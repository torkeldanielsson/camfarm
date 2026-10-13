mod gen;
mod prof;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_rtsp_server as rtsp;
use prof::{plot, zone};
use rtsp::prelude::*;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Pictures are rendered this far ahead of their due time, which absorbs GPU and scheduling stalls.
const RENDER_AHEAD: Duration = Duration::from_millis(200);
const RENDER_TICK: Duration = Duration::from_millis(10);
const MIN_BATCH: usize = 64;
const START_DELAY: Duration = Duration::from_millis(1500);
const LATE: Duration = Duration::from_millis(2);
#[cfg(target_os = "linux")]
const PUSH_THREAD_PRIORITY: i32 = 10;
const APPSRC_MAX_BUFFERS: u64 = 4;
const PEAK_BITRATE_PERCENT: u64 = 120;
/// Picture time runs back and forth over this many seconds, so the shader's f32 time keeps sub-millisecond steps.
const PICTURE_TIME_TURN_S: f64 = 3600.0;

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq)]
enum Codec {
    H264,
    H265,
    Mjpeg,
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq)]
enum Encoder {
    Auto,
    Jetson,
    Nvcodec,
    Software,
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq)]
enum Phase {
    /// camera k (from 0) starts k/N of a frame period after camera 0
    Spread,
    /// uniformly random phase per camera (--seed)
    Random,
    /// all cameras capture at the same instant
    Aligned,
}

#[derive(Parser, Clone, Debug)]
#[command(name = "camfarm", version, about = "Synthetic RTSP cameras with a frame identity code")]
struct Args {
    /// number of cameras
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..))]
    cameras: u32,
    /// id of the first camera, in the code and the mount name (ids go up to 255); use different ranges on several hosts
    #[arg(long, default_value_t = 1)]
    first_id: u32,
    /// picture width in pixels, a multiple of 4
    #[arg(long, default_value_t = 1280)]
    width: u32,
    /// picture height in pixels, even
    #[arg(long, default_value_t = 720)]
    height: u32,
    /// frames per second
    #[arg(long, default_value_t = 15, value_parser = clap::value_parser!(u32).range(1..))]
    fps: u32,
    /// video codec
    #[arg(long, value_enum, default_value_t = Codec::H264)]
    codec: Codec,
    /// encoder elements; auto picks jetson, then nvcodec (NVIDIA desktop), then software
    #[arg(long, value_enum, default_value_t = Encoder::Auto)]
    encoder: Encoder,
    /// H.265/H.264 bitrate in kbit/s
    #[arg(long, default_value_t = 2000)]
    bitrate_kbps: u32,
    /// variable instead of constant bitrate
    #[arg(long)]
    vbr: bool,
    /// key frame interval in seconds
    #[arg(long, default_value_t = 2.0)]
    gop_s: f32,
    /// MJPEG quality, 0 to 100
    #[arg(long, default_value_t = 85, value_parser = clap::value_parser!(u32).range(0..=100))]
    jpeg_quality: u32,
    /// white noise amplitude in luma levels (entropy: more noise = more bits needed)
    #[arg(long, default_value_t = 6.0)]
    noise: f32,
    /// when each camera captures within the frame period
    #[arg(long, value_enum, default_value_t = Phase::Spread)]
    phase: Phase,
    /// seed for --phase random and for the picture content
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// RTSP port
    #[arg(long, default_value_t = 8554)]
    port: u16,
    /// mount names are <prefix>NN, e.g. cam01
    #[arg(long, default_value = "cam")]
    mount_prefix: String,
    /// add the ONVIF replay RTP header extension (NTP capture time per frame), like Axis onvifreplayext=1
    #[arg(long)]
    onvif: bool,
    /// largest RTP packet in bytes
    #[arg(long, default_value_t = 1400)]
    mtu: u32,
    /// seconds between stats lines
    #[arg(long, default_value_t = 10)]
    stats_s: u64,
}

#[derive(Default)]
struct Camera {
    id: u32,
    phase: Duration,
    appsrc: Mutex<Option<gst_app::AppSrc>>,
    pushed: AtomicU64,
    unconnected: AtomicU64,
    skipped: AtomicU64,
    late: AtomicU64,
    max_late_us: AtomicU64,
}

/// Rendered pictures waiting for their due time, by (due, camera index).
#[derive(Default)]
struct Ready {
    frames: Mutex<BTreeMap<(Instant, usize), gst::Buffer>>,
    changed: Condvar,
}

/// Frame n of camera c is due at t0 + phase_c + n * period (monotonic), its capture time.
struct Timeline {
    t0: Instant,
    rt0_us: u64,
    period: Duration,
}

impl Timeline {
    fn start(period: Duration) -> Result<Self> {
        let unix_now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        Ok(Self { t0: Instant::now() + START_DELAY, rt0_us: (unix_now + START_DELAY).as_micros() as u64, period })
    }

    fn due(&self, c: &Camera, n: u64) -> Instant {
        self.t0 + c.phase + self.period * n as u32
    }

    fn capture_us(&self, due: Instant) -> u64 {
        self.rt0_us + (due - self.t0).as_micros() as u64
    }
}

fn element_exists(name: &str) -> bool {
    gst::ElementFactory::find(name).is_some()
}

/// MJPEG cameras render I420, which the JPEG encoders take directly (on Jetson that skips a VIC conversion).
fn renders_i420(a: &Args) -> bool {
    a.codec == Codec::Mjpeg && a.width.is_multiple_of(8)
}

fn pipeline_desc(a: &Args, enc: Encoder) -> String {
    let format = if renders_i420(a) { "I420" } else { "NV12" };
    let caps = format!("video/x-raw,format={format},width={},height={},framerate={}/1", a.width, a.height, a.fps);
    let gop = ((a.fps as f32) * a.gop_s).round().max(1.0) as u32;
    let kbps = a.bitrate_kbps as u64;
    let peak_kbps = kbps * PEAK_BITRATE_PERCENT / 100;
    let quality = a.jpeg_quality;
    let codec = match a.codec {
        Codec::H264 => "h264",
        Codec::H265 => "h265",
        Codec::Mjpeg => "jpeg",
    };
    let encoder = match (enc, a.codec) {
        (Encoder::Jetson, Codec::Mjpeg) if renders_i420(a) => format!("nvjpegenc quality={quality}"),
        (Encoder::Jetson, Codec::Mjpeg) => {
            format!("nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! nvjpegenc quality={quality}")
        }
        (_, Codec::Mjpeg) if renders_i420(a) => format!("jpegenc quality={quality}"),
        (_, Codec::Mjpeg) => format!("videoconvert ! video/x-raw,format=I420 ! jpegenc quality={quality}"),
        (Encoder::Jetson, _) => format!(
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2{codec}enc bitrate={} peak-bitrate={} \
             control-rate={} iframeinterval={gop} idrinterval={gop} insert-sps-pps=true insert-vui=true \
             maxperf-enable=true num-B-Frames=0{}",
            kbps * 1000,
            kbps * 1000 * PEAK_BITRATE_PERCENT / 100,
            if a.vbr { 0 } else { 1 },
            if a.codec == Codec::H264 { " profile=4" } else { "" }
        ),
        (Encoder::Nvcodec, _) => format!(
            "nv{codec}enc bitrate={kbps} max-bitrate={peak_kbps} rc-mode={} gop-size={gop} bframes=0 zerolatency=true",
            if a.vbr { "vbr" } else { "cbr" }
        ),
        (Encoder::Software, Codec::H264) => format!(
            "videoconvert ! x264enc tune=zerolatency speed-preset=veryfast bitrate={kbps} key-int-max={gop} bframes=0"
        ),
        (Encoder::Software, _) => {
            format!("videoconvert ! x265enc tune=zerolatency speed-preset=veryfast bitrate={kbps} key-int-max={gop}")
        }
        (Encoder::Auto, _) => unreachable!("the encoder is resolved first"),
    };
    let (pay, mtu) = (if a.onvif { "pay" } else { "pay0" }, a.mtu);
    let payloader = match a.codec {
        Codec::Mjpeg => format!("rtpjpegpay name={pay} pt=26 mtu={mtu}"),
        _ => format!("{codec}parse ! rtp{codec}pay name={pay} pt=96 config-interval=-1 mtu={mtu}"),
    };
    let onvif = if a.onvif { " ! rtponviftimestamp name=pay0" } else { "" };
    format!("( appsrc name=src is-live=true format=time caps=\"{caps}\" ! {encoder} ! {payloader}{onvif} )")
}

fn resolve_encoder(a: &Args) -> Encoder {
    match a.encoder {
        Encoder::Auto if element_exists("nvv4l2h264enc") => Encoder::Jetson,
        Encoder::Auto if element_exists("nvh264enc") => Encoder::Nvcodec,
        Encoder::Auto => Encoder::Software,
        e => e,
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn cameras(a: &Args, period: Duration) -> Vec<Camera> {
    (0..a.cameras)
        .map(|k| {
            let phase = match a.phase {
                Phase::Spread => period * k / a.cameras,
                Phase::Random => Duration::from_nanos(splitmix64(a.seed ^ (k as u64 + 1)) % period.as_nanos() as u64),
                Phase::Aligned => Duration::ZERO,
            };
            Camera { id: a.first_id + k, phase, ..Default::default() }
        })
        .collect()
}

fn picture_time(n: u64, fps: u32) -> f32 {
    let t = (n as f64 / fps as f64) % (2.0 * PICTURE_TIME_TURN_S);
    (if t < PICTURE_TIME_TURN_S { t } else { 2.0 * PICTURE_TIME_TURN_S - t }) as f32
}

fn picture_params(a: &Args, c: &Camera, n: u64, capture_us: u64) -> gen::Params {
    let geometry = framecode::Geometry::for_picture(a.width, a.height);
    gen::Params {
        width: a.width,
        height: a.height,
        band_rows: geometry.band_rows(),
        cell_w: geometry.cell_w,
        cell_h: geometry.cell_h,
        cols: framecode::COLS,
        frame: n as u32,
        seed: (a.seed as u32) ^ c.id.wrapping_mul(2654435761),
        time: picture_time(n, a.fps),
        noise: a.noise,
        hue: (c.id as f32 * 0.618_034).fract(),
        speed: 0.3 + 0.05 * (c.id % 5) as f32,
        bits: framecode::encode(&framecode::FrameId { camera: c.id as u8, frame: n as u32, capture_us }),
        i420: renders_i420(a) as u32,
        _pad: [0; 3],
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    let last_id = a.first_id as u64 + a.cameras as u64 - 1;
    if last_id > u8::MAX as u64 {
        bail!("camera ids {}..={last_id} do not fit the frame code, which carries ids up to 255", a.first_id);
    }
    prof::start();
    gst::init()?;
    let enc = resolve_encoder(&a);
    let desc = pipeline_desc(&a, enc);
    gst::parse::launch(&desc).with_context(|| format!("cannot build the camera pipeline {desc}"))?;
    let mut generator = gen::Generator::new(a.width, a.height)?;
    let period = Duration::from_nanos(1_000_000_000 / a.fps as u64);
    let cams = Arc::new(cameras(&a, period));

    let rate = match a.codec {
        Codec::Mjpeg => format!("quality {}", a.jpeg_quality),
        _ => format!("{} kbit/s {}, GOP {} s", a.bitrate_kbps, if a.vbr { "VBR" } else { "CBR" }, a.gop_s),
    };
    println!(
        "camfarm: {} x {}x{}@{} {:?} via {:?}, {rate}, noise {}, phase {:?}, GPU {}",
        a.cameras, a.width, a.height, a.fps, a.codec, enc, a.noise, a.phase, generator.adapter_name
    );
    println!("pipeline: {desc}");
    for c in cams.iter() {
        println!(
            "camera {:3}: rtsp://<host>:{}/{}{:02}  phase {:.2} ms",
            c.id,
            a.port,
            a.mount_prefix,
            c.id,
            c.phase.as_secs_f64() * 1000.0
        );
    }
    let _server = start_rtsp_server(&a, &desc, &cams)?;

    // Allocate every GPU buffer and run one batch before the time base is set, so the first frames are on time.
    let frames_per_tick = (cams.len() as u64 * a.fps as u64 * RENDER_TICK.as_millis() as u64).div_ceil(1000) as usize;
    let max_batch = (2 * frames_per_tick).max(MIN_BATCH);
    let ahead_frames = (RENDER_AHEAD.as_millis() as u64 * a.fps as u64).div_ceil(1000) as usize;
    generator.reserve(max_batch, cams.len() * (ahead_frames + 2) + max_batch);
    generator.submit(&vec![picture_params(&a, &cams[0], 0, 0); max_batch]);
    drop(generator.collect()?);

    let timeline = Arc::new(Timeline::start(period)?);
    let ready = Arc::new(Ready::default());
    {
        let (a, cams, ready, timeline) = (a.clone(), cams.clone(), ready.clone(), timeline.clone());
        std::thread::Builder::new().name("render".into()).spawn(move || {
            let Err(e) = render_loop(generator, &a, &cams, &ready, &timeline, max_batch);
            eprintln!("camfarm: render failed: {e:#}");
            std::process::exit(1);
        })?;
    }
    {
        let cams = cams.clone();
        std::thread::Builder::new().name("push".into()).spawn(move || push_loop(&cams, &ready, period))?;
    }
    {
        let every = Duration::from_secs(a.stats_s.max(1));
        std::thread::Builder::new().name("stats".into()).spawn(move || stats_loop(&cams, every))?;
    }
    gst::glib::MainLoop::new(None, false).run();
    Ok(())
}

fn start_rtsp_server(a: &Args, desc: &str, cams: &Arc<Vec<Camera>>) -> Result<gst::glib::SourceId> {
    let server = rtsp::RTSPServer::new();
    server.set_service(&a.port.to_string());
    let mounts = server.mount_points().ok_or_else(|| anyhow!("no mount points"))?;
    for (k, c) in cams.iter().enumerate() {
        let factory = rtsp::RTSPMediaFactory::new();
        factory.set_launch(desc);
        factory.set_shared(true);
        factory.set_suspend_mode(rtsp::RTSPSuspendMode::None);
        factory.set_eos_shutdown(false);
        let cams = cams.clone();
        factory.connect_media_configure(move |_, media| {
            let bin = media.element().downcast::<gst::Bin>().expect("media element is a bin");
            let src = bin.by_name("src").expect("appsrc").downcast::<gst_app::AppSrc>().expect("appsrc type");
            src.set_max_buffers(APPSRC_MAX_BUFFERS);
            src.set_leaky_type(gst_app::AppLeakyType::Downstream);
            *cams[k].appsrc.lock().unwrap() = Some(src.clone());
            let cams = cams.clone();
            // a reconnecting client may have configured a new media by now; leave that one alone
            media.connect_unprepared(move |_| {
                let mut current = cams[k].appsrc.lock().unwrap();
                if current.as_ref() == Some(&src) {
                    *current = None;
                }
            });
        });
        mounts.add_factory(&format!("/{}{:02}", a.mount_prefix, c.id), factory);
    }
    server.attach(None).map_err(|e| anyhow!("cannot listen on RTSP port {} (in use?): {e}", a.port))
}

fn render_loop(
    mut generator: gen::Generator,
    a: &Args,
    cams: &[Camera],
    ready: &Ready,
    timeline: &Timeline,
    max_batch: usize,
) -> Result<std::convert::Infallible> {
    let mut schedule: BinaryHeap<Reverse<(Instant, usize, u64)>> =
        cams.iter().enumerate().map(|(ci, c)| Reverse((timeline.due(c, 0), ci, 0))).collect();
    let mut pending: Vec<(Instant, usize)> = Vec::new();
    let mut jobs: Vec<gen::Params> = Vec::new();
    loop {
        let tick = Instant::now();
        {
            zone!("render loop");
            let pictures = generator.collect()?;
            if !pictures.is_empty() {
                let buffers: Vec<_> = pictures.into_iter().map(gst::Buffer::from_slice).collect();
                ready.frames.lock().unwrap().extend(pending.drain(..).zip(buffers));
                ready.changed.notify_one();
            }
            let now = Instant::now();
            jobs.clear();
            while let Some(&Reverse((due, ci, n))) = schedule.peek() {
                if due >= now + RENDER_AHEAD || jobs.len() >= max_batch {
                    break;
                }
                let c = &cams[ci];
                schedule.pop();
                schedule.push(Reverse((timeline.due(c, n + 1), ci, n + 1)));
                if due + timeline.period < now {
                    c.skipped.fetch_add(1, Relaxed);
                    continue;
                }
                pending.push((due, ci));
                jobs.push(picture_params(a, c, n, timeline.capture_us(due)));
            }
            plot!("batch frames", jobs.len());
            generator.submit(&jobs);
        }
        let spent = tick.elapsed();
        if spent < RENDER_TICK {
            std::thread::sleep(RENDER_TICK - spent);
        }
    }
}

fn stats_loop(cams: &[Camera], every: Duration) {
    loop {
        std::thread::sleep(every);
        let sum = |f: fn(&Camera) -> u64| cams.iter().map(f).sum::<u64>();
        let connected = cams.iter().filter(|c| c.appsrc.lock().unwrap().is_some()).count();
        println!(
            "stats: {connected}/{} cameras streaming, frames pushed {}, not connected {}, late >{} ms {}, \
             skipped {}, max late {:.2} ms",
            cams.len(),
            sum(|c| c.pushed.load(Relaxed)),
            sum(|c| c.unconnected.load(Relaxed)),
            LATE.as_millis(),
            sum(|c| c.late.load(Relaxed)),
            sum(|c| c.skipped.load(Relaxed)),
            cams.iter().map(|c| c.max_late_us.load(Relaxed)).max().unwrap_or(0) as f64 / 1000.0
        );
    }
}

fn push_loop(cams: &[Camera], ready: &Ready, period: Duration) {
    #[cfg(target_os = "linux")]
    unsafe {
        // a 1 ns timer slack lets this thread sleep until a frame is due instead of spinning
        libc::prctl(libc::PR_SET_TIMERSLACK, 1);
        let param = libc::sched_param { sched_priority: PUSH_THREAD_PRIORITY };
        if libc::pthread_setschedparam(libc::pthread_self(), libc::SCHED_FIFO, &param) == 0 {
            println!("push thread: real-time priority");
        } else {
            println!("push thread: normal priority (real-time needs root or CAP_SYS_NICE)");
        }
    }
    let mut frames = ready.frames.lock().unwrap();
    loop {
        let Some(entry) = frames.first_entry() else {
            frames = ready.changed.wait(frames).unwrap();
            continue;
        };
        let (due, ci) = *entry.key();
        let now = Instant::now();
        if due > now {
            frames = ready.changed.wait_timeout(frames, due - now).unwrap().0;
            continue;
        }
        let buf = entry.remove();
        drop(frames);
        push_frame(&cams[ci], due, buf, period);
        frames = ready.frames.lock().unwrap();
    }
}

fn push_frame(c: &Camera, due: Instant, mut buf: gst::Buffer, period: Duration) {
    zone!("push frame");
    let late = Instant::now().saturating_duration_since(due);
    if late > LATE {
        c.late.fetch_add(1, Relaxed);
    }
    let late_us = late.as_micros() as u64;
    c.max_late_us.fetch_max(late_us, Relaxed);
    plot!("push late ms", late_us as f64 / 1000.0);
    let src = c.appsrc.lock().unwrap().clone();
    let Some(src) = src else {
        c.unconnected.fetch_add(1, Relaxed);
        return;
    };
    let b = buf.make_mut();
    b.set_pts(src.current_running_time());
    b.set_duration(gst::ClockTime::from_nseconds(period.as_nanos() as u64));
    let pushed = {
        zone!("appsrc push");
        src.push_buffer(buf).is_ok()
    };
    // a failed push means the media is going away; its unprepared handler clears the appsrc
    if pushed {
        c.pushed.fetch_add(1, Relaxed);
    } else {
        c.unconnected.fetch_add(1, Relaxed);
    }
}
