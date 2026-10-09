//! camfarm: synthetic RTSP cameras. Every camera renders its pictures on the GPU (animated noise field, moving discs,
//! white noise, and a frame identity code in the top band: camera id, frame number, capture time), hardware-encodes them (Jetson V4L2/NVJPG, desktop
//! NVENC, or software) and serves them as rtsp://<host>:<port>/<prefix>NN. Frame n of a camera is pushed into
//! its encoder at exactly t0 + phase + n / fps; its capture time (in the code, and in the ONVIF RTP header
//! extension with --onvif) is that instant.

mod gen;

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_rtsp_server as rtsp;
use rtsp::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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

#[derive(Parser, Debug)]
#[command(name = "camfarm", version, about = "Synthetic RTSP cameras with a frame identity code")]
struct Args {
    #[arg(long, default_value_t = 32)]
    cameras: u32,
    /// id of the first camera (code + mount name); use different ranges on several hosts
    #[arg(long, default_value_t = 1)]
    first_id: u32,
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    #[arg(long, default_value_t = 15)]
    fps: u32,
    #[arg(long, value_enum, default_value_t = Codec::H264)]
    codec: Codec,
    #[arg(long, value_enum, default_value_t = Encoder::Auto)]
    encoder: Encoder,
    #[arg(long, default_value_t = 2000)]
    bitrate_kbps: u32,
    /// variable instead of constant bitrate
    #[arg(long)]
    vbr: bool,
    /// key frame interval in seconds
    #[arg(long, default_value_t = 2.0)]
    gop_s: f32,
    #[arg(long, default_value_t = 85)]
    jpeg_quality: u32,
    /// white noise amplitude in luma levels (entropy: more noise = more bits needed)
    #[arg(long, default_value_t = 6.0)]
    noise: f32,
    #[arg(long, value_enum, default_value_t = Phase::Spread)]
    phase: Phase,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value_t = 8554)]
    port: u16,
    #[arg(long, default_value = "cam")]
    mount_prefix: String,
    /// add the ONVIF replay RTP header extension (NTP capture time per frame), like Axis onvifreplayext=1
    #[arg(long)]
    onvif: bool,
    #[arg(long, default_value_t = 1400)]
    mtu: u32,
    #[arg(long, default_value_t = 10)]
    stats_s: u64,
}

struct Camera {
    id: u32,
    phase: Duration,
    appsrc: Mutex<Option<gst_app::AppSrc>>,
    ready: Mutex<VecDeque<(u64, Instant, gst::Buffer)>>,
    next_render: AtomicU64,
    pushed: AtomicU64,
    unconnected: AtomicU64,
    skipped: AtomicU64,
    late: AtomicU64,
    max_late_us: AtomicU64,
}

fn element_exists(name: &str) -> bool {
    gst::ElementFactory::find(name).is_some()
}

fn pipeline_desc(a: &Args, enc: Encoder) -> Result<String> {
    let caps = format!("video/x-raw,format=NV12,width={},height={},framerate={}/1", a.width, a.height, a.fps);
    let head = format!("appsrc name=src is-live=true format=time caps=\"{caps}\"");
    let gop = ((a.fps as f32) * a.gop_s).round().max(1.0) as u32;
    let (bps, kbps) = (a.bitrate_kbps as u64 * 1000, a.bitrate_kbps);
    let pay = if a.onvif { "pay" } else { "pay0" };
    let mtu = a.mtu;
    let body = match (enc, a.codec) {
        (Encoder::Jetson, Codec::H264) => format!(
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc bitrate={bps} peak-bitrate={} \
             control-rate={} iframeinterval={gop} idrinterval={gop} insert-sps-pps=true insert-vui=true \
             maxperf-enable=true num-B-Frames=0 profile=4 ! h264parse ! rtph264pay name={pay} pt=96 \
             config-interval=-1 mtu={mtu}",
            bps * 6 / 5,
            if a.vbr { 0 } else { 1 }
        ),
        (Encoder::Jetson, Codec::H265) => format!(
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h265enc bitrate={bps} peak-bitrate={} \
             control-rate={} iframeinterval={gop} idrinterval={gop} insert-sps-pps=true insert-vui=true \
             maxperf-enable=true num-B-Frames=0 ! h265parse ! rtph265pay name={pay} pt=96 config-interval=-1 mtu={mtu}",
            bps * 6 / 5,
            if a.vbr { 0 } else { 1 }
        ),
        (Encoder::Jetson, Codec::Mjpeg) => format!(
            "nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! nvjpegenc quality={} ! rtpjpegpay name={pay} pt=26 mtu={mtu}",
            a.jpeg_quality
        ),
        (Encoder::Nvcodec, Codec::H264) => format!(
            "nvh264enc bitrate={kbps} max-bitrate={} rc-mode={} gop-size={gop} bframes=0 zerolatency=true ! \
             h264parse ! rtph264pay name={pay} pt=96 config-interval=-1 mtu={mtu}",
            kbps * 6 / 5,
            if a.vbr { "vbr" } else { "cbr" }
        ),
        (Encoder::Nvcodec, Codec::H265) => format!(
            "nvh265enc bitrate={kbps} max-bitrate={} rc-mode={} gop-size={gop} bframes=0 zerolatency=true ! \
             h265parse ! rtph265pay name={pay} pt=96 config-interval=-1 mtu={mtu}",
            kbps * 6 / 5,
            if a.vbr { "vbr" } else { "cbr" }
        ),
        (Encoder::Software, Codec::H264) => format!(
            "videoconvert ! x264enc tune=zerolatency speed-preset=veryfast bitrate={kbps} key-int-max={gop} bframes=0 ! \
             h264parse ! rtph264pay name={pay} pt=96 config-interval=-1 mtu={mtu}"
        ),
        (Encoder::Software, Codec::H265) => format!(
            "videoconvert ! x265enc tune=zerolatency speed-preset=veryfast bitrate={kbps} key-int-max={gop} ! \
             h265parse ! rtph265pay name={pay} pt=96 config-interval=-1 mtu={mtu}"
        ),
        (_, Codec::Mjpeg) => format!(
            "videoconvert ! video/x-raw,format=I420 ! jpegenc quality={} ! rtpjpegpay name={pay} pt=26 mtu={mtu}",
            a.jpeg_quality
        ),
        (Encoder::Auto, _) => return Err(anyhow!("encoder not resolved")),
    };
    let tail = if a.onvif { " ! rtponviftimestamp name=pay0" } else { "" };
    Ok(format!("( {head} ! {body}{tail} )"))
}

fn resolve_encoder(a: &Args) -> Encoder {
    match a.encoder {
        Encoder::Auto if element_exists("nvv4l2h264enc") => Encoder::Jetson,
        Encoder::Auto if element_exists("nvh264enc") => Encoder::Nvcodec,
        Encoder::Auto => Encoder::Software,
        e => e,
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn main() -> Result<()> {
    let a = Args::parse();
    gst::init()?;
    let enc = resolve_encoder(&a);
    let desc = pipeline_desc(&a, enc)?;
    let generator = gen::Generator::new(a.width, a.height)?;
    let period = Duration::from_nanos(1_000_000_000 / a.fps as u64);
    let n = a.cameras.max(1);
    let cams: Arc<Vec<Camera>> = Arc::new(
        (0..n)
            .map(|k| {
                let phase = match a.phase {
                    Phase::Spread => period * k / n,
                    Phase::Random => Duration::from_nanos(splitmix(a.seed ^ (k as u64 + 1)) % period.as_nanos() as u64),
                    Phase::Aligned => Duration::ZERO,
                };
                Camera {
                    id: a.first_id + k,
                    phase,
                    appsrc: Mutex::new(None),
                    ready: Mutex::new(VecDeque::new()),
                    next_render: AtomicU64::new(0),
                    pushed: AtomicU64::new(0),
                    unconnected: AtomicU64::new(0),
                    skipped: AtomicU64::new(0),
                    late: AtomicU64::new(0),
                    max_late_us: AtomicU64::new(0),
                }
            })
            .collect(),
    );

    // Common time base: frame n of camera c is captured at t0 + phase_c + n * period (monotonic), which is
    // rt0_us + (phase_c + n * period) in Unix microseconds.
    let t0 = Instant::now() + Duration::from_millis(1500);
    let rt0_us = (SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros() + 1_500_000) as u64;

    println!(
        "camfarm: {n} x {}x{}@{} {:?} via {:?}, {} kbit/s {}, GOP {} s, noise {}, phase {:?}, GPU {}",
        a.width,
        a.height,
        a.fps,
        a.codec,
        enc,
        a.bitrate_kbps,
        if a.vbr { "VBR" } else { "CBR" },
        a.gop_s,
        a.noise,
        a.phase,
        generator.adapter_name
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

    // RTSP server: one shared media per camera, its appsrc registered with the camera when configured.
    let server = rtsp::RTSPServer::new();
    server.set_service(&a.port.to_string());
    let mounts = server.mount_points().ok_or_else(|| anyhow!("no mount points"))?;
    for (k, c) in cams.iter().enumerate() {
        let factory = rtsp::RTSPMediaFactory::new();
        factory.set_launch(&desc);
        factory.set_shared(true);
        factory.set_suspend_mode(rtsp::RTSPSuspendMode::None);
        factory.set_eos_shutdown(false);
        let cams_c = cams.clone();
        factory.connect_media_configure(move |_, media| {
            let bin = media.element().downcast::<gst::Bin>().expect("media element is a bin");
            let src = bin.by_name("src").expect("appsrc").downcast::<gst_app::AppSrc>().expect("appsrc type");
            src.set_max_buffers(4);
            src.set_leaky_type(gst_app::AppLeakyType::Downstream);
            *cams_c[k].appsrc.lock().unwrap() = Some(src);
            let cams_u = cams_c.clone();
            media.connect_unprepared(move |_| {
                *cams_u[k].appsrc.lock().unwrap() = None;
            });
        });
        mounts.add_factory(&format!("/{}{:02}", a.mount_prefix, c.id), factory);
    }
    let _server_id = server.attach(None).map_err(|e| anyhow!("cannot listen on RTSP port {} (in use?): {e}", a.port))?;

    // Render thread: renders every frame due within the next 80 ms, all cameras in one GPU batch.
    {
        let cams = cams.clone();
        let (w, h, fps, noise, seed) = (a.width, a.height, a.fps, a.noise, a.seed);
        let geometry = framecode::Geometry::for_picture(w, h);
        std::thread::Builder::new().name("render".into()).spawn(move || {
            const RING: u64 = 4;
            let slots: Vec<Vec<gen::Slot>> = (0..cams.len()).map(|_| (0..RING).map(|_| generator.slot()).collect()).collect();
            loop {
                let now = Instant::now();
                let horizon = now + Duration::from_millis(80);
                let mut jobs: Vec<(usize, u64, Instant)> = Vec::new();
                for (ci, c) in cams.iter().enumerate() {
                    let mut taken = 0;
                    while taken < RING {
                        let n = c.next_render.load(Relaxed);
                        let due = t0 + c.phase + period * n as u32;
                        if due >= horizon {
                            break;
                        }
                        c.next_render.store(n + 1, Relaxed);
                        if due + period < now {
                            c.skipped.fetch_add(1, Relaxed);
                            continue;
                        }
                        jobs.push((ci, n, due));
                        taken += 1;
                    }
                }
                let params: Vec<(&gen::Slot, gen::Params)> = jobs
                    .iter()
                    .map(|&(ci, n, due)| {
                        let c = &cams[ci];
                        let capture_us = rt0_us + (due - t0).as_micros() as u64;
                        let bits = framecode::encode(&framecode::FrameId {
                            camera: c.id as u8,
                            frame: n as u32,
                            capture_us,
                        });
                        let p = gen::Params {
                            width: w,
                            height: h,
                            band_rows: geometry.band_rows(),
                            cell_w: geometry.cell_w,
                            cell_h: geometry.cell_h,
                            cols: framecode::COLS,
                            frame: n as u32,
                            seed: (seed as u32) ^ (c.id * 2654435761),
                            time: n as f32 / fps as f32,
                            noise,
                            hue: (c.id as f32 * 0.618_034).fract(),
                            speed: 0.3 + 0.05 * (c.id % 5) as f32,
                            bits,
                        };
                        (&slots[ci][(n % RING) as usize], p)
                    })
                    .collect();
                let result = generator.render_batch(&params, |i, data| {
                    let (ci, n, due) = jobs[i];
                    let mut buf = gst::Buffer::with_size(data.len()).expect("buffer");
                    buf.get_mut().unwrap().map_writable().unwrap().copy_from_slice(data);
                    cams[ci].ready.lock().unwrap().push_back((n, due, buf));
                });
                if let Err(e) = result {
                    eprintln!("render failed: {e}");
                }
                let spent = now.elapsed();
                if spent < Duration::from_millis(10) {
                    std::thread::sleep(Duration::from_millis(10) - spent);
                }
            }
        })?;
    }

    // Push thread: hands each frame to its camera's encoder at its exact due time.
    {
        let cams = cams.clone();
        std::thread::Builder::new().name("push".into()).spawn(move || loop {
            let mut best: Option<(usize, Instant)> = None;
            for (ci, c) in cams.iter().enumerate() {
                if let Some((_, due, _)) = c.ready.lock().unwrap().front() {
                    if best.map_or(true, |(_, b)| *due < b) {
                        best = Some((ci, *due));
                    }
                }
            }
            let Some((ci, due)) = best else {
                std::thread::sleep(Duration::from_micros(500));
                continue;
            };
            let now = Instant::now();
            if due > now + Duration::from_millis(2) {
                std::thread::sleep(due - now - Duration::from_millis(1));
                continue;
            }
            while Instant::now() < due {
                std::hint::spin_loop();
            }
            let c = &cams[ci];
            let Some((_, due, mut buf)) = c.ready.lock().unwrap().pop_front() else { continue };
            let late = Instant::now().saturating_duration_since(due).as_micros() as u64;
            if late > 2000 {
                c.late.fetch_add(1, Relaxed);
            }
            c.max_late_us.fetch_max(late, Relaxed);
            let src = c.appsrc.lock().unwrap().clone();
            match src {
                Some(src) => {
                    let b = buf.get_mut().unwrap();
                    b.set_pts(src.current_running_time());
                    b.set_duration(gst::ClockTime::from_nseconds(period.as_nanos() as u64));
                    if src.push_buffer(buf).is_ok() {
                        c.pushed.fetch_add(1, Relaxed);
                    } else {
                        *c.appsrc.lock().unwrap() = None;
                    }
                }
                None => {
                    c.unconnected.fetch_add(1, Relaxed);
                }
            }
        })?;
    }

    // Stats.
    {
        let cams = cams.clone();
        let every = Duration::from_secs(a.stats_s.max(1));
        std::thread::spawn(move || loop {
            std::thread::sleep(every);
            let sum = |f: fn(&Camera) -> u64| cams.iter().map(f).sum::<u64>();
            let connected = cams.iter().filter(|c| c.appsrc.lock().unwrap().is_some()).count();
            println!(
                "stats: {connected}/{} cameras streaming, frames pushed {}, not connected {}, late >2 ms {}, \
                 skipped {}, max late {:.2} ms",
                cams.len(),
                sum(|c| c.pushed.load(Relaxed)),
                sum(|c| c.unconnected.load(Relaxed)),
                sum(|c| c.late.load(Relaxed)),
                sum(|c| c.skipped.load(Relaxed)),
                cams.iter().map(|c| c.max_late_us.load(Relaxed)).max().unwrap_or(0) as f64 / 1000.0
            );
        });
    }

    gst::glib::MainLoop::new(None, false).run();
    Ok(())
}
