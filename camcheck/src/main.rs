//! camcheck: reads camfarm pictures and checks the frame identity code (framecode) in every one of them.
//!
//! `rtsp`: pull the cameras directly (reference: proves the cameras and the network are clean).
//! `cuda-ipc`: read GStreamer CUDA IPC exports (cudaipcsink servers, one socket per camera). Only the code band rows
//!   are copied out of GPU memory, so the reader hardly loads the machine it runs on.
//! `grid`: decode a recorded video in which the cameras are tiled in a grid (e.g. a composed stream) and read every
//!   cell.
//!
//! Per camera: frames received, distinct camera frames, lost (gaps in the frame numbers), repeated, frames that went
//! backwards (an older picture shown after a newer one), unreadable, pictures of the wrong camera, and the latency
//! from the capture time in the code to the arrival here (sender and receiver clocks must be in sync, e.g. NTP).

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use framecode::{DecodeError, FrameId};
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(name = "camcheck", about = "Check camfarm frame codes from RTSP, GStreamer CUDA IPC exports or a recorded grid")]
struct Args {
    #[command(subcommand)]
    mode: Mode,
    /// first camera id
    #[arg(long, global = true, default_value_t = 1)]
    first: u32,
    #[arg(long, global = true, default_value_t = 32)]
    count: u32,
    /// measurement time after the warm-up
    #[arg(long, global = true, default_value_t = 30.0)]
    seconds: f64,
    #[arg(long, global = true, default_value_t = 3.0)]
    warmup: f64,
    /// write one line per received picture
    #[arg(long, global = true)]
    csv: Option<String>,
    /// write the per-camera summary as JSON
    #[arg(long, global = true)]
    json: Option<String>,
}

#[derive(Subcommand)]
enum Mode {
    /// pull rtsp://.../<base>NN directly
    Rtsp {
        /// e.g. rtsp://192.0.2.10:8554/cam (camera NN is appended as two digits)
        #[arg(long)]
        url_base: String,
        #[arg(long)]
        tcp: bool,
    },
    /// read GStreamer CUDA IPC exports (cudaipcsink sockets) <path-base>NN
    CudaIpc {
        /// e.g. /tmp/cam (camera NN is appended as two digits)
        #[arg(long)]
        path_base: String,
    },
    /// decode a recorded video (H.264/H.265 elementary stream or any container) with the cameras tiled in a grid
    Grid {
        #[arg(long)]
        file: String,
        #[arg(long, default_value_t = 8)]
        cols: u32,
        #[arg(long, default_value_t = 960)]
        cell_w: u32,
        #[arg(long, default_value_t = 540)]
        cell_h: u32,
        /// frame rate of the recording, for the time axis (latency is not meaningful for a file)
        #[arg(long, default_value_t = 15.0)]
        fps: f64,
    },
}

struct Rec {
    arrival_us: u64,
    result: Result<FrameId, DecodeError>,
}

fn now_us() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64
}

/// CUDA access for the cuda-ipc mode, loaded at run time so the binary also runs where CUDA is not installed.
mod cuda {
    use gstreamer as gst;
    use std::os::raw::{c_int, c_void};
    use std::sync::OnceLock;

    // Leading fields of GstCudaMemory (gst/cuda/gstcudamemory.h).
    #[repr(C)]
    struct CudaMemoryHead {
        mem: gst::ffi::GstMemory,
        context: *mut c_void,
    }

    #[repr(C)]
    struct Memcpy2D {
        src_x_in_bytes: usize,
        src_y: usize,
        src_memory_type: u32,
        src_host: *const c_void,
        src_device: u64,
        src_array: *mut c_void,
        src_pitch: usize,
        dst_x_in_bytes: usize,
        dst_y: usize,
        dst_memory_type: u32,
        dst_host: *mut c_void,
        dst_device: u64,
        dst_array: *mut c_void,
        dst_pitch: usize,
        width_in_bytes: usize,
        height: usize,
    }

    const MEMORYTYPE_HOST: u32 = 1;
    const MEMORYTYPE_DEVICE: u32 = 2;
    const GST_MAP_CUDA: u32 = 1 << 17;

    struct Api {
        is_cuda_memory: unsafe extern "C" fn(*mut gst::ffi::GstMemory) -> gst::glib::ffi::gboolean,
        context_push: unsafe extern "C" fn(*mut c_void) -> gst::glib::ffi::gboolean,
        context_pop: unsafe extern "C" fn(*mut *mut c_void) -> gst::glib::ffi::gboolean,
        memcpy_2d: unsafe extern "C" fn(*const Memcpy2D) -> c_int,
        _libs: (libloading::Library, libloading::Library),
    }

    fn api() -> Result<&'static Api, String> {
        static API: OnceLock<Result<Api, String>> = OnceLock::new();
        API.get_or_init(|| unsafe {
            let gstcuda = libloading::Library::new("libgstcuda-1.0.so.0").map_err(|e| format!("libgstcuda-1.0: {e}"))?;
            let cuda = libloading::Library::new("libcuda.so.1").map_err(|e| format!("libcuda: {e}"))?;
            let api = Api {
                is_cuda_memory: *gstcuda.get(b"gst_is_cuda_memory\0").map_err(|e| e.to_string())?,
                context_push: *gstcuda.get(b"gst_cuda_context_push\0").map_err(|e| e.to_string())?,
                context_pop: *gstcuda.get(b"gst_cuda_context_pop\0").map_err(|e| e.to_string())?,
                memcpy_2d: *cuda.get(b"cuMemcpy2D_v2\0").map_err(|e| e.to_string())?,
                _libs: (gstcuda, cuda),
            };
            Ok(api)
        })
        .as_ref()
        .map_err(|e| e.clone())
    }

    /// Copy `rows` rows (`width` bytes each) of plane 0 of a CUDA memory buffer to host memory.
    pub fn copy_band(buffer: &gst::BufferRef, offset: usize, stride: usize, width: usize, rows: usize) -> Result<Vec<u8>, String> {
        let api = api()?;
        unsafe {
            let mem = gst::ffi::gst_buffer_peek_memory(buffer.as_mut_ptr(), 0);
            if mem.is_null() || (api.is_cuda_memory)(mem) == 0 {
                return Err("not CUDA memory".into());
            }
            let ctx = (*(mem as *mut CudaMemoryHead)).context;
            let mut info = std::mem::MaybeUninit::<gst::ffi::GstMapInfo>::zeroed();
            if gst::ffi::gst_buffer_map(buffer.as_mut_ptr(), info.as_mut_ptr(), gst::ffi::GST_MAP_READ | GST_MAP_CUDA) == 0 {
                return Err("CUDA map failed".into());
            }
            let mut info = info.assume_init();
            let mut out = vec![0u8; width * rows];
            let p = Memcpy2D {
                src_x_in_bytes: 0,
                src_y: 0,
                src_memory_type: MEMORYTYPE_DEVICE,
                src_host: std::ptr::null(),
                src_device: info.data as u64 + offset as u64,
                src_array: std::ptr::null_mut(),
                src_pitch: stride,
                dst_x_in_bytes: 0,
                dst_y: 0,
                dst_memory_type: MEMORYTYPE_HOST,
                dst_host: out.as_mut_ptr() as *mut c_void,
                dst_device: 0,
                dst_array: std::ptr::null_mut(),
                dst_pitch: width,
                width_in_bytes: width,
                height: rows,
            };
            let pushed = (api.context_push)(ctx) != 0;
            let r = if pushed { (api.memcpy_2d)(&p) } else { -1 };
            if pushed {
                (api.context_pop)(std::ptr::null_mut());
            }
            gst::ffi::gst_buffer_unmap(buffer.as_mut_ptr(), &mut info);
            if r != 0 {
                return Err(format!("cuMemcpy2D failed: {r}"));
            }
            Ok(out)
        }
    }
}

fn pipeline_for(mode: &Mode, cam: u32, k: usize) -> String {
    match mode {
        Mode::Rtsp { url_base, tcp } => format!(
            "rtspsrc location={url_base}{cam:02} protocols={} latency=0 ! decodebin ! {} ! \
             video/x-raw,format=GRAY8 ! appsink name=sink{k} sync=false max-buffers=16 drop=false",
            if *tcp { "tcp" } else { "udp" },
            // Jetson hardware decoders output NVMM memory, which only nvvidconv can read
            if gst::ElementFactory::find("nvvidconv").is_some() { "nvvidconv" } else { "videoconvert" }
        ),
        Mode::CudaIpc { path_base } => format!(
            "cudaipcsrc address={path_base}{cam:02} ! appsink name=sink{k} sync=false max-buffers=16 drop=false \
             caps=video/x-raw(memory:CUDAMemory)"
        ),
        Mode::Grid { .. } => unreachable!(),
    }
}

fn decode_sample(sample: &gst::Sample, cuda: bool) -> Result<FrameId, DecodeError> {
    let bad = Err(DecodeError::NoContrast { black: 0.0, white: 0.0 });
    let Some(buffer) = sample.buffer() else { return bad };
    let Some(caps) = sample.caps() else { return bad };
    let Ok(vinfo) = gst_video::VideoInfo::from_caps(caps) else { return bad };
    let (w, h) = (vinfo.width(), vinfo.height());
    let (offset, stride) = match buffer.meta::<gst_video::VideoMeta>() {
        Some(m) => (m.offset()[0], m.stride()[0] as usize),
        None => (vinfo.offset()[0], vinfo.stride()[0] as usize),
    };
    let rows = framecode::Geometry::for_picture(w, h).band_rows() as usize;
    if cuda {
        return match cuda::copy_band(buffer, offset, stride, w as usize, rows) {
            Ok(band) => framecode::decode(&band, w as usize, w, h),
            Err(e) => {
                eprintln!("{e}");
                bad
            }
        };
    }
    let Ok(map) = buffer.map_readable() else { return bad };
    framecode::decode(&map[offset..offset + stride * rows], stride, w, h)
}

#[derive(Default)]
struct Summary {
    cam: u32,
    samples: usize,
    unreadable: usize,
    wrong_camera: usize,
    distinct: usize,
    span: u64,
    lost: u64,
    repeated: usize,
    backwards: usize,
    lat_ms: Vec<f64>,
    fps: f64,
}

fn summarize(cam: u32, recs: &[Rec], t_from: u64, t_to: u64) -> Summary {
    let mut s = Summary { cam, ..Default::default() };
    let mut seen = std::collections::BTreeMap::new();
    let mut prev: Option<u32> = None;
    let mut max_frame: Option<u32> = None;
    for r in recs.iter().filter(|r| r.arrival_us >= t_from && r.arrival_us < t_to) {
        s.samples += 1;
        let id = match r.result {
            Ok(id) => id,
            Err(_) => {
                s.unreadable += 1;
                continue;
            }
        };
        if id.camera as u32 != cam {
            s.wrong_camera += 1;
            continue;
        }
        if prev == Some(id.frame) {
            s.repeated += 1;
        }
        if let Some(m) = max_frame {
            if id.frame < m && Some(id.frame) != prev {
                s.backwards += 1;
            }
        }
        prev = Some(id.frame);
        max_frame = Some(max_frame.map_or(id.frame, |m| m.max(id.frame)));
        seen.entry(id.frame).or_insert_with(|| {
            let capture = framecode::unwrap_capture_us(id.capture_us, r.arrival_us);
            (r.arrival_us as f64 - capture as f64) / 1000.0
        });
    }
    if let (Some((&first, _)), Some((&last, _))) = (seen.iter().next(), seen.iter().next_back()) {
        s.distinct = seen.len();
        s.span = (last - first) as u64 + 1;
        s.lost = s.span - s.distinct as u64;
    }
    s.lat_ms = seen.values().copied().collect();
    s.lat_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s.fps = s.samples as f64 / ((t_to - t_from) as f64 / 1e6);
    s
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v[((v.len() - 1) as f64 * p).round() as usize]
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    gst::init()?;
    let cuda = matches!(args.mode, Mode::CudaIpc { .. });
    let cams: Vec<u32> = (args.first..args.first + args.count).collect();
    let recs: Arc<Vec<Mutex<Vec<Rec>>>> = Arc::new(cams.iter().map(|_| Mutex::new(Vec::new())).collect());

    if let Mode::Grid { file, cols, cell_w, cell_h, fps } = &args.mode {
        let pipeline = gst::parse::launch(&format!(
            "filesrc location={file} ! parsebin ! decodebin ! videoconvert ! video/x-raw,format=GRAY8 ! \
             appsink name=sink sync=false max-buffers=4 drop=false"
        ))?
        .downcast::<gst::Pipeline>()
        .map_err(|_| anyhow!("not a pipeline"))?;
        let sink = pipeline.by_name("sink").unwrap().downcast::<gst_app::AppSink>().unwrap();
        pipeline.set_state(gst::State::Playing)?;
        let mut index = 0u64;
        while let Ok(sample) = sink.pull_sample() {
            let (Some(buffer), Some(caps)) = (sample.buffer(), sample.caps()) else { continue };
            let Ok(vinfo) = gst_video::VideoInfo::from_caps(caps) else { continue };
            let stride = vinfo.stride()[0] as usize;
            let Ok(map) = buffer.map_readable() else { continue };
            let t = (index as f64 * 1e6 / fps) as u64;
            for (k, _) in cams.iter().enumerate() {
                let (x0, y0) = ((k as u32 % cols) * cell_w, (k as u32 / cols) * cell_h);
                if y0 + cell_h > vinfo.height() {
                    continue;
                }
                let start = y0 as usize * stride + x0 as usize;
                let result = framecode::decode(&map[start..], stride, *cell_w, *cell_h);
                recs[k].lock().unwrap().push(Rec { arrival_us: t, result });
            }
            index += 1;
        }
        let _ = pipeline.set_state(gst::State::Null);
        let t_to = (index as f64 * 1e6 / fps) as u64 + 1;
        return report(&args, &cams, &recs, (args.warmup * 1e6) as u64, t_to);
    }

    // One pipeline for all cameras, so all CUDA IPC readers share one CUDA context.
    let desc = cams.iter().enumerate().map(|(k, &cam)| pipeline_for(&args.mode, cam, k)).collect::<Vec<_>>().join(" ");
    let pipeline = gst::parse::launch(&desc)?.downcast::<gst::Pipeline>().map_err(|_| anyhow!("not a pipeline"))?;
    for (k, _) in cams.iter().enumerate() {
        let sink = pipeline.by_name(&format!("sink{k}")).unwrap().downcast::<gst_app::AppSink>().unwrap();
        let recs_c = recs.clone();
        sink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                    let arrival_us = now_us();
                    let result = decode_sample(&sample, cuda);
                    recs_c[k].lock().unwrap().push(Rec { arrival_us, result });
                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );
    }
    pipeline.set_state(gst::State::Playing)?;
    let start = Instant::now();
    let t_from = now_us() + (args.warmup * 1e6) as u64;
    let t_to = t_from + (args.seconds * 1e6) as u64;
    while start.elapsed() < Duration::from_secs_f64(args.warmup + args.seconds + 0.2) {
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = pipeline.set_state(gst::State::Null);
    report(&args, &cams, &recs, t_from, t_to)
}

fn report(args: &Args, cams: &[u32], recs: &Arc<Vec<Mutex<Vec<Rec>>>>, t_from: u64, t_to: u64) -> Result<()> {
    let file_mode = matches!(args.mode, Mode::Grid { .. });
    if let Some(path) = &args.csv {
        let mut out = String::from("camera,arrival_us,frame_camera,frame,capture_us,error\n");
        for (k, &cam) in cams.iter().enumerate() {
            for r in recs[k].lock().unwrap().iter() {
                match r.result {
                    Ok(id) => writeln!(out, "{cam},{},{},{},{},", r.arrival_us, id.camera, id.frame, id.capture_us)?,
                    Err(e) => writeln!(out, "{cam},{},,,,{e:?}", r.arrival_us)?,
                }
            }
        }
        std::fs::write(path, out)?;
    }
    let sums: Vec<Summary> = cams.iter().enumerate().map(|(k, &cam)| summarize(cam, &recs[k].lock().unwrap(), t_from, t_to)).collect();
    println!(
        "{:>4} {:>7} {:>8} {:>6} {:>5} {:>6} {:>6} {:>6} {:>7} {:>9} {:>9}",
        "cam", "fps", "distinct", "lost", "rep", "back", "unread", "wrong", "deliv%", "lat med", "lat p95"
    );
    let mut json = String::from("[\n");
    for s in &sums {
        let deliv = if s.span > 0 { 100.0 * s.distinct as f64 / s.span as f64 } else { 0.0 };
        let (lm, lp) = if file_mode { (f64::NAN, f64::NAN) } else { (pct(&s.lat_ms, 0.5), pct(&s.lat_ms, 0.95)) };
        println!(
            "{:>4} {:>7.2} {:>8} {:>6} {:>5} {:>6} {:>6} {:>6} {:>7.2} {:>9.1} {:>9.1}",
            s.cam, s.fps, s.distinct, s.lost, s.repeated, s.backwards, s.unreadable, s.wrong_camera, deliv, lm, lp
        );
        let _ = writeln!(
            json,
            "  {{\"camera\": {}, \"fps\": {:.3}, \"samples\": {}, \"distinct\": {}, \"span\": {}, \"lost\": {}, \"repeated\": {}, \
             \"backwards\": {}, \"unreadable\": {}, \"wrong_camera\": {}, \"delivered_pct\": {:.3}, \"lat_median_ms\": {:.2}, \"lat_p95_ms\": {:.2}}},",
            s.cam, s.fps, s.samples, s.distinct, s.span, s.lost, s.repeated, s.backwards, s.unreadable, s.wrong_camera, deliv, lm, lp
        );
    }
    let mut all_lat: Vec<f64> = sums.iter().flat_map(|s| s.lat_ms.iter().copied()).collect();
    all_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (distinct, span, lost) = sums.iter().fold((0usize, 0u64, 0u64), |a, s| (a.0 + s.distinct, a.1 + s.span, a.2 + s.lost));
    let total = |f: fn(&Summary) -> usize| sums.iter().map(f).sum::<usize>();
    let latency = if file_mode {
        String::new()
    } else {
        format!(", latency median {:.1} ms p95 {:.1} ms", pct(&all_lat, 0.5), pct(&all_lat, 0.95))
    };
    println!(
        "ALL: {} cameras, distinct {distinct} of {span} camera frames ({:.3} %), lost {lost}, repeated {}, backwards {}, \
         unreadable {}, wrong camera {}{latency}, worst camera {:.2} %",
        sums.len(),
        if span > 0 { 100.0 * distinct as f64 / span as f64 } else { 0.0 },
        total(|s| s.repeated),
        total(|s| s.backwards),
        total(|s| s.unreadable),
        total(|s| s.wrong_camera),
        sums.iter().filter(|s| s.span > 0).map(|s| 100.0 * s.distinct as f64 / s.span as f64).fold(f64::INFINITY, f64::min)
    );
    if let Some(path) = &args.json {
        json.truncate(json.trim_end_matches(",\n").len());
        json.push_str("\n]\n");
        std::fs::write(path, json)?;
    }
    Ok(())
}
