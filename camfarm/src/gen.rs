//! GPU picture generator: renders test camera pictures (NV12) with a compute shader into mappable staging buffers,
//! one submit per batch. A batch is collected one tick after its submit, so the render thread never waits for the
//! GPU, and its pictures are handed out zero-copy: a staging buffer stays mapped until GStreamer drops the picture.

use crate::prof::zone;
use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    pub width: u32,
    pub height: u32,
    pub band_rows: u32,
    pub cell_w: u32,
    pub cell_h: u32,
    pub cols: u32,
    pub frame: u32,
    pub seed: u32,
    pub time: f32,
    pub noise: f32,
    pub hue: f32,
    pub speed: f32,
    pub bits: [u32; 4],
}

/// Per-job GPU resources of a batch, reused by the next batches (the queue orders them).
struct Scratch {
    params: wgpu::Buffer,
    out: wgpu::Buffer,
    _low: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

struct Batch {
    submission: wgpu::SubmissionIndex,
    staging: Vec<wgpu::Buffer>,
}

/// One rendered picture, readable until dropped; then its staging buffer goes back to the generator.
pub struct Picture {
    view: Option<wgpu::BufferView>,
    buffer: Option<wgpu::Buffer>,
    released: Arc<Mutex<Vec<wgpu::Buffer>>>,
}

impl AsRef<[u8]> for Picture {
    fn as_ref(&self) -> &[u8] {
        self.view.as_deref().unwrap_or(&[])
    }
}

impl Drop for Picture {
    fn drop(&mut self) {
        self.view = None;
        if let Some(buffer) = self.buffer.take() {
            self.released.lock().unwrap().push(buffer);
        }
    }
}

pub struct Generator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    field: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pub width: u32,
    pub height: u32,
    pub frame_bytes: u64,
    pub adapter_name: String,
    scratch: Vec<Scratch>,
    free: Vec<wgpu::Buffer>,
    released: Arc<Mutex<Vec<wgpu::Buffer>>>,
    in_flight: Option<Batch>,
}

impl Generator {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        if width % 4 != 0 || height % 2 != 0 {
            return Err(anyhow!("width must be a multiple of 4 and height even"));
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let adapter_name = format!("{} ({:?})", adapter.get_info().name, adapter.get_info().backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("picture"),
            source: wgpu::ShaderSource::Wgsl(include_str!("picture.wgsl").into()),
        });
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1),
                storage(2),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let compute = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (pipeline, field) = (compute("main"), compute("field"));
        Ok(Self {
            device,
            queue,
            pipeline,
            field,
            layout,
            width,
            height,
            frame_bytes: (width * height * 3 / 2) as u64,
            adapter_name,
            scratch: Vec::new(),
            free: Vec::new(),
            released: Arc::new(Mutex::new(Vec::new())),
            in_flight: None,
        })
    }

    /// Allocate up front: GPU resources for `batch` jobs per submit and `staging` pictures in flight.
    pub fn reserve(&mut self, batch: usize, staging: usize) {
        while self.scratch.len() < batch {
            let scratch = self.new_scratch();
            self.scratch.push(scratch);
        }
        while self.free.len() < staging {
            let buffer = self.new_staging();
            self.free.push(buffer);
        }
    }

    fn new_scratch(&self) -> Scratch {
        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.frame_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let low = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: ((self.width / 4) * (self.height / 4) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: low.as_entire_binding() },
            ],
        });
        Scratch { params, out, _low: low, bind }
    }

    fn new_staging(&self) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.frame_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Render all jobs in one submit. Their pictures come back from the next `collect`.
    pub fn submit(&mut self, jobs: &[Params]) {
        zone!("gpu submit");
        for buffer in std::mem::take(&mut *self.released.lock().unwrap()) {
            buffer.unmap();
            self.free.push(buffer);
        }
        if jobs.is_empty() {
            return;
        }
        self.reserve(jobs.len(), jobs.len());
        let staging: Vec<wgpu::Buffer> = self.free.split_off(self.free.len() - jobs.len());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for (scratch, params) in self.scratch.iter().zip(jobs) {
            self.queue.write_buffer(&scratch.params, 0, bytemuck::bytes_of(params));
        }
        {
            // all fields first, then all pictures: one pipeline switch per batch
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.field);
            for scratch in &self.scratch[..jobs.len()] {
                pass.set_bind_group(0, &scratch.bind, &[]);
                pass.dispatch_workgroups((self.width / 4).div_ceil(64), self.height / 4, 1);
            }
            pass.set_pipeline(&self.pipeline);
            for scratch in &self.scratch[..jobs.len()] {
                pass.set_bind_group(0, &scratch.bind, &[]);
                pass.dispatch_workgroups((self.width / 4).div_ceil(64), self.height + self.height / 2, 1);
            }
        }
        for (scratch, buffer) in self.scratch.iter().zip(&staging) {
            enc.copy_buffer_to_buffer(&scratch.out, 0, buffer, 0, self.frame_bytes);
        }
        let submission = self.queue.submit([enc.finish()]);
        for buffer in &staging {
            buffer.slice(..).map_async(wgpu::MapMode::Read, |r| {
                if let Err(e) = r {
                    eprintln!("map failed: {e:?}");
                }
            });
        }
        self.in_flight = Some(Batch { submission, staging });
    }

    /// Wait for the last submitted batch (normally finished long ago) and return its pictures in job order.
    pub fn collect(&mut self) -> Result<Vec<Picture>> {
        let Some(batch) = self.in_flight.take() else { return Ok(Vec::new()) };
        {
            zone!("gpu wait");
            self.device
                .poll(wgpu::PollType::Wait { submission_index: Some(batch.submission), timeout: None })?;
        }
        batch
            .staging
            .into_iter()
            .map(|buffer| {
                let view = buffer.slice(..).get_mapped_range().map_err(|e| anyhow!("map range: {e:?}"))?;
                Ok(Picture { view: Some(view), buffer: Some(buffer), released: self.released.clone() })
            })
            .collect()
    }
}
