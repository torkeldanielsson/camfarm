//! GPU picture generator: renders test camera pictures (NV12) with a compute shader and reads them back in
//! batches, one submit and one wait per batch.

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};

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

pub struct Slot {
    params: wgpu::Buffer,
    out: wgpu::Buffer,
    _low: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("picture"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let field = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("field"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("field"),
            compilation_options: Default::default(),
            cache: None,
        });
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
        })
    }

    pub fn slot(&self) -> Slot {
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
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.frame_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
        Slot { params, out, _low: low, staging, bind }
    }

    /// Render all jobs in one submit, then hand each picture's bytes to `sink` (job index, NV12 bytes).
    pub fn render_batch(&self, jobs: &[(&Slot, Params)], mut sink: impl FnMut(usize, &[u8])) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for (slot, params) in jobs {
            self.queue.write_buffer(&slot.params, 0, bytemuck::bytes_of(params));
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.field);
                pass.set_bind_group(0, &slot.bind, &[]);
                pass.dispatch_workgroups((self.width / 4).div_ceil(64), self.height / 4, 1);
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &slot.bind, &[]);
                pass.dispatch_workgroups((self.width / 4).div_ceil(64), self.height + self.height / 2, 1);
            }
            enc.copy_buffer_to_buffer(&slot.out, 0, &slot.staging, 0, self.frame_bytes);
        }
        self.queue.submit([enc.finish()]);
        for (slot, _) in jobs {
            slot.staging.slice(..).map_async(wgpu::MapMode::Read, |r| {
                if let Err(e) = r {
                    eprintln!("map failed: {e:?}");
                }
            });
        }
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        for (i, (slot, _)) in jobs.iter().enumerate() {
            {
                let data = slot.staging.slice(..).get_mapped_range().map_err(|e| anyhow!("map range: {e:?}"))?;
                sink(i, &data);
            }
            slot.staging.unmap();
        }
        Ok(())
    }
}
