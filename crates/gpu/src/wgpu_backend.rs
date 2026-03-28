use bytemuck::{Pod, Zeroable};
use simucad_core::error::GpuError;
use simucad_core::types::{Particle, Vec3};
use tracing::{debug, info};

use crate::backend::ComputeBackend;
use crate::capabilities::GpuCapabilities;
use crate::pipeline;
use crate::profiler::GpuProfiler;

// ---------------------------------------------------------------------------
// GPU-compatible data types (f32, 16-byte aligned)
// ---------------------------------------------------------------------------

/// GPU-side particle representation.
///
/// GPU compute shaders work with f32 and require vec4-aligned data. Each
/// particle occupies two vec4 slots: position + padding, velocity + padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuParticle {
    pub px: f32,
    pub py: f32,
    pub pz: f32,
    pub _pad0: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub _pad1: f32,
}

impl GpuParticle {
    fn from_particle(p: &Particle) -> Self {
        Self {
            px: p.position.x as f32,
            py: p.position.y as f32,
            pz: p.position.z as f32,
            _pad0: 0.0,
            vx: p.velocity.x as f32,
            vy: p.velocity.y as f32,
            vz: p.velocity.z as f32,
            _pad1: 0.0,
        }
    }

    fn to_particle(&self) -> Particle {
        Particle {
            position: Vec3::new(self.px as f64, self.py as f64, self.pz as f64),
            velocity: Vec3::new(self.vx as f64, self.vy as f64, self.vz as f64),
        }
    }
}

/// GPU-side vec3 with padding to vec4 alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

impl GpuVec3 {
    fn from_vec3(v: &Vec3) -> Self {
        Self {
            x: v.x as f32,
            y: v.y as f32,
            z: v.z as f32,
            _pad: 0.0,
        }
    }

    fn to_vec3(&self) -> Vec3 {
        Vec3::new(self.x as f64, self.y as f64, self.z as f64)
    }
}

/// Uniform data for the particle advection shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AdvectUniforms {
    vx: f32,
    vy: f32,
    vz: f32,
    dt: f32,
}

/// Uniform data for the velocity field shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VelocityFieldUniforms {
    num_particles: u32,
    num_nodes: u32,
    _pad0: u32,
    _pad1: u32,
}

// ---------------------------------------------------------------------------
// WgpuBackend
// ---------------------------------------------------------------------------

/// GPU compute backend powered by wgpu.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    capabilities: GpuCapabilities,
    profiler: GpuProfiler,
}

impl WgpuBackend {
    /// Request a GPU adapter and create a device + queue.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::NoAdapter` if no suitable adapter is found, or
    /// `GpuError::DeviceCreation` if the device cannot be created.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuError::NoAdapter)?;

        debug!("wgpu adapter: {:?}", adapter.get_info());

        // Request timestamp query features if available, otherwise proceed without.
        let adapter_features = adapter.features();
        let mut required_features = wgpu::Features::empty();
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("simucad_compute_device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuError::DeviceCreation(e.to_string()))?;

        let capabilities = GpuCapabilities::detect(&adapter, &device);
        info!("GPU capabilities detected:\n{capabilities}");

        // Enable profiling by default -- it no-ops if timestamps are unsupported.
        let profiler = GpuProfiler::new(&device, &queue, true);

        Ok(Self {
            device,
            queue,
            capabilities,
            profiler,
        })
    }

    /// Access the detected GPU capabilities.
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }
}

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &str {
        "wgpu"
    }

    fn advect_particles(
        &self,
        particles: &mut [Particle],
        velocity: Vec3,
        dt: f64,
    ) -> Result<(), GpuError> {
        if particles.is_empty() {
            return Ok(());
        }

        let num_particles = particles.len();

        // Convert to GPU layout
        let gpu_particles: Vec<GpuParticle> =
            particles.iter().map(GpuParticle::from_particle).collect();

        // Create buffers
        let particle_buf =
            pipeline::create_storage_buffer(&self.device, &gpu_particles, "advect_particles");

        let uniforms = AdvectUniforms {
            vx: velocity.x as f32,
            vy: velocity.y as f32,
            vz: velocity.z as f32,
            dt: dt as f32,
        };
        let uniform_buf =
            pipeline::create_uniform_buffer(&self.device, &uniforms, "advect_uniforms");

        // Build compute pipeline
        let shader_source = include_str!("../shaders/particle_advect.wgsl");
        let compute_pipeline =
            pipeline::create_compute_pipeline(&self.device, shader_source, "advect_main")?;

        // Bind group -- auto-layout from the pipeline
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("advect_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch -- use recommended workgroup size (shader compiled with 256)
        let wg_size = self.capabilities.recommend_workgroup_size(num_particles);
        let workgroup_count = ((num_particles as u32) + wg_size - 1) / wg_size;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("advect_encoder"),
            });

        self.profiler.write_start_timestamp(&mut encoder);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("advect_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        self.profiler.write_end_timestamp(&mut encoder);
        self.profiler.resolve(&mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()));

        if let Some(ms) = self.profiler.read_elapsed_ms(&self.device, &self.queue) {
            debug!("advect_particles GPU time: {ms:.3} ms ({num_particles} particles)");
        }

        // Read back
        let result: Vec<GpuParticle> =
            pipeline::read_buffer(&self.device, &self.queue, &particle_buf, num_particles)?;

        // Write results back into the particle slice
        for (dst, src) in particles.iter_mut().zip(result.iter()) {
            *dst = src.to_particle();
        }

        Ok(())
    }

    fn compute_velocity_field(
        &self,
        particles: &[Particle],
        node_positions: &[Vec3],
    ) -> Result<Vec<Vec3>, GpuError> {
        if particles.is_empty() {
            return Ok(vec![Vec3::ZERO; node_positions.len()]);
        }
        if node_positions.is_empty() {
            return Ok(Vec::new());
        }

        let num_particles = particles.len();
        let num_nodes = node_positions.len();

        // Convert to GPU layout
        let gpu_particles: Vec<GpuVec3> = particles
            .iter()
            .map(|p| GpuVec3::from_vec3(&p.position))
            .collect();

        let gpu_nodes: Vec<GpuVec3> = node_positions
            .iter()
            .map(GpuVec3::from_vec3)
            .collect();

        let gpu_output = vec![GpuVec3 { x: 0.0, y: 0.0, z: 0.0, _pad: 0.0 }; num_nodes];

        // Create buffers
        let particle_buf =
            pipeline::create_storage_buffer(&self.device, &gpu_particles, "vf_particles");
        let node_buf =
            pipeline::create_storage_buffer(&self.device, &gpu_nodes, "vf_nodes");
        let output_buf =
            pipeline::create_storage_buffer(&self.device, &gpu_output, "vf_output");

        let uniforms = VelocityFieldUniforms {
            num_particles: num_particles as u32,
            num_nodes: num_nodes as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf =
            pipeline::create_uniform_buffer(&self.device, &uniforms, "vf_uniforms");

        // Build compute pipeline
        let shader_source = include_str!("../shaders/velocity_field.wgsl");
        let compute_pipeline =
            pipeline::create_compute_pipeline(&self.device, shader_source, "velocity_field_main")?;

        // Bind group
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vf_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch -- one workgroup per node
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vf_encoder"),
            });

        self.profiler.write_start_timestamp(&mut encoder);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_nodes as u32, 1, 1);
        }
        self.profiler.write_end_timestamp(&mut encoder);
        self.profiler.resolve(&mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()));

        if let Some(ms) = self.profiler.read_elapsed_ms(&self.device, &self.queue) {
            debug!(
                "compute_velocity_field GPU time: {ms:.3} ms ({num_nodes} nodes, {num_particles} particles)"
            );
        }

        // Read back
        let result: Vec<GpuVec3> =
            pipeline::read_buffer(&self.device, &self.queue, &output_buf, num_nodes)?;

        Ok(result.iter().map(GpuVec3::to_vec3).collect())
    }
}
