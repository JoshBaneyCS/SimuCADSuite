//! 3D renderer using wgpu, integrated with egui via paint callbacks.
//!
//! Provides GPU render pipelines for point clouds (particles) and line
//! primitives (mesh wireframe, grid, axes). Resources are stored in
//! `egui_wgpu::CallbackResources` and drawn via [`SceneCallback`].

use bytemuck::{Pod, Zeroable};
use eframe::wgpu;

// ---------------------------------------------------------------------------
// Vertex type
// ---------------------------------------------------------------------------

/// A colored 3D vertex shared by all render primitives.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

// ---------------------------------------------------------------------------
// GPU resources (stored in CallbackResources)
// ---------------------------------------------------------------------------

/// Maximum number of point vertices (particles) per frame.
const MAX_POINTS: u32 = 200_000;
/// Maximum number of line vertices (grid + wireframe + axes) per frame.
const MAX_LINE_VERTICES: u32 = 500_000;

/// Long-lived GPU resources for 3D rendering.
///
/// Created once during app initialization and stored in
/// `egui_wgpu::CallbackResources`.
pub struct Renderer3DResources {
    pub point_pipeline: wgpu::RenderPipeline,
    pub line_pipeline: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub point_buffer: wgpu::Buffer,
    pub line_buffer: wgpu::Buffer,
    pub point_count: u32,
    pub line_vertex_count: u32,
}

// ---------------------------------------------------------------------------
// WGSL shader
// ---------------------------------------------------------------------------

const SHADER_SOURCE: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

// ---------------------------------------------------------------------------
// Resource creation
// ---------------------------------------------------------------------------

impl Renderer3DResources {
    /// Create all GPU resources (pipelines, buffers, bind groups).
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("3d_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("3d_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // 64 bytes = mat4x4<f32>.
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3d_uniform_buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("3d_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("3d_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let make_pipeline = |topology: wgpu::PrimitiveTopology, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let point_pipeline =
            make_pipeline(wgpu::PrimitiveTopology::PointList, "3d_point_pipeline");
        let line_pipeline =
            make_pipeline(wgpu::PrimitiveTopology::LineList, "3d_line_pipeline");

        let vertex_size = std::mem::size_of::<Vertex3D>() as u64;

        let point_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3d_point_buffer"),
            size: MAX_POINTS as u64 * vertex_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3d_line_buffer"),
            size: MAX_LINE_VERTICES as u64 * vertex_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            point_pipeline,
            line_pipeline,
            uniform_buffer,
            bind_group,
            point_buffer,
            line_buffer,
            point_count: 0,
            line_vertex_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame paint callback
// ---------------------------------------------------------------------------

/// Per-frame data sent to the GPU via the egui paint callback system.
pub struct SceneCallback {
    /// Combined view-projection matrix (column-major).
    pub view_proj: [[f32; 4]; 4],
    /// Point cloud vertices (particles).
    pub point_vertices: Vec<Vertex3D>,
    /// Line vertices (grid, axes, wireframe).
    pub line_vertices: Vec<Vertex3D>,
}

impl eframe::egui_wgpu::CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &mut Renderer3DResources = callback_resources.get_mut().unwrap();

        // Upload view-projection uniform.
        queue.write_buffer(
            &resources.uniform_buffer,
            0,
            bytemuck::cast_slice(&self.view_proj),
        );

        // Upload point vertices.
        let point_count = self.point_vertices.len().min(MAX_POINTS as usize);
        if point_count > 0 {
            queue.write_buffer(
                &resources.point_buffer,
                0,
                bytemuck::cast_slice(&self.point_vertices[..point_count]),
            );
        }
        resources.point_count = point_count as u32;

        // Upload line vertices.
        let line_count = self.line_vertices.len().min(MAX_LINE_VERTICES as usize);
        if line_count > 0 {
            queue.write_buffer(
                &resources.line_buffer,
                0,
                bytemuck::cast_slice(&self.line_vertices[..line_count]),
            );
        }
        resources.line_vertex_count = line_count as u32;

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        let resources: &Renderer3DResources = callback_resources.get().unwrap();

        // Draw lines (grid, wireframe, axes) first.
        if resources.line_vertex_count > 0 {
            render_pass.set_pipeline(&resources.line_pipeline);
            render_pass.set_bind_group(0, &resources.bind_group, &[]);
            render_pass.set_vertex_buffer(0, resources.line_buffer.slice(..));
            render_pass.draw(0..resources.line_vertex_count, 0..1);
        }

        // Draw points (particles) on top.
        if resources.point_count > 0 {
            render_pass.set_pipeline(&resources.point_pipeline);
            render_pass.set_bind_group(0, &resources.bind_group, &[]);
            render_pass.set_vertex_buffer(0, resources.point_buffer.slice(..));
            render_pass.draw(0..resources.point_count, 0..1);
        }
    }
}
