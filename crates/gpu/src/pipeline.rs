use simucad_core::error::GpuError;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Compute pipeline helpers
// ---------------------------------------------------------------------------

/// Compile a WGSL shader source and build a `ComputePipeline` from it.
///
/// # Errors
///
/// Returns `GpuError::ShaderCompilation` if the shader source is invalid, or
/// `GpuError::DispatchError` if the pipeline layout cannot be inferred.
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    shader_source: &str,
    entry_point: &str,
) -> Result<wgpu::ComputePipeline, GpuError> {
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: None, // auto layout
        module: &shader_module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok(pipeline)
}

/// Create a GPU storage buffer initialised with `data`.
///
/// The buffer is created with `STORAGE | COPY_SRC | COPY_DST` usage flags so
/// it can be bound to compute shaders and also read back to the host.
pub fn create_storage_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    data: &[T],
    label: &str,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create a GPU uniform buffer initialised with a single value.
pub fn create_uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    data: &T,
    label: &str,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Read `count` elements of type `T` back from a GPU buffer.
///
/// Internally this creates a staging buffer with `MAP_READ | COPY_DST`,
/// copies the source buffer into it, maps it, and returns the data.
///
/// # Errors
///
/// Returns `GpuError::BufferError` if the map or copy fails.
pub fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<T>, GpuError> {
    let size = (count * std::mem::size_of::<T>()) as u64;

    // Create a staging buffer that can be mapped for reading.
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Encode a copy from the source buffer to the staging buffer.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("read_buffer_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    // Map the staging buffer and block until ready.
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    rx.recv()
        .map_err(|e| GpuError::BufferError(format!("Channel recv failed: {e}")))?
        .map_err(|e| GpuError::BufferError(format!("Buffer map failed: {e}")))?;

    let data = slice.get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

    drop(data);
    staging.unmap();

    Ok(result)
}
