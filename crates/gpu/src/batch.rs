use std::sync::Arc;

use simucad_core::error::GpuError;
use tracing::{debug, info};
use wgpu::util::DeviceExt;

use crate::capabilities::GpuCapabilities;
use crate::pipeline;

// ---------------------------------------------------------------------------
// BatchDispatcher -- split oversized dispatches into GPU-safe chunks
// ---------------------------------------------------------------------------

/// Splits large data arrays into batches that fit within GPU buffer limits,
/// dispatches each batch through a compute pipeline, and reassembles results.
pub struct BatchDispatcher {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_batch_bytes: u64,
}

impl BatchDispatcher {
    /// Create a new dispatcher. `caps` is used to determine the maximum bytes
    /// per batch (derived from `max_buffer_size` with a safety margin).
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        caps: &GpuCapabilities,
    ) -> Self {
        // Leave 10% headroom so uniform / staging buffers fit too.
        let max_batch_bytes = (caps.max_buffer_size as f64 * 0.9) as u64;
        debug!(
            "BatchDispatcher created, max batch size: {} MiB",
            max_batch_bytes / (1024 * 1024)
        );
        Self {
            device,
            queue,
            max_batch_bytes,
        }
    }

    /// Dispatch a compute pipeline over `data`, automatically splitting into
    /// batches if the data exceeds the GPU buffer limit.
    ///
    /// Each batch:
    /// 1. Uploads a slice of `data` to a storage buffer at binding 0.
    /// 2. Uploads `uniforms` to a uniform buffer at binding 1.
    /// 3. Dispatches the pipeline with `ceil(batch_len / workgroup_size)` groups.
    /// 4. Reads back the storage buffer and writes results into `data`.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::BufferError` or `GpuError::DispatchError` on failure.
    pub fn dispatch_batched<T: bytemuck::Pod>(
        &self,
        pipeline: &wgpu::ComputePipeline,
        data: &mut [T],
        uniforms: &[u8],
        workgroup_size: u32,
    ) -> Result<(), GpuError> {
        if data.is_empty() {
            return Ok(());
        }

        let elem_size = std::mem::size_of::<T>() as u64;
        let max_elements = (self.max_batch_bytes / elem_size) as usize;
        let total = data.len();
        let num_batches = (total + max_elements - 1) / max_elements;

        if num_batches > 1 {
            info!(
                "Data exceeds single buffer limit, splitting into {num_batches} batches \
                 ({total} elements, max {max_elements} per batch)"
            );
        }

        for batch_idx in 0..num_batches {
            let start = batch_idx * max_elements;
            let end = (start + max_elements).min(total);
            let batch = &data[start..end];
            let batch_len = batch.len();

            debug!(
                "Batch {}/{}: elements {}..{} ({batch_len})",
                batch_idx + 1,
                num_batches,
                start,
                end
            );

            // Upload data slice
            let data_buf =
                pipeline::create_storage_buffer(&self.device, batch, "batch_data");

            // Upload uniforms
            let uniform_buf = self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("batch_uniforms"),
                    contents: uniforms,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                },
            );

            // Bind group from auto-layout
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group =
                self.device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("batch_bind_group"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: data_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: uniform_buf.as_entire_binding(),
                            },
                        ],
                    });

            // Dispatch
            let workgroup_count =
                ((batch_len as u32) + workgroup_size - 1) / workgroup_size;
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("batch_encoder"),
                },
            );
            {
                let mut pass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("batch_pass"),
                        timestamp_writes: None,
                    });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroup_count, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));

            // Read back
            let result: Vec<T> = pipeline::read_buffer(
                &self.device,
                &self.queue,
                &data_buf,
                batch_len,
            )?;

            // Write back into caller's slice
            data[start..end].copy_from_slice(&result);
        }

        Ok(())
    }

    /// Compute how many batches would be needed for the given element count.
    pub fn estimate_batches<T>(&self, element_count: usize) -> usize {
        let elem_size = std::mem::size_of::<T>() as u64;
        let max_elements = (self.max_batch_bytes / elem_size) as usize;
        if max_elements == 0 {
            return element_count; // degenerate case
        }
        (element_count + max_elements - 1) / max_elements
    }
}

// ---------------------------------------------------------------------------
// Tests (CPU-side logic -- no GPU needed)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capabilities::GpuCapabilities;

    fn fake_caps(max_buffer: u64) -> GpuCapabilities {
        GpuCapabilities {
            adapter_name: "Test".to_string(),
            backend: "Vulkan".to_string(),
            max_buffer_size: max_buffer,
            max_compute_workgroup_size: [256, 1, 1],
            max_compute_workgroups_per_dimension: 65535,
            max_storage_buffers: 8,
            supports_f16: false,
            supports_timestamp_query: false,
        }
    }

    #[test]
    fn estimate_batches_single() {
        // 1 GiB buffer, 4-byte elements, 1000 elements => 1 batch
        let caps = fake_caps(1024 * 1024 * 1024);
        // We can't create a real BatchDispatcher without a device, so test
        // the math directly.
        let max_batch_bytes = (caps.max_buffer_size as f64 * 0.9) as u64;
        let elem_size = std::mem::size_of::<f32>() as u64;
        let max_elements = (max_batch_bytes / elem_size) as usize;
        let total = 1000usize;
        let batches = (total + max_elements - 1) / max_elements;
        assert_eq!(batches, 1);
    }

    #[test]
    fn estimate_batches_multiple() {
        // 100-byte buffer (90 after margin), 4-byte elements, 100 elements
        let max_batch_bytes = (100_u64 as f64 * 0.9) as u64; // 90
        let elem_size = std::mem::size_of::<f32>() as u64; // 4
        let max_elements = (max_batch_bytes / elem_size) as usize; // 22
        let total = 100usize;
        let batches = (total + max_elements - 1) / max_elements;
        assert_eq!(batches, 5); // ceil(100/22) = 5
    }

    #[test]
    fn estimate_batches_exact_fit() {
        let max_batch_bytes = 40_u64; // exactly 10 f32s
        let elem_size = std::mem::size_of::<f32>() as u64;
        let max_elements = (max_batch_bytes / elem_size) as usize; // 10
        let total = 10usize;
        let batches = (total + max_elements - 1) / max_elements;
        assert_eq!(batches, 1);
    }

    #[test]
    fn estimate_batches_empty() {
        let max_batch_bytes = 40_u64;
        let elem_size = std::mem::size_of::<f32>() as u64;
        let max_elements = (max_batch_bytes / elem_size) as usize;
        let total = 0usize;
        let batches = if total == 0 {
            0
        } else {
            (total + max_elements - 1) / max_elements
        };
        assert_eq!(batches, 0);
    }
}
