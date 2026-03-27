use std::fmt;

// ---------------------------------------------------------------------------
// GPU capability detection
// ---------------------------------------------------------------------------

/// Describes the capabilities and limits of the active GPU adapter.
///
/// Used to make runtime decisions about workgroup sizes, buffer allocation
/// strategies, and whether optional features (f16, timestamp queries) are
/// available.
pub struct GpuCapabilities {
    pub adapter_name: String,
    /// Backend API name: "Metal", "Vulkan", "DX12", "GL", etc.
    pub backend: String,
    pub max_buffer_size: u64,
    pub max_compute_workgroup_size: [u32; 3],
    pub max_compute_workgroups_per_dimension: u32,
    pub max_storage_buffers: u32,
    pub supports_f16: bool,
    pub supports_timestamp_query: bool,
}

impl GpuCapabilities {
    /// Query the adapter and device to populate capability information.
    pub fn detect(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Self {
        let info = adapter.get_info();
        let limits = device.limits();
        let features = device.features();

        let backend = match info.backend {
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Dx12 => "DX12",
            wgpu::Backend::Gl => "GL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Unknown",
        };

        Self {
            adapter_name: info.name.clone(),
            backend: backend.to_string(),
            max_buffer_size: limits.max_buffer_size as u64,
            max_compute_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_buffers: limits.max_storage_buffers_per_shader_stage,
            supports_f16: features.contains(wgpu::Features::SHADER_F16),
            supports_timestamp_query: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        }
    }

    /// Recommend a workgroup size for a 1-D dispatch over `particle_count` items.
    ///
    /// Picks the largest power-of-two that is <= the device's max workgroup X
    /// dimension, capped at 256 (a common sweet spot for compute shaders).
    pub fn recommend_workgroup_size(&self, _particle_count: usize) -> u32 {
        let max_x = self.max_compute_workgroup_size[0];
        // Cap at 256 -- most shaders are compiled with @workgroup_size(256).
        let cap = max_x.min(256);
        // Round down to nearest power of two.
        if cap == 0 {
            return 1;
        }
        1 << (31 - cap.leading_zeros())
    }

    /// Returns `true` if `byte_count` fits in a single GPU buffer.
    pub fn can_fit_in_buffer(&self, byte_count: u64) -> bool {
        byte_count <= self.max_buffer_size
    }
}

impl fmt::Display for GpuCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GPU Capabilities:")?;
        writeln!(f, "  Adapter:          {}", self.adapter_name)?;
        writeln!(f, "  Backend:          {}", self.backend)?;
        writeln!(
            f,
            "  Max buffer size:  {} MiB",
            self.max_buffer_size / (1024 * 1024)
        )?;
        writeln!(
            f,
            "  Max workgroup:    [{}, {}, {}]",
            self.max_compute_workgroup_size[0],
            self.max_compute_workgroup_size[1],
            self.max_compute_workgroup_size[2],
        )?;
        writeln!(
            f,
            "  Max workgroups/dim: {}",
            self.max_compute_workgroups_per_dimension
        )?;
        writeln!(
            f,
            "  Max storage bufs: {}",
            self.max_storage_buffers
        )?;
        writeln!(f, "  f16 support:      {}", self.supports_f16)?;
        write!(
            f,
            "  Timestamp query:  {}",
            self.supports_timestamp_query
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a GpuCapabilities with controllable values (no GPU needed).
    fn fake_caps(max_x: u32, max_buffer: u64) -> GpuCapabilities {
        GpuCapabilities {
            adapter_name: "Test Adapter".to_string(),
            backend: "Vulkan".to_string(),
            max_buffer_size: max_buffer,
            max_compute_workgroup_size: [max_x, 1, 1],
            max_compute_workgroups_per_dimension: 65535,
            max_storage_buffers: 8,
            supports_f16: false,
            supports_timestamp_query: false,
        }
    }

    #[test]
    fn recommend_workgroup_size_capped_at_256() {
        let caps = fake_caps(1024, 128 * 1024 * 1024);
        assert_eq!(caps.recommend_workgroup_size(10_000), 256);
    }

    #[test]
    fn recommend_workgroup_size_small_device() {
        let caps = fake_caps(64, 128 * 1024 * 1024);
        assert_eq!(caps.recommend_workgroup_size(10_000), 64);
    }

    #[test]
    fn recommend_workgroup_size_non_power_of_two_max() {
        // If the device reports a non-power-of-two max, round down.
        let caps = fake_caps(200, 128 * 1024 * 1024);
        assert_eq!(caps.recommend_workgroup_size(10_000), 128);
    }

    #[test]
    fn recommend_workgroup_size_zero_max() {
        let caps = fake_caps(0, 128 * 1024 * 1024);
        assert_eq!(caps.recommend_workgroup_size(10_000), 1);
    }

    #[test]
    fn can_fit_in_buffer_yes() {
        let caps = fake_caps(256, 256 * 1024 * 1024);
        assert!(caps.can_fit_in_buffer(100 * 1024 * 1024));
    }

    #[test]
    fn can_fit_in_buffer_exact() {
        let caps = fake_caps(256, 1000);
        assert!(caps.can_fit_in_buffer(1000));
    }

    #[test]
    fn can_fit_in_buffer_no() {
        let caps = fake_caps(256, 1000);
        assert!(!caps.can_fit_in_buffer(1001));
    }

    #[test]
    fn display_impl_does_not_panic() {
        let caps = fake_caps(256, 256 * 1024 * 1024);
        let s = format!("{caps}");
        assert!(s.contains("Test Adapter"));
        assert!(s.contains("Vulkan"));
        assert!(s.contains("256 MiB"));
    }
}
