use simucad_core::error::GpuError;
use simucad_core::types::{Particle, Vec3};
use tracing::{info, warn};

use crate::cpu_backend::CpuBackend;
use crate::wgpu_backend::WgpuBackend;

// ---------------------------------------------------------------------------
// ComputeBackend trait -- abstraction over GPU and CPU compute
// ---------------------------------------------------------------------------

/// Trait that all compute backends (GPU, CPU fallback) must implement.
///
/// Every method takes shared or mutable references to domain types and returns
/// `Result<_, GpuError>` so callers can handle failures uniformly regardless
/// of which backend is active.
pub trait ComputeBackend: Send + Sync {
    /// Human-readable name of this backend (e.g. "wgpu", "cpu-rayon").
    fn name(&self) -> &str;

    /// Advect every particle by `velocity * dt`.
    ///
    /// After this call, each particle's position is updated in-place:
    ///   `particle.position += velocity * dt`
    fn advect_particles(
        &self,
        particles: &mut [Particle],
        velocity: Vec3,
        dt: f64,
    ) -> Result<(), GpuError>;

    /// Compute a velocity field at the given node positions.
    ///
    /// For each node, the velocity is the mean displacement from the node to
    /// every particle:
    ///   `v_node = (1/N) * sum(particle.position - node_position)`
    ///
    /// Returns one `Vec3` per node position.
    fn compute_velocity_field(
        &self,
        particles: &[Particle],
        node_positions: &[Vec3],
    ) -> Result<Vec<Vec3>, GpuError>;
}

// ---------------------------------------------------------------------------
// Backend selection -- tries wgpu first, falls back to CPU
// ---------------------------------------------------------------------------

/// Attempt to create a wgpu-backed compute backend. If the GPU is not
/// available (no adapter, device creation failure, CI environment, etc.),
/// fall back to the multi-threaded CPU backend built on rayon.
pub fn select_backend() -> Box<dyn ComputeBackend> {
    match WgpuBackend::new() {
        Ok(backend) => {
            info!("GPU compute backend initialised (wgpu)");
            Box::new(backend)
        }
        Err(e) => {
            warn!("GPU unavailable ({e}), falling back to CPU backend");
            Box::new(CpuBackend::new())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_backend_returns_something() {
        // Should never panic -- at worst we get the CPU fallback.
        let backend = select_backend();
        let name = backend.name();
        assert!(name == "wgpu" || name == "cpu-rayon");
    }
}
