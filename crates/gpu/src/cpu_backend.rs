use rayon::prelude::*;
use simucad_core::error::GpuError;
use simucad_core::types::{Particle, Vec3};
use tracing::debug;

use crate::backend::ComputeBackend;

// ---------------------------------------------------------------------------
// CpuBackend -- multi-threaded reference implementation using rayon
// ---------------------------------------------------------------------------

/// A purely CPU-based compute backend that uses rayon for data-parallel work.
///
/// This serves as:
/// 1. The automatic fallback when no GPU adapter is available.
/// 2. The reference implementation against which GPU results are validated.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        debug!("CpuBackend created (rayon thread pool)");
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu-rayon"
    }

    /// Advect particles in parallel: `p.position += velocity * dt`.
    fn advect_particles(
        &self,
        particles: &mut [Particle],
        velocity: Vec3,
        dt: f64,
    ) -> Result<(), GpuError> {
        let dv = velocity * dt;
        particles.par_iter_mut().for_each(|p| {
            p.position = p.position + dv;
        });
        Ok(())
    }

    /// For each node, compute the mean displacement from node to particles:
    ///   `v_node = (1/N) * sum_i(particle_i.position - node_position)`
    fn compute_velocity_field(
        &self,
        particles: &[Particle],
        node_positions: &[Vec3],
    ) -> Result<Vec<Vec3>, GpuError> {
        if particles.is_empty() {
            return Ok(vec![Vec3::ZERO; node_positions.len()]);
        }

        let n = particles.len() as f64;

        // Pre-compute mean particle position in O(N)
        let (sum_x, sum_y, sum_z) = particles.par_iter().fold(
            || (0.0f64, 0.0f64, 0.0f64),
            |(sx, sy, sz), p| (sx + p.position.x, sy + p.position.y, sz + p.position.z),
        ).reduce(
            || (0.0, 0.0, 0.0),
            |(sx1, sy1, sz1), (sx2, sy2, sz2)| (sx1 + sx2, sy1 + sy2, sz1 + sz2),
        );
        let mean = Vec3::new(sum_x / n, sum_y / n, sum_z / n);

        // For each node: mean_displacement = mean_position - node (O(M))
        let result: Vec<Vec3> = node_positions
            .par_iter()
            .map(|node| mean - *node)
            .collect();

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use simucad_core::types::{Particle, Vec3};

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn vec3_approx_eq(a: &Vec3, b: &Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    #[test]
    fn cpu_backend_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu-rayon");
    }

    #[test]
    fn advect_single_particle() {
        let backend = CpuBackend::new();
        let mut particles = vec![Particle::at_rest(Vec3::new(1.0, 2.0, 3.0))];
        let velocity = Vec3::new(10.0, 20.0, 30.0);
        let dt = 0.5;

        backend
            .advect_particles(&mut particles, velocity, dt)
            .unwrap();

        assert!(vec3_approx_eq(
            &particles[0].position,
            &Vec3::new(6.0, 12.0, 18.0)
        ));
    }

    #[test]
    fn advect_preserves_particle_velocity() {
        let backend = CpuBackend::new();
        let original_vel = Vec3::new(1.0, 1.0, 1.0);
        let mut particles = vec![Particle::new(Vec3::ZERO, original_vel)];

        backend
            .advect_particles(&mut particles, Vec3::new(5.0, 0.0, 0.0), 1.0)
            .unwrap();

        // The particle's own velocity field should be untouched.
        assert!(vec3_approx_eq(&particles[0].velocity, &original_vel));
    }

    #[test]
    fn advect_zero_dt() {
        let backend = CpuBackend::new();
        let mut particles = vec![Particle::at_rest(Vec3::new(1.0, 2.0, 3.0))];

        backend
            .advect_particles(&mut particles, Vec3::new(100.0, 100.0, 100.0), 0.0)
            .unwrap();

        assert!(vec3_approx_eq(
            &particles[0].position,
            &Vec3::new(1.0, 2.0, 3.0)
        ));
    }

    #[test]
    fn advect_many_particles() {
        let backend = CpuBackend::new();
        let mut particles: Vec<Particle> = (0..1000)
            .map(|i| {
                let f = i as f64;
                Particle::at_rest(Vec3::new(f, f * 2.0, f * 3.0))
            })
            .collect();

        let vel = Vec3::new(1.0, 0.0, -1.0);
        let dt = 0.1;
        backend.advect_particles(&mut particles, vel, dt).unwrap();

        for (i, p) in particles.iter().enumerate() {
            let f = i as f64;
            assert!(approx_eq(p.position.x, f + 0.1));
            assert!(approx_eq(p.position.y, f * 2.0));
            assert!(approx_eq(p.position.z, f * 3.0 - 0.1));
        }
    }

    #[test]
    fn velocity_field_single_node_single_particle() {
        let backend = CpuBackend::new();
        let particles = vec![Particle::at_rest(Vec3::new(3.0, 4.0, 5.0))];
        let nodes = vec![Vec3::new(1.0, 1.0, 1.0)];

        let field = backend
            .compute_velocity_field(&particles, &nodes)
            .unwrap();

        assert_eq!(field.len(), 1);
        assert!(vec3_approx_eq(&field[0], &Vec3::new(2.0, 3.0, 4.0)));
    }

    #[test]
    fn velocity_field_mean_of_two_particles() {
        let backend = CpuBackend::new();
        let particles = vec![
            Particle::at_rest(Vec3::new(4.0, 0.0, 0.0)),
            Particle::at_rest(Vec3::new(6.0, 0.0, 0.0)),
        ];
        let nodes = vec![Vec3::new(0.0, 0.0, 0.0)];

        let field = backend
            .compute_velocity_field(&particles, &nodes)
            .unwrap();

        // mean displacement = ((4,0,0) + (6,0,0)) / 2 = (5,0,0)
        assert!(vec3_approx_eq(&field[0], &Vec3::new(5.0, 0.0, 0.0)));
    }

    #[test]
    fn velocity_field_empty_particles() {
        let backend = CpuBackend::new();
        let nodes = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO];

        let field = backend
            .compute_velocity_field(&[], &nodes)
            .unwrap();

        assert_eq!(field.len(), 2);
        assert!(vec3_approx_eq(&field[0], &Vec3::ZERO));
        assert!(vec3_approx_eq(&field[1], &Vec3::ZERO));
    }

    #[test]
    fn velocity_field_multiple_nodes() {
        let backend = CpuBackend::new();
        let particles = vec![
            Particle::at_rest(Vec3::new(2.0, 2.0, 2.0)),
            Particle::at_rest(Vec3::new(4.0, 4.0, 4.0)),
        ];
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 3.0, 3.0),
        ];

        let field = backend
            .compute_velocity_field(&particles, &nodes)
            .unwrap();

        // Node 0: mean of ((2,2,2)-(0,0,0), (4,4,4)-(0,0,0)) = (3,3,3)
        assert!(vec3_approx_eq(&field[0], &Vec3::new(3.0, 3.0, 3.0)));
        // Node 1: mean of ((2,2,2)-(3,3,3), (4,4,4)-(3,3,3)) = (0,0,0)
        assert!(vec3_approx_eq(&field[1], &Vec3::ZERO));
    }
}
