//! Particle-based fluid simulation.
//!
//! Provides a simple Lagrangian particle system for advection and velocity
//! field reconstruction. Particles are confined within a [`BoundingBox3`]
//! and can be advected by a uniform velocity field. A velocity field can be
//! reconstructed at arbitrary node positions by averaging nearby particle
//! displacements.
//!
//! Parallelism is achieved via [`rayon`] for both advection and velocity
//! field computation.

use rayon::prelude::*;
use simucad_core::types::{BoundingBox3, Particle, Vec3};

// ---------------------------------------------------------------------------
// Pseudo-random number generator (xorshift64)
// ---------------------------------------------------------------------------

/// A minimal xorshift64 PRNG to avoid pulling in an external RNG crate.
///
/// This is adequate for distributing particles within a domain. It is NOT
/// cryptographically secure.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Seed the generator. A seed of 0 is replaced with a non-zero default.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x1234_5678_9ABC_DEF0 } else { seed },
        }
    }

    /// Generate the next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a pseudo-random `f64` in the range [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Use the upper 53 bits for a uniform double in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// ParticleSystem
// ---------------------------------------------------------------------------

/// A system of Lagrangian particles within a 3D bounding box.
///
/// Particles are represented as position + velocity pairs and can be
/// advected, reflected at boundaries, and queried for velocity field
/// reconstruction.
pub struct ParticleSystem {
    /// The collection of particles.
    pub particles: Vec<Particle>,
    /// Spatial bounds for the simulation domain.
    pub bounds: BoundingBox3,
}

impl ParticleSystem {
    /// Create a new particle system with `count` particles randomly
    /// distributed within `bounds`. All particles start at rest.
    ///
    /// The distribution is deterministic for a given `count` (seeded by
    /// `count` itself) to make simulations reproducible.
    pub fn initialize(bounds: BoundingBox3, count: usize) -> Self {
        let size = bounds.size();
        let mut rng = Xorshift64::new(count as u64);

        let particles = (0..count)
            .map(|_| {
                let x = bounds.min.x + rng.next_f64() * size.x;
                let y = bounds.min.y + rng.next_f64() * size.y;
                let z = bounds.min.z + rng.next_f64() * size.z;
                Particle::at_rest(Vec3::new(x, y, z))
            })
            .collect();

        Self { particles, bounds }
    }

    /// Advect all particles by a uniform velocity field for a timestep `dt`.
    ///
    /// Each particle's position is updated as:
    /// ```text
    /// position += velocity_field * dt
    /// ```
    ///
    /// Particles that leave the bounding box are reflected back inside
    /// (elastic boundary reflection) so that the domain remains populated.
    ///
    /// Uses rayon for parallel iteration.
    pub fn advect(&mut self, velocity: Vec3, dt: f64) {
        let bounds = self.bounds;
        self.particles.par_iter_mut().for_each(|p| {
            p.velocity = velocity;
            p.position = p.position + velocity * dt;

            // Reflect particles back into the domain
            reflect_into_bounds(&mut p.position, &bounds);
        });
    }

    /// Compute a velocity field at the given `node_positions`.
    ///
    /// For each node, the velocity is computed as the mean of
    /// `(particle.position - node_position)` over all particles. This gives
    /// a simple displacement-based velocity estimate.
    ///
    /// Uses rayon for parallel iteration over nodes.
    pub fn compute_velocity_field(&self, node_positions: &[Vec3]) -> Vec<Vec3> {
        if self.particles.is_empty() {
            return vec![Vec3::ZERO; node_positions.len()];
        }

        let inv_count = 1.0 / self.particles.len() as f64;

        // Pre-compute the sum of all particle positions once (O(N) instead
        // of O(N*M) where N = particles, M = nodes).
        let sum = self
            .particles
            .par_iter()
            .fold(
                || Vec3::ZERO,
                |acc, p| acc + p.position,
            )
            .reduce(|| Vec3::ZERO, |a, b| a + b);

        node_positions
            .par_iter()
            .map(|node| {
                // mean(particle_pos - node) = mean(particle_pos) - node
                let mean_pos = sum * inv_count;
                mean_pos - *node
            })
            .collect()
    }

    /// Return the number of particles currently in the system.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Return the number of particles that are inside the bounding box.
    pub fn particles_in_bounds(&self) -> usize {
        self.particles
            .par_iter()
            .filter(|p| self.bounds.contains(&p.position))
            .count()
    }
}

// ---------------------------------------------------------------------------
// Boundary reflection helper
// ---------------------------------------------------------------------------

/// Reflect a position back into the bounding box along each axis
/// independently (elastic reflection).
fn reflect_into_bounds(pos: &mut Vec3, bounds: &BoundingBox3) {
    reflect_axis(&mut pos.x, bounds.min.x, bounds.max.x);
    reflect_axis(&mut pos.y, bounds.min.y, bounds.max.y);
    reflect_axis(&mut pos.z, bounds.min.z, bounds.max.z);
}

/// Reflect a scalar coordinate into [lo, hi] using a fold-back strategy.
fn reflect_axis(val: &mut f64, lo: f64, hi: f64) {
    let span = hi - lo;
    if span <= 0.0 {
        *val = lo;
        return;
    }

    // Normalize to [0, span]
    let mut v = *val - lo;

    // Number of full spans from origin
    let periods = (v / span).floor() as i64;
    v -= periods as f64 * span;

    // If we ended up on an odd period, reflect
    if periods.rem_euclid(2) == 1 {
        v = span - v;
    }

    *val = lo + v;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box() -> BoundingBox3 {
        BoundingBox3::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0))
    }

    fn big_box() -> BoundingBox3 {
        BoundingBox3::new(Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0))
    }

    // ----- Xorshift64 tests -----

    #[test]
    fn xorshift_produces_different_values() {
        let mut rng = Xorshift64::new(42);
        let a = rng.next_u64();
        let b = rng.next_u64();
        let c = rng.next_u64();
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn xorshift_f64_in_range() {
        let mut rng = Xorshift64::new(123);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "value {v} out of [0, 1)");
        }
    }

    #[test]
    fn xorshift_zero_seed_handled() {
        let mut rng = Xorshift64::new(0);
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }

    // ----- ParticleSystem::initialize tests -----

    #[test]
    fn initialize_correct_count() {
        for count in [0, 1, 100, 5000] {
            let sys = ParticleSystem::initialize(unit_box(), count);
            assert_eq!(sys.particle_count(), count);
        }
    }

    #[test]
    fn initialize_particles_in_bounds() {
        let bounds = big_box();
        let sys = ParticleSystem::initialize(bounds, 10_000);

        for p in &sys.particles {
            assert!(
                bounds.contains(&p.position),
                "particle at {:?} is outside bounds",
                p.position
            );
        }
    }

    #[test]
    fn initialize_particles_at_rest() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        for p in &sys.particles {
            assert_eq!(p.velocity, Vec3::ZERO);
        }
    }

    #[test]
    fn initialize_deterministic() {
        let a = ParticleSystem::initialize(unit_box(), 500);
        let b = ParticleSystem::initialize(unit_box(), 500);
        for (pa, pb) in a.particles.iter().zip(b.particles.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }

    // ----- advect tests -----

    #[test]
    fn advect_updates_positions() {
        let mut sys = ParticleSystem::initialize(big_box(), 100);
        let initial_positions: Vec<Vec3> = sys.particles.iter().map(|p| p.position).collect();

        let vel = Vec3::new(1.0, 0.0, 0.0);
        sys.advect(vel, 0.5);

        for (i, p) in sys.particles.iter().enumerate() {
            let expected_x = initial_positions[i].x + 0.5;
            // May differ slightly if reflected, but within big_box should be fine
            if big_box().contains(&Vec3::new(expected_x, p.position.y, p.position.z)) {
                assert!(
                    (p.position.x - expected_x).abs() < 1e-10,
                    "particle {i} x: {} vs expected: {}",
                    p.position.x,
                    expected_x
                );
            }
        }
    }

    #[test]
    fn advect_sets_velocity() {
        let mut sys = ParticleSystem::initialize(unit_box(), 50);
        let vel = Vec3::new(2.0, -1.0, 0.5);
        sys.advect(vel, 0.1);

        for p in &sys.particles {
            assert_eq!(p.velocity, vel);
        }
    }

    #[test]
    fn advect_particles_remain_in_bounds() {
        let bounds = unit_box();
        let mut sys = ParticleSystem::initialize(bounds, 1000);

        // Advect with a large velocity to force boundary reflections
        let vel = Vec3::new(100.0, -50.0, 75.0);
        for _ in 0..10 {
            sys.advect(vel, 0.1);
        }

        for p in &sys.particles {
            assert!(
                bounds.contains(&p.position),
                "particle at {:?} escaped bounds after advection",
                p.position
            );
        }
    }

    #[test]
    fn advect_zero_velocity_no_change() {
        let mut sys = ParticleSystem::initialize(unit_box(), 100);
        let positions_before: Vec<Vec3> = sys.particles.iter().map(|p| p.position).collect();
        sys.advect(Vec3::ZERO, 1.0);

        for (i, p) in sys.particles.iter().enumerate() {
            assert_eq!(p.position, positions_before[i]);
        }
    }

    // ----- compute_velocity_field tests -----

    #[test]
    fn velocity_field_empty_particles() {
        let sys = ParticleSystem::initialize(unit_box(), 0);
        let nodes = vec![Vec3::new(0.5, 0.5, 0.5)];
        let field = sys.compute_velocity_field(&nodes);
        assert_eq!(field.len(), 1);
        assert_eq!(field[0], Vec3::ZERO);
    }

    #[test]
    fn velocity_field_empty_nodes() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        let field = sys.compute_velocity_field(&[]);
        assert!(field.is_empty());
    }

    #[test]
    fn velocity_field_correct_count() {
        let sys = ParticleSystem::initialize(unit_box(), 100);
        let nodes: Vec<Vec3> = (0..25).map(|i| Vec3::new(i as f64 * 0.04, 0.5, 0.5)).collect();
        let field = sys.compute_velocity_field(&nodes);
        assert_eq!(field.len(), 25);
    }

    #[test]
    fn velocity_field_single_particle() {
        // With a single particle at a known position, the velocity at a node
        // should be (particle_pos - node_pos).
        let bounds = BoundingBox3::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        let mut sys = ParticleSystem {
            particles: vec![Particle::at_rest(Vec3::new(5.0, 5.0, 5.0))],
            bounds,
        };

        let nodes = vec![Vec3::new(1.0, 2.0, 3.0)];
        let field = sys.compute_velocity_field(&nodes);

        assert!((field[0].x - 4.0).abs() < 1e-10);
        assert!((field[0].y - 3.0).abs() < 1e-10);
        assert!((field[0].z - 2.0).abs() < 1e-10);
    }

    #[test]
    fn velocity_field_symmetric_particles() {
        // Two particles symmetric around the origin: their mean position is
        // the origin, so the velocity at the origin should be ZERO.
        let bounds = BoundingBox3::new(Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0));
        let sys = ParticleSystem {
            particles: vec![
                Particle::at_rest(Vec3::new(1.0, 0.0, 0.0)),
                Particle::at_rest(Vec3::new(-1.0, 0.0, 0.0)),
            ],
            bounds,
        };

        let nodes = vec![Vec3::ZERO];
        let field = sys.compute_velocity_field(&nodes);

        assert!((field[0].x).abs() < 1e-10);
        assert!((field[0].y).abs() < 1e-10);
        assert!((field[0].z).abs() < 1e-10);
    }

    // ----- reflect_axis tests -----

    #[test]
    fn reflect_axis_inside_unchanged() {
        let mut v = 0.5;
        reflect_axis(&mut v, 0.0, 1.0);
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reflect_axis_at_boundary() {
        let mut v = 1.0;
        reflect_axis(&mut v, 0.0, 1.0);
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reflect_axis_beyond_upper() {
        let mut v = 1.3;
        reflect_axis(&mut v, 0.0, 1.0);
        // 1.3 -> reflects to 0.7
        assert!((v - 0.7).abs() < 1e-10, "reflected value: {v}");
    }

    #[test]
    fn reflect_axis_beyond_lower() {
        let mut v = -0.3;
        reflect_axis(&mut v, 0.0, 1.0);
        // -0.3 -> reflects to 0.3
        assert!((v - 0.3).abs() < 1e-10, "reflected value: {v}");
    }

    #[test]
    fn reflect_axis_multiple_reflections() {
        let mut v = 2.7; // 2.7 spans from 0: period 2 (even), remainder 0.7
        reflect_axis(&mut v, 0.0, 1.0);
        assert!((v - 0.7).abs() < 1e-10, "reflected value: {v}");
    }

    #[test]
    fn reflect_axis_negative_side_multiple() {
        let mut v = -1.4; // -1.4 from 0: normalize => -1.4, floor(-1.4) = -2, periods = -2
        reflect_axis(&mut v, 0.0, 1.0);
        // After normalization: v = -1.4 - (-2)*1 = 0.6, periods=-2 (even), so v=0.6
        assert!((v - 0.6).abs() < 1e-10, "reflected value: {v}");
    }

    // ----- particles_in_bounds test -----

    #[test]
    fn particles_in_bounds_after_init() {
        let sys = ParticleSystem::initialize(unit_box(), 500);
        assert_eq!(sys.particles_in_bounds(), 500);
    }

    // ----- Integration test: advect then compute field -----

    #[test]
    fn advect_then_compute_field() {
        let bounds = BoundingBox3::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        let mut sys = ParticleSystem::initialize(bounds, 5000);

        // Advect a little
        sys.advect(Vec3::new(0.1, 0.2, 0.05), 1.0);

        // Compute velocity field at center
        let nodes = vec![Vec3::new(5.0, 5.0, 5.0)];
        let field = sys.compute_velocity_field(&nodes);

        // Should produce a finite, non-zero vector
        assert!(field[0].magnitude().is_finite());
    }
}
