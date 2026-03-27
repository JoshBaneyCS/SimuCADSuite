//! End-to-end integration tests for the fluid simulation subsystem.
//!
//! These tests exercise ParticleSystem creation, advection, boundary
//! reflection, SpatialHashGrid queries, FluidSimulation step/run/progress,
//! and velocity field computation.

use simucad_core::types::{BoundingBox3, Vec3};
use simucad_physics::fluid::{FluidSimulation, ParticleSystem, SpatialHashGrid};

fn unit_box() -> BoundingBox3 {
    BoundingBox3::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0))
}

fn big_box() -> BoundingBox3 {
    BoundingBox3::new(
        Vec3::new(-10.0, -10.0, -10.0),
        Vec3::new(10.0, 10.0, 10.0),
    )
}

// ---------------------------------------------------------------------------
// 1. Create 10k particles, run 5 steps, verify all particles in bounds
// ---------------------------------------------------------------------------

#[test]
fn particle_system_10k_particles_stay_in_bounds() {
    let bounds = big_box();
    let mut system = ParticleSystem::initialize(bounds, 10_000);

    assert_eq!(system.particle_count(), 10_000);

    // All particles should start in bounds
    assert_eq!(
        system.particles_in_bounds(),
        10_000,
        "all particles should start in bounds",
    );

    // Advect 5 steps with a velocity that would push particles out of bounds
    // if reflection were not working
    let velocity = Vec3::new(5.0, -3.0, 7.0);
    let dt = 1.0;
    for _ in 0..5 {
        system.advect(velocity, dt);
    }

    // After reflection, all particles should still be in bounds
    let in_bounds = system.particles_in_bounds();
    assert_eq!(
        in_bounds, 10_000,
        "all particles should remain in bounds after advection, got {} in bounds",
        in_bounds,
    );

    // Verify particle count is unchanged
    assert_eq!(system.particle_count(), 10_000);
}

// ---------------------------------------------------------------------------
// 2. Build spatial hash grid, verify query_radius returns correct particles
// ---------------------------------------------------------------------------

#[test]
fn spatial_hash_grid_query_radius() {
    let bounds = BoundingBox3::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
    let system = ParticleSystem::initialize(bounds, 5_000);

    let cell_size = 1.0;
    let mut grid = SpatialHashGrid::new(cell_size);
    grid.build(&system.particles);

    // Query a sphere centered at (5, 5, 5) with radius 2.0
    let center = Vec3::new(5.0, 5.0, 5.0);
    let radius = 2.0;
    let nearby = grid.query_radius(center, radius);

    // Should find some particles (not empty for 5000 particles in a 10^3 box)
    assert!(
        !nearby.is_empty(),
        "should find particles near center of domain",
    );

    // All returned indices should be valid
    for &idx in &nearby {
        assert!(
            idx < system.particle_count(),
            "index {} out of bounds (count={})",
            idx,
            system.particle_count(),
        );
    }

    // The spatial hash grid returns ALL particles in cells whose integer
    // coordinates overlap the query sphere. A cell at the far corner can
    // extend up to `cell_size` beyond the sphere boundary along each axis.
    // The worst case for a 3D cell corner is radius + cell_size*sqrt(3)
    // per the cell diagonal, but we also need to account for the fact that
    // the floor-based cell selection can add up to one extra cell_size on
    // each side. Use a generous bound:
    let max_overshoot = radius + cell_size * 2.0 * 3.0_f64.sqrt();
    for &idx in &nearby {
        let pos = system.particles[idx].position;
        let dx = pos.x - center.x;
        let dy = pos.y - center.y;
        let dz = pos.z - center.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            dist < max_overshoot,
            "particle at ({}, {}, {}) is distance {} from center, exceeds {}",
            pos.x,
            pos.y,
            pos.z,
            dist,
            max_overshoot,
        );
    }

    // Query at the corner with a tiny radius -- should find very few or zero
    let corner = Vec3::new(0.0, 0.0, 0.0);
    let tiny_result = grid.query_radius(corner, 0.01);
    // With 5000 particles in 1000 cubic units, density is ~5 per unit,
    // so a sphere of radius 0.01 encompasses ~0.000004 cubic units.
    // It is unlikely but possible to find a particle; either way the result
    // should be small.
    assert!(
        tiny_result.len() < 50,
        "tiny query should return very few particles, got {}",
        tiny_result.len(),
    );
}

// ---------------------------------------------------------------------------
// 3. Create FluidSimulation, run all steps, verify progress reaches 1.0
// ---------------------------------------------------------------------------

#[test]
fn fluid_simulation_run_to_completion() {
    let bounds = unit_box();
    let system = ParticleSystem::initialize(bounds, 1_000);

    let step_count = 10;
    let mut sim = FluidSimulation::new(system, 0.1, step_count);

    // Initial progress should be 0
    assert!(
        (sim.progress() - 0.0).abs() < f32::EPSILON,
        "initial progress should be 0, got {}",
        sim.progress(),
    );

    // Step halfway
    let velocity = Vec3::new(0.1, 0.0, 0.0);
    let dt = 0.01;
    for _ in 0..5 {
        sim.step(velocity, dt);
    }
    assert!(
        (sim.progress() - 0.5).abs() < f32::EPSILON,
        "progress should be 0.5 after 5/10 steps, got {}",
        sim.progress(),
    );

    // Run remaining steps
    sim.run(velocity, dt);

    // Progress should be 1.0
    assert!(
        (sim.progress() - 1.0).abs() < f32::EPSILON,
        "final progress should be 1.0, got {}",
        sim.progress(),
    );

    // Step count should match
    assert_eq!(sim.current_step, step_count);

    // All particles should still be in bounds
    let in_bounds = sim.particle_system.particles_in_bounds();
    assert_eq!(
        in_bounds,
        sim.particle_system.particle_count(),
        "all particles should be in bounds after simulation",
    );
}

// ---------------------------------------------------------------------------
// 4. Velocity field with known uniform velocity -- all field values similar
// ---------------------------------------------------------------------------

#[test]
fn velocity_field_with_uniform_particle_distribution() {
    // Create a particle system in a unit box
    let bounds = unit_box();
    let system = ParticleSystem::initialize(bounds, 5_000);

    // Query nodes at the center of the domain
    let center = Vec3::new(0.5, 0.5, 0.5);
    let nodes = vec![center];

    let field = system.compute_velocity_field(&nodes);
    assert_eq!(field.len(), 1);

    // For a uniform distribution in [0,1]^3, the mean position is ~(0.5, 0.5, 0.5).
    // The field at center is mean(particle_pos) - center.
    // Since mean(particle_pos) ~ (0.5, 0.5, 0.5), the field should be near zero.
    let v = field[0];
    let magnitude = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    assert!(
        magnitude < 0.1,
        "velocity field at center should be near zero for uniform distribution, got ({}, {}, {}), magnitude {}",
        v.x, v.y, v.z, magnitude,
    );

    // Query two symmetric nodes -- their field values should be roughly opposite
    let node_a = Vec3::new(0.25, 0.5, 0.5);
    let node_b = Vec3::new(0.75, 0.5, 0.5);
    let field_ab = system.compute_velocity_field(&[node_a, node_b]);

    // field[0].x should be positive (particles are generally to the right of 0.25)
    // field[1].x should be negative (particles are generally to the left of 0.75)
    assert!(
        field_ab[0].x > 0.0,
        "field at x=0.25 should have positive x component, got {}",
        field_ab[0].x,
    );
    assert!(
        field_ab[1].x < 0.0,
        "field at x=0.75 should have negative x component, got {}",
        field_ab[1].x,
    );

    // The magnitudes should be similar (by symmetry)
    let diff = (field_ab[0].x.abs() - field_ab[1].x.abs()).abs();
    assert!(
        diff < 0.1,
        "symmetric nodes should have similar magnitude field, diff = {}",
        diff,
    );
}

#[test]
fn fluid_simulation_velocity_field_consistent() {
    let bounds = big_box();
    let system = ParticleSystem::initialize(bounds, 2_000);

    let mut sim = FluidSimulation::new(system, 1.0, 5);

    // Advect with a uniform velocity
    let velocity = Vec3::new(1.0, 0.0, 0.0);
    let dt = 0.1;
    sim.run(velocity, dt);

    // Query velocity field at origin
    let nodes = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
    let field = sim.velocity_field(&nodes);

    assert_eq!(field.len(), 2);

    // Both field values should be finite
    for v in &field {
        assert!(v.x.is_finite());
        assert!(v.y.is_finite());
        assert!(v.z.is_finite());
    }
}
