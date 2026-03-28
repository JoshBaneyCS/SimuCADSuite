//! CPU backend integration tests.
//!
//! These tests exercise the CpuBackend implementation of the ComputeBackend
//! trait: particle advection, velocity field computation, and verification
//! against manual calculation.

use simucad_core::types::{Particle, Vec3};
use simucad_gpu::backend::{select_backend, ComputeBackend};
use simucad_gpu::cpu_backend::CpuBackend;

const TOL: f64 = 1e-10;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < TOL
}

fn vec3_approx_eq(a: &Vec3, b: &Vec3) -> bool {
    approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
}

// ---------------------------------------------------------------------------
// 1. Create 1000 particles, advect, verify positions updated correctly
// ---------------------------------------------------------------------------

#[test]
fn advect_1000_particles() {
    let backend = CpuBackend::new();

    // Create 1000 particles at known positions
    let mut particles: Vec<Particle> = (0..1000)
        .map(|i| {
            let f = i as f64;
            Particle::at_rest(Vec3::new(f * 0.1, f * 0.2, f * 0.3))
        })
        .collect();

    let velocity = Vec3::new(1.0, -0.5, 2.0);
    let dt = 0.25;

    backend
        .advect_particles(&mut particles, velocity, dt)
        .expect("advection should succeed");

    // Verify each particle's position
    for (i, p) in particles.iter().enumerate() {
        let f = i as f64;
        let expected_x = f * 0.1 + velocity.x * dt;
        let expected_y = f * 0.2 + velocity.y * dt;
        let expected_z = f * 0.3 + velocity.z * dt;

        assert!(
            approx_eq(p.position.x, expected_x),
            "particle {} x: {} vs expected {}",
            i,
            p.position.x,
            expected_x,
        );
        assert!(
            approx_eq(p.position.y, expected_y),
            "particle {} y: {} vs expected {}",
            i,
            p.position.y,
            expected_y,
        );
        assert!(
            approx_eq(p.position.z, expected_z),
            "particle {} z: {} vs expected {}",
            i,
            p.position.z,
            expected_z,
        );
    }
}

#[test]
fn advect_multiple_steps_accumulate() {
    let backend = CpuBackend::new();

    let mut particles = vec![Particle::at_rest(Vec3::new(0.0, 0.0, 0.0))];
    let velocity = Vec3::new(1.0, 2.0, 3.0);
    let dt = 0.1;

    // Run 10 steps
    for _ in 0..10 {
        backend
            .advect_particles(&mut particles, velocity, dt)
            .unwrap();
    }

    // After 10 steps of dt=0.1 with velocity (1,2,3):
    // position = (1, 2, 3) * 10 * 0.1 = (1, 2, 3)
    let expected = Vec3::new(1.0, 2.0, 3.0);
    assert!(
        approx_eq(particles[0].position.x, expected.x),
        "x: {} vs {}",
        particles[0].position.x,
        expected.x,
    );
    assert!(
        approx_eq(particles[0].position.y, expected.y),
        "y: {} vs {}",
        particles[0].position.y,
        expected.y,
    );
    assert!(
        approx_eq(particles[0].position.z, expected.z),
        "z: {} vs {}",
        particles[0].position.z,
        expected.z,
    );
}

#[test]
fn advect_zero_velocity_no_change() {
    let backend = CpuBackend::new();

    let original_pos = Vec3::new(3.0, 7.0, -2.0);
    let mut particles = vec![Particle::at_rest(original_pos)];

    backend
        .advect_particles(&mut particles, Vec3::ZERO, 1.0)
        .unwrap();

    assert!(
        vec3_approx_eq(&particles[0].position, &original_pos),
        "position should not change with zero velocity",
    );
}

// ---------------------------------------------------------------------------
// 2. Compute velocity field for 10 nodes, verify reasonable values
// ---------------------------------------------------------------------------

#[test]
fn velocity_field_10_nodes() {
    let backend = CpuBackend::new();

    // Create particles at known positions
    let particles: Vec<Particle> = vec![
        Particle::at_rest(Vec3::new(1.0, 0.0, 0.0)),
        Particle::at_rest(Vec3::new(3.0, 0.0, 0.0)),
        Particle::at_rest(Vec3::new(2.0, 2.0, 0.0)),
        Particle::at_rest(Vec3::new(2.0, -2.0, 0.0)),
    ];

    // Mean particle position = (8/4, 0/4, 0/4) = (2, 0, 0)
    let mean_pos = Vec3::new(2.0, 0.0, 0.0);

    // Create 10 node positions along the x-axis
    let nodes: Vec<Vec3> = (0..10)
        .map(|i| Vec3::new(i as f64 * 0.5, 0.0, 0.0))
        .collect();

    let field = backend
        .compute_velocity_field(&particles, &nodes)
        .expect("velocity field computation should succeed");

    assert_eq!(field.len(), 10);

    // For each node, verify the field value
    // v_node = mean(particle_pos - node) = mean_pos - node
    for (i, v) in field.iter().enumerate() {
        let node = &nodes[i];
        let expected = Vec3::new(
            mean_pos.x - node.x,
            mean_pos.y - node.y,
            mean_pos.z - node.z,
        );
        assert!(
            vec3_approx_eq(v, &expected),
            "node {}: got ({}, {}, {}), expected ({}, {}, {})",
            i,
            v.x,
            v.y,
            v.z,
            expected.x,
            expected.y,
            expected.z,
        );
    }
}

#[test]
fn velocity_field_empty_particles_returns_zero() {
    let backend = CpuBackend::new();
    let nodes = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO, Vec3::new(-1.0, -2.0, -3.0)];

    let field = backend
        .compute_velocity_field(&[], &nodes)
        .expect("should handle empty particles");

    assert_eq!(field.len(), 3);
    for v in &field {
        assert!(
            vec3_approx_eq(v, &Vec3::ZERO),
            "field should be zero for empty particles, got ({}, {}, {})",
            v.x,
            v.y,
            v.z,
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Compare CPU backend results against manual computation
// ---------------------------------------------------------------------------

#[test]
fn advect_then_velocity_field_consistency() {
    let backend = CpuBackend::new();

    // Start with particles at known positions
    let mut particles: Vec<Particle> = (0..100)
        .map(|i| {
            let f = i as f64;
            Particle::at_rest(Vec3::new(f, 0.0, 0.0))
        })
        .collect();

    // Advect by (10, 5, 0) for dt=1
    let velocity = Vec3::new(10.0, 5.0, 0.0);
    let dt = 1.0;
    backend
        .advect_particles(&mut particles, velocity, dt)
        .unwrap();

    // Verify positions manually
    for (i, p) in particles.iter().enumerate() {
        let f = i as f64;
        assert!(
            approx_eq(p.position.x, f + 10.0),
            "particle {} x after advection: {} vs {}",
            i,
            p.position.x,
            f + 10.0,
        );
        assert!(
            approx_eq(p.position.y, 5.0),
            "particle {} y after advection: {} vs 5.0",
            i,
            p.position.y,
        );
    }

    // Compute velocity field at origin
    let nodes = vec![Vec3::ZERO];
    let field = backend
        .compute_velocity_field(&particles, &nodes)
        .unwrap();

    // Mean particle position after advection:
    // x: mean(0..99) + 10 = 49.5 + 10 = 59.5
    // y: 5.0
    // z: 0.0
    // Field at origin = mean_pos - (0,0,0) = (59.5, 5.0, 0.0)
    let expected = Vec3::new(59.5, 5.0, 0.0);
    assert!(
        approx_eq(field[0].x, expected.x),
        "field x: {} vs {}",
        field[0].x,
        expected.x,
    );
    assert!(
        approx_eq(field[0].y, expected.y),
        "field y: {} vs {}",
        field[0].y,
        expected.y,
    );
    assert!(
        approx_eq(field[0].z, expected.z),
        "field z: {} vs {}",
        field[0].z,
        expected.z,
    );
}

#[test]
fn backend_name_is_cpu_rayon() {
    let backend = CpuBackend::new();
    assert_eq!(backend.name(), "cpu-rayon");
}

#[test]
fn select_backend_advect_matches_cpu() {
    // Verify that select_backend() (GPU or CPU) produces the same results as
    // the explicit CPU backend for a non-trivial advection.
    let auto_backend = select_backend();
    let cpu_backend = CpuBackend::new();

    let make_particles = || -> Vec<Particle> {
        (0..500)
            .map(|i| {
                let f = i as f64;
                Particle::at_rest(Vec3::new(f * 0.1, f * 0.05, f * 0.02))
            })
            .collect()
    };

    let mut auto_particles = make_particles();
    let mut cpu_particles = make_particles();

    let velocity = Vec3::new(3.0, -1.5, 0.7);
    let dt = 0.1;

    // Run 5 steps on both backends
    for _ in 0..5 {
        auto_backend
            .advect_particles(&mut auto_particles, velocity, dt)
            .expect("auto backend advection should succeed");
        cpu_backend
            .advect_particles(&mut cpu_particles, velocity, dt)
            .expect("cpu backend advection should succeed");
    }

    // GPU uses f32 internally so tolerance must be looser than f64 epsilon
    let gpu_tol = 1e-4;
    for (i, (a, c)) in auto_particles.iter().zip(cpu_particles.iter()).enumerate() {
        assert!(
            (a.position.x - c.position.x).abs() < gpu_tol
                && (a.position.y - c.position.y).abs() < gpu_tol
                && (a.position.z - c.position.z).abs() < gpu_tol,
            "particle {} mismatch after 5 steps: auto={:?} cpu={:?} (backend={})",
            i,
            a.position,
            c.position,
            auto_backend.name(),
        );
    }
}

#[test]
fn velocity_field_symmetric_particles() {
    let backend = CpuBackend::new();

    // Two particles symmetric about the origin
    let particles = vec![
        Particle::at_rest(Vec3::new(5.0, 0.0, 0.0)),
        Particle::at_rest(Vec3::new(-5.0, 0.0, 0.0)),
    ];

    // At origin, mean displacement is (0, 0, 0)
    let field = backend
        .compute_velocity_field(&particles, &[Vec3::ZERO])
        .unwrap();
    assert!(
        vec3_approx_eq(&field[0], &Vec3::ZERO),
        "symmetric particles should produce zero field at origin, got ({}, {}, {})",
        field[0].x,
        field[0].y,
        field[0].z,
    );

    // At (5, 0, 0), mean displacement is mean((0,0,0), (-10,0,0)) = (-5, 0, 0)
    let field2 = backend
        .compute_velocity_field(&particles, &[Vec3::new(5.0, 0.0, 0.0)])
        .unwrap();
    assert!(
        vec3_approx_eq(&field2[0], &Vec3::new(-5.0, 0.0, 0.0)),
        "field at (5,0,0) should be (-5,0,0), got ({}, {}, {})",
        field2[0].x,
        field2[0].y,
        field2[0].z,
    );
}
