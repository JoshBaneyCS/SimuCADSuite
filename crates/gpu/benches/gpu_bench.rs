use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simucad_core::types::{Particle, Vec3};
use simucad_gpu::backend::ComputeBackend;
use simucad_gpu::cpu_backend::CpuBackend;

fn make_particles(count: usize) -> Vec<Particle> {
    (0..count)
        .map(|i| {
            let f = i as f64;
            Particle::at_rest(Vec3::new(f * 0.001, f * 0.002, f * 0.003))
        })
        .collect()
}

fn bench_advect_particles(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_advect_particles");
    let backend = CpuBackend::new();
    let velocity = Vec3::new(1.0, 0.0, 0.0);
    let dt = 0.01;

    for count in [100_000, 1_000_000] {
        group.sample_size(if count >= 1_000_000 { 10 } else { 30 });
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, &n| {
                let mut particles = make_particles(n);
                b.iter(|| {
                    backend
                        .advect_particles(
                            black_box(&mut particles),
                            black_box(velocity),
                            black_box(dt),
                        )
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_compute_velocity_field(c: &mut Criterion) {
    let backend = CpuBackend::new();
    let particles = make_particles(10_000);
    let node_positions: Vec<Vec3> = (0..100)
        .map(|i| Vec3::new(i as f64, 0.0, 0.0))
        .collect();

    c.bench_function("cpu_compute_velocity_field_10k", |b| {
        b.iter(|| {
            backend
                .compute_velocity_field(black_box(&particles), black_box(&node_positions))
                .unwrap();
        });
    });
}

criterion_group!(benches, bench_advect_particles, bench_compute_velocity_field);
criterion_main!(benches);
