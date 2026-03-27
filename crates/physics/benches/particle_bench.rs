use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simucad_core::types::{BoundingBox3, Particle, Vec3};
use simucad_physics::fluid::{ParticleSystem, SpatialHashGrid};

fn make_system(count: usize) -> ParticleSystem {
    let bounds = BoundingBox3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 10.0));
    ParticleSystem::initialize(bounds, count)
}

fn bench_advection_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("advect_particles");
    let velocity = Vec3::new(1.0, 0.0, 0.0);

    for count in [10_000, 100_000, 1_000_000] {
        let mut system = make_system(count);
        group.sample_size(if count >= 1_000_000 { 10 } else { 30 });
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, _| {
                b.iter(|| {
                    system.advect(black_box(velocity), black_box(0.01));
                });
            },
        );
    }
    group.finish();
}

fn bench_spatial_hash_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_hash_build");

    for count in [10_000, 100_000, 1_000_000] {
        let system = make_system(count);
        let particles = &system.particles;
        group.sample_size(if count >= 1_000_000 { 10 } else { 30 });
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, _| {
                b.iter(|| {
                    let mut grid = SpatialHashGrid::new(1.0);
                    grid.build(black_box(particles));
                });
            },
        );
    }
    group.finish();
}

fn bench_spatial_hash_query(c: &mut Criterion) {
    let system = make_system(100_000);
    let mut grid = SpatialHashGrid::new(1.0);
    grid.build(&system.particles);

    let center = Vec3::new(5.0, 5.0, 5.0);

    c.bench_function("spatial_hash_query_radius_100k", |b| {
        b.iter(|| {
            grid.query_radius(black_box(center), black_box(2.0));
        });
    });
}

criterion_group!(
    benches,
    bench_advection_scaling,
    bench_spatial_hash_build,
    bench_spatial_hash_query,
);
criterion_main!(benches);
