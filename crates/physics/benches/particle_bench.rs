use criterion::{black_box, criterion_group, criterion_main, Criterion};
use simucad_core::types::{BoundingBox3, Vec3};
use simucad_physics::fluid::ParticleSystem;

fn bench_particle_advection(c: &mut Criterion) {
    let bounds = BoundingBox3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 10.0));
    let mut system = ParticleSystem::initialize(bounds, 100_000, 42);
    let velocity = Vec3::new(1.0, 0.0, 0.0);

    c.bench_function("advect_100k_particles", |b| {
        b.iter(|| {
            system.advect(black_box(velocity), black_box(0.01));
        })
    });
}

criterion_group!(benches, bench_particle_advection);
criterion_main!(benches);
