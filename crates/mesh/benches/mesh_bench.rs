use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simucad_core::types::Vec3;
use simucad_mesh::quality::compute_mesh_quality;
use simucad_mesh::spatial::BVH;
use simucad_mesh::types::{ElementType, Mesh, MeshElement};

/// Build a triangulated mesh with approximately `n` triangles arranged in a
/// grid on the XY plane. Each grid cell is split into 2 triangles.
fn make_triangle_mesh(num_elements: usize) -> Mesh {
    // Determine grid size: grid_side^2 * 2 >= num_elements
    let grid_side = ((num_elements as f64 / 2.0).sqrt().ceil()) as usize;
    let step = 1.0;

    // Generate nodes: (grid_side + 1) x (grid_side + 1) grid.
    let node_count = (grid_side + 1) * (grid_side + 1);
    let mut nodes = Vec::with_capacity(node_count);
    for j in 0..=grid_side {
        for i in 0..=grid_side {
            nodes.push(Vec3::new(i as f64 * step, j as f64 * step, 0.0));
        }
    }

    // Generate triangles (2 per grid cell).
    let cols = grid_side + 1;
    let mut elements = Vec::with_capacity(grid_side * grid_side * 2);
    for j in 0..grid_side {
        for i in 0..grid_side {
            let bl = j * cols + i;
            let br = bl + 1;
            let tl = bl + cols;
            let tr = tl + 1;
            elements.push(MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![bl, br, tl],
            });
            elements.push(MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![br, tr, tl],
            });
            if elements.len() >= num_elements {
                break;
            }
        }
        if elements.len() >= num_elements {
            break;
        }
    }

    Mesh {
        nodes,
        elements,
        dimension: 2,
    }
}

fn bench_bvh_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_build");

    for count in [100, 1_000, 10_000] {
        let mesh = make_triangle_mesh(count);
        group.sample_size(if count >= 10_000 { 10 } else { 30 });
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, _| {
                b.iter(|| {
                    BVH::build(black_box(&mesh));
                });
            },
        );
    }
    group.finish();
}

fn bench_bvh_query_point(c: &mut Criterion) {
    let mesh = make_triangle_mesh(10_000);
    let bvh = BVH::build(&mesh);
    let point = Vec3::new(5.0, 5.0, 0.0);

    c.bench_function("bvh_query_point_10k", |b| {
        b.iter(|| {
            bvh.query_point(black_box(&point));
        });
    });
}

fn bench_mesh_quality(c: &mut Criterion) {
    let mesh = make_triangle_mesh(10_000);

    c.bench_function("mesh_quality_10k", |b| {
        b.iter(|| {
            compute_mesh_quality(black_box(&mesh));
        });
    });
}

criterion_group!(benches, bench_bvh_build, bench_bvh_query_point, bench_mesh_quality);
criterion_main!(benches);
