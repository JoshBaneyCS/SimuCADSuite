//! Structured mesh generation utilities.
//!
//! Provides functions for generating simple meshes on canonical geometries
//! (rectangles, boxes, disks) that are useful for testing, prototyping, and
//! as starting points for adaptive refinement.

use std::f64::consts::TAU;

use simucad_core::types::Vec3;

use crate::types::{ElementType, Mesh, MeshElement};

// ---------------------------------------------------------------------------
// Rectangle mesh (2D)
// ---------------------------------------------------------------------------

/// Generate a structured triangular mesh over a rectangular domain.
///
/// The rectangle spans `[x_min, x_max] x [y_min, y_max]` and is divided into
/// `nx` columns and `ny` rows.  Each rectangular cell is split into two
/// [`Triangle3`](ElementType::Triangle3) elements.
///
/// Returns an empty mesh when `nx == 0` or `ny == 0`.
pub fn generate_rectangle_mesh(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
) -> Mesh {
    if nx == 0 || ny == 0 {
        return Mesh {
            nodes: Vec::new(),
            elements: Vec::new(),
            dimension: 2,
        };
    }

    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;

    // Nodes: (nx+1) * (ny+1), indexed as j * (nx+1) + i
    let num_nodes = (nx + 1) * (ny + 1);
    let mut nodes = Vec::with_capacity(num_nodes);

    for j in 0..=ny {
        for i in 0..=nx {
            nodes.push(Vec3::new(
                x_min + i as f64 * dx,
                y_min + j as f64 * dy,
                0.0,
            ));
        }
    }

    // Elements: 2 triangles per cell, nx * ny cells
    let num_elements = 2 * nx * ny;
    let mut elements = Vec::with_capacity(num_elements);

    let node_idx = |i: usize, j: usize| -> usize { j * (nx + 1) + i };

    for j in 0..ny {
        for i in 0..nx {
            let n00 = node_idx(i, j);
            let n10 = node_idx(i + 1, j);
            let n11 = node_idx(i + 1, j + 1);
            let n01 = node_idx(i, j + 1);

            // Lower-right triangle
            elements.push(MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![n00, n10, n11],
            });
            // Upper-left triangle
            elements.push(MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![n00, n11, n01],
            });
        }
    }

    Mesh {
        nodes,
        elements,
        dimension: 2,
    }
}

// ---------------------------------------------------------------------------
// Box mesh (3D)
// ---------------------------------------------------------------------------

/// Generate a structured tetrahedral mesh over a box domain.
///
/// The box spans `[x_min, x_max] x [y_min, y_max] x [z_min, z_max]` and is
/// divided into `nx x ny x nz` hexahedral cells.  Each hex cell is decomposed
/// into 6 [`Tetrahedron4`](ElementType::Tetrahedron4) elements using the
/// standard 6-tet decomposition that shares the main diagonal of the cube.
///
/// Returns an empty mesh when any of `nx`, `ny`, or `nz` is zero.
pub fn generate_box_mesh(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Mesh {
    if nx == 0 || ny == 0 || nz == 0 {
        return Mesh {
            nodes: Vec::new(),
            elements: Vec::new(),
            dimension: 3,
        };
    }

    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let dz = (z_max - z_min) / nz as f64;

    // Nodes: (nx+1) * (ny+1) * (nz+1)
    let num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    let mut nodes = Vec::with_capacity(num_nodes);

    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push(Vec3::new(
                    x_min + i as f64 * dx,
                    y_min + j as f64 * dy,
                    z_min + k as f64 * dz,
                ));
            }
        }
    }

    // Node index helper: k * (ny+1)*(nx+1) + j * (nx+1) + i
    let node_idx =
        |i: usize, j: usize, k: usize| -> usize { k * (ny + 1) * (nx + 1) + j * (nx + 1) + i };

    // Elements: 6 tets per hex cell
    let num_elements = 6 * nx * ny * nz;
    let mut elements = Vec::with_capacity(num_elements);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                // Eight corners of the hex cell, labelled by binary (i-offset, j-offset, k-offset)
                let v000 = node_idx(i, j, k);
                let v100 = node_idx(i + 1, j, k);
                let v010 = node_idx(i, j + 1, k);
                let v110 = node_idx(i + 1, j + 1, k);
                let v001 = node_idx(i, j, k + 1);
                let v101 = node_idx(i + 1, j, k + 1);
                let v011 = node_idx(i, j + 1, k + 1);
                let v111 = node_idx(i + 1, j + 1, k + 1);

                // Standard 6-tet decomposition sharing the main diagonal v000-v111.
                let tets: [[usize; 4]; 6] = [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ];

                for tet in &tets {
                    elements.push(MeshElement {
                        element_type: ElementType::Tetrahedron4,
                        node_indices: tet.to_vec(),
                    });
                }
            }
        }
    }

    Mesh {
        nodes,
        elements,
        dimension: 3,
    }
}

// ---------------------------------------------------------------------------
// Disk mesh (2D)
// ---------------------------------------------------------------------------

/// Generate a structured triangular mesh over a disk.
///
/// The disk is centred at `(cx, cy, 0)` with the given `radius`.  It is
/// divided into `num_rings` concentric rings and `num_sectors` angular
/// sectors.
///
/// * Node 0 is the centre point.
/// * Ring `r` (1..=`num_rings`) contains `num_sectors` nodes evenly
///   distributed at radius `r * radius / num_rings`.
/// * The innermost ring uses a triangle fan from the centre.
/// * Outer rings use quad strips, each quad split into two triangles.
///
/// Returns an empty mesh when `num_rings == 0` or `num_sectors < 3`.
pub fn generate_disk_mesh(
    cx: f64,
    cy: f64,
    radius: f64,
    num_rings: usize,
    num_sectors: usize,
) -> Mesh {
    if num_rings == 0 || num_sectors < 3 {
        return Mesh {
            nodes: Vec::new(),
            elements: Vec::new(),
            dimension: 2,
        };
    }

    // --- Nodes ---
    // Centre + num_rings * num_sectors
    let num_nodes = 1 + num_rings * num_sectors;
    let mut nodes = Vec::with_capacity(num_nodes);

    // Node 0: centre
    nodes.push(Vec3::new(cx, cy, 0.0));

    for r in 1..=num_rings {
        let ring_radius = r as f64 * radius / num_rings as f64;
        for s in 0..num_sectors {
            let angle = TAU * s as f64 / num_sectors as f64;
            nodes.push(Vec3::new(
                cx + ring_radius * angle.cos(),
                cy + ring_radius * angle.sin(),
                0.0,
            ));
        }
    }

    // --- Elements ---
    // Innermost ring: num_sectors triangles (fan from centre)
    // Each outer ring: 2 * num_sectors triangles (quad strip)
    let num_elements = num_sectors + 2 * num_sectors * (num_rings - 1);
    let mut elements = Vec::with_capacity(num_elements);

    // Helper: node index for ring r (1-based), sector s (wraps around)
    let ring_node = |r: usize, s: usize| -> usize { 1 + (r - 1) * num_sectors + s % num_sectors };

    // Inner fan: centre (0) → ring 1
    for s in 0..num_sectors {
        let s_next = (s + 1) % num_sectors;
        elements.push(MeshElement {
            element_type: ElementType::Triangle3,
            node_indices: vec![0, ring_node(1, s), ring_node(1, s_next)],
        });
    }

    // Outer rings
    for r in 2..=num_rings {
        for s in 0..num_sectors {
            let s_next = (s + 1) % num_sectors;

            let inner_cur = ring_node(r - 1, s);
            let inner_next = ring_node(r - 1, s_next);
            let outer_cur = ring_node(r, s);
            let outer_next = ring_node(r, s_next);

            // Two triangles for each quad
            elements.push(MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![inner_cur, outer_cur, outer_next],
            });
            elements.push(MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![inner_cur, outer_next, inner_next],
            });
        }
    }

    Mesh {
        nodes,
        elements,
        dimension: 2,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::validate_mesh;

    #[test]
    fn rectangle_mesh_counts() {
        let mesh = generate_rectangle_mesh(0.0, 3.0, 0.0, 2.0, 3, 2);
        // Nodes: (3+1)*(2+1) = 12
        assert_eq!(mesh.node_count(), 12);
        // Elements: 2 * 3 * 2 = 12
        assert_eq!(mesh.element_count(), 12);
        assert_eq!(mesh.dimension, 2);
    }

    #[test]
    fn rectangle_mesh_validates() {
        let mesh = generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 3, 2);
        assert!(validate_mesh(&mesh).is_ok());
    }

    #[test]
    fn box_mesh_counts_and_validates() {
        let mesh = generate_box_mesh(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2, 2);
        // Nodes: (2+1)*(2+1)*(2+1) = 27
        assert_eq!(mesh.node_count(), 27);
        // Elements: 6 * 2 * 2 * 2 = 48
        assert_eq!(mesh.element_count(), 48);
        assert_eq!(mesh.dimension, 3);
        assert!(validate_mesh(&mesh).is_ok());
    }

    #[test]
    fn disk_mesh_counts_and_validates() {
        let mesh = generate_disk_mesh(0.0, 0.0, 1.0, 3, 8);
        // Nodes: 1 + 3 * 8 = 25
        assert_eq!(mesh.node_count(), 25);
        // Elements: 8 (inner fan) + 2 * 8 * (3 - 1) = 8 + 32 = 40
        assert_eq!(mesh.element_count(), 40);
        assert_eq!(mesh.dimension, 2);
        assert!(validate_mesh(&mesh).is_ok());
    }

    #[test]
    fn empty_rectangle() {
        let mesh = generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 0, 5);
        assert_eq!(mesh.node_count(), 0);
        assert_eq!(mesh.element_count(), 0);
        assert_eq!(mesh.dimension, 2);
    }
}
