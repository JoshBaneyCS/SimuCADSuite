use std::collections::HashMap;

use simucad_core::types::Vec3;

use crate::types::{ElementType, Mesh, MeshElement};

/// Refine a mesh by subdividing each triangle into 4 via edge midpoint insertion.
///
/// - `Triangle3` elements are split into 4 sub-triangles.
/// - `Line2` elements are split into 2 sub-lines.
/// - `Tetrahedron4` elements are passed through unchanged.
/// - Shared edges produce shared midpoint nodes (no duplicates).
pub fn subdivide_midpoint(mesh: &Mesh) -> Mesh {
    let mut new_nodes: Vec<Vec3> = mesh.nodes.clone();
    let mut new_elements: Vec<MeshElement> = Vec::new();
    let mut edge_midpoints: HashMap<(usize, usize), usize> = HashMap::new();

    let get_or_create_midpoint =
        |a: usize, b: usize, nodes: &mut Vec<Vec3>, map: &mut HashMap<(usize, usize), usize>| -> usize {
            let key = (a.min(b), a.max(b));
            if let Some(&idx) = map.get(&key) {
                return idx;
            }
            let midpoint = (mesh.nodes[a] + mesh.nodes[b]) * 0.5;
            let idx = nodes.len();
            nodes.push(midpoint);
            map.insert(key, idx);
            idx
        };

    for elem in &mesh.elements {
        match elem.element_type {
            ElementType::Triangle3 => {
                let a = elem.node_indices[0];
                let b = elem.node_indices[1];
                let c = elem.node_indices[2];

                let m_ab = get_or_create_midpoint(a, b, &mut new_nodes, &mut edge_midpoints);
                let m_bc = get_or_create_midpoint(b, c, &mut new_nodes, &mut edge_midpoints);
                let m_ca = get_or_create_midpoint(c, a, &mut new_nodes, &mut edge_midpoints);

                new_elements.push(MeshElement {
                    element_type: ElementType::Triangle3,
                    node_indices: vec![a, m_ab, m_ca],
                });
                new_elements.push(MeshElement {
                    element_type: ElementType::Triangle3,
                    node_indices: vec![m_ab, b, m_bc],
                });
                new_elements.push(MeshElement {
                    element_type: ElementType::Triangle3,
                    node_indices: vec![m_ca, m_bc, c],
                });
                new_elements.push(MeshElement {
                    element_type: ElementType::Triangle3,
                    node_indices: vec![m_ab, m_bc, m_ca],
                });
            }
            ElementType::Line2 => {
                let a = elem.node_indices[0];
                let b = elem.node_indices[1];

                let m = get_or_create_midpoint(a, b, &mut new_nodes, &mut edge_midpoints);

                new_elements.push(MeshElement {
                    element_type: ElementType::Line2,
                    node_indices: vec![a, m],
                });
                new_elements.push(MeshElement {
                    element_type: ElementType::Line2,
                    node_indices: vec![m, b],
                });
            }
            ElementType::Tetrahedron4 => {
                new_elements.push(elem.clone());
            }
        }
    }

    Mesh {
        nodes: new_nodes,
        elements: new_elements,
        dimension: mesh.dimension,
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
    fn subdivide_single_triangle() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
            ],
            elements: vec![MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![0, 1, 2],
            }],
            dimension: 2,
        };

        let refined = subdivide_midpoint(&mesh);

        // 3 original + 3 edge midpoints = 6 nodes
        assert_eq!(refined.nodes.len(), 6);
        // 1 triangle -> 4 triangles
        assert_eq!(refined.elements.len(), 4);
        for elem in &refined.elements {
            assert_eq!(elem.element_type, ElementType::Triangle3);
            assert_eq!(elem.node_indices.len(), 3);
        }
    }

    #[test]
    fn subdivide_two_sharing_edge() {
        // Two triangles sharing edge (1, 2):
        //   tri0 = [0, 1, 2]
        //   tri1 = [1, 3, 2]
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
                Vec3::new(1.5, 1.0, 0.0),
            ],
            elements: vec![
                MeshElement {
                    element_type: ElementType::Triangle3,
                    node_indices: vec![0, 1, 2],
                },
                MeshElement {
                    element_type: ElementType::Triangle3,
                    node_indices: vec![1, 3, 2],
                },
            ],
            dimension: 2,
        };

        let refined = subdivide_midpoint(&mesh);

        // 2 triangles -> 8 triangles
        assert_eq!(refined.elements.len(), 8);

        // Unique edges across both triangles:
        //   tri0 edges: (0,1), (1,2), (0,2)
        //   tri1 edges: (1,3), (2,3), (1,2)  <-- (1,2) is shared
        // 5 unique edges -> 5 midpoints + 4 original = 9 nodes
        assert_eq!(refined.nodes.len(), 9);
    }

    #[test]
    fn subdivide_preserves_dimension() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
            ],
            elements: vec![MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![0, 1, 2],
            }],
            dimension: 3,
        };

        let refined = subdivide_midpoint(&mesh);
        assert_eq!(refined.dimension, 3);
    }

    #[test]
    fn subdivide_line_elements() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
            elements: vec![MeshElement {
                element_type: ElementType::Line2,
                node_indices: vec![0, 1],
            }],
            dimension: 1,
        };

        let refined = subdivide_midpoint(&mesh);

        // 2 original + 1 midpoint = 3 nodes
        assert_eq!(refined.nodes.len(), 3);
        // 1 line -> 2 lines
        assert_eq!(refined.elements.len(), 2);
        for elem in &refined.elements {
            assert_eq!(elem.element_type, ElementType::Line2);
            assert_eq!(elem.node_indices.len(), 2);
        }

        // Verify midpoint coordinate
        let mid = refined.nodes[2];
        assert!((mid.x - 1.0).abs() < 1e-12);
        assert!((mid.y - 0.0).abs() < 1e-12);
        assert!((mid.z - 0.0).abs() < 1e-12);
    }

    #[test]
    fn subdivide_validates() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
                Vec3::new(1.5, 1.0, 0.0),
            ],
            elements: vec![
                MeshElement {
                    element_type: ElementType::Triangle3,
                    node_indices: vec![0, 1, 2],
                },
                MeshElement {
                    element_type: ElementType::Triangle3,
                    node_indices: vec![1, 3, 2],
                },
                MeshElement {
                    element_type: ElementType::Line2,
                    node_indices: vec![0, 1],
                },
            ],
            dimension: 2,
        };

        let refined = subdivide_midpoint(&mesh);
        assert!(validate_mesh(&refined).is_ok());
    }
}
