use simucad_core::error::MeshError;

use crate::types::Mesh;

// ---------------------------------------------------------------------------
// Mesh validation
// ---------------------------------------------------------------------------

/// Validate the integrity of a [`Mesh`].
///
/// Checks performed:
/// 1. The mesh is not empty (has at least one element).
/// 2. Every element's `node_indices` length matches its `element_type.node_count()`.
/// 3. Every node index is within bounds of `mesh.nodes`.
pub fn validate_mesh(mesh: &Mesh) -> Result<(), MeshError> {
    // 1. Non-empty
    if mesh.elements.is_empty() {
        return Err(MeshError::EmptyMesh);
    }

    let node_count = mesh.nodes.len();

    for (i, elem) in mesh.elements.iter().enumerate() {
        // 2. Node count matches element type
        let expected = elem.element_type.node_count();
        if elem.node_indices.len() != expected {
            return Err(MeshError::InvalidTopology(format!(
                "Element {i} is {:?} and requires {expected} nodes but has {}",
                elem.element_type,
                elem.node_indices.len()
            )));
        }

        // 3. All node indices in bounds
        for &idx in &elem.node_indices {
            if idx >= node_count {
                return Err(MeshError::NodeIndexOutOfBounds {
                    index: idx,
                    node_count,
                });
            }
        }
    }

    Ok(())
}

/// Detect the spatial dimension of the mesh from its element types.
///
/// Returns the maximum `element_type.dimension()` across all elements, or 0
/// for an empty mesh.
pub fn detect_dimension(mesh: &Mesh) -> u8 {
    mesh.elements
        .iter()
        .map(|e| e.element_type.dimension())
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ElementType, MeshElement};
    use simucad_core::types::Vec3;

    fn valid_2d_mesh() -> Mesh {
        Mesh {
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
        }
    }

    #[test]
    fn valid_mesh_passes() {
        assert!(validate_mesh(&valid_2d_mesh()).is_ok());
    }

    #[test]
    fn empty_mesh_fails() {
        let mesh = Mesh {
            nodes: vec![Vec3::ZERO],
            elements: vec![],
            dimension: 2,
        };
        match validate_mesh(&mesh) {
            Err(MeshError::EmptyMesh) => {}
            other => panic!("Expected EmptyMesh, got {other:?}"),
        }
    }

    #[test]
    fn out_of_bounds_index_fails() {
        let mesh = Mesh {
            nodes: vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)],
            elements: vec![MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![0, 1, 99],
            }],
            dimension: 2,
        };
        match validate_mesh(&mesh) {
            Err(MeshError::NodeIndexOutOfBounds { index: 99, .. }) => {}
            other => panic!("Expected NodeIndexOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn wrong_node_count_fails() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            elements: vec![MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![0, 1, 2, 3], // 4 nodes for a triangle
            }],
            dimension: 2,
        };
        match validate_mesh(&mesh) {
            Err(MeshError::InvalidTopology(_)) => {}
            other => panic!("Expected InvalidTopology, got {other:?}"),
        }
    }

    #[test]
    fn detect_dimension_2d() {
        let mesh = valid_2d_mesh();
        assert_eq!(detect_dimension(&mesh), 2);
    }

    #[test]
    fn detect_dimension_3d() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            elements: vec![
                MeshElement {
                    element_type: ElementType::Triangle3,
                    node_indices: vec![0, 1, 2],
                },
                MeshElement {
                    element_type: ElementType::Tetrahedron4,
                    node_indices: vec![0, 1, 2, 3],
                },
            ],
            dimension: 3,
        };
        assert_eq!(detect_dimension(&mesh), 3);
    }

    #[test]
    fn detect_dimension_empty() {
        let mesh = Mesh {
            nodes: vec![],
            elements: vec![],
            dimension: 0,
        };
        assert_eq!(detect_dimension(&mesh), 0);
    }
}
