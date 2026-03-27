use serde::{Deserialize, Serialize};
use simucad_core::types::{BoundingBox3, Vec3};

// ---------------------------------------------------------------------------
// ElementType — supported finite-element topologies
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementType {
    /// 2-node line segment (1D element).
    Line2,
    /// 3-node triangle (2D element).
    Triangle3,
    /// 4-node tetrahedron (3D element).
    Tetrahedron4,
}

impl ElementType {
    /// Number of nodes that define this element.
    pub fn node_count(&self) -> usize {
        match self {
            Self::Line2 => 2,
            Self::Triangle3 => 3,
            Self::Tetrahedron4 => 4,
        }
    }

    /// Spatial dimension of this element (1D, 2D, or 3D).
    pub fn dimension(&self) -> u8 {
        match self {
            Self::Line2 => 1,
            Self::Triangle3 => 2,
            Self::Tetrahedron4 => 3,
        }
    }

    /// Convert a Gmsh integer element-type code to an `ElementType`.
    /// Returns `None` for unsupported codes.
    pub fn from_gmsh_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::Line2),
            2 => Some(Self::Triangle3),
            4 => Some(Self::Tetrahedron4),
            _ => None,
        }
    }

    /// Return the Gmsh integer code for this element type.
    pub fn to_gmsh_code(&self) -> u32 {
        match self {
            Self::Line2 => 1,
            Self::Triangle3 => 2,
            Self::Tetrahedron4 => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// MeshElement — one element in the mesh
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshElement {
    pub element_type: ElementType,
    /// Zero-based indices into `Mesh::nodes`.
    pub node_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Mesh — the top-level mesh data structure
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mesh {
    /// Node coordinates.
    pub nodes: Vec<Vec3>,
    /// Finite elements that reference `nodes` by index.
    pub elements: Vec<MeshElement>,
    /// Spatial dimension of the mesh (1, 2, or 3).
    pub dimension: u8,
}

impl Mesh {
    /// Compute the axis-aligned bounding box of all nodes.
    ///
    /// If the mesh has no nodes the returned box collapses to the origin.
    pub fn bounding_box(&self) -> BoundingBox3 {
        if self.nodes.is_empty() {
            return BoundingBox3::new(Vec3::ZERO, Vec3::ZERO);
        }

        let first = self.nodes[0];
        let mut min = first;
        let mut max = first;

        for node in &self.nodes[1..] {
            if node.x < min.x {
                min.x = node.x;
            }
            if node.y < min.y {
                min.y = node.y;
            }
            if node.z < min.z {
                min.z = node.z;
            }
            if node.x > max.x {
                max.x = node.x;
            }
            if node.y > max.y {
                max.y = node.y;
            }
            if node.z > max.z {
                max.z = node.z;
            }
        }

        BoundingBox3::new(min, max)
    }

    /// Total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of elements.
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }
}

impl MeshElement {
    /// Compute the centroid (geometric centre) of this element.
    pub fn element_centroid(&self, nodes: &[Vec3]) -> Vec3 {
        let n = self.node_indices.len() as f64;
        let mut sum = Vec3::ZERO;
        for &idx in &self.node_indices {
            let v = nodes[idx];
            sum = sum + v;
        }
        sum * (1.0 / n)
    }

    /// Compute the axis-aligned bounding box enclosing this element.
    pub fn element_bounding_box(&self, nodes: &[Vec3]) -> BoundingBox3 {
        let first = nodes[self.node_indices[0]];
        let mut min = first;
        let mut max = first;

        for &idx in &self.node_indices[1..] {
            let v = nodes[idx];
            if v.x < min.x { min.x = v.x; }
            if v.y < min.y { min.y = v.y; }
            if v.z < min.z { min.z = v.z; }
            if v.x > max.x { max.x = v.x; }
            if v.y > max.y { max.y = v.y; }
            if v.z > max.z { max.z = v.z; }
        }

        BoundingBox3::new(min, max)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mesh() -> Mesh {
        Mesh {
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
        }
    }

    #[test]
    fn element_type_node_count() {
        assert_eq!(ElementType::Line2.node_count(), 2);
        assert_eq!(ElementType::Triangle3.node_count(), 3);
        assert_eq!(ElementType::Tetrahedron4.node_count(), 4);
    }

    #[test]
    fn element_type_dimension() {
        assert_eq!(ElementType::Line2.dimension(), 1);
        assert_eq!(ElementType::Triangle3.dimension(), 2);
        assert_eq!(ElementType::Tetrahedron4.dimension(), 3);
    }

    #[test]
    fn gmsh_code_round_trip() {
        for et in [
            ElementType::Line2,
            ElementType::Triangle3,
            ElementType::Tetrahedron4,
        ] {
            assert_eq!(ElementType::from_gmsh_code(et.to_gmsh_code()), Some(et));
        }
        assert_eq!(ElementType::from_gmsh_code(99), None);
    }

    #[test]
    fn mesh_bounding_box() {
        let mesh = sample_mesh();
        let bb = mesh.bounding_box();
        assert_eq!(bb.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(bb.max, Vec3::new(1.5, 1.0, 0.0));
    }

    #[test]
    fn mesh_counts() {
        let mesh = sample_mesh();
        assert_eq!(mesh.node_count(), 4);
        assert_eq!(mesh.element_count(), 2);
    }

    #[test]
    fn empty_mesh_bounding_box() {
        let mesh = Mesh {
            nodes: vec![],
            elements: vec![],
            dimension: 2,
        };
        let bb = mesh.bounding_box();
        assert_eq!(bb.min, Vec3::ZERO);
        assert_eq!(bb.max, Vec3::ZERO);
    }

    #[test]
    fn element_centroid_triangle() {
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let elem = MeshElement {
            element_type: ElementType::Triangle3,
            node_indices: vec![0, 1, 2],
        };
        let c = elem.element_centroid(&nodes);
        assert!((c.x - 1.0).abs() < 1e-12);
        assert!((c.y - 1.0).abs() < 1e-12);
        assert!((c.z - 0.0).abs() < 1e-12);
    }

    #[test]
    fn element_bounding_box_triangle() {
        let nodes = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 0.0, 1.0),
            Vec3::new(2.0, 5.0, 0.0),
        ];
        let elem = MeshElement {
            element_type: ElementType::Triangle3,
            node_indices: vec![0, 1, 2],
        };
        let bb = elem.element_bounding_box(&nodes);
        assert_eq!(bb.min, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(bb.max, Vec3::new(4.0, 5.0, 3.0));
    }
}
