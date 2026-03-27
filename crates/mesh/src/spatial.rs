use simucad_core::types::{BoundingBox3, Vec3};

use crate::types::{Mesh, MeshElement};

// ---------------------------------------------------------------------------
// BVH — bounding volume hierarchy for spatial queries
// ---------------------------------------------------------------------------

/// A bounding volume hierarchy built from mesh elements for fast spatial
/// queries (point containment, ray casting).
#[derive(Debug, Clone)]
pub struct BVH {
    nodes: Vec<BVHNode>,
}

#[derive(Debug, Clone)]
struct BVHNode {
    bounds: BoundingBox3,
    left: Option<usize>,
    right: Option<usize>,
    /// Set only on leaf nodes — index into the mesh's element list.
    element_index: Option<usize>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl BVH {
    /// Build a BVH from a mesh. Each leaf node wraps the bounding box of one
    /// element; internal nodes are built via recursive median-split along the
    /// longest axis.
    pub fn build(mesh: &Mesh) -> Self {
        if mesh.elements.is_empty() {
            return Self { nodes: Vec::new() };
        }

        // Pre-compute bounding boxes and centroids for each element.
        let mut entries: Vec<BVHEntry> = mesh
            .elements
            .iter()
            .enumerate()
            .map(|(i, elem)| {
                let bb = element_bounding_box(&mesh.nodes, elem);
                let cx = (bb.min.x + bb.max.x) * 0.5;
                let cy = (bb.min.y + bb.max.y) * 0.5;
                let cz = (bb.min.z + bb.max.z) * 0.5;
                BVHEntry {
                    element_index: i,
                    bounds: bb,
                    centroid: Vec3::new(cx, cy, cz),
                }
            })
            .collect();

        let mut bvh_nodes: Vec<BVHNode> = Vec::new();
        build_recursive(&mut entries, &mut bvh_nodes);

        Self { nodes: bvh_nodes }
    }

    /// Return `true` if the BVH is empty (no elements).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // -----------------------------------------------------------------------
    // Point query
    // -----------------------------------------------------------------------

    /// Find all element indices whose bounding box contains `point`.
    pub fn query_point(&self, point: &Vec3) -> Vec<usize> {
        let mut results = Vec::new();
        if !self.nodes.is_empty() {
            self.query_point_recursive(0, point, &mut results);
        }
        results
    }

    fn query_point_recursive(&self, node_idx: usize, point: &Vec3, results: &mut Vec<usize>) {
        let node = &self.nodes[node_idx];
        if !node.bounds.contains(point) {
            return;
        }

        if let Some(elem_idx) = node.element_index {
            results.push(elem_idx);
            return;
        }

        if let Some(left) = node.left {
            self.query_point_recursive(left, point, results);
        }
        if let Some(right) = node.right {
            self.query_point_recursive(right, point, results);
        }
    }

    // -----------------------------------------------------------------------
    // Ray query
    // -----------------------------------------------------------------------

    /// Find all element indices whose bounding box is intersected by the ray
    /// defined by `origin + t * direction` (t >= 0). Returns `(element_index,
    /// t_entry)` pairs sorted by ascending t.
    pub fn query_ray(&self, origin: &Vec3, direction: &Vec3) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        if !self.nodes.is_empty() {
            let inv_dir = Vec3::new(
                if direction.x.abs() > f64::EPSILON {
                    1.0 / direction.x
                } else {
                    f64::INFINITY
                },
                if direction.y.abs() > f64::EPSILON {
                    1.0 / direction.y
                } else {
                    f64::INFINITY
                },
                if direction.z.abs() > f64::EPSILON {
                    1.0 / direction.z
                } else {
                    f64::INFINITY
                },
            );
            self.query_ray_recursive(0, origin, &inv_dir, &mut results);
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn query_ray_recursive(
        &self,
        node_idx: usize,
        origin: &Vec3,
        inv_dir: &Vec3,
        results: &mut Vec<(usize, f64)>,
    ) {
        let node = &self.nodes[node_idx];
        let Some(t_entry) = ray_aabb_intersect(&node.bounds, origin, inv_dir) else {
            return;
        };

        if let Some(elem_idx) = node.element_index {
            results.push((elem_idx, t_entry));
            return;
        }

        if let Some(left) = node.left {
            self.query_ray_recursive(left, origin, inv_dir, results);
        }
        if let Some(right) = node.right {
            self.query_ray_recursive(right, origin, inv_dir, results);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct BVHEntry {
    element_index: usize,
    bounds: BoundingBox3,
    centroid: Vec3,
}

/// Compute the AABB for one element from its node positions.
fn element_bounding_box(nodes: &[Vec3], elem: &MeshElement) -> BoundingBox3 {
    let first = nodes[elem.node_indices[0]];
    let mut min = first;
    let mut max = first;

    for &idx in &elem.node_indices[1..] {
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

/// Merge two AABBs into the enclosing AABB.
fn merge_aabb(a: &BoundingBox3, b: &BoundingBox3) -> BoundingBox3 {
    BoundingBox3::new(
        Vec3::new(a.min.x.min(b.min.x), a.min.y.min(b.min.y), a.min.z.min(b.min.z)),
        Vec3::new(a.max.x.max(b.max.x), a.max.y.max(b.max.y), a.max.z.max(b.max.z)),
    )
}

/// Recursively build the BVH, returning the index of the root node for this
/// sub-tree.
fn build_recursive(entries: &mut [BVHEntry], out: &mut Vec<BVHNode>) -> usize {
    if entries.len() == 1 {
        let e = &entries[0];
        let idx = out.len();
        out.push(BVHNode {
            bounds: e.bounds,
            left: None,
            right: None,
            element_index: Some(e.element_index),
        });
        return idx;
    }

    // Compute combined AABB
    let mut combined = entries[0].bounds;
    for e in entries.iter().skip(1) {
        combined = merge_aabb(&combined, &e.bounds);
    }

    // Split along the longest axis of the centroid spread.
    let mut cmin = entries[0].centroid;
    let mut cmax = entries[0].centroid;
    for e in entries.iter().skip(1) {
        if e.centroid.x < cmin.x { cmin.x = e.centroid.x; }
        if e.centroid.y < cmin.y { cmin.y = e.centroid.y; }
        if e.centroid.z < cmin.z { cmin.z = e.centroid.z; }
        if e.centroid.x > cmax.x { cmax.x = e.centroid.x; }
        if e.centroid.y > cmax.y { cmax.y = e.centroid.y; }
        if e.centroid.z > cmax.z { cmax.z = e.centroid.z; }
    }

    let span = cmax - cmin;
    let axis = if span.x >= span.y && span.x >= span.z {
        0
    } else if span.y >= span.z {
        1
    } else {
        2
    };

    // Partition at median along chosen axis (O(n) instead of O(n log n) sort).
    let mid = entries.len() / 2;
    entries.select_nth_unstable_by(mid, |a, b| {
        let va = match axis {
            0 => a.centroid.x,
            1 => a.centroid.y,
            _ => a.centroid.z,
        };
        let vb = match axis {
            0 => b.centroid.x,
            1 => b.centroid.y,
            _ => b.centroid.z,
        };
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let (left_entries, right_entries) = entries.split_at_mut(mid);

    // Reserve a slot for this internal node.
    let node_idx = out.len();
    out.push(BVHNode {
        bounds: combined,
        left: None,
        right: None,
        element_index: None,
    });

    let left_idx = build_recursive(left_entries, out);
    let right_idx = build_recursive(right_entries, out);

    out[node_idx].left = Some(left_idx);
    out[node_idx].right = Some(right_idx);

    node_idx
}

/// Ray-AABB slab test. Returns `Some(t_entry)` if the ray hits the box with
/// t >= 0, or `None` if it misses.
fn ray_aabb_intersect(bb: &BoundingBox3, origin: &Vec3, inv_dir: &Vec3) -> Option<f64> {
    let t1x = (bb.min.x - origin.x) * inv_dir.x;
    let t2x = (bb.max.x - origin.x) * inv_dir.x;
    let t1y = (bb.min.y - origin.y) * inv_dir.y;
    let t2y = (bb.max.y - origin.y) * inv_dir.y;
    let t1z = (bb.min.z - origin.z) * inv_dir.z;
    let t2z = (bb.max.z - origin.z) * inv_dir.z;

    let tmin_x = t1x.min(t2x);
    let tmax_x = t1x.max(t2x);
    let tmin_y = t1y.min(t2y);
    let tmax_y = t1y.max(t2y);
    let tmin_z = t1z.min(t2z);
    let tmax_z = t1z.max(t2z);

    let tmin = tmin_x.max(tmin_y).max(tmin_z);
    let tmax = tmax_x.min(tmax_y).min(tmax_z);

    if tmax < 0.0 || tmin > tmax {
        None
    } else {
        Some(tmin.max(0.0))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ElementType, MeshElement};

    fn make_tri(indices: Vec<usize>) -> MeshElement {
        MeshElement {
            element_type: ElementType::Triangle3,
            node_indices: indices,
        }
    }

    fn sample_mesh() -> Mesh {
        // Two triangles forming a unit square in the XY plane.
        Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            elements: vec![
                make_tri(vec![0, 1, 2]),
                make_tri(vec![0, 2, 3]),
            ],
            dimension: 2,
        }
    }

    #[test]
    fn bvh_build_empty() {
        let mesh = Mesh {
            nodes: vec![],
            elements: vec![],
            dimension: 2,
        };
        let bvh = BVH::build(&mesh);
        assert!(bvh.is_empty());
        assert!(bvh.query_point(&Vec3::ZERO).is_empty());
    }

    #[test]
    fn bvh_build_single_element() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
            ],
            elements: vec![make_tri(vec![0, 1, 2])],
            dimension: 2,
        };

        let bvh = BVH::build(&mesh);
        assert_eq!(bvh.nodes.len(), 1);

        // Point inside the bounding box
        let hits = bvh.query_point(&Vec3::new(0.5, 0.5, 0.0));
        assert_eq!(hits, vec![0]);

        // Point outside
        let hits = bvh.query_point(&Vec3::new(5.0, 5.0, 0.0));
        assert!(hits.is_empty());
    }

    #[test]
    fn bvh_point_query_multi() {
        let mesh = sample_mesh();
        let bvh = BVH::build(&mesh);

        // Center of the square — should hit both triangles' bounding boxes.
        let hits = bvh.query_point(&Vec3::new(0.5, 0.5, 0.0));
        assert_eq!(hits.len(), 2);

        // Outside the square entirely.
        let hits = bvh.query_point(&Vec3::new(2.0, 2.0, 0.0));
        assert!(hits.is_empty());
    }

    #[test]
    fn bvh_ray_query_hits() {
        let mesh = sample_mesh();
        let bvh = BVH::build(&mesh);

        // Ray from above pointing down through the square.
        let origin = Vec3::new(0.5, 0.5, 5.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);
        let hits = bvh.query_ray(&origin, &direction);

        // Should hit both triangle bounding boxes.
        assert_eq!(hits.len(), 2);
        // t values should be positive (ray starts above the z=0 plane).
        for &(_, t) in &hits {
            assert!(t >= 0.0);
        }
    }

    #[test]
    fn bvh_ray_query_misses() {
        let mesh = sample_mesh();
        let bvh = BVH::build(&mesh);

        // Ray pointing away from the mesh.
        let origin = Vec3::new(5.0, 5.0, 5.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);
        let hits = bvh.query_ray(&origin, &direction);
        assert!(hits.is_empty());
    }

    #[test]
    fn bvh_ray_query_sorted_by_t() {
        // Two triangles at different z-levels.
        let mesh = Mesh {
            nodes: vec![
                // Triangle at z = 0
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
                // Triangle at z = 2
                Vec3::new(0.0, 0.0, 2.0),
                Vec3::new(1.0, 0.0, 2.0),
                Vec3::new(0.5, 1.0, 2.0),
            ],
            elements: vec![
                make_tri(vec![0, 1, 2]),
                make_tri(vec![3, 4, 5]),
            ],
            dimension: 3,
        };

        let bvh = BVH::build(&mesh);

        // Ray from z=10 pointing down.
        let origin = Vec3::new(0.5, 0.3, 10.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);
        let hits = bvh.query_ray(&origin, &direction);

        assert_eq!(hits.len(), 2);
        // First hit should have smaller t (closer to z=10, i.e. the z=2 triangle).
        assert!(hits[0].1 <= hits[1].1);
    }

    #[test]
    fn bvh_many_elements() {
        // Build a mesh with many triangles to exercise the recursive split.
        let mut nodes = Vec::new();
        let mut elements = Vec::new();

        for i in 0..20 {
            let x = i as f64;
            let base = nodes.len();
            nodes.push(Vec3::new(x, 0.0, 0.0));
            nodes.push(Vec3::new(x + 1.0, 0.0, 0.0));
            nodes.push(Vec3::new(x + 0.5, 1.0, 0.0));
            elements.push(make_tri(vec![base, base + 1, base + 2]));
        }

        let mesh = Mesh {
            nodes,
            elements,
            dimension: 2,
        };

        let bvh = BVH::build(&mesh);

        // Query a point that should be in triangle 10 (x in [10, 11]).
        let hits = bvh.query_point(&Vec3::new(10.5, 0.5, 0.0));
        assert!(hits.contains(&10));

        // Query a point outside all triangles.
        let hits = bvh.query_point(&Vec3::new(100.0, 0.0, 0.0));
        assert!(hits.is_empty());
    }
}
