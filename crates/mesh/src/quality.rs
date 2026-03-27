use simucad_core::types::Vec3;

use crate::types::{ElementType, Mesh, MeshElement};

/// Tolerance for detecting degenerate (near-zero area/volume) elements.
const DEGENERATE_EPSILON: f64 = 1e-12;

// ---------------------------------------------------------------------------
// ElementQuality — per-element quality metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ElementQuality {
    /// Ratio of longest edge to shortest edge (>= 1.0, 1.0 = ideal).
    pub aspect_ratio: f64,
    /// Skewness measure: 0 = ideal, 1 = degenerate.
    pub skewness: f64,
    /// Element area (2D) or volume (3D); for Line2 this is the length.
    pub area_or_volume: f64,
    /// Minimum interior angle in radians.
    pub min_angle: f64,
    /// Maximum interior angle in radians.
    pub max_angle: f64,
}

// ---------------------------------------------------------------------------
// MeshQualityReport — aggregate statistics for the entire mesh
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct MeshQualityReport {
    pub element_count: usize,
    pub node_count: usize,
    pub dimension: u8,
    pub min_quality: ElementQuality,
    pub max_quality: ElementQuality,
    pub mean_aspect_ratio: f64,
    pub mean_skewness: f64,
    pub total_area_or_volume: f64,
    /// Number of elements with area/volume below the degenerate threshold.
    pub degenerate_count: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the distance between two Vec3 points.
fn distance(a: &Vec3, b: &Vec3) -> f64 {
    (*a - *b).magnitude()
}

/// Compute the angle at vertex B in triangle ABC using the law of cosines.
/// Returns angle in radians.
fn angle_at_vertex(a: &Vec3, b: &Vec3, c: &Vec3) -> f64 {
    let ba = *a - *b;
    let bc = *c - *b;
    let dot = ba.dot(&bc);
    let denom = ba.magnitude() * bc.magnitude();
    if denom < f64::EPSILON {
        return 0.0;
    }
    let cos_theta = (dot / denom).clamp(-1.0, 1.0);
    cos_theta.acos()
}

// ---------------------------------------------------------------------------
// compute_element_quality
// ---------------------------------------------------------------------------

/// Compute quality metrics for a single mesh element.
///
/// # Panics
///
/// Panics if the element references node indices that are out of bounds.
pub fn compute_element_quality(nodes: &[Vec3], element: &MeshElement) -> ElementQuality {
    match element.element_type {
        ElementType::Line2 => quality_line2(nodes, &element.node_indices),
        ElementType::Triangle3 => quality_triangle3(nodes, &element.node_indices),
        ElementType::Tetrahedron4 => quality_tetrahedron4(nodes, &element.node_indices),
    }
}

fn quality_line2(nodes: &[Vec3], idx: &[usize]) -> ElementQuality {
    let length = distance(&nodes[idx[0]], &nodes[idx[1]]);
    ElementQuality {
        aspect_ratio: 1.0,
        skewness: 0.0,
        area_or_volume: length,
        min_angle: std::f64::consts::PI,
        max_angle: std::f64::consts::PI,
    }
}

fn quality_triangle3(nodes: &[Vec3], idx: &[usize]) -> ElementQuality {
    let (a, b, c) = (nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]);

    // Edge lengths
    let ab = distance(&a, &b);
    let bc = distance(&b, &c);
    let ca = distance(&c, &a);

    let edges = [ab, bc, ca];
    let longest = edges.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shortest = edges.iter().cloned().fold(f64::INFINITY, f64::min);

    let aspect_ratio = if shortest > f64::EPSILON {
        longest / shortest
    } else {
        f64::INFINITY
    };

    // Interior angles
    let angle_a = angle_at_vertex(&b, &a, &c);
    let angle_b = angle_at_vertex(&a, &b, &c);
    let angle_c = angle_at_vertex(&a, &c, &b);
    let angles = [angle_a, angle_b, angle_c];
    let min_angle = angles.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_angle = angles.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Area via cross product: 0.5 * |AB x AC|
    let ab_vec = b - a;
    let ac_vec = c - a;
    let area = 0.5 * ab_vec.cross(&ac_vec).magnitude();

    // Skewness: compare max angle to the equilateral ideal (pi/3).
    // skewness = (max_angle - ideal_angle) / (pi - ideal_angle)
    let ideal_angle = std::f64::consts::PI / 3.0;
    let skewness = ((max_angle - ideal_angle) / (std::f64::consts::PI - ideal_angle)).clamp(0.0, 1.0);

    ElementQuality {
        aspect_ratio,
        skewness,
        area_or_volume: area,
        min_angle,
        max_angle,
    }
}

fn quality_tetrahedron4(nodes: &[Vec3], idx: &[usize]) -> ElementQuality {
    let (a, b, c, d) = (nodes[idx[0]], nodes[idx[1]], nodes[idx[2]], nodes[idx[3]]);

    // Six edges
    let edge_lengths = [
        distance(&a, &b),
        distance(&a, &c),
        distance(&a, &d),
        distance(&b, &c),
        distance(&b, &d),
        distance(&c, &d),
    ];

    let longest = edge_lengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shortest = edge_lengths.iter().cloned().fold(f64::INFINITY, f64::min);

    let aspect_ratio = if shortest > f64::EPSILON {
        longest / shortest
    } else {
        f64::INFINITY
    };

    // Volume via scalar triple product: V = |det([AB, AC, AD])| / 6
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let volume = (ab.dot(&ac.cross(&ad))).abs() / 6.0;

    // Dihedral / face angles: compute angles at each vertex of the tet.
    // At vertex A: angle between edges AB, AC, AD (take pairwise).
    // We compute angles of all 12 face-vertex combinations (4 faces x 3 vertices).
    let faces: [(usize, usize, usize); 4] = [
        (0, 1, 2), // face ABC
        (0, 1, 3), // face ABD
        (0, 2, 3), // face ACD
        (1, 2, 3), // face BCD
    ];
    let verts = [a, b, c, d];

    let mut min_angle = f64::INFINITY;
    let mut max_angle = f64::NEG_INFINITY;

    for &(i, j, k) in &faces {
        // Angle at vertex i
        let ang_i = angle_at_vertex(&verts[j], &verts[i], &verts[k]);
        // Angle at vertex j
        let ang_j = angle_at_vertex(&verts[i], &verts[j], &verts[k]);
        // Angle at vertex k
        let ang_k = angle_at_vertex(&verts[i], &verts[k], &verts[j]);

        for &ang in &[ang_i, ang_j, ang_k] {
            if ang < min_angle {
                min_angle = ang;
            }
            if ang > max_angle {
                max_angle = ang;
            }
        }
    }

    // Skewness: for a regular tetrahedron the ideal face angle is acos(1/3) ~ 70.53 deg.
    let ideal_angle = (1.0_f64 / 3.0).acos();
    let skewness = ((max_angle - ideal_angle) / (std::f64::consts::PI - ideal_angle)).clamp(0.0, 1.0);

    ElementQuality {
        aspect_ratio,
        skewness,
        area_or_volume: volume,
        min_angle,
        max_angle,
    }
}

// ---------------------------------------------------------------------------
// compute_mesh_quality
// ---------------------------------------------------------------------------

/// Compute aggregate quality statistics for an entire mesh.
///
/// Returns `None` if the mesh has no elements.
pub fn compute_mesh_quality(mesh: &Mesh) -> Option<MeshQualityReport> {
    if mesh.elements.is_empty() {
        return None;
    }

    let mut sum_aspect = 0.0;
    let mut sum_skewness = 0.0;
    let mut total_area_volume = 0.0;
    let mut degenerate_count = 0usize;

    // Track min/max quality records.
    let mut min_quality: Option<ElementQuality> = None;
    let mut max_quality: Option<ElementQuality> = None;

    for elem in &mesh.elements {
        let q = compute_element_quality(&mesh.nodes, elem);

        sum_aspect += q.aspect_ratio;
        sum_skewness += q.skewness;
        total_area_volume += q.area_or_volume;

        if q.area_or_volume < DEGENERATE_EPSILON {
            degenerate_count += 1;
        }

        // Track element with worst (highest) aspect ratio as max_quality,
        // and element with best (lowest) aspect ratio as min_quality.
        match &min_quality {
            None => min_quality = Some(q.clone()),
            Some(cur) if q.aspect_ratio < cur.aspect_ratio => min_quality = Some(q.clone()),
            _ => {}
        }
        match &max_quality {
            None => max_quality = Some(q.clone()),
            Some(cur) if q.aspect_ratio > cur.aspect_ratio => max_quality = Some(q.clone()),
            _ => {}
        }
    }

    let n = mesh.elements.len() as f64;

    Some(MeshQualityReport {
        element_count: mesh.elements.len(),
        node_count: mesh.nodes.len(),
        dimension: mesh.dimension,
        min_quality: min_quality.unwrap(),
        max_quality: max_quality.unwrap(),
        mean_aspect_ratio: sum_aspect / n,
        mean_skewness: sum_skewness / n,
        total_area_or_volume: total_area_volume,
        degenerate_count,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ElementType, MeshElement};

    fn make_element(et: ElementType, indices: Vec<usize>) -> MeshElement {
        MeshElement {
            element_type: et,
            node_indices: indices,
        }
    }

    // ---- Triangle tests ----

    #[test]
    fn equilateral_triangle_quality() {
        // Equilateral triangle with side length 2
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 3.0_f64.sqrt(), 0.0),
        ];
        let elem = make_element(ElementType::Triangle3, vec![0, 1, 2]);
        let q = compute_element_quality(&nodes, &elem);

        // All edges equal => aspect ratio = 1
        assert!((q.aspect_ratio - 1.0).abs() < 1e-10);
        // All angles = pi/3
        assert!((q.min_angle - std::f64::consts::PI / 3.0).abs() < 1e-10);
        assert!((q.max_angle - std::f64::consts::PI / 3.0).abs() < 1e-10);
        // Skewness = 0 for equilateral
        assert!(q.skewness.abs() < 1e-10);
        // Area = sqrt(3)
        assert!((q.area_or_volume - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn right_triangle_quality() {
        // 3-4-5 right triangle
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
        ];
        let elem = make_element(ElementType::Triangle3, vec![0, 1, 2]);
        let q = compute_element_quality(&nodes, &elem);

        // Aspect ratio = 5/3
        assert!((q.aspect_ratio - 5.0 / 3.0).abs() < 1e-10);
        // Min angle: atan(3/4) ~ 0.6435 rad
        assert!((q.min_angle - (3.0_f64 / 4.0).atan()).abs() < 1e-10);
        // Max angle: pi/2
        assert!((q.max_angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        // Area = 6
        assert!((q.area_or_volume - 6.0).abs() < 1e-10);
    }

    #[test]
    fn degenerate_triangle_quality() {
        // Collinear points => degenerate
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let elem = make_element(ElementType::Triangle3, vec![0, 1, 2]);
        let q = compute_element_quality(&nodes, &elem);

        assert!(q.area_or_volume < DEGENERATE_EPSILON);
        // Max angle should be ~pi (degenerate)
        assert!((q.max_angle - std::f64::consts::PI).abs() < 1e-10);
        // Skewness should be ~1.0
        assert!((q.skewness - 1.0).abs() < 1e-10);
    }

    // ---- Line2 tests ----

    #[test]
    fn line2_quality() {
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 4.0, 0.0),
        ];
        let elem = make_element(ElementType::Line2, vec![0, 1]);
        let q = compute_element_quality(&nodes, &elem);

        assert!((q.aspect_ratio - 1.0).abs() < 1e-12);
        assert!((q.skewness).abs() < 1e-12);
        assert!((q.area_or_volume - 5.0).abs() < 1e-12);
    }

    // ---- Tetrahedron tests ----

    #[test]
    fn regular_tetrahedron_quality() {
        // Regular tetrahedron with edge length sqrt(2)
        let nodes = vec![
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];
        let elem = make_element(ElementType::Tetrahedron4, vec![0, 1, 2, 3]);
        let q = compute_element_quality(&nodes, &elem);

        // All edges equal => aspect ratio = 1
        assert!((q.aspect_ratio - 1.0).abs() < 1e-10);
        // Volume = 8/3 for this specific regular tet
        assert!((q.area_or_volume - 8.0 / 3.0).abs() < 1e-10);
        // Skewness should be ~0 for a regular tet
        assert!(q.skewness.abs() < 1e-10);
    }

    #[test]
    fn degenerate_tetrahedron_quality() {
        // All four points coplanar => volume ~ 0
        let nodes = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let elem = make_element(ElementType::Tetrahedron4, vec![0, 1, 2, 3]);
        let q = compute_element_quality(&nodes, &elem);

        assert!(q.area_or_volume < DEGENERATE_EPSILON);
    }

    // ---- Mesh-level report tests ----

    #[test]
    fn mesh_quality_report_basic() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(1.0, 3.0_f64.sqrt(), 0.0),
                Vec3::new(3.0, 3.0_f64.sqrt(), 0.0),
            ],
            elements: vec![
                make_element(ElementType::Triangle3, vec![0, 1, 2]),
                make_element(ElementType::Triangle3, vec![1, 3, 2]),
            ],
            dimension: 2,
        };

        let report = compute_mesh_quality(&mesh).expect("should produce report");
        assert_eq!(report.element_count, 2);
        assert_eq!(report.node_count, 4);
        assert_eq!(report.dimension, 2);
        assert_eq!(report.degenerate_count, 0);
        assert!(report.total_area_or_volume > 0.0);
        assert!(report.mean_aspect_ratio >= 1.0);
    }

    #[test]
    fn mesh_quality_empty() {
        let mesh = Mesh {
            nodes: vec![],
            elements: vec![],
            dimension: 2,
        };
        assert!(compute_mesh_quality(&mesh).is_none());
    }

    #[test]
    fn mesh_quality_detects_degenerates() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0), // collinear
                Vec3::new(0.5, 1.0, 0.0),
            ],
            elements: vec![
                // Degenerate triangle (collinear)
                make_element(ElementType::Triangle3, vec![0, 1, 2]),
                // Valid triangle
                make_element(ElementType::Triangle3, vec![0, 1, 3]),
            ],
            dimension: 2,
        };

        let report = compute_mesh_quality(&mesh).expect("should produce report");
        assert_eq!(report.degenerate_count, 1);
    }
}
