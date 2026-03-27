//! End-to-end integration tests for mesh parsing, validation, quality, and I/O.
//!
//! These tests exercise cross-module workflows: loading .msh files, validating
//! them, computing quality metrics, parsing STL, building BVH, and mesh
//! round-trips (load -> save -> reload -> compare).

use std::path::PathBuf;

use simucad_core::types::Vec3;
use simucad_mesh::io::{GmshLoader, MeshLoader};
use simucad_mesh::parser::parse_msh;
use simucad_mesh::quality::{compute_element_quality, compute_mesh_quality};
use simucad_mesh::spatial::BVH;
use simucad_mesh::stl::parse_stl_ascii;
use simucad_mesh::types::{ElementType, Mesh, MeshElement};
use simucad_mesh::validation::validate_mesh;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_fixtures")
}

// ---------------------------------------------------------------------------
// 1. Load sample.msh, validate, compute quality metrics
// ---------------------------------------------------------------------------

#[test]
fn load_sample_msh_validate_and_quality() {
    let path = fixtures_dir().join("sample.msh");
    let mesh = GmshLoader::load(&path).expect("should load sample.msh");

    // Basic structure checks
    assert_eq!(mesh.node_count(), 4);
    assert_eq!(mesh.element_count(), 2);
    assert_eq!(mesh.dimension, 2);

    // Validation should pass (GmshLoader already validates, but test explicitly)
    validate_mesh(&mesh).expect("mesh should be valid");

    // Compute quality
    let report = compute_mesh_quality(&mesh).expect("should produce quality report");
    assert_eq!(report.element_count, 2);
    assert_eq!(report.node_count, 4);
    assert_eq!(report.dimension, 2);
    assert_eq!(report.degenerate_count, 0);
    assert!(report.total_area_or_volume > 0.0);
    assert!(report.mean_aspect_ratio >= 1.0);
    assert!(report.mean_skewness >= 0.0);
    assert!(report.mean_skewness <= 1.0);

    // Per-element quality
    for elem in &mesh.elements {
        let q = compute_element_quality(&mesh.nodes, elem);
        assert!(q.aspect_ratio >= 1.0);
        assert!(q.area_or_volume > 0.0);
        assert!(q.min_angle > 0.0);
        assert!(q.max_angle > q.min_angle || (q.max_angle - q.min_angle).abs() < 1e-10);
    }
}

// ---------------------------------------------------------------------------
// 2. Parse a manually constructed STL ASCII string, validate the mesh
// ---------------------------------------------------------------------------

#[test]
fn parse_stl_ascii_and_validate() {
    let stl_str = "\
solid test_cube_face
  facet normal 0 0 1
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 0.0 0.0
      vertex 0.0 1.0 0.0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 1.0 0.0 0.0
      vertex 1.0 1.0 0.0
      vertex 0.0 1.0 0.0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0.0 0.0 1.0
      vertex 0.0 1.0 1.0
      vertex 1.0 0.0 1.0
    endloop
  endfacet
endsolid test_cube_face
";

    let mesh = parse_stl_ascii(stl_str).expect("should parse STL ASCII");

    // 3 triangles
    assert_eq!(mesh.element_count(), 3);

    // All elements should be Triangle3
    for elem in &mesh.elements {
        assert_eq!(elem.element_type, ElementType::Triangle3);
        assert_eq!(elem.node_indices.len(), 3);
    }

    // Mesh should be valid
    validate_mesh(&mesh).expect("parsed STL should be valid");

    // Compute quality -- no degenerate triangles
    let report = compute_mesh_quality(&mesh).expect("should produce report");
    assert_eq!(report.degenerate_count, 0);
    assert!(report.total_area_or_volume > 0.0);

    // Bounding box should span [0,1] x [0,1] x [0,1]
    let bb = mesh.bounding_box();
    assert!((bb.min.x - 0.0).abs() < 1e-9);
    assert!((bb.min.y - 0.0).abs() < 1e-9);
    assert!((bb.min.z - 0.0).abs() < 1e-9);
    assert!((bb.max.x - 1.0).abs() < 1e-9);
    assert!((bb.max.y - 1.0).abs() < 1e-9);
    assert!((bb.max.z - 1.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// 3. Build BVH from mesh, query a point, verify it returns correct elements
// ---------------------------------------------------------------------------

#[test]
fn bvh_query_point() {
    // Build a mesh with two triangles sharing an edge
    let mesh = Mesh {
        nodes: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(3.0, 2.0, 0.0),
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

    let bvh = BVH::build(&mesh);
    assert!(!bvh.is_empty());

    // Query a point inside the first triangle
    let inside_first = Vec3::new(0.8, 0.5, 0.0);
    let results = bvh.query_point(&inside_first);
    assert!(
        !results.is_empty(),
        "should find at least one element containing (0.8, 0.5, 0.0)",
    );
    // The result should include element 0
    assert!(
        results.contains(&0),
        "should find element 0, got {:?}",
        results,
    );

    // Query a point inside the second triangle (far right)
    let inside_second = Vec3::new(2.5, 1.5, 0.0);
    let results2 = bvh.query_point(&inside_second);
    assert!(
        !results2.is_empty(),
        "should find element near (2.5, 1.5, 0.0)",
    );
    assert!(
        results2.contains(&1),
        "should find element 1, got {:?}",
        results2,
    );

    // Query a point far outside the mesh
    let outside = Vec3::new(100.0, 100.0, 100.0);
    let results3 = bvh.query_point(&outside);
    assert!(
        results3.is_empty(),
        "point far outside should return no elements, got {:?}",
        results3,
    );
}

// ---------------------------------------------------------------------------
// 4. Mesh round-trip: load .msh -> save -> reload -> compare
// ---------------------------------------------------------------------------

#[test]
fn mesh_round_trip_msh() {
    let path = fixtures_dir().join("sample.msh");
    let original = GmshLoader::load(&path).expect("should load original");

    // Save to a temporary file
    let tmp_dir = std::env::temp_dir().join("simucad_mesh_roundtrip_test");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let tmp_path = tmp_dir.join("roundtrip.msh");

    GmshLoader::save(&original, &tmp_path).expect("should save mesh");

    // Reload the saved file
    let reloaded = GmshLoader::load(&tmp_path).expect("should reload mesh");

    // Compare structure
    assert_eq!(original.node_count(), reloaded.node_count());
    assert_eq!(original.element_count(), reloaded.element_count());
    assert_eq!(original.dimension, reloaded.dimension);

    // Compare nodes (within tolerance for floating-point round-trip)
    for (i, (orig, reload)) in original.nodes.iter().zip(reloaded.nodes.iter()).enumerate() {
        assert!(
            (orig.x - reload.x).abs() < 1e-6
                && (orig.y - reload.y).abs() < 1e-6
                && (orig.z - reload.z).abs() < 1e-6,
            "node {} mismatch: {:?} vs {:?}",
            i,
            orig,
            reload,
        );
    }

    // Compare element types and connectivity
    for (i, (orig, reload)) in original
        .elements
        .iter()
        .zip(reloaded.elements.iter())
        .enumerate()
    {
        assert_eq!(
            orig.element_type, reload.element_type,
            "element {} type mismatch",
            i,
        );
        assert_eq!(
            orig.node_indices, reload.node_indices,
            "element {} connectivity mismatch",
            i,
        );
    }

    // Quality should be identical
    let orig_quality = compute_mesh_quality(&original).unwrap();
    let reload_quality = compute_mesh_quality(&reloaded).unwrap();
    assert!(
        (orig_quality.total_area_or_volume - reload_quality.total_area_or_volume).abs() < 1e-6,
        "total area/volume should match after round-trip",
    );
    assert!(
        (orig_quality.mean_aspect_ratio - reload_quality.mean_aspect_ratio).abs() < 1e-6,
        "mean aspect ratio should match after round-trip",
    );

    // Clean up
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn parse_msh_string_directly_and_validate() {
    let msh_content = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 0.0 0.0 1.0
$EndNodes
$Elements
1
1 4 0 1 2 3 4
$EndElements
";

    let mesh = parse_msh(msh_content).expect("should parse inline MSH");
    assert_eq!(mesh.node_count(), 4);
    assert_eq!(mesh.element_count(), 1);
    assert_eq!(mesh.elements[0].element_type, ElementType::Tetrahedron4);
    assert_eq!(mesh.dimension, 3);

    validate_mesh(&mesh).expect("tet mesh should be valid");

    let report = compute_mesh_quality(&mesh).unwrap();
    assert_eq!(report.degenerate_count, 0);
    assert!(report.total_area_or_volume > 0.0);
}
