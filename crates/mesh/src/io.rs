use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use simucad_core::error::MeshError;
use tracing::{debug, info};

use crate::parser::parse_msh;
use crate::types::Mesh;
use crate::validation::validate_mesh;

// ---------------------------------------------------------------------------
// MeshLoader trait
// ---------------------------------------------------------------------------

/// Trait for loading and saving meshes from/to files.
pub trait MeshLoader {
    /// Read a mesh from the file at `path`.
    fn load(path: &Path) -> Result<Mesh, MeshError>;

    /// Write `mesh` to the file at `path`.
    fn save(mesh: &Mesh, path: &Path) -> Result<(), MeshError>;
}

// ---------------------------------------------------------------------------
// GmshLoader — Gmsh .msh v2.2 ASCII
// ---------------------------------------------------------------------------

/// Loader for Gmsh `.msh` v2.2 ASCII files.
pub struct GmshLoader;

impl MeshLoader for GmshLoader {
    fn load(path: &Path) -> Result<Mesh, MeshError> {
        info!("Loading Gmsh mesh from {}", path.display());

        let contents = fs::read_to_string(path)?;
        let mesh = parse_msh(&contents)?;

        debug!(
            "Parsed mesh: {} nodes, {} elements, {}D",
            mesh.node_count(),
            mesh.element_count(),
            mesh.dimension
        );

        validate_mesh(&mesh)?;
        info!("Mesh validated successfully");

        Ok(mesh)
    }

    fn save(mesh: &Mesh, path: &Path) -> Result<(), MeshError> {
        info!("Saving Gmsh mesh to {}", path.display());

        let mut out = String::new();

        // $MeshFormat
        writeln!(out, "$MeshFormat").unwrap();
        writeln!(out, "2.2 0 8").unwrap();
        writeln!(out, "$EndMeshFormat").unwrap();

        // $Nodes — use 1-based IDs
        writeln!(out, "$Nodes").unwrap();
        writeln!(out, "{}", mesh.nodes.len()).unwrap();
        for (i, node) in mesh.nodes.iter().enumerate() {
            writeln!(out, "{} {} {} {}", i + 1, node.x, node.y, node.z).unwrap();
        }
        writeln!(out, "$EndNodes").unwrap();

        // $Elements — use 1-based IDs, 0 tags
        writeln!(out, "$Elements").unwrap();
        writeln!(out, "{}", mesh.elements.len()).unwrap();
        for (i, elem) in mesh.elements.iter().enumerate() {
            let gmsh_type = elem.element_type.to_gmsh_code();
            // Write: elem-id  elem-type  num-tags  node-ids...
            // We write 0 tags for simplicity.
            let mut line = format!("{} {} 0", i + 1, gmsh_type);
            for &idx in &elem.node_indices {
                write!(line, " {}", idx + 1).unwrap(); // convert back to 1-based
            }
            writeln!(out, "{}", line).unwrap();
        }
        writeln!(out, "$EndElements").unwrap();

        fs::write(path, out)?;
        info!("Mesh saved ({} bytes)", path.metadata()?.len());

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ElementType, MeshElement};
    use simucad_core::types::Vec3;
    use std::path::PathBuf;

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
    fn round_trip_save_load() {
        let mesh = sample_mesh();
        let dir = std::env::temp_dir().join("simucad_mesh_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("round_trip.msh");

        GmshLoader::save(&mesh, &path).expect("save should succeed");
        let loaded = GmshLoader::load(&path).expect("load should succeed");

        assert_eq!(loaded.node_count(), mesh.node_count());
        assert_eq!(loaded.element_count(), mesh.element_count());
        assert_eq!(loaded.dimension, mesh.dimension);

        // Compare node coordinates
        for (a, b) in loaded.nodes.iter().zip(mesh.nodes.iter()) {
            assert!((a.x - b.x).abs() < 1e-12);
            assert!((a.y - b.y).abs() < 1e-12);
            assert!((a.z - b.z).abs() < 1e-12);
        }

        // Compare element connectivity
        for (a, b) in loaded.elements.iter().zip(mesh.elements.iter()) {
            assert_eq!(a.element_type, b.element_type);
            assert_eq!(a.node_indices, b.node_indices);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_fixture() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_fixtures")
            .join("sample.msh");
        if fixture.exists() {
            let mesh = GmshLoader::load(&fixture).expect("should load fixture");
            assert_eq!(mesh.node_count(), 4);
            assert_eq!(mesh.element_count(), 2);
        }
    }

    #[test]
    fn load_nonexistent_file_fails() {
        let result = GmshLoader::load(Path::new("/tmp/nonexistent_simucad_test.msh"));
        assert!(result.is_err());
    }
}
