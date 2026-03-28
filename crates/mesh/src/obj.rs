use std::fs;
use std::io::Write;
use std::path::Path;

use simucad_core::error::MeshError;
use simucad_core::types::Vec3;
use tracing::{info, warn};

use crate::types::{ElementType, Mesh, MeshElement};

// ---------------------------------------------------------------------------
// OBJ parser
// ---------------------------------------------------------------------------

/// Parse a Wavefront OBJ string into a [`Mesh`].
///
/// Supports: `v x y z` vertex lines, `f i j k` triangular faces (1-based).
/// Quads `f i j k l` are split into two triangles.
/// Handles `v/vt/vn` face format (takes first component).
/// Supports negative indices (relative to current vertex count).
/// Ignores: vn, vt, g, o, s, mtllib, usemtl, # comments.
pub fn parse_obj(input: &str) -> Result<Mesh, MeshError> {
    let mut nodes: Vec<Vec3> = Vec::new();
    let mut elements: Vec<MeshElement> = Vec::new();

    for (line_num, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let mut tokens = line.split_whitespace();
        let Some(keyword) = tokens.next() else {
            continue;
        };

        match keyword {
            "v" => {
                let coords: Vec<&str> = tokens.collect();
                if coords.len() < 3 {
                    return Err(MeshError::ParseError(format!(
                        "Line {}: vertex needs at least 3 coordinates, got {}",
                        line_num + 1,
                        coords.len()
                    )));
                }
                let x: f64 = coords[0].parse().map_err(|e| {
                    MeshError::ParseError(format!("Line {}: bad vertex x: {e}", line_num + 1))
                })?;
                let y: f64 = coords[1].parse().map_err(|e| {
                    MeshError::ParseError(format!("Line {}: bad vertex y: {e}", line_num + 1))
                })?;
                let z: f64 = coords[2].parse().map_err(|e| {
                    MeshError::ParseError(format!("Line {}: bad vertex z: {e}", line_num + 1))
                })?;
                nodes.push(Vec3::new(x, y, z));
            }
            "f" => {
                let raw_indices: Vec<&str> = tokens.collect();
                if raw_indices.len() < 3 {
                    warn!(
                        "Line {}: face has fewer than 3 vertices, skipping",
                        line_num + 1
                    );
                    continue;
                }

                // Parse each face component: could be "idx", "idx/vt", "idx/vt/vn", or "idx//vn"
                let mut indices: Vec<usize> = Vec::with_capacity(raw_indices.len());
                for component in &raw_indices {
                    let idx_str = component.split('/').next().unwrap_or("");
                    let idx: isize = idx_str.parse().map_err(|e| {
                        MeshError::ParseError(format!(
                            "Line {}: bad face index '{component}': {e}",
                            line_num + 1
                        ))
                    })?;

                    let resolved = if idx > 0 {
                        (idx - 1) as usize
                    } else if idx < 0 {
                        let abs_idx = (-idx) as usize;
                        if abs_idx > nodes.len() {
                            return Err(MeshError::ParseError(format!(
                                "Line {}: negative index {idx} exceeds vertex count {}",
                                line_num + 1,
                                nodes.len()
                            )));
                        }
                        nodes.len() - abs_idx
                    } else {
                        return Err(MeshError::ParseError(format!(
                            "Line {}: face index 0 is invalid in OBJ format",
                            line_num + 1
                        )));
                    };

                    if resolved >= nodes.len() {
                        return Err(MeshError::NodeIndexOutOfBounds {
                            index: resolved,
                            node_count: nodes.len(),
                        });
                    }
                    indices.push(resolved);
                }

                // Fan triangulation from the first vertex
                for i in 1..indices.len() - 1 {
                    elements.push(MeshElement {
                        element_type: ElementType::Triangle3,
                        node_indices: vec![indices[0], indices[i], indices[i + 1]],
                    });
                }
            }
            // Skip everything else: comments, vn, vt, g, o, s, mtllib, usemtl, etc.
            _ => {}
        }
    }

    if nodes.is_empty() {
        return Err(MeshError::EmptyMesh);
    }
    if elements.is_empty() {
        return Err(MeshError::ParseError(
            "OBJ contains vertices but no faces".into(),
        ));
    }

    info!(
        "Parsed OBJ: {} nodes, {} triangles",
        nodes.len(),
        elements.len()
    );

    Ok(Mesh {
        nodes,
        elements,
        dimension: 3,
    })
}

// ---------------------------------------------------------------------------
// OBJ writer
// ---------------------------------------------------------------------------

/// Write a mesh as Wavefront OBJ format.
///
/// Only [`ElementType::Triangle3`] elements are written; other element types
/// are skipped with a warning.
pub fn write_obj(mesh: &Mesh, path: &Path) -> Result<(), MeshError> {
    info!("Writing OBJ to {}", path.display());

    let mut file = fs::File::create(path)?;
    writeln!(file, "# SimuCAD OBJ export")?;
    writeln!(file)?;

    // Write vertices
    for node in &mesh.nodes {
        writeln!(file, "v {} {} {}", node.x, node.y, node.z)?;
    }

    writeln!(file)?;

    // Write faces (1-based indices)
    for elem in &mesh.elements {
        if elem.element_type != ElementType::Triangle3 {
            warn!(
                "Skipping non-triangle element ({:?}) in OBJ export",
                elem.element_type
            );
            continue;
        }
        writeln!(
            file,
            "f {} {} {}",
            elem.node_indices[0] + 1,
            elem.node_indices[1] + 1,
            elem.node_indices[2] + 1,
        )?;
    }

    info!("OBJ written successfully");
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_triangle() {
        let obj = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
";
        let mesh = parse_obj(obj).expect("should parse");
        assert_eq!(mesh.nodes.len(), 3);
        assert_eq!(mesh.elements.len(), 1);
        assert_eq!(mesh.elements[0].element_type, ElementType::Triangle3);
        assert_eq!(mesh.elements[0].node_indices, vec![0, 1, 2]);
        assert_eq!(mesh.dimension, 3);
    }

    #[test]
    fn parse_cube_obj() {
        // A unit cube: 8 vertices, 6 quad faces -> 12 triangles
        let obj = "\
# Unit cube
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1

f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
";
        let mesh = parse_obj(obj).expect("should parse cube");
        assert_eq!(mesh.nodes.len(), 8);
        assert_eq!(mesh.elements.len(), 12); // 6 quads -> 12 triangles

        for elem in &mesh.elements {
            assert_eq!(elem.element_type, ElementType::Triangle3);
            assert_eq!(elem.node_indices.len(), 3);
            for &idx in &elem.node_indices {
                assert!(idx < 8, "index {idx} out of bounds");
            }
        }
    }

    #[test]
    fn parse_negative_indices() {
        let obj = "\
v 0 0 0
v 1 0 0
v 0 1 0
f -3 -2 -1
";
        let mesh = parse_obj(obj).expect("should parse negative indices");
        assert_eq!(mesh.nodes.len(), 3);
        assert_eq!(mesh.elements.len(), 1);
        assert_eq!(mesh.elements[0].node_indices, vec![0, 1, 2]);
    }

    #[test]
    fn parse_vertex_texture_normal_format() {
        let obj = "\
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
vn 0 0 1
vn 0 0 1
f 1/1/1 2/2/2 3/3/3
";
        let mesh = parse_obj(obj).expect("should parse v/vt/vn format");
        assert_eq!(mesh.nodes.len(), 3);
        assert_eq!(mesh.elements.len(), 1);
        assert_eq!(mesh.elements[0].node_indices, vec![0, 1, 2]);
    }

    #[test]
    fn round_trip_obj() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
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
            dimension: 3,
        };

        let dir = std::env::temp_dir().join("simucad_obj_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("round_trip.obj");

        write_obj(&mesh, &path).expect("should write OBJ");

        let contents = fs::read_to_string(&path).expect("should read back");
        let reloaded = parse_obj(&contents).expect("should re-parse");

        assert_eq!(reloaded.nodes.len(), mesh.nodes.len());
        assert_eq!(reloaded.elements.len(), mesh.elements.len());
        assert_eq!(reloaded.dimension, 3);

        // Verify vertex coordinates survived the round trip
        for (orig, loaded) in mesh.nodes.iter().zip(reloaded.nodes.iter()) {
            assert!((orig.x - loaded.x).abs() < 1e-12);
            assert!((orig.y - loaded.y).abs() < 1e-12);
            assert!((orig.z - loaded.z).abs() < 1e-12);
        }

        // Verify element indices survived
        for (orig, loaded) in mesh.elements.iter().zip(reloaded.elements.iter()) {
            assert_eq!(orig.node_indices, loaded.node_indices);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_obj_fails() {
        let obj = "# just a comment\n";
        let result = parse_obj(obj);
        assert!(result.is_err());

        // Also test with no content at all
        let result2 = parse_obj("");
        assert!(result2.is_err());
    }
}
