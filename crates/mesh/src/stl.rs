use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use simucad_core::error::MeshError;
use simucad_core::types::Vec3;
use tracing::{debug, info};

use crate::types::{ElementType, Mesh, MeshElement};

/// Vertex merge tolerance for deduplication.
const MERGE_EPSILON: f64 = 1e-9;

// ---------------------------------------------------------------------------
// STL format detection
// ---------------------------------------------------------------------------

/// Detected STL format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StlFormat {
    Ascii,
    Binary,
}

/// Heuristic to detect whether raw bytes represent ASCII or binary STL.
///
/// ASCII STL begins with "solid " (optionally followed by a name). However,
/// some binary files also start with "solid" in their 80-byte header, so we
/// check further: if the file also contains "facet" within the first ~200
/// bytes, it is likely ASCII.
pub fn detect_stl_format(data: &[u8]) -> StlFormat {
    if data.len() < 84 {
        // Too short for a valid binary STL; assume ASCII.
        return StlFormat::Ascii;
    }

    // Check if it starts with "solid"
    if data.starts_with(b"solid") {
        // Look for "facet" or "endsolid" in the first 300 bytes to confirm ASCII.
        let check_len = data.len().min(300);
        let prefix = String::from_utf8_lossy(&data[..check_len]);
        if prefix.contains("facet") || prefix.contains("endsolid") {
            return StlFormat::Ascii;
        }
    }

    StlFormat::Binary
}

// ---------------------------------------------------------------------------
// Vertex deduplication helper
// ---------------------------------------------------------------------------

/// Quantise a coordinate to an integer key for hash-based dedup.
fn quantise(v: f64) -> i64 {
    (v / MERGE_EPSILON).round() as i64
}

fn vertex_key(v: &Vec3) -> (i64, i64, i64) {
    (quantise(v.x), quantise(v.y), quantise(v.z))
}

/// Insert a vertex, returning its (possibly deduplicated) zero-based index.
fn insert_or_get(
    nodes: &mut Vec<Vec3>,
    map: &mut HashMap<(i64, i64, i64), usize>,
    v: Vec3,
) -> usize {
    let key = vertex_key(&v);
    if let Some(&idx) = map.get(&key) {
        idx
    } else {
        let idx = nodes.len();
        nodes.push(v);
        map.insert(key, idx);
        idx
    }
}

// ---------------------------------------------------------------------------
// ASCII STL parser
// ---------------------------------------------------------------------------

/// Parse an ASCII STL string into a [`Mesh`].
///
/// The format looks like:
/// ```text
/// solid name
///   facet normal ni nj nk
///     outer loop
///       vertex x y z
///       vertex x y z
///       vertex x y z
///     endloop
///   endfacet
/// endsolid name
/// ```
pub fn parse_stl_ascii(input: &str) -> Result<Mesh, MeshError> {
    let mut nodes: Vec<Vec3> = Vec::new();
    let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut elements: Vec<MeshElement> = Vec::new();

    let mut lines = input.lines().map(str::trim).peekable();

    // Expect "solid ..."
    match lines.next() {
        Some(line) if line.starts_with("solid") => {}
        _ => {
            return Err(MeshError::ParseError(
                "ASCII STL must start with 'solid'".into(),
            ))
        }
    }

    loop {
        // Skip blank lines
        while let Some(&line) = lines.peek() {
            if line.is_empty() {
                lines.next();
            } else {
                break;
            }
        }

        let Some(line) = lines.next() else { break };

        if line.starts_with("endsolid") {
            break;
        }

        // Expect "facet normal ..."
        if !line.starts_with("facet") {
            return Err(MeshError::ParseError(format!(
                "Expected 'facet normal' or 'endsolid', got: '{line}'"
            )));
        }

        // Expect "outer loop"
        let loop_line = lines
            .next()
            .ok_or_else(|| MeshError::ParseError("Unexpected end of STL".into()))?;
        if !loop_line.starts_with("outer loop") {
            return Err(MeshError::ParseError(format!(
                "Expected 'outer loop', got: '{loop_line}'"
            )));
        }

        // Read three vertices
        let mut tri_indices = Vec::with_capacity(3);
        for _ in 0..3 {
            let vline = lines
                .next()
                .ok_or_else(|| MeshError::ParseError("Unexpected end of STL in vertex".into()))?;
            let v = parse_vertex_line(vline)?;
            let idx = insert_or_get(&mut nodes, &mut map, v);
            tri_indices.push(idx);
        }

        // Expect "endloop"
        let endloop = lines
            .next()
            .ok_or_else(|| MeshError::ParseError("Unexpected end of STL, expected endloop".into()))?;
        if !endloop.starts_with("endloop") {
            return Err(MeshError::ParseError(format!(
                "Expected 'endloop', got: '{endloop}'"
            )));
        }

        // Expect "endfacet"
        let endfacet = lines
            .next()
            .ok_or_else(|| MeshError::ParseError("Unexpected end of STL, expected endfacet".into()))?;
        if !endfacet.starts_with("endfacet") {
            return Err(MeshError::ParseError(format!(
                "Expected 'endfacet', got: '{endfacet}'"
            )));
        }

        elements.push(MeshElement {
            element_type: ElementType::Triangle3,
            node_indices: tri_indices,
        });
    }

    debug!(
        "Parsed ASCII STL: {} nodes, {} triangles",
        nodes.len(),
        elements.len()
    );

    Ok(Mesh {
        nodes,
        elements,
        dimension: 2,
    })
}

/// Parse a "vertex x y z" line.
fn parse_vertex_line(line: &str) -> Result<Vec3, MeshError> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 4 || parts[0] != "vertex" {
        return Err(MeshError::ParseError(format!(
            "Invalid vertex line: '{line}'"
        )));
    }
    let x: f64 = parts[1]
        .parse()
        .map_err(|e| MeshError::ParseError(format!("Bad vertex x: {e}")))?;
    let y: f64 = parts[2]
        .parse()
        .map_err(|e| MeshError::ParseError(format!("Bad vertex y: {e}")))?;
    let z: f64 = parts[3]
        .parse()
        .map_err(|e| MeshError::ParseError(format!("Bad vertex z: {e}")))?;
    Ok(Vec3::new(x, y, z))
}

// ---------------------------------------------------------------------------
// Binary STL parser
// ---------------------------------------------------------------------------

/// Parse a binary STL byte slice into a [`Mesh`].
///
/// Binary format:
/// - 80-byte header (ignored)
/// - 4-byte little-endian u32: triangle count
/// - Per triangle (50 bytes):
///   - 12 bytes: normal (3 x f32)
///   - 36 bytes: 3 vertices (3 x 3 x f32)
///   - 2 bytes: attribute byte count (ignored)
pub fn parse_stl_binary(data: &[u8]) -> Result<Mesh, MeshError> {
    if data.len() < 84 {
        return Err(MeshError::ParseError(
            "Binary STL too short (need at least 84 bytes)".into(),
        ));
    }

    let tri_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let expected_len = 84 + tri_count * 50;

    if data.len() < expected_len {
        return Err(MeshError::ParseError(format!(
            "Binary STL truncated: expected {} bytes for {} triangles, got {}",
            expected_len,
            tri_count,
            data.len()
        )));
    }

    let mut nodes: Vec<Vec3> = Vec::new();
    let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut elements: Vec<MeshElement> = Vec::with_capacity(tri_count);

    let mut offset = 84;
    for _ in 0..tri_count {
        // Skip normal (12 bytes)
        offset += 12;

        // Read 3 vertices
        let mut tri_indices = Vec::with_capacity(3);
        for _ in 0..3 {
            let x = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as f64;
            let y = f32::from_le_bytes([
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]) as f64;
            let z = f32::from_le_bytes([
                data[offset + 8],
                data[offset + 9],
                data[offset + 10],
                data[offset + 11],
            ]) as f64;
            offset += 12;

            let v = Vec3::new(x, y, z);
            let idx = insert_or_get(&mut nodes, &mut map, v);
            tri_indices.push(idx);
        }

        // Skip attribute byte count (2 bytes)
        offset += 2;

        elements.push(MeshElement {
            element_type: ElementType::Triangle3,
            node_indices: tri_indices,
        });
    }

    debug!(
        "Parsed binary STL: {} nodes, {} triangles",
        nodes.len(),
        elements.len()
    );

    Ok(Mesh {
        nodes,
        elements,
        dimension: 2,
    })
}

// ---------------------------------------------------------------------------
// STL writer (ASCII)
// ---------------------------------------------------------------------------

/// Write a mesh to an ASCII STL file.
///
/// Only [`ElementType::Triangle3`] elements are written; other element types
/// are silently skipped.
pub fn write_stl_ascii(mesh: &Mesh, path: &Path) -> Result<(), MeshError> {
    info!("Writing ASCII STL to {}", path.display());

    let mut file = fs::File::create(path)?;
    writeln!(file, "solid mesh").map_err(|e| MeshError::Io(e))?;

    for elem in &mesh.elements {
        if elem.element_type != ElementType::Triangle3 {
            continue;
        }
        let a = mesh.nodes[elem.node_indices[0]];
        let b = mesh.nodes[elem.node_indices[1]];
        let c = mesh.nodes[elem.node_indices[2]];

        // Compute face normal
        let ab = b - a;
        let ac = c - a;
        let n = ab.cross(&ac).normalized();

        writeln!(file, "  facet normal {} {} {}", n.x, n.y, n.z)
            .map_err(|e| MeshError::Io(e))?;
        writeln!(file, "    outer loop").map_err(|e| MeshError::Io(e))?;
        writeln!(file, "      vertex {} {} {}", a.x, a.y, a.z)
            .map_err(|e| MeshError::Io(e))?;
        writeln!(file, "      vertex {} {} {}", b.x, b.y, b.z)
            .map_err(|e| MeshError::Io(e))?;
        writeln!(file, "      vertex {} {} {}", c.x, c.y, c.z)
            .map_err(|e| MeshError::Io(e))?;
        writeln!(file, "    endloop").map_err(|e| MeshError::Io(e))?;
        writeln!(file, "  endfacet").map_err(|e| MeshError::Io(e))?;
    }

    writeln!(file, "endsolid mesh").map_err(|e| MeshError::Io(e))?;

    info!("STL written successfully");
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_ASCII_STL: &str = "\
solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 1 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
";

    #[test]
    fn parse_ascii_stl_basic() {
        let mesh = parse_stl_ascii(SAMPLE_ASCII_STL).expect("should parse");
        // 4 unique vertices (shared edge), 2 triangles
        assert_eq!(mesh.elements.len(), 2);
        assert_eq!(mesh.nodes.len(), 4);

        for elem in &mesh.elements {
            assert_eq!(elem.element_type, ElementType::Triangle3);
            assert_eq!(elem.node_indices.len(), 3);
        }
    }

    #[test]
    fn parse_ascii_stl_missing_solid_fails() {
        let bad = "facet normal 0 0 1\n  outer loop\n  endloop\nendfacet\n";
        assert!(parse_stl_ascii(bad).is_err());
    }

    #[test]
    fn detect_ascii_format() {
        let data = SAMPLE_ASCII_STL.as_bytes();
        assert_eq!(detect_stl_format(data), StlFormat::Ascii);
    }

    #[test]
    fn detect_binary_format() {
        // Build a minimal binary STL with 0 triangles.
        let mut data = vec![0u8; 84];
        // Header: 80 bytes (fill with non-"solid" content to be unambiguous)
        data[0] = b'X';
        // Triangle count = 0
        data[80] = 0;
        data[81] = 0;
        data[82] = 0;
        data[83] = 0;

        assert_eq!(detect_stl_format(&data), StlFormat::Binary);
    }

    #[test]
    fn parse_binary_stl_basic() {
        // Build a binary STL with 1 triangle.
        let mut data = vec![0u8; 80]; // header
        // Triangle count = 1
        data.extend_from_slice(&1u32.to_le_bytes());

        // Normal (0, 0, 1)
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());

        // Vertex 1: (0, 0, 0)
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());

        // Vertex 2: (1, 0, 0)
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());

        // Vertex 3: (0, 1, 0)
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());

        // Attribute byte count
        data.extend_from_slice(&0u16.to_le_bytes());

        let mesh = parse_stl_binary(&data).expect("should parse binary STL");
        assert_eq!(mesh.elements.len(), 1);
        assert_eq!(mesh.nodes.len(), 3);
    }

    #[test]
    fn parse_binary_stl_truncated_fails() {
        let data = vec![0u8; 50]; // too short
        assert!(parse_stl_binary(&data).is_err());
    }

    #[test]
    fn binary_stl_vertex_dedup() {
        // Two triangles sharing an edge -> 4 unique vertices instead of 6.
        let mut data = vec![0u8; 80]; // header
        data.extend_from_slice(&2u32.to_le_bytes());

        // Triangle 1: (0,0,0) (1,0,0) (0,1,0)
        // Normal
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        // Vertices
        for v in &[(0.0f32, 0.0f32, 0.0f32), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)] {
            data.extend_from_slice(&v.0.to_le_bytes());
            data.extend_from_slice(&v.1.to_le_bytes());
            data.extend_from_slice(&v.2.to_le_bytes());
        }
        data.extend_from_slice(&0u16.to_le_bytes());

        // Triangle 2: (1,0,0) (1,1,0) (0,1,0)
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        for v in &[(1.0f32, 0.0f32, 0.0f32), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)] {
            data.extend_from_slice(&v.0.to_le_bytes());
            data.extend_from_slice(&v.1.to_le_bytes());
            data.extend_from_slice(&v.2.to_le_bytes());
        }
        data.extend_from_slice(&0u16.to_le_bytes());

        let mesh = parse_stl_binary(&data).expect("should parse");
        assert_eq!(mesh.elements.len(), 2);
        assert_eq!(mesh.nodes.len(), 4); // shared vertices deduplicated
    }

    #[test]
    fn write_and_reparse_ascii_stl() {
        let mesh = Mesh {
            nodes: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            elements: vec![MeshElement {
                element_type: ElementType::Triangle3,
                node_indices: vec![0, 1, 2],
            }],
            dimension: 2,
        };

        let dir = std::env::temp_dir().join("simucad_stl_test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_output.stl");

        write_stl_ascii(&mesh, &path).expect("should write STL");

        let contents = fs::read_to_string(&path).expect("should read back");
        let reloaded = parse_stl_ascii(&contents).expect("should re-parse");

        assert_eq!(reloaded.elements.len(), 1);
        assert_eq!(reloaded.nodes.len(), 3);

        let _ = fs::remove_dir_all(&dir);
    }
}
