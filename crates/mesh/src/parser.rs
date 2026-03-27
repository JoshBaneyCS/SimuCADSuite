use nom::{
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{line_ending, multispace0, space0, space1},
    combinator::{map_res, opt},
    number::complete::double,
    IResult,
};
use simucad_core::error::MeshError;
use simucad_core::types::Vec3;
use tracing::debug;

use crate::types::{ElementType, Mesh, MeshElement};

// ---------------------------------------------------------------------------
// Low-level nom helpers
// ---------------------------------------------------------------------------

/// Consume optional whitespace including newlines.
fn ws(input: &str) -> IResult<&str, &str> {
    multispace0(input)
}

/// Parse an unsigned integer from decimal digits.
fn parse_usize(input: &str) -> IResult<&str, usize> {
    map_res(take_while1(|c: char| c.is_ascii_digit()), |s: &str| {
        s.parse::<usize>()
    })(input)
}

/// Parse an unsigned 32-bit integer.
fn parse_u32(input: &str) -> IResult<&str, u32> {
    map_res(take_while1(|c: char| c.is_ascii_digit()), |s: &str| {
        s.parse::<u32>()
    })(input)
}

/// Consume characters until end-of-line (but not the newline itself).
fn rest_of_line(input: &str) -> IResult<&str, &str> {
    take_while(|c: char| c != '\n' && c != '\r')(input)
}

/// Consume a single line ending (\n or \r\n).
fn eol(input: &str) -> IResult<&str, &str> {
    line_ending(input)
}

/// Consume optional trailing whitespace + a line ending.
fn line_end(input: &str) -> IResult<&str, ()> {
    let (input, _) = space0(input)?;
    let (input, _) = eol(input)?;
    Ok((input, ()))
}

// ---------------------------------------------------------------------------
// $MeshFormat section
// ---------------------------------------------------------------------------

fn parse_mesh_format_section(input: &str) -> IResult<&str, ()> {
    let (input, _) = ws(input)?;
    let (input, _) = tag("$MeshFormat")(input)?;
    let (input, _) = line_end(input)?;

    // version file-type data-size  (e.g. "2.2 0 8")
    let (input, _) = rest_of_line(input)?;
    let (input, _) = eol(input)?;

    let (input, _) = ws(input)?;
    let (input, _) = tag("$EndMeshFormat")(input)?;
    let (input, _) = opt(line_end)(input)?;

    Ok((input, ()))
}

// ---------------------------------------------------------------------------
// $Nodes section
// ---------------------------------------------------------------------------

/// Parse a single node line: "id x y z"
fn parse_node_line(input: &str) -> IResult<&str, (usize, Vec3)> {
    let (input, _) = space0(input)?;
    let (input, id) = parse_usize(input)?;
    let (input, _) = space1(input)?;
    let (input, x) = double(input)?;
    let (input, _) = space1(input)?;
    let (input, y) = double(input)?;
    let (input, _) = space1(input)?;
    let (input, z) = double(input)?;
    let (input, _) = opt(line_end)(input)?;
    Ok((input, (id, Vec3::new(x, y, z))))
}

fn parse_nodes_section(input: &str) -> IResult<&str, Vec<(usize, Vec3)>> {
    let (input, _) = ws(input)?;
    let (input, _) = tag("$Nodes")(input)?;
    let (input, _) = line_end(input)?;

    // Node count
    let (input, _) = space0(input)?;
    let (input, count) = parse_usize(input)?;
    let (input, _) = line_end(input)?;

    let mut nodes = Vec::with_capacity(count);
    let mut remaining = input;
    for _ in 0..count {
        let (rest, node) = parse_node_line(remaining)?;
        nodes.push(node);
        remaining = rest;
    }

    let (remaining, _) = ws(remaining)?;
    let (remaining, _) = tag("$EndNodes")(remaining)?;
    let (remaining, _) = opt(line_end)(remaining)?;

    Ok((remaining, nodes))
}

// ---------------------------------------------------------------------------
// $Elements section
// ---------------------------------------------------------------------------

/// Parse a single element line:
/// "elem-id  elem-type  num-tags  tag1 tag2 ...  node1 node2 ..."
fn parse_element_line(input: &str) -> IResult<&str, Option<(u32, Vec<usize>)>> {
    let (input, _) = space0(input)?;
    let (input, _elem_id) = parse_usize(input)?;
    let (input, _) = space1(input)?;
    let (input, elem_type) = parse_u32(input)?;
    let (input, _) = space1(input)?;
    let (input, num_tags) = parse_usize(input)?;

    // Skip tags
    let mut remaining = input;
    for _ in 0..num_tags {
        let (r, _) = space1(remaining)?;
        let (r, _) = take_while1(|c: char| c.is_ascii_digit() || c == '-')(r)?;
        remaining = r;
    }

    // Determine how many nodes to read from the element type
    let expected_nodes = match ElementType::from_gmsh_code(elem_type) {
        Some(et) => et.node_count(),
        None => {
            // Unknown element type -- skip rest of line
            let (remaining, _) = rest_of_line(remaining)?;
            let (remaining, _) = opt(eol)(remaining)?;
            return Ok((remaining, None));
        }
    };

    let mut node_ids = Vec::with_capacity(expected_nodes);
    for _ in 0..expected_nodes {
        let (r, _) = space1(remaining)?;
        let (r, nid) = parse_usize(r)?;
        node_ids.push(nid);
        remaining = r;
    }

    let (remaining, _) = opt(line_end)(remaining)?;
    Ok((remaining, Some((elem_type, node_ids))))
}

fn parse_elements_section(input: &str) -> IResult<&str, Vec<(u32, Vec<usize>)>> {
    let (input, _) = ws(input)?;
    let (input, _) = tag("$Elements")(input)?;
    let (input, _) = line_end(input)?;

    let (input, _) = space0(input)?;
    let (input, count) = parse_usize(input)?;
    let (input, _) = line_end(input)?;

    let mut elements = Vec::with_capacity(count);
    let mut remaining = input;
    for _ in 0..count {
        let (rest, maybe_elem) = parse_element_line(remaining)?;
        if let Some(elem) = maybe_elem {
            elements.push(elem);
        }
        remaining = rest;
    }

    let (remaining, _) = ws(remaining)?;
    let (remaining, _) = tag("$EndElements")(remaining)?;
    let (remaining, _) = opt(line_end)(remaining)?;

    Ok((remaining, elements))
}

// ---------------------------------------------------------------------------
// Skip an unrecognised section ($PhysicalNames, $Comment, etc.)
// ---------------------------------------------------------------------------

fn skip_unknown_section(input: &str) -> IResult<&str, ()> {
    let (input, _) = ws(input)?;
    let (input, _) = tag("$")(input)?;
    let (input, section_name) = take_while1(|c: char| c.is_alphanumeric())(input)?;
    let (input, _) = opt(line_end)(input)?;

    let end_tag = format!("$End{}", section_name);
    // Scan forward for the closing tag.
    if let Some(pos) = input.find(&end_tag) {
        let after = &input[pos + end_tag.len()..];
        let (after, _) = opt(line_end)(after)?;
        Ok((after, ()))
    } else {
        // If we cannot find the end tag, just bail with an error.
        Err(nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )))
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a Gmsh `.msh` v2.2 ASCII file from a string and return a [`Mesh`].
pub fn parse_msh(input: &str) -> Result<Mesh, MeshError> {
    let mut remaining = input;

    // Parse required $MeshFormat first
    let (r, _) = parse_mesh_format_section(remaining)
        .map_err(|e| MeshError::ParseError(format!("Failed to parse $MeshFormat: {e}")))?;
    remaining = r;

    let mut raw_nodes: Option<Vec<(usize, Vec3)>> = None;
    let mut raw_elements: Option<Vec<(u32, Vec<usize>)>> = None;

    // Parse remaining sections in any order
    loop {
        // Skip whitespace
        let trimmed = remaining.trim_start();
        if trimmed.is_empty() {
            break;
        }
        remaining = trimmed;

        // Peek at the section name
        if remaining.starts_with("$Nodes") {
            let (r, nodes) = parse_nodes_section(remaining)
                .map_err(|e| MeshError::ParseError(format!("Failed to parse $Nodes: {e}")))?;
            raw_nodes = Some(nodes);
            remaining = r;
        } else if remaining.starts_with("$Elements") {
            let (r, elems) = parse_elements_section(remaining)
                .map_err(|e| MeshError::ParseError(format!("Failed to parse $Elements: {e}")))?;
            raw_elements = Some(elems);
            remaining = r;
        } else if remaining.starts_with('$') {
            let (r, _) = skip_unknown_section(remaining).map_err(|e| {
                MeshError::ParseError(format!("Failed to skip unknown section: {e}"))
            })?;
            remaining = r;
        } else {
            break;
        }
    }

    // Build node array -- Gmsh node IDs are 1-based. We store them in order
    // and build a mapping from Gmsh ID -> 0-based index.
    let raw_nodes =
        raw_nodes.ok_or_else(|| MeshError::ParseError("Missing $Nodes section".into()))?;
    let raw_elements =
        raw_elements.ok_or_else(|| MeshError::ParseError("Missing $Elements section".into()))?;

    // Determine max node ID so we can map Gmsh IDs to 0-based indices.
    let max_id = raw_nodes.iter().map(|(id, _)| *id).max().unwrap_or(0);
    let mut id_to_index = vec![usize::MAX; max_id + 1];
    let mut nodes = Vec::with_capacity(raw_nodes.len());
    for (id, coord) in &raw_nodes {
        id_to_index[*id] = nodes.len();
        nodes.push(*coord);
    }

    debug!("Parsed {} nodes from .msh file", nodes.len());

    // Convert raw elements to MeshElements with 0-based indices.
    let mut elements = Vec::with_capacity(raw_elements.len());
    for (gmsh_type, gmsh_node_ids) in &raw_elements {
        let element_type = ElementType::from_gmsh_code(*gmsh_type).ok_or_else(|| {
            MeshError::UnsupportedFormat(format!("Unsupported Gmsh element type: {gmsh_type}"))
        })?;

        let node_indices: Vec<usize> = gmsh_node_ids
            .iter()
            .map(|&gid| {
                if gid > max_id || id_to_index[gid] == usize::MAX {
                    Err(MeshError::NodeIndexOutOfBounds {
                        index: gid,
                        node_count: nodes.len(),
                    })
                } else {
                    Ok(id_to_index[gid])
                }
            })
            .collect::<Result<_, _>>()?;

        elements.push(MeshElement {
            element_type,
            node_indices,
        });
    }

    debug!("Parsed {} elements from .msh file", elements.len());

    let dimension = detect_dimension_from_elements(&elements);

    Ok(Mesh {
        nodes,
        elements,
        dimension,
    })
}

/// Infer dimension from the highest-dimensional element type present.
fn detect_dimension_from_elements(elements: &[MeshElement]) -> u8 {
    elements
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

    const SAMPLE_MSH: &str = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.5 1.0 0.0
4 1.5 1.0 0.0
$EndNodes
$Elements
2
1 2 2 0 1 1 2 3
2 2 2 0 1 2 4 3
$EndElements
";

    #[test]
    fn parse_sample_mesh() {
        let mesh = parse_msh(SAMPLE_MSH).expect("should parse sample msh");
        assert_eq!(mesh.node_count(), 4);
        assert_eq!(mesh.element_count(), 2);
        assert_eq!(mesh.dimension, 2);

        // Verify first node
        assert_eq!(mesh.nodes[0], Vec3::new(0.0, 0.0, 0.0));
        // Verify second node
        assert_eq!(mesh.nodes[1], Vec3::new(1.0, 0.0, 0.0));

        // Verify element types
        assert_eq!(mesh.elements[0].element_type, ElementType::Triangle3);
        assert_eq!(mesh.elements[1].element_type, ElementType::Triangle3);

        // Verify 0-based indices
        assert_eq!(mesh.elements[0].node_indices, vec![0, 1, 2]);
        assert_eq!(mesh.elements[1].node_indices, vec![1, 3, 2]);
    }

    #[test]
    fn parse_with_unknown_section() {
        let input = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 \"surface\"
$EndPhysicalNames
$Nodes
3
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.5 1.0 0.0
$EndNodes
$Elements
1
1 2 2 0 1 1 2 3
$EndElements
";
        let mesh = parse_msh(input).expect("should skip unknown sections");
        assert_eq!(mesh.node_count(), 3);
        assert_eq!(mesh.element_count(), 1);
    }

    #[test]
    fn parse_missing_nodes_fails() {
        let input = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Elements
1
1 2 2 0 1 1 2 3
$EndElements
";
        let result = parse_msh(input);
        assert!(result.is_err());
    }

    #[test]
    fn parse_line_elements() {
        let input = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
2
1 0.0 0.0 0.0
2 1.0 0.0 0.0
$EndNodes
$Elements
1
1 1 2 0 1 1 2
$EndElements
";
        let mesh = parse_msh(input).expect("should parse line elements");
        assert_eq!(mesh.elements[0].element_type, ElementType::Line2);
        assert_eq!(mesh.dimension, 1);
    }

    #[test]
    fn parse_tet_elements() {
        let input = "\
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
1 4 2 0 1 1 2 3 4
$EndElements
";
        let mesh = parse_msh(input).expect("should parse tet elements");
        assert_eq!(mesh.elements[0].element_type, ElementType::Tetrahedron4);
        assert_eq!(mesh.dimension, 3);
    }
}
