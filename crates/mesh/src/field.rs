//! Scalar and vector field types for FEM-style visualization.
//!
//! Provides per-node field data, colormap functions, and utilities for
//! interpolating and visualizing simulation results on a mesh.

use serde::{Deserialize, Serialize};
use simucad_core::types::Vec3;

use crate::types::Mesh;

// ---------------------------------------------------------------------------
// MeshScalarField
// ---------------------------------------------------------------------------

/// A per-node scalar field defined over a mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshScalarField {
    /// One scalar value per mesh node.
    pub values: Vec<f64>,
}

/// A per-node vector field defined over a mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshVectorField {
    /// One vector value per mesh node.
    pub values: Vec<Vec3>,
}

// ---------------------------------------------------------------------------
// MeshScalarField impl
// ---------------------------------------------------------------------------

impl MeshScalarField {
    /// Create a scalar field from pre-computed per-node values.
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Compute a scalar field by evaluating `f` at every mesh node.
    pub fn from_function(mesh: &Mesh, f: impl Fn(&Vec3) -> f64) -> Self {
        let values = mesh.nodes.iter().map(|node| f(node)).collect();
        Self { values }
    }

    /// Return the minimum value in the field, or [`f64::MAX`] if empty.
    pub fn min(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::MAX, f64::min)
    }

    /// Return the maximum value in the field, or [`f64::MIN`] if empty.
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::MIN, f64::max)
    }

    /// Barycentric interpolation of the scalar field within an element.
    ///
    /// `bary_coords` must have one weight per node in the element.  The
    /// result is the weighted sum of the nodal values.
    pub fn interpolate_at_element(
        &self,
        mesh: &Mesh,
        element_idx: usize,
        bary_coords: &[f64],
    ) -> f64 {
        let elem = &mesh.elements[element_idx];
        elem.node_indices
            .iter()
            .zip(bary_coords.iter())
            .map(|(&ni, &w)| self.values[ni] * w)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// MeshVectorField impl
// ---------------------------------------------------------------------------

impl MeshVectorField {
    /// Create a vector field from pre-computed per-node values.
    pub fn new(values: Vec<Vec3>) -> Self {
        Self { values }
    }

    /// Compute a vector field by evaluating `f` at every mesh node.
    pub fn from_function(mesh: &Mesh, f: impl Fn(&Vec3) -> Vec3) -> Self {
        let values = mesh.nodes.iter().map(|node| f(node)).collect();
        Self { values }
    }

    /// Compute the magnitude of each vector, yielding a scalar field.
    ///
    /// Magnitude is computed as `sqrt(x^2 + y^2 + z^2)`.
    pub fn magnitude(&self) -> MeshScalarField {
        let values = self
            .values
            .iter()
            .map(|v| (v.x * v.x + v.y * v.y + v.z * v.z).sqrt())
            .collect();
        MeshScalarField { values }
    }
}

// ---------------------------------------------------------------------------
// Colormaps
// ---------------------------------------------------------------------------

/// Colormap function signature: a normalized parameter in `[0, 1]` mapped to
/// an RGBA color with components in `[0, 1]`.
pub type ColormapFn = fn(f64) -> [f32; 4];

/// Viridis-like colormap (dark purple -> teal -> yellow).
pub fn colormap_viridis_rgba(t: f64) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);

    // Key control points: (parameter, R, G, B)
    const STOPS: &[(f64, f32, f32, f32)] = &[
        (0.00, 0.267, 0.004, 0.329), // dark purple
        (0.25, 0.282, 0.140, 0.458), // purple
        (0.50, 0.127, 0.566, 0.551), // teal
        (0.75, 0.544, 0.774, 0.247), // green-yellow
        (1.00, 0.993, 0.906, 0.144), // yellow
    ];

    lerp_colormap(t, STOPS)
}

/// Cool-warm diverging colormap (blue -> white -> red).
pub fn colormap_coolwarm_rgba(t: f64) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);

    const STOPS: &[(f64, f32, f32, f32)] = &[
        (0.0, 0.230, 0.299, 0.754), // blue
        (0.5, 0.865, 0.865, 0.865), // white-ish
        (1.0, 0.706, 0.016, 0.150), // red
    ];

    lerp_colormap(t, STOPS)
}

/// Jet colormap (blue -> cyan -> green -> yellow -> red).
pub fn colormap_jet_rgba(t: f64) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);

    const STOPS: &[(f64, f32, f32, f32)] = &[
        (0.00, 0.0, 0.0, 0.5),  // dark blue
        (0.12, 0.0, 0.0, 1.0),  // blue
        (0.37, 0.0, 1.0, 1.0),  // cyan
        (0.50, 0.0, 1.0, 0.0),  // green
        (0.63, 1.0, 1.0, 0.0),  // yellow
        (0.88, 1.0, 0.0, 0.0),  // red
        (1.00, 0.5, 0.0, 0.0),  // dark red
    ];

    lerp_colormap(t, STOPS)
}

/// Linearly interpolate between piecewise-linear colour stops.
fn lerp_colormap(t: f64, stops: &[(f64, f32, f32, f32)]) -> [f32; 4] {
    // Find the surrounding stop pair.
    let mut i = 0;
    while i + 1 < stops.len() && stops[i + 1].0 < t {
        i += 1;
    }
    if i + 1 >= stops.len() {
        let s = stops[stops.len() - 1];
        return [s.1, s.2, s.3, 1.0];
    }

    let (t0, r0, g0, b0) = stops[i];
    let (t1, r1, g1, b1) = stops[i + 1];
    let frac = if (t1 - t0).abs() < 1e-12 {
        0.0
    } else {
        ((t - t0) / (t1 - t0)) as f32
    };

    [
        r0 + (r1 - r0) * frac,
        g0 + (g1 - g0) * frac,
        b0 + (b1 - b0) * frac,
        1.0,
    ]
}

// ---------------------------------------------------------------------------
// Colorization
// ---------------------------------------------------------------------------

/// Generate per-node RGBA colours from a scalar field and a colormap.
///
/// The scalar values are linearly normalized to `[0, 1]` based on the field's
/// min/max range.  If all values are equal, every node receives the midpoint
/// colour (`t = 0.5`).
pub fn colorize_nodes(field: &MeshScalarField, colormap: ColormapFn) -> Vec<[f32; 4]> {
    let min = field.min();
    let max = field.max();
    let range = max - min;

    field
        .values
        .iter()
        .map(|&v| {
            let t = if range.abs() < 1e-15 {
                0.5
            } else {
                (v - min) / range
            };
            colormap(t)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ElementType, Mesh, MeshElement};
    use simucad_core::types::Vec3;

    /// Helper: build a simple 3-node triangle mesh.
    fn triangle_mesh() -> Mesh {
        Mesh {
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
        }
    }

    #[test]
    fn scalar_field_min_max() {
        let field = MeshScalarField::new(vec![3.0, -1.0, 7.5, 2.0]);
        assert!((field.min() - (-1.0)).abs() < 1e-12);
        assert!((field.max() - 7.5).abs() < 1e-12);

        // Empty field edge case.
        let empty = MeshScalarField::new(vec![]);
        assert_eq!(empty.min(), f64::MAX);
        assert_eq!(empty.max(), f64::MIN);
    }

    #[test]
    fn scalar_field_from_function() {
        let mesh = triangle_mesh();
        let field = MeshScalarField::from_function(&mesh, |v| v.x + v.y);

        // Node 0: (0,0,0) -> 0
        // Node 1: (1,0,0) -> 1
        // Node 2: (0,1,0) -> 1
        assert!((field.values[0] - 0.0).abs() < 1e-12);
        assert!((field.values[1] - 1.0).abs() < 1e-12);
        assert!((field.values[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn interpolation_at_centroid() {
        let mesh = triangle_mesh();
        // Assign distinct values to the 3 nodes.
        let field = MeshScalarField::new(vec![3.0, 6.0, 9.0]);

        // Equal barycentric coords at the centroid.
        let third = 1.0 / 3.0;
        let interp = field.interpolate_at_element(&mesh, 0, &[third, third, third]);

        // Expected: (3 + 6 + 9) / 3 = 6
        assert!((interp - 6.0).abs() < 1e-12);
    }

    #[test]
    fn colorize_uniform_field() {
        // All values the same -> all colours identical (midpoint colour).
        let field = MeshScalarField::new(vec![5.0, 5.0, 5.0, 5.0]);
        let colours = colorize_nodes(&field, colormap_viridis_rgba);

        let first = colours[0];
        for c in &colours {
            assert_eq!(c, &first);
        }
    }

    #[test]
    fn vector_field_magnitude() {
        let field = MeshVectorField::new(vec![
            Vec3::new(3.0, 4.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 2.0),
        ]);

        let mag = field.magnitude();
        assert!((mag.values[0] - 5.0).abs() < 1e-12);
        assert!((mag.values[1] - 0.0).abs() < 1e-12);
        assert!((mag.values[2] - 3.0).abs() < 1e-12);
    }
}
