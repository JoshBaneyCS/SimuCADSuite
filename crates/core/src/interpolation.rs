use crate::error::SimuError;
use crate::types::{Vec2, Vec3};

// ---------------------------------------------------------------------------
// Scalar interpolation functions
// ---------------------------------------------------------------------------

/// Linear interpolation between two values.
///
/// Returns `a` when `t = 0`, `b` when `t = 1`, and linearly blends for
/// values of `t` in between.  Values of `t` outside `[0, 1]` extrapolate.
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Bilinear interpolation on a 2D grid.
///
/// `c00`, `c10`, `c01`, `c11` are the values at the four corners of a unit
/// cell.  `tx` interpolates along x, `ty` along y.
pub fn bilerp(c00: f64, c10: f64, c01: f64, c11: f64, tx: f64, ty: f64) -> f64 {
    let x0 = lerp(c00, c10, tx);
    let x1 = lerp(c01, c11, tx);
    lerp(x0, x1, ty)
}

/// Cubic Hermite spline interpolation.
///
/// Interpolates between `p0` (at `t = 0`) and `p1` (at `t = 1`) with
/// tangents `m0` and `m1` at those endpoints.
pub fn hermite(p0: f64, p1: f64, m0: f64, m1: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

// ---------------------------------------------------------------------------
// LookupTable1D
// ---------------------------------------------------------------------------

/// 1-D lookup table with linear interpolation.
///
/// The `xs` vector must be strictly monotonically increasing and have the
/// same length as `ys`.  Evaluations outside the table domain are clamped
/// to the boundary values.
#[derive(Debug, Clone)]
pub struct LookupTable1D {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl LookupTable1D {
    /// Create a new lookup table.
    ///
    /// # Errors
    /// Returns `SimuError::Config` if:
    /// - `xs` and `ys` have different lengths
    /// - fewer than 2 entries are provided
    /// - `xs` is not strictly monotonically increasing
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Result<Self, SimuError> {
        if xs.len() != ys.len() {
            return Err(SimuError::Config(format!(
                "LookupTable1D: xs length ({}) != ys length ({})",
                xs.len(),
                ys.len()
            )));
        }
        if xs.len() < 2 {
            return Err(SimuError::Config(
                "LookupTable1D: need at least 2 data points".into(),
            ));
        }
        for w in xs.windows(2) {
            if w[1] <= w[0] {
                return Err(SimuError::Config(format!(
                    "LookupTable1D: xs must be strictly increasing, found {} followed by {}",
                    w[0], w[1]
                )));
            }
        }
        Ok(Self { xs, ys })
    }

    /// Evaluate the table at `x` using linear interpolation.
    ///
    /// Values outside the domain are clamped to the first / last y value.
    pub fn evaluate(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            return self.ys[0];
        }
        let last = self.xs.len() - 1;
        if x >= self.xs[last] {
            return self.ys[last];
        }

        // Binary search for the interval containing x.
        let idx = match self.xs.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
            Ok(i) => return self.ys[i], // exact match
            Err(i) => i - 1,            // x is between xs[i-1] and xs[i]
        };

        let t = (x - self.xs[idx]) / (self.xs[idx + 1] - self.xs[idx]);
        lerp(self.ys[idx], self.ys[idx + 1], t)
    }
}

// ---------------------------------------------------------------------------
// Barycentric / geometric helpers
// ---------------------------------------------------------------------------

/// Compute barycentric coordinates `(u, v, w)` of point `p` with respect
/// to triangle `(a, b, c)`.  The coordinates satisfy `u + v + w = 1` when
/// `p` lies in the plane of the triangle.
pub fn barycentric_coords(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> (f64, f64, f64) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;

    let d00 = v0.dot(&v0);
    let d01 = v0.dot(&v1);
    let d11 = v1.dot(&v1);
    let d20 = v2.dot(&v0);
    let d21 = v2.dot(&v1);

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, v, w)
}

/// Returns `true` if point `p` is inside triangle `(a, b, c)` (inclusive of
/// edges) using barycentric coordinates.
pub fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let (u, v, w) = barycentric_coords(p, a, b, c);
    u >= 0.0 && v >= 0.0 && w >= 0.0
}

/// Returns `true` if point `p` is inside tetrahedron `(a, b, c, d)`.
///
/// Uses the sign-of-determinant method: `p` is inside iff it is on the same
/// side of each face as the opposing vertex.
pub fn point_in_tetrahedron(p: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool {
    fn sign(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
        let ab = b - a;
        let ac = c - a;
        let ad = d - a;
        ab.cross(&ac).dot(&ad)
    }

    let s0 = sign(a, b, c, d);
    let s1 = sign(a, b, c, p);
    let s2 = sign(a, b, d, p);
    let s3 = sign(a, c, d, p);
    let s4 = sign(b, c, d, p);

    // p is inside iff it is on the same side of each face as the opposite vertex.
    // Face (a,b,c) opposite d => s0 and s1 same sign
    // Face (a,b,d) opposite c => sign(a,b,d,c) and s2 same sign
    // Face (a,c,d) opposite b => sign(a,c,d,b) and s3 same sign
    // Face (b,c,d) opposite a => sign(b,c,d,a) and s4 same sign
    let same_sign = |x: f64, y: f64| (x >= 0.0 && y >= 0.0) || (x <= 0.0 && y <= 0.0);

    same_sign(s0, s1)
        && same_sign(sign(a, b, d, c), s2)
        && same_sign(sign(a, c, d, b), s3)
        && same_sign(sign(b, c, d, a), s4)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn approx(a: f64, b: f64) {
        assert!(
            (a - b).abs() < EPS,
            "expected {b}, got {a}, diff = {}",
            (a - b).abs()
        );
    }

    // -- lerp ---------------------------------------------------------------

    #[test]
    fn lerp_endpoints() {
        approx(lerp(2.0, 8.0, 0.0), 2.0);
        approx(lerp(2.0, 8.0, 1.0), 8.0);
    }

    #[test]
    fn lerp_midpoint() {
        approx(lerp(0.0, 10.0, 0.5), 5.0);
    }

    #[test]
    fn lerp_extrapolation() {
        approx(lerp(0.0, 10.0, 2.0), 20.0);
        approx(lerp(0.0, 10.0, -1.0), -10.0);
    }

    // -- bilerp -------------------------------------------------------------

    #[test]
    fn bilerp_corners() {
        approx(bilerp(1.0, 2.0, 3.0, 4.0, 0.0, 0.0), 1.0);
        approx(bilerp(1.0, 2.0, 3.0, 4.0, 1.0, 0.0), 2.0);
        approx(bilerp(1.0, 2.0, 3.0, 4.0, 0.0, 1.0), 3.0);
        approx(bilerp(1.0, 2.0, 3.0, 4.0, 1.0, 1.0), 4.0);
    }

    #[test]
    fn bilerp_center() {
        approx(bilerp(0.0, 2.0, 2.0, 4.0, 0.5, 0.5), 2.0);
    }

    // -- hermite ------------------------------------------------------------

    #[test]
    fn hermite_endpoints() {
        approx(hermite(1.0, 4.0, 0.0, 0.0, 0.0), 1.0);
        approx(hermite(1.0, 4.0, 0.0, 0.0, 1.0), 4.0);
    }

    #[test]
    fn hermite_linear_with_matching_tangents() {
        // When tangents match the slope (p1-p0), hermite reduces to linear.
        let p0 = 2.0;
        let p1 = 8.0;
        let slope = p1 - p0;
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            approx(hermite(p0, p1, slope, slope, t), lerp(p0, p1, t));
        }
    }

    // -- LookupTable1D ------------------------------------------------------

    #[test]
    fn lookup_table_basic() {
        let table = LookupTable1D::new(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 20.0]).unwrap();
        approx(table.evaluate(0.5), 5.0);
        approx(table.evaluate(1.5), 15.0);
    }

    #[test]
    fn lookup_table_clamp() {
        let table = LookupTable1D::new(vec![0.0, 1.0], vec![5.0, 15.0]).unwrap();
        approx(table.evaluate(-100.0), 5.0);
        approx(table.evaluate(100.0), 15.0);
    }

    #[test]
    fn lookup_table_exact_hit() {
        let table = LookupTable1D::new(vec![0.0, 1.0, 2.0], vec![10.0, 20.0, 30.0]).unwrap();
        approx(table.evaluate(1.0), 20.0);
    }

    #[test]
    fn lookup_table_validation_length_mismatch() {
        assert!(LookupTable1D::new(vec![0.0, 1.0], vec![5.0]).is_err());
    }

    #[test]
    fn lookup_table_validation_too_few() {
        assert!(LookupTable1D::new(vec![0.0], vec![5.0]).is_err());
    }

    #[test]
    fn lookup_table_validation_not_increasing() {
        assert!(LookupTable1D::new(vec![0.0, 0.0], vec![1.0, 2.0]).is_err());
        assert!(LookupTable1D::new(vec![1.0, 0.0], vec![1.0, 2.0]).is_err());
    }

    // -- barycentric_coords -------------------------------------------------

    #[test]
    fn barycentric_at_vertices() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.0, 1.0);

        let (u, v, w) = barycentric_coords(a, a, b, c);
        approx(u, 1.0);
        approx(v, 0.0);
        approx(w, 0.0);

        let (u, v, w) = barycentric_coords(b, a, b, c);
        approx(u, 0.0);
        approx(v, 1.0);
        approx(w, 0.0);

        let (u, v, w) = barycentric_coords(c, a, b, c);
        approx(u, 0.0);
        approx(v, 0.0);
        approx(w, 1.0);
    }

    #[test]
    fn barycentric_centroid() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 0.0);
        let c = Vec2::new(0.0, 3.0);
        let centroid = Vec2::new(1.0, 1.0);
        let (u, v, w) = barycentric_coords(centroid, a, b, c);
        approx(u, 1.0 / 3.0);
        approx(v, 1.0 / 3.0);
        approx(w, 1.0 / 3.0);
    }

    // -- point_in_triangle --------------------------------------------------

    #[test]
    fn point_inside_triangle() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(4.0, 0.0);
        let c = Vec2::new(0.0, 4.0);
        assert!(point_in_triangle(Vec2::new(1.0, 1.0), a, b, c));
    }

    #[test]
    fn point_outside_triangle() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(1.0, 0.0);
        let c = Vec2::new(0.0, 1.0);
        assert!(!point_in_triangle(Vec2::new(1.0, 1.0), a, b, c));
    }

    #[test]
    fn point_on_edge() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(2.0, 0.0);
        let c = Vec2::new(0.0, 2.0);
        assert!(point_in_triangle(Vec2::new(1.0, 0.0), a, b, c));
    }

    // -- point_in_tetrahedron -----------------------------------------------

    #[test]
    fn point_inside_tetrahedron() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let d = Vec3::new(0.0, 0.0, 1.0);
        // Centroid is at (0.25, 0.25, 0.25)
        assert!(point_in_tetrahedron(Vec3::new(0.1, 0.1, 0.1), a, b, c, d));
    }

    #[test]
    fn point_outside_tetrahedron() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let d = Vec3::new(0.0, 0.0, 1.0);
        assert!(!point_in_tetrahedron(Vec3::new(2.0, 2.0, 2.0), a, b, c, d));
    }

    #[test]
    fn point_at_vertex_of_tetrahedron() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let d = Vec3::new(0.0, 0.0, 1.0);
        assert!(point_in_tetrahedron(a, a, b, c, d));
    }
}
