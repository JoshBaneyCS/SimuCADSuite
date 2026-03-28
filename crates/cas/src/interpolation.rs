//! Polynomial interpolation and regression.
//!
//! Given a set of data points, constructs a symbolic polynomial expression
//! that passes through every point (Lagrange interpolation) or best fits
//! the data in a least-squares sense (polynomial regression).

use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::simplify::simplify;

// ---------------------------------------------------------------------------
// Lagrange interpolation (exact fit)
// ---------------------------------------------------------------------------

/// Construct the unique polynomial of degree `n-1` that passes through all
/// `n` data points using Lagrange interpolation.
///
/// Returns a symbolic [`Expr`] in the variable `var`.
///
/// # Parameters
///
/// - `points` -- `(x, y)` data points. Must have at least 1 point and all
///   x-values must be distinct.
/// - `var` -- the variable name for the resulting polynomial (e.g. `"x"`).
///
/// # Examples
///
/// ```ignore
/// // Points (0,1), (1,3), (2,7) => quadratic through all three
/// let poly = lagrange_interpolate(&[(0.0,1.0),(1.0,3.0),(2.0,7.0)], "x")?;
/// ```
pub fn lagrange_interpolate(
    points: &[(f64, f64)],
    var: &str,
) -> Result<Expr, CasError> {
    if points.is_empty() {
        return Err(CasError::DomainError(
            "At least one data point is required".into(),
        ));
    }

    // Verify distinct x-values.
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            if (points[i].0 - points[j].0).abs() < 1e-15 {
                return Err(CasError::DomainError(format!(
                    "Duplicate x-value: {} at indices {} and {}",
                    points[i].0, i, j
                )));
            }
        }
    }

    let n = points.len();

    if n == 1 {
        return Ok(Expr::num(points[0].1));
    }

    // L(x) = sum_i y_i * prod_{j != i} (x - x_j) / (x_i - x_j)
    let mut terms: Vec<Expr> = Vec::with_capacity(n);

    for i in 0..n {
        let yi = points[i].1;
        if yi.abs() < 1e-15 {
            continue; // skip zero terms
        }

        let mut basis = Expr::num(1.0);
        for j in 0..n {
            if j == i {
                continue;
            }
            let xi = points[i].0;
            let xj = points[j].0;
            let denom = xi - xj;

            // (x - xj) / denom
            let factor = Expr::div(
                Expr::sub(Expr::var(var), Expr::num(xj)),
                Expr::num(denom),
            );
            basis = Expr::mul(basis, factor);
        }

        terms.push(Expr::mul(Expr::num(yi), basis));
    }

    if terms.is_empty() {
        return Ok(Expr::num(0.0));
    }

    let mut result = terms.remove(0);
    for t in terms {
        result = Expr::add(result, t);
    }

    Ok(simplify(&result))
}

// ---------------------------------------------------------------------------
// Newton's divided differences interpolation
// ---------------------------------------------------------------------------

/// Construct the interpolating polynomial using Newton's divided differences.
///
/// Mathematically equivalent to Lagrange but often simpler to expand. The
/// result is a symbolic polynomial in `var`.
pub fn newton_interpolate(
    points: &[(f64, f64)],
    var: &str,
) -> Result<Expr, CasError> {
    if points.is_empty() {
        return Err(CasError::DomainError(
            "At least one data point is required".into(),
        ));
    }

    let n = points.len();

    // Build divided difference table.
    let mut dd: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        dd[i][0] = points[i].1;
    }
    for j in 1..n {
        for i in 0..(n - j) {
            let dx = points[i + j].0 - points[i].0;
            if dx.abs() < 1e-15 {
                return Err(CasError::DomainError(format!(
                    "Duplicate x-value near {} in divided differences",
                    points[i].0
                )));
            }
            dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / dx;
        }
    }

    // Build the polynomial: sum_j dd[0][j] * prod_{k=0..j-1} (x - x_k)
    let mut result = Expr::num(dd[0][0]);

    for j in 1..n {
        let coeff = dd[0][j];
        if coeff.abs() < 1e-15 {
            continue;
        }

        let mut product = Expr::num(coeff);
        for k in 0..j {
            product = Expr::mul(
                product,
                Expr::sub(Expr::var(var), Expr::num(points[k].0)),
            );
        }
        result = Expr::add(result, product);
    }

    Ok(simplify(&result))
}

// ---------------------------------------------------------------------------
// Polynomial regression (least-squares fit)
// ---------------------------------------------------------------------------

/// Fit a polynomial of the given `degree` to the data points using
/// least-squares regression.
///
/// Unlike interpolation, the polynomial need not pass through every point.
/// Instead it minimizes the sum of squared residuals. The number of points
/// must exceed the degree.
///
/// Returns a symbolic polynomial in `var` with numeric coefficients.
pub fn polynomial_regression(
    points: &[(f64, f64)],
    var: &str,
    degree: usize,
) -> Result<Expr, CasError> {
    let n = points.len();
    if n == 0 {
        return Err(CasError::DomainError(
            "At least one data point is required".into(),
        ));
    }
    if degree >= n {
        return Err(CasError::DomainError(format!(
            "Degree ({degree}) must be less than number of points ({n})"
        )));
    }

    let m = degree + 1; // number of coefficients

    // Build the Vandermonde system: V^T V c = V^T y
    // V[i][j] = x_i^j
    let mut vtv = vec![vec![0.0_f64; m]; m];
    let mut vty = vec![0.0_f64; m];

    for &(xi, yi) in points {
        let mut xi_pow = 1.0;
        for j in 0..m {
            vty[j] += yi * xi_pow;
            let mut xi_pow2 = 1.0;
            for k in 0..m {
                vtv[j][k] += xi_pow * xi_pow2;
                xi_pow2 *= xi;
            }
            xi_pow *= xi;
        }
    }

    // Solve via Gaussian elimination with partial pivoting.
    let coeffs = solve_linear_system(&mut vtv, &mut vty)?;

    // Build the polynomial expression: c0 + c1*x + c2*x^2 + ...
    let mut result = Expr::num(0.0);
    for (k, &c) in coeffs.iter().enumerate() {
        if c.abs() < 1e-15 {
            continue;
        }
        let term = if k == 0 {
            Expr::num(c)
        } else if k == 1 {
            Expr::mul(Expr::num(c), Expr::var(var))
        } else {
            Expr::mul(
                Expr::num(c),
                Expr::pow(Expr::var(var), Expr::num(k as f64)),
            )
        };
        result = Expr::add(result, term);
    }

    Ok(simplify(&result))
}

/// Solve a small dense linear system Ax = b via Gaussian elimination with
/// partial pivoting. Modifies A and b in place.
fn solve_linear_system(
    a: &mut [Vec<f64>],
    b: &mut [f64],
) -> Result<Vec<f64>, CasError> {
    let n = b.len();

    // Forward elimination.
    for col in 0..n {
        // Partial pivoting: find row with largest |a[row][col]|.
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(CasError::DomainError(
                "Singular or near-singular Vandermonde matrix".into(),
            ));
        }

        // Swap rows.
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        // Eliminate below.
        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = b[col];
        for k in (col + 1)..n {
            sum -= a[col][k] * x[k];
        }
        x[col] = sum / a[col][col];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::{evaluate, Environment};

    fn eval_at(expr: &Expr, var: &str, x: f64) -> f64 {
        let mut env = Environment::new();
        env.set(var, x);
        evaluate(expr, &env).unwrap()
    }

    // ----- Lagrange interpolation -----

    #[test]
    fn lagrange_single_point() {
        let poly = lagrange_interpolate(&[(3.0, 7.0)], "x").unwrap();
        let val = eval_at(&poly, "x", 999.0);
        assert!((val - 7.0).abs() < 1e-10, "constant poly should be 7, got {val}");
    }

    #[test]
    fn lagrange_two_points_linear() {
        // (0, 1) and (2, 5) => y = 2x + 1
        let poly = lagrange_interpolate(&[(0.0, 1.0), (2.0, 5.0)], "x").unwrap();
        assert!((eval_at(&poly, "x", 0.0) - 1.0).abs() < 1e-10);
        assert!((eval_at(&poly, "x", 2.0) - 5.0).abs() < 1e-10);
        assert!((eval_at(&poly, "x", 1.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn lagrange_three_points_quadratic() {
        // (0,1), (1,3), (2,7) => f(x) = x^2 + x + 1
        let pts = [(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let poly = lagrange_interpolate(&pts, "x").unwrap();

        for &(x, y) in &pts {
            let val = eval_at(&poly, "x", x);
            assert!((val - y).abs() < 1e-10, "at x={x}: got {val}, expected {y}");
        }

        // Also check intermediate point: f(1.5) = 2.25 + 1.5 + 1 = 4.75
        let val = eval_at(&poly, "x", 1.5);
        assert!((val - 4.75).abs() < 1e-8, "at x=1.5: got {val}, expected 4.75");
    }

    #[test]
    fn lagrange_five_points() {
        let pts = [
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 8.0),
            (3.0, 27.0),
            (4.0, 64.0),
        ];
        let poly = lagrange_interpolate(&pts, "x").unwrap();

        for &(x, y) in &pts {
            let val = eval_at(&poly, "x", x);
            assert!(
                (val - y).abs() < 1e-6,
                "at x={x}: got {val}, expected {y}"
            );
        }
    }

    #[test]
    fn lagrange_duplicate_x_error() {
        let pts = [(1.0, 2.0), (1.0, 3.0)];
        assert!(lagrange_interpolate(&pts, "x").is_err());
    }

    #[test]
    fn lagrange_empty_error() {
        assert!(lagrange_interpolate(&[], "x").is_err());
    }

    // ----- Newton interpolation -----

    #[test]
    fn newton_matches_lagrange() {
        let pts = [(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let lagrange = lagrange_interpolate(&pts, "x").unwrap();
        let newton = newton_interpolate(&pts, "x").unwrap();

        // Both should give the same values at test points.
        for x in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0] {
            let lv = eval_at(&lagrange, "x", x);
            let nv = eval_at(&newton, "x", x);
            assert!(
                (lv - nv).abs() < 1e-8,
                "at x={x}: lagrange={lv}, newton={nv}"
            );
        }
    }

    #[test]
    fn newton_single_point() {
        let poly = newton_interpolate(&[(5.0, 42.0)], "x").unwrap();
        let val = eval_at(&poly, "x", 100.0);
        assert!((val - 42.0).abs() < 1e-10);
    }

    // ----- Polynomial regression -----

    #[test]
    fn regression_linear_exact() {
        // Points on the line y = 2x + 1
        let pts: Vec<(f64, f64)> = (0..5).map(|i| {
            let x = i as f64;
            (x, 2.0 * x + 1.0)
        }).collect();

        let poly = polynomial_regression(&pts, "x", 1).unwrap();

        for &(x, y) in &pts {
            let val = eval_at(&poly, "x", x);
            assert!(
                (val - y).abs() < 1e-8,
                "at x={x}: got {val}, expected {y}"
            );
        }
    }

    #[test]
    fn regression_quadratic_exact() {
        // Points on y = x^2 - 3x + 2
        let pts: Vec<(f64, f64)> = (0..10).map(|i| {
            let x = i as f64;
            (x, x * x - 3.0 * x + 2.0)
        }).collect();

        let poly = polynomial_regression(&pts, "x", 2).unwrap();

        for &(x, y) in &pts {
            let val = eval_at(&poly, "x", x);
            assert!(
                (val - y).abs() < 1e-6,
                "at x={x}: got {val}, expected {y}"
            );
        }
    }

    #[test]
    fn regression_degree_too_high() {
        let pts = [(0.0, 1.0), (1.0, 2.0)];
        // degree 2 requires at least 3 points
        assert!(polynomial_regression(&pts, "x", 2).is_err());
    }

    #[test]
    fn regression_noisy_data() {
        // Noisy linear data — regression should approximate y ≈ x
        let pts = [
            (0.0, 0.1),
            (1.0, 0.9),
            (2.0, 2.1),
            (3.0, 2.9),
            (4.0, 4.1),
        ];
        let poly = polynomial_regression(&pts, "x", 1).unwrap();

        // Slope should be near 1, intercept near 0
        let at0 = eval_at(&poly, "x", 0.0);
        let at4 = eval_at(&poly, "x", 4.0);
        assert!(at0.abs() < 0.5, "intercept too far from 0: {at0}");
        assert!((at4 - 4.0).abs() < 0.5, "at x=4 too far from 4: {at4}");
    }
}
