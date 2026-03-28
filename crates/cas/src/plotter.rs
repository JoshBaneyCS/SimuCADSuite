use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::evaluator::{evaluate, Environment};

/// Generate a set of 2D `(x, y)` points by evaluating `expr` over the variable
/// `var` at evenly spaced values in `[x_min, x_max]`.
///
/// Points where the evaluation fails (e.g., domain errors) are silently
/// skipped, so the result may contain fewer than `num_points` entries.
pub fn generate_2d_points(
    expr: &Expr,
    var: &str,
    x_min: f64,
    x_max: f64,
    num_points: usize,
) -> Result<Vec<(f64, f64)>, CasError> {
    if num_points < 2 {
        return Err(CasError::DomainError(
            "num_points must be at least 2".into(),
        ));
    }
    if x_min >= x_max {
        return Err(CasError::DomainError(
            "x_min must be less than x_max".into(),
        ));
    }

    let step = (x_max - x_min) / (num_points - 1) as f64;
    let mut points = Vec::with_capacity(num_points);
    let mut env = Environment::new();

    for i in 0..num_points {
        let x = x_min + step * i as f64;
        env.set(var, x);
        match evaluate(expr, &env) {
            Ok(y) if y.is_finite() => points.push((x, y)),
            _ => { /* skip domain errors / NaN / Inf */ }
        }
    }

    Ok(points)
}

/// Generate a set of 3D `(x, y, z)` points by evaluating `expr` over a grid
/// defined by `x_var` in `x_range` and `y_var` in `y_range`, each with
/// `resolution` samples.
///
/// Points where the evaluation fails are silently skipped.
pub fn generate_3d_points(
    expr: &Expr,
    x_var: &str,
    y_var: &str,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
) -> Result<Vec<(f64, f64, f64)>, CasError> {
    if resolution < 2 {
        return Err(CasError::DomainError(
            "resolution must be at least 2".into(),
        ));
    }
    if x_range.0 >= x_range.1 {
        return Err(CasError::DomainError(
            "x_range min must be less than max".into(),
        ));
    }
    if y_range.0 >= y_range.1 {
        return Err(CasError::DomainError(
            "y_range min must be less than max".into(),
        ));
    }

    let x_step = (x_range.1 - x_range.0) / (resolution - 1) as f64;
    let y_step = (y_range.1 - y_range.0) / (resolution - 1) as f64;
    let mut points = Vec::with_capacity(resolution * resolution);
    let mut env = Environment::new();

    for i in 0..resolution {
        let x = x_range.0 + x_step * i as f64;
        env.set(x_var, x);
        for j in 0..resolution {
            let y = y_range.0 + y_step * j as f64;
            env.set(y_var, y);
            match evaluate(expr, &env) {
                Ok(z) if z.is_finite() => points.push((x, y, z)),
                _ => { /* skip */ }
            }
        }
    }

    Ok(points)
}

// ---------------------------------------------------------------------------
// Polar point generation
// ---------------------------------------------------------------------------

/// Generate 2D `(x, y)` points from a polar expression `r = f(theta)`.
///
/// Evaluates the expression at evenly spaced theta values in
/// `[theta_min, theta_max]` and converts to Cartesian: `(r*cos(theta),
/// r*sin(theta))`. Points where the evaluation fails or `r` is negative
/// are silently skipped.
pub fn generate_polar_points(
    expr: &Expr,
    theta_var: &str,
    theta_min: f64,
    theta_max: f64,
    num_points: usize,
) -> Result<Vec<(f64, f64)>, CasError> {
    if num_points < 2 {
        return Err(CasError::DomainError(
            "num_points must be at least 2".into(),
        ));
    }
    if theta_min >= theta_max {
        return Err(CasError::DomainError(
            "theta_min must be less than theta_max".into(),
        ));
    }

    let step = (theta_max - theta_min) / (num_points - 1) as f64;
    let mut points = Vec::with_capacity(num_points);
    let mut env = Environment::new();

    for i in 0..num_points {
        let theta = theta_min + step * i as f64;
        env.set(theta_var, theta);
        match evaluate(expr, &env) {
            Ok(r) if r.is_finite() => {
                let x = r * theta.cos();
                let y = r * theta.sin();
                points.push((x, y));
            }
            _ => { /* skip */ }
        }
    }

    Ok(points)
}

// ---------------------------------------------------------------------------
// Parametric point generation
// ---------------------------------------------------------------------------

/// Generate 2D `(x, y)` points from parametric expressions `x = f(t)` and
/// `y = g(t)`.
///
/// Both expressions are evaluated at evenly spaced `t` values in
/// `[t_min, t_max]`. Points where either evaluation fails are silently
/// skipped.
pub fn generate_parametric_points(
    x_expr: &Expr,
    y_expr: &Expr,
    t_var: &str,
    t_min: f64,
    t_max: f64,
    num_points: usize,
) -> Result<Vec<(f64, f64)>, CasError> {
    if num_points < 2 {
        return Err(CasError::DomainError(
            "num_points must be at least 2".into(),
        ));
    }
    if t_min >= t_max {
        return Err(CasError::DomainError(
            "t_min must be less than t_max".into(),
        ));
    }

    let step = (t_max - t_min) / (num_points - 1) as f64;
    let mut points = Vec::with_capacity(num_points);
    let mut env = Environment::new();

    for i in 0..num_points {
        let t = t_min + step * i as f64;
        env.set(t_var, t);
        let xv = evaluate(x_expr, &env);
        let yv = evaluate(y_expr, &env);
        match (xv, yv) {
            (Ok(x), Ok(y)) if x.is_finite() && y.is_finite() => {
                points.push((x, y));
            }
            _ => { /* skip */ }
        }
    }

    Ok(points)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_2d_linear() {
        // y = x over [0, 1] with 11 points
        let expr = Expr::var("x");
        let pts = generate_2d_points(&expr, "x", 0.0, 1.0, 11).unwrap();
        assert_eq!(pts.len(), 11);
        for &(x, y) in &pts {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn test_generate_2d_constant() {
        let expr = Expr::num(5.0);
        let pts = generate_2d_points(&expr, "x", -1.0, 1.0, 5).unwrap();
        assert_eq!(pts.len(), 5);
        for &(_x, y) in &pts {
            assert!((y - 5.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_generate_2d_skips_domain_errors() {
        // y = sqrt(x) over [-1, 1] — negative x values skipped
        let expr = Expr::func("sqrt", vec![Expr::var("x")]);
        let pts = generate_2d_points(&expr, "x", -1.0, 1.0, 21).unwrap();
        // Should have roughly half the points (the ones >= 0)
        assert!(pts.len() < 21);
        assert!(pts.len() >= 10);
        for &(x, _y) in &pts {
            assert!(x >= 0.0 || (x - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_generate_2d_error_bad_range() {
        let expr = Expr::var("x");
        assert!(generate_2d_points(&expr, "x", 1.0, 0.0, 10).is_err());
    }

    #[test]
    fn test_generate_2d_error_too_few_points() {
        let expr = Expr::var("x");
        assert!(generate_2d_points(&expr, "x", 0.0, 1.0, 1).is_err());
    }

    #[test]
    fn test_generate_3d_plane() {
        // z = x + y
        let expr = Expr::add(Expr::var("x"), Expr::var("y"));
        let pts = generate_3d_points(
            &expr,
            "x",
            "y",
            (0.0, 1.0),
            (0.0, 1.0),
            5,
        )
        .unwrap();
        assert_eq!(pts.len(), 25); // 5x5 grid, no domain errors
        for &(x, y, z) in &pts {
            assert!((z - (x + y)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_generate_3d_constant() {
        let expr = Expr::num(1.0);
        let pts = generate_3d_points(
            &expr,
            "x",
            "y",
            (-1.0, 1.0),
            (-1.0, 1.0),
            3,
        )
        .unwrap();
        assert_eq!(pts.len(), 9);
        for &(_x, _y, z) in &pts {
            assert!((z - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_generate_3d_skips_errors() {
        // z = log(x * y) — will fail for non-positive products
        let expr = Expr::func(
            "log",
            vec![Expr::mul(Expr::var("x"), Expr::var("y"))],
        );
        let pts = generate_3d_points(
            &expr,
            "x",
            "y",
            (-1.0, 1.0),
            (-1.0, 1.0),
            5,
        )
        .unwrap();
        // Only points where x*y > 0 survive
        assert!(pts.len() < 25);
        for &(x, y, z) in &pts {
            assert!((z - (x * y).ln()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_generate_3d_error_bad_range() {
        let expr = Expr::var("x");
        assert!(generate_3d_points(&expr, "x", "y", (1.0, 0.0), (0.0, 1.0), 5).is_err());
    }

    #[test]
    fn test_generate_3d_error_too_few() {
        let expr = Expr::var("x");
        assert!(generate_3d_points(&expr, "x", "y", (0.0, 1.0), (0.0, 1.0), 1).is_err());
    }

    #[test]
    fn test_generate_2d_endpoints() {
        // Verify first and last x values match the range exactly
        let expr = Expr::var("x");
        let pts = generate_2d_points(&expr, "x", -5.0, 5.0, 101).unwrap();
        assert!((pts.first().unwrap().0 - (-5.0)).abs() < 1e-12);
        assert!((pts.last().unwrap().0 - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_generate_2d_quadratic() {
        // y = x^2 over [-2, 2]
        let expr = Expr::pow(Expr::var("x"), Expr::num(2.0));
        let pts = generate_2d_points(&expr, "x", -2.0, 2.0, 5).unwrap();
        assert_eq!(pts.len(), 5);
        for &(x, y) in &pts {
            assert!((y - x * x).abs() < 1e-10);
        }
    }

    // ----- Polar tests -----

    #[test]
    fn test_polar_unit_circle() {
        // r = 1 => should produce points on the unit circle
        let expr = Expr::num(1.0);
        let pts = generate_polar_points(
            &expr, "t", 0.0, std::f64::consts::TAU, 100,
        ).unwrap();
        assert_eq!(pts.len(), 100);
        for &(x, y) in &pts {
            let r = (x * x + y * y).sqrt();
            assert!((r - 1.0).abs() < 1e-10, "point ({x},{y}) not on unit circle: r={r}");
        }
    }

    #[test]
    fn test_polar_error_bad_range() {
        let expr = Expr::num(1.0);
        assert!(generate_polar_points(&expr, "t", 1.0, 0.0, 10).is_err());
    }

    // ----- Parametric tests -----

    #[test]
    fn test_parametric_circle() {
        // x = cos(t), y = sin(t) => unit circle
        let x_expr = Expr::func("cos", vec![Expr::var("t")]);
        let y_expr = Expr::func("sin", vec![Expr::var("t")]);
        let pts = generate_parametric_points(
            &x_expr, &y_expr, "t", 0.0, std::f64::consts::TAU, 100,
        ).unwrap();
        assert_eq!(pts.len(), 100);
        for &(x, y) in &pts {
            let r = (x * x + y * y).sqrt();
            assert!((r - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parametric_line() {
        // x = t, y = 2*t
        let x_expr = Expr::var("t");
        let y_expr = Expr::mul(Expr::num(2.0), Expr::var("t"));
        let pts = generate_parametric_points(
            &x_expr, &y_expr, "t", 0.0, 1.0, 11,
        ).unwrap();
        assert_eq!(pts.len(), 11);
        for &(x, y) in &pts {
            assert!((y - 2.0 * x).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parametric_error_bad_range() {
        let x_expr = Expr::var("t");
        let y_expr = Expr::var("t");
        assert!(generate_parametric_points(&x_expr, &y_expr, "t", 1.0, 0.0, 10).is_err());
    }
}
