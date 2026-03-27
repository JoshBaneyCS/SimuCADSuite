use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::derivative::differentiate;
use crate::evaluator::{evaluate, Environment};
use crate::simplify::simplify;

/// Solve `expr = 0` for `var` using the Newton-Raphson method.
///
/// Starting from `initial_guess`, iterates until `|f(x)| < tolerance` or
/// `max_iter` iterations have been performed.
pub fn solve_numeric(
    expr: &Expr,
    var: &str,
    initial_guess: f64,
    tolerance: f64,
    max_iter: usize,
) -> Result<f64, CasError> {
    let deriv = simplify(&differentiate(expr, var)?);
    let mut env = Environment::new();
    let mut x = initial_guess;

    for _ in 0..max_iter {
        env.set(var, x);
        let fx = evaluate(expr, &env)?;

        if fx.abs() < tolerance {
            return Ok(x);
        }

        let dfx = evaluate(&deriv, &env)?;

        if dfx.abs() < 1e-15 {
            return Err(CasError::UnsupportedOperation(
                "Newton-Raphson: derivative is zero, cannot continue".into(),
            ));
        }

        x = x - fx / dfx;
    }

    Err(CasError::UnsupportedOperation(format!(
        "Newton-Raphson did not converge within {} iterations",
        max_iter
    )))
}

/// Find roots of `expr = 0` for `var` in the interval `[x_min, x_max]`.
///
/// Scans the interval on a grid of `resolution` points looking for sign
/// changes, then refines each root with the bisection method.
pub fn find_roots(
    expr: &Expr,
    var: &str,
    x_min: f64,
    x_max: f64,
    resolution: usize,
) -> Result<Vec<f64>, CasError> {
    if resolution < 2 {
        return Err(CasError::DomainError(
            "resolution must be at least 2".into(),
        ));
    }
    if x_min >= x_max {
        return Err(CasError::DomainError(
            "x_min must be less than x_max".into(),
        ));
    }

    let step = (x_max - x_min) / (resolution - 1) as f64;
    let mut env = Environment::new();
    let mut roots = Vec::new();

    // Evaluate on grid
    let mut grid: Vec<(f64, f64)> = Vec::with_capacity(resolution);
    for i in 0..resolution {
        let x = x_min + step * i as f64;
        env.set(var, x);
        if let Ok(y) = evaluate(expr, &env) {
            if y.is_finite() {
                grid.push((x, y));
            }
        }
    }

    // Check for exact zeros and sign changes
    for i in 0..grid.len() {
        let (x, y) = grid[i];

        // Exact zero (or very close)
        if y.abs() < 1e-12 {
            // Avoid duplicates
            if roots.last().map_or(true, |&last: &f64| (x - last).abs() > step * 0.5) {
                roots.push(x);
            }
            continue;
        }

        // Sign change between grid[i] and grid[i+1]
        if i + 1 < grid.len() {
            let (x2, y2) = grid[i + 1];
            if y * y2 < 0.0 {
                // Bisection refinement
                match bisect(expr, var, x, y, x2, y2, 60) {
                    Ok(root) => {
                        if roots
                            .last()
                            .map_or(true, |&last: &f64| (root - last).abs() > step * 0.5)
                        {
                            roots.push(root);
                        }
                    }
                    Err(_) => { /* skip this interval */ }
                }
            }
        }
    }

    Ok(roots)
}

/// Bisection method to refine a root in [a, b] where f(a) and f(b) have
/// opposite signs.
fn bisect(
    expr: &Expr,
    var: &str,
    mut a: f64,
    mut fa: f64,
    mut b: f64,
    mut _fb: f64,
    max_iter: usize,
) -> Result<f64, CasError> {
    let mut env = Environment::new();

    for _ in 0..max_iter {
        let mid = (a + b) / 2.0;
        env.set(var, mid);
        let fmid = evaluate(expr, &env)?;

        if fmid.abs() < 1e-12 || (b - a).abs() < 1e-14 {
            return Ok(mid);
        }

        if fa * fmid < 0.0 {
            b = mid;
            _fb = fmid;
        } else {
            a = mid;
            fa = fmid;
        }
    }

    Ok((a + b) / 2.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_x_squared_minus_4() {
        // x^2 - 4 = 0, root at x = 2 (starting from 3)
        let expr = Expr::sub(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::num(4.0),
        );
        let root = solve_numeric(&expr, "x", 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_newton_x_squared_minus_4_negative_root() {
        // x^2 - 4 = 0, root at x = -2 (starting from -3)
        let expr = Expr::sub(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::num(4.0),
        );
        let root = solve_numeric(&expr, "x", -3.0, 1e-10, 100).unwrap();
        assert!((root - (-2.0)).abs() < 1e-8);
    }

    #[test]
    fn test_newton_sin_near_pi() {
        // sin(x) = 0, root near pi
        let expr = Expr::func("sin", vec![Expr::var("x")]);
        let root = solve_numeric(&expr, "x", 3.0, 1e-10, 100).unwrap();
        assert!((root - std::f64::consts::PI).abs() < 1e-8);
    }

    #[test]
    fn test_newton_linear() {
        // 2*x - 6 = 0, root at x = 3
        let expr = Expr::sub(
            Expr::mul(Expr::num(2.0), Expr::var("x")),
            Expr::num(6.0),
        );
        let root = solve_numeric(&expr, "x", 0.0, 1e-10, 100).unwrap();
        assert!((root - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_newton_max_iter_exceeded() {
        // With a very tight tolerance and very few iterations
        let expr = Expr::sub(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::num(2.0),
        );
        let result = solve_numeric(&expr, "x", 100.0, 1e-15, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_roots_x_squared_minus_4() {
        // x^2 - 4 = 0 has roots at -2 and 2
        let expr = Expr::sub(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::num(4.0),
        );
        let roots = find_roots(&expr, "x", -5.0, 5.0, 1000).unwrap();
        assert!(roots.len() >= 2);

        let has_neg2 = roots.iter().any(|r| (r - (-2.0)).abs() < 1e-6);
        let has_pos2 = roots.iter().any(|r| (r - 2.0).abs() < 1e-6);
        assert!(has_neg2, "should find root at -2, got {:?}", roots);
        assert!(has_pos2, "should find root at 2, got {:?}", roots);
    }

    #[test]
    fn test_find_roots_sin() {
        // sin(x) = 0 in [-4, 4] has roots at -pi, 0, pi
        let expr = Expr::func("sin", vec![Expr::var("x")]);
        let roots = find_roots(&expr, "x", -4.0, 4.0, 1000).unwrap();

        let has_zero = roots.iter().any(|r| r.abs() < 1e-6);
        let has_neg_pi = roots.iter().any(|r| (r - (-std::f64::consts::PI)).abs() < 1e-6);
        let has_pi = roots.iter().any(|r| (r - std::f64::consts::PI).abs() < 1e-6);

        assert!(has_zero, "should find root at 0, got {:?}", roots);
        assert!(has_neg_pi, "should find root near -pi, got {:?}", roots);
        assert!(has_pi, "should find root near pi, got {:?}", roots);
    }

    #[test]
    fn test_find_roots_linear() {
        // x - 3 = 0 has root at 3
        let expr = Expr::sub(Expr::var("x"), Expr::num(3.0));
        let roots = find_roots(&expr, "x", 0.0, 10.0, 100).unwrap();
        assert!(!roots.is_empty());
        assert!(roots.iter().any(|r| (r - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_find_roots_no_roots() {
        // x^2 + 1 = 0 has no real roots
        let expr = Expr::add(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::num(1.0),
        );
        let roots = find_roots(&expr, "x", -10.0, 10.0, 100).unwrap();
        assert!(roots.is_empty());
    }

    #[test]
    fn test_find_roots_cubic() {
        // x^3 - x = x(x-1)(x+1) has roots at -1, 0, 1
        let expr = Expr::sub(
            Expr::pow(Expr::var("x"), Expr::num(3.0)),
            Expr::var("x"),
        );
        let roots = find_roots(&expr, "x", -2.0, 2.0, 1000).unwrap();
        assert!(roots.len() >= 3, "expected 3 roots, got {:?}", roots);

        let has_neg1 = roots.iter().any(|r| (r - (-1.0)).abs() < 1e-6);
        let has_zero = roots.iter().any(|r| r.abs() < 1e-6);
        let has_pos1 = roots.iter().any(|r| (r - 1.0).abs() < 1e-6);

        assert!(has_neg1, "should find root at -1, got {:?}", roots);
        assert!(has_zero, "should find root at 0, got {:?}", roots);
        assert!(has_pos1, "should find root at 1, got {:?}", roots);
    }

    #[test]
    fn test_find_roots_bad_range() {
        let expr = Expr::var("x");
        assert!(find_roots(&expr, "x", 5.0, 0.0, 100).is_err());
    }

    #[test]
    fn test_find_roots_bad_resolution() {
        let expr = Expr::var("x");
        assert!(find_roots(&expr, "x", 0.0, 5.0, 1).is_err());
    }
}
