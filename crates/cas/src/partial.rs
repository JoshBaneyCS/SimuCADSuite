//! Partial derivative computation for multivariable expressions.
//!
//! Extends the single-variable [`crate::derivative::differentiate`] with
//! higher-order and mixed partial derivatives, gradient vectors, Jacobian
//! matrices, Hessian matrices, and the Laplacian operator.

use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::derivative::differentiate;
use crate::matrix::Matrix;
use crate::simplify::simplify;

// ---------------------------------------------------------------------------
// Single partial derivative (delegates to existing differentiate)
// ---------------------------------------------------------------------------

/// Compute the partial derivative of `expr` with respect to `var`.
///
/// This is syntactically identical to [`differentiate`] but serves as the
/// entry point for the partial derivative API. All other variables are
/// treated as constants.
pub fn partial_derivative(expr: &Expr, var: &str) -> Result<Expr, CasError> {
    let d = differentiate(expr, var)?;
    Ok(simplify(&d))
}

// ---------------------------------------------------------------------------
// Higher-order partial derivatives
// ---------------------------------------------------------------------------

/// Compute an n-th order partial derivative of `expr` with respect to `var`.
///
/// Equivalent to applying `∂/∂var` n times.
pub fn nth_partial_derivative(
    expr: &Expr,
    var: &str,
    order: usize,
) -> Result<Expr, CasError> {
    let mut result = expr.clone();
    for _ in 0..order {
        result = differentiate(&result, var)?;
        result = simplify(&result);
    }
    Ok(result)
}

/// Compute a mixed partial derivative.
///
/// `vars` is a sequence of variables to differentiate with respect to, in
/// order. For example, `["x", "y"]` computes `∂²f / ∂y∂x` (first
/// differentiate w.r.t. x, then w.r.t. y).
pub fn mixed_partial_derivative(
    expr: &Expr,
    vars: &[&str],
) -> Result<Expr, CasError> {
    let mut result = expr.clone();
    for &var in vars {
        result = differentiate(&result, var)?;
        result = simplify(&result);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Gradient, Jacobian, Hessian, Laplacian
// ---------------------------------------------------------------------------

/// Compute the gradient of a scalar expression.
///
/// Returns a vector of partial derivatives `[∂f/∂x₁, ∂f/∂x₂, …]` for the
/// given variable names.
pub fn gradient(expr: &Expr, vars: &[&str]) -> Result<Vec<Expr>, CasError> {
    vars.iter()
        .map(|v| partial_derivative(expr, v))
        .collect()
}

/// Compute the Jacobian matrix of a vector of expressions.
///
/// `exprs[i]` is the i-th component function, `vars[j]` is the j-th
/// variable. Returns a matrix where entry `(i, j) = ∂exprs[i]/∂vars[j]`.
pub fn jacobian(exprs: &[Expr], vars: &[&str]) -> Result<Matrix, CasError> {
    let data: Result<Vec<Vec<Expr>>, CasError> = exprs
        .iter()
        .map(|e| {
            vars.iter()
                .map(|v| partial_derivative(e, v))
                .collect()
        })
        .collect();
    Matrix::new(data?)
}

/// Compute the Hessian matrix of a scalar expression.
///
/// Entry `(i, j) = ∂²f / ∂vars[i]∂vars[j]`.
pub fn hessian(expr: &Expr, vars: &[&str]) -> Result<Matrix, CasError> {
    let data: Result<Vec<Vec<Expr>>, CasError> = vars
        .iter()
        .map(|vi| {
            let di = differentiate(expr, vi)?;
            let di = simplify(&di);
            vars.iter()
                .map(|vj| partial_derivative(&di, vj))
                .collect()
        })
        .collect();
    Matrix::new(data?)
}

/// Compute the Laplacian of a scalar expression.
///
/// `∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + …`
pub fn laplacian(expr: &Expr, vars: &[&str]) -> Result<Expr, CasError> {
    let mut sum: Option<Expr> = None;
    for &v in vars {
        let d2 = nth_partial_derivative(expr, v, 2)?;
        sum = Some(match sum {
            None => d2,
            Some(s) => simplify(&Expr::add(s, d2)),
        });
    }
    Ok(sum.unwrap_or(Expr::num(0.0)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr;
    use crate::evaluator::{evaluate, Environment};

    fn eval_at(expr: &Expr, bindings: &[(&str, f64)]) -> f64 {
        let mut env = Environment::new();
        for &(name, val) in bindings {
            env.set(name, val);
        }
        evaluate(expr, &env).unwrap()
    }

    #[test]
    fn partial_x_of_x_times_y() {
        // ∂/∂x (x * y) = y
        let e = Expr::mul(Expr::var("x"), Expr::var("y"));
        let dx = partial_derivative(&e, "x").unwrap();
        let val = eval_at(&dx, &[("x", 3.0), ("y", 7.0)]);
        assert!((val - 7.0).abs() < 1e-10);
    }

    #[test]
    fn partial_y_of_x_times_y() {
        // ∂/∂y (x * y) = x
        let e = Expr::mul(Expr::var("x"), Expr::var("y"));
        let dy = partial_derivative(&e, "y").unwrap();
        let val = eval_at(&dy, &[("x", 3.0), ("y", 7.0)]);
        assert!((val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn second_partial_derivative() {
        // f = x^3, d²f/dx² = 6x, at x=2 → 12
        let e = Expr::pow(Expr::var("x"), Expr::num(3.0));
        let d2 = nth_partial_derivative(&e, "x", 2).unwrap();
        let val = eval_at(&d2, &[("x", 2.0)]);
        assert!((val - 12.0).abs() < 1e-10);
    }

    #[test]
    fn mixed_partial_xy() {
        // f = x^2 * y^3
        // ∂²f/∂y∂x = ∂/∂y(2x * y^3) = 6x * y^2
        // at (x=1, y=2) → 6*1*4 = 24
        let e = Expr::mul(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::pow(Expr::var("y"), Expr::num(3.0)),
        );
        let d = mixed_partial_derivative(&e, &["x", "y"]).unwrap();
        let val = eval_at(&d, &[("x", 1.0), ("y", 2.0)]);
        assert!((val - 24.0).abs() < 1e-10);
    }

    #[test]
    fn gradient_of_quadratic() {
        // f = x^2 + y^2
        // ∇f = (2x, 2y), at (3, 4) → (6, 8)
        let e = Expr::add(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::pow(Expr::var("y"), Expr::num(2.0)),
        );
        let g = gradient(&e, &["x", "y"]).unwrap();
        assert_eq!(g.len(), 2);

        let gx = eval_at(&g[0], &[("x", 3.0), ("y", 4.0)]);
        let gy = eval_at(&g[1], &[("x", 3.0), ("y", 4.0)]);
        assert!((gx - 6.0).abs() < 1e-10);
        assert!((gy - 8.0).abs() < 1e-10);
    }

    #[test]
    fn jacobian_of_vector_field() {
        // F = (x*y, x+y)
        // J = [[y, x], [1, 1]]
        let f1 = Expr::mul(Expr::var("x"), Expr::var("y"));
        let f2 = Expr::add(Expr::var("x"), Expr::var("y"));
        let j = jacobian(&[f1, f2], &["x", "y"]).unwrap();
        assert_eq!(j.rows, 2);
        assert_eq!(j.cols, 2);

        let bindings = &[("x", 2.0), ("y", 3.0)];
        // J[0][0] = y = 3
        assert!((eval_at(&j.data[0][0], bindings) - 3.0).abs() < 1e-10);
        // J[0][1] = x = 2
        assert!((eval_at(&j.data[0][1], bindings) - 2.0).abs() < 1e-10);
        // J[1][0] = 1
        assert!((eval_at(&j.data[1][0], bindings) - 1.0).abs() < 1e-10);
        // J[1][1] = 1
        assert!((eval_at(&j.data[1][1], bindings) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn hessian_of_quadratic() {
        // f = x^2 + 3*x*y + y^2
        // H = [[2, 3], [3, 2]]
        let e = Expr::add(
            Expr::add(
                Expr::pow(Expr::var("x"), Expr::num(2.0)),
                Expr::mul(
                    Expr::num(3.0),
                    Expr::mul(Expr::var("x"), Expr::var("y")),
                ),
            ),
            Expr::pow(Expr::var("y"), Expr::num(2.0)),
        );
        let h = hessian(&e, &["x", "y"]).unwrap();
        assert_eq!(h.rows, 2);
        assert_eq!(h.cols, 2);

        let bindings = &[("x", 1.0), ("y", 1.0)];
        assert!((eval_at(&h.data[0][0], bindings) - 2.0).abs() < 1e-10);
        assert!((eval_at(&h.data[0][1], bindings) - 3.0).abs() < 1e-10);
        assert!((eval_at(&h.data[1][0], bindings) - 3.0).abs() < 1e-10);
        assert!((eval_at(&h.data[1][1], bindings) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn laplacian_of_harmonic() {
        // f = x^2 + y^2 + z^2
        // ∇²f = 2 + 2 + 2 = 6
        let e = Expr::add(
            Expr::add(
                Expr::pow(Expr::var("x"), Expr::num(2.0)),
                Expr::pow(Expr::var("y"), Expr::num(2.0)),
            ),
            Expr::pow(Expr::var("z"), Expr::num(2.0)),
        );
        let lap = laplacian(&e, &["x", "y", "z"]).unwrap();
        let val = eval_at(&lap, &[("x", 0.0), ("y", 0.0), ("z", 0.0)]);
        assert!((val - 6.0).abs() < 1e-10);
    }

    #[test]
    fn laplacian_of_harmonic_function() {
        // f = x^2 - y^2 is harmonic: ∇²f = 2 - 2 = 0
        let e = Expr::sub(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::pow(Expr::var("y"), Expr::num(2.0)),
        );
        let lap = laplacian(&e, &["x", "y"]).unwrap();
        let val = eval_at(&lap, &[("x", 5.0), ("y", 3.0)]);
        assert!(val.abs() < 1e-10, "harmonic function should have zero laplacian, got {val}");
    }
}
