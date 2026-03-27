use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::evaluator::{Environment, evaluate};
use crate::simplify::simplify;

/// Solve `expr = 0` for `var` when `expr` is linear in `var`.
///
/// Strategy: evaluate at `var=0` and `var=1` to extract the line `f(x) = a*x + b`,
/// then solve `a*x + b = 0` => `x = -b/a`.
pub fn solve_linear(expr: &Expr, var: &str) -> Result<Expr, CasError> {
    let simplified = simplify(expr);

    let mut env = Environment::new();

    // f(0) = b
    env.set(var, 0.0);
    let b = evaluate(&simplified, &env)?;

    // f(1) = a + b
    env.set(var, 1.0);
    let f1 = evaluate(&simplified, &env)?;

    let a = f1 - b;

    if a.abs() < 1e-15 {
        return Err(CasError::UnsupportedOperation(
            "expression is constant in the variable — no unique solution".into(),
        ));
    }

    // x = -b / a
    let root = -b / a;
    Ok(Expr::num(root))
}

/// Solve `expr = 0` for `var` when `expr` is quadratic.
///
/// Strategy: evaluate at three points (`var=0`, `var=1`, `var=-1`) to extract
/// coefficients `a`, `b`, `c` of `a*x^2 + b*x + c`, then apply the quadratic
/// formula.
pub fn solve_quadratic(expr: &Expr, var: &str) -> Result<Vec<Expr>, CasError> {
    let simplified = simplify(expr);
    let mut env = Environment::new();

    // f(0) = c
    env.set(var, 0.0);
    let f0 = evaluate(&simplified, &env)?;

    // f(1) = a + b + c
    env.set(var, 1.0);
    let f1 = evaluate(&simplified, &env)?;

    // f(-1) = a - b + c
    env.set(var, -1.0);
    let fm1 = evaluate(&simplified, &env)?;

    // From the system:
    //   f(0)  = c
    //   f(1)  = a + b + c
    //   f(-1) = a - b + c
    let c = f0;
    let a = (f1 + fm1) / 2.0 - c;
    let b = (f1 - fm1) / 2.0;

    // If a is approximately zero, delegate to linear.
    if a.abs() < 1e-12 {
        let root = solve_linear(expr, var)?;
        return Ok(vec![root]);
    }

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < -1e-12 {
        return Err(CasError::DomainError("no real roots".into()));
    }

    if discriminant.abs() < 1e-12 {
        // Double root.
        let root = -b / (2.0 * a);
        return Ok(vec![Expr::num(root)]);
    }

    let sqrt_disc = discriminant.sqrt();
    let r1 = (-b - sqrt_disc) / (2.0 * a);
    let r2 = (-b + sqrt_disc) / (2.0 * a);

    let mut roots = vec![r1, r2];
    roots.sort_by(|x, y| x.partial_cmp(y).unwrap());

    Ok(roots.into_iter().map(Expr::num).collect())
}

/// Solve `expr = 0` for `var` given that `expr` is a polynomial of the
/// specified `degree`.
///
/// For degree 1 and 2, delegates to the specialised solvers. For higher
/// degrees the coefficients are extracted via a Vandermonde system and roots
/// are found numerically using `solver::find_roots`.
pub fn solve_polynomial(
    expr: &Expr,
    var: &str,
    degree: usize,
) -> Result<Vec<Expr>, CasError> {
    match degree {
        0 => Err(CasError::UnsupportedOperation(
            "cannot solve a degree-0 polynomial".into(),
        )),
        1 => {
            let root = solve_linear(expr, var)?;
            Ok(vec![root])
        }
        2 => solve_quadratic(expr, var),
        _ => {
            // For degree > 2 use numeric root finding.
            let roots = crate::solver::find_roots(expr, var, -100.0, 100.0, 10_000)?;
            Ok(roots.into_iter().map(Expr::num).collect())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to get the f64 value out of an Expr::Num.
    fn num_val(e: &Expr) -> f64 {
        match e {
            Expr::Num(v) => *v,
            other => panic!("expected Num, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // solve_linear
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_linear_2x_plus_6() {
        // 2*x + 6 = 0  =>  x = -3
        let expr = Expr::add(
            Expr::mul(Expr::num(2.0), Expr::var("x")),
            Expr::num(6.0),
        );
        let root = solve_linear(&expr, "x").unwrap();
        assert!(
            (num_val(&root) - (-3.0)).abs() < 1e-10,
            "expected -3, got {}",
            num_val(&root)
        );
    }

    #[test]
    fn test_solve_linear_x_minus_5() {
        // x - 5 = 0  =>  x = 5
        let expr = Expr::sub(Expr::var("x"), Expr::num(5.0));
        let root = solve_linear(&expr, "x").unwrap();
        assert!(
            (num_val(&root) - 5.0).abs() < 1e-10,
            "expected 5, got {}",
            num_val(&root)
        );
    }

    #[test]
    fn test_solve_linear_constant_error() {
        // 3 = 0  =>  no variable, should error
        let expr = Expr::num(3.0);
        let result = solve_linear(&expr, "x");
        assert!(result.is_err(), "constant expression should fail");
    }

    // -----------------------------------------------------------------------
    // solve_quadratic
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_quadratic_x2_minus_4() {
        // x^2 - 4 = 0  =>  x = -2, 2
        let expr = Expr::sub(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::num(4.0),
        );
        let roots = solve_quadratic(&expr, "x").unwrap();
        assert_eq!(roots.len(), 2);

        let vals: Vec<f64> = roots.iter().map(num_val).collect();
        assert!(
            (vals[0] - (-2.0)).abs() < 1e-10,
            "expected -2, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 2.0).abs() < 1e-10,
            "expected 2, got {}",
            vals[1]
        );
    }

    #[test]
    fn test_solve_quadratic_double_root() {
        // x^2 + 2*x + 1 = 0  =>  x = -1  (double root)
        let expr = Expr::add(
            Expr::add(
                Expr::pow(Expr::var("x"), Expr::num(2.0)),
                Expr::mul(Expr::num(2.0), Expr::var("x")),
            ),
            Expr::num(1.0),
        );
        let roots = solve_quadratic(&expr, "x").unwrap();
        assert_eq!(roots.len(), 1, "expected 1 root (double), got {:?}", roots);
        assert!(
            (num_val(&roots[0]) - (-1.0)).abs() < 1e-10,
            "expected -1, got {}",
            num_val(&roots[0])
        );
    }

    #[test]
    fn test_solve_quadratic_no_real_roots() {
        // x^2 + 1 = 0  =>  no real roots
        let expr = Expr::add(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::num(1.0),
        );
        let result = solve_quadratic(&expr, "x");
        assert!(result.is_err(), "x^2+1=0 should have no real roots");
    }

    #[test]
    fn test_solve_quadratic_x2_minus_5x_plus_6() {
        // x^2 - 5*x + 6 = 0  =>  x = 2, 3
        let expr = Expr::add(
            Expr::sub(
                Expr::pow(Expr::var("x"), Expr::num(2.0)),
                Expr::mul(Expr::num(5.0), Expr::var("x")),
            ),
            Expr::num(6.0),
        );
        let roots = solve_quadratic(&expr, "x").unwrap();
        assert_eq!(roots.len(), 2);

        let vals: Vec<f64> = roots.iter().map(num_val).collect();
        assert!(
            (vals[0] - 2.0).abs() < 1e-10,
            "expected 2, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 3.0).abs() < 1e-10,
            "expected 3, got {}",
            vals[1]
        );
    }
}
