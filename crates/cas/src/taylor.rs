use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::derivative::differentiate;
use crate::evaluator::{Environment, evaluate};
use crate::simplify::simplify;

/// Compute `n!` as an `f64`.
pub fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, i| acc * i as f64)
}

/// Compute the Taylor (or Maclaurin when `center == 0`) series expansion of
/// `expr` around `center` up to `order` terms.
///
/// Formula: T(x) = sum_{n=0}^{order} f^(n)(a) / n! * (x - a)^n
pub fn taylor_expand(
    expr: &Expr,
    var: &str,
    center: f64,
    order: usize,
) -> Result<Expr, CasError> {
    let mut env = Environment::new();
    env.set(var, center);

    let is_maclaurin = center == 0.0;

    // Build up the polynomial by accumulating terms.
    let mut result: Option<Expr> = None;
    let mut current_deriv = expr.clone();

    for n in 0..=order {
        // Simplify the current derivative before evaluating.
        let simplified = simplify(&current_deriv);

        // Evaluate f^(n)(center).
        let coeff_value = evaluate(&simplified, &env)?;
        let coeff = coeff_value / factorial(n);

        // Skip near-zero coefficients.
        if coeff.abs() > 1e-15 {
            // Build the term: coeff * (x - center)^n
            let term = if n == 0 {
                Expr::num(coeff)
            } else {
                let base = if is_maclaurin {
                    Expr::var(var)
                } else {
                    Expr::sub(Expr::var(var), Expr::num(center))
                };

                let power_expr = if n == 1 {
                    base
                } else {
                    Expr::pow(base, Expr::num(n as f64))
                };

                if (coeff - 1.0).abs() < 1e-15 {
                    power_expr
                } else if (coeff - (-1.0)).abs() < 1e-15 {
                    Expr::neg(power_expr)
                } else {
                    Expr::mul(Expr::num(coeff), power_expr)
                }
            };

            result = Some(match result {
                None => term,
                Some(acc) => Expr::add(acc, term),
            });
        }

        // Compute the next derivative (unless this is the last iteration).
        if n < order {
            current_deriv = differentiate(&current_deriv, var)?;
        }
    }

    // If all coefficients were zero, return 0.
    Ok(result.unwrap_or_else(|| Expr::num(0.0)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: evaluate a Taylor expansion at a given point.
    fn eval_at(expr: &Expr, var: &str, val: f64) -> f64 {
        let mut env = Environment::new();
        env.set(var, val);
        evaluate(expr, &env).unwrap()
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3628800.0);
    }

    #[test]
    fn test_taylor_sin_at_zero_order_5() {
        // sin(x) ~ x - x^3/6 + x^5/120
        let expr = Expr::func("sin", vec![Expr::var("x")]);
        let taylor = taylor_expand(&expr, "x", 0.0, 5).unwrap();

        let approx = eval_at(&taylor, "x", 0.5);
        let exact = 0.5_f64.sin();
        assert!(
            (approx - exact).abs() < 1e-5,
            "sin Taylor at 0.5: approx={}, exact={}",
            approx,
            exact
        );
    }

    #[test]
    fn test_taylor_cos_at_zero_order_4() {
        // cos(x) ~ 1 - x^2/2 + x^4/24
        let expr = Expr::func("cos", vec![Expr::var("x")]);
        let taylor = taylor_expand(&expr, "x", 0.0, 4).unwrap();

        let approx = eval_at(&taylor, "x", 0.5);
        let exact = 0.5_f64.cos();
        assert!(
            (approx - exact).abs() < 1e-4,
            "cos Taylor at 0.5: approx={}, exact={}",
            approx,
            exact
        );
    }

    #[test]
    fn test_taylor_exp_at_zero_order_4() {
        // exp(x) ~ 1 + x + x^2/2 + x^3/6 + x^4/24
        let expr = Expr::func("exp", vec![Expr::var("x")]);
        let taylor = taylor_expand(&expr, "x", 0.0, 4).unwrap();

        let approx = eval_at(&taylor, "x", 1.0);
        let exact = 1.0 + 1.0 + 0.5 + 1.0 / 6.0 + 1.0 / 24.0;
        assert!(
            (approx - exact).abs() < 1e-10,
            "exp Taylor at 1.0: approx={}, exact={}",
            approx,
            exact
        );
    }

    #[test]
    fn test_taylor_polynomial_at_center_1() {
        // f(x) = x^2 + 1 expanded at center=1
        // f(1) = 2, f'(1) = 2, f''(1) = 2
        // Taylor: 2 + 2*(x-1) + (x-1)^2
        // At x=3: 2 + 2*2 + 4 = 10 = 3^2+1 = 10 (exact)
        let expr = Expr::add(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::num(1.0),
        );
        let taylor = taylor_expand(&expr, "x", 1.0, 2).unwrap();

        let approx = eval_at(&taylor, "x", 3.0);
        let exact = 10.0;
        assert!(
            (approx - exact).abs() < 1e-10,
            "polynomial Taylor at 3.0: approx={}, exact={}",
            approx,
            exact
        );
    }

    #[test]
    fn test_taylor_ln_at_center_1_order_3() {
        // ln(x) at center=1, order 3
        // ln(1)=0, f'(1)=1, f''(1)=-1, f'''(1)=2
        // Taylor: (x-1) - (x-1)^2/2 + (x-1)^3/3
        let expr = Expr::func("ln", vec![Expr::var("x")]);
        let taylor = taylor_expand(&expr, "x", 1.0, 3).unwrap();

        let approx = eval_at(&taylor, "x", 1.5);
        let exact = 1.5_f64.ln();
        assert!(
            (approx - exact).abs() < 0.02,
            "ln Taylor at 1.5: approx={}, exact={}",
            approx,
            exact
        );
    }

    #[test]
    fn test_taylor_order_0_gives_constant() {
        let expr = Expr::func("sin", vec![Expr::var("x")]);
        let taylor = taylor_expand(&expr, "x", 0.0, 0).unwrap();

        // sin(0) = 0, so the result should be 0
        let approx = eval_at(&taylor, "x", 42.0);
        assert!(
            approx.abs() < 1e-15,
            "order 0 Taylor of sin at 0 should be 0, got {}",
            approx
        );
    }

    #[test]
    fn test_taylor_error_for_undefined() {
        // ln(x) at x=0 is undefined
        let expr = Expr::func("ln", vec![Expr::var("x")]);
        let result = taylor_expand(&expr, "x", 0.0, 3);
        assert!(result.is_err(), "ln(x) at center=0 should produce an error");
    }
}
