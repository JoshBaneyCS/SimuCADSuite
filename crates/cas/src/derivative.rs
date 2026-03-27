use simucad_core::error::CasError;

use crate::ast::{BinOp, Expr, UnaryOp};

/// Symbolically differentiate `expr` with respect to the variable `var`.
///
/// Returns an unsimplified expression tree. Use [`crate::simplify::simplify`]
/// to clean up the result.
pub fn differentiate(expr: &Expr, var: &str) -> Result<Expr, CasError> {
    match expr {
        // d/dx(c) = 0
        Expr::Num(_) => Ok(Expr::num(0.0)),

        // d/dx(x) = 1,  d/dx(y) = 0  for y != x
        Expr::Var(name) => {
            if name == var {
                Ok(Expr::num(1.0))
            } else {
                Ok(Expr::num(0.0))
            }
        }

        Expr::BinOp { op, lhs, rhs } => match op {
            // Sum / difference rule
            BinOp::Add => {
                let dl = differentiate(lhs, var)?;
                let dr = differentiate(rhs, var)?;
                Ok(Expr::add(dl, dr))
            }
            BinOp::Sub => {
                let dl = differentiate(lhs, var)?;
                let dr = differentiate(rhs, var)?;
                Ok(Expr::sub(dl, dr))
            }

            // Product rule: d(f*g) = f'*g + f*g'
            BinOp::Mul => {
                let dl = differentiate(lhs, var)?;
                let dr = differentiate(rhs, var)?;
                Ok(Expr::add(
                    Expr::mul(dl, rhs.as_ref().clone()),
                    Expr::mul(lhs.as_ref().clone(), dr),
                ))
            }

            // Quotient rule: d(f/g) = (f'*g - f*g') / g^2
            BinOp::Div => {
                let dl = differentiate(lhs, var)?;
                let dr = differentiate(rhs, var)?;
                let numerator = Expr::sub(
                    Expr::mul(dl, rhs.as_ref().clone()),
                    Expr::mul(lhs.as_ref().clone(), dr),
                );
                let denominator = Expr::pow(rhs.as_ref().clone(), Expr::num(2.0));
                Ok(Expr::div(numerator, denominator))
            }

            // Power rule (general): d(f^g) = f^g * (g' * ln(f) + g * f'/f)
            // Special case: when g is a constant, d(f^n) = n * f^(n-1) * f'
            BinOp::Pow => {
                let base = lhs.as_ref();
                let exp = rhs.as_ref();

                // Check if exponent is constant w.r.t. var
                if !contains_var(exp, var) {
                    // d(f^n) = n * f^(n-1) * f'
                    let df = differentiate(base, var)?;
                    Ok(Expr::mul(
                        Expr::mul(
                            exp.clone(),
                            Expr::pow(base.clone(), Expr::sub(exp.clone(), Expr::num(1.0))),
                        ),
                        df,
                    ))
                } else if !contains_var(base, var) {
                    // d(a^g) = a^g * ln(a) * g'
                    let dg = differentiate(exp, var)?;
                    Ok(Expr::mul(
                        Expr::mul(
                            Expr::pow(base.clone(), exp.clone()),
                            Expr::func("log", vec![base.clone()]),
                        ),
                        dg,
                    ))
                } else {
                    // General case: d(f^g) = f^g * (g' * ln(f) + g * f'/f)
                    let df = differentiate(base, var)?;
                    let dg = differentiate(exp, var)?;
                    let term1 = Expr::mul(dg, Expr::func("log", vec![base.clone()]));
                    let term2 = Expr::mul(exp.clone(), Expr::div(df, base.clone()));
                    Ok(Expr::mul(
                        Expr::pow(base.clone(), exp.clone()),
                        Expr::add(term1, term2),
                    ))
                }
            }
        },

        // d/dx(-f) = -(f')
        Expr::UnaryOp {
            op: UnaryOp::Neg,
            operand,
        } => {
            let d = differentiate(operand, var)?;
            Ok(Expr::neg(d))
        }

        // Chain rule for known functions
        Expr::Func { name, args } => {
            if args.len() != 1 {
                return Err(CasError::UnsupportedOperation(format!(
                    "differentiation of multi-argument function '{}' is not supported",
                    name
                )));
            }
            let inner = &args[0];
            let d_inner = differentiate(inner, var)?;

            let outer_deriv = match name.as_str() {
                // d/du sin(u) = cos(u)
                "sin" => Expr::func("cos", vec![inner.clone()]),

                // d/du cos(u) = -sin(u)
                "cos" => Expr::neg(Expr::func("sin", vec![inner.clone()])),

                // d/du tan(u) = 1 / cos(u)^2
                "tan" => Expr::div(
                    Expr::num(1.0),
                    Expr::pow(Expr::func("cos", vec![inner.clone()]), Expr::num(2.0)),
                ),

                // d/du exp(u) = exp(u)
                "exp" => Expr::func("exp", vec![inner.clone()]),

                // d/du log(u) = 1/u
                "log" | "ln" => Expr::div(Expr::num(1.0), inner.clone()),

                // d/du sqrt(u) = 1 / (2 * sqrt(u))
                "sqrt" => Expr::div(
                    Expr::num(1.0),
                    Expr::mul(Expr::num(2.0), Expr::func("sqrt", vec![inner.clone()])),
                ),

                // d/du sinh(u) = cosh(u)
                "sinh" => Expr::func("cosh", vec![inner.clone()]),

                // d/du cosh(u) = sinh(u)
                "cosh" => Expr::func("sinh", vec![inner.clone()]),

                // d/du tanh(u) = 1 / cosh(u)^2
                "tanh" => Expr::div(
                    Expr::num(1.0),
                    Expr::pow(Expr::func("cosh", vec![inner.clone()]), Expr::num(2.0)),
                ),

                // d/du asinh(u) = 1 / sqrt(u^2 + 1)
                "asinh" => Expr::div(
                    Expr::num(1.0),
                    Expr::func(
                        "sqrt",
                        vec![Expr::add(
                            Expr::pow(inner.clone(), Expr::num(2.0)),
                            Expr::num(1.0),
                        )],
                    ),
                ),

                // d/du acosh(u) = 1 / sqrt(u^2 - 1)
                "acosh" => Expr::div(
                    Expr::num(1.0),
                    Expr::func(
                        "sqrt",
                        vec![Expr::sub(
                            Expr::pow(inner.clone(), Expr::num(2.0)),
                            Expr::num(1.0),
                        )],
                    ),
                ),

                // d/du atanh(u) = 1 / (1 - u^2)
                "atanh" => Expr::div(
                    Expr::num(1.0),
                    Expr::sub(
                        Expr::num(1.0),
                        Expr::pow(inner.clone(), Expr::num(2.0)),
                    ),
                ),

                // d/du abs(u) = sign(u)
                "abs" => Expr::func("sign", vec![inner.clone()]),

                _ => {
                    return Err(CasError::UnsupportedOperation(format!(
                        "differentiation of function '{}' is not supported",
                        name
                    )));
                }
            };

            // Chain rule: outer_deriv * d_inner
            Ok(Expr::mul(outer_deriv, d_inner))
        }
    }
}

/// Returns `true` if `expr` contains the variable `var`.
fn contains_var(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Num(_) => false,
        Expr::Var(name) => name == var,
        Expr::BinOp { lhs, rhs, .. } => contains_var(lhs, var) || contains_var(rhs, var),
        Expr::UnaryOp { operand, .. } => contains_var(operand, var),
        Expr::Func { args, .. } => args.iter().any(|a| contains_var(a, var)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::{evaluate, Environment};
    use crate::simplify::simplify;

    /// Helper: parse, differentiate, simplify, then evaluate at x = val.
    fn diff_eval(expr: &Expr, var: &str, val: f64) -> f64 {
        let d = differentiate(expr, var).unwrap();
        let s = simplify(&d);
        let mut env = Environment::new();
        env.set(var, val);
        evaluate(&s, &env).unwrap()
    }

    #[test]
    fn test_diff_constant() {
        let d = differentiate(&Expr::num(5.0), "x").unwrap();
        assert_eq!(d, Expr::num(0.0));
    }

    #[test]
    fn test_diff_var() {
        let d = differentiate(&Expr::var("x"), "x").unwrap();
        assert_eq!(d, Expr::num(1.0));
    }

    #[test]
    fn test_diff_other_var() {
        let d = differentiate(&Expr::var("y"), "x").unwrap();
        assert_eq!(d, Expr::num(0.0));
    }

    #[test]
    fn test_diff_sum() {
        // d/dx(x + 5) = 1
        let e = Expr::add(Expr::var("x"), Expr::num(5.0));
        let val = diff_eval(&e, "x", 1.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_product() {
        // d/dx(x * x) = 2x,  at x=3 → 6
        let e = Expr::mul(Expr::var("x"), Expr::var("x"));
        let val = diff_eval(&e, "x", 3.0);
        assert!((val - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_quotient() {
        // d/dx(1/x) = -1/x^2,  at x=2 → -0.25
        let e = Expr::div(Expr::num(1.0), Expr::var("x"));
        let val = diff_eval(&e, "x", 2.0);
        assert!((val - (-0.25)).abs() < 1e-10);
    }

    #[test]
    fn test_diff_power() {
        // d/dx(x^3) = 3x^2,  at x=2 → 12
        let e = Expr::pow(Expr::var("x"), Expr::num(3.0));
        let val = diff_eval(&e, "x", 2.0);
        assert!((val - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sin() {
        // d/dx(sin(x)) = cos(x),  at x=0 → 1
        let e = Expr::func("sin", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 0.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_cos() {
        // d/dx(cos(x)) = -sin(x),  at x=0 → 0
        let e = Expr::func("cos", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 0.0);
        assert!(val.abs() < 1e-10);
    }

    #[test]
    fn test_diff_exp() {
        // d/dx(exp(x)) = exp(x),  at x=1 → e
        let e = Expr::func("exp", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 1.0);
        assert!((val - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_diff_log() {
        // d/dx(log(x)) = 1/x,  at x=4 → 0.25
        let e = Expr::func("log", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 4.0);
        assert!((val - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sqrt() {
        // d/dx(sqrt(x)) = 1/(2*sqrt(x)),  at x=4 → 0.25
        let e = Expr::func("sqrt", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 4.0);
        assert!((val - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_diff_chain_rule() {
        // d/dx(sin(x^2)) = cos(x^2) * 2x,  at x=0 → 0
        let e = Expr::func("sin", vec![Expr::pow(Expr::var("x"), Expr::num(2.0))]);
        let val = diff_eval(&e, "x", 0.0);
        assert!(val.abs() < 1e-10);
    }

    #[test]
    fn test_diff_neg() {
        // d/dx(-x) = -1
        let e = Expr::neg(Expr::var("x"));
        let d = simplify(&differentiate(&e, "x").unwrap());
        let env = Environment::new();
        // Should simplify to -1
        let val = evaluate(&d, &env).unwrap();
        assert!((val - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_diff_exponential_base() {
        // d/dx(2^x) = 2^x * ln(2),  at x=0 → ln(2) ≈ 0.693
        let e = Expr::pow(Expr::num(2.0), Expr::var("x"));
        let val = diff_eval(&e, "x", 0.0);
        assert!((val - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sinh() {
        // d/dx(sinh(x)) = cosh(x), at x=0 → cosh(0) = 1
        let e = Expr::func("sinh", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 0.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_cosh() {
        // d/dx(cosh(x)) = sinh(x), at x=0 → sinh(0) = 0
        let e = Expr::func("cosh", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 0.0);
        assert!(val.abs() < 1e-10);
    }

    #[test]
    fn test_diff_tanh_numerically() {
        // d/dx(tanh(x)) = 1/cosh²(x), verify numerically at x=0.5
        let e = Expr::func("tanh", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 0.5);
        let expected = 1.0 / (0.5_f64.cosh().powi(2));
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sinh_chain_rule() {
        // d/dx(sinh(x²)) = cosh(x²) * 2x, verify numerically at x=1
        let e = Expr::func("sinh", vec![Expr::pow(Expr::var("x"), Expr::num(2.0))]);
        let val = diff_eval(&e, "x", 1.0);
        let expected = 1.0_f64.cosh() * 2.0;
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_diff_asinh() {
        // d/dx(asinh(x)) = 1/sqrt(x²+1), at x=0 → 1
        let e = Expr::func("asinh", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 0.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_acosh() {
        // d/dx(acosh(x)) = 1/sqrt(x²-1), at x=2 → 1/sqrt(3)
        let e = Expr::func("acosh", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 2.0);
        let expected = 1.0 / 3.0_f64.sqrt();
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_diff_atanh() {
        // d/dx(atanh(x)) = 1/(1-x²), at x=0.5 → 1/(1-0.25) = 4/3
        let e = Expr::func("atanh", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 0.5);
        let expected = 1.0 / (1.0 - 0.25);
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_diff_abs() {
        // d/dx(abs(x)) = sign(x), at x=3 → 1
        let e = Expr::func("abs", vec![Expr::var("x")]);
        let val = diff_eval(&e, "x", 3.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_unsupported_func() {
        let e = Expr::func("foobar", vec![Expr::var("x")]);
        assert!(matches!(
            differentiate(&e, "x"),
            Err(CasError::UnsupportedOperation(_))
        ));
    }

    #[test]
    fn test_diff_polynomial_numerical() {
        // d/dx(x^3 + 2*x^2 + x + 1) = 3x^2 + 4x + 1,  at x=2 → 21
        let e = Expr::add(
            Expr::add(
                Expr::add(
                    Expr::pow(Expr::var("x"), Expr::num(3.0)),
                    Expr::mul(Expr::num(2.0), Expr::pow(Expr::var("x"), Expr::num(2.0))),
                ),
                Expr::var("x"),
            ),
            Expr::num(1.0),
        );
        let val = diff_eval(&e, "x", 2.0);
        assert!((val - 21.0).abs() < 1e-10);
    }
}
