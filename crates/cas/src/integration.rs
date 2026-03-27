use simucad_core::error::CasError;

use crate::ast::{BinOp, Expr, UnaryOp};

/// Attempt symbolic integration of `expr` with respect to `var`.
/// Returns `Err` if the integral cannot be found symbolically.
pub fn integrate(expr: &Expr, var: &str) -> Result<Expr, CasError> {
    match expr {
        // ∫c dx = c*x
        Expr::Num(_) => Ok(Expr::mul(expr.clone(), Expr::var(var))),

        Expr::Var(name) => {
            if name == var {
                // ∫x dx = x^2 / 2
                Ok(Expr::div(
                    Expr::pow(Expr::var(var), Expr::num(2.0)),
                    Expr::num(2.0),
                ))
            } else {
                // Treat as constant w.r.t. var: ∫c dx = c*x
                Ok(Expr::mul(expr.clone(), Expr::var(var)))
            }
        }

        Expr::UnaryOp {
            op: UnaryOp::Neg,
            operand,
        } => {
            // ∫-f dx = -(∫f dx)
            let inner = integrate(operand, var)?;
            Ok(Expr::neg(inner))
        }

        Expr::BinOp { op, lhs, rhs } => match op {
            // ∫(f + g) dx = ∫f dx + ∫g dx
            BinOp::Add => {
                let il = integrate(lhs, var)?;
                let ir = integrate(rhs, var)?;
                Ok(Expr::add(il, ir))
            }
            // ∫(f - g) dx = ∫f dx - ∫g dx
            BinOp::Sub => {
                let il = integrate(lhs, var)?;
                let ir = integrate(rhs, var)?;
                Ok(Expr::sub(il, ir))
            }
            // Constant multiple: ∫c*f dx = c * ∫f dx
            BinOp::Mul => {
                let l_has_var = contains_var(lhs, var);
                let r_has_var = contains_var(rhs, var);

                if !l_has_var && !r_has_var {
                    // Both constant: ∫c dx = c*x
                    Ok(Expr::mul(
                        Expr::mul(lhs.as_ref().clone(), rhs.as_ref().clone()),
                        Expr::var(var),
                    ))
                } else if !l_has_var {
                    // c * f(x): ∫c*f dx = c * ∫f dx
                    let ir = integrate(rhs, var)?;
                    Ok(Expr::mul(lhs.as_ref().clone(), ir))
                } else if !r_has_var {
                    // f(x) * c: ∫f*c dx = c * ∫f dx
                    let il = integrate(lhs, var)?;
                    Ok(Expr::mul(rhs.as_ref().clone(), il))
                } else {
                    Err(CasError::UnsupportedOperation(
                        "integration of product of two variable expressions is not supported"
                            .into(),
                    ))
                }
            }
            // ∫f/g — handle constant denominator and 1/x
            BinOp::Div => {
                let l_has_var = contains_var(lhs, var);
                let r_has_var = contains_var(rhs, var);

                if !l_has_var && !r_has_var {
                    // constant / constant
                    Ok(Expr::mul(
                        Expr::div(lhs.as_ref().clone(), rhs.as_ref().clone()),
                        Expr::var(var),
                    ))
                } else if l_has_var && !r_has_var {
                    // f(x) / c = (1/c) * f(x)
                    let il = integrate(lhs, var)?;
                    Ok(Expr::div(il, rhs.as_ref().clone()))
                } else if !l_has_var && r_has_var {
                    // c / f(x) — only handle c / x => c * ln(x)
                    if let Expr::Var(name) = rhs.as_ref() {
                        if name == var {
                            // ∫c/x dx = c * ln(x)
                            return Ok(Expr::mul(
                                lhs.as_ref().clone(),
                                Expr::func("ln", vec![Expr::var(var)]),
                            ));
                        }
                    }
                    Err(CasError::UnsupportedOperation(
                        "integration of constant/f(x) for non-trivial f is not supported".into(),
                    ))
                } else {
                    // Both have var — check for 1/x pattern where lhs is Num(1) and rhs is Var
                    // This is handled above, so this is the general unsupported case
                    Err(CasError::UnsupportedOperation(
                        "integration of variable/variable expressions is not supported".into(),
                    ))
                }
            }
            // Power rule: ∫x^n dx = x^(n+1)/(n+1) for n ≠ -1
            BinOp::Pow => {
                let base_has_var = contains_var(lhs, var);
                let exp_has_var = contains_var(rhs, var);

                if !base_has_var && !exp_has_var {
                    // constant^constant is just a constant
                    Ok(Expr::mul(
                        Expr::pow(lhs.as_ref().clone(), rhs.as_ref().clone()),
                        Expr::var(var),
                    ))
                } else if base_has_var && !exp_has_var {
                    // f(x)^n — only handle x^n (simple variable)
                    if let Expr::Var(name) = lhs.as_ref() {
                        if name == var {
                            // Check for n == -1
                            if let Expr::Num(n) = rhs.as_ref() {
                                if (*n - (-1.0)).abs() < 1e-15 {
                                    // ∫x^(-1) dx = ln(x)
                                    return Ok(Expr::func("ln", vec![Expr::var(var)]));
                                }
                            }
                            // ∫x^n dx = x^(n+1)/(n+1)
                            let n_plus_1 =
                                Expr::add(rhs.as_ref().clone(), Expr::num(1.0));
                            return Ok(Expr::div(
                                Expr::pow(Expr::var(var), n_plus_1.clone()),
                                n_plus_1,
                            ));
                        }
                    }
                    Err(CasError::UnsupportedOperation(
                        "integration of f(x)^n for non-trivial f is not supported".into(),
                    ))
                } else if !base_has_var && exp_has_var {
                    // a^x where a is a constant — only handle e^x = exp(x)
                    if let Expr::Var(name) = rhs.as_ref() {
                        if name == var {
                            if let Expr::Var(base_name) = lhs.as_ref() {
                                if base_name == "e" {
                                    // ∫e^x dx = e^x
                                    return Ok(Expr::pow(
                                        Expr::var("e"),
                                        Expr::var(var),
                                    ));
                                }
                            }
                        }
                    }
                    Err(CasError::UnsupportedOperation(
                        "integration of a^f(x) is not supported".into(),
                    ))
                } else {
                    Err(CasError::UnsupportedOperation(
                        "integration of f(x)^g(x) is not supported".into(),
                    ))
                }
            }
        },

        Expr::Func { name, args } => {
            if args.len() != 1 {
                return Err(CasError::UnsupportedOperation(
                    "integration of multi-argument functions is not supported".into(),
                ));
            }
            let inner = &args[0];

            // Only handle f(x) where inner is just the variable
            if let Expr::Var(inner_name) = inner {
                if inner_name == var {
                    return match name.as_str() {
                        // ∫sin(x) dx = -cos(x)
                        "sin" => Ok(Expr::neg(Expr::func("cos", vec![Expr::var(var)]))),
                        // ∫cos(x) dx = sin(x)
                        "cos" => Ok(Expr::func("sin", vec![Expr::var(var)])),
                        // ∫exp(x) dx = exp(x)
                        "exp" => Ok(Expr::func("exp", vec![Expr::var(var)])),
                        _ => Err(CasError::UnsupportedOperation(format!(
                            "integration of function '{}' is not supported",
                            name
                        ))),
                    };
                }
            }

            // Constant argument — treat function as constant
            if !contains_var(inner, var) {
                return Ok(Expr::mul(expr.clone(), Expr::var(var)));
            }

            Err(CasError::UnsupportedOperation(format!(
                "integration of {}(f(x)) for non-trivial f is not supported",
                name
            )))
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
    use crate::derivative::differentiate;
    use crate::evaluator::{evaluate, Environment};
    use crate::simplify::simplify;

    /// Helper: evaluate an expression at a given x value.
    fn eval_at(expr: &Expr, var: &str, val: f64) -> f64 {
        let mut env = Environment::new();
        env.set(var, val);
        evaluate(expr, &env).unwrap()
    }

    #[test]
    fn test_integrate_constant() {
        // ∫5 dx = 5*x
        let result = integrate(&Expr::num(5.0), "x").unwrap();
        let s = simplify(&result);
        // At x=3: 5*3 = 15
        assert!((eval_at(&s, "x", 3.0) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_variable() {
        // ∫x dx = x^2/2
        let result = integrate(&Expr::var("x"), "x").unwrap();
        let s = simplify(&result);
        // At x=4: 16/2 = 8
        assert!((eval_at(&s, "x", 4.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_other_var_as_constant() {
        // ∫y dx = y*x
        let result = integrate(&Expr::var("y"), "x").unwrap();
        let s = simplify(&result);
        let mut env = Environment::new();
        env.set("x", 3.0);
        env.set("y", 2.0);
        assert!((evaluate(&s, &env).unwrap() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_power() {
        // ∫x^3 dx = x^4/4
        let expr = Expr::pow(Expr::var("x"), Expr::num(3.0));
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        // At x=2: 16/4 = 4
        assert!((eval_at(&s, "x", 2.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_reciprocal() {
        // ∫x^(-1) dx = ln(x)
        let expr = Expr::pow(Expr::var("x"), Expr::num(-1.0));
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        // At x=e: ln(e) = 1
        assert!((eval_at(&s, "x", std::f64::consts::E) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_one_over_x() {
        // ∫1/x dx = ln(x)  (via Div node)
        let expr = Expr::div(Expr::num(1.0), Expr::var("x"));
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        assert!((eval_at(&s, "x", std::f64::consts::E) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_sin() {
        // ∫sin(x) dx = -cos(x)
        let expr = Expr::func("sin", vec![Expr::var("x")]);
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        // At x=0: -cos(0) = -1
        assert!((eval_at(&s, "x", 0.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_cos() {
        // ∫cos(x) dx = sin(x)
        let expr = Expr::func("cos", vec![Expr::var("x")]);
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        // At x=pi/2: sin(pi/2) = 1
        assert!((eval_at(&s, "x", std::f64::consts::FRAC_PI_2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_exp() {
        // ∫exp(x) dx = exp(x)
        let expr = Expr::func("exp", vec![Expr::var("x")]);
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        // At x=1: exp(1) = e
        assert!((eval_at(&s, "x", 1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_sum() {
        // ∫(x + 1) dx = x^2/2 + x
        let expr = Expr::add(Expr::var("x"), Expr::num(1.0));
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        // At x=4: 16/2 + 4 = 12
        assert!((eval_at(&s, "x", 4.0) - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_difference() {
        // ∫(x - 1) dx = x^2/2 - x
        let expr = Expr::sub(Expr::var("x"), Expr::num(1.0));
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        // At x=4: 16/2 - 4 = 4
        assert!((eval_at(&s, "x", 4.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_constant_multiple() {
        // ∫3*x dx = 3 * x^2/2
        let expr = Expr::mul(Expr::num(3.0), Expr::var("x"));
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        // At x=4: 3*16/2 = 24
        assert!((eval_at(&s, "x", 4.0) - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_negation() {
        // ∫-x dx = -x^2/2
        let expr = Expr::neg(Expr::var("x"));
        let result = integrate(&expr, "x").unwrap();
        let s = simplify(&result);
        // At x=4: -16/2 = -8
        assert!((eval_at(&s, "x", 4.0) - (-8.0)).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_unsupported() {
        // ∫x*x dx — product of two variable expressions
        let expr = Expr::mul(Expr::var("x"), Expr::var("x"));
        assert!(integrate(&expr, "x").is_err());
    }

    #[test]
    fn test_integrate_then_differentiate_x_squared() {
        // d/dx(∫x dx) should give back x (up to simplification)
        let expr = Expr::var("x");
        let integral = integrate(&expr, "x").unwrap();
        let derivative = differentiate(&integral, "x").unwrap();
        let s = simplify(&derivative);
        // Should evaluate to x at any point
        assert!((eval_at(&s, "x", 3.0) - 3.0).abs() < 1e-10);
        assert!((eval_at(&s, "x", 7.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_then_differentiate_sin() {
        // d/dx(∫sin(x) dx) = d/dx(-cos(x)) = sin(x)
        let expr = Expr::func("sin", vec![Expr::var("x")]);
        let integral = integrate(&expr, "x").unwrap();
        let derivative = differentiate(&integral, "x").unwrap();
        let s = simplify(&derivative);
        // sin(pi/6) = 0.5
        assert!((eval_at(&s, "x", std::f64::consts::FRAC_PI_6) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_then_differentiate_exp() {
        // d/dx(∫exp(x) dx) = exp(x)
        let expr = Expr::func("exp", vec![Expr::var("x")]);
        let integral = integrate(&expr, "x").unwrap();
        let derivative = differentiate(&integral, "x").unwrap();
        let s = simplify(&derivative);
        assert!((eval_at(&s, "x", 1.0) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_then_differentiate_power() {
        // d/dx(∫x^3 dx) = x^3
        let expr = Expr::pow(Expr::var("x"), Expr::num(3.0));
        let integral = integrate(&expr, "x").unwrap();
        let derivative = differentiate(&integral, "x").unwrap();
        let s = simplify(&derivative);
        // At x=2: 2^3 = 8
        assert!((eval_at(&s, "x", 2.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_then_differentiate_constant() {
        // d/dx(∫5 dx) = 5
        let expr = Expr::num(5.0);
        let integral = integrate(&expr, "x").unwrap();
        let derivative = differentiate(&integral, "x").unwrap();
        let s = simplify(&derivative);
        assert!((eval_at(&s, "x", 42.0) - 5.0).abs() < 1e-10);
    }
}
