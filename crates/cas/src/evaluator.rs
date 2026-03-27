use std::collections::HashMap;

use simucad_core::error::CasError;

use crate::ast::{BinOp, Expr, UnaryOp};

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// A set of variable bindings used during numeric evaluation.
#[derive(Debug, Clone, Default)]
pub struct Environment {
    vars: HashMap<String, f64>,
}

impl Environment {
    /// Create an empty environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a variable binding.
    pub fn set(&mut self, name: impl Into<String>, value: f64) {
        self.vars.insert(name.into(), value);
    }

    /// Look up a variable, returning `None` if it is not bound.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.vars.get(name).copied()
    }

    /// Create an environment from an iterator of `(name, value)` pairs.
    pub fn from_iter(iter: impl IntoIterator<Item = (String, f64)>) -> Self {
        Self {
            vars: iter.into_iter().collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Evaluate an expression tree to a numeric value given an environment of
/// variable bindings.
///
/// Built-in constants `pi` and `e` are recognised automatically.
pub fn evaluate(expr: &Expr, env: &Environment) -> Result<f64, CasError> {
    match expr {
        Expr::Num(n) => Ok(*n),

        Expr::Var(name) => {
            // Built-in constants
            match name.as_str() {
                "pi" => Ok(std::f64::consts::PI),
                "e" => Ok(std::f64::consts::E),
                _ => env.get(name).ok_or_else(|| {
                    CasError::UndefinedVariable(name.clone())
                }),
            }
        }

        Expr::BinOp { op, lhs, rhs } => {
            let l = evaluate(lhs, env)?;
            let r = evaluate(rhs, env)?;
            match op {
                BinOp::Add => Ok(l + r),
                BinOp::Sub => Ok(l - r),
                BinOp::Mul => Ok(l * r),
                BinOp::Div => {
                    if r == 0.0 {
                        Err(CasError::DivisionByZero)
                    } else {
                        Ok(l / r)
                    }
                }
                BinOp::Pow => Ok(l.powf(r)),
            }
        }

        Expr::UnaryOp {
            op: UnaryOp::Neg,
            operand,
        } => {
            let v = evaluate(operand, env)?;
            Ok(-v)
        }

        Expr::Func { name, args } => {
            if args.is_empty() {
                return Err(CasError::ParseError {
                    position: 0,
                    message: format!("function '{}' requires at least one argument", name),
                });
            }
            let arg = evaluate(&args[0], env)?;
            match name.as_str() {
                "sin" => Ok(arg.sin()),
                "cos" => Ok(arg.cos()),
                "tan" => Ok(arg.tan()),
                "asin" => {
                    if !(-1.0..=1.0).contains(&arg) {
                        Err(CasError::DomainError(format!(
                            "asin argument {} out of range [-1, 1]",
                            arg
                        )))
                    } else {
                        Ok(arg.asin())
                    }
                }
                "acos" => {
                    if !(-1.0..=1.0).contains(&arg) {
                        Err(CasError::DomainError(format!(
                            "acos argument {} out of range [-1, 1]",
                            arg
                        )))
                    } else {
                        Ok(arg.acos())
                    }
                }
                "atan" => Ok(arg.atan()),
                "exp" => Ok(arg.exp()),
                "log" | "ln" => {
                    if arg <= 0.0 {
                        Err(CasError::DomainError(format!(
                            "log argument {} must be positive",
                            arg
                        )))
                    } else {
                        Ok(arg.ln())
                    }
                }
                "sqrt" => {
                    if arg < 0.0 {
                        Err(CasError::DomainError(format!(
                            "sqrt argument {} must be non-negative",
                            arg
                        )))
                    } else {
                        Ok(arg.sqrt())
                    }
                }
                "abs" => Ok(arg.abs()),
                "sinh" => Ok(arg.sinh()),
                "cosh" => Ok(arg.cosh()),
                "tanh" => Ok(arg.tanh()),
                "asinh" => Ok(arg.asinh()),
                "acosh" => {
                    if arg < 1.0 {
                        Err(CasError::DomainError(format!(
                            "acosh argument {} must be >= 1",
                            arg
                        )))
                    } else {
                        Ok(arg.acosh())
                    }
                }
                "atanh" => {
                    if arg <= -1.0 || arg >= 1.0 {
                        Err(CasError::DomainError(format!(
                            "atanh argument {} must be in (-1, 1)",
                            arg
                        )))
                    } else {
                        Ok(arg.atanh())
                    }
                }
                "floor" => Ok(arg.floor()),
                "ceil" => Ok(arg.ceil()),
                "sign" => Ok(arg.signum()),
                _ => Err(CasError::UnsupportedOperation(format!(
                    "unknown function: {}",
                    name
                ))),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn env_with_x(x: f64) -> Environment {
        let mut env = Environment::new();
        env.set("x", x);
        env
    }

    #[test]
    fn test_eval_num() {
        let env = Environment::new();
        assert_eq!(evaluate(&Expr::num(42.0), &env).unwrap(), 42.0);
    }

    #[test]
    fn test_eval_var() {
        let env = env_with_x(5.0);
        assert_eq!(evaluate(&Expr::var("x"), &env).unwrap(), 5.0);
    }

    #[test]
    fn test_eval_undefined_var() {
        let env = Environment::new();
        assert!(matches!(
            evaluate(&Expr::var("z"), &env),
            Err(CasError::UndefinedVariable(_))
        ));
    }

    #[test]
    fn test_eval_pi() {
        let env = Environment::new();
        assert!((evaluate(&Expr::var("pi"), &env).unwrap() - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn test_eval_e() {
        let env = Environment::new();
        assert!((evaluate(&Expr::var("e"), &env).unwrap() - std::f64::consts::E).abs() < 1e-15);
    }

    #[test]
    fn test_eval_add() {
        let env = Environment::new();
        let e = Expr::add(Expr::num(2.0), Expr::num(3.0));
        assert_eq!(evaluate(&e, &env).unwrap(), 5.0);
    }

    #[test]
    fn test_eval_sub() {
        let env = Environment::new();
        let e = Expr::sub(Expr::num(10.0), Expr::num(4.0));
        assert_eq!(evaluate(&e, &env).unwrap(), 6.0);
    }

    #[test]
    fn test_eval_mul() {
        let env = Environment::new();
        let e = Expr::mul(Expr::num(3.0), Expr::num(7.0));
        assert_eq!(evaluate(&e, &env).unwrap(), 21.0);
    }

    #[test]
    fn test_eval_div() {
        let env = Environment::new();
        let e = Expr::div(Expr::num(10.0), Expr::num(4.0));
        assert_eq!(evaluate(&e, &env).unwrap(), 2.5);
    }

    #[test]
    fn test_eval_div_by_zero() {
        let env = Environment::new();
        let e = Expr::div(Expr::num(1.0), Expr::num(0.0));
        assert!(matches!(evaluate(&e, &env), Err(CasError::DivisionByZero)));
    }

    #[test]
    fn test_eval_pow() {
        let env = Environment::new();
        let e = Expr::pow(Expr::num(2.0), Expr::num(10.0));
        assert_eq!(evaluate(&e, &env).unwrap(), 1024.0);
    }

    #[test]
    fn test_eval_neg() {
        let env = Environment::new();
        let e = Expr::neg(Expr::num(5.0));
        assert_eq!(evaluate(&e, &env).unwrap(), -5.0);
    }

    #[test]
    fn test_eval_sin() {
        let env = Environment::new();
        let e = Expr::func("sin", vec![Expr::num(0.0)]);
        assert!(evaluate(&e, &env).unwrap().abs() < 1e-15);
    }

    #[test]
    fn test_eval_cos() {
        let env = Environment::new();
        let e = Expr::func("cos", vec![Expr::num(0.0)]);
        assert!((evaluate(&e, &env).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_eval_exp() {
        let env = Environment::new();
        let e = Expr::func("exp", vec![Expr::num(0.0)]);
        assert!((evaluate(&e, &env).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_eval_log() {
        let env = Environment::new();
        let e = Expr::func("log", vec![Expr::num(1.0)]);
        assert!(evaluate(&e, &env).unwrap().abs() < 1e-15);
    }

    #[test]
    fn test_eval_log_domain_error() {
        let env = Environment::new();
        let e = Expr::func("log", vec![Expr::num(-1.0)]);
        assert!(matches!(evaluate(&e, &env), Err(CasError::DomainError(_))));
    }

    #[test]
    fn test_eval_sqrt() {
        let env = Environment::new();
        let e = Expr::func("sqrt", vec![Expr::num(9.0)]);
        assert_eq!(evaluate(&e, &env).unwrap(), 3.0);
    }

    #[test]
    fn test_eval_sqrt_domain_error() {
        let env = Environment::new();
        let e = Expr::func("sqrt", vec![Expr::num(-4.0)]);
        assert!(matches!(evaluate(&e, &env), Err(CasError::DomainError(_))));
    }

    #[test]
    fn test_eval_abs() {
        let env = Environment::new();
        let e = Expr::func("abs", vec![Expr::num(-7.0)]);
        assert_eq!(evaluate(&e, &env).unwrap(), 7.0);
    }

    #[test]
    fn test_eval_asin() {
        let env = Environment::new();
        let e = Expr::func("asin", vec![Expr::num(0.0)]);
        assert!(evaluate(&e, &env).unwrap().abs() < 1e-15);
    }

    #[test]
    fn test_eval_asin_domain_error() {
        let env = Environment::new();
        let e = Expr::func("asin", vec![Expr::num(2.0)]);
        assert!(matches!(evaluate(&e, &env), Err(CasError::DomainError(_))));
    }

    #[test]
    fn test_eval_unknown_function() {
        let env = Environment::new();
        let e = Expr::func("foobar", vec![Expr::num(1.0)]);
        assert!(matches!(
            evaluate(&e, &env),
            Err(CasError::UnsupportedOperation(_))
        ));
    }

    #[test]
    fn test_eval_sinh() {
        let env = Environment::new();
        let e = Expr::func("sinh", vec![Expr::num(0.0)]);
        assert!(evaluate(&e, &env).unwrap().abs() < 1e-15);
    }

    #[test]
    fn test_eval_sinh_value() {
        let env = Environment::new();
        let e = Expr::func("sinh", vec![Expr::num(1.0)]);
        assert!((evaluate(&e, &env).unwrap() - 1.0_f64.sinh()).abs() < 1e-15);
    }

    #[test]
    fn test_eval_cosh() {
        let env = Environment::new();
        let e = Expr::func("cosh", vec![Expr::num(0.0)]);
        assert!((evaluate(&e, &env).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_eval_tanh() {
        let env = Environment::new();
        let e = Expr::func("tanh", vec![Expr::num(0.0)]);
        assert!(evaluate(&e, &env).unwrap().abs() < 1e-15);
    }

    #[test]
    fn test_eval_asinh() {
        let env = Environment::new();
        let e = Expr::func("asinh", vec![Expr::num(0.0)]);
        assert!(evaluate(&e, &env).unwrap().abs() < 1e-15);
    }

    #[test]
    fn test_eval_acosh() {
        let env = Environment::new();
        let e = Expr::func("acosh", vec![Expr::num(1.0)]);
        assert!(evaluate(&e, &env).unwrap().abs() < 1e-15);
    }

    #[test]
    fn test_eval_acosh_domain_error() {
        let env = Environment::new();
        let e = Expr::func("acosh", vec![Expr::num(0.5)]);
        assert!(matches!(evaluate(&e, &env), Err(CasError::DomainError(_))));
    }

    #[test]
    fn test_eval_atanh() {
        let env = Environment::new();
        let e = Expr::func("atanh", vec![Expr::num(0.0)]);
        assert!(evaluate(&e, &env).unwrap().abs() < 1e-15);
    }

    #[test]
    fn test_eval_atanh_domain_error() {
        let env = Environment::new();
        let e = Expr::func("atanh", vec![Expr::num(1.0)]);
        assert!(matches!(evaluate(&e, &env), Err(CasError::DomainError(_))));
    }

    #[test]
    fn test_eval_floor() {
        let env = Environment::new();
        let e = Expr::func("floor", vec![Expr::num(3.7)]);
        assert_eq!(evaluate(&e, &env).unwrap(), 3.0);
    }

    #[test]
    fn test_eval_ceil() {
        let env = Environment::new();
        let e = Expr::func("ceil", vec![Expr::num(3.2)]);
        assert_eq!(evaluate(&e, &env).unwrap(), 4.0);
    }

    #[test]
    fn test_eval_sign_positive() {
        let env = Environment::new();
        let e = Expr::func("sign", vec![Expr::num(5.0)]);
        assert_eq!(evaluate(&e, &env).unwrap(), 1.0);
    }

    #[test]
    fn test_eval_sign_negative() {
        let env = Environment::new();
        let e = Expr::func("sign", vec![Expr::num(-3.0)]);
        assert_eq!(evaluate(&e, &env).unwrap(), -1.0);
    }

    #[test]
    fn test_eval_sign_zero() {
        // Rust's f64::signum(0.0) returns 1.0 (IEEE standard for positive zero)
        let env = Environment::new();
        let e = Expr::func("sign", vec![Expr::num(0.0)]);
        assert_eq!(evaluate(&e, &env).unwrap(), 1.0);
    }

    #[test]
    fn test_eval_complex_expr() {
        // sin(pi/2) should be 1
        let env = Environment::new();
        let e = Expr::func(
            "sin",
            vec![Expr::div(Expr::var("pi"), Expr::num(2.0))],
        );
        assert!((evaluate(&e, &env).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_eval_polynomial() {
        // x^2 + 2*x + 1 at x=3 → 16
        let env = env_with_x(3.0);
        let e = Expr::add(
            Expr::add(
                Expr::pow(Expr::var("x"), Expr::num(2.0)),
                Expr::mul(Expr::num(2.0), Expr::var("x")),
            ),
            Expr::num(1.0),
        );
        assert_eq!(evaluate(&e, &env).unwrap(), 16.0);
    }
}
