//! Numerical and symbolic ODE solvers.
//!
//! Solves first-order ordinary differential equations of the form
//! `dy/dx = f(x, y)` using numerical methods (Euler, RK4) and provides
//! symbolic solutions for separable and linear first-order ODEs.

use simucad_core::error::CasError;

use crate::ast::{BinOp, Expr};
use crate::evaluator::{evaluate, Environment};
use crate::simplify::simplify;

// ---------------------------------------------------------------------------
// Numerical solution types
// ---------------------------------------------------------------------------

/// A single point in the numerical ODE solution.
#[derive(Debug, Clone, PartialEq)]
pub struct OdePoint {
    pub x: f64,
    pub y: f64,
}

/// Method used for numerical integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeMethod {
    /// Forward Euler (1st order).
    Euler,
    /// Classic Runge-Kutta (4th order).
    RK4,
}

/// Configuration for the numerical ODE solver.
#[derive(Debug, Clone)]
pub struct OdeConfig {
    /// Integration method.
    pub method: OdeMethod,
    /// Step size.
    pub step_size: f64,
    /// Number of steps.
    pub num_steps: usize,
}

impl Default for OdeConfig {
    fn default() -> Self {
        Self {
            method: OdeMethod::RK4,
            step_size: 0.01,
            num_steps: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Numerical ODE solver
// ---------------------------------------------------------------------------

/// Solve the initial value problem `dy/dx = f(x, y)` numerically.
///
/// # Parameters
///
/// - `rhs` -- the right-hand side expression `f(x, y)`. Must use variables
///   named `x_var` and `y_var`.
/// - `x_var` -- name of the independent variable (e.g. "x" or "t").
/// - `y_var` -- name of the dependent variable (e.g. "y").
/// - `x0` -- initial x value.
/// - `y0` -- initial y value.
/// - `config` -- solver configuration.
///
/// # Returns
///
/// A vector of `(x, y)` points along the solution curve.
pub fn solve_ivp(
    rhs: &Expr,
    x_var: &str,
    y_var: &str,
    x0: f64,
    y0: f64,
    config: &OdeConfig,
) -> Result<Vec<OdePoint>, CasError> {
    if config.step_size <= 0.0 {
        return Err(CasError::DomainError(
            "step size must be positive".to_string(),
        ));
    }

    let mut points = Vec::with_capacity(config.num_steps + 1);
    let mut x = x0;
    let mut y = y0;

    points.push(OdePoint { x, y });

    let f = |xv: f64, yv: f64| -> Result<f64, CasError> {
        let mut env = Environment::new();
        env.set(x_var, xv);
        env.set(y_var, yv);
        evaluate(rhs, &env)
    };

    let h = config.step_size;

    for _ in 0..config.num_steps {
        y = match config.method {
            OdeMethod::Euler => {
                let slope = f(x, y)?;
                y + h * slope
            }
            OdeMethod::RK4 => {
                let k1 = f(x, y)?;
                let k2 = f(x + h / 2.0, y + h * k1 / 2.0)?;
                let k3 = f(x + h / 2.0, y + h * k2 / 2.0)?;
                let k4 = f(x + h, y + h * k3)?;
                y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            }
        };

        x += h;
        points.push(OdePoint { x, y });

        if !y.is_finite() {
            return Err(CasError::DomainError(format!(
                "solution diverged at x = {x}"
            )));
        }
    }

    Ok(points)
}

// ---------------------------------------------------------------------------
// Symbolic ODE solver (classification + solution)
// ---------------------------------------------------------------------------

/// Classification of a first-order ODE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdeType {
    /// dy/dx = g(x) — directly integrable.
    DirectlyIntegrable,
    /// dy/dx = a*y + b(x) — linear first-order.
    LinearFirstOrder,
    /// dy/dx = f(x)*g(y) — separable.
    Separable,
    /// Could not classify.
    Unknown,
}

/// Attempt to classify a first-order ODE `dy/dx = rhs(x, y)`.
pub fn classify_ode(rhs: &Expr, x_var: &str, y_var: &str) -> OdeType {
    // Check if rhs contains y at all.
    if !contains_var(rhs, y_var) {
        return OdeType::DirectlyIntegrable;
    }

    // Check for linear form: a(x)*y + b(x)
    if is_linear_in(rhs, y_var) {
        return OdeType::LinearFirstOrder;
    }

    // Check for separable form: f(x) * g(y)
    if is_separable(rhs, x_var, y_var) {
        return OdeType::Separable;
    }

    OdeType::Unknown
}

/// Attempt to find a symbolic solution to a first-order ODE.
///
/// Returns the solution as an expression for `y` in terms of `x_var` and
/// an integration constant `C`. Only works for directly integrable and
/// simple separable/linear cases.
pub fn solve_symbolic(
    rhs: &Expr,
    x_var: &str,
    y_var: &str,
) -> Result<Expr, CasError> {
    let ode_type = classify_ode(rhs, x_var, y_var);

    match ode_type {
        OdeType::DirectlyIntegrable => {
            // dy/dx = f(x) => y = ∫f(x)dx + C
            let integral = crate::integration::integrate(rhs, x_var)?;
            Ok(simplify(&Expr::add(integral, Expr::var("C"))))
        }

        OdeType::LinearFirstOrder => {
            // dy/dx = a*y means y = C*exp(a*x) when b(x) = 0
            // Extract coefficient of y.
            if let Some(coeff) = extract_y_coefficient(rhs, y_var) {
                if !contains_var(&remainder_after_y(rhs, y_var), x_var)
                    && !contains_var(&remainder_after_y(rhs, y_var), y_var)
                {
                    // dy/dx = a*y + b, with a and b constant
                    let b = remainder_after_y(rhs, y_var);
                    let b_simplified = simplify(&b);

                    if b_simplified.is_zero() {
                        // y = C * exp(a*x)
                        let ax = Expr::mul(coeff, Expr::var(x_var));
                        return Ok(Expr::mul(Expr::var("C"), Expr::func("exp", vec![ax])));
                    }
                }
            }

            Err(CasError::UnsupportedOperation(
                "symbolic solution of this linear ODE form is not yet supported".to_string(),
            ))
        }

        _ => Err(CasError::UnsupportedOperation(format!(
            "symbolic solution of {:?} ODEs is not yet supported",
            ode_type
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn contains_var(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Num(_) => false,
        Expr::Var(name) => name == var,
        Expr::BinOp { lhs, rhs, .. } => contains_var(lhs, var) || contains_var(rhs, var),
        Expr::UnaryOp { operand, .. } => contains_var(operand, var),
        Expr::Func { args, .. } => args.iter().any(|a| contains_var(a, var)),
    }
}

/// Check if `expr` is linear in `var` (of the form `a * var + b` where `a`
/// and `b` do not depend on `var`).
fn is_linear_in(expr: &Expr, var: &str) -> bool {
    match expr {
        Expr::Var(name) => name == var,
        Expr::Num(_) => true,
        Expr::BinOp { op, lhs, rhs } => match op {
            BinOp::Add | BinOp::Sub => is_linear_in(lhs, var) && is_linear_in(rhs, var),
            BinOp::Mul => {
                // a*y is linear if a doesn't contain y
                (!contains_var(lhs, var) && is_linear_in(rhs, var))
                    || (!contains_var(rhs, var) && is_linear_in(lhs, var))
            }
            _ => !contains_var(expr, var),
        },
        Expr::UnaryOp { operand, .. } => is_linear_in(operand, var),
        Expr::Func { .. } => !contains_var(expr, var),
    }
}

/// Check if `expr` can be written as f(x) * g(y).
fn is_separable(expr: &Expr, x_var: &str, y_var: &str) -> bool {
    match expr {
        Expr::BinOp {
            op: BinOp::Mul,
            lhs,
            rhs,
        } => {
            // One factor depends only on x, the other only on y.
            (!contains_var(lhs, y_var) && !contains_var(rhs, x_var))
                || (!contains_var(lhs, x_var) && !contains_var(rhs, y_var))
        }
        // A pure function of y (like y^2) is separable with f(x) = 1.
        _ => !contains_var(expr, x_var) || !contains_var(expr, y_var),
    }
}

/// Extract the coefficient of `y` in a linear expression.
fn extract_y_coefficient(expr: &Expr, y_var: &str) -> Option<Expr> {
    match expr {
        Expr::Var(name) if name == y_var => Some(Expr::num(1.0)),
        Expr::BinOp {
            op: BinOp::Mul,
            lhs,
            rhs,
        } => {
            if let Expr::Var(name) = rhs.as_ref() {
                if name == y_var && !contains_var(lhs, y_var) {
                    return Some(lhs.as_ref().clone());
                }
            }
            if let Expr::Var(name) = lhs.as_ref() {
                if name == y_var && !contains_var(rhs, y_var) {
                    return Some(rhs.as_ref().clone());
                }
            }
            None
        }
        Expr::BinOp {
            op: BinOp::Add,
            lhs,
            rhs,
        } => {
            extract_y_coefficient(lhs, y_var)
                .or_else(|| extract_y_coefficient(rhs, y_var))
        }
        Expr::BinOp {
            op: BinOp::Sub,
            lhs,
            rhs,
        } => {
            extract_y_coefficient(lhs, y_var)
                .or_else(|| extract_y_coefficient(rhs, y_var).map(|c| Expr::neg(c)))
        }
        _ => None,
    }
}

/// Get the part of a linear expression that doesn't contain `y`.
fn remainder_after_y(expr: &Expr, y_var: &str) -> Expr {
    match expr {
        Expr::Var(name) if name == y_var => Expr::num(0.0),
        Expr::BinOp {
            op: BinOp::Mul, ..
        } if contains_var(expr, y_var) => Expr::num(0.0),
        Expr::BinOp {
            op: BinOp::Add,
            lhs,
            rhs,
        } => {
            if contains_var(lhs, y_var) && !contains_var(rhs, y_var) {
                rhs.as_ref().clone()
            } else if !contains_var(lhs, y_var) && contains_var(rhs, y_var) {
                lhs.as_ref().clone()
            } else {
                expr.clone()
            }
        }
        _ => {
            if contains_var(expr, y_var) {
                Expr::num(0.0)
            } else {
                expr.clone()
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
    use crate::ast::Expr;

    // ----- Numerical solver tests -----

    #[test]
    fn euler_exponential_growth() {
        // dy/dx = y, y(0) = 1 => y = e^x
        // Test at x = 1: y ≈ e ≈ 2.71828
        let rhs = Expr::var("y");
        let config = OdeConfig {
            method: OdeMethod::Euler,
            step_size: 0.001,
            num_steps: 1000,
        };

        let solution = solve_ivp(&rhs, "x", "y", 0.0, 1.0, &config).unwrap();
        let last = solution.last().unwrap();
        assert!(
            (last.x - 1.0).abs() < 1e-10,
            "should end at x=1, got {}",
            last.x
        );
        assert!(
            (last.y - std::f64::consts::E).abs() < 0.01,
            "Euler should approximate e, got {}",
            last.y
        );
    }

    #[test]
    fn rk4_exponential_growth() {
        // dy/dx = y, y(0) = 1 => y = e^x
        // RK4 should be much more accurate than Euler.
        let rhs = Expr::var("y");
        let config = OdeConfig {
            method: OdeMethod::RK4,
            step_size: 0.01,
            num_steps: 100,
        };

        let solution = solve_ivp(&rhs, "x", "y", 0.0, 1.0, &config).unwrap();
        let last = solution.last().unwrap();
        assert!(
            (last.y - std::f64::consts::E).abs() < 1e-8,
            "RK4 should be very accurate, got {}",
            last.y
        );
    }

    #[test]
    fn rk4_linear_ode() {
        // dy/dx = 2x, y(0) = 0 => y = x^2
        // At x = 3: y = 9
        let rhs = Expr::mul(Expr::num(2.0), Expr::var("x"));
        let config = OdeConfig {
            method: OdeMethod::RK4,
            step_size: 0.01,
            num_steps: 300,
        };

        let solution = solve_ivp(&rhs, "x", "y", 0.0, 0.0, &config).unwrap();
        let last = solution.last().unwrap();
        assert!(
            (last.y - 9.0).abs() < 1e-6,
            "should be 9 at x=3, got {}",
            last.y
        );
    }

    #[test]
    fn rk4_harmonic_like() {
        // dy/dx = -x, y(0) = 0 => y = -x^2/2
        // At x = 2: y = -2
        let rhs = Expr::neg(Expr::var("x"));
        let config = OdeConfig {
            method: OdeMethod::RK4,
            step_size: 0.01,
            num_steps: 200,
        };

        let solution = solve_ivp(&rhs, "x", "y", 0.0, 0.0, &config).unwrap();
        let last = solution.last().unwrap();
        assert!(
            (last.y - (-2.0)).abs() < 1e-6,
            "should be -2 at x=2, got {}",
            last.y
        );
    }

    #[test]
    fn rk4_returns_correct_point_count() {
        let rhs = Expr::num(1.0);
        let config = OdeConfig {
            method: OdeMethod::RK4,
            step_size: 0.1,
            num_steps: 50,
        };

        let solution = solve_ivp(&rhs, "x", "y", 0.0, 0.0, &config).unwrap();
        assert_eq!(solution.len(), 51); // initial + 50 steps
    }

    #[test]
    fn invalid_step_size() {
        let rhs = Expr::var("y");
        let config = OdeConfig {
            method: OdeMethod::RK4,
            step_size: 0.0,
            num_steps: 100,
        };

        assert!(solve_ivp(&rhs, "x", "y", 0.0, 1.0, &config).is_err());
    }

    // ----- Classification tests -----

    #[test]
    fn classify_directly_integrable() {
        // dy/dx = x^2 (no y)
        let rhs = Expr::pow(Expr::var("x"), Expr::num(2.0));
        assert_eq!(classify_ode(&rhs, "x", "y"), OdeType::DirectlyIntegrable);
    }

    #[test]
    fn classify_linear() {
        // dy/dx = 3*y
        let rhs = Expr::mul(Expr::num(3.0), Expr::var("y"));
        assert_eq!(classify_ode(&rhs, "x", "y"), OdeType::LinearFirstOrder);
    }

    #[test]
    fn classify_separable() {
        // dy/dx = x * y^2
        let rhs = Expr::mul(
            Expr::var("x"),
            Expr::pow(Expr::var("y"), Expr::num(2.0)),
        );
        assert_eq!(classify_ode(&rhs, "x", "y"), OdeType::Separable);
    }

    // ----- Symbolic solver tests -----

    #[test]
    fn symbolic_directly_integrable() {
        // dy/dx = x => y = x^2/2 + C
        let rhs = Expr::var("x");
        let sol = solve_symbolic(&rhs, "x", "y").unwrap();
        // Evaluate at x=4: should be 4^2/2 + C = 8 + C
        let mut env = Environment::new();
        env.set("x", 4.0);
        env.set("C", 0.0);
        let val = evaluate(&sol, &env).unwrap();
        assert!(
            (val - 8.0).abs() < 1e-10,
            "integral of x at x=4 should be 8, got {val}"
        );
    }

    #[test]
    fn symbolic_exponential_decay() {
        // dy/dx = -2*y => y = C*exp(-2x)
        let rhs = Expr::mul(Expr::num(-2.0), Expr::var("y"));
        let sol = solve_symbolic(&rhs, "x", "y").unwrap();
        // Evaluate at x=0, C=5: y = 5*exp(0) = 5
        let mut env = Environment::new();
        env.set("x", 0.0);
        env.set("C", 5.0);
        let val = evaluate(&sol, &env).unwrap();
        assert!(
            (val - 5.0).abs() < 1e-10,
            "at x=0 with C=5 should be 5, got {val}"
        );
    }
}
