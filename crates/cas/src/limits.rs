//! Symbolic and numerical limit evaluation.
//!
//! Computes limits of expressions as a variable approaches a target value,
//! including one-sided limits and limits at infinity.

use simucad_core::error::CasError;

use crate::ast::Expr;
use crate::evaluator::{evaluate, Environment};
use crate::simplify::simplify;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Direction from which the limit is approached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitDirection {
    /// Two-sided limit.
    Both,
    /// Left-sided limit (from below).
    Left,
    /// Right-sided limit (from above).
    Right,
}

/// Result of a limit computation.
#[derive(Debug, Clone, PartialEq)]
pub enum LimitResult {
    /// The limit evaluates to a finite value.
    Finite(f64),
    /// The limit diverges to positive infinity.
    PosInfinity,
    /// The limit diverges to negative infinity.
    NegInfinity,
    /// The limit does not exist (e.g. oscillation or left != right).
    DoesNotExist,
}

/// Compute the limit of `expr` as `var` approaches `target`.
///
/// Uses direct substitution first, then L'Hopital's rule for 0/0 and inf/inf
/// indeterminate forms, and finally numerical estimation.
///
/// # Parameters
///
/// - `expr` -- the expression to evaluate.
/// - `var` -- the variable approaching the target.
/// - `target` -- the value being approached (finite).
/// - `direction` -- two-sided, left, or right.
///
/// # Examples
///
/// ```ignore
/// // lim x->0 sin(x)/x = 1
/// let expr = Expr::div(Expr::func("sin", vec![Expr::var("x")]), Expr::var("x"));
/// let result = compute_limit(&expr, "x", 0.0, LimitDirection::Both).unwrap();
/// assert_eq!(result, LimitResult::Finite(1.0));
/// ```
pub fn compute_limit(
    expr: &Expr,
    var: &str,
    target: f64,
    direction: LimitDirection,
) -> Result<LimitResult, CasError> {
    let simplified = simplify(expr);

    // Try direct substitution first.
    if let Some(val) = try_direct(&simplified, var, target) {
        return Ok(LimitResult::Finite(val));
    }

    // Check if the expression is a quotient (potential L'Hopital).
    if let Some(result) = try_lhopital(&simplified, var, target, direction, 0)? {
        return Ok(result);
    }

    // Fall back to numerical estimation.
    numerical_limit(&simplified, var, target, direction)
}

/// Compute the limit of `expr` as `var` approaches positive or negative
/// infinity.
///
/// `positive` controls the direction: `true` for +inf, `false` for -inf.
pub fn compute_limit_at_infinity(
    expr: &Expr,
    var: &str,
    positive: bool,
) -> Result<LimitResult, CasError> {
    let simplified = simplify(expr);

    // Evaluate at increasingly large values and check convergence.
    let magnitudes = [1e2, 1e4, 1e6, 1e8, 1e12];
    let sign = if positive { 1.0 } else { -1.0 };

    let mut values = Vec::new();
    for &m in &magnitudes {
        let x = sign * m;
        let mut env = Environment::new();
        env.set(var, x);
        match evaluate(&simplified, &env) {
            Ok(v) if v.is_finite() => values.push(v),
            Ok(v) if v.is_infinite() => {
                return Ok(if v > 0.0 {
                    LimitResult::PosInfinity
                } else {
                    LimitResult::NegInfinity
                });
            }
            _ => return Ok(LimitResult::DoesNotExist),
        }
    }

    // Check if the sequence converges.
    if values.len() >= 3 {
        let last = *values.last().unwrap();
        let prev = values[values.len() - 2];
        let prev2 = values[values.len() - 3];

        // If the last three values are close, the limit converges.
        if (last - prev).abs() < 1e-6 * last.abs().max(1.0)
            && (prev - prev2).abs() < 1e-4 * prev.abs().max(1.0)
        {
            return Ok(LimitResult::Finite(last));
        }

        // Check for divergence.
        if last.abs() > prev.abs() && prev.abs() > prev2.abs() && last.abs() > 1e10 {
            return Ok(if last > 0.0 {
                LimitResult::PosInfinity
            } else {
                LimitResult::NegInfinity
            });
        }
    }

    Ok(LimitResult::DoesNotExist)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Try direct substitution. Returns `Some(value)` if the result is finite.
fn try_direct(expr: &Expr, var: &str, target: f64) -> Option<f64> {
    let mut env = Environment::new();
    env.set(var, target);
    match evaluate(expr, &env) {
        Ok(v) if v.is_finite() => Some(v),
        _ => None,
    }
}

/// Attempt L'Hopital's rule on a quotient expression.
///
/// Recursively applies the rule up to `MAX_LHOPITAL_DEPTH` times.
const MAX_LHOPITAL_DEPTH: usize = 5;

fn try_lhopital(
    expr: &Expr,
    var: &str,
    target: f64,
    direction: LimitDirection,
    depth: usize,
) -> Result<Option<LimitResult>, CasError> {
    if depth > MAX_LHOPITAL_DEPTH {
        return Ok(None);
    }

    // Only works on f/g quotients.
    let (numer, denom) = match expr {
        Expr::BinOp {
            op: crate::ast::BinOp::Div,
            lhs,
            rhs,
        } => (lhs.as_ref(), rhs.as_ref()),
        _ => return Ok(None),
    };

    // Evaluate numerator and denominator at the target.
    let mut env = Environment::new();
    env.set(var, target);

    let n_val = evaluate(numer, &env).unwrap_or(f64::NAN);
    let d_val = evaluate(denom, &env).unwrap_or(f64::NAN);

    // Check for 0/0 or inf/inf indeterminate forms.
    let is_zero_zero = n_val.abs() < 1e-12 && d_val.abs() < 1e-12;
    let is_inf_inf = n_val.is_infinite() && d_val.is_infinite();

    if !is_zero_zero && !is_inf_inf {
        return Ok(None);
    }

    // Differentiate numerator and denominator.
    let dn = crate::derivative::differentiate(numer, var)?;
    let dd = crate::derivative::differentiate(denom, var)?;
    let new_expr = simplify(&Expr::div(dn, dd));

    // Try direct substitution on the new quotient.
    if let Some(val) = try_direct(&new_expr, var, target) {
        return Ok(Some(LimitResult::Finite(val)));
    }

    // Recurse.
    try_lhopital(&new_expr, var, target, direction, depth + 1)
}

/// Estimate the limit numerically by approaching `target` from the given
/// direction with decreasing step sizes.
fn numerical_limit(
    expr: &Expr,
    var: &str,
    target: f64,
    direction: LimitDirection,
) -> Result<LimitResult, CasError> {
    let offsets = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10];

    let eval_at = |x: f64| -> Option<f64> {
        let mut env = Environment::new();
        env.set(var, x);
        evaluate(expr, &env).ok().filter(|v| v.is_finite())
    };

    match direction {
        LimitDirection::Right => estimate_one_sided(target, &offsets, 1.0, eval_at),
        LimitDirection::Left => estimate_one_sided(target, &offsets, -1.0, eval_at),
        LimitDirection::Both => {
            let right = estimate_one_sided(target, &offsets, 1.0, &eval_at);
            let left = estimate_one_sided(target, &offsets, -1.0, &eval_at);
            match (&left, &right) {
                (Ok(LimitResult::Finite(l)), Ok(LimitResult::Finite(r))) => {
                    if (l - r).abs() < 1e-6 * l.abs().max(r.abs()).max(1.0) {
                        Ok(LimitResult::Finite((l + r) / 2.0))
                    } else {
                        Ok(LimitResult::DoesNotExist)
                    }
                }
                (Ok(l), Ok(r)) if l == r => Ok(l.clone()),
                _ => Ok(LimitResult::DoesNotExist),
            }
        }
    }
}

fn estimate_one_sided(
    target: f64,
    offsets: &[f64],
    sign: f64,
    eval_at: impl Fn(f64) -> Option<f64>,
) -> Result<LimitResult, CasError> {
    let mut values = Vec::new();
    for &h in offsets {
        if let Some(v) = eval_at(target + sign * h) {
            values.push(v);
        }
    }

    if values.is_empty() {
        return Ok(LimitResult::DoesNotExist);
    }

    // Check convergence or divergence of the sequence.
    if values.len() >= 2 {
        let last = *values.last().unwrap();
        let prev = values[values.len() - 2];

        // Check for divergence first: if magnitudes are growing and large.
        if values.len() >= 3 {
            let prev2 = values[values.len() - 3];
            if last.abs() > prev.abs()
                && prev.abs() > prev2.abs()
                && last.abs() > 1e6
            {
                return Ok(if last > 0.0 {
                    LimitResult::PosInfinity
                } else {
                    LimitResult::NegInfinity
                });
            }
        }

        // Check for convergence.
        if (last - prev).abs() < 1e-6 * last.abs().max(1.0) {
            return Ok(LimitResult::Finite(last));
        }

        // Large absolute value without convergence => diverges.
        if last.abs() > 1e8 {
            return Ok(if last > 0.0 {
                LimitResult::PosInfinity
            } else {
                LimitResult::NegInfinity
            });
        }
    }

    // Best estimate is the last computed value.
    let last = *values.last().unwrap();
    Ok(LimitResult::Finite(last))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr;

    fn assert_finite(result: &LimitResult, expected: f64, tol: f64) {
        match result {
            LimitResult::Finite(v) => {
                assert!(
                    (v - expected).abs() < tol,
                    "expected ~{expected}, got {v}"
                );
            }
            other => panic!("expected Finite({expected}), got {other:?}"),
        }
    }

    #[test]
    fn limit_polynomial_direct_sub() {
        // lim x->2 (x^2 + 1) = 5
        let e = Expr::add(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::num(1.0),
        );
        let r = compute_limit(&e, "x", 2.0, LimitDirection::Both).unwrap();
        assert_finite(&r, 5.0, 1e-10);
    }

    #[test]
    fn limit_sin_x_over_x() {
        // lim x->0 sin(x)/x = 1  (0/0 form, L'Hopital)
        let e = Expr::div(
            Expr::func("sin", vec![Expr::var("x")]),
            Expr::var("x"),
        );
        let r = compute_limit(&e, "x", 0.0, LimitDirection::Both).unwrap();
        assert_finite(&r, 1.0, 1e-6);
    }

    #[test]
    fn limit_1_minus_cos_over_x_squared() {
        // lim x->0 (1 - cos(x)) / x^2 = 1/2
        let e = Expr::div(
            Expr::sub(Expr::num(1.0), Expr::func("cos", vec![Expr::var("x")])),
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
        );
        let r = compute_limit(&e, "x", 0.0, LimitDirection::Both).unwrap();
        assert_finite(&r, 0.5, 1e-5);
    }

    #[test]
    fn limit_at_positive_infinity() {
        // lim x->+inf 1/x = 0
        let e = Expr::div(Expr::num(1.0), Expr::var("x"));
        let r = compute_limit_at_infinity(&e, "x", true).unwrap();
        assert_finite(&r, 0.0, 1e-6);
    }

    #[test]
    fn limit_at_negative_infinity_diverges() {
        // lim x->-inf x^2 = +inf
        let e = Expr::pow(Expr::var("x"), Expr::num(2.0));
        let r = compute_limit_at_infinity(&e, "x", false).unwrap();
        assert_eq!(r, LimitResult::PosInfinity);
    }

    #[test]
    fn limit_one_sided_1_over_x() {
        // lim x->0+ 1/x = +inf
        let e = Expr::div(Expr::num(1.0), Expr::var("x"));
        let r = compute_limit(&e, "x", 0.0, LimitDirection::Right).unwrap();
        assert_eq!(r, LimitResult::PosInfinity);

        // lim x->0- 1/x = -inf
        let r = compute_limit(&e, "x", 0.0, LimitDirection::Left).unwrap();
        assert_eq!(r, LimitResult::NegInfinity);
    }

    #[test]
    fn limit_exp_minus_1_over_x() {
        // lim x->0 (exp(x) - 1)/x = 1  (0/0 form)
        let e = Expr::div(
            Expr::sub(
                Expr::func("exp", vec![Expr::var("x")]),
                Expr::num(1.0),
            ),
            Expr::var("x"),
        );
        let r = compute_limit(&e, "x", 0.0, LimitDirection::Both).unwrap();
        assert_finite(&r, 1.0, 1e-5);
    }

    #[test]
    fn limit_constant_expression() {
        let e = Expr::num(42.0);
        let r = compute_limit(&e, "x", 999.0, LimitDirection::Both).unwrap();
        assert_finite(&r, 42.0, 1e-10);
    }
}
