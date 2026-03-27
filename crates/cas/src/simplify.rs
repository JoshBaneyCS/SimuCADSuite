use crate::ast::{BinOp, Expr, UnaryOp};

/// Recursively simplify an expression using basic algebraic identities.
///
/// Applies rules until a fixed point is reached (the expression does not
/// change between iterations).
pub fn simplify(expr: &Expr) -> Expr {
    let mut current = expr.clone();
    for _ in 0..100 {
        let next = simplify_once(&current);
        if next == current {
            return current; // return the already-owned value, no extra clone
        }
        current = next;
    }
    current
}

/// A single simplification pass.
fn simplify_once(expr: &Expr) -> Expr {
    match expr {
        Expr::Num(_) | Expr::Var(_) => expr.clone(),

        Expr::UnaryOp {
            op: UnaryOp::Neg,
            operand,
        } => {
            let s = simplify_once(operand);
            match &s {
                // -c  → fold to Num(-c)
                Expr::Num(n) => Expr::num(-n),
                // --x → x
                Expr::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: inner,
                } => inner.as_ref().clone(),
                _ => Expr::neg(s),
            }
        }

        Expr::BinOp { op, lhs, rhs } => {
            let l = simplify_once(lhs);
            let r = simplify_once(rhs);

            // Constant folding: Num op Num
            if let (Expr::Num(a), Expr::Num(b)) = (&l, &r) {
                if let Some(result) = fold_constants(op, *a, *b) {
                    return Expr::num(result);
                }
            }

            match op {
                BinOp::Add => {
                    // x + 0 = x
                    if r.is_zero() {
                        return l;
                    }
                    // 0 + x = x
                    if l.is_zero() {
                        return r;
                    }
                    // Combine like terms: a*x + b*x = (a+b)*x
                    if let Some(result) = try_combine_like_terms(&l, &r, true) {
                        return result;
                    }
                    Expr::add(l, r)
                }
                BinOp::Sub => {
                    // x - 0 = x
                    if r.is_zero() {
                        return l;
                    }
                    // 0 - x = -x
                    if l.is_zero() {
                        return Expr::neg(r);
                    }
                    // x - x = 0
                    if l == r {
                        return Expr::num(0.0);
                    }
                    // Combine like terms: a*x - b*x = (a-b)*x
                    if let Some(result) = try_combine_like_terms(&l, &r, false) {
                        return result;
                    }
                    Expr::sub(l, r)
                }
                BinOp::Mul => {
                    // x * 0 = 0
                    if l.is_zero() || r.is_zero() {
                        return Expr::num(0.0);
                    }
                    // x * 1 = x
                    if r.is_one() {
                        return l;
                    }
                    // 1 * x = x
                    if l.is_one() {
                        return r;
                    }
                    // (-a) * (-b) = a * b
                    if let (
                        Expr::UnaryOp { op: UnaryOp::Neg, operand: a },
                        Expr::UnaryOp { op: UnaryOp::Neg, operand: b },
                    ) = (&l, &r)
                    {
                        return Expr::mul(a.as_ref().clone(), b.as_ref().clone());
                    }
                    // Combine like terms: (a * x) + (b * x) is handled in Add,
                    // but here handle Num * Num * Var consolidation if needed
                    Expr::mul(l, r)
                }
                BinOp::Div => {
                    // 0 / x = 0  (x != 0, but we simplify symbolically)
                    if l.is_zero() {
                        return Expr::num(0.0);
                    }
                    // x / 1 = x
                    if r.is_one() {
                        return l;
                    }
                    // x / x = 1
                    if l == r {
                        return Expr::num(1.0);
                    }
                    Expr::div(l, r)
                }
                BinOp::Pow => {
                    // x^0 = 1
                    if r.is_zero() {
                        return Expr::num(1.0);
                    }
                    // x^1 = x
                    if r.is_one() {
                        return l;
                    }
                    // 0^x = 0  (for positive x, good enough symbolically)
                    if l.is_zero() {
                        return Expr::num(0.0);
                    }
                    // 1^x = 1
                    if l.is_one() {
                        return Expr::num(1.0);
                    }
                    Expr::pow(l, r)
                }
            }
        }

        Expr::Func { name, args } => {
            let simplified_args: Vec<Expr> = args.iter().map(simplify_once).collect();

            // Trigonometric simplifications for single-argument functions
            if simplified_args.len() == 1 {
                let arg = &simplified_args[0];
                match name.as_str() {
                    "sin" => {
                        // sin(0) = 0
                        if arg.is_zero() {
                            return Expr::num(0.0);
                        }
                        // sin(pi) = 0
                        if *arg == Expr::var("pi") {
                            return Expr::num(0.0);
                        }
                    }
                    "cos" => {
                        // cos(0) = 1
                        if arg.is_zero() {
                            return Expr::num(1.0);
                        }
                        // cos(pi) = -1
                        if *arg == Expr::var("pi") {
                            return Expr::num(-1.0);
                        }
                    }
                    _ => {}
                }
            }

            Expr::Func {
                name: name.clone(),
                args: simplified_args,
            }
        }
    }
}

/// Extract the coefficient and variable from a term that looks like `Num * Expr`
/// or just `Expr` (coefficient = 1).
fn extract_coeff_and_term(expr: &Expr) -> Option<(f64, &Expr)> {
    match expr {
        Expr::BinOp {
            op: BinOp::Mul,
            lhs,
            rhs,
        } => {
            if let Expr::Num(c) = lhs.as_ref() {
                Some((*c, rhs.as_ref()))
            } else if let Expr::Num(c) = rhs.as_ref() {
                Some((*c, lhs.as_ref()))
            } else {
                None
            }
        }
        _ => Some((1.0, expr)),
    }
}

/// Try to combine like terms: `a*x + b*x = (a+b)*x` or `a*x - b*x = (a-b)*x`.
/// `is_add` indicates whether this is addition (true) or subtraction (false).
fn try_combine_like_terms(l: &Expr, r: &Expr, is_add: bool) -> Option<Expr> {
    let (cl, tl) = extract_coeff_and_term(l)?;
    let (cr, tr) = extract_coeff_and_term(r)?;

    // Terms must match (e.g., both are `x` or `sin(x)`)
    if tl != tr {
        return None;
    }

    // Don't combine plain numbers — that's handled by constant folding
    if matches!(tl, Expr::Num(_)) {
        return None;
    }

    let combined = if is_add { cl + cr } else { cl - cr };

    if combined == 0.0 {
        Some(Expr::num(0.0))
    } else if combined == 1.0 {
        Some(tl.clone())
    } else if combined == -1.0 {
        Some(Expr::neg(tl.clone()))
    } else {
        Some(Expr::mul(Expr::num(combined), tl.clone()))
    }
}

/// Try to fold a binary operation on two constants. Returns `None` for
/// division by zero or other problematic cases to leave the expression as-is.
fn fold_constants(op: &BinOp, a: f64, b: f64) -> Option<f64> {
    match op {
        BinOp::Add => Some(a + b),
        BinOp::Sub => Some(a - b),
        BinOp::Mul => Some(a * b),
        BinOp::Div => {
            if b == 0.0 {
                None // don't fold division by zero
            } else {
                Some(a / b)
            }
        }
        BinOp::Pow => Some(a.powf(b)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_add_zero_right() {
        let e = Expr::add(Expr::var("x"), Expr::num(0.0));
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_add_zero_left() {
        let e = Expr::add(Expr::num(0.0), Expr::var("x"));
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_sub_zero() {
        let e = Expr::sub(Expr::var("x"), Expr::num(0.0));
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_zero_minus_x() {
        let e = Expr::sub(Expr::num(0.0), Expr::var("x"));
        assert_eq!(simplify(&e), Expr::neg(Expr::var("x")));
    }

    #[test]
    fn test_simplify_x_minus_x() {
        let e = Expr::sub(Expr::var("x"), Expr::var("x"));
        assert_eq!(simplify(&e), Expr::num(0.0));
    }

    #[test]
    fn test_simplify_mul_one_right() {
        let e = Expr::mul(Expr::var("x"), Expr::num(1.0));
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_mul_one_left() {
        let e = Expr::mul(Expr::num(1.0), Expr::var("x"));
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_mul_zero() {
        let e = Expr::mul(Expr::var("x"), Expr::num(0.0));
        assert_eq!(simplify(&e), Expr::num(0.0));
    }

    #[test]
    fn test_simplify_div_one() {
        let e = Expr::div(Expr::var("x"), Expr::num(1.0));
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_div_self() {
        let e = Expr::div(Expr::var("x"), Expr::var("x"));
        assert_eq!(simplify(&e), Expr::num(1.0));
    }

    #[test]
    fn test_simplify_zero_div() {
        let e = Expr::div(Expr::num(0.0), Expr::var("x"));
        assert_eq!(simplify(&e), Expr::num(0.0));
    }

    #[test]
    fn test_simplify_pow_zero() {
        let e = Expr::pow(Expr::var("x"), Expr::num(0.0));
        assert_eq!(simplify(&e), Expr::num(1.0));
    }

    #[test]
    fn test_simplify_pow_one() {
        let e = Expr::pow(Expr::var("x"), Expr::num(1.0));
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_one_pow() {
        let e = Expr::pow(Expr::num(1.0), Expr::var("x"));
        assert_eq!(simplify(&e), Expr::num(1.0));
    }

    #[test]
    fn test_simplify_constant_fold_add() {
        let e = Expr::add(Expr::num(2.0), Expr::num(3.0));
        assert_eq!(simplify(&e), Expr::num(5.0));
    }

    #[test]
    fn test_simplify_constant_fold_mul() {
        let e = Expr::mul(Expr::num(3.0), Expr::num(4.0));
        assert_eq!(simplify(&e), Expr::num(12.0));
    }

    #[test]
    fn test_simplify_constant_fold_pow() {
        let e = Expr::pow(Expr::num(2.0), Expr::num(3.0));
        assert_eq!(simplify(&e), Expr::num(8.0));
    }

    #[test]
    fn test_simplify_double_neg() {
        let e = Expr::neg(Expr::neg(Expr::var("x")));
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_neg_constant() {
        let e = Expr::neg(Expr::num(3.0));
        assert_eq!(simplify(&e), Expr::num(-3.0));
    }

    #[test]
    fn test_simplify_nested() {
        // (x + 0) * 1 → x
        let e = Expr::mul(Expr::add(Expr::var("x"), Expr::num(0.0)), Expr::num(1.0));
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_deeply_nested() {
        // ((x * 1) + 0)^1 → x
        let e = Expr::pow(
            Expr::add(
                Expr::mul(Expr::var("x"), Expr::num(1.0)),
                Expr::num(0.0),
            ),
            Expr::num(1.0),
        );
        assert_eq!(simplify(&e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_preserves_non_trivial() {
        // x + y should not change
        let e = Expr::add(Expr::var("x"), Expr::var("y"));
        assert_eq!(simplify(&e), Expr::add(Expr::var("x"), Expr::var("y")));
    }

    #[test]
    fn test_simplify_func_args() {
        // sin(x + 0) → sin(x)
        let e = Expr::func("sin", vec![Expr::add(Expr::var("x"), Expr::num(0.0))]);
        assert_eq!(simplify(&e), Expr::func("sin", vec![Expr::var("x")]));
    }

    #[test]
    fn test_simplify_div_by_zero_not_folded() {
        // 1 / 0 should stay as-is (not fold to Inf)
        let e = Expr::div(Expr::num(1.0), Expr::num(0.0));
        assert_eq!(simplify(&e), Expr::div(Expr::num(1.0), Expr::num(0.0)));
    }

    // --- New simplification rules ---

    #[test]
    fn test_simplify_double_neg_in_mul() {
        // (-a) * (-b) = a * b
        let e = Expr::mul(Expr::neg(Expr::var("a")), Expr::neg(Expr::var("b")));
        assert_eq!(simplify(&e), Expr::mul(Expr::var("a"), Expr::var("b")));
    }

    #[test]
    fn test_simplify_combine_like_terms_add() {
        // 2*x + 3*x = 5*x
        let e = Expr::add(
            Expr::mul(Expr::num(2.0), Expr::var("x")),
            Expr::mul(Expr::num(3.0), Expr::var("x")),
        );
        assert_eq!(simplify(&e), Expr::mul(Expr::num(5.0), Expr::var("x")));
    }

    #[test]
    fn test_simplify_combine_like_terms_sub() {
        // 5*x - 3*x = 2*x
        let e = Expr::sub(
            Expr::mul(Expr::num(5.0), Expr::var("x")),
            Expr::mul(Expr::num(3.0), Expr::var("x")),
        );
        assert_eq!(simplify(&e), Expr::mul(Expr::num(2.0), Expr::var("x")));
    }

    #[test]
    fn test_simplify_combine_like_terms_cancel() {
        // 3*x - 3*x = 0
        let e = Expr::sub(
            Expr::mul(Expr::num(3.0), Expr::var("x")),
            Expr::mul(Expr::num(3.0), Expr::var("x")),
        );
        assert_eq!(simplify(&e), Expr::num(0.0));
    }

    #[test]
    fn test_simplify_combine_like_terms_implicit_coeff() {
        // x + 3*x = 4*x (x has implicit coefficient 1)
        let e = Expr::add(
            Expr::var("x"),
            Expr::mul(Expr::num(3.0), Expr::var("x")),
        );
        assert_eq!(simplify(&e), Expr::mul(Expr::num(4.0), Expr::var("x")));
    }

    #[test]
    fn test_simplify_sin_zero() {
        let e = Expr::func("sin", vec![Expr::num(0.0)]);
        assert_eq!(simplify(&e), Expr::num(0.0));
    }

    #[test]
    fn test_simplify_cos_zero() {
        let e = Expr::func("cos", vec![Expr::num(0.0)]);
        assert_eq!(simplify(&e), Expr::num(1.0));
    }

    #[test]
    fn test_simplify_sin_pi() {
        let e = Expr::func("sin", vec![Expr::var("pi")]);
        assert_eq!(simplify(&e), Expr::num(0.0));
    }

    #[test]
    fn test_simplify_cos_pi() {
        let e = Expr::func("cos", vec![Expr::var("pi")]);
        assert_eq!(simplify(&e), Expr::num(-1.0));
    }
}
