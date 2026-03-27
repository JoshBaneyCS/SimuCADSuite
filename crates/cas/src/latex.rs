use crate::ast::{BinOp, Expr, UnaryOp};

/// Render an expression as a LaTeX string.
pub fn to_latex(expr: &Expr) -> String {
    match expr {
        Expr::Num(n) => {
            if *n == (*n as i64) as f64 && n.is_finite() {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }

        Expr::Var(name) => name.clone(),

        Expr::BinOp { op, lhs, rhs } => match op {
            BinOp::Add => {
                format!("{} + {}", to_latex(lhs), to_latex(rhs))
            }
            BinOp::Sub => {
                let rhs_str = match rhs.as_ref() {
                    Expr::BinOp {
                        op: BinOp::Add | BinOp::Sub,
                        ..
                    } => format!("\\left({}\\right)", to_latex(rhs)),
                    _ => to_latex(rhs),
                };
                format!("{} - {}", to_latex(lhs), rhs_str)
            }
            BinOp::Mul => {
                let lhs_str = match lhs.as_ref() {
                    Expr::BinOp {
                        op: BinOp::Add | BinOp::Sub,
                        ..
                    } => format!("\\left({}\\right)", to_latex(lhs)),
                    _ => to_latex(lhs),
                };
                let rhs_str = match rhs.as_ref() {
                    Expr::BinOp {
                        op: BinOp::Add | BinOp::Sub,
                        ..
                    } => format!("\\left({}\\right)", to_latex(rhs)),
                    _ => to_latex(rhs),
                };
                format!("{} \\cdot {}", lhs_str, rhs_str)
            }
            BinOp::Div => {
                format!("\\frac{{{}}}{{{}}}", to_latex(lhs), to_latex(rhs))
            }
            BinOp::Pow => {
                let base_str = match lhs.as_ref() {
                    Expr::Num(_) | Expr::Var(_) => to_latex(lhs),
                    Expr::Func { .. } => to_latex(lhs),
                    _ => format!("\\left({}\\right)", to_latex(lhs)),
                };
                format!("{}^{{{}}}", base_str, to_latex(rhs))
            }
        },

        Expr::UnaryOp {
            op: UnaryOp::Neg,
            operand,
        } => match operand.as_ref() {
            Expr::Num(_) | Expr::Var(_) => format!("-{}", to_latex(operand)),
            _ => format!("-\\left({}\\right)", to_latex(operand)),
        },

        Expr::Func { name, args } => {
            let args_str: Vec<String> = args.iter().map(|a| to_latex(a)).collect();
            let joined = args_str.join(", ");

            match name.as_str() {
                "sin" => format!("\\sin\\left({}\\right)", joined),
                "cos" => format!("\\cos\\left({}\\right)", joined),
                "tan" => format!("\\tan\\left({}\\right)", joined),
                "asin" => format!("\\arcsin\\left({}\\right)", joined),
                "acos" => format!("\\arccos\\left({}\\right)", joined),
                "atan" => format!("\\arctan\\left({}\\right)", joined),
                "exp" => format!("e^{{{}}}", joined),
                "log" | "ln" => format!("\\ln\\left({}\\right)", joined),
                "sqrt" => format!("\\sqrt{{{}}}", joined),
                "abs" => format!("\\left|{}\\right|", joined),
                "sinh" => format!("\\sinh\\left({}\\right)", joined),
                "cosh" => format!("\\cosh\\left({}\\right)", joined),
                "tanh" => format!("\\tanh\\left({}\\right)", joined),
                "asinh" => format!("\\operatorname{{asinh}}\\left({}\\right)", joined),
                "acosh" => format!("\\operatorname{{acosh}}\\left({}\\right)", joined),
                "atanh" => format!("\\operatorname{{atanh}}\\left({}\\right)", joined),
                "floor" => format!("\\lfloor {} \\rfloor", joined),
                "ceil" => format!("\\lceil {} \\rceil", joined),
                "sign" => format!("\\operatorname{{sgn}}\\left({}\\right)", joined),
                _ => format!("\\mathrm{{{}}}\\left({}\\right)", name, joined),
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

    #[test]
    fn test_latex_number() {
        assert_eq!(to_latex(&Expr::num(42.0)), "42");
    }

    #[test]
    fn test_latex_float() {
        assert_eq!(to_latex(&Expr::num(3.14)), "3.14");
    }

    #[test]
    fn test_latex_variable() {
        assert_eq!(to_latex(&Expr::var("x")), "x");
    }

    #[test]
    fn test_latex_add() {
        let e = Expr::add(Expr::var("x"), Expr::num(1.0));
        assert_eq!(to_latex(&e), "x + 1");
    }

    #[test]
    fn test_latex_sub() {
        let e = Expr::sub(Expr::var("x"), Expr::num(1.0));
        assert_eq!(to_latex(&e), "x - 1");
    }

    #[test]
    fn test_latex_sub_with_grouped_rhs() {
        let e = Expr::sub(
            Expr::var("x"),
            Expr::add(Expr::var("y"), Expr::num(1.0)),
        );
        assert_eq!(to_latex(&e), "x - \\left(y + 1\\right)");
    }

    #[test]
    fn test_latex_mul() {
        let e = Expr::mul(Expr::var("x"), Expr::var("y"));
        assert_eq!(to_latex(&e), "x \\cdot y");
    }

    #[test]
    fn test_latex_mul_grouping() {
        let e = Expr::mul(
            Expr::add(Expr::var("x"), Expr::num(1.0)),
            Expr::var("y"),
        );
        assert_eq!(to_latex(&e), "\\left(x + 1\\right) \\cdot y");
    }

    #[test]
    fn test_latex_div() {
        let e = Expr::div(Expr::var("x"), Expr::num(2.0));
        assert_eq!(to_latex(&e), "\\frac{x}{2}");
    }

    #[test]
    fn test_latex_pow() {
        let e = Expr::pow(Expr::var("x"), Expr::num(2.0));
        assert_eq!(to_latex(&e), "x^{2}");
    }

    #[test]
    fn test_latex_pow_complex_base() {
        let e = Expr::pow(
            Expr::add(Expr::var("x"), Expr::num(1.0)),
            Expr::num(2.0),
        );
        assert_eq!(to_latex(&e), "\\left(x + 1\\right)^{2}");
    }

    #[test]
    fn test_latex_neg_var() {
        let e = Expr::neg(Expr::var("x"));
        assert_eq!(to_latex(&e), "-x");
    }

    #[test]
    fn test_latex_neg_compound() {
        let e = Expr::neg(Expr::add(Expr::var("x"), Expr::num(1.0)));
        assert_eq!(to_latex(&e), "-\\left(x + 1\\right)");
    }

    #[test]
    fn test_latex_sin() {
        let e = Expr::func("sin", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\sin\\left(x\\right)");
    }

    #[test]
    fn test_latex_cos() {
        let e = Expr::func("cos", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\cos\\left(x\\right)");
    }

    #[test]
    fn test_latex_tan() {
        let e = Expr::func("tan", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\tan\\left(x\\right)");
    }

    #[test]
    fn test_latex_exp() {
        let e = Expr::func("exp", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "e^{x}");
    }

    #[test]
    fn test_latex_ln() {
        let e = Expr::func("ln", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\ln\\left(x\\right)");
    }

    #[test]
    fn test_latex_log() {
        let e = Expr::func("log", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\ln\\left(x\\right)");
    }

    #[test]
    fn test_latex_sqrt() {
        let e = Expr::func("sqrt", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\sqrt{x}");
    }

    #[test]
    fn test_latex_abs() {
        let e = Expr::func("abs", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\left|x\\right|");
    }

    #[test]
    fn test_latex_unknown_function() {
        let e = Expr::func("foobar", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\mathrm{foobar}\\left(x\\right)");
    }

    #[test]
    fn test_latex_complex_expression() {
        // x^2 + sin(x) / 2
        let e = Expr::add(
            Expr::pow(Expr::var("x"), Expr::num(2.0)),
            Expr::div(
                Expr::func("sin", vec![Expr::var("x")]),
                Expr::num(2.0),
            ),
        );
        assert_eq!(
            to_latex(&e),
            "x^{2} + \\frac{\\sin\\left(x\\right)}{2}"
        );
    }

    #[test]
    fn test_latex_nested_fraction() {
        // (x + 1) / (x - 1)
        let e = Expr::div(
            Expr::add(Expr::var("x"), Expr::num(1.0)),
            Expr::sub(Expr::var("x"), Expr::num(1.0)),
        );
        assert_eq!(to_latex(&e), "\\frac{x + 1}{x - 1}");
    }

    #[test]
    fn test_latex_arcsin() {
        let e = Expr::func("asin", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\arcsin\\left(x\\right)");
    }

    #[test]
    fn test_latex_arccos() {
        let e = Expr::func("acos", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\arccos\\left(x\\right)");
    }

    #[test]
    fn test_latex_arctan() {
        let e = Expr::func("atan", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\arctan\\left(x\\right)");
    }

    #[test]
    fn test_latex_sinh() {
        let e = Expr::func("sinh", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\sinh\\left(x\\right)");
    }

    #[test]
    fn test_latex_cosh() {
        let e = Expr::func("cosh", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\cosh\\left(x\\right)");
    }

    #[test]
    fn test_latex_tanh() {
        let e = Expr::func("tanh", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\tanh\\left(x\\right)");
    }

    #[test]
    fn test_latex_asinh() {
        let e = Expr::func("asinh", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\operatorname{asinh}\\left(x\\right)");
    }

    #[test]
    fn test_latex_acosh() {
        let e = Expr::func("acosh", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\operatorname{acosh}\\left(x\\right)");
    }

    #[test]
    fn test_latex_atanh() {
        let e = Expr::func("atanh", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\operatorname{atanh}\\left(x\\right)");
    }

    #[test]
    fn test_latex_floor() {
        let e = Expr::func("floor", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\lfloor x \\rfloor");
    }

    #[test]
    fn test_latex_ceil() {
        let e = Expr::func("ceil", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\lceil x \\rceil");
    }

    #[test]
    fn test_latex_sign() {
        let e = Expr::func("sign", vec![Expr::var("x")]);
        assert_eq!(to_latex(&e), "\\operatorname{sgn}\\left(x\\right)");
    }

    #[test]
    fn test_latex_pow_with_func_base() {
        // sin(x)^2
        let e = Expr::pow(
            Expr::func("sin", vec![Expr::var("x")]),
            Expr::num(2.0),
        );
        assert_eq!(to_latex(&e), "\\sin\\left(x\\right)^{2}");
    }
}
