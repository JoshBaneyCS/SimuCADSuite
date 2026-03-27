use std::fmt;

/// Binary operators supported by the CAS.
#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Unary operators supported by the CAS.
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
}

/// The core expression tree for the computer algebra system.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A numeric literal.
    Num(f64),
    /// A named variable (or constant like `pi`, `e`).
    Var(String),
    /// A binary operation: `lhs op rhs`.
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// A unary operation applied to an operand.
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    /// A function call, e.g. `sin(x)`.
    Func {
        name: String,
        args: Vec<Expr>,
    },
}

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

impl Expr {
    pub fn num(n: f64) -> Self {
        Expr::Num(n)
    }

    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    pub fn add(lhs: Expr, rhs: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn sub(lhs: Expr, rhs: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Sub,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn mul(lhs: Expr, rhs: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Mul,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn div(lhs: Expr, rhs: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Div,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn pow(lhs: Expr, rhs: Expr) -> Self {
        Expr::BinOp {
            op: BinOp::Pow,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn neg(operand: Expr) -> Self {
        Expr::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(operand),
        }
    }

    pub fn func(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::Func {
            name: name.into(),
            args,
        }
    }

    /// Returns `true` when the expression is the numeric literal `0`.
    pub fn is_zero(&self) -> bool {
        matches!(self, Expr::Num(v) if *v == 0.0)
    }

    /// Returns `true` when the expression is the numeric literal `1`.
    pub fn is_one(&self) -> bool {
        matches!(self, Expr::Num(v) if *v == 1.0)
    }
}

// ---------------------------------------------------------------------------
// Display — human-readable infix notation
// ---------------------------------------------------------------------------

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Num(n) => {
                if *n == (*n as i64) as f64 && n.is_finite() {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            Expr::Var(name) => write!(f, "{}", name),
            Expr::BinOp { op, lhs, rhs } => {
                let op_str = match op {
                    BinOp::Add => " + ",
                    BinOp::Sub => " - ",
                    BinOp::Mul => " * ",
                    BinOp::Div => " / ",
                    BinOp::Pow => "^",
                };
                let needs_lhs_parens = match op {
                    BinOp::Mul | BinOp::Div => matches!(
                        lhs.as_ref(),
                        Expr::BinOp {
                            op: BinOp::Add | BinOp::Sub,
                            ..
                        }
                    ),
                    BinOp::Pow => !matches!(lhs.as_ref(), Expr::Num(_) | Expr::Var(_)),
                    _ => false,
                };
                let needs_rhs_parens = match op {
                    BinOp::Mul | BinOp::Div => matches!(
                        rhs.as_ref(),
                        Expr::BinOp {
                            op: BinOp::Add | BinOp::Sub,
                            ..
                        }
                    ),
                    BinOp::Sub => matches!(
                        rhs.as_ref(),
                        Expr::BinOp {
                            op: BinOp::Add | BinOp::Sub,
                            ..
                        }
                    ),
                    BinOp::Pow => !matches!(rhs.as_ref(), Expr::Num(_) | Expr::Var(_)),
                    _ => false,
                };

                if needs_lhs_parens {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, "{}", op_str)?;
                if needs_rhs_parens {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::UnaryOp {
                op: UnaryOp::Neg,
                operand,
            } => match operand.as_ref() {
                Expr::Num(_) | Expr::Var(_) => write!(f, "-{}", operand),
                _ => write!(f, "-({})", operand),
            },
            Expr::Func { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Pow => write!(f, "^"),
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
    fn test_display_num_integer() {
        assert_eq!(Expr::num(3.0).to_string(), "3");
    }

    #[test]
    fn test_display_num_float() {
        assert_eq!(Expr::num(3.14).to_string(), "3.14");
    }

    #[test]
    fn test_display_var() {
        assert_eq!(Expr::var("x").to_string(), "x");
    }

    #[test]
    fn test_display_add() {
        let e = Expr::add(Expr::var("x"), Expr::num(1.0));
        assert_eq!(e.to_string(), "x + 1");
    }

    #[test]
    fn test_display_mul_with_add_parens() {
        // (x + 1) * y  →  should parenthesise the addition
        let e = Expr::mul(Expr::add(Expr::var("x"), Expr::num(1.0)), Expr::var("y"));
        assert_eq!(e.to_string(), "(x + 1) * y");
    }

    #[test]
    fn test_display_pow() {
        let e = Expr::pow(Expr::var("x"), Expr::num(2.0));
        assert_eq!(e.to_string(), "x^2");
    }

    #[test]
    fn test_display_neg_var() {
        assert_eq!(Expr::neg(Expr::var("x")).to_string(), "-x");
    }

    #[test]
    fn test_display_neg_compound() {
        let e = Expr::neg(Expr::add(Expr::var("x"), Expr::num(1.0)));
        assert_eq!(e.to_string(), "-(x + 1)");
    }

    #[test]
    fn test_display_func() {
        let e = Expr::func("sin", vec![Expr::var("x")]);
        assert_eq!(e.to_string(), "sin(x)");
    }

    #[test]
    fn test_is_zero() {
        assert!(Expr::num(0.0).is_zero());
        assert!(!Expr::num(1.0).is_zero());
        assert!(!Expr::var("x").is_zero());
    }

    #[test]
    fn test_is_one() {
        assert!(Expr::num(1.0).is_one());
        assert!(!Expr::num(0.0).is_one());
    }

    #[test]
    fn test_clone_and_eq() {
        let e = Expr::add(Expr::var("x"), Expr::num(2.0));
        let e2 = e.clone();
        assert_eq!(e, e2);
    }
}
