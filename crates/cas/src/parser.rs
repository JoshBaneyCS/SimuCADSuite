use crate::ast::{BinOp, Expr, UnaryOp};
use simucad_core::error::CasError;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a mathematical expression string into an `Expr` AST.
///
/// Operator precedence (lowest to highest):
///   1. `+`, `-`
///   2. `*`, `/`
///   3. unary `-`
///   4. `^` (right-associative)
///   5. atoms: numbers, variables, function calls, parenthesised sub-expressions
pub fn parse(input: &str) -> Result<Expr, CasError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser {
        tokens,
        pos: 0,
        input,
    };
    let expr = parser.parse_expr()?;
    if parser.pos < parser.tokens.len() {
        let tok = &parser.tokens[parser.pos];
        return Err(CasError::ParseError {
            position: tok.start,
            message: format!("unexpected token: {:?}", tok.kind),
        });
    }
    Ok(expr)
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
    start: usize,
}

// ---------------------------------------------------------------------------
// Tokeniser (lexer)
// ---------------------------------------------------------------------------

fn tokenize(input: &str) -> Result<Vec<Token>, CasError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        // Skip whitespace
        if ch.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        let start = i;

        match ch {
            '+' => {
                tokens.push(Token { kind: TokenKind::Plus, start });
                i += 1;
            }
            '-' => {
                tokens.push(Token { kind: TokenKind::Minus, start });
                i += 1;
            }
            '*' => {
                tokens.push(Token { kind: TokenKind::Star, start });
                i += 1;
            }
            '/' => {
                tokens.push(Token { kind: TokenKind::Slash, start });
                i += 1;
            }
            '^' => {
                tokens.push(Token { kind: TokenKind::Caret, start });
                i += 1;
            }
            '(' => {
                tokens.push(Token { kind: TokenKind::LParen, start });
                i += 1;
            }
            ')' => {
                tokens.push(Token { kind: TokenKind::RParen, start });
                i += 1;
            }
            ',' => {
                tokens.push(Token { kind: TokenKind::Comma, start });
                i += 1;
            }
            _ if ch.is_ascii_digit() || ch == '.' => {
                let mut end = i;
                let mut has_dot = ch == '.';
                end += 1;
                while end < chars.len() && (chars[end].is_ascii_digit() || chars[end] == '.') {
                    if chars[end] == '.' {
                        if has_dot {
                            break;
                        }
                        has_dot = true;
                    }
                    end += 1;
                }
                let num_str: String = chars[i..end].iter().collect();
                let value: f64 = num_str.parse().map_err(|_| CasError::ParseError {
                    position: start,
                    message: format!("invalid number: {}", num_str),
                })?;
                tokens.push(Token {
                    kind: TokenKind::Number(value),
                    start,
                });
                i = end;
            }
            _ if ch.is_ascii_alphabetic() || ch == '_' => {
                let mut end = i + 1;
                while end < chars.len()
                    && (chars[end].is_ascii_alphanumeric() || chars[end] == '_')
                {
                    end += 1;
                }
                let name: String = chars[i..end].iter().collect();
                tokens.push(Token {
                    kind: TokenKind::Ident(name),
                    start,
                });
                i = end;
            }
            _ => {
                return Err(CasError::ParseError {
                    position: start,
                    message: format!("unexpected character: '{}'", ch),
                });
            }
        }
    }

    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Recursive-descent parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    input: &'a str,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&TokenKind> {
        self.tokens.get(self.pos).map(|t| &t.kind)
    }

    fn current_position(&self) -> usize {
        self.tokens
            .get(self.pos)
            .map(|t| t.start)
            .unwrap_or(self.input.len())
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &TokenKind) -> Result<Token, CasError> {
        if self.peek() == Some(expected) {
            Ok(self.advance())
        } else {
            Err(CasError::ParseError {
                position: self.current_position(),
                message: format!(
                    "expected {:?}, found {:?}",
                    expected,
                    self.peek()
                ),
            })
        }
    }

    // expr = term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<Expr, CasError> {
        let mut lhs = self.parse_term()?;
        loop {
            match self.peek() {
                Some(TokenKind::Plus) => {
                    self.advance();
                    let rhs = self.parse_term()?;
                    lhs = Expr::BinOp {
                        op: BinOp::Add,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                Some(TokenKind::Minus) => {
                    self.advance();
                    let rhs = self.parse_term()?;
                    lhs = Expr::BinOp {
                        op: BinOp::Sub,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    // term = unary (('*' | '/') unary)*
    fn parse_term(&mut self) -> Result<Expr, CasError> {
        let mut lhs = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(TokenKind::Star) => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    lhs = Expr::BinOp {
                        op: BinOp::Mul,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                Some(TokenKind::Slash) => {
                    self.advance();
                    let rhs = self.parse_unary()?;
                    lhs = Expr::BinOp {
                        op: BinOp::Div,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    // unary = '-' unary | power
    fn parse_unary(&mut self) -> Result<Expr, CasError> {
        if self.peek() == Some(&TokenKind::Minus) {
            self.advance();
            let operand = self.parse_unary()?;
            Ok(Expr::UnaryOp {
                op: UnaryOp::Neg,
                operand: Box::new(operand),
            })
        } else {
            self.parse_power()
        }
    }

    // power = atom ('^' unary)?   (right-associative)
    fn parse_power(&mut self) -> Result<Expr, CasError> {
        let base = self.parse_atom()?;
        if self.peek() == Some(&TokenKind::Caret) {
            self.advance();
            // Right-associative: parse the exponent as a unary (which recurses into power)
            let exp = self.parse_unary()?;
            Ok(Expr::BinOp {
                op: BinOp::Pow,
                lhs: Box::new(base),
                rhs: Box::new(exp),
            })
        } else {
            Ok(base)
        }
    }

    // atom = Number | Ident | Ident '(' args ')' | '(' expr ')'
    fn parse_atom(&mut self) -> Result<Expr, CasError> {
        match self.peek().cloned() {
            Some(TokenKind::Number(n)) => {
                self.advance();
                Ok(Expr::Num(n))
            }
            Some(TokenKind::Ident(name)) => {
                self.advance();
                // Check for function call
                if self.peek() == Some(&TokenKind::LParen) {
                    self.advance(); // consume '('
                    let mut args = Vec::new();
                    if self.peek() != Some(&TokenKind::RParen) {
                        args.push(self.parse_expr()?);
                        while self.peek() == Some(&TokenKind::Comma) {
                            self.advance();
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(&TokenKind::RParen)?;
                    Ok(Expr::Func { name, args })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Some(TokenKind::LParen) => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(expr)
            }
            other => Err(CasError::ParseError {
                position: self.current_position(),
                message: format!("expected number, variable, or '(', found {:?}", other),
            }),
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
    fn test_parse_number() {
        let e = parse("42").unwrap();
        assert_eq!(e, Expr::num(42.0));
    }

    #[test]
    fn test_parse_float() {
        let e = parse("3.14").unwrap();
        assert_eq!(e, Expr::num(3.14));
    }

    #[test]
    fn test_parse_variable() {
        let e = parse("x").unwrap();
        assert_eq!(e, Expr::var("x"));
    }

    #[test]
    fn test_parse_addition() {
        let e = parse("x + 1").unwrap();
        assert_eq!(e, Expr::add(Expr::var("x"), Expr::num(1.0)));
    }

    #[test]
    fn test_parse_subtraction() {
        let e = parse("x - 2").unwrap();
        assert_eq!(e, Expr::sub(Expr::var("x"), Expr::num(2.0)));
    }

    #[test]
    fn test_parse_multiplication() {
        let e = parse("x * y").unwrap();
        assert_eq!(e, Expr::mul(Expr::var("x"), Expr::var("y")));
    }

    #[test]
    fn test_parse_division() {
        let e = parse("x / 2").unwrap();
        assert_eq!(e, Expr::div(Expr::var("x"), Expr::num(2.0)));
    }

    #[test]
    fn test_parse_power() {
        let e = parse("x^2").unwrap();
        assert_eq!(e, Expr::pow(Expr::var("x"), Expr::num(2.0)));
    }

    #[test]
    fn test_parse_unary_neg() {
        let e = parse("-x").unwrap();
        assert_eq!(e, Expr::neg(Expr::var("x")));
    }

    #[test]
    fn test_parse_function_call() {
        let e = parse("sin(x)").unwrap();
        assert_eq!(e, Expr::func("sin", vec![Expr::var("x")]));
    }

    #[test]
    fn test_parse_nested_function() {
        let e = parse("cos(sin(x))").unwrap();
        assert_eq!(
            e,
            Expr::func("cos", vec![Expr::func("sin", vec![Expr::var("x")])])
        );
    }

    #[test]
    fn test_parse_precedence_add_mul() {
        // 1 + 2 * 3 → 1 + (2 * 3)
        let e = parse("1 + 2 * 3").unwrap();
        assert_eq!(
            e,
            Expr::add(Expr::num(1.0), Expr::mul(Expr::num(2.0), Expr::num(3.0)))
        );
    }

    #[test]
    fn test_parse_precedence_mul_pow() {
        // 2 * x^3 → 2 * (x^3)
        let e = parse("2 * x^3").unwrap();
        assert_eq!(
            e,
            Expr::mul(Expr::num(2.0), Expr::pow(Expr::var("x"), Expr::num(3.0)))
        );
    }

    #[test]
    fn test_parse_parens_override() {
        // (1 + 2) * 3
        let e = parse("(1 + 2) * 3").unwrap();
        assert_eq!(
            e,
            Expr::mul(Expr::add(Expr::num(1.0), Expr::num(2.0)), Expr::num(3.0))
        );
    }

    #[test]
    fn test_parse_power_right_associative() {
        // 2^3^4 → 2^(3^4)
        let e = parse("2^3^4").unwrap();
        assert_eq!(
            e,
            Expr::pow(
                Expr::num(2.0),
                Expr::pow(Expr::num(3.0), Expr::num(4.0))
            )
        );
    }

    #[test]
    fn test_parse_pi() {
        let e = parse("pi").unwrap();
        assert_eq!(e, Expr::var("pi"));
    }

    #[test]
    fn test_parse_complex_expression() {
        // sin(x)^2 + cos(x)^2
        let e = parse("sin(x)^2 + cos(x)^2").unwrap();
        assert_eq!(
            e,
            Expr::add(
                Expr::pow(Expr::func("sin", vec![Expr::var("x")]), Expr::num(2.0)),
                Expr::pow(Expr::func("cos", vec![Expr::var("x")]), Expr::num(2.0)),
            )
        );
    }

    #[test]
    fn test_parse_negative_in_expression() {
        // -x + 1
        let e = parse("-x + 1").unwrap();
        assert_eq!(e, Expr::add(Expr::neg(Expr::var("x")), Expr::num(1.0)));
    }

    #[test]
    fn test_parse_error_unexpected_char() {
        let result = parse("x & y");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_unmatched_paren() {
        let result = parse("(x + 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_empty() {
        let result = parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_whitespace_handling() {
        let e = parse("  x  +  1  ").unwrap();
        assert_eq!(e, Expr::add(Expr::var("x"), Expr::num(1.0)));
    }

    #[test]
    fn test_parse_multi_arg_function() {
        // Hypothetical: max(x, y)  — the parser supports it even if evaluator may not
        let e = parse("max(x, y)").unwrap();
        assert_eq!(
            e,
            Expr::Func {
                name: "max".into(),
                args: vec![Expr::var("x"), Expr::var("y")],
            }
        );
    }
}
