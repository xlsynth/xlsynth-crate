// SPDX-License-Identifier: Apache-2.0

use crate::Error;
use crate::Result;
use crate::ast::BinaryOp;
use crate::ast::Expr;
use crate::ast::UnaryOp;
use crate::lexer::Lexer;
use crate::lexer::Token;
use crate::value::LogicBit;
use crate::value::Signedness;
use crate::value::Value4;

const MAX_LITERAL_WIDTH_BITS: u32 = 1_000_000;

fn validate_literal_width(width: u32, literal: &str) -> Result<()> {
    if width > MAX_LITERAL_WIDTH_BITS {
        return Err(Error::Parse(format!(
            "literal width {width} exceeds max supported width {MAX_LITERAL_WIDTH_BITS}: {literal}"
        )));
    }
    Ok(())
}

fn expr_is_constant(expr: &Expr) -> bool {
    match expr {
        Expr::Ident(_) => false,
        Expr::Literal(_) | Expr::UnsizedNumber(_) | Expr::UnbasedUnsized(_) => true,
        Expr::Call { .. } => false,
        Expr::Concat(parts) => parts.iter().all(expr_is_constant),
        Expr::Replicate { count, expr } => expr_is_constant(count) && expr_is_constant(expr),
        Expr::Cast { width, expr } => expr_is_constant(width) && expr_is_constant(expr),
        Expr::Index { expr, index } => expr_is_constant(expr) && expr_is_constant(index),
        Expr::Slice { expr, msb, lsb } => {
            expr_is_constant(expr) && expr_is_constant(msb) && expr_is_constant(lsb)
        }
        Expr::IndexedSlice {
            expr, base, width, ..
        } => expr_is_constant(expr) && expr_is_constant(base) && expr_is_constant(width),
        Expr::Unary { expr, .. } => expr_is_constant(expr),
        Expr::Binary { lhs, rhs, .. } => expr_is_constant(lhs) && expr_is_constant(rhs),
        Expr::Ternary { cond, t, f } => {
            expr_is_constant(cond) && expr_is_constant(t) && expr_is_constant(f)
        }
    }
}

pub fn parse_expr(input: &str) -> Result<Expr> {
    let mut p = Parser::new(input)?;
    let expr = p.parse_ternary()?;
    match p.cur {
        Token::End => Ok(expr),
        _ => Err(Error::Parse(format!(
            "unexpected trailing token: {:?}",
            p.cur
        ))),
    }
}

struct Parser<'a> {
    lex: Lexer<'a>,
    cur: Token,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Result<Self> {
        let mut lex = Lexer::new(s);
        let cur = lex.next_token()?;
        Ok(Self { lex, cur })
    }

    fn bump(&mut self) -> Result<()> {
        self.cur = self.lex.next_token()?;
        Ok(())
    }

    fn expect_and_bump(&mut self, want: Token) -> Result<()> {
        if self.cur == want {
            self.bump()
        } else {
            Err(Error::Parse(format!(
                "expected {:?}, got {:?}",
                want, self.cur
            )))
        }
    }

    // Lowest precedence: ternary (right-associative).
    fn parse_ternary(&mut self) -> Result<Expr> {
        let mut cond = self.parse_logical_or()?;
        if self.cur == Token::Question {
            self.bump()?;
            let t = self.parse_ternary()?;
            self.expect_and_bump(Token::Colon)?;
            let f = self.parse_ternary()?;
            cond = Expr::Ternary {
                cond: Box::new(cond),
                t: Box::new(t),
                f: Box::new(f),
            };
        }
        Ok(cond)
    }

    fn parse_logical_or(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_logical_and()?;
        while self.cur == Token::OrOr {
            self.bump()?;
            let rhs = self.parse_logical_and()?;
            lhs = Expr::Binary {
                op: BinaryOp::LogicalOr,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_logical_and(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_bitwise_or()?;
        while self.cur == Token::AndAnd {
            self.bump()?;
            let rhs = self.parse_bitwise_or()?;
            lhs = Expr::Binary {
                op: BinaryOp::LogicalAnd,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_bitwise_or(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_bitwise_xor()?;
        while self.cur == Token::Or {
            self.bump()?;
            let rhs = self.parse_bitwise_xor()?;
            lhs = Expr::Binary {
                op: BinaryOp::BitOr,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_bitwise_xor(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_bitwise_and()?;
        while self.cur == Token::Caret {
            self.bump()?;
            let rhs = self.parse_bitwise_and()?;
            lhs = Expr::Binary {
                op: BinaryOp::BitXor,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_bitwise_and(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_equality()?;
        while self.cur == Token::And {
            self.bump()?;
            let rhs = self.parse_equality()?;
            lhs = Expr::Binary {
                op: BinaryOp::BitAnd,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_equality(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_relational()?;
        loop {
            let op = match self.cur {
                Token::EqEq => Some(BinaryOp::Eq),
                Token::BangEq => Some(BinaryOp::Neq),
                Token::EqEqEq => Some(BinaryOp::CaseEq),
                Token::BangEqEq => Some(BinaryOp::CaseNeq),
                _ => None,
            };
            let Some(op) = op else { break };
            self.bump()?;
            let rhs = self.parse_relational()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_relational(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_shift()?;
        loop {
            let op = match self.cur {
                Token::Lt => Some(BinaryOp::Lt),
                Token::Le => Some(BinaryOp::Le),
                Token::Gt => Some(BinaryOp::Gt),
                Token::Ge => Some(BinaryOp::Ge),
                _ => None,
            };
            let Some(op) = op else { break };
            self.bump()?;
            let rhs = self.parse_shift()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = match self.cur {
                Token::Shl => Some(BinaryOp::Shl),
                Token::Shr => Some(BinaryOp::Shr),
                Token::Sshr => Some(BinaryOp::Sshr),
                _ => None,
            };
            let Some(op) = op else { break };
            self.bump()?;
            let rhs = self.parse_additive()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.cur {
                Token::Plus => Some(BinaryOp::Add),
                Token::Minus => Some(BinaryOp::Sub),
                _ => None,
            };
            let Some(op) = op else { break };
            self.bump()?;
            let rhs = self.parse_multiplicative()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.cur {
                Token::Star => Some(BinaryOp::Mul),
                Token::Slash => Some(BinaryOp::Div),
                Token::Percent => Some(BinaryOp::Mod),
                _ => None,
            };
            let Some(op) = op else { break };
            self.bump()?;
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.cur.clone() {
            Token::Bang => {
                self.bump()?;
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::LogicalNot,
                    expr: Box::new(e),
                })
            }
            Token::Tilde => {
                self.bump()?;
                // Could be ~, ~&, ~|, ~^
                match self.cur.clone() {
                    Token::And => {
                        self.bump()?;
                        let e = self.parse_unary()?;
                        Ok(Expr::Unary {
                            op: UnaryOp::ReduceNand,
                            expr: Box::new(e),
                        })
                    }
                    Token::Or => {
                        self.bump()?;
                        let e = self.parse_unary()?;
                        Ok(Expr::Unary {
                            op: UnaryOp::ReduceNor,
                            expr: Box::new(e),
                        })
                    }
                    Token::Caret => {
                        self.bump()?;
                        let e = self.parse_unary()?;
                        Ok(Expr::Unary {
                            op: UnaryOp::ReduceXnor,
                            expr: Box::new(e),
                        })
                    }
                    _ => {
                        let e = self.parse_unary()?;
                        Ok(Expr::Unary {
                            op: UnaryOp::BitNot,
                            expr: Box::new(e),
                        })
                    }
                }
            }
            Token::And => {
                self.bump()?;
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::ReduceAnd,
                    expr: Box::new(e),
                })
            }
            Token::Or => {
                self.bump()?;
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::ReduceOr,
                    expr: Box::new(e),
                })
            }
            Token::Caret => {
                self.bump()?;
                // ^~ is tokenized as '^' then '~'; handle both ^~ and plain ^.
                let op = if self.cur == Token::Tilde {
                    self.bump()?;
                    UnaryOp::ReduceXnor
                } else {
                    UnaryOp::ReduceXor
                };
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op,
                    expr: Box::new(e),
                })
            }
            Token::Plus => {
                self.bump()?;
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::UnaryPlus,
                    expr: Box::new(e),
                })
            }
            Token::Minus => {
                self.bump()?;
                let e = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::UnaryMinus,
                    expr: Box::new(e),
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        let mut expr = match self.cur.clone() {
            Token::Ident(name) => {
                self.bump()?;
                // Function call: ident '(' args ')'
                if self.cur == Token::LParen {
                    self.bump()?;
                    let mut args: Vec<Expr> = Vec::new();
                    if self.cur != Token::RParen {
                        args.push(self.parse_ternary()?);
                        while self.cur == Token::Comma {
                            self.bump()?;
                            args.push(self.parse_ternary()?);
                        }
                    }
                    self.expect_and_bump(Token::RParen)?;
                    Expr::Call { name, args }
                } else {
                    Expr::Ident(name)
                }
            }
            Token::Number(lit) => {
                self.bump()?;
                // Unbased unsized literals: `'0`, `'1`, `'x`, `'z`.
                if let Some(rest) = lit.strip_prefix('\'') {
                    if rest.len() == 1 {
                        let ch = rest.chars().next().unwrap();
                        let lb = match ch {
                            '0' => Some(LogicBit::Zero),
                            '1' => Some(LogicBit::One),
                            'x' | 'X' => Some(LogicBit::X),
                            'z' | 'Z' => Some(LogicBit::Z),
                            _ => None,
                        };
                        if let Some(lb) = lb {
                            return Ok(Expr::UnbasedUnsized(lb));
                        }
                    }
                }
                let (v, is_unsized_number) = parse_verilog_number_with_origin(&lit)?;
                if is_unsized_number {
                    Expr::UnsizedNumber(v)
                } else {
                    Expr::Literal(v)
                }
            }
            Token::LParen => {
                self.bump()?;
                let e = self.parse_ternary()?;
                self.expect_and_bump(Token::RParen)?;
                e
            }
            Token::LBrace => {
                self.bump()?;
                // Either concat: {e1, e2, ...} or replication: {N{e}}
                let first = self.parse_ternary()?;
                if self.cur == Token::LBrace {
                    self.bump()?;
                    let inner = self.parse_ternary()?;
                    self.expect_and_bump(Token::RBrace)?;
                    self.expect_and_bump(Token::RBrace)?;
                    if !expr_is_constant(&first) {
                        return Err(Error::Parse(
                            "replication count must be a constant expression".to_string(),
                        ));
                    }
                    Expr::Replicate {
                        count: Box::new(first),
                        expr: Box::new(inner),
                    }
                } else {
                    let mut parts = vec![first];
                    while self.cur == Token::Comma {
                        self.bump()?;
                        parts.push(self.parse_ternary()?);
                    }
                    self.expect_and_bump(Token::RBrace)?;
                    if parts.iter().any(|p| matches!(p, Expr::UnsizedNumber(_))) {
                        return Err(Error::Parse(
                            "unsized constant numbers are illegal in concatenations".to_string(),
                        ));
                    }
                    Expr::Concat(parts)
                }
            }
            t => {
                return Err(Error::Parse(format!(
                    "unexpected token in primary: {:?}",
                    t
                )));
            }
        };

        // Postfix sized-casts/selects: a'(expr), a[expr], a[msb:lsb]
        loop {
            match self.cur {
                Token::Apostrophe => {
                    self.bump()?;
                    self.expect_and_bump(Token::LParen)?;
                    let inner = self.parse_ternary()?;
                    self.expect_and_bump(Token::RParen)?;
                    expr = Expr::Cast {
                        width: Box::new(expr),
                        expr: Box::new(inner),
                    };
                }
                Token::LBracket => {
                    self.bump()?;
                    let idx_or_msb = self.parse_ternary()?;
                    if self.cur == Token::Colon {
                        self.bump()?;
                        let lsb = self.parse_ternary()?;
                        self.expect_and_bump(Token::RBracket)?;
                        if !expr_is_constant(&idx_or_msb) || !expr_is_constant(&lsb) {
                            return Err(Error::Parse(
                                "part-select bounds must be constant expressions".to_string(),
                            ));
                        }
                        expr = Expr::Slice {
                            expr: Box::new(expr),
                            msb: Box::new(idx_or_msb),
                            lsb: Box::new(lsb),
                        };
                    } else if matches!(self.cur, Token::PlusColon | Token::MinusColon) {
                        let upward = self.cur == Token::PlusColon;
                        self.bump()?;
                        let width = self.parse_ternary()?;
                        self.expect_and_bump(Token::RBracket)?;
                        if !expr_is_constant(&width) {
                            return Err(Error::Parse(
                                "indexed part-select width must be a constant expression"
                                    .to_string(),
                            ));
                        }
                        expr = Expr::IndexedSlice {
                            expr: Box::new(expr),
                            base: Box::new(idx_or_msb),
                            width: Box::new(width),
                            upward,
                        };
                    } else {
                        self.expect_and_bump(Token::RBracket)?;
                        expr = Expr::Index {
                            expr: Box::new(expr),
                            index: Box::new(idx_or_msb),
                        };
                    }
                }
                _ => break,
            }
        }

        Ok(expr)
    }
}

fn parse_verilog_number_with_origin(s: &str) -> Result<(Value4, bool)> {
    // Strip underscores; they are allowed as separators.
    let compact: String = s.chars().filter(|c| *c != '_').collect();
    let s = compact.as_str();

    // Based form: [size] ' [s] [base] digits
    if let Some(tick_idx) = s.find('\'') {
        let (size_part, rest) = s.split_at(tick_idx);
        let rest = &rest[1..]; // drop '
        if rest.is_empty() {
            return Err(Error::Parse(format!(
                "bad based literal (missing base): {s}"
            )));
        }

        let mut chars = rest.chars();
        let mut signed = false;
        let base_ch = match chars.next().unwrap() {
            's' | 'S' => {
                signed = true;
                chars
                    .next()
                    .ok_or_else(|| Error::Parse(format!("bad based literal: {s}")))?
            }
            c => c,
        };

        let base = match base_ch {
            'b' | 'B' => 2,
            'o' | 'O' => 8,
            'd' | 'D' => 10,
            'h' | 'H' => 16,
            _ => return Err(Error::Parse(format!("unknown base specifier in {s}"))),
        };

        let digits: String = chars.collect();
        if digits.is_empty() {
            return Err(Error::Parse(format!(
                "bad based literal (missing digits): {s}"
            )));
        }

        let width_opt = if size_part.is_empty() {
            None
        } else {
            let width = size_part
                .parse::<u32>()
                .map_err(|_| Error::Parse(format!("bad sized literal width: {s}")))?;
            validate_literal_width(width, s)?;
            Some(width)
        };

        let signedness = if signed {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        };

        return Ok((
            parse_based_digits(width_opt, signedness, base, &digits)?,
            width_opt.is_none(),
        ));
    }

    // Unsized decimal literals are at least 32 bits and signed unless marked
    // otherwise.
    let v = Value4::parse_unsized_decimal_token(Signedness::Signed, s)?;
    validate_literal_width(v.width, s)?;
    Ok((v, true))
}

fn parse_based_digits(
    width_opt: Option<u32>,
    signedness: Signedness,
    base: u32,
    digits: &str,
) -> Result<Value4> {
    match base {
        2 => parse_based_bits(width_opt, signedness, 1, digits, parse_bin_digit),
        8 => parse_based_bits(width_opt, signedness, 3, digits, parse_oct_digit),
        16 => parse_based_bits(width_opt, signedness, 4, digits, parse_hex_digit),
        10 => parse_dec_based(width_opt, signedness, digits),
        _ => Err(Error::Parse(format!("unsupported base {base}"))),
    }
}

fn parse_dec_based(width_opt: Option<u32>, signedness: Signedness, digits: &str) -> Result<Value4> {
    let width = width_opt.unwrap_or(32);
    if digits.eq_ignore_ascii_case("?") {
        validate_literal_width(width, digits)?;
        return Ok(Value4::new(
            width,
            signedness,
            vec![LogicBit::Z; width as usize],
        ));
    }
    if digits.eq_ignore_ascii_case("x") {
        validate_literal_width(width, digits)?;
        return Ok(Value4::new(
            width,
            signedness,
            vec![LogicBit::X; width as usize],
        ));
    }
    if digits.eq_ignore_ascii_case("z") {
        validate_literal_width(width, digits)?;
        return Ok(Value4::new(
            width,
            signedness,
            vec![LogicBit::Z; width as usize],
        ));
    }
    if width_opt.is_none() {
        let v = Value4::parse_unsized_decimal_token(signedness, digits)?;
        validate_literal_width(v.width, digits)?;
        return Ok(v);
    }
    validate_literal_width(width, digits)?;
    if digits.chars().any(|c| !c.is_ascii_digit()) {
        return Err(Error::Parse(format!(
            "unsupported decimal digits in literal: {digits}"
        )));
    }
    Value4::parse_numeric_token(width, signedness, digits)
}

fn parse_based_bits<F>(
    width_opt: Option<u32>,
    signedness: Signedness,
    bits_per_digit: u32,
    digits: &str,
    parse_digit: F,
) -> Result<Value4>
where
    F: Fn(char) -> Result<Vec<LogicBit>>,
{
    let mut bits_msb_first: Vec<LogicBit> = Vec::new();
    for c in digits.chars() {
        let mut digit_bits = parse_digit(c)?;
        if digit_bits.len() != bits_per_digit as usize {
            return Err(Error::Parse(format!(
                "internal digit parse width mismatch for {c}"
            )));
        }
        bits_msb_first.append(&mut digit_bits);
    }

    let implied_width = bits_msb_first.len() as u32;
    let width = width_opt.unwrap_or(implied_width.max(32));
    validate_literal_width(width, digits)?;

    // If specified width is smaller, truncate from the left (MSB side).
    let bits_msb_first = if width < implied_width {
        bits_msb_first[(implied_width - width) as usize..].to_vec()
    } else if width > implied_width {
        let ext_bit = match bits_msb_first.first().copied().unwrap_or(LogicBit::Zero) {
            LogicBit::X => LogicBit::X,
            LogicBit::Z => LogicBit::Z,
            LogicBit::Zero | LogicBit::One => LogicBit::Zero,
        };
        let mut ext = vec![ext_bit; (width - implied_width) as usize];
        ext.extend(bits_msb_first);
        ext
    } else {
        bits_msb_first
    };

    Ok(Value4::from_bits_msb_first(
        width,
        signedness,
        &bits_msb_first,
    ))
}

fn parse_bin_digit(c: char) -> Result<Vec<LogicBit>> {
    let b = match c {
        '0' => LogicBit::Zero,
        '1' => LogicBit::One,
        'x' | 'X' => LogicBit::X,
        'z' | 'Z' | '?' => LogicBit::Z,
        _ => return Err(Error::Parse(format!("bad binary digit: {c}"))),
    };
    Ok(vec![b])
}

fn parse_oct_digit(c: char) -> Result<Vec<LogicBit>> {
    match c {
        'x' | 'X' => Ok(vec![LogicBit::X, LogicBit::X, LogicBit::X]),
        'z' | 'Z' | '?' => Ok(vec![LogicBit::Z, LogicBit::Z, LogicBit::Z]),
        _ => {
            let v = c
                .to_digit(8)
                .ok_or_else(|| Error::Parse(format!("bad octal digit: {c}")))?;
            let b2 = if (v & 4) != 0 {
                LogicBit::One
            } else {
                LogicBit::Zero
            };
            let b1 = if (v & 2) != 0 {
                LogicBit::One
            } else {
                LogicBit::Zero
            };
            let b0 = if (v & 1) != 0 {
                LogicBit::One
            } else {
                LogicBit::Zero
            };
            Ok(vec![b2, b1, b0])
        }
    }
}

fn parse_hex_digit(c: char) -> Result<Vec<LogicBit>> {
    match c {
        'x' | 'X' => Ok(vec![LogicBit::X, LogicBit::X, LogicBit::X, LogicBit::X]),
        'z' | 'Z' | '?' => Ok(vec![LogicBit::Z, LogicBit::Z, LogicBit::Z, LogicBit::Z]),
        _ => {
            let v = c
                .to_digit(16)
                .ok_or_else(|| Error::Parse(format!("bad hex digit: {c}")))?;
            let b3 = if (v & 8) != 0 {
                LogicBit::One
            } else {
                LogicBit::Zero
            };
            let b2 = if (v & 4) != 0 {
                LogicBit::One
            } else {
                LogicBit::Zero
            };
            let b1 = if (v & 2) != 0 {
                LogicBit::One
            } else {
                LogicBit::Zero
            };
            let b0 = if (v & 1) != 0 {
                LogicBit::One
            } else {
                LogicBit::Zero
            };
            Ok(vec![b3, b2, b1, b0])
        }
    }
}
