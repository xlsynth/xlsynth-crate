// SPDX-License-Identifier: Apache-2.0

use crate::ast::BinaryOp;
use crate::ast::Expr;
use crate::ast::UnaryOp;
use crate::ast_spanned::SpannedExpr;
use crate::ast_spanned::SpannedExprKind;
use crate::ast_spanned::SpannedTok;
use crate::lexer::Token;
use crate::lexer_spanned::LexerSpanned;
use crate::sv_ast::Span;
use crate::Error;
use crate::Result;

pub fn parse_expr_spanned(input: &str) -> Result<SpannedExpr> {
    let mut p = Parser::new(input)?;
    let expr = p.parse_ternary()?;
    match p.cur.tok {
        Token::End => Ok(expr),
        _ => Err(Error::Parse(format!(
            "unexpected trailing token: {:?}",
            p.cur.tok
        ))),
    }
}

struct Parser<'a> {
    lex: LexerSpanned<'a>,
    cur: SpannedTok,
}

fn spanned_expr_is_constant(expr: &SpannedExpr) -> bool {
    match &expr.kind {
        SpannedExprKind::Ident(_) => false,
        SpannedExprKind::Literal(_)
        | SpannedExprKind::UnsizedNumber(_)
        | SpannedExprKind::UnbasedUnsized(_) => true,
        SpannedExprKind::Call { .. } => false,
        SpannedExprKind::Concat(parts) => parts.iter().all(spanned_expr_is_constant),
        SpannedExprKind::Replicate { count, expr } => {
            spanned_expr_is_constant(count) && spanned_expr_is_constant(expr)
        }
        SpannedExprKind::Cast { width, expr } => {
            spanned_expr_is_constant(width) && spanned_expr_is_constant(expr)
        }
        SpannedExprKind::Index { expr, index } => {
            spanned_expr_is_constant(expr) && spanned_expr_is_constant(index)
        }
        SpannedExprKind::Slice { expr, msb, lsb } => {
            spanned_expr_is_constant(expr)
                && spanned_expr_is_constant(msb)
                && spanned_expr_is_constant(lsb)
        }
        SpannedExprKind::IndexedSlice {
            expr, base, width, ..
        } => {
            spanned_expr_is_constant(expr)
                && spanned_expr_is_constant(base)
                && spanned_expr_is_constant(width)
        }
        SpannedExprKind::Unary { expr, .. } => spanned_expr_is_constant(expr),
        SpannedExprKind::Binary { lhs, rhs, .. } => {
            spanned_expr_is_constant(lhs) && spanned_expr_is_constant(rhs)
        }
        SpannedExprKind::Ternary { cond, t, f } => {
            spanned_expr_is_constant(cond)
                && spanned_expr_is_constant(t)
                && spanned_expr_is_constant(f)
        }
    }
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Result<Self> {
        let mut lex = LexerSpanned::new(s);
        let cur = lex.next_token()?;
        Ok(Self { lex, cur })
    }

    fn bump(&mut self) -> Result<()> {
        self.cur = self.lex.next_token()?;
        Ok(())
    }

    fn expect_and_bump(&mut self, want: Token) -> Result<Span> {
        if self.cur.tok == want {
            let s = self.cur.span;
            self.bump()?;
            Ok(s)
        } else {
            Err(Error::Parse(format!(
                "expected {:?}, got {:?}",
                want, self.cur.tok
            )))
        }
    }

    fn parse_ternary(&mut self) -> Result<SpannedExpr> {
        let mut cond = self.parse_logical_or()?;
        if self.cur.tok == Token::Question {
            let q_span = self.expect_and_bump(Token::Question)?;
            let t = self.parse_ternary()?;
            let _ = self.expect_and_bump(Token::Colon)?;
            let f = self.parse_ternary()?;
            let span = Span {
                start: cond.span.start,
                end: f.span.end,
            };
            let _ = q_span;
            cond = SpannedExpr {
                span,
                kind: SpannedExprKind::Ternary {
                    cond: Box::new(cond),
                    t: Box::new(t),
                    f: Box::new(f),
                },
            };
        }
        Ok(cond)
    }

    fn parse_logical_or(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_logical_and()?;
        while self.cur.tok == Token::OrOr {
            let op_span = self.expect_and_bump(Token::OrOr)?;
            let rhs = self.parse_logical_and()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            let _ = op_span;
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op: BinaryOp::LogicalOr,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_logical_and(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_bitwise_or()?;
        while self.cur.tok == Token::AndAnd {
            let op_span = self.expect_and_bump(Token::AndAnd)?;
            let rhs = self.parse_bitwise_or()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            let _ = op_span;
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op: BinaryOp::LogicalAnd,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_bitwise_or(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_bitwise_xor()?;
        while self.cur.tok == Token::Or {
            let op_span = self.expect_and_bump(Token::Or)?;
            let rhs = self.parse_bitwise_xor()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            let _ = op_span;
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op: BinaryOp::BitOr,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_bitwise_xor(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_bitwise_and()?;
        while self.cur.tok == Token::Caret {
            let op_span = self.expect_and_bump(Token::Caret)?;
            let rhs = self.parse_bitwise_and()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            let _ = op_span;
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op: BinaryOp::BitXor,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_bitwise_and(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_equality()?;
        while self.cur.tok == Token::And {
            let op_span = self.expect_and_bump(Token::And)?;
            let rhs = self.parse_equality()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            let _ = op_span;
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_equality(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_relational()?;
        loop {
            let op = match self.cur.tok {
                Token::EqEq => Some(BinaryOp::Eq),
                Token::BangEq => Some(BinaryOp::Neq),
                Token::EqEqEq => Some(BinaryOp::CaseEq),
                Token::BangEqEq => Some(BinaryOp::CaseNeq),
                _ => None,
            };
            let Some(op) = op else { break };
            let _ = self.cur.span;
            self.bump()?;
            let rhs = self.parse_relational()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_relational(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_shift()?;
        loop {
            let op = match self.cur.tok {
                Token::Lt => Some(BinaryOp::Lt),
                Token::Le => Some(BinaryOp::Le),
                Token::Gt => Some(BinaryOp::Gt),
                Token::Ge => Some(BinaryOp::Ge),
                _ => None,
            };
            let Some(op) = op else { break };
            let _ = self.cur.span;
            self.bump()?;
            let rhs = self.parse_shift()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = match self.cur.tok {
                Token::Shl => Some(BinaryOp::Shl),
                Token::Shr => Some(BinaryOp::Shr),
                Token::Sshr => Some(BinaryOp::Sshr),
                _ => None,
            };
            let Some(op) = op else { break };
            let _ = self.cur.span;
            self.bump()?;
            let rhs = self.parse_additive()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.cur.tok {
                Token::Plus => Some(BinaryOp::Add),
                Token::Minus => Some(BinaryOp::Sub),
                _ => None,
            };
            let Some(op) = op else { break };
            let _ = self.cur.span;
            self.bump()?;
            let rhs = self.parse_multiplicative()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<SpannedExpr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.cur.tok {
                Token::Star => Some(BinaryOp::Mul),
                Token::Slash => Some(BinaryOp::Div),
                Token::Percent => Some(BinaryOp::Mod),
                _ => None,
            };
            let Some(op) = op else { break };
            let _ = self.cur.span;
            self.bump()?;
            let rhs = self.parse_unary()?;
            let span = Span {
                start: lhs.span.start,
                end: rhs.span.end,
            };
            lhs = SpannedExpr {
                span,
                kind: SpannedExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<SpannedExpr> {
        match self.cur.tok.clone() {
            Token::Bang => {
                let op_span = self.expect_and_bump(Token::Bang)?;
                let e = self.parse_unary()?;
                Ok(SpannedExpr {
                    span: Span {
                        start: op_span.start,
                        end: e.span.end,
                    },
                    kind: SpannedExprKind::Unary {
                        op: UnaryOp::LogicalNot,
                        expr: Box::new(e),
                    },
                })
            }
            Token::Tilde => {
                let op_span = self.expect_and_bump(Token::Tilde)?;
                // Could be ~, ~&, ~|, ~^
                match self.cur.tok.clone() {
                    Token::And => {
                        self.bump()?;
                        let e = self.parse_unary()?;
                        Ok(SpannedExpr {
                            span: Span {
                                start: op_span.start,
                                end: e.span.end,
                            },
                            kind: SpannedExprKind::Unary {
                                op: UnaryOp::ReduceNand,
                                expr: Box::new(e),
                            },
                        })
                    }
                    Token::Or => {
                        self.bump()?;
                        let e = self.parse_unary()?;
                        Ok(SpannedExpr {
                            span: Span {
                                start: op_span.start,
                                end: e.span.end,
                            },
                            kind: SpannedExprKind::Unary {
                                op: UnaryOp::ReduceNor,
                                expr: Box::new(e),
                            },
                        })
                    }
                    Token::Caret => {
                        self.bump()?;
                        let e = self.parse_unary()?;
                        Ok(SpannedExpr {
                            span: Span {
                                start: op_span.start,
                                end: e.span.end,
                            },
                            kind: SpannedExprKind::Unary {
                                op: UnaryOp::ReduceXnor,
                                expr: Box::new(e),
                            },
                        })
                    }
                    _ => {
                        let e = self.parse_unary()?;
                        Ok(SpannedExpr {
                            span: Span {
                                start: op_span.start,
                                end: e.span.end,
                            },
                            kind: SpannedExprKind::Unary {
                                op: UnaryOp::BitNot,
                                expr: Box::new(e),
                            },
                        })
                    }
                }
            }
            Token::Plus => {
                let op_span = self.expect_and_bump(Token::Plus)?;
                let e = self.parse_unary()?;
                Ok(SpannedExpr {
                    span: Span {
                        start: op_span.start,
                        end: e.span.end,
                    },
                    kind: SpannedExprKind::Unary {
                        op: UnaryOp::UnaryPlus,
                        expr: Box::new(e),
                    },
                })
            }
            Token::Minus => {
                let op_span = self.expect_and_bump(Token::Minus)?;
                let e = self.parse_unary()?;
                Ok(SpannedExpr {
                    span: Span {
                        start: op_span.start,
                        end: e.span.end,
                    },
                    kind: SpannedExprKind::Unary {
                        op: UnaryOp::UnaryMinus,
                        expr: Box::new(e),
                    },
                })
            }
            Token::And => {
                let op_span = self.expect_and_bump(Token::And)?;
                let e = self.parse_unary()?;
                Ok(SpannedExpr {
                    span: Span {
                        start: op_span.start,
                        end: e.span.end,
                    },
                    kind: SpannedExprKind::Unary {
                        op: UnaryOp::ReduceAnd,
                        expr: Box::new(e),
                    },
                })
            }
            Token::Or => {
                let op_span = self.expect_and_bump(Token::Or)?;
                let e = self.parse_unary()?;
                Ok(SpannedExpr {
                    span: Span {
                        start: op_span.start,
                        end: e.span.end,
                    },
                    kind: SpannedExprKind::Unary {
                        op: UnaryOp::ReduceOr,
                        expr: Box::new(e),
                    },
                })
            }
            Token::Caret => {
                let op_span = self.expect_and_bump(Token::Caret)?;
                let e = self.parse_unary()?;
                Ok(SpannedExpr {
                    span: Span {
                        start: op_span.start,
                        end: e.span.end,
                    },
                    kind: SpannedExprKind::Unary {
                        op: UnaryOp::ReduceXor,
                        expr: Box::new(e),
                    },
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<SpannedExpr> {
        let mut base = self.parse_primary()?;
        loop {
            match self.cur.tok.clone() {
                Token::Apostrophe => {
                    let _ = self.expect_and_bump(Token::Apostrophe)?;
                    let _ = self.expect_and_bump(Token::LParen)?;
                    let inner = self.parse_ternary()?;
                    let rpar = self.expect_and_bump(Token::RParen)?;
                    let span = Span {
                        start: base.span.start,
                        end: rpar.end,
                    };
                    base = SpannedExpr {
                        span,
                        kind: SpannedExprKind::Cast {
                            width: Box::new(base),
                            expr: Box::new(inner),
                        },
                    };
                }
                Token::LBracket => {
                    let lbr = self.expect_and_bump(Token::LBracket)?;
                    let first = self.parse_ternary()?;
                    if self.cur.tok == Token::Colon {
                        self.bump()?;
                        let lsb = self.parse_ternary()?;
                        let rbr = self.expect_and_bump(Token::RBracket)?;
                        if !spanned_expr_is_constant(&first) || !spanned_expr_is_constant(&lsb) {
                            return Err(Error::Parse(
                                "part-select bounds must be constant expressions".to_string(),
                            ));
                        }
                        let span = Span {
                            start: base.span.start,
                            end: rbr.end,
                        };
                        let _ = lbr;
                        base = SpannedExpr {
                            span,
                            kind: SpannedExprKind::Slice {
                                expr: Box::new(base),
                                msb: Box::new(first),
                                lsb: Box::new(lsb),
                            },
                        };
                    } else if matches!(self.cur.tok, Token::PlusColon | Token::MinusColon) {
                        let upward = self.cur.tok == Token::PlusColon;
                        self.bump()?;
                        let width = self.parse_ternary()?;
                        let rbr = self.expect_and_bump(Token::RBracket)?;
                        if !spanned_expr_is_constant(&width) {
                            return Err(Error::Parse(
                                "indexed part-select width must be a constant expression"
                                    .to_string(),
                            ));
                        }
                        let span = Span {
                            start: base.span.start,
                            end: rbr.end,
                        };
                        let _ = lbr;
                        base = SpannedExpr {
                            span,
                            kind: SpannedExprKind::IndexedSlice {
                                expr: Box::new(base),
                                base: Box::new(first),
                                width: Box::new(width),
                                upward,
                            },
                        };
                    } else {
                        let rbr = self.expect_and_bump(Token::RBracket)?;
                        let span = Span {
                            start: base.span.start,
                            end: rbr.end,
                        };
                        let _ = lbr;
                        base = SpannedExpr {
                            span,
                            kind: SpannedExprKind::Index {
                                expr: Box::new(base),
                                index: Box::new(first),
                            },
                        };
                    }
                }
                Token::LParen => {
                    // Function call on identifier only, for parity with existing parser.
                    let SpannedExprKind::Ident(name) = &base.kind else {
                        break;
                    };
                    let call_start = base.span.start;
                    let name = name.clone();
                    let _ = self.expect_and_bump(Token::LParen)?;
                    let mut args: Vec<SpannedExpr> = Vec::new();
                    if self.cur.tok != Token::RParen {
                        loop {
                            args.push(self.parse_ternary()?);
                            if self.cur.tok == Token::Comma {
                                self.bump()?;
                                continue;
                            }
                            break;
                        }
                    }
                    let rpar = self.expect_and_bump(Token::RParen)?;
                    base = SpannedExpr {
                        span: Span {
                            start: call_start,
                            end: rpar.end,
                        },
                        kind: SpannedExprKind::Call { name, args },
                    };
                }
                _ => break,
            }
        }
        Ok(base)
    }

    fn parse_primary(&mut self) -> Result<SpannedExpr> {
        match self.cur.tok.clone() {
            Token::LParen => {
                let l = self.expect_and_bump(Token::LParen)?;
                let e = self.parse_ternary()?;
                let r = self.expect_and_bump(Token::RParen)?;
                Ok(SpannedExpr {
                    span: Span {
                        start: l.start,
                        end: r.end,
                    },
                    kind: e.kind,
                })
            }
            Token::LBrace => {
                let l = self.expect_and_bump(Token::LBrace)?;
                // Either concat `{a,b,c}` or replicate `{N{expr}}`
                let first = self.parse_ternary()?;
                if self.cur.tok == Token::LBrace {
                    let _ = self.expect_and_bump(Token::LBrace)?;
                    let inner = self.parse_ternary()?;
                    let _ = self.expect_and_bump(Token::RBrace)?;
                    let r = self.expect_and_bump(Token::RBrace)?;
                    if !spanned_expr_is_constant(&first) {
                        return Err(Error::Parse(
                            "replication count must be a constant expression".to_string(),
                        ));
                    }
                    Ok(SpannedExpr {
                        span: Span {
                            start: l.start,
                            end: r.end,
                        },
                        kind: SpannedExprKind::Replicate {
                            count: Box::new(first),
                            expr: Box::new(inner),
                        },
                    })
                } else {
                    let mut parts: Vec<SpannedExpr> = vec![first];
                    while self.cur.tok == Token::Comma {
                        self.bump()?;
                        parts.push(self.parse_ternary()?);
                    }
                    let r = self.expect_and_bump(Token::RBrace)?;
                    if parts
                        .iter()
                        .any(|p| matches!(p.kind, SpannedExprKind::UnsizedNumber(_)))
                    {
                        return Err(Error::Parse(
                            "unsized constant numbers are illegal in concatenations".to_string(),
                        ));
                    }
                    Ok(SpannedExpr {
                        span: Span {
                            start: l.start,
                            end: r.end,
                        },
                        kind: SpannedExprKind::Concat(parts),
                    })
                }
            }
            Token::Ident(name) => {
                let s = self.cur.span;
                self.bump()?;
                Ok(SpannedExpr {
                    span: s,
                    kind: SpannedExprKind::Ident(name),
                })
            }
            Token::Number(n) => {
                let s = self.cur.span;
                self.bump()?;
                Ok(SpannedExpr {
                    span: s,
                    kind: parse_number_to_spanned_kind(&n)?,
                })
            }
            _ => Err(Error::Parse(format!(
                "unexpected token in primary: {:?}",
                self.cur.tok
            ))),
        }
    }
}

fn parse_number_to_spanned_kind(n: &str) -> Result<SpannedExprKind> {
    // Delegate to existing parser for literal semantics.
    let e: Expr = crate::parser::parse_expr(n)?;
    match e {
        Expr::Literal(v) => Ok(SpannedExprKind::Literal(v)),
        Expr::UnsizedNumber(v) => Ok(SpannedExprKind::UnsizedNumber(v)),
        Expr::UnbasedUnsized(b) => Ok(SpannedExprKind::UnbasedUnsized(b)),
        _ => Err(Error::Parse(format!("unsupported numeric token `{n}`"))),
    }
}
