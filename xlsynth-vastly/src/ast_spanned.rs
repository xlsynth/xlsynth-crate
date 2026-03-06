// SPDX-License-Identifier: Apache-2.0

use crate::ast::BinaryOp;
use crate::ast::UnaryOp;
use crate::lexer::Token;
use crate::sv_ast::Span;
use crate::value::LogicBit;
use crate::value::Value4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpannedExpr {
    pub span: Span, // byte offsets in the expression slice
    pub kind: SpannedExprKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpannedExprKind {
    Ident(String),
    Literal(Value4),
    UnsizedNumber(Value4),
    UnbasedUnsized(LogicBit),
    Call {
        name: String,
        args: Vec<SpannedExpr>,
    },
    Concat(Vec<SpannedExpr>),
    Replicate {
        count: Box<SpannedExpr>,
        expr: Box<SpannedExpr>,
    },
    Index {
        expr: Box<SpannedExpr>,
        index: Box<SpannedExpr>,
    },
    Slice {
        expr: Box<SpannedExpr>,
        msb: Box<SpannedExpr>,
        lsb: Box<SpannedExpr>,
    },
    IndexedSlice {
        expr: Box<SpannedExpr>,
        base: Box<SpannedExpr>,
        width: Box<SpannedExpr>,
        upward: bool,
    },
    Unary {
        op: UnaryOp,
        expr: Box<SpannedExpr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<SpannedExpr>,
        rhs: Box<SpannedExpr>,
    },
    Ternary {
        cond: Box<SpannedExpr>,
        t: Box<SpannedExpr>,
        f: Box<SpannedExpr>,
    },
}

impl SpannedExpr {
    pub fn shift_spans(&mut self, delta: usize) {
        self.span.start += delta;
        self.span.end += delta;
        match &mut self.kind {
            SpannedExprKind::Ident(_) => {}
            SpannedExprKind::Literal(_) => {}
            SpannedExprKind::UnsizedNumber(_) => {}
            SpannedExprKind::UnbasedUnsized(_) => {}
            SpannedExprKind::Call { args, .. } => {
                for a in args {
                    a.shift_spans(delta);
                }
            }
            SpannedExprKind::Concat(ps) => {
                for p in ps {
                    p.shift_spans(delta);
                }
            }
            SpannedExprKind::Replicate { count, expr } => {
                count.shift_spans(delta);
                expr.shift_spans(delta);
            }
            SpannedExprKind::Index { expr, index } => {
                expr.shift_spans(delta);
                index.shift_spans(delta);
            }
            SpannedExprKind::Slice { expr, msb, lsb } => {
                expr.shift_spans(delta);
                msb.shift_spans(delta);
                lsb.shift_spans(delta);
            }
            SpannedExprKind::IndexedSlice {
                expr, base, width, ..
            } => {
                expr.shift_spans(delta);
                base.shift_spans(delta);
                width.shift_spans(delta);
            }
            SpannedExprKind::Unary { expr, .. } => {
                expr.shift_spans(delta);
            }
            SpannedExprKind::Binary { lhs, rhs, .. } => {
                lhs.shift_spans(delta);
                rhs.shift_spans(delta);
            }
            SpannedExprKind::Ternary { cond, t, f } => {
                cond.shift_spans(delta);
                t.shift_spans(delta);
                f.shift_spans(delta);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpannedTok {
    pub tok: Token,
    pub span: Span,
}
