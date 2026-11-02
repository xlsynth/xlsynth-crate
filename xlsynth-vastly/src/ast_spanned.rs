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
    Cast {
        width: Box<SpannedExpr>,
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
    /// Visits every expression node so elaboration can rewrite names in place.
    pub(crate) fn visit_mut(&mut self, visit: &mut impl FnMut(&mut Self)) {
        visit(self);
        match &mut self.kind {
            SpannedExprKind::Ident(_)
            | SpannedExprKind::Literal(_)
            | SpannedExprKind::UnsizedNumber(_)
            | SpannedExprKind::UnbasedUnsized(_) => {
                // Leaf expressions have no children.
            }
            SpannedExprKind::Call { args, .. } | SpannedExprKind::Concat(args) => {
                for arg in args {
                    arg.visit_mut(visit);
                }
            }
            SpannedExprKind::Replicate { count, expr } => {
                count.visit_mut(visit);
                expr.visit_mut(visit);
            }
            SpannedExprKind::Cast { width, expr } => {
                width.visit_mut(visit);
                expr.visit_mut(visit);
            }
            SpannedExprKind::Index { expr, index } => {
                expr.visit_mut(visit);
                index.visit_mut(visit);
            }
            SpannedExprKind::Slice { expr, msb, lsb } => {
                expr.visit_mut(visit);
                msb.visit_mut(visit);
                lsb.visit_mut(visit);
            }
            SpannedExprKind::IndexedSlice {
                expr, base, width, ..
            } => {
                expr.visit_mut(visit);
                base.visit_mut(visit);
                width.visit_mut(visit);
            }
            SpannedExprKind::Binary { lhs, rhs, .. } => {
                lhs.visit_mut(visit);
                rhs.visit_mut(visit);
            }
            SpannedExprKind::Ternary { cond, t, f } => {
                cond.visit_mut(visit);
                t.visit_mut(visit);
                f.visit_mut(visit);
            }
            SpannedExprKind::Unary { expr, .. } => expr.visit_mut(visit),
        }
    }

    pub fn shift_spans(&mut self, delta: usize) {
        self.visit_mut(&mut |node| {
            node.span.start += delta;
            node.span.end += delta;
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpannedTok {
    pub tok: Token,
    pub span: Span,
}
