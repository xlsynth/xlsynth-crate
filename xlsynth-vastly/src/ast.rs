// SPDX-License-Identifier: Apache-2.0

use crate::value::LogicBit;
use crate::value::Value4;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Ident(String),
    Literal(Value4),
    /// Unsized numeric literal such as `123` or `'hff`.
    ///
    /// This remains distinct from sized literals so parser-enforced context
    /// rules, such as concatenation legality, can follow the standard.
    UnsizedNumber(Value4),
    /// SystemVerilog unbased unsized literal: `'0`, `'1`, `'x`, `'z`.
    ///
    /// This is context-sized; our combo evaluator supplies the expected width
    /// (LHS width).
    UnbasedUnsized(LogicBit),
    /// Function call like `foo(a, b, c)`.
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Concat(Vec<Expr>),
    Replicate {
        count: Box<Expr>,
        expr: Box<Expr>,
    },
    Cast {
        width: Box<Expr>,
        expr: Box<Expr>,
    },
    Index {
        expr: Box<Expr>,
        index: Box<Expr>,
    },
    Slice {
        expr: Box<Expr>,
        msb: Box<Expr>,
        lsb: Box<Expr>,
    },
    IndexedSlice {
        expr: Box<Expr>,
        base: Box<Expr>,
        width: Box<Expr>,
        upward: bool,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Ternary {
        cond: Box<Expr>,
        t: Box<Expr>,
        f: Box<Expr>,
    },
}

impl Expr {
    /// Collects identifier reads and user-defined helper calls from this
    /// expression.
    pub(crate) fn collect_refs(&self, idents: &mut BTreeSet<String>, calls: &mut BTreeSet<String>) {
        match self {
            Expr::Ident(s) => {
                idents.insert(s.clone());
            }
            Expr::Literal(_) | Expr::UnsizedNumber(_) | Expr::UnbasedUnsized(_) => {
                // Constants have no dependencies.
            }
            Expr::Call { name, args } => {
                if !name.starts_with('$') {
                    calls.insert(name.clone());
                }
                for a in args {
                    a.collect_refs(idents, calls);
                }
            }
            Expr::Concat(ps) => {
                for p in ps {
                    p.collect_refs(idents, calls);
                }
            }
            Expr::Replicate { count, expr } => {
                count.collect_refs(idents, calls);
                expr.collect_refs(idents, calls);
            }
            Expr::Cast { width, expr } => {
                width.collect_refs(idents, calls);
                expr.collect_refs(idents, calls);
            }
            Expr::Index { expr, index } => {
                expr.collect_refs(idents, calls);
                index.collect_refs(idents, calls);
            }
            Expr::Slice { expr, msb, lsb } => {
                expr.collect_refs(idents, calls);
                msb.collect_refs(idents, calls);
                lsb.collect_refs(idents, calls);
            }
            Expr::IndexedSlice {
                expr, base, width, ..
            } => {
                expr.collect_refs(idents, calls);
                base.collect_refs(idents, calls);
                width.collect_refs(idents, calls);
            }
            Expr::Unary { expr, .. } => {
                expr.collect_refs(idents, calls);
            }
            Expr::Binary { lhs, rhs, .. } => {
                lhs.collect_refs(idents, calls);
                rhs.collect_refs(idents, calls);
            }
            Expr::Ternary { cond, t, f } => {
                cond.collect_refs(idents, calls);
                t.collect_refs(idents, calls);
                f.collect_refs(idents, calls);
            }
        }
    }

    /// Visits every expression node so elaboration can rewrite names in place.
    pub(crate) fn visit_mut(&mut self, visit: &mut impl FnMut(&mut Self)) {
        visit(self);
        match self {
            Expr::Ident(_)
            | Expr::Literal(_)
            | Expr::UnsizedNumber(_)
            | Expr::UnbasedUnsized(_) => {
                // Leaf expressions have no children.
            }
            Expr::Call { args, .. } | Expr::Concat(args) => {
                for arg in args {
                    arg.visit_mut(visit);
                }
            }
            Expr::Replicate { count, expr } => {
                count.visit_mut(visit);
                expr.visit_mut(visit);
            }
            Expr::Cast { width, expr } => {
                width.visit_mut(visit);
                expr.visit_mut(visit);
            }
            Expr::Index { expr, index } => {
                expr.visit_mut(visit);
                index.visit_mut(visit);
            }
            Expr::Slice { expr, msb, lsb } => {
                expr.visit_mut(visit);
                msb.visit_mut(visit);
                lsb.visit_mut(visit);
            }
            Expr::IndexedSlice {
                expr, base, width, ..
            } => {
                expr.visit_mut(visit);
                base.visit_mut(visit);
                width.visit_mut(visit);
            }
            Expr::Binary { lhs, rhs, .. } => {
                lhs.visit_mut(visit);
                rhs.visit_mut(visit);
            }
            Expr::Ternary { cond, t, f } => {
                cond.visit_mut(visit);
                t.visit_mut(visit);
                f.visit_mut(visit);
            }
            Expr::Unary { expr, .. } => expr.visit_mut(visit),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    LogicalNot, // !
    BitNot,     // ~
    UnaryPlus,  // +
    UnaryMinus, // -
    ReduceAnd,  // &
    ReduceNand, // ~&
    ReduceOr,   // |
    ReduceNor,  // ~|
    ReduceXor,  // ^
    ReduceXnor, // ^~ or ~^
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add, // +
    Sub, // -
    Mul, // *
    Div, // /
    Mod, // %

    // Shifts
    Shl,  // <<
    Shr,  // >>
    Sshr, // >>>

    // Bitwise
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^

    LogicalAnd, // &&
    LogicalOr,  // ||

    // Relational
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=

    Eq,      // ==
    Neq,     // !=
    CaseEq,  // ===
    CaseNeq, // !==
}
