// SPDX-License-Identifier: Apache-2.0

use crate::value::LogicBit;
use crate::value::Value4;

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
