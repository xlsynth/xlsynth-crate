// SPDX-License-Identifier: Apache-2.0

use crate::ast::Expr as VExpr;
use crate::value::Value4;
use std::collections::BTreeMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PortDir {
    Input,
    Output,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PortTy {
    Wire,
    Logic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortDecl {
    pub dir: PortDir,
    pub ty: PortTy,
    pub signed: bool,
    pub width: u32,
    pub packed_dims: Vec<u32>,
    pub unpacked_dims: Vec<u32>,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedModule {
    pub name: String,
    pub params: BTreeMap<String, Value4>,
    pub ports: Vec<PortDecl>,
    pub header_span: Span,
    pub endmodule_span: Span,
    pub items: Vec<ModuleItem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDecl {
    pub name: String,
    pub ret_width: u32,
    pub ret_signed: bool,
    pub args: Vec<Decl>,
    pub locals: Vec<Decl>,
    pub body: FunctionBody,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionBody {
    UniqueCasez {
        casez_span: Span,
        selector: Span,
        endcase_span: Span,
        arms: Vec<CasezArm>,
    },
    Assign {
        value: Span,
    },
    Procedure {
        assigns: Vec<FunctionAssign>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionAssign {
    pub lhs: String,
    pub value: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CasezPattern {
    pub width: u32,
    /// Verbatim bits string (MSB-first), may include '?'.
    pub bits_msb: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CasezArm {
    pub pat: Option<CasezPattern>, // None => default
    pub pat_span: Option<Span>,
    pub arm_span: Span,
    pub value: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decl {
    pub name: String,
    pub signed: bool,
    pub width: u32,
    pub packed_dims: Vec<u32>,
    pub unpacked_dims: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlwaysFf {
    pub clk_name: String,
    pub body: Stmt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleItem {
    Decl {
        decl: Decl,
        span: Span,
    },
    Assign {
        lhs: Lhs,
        rhs: Span,
        rhs_text: Option<String>,
        span: Span,
    },
    Function {
        func: FunctionDecl,
        span: Span,
        body_span: Span,
        begin_span: Span,
        end_span: Span,
    },
    AlwaysFf {
        always_ff: AlwaysFf,
        span: Span,
    },
    GenerateFor {
        genvar: String,
        start: VExpr,
        limit: VExpr,
        body: Vec<ModuleItem>,
        span: Span,
    },
    GenerateIf {
        branches: Vec<GenerateBranch>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateBranch {
    pub cond: Option<VExpr>,
    pub body: Vec<ModuleItem>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Begin(Vec<Stmt>),
    If {
        cond: VExpr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    NbaAssign {
        lhs: Lhs,
        rhs: VExpr,
    },
    Display {
        fmt: String,
        args: Vec<VExpr>,
    },
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lhs {
    Ident(String),
    Index {
        base: String,
        index: VExpr,
    },
    PackedIndex {
        base: String,
        indices: Vec<VExpr>,
    },
    Slice {
        base: String,
        msb: VExpr,
        lsb: VExpr,
    },
}
