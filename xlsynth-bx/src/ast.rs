// SPDX-License-Identifier: Apache-2.0

//! Source-level representation of a BX module, before name and type checking.

use std::fmt;

// Generate both directions of the spelling map from the same operator list.
macro_rules! operator_enum {
    ($(#[$attribute:meta])* $name:ident { $($variant:ident => $spelling:literal),+ $(,)? }) => {
        $(#[$attribute])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
        pub enum $name { $($variant),+ }

        impl $name {
            pub(crate) fn from_spelling(text: &str) -> Option<Self> {
                match text {
                    $($spelling => Some(Self::$variant),)+
                    _ => None,
                }
            }

            pub fn spelling(self) -> &'static str {
                match self { $(Self::$variant => $spelling),+ }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.spelling())
            }
        }
    }
}

operator_enum! {
    /// Unary operations on BX bit values.
    UnaryOp { Not => "!", Neg => "-" }
}

operator_enum! {
    /// Binary operations in a BX value expression.
    BinaryOp {
        LogicalOr => "||", LogicalAnd => "&&",
        Eq => "==", Ne => "!=", Lt => "<", Le => "<=", Gt => ">", Ge => ">=",
        Or => "|", Xor => "^", And => "&", Shl => "<<", Shr => ">>",
        Add => "+", Sub => "-", Concat => "++", Mul => "*", Div => "/", Mod => "%",
    }
}

impl BinaryOp {
    /// DSLX operator precedence, with larger numbers binding more tightly.
    pub(crate) fn precedence(self) -> u8 {
        match self {
            Self::LogicalOr => 1,
            Self::LogicalAnd => 2,
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge => 3,
            Self::Or => 4,
            Self::Xor => 5,
            Self::And => 6,
            Self::Shl | Self::Shr => 7,
            Self::Add | Self::Sub | Self::Concat => 8,
            Self::Mul | Self::Div | Self::Mod => 9,
        }
    }
}

/// Concrete value types shared by block ports, registers, and function calls.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Type {
    Bits { signed: bool, width: usize },
    Bool,
    Tuple(Vec<Type>),
    Array { element: Box<Type>, size: usize },
    Named(String),
}

/// A typed name in a parameter list or struct declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

/// A named aggregate, with fields in declaration order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub public: bool,
    pub fields: Vec<Param>,
}

/// The source signature and optional package symbol for an external function.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternFn {
    /// Source name, spelled either `function` or `module::function`.
    pub name: String,
    pub params: Vec<Param>,
    pub result: Type,
    pub symbol: Option<String>,
}

/// The BX signature of a clocked block supplied by the linked IR package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternBlock {
    /// Exact block name in the linked package.
    pub name: String,
    pub clock: String,
    pub reset: Option<String>,
    pub params: Vec<Param>,
    pub result: Type,
}

/// A value computed from this cycle's ports and register contents.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Expr {
    Name(String),
    Bool(bool),
    Literal {
        ty: Option<Type>,
        text: String,
    },
    Tuple(Vec<Expr>),
    Array(Vec<Expr>),
    Struct {
        name: String,
        fields: Vec<(String, Expr)>,
    },
    Field {
        value: Box<Expr>,
        name: String,
    },
    Index {
        value: Box<Expr>,
        index: Box<Expr>,
    },
    Call {
        /// A function name (including qualified imported or extern functions),
        /// or a block name when the call initializes a let binding.
        name: String,
        args: Vec<Expr>,
    },
    Unary {
        op: UnaryOp,
        value: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Cast {
        value: Box<Expr>,
        ty: Type,
    },
    If {
        condition: Box<Expr>,
        then_value: Box<Expr>,
        else_value: Box<Expr>,
    },
}

/// Structural declarations, value bindings, and register or child port drivers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Stmt {
    Reg {
        name: String,
        ty: Type,
        reset: Option<Expr>,
    },
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
    },
    Write {
        name: String,
        value: Expr,
    },
    If {
        condition: Expr,
        body: Vec<Stmt>,
    },
    Inst {
        name: String,
        block: String,
    },
    Connect {
        instance: String,
        port: String,
        value: Expr,
    },
}

/// A block definition, including its implicit timing domain and output
/// expression.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Block {
    pub name: String,
    pub public: bool,
    pub clock: String,
    pub reset: Option<String>,
    pub params: Vec<Param>,
    pub result: Type,
    pub statements: Vec<Stmt>,
    pub output: Expr,
}

/// A DSLX module imported under a local qualifier for function calls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Import {
    /// The full DSLX import subject, with components separated by dots.
    pub subject: String,
    /// The explicit local name following `as`, when present.
    pub alias: Option<String>,
}

impl Import {
    /// The qualifier in `qualifier::function(...)`. DSLX defaults this to the
    /// final component of the import subject.
    pub fn binding(&self) -> &str {
        self.alias.as_deref().unwrap_or_else(|| {
            self.subject
                .rsplit_once('.')
                .map_or(self.subject.as_str(), |(_, last)| last)
        })
    }
}

/// All declarations in one source file, retaining order within each kind.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Module {
    pub imports: Vec<Import>,
    pub structs: Vec<StructDef>,
    pub externs: Vec<ExternFn>,
    pub extern_blocks: Vec<ExternBlock>,
    pub blocks: Vec<Block>,
}
