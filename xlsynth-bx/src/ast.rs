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
    pub(crate) fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge
        )
    }

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

/// Value types shared by block ports, registers, and function calls.
/// A named type can refer to a declaration or an enclosing block type
/// parameter.
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

/// A local binding in a pure function, evaluated before subsequent bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LetBinding {
    pub pattern: BindingPattern,
    pub ty: Option<Type>,
    pub value: Expr,
}

/// A DSLX let pattern. A rest marker consumes any remaining tuple members.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BindingPattern {
    Name(String),
    Tuple(Vec<BindingPattern>),
    Rest,
}

impl BindingPattern {
    /// Visit the names introduced in source order and return the number of
    /// pattern nodes traversed, including wildcards and rest markers.
    pub(crate) fn visit_names(&self, visit: &mut impl FnMut(&str)) -> usize {
        let mut nodes = 1;
        match self {
            Self::Name(name) if name != "_" => visit(name),
            Self::Tuple(fields) => {
                for field in fields {
                    nodes += field.visit_names(visit);
                }
            }
            Self::Name(_) | Self::Rest => { /* Wildcards and rest markers bind no value. */ }
        }
        nodes
    }
}

/// Alternatives in one match arm share a result expression.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchArm {
    pub patterns: Vec<MatchPattern>,
    pub value: Expr,
}

/// A wildcard, scalar literal, or range between two scalar literals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MatchPattern {
    Wildcard,
    Literal(Expr),
    Range {
        start: Expr,
        end: Expr,
        inclusive: bool,
    },
}

impl MatchPattern {
    /// Visit literal expressions in the order they occur in the pattern.
    pub(crate) fn literals(&self) -> impl Iterator<Item = &Expr> {
        match self {
            Self::Wildcard => [None, None],
            Self::Literal(value) => [Some(value), None],
            Self::Range { start, end, .. } => [Some(start), Some(end)],
        }
        .into_iter()
        .flatten()
    }

    /// Visit the same expressions when capturing their type annotations.
    pub(crate) fn literals_mut(&mut self) -> impl Iterator<Item = &mut Expr> {
        match self {
            Self::Wildcard => [None, None],
            Self::Literal(value) => [Some(value), None],
            Self::Range { start, end, .. } => [Some(start), Some(end)],
        }
        .into_iter()
        .flatten()
    }
}

/// A combinational function defined in this source module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Function {
    pub name: String,
    pub public: bool,
    pub params: Vec<Param>,
    pub result: Type,
    pub body: Expr,
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
    /// A lexical value scope, evaluated in source order.
    Scope {
        bindings: Vec<LetBinding>,
        result: Box<Expr>,
    },
    Name(String),
    Bool(bool),
    Literal {
        ty: Option<Type>,
        /// Numeric text, or the character or boolean spelling when a generic
        /// annotation defers checking that literal's required width.
        text: String,
    },
    /// An all-zero value of the specified type, including aggregate types.
    Zero(Type),
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
    /// A packed slice with a bit offset and a statically declared result type.
    WidthSlice {
        value: Box<Expr>,
        start: Box<Expr>,
        width: WidthSliceType,
    },
    Call {
        /// A function name (including qualified imported or extern functions),
        /// or a block name when the call initializes a let binding.
        name: String,
        args: Vec<Expr>,
    },
    Map(Box<MapExpr>),
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
    Match {
        subject: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    UnrollFor(Box<UnrollFor>),
    For(Box<ForLoop>),
}

/// An array mapped by an inline unary lambda. The checker verifies that the
/// body is closed over its parameter and visible module declarations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MapExpr {
    /// The callee name, retained so lexical scope preparation preserves
    /// shadowing.
    pub target: String,
    pub iterable: ForIterable,
    pub parameter: String,
    pub parameter_type: Option<Type>,
    pub result_type: Option<Type>,
    pub body: Expr,
}

/// A counted fold, with an iteration element and a loop-carried accumulator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForLoop {
    pub element: BindingPattern,
    pub accumulator: BindingPattern,
    pub annotation: Option<(Type, Type)>,
    pub iterable: ForIterable,
    pub body: Expr,
    pub seed: Expr,
}

/// A counted loop or map visits a statically bounded integer range or an array
/// value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForIterable {
    Range {
        start: Expr,
        limit: Expr,
        inclusive: bool,
    },
    Array(Expr),
}

/// A fold over a compile-time half-open range. Each iteration's local lets
/// and result expression can use the index and accumulator bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnrollFor {
    pub index: String,
    pub accumulator: String,
    pub annotation: Option<(Type, Type)>,
    pub start: Expr,
    pub limit: Expr,
    pub body: Vec<Stmt>,
    pub result: Expr,
    pub seed: Expr,
}

/// The result type in `value[start+:type]`, before resolving its constant
/// parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WidthSliceType {
    Concrete(Type),
    Bits {
        signed: Box<Expr>,
        bit_count: Box<Expr>,
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
    /// Type parameter names in declaration order.
    pub type_params: Vec<String>,
    pub clock: String,
    pub reset: Option<String>,
    pub params: Vec<Param>,
    pub result: Type,
    pub statements: Vec<Stmt>,
    pub output: Expr,
}

/// One leaf of a DSLX `use` declaration. Its final component is bound in BX.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UsePath {
    /// All path components in order, including the prefixes of enclosing
    /// groups.
    pub components: Vec<String>,
}

impl UsePath {
    pub fn binding(&self) -> Option<&str> {
        self.components.last().map(String::as_str)
    }
}

/// All declarations in one source file, retaining order within each kind.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Module {
    /// Bound names in declaration order, including each leaf of a use tree.
    pub declaration_order: Vec<String>,
    pub uses: Vec<UsePath>,
    pub structs: Vec<StructDef>,
    pub externs: Vec<ExternFn>,
    pub functions: Vec<Function>,
    pub extern_blocks: Vec<ExternBlock>,
    pub blocks: Vec<Block>,
}
