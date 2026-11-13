// SPDX-License-Identifier: Apache-2.0

//! Source-level representation of a BX module, before name and type checking.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::Arc,
};

use crate::BxWarningKind;

mod source_nodes;
pub(crate) use source_nodes::SourceNode;

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
/// A named type can refer to a declaration or an enclosing function or block
/// type parameter.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Type {
    Bits {
        signed: bool,
        width: usize,
    },
    /// Signedness and width evaluated from constants and any bound function
    /// value parametrics.
    ParametricBits {
        signed: Box<Expr>,
        width: Box<Expr>,
    },
    Bool,
    Tuple(Vec<Type>),
    Array {
        element: Box<Type>,
        size: usize,
    },
    /// Array dimension evaluated from constants and any bound function value
    /// parametrics.
    ParametricArray {
        element: Box<Type>,
        size: Box<Expr>,
    },
    Named(String),
    /// A resolved numeric enum with its nominal identity and IR representation.
    Enum {
        name: String,
        signed: bool,
        width: usize,
    },
    /// A struct application with concrete arguments, before nominal resolution.
    Applied {
        name: String,
        args: Vec<u32>,
    },
    /// A struct application with arguments evaluated at compile time.
    ParametricStruct {
        name: String,
        args: Vec<Expr>,
    },
    /// A native struct application with type arguments.
    TypeApplied {
        name: String,
        args: Vec<Type>,
    },
}

impl Type {
    /// Construct an unsigned bit-vector type of the given width.
    pub(crate) fn unsigned(width: usize) -> Self {
        Self::Bits {
            signed: false,
            width,
        }
    }

    /// Construct an array type from its element type and length.
    pub(crate) fn array(element: Type, size: usize) -> Self {
        Self::Array {
            element: Box::new(element),
            size,
        }
    }
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
    /// Type parameters in declaration order.
    pub type_params: Vec<String>,
    /// Compile-time values in declaration order.
    pub value_parametrics: Vec<ParametricBinding>,
    pub fields: Vec<Param>,
}

impl StructDef {
    pub fn is_parametric(&self) -> bool {
        !self.type_params.is_empty() || !self.value_parametrics.is_empty()
    }
}

/// One named value in a numeric enum, before constant evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnumMember {
    pub name: String,
    pub value: Expr,
}

/// A module-level name for a concrete value type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeAlias {
    pub name: String,
    pub public: bool,
    pub ty: Type,
}

/// A nominal bits-backed type with ordered member initializers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnumDef {
    pub name: String,
    pub public: bool,
    pub underlying: Option<Type>,
    pub members: Vec<EnumMember>,
}

/// A module-level value evaluated at compile time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstDef {
    pub name: String,
    pub public: bool,
    pub ty: Option<Type>,
    pub value: Expr,
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

/// A let initializer and its pattern, evaluated before introducing the bound
/// names.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LetBinding {
    pub pattern: BindingPattern,
    pub ty: Option<Type>,
    pub value: Expr,
}

/// A declaration requiring a Boolean expression to evaluate to true at compile
/// time.
#[derive(Clone, Debug, Eq)]
pub struct ConstAssert {
    pub predicate: Expr,
    pub position: SourcePosition,
}

// Diagnostic locations do not affect equality of expressions in type
// dimensions.
impl PartialEq for ConstAssert {
    fn eq(&self, other: &Self) -> bool {
        self.predicate == other.predicate
    }
}

impl Hash for ConstAssert {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.predicate.hash(state);
    }
}

/// An ordered declaration in a pure value scope.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ScopeStatement {
    Let(LetBinding),
    ConstAssert(ConstAssert),
    /// A value expression followed by a semicolon. DSLX requires its value
    /// when evaluating the enclosing scope as a constant expression.
    Expression(Expr),
}

/// A DSLX let pattern. A rest marker consumes any remaining tuple members.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum BindingPattern {
    Name(String),
    Tuple(Vec<BindingPattern>),
    Rest,
}

impl BindingPattern {
    /// Visit the names introduced in source order and return the number of
    /// pattern nodes traversed, including wildcards and rest markers.
    pub(crate) fn visit_names(&self, visit: &mut impl FnMut(&str)) -> usize {
        let mut nodes = 0;
        let mut pending = Vec::new();
        let mut next = Some(self);
        while let Some(pattern) = next {
            nodes += 1;
            match pattern {
                Self::Name(name) if name != "_" => visit(name),
                Self::Tuple(fields) => pending.extend(fields.iter().rev()),
                Self::Name(_) | Self::Rest => { /* Wildcards and rest markers bind no value. */ }
            }
            next = pending.pop();
        }
        nodes
    }
}

/// Alternatives in one match arm share a result expression.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MatchArm {
    pub patterns: Vec<MatchPattern>,
    pub value: Expr,
}

/// A tuple of patterns or a leaf matching a wildcard, literal, value
/// parametric, module constant, enum member, or literal range.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum MatchPattern {
    Wildcard,
    Literal(Expr),
    Range {
        start: Expr,
        end: Expr,
        inclusive: bool,
    },
    Tuple(Vec<MatchPattern>),
}

impl MatchPattern {
    /// Visit each pattern node in source order, starting at depth one.
    pub(crate) fn nodes(&self) -> impl Iterator<Item = (usize, &Self)> {
        let mut pending: Vec<std::slice::Iter<'_, Self>> = Vec::new();
        let mut next = Some(self);
        std::iter::from_fn(move || {
            loop {
                if let Some(pattern) = next.take() {
                    let depth = pending.len() + 1;
                    if let Self::Tuple(fields) = pattern {
                        pending.push(fields.iter());
                    }
                    return Some((depth, pattern));
                }
                next = pending.last_mut()?.next();
                if next.is_none() {
                    pending.pop();
                }
            }
        })
    }

    /// Visit literal expressions in the order they occur in the pattern.
    pub(crate) fn literals(&self) -> impl Iterator<Item = &Expr> {
        self.nodes()
            .flat_map(|(_, pattern)| match pattern {
                Self::Wildcard | Self::Tuple(_) => [None, None],
                Self::Literal(value) => [Some(value), None],
                Self::Range { start, end, .. } => [Some(start), Some(end)],
            })
            .flatten()
    }

    /// Visit the same expressions when capturing their type annotations.
    pub(crate) fn literals_mut(&mut self) -> impl Iterator<Item = &mut Expr> {
        let mut pending: Vec<std::slice::IterMut<'_, Self>> = Vec::new();
        let mut next = Some(self);
        std::iter::from_fn(move || {
            loop {
                if let Some(pattern) = next.take() {
                    match pattern {
                        Self::Wildcard => { /* No literal expressions to visit. */ }
                        Self::Literal(value) => return Some([Some(value), None]),
                        Self::Range { start, end, .. } => return Some([Some(start), Some(end)]),
                        Self::Tuple(fields) => pending.push(fields.iter_mut()),
                    }
                }
                next = pending.last_mut()?.next();
                if next.is_none() {
                    pending.pop();
                }
            }
        })
        .flatten()
        .flatten()
    }
}

/// A combinational function defined in this source module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Function {
    pub name: String,
    pub public: bool,
    /// Type parameters inferred from data inputs and callback results.
    pub type_params: Vec<String>,
    /// Named, compile-time function parameters in declaration order.
    pub callback_params: Vec<CallbackParam>,
    /// Compile-time values in declaration order.
    pub value_parametrics: Vec<ParametricBinding>,
    pub params: Vec<Param>,
    pub result: Type,
    pub body: Expr,
}

/// A value parametric, optionally derived from earlier compile-time
/// bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParametricBinding {
    pub name: String,
    pub ty: Type,
    pub default: Option<Expr>,
    /// Position in the declaration's combined list of value and callback
    /// parameters.
    pub index: usize,
}

/// A value or callback formal that accepts an explicit call argument.
#[derive(Clone, Copy, Debug)]
pub enum FunctionParametricRef<'a> {
    Value(&'a ParametricBinding),
    Callback(&'a CallbackParam),
}

impl Function {
    /// Whether a call must bind type or value parameters to specialize this
    /// function.
    pub fn has_inferred_parametrics(&self) -> bool {
        !self.type_params.is_empty() || !self.value_parametrics.is_empty()
    }

    /// Visit interleaved value and callback formals in call-site argument
    /// order.
    pub fn parametric_bindings(&self) -> impl Iterator<Item = FunctionParametricRef<'_>> {
        explicit_parametric_bindings(&self.value_parametrics, &self.callback_params)
    }
}

/// Merge the value and callback formals in explicit call argument order.
fn explicit_parametric_bindings<'a>(
    values: &'a [ParametricBinding],
    callbacks: &'a [CallbackParam],
) -> impl Iterator<Item = FunctionParametricRef<'a>> {
    let count = values.len() + callbacks.len();
    let mut values = values.iter().peekable();
    let mut callbacks = callbacks.iter();
    (0..count).filter_map(move |index| {
        if let Some(value) = values.next_if(|p| p.index == index) {
            Some(FunctionParametricRef::Value(value))
        } else {
            callbacks.next().map(FunctionParametricRef::Callback)
        }
    })
}

/// A callable that a function or block template binds to a named pure
/// function. Its signature can refer to the template's type parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackParam {
    pub name: String,
    pub params: Vec<Type>,
    pub result: Type,
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

/// The base expression in a struct update and the source location of its `..`.
#[derive(Clone, Debug, Eq)]
pub struct StructSplat {
    pub value: Expr,
    pub position: Option<SourcePosition>,
}

// Source locations are diagnostic metadata. Expressions in type dimensions
// must compare the same when written in different places.
impl PartialEq for StructSplat {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Hash for StructSplat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

/// A value computed from this cycle's ports and register contents.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Expr {
    /// A lexical value scope, evaluated in source order.
    Scope {
        statements: Vec<ScopeStatement>,
        result: Box<Expr>,
    },
    Name(String),
    Bool(bool),
    /// A DSLX string literal, with its UTF-8 and escaped bytes decoded.
    ByteString(Vec<u8>),
    Literal {
        ty: Option<Type>,
        /// Numeric text, or the character or boolean spelling when a generic
        /// annotation defers checking that literal's required width.
        text: String,
    },
    /// An all-zero value of the specified type, including aggregate types.
    Zero(Type),
    /// A DSLX builtin invocation with explicit type arguments.
    TypeBuiltin {
        name: String,
        types: Vec<Type>,
        args: Vec<Expr>,
    },
    Tuple(Vec<Expr>),
    Array(Vec<Expr>),
    /// An array with an explicit type or a trailing ellipsis that repeats
    /// its final member up to the declared size.
    ArrayLiteral {
        annotation: Option<Type>,
        members: Vec<Expr>,
        has_ellipsis: bool,
    },
    Struct {
        name: String,
        fields: Vec<(String, Expr)>,
        /// A final `..base` expression supplying omitted fields.
        base: Option<Box<StructSplat>>,
    },
    /// A struct literal with explicit concrete DSLX value arguments.
    AppliedStruct {
        name: String,
        args: Vec<u32>,
        fields: Vec<(String, Expr)>,
        base: Option<Box<StructSplat>>,
    },
    /// A struct literal with arguments evaluated at compile time.
    ParametricStruct {
        name: String,
        args: Vec<Expr>,
        fields: Vec<(String, Expr)>,
        base: Option<Box<StructSplat>>,
    },
    /// A native struct literal with explicit type arguments.
    TypeAppliedStruct {
        name: String,
        args: Vec<Type>,
        fields: Vec<(String, Expr)>,
        base: Option<Box<StructSplat>>,
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
    /// A call before a later `map` import. The imported callable takes ordinary
    /// block arguments. A later enum leaves the builtin visible, with its own
    /// operand grammar. Absent builtin arguments mean that grammar rejected
    /// the source.
    MapBeforeUse {
        name: String,
        args: Vec<Expr>,
        builtin_args: Option<Vec<Expr>>,
    },
    /// Explicit compile-time arguments to a function or child block. A callback
    /// argument is represented by the name of its function.
    ParametricCall {
        name: String,
        parametrics: Vec<Expr>,
        args: Vec<Expr>,
    },
    Map(Box<MapExpr>),
    /// A named callback with an iterable or binding that requires the builtin.
    /// Ordinary array arguments can also be represented as a call.
    NamedMap {
        target: String,
        iterable: Box<ForIterable>,
        callback: String,
        /// Explicit compile-time arguments on the callback reference.
        parametrics: Vec<Expr>,
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
    Match {
        subject: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    UnrollFor(Box<UnrollFor>),
    For(Box<ForLoop>),
}

/// The argument syntax written on a struct constructor.
#[derive(Clone, Copy)]
pub(crate) enum StructLiteralArgs<'a> {
    Omitted,
    Integers(&'a [u32]),
    Expressions(&'a [Expr]),
    Types(&'a [Type]),
}

/// A borrowed view of a struct constructor's arguments, fields, and splat.
pub(crate) struct StructLiteral<'a> {
    pub(crate) name: &'a str,
    pub(crate) args: StructLiteralArgs<'a>,
    pub(crate) fields: &'a [(String, Expr)],
    pub(crate) base: Option<&'a StructSplat>,
}

impl Expr {
    /// Prepared scopes have already replaced their discarded expressions with
    /// lexical aliases. Only the trailing result still needs contextual typing.
    pub(crate) fn needs_scope_preparation(&self) -> bool {
        matches!(self, Self::Scope { statements, .. } if statements.is_empty() || !statements.iter().all(|s|
            matches!(s, ScopeStatement::Expression(Expr::Name(name)) if name.starts_with("$scope_"))))
    }

    /// Borrow the common parts of a struct literal, preserving its argument
    /// syntax.
    pub(crate) fn struct_literal(&self) -> Option<StructLiteral<'_>> {
        let (name, args, fields, base) = match self {
            Self::Struct { name, fields, base } => (name, StructLiteralArgs::Omitted, fields, base),
            Self::AppliedStruct {
                name,
                args,
                fields,
                base,
            } => (name, StructLiteralArgs::Integers(args), fields, base),
            Self::ParametricStruct {
                name,
                args,
                fields,
                base,
            } => (name, StructLiteralArgs::Expressions(args), fields, base),
            Self::TypeAppliedStruct {
                name,
                args,
                fields,
                base,
            } => (name, StructLiteralArgs::Types(args), fields, base),
            _ => return None,
        };
        Some(StructLiteral {
            name,
            args,
            fields,
            base: base.as_deref(),
        })
    }

    /// A loop without an accumulator annotation takes its carry type from its
    /// seed.
    pub(crate) fn unannotated_loop_seed(&self) -> Option<&Expr> {
        match self {
            Self::For(fold) if fold.annotation.is_none() => Some(&fold.seed),
            Self::UnrollFor(fold) if fold.annotation.is_none() => Some(&fold.seed),
            _ => None,
        }
    }

    /// The written elements and fill flag of an array without an explicit type.
    pub(crate) fn unannotated_array(&self) -> Option<(&[Expr], bool)> {
        match self {
            Self::Array(members) => Some((members, false)),
            Self::ArrayLiteral {
                annotation: None,
                members,
                has_ellipsis,
            } => Some((members, *has_ellipsis)),
            _ => None,
        }
    }
}

/// An array mapped by an inline unary lambda. The checker verifies that the
/// body is closed over its parameter and visible module declarations.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum WidthSliceType {
    Concrete(Type),
    Bits {
        signed: Box<Expr>,
        bit_count: Box<Expr>,
    },
}

/// Structural declarations, value bindings, and register or child port drivers.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Stmt {
    /// Check a Boolean constexpr when elaborating this block.
    ConstAssert(ConstAssert),
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
    /// Declare a typed combinational value before its driver is available.
    Wire {
        name: String,
        ty: Type,
    },
    /// Supply the single driver for a previously declared wire.
    DriveWire {
        name: String,
        value: Expr,
    },
    /// Destructure a tuple or discard an initializer using the DSLX let
    /// grammar.
    LetPattern(LetBinding),
    Write {
        name: String,
        value: Expr,
    },
    If {
        condition: Expr,
        body: Vec<Stmt>,
    },
}

/// A one-based source line and character column within a BX file.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourcePosition {
    pub line: usize,
    pub column: usize,
}

/// Function tests evaluate pure expressions. Block tests can drive and clock
/// hardware.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TestKind {
    Function,
    Block,
}

/// A property block's random inputs and its source-level predicate result type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuickCheckSignature {
    pub params: Vec<Param>,
    pub result: Type,
}

/// A source test with its declared name and ordered procedural body.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TestDecl {
    pub name: String,
    pub kind: TestKind,
    pub position: SourcePosition,
    pub quickcheck: Option<QuickCheckSignature>,
    pub statements: Vec<TestStmt>,
    /// Module names referenced in annotations replaced by an outer annotation.
    pub overwritten_literal_types: BTreeSet<String>,
}

/// A named Boolean fact checked on each observation of an invariant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvariantLemma {
    pub name: String,
    pub position: SourcePosition,
}

impl InvariantLemma {
    /// Capture the marked binding before later lets can shadow its spelling.
    /// A dot cannot occur in a source BX binding identifier.
    pub(crate) fn observation_name(index: usize) -> String {
        format!("bx.lemma.{index}")
    }
}

/// A private clocked property with named lemmas sampled in the same cycle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvariantDecl {
    pub block: Block,
    pub position: SourcePosition,
    pub lemmas: Vec<InvariantLemma>,
    pub overwritten_literal_types: BTreeSet<String>,
}

/// One test action and the location used to report its outcome.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TestStmt {
    pub position: SourcePosition,
    pub kind: TestStmtKind,
}

/// Procedural operations available inside BX test declarations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TestStmtKind {
    /// The trailing boolean expression of a quickcheck block.
    Property(Expr),
    Reg {
        name: String,
        ty: Type,
        /// An omitted initializer drives the type's all-zero value.
        init: Option<Expr>,
    },
    Let(LetBinding),
    Assign {
        name: String,
        value: Expr,
    },
    Tick {
        reset: Option<Expr>,
    },
    Assert {
        predicate: Expr,
        label: Option<String>,
    },
    AssertEq {
        actual: Expr,
        expected: Expr,
    },
    For {
        pattern: BindingPattern,
        iterable: ForIterable,
        body: Vec<TestStmt>,
    },
    If {
        condition: Expr,
        then_body: Vec<TestStmt>,
        else_body: Vec<TestStmt>,
    },
}

/// The byte range of a construct within a BX source file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceSpan {
    pub start: usize,
    pub end: usize,
}

/// The source snapshot used to render BX diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceFile {
    pub text: Arc<str>,
    pub path: Option<PathBuf>,
}

/// Diagnostic source ranges for the statements in a BX block body.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockSource {
    pub file: SourceFile,
    pub(crate) statements: Vec<SourceSpan>,
    pub(crate) output: SourceSpan,
}

/// A stage as written in BX, before a module constant has been looked up.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StageAnnotation {
    Literal(u64),
    Constant(String),
}

impl From<u64> for StageAnnotation {
    fn from(stage: u64) -> Self {
        Self::Literal(stage)
    }
}

/// Output stages declared in this block's local stage numbering. Checked
/// blocks carry numbers, while parsed blocks can still refer to constants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutputCycles<T = u64> {
    /// `-> Ports @1` places every output port at stage 1.
    Uniform(T),
    /// `-> Ports @(live: 0, saved: 1)` gives each physical port its own stage.
    ByPort(BTreeMap<String, T>),
}

/// The stage declarations on a single block signature.
#[derive(Clone)]
pub(crate) struct BlockStages<T> {
    pub inputs: BTreeMap<String, T>,
    pub output: Option<OutputCycles<T>>,
}

/// A block definition, including its implicit timing domain and output
/// expression.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Block {
    pub name: String,
    /// Location of the declared name, when parsed from a source file.
    pub name_position: Option<SourcePosition>,
    /// Source text and ranges for blocks that declare deferred bindings.
    pub source: Option<BlockSource>,
    /// Warning kinds suppressed for this definition by `#[allow(...)]`.
    pub allowed_warnings: BTreeSet<BxWarningKind>,
    pub public: bool,
    /// Type parameter names in declaration order.
    pub type_params: Vec<String>,
    /// Named callable parameters, in the order supplied at a block call site.
    pub callback_params: Vec<CallbackParam>,
    /// Explicit compile-time values in declaration order.
    pub value_parametrics: Vec<ParametricBinding>,
    pub clock: String,
    pub reset: Option<String>,
    pub params: Vec<Param>,
    /// Explicit cycle offsets by data input name.
    pub input_cycles: BTreeMap<String, StageAnnotation>,
    pub result: Type,
    /// Declared output timing, if the block opts into checking it.
    pub output_cycles: Option<OutputCycles<StageAnnotation>>,
    pub statements: Vec<Stmt>,
    pub output: Expr,
}

impl Block {
    /// Visit annotations in signature scope. An input enters scope at its
    /// declaration and can shadow a module constant for subsequent stages.
    pub(crate) fn map_stages<T: From<u64>>(
        &self,
        constant: impl Fn(&str) -> Result<T, String>,
    ) -> Result<BlockStages<T>, String> {
        let mut stages = BlockStages {
            inputs: BTreeMap::new(),
            output: None,
        };
        if self.input_cycles.is_empty() && self.output_cycles.is_none() {
            return Ok(stages);
        }
        let mut locals: BTreeSet<&str> = std::iter::once(self.clock.as_str())
            .chain(self.reset.as_deref())
            .chain(self.type_params.iter().map(String::as_str))
            .chain(self.value_parametrics.iter().map(|p| p.name.as_str()))
            .chain(self.callback_params.iter().map(|p| p.name.as_str()))
            .collect();
        let stage = |annotation: &StageAnnotation, locals: &BTreeSet<&str>| match annotation {
            StageAnnotation::Literal(value) => Ok(T::from(*value)),
            StageAnnotation::Constant(name) => {
                let head = name
                    .split_once("::")
                    .map_or(name.as_str(), |(head, _)| head);
                if locals.contains(head) {
                    return Err(format!(
                        "stage '{name}' refers to a block parameter, expected a module constant"
                    ));
                }
                constant(name)
            }
        };
        for param in &self.params {
            locals.insert(&param.name);
            if let Some(annotation) = self.input_cycles.get(&param.name) {
                stages
                    .inputs
                    .insert(param.name.clone(), stage(annotation, &locals)?);
            }
        }
        // Parsed annotations are attached to declared inputs. Check that
        // invariant for callers that construct a Block directly as well.
        if let Some(unknown) = self
            .input_cycles
            .keys()
            .find(|p| !stages.inputs.contains_key(*p))
        {
            return Err(format!(
                "unknown input port '{unknown}' in stage annotation"
            ));
        }
        stages.output = match &self.output_cycles {
            None => None,
            Some(OutputCycles::Uniform(value)) => {
                Some(OutputCycles::Uniform(stage(value, &locals)?))
            }
            Some(OutputCycles::ByPort(ports)) => Some(OutputCycles::ByPort(
                ports
                    .iter()
                    .map(|(port, value)| Ok((port.clone(), stage(value, &locals)?)))
                    .collect::<Result<_, String>>()?,
            )),
        };
        Ok(stages)
    }

    /// Whether this declaration needs a call to produce a concrete block.
    pub fn is_parametric(&self) -> bool {
        !self.type_params.is_empty()
            || !self.callback_params.is_empty()
            || !self.value_parametrics.is_empty()
    }

    /// Visit explicitly supplied formals in call-site argument order.
    pub fn parametric_bindings(&self) -> impl Iterator<Item = FunctionParametricRef<'_>> {
        explicit_parametric_bindings(&self.value_parametrics, &self.callback_params)
    }
}

/// One leaf of a BX or DSLX module import. Its final component is bound in BX.
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
    /// `#![allow(block_reset_signal)]` permits ordinary blocks in this file
    /// to declare an explicit reset signal.
    pub allow_block_reset_signal: bool,
    /// `#![require_block_timing]` requires output timing on this file's block
    /// definitions.
    pub require_block_timing: bool,
    /// `#![require_define_before_use]` requires this file's declarations to
    /// precede their uses.
    pub require_define_before_use: bool,
    /// Declaration labels in source order, including each leaf of a use tree
    /// and the source location label of each module assertion.
    /// AST callers opting into `require_define_before_use` must list every
    /// module declaration and use binding exactly once.
    pub declaration_order: Vec<String>,
    /// Explicit warning allowances on declarations other than block
    /// definitions.
    pub declaration_allowances: BTreeMap<String, BTreeSet<BxWarningKind>>,
    /// Module references in literal annotations replaced by an outer
    /// annotation, grouped by the declaration that used them. A bare name must
    /// denote a type. A qualified name requires only its head to be in scope.
    pub overwritten_literal_types: BTreeMap<String, BTreeSet<String>>,
    pub uses: Vec<UsePath>,
    pub structs: Vec<StructDef>,
    pub type_aliases: Vec<TypeAlias>,
    pub enums: Vec<EnumDef>,
    pub constants: Vec<ConstDef>,
    /// Compile-time assertions keyed by source location label. These labels
    /// identify declarations for diagnostics and ordering without binding
    /// names.
    pub const_asserts: BTreeMap<String, ConstAssert>,
    pub externs: Vec<ExternFn>,
    pub functions: Vec<Function>,
    pub extern_blocks: Vec<ExternBlock>,
    pub blocks: Vec<Block>,
    /// Tests declared in this source file, in execution order.
    pub tests: Vec<TestDecl>,
    /// Private property circuits compiled by the verification commands.
    pub invariants: Vec<InvariantDecl>,
}
