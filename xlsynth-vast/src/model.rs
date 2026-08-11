// SPDX-License-Identifier: Apache-2.0

//! File-owned Verilog AST nodes and lightweight, copyable node handles.

use std::collections::BTreeSet;

/// Selects the spelling rules used when emitting a hardware description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VastFileType {
    Verilog,
    SystemVerilog,
}

/// Describes the direction of a module port.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModulePortDirection {
    Input,
    Output,
    InOut,
}

/// Selects the declaration keyword for a Verilog definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataKind {
    Reg,
    Wire,
    Logic,
    Integer,
    Int,
    User,
    UntypedEnum,
    Genvar,
}

/// Distinguishes the supported procedural block forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AlwaysKind {
    AlwaysFF,
    AlwaysAt,
    AlwaysComb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Id {
    pub(crate) file_id: u64,
    pub(crate) index: usize,
}

macro_rules! define_handle {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) Id);
    };
}

define_handle!(
    VastModule,
    "Identifies a module owned by a [`crate::VastFile`]."
);
define_handle!(
    ModulePort,
    "Identifies a module port owned by a [`crate::VastFile`]."
);
define_handle!(
    VastDataType,
    "Identifies a data type owned by a [`crate::VastFile`]."
);
define_handle!(
    LogicRef,
    "References a named logic expression owned by a file."
);
define_handle!(
    Expr,
    "References an expression owned by a [`crate::VastFile`]."
);
define_handle!(
    IndexableExpr,
    "References an expression suitable for indexing."
);
define_handle!(
    Slice,
    "References a part-select expression owned by a file."
);
define_handle!(Index, "References a bit-select expression owned by a file.");
define_handle!(
    Instantiation,
    "Identifies a module instantiation owned by a file."
);
define_handle!(
    BlankLine,
    "Identifies a reusable blank-line formatting artifact."
);
define_handle!(
    InlineVerilogStatement,
    "Identifies inline Verilog text owned by a file."
);
define_handle!(Comment, "Identifies a Verilog line-comment artifact.");
define_handle!(MacroRef, "References a preprocessor macro expression.");
define_handle!(MacroStatement, "Identifies a preprocessor macro statement.");
define_handle!(ContinuousAssignment, "Identifies a continuous assignment.");
define_handle!(VastAlwaysBase, "Identifies a procedural always block.");
define_handle!(
    VastStatementBlock,
    "Identifies a sequence of procedural statements."
);
define_handle!(VastStatement, "Identifies a reusable assignment statement.");
define_handle!(Conditional, "Identifies an if/else conditional statement.");
define_handle!(CaseStatement, "Identifies a case statement and its arms.");
define_handle!(
    GenerateLoop,
    "Identifies a generate loop and its module members."
);
define_handle!(ParameterRef, "References a module parameter expression.");
define_handle!(LocalparamRef, "References a local parameter expression.");
define_handle!(Def, "Identifies a typed declaration owned by a file.");

impl LogicRef {
    /// Returns this named reference as an ordinary expression.
    pub fn to_expr(&self) -> Expr {
        Expr(self.0)
    }

    /// Returns this named reference as an indexable expression.
    pub fn to_indexable_expr(&self) -> IndexableExpr {
        IndexableExpr(self.0)
    }
}

impl IndexableExpr {
    /// Returns the indexable expression as an ordinary expression.
    pub fn to_expr(&self) -> Expr {
        Expr(self.0)
    }
}

impl Slice {
    /// Returns the part select as an ordinary expression.
    pub fn to_expr(&self) -> Expr {
        Expr(self.0)
    }
}

impl Index {
    /// Returns the bit select as an ordinary expression.
    pub fn to_expr(&self) -> Expr {
        Expr(self.0)
    }

    /// Allows a multidimensional bit select to be indexed again.
    pub fn to_indexable_expr(&self) -> IndexableExpr {
        IndexableExpr(self.0)
    }
}

impl ParameterRef {
    /// Returns this parameter reference as an ordinary expression.
    pub fn to_expr(&self) -> Expr {
        Expr(self.0)
    }

    /// Returns this parameter reference as an indexable expression.
    pub fn to_indexable_expr(&self) -> IndexableExpr {
        IndexableExpr(self.0)
    }
}

impl LocalparamRef {
    /// Returns this local parameter reference as an ordinary expression.
    pub fn to_expr(&self) -> Expr {
        Expr(self.0)
    }
}

impl MacroRef {
    /// Returns this macro reference as an ordinary expression.
    pub fn to_expr(&self) -> Expr {
        Expr(self.0)
    }
}

/// Ordered arenas containing all nodes belonging to one Verilog file.
#[derive(Default, Debug)]
pub(crate) struct Ast {
    pub(crate) file_items: Vec<FileItem>,
    pub(crate) modules: Vec<ModuleData>,
    pub(crate) expressions: Vec<ExprData>,
    pub(crate) data_types: Vec<TypeData>,
    pub(crate) ports: Vec<PortData>,
    pub(crate) blocks: Vec<BlockData>,
    pub(crate) conditionals: Vec<ConditionalData>,
    pub(crate) cases: Vec<CaseData>,
    pub(crate) generates: Vec<GenerateData>,
    pub(crate) always: Vec<AlwaysData>,
    pub(crate) instantiations: Vec<InstantiationData>,
    pub(crate) defs: Vec<DefData>,
    pub(crate) artifacts: Vec<Artifact>,
    pub(crate) assignments: Vec<AssignmentData>,
}

#[derive(Clone, Debug)]
pub(crate) enum FileItem {
    Module(VastModule),
    Include(String),
    Comment(String),
    BlankLine,
}

#[derive(Clone, Debug)]
pub(crate) struct ModuleData {
    pub(crate) name: String,
    pub(crate) parameter_ports: Vec<ParameterPortData>,
    pub(crate) ports: Vec<ModulePort>,
    pub(crate) members: Vec<MemberData>,
    pub(crate) defined_names: BTreeSet<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct ParameterPortData {
    pub(crate) name: String,
    pub(crate) data_type: Option<VastDataType>,
    pub(crate) value: Expr,
}

#[derive(Clone, Debug)]
pub(crate) struct PortData {
    pub(crate) direction: ModulePortDirection,
    pub(crate) name: String,
    pub(crate) data_type: VastDataType,
    pub(crate) kind: DataKind,
    pub(crate) logic_ref: LogicRef,
}

#[derive(Clone, Debug)]
pub(crate) enum ExprData {
    Name {
        name: String,
        data_type: Option<VastDataType>,
    },
    Literal {
        text: String,
        value: Option<i64>,
    },
    Unary {
        op: UnaryOp,
        arg: Expr,
    },
    Binary {
        op: BinaryOp,
        lhs: Expr,
        rhs: Expr,
    },
    Ternary {
        condition: Expr,
        consequent: Expr,
        alternate: Expr,
    },
    Index {
        subject: Expr,
        index: Expr,
    },
    Slice {
        subject: Expr,
        hi: Expr,
        lo: Expr,
    },
    Concat {
        replication: Option<Expr>,
        elements: Vec<Expr>,
    },
    ArrayAssignmentPattern(Vec<Expr>),
    WidthCast {
        width: Expr,
        value: Expr,
    },
    TypeCast {
        data_type: VastDataType,
        value: Expr,
    },
    PosEdge(Expr),
    NegEdge(Expr),
    Macro {
        name: String,
        args: Option<Vec<Expr>>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnaryOp {
    Negate,
    BitwiseNot,
    LogicalNot,
    AndReduce,
    OrReduce,
    XorReduce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Power,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Shll,
    Shra,
    Shrl,
    Ne,
    CaseNe,
    Eq,
    CaseEq,
    Ge,
    Gt,
    Le,
    Lt,
    LogicalAnd,
    LogicalOr,
    NeX,
    EqX,
}

/// Keeps Verilog `integer` and SystemVerilog `int` semantically distinct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntegerTypeKind {
    Integer,
    Int,
}

#[derive(Clone, Debug)]
pub(crate) enum TypeData {
    Scalar {
        signed: bool,
    },
    BitVector {
        width: Expr,
        width_value: Option<i64>,
        signed: bool,
    },
    Extern {
        package: Option<String>,
        name: String,
    },
    Integer {
        signed: bool,
        kind: IntegerTypeKind,
    },
    PackedArray {
        element: VastDataType,
        dimensions: Vec<i64>,
    },
    UnpackedArray {
        element: VastDataType,
        dimensions: Vec<i64>,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum MemberData {
    Declaration {
        kind: DataKind,
        name: String,
        data_type: VastDataType,
    },
    ContinuousAssignment {
        lhs: Expr,
        rhs: Expr,
    },
    Instantiation(Instantiation),
    Always(VastAlwaysBase),
    Generate(GenerateLoop),
    Parameter {
        local: bool,
        def: Option<Def>,
        name: String,
        rhs: Expr,
        int_kind: bool,
    },
    Conditional(Conditional),
    Comment(String),
    BlankLine,
    Inline(String),
    Macro {
        expr: Expr,
        semicolon: bool,
    },
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BlockData {
    pub(crate) statements: Vec<StatementData>,
}

#[derive(Clone, Debug)]
pub(crate) enum StatementData {
    BlockingAssignment { lhs: Expr, rhs: Expr },
    NonblockingAssignment { lhs: Expr, rhs: Expr },
    ContinuousAssignment { lhs: Expr, rhs: Expr },
    Conditional(Conditional),
    Case(CaseStatement),
    Comment(String),
    BlankLine,
    Inline(String),
}

#[derive(Clone, Debug)]
pub(crate) struct ConditionalData {
    pub(crate) condition: Expr,
    pub(crate) consequent: VastStatementBlock,
    pub(crate) alternates: Vec<(Option<Expr>, VastStatementBlock)>,
}

#[derive(Clone, Debug)]
pub(crate) struct CaseData {
    pub(crate) selector: Expr,
    pub(crate) arms: Vec<(Option<Expr>, VastStatementBlock)>,
}

#[derive(Clone, Debug)]
pub(crate) struct GenerateData {
    pub(crate) genvar: LogicRef,
    pub(crate) name: String,
    pub(crate) start: Expr,
    pub(crate) end: Expr,
    pub(crate) label: Option<String>,
    pub(crate) members: Vec<MemberData>,
}

#[derive(Clone, Debug)]
pub(crate) struct AlwaysData {
    pub(crate) kind: AlwaysKind,
    pub(crate) sensitivity_list: Vec<Expr>,
    pub(crate) block: VastStatementBlock,
}

#[derive(Clone, Debug)]
pub(crate) struct InstantiationData {
    pub(crate) module_name: String,
    pub(crate) instance_name: String,
    pub(crate) parameters: Vec<(String, Expr)>,
    pub(crate) ports: Vec<(String, Option<Expr>)>,
}

#[derive(Clone, Debug)]
pub(crate) struct DefData {
    pub(crate) name: String,
    pub(crate) kind: DataKind,
    pub(crate) data_type: VastDataType,
}

#[derive(Clone, Debug)]
pub(crate) enum Artifact {
    BlankLine,
    Comment(String),
    Inline(String),
    MacroStatement { expr: Expr, semicolon: bool },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AssignmentData {
    pub(crate) lhs: Expr,
    pub(crate) rhs: Expr,
    pub(crate) nonblocking: bool,
}
