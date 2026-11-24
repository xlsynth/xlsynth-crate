// SPDX-License-Identifier: Apache-2.0

//! Rust-native builders and deterministic emitters for Verilog and
//! SystemVerilog.
//!
//! A [`VastFile`] owns every AST node. Its inexpensive, copyable handles can
//! only be inspected or mutated through their originating file, so the
//! implementation needs neither shared ownership nor interior mutability.

mod emit;
pub mod helpers;
mod literal;
mod model;

use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use literal::Literal;
use model::{
    AlwaysData, Artifact, AssignmentData, Ast, BinaryOp, BlockData, CaseData, ConditionalData,
    DefData, ExprData, FileItem, FunctionData, FunctionInputData, FunctionLocalData, GenerateData,
    Id, InstantiationData, IntegerTypeKind, LogicState, MemberData, ModuleData, PackageImport,
    ParameterPortData, ParameterPortDeclaration, PortData, StatementData, TypeData, UnaryOp,
};

pub use model::{
    AlwaysKind, BlankLine, CaseStatement, Comment, Conditional, ContinuousAssignment, DataKind,
    Def, Expr, GenerateLoop, Index, IndexableExpr, InlineVerilogStatement, Instantiation,
    LocalparamRef, LogicRef, MacroRef, MacroStatement, ModulePort, ModulePortDirection,
    ParameterKind, ParameterRef, Slice, SvArgument, TimeScaleMagnitude, TimeScaleValue, TimeUnit,
    VastAlwaysBase, VastDataType, VastFileType, VastFunction, VastModule, VastStatement,
    VastStatementBlock,
};

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(1);

/// Reports whether a name is an unescaped, non-keyword SystemVerilog
/// identifier.
pub fn is_system_verilog_simple_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    matches!(bytes.next(), Some(b'A'..=b'Z' | b'a'..=b'z' | b'_'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$'))
        && !is_system_verilog_keyword(name)
}

/// Reports whether `identifier` is reserved in Verilog or SystemVerilog.
///
/// Keywords are case-sensitive. This includes the complete Verilog keyword
/// set together with the SystemVerilog additions through IEEE 1800-2012.
pub fn is_system_verilog_keyword(identifier: &str) -> bool {
    matches!(
        identifier,
        "accept_on"
            | "alias"
            | "always"
            | "always_comb"
            | "always_ff"
            | "always_latch"
            | "and"
            | "assert"
            | "assign"
            | "assume"
            | "automatic"
            | "before"
            | "begin"
            | "bind"
            | "bins"
            | "binsof"
            | "bit"
            | "break"
            | "buf"
            | "bufif0"
            | "bufif1"
            | "byte"
            | "case"
            | "casex"
            | "casez"
            | "cell"
            | "chandle"
            | "checker"
            | "class"
            | "clocking"
            | "cmos"
            | "config"
            | "const"
            | "constraint"
            | "context"
            | "continue"
            | "cover"
            | "covergroup"
            | "coverpoint"
            | "cross"
            | "deassign"
            | "default"
            | "defparam"
            | "design"
            | "disable"
            | "dist"
            | "do"
            | "edge"
            | "else"
            | "end"
            | "endcase"
            | "endchecker"
            | "endclass"
            | "endclocking"
            | "endconfig"
            | "endfunction"
            | "endgenerate"
            | "endgroup"
            | "endinterface"
            | "endmodule"
            | "endpackage"
            | "endprimitive"
            | "endprogram"
            | "endproperty"
            | "endsequence"
            | "endspecify"
            | "endtable"
            | "endtask"
            | "enum"
            | "event"
            | "eventually"
            | "expect"
            | "export"
            | "extends"
            | "extern"
            | "final"
            | "first_match"
            | "for"
            | "force"
            | "foreach"
            | "forever"
            | "fork"
            | "forkjoin"
            | "function"
            | "generate"
            | "genvar"
            | "global"
            | "highz0"
            | "highz1"
            | "if"
            | "iff"
            | "ifnone"
            | "ignore_bins"
            | "illegal_bins"
            | "implements"
            | "implies"
            | "import"
            | "incdir"
            | "include"
            | "initial"
            | "inout"
            | "input"
            | "inside"
            | "instance"
            | "int"
            | "integer"
            | "interconnect"
            | "interface"
            | "intersect"
            | "join"
            | "join_any"
            | "join_none"
            | "large"
            | "let"
            | "liblist"
            | "library"
            | "local"
            | "localparam"
            | "logic"
            | "longint"
            | "macromodule"
            | "matches"
            | "medium"
            | "modport"
            | "module"
            | "nand"
            | "negedge"
            | "nettype"
            | "new"
            | "nexttime"
            | "nmos"
            | "nor"
            | "noshowcancelled"
            | "not"
            | "notif0"
            | "notif1"
            | "null"
            | "or"
            | "output"
            | "package"
            | "packed"
            | "parameter"
            | "pmos"
            | "posedge"
            | "primitive"
            | "priority"
            | "program"
            | "property"
            | "protected"
            | "pull0"
            | "pull1"
            | "pulldown"
            | "pullup"
            | "pulsestyle_ondetect"
            | "pulsestyle_onevent"
            | "pure"
            | "rand"
            | "randc"
            | "randcase"
            | "randsequence"
            | "rcmos"
            | "real"
            | "realtime"
            | "ref"
            | "reg"
            | "reject_on"
            | "release"
            | "repeat"
            | "restrict"
            | "return"
            | "rnmos"
            | "rpmos"
            | "rtran"
            | "rtranif0"
            | "rtranif1"
            | "s_always"
            | "s_eventually"
            | "s_nexttime"
            | "s_until"
            | "s_until_with"
            | "scalared"
            | "sequence"
            | "shortint"
            | "shortreal"
            | "showcancelled"
            | "signed"
            | "small"
            | "soft"
            | "solve"
            | "specify"
            | "specparam"
            | "static"
            | "string"
            | "strong"
            | "strong0"
            | "strong1"
            | "struct"
            | "super"
            | "supply0"
            | "supply1"
            | "sync_accept_on"
            | "sync_reject_on"
            | "table"
            | "tagged"
            | "task"
            | "this"
            | "throughout"
            | "time"
            | "timeprecision"
            | "timeunit"
            | "tran"
            | "tranif0"
            | "tranif1"
            | "tri"
            | "tri0"
            | "tri1"
            | "triand"
            | "trior"
            | "trireg"
            | "type"
            | "typedef"
            | "union"
            | "unique"
            | "unique0"
            | "unsigned"
            | "until"
            | "until_with"
            | "untyped"
            | "use"
            | "uwire"
            | "var"
            | "vectored"
            | "virtual"
            | "void"
            | "wait"
            | "wait_order"
            | "wand"
            | "weak"
            | "weak0"
            | "weak1"
            | "while"
            | "wildcard"
            | "wire"
            | "with"
            | "within"
            | "wor"
            | "xnor"
            | "xor"
    )
}

/// Builds a package import from identifier tokens or a wildcard item.
fn package_import(package: &str, item: &str) -> Result<PackageImport, VastError> {
    if !is_system_verilog_simple_identifier(package) {
        return Err(VastError(format!("invalid package identifier: {package}")));
    }
    if item != "*" && !is_system_verilog_simple_identifier(item) {
        return Err(VastError(format!("invalid import item: {item}")));
    }
    Ok(PackageImport {
        package: package.to_owned(),
        item: item.to_owned(),
    })
}

/// Controls how an arbitrary-width integer appears in generated Verilog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LiteralFormat {
    /// A sized, zero-padded unsigned binary literal, such as `8'b0000_0011`.
    Binary,
    /// A sized, zero-padded unsigned hexadecimal literal, such as `8'h03`.
    Hex,
    /// A sized, signed two's-complement decimal literal, such as `-8'sd1`.
    SignedDecimal,
    /// A sized unsigned decimal literal, such as `8'd255`.
    UnsignedDecimal,
    /// An unsized unsigned binary literal, such as `'b11`.
    UnsizedBinary,
    /// An unsized decimal literal in the signed 32-bit range, such as `255`.
    UnsizedDecimal,
    /// An unsized unsigned hexadecimal literal, such as `'hff`.
    UnsizedHex,
}

/// Describes invalid builder input or an unsupported VAST operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VastError(pub String);

impl fmt::Display for VastError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for VastError {}

/// Owns every module, declaration, expression, and statement in one output
/// file.
#[derive(Debug)]
pub struct VastFile {
    pub(crate) file_type: VastFileType,
    pub(crate) ast: Ast,
    pub(crate) file_id: u64,
}

/// Returns the declaration kind used in duplicate-name diagnostics.
fn parameter_kind_name(kind: ParameterKind, is_type: bool) -> &'static str {
    match (kind, is_type) {
        (ParameterKind::Parameter, false) => "parameter",
        (ParameterKind::Localparam, false) => "localparam",
        (ParameterKind::Parameter, true) => "type parameter",
        (ParameterKind::Localparam, true) => "local type parameter",
    }
}

impl VastFile {
    /// Creates an empty file using the requested Verilog dialect.
    pub fn new(file_type: VastFileType) -> Self {
        Self {
            file_type,
            ast: Ast::default(),
            file_id: NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Returns the Verilog dialect used by this file.
    pub fn file_type(&self) -> VastFileType {
        self.file_type
    }

    /// Checks that a lightweight handle belongs to this file.
    fn check(&self, id: Id) -> usize {
        assert_eq!(
            id.file_id, self.file_id,
            "VAST handle belongs to a different file"
        );
        id.index
    }

    /// Creates an identifier for a node in one of this file's arenas.
    fn id(&self, index: usize) -> Id {
        Id {
            file_id: self.file_id,
            index,
        }
    }

    /// Appends an expression to its stable, file-owned arena.
    fn add_expression(&mut self, expression: ExprData) -> Expr {
        let handle = Expr(self.id(self.ast.expressions.len()));
        self.ast.expressions.push(expression);
        handle
    }

    /// Appends a data type to its stable, file-owned arena.
    fn add_data_type(&mut self, data_type: TypeData) -> VastDataType {
        let handle = VastDataType(self.id(self.ast.data_types.len()));
        self.ast.data_types.push(data_type);
        handle
    }

    /// Creates an empty procedural statement block.
    fn make_statement_block(&mut self) -> VastStatementBlock {
        let handle = VastStatementBlock(self.id(self.ast.blocks.len()));
        self.ast.blocks.push(BlockData::default());
        handle
    }

    /// Creates a named reference, retaining its declared type when available.
    fn make_named_ref(&mut self, name: &str, data_type: Option<VastDataType>) -> LogicRef {
        if let Some(data_type) = data_type {
            self.check(data_type.0);
        }
        LogicRef(
            self.add_expression(ExprData::Name {
                name: name.to_owned(),
                data_type,
            })
            .0,
        )
    }

    /// Validates and appends a module port without changing the file on
    /// failure.
    fn add_port(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
        direction: ModulePortDirection,
        kind: DataKind,
    ) -> Result<LogicRef, VastError> {
        let module_index = self.check(module.0);
        let direction_name = match direction {
            ModulePortDirection::Input => "input",
            ModulePortDirection::Output => "output",
            ModulePortDirection::InOut => "inout",
        };
        self.check_unique_module_name(module, name, direction_name)?;
        self.check(data_type.0);
        if direction == ModulePortDirection::InOut && self.is_non_net_builtin(*data_type) {
            return Err(VastError(
                "this built-in type cannot be used as an inout net".into(),
            ));
        }
        let logic_ref = self.make_named_ref(name, Some(*data_type));
        let port = ModulePort(self.id(self.ast.ports.len()));
        self.ast.ports.push(PortData {
            direction,
            name: name.to_owned(),
            data_type: *data_type,
            kind,
            logic_ref,
        });
        self.ast.modules[module_index].ports.push(port);
        self.ast.modules[module_index]
            .defined_names
            .insert(name.to_owned());
        Ok(logic_ref)
    }

    /// Appends an ordered module member.
    fn push_member(&mut self, module: VastModule, member: MemberData) {
        let module_index = self.check(module.0);
        self.ast.modules[module_index].members.push(member);
    }

    /// Reports whether a built-in type cannot be declared as a net.
    fn is_non_net_builtin(&self, data_type: VastDataType) -> bool {
        match &self.ast.data_types[self.check(data_type.0)] {
            TypeData::Scalar {
                state: LogicState::TwoState,
                ..
            }
            | TypeData::BitVector {
                state: LogicState::TwoState,
                ..
            }
            | TypeData::String => true,
            TypeData::Integer { kind, .. } => matches!(
                kind,
                IntegerTypeKind::Byte
                    | IntegerTypeKind::Shortint
                    | IntegerTypeKind::Int
                    | IntegerTypeKind::Longint
            ),
            TypeData::PackedArray { element, .. } | TypeData::UnpackedArray { element, .. } => {
                self.is_non_net_builtin(*element)
            }
            _ => false,
        }
    }

    /// Enforces the common namespace shared by module ports and declarations.
    fn check_unique_module_name(
        &self,
        module: VastModule,
        name: &str,
        kind: &str,
    ) -> Result<(), VastError> {
        let module = &self.ast.modules[self.check(module.0)];
        if !module.defined_names.contains(name) {
            return Ok(());
        }
        let names = module
            .defined_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        Err(VastError(format!(
            "FAILED_PRECONDITION: Attempted to declare {kind} with name '{name}' multiple times \
             in the same module. Already defined: [{}]",
            names.join(", ")
        )))
    }

    /// Appends an ordered generate-loop member.
    fn push_generate_member(&mut self, generate: GenerateLoop, member: MemberData) {
        let generate_index = self.check(generate.0);
        self.ast.generates[generate_index].members.push(member);
    }

    /// Appends an ordered statement to a procedural block.
    fn push_statement(&mut self, block: VastStatementBlock, statement: StatementData) {
        let block_index = self.check(block.0);
        self.ast.blocks[block_index].statements.push(statement);
    }

    /// Adds a preprocessor include directive to the file.
    pub fn add_include(&mut self, include: &str) {
        self.ast
            .file_items
            .push(FileItem::Include(include.to_owned()));
    }

    /// Adds a file-level blank line.
    pub fn add_blank_line(&mut self, blank_line: BlankLine) {
        self.check(blank_line.0);
        self.ast.file_items.push(FileItem::BlankLine);
    }

    /// Adds a file-level single-line Verilog comment.
    pub fn add_comment_text(&mut self, text: &str) {
        self.ast.file_items.push(FileItem::Comment(text.to_owned()));
    }

    /// Adds an ordered compilation-unit package import. Use `"*"` for all
    /// items. Returns an error for invalid package or item identifiers.
    pub fn add_file_import(&mut self, package: &str, item: &str) -> Result<(), VastError> {
        let import = package_import(package, item)?;
        self.ast.file_items.push(FileItem::Import(import));
        Ok(())
    }

    /// Adds a module at the current position in the output file.
    pub fn add_module(&mut self, name: &str) -> VastModule {
        let handle = VastModule(self.id(self.ast.modules.len()));
        self.ast.modules.push(ModuleData {
            name: name.to_owned(),
            imports: Vec::new(),
            parameter_ports: Vec::new(),
            ports: Vec::new(),
            members: Vec::new(),
            defined_names: Default::default(),
        });
        self.ast.file_items.push(FileItem::Module(handle));
        handle
    }

    /// Adds a package import to a module's header scope. Use `"*"` to import
    /// all items. Invalid identifiers or explicit-import name conflicts return
    /// an error; importing the same package item more than once is allowed.
    pub fn add_module_import(
        &mut self,
        module: VastModule,
        package: &str,
        item: &str,
    ) -> Result<(), VastError> {
        let index = self.check(module.0);
        let import = package_import(package, item)?;
        self.register_module_import(module, &import)?;
        self.ast.modules[index].imports.push(import);
        Ok(())
    }

    /// Adds a package import in the module body at the current member position.
    /// Use `"*"` for a wildcard import. Invalid identifiers or explicit-import
    /// name conflicts return an error; repeat imports of one item are allowed.
    pub fn add_member_import(
        &mut self,
        module: VastModule,
        package: &str,
        item: &str,
    ) -> Result<(), VastError> {
        self.check(module.0);
        let import = package_import(package, item)?;
        self.register_module_import(module, &import)?;
        self.push_member(module, MemberData::Import(import));
        Ok(())
    }

    /// Registers an explicit import in the module namespace, permitting an
    /// identical package import already present in either module scope.
    fn register_module_import(
        &mut self,
        module: VastModule,
        import: &PackageImport,
    ) -> Result<(), VastError> {
        if import.item == "*" {
            return Ok(());
        }
        let index = self.check(module.0);
        let module_data = &self.ast.modules[index];
        let same_import = module_data
            .imports
            .iter()
            .any(|existing| existing.package == import.package && existing.item == import.item)
            || module_data.members.iter().any(|member| {
                matches!(member, MemberData::Import(existing)
                if existing.package == import.package && existing.item == import.item)
            });
        if !same_import {
            self.check_unique_module_name(module, &import.item, "import")?;
        }
        self.ast.modules[index]
            .defined_names
            .insert(import.item.clone());
        Ok(())
    }

    /// Returns the name of a module belonging to this file.
    pub fn module_name(&self, module: VastModule) -> String {
        self.ast.modules[self.check(module.0)].name.clone()
    }

    /// Returns the insertion position for the next module member.
    pub fn module_member_count(&self, module: VastModule) -> usize {
        self.ast.modules[self.check(module.0)].members.len()
    }

    /// Moves signal declarations ahead of statements in the suffix starting at
    /// `start`, preserving order within both groups and leaving nested scopes
    /// and earlier members untouched. Declarations stay on their original side
    /// of imports, parameter declarations, and timescales, preserving their
    /// scope and order.
    pub fn hoist_module_declarations(&mut self, module: VastModule, start: usize) {
        let module_index = self.check(module.0);
        for section in self.ast.modules[module_index].members[start..].split_mut(|member| {
            matches!(
                member,
                MemberData::Import(_)
                    | MemberData::Parameter { .. }
                    | MemberData::TypeParameter { .. }
                    | MemberData::TimeUnit(_)
                    | MemberData::TimePrecision(_)
            )
        }) {
            section.sort_by_key(|member| !matches!(member, MemberData::Declaration { .. }));
        }
    }

    /// Adds an automatic, typed function at the current module-member position.
    pub fn add_function(
        &mut self,
        module: VastModule,
        name: &str,
        result_type: &VastDataType,
    ) -> Result<VastFunction, VastError> {
        let module_index = self.check(module.0);
        self.check(result_type.0);
        if matches!(
            self.ast.data_types[self.check(result_type.0)],
            TypeData::UnpackedArray { .. }
        ) {
            return Err(VastError(
                concat!(
                    "FAILED_PRECONDITION: function result type must not be an anonymous ",
                    "unpacked array; use a named typedef"
                )
                .to_owned(),
            ));
        }
        self.check_unique_module_name(module, name, "function")?;

        let result_ref = self.make_named_ref(name, Some(*result_type));
        let body = self.make_statement_block();
        let handle = VastFunction(self.id(self.ast.functions.len()));
        let mut defined_names = std::collections::BTreeSet::new();
        defined_names.insert(name.to_owned());
        self.ast.functions.push(FunctionData {
            name: name.to_owned(),
            result_type: *result_type,
            result_ref,
            inputs: Vec::new(),
            locals: Vec::new(),
            body,
            defined_names,
        });
        self.ast.modules[module_index]
            .defined_names
            .insert(name.to_owned());
        self.push_member(module, MemberData::Function(handle));
        Ok(handle)
    }

    /// Adds an ordered, typed input to an existing function.
    pub fn function_add_input(
        &mut self,
        function: VastFunction,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.add_function_input(function, name, data_type, DataKind::Reg)
    }

    /// Adds a function input explicitly declared with the `logic` keyword.
    pub fn function_add_logic_input(
        &mut self,
        function: VastFunction,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.add_function_input(function, name, data_type, DataKind::Logic)
    }

    /// Adds a typed function input with its requested declaration keyword.
    fn add_function_input(
        &mut self,
        function: VastFunction,
        name: &str,
        data_type: &VastDataType,
        kind: DataKind,
    ) -> Result<LogicRef, VastError> {
        let function_index = self.check(function.0);
        self.check(data_type.0);
        self.check_unique_function_name(function, name, "input")?;

        let reference = self.make_named_ref(name, Some(*data_type));
        let function = &mut self.ast.functions[function_index];
        function.inputs.push(FunctionInputData {
            kind,
            name: name.to_owned(),
            data_type: *data_type,
        });
        function.defined_names.insert(name.to_owned());
        Ok(reference)
    }

    /// Adds a register declaration local to a function.
    pub fn function_add_reg(
        &mut self,
        function: VastFunction,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.add_function_local(function, name, data_type, DataKind::Reg)
    }

    /// Adds a SystemVerilog logic declaration local to a function.
    pub fn function_add_logic(
        &mut self,
        function: VastFunction,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.add_function_local(function, name, data_type, DataKind::Logic)
    }

    /// Returns the procedural statement block belonging to a function.
    pub fn function_body(&self, function: VastFunction) -> VastStatementBlock {
        self.ast.functions[self.check(function.0)].body
    }

    /// Returns the implicit function-result variable used for assignments.
    pub fn function_result(&self, function: VastFunction) -> LogicRef {
        self.ast.functions[self.check(function.0)].result_ref
    }

    /// Appends a checked, uniquely named declaration to a function.
    fn add_function_local(
        &mut self,
        function: VastFunction,
        name: &str,
        data_type: &VastDataType,
        kind: DataKind,
    ) -> Result<LogicRef, VastError> {
        let function_index = self.check(function.0);
        self.check(data_type.0);
        let kind_name = match kind {
            DataKind::Reg => "reg",
            DataKind::Logic => "logic",
            _ => unreachable!("function locals only use reg or logic declarations"),
        };
        self.check_unique_function_name(function, name, kind_name)?;

        let reference = self.make_named_ref(name, Some(*data_type));
        let function = &mut self.ast.functions[function_index];
        function.locals.push(FunctionLocalData {
            kind,
            name: name.to_owned(),
            data_type: *data_type,
        });
        function.defined_names.insert(name.to_owned());
        Ok(reference)
    }

    /// Enforces the namespace shared by a function result, inputs, and locals.
    fn check_unique_function_name(
        &self,
        function: VastFunction,
        name: &str,
        kind: &str,
    ) -> Result<(), VastError> {
        let function = &self.ast.functions[self.check(function.0)];
        if !function.defined_names.contains(name) {
            return Ok(());
        }
        let names = function
            .defined_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        Err(VastError(format!(
            "FAILED_PRECONDITION: Attempted to declare {kind} with name '{name}' multiple times \
             in function '{}'. Already defined: [{}]",
            function.name,
            names.join(", ")
        )))
    }

    /// Returns all of a module's ports in declaration order.
    pub fn module_ports(&self, module: VastModule) -> Vec<ModulePort> {
        self.ast.modules[self.check(module.0)].ports.clone()
    }

    /// Returns the input ports of a module in declaration order.
    pub fn module_input_ports(&self, module: VastModule) -> Vec<ModulePort> {
        self.module_ports(module)
            .into_iter()
            .filter(|port| self.port_direction(*port) == ModulePortDirection::Input)
            .collect()
    }

    /// Returns the output ports of a module in declaration order.
    pub fn module_output_ports(&self, module: VastModule) -> Vec<ModulePort> {
        self.module_ports(module)
            .into_iter()
            .filter(|port| self.port_direction(*port) == ModulePortDirection::Output)
            .collect()
    }

    /// Returns the declared direction of a module port.
    pub fn port_direction(&self, port: ModulePort) -> ModulePortDirection {
        self.ast.ports[self.check(port.0)].direction
    }

    /// Returns the name of a module port.
    pub fn port_name(&self, port: ModulePort) -> String {
        self.ast.ports[self.check(port.0)].name.clone()
    }

    /// Returns the data type declared for a module port.
    pub fn port_data_type(&self, port: ModulePort) -> VastDataType {
        self.ast.ports[self.check(port.0)].data_type
    }

    /// Returns the flat width of a port, or zero when the width is symbolic.
    pub fn port_width(&self, port: ModulePort) -> i64 {
        self.type_flat_bit_count_as_int64(self.port_data_type(port))
            .unwrap_or_default()
    }

    /// Returns the logic expression corresponding to a module port.
    pub fn port_logic_ref(&self, port: ModulePort) -> LogicRef {
        self.ast.ports[self.check(port.0)].logic_ref
    }

    /// Returns the identifier associated with a named logic expression.
    pub fn logic_ref_name(&self, logic_ref: LogicRef) -> String {
        match &self.ast.expressions[self.check(logic_ref.0)] {
            ExprData::Name { name, .. } => name.clone(),
            _ => panic!("logic reference does not identify a named expression"),
        }
    }

    /// Returns the expression used as the declared width, when available.
    pub fn type_width_expr(&self, data_type: VastDataType) -> Option<Expr> {
        match &self.ast.data_types[self.check(data_type.0)] {
            TypeData::BitVector { width, .. } => Some(*width),
            TypeData::PackedArray { element, .. } | TypeData::UnpackedArray { element, .. } => {
                self.type_width_expr(*element)
            }
            TypeData::Scalar { .. }
            | TypeData::Extern { .. }
            | TypeData::Integer { .. }
            | TypeData::String => None,
        }
    }

    /// Returns whether a type uses signed arithmetic.
    pub fn type_is_signed(&self, data_type: VastDataType) -> bool {
        match &self.ast.data_types[self.check(data_type.0)] {
            TypeData::BitVector { signed, .. } | TypeData::Integer { signed, .. } => *signed,
            TypeData::PackedArray { element, .. } | TypeData::UnpackedArray { element, .. } => {
                self.type_is_signed(*element)
            }
            TypeData::Scalar { signed, .. } => *signed,
            TypeData::Extern { .. } | TypeData::String => false,
        }
    }

    /// Returns the declared scalar or bit-vector width.
    pub fn type_width_as_int64(&self, data_type: VastDataType) -> Result<i64, VastError> {
        match &self.ast.data_types[self.check(data_type.0)] {
            TypeData::Scalar { .. } => Ok(1),
            TypeData::BitVector {
                width_value: Some(width),
                ..
            } => Ok(*width),
            TypeData::BitVector { width, .. } => Err(VastError(format!(
                "Width is not a literal: {}",
                self.emit_expression(width)
            ))),
            TypeData::PackedArray { element, .. } | TypeData::UnpackedArray { element, .. } => {
                self.type_width_as_int64(*element)
            }
            TypeData::Extern { .. } | TypeData::Integer { .. } | TypeData::String => Err(
                VastError("data type does not have a declared bit-vector width".into()),
            ),
        }
    }

    /// Returns the total width of a scalar, integer, vector, or array.
    pub fn type_flat_bit_count_as_int64(&self, data_type: VastDataType) -> Result<i64, VastError> {
        match &self.ast.data_types[self.check(data_type.0)] {
            TypeData::Scalar { .. } => Ok(1),
            TypeData::BitVector { .. } => self.type_width_as_int64(data_type),
            TypeData::Integer { kind, .. } => Ok(kind.bit_width()),
            TypeData::String => Err(VastError("string type has a dynamic bit width".into())),
            TypeData::PackedArray {
                element,
                dimensions,
            }
            | TypeData::UnpackedArray {
                element,
                dimensions,
            } => dimensions.iter().try_fold(
                self.type_flat_bit_count_as_int64(*element)?,
                |width, dimension| {
                    width
                        .checked_mul(*dimension)
                        .ok_or_else(|| VastError("array bit count exceeds i64".into()))
                },
            ),
            TypeData::Extern { package, name } => Err(VastError(format!(
                "external type `{}` has no known bit width",
                package
                    .as_ref()
                    .map(|package| format!("{package}::{name}"))
                    .unwrap_or_else(|| name.clone())
            ))),
        }
    }

    /// Adds an untyped parameter with a default to a module's parameter-port
    /// list, rejecting duplicate names.
    pub fn add_parameter_port(
        &mut self,
        module: VastModule,
        name: &str,
        value: &Expr,
    ) -> Result<LogicRef, VastError> {
        self.add_value_parameter_port(module, name, ParameterKind::Parameter, None, Some(value))
    }

    /// Adds a typed parameter with a default to a module's parameter-port list,
    /// rejecting duplicate names.
    pub fn add_typed_parameter_port(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
        value: &Expr,
    ) -> Result<LogicRef, VastError> {
        self.add_value_parameter_port(
            module,
            name,
            ParameterKind::Parameter,
            Some(data_type),
            Some(value),
        )
    }

    /// Appends a value parameter or local parameter to the parameter-port list.
    ///
    /// Omitting the default makes an overridable parameter required. Local
    /// parameters must have defaults. Duplicate names return an error. All
    /// supplied handles must belong to this file.
    pub fn add_value_parameter_port(
        &mut self,
        module: VastModule,
        name: &str,
        kind: ParameterKind,
        data_type: Option<&VastDataType>,
        default: Option<&Expr>,
    ) -> Result<LogicRef, VastError> {
        let module_index = self.check(module.0);
        if kind == ParameterKind::Localparam && default.is_none() {
            return Err(VastError("local parameters require a default".into()));
        }
        if let Some(data_type) = data_type {
            self.check(data_type.0);
        }
        if let Some(default) = default {
            self.check(default.0);
        }
        self.check_unique_module_name(module, name, parameter_kind_name(kind, false))?;
        let reference = self.make_named_ref(name, data_type.copied());
        self.ast.modules[module_index]
            .parameter_ports
            .push(ParameterPortData {
                name: name.to_owned(),
                kind,
                declaration: ParameterPortDeclaration::Value {
                    data_type: data_type.copied(),
                    default: default.copied(),
                },
            });
        self.ast.modules[module_index]
            .defined_names
            .insert(name.to_owned());
        Ok(reference)
    }

    /// Appends a type parameter and returns a reference to its declared type.
    ///
    /// An overridable type parameter can omit its default. A local type
    /// parameter cannot. Returns an error for a missing local default or an
    /// anonymous unpacked array default, which requires a named typedef, or a
    /// duplicate name.
    pub fn add_type_parameter_port(
        &mut self,
        module: VastModule,
        name: &str,
        kind: ParameterKind,
        default: Option<&VastDataType>,
    ) -> Result<VastDataType, VastError> {
        let module_index = self.check(module.0);
        if kind == ParameterKind::Localparam && default.is_none() {
            return Err(VastError("local parameters require a default".into()));
        }
        if let Some(default) = default {
            self.check_standalone_type(*default)?;
        }
        self.check_unique_module_name(module, name, parameter_kind_name(kind, true))?;
        let reference = self.make_extern_type(name);
        self.ast.modules[module_index]
            .parameter_ports
            .push(ParameterPortData {
                name: name.to_owned(),
                kind,
                declaration: ParameterPortDeclaration::Type {
                    default: default.copied(),
                },
            });
        self.ast.modules[module_index]
            .defined_names
            .insert(name.to_owned());
        Ok(reference)
    }

    /// Adds an input port to the module, rejecting duplicate names.
    pub fn add_input(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.add_port(
            module,
            name,
            data_type,
            ModulePortDirection::Input,
            DataKind::Wire,
        )
    }

    /// Adds an output port to the module, rejecting duplicate names.
    pub fn add_output(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.add_port(
            module,
            name,
            data_type,
            ModulePortDirection::Output,
            DataKind::Wire,
        )
    }

    /// Adds a bidirectional port, returning an error for an invalid net type or
    /// duplicate name.
    pub fn add_inout(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.add_port(
            module,
            name,
            data_type,
            ModulePortDirection::InOut,
            DataKind::Wire,
        )
    }

    /// Adds a `logic` input port, rejecting duplicate names.
    pub fn add_logic_input(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.add_port(
            module,
            name,
            data_type,
            ModulePortDirection::Input,
            DataKind::Logic,
        )
    }

    /// Adds a `logic` output port, rejecting duplicate names.
    pub fn add_logic_output(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.add_port(
            module,
            name,
            data_type,
            ModulePortDirection::Output,
            DataKind::Logic,
        )
    }

    /// Appends a named declaration to a module and returns its logic reference.
    fn add_declaration(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
        kind: DataKind,
    ) -> LogicRef {
        let module_index = self.check(module.0);
        let logic_ref = self.make_named_ref(name, Some(*data_type));
        self.push_member(
            module,
            MemberData::Declaration {
                kind,
                name: name.to_owned(),
                data_type: *data_type,
            },
        );
        self.ast.modules[module_index]
            .defined_names
            .insert(name.to_owned());
        logic_ref
    }

    /// Adds a wire, returning an error for an invalid net type or duplicate
    /// name.
    pub fn add_wire(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        if self.is_non_net_builtin(*data_type) {
            return Err(VastError(
                "this built-in type cannot be used as a wire net".into(),
            ));
        }
        self.check_unique_module_name(module, name, "wire")?;
        Ok(self.add_declaration(module, name, data_type, DataKind::Wire))
    }

    /// Ties corresponding bits of two nets with bidirectional Verilog
    /// primitives.
    pub fn add_tran(
        &mut self,
        module: VastModule,
        name: &str,
        width: usize,
        left: LogicRef,
        right: LogicRef,
    ) -> Result<(), VastError> {
        if width == 0 {
            return Err(VastError("tran requires a positive width".into()));
        }
        self.check(left.0);
        self.check(right.0);
        self.check_unique_module_name(module, name, "tran")?;
        let index = self.check(module.0);
        self.ast.modules[index]
            .defined_names
            .insert(name.to_owned());
        self.push_member(
            module,
            MemberData::Tran {
                name: name.to_owned(),
                width,
                left,
                right,
            },
        );
        Ok(())
    }

    /// Appends a register declaration to a module.
    pub fn add_reg(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.check_unique_module_name(module, name, "reg")?;
        Ok(self.add_declaration(module, name, data_type, DataKind::Reg))
    }

    /// Appends a SystemVerilog logic declaration to a module.
    pub fn add_logic(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, VastError> {
        self.check_unique_module_name(module, name, "logic")?;
        Ok(self.add_declaration(module, name, data_type, DataKind::Logic))
    }

    /// Creates a width-annotated, arbitrary-precision integer literal.
    ///
    /// Literal widths must be between one and 1,048,576 bits.
    /// Unsized decimal magnitudes must fit in [`i32::MAX`]; unsized binary and
    /// hexadecimal magnitudes must fit in [`u32::MAX`]. These portable bounds
    /// apply to the value, regardless of its declared source width.
    pub fn make_literal(&mut self, text: &str, format: &LiteralFormat) -> Result<Expr, VastError> {
        let literal = Literal::parse(text)?;
        literal.validate_format(*format)?;
        let value = literal.value.to_str_radix(10).parse::<i64>().ok();
        Ok(self.add_expression(ExprData::Literal {
            text: literal.format(*format),
            value,
        }))
    }

    /// Creates a sized binary pattern from MSB-first `0`, `1`, and `?` digits.
    pub fn make_binary_pattern(&mut self, bits_msb: &str) -> Result<Expr, VastError> {
        self.make_checked_binary_literal(
            bits_msb,
            b"01?",
            "binary patterns require 1 to 1048576 MSB-first 0, 1, or ? digits",
        )
    }

    /// Creates a sized four-state binary literal from MSB-first `0`, `1`, `x`,
    /// and `z` digits. Letter case and all unknown or high-impedance bits
    /// are preserved.
    pub fn make_four_state_binary_literal(&mut self, bits_msb: &str) -> Result<Expr, VastError> {
        self.make_checked_binary_literal(
            bits_msb,
            b"01xXzZ",
            "four-state binary literals require 1 to 1048576 MSB-first 0, 1, x, or z digits",
        )
    }

    /// Checks a bounded binary digit sequence and creates its sized literal
    /// expression.
    fn make_checked_binary_literal(
        &mut self,
        bits_msb: &str,
        allowed: &[u8],
        error: &str,
    ) -> Result<Expr, VastError> {
        if bits_msb.is_empty()
            || bits_msb.len() > (1 << 20)
            || !bits_msb.bytes().all(|bit| allowed.contains(&bit))
        {
            return Err(VastError(error.to_owned()));
        }
        let value = bits_msb.bytes().try_fold(0_i64, |value, bit| match bit {
            b'0' | b'1' => value.checked_mul(2)?.checked_add(i64::from(bit - b'0')),
            _ => None,
        });
        Ok(self.add_expression(ExprData::Literal {
            text: format!("{}'b{bits_msb}", bits_msb.len()),
            value,
        }))
    }

    /// Creates an unsized decimal literal from any signed 32-bit value.
    ///
    /// Negative values, including [`i32::MIN`], are emitted with their leading
    /// minus sign; nonnegative values are emitted without a size or radix.
    pub fn make_unsized_decimal_literal(&mut self, value: i32) -> Expr {
        self.add_expression(ExprData::Literal {
            text: value.to_string(),
            value: Some(i64::from(value)),
        })
    }

    /// Creates a quoted Verilog string literal from decoded Rust text.
    ///
    /// The value is stored without Verilog quotes or escapes. Emission adds the
    /// surrounding quotes and escapes special characters as needed.
    pub fn make_string_literal(&mut self, value: &str) -> Expr {
        self.add_expression(ExprData::StringLiteral {
            value: value.to_owned(),
        })
    }

    /// Creates a time literal from a decimal magnitude such as `1`, `1.5`, or
    /// `1_000`.
    pub fn make_time_literal(
        &mut self,
        magnitude: &str,
        unit: TimeUnit,
    ) -> Result<Expr, VastError> {
        let mut parts = magnitude.split('.');
        let integer = parts.next().unwrap_or_default();
        let fraction = parts.next();
        let valid = |part: &str| {
            let bytes = part.as_bytes();
            !bytes.is_empty()
                && bytes.first().is_some_and(u8::is_ascii_digit)
                && bytes
                    .iter()
                    .all(|digit| digit.is_ascii_digit() || *digit == b'_')
        };
        if !valid(integer) || fraction.is_some_and(|part| !valid(part)) || parts.next().is_some() {
            return Err(VastError(
                "time literal magnitude must be an unsigned decimal or fixed-point number".into(),
            ));
        }
        Ok(self.add_expression(ExprData::TimeLiteral {
            magnitude: magnitude.to_owned(),
            unit,
        }))
    }

    /// Creates an expression from complete Verilog source text.
    ///
    /// Supply any required Verilog quotes and escapes as part of the text.
    /// VAST preserves that text and parenthesizes it when nested in another
    /// expression that requires grouping.
    pub fn make_inline_verilog_expression(&mut self, text: &str) -> Expr {
        self.add_expression(ExprData::InlineVerilog {
            text: text.to_owned(),
        })
    }

    /// Refers to an external value, returning an error for an invalid simple
    /// identifier.
    pub fn make_extern_value_ref(&mut self, name: &str) -> Result<Expr, VastError> {
        if !is_system_verilog_simple_identifier(name) {
            return Err(VastError(format!(
                "invalid external value identifier: {name}"
            )));
        }
        Ok(self.add_expression(ExprData::ExternalValue {
            package: None,
            name: name.to_owned(),
        }))
    }

    /// Refers to an external value using separate simple package and value
    /// identifiers. Invalid package or value identifiers return an error.
    pub fn make_extern_package_value_ref(
        &mut self,
        package: &str,
        name: &str,
    ) -> Result<Expr, VastError> {
        if !is_system_verilog_simple_identifier(package) {
            return Err(VastError(format!("invalid package identifier: {package}")));
        }
        if !is_system_verilog_simple_identifier(name) {
            return Err(VastError(format!(
                "invalid external value identifier: {name}"
            )));
        }
        Ok(self.add_expression(ExprData::ExternalValue {
            package: Some(package.to_owned()),
            name: name.to_owned(),
        }))
    }

    /// Checks that a type can be emitted without a declarator separating its
    /// array dimensions.
    fn check_standalone_type(&self, data_type: VastDataType) -> Result<(), VastError> {
        let index = self.check(data_type.0);
        if matches!(self.ast.data_types[index], TypeData::UnpackedArray { .. }) {
            return Err(VastError(
                "anonymous unpacked array types require a named typedef".into(),
            ));
        }
        Ok(())
    }

    /// Creates an unsigned, four-state one-bit logic scalar type.
    pub fn make_scalar_type(&mut self) -> VastDataType {
        self.add_data_type(TypeData::Scalar {
            signed: false,
            state: LogicState::FourState,
        })
    }

    /// Creates an unsigned, two-state SystemVerilog `bit` scalar type.
    pub fn make_bit_type(&mut self) -> VastDataType {
        self.add_data_type(TypeData::Scalar {
            signed: false,
            state: LogicState::TwoState,
        })
    }

    /// Creates a two-state SystemVerilog `bit` vector, collapsing width one to
    /// a scalar. Returns an error if the width is not positive.
    pub fn make_two_state_bit_vector_type(
        &mut self,
        bit_count: i64,
        signed: bool,
    ) -> Result<VastDataType, VastError> {
        self.make_integral_vector_type(bit_count, signed, LogicState::TwoState)
    }

    /// Creates a two-state SystemVerilog `bit` vector with an expression-valued
    /// width. Returns an error if a known width is not positive.
    pub fn make_two_state_bit_vector_type_expr(
        &mut self,
        width: &Expr,
        signed: bool,
    ) -> Result<VastDataType, VastError> {
        self.make_integral_vector_type_expr(width, signed, LogicState::TwoState)
    }

    /// Creates a four-state logic vector, collapsing width one to a scalar.
    /// Despite its legacy name, this does not create a SystemVerilog two-state
    /// `bit` vector. Returns an error if the width is not positive.
    pub fn make_bit_vector_type(
        &mut self,
        bit_count: i64,
        signed: bool,
    ) -> Result<VastDataType, VastError> {
        self.make_integral_vector_type(bit_count, signed, LogicState::FourState)
    }

    /// Creates a four-state logic vector whose width is determined by an
    /// expression. Returns an error if a known width is not positive.
    pub fn make_bit_vector_type_expr(
        &mut self,
        width: &Expr,
        signed: bool,
    ) -> Result<VastDataType, VastError> {
        self.make_integral_vector_type_expr(width, signed, LogicState::FourState)
    }

    /// Creates an integral vector of the requested state and constant width.
    fn make_integral_vector_type(
        &mut self,
        bit_count: i64,
        signed: bool,
        state: LogicState,
    ) -> Result<VastDataType, VastError> {
        if bit_count <= 0 {
            return Err(VastError(
                "bit-vector width must be greater than zero".into(),
            ));
        }
        if bit_count == 1 {
            return Ok(self.add_data_type(TypeData::Scalar { signed, state }));
        }
        let width = self.add_expression(ExprData::Literal {
            text: bit_count.to_string(),
            value: Some(bit_count),
        });
        Ok(self.add_data_type(TypeData::BitVector {
            width,
            width_value: Some(bit_count),
            signed,
            state,
        }))
    }

    /// Creates an integral vector of the requested state and expression-valued
    /// width.
    fn make_integral_vector_type_expr(
        &mut self,
        width: &Expr,
        signed: bool,
        state: LogicState,
    ) -> Result<VastDataType, VastError> {
        let expression_index = self.check(width.0);
        let width_value = match &self.ast.expressions[expression_index] {
            ExprData::Literal { value, .. } => *value,
            _ => None,
        };
        if width_value.is_some_and(|width| width <= 0) {
            return Err(VastError(
                "bit-vector width must be greater than zero".into(),
            ));
        }
        Ok(self.add_data_type(TypeData::BitVector {
            width: *width,
            width_value,
            signed,
            state,
        }))
    }

    /// Creates a user-defined type in the current namespace.
    pub fn make_extern_type(&mut self, name: &str) -> VastDataType {
        self.add_data_type(TypeData::Extern {
            package: None,
            name: name.to_owned(),
        })
    }

    /// Creates a package-qualified, user-defined type.
    pub fn make_extern_package_type(&mut self, package: &str, name: &str) -> VastDataType {
        self.add_data_type(TypeData::Extern {
            package: Some(package.to_owned()),
            name: name.to_owned(),
        })
    }

    /// Creates a Verilog `integer` type.
    pub fn make_integer_type(&mut self, signed: bool) -> VastDataType {
        self.add_data_type(TypeData::Integer {
            signed,
            kind: IntegerTypeKind::Integer,
        })
    }

    /// Creates a SystemVerilog `int` type.
    pub fn make_int_type(&mut self, signed: bool) -> VastDataType {
        self.add_data_type(TypeData::Integer {
            signed,
            kind: IntegerTypeKind::Int,
        })
    }

    /// Creates a dynamically sized SystemVerilog `string` type.
    pub fn make_string_type(&mut self) -> VastDataType {
        self.add_data_type(TypeData::String)
    }

    /// Creates a signed or unsigned, 64-bit, four-state SystemVerilog `time`
    /// type.
    pub fn make_time_type(&mut self, signed: bool) -> VastDataType {
        self.add_data_type(TypeData::Integer {
            signed,
            kind: IntegerTypeKind::Time,
        })
    }

    /// Creates a signed or unsigned 8-bit SystemVerilog `byte` type.
    pub fn make_byte_type(&mut self, signed: bool) -> VastDataType {
        self.add_data_type(TypeData::Integer {
            signed,
            kind: IntegerTypeKind::Byte,
        })
    }

    /// Creates a signed or unsigned 16-bit SystemVerilog `shortint` type.
    pub fn make_shortint_type(&mut self, signed: bool) -> VastDataType {
        self.add_data_type(TypeData::Integer {
            signed,
            kind: IntegerTypeKind::Shortint,
        })
    }

    /// Creates a signed or unsigned 64-bit SystemVerilog `longint` type.
    pub fn make_longint_type(&mut self, signed: bool) -> VastDataType {
        self.add_data_type(TypeData::Integer {
            signed,
            kind: IntegerTypeKind::Longint,
        })
    }

    /// Creates a packed multidimensional array around its element type.
    ///
    /// Dimensions are ordered from outermost to innermost. Nesting packed
    /// array types adds the new dimensions before the element's dimensions.
    /// Returns an error for invalid element types or empty or nonpositive
    /// dimensions.
    pub fn make_packed_array_type(
        &mut self,
        element: VastDataType,
        dimensions: &[i64],
    ) -> Result<VastDataType, VastError> {
        let element_index = self.check(element.0);
        let invalid_element = match &self.ast.data_types[element_index] {
            TypeData::UnpackedArray { .. } => {
                Some("packed arrays cannot contain unpacked array elements")
            }
            TypeData::Integer { .. } => {
                Some("packed arrays cannot contain integer or int elements")
            }
            TypeData::String => Some("packed arrays cannot contain string elements"),
            _ => None,
        };
        if let Some(message) = invalid_element {
            return Err(VastError(message.into()));
        }
        if dimensions.is_empty() {
            return Err(VastError(
                "packed array requires at least one dimension".into(),
            ));
        }
        if dimensions.iter().any(|dimension| *dimension <= 0) {
            return Err(VastError(
                "packed-array dimensions must be greater than zero".into(),
            ));
        }
        Ok(self.add_data_type(TypeData::PackedArray {
            element,
            dimensions: dimensions.to_vec(),
        }))
    }

    /// Creates an unpacked multidimensional array around its element type.
    ///
    /// Dimensions are ordered from outermost to innermost. Nesting unpacked
    /// array types adds the new dimensions before the element's dimensions.
    /// Returns an error for empty or nonpositive dimensions.
    pub fn make_unpacked_array_type(
        &mut self,
        element: VastDataType,
        dimensions: &[i64],
    ) -> Result<VastDataType, VastError> {
        self.check(element.0);
        if dimensions.is_empty() {
            return Err(VastError(
                "unpacked array requires at least one dimension".into(),
            ));
        }
        if dimensions.iter().any(|dimension| *dimension <= 0) {
            return Err(VastError(
                "unpacked-array dimensions must be greater than zero".into(),
            ));
        }
        Ok(self.add_data_type(TypeData::UnpackedArray {
            element,
            dimensions: dimensions.to_vec(),
        }))
    }

    /// Builds a nonnegative index, matching XLS's sized spelling above i32.
    fn make_index_literal(&mut self, index: i64) -> Expr {
        assert!(
            index >= 0,
            "bit and part-select indices must be nonnegative"
        );
        let text = if index <= i64::from(i32::MAX) {
            index.to_string()
        } else {
            Literal::parse(&format!("bits[64]:{index}"))
                .expect("a nonnegative i64 always fits in 64 bits")
                .format(LiteralFormat::Hex)
        };
        self.add_expression(ExprData::Literal {
            text,
            value: Some(index),
        })
    }

    /// Treats an owned expression as an indexable SystemVerilog expression.
    ///
    /// Named signals and arrays are portable index subjects. Common parsers
    /// reject selects applied directly to literals, concatenations, casts, or
    /// other compound expressions even when those expressions are enclosed in
    /// parentheses. Materialize such values first, or lower the selection to a
    /// shift and width cast when targeting multiple SystemVerilog tools.
    pub fn make_indexable_expression(&self, expression: &Expr) -> IndexableExpr {
        self.check(expression.0);
        IndexableExpr(expression.0)
    }

    /// Creates a fixed-index part select.
    pub fn make_slice(&mut self, subject: &IndexableExpr, hi: i64, lo: i64) -> Slice {
        self.check(subject.0);
        let hi = self.make_index_literal(hi);
        let lo = self.make_index_literal(lo);
        self.make_slice_expr(subject, &hi, &lo)
    }

    /// Creates an expression-indexed part select.
    pub fn make_slice_expr(&mut self, subject: &IndexableExpr, hi: &Expr, lo: &Expr) -> Slice {
        self.check(subject.0);
        self.check(hi.0);
        self.check(lo.0);
        Slice(
            self.add_expression(ExprData::Slice {
                subject: subject.to_expr(),
                hi: *hi,
                lo: *lo,
            })
            .0,
        )
    }

    /// Creates an ascending indexed part select: `subject[start +: width]`.
    ///
    /// The subject must be a packed vector, and the width must be positive.
    pub fn make_indexed_part_select(
        &mut self,
        subject: &IndexableExpr,
        start: &Expr,
        width: i64,
    ) -> Expr {
        self.check(subject.0);
        self.check(start.0);
        assert!(width > 0, "indexed part-select width must be positive");
        self.add_expression(ExprData::IndexedPartSelect {
            subject: subject.to_expr(),
            start: *start,
            width,
        })
    }

    /// Creates a constant-index bit select.
    pub fn make_index(&mut self, subject: &IndexableExpr, index: i64) -> Index {
        self.check(subject.0);
        let index = self.make_index_literal(index);
        self.make_index_expr(subject, &index)
    }

    /// Creates an expression-indexed bit select.
    pub fn make_index_expr(&mut self, subject: &IndexableExpr, index: &Expr) -> Index {
        self.check(subject.0);
        self.check(index.0);
        Index(
            self.add_expression(ExprData::Index {
                subject: subject.to_expr(),
                index: *index,
            })
            .0,
        )
    }

    /// Copies expression references after checking their owning file.
    fn checked_expressions(&self, expressions: &[&Expr]) -> Vec<Expr> {
        expressions
            .iter()
            .map(|expression| {
                self.check(expression.0);
                **expression
            })
            .collect()
    }

    /// Calls an ordinary or system function with value expressions, rejecting
    /// an empty name.
    pub fn make_function_call(
        &mut self,
        name: &str,
        arguments: &[&Expr],
    ) -> Result<Expr, VastError> {
        self.make_function_call_from_arguments(
            name,
            arguments
                .iter()
                .map(|argument| SvArgument::Value(**argument)),
        )
    }

    /// Calls a function with value or type arguments, such as `$bits(T)`.
    ///
    /// Type arguments are accepted only for system functions. The caller is
    /// responsible for choosing a system function and argument position that
    /// accept a type. Invalid names or type arguments return an error.
    pub fn make_function_call_with_arguments(
        &mut self,
        name: &str,
        arguments: &[SvArgument],
    ) -> Result<Expr, VastError> {
        self.make_function_call_from_arguments(name, arguments.iter().copied())
    }

    /// Creates a call while collecting its checked arguments exactly once.
    fn make_function_call_from_arguments(
        &mut self,
        name: &str,
        arguments: impl IntoIterator<Item = SvArgument>,
    ) -> Result<Expr, VastError> {
        if name.is_empty() {
            return Err(VastError("function call name must not be empty".into()));
        }
        let args = self.checked_arguments(arguments)?;
        if !name.starts_with('$')
            && args
                .iter()
                .any(|argument| matches!(argument, SvArgument::Type(_)))
        {
            return Err(VastError(
                "ordinary function calls do not accept type arguments".into(),
            ));
        }
        Ok(self.add_expression(ExprData::FunctionCall {
            name: name.to_owned(),
            args,
        }))
    }

    /// Checks and collects ordered arguments into their AST-owned storage.
    fn checked_arguments(
        &self,
        arguments: impl IntoIterator<Item = SvArgument>,
    ) -> Result<Vec<SvArgument>, VastError> {
        arguments
            .into_iter()
            .map(|argument| self.check_argument(argument).map(|()| argument))
            .collect()
    }

    /// Checks an argument's owning file and standalone type spelling.
    fn check_argument(&self, argument: SvArgument) -> Result<(), VastError> {
        match argument {
            SvArgument::Value(value) => {
                self.check(value.0);
                Ok(())
            }
            SvArgument::Type(data_type) => self.check_standalone_type(data_type),
        }
    }

    /// Creates an ordinary concatenation expression.
    pub fn make_concat(&mut self, expressions: &[&Expr]) -> Expr {
        let elements = self.checked_expressions(expressions);
        self.add_expression(ExprData::Concat {
            replication: None,
            elements,
        })
    }

    /// Creates an expression-controlled replicated concatenation.
    pub fn make_replicated_concat(&mut self, replication: &Expr, elements: &[&Expr]) -> Expr {
        self.check(replication.0);
        let elements = self.checked_expressions(elements);
        self.add_expression(ExprData::Concat {
            replication: Some(*replication),
            elements,
        })
    }

    /// Creates a constant-count replicated concatenation.
    pub fn make_replicated_concat_i64(&mut self, replication: i64, elements: &[&Expr]) -> Expr {
        let count = self.add_expression(ExprData::Literal {
            text: replication.to_string(),
            value: Some(replication),
        });
        self.make_replicated_concat(&count, elements)
    }

    /// Creates a SystemVerilog array assignment pattern.
    pub fn make_array_assignment_pattern(&mut self, elements: &[&Expr]) -> Expr {
        let elements = self.checked_expressions(elements);
        self.add_expression(ExprData::ArrayAssignmentPattern(elements))
    }

    /// Creates an ordered SystemVerilog struct assignment pattern with named
    /// members.
    ///
    /// Keys must be simple identifiers or the `default` key. Returns an error
    /// for an empty pattern or invalid or duplicate keys.
    pub fn make_struct_assignment_pattern(
        &mut self,
        members: &[(&str, &Expr)],
    ) -> Result<Expr, VastError> {
        if members.is_empty() {
            return Err(VastError(
                "struct assignment pattern must not be empty".into(),
            ));
        }
        let mut names = HashSet::with_capacity(members.len());
        let members = members
            .iter()
            .map(|(name, value)| {
                if *name != "default" && !is_system_verilog_simple_identifier(name) {
                    return Err(VastError(format!(
                        "invalid struct assignment member identifier: {name}"
                    )));
                }
                if !names.insert(*name) {
                    return Err(VastError(format!("duplicate struct member {name}")));
                }
                self.check(value.0);
                Ok(((*name).to_owned(), **value))
            })
            .collect::<Result<Vec<_>, VastError>>()?;
        Ok(self.add_expression(ExprData::StructAssignmentPattern(members)))
    }

    /// Adds a unary expression to the file-owned arena.
    fn make_unary(&mut self, op: UnaryOp, expression: &Expr) -> Expr {
        self.check(expression.0);
        self.add_expression(ExprData::Unary {
            op,
            arg: *expression,
        })
    }

    /// Creates a bitwise complement.
    pub fn make_not(&mut self, expression: &Expr) -> Expr {
        self.make_unary(UnaryOp::BitwiseNot, expression)
    }

    /// Creates an arithmetic negation.
    pub fn make_negate(&mut self, expression: &Expr) -> Expr {
        self.make_unary(UnaryOp::Negate, expression)
    }

    /// Creates a logical negation.
    pub fn make_logical_not(&mut self, expression: &Expr) -> Expr {
        self.make_unary(UnaryOp::LogicalNot, expression)
    }

    /// Creates a reduction AND.
    pub fn make_and_reduce(&mut self, expression: &Expr) -> Expr {
        self.make_unary(UnaryOp::AndReduce, expression)
    }

    /// Creates a reduction OR.
    pub fn make_or_reduce(&mut self, expression: &Expr) -> Expr {
        self.make_unary(UnaryOp::OrReduce, expression)
    }

    /// Creates a reduction XOR.
    pub fn make_xor_reduce(&mut self, expression: &Expr) -> Expr {
        self.make_unary(UnaryOp::XorReduce, expression)
    }

    /// Creates a bitwise complement.
    pub fn make_bitwise_not(&mut self, expression: &Expr) -> Expr {
        self.make_unary(UnaryOp::BitwiseNot, expression)
    }

    /// Adds a binary expression to the file-owned arena.
    fn make_binary(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) -> Expr {
        self.check(lhs.0);
        self.check(rhs.0);
        self.add_expression(ExprData::Binary {
            op,
            lhs: *lhs,
            rhs: *rhs,
        })
    }

    /// Creates an addition expression.
    pub fn make_add(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Add, lhs, rhs)
    }

    /// Creates a subtraction expression.
    pub fn make_sub(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Sub, lhs, rhs)
    }

    /// Creates a multiplication expression.
    pub fn make_mul(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Mul, lhs, rhs)
    }

    /// Creates a division expression.
    pub fn make_div(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Div, lhs, rhs)
    }

    /// Creates a modulo expression.
    pub fn make_mod(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Mod, lhs, rhs)
    }

    /// Creates an exponentiation expression.
    pub fn make_power(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Power, lhs, rhs)
    }

    /// Creates a bitwise AND expression.
    pub fn make_bitwise_and(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::BitwiseAnd, lhs, rhs)
    }

    /// Creates a bitwise OR expression.
    pub fn make_bitwise_or(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::BitwiseOr, lhs, rhs)
    }

    /// Creates a bitwise XOR expression.
    pub fn make_bitwise_xor(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::BitwiseXor, lhs, rhs)
    }

    /// Creates a logical left shift.
    pub fn make_shll(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Shll, lhs, rhs)
    }

    /// Creates an arithmetic right shift.
    pub fn make_shra(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Shra, lhs, rhs)
    }

    /// Creates a logical right shift.
    pub fn make_shrl(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Shrl, lhs, rhs)
    }

    /// Creates an inequality comparison.
    pub fn make_ne(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Ne, lhs, rhs)
    }

    /// Creates a four-valued inequality comparison.
    pub fn make_case_ne(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::CaseNe, lhs, rhs)
    }

    /// Creates an equality comparison.
    pub fn make_eq(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Eq, lhs, rhs)
    }

    /// Creates a four-valued equality comparison.
    pub fn make_case_eq(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::CaseEq, lhs, rhs)
    }

    /// Creates a greater-than-or-equal comparison.
    pub fn make_ge(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Ge, lhs, rhs)
    }

    /// Creates a greater-than comparison.
    pub fn make_gt(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Gt, lhs, rhs)
    }

    /// Creates a less-than-or-equal comparison.
    pub fn make_le(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Le, lhs, rhs)
    }

    /// Creates a less-than comparison.
    pub fn make_lt(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::Lt, lhs, rhs)
    }

    /// Creates a logical conjunction.
    pub fn make_logical_and(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::LogicalAnd, lhs, rhs)
    }

    /// Creates a logical disjunction.
    pub fn make_logical_or(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::LogicalOr, lhs, rhs)
    }

    /// Creates an X-aware inequality comparison.
    pub fn make_ne_x(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::NeX, lhs, rhs)
    }

    /// Creates an X-aware equality comparison.
    pub fn make_eq_x(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(BinaryOp::EqX, lhs, rhs)
    }

    /// Creates a ternary conditional expression.
    pub fn make_ternary(&mut self, condition: &Expr, consequent: &Expr, alternate: &Expr) -> Expr {
        self.check(condition.0);
        self.check(consequent.0);
        self.check(alternate.0);
        self.add_expression(ExprData::Ternary {
            condition: *condition,
            consequent: *consequent,
            alternate: *alternate,
        })
    }

    /// Creates a SystemVerilog expression-width cast.
    pub fn make_width_cast(&mut self, width: &Expr, value: &Expr) -> Expr {
        self.check(width.0);
        self.check(value.0);
        self.add_expression(ExprData::WidthCast {
            width: *width,
            value: *value,
        })
    }

    /// Creates a SystemVerilog data-type cast, returning an error for anonymous
    /// array targets. Cast to an existing typedef through a named external
    /// type.
    pub fn make_type_cast(
        &mut self,
        data_type: &VastDataType,
        value: &Expr,
    ) -> Result<Expr, VastError> {
        let type_index = self.check(data_type.0);
        self.check(value.0);
        match &self.ast.data_types[type_index] {
            TypeData::UnpackedArray { .. } => {
                return Err(VastError(
                    "unpacked array types cannot be used in type casts".into(),
                ));
            }
            TypeData::PackedArray { .. } => {
                return Err(VastError(
                    "packed array types cannot be used in type casts; use a named typedef instead"
                        .into(),
                ));
            }
            _ => {}
        }
        Ok(self.add_expression(ExprData::TypeCast {
            data_type: *data_type,
            value: *value,
        }))
    }

    /// Creates a positive-edge sensitivity expression.
    pub fn make_pos_edge(&mut self, expression: &Expr) -> Expr {
        self.check(expression.0);
        self.add_expression(ExprData::PosEdge(*expression))
    }

    /// Creates a negative-edge sensitivity expression.
    pub fn make_neg_edge(&mut self, expression: &Expr) -> Expr {
        self.check(expression.0);
        self.add_expression(ExprData::NegEdge(*expression))
    }

    /// Creates the unsized SystemVerilog all-ones literal.
    pub fn make_unsized_one_literal(&mut self) -> Expr {
        self.add_expression(ExprData::Literal {
            text: "'1".into(),
            value: Some(1),
        })
    }

    /// Creates the unsized SystemVerilog all-zeroes literal.
    pub fn make_unsized_zero_literal(&mut self) -> Expr {
        self.add_expression(ExprData::Literal {
            text: "'0".into(),
            value: Some(0),
        })
    }

    /// Creates the unsized SystemVerilog unknown-value literal.
    pub fn make_unsized_x_literal(&mut self) -> Expr {
        self.add_expression(ExprData::Literal {
            text: "'X".into(),
            value: None,
        })
    }

    /// Creates the context-sized SystemVerilog high-impedance literal.
    pub fn make_unsized_z_literal(&mut self) -> Expr {
        self.add_expression(ExprData::Literal {
            text: "'z".into(),
            value: None,
        })
    }

    /// Creates a reusable declaration describing its name, kind, and data type.
    pub fn make_def(&mut self, name: &str, kind: DataKind, data_type: &VastDataType) -> Def {
        self.check(data_type.0);
        let handle = Def(self.id(self.ast.defs.len()));
        self.ast.defs.push(DefData {
            name: name.to_owned(),
            kind,
            data_type: *data_type,
        });
        handle
    }

    /// Creates a reusable blank-line artifact.
    pub fn make_blank_line(&mut self) -> BlankLine {
        let handle = BlankLine(self.id(self.ast.artifacts.len()));
        self.ast.artifacts.push(Artifact::BlankLine);
        handle
    }

    /// Creates a reusable single- or multiline Verilog comment.
    pub fn make_comment(&mut self, text: &str) -> Comment {
        let handle = Comment(self.id(self.ast.artifacts.len()));
        self.ast.artifacts.push(Artifact::Comment(text.to_owned()));
        handle
    }

    /// Creates a reusable inline Verilog statement.
    pub fn make_inline_verilog_statement(&mut self, text: &str) -> InlineVerilogStatement {
        let handle = InlineVerilogStatement(self.id(self.ast.artifacts.len()));
        self.ast.artifacts.push(Artifact::Inline(text.to_owned()));
        handle
    }

    /// Creates a preprocessor macro reference without explicit arguments.
    pub fn make_macro_ref(&mut self, name: &str) -> MacroRef {
        MacroRef(
            self.add_expression(ExprData::Macro {
                name: name.to_owned(),
                args: None,
            })
            .0,
        )
    }

    /// Creates a preprocessor macro reference with explicit value arguments.
    pub fn make_macro_ref_with_args(&mut self, name: &str, arguments: &[&Expr]) -> MacroRef {
        self.make_macro_ref_from_arguments(
            name,
            arguments
                .iter()
                .map(|argument| SvArgument::Value(**argument)),
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Creates a preprocessor macro reference, returning an error for
    /// unsupported type arguments.
    pub fn make_macro_ref_with_arguments(
        &mut self,
        name: &str,
        arguments: &[SvArgument],
    ) -> Result<MacroRef, VastError> {
        self.make_macro_ref_from_arguments(name, arguments.iter().copied())
    }

    /// Creates a macro reference while collecting checked arguments once.
    fn make_macro_ref_from_arguments(
        &mut self,
        name: &str,
        arguments: impl IntoIterator<Item = SvArgument>,
    ) -> Result<MacroRef, VastError> {
        let args = self.checked_arguments(arguments)?;
        Ok(MacroRef(
            self.add_expression(ExprData::Macro {
                name: name.to_owned(),
                args: Some(args),
            })
            .0,
        ))
    }

    /// Wraps a preprocessor macro reference as a module or generate statement.
    pub fn make_macro_statement(
        &mut self,
        macro_ref: &MacroRef,
        semicolon: bool,
    ) -> MacroStatement {
        self.check(macro_ref.0);
        let handle = MacroStatement(self.id(self.ast.artifacts.len()));
        self.ast.artifacts.push(Artifact::MacroStatement {
            expr: macro_ref.to_expr(),
            semicolon,
        });
        handle
    }

    /// Creates a reusable continuous assignment.
    pub fn make_continuous_assignment(&mut self, lhs: &Expr, rhs: &Expr) -> ContinuousAssignment {
        self.check(lhs.0);
        self.check(rhs.0);
        let handle = ContinuousAssignment(self.id(self.ast.assignments.len()));
        self.ast.assignments.push(AssignmentData {
            lhs: *lhs,
            rhs: *rhs,
            nonblocking: false,
        });
        handle
    }

    /// Creates a reusable procedural nonblocking assignment.
    pub fn make_nonblocking_assignment(&mut self, lhs: &Expr, rhs: &Expr) -> VastStatement {
        self.make_assignment(lhs, rhs, true)
    }

    /// Creates a reusable procedural blocking assignment.
    pub fn make_blocking_assignment(&mut self, lhs: &Expr, rhs: &Expr) -> VastStatement {
        self.make_assignment(lhs, rhs, false)
    }

    /// Creates a stable assignment handle usable by statement-building methods.
    fn make_assignment(&mut self, lhs: &Expr, rhs: &Expr, nonblocking: bool) -> VastStatement {
        self.check(lhs.0);
        self.check(rhs.0);
        let handle = VastStatement(self.id(self.ast.assignments.len()));
        self.ast.assignments.push(AssignmentData {
            lhs: *lhs,
            rhs: *rhs,
            nonblocking,
        });
        handle
    }

    /// Constructs a named module instantiation with value parameters and ports,
    /// returning an error for mismatched argument lists.
    pub fn make_instantiation(
        &mut self,
        module_name: &str,
        instance_name: &str,
        parameter_names: &[&str],
        parameter_expressions: &[&Expr],
        connection_names: &[&str],
        connection_expressions: &[Option<&Expr>],
    ) -> Result<Instantiation, VastError> {
        self.make_instantiation_from_arguments(
            module_name,
            instance_name,
            parameter_names,
            parameter_expressions
                .iter()
                .map(|value| SvArgument::Value(**value)),
            connection_names,
            connection_expressions,
        )
    }

    /// Constructs a module instantiation, returning an error for mismatched
    /// argument lists or unsupported types.
    pub fn make_instantiation_with_arguments(
        &mut self,
        module_name: &str,
        instance_name: &str,
        parameter_names: &[&str],
        parameter_arguments: &[SvArgument],
        connection_names: &[&str],
        connection_expressions: &[Option<&Expr>],
    ) -> Result<Instantiation, VastError> {
        self.make_instantiation_from_arguments(
            module_name,
            instance_name,
            parameter_names,
            parameter_arguments.iter().copied(),
            connection_names,
            connection_expressions,
        )
    }

    /// Creates an instantiation while checking and storing each parameter once.
    fn make_instantiation_from_arguments(
        &mut self,
        module_name: &str,
        instance_name: &str,
        parameter_names: &[&str],
        parameter_arguments: impl ExactSizeIterator<Item = SvArgument>,
        connection_names: &[&str],
        connection_expressions: &[Option<&Expr>],
    ) -> Result<Instantiation, VastError> {
        if parameter_names.len() != parameter_arguments.len() {
            return Err(VastError(
                "module-instantiation parameter names and values must have equal lengths".into(),
            ));
        }
        if connection_names.len() != connection_expressions.len() {
            return Err(VastError(
                "module-instantiation port names and values must have equal lengths".into(),
            ));
        }
        let parameters = parameter_names
            .iter()
            .zip(parameter_arguments)
            .map(|(name, argument)| {
                self.check_argument(argument)?;
                Ok(((*name).to_owned(), argument))
            })
            .collect::<Result<Vec<_>, VastError>>()?;
        let ports = connection_names
            .iter()
            .zip(connection_expressions)
            .map(|(name, expression)| {
                let expression = expression.map(|expression| {
                    self.check(expression.0);
                    *expression
                });
                ((*name).to_owned(), expression)
            })
            .collect();
        let handle = Instantiation(self.id(self.ast.instantiations.len()));
        self.ast.instantiations.push(InstantiationData {
            module_name: module_name.to_owned(),
            instance_name: instance_name.to_owned(),
            parameters,
            ports,
        });
        Ok(handle)
    }

    /// Adds an existing instantiation to a module, rejecting duplicate names.
    pub fn add_member_instantiation(
        &mut self,
        module: VastModule,
        instantiation: Instantiation,
    ) -> Result<(), VastError> {
        let name = self.ast.instantiations[self.check(instantiation.0)]
            .instance_name
            .clone();
        let module_index = self.check(module.0);
        self.check_unique_module_name(module, &name, "instance")?;
        self.push_member(module, MemberData::Instantiation(instantiation));
        self.ast.modules[module_index].defined_names.insert(name);
        Ok(())
    }

    /// Adds an existing continuous assignment to a module.
    pub fn add_member_continuous_assignment(
        &mut self,
        module: VastModule,
        assignment: ContinuousAssignment,
    ) {
        let data = self.ast.assignments[self.check(assignment.0)];
        assert!(
            !data.nonblocking,
            "continuous assignments cannot use nonblocking semantics"
        );
        self.push_member(
            module,
            MemberData::ContinuousAssignment {
                lhs: data.lhs,
                rhs: data.rhs,
            },
        );
    }

    /// Adds an existing blank line to a module.
    pub fn add_member_blank_line(&mut self, module: VastModule, blank: BlankLine) {
        assert!(matches!(
            self.ast.artifacts[self.check(blank.0)],
            Artifact::BlankLine
        ));
        self.push_member(module, MemberData::BlankLine);
    }

    /// Adds an existing comment to a module.
    pub fn add_member_comment(&mut self, module: VastModule, comment: Comment) {
        let text = match &self.ast.artifacts[self.check(comment.0)] {
            Artifact::Comment(text) => text.clone(),
            _ => panic!("comment handle does not identify a comment artifact"),
        };
        self.push_member(module, MemberData::Comment(text));
    }

    /// Adds an existing inline Verilog statement to a module.
    pub fn add_member_inline_statement(
        &mut self,
        module: VastModule,
        statement: InlineVerilogStatement,
    ) {
        let text = match &self.ast.artifacts[self.check(statement.0)] {
            Artifact::Inline(text) => text.clone(),
            _ => panic!("inline-statement handle does not identify inline text"),
        };
        self.push_member(module, MemberData::Inline(text));
    }

    /// Adds an existing macro statement to a module.
    pub fn add_member_macro_statement(&mut self, module: VastModule, statement: MacroStatement) {
        let (expr, semicolon) = self.macro_statement_data(statement);
        self.push_member(module, MemberData::Macro { expr, semicolon });
    }

    /// Reads the expression and terminator attached to a macro-statement
    /// handle.
    fn macro_statement_data(&self, statement: MacroStatement) -> (Expr, bool) {
        match self.ast.artifacts[self.check(statement.0)] {
            Artifact::MacroStatement { expr, semicolon } => (expr, semicolon),
            _ => panic!("macro-statement handle does not identify a macro statement"),
        }
    }

    /// Creates the shared representation of a parameter or local parameter.
    fn parameter_member(
        &mut self,
        name: &str,
        rhs: &Expr,
        local: bool,
        definition: Option<Def>,
        int_kind: bool,
    ) -> (LogicRef, MemberData) {
        self.check(rhs.0);
        let data_type =
            definition.map(|definition| self.ast.defs[self.check(definition.0)].data_type);
        let reference = self.make_named_ref(name, data_type);
        (
            reference,
            MemberData::Parameter {
                local,
                def: definition,
                name: name.to_owned(),
                rhs: *rhs,
                int_kind,
            },
        )
    }

    /// Appends a module-body type parameter and returns a reference to its
    /// type.
    ///
    /// SystemVerilog treats body parameters as local when the module has a
    /// parameter-port list, regardless of the declaration keyword.
    /// The returned type is available to later body declarations, not to the
    /// module's preceding port list; use [`Self::add_type_parameter_port`] for
    /// a port type. Anonymous unpacked array defaults return an error and
    /// require a named typedef. Duplicate names also return an error.
    pub fn add_type_parameter(
        &mut self,
        module: VastModule,
        name: &str,
        kind: ParameterKind,
        default: &VastDataType,
    ) -> Result<VastDataType, VastError> {
        let module_index = self.check(module.0);
        self.check_standalone_type(*default)?;
        self.check_unique_module_name(module, name, parameter_kind_name(kind, true))?;
        let reference = self.make_extern_type(name);
        self.push_member(
            module,
            MemberData::TypeParameter {
                kind,
                name: name.to_owned(),
                default: *default,
            },
        );
        self.ast.modules[module_index]
            .defined_names
            .insert(name.to_owned());
        Ok(reference)
    }

    /// Appends a legal `timeunit` declaration to the module body.
    /// Returns an error if it follows other scope items or conflicts with a
    /// local time scale.
    pub fn add_timeunit(
        &mut self,
        module: VastModule,
        value: TimeScaleValue,
    ) -> Result<(), VastError> {
        self.add_timescale_declaration(module, MemberData::TimeUnit(value))
    }

    /// Appends a legal `timeprecision` declaration to the module body.
    /// Returns an error if it follows other scope items or is coarser than the
    /// local time unit.
    pub fn add_timeprecision(
        &mut self,
        module: VastModule,
        value: TimeScaleValue,
    ) -> Result<(), VastError> {
        self.add_timescale_declaration(module, MemberData::TimePrecision(value))
    }

    /// Validates local time-scale declarations before appending them in source
    /// order.
    fn add_timescale_declaration(
        &mut self,
        module: VastModule,
        declaration: MemberData,
    ) -> Result<(), VastError> {
        let module_index = self.check(module.0);
        let members = &self.ast.modules[module_index].members;
        for member in members {
            match (member, &declaration) {
                (MemberData::TimeUnit(existing), MemberData::TimeUnit(value))
                | (MemberData::TimePrecision(existing), MemberData::TimePrecision(value)) => {
                    if existing != value {
                        return Err(VastError("conflicting time-scale declarations".into()));
                    }
                }
                (MemberData::TimeUnit(unit), MemberData::TimePrecision(precision))
                | (MemberData::TimePrecision(precision), MemberData::TimeUnit(unit)) => {
                    if precision.femtosecond_exponent() > unit.femtosecond_exponent() {
                        return Err(VastError(
                            "time precision cannot be coarser than time unit".into(),
                        ));
                    }
                }
                (MemberData::Comment(_) | MemberData::BlankLine, _) => {
                    // Comments and whitespace do not declare scope items.
                }
                _ => {
                    return Err(VastError(
                        "time-scale declarations must precede other module members".into(),
                    ));
                }
            }
        }
        self.push_member(module, declaration);
        Ok(())
    }

    /// Validates and appends a module-body value parameter declaration.
    fn add_module_parameter(
        &mut self,
        module: VastModule,
        name: &str,
        rhs: &Expr,
        local: bool,
        definition: Option<Def>,
        int_kind: bool,
    ) -> Result<LogicRef, VastError> {
        let module_index = self.check(module.0);
        let kind = if local { "localparam" } else { "parameter" };
        self.check_unique_module_name(module, name, kind)?;
        let (reference, member) = self.parameter_member(name, rhs, local, definition, int_kind);
        self.push_member(module, member);
        self.ast.modules[module_index]
            .defined_names
            .insert(name.to_owned());
        Ok(reference)
    }

    /// Adds an untyped module parameter declaration, rejecting duplicate names.
    pub fn add_parameter(
        &mut self,
        module: VastModule,
        name: &str,
        rhs: &Expr,
    ) -> Result<ParameterRef, VastError> {
        self.add_module_parameter(module, name, rhs, false, None, false)
            .map(|reference| ParameterRef(reference.0))
    }

    /// Adds a typed module parameter declaration, rejecting duplicate names.
    pub fn add_parameter_with_def(
        &mut self,
        module: VastModule,
        definition: &Def,
        rhs: &Expr,
    ) -> Result<ParameterRef, VastError> {
        let name = self.ast.defs[self.check(definition.0)].name.clone();
        self.add_module_parameter(module, &name, rhs, false, Some(*definition), false)
            .map(|reference| ParameterRef(reference.0))
    }

    /// Adds an untyped module-local parameter, rejecting duplicate names.
    pub fn add_localparam(
        &mut self,
        module: VastModule,
        name: &str,
        rhs: &Expr,
    ) -> Result<LocalparamRef, VastError> {
        self.add_module_parameter(module, name, rhs, true, None, false)
            .map(|reference| LocalparamRef(reference.0))
    }

    /// Adds a typed module-local parameter, rejecting duplicate names.
    pub fn add_typed_localparam(
        &mut self,
        module: VastModule,
        definition: &Def,
        rhs: &Expr,
    ) -> Result<LocalparamRef, VastError> {
        let name = self.ast.defs[self.check(definition.0)].name.clone();
        self.add_module_parameter(module, &name, rhs, true, Some(*definition), false)
            .map(|reference| LocalparamRef(reference.0))
    }

    /// Adds an `int`-typed module-local parameter, rejecting duplicate names.
    pub fn add_int_localparam(
        &mut self,
        module: VastModule,
        name: &str,
        rhs: &Expr,
    ) -> Result<LocalparamRef, VastError> {
        self.add_module_parameter(module, name, rhs, true, None, true)
            .map(|reference| LocalparamRef(reference.0))
    }

    /// Allocates an always block and its initially empty procedural block.
    fn make_always(&mut self, kind: AlwaysKind, sensitivity_list: &[&Expr]) -> VastAlwaysBase {
        let sensitivity_list = self.checked_expressions(sensitivity_list);
        let block = self.make_statement_block();
        let handle = VastAlwaysBase(self.id(self.ast.always.len()));
        self.ast.always.push(AlwaysData {
            kind,
            sensitivity_list,
            block,
        });
        handle
    }

    /// Rejects sensitivity expressions that cannot represent a Verilog event.
    fn validate_sensitivity_list(
        &self,
        kind: AlwaysKind,
        sensitivity_list: &[&Expr],
    ) -> Result<(), VastError> {
        for (index, expression) in sensitivity_list.iter().enumerate() {
            if !matches!(
                self.ast.expressions[self.check(expression.0)],
                ExprData::PosEdge(_) | ExprData::NegEdge(_) | ExprData::Name { .. }
            ) {
                let block_kind = match kind {
                    AlwaysKind::AlwaysFF => "always_ff",
                    AlwaysKind::AlwaysAt => "always @",
                    AlwaysKind::AlwaysComb => "always_comb",
                };
                return Err(VastError(format!(
                    "Unsupported expression type passed to sensitivity list for {block_kind} \
                     at index {index}. Only posedge, negedge, or logic-reference expressions \
                     are supported."
                )));
            }
        }
        Ok(())
    }

    /// Adds an `always_ff` procedural block to a module.
    pub fn add_always_ff(
        &mut self,
        module: VastModule,
        sensitivity_list: &[&Expr],
    ) -> Result<VastAlwaysBase, VastError> {
        self.validate_sensitivity_list(AlwaysKind::AlwaysFF, sensitivity_list)?;
        let block = self.make_always(AlwaysKind::AlwaysFF, sensitivity_list);
        self.push_member(module, MemberData::Always(block));
        Ok(block)
    }

    /// Adds an ordinary `always @ (...)` procedural block to a module.
    pub fn add_always_at(
        &mut self,
        module: VastModule,
        sensitivity_list: &[&Expr],
    ) -> Result<VastAlwaysBase, VastError> {
        self.validate_sensitivity_list(AlwaysKind::AlwaysAt, sensitivity_list)?;
        let block = self.make_always(AlwaysKind::AlwaysAt, sensitivity_list);
        self.push_member(module, MemberData::Always(block));
        Ok(block)
    }

    /// Adds an `always_comb` procedural block to a module.
    pub fn add_always_comb(&mut self, module: VastModule) -> Result<VastAlwaysBase, VastError> {
        let block = self.make_always(AlwaysKind::AlwaysComb, &[]);
        self.push_member(module, MemberData::Always(block));
        Ok(block)
    }

    /// Returns the procedural statement block owned by an always block.
    pub fn statement_block(&self, always: VastAlwaysBase) -> VastStatementBlock {
        self.ast.always[self.check(always.0)].block
    }

    /// Adds a blocking assignment to a procedural statement block.
    pub fn block_add_blocking_assignment(
        &mut self,
        block: VastStatementBlock,
        lhs: &Expr,
        rhs: &Expr,
    ) -> VastStatement {
        let statement = self.make_assignment(lhs, rhs, false);
        self.push_statement(
            block,
            StatementData::BlockingAssignment {
                lhs: *lhs,
                rhs: *rhs,
            },
        );
        statement
    }

    /// Adds a continuous assignment to a procedural or conditional block.
    pub fn block_add_continuous_assignment(
        &mut self,
        block: VastStatementBlock,
        lhs: &Expr,
        rhs: &Expr,
    ) -> VastStatement {
        let statement = self.make_assignment(lhs, rhs, false);
        self.push_statement(
            block,
            StatementData::ContinuousAssignment {
                lhs: *lhs,
                rhs: *rhs,
            },
        );
        statement
    }

    /// Adds a nonblocking assignment to a procedural statement block.
    pub fn block_add_nonblocking_assignment(
        &mut self,
        block: VastStatementBlock,
        lhs: &Expr,
        rhs: &Expr,
    ) -> VastStatement {
        let statement = self.make_assignment(lhs, rhs, true);
        self.push_statement(
            block,
            StatementData::NonblockingAssignment {
                lhs: *lhs,
                rhs: *rhs,
            },
        );
        statement
    }

    /// Adds a comment to a procedural statement block.
    pub fn block_add_comment_text(
        &mut self,
        block: VastStatementBlock,
        text: &str,
    ) -> VastStatement {
        self.push_statement(block, StatementData::Comment(text.to_owned()));
        VastStatement(block.0)
    }

    /// Adds a blank line to a procedural statement block.
    pub fn block_add_blank_line(&mut self, block: VastStatementBlock) -> VastStatement {
        self.push_statement(block, StatementData::BlankLine);
        VastStatement(block.0)
    }

    /// Adds inline Verilog text to a procedural statement block.
    pub fn block_add_inline_text(
        &mut self,
        block: VastStatementBlock,
        text: &str,
    ) -> VastStatement {
        self.push_statement(block, StatementData::Inline(text.to_owned()));
        VastStatement(block.0)
    }

    /// Allocates a conditional and its initially empty consequent block.
    fn make_conditional(&mut self, condition: &Expr) -> Conditional {
        self.check(condition.0);
        let consequent = self.make_statement_block();
        let handle = Conditional(self.id(self.ast.conditionals.len()));
        self.ast.conditionals.push(ConditionalData {
            condition: *condition,
            consequent,
            alternates: Vec::new(),
        });
        handle
    }

    /// Adds a module-level conditional member.
    pub fn add_conditional(&mut self, module: VastModule, condition: &Expr) -> Conditional {
        let conditional = self.make_conditional(condition);
        self.push_member(module, MemberData::Conditional(conditional));
        conditional
    }

    /// Adds an if/else conditional to a statement block.
    pub fn block_add_cond(&mut self, block: VastStatementBlock, condition: &Expr) -> Conditional {
        let conditional = self.make_conditional(condition);
        self.push_statement(block, StatementData::Conditional(conditional));
        conditional
    }

    /// Returns the consequent statement block of an existing conditional.
    pub fn conditional_then_block(&self, conditional: Conditional) -> VastStatementBlock {
        self.ast.conditionals[self.check(conditional.0)].consequent
    }

    /// Appends an else-if arm and returns its newly allocated statement block.
    pub fn conditional_add_else_if(
        &mut self,
        conditional: Conditional,
        condition: &Expr,
    ) -> VastStatementBlock {
        let conditional_index = self.check(conditional.0);
        self.check(condition.0);
        assert!(
            !matches!(
                self.ast.conditionals[conditional_index].alternates.last(),
                Some((None, _))
            ),
            "cannot add an else-if arm after an unconditional else"
        );
        let block = self.make_statement_block();
        self.ast.conditionals[conditional_index]
            .alternates
            .push((Some(*condition), block));
        block
    }

    /// Appends an else arm and returns its newly allocated statement block.
    pub fn conditional_add_else(&mut self, conditional: Conditional) -> VastStatementBlock {
        let conditional_index = self.check(conditional.0);
        assert!(
            !matches!(
                self.ast.conditionals[conditional_index].alternates.last(),
                Some((None, _))
            ),
            "cannot add another arm after an unconditional else"
        );
        let block = self.make_statement_block();
        self.ast.conditionals[conditional_index]
            .alternates
            .push((None, block));
        block
    }

    /// Adds a case statement to a procedural block.
    pub fn block_add_case(&mut self, block: VastStatementBlock, selector: &Expr) -> CaseStatement {
        self.add_case(block, selector, false)
    }

    /// Adds a `casez` statement, treating `z` and `?` bits as wildcards.
    pub fn block_add_casez(&mut self, block: VastStatementBlock, selector: &Expr) -> CaseStatement {
        self.add_case(block, selector, true)
    }

    fn add_case(
        &mut self,
        block: VastStatementBlock,
        selector: &Expr,
        wildcard_z: bool,
    ) -> CaseStatement {
        self.check(selector.0);
        let case = CaseStatement(self.id(self.ast.cases.len()));
        self.ast.cases.push(CaseData {
            wildcard_z,
            unique: false,
            selector: *selector,
            arms: Vec::new(),
        });
        self.push_statement(block, StatementData::Case(case));
        case
    }

    /// Adds a matching case arm and returns its procedural statement block.
    pub fn case_add_item(&mut self, case: CaseStatement, pattern: &Expr) -> VastStatementBlock {
        let case_index = self.check(case.0);
        self.check(pattern.0);
        let block = self.make_statement_block();
        self.ast.cases[case_index]
            .arms
            .push((Some(*pattern), block));
        block
    }

    /// Requests SystemVerilog's uniqueness check for disjoint case patterns.
    pub fn case_set_unique(&mut self, case: CaseStatement, unique: bool) -> Result<(), VastError> {
        let index = self.check(case.0);
        if unique && self.file_type != VastFileType::SystemVerilog {
            return Err(VastError("unique case requires SystemVerilog".to_owned()));
        }
        self.ast.cases[index].unique = unique;
        Ok(())
    }

    /// Adds a default case arm and returns its procedural statement block.
    pub fn case_add_default(&mut self, case: CaseStatement) -> VastStatementBlock {
        let case_index = self.check(case.0);
        let block = self.make_statement_block();
        self.ast.cases[case_index].arms.push((None, block));
        block
    }

    /// Allocates a generate loop and the corresponding genvar reference.
    fn make_generate_loop(
        &mut self,
        name: &str,
        start: &Expr,
        end: &Expr,
        label: Option<&str>,
    ) -> GenerateLoop {
        self.check(start.0);
        self.check(end.0);
        let genvar = self.make_named_ref(name, None);
        let handle = GenerateLoop(self.id(self.ast.generates.len()));
        self.ast.generates.push(GenerateData {
            genvar,
            name: name.to_owned(),
            start: *start,
            end: *end,
            label: label.map(str::to_owned),
            members: Vec::new(),
        });
        handle
    }

    /// Adds a generate loop to a module, rejecting a duplicate block label.
    pub fn add_generate_loop(
        &mut self,
        module: VastModule,
        name: &str,
        start: &Expr,
        end: &Expr,
        label: Option<&str>,
    ) -> Result<GenerateLoop, VastError> {
        let module_index = self.check(module.0);
        self.check(start.0);
        self.check(end.0);
        if let Some(label) = label {
            self.check_unique_module_name(module, label, "generate block")?;
        }
        let generate = self.make_generate_loop(name, start, end, label);
        self.push_member(module, MemberData::Generate(generate));
        if let Some(label) = label {
            self.ast.modules[module_index]
                .defined_names
                .insert(label.to_owned());
        }
        Ok(generate)
    }

    /// Adds a nested generate loop inside an existing generate loop.
    pub fn generate_add_generate_loop(
        &mut self,
        parent: GenerateLoop,
        name: &str,
        start: &Expr,
        end: &Expr,
        label: Option<&str>,
    ) -> GenerateLoop {
        let generate = self.make_generate_loop(name, start, end, label);
        self.push_generate_member(parent, MemberData::Generate(generate));
        generate
    }

    /// Returns the genvar logic reference belonging to a generate loop.
    pub fn generate_genvar(&self, generate: GenerateLoop) -> LogicRef {
        self.ast.generates[self.check(generate.0)].genvar
    }

    /// Adds an `always_comb` block inside a generate loop.
    pub fn generate_add_always_comb(
        &mut self,
        generate: GenerateLoop,
    ) -> Result<VastAlwaysBase, VastError> {
        let block = self.make_always(AlwaysKind::AlwaysComb, &[]);
        self.push_generate_member(generate, MemberData::Always(block));
        Ok(block)
    }

    /// Adds an `always_ff` block inside a generate loop.
    pub fn generate_add_always_ff(
        &mut self,
        generate: GenerateLoop,
        sensitivity_list: &[&Expr],
    ) -> Result<VastAlwaysBase, VastError> {
        self.validate_sensitivity_list(AlwaysKind::AlwaysFF, sensitivity_list)?;
        let block = self.make_always(AlwaysKind::AlwaysFF, sensitivity_list);
        self.push_generate_member(generate, MemberData::Always(block));
        Ok(block)
    }

    /// Adds an ordinary `always @ (...)` block inside a generate loop.
    pub fn generate_add_always_at(
        &mut self,
        generate: GenerateLoop,
        sensitivity_list: &[&Expr],
    ) -> Result<VastAlwaysBase, VastError> {
        self.validate_sensitivity_list(AlwaysKind::AlwaysAt, sensitivity_list)?;
        let block = self.make_always(AlwaysKind::AlwaysAt, sensitivity_list);
        self.push_generate_member(generate, MemberData::Always(block));
        Ok(block)
    }

    /// Adds an untyped local parameter inside a generate loop.
    pub fn generate_add_localparam(
        &mut self,
        generate: GenerateLoop,
        name: &str,
        rhs: &Expr,
    ) -> LocalparamRef {
        let (reference, member) = self.parameter_member(name, rhs, true, None, false);
        self.push_generate_member(generate, member);
        LocalparamRef(reference.0)
    }

    /// Adds a typed local parameter inside a generate loop.
    pub fn generate_add_typed_localparam(
        &mut self,
        generate: GenerateLoop,
        definition: &Def,
        rhs: &Expr,
    ) -> LocalparamRef {
        let name = self.ast.defs[self.check(definition.0)].name.clone();
        let (reference, member) = self.parameter_member(&name, rhs, true, Some(*definition), false);
        self.push_generate_member(generate, member);
        LocalparamRef(reference.0)
    }

    /// Adds a continuous assignment inside a generate loop.
    pub fn generate_add_continuous_assignment(
        &mut self,
        generate: GenerateLoop,
        lhs: &Expr,
        rhs: &Expr,
    ) -> VastStatement {
        let statement = self.make_assignment(lhs, rhs, false);
        self.push_generate_member(
            generate,
            MemberData::ContinuousAssignment {
                lhs: *lhs,
                rhs: *rhs,
            },
        );
        statement
    }

    /// Adds a conditional module member inside a generate loop.
    pub fn generate_add_conditional(
        &mut self,
        generate: GenerateLoop,
        condition: &Expr,
    ) -> Conditional {
        let conditional = self.make_conditional(condition);
        self.push_generate_member(generate, MemberData::Conditional(conditional));
        conditional
    }

    /// Adds a blank line inside a generate loop.
    pub fn generate_add_blank_line(&mut self, generate: GenerateLoop) {
        self.push_generate_member(generate, MemberData::BlankLine);
    }

    /// Adds an existing comment inside a generate loop.
    pub fn generate_add_comment(&mut self, generate: GenerateLoop, comment: &Comment) {
        let text = match &self.ast.artifacts[self.check(comment.0)] {
            Artifact::Comment(text) => text.clone(),
            _ => panic!("comment handle does not identify a comment artifact"),
        };
        self.push_generate_member(generate, MemberData::Comment(text));
    }

    /// Adds an existing module instantiation inside a generate loop.
    pub fn generate_add_instantiation(
        &mut self,
        generate: GenerateLoop,
        instantiation: &Instantiation,
    ) {
        self.check(instantiation.0);
        self.push_generate_member(generate, MemberData::Instantiation(*instantiation));
    }

    /// Adds an existing inline Verilog statement inside a generate loop.
    pub fn generate_add_inline_statement(
        &mut self,
        generate: GenerateLoop,
        statement: &InlineVerilogStatement,
    ) {
        let text = match &self.ast.artifacts[self.check(statement.0)] {
            Artifact::Inline(text) => text.clone(),
            _ => panic!("inline-statement handle does not identify inline text"),
        };
        self.push_generate_member(generate, MemberData::Inline(text));
    }

    /// Adds an existing macro statement inside a generate loop.
    pub fn generate_add_macro_statement(
        &mut self,
        generate: GenerateLoop,
        statement: &MacroStatement,
    ) {
        let (expr, semicolon) = self.macro_statement_data(*statement);
        self.push_generate_member(generate, MemberData::Macro { expr, semicolon });
    }

    /// Emits one expression without requiring shared ownership of its file.
    pub fn emit_expression(&self, expression: &Expr) -> String {
        self.check(expression.0);
        emit::emit_expr(self, *expression)
    }

    /// Emits the complete Verilog or SystemVerilog source file.
    pub fn emit(&self) -> String {
        emit::emit_file(self)
    }
}
