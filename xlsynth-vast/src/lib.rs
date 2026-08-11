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

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use literal::Literal;
use model::{
    AlwaysData, Artifact, AssignmentData, Ast, BinaryOp, BlockData, CaseData, ConditionalData,
    DefData, ExprData, FileItem, GenerateData, Id, InstantiationData, IntegerTypeKind, MemberData,
    ModuleData, ParameterPortData, PortData, StatementData, TypeData, UnaryOp,
};

pub use model::{
    AlwaysKind, BlankLine, CaseStatement, Comment, Conditional, ContinuousAssignment, DataKind,
    Def, Expr, GenerateLoop, Index, IndexableExpr, InlineVerilogStatement, Instantiation,
    LocalparamRef, LogicRef, MacroRef, MacroStatement, ModulePort, ModulePortDirection,
    ParameterRef, Slice, VastAlwaysBase, VastDataType, VastFileType, VastModule, VastStatement,
    VastStatementBlock,
};

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(1);

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

/// Describes an invalid literal, unavailable width, or unsupported declaration.
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

    /// Adds a declaration-ordered module port and its corresponding reference.
    fn add_port(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
        direction: ModulePortDirection,
        kind: DataKind,
    ) -> LogicRef {
        let module_index = self.check(module.0);
        let direction_name = match direction {
            ModulePortDirection::Input => "input",
            ModulePortDirection::Output => "output",
            ModulePortDirection::InOut => "inout",
        };
        self.check_unique_module_name(module, name, direction_name)
            .expect("module ports must have unique names");
        self.check(data_type.0);
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
        logic_ref
    }

    /// Appends an ordered module member.
    fn push_member(&mut self, module: VastModule, member: MemberData) {
        let module_index = self.check(module.0);
        self.ast.modules[module_index].members.push(member);
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

    /// Adds a module at the current position in the output file.
    pub fn add_module(&mut self, name: &str) -> VastModule {
        let handle = VastModule(self.id(self.ast.modules.len()));
        self.ast.modules.push(ModuleData {
            name: name.to_owned(),
            parameter_ports: Vec::new(),
            ports: Vec::new(),
            members: Vec::new(),
            defined_names: Default::default(),
        });
        self.ast.file_items.push(FileItem::Module(handle));
        handle
    }

    /// Returns the name of a module belonging to this file.
    pub fn module_name(&self, module: VastModule) -> String {
        self.ast.modules[self.check(module.0)].name.clone()
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
            TypeData::Scalar { .. } | TypeData::Extern { .. } | TypeData::Integer { .. } => None,
        }
    }

    /// Returns whether a type uses signed arithmetic.
    pub fn type_is_signed(&self, data_type: VastDataType) -> bool {
        match &self.ast.data_types[self.check(data_type.0)] {
            TypeData::BitVector { signed, .. } | TypeData::Integer { signed, .. } => *signed,
            TypeData::PackedArray { element, .. } | TypeData::UnpackedArray { element, .. } => {
                self.type_is_signed(*element)
            }
            TypeData::Scalar { signed } => *signed,
            TypeData::Extern { .. } => false,
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
            TypeData::Extern { .. } | TypeData::Integer { .. } => Err(VastError(
                "data type does not have a declared bit-vector width".into(),
            )),
        }
    }

    /// Returns the total width of a scalar, integer, vector, or array.
    pub fn type_flat_bit_count_as_int64(&self, data_type: VastDataType) -> Result<i64, VastError> {
        match &self.ast.data_types[self.check(data_type.0)] {
            TypeData::Scalar { .. } => Ok(1),
            TypeData::BitVector { .. } => self.type_width_as_int64(data_type),
            TypeData::Integer { .. } => Ok(32),
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

    /// Adds an untyped parameter to a module's parameter-port list.
    pub fn add_parameter_port(&mut self, module: VastModule, name: &str, value: &Expr) -> LogicRef {
        let module_index = self.check(module.0);
        self.check(value.0);
        let logic_ref = self.make_named_ref(name, None);
        self.ast.modules[module_index]
            .parameter_ports
            .push(ParameterPortData {
                name: name.to_owned(),
                data_type: None,
                value: *value,
            });
        logic_ref
    }

    /// Adds a typed parameter to a module's parameter-port list.
    pub fn add_typed_parameter_port(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
        value: &Expr,
    ) -> LogicRef {
        let module_index = self.check(module.0);
        self.check(value.0);
        let logic_ref = self.make_named_ref(name, Some(*data_type));
        self.ast.modules[module_index]
            .parameter_ports
            .push(ParameterPortData {
                name: name.to_owned(),
                data_type: Some(*data_type),
                value: *value,
            });
        logic_ref
    }

    /// Adds an input port to the indicated module.
    pub fn add_input(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> LogicRef {
        self.add_port(
            module,
            name,
            data_type,
            ModulePortDirection::Input,
            DataKind::Wire,
        )
    }

    /// Adds an output port to the indicated module.
    pub fn add_output(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> LogicRef {
        self.add_port(
            module,
            name,
            data_type,
            ModulePortDirection::Output,
            DataKind::Wire,
        )
    }

    /// Adds a bidirectional port to the indicated module.
    pub fn add_inout(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> LogicRef {
        self.add_port(
            module,
            name,
            data_type,
            ModulePortDirection::InOut,
            DataKind::Wire,
        )
    }

    /// Adds an input port explicitly declared with the `logic` keyword.
    pub fn add_logic_input(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> LogicRef {
        self.add_port(
            module,
            name,
            data_type,
            ModulePortDirection::Input,
            DataKind::Logic,
        )
    }

    /// Adds an output port explicitly declared with the `logic` keyword.
    pub fn add_logic_output(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> LogicRef {
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

    /// Appends a wire declaration to a module.
    pub fn add_wire(
        &mut self,
        module: VastModule,
        name: &str,
        data_type: &VastDataType,
    ) -> LogicRef {
        self.check_unique_module_name(module, name, "wire")
            .expect("module wires must have unique names");
        self.add_declaration(module, name, data_type, DataKind::Wire)
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

    /// Creates an unsigned one-bit scalar type.
    pub fn make_scalar_type(&mut self) -> VastDataType {
        self.add_data_type(TypeData::Scalar { signed: false })
    }

    /// Creates a bit-vector type, collapsing one-bit vectors to scalars.
    pub fn make_bit_vector_type(&mut self, bit_count: i64, signed: bool) -> VastDataType {
        assert!(bit_count > 0, "bit-vector width must be greater than zero");
        if bit_count == 1 {
            return self.add_data_type(TypeData::Scalar { signed });
        }
        let width = self.add_expression(ExprData::Literal {
            text: bit_count.to_string(),
            value: Some(bit_count),
        });
        self.add_data_type(TypeData::BitVector {
            width,
            width_value: Some(bit_count),
            signed,
        })
    }

    /// Creates a bit-vector type whose width is determined by an expression.
    pub fn make_bit_vector_type_expr(&mut self, width: &Expr, signed: bool) -> VastDataType {
        let expression_index = self.check(width.0);
        let width_value = match &self.ast.expressions[expression_index] {
            ExprData::Literal { value, .. } => *value,
            _ => None,
        };
        if let Some(width) = width_value {
            assert!(width > 0, "bit-vector width must be greater than zero");
        }
        self.add_data_type(TypeData::BitVector {
            width: *width,
            width_value,
            signed,
        })
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

    /// Creates a packed multidimensional array around its element type.
    ///
    /// Dimensions are ordered from outermost to innermost. Nesting packed
    /// array types adds the new dimensions before the element's dimensions.
    ///
    /// # Panics
    ///
    /// Panics with `packed arrays cannot contain unpacked array elements` when
    /// the element contains any unpacked array dimensions. Also panics when
    /// dimensions are empty or contain a nonpositive size.
    pub fn make_packed_array_type(
        &mut self,
        element: VastDataType,
        dimensions: &[i64],
    ) -> VastDataType {
        let element_index = self.check(element.0);
        assert!(
            !matches!(
                &self.ast.data_types[element_index],
                TypeData::UnpackedArray { .. }
            ),
            "packed arrays cannot contain unpacked array elements"
        );

        assert!(
            !dimensions.is_empty(),
            "packed array requires at least one dimension"
        );
        assert!(
            dimensions.iter().all(|dimension| *dimension > 0),
            "packed-array dimensions must be greater than zero"
        );
        self.add_data_type(TypeData::PackedArray {
            element,
            dimensions: dimensions.to_vec(),
        })
    }

    /// Creates an unpacked multidimensional array around its element type.
    ///
    /// Dimensions are ordered from outermost to innermost. Nesting unpacked
    /// array types adds the new dimensions before the element's dimensions.
    pub fn make_unpacked_array_type(
        &mut self,
        element: VastDataType,
        dimensions: &[i64],
    ) -> VastDataType {
        self.check(element.0);
        assert!(
            !dimensions.is_empty(),
            "unpacked array requires at least one dimension"
        );
        assert!(
            dimensions.iter().all(|dimension| *dimension > 0),
            "unpacked-array dimensions must be greater than zero"
        );
        self.add_data_type(TypeData::UnpackedArray {
            element,
            dimensions: dimensions.to_vec(),
        })
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

    /// Creates a SystemVerilog data-type cast.
    ///
    /// # Panics
    ///
    /// Panics with `unpacked array types cannot be used in type casts` when
    /// the target contains any unpacked array dimensions.
    pub fn make_type_cast(&mut self, data_type: &VastDataType, value: &Expr) -> Expr {
        let mut type_index = self.check(data_type.0);
        self.check(value.0);

        loop {
            let current_type = &self.ast.data_types[type_index];
            assert!(
                !matches!(current_type, TypeData::UnpackedArray { .. }),
                "unpacked array types cannot be used in type casts"
            );
            match current_type {
                TypeData::PackedArray { element, .. } => type_index = self.check(element.0),
                _ => break,
            }
        }

        self.add_expression(ExprData::TypeCast {
            data_type: *data_type,
            value: *value,
        })
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

    /// Creates a preprocessor macro reference with explicit arguments.
    pub fn make_macro_ref_with_args(&mut self, name: &str, arguments: &[&Expr]) -> MacroRef {
        let arguments = self.checked_expressions(arguments);
        MacroRef(
            self.add_expression(ExprData::Macro {
                name: name.to_owned(),
                args: Some(arguments),
            })
            .0,
        )
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

    /// Constructs a named module instantiation with ordered parameters and
    /// ports.
    pub fn make_instantiation(
        &mut self,
        module_name: &str,
        instance_name: &str,
        parameter_names: &[&str],
        parameter_expressions: &[&Expr],
        connection_names: &[&str],
        connection_expressions: &[Option<&Expr>],
    ) -> Instantiation {
        assert_eq!(
            parameter_names.len(),
            parameter_expressions.len(),
            "module-instantiation parameter names and values must have equal lengths"
        );
        assert_eq!(
            connection_names.len(),
            connection_expressions.len(),
            "module-instantiation port names and values must have equal lengths"
        );
        let parameters = parameter_names
            .iter()
            .zip(parameter_expressions)
            .map(|(name, expression)| {
                self.check(expression.0);
                ((*name).to_owned(), **expression)
            })
            .collect();
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
        handle
    }

    /// Adds an existing instantiation to a module.
    pub fn add_member_instantiation(&mut self, module: VastModule, instantiation: Instantiation) {
        self.check(instantiation.0);
        self.push_member(module, MemberData::Instantiation(instantiation));
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

    /// Adds an untyped module parameter declaration.
    pub fn add_parameter(&mut self, module: VastModule, name: &str, rhs: &Expr) -> ParameterRef {
        let (reference, member) = self.parameter_member(name, rhs, false, None, false);
        self.push_member(module, member);
        ParameterRef(reference.0)
    }

    /// Adds a typed module parameter declaration.
    pub fn add_parameter_with_def(
        &mut self,
        module: VastModule,
        definition: &Def,
        rhs: &Expr,
    ) -> ParameterRef {
        let name = self.ast.defs[self.check(definition.0)].name.clone();
        let (reference, member) =
            self.parameter_member(&name, rhs, false, Some(*definition), false);
        self.push_member(module, member);
        ParameterRef(reference.0)
    }

    /// Adds an untyped module-local parameter declaration.
    pub fn add_localparam(&mut self, module: VastModule, name: &str, rhs: &Expr) -> LocalparamRef {
        let (reference, member) = self.parameter_member(name, rhs, true, None, false);
        self.push_member(module, member);
        LocalparamRef(reference.0)
    }

    /// Adds a typed module-local parameter declaration.
    pub fn add_typed_localparam(
        &mut self,
        module: VastModule,
        definition: &Def,
        rhs: &Expr,
    ) -> LocalparamRef {
        let name = self.ast.defs[self.check(definition.0)].name.clone();
        let (reference, member) = self.parameter_member(&name, rhs, true, Some(*definition), false);
        self.push_member(module, member);
        LocalparamRef(reference.0)
    }

    /// Adds a SystemVerilog `int`-typed local parameter declaration.
    pub fn add_int_localparam(
        &mut self,
        module: VastModule,
        name: &str,
        rhs: &Expr,
    ) -> LocalparamRef {
        let (reference, member) = self.parameter_member(name, rhs, true, None, true);
        self.push_member(module, member);
        LocalparamRef(reference.0)
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
        self.check(selector.0);
        let case = CaseStatement(self.id(self.ast.cases.len()));
        self.ast.cases.push(CaseData {
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

    /// Adds a labeled or unlabeled generate loop to a module.
    pub fn add_generate_loop(
        &mut self,
        module: VastModule,
        name: &str,
        start: &Expr,
        end: &Expr,
        label: Option<&str>,
    ) -> GenerateLoop {
        let generate = self.make_generate_loop(name, start, end, label);
        self.push_member(module, MemberData::Generate(generate));
        generate
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
