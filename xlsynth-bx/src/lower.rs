// SPDX-License-Identifier: Apache-2.0

//! Links checked BX declarations to IR functions and blocks and builds their
//! enclosing hardware.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use xlsynth_pir::ir::{self, NodePayload, Package, PackageMember};
use xlsynth_pir::ir_utils::external_function_references;
use xlsynth_pir::ir_verify::verify_package;
use xlsynth_pir::{
    BInstantiation, BRegister, BValue, BlockBuilder, BuilderError, RegisterWriteOptions,
    ResetBehavior,
};

use crate::ast::{BinaryOp, Param, StructDef, Type, UnaryOp};
use crate::check::{
    CheckedBlock, CheckedModule, CheckedStmt, TypedExpr, TypedExprKind, output_ports,
};
use crate::import_link::link_imported_functions;
use crate::layout::Layouts;

/// Destination package indices for explicitly declared externs and inferred
/// imported DSLX specializations.
struct LinkedFunctions {
    externs: BTreeMap<String, usize>,
    imported: BTreeMap<usize, usize>,
}

/// Links every extern and appends verified blocks in child-before-parent order.
pub fn lower_module(
    module: &CheckedModule,
    mut package: Package,
    reset: ResetBehavior,
) -> Result<Package, String> {
    verify_package(&package).map_err(|error| error.to_string())?;
    let mut layouts = Layouts::new(&module.structs);
    let externs = link_externs(module, &package, &mut layouts)?;
    check_extern_blocks(module, &package, &mut layouts, reset)?;
    let block_names = module
        .blocks
        .iter()
        .map(|block| block.name.clone())
        .collect::<Vec<_>>();
    let mut imported = BTreeMap::new();
    for batch in &module.imports.batches {
        let indices = link_imported_functions(
            &mut package,
            batch.package.clone(),
            &batch.roots,
            &block_names,
        )?;
        for (&id, index) in batch.call_ids.iter().zip(indices) {
            let call = &module.imports.calls[id];
            let PackageMember::Function(function) = &package.members[index] else {
                unreachable!("imported roots are function members")
            };
            check_signature(
                &format!("{}::{}", call.module, call.function),
                call.args.iter(),
                &call.result,
                function,
                &mut layouts,
            )?;
            imported.insert(id, index);
        }
    }
    let functions = LinkedFunctions { externs, imported };
    for block in &module.blocks {
        let builder = LowerBlock::build(
            block,
            &package,
            &functions,
            &mut layouts,
            &module.structs,
            reset,
        )
        .map_err(|error| format!("block '{}': {error}", block.name))?;
        builder
            .build_into_package(&mut package)
            .map_err(|error| format!("block '{}': {error}", block.name))?;
    }
    if !module.blocks.is_empty() {
        package.top = None;
        let mut public = module.blocks.iter().filter(|block| block.public);
        let top = if module.blocks.len() == 1 {
            module.blocks.first()
        } else {
            let first = public.next();
            first.filter(|_| public.next().is_none())
        };
        if let Some(block) = top {
            package.set_top_block(&block.name)?;
        }
    }
    Ok(package)
}

/// Matches imported block signatures by timing role and named data ports before
/// instantiation.
fn check_extern_blocks(
    module: &CheckedModule,
    package: &Package,
    layouts: &mut Layouts,
    reset_behavior: ResetBehavior,
) -> Result<(), String> {
    for declaration in &module.extern_blocks {
        let name = &declaration.name;
        let block = package
            .get_block(name)
            .ok_or_else(|| format!("extern block '{name}': no linked IR block of that name"))?;
        if block.clock_port_name().is_none() {
            return Err(format!(
                "extern block '{name}' requires a clock but the linked block has none"
            ));
        }
        let reset_port = match (&declaration.reset, &block.reset) {
            (Some(_), Some(actual)) => {
                if actual.asynchronous != reset_behavior.asynchronous
                    || actual.active_low != reset_behavior.active_low
                {
                    return Err(format!(
                        "extern block '{name}' expects reset asynchronous={}, active_low={}, but the linked block has asynchronous={}, active_low={}",
                        reset_behavior.asynchronous,
                        reset_behavior.active_low,
                        actual.asynchronous,
                        actual.active_low
                    ));
                }
                Some(actual.port)
            }
            (None, None) => None,
            (Some(_), None) => {
                return Err(format!(
                    "extern block '{name}' declares a reset but the linked block has none"
                ));
            }
            (None, Some(_)) => {
                return Err(format!(
                    "extern block '{name}' omits the linked block's reset"
                ));
            }
        };

        // The physical names of the clock and reset belong to the child. Only
        // its ordinary inputs and outputs participate in the BX named data ABI.
        let inputs = block.input_ports().filter(|port| Some(*port) != reset_port);
        check_data_ports(block, "input", &declaration.params, inputs, layouts)?;
        let outputs = output_ports(&declaration.result, &module.structs)?;
        check_data_ports(block, "output", &outputs, block.output_ports(), layouts)?;
    }
    Ok(())
}

/// Checks a complete set of named ports, independent of the linked IR
/// declaration order.
fn check_data_ports(
    block: &ir::Block,
    direction: &str,
    expected: &[Param],
    ports: impl Iterator<Item = ir::NodeRef>,
    layouts: &mut Layouts,
) -> Result<(), String> {
    let mut actual: BTreeMap<_, _> = ports
        .map(|port| (block.port_name(port), block.port_type(port)))
        .collect();
    for param in expected {
        let name = param.name.as_str();
        let Some(actual_type) = actual.remove(name) else {
            return Err(format!(
                "extern block '{}': linked IR has no data {direction} named '{name}'",
                block.name
            ));
        };
        let ty = layouts
            .lower(&param.ty)
            .map_err(|error| error.to_string())?;
        if ty != *actual_type {
            return Err(format!(
                "extern block '{}': {direction} '{name}' expects {ty}, but linked IR has {actual_type}",
                block.name
            ));
        }
    }
    if let Some((extra, _)) = actual.first_key_value() {
        return Err(format!(
            "extern block '{}': linked IR has undeclared data {direction} '{extra}'",
            block.name
        ));
    }
    Ok(())
}

/// Resolves declarations to member indices, which stay stable as blocks are
/// appended.
fn link_externs(
    module: &CheckedModule,
    package: &Package,
    layouts: &mut Layouts,
) -> Result<BTreeMap<String, usize>, String> {
    let symbols: BTreeMap<_, _> = package
        .members
        .iter()
        .enumerate()
        .filter_map(|(index, member)| match member {
            PackageMember::Function(function) => Some((function.name.as_str(), index)),
            PackageMember::Block(_) => None,
        })
        .collect();
    let mut bindings = BTreeMap::new();
    for declaration in &module.externs {
        let name = declaration.name.as_str();
        let symbol = match (&declaration.symbol, name.split_once("::")) {
            (Some(symbol), _) => symbol.clone(),
            (None, None) if symbols.contains_key(name) => name.to_owned(),
            (None, qualifier) => {
                let (module, function) = qualifier.unwrap_or((&package.name, name));
                xlsynth::mangle_dslx_name(module, function)
                    .map_err(|error| format!("extern fn '{name}': {error}"))?
            }
        };
        let index = symbols
            .get(symbol.as_str())
            .copied()
            .ok_or_else(|| format!("extern fn '{name}': no linked IR function for '{symbol}'"))?;
        let PackageMember::Function(function) = &package.members[index] else {
            unreachable!("symbol table contains only function indices")
        };
        check_signature(
            &format!("extern fn '{name}'"),
            declaration.params.iter().map(|param| &param.ty),
            &declaration.result,
            function,
            layouts,
        )?;
        bindings.insert(declaration.name.clone(), index);
    }
    // Both select arms can run. Check reachable bodies for effects, including
    // dead nodes and calls through maps or loops.
    let mut pending: Vec<usize> = bindings.values().copied().collect();
    let mut visited = BTreeSet::new();
    while let Some(index) = pending.pop() {
        if !visited.insert(index) {
            continue;
        }
        let PackageMember::Function(function) = &package.members[index] else {
            unreachable!("reachable function indices come from the symbol table")
        };
        check_function_purity(function)?;
        for callee in external_function_references(function) {
            pending.push(
                *symbols
                    .get(callee.as_str())
                    .ok_or_else(|| format!("missing callee '{callee}'"))?,
            );
        }
    }
    Ok(bindings)
}

/// Checks that a resolved IR function has the argument and result layouts
/// used to type-check the BX call.
fn check_signature<'a>(
    source: &str,
    params: impl Iterator<Item = &'a Type>,
    result: &Type,
    function: &ir::Fn,
    layouts: &mut Layouts,
) -> Result<(), String> {
    let params = params
        .map(|ty| layouts.lower(ty))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let result = layouts.lower(result).map_err(|error| error.to_string())?;
    if params
        .iter()
        .ne(function.param_nodes().map(|node| &node.ty))
        || result != function.ret_ty
    {
        let expected = params
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let actual = function
            .param_nodes()
            .map(|node| node.ty.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "{source} expects ({expected}) -> {result}, but linked IR function '{}' has ({actual}) -> {}",
            function.name, function.ret_ty
        ));
    }
    Ok(())
}

/// Rejects effects even on dead nodes because both arms of a BX select can run.
pub(crate) fn check_function_purity(function: &ir::Fn) -> Result<(), String> {
    if function.ffi_proto().is_some() {
        return Err(format!(
            "cannot lift IR function '{}' with a foreign implementation into BX",
            function.name
        ));
    }
    for node in &function.nodes {
        if matches!(
            node.payload,
            NodePayload::Assert { .. } | NodePayload::Trace { .. } | NodePayload::Cover { .. }
        ) {
            return Err(format!(
                "cannot lift effectful IR function '{}' into BX because it contains '{}'",
                function.name,
                node.payload.get_operator()
            ));
        }
    }
    Ok(())
}

/// A BX register and the physical reset signal it uses, if any.
#[derive(Clone, Copy)]
struct Register {
    handle: BRegister,
    reset: Option<BValue>,
}

/// Block-local values and resources being wired into a checked PIR builder.
struct LowerBlock<'a, 'defs> {
    builder: BlockBuilder,
    package: &'a Package,
    functions: &'a LinkedFunctions,
    layouts: &'a mut Layouts<'defs>,
    values: BTreeMap<String, BValue>,
    tuple_elements: HashMap<BValue, Vec<BValue>>,
    registers: BTreeMap<String, Register>,
    instances: BTreeMap<String, BInstantiation>,
}

impl<'a, 'defs> LowerBlock<'a, 'defs> {
    /// Allocates the interface and instances before connecting their input
    /// drivers.
    fn build(
        block: &CheckedBlock,
        package: &'a Package,
        functions: &'a LinkedFunctions,
        layouts: &'a mut Layouts<'defs>,
        structs: &[StructDef],
        reset_behavior: ResetBehavior,
    ) -> Result<BlockBuilder, BuilderError> {
        let mut state = Self {
            builder: BlockBuilder::new(&block.name),
            package,
            functions,
            layouts,
            values: BTreeMap::new(),
            tuple_elements: HashMap::new(),
            registers: BTreeMap::new(),
            instances: BTreeMap::new(),
        };
        state.builder.clock_port(&block.clock)?;
        let reset = if let Some(name) = &block.reset {
            let physical = state.builder.input_port(name, ir::Type::Bits(1))?;
            state.builder.set_reset(physical, reset_behavior)?;
            let asserted = if reset_behavior.active_low {
                state.builder.not(physical)?
            } else {
                physical
            };
            state.values.insert(name.clone(), asserted);
            Some(physical)
        } else {
            None
        };
        for param in &block.params {
            let value = state
                .builder
                .input_port(&param.name, state.layouts.lower(&param.ty)?)?;
            state.values.insert(param.name.clone(), value);
        }
        for statement in &block.statements {
            if let CheckedStmt::Inst {
                name,
                block: target,
            } = statement
            {
                state.add_instance(name, target, reset)?;
            }
        }
        state.wire_statements(&block.statements, reset, None)?;
        let ports = output_ports(&block.result, structs).map_err(BuilderError::InvalidOperation)?;
        if let TypedExprKind::Tuple(fields) = &block.output.kind {
            // Drive fields separately to preserve their feedback dependencies.
            for (port, field) in ports.iter().zip(fields) {
                let value = state.expr(field)?;
                state.builder.output_port(&port.name, value)?;
            }
            return Ok(state.builder);
        }
        let value = state.expr(&block.output)?;
        for (index, port) in ports.iter().enumerate() {
            let field = match block.result {
                Type::Named(_) | Type::Tuple(_) => state.project(value, index)?,
                Type::Bits { .. } | Type::Bool | Type::Array { .. } => value,
            };
            state.builder.output_port(&port.name, field)?;
        }
        Ok(state.builder)
    }

    /// Wires register updates using one shared load-enable node for each
    /// conditional scope.
    fn wire_statements(
        &mut self,
        statements: &[CheckedStmt],
        reset: Option<BValue>,
        guard: Option<BValue>,
    ) -> Result<(), BuilderError> {
        for statement in statements {
            match statement {
                CheckedStmt::Reg {
                    name,
                    ty,
                    reset: value,
                } => {
                    let handle =
                        self.builder
                            .register(name, self.layouts.lower(ty)?, value.clone())?;
                    let current = self.builder.register_read(handle)?;
                    self.values.insert(name.clone(), current);
                    self.registers.insert(
                        name.clone(),
                        Register {
                            handle,
                            reset: value.as_ref().and(reset),
                        },
                    );
                }
                CheckedStmt::Let { name, value } => {
                    let value = self.expr(value)?;
                    self.values.insert(name.clone(), value);
                }
                CheckedStmt::Write { name, value } => {
                    let reg = self.registers[name];
                    let data = self.expr(value)?;
                    self.builder.register_write(
                        reg.handle,
                        data,
                        RegisterWriteOptions {
                            load_enable: guard,
                            reset: reg.reset,
                        },
                    )?;
                }
                CheckedStmt::If { condition, body } => {
                    let condition = self.expr(condition)?;
                    let active = match guard {
                        Some(parent) => self.builder.and(parent, condition)?,
                        None => condition,
                    };
                    self.wire_statements(body, reset, Some(active))?;
                }
                CheckedStmt::Inst { .. } => {
                    // Instance outputs were made available before wiring any
                    // inputs.
                }
                CheckedStmt::Connect {
                    instance,
                    port,
                    value,
                } => {
                    let value = self.expr(value)?;
                    self.builder
                        .instantiation_input(self.instances[instance], port, value)?;
                }
            }
        }
        Ok(())
    }

    /// Selects the actual field wire for a locally constructed tuple, even
    /// through let aliases.
    fn project(&mut self, tuple: BValue, index: usize) -> Result<BValue, BuilderError> {
        if let Some(elements) = self.tuple_elements.get(&tuple) {
            Ok(elements[index])
        } else {
            self.builder.tuple_index(tuple, index)
        }
    }

    /// Selects tuple fields independently to preserve each port's combinational
    /// dependencies.
    fn select_value(
        &mut self,
        condition: BValue,
        yes: BValue,
        no: BValue,
    ) -> Result<BValue, BuilderError> {
        if yes == no {
            return Ok(yes);
        }
        if let ir::Type::Tuple(fields) = self.builder.get_type(yes)? {
            let count = fields.len();
            let mut elements = Vec::with_capacity(count);
            for index in 0..count {
                let yes = self.project(yes, index)?;
                let no = self.project(no, index)?;
                elements.push(self.select_value(condition, yes, no)?);
            }
            let tuple = self.builder.tuple(&elements)?;
            self.tuple_elements.insert(tuple, elements);
            Ok(tuple)
        } else {
            self.builder.select(condition, &[no, yes], None)
        }
    }

    /// Connects the common timing domain and declares all child outputs,
    /// including unused ports.
    fn add_instance(
        &mut self,
        name: &str,
        target: &str,
        parent_reset: Option<BValue>,
    ) -> Result<(), BuilderError> {
        let child = self.package.get_block(target).ok_or_else(|| {
            BuilderError::InvalidOperation(format!("unknown child block '{target}'"))
        })?;
        let instance = self.builder.instantiate(name, child)?;
        if let Some(reset) = &child.reset {
            let signal = parent_reset.ok_or_else(|| {
                BuilderError::InvalidOperation(format!("child '{name}' requires a reset"))
            })?;
            self.builder
                .instantiation_input(instance, child.port_name(reset.port), signal)?;
        }
        for port in child.output_ports() {
            self.builder
                .instantiation_output(instance, child.port_name(port))?;
        }
        self.instances.insert(name.to_owned(), instance);
        Ok(())
    }

    /// Invokes a linked function with the lowered BX arguments in source order.
    fn invoke(&mut self, index: usize, args: &[TypedExpr]) -> Result<BValue, BuilderError> {
        let args = args
            .iter()
            .map(|arg| self.expr(arg))
            .collect::<Result<Vec<_>, _>>()?;
        let PackageMember::Function(function) = &self.package.members[index] else {
            unreachable!("linked calls refer to function members")
        };
        self.builder.invoke(function, &args)
    }

    /// Lowers a checked expression at the current instant, without inserting
    /// registers.
    fn expr(&mut self, expr: &TypedExpr) -> Result<BValue, BuilderError> {
        match &expr.kind {
            TypedExprKind::Name(name) => Ok(self.values[name]),
            TypedExprKind::Literal(value) => self.builder.literal(value.clone()),
            TypedExprKind::Array(items) => {
                let Type::Array { element, .. } = &expr.ty else {
                    return Err(BuilderError::InvalidOperation(
                        "array expression requires an array type".into(),
                    ));
                };
                let values = items
                    .iter()
                    .map(|item| self.expr(item))
                    .collect::<Result<Vec<_>, _>>()?;
                self.builder.array(self.layouts.lower(element)?, &values)
            }
            TypedExprKind::Index { value, index } => {
                let value = self.expr(value)?;
                let index = self.expr(index)?;
                self.builder.array_index(value, index)
            }
            TypedExprKind::WidthSlice { value, start } => {
                let value = self.expr(value)?;
                let start = self.expr(start)?;
                let width = self.layouts.lower(&expr.ty)?.bit_count();
                self.builder.dynamic_bit_slice(value, start, width)
            }
            TypedExprKind::Tuple(fields) => {
                let fields = fields
                    .iter()
                    .map(|field| self.expr(field))
                    .collect::<Result<Vec<_>, _>>()?;
                let value = self.builder.tuple(&fields)?;
                self.tuple_elements.insert(value, fields);
                Ok(value)
            }
            TypedExprKind::TupleIndex { value, index } => {
                let value = self.expr(value)?;
                self.project(value, *index)
            }
            TypedExprKind::InstanceOutput { instance, port } => self
                .builder
                .instantiation_output(self.instances[instance], port),
            TypedExprKind::Call { name, args } => self.invoke(self.functions.externs[name], args),
            TypedExprKind::ImportedCall { id, args } => {
                self.invoke(self.functions.imported[id], args)
            }
            TypedExprKind::Unary { op, value } => {
                let value = self.expr(value)?;
                match op {
                    UnaryOp::Not => self.builder.not(value),
                    UnaryOp::Neg => self.builder.neg(value),
                }
            }
            TypedExprKind::Binary { op, left, right } => {
                let signed = matches!(left.ty, Type::Bits { signed: true, .. });
                let lhs = self.expr(left)?;
                let rhs = self.expr(right)?;
                self.binary(*op, signed, lhs, rhs)
            }
            TypedExprKind::Cast { value } => {
                let src = self.expr(value)?;
                let src_width = self.builder.get_type(src)?.bit_count();
                let dst_width = match expr.ty {
                    Type::Bits { width, .. } => width,
                    Type::Bool => 1,
                    _ => {
                        return Err(BuilderError::InvalidOperation(
                            "cast target must be bits".into(),
                        ));
                    }
                };
                if src_width == dst_width {
                    Ok(src)
                } else if dst_width < src_width {
                    self.builder.bit_slice(src, 0, dst_width)
                } else if src_width > 0 && matches!(value.ty, Type::Bits { signed: true, .. }) {
                    self.builder.sign_extend(src, dst_width)
                } else {
                    self.builder.zero_extend(src, dst_width)
                }
            }
            TypedExprKind::If {
                condition,
                then_value,
                else_value,
            } => {
                let condition = self.expr(condition)?;
                let yes = self.expr(then_value)?;
                let no = self.expr(else_value)?;
                self.select_value(condition, yes, no)
            }
        }
    }

    /// Chooses signed PIR operations using BX operand types before erasing
    /// signedness.
    fn binary(
        &mut self,
        op: BinaryOp,
        signed: bool,
        lhs: BValue,
        rhs: BValue,
    ) -> Result<BValue, BuilderError> {
        let b = &mut self.builder;
        match op {
            BinaryOp::Add => b.add(lhs, rhs),
            BinaryOp::Sub => b.sub(lhs, rhs),
            BinaryOp::Mul if signed => b.smul(lhs, rhs),
            BinaryOp::Mul => b.umul(lhs, rhs),
            BinaryOp::Div if signed => b.sdiv(lhs, rhs),
            BinaryOp::Div => b.udiv(lhs, rhs),
            BinaryOp::Mod if signed => b.smod(lhs, rhs),
            BinaryOp::Mod => b.umod(lhs, rhs),
            BinaryOp::And | BinaryOp::LogicalAnd => b.and(lhs, rhs),
            BinaryOp::Or | BinaryOp::LogicalOr => b.or(lhs, rhs),
            BinaryOp::Xor => b.xor(lhs, rhs),
            BinaryOp::Concat => b.concat(&[lhs, rhs]),
            BinaryOp::Shl => b.shll(lhs, rhs),
            BinaryOp::Shr if signed => b.shra(lhs, rhs),
            BinaryOp::Shr => b.shrl(lhs, rhs),
            BinaryOp::Eq => b.eq(lhs, rhs),
            BinaryOp::Ne => b.ne(lhs, rhs),
            BinaryOp::Lt if signed => b.slt(lhs, rhs),
            BinaryOp::Lt => b.ult(lhs, rhs),
            BinaryOp::Le if signed => b.sle(lhs, rhs),
            BinaryOp::Le => b.ule(lhs, rhs),
            BinaryOp::Gt if signed => b.sgt(lhs, rhs),
            BinaryOp::Gt => b.ugt(lhs, rhs),
            BinaryOp::Ge if signed => b.sge(lhs, rhs),
            BinaryOp::Ge => b.uge(lhs, rhs),
        }
    }
}
