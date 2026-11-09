// SPDX-License-Identifier: Apache-2.0

//! Links checked BX declarations to IR functions and blocks and builds their
//! enclosing hardware.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use xlsynth_pir::ir::{self, NodePayload, Package, PackageMember};
use xlsynth_pir::ir_builder::{
    BlockState, Builder, BuilderState, FunctionBatch, InvokeSignature, is_valid_identifier,
};
use xlsynth_pir::ir_utils::external_function_references;
use xlsynth_pir::ir_verify::verify_package;
use xlsynth_pir::math::ceil_log2;
use xlsynth_pir::{
    BInstantiation, BRegister, BValue, BlockBuilder, BuilderError, FnBuilder, IrBits, IrValue,
    RegisterWriteOptions, ResetBehavior,
};

use crate::ast::{BinaryOp, Param, StructDef, Type, UnaryOp};
use crate::bx_import::library_item;
use crate::check::{
    CheckedBlock, CheckedFunction, CheckedModule, CheckedStmt, ScalarPattern, TypedExpr,
    TypedExprKind, TypedMatchArm, output_ports,
};
use crate::import_link::link_imported_functions;
use crate::layout::Layouts;
use crate::typed::{CheckedBinding, TypedFor, TypedIterable, TypedMap};

/// Destination package indices for named BX functions, externs, and inferred
/// DSLX specializations.
struct LinkedFunctions {
    named: BTreeMap<String, usize>,
    imported: BTreeMap<usize, usize>,
    signatures: HashMap<usize, InvokeSignature>,
}

/// Captures the checked signature once per distinct linked function.
fn invoke_signature(package: &Package, index: usize) -> Result<InvokeSignature, String> {
    let PackageMember::Function(function) = &package.members[index] else {
        unreachable!("linked calls refer to function members")
    };
    InvokeSignature::from_function(function).map_err(|error| error.to_string())
}

/// Links external declarations and emits functions and blocks after their
/// dependencies.
pub fn lower_module(
    module: &CheckedModule,
    package: Package,
    reset: ResetBehavior,
) -> Result<Package, String> {
    lower_checked_module(module, package, reset).map_err(crate::bx_import::diagnostic)
}

fn lower_checked_module(
    module: &CheckedModule,
    mut package: Package,
    reset: ResetBehavior,
) -> Result<Package, String> {
    verify_package(&package).map_err(|error| error.to_string())?;
    let mut layouts = Layouts::new(&module.structs);
    let named = link_externs(module, &package, &mut layouts)?;
    check_extern_blocks(module, &package, &mut layouts, reset)?;
    let block_names = module.blocks.iter().map(|block| block.name.as_str());
    let function_symbols = local_ir_names(
        module
            .functions
            .iter()
            .map(|function| function.name.as_str()),
        block_names.clone().chain(
            package
                .members
                .iter()
                .map(|member| member.graph().name.as_str()),
        ),
        &module.specialized_functions,
    );
    let block_symbols = BlockSymbols::new(module, &package, &function_symbols);
    let reserved_names = block_names
        .map(|name| block_symbols.get(name).to_owned())
        .chain(function_symbols.iter().cloned())
        .collect::<Vec<_>>();
    let mut imported = BTreeMap::new();
    for batch in &module.imports.batches {
        let indices = link_imported_functions(
            &mut package,
            batch.package.clone(),
            &batch.roots,
            &reserved_names,
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
    let mut signatures = HashMap::new();
    for &index in named.values().chain(imported.values()) {
        if let std::collections::hash_map::Entry::Vacant(entry) = signatures.entry(index) {
            entry.insert(invoke_signature(&package, index)?);
        }
    }
    let mut functions = LinkedFunctions {
        named,
        imported,
        signatures,
    };
    if module.functions.is_empty() && module.blocks.is_empty() {
        return Ok(package);
    }
    let occupied = reserved_names
        .into_iter()
        .chain(package.members.iter().map(|m| m.graph().name.clone()))
        .collect();
    let mut destination = Destination {
        batch: FunctionBatch::new(&mut package).map_err(|error| error.to_string())?,
        occupied,
        next_body: 0,
    };
    for (function, symbol) in module.functions.iter().zip(function_symbols) {
        let index = lower_function(
            function,
            &symbol,
            &mut destination,
            &functions,
            &mut layouts,
        )
        .map_err(|error| format!("fn '{}': {error}", function.name))?;
        functions
            .signatures
            .insert(index, invoke_signature(destination.batch.package(), index)?);
        functions.named.insert(function.name.clone(), index);
    }
    for block in &module.blocks {
        let builder = LowerBody::build_block(
            block,
            &block_symbols,
            &mut destination,
            &functions,
            &mut layouts,
            &module.structs,
            reset,
        )
        .map_err(|error| format!("block '{}': {error}", block.name))?;
        destination
            .batch
            .append_block(builder)
            .map_err(|error| format!("block '{}': {error}", block.name))?;
    }
    let mut concrete = module.blocks.iter().filter(|block| {
        library_item(&block.name).is_none() && !module.specialized_blocks.contains(&block.name)
    });
    if let Some(first) = concrete.next() {
        package.top = None;
        let top = if concrete.clone().next().is_none() {
            Some(first)
        } else {
            let mut public = std::iter::once(first).chain(concrete).filter(|b| b.public);
            let first = public.next();
            first.filter(|_| public.next().is_none())
        };
        if let Some(block) = top {
            package.set_top_block(&block.name)?;
        }
    }
    Ok(package)
}

/// IR symbols for library blocks, their prelinked externs, and generated
/// specializations. Parent instances use the same mapping as definitions.
struct BlockSymbols {
    renamed: BTreeMap<String, String>,
}

impl BlockSymbols {
    fn new(module: &CheckedModule, package: &Package, function_symbols: &[String]) -> Self {
        let mut renamed: BTreeMap<_, _> = module
            .library_extern_blocks
            .iter()
            .map(|(&index, key)| (key.clone(), module.extern_blocks[index].name.clone()))
            .collect();
        if module.specialized_blocks.is_empty()
            && module
                .blocks
                .iter()
                .all(|b| library_item(&b.name).is_none())
        {
            return Self { renamed };
        }
        let linked: BTreeSet<_> = package
            .members
            .iter()
            .map(|member| member.graph().name.as_str())
            .chain(function_symbols.iter().map(String::as_str))
            .collect();
        let mut occupied = linked
            .iter()
            .copied()
            .chain(module.blocks.iter().map(|block| block.name.as_str()))
            .map(str::to_owned)
            .collect();
        for block in &module.blocks {
            let name = &block.name;
            if library_item(name).is_some()
                || (module.specialized_blocks.contains(name)
                    && (!is_valid_identifier(name) || linked.contains(name.as_str())))
            {
                renamed.insert(name.clone(), fresh_ir_name(name, &mut occupied));
            }
        }
        Self { renamed }
    }

    fn get<'a>(&'a self, name: &'a str) -> &'a str {
        self.renamed.get(name).map_or(name, String::as_str)
    }
}

/// Turn a checked name into a legal IR symbol and disambiguate collisions.
fn fresh_ir_name(name: &str, occupied: &mut BTreeSet<String>) -> String {
    let stem = name.replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_");
    // Library identities and their specializations already contain a namespace.
    if name.contains('$') && occupied.insert(stem.clone()) {
        return stem;
    }
    let mut suffix = 1;
    loop {
        let candidate = format!("{stem}__bx_{suffix}");
        if occupied.insert(candidate.clone()) {
            return candidate;
        }
        suffix += 1;
    }
}

/// Keep ordinary source identifiers and allocate collision-free IR names for
/// invalid identifiers or generated functions shadowed in the linked package.
fn local_ir_names<'a>(
    names: impl Iterator<Item = &'a str> + Clone,
    existing: impl Iterator<Item = &'a str> + Clone,
    generated: &BTreeSet<String>,
) -> Vec<String> {
    let conflicts: BTreeSet<_> = if generated.is_empty() {
        BTreeSet::new()
    } else {
        existing
            .clone()
            .filter(|&name| generated.contains(name))
            .collect()
    };
    if conflicts.is_empty() && names.clone().all(is_valid_identifier) {
        return names.map(str::to_owned).collect();
    }
    let mut occupied: BTreeSet<String> = names.clone().chain(existing).map(str::to_owned).collect();
    names
        .map(|name| {
            if is_valid_identifier(name) && !conflicts.contains(name) {
                return name.to_owned();
            }
            fresh_ir_name(name, &mut occupied)
        })
        .collect()
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
        let name = declaration.name.as_str();
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
    for (i, declaration) in module.externs.iter().enumerate() {
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
        let key = module
            .library_extern_functions
            .get(&i)
            .unwrap_or(&declaration.name);
        bindings.insert(key.clone(), index);
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

/// Hardware resources used to wire register and instance inputs in one block.
#[derive(Default)]
struct BlockResources {
    registers: BTreeMap<String, Register>,
    instances: BTreeMap<String, BInstantiation>,
}

/// The destination package and symbols reserved for user definitions and
/// generated loop and map bodies.
struct Destination<'p> {
    batch: FunctionBatch<'p>,
    occupied: BTreeSet<String>,
    next_body: usize,
}

impl Destination<'_> {
    /// Assign generated functions collision-free names in source traversal
    /// order.
    fn body_symbol(&mut self, kind: &str) -> String {
        loop {
            let name = format!("__bx_{kind}_{}", self.next_body);
            self.next_body += 1;
            if self.occupied.insert(name.clone()) {
                return name;
            }
        }
    }
}

/// Builds checked value expressions in either a pure function or a clocked
/// block.
struct LowerBody<'a, 'defs, 'p, S: BuilderState> {
    builder: Builder<S>,
    destination: &'a mut Destination<'p>,
    functions: &'a LinkedFunctions,
    layouts: &'a mut Layouts<'defs>,
    values: BTreeMap<String, BValue>,
    tuple_elements: HashMap<BValue, Vec<BValue>>,
}

/// Builds a local function in the destination package's callee and node-ID
/// context.
fn lower_function(
    function: &CheckedFunction,
    symbol: &str,
    destination: &mut Destination,
    functions: &LinkedFunctions,
    layouts: &mut Layouts,
) -> Result<usize, BuilderError> {
    let mut body = LowerBody::new(FnBuilder::new(symbol), destination, functions, layouts);
    let param_symbols = local_ir_names(
        function.params.iter().map(|param| param.name.as_str()),
        std::iter::empty(),
        &BTreeSet::new(),
    );
    for (param, param_symbol) in function.params.iter().zip(param_symbols) {
        let value = body
            .builder
            .param(&param_symbol, body.layouts.lower(&param.ty)?)?;
        body.values.insert(param.name.clone(), value);
    }
    let result = body.bound_expr(&function.bindings, &function.output)?;
    body.destination.batch.append(body.builder, result)
}

impl<'a, 'defs, 'p> LowerBody<'a, 'defs, 'p, BlockState> {
    /// Allocates the interface and instances before connecting their input
    /// drivers.
    fn build_block(
        block: &CheckedBlock,
        symbols: &BlockSymbols,
        destination: &'a mut Destination<'p>,
        functions: &'a LinkedFunctions,
        layouts: &'a mut Layouts<'defs>,
        structs: &[StructDef],
        reset_behavior: ResetBehavior,
    ) -> Result<BlockBuilder, BuilderError> {
        let mut state = Self::new(
            BlockBuilder::new(symbols.get(&block.name)),
            destination,
            functions,
            layouts,
        );
        let mut resources = BlockResources::default();
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
                let instance = state.add_instance(name, symbols.get(target), reset)?;
                resources.instances.insert(name.clone(), instance);
            }
        }
        state.wire_statements(&block.statements, &mut resources, reset, None)?;
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
                Type::Named(_) | Type::Applied { .. } | Type::Tuple(_) => {
                    state.project(value, index)?
                }
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
        resources: &mut BlockResources,
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
                    resources.registers.insert(
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
                    let reg = resources.registers[name];
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
                    self.wire_statements(body, resources, reset, Some(active))?;
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
                        .instantiation_input(resources.instances[instance], port, value)?;
                }
            }
        }
        Ok(())
    }

    /// Connects the common timing domain and declares all child outputs,
    /// including unused ports.
    fn add_instance(
        &mut self,
        name: &str,
        target: &str,
        parent_reset: Option<BValue>,
    ) -> Result<BInstantiation, BuilderError> {
        let child = self
            .destination
            .batch
            .package()
            .get_block(target)
            .ok_or_else(|| {
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
            let port_name = child.port_name(port);
            let value = self.builder.instantiation_output(instance, port_name)?;
            // Ordinary BX identifiers cannot contain a dot.
            self.values.insert(format!("{name}.{port_name}"), value);
        }
        Ok(instance)
    }
}

impl<'a, 'defs, 'p, S: BuilderState> LowerBody<'a, 'defs, 'p, S> {
    fn new(
        builder: Builder<S>,
        destination: &'a mut Destination<'p>,
        functions: &'a LinkedFunctions,
        layouts: &'a mut Layouts<'defs>,
    ) -> Self {
        Self {
            builder,
            destination,
            functions,
            layouts,
            values: BTreeMap::new(),
            tuple_elements: HashMap::new(),
        }
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

    /// Selects the first asserted case, with case zero in the low bit.
    /// Select tuple fields independently to preserve each port's dependencies.
    fn select_value(
        &mut self,
        selector: BValue,
        cases: &[BValue],
        default: BValue,
    ) -> Result<BValue, BuilderError> {
        if cases.iter().all(|case| *case == default) {
            return Ok(default);
        }
        if let ir::Type::Tuple(fields) = self.builder.get_type(default)? {
            let count = fields.len();
            let mut elements = Vec::with_capacity(count);
            for index in 0..count {
                let fields = cases
                    .iter()
                    .map(|&case| self.project(case, index))
                    .collect::<Result<Vec<_>, _>>()?;
                let field_default = self.project(default, index)?;
                elements.push(self.select_value(selector, &fields, field_default)?);
            }
            let tuple = self.builder.tuple(&elements)?;
            self.tuple_elements.insert(tuple, elements);
            Ok(tuple)
        } else if let [yes] = cases {
            self.builder.select(selector, &[default, *yes], None)
        } else {
            self.builder.priority_select(selector, cases, default)
        }
    }

    /// Build each predicate from the same subject wire. The checker marks
    /// the arm that completes coverage as the unconditional default.
    fn lower_match(
        &mut self,
        subject: &TypedExpr,
        arms: &[TypedMatchArm],
    ) -> Result<BValue, BuilderError> {
        let signed = matches!(subject.ty, Type::Bits { signed: true, .. });
        let subject = self.expr(subject)?;
        let mut predicates = Vec::new();
        let mut cases = Vec::new();
        for arm in arms {
            if arm.patterns.is_empty() {
                // Every source alternative in this arm was an empty range.
                continue;
            }
            if matches!(arm.patterns.as_slice(), [ScalarPattern::Wildcard]) {
                let default = self.expr(&arm.value)?;
                if predicates.is_empty() {
                    return Ok(default);
                }
                predicates.reverse();
                let selector = if let [predicate] = predicates.as_slice() {
                    *predicate
                } else {
                    self.builder.concat(&predicates)?
                };
                return self.select_value(selector, &cases, default);
            }
            let mut alternatives = Vec::with_capacity(arm.patterns.len());
            for pattern in &arm.patterns {
                alternatives.push(match pattern {
                    ScalarPattern::Literal(bits) => {
                        let literal = self.builder.literal(IrValue::Bits(bits.clone()))?;
                        self.builder.eq(subject, literal)?
                    }
                    ScalarPattern::Range {
                        start,
                        end,
                        inclusive,
                    } => {
                        let start = self.builder.literal(IrValue::Bits(start.clone()))?;
                        let end = self.builder.literal(IrValue::Bits(end.clone()))?;
                        let ge = self.binary(BinaryOp::Ge, signed, subject, start)?;
                        let op = if *inclusive {
                            BinaryOp::Le
                        } else {
                            BinaryOp::Lt
                        };
                        let upper = self.binary(op, signed, subject, end)?;
                        self.builder.and(ge, upper)?
                    }
                    ScalarPattern::Wildcard => unreachable!("exhaustive arm is normalized"),
                });
            }
            let predicate = if let [only] = alternatives.as_slice() {
                *only
            } else {
                let bits = self.builder.concat(&alternatives)?;
                self.builder.or_reduce(bits)?
            };
            predicates.push(predicate);
            cases.push(self.expr(&arm.value)?);
        }
        Err(BuilderError::InvalidOperation(
            "match has no exhaustive arm".into(),
        ))
    }

    /// Invokes a linked function with the lowered BX arguments in source order.
    fn invoke(&mut self, index: usize, args: &[TypedExpr]) -> Result<BValue, BuilderError> {
        let args = args
            .iter()
            .map(|arg| self.expr(arg))
            .collect::<Result<Vec<_>, _>>()?;
        self.builder
            .invoke_with_signature(&self.functions.signatures[&index], &args)
    }

    /// Lower a result after evaluating its checked lexical bindings in order.
    fn bound_expr(
        &mut self,
        bindings: &[CheckedBinding],
        output: &TypedExpr,
    ) -> Result<BValue, BuilderError> {
        for binding in bindings {
            let value = self.expr(&binding.value)?;
            self.values.insert(binding.name.clone(), value);
        }
        self.expr(output)
    }

    /// Emit a unary IR map for a closed lambda or a counted loop carrying
    /// the result array when the lambda reads enclosing wires.
    fn map(&mut self, map: &TypedMap) -> Result<BValue, BuilderError> {
        let array = self.expr(&map.array)?;
        let symbol = self.destination.body_symbol("map");
        let mut builder = FnBuilder::new(&symbol);
        let element = builder.param("element", self.layouts.lower(&map.element.ty)?)?;
        let mut capture_wires = Vec::new();
        let captures = self.capture_params(&mut builder, &map.captures, &mut capture_wires)?;
        let mut body = LowerBody::new(builder, self.destination, self.functions, self.layouts);
        body.values = captures;
        body.values.insert(map.element.name.clone(), element);
        let result = body.bound_expr(&map.bindings, &map.output)?;
        let member = body.destination.batch.append(body.builder, result)?;
        let PackageMember::Function(function) = &self.destination.batch.package().members[member]
        else {
            unreachable!("appended map callback is a function");
        };
        if capture_wires.is_empty() {
            return self.builder.map(array, function);
        }
        let callback = InvokeSignature::from_function(function)?;
        self.captured_map(array, &callback, &capture_wires)
    }

    /// Broadcast the first real result to seed the output, then replace each
    /// remaining slot by invoking the same callback on its input element.
    fn captured_map(
        &mut self,
        array: BValue,
        callback: &InvokeSignature,
        captures: &[BValue],
    ) -> Result<BValue, BuilderError> {
        let ir::Type::Array(input) = self.builder.get_type(array)? else {
            unreachable!("checked map input is an array");
        };
        let count = input.element_count;
        let zero = self.builder.literal(IrValue::bool(false))?;
        let first = self.builder.array_index(array, zero)?;
        let mut args = vec![first];
        args.extend_from_slice(captures);
        let result = self.builder.invoke_with_signature(callback, &args)?;
        let singleton = self
            .builder
            .array(self.builder.get_type(result)?.clone(), &[result])?;
        if count == 1 {
            return Ok(singleton);
        }
        // Saturating array_slice repeats the singleton. This seed also works
        // for runtime aggregate values that XLS cannot represent as literals.
        let init = self.builder.array_slice(singleton, zero, count)?;
        let symbol = self.destination.body_symbol("map_loop");
        let mut builder = FnBuilder::new(&symbol);
        let width = ceil_log2(count);
        let ordinal = builder.param("index", ir::Type::Bits(width))?;
        let carry = builder.param("carry", self.builder.get_type(init)?.clone())?;
        let iterable = builder.param("iterable", self.builder.get_type(array)?.clone())?;
        let params = self.wire_params(&mut builder, captures)?;
        let one = builder.literal(IrValue::Bits(IrBits::from_lsb_fn(width, |i| i == 0)))?;
        let index = builder.add(ordinal, one)?;
        let element = builder.array_index(iterable, index)?;
        let mut callback_args = vec![element];
        callback_args.extend(params);
        let replacement = builder.invoke_with_signature(callback, &callback_args)?;
        let result = builder.array_update(carry, replacement, &[index])?;
        let member = self.destination.batch.append(builder, result)?;
        let PackageMember::Function(function) = &self.destination.batch.package().members[member]
        else {
            unreachable!("appended map iteration is a function");
        };
        let mut invariants = vec![array];
        invariants.extend_from_slice(captures);
        self.builder
            .counted_for(init, count - 1, 1, function, &invariants)
    }

    /// Create one capture parameter per enclosing wire, in invariant order.
    fn wire_params(
        &self,
        builder: &mut FnBuilder,
        wires: &[BValue],
    ) -> Result<Vec<BValue>, BuilderError> {
        wires
            .iter()
            .enumerate()
            .map(|(ordinal, &wire)| {
                builder.param(
                    &format!("capture_{ordinal}"),
                    self.builder.get_type(wire)?.clone(),
                )
            })
            .collect()
    }

    /// Pair the sorted enclosing wires with body parameters in invariant order.
    fn capture_params(
        &self,
        builder: &mut FnBuilder,
        captures: &BTreeSet<String>,
        invariants: &mut Vec<BValue>,
    ) -> Result<BTreeMap<String, BValue>, BuilderError> {
        let start = invariants.len();
        invariants.extend(captures.iter().map(|name| self.values[name]));
        let params = self.wire_params(builder, &invariants[start..])?;
        Ok(captures.iter().cloned().zip(params).collect())
    }

    /// Build one pure iteration function. Captures become invariant
    /// arguments; the array or range element is derived from the ordinal.
    fn counted_for(&mut self, fold: &TypedFor) -> Result<BValue, BuilderError> {
        let carry = self.expr(&fold.seed)?;
        if fold.trip_count == 0 {
            return Ok(carry);
        }
        let mut invariants = Vec::new();
        let symbol = self.destination.body_symbol("for");
        let mut builder = FnBuilder::new(&symbol);
        let index_width = match &fold.iterable {
            TypedIterable::Range { start } => start.get_bit_count().max(1),
            TypedIterable::Array(_) => ceil_log2(fold.trip_count).max(1),
        };
        let index = builder.param("index", ir::Type::Bits(index_width))?;
        let accumulator = builder.param("carry", self.builder.get_type(carry)?.clone())?;
        let array = match &fold.iterable {
            TypedIterable::Array(array) => {
                let wire = self.expr(array)?;
                invariants.push(wire);
                Some(builder.param("iterable", self.builder.get_type(wire)?.clone())?)
            }
            TypedIterable::Range { .. } => None,
        };
        let captures = self.capture_params(&mut builder, &fold.captures, &mut invariants)?;
        let mut body = LowerBody::new(builder, self.destination, self.functions, self.layouts);
        body.values = captures;
        body.values
            .insert(fold.accumulator.name.clone(), accumulator);
        let element = match &fold.iterable {
            TypedIterable::Range { start } if start.get_bit_count() == 0 => {
                body.builder.literal(IrValue::Bits(start.clone()))?
            }
            TypedIterable::Range { start } if start.is_zero() => index,
            TypedIterable::Range { start } => {
                let start = body.builder.literal(IrValue::Bits(start.clone()))?;
                body.builder.add(start, index)?
            }
            TypedIterable::Array(_) => body
                .builder
                .array_index(array.expect("array invariant"), index)?,
        };
        body.values.insert(fold.element.name.clone(), element);
        let result = body.bound_expr(&fold.bindings, &fold.output)?;
        let index = body.destination.batch.append(body.builder, result)?;
        let PackageMember::Function(function) = &self.destination.batch.package().members[index]
        else {
            unreachable!("appended loop body is a function");
        };
        self.builder
            .counted_for(carry, fold.trip_count, 1, function, &invariants)
    }

    /// Lowers a checked expression at the current instant, without inserting
    /// registers.
    fn expr(&mut self, expr: &TypedExpr) -> Result<BValue, BuilderError> {
        match &expr.kind {
            TypedExprKind::Name(name) => Ok(self.values[name]),
            TypedExprKind::Literal(value) => self.builder.literal(value.clone()),
            TypedExprKind::For(fold) => self.counted_for(fold),
            TypedExprKind::Map(map) => self.map(map),
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
            TypedExprKind::ArrayUpdate {
                array,
                index,
                replacement,
            } => {
                let array = self.expr(array)?;
                let index = self.expr(index)?;
                let replacement = self.expr(replacement)?;
                self.builder.array_update(array, replacement, &[index])
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
            TypedExprKind::BindingTupleIndex { name, index } => {
                self.project(self.values[name], *index)
            }
            TypedExprKind::InstanceOutput { instance, port } => {
                Ok(self.values[&format!("{instance}.{port}")])
            }
            TypedExprKind::Call { name, args } => self.invoke(self.functions.named[name], args),
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
                self.select_value(condition, &[yes], no)
            }
            TypedExprKind::Match { subject, arms } => self.lower_match(subject, arms),
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
