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

use crate::ast::{BinaryOp, Param, Type, UnaryOp};
use crate::bx_import::library_item;
use crate::check::{
    CheckedBlock, CheckedFunction, CheckedModule, CheckedStmt, TypedExpr, TypedExprKind,
    TypedMatchArm, literal_copy_work, output_ports,
};
use crate::cycle_diagnostic::{CycleSources, WireDriver};
use crate::import_link::link_imported_functions;
use crate::layout::Layouts;
use crate::typed::{
    CaptureUse, CheckedBinding, TypedFor, TypedIterable, TypedMap, TypedMatchPattern, repr_bits,
};

/// Destination package indices for named BX functions, externs, and inferred
/// DSLX specializations.
struct LinkedFunctions {
    named: BTreeMap<String, usize>,
    imported: BTreeMap<usize, usize>,
    signatures: HashMap<usize, InvokeSignature>,
}

impl LinkedFunctions {
    /// Recognize a map body that passes its element directly to one linked
    /// function. The IR map can use that function's cached signature.
    fn forwarded_map_target(
        &self,
        map: &TypedMap,
        evaluate_test_discards: bool,
    ) -> Option<&InvokeSignature> {
        if map
            .bindings
            .iter()
            .any(|b| evaluate_test_discards || !b.is_test_discard())
            || map
                .captures
                .values()
                .any(|usage| evaluate_test_discards || *usage == CaptureUse::Value)
        {
            return None;
        }
        let (index, args) = match &map.output.kind {
            TypedExprKind::Call { name, args } => (self.named[name], args),
            TypedExprKind::ImportedCall { id, args } => (self.imported[id], args),
            _ => return None,
        };
        match args.as_slice() {
            [arg] if matches!(&arg.kind, TypedExprKind::Name(name) if *name == map.element.name) => {
                Some(&self.signatures[&index])
            }
            _ => None,
        }
    }
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
    lower_checked_module(module, package, reset, crate::check::MAX_TYPE_SIZE)
        .map_err(|error| error.0)
}

/// A lowering diagnostic ready for display, including any source excerpt.
#[derive(Debug)]
struct LowerError(String);

impl From<String> for LowerError {
    fn from(message: String) -> Self {
        Self(crate::bx_import::diagnostic(message))
    }
}

fn lower_checked_module(
    module: &CheckedModule,
    mut package: Package,
    reset: ResetBehavior,
    empty_array_limit: usize,
) -> Result<Package, LowerError> {
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
        evaluate_test_discards: false,
    };
    let mut function_specializations =
        EmptyArrayBudget::new("function specialization", empty_array_limit);
    for (function, symbol) in module.functions.iter().zip(function_symbols) {
        destination.evaluate_test_discards =
            module.functions_with_test_discards.contains(&function.name);
        let budget = if module.specialized_functions.contains(&function.name) {
            &mut function_specializations
        } else {
            &mut EmptyArrayBudget::new("body", empty_array_limit)
        };
        let index = lower_function(
            function,
            &symbol,
            &mut destination,
            &functions,
            &mut layouts,
            budget,
        )
        .map_err(|error| format!("fn '{}': {error}", function.name))?;
        functions
            .signatures
            .insert(index, invoke_signature(destination.batch.package(), index)?);
        functions.named.insert(function.name.clone(), index);
    }
    let mut block_specializations =
        EmptyArrayBudget::new("generic block specialization", empty_array_limit);
    destination.evaluate_test_discards = false;
    for block in &module.blocks {
        let ir_block = block_symbols.get(&block.name);
        let budget = if module.specialized_blocks.contains(&block.name) {
            &mut block_specializations
        } else {
            &mut EmptyArrayBudget::new("body", empty_array_limit)
        };
        let body = LowerBody::new(
            BlockBuilder::new(ir_block),
            &mut destination,
            &functions,
            &mut layouts,
            budget,
        );
        let (builder, sources) = body
            .build_block(block, &block_symbols, reset)
            .map_err(|error| format!("block '{}': {error}", block.name))?;
        destination.batch.append_block(builder).map_err(|error| {
            sources.render(&error, block, ir_block).map_or_else(
                || format!("block '{}': {error}", block.name).into(),
                LowerError,
            )
        })?;
        if let Some(error) = &block.timing_error {
            return Err(format!("block '{}': {error}", block.name).into());
        }
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

/// A BX register, its physical reset signal, and any storage type needed to
/// encode empty-array leaves of its reset value.
struct Register {
    handle: BRegister,
    reset: Option<BValue>,
    storage_type: Option<ir::Type>,
}

/// Hardware resources used to wire register and instance inputs in one block.
#[derive(Default)]
struct BlockResources {
    registers: BTreeMap<String, Register>,
    instances: BTreeMap<String, BInstantiation>,
    cycle_sources: CycleSources,
}

/// The destination package and symbols reserved for user definitions and
/// generated loop and map bodies.
struct Destination<'p> {
    batch: FunctionBatch<'p>,
    occupied: BTreeSet<String>,
    next_body: usize,
    evaluate_test_discards: bool,
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
    empty_array_budget: &'a mut EmptyArrayBudget,
}

/// Bounds empty-array lowering for a source body and its generated helpers.
/// Concrete specializations share one budget per kind across the module.
struct EmptyArrayBudget {
    scope: &'static str,
    limit: usize,
    used: usize,
}

impl EmptyArrayBudget {
    fn new(scope: &'static str, limit: usize) -> Self {
        Self {
            scope,
            limit,
            used: 0,
        }
    }

    /// Charge constructors and wire conversions before allocating their nodes.
    fn charge(&mut self, work: usize) -> Result<(), BuilderError> {
        self.used = self.used.saturating_add(work);
        if self.used > self.limit {
            return Err(BuilderError::InvalidOperation(format!(
                "empty array lowering exceeds the {} expansion limit of {}",
                self.scope, self.limit,
            )));
        }
        Ok(())
    }
}

/// XLS requires a typed array constructor for an empty array. In a
/// homogeneous array the first element determines whether its type contains
/// one.
fn contains_empty_array(value: &IrValue) -> bool {
    match value {
        IrValue::Array(array) => array.elements().first().is_none_or(contains_empty_array),
        IrValue::Tuple(fields) => fields.iter().any(contains_empty_array),
        IrValue::Bits(_) | IrValue::Token => false,
    }
}

/// Owned IR type nodes and lowering work for an empty-array constant and one
/// register wire of its shape. A matching wire subtree needs no adaptation.
struct EmptyArrayExpansion {
    type_nodes: usize,
    literal_work: usize,
    wire_work: usize,
}

impl EmptyArrayExpansion {
    /// Count each tuple field and one representative of each homogeneous
    /// array. Include type trees that PIR constructs or copies alongside the
    /// emitted nodes, even for a field forwarded as a single literal or wire.
    fn for_value(value: &IrValue) -> Self {
        match value {
            IrValue::Array(array) => {
                let Some(first) = array.elements().first() else {
                    // The array owns its element type even when it is empty.
                    let type_nodes = 2 + literal_copy_work(value);
                    return Self {
                        type_nodes,
                        literal_work: type_nodes,
                        wire_work: type_nodes,
                    };
                };
                let child = Self::for_value(first);
                let type_nodes = 1 + child.type_nodes;
                if child.wire_work == 0 {
                    return Self {
                        type_nodes,
                        // Cloning the array and constructing its literal type
                        // each copy its owned element layout.
                        literal_work: 2 * type_nodes,
                        wire_work: 0,
                    };
                }
                let count = array.elements().len();
                // An array constructor gathers the element types and builds
                // the result type. Each array_index copies the array type
                // during operand gathering and result deduction.
                let constructor = count * child.type_nodes + 2 * type_nodes;
                let projection = 2 + 2 * type_nodes + child.type_nodes;
                Self {
                    type_nodes,
                    literal_work: count * child.literal_work + constructor,
                    wire_work: count * (child.wire_work + projection) + constructor,
                }
            }
            IrValue::Tuple(fields) => {
                let mut type_nodes = 1;
                let mut literal_work = 0;
                let mut wire_work = 0;
                for field in fields.iter() {
                    let child = Self::for_value(field);
                    type_nodes += child.type_nodes;
                    literal_work += child.literal_work;
                    wire_work += child.wire_work;
                }
                if wire_work == 0 {
                    // Tuple values share their storage, but builder.literal
                    // still materializes their full result type.
                    return Self {
                        type_nodes,
                        literal_work: type_nodes,
                        wire_work: 0,
                    };
                }
                // The tuple constructor gathers the field types and copies
                // them into its result. A wire also projects each field.
                Self {
                    type_nodes,
                    literal_work: literal_work + 2 * type_nodes,
                    wire_work: wire_work + 3 * type_nodes,
                }
            }
            IrValue::Bits(_) | IrValue::Token => Self {
                type_nodes: 1,
                literal_work: 1 + literal_copy_work(value),
                wire_work: 0,
            },
        }
    }
}

/// Replace each empty array in a reset literal with its zero-bit storage
/// placeholder, preserving the other fields and array elements.
fn encode_register_reset(value: &IrValue) -> Result<IrValue, BuilderError> {
    if !contains_empty_array(value) {
        return Ok(value.clone());
    }
    match value {
        IrValue::Array(array) if array.elements().is_empty() => Ok(IrValue::Bits(IrBits::zero(0))),
        IrValue::Array(array) => {
            let elements = array
                .elements()
                .iter()
                .map(encode_register_reset)
                .collect::<Result<Vec<_>, _>>()?;
            IrValue::make_array(&elements)
                .map_err(|e| BuilderError::InvalidOperation(e.to_string()))
        }
        IrValue::Tuple(fields) => Ok(IrValue::Tuple(
            fields
                .iter()
                .map(encode_register_reset)
                .collect::<Result<_, _>>()?,
        )),
        _ => unreachable!("only aggregates can contain an empty array"),
    }
}

/// Builds a local function in the destination package's callee and node-ID
/// context.
fn lower_function(
    function: &CheckedFunction,
    symbol: &str,
    destination: &mut Destination,
    functions: &LinkedFunctions,
    layouts: &mut Layouts,
    budget: &mut EmptyArrayBudget,
) -> Result<usize, BuilderError> {
    let mut body = LowerBody::new(
        FnBuilder::new(symbol),
        destination,
        functions,
        layouts,
        budget,
    );
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
        mut self,
        block: &CheckedBlock,
        symbols: &BlockSymbols,
        reset_behavior: ResetBehavior,
    ) -> Result<(BlockBuilder, CycleSources), BuilderError> {
        let mut resources = BlockResources::default();
        self.builder.clock_port(&block.clock)?;
        let reset = if let Some(name) = &block.reset {
            let physical = self.builder.input_port(name, ir::Type::Bits(1))?;
            self.builder.set_reset(physical, reset_behavior)?;
            let asserted = if reset_behavior.active_low {
                self.builder.not(physical)?
            } else {
                physical
            };
            self.values.insert(name.clone(), asserted);
            Some(physical)
        } else {
            None
        };
        for param in &block.params {
            let value = self
                .builder
                .input_port(&param.name, self.layouts.lower(&param.ty)?)?;
            self.values.insert(param.name.clone(), value);
        }
        for statement in &block.statements {
            if let CheckedStmt::Inst {
                name,
                block: target,
            } = statement
            {
                let instance = self.add_instance(name, symbols.get(target), reset)?;
                resources.instances.insert(name.clone(), instance);
            }
        }
        self.wire_statements(&block.statements, &mut resources, reset, None)?;
        let ports = &block.output_ports;
        if let TypedExprKind::Tuple(fields) = &block.output.kind {
            // Drive fields separately to preserve their feedback dependencies.
            for (port, field) in ports.iter().zip(fields) {
                let value = self.expr(field)?;
                self.builder.output_port(&port.name, value)?;
            }
            return Ok((self.builder, resources.cycle_sources));
        }
        let value = self.expr(&block.output)?;
        for (index, port) in ports.iter().enumerate() {
            let field = match block.result {
                Type::Named(_) | Type::Tuple(_) => self.project(value, index)?,
                Type::Bits { .. } | Type::Enum { .. } | Type::Bool | Type::Array { .. } => value,
                Type::ParametricBits { .. }
                | Type::ParametricArray { .. }
                | Type::Applied { .. }
                | Type::ParametricStruct { .. }
                | Type::TypeApplied { .. } => {
                    return Err(BuilderError::InvalidOperation(format!(
                        "cannot lower block result with unresolved type '{}'",
                        block.result
                    )));
                }
            };
            self.builder.output_port(&port.name, field)?;
        }
        Ok((self.builder, resources.cycle_sources))
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
                    let ty = self.layouts.lower(ty)?;
                    let mut storage_type = None;
                    let value = match value {
                        Some(value) if contains_empty_array(value) => {
                            if ty.bit_count() == 0 {
                                // A zero-bit register has only one possible
                                // value.
                                None
                            } else {
                                // Each checked register has one read and one
                                // write.
                                let expansion = EmptyArrayExpansion::for_value(value);
                                self.empty_array_budget
                                    .charge(2 * expansion.wire_work + expansion.literal_work)?;
                                let encoded = encode_register_reset(value)?;
                                storage_type = Some(encoded.type_());
                                Some(encoded)
                            }
                        }
                        _ => value.clone(),
                    };
                    let register_reset = value.as_ref().and(reset);
                    let handle = self.builder.register(
                        name,
                        storage_type.as_ref().unwrap_or(&ty).clone(),
                        value,
                    )?;
                    let current = self.builder.register_read(handle)?;
                    let current = if storage_type.is_some() {
                        self.adapt_register_wire(current, &ty)?
                    } else {
                        current
                    };
                    self.values.insert(name.clone(), current);
                    resources.registers.insert(
                        name.clone(),
                        Register {
                            handle,
                            reset: register_reset,
                            storage_type,
                        },
                    );
                }
                CheckedStmt::Let { name, value } => {
                    let value = self.expr(value)?;
                    self.values.insert(name.clone(), value);
                }
                CheckedStmt::Wire { name, ty, .. } => {
                    let wire = self.builder.declare_wire(self.layouts.lower(ty)?)?;
                    self.values.insert(name.clone(), wire);
                }
                CheckedStmt::DriveWire { name, value, span } => {
                    let wire = self.values[name];
                    let value = self.expr(value)?;
                    self.builder.drive_wire(wire, value)?;
                    if let Some(span) = span {
                        let id = self.builder.value_id(wire)?;
                        resources.cycle_sources.wires.insert(
                            id,
                            WireDriver {
                                name: name.clone(),
                                span: *span,
                            },
                        );
                    }
                }
                CheckedStmt::Write { name, value } => {
                    let reg = &resources.registers[name];
                    let data = self.expr(value)?;
                    let data = if let Some(ty) = &reg.storage_type {
                        self.adapt_register_wire(data, ty)?
                    } else {
                        data
                    };
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
                    span,
                } => {
                    let value = self.expr(value)?;
                    let sink = self.builder.instantiation_input(
                        resources.instances[instance],
                        port,
                        value,
                    )?;
                    if let Some(span) = span {
                        let id = self.builder.value_id(sink)?;
                        resources.cycle_sources.inputs.insert(id, *span);
                    }
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
        empty_array_budget: &'a mut EmptyArrayBudget,
    ) -> Self {
        Self {
            builder,
            destination,
            functions,
            layouts,
            values: BTreeMap::new(),
            tuple_elements: HashMap::new(),
            empty_array_budget,
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

    /// Adapt a wire between a declared register layout and its reset storage
    /// layout, forwarding each subtree whose type already matches.
    fn adapt_register_wire(
        &mut self,
        value: BValue,
        target: &ir::Type,
    ) -> Result<BValue, BuilderError> {
        if self.builder.get_type(value)? == target {
            return Ok(value);
        }
        match target {
            ir::Type::Bits(0) => self.builder.literal(IrValue::Bits(IrBits::zero(0))),
            ir::Type::Array(array) if array.element_count == 0 => {
                self.builder.array((*array.element_type).clone(), &[])
            }
            ir::Type::Tuple(types) => {
                let mut fields = Vec::with_capacity(types.len());
                for (index, ty) in types.iter().enumerate() {
                    let field = self.project(value, index)?;
                    fields.push(self.adapt_register_wire(field, ty)?);
                }
                let tuple = self.builder.tuple(&fields)?;
                self.tuple_elements.insert(tuple, fields);
                Ok(tuple)
            }
            ir::Type::Array(array) => {
                let mut elements = Vec::with_capacity(array.element_count);
                for i in 0..array.element_count {
                    let index = self.builder.literal(IrValue::u32(i as u32))?;
                    let element = self.builder.array_index(value, index)?;
                    elements.push(self.adapt_register_wire(element, &array.element_type)?);
                }
                self.builder.array((*array.element_type).clone(), &elements)
            }
            _ => unreachable!("register layouts only differ at empty-array leaves"),
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
        let subject_value = self.expr(subject)?;
        let mut predicates = Vec::new();
        let mut cases = Vec::new();
        for arm in arms {
            if arm.patterns.is_empty() {
                // Every source alternative in this arm was an empty range.
                continue;
            }
            if arm.patterns.iter().any(TypedMatchPattern::is_irrefutable) {
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
                alternatives.push(self.lower_match_pattern(pattern, subject_value, &subject.ty)?);
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

    /// Combine the predicates of constrained tuple members and compare scalar
    /// leaves.
    fn lower_match_pattern(
        &mut self,
        pattern: &TypedMatchPattern,
        subject: BValue,
        ty: &Type,
    ) -> Result<BValue, BuilderError> {
        match pattern {
            TypedMatchPattern::Wildcard => {
                unreachable!("irrefutable pattern requires no predicate")
            }
            TypedMatchPattern::Literal(bits) => {
                let literal = self.builder.literal(IrValue::Bits(bits.clone()))?;
                self.builder.eq(subject, literal)
            }
            TypedMatchPattern::Range {
                start,
                end,
                inclusive,
            } => {
                let (signed, _) = repr_bits(ty).expect("checked range subject is scalar");
                let start = self.builder.literal(IrValue::Bits(start.clone()))?;
                let end = self.builder.literal(IrValue::Bits(end.clone()))?;
                let ge = self.binary(BinaryOp::Ge, signed, subject, start)?;
                let op = if *inclusive {
                    BinaryOp::Le
                } else {
                    BinaryOp::Lt
                };
                let upper = self.binary(op, signed, subject, end)?;
                self.builder.and(ge, upper)
            }
            TypedMatchPattern::Tuple(patterns) => {
                let Type::Tuple(types) = ty else {
                    unreachable!("checked tuple pattern requires a tuple subject");
                };
                let mut predicates = Vec::new();
                for (index, (pattern, ty)) in patterns.iter().zip(types).enumerate() {
                    if !pattern.is_irrefutable() {
                        let member = self.builder.tuple_index(subject, index)?;
                        predicates.push(self.lower_match_pattern(pattern, member, ty)?);
                    }
                }
                if let [only] = predicates.as_slice() {
                    Ok(*only)
                } else {
                    let bits = self.builder.concat(&predicates)?;
                    self.builder.and_reduce(bits)
                }
            }
        }
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
            if !self.destination.evaluate_test_discards && binding.is_test_discard() {
                continue;
            }
            let value = self.expr(&binding.value)?;
            self.values.insert(binding.name.clone(), value);
        }
        self.expr(output)
    }

    /// Emit a unary IR map for a closed lambda or a counted loop carrying
    /// the result array when the lambda reads enclosing wires.
    fn map(&mut self, map: &TypedMap) -> Result<BValue, BuilderError> {
        let array = self.expr(&map.array)?;
        if let Some(target) = self
            .functions
            .forwarded_map_target(map, self.destination.evaluate_test_discards)
        {
            return self.builder.map_with_signature(array, target);
        }
        let symbol = self.destination.body_symbol("map");
        let mut builder = FnBuilder::new(&symbol);
        let element = builder.param("element", self.layouts.lower(&map.element.ty)?)?;
        let mut capture_wires = Vec::new();
        let captures = self.capture_params(&mut builder, &map.captures, &mut capture_wires)?;
        let mut body = LowerBody::new(
            builder,
            self.destination,
            self.functions,
            self.layouts,
            self.empty_array_budget,
        );
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
        captures: &BTreeMap<String, CaptureUse>,
        invariants: &mut Vec<BValue>,
    ) -> Result<BTreeMap<String, BValue>, BuilderError> {
        let start = invariants.len();
        let names = captures
            .iter()
            .filter(|(_, usage)| {
                self.destination.evaluate_test_discards || **usage == CaptureUse::Value
            })
            .map(|(name, _)| name);
        invariants.extend(names.clone().map(|name| self.values[name]));
        let params = self.wire_params(builder, &invariants[start..])?;
        Ok(names.cloned().zip(params).collect())
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
        let mut body = LowerBody::new(
            builder,
            self.destination,
            self.functions,
            self.layouts,
            self.empty_array_budget,
        );
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
            TypedExprKind::Literal(value) => self.literal(value),
            TypedExprKind::For(fold) => self.counted_for(fold),
            TypedExprKind::Map(map) => self.map(map),
            TypedExprKind::Array(items) => {
                let Type::Array { element, size } = &expr.ty else {
                    return Err(BuilderError::InvalidOperation(
                        "array expression requires an array type".into(),
                    ));
                };
                let mut values = items
                    .iter()
                    .map(|item| self.expr(item))
                    .collect::<Result<Vec<_>, _>>()?;
                if values.len() < *size {
                    let last = values.last().ok_or_else(|| {
                        BuilderError::InvalidOperation("array fill requires a value".into())
                    })?;
                    values.resize(*size, *last);
                }
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
            TypedExprKind::Builtin { op, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                op.emit(&mut self.builder, &args)
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
                let lhs = self.expr(left)?;
                let rhs = self.expr(right)?;
                if *op == BinaryOp::Concat && matches!(expr.ty, Type::Array { .. }) {
                    self.builder.array_concat(&[lhs, rhs])
                } else {
                    let signed = matches!(left.ty, Type::Bits { signed: true, .. });
                    self.binary(*op, signed, lhs, rhs)
                }
            }
            TypedExprKind::Cast { value } => self.cast(value, &expr.ty),
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

    fn literal(&mut self, value: &IrValue) -> Result<BValue, BuilderError> {
        if !contains_empty_array(value) {
            return self.builder.literal(value.clone());
        }
        self.empty_array_budget
            .charge(EmptyArrayExpansion::for_value(value).literal_work)?;
        self.literal_with_constructors(value)
    }

    /// Use typed IR constructors for constant aggregates containing empty
    /// arrays so the emitted package can be parsed by XLS.
    fn literal_with_constructors(&mut self, value: &IrValue) -> Result<BValue, BuilderError> {
        if !contains_empty_array(value) {
            return self.builder.literal(value.clone());
        }
        match value {
            IrValue::Array(array) => {
                let elements = array
                    .elements()
                    .iter()
                    .map(|e| self.literal_with_constructors(e))
                    .collect::<Result<Vec<_>, _>>()?;
                self.builder.array(array.element_type().clone(), &elements)
            }
            IrValue::Tuple(fields) => {
                let fields = fields
                    .iter()
                    .map(|field| self.literal_with_constructors(field))
                    .collect::<Result<Vec<_>, _>>()?;
                self.builder.tuple(&fields)
            }
            _ => unreachable!("only aggregates can contain an empty array"),
        }
    }

    /// Resize a scalar or convert an equal-width array with element zero at
    /// the most significant end of the packed bits.
    fn cast(&mut self, value: &TypedExpr, target: &Type) -> Result<BValue, BuilderError> {
        let src = self.expr(value)?;
        if let Type::Array { element, size } = target {
            let element_type = self.layouts.lower(element)?;
            let width = element_type.bit_count();
            let slices = if width == 0 {
                // Each element is the same zero-bit input wire.
                vec![src; *size]
            } else {
                (0..*size)
                    .rev()
                    .map(|i| self.builder.bit_slice(src, i * width, width))
                    .collect::<Result<Vec<_>, _>>()?
            };
            return self.builder.array(element_type, &slices);
        }
        let (_, dst_width) = repr_bits(target).expect("checked scalar cast target");
        if let Type::Array { size, .. } = &value.ty {
            if dst_width == 0 {
                return self.builder.literal(IrValue::Bits(IrBits::zero(0)));
            }
            let mut pieces = Vec::with_capacity(*size);
            for i in 0..*size {
                // Concrete BX arrays fit within the DSLX u32 index type.
                let index = self.builder.literal(IrValue::u32(i as u32))?;
                pieces.push(self.builder.array_index(src, index)?);
            }
            return self.builder.concat(&pieces);
        }
        let (signed, src_width) = repr_bits(&value.ty).expect("checked scalar cast source");
        if src_width == dst_width {
            Ok(src)
        } else if dst_width < src_width {
            self.builder.bit_slice(src, 0, dst_width)
        } else if signed && src_width > 0 {
            self.builder.sign_extend(src, dst_width)
        } else {
            self.builder.zero_extend(src, dst_width)
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

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use super::{EmptyArrayExpansion, lower_checked_module};
    use crate::{ResetBehavior, check_module, empty_package, lower_module, parse_module};

    #[test]
    fn empty_array_expansion_follows_source_and_specialization_scopes() -> Result<(), String> {
        let constants = r#"
const E = bits[0]:0 as u1[0];
const A = [(E, u1:0), (E, u1:1)];
"#;
        for (bodies, exhausted_scope) in [
            (
                r#"
fn f<N: u32>(x: u1) -> u1 { A[x].1 }
fn both(x: u1) -> u1 { f<1>(x) ^ f<2>(x) }
"#,
                Some("function specialization"),
            ),
            (
                r#"
#[allow(combinational_block)]
block stage<T>(clk, x: T) -> u1 { A[x as u1].1 }
block both(clk, x: u1, y: u2) -> u1 {
    let a = stage(x);
    let b = stage(y);
    a ^ b
}
"#,
                Some("generic block specialization"),
            ),
            (
                r#"
fn both(xs: u1[2]) -> (u1[2], u1) {
    let m = map(xs, |i| A[i].1);
    let f = for (i, acc) in xs { acc ^ A[i].1 }(u1:0);
    (m, f)
}
"#,
                Some("body"),
            ),
            (
                r#"
fn f<N: u32>(x: u1) -> u1 { A[x].1 }
fn a(x: u1) -> u1 { f<1>(x) ^ A[x].1 }
fn b(x: u1) -> u1 { A[x].1 }
#[allow(combinational_block)]
block stage<T>(clk, x: T) -> u1 { A[x as u1].1 }
#[allow(combinational_block)]
block ordinary(clk, x: u1) -> u1 { A[x].1 }
block both(clk, x: u1) -> u1 {
    let a = stage(x);
    a ^ A[x].1
}
"#,
                None,
            ),
        ] {
            let checked = check_module(&parse_module(&format!("{constants}{bodies}"))?)?;
            // One materialization fits exactly in each independent scope.
            let limit = EmptyArrayExpansion::for_value(&checked.constants()[1].value).literal_work;
            let result = lower_checked_module(
                &checked,
                empty_package("design"),
                ResetBehavior::default(),
                limit,
            )
            .map_err(|error| error.0);
            if let Some(scope) = exhausted_scope {
                let error = result.unwrap_err();
                assert!(
                    error.contains(&format!(
                        "empty array lowering exceeds the {scope} expansion limit of {limit}"
                    )),
                    "{error}"
                );
            } else {
                result?;
            }
        }
        Ok(())
    }

    #[test]
    fn empty_array_expansion_charges_nested_plain_tuple_layouts() -> Result<(), String> {
        // Sharing keeps the checked constant small. Each emitted literal or
        // projection still needs its own copy of the 767-node tuple type.
        let mut source = String::from("struct S0 { a: u1 }\n");
        for i in 1..=8 {
            let child = i - 1;
            writeln!(source, "struct S{i} {{ a: S{child}, b: S{child} }}").unwrap();
        }
        source.push_str("const A0 = (bits[0]:0 as u1[0], zero!<S8>());\n");
        for i in 1..=8 {
            let child = i - 1;
            writeln!(source, "const A{i} = [A{child}, A{child}];").unwrap();
        }
        // At seven array dimensions the literal fits, while adapting the
        // register read and write takes the combined cost over the limit.
        let ty = format!("(u1[0], S8){}", "[2]".repeat(7));
        for body in [
            "#[allow(combinational_block)] block b(clk) -> u1 { let _v = A8; u1:0 }".into(),
            format!("block b(clk, rst) -> u1 {{ reg q: {ty} = A7; q := q; u1:0 }}"),
        ] {
            let checked = check_module(&parse_module(&format!("{source}{body}"))?)?;
            let error = lower_module(&checked, empty_package("design"), ResetBehavior::default())
                .unwrap_err();
            assert!(
                error.contains("empty array lowering exceeds the body expansion limit"),
                "{error}"
            );
        }
        Ok(())
    }
}
