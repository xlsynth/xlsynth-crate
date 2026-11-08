// SPDX-License-Identifier: Apache-2.0

//! BX name resolution, value type checking, and structural driver checks.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::rc::Rc;
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::Type as IrType;
use xlsynth_pir::ir_value_utils::{ir_bits_to_usize, zero_ir_value_for_type};

use crate::ImportOptions;
use crate::ast::{
    BinaryOp, Block, CallbackParam, Expr, ExternBlock, ExternFn, Function, LetBinding, MatchArm,
    MatchPattern, Module, Param, Stmt, StructDef, Type, UnaryOp, WidthSliceType,
};
use crate::dslx_import::ImportResolver;
use crate::infer_literal::{
    CommonTypeHint, ContextOperands, InferredBindings, WidthLookup, binding_widths,
    check_argument_widths, check_match_operand_width, check_result_width, common_type_hint,
    context_operands, has_common_bit_type, has_concrete_element_type, minimum_operand_type,
    minimum_type, projected_type_compatible, unify_inferred_members,
};
use crate::parser::{character_or_bool_digits, explicit_width_hint};
mod callables;
mod counted_for;
mod function_specializations;
mod generic_blocks;
mod generic_inputs;
mod map;
mod match_patterns;
mod module_constants;
mod scope;
mod unroll;

pub use crate::typed::{
    CheckedBinding, CheckedBlock, CheckedConstant, CheckedFunction, CheckedModule, CheckedStmt,
    ScalarPattern, TypedExpr, TypedExprKind, TypedFor, TypedIterable, TypedMap, TypedMatchArm,
};
use crate::typed::{ConstantBinding, ConstantFailure, ConstantFunctions, TypedExprKind as K, bits};
use callables::{BlockCallbackBinding, BlockCallbackFormal, BoundCallback};
use counted_for::CarryContext;
use function_specializations::{CallbackTarget, FunctionSpecialization, FunctionSpecializations};
use generic_blocks::{Bindings, TypeArgument};

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Bits { signed, width } => {
                let prefix = if *signed { "s" } else { "u" };
                if (1..=64).contains(width) {
                    write!(f, "{prefix}{width}")
                } else {
                    write!(f, "{prefix}N[{width}]")
                }
            }
            Type::Bool => write!(f, "bool"),
            Type::Named(name) => write!(f, "{name}"),
            Type::Array { element, size } => write!(f, "{element}[{size}]"),
            Type::Tuple(items) => {
                write!(f, "(")?;
                for (i, ty) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                if items.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Add a bit-width spelling hint only after nominal name resolution fails.
fn unknown_declaration(kind: &str, name: &str) -> String {
    let error = format!("unknown {kind} '{name}'");
    if kind == "struct"
        && let Some(hint) = explicit_width_hint(name)
    {
        return format!("{error} ({hint})");
    }
    error
}

/// Choose the physical output ports from the source result's outermost type.
pub(crate) fn output_ports(result: &Type, structs: &[StructDef]) -> Result<Vec<Param>, String> {
    match result {
        Type::Named(name) => structs
            .iter()
            .find(|s| s.name == *name)
            .map(|s| s.fields.clone())
            .ok_or_else(|| unknown_declaration("struct", name)),
        Type::Tuple(items) => Ok(items
            .iter()
            .enumerate()
            .map(|(i, ty)| Param {
                name: format!("out_{i}"),
                ty: ty.clone(),
            })
            .collect()),
        _ => Ok(vec![Param {
            name: "out".into(),
            ty: result.clone(),
        }]),
    }
}

/// Aggregate constructors share member types with the other operand or branch.
fn has_aggregate_literal(expr: &Expr) -> bool {
    match expr {
        Expr::For(fold) if fold.annotation.is_none() => has_aggregate_literal(&fold.seed),
        Expr::Scope { result, .. } => has_aggregate_literal(result),
        Expr::Array(_) | Expr::Tuple(_) => true,
        Expr::If {
            then_value,
            else_value,
            ..
        } => has_aggregate_literal(then_value) || has_aggregate_literal(else_value),
        Expr::Match { arms, .. } => arms.iter().any(|arm| has_aggregate_literal(&arm.value)),
        _ => false,
    }
}

/// A match subject and its patterns, checked independently of the result
/// group. Each pattern group consumes exactly one checked arm value.
struct MatchSelector {
    subject: TypedExpr,
    patterns: Vec<Vec<ScalarPattern>>,
}

impl MatchSelector {
    fn with_values(self, values: &mut impl Iterator<Item = TypedExpr>) -> TypedExpr {
        let arms: Vec<_> = self
            .patterns
            .into_iter()
            .map(|patterns| TypedMatchArm {
                patterns,
                value: values.next().expect("one checked result per match arm"),
            })
            .collect();
        TypedExpr {
            ty: arms[0].value.ty.clone(),
            kind: K::Match {
                subject: Box::new(self.subject),
                arms,
            },
        }
    }
}

/// Retains each selector while nested branch results are inferred together.
enum BranchSelector {
    Value,
    If(TypedExpr),
    Match(MatchSelector),
}

/// Collect the subject and literal constraints in source order. AST callers
/// can construct patterns directly, including inside an empty unrolled loop.
fn match_operands<'e>(
    subject: &'e Expr,
    arms: &'e [MatchArm],
    is_module_constant: impl Fn(&str) -> bool,
) -> Result<Vec<&'e Expr>, String> {
    if arms.is_empty() || arms.iter().any(|arm| arm.patterns.is_empty()) {
        return Err("match requires at least one arm and a pattern for each arm".into());
    }
    let mut operands = vec![subject];
    for pattern in arms.iter().flat_map(|arm| &arm.patterns) {
        for literal in pattern.literals() {
            if !matches!(literal, Expr::Bool(_) | Expr::Literal { .. })
                && !matches!(pattern, MatchPattern::Literal(Expr::Name(name)) if is_module_constant(name))
            {
                return Err(
                    "scalar match patterns require literal values or a visible module constant"
                        .into(),
                );
            }
            operands.push(literal);
        }
    }
    Ok(operands)
}

/// DSLX's bool and u1 are the same unsigned bit type, including in tuples.
fn compatible(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Tuple(x), Type::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| compatible(x, y))
        }
        (Type::Named(x), Type::Named(y)) => x == y,
        (
            Type::Array {
                element: x,
                size: n,
            },
            Type::Array {
                element: y,
                size: m,
            },
        ) => n == m && compatible(x, y),
        _ => bits(a).is_some_and(|t| Some(t) == bits(b)),
    }
}

/// Data arguments, output wires, and reset requirement of a local or linked
/// child.
struct BlockInterface {
    params: Vec<Param>,
    result: Type,
    outputs: Vec<Param>,
    requires_reset: bool,
    /// Concrete type nodes for wiring one instance of this interface.
    port_type_nodes: usize,
}

/// Output wires and the cost of materializing the checked data ports.
struct BlockPortLayout {
    outputs: Vec<Param>,
    type_nodes: usize,
}

/// A source block with symbolic data ports and a result in its declaration
/// scope.
struct BlockTemplate<'m> {
    block: &'m Block,
    params: Vec<Param>,
    result: Type,
    callback_params: Vec<BlockCallbackFormal>,
    callback_indices: BTreeMap<String, usize>,
}

/// The concrete interface shared by instances with identical type bindings.
struct BlockSpecialization<'m> {
    block: &'m Block,
    name: String,
    interface: BlockInterface,
    bindings: Bindings,
    callbacks: Vec<Rc<BoundCallback<'m>>>,
    path: Vec<&'m str>,
}

/// Required dimensions and reset constants retain their own counted-loop
/// evaluation allowance when speculative propagation exhausts its allowance.
struct ConstantBudgets {
    speculative: usize,
    required: usize,
}

impl Default for ConstantBudgets {
    fn default() -> Self {
        Self {
            speculative: MAX_TYPE_SIZE,
            required: MAX_TYPE_SIZE,
        }
    }
}

/// Lexical constants and their checked definitions share the active body's
/// budgets. Statements precede every initializer that can depend on them.
struct ConstantContext<'a, 'f> {
    names: &'a mut BTreeMap<String, ConstantBinding>,
    statements: &'a [CheckedStmt],
    budgets: &'a mut ConstantBudgets,
    /// Module initializers and specialized functions charge intermediate
    /// values outside loops as well as in their bodies.
    charge_values: bool,
    functions: Option<&'a mut ConstantFunctions<'f>>,
}

/// The concrete source types and directly selected functions of a block.
#[derive(Eq, Hash, PartialEq)]
struct BlockKey<'m> {
    name: &'m str,
    types: Vec<TypeArgument>,
    callbacks: Vec<CallbackTarget<'m>>,
}

/// Interns specialized block interfaces as calls are checked. Checking queued
/// bodies iteratively also discovers calls inside those bodies.
struct BlockSpecializations<'m> {
    cache: HashMap<BlockKey<'m>, Rc<BlockSpecialization<'m>>>,
    validated_callbacks: HashMap<BlockCallbackBinding<'m>, Rc<BoundCallback<'m>>>,
    pending: VecDeque<Rc<BlockSpecialization<'m>>>,
    reserved: BTreeSet<String>,
    next_id: usize,
    expansion_work: usize,
    reference_copy_work: Cell<usize>,
    constant_budgets: ConstantBudgets,
}

impl<'m> BlockSpecializations<'m> {
    fn new(reserved: BTreeSet<String>) -> Self {
        Self {
            cache: HashMap::new(),
            validated_callbacks: HashMap::new(),
            pending: VecDeque::new(),
            reserved,
            next_id: 0,
            expansion_work: 0,
            reference_copy_work: Cell::new(0),
            constant_budgets: ConstantBudgets::default(),
        }
    }

    /// Assign one name and interface per binding tuple. Recursive calls can
    /// otherwise request a different (and growing) payload type at each step.
    fn intern(
        &mut self,
        template: &BlockTemplate<'m>,
        bindings: Bindings,
        callbacks: Vec<Rc<BoundCallback<'m>>>,
        path: &[&'m str],
        structs: &StructTable,
    ) -> Result<Rc<BlockSpecialization<'m>>, String> {
        let block = template.block;
        let name = block.name.as_str();
        if path.contains(&name) {
            return Err(format!("recursive block dependency involving '{name}'"));
        }
        if path.len() >= MAX_ELABORATION_DEPTH {
            return Err(format!(
                "block dependency depth exceeds {MAX_ELABORATION_DEPTH} levels at '{name}'"
            ));
        }
        let key = BlockKey {
            name,
            types: bindings.arguments(&block.type_params)?,
            callbacks: callbacks.iter().map(|bound| bound.target).collect(),
        };
        if let Some(prior) = self.cache.get(&key) {
            return Ok(Rc::clone(prior));
        }
        // An acyclic template graph can still request exponentially many
        // distinct specializations. Repeated instances of one type share this
        // budget entry.
        const MAX_SPECIALIZATIONS: usize = 4096;
        if self.cache.len() >= MAX_SPECIALIZATIONS {
            return Err(format!(
                "generic block specialization limit of {MAX_SPECIALIZATIONS} exceeded"
            ));
        }
        // Distinct callback identities can replicate the same wide ports.
        // Charge the fully bound interface before cloning its concrete types.
        if !callbacks.is_empty() {
            for ty in template
                .params
                .iter()
                .map(|p| &p.ty)
                .chain([&template.result])
            {
                let shape = structs.formal_shape(ty, MAX_ELABORATION_DEPTH, Some(&bindings))?;
                self.expansion_work = self.expansion_work.saturating_add(shape.ir_nodes);
                if self.expansion_work > MAX_TYPE_SIZE {
                    return Err(format!(
                        "block signature exceeds the generic block specialization expansion limit of {MAX_TYPE_SIZE}"
                    ));
                }
            }
        }
        let params: Vec<Param> = template
            .params
            .iter()
            .map(|p| {
                Ok(Param {
                    name: p.name.clone(),
                    ty: bindings.concrete_type(&p.ty)?,
                })
            })
            .collect::<Result<_, String>>()?;
        let result = bindings.concrete_type(&template.result)?;
        let interface = BlockInterface::new(
            &block.clock,
            block.reset.as_deref(),
            params,
            result,
            structs,
        )?;
        let symbol = loop {
            let candidate = format!("{name}__{}", self.next_id);
            self.next_id += 1;
            if self.reserved.insert(candidate.clone()) {
                break candidate;
            }
        };
        let mut path = path.to_vec();
        path.push(name);
        let spec = Rc::new(BlockSpecialization {
            block,
            name: symbol,
            interface,
            bindings,
            callbacks,
            path,
        });
        self.cache.insert(key, Rc::clone(&spec));
        self.pending.push_back(Rc::clone(&spec));
        Ok(spec)
    }
}

/// Resolved signature used to check both local and externally linked calls.
struct FunctionInterface {
    params: Vec<Param>,
    result: Type,
    callback_params: Vec<CallbackFormal>,
    /// Lookup in the body's lexical scope while preserving declaration order
    /// in the specialization key.
    callback_indices: BTreeMap<String, usize>,
    local: bool,
}

/// A resolved callback signature and the cost of validating a new actual
/// against it. The declaration order determines the cache's formal index.
struct CallbackFormal {
    signature: CallbackParam,
    binding_work: usize,
}

/// Nodes visited when comparing or cloning a checked BX source type.
/// A nominal reference is a single name and an array carries one element type.
fn source_type_nodes(ty: &Type) -> usize {
    match ty {
        Type::Tuple(fields) => 1 + fields.iter().map(source_type_nodes).sum::<usize>(),
        Type::Array { element, .. } => 1 + source_type_nodes(element),
        Type::Named(_) | Type::Bool | Type::Bits { .. } => 1,
    }
}

/// Storage copied when cloning a literal: wide bits own their limbs, and an
/// array owns its IR element type. Tuple and array element values are shared.
fn literal_copy_work(value: &IrValue) -> usize {
    fn descendants(ty: &IrType) -> usize {
        match ty {
            IrType::Tuple(fields) => fields.iter().map(|ty| 1 + descendants(ty)).sum(),
            IrType::Array(array) => 1 + descendants(&array.element_type),
            IrType::Token | IrType::Bits(_) => 0,
        }
    }
    match value {
        IrValue::Bits(bits) if bits.get_bit_count() > 64 => bits.get_bit_count().div_ceil(64),
        IrValue::Array(array) => descendants(array.element_type()),
        _ => 0,
    }
}

/// Source declaration indices for lookup during checking.
#[derive(Default)]
struct Declarations<'a> {
    positions: BTreeMap<&'a str, usize>,
    constants: BTreeMap<&'a str, CheckedConstant>,
    functions: BTreeMap<&'a str, FunctionInterface>,
    function_templates: BTreeMap<&'a str, &'a Function>,
    blocks: BTreeMap<&'a str, BlockInterface>,
    templates: BTreeMap<&'a str, BlockTemplate<'a>>,
}

impl Declarations<'_> {
    fn is_block(&self, name: &str) -> bool {
        self.blocks.contains_key(name) || self.templates.contains_key(name)
    }
    /// Qualified imports are visible from their use binding; qualified
    /// externs are introduced by their full source name.
    fn visible(&self, name: &str, at: Option<usize>) -> bool {
        let root = name.split_once("::").map_or(name, |(root, _)| root);
        let position = self
            .positions
            .get(name)
            .or_else(|| self.positions.get(root));
        at.zip(position).is_none_or(|(at, &p)| p <= at)
    }

    fn check_type_visibility(&self, ty: &Type, at: Option<usize>) -> Result<(), String> {
        if at.is_none() {
            return Ok(());
        }
        let mut pending = vec![ty];
        while let Some(ty) = pending.pop() {
            match ty {
                Type::Named(name) if !self.visible(name, at) => {
                    return Err(format!("type '{name}' is declared later in the module"));
                }
                Type::Tuple(fields) => pending.extend(fields.iter().rev()),
                Type::Array { element, .. } => pending.push(element),
                Type::Bool | Type::Bits { .. } | Type::Named(_) => { /* A primitive or visible type. */
                }
            }
        }
        Ok(())
    }
}

/// Checked BX and imported DSLX structs, with cached sizes for layout
/// validation.
#[derive(Default)]
struct StructTable {
    definitions: Vec<StructDef>,
    indices: BTreeMap<String, (usize, TypeShape)>,
    imported_count: usize,
}

/// Look up a DSLX module member in the source scope. Block-local values
/// shadow the module qualifier or the directly bound member.
fn lookup_use(
    name: &str,
    resolver: &ImportResolver,
    is_local: impl FnOnce(&str) -> bool,
) -> Result<Option<(String, String)>, String> {
    let member = resolver.lookup(name)?;
    let binding = name.split_once("::").map_or(name, |(head, _)| head);
    if member.is_some() && is_local(binding) {
        return Err(format!("'{binding}' names a local value here"));
    }
    Ok(member)
}

/// Copy a source type into the checked module, resolving qualified names as
/// we go. Bound the walk before cloning any recursive types from AST callers.
fn resolve_type(
    ty: &Type,
    is_local: &dyn Fn(&str) -> bool,
    bindings: Option<&Bindings>,
    resolver: &mut ImportResolver,
    remaining: usize,
) -> Result<Type, String> {
    match ty {
        Type::Named(name) => {
            if is_local(name) {
                return Err(format!("'{name}' names a local value here"));
            }
            if let Some(bindings) = bindings {
                if bindings.declares(name) {
                    // An actual type is already resolved in the caller's scope.
                    // During template declaration the name is still symbolic.
                    return Ok(bindings.get(name).unwrap_or(ty).clone());
                }
                bindings.reject_value(name)?;
            }
            if let Some((subject, member)) = lookup_use(name, resolver, is_local)? {
                resolver.resolve_named(&subject, &member)
            } else if let Some((qualifier, _)) = name.split_once("::") {
                Err(format!("unknown use binding '{qualifier}' in '{name}'"))
            } else {
                Ok(ty.clone())
            }
        }
        Type::Array { element, size } => Ok(Type::Array {
            element: Box::new(resolve_type(
                element,
                is_local,
                bindings,
                resolver,
                child_type_budget(remaining)?,
            )?),
            size: *size,
        }),
        Type::Tuple(items) => {
            let child_budget = child_type_budget(remaining)?;
            items
                .iter()
                .map(|item| resolve_type(item, is_local, bindings, resolver, child_budget))
                .collect::<Result<_, _>>()
                .map(Type::Tuple)
        }
        Type::Bits { .. } | Type::Bool => Ok(ty.clone()),
    }
}

/// Resolve struct fields in the enclosing module scope.
fn resolve_fields(params: &[Param], resolver: &mut ImportResolver) -> Result<Vec<Param>, String> {
    params
        .iter()
        .map(|p| {
            Ok(Param {
                name: p.name.clone(),
                ty: resolve_type(&p.ty, &|_| false, None, resolver, MAX_ELABORATION_DEPTH)?,
            })
        })
        .collect()
}

/// A parameter name enters scope before its type annotation and remains
/// visible in later parameters and the result type.
fn resolve_signature<'a>(
    params: &[Param],
    result: &Type,
    timing_ports: impl IntoIterator<Item = &'a String>,
    bindings: Option<&Bindings>,
    resolver: &mut ImportResolver,
) -> Result<(Vec<Param>, Type), String> {
    let mut scope: BTreeSet<_> = timing_ports.into_iter().cloned().collect();
    let mut resolved = Vec::with_capacity(params.len());
    for p in params {
        scope.insert(p.name.clone());
        resolved.push(Param {
            name: p.name.clone(),
            ty: resolve_type(
                &p.ty,
                &|name| scope.contains(name),
                bindings,
                resolver,
                MAX_ELABORATION_DEPTH,
            )?,
        });
    }
    let result = resolve_type(
        result,
        &|name| scope.contains(name),
        bindings,
        resolver,
        MAX_ELABORATION_DEPTH,
    )?;
    Ok((resolved, result))
}

/// Apply BX's concrete layout budget to a value type with no named
/// declarations.
pub(crate) fn validate_structural_type(ty: &Type) -> Result<(), String> {
    StructTable::default().validate_type(ty)
}

fn declare(names: &mut BTreeSet<String>, name: &str) -> Result<(), String> {
    if names.insert(name.to_owned()) {
        Ok(())
    } else {
        Err(format!("duplicate name '{name}'"))
    }
}

/// DSLX installs these builtin bindings before parsing any module constants.
/// Keep the spellings in sync with GetParametricBuiltins and
/// GetIoObjectBuiltins in the pinned frontend. Sized type keywords are already
/// excluded by the lexer.
fn dslx_builtin_binding(name: &str) -> bool {
    crate::import_structs::is_token_builtin(name)
        || matches!(
            name,
            "all_ones!"
                | "zero!"
                | "bit_slice_update"
                | "ceillog2"
                | "clz"
                | "configured_value_or"
                | "ctz"
                | "gate!"
                | "map"
                | "decode"
                | "encode"
                | "one_hot"
                | "one_hot_sel"
                | "priority_sel"
                | "rev"
                | "umulp"
                | "smulp"
                | "array_rev"
                | "array_size"
                | "bit_count"
                | "element_count"
                | "and_reduce"
                | "or_reduce"
                | "xor_reduce"
                | "signex"
                | "array_slice"
                | "update"
                | "widening_cast"
                | "checked_cast"
                | "range"
                | "zip"
                | "send"
                | "send_if"
                | "recv"
                | "recv_if"
                | "recv_non_blocking"
                | "recv_if_non_blocking"
                | "join"
                | "token"
                | "read"
                | "write"
                | "Source"
                | "Sink"
                | "IOResult"
                | "ChannelConfig"
        )
}

/// Collect BX dependencies. Imported layouts are registered before local
/// structs.
fn type_deps<'a>(ty: &'a Type, result: &mut Vec<&'a str>) {
    let mut pending = vec![ty];
    while let Some(ty) = pending.pop() {
        match ty {
            Type::Named(name) => {
                if !name.contains("::") {
                    result.push(name);
                }
            }
            Type::Array { element, .. } => pending.push(element),
            Type::Tuple(fields) => pending.extend(fields.iter().rev()),
            Type::Bool | Type::Bits { .. } => { /* Primitive types have no dependencies. */ }
        }
    }
}

/// Bound concrete value materialization, including expanded zero-width members.
/// A bit type costs its width (at least one), and an aggregate costs one plus
/// the sum of its members' costs. This accounts for aggregate storage even
/// when its packed width is zero. An empty array still stores its element type.
pub(crate) const MAX_TYPE_SIZE: usize = 1 << 20;

/// Bound recursion in downstream layout construction and hierarchy elaboration.
pub(crate) const MAX_ELABORATION_DEPTH: usize = 64;

/// Storage cost and maximum aggregate nesting of a fully expanded source type.
#[derive(Clone, Copy)]
struct TypeShape {
    size: usize,
    depth: usize,
    /// Nodes in the lowered IR type. Arrays store their element type once,
    /// and nominal structs expand to tuples.
    ir_nodes: usize,
}

impl<'a> Declarations<'a> {
    /// Resolve and validate each function signature before checking call sites.
    fn add_function(
        &mut self,
        name: &'a str,
        params: &[Param],
        result: &Type,
        local: Option<&'a Function>,
        resolver: &mut ImportResolver,
        structs: &mut StructTable,
    ) -> Result<(), String> {
        let mut names = BTreeSet::new();
        let mut callback_params = Vec::new();
        if let Some(function) = local {
            let at = self.positions.get(name).copied();
            for callback in &function.callback_params {
                declare(&mut names, &callback.name)?;
                let mut resolve = |ty: &Type| {
                    self.check_type_visibility(ty, at)?;
                    resolve_type(
                        ty,
                        &|name| names.contains(name),
                        None,
                        resolver,
                        MAX_ELABORATION_DEPTH,
                    )
                };
                callback_params.push(CallbackParam {
                    name: callback.name.clone(),
                    params: callback
                        .params
                        .iter()
                        .map(&mut resolve)
                        .collect::<Result<_, _>>()?,
                    result: resolve(&callback.result)?,
                });
            }
            if !callback_params.is_empty() {
                self.function_templates.insert(name, function);
            }
        }
        let (params, result) = resolve_signature(params, result, &names, None, resolver)?;
        structs.update_imports(resolver)?;
        let callback_params = callback_params
            .into_iter()
            .map(|signature| {
                let mut binding_work = 0usize;
                for ty in signature.params.iter().chain([&signature.result]) {
                    structs
                        .validate_type(ty)
                        .map_err(|e| format!("callback '{}': {e}", signature.name))?;
                    binding_work = binding_work.saturating_add(source_type_nodes(ty));
                }
                Ok(CallbackFormal {
                    signature,
                    binding_work,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        for p in &params {
            declare(&mut names, &p.name)?;
            structs
                .validate_type(&p.ty)
                .map_err(|e| format!("parameter '{}': {e}", p.name))?;
        }
        structs
            .validate_type(&result)
            .map_err(|e| format!("result: {e}"))?;
        self.functions.insert(
            name,
            FunctionInterface {
                params,
                result,
                callback_indices: callback_params
                    .iter()
                    .enumerate()
                    .map(|(i, callback)| (callback.signature.name.clone(), i))
                    .collect(),
                callback_params,
                local: local.is_some(),
            },
        );
        Ok(())
    }

    /// Check the shared physical port ABI before any parent uses this
    /// signature.
    fn add_block(
        &mut self,
        name: &'a str,
        clock: &str,
        reset: Option<&str>,
        params: Vec<Param>,
        result: Type,
        structs: &StructTable,
    ) -> Result<(), String> {
        let interface = BlockInterface::new(clock, reset, params, result, structs)?;
        self.blocks.insert(name, interface);
        Ok(())
    }
}

impl BlockInterface {
    /// Validate the concrete physical port ABI for a block definition.
    fn new(
        clock: &str,
        reset: Option<&str>,
        params: Vec<Param>,
        result: Type,
        structs: &StructTable,
    ) -> Result<Self, String> {
        let layout = Self::check_ports(clock, reset, &params, &result, structs, None)?;
        Ok(Self {
            params,
            result,
            outputs: layout.outputs,
            requires_reset: reset.is_some(),
            port_type_nodes: layout.type_nodes,
        })
    }

    /// Check physical port names and layout bounds. A bare type parameter can
    /// introduce struct fields or tuple outputs once its binding is known.
    fn check_ports(
        clock: &str,
        reset: Option<&str>,
        params: &[Param],
        result: &Type,
        structs: &StructTable,
        parameters: Option<&Bindings>,
    ) -> Result<BlockPortLayout, String> {
        let mut ports = BTreeSet::new();
        declare(&mut ports, clock)?;
        if let Some(reset) = reset {
            declare(&mut ports, reset)?;
        }
        let mut type_nodes = 0usize;
        for p in params {
            declare(&mut ports, &p.name)?;
            let shape = structs
                .formal_shape(&p.ty, MAX_ELABORATION_DEPTH, parameters)
                .map_err(|e| format!("input '{}': {e}", p.name))?;
            type_nodes = type_nodes.saturating_add(shape.ir_nodes);
        }
        let shape = structs
            .formal_shape(result, MAX_ELABORATION_DEPTH, parameters)
            .map_err(|e| format!("result: {e}"))?;
        type_nodes = type_nodes.saturating_add(shape.ir_nodes);
        if matches!(result, Type::Named(name) if parameters.is_some_and(|p| p.declares(name))) {
            return Ok(BlockPortLayout {
                outputs: Vec::new(),
                type_nodes,
            });
        }
        let outputs = output_ports(result, &structs.definitions)?;
        for output in &outputs {
            if !xlsynth_pir::ir_builder::is_valid_identifier(&output.name) {
                return Err(format!(
                    "output port '{}' is not a valid XLS IR identifier",
                    output.name
                ));
            }
            declare(&mut ports, &output.name)?;
        }
        Ok(BlockPortLayout {
            outputs,
            type_nodes,
        })
    }
}

impl StructTable {
    /// Validate a definition after its dependencies and index it for field
    /// selection.
    fn index(&mut self, index: usize) -> Result<(), String> {
        let def = &self.definitions[index];
        let shape = TypeShape::aggregate(
            def.fields.iter().map(|p| &p.ty),
            MAX_ELABORATION_DEPTH,
            &|name| self.named_shape(name),
        )
        .map_err(|e| format!("struct '{}': {e}", def.name))?;
        self.indices.insert(def.name.clone(), (index, shape));
        Ok(())
    }

    /// Import newly reflected definitions in dependency order. A call can
    /// introduce a type even when the BX source has no explicit annotation
    /// for its result.
    fn update_imports(&mut self, resolver: &ImportResolver) -> Result<(), String> {
        for def in &resolver.struct_defs()[self.imported_count..] {
            let index = self.definitions.len();
            self.definitions.push(def.clone());
            self.index(index)?;
            self.imported_count += 1;
        }
        Ok(())
    }

    fn definition(&self, name: &str) -> Result<&StructDef, String> {
        self.indices
            .get(name)
            .map(|(i, _)| &self.definitions[*i])
            .ok_or_else(|| unknown_declaration("struct", name))
    }

    fn validate_type(&self, ty: &Type) -> Result<(), String> {
        self.type_shape(ty, MAX_ELABORATION_DEPTH).map(|_| ())
    }

    /// Bound the concrete layout, including the full depth of memoized named
    /// types.
    fn type_shape(&self, ty: &Type, remaining: usize) -> Result<TypeShape, String> {
        self.formal_shape(ty, remaining, None)
    }

    fn named_shape(&self, name: &str) -> Result<TypeShape, String> {
        self.indices
            .get(name)
            .map(|(_, shape)| *shape)
            .ok_or_else(|| unknown_declaration("struct", name))
    }

    /// An unbound type occupies at least one unit. This lower bound lets
    /// declarations check fixed dimensions and nominal names before any call.
    fn formal_shape(
        &self,
        ty: &Type,
        remaining: usize,
        parameters: Option<&Bindings>,
    ) -> Result<TypeShape, String> {
        TypeShape::for_type(ty, remaining, &|name| {
            parameters
                .and_then(|p| p.shape(name))
                .map_or_else(|| self.named_shape(name), Ok)
        })
    }
}

impl TypeShape {
    const LEAF: Self = Self {
        size: 1,
        depth: 0,
        ir_nodes: 1,
    };

    /// Compute the expanded layout without allocating its substituted type.
    /// Named leaves supply a previously checked struct or type argument shape.
    fn for_type(
        ty: &Type,
        remaining: usize,
        named_shape: &impl Fn(&str) -> Result<Self, String>,
    ) -> Result<Self, String> {
        let shape = match ty {
            Type::Bool => Self::LEAF,
            Type::Bits { width, .. } => TypeShape {
                size: (*width).max(1),
                ..Self::LEAF
            },
            Type::Tuple(items) => Self::aggregate(items, remaining, named_shape)?,
            Type::Array { element, size } => {
                let child = Self::for_type(element, child_type_budget(remaining)?, named_shape)?;
                TypeShape {
                    size: child.size.saturating_mul((*size).max(1)).saturating_add(1),
                    depth: child.depth + 1,
                    ir_nodes: child.ir_nodes.saturating_add(1),
                }
            }
            Type::Named(name) => named_shape(name)?,
        };
        if shape.depth > remaining {
            return Err(format!(
                "expanded type nesting exceeds {MAX_ELABORATION_DEPTH} levels"
            ));
        }
        if shape.size > MAX_TYPE_SIZE {
            return Err(format!(
                "type {ty} exceeds maximum concrete type size {MAX_TYPE_SIZE}"
            ));
        }
        Ok(shape)
    }

    fn aggregate<'t>(
        fields: impl IntoIterator<Item = &'t Type>,
        remaining: usize,
        named_shape: &impl Fn(&str) -> Result<Self, String>,
    ) -> Result<TypeShape, String> {
        let child_budget = child_type_budget(remaining)?;
        let mut shape = TypeShape {
            depth: 1,
            ..Self::LEAF
        };
        for field in fields {
            let child = Self::for_type(field, child_budget, named_shape)?;
            shape.size = shape.size.saturating_add(child.size);
            shape.ir_nodes = shape.ir_nodes.saturating_add(child.ir_nodes);
            shape.depth = shape.depth.max(child.depth + 1);
            if shape.size > MAX_TYPE_SIZE {
                return Err(format!(
                    "aggregate exceeds maximum concrete type size {MAX_TYPE_SIZE}"
                ));
            }
        }
        Ok(shape)
    }
}

fn child_type_budget(remaining: usize) -> Result<usize, String> {
    remaining
        .checked_sub(1)
        .ok_or_else(|| format!("expanded type nesting exceeds {MAX_ELABORATION_DEPTH} levels"))
}

/// A dependency in progress or its fully checked depth in the dependency graph.
#[derive(Clone, Copy)]
enum Visit {
    Unseen,
    Active,
    Complete(usize),
}

/// Visit named dependencies in source order, rejecting cycles and excessive
/// depth.
fn dependency_order<'a>(
    kind: &str,
    graph: &[(&'a str, Vec<&'a str>)],
) -> Result<Vec<usize>, String> {
    fn visit(
        i: usize,
        kind: &str,
        graph: &[(&str, Vec<&str>)],
        indices: &BTreeMap<&str, usize>,
        status: &mut [Visit],
        order: &mut Vec<usize>,
        remaining: usize,
    ) -> Result<usize, String> {
        let too_deep = || {
            format!(
                "{kind} dependency depth exceeds {MAX_ELABORATION_DEPTH} levels at '{}'",
                graph[i].0
            )
        };
        if remaining == 0 {
            return Err(too_deep());
        }
        match status[i] {
            Visit::Complete(depth) if depth > remaining => return Err(too_deep()),
            Visit::Complete(depth) => return Ok(depth),
            Visit::Active => {
                return Err(format!(
                    "recursive {kind} dependency involving '{}'",
                    graph[i].0
                ));
            }
            Visit::Unseen => status[i] = Visit::Active,
        }
        let mut depth = 1;
        for dep in &graph[i].1 {
            let next = *indices
                .get(dep)
                .ok_or_else(|| format!("{} in '{}'", unknown_declaration(kind, dep), graph[i].0))?;
            let child_depth = visit(next, kind, graph, indices, status, order, remaining - 1)?;
            depth = depth.max(child_depth + 1);
        }
        status[i] = Visit::Complete(depth);
        order.push(i);
        Ok(depth)
    }
    let indices = graph
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (*name, i))
        .collect();
    let mut order = Vec::with_capacity(graph.len());
    let mut status = vec![Visit::Unseen; graph.len()];
    for i in 0..graph.len() {
        visit(
            i,
            kind,
            graph,
            &indices,
            &mut status,
            &mut order,
            MAX_ELABORATION_DEPTH,
        )?;
    }
    Ok(order)
}

/// Type-check a parsed module without requiring the external functions' IR
/// bodies.
pub fn check_module(module: &Module) -> Result<CheckedModule, String> {
    check_module_with_options(module, &ImportOptions::default())
}

/// Check BX declarations and resolve the imported DSLX functions for lowering.
pub fn check_module_with_options(
    module: &Module,
    options: &ImportOptions<'_>,
) -> Result<CheckedModule, String> {
    let mut names = BTreeSet::new();
    for path in &module.uses {
        let name = path.binding().ok_or("empty DSLX use path")?;
        if !names.insert(name.to_owned()) {
            return Err(format!("duplicate use binding '{name}'"));
        }
    }
    for f in &module.externs {
        if let Some((qualifier, _)) = f.name.split_once("::")
            && names.contains(qualifier)
        {
            return Err(format!(
                "extern fn '{}' conflicts with use binding '{qualifier}'",
                f.name
            ));
        }
    }
    for constant in &module.constants {
        let name = constant.name.as_str();
        if !(crate::parser::is_use_identifier(name) || name == "self") {
            return Err(format!("invalid module constant name '{name}'"));
        }
        if dslx_builtin_binding(name) {
            return Err(format!("constant '{name}' shadows a builtin"));
        }
        declare(&mut names, name)?;
    }
    let mut resolver = ImportResolver::new(&module.uses, options)?;
    let mut structs = StructTable::default();
    for def in &module.structs {
        declare(&mut names, &def.name)?;
        if def.name.contains("::") {
            return Err(format!(
                "BX struct '{}' must have an unqualified name",
                def.name
            ));
        }
        let mut fields = BTreeSet::new();
        for field in &def.fields {
            declare(&mut fields, &field.name).map_err(|e| format!("struct '{}': {e}", def.name))?;
        }
        structs.definitions.push(StructDef {
            name: def.name.clone(),
            public: def.public,
            fields: resolve_fields(&def.fields, &mut resolver)?,
        });
    }
    let mut struct_graph = Vec::new();
    for def in &structs.definitions {
        let mut deps = Vec::new();
        for field in &def.fields {
            type_deps(&field.ty, &mut deps);
        }
        struct_graph.push((def.name.as_str(), deps));
    }
    let struct_order = dependency_order("struct", &struct_graph)?;
    structs.update_imports(&resolver)?;
    for index in struct_order {
        structs.index(index)?;
    }

    let mut declarations = Declarations {
        positions: module
            .declaration_order
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect(),
        ..Declarations::default()
    };
    // Without an explicit cross-kind order, programmatic modules introduce
    // their constants first and preserve source order within each list.
    if declarations.positions.is_empty() {
        declarations.positions.extend(
            module
                .constants
                .iter()
                .map(|c| c.name.as_str())
                .chain(module.functions.iter().map(|f| f.name.as_str()))
                .enumerate()
                .map(|(i, name)| (name, i)),
        );
    }
    let mut externs = Vec::with_capacity(module.externs.len());
    for f in &module.externs {
        declare(&mut names, &f.name)?;
        declarations
            .add_function(
                &f.name,
                &f.params,
                &f.result,
                None,
                &mut resolver,
                &mut structs,
            )
            .map_err(|e| format!("extern '{}': {e}", f.name))?;
        let interface = &declarations.functions[f.name.as_str()];
        externs.push(ExternFn {
            name: f.name.clone(),
            params: interface.params.clone(),
            result: interface.result.clone(),
            symbol: f.symbol.clone(),
        });
    }
    for f in &module.functions {
        declare(&mut names, &f.name)?;
        if f.name.contains("::") {
            return Err(format!(
                "BX function '{}' must have an unqualified name",
                f.name
            ));
        }
        let at = declarations.positions.get(f.name.as_str()).copied();
        for ty in f.params.iter().map(|p| &p.ty).chain([&f.result]) {
            declarations.check_type_visibility(ty, at)?;
        }
        declarations
            .add_function(
                &f.name,
                &f.params,
                &f.result,
                Some(f),
                &mut resolver,
                &mut structs,
            )
            .map_err(|e| format!("fn '{}': {e}", f.name))?;
    }
    for b in &module.blocks {
        declare(&mut names, &b.name)?;
        let reserved_ports = std::iter::once(&b.clock)
            .chain(&b.reset)
            .chain(b.callback_params.iter().map(|p| &p.name));
        let symbolic = if b.type_params.is_empty() && b.callback_params.is_empty() {
            None
        } else {
            generic_blocks::validate_declaration(b)
                .map_err(|e| format!("block '{}': {e}", b.name))?;
            Some(Bindings::new(&b.type_params))
        };
        let (params, result) = resolve_signature(
            &b.params,
            &b.result,
            reserved_ports,
            symbolic.as_ref(),
            &mut resolver,
        )?;
        if let Some(bindings) = &symbolic {
            declarations.templates.insert(
                &b.name,
                BlockTemplate::new(b, params, result, bindings, &mut resolver, &mut structs)
                    .map_err(|e| format!("block '{}': {e}", b.name))?,
            );
            continue;
        }
        structs.update_imports(&resolver)?;
        declarations
            .add_block(
                &b.name,
                &b.clock,
                b.reset.as_deref(),
                params,
                result,
                &structs,
            )
            .map_err(|e| format!("block '{}': {e}", b.name))?;
    }
    let mut extern_blocks = Vec::with_capacity(module.extern_blocks.len());
    for b in &module.extern_blocks {
        declare(&mut names, &b.name)?;
        let timing_ports = std::iter::once(&b.clock).chain(&b.reset);
        let (params, result) =
            resolve_signature(&b.params, &b.result, timing_ports, None, &mut resolver)?;
        structs.update_imports(&resolver)?;
        declarations
            .add_block(
                &b.name,
                &b.clock,
                b.reset.as_deref(),
                params.clone(),
                result.clone(),
                &structs,
            )
            .map_err(|e| format!("extern block '{}': {e}", b.name))?;
        extern_blocks.push(ExternBlock {
            name: b.name.clone(),
            clock: b.clock.clone(),
            reset: b.reset.clone(),
            params,
            result,
        });
    }
    let mut function_registry = FunctionSpecializations::new(names.clone());
    let function_bodies =
        function_registry.check_sources(module, &mut declarations, &mut resolver, &mut structs)?;
    // Static loops can instantiate children. Order the checked blocks using
    // the instances actually elaborated, including children declared later.
    let mut specializations = BlockSpecializations::new(names);
    let mut checked = Vec::new();
    for block in module
        .blocks
        .iter()
        .filter(|b| b.type_params.is_empty() && b.callback_params.is_empty())
    {
        checked.push(
            BodyChecker::check_block(
                &declarations,
                block,
                &mut resolver,
                &mut structs,
                &mut specializations,
                None,
                &mut function_registry,
            )
            .map_err(|e| format!("block '{}': {e}", block.name))?,
        );
    }
    let mut specialized_blocks = BTreeSet::new();
    while let Some(spec) = specializations.pending.pop_front() {
        let block = BodyChecker::check_block(
            &declarations,
            spec.block,
            &mut resolver,
            &mut structs,
            &mut specializations,
            Some(&spec),
            &mut function_registry,
        )
        .map_err(|e| format!("block '{}' ({}): {e}", spec.block.name, spec.name))?;
        specialized_blocks.insert(spec.name.clone());
        checked.push(block);
    }
    let functions =
        function_registry.finish(function_bodies, &declarations, &mut resolver, &mut structs)?;
    let mut graph = Vec::new();
    for block in &checked {
        let children = block
            .statements
            .iter()
            .filter_map(|stmt| match stmt {
                CheckedStmt::Inst { block, .. } => Some(block.as_str()),
                _ => None,
            })
            .collect();
        graph.push((block.name.as_str(), children));
    }
    graph.extend(extern_blocks.iter().map(|b| (b.name.as_str(), Vec::new())));
    let order = dependency_order("block", &graph)?;
    let mut checked: Vec<_> = checked.into_iter().map(Some).collect();
    let blocks = order
        .into_iter()
        .filter_map(|i| checked.get_mut(i)?.take())
        .collect();
    Ok(CheckedModule {
        structs: structs.definitions,
        constants: module
            .constants
            .iter()
            .map(|c| {
                declarations
                    .constants
                    .remove(c.name.as_str())
                    .expect("checked module constant")
            })
            .collect(),
        externs,
        functions,
        specialized_functions: function_registry.generated_names(),
        extern_blocks,
        imports: resolver.finish()?,
        blocks,
        specialized_blocks,
    })
}

/// Controls which declarations and bindings are available in a BX body.
#[derive(Clone, Copy, Eq, PartialEq)]
enum BodyKind {
    Function,
    Block { has_reset: bool },
}

/// Lexical values, loop elaboration, and any hardware declared in one body.
struct BodyChecker<'a, 'm> {
    decls: &'a Declarations<'m>,
    structs: &'a mut StructTable,
    resolver: &'a mut ImportResolver,
    kind: BodyKind,
    /// The source body owns the expansion budget even while checking a lambda
    /// with function-local rules.
    source_kind: BodyKind,
    specializations: Option<&'a mut BlockSpecializations<'m>>,
    specialization: Option<&'a BlockSpecialization<'m>>,
    function_registry: Option<&'a mut FunctionSpecializations<'m>>,
    function_spec: Option<&'a FunctionSpecialization<'m>>,
    /// The enclosing template's signature for abstract or concrete checking.
    source_template: Option<&'a FunctionInterface>,
    /// Use DSLX literal inference inside a local call argument or scalar match.
    dslx_value_context: bool,
    /// Bound materialized operands while inferring a callback block's data
    /// ports.
    checking_callback_inputs: bool,
    function_position: Option<usize>,
    /// Blocks allow forward hardware references, but introduce constants in
    /// source order.
    block_position: Option<usize>,
    local_callees: BTreeSet<String>,
    names: BTreeSet<String>,
    outputs: BTreeSet<String>,
    reserved_hardware_names: BTreeSet<String>,
    values: BTreeMap<String, Type>,
    constants: BTreeMap<String, ConstantBinding>,
    /// Earlier local functions available to a module constant initializer.
    constant_functions: Option<ConstantFunctions<'a>>,
    /// Bindings whose entire initializer depends only on compile-time inputs.
    constexpr_names: BTreeSet<String>,
    /// Minimum widths of unannotated bindings and their aggregate members.
    inferred_bindings: InferredBindings,
    regs: BTreeMap<String, Type>,
    writes: BTreeSet<String>,
    instances: BTreeMap<String, &'a BlockInterface>,
    connections: BTreeSet<(String, String)>,
    statements: Vec<CheckedStmt>,
    /// Source names bound by currently active loop iterations.
    loop_values: BTreeMap<String, TypedExpr>,
    /// Function locals resolve to an IR value that stays valid after their
    /// lexical scope ends, including when another arm binds the same name.
    function_values: BTreeMap<String, TypedExpr>,
    /// Source spelling behind each live `$scope_N` alias. Prepared loop
    /// iterables have no binding name.
    scope_alias_names: Vec<Option<String>>,
    /// Iterables checked before a scoped seed is prepared for result inference.
    prepared_counted_iterables: BTreeMap<String, counted_for::Iteration>,
    /// Counted bodies bind runtime loop values even outside a structural
    /// unroll.
    in_counted_body: bool,
    /// Structural iterations enclosing the current closed lambda. They still
    /// consume expansion work, but do not supply loop-local type inference.
    map_outer_iterations: Option<usize>,
    next_counted_loop: usize,
    iteration_path: Vec<usize>,
    expansion_work: usize,
    reference_copy_work: Cell<usize>,
    constant_budgets: ConstantBudgets,
    unroll_binding_count: usize,
}

impl<'a, 'm> BodyChecker<'a, 'm> {
    /// Generated bodies charge their common registry. Ordinary source bodies
    /// own independent budgets just as they do for structural unrolling.
    fn constant_context(&mut self) -> ConstantContext<'_, 'a> {
        let budgets = match (
            &mut self.function_registry,
            self.function_spec,
            &mut self.specializations,
            self.specialization,
        ) {
            (Some(registry), Some(_), _, _) => &mut registry.constant_budgets,
            (_, _, Some(registry), Some(_)) => &mut registry.constant_budgets,
            _ => &mut self.constant_budgets,
        };
        ConstantContext {
            names: &mut self.constants,
            statements: &self.statements,
            budgets,
            charge_values: self.constant_functions.is_some() || self.function_spec.is_some(),
            functions: self.constant_functions.as_mut(),
        }
    }

    /// Record a constant if speculative folding has work remaining.
    fn fold_constant(&mut self, value: &TypedExpr) -> Option<ConstantBinding> {
        let ctx = self.constant_context();
        let deferred = ConstantBinding::Deferred(ctx.statements.len());
        // A projected wide scalar owns its limbs in the speculative cache.
        // Avoid materializing it once the remaining allowance cannot retain it.
        if !ctx.charge_values
            && bits(&value.ty).is_some_and(|(_, width)| {
                width > 64 && width.div_ceil(64) > ctx.budgets.speculative
            })
        {
            return Some(deferred);
        }
        match value.constant(
            ctx.names,
            &mut ctx.budgets.speculative,
            ctx.charge_values,
            ctx.functions,
        ) {
            Ok(Some(value)) if !ctx.charge_values => {
                // Ordinary bodies retain the result without charging other
                // straight-line intermediate values computed by the evaluator.
                if let Some(remaining) = ctx
                    .budgets
                    .speculative
                    .checked_sub(literal_copy_work(&value))
                {
                    ctx.budgets.speculative = remaining;
                    Some(ConstantBinding::Known(value))
                } else {
                    Some(deferred)
                }
            }
            Ok(value) => value.map(ConstantBinding::Known),
            Err(_) => Some(deferred),
        }
    }

    /// Replay only required earlier lets, using an explicit stack so long
    /// chains cannot overflow the checker. Cache each result in its binding.
    fn checked_constant(&mut self, value: &TypedExpr) -> Result<Option<IrValue>, String> {
        let mut ctx = self.constant_context();
        let mut charge_values = ctx.charge_values;
        let mut pending: Vec<usize> = Vec::new();
        loop {
            let definition = pending.last().copied();
            let (expr, name) = if let Some(index) = definition {
                let CheckedStmt::Let { name, value } = &ctx.statements[index] else {
                    unreachable!("deferred constant refers to a checked let")
                };
                if !matches!(ctx.names.get(name), Some(ConstantBinding::Deferred(i)) if *i == index)
                {
                    pending.pop();
                    continue;
                }
                (value, Some(name))
            } else {
                (value, None)
            };
            match expr.constant(
                ctx.names,
                &mut ctx.budgets.required,
                charge_values,
                ctx.functions.as_deref_mut(),
            ) {
                Ok(result) => {
                    let Some(name) = name else { return Ok(result) };
                    if let Some(result) = result {
                        ctx.names
                            .insert(name.clone(), ConstantBinding::Known(result));
                    } else {
                        ctx.names.remove(name);
                    }
                    pending.pop();
                }
                Err(ConstantFailure::Deferred(index)) => {
                    charge_values = true;
                    let mut earlier = BTreeSet::from([index]);
                    expr.deferred_inputs(ctx.names, &mut earlier);
                    debug_assert!(
                        earlier
                            .last()
                            .is_some_and(|i| *i < definition.unwrap_or(ctx.statements.len()))
                    );
                    pending.extend(earlier.into_iter().rev());
                }
                Err(ConstantFailure::WorkLimit) => {
                    return Err(format!(
                        "constant evaluation exceeds the work limit of {MAX_TYPE_SIZE}"
                    ));
                }
            }
        }
    }

    /// Start an independent lexical and elaboration scope for a function or
    /// block.
    fn new(
        decls: &'a Declarations<'m>,
        resolver: &'a mut ImportResolver,
        structs: &'a mut StructTable,
        kind: BodyKind,
    ) -> Self {
        Self {
            decls,
            structs,
            resolver,
            kind,
            source_kind: kind,
            specializations: None,
            specialization: None,
            function_registry: None,
            function_spec: None,
            source_template: None,
            dslx_value_context: false,
            checking_callback_inputs: false,
            function_position: None,
            block_position: None,
            local_callees: BTreeSet::new(),
            names: BTreeSet::new(),
            outputs: BTreeSet::new(),
            reserved_hardware_names: BTreeSet::new(),
            values: BTreeMap::new(),
            constants: BTreeMap::new(),
            constant_functions: None,
            constexpr_names: BTreeSet::new(),
            inferred_bindings: BTreeMap::new(),
            regs: BTreeMap::new(),
            writes: BTreeSet::new(),
            instances: BTreeMap::new(),
            connections: BTreeSet::new(),
            statements: Vec::new(),
            loop_values: BTreeMap::new(),
            function_values: BTreeMap::new(),
            scope_alias_names: Vec::new(),
            prepared_counted_iterables: BTreeMap::new(),
            in_counted_body: false,
            map_outer_iterations: None,
            next_counted_loop: 0,
            iteration_path: Vec::new(),
            expansion_work: 0,
            reference_copy_work: Cell::new(0),
            constant_budgets: ConstantBudgets::default(),
            unroll_binding_count: 0,
        }
    }

    /// Check ordered let bindings and a return expression in a pure scope.
    fn check_function(&mut self, function: &Function) -> Result<CheckedFunction, String> {
        self.function_position = self.decls.positions.get(function.name.as_str()).copied();
        let interface = &self.decls.functions[function.name.as_str()];
        self.source_template = (!interface.callback_params.is_empty()).then_some(interface);
        if self.function_spec.is_some() {
            for ty in interface
                .params
                .iter()
                .map(|p| &p.ty)
                .chain([&interface.result])
            {
                let nodes = self.structs.type_shape(ty, MAX_ELABORATION_DEPTH)?.ir_nodes;
                self.charge_expansion("function signature", nodes)?;
            }
        }
        for p in &interface.params {
            self.names.insert(p.name.clone());
            self.values.insert(p.name.clone(), p.ty.clone());
        }
        let output = self
            .scoped_result(&function.body, Some(&interface.result), false)
            .map_err(|e| format!("result: {e}"))?;
        let bindings = self
            .statements
            .drain(..)
            .map(|stmt| match stmt {
                CheckedStmt::Let { name, value } => CheckedBinding { name, value },
                _ => unreachable!("pure expressions can only elaborate let bindings"),
            })
            .collect();
        Ok(CheckedFunction {
            name: self
                .function_spec
                .map_or(&function.name, |s| &s.name)
                .clone(),
            public: self.function_spec.is_none() && function.public,
            params: interface.params.clone(),
            result: interface.result.clone(),
            bindings,
            output,
        })
    }

    /// Reserve interface and instance names before checking dataflow
    /// expressions.
    fn check_block(
        decls: &'a Declarations<'m>,
        block: &'m Block,
        resolver: &'a mut ImportResolver,
        structs: &'a mut StructTable,
        specializations: &'a mut BlockSpecializations<'m>,
        specialization: Option<&'a BlockSpecialization<'m>>,
        function_registry: &'a mut FunctionSpecializations<'m>,
    ) -> Result<CheckedBlock, String> {
        let interface = match specialization {
            Some(spec) => &spec.interface,
            None => &decls.blocks[block.name.as_str()],
        };
        let outputs = interface
            .outputs
            .iter()
            .map(|p| p.name.clone())
            .collect::<BTreeSet<_>>();
        // Generated names must avoid future declarations as well as visible
        // ones.
        let mut reserved_hardware_names = outputs.clone();
        reserved_hardware_names.insert(block.clock.clone());
        reserved_hardware_names.extend(block.reset.iter().cloned());
        reserved_hardware_names.extend(block.params.iter().map(|p| p.name.clone()));
        for statement in &block.statements {
            match statement {
                Stmt::Reg { name, .. } | Stmt::Let { name, .. } | Stmt::Inst { name, .. } => {
                    reserved_hardware_names.insert(name.clone());
                }
                Stmt::LetPattern(binding) => {
                    binding.pattern.visit_names(&mut |name| {
                        reserved_hardware_names.insert(name.to_owned());
                    });
                }
                _ => { /* Drivers and guards introduce no top-level names. */ }
            }
        }
        let mut c = Self::new(
            decls,
            resolver,
            structs,
            BodyKind::Block {
                has_reset: block.reset.is_some(),
            },
        );
        c.outputs = outputs;
        c.block_position = decls.positions.get(block.name.as_str()).copied();
        c.specializations = Some(specializations);
        c.specialization = specialization;
        c.function_registry = Some(function_registry);
        c.reserved_hardware_names = reserved_hardware_names;
        c.hardware_name(&block.clock)?;
        if let Some(rst) = &block.reset {
            c.hardware_name(rst)?;
            c.values.insert(rst.clone(), Type::Bool);
        }
        for p in &interface.params {
            c.hardware_name(&p.name)?;
            c.values.insert(p.name.clone(), p.ty.clone());
        }
        for stmt in &block.statements {
            if let Stmt::Inst { name, block: child } = stmt {
                c.hardware_name(name)?;
                if c.callback_index(child).is_some() {
                    return Err(format!(
                        "callable parameter '{child}' requires a direct call"
                    ));
                }
                c.reject_type_parameter_value(child)?;
                if let Some(template) = decls.templates.get(child.as_str()) {
                    if !template.callback_params.is_empty() {
                        return Err(format!(
                            "block '{child}' requires a let call with explicit named callbacks"
                        ));
                    }
                    return Err(format!(
                        "generic block '{child}' needs data arguments to infer its type parameters; use 'let {name} = {child}(...)'"
                    ));
                }
                let child_block = decls
                    .blocks
                    .get(child.as_str())
                    .ok_or_else(|| format!("unknown child block '{child}'"))?;
                c.check_child_instance(name, child, child_block, /* has_callbacks= */ false)?;
                c.instances.insert(name.clone(), child_block);
            }
        }
        c.body(block, &block.statements, false)?;
        for reg in c.regs.keys() {
            if !c.writes.contains(reg) {
                return Err(format!("register '{reg}' has no update"));
            }
        }
        for (name, child) in &c.instances {
            for p in &child.params {
                if !c.connections.contains(&(name.clone(), p.name.clone())) {
                    return Err(format!("instance input '{name}.{}' has no driver", p.name));
                }
            }
        }
        let output = c
            .scoped_result(&block.output, Some(&interface.result), true)
            .map_err(|e| format!("result: {e}"))?;
        Ok(CheckedBlock {
            name: specialization.map_or(&block.name, |s| &s.name).clone(),
            public: specialization.is_none() && block.public,
            clock: block.clock.clone(),
            reset: block.reset.clone(),
            params: interface.params.clone(),
            result: interface.result.clone(),
            statements: c.statements,
            output,
        })
    }

    /// Resolve a type written in a block body and register its concrete fields.
    fn source_type(&mut self, source: &Type) -> Result<Type, String> {
        self.decls
            .check_type_visibility(source, self.function_position)?;
        if let Some(spec) = self.specialization {
            spec.bindings.validate_substitution(source)?;
        }
        let callbacks = self.callback_indices();
        let ty = resolve_type(
            source,
            &|name| {
                self.names.contains(name)
                    || callbacks.is_some_and(|indices| indices.contains_key(name))
            },
            self.specialization.map(|s| &s.bindings),
            self.resolver,
            MAX_ELABORATION_DEPTH,
        )?;
        self.structs.update_imports(self.resolver)?;
        self.structs.validate_type(&ty)?;
        Ok(ty)
    }

    fn hardware_name(&mut self, name: &str) -> Result<(), String> {
        if self.outputs.contains(name) {
            return Err(format!("name '{name}' collides with an output port"));
        }
        declare(&mut self.names, name)
    }

    /// Check the reset and account for the data ports wired for this instance.
    fn check_child_instance(
        &mut self,
        name: &str,
        target: &str,
        child: &BlockInterface,
        has_callbacks: bool,
    ) -> Result<(), String> {
        if child.requires_reset && !matches!(self.kind, BodyKind::Block { has_reset: true }) {
            return Err(format!(
                "instance '{name}' of '{target}' requires a parent reset"
            ));
        }
        // Every instance materializes its own ports, even when its definition
        // is shared and its outputs are unused.
        if self.replaying_body() || has_callbacks {
            self.charge_expansion("child block wiring", child.port_type_nodes)?;
        }
        Ok(())
    }

    /// Keep the source binding as the generated instance name when available
    /// in IR. An output port or an IR keyword such as `next` needs a suffix.
    fn call_instance_name(&mut self, binding: &str) -> String {
        if !self.outputs.contains(binding) && xlsynth_pir::ir_builder::is_valid_identifier(binding)
        {
            return binding.to_owned();
        }
        let base = format!("{binding}_inst");
        let mut name = base.clone();
        let mut suffix = 0;
        while !self.reserved_hardware_names.insert(name.clone()) {
            suffix += 1;
            name = format!("{base}_{suffix}");
        }
        name
    }

    /// Elaborate a call to static hardware, then assemble its outputs into a BX
    /// value.
    fn block_call(
        &mut self,
        binding: &str,
        target: &str,
        callbacks: &[String],
        args: &[Expr],
        expected: Option<&Type>,
    ) -> Result<TypedExpr, String> {
        let mut prepared = None;
        let spec;
        let (symbol, child) = if let Some(template) = self.decls.templates.get(target) {
            let call = self.generic_inputs(template, args, callbacks)?;
            let path = self.specialization.map_or(&[][..], |s| s.path.as_slice());
            spec = self
                .specializations
                .as_mut()
                .expect("block elaboration")
                .intern(template, call.bindings, call.callbacks, path, self.structs)
                .map_err(|e| format!("call to block '{target}': {e}"))?;
            prepared = Some(call.values.into_iter());
            (spec.name.as_str(), &spec.interface)
        } else {
            (target, &self.decls.blocks[target])
        };
        self.check_child_instance(binding, target, child, !callbacks.is_empty())?;
        if let Some(ty) = expected
            && !compatible(ty, &child.result)
        {
            return Err(format!("expected {ty}, got {}", child.result));
        }
        if args.len() != child.params.len() {
            return Err(format!(
                "block '{target}' expects {} data arguments, got {}",
                child.params.len(),
                args.len()
            ));
        }
        let instance = self.call_instance_name(binding);
        self.statements.push(CheckedStmt::Inst {
            name: instance.clone(),
            block: symbol.to_owned(),
        });
        for (arg, param) in args.iter().zip(&child.params) {
            let value = match &mut prepared {
                Some(values) => values.next().expect("one checked value per generic input"),
                None => self.expr(arg, Some(&param.ty)).map_err(|e| {
                    format!("input '{}', call to block '{target}': {e}", param.name)
                })?,
            };
            self.statements.push(CheckedStmt::Connect {
                instance: instance.clone(),
                port: param.name.clone(),
                value,
            });
        }
        let mut outputs = child.outputs.iter().map(|port| TypedExpr {
            ty: port.ty.clone(),
            kind: K::InstanceOutput {
                instance: instance.clone(),
                port: port.name.clone(),
            },
        });
        match child.result {
            Type::Tuple(_) | Type::Named(_) => Ok(TypedExpr {
                ty: child.result.clone(),
                kind: K::Tuple(outputs.collect()),
            }),
            Type::Bool | Type::Bits { .. } | Type::Array { .. } => Ok(outputs
                .next()
                .expect("scalar and array interfaces have one output")),
        }
    }

    /// Check the explicit callback list even in a skipped structural loop,
    /// where no concrete data types or hardware instance are needed.
    fn check_block_callback_list(
        &self,
        target: &str,
        count: usize,
        hidden: bool,
    ) -> Result<(), String> {
        if hidden {
            let name = scope::source_binding_name(target, &self.scope_alias_names);
            return Err(format!("'{name}' names a local value here"));
        }
        self.reject_type_parameter_value(target)?;
        let expected = self
            .decls
            .templates
            .get(target)
            .map_or(0, |template| template.callback_params.len());
        if expected != count {
            return Err(format!(
                "block '{target}' expects {expected} callable arguments, got {count}"
            ));
        }
        Ok(())
    }

    /// A type binder shadows a module member or import in callee position.
    fn reject_type_parameter_value(&self, name: &str) -> Result<(), String> {
        self.specialization
            .map_or(Ok(()), |s| s.bindings.reject_value(name))
    }

    /// Check statements in lexical order, preserving a single expression per
    /// guard.
    fn body(
        &mut self,
        block: &Block,
        statements: &[Stmt],
        conditional: bool,
    ) -> Result<(), String> {
        let mut locals = Vec::new();
        let mut pattern_bindings = Vec::new();
        for stmt in statements {
            let checked = match stmt {
                Stmt::Reg { name, ty, reset } => {
                    if conditional {
                        return Err(format!("conditional declaration of register '{name}'"));
                    }
                    let ty = self.source_type(ty)?;
                    self.hardware_name(name)?;
                    let reset = match reset {
                        None => None,
                        Some(expr) => {
                            if !matches!(self.kind, BodyKind::Block { has_reset: true }) {
                                return Err(format!(
                                    "initializer on register '{name}' requires a reset"
                                ));
                            }
                            let value = self
                                .expr(expr, Some(&ty))
                                .map_err(|e| format!("reset of register '{name}': {e}"))?;
                            Some(self.checked_constant(&value)?.ok_or_else(|| {
                                format!(
                                    "reset of register '{name}' must be a compile-time constant"
                                )
                            })?)
                        }
                    };
                    self.regs.insert(name.clone(), ty.clone());
                    self.values.insert(name.clone(), ty.clone());
                    CheckedStmt::Reg {
                        name: name.clone(),
                        ty: ty.clone(),
                        reset,
                    }
                }
                Stmt::Inst { name, block } => {
                    if conditional {
                        return Err(format!("conditional declaration of instance '{name}'"));
                    }
                    if self.function_values.contains_key(block) {
                        return Err(format!("'{block}' names a local value here"));
                    }
                    CheckedStmt::Inst {
                        name: name.clone(),
                        block: block.clone(),
                    }
                }
                Stmt::Connect {
                    instance,
                    port,
                    value,
                } => {
                    if conditional {
                        return Err(format!("conditional connection of '{instance}.{port}'"));
                    }
                    let child = self
                        .instances
                        .get(instance)
                        .ok_or_else(|| format!("unknown instance '{instance}'"))?;
                    let param = child
                        .params
                        .iter()
                        .find(|p| p.name == *port)
                        .ok_or_else(|| format!("'{instance}.{port}' is not a data input"))?;
                    let value = self
                        .expr(value, Some(&param.ty))
                        .map_err(|e| format!("input '{instance}.{port}': {e}"))?;
                    if !self.connections.insert((instance.clone(), port.clone())) {
                        return Err(format!("multiple drivers for input '{instance}.{port}'"));
                    }
                    CheckedStmt::Connect {
                        instance: instance.clone(),
                        port: port.clone(),
                        value,
                    }
                }
                Stmt::Let {
                    name,
                    ty,
                    value: initializer,
                } => {
                    if self.function_values.contains_key(name) {
                        // Rebinding a pattern leaf uses the previous lexical
                        // value in the initializer and gives a child its own
                        // name.
                        let init =
                            self.block_initializer(None, ty.as_ref(), initializer, !conditional)?;
                        let value = self.share_scoped_value(init.value);
                        self.bind_scoped_value(name, value, init.widths, &mut pattern_bindings);
                        if !conditional {
                            pattern_bindings.clear();
                        }
                        continue;
                    }
                    if conditional {
                        locals.push(name);
                    }
                    self.check_let(name, ty.as_ref(), initializer, !conditional)?
                }
                Stmt::LetPattern(binding) => {
                    let init = self.pattern_initializer(binding, !conditional)?;
                    let mut collision = None;
                    binding.pattern.visit_names(&mut |name| {
                        if name == block.clock
                            || block.reset.as_deref() == Some(name)
                            || self.regs.contains_key(name)
                            || self.instances.contains_key(name)
                        {
                            collision = Some(name.to_owned());
                        }
                    });
                    if let Some(name) = collision {
                        return Err(format!("duplicate name '{name}'"));
                    }
                    self.bind_pattern(
                        &binding.pattern,
                        init.value,
                        init.widths,
                        &mut pattern_bindings,
                    )?;
                    if !conditional {
                        // A top-level rebinding stays visible in the block
                        // result.
                        pattern_bindings.clear();
                    }
                    continue;
                }
                Stmt::Write { name, value } => {
                    let ty = self
                        .regs
                        .get(name)
                        .ok_or_else(|| format!("'{name}' is not a declared register"))?
                        .clone();
                    let value = self
                        .expr(value, Some(&ty))
                        .map_err(|e| format!("update of '{name}': {e}"))?;
                    if !self.writes.insert(name.clone()) {
                        return Err(format!("multiple updates to register '{name}'"));
                    }
                    CheckedStmt::Write {
                        name: name.clone(),
                        value,
                    }
                }
                Stmt::If { condition, body } => {
                    let condition = self
                        .expr(condition, Some(&Type::Bool))
                        .map_err(|e| format!("write guard: {e}"))?;
                    let start = self.statements.len();
                    self.body(block, body, true)?;
                    CheckedStmt::If {
                        condition,
                        body: self.statements.split_off(start),
                    }
                }
            };
            self.statements.push(checked);
        }
        self.leave_scope(pattern_bindings);
        for name in locals {
            self.names.remove(name);
            self.values.remove(name);
            self.constants.remove(name);
            self.constexpr_names.remove(name);
            self.inferred_bindings.remove(name);
        }
        Ok(())
    }

    /// Resolve an initializer against the previous scope before installing
    /// its type, inferred widths, and any compile-time value.
    fn check_let(
        &mut self,
        name: &str,
        annotation: Option<&Type>,
        initializer: &Expr,
        structural: bool,
    ) -> Result<CheckedStmt, String> {
        let scope::BindingInitializer { value, widths } = self
            .block_initializer(Some(name), annotation, initializer, structural)
            .map_err(|e| format!("let '{name}': {e}"))?;
        let constexpr = value.has_constant_inputs(&self.constexpr_names);
        let constant = self.fold_constant(&value);
        declare(&mut self.names, name)?;
        self.values.insert(name.to_owned(), value.ty.clone());
        if constexpr {
            self.constexpr_names.insert(name.to_owned());
        }
        if let Some(v) = constant {
            self.constants.insert(name.to_owned(), v);
        }
        if let Some(widths) = widths {
            self.inferred_bindings.insert(name.to_owned(), widths);
        }
        Ok(CheckedStmt::Let {
            name: name.to_owned(),
            value,
        })
    }

    /// Elaborate a destructured statement's initializer before installing any
    /// of its names. An unconditional direct block call creates one instance.
    fn pattern_initializer(
        &mut self,
        binding: &LetBinding,
        structural: bool,
    ) -> Result<scope::BindingInitializer, String> {
        self.validate_source_pattern(&binding.pattern)?;
        self.block_initializer(None, binding.ty.as_ref(), &binding.value, structural)
    }

    /// Check a block or unrolled let before entering its source names.
    fn block_initializer(
        &mut self,
        name: Option<&str>,
        annotation: Option<&Type>,
        initializer: &Expr,
        structural: bool,
    ) -> Result<scope::BindingInitializer, String> {
        let ty = annotation.map(|t| self.source_type(t)).transpose()?;
        // Preparing a lexical scope exposes its trailing expression, which
        // still occupies a pure value position inside the original scope.
        let structural = structural && !matches!(initializer, Expr::Scope { .. });
        self.binding_initializer(initializer, ty.as_ref(), |checker, expr| {
            checker.let_value(name, expr, ty.as_ref(), structural)
        })
    }

    /// Propagate expected types to literals, aggregate members, and pure call
    /// arguments.
    fn expr(&mut self, expr: &Expr, expected: Option<&Type>) -> Result<TypedExpr, String> {
        if expected.is_some() {
            return self.expr_in_group(expr, expected);
        }
        if crate::infer_literal::has_scalar_match_result(expr) {
            // Check the inferred group while arm-local aliases still retain
            // their literal widths, including in block initializers.
            return self.with_dslx_value_context(|checker| {
                checker.with_prepared_value(expr, Self::inferred_group_result)
            });
        }
        self.inferred_group_result(expr)
    }

    /// Validate a self-determined result after elaborating its concrete type.
    fn inferred_group_result(&mut self, expr: &Expr) -> Result<TypedExpr, String> {
        let value = self.expr_in_group(expr, None)?;
        check_result_width(&[expr], &value.ty, self.dslx_widths(), &|| {
            self.zero_calls_function()
        })?;
        Ok(value)
    }

    /// Check one member of a group of expressions that share their result
    /// type. The enclosing group checks its complete prefix of unsized
    /// literals.
    fn expr_in_group(&mut self, expr: &Expr, expected: Option<&Type>) -> Result<TypedExpr, String> {
        // Keep the recursive dispatch frame small for expressions at the depth
        // limit.
        let (ty, kind) = match expr {
            Expr::Scope { bindings, result } => {
                return self.value_scope(bindings, result, expected);
            }
            Expr::UnrollFor(fold) => return self.unroll(fold, expected, false),
            Expr::For(fold) => {
                return self.with_dslx_value_context(|checker| {
                    checker.counted_for(fold, CarryContext::Expected(expected))
                });
            }
            Expr::Map(map) => self.with_dslx_value_context(|checker| checker.map_lambda(map)),
            Expr::Name(name) => self.named_value(name),
            Expr::Bool(value) => Ok((Type::Bool, K::Literal(IrValue::bool(*value)))),
            Expr::Zero(ty) => self.zero_value(ty),
            Expr::Literal { ty, text } => self.bit_literal(ty.as_ref(), text, expected),
            Expr::Tuple(items) => self.tuple_literal(items, expected),
            Expr::Array(items) => self.array_literal(items, expected),
            Expr::Index { value, index } => self.array_index(value, index),
            Expr::WidthSlice {
                value,
                start,
                width,
            } => self.width_slice(value, start, width),
            Expr::Struct { name, fields } => self.struct_literal(name, fields),
            Expr::Field { value, name } => self.project_field(value, name),
            Expr::Call { name, args } => self.function_call(name, args),
            Expr::ParametricCall {
                name,
                callbacks,
                args,
            } => {
                if self.decls.is_block(name) && self.callback_index(name).is_none() {
                    self.check_block_callback_list(
                        name,
                        callbacks.len(),
                        self.names.contains(name),
                    )?;
                    return Err(self.block_in_value_error(name));
                }
                self.parametric_call(name, callbacks, args)
            }
            Expr::Match { subject, arms } => self.with_dslx_value_context(|checker| {
                let selector = checker.match_selector(subject, arms)?;
                let results: Vec<_> = arms.iter().map(|arm| &arm.value).collect();
                let values = checker.check_results(&results, expected)?;
                let value = selector.with_values(&mut values.into_iter());
                Ok((value.ty, value.kind))
            }),
            Expr::Cast { value, ty } => self.bit_cast(value, ty),
            Expr::If {
                condition,
                then_value,
                else_value,
            } => self.conditional(condition, then_value, else_value, expected),
            Expr::Unary { op, value } => self.unary(*op, value, expected),
            Expr::Binary { op, left, right } => self.binary(*op, left, right, expected),
        }?;
        self.finish_expr(expr, expected, ty, kind)
    }

    /// Look up a lexical value before consulting the block interface.
    fn named_value(&self, name: &str) -> Result<(Type, K), String> {
        if name == "_" && (self.kind == BodyKind::Function || self.in_loop_scope()) {
            return Err("cannot read the wildcard binding '_'".into());
        }
        if let Some(value) = self.lexical_value(name) {
            let literal = match &value.kind {
                K::Literal(literal) => Some(literal),
                _ => None,
            };
            let work = source_type_nodes(&value.ty) - 1 + literal.map_or(0, literal_copy_work);
            self.charge_reference_copy(work)?;
            return Ok((value.ty.clone(), value.kind.clone()));
        }
        if let Some(ty) = self.values.get(name) {
            let work = source_type_nodes(ty) - 1;
            self.charge_reference_copy(work)?;
            return Ok((ty.clone(), K::Name(name.to_owned())));
        }
        if self.callback_index(name).is_some() {
            return Err(format!(
                "callable parameter '{name}' may only be used as a callee"
            ));
        }
        self.reject_type_parameter_value(name)?;
        if !self.names.contains(name)
            && let Some(value) = self.module_constant(name)
        {
            let work = source_type_nodes(&value.ty) - 1 + literal_copy_work(&value.value);
            self.charge_reference_copy(work)?;
            return Ok((value.ty.clone(), K::Literal(value.value.clone())));
        }
        Err(format!("unknown value '{name}'"))
    }

    /// Bound owned type nodes and literal storage copied by checked references.
    /// Generated bodies charge their shared specialization registry. Keeping
    /// this allowance separate preserves the structural unroll budget.
    fn charge_reference_copy(&self, work: usize) -> Result<(), String> {
        if work == 0 {
            return Ok(());
        }
        let total = match (
            &self.function_registry,
            self.function_spec,
            &self.specializations,
            self.specialization,
        ) {
            (Some(registry), Some(_), _, _) => &registry.reference_copy_work,
            (_, _, Some(registry), Some(_)) => &registry.reference_copy_work,
            _ => &self.reference_copy_work,
        };
        let updated = total.get().saturating_add(work);
        if updated > MAX_TYPE_SIZE {
            return Err(format!(
                "value reference exceeds the expansion limit of {MAX_TYPE_SIZE}"
            ));
        }
        total.set(updated);
        Ok(())
    }

    fn lexical_value(&self, name: &str) -> Option<&TypedExpr> {
        self.loop_values
            .get(name)
            .or_else(|| self.function_values.get(name))
    }

    fn module_constant(&self, name: &str) -> Option<&CheckedConstant> {
        self.decls.constants.get(name).filter(|_| {
            self.decls
                .visible(name, self.function_position.or(self.block_position))
        })
    }

    /// Resolve zero! to its local function or construct the intrinsic zero
    /// value.
    fn zero_value(&mut self, ty: &Type) -> Result<(Type, K), String> {
        let ty = self.source_type(ty)?;
        if self.zero_calls_function()? {
            self.function_call("zero!", &[])
        } else {
            let layout = crate::layout::Layouts::new(&self.structs.definitions)
                .lower(&ty)
                .map_err(|e| e.to_string())?;
            Ok((ty, K::Literal(zero_ir_value_for_type(&layout))))
        }
    }

    /// Check a bit literal at its annotated or inferred width.
    fn bit_literal(
        &mut self,
        annotation: Option<&Type>,
        text: &str,
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        let resolved;
        let annotation = if let Some(ty @ Type::Named(_)) = annotation {
            resolved = self.source_type(ty)?;
            Some(&resolved)
        } else {
            annotation
        };
        let ty = annotation
            .or(expected)
            .ok_or_else(|| format!("literal '{text}' requires a concrete bit type"))?;
        let normalized;
        let text = if text.starts_with('\'') || matches!(text, "true" | "false") {
            normalized = character_or_bool_digits(text, ty)?;
            normalized.as_str()
        } else {
            text
        };
        // An explicitly typed DSLX negative literal negates a
        // width-checked magnitude. For example, s3:-5 represents 3.
        let magnitude = if annotation.is_some() && self.dslx_widths().is_some() {
            text.strip_prefix('-')
        } else {
            None
        };
        let (signed, width) =
            bits(ty).ok_or_else(|| format!("integer literal cannot have type {ty}"))?;
        self.structs.validate_type(ty)?;
        if !signed && text.starts_with('-') {
            return Err(format!("negative literal '{text}' has unsigned type {ty}"));
        }
        let digits = magnitude.unwrap_or(text);
        let value = IrValue::parse_typed(&format!("bits[{width}]:{digits}"))
            .map_err(|e| format!("invalid literal '{ty}:{text}': {e}"))?;
        let value = if magnitude.is_some() {
            IrValue::Bits(value.to_bits().map_err(|e| e.to_string())?.negate())
        } else {
            value
        };
        // Like DSLX, hexadecimal and binary literals can spell a signed
        // bit pattern directly, e.g. s8:0xff represents -1.
        if signed
            && !text.starts_with('-')
            && !text.starts_with("0x")
            && !text.starts_with("0b")
            && value.as_bits().map_err(|e| e.to_string())?.is_negative()
        {
            return Err(format!("literal '{text}' does not fit in {ty}"));
        }
        Ok((ty.clone(), K::Literal(value)))
    }

    /// Check tuple fields against their respective contextual types.
    fn tuple_literal(
        &mut self,
        items: &[Expr],
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        let hints = match expected {
            Some(Type::Tuple(ts)) if ts.len() == items.len() => Some(ts),
            _ => None,
        };
        let values = items
            .iter()
            .enumerate()
            .map(|(i, value)| match hints {
                Some(types) => self.expr(value, Some(&types[i])),
                None => self.inferred_expr(value),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((
            Type::Tuple(values.iter().map(|v| v.ty.clone()).collect()),
            K::Tuple(values),
        ))
    }

    /// Infer a common element type and preserve the type of an empty array.
    fn array_literal(
        &mut self,
        items: &[Expr],
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        let hint = match expected {
            Some(Type::Array { element, size }) => {
                if *size != items.len() {
                    return Err(format!("expected array length {size}, got {}", items.len()));
                }
                Some(element.as_ref())
            }
            Some(other) => return Err(format!("expected {other}, got an array")),
            None => None,
        };
        Ok(if !items.is_empty() {
            let values = match hint {
                Some(element) => items
                    .iter()
                    .map(|item| self.expr(item, Some(element)))
                    .collect::<Result<Vec<_>, _>>()?,
                None => self.unify_values(&items.iter().collect::<Vec<_>>())?,
            };
            (
                Type::Array {
                    element: Box::new(values[0].ty.clone()),
                    size: items.len(),
                },
                K::Array(values),
            )
        } else {
            let element =
                hint.ok_or_else(|| "empty array literal requires an array type".to_owned())?;
            let mut layouts = crate::layout::Layouts::new(&self.structs.definitions);
            let element_ir = layouts.lower(element).map_err(|e| e.to_string())?;
            let empty = IrValue::make_array_typed(element_ir, &[]).map_err(|e| e.to_string())?;
            (
                Type::Array {
                    element: Box::new(element.clone()),
                    size: 0,
                },
                K::Literal(empty),
            )
        })
    }

    /// Check the index against the array type and any compile-time bounds.
    fn array_index(&mut self, value: &Expr, index: &Expr) -> Result<(Type, K), String> {
        let array = self.expr(value, None)?;
        let Type::Array { element, size } = &array.ty else {
            return Err(format!("indexing requires an array, got {}", array.ty));
        };
        if *size == 0 {
            return Err("cannot index an empty array".into());
        }
        let index = self.unsigned_index(index, Some(32))?;
        if index.has_constant_inputs(&self.constexpr_names)
            && let Some(value) = self.checked_constant(&index)?
            && ir_bits_to_usize(value.as_bits().map_err(|e| e.to_string())?)
                .is_none_or(|i| i >= *size)
        {
            return Err(format!(
                "constant index {value} out of bounds for {}",
                array.ty
            ));
        }
        Ok((
            element.as_ref().clone(),
            K::Index {
                value: Box::new(array),
                index: Box::new(index),
            },
        ))
    }

    /// Check the bit slice and resolve its signedness and width.
    fn width_slice(
        &mut self,
        value: &Expr,
        start: &Expr,
        width: &WidthSliceType,
    ) -> Result<(Type, K), String> {
        let value = self.inferred_expr(value)?;
        let Some((false, source_width)) = bits(&value.ty) else {
            return Err(format!(
                "width slice requires an unsigned bit value, got {}",
                value.ty
            ));
        };
        let ty = self.slice_result_type(width)?;
        let Some((_, result_width)) = bits(&ty) else {
            return Err(format!("width slice requires a bit result type, got {ty}"));
        };
        if result_width > source_width {
            return Err(format!(
                "slice width {result_width} exceeds input width {source_width}"
            ));
        }
        let start = self.unsigned_index(start, Some(source_width))?;
        Ok((
            ty,
            K::WidthSlice {
                value: Box::new(value),
                start: Box::new(start),
            },
        ))
    }

    /// Resolve the struct and check its fields in declaration order.
    fn struct_literal(
        &mut self,
        name: &str,
        fields: &[(String, Expr)],
    ) -> Result<(Type, K), String> {
        let ty = self.source_type(&Type::Named(name.to_owned()))?;
        let Type::Named(canonical) = &ty else {
            return Err(format!("'{name}' is not a struct"));
        };
        let members = self.structs.definition(canonical)?.fields.clone();
        let mut supplied = BTreeMap::new();
        for (field, value) in fields {
            if supplied.insert(field.as_str(), value).is_some() {
                return Err(format!("duplicate field '{name}.{field}'"));
            }
        }
        let mut values = Vec::new();
        for p in &members {
            let v = supplied
                .remove(p.name.as_str())
                .ok_or_else(|| format!("missing field '{name}.{}'", p.name))?;
            values.push(
                self.expr(v, Some(&p.ty))
                    .map_err(|e| format!("field '{name}.{}': {e}", p.name))?,
            );
        }
        if let Some(field) = supplied.keys().next() {
            return Err(format!("unknown field '{name}.{field}'"));
        }
        Ok((ty, K::Tuple(values)))
    }

    /// Resolve an instance output, tuple element, or named struct field.
    fn project_field(&mut self, value: &Expr, name: &str) -> Result<(Type, K), String> {
        Ok(
            if let Expr::Name(instance) = value
                && !self.loop_values.contains_key(instance)
                && !self.function_values.contains_key(instance)
                && let Some(child) = self.instances.get(instance)
            {
                let p = child
                    .outputs
                    .iter()
                    .find(|p| p.name == name)
                    .ok_or_else(|| format!("'{instance}.{name}' is not an output"))?;
                (
                    p.ty.clone(),
                    K::InstanceOutput {
                        instance: instance.clone(),
                        port: name.to_owned(),
                    },
                )
            } else {
                let v = self.expr(value, None)?;
                let (index, ty) = match &v.ty {
                    Type::Tuple(items) => {
                        let index: usize = name
                            .parse()
                            .map_err(|_| format!("invalid tuple field '{name}'"))?;
                        (
                            index,
                            items
                                .get(index)
                                .ok_or_else(|| format!("tuple field {index} out of bounds"))?
                                .clone(),
                        )
                    }
                    Type::Named(s) => {
                        let fields = &self.structs.definition(s)?.fields;
                        let (i, p) = fields
                            .iter()
                            .enumerate()
                            .find(|(_, p)| p.name == name)
                            .ok_or_else(|| format!("unknown field '{s}.{name}'"))?;
                        (i, p.ty.clone())
                    }
                    other => return Err(format!("cannot project field '{name}' from {other}")),
                };
                (
                    ty,
                    K::TupleIndex {
                        value: Box::new(v),
                        index,
                    },
                )
            },
        )
    }

    /// Check a cast between two bit types.
    fn bit_cast(&mut self, value: &Expr, ty: &Type) -> Result<(Type, K), String> {
        let ty = self.source_type(ty)?;
        let v = self.inferred_expr(value)?;
        if bits(&v.ty).is_none() || bits(&ty).is_none() {
            return Err(format!("cast from {} to {ty} requires bit types", v.ty));
        }
        Ok((ty, K::Cast { value: Box::new(v) }))
    }

    /// Check a boolean condition and infer its branch results together.
    fn conditional(
        &mut self,
        condition: &Expr,
        then_value: &Expr,
        else_value: &Expr,
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        let cond = self.expr(condition, Some(&Type::Bool))?;
        let [t, f] = self
            .check_results(&[then_value, else_value], expected)?
            .try_into()
            .expect("two checked branches");
        Ok((
            t.ty.clone(),
            K::If {
                condition: Box::new(cond),
                then_value: Box::new(t),
                else_value: Box::new(f),
            },
        ))
    }

    /// Check a unary bit operation in its operand context.
    fn unary(
        &mut self,
        op: UnaryOp,
        value: &Expr,
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        let v = self.expr_in_group(value, expected)?;
        if bits(&v.ty).is_none() {
            return Err(format!("invalid unary '{op}' for {}", v.ty));
        }
        Ok((
            v.ty.clone(),
            K::Unary {
                op,
                value: Box::new(v),
            },
        ))
    }

    /// Check the result against its context after recursive operands return.
    fn finish_expr(
        &mut self,
        expr: &Expr,
        expected: Option<&Type>,
        ty: Type,
        kind: K,
    ) -> Result<TypedExpr, String> {
        let callback_block = self
            .specialization
            .is_some_and(|spec| !spec.callbacks.is_empty());
        if self.checking_callback_inputs
            || ((self.source_template.is_some()
                || callback_block
                || self.map_outer_iterations.is_some())
                && self.replaying_body())
        {
            // Materialize checked IR types incrementally so neither a copied
            // body nor a long list of aggregate inputs can overrun the budget.
            let nodes = self
                .structs
                .type_shape(&ty, MAX_ELABORATION_DEPTH)?
                .ir_nodes;
            let literal_work = match &kind {
                K::Literal(value) => crate::typed::constant_work(value),
                _ => 0,
            };
            let body = if self.checking_callback_inputs {
                "callback block input"
            } else if self.map_outer_iterations.is_some() {
                "map lambda"
            } else if callback_block {
                "block body"
            } else {
                "function body"
            };
            self.charge_expansion(body, nodes.saturating_add(literal_work))?;
        }
        let scoped_projection = self.reference_type(expr).is_some_and(|(name, _)| {
            self.inferred_bindings
                .get(name)
                .is_some_and(|widths| widths.infer_at_use)
        });
        let dslx_projection =
            (self.kind == BodyKind::Function || self.dslx_value_context || scoped_projection)
                && matches!(
                    expr,
                    Expr::Name(_) | Expr::Field { .. } | Expr::Index { .. }
                );
        let dslx_bindings =
            (self.kind == BodyKind::Function || self.dslx_value_context || scoped_projection)
                .then_some(&self.inferred_bindings);
        if dslx_projection {
            // Each nested aggregate infers its fields before a typed sibling
            // or the enclosing expression supplies a signed result type.
            check_argument_widths(expr, expected.unwrap_or(&ty), dslx_bindings, None)?;
        }
        if let Some(expected) = expected
            && !compatible(&ty, expected)
        {
            if dslx_projection
                && projected_type_compatible(expr, &ty, expected, dslx_bindings, false)?
            {
                // Only signedness differs. Preserve the complete IR layout.
                return Ok(TypedExpr {
                    ty: expected.clone(),
                    kind,
                });
            }
            if scoped_projection
                && projected_type_compatible(expr, &ty, expected, dslx_bindings, true)?
            {
                return self.widen_projection(TypedExpr { ty, kind }, expected);
            }
            // DSLX retains the minimum type at the binder and can give a
            // reference to that value a wider concrete type.
            if let Some(locals) = self.scoped_widths()
                && locals.is_inferred_reference(expr)?
                && let (Some((signed, width)), Some((target_signed, target_width))) =
                    (bits(&ty), bits(expected))
                && (!signed || target_signed)
                && target_width >= width + usize::from(!signed && target_signed)
            {
                return Ok(TypedExpr {
                    ty: expected.clone(),
                    kind: K::Cast {
                        value: Box::new(TypedExpr { ty, kind }),
                    },
                });
            }
            return Err(format!("expected {expected}, got {ty}"));
        }
        if matches!(expr, Expr::Tuple(_) | Expr::Array(_)) {
            self.structs.validate_type(&ty)?;
        }
        let value = TypedExpr { ty, kind };
        if !self.iteration_path.is_empty()
            && matches!(
                value.kind,
                K::ArrayUpdate { .. } | K::Array(_) | K::Tuple(_) | K::Literal(_)
            )
            && value.has_constant_inputs(&self.constexpr_names)
        {
            // Rebuilding a constant array each iteration can take quadratic
            // work. Charge its size in machine words when elaborating the fold.
            let size = self
                .structs
                .type_shape(&value.ty, MAX_ELABORATION_DEPTH)?
                .size;
            self.charge_expansion("unroll_for!", size.div_ceil(64))?;
        }
        Ok(value)
    }

    /// Local function expressions and their arguments can use an unannotated
    /// binding's literal minimum to find the first concrete operand.
    fn dslx_widths(&self) -> Option<WidthLookup<'_>> {
        (self.kind == BodyKind::Function || self.dslx_value_context)
            .then_some(WidthLookup::DslxLiterals(Some(&self.inferred_bindings)))
    }

    fn common_widths(&self) -> Option<WidthLookup<'_>> {
        self.dslx_widths().or_else(|| self.scoped_widths())
    }

    /// Find the stored type and binding behind a reference without evaluating
    /// its index or copying the aggregate value.
    fn reference_type<'e>(&'e self, expr: &'e Expr) -> Option<(&'e str, &'e Type)> {
        match expr {
            Expr::Name(name) => {
                let ty = self
                    .lexical_value(name)
                    .map(|v| &v.ty)
                    .or_else(|| self.values.get(name))
                    .or_else(|| self.module_constant(name).map(|c| &c.ty))?;
                Some((name, ty))
            }
            Expr::Field { value, name } => {
                let (binding, Type::Tuple(fields)) = self.reference_type(value)? else {
                    return None;
                };
                Some((binding, fields.get(name.parse::<usize>().ok()?)?))
            }
            Expr::Index { value, .. } => {
                let (binding, Type::Array { element, .. }) = self.reference_type(value)? else {
                    return None;
                };
                Some((binding, element))
            }
            _ => None,
        }
    }

    /// An inferred aggregate can receive a leaf's type from another operand,
    /// including when the binding appears inside a tuple or array constructor.
    fn contains_inferred_aggregate(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Tuple(items) | Expr::Array(items) => items
                .iter()
                .any(|item| self.contains_inferred_aggregate(item)),
            Expr::If {
                then_value,
                else_value,
                ..
            } => {
                self.contains_inferred_aggregate(then_value)
                    || self.contains_inferred_aggregate(else_value)
            }
            Expr::Match { arms, .. } => arms
                .iter()
                .any(|arm| self.contains_inferred_aggregate(&arm.value)),
            _ => self.reference_type(expr).is_some_and(|(name, ty)| {
                matches!(ty, Type::Array { .. } | Type::Tuple(_))
                    && self.inferred_bindings.get(name).is_some_and(|widths| {
                        self.dslx_widths().is_some()
                            || self.function_values.contains_key(name)
                            || widths.infer_at_use
                    })
            }),
        }
    }

    fn common_hint(&self, items: &[&Expr]) -> Result<CommonTypeHint, String> {
        if items
            .iter()
            .any(|item| self.contains_inferred_aggregate(item))
        {
            for (i, item) in items.iter().enumerate() {
                if !self.contains_inferred_aggregate(item)
                    && has_concrete_element_type(item, self.common_widths())?
                {
                    return Ok(CommonTypeHint::FromExpr(i));
                }
            }
        }
        common_type_hint(items, self.common_widths())
    }

    /// Infer corresponding signed leaves across already evaluated aggregate
    /// references. Different tuple fields may get their concrete signed type
    /// from different operands. Each operand is subsequently checked against
    /// the result, including the original literal constraint at each leaf.
    fn bound_aggregate_hint(&self, items: &[&Expr]) -> Result<Option<Type>, String> {
        fn merge(ty: &mut Type, other: &Type) -> bool {
            if let (Some((a, width)), Some((b, other_width))) = (bits(ty), bits(other)) {
                if other_width != width || b && !a {
                    *ty = Type::Bits {
                        signed: a || b,
                        width: width.max(other_width),
                    };
                }
                return true;
            }
            match (ty, other) {
                (Type::Tuple(a), Type::Tuple(b)) => {
                    a.len() == b.len() && a.iter_mut().zip(b).all(|(a, b)| merge(a, b))
                }
                (
                    Type::Array {
                        element: a,
                        size: n,
                    },
                    Type::Array {
                        element: b,
                        size: m,
                    },
                ) => n == m && merge(a, b),
                (a, b) => a == b,
            }
        }
        fn operand(checker: &BodyChecker<'_, '_>, expr: &Expr) -> Result<Option<Type>, String> {
            match expr {
                Expr::Tuple(fields) => {
                    let mut types = Vec::with_capacity(fields.len());
                    for field in fields {
                        let Some(ty) = operand(checker, field)? else {
                            return Ok(None);
                        };
                        types.push(ty);
                    }
                    Ok(Some(Type::Tuple(types)))
                }
                Expr::Array(items) => {
                    Ok(
                        merged_operands(checker, items.iter())?.map(|element| Type::Array {
                            element: Box::new(element),
                            size: items.len(),
                        }),
                    )
                }
                Expr::If {
                    then_value,
                    else_value,
                    ..
                } => merged_operands(
                    checker,
                    [then_value.as_ref(), else_value.as_ref()].into_iter(),
                ),
                Expr::Match { arms, .. } => {
                    merged_operands(checker, arms.iter().map(|arm| &arm.value))
                }
                Expr::Bool(_) => Ok(Some(Type::Bool)),
                Expr::Literal { ty: Some(ty), .. } => {
                    let ty = match (ty, checker.specialization) {
                        (Type::Named(name), Some(spec)) => spec.bindings.get(name),
                        _ => Some(ty),
                    };
                    Ok(ty.filter(|ty| bits(ty).is_some()).cloned())
                }
                _ => {
                    if let Some((_, ty)) = checker.reference_type(expr) {
                        Ok(Some(ty.clone()))
                    } else {
                        minimum_operand_type(expr, checker.common_widths())
                    }
                }
            }
        }
        fn merged_operands<'e>(
            checker: &BodyChecker<'_, '_>,
            mut items: impl Iterator<Item = &'e Expr>,
        ) -> Result<Option<Type>, String> {
            let Some(first) = items.next() else {
                return Ok(None);
            };
            let Some(mut ty) = operand(checker, first)? else {
                return Ok(None);
            };
            for item in items {
                if !operand(checker, item)?.is_some_and(|other| merge(&mut ty, &other)) {
                    return Ok(None);
                }
            }
            Ok(Some(ty))
        }
        if !items
            .iter()
            .any(|item| self.contains_inferred_aggregate(item))
        {
            return Ok(None);
        }
        let Some(mut ty) = merged_operands(self, items.iter().copied())? else {
            return Ok(None);
        };
        unify_inferred_members(&mut ty, items, &self.inferred_bindings)?;
        Ok(Some(ty))
    }

    /// Check values against a common type, including the corresponding
    /// members of nested aggregate literals. Keep the resulting values in
    /// source order and reuse the expression that supplied a concrete type.
    fn unify_values(&mut self, items: &[&Expr]) -> Result<Vec<TypedExpr>, String> {
        if items.iter().any(|e| matches!(e, Expr::For(fold) if fold.annotation.is_none() && has_aggregate_literal(&fold.seed))) {
            return self.with_dslx_value_context(|checker| checker.unify_loop_seeds(items));
        }
        if self.dslx_widths().is_none()
            && items.iter().any(|item| matches!(item, Expr::Match { .. }))
        {
            return self.with_dslx_value_context(|checker| checker.unify_values(items));
        }
        if items.iter().any(|e| matches!(e, Expr::Scope { .. })) {
            return self.with_scoped_results(items, Self::unify_values);
        }
        if items.iter().any(|e| self.contains_inferred_aggregate(e))
            && items.iter().any(|e| has_aggregate_literal(e))
        {
            return self.with_aggregate_operands(items, Self::unify_prepared_values);
        }
        self.unify_prepared_values(items)
    }

    /// Elaborate operands while any concrete anchors and lexical scope aliases
    /// from preparation are still available.
    fn unify_prepared_values(&mut self, items: &[&Expr]) -> Result<Vec<TypedExpr>, String> {
        if let Some(ty) = self.bound_aggregate_hint(items)? {
            return items
                .iter()
                .map(|item| self.expr(item, Some(&ty)))
                .collect();
        }
        let first_has_type = !self.contains_inferred_aggregate(items[0])
            && has_concrete_element_type(items[0], self.common_widths())?;
        if !first_has_type
            && items
                .iter()
                .any(|item| matches!(item, Expr::If { .. } | Expr::Match { .. }))
        {
            // Every conditional branch contributes to the same element type.
            let mut branches = Vec::new();
            let mut conditions = Vec::new();
            for item in items {
                match item {
                    Expr::If {
                        condition,
                        then_value,
                        else_value,
                    } => {
                        conditions
                            .push(BranchSelector::If(self.expr(condition, Some(&Type::Bool))?));
                        branches.extend([then_value.as_ref(), else_value.as_ref()]);
                    }
                    Expr::Match { subject, arms } => {
                        conditions.push(BranchSelector::Match(self.match_selector(subject, arms)?));
                        branches.extend(arms.iter().map(|arm| &arm.value));
                    }
                    _ => {
                        conditions.push(BranchSelector::Value);
                        branches.push(*item);
                    }
                }
            }
            let mut branches = self.unify_values(&branches)?.into_iter();
            return Ok(conditions
                .into_iter()
                .map(|condition| match condition {
                    BranchSelector::Value => branches.next().expect("one value per item"),
                    BranchSelector::Match(selector) => selector.with_values(&mut branches),
                    BranchSelector::If(condition) => {
                        let value = branches.next().expect("then branch");
                        TypedExpr {
                            ty: value.ty.clone(),
                            kind: K::If {
                                condition: Box::new(condition),
                                then_value: Box::new(value),
                                else_value: Box::new(
                                    branches.next().expect("two branches per conditional"),
                                ),
                            },
                        }
                    }
                })
                .collect());
        }
        if !first_has_type
            && let Expr::Tuple(first) = items[0]
            && let Some(tuples) = items
                .iter()
                .map(|item| match item {
                    Expr::Tuple(fields) if fields.len() == first.len() => Some(fields),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        {
            let mut values = vec![Vec::with_capacity(first.len()); tuples.len()];
            let mut types = Vec::with_capacity(first.len());
            for i in 0..first.len() {
                let fields: Vec<_> = tuples.iter().map(|tuple| &tuple[i]).collect();
                let fields = self.unify_values(&fields)?;
                types.push(fields[0].ty.clone());
                for (tuple, field) in values.iter_mut().zip(fields) {
                    tuple.push(field);
                }
            }
            let ty = Type::Tuple(types);
            return Ok(values
                .into_iter()
                .map(|fields| TypedExpr {
                    ty: ty.clone(),
                    kind: K::Tuple(fields),
                })
                .collect());
        }
        if !first_has_type
            && let Expr::Array(first) = items[0]
            && let Some(arrays) = items
                .iter()
                .map(|item| match item {
                    Expr::Array(elements) if elements.len() == first.len() => Some(elements),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        {
            let fields: Vec<_> = arrays.into_iter().flatten().collect();
            if fields.is_empty() {
                return Err("empty array literal requires an array type".into());
            }
            let fields = self.unify_values(&fields)?;
            let ty = Type::Array {
                element: Box::new(fields[0].ty.clone()),
                size: first.len(),
            };
            let mut fields = fields.into_iter();
            return Ok(items
                .iter()
                .map(|_| TypedExpr {
                    ty: ty.clone(),
                    kind: K::Array(fields.by_ref().take(first.len()).collect()),
                })
                .collect());
        }
        let hint = if first_has_type {
            CommonTypeHint::FromExpr(0)
        } else {
            self.common_hint(items)?
        };
        match hint {
            CommonTypeHint::Minimum(ty) => items
                .iter()
                .map(|item| self.expr(item, Some(&ty)))
                .collect(),
            CommonTypeHint::FromExpr(i) => {
                let anchor = self.expr(items[i], None)?;
                if let Some(widths) = self.dslx_widths() {
                    check_result_width(items, &anchor.ty, Some(widths), &|| {
                        self.zero_calls_function()
                    })?;
                }
                self.check_siblings(items, i, anchor)
            }
        }
    }

    /// An aggregate carry may obtain different field types from different
    /// siblings. Unify its seed once and use the resulting value for the loop.
    fn unify_loop_seeds(&mut self, items: &[&Expr]) -> Result<Vec<TypedExpr>, String> {
        let folds: Vec<_> = items
            .iter()
            .map(|item| match item {
                Expr::For(fold)
                    if fold.annotation.is_none() && has_aggregate_literal(&fold.seed) =>
                {
                    Some(fold)
                }
                _ => None,
            })
            .collect();
        let seeds: Vec<_> = items
            .iter()
            .zip(&folds)
            .map(|(item, fold)| fold.map_or(*item, |f| &f.seed))
            .collect();
        let values = self.unify_values(&seeds)?;
        folds
            .into_iter()
            .zip(values)
            .map(|(fold, value)| match fold {
                Some(f) => self.with_dslx_value_context(|checker| {
                    checker.counted_for(f, CarryContext::Prepared(value))
                }),
                None => Ok(value),
            })
            .collect()
    }

    /// Check the other expressions against the anchor, returning each checked
    /// value in source order without evaluating the anchor a second time.
    fn check_siblings(
        &mut self,
        items: &[&Expr],
        index: usize,
        anchor: TypedExpr,
    ) -> Result<Vec<TypedExpr>, String> {
        let ty = anchor.ty.clone();
        let mut values = Vec::with_capacity(items.len());
        for item in &items[..index] {
            values.push(self.expr_in_group(item, Some(&ty))?);
        }
        values.push(anchor);
        for item in &items[index + 1..] {
            values.push(self.expr_in_group(item, Some(&ty))?);
        }
        Ok(values)
    }

    /// Use DSLX inference for new value expressions in either kind of body.
    fn with_dslx_value_context<T>(
        &mut self,
        check: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let outer = std::mem::replace(&mut self.dslx_value_context, true);
        let result = check(self);
        self.dslx_value_context = outer;
        result
    }

    /// Resolve the spelling shared by the zero intrinsic and a function.
    fn zero_calls_function(&self) -> Result<bool, String> {
        match self.kind {
            _ if self.names.contains("zero!") => Err("'zero!' names a local value here".into()),
            _ if self.callback_index("zero!").is_some() => {
                Err("callable parameter 'zero!' requires a direct call".into())
            }
            BodyKind::Function => Ok(self.decls.visible("zero!", self.function_position)
                && (self.decls.functions.contains_key("zero!")
                    || self.resolver.lookup("zero!")?.is_some())),
            // Only a local function can override zero! in a BX block.
            BodyKind::Block { .. } => {
                Ok(self.decls.functions.get("zero!").is_some_and(|f| f.local))
            }
        }
    }

    /// The subject and literal patterns share one type. An explicitly typed
    /// endpoint can supply the width for an otherwise untyped subject.
    fn match_selector(
        &mut self,
        subject: &Expr,
        arms: &[MatchArm],
    ) -> Result<MatchSelector, String> {
        self.with_prepared_value(subject, |checker, subject| {
            let operands = match_operands(subject, arms, |name| {
                !checker.names.contains(name)
                    && checker.callback_index(name).is_none()
                    && checker.module_constant(name).is_some()
            })?;
            let mut values = checker.check_results(&operands, None)?.into_iter();
            let subject = values.next().expect("subject is the first operand");
            check_match_operand_width(&operands, &subject.ty, checker.dslx_widths(), &|| {
                checker.zero_calls_function()
            })?;
            let patterns = match_patterns::check_patterns(&subject.ty, arms, values)?;
            Ok(MatchSelector { subject, patterns })
        })
    }

    /// Check the alternatives of a value-producing expression as one result
    /// group. Aggregate literals can obtain their member types from later arms.
    /// The enclosing expression must include all arms when checking the
    /// inferred width of a prefix of signed and unsigned literals.
    fn check_results(
        &mut self,
        items: &[&Expr],
        expected: Option<&Type>,
    ) -> Result<Vec<TypedExpr>, String> {
        if expected.is_none() && items.iter().any(|e| matches!(e, Expr::Scope { .. })) {
            return self.with_scoped_results(items, |checker, items| {
                checker.check_results(items, expected)
            });
        }
        if expected.is_none() && items.iter().any(|e| has_aggregate_literal(e)) {
            return self.unify_values(items);
        }
        let (index, anchor) = self.result_anchor(items, expected)?;
        self.check_siblings(items, index, anchor)
    }

    /// Start at the first result with a concrete type, or check the final
    /// result against the group's minimum if all results are implicit.
    fn result_anchor(
        &mut self,
        items: &[&Expr],
        expected: Option<&Type>,
    ) -> Result<(usize, TypedExpr), String> {
        let first = items
            .first()
            .ok_or("expected at least one result expression")?;
        if let Some(ty) = expected {
            return Ok((0, self.expr_in_group(first, Some(ty))?));
        }
        if let Some(ty) = self.bound_aggregate_hint(items)? {
            return Ok((0, self.expr_in_group(first, Some(&ty))?));
        }
        match self.common_hint(items)? {
            CommonTypeHint::FromExpr(i) => {
                let anchor = self.expr_in_group(items[i], None)?;
                if let Some(widths) = self.dslx_widths() {
                    check_result_width(items, &anchor.ty, Some(widths), &|| {
                        self.zero_calls_function()
                    })?;
                }
                Ok((i, anchor))
            }
            CommonTypeHint::Minimum(ty) => {
                let i = items.len() - 1;
                Ok((i, self.expr_in_group(items[i], Some(&ty))?))
            }
        }
    }

    /// Hardware instances require a structural binding in a block.
    fn block_in_value_error(&self, name: &str) -> String {
        if self.kind == BodyKind::Function {
            format!("pure function cannot call block '{name}'")
        } else {
            format!(
                "call to block '{name}' must be the entire right-hand side of an unconditional top-level let"
            )
        }
    }

    /// Resolve a local or extern signature or infer an imported DSLX
    /// specialization.
    fn function_call(&mut self, name: &str, args: &[Expr]) -> Result<(Type, K), String> {
        if self.loop_values.contains_key(name) {
            return Err(format!("'{name}' names a loop-local value"));
        }
        if self.function_values.contains_key(name) {
            return Err(format!("'{name}' names a local value here"));
        }
        if let Some(index) = self.callback_index(name) {
            if self.names.contains(name) {
                return Err(format!("'{name}' names a local value here"));
            }
            return self.call_formal(index, args);
        }
        if let Some((head, _)) = name.split_once("::")
            && self.callback_index(head).is_some()
        {
            return Err(format!(
                "callable parameter '{head}' may only be used as a callee"
            ));
        }
        self.reject_type_parameter_value(name)?;
        let visible = self.decls.visible(name, self.function_position);
        if !visible && name != "update" {
            return Err(format!("function '{name}' is declared later in the module"));
        }
        if visible
            && let Some((subject, function)) =
                lookup_use(name, self.resolver, |binding| self.names.contains(binding))?
        {
            let args = args
                .iter()
                .enumerate()
                .map(|(i, arg)| {
                    self.expr(arg, None)
                        .map_err(|e| format!("argument {}, call to '{name}': {e}", i + 1))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let types = args.iter().map(|arg| arg.ty.clone()).collect();
            let (id, ty) = self.resolver.resolve(&subject, &function, types)?;
            self.structs.update_imports(self.resolver)?;
            self.structs.validate_type(&ty)?;
            return Ok((ty, K::ImportedCall { id, args }));
        }
        if visible && self.decls.is_block(name) {
            return Err(self.block_in_value_error(name));
        }
        if name == "update" && (!visible || !self.decls.functions.contains_key(name)) {
            if self.names.contains(name) {
                return Err(format!("'{name}' names a local value here"));
            }
            return self.array_update(args);
        }
        let (&callee, f) = self
            .decls
            .functions
            .get_key_value(name)
            .ok_or_else(|| format!("unknown function '{name}'"))?;
        if self.names.contains(name) && (f.local || self.kind == BodyKind::Function) {
            return Err(format!("'{name}' names a local value here"));
        }
        if !f.callback_params.is_empty() {
            return Err(format!(
                "'{name}' requires {} callable arguments",
                f.callback_params.len()
            ));
        }
        if f.local && self.kind == BodyKind::Function {
            self.local_callees.insert(callee.to_owned());
        }
        // BX block-to-extern calls use their declared port width as context.
        // Restore the outer argument's rules after checking a nested call.
        let dslx_arguments = f.local || self.kind == BodyKind::Function;
        let args = self.call_arguments(
            name,
            args,
            f.params.iter().map(|p| (Some(p.name.as_str()), &p.ty)),
            dslx_arguments,
        )?;
        Ok((
            f.result.clone(),
            K::Call {
                name: name.to_owned(),
                args,
            },
        ))
    }

    /// Check DSLX's single-index array update with independently inferred
    /// arguments.
    fn array_update(&mut self, args: &[Expr]) -> Result<(Type, K), String> {
        let [array, index, replacement] = args else {
            return Err(format!("'update' expects 3 arguments, got {}", args.len()));
        };
        let array = self.expr(array, None)?;
        let Type::Array { element, .. } = &array.ty else {
            return Err(format!("'update' requires an array, got {}", array.ty));
        };
        // DSLX infers the index and replacement type parameters separately.
        // An inline unsized scalar cannot supply either type parameter.
        for (arg, role) in [(index, "index"), (replacement, "replacement")] {
            if minimum_type(arg)?.is_some() {
                return Err(format!("cannot infer the type of 'update' {role}"));
            }
        }
        let index = self.expr(index, None)?;
        if !bits(&index.ty).is_some_and(|(signed, _)| !signed) {
            return Err(format!(
                "'update' requires an unsigned index, got {}",
                index.ty
            ));
        }
        let replacement = self.expr(replacement, None)?;
        if !compatible(element, &replacement.ty) {
            return Err(format!(
                "'update' replacement expects {element}, got {}",
                replacement.ty
            ));
        }
        Ok((
            array.ty.clone(),
            K::ArrayUpdate {
                array: Box::new(array),
                index: Box::new(index),
                replacement: Box::new(replacement),
            },
        ))
    }

    /// Check a dimension's operands at the widths used by DSLX constexpr
    /// evaluation.
    fn dimension_expr(&mut self, expr: &Expr, context_width: usize) -> Result<TypedExpr, String> {
        match expr {
            Expr::Match { subject, arms } => self.with_dslx_value_context(|checker| {
                let selector = checker.match_selector(subject, arms)?;
                let subject = checker
                    .checked_constant(&selector.subject)?
                    .ok_or("slice type dimension must be a compile-time constant")?;
                let value = subject.as_bits().map_err(|e| e.to_string())?;
                let signed = bits(&selector.subject.ty).expect("checked scalar match").0;
                let selected = selector
                    .patterns
                    .iter()
                    .position(|patterns| patterns.iter().any(|p| p.matches(value, signed)))
                    .expect("exhaustive match");
                let values = arms
                    .iter()
                    .map(|arm| checker.dimension_expr(&arm.value, context_width))
                    .collect::<Result<Vec<_>, _>>()?;
                let ty = values[selected].ty.clone();
                let mut value = selector.with_values(&mut values.into_iter());
                value.ty = ty;
                Ok(value)
            }),
            Expr::If {
                condition,
                then_value,
                else_value,
            } => {
                // The chosen branch's physical width is preserved when it
                // becomes an operand of further constexpr arithmetic.
                let condition = self.expr(condition, Some(&Type::Bool))?;
                let select_then = self
                    .checked_constant(&condition)?
                    .ok_or("slice type dimension must be a compile-time constant")?
                    .to_bool()
                    .map_err(|e| e.to_string())?;
                let then_value = self.dimension_expr(then_value, context_width)?;
                let else_value = self.dimension_expr(else_value, context_width)?;
                let ty = if select_then {
                    &then_value.ty
                } else {
                    &else_value.ty
                };
                Ok(TypedExpr {
                    ty: ty.clone(),
                    kind: K::If {
                        condition: Box::new(condition),
                        then_value: Box::new(then_value),
                        else_value: Box::new(else_value),
                    },
                })
            }
            Expr::Unary { op, value } => {
                let value = self.dimension_expr(value, context_width)?;
                Ok(TypedExpr {
                    ty: value.ty.clone(),
                    kind: K::Unary {
                        op: *op,
                        value: Box::new(value),
                    },
                })
            }
            Expr::Binary { op, left, right }
                if has_common_bit_type(*op) || matches!(op, BinaryOp::Shl | BinaryOp::Shr) =>
            {
                let left = self.dimension_expr(left, context_width)?;
                let right = if has_common_bit_type(*op) {
                    let right = self.dimension_expr(right, context_width)?;
                    if !compatible(&left.ty, &right.ty) {
                        return Err(format!(
                            "'{op}' in slice dimension requires matching operand types, got {} and {}",
                            left.ty, right.ty
                        ));
                    }
                    right
                } else {
                    // DSLX checks the shift against the dimension's required
                    // type, then evaluates it on the selected physical bits.
                    self.shift_distance(right, context_width)?
                };
                Ok(TypedExpr {
                    ty: left.ty.clone(),
                    kind: K::Binary {
                        op: *op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                })
            }
            _ => {
                let operands = context_operands(expr, &self.inferred_bindings, context_width)?;
                let context = Type::Bits {
                    signed: false,
                    width: context_width,
                };
                self.expr(
                    expr,
                    (operands != ContextOperands::Bound).then_some(&context),
                )
            }
        }
    }

    /// Evaluate an unsigned DSLX type dimension under its required bit width.
    fn slice_dimension(&mut self, expr: &Expr, context_width: usize) -> Result<usize, String> {
        let dimension = self.dimension_expr(expr, context_width)?;
        if !dimension.has_constant_inputs(&self.constexpr_names) {
            return Err("slice type dimension must be a compile-time constant".into());
        }
        let value = self
            .checked_constant(&dimension)?
            .ok_or("slice type dimension must be a compile-time constant")?;
        ir_bits_to_usize(value.as_bits().map_err(|e| e.to_string())?)
            .ok_or("slice type dimension does not fit usize".into())
    }

    /// Resolve the bit type after `+:`, including constant width and signedness
    /// expressions.
    fn slice_result_type(&mut self, width: &WidthSliceType) -> Result<Type, String> {
        match width {
            WidthSliceType::Concrete(ty) => self.source_type(ty),
            WidthSliceType::Bits { signed, bit_count } => Ok(Type::Bits {
                signed: self.slice_dimension(signed, 1)? != 0,
                width: self.slice_dimension(bit_count, 32)?,
            }),
        }
    }

    /// In a self-determined expression, bare numbers take the smallest common
    /// type that fits.
    fn inferred_expr(&mut self, expr: &Expr) -> Result<TypedExpr, String> {
        let minimum = minimum_type(expr)?;
        self.expr(expr, minimum.as_ref())
    }

    /// Unify same-type operands using the result context, an explicit operand,
    /// or their minimum widths.
    fn pair(
        &mut self,
        left: &Expr,
        right: &Expr,
        expected: Option<&Type>,
    ) -> Result<(TypedExpr, TypedExpr), String> {
        if expected.is_none() && (has_aggregate_literal(left) || has_aggregate_literal(right)) {
            let [left, right] = self
                .unify_values(&[left, right])?
                .try_into()
                .expect("one typed expression per operand");
            return Ok((left, right));
        }
        let (index, anchor) = self.result_anchor(&[left, right], expected)?;
        if index == 0 {
            let r = self.expr_in_group(right, Some(&anchor.ty))?;
            Ok((anchor, r))
        } else {
            let l = self.expr_in_group(left, Some(&anchor.ty))?;
            Ok((l, anchor))
        }
    }

    /// Comparison operands infer their shared type independently of the
    /// boolean result. Keep any scoped bindings available to that inference.
    fn comparison_pair(
        &mut self,
        left: &Expr,
        right: &Expr,
    ) -> Result<(TypedExpr, TypedExpr), String> {
        if self.dslx_widths().is_some() {
            return self.with_prepared_value(left, |checker, lhs| {
                checker.with_prepared_value(right, |checker, rhs| checker.pair(lhs, rhs, None))
            });
        }
        let (l, r) = self.pair(left, right, None)?;
        check_result_width(&[left, right], &l.ty, None, &|| self.zero_calls_function())?;
        Ok((l, r))
    }

    /// Use the index context's standard width only when all its operands are
    /// unsized.
    fn unsigned_index(
        &mut self,
        expr: &Expr,
        standard_width: Option<usize>,
    ) -> Result<TypedExpr, String> {
        let hint = match standard_width {
            Some(width) => {
                let operands = context_operands(expr, &self.inferred_bindings, width)?;
                (operands == ContextOperands::Inline).then_some(Type::Bits {
                    signed: false,
                    width,
                })
            }
            None => minimum_type(expr)?,
        };
        let v = self.expr(expr, hint.as_ref())?;
        if !bits(&v.ty).is_some_and(|(s, _)| !s) {
            return Err(format!("expected unsigned index, got {}", v.ty));
        }
        Ok(v)
    }

    /// Check the independently typed shift distance against the left operand.
    fn shift_distance(&mut self, expr: &Expr, width: usize) -> Result<TypedExpr, String> {
        let right = self.unsigned_index(expr, None)?;
        if matches!(expr, Expr::Literal { .. } | Expr::Bool(_))
            && let K::Literal(value) = &right.kind
            && ir_bits_to_usize(value.as_bits().map_err(|e| e.to_string())?)
                .is_none_or(|shift| shift > width)
        {
            return Err(format!("literal shift amount exceeds input width {width}"));
        }
        Ok(right)
    }

    /// Select operator signatures before lowering erases signedness and nominal
    /// types.
    fn binary(
        &mut self,
        op: BinaryOp,
        left: &Expr,
        right: &Expr,
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        let (ty, l, r) = match op {
            BinaryOp::Shl | BinaryOp::Shr => {
                let l = self.expr_in_group(left, expected)?;
                let (_, width) =
                    bits(&l.ty).ok_or_else(|| format!("'{op}' requires a bit value"))?;
                let r = self.shift_distance(right, width)?;
                (l.ty.clone(), l, r)
            }
            BinaryOp::Concat => {
                let l = self.inferred_expr(left)?;
                let r = self.inferred_expr(right)?;
                let (Some((false, left_width)), Some((false, right_width))) =
                    (bits(&l.ty), bits(&r.ty))
                else {
                    return Err(format!(
                        "bit concatenation requires unsigned operands, got {} and {}",
                        l.ty, r.ty
                    ));
                };
                let width = left_width
                    .checked_add(right_width)
                    .ok_or_else(|| "concatenation width overflow".to_owned())?;
                let ty = Type::Bits {
                    signed: false,
                    width,
                };
                self.structs.validate_type(&ty)?;
                (ty, l, r)
            }
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => {
                let (l, r) = self.comparison_pair(left, right)?;
                if !matches!(op, BinaryOp::Eq | BinaryOp::Ne) && bits(&l.ty).is_none() {
                    return Err(format!("comparison '{op}' requires bit operands"));
                }
                (Type::Bool, l, r)
            }
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => {
                let (l, r) = self.pair(left, right, Some(&Type::Bool))?;
                (Type::Bool, l, r)
            }
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Mod
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Xor => {
                let (l, r) = self.pair(left, right, expected)?;
                if bits(&l.ty).is_none() {
                    return Err(format!("'{op}' requires bit operands, got {}", l.ty));
                }
                (l.ty.clone(), l, r)
            }
        };
        Ok((
            ty,
            K::Binary {
                op,
                left: Box::new(l),
                right: Box::new(r),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{CheckedStmt, MAX_ELABORATION_DEPTH, check_module};
    use crate::ast::{BindingPattern, Expr, Type};
    use crate::parser::parse_module;
    use crate::{ResetBehavior, compile, empty_package};
    use xlsynth_pir::IrValue;

    #[test]
    fn function_unrolls_resolve_calls_without_elaborating_hardware() {
        let fold = r#"
fn step(x: u8) -> u8 { x + u8:1 }
fn fold(x: u8) -> u8 {
    unroll_for! (_, acc) in u32:0..u32:3 {
        let next = step(acc);
        next
    }(x)
}
"#;
        let checked = check_module(&parse_module(fold).unwrap()).unwrap();
        assert_eq!(
            checked
                .functions()
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>(),
            ["step", "fold"],
        );
        let recursive = fold.replace("step(acc)", "fold(acc)");
        assert!(check_module(&parse_module(&recursive).unwrap()).is_err());
        let empty = recursive.replace("u32:3", "u32:0");
        check_module(&parse_module(&empty).unwrap()).unwrap();

        let clocked = fold.replace("fn step(x: u8)", "block step(clk, x: u8)");
        for source in [clocked.replace("u32:3", "u32:0"), clocked] {
            assert!(check_module(&parse_module(&source).unwrap()).is_err());
        }
    }

    #[test]
    fn function_unroll_charges_lexical_bindings_and_projections() {
        // The range consumes all but one unit. The let and its selected
        // tuple field must exhaust the budget during the first iteration.
        let source = r#"
fn fold(seed: (u8, u8)) -> (u8, u8) {
    unroll_for! (_, acc) in u32:0..u32:1048575 {
        let (x, _) = acc;
        acc
    }(seed)
}"#;
        let error = check_module(&parse_module(source).unwrap()).unwrap_err();
        assert!(
            error.contains("iteration 0") && error.contains("function expansion limit"),
            "{error}"
        );
    }

    #[test]
    fn qualified_extern_conflicts_only_with_its_bound_qualifier() {
        let decl = "extern fn math::id(x: u8) -> u8;";
        let body = "block b(clk, x: u8) -> u8 { math::id(x) }";
        let module = parse_module(&format!("use std; {decl} {body}")).unwrap();
        assert!(check_module(&module).unwrap().imported_calls().is_empty());
        for source in [
            format!("use vendor::math; {decl}"),
            format!("{decl} use vendor::math;"),
        ] {
            let module = parse_module(&source).unwrap();
            assert_eq!(
                check_module(&module).unwrap_err(),
                "extern fn 'math::id' conflicts with use binding 'math'"
            );
        }
    }

    #[test]
    fn checker_validates_use_bindings_on_programmatically_constructed_modules() {
        let mut module = parse_module("use std;").unwrap();
        module.uses.push(module.uses[0].clone());
        assert_eq!(
            check_module(&module).unwrap_err(),
            "duplicate use binding 'std'"
        );
        module.uses[1].components.clear();
        assert_eq!(check_module(&module).unwrap_err(), "empty DSLX use path");
    }

    #[test]
    fn checker_validates_programmatically_constructed_let_patterns() {
        let mut module =
            parse_module("fn probe(p: (u8, (u8, u8))) -> u8 { let (_, (.., _)) = p; u8:3 }")
                .unwrap();
        check_module(&module).unwrap();
        for (pattern, error) in [
            (
                BindingPattern::Tuple(vec![
                    BindingPattern::Name("x".into()),
                    BindingPattern::Tuple(vec![
                        BindingPattern::Rest,
                        BindingPattern::Name("x".into()),
                    ]),
                ]),
                "duplicate name 'x' in let pattern",
            ),
            (
                BindingPattern::Tuple(vec![
                    BindingPattern::Rest,
                    BindingPattern::Tuple(vec![BindingPattern::Rest, BindingPattern::Rest]),
                ]),
                "duplicate rest marker in tuple pattern",
            ),
            (BindingPattern::Rest, "rest marker requires a tuple pattern"),
        ] {
            let Expr::Scope { bindings, .. } = &mut module.functions[0].body else {
                panic!("expected the parsed function's let scope");
            };
            bindings[0].pattern = pattern;
            assert_eq!(
                check_module(&module).unwrap_err(),
                format!("fn 'probe': result: {error}")
            );
        }
    }

    #[test]
    fn programmatic_patterns_are_checked_in_empty_and_nonempty_folds() {
        for count in [0, 2] {
            let source = format!(
                r#"fn probe(p: (u8, u8)) -> u8 {{
                    unroll_for! (_, acc) in u32:0..u32:{count} {{
                        let (_, _) = p;
                        acc
                    }}(u8:0)
                }}"#
            );
            let mut module = parse_module(&source).unwrap();
            let Expr::UnrollFor(fold) = &mut module.functions[0].body else {
                panic!("expected the parsed fold");
            };
            let Expr::Scope { bindings, .. } = &mut fold.result else {
                panic!("expected the fold's let scope");
            };
            bindings[0].pattern =
                BindingPattern::Tuple(vec![BindingPattern::Rest, BindingPattern::Rest]);
            let iteration = if count == 0 {
                ""
            } else {
                "unroll_for! iteration 0: "
            };
            assert_eq!(
                check_module(&module).unwrap_err(),
                format!("fn 'probe': result: {iteration}duplicate rest marker in tuple pattern")
            );
        }
    }

    #[test]
    fn grouped_operator_chains_obey_the_expression_tree_depth_limit() {
        let check = |expression: &str, result: &str| {
            let source = format!("block b(clk, x: u8) -> {result} {{ {expression} }}");
            let module = parse_module(&source)?;
            check_module(&module)
        };
        // Each group has only 16 operators; composed, the tree has 801 levels.
        let chain = format!(
            "{}x{}",
            "(".repeat(50),
            format!("{})", " + x".repeat(16)).repeat(50)
        );
        assert!(
            check(&chain, "u8")
                .unwrap_err()
                .contains("expression tree exceeds")
        );
        check(&format!("x{}", " + x".repeat(63)), "u8").unwrap();
        let wide = format!("[{}]", "x + x,".repeat(1000));
        check(&wide, "u8[1000]").unwrap();
    }

    #[test]
    fn bit_concatenation_requires_unsigned_operands_on_both_sides() {
        for (left, right) in [("s4", "u4"), ("u4", "s4"), ("s4", "s4")] {
            let source = format!("block join(clk, x: {left}, y: {right}) -> u8 {{ x ++ y }}");
            assert_eq!(
                check_module(&parse_module(&source).unwrap()).unwrap_err(),
                format!(
                    "block 'join': result: bit concatenation requires unsigned operands, got {left} and {right}"
                )
            );
        }
        let source = "block join(clk, x: s4, y: u4) -> u8 { (x as u4) ++ y }";
        check_module(&parse_module(source).unwrap()).unwrap();
    }

    #[test]
    fn constant_aggregate_resets_preserve_field_order_and_wide_signed_values() {
        let source = r#"
struct Payload { words: u8[2], wide: sN[130] }
block seeded(clk, rst) -> Payload {
    let seed = u8:9 * u8:4;
    reg q: Payload = Payload { wide: sN[130]:-2, words: [seed, seed + 1] };
    q := q;
    q
}"#;
        let module = check_module(&parse_module(source).unwrap()).unwrap();
        let CheckedStmt::Reg { reset, .. } = &module.blocks()[0].statements[1] else {
            panic!("expected a register after seed");
        };
        assert_eq!(
            *reset,
            Some(IrValue::parse_typed("([bits[8]:36, bits[8]:37], bits[130]:-2)").unwrap())
        );
    }

    #[test]
    fn types_are_nominal_and_signed_until_linking() {
        for expression in ["x", "B { data: u8:1 }", "A { data: s8:1 }"] {
            let source = format!(
                r#"
struct A {{ data: u8 }}
struct B {{ data: u8 }}
block nominal(clk, x: B) -> A {{ {expression} }}"#
            );
            let module = parse_module(&source).unwrap();
            assert!(check_module(&module).is_err(), "{expression}");
        }
    }

    #[test]
    fn hierarchy_requires_exactly_one_unconditional_data_driver() {
        for connections in [
            "",
            "d.x := 0; d.x := 1;",
            "if en { d.x := 0; }",
            "d.out := 0;",
        ] {
            let source = format!(
                r#"
block parent(clk, en: bool) -> u8 {{ inst d: child; {connections} d.out }}
block child(clk, x: u8) -> u8 {{ x }}"#
            );
            let module = parse_module(&source).unwrap();
            assert!(check_module(&module).is_err(), "{connections}");
        }
    }

    #[test]
    fn call_instance_names_avoid_parent_output_and_future_register_names() {
        let source = r#"
extern block leaf(clk) -> u8;
block wrap(clk) -> u8 {
    let out = leaf();
    reg out_inst: u8;
    out_inst := out;
    out_inst
}"#;
        let checked = check_module(&parse_module(source).unwrap()).unwrap();
        assert_eq!(checked.extern_blocks()[0].name, "leaf");
        let CheckedStmt::Inst { name, block } = &checked.blocks()[0].statements[0] else {
            panic!("the call must declare an instance");
        };
        assert_eq!((name.as_str(), block.as_str()), ("out_inst_1", "leaf"));
    }

    #[test]
    fn only_explicit_instances_expose_ports_before_the_declaration() {
        let source = r#"
extern block leaf(clk, d: u8) -> u8;
block wrap(clk, x: u8) -> u8 {
    let before = lane.out;
    inst lane: leaf;
    lane.d := x;
    before
}"#;
        check_module(&parse_module(source).unwrap()).unwrap();
        let lexical = source.replace("inst lane: leaf;\n    lane.d := x;", "let lane = leaf(x);");
        assert!(check_module(&parse_module(&lexical).unwrap()).is_err());
    }

    #[test]
    fn extern_block_names_and_physical_ports_are_unique() {
        for source in [
            "extern block a(clk, x: u8, x: u8) -> ();",
            "extern block a(clk, out: u8) -> u8;",
            "extern block a(clk) -> (); block a(clk) -> () { () }",
            "extern fn a() -> (); extern block a(clk) -> ();",
        ] {
            assert!(
                check_module(&parse_module(source).unwrap()).is_err(),
                "{source}"
            );
        }
    }

    #[test]
    fn guards_have_lexical_scope_and_reset_is_a_logical_value() {
        let source = r#"block stage(clk, rst, x: u8) -> u8 {
    reg q: u8;
    if !rst { let local = x; q := local; }
    q
}"#;
        check_module(&parse_module(source).unwrap()).unwrap();
        for bad in [
            source.replace("    q\n", "    local\n"),
            source.replace("!rst", "clk"),
        ] {
            assert!(check_module(&parse_module(&bad).unwrap()).is_err());
        }
    }

    #[test]
    fn type_sizes_include_zero_width_members_and_are_checked_before_materialization() {
        for ty in ["uN[18446744073709551615]", "uN[0][1048576]", "u16[65536]"] {
            let source = format!("block sized(clk, x: {ty}) -> bool {{ false }}");
            assert!(
                parse_module(&source)
                    .and_then(|module| check_module(&module))
                    .is_err(),
                "{ty}"
            );
        }
    }

    #[test]
    fn hierarchy_and_expanded_type_depth_include_already_checked_dependencies() {
        for levels in [MAX_ELABORATION_DEPTH, MAX_ELABORATION_DEPTH + 1, 512] {
            let mut blocks = vec!["block b0(clk) -> () { () }".to_owned()];
            let mut structs = vec!["struct S0 { f: u1 }".to_owned()];
            for i in 1..levels {
                blocks.push(format!(
                    "block b{i}(clk) -> () {{ inst child: b{}; () }}",
                    i - 1
                ));
                structs.push(format!("struct S{i} {{ f: S{} }}", i - 1));
            }
            for declarations in [&mut blocks, &mut structs] {
                for _ in 0..2 {
                    let module = parse_module(&declarations.join("\n")).unwrap();
                    assert_eq!(
                        check_module(&module).is_ok(),
                        levels <= MAX_ELABORATION_DEPTH
                    );
                    declarations.reverse();
                }
            }
        }
    }

    #[test]
    fn type_resolution_bounds_deep_programmatic_ports_before_copying() {
        let mut module = parse_module("block probe(clk, x: u1) -> u1 { u1:0 }").unwrap();
        let input = &mut module.blocks[0].params[0].ty;
        for _ in 0..16_384 {
            *input = Type::Array {
                element: Box::new(std::mem::replace(input, Type::Bool)),
                size: 1,
            };
        }
        let result = check_module(&module);
        // Unwrap the synthetic deep tree iteratively on the test thread too.
        let input = &mut module.blocks[0].params[0].ty;
        while let Type::Array { element, .. } = std::mem::replace(input, Type::Bool) {
            *input = *element;
        }
        assert_eq!(
            result.unwrap_err(),
            "expanded type nesting exceeds 64 levels"
        );
    }

    #[test]
    fn expanded_types_count_arrays_and_tuples_between_named_structs() {
        let mut definitions = vec!["struct S0 { f: u1 }".to_owned()];
        for i in 1..=MAX_ELABORATION_DEPTH / 3 {
            definitions.push(format!("struct S{i} {{ f: (S{}[1],) }}", i - 1));
        }
        // Depth 1 + 3*21 = 64, so an additional aggregate exceeds the limit.
        let name = format!("S{}", MAX_ELABORATION_DEPTH / 3);
        let valid = definitions.join("\n");
        check_module(&parse_module(&valid).unwrap()).unwrap();
        let too_deep = format!("{valid}\nextern fn f(x: {name}[1]) -> bool;");
        assert!(check_module(&parse_module(&too_deep).unwrap()).is_err());
    }

    #[test]
    fn empty_arrays_account_for_their_expanded_element_types() {
        let mut definitions = vec!["struct S0 { f: u1 }".to_owned()];
        for i in 1..=20 {
            definitions.push(format!(
                "struct S{i} {{ a: S{}[0], b: S{}[0] }}",
                i - 1,
                i - 1
            ));
        }
        let module = parse_module(&definitions.join("\n")).unwrap();
        assert!(
            check_module(&module)
                .unwrap_err()
                .contains("maximum concrete type size")
        );
    }

    #[test]
    fn aggregate_equality_accepts_matching_arrays_and_tuples_in_block_ir() {
        let source = r#"
block same(clk, a: (s8, u8[3]), b: (s8, u8[3])) -> bool { a == b }
"#;
        compile(
            source,
            empty_package("comparison"),
            ResetBehavior::default(),
        )
        .unwrap();
    }

    #[test]
    fn constant_array_index_and_zero_width_cast_in_register_reset() {
        let source = r#"
block constants(clk, rst) -> (s8, u8) {
    let samples = [s8:0xff, s8:0b1000_0000];
    reg q: (s8, u8) = (samples[uN[100]:1], sN[0]:0 as u8);
    q := q;
    q
}"#;
        let checked = check_module(&parse_module(source).unwrap()).unwrap();
        let CheckedStmt::Reg { reset, .. } = &checked.blocks()[0].statements[1] else {
            panic!("expected a register after samples");
        };
        assert_eq!(
            *reset,
            Some(IrValue::parse_typed("(bits[8]:-128, bits[8]:0)").unwrap())
        );
        for expression in [
            "s8:128",
            "s8:-129",
            "u8:-1",
            "s8:1[u1:0]",
            "[s8:1,s8:2][uN[100]:0xffffffffffffffffffff]",
        ] {
            let source = format!("block invalid(clk) -> s8 {{ {expression} }}");
            assert!(
                check_module(&parse_module(&source).unwrap()).is_err(),
                "{expression}"
            );
        }
    }
}
