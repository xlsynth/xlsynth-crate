// SPDX-License-Identifier: Apache-2.0

//! BX name resolution, value type checking, and structural driver checks.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::rc::Rc;
use xlsynth_pir::ir::Type as IrType;
use xlsynth_pir::ir_value_utils::{ir_bits_to_usize, zero_ir_value_for_type};
use xlsynth_pir::{IrBits, IrValue};

use crate::ImportOptions;
use crate::ast::{
    BinaryOp, Block, CallbackParam, Expr, ExternBlock, ExternFn, Function, FunctionParametricRef,
    InvariantDecl, InvariantLemma, LetBinding, MatchArm, MatchPattern, Module, Param, SourceSpan,
    Stmt, StructDef, StructLiteralArgs, StructSplat, Type, UnaryOp, WidthSliceType,
};
use crate::dslx_import::ImportResolver;
use crate::infer_literal::{
    CommonTypeHint, ContextOperands, InferredBindings, WidthLookup, binding_widths,
    check_argument_widths, check_match_operand_width, check_result_width, common_type_hint,
    context_operands, has_common_bit_type, has_concrete_element_type, minimum_operand_type,
    minimum_type, projected_type_compatible, unify_inferred_members,
};
use crate::parser::{character_or_bool_digits, explicit_width_hint};
use crate::warnings::WarningCollector;
mod builtin_arrays;
mod builtin_calls;
mod builtin_types;
pub(crate) use builtin_calls::{BUILTIN_PREFIX, is_call_builtin};
mod callables;
mod counted_for;
mod enums;
mod function_specializations;
mod generic_blocks;
mod generic_inputs;
mod imported_parametrics;
mod map;
mod match_patterns;
mod module_constants;
mod scope;
mod source_patterns;
mod struct_parametrics;
use struct_parametrics::StructApplication;
mod test_program;
mod timing;
mod tuple_match;
mod unroll;
mod value_parametrics;

pub use crate::builtin::BuiltinOp;
pub use crate::typed::TypedMatchPattern as ScalarPattern;
use crate::typed::{
    BoundParametricValue, ConstantBinding, ConstantFailure, ConstantFunctions, ConstantUse,
    TypedExprKind as K, bits, repr_bits,
};
pub use crate::typed::{
    CheckedBinding, CheckedBlock, CheckedConstant, CheckedEnum, CheckedFunction, CheckedModule,
    CheckedStmt, TypedExpr, TypedExprKind, TypedFor, TypedIterable, TypedMap, TypedMatchArm,
    TypedMatchPattern,
};
use callables::{BlockCallbackBinding, BlockCallbackFormal, BoundCallback};
use counted_for::CarryContext;
use function_specializations::{CallbackTarget, FunctionSpecialization, FunctionSpecializations};
use generic_blocks::{Bindings, TypeArgument};
use module_constants::ModuleDimensions;

/// Render the common DSLX expressions used in a symbolic type annotation.
struct Dimension<'e>(&'e Expr);

impl fmt::Display for Dimension<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Expr::Name(name) => f.write_str(name),
            Expr::Bool(value) => write!(f, "{value}"),
            Expr::Literal { ty: Some(ty), text } => write!(f, "{ty}:{text}"),
            Expr::Literal { ty: None, text } => f.write_str(text),
            Expr::Unary { op, value } => write!(f, "{op}{}", Dimension(value)),
            Expr::Binary { op, left, right } => {
                write!(f, "({} {op} {})", Dimension(left), Dimension(right))
            }
            Expr::Call { name, args } => {
                write!(f, "{name}(")?;
                for (index, arg) in args.iter().enumerate() {
                    if index != 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", Dimension(arg))?;
                }
                f.write_str(")")
            }
            expr => write!(f, "{expr:?}"),
        }
    }
}

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
            Type::Named(name) | Type::Enum { name, .. } => write!(f, "{name}"),
            Type::Applied { name, args } => {
                write!(f, "{name}<")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ">")
            }
            Type::ParametricStruct { name, args } => {
                write!(f, "{name}<")?;
                for (i, arg) in args.iter().enumerate() {
                    if i != 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", Dimension(arg))?;
                }
                f.write_str(">")
            }
            Type::TypeApplied { name, args } => {
                write!(f, "{name}<")?;
                for (index, arg) in args.iter().enumerate() {
                    if index != 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                f.write_str(">")
            }
            Type::Array { element, size } => write!(f, "{element}[{size}]"),
            Type::ParametricBits { signed, width } => match &**signed {
                Expr::Bool(false) => write!(f, "uN[{}]", Dimension(width)),
                Expr::Bool(true) => write!(f, "sN[{}]", Dimension(width)),
                signed => write!(f, "xN[{}][{}]", Dimension(signed), Dimension(width)),
            },
            Type::ParametricArray { element, size } => write!(f, "{element}[{}]", Dimension(size)),
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
    if let Some(seed) = expr.unannotated_loop_seed() {
        return has_aggregate_literal(seed);
    }
    match expr {
        Expr::Scope { result, .. } => has_aggregate_literal(result),
        Expr::Array(_)
        | Expr::Tuple(_)
        | Expr::ArrayLiteral {
            annotation: None, ..
        } => true,
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
    patterns: Vec<Vec<TypedMatchPattern>>,
}

impl MatchSelector {
    /// Select the first matching arm for a compile-time subject.
    fn selected_arm(&self, value: &IrValue) -> Result<usize, String> {
        for (index, patterns) in self.patterns.iter().enumerate() {
            for pattern in patterns {
                if pattern.matches_value(value, &self.subject.ty)? {
                    return Ok(index);
                }
            }
        }
        Err("constant value did not select an arm of an exhaustive match".into())
    }

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

/// Collect the subject and literal constraints in source order.
fn match_operands<'e>(
    subject: &'e Expr,
    arms: &'e [MatchArm],
    is_named_pattern: impl Fn(&str) -> Result<bool, String>,
) -> Result<Vec<&'e Expr>, String> {
    if arms.is_empty() || arms.iter().any(|arm| arm.patterns.is_empty()) {
        return Err("match requires at least one arm and a pattern for each arm".into());
    }
    source_patterns::validate_match_pattern_size(arms)?;
    let mut operands = vec![subject];
    for (_, pattern) in arms
        .iter()
        .flat_map(|arm| &arm.patterns)
        .flat_map(MatchPattern::nodes)
    {
        if matches!(pattern, MatchPattern::Tuple(_)) {
            continue;
        }
        for literal in pattern.literals() {
            if !matches!(literal, Expr::Bool(_) | Expr::Literal { .. })
                && !matches!(pattern, MatchPattern::Literal(Expr::Name(name)) if is_named_pattern(name)?)
            {
                return Err(
                    "scalar match patterns require a literal, value parametric, module constant, or enum member"
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
        (Type::Enum { .. }, Type::Enum { .. }) => a == b,
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

/// The concrete interface shared by instances with identical parametric
/// bindings.
struct BlockSpecialization<'m> {
    block: &'m Block,
    name: String,
    interface: BlockInterface,
    bindings: Bindings,
    parametric_values: Rc<[BoundParametricValue]>,
    callbacks: Vec<Rc<BoundCallback<'m>>>,
    path: Vec<&'m str>,
}

/// A concrete block can have a private verification-only interface. A
/// specialization also supplies type, value, and callback bindings.
enum BlockDefinition<'a, 'm> {
    Concrete(&'m Block, &'a BlockInterface),
    Invariant(&'m InvariantDecl, &'a BlockInterface),
    Specialized(&'a BlockSpecialization<'m>),
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
struct ConstantContext<'a> {
    enums: &'a BTreeMap<String, CheckedEnum>,
    names: &'a mut BTreeMap<String, ConstantBinding>,
    constexpr_names: Option<&'a BTreeSet<String>>,
    statements: &'a [CheckedStmt],
    budgets: &'a mut ConstantBudgets,
    /// Module initializers and specialized functions charge intermediate
    /// values outside loops as well as in their bodies.
    charge_values: bool,
    functions: Option<&'a mut ConstantFunctions>,
}

impl ConstantContext<'_> {
    /// Keep the defining statement when a folded let read only a selected
    /// path and could later supply a DSLX value parametric.
    fn folded(&self, expr: &TypedExpr, value: IrValue, definition: usize) -> ConstantBinding {
        if self
            .constexpr_names
            .is_some_and(|names| !expr.has_constant_inputs(names))
        {
            ConstantBinding::Selected { value, definition }
        } else {
            ConstantBinding::Known(value)
        }
    }
}

/// The concrete source types, compile-time values, and selected functions of a
/// block.
#[derive(Eq, Hash, PartialEq)]
struct BlockKey<'m> {
    name: &'m str,
    types: Vec<TypeArgument>,
    values: Rc<[BoundParametricValue]>,
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

/// Concrete compile-time arguments selected for a block call.
struct BlockParametricArguments<'m> {
    bindings: Bindings,
    values: Vec<BoundParametricValue>,
    callbacks: Vec<Rc<BoundCallback<'m>>>,
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
        arguments: BlockParametricArguments<'m>,
        path: &[&'m str],
        dimensions: ModuleDimensions<'_, 'm>,
        types: &mut TypeResolver<'_, 'm>,
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
        let BlockParametricArguments {
            bindings,
            values,
            callbacks,
        } = arguments;
        let parametric_values: Rc<[_]> = values.into();
        let key = BlockKey {
            name,
            types: bindings.arguments(&block.type_params)?,
            values: Rc::clone(&parametric_values),
            callbacks: callbacks.iter().map(|bound| bound.target).collect(),
        };
        if let Some(prior) = self.cache.get(&key) {
            return Ok(Rc::clone(prior));
        }
        // An acyclic template graph can still request exponentially many
        // distinct specializations. Repeated instances of one type share this
        // budget entry.
        if self.cache.len() >= MAX_SPECIALIZATIONS {
            return Err(format!(
                "generic block specialization limit of {MAX_SPECIALIZATIONS} exceeded"
            ));
        }
        // Distinct explicit bindings can replicate the same wide ports.
        // Charge the fully bound interface before cloning its concrete types.
        let mut formal_nodes = 0;
        let charge_signature = !callbacks.is_empty() || !parametric_values.is_empty();
        if charge_signature {
            self.expansion_work = self.expansion_work.saturating_add(parametric_values.len());
            for ty in template
                .params
                .iter()
                .map(|p| &p.ty)
                .chain([&template.result])
            {
                let shape =
                    types
                        .structs
                        .formal_shape(ty, MAX_ELABORATION_DEPTH, Some(&bindings))?;
                formal_nodes += shape.ir_nodes;
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
                    ty: types.materialize_type_applications(
                        bindings.concrete_type(&p.ty)?,
                        dimensions,
                        MAX_ELABORATION_DEPTH,
                    )?,
                })
            })
            .collect::<Result<_, String>>()?;
        let result = types.materialize_type_applications(
            bindings.concrete_type(&template.result)?,
            dimensions,
            MAX_ELABORATION_DEPTH,
        )?;
        let interface = BlockInterface::new(
            &block.clock,
            block.reset.as_deref(),
            params,
            result,
            types.structs,
            None,
        )?;
        if charge_signature {
            self.expansion_work = self
                .expansion_work
                .saturating_add(interface.port_type_nodes.saturating_sub(formal_nodes));
            if self.expansion_work > MAX_TYPE_SIZE {
                return Err(format!(
                    "block signature exceeds the generic block specialization expansion limit of {MAX_TYPE_SIZE}"
                ));
            }
        }
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
            parametric_values,
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
pub(crate) fn source_type_nodes(ty: &Type) -> usize {
    match ty {
        Type::Tuple(fields) => 1 + fields.iter().map(source_type_nodes).sum::<usize>(),
        Type::Array { element, .. } | Type::ParametricArray { element, .. } => {
            1 + source_type_nodes(element)
        }
        Type::ParametricBits { .. } => 1,
        Type::Applied { args, .. } => 1 + args.len(),
        Type::ParametricStruct { args, .. } => 1 + args.len(),
        Type::TypeApplied { args, .. } => 1 + args.iter().map(source_type_nodes).sum::<usize>(),
        Type::Named(_) | Type::Enum { .. } | Type::Bool | Type::Bits { .. } => 1,
    }
}

/// Storage copied when cloning a literal: wide bits own their limbs, and an
/// array owns its IR element type. Tuple and array element values are shared.
pub(crate) fn literal_copy_work(value: &IrValue) -> usize {
    fn descendants(ty: &IrType) -> usize {
        match ty {
            IrType::Tuple(fields) => fields.iter().map(|ty| 1 + descendants(ty)).sum(),
            IrType::Array(array) => 1 + descendants(&array.element_type),
            IrType::Token | IrType::Bits(_) => 0,
        }
    }
    match value {
        IrValue::Bits(bits) => bits_copy_work(bits),
        IrValue::Array(array) => descendants(array.element_type()),
        _ => 0,
    }
}

/// A cloned scalar only allocates when its bits exceed one inline limb.
fn bits_copy_work(bits: &IrBits) -> usize {
    let width = bits.get_bit_count();
    if width > 64 { width.div_ceil(64) } else { 0 }
}

/// Source declaration indices for lookup during checking.
#[derive(Default)]
struct Declarations<'a> {
    warnings: RefCell<WarningCollector<'a>>,
    /// Pure tests observe IR contracts in evaluated but discarded tuple fields.
    evaluate_discarded_tuple_fields: bool,
    has_test_discards: Cell<bool>,
    block_names: BTreeSet<&'a str>,
    zero_uses: Option<&'a BTreeMap<String, String>>,
    positions: BTreeMap<&'a str, usize>,
    /// Canonical types evaluated in each alias's declaration scope.
    type_aliases: BTreeMap<&'a str, CheckedTypeAlias>,
    /// Aggregate type nodes copied when substituting transparent aliases.
    alias_copy_work: Cell<usize>,
    constants: BTreeMap<&'a str, CheckedConstant>,
    functions: BTreeMap<&'a str, FunctionInterface>,
    function_templates: BTreeMap<&'a str, &'a Function>,
    /// Value-template source types are borrowed from function_templates until
    /// instantiation. Only their callable lookup table is needed beforehand.
    value_template_callbacks: BTreeMap<&'a str, BTreeMap<String, usize>>,
    has_block_value_parametrics: bool,
    blocks: BTreeMap<&'a str, BlockInterface>,
    templates: BTreeMap<&'a str, BlockTemplate<'a>>,
}

/// A transparent alias and the cost of copying its expanded type at a use site.
struct CheckedTypeAlias {
    ty: Type,
    copy_work: usize,
}

impl Declarations<'_> {
    /// Resolve the spelling shared by a source module's zero intrinsic and
    /// functions.
    fn zero_binding(&self, owner: &str) -> String {
        self.zero_uses
            .and_then(|uses| uses.get(owner))
            .cloned()
            .unwrap_or_else(|| crate::bx_import::zero_binding(owner))
    }

    fn is_block(&self, name: &str) -> bool {
        self.block_names.contains(name)
            || self.blocks.contains_key(name)
            || self.templates.contains_key(name)
    }

    fn declares_function(&self, name: &str) -> bool {
        self.functions.contains_key(name) || self.value_template_callbacks.contains_key(name)
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

    fn check_type_name_visibility(&self, name: &str, at: Option<usize>) -> Result<(), String> {
        if self.visible(name, at) {
            Ok(())
        } else {
            Err(format!("type '{name}' is declared later in the module"))
        }
    }

    fn check_type_visibility(
        &self,
        ty: &Type,
        at: Option<usize>,
        bindings: Option<&Bindings>,
    ) -> Result<(), String> {
        if at.is_none() {
            return Ok(());
        }
        let mut pending = vec![ty];
        while let Some(ty) = pending.pop() {
            match ty {
                Type::Named(name) if bindings.is_some_and(|b| b.declares(name)) => {
                    // The template introduces this name in its own scope.
                }
                Type::Named(name)
                | Type::Applied { name, .. }
                | Type::ParametricStruct { name, .. } => {
                    self.check_type_name_visibility(name, at)?
                }
                Type::TypeApplied { name, args } => {
                    self.check_type_name_visibility(name, at)?;
                    pending.extend(args.iter().rev());
                }
                Type::Tuple(fields) => pending.extend(fields.iter().rev()),
                Type::Array { element, .. } | Type::ParametricArray { element, .. } => {
                    pending.push(element)
                }
                Type::Bool
                | Type::Bits { .. }
                | Type::ParametricBits { .. }
                | Type::Enum { .. } => { /* Resolved types have no declaration to look up. */ }
            }
        }
        Ok(())
    }
}

/// Checked nominal BX and DSLX types, with cached sizes for struct layouts.
#[derive(Default)]
struct StructTable<'m> {
    enums: BTreeMap<String, CheckedEnum>,
    imported_enum_count: usize,
    definitions: Vec<StructDef>,
    indices: BTreeMap<String, (usize, TypeShape)>,
    imported_count: usize,
    /// Ordinary BX declarations, including records whose layout is still
    /// pending.
    declared_records: BTreeSet<&'m str>,
    templates: BTreeMap<&'m str, &'m StructDef>,
    instances: BTreeMap<String, struct_parametrics::StructInstance<'m>>,
    /// Explicit prefixes whose trailing defaults have already been evaluated.
    default_instances: BTreeMap<String, usize>,
    instance_layout_nodes: usize,
    active_templates: BTreeSet<&'m str>,
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
fn resolve_type<'m>(
    ty: &Type,
    is_local: &dyn Fn(&str) -> bool,
    bindings: Option<&Bindings>,
    dimensions: ModuleDimensions<'_, 'm>,
    types: &mut TypeResolver<'_, 'm>,
    remaining: usize,
) -> Result<Type, String> {
    match ty {
        Type::TypeApplied { name, args } => {
            if is_local(name) {
                return Err(format!("'{name}' names a local value here"));
            }
            if let Some(bindings) = bindings {
                bindings.reject_value(name)?;
            }
            let template = types
                .structs
                .template(name)
                .ok_or_else(|| format!("unknown type-parametric struct '{name}'"))?;
            dimensions
                .decls
                .check_type_name_visibility(name, dimensions.position)?;
            struct_parametrics::check_type_arity(template, args.len())?;
            let budget = child_type_budget(remaining)?;
            let args = args
                .iter()
                .map(|arg| resolve_type(arg, is_local, bindings, dimensions, types, budget))
                .collect::<Result<Vec<_>, _>>()?;
            types.resolve_struct_type_application(name, args, bindings, dimensions, remaining)
        }
        Type::ParametricStruct { name, args } => {
            let mut checker = types.struct_scope(dimensions);
            checker.enclosing_type_locals = Some(is_local);
            checker.function_type_bindings = bindings.cloned();
            let args = checker.struct_value_prefix(name, StructLiteralArgs::Expressions(args))?;
            checker.struct_value_type(name, &args, remaining)
        }
        Type::Named(name) | Type::Applied { name, .. } => {
            if is_local(name) {
                return Err(format!("'{name}' names a local value here"));
            }
            if let Some(bindings) = bindings {
                if matches!(ty, Type::Named(_)) && bindings.declares(name) {
                    // Keep type parameters symbolic until the dimensions have
                    // been resolved and the substituted layout has been
                    // bounded.
                    return Ok(ty.clone());
                }
                bindings.reject_value(name)?;
            }
            if let Some(alias) = dimensions.decls.type_aliases.get(name.as_str()) {
                if !matches!(ty, Type::Named(_)) {
                    return Err(format!("type alias '{name}' takes no arguments"));
                }
                // A short chain of tuple aliases can describe exponentially
                // many source nodes. Budget the expansion before cloning it.
                let work = dimensions
                    .decls
                    .alias_copy_work
                    .get()
                    .saturating_add(alias.copy_work);
                if work > MAX_TYPE_SIZE {
                    return Err(format!(
                        "type alias '{name}' exceeds the type expansion limit of {MAX_TYPE_SIZE}"
                    ));
                }
                dimensions.decls.alias_copy_work.set(work);
                return Ok(alias.ty.clone());
            }
            if let Some((subject, member)) = lookup_use(name, types.resolver, is_local)? {
                if let Type::Applied { args, .. } = ty {
                    types.resolver.resolve_applied(&subject, &member, args)
                } else {
                    types.resolver.resolve_named(&subject, &member)
                }
            } else if let Some((qualifier, _)) = name.split_once("::") {
                Err(format!("unknown use binding '{qualifier}' in '{name}'"))
            } else if let Some(template) = types.structs.template(name) {
                dimensions
                    .decls
                    .check_type_name_visibility(name, dimensions.position)?;
                let args = match ty {
                    Type::Applied { args, .. } => args.as_slice(),
                    _ => &[],
                };
                let mut checker = types.struct_scope(dimensions);
                let args = checker.struct_value_prefix(name, StructLiteralArgs::Integers(args))?;
                checker.specialize_struct(
                    template,
                    struct_parametrics::StructArguments::Values {
                        explicit: &args,
                        inferred: None,
                    },
                    remaining,
                )
            } else if matches!(ty, Type::Applied { .. }) {
                Err(format!(
                    "'{name}' requires an imported DSLX struct template"
                ))
            } else {
                Ok(types
                    .structs
                    .enums
                    .get(name)
                    .map_or_else(|| ty.clone(), CheckedEnum::ty))
            }
        }
        Type::Array { element, size } => Ok(Type::array(
            resolve_type(
                element,
                is_local,
                bindings,
                dimensions,
                types,
                child_type_budget(remaining)?,
            )?,
            *size,
        )),
        Type::Tuple(items) => {
            let child_budget = child_type_budget(remaining)?;
            items
                .iter()
                .map(|item| resolve_type(item, is_local, bindings, dimensions, types, child_budget))
                .collect::<Result<_, _>>()
                .map(Type::Tuple)
        }
        Type::Enum {
            name,
            signed,
            width,
        } => {
            let definition = types.structs.enums.get(name).or_else(|| {
                types
                    .resolver
                    .enum_definitions()
                    .iter()
                    .find(|def| def.name == *name)
            });
            if definition.is_some_and(|def| def.signed == *signed && def.width == *width) {
                Ok(ty.clone())
            } else {
                Err(format!(
                    "enum '{name}' has no matching declaration for its representation"
                ))
            }
        }
        Type::ParametricBits { signed, width } => Ok(Type::Bits {
            signed: dimensions.signed(signed, is_local, bindings)?,
            width: dimensions.size(width, is_local, bindings)?,
        }),
        Type::ParametricArray { element, size } => Ok(Type::array(
            resolve_type(
                element,
                is_local,
                bindings,
                dimensions,
                types,
                child_type_budget(remaining)?,
            )?,
            dimensions.size(size, is_local, bindings)?,
        )),
        Type::Bits { .. } | Type::Bool => Ok(ty.clone()),
    }
}

/// Shared module state used to resolve types and evaluate struct arguments.
struct TypeResolver<'a, 'm> {
    resolver: &'a mut ImportResolver,
    structs: &'a mut StructTable<'m>,
    functions: Option<&'a mut FunctionSpecializations<'m>>,
}

impl<'m> TypeResolver<'_, 'm> {
    /// Keep a struct application symbolic until its type arguments are bound,
    /// then substitute the caller's bindings and resolve the struct's fields.
    fn resolve_struct_type_application(
        &mut self,
        name: &str,
        args: Vec<Type>,
        bindings: Option<&Bindings>,
        dimensions: ModuleDimensions<'_, 'm>,
        remaining: usize,
    ) -> Result<Type, String> {
        let application = Type::TypeApplied {
            name: name.to_owned(),
            args,
        };
        let ty = match bindings {
            Some(bindings) if bindings.has_unbound_variable(&application) => {
                return Ok(application);
            }
            Some(bindings) => bindings.concrete_type(&application)?,
            None => application,
        };
        self.materialize_type_applications(ty, dimensions, remaining)
    }

    /// Resolve struct applications whose arguments have already been resolved
    /// and substituted. Their fields are evaluated in the declaration scope.
    fn materialize_type_applications(
        &mut self,
        ty: Type,
        dimensions: ModuleDimensions<'_, 'm>,
        remaining: usize,
    ) -> Result<Type, String> {
        match ty {
            Type::TypeApplied { name, args } => {
                let template = self
                    .structs
                    .template(&name)
                    .ok_or_else(|| format!("unknown type-parametric struct '{name}'"))?;
                let budget = child_type_budget(remaining)?;
                let args = args
                    .into_iter()
                    .map(|arg| self.materialize_type_applications(arg, dimensions, budget))
                    .collect::<Result<Vec<_>, _>>()?;
                self.structs.update_imports(self.resolver)?;
                self.struct_scope(dimensions).specialize_struct(
                    template,
                    struct_parametrics::StructArguments::Types(&args),
                    remaining,
                )
            }
            Type::Tuple(fields) => {
                let budget = child_type_budget(remaining)?;
                fields
                    .into_iter()
                    .map(|field| self.materialize_type_applications(field, dimensions, budget))
                    .collect::<Result<_, _>>()
                    .map(Type::Tuple)
            }
            Type::Array { element, size } => Ok(Type::array(
                self.materialize_type_applications(
                    *element,
                    dimensions,
                    child_type_budget(remaining)?,
                )?,
                size,
            )),
            _ => Ok(ty),
        }
    }

    /// Use the module registry so a type argument can specialize a generic
    /// helper.
    fn struct_scope<'a>(&'a mut self, dimensions: ModuleDimensions<'a, 'm>) -> BodyChecker<'a, 'm> {
        BodyChecker::struct_type_scope(
            dimensions,
            self.resolver,
            self.structs,
            self.functions.as_deref_mut(),
        )
    }
}

/// Resolve struct fields in the enclosing module scope.
fn resolve_fields<'m>(
    params: &[Param],
    dimensions: ModuleDimensions<'_, 'm>,
    types: &mut TypeResolver<'_, 'm>,
) -> Result<Vec<Param>, String> {
    params
        .iter()
        .map(|p| {
            Ok(Param {
                name: p.name.clone(),
                ty: resolve_type(
                    &p.ty,
                    &|_| false,
                    None,
                    dimensions,
                    types,
                    MAX_ELABORATION_DEPTH,
                )?,
            })
        })
        .collect()
}

/// A parameter name enters scope before its type annotation and remains
/// visible in later parameters and the result type.
fn resolve_signature<'a, 'm>(
    params: &[Param],
    result: &Type,
    timing_ports: impl IntoIterator<Item = &'a String>,
    bindings: Option<&Bindings>,
    dimensions: ModuleDimensions<'_, 'm>,
    types: &mut TypeResolver<'_, 'm>,
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
                dimensions,
                types,
                MAX_ELABORATION_DEPTH,
            )?,
        });
    }
    let result = resolve_type(
        result,
        &|name| scope.contains(name),
        bindings,
        dimensions,
        types,
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
pub(crate) fn is_dslx_builtin_value(name: &str) -> bool {
    is_call_builtin(name)
        || crate::import_structs::is_token_builtin(name)
        || matches!(
            name,
            "all_ones!"
                | "zero!"
                | "map"
                | "decode"
                | "bit_count"
                | "element_count"
                | "widening_cast"
                | "checked_cast"
                | "range"
                | "send"
                | "send_if"
                | "recv"
                | "recv_if"
                | "recv_non_blocking"
                | "recv_if_non_blocking"
                | "join"
                | "read"
                | "write"
        )
}

/// DSLX builtin types can qualify a member in an overwritten annotation.
pub(crate) fn is_dslx_builtin_type(name: &str) -> bool {
    matches!(
        name,
        "token" | "Source" | "Sink" | "IOResult" | "ChannelConfig"
    )
}

/// Collect unresolved local names that can refer to type parameters.
fn type_parameter_references<'a>(ty: &'a Type, result: &mut Vec<&'a str>) {
    type_dependencies(
        ty,
        |ty, name| {
            if matches!(ty, Type::Named(_)) {
                result.push(name);
            }
        },
        |_| {},
    );
}

/// Walk nominal dependencies and the expressions supplying dimensions without
/// recursively copying potentially nested source types.
fn type_dependencies<'a>(
    ty: &'a Type,
    mut nominal: impl FnMut(&'a Type, &'a str),
    mut dimension: impl FnMut(&'a Expr),
) {
    let mut pending = vec![ty];
    while let Some(ty) = pending.pop() {
        match ty {
            Type::Named(name)
            | Type::Enum { name, .. }
            | Type::Applied { name, .. }
            | Type::ParametricStruct { name, .. } => {
                nominal(ty, name);
                if let Type::ParametricStruct { args, .. } = ty {
                    for arg in args {
                        dimension(arg);
                    }
                }
            }
            Type::TypeApplied { name, args } => {
                nominal(ty, name);
                pending.extend(args.iter().rev());
            }
            Type::Array { element, .. } => pending.push(element),
            Type::ParametricArray { element, size } => {
                dimension(size);
                pending.push(element);
            }
            Type::ParametricBits { signed, width } => {
                dimension(signed);
                dimension(width);
            }
            Type::Tuple(fields) => pending.extend(fields.iter().rev()),
            Type::Bool | Type::Bits { .. } => { /* Primitives have no local dependencies. */ }
        }
    }
}

/// Bound concrete value materialization, including expanded zero-width members.
/// A bit type costs its width (at least one), and an aggregate costs one plus
/// the sum of its members' costs. This accounts for aggregate storage even
/// when its packed width is zero. An empty array still stores its element type.
pub(crate) const MAX_TYPE_SIZE: usize = 1 << 20;

/// Bound the number of distinct instances in each BX or native function
/// registry.
pub(crate) const MAX_SPECIALIZATIONS: usize = 4096;

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
    /// Nodes in the source type copied during substitution.
    source_nodes: usize,
    /// Bytes in nominal names retained when this type is copied.
    name_bytes: usize,
}

impl<'a> Declarations<'a> {
    /// Resolve and validate each function signature before checking call sites.
    fn add_function(
        &mut self,
        name: &'a str,
        params: &[Param],
        result: &Type,
        local: Option<&'a Function>,
        types: &mut TypeResolver<'_, 'a>,
    ) -> Result<(), String> {
        if let Some(f) = local
            && f.has_inferred_parametrics()
        {
            return self.add_value_template(f, types.structs);
        }
        let mut names = BTreeSet::new();
        let mut callback_params = Vec::new();
        if let Some(function) = local {
            let dimensions = self.module_dimensions(name);
            for callback in &function.callback_params {
                declare(&mut names, &callback.name)?;
                let mut resolve = |ty: &Type| {
                    self.check_type_visibility(ty, dimensions.position, None)?;
                    resolve_type(
                        ty,
                        &|name| names.contains(name),
                        None,
                        dimensions,
                        types,
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
        let (params, result) = resolve_signature(
            params,
            result,
            &names,
            None,
            self.module_dimensions(name),
            types,
        )?;
        let TypeResolver {
            resolver, structs, ..
        } = types;
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
        let interface = BlockInterface::new(clock, reset, params, result, structs, None)?;
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
        private_outputs: Option<&BTreeSet<String>>,
    ) -> Result<Self, String> {
        let layout = Self::check_ports(
            clock,
            reset,
            &params,
            &result,
            structs,
            None,
            private_outputs,
        )?;
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
        private_outputs: Option<&BTreeSet<String>>,
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
        let mut outputs = if let Type::TypeApplied { name, .. } = result {
            structs
                .template(name)
                .ok_or_else(|| unknown_declaration("struct", name))?
                .fields
                .clone()
        } else {
            output_ports(result, &structs.definitions)?
        };
        for output in &mut outputs {
            if let Some(reserved) = private_outputs {
                while reserved.contains(&output.name) {
                    output.name.push('_');
                }
            }
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

impl StructTable<'_> {
    /// Validate a definition after its dependencies and index it for field
    /// selection.
    fn index(&mut self, index: usize) -> Result<TypeShape, String> {
        let def = &self.definitions[index];
        let shape = self.definition_shape(def)?;
        self.indices.insert(def.name.clone(), (index, shape));
        Ok(shape)
    }

    /// Check the complete field layout before registering the nominal type.
    fn definition_shape(&self, def: &StructDef) -> Result<TypeShape, String> {
        TypeShape::aggregate(
            def.fields.iter().map(|p| &p.ty),
            MAX_ELABORATION_DEPTH,
            &|name| self.named_shape(name),
        )
        .map_err(|e| format!("struct '{}': {e}", def.name))
    }

    /// Import newly reflected definitions in dependency order. A call can
    /// introduce a type even when the BX source has no explicit annotation
    /// for its result.
    fn update_imports(&mut self, resolver: &ImportResolver) -> Result<(), String> {
        for def in &resolver.enum_definitions()[self.imported_enum_count..] {
            self.enums.insert(def.name.clone(), def.clone());
            self.imported_enum_count += 1;
        }
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
            .map(|(_, shape)| TypeShape {
                source_nodes: 1,
                name_bytes: name.len(),
                ..*shape
            })
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
        source_nodes: 1,
        name_bytes: 0,
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
            Type::Enum { name, width, .. } => TypeShape {
                size: (*width).max(1),
                name_bytes: name.len(),
                ..Self::LEAF
            },
            Type::Tuple(items) => Self::aggregate(items, remaining, named_shape)?,
            Type::Array { element, size } => {
                let child = Self::for_type(element, child_type_budget(remaining)?, named_shape)?;
                TypeShape {
                    size: child.size.saturating_mul((*size).max(1)).saturating_add(1),
                    depth: child.depth + 1,
                    ir_nodes: child.ir_nodes.saturating_add(1),
                    source_nodes: child.source_nodes.saturating_add(1),
                    name_bytes: child.name_bytes,
                }
            }
            Type::Named(name) => {
                let mut shape = named_shape(name)?;
                shape.name_bytes = shape.name_bytes.max(name.len());
                shape
            }
            // Substitution checks visit source annotations before import
            // resolution. An application is a single nominal leaf there;
            // the checker validates its full layout after resolution.
            Type::Applied { .. } | Type::ParametricStruct { .. } => Self::LEAF,
            Type::TypeApplied { name, args } => {
                // Type arguments can be phantom. Count their syntax for
                // substitution and validate the physical fields at interning.
                let budget = child_type_budget(remaining)?;
                let mut shape = Self {
                    depth: 1,
                    name_bytes: name.len(),
                    ..Self::LEAF
                };
                for arg in args {
                    let arg = Self::for_type(arg, budget, named_shape)?;
                    shape.depth = shape.depth.max(arg.depth + 1);
                    shape.source_nodes = shape.source_nodes.saturating_add(arg.source_nodes);
                    shape.name_bytes = shape.name_bytes.saturating_add(arg.name_bytes);
                }
                shape
            }
            Type::ParametricBits { .. } | Type::ParametricArray { .. } => {
                return Err("unresolved value-parametric dimension".into());
            }
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
        shape.check_source_budget()?;
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
            shape.source_nodes = shape.source_nodes.saturating_add(child.source_nodes);
            shape.depth = shape.depth.max(child.depth + 1);
            shape.name_bytes = shape.name_bytes.saturating_add(child.name_bytes);
            if shape.size > MAX_TYPE_SIZE {
                return Err(format!(
                    "aggregate exceeds maximum concrete type size {MAX_TYPE_SIZE}"
                ));
            }
            shape.check_source_budget()?;
        }
        Ok(shape)
    }

    fn check_source_budget(&self) -> Result<(), String> {
        if self.source_nodes > MAX_TYPE_SIZE {
            return Err(format!(
                "type syntax exceeds the limit of {MAX_TYPE_SIZE} nodes"
            ));
        }
        if self.name_bytes > MAX_TYPE_SIZE {
            return Err(format!(
                "type names exceed the limit of {MAX_TYPE_SIZE} bytes"
            ));
        }
        Ok(())
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
    source_patterns::validate_source_match_patterns(module)?;
    let imports = crate::bx_import::link(module, options)?;
    check_sources(module, imports, options).map_err(crate::bx_import::diagnostic)
}

/// Check the root and all loaded libraries against one DSLX source snapshot.
pub(crate) fn check_sources(
    root: &Module,
    imports: crate::bx_import::SourceImports<'_>,
    options: &ImportOptions<'_>,
) -> Result<CheckedModule, String> {
    check_sources_with_tests(root, imports, options, None)
}

/// Elaborate test-only harnesses before draining the shared specialization
/// queues.
pub(crate) fn check_sources_with_tests(
    root: &Module,
    imports: crate::bx_import::SourceImports<'_>,
    options: &ImportOptions<'_>,
    mut tests: Option<&mut crate::test_runner::TestSuite>,
) -> Result<CheckedModule, String> {
    let linked = imports.libraries.as_ref();
    let module = linked.map_or(root, |l| &l.module);
    let mut names = BTreeSet::new();
    for (i, path) in module.uses.iter().enumerate() {
        let leaf = path.binding().ok_or("empty DSLX use path")?;
        let name = linked.map_or(leaf, |l| l.dslx_bindings[i].as_str());
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
        let source_name = constant.name.as_str();
        let name = linked
            .and_then(|_| crate::bx_import::library_item(source_name))
            .map_or(source_name, |(_, n)| n);
        if !(crate::parser::is_use_identifier(name) || name == "self") {
            return Err(format!("invalid module constant name '{name}'"));
        }
        if is_dslx_builtin_value(name) || is_dslx_builtin_type(name) {
            return Err(format!("constant '{name}' shadows a builtin"));
        }
        declare(&mut names, source_name)?;
    }
    let mut resolver = ImportResolver::with_bindings(
        &module.uses,
        options,
        linked.map(|l| l.dslx_bindings.as_slice()),
        imports.native,
    )?;
    for order in imports.declaration_orders {
        order.verify(&resolver)?;
    }
    let mut structs = StructTable::default();
    for alias in &module.type_aliases {
        let name =
            crate::bx_import::library_item(&alias.name).map_or(alias.name.as_str(), |(_, n)| n);
        if !(crate::parser::is_use_identifier(name) || name == "self") {
            return Err(format!("invalid type alias name '{name}'"));
        }
        declare(&mut names, &alias.name)?;
    }
    for def in &module.enums {
        let name = crate::bx_import::library_item(&def.name).map_or(def.name.as_str(), |(_, n)| n);
        if !(crate::parser::is_use_identifier(name) || name == "self") {
            return Err(format!("invalid enum name '{name}'"));
        }
        declare(&mut names, &def.name)?;
    }
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
        structs.register_template(def)?;
        if !def.is_parametric() {
            structs.declared_records.insert(&def.name);
        }
        structs.definitions.push(StructDef {
            name: def.name.clone(),
            public: def.public,
            // Reserve local indices before reflection appends imported layouts.
            fields: Vec::new(),
            value_parametrics: Vec::new(),
            type_params: Vec::new(),
        });
    }
    let mut declarations = Declarations {
        evaluate_discarded_tuple_fields: tests.is_some()
            && module
                .tests
                .iter()
                .any(|t| t.kind == crate::ast::TestKind::Function),
        block_names: module
            .blocks
            .iter()
            .map(|b| b.name.as_str())
            .chain(module.extern_blocks.iter().map(|b| b.name.as_str()))
            .collect(),
        zero_uses: linked.map(|l| &l.zero_uses),
        positions: module
            .declaration_order
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i + 1))
            .collect(),
        ..Declarations::default()
    };
    // Without an explicit cross-kind order, programmatic modules introduce
    // their constants first and preserve source order within each list.
    if declarations.positions.is_empty() {
        declarations.positions.extend(
            module
                .enums
                .iter()
                .map(|e| e.name.as_str())
                .chain(module.externs.iter().map(|f| f.name.as_str()))
                .chain(module.constants.iter().map(|c| c.name.as_str()))
                .chain(module.structs.iter().map(|s| s.name.as_str()))
                .chain(module.type_aliases.iter().map(|a| a.name.as_str()))
                .chain(module.functions.iter().map(|f| f.name.as_str()))
                .enumerate()
                .map(|(i, name)| (name, i + 1)),
        );
    }
    for f in &module.externs {
        declare(&mut names, &f.name)?;
    }
    for f in &module.functions {
        declare(&mut names, &f.name)?;
        if f.name.contains("::") {
            return Err(format!(
                "BX function '{}' must have an unqualified name",
                f.name
            ));
        }
    }
    for name in module
        .blocks
        .iter()
        .map(|b| &b.name)
        .chain(module.extern_blocks.iter().map(|b| &b.name))
    {
        declare(&mut names, name)?;
    }
    let mut harness_names = Vec::new();
    let mut invariant_harnesses = Vec::new();
    if let Some(suite) = &mut tests {
        suite.reserve_names(&names);
        for test in &module.tests {
            let harness = suite.harness_name(&test.name);
            names.insert(harness.clone());
            harness_names.push(harness);
        }
        for property in &module.invariants {
            let harness = suite.harness_name(&property.block.name);
            names.insert(harness.clone());
            invariant_harnesses.push(harness);
        }
    }
    let mut function_registry = FunctionSpecializations::new(names.clone());
    let function_bodies =
        function_registry.check_sources(module, &mut declarations, &mut resolver, &mut structs)?;
    for (owner, types) in &module.overwritten_literal_types {
        for name in types {
            if let Some((head, _)) = name.split_once("::") {
                // An unlinked module retains its qualified source spelling.
                // Only the qualifier's scope matters after the outer type wins.
                if !names.contains(head) {
                    return Err(format!("unknown name '{head}' in literal annotation"));
                }
                continue;
            }
            let is_type = match resolver.lookup(name)? {
                Some((subject, member)) => resolver.has_public_type(&subject, &member),
                None => {
                    declarations.type_aliases.contains_key(name.as_str())
                        || structs.enums.contains_key(name)
                        || structs.declared_records.contains(name.as_str())
                        || structs.template(name).is_some()
                }
            };
            if !is_type {
                return Err(format!(
                    "in '{owner}': '{name}' does not name a type in literal annotation"
                ));
            }
        }
    }
    let externs = module
        .externs
        .iter()
        .map(|f| {
            let interface = &declarations.functions[f.name.as_str()];
            ExternFn {
                name: f.name.clone(),
                params: interface.params.clone(),
                result: interface.result.clone(),
                symbol: f.symbol.clone(),
            }
        })
        .collect();
    for b in &module.blocks {
        let reserved_ports = std::iter::once(&b.clock)
            .chain(&b.reset)
            .chain(b.callback_params.iter().map(|p| &p.name))
            .chain(b.value_parametrics.iter().map(|p| &p.name));
        let symbolic = if !b.is_parametric() {
            None
        } else {
            generic_blocks::validate_declaration(b)
                .map_err(|e| format!("block '{}': {e}", b.name))?;
            Some(Bindings::new(&b.type_params))
        };
        let dimensions = declarations.module_dimensions(&b.name);
        let mut types = TypeResolver {
            resolver: &mut resolver,
            structs: &mut structs,
            functions: Some(&mut function_registry),
        };
        let (params, result) = resolve_signature(
            &b.params,
            &b.result,
            reserved_ports,
            symbolic.as_ref(),
            dimensions,
            &mut types,
        )?;
        if let Some(bindings) = &symbolic {
            let template = BlockTemplate::new(b, params, result, bindings, dimensions, &mut types)
                .map_err(|e| format!("block '{}': {e}", b.name))?;
            declarations.has_block_value_parametrics |= !b.value_parametrics.is_empty();
            declarations.templates.insert(&b.name, template);
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
        let timing_ports = std::iter::once(&b.clock).chain(&b.reset);
        let (params, result) = resolve_signature(
            &b.params,
            &b.result,
            timing_ports,
            None,
            declarations.module_dimensions(&b.name),
            &mut TypeResolver {
                resolver: &mut resolver,
                structs: &mut structs,
                functions: Some(&mut function_registry),
            },
        )?;
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
    // Static loops can instantiate children. Order the checked blocks using
    // the instances actually elaborated, including children declared later.
    let mut specializations = BlockSpecializations::new(names);
    let mut checked = Vec::new();
    for block in module.blocks.iter().filter(|b| !b.is_parametric()) {
        let body = BodyChecker::check_block(
            &declarations,
            BlockDefinition::Concrete(block, &declarations.blocks[block.name.as_str()]),
            &mut resolver,
            &mut structs,
            &mut specializations,
            &mut function_registry,
        )
        .map_err(|e| format!("block '{}': {e}", block.name))?;
        declarations
            .warnings
            .borrow_mut()
            .record_block(&block.name, body.requires_hardware);
        checked.push(body.block);
    }
    let mut test_functions = Vec::new();
    if let Some(suite) = tests.as_deref_mut() {
        for (test, harness) in module.tests.iter().zip(harness_names) {
            let mut checker = BodyChecker::new(
                &declarations,
                &mut resolver,
                &mut structs,
                match test.kind {
                    crate::ast::TestKind::Function => BodyKind::Function,
                    crate::ast::TestKind::Block => BodyKind::Block { has_reset: true },
                },
            );
            checker.specializations = Some(&mut specializations);
            checker.function_registry = Some(&mut function_registry);
            checker.warning_owner = Some(&test.name);
            checker.zero_binding = declarations.zero_binding(&test.name);
            let position = declarations.positions.get(test.name.as_str()).copied();
            match test.kind {
                crate::ast::TestKind::Function => checker.function_position = position,
                crate::ast::TestKind::Block => checker.block_position = position,
            }
            checker.check_test(test, harness, suite, &mut test_functions, &mut checked)?;
        }
        for (property, harness) in module.invariants.iter().zip(invariant_harnesses) {
            let block = &property.block;
            let (params, result) = resolve_signature(
                &block.params,
                &block.result,
                std::iter::once(&block.clock).chain(&block.reset),
                None,
                declarations.module_dimensions(&block.name),
                &mut TypeResolver {
                    resolver: &mut resolver,
                    structs: &mut structs,
                    functions: Some(&mut function_registry),
                },
            )?;
            if bits(&result) != Some((false, 1)) {
                return Err(format!(
                    "invariant '{}' must return bool, got {result}",
                    block.name
                ));
            }
            let mut fields = vec![result];
            fields.resize(property.lemmas.len() + 1, Type::Bool);
            structs.update_imports(&resolver)?;
            let interface = BlockInterface::new(
                &block.clock,
                block.reset.as_deref(),
                params,
                Type::Tuple(fields),
                &structs,
                Some(&block_binding_names(block)),
            )?;
            let mut body = BodyChecker::check_block(
                &declarations,
                BlockDefinition::Invariant(property, &interface),
                &mut resolver,
                &mut structs,
                &mut specializations,
                &mut function_registry,
            )
            .map_err(|e| format!("invariant '{}': {e}", block.name))?
            .block;
            body.name.clone_from(&harness);
            suite.invariants.push(crate::test_runner::InvariantProgram {
                name: block.name.clone(),
                position: property.position,
                harness,
                params: body.params.clone(),
                lemmas: property.lemmas.clone(),
            });
            checked.push(body);
        }
    }
    let mut specialized_blocks = BTreeSet::new();
    while let Some(spec) = specializations.pending.pop_front() {
        let block = BodyChecker::check_block(
            &declarations,
            BlockDefinition::Specialized(&spec),
            &mut resolver,
            &mut structs,
            &mut specializations,
            &mut function_registry,
        )
        .map_err(|e| format!("block '{}' ({}): {e}", spec.block.name, spec.name))?;
        declarations
            .warnings
            .borrow_mut()
            .record_block(&spec.block.name, block.requires_hardware);
        specialized_blocks.insert(spec.name.clone());
        checked.push(block.block);
    }
    let mut functions = function_registry.finish(function_bodies)?;
    functions.extend(test_functions);
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
    let mut blocks: Vec<_> = order
        .into_iter()
        .filter_map(|i| checked.get_mut(i)?.take())
        .collect();
    // Harnesses are now mixed in with ordinary blocks. Their generated
    // names let timing find the ones whose source result is just a predicate.
    let invariant_harnesses = tests
        .as_deref()
        .into_iter()
        .flat_map(|suite| &suite.invariants)
        .map(|property| property.harness.as_str())
        .collect();
    // A parent may call a generic child defined later in the file. At this
    // point its concrete specialization appears before the parent in `blocks`.
    timing::check_blocks(&mut blocks, &structs, &invariant_harnesses)?;
    let warnings =
        declarations
            .warnings
            .take()
            .finish(root, linked, &declarations.positions, |name| {
                declarations.is_block(name)
            });
    let mut checked = CheckedModule {
        warnings,
        functions_with_test_discards: BTreeSet::new(),
        structs: structs
            .definitions
            .into_iter()
            .filter(|def| !structs.templates.contains_key(def.name.as_str()))
            .collect(),
        enums: structs.enums,
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
        library_extern_functions: BTreeMap::new(),
        functions,
        specialized_functions: function_registry.generated_names(),
        extern_blocks,
        library_extern_blocks: BTreeMap::new(),
        imports: Default::default(),
        blocks,
        specialized_blocks,
    };
    if declarations.has_test_discards.get()
        && let Some(suite) = tests
    {
        let harnesses = suite
            .programs
            .iter()
            .filter(|p| p.kind == crate::ast::TestKind::Function)
            .map(|p| p.harness.clone())
            .collect();
        checked.resolve_pure_test_dependencies(harnesses);
    }
    checked.imports = if linked.is_some() {
        let calls = checked.retain_library_dependencies();
        checked.restore_extern_names();
        resolver.finish_with_calls(Some(&calls))?
    } else {
        resolver.finish()?
    };
    Ok(checked)
}

/// Reserve source bindings before allocating instance or observation port
/// names.
fn block_binding_names(block: &Block) -> BTreeSet<String> {
    let mut names: BTreeSet<_> = std::iter::once(&block.clock)
        .chain(&block.reset)
        .chain(block.params.iter().map(|p| &p.name))
        .cloned()
        .collect();
    let mut pending: Vec<_> = block.statements.iter().collect();
    while let Some(statement) = pending.pop() {
        match statement {
            Stmt::Reg { name, .. }
            | Stmt::Let { name, .. }
            | Stmt::Wire { name, .. }
            | Stmt::Inst { name, .. } => {
                names.insert(name.clone());
            }
            Stmt::LetPattern(binding) => {
                binding.pattern.visit_names(&mut |name| {
                    names.insert(name.to_owned());
                });
            }
            Stmt::If { body, .. } => pending.extend(body),
            _ => { /* Drivers introduce no local bindings. */ }
        }
    }
    names
}

/// Controls which declarations and bindings are available in a BX body.
#[derive(Clone, Copy, Eq, PartialEq)]
enum BodyKind {
    Function,
    Block { has_reset: bool },
}

/// The callback that owns local width inference and any loops opened inside it.
#[derive(Clone, Copy)]
struct MapScope {
    id: usize,
    outer_iterations: usize,
}

/// The checked body and whether its source needs hardware, including child
/// references resolved inside structural iterations that were skipped.
struct BlockBody {
    block: CheckedBlock,
    requires_hardware: bool,
}

/// Lexical values, loop elaboration, and any hardware declared in one body.
struct BodyChecker<'a, 'm> {
    decls: &'a Declarations<'m>,
    structs: &'a mut StructTable<'m>,
    resolver: &'a mut ImportResolver,
    /// Runtime names that shadow module declarations in a type annotation.
    enclosing_type_locals: Option<&'a dyn Fn(&str) -> bool>,
    kind: BodyKind,
    uses_child: Cell<bool>,
    /// The source body owns the expansion budget even while checking a lambda
    /// with function-local rules.
    source_kind: BodyKind,
    specializations: Option<&'a mut BlockSpecializations<'m>>,
    specialization: Option<&'a BlockSpecialization<'m>>,
    function_registry: Option<&'a mut FunctionSpecializations<'m>>,
    function_spec: Option<&'a FunctionSpecialization<'m>>,
    /// Type arguments in the function's declaration scope, including partial
    /// bindings while checking a callback signature.
    function_type_bindings: Option<Bindings>,
    /// The enclosing template's signature for abstract or concrete checking.
    source_template: Option<&'a FunctionInterface>,
    /// Callable names available during lexical checking of a value template.
    template_callbacks: Option<&'a BTreeMap<String, usize>>,
    /// Bound recursive type and expression walks over programmatically supplied
    /// templates.
    template_name_depth: Option<Cell<usize>>,
    /// Use DSLX literal inference inside a local call argument or scalar match.
    dslx_value_context: bool,
    /// A selected function parametric can supply a wider type at a reference
    /// to a let whose scalar initializer contained only unsized literals.
    parametric_argument_context: bool,
    /// Bound materialized operands while inferring a callback block's data
    /// ports.
    checking_callback_inputs: bool,
    function_position: Option<usize>,
    /// The source declaration whose expression is being checked, including
    /// while elaborating its parametric defaults and specializations.
    warning_owner: Option<&'m str>,
    /// The zero intrinsic can be shadowed in the body's own source module.
    zero_binding: String,
    /// Blocks allow forward hardware references, but introduce constants in
    /// source order.
    block_position: Option<usize>,
    local_callees: BTreeSet<String>,
    names: BTreeSet<String>,
    /// Visible function value parametrics, which can appear as named match
    /// patterns.
    value_parametric_names: BTreeSet<String>,
    /// Bare dimension bindings in a struct declaration can have any integer
    /// type.
    in_struct_definition: bool,
    outputs: BTreeSet<String>,
    reserved_hardware_names: BTreeSet<String>,
    values: BTreeMap<String, Type>,
    constants: BTreeMap<String, ConstantBinding>,
    /// Earlier local functions available to a module constant initializer.
    constant_functions: Option<ConstantFunctions>,
    /// Bindings whose entire initializer depends only on compile-time inputs.
    constexpr_names: BTreeSet<String>,
    /// Minimum widths of unannotated bindings and their aggregate members.
    inferred_bindings: InferredBindings,
    regs: BTreeMap<String, Type>,
    writes: BTreeSet<String>,
    /// Deferred bindings in lexical scope and whether each has a driver.
    wires: BTreeMap<String, bool>,
    /// Location of the enclosing block statement during elaboration.
    statement_span: Option<SourceSpan>,
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
    map_scope: Option<MapScope>,
    next_map: usize,
    next_counted_loop: usize,
    iteration_path: Vec<usize>,
    expansion_work: usize,
    reference_copy_work: Cell<usize>,
    constant_budgets: ConstantBudgets,
    unroll_binding_count: usize,
}

impl<'a, 'm> BodyChecker<'a, 'm> {
    /// The source body has scope zero, and callbacks receive increasing IDs.
    fn width_scope(&self) -> usize {
        self.map_scope.map_or(0, |s| s.id)
    }

    /// Make already-checked helpers available for constexpr calls.
    fn enable_constant_functions(&mut self) {
        if self.constant_functions.is_none()
            && let Some(registry) = &self.function_registry
        {
            self.constant_functions = Some(ConstantFunctions::new(Rc::clone(&registry.available)));
        }
    }

    /// Generated bodies share their registry's evaluation budget. Ordinary
    /// source bodies own independent budgets as they do for unrolling.
    fn constant_context(&mut self) -> ConstantContext<'_> {
        let budgets = match (
            &mut self.function_registry,
            self.function_spec,
            &mut self.specializations,
            self.specialization,
        ) {
            (Some(registry), spec, _, _) if spec.is_some() || registry.checking_definition() => {
                &mut registry.constant_budgets
            }
            (_, _, Some(registry), Some(_)) => &mut registry.constant_budgets,
            _ => &mut self.constant_budgets,
        };
        ConstantContext {
            enums: &self.structs.enums,
            names: &mut self.constants,
            constexpr_names: (self.resolver.has_imports()
                || !self.decls.value_template_callbacks.is_empty()
                || self.decls.has_block_value_parametrics)
                .then_some(&self.constexpr_names),
            statements: &self.statements,
            budgets,
            charge_values: self.constant_functions.is_some() || self.function_spec.is_some(),
            functions: self.constant_functions.as_mut(),
        }
    }

    /// Record a constant if speculative folding has work remaining.
    fn fold_constant(&mut self, value: &TypedExpr) -> Option<ConstantBinding> {
        let mut ctx = self.constant_context();
        let deferred = ConstantBinding::Deferred(ctx.statements.len());
        // A projected wide scalar owns its limbs in the speculative cache.
        // Avoid materializing it once the remaining allowance cannot retain it.
        if !ctx.charge_values
            && repr_bits(&value.ty).is_some_and(|(_, width)| {
                width > 64 && width.div_ceil(64) > ctx.budgets.speculative
            })
        {
            return Some(deferred);
        }
        match value.constant(
            ctx.names,
            &mut ctx.budgets.speculative,
            ctx.charge_values,
            ConstantUse::Bx,
            ctx.functions.as_deref_mut(),
            ctx.enums,
        ) {
            Ok(Some(result)) if !ctx.charge_values => {
                // Ordinary bodies retain the result without charging other
                // straight-line intermediate values computed by the evaluator.
                if let Some(remaining) = ctx
                    .budgets
                    .speculative
                    .checked_sub(literal_copy_work(&result))
                {
                    ctx.budgets.speculative = remaining;
                    Some(ctx.folded(value, result, ctx.statements.len()))
                } else {
                    Some(deferred)
                }
            }
            Ok(result) => result.map(|v| ctx.folded(value, v, ctx.statements.len())),
            Err(ConstantFailure::NeedsFunctions) => {
                ctx.constexpr_names.is_some().then_some(deferred)
            }
            Err(_) => Some(deferred),
        }
    }

    /// Evaluate an ordinary BX compile-time value, such as a register reset.
    fn checked_constant(&mut self, value: &TypedExpr) -> Result<Option<IrValue>, String> {
        self.checked_constant_for(value, ConstantUse::Bx)
    }

    /// Replay required lets on an explicit stack. A successful parametric
    /// replay establishes that the folded value satisfies both uses.
    fn checked_constant_for(
        &mut self,
        value: &TypedExpr,
        usage: ConstantUse,
    ) -> Result<Option<IrValue>, String> {
        let mut ctx = self.constant_context();
        let mut charge_values = ctx.charge_values;
        let mut pending: Vec<usize> = Vec::new();
        loop {
            let definition = pending.last().copied();
            let (expr, name) = if let Some(index) = definition {
                let CheckedStmt::Let { name, value } = &ctx.statements[index] else {
                    unreachable!("deferred constant refers to a checked let")
                };
                if ctx
                    .names
                    .get(name)
                    .and_then(|b| b.deferred_definition(usage))
                    != Some(index)
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
                usage,
                ctx.functions.as_deref_mut(),
                ctx.enums,
            ) {
                Ok(result) => {
                    let Some(name) = name else { return Ok(result) };
                    if let Some(result) = result {
                        let binding = if usage == ConstantUse::ValueParametric {
                            ConstantBinding::Known(result)
                        } else {
                            ctx.folded(expr, result, definition.expect("replayed let"))
                        };
                        ctx.names.insert(name.clone(), binding);
                    } else if usage == ConstantUse::ValueParametric {
                        // The ordinary BX constant may still have a value.
                        return Ok(None);
                    } else {
                        ctx.names.remove(name);
                    }
                    pending.pop();
                }
                Err(ConstantFailure::Deferred(index)) => {
                    charge_values = true;
                    let mut earlier = BTreeSet::from([index]);
                    expr.deferred_inputs(ctx.names, &mut earlier, usage);
                    debug_assert!(
                        earlier
                            .last()
                            .is_some_and(|i| *i < definition.unwrap_or(ctx.statements.len()))
                    );
                    pending.extend(earlier.into_iter().rev());
                }
                // A later constexpr actual may replay the same definition
                // with its callees available.
                Err(ConstantFailure::NeedsFunctions) => return Ok(None),
                Err(ConstantFailure::InvalidEnumCast(name)) => {
                    return Err(format!(
                        "constant cast to enum '{name}' requires bits of its declared width encoding a member"
                    ));
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
        structs: &'a mut StructTable<'m>,
        kind: BodyKind,
    ) -> Self {
        Self {
            decls,
            structs,
            resolver,
            enclosing_type_locals: None,
            kind,
            uses_child: Cell::new(false),
            source_kind: kind,
            specializations: None,
            specialization: None,
            function_registry: None,
            function_spec: None,
            function_type_bindings: None,
            source_template: None,
            template_callbacks: None,
            template_name_depth: None,
            dslx_value_context: false,
            parametric_argument_context: false,
            checking_callback_inputs: false,
            function_position: None,
            warning_owner: None,
            zero_binding: "zero!".into(),
            block_position: None,
            local_callees: BTreeSet::new(),
            names: BTreeSet::new(),
            value_parametric_names: BTreeSet::new(),
            in_struct_definition: false,
            outputs: BTreeSet::new(),
            reserved_hardware_names: BTreeSet::new(),
            values: BTreeMap::new(),
            constants: BTreeMap::new(),
            constant_functions: None,
            constexpr_names: BTreeSet::new(),
            inferred_bindings: BTreeMap::new(),
            regs: BTreeMap::new(),
            writes: BTreeSet::new(),
            wires: BTreeMap::new(),
            statement_span: None,
            instances: BTreeMap::new(),
            connections: BTreeSet::new(),
            statements: Vec::new(),
            loop_values: BTreeMap::new(),
            function_values: BTreeMap::new(),
            scope_alias_names: Vec::new(),
            prepared_counted_iterables: BTreeMap::new(),
            in_counted_body: false,
            map_scope: None,
            next_map: 1,
            next_counted_loop: 0,
            iteration_path: Vec::new(),
            expansion_work: 0,
            reference_copy_work: Cell::new(0),
            constant_budgets: ConstantBudgets::default(),
            unroll_binding_count: 0,
        }
    }

    /// Check ordered let bindings and a return expression in a pure scope.
    fn check_function(&mut self, function: &'m Function) -> Result<CheckedFunction, String> {
        self.warning_owner = Some(&function.name);
        self.zero_binding = self.decls.zero_binding(&function.name);
        self.function_position = self.decls.positions.get(function.name.as_str()).copied();
        let bound = self
            .function_spec
            .and_then(|spec| spec.value_signature.as_deref());
        let interface = match bound {
            Some(bound) => &bound.interface,
            None => &self.decls.functions[function.name.as_str()],
        };
        if let Some(bound) = bound {
            if !function.type_params.is_empty() {
                self.function_type_bindings = Some(bound.type_bindings.clone());
            }
            // Width slices and counted ranges may call earlier pure functions
            // with this specialization's compile-time values.
            self.enable_constant_functions();
            for (binding, value) in function.value_parametrics.iter().zip(&bound.values) {
                self.bind_parametric_value(&binding.name, value);
            }
        }
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
        definition: BlockDefinition<'a, 'm>,
        resolver: &'a mut ImportResolver,
        structs: &'a mut StructTable<'m>,
        specializations: &'a mut BlockSpecializations<'m>,
        function_registry: &'a mut FunctionSpecializations<'m>,
    ) -> Result<BlockBody, String> {
        let (block, interface, specialization, invariant) = match definition {
            BlockDefinition::Concrete(block, interface) => (block, interface, None, None),
            BlockDefinition::Invariant(property, interface) => {
                (&property.block, interface, None, Some(property))
            }
            BlockDefinition::Specialized(spec) => (spec.block, &spec.interface, Some(spec), None),
        };
        let outputs = interface
            .outputs
            .iter()
            .map(|p| p.name.clone())
            .collect::<BTreeSet<_>>();
        let mut reserved_hardware_names = block_binding_names(block);
        reserved_hardware_names.extend(outputs.iter().cloned());
        let mut c = Self::new(
            decls,
            resolver,
            structs,
            BodyKind::Block {
                has_reset: block.reset.is_some(),
            },
        );
        c.zero_binding = decls.zero_binding(&block.name);
        c.warning_owner = Some(&block.name);
        c.outputs = outputs;
        c.block_position = decls.positions.get(block.name.as_str()).copied();
        c.specializations = Some(specializations);
        c.specialization = specialization;
        c.function_registry = Some(function_registry);
        c.reserved_hardware_names = reserved_hardware_names;
        if let Some(spec) = specialization
            && !spec.parametric_values.is_empty()
        {
            // Structural bounds can call a local pure helper on the bound
            // values.
            c.enable_constant_functions();
            for (formal, value) in block
                .value_parametrics
                .iter()
                .zip(spec.parametric_values.iter())
            {
                c.bind_parametric_value(&formal.name, value);
            }
        }
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
                    if !template.block.value_parametrics.is_empty() {
                        return Err(format!(
                            "block '{child}' requires a let call with explicit value parametrics"
                        ));
                    }
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
                c.check_child_instance(
                    name,
                    child,
                    child_block,
                    /* has_explicit_parametrics= */ false,
                )?;
                c.instances.insert(name.clone(), child_block);
            }
        }
        c.body(block, &block.statements, false)?;
        for (name, driven) in &c.wires {
            if !driven {
                return Err(format!("wire '{name}' has no driver"));
            }
        }
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
        c.statement_span = block.source.as_ref().map(|source| source.output);
        // An invariant returns one Boolean. Check the trailing expression
        // against that type before building the tuple the prover reads.
        let result = if invariant.is_some() {
            &interface.outputs[0].ty
        } else {
            &interface.result
        };
        let main = c
            .scoped_result(&block.output, Some(result), true)
            .map_err(|e| format!("result: {e}"))?;
        let output = if let Some(property) = invariant {
            // Field 0 is the returned predicate. Each #[lemma] gets the next
            // field in declaration order so the prover can sample it by index.
            let mut observations = vec![main];
            for index in 0..property.lemmas.len() {
                let value = Expr::Name(InvariantLemma::observation_name(index));
                observations.push(c.expr(&value, Some(&Type::Bool))?);
            }
            TypedExpr {
                ty: interface.result.clone(),
                kind: K::Tuple(observations),
            }
        } else {
            main
        };
        let requires_hardware = !c.regs.is_empty() || c.uses_child.get();
        let block = CheckedBlock {
            name: specialization.map_or(&block.name, |s| &s.name).clone(),
            source_name: block.name.clone(),
            source: block.source.as_ref().map(|source| source.file.clone()),
            public: specialization.is_none() && block.public,
            clock: block.clock.clone(),
            reset: block.reset.clone(),
            params: interface.params.clone(),
            input_cycles: block.input_cycles.clone(),
            result: interface.result.clone(),
            output_ports: interface.outputs.clone(),
            output_cycles: block.output_cycles.clone(),
            timing_error: None,
            statements: c.statements,
            output,
        };
        Ok(BlockBody {
            block,
            requires_hardware,
        })
    }

    /// Inferred type arguments in the current block or pure function.
    fn type_bindings(&self) -> Option<&Bindings> {
        self.function_type_bindings
            .as_ref()
            .or_else(|| self.specialization.map(|s| &s.bindings))
    }

    /// Resolve a source annotation and register its concrete fields.
    fn source_type(&mut self, source: &Type) -> Result<Type, String> {
        self.decls
            .check_type_visibility(source, self.function_position, self.type_bindings())?;
        let ty = self.resolve_dimensions(source, MAX_ELABORATION_DEPTH)?;
        let ty = match self.type_bindings() {
            Some(bindings) => bindings.concrete_type(&ty)?,
            None => ty,
        };
        self.structs.update_imports(self.resolver)?;
        self.structs.validate_type(&ty)?;
        Ok(ty)
    }

    /// Instantiate a type from a resolved block signature at a call site.
    fn bound_block_type(&mut self, ty: &Type, bindings: &Bindings) -> Result<Type, String> {
        let ty = bindings.concrete_type(ty)?;
        self.with_type_resolver(|types, dimensions, _| {
            types.materialize_type_applications(ty, dimensions, MAX_ELABORATION_DEPTH)
        })
    }

    /// Borrow the module type resolver alongside this body's lexical scope and
    /// bindings.
    fn with_type_resolver(
        &mut self,
        resolve: impl FnOnce(
            &mut TypeResolver<'_, 'm>,
            ModuleDimensions<'_, 'm>,
            Option<&Bindings>,
        ) -> Result<Type, String>,
    ) -> Result<Type, String> {
        let dimensions = self.decls.dimensions_at(
            self.function_position.or(self.block_position),
            &self.zero_binding,
            self.warning_owner,
        );
        let bindings = self
            .function_type_bindings
            .as_ref()
            .or_else(|| self.specialization.map(|s| &s.bindings));
        let mut types = TypeResolver {
            resolver: self.resolver,
            structs: self.structs,
            functions: self.function_registry.as_deref_mut(),
        };
        resolve(&mut types, dimensions, bindings)
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
        has_explicit_parametrics: bool,
    ) -> Result<(), String> {
        if child.requires_reset && !matches!(self.kind, BodyKind::Block { has_reset: true }) {
            return Err(format!(
                "instance '{name}' of '{target}' requires a parent reset"
            ));
        }
        // Every instance materializes its own ports, even when its definition
        // is shared and its outputs are unused.
        if self.replaying_body() || has_explicit_parametrics {
            self.charge_expansion("child block wiring", child.port_type_nodes)?;
        }
        self.uses_child.set(true);
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
        parametrics: &[Expr],
        args: &[Expr],
        expected: Option<&Type>,
    ) -> Result<TypedExpr, String> {
        let mut prepared = None;
        let spec;
        let (symbol, child) = if let Some(template) = self.decls.templates.get(target) {
            let mut values = Vec::with_capacity(template.block.value_parametrics.len());
            let mut callbacks = Vec::with_capacity(template.callback_params.len());
            for (formal, actual) in template.block.parametric_bindings().zip(parametrics) {
                match formal {
                    FunctionParametricRef::Value(formal) => {
                        // Block template validation resolves each value formal
                        // to u32.
                        let ty = value_parametrics::dimension_type();
                        let value = self.parametric_value(actual, &ty).map_err(|e| {
                            format!(
                                "value parametric '{}' of block '{target}': {e}",
                                formal.name
                            )
                        })?;
                        values.push(BoundParametricValue { ty, value });
                    }
                    FunctionParametricRef::Callback(_) => {
                        callbacks.push(generic_blocks::callback_argument_name(actual)?.to_owned());
                    }
                }
            }
            let call = self.generic_inputs(template, args, &callbacks)?;
            let path = self.specialization.map_or(&[][..], |s| s.path.as_slice());
            spec = self
                .specializations
                .as_mut()
                .expect("block elaboration")
                .intern(
                    template,
                    BlockParametricArguments {
                        bindings: call.bindings,
                        values,
                        callbacks: call.callbacks,
                    },
                    path,
                    self.decls.module_dimensions(target),
                    &mut TypeResolver {
                        resolver: self.resolver,
                        structs: self.structs,
                        functions: self.function_registry.as_deref_mut(),
                    },
                )
                .map_err(|e| format!("call to block '{target}': {e}"))?;
            prepared = Some(call.values.into_iter());
            (spec.name.as_str(), &spec.interface)
        } else {
            (target, &self.decls.blocks[target])
        };
        self.check_child_instance(binding, target, child, !parametrics.is_empty())?;
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
                span: self.statement_span,
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
            Type::ParametricBits { .. }
            | Type::ParametricArray { .. }
            | Type::TypeApplied { .. } => Err("unresolved block result dimension".into()),
            Type::Tuple(_)
            | Type::Named(_)
            | Type::Applied { .. }
            | Type::ParametricStruct { .. } => Ok(TypedExpr {
                ty: child.result.clone(),
                kind: K::Tuple(outputs.collect()),
            }),
            Type::Bool | Type::Bits { .. } | Type::Enum { .. } | Type::Array { .. } => Ok(outputs
                .next()
                .expect("scalar and array interfaces have one output")),
        }
    }

    /// Check the explicit parametric count even in a skipped structural loop,
    /// where no concrete data types or hardware instance are needed.
    fn check_block_parametric_count(
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
        let (values, callbacks) = self.decls.templates.get(target).map_or((0, 0), |template| {
            (
                template.block.value_parametrics.len(),
                template.callback_params.len(),
            )
        });
        let expected = values + callbacks;
        if expected != count {
            let kind = if values == 0 {
                "callable"
            } else {
                "explicit parametric"
            };
            return Err(format!(
                "block '{target}' expects {expected} {kind} arguments, got {count}"
            ));
        }
        Ok(())
    }

    /// A type binder shadows a module member or import in callee position.
    fn reject_type_parameter_value(&self, name: &str) -> Result<(), String> {
        self.type_bindings()
            .map_or(Ok(()), |bindings| bindings.reject_value(name))
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
        for (index, stmt) in statements.iter().enumerate() {
            if !conditional {
                self.statement_span = block
                    .source
                    .as_ref()
                    .and_then(|source| source.statements.get(index))
                    .copied();
            }
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
                        span: self.statement_span,
                    }
                }
                Stmt::Wire { name, ty } => {
                    if conditional {
                        return Err(format!("conditional declaration of wire '{name}'"));
                    }
                    let ty = self.source_type(ty)?;
                    if self.function_values.contains_key(name) {
                        return Err(format!("duplicate name '{name}'"));
                    }
                    declare(&mut self.names, name)?;
                    self.values.insert(name.clone(), ty.clone());
                    self.wires.insert(name.clone(), false);
                    CheckedStmt::Wire {
                        name: name.clone(),
                        ty,
                    }
                }
                Stmt::DriveWire { name, value } => {
                    if conditional {
                        return Err(format!("conditional driver for wire '{name}'"));
                    }
                    let driven = self
                        .wires
                        .get_mut(name)
                        .ok_or_else(|| format!("'{name}' is not a declared wire"))?;
                    if std::mem::replace(driven, true) {
                        return Err(format!("multiple drivers for wire '{name}'"));
                    }
                    let ty = self.values[name].clone();
                    let value = self
                        .expr(value, Some(&ty))
                        .map_err(|e| format!("driver of wire '{name}': {e}"))?;
                    CheckedStmt::DriveWire {
                        name: name.clone(),
                        value,
                        span: self.statement_span,
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
                            || self.wires.contains_key(name)
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
                    if self.wires.contains_key(name) {
                        return Err(format!("wire '{name}' must be driven with '='"));
                    }
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
        if let Some(mut widths) = widths {
            widths.bind_in_scope(self.width_scope());
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
        // Guard checking can already occupy much of the caller's stack when
        // the expression starts. Check available space on each descent.
        stacker::maybe_grow(256 * 1024, 2 * 1024 * 1024, || {
            self.check_group_member(expr, expected)
        })
    }

    /// Dispatch an expression once its recursive frame has sufficient stack
    /// space.
    fn check_group_member(
        &mut self,
        expr: &Expr,
        expected: Option<&Type>,
    ) -> Result<TypedExpr, String> {
        // Keep the recursive dispatch frame small for expressions at the depth
        // limit.
        let (ty, kind) = match expr {
            Expr::Scope { bindings, result } => {
                return self.value_scope(bindings, result, expected);
            }
            Expr::UnrollFor(fold) => {
                return self.unroll(fold, CarryContext::Expected(expected), false);
            }
            Expr::For(fold) => {
                return self.with_dslx_value_context(|checker| {
                    checker.counted_for(fold, CarryContext::Expected(expected))
                });
            }
            Expr::Map(map) => self.with_dslx_value_context(|checker| checker.map_lambda(map)),
            Expr::NamedMap {
                target,
                iterable,
                callback,
                parametrics,
            } => self.with_dslx_value_context(|checker| {
                checker.named_range_map(target, iterable, callback, parametrics)
            }),
            Expr::Name(name) => self.named_value(name),
            Expr::Bool(value) => Ok((Type::Bool, K::Literal(IrValue::bool(*value)))),
            Expr::ByteString(bytes) => builtin_types::byte_string(bytes),
            Expr::TypeBuiltin { name, types, args } => self.type_builtin(name, types, args),
            Expr::Zero(ty) => self.zero_value(ty),
            Expr::Literal { ty, text } => self.bit_literal(ty.as_ref(), text, expected),
            Expr::Tuple(items) => self.tuple_literal(items, expected),
            Expr::Array(items) => self.array_literal(items, expected, false),
            Expr::ArrayLiteral {
                annotation,
                members,
                has_ellipsis,
            } => {
                let ty = annotation
                    .as_ref()
                    .map(|ty| self.source_type(ty))
                    .transpose()?;
                let (inferred, kind) =
                    self.array_literal(members, ty.as_ref().or(expected), *has_ellipsis)?;
                Ok((ty.unwrap_or(inferred), kind))
            }
            Expr::Index { value, index } => self.array_index(value, index),
            Expr::WidthSlice {
                value,
                start,
                width,
            } => self.width_slice(value, start, width),
            Expr::Struct { .. }
            | Expr::AppliedStruct { .. }
            | Expr::ParametricStruct { .. }
            | Expr::TypeAppliedStruct { .. } => {
                let literal = expr.struct_literal().expect("struct literal");
                let values;
                let application = match literal.args {
                    StructLiteralArgs::Omitted => StructApplication::Named(literal.name),
                    StructLiteralArgs::Types(types) => {
                        StructApplication::Types(literal.name, types)
                    }
                    _ => {
                        values = self.struct_value_prefix(literal.name, literal.args)?;
                        StructApplication::Values(literal.name, &values)
                    }
                };
                self.struct_literal(application, literal.fields, literal.base, expected)
            }
            Expr::Field { value, name } => self.project_field(value, name),
            Expr::Call { name, args } => self.function_call(name, args),
            Expr::MapBeforeUse {
                name,
                args,
                builtin_args,
            } => {
                if self.is_builtin_map(name, |target| self.has_local_name(target))? {
                    let args = builtin_args
                        .as_deref()
                        .ok_or("invalid operand syntax for the built-in map")?;
                    self.with_dslx_value_context(|checker| checker.named_array_map(args))
                } else {
                    self.function_call(name, args)
                }
            }
            Expr::ParametricCall {
                name,
                parametrics,
                args,
            } => {
                if self.decls.is_block(name) && self.callback_index(name).is_none() {
                    self.check_block_parametric_count(
                        name,
                        parametrics.len(),
                        self.has_local_name(name),
                    )?;
                    return Err(self.block_in_value_error(name));
                }
                self.parametric_call(name, parametrics, args)
            }
            Expr::Match { subject, arms } => self.with_dslx_value_context(|checker| {
                let selector = checker.match_selector(subject, arms)?;
                let results: Vec<_> = arms.iter().map(|arm| &arm.value).collect();
                let values = checker.check_results(&results, expected)?;
                let value = selector.with_values(&mut values.into_iter());
                Ok((value.ty, value.kind))
            }),
            Expr::Cast { value, ty } => self.cast(value, ty),
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
    fn has_local_name(&self, name: &str) -> bool {
        self.names.contains(name)
            || self.enclosing_type_locals.is_some_and(|contains| {
                contains(name.split_once("::").map_or(name, |(head, _)| head))
            })
    }

    fn named_value(&mut self, name: &str) -> Result<(Type, K), String> {
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
        if !self.has_local_name(name)
            && let Some(value) = self.module_constant(name)
        {
            let work = source_type_nodes(&value.ty) - 1 + literal_copy_work(&value.value);
            self.charge_reference_copy(work)?;
            return Ok((value.ty.clone(), K::Literal(value.value.clone())));
        }
        if let Some((ty, value)) = self.enum_member(name)? {
            return Ok((ty, K::Literal(IrValue::Bits(value))));
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
            (Some(registry), spec, _, _) if spec.is_some() || registry.checking_definition() => {
                &registry.reference_copy_work
            }
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
            self.function_call(&self.zero_binding.clone(), &[])
        } else {
            let zero = self.intrinsic_zero(&ty)?;
            Ok((ty, zero))
        }
    }

    /// Construct a zero of a concrete BX type, including enum membership
    /// checks.
    fn intrinsic_zero(&self, ty: &Type) -> Result<K, String> {
        self.structs.check_zero_members(ty)?;
        let layout = crate::layout::Layouts::new(&self.structs.definitions)
            .lower(ty)
            .map_err(|e| e.to_string())?;
        Ok(K::Literal(zero_ir_value_for_type(&layout)))
    }

    /// Check a bit literal at its annotated or inferred width.
    fn bit_literal(
        &mut self,
        annotation: Option<&Type>,
        text: &str,
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        let resolved;
        let annotation = if let Some(
            ty @ (Type::Named(_)
            | Type::Applied { .. }
            | Type::ParametricStruct { .. }
            | Type::TypeApplied { .. }
            | Type::ParametricBits { .. }),
        ) = annotation
        {
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

    /// Validate the explicit prefix and budget the implicit IR operands.
    fn check_array_length(
        &mut self,
        written: usize,
        length: usize,
        fill: bool,
        charged_fill: usize,
    ) -> Result<(), String> {
        if written > length || (!fill && written != length) {
            return Err(format!("expected array length {length}, got {written}"));
        }
        if fill {
            if written == 0 {
                return Err("array ellipsis requires an element to repeat".into());
            }
            self.charge_expansion(
                "array ellipsis",
                (length - written).saturating_sub(charged_fill),
            )?;
        }
        Ok(())
    }

    /// Infer a common element type and preserve the type of an empty array.
    fn array_literal(
        &mut self,
        items: &[Expr],
        expected: Option<&Type>,
        has_ellipsis: bool,
    ) -> Result<(Type, K), String> {
        let (hint, length) = match expected {
            Some(Type::Array { element, size }) => (Some(element.as_ref()), *size),
            Some(other) => return Err(format!("expected {other}, got an array")),
            None if has_ellipsis => {
                return Err("array ellipsis requires a known array length".into());
            }
            None => (None, items.len()),
        };
        self.check_array_length(items.len(), length, has_ellipsis, 0)?;
        Ok(if !items.is_empty() {
            let values = match hint {
                Some(element) => items
                    .iter()
                    .map(|item| self.expr(item, Some(element)))
                    .collect::<Result<Vec<_>, _>>()?,
                None => self.unify_values(&items.iter().collect::<Vec<_>>())?,
            };
            (Type::array(values[0].ty.clone(), length), K::Array(values))
        } else {
            let element =
                hint.ok_or_else(|| "empty array literal requires an array type".to_owned())?;
            let mut layouts = crate::layout::Layouts::new(&self.structs.definitions);
            let element_ir = layouts.lower(element).map_err(|e| e.to_string())?;
            let empty = IrValue::make_array_typed(element_ir, &[]).map_err(|e| e.to_string())?;
            (Type::array(element.clone(), 0), K::Literal(empty))
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

    /// Discard lets emitted while checking a compile-time or deferred value.
    fn discard_temporary_statements(&mut self, first: usize) {
        for stmt in self.statements.drain(first..) {
            if let CheckedStmt::Let { name, .. } = stmt {
                self.constants.remove(&name);
                self.constexpr_names.remove(&name);
            }
        }
    }

    /// Resolve the struct and check its fields in declaration order.
    fn struct_literal(
        &mut self,
        source: StructApplication<'_>,
        fields: &[(String, Expr)],
        base: Option<&StructSplat>,
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        let base = base
            .map(|splat| {
                self.struct_splat_base(source, &splat.value, expected)
                    .map(|value| (splat, value))
            })
            .transpose()?;
        self.check_struct_literal(source, fields, base, expected)
            .map_err(struct_parametrics::StructLiteralError::into_message)
    }

    /// Check a constructor while distinguishing incomplete type inference
    /// from invalid fields or types.
    fn check_struct_literal(
        &mut self,
        source: StructApplication<'_>,
        fields: &[(String, Expr)],
        base: Option<(&StructSplat, TypedExpr)>,
        expected: Option<&Type>,
    ) -> Result<(Type, K), struct_parametrics::StructLiteralError> {
        let mut supplied = BTreeMap::new();
        for (field, value) in fields {
            if supplied.insert(field.as_str(), value).is_some() {
                return Err(format!("duplicate field '{source}.{field}'").into());
            }
        }
        let (ty, mut inputs, base) = if let Some((splat, base)) = base {
            self.check_struct_splat_type(source, &base.ty)?;
            if let Some(owner) = self.warning_owner
                && let Type::Named(name) = &base.ty
                && self.structs.definition(name)?.fields.len() == supplied.len()
            {
                self.decls
                    .warnings
                    .borrow_mut()
                    .record_useless_splat(owner, splat.position);
            }
            if supplied.is_empty() {
                return Ok((base.ty, base.kind));
            }
            let ty = base.ty.clone();
            let work = self
                .structs
                .type_shape(&ty, MAX_ELABORATION_DEPTH)?
                .ir_nodes;
            self.charge_expansion("struct splat", work)?;
            (ty, BTreeMap::new(), Some(self.share_scoped_value(base)))
        } else {
            let (ty, inputs) = self.struct_literal_type(source, &supplied, expected)?;
            (ty, inputs, None)
        };
        let Type::Named(canonical) = &ty else {
            return Err(format!("'{source}' is not a struct").into());
        };
        let members = self.structs.definition(canonical)?.fields.clone();
        let mut values = Vec::new();
        for (index, p) in members.iter().enumerate() {
            if let Some(v) = supplied.remove(p.name.as_str()) {
                let value = if let Some(input) = inputs.remove(&p.name) {
                    self.finish_input(input, &p.ty)
                } else {
                    self.expr(v, Some(&p.ty))
                };
                values.push(value.map_err(|e| format!("field '{source}.{}': {e}", p.name))?);
            } else if let Some(base) = &base {
                let kind = match &base.kind {
                    K::Name(name) => K::BindingTupleIndex {
                        name: name.clone(),
                        index,
                    },
                    K::Literal(value) => {
                        let member = value.get_element(index).map_err(|e| e.to_string())?;
                        self.charge_expansion("struct splat", literal_copy_work(&member))?;
                        K::Literal(member)
                    }
                    _ => unreachable!("the splat base is shared by its field projections"),
                };
                values.push(TypedExpr {
                    ty: p.ty.clone(),
                    kind,
                });
            } else {
                return Err(format!("missing field '{source}.{}'", p.name).into());
            }
        }
        if let Some(field) = supplied.keys().next() {
            return Err(format!("unknown field '{source}.{field}'").into());
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

    /// Check scalar casts and equal-width casts between bits and a flat array.
    fn cast(&mut self, value: &Expr, ty: &Type) -> Result<(Type, K), String> {
        let ty = self.source_type(ty)?;
        let v = self.inferred_expr(value)?;
        match (&v.ty, &ty) {
            (Type::Array { element, size }, scalar) | (scalar, Type::Array { element, size }) => {
                let (Some((_, element_width)), Some((_, scalar_width))) =
                    (bits(element), bits(scalar))
                else {
                    return Err(format!(
                        "cast from {} to {ty} requires a bit type and a one-dimensional array of bits",
                        v.ty
                    ));
                };
                // Both layouts have already passed the concrete type size
                // check.
                let array_width = element_width * size;
                if array_width != scalar_width {
                    return Err(format!(
                        "cast from {} to {ty} requires equal total widths: array has {array_width} bits, scalar has {scalar_width}",
                        v.ty
                    ));
                }
                // Unpacking slices each element. Packing indexes each element
                // with a literal before concatenating the pieces.
                let element_work = if matches!(v.ty, Type::Array { .. }) {
                    if scalar_width == 0 { 0 } else { 2 * size }
                } else {
                    *size
                };
                self.charge_expansion("array cast", element_work + 1)?;
            }
            (Type::Enum { .. }, Type::Enum { .. }) if v.ty != ty => {
                return Err(format!(
                    "cannot cast between distinct enum types {} and {ty}",
                    v.ty
                ));
            }
            (from, to) if repr_bits(from).is_none() || repr_bits(to).is_none() => {
                return Err(format!("cast from {from} to {to} requires bit types"));
            }
            _ => { /* Scalar casts can change width and signedness. */ }
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
                || self.function_type_bindings.is_some()
                || callback_block
                || self.map_scope.is_some())
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
            } else if self.map_scope.is_some() {
                "map lambda"
            } else if callback_block {
                "block body"
            } else {
                "function body"
            };
            self.charge_expansion(body, nodes.saturating_add(literal_work))?;
        }
        // A capture keeps its materialized width across a map boundary. Its
        // literal radix can still determine signedness at that width.
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
                && projected_type_compatible(expr, &ty, expected, dslx_bindings, None)?
            {
                // Only signedness differs. Preserve the complete IR layout.
                return Ok(TypedExpr {
                    ty: expected.clone(),
                    kind,
                });
            }
            if scoped_projection
                && projected_type_compatible(
                    expr,
                    &ty,
                    expected,
                    dslx_bindings,
                    Some(self.width_scope()),
                )?
            {
                return self.widen_projection(TypedExpr { ty, kind }, expected);
            }
            // DSLX retains the minimum type at the binder and can give a
            // reference to that value a wider concrete type.
            let widths = if self.parametric_argument_context {
                self.common_widths()
            } else {
                self.scoped_widths()
            };
            if let Some(locals) = widths
                && locals.can_widen_reference(expr, self.width_scope())?
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
        if matches!(
            expr,
            Expr::Tuple(_) | Expr::Array(_) | Expr::ArrayLiteral { .. }
        ) {
            self.structs.validate_type(&ty)?;
        }
        let value = TypedExpr { ty, kind };
        if !self.iteration_path.is_empty()
            && matches!(
                value.kind,
                K::ArrayUpdate { .. }
                    | K::Builtin { .. }
                    | K::Array(_)
                    | K::Tuple(_)
                    | K::Literal(_)
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

    /// An inferred aggregate can receive a leaf's type or missing struct
    /// arguments from another operand.
    fn contains_inferred_aggregate(&self, expr: &Expr) -> bool {
        match expr {
            Expr::For(fold) if fold.annotation.is_none() => {
                self.contains_inferred_aggregate(&fold.seed)
            }
            Expr::UnrollFor(fold) if fold.annotation.is_none() => {
                self.contains_inferred_aggregate(&fold.seed)
            }
            Expr::Struct { .. } | Expr::AppliedStruct { .. } | Expr::ParametricStruct { .. } => {
                self.contextual_struct_constructor(expr).is_some()
            }
            Expr::Tuple(items) | Expr::Array(items) => items
                .iter()
                .any(|item| self.contains_inferred_aggregate(item)),
            Expr::ArrayLiteral {
                annotation: None, ..
            } => true,
            Expr::Scope { result, .. } => self.contains_inferred_aggregate(result),
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
                Expr::Array(items) => Ok(merged_operands(checker, items.iter())?
                    .map(|element| Type::array(element, items.len()))),
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
                    let ty = match (ty, checker.type_bindings()) {
                        (Type::Named(name), Some(bindings)) => bindings.get(name),
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
        if items.iter().any(|expr| {
            expr.unannotated_loop_seed()
                .is_some_and(has_aggregate_literal)
        }) {
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

    /// Use a complete sibling row for the length and any written element for
    /// type inference, e.g. `[[x, ...], [1, 2]]` with `x: u8`.
    fn check_array_rows(
        &mut self,
        items: &[&Expr],
        rows: Vec<(&[Expr], bool)>,
        check_elements: impl FnOnce(&mut Self, &[&Expr]) -> Result<Option<Vec<TypedExpr>>, String>,
    ) -> Result<Option<Vec<TypedExpr>>, String> {
        let Some(length) = rows
            .iter()
            .find_map(|(row, fill)| (!fill).then_some(row.len()))
        else {
            // A caller may still obtain the complete array type from another
            // argument.
            return Ok(None);
        };
        let fields: Vec<_> = rows.iter().flat_map(|(row, _)| row.iter()).collect();
        let Some(fields) = check_elements(self, &fields)? else {
            return Ok(None);
        };
        let ty = Type::array(fields[0].ty.clone(), length);
        self.structs.validate_type(&ty)?;
        let mut fields = fields.into_iter();
        items
            .iter()
            .zip(rows)
            .map(|(expr, (row, fill))| {
                self.check_array_length(row.len(), length, fill, 0)?;
                self.finish_expr(
                    expr,
                    None,
                    ty.clone(),
                    K::Array(fields.by_ref().take(row.len()).collect()),
                )
            })
            .collect::<Result<_, _>>()
            .map(Some)
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
            return self.check_result_branches(items, Self::unify_values);
        }
        if !first_has_type
            && let Expr::Tuple(first) = items[0]
            && let Some(tuples) = items
                .iter()
                .map(|item| match item {
                    Expr::Tuple(fields) if fields.len() == first.len() => Some(fields.as_slice()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        {
            return self
                .check_tuple_results(&tuples, |checker, _, fields| checker.unify_values(fields));
        }
        if !first_has_type
            && let Some(rows) = items
                .iter()
                .map(|item| item.unannotated_array())
                .collect::<Option<Vec<_>>>()
            && let Some(arrays) = self.check_array_rows(items, rows, |checker, fields| {
                if fields.is_empty() {
                    return Err("empty array literal requires an array type".into());
                }
                checker.unify_values(fields).map(Some)
            })?
        {
            return Ok(arrays);
        }
        let hint = if first_has_type {
            CommonTypeHint::FromExpr(0)
        } else {
            if let Some(values) = self.infer_struct_siblings(items, None)? {
                return Ok(values);
            }
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

    /// Check all branch results together, then rebuild their typed selectors.
    fn check_result_branches(
        &mut self,
        items: &[&Expr],
        check: impl FnOnce(&mut Self, &[&Expr]) -> Result<Vec<TypedExpr>, String>,
    ) -> Result<Vec<TypedExpr>, String> {
        let (selectors, branches) = self.result_branches(items)?;
        let values = check(self, &branches)?;
        Ok(Self::rebuild_branch_results(selectors, values))
    }

    /// Separate branch selectors from their results so all results can share
    /// the same type evidence.
    fn result_branches<'e>(
        &mut self,
        items: &[&'e Expr],
    ) -> Result<(Vec<BranchSelector>, Vec<&'e Expr>), String> {
        let mut branches = Vec::new();
        let mut selectors = Vec::new();
        for item in items {
            match item {
                Expr::If {
                    condition,
                    then_value,
                    else_value,
                } => {
                    selectors.push(BranchSelector::If(self.expr(condition, Some(&Type::Bool))?));
                    branches.extend([then_value.as_ref(), else_value.as_ref()]);
                }
                Expr::Match { subject, arms } => {
                    selectors.push(BranchSelector::Match(self.match_selector(subject, arms)?));
                    branches.extend(arms.iter().map(|arm| &arm.value));
                }
                _ => {
                    selectors.push(BranchSelector::Value);
                    branches.push(*item);
                }
            }
        }
        Ok((selectors, branches))
    }

    /// Attach checked results to the original branch selectors.
    fn rebuild_branch_results(
        selectors: Vec<BranchSelector>,
        values: Vec<TypedExpr>,
    ) -> Vec<TypedExpr> {
        let mut branches = values.into_iter();
        selectors
            .into_iter()
            .map(|selector| match selector {
                BranchSelector::Value => branches.next().expect("one value per item"),
                BranchSelector::Match(selector) => selector.with_values(&mut branches),
                BranchSelector::If(condition) => {
                    let then_value = branches.next().expect("then branch");
                    let else_value = branches.next().expect("else branch");
                    TypedExpr {
                        ty: then_value.ty.clone(),
                        kind: K::If {
                            condition: Box::new(condition),
                            then_value: Box::new(then_value),
                            else_value: Box::new(else_value),
                        },
                    }
                }
            })
            .collect()
    }

    /// Check each column of same-arity tuple constructors as one type group.
    fn check_tuple_results(
        &mut self,
        tuples: &[&[Expr]],
        mut check: impl FnMut(&mut Self, usize, &[&Expr]) -> Result<Vec<TypedExpr>, String>,
    ) -> Result<Vec<TypedExpr>, String> {
        let arity = tuples[0].len();
        let mut values: Vec<_> = tuples.iter().map(|_| Vec::with_capacity(arity)).collect();
        let mut types = Vec::with_capacity(arity);
        for i in 0..arity {
            let fields: Vec<_> = tuples.iter().map(|tuple| &tuple[i]).collect();
            let fields = check(self, i, &fields)?;
            types.push(fields[0].ty.clone());
            for (tuple, field) in values.iter_mut().zip(fields) {
                tuple.push(field);
            }
        }
        let ty = Type::Tuple(types);
        Ok(values
            .into_iter()
            .map(|fields| TypedExpr {
                ty: ty.clone(),
                kind: K::Tuple(fields),
            })
            .collect())
    }

    /// An aggregate carry may obtain different field types from different
    /// siblings. Unify its seed once and use the resulting value for the loop.
    fn unify_loop_seeds(&mut self, items: &[&Expr]) -> Result<Vec<TypedExpr>, String> {
        let (seeds, loops): (Vec<_>, Vec<_>) = items
            .iter()
            .map(|expr| match expr.unannotated_loop_seed() {
                Some(seed) if has_aggregate_literal(seed) => (seed, Some(*expr)),
                _ => (*expr, None),
            })
            .unzip();
        let values = self.unify_values(&seeds)?;
        loops
            .into_iter()
            .zip(values)
            .map(|(expr, value)| match expr {
                Some(Expr::For(fold)) => self.counted_for(fold, CarryContext::Prepared(value)),
                Some(Expr::UnrollFor(fold)) => {
                    self.unroll(fold, CarryContext::Prepared(value), false)
                }
                Some(_) => unreachable!("only loop expressions have accumulator seeds"),
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
            _ if self.has_local_name("zero!") => Err("'zero!' names a local value here".into()),
            _ if self.callback_index("zero!").is_some() => {
                Err("callable parameter 'zero!' requires a direct call".into())
            }
            _ if self.structs.enums.contains_key(&self.zero_binding)
                || self
                    .decls
                    .type_aliases
                    .contains_key(self.zero_binding.as_str())
                || self
                    .resolver
                    .lookup_enum_prefix(&self.zero_binding)?
                    .is_some_and(|(module, name)| {
                        self.resolver.is_enum_or_alias(&module, &name)
                    }) =>
            {
                if self.decls.visible(
                    &self.zero_binding,
                    self.function_position.or(self.block_position),
                ) {
                    Err("type 'zero!' cannot be called as a function".into())
                } else {
                    Ok(false)
                }
            }
            BodyKind::Function => Ok(self
                .decls
                .visible(&self.zero_binding, self.function_position)
                && (self.decls.declares_function(&self.zero_binding)
                    || self.resolver.lookup(&self.zero_binding)?.is_some())),
            // Only a local function can override zero! in a BX block.
            BodyKind::Block { .. } => Ok(self
                .decls
                .value_template_callbacks
                .contains_key(self.zero_binding.as_str())
                || self
                    .decls
                    .functions
                    .get(self.zero_binding.as_str())
                    .is_some_and(|f| f.local)),
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
                let head = name.split_once("::").map_or(name, |(head, _)| head);
                Ok(checker.value_parametric_names.contains(name)
                    || (!checker.names.contains(head)
                        && checker.callback_index(head).is_none()
                        && (checker.module_constant(name).is_some()
                            || checker.is_enum_member_name(name)?)))
            })?;
            if arms
                .iter()
                .flat_map(|arm| &arm.patterns)
                .any(|p| matches!(p, MatchPattern::Tuple(_)))
            {
                return checker.tuple_match_selector(subject, arms);
            }
            let mut values = checker.check_results(&operands, None)?.into_iter();
            let subject = values.next().expect("subject is the first operand");
            check_match_operand_width(&operands, &subject.ty, checker.dslx_widths(), &|| {
                checker.zero_calls_function()
            })?;
            let patterns = match &subject.ty {
                Type::Enum { name, .. } => {
                    let def = checker
                        .structs
                        .enums
                        .get(name)
                        .ok_or_else(|| format!("unknown enum '{name}'"))?;
                    match_patterns::check_enum_patterns(def, &subject.ty, arms, values)?
                }
                Type::Tuple(_) => return checker.finish_tuple_match(subject, arms),
                _ => match_patterns::check_patterns(&subject.ty, arms, values)?,
            };
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
        if expected.is_none()
            && (items.iter().any(|e| has_aggregate_literal(e))
                || items
                    .iter()
                    .any(|e| matches!(e, Expr::If { .. } | Expr::Match { .. }))
                    && items.iter().any(|e| self.contains_inferred_aggregate(e)))
        {
            return self.unify_values(items);
        }
        if expected.is_none()
            && let Some(values) = self.infer_struct_siblings(items, None)?
        {
            return Ok(values);
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
        let name = self.qualified_extern(name).unwrap_or(name);
        let local = crate::bx_import::local_callee_name(name);
        let binding = local.unwrap_or(name);
        let source_name = scope::source_binding_name(binding, &self.scope_alias_names);
        if self.loop_values.contains_key(binding) {
            return Err(format!("'{source_name}' names a loop-local value"));
        }
        if local.is_some() || self.function_values.contains_key(name) {
            return Err(format!("'{source_name}' names a local value here"));
        }
        if let Some(index) = self.callback_index(name) {
            if self.has_local_name(name) {
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
        if self.is_builtin_map(name, |target| self.has_local_name(target))? {
            return self.with_dslx_value_context(|checker| checker.named_array_map(args));
        }
        if let Some(builtin) = name.strip_prefix(BUILTIN_PREFIX)
            && is_call_builtin(builtin)
        {
            return self.builtin_call(builtin, args);
        }
        let visible = self.decls.visible(name, self.function_position);
        if !visible {
            if is_call_builtin(name) && !self.has_local_name(name) {
                return self.builtin_call(name, args);
            }
            return Err(format!("function '{name}' is declared later in the module"));
        }
        let visible_value = self
            .decls
            .visible(name, self.function_position.or(self.block_position));
        let spelling = crate::bx_import::library_item(name).map_or(name, |(_, item)| item);
        if visible
            && let Some((subject, function)) =
                lookup_use(name, self.resolver, |binding| self.has_local_name(binding))?
        {
            if is_call_builtin(spelling) && self.resolver.has_public_type(&subject, &function) {
                if !visible_value {
                    return self.builtin_call(spelling, args);
                }
                let kind = if self.resolver.is_enum_type(&subject, &function) {
                    "enum"
                } else if self.resolver.is_struct_type(&subject, &function) {
                    "struct"
                } else {
                    "type"
                };
                return Err(format!(
                    "{kind} '{spelling}' cannot be called as a function"
                ));
            }
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
        if visible_value && self.structs.enums.contains_key(name) {
            return Err(format!("enum '{name}' cannot be called as a function"));
        }
        if visible_value && self.decls.type_aliases.contains_key(name) {
            return Err(format!(
                "type alias '{spelling}' cannot be called as a function"
            ));
        }
        if visible_value
            && is_call_builtin(spelling)
            && (self.structs.indices.contains_key(name) || self.structs.template(name).is_some())
        {
            return Err(format!(
                "struct '{spelling}' cannot be called as a function"
            ));
        }
        if is_call_builtin(name) && !self.decls.declares_function(name) {
            if self.has_local_name(name) {
                return Err(format!("'{name}' names a local value here"));
            }
            return self.builtin_call(name, args);
        }
        if self.decls.value_template_callbacks.contains_key(name) {
            return self.parametric_call(name, &[], args);
        }
        let (&callee, f) = self
            .decls
            .functions
            .get_key_value(name)
            .ok_or_else(|| format!("unknown function '{name}'"))?;
        if self.has_local_name(name) && (f.local || self.kind == BodyKind::Function) {
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

    /// Check a dimension's operands at the widths used by DSLX constexpr
    /// evaluation.
    fn dimension_expr(&mut self, expr: &Expr, context_width: usize) -> Result<TypedExpr, String> {
        match expr {
            Expr::Match { subject, arms } => self.with_dslx_value_context(|checker| {
                let selector = checker.match_selector(subject, arms)?;
                let subject = checker
                    .checked_constant(&selector.subject)?
                    .ok_or("slice type dimension must be a compile-time constant")?;
                let selected = selector.selected_arm(&subject)?;
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
                let context = Type::unsigned(context_width);
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
                (operands == ContextOperands::Inline).then_some(Type::unsigned(width))
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
                // DSLX infers each operand independently, including array
                // literals. The result type does not size their elements.
                let l = self.inferred_expr(left)?;
                let r = self.inferred_expr(right)?;
                let ty = match (&l.ty, &r.ty) {
                    (
                        Type::Array { element, size },
                        Type::Array {
                            element: other,
                            size: tail,
                        },
                    ) => {
                        if !compatible(element, other) {
                            return Err(format!(
                                "array concatenation requires matching element types, got {element} and {other}"
                            ));
                        }
                        Type::Array {
                            element: element.clone(),
                            size: size
                                .checked_add(*tail)
                                .ok_or("concatenation length overflow")?,
                        }
                    }
                    (Type::Array { .. }, _) | (_, Type::Array { .. }) => {
                        return Err(format!(
                            "array concatenation requires two arrays, got {} and {}",
                            l.ty, r.ty
                        ));
                    }
                    _ => {
                        let (Some((false, left_width)), Some((false, right_width))) =
                            (bits(&l.ty), bits(&r.ty))
                        else {
                            return Err(format!(
                                "bit concatenation requires unsigned operands, got {} and {}",
                                l.ty, r.ty
                            ));
                        };
                        Type::Bits {
                            signed: false,
                            width: left_width
                                .checked_add(right_width)
                                .ok_or("concatenation width overflow")?,
                        }
                    }
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
                    if matches!(l.ty, Type::Enum { .. }) {
                        return Err(format!(
                            "ordered comparison of enum values requires an explicit cast to bits before '{op}'"
                        ));
                    }
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
    use super::{
        BodyChecker, BodyKind, CheckedModule, CheckedStmt, ConstantBinding, Declarations,
        ImportOptions, ImportResolver, MAX_ELABORATION_DEPTH, StructTable, check_module,
    };
    use crate::ast::{BindingPattern, Expr, Type};
    use crate::parser::parse_module;
    use crate::{ResetBehavior, compile, empty_package};
    use xlsynth_pir::IrValue;

    fn check_bx(source: &str) -> Result<CheckedModule, String> {
        check_module(&parse_module(source)?)
    }

    #[test]
    fn sibling_struct_constructors_supply_inferred_type_arguments() -> Result<(), String> {
        check_bx(
            r#"
struct Wrap<T> { value: T }
fn choose(c: bool, x: u8, w: Wrap<u8>) -> (u8, u8, u8, u8, u8, u8, u8, u8) {
    let inferred = [Wrap { value: 1 }, Wrap { value: x }];
    let reversed = [Wrap { value: x }, Wrap { value: 2 }];
    let explicit = [Wrap { value: 3 }, Wrap<u8> { value: x }];
    let reference = [Wrap { value: 4 }, w];
    let nested = [[Wrap { value: 5 }], [Wrap { value: x }]];
    let selected = if c { Wrap { value: 6 } } else { Wrap { value: x } };
    let scoped = if c { Wrap { value: 7 } } else {
        if c { let y = x; Wrap { value: y } } else { Wrap { value: 8 } }
    };
    let matched = match c { true => Wrap { value: 9 }, false => Wrap<u8> { value: x } };
    (inferred[u32:0].value, reversed[u32:1].value, explicit[u32:0].value,
     reference[u32:0].value, nested[u32:0][u32:0].value, selected.value,
     scoped.value, matched.value)
}

struct Output { result: u8 }
fn nested_scopes(c: bool, x: u8) -> Output {
    Output { result: (if c {
        if c { let _a = x; Wrap { value: 10 } } else { let _b = x; Wrap { value: 11 } }
    } else { Wrap { value: x } }).value }
}
"#,
        )?;
        Ok(())
    }

    const TAP_PORTS: &str = "Ports { live: x, saved: q }";

    /// Common two-port pipeline used to check result annotations.
    fn check_tap(annotation: &str, result: &str) -> Result<CheckedModule, String> {
        check_bx(&format!(
            r#"
struct Ports {{ saved: u8, live: u8 }}
block tap(clk, x: u8) -> Ports {annotation} {{
    reg q: u8;
    q := x;
    {result}
}}
"#
        ))
    }

    #[test]
    fn array_casts_share_the_function_expansion_budget() -> Result<(), String> {
        let unpack = "fn unpack(x: uN[400000]) -> bool[400000] { x as bool[400000] }";
        check_module(&parse_module(unpack)?)?;
        let round_trip = "fn f(x: uN[400000]) -> uN[400000] { (x as bool[400000]) as uN[400000] }";
        let error = check_module(&parse_module(round_trip)?).unwrap_err();
        assert!(
            error.contains("array cast exceeds the function expansion limit"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn implicit_array_elements_share_the_function_expansion_budget() -> Result<(), String> {
        let fill = "bool[600000]:[x, ...]";
        check_bx(&format!("fn fill(x: bool) -> bool[600000] {{ {fill} }}"))?;
        let error = check_bx(&format!(
            "fn compare(x: bool) -> bool {{ {fill} == {fill} }}"
        ))
        .unwrap_err();
        assert_eq!(
            error,
            "fn 'compare': result: array ellipsis exceeds the function expansion limit of 1048576"
        );
        Ok(())
    }

    #[test]
    fn inferred_generic_array_inputs_charge_each_fill_once() -> Result<(), String> {
        let declarations = r#"
fn first<T>(a: T[550000]) -> T { a[0] }
fn id<T>(a: T) -> T { a }
"#;
        for body in [
            "first(if c { [x, ...] } else { y })",
            "id(if c { [x, ...] } else { y })[0]",
            "id(if c { ([x, ...],) } else { (y,) }).0[0]",
            "id(if c { for (_, a) in u32:0..u32:1 { a }([x, ...]) } else { y })[0]",
        ] {
            check_bx(&format!(
                "{declarations}\nfn f(x: bool, c: bool, y: bool[550000]) -> bool {{ {body} }}"
            ))?;
        }
        // Two distinct prefixes each still contribute 549999 implicit operands.
        let error = check_bx(&format!(
            "{declarations}\nfn f(x: bool, c: bool, y: bool[550000]) -> bool {{
                first(if c {{ [x, ...] }} else {{ y }}) == first(if c {{ [x, ...] }} else {{ y }})
            }}"
        ))
        .unwrap_err();
        assert!(
            error.contains("array ellipsis exceeds the function expansion limit"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn typed_array_literals_supply_generic_arguments() -> Result<(), String> {
        check_bx(
            r#"
fn first<T, N: u32>(xs: T[N]) -> T { xs[0] }
fn repeat<T, N: u32>(x: T) -> T[N] { T[N]:[x, ...] }
fn sample(x: u8) -> (u8, u8[4]) { (first(u8[3]:[x, ...]), repeat<4>(x)) }
"#,
        )?;
        Ok(())
    }

    #[test]
    fn speculative_fold_preserves_work_when_a_wide_enum_will_not_fit() -> Result<(), String> {
        let source = r#"
enum Mode:uN[65] { Idle=0 }
fn wide()->Mode { for (_, state) in u32:0..u32:1 { state }(Mode::Idle) }
"#;
        let module = check_module(&parse_module(source)?)?;
        let decls = Declarations::default();
        let mut resolver = ImportResolver::new(&[], &ImportOptions::default())?;
        let mut structs = StructTable::default();
        let mut checker = BodyChecker::new(&decls, &mut resolver, &mut structs, BodyKind::Function);
        checker.constant_budgets.speculative = 1;
        assert!(matches!(
            checker.fold_constant(&module.functions()[0].output),
            Some(ConstantBinding::Deferred(0))
        ));
        assert_eq!(checker.constant_budgets.speculative, 1);
        Ok(())
    }

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
    fn declared_output_cycle_must_match_the_inferred_cycle() {
        let check_delay = |cycle| {
            let source = format!("block b(clk, x: u8) -> u8 @{cycle} {{ reg q: u8; q := x; q }}");
            check_bx(&source)
        };
        check_delay(1).unwrap();
        assert_eq!(
            check_delay(0).unwrap_err(),
            "block 'b': expected cycle 0, got cycle 1"
        );
    }

    #[test]
    fn uniform_output_cycle_applies_to_each_field() {
        check_tap("@1", "Ports { saved: q, live: q }").unwrap();
        assert_eq!(
            check_tap("@1", TAP_PORTS).unwrap_err(),
            "block 'tap': output 'live': expected cycle 1, got cycle 0"
        );
    }

    #[test]
    fn uniform_output_cycle_can_check_an_aggregate_input_as_a_whole() {
        let source = r#"
struct Ports { saved: u8, live: u8 }
block pass(clk, p: Ports @1) -> Ports @1 { p }
"#;
        check_bx(source).unwrap();
    }

    #[test]
    fn field_projection_inherits_the_aggregate_cycle() {
        let check_pick = |cycle| {
            let source = format!(
                r#"
struct Ports {{ saved: u8, live: u8 }}
block pick(clk, p: Ports @1) -> u8 @{cycle} {{ p.saved }}
"#
            );
            check_bx(&source)
        };
        check_pick(1).unwrap();
        assert_eq!(
            check_pick(0).unwrap_err(),
            "block 'pick': expected cycle 0, got cycle 1"
        );
    }

    #[test]
    fn field_projection_follows_the_selected_initializer() {
        let check_pick = |field, cycle| {
            let source = format!(
                r#"
struct Ports {{ saved: u8, live: u8 }}
struct Envelope {{ tag: u8, ports: Ports }}
block pick(clk, x: u8) -> u8 @{cycle} {{
    reg q: u8;
    q := x;
    let fields = Ports {{ saved: q, live: x }};
    let envelope = Envelope {{ tag: x, ports: fields }};
    let p = envelope;
    p.ports.{field}
}}
"#
            );
            check_bx(&source)
        };
        check_pick("saved", 1).unwrap();
        check_pick("live", 0).unwrap();
        assert_eq!(
            check_pick("live", 1).unwrap_err(),
            "block 'pick': expected cycle 1, got cycle 0"
        );
    }

    #[test]
    fn field_projection_adds_the_register_cycle() {
        let check_pick = |saved| {
            let source = format!(
                r#"
struct Ports {{ saved: u8, live: u8 }}
block pick(clk, x: u8) -> Ports @(saved: 2, live: 1) {{
    reg q: u8;
    reg p: Ports;
    q := x;
    p := Ports {{ saved: {saved}, live: x }};
    Ports {{ saved: p.saved, live: p.live }}
}}
"#
            );
            check_bx(&source)
        };
        check_pick("q").unwrap();
        check_pick("p.live").unwrap(); // x -> p.live @1 -> p.saved @2
        assert_eq!(
            check_pick("p.saved").unwrap_err(),
            "block 'pick': output 'saved': timing feedback through 'p' is unsupported"
        );
    }

    #[test]
    fn declared_output_cycles_match_fields_by_port_name() {
        check_tap("@(live: 0, saved: 1)", TAP_PORTS).unwrap();
        assert_eq!(
            check_tap("@(live: 0, saved: 0)", TAP_PORTS).unwrap_err(),
            "block 'tap': output 'saved': expected cycle 0, got cycle 1"
        );
    }

    #[test]
    fn forwarding_a_registered_struct_checks_the_same_fields_as_rebuilding_it() {
        for result in ["q", "Payload { data: q.data, tag: q.tag }"] {
            let source = format!(
                r#"
struct Payload {{ data: u8, tag: u8 }}
struct Bus {{ payload: Payload }}
block b(clk, x: u8) -> Bus @1 {{
    reg q: Payload;
    q := Payload {{ data: x, tag: u8:7 }};
    Bus {{ payload: {result} }}
}}
"#
            );
            assert_eq!(
                check_bx(&source).unwrap_err(),
                "block 'b': output 'payload': cannot infer output cycle for this expression"
            );
            check_bx(&source.replace("tag: u8:7", "tag: x")).unwrap();
        }
    }

    #[test]
    fn forwarding_a_registered_tuple_field_uses_its_write_enable() {
        let source = r#"
struct Bus { payload: (u8, u8) }
block b(clk, en: bool @2, x: u8 @2) -> Bus @3 {
    reg q: (u8, (u8, u8));
    if en { q := (x, (x, u8:7)); }
    Bus { payload: q.1 }
}
"#;
        check_bx(source).unwrap();
        assert_eq!(
            check_bx(&source.replace("if en", "if true")).unwrap_err(),
            "block 'b': output 'payload': cannot infer output cycle for this expression"
        );
    }

    #[test]
    fn long_timing_dependencies_are_independent_of_operand_order() {
        let mut aliases = String::from("let v0 = x;");
        let mut pipeline = String::from("reg q0: u8; q0 := x;");
        for i in 1..256 {
            let prev = i - 1;
            aliases.push_str(&format!("let v{i} = v{prev};"));
            pipeline.push_str(&format!("reg q{i}: u8; q{i} := q{prev} + u8:1;"));
        }
        for sum in ["v1 + v255", "v255 + v1"] {
            check_bx(&format!(
                "block b(clk, x: u8) -> u8 @0 {{ {aliases} {sum} }}"
            ))
            .unwrap();
        }
        let source = format!("block b(clk, x: u8) -> u8 @256 {{ {pipeline} q255 }}");
        check_bx(&source).unwrap();
        assert_eq!(
            check_bx(&source.replace("@256", "@255")).unwrap_err(),
            "block 'b': expected cycle 255, got cycle 256"
        );
    }

    #[test]
    fn per_port_timing_follows_aggregate_let_bindings() {
        let result = "let ports = Ports { saved: q, live: x }; let result = ports; result";
        check_tap("@(saved: 1, live: 0)", result).unwrap();
    }

    #[test]
    fn per_port_timing_projects_a_returned_register() {
        let result = r#"
    reg p: Ports;
    p := Ports { saved: q, live: x };
    p
"#;
        check_tap("@(live: 1, saved: 2)", result).unwrap();
        assert_eq!(
            check_tap("@(live: 1, saved: 1)", result).unwrap_err(),
            "block 'tap': output 'saved': expected cycle 1, got cycle 2"
        );
    }

    #[test]
    fn output_cycle_annotations_reject_unknown_ports() {
        assert_eq!(
            check_tap("@(svaed: 1)", TAP_PORTS).unwrap_err(),
            "block 'tap': unknown output port 'svaed' in cycle annotation"
        );
    }

    #[test]
    fn output_cycle_annotations_require_every_port() {
        assert_eq!(
            check_tap("@(saved: 1)", TAP_PORTS).unwrap_err(),
            "block 'tap': missing cycle annotation for output port 'live'"
        );
    }

    #[test]
    fn declared_input_cycles_seed_timing_inference() {
        let source = r#"
block b(clk, x: u8, y: u8 @1) -> u8 @1 {
    reg q: u8;
    q := x;
    q + y
}
"#;
        check_bx(source).unwrap();
    }

    #[test]
    fn let_binding_inherits_the_cycle_of_its_initializer() {
        let source = "block b(clk, x: u8) -> u8 @1 { reg q: u8; let saved = q; q := x; saved }";
        check_bx(source).unwrap();
    }

    #[test]
    fn binary_operands_must_agree_on_cycle() {
        let check_sum = |sum| {
            let source = format!("block b(clk, x: u8) -> u8 @1 {{ reg q: u8; q := x; {sum} }}");
            check_bx(&source)
        };
        check_sum("q + q").unwrap();
        check_sum("q + u8:1").unwrap();
        check_sum("u8:1 + q").unwrap();
        assert_eq!(
            check_sum("q + x").unwrap_err(),
            "block 'b': operator '+' combines cycle 1 with cycle 0"
        );
    }

    #[test]
    fn guarded_write_checks_enable_against_data_at_the_write_edge() {
        let check_capture = |enable_cycle| {
            let source = format!(
                r#"
block capture(clk, en: bool @{enable_cycle}, x: u8 @2, y: u8 @2) -> u8 @3 {{
    reg q: u8;
    if en {{ q := x + y; }}
    q
}}
"#
            );
            check_bx(&source)
        };
        assert_eq!(
            check_capture(1).unwrap_err(),
            "block 'capture': write to 'q' has guard at cycle 1 and data at cycle 2"
        );
        check_capture(2).unwrap();
    }

    #[test]
    fn held_register_keeps_its_stage_at_later_reads() {
        let source = r#"
block add_held(clk, en: bool, x: u8, r: u8 @1) -> u8 @1 {
    reg q: u8;
    if en { q := x; }
    q + r
}
"#;
        check_bx(source).unwrap();
        assert_eq!(
            check_bx(&source.replace("r: u8 @1", "r: u8 @0")).unwrap_err(),
            "block 'add_held': operator '+' combines cycle 1 with cycle 0"
        );
        let registered = source
            .replace("-> u8 @1", "-> u8 @2")
            .replace("    q + r", "    reg m: u8;\n    m := q + r;\n    m");
        check_bx(&registered).unwrap();
    }

    #[test]
    fn nested_guards_and_multiple_writes_resolve_local_bindings_in_scope() {
        let source = r#"
block b(clk, en: bool, x: u8, late_en: bool @1, y: u8 @1)
    -> (u8, u8, u8, u8) @(out_0: 1, out_1: 1, out_2: 2, out_3: 1) {
    reg a: u8;
    reg b0: u8;
    reg c: u8;
    if en {
        let local = x;
        let inner = local == u8:1;
        if inner { a := local; }
        b0 := local;
    }
    if late_en {
        let local = y;
        c := local;
    }
    let local = y;
    (a, b0, c, local)
}
"#;
        check_bx(source).unwrap();
        let bad = source.replace("if inner", "if late_en");
        assert_eq!(
            check_bx(&bad).unwrap_err(),
            "block 'b': output 'out_0': nested write guards combine cycle 0 with cycle 1"
        );
    }

    #[test]
    fn guarded_constant_uses_the_enables_stage_and_checks_offset_overflow() {
        let source = "block b(clk, en: bool @2) -> u8 @3 { reg q: u8; if en { q := u8:7; } q }";
        check_bx(source).unwrap();
        let overflow = source.replace("en: bool @2", &format!("en: bool @{}", u64::MAX));
        assert_eq!(
            check_bx(&overflow).unwrap_err(),
            "block 'b': cycle offset overflow at 'q'"
        );
    }

    #[test]
    fn combinational_if_checks_the_select_and_both_data_inputs() {
        let source =
            "block b(clk, en: bool @1, x: u8 @1, y: u8 @1) -> u8 @1 { if en { x } else { y } }";
        check_bx(source).unwrap();
        assert_eq!(
            check_bx(&source.replace("en: bool @1", "en: bool @0")).unwrap_err(),
            "block 'b': if expression has condition at cycle 0 and data at cycle 1"
        );
        assert_eq!(
            check_bx(&source.replace("y: u8 @1", "y: u8 @0")).unwrap_err(),
            "block 'b': if branches combine cycle 1 with cycle 0"
        );
    }

    #[test]
    fn field_projection_checks_the_register_write_guard() {
        let source = r#"
struct Ports { saved: u8 }
block capture(clk, en: bool @1, x: u8 @2) -> u8 @3 {
    reg q: Ports;
    if en { q := Ports { saved: x }; }
    q.saved
}
"#;
        assert_eq!(
            check_bx(source).unwrap_err(),
            "block 'capture': write to 'q' has guard at cycle 1 and data at cycle 2"
        );
    }

    #[test]
    fn unary_and_cast_expressions_inherit_the_operand_cycle() {
        let source = "block b(clk, x: u8) -> u16 @1 { reg q: u8; q := x; (!q) as u16 }";
        check_bx(source).unwrap();
    }

    #[test]
    fn annotated_blocks_require_an_inferred_output_cycle() {
        check_bx("block b(clk, x: u8[2]) -> u8 { x[u1:0] }").unwrap();
        let timed = "block b(clk, x: u8[2]) -> u8 @0 { x[u1:0] }";
        assert_eq!(
            check_bx(timed).unwrap_err(),
            "block 'b': cannot infer output cycle for this expression"
        );
    }

    #[test]
    fn two_instances_of_a_three_stage_tail_translate_independently() -> Result<(), String> {
        let source = r#"
block tail(clk, data: u8) -> u8 @3 {
    reg a: u8; reg b: u8; reg c: u8;
    a := data; b := a; c := b; c
}
block top(clk, x: u8 @4) -> (u8, u8) @(out_0: 7, out_1: 10) {
    let first = tail(x);
    let second = tail(first);
    (first, second)
}
block outer(clk, x: u8) -> u8 @6 {
    let pair = top(x);
    pair.1
}
"#;
        check_bx(source)?;
        assert_eq!(
            check_bx(&source.replace("out_0: 7", "out_0: 6")).unwrap_err(),
            "block 'top': output 'out_0': expected cycle 6, got cycle 7"
        );
        Ok(())
    }

    #[test]
    fn all_child_inputs_must_agree_on_one_instance_offset() -> Result<(), String> {
        let source = r#"
block child(clk, x: u8 @0, r: u8 @1) -> u8 @1 {
    reg q: u8; q := x; q + r
}
block top(clk, x: u8 @4, r: u8 @5) -> u8 @5 {
    let joined = child(x, r);
    joined
}
"#;
        check_bx(source)?;
        assert_eq!(
            check_bx(&source.replace("r: u8 @5", "r: u8 @4")).unwrap_err(),
            "block 'top': instance 'joined' input 'r': expected parent cycle 5, got 4"
        );
        check_bx(&source.replace("child(x, r)", "child(u8:7, r)"))?;
        assert_eq!(
            check_bx(&source.replace("child(x, r)", "child(u8:7, u8:7)")).unwrap_err(),
            "block 'top': cannot infer output cycle for this expression"
        );
        Ok(())
    }

    #[test]
    fn explicit_instance_translates_each_output_port_with_a_signed_offset() -> Result<(), String> {
        let source = r#"
struct Ports { live: u8, saved: u8 }
block tap(clk, x: u8 @3) -> Ports @(live: 3, saved: 4) {
    reg q: u8; q := x; Ports { live: x, saved: q }
}
block top(clk, x: u8) -> Ports @(live: 0, saved: 1) {
    lane.x := x;
    inst lane: tap;
    // Reading lane.live must leave x on the adder's operand stack.
    Ports { live: x + lane.live, saved: lane.saved }
}
"#;
        check_bx(source)?;
        assert_eq!(
            check_bx(&source.replace("live: 0, saved: 1", "live: 0, saved: 0")).unwrap_err(),
            "block 'top': output 'saved': expected cycle 0, got cycle 1"
        );
        Ok(())
    }

    #[test]
    fn missing_child_timing_or_an_unknown_connection_stays_unknown() {
        for (child, argument) in [
            ("extern block child(clk, x: u8, r: u8) -> u8;", "x"),
            ("block child(clk, x: u8, r: u8) -> u8 { x + r }", "x"),
            ("block child(clk, x: u8 @0, r: u8 @0) -> u8 { x + r }", "x"),
            (
                "block child(clk, x: u8, r: u8) -> u8 @0 { x + r }",
                "arr[u1:0]",
            ),
        ] {
            let source = format!(
                "{child}\nblock top(clk, x: u8, arr: u8[2]) -> u8 @0 {{ let y = child({argument}, x); y }}"
            );
            assert_eq!(
                check_bx(&source).unwrap_err(),
                "block 'top': cannot infer output cycle for this expression"
            );
        }
    }

    #[test]
    fn child_translation_checks_stage_range_and_feedback() {
        let source = r#"
block delay(clk, x: u8) -> u8 @1 { reg q: u8; q := x; q }
block top(clk, x: u8 @18446744073709551615) -> u8 @0 {
    inst lane: delay; lane.x := x; lane.out
}
"#;
        assert_eq!(
            check_bx(source).unwrap_err(),
            "block 'top': instance 'lane.out': translated cycle 18446744073709551616 is out of range"
        );
        assert_eq!(
            check_bx(&source.replace("lane.x := x", "lane.x := lane.out")).unwrap_err(),
            "block 'top': timing feedback through instance 'lane' is unsupported"
        );
    }

    #[test]
    fn a_deferred_child_timing_error_cannot_supply_a_parent_stage() -> Result<(), String> {
        let checked = check_bx(
            r#"
block child(clk, x: u8) -> u8 @1 { let w: u8; w = x; w }
block top(clk, x: u8) -> u8 @1 { let y = child(x); y }
"#,
        )?;
        let blocks = checked.blocks();
        assert_eq!(
            blocks[0].timing_error.as_deref(),
            Some("expected cycle 1, got cycle 0")
        );
        assert_eq!(
            blocks[1].timing_error.as_deref(),
            Some("cannot infer output cycle for this expression")
        );
        Ok(())
    }

    #[test]
    fn output_cycle_inference_rejects_register_feedback() {
        for rhs in ["q", "q + x"] {
            let source = format!("block b(clk, x: u8) -> u8 {{ reg q: u8; q := {rhs}; q }}");
            check_bx(&source).unwrap();
            let timed = source.replace("-> u8", "-> u8 @1");
            assert_eq!(
                check_bx(&timed).unwrap_err(),
                "block 'b': timing feedback through 'q' is unsupported"
            );
        }
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
#[allow(combinational_block)]
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
