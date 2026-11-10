// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use xlsynth_pir::{
    IrBits, IrValue,
    ir::Type as IrType,
    ir_value_utils::{ir_bits_to_usize, ir_bits_to_usize_in_range},
};

use crate::ast::{BinaryOp, ExternBlock, ExternFn, Param, StructDef, Type, UnaryOp};
use crate::builtin::BuiltinOp;
use crate::bx_import::library_item;
use crate::dslx_import::{ImportedCall, ResolvedImports};
use crate::warnings::BxWarning;
use TypedExprKind as K;

/// A module with concrete signatures, single drivers, and an acyclic instance
/// hierarchy.
#[derive(Clone, Debug)]
pub struct CheckedModule {
    pub(crate) warnings: Vec<BxWarning>,
    pub(crate) structs: Vec<StructDef>,
    pub(crate) enums: BTreeMap<String, CheckedEnum>,
    pub(crate) constants: Vec<CheckedConstant>,
    pub(crate) externs: Vec<ExternFn>,
    /// Qualified call targets of imported extern functions, by declaration
    /// index.
    pub(crate) library_extern_functions: BTreeMap<usize, String>,
    pub(crate) functions: Vec<CheckedFunction>,
    /// Generated function definitions that can be renamed when linked into
    /// an existing package.
    pub(crate) specialized_functions: BTreeSet<String>,
    pub(crate) extern_blocks: Vec<ExternBlock>,
    /// Qualified instance targets of imported extern blocks, by declaration
    /// index.
    pub(crate) library_extern_blocks: BTreeMap<usize, String>,
    pub(crate) imports: ResolvedImports,
    pub(crate) blocks: Vec<CheckedBlock>,
    /// Generated block definitions referenced by inferred call sites.
    pub(crate) specialized_blocks: BTreeSet<String>,
}

impl CheckedModule {
    /// Numeric BX and imported DSLX enum definitions, keyed by their nominal
    /// names.
    pub fn enums(&self) -> &BTreeMap<String, CheckedEnum> {
        &self.enums
    }
    /// Restore source spellings after filtering unreachable externs. Calls and
    /// instances keep their qualified keys to distinguish library declarations.
    pub(crate) fn restore_extern_names(&mut self) {
        self.library_extern_functions =
            take_qualified_names(self.externs.iter_mut().map(|f| &mut f.name));
        self.library_extern_blocks =
            take_qualified_names(self.extern_blocks.iter_mut().map(|b| &mut b.name));
    }

    /// Warnings from the root and its BX libraries in dependency and source
    /// order.
    pub fn warnings(&self) -> &[BxWarning] {
        &self.warnings
    }

    pub fn structs(&self) -> &[StructDef] {
        &self.structs
    }
    /// BX constants and their compile-time values, in dependency and source
    /// order across modules.
    pub fn constants(&self) -> &[CheckedConstant] {
        &self.constants
    }
    /// Extern function declarations with their original names and optional link
    /// symbols.
    pub fn externs(&self) -> &[ExternFn] {
        &self.externs
    }
    /// BX functions from the source and its libraries in dependency order,
    /// with callees preceding callers.
    pub fn functions(&self) -> &[CheckedFunction] {
        &self.functions
    }
    /// Clocked block interfaces that must be supplied by the linked IR package.
    pub fn extern_blocks(&self) -> &[ExternBlock] {
        &self.extern_blocks
    }
    /// Concrete DSLX specializations, indexed by
    /// [`TypedExprKind::ImportedCall`].
    pub fn imported_calls(&self) -> &[ImportedCall] {
        &self.imports.calls
    }
    /// BX blocks from the source and its libraries in dependency order, with
    /// child definitions preceding their parents.
    pub fn blocks(&self) -> &[CheckedBlock] {
        &self.blocks
    }
}

/// Keep the checked key for each library declaration and expose its source
/// name.
fn take_qualified_names<'a>(
    names: impl Iterator<Item = &'a mut String>,
) -> BTreeMap<usize, String> {
    names
        .enumerate()
        .filter_map(|(i, name)| {
            let source = library_item(name)?.1.to_owned();
            Some((i, std::mem::replace(name, source)))
        })
        .collect()
}

/// A nominal hardware enum and the bit patterns of its named members.
#[derive(Clone, Debug)]
pub struct CheckedEnum {
    pub name: String,
    pub public: bool,
    pub signed: bool,
    pub width: usize,
    pub members: BTreeMap<String, IrBits>,
    values: HashSet<IrBits>,
}

impl CheckedEnum {
    /// Bound storage before materializing the discriminants and their lookup
    /// index.
    pub(crate) fn validate_size(width: usize, count: usize) -> Result<(), String> {
        let limit = crate::check::MAX_TYPE_SIZE;
        if width > limit || width.div_ceil(64).max(1).saturating_mul(count) > limit {
            return Err(format!(
                "enum representation exceeds the size limit of {limit}"
            ));
        }
        Ok(())
    }

    pub(crate) fn new(
        name: String,
        public: bool,
        signed: bool,
        width: usize,
        members: BTreeMap<String, IrBits>,
    ) -> Self {
        let values = members.values().cloned().collect();
        Self {
            name,
            public,
            signed,
            width,
            members,
            values,
        }
    }

    pub(crate) fn ty(&self) -> Type {
        Type::Enum {
            name: self.name.clone(),
            signed: self.signed,
            width: self.width,
        }
    }

    pub(crate) fn value_count(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn contains(&self, value: &IrBits) -> bool {
        self.values.contains(value)
    }
}

/// A function value parametric after its source type and constexpr value
/// have been resolved in BX or DSLX.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BoundParametricValue {
    pub ty: Type,
    pub value: IrValue,
}

/// A typed module constant, evaluated before checking its consumers.
#[derive(Clone, Debug)]
pub struct CheckedConstant {
    pub name: String,
    pub public: bool,
    pub ty: Type,
    pub value: IrValue,
}

/// A pure function whose bindings and return value have concrete BX types.
#[derive(Clone, Debug)]
pub struct CheckedFunction {
    pub name: String,
    pub public: bool,
    pub params: Vec<Param>,
    pub result: Type,
    pub bindings: Vec<CheckedBinding>,
    pub output: TypedExpr,
}

/// A checked function binding in evaluation order, including unrolled lets.
#[derive(Clone, Debug)]
pub struct CheckedBinding {
    pub name: String,
    pub value: TypedExpr,
}

/// Ports, static hardware declarations, and connections of a type-checked
/// block.
#[derive(Clone, Debug)]
pub struct CheckedBlock {
    pub name: String,
    pub public: bool,
    pub clock: String,
    pub reset: Option<String>,
    pub params: Vec<Param>,
    pub result: Type,
    pub statements: Vec<CheckedStmt>,
    pub output: TypedExpr,
}

/// A structural declaration or a single driver with its complete enable
/// condition.
#[derive(Clone, Debug)]
pub enum CheckedStmt {
    Reg {
        name: String,
        ty: Type,
        reset: Option<IrValue>,
    },
    Inst {
        name: String,
        block: String,
    },
    Let {
        name: String,
        value: TypedExpr,
    },
    Write {
        name: String,
        value: TypedExpr,
    },
    If {
        condition: TypedExpr,
        body: Vec<CheckedStmt>,
    },
    Connect {
        instance: String,
        port: String,
        value: TypedExpr,
    },
}

/// An expression whose type retains source-level signedness and aggregate
/// names.
#[derive(Clone, Debug)]
pub struct TypedExpr {
    pub ty: Type,
    pub kind: TypedExprKind,
}

/// A checked pattern whose bit values have the subject's width. Range
/// comparisons use the subject's signedness.
#[derive(Clone, Debug)]
pub enum ScalarPattern {
    Wildcard,
    Literal(IrBits),
    Range {
        start: IrBits,
        end: IrBits,
        inclusive: bool,
    },
}

impl ScalarPattern {
    pub(super) fn matches(&self, value: &IrBits, signed: bool) -> bool {
        match self {
            Self::Wildcard => true,
            Self::Literal(literal) => value == literal,
            Self::Range {
                start,
                end,
                inclusive,
            } => {
                let at_or_above = if signed {
                    value.sge(start)
                } else {
                    value.uge(start)
                };
                let below = if signed {
                    value.slt(end)
                } else {
                    value.ult(end)
                };
                at_or_above && (below || *inclusive && value == end)
            }
        }
    }
}

/// Alternatives sharing one checked result in a source-ordered scalar match.
#[derive(Clone, Debug)]
pub struct TypedMatchArm {
    /// An empty list means every source alternative was an empty range.
    pub patterns: Vec<ScalarPattern>,
    pub value: TypedExpr,
}

/// The values visited by a counted fold. A range advances by one in the
/// element's bit type. An array is evaluated once at loop entry.
#[derive(Clone, Debug)]
pub enum TypedIterable {
    Range { start: IrBits },
    Array(TypedExpr),
}

/// A pure counted fold with one checked body shared by all iterations.
#[derive(Clone, Debug)]
pub struct TypedFor {
    pub element: Param,
    pub accumulator: Param,
    pub iterable: TypedIterable,
    pub trip_count: usize,
    pub seed: TypedExpr,
    pub bindings: Vec<CheckedBinding>,
    pub output: TypedExpr,
    /// Deterministic free names computed once for the immutable body.
    pub(super) captures: BTreeSet<String>,
    uses_element: bool,
}

/// An array transformation and the values its checked lambda reads at map
/// entry.
#[derive(Clone, Debug)]
pub struct TypedMap {
    pub array: TypedExpr,
    pub element: Param,
    pub bindings: Vec<CheckedBinding>,
    pub output: TypedExpr,
    pub(super) captures: BTreeSet<String>,
}

/// Resolved operations, including field projections in declaration order.
#[derive(Clone, Debug)]
pub enum TypedExprKind {
    For(Box<TypedFor>),
    Map(Box<TypedMap>),
    Name(String),
    Literal(IrValue),
    Tuple(Vec<TypedExpr>),
    Array(Vec<TypedExpr>),
    Index {
        value: Box<TypedExpr>,
        index: Box<TypedExpr>,
    },
    ArrayUpdate {
        array: Box<TypedExpr>,
        index: Box<TypedExpr>,
        replacement: Box<TypedExpr>,
    },
    Builtin {
        op: BuiltinOp,
        args: Vec<TypedExpr>,
    },
    WidthSlice {
        value: Box<TypedExpr>,
        start: Box<TypedExpr>,
    },
    TupleIndex {
        value: Box<TypedExpr>,
        index: usize,
    },
    /// Project one field of a shared tuple without copying the complete tuple
    /// type and expression into every projection of a destructuring pattern.
    BindingTupleIndex {
        name: String,
        index: usize,
    },
    Call {
        name: String,
        args: Vec<TypedExpr>,
    },
    /// A concrete imported DSLX function specialization.
    ImportedCall {
        /// Index of the specialization in [`CheckedModule::imported_calls`].
        id: usize,
        args: Vec<TypedExpr>,
    },
    Unary {
        op: UnaryOp,
        value: Box<TypedExpr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<TypedExpr>,
        right: Box<TypedExpr>,
    },
    Cast {
        value: Box<TypedExpr>,
    },
    If {
        condition: Box<TypedExpr>,
        then_value: Box<TypedExpr>,
        else_value: Box<TypedExpr>,
    },
    Match {
        subject: Box<TypedExpr>,
        arms: Vec<TypedMatchArm>,
    },
    InstanceOutput {
        instance: String,
        port: String,
    },
}

pub(super) fn bits(ty: &Type) -> Option<(bool, usize)> {
    match ty {
        Type::Bool => Some((false, 1)),
        Type::Bits { signed, width } => Some((*signed, *width)),
        Type::Enum { .. }
        | Type::Tuple(_)
        | Type::Named(_)
        | Type::Applied { .. }
        | Type::ParametricStruct { .. }
        | Type::ParametricBits { .. }
        | Type::ParametricArray { .. }
        | Type::Array { .. } => None,
    }
}

/// Scalar IR representation, including nominal enum types.
pub(crate) fn repr_bits(ty: &Type) -> Option<(bool, usize)> {
    match ty {
        Type::Enum { signed, width, .. } => Some((*signed, *width)),
        _ => bits(ty),
    }
}

/// A value parametric follows DSLX's constexpr rules for lexical inputs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ConstantUse {
    Bx,
    ValueParametric,
}

/// A known lexical constant, or the checked let to replay if a required
/// constant later needs a value that speculative folding deferred.
pub(super) enum ConstantBinding {
    Known(IrValue),
    /// BX folded a selected path. DSLX may need to inspect other inputs
    /// when this let is used as a value parametric.
    Selected {
        value: IrValue,
        definition: usize,
    },
    Deferred(usize),
}

/// A checked expression could not finish evaluating in the current context.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ConstantFailure {
    InvalidEnumCast(String),
    WorkLimit,
    Deferred(usize),
    /// An ordinary speculative fold reached a call. A value-parametric use
    /// can replay the let with the module's concrete functions available.
    NeedsFunctions,
}

/// Memoize concrete local function calls while evaluating an initializer or
/// a value-parametric binding in its lexical scope.
pub(super) struct ConstantFunctions {
    available: AvailableFunctions,
    values: HashMap<ConstantCall, IrValue>,
}

/// Concrete checked functions available to constexpr evaluation, including
/// specializations first requested within the initializer itself.
pub(super) type AvailableFunctions = Rc<RefCell<BTreeMap<String, Rc<CheckedFunction>>>>;

/// The cache key for a pure local helper applied to concrete arguments.
#[derive(Eq, PartialEq, Hash)]
struct ConstantCall {
    name: String,
    arguments: Vec<IrValue>,
}

impl ConstantFunctions {
    pub(super) fn new(available: AvailableFunctions) -> Self {
        Self {
            available,
            values: HashMap::new(),
        }
    }
}

impl ConstantBinding {
    pub(super) fn deferred_definition(&self, usage: ConstantUse) -> Option<usize> {
        match self {
            Self::Deferred(index) => Some(*index),
            Self::Selected { definition, .. } if usage == ConstantUse::ValueParametric => {
                Some(*definition)
            }
            _ => None,
        }
    }

    fn read(&self, work: &mut CountedWork<'_>) -> Option<&IrValue> {
        if let Some(index) = self.deferred_definition(work.usage) {
            work.failure = Some(ConstantFailure::Deferred(index));
            return None;
        }
        match self {
            Self::Known(value) | Self::Selected { value, .. } => Some(value),
            Self::Deferred(_) => unreachable!("deferred definition handled above"),
        }
    }
}

/// Read outer constants and iteration locals only along the selected
/// evaluation path. An uncomputed local identifies its defining statement
/// in the active iteration so the evaluator can resume after computing it.
enum ConstantScope<'a> {
    Outer(&'a BTreeMap<String, ConstantBinding>),
    Iteration {
        parent: &'a Self,
        locals: &'a BTreeMap<&'a str, IrValue>,
        definitions: &'a BTreeMap<&'a str, usize>,
        before: usize,
        depth: usize,
    },
}

impl ConstantScope<'_> {
    fn depth(&self) -> usize {
        match self {
            Self::Outer(_) => 0,
            Self::Iteration { depth, .. } => *depth,
        }
    }

    fn read(&self, name: &str, work: &mut CountedWork<'_>) -> Option<&IrValue> {
        let mut scope = self;
        loop {
            match scope {
                Self::Outer(bindings) => return bindings.get(name)?.read(work),
                Self::Iteration {
                    parent,
                    locals,
                    definitions,
                    before,
                    depth,
                } => {
                    match definitions.get(name) {
                        Some(&index) if index < *before => {
                            if let Some(value) = locals.get(name) {
                                return Some(value);
                            }
                            work.missing_local = Some((*depth, index));
                            return None;
                        }
                        None if let Some(value) = locals.get(name) => return Some(value),
                        _ => { /* This scope has no preceding binding of the name. */ }
                    }
                    scope = parent;
                }
            }
        }
    }
}

/// One constant evaluation charges loop and map iterations to its caller's
/// budget.
struct CountedWork<'a> {
    enums: &'a BTreeMap<String, CheckedEnum>,
    remaining: &'a mut usize,
    inside_loop: bool,
    usage: ConstantUse,
    failure: Option<ConstantFailure>,
    missing_local: Option<(usize, usize)>,
    functions: Option<&'a mut ConstantFunctions>,
}

/// Reserve room for one syntax-depth-limited expression tree at initializer
/// and helper entry. In debug builds its 64 recursive frames can consume
/// several hundred KiB. A stack segment preserves the caller's partially
/// evaluated maps, loops, and arguments while a nested helper runs.
fn on_constant_stack<T>(evaluate: impl FnOnce() -> T) -> T {
    const RED_ZONE: usize = 1024 * 1024;
    stacker::maybe_grow(RED_ZONE, 2 * RED_ZONE, evaluate)
}

impl CountedWork<'_> {
    fn charge(&mut self, work: usize) -> Option<()> {
        if let Some(rest) = self.remaining.checked_sub(work) {
            *self.remaining = rest;
            Some(())
        } else {
            *self.remaining = 0;
            self.failure = Some(ConstantFailure::WorkLimit);
            None
        }
    }

    /// Convert a checked constant, charging the array elements before
    /// allocating or scanning their payload.
    fn cast(&mut self, value: &IrValue, from: &Type, to: &Type) -> Option<IrValue> {
        if let Type::Array { element, size } = to {
            let source = value.as_bits().ok()?;
            let width = bits(element)?.1;
            self.charge(size.saturating_mul(width.div_ceil(64).max(1)).max(1))?;
            let elements = (0..*size)
                .rev()
                .map(|i| IrValue::Bits(source.width_slice((i * width) as i64, width as i64)))
                .collect::<Vec<_>>();
            return IrValue::make_array_typed(IrType::Bits(width), &elements).ok();
        }
        let (_, width) = repr_bits(to)?;
        if let Type::Array { element, size } = from {
            if width == 0 {
                return Some(IrValue::Bits(IrBits::zero(0)));
            }
            let element_width = bits(element)?.1;
            self.charge(constant_work(value))?;
            let elements = value.as_elements().ok()?;
            // Visit each output bit once, starting in the last array element.
            return Some(IrValue::Bits(IrBits::from_lsb_fn(width, |i| {
                elements[size - 1 - i / element_width]
                    .as_bits()
                    .expect("checked cast from a bits array")
                    .get_bit(i % element_width)
                    .expect("bit index within element")
            })));
        }
        let source = value.as_bits().ok()?;
        if let Type::Enum { name, .. } = to {
            // DSLX bytecode checks the original bits before constructing an
            // enum. Hardware IR uses the ordinary bit conversion.
            if matches!(from, Type::Enum { .. }) || !self.enums.get(name)?.contains(source) {
                self.failure = Some(ConstantFailure::InvalidEnumCast(name.clone()));
                return None;
            }
        }
        let fill = repr_bits(from)?.0 && source.is_negative();
        Some(IrValue::Bits(IrBits::from_lsb_fn(width, |i| {
            source.get_bit(i).unwrap_or(fill)
        })))
    }

    /// Apply a preceding pure helper to concrete arguments in its own lexical
    /// scope. Charge arguments, definitions and body evaluation to the caller.
    fn call(
        &mut self,
        name: &str,
        args: &[TypedExpr],
        names: &ConstantScope<'_>,
    ) -> Option<IrValue> {
        let Some(functions) = self.functions.as_ref() else {
            self.failure = Some(ConstantFailure::NeedsFunctions);
            return None;
        };
        let function = functions.available.borrow().get(name)?.clone();
        let args = args
            .iter()
            .map(|arg| arg.constant_with_budget(names, self))
            .collect::<Option<Vec<_>>>()?;
        self.charge(
            args.iter()
                .fold(1usize, |work, arg| work.saturating_add(constant_work(arg))),
        )?;
        let call = ConstantCall {
            name: name.to_owned(),
            arguments: args,
        };
        if let Some(value) = self.functions.as_ref()?.values.get(&call) {
            return Some(value.clone());
        }
        let value = on_constant_stack(|| self.function_body(&function, &call.arguments))?;
        self.functions.as_mut()?.values.insert(call, value.clone());
        Some(value)
    }

    /// Evaluate the checked bindings and output with only the callee's
    /// concrete parameters in scope.
    fn function_body(&mut self, function: &CheckedFunction, args: &[IrValue]) -> Option<IrValue> {
        self.charge(function.bindings.len())?;
        let definitions = function
            .bindings
            .iter()
            .enumerate()
            .map(|(i, b)| (b.name.as_str(), i))
            .collect();
        let mut locals = function
            .params
            .iter()
            .zip(args)
            .map(|(p, value)| (p.name.as_str(), value.clone()))
            .collect();
        let outer = std::mem::replace(&mut self.inside_loop, true);
        let result = constant_iteration(
            &function.bindings,
            &function.output,
            &ConstantScope::Outer(&BTreeMap::new()),
            &mut locals,
            &definitions,
            self,
        );
        self.inside_loop = outer;
        result
    }
}

/// Collect free value names in source order so a let can read an enclosing
/// binding before introducing its own. Nested bodies supply their captures.
fn body_inputs(bindings: &[CheckedBinding], output: &TypedExpr) -> BTreeSet<String> {
    let mut bound = BTreeSet::new();
    let mut inputs = BTreeSet::new();
    let mut record = |expr: &TypedExpr, bound: &BTreeSet<&str>| {
        expr.all_inputs(true, &mut |name| {
            if !bound.contains(name) {
                inputs.insert(name.to_owned());
            }
            true
        });
    };
    for binding in bindings {
        record(&binding.value, &bound);
        bound.insert(binding.name.as_str());
    }
    record(output, &bound);
    inputs
}

impl TypedFor {
    /// Resolve body dependencies once, removing local binders and retaining
    /// whether constant evaluation needs the induction element.
    pub(super) fn new(
        element: Param,
        accumulator: Param,
        iterable: TypedIterable,
        trip_count: usize,
        seed: TypedExpr,
        bindings: Vec<CheckedBinding>,
        output: TypedExpr,
    ) -> Self {
        let mut captures = body_inputs(&bindings, &output);
        let uses_element = captures.remove(&element.name);
        captures.remove(&accumulator.name);
        Self {
            element,
            accumulator,
            iterable,
            trip_count,
            seed,
            bindings,
            output,
            captures,
            uses_element,
        }
    }

    /// Evaluate a fold with constant inputs, sharing the work budget across
    /// nested loops. Each evaluation of a typed body expression spends from
    /// this budget, so speculative folding cannot expand loops indefinitely.
    fn constant(
        &self,
        names: &ConstantScope<'_>,
        remaining: &mut CountedWork<'_>,
    ) -> Option<IrValue> {
        remaining.charge(1)?;
        let mut carry = self.seed.constant_with_budget(names, remaining)?;
        if self.trip_count == 0 {
            return Some(carry);
        }
        let array = match &self.iterable {
            TypedIterable::Array(array) => Some(array.constant_with_budget(names, remaining)?),
            TypedIterable::Range { .. } => None,
        };
        let element_work = match &self.iterable {
            TypedIterable::Range { start } if self.uses_element => {
                start.get_bit_count().div_ceil(64).max(1)
            }
            _ => 1,
        };
        remaining.charge(self.trip_count.saturating_mul(element_work))?;
        remaining.charge(self.bindings.len())?;
        let definitions = self
            .bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| (binding.name.as_str(), index))
            .collect();
        let mut locals = BTreeMap::new();
        for ordinal in 0..self.trip_count {
            locals.clear();
            if self.uses_element {
                let element = match &self.iterable {
                    TypedIterable::Array(_) => array.as_ref()?.get_element(ordinal).ok()?,
                    TypedIterable::Range { start } => {
                        let offset =
                            IrBits::make_ubits(start.get_bit_count(), ordinal.try_into().ok()?)
                                .ok()?;
                        IrValue::Bits(start.add(&offset))
                    }
                };
                locals.insert(self.element.name.as_str(), element);
            }
            locals.insert(self.accumulator.name.as_str(), carry);
            carry = constant_iteration(
                &self.bindings,
                &self.output,
                names,
                &mut locals,
                &definitions,
                remaining,
            )?;
        }
        Some(carry)
    }
}

impl TypedMap {
    /// Record the checked body's free names after binding the map element.
    pub fn new(
        array: TypedExpr,
        element: Param,
        bindings: Vec<CheckedBinding>,
        output: TypedExpr,
    ) -> Self {
        let mut captures = body_inputs(&bindings, &output);
        captures.remove(&element.name);
        Self {
            array,
            element,
            bindings,
            output,
            captures,
        }
    }

    /// Evaluate each element in a child scope of the caller, sharing its work
    /// limit and resolving captured constants only when the body reads them.
    fn constant(&self, names: &ConstantScope<'_>, work: &mut CountedWork<'_>) -> Option<IrValue> {
        let array = self.array.constant_with_budget(names, work)?;
        let elements = array.as_elements().ok()?;
        work.charge(elements.len())?;
        work.charge(self.bindings.len())?;
        let definitions = self
            .bindings
            .iter()
            .enumerate()
            .map(|(index, b)| (b.name.as_str(), index))
            .collect();
        let mut locals = BTreeMap::new();
        let mut values = Vec::with_capacity(elements.len());
        for element in elements {
            locals.clear();
            locals.insert(self.element.name.as_str(), element.clone());
            values.push(constant_iteration(
                &self.bindings,
                &self.output,
                names,
                &mut locals,
                &definitions,
                work,
            )?);
        }
        IrValue::make_array(&values).ok()
    }
}

/// Evaluate only the lets that the iteration result reads. The explicit
/// stack preserves lexical order even for long chains of local aliases.
fn constant_iteration<'a>(
    bindings: &'a [CheckedBinding],
    output: &TypedExpr,
    parent: &ConstantScope<'_>,
    locals: &mut BTreeMap<&'a str, IrValue>,
    definitions: &BTreeMap<&str, usize>,
    work: &mut CountedWork<'_>,
) -> Option<IrValue> {
    let depth = parent.depth() + 1;
    let mut pending: Vec<usize> = Vec::new();
    loop {
        let index = pending.last().copied().unwrap_or(bindings.len());
        let expr = match bindings.get(index) {
            Some(binding) if locals.contains_key(binding.name.as_str()) => {
                pending.pop();
                continue;
            }
            Some(binding) => &binding.value,
            None => output,
        };
        let scope = ConstantScope::Iteration {
            parent,
            locals,
            definitions,
            before: index,
            depth,
        };
        if let Some(value) = expr.constant_with_budget(&scope, work) {
            let Some(index) = pending.pop() else {
                return Some(value);
            };
            locals.insert(bindings[index].name.as_str(), value);
        } else if let Some((owner, missing)) = work.missing_local
            && owner == depth
        {
            work.missing_local = None;
            let mut required = BTreeSet::from([missing]);
            expr.all_inputs(false, &mut |name| {
                if let Some(&i) = definitions.get(name)
                    && i < index
                    && !locals.contains_key(name)
                {
                    required.insert(i);
                }
                true
            });
            work.charge(required.len())?;
            pending.extend(required.into_iter().rev());
        } else {
            return None;
        }
    }
}

/// Storage and arithmetic work of a concrete value produced inside a loop.
/// Charge array elements through their cached type, including nominal
/// payloads whose lowered representation is a tuple.
pub(crate) fn constant_work(value: &IrValue) -> usize {
    fn layout(ty: &IrType) -> usize {
        match ty {
            IrType::Token => 1,
            IrType::Bits(width) => width.div_ceil(64).max(1),
            IrType::Tuple(fields) => fields
                .iter()
                .fold(1usize, |work, field| work.saturating_add(layout(field))),
            IrType::Array(array) => layout(&array.element_type)
                .saturating_mul(array.element_count.max(1))
                .saturating_add(1),
        }
    }
    match value {
        IrValue::Token => 1,
        IrValue::Bits(bits) => bits.get_bit_count().div_ceil(64).max(1),
        IrValue::Tuple(fields) => fields.iter().fold(1usize, |work, field| {
            work.saturating_add(constant_work(field))
        }),
        IrValue::Array(array) => layout(array.element_type())
            .saturating_mul(array.elements().len().max(1))
            .saturating_add(1),
    }
}

/// Select the first matching arm, charging only the alternatives inspected
/// during counted constant evaluation. Range comparisons scan the bit value
/// for each endpoint and may also compare equality at the inclusive limit.
fn constant_match_arm<'a>(
    arms: &'a [TypedMatchArm],
    value: &IrBits,
    signed: bool,
    work: &mut CountedWork<'_>,
) -> Option<&'a TypedExpr> {
    let words = value.get_bit_count().div_ceil(64).max(1);
    for arm in arms {
        if work.inside_loop {
            work.charge(1)?;
        }
        for pattern in &arm.patterns {
            if work.inside_loop {
                let comparisons = match pattern {
                    ScalarPattern::Wildcard => 1,
                    ScalarPattern::Literal(_) => words,
                    ScalarPattern::Range { inclusive, .. } => words * (2 + usize::from(*inclusive)),
                };
                work.charge(comparisons)?;
            }
            if pattern.matches(value, signed) {
                return Some(&arm.value);
            }
        }
    }
    None
}

impl TypedExpr {
    /// Visit the value dependencies of an expression, with lexical binders
    /// removed from nested loop and map bodies.
    fn all_inputs(&self, branches: bool, accept: &mut dyn FnMut(&str) -> bool) -> bool {
        match &self.kind {
            K::Name(name) | K::BindingTupleIndex { name, .. } => accept(name),
            K::Literal(_) => true,
            K::InstanceOutput { instance, port } => accept(&format!("{instance}.{port}")),
            K::For(fold) if !branches && fold.trip_count == 0 => {
                fold.seed.all_inputs(branches, accept)
            }
            K::Map(map) => {
                map.array.all_inputs(branches, accept)
                    && (!branches || map.captures.iter().all(|name| accept(name)))
            }
            K::Call { .. } | K::ImportedCall { .. } if !branches => true,
            K::For(fold) => {
                fold.seed.all_inputs(branches, accept)
                    && match &fold.iterable {
                        TypedIterable::Array(array) => array.all_inputs(branches, accept),
                        TypedIterable::Range { .. } => true,
                    }
                    && (!branches || fold.captures.iter().all(|name| accept(name)))
            }
            K::Tuple(items)
            | K::Array(items)
            | K::Builtin { args: items, .. }
            | K::Call { args: items, .. }
            | K::ImportedCall { args: items, .. } => {
                items.iter().all(|item| item.all_inputs(branches, accept))
            }
            K::TupleIndex { value, .. } | K::Cast { value } | K::Unary { value, .. } => {
                value.all_inputs(branches, accept)
            }
            K::Index {
                value: left,
                index: right,
            }
            | K::WidthSlice {
                value: left,
                start: right,
            }
            | K::Binary { left, right, .. } => {
                left.all_inputs(branches, accept) && right.all_inputs(branches, accept)
            }
            K::If {
                condition,
                then_value,
                else_value,
            } => {
                condition.all_inputs(branches, accept)
                    && (!branches
                        || (then_value.all_inputs(branches, accept)
                            && else_value.all_inputs(branches, accept)))
            }
            K::ArrayUpdate {
                array,
                index,
                replacement,
            } => {
                array.all_inputs(branches, accept)
                    && index.all_inputs(branches, accept)
                    && replacement.all_inputs(branches, accept)
            }
            K::Match { subject, arms } => {
                subject.all_inputs(branches, accept)
                    && (!branches
                        || arms
                            .iter()
                            .all(|arm| arm.value.all_inputs(branches, accept)))
            }
        }
    }

    /// Collect deferred lets needed unconditionally. Once a conditional's
    /// selector is known, the evaluator requests its chosen branch on demand.
    pub(super) fn deferred_inputs(
        &self,
        names: &BTreeMap<String, ConstantBinding>,
        pending: &mut BTreeSet<usize>,
        usage: ConstantUse,
    ) {
        self.all_inputs(false, &mut |name| {
            if let Some(index) = names.get(name).and_then(|b| b.deferred_definition(usage)) {
                pending.insert(index);
            }
            true
        });
    }

    /// A DSLX type dimension requires constant inputs on every branch.
    pub(super) fn has_constant_inputs(&self, names: &BTreeSet<String>) -> bool {
        self.all_inputs(true, &mut |name| names.contains(name))
    }

    /// Evaluate a lexical constant. Module initializers may additionally
    /// invoke the checked local functions visible at their declaration.
    pub(super) fn constant(
        &self,
        names: &BTreeMap<String, ConstantBinding>,
        remaining: &mut usize,
        replaying: bool,
        usage: ConstantUse,
        functions: Option<&mut ConstantFunctions>,
        enums: &BTreeMap<String, CheckedEnum>,
    ) -> Result<Option<IrValue>, ConstantFailure> {
        let may_call = functions.is_some();
        let mut work = CountedWork {
            enums,
            remaining,
            inside_loop: replaying,
            usage,
            failure: None,
            missing_local: None,
            functions,
        };
        if replaying && work.charge(1).is_none() {
            return Err(ConstantFailure::WorkLimit);
        }
        let mut evaluate = || self.constant_with_budget(&ConstantScope::Outer(names), &mut work);
        let value = if may_call {
            on_constant_stack(evaluate)
        } else {
            evaluate()
        };
        if let Some(failure) = work.failure {
            Err(failure)
        } else {
            Ok(value)
        }
    }

    fn constant_with_budget(
        &self,
        names: &ConstantScope<'_>,
        remaining: &mut CountedWork<'_>,
    ) -> Option<IrValue> {
        // DSLX emits bytecode for a constexpr match or loop with bindings for
        // all its free names, including unselected arms. A helper's parameters
        // and iteration locals already have concrete values when executed.
        if remaining.usage == ConstantUse::ValueParametric
            && names.depth() == 0
            && matches!(self.kind, K::Match { .. } | K::For(_) | K::Map(_))
            && !self.all_inputs(true, &mut |name| {
                remaining.charge(1).is_some() && names.read(name, remaining).is_some()
            })
        {
            return None;
        }
        let value = match &self.kind {
            K::Map(map) => {
                let outer = std::mem::replace(&mut remaining.inside_loop, true);
                let value = map.constant(names, remaining);
                remaining.inside_loop = outer;
                value
            }
            K::For(fold) => {
                let outer = std::mem::replace(&mut remaining.inside_loop, true);
                let value = fold.constant(names, remaining);
                remaining.inside_loop = outer;
                value
            }
            K::Literal(v) => Some(v.clone()),
            K::Name(name) => Some(names.read(name, remaining)?.clone()),
            K::Tuple(fields) => Some(IrValue::Tuple(
                fields
                    .iter()
                    .map(|f| f.constant_with_budget(names, remaining))
                    .collect::<Option<_>>()?,
            )),
            K::Array(elements) => {
                let elements = elements
                    .iter()
                    .map(|e| e.constant_with_budget(names, remaining))
                    .collect::<Option<Vec<_>>>()?;
                IrValue::make_array(&elements).ok()
            }
            K::Index { value, index } => {
                let array = value.constant_with_budget(names, remaining)?;
                let size = array.get_element_count().ok()?;
                let index = index.constant_with_budget(names, remaining)?;
                let bits = index.as_bits().ok()?;
                let i = if names.depth() > 0 || remaining.usage == ConstantUse::ValueParametric {
                    // An iteration or constexpr actual can make an index
                    // constant for the first time. Check its array bounds.
                    ir_bits_to_usize_in_range(bits, size)?
                } else {
                    let last = size.checked_sub(1)?;
                    ir_bits_to_usize(bits).unwrap_or(last).min(last)
                };
                array.get_element(i).ok()
            }
            K::ArrayUpdate {
                array,
                index,
                replacement,
            } => {
                let IrValue::Array(array) = array.constant_with_budget(names, remaining)? else {
                    return None;
                };
                // DSLX constexpr evaluation requires an in-bounds update.
                // Runtime IR handles an out-of-bounds index as a no-op.
                let i = ir_bits_to_usize_in_range(
                    index
                        .constant_with_budget(names, remaining)?
                        .as_bits()
                        .ok()?,
                    array.elements().len(),
                )?;
                let replacement = replacement.constant_with_budget(names, remaining)?;
                let mut elements = array.elements().to_vec();
                elements[i] = replacement;
                IrValue::make_array_typed(array.element_type().clone(), &elements).ok()
            }
            K::Builtin { op, args } => {
                let args = args
                    .iter()
                    .map(|arg| arg.constant_with_budget(names, remaining))
                    .collect::<Option<Vec<_>>>()?;
                remaining.charge(op.evaluation_work(&args))?;
                op.evaluate(&args)
            }
            K::WidthSlice { value, start } => {
                let value = value.constant_with_budget(names, remaining)?;
                let source = value.as_bits().ok()?;
                let start = start.constant_with_budget(names, remaining)?;
                let width = bits(&self.ty)?.1;
                let slice = match ir_bits_to_usize_in_range(
                    start.as_bits().ok()?,
                    source.get_bit_count(),
                ) {
                    Some(offset) => source.width_slice(offset as i64, width as i64),
                    None => IrBits::zero(width),
                };
                Some(IrValue::Bits(slice))
            }
            K::TupleIndex { value, index } => value
                .constant_with_budget(names, remaining)?
                .get_element(*index)
                .ok(),
            K::BindingTupleIndex { name, index } => {
                names.read(name, remaining)?.get_element(*index).ok()
            }
            K::Call { name, args } => remaining.call(name, args, names),
            K::ImportedCall { .. } | K::InstanceOutput { .. } => None,
            K::Match { subject, arms } => {
                let value = subject.constant_with_budget(names, remaining)?;
                let bits_value = value.as_bits().ok()?;
                let signed = repr_bits(&subject.ty)?.0;
                constant_match_arm(arms, bits_value, signed, remaining)?
                    .constant_with_budget(names, remaining)
            }
            K::If {
                condition,
                then_value,
                else_value,
            } => {
                if condition
                    .constant_with_budget(names, remaining)?
                    .to_bool()
                    .ok()?
                {
                    then_value.constant_with_budget(names, remaining)
                } else {
                    else_value.constant_with_budget(names, remaining)
                }
            }
            K::Cast { value } => {
                let source = value.constant_with_budget(names, remaining)?;
                remaining.cast(&source, &value.ty, &self.ty)
            }
            K::Unary { op, value } => {
                let v = value
                    .constant_with_budget(names, remaining)?
                    .to_bits()
                    .ok()?;
                match op {
                    UnaryOp::Not => Some(IrValue::Bits(v.not())),
                    UnaryOp::Neg => Some(IrValue::Bits(v.negate())),
                }
            }
            K::Binary { op, left, right } => {
                let lhs = left.constant_with_budget(names, remaining)?;
                let rhs = right.constant_with_budget(names, remaining)?;
                if matches!(op, BinaryOp::Eq | BinaryOp::Ne) {
                    return Some(IrValue::bool((lhs == rhs) == (*op == BinaryOp::Eq)));
                }
                let a = lhs.as_bits().ok()?;
                let b = rhs.as_bits().ok()?;
                let (signed, width) = bits(&left.ty)?;
                let result = match op {
                    BinaryOp::Add => a.add(b),
                    BinaryOp::Sub => a.sub(b),
                    BinaryOp::Mul => a.umul(b).width_slice(0, width as i64),
                    BinaryOp::Div if signed => a.sdiv(b),
                    BinaryOp::Div => a.udiv(b),
                    BinaryOp::Mod if signed => a.smod(b),
                    BinaryOp::Mod => a.umod(b),
                    BinaryOp::And | BinaryOp::LogicalAnd => a.and(b),
                    BinaryOp::Or | BinaryOp::LogicalOr => a.or(b),
                    BinaryOp::Xor => a.xor(b),
                    BinaryOp::Lt => IrBits::bool(if signed { a.slt(b) } else { a.ult(b) }),
                    BinaryOp::Le => IrBits::bool(if signed { a.sle(b) } else { a.ule(b) }),
                    BinaryOp::Gt => IrBits::bool(if signed { a.sgt(b) } else { a.ugt(b) }),
                    BinaryOp::Ge => IrBits::bool(if signed { a.sge(b) } else { a.uge(b) }),
                    BinaryOp::Shl | BinaryOp::Shr => {
                        let shift = ir_bits_to_usize(b).unwrap_or(width).min(width) as i64;
                        if *op == BinaryOp::Shl {
                            a.shll(shift)
                        } else if signed {
                            a.shra(shift)
                        } else {
                            a.shrl(shift)
                        }
                    }
                    BinaryOp::Concat => IrBits::from_lsb_fn(bits(&self.ty)?.1, |i| {
                        if i < b.get_bit_count() {
                            b.get_bit(i).expect("index within right operand")
                        } else {
                            a.get_bit(i - b.get_bit_count())
                                .expect("index within left operand")
                        }
                    }),
                    BinaryOp::Eq | BinaryOp::Ne => {
                        unreachable!("equality evaluated on arbitrary values above")
                    }
                };
                Some(IrValue::Bits(result))
            }
        }?;
        if remaining.inside_loop {
            remaining.charge(constant_work(&value))?;
        }
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use xlsynth_pir::{IrBits, IrValue};

    use super::{
        CheckedStmt, ConstantBinding, ConstantFailure, ConstantFunctions, ConstantUse, TypedExpr,
        TypedExprKind,
    };
    use crate::{check_module, parse_module};

    #[test]
    fn nested_constant_helper_expressions_fit_a_small_stack() -> Result<(), String> {
        let source = format!(
            "fn inner(x:u8)->u8 {{ {}x }} fn outer(x:u8)->u8 {{ {}inner(x) }} fn result()->u8 {{ outer(u8:1) }}",
            "!".repeat(60),
            "!".repeat(27),
        );
        let functions = check_module(&parse_module(&source)?)?.functions;
        // Exercise only evaluation, after the parser and checker have released
        // their frames. Each helper expression fits individually on this stack.
        std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(move || {
                let available: super::AvailableFunctions = Default::default();
                for f in functions {
                    available
                        .borrow_mut()
                        .insert(f.name.clone(), super::Rc::new(f));
                }
                let result = available.borrow()["result"].clone();
                let mut calls = ConstantFunctions::new(available);
                let mut remaining = crate::check::MAX_TYPE_SIZE;
                assert_eq!(
                    result.output.constant(
                        &BTreeMap::new(),
                        &mut remaining,
                        true,
                        ConstantUse::Bx,
                        Some(&mut calls),
                        &BTreeMap::new(),
                    ),
                    Ok(Some(IrValue::make_ubits(8, 254).unwrap()))
                );
            })
            .map_err(|e| e.to_string())?
            .join()
            .expect("constant evaluator thread completed");
        Ok(())
    }

    #[test]
    fn deeply_nested_constant_map_stays_within_the_work_budget() -> Result<(), String> {
        let mut expr = "map(A, |x| inc(x))".to_owned();
        for _ in 0..32 {
            expr = format!("if true {{ {expr} }} else {{ A }}");
        }
        let source = format!(
            "fn inc(x:u16)->u16 {{ x + u16:1 }} const A = map(u16:0..u16:1024, |i| i); const C = ({expr})[u32:1023];"
        );
        let checked = check_module(&parse_module(&source)?)?;
        assert_eq!(
            checked.constants()[1].value,
            IrValue::make_ubits(16, 1024).unwrap()
        );
        Ok(())
    }

    #[test]
    fn counted_reset_checks_indices_on_the_selected_iteration_path() -> Result<(), String> {
        let source = |body| {
            format!(
                r#"
block folded(clk, rst, x: u8) -> u8 {{
    let xs = [u8:2, u8:3];
    reg q: u8 = for (i, a) in u32:0..u32:3 {{ {body} }}(u8:0);
    q := x;
    q
}}
"#
            )
        };
        let invalid = parse_module(&source("xs[i]"))?;
        assert!(
            check_module(&invalid)
                .unwrap_err()
                .contains("must be a compile-time constant")
        );
        let valid = check_module(&parse_module(&source(
            "if i < u32:2 { a + xs[i] } else { a }",
        ))?)?;
        let expected = IrValue::make_ubits(8, 5).unwrap();
        assert!(valid.blocks()[0].statements.iter().any(|stmt| matches!(
            stmt, CheckedStmt::Reg { reset: Some(value), .. } if value == &expected
        )));
        Ok(())
    }

    #[test]
    fn counted_match_charges_only_inspected_alternatives() -> Result<(), String> {
        let alternatives = (0..64)
            .map(|n| format!("u16:{n}"))
            .collect::<Vec<_>>()
            .join(" | ");
        for (seed, expected) in [
            (0, Ok(Some(IrValue::make_ubits(16, 0).unwrap()))),
            (65535, Err(ConstantFailure::WorkLimit)),
        ] {
            let source = format!(
                r#"
block folded(clk) -> u16 {{
    for (_, a) in u32:0..u32:4 {{
        match a {{ {alternatives} => u16:0, _ => a }}
    }}(u16:{seed})
}}
"#
            );
            let module = check_module(&parse_module(&source)?)?;
            let mut remaining = 128;
            assert_eq!(
                module.blocks()[0].output.constant(
                    &BTreeMap::new(),
                    &mut remaining,
                    false,
                    ConstantUse::Bx,
                    None,
                    &BTreeMap::new()
                ),
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn constant_casts_charge_array_work_outside_loops() -> Result<(), String> {
        let source = r#"
fn unpack(x: uN[0]) -> uN[0][64] { x as uN[0][64] }
fn pack(x: uN[256][2]) -> uN[512] { x as uN[512] }
"#;
        let module = check_module(&parse_module(source)?)?;
        for function in module.functions() {
            let input = match function.name.as_str() {
                "unpack" => IrValue::Bits(IrBits::zero(0)),
                "pack" => IrValue::make_array(&[
                    IrValue::Bits(IrBits::all_ones(256)),
                    IrValue::Bits(IrBits::zero(256)),
                ])
                .map_err(|e| e.to_string())?,
                _ => unreachable!("only the two cast functions are declared"),
            };
            let names = BTreeMap::from([("x".to_owned(), ConstantBinding::Known(input))]);
            let mut remaining = 4;
            assert_eq!(
                function.output.constant(
                    &names,
                    &mut remaining,
                    false,
                    ConstantUse::Bx,
                    None,
                    &BTreeMap::new(),
                ),
                Err(ConstantFailure::WorkLimit),
            );
        }
        Ok(())
    }

    #[test]
    fn nested_maps_replay_local_arrays_within_one_constant_work_budget() -> Result<(), String> {
        let source = r#"
block folded(clk, rows: u8[2][2]) -> u8[2][2] {
    map(rows, |row| {
        let cells = row;
        map(cells, |cell| {
            let plus = cell + u8:1;
            plus
        })
    })
}
"#;
        let module = check_module(&parse_module(source)?)?;
        let output = &module.blocks()[0].output;
        let input = IrValue::parse_typed("[[bits[8]:0, bits[8]:1], [bits[8]:255, bits[8]:2]]")
            .map_err(|e| e.to_string())?;
        let expected = IrValue::parse_typed("[[bits[8]:1, bits[8]:2], [bits[8]:0, bits[8]:3]]")
            .map_err(|e| e.to_string())?;
        let names = BTreeMap::from([("rows".to_string(), ConstantBinding::Known(input))]);
        let mut ample = 1024;
        assert_eq!(
            output.constant(
                &names,
                &mut ample,
                false,
                ConstantUse::Bx,
                None,
                &BTreeMap::new()
            ),
            Ok(Some(expected))
        );
        let mut limited = 26;
        assert_eq!(
            output.constant(
                &names,
                &mut limited,
                false,
                ConstantUse::Bx,
                None,
                &BTreeMap::new()
            ),
            Err(ConstantFailure::WorkLimit)
        );
        Ok(())
    }

    #[test]
    fn structural_iterations_and_specializations_share_speculative_work() -> Result<(), String> {
        let body = r#"
            let sum = unroll_for! (_, a) in u8:0..u8:TRIPS {
                for (_, b) in u8:0..u8:64 { b + uN[65536]:1 }(a)
            }(uN[65536]:0);
            let two = u32:1 + u32:1;
            reg q: u32 = for (_, b) in u32:0..two { b + u32:1 }(two);
            q := sum as u32;
            q
        "#;
        let source = format!(
            r#"
block ordinary(clk, rst) -> u32 {{ {} }}
block folded<T>(clk, rst, unused: T) -> u32 {{ {} }}
block wrapper(clk, rst, x: u8, y: u16) -> (u32, u32) {{
    let a = folded(x);
    let b = folded(y);
    (a, b)
}}
"#,
            body.replace("TRIPS", "8"),
            body.replace("TRIPS", "4")
        );
        let module = check_module(&parse_module(&source)?)?;
        let mut specialized_runtime = Vec::new();
        for block in module.blocks().iter().filter(|b| b.name != "wrapper") {
            let runtime = block.statements.iter().any(|stmt| {
                matches!(
                    stmt,
                    CheckedStmt::Let {
                        value: TypedExpr {
                            kind: TypedExprKind::For(_),
                            ..
                        },
                        ..
                    }
                )
            });
            if block.name == "ordinary" {
                assert!(runtime);
            } else {
                specialized_runtime.push(runtime);
            }
            assert!(block.statements.iter().any(|stmt| matches!(stmt,
                CheckedStmt::Reg { reset: Some(value), .. } if *value == IrValue::u32(4)
            )));
        }
        // Four wide folds fit in one body allowance. The second distinct
        // specialization must emit remaining iterations as runtime loops.
        assert_eq!(specialized_runtime, [false, true]);
        Ok(())
    }

    #[test]
    fn required_uses_replay_a_small_counted_let_after_speculative_exhaustion() -> Result<(), String>
    {
        let source = r#"
block stage(clk, rst, input: u8) -> (u8, u8) {
    let spent = for (i, a) in uN[1048576]:0..uN[1048576]:128 { a + (i as u8) }(u8:0);
    let small = for (_, a) in u32:0..u32:3 { a + u8:2 }(u8:0);
    let alias = if true { small + u8:1 } else { spent };
    let selected = if false { input } else { alias };
    reg qa: u8 = selected;
    reg qb: u8 = (for (_, a) in u32:0..u32:3 { a + u8:2 }(u8:0)) + u8:1;
    qa := u8:0;
    qb := u8:0;
    (qa, qb)
}
"#;
        let module = check_module(&parse_module(source)?)?;
        let resets: Vec<_> = module.blocks()[0]
            .statements
            .iter()
            .filter_map(|stmt| {
                if let CheckedStmt::Reg { reset, .. } = stmt {
                    reset.clone()
                } else {
                    None
                }
            })
            .collect();
        let seven = IrValue::make_ubits(8, 7).map_err(|e| e.to_string())?;
        assert_eq!(resets, [seven.clone(), seven]);
        Ok(())
    }

    #[test]
    fn counted_constants_read_captures_and_body_lets_only_on_selected_paths() -> Result<(), String>
    {
        let source = r#"
block stage(clk, rst, x: u32) -> u32 {
    let spent = for (i, b) in uN[1048576]:0..uN[1048576]:128 { b + (i as u32) }(u32:0);
    reg q: u32 = for (_, a) in u32:0..u32:2 {
        let runtime = x + u32:1;
        let costly = spent + u32:1;
        let next = a + u32:1;
        if true { for (_, b) in u32:0..u32:1 { b + next }(u32:0) } else { runtime + costly }
    }(u32:0);
    q := x;
    q
}
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert!(checked.blocks()[0].statements.iter().any(|s| matches!(s,
            CheckedStmt::Reg { reset: Some(value), .. } if *value == IrValue::u32(2)
        )));
        for (live, expected) in [
            ("runtime", "must be a compile-time constant"),
            ("costly", "constant evaluation exceeds the work limit"),
        ] {
            let source = source
                .replace("if true", "if false")
                .replace("runtime + costly }", &format!("{live} }}"));
            let error = check_module(&parse_module(&source)?).unwrap_err();
            assert!(error.contains(expected), "{live}: {error}");
        }
        Ok(())
    }

    #[test]
    fn required_constant_reports_a_work_limited_let_through_its_alias() -> Result<(), String> {
        let source = r#"
block wrapper(clk, rst) -> u8 {
    let deferred = for (i, acc) in uN[1048576]:0..uN[1048576]:128 { acc + (i as u8) }(u8:0);
    let alias = deferred + u8:1;
    reg q: u8 = alias;
    q := u8:0;
    q
}
"#;
        let error = check_module(&parse_module(source)?).unwrap_err();
        assert!(
            error.contains("constant evaluation exceeds the work limit"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn speculative_counted_folding_charges_wide_indices_and_nominal_captures() -> Result<(), String>
    {
        let module = parse_module(
            r#"
struct Payload { word: uN[65536] }

block wide_index(clk) -> u8 {
    for (i, acc) in uN[1048576]:0..uN[1048576]:128 { acc + (i as u8) }(u8:0)
}

block wide_capture(clk, data: Payload) -> bool {
    for (_, acc) in u32:0..u32:8192 { acc && (data == data) }(true)
}
"#,
        )?;
        let checked = check_module(&module)?;
        let values = BTreeMap::from([(
            "data".to_owned(),
            ConstantBinding::Known(IrValue::Tuple(
                vec![IrValue::Bits(IrBits::zero(65536))].into(),
            )),
        )]);
        for block in checked.blocks() {
            let mut remaining = crate::check::MAX_TYPE_SIZE;
            assert_eq!(
                block.output.constant(
                    &values,
                    &mut remaining,
                    false,
                    ConstantUse::Bx,
                    None,
                    &BTreeMap::new()
                ),
                Err(ConstantFailure::WorkLimit),
                "{}",
                block.name
            );
        }
        Ok(())
    }
}
