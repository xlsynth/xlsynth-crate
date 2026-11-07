// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use xlsynth_pir::{
    IrBits, IrValue,
    ir::Type as IrType,
    ir_value_utils::{ir_bits_to_usize, ir_bits_to_usize_in_range},
};

use crate::ast::{BinaryOp, ExternBlock, ExternFn, Param, StructDef, Type, UnaryOp};
use crate::dslx_import::{ImportedCall, ResolvedImports};
use TypedExprKind as K;

/// A module with concrete signatures, single drivers, and an acyclic instance
/// hierarchy.
#[derive(Clone, Debug)]
pub struct CheckedModule {
    pub(crate) structs: Vec<StructDef>,
    pub(crate) externs: Vec<ExternFn>,
    pub(crate) functions: Vec<CheckedFunction>,
    pub(crate) extern_blocks: Vec<ExternBlock>,
    pub(crate) imports: ResolvedImports,
    pub(crate) blocks: Vec<CheckedBlock>,
    /// Generated block definitions referenced by inferred call sites.
    pub(crate) specialized_blocks: BTreeSet<String>,
}

impl CheckedModule {
    pub fn structs(&self) -> &[StructDef] {
        &self.structs
    }
    pub fn externs(&self) -> &[ExternFn] {
        &self.externs
    }
    /// Local functions in dependency order, with callees preceding callers.
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
    /// Blocks in dependency order, with child definitions preceding their
    /// parents.
    pub fn blocks(&self) -> &[CheckedBlock] {
        &self.blocks
    }
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

/// Resolved operations, including field projections in declaration order.
#[derive(Clone, Debug)]
pub enum TypedExprKind {
    For(Box<TypedFor>),
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
        Type::Tuple(_) | Type::Named(_) | Type::Array { .. } => None,
    }
}

/// A known lexical constant, or the checked let to replay if a required
/// constant later needs a value that speculative folding deferred.
pub(super) enum ConstantBinding {
    Known(IrValue),
    Deferred(usize),
}

/// Evaluation either exceeded its work budget or needs an earlier let.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ConstantFailure {
    WorkLimit,
    Deferred(usize),
}

impl ConstantBinding {
    fn read(&self, work: &mut CountedWork<'_>) -> Option<&IrValue> {
        match self {
            Self::Known(value) => Some(value),
            Self::Deferred(index) => {
                work.failure = Some(ConstantFailure::Deferred(*index));
                None
            }
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

/// One constant evaluation charges counted-loop work to its caller's budget.
struct CountedWork<'a> {
    remaining: &'a mut usize,
    inside_loop: bool,
    failure: Option<ConstantFailure>,
    missing_local: Option<(usize, usize)>,
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
        let mut bound = BTreeSet::new();
        let mut captures = BTreeSet::new();
        let mut record = |expr: &TypedExpr, bound: &BTreeSet<&str>| {
            expr.all_inputs(true, &mut |name| {
                if !bound.contains(name) {
                    captures.insert(name.to_owned());
                }
                true
            });
        };
        for binding in &bindings {
            record(&binding.value, &bound);
            bound.insert(&binding.name);
        }
        record(&output, &bound);
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
            carry = self.constant_iteration(names, &mut locals, &definitions, remaining)?;
        }
        Some(carry)
    }

    /// Evaluate only the lets that the iteration result reads. The explicit
    /// stack preserves lexical order even for long chains of local aliases.
    fn constant_iteration<'a>(
        &'a self,
        parent: &ConstantScope<'_>,
        locals: &mut BTreeMap<&'a str, IrValue>,
        definitions: &BTreeMap<&str, usize>,
        work: &mut CountedWork<'_>,
    ) -> Option<IrValue> {
        let depth = parent.depth() + 1;
        let mut pending: Vec<usize> = Vec::new();
        loop {
            let index = pending.last().copied().unwrap_or(self.bindings.len());
            let expr = match self.bindings.get(index) {
                Some(binding) if locals.contains_key(binding.name.as_str()) => {
                    pending.pop();
                    continue;
                }
                Some(binding) => &binding.value,
                None => &self.output,
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
                locals.insert(self.bindings[index].name.as_str(), value);
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
}

/// Storage and arithmetic work of a concrete value produced inside a loop.
/// Charge array elements through their cached type, including nominal
/// payloads whose lowered representation is a tuple.
fn constant_work(value: &IrValue) -> usize {
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
    /// removed from nested loop bodies.
    fn all_inputs(&self, branches: bool, accept: &mut dyn FnMut(&str) -> bool) -> bool {
        match &self.kind {
            K::Name(name) | K::BindingTupleIndex { name, .. } => accept(name),
            K::Literal(_) => true,
            K::InstanceOutput { instance, port } => accept(&format!("{instance}.{port}")),
            K::For(fold) if !branches && fold.trip_count == 0 => {
                fold.seed.all_inputs(branches, accept)
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
    ) {
        self.all_inputs(false, &mut |name| {
            if let Some(ConstantBinding::Deferred(index)) = names.get(name) {
                pending.insert(*index);
            }
            true
        });
    }

    /// A DSLX type dimension requires constant inputs on every branch.
    pub(super) fn has_constant_inputs(&self, names: &BTreeSet<String>) -> bool {
        self.all_inputs(true, &mut |name| names.contains(name))
    }

    /// Evaluate register reset constants, allowing prior constant let bindings.
    pub(super) fn constant(
        &self,
        names: &BTreeMap<String, ConstantBinding>,
        remaining: &mut usize,
        replaying: bool,
    ) -> Result<Option<IrValue>, ConstantFailure> {
        let mut work = CountedWork {
            remaining,
            inside_loop: replaying,
            failure: None,
            missing_local: None,
        };
        if replaying && work.charge(1).is_none() {
            return Err(ConstantFailure::WorkLimit);
        }
        let value = self.constant_with_budget(&ConstantScope::Outer(names), &mut work);
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
        let value = match &self.kind {
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
                let i = if names.depth() > 0 {
                    // An iteration can make an index constant for the first
                    // time. Apply the same bounds check as a constant BX index.
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
            K::Call { .. } | K::ImportedCall { .. } | K::InstanceOutput { .. } => None,
            K::Match { subject, arms } => {
                let value = subject.constant_with_budget(names, remaining)?;
                let bits_value = value.as_bits().ok()?;
                let signed = bits(&subject.ty)?.0;
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
                let source = value
                    .constant_with_budget(names, remaining)?
                    .to_bits()
                    .ok()?;
                let fill = bits(&value.ty)?.0 && source.is_negative();
                let width = bits(&self.ty)?.1;
                Some(IrValue::Bits(IrBits::from_lsb_fn(width, |i| {
                    source.get_bit(i).unwrap_or(fill)
                })))
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

    use super::{CheckedStmt, ConstantBinding, ConstantFailure, TypedExpr, TypedExprKind};
    use crate::{check_module, parse_module};

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
                module.blocks()[0]
                    .output
                    .constant(&BTreeMap::new(), &mut remaining, false),
                expected
            );
        }
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
            (
                "costly",
                "counted for constant evaluation exceeds the work limit",
            ),
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
            error.contains("counted for constant evaluation exceeds the work limit"),
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
                block.output.constant(&values, &mut remaining, false),
                Err(ConstantFailure::WorkLimit),
                "{}",
                block.name
            );
        }
        Ok(())
    }
}
