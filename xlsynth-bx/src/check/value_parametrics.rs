// SPDX-License-Identifier: Apache-2.0

//! Bind DSLX-style function value parametrics from constexpr actuals, argument
//! types and defaults, then typecheck each concrete function specialization.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use xlsynth_pir::{IrValue, ir_value_utils::ir_bits_to_usize};

use super::callables::CallbackSource;
use super::function_specializations::{
    BoundValueSignature, ExplicitEnvironment, InferredCallback, SuppliedParametrics,
};
use super::generic_blocks::{Bindings, callback_argument_name, validate_inference_sources};
use super::generic_inputs::{Input, InputPattern};
use super::{
    BodyChecker, BodyKind, CallbackFormal, CallbackTarget, Declarations, FunctionInterface, K,
    MAX_ELABORATION_DEPTH, StructTable, TypeResolver, TypeShape, bits, child_type_budget,
    compatible, declare, literal_copy_work, minimum_type, resolve_type, source_type_nodes,
};
use crate::ast::{
    CallbackParam, Expr, Function, FunctionParametricRef, Param, ParametricBinding,
    StructLiteralArgs, Type,
};
use crate::dslx_import::ImportResolver;
use crate::typed::{BoundParametricValue, TypedExpr};

pub(super) fn dimension_type() -> Type {
    Type::unsigned(32)
}

/// Total source storage copied from bound type parameters in one annotation.
#[derive(Default)]
struct TypeArgumentCopies {
    nodes: usize,
    name_bytes: usize,
}

impl<'m> Declarations<'m> {
    /// Retain the declaration's symbolic signature until a call supplies its
    /// environment. Formal values occupy the same lexical namespace as data.
    pub(super) fn add_value_template(
        &mut self,
        f: &'m Function,
        structs: &StructTable,
    ) -> Result<(), String> {
        let mut names = BTreeSet::new();
        for name in &f.type_params {
            if !crate::parser::is_use_identifier(name) {
                return Err(format!("invalid type parameter name '{name}'"));
            }
            declare(&mut names, name)?;
        }
        let needs_result_type = !f.type_params.is_empty()
            && validate_inference_sources(&f.type_params, &f.params, &f.callback_params, None)
                .is_err();
        if needs_result_type {
            validate_inference_sources(
                &f.type_params,
                &f.params,
                &f.callback_params,
                Some(&f.result),
            )?;
        }
        let mut last = None;
        for p in &f.value_parametrics {
            if p.index >= f.value_parametrics.len() + f.callback_params.len()
                || last.is_some_and(|i| i >= p.index)
            {
                return Err("invalid value parametric declaration order".into());
            }
            last = Some(p.index);
            declare(&mut names, &p.name)?;
            match &p.ty {
                Type::Bits { .. } | Type::Bool | Type::Enum { .. } => {
                    structs.validate_type(&p.ty)?
                }
                Type::ParametricBits { .. }
                | Type::Named(_)
                | Type::Applied { .. }
                | Type::ParametricStruct { .. }
                | Type::TypeApplied { .. } => {
                    // Resolve dependent widths and nominal enum or struct
                    // annotations in the template's declaration scope.
                }
                _ => {
                    return Err(format!(
                        "value parametric '{}' requires a bits, bool, enum, or struct type",
                        p.name
                    ));
                }
            }
        }
        let mut callback_indices = BTreeMap::new();
        for (index, callback) in f.callback_params.iter().enumerate() {
            declare(&mut names, &callback.name)?;
            callback_indices.insert(callback.name.clone(), index);
        }
        for p in &f.params {
            declare(&mut names, &p.name)?;
        }
        if needs_result_type {
            self.result_context
                .insert(&f.name, super::ResultContext::TypeParameter);
        } else if !f.value_parametrics.is_empty() {
            // The specialization binds missing dimensions from data arguments
            // before it checks callback signatures.
            let mut dimensions = BTreeSet::new();
            for param in &f.params {
                super::type_dependencies(
                    &param.ty,
                    |_, _| {},
                    |expr| {
                        if let Expr::Name(name) = expr {
                            dimensions.insert(name.as_str());
                        }
                    },
                );
            }
            if let Some(formal) = f
                .value_parametrics
                .iter()
                .rev()
                .find(|p| p.default.is_none() && !dimensions.contains(p.name.as_str()))
            {
                self.result_context.insert(
                    &f.name,
                    super::ResultContext::ExplicitValues(formal.index + 1),
                );
            }
        }
        self.value_template_callbacks
            .insert(&f.name, callback_indices);
        self.function_templates.insert(&f.name, f);
        Ok(())
    }
}

/// Convert a bits or bool value to a host-sized dimension without truncating
/// high bits or interpreting a negative signed value as a width.
fn nonnegative_dimension(ty: &Type, value: &IrValue) -> Result<usize, String> {
    let (signed, _) =
        bits(ty).ok_or_else(|| format!("type dimension requires bits or bool, got {ty}"))?;
    let bits = value.as_bits().map_err(|e| e.to_string())?;
    if signed && bits.msb() {
        return Err("type dimension cannot be negative".into());
    }
    ir_bits_to_usize(bits).ok_or_else(|| "dimension does not fit usize".to_owned())
}

/// A bit width or sign can adopt a struct formal's type. Array lengths and
/// forwarded nominal bindings supply a value of a specific type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WitnessType {
    BitDimension,
    Exact,
}

/// Only a bare binding in a dimension provides an inference constraint. A
/// computed dimension is checked after its operands have concrete values.
pub(super) struct InferredBindings<'f> {
    formals: &'f [ParametricBinding],
    indices: BTreeMap<&'f str, usize>,
    pub(super) values: Vec<Option<BoundParametricValue>>,
    explicit_count: usize,
    // Struct bit widths and signedness provide unsized integer witnesses.
    // Array lengths and nested nominal arguments retain their source types.
    struct_bit_dimensions: Option<BTreeSet<usize>>,
    // Gather all witnesses from one data argument before its bindings become
    // context for later arguments. Tuple fields must agree with one another.
    witnesses: BTreeMap<usize, BoundParametricValue>,
}

impl<'f> InferredBindings<'f> {
    pub(super) fn new(formals: &'f [ParametricBinding], explicit_count: usize) -> Self {
        Self {
            formals,
            indices: formals
                .iter()
                .enumerate()
                .map(|(i, p)| (p.name.as_str(), i))
                .collect(),
            values: vec![None; formals.len()],
            explicit_count,
            struct_bit_dimensions: None,
            witnesses: BTreeMap::new(),
        }
    }

    pub(super) fn for_struct(formals: &'f [ParametricBinding], explicit_count: usize) -> Self {
        let mut bindings = Self::new(formals, explicit_count);
        bindings.struct_bit_dimensions = Some(BTreeSet::new());
        bindings
    }

    /// Give a bit-width or signedness witness its resolved declaration type.
    /// Other witnesses, including a forwarded nominal argument, must match
    /// exactly.
    pub(super) fn struct_value(
        &self,
        index: usize,
        ty: &Type,
    ) -> Result<Option<BoundParametricValue>, String> {
        let Some(bound) = self.values[index].as_ref() else {
            return Ok(None);
        };
        if !self
            .struct_bit_dimensions
            .as_ref()
            .is_some_and(|dims| dims.contains(&index))
        {
            return Ok(Some(bound.clone()));
        }
        let (signed, width) = bits(ty).ok_or_else(|| {
            format!(
                "parametric '{}' cannot infer {ty} from a bit dimension",
                self.formals[index].name
            )
        })?;
        let n = nonnegative_dimension(&bound.ty, &bound.value)?;
        let value = if signed {
            IrValue::make_sbits(
                width,
                i64::try_from(n).map_err(|_| "dimension exceeds i64")?,
            )
        } else {
            IrValue::make_ubits(width, n as u64)
        }
        .map_err(|_| {
            format!(
                "inferred dimension {n} does not fit {ty} for parametric '{}'",
                self.formals[index].name
            )
        })?;
        Ok(Some(BoundParametricValue {
            ty: ty.clone(),
            value,
        }))
    }

    fn missing(&self, dimension: &Expr) -> Option<usize> {
        let Expr::Name(name) = dimension else {
            return None;
        };
        let i = *self.indices.get(name.as_str())?;
        (self.formals[i].index >= self.explicit_count
            && self.values[i].is_none()
            && self.formals[i].default.is_none())
        .then_some(i)
    }

    fn needs_witness(&self, ty: &Type) -> bool {
        match ty {
            Type::ParametricBits { signed, width } => {
                self.missing(signed).is_some() || self.missing(width).is_some()
            }
            Type::ParametricArray { element, size } => {
                self.missing(size).is_some_and(|i| {
                    self.witnesses.get(&i).is_none_or(|previous| {
                        // An array length supplies an exact u32. A prior bit
                        // dimension or a differently typed nominal argument
                        // still needs to agree with that constraint.
                        self.struct_bit_dimensions.as_ref().is_some_and(|dims| {
                            dims.contains(&i) || previous.ty != dimension_type()
                        })
                    })
                }) || self.needs_witness(element)
            }
            Type::Array { element, .. } => self.needs_witness(element),
            Type::Tuple(fields) | Type::TypeApplied { args: fields, .. } => {
                fields.iter().any(|f| self.needs_witness(f))
            }
            Type::ParametricStruct { args, .. } => {
                args.iter().any(|arg| self.missing(arg).is_some())
            }
            _ => false,
        }
    }

    /// Merge evidence from one field. A typed nominal argument can also
    /// constrain a binding that supplies a numeric bit width in that field.
    fn bind_witness(
        &mut self,
        dimension: &Expr,
        ty: &Type,
        value: IrValue,
        witness_type: WitnessType,
    ) -> Result<(), String> {
        let Some(index) = self.missing(dimension) else {
            return Ok(());
        };
        let contextual =
            self.struct_bit_dimensions.is_some() && witness_type == WitnessType::BitDimension;
        let witness = BoundParametricValue {
            ty: ty.clone(),
            value,
        };
        if let Some(previous) = self.witnesses.get(&index) {
            let previous_contextual = self
                .struct_bit_dimensions
                .as_ref()
                .is_some_and(|dims| dims.contains(&index));
            let equal = (compatible(&previous.ty, &witness.ty) && previous.value == witness.value)
                || ((previous_contextual || contextual)
                    && nonnegative_dimension(&previous.ty, &previous.value)
                        .ok()
                        .is_some_and(|n| {
                            nonnegative_dimension(&witness.ty, &witness.value) == Ok(n)
                        }));
            if !equal {
                return Err(format!(
                    "conflicting values inferred for parametric '{}': {} and {}",
                    self.formals[index].name, previous.value, witness.value
                ));
            }
            if contextual && !previous_contextual {
                return Ok(());
            }
        }
        if let Some(dims) = &mut self.struct_bit_dimensions {
            if contextual {
                dims.insert(index);
            } else {
                dims.remove(&index);
            }
        }
        self.witnesses.insert(index, witness);
        Ok(())
    }

    pub(super) fn finish_argument(&mut self) {
        for (index, value) in std::mem::take(&mut self.witnesses) {
            self.values[index] = Some(value);
        }
    }

    fn bind_size(
        &mut self,
        dimension: &Expr,
        size: usize,
        witness_type: WitnessType,
    ) -> Result<(), String> {
        if self.missing(dimension).is_some() {
            let value = IrValue::make_ubits(32, size as u64).map_err(|e| e.to_string())?;
            self.bind_witness(dimension, &dimension_type(), value, witness_type)?;
        }
        Ok(())
    }

    fn infer(
        &mut self,
        formal: &Type,
        actual: &Type,
        structs: &StructTable,
        resolver: &ImportResolver,
    ) -> Result<(), String> {
        match (formal, actual) {
            (Type::ParametricBits { signed, width }, actual) => {
                if let Some((sign, count)) = bits(actual) {
                    self.bind_witness(
                        signed,
                        &Type::Bool,
                        IrValue::bool(sign),
                        WitnessType::BitDimension,
                    )?;
                    self.bind_size(width, count, WitnessType::BitDimension)?;
                }
            }
            (
                Type::ParametricArray { element, size },
                Type::Array {
                    element: actual,
                    size: count,
                },
            ) => {
                self.bind_size(size, *count, WitnessType::Exact)?;
                self.infer(element, actual, structs, resolver)?;
            }
            (
                Type::Array { element, .. },
                Type::Array {
                    element: actual, ..
                },
            ) => self.infer(element, actual, structs, resolver)?,
            (Type::Tuple(fields), Type::Tuple(actuals)) => {
                for (field, actual) in fields.iter().zip(actuals) {
                    self.infer(field, actual, structs, resolver)?;
                }
            }
            (Type::ParametricStruct { name, args }, Type::Named(canonical)) => {
                if let Some(instance) = structs.instances.get(canonical) {
                    if instance.template.name == *name {
                        for (arg, bound) in args.iter().zip(&instance.values) {
                            self.bind_witness(
                                arg,
                                &bound.ty,
                                bound.value.clone(),
                                WitnessType::Exact,
                            )?;
                        }
                    }
                } else if let Some(values) = resolver.struct_parametric_values(name, canonical)? {
                    for (arg, &value) in args.iter().zip(values) {
                        self.bind_witness(
                            arg,
                            &dimension_type(),
                            IrValue::u32(value),
                            WitnessType::Exact,
                        )?;
                    }
                }
            }
            (Type::TypeApplied { name, args }, Type::Named(canonical)) => {
                if let Some(instance) = structs.instances.get(canonical)
                    && instance.template.name == *name
                {
                    for (arg, actual) in args.iter().zip(&instance.types) {
                        self.infer(arg, actual, structs, resolver)?;
                    }
                }
            }
            _ => { /* Concrete constraints are checked against the final signature. */ }
        }
        Ok(())
    }

    /// An array literal provides its length even when its element needs the
    /// formal parameter's width as context. An untyped empty literal needs its
    /// complete array type from context before it can supply a length.
    pub(super) fn literal_lengths(&mut self, pattern: &Type, expr: &Expr) -> Result<(), String> {
        match (pattern, expr) {
            (Type::ParametricArray { size, element }, Expr::Array(items)) => {
                if let Some(first) = items.first() {
                    self.bind_size(size, items.len(), WitnessType::Exact)?;
                    self.literal_lengths(element, first)?;
                }
            }
            (Type::Array { element, .. }, Expr::Array(items)) => {
                if let Some(first) = items.first() {
                    self.literal_lengths(element, first)?;
                }
            }
            (Type::Tuple(fields), Expr::Tuple(items)) => {
                for (field, item) in fields.iter().zip(items) {
                    self.literal_lengths(field, item)?;
                }
            }
            _ => { /* Nonliteral arguments provide their checked type instead. */ }
        }
        Ok(())
    }
}

/// A data operand may already have a type from inference or from map's element.
enum DataArgument<'e> {
    Source(&'e Expr),
    Checked(TypedExpr),
}

/// Earlier explicit formals stay in declaration scope while the caller
/// evaluates each constexpr actual with the type of the next formal.
#[derive(Default)]
pub(super) struct ExplicitScope {
    types: Option<Bindings>,
    names: BTreeSet<String>,
    values: BTreeMap<String, Type>,
    literals: BTreeMap<String, TypedExpr>,
    constexpr_names: BTreeSet<String>,
    constant_pattern_names: BTreeSet<String>,
}

impl ExplicitScope {
    /// Move the lexical maps between successive checks without copying the
    /// types or literals of every preceding formal at each step.
    pub(super) fn exchange(&mut self, checker: &mut BodyChecker<'_, '_>) {
        std::mem::swap(&mut self.types, &mut checker.function_type_bindings);
        std::mem::swap(&mut self.names, &mut checker.names);
        std::mem::swap(&mut self.values, &mut checker.values);
        std::mem::swap(&mut self.literals, &mut checker.function_values);
        std::mem::swap(&mut self.constexpr_names, &mut checker.constexpr_names);
        std::mem::swap(
            &mut self.constant_pattern_names,
            &mut checker.constant_pattern_names,
        );
    }

    fn evaluate<'m, T>(
        &mut self,
        caller: &mut BodyChecker<'_, 'm>,
        f: &'m Function,
        preceding_value: Option<(&str, &BoundParametricValue)>,
        evaluate: impl FnOnce(&mut BodyChecker<'_, 'm>) -> Result<T, String>,
    ) -> Result<T, String> {
        caller.with_value_definition(f, |definition| {
            self.exchange(definition);
            if let Some((name, value)) = preceding_value {
                definition.bind_parametric_value(name, value);
            }
            let result = evaluate(definition);
            self.exchange(definition);
            result
        })
    }
}

impl<'a, 'm> BodyChecker<'a, 'm> {
    /// Apply the selected parametric signature to unsized expressions and
    /// materialized references in the caller's lexical scope.
    pub(super) fn with_parametric_context<T>(
        &mut self,
        check: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let saved = std::mem::replace(&mut self.parametric_argument_context, true);
        let result = self.with_dslx_value_context(check);
        self.parametric_argument_context = saved;
        result
    }

    pub(super) fn bind_parametric_value(&mut self, name: &str, bound: &BoundParametricValue) {
        self.names.insert(name.to_owned());
        self.constant_pattern_names.insert(name.to_owned());
        self.values.insert(name.to_owned(), bound.ty.clone());
        self.constexpr_names.insert(name.to_owned());
        self.function_values.insert(
            name.to_owned(),
            TypedExpr {
                ty: bound.ty.clone(),
                kind: K::Literal(bound.value.clone()),
            },
        );
    }

    /// Evaluate in the active lexical scope with the binding's declared type.
    pub(super) fn parametric_value(&mut self, expr: &Expr, ty: &Type) -> Result<IrValue, String> {
        let retain = self.assert_operands.is_some();
        let (constant, operand) = self.parametric_value_and_operand(expr, ty, retain)?;
        self.record_assert_operand(operand);
        Ok(constant)
    }

    /// Fold an actual while optionally keeping its typed expression and
    /// constexpr-only lexical lets for an operand erased during lowering.
    pub(super) fn parametric_value_and_operand(
        &mut self,
        expr: &Expr,
        ty: &Type,
        retain_operand: bool,
    ) -> Result<(IrValue, TypedExpr), String> {
        let had_functions = self.constant_functions.is_some();
        let first = self.statements.len();
        let callees = std::mem::take(&mut self.local_callees);
        let outer_erased = self.in_erased_operand;
        self.in_erased_operand |= retain_operand;
        self.enable_constant_functions();
        let result = self.with_parametric_context(|checker| {
            let value = checker.dslx_argument(expr, ty)?;
            let constant = checker
                .checked_constant_for(&value, checker.dslx_constant_use)?
                .ok_or_else(|| "value parametric requires a compile-time constant".to_owned())?;
            Ok((constant, value))
        });
        // Scope elaboration may materialize lets while computing an actual
        // or dimension. Their values have been consumed at compile time.
        if !retain_operand || result.is_err() {
            self.discard_temporary_statements(first);
        }
        self.in_erased_operand = outer_erased;
        self.local_callees = callees;
        if !had_functions {
            self.constant_functions = None;
        }
        result
    }

    /// A directly named struct value parameter can supply a dimension at its
    /// declared width. Computed dimensions use the usual bool or u32 context.
    fn struct_dimension(&self, expr: &Expr) -> Result<Option<usize>, String> {
        self.direct_struct_binding(expr)
            .map(|(ty, value)| nonnegative_dimension(ty, value))
            .transpose()
    }

    pub(super) fn resolve_dimensions(
        &mut self,
        ty: &Type,
        remaining: usize,
    ) -> Result<Type, String> {
        self.resolve_annotation(ty, remaining, &mut None)
    }

    /// Substitute each written type parameter once, at its source reference.
    /// Already concrete names inside an alias or another argument retain their
    /// original identity even if a formal has the same spelling.
    pub(super) fn concrete_annotation(&mut self, ty: &Type) -> Result<Type, String> {
        let mut copies = Some(TypeArgumentCopies::default());
        self.resolve_annotation(ty, MAX_ELABORATION_DEPTH, &mut copies)
    }

    /// Resolve dimensions and aliases, keeping formals symbolic when checking
    /// an inference pattern. Concrete annotations budget all copied formals
    /// together before cloning their bound types.
    fn resolve_annotation(
        &mut self,
        ty: &Type,
        remaining: usize,
        copies: &mut Option<TypeArgumentCopies>,
    ) -> Result<Type, String> {
        let size = |checker: &mut Self, expr: &Expr| {
            if let Some(n) = checker.struct_dimension(expr)? {
                return Ok(n);
            }
            let value = checker.parametric_value(expr, &dimension_type())?;
            ir_bits_to_usize(value.as_bits().map_err(|e| e.to_string())?)
                .ok_or_else(|| "dimension does not fit usize".to_owned())
        };
        match ty {
            Type::Named(name) if self.local_type_aliases.contains_key(name) => {
                let alias = &self.local_type_aliases[name].ty;
                self.charge_reference_copy(super::source_type_nodes(alias))?;
                Ok(alias.clone())
            }
            Type::TypeApplied { name, args } => {
                if self.has_local_name(name)
                    || self
                        .callback_indices()
                        .is_some_and(|indices| indices.contains_key(name))
                {
                    return Err(format!("'{name}' names a local value here"));
                }
                self.reject_type_parameter_value(name)?;
                self.decls.check_type_name_visibility(
                    name,
                    self.function_position.or(self.block_position),
                )?;
                let template = self
                    .structs
                    .template(name)
                    .ok_or_else(|| format!("unknown type-parametric struct '{name}'"))?;
                super::struct_parametrics::check_type_arity(template, args.len())?;
                let budget = child_type_budget(remaining)?;
                let args = args
                    .iter()
                    .map(|arg| self.resolve_annotation(arg, budget, copies))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_type_resolver(|types, dimensions, bindings| {
                    let bindings = if copies.is_some() { None } else { bindings };
                    types.resolve_struct_type_application(
                        name, args, bindings, dimensions, remaining,
                    )
                })
            }
            Type::ParametricStruct { name, args } => {
                let args = self.struct_value_prefix(name, StructLiteralArgs::Expressions(args))?;
                self.struct_value_type(name, &args, remaining)
            }
            Type::ParametricBits { signed, width } => {
                let signed = match self.struct_dimension(signed)? {
                    Some(0) => false,
                    Some(1) => true,
                    Some(n) => return Err(format!("bit signedness must be 0 or 1, got {n}")),
                    None => self
                        .parametric_value(signed, &Type::Bool)?
                        .to_bool()
                        .map_err(|e| e.to_string())?,
                };
                Ok(Type::Bits {
                    signed,
                    width: size(self, width)?,
                })
            }
            Type::ParametricArray {
                element,
                size: expr,
            } => Ok(Type::array(
                self.resolve_annotation(element, child_type_budget(remaining)?, copies)?,
                size(self, expr)?,
            )),
            Type::Array { element, size } => Ok(Type::array(
                self.resolve_annotation(element, child_type_budget(remaining)?, copies)?,
                *size,
            )),
            Type::Tuple(fields) => {
                let depth = child_type_budget(remaining)?;
                fields
                    .iter()
                    .map(|t| self.resolve_annotation(t, depth, copies))
                    .collect::<Result<_, _>>()
                    .map(Type::Tuple)
            }
            _ => {
                let callbacks = self.callback_indices();
                let resolved = resolve_type(
                    ty,
                    &|name| {
                        self.names.contains(name)
                            || self
                                .enclosing_type_locals
                                .is_some_and(|contains| contains(name))
                            || callbacks.is_some_and(|indices| indices.contains_key(name))
                    },
                    self.function_type_bindings
                        .as_ref()
                        .or_else(|| self.specialization.map(|s| &s.bindings)),
                    self.decls.dimensions_at(
                        self.function_position.or(self.block_position),
                        &self.zero_binding,
                        self.warning_owner,
                    ),
                    &mut TypeResolver {
                        resolver: self.resolver,
                        structs: self.structs,
                        functions: self.function_registry.as_deref_mut(),
                    },
                    remaining,
                )?;
                if let Some(copies) = copies
                    && let Type::Named(name) = ty
                    && let Some(bindings) = self.type_bindings()
                    && let Some(shape) = bindings.shape(name)
                {
                    let bound = bindings
                        .get(name)
                        .ok_or_else(|| format!("cannot infer type parameter '{name}'"))?;
                    copies.nodes = copies.nodes.saturating_add(shape.source_nodes);
                    copies.name_bytes = copies.name_bytes.saturating_add(shape.name_bytes);
                    TypeShape::check_source_budget(copies.nodes, copies.name_bytes)?;
                    return Ok(bound.clone());
                }
                Ok(resolved)
            }
        }
    }

    pub(super) fn value_function_call(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        args: &[Expr],
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        for arg in args {
            crate::infer_literal::check_argument_array_sizes(arg)?;
        }
        self.with_parametric_context(|checker| {
            checker.with_scope_aliases(|checker| {
                // Keep branch-local and aggregate-member bindings visible
                // through both witness collection and application of the final
                // signature.
                let prepared = args
                    .iter()
                    .map(|expr| checker.prepare_value(expr))
                    .collect::<Result<Vec<_>, _>>()?;
                let args = prepared
                    .iter()
                    .map(|expr| DataArgument::Source(expr.as_ref()))
                    .collect();
                checker.instantiate_value_function(f, actuals, args, expected)
            })
        })
    }

    pub(super) fn value_map_call_with_actuals(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        element: TypedExpr,
    ) -> Result<TypedExpr, String> {
        if !f.callback_params.is_empty() || actuals.len() > f.value_parametrics.len() {
            return Err(format!(
                "invalid parametric arguments for map callback '{}'",
                f.name
            ));
        }
        let (ty, kind) = self.instantiate_value_function(
            f,
            actuals,
            vec![DataArgument::Checked(element)],
            None,
        )?;
        Ok(TypedExpr { ty, kind })
    }

    /// Evaluate a declaration's annotations and defaults with its own imports
    /// and earlier module declarations, even when called from a different
    /// scope.
    fn with_value_definition<T>(
        &mut self,
        f: &'m Function,
        evaluate: impl FnOnce(&mut BodyChecker<'_, 'm>) -> Result<T, String>,
    ) -> Result<T, String> {
        let registry = self.function_registry.as_deref_mut().ok_or_else(|| {
            format!(
                "cannot specialize function '{}' in this type annotation",
                f.name
            )
        })?;
        registry.enter(&f.name)?;
        let result = stacker::maybe_grow(1024 * 1024, 2 * 1024 * 1024, || {
            let mut definition =
                BodyChecker::new(self.decls, self.resolver, self.structs, BodyKind::Function);
            definition.function_position = self
                .decls
                .positions
                .get(f.name.as_str())
                .map(|at| at.saturating_sub(1));
            definition.warning_owner = Some(&f.name);
            definition.zero_binding = self.decls.zero_binding(&f.name);
            definition.function_registry = Some(registry);
            evaluate(&mut definition)
        });
        registry.leave();
        result
    }

    /// Bind defaults in declaration order, then resolve each runtime parameter
    /// and the result in the completed value environment.
    fn bound_value_signature(
        &mut self,
        f: &'m Function,
        provided: &[Option<BoundParametricValue>],
    ) -> Result<BoundValueSignature, String> {
        let mut values = Vec::with_capacity(f.value_parametrics.len());
        let mut callbacks = Vec::with_capacity(f.callback_params.len());
        // One inferred tuple can expand at every formal. Bound the retained
        // signature incrementally, before resolving the next annotation.
        let charge = |checker: &mut Self, work| {
            checker
                .function_registry
                .as_deref_mut()
                .expect("module function registry")
                .charge_bindings(work)
        };
        for formal in f.parametric_bindings() {
            let formal = match formal {
                FunctionParametricRef::Value(formal) => formal,
                FunctionParametricRef::Callback(formal) => {
                    let mut binding_work = 0;
                    let mut resolve = |source: &Type| -> Result<Type, String> {
                        let ty = self.source_type(source)?;
                        let nodes = source_type_nodes(&ty);
                        charge(self, nodes)?;
                        binding_work += nodes;
                        Ok(ty)
                    };
                    let params = formal
                        .params
                        .iter()
                        .map(&mut resolve)
                        .collect::<Result<Vec<_>, _>>()?;
                    let result = resolve(&formal.result)?;
                    callbacks.push(CallbackFormal {
                        signature: CallbackParam {
                            name: formal.name.clone(),
                            params,
                            result,
                        },
                        binding_work,
                    });
                    // Callback targets are selected after this signature has
                    // been resolved. The formal still hides
                    // the outer declaration.
                    self.names.insert(formal.name.clone());
                    continue;
                }
            };
            let supplied = &provided[values.len()];
            let ty = self.value_binding_type(formal)?;
            self.names.insert(formal.name.clone());
            let bound = match (supplied, &formal.default) {
                (Some(bound), default) => {
                    if !compatible(&ty, &bound.ty) {
                        return Err(format!(
                            "parametric '{}' expects {ty}, got {}",
                            formal.name, bound.ty
                        ));
                    }
                    if let Some(expr) = default {
                        // An overridden default still has to typecheck. The
                        // formal has its supplied value inside its own default.
                        self.bind_parametric_value(&formal.name, bound);
                        self.with_parametric_context(|c| c.dslx_argument(expr, &ty))
                            .map_err(|e| format!("default for '{}': {e}", formal.name))?;
                    }
                    BoundParametricValue {
                        ty,
                        value: bound.value.clone(),
                    }
                }
                (None, Some(expr)) => {
                    let value = self
                        .parametric_value(expr, &ty)
                        .map_err(|e| format!("default for '{}': {e}", formal.name))?;
                    BoundParametricValue { ty, value }
                }
                (None, None) => {
                    return Err(format!(
                        "cannot infer value parametric '{}' of '{}'",
                        formal.name, f.name
                    ));
                }
            };
            charge(self, 1 + literal_copy_work(&bound.value))?;
            self.bind_parametric_value(&formal.name, &bound);
            values.push(bound);
        }
        let mut params = Vec::new();
        for p in &f.params {
            let ty = self.source_type(&p.ty)?;
            charge(self, source_type_nodes(&ty))?;
            self.names.insert(p.name.clone());
            params.push(Param {
                name: p.name.clone(),
                ty,
            });
        }
        let result = self.source_type(&f.result)?;
        Ok(BoundValueSignature {
            values,
            type_bindings: self
                .function_type_bindings
                .take()
                .unwrap_or_else(|| Bindings::new(&[])),
            interface: FunctionInterface {
                params,
                result,
                callback_params: callbacks,
                callback_indices: self.decls.value_template_callbacks[f.name.as_str()].clone(),
                local: true,
            },
        })
    }

    /// An inferred annotation on a constexpr formal must select one of the
    /// value types accepted for an explicitly annotated binding.
    fn value_binding_type(&mut self, formal: &ParametricBinding) -> Result<Type, String> {
        let ty = self.source_type(&formal.ty)?;
        if let Type::Named(name) = &formal.ty
            && self
                .type_bindings()
                .is_some_and(|bindings| bindings.declares(name))
            && !matches!(
                ty,
                Type::Bits { .. } | Type::Bool | Type::Enum { .. } | Type::Named(_)
            )
        {
            return Err(format!(
                "value parametric '{}' requires a bits, bool, enum, or struct type, got {ty}",
                formal.name
            ));
        }
        Ok(ty)
    }

    /// Explicit actuals and input types come from the caller. Defaults and
    /// formal annotations are evaluated in the function's definition scope.
    fn specialized_signature(
        &mut self,
        f: &'m Function,
        types: Bindings,
        values: Vec<Option<BoundParametricValue>>,
    ) -> Result<Rc<BoundValueSignature>, String> {
        let supplied = SuppliedParametrics {
            types: types.arguments(&f.type_params)?,
            values,
        };
        let registry = self
            .function_registry
            .as_deref_mut()
            .expect("module function registry");
        if let Some(bound) = registry.cached_value_signature(f, &supplied)? {
            return Ok(bound);
        }
        let signature = self.with_value_definition(f, |definition| {
            definition.function_type_bindings = (!f.type_params.is_empty()).then_some(types);
            definition.bound_value_signature(f, &supplied.values)
        })?;
        self.function_registry
            .as_deref_mut()
            .expect("module function registry")
            .cache_value_signature(f, supplied, signature)
    }

    /// An unsized scalar in a tuple or struct supplies its minimum width.
    /// An unsized scalar passed directly to a function can wait for context.
    fn dimension_witness(
        &mut self,
        formal: &Type,
        expr: &Expr,
        inferred: &mut InferredBindings<'_>,
        tuple_member: bool,
    ) -> Result<Option<TypedExpr>, String> {
        if !inferred.needs_witness(formal) {
            return Ok(None);
        }
        let value = if self.contains_type_struct_constructor(expr) {
            let Some(value) = self.constructor_dimension_witness(expr)? else {
                return Ok(None);
            };
            value
        } else if tuple_member {
            self.inferred_expr(expr)?
        } else if minimum_type(expr)?.is_none() {
            self.expr(expr, None)?
        } else {
            return Ok(None);
        };
        inferred.infer(formal, &value.ty, self.structs, self.resolver)?;
        Ok(Some(value))
    }

    /// Tuple fields have independent widths. Checking a field that supplies a
    /// dimension must leave the other fields free to use their inferred types.
    pub(super) fn infer_type_input<'e>(
        &mut self,
        formal: &Type,
        expr: &'e Expr,
        types: &mut Bindings,
        inferred: &mut InferredBindings<'_>,
        tuple_member: bool,
    ) -> Result<Input<'e>, String> {
        if inferred.needs_witness(formal)
            && let (Type::Tuple(fields), Expr::Tuple(items)) = (formal, expr)
            && fields.len() == items.len()
        {
            let members = fields
                .iter()
                .zip(items)
                .map(|(ty, item)| {
                    self.infer_type_input(ty, item, types, inferred, /* tuple_member= */ true)
                })
                .collect::<Result<_, _>>()?;
            return Ok(Input::Aggregate {
                expr,
                members,
                charged_fill: 0,
            });
        }
        let witness = self.dimension_witness(formal, expr, inferred, tuple_member)?;
        self.infer_input(formal, expr, types, InputPattern::Function(witness))
    }

    /// Gather dimension witnesses from the same field across array rows or
    /// conditional arms. Other fields receive the completed signature later.
    fn dimension_input_group<'e>(
        &mut self,
        formal: &Type,
        exprs: &[&'e Expr],
        inferred: &mut InferredBindings<'_>,
        aggregate_member: bool,
    ) -> Result<Vec<Input<'e>>, String> {
        if exprs.is_empty() || !inferred.needs_witness(formal) {
            return Ok(exprs
                .iter()
                .map(|&expr| Input::Contextual {
                    expr,
                    prepared: None,
                })
                .collect());
        }
        if exprs
            .iter()
            .any(|expr| matches!(expr, Expr::If { .. } | Expr::Match { .. }))
            && exprs.iter().any(|expr| super::has_aggregate_literal(expr))
        {
            let (selectors, branches) = self.result_branches(exprs)?;
            let mut inputs = self
                .dimension_input_group(formal, &branches, inferred, aggregate_member)?
                .into_iter();
            return Ok(exprs
                .iter()
                .zip(selectors)
                .map(|(&expr, selector)| {
                    let count = match expr {
                        Expr::If { .. } => 2,
                        Expr::Match { arms, .. } => arms.len(),
                        _ => return inputs.next().expect("one input for an unbranched operand"),
                    };
                    Input::Branches {
                        expr,
                        selector: Some(selector),
                        results: inputs.by_ref().take(count).collect(),
                    }
                })
                .collect());
        }
        if let Type::Tuple(fields) = formal
            && let Some(rows) = exprs
                .iter()
                .map(|expr| match expr {
                    Expr::Tuple(items) if items.len() == fields.len() => Some(items),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        {
            let mut members: Vec<Vec<Input>> = rows
                .iter()
                .map(|_| Vec::with_capacity(fields.len()))
                .collect();
            for (index, field) in fields.iter().enumerate() {
                let column = rows.iter().map(|row| &row[index]).collect::<Vec<_>>();
                let inputs = self.dimension_input_group(
                    field, &column, inferred, /* aggregate_member= */ true,
                )?;
                for (row, input) in members.iter_mut().zip(inputs) {
                    row.push(input);
                }
            }
            return Ok(exprs
                .iter()
                .zip(members)
                .map(|(&expr, members)| Input::Aggregate {
                    expr,
                    members,
                    charged_fill: 0,
                })
                .collect());
        }
        if let Type::Array { element, .. } | Type::ParametricArray { element, .. } = formal
            && let Some(rows) = exprs
                .iter()
                .map(|expr| expr.unannotated_array())
                .collect::<Option<Vec<_>>>()
        {
            if let Type::ParametricArray { size, .. } = formal {
                for (row, fill) in &rows {
                    if !fill && !row.is_empty() {
                        inferred.bind_size(size, row.len(), WitnessType::Exact)?;
                    }
                }
            }
            let column = rows
                .iter()
                .flat_map(|(items, _)| items.iter())
                .collect::<Vec<_>>();
            let mut inputs = self
                .dimension_input_group(
                    element, &column, inferred, /* aggregate_member= */ true,
                )?
                .into_iter();
            return Ok(exprs
                .iter()
                .zip(rows)
                .map(|(&expr, (row, _))| {
                    if row.is_empty() {
                        // The complete element type is needed to materialize an
                        // empty array.
                        Input::Contextual {
                            expr,
                            prepared: None,
                        }
                    } else {
                        Input::Aggregate {
                            expr,
                            members: inputs.by_ref().take(row.len()).collect(),
                            charged_fill: 0,
                        }
                    }
                })
                .collect());
        }
        let mut needs_sibling = true;
        for expr in exprs {
            if !crate::infer_literal::requires_sibling_type(
                expr,
                self.common_widths(),
                &|name, count| self.call_needs_result(name, count),
            )? {
                needs_sibling = false;
                break;
            }
        }
        if needs_sibling {
            return Ok(exprs
                .iter()
                .map(|&expr| Input::Contextual {
                    expr,
                    prepared: None,
                })
                .collect());
        }
        if let [expr] = exprs {
            let witness = self.dimension_witness(formal, expr, inferred, aggregate_member)?;
            return Ok(vec![Input::Contextual {
                expr,
                prepared: witness,
            }]);
        }
        let witnesses = self.infer_constructors(exprs, /* infer_literal_widths= */ true)?;
        let mut witnesses = witnesses.into_iter().flatten().peekable();
        if let Some(value) = witnesses.peek() {
            // Sibling checking has given the entire column one type.
            inferred.infer(formal, &value.ty, self.structs, self.resolver)?;
        }
        Ok(exprs
            .iter()
            .map(|&expr| Input::Contextual {
                expr,
                prepared: witnesses.next(),
            })
            .collect())
    }

    /// Check each data argument for witnesses in source order, retaining its
    /// checked leaves for application of the complete function signature.
    fn infer_function_inputs<'e>(
        &mut self,
        f: &Function,
        args: Vec<DataArgument<'e>>,
        types: &mut Bindings,
        inferred: &mut InferredBindings<'_>,
    ) -> Result<Vec<Input<'e>>, String> {
        let mut inputs = Vec::with_capacity(args.len());
        for (arg, param) in args.into_iter().zip(&f.params) {
            let input = match arg {
                DataArgument::Source(expr) => {
                    inferred.literal_lengths(&param.ty, expr)?;
                    let mut input = self
                        .dimension_input_group(
                            &param.ty,
                            &[expr],
                            inferred,
                            /* aggregate_member= */ false,
                        )?
                        .pop()
                        .expect("one input per argument");
                    if !f.type_params.is_empty() {
                        // Dimension witnesses share a column; type parameters
                        // bind in the original order of the caller's rows and
                        // fields.
                        input.visit_leaves(&param.ty, &mut |formal, leaf| {
                            if let Input::Contextual { expr, prepared } = leaf {
                                *leaf = self.infer_input(
                                    formal,
                                    expr,
                                    types,
                                    InputPattern::Function(prepared.take()),
                                )?;
                            }
                            Ok(())
                        })?;
                    }
                    input
                }
                DataArgument::Checked(value) => {
                    types.infer_source_type(
                        &param.ty,
                        &value.ty,
                        self.structs,
                        /* contextual_projection= */ false,
                    )?;
                    inferred.infer(&param.ty, &value.ty, self.structs, self.resolver)?;
                    Input::Checked { expr: None, value }
                }
            };
            inferred.finish_argument();
            inputs.push(input);
        }
        Ok(inputs)
    }

    /// Apply the bound signature to remaining operands and reuse the witnesses.
    fn finish_function_inputs(
        &mut self,
        f: &Function,
        inputs: Vec<Input>,
        params: &[Param],
    ) -> Result<Vec<TypedExpr>, String> {
        let mut checked = Vec::with_capacity(inputs.len());
        for (input, param) in inputs.into_iter().zip(params) {
            if let Some(expr) = input.source() {
                super::check_argument_widths(
                    expr,
                    &param.ty,
                    Some(&self.inferred_bindings),
                    Some(&|| self.zero_calls_function()),
                )?;
            }
            checked
                .push(self.finish_input(input, &param.ty).map_err(|e| {
                    format!("argument '{}', call to '{}': {e}", param.name, f.name)
                })?);
        }
        Ok(checked)
    }

    /// Infer type variables from the result context, data and callbacks before
    /// checking contextual arguments and the specialized body.
    fn instantiate_type_function(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        args: Vec<DataArgument<'_>>,
        expected: Option<&Type>,
        mut inferred: InferredBindings<'_>,
    ) -> Result<(Type, K), String> {
        let mut types = Bindings::new(&f.type_params);
        if let Some(result) = expected {
            types.infer_source_type(
                &f.result,
                result,
                self.structs,
                /* contextual_projection= */ false,
            )?;
        }
        let mut inputs = self.infer_function_inputs(f, args, &mut types, &mut inferred)?;
        self.infer_input_groups(&mut inputs, &f.params, &mut types, &format!("'{}'", f.name))?;
        let (types, targets) = self.infer_explicit_formals(f, types, actuals, &mut inferred)?;
        let bound = self.specialized_signature(f, types, inferred.values)?;
        let checked = self.finish_function_inputs(f, inputs, &bound.interface.params)?;
        let result = bound.interface.result.clone();
        let name = self.intern_function(f, targets, Some(bound))?;
        Ok((
            result,
            K::Call {
                name,
                args: checked,
            },
        ))
    }

    /// Visit explicit formals in order. A callback's return can determine
    /// later callback inputs, constexpr annotations, and the function result.
    fn infer_explicit_formals(
        &mut self,
        f: &'m Function,
        types: Bindings,
        actuals: &[Expr],
        inferred: &mut InferredBindings<'_>,
    ) -> Result<(Bindings, Vec<CallbackTarget<'m>>), String> {
        if actuals.is_empty() {
            return Ok((types, Vec::new()));
        }
        let mut environment = self
            .function_registry
            .as_deref_mut()
            .expect("module function registry")
            .explicit_environment(ExplicitEnvironment::DataTypes(
                &f.name,
                types.partial_arguments(),
            ))?;
        let mut scope = ExplicitScope {
            types: Some(types),
            ..Default::default()
        };
        let mut targets = Vec::with_capacity(f.callback_params.len());
        let mut preceding_value: Option<(&str, &BoundParametricValue)> = None;
        let mut explicit_values = inferred.values.iter_mut();
        for (index, (formal, actual)) in f.parametric_bindings().zip(actuals).enumerate() {
            let prior = preceding_value.take();
            match formal {
                FunctionParametricRef::Value(value) => {
                    let ty = scope.evaluate(self, f, prior, |d| {
                        if let Some(ty) = d
                            .function_registry
                            .as_deref()
                            .expect("module function registry")
                            .explicit_value_type(environment)
                        {
                            return Ok(ty);
                        }
                        let ty = d.value_binding_type(value)?;
                        d.function_registry
                            .as_deref_mut()
                            .expect("module function registry")
                            .record_explicit_value_type(environment, &ty);
                        Ok(ty)
                    })?;
                    let actual = self
                        .parametric_value(actual, &ty)
                        .map_err(|e| format!("parametric '{}': {e}", value.name))?;
                    let slot = explicit_values
                        .next()
                        .expect("value formal in declaration order");
                    let bound = slot.insert(BoundParametricValue { ty, value: actual });
                    if index + 1 < actuals.len() {
                        environment = self
                            .function_registry
                            .as_deref_mut()
                            .expect("module function registry")
                            .explicit_environment(ExplicitEnvironment::Value(
                                environment,
                                bound.clone(),
                            ))?;
                    }
                    preceding_value = Some((&value.name, bound));
                }
                FunctionParametricRef::Callback(callback) => {
                    let name = callback_argument_name(actual)?;
                    let source = self.callback_source(name, |head| self.has_local_name(head))?;
                    environment = self
                        .function_registry
                        .as_deref_mut()
                        .expect("module function registry")
                        .explicit_environment(ExplicitEnvironment::Callback(
                            environment,
                            source.clone(),
                        ))?;
                    let target = scope
                        .evaluate(self, f, prior, |d| {
                            d.infer_callback_result(callback, &source, environment)
                        })
                        .map_err(|e| {
                            format!(
                                "callable parameter '{}' of '{}': {e}",
                                callback.name, f.name
                            )
                        })?;
                    targets.push(target);
                }
            }
        }
        Ok((scope.types.expect("generic function bindings"), targets))
    }

    /// A callback's concrete result binds variables in its return pattern
    /// after its input annotations resolve in the template's lexical scope.
    fn infer_callback_result(
        &mut self,
        callback: &CallbackParam,
        source: &CallbackSource<'m>,
        environment: usize,
    ) -> Result<CallbackTarget<'m>, String> {
        let bound = match self
            .function_registry
            .as_deref()
            .expect("module function registry")
            .inferred_callback(environment)
        {
            Some(bound) => bound,
            None => {
                let bound = Rc::new(self.validate_inferred_callback(callback, source)?);
                self.function_registry
                    .as_deref_mut()
                    .expect("module function registry")
                    .record_inferred_callback(environment, Rc::clone(&bound));
                bound
            }
        };
        let registry = self
            .function_registry
            .as_deref_mut()
            .expect("module function registry");
        self.function_type_bindings
            .as_mut()
            .expect("generic function bindings")
            .merge_callback_results(&bound.results, |nodes| registry.charge_bindings(nodes))?;
        self.names.insert(callback.name.clone());
        Ok(bound.target)
    }

    /// Resolve the dependent annotations once and retain only the bindings
    /// contributed by the callback's result.
    fn validate_inferred_callback(
        &mut self,
        callback: &CallbackParam,
        source: &CallbackSource<'m>,
    ) -> Result<InferredCallback<'m>, String> {
        let mut params = Vec::with_capacity(callback.params.len());
        for ty in &callback.params {
            let ty = self.source_type(ty)?;
            self.charge_expansion("callback bindings", source_type_nodes(&ty))?;
            params.push(ty);
        }
        let pattern = self.resolve_dimensions(&callback.result, MAX_ELABORATION_DEPTH)?;
        self.structs.update_imports(self.resolver)?;
        self.structs
            .formal_shape(&pattern, MAX_ELABORATION_DEPTH, self.type_bindings())?;
        let input_nodes = if matches!(source, CallbackSource::Imported { .. }) {
            Some(params.iter().try_fold(0usize, |sum, ty| {
                self.structs
                    .type_shape(ty, MAX_ELABORATION_DEPTH)
                    .map(|shape| sum.saturating_add(shape.ir_nodes))
            })?)
        } else {
            None
        };
        let (target, result) = self.callback_result(&params, source, input_nodes)?;
        self.charge_expansion("callback bindings", source_type_nodes(&result))?;
        let bindings = self
            .function_type_bindings
            .as_ref()
            .expect("generic function bindings");
        let mut results = Bindings::new(&bindings.referenced_parameters([&pattern]));
        results.match_input(&pattern, &result, self.structs)?;
        Ok(InferredCallback { target, results })
    }

    /// Bind a local template to its actuals and emit a call to its concrete
    /// symbol.
    fn instantiate_value_function(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        args: Vec<DataArgument<'_>>,
        expected: Option<&Type>,
    ) -> Result<(Type, K), String> {
        if self.function_registry.is_none() {
            return Err(format!(
                "cannot specialize function '{}' in this type annotation",
                f.name
            ));
        }
        if args.len() != f.params.len() {
            return Err(format!(
                "'{}' expects {} arguments, got {}",
                f.name,
                f.params.len(),
                args.len()
            ));
        }
        let mut inferred = InferredBindings::new(&f.value_parametrics, actuals.len());
        if let Some(result) = expected {
            inferred.infer(&f.result, result, self.structs, self.resolver)?;
            inferred.finish_argument();
        }
        if !f.type_params.is_empty() {
            return self.instantiate_type_function(f, actuals, args, expected, inferred);
        }
        if !actuals.is_empty() {
            // Check explicit annotations before instantiating the callee.
            // Preceding formals hide module declarations even though none of
            // their values are available yet in this definition scope.
            let types = self.with_value_definition(f, |definition| {
                let mut types = Vec::new();
                for formal in f.parametric_bindings().take(actuals.len()) {
                    let name = match formal {
                        FunctionParametricRef::Value(value) => {
                            types.push(definition.source_type(&value.ty)?);
                            &value.name
                        }
                        FunctionParametricRef::Callback(callback) => &callback.name,
                    };
                    definition.names.insert(name.clone());
                }
                Ok(types)
            })?;
            for (i, (formal, ty)) in f.value_parametrics.iter().zip(types).enumerate() {
                let value = self
                    .parametric_value(&actuals[formal.index], &ty)
                    .map_err(|e| format!("parametric '{}': {e}", formal.name))?;
                inferred.values[i] = Some(BoundParametricValue { ty, value });
            }
        }
        let mut types = Bindings::new(&[]);
        let inputs = self.infer_function_inputs(f, args, &mut types, &mut inferred)?;
        let bound = self.specialized_signature(f, types, inferred.values)?;
        let interface = &bound.interface;
        let mut targets = Vec::new();
        for (i, formal) in f.parametric_bindings().enumerate() {
            if let FunctionParametricRef::Callback(_) = formal {
                let actual = actuals
                    .get(i)
                    .ok_or_else(|| format!("'{}' requires explicit callable arguments", f.name))?;
                let name = super::generic_blocks::callback_argument_name(actual)?;
                let source = self.callback_source(name, |head| self.has_local_name(head))?;
                let formal = &interface.callback_params[targets.len()];
                if self.replaying_body() {
                    self.charge_expansion("callback bindings", formal.binding_work)?;
                }
                let signature = &formal.signature;
                let (target, result) = self.callback_result(&signature.params, &source, None)?;
                if !compatible(&result, &signature.result) {
                    return Err(format!(
                        "callback '{name}' returns {result}, expected {}",
                        signature.result
                    ));
                }
                targets.push(target);
            }
        }
        let checked = self.finish_function_inputs(f, inputs, &interface.params)?;
        let result = interface.result.clone();
        let name = self.intern_function(f, targets, Some(bound))?;
        Ok((
            result,
            K::Call {
                name,
                args: checked,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{Bindings, BodyChecker, BodyKind, Declarations, StructTable, Type};
    use crate::check::{FunctionSpecializations, ImportResolver, MAX_TYPE_SIZE};
    use crate::{ImportOptions, check_module, parse_module};
    use xlsynth_pir::IrValue;

    #[test]
    fn repeated_type_formals_stop_signature_construction_at_the_expansion_limit()
    -> Result<(), String> {
        let module = parse_module("fn repeat<T>(a:T,b:T)->T { a }")?;
        let f = &module.functions[0];
        let mut structs = StructTable::default();
        let mut decls = Declarations::default();
        decls.add_value_template(f, &structs)?;
        let mut types = Bindings::new(&f.type_params);
        let tuple = Type::Tuple(vec![Type::Bool, Type::Bool]);
        types.match_input(&f.params[0].ty, &tuple, &structs)?;
        let mut resolver = ImportResolver::new(&[], &ImportOptions::default())?;
        let mut registry = FunctionSpecializations::new(Default::default());
        // The first tuple fits. Resolving the second must stop construction.
        registry.expansion_work = MAX_TYPE_SIZE - 3;
        let mut checker = BodyChecker::new(&decls, &mut resolver, &mut structs, BodyKind::Function);
        checker.function_registry = Some(&mut registry);
        checker.function_type_bindings = Some(types);
        let error = checker.bound_value_signature(f, &[]).err();
        assert_eq!(
            error,
            Some(format!(
                "parametric bindings exceed the function specialization expansion limit of {MAX_TYPE_SIZE}"
            ))
        );
        Ok(())
    }

    #[test]
    fn callback_results_type_later_explicit_values_and_callbacks() -> Result<(), String> {
        let source = r#"
fn widen(x:u8)->u16 { x as u16 }
fn add(a:u16,b:u16)->u16 { a+b }
fn apply<T,R,F:fn(T)->R,V:R,G:fn(R,R)->R>(x:T)->R { G(F(x),V) }
const ANSWER = apply<widen,37,add>(u8:5);
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::make_ubits(16, 42).map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn aggregate_arguments_share_complementary_type_witnesses() -> Result<(), String> {
        let source = r#"
fn pick<T>(a:T,b:T)->T { a }
const ANSWER = pick((u4:1,2), (3,u8:4));
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::parse_typed("(bits[4]:1,bits[8]:2)").map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn generic_functions_preserve_contextual_aggregate_signedness() -> Result<(), String> {
        let source = r#"
fn width<T,N:u32>(t:T,x:sN[N][1])->u32 { N }
fn fixed<T>(t:T,x:(s4,s4))->u32 { u32:4 }
fn mixed<T,N:u32>(x:(T,sN[N][1]))->u32 { N }
fn test()->(u32,u32,u32,u32,u32) {
    let a = [0xf];
    let t = (0xf,0xf);
    let m = (u8:0,[0xf]);
    (width<4>(u8:0,a), fixed(u8:0,t), mixed(m),
     width(u8:0,if true { a } else { a }),
     mixed(match false { true => m, _ => m }))
}
const ANSWER = test();
"#;
        let checked = check_module(&parse_module(source)?)?;
        assert_eq!(
            checked.constants()[0].value,
            IrValue::parse_typed("(bits[32]:4,bits[32]:4,bits[32]:4,bits[32]:4,bits[32]:4)")
                .map_err(|e| e.to_string())?
        );
        Ok(())
    }

    #[test]
    fn generic_function_dimensions_use_each_arguments_original_witness() -> Result<(), String> {
        let source = r#"
fn leading<T,N:u32>(t:T,x:uN[N],xs:uN[N][1])->u32 { N }
fn trailing<T,N:u32>(t:T,xs:uN[N][1],x:uN[N])->u32 { N }
fn nested<T,N:u32>(p:(T,uN[N]),x:uN[N])->u32 { N }
fn tuple_width<T,N:u32>(p:(T,uN[N]))->u32 { N }
"#;
        let accepted =
            format!("{source} const WIDTH = (leading(u8:0,u8:3,[1]),tuple_width((u8:0,1)));");
        assert_eq!(
            check_module(&parse_module(&accepted)?)?.constants()[0].value,
            IrValue::parse_typed("(bits[32]:8,bits[32]:1)").map_err(|e| e.to_string())?
        );
        for call in ["trailing(u8:0,[1],u8:3)", "nested((u8:0,1),u8:3)"] {
            let rejected = format!("{source} const WIDTH = {call};");
            let err = check_module(&parse_module(&rejected)?).unwrap_err();
            assert!(err.contains("expected u1, got u8"), "{call}: {err}");
        }
        Ok(())
    }

    #[test]
    fn value_formals_keep_their_supported_types_after_inference() -> Result<(), String> {
        let source = "fn f<T,V:T>(x:T)->T { V } fn test()->u8[1] { f<{[u8:0]}>([u8:1]) }";
        let error = check_module(&parse_module(source)?).unwrap_err();
        assert!(
            error.contains("requires a bits, bool, enum, or struct type"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn mixed_formals_resolve_callback_types_in_declaration_order() -> Result<(), String> {
        let source = r#"
const N = u32:8;
fn step(x:u8)->u8{x}
fn f<F:fn(uN[N])->uN[N],N:u32>(x:u8)->u8{F(x)}
fn sample(x:u8)->u8{f<step,3>(x)}
"#;
        check_module(&parse_module(source)?)?;
        Ok(())
    }

    #[test]
    fn a_callable_formal_hides_globals_in_defaults_and_dimensions() -> Result<(), String> {
        let definitions = "fn F(x:u32)->u32{x+u32:100} fn step(x:u32)->u32{x+u32:1}";
        for (signature, call) in [
            (
                "fn f<F:fn(u32)->u32,N:u32={F(u32:1)}>()->u32{N}",
                "f<step>()",
            ),
            (
                "fn f<F:fn(u32)->u32,N:u32={F(u32:1)}>()->u32{N}",
                "f<step,2>()",
            ),
            (
                "fn f<F:fn(u32)->u32,N:u32>(x:uN[F(N)])->u32{x as u32}",
                "f<step,1>(uN[101]:1)",
            ),
            (
                "fn f<F:fn(u32)->u32,N:uN[F(u32:1)]>()->u32{N as u32}",
                "f<step,uN[101]:1>()",
            ),
        ] {
            let source = format!("{definitions}\n{signature}\nfn sample()->u32{{{call}}}");
            let error = check_module(&parse_module(&source)?).unwrap_err();
            assert!(error.contains("'F' names a local value"), "{error}");
        }
        Ok(())
    }
}
