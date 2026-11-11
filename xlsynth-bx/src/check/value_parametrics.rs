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
    MAX_ELABORATION_DEPTH, StructTable, TypeResolver, bits, child_type_budget, compatible, declare,
    literal_copy_work, minimum_type, resolve_type, source_type_nodes,
};
use crate::ast::{
    CallbackParam, Expr, Function, FunctionParametricRef, Param, ParametricBinding, Type,
};
use crate::dslx_import::ImportResolver;
use crate::typed::{BoundParametricValue, ConstantUse, TypedExpr};

pub(super) fn dimension_type() -> Type {
    Type::Bits {
        signed: false,
        width: 32,
    }
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
        if !f.type_params.is_empty() {
            validate_inference_sources(&f.type_params, &f.params, &f.callback_params)?;
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
        self.value_template_callbacks
            .insert(&f.name, callback_indices);
        self.function_templates.insert(&f.name, f);
        Ok(())
    }
}

/// Only a bare binding in a dimension provides an inference constraint. A
/// computed dimension is checked after its operands have concrete values.
pub(super) struct InferredBindings<'f> {
    formals: &'f [ParametricBinding],
    indices: BTreeMap<&'f str, usize>,
    pub(super) values: Vec<Option<BoundParametricValue>>,
    explicit_count: usize,
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
            witnesses: BTreeMap::new(),
        }
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
                self.missing(size)
                    .is_some_and(|i| !self.witnesses.contains_key(&i))
                    || self.needs_witness(element)
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

    fn bind_dimension(
        &mut self,
        dimension: &Expr,
        ty: &Type,
        value: IrValue,
    ) -> Result<(), String> {
        if let Some(index) = self.missing(dimension) {
            let witness = BoundParametricValue {
                ty: ty.clone(),
                value,
            };
            if let Some(previous) = self.witnesses.insert(index, witness.clone())
                && previous != witness
            {
                return Err(format!(
                    "conflicting values inferred for parametric '{}': {} and {}",
                    self.formals[index].name, previous.value, witness.value
                ));
            }
        }
        Ok(())
    }

    pub(super) fn finish_argument(&mut self) {
        for (index, value) in std::mem::take(&mut self.witnesses) {
            self.values[index] = Some(value);
        }
    }

    fn bind_size(&mut self, dimension: &Expr, size: usize) -> Result<(), String> {
        if self.missing(dimension).is_some() {
            let value = IrValue::make_ubits(32, size as u64).map_err(|e| e.to_string())?;
            self.bind_dimension(dimension, &dimension_type(), value)?;
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
                    self.bind_dimension(signed, &Type::Bool, IrValue::bool(sign))?;
                    self.bind_size(width, count)?;
                }
            }
            (
                Type::ParametricArray { element, size },
                Type::Array {
                    element: actual,
                    size: count,
                },
            ) => {
                self.bind_size(size, *count)?;
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
                let values = if let Some(instance) = structs.instances.get(canonical) {
                    (instance.template.name == *name).then_some(instance.values.as_slice())
                } else {
                    resolver.struct_parametric_values(name, canonical)?
                };
                if let Some(values) = values {
                    for (arg, &value) in args.iter().zip(values) {
                        self.bind_dimension(arg, &dimension_type(), IrValue::u32(value))?;
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
                    self.bind_size(size, items.len())?;
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
struct DataArgument<'e> {
    source: Option<&'e Expr>,
    checked: Option<TypedExpr>,
}

/// Earlier explicit formals stay in declaration scope while the caller
/// evaluates each constexpr actual with the type of the next formal.
#[derive(Default)]
struct ExplicitScope {
    types: Option<Bindings>,
    names: BTreeSet<String>,
    values: BTreeMap<String, Type>,
    literals: BTreeMap<String, TypedExpr>,
    constexpr_names: BTreeSet<String>,
    value_parametric_names: BTreeSet<String>,
}

impl ExplicitScope {
    /// Move the lexical maps between successive checks without copying the
    /// types or literals of every preceding formal at each step.
    fn exchange(&mut self, checker: &mut BodyChecker<'_, '_>) {
        std::mem::swap(&mut self.types, &mut checker.function_type_bindings);
        std::mem::swap(&mut self.names, &mut checker.names);
        std::mem::swap(&mut self.values, &mut checker.values);
        std::mem::swap(&mut self.literals, &mut checker.function_values);
        std::mem::swap(&mut self.constexpr_names, &mut checker.constexpr_names);
        std::mem::swap(
            &mut self.value_parametric_names,
            &mut checker.value_parametric_names,
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
        self.value_parametric_names.insert(name.to_owned());
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
        let had_functions = self.constant_functions.is_some();
        let first = self.statements.len();
        let callees = std::mem::take(&mut self.local_callees);
        self.enable_constant_functions();
        let result = self.with_parametric_context(|checker| {
            let value = checker.dslx_argument(expr, ty)?;
            checker
                .checked_constant_for(&value, ConstantUse::ValueParametric)?
                .ok_or_else(|| "value parametric requires a compile-time constant".to_owned())
        });
        // Scope elaboration may materialize lets while computing an actual
        // or dimension. Their values have been consumed at compile time.
        self.discard_temporary_statements(first);
        self.local_callees = callees;
        if !had_functions {
            self.constant_functions = None;
        }
        result
    }

    pub(super) fn resolve_dimensions(
        &mut self,
        ty: &Type,
        remaining: usize,
    ) -> Result<Type, String> {
        let size = |checker: &mut Self, expr: &Expr| {
            let value = checker.parametric_value(expr, &dimension_type())?;
            ir_bits_to_usize(value.as_bits().map_err(|e| e.to_string())?)
                .ok_or_else(|| "dimension does not fit usize".to_owned())
        };
        match ty {
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
                    .map(|arg| self.resolve_dimensions(arg, budget))
                    .collect::<Result<Vec<_>, _>>()?;
                let ty = Type::TypeApplied {
                    name: name.clone(),
                    args,
                };
                let ty = match self.type_bindings() {
                    Some(bindings) if bindings.has_unbound_variable(&ty) => return Ok(ty),
                    Some(bindings) => bindings.concrete_type(&ty)?,
                    None => ty,
                };
                self.materialize_type_applications(ty, remaining)
            }
            Type::ParametricStruct { name, args } => {
                let ty = self.struct_actual_type(name, args)?;
                self.resolve_dimensions(&ty, remaining)
            }
            Type::ParametricBits { signed, width } => {
                let signed = self
                    .parametric_value(signed, &Type::Bool)?
                    .to_bool()
                    .map_err(|e| e.to_string())?;
                Ok(Type::Bits {
                    signed,
                    width: size(self, width)?,
                })
            }
            Type::ParametricArray {
                element,
                size: expr,
            } => Ok(Type::Array {
                element: Box::new(self.resolve_dimensions(element, child_type_budget(remaining)?)?),
                size: size(self, expr)?,
            }),
            Type::Array { element, size } => Ok(Type::Array {
                element: Box::new(self.resolve_dimensions(element, child_type_budget(remaining)?)?),
                size: *size,
            }),
            Type::Tuple(fields) => {
                let depth = child_type_budget(remaining)?;
                fields
                    .iter()
                    .map(|t| self.resolve_dimensions(t, depth))
                    .collect::<Result<_, _>>()
                    .map(Type::Tuple)
            }
            _ => {
                let callbacks = self.callback_indices();
                resolve_type(
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
                )
            }
        }
    }

    pub(super) fn value_function_call(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        args: &[Expr],
    ) -> Result<(Type, K), String> {
        let instantiate = |checker: &mut Self, args: &[&Expr]| {
            let args = args
                .iter()
                .map(|&expr| DataArgument {
                    source: Some(expr),
                    checked: None,
                })
                .collect();
            checker.instantiate_value_function(f, actuals, args)
        };
        self.with_parametric_context(|checker| {
            if f.type_params.is_empty() {
                let sources: Vec<_> = args.iter().collect();
                return checker.with_scoped_results(&sources, instantiate);
            }
            checker.with_scope_aliases(|checker| {
                // Preserve scoped aggregate members and branch results until
                // both their type evidence and the final context are known.
                let prepared = args
                    .iter()
                    .map(|expr| checker.prepare_value(expr))
                    .collect::<Result<Vec<_>, _>>()?;
                let sources = prepared
                    .iter()
                    .map(|expr| expr.as_ref())
                    .collect::<Vec<_>>();
                instantiate(checker, &sources)
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
            vec![DataArgument {
                source: None,
                checked: Some(element),
            }],
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

    /// Preserve DSLX's source-order dimension inference. An unsized scalar
    /// can wait for later evidence. A constructor supplies the type inferred
    /// for its members when this argument needs to bind a width.
    fn infer_dimension_input(
        &mut self,
        formal: &Type,
        expr: &Expr,
        inferred: &mut InferredBindings<'_>,
    ) -> Result<Option<TypedExpr>, String> {
        inferred.literal_lengths(formal, expr)?;
        self.dimension_witness(formal, expr, inferred, /* tuple_member= */ false)
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
            return Ok(Input::Aggregate { expr, members });
        }
        let witness = self.dimension_witness(formal, expr, inferred, tuple_member)?;
        self.infer_input(formal, expr, types, InputPattern::Function(witness))
    }

    /// Infer type variables from data and callback results before applying
    /// the resulting signature to contextual literals and the specialized body.
    fn instantiate_type_function(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        mut args: Vec<DataArgument<'_>>,
    ) -> Result<(Type, K), String> {
        let mut types = Bindings::new(&f.type_params);
        let mut inferred = InferredBindings::new(&f.value_parametrics, actuals.len());
        let mut inputs = Vec::with_capacity(args.len());
        for (arg, param) in args.iter_mut().zip(&f.params) {
            let input = match (arg.source, arg.checked.take()) {
                (Some(expr), _) => {
                    inferred.literal_lengths(&param.ty, expr)?;
                    self.infer_type_input(
                        &param.ty,
                        expr,
                        &mut types,
                        &mut inferred,
                        /* tuple_member= */ false,
                    )?
                }
                (None, Some(value)) => {
                    types.infer_source_type(
                        &param.ty,
                        &value.ty,
                        self.structs,
                        /* contextual_projection= */ false,
                    )?;
                    inferred.infer(&param.ty, &value.ty, self.structs, self.resolver)?;
                    Input::Checked { expr: None, value }
                }
                (None, None) => unreachable!("map supplies a typed argument"),
            };
            inferred.finish_argument();
            inputs.push(input);
        }
        self.infer_input_groups(&mut inputs, &f.params, &mut types, &format!("'{}'", f.name))?;
        let (types, targets) = self.infer_explicit_formals(f, types, actuals, &mut inferred)?;
        let bound = self.specialized_signature(f, types, inferred.values)?;
        let mut checked = Vec::with_capacity(inputs.len());
        for ((input, arg), param) in inputs.into_iter().zip(args).zip(&bound.interface.params) {
            if let Some(expr) = arg.source {
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
        mut args: Vec<DataArgument<'_>>,
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
        if !f.type_params.is_empty() {
            return self.instantiate_type_function(f, actuals, args);
        }
        let mut inferred = InferredBindings::new(&f.value_parametrics, actuals.len());
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
        for (arg, param) in args.iter_mut().zip(&f.params) {
            if let Some(expr) = arg.source {
                arg.checked = self.infer_dimension_input(&param.ty, expr, &mut inferred)?;
            } else if let Some(value) = &arg.checked {
                inferred.infer(&param.ty, &value.ty, self.structs, self.resolver)?;
            }
            inferred.finish_argument();
        }
        let bound = self.specialized_signature(f, Bindings::new(&[]), inferred.values)?;
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
        let mut checked = Vec::new();
        for (arg, param) in args.into_iter().zip(&interface.params) {
            let value = match (arg.source, arg.checked) {
                (Some(expr), Some(value)) if compatible(&param.ty, &value.ty) => {
                    super::check_argument_widths(
                        expr,
                        &param.ty,
                        Some(&self.inferred_bindings),
                        Some(&|| self.zero_calls_function()),
                    )?;
                    value
                }
                (Some(expr), _) => self.dslx_argument(expr, &param.ty)?,
                (None, Some(value)) if compatible(&param.ty, &value.ty) => value,
                (None, Some(value)) => {
                    return Err(format!(
                        "argument '{}' expects {}, got {}",
                        param.name, param.ty, value.ty
                    ));
                }
                (None, None) => unreachable!("map supplies a typed argument"),
            };
            checked.push(value);
        }
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
