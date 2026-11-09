// SPDX-License-Identifier: Apache-2.0

//! Bind DSLX-style function value parametrics from constexpr actuals, argument
//! types and defaults, then typecheck each concrete function specialization.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth_pir::{IrValue, ir_value_utils::ir_bits_to_usize};

use super::function_specializations::BoundValueSignature;
use super::{
    BodyChecker, BodyKind, CallbackFormal, Declarations, FunctionInterface, K, StructTable, bits,
    child_type_budget, compatible, declare, minimum_type, resolve_type, source_type_nodes,
};
use crate::ast::{
    CallbackParam, Expr, Function, FunctionParametricRef, Param, ParametricBinding, Type,
};
use crate::typed::{BoundParametricValue, ConstantUse, TypedExpr};

fn dimension_type() -> Type {
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
                Type::ParametricBits { .. } | Type::Named(_) | Type::Applied { .. } => {
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
struct InferredBindings<'f> {
    formals: &'f [ParametricBinding],
    indices: BTreeMap<&'f str, usize>,
    values: Vec<Option<BoundParametricValue>>,
    // Gather all witnesses from one data argument before its bindings become
    // context for later arguments. Tuple fields must agree with one another.
    witnesses: BTreeMap<usize, BoundParametricValue>,
}

impl<'f> InferredBindings<'f> {
    fn new(formals: &'f [ParametricBinding]) -> Self {
        Self {
            formals,
            indices: formals
                .iter()
                .enumerate()
                .map(|(i, p)| (p.name.as_str(), i))
                .collect(),
            values: vec![None; formals.len()],
            witnesses: BTreeMap::new(),
        }
    }

    fn missing(&self, dimension: &Expr) -> Option<usize> {
        let Expr::Name(name) = dimension else {
            return None;
        };
        let i = *self.indices.get(name.as_str())?;
        (self.values[i].is_none() && self.formals[i].default.is_none()).then_some(i)
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
            Type::Tuple(fields) => fields.iter().any(|f| self.needs_witness(f)),
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

    fn finish_argument(&mut self) {
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

    fn infer(&mut self, formal: &Type, actual: &Type) -> Result<(), String> {
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
                self.infer(element, actual)?;
            }
            (
                Type::Array { element, .. },
                Type::Array {
                    element: actual, ..
                },
            ) => self.infer(element, actual)?,
            (Type::Tuple(fields), Type::Tuple(actuals)) => {
                for (field, actual) in fields.iter().zip(actuals) {
                    self.infer(field, actual)?;
                }
            }
            _ => { /* Concrete constraints are checked against the final signature. */ }
        }
        Ok(())
    }

    /// An array literal provides its length even when its element needs the
    /// formal parameter's width as context. An untyped empty literal needs its
    /// complete array type from context before it can supply a length.
    fn literal_lengths(&mut self, pattern: &Type, expr: &Expr) -> Result<(), String> {
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
        for stmt in self.statements.drain(first..) {
            if let super::CheckedStmt::Let { name, .. } = stmt {
                self.constants.remove(&name);
                self.constexpr_names.remove(&name);
            }
        }
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
                            || callbacks.is_some_and(|indices| indices.contains_key(name))
                    },
                    self.specialization.map(|s| &s.bindings),
                    self.resolver,
                    self.structs,
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
        let sources: Vec<_> = args.iter().collect();
        self.with_parametric_context(|checker| {
            checker.with_scoped_results(&sources, |checker, args| {
                let args = args
                    .iter()
                    .map(|&expr| DataArgument {
                        source: Some(expr),
                        checked: None,
                    })
                    .collect();
                checker.instantiate_value_function(f, actuals, args)
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
        let registry = self
            .function_registry
            .as_deref_mut()
            .expect("module function registry");
        registry.enter(&f.name)?;
        let result = stacker::maybe_grow(1024 * 1024, 2 * 1024 * 1024, || {
            let mut definition =
                BodyChecker::new(self.decls, self.resolver, self.structs, BodyKind::Function);
            definition.function_position = self
                .decls
                .positions
                .get(f.name.as_str())
                .map(|at| at.saturating_sub(1));
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
        for formal in f.parametric_bindings() {
            let formal = match formal {
                FunctionParametricRef::Value(formal) => formal,
                FunctionParametricRef::Callback(formal) => {
                    let params = formal
                        .params
                        .iter()
                        .map(|ty| self.source_type(ty))
                        .collect::<Result<Vec<_>, _>>()?;
                    let result = self.source_type(&formal.result)?;
                    let binding_work = params.iter().chain([&result]).map(source_type_nodes).sum();
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
            let ty = self.source_type(&formal.ty)?;
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
            self.bind_parametric_value(&formal.name, &bound);
            values.push(bound);
        }
        let mut params = Vec::new();
        for p in &f.params {
            let ty = self.source_type(&p.ty)?;
            self.names.insert(p.name.clone());
            params.push(Param {
                name: p.name.clone(),
                ty,
            });
        }
        let result = self.source_type(&f.result)?;
        Ok(BoundValueSignature {
            values,
            interface: FunctionInterface {
                params,
                result,
                callback_params: callbacks,
                callback_indices: self.decls.value_template_callbacks[f.name.as_str()].clone(),
                local: true,
            },
        })
    }

    /// Explicit actuals and input types come from the caller. Defaults and
    /// formal annotations are evaluated in the function's definition scope.
    fn instantiate_value_function(
        &mut self,
        f: &'m Function,
        actuals: &[Expr],
        mut args: Vec<DataArgument<'_>>,
    ) -> Result<(Type, K), String> {
        if args.len() != f.params.len() {
            return Err(format!(
                "'{}' expects {} arguments, got {}",
                f.name,
                f.params.len(),
                args.len()
            ));
        }
        let mut inferred = InferredBindings::new(&f.value_parametrics);
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
                inferred.literal_lengths(&param.ty, expr)?;
            }
            if inferred.needs_witness(&param.ty) {
                if let Some(expr) = arg.source {
                    if minimum_type(expr)?.is_some() {
                        continue;
                    }
                    arg.checked = Some(self.expr(expr, None)?);
                }
                inferred.infer(
                    &param.ty,
                    &arg.checked.as_ref().expect("checked argument").ty,
                )?;
            }
            inferred.finish_argument();
        }
        let cached = self
            .function_registry
            .as_deref_mut()
            .expect("module function registry")
            .cached_value_signature(f, &inferred.values)?;
        let bound = match cached {
            Some(bound) => bound,
            None => {
                let signature = self.with_value_definition(f, |definition| {
                    definition.bound_value_signature(f, &inferred.values)
                })?;
                self.function_registry
                    .as_deref_mut()
                    .expect("module function registry")
                    .cache_value_signature(f, inferred.values, signature)?
            }
        };
        let interface = &bound.interface;
        let mut targets = Vec::new();
        for (i, formal) in f.parametric_bindings().enumerate() {
            if let FunctionParametricRef::Callback(_) = formal {
                let actual = actuals
                    .get(i)
                    .ok_or_else(|| format!("'{}' requires explicit callable arguments", f.name))?;
                let name = super::generic_blocks::callback_argument_name(actual)?;
                let source = self.callback_source(name, |head| self.names.contains(head))?;
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
    use crate::{check_module, parse_module};

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
