// SPDX-License-Identifier: Apache-2.0

//! Inference of concrete BX types for generic block input ports.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;

use crate::ast::{Block, CallbackParam, Expr, FunctionParametricRef, Param, Type};
use crate::parser::{is_identifier, is_use_identifier};

use super::{
    BlockCallbackFormal, BlockInterface, BlockTemplate, MAX_ELABORATION_DEPTH, ModuleDimensions,
    StructTable, TypeResolver, TypeShape, compatible, resolve_type, type_parameter_references,
};

/// A callable argument identifies a function in the caller's lexical scope.
pub(super) fn callback_argument_name(expr: &Expr) -> Result<&str, String> {
    match expr {
        Expr::Name(name) => Ok(name),
        _ => Err("callable argument must name a function".into()),
    }
}

/// Check the template's parameter namespace and ensure every type parameter
/// can be inferred from data inputs and then callback results, in order.
pub(super) fn validate_declaration(block: &Block) -> Result<(), String> {
    let mut type_params = BTreeSet::new();
    for name in &block.type_params {
        if !is_identifier(name) {
            return Err(format!("invalid type parameter name '{name}'"));
        }
        if !type_params.insert(name.as_str()) {
            return Err(format!("duplicate type parameter '{name}'"));
        }
    }
    let mut callback_names = BTreeSet::new();
    for callback in &block.callback_params {
        let name = &callback.name;
        if !is_use_identifier(name) {
            return Err(format!("invalid callable parameter name '{name}'"));
        }
        if type_params.contains(name.as_str()) || !callback_names.insert(name.as_str()) {
            return Err(format!("duplicate block parameter '{name}'"));
        }
    }
    let mut value_names = BTreeSet::new();
    let explicit_count = block.value_parametrics.len() + block.callback_params.len();
    let mut previous_index = None;
    for parametric in &block.value_parametrics {
        let name = parametric.name.as_str();
        if !is_identifier(name) {
            return Err(format!("invalid value parametric name '{name}'"));
        }
        if type_params.contains(name) || callback_names.contains(name) || !value_names.insert(name)
        {
            return Err(format!("duplicate block parameter '{name}'"));
        }
        if parametric.default.is_some() {
            return Err(format!(
                "block value parametric '{name}' requires an explicit argument"
            ));
        }
        if parametric.index >= explicit_count
            || previous_index.is_some_and(|i| i >= parametric.index)
        {
            return Err(format!(
                "invalid argument position for block value parametric '{name}'"
            ));
        }
        previous_index = Some(parametric.index);
    }
    for name in std::iter::once(&block.clock)
        .chain(&block.reset)
        .chain(block.params.iter().map(|param| &param.name))
    {
        if type_params.contains(name.as_str()) {
            return Err(format!("port '{name}' collides with a type parameter"));
        }
        if callback_names.contains(name.as_str()) {
            return Err(format!("port '{name}' collides with a callable parameter"));
        }
        if value_names.contains(name.as_str()) {
            return Err(format!("port '{name}' collides with a value parametric"));
        }
    }
    validate_inference_sources(&block.type_params, &block.params, &block.callback_params)
}

/// Every type formal needs evidence from a data argument or the result of a
/// callback whose inputs have already been inferred.
pub(super) fn validate_inference_sources(
    types: &[String],
    params: &[Param],
    callbacks: &[CallbackParam],
) -> Result<(), String> {
    let mut type_params: BTreeSet<_> = types.iter().map(String::as_str).collect();
    let mut input_names = Vec::new();
    for param in params {
        type_parameter_references(&param.ty, &mut input_names);
    }
    for name in input_names {
        type_params.remove(name);
    }
    for callback in callbacks {
        let mut dependencies = Vec::new();
        for ty in &callback.params {
            type_parameter_references(ty, &mut dependencies);
        }
        if let Some(missing) = dependencies
            .iter()
            .find(|name| type_params.contains(**name))
        {
            return Err(format!(
                "cannot infer type parameter '{missing}' before callable parameter '{}'; its input types must come from a data input or an earlier callback result",
                callback.name
            ));
        }
        dependencies.clear();
        type_parameter_references(&callback.result, &mut dependencies);
        for name in dependencies {
            type_params.remove(name);
        }
    }
    if let Some(missing) = types
        .iter()
        .find(|name| type_params.contains(name.as_str()))
    {
        let source = if callbacks.is_empty() {
            "any data input"
        } else {
            "any data input or callback result"
        };
        return Err(format!(
            "cannot infer type parameter '{missing}': it does not appear in {source}"
        ));
    }
    Ok(())
}

impl<'m> BlockTemplate<'m> {
    /// Resolve module types while retaining the block's symbolic type names
    /// for data inference and callback specialization at each call site.
    pub(super) fn new(
        block: &'m Block,
        params: Vec<Param>,
        result: Type,
        bindings: &Bindings,
        dimensions: ModuleDimensions<'_, 'm>,
        types: &mut TypeResolver<'_, 'm>,
    ) -> Result<Self, String> {
        let dimension_type = super::value_parametrics::dimension_type();
        let mut preceding = BTreeSet::new();
        for formal in block.parametric_bindings() {
            let name = match formal {
                FunctionParametricRef::Callback(callback) => &callback.name,
                FunctionParametricRef::Value(value) => {
                    let is_dimension = if matches!(value.ty, Type::Named(_)) {
                        resolve_type(
                            &value.ty,
                            &|name| preceding.contains(name),
                            Some(bindings),
                            dimensions,
                            types,
                            MAX_ELABORATION_DEPTH,
                        )? == dimension_type
                    } else {
                        value.ty == dimension_type
                    };
                    if !is_dimension {
                        return Err(format!(
                            "block value parametric '{}' must have type u32",
                            value.name
                        ));
                    }
                    &value.name
                }
            };
            preceding.insert(name.as_str());
        }
        let value_names: BTreeSet<_> = block
            .value_parametrics
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        let mut callback_indices = BTreeMap::new();
        let mut callback_params = Vec::with_capacity(block.callback_params.len());
        for (index, callback) in block.callback_params.iter().enumerate() {
            callback_indices.insert(callback.name.clone(), index);
            let mut resolve = |ty: &Type| {
                resolve_type(
                    ty,
                    &|name| callback_indices.contains_key(name) || value_names.contains(name),
                    Some(bindings),
                    dimensions,
                    types,
                    MAX_ELABORATION_DEPTH,
                )
            };
            let signature = CallbackParam {
                name: callback.name.clone(),
                params: callback
                    .params
                    .iter()
                    .map(&mut resolve)
                    .collect::<Result<_, _>>()?,
                result: resolve(&callback.result)?,
            };
            callback_params.push(BlockCallbackFormal::new(signature, bindings));
        }
        let TypeResolver {
            resolver, structs, ..
        } = types;
        structs.update_imports(resolver)?;
        for formal in &callback_params {
            let signature = &formal.signature;
            for ty in signature.params.iter().chain([&signature.result]) {
                structs
                    .formal_shape(ty, MAX_ELABORATION_DEPTH, Some(bindings))
                    .map_err(|e| format!("callable parameter '{}': {e}", signature.name))?;
            }
        }
        BlockInterface::check_ports(
            &block.clock,
            block.reset.as_deref(),
            &params,
            &result,
            structs,
            Some(bindings),
            None,
        )?;
        Ok(Self {
            block,
            stages: dimensions.block_stages(block)?,
            params,
            result,
            callback_params,
            callback_indices,
        })
    }
}

/// Type parameters of one template and their bindings at a single call site.
#[derive(Clone)]
pub(super) struct Bindings {
    types: BTreeMap<String, Option<TypeArgument>>,
}

/// A shared specialization-key component. Callback results reused as inputs
/// to another callback keep the same canonical type across call sites.
pub(super) type TypeArgument = Rc<BoundType>;

/// The concrete argument and its cached expanded layout, reused whenever the
/// parameter occurs in an annotation or port type.
pub(super) struct BoundType {
    ty: Type,
    shape: TypeShape,
    pub(super) source_nodes: usize,
    type_hash: u64,
}

impl BoundType {
    /// Only independently inferred types with the same hash require a
    /// structural comparison to resolve equality.
    pub(super) fn comparison_nodes(&self, other: &Self) -> usize {
        if std::ptr::eq(self, other) || self.type_hash != other.type_hash {
            0
        } else {
            self.source_nodes.min(other.source_nodes)
        }
    }
}

impl PartialEq for BoundType {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (self.type_hash == other.type_hash && self.ty == other.ty)
    }
}

impl Eq for BoundType {}

impl Hash for BoundType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_hash.hash(state);
    }
}

/// Whether dimensions are resolved and a reference can still use context.
#[derive(Clone, Copy, Eq, PartialEq)]
enum TypePattern {
    Resolved,
    Source,
    SourceProjection,
}

impl Bindings {
    pub(super) fn new(type_params: &[String]) -> Self {
        Self {
            types: type_params
                .iter()
                .map(|name| (name.clone(), None))
                .collect(),
        }
    }

    pub(super) fn get(&self, name: &str) -> Option<&Type> {
        Some(&self.types.get(name)?.as_ref()?.ty)
    }

    /// Use a concrete binding's cached layout, or the minimum layout of a
    /// still-symbolic type parameter when checking the declaration.
    pub(super) fn shape(&self, name: &str) -> Option<TypeShape> {
        Some(
            self.types
                .get(name)?
                .as_ref()
                .map_or(TypeShape::LEAF, |t| t.shape),
        )
    }

    pub(super) fn declares(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    /// Check whether the pattern still needs a type witness.
    pub(super) fn has_unbound_variable(&self, pattern: &Type) -> bool {
        match pattern {
            Type::Named(name) => self.types.get(name).is_some_and(Option::is_none),
            Type::Tuple(fields) => fields.iter().any(|field| self.has_unbound_variable(field)),
            Type::TypeApplied { args, .. } => args.iter().any(|arg| self.has_unbound_variable(arg)),
            Type::Array { element, .. } | Type::ParametricArray { element, .. } => {
                self.has_unbound_variable(element)
            }
            _ => false,
        }
    }

    /// A type parameter cannot resolve to a callee or an import qualifier.
    pub(super) fn reject_value(&self, name: &str) -> Result<(), String> {
        let head = name.split_once("::").map_or(name, |(head, _)| head);
        if self.declares(head) {
            return Err(format!("'{head}' names a type parameter here"));
        }
        Ok(())
    }

    pub(super) fn source_nodes(&self, name: &str) -> Option<usize> {
        Some(self.types.get(name)?.as_ref()?.source_nodes)
    }

    /// Apply a callback result's inferred variables in the caller. Unbound
    /// variables share their cached concrete type and layout. Only an existing
    /// independent binding requires another comparison.
    pub(super) fn merge_callback_results(
        &mut self,
        result: &Bindings,
        mut charge: impl FnMut(usize) -> Result<(), String>,
    ) -> Result<(), String> {
        charge(result.types.len())?;
        for (name, inferred) in &result.types {
            let inferred = inferred.as_ref().expect("matched result type parameter");
            let binding = self.types.get_mut(name).expect("declared type parameter");
            if let Some(prior) = binding {
                charge(prior.comparison_nodes(inferred))?;
                if prior != inferred {
                    return Err(format!(
                        "conflicting types for type parameter '{name}': inferred {} and {}",
                        prior.ty, inferred.ty
                    ));
                }
            }
            *binding = Some(Rc::clone(inferred));
        }
        Ok(())
    }

    /// Match a resolved formal port shape against the argument's checked type.
    /// Named structs retain their identity even when their layouts coincide.
    pub(super) fn match_input(
        &mut self,
        formal: &Type,
        actual: &Type,
        structs: &StructTable,
    ) -> Result<(), String> {
        self.match_pattern(formal, actual, structs, TypePattern::Resolved)
    }

    /// Infer type variables while dimensions and nominal imports in a source
    /// annotation still await resolution in the function's declaration scope.
    /// An inferred projection still needs to be checked with its bound type.
    pub(super) fn infer_source_type(
        &mut self,
        formal: &Type,
        actual: &Type,
        structs: &StructTable,
        contextual_projection: bool,
    ) -> Result<(), String> {
        let pattern = if contextual_projection {
            TypePattern::SourceProjection
        } else {
            TypePattern::Source
        };
        self.match_pattern(formal, actual, structs, pattern)
    }

    fn match_pattern(
        &mut self,
        formal: &Type,
        actual: &Type,
        structs: &StructTable,
        pattern: TypePattern,
    ) -> Result<(), String> {
        if let Type::Named(name) = formal
            && let Some(binding) = self.types.get_mut(name)
        {
            match binding {
                Some(prior)
                    if pattern != TypePattern::SourceProjection
                        && !compatible(&prior.ty, actual) =>
                {
                    return Err(format!(
                        "conflicting types for type parameter '{name}': inferred {} and {actual}",
                        prior.ty
                    ));
                }
                Some(_) => { /* Keep the first binding as context for checking the argument. */ }
                None => {
                    let shape = structs.type_shape(actual, MAX_ELABORATION_DEPTH)?;
                    let ty = canonical_argument(actual);
                    let mut hasher = DefaultHasher::new();
                    ty.hash(&mut hasher);
                    *binding = Some(Rc::new(BoundType {
                        ty,
                        shape,
                        source_nodes: shape.source_nodes,
                        type_hash: hasher.finish(),
                    }));
                }
            }
            return Ok(());
        }
        match (formal, actual) {
            (Type::TypeApplied { name, args }, Type::Named(actual)) => {
                if let Some(instance) = structs.instances.get(actual)
                    && instance.template.name == *name
                    && instance.types.len() == args.len()
                {
                    for (formal, actual) in args.iter().zip(&instance.types) {
                        self.match_pattern(formal, actual, structs, pattern)?;
                    }
                    Ok(())
                } else if pattern != TypePattern::Resolved {
                    Ok(())
                } else {
                    Err(format!("expected {formal}, got {actual}"))
                }
            }
            (Type::Tuple(formals), Type::Tuple(actuals)) if formals.len() == actuals.len() => {
                for (formal, actual) in formals.iter().zip(actuals) {
                    self.match_pattern(formal, actual, structs, pattern)?;
                }
                Ok(())
            }
            (
                Type::Array {
                    element: formal,
                    size,
                },
                Type::Array {
                    element: actual,
                    size: actual_size,
                },
            ) if size == actual_size => self.match_pattern(formal, actual, structs, pattern),
            (
                Type::ParametricArray { element, .. },
                Type::Array {
                    element: actual, ..
                },
            ) if pattern != TypePattern::Resolved => {
                self.match_pattern(element, actual, structs, pattern)
            }
            _ if pattern != TypePattern::Resolved => Ok(()),
            _ if compatible(formal, actual) => Ok(()),
            _ => Err(format!("expected {formal}, got {actual}")),
        }
    }

    /// Bound size and depth before copying type arguments into their use sites.
    /// A non-parameter name requires one AST node. Its full nominal layout is
    /// checked after source names (including import aliases) are resolved.
    pub(super) fn validate_substitution(&self, formal: &Type) -> Result<(), String> {
        TypeShape::for_type(
            formal,
            MAX_ELABORATION_DEPTH,
            &|name| match self.types.get(name) {
                Some(Some(binding)) => Ok(binding.shape),
                Some(None) => Err(format!("cannot infer type parameter '{name}'")),
                None => Ok(TypeShape::LEAF),
            },
        )
        .map(|_| ())
    }

    /// Instantiate a resolved formal type once its complete layout fits.
    pub(super) fn concrete_type(&self, formal: &Type) -> Result<Type, String> {
        self.validate_substitution(formal)?;
        Ok(self.substitute(formal))
    }

    fn substitute(&self, formal: &Type) -> Type {
        match formal {
            Type::Named(name) => self.get(name).unwrap_or(formal).clone(),
            Type::Tuple(fields) => {
                Type::Tuple(fields.iter().map(|field| self.substitute(field)).collect())
            }
            Type::Array { element, size } => Type::array(self.substitute(element), *size),
            Type::TypeApplied { name, args } => Type::TypeApplied {
                name: name.clone(),
                args: args.iter().map(|arg| self.substitute(arg)).collect(),
            },
            Type::Bool
            | Type::Bits { .. }
            | Type::Enum { .. }
            | Type::ParametricBits { .. }
            | Type::ParametricArray { .. }
            | Type::Applied { .. }
            | Type::ParametricStruct { .. } => formal.clone(),
        }
    }

    /// Identify the type parameters referenced by callback annotations.
    pub(super) fn referenced_parameters<'t>(
        &self,
        types: impl IntoIterator<Item = &'t Type>,
    ) -> Vec<String> {
        let mut names = Vec::new();
        for ty in types {
            type_parameter_references(ty, &mut names);
        }
        names.retain(|name| self.declares(name));
        names.sort_unstable();
        names.dedup();
        names.into_iter().map(str::to_owned).collect()
    }

    /// Before callbacks are resolved, a template's data arguments can leave
    /// some types unbound. Name order is stable within that declaration.
    pub(super) fn partial_arguments(&self) -> Vec<Option<TypeArgument>> {
        self.types.values().cloned().collect()
    }

    /// Build the specialization key in the template's declaration order.
    pub(super) fn arguments(&self, type_params: &[String]) -> Result<Vec<TypeArgument>, String> {
        type_params
            .iter()
            .map(|name| {
                self.types
                    .get(name)
                    .and_then(Option::as_ref)
                    .cloned()
                    .ok_or_else(|| format!("cannot infer type parameter '{name}'"))
            })
            .collect()
    }
}

/// Use one specialization for bool and u1, including inside aggregate inputs.
fn canonical_argument(ty: &Type) -> Type {
    match ty {
        Type::Bool => Type::unsigned(1),
        Type::Tuple(fields) => Type::Tuple(fields.iter().map(canonical_argument).collect()),
        Type::Array { element, size } => Type::array(canonical_argument(element), *size),
        Type::Bits { .. }
        | Type::Enum { .. }
        | Type::Named(_)
        | Type::ParametricBits { .. }
        | Type::ParametricArray { .. }
        | Type::Applied { .. }
        | Type::ParametricStruct { .. }
        | Type::TypeApplied { .. } => ty.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::MAX_TYPE_SIZE;

    #[test]
    fn substitution_checks_combined_size_and_depth_before_copying() -> Result<(), String> {
        let name = String::from("T");
        let formal = Type::Named(name.clone());
        let structs = StructTable::default();
        let mut bindings = Bindings::new(std::slice::from_ref(&name));
        let actual = Type::Tuple(vec![Type::Bool; 1024]);
        bindings.match_input(&formal, &actual, &structs)?;
        let repeated = Type::Tuple(vec![formal.clone(); 1024]);
        assert_eq!(
            bindings.concrete_type(&repeated).unwrap_err(),
            format!("aggregate exceeds maximum concrete type size {MAX_TYPE_SIZE}"),
        );

        let nested = (0..MAX_ELABORATION_DEPTH).fold(formal, |element, _| Type::Array {
            element: Box::new(element),
            size: 1,
        });
        assert_eq!(
            bindings.concrete_type(&nested).unwrap_err(),
            format!("expanded type nesting exceeds {MAX_ELABORATION_DEPTH} levels"),
        );
        Ok(())
    }
}
