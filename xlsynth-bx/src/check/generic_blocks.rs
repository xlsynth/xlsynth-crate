// SPDX-License-Identifier: Apache-2.0

//! Inference of concrete BX types for generic block input ports.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::rc::Rc;

use crate::ast::{Block, CallbackParam, Param, Type};
use crate::parser::{is_identifier, is_use_identifier};

use super::{
    BlockCallbackFormal, BlockInterface, BlockTemplate, ImportResolver, MAX_ELABORATION_DEPTH,
    StructTable, TypeShape, compatible, resolve_type, source_type_nodes, type_deps,
};

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
    }
    let mut input_names = Vec::new();
    for param in &block.params {
        type_deps(&param.ty, &mut input_names);
    }
    for name in input_names {
        type_params.remove(name);
    }
    for callback in &block.callback_params {
        let mut dependencies = Vec::new();
        for ty in &callback.params {
            type_deps(ty, &mut dependencies);
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
        type_deps(&callback.result, &mut dependencies);
        for name in dependencies {
            type_params.remove(name);
        }
    }
    if let Some(missing) = block
        .type_params
        .iter()
        .find(|name| type_params.contains(name.as_str()))
    {
        let source = if block.callback_params.is_empty() {
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
        resolver: &mut ImportResolver,
        structs: &mut StructTable,
    ) -> Result<Self, String> {
        let mut callback_indices = BTreeMap::new();
        let mut callback_params = Vec::with_capacity(block.callback_params.len());
        for (index, callback) in block.callback_params.iter().enumerate() {
            callback_indices.insert(callback.name.clone(), index);
            let mut resolve = |ty: &Type| {
                resolve_type(
                    ty,
                    &|name| callback_indices.contains_key(name),
                    Some(bindings),
                    resolver,
                    structs,
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
        )?;
        Ok(Self {
            block,
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
        if let Type::Named(name) = formal
            && let Some(binding) = self.types.get_mut(name)
        {
            match binding {
                Some(prior) if !compatible(&prior.ty, actual) => {
                    return Err(format!(
                        "conflicting types for type parameter '{name}': inferred {} and {actual}",
                        prior.ty
                    ));
                }
                Some(_) => { /* A repeated occurrence agrees with the first binding. */ }
                None => {
                    let shape = structs.type_shape(actual, MAX_ELABORATION_DEPTH)?;
                    let ty = canonical_argument(actual);
                    let mut hasher = DefaultHasher::new();
                    ty.hash(&mut hasher);
                    *binding = Some(Rc::new(BoundType {
                        ty,
                        shape,
                        source_nodes: source_type_nodes(actual),
                        type_hash: hasher.finish(),
                    }));
                }
            }
            return Ok(());
        }
        match (formal, actual) {
            (Type::Tuple(formals), Type::Tuple(actuals)) if formals.len() == actuals.len() => {
                for (formal, actual) in formals.iter().zip(actuals) {
                    self.match_input(formal, actual, structs)?;
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
            ) if size == actual_size => self.match_input(formal, actual, structs),
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
            Type::Array { element, size } => Type::Array {
                element: Box::new(self.substitute(element)),
                size: *size,
            },
            Type::Bool | Type::Bits { .. } | Type::Enum { .. } | Type::Applied { .. } => {
                formal.clone()
            }
        }
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
        Type::Bool => Type::Bits {
            signed: false,
            width: 1,
        },
        Type::Tuple(fields) => Type::Tuple(fields.iter().map(canonical_argument).collect()),
        Type::Array { element, size } => Type::Array {
            element: Box::new(canonical_argument(element)),
            size: *size,
        },
        Type::Bits { .. } | Type::Enum { .. } | Type::Named(_) | Type::Applied { .. } => ty.clone(),
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
