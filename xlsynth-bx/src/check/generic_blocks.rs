// SPDX-License-Identifier: Apache-2.0

//! Inference of concrete BX types for generic block input ports.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{Block, Type};
use crate::parser::is_identifier;

use super::{MAX_ELABORATION_DEPTH, StructTable, TypeShape, compatible, type_deps};

/// Check the template's parameter namespace and ensure every type parameter
/// can be inferred solely from its data inputs.
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
    for name in std::iter::once(&block.clock)
        .chain(&block.reset)
        .chain(block.params.iter().map(|param| &param.name))
    {
        if type_params.contains(name.as_str()) {
            return Err(format!("port '{name}' collides with a type parameter"));
        }
    }
    let mut input_names = Vec::new();
    for param in &block.params {
        type_deps(&param.ty, &mut input_names);
    }
    for name in input_names {
        type_params.remove(name);
    }
    if let Some(missing) = block
        .type_params
        .iter()
        .find(|name| type_params.contains(name.as_str()))
    {
        return Err(format!(
            "cannot infer type parameter '{missing}': it does not appear in any data input"
        ));
    }
    Ok(())
}

/// Type parameters of one template and their bindings at a single call site.
#[derive(Clone)]
pub(super) struct Bindings {
    types: BTreeMap<String, Option<BoundType>>,
}

/// The concrete argument and its cached expanded layout, reused whenever the
/// parameter occurs in an annotation or port type.
#[derive(Clone)]
struct BoundType {
    ty: Type,
    shape: TypeShape,
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

    pub(super) fn declares(&self, name: &str) -> bool {
        self.types.contains_key(name)
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
                    *binding = Some(BoundType {
                        ty: canonical_argument(actual),
                        shape,
                    });
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
            Type::Bool | Type::Bits { .. } => formal.clone(),
        }
    }

    /// Build the specialization key in the template's declaration order.
    pub(super) fn arguments(&self, type_params: &[String]) -> Result<Vec<Type>, String> {
        type_params
            .iter()
            .map(|name| {
                self.get(name)
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
        Type::Bits { .. } | Type::Named(_) => ty.clone(),
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
