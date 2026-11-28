// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap, btree_map::Entry};
use std::rc::Rc;

use xlsynth::dslx;
use xlsynth_pir::{ir, libxls_bridge::value_from_libxls};

use super::binding_types::witness_names;
use super::{DeclId, ImportedStructs, instance_spelling};
use crate::ast::Type;
use crate::check::MAX_TYPE_SIZE;
use crate::dslx_import::ProbeCapture;
use crate::parser::dslx_tokens;
use crate::typed::{BoundParametricValue, bits};

/// The earlier formals mentioned in an annotation and its resolved type for
/// each combination of their actuals. An alias with no dependencies has one
/// entry regardless of preceding arguments.
pub(super) struct StructFormalTypes {
    prior_indices: Vec<usize>,
    types: HashMap<Vec<BoundParametricValue>, Type>,
}

impl ImportedStructs {
    /// Resolve the next formal in its declaring scope with previously bound
    /// arguments. Native unsigned one-bit types are DSLX booleans.
    pub(crate) fn struct_parametric_type(
        &mut self,
        subject: &str,
        member: &str,
        previous: &[BoundParametricValue],
    ) -> Result<Type, String> {
        let graph = Rc::clone(&self.graph);
        let formals = graph
            .struct_formals(subject, member)
            .ok_or_else(|| format!("missing DSLX struct template '{subject}::{member}'"))?;
        if formals.len() > MAX_TYPE_SIZE {
            return Err(format!(
                "DSLX struct has more than {MAX_TYPE_SIZE} parametric bindings"
            ));
        }
        let index = previous.len();
        let formal = formals
            .get(index)
            .ok_or_else(|| format!("'{subject}::{member}' has no formal at index {index}"))?;
        let syntax = formal.ty.syntax.as_str();
        match syntax {
            "bool" | "u1" => return Ok(Type::Bool),
            "u32" => {
                return Ok(Type::Bits {
                    signed: false,
                    width: 32,
                });
            }
            "type" => {
                return Err(format!(
                    "DSLX parametric '{}' of '{subject}::{member}' expects a type",
                    formal.name
                ));
            }
            _ => { /* Resolve aliases and computed types in the native declaration scope. */ }
        }

        let key = (
            DeclId {
                subject: subject.into(),
                name: member.into(),
            },
            index,
        );
        let cache = match self.struct_formal_types.entry(key.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let names: BTreeSet<_> = dslx_tokens(syntax)?.into_iter().map(|t| t.text).collect();
                let prior_indices = formals[..index]
                    .iter()
                    .enumerate()
                    .filter_map(|(i, f)| names.contains(f.name.as_str()).then_some(i))
                    .collect();
                entry.insert(StructFormalTypes {
                    prior_indices,
                    types: HashMap::new(),
                })
            }
        };
        let actuals: Vec<_> = cache
            .prior_indices
            .iter()
            .map(|&i| previous[i].clone())
            .collect();
        if let Some(ty) = cache.types.get(&actuals) {
            return Ok(ty.clone());
        }

        let owner = graph
            .module_reference(subject, &[])
            .expect("template declaring module");
        let name = witness_names(&owner.source, "BXStructBindingType")
            .next()
            .expect("finite source reserves finitely many names");
        let formals: Vec<_> = cache
            .prior_indices
            .iter()
            .map(|&i| format!("{}: {}", formals[i].name, previous[i].ty))
            .collect();
        let (formals, invocation) = if formals.is_empty() {
            (String::new(), name.clone())
        } else {
            (
                format!("<{}>", formals.join(", ")),
                instance_spelling(&name, &actuals)?,
            )
        };
        let mut bytes = owner.source.clone();
        bytes.extend_from_slice(
            format!("\npub fn {name}{formals}() -> {syntax} {{ zero!<{syntax}>() }}\n").as_bytes(),
        );
        let (registry, mut isolated) = self.annotation_snapshot(subject, bytes)?;
        let source = format!(
            r#"#![feature(generics)]
import {subject} as bx_owner;
fn capture<T: type>(_v: T) {{}}
fn probe() {{ capture(bx_owner::{invocation}()) }}"#
        );
        let checked = registry
            .check_probe(&source, "0bx_struct_binding_type", &mut isolated)
            .map_err(|e| {
                format!(
                    "checking parametric '{}' of '{subject}::{member}': {}",
                    formal.name,
                    registry.remap_diagnostic(e)
                )
            })?;
        let native = ProbeCapture::new(&checked, |_| None)?.param_type(0)?;
        let ty = match (!native.is_enum()).then(|| native.is_bits_like()).flatten() {
            Some((false, 1)) => Type::Bool,
            Some((false, 32)) => Type::Bits {
                signed: false,
                width: 32,
            },
            _ => {
                return Err(format!(
                    "DSLX struct parametric '{}' of '{subject}::{member}' has unsupported type '{native}'; expected bool or u32",
                    formal.name
                ));
            }
        };
        self.struct_formal_types
            .get_mut(&key)
            .expect("registered formal")
            .types
            .insert(actuals, ty.clone());
        Ok(ty)
    }

    /// Read actuals in declaration order from the checked struct instance.
    /// Its environment includes defaults and arguments unused by the fields.
    pub(super) fn read_struct_bindings(
        &mut self,
        id: &DeclId,
        def: &dslx::StructDef,
        native: &dslx::Type,
    ) -> Result<Vec<BoundParametricValue>, String> {
        let count = def.get_parametric_binding_count();
        if self.public_struct_template_arity(&id.subject, &id.name) != Some(count) {
            return Err(format!(
                "Rust and DSLX disagree on the number of parametrics for '{}::{}'",
                id.subject, id.name
            ));
        }
        if count > MAX_TYPE_SIZE {
            return Err(format!(
                "DSLX struct has more than {MAX_TYPE_SIZE} parametric bindings"
            ));
        }
        let env = native
            .get_struct_parametric_env()
            .map_err(|e| e.to_string())?;
        let bindings: BTreeMap<_, _> = env.bindings().into_iter().collect();
        let mut actuals = Vec::with_capacity(count);
        for formal in def.parametric_bindings() {
            let ty = self.struct_parametric_type(&id.subject, &id.name, &actuals)?;
            let name = formal.get_identifier();
            let value = bindings
                .get(&name)
                .ok_or_else(|| format!("missing DSLX struct value binding '{name}'"))?
                .convert_to_ir()
                .map_err(|e| e.to_string())?;
            let (_, width) = bits(&ty).expect("validated unsigned struct formal");
            let value =
                value_from_libxls(&value, &ir::Type::Bits(width)).map_err(|e| e.to_string())?;
            actuals.push(BoundParametricValue { ty, value });
        }
        Ok(actuals)
    }
}
