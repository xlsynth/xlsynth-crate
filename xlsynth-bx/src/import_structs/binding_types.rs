// SPDX-License-Identifier: Apache-2.0

//! Resolve native function value-binding annotations before BX elaborates the
//! actuals.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::rc::Rc;

use xlsynth::dslx::ImportData;

use super::{ImportedStructs, NativeSourceFiles, ResolvedShape, probe_function};
use crate::OutlineShape;
use crate::ast::Type;
use crate::dslx_import::{ProbeCapture, TypeWitness};
use crate::parser::outline::argument_identity;
use crate::parser::source_bytes::decode_dslx_source;
use crate::parser::{parse_value_type, source_bytes::RelocatedSource};

/// Allocate witness declaration names outside the source's existing
/// identifiers.
fn witness_names<'a>(source: &[u8], prefix: &'a str) -> impl Iterator<Item = String> + 'a {
    let occupied: BTreeSet<_> = source
        .split(|b| !b.is_ascii_alphanumeric() && *b != b'_')
        .filter_map(|word| {
            std::str::from_utf8(word.strip_prefix(prefix.as_bytes())?)
                .ok()?
                .parse::<usize>()
                .ok()
        })
        .collect();
    (0..)
        .filter(move |id| !occupied.contains(id))
        .map(move |id| format!("{prefix}{id}"))
}

impl ImportedStructs {
    /// libxls v0.61 leaves generic function annotations unresolved until an
    /// invocation. Check the supplied prefix in the original declaration scope
    /// so module constants, private aliases and computed widths keep their
    /// meaning.
    pub(crate) fn binding_types(
        &mut self,
        subject: &str,
        function: &str,
        range: Range<usize>,
        data: &mut ImportData,
    ) -> Result<Vec<Type>, String> {
        let graph = Rc::clone(&self.graph);
        let owner = graph
            .module_reference(subject, &[])
            .ok_or_else(|| format!("unknown DSLX module '{subject}'"))?;
        let formals = owner
            .outline
            .function_parametrics
            .get(function)
            .ok_or_else(|| {
                format!("'{subject}::{function}' requires a public value-parametric DSLX function")
            })?;
        let formals = formals.get(range).ok_or_else(|| {
            format!(
                "'{subject}::{function}' expects at most {} parametric arguments",
                formals.len()
            )
        })?;
        let mut types = Vec::with_capacity(formals.len());
        let mut pending = Vec::new();
        for (i, binding) in formals.iter().enumerate() {
            if binding.ty.syntax == "type" {
                return Err(format!(
                    "DSLX parametric '{}' of '{subject}::{function}' expects a type",
                    binding.name
                ));
            }
            let mut concrete = parse_value_type(&binding.ty.syntax).ok();
            if concrete.is_none()
                && let OutlineShape::Name(path) = &binding.ty.shape
                && !binding.ty.syntax.contains('<')
                && let Some(id) = graph.reference(subject, path)
                && graph.has_public_member(&id.subject, &id.name)
                && matches!(
                    graph.named(&id.subject, &id.name),
                    Some(
                        ResolvedShape::Enum(_)
                            | ResolvedShape::Struct(_)
                            | ResolvedShape::Tuple(_)
                            | ResolvedShape::Array(_)
                    )
                )
            {
                concrete = Some(self.resolve_exported_type(&id.subject, &id.name, data)?);
            }
            if concrete.is_none() {
                pending.push((i, &binding.ty));
            }
            types.push(concrete);
        }
        if !pending.is_empty() {
            // Only annotations needing native resolution pay for the second
            // snapshot. Keep its temporary type aliases outside the graph
            // used for semantic checking and final IR conversion.
            let mut bytes = owner.source.clone();
            let mut params = Vec::new();
            for ((i, annotation), alias) in pending
                .iter()
                .zip(witness_names(&owner.source, "BXBindingType"))
            {
                bytes.extend_from_slice(
                    format!("\npub type {alias} = {};\n", annotation.syntax).as_bytes(),
                );
                params.push(format!("value_{i}: bx_owner::{alias}"));
            }
            let (mut registry, mut isolated) = self.annotation_snapshot(subject, bytes)?;
            let imports = format!("import {subject} as bx_owner;");
            let params = params.join(", ");
            let source = format!("{imports}\nfn probe({params}) {{}}");
            let checked = registry
                .check_probe(&source, "0bx_binding_types", &mut isolated)
                .map_err(|e| {
                    format!("checking explicit value annotations of '{subject}::{function}': {e}")
                })?;
            let probe = probe_function(&checked)?;
            for (index, (i, annotation)) in pending.into_iter().enumerate() {
                let ty = checked
                    .get_type_info()
                    .get_type_for_type_annotation(&probe.get_param(index).get_type_annotation())
                    .ok_or("missing concrete DSLX binding annotation")?;
                let shape = graph.resolve_type(subject, annotation);
                let witness = TypeWitness {
                    imports: imports.clone(),
                    params: params.clone(),
                    expr: format!("value_{i}"),
                };
                let ty =
                    registry.reflect_at(&ty, &mut isolated, &witness, Some(&shape), false, 0)?;
                types[i] = Some(self.import_binding_type(&registry, ty, data)?);
            }
        }
        Ok(types
            .into_iter()
            .map(|t| t.expect("all annotations resolved"))
            .collect())
    }

    /// An empty array supplies no nominal element witness. A sibling with
    /// the same signature can return that parameter, giving reflection the
    /// original constrained value to project through, including latent struct
    /// bindings.
    pub(crate) fn contextual_argument_type(
        &mut self,
        subject: &str,
        function: &str,
        index: usize,
        make_witness: impl FnOnce(&str) -> TypeWitness,
        data: &mut ImportData,
    ) -> Result<Type, String> {
        let owner = self
            .graph
            .module_reference(subject, &[])
            .ok_or("missing DSLX declaring module")?;
        let decoded = decode_dslx_source(&owner.source)?;
        let witness_name = witness_names(&owner.source, "BXArgumentWitness")
            .next()
            .expect("finite source reserves finitely many names");
        let (declaration, annotation) =
            argument_identity(&decoded, function, index, &witness_name)?;
        let shape = self.graph.resolve_type(subject, &annotation);
        let mut bytes = owner.source.clone();
        bytes.extend_from_slice(format!("\n{declaration}\n").as_bytes());
        let (mut registry, mut isolated) = self.annotation_snapshot(subject, bytes)?;
        let witness = make_witness(&witness_name);
        let source = format!(
            "#![feature(generics)]\n{}fn capture<T:type>(_v:T) {{}}\nfn probe({}) {{ capture({}) }}",
            witness.imports, witness.params, witness.expr
        );
        let checked = registry.check_probe(&source, "0bx_argument_type", &mut isolated)?;
        let capture = ProbeCapture::new(&checked, |_| None)?;
        let ty = registry.reflect_at(
            &capture.param_type(0)?,
            &mut isolated,
            &witness,
            Some(&shape),
            false,
            0,
        )?;
        self.import_binding_type(&registry, ty, data)
    }

    /// Typecheck added witness declarations without changing the native module
    /// used for IR.
    fn annotation_snapshot(
        &self,
        subject: &str,
        bytes: Vec<u8>,
    ) -> Result<(Self, ImportData), String> {
        let overrides = BTreeMap::from([(
            subject.to_owned(),
            RelocatedSource {
                bytes,
                columns: Default::default(),
            },
        )]);
        let files = NativeSourceFiles::capture_with_overrides(&self.graph, overrides)?;
        let isolated = files.import_data();
        Ok((
            Self::for_snapshot(Rc::clone(&self.graph), Some(files)),
            isolated,
        ))
    }

    /// Rebind nominal layouts reflected in the type-only snapshot to handles
    /// in the main import graph. Concrete struct spellings include every
    /// binding.
    fn import_binding_type(
        &mut self,
        registry: &Self,
        ty: Type,
        data: &mut ImportData,
    ) -> Result<Type, String> {
        if let Some((subject, member)) = registry
            .aggregate_spelling(&ty)
            .and_then(|alias| alias.split_once("::"))
        {
            return self.resolve_exported_type(subject, member, data);
        }
        match ty {
            Type::Enum { ref name, .. } | Type::Named(ref name) => {
                let (subject, spelling) = name
                    .split_once("::")
                    .ok_or_else(|| format!("missing DSLX module for binding type '{ty}'"))?;
                if !self.graph.has_module(subject) {
                    return Err(format!(
                        "DSLX value parametric requires a public spelling for nominal type '{name}'"
                    ));
                }
                let member = spelling
                    .split('<')
                    .next()
                    .expect("spelling has a first component");
                if matches!(ty, Type::Named(_)) {
                    self.resolve_spelling(subject, member, spelling, data)
                        .map(Type::Named)
                } else {
                    self.resolve_exported_type(subject, member, data)
                }
            }
            Type::Tuple(fields) => fields
                .into_iter()
                .map(|f| self.import_binding_type(registry, f, data))
                .collect::<Result<Vec<_>, _>>()
                .map(Type::Tuple),
            Type::Array { element, size } => Ok(Type::array(
                self.import_binding_type(registry, *element, data)?,
                size,
            )),
            _ => Ok(ty),
        }
    }
}
