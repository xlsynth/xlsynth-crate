// SPDX-License-Identifier: Apache-2.0

//! Bridge Rust's DSLX declaration identities to concrete types in libxls.

mod binding_types;
mod effect_source;
mod enum_conversion;
mod enums;
mod outline_graph;
mod runfiles;
mod specialization;

use std::collections::{BTreeMap, BTreeSet, HashMap};
#[cfg(test)]
use std::path::Path;
use std::rc::Rc;

use xlsynth::dslx::{self, ImportData, MatchableModuleMember};

use crate::ast::{Param, StructDef, Type, UsePath};
use crate::check::MAX_ELABORATION_DEPTH;
use crate::dslx_import::TypeWitness;
use crate::parser::parse_value_type;
use crate::typed::CheckedEnum;
pub(crate) use effect_source::{EffectFunctionSource, EffectSources, is_token_builtin};
pub(crate) use outline_graph::DslxSourceLoader;
use outline_graph::{DeclId, ImportFeatures, NativeSourceFiles, OutlineGraph, ResolvedShape};

/// The Rust declaration and BX layout associated with a checked native type.
#[derive(Default)]
struct StructBinding {
    declaration: Option<DeclId>,
    public_name: Option<String>,
    /// Ordered value bindings distinguish instances even when all fields
    /// coincide.
    layouts: BTreeMap<Rc<[u32]>, usize>,
}

/// The template and concrete bindings of a reflected DSLX layout.
struct ImportedStructInstance {
    template: DeclId,
    values: Rc<[u32]>,
}

/// A projection from a public DSLX aggregate to a type containing private
/// nominal declarations.
struct AggregateProjection {
    alias: Rc<str>,
    suffix: String,
}

/// A frozen module graph and its concrete struct types. Ordinary DSLX imports
/// check dependencies with the same rules as native DSLX callers.
pub(crate) struct ImportedStructs {
    graph: Rc<OutlineGraph>,
    files: Option<NativeSourceFiles>,
    bindings: HashMap<dslx::StructDef, StructBinding>,
    declarations: BTreeMap<DeclId, dslx::StructDef>,
    explicit_types: BTreeMap<String, dslx::Type>,
    /// Concrete non-record exports already reflected from the native snapshot.
    reflected_aliases: BTreeMap<String, Type>,
    /// Public DSLX names for whole aggregates, including private nominal
    /// leaves.
    aggregate_spellings: HashMap<Type, String>,
    /// Snapshot constructor or cast names for reflected private nominal values.
    value_spellings: HashMap<String, String>,
    aggregate_projections: HashMap<Type, AggregateProjection>,
    /// A native type display narrows the public alias witnesses to reflect.
    /// The canonical BX type, including declaration identities, selects the
    /// spelling.
    aggregate_candidates: HashMap<String, Vec<DeclId>>,
    named_layouts: BTreeMap<String, usize>,
    parametric_instances: BTreeMap<usize, ImportedStructInstance>,
    /// Reflected layouts that contain a concrete parametric struct, including
    /// through fields of another struct.
    instance_layouts: BTreeSet<usize>,
    exports_bound: bool,
    #[cfg(test)]
    type_witnesses: usize,
    definitions: Vec<StructDef>,
    enums: enums::ImportedEnums,
    enum_conversion_snapshot: std::cell::OnceCell<Result<Option<NativeSourceFiles>, String>>,
    probe_index: usize,
}

impl ImportedStructs {
    /// Capture the reachable sources and validate the requested BX uses with
    /// native dependency-import semantics.
    #[cfg(test)]
    pub(crate) fn load(
        uses: &[UsePath],
        stdlib: &Path,
        paths: &[&Path],
    ) -> Result<(Self, Option<ImportData>), String> {
        Self::from_sources(uses, DslxSourceLoader::new(stdlib, paths))
    }

    pub(crate) fn from_sources(
        uses: &[UsePath],
        loader: DslxSourceLoader<'_>,
    ) -> Result<(Self, Option<ImportData>), String> {
        let graph = loader.finish(uses)?;
        let files = (!uses.is_empty())
            .then(|| graph.native_source_files())
            .transpose()?;
        let mut data = files.as_ref().map(NativeSourceFiles::import_data);
        let registry = Self::for_snapshot(Rc::new(graph), files);
        if let Some(data) = data.as_mut() {
            for (index, path) in uses.iter().enumerate() {
                let subject = path.components.join(".");
                let use_path = path.components.join("::");
                let source = if registry.graph.has_module(&subject) {
                    format!("import {subject} as bx_module;")
                } else {
                    format!("#![feature(use_syntax)]\nuse {use_path};")
                };
                registry
                    .check_probe(&source, &format!("0bx_use_{index}"), data)
                    .map_err(|e| {
                        format!(
                            "checking use '{use_path}': {}",
                            registry.remap_diagnostic(e)
                        )
                    })?;
            }
        }
        Ok((registry, data))
    }

    /// Pair the frozen declaration graph with the native handles of one
    /// snapshot.
    fn for_snapshot(graph: Rc<OutlineGraph>, files: Option<NativeSourceFiles>) -> Self {
        Self {
            graph,
            files,
            bindings: HashMap::new(),
            declarations: BTreeMap::new(),
            explicit_types: BTreeMap::new(),
            reflected_aliases: BTreeMap::new(),
            aggregate_spellings: HashMap::new(),
            value_spellings: HashMap::new(),
            aggregate_projections: HashMap::new(),
            aggregate_candidates: HashMap::new(),
            named_layouts: BTreeMap::new(),
            parametric_instances: BTreeMap::new(),
            instance_layouts: BTreeSet::new(),
            exports_bound: false,
            #[cfg(test)]
            type_witnesses: 0,
            definitions: Vec::new(),
            enums: Default::default(),
            enum_conversion_snapshot: Default::default(),
            probe_index: 0,
        }
    }

    /// Pair public, concrete structs and enums, including aliases, with their
    /// native DSLX declaration handles.
    fn bind_exports(&mut self, data: &mut ImportData) -> Result<(), String> {
        if self.exports_bound {
            return Ok(());
        }
        let exports: Vec<_> = self
            .graph
            .nominal_exports()
            .map(|(name, id)| (name.clone(), Some(id.clone())))
            .chain(
                self.graph
                    .unresolved_aliases()
                    .cloned()
                    .map(|name| (name, None)),
            )
            .chain(
                self.graph
                    .private_aggregate_exports
                    .iter()
                    .cloned()
                    .map(|name| (name, None)),
            )
            .collect();
        if exports.is_empty() {
            self.exports_bound = true;
            return Ok(());
        }
        let mut modules = BTreeMap::new();
        let mut params = Vec::with_capacity(exports.len());
        for (index, (export, _)) in exports.iter().enumerate() {
            let next = modules.len();
            let qualifier = modules.entry(&export.subject).or_insert(next);
            params.push(format!(
                "arg_{index}: bx_module_{qualifier}::{}",
                export.name
            ));
        }
        let mut source = String::new();
        for (subject, index) in modules {
            source.push_str(&format!("import {subject} as bx_module_{index};\n"));
        }
        source.push_str(&format!("fn exported_types({}) {{}}", params.join(", ")));
        #[cfg(test)]
        {
            self.type_witnesses += 1;
        }
        let checked = self.check_probe(&source, "0bx_declarations", data)?;
        let function = probe_function(&checked)?;
        let info = checked.get_type_info();
        for (index, (export, target)) in exports.into_iter().enumerate() {
            let ty = info
                .get_type_for_type_annotation(&function.get_param(index).get_type_annotation())
                .ok_or_else(|| {
                    format!(
                        "missing concrete DSLX type for '{}::{}'",
                        export.subject, export.name
                    )
                })?;
            if ty.is_enum() {
                let path = format!("{}::{}", export.subject, export.name);
                let def = ty.get_enum_def().map_err(|e| e.to_string())?;
                let canonical = target
                    .as_ref()
                    .and_then(|id| self.graph.public_path(id))
                    .unwrap_or(&path);
                if target.is_none() {
                    self.enums
                        .native_member_names
                        .entry(def.clone())
                        .or_insert_with(|| {
                            (0..def.get_member_count())
                                .map(|i| def.get_member(i).get_name())
                                .collect()
                        });
                }
                self.enums
                    .exports
                    .entry(def)
                    .or_insert_with(|| canonical.to_owned());
                self.explicit_types.insert(path, ty);
                continue;
            }
            let Some(target) = target else {
                // This declared alias needs DSLX evaluation (e.g. an
                // associated type). Record any resulting public struct name.
                if let Ok(def) = ty.get_struct_def() {
                    let name = format!("{}::{}", export.subject, export.name);
                    if !def.is_parametric() {
                        self.bindings
                            .entry(def)
                            .or_default()
                            .public_name
                            .get_or_insert_with(|| name.clone());
                    }
                    self.explicit_types.insert(name, ty);
                } else {
                    let display = ty.to_string().map_err(|e| e.to_string())?;
                    if ty.is_array() || display.starts_with('(') {
                        self.explicit_types
                            .insert(format!("{}::{}", export.subject, export.name), ty);
                        self.aggregate_candidates
                            .entry(display)
                            .or_default()
                            .push(export);
                    }
                }
                continue;
            };
            let def = ty.get_struct_def().map_err(|e| e.to_string())?;
            self.bind_declaration(&def, &target)?;
            let binding = self.bindings.get_mut(&def).expect("registered declaration");
            binding.public_name = self.graph.public_path(&target).map(str::to_owned);
            self.explicit_types
                .insert(format!("{}::{}", export.subject, export.name), ty);
        }
        self.exports_bound = true;
        Ok(())
    }

    /// Pair an outlined declaration with its native handle in both directions.
    fn bind_declaration(&mut self, def: &dslx::StructDef, id: &DeclId) -> Result<(), String> {
        let binding = self.bindings.entry(def.clone()).or_default();
        if def.get_identifier() != id.name
            || binding
                .declaration
                .as_ref()
                .is_some_and(|known| known != id)
            || self.declarations.get(id).is_some_and(|known| known != def)
        {
            return Err(format!(
                "Rust and DSLX disagree on the struct declaration for '{}::{}'",
                id.subject, id.name
            ));
        }
        binding.declaration.get_or_insert_with(|| id.clone());
        self.declarations
            .entry(id.clone())
            .or_insert_with(|| def.clone());
        Ok(())
    }

    pub(crate) fn remap_diagnostic(&self, error: String) -> String {
        match &self.files {
            Some(files) => files.remap_diagnostic(error),
            None => error,
        }
    }

    /// Check a generated signature or call using only snapshot-qualified
    /// imports.
    pub(crate) fn check_probe(
        &self,
        source: &str,
        name: &str,
        data: &mut ImportData,
    ) -> Result<dslx::TypecheckedModule, String> {
        let files = self
            .files
            .as_ref()
            .expect("DSLX probe has a source snapshot");
        let source = files.native_source(source);
        dslx::parse_and_typecheck(&source, &format!("{name}.x"), name, data)
            .map_err(|e| self.remap_diagnostic(e.to_string()))
    }

    /// Start an independent native type context over the same frozen sources.
    pub(crate) fn fresh_import_data(&self) -> ImportData {
        self.files
            .as_ref()
            .expect("DSLX batch has a source snapshot")
            .import_data()
    }

    pub(crate) fn has_module(&self, subject: &str) -> bool {
        self.graph.has_module(subject)
    }

    pub(crate) fn import_features(&self, subject: &str) -> ImportFeatures {
        self.graph.features(subject)
    }

    pub(crate) fn has_public_member(&self, subject: &str, member: &str) -> bool {
        self.graph.has_public_member(subject, member)
    }

    pub(crate) fn has_public_function(&self, subject: &str, member: &str) -> bool {
        self.graph.has_public_function(subject, member)
    }

    pub(crate) fn has_public_type(&self, subject: &str, member: &str) -> bool {
        self.graph.has_public_type(subject, member)
    }

    /// A public alias keeps its declaration kind even when it names a struct.
    pub(crate) fn is_public_alias(&self, subject: &str, member: &str) -> bool {
        self.graph
            .module_reference(subject, &[])
            .and_then(|module| module.outline.declarations.get(member))
            .is_some_and(|decl| {
                decl.has_public_spelling(member)
                    && matches!(decl.kind, crate::OutlineKind::Alias(_))
            })
    }

    pub(crate) fn public_struct_template_arity(
        &self,
        subject: &str,
        member: &str,
    ) -> Option<usize> {
        self.graph.public_struct_template_arity(subject, member)
    }

    pub(crate) fn has_enum_member(&self, subject: &str, member: &str, variant: &str) -> bool {
        self.graph.has_enum_member(subject, member, variant)
            || self
                .explicit_types
                .get(&format!("{subject}::{member}"))
                .and_then(|ty| ty.get_enum_def().ok())
                .and_then(|def| self.enums.native_member_names.get(&def))
                .is_some_and(|names| names.contains(variant))
    }

    /// Identify enum declarations and aliases without elaborating a skipped
    /// expression.
    pub(crate) fn is_enum_type(&self, subject: &str, member: &str) -> bool {
        matches!(
            self.graph.named(subject, member),
            Some(ResolvedShape::Enum(_))
        ) || self
            .explicit_types
            .get(&format!("{subject}::{member}"))
            .is_some_and(dslx::Type::is_enum)
    }

    /// Identify a struct declaration or alias without elaborating a call.
    pub(crate) fn is_struct_type(&self, subject: &str, member: &str) -> bool {
        matches!(
            self.graph.named(subject, member),
            Some(ResolvedShape::Struct(_))
        ) || self
            .explicit_types
            .get(&format!("{subject}::{member}"))
            .is_some_and(dslx::Type::is_struct)
    }

    #[cfg(test)]
    pub(crate) fn type_witnesses(&self) -> usize {
        self.type_witnesses
    }

    pub(crate) fn definitions(&self) -> &[StructDef] {
        &self.definitions
    }

    pub(crate) fn definition(&self, name: &str) -> Option<&StructDef> {
        self.named_layouts.get(name).map(|&i| &self.definitions[i])
    }

    pub(crate) fn enum_definitions(&self) -> &[CheckedEnum] {
        &self.enums.definitions
    }

    pub(crate) fn aggregate_spelling(&self, ty: &Type) -> Option<&str> {
        self.aggregate_spellings.get(ty).map(String::as_str)
    }

    pub(crate) fn value_spelling(&self, ty: &Type) -> Option<&str> {
        match ty {
            Type::Named(name) | Type::Enum { name, .. } => {
                self.value_spellings.get(name).map(String::as_str)
            }
            _ => None,
        }
    }

    /// Supply a typed expression for a probe operand that has no public name.
    pub(crate) fn aggregate_projection(&self, ty: &Type) -> Option<(&str, &str)> {
        let member = self.aggregate_projections.get(ty)?;
        Some((&member.alias, &member.suffix))
    }

    /// Index projections for private members without a snapshot spelling.
    /// A chain of array aliases can exceed the outline traversal limit while
    /// its concrete native layout still fits the BX type-depth limit.
    fn index_aggregate_members(
        &mut self,
        ty: &Type,
        alias: &Rc<str>,
        suffix: &mut String,
        reachable: bool,
    ) -> bool {
        let start = suffix.len();
        let private = match ty {
            Type::Named(name) | Type::Enum { name, .. } => {
                self.value_spelling(ty).is_none()
                    && name
                        .split_once("::")
                        .is_some_and(|(subject, _)| !self.graph.has_module(subject))
            }
            Type::Array { element, size } => {
                suffix.push_str("[u32:0]");
                let private =
                    self.index_aggregate_members(element, alias, suffix, reachable && *size > 0);
                suffix.truncate(start);
                private
            }
            Type::Tuple(fields) => {
                let mut private = false;
                for (i, field) in fields.iter().enumerate() {
                    suffix.push_str(&format!(".{i}"));
                    private |= self.index_aggregate_members(field, alias, suffix, reachable);
                    suffix.truncate(start);
                }
                private
            }
            _ => false,
        };
        if private && reachable && start != 0 && !self.aggregate_projections.contains_key(ty) {
            self.aggregate_projections.insert(
                ty.clone(),
                AggregateProjection {
                    alias: Rc::clone(alias),
                    suffix: suffix.clone(),
                },
            );
        }
        private
    }

    pub(crate) fn contains_parametric_instance(&self, ty: &Type) -> bool {
        match ty {
            Type::Named(name) => self
                .named_layouts
                .get(name)
                .is_some_and(|i| self.instance_layouts.contains(i)),
            Type::Array { element, .. } | Type::ParametricArray { element, .. } => {
                self.contains_parametric_instance(element)
            }
            Type::Tuple(members) => members.iter().any(|m| self.contains_parametric_instance(m)),
            Type::Applied { .. } => true,
            Type::Bits { .. }
            | Type::ParametricBits { .. }
            | Type::ParametricStruct { .. }
            | Type::TypeApplied { .. }
            | Type::Bool
            | Type::Enum { .. } => false,
        }
    }

    /// Names of public layouts are valid directly in a generated parameter.
    pub(crate) fn public_spelling<'a>(&'a self, name: &'a str) -> Result<&'a str, String> {
        if let Some((subject, member)) = name.split_once("::") {
            if let Some(index) = subject
                .strip_prefix("0bx_private_")
                .and_then(|s| s.parse::<usize>().ok())
            {
                // An explicitly requested generated/opaque type can expose a
                // previously unnamed layout without changing its BX identity.
                if let Some(path) = self
                    .bindings
                    .values()
                    .find(|binding| binding.layouts.get(&[][..]) == Some(&index))
                    .and_then(|binding| binding.public_name.as_deref())
                {
                    return Ok(path);
                }
                return Err(format!(
                    "cannot pass DSLX struct '{member}' to an imported function without a public type name"
                ));
            }
            if self.named_layouts.contains_key(name)
                || self.explicit_types.contains_key(name)
                || matches!(
                    self.graph.named(subject, member),
                    Some(ResolvedShape::Struct(_))
                )
            {
                return Ok(name);
            }
        }
        Err(format!(
            "unsupported nominal type '{name}' in imported function signature"
        ))
    }

    pub(crate) fn resolve_exported_type(
        &mut self,
        subject: &str,
        member: &str,
        data: &mut ImportData,
    ) -> Result<Type, String> {
        if member.starts_with("BXConstexprType")
            && self
                .graph
                .value_aliases
                .get(subject)
                .is_some_and(|aliases| aliases.values().any(|alias| alias == member))
        {
            return Err(format!("unknown DSLX type '{subject}::{member}'"));
        }
        if self.graph.has_public_member(subject, member)
            && self.graph.is_sum(subject, &[member.to_owned()])
        {
            return Err(format!(
                "BX supports numeric enums and '{subject}::{member}' is a DSLX sum enum"
            ));
        }
        self.reflect_spelling(subject, member, member, data)
    }

    #[cfg(test)]
    fn resolve_named(
        &mut self,
        subject: &str,
        member: &str,
        data: &mut ImportData,
    ) -> Result<String, String> {
        self.resolve_spelling(subject, member, member, data)
    }

    /// Supply typed literals so DSLX checks supplied bindings against u32 and
    /// evaluates defaults in the template's declaration context.
    pub(crate) fn resolve_applied(
        &mut self,
        subject: &str,
        member: &str,
        args: &[u32],
        data: &mut ImportData,
    ) -> Result<String, String> {
        let spelling = instance_spelling(member, args);
        self.resolve_spelling(subject, member, &spelling, data)
            .map_err(|e| format!("{subject}::{spelling}: {}", self.remap_diagnostic(e)))
    }

    /// Get the captured bindings when a layout belongs to the named template.
    pub(crate) fn struct_parametric_values(
        &self,
        canonical: &str,
        subject: &str,
        member: &str,
    ) -> Option<&[u32]> {
        let index = self.named_layouts.get(canonical)?;
        let instance = self.parametric_instances.get(index)?;
        (instance.template.subject == subject && instance.template.name == member)
            .then_some(instance.values.as_ref())
    }

    /// Reflect a named struct or a concrete application. The base declaration
    /// identifies the template, while DSLX evaluates the written arguments.
    fn resolve_spelling(
        &mut self,
        subject: &str,
        member: &str,
        spelling: &str,
        data: &mut ImportData,
    ) -> Result<String, String> {
        match self.reflect_spelling(subject, member, spelling, data)? {
            Type::Named(name) => Ok(name),
            _ => Err(format!("DSLX type '{subject}::{spelling}' is not a struct")),
        }
    }

    /// Reflect a publicly accessible type, preserving the identity of any
    /// structs and enums nested inside a transparent alias.
    fn reflect_spelling(
        &mut self,
        subject: &str,
        member: &str,
        spelling: &str,
        data: &mut ImportData,
    ) -> Result<Type, String> {
        self.bind_exports(data)?;
        let path = format!("{subject}::{spelling}");
        if let Some(&index) = self.named_layouts.get(&path) {
            return Ok(Type::Named(self.definitions[index].name.clone()));
        }
        if let Some(ty) = self.reflected_aliases.get(&path) {
            return Ok(ty.clone());
        }
        let witness = TypeWitness {
            imports: format!("import {subject} as bx_module;\n"),
            params: format!("value: bx_module::{spelling}"),
            expr: "value".into(),
        };
        let ty = if let Some(ty) = self.explicit_types.get(&path) {
            ty.clone()
        } else {
            // Resolve the exact requested alias or generated declaration. Its
            // target may have value bindings that require native evaluation.
            let source = format!("{}fn inspect({}) {{}}", witness.imports, witness.params);
            let probe = format!("0bx_named_{}", self.probe_index);
            self.probe_index += 1;
            #[cfg(test)]
            {
                self.type_witnesses += 1;
            }
            let checked = self.check_probe(&source, &probe, data)?;
            let function = probe_function(&checked)?;
            let ty = checked
                .get_type_info()
                .get_type_for_type_annotation(&function.get_param(0).get_type_annotation())
                .ok_or_else(|| format!("could not resolve DSLX type '{path}'"))?;
            if let Ok(def) = ty.get_struct_def()
                && !def.is_parametric()
            {
                self.bindings
                    .entry(def)
                    .or_default()
                    .public_name
                    .get_or_insert_with(|| path.clone());
            }
            ty
        };
        let written = self.graph.named(subject, member);
        let reflected = self.reflect_at(&ty, data, &witness, written.as_ref(), 0)?;
        if let Type::Named(canonical) = &reflected {
            let index = self.named_layouts[canonical];
            self.named_layouts.insert(path, index);
        } else {
            if matches!(reflected, Type::Tuple(_) | Type::Array { .. })
                && !self.aggregate_spellings.contains_key(&reflected)
            {
                self.index_aggregate_members(
                    &reflected,
                    &Rc::from(path.as_str()),
                    &mut String::new(),
                    true,
                );
                self.aggregate_spellings
                    .insert(reflected.clone(), path.clone());
            }
            self.reflected_aliases.insert(path, reflected.clone());
        }
        Ok(reflected)
    }

    pub(crate) fn reflect_result(
        &mut self,
        subject: &str,
        function: &str,
        ty: &dslx::Type,
        data: &mut ImportData,
        witness: &TypeWitness,
    ) -> Result<Type, String> {
        let written = self.graph.return_type(subject, function);
        self.reflect_at(ty, data, witness, written.as_ref(), 0)
    }

    /// Walk the checked value type together with its written nominal leaves.
    fn reflect_at(
        &mut self,
        ty: &dslx::Type,
        data: &mut ImportData,
        witness: &TypeWitness,
        written: Option<&ResolvedShape>,
        depth: usize,
    ) -> Result<Type, String> {
        if depth > MAX_ELABORATION_DEPTH {
            return Err(format!(
                "DSLX import type nesting exceeds {MAX_ELABORATION_DEPTH} levels"
            ));
        }
        if ty.is_enum() || ty.is_struct() {
            let (reflected, id) = if ty.is_enum() {
                self.bind_exports(data)?;
                let id = match written {
                    Some(ResolvedShape::Enum(id)) => Some(id),
                    _ => None,
                };
                (self.reflect_enum(ty, data)?, id)
            } else {
                let id = match written {
                    Some(ResolvedShape::Struct(id)) => Some(id),
                    _ => None,
                };
                (
                    Type::Named(self.reflect_struct(ty, data, witness, id, depth)?),
                    id,
                )
            };
            if let Some(id) = id
                && let Some(alias) = self
                    .graph
                    .value_aliases
                    .get(&id.subject)
                    .and_then(|aliases| aliases.get(&id.name))
                && let Type::Named(name) | Type::Enum { name, .. } = &reflected
            {
                self.value_spellings
                    .entry(name.clone())
                    .or_insert_with(|| format!("{}::{alias}", id.subject));
            }
            return Ok(reflected);
        }
        if let Some((signed, width)) = ty.is_bits_like() {
            return Ok(Type::Bits { signed, width });
        }
        if ty.is_array() {
            let size = ty.get_array_size();
            let index = if size == 0 {
                "[_bx_array_index]"
            } else {
                "[u32:0]"
            };
            let element = match written {
                Some(ResolvedShape::Array(element)) => Some(element.as_ref()),
                _ => None,
            };
            let result = Type::array(
                self.reflect_at(
                    &ty.get_array_element_type(),
                    data,
                    &witness.project(index),
                    element,
                    depth + 1,
                )?,
                size,
            );
            self.bind_matching_aggregates(ty, data)?;
            return Ok(result);
        }
        let text = ty.to_string().map_err(|e| e.to_string())?;
        match parse_value_type(&text) {
            Ok(ty) => Ok(ty),
            Err(_) if text.starts_with('(') => {
                let index = self.probe_index;
                self.probe_index += 1;
                let items = witness.tuple_members(ty, index, data, self)?;
                let mut members = Vec::with_capacity(items.len());
                for (index, item) in items.iter().enumerate() {
                    let field = match written {
                        Some(ResolvedShape::Tuple(fields)) => fields.get(index),
                        _ => None,
                    };
                    members.push(self.reflect_at(
                        item,
                        data,
                        &witness.project(&format!(".{index}")),
                        field,
                        depth + 1,
                    )?);
                }
                self.bind_matching_aggregates(ty, data)?;
                Ok(Type::Tuple(members))
            }
            Err(e) => Err(format!("unsupported result type '{text}': {e}")),
        }
    }

    /// Resolve matching public aliases against their own native declarations.
    /// Equal type displays from different modules can contain distinct private
    /// records or enums, so the display never supplies a BX type identity.
    fn bind_matching_aggregates(
        &mut self,
        native: &dslx::Type,
        data: &mut ImportData,
    ) -> Result<(), String> {
        if !self.aggregate_candidates.is_empty() {
            let display = native.to_string().map_err(|e| e.to_string())?;
            self.bind_aggregate_display(&display, data)?;
        }
        Ok(())
    }

    /// Consume candidates before reflecting them, since walking an alias can
    /// encounter the same native aggregate.
    pub(crate) fn bind_aggregate_display(
        &mut self,
        display: &str,
        data: &mut ImportData,
    ) -> Result<(), String> {
        if let Some(exports) = self.aggregate_candidates.remove(display) {
            for export in exports {
                self.resolve_exported_type(&export.subject, &export.name, data)?;
            }
        }
        Ok(())
    }

    /// Use the Rust declaration's field order with DSLX's concrete field types.
    fn reflect_struct(
        &mut self,
        ty: &dslx::Type,
        data: &mut ImportData,
        witness: &TypeWitness,
        written: Option<&DeclId>,
        depth: usize,
    ) -> Result<String, String> {
        self.bind_exports(data)?;
        let def = ty.get_struct_def().map_err(|e| e.to_string())?;
        let identifier = def.get_identifier();
        self.bindings.entry(def.clone()).or_default();
        let (template, actuals) = if def.is_parametric() {
            let id = written.or(self.bindings[&def].declaration.as_ref())
                .filter(|id| self.public_struct_template_arity(&id.subject, &id.name).is_some())
                .cloned().ok_or_else(|| format!(
                    "cannot identify a public DSLX struct template for '{identifier}'; name the concrete struct or a public alias in the DSLX type annotation"
                ))?;
            let values = self.capture_struct_bindings(&id, &def, witness, data)?;
            self.bind_declaration(&def, &id)?;
            (Some(id), values)
        } else {
            if let Some(id) = written {
                self.bind_declaration(&def, id)?;
            }
            (None, Vec::new())
        };
        let binding = &self.bindings[&def];
        if let Some(&index) = binding.layouts.get(actuals.as_slice()) {
            return Ok(self.definitions[index].name.clone());
        }
        let path = if let Some(id) = &template {
            Some(format!(
                "{}::{}",
                id.subject,
                instance_spelling(&id.name, &actuals)
            ))
        } else {
            binding.public_name.clone()
        };
        let label = path.as_deref().unwrap_or(&identifier);
        let members = if let Some(id) = &binding.declaration
            && let Some(fields) = self.graph.fields(id)
        {
            if fields.len() != ty.get_struct_member_count() {
                return Err(format!(
                    "Rust and DSLX disagree on the number of fields in '{label}'"
                ));
            }
            fields
                .iter()
                .map(|(name, annotation)| {
                    (
                        name.clone(),
                        self.graph.resolve_type(&id.subject, annotation),
                    )
                })
                .collect::<Vec<_>>()
        } else {
            // A specialization can return a private or generated struct with
            // no written type reference. Its checked handle still identifies
            // the layout for projection and keeps it distinct from other types.
            (0..ty.get_struct_member_count())
                .map(|i| (def.get_member(i).get_name(), ResolvedShape::Other))
                .collect()
        };
        let mut fields = Vec::with_capacity(members.len());
        for (index, (name, shape)) in members.into_iter().enumerate() {
            if name != def.get_member(index).get_name() {
                return Err(format!(
                    "Rust and DSLX disagree on field {index} of '{label}'"
                ));
            }
            let ty = self
                .reflect_at(
                    &ty.get_struct_member_type(index),
                    data,
                    &witness.project(&format!(".{name}")),
                    Some(&shape),
                    depth + 1,
                )
                .map_err(|e| format!("DSLX struct field '{label}.{name}': {e}"))?;
            fields.push(Param { name, ty });
        }
        let index = self.definitions.len();
        let public = path.is_some();
        let name = path.unwrap_or_else(|| format!("0bx_private_{index}::{identifier}"));
        let actuals: Rc<[u32]> = actuals.into();
        if let Some(template) = template {
            self.parametric_instances.insert(
                index,
                ImportedStructInstance {
                    template,
                    values: Rc::clone(&actuals),
                },
            );
        }
        self.bindings
            .get_mut(&def)
            .expect("registered native declaration")
            .layouts
            .insert(actuals, index);
        self.named_layouts.insert(name.clone(), index);
        if def.is_parametric()
            || fields
                .iter()
                .any(|f| self.contains_parametric_instance(&f.ty))
        {
            self.instance_layouts.insert(index);
        }
        self.definitions.push(StructDef {
            type_params: Vec::new(),
            name: name.clone(),
            public,
            value_parametrics: Vec::new(),
            fields,
        });
        Ok(name)
    }
}

/// Render value arguments in the type spelling used for native witnesses and
/// canonical BX struct identities.
fn instance_spelling(member: &str, values: &[u32]) -> String {
    let values = values
        .iter()
        .map(|v| format!("u32:{v}"))
        .collect::<Vec<_>>();
    format!("{member}<{}>", values.join(", "))
}

/// The declaration/explicit-type witnesses each contain one function.
pub(crate) fn probe_function(checked: &dslx::TypecheckedModule) -> Result<dslx::Function, String> {
    let module = checked.get_module();
    (0..module.get_member_count())
        .find_map(|i| match module.get_member(i).to_matchable() {
            Some(MatchableModuleMember::Function(f)) => Some(f),
            _ => None,
        })
        .ok_or_else(|| "missing DSLX type witness".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn load(uses: &str, paths: &[&Path]) -> Result<(ImportedStructs, ImportData), String> {
        let module = crate::parse_module(uses)?;
        let (registry, data) =
            ImportedStructs::load(&module.uses, xlsynth::default_dslx_stdlib_path(), paths)?;
        Ok((registry, data.ok_or("expected imported modules")?))
    }

    #[test]
    fn concrete_alias_reuses_its_layout_and_inferred_defaults()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        fs::write(
            root.path().join("wire.x"),
            r#"pub struct Tagged<A: u32, B: u32 = {A + A}> { tag: u8 }
pub type Short = Tagged<u32:3>;
pub type Full = Tagged<u32:3, u32:6>;"#,
        )?;
        let (mut registry, mut data) = load("use wire;", &[root.path()])?;
        let expected = "wire::Tagged<u32:3, u32:6>";
        for alias in ["Short", "Full"] {
            assert_eq!(registry.resolve_named("wire", alias, &mut data)?, expected);
        }
        let witnesses = registry.type_witnesses();
        for alias in ["Short", "Full"] {
            assert_eq!(registry.resolve_named("wire", alias, &mut data)?, expected);
        }
        assert_eq!(registry.type_witnesses(), witnesses);
        assert_eq!(registry.definitions().len(), 1);
        Ok(())
    }

    #[test]
    fn unsupported_templates_only_affect_references_to_their_instances()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        fs::write(
            root.path().join("formats.x"),
            r#"pub struct Format<N: u32> { value: uN[N] }
pub type Byte = Format<8>;
pub struct Small<N: u8> { value: u8 }
pub type NarrowBinding = Small<u8:2>;
struct Hidden<N: u32> { value: u8 }
pub type HiddenBinding = Hidden<2>;"#,
        )?;
        for unsupported in ["NarrowBinding", "HiddenBinding"] {
            let (mut registry, mut data) = load("use formats;", &[root.path()])?;
            assert_eq!(
                registry.resolve_named("formats", "Byte", &mut data)?,
                "formats::Format<u32:8>"
            );
            assert!(
                registry
                    .resolve_named("formats", unsupported, &mut data)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn attributed_struct_visibility_comes_from_dslx() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let source = r#"#[sv_type("hidden\npub struct Decoy")]
struct Hidden { tag: u8 }
#[sv_type("pkg::visible_t")]
pub struct Visible { tag: u8 }
pub struct Frame { hidden: Hidden, visible: Visible }"#;
        let path = root.path().join("wire.x");
        fs::write(&path, source)?;
        let paths = [root.path()];
        let (mut registry, mut data) = load("use wire;", &paths)?;
        registry.resolve_named("wire", "Frame", &mut data)?;
        let [hidden, visible, _frame] = registry.definitions() else {
            panic!("expected two nested structs and their frame")
        };
        assert!(registry.public_spelling(&hidden.name).is_err());
        assert_eq!(registry.public_spelling(&visible.name)?, "wire::Visible");
        Ok(())
    }

    #[test]
    fn snapshot_declaration_precedes_aliases_added_on_disk()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let noise = root.path().join("a_noise.x");
        let wire = root.path().join("wire.x");
        let noise_source = "struct Leaf { x: u8 }";
        let wire_source = "pub struct Leaf { x: u8 } pub struct Frame { leaf: Leaf }";
        fs::write(&noise, noise_source)?;
        fs::write(&wire, wire_source)?;
        let paths = [root.path()];
        let (mut registry, mut data) = load("use a_noise; use wire;", &paths)?;
        fs::write(noise, format!("{noise_source} pub type A = Leaf;"))?;
        fs::write(wire, format!("{wire_source} pub type A = Leaf;"))?;
        registry.resolve_named("wire", "Frame", &mut data)?;
        let [leaf, _frame] = registry.definitions() else {
            panic!("expected a nested leaf and its frame")
        };
        assert_eq!(registry.public_spelling(&leaf.name)?, "wire::Leaf");
        Ok(())
    }

    #[test]
    fn nested_snapshot_struct_keeps_its_name_when_the_export_becomes_a_different_type()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let wire = root.path().join("wire.x");
        let paths = [root.path()];
        for replacement in [
            "pub enum Leaf : u8 { ZERO = 0 }",
            "pub type Leaf = u32;",
            "pub enum Leaf { None, Some(u8), }",
        ] {
            fs::write(
                &wire,
                "pub struct Leaf { x: u8 } pub struct Frame { leaf: Leaf }",
            )?;
            let (mut registry, mut data) = load("use wire;", &paths)?;
            fs::write(&wire, replacement)?;

            registry.resolve_named("wire", "Frame", &mut data)?;
            let [leaf, _frame] = registry.definitions() else {
                panic!("expected the original Leaf and Frame layouts")
            };
            let leaf_name = leaf.name.clone();
            assert_eq!(
                registry.public_spelling(&leaf_name)?,
                "wire::Leaf",
                "{replacement}"
            );
        }
        Ok(())
    }

    #[test]
    fn multiple_named_structs_use_the_same_source_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let path = root.path().join("wire.x");
        fs::write(
            &path,
            "pub struct First { x: u8 } pub struct Second { x: u8 }",
        )?;
        let paths = [root.path()];
        let (mut registry, mut data) = load("use wire;", &paths)?;
        fs::remove_file(path)?;
        let first = registry.resolve_named("wire", "First", &mut data)?;
        let second = registry.resolve_named("wire", "Second", &mut data)?;
        assert_ne!(first, second);
        Ok(())
    }

    #[test]
    fn a_public_alias_keeps_a_projected_structs_identity_after_source_removal()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let source = root.path().join("wire.x");
        fs::write(
            &source,
            "struct Internal { x: u8 } pub struct Outer { inner: Internal } pub type Payload = Internal;",
        )?;
        let paths = [root.path()];
        let (mut registry, mut data) = load("use wire;", &paths)?;
        fs::remove_file(source)?;
        registry.resolve_named("wire", "Outer", &mut data)?;
        let Type::Named(inner) = registry.definitions()[1].fields[0].ty.clone() else {
            panic!("expected a nominal field");
        };
        assert_eq!(registry.public_spelling(&inner)?, "wire::Payload");
        assert_eq!(registry.resolve_named("wire", "Payload", &mut data)?, inner);
        assert_eq!(registry.public_spelling(&inner)?, "wire::Payload");
        assert!(
            registry
                .resolve_named("wire", "Internal", &mut data)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn discovery_uses_dslxs_stripped_path_for_a_single_component_import()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        fs::write(
            root.path().join(".x"),
            "pub struct Leaf { tag: u8 } pub struct Frame { leaf: Leaf }",
        )?;
        let paths = [root.path()];
        let (mut registry, mut data) = load("use wire;", &paths)?;
        registry.resolve_named("wire", "Frame", &mut data)?;
        let [leaf, _frame] = registry.definitions() else {
            panic!("expected a nested leaf and its frame")
        };
        assert_eq!(registry.public_spelling(&leaf.name)?, "wire::Leaf");
        Ok(())
    }

    #[test]
    fn type_only_reexport_keeps_its_nested_structs_defining_module()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let shadow = tempdir()?;
        fs::write(
            shadow.path().join("facade.x"),
            "pub struct Header { wrong: u32 }",
        )?;
        fs::write(
            root.path().join("fields.x"),
            "pub struct Leaf { x: u8 } pub struct Outer { nested: Leaf, empty: (Leaf, bool)[0] }",
        )?;
        fs::write(
            root.path().join("wire.x"),
            r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use fields::Outer;
pub type Header = Outer;"#,
        )?;
        fs::write(
            root.path().join("facade.x"),
            "import wire; pub type Header = wire::Header;",
        )?;
        let paths = [root.path(), shadow.path()];
        let (mut registry, mut data) = load("use facade;", &paths)?;
        registry.resolve_named("facade", "Header", &mut data)?;
        let [leaf, outer] = registry.definitions() else {
            panic!("expected nested and outer struct declarations")
        };
        assert_eq!(leaf.name, "fields::Leaf");
        assert_eq!(
            outer.fields,
            [
                Param {
                    name: "nested".into(),
                    ty: Type::Named(leaf.name.clone())
                },
                Param {
                    name: "empty".into(),
                    ty: Type::Array {
                        element: Box::new(Type::Tuple(vec![
                            Type::Named(leaf.name.clone()),
                            Type::Bits {
                                signed: false,
                                width: 1
                            }
                        ])),
                        size: 0,
                    },
                },
            ]
        );
        Ok(())
    }
    #[test]
    fn a_tuple_field_preserves_an_enums_identity_even_if_its_name_looks_like_bits()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        fs::write(
            root.path().join("ambig.x"),
            "pub enum u84 : u8 { Z = 0 } pub struct Envelope { item: (u84, u8) }",
        )?;
        let paths = [root.path()];
        let (mut registry, mut data) = load("use ambig;", &paths)?;
        registry.resolve_named("ambig", "Envelope", &mut data)?;
        assert_eq!(
            registry.definitions()[0].fields[0].ty,
            Type::Tuple(vec![
                Type::Enum {
                    name: "ambig::u84".into(),
                    signed: false,
                    width: 8
                },
                Type::Bits {
                    signed: false,
                    width: 8
                }
            ])
        );
        Ok(())
    }

    #[test]
    fn enum_aliases_use_the_owner_module_values_and_nominal_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        fs::create_dir(dir.path().join("pkg"))?;
        fs::write(
            dir.path().join("pkg/flags.x"),
            r#"
const HIGH = sN[129]:-2;
enum Hidden : sN[129] { Clear = 0, Set = HIGH }
pub type Mode = Hidden;
pub enum Empty : sN[33] {}
"#,
        )?;
        fs::write(
            dir.path().join("forward.x"),
            "import pkg.flags; pub type Forward = flags::Mode;",
        )?;
        let (mut registry, mut data) = load("use forward; use pkg::flags;", &[dir.path()])?;
        let first = registry.resolve_exported_type("forward", "Forward", &mut data)?;
        let second = registry.resolve_exported_type("pkg.flags", "Mode", &mut data)?;
        assert_eq!(first, second);
        let def = &registry.enum_definitions()[0];
        assert!(def.signed);
        assert_eq!(def.width, 129);
        assert_eq!(
            def.members["Set"].to_le_bytes(),
            [vec![254], vec![255; 15], vec![1]].concat()
        );
        assert_eq!(
            registry.resolve_exported_type("pkg.flags", "Empty", &mut data)?,
            Type::Enum {
                name: "pkg.flags::Empty".into(),
                signed: true,
                width: 33
            }
        );
        Ok(())
    }
}
