// SPDX-License-Identifier: Apache-2.0

//! Preserve DSLX nominal type identity when importing structs into BX.

mod annotation;
mod runfiles;

use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use xlsynth::dslx::{self, ImportData, MatchableModuleMember, ModuleMemberKind};

use crate::ast::{Param, StructDef, Type};
use crate::check::MAX_ELABORATION_DEPTH;
use crate::dslx_import::{TypeWitness, check_probe};
use crate::parser::{is_dslx_member_name, parse_value_type};

/// The first public spelling and the layout, if already reflected, of one
/// declaration in the authoritative DSLX import graph.
#[derive(Default)]
struct StructIdentity {
    public_path: Option<String>,
    layout: Option<usize>,
}

/// Only handles obtained from the authoritative graph can establish identity.
/// A separately inspected source can supply candidate spellings for that graph.
#[derive(Clone, Copy, PartialEq)]
enum ModuleOrigin {
    Authoritative,
    Discovery,
}

/// Structs reflected in one DSLX import graph, with public spellings checked
/// by that graph's typechecker. Definitions are appended in dependency order.
pub(crate) struct ImportedStructs {
    subjects: BTreeSet<String>,
    public_types: BTreeMap<String, dslx::Type>,
    rejected_candidates: BTreeSet<String>,
    /// Potential type exports indexed by their member name and, for structs,
    /// by the underlying declaration's identifier. The authoritative graph
    /// must confirm visibility and identity.
    candidates: BTreeMap<String, BTreeSet<String>>,
    /// A use leaf in the checked graph can still name the original type if
    /// the declaration behind that spelling changes on disk.
    checked_use_members: BTreeSet<String>,
    /// Some native type declarations expose their kind but no member name
    /// through the pinned module-member API.
    opaque_type_subjects: BTreeSet<String>,
    /// Struct aliases and type names selected by checked uses can remain
    /// addressable even if their underlying declaration changes on disk.
    discovered_exports: BTreeSet<String>,
    discovery: Option<ImportData>,
    #[cfg(test)]
    candidate_trials: usize,
    observed_modules: BTreeSet<String>,
    identities: HashMap<dslx::StructDef, StructIdentity>,
    declarations_by_name: BTreeMap<String, dslx::StructDef>,
    definitions: Vec<StructDef>,
    probe_index: usize,
    stdlib: PathBuf,
    search_paths: Vec<PathBuf>,
    inspected_sources: BTreeSet<(String, PathBuf)>,
    runfiles: OnceCell<runfiles::XlsRunfiles>,
}

impl ImportedStructs {
    pub(crate) fn new(
        subjects: &BTreeSet<String>,
        stdlib_path: &Path,
        search_paths: &[&Path],
    ) -> Self {
        Self {
            subjects: subjects.clone(),
            public_types: BTreeMap::new(),
            rejected_candidates: BTreeSet::new(),
            candidates: BTreeMap::new(),
            checked_use_members: BTreeSet::new(),
            opaque_type_subjects: BTreeSet::new(),
            discovered_exports: BTreeSet::new(),
            discovery: None,
            #[cfg(test)]
            candidate_trials: 0,
            observed_modules: BTreeSet::new(),
            identities: HashMap::new(),
            declarations_by_name: BTreeMap::new(),
            definitions: Vec::new(),
            probe_index: 0,
            stdlib: stdlib_path.to_owned(),
            search_paths: search_paths.iter().map(|path| (*path).to_owned()).collect(),
            inspected_sources: BTreeSet::new(),
            runfiles: OnceCell::new(),
        }
    }

    pub(crate) fn definitions(&self) -> &[StructDef] {
        &self.definitions
    }

    /// A generated DSLX parameter must name its type, even when BX first
    /// encountered that type by projecting a private field from a public value.
    pub(crate) fn public_spelling(&self, name: &str) -> Result<&str, String> {
        let declaration = self.declarations_by_name.get(name).ok_or_else(|| {
            format!("unsupported nominal type '{name}' in imported function signature")
        })?;
        self.identities[declaration].public_path.as_deref().ok_or_else(|| {
            format!(
                "cannot pass DSLX struct '{}' to an imported function without a public type name",
                declaration.get_identifier()
            )
        })
    }

    /// When a generated parameter requires a name that the catalogue couldn't
    /// recover, try the checked use leaves and the original declaration name.
    pub(crate) fn parameter_spelling(
        &mut self,
        name: &str,
        data: &mut ImportData,
    ) -> Result<&str, String> {
        if let Some(declaration) = self.declarations_by_name.get(name)
            && self.identities[declaration].public_path.is_none()
        {
            let declaration = declaration.clone();
            let identifier = declaration.get_identifier();
            let candidates = self
                .checked_use_members
                .iter()
                .cloned()
                .chain(
                    self.subjects
                        .iter()
                        .map(|subject| format!("{subject}::{identifier}"))
                        .filter(|path| !self.checked_use_members.contains(path)),
                )
                .collect();
            self.match_public_declaration(candidates, &declaration, data)?;
        }
        self.public_spelling(name)
    }

    #[cfg(test)]
    pub(crate) fn discovery_counts(&self) -> (usize, usize) {
        (self.candidate_trials, self.inspected_sources.len())
    }

    /// Record a name and its resolved type together, before walking its fields.
    /// A later alias can make an existing private layout publicly nameable.
    fn record_annotation_path(&mut self, ty: &dslx::Type, path: String) {
        if let Ok(declaration) = ty.get_struct_def() {
            self.identities
                .entry(declaration)
                .or_default()
                .public_path
                .get_or_insert_with(|| path.clone());
            self.public_types.entry(path).or_insert_with(|| ty.clone());
        }
    }

    /// Index public structs and aliases once from the actual callee's module
    /// and its type information, preserving native declaration identity.
    pub(crate) fn observe_module(
        &mut self,
        module: &dslx::Module,
        checked: &dslx::TypecheckedModule,
    ) {
        let subject = module.get_name();
        if !self.observed_modules.contains(&subject)
            && let Some(info) = checked.get_type_info_for_module(module)
        {
            self.observed_modules.insert(subject.clone());
            self.observe_members(&subject, module, &info, ModuleOrigin::Authoritative);
        }
    }

    fn observe_members(
        &mut self,
        subject: &str,
        module: &dslx::Module,
        info: &dslx::TypeInfo,
        origin: ModuleOrigin,
    ) {
        self.subjects.insert(subject.to_owned());
        let mut has_uses = false;
        for index in 0..module.get_member_count() {
            let member = module.get_member(index);
            if origin == ModuleOrigin::Discovery
                && matches!(
                    member.kind(),
                    ModuleMemberKind::SumDef
                        | ModuleMemberKind::ProcDef
                        | ModuleMemberKind::ProcAlias
                )
            {
                self.opaque_type_subjects.insert(subject.to_owned());
                continue;
            }
            if member.kind() == ModuleMemberKind::Use {
                has_uses = true;
            } else if let Some(import) = member.to_import() {
                self.subjects.insert(import.get_subject().join("."));
            } else {
                let (name, ty, declaration) = match member.to_matchable() {
                    Some(MatchableModuleMember::StructDef(def)) => {
                        if origin == ModuleOrigin::Authoritative && def.is_parametric() {
                            continue;
                        }
                        (
                            def.get_identifier(),
                            info.get_type_for_struct_def(&def),
                            def.to_text(),
                        )
                    }
                    Some(MatchableModuleMember::TypeAlias(alias)) => {
                        let annotation = alias.get_type_annotation();
                        let ty = info.get_type_for_type_annotation(&annotation);
                        (alias.get_identifier(), ty, alias.to_text())
                    }
                    Some(MatchableModuleMember::EnumDef(def))
                        if origin == ModuleOrigin::Discovery =>
                    {
                        (def.get_identifier(), None, def.to_text())
                    }
                    _ => continue,
                };
                // The pinned API prints visibility but has no visibility
                // getter. An attribute precedes the header and
                // can contain arbitrary text. Verify such names
                // with the native typechecker on demand.
                let public = declaration.starts_with("pub ");
                if !public && !declaration.starts_with("#[") {
                    continue;
                }
                let path = format!("{subject}::{name}");
                if origin == ModuleOrigin::Discovery {
                    // The name can still resolve to a struct in the original
                    // graph even if its on-disk definition is now another type.
                    self.candidates
                        .entry(name)
                        .or_default()
                        .insert(path.clone());
                    if self.checked_use_members.contains(&path) {
                        self.discovered_exports.insert(path.clone());
                    }
                }
                let Some(ty) = ty else { continue };
                let Ok(def) = ty.get_struct_def() else {
                    continue;
                };
                if origin == ModuleOrigin::Authoritative && public {
                    self.record_annotation_path(&ty, path);
                } else {
                    if origin == ModuleOrigin::Discovery
                        && member.kind() == ModuleMemberKind::TypeAlias
                    {
                        self.discovered_exports.insert(path.clone());
                    }
                    self.candidates
                        .entry(def.get_identifier())
                        .or_default()
                        .insert(path);
                }
            }
        }
        if has_uses {
            for tree in module
                .to_text()
                .lines()
                .filter_map(|line| line.strip_prefix("use ")?.strip_suffix(';'))
            {
                self.observe_use_tree(tree, origin);
            }
        }
    }

    /// libxls prints each checked use tree on one line but does not expose
    /// its leaves. A leaf can select a whole module or a member of its parent.
    /// Keep both spellings as candidates for the original graph to verify.
    fn observe_use_tree(&mut self, tree: &str, origin: ModuleOrigin) {
        let mut prefix = Vec::new();
        let mut groups = Vec::new();
        for segment in tree.split_inclusive(['{', '}', ',']) {
            let previous = prefix.len();
            let path = segment
                .trim_end_matches(['{', '}', ','])
                .trim()
                .trim_end_matches("::");
            if !path.is_empty() {
                prefix.extend(path.split("::"));
            }
            if segment.ends_with('{') {
                groups.push(previous);
                continue;
            }
            if prefix.len() > previous
                && let Some((member, parents)) = prefix.split_last()
                && parents.iter().all(|part| is_dslx_member_name(part))
                && is_dslx_member_name(member)
            {
                self.subjects.insert(prefix.join("."));
                if !parents.is_empty() {
                    let subject = parents.join(".");
                    if origin == ModuleOrigin::Authoritative {
                        self.checked_use_members
                            .insert(format!("{subject}::{member}"));
                    }
                    self.subjects.insert(subject);
                }
            }
            prefix.truncate(previous);
            if segment.ends_with('}') {
                let Some(parent) = groups.pop() else { return };
                prefix.truncate(parent);
            }
        }
    }

    /// Bind a type written in BX to its original DSLX declaration without
    /// discovering or reflecting the layouts of its fields.
    pub(crate) fn bind_named(
        &mut self,
        subject: &str,
        member: &str,
        data: &mut ImportData,
    ) -> Result<dslx::Type, String> {
        if !is_dslx_member_name(member) {
            return Err(format!("invalid DSLX struct name '{member}'"));
        }
        let name = format!("{subject}::{member}");
        let ty = self.public_type(&name, data)?;
        if !ty.is_struct() {
            return Err(format!("DSLX type '{name}' is not a struct"));
        }
        Ok(ty)
    }

    /// Reflect a named type using the spelling authorized by the DSLX checker.
    pub(crate) fn resolve_named(
        &mut self,
        subject: &str,
        member: &str,
        data: &mut ImportData,
    ) -> Result<String, String> {
        let ty = self.bind_named(subject, member, data)?;
        let name = format!("{subject}::{member}");
        let witness = TypeWitness {
            imports: format!("import {subject} as bx_module;\n"),
            params: format!("value: bx_module::{member}"),
            expr: "value".into(),
        };
        self.reflect_struct(&ty, Some(&name), data, &witness, 0)
    }

    fn type_probe_source(name: &str) -> String {
        let (subject, member) = name.split_once("::").expect("qualified struct path");
        format!("import {subject} as bx_module;\nfn inspect(value: bx_module::{member}) {{}}")
    }

    /// Read the type and nominal identity of a public name in the shared graph.
    fn public_type(&mut self, name: &str, data: &mut ImportData) -> Result<dslx::Type, String> {
        if let Some(ty) = self.public_types.get(name) {
            return Ok(ty.clone());
        }
        let probe_name = format!("0bx_struct_{}", self.probe_index);
        self.probe_index += 1;
        let checked = check_probe(&Self::type_probe_source(name), &probe_name, data)?;
        let module = checked.get_module();
        let Some(function) = (0..module.get_member_count()).find_map(|i| {
            match module.get_member(i).to_matchable() {
                Some(MatchableModuleMember::Function(f)) => Some(f),
                _ => None,
            }
        }) else {
            return Err("missing DSLX type probe".into());
        };
        let ty = checked
            .get_type_info()
            .get_type_for_type_annotation(&function.get_param(0).get_type_annotation())
            .ok_or_else(|| format!("could not resolve DSLX type '{name}'"))?;
        self.record_annotation_path(&ty, name.to_owned());
        Ok(ty)
    }

    /// A speculative type name can denote a function or a constant. With
    /// libxls v0.61 a failed type annotation can crash the next probe on the
    /// same inference graph. Isolate the trial, then obtain the declaration's
    /// identity from the shared graph.
    fn candidate_type(
        &mut self,
        name: &str,
        data: &mut ImportData,
    ) -> Result<Option<dslx::Type>, String> {
        if let Some(ty) = self.public_types.get(name) {
            return Ok(Some(ty.clone()));
        }
        if self.rejected_candidates.contains(name) {
            return Ok(None);
        }
        #[cfg(test)]
        {
            self.candidate_trials += 1;
        }
        if self
            .check_discovery(&Self::type_probe_source(name), "0bx_type_trial.x")
            .is_none()
        {
            self.rejected_candidates.insert(name.to_owned());
            return Ok(None);
        }
        self.public_type(name, data).map(Some)
    }

    /// Create a separate typechecking graph with the resolver's import paths.
    fn new_import_data(&self) -> ImportData {
        let paths = self
            .search_paths
            .iter()
            .map(PathBuf::as_path)
            .collect::<Vec<_>>();
        ImportData::new(Some(&self.stdlib), &paths)
    }

    /// Reuse successful discovery work, including already loaded dependencies.
    /// Drop a failed graph before another probe can enter its inference state.
    fn check_discovery(&mut self, source: &str, filename: &str) -> Option<dslx::TypecheckedModule> {
        let mut graph = self
            .discovery
            .take()
            .unwrap_or_else(|| self.new_import_data());
        let name = format!("0bx_discovery_{}", self.probe_index);
        self.probe_index += 1;
        let checked = dslx::parse_and_typecheck(source, filename, &name, &mut graph).ok()?;
        self.discovery = Some(graph);
        Some(checked)
    }

    /// Read concrete value types. A tuple's displayed spelling can describe
    /// structural leaves, but only native types can identify struct elements.
    pub(crate) fn reflect(
        &mut self,
        ty: &dslx::Type,
        data: &mut ImportData,
        witness: &TypeWitness,
        annotation: Option<dslx::TypeAnnotation>,
    ) -> Result<Type, String> {
        self.reflect_at(ty, data, witness, annotation, 0)
    }

    fn reflect_at(
        &mut self,
        ty: &dslx::Type,
        data: &mut ImportData,
        witness: &TypeWitness,
        annotation: Option<dslx::TypeAnnotation>,
        depth: usize,
    ) -> Result<Type, String> {
        if depth > MAX_ELABORATION_DEPTH {
            return Err(format!(
                "DSLX import type nesting exceeds {MAX_ELABORATION_DEPTH} levels"
            ));
        }
        if ty.is_enum() {
            let definition = ty.get_enum_def().map_err(|e| e.to_string())?;
            return Err(format!(
                "unsupported result type '{}' (DSLX enum)",
                definition.get_identifier()
            ));
        }
        if let Some((signed, width)) = ty.is_bits_like() {
            return Ok(Type::Bits { signed, width });
        }
        let annotation = annotation.map(annotation::underlying_annotation);
        if let Some(written) = &annotation {
            self.observe_typed_annotation(written, ty);
        }
        if ty.is_struct() {
            return Ok(Type::Named(
                self.reflect_struct(ty, None, data, witness, depth)?,
            ));
        }
        if ty.is_array() {
            let size = ty.get_array_size();
            // The probe typechecks a projection without evaluating it. A
            // dynamic index also preserves the element type of an empty array.
            let index = if size == 0 {
                "[_bx_array_index]"
            } else {
                "[u32:0]"
            };
            return Ok(Type::Array {
                element: Box::new(
                    self.reflect_at(
                        &ty.get_array_element_type(),
                        data,
                        &witness.project(index),
                        annotation
                            .and_then(|a| a.to_array_type_annotation())
                            .map(|a| a.get_element_type()),
                        depth + 1,
                    )?,
                ),
                size,
            });
        }
        let text = ty.to_string().map_err(|e| e.to_string())?;
        match parse_value_type(&text) {
            Ok(ty) => Ok(ty),
            Err(_) if text.starts_with('(') => {
                let probe_index = self.probe_index;
                self.probe_index += 1;
                let items = witness.tuple_members(ty, probe_index, data)?;
                let mut written = annotation
                    .and_then(|a| a.tuple_members())
                    .filter(|members| members.len() == items.len())
                    .unwrap_or_default()
                    .into_iter();
                let members = items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        self.reflect_at(
                            item,
                            data,
                            &witness.project(&format!(".{i}")),
                            written.next(),
                            depth + 1,
                        )
                    })
                    .collect::<Result<_, _>>()?;
                Ok(Type::Tuple(members))
            }
            Err(e) => Err(format!("unsupported result type '{text}': {e}")),
        }
    }

    /// Intern the original declaration, so aliases share a BX type and two
    /// modules with identically named and shaped structs keep distinct types.
    fn reflect_struct(
        &mut self,
        ty: &dslx::Type,
        spelling: Option<&str>,
        data: &mut ImportData,
        witness: &TypeWitness,
        depth: usize,
    ) -> Result<String, String> {
        let def = ty.get_struct_def().map_err(|e| e.to_string())?;
        if def.is_parametric() {
            return Err(format!(
                "parametric DSLX struct '{}' is unsupported in BX",
                def.get_identifier()
            ));
        }
        if let Some(index) = self.identities.get(&def).and_then(|id| id.layout) {
            return Ok(self.definitions[index].name.clone());
        }
        let public_path = match spelling {
            Some(name) => Some(name.to_owned()),
            None => self.public_path(&def, data)?,
        };
        let identifier = def.get_identifier();
        let display_name = public_path.as_deref().unwrap_or(&identifier);
        let mut fields = Vec::with_capacity(ty.get_struct_member_count());
        for index in 0..ty.get_struct_member_count() {
            let member = def.get_member(index);
            let field = member.get_name();
            let written = member.get_type();
            let ty = self
                .reflect_at(
                    &ty.get_struct_member_type(index),
                    data,
                    &witness.project(&format!(".{field}")),
                    Some(written),
                    depth + 1,
                )
                .map_err(|e| format!("DSLX struct field '{display_name}.{field}': {e}"))?;
            fields.push(Param { name: field, ty });
        }
        // The numeric prefix cannot be written as a BX or DSLX use binding.
        // Assign the ID after visiting the fields so nested declarations are
        // each interned with their own index.
        let index = self.definitions.len();
        let name = public_path
            .clone()
            .unwrap_or_else(|| format!("0bx_private_{index}::{identifier}"));
        self.definitions.push(StructDef {
            name: name.clone(),
            public: public_path.is_some(),
            fields,
        });
        self.declarations_by_name.insert(name.clone(), def.clone());
        self.identities.entry(def).or_default().layout = Some(index);
        Ok(name)
    }

    /// Find an addressable name for an inferred type without conflating a
    /// same-named declaration in a different module.
    fn public_path(
        &mut self,
        def: &dslx::StructDef,
        data: &mut ImportData,
    ) -> Result<Option<String>, String> {
        if let Some(path) = self
            .identities
            .get(def)
            .and_then(|id| id.public_path.clone())
        {
            return Ok(Some(path));
        }
        let identifier = def.get_identifier();
        let mut attempted = BTreeSet::new();
        loop {
            // Prefer the original declaration name across all reachable
            // modules before trying aliases found in separately read sources.
            // A newly added alias may be absent from the import snapshot.
            let paths = self
                .candidates
                .get(&identifier)
                .into_iter()
                .flatten()
                .filter(|path| {
                    path.rsplit_once("::")
                        .is_some_and(|(_, name)| name == identifier)
                })
                .filter(|path| attempted.insert((*path).clone()))
                .cloned()
                .collect::<Vec<_>>();
            if let Some(path) = self.match_public_declaration(paths, def, data)? {
                return Ok(Some(path));
            }
            let Some((subject, path)) = self.next_type_only_source() else {
                break;
            };
            self.inspected_sources
                .insert((subject.clone(), path.clone()));
            self.discover_type_only_imports(&subject, &path);
        }
        // A source with opaque type members may still export the declaration
        // under its original name. Check that spelling before on-disk aliases.
        let opaque = self
            .opaque_type_subjects
            .iter()
            .map(|subject| format!("{subject}::{identifier}"))
            .filter(|path| attempted.insert(path.clone()))
            .collect();
        if let Some(path) = self.match_public_declaration(opaque, def, data)? {
            return Ok(Some(path));
        }
        // Prefer type exports named by the checked module, then aliases of a
        // same-named declaration, then other exports that may have been
        // retargeted on disk. All must resolve to the original identity.
        let paths = self
            .checked_use_members
            .intersection(&self.discovered_exports)
            .chain(self.candidates.get(&identifier).into_iter().flatten())
            .chain(&self.discovered_exports)
            .filter(|path| attempted.insert((*path).clone()))
            .cloned()
            .collect();
        self.match_public_declaration(paths, def, data)
    }

    /// Accept a spelling only when it resolves to the requested declaration
    /// in the graph that checked the BX import.
    fn match_public_declaration(
        &mut self,
        paths: Vec<String>,
        def: &dslx::StructDef,
        data: &mut ImportData,
    ) -> Result<Option<String>, String> {
        for path in paths {
            if let Some(ty) = self.candidate_type(&path, data)?
                && let Ok(candidate) = ty.get_struct_def()
                && candidate == *def
            {
                return Ok(Some(path));
            }
        }
        Ok(None)
    }

    /// Try reachable subjects under the configured search roots in order.
    /// A root may also contain a subject with its first component stripped.
    fn next_type_only_source(&self) -> Option<(String, PathBuf)> {
        // The native importer tries CWD and Bazel runfiles for both the
        // full and depot-root-stripped suffix before its explicit search roots.
        let roots = std::iter::once(None)
            .chain(std::iter::once(Some(self.stdlib.as_path())))
            .chain(self.search_paths.iter().map(|root| Some(root.as_path())));
        for root in roots {
            for subject in self.subjects.difference(&self.observed_modules) {
                let relative = PathBuf::from(subject.replace('.', "/")).with_extension("x");
                // libxls appends ".x" even when stripping the depot root
                // leaves an empty path for a single-component import.
                let tail = subject.split_once('.').map_or("", |(_, rest)| rest);
                let stripped = PathBuf::from(format!("{}.x", tail.replace('.', "/")));
                for suffix in [relative, stripped] {
                    let path = root.unwrap_or_else(|| Path::new("")).join(&suffix);
                    let runfile = std::iter::once_with(|| {
                        if root.is_some() {
                            return None;
                        }
                        self.runfiles
                            .get_or_init(runfiles::XlsRunfiles::from_env)
                            .lookup(&suffix)
                    });
                    let candidates = std::iter::once(path).chain(runfile.flatten());
                    for path in candidates {
                        let candidate = (subject.clone(), path);
                        if !self.inspected_sources.contains(&candidate) && candidate.1.is_file() {
                            return Some(candidate);
                        }
                    }
                }
            }
        }
        None
    }

    /// A type-only re-export may be the only path from BX to a nested struct.
    /// Released libxls exposes an import's subject but not its loaded module.
    /// Use a separate graph solely to discover possible qualified spellings.
    /// Visibility, identity and field types come from the original graph.
    fn discover_type_only_imports(&mut self, subject: &str, path: &Path) {
        let (Ok(source), Some(filename)) = (fs::read_to_string(path), path.to_str()) else {
            return;
        };
        if let Some(checked) = self.check_discovery(&source, filename) {
            self.observe_members(
                subject,
                &checked.get_module(),
                &checked.get_type_info(),
                ModuleOrigin::Discovery,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let paths = [root.path()];
        let mut data = ImportData::new(Some(stdlib), &paths);
        let checked = dslx::parse_and_typecheck(source, path.to_str().unwrap(), "wire", &mut data)?;
        let mut registry = ImportedStructs::new(&BTreeSet::from(["wire".into()]), stdlib, &paths);
        registry.observe_module(&checked.get_module(), &checked);
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
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let paths = [root.path()];
        let mut data = ImportData::new(Some(stdlib), &paths);
        check_probe("import a_noise; import wire;", "0bx_snapshot", &mut data)?;
        fs::write(noise, format!("{noise_source} pub type A = Leaf;"))?;
        fs::write(wire, format!("{wire_source} pub type A = Leaf;"))?;
        let subjects = BTreeSet::from(["a_noise".into(), "wire".into()]);
        let mut registry = ImportedStructs::new(&subjects, stdlib, &paths);
        registry.resolve_named("wire", "Frame", &mut data)?;
        let [leaf, _frame] = registry.definitions() else {
            panic!("expected a nested leaf and its frame")
        };
        assert_eq!(registry.public_spelling(&leaf.name)?, "wire::Leaf");
        assert_eq!(registry.discovery_counts().0, 1);
        Ok(())
    }

    #[test]
    fn nested_snapshot_struct_keeps_its_name_when_the_export_becomes_a_different_type()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let wire = root.path().join("wire.x");
        let stdlib = xlsynth::default_dslx_stdlib_path();
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
            let mut data = ImportData::new(Some(stdlib), &paths);
            check_probe("import wire;", "0bx_snapshot", &mut data)?;
            fs::write(&wire, replacement)?;

            let mut registry =
                ImportedStructs::new(&BTreeSet::from(["wire".into()]), stdlib, &paths);
            registry.resolve_named("wire", "Frame", &mut data)?;
            let [leaf, _frame] = registry.definitions() else {
                panic!("expected the original Leaf and Frame layouts")
            };
            let leaf_name = leaf.name.clone();
            assert_eq!(
                registry.parameter_spelling(&leaf_name, &mut data)?,
                "wire::Leaf",
                "{replacement}"
            );
        }
        Ok(())
    }

    #[test]
    fn fallback_probes_reuse_already_loaded_dependencies() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = tempdir()?;
        let path = root.path().join("wire.x");
        fs::write(
            &path,
            "pub struct First { x: u8 } pub struct Second { x: u8 }",
        )?;
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let paths = [root.path()];
        let mut data = ImportData::new(Some(stdlib), &paths);
        let mut registry = ImportedStructs::new(&BTreeSet::from(["wire".into()]), stdlib, &paths);
        let first = registry.candidate_type("wire::First", &mut data)?.unwrap();
        // Both graphs loaded wire for the first probe. Starting a fresh
        // discovery graph for the second would now fail to import it.
        fs::remove_file(path)?;
        let second = registry.candidate_type("wire::Second", &mut data)?.unwrap();
        assert!(first.get_struct_def()? != second.get_struct_def()?);
        Ok(())
    }

    #[test]
    fn a_later_public_spelling_keeps_a_projected_structs_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let source = root.path().join("wire.x");
        fs::write(
            &source,
            "struct Internal { x: u8 } pub struct Outer { inner: Internal } pub type Payload = Internal;",
        )?;
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let paths = [root.path()];
        let mut data = ImportData::new(Some(stdlib), &paths);
        check_probe("import wire;", "0bx_snapshot", &mut data)?;
        // The import graph has both public types. Without the source file the
        // registry first sees Internal solely through the Outer.inner field.
        fs::remove_file(source)?;
        let mut registry = ImportedStructs::new(&BTreeSet::from(["wire".into()]), stdlib, &paths);
        registry.resolve_named("wire", "Outer", &mut data)?;
        let Type::Named(inner) = registry.definitions()[1].fields[0].ty.clone() else {
            panic!("expected a nominal field");
        };
        assert!(registry.public_spelling(&inner).is_err());
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
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let paths = [root.path()];
        let mut data = ImportData::new(Some(stdlib), &paths);
        let mut registry = ImportedStructs::new(&BTreeSet::from(["wire".into()]), stdlib, &paths);
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
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let paths = [root.path(), shadow.path()];
        let mut data = ImportData::new(Some(stdlib), &paths);
        let mut registry =
            ImportedStructs::new(&BTreeSet::from(["facade".to_string()]), stdlib, &paths);
        registry.resolve_named("facade", "Header", &mut data)?;
        // Only the catalogued fields::Leaf declaration needs a trial.
        assert_eq!(registry.discovery_counts().0, 1);
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
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let paths = [root.path()];
        let mut data = ImportData::new(Some(stdlib), &paths);
        let mut registry =
            ImportedStructs::new(&BTreeSet::from(["ambig".to_owned()]), stdlib, &paths);
        let result = registry.resolve_named("ambig", "Envelope", &mut data);
        assert_eq!(
            result.unwrap_err(),
            "DSLX struct field 'ambig::Envelope.item': unsupported result type 'u84' (DSLX enum)"
        );
        Ok(())
    }
}
