// SPDX-License-Identifier: Apache-2.0

//! Preserve DSLX nominal type identity when importing structs into BX.

mod runfiles;

use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use xlsynth::dslx::{self, ImportData, MatchableModuleMember, ModuleMemberKind};

use crate::ast::{Param, StructDef, Type};
use crate::check::MAX_ELABORATION_DEPTH;
use crate::dslx_import::{TypeWitness, check_probe};
use crate::parser::{is_dslx_member_name, parse_value_type};

/// A DSLX declaration's BX identity and an optional name usable in DSLX probes.
struct StructIdentity {
    declaration: dslx::StructDef,
    name: String,
    public_path: Option<String>,
}

/// Structs reflected in one DSLX import graph, with public spellings checked
/// by that graph's typechecker. Definitions are appended in dependency order.
pub(crate) struct ImportedStructs {
    subjects: BTreeSet<String>,
    public_types: BTreeMap<String, dslx::Type>,
    rejected_candidates: BTreeSet<String>,
    aliases: BTreeSet<String>,
    observed_modules: BTreeSet<String>,
    identities: Vec<StructIdentity>,
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
            aliases: BTreeSet::new(),
            observed_modules: BTreeSet::new(),
            identities: Vec::new(),
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
        let identity = self
            .identities
            .iter()
            .find(|id| id.name == name)
            .ok_or_else(|| {
                format!("unsupported nominal type '{name}' in imported function signature")
            })?;
        identity.public_path.as_deref().ok_or_else(|| {
            format!(
                "cannot pass DSLX struct '{}' to an imported function without a public type name",
                identity.declaration.get_identifier()
            )
        })
    }

    /// Discover transitive subjects and type aliases from the actual callee's
    /// owning module. A candidate spelling is only used after DSLX resolves it
    /// to the exact struct declaration.
    pub(crate) fn observe_module(
        &mut self,
        module: &dslx::Module,
        checked: &dslx::TypecheckedModule,
    ) {
        let subject = module.get_name();
        if self.observed_modules.insert(subject.clone()) {
            self.observe_members(&subject, module, checked);
        }
    }

    fn observe_members(
        &mut self,
        subject: &str,
        module: &dslx::Module,
        checked: &dslx::TypecheckedModule,
    ) {
        self.subjects.insert(subject.to_owned());
        let context = checked.get_type_info_for_module(module);
        let mut has_uses = false;
        for index in 0..module.get_member_count() {
            let member = module.get_member(index);
            if member.kind() == ModuleMemberKind::Use {
                has_uses = true;
            } else if let Some(import) = member.to_import() {
                self.subjects.insert(import.get_subject().join("."));
            } else if let Some(MatchableModuleMember::TypeAlias(alias)) = member.to_matchable() {
                let annotation = alias.get_type_annotation();
                if context
                    .as_ref()
                    .and_then(|info| info.get_type_for_type_annotation(&annotation))
                    .is_some_and(|ty| ty.is_struct())
                {
                    self.aliases
                        .insert(format!("{subject}::{}", alias.get_identifier()));
                    self.observe_annotation(annotation, 0);
                }
            }
        }
        if has_uses {
            for tree in module
                .to_text()
                .lines()
                .filter_map(|line| line.strip_prefix("use ")?.strip_suffix(';'))
            {
                self.observe_use_tree(tree);
            }
        }
    }

    /// libxls prints each checked use tree on one line but does not expose
    /// its leaves. A leaf can select a whole module or a member of its parent.
    /// Keep both spellings as candidates for the original graph to verify.
    fn observe_use_tree(&mut self, tree: &str) {
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
                    self.aliases.insert(format!("{subject}::{member}"));
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

    /// Resolve a source type or constructor spelling, preserving DSLX's own
    /// visibility and alias resolution.
    pub(crate) fn resolve_named(
        &mut self,
        subject: &str,
        member: &str,
        data: &mut ImportData,
    ) -> Result<String, String> {
        if !is_dslx_member_name(member) {
            return Err(format!("invalid DSLX struct name '{member}'"));
        }
        let name = format!("{subject}::{member}");
        let ty = self.public_type(&name, data)?;
        if !ty.is_struct() {
            return Err(format!("DSLX type '{name}' is not a struct"));
        }
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
        self.public_types.insert(name.to_owned(), ty.clone());
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
        let mut trial = self.new_import_data();
        if check_probe(&Self::type_probe_source(name), "0bx_type_trial", &mut trial).is_err() {
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

    /// Read concrete value types. A tuple's displayed spelling can describe
    /// structural leaves, but only native types can identify struct elements.
    pub(crate) fn reflect(
        &mut self,
        ty: &dslx::Type,
        data: &mut ImportData,
        witness: &TypeWitness,
    ) -> Result<Type, String> {
        self.reflect_at(ty, data, witness, 0)
    }

    fn reflect_at(
        &mut self,
        ty: &dslx::Type,
        data: &mut ImportData,
        witness: &TypeWitness,
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
                element: Box::new(self.reflect_at(
                    &ty.get_array_element_type(),
                    data,
                    &witness.project(index),
                    depth + 1,
                )?),
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
                let members = items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        self.reflect_at(item, data, &witness.project(&format!(".{i}")), depth + 1)
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
        if let Some(identity) = self
            .identities
            .iter_mut()
            .find(|id| id.declaration.is_same_definition(&def))
        {
            if identity.public_path.is_none() {
                identity.public_path = spelling.map(str::to_owned);
            }
            return Ok(identity.name.clone());
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
            self.observe_annotation(member.get_type(), 0);
            let ty = self
                .reflect_at(
                    &ty.get_struct_member_type(index),
                    data,
                    &witness.project(&format!(".{field}")),
                    depth + 1,
                )
                .map_err(|e| format!("DSLX struct field '{display_name}.{field}': {e}"))?;
            fields.push(Param { name: field, ty });
        }
        // The numeric prefix cannot be written as a BX or DSLX use binding.
        // Assign the ID after visiting the fields so nested declarations are
        // each interned with their own index.
        let name = public_path
            .clone()
            .unwrap_or_else(|| format!("0bx_private_{}::{identifier}", self.identities.len()));
        self.definitions.push(StructDef {
            name: name.clone(),
            public: public_path.is_some(),
            fields,
        });
        self.identities.push(StructIdentity {
            declaration: def,
            name: name.clone(),
            public_path,
        });
        Ok(name)
    }

    /// Find an addressable name for an inferred type without conflating a
    /// same-named declaration in a different module.
    fn public_path(
        &mut self,
        def: &dslx::StructDef,
        data: &mut ImportData,
    ) -> Result<Option<String>, String> {
        let identifier = def.get_identifier();
        let mut attempted = BTreeSet::new();
        loop {
            let paths: Vec<_> = self
                .subjects
                .iter()
                .map(|subject| format!("{subject}::{identifier}"))
                .chain(self.aliases.iter().cloned())
                .filter(|name| attempted.insert(name.clone()))
                .collect();
            for path in paths {
                if let Some(ty) = self.candidate_type(&path, data)?
                    && let Ok(candidate) = ty.get_struct_def()
                    && candidate.is_same_definition(def)
                {
                    return Ok(Some(path));
                }
            }
            let Some((subject, path)) = self.next_type_only_source() else {
                return Ok(None);
            };
            self.inspected_sources
                .insert((subject.clone(), path.clone()));
            self.discover_type_only_imports(&subject, &path);
        }
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
                let tail: PathBuf = relative.components().skip(1).collect();
                for suffix in
                    std::iter::once(relative).chain((!tail.as_os_str().is_empty()).then_some(tail))
                {
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
        let mut discovery = self.new_import_data();
        if let Ok(checked) = dslx::parse_and_typecheck(&source, filename, subject, &mut discovery) {
            self.observe_members(subject, &checked.get_module(), &checked);
        }
    }

    /// Record qualified names referenced by an actual field or alias, even
    /// when that module is only imported transitively by DSLX.
    pub(crate) fn observe_annotation(&mut self, annotation: dslx::TypeAnnotation, depth: usize) {
        if depth > MAX_ELABORATION_DEPTH {
            return;
        }
        if let Some(reference) = annotation.to_type_ref_type_annotation() {
            let definition = reference.get_type_ref().get_type_definition();
            if let Some(colon) = definition.to_colon_ref()
                && let Some(import) = colon.resolve_import_subject()
            {
                let subject = import.get_subject().join(".");
                self.aliases
                    .insert(format!("{subject}::{}", colon.get_attr()));
                self.subjects.insert(subject);
            } else if let Some(alias) = definition.to_type_alias() {
                self.observe_annotation(alias.get_type_annotation(), depth + 1);
            }
        } else if let Some(array) = annotation.to_array_type_annotation() {
            self.observe_annotation(array.get_element_type(), depth + 1);
        } else if let Some(members) = annotation.tuple_members() {
            for member in members {
                self.observe_annotation(member, depth + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
