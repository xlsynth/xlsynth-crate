// SPDX-License-Identifier: Apache-2.0

//! Bridge Rust's DSLX declaration identities to concrete types in libxls.

mod outline_graph;
mod runfiles;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use xlsynth::dslx::{self, ImportData, MatchableModuleMember};

use crate::ast::{Param, StructDef, Type, UsePath};
use crate::check::MAX_ELABORATION_DEPTH;
use crate::dslx_import::TypeWitness;
use crate::parser::parse_value_type;
use outline_graph::{DeclId, NativeSourceFiles, OutlineGraph, ResolvedShape};

/// The Rust declaration and BX layout associated with a checked native type.
#[derive(Default)]
struct StructBinding {
    declaration: Option<DeclId>,
    public_name: Option<String>,
    layout: Option<usize>,
}

/// A frozen module graph and its concrete struct types. Ordinary DSLX imports
/// check dependencies with the same rules as native DSLX callers.
pub(crate) struct ImportedStructs {
    graph: OutlineGraph,
    files: Option<NativeSourceFiles>,
    bindings: HashMap<dslx::StructDef, StructBinding>,
    types: BTreeMap<DeclId, dslx::Type>,
    explicit_types: BTreeMap<String, dslx::Type>,
    exports_bound: bool,
    #[cfg(test)]
    type_witnesses: usize,
    definitions: Vec<StructDef>,
    probe_index: usize,
}

impl ImportedStructs {
    /// Capture the reachable sources and validate the requested BX uses with
    /// native dependency-import semantics.
    pub(crate) fn load(
        uses: &[UsePath],
        stdlib: &Path,
        paths: &[&Path],
    ) -> Result<(Self, Option<ImportData>), String> {
        let graph = OutlineGraph::load_uses(uses, stdlib, paths)?;
        let files = (!uses.is_empty())
            .then(|| graph.native_source_files())
            .transpose()?;
        let mut data = files.as_ref().map(NativeSourceFiles::import_data);
        let registry = Self {
            graph,
            files,
            bindings: HashMap::new(),
            types: BTreeMap::new(),
            explicit_types: BTreeMap::new(),
            exports_bound: false,
            #[cfg(test)]
            type_witnesses: 0,
            definitions: Vec::new(),
            probe_index: 0,
        };
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

    /// Pair every addressable non-parametric struct and its aliases with DSLX
    /// declaration handles. A name here is an actual outlined export.
    fn bind_exports(&mut self, data: &mut ImportData) -> Result<(), String> {
        if self.exports_bound {
            return Ok(());
        }
        let exports: Vec<_> = self
            .graph
            .struct_exports()
            .map(|(name, id)| (name, Some(id)))
            .chain(self.graph.unresolved_aliases().map(|name| (name, None)))
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
            let Some(target) = target else {
                // This declared alias needs DSLX evaluation (e.g. an
                // associated type). Record any resulting public struct name.
                if let Ok(def) = ty.get_struct_def() {
                    let name = format!("{}::{}", export.subject, export.name);
                    self.bindings
                        .entry(def)
                        .or_default()
                        .public_name
                        .get_or_insert_with(|| name.clone());
                    self.explicit_types.insert(name, ty);
                }
                continue;
            };
            let def = ty.get_struct_def().map_err(|e| e.to_string())?;
            let binding = self.bindings.entry(def.clone()).or_default();
            if def.get_identifier() != target.name
                || binding.declaration.as_ref().is_some_and(|id| id != target)
                || self
                    .types
                    .get(target)
                    .is_some_and(|other| other.get_struct_def().ok().as_ref() != Some(&def))
            {
                return Err(format!(
                    "Rust and DSLX resolve '{}::{}' to different struct declarations",
                    export.subject, export.name
                ));
            }
            binding.declaration = Some(target.clone());
            binding.public_name = self.graph.public_path(target).map(str::to_owned);
            self.types.entry(target.clone()).or_insert(ty);
        }
        self.exports_bound = true;
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

    pub(crate) fn restore_ir_names(&self, ir: &str) -> String {
        self.files
            .as_ref()
            .expect("imported symbol has a source snapshot")
            .restore_ir_names(ir)
    }

    pub(crate) fn original_column(&self, path: &Path, line: usize, column: usize) -> usize {
        self.files
            .as_ref()
            .map_or(column, |files| files.original_column(path, line, column))
    }

    pub(crate) fn logical_path<'a>(&'a self, path: &'a Path) -> &'a Path {
        self.files
            .as_ref()
            .and_then(|f| f.logical_path(path))
            .unwrap_or(path)
    }

    pub(crate) fn has_module(&self, subject: &str) -> bool {
        self.graph.has_module(subject)
    }

    pub(crate) fn has_public_member(&self, subject: &str, member: &str) -> bool {
        self.graph.has_public_member(subject, member)
    }

    #[cfg(test)]
    pub(crate) fn type_witnesses(&self) -> usize {
        self.type_witnesses
    }

    pub(crate) fn definitions(&self) -> &[StructDef] {
        &self.definitions
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
                    .find(|binding| binding.layout == Some(index))
                    .and_then(|binding| binding.public_name.as_deref())
                {
                    return Ok(path);
                }
                return Err(format!(
                    "cannot pass DSLX struct '{member}' to an imported function without a public type name"
                ));
            }
            if self.explicit_types.contains_key(name)
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

    pub(crate) fn resolve_named(
        &mut self,
        subject: &str,
        member: &str,
        data: &mut ImportData,
    ) -> Result<String, String> {
        self.bind_exports(data)?;
        let path = format!("{subject}::{member}");
        let witness = TypeWitness {
            imports: format!("import {subject} as bx_module;\n"),
            params: format!("value: bx_module::{member}"),
            expr: "value".into(),
        };
        let ty = match self.graph.named(subject, member) {
            Some(ResolvedShape::Struct(id)) => self.types.get(&id).cloned().ok_or_else(|| {
                format!("DSLX type '{path}' is not a concrete non-parametric struct")
            })?,
            _ => {
                // A requested name can be generated by a DSLX attribute or
                // be an alias whose target requires type evaluation. Ask
                // about that exact user-written type, never a candidate name.
                if let Some(ty) = self.explicit_types.get(&path) {
                    ty.clone()
                } else {
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
                    let def = ty
                        .get_struct_def()
                        .map_err(|_| format!("DSLX type '{path}' is not a struct"))?;
                    self.bindings
                        .entry(def)
                        .or_default()
                        .public_name
                        .get_or_insert_with(|| path.clone());
                    self.explicit_types.insert(path, ty.clone());
                    ty
                }
            }
        };
        self.reflect_struct(&ty, data, &witness, None, 0)
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
        if ty.is_enum() {
            let def = ty.get_enum_def().map_err(|e| e.to_string())?;
            return Err(format!(
                "unsupported result type '{}' (DSLX enum)",
                def.get_identifier()
            ));
        }
        if let Some((signed, width)) = ty.is_bits_like() {
            return Ok(Type::Bits { signed, width });
        }
        if ty.is_struct() {
            let id = match written {
                Some(ResolvedShape::Struct(id)) => Some(id),
                _ => None,
            };
            return self
                .reflect_struct(ty, data, witness, id, depth)
                .map(Type::Named);
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
            return Ok(Type::Array {
                element: Box::new(self.reflect_at(
                    &ty.get_array_element_type(),
                    data,
                    &witness.project(index),
                    element,
                    depth + 1,
                )?),
                size,
            });
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
                Ok(Type::Tuple(members))
            }
            Err(e) => Err(format!("unsupported result type '{text}': {e}")),
        }
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
        if def.is_parametric() {
            return Err(format!(
                "parametric DSLX struct '{identifier}' is unsupported in BX"
            ));
        }
        let binding = self.bindings.entry(def.clone()).or_default();
        if let Some(id) = written {
            if identifier != id.name
                || binding
                    .declaration
                    .as_ref()
                    .is_some_and(|known| known != id)
                || self
                    .types
                    .get(id)
                    .is_some_and(|known| known.get_struct_def().ok().as_ref() != Some(&def))
            {
                return Err(format!(
                    "Rust and DSLX disagree on the struct declaration for '{}::{}'",
                    id.subject, id.name
                ));
            }
            binding.declaration.get_or_insert_with(|| id.clone());
            self.types.entry(id.clone()).or_insert_with(|| ty.clone());
        }
        if let Some(index) = binding.layout {
            return Ok(self.definitions[index].name.clone());
        }
        let path = binding.public_name.clone();
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
        self.bindings
            .get_mut(&def)
            .expect("registered native declaration")
            .layout = Some(index);
        self.definitions.push(StructDef {
            name: name.clone(),
            public,
            fields,
        });
        Ok(name)
    }
}
/// The declaration/explicit-type witnesses each contain one function.
fn probe_function(checked: &dslx::TypecheckedModule) -> Result<dslx::Function, String> {
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
        let result = registry.resolve_named("ambig", "Envelope", &mut data);
        assert_eq!(
            result.unwrap_err(),
            "DSLX struct field 'ambig::Envelope.item': unsupported result type 'u84' (DSLX enum)"
        );
        Ok(())
    }
}
