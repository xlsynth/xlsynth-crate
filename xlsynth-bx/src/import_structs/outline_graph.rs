// SPDX-License-Identifier: Apache-2.0

//! Resolve DSLX declarations in the source snapshot captured for a BX check.

use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[path = "native_files.rs"]
mod native_files;

use super::runfiles::XlsRunfiles;
use crate::ast::UsePath;
use crate::check::{MAX_ELABORATION_DEPTH, MAX_TYPE_SIZE};
use crate::parser::outline::outline_with_sum_names;
use crate::parser::source_bytes::decode_dslx_source;
use crate::{DslxOutline, OutlineKind, OutlineShape, OutlineType};
pub(super) use native_files::NativeSourceFiles;

// XLS generates these names from the stdlib sources of the pinned DSO. A
// caller's stdlib override changes their directory only.
const STDLIB_MODULES: &[&str] = &[
    "abs_diff",
    "acm_random",
    "apfloat",
    "bfloat16",
    "fixed_point",
    "float32",
    "float64",
    "float_instances",
    "hfloat16",
    "lza",
    "round",
    "std",
];

/// A declaration in a specific DSLX module, independent of aliases and layout.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct DeclId {
    pub subject: String,
    pub name: String,
}

/// Nominal leaves of a written type. DSLX supplies concrete widths and sizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ResolvedShape {
    Struct(DeclId),
    Enum(DeclId),
    Array(Box<ResolvedShape>),
    Tuple(Vec<ResolvedShape>),
    Other,
}

/// A local binding established by a DSLX import or an expanded use leaf.
enum Binding {
    Module(String),
    Member(DeclId),
}

impl Binding {
    fn subject(&self) -> &str {
        match self {
            Self::Module(subject) => subject,
            Self::Member(id) => &id.subject,
        }
    }
}

/// Source bytes and logical filename shared by the Rust and native frontends.
pub(super) struct SourceModule {
    pub subject: String,
    pub source_path: PathBuf,
    pub source: Vec<u8>,
    pub outline: DslxOutline,
    sums: BTreeSet<String>,
    bindings: BTreeMap<String, Binding>,
}

impl SourceModule {
    /// Resolve the qualifier or imported member in an ordinary function call.
    pub(super) fn call_subject(&self, qualifier: Option<&str>, name: &str) -> Option<&str> {
        match (qualifier, self.bindings.get(qualifier.unwrap_or(name))) {
            (None, Some(Binding::Member(id))) if id.name == name => Some(&id.subject),
            (None, None) => Some(&self.subject),
            (Some(_), Some(Binding::Module(subject))) => Some(subject),
            _ => None,
        }
    }
}

/// Native logical and physical filenames can differ for Bazel runfiles.
pub(crate) struct SourcePath {
    logical: PathBuf,
    pub(crate) physical: PathBuf,
}

/// BX module paths are root-relative. Native DSLX additionally tries the
/// suffix with its depot root removed, including ".x" for a single component.
fn module_suffixes(subject: &str, extension: &str) -> (PathBuf, Option<PathBuf>) {
    let full = PathBuf::from(format!("{}.{extension}", subject.replace('.', "/")));
    let stripped = (extension == "x").then(|| {
        let tail = subject.split_once('.').map_or("", |(_, tail)| tail);
        PathBuf::from(format!("{}.x", tail.replace('.', "/")))
    });
    (full, stripped)
}

/// Search roots and lazily discovered Bazel runfiles for module imports.
pub(crate) struct ModuleSearch<'a> {
    stdlib: &'a Path,
    roots: &'a [&'a Path],
    runfiles: OnceCell<XlsRunfiles>,
}

/// File lookup decisions, including missing full-module use leaves, for one
/// check.
struct SourceLocator<'a> {
    search: ModuleSearch<'a>,
    locations: BTreeMap<String, Option<SourcePath>>,
    member_use_names: BTreeSet<String>,
}

/// Try the implicit roots that the native importer always searches first.
fn implicit_source(
    full: &Path,
    stripped: Option<&Path>,
    runfiles: &OnceCell<XlsRunfiles>,
) -> Option<SourcePath> {
    for suffix in std::iter::once(full).chain(stripped) {
        if suffix.exists() {
            return Some(SourcePath {
                logical: suffix.to_owned(),
                physical: suffix.to_owned(),
            });
        }
        if let Some(path) = runfiles.get_or_init(XlsRunfiles::from_env).lookup(suffix) {
            // XLS retains the full logical path for a stripped runfile.
            return Some(SourcePath {
                logical: full.to_owned(),
                physical: path,
            });
        }
    }
    None
}

impl<'a> ModuleSearch<'a> {
    pub(crate) fn new(stdlib: &'a Path, roots: &'a [&'a Path]) -> Self {
        Self {
            stdlib,
            roots,
            runfiles: OnceCell::new(),
        }
    }

    /// Look up a dotted module subject using the native DSLX search order.
    pub(crate) fn file(&self, subject: &str, extension: &str) -> Option<SourcePath> {
        let (full, stripped) = if extension == "x" && STDLIB_MODULES.contains(&subject) {
            (self.stdlib.join(format!("{subject}.x")), None)
        } else {
            module_suffixes(subject, extension)
        };
        if let Some(path) = implicit_source(&full, stripped.as_deref(), &self.runfiles) {
            return Some(path);
        }
        for root in self.roots {
            for suffix in std::iter::once(&full).chain(stripped.as_ref()) {
                let physical = root.join(suffix);
                if physical.exists() {
                    return Some(SourcePath {
                        logical: physical.clone(),
                        physical,
                    });
                }
            }
        }
        None
    }
}

impl SourceLocator<'_> {
    fn lookup(&mut self, subject: &str) -> Option<&SourcePath> {
        self.locations
            .entry(subject.to_owned())
            .or_insert_with(|| self.search.file(subject, "x"))
            .as_ref()
    }

    /// A full module wins over a same-named member of its parent.
    fn use_binding(&mut self, components: &[String]) -> Result<Binding, String> {
        let (name, parent) = components.split_last().ok_or("empty DSLX use path")?;
        let full = components.join(".");
        if self.lookup(&full).is_some() {
            return Ok(Binding::Module(full));
        }
        let parent = parent.join(".");
        if !parent.is_empty() && self.lookup(&parent).is_some() {
            self.member_use_names.insert(name.clone());
            return Ok(Binding::Member(DeclId {
                subject: parent,
                name: name.clone(),
            }));
        }
        Err(format!(
            "could not resolve DSLX use '{}'",
            components.join("::")
        ))
    }
}

/// Reachable outlines, their dependency order, and the resolved aliases of
/// one immutable import graph.
pub(super) struct OutlineGraph {
    modules: BTreeMap<String, SourceModule>,
    order: Vec<String>,
    targets: BTreeMap<DeclId, DeclId>,
    exports: BTreeMap<DeclId, String>,
    unresolved_aliases: BTreeSet<DeclId>,
    member_use_names: BTreeSet<String>,
}

impl OutlineGraph {
    pub(super) fn load_uses(
        uses: &[UsePath],
        stdlib: &Path,
        search_paths: &[&Path],
    ) -> Result<Self, String> {
        let mut locator = SourceLocator {
            search: ModuleSearch::new(stdlib, search_paths),
            locations: BTreeMap::new(),
            member_use_names: BTreeSet::new(),
        };
        let roots = uses
            .iter()
            .map(|path| {
                locator
                    .use_binding(&path.components)
                    .map(|binding| binding.subject().to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::capture(roots, &mut locator)
    }

    /// Freeze each module once. Finishing a DFS node after its dependencies
    /// gives the installation order for the native import graph.
    fn capture(roots: Vec<String>, locator: &mut SourceLocator<'_>) -> Result<Self, String> {
        enum Visit {
            Enter(String),
            Finish(String),
        }
        let mut pending: Vec<_> = roots.into_iter().rev().map(Visit::Enter).collect();
        let mut active = BTreeSet::new();
        let mut modules = BTreeMap::new();
        let mut order = Vec::new();
        while let Some(visit) = pending.pop() {
            let subject = match visit {
                Visit::Finish(subject) => {
                    active.remove(&subject);
                    order.push(subject);
                    continue;
                }
                Visit::Enter(subject) => subject,
            };
            if active.contains(&subject) {
                return Err(format!("cyclic DSLX import of '{subject}'"));
            }
            if modules.contains_key(&subject) {
                continue;
            }
            let path = locator
                .lookup(&subject)
                .ok_or_else(|| format!("could not find DSLX module '{subject}'"))?;
            let source_path = path.logical.clone();
            let source = fs::read(&path.physical).map_err(|e| {
                format!(
                    "reading DSLX module '{subject}' at {}: {e}",
                    path.physical.display()
                )
            })?;
            let decoded = decode_dslx_source(&source).map_err(|e| {
                format!(
                    "decoding DSLX module '{subject}' at {}: {e}",
                    source_path.display()
                )
            })?;
            let mut sums = BTreeSet::new();
            let outline = outline_with_sum_names(&decoded, &mut sums).map_err(|e| {
                format!(
                    "outlining DSLX module '{subject}' at {}: {e}",
                    source_path.display()
                )
            })?;
            let mut bindings: BTreeMap<_, _> = outline
                .imports
                .iter()
                .map(|(name, module)| (name.clone(), Binding::Module(module.clone())))
                .collect();
            let mut dependencies: BTreeSet<_> = outline.imports.values().cloned().collect();
            for leaf in &outline.uses {
                let binding = locator.use_binding(leaf)?;
                // Distinct leaves in a use tree may have the same local name.
                // Both modules are loaded even when the later binding wins.
                dependencies.insert(binding.subject().to_owned());
                bindings.insert(leaf.last().ok_or("empty DSLX use leaf")?.clone(), binding);
            }
            active.insert(subject.clone());
            pending.push(Visit::Finish(subject.clone()));
            pending.extend(dependencies.into_iter().rev().map(Visit::Enter));
            modules.insert(
                subject.clone(),
                SourceModule {
                    subject,
                    source_path,
                    source,
                    outline,
                    sums,
                    bindings,
                },
            );
        }
        let mut result = Self {
            modules,
            order,
            targets: BTreeMap::new(),
            exports: BTreeMap::new(),
            unresolved_aliases: BTreeSet::new(),
            member_use_names: std::mem::take(&mut locator.member_use_names),
        };
        result.index_declarations();
        Ok(result)
    }

    pub(super) fn sources(&self) -> impl Iterator<Item = &SourceModule> {
        self.order.iter().map(|subject| &self.modules[subject])
    }

    /// Give native checking a private module namespace for this snapshot.
    pub(super) fn native_source_files(&self) -> Result<NativeSourceFiles, String> {
        NativeSourceFiles::capture(self)
    }

    pub(super) fn has_module(&self, subject: &str) -> bool {
        self.modules.contains_key(subject)
    }

    pub(super) fn has_public_member(&self, subject: &str, name: &str) -> bool {
        self.modules.get(subject).is_some_and(|module| {
            module.outline.public_values.contains(name)
                || module
                    .outline
                    .declarations
                    .get(name)
                    .is_some_and(|decl| decl.public)
        })
    }

    pub(super) fn has_public_function(&self, subject: &str, name: &str) -> bool {
        self.modules
            .get(subject)
            .is_some_and(|module| module.outline.public_functions.contains(name))
    }

    /// Resolve a type name in the module that wrote it. Native typechecking
    /// enforces access rules at each source reference (including v1 use
    /// syntax).
    fn reference(&self, subject: &str, components: &[String]) -> Option<DeclId> {
        let (name, qualifiers) = components.split_last()?;
        let module = self.module_reference(subject, qualifiers)?;
        if module.outline.declarations.contains_key(name) {
            Some(DeclId {
                subject: module.subject.clone(),
                name: name.clone(),
            })
        } else if let Some(Binding::Member(target)) = module.bindings.get(name) {
            Some(target.clone())
        } else {
            None
        }
    }

    /// Resolve every component of a module qualifier in its declaration scope.
    pub(super) fn module_reference(&self, subject: &str, path: &[String]) -> Option<&SourceModule> {
        let mut module = self.modules.get(subject)?;
        for qualifier in path {
            let Binding::Module(target) = module.bindings.get(qualifier)? else {
                return None;
            };
            module = self.modules.get(target)?;
        }
        Some(module)
    }

    /// Follow a checked qualifier through module imports and type aliases.
    pub(super) fn is_sum(&self, subject: &str, path: &[String]) -> bool {
        self.reference(subject, path)
            .and_then(|id| self.targets.get(&id))
            .is_some_and(|id| self.modules[&id.subject].sums.contains(&id.name))
    }

    /// Collapse name-only alias chains once, without expanding tuple or array
    /// types. Private originals are indexed whenever a public alias exposes
    /// them.
    fn index_declarations(&mut self) {
        let mut links = BTreeMap::new();
        for (subject, module) in &self.modules {
            for (name, decl) in &module.outline.declarations {
                let id = DeclId {
                    subject: subject.clone(),
                    name: name.clone(),
                };
                if let OutlineKind::Alias(OutlineType {
                    shape: OutlineShape::Name(path),
                    ..
                }) = &decl.kind
                {
                    if decl.has_public_spelling(name) && !decl.parametric {
                        self.unresolved_aliases.insert(id.clone());
                    }
                    links.insert(id, self.reference(subject, path));
                } else {
                    self.targets.insert(id.clone(), id);
                }
            }
        }
        while let Some((id, mut next)) = links.pop_first() {
            let mut chain = vec![id];
            while let Some(id) = next {
                if let Some(target) = self.targets.get(&id).cloned() {
                    for alias in chain {
                        self.unresolved_aliases.remove(&alias);
                        self.targets.insert(alias, target.clone());
                    }
                    break;
                }
                next = links.remove(&id).flatten();
                chain.push(id);
            }
        }
        for (id, target) in &self.targets {
            if !self.modules[&id.subject].outline.declarations[&id.name]
                .has_public_spelling(&id.name)
            {
                continue;
            }
            let path = format!("{}::{}", id.subject, id.name);
            if id == target {
                self.exports.insert(target.clone(), path);
            } else {
                self.exports.entry(target.clone()).or_insert(path);
            }
        }
    }

    /// Every public declaration or alias whose resolved original is a concrete
    /// struct.
    pub(super) fn struct_exports(&self) -> impl Iterator<Item = (&DeclId, &DeclId)> {
        self.targets.iter().filter(|(id, target)| {
            let decl = &self.modules[&id.subject].outline.declarations[&id.name];
            let original = &self.modules[&target.subject].outline.declarations[&target.name];
            decl.has_public_spelling(&id.name)
                && !decl.parametric
                && !original.parametric
                && matches!(
                    original.kind,
                    OutlineKind::Struct { .. } | OutlineKind::GeneratedStruct
                )
        })
    }

    /// Public concrete alias spellings whose named target requires DSLX
    /// resolution, for example an associated type on a struct.
    pub(super) fn unresolved_aliases(&self) -> impl Iterator<Item = &DeclId> {
        self.unresolved_aliases.iter()
    }

    pub(super) fn public_path(&self, id: &DeclId) -> Option<&str> {
        self.exports.get(id).map(String::as_str)
    }

    /// A concrete alias can expose an instance of a private struct without
    /// exposing its template. Binding inference needs the template itself.
    pub(super) fn public_parametric_struct(&self, subject: &str, name: &str) -> bool {
        self.modules
            .get(subject)
            .and_then(|module| module.outline.declarations.get(name))
            .is_some_and(|decl| decl.is_public_struct_template(name))
    }

    pub(super) fn named(&self, subject: &str, name: &str) -> Option<ResolvedShape> {
        let module = self.modules.get(subject)?;
        if !module
            .outline
            .declarations
            .get(name)?
            .has_public_spelling(name)
        {
            return None;
        }
        let id = DeclId {
            subject: subject.to_owned(),
            name: name.to_owned(),
        };
        let mut budget = MAX_TYPE_SIZE;
        Some(self.decl_shape(&id, MAX_ELABORATION_DEPTH, &mut budget))
    }

    pub(super) fn return_type(&self, subject: &str, function: &str) -> Option<ResolvedShape> {
        let ty = self.modules.get(subject)?.outline.functions.get(function)?;
        Some(self.resolve_type(subject, ty))
    }

    pub(super) fn fields(&self, id: &DeclId) -> Option<&[(String, OutlineType)]> {
        let decl = self
            .modules
            .get(&id.subject)?
            .outline
            .declarations
            .get(&id.name)?;
        let OutlineKind::Struct { fields } = &decl.kind else {
            return None;
        };
        Some(fields)
    }

    pub(super) fn resolve_type(&self, subject: &str, ty: &OutlineType) -> ResolvedShape {
        let mut budget = MAX_TYPE_SIZE;
        self.shape(subject, ty, MAX_ELABORATION_DEPTH, &mut budget)
    }

    fn decl_shape(&self, id: &DeclId, depth: usize, budget: &mut usize) -> ResolvedShape {
        let Some(target) = self.targets.get(id) else {
            return ResolvedShape::Other;
        };
        match &self.modules[&target.subject].outline.declarations[&target.name].kind {
            OutlineKind::Struct { .. } | OutlineKind::GeneratedStruct => {
                ResolvedShape::Struct(target.clone())
            }
            OutlineKind::Enum { .. } => ResolvedShape::Enum(target.clone()),
            OutlineKind::Alias(ty) => self.shape(&target.subject, ty, depth, budget),
            OutlineKind::Opaque => ResolvedShape::Other,
        }
    }

    fn shape(
        &self,
        subject: &str,
        ty: &OutlineType,
        depth: usize,
        budget: &mut usize,
    ) -> ResolvedShape {
        if *budget == 0 {
            return ResolvedShape::Other;
        }
        *budget -= 1;
        if depth == 0 {
            return ResolvedShape::Other;
        }
        match &ty.shape {
            OutlineShape::Name(path) => self
                .reference(subject, path)
                .map_or(ResolvedShape::Other, |id| {
                    self.decl_shape(&id, depth - 1, budget)
                }),
            OutlineShape::Array(element) => {
                ResolvedShape::Array(Box::new(self.shape(subject, element, depth - 1, budget)))
            }
            OutlineShape::Tuple(members) => {
                let mut result = Vec::new();
                for member in members {
                    if *budget == 0 {
                        return ResolvedShape::Other;
                    }
                    result.push(self.shape(subject, member, depth - 1, budget));
                }
                ResolvedShape::Tuple(result)
            }
            OutlineShape::Other => ResolvedShape::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};

    #[test]
    fn a_depot_component_longer_than_a_filename_keeps_its_native_symbol()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let depot = "a".repeat(260);
        fs::write(
            dir.path().join("leaf.x"),
            "pub fn inc(x: u8) -> u8 { x + u8:1 }",
        )?;
        let uses = crate::parse_module(&format!("use {depot}::leaf;"))?.uses;
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let paths = [dir.path()];
        let source = format!("import {depot}.leaf; fn run(x: u8) -> u8 {{ leaf::inc(x) }}");
        let mut native = ImportData::new(Some(stdlib), &paths);
        let original =
            parse_and_typecheck(&source, "probe.x", "probe", &mut native)?.to_ir_text()?;
        let graph = OutlineGraph::load_uses(&uses, stdlib, &paths)?;
        let files = graph.native_source_files()?;
        let mut snapshot = files.import_data();
        let relocated = parse_and_typecheck(
            &files.native_source(&source),
            "probe.x",
            "probe",
            &mut snapshot,
        )?
        .to_ir_text()?;
        let symbols = |ir: &str| -> BTreeSet<String> {
            ir.lines()
                .filter_map(|line| {
                    line.strip_prefix("fn ")?
                        .split_once('(')
                        .map(|(name, _)| name.to_owned())
                })
                .collect()
        };
        let expected = symbols(&original);
        assert!(expected.contains(&format!("__{depot}_leaf__inc")));
        assert_eq!(symbols(&files.restore_ir_names(&relocated)), expected);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_cwd_fifo_supplies_bytes_only_to_snapshot_capture() -> Result<(), Box<dyn std::error::Error>>
    {
        fs::create_dir_all("target")?;
        let cwd = tempfile::Builder::new()
            .prefix("bx_fifo_")
            .tempdir_in("target")?;
        let name = cwd.path().file_name().unwrap().to_str().unwrap();
        let source = cwd.path().join("wire.x");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&source)
                .status()?
                .success()
        );
        let uses = crate::parse_module(&format!("use target::{name}::wire;"))?.uses;
        let graph = std::thread::scope(|scope| {
            let writer = scope.spawn(|| fs::write(&source, "pub type Kind = u8;"));
            let graph = OutlineGraph::load_uses(&uses, xlsynth::default_dslx_stdlib_path(), &[]);
            writer.join().expect("FIFO writer").expect("write source");
            graph
        })?;
        fs::remove_file(&source)?;
        fs::write(&source, "pub type Kind = u16;")?;
        let files = graph.native_source_files()?;
        let mut data = files.import_data();
        let probe =
            format!("import target.{name}.wire as wire; fn probe(x: wire::Kind) -> u8 {{ x }}");
        parse_and_typecheck(&files.native_source(&probe), "probe.x", "probe", &mut data)?;
        Ok(())
    }

    #[test]
    fn frozen_imports_preserve_search_root_priority_raw_bytes_and_filenames()
    -> Result<(), Box<dyn std::error::Error>> {
        let first = tempfile::Builder::new()
            .prefix("bx original source ")
            .tempdir()?;
        let second = tempdir()?;
        let first_file = first.path().join("bx_outline_rank.x");
        let second_file = second.path().join("a/bx_outline_rank.x");
        fs::create_dir(second.path().join("a"))?;
        let first_source = b"pub type Width = u8; pub fn byte() -> u8 { '\xff' }";
        fs::write(&first_file, first_source)?;
        fs::write(&second_file, "pub type Width = u16;")?;
        // a.bx_outline_rank uses root 1's stripped file, despite root 2
        // containing its full path. prefix.a.bx_outline_rank needs root 2.
        let facade = first.path().join("bx_outline_ranks.x");
        fs::write(
            &facade,
            "import a.bx_outline_rank as a; import prefix.a.bx_outline_rank as b; pub fn pair(x: a::Width, y: b::Width) -> (u8, u16) { (x, y) }",
        )?;
        let roots = crate::parse_module("use bx_outline_ranks;")?.uses;
        let graph = OutlineGraph::load_uses(
            &roots,
            xlsynth::default_dslx_stdlib_path(),
            &[first.path(), second.path()],
        )?;
        first.close()?;
        second.close()?;
        let files = graph.native_source_files()?;
        let frozen_first = files.physical_path(&first_file).unwrap();
        let frozen_second = files.physical_path(&second_file).unwrap();
        assert_eq!(fs::read(frozen_first)?, first_source);
        assert_eq!(files.logical_path(frozen_first), Some(first_file.as_path()));
        assert_eq!(
            files.logical_path(frozen_second),
            Some(second_file.as_path())
        );
        assert_eq!(
            files.remap_diagnostic(format!(
                "{}:1:2 (see {}:3:4)",
                frozen_first.display(),
                frozen_second.display()
            )),
            format!(
                "{}:1:2 (see {}:3:4)",
                first_file.display(),
                second_file.display()
            )
        );
        let mut data = files.import_data();
        parse_and_typecheck(
            &files.native_source("import bx_outline_ranks;"),
            "load.x",
            "load",
            &mut data,
        )?;
        Ok(())
    }

    #[test]
    fn private_namespace_preserves_stripped_module_and_member_uses_after_source_deletion()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        // b.a.m comes from its stripped a/m.x. Materializing b/a/m.x
        // instead would make prefix::b::a::m resolve as a full module through
        // its stripped suffix, shadowing member m in prefix.b.a.
        fs::create_dir(root.path().join("a"))?;
        fs::create_dir_all(root.path().join("prefix/b"))?;
        fs::write(root.path().join("a/m.x"), "pub struct Item { x: u8 }")?;
        fs::write(root.path().join("prefix/b/a.x"), "pub struct m { x: u16 }")?;
        fs::write(
            root.path().join("bx_outline_native_use.x"),
            r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use b::a::m::Item;
use prefix::b::a::m;
pub fn checked(x: Item, y: m) -> (u8, u16) { (x.x, y.x) }"#,
        )?;
        let uses = crate::parse_module("use bx_outline_native_use; use b::a::m::Item;")?.uses;
        let graph =
            OutlineGraph::load_uses(&uses, xlsynth::default_dslx_stdlib_path(), &[root.path()])?;
        root.close()?;
        let files = graph.native_source_files()?;
        let mut data = files.import_data();
        parse_and_typecheck(
            &files.native_source("import bx_outline_native_use;"),
            "load.x",
            "load",
            &mut data,
        )?;
        parse_and_typecheck(
            &files.native_source("#![feature(use_syntax)]\nuse b::a::m::Item;"),
            "probe.x",
            "probe",
            &mut data,
        )?;
        Ok(())
    }

    #[test]
    fn a_new_cwd_module_does_not_capture_a_member_use() -> Result<(), Box<dyn std::error::Error>> {
        // Keep the short-lived CWD source in the ignored build directory.
        fs::create_dir_all("target")?;
        let cwd = tempfile::Builder::new()
            .prefix("bx_outline_shadow_")
            .tempdir_in("target")?;
        let name = cwd.path().file_name().unwrap().to_str().unwrap();
        let root = tempdir()?;
        fs::create_dir(root.path().join("target"))?;
        fs::write(
            root.path().join("target").join(format!("{name}.x")),
            "pub type Item = u8;",
        )?;
        let uses = crate::parse_module(&format!("use target::{name}::Item;"))?.uses;
        let graph =
            OutlineGraph::load_uses(&uses, xlsynth::default_dslx_stdlib_path(), &[root.path()])?;
        fs::write(cwd.path().join("Item.x"), "pub const injected = u8:42;")?;
        let files = graph.native_source_files()?;
        let mut data = files.import_data();
        let source = format!("#![feature(use_syntax)]\nuse target::{name}::Item;");
        parse_and_typecheck(&files.native_source(&source), "probe.x", "probe", &mut data)?;
        Ok(())
    }

    #[test]
    fn stdlib_overrides_follow_the_native_builtin_module_list()
    -> Result<(), Box<dyn std::error::Error>> {
        let stdlib = tempdir()?;
        let search = tempdir()?;
        let paths = [search.path()];
        let mut subjects: BTreeSet<String> =
            STDLIB_MODULES.iter().map(|name| (*name).into()).collect();
        // Include newly shipped files too, so a release that adds a native
        // builtin detects a stale Rust list by actually typechecking the
        // import.
        for file in fs::read_dir(xlsynth::default_dslx_stdlib_path())? {
            let path = file?.path();
            if path.extension().is_some_and(|ext| ext == "x")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && crate::parser::is_use_identifier(stem)
            {
                subjects.insert(stem.to_owned());
            }
        }
        subjects.insert("bx_outline_user".into());
        for subject in subjects {
            let file = format!("{subject}.x");
            fs::write(stdlib.path().join(&file), "pub type Tag = u8;")?;
            fs::write(search.path().join(&file), "pub type Tag = u16;")?;
            let uses = crate::parse_module(&format!("use {subject};"))?.uses;
            let graph = OutlineGraph::load_uses(&uses, stdlib.path(), &paths)?;
            let builtin = STDLIB_MODULES.contains(&subject.as_str());
            let root = if builtin {
                stdlib.path()
            } else {
                search.path()
            };
            assert_eq!(graph.sources().next().unwrap().source_path, root.join(file));
            let expected = if builtin { "u8" } else { "u16" };
            let source = format!(
                "import {subject} as chosen; fn probe(x: chosen::Tag) -> {expected} {{ x }}"
            );
            let mut data = ImportData::new(Some(stdlib.path()), &paths);
            parse_and_typecheck(&source, "probe.x", "probe", &mut data)?;
        }
        Ok(())
    }

    #[test]
    fn aliases_and_aggregate_leaves_keep_the_defining_module()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        for (module, source) in [
            (
                "wire",
                "struct Hidden { x: u8 } pub type Item = Hidden; pub enum Mode : u1 { A = 0 }",
            ),
            ("other", "pub struct Hidden { x: u8 }"),
            (
                "facade",
                "import wire as w; pub type Payload = w::Item; pub type Flag = w::Mode;",
            ),
            (
                "factory",
                r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use facade::{Payload, Flag};
use other::Hidden;
type Pair = (Payload[2], Hidden);
pub struct Frame { data: Pair, mode: Flag }
pub fn make() -> Frame { zero!<Frame>() }
"#,
            ),
        ] {
            fs::write(dir.path().join(format!("{module}.x")), source)?;
        }
        let uses = crate::parse_module("use factory;")?.uses;
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let paths = [dir.path()];
        let mut native = ImportData::new(Some(stdlib), &paths);
        parse_and_typecheck("import factory;", "probe.x", "probe", &mut native)?;
        let graph = OutlineGraph::load_uses(&uses, stdlib, &paths)?;
        let frame = DeclId {
            subject: "factory".into(),
            name: "Frame".into(),
        };
        let wire = DeclId {
            subject: "wire".into(),
            name: "Hidden".into(),
        };
        let other = DeclId {
            subject: "other".into(),
            name: "Hidden".into(),
        };
        let mode = DeclId {
            subject: "wire".into(),
            name: "Mode".into(),
        };
        assert_eq!(graph.named("wire", "Hidden"), None);
        assert_eq!(
            graph.named("facade", "Payload"),
            Some(ResolvedShape::Struct(wire.clone()))
        );
        assert_eq!(graph.public_path(&wire), Some("facade::Payload"));
        assert_eq!(graph.public_path(&mode), Some("wire::Mode"));
        assert_eq!(
            graph.return_type("factory", "make"),
            Some(ResolvedShape::Struct(frame.clone()))
        );
        let fields = graph.fields(&frame).unwrap();
        assert_eq!(fields[0].0, "data");
        assert_eq!(
            graph.resolve_type("factory", &fields[0].1),
            ResolvedShape::Tuple(vec![
                ResolvedShape::Array(Box::new(ResolvedShape::Struct(wire))),
                ResolvedShape::Struct(other)
            ])
        );
        assert_eq!(
            graph.resolve_type("factory", &fields[1].1),
            ResolvedShape::Enum(mode)
        );
        let subjects: Vec<_> = graph.sources().map(|s| s.subject.as_str()).collect();
        assert!(
            subjects.iter().position(|s| *s == "wire")
                < subjects.iter().position(|s| *s == "facade")
        );
        assert_eq!(subjects.last(), Some(&"factory"));
        Ok(())
    }

    #[test]
    fn stripped_module_in_first_root_wins_over_full_module_in_second_root()
    -> Result<(), Box<dyn std::error::Error>> {
        let first = tempdir()?;
        let second = tempdir()?;
        let chosen = first.path().join("bx_outline_paths.x");
        let source = "pub struct Chosen { x: u8 }";
        fs::write(&chosen, source)?;
        fs::create_dir(second.path().join("depot"))?;
        fs::write(
            second.path().join("depot/bx_outline_paths.x"),
            "pub struct Shadow { x: u8 }",
        )?;
        let uses = crate::parse_module("use depot::bx_outline_paths;")?.uses;
        let graph = OutlineGraph::load_uses(
            &uses,
            xlsynth::default_dslx_stdlib_path(),
            &[first.path(), second.path()],
        )?;
        fs::write(&chosen, "pub struct New { other: u32 }")?;
        let module = graph.sources().next().unwrap();
        assert_eq!(module.subject, "depot.bx_outline_paths");
        assert_eq!(module.source_path, chosen);
        assert_eq!(module.source, source.as_bytes());
        assert!(graph.named(&module.subject, "Chosen").is_some());
        assert!(graph.named(&module.subject, "New").is_none());
        Ok(())
    }

    #[test]
    fn use_leaf_selects_existing_module_before_parent_member()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempdir()?;
        let child = root.path().join("bx_outline_subject");
        fs::create_dir(&child)?;
        fs::write(
            root.path().join("bx_outline_subject.x"),
            "pub struct Item { x: u8 }",
        )?;
        fs::write(child.join("Item.x"), "pub struct Data { x: u16 }")?;
        let uses = crate::parse_module("use bx_outline_subject::Item;")?.uses;
        let stdlib = xlsynth::default_dslx_stdlib_path();
        let first = OutlineGraph::load_uses(&uses, stdlib, &[root.path()])?;
        fs::remove_file(child.join("Item.x"))?;
        let second = OutlineGraph::load_uses(&uses, stdlib, &[root.path()])?;
        assert!(first.has_module("bx_outline_subject.Item"));
        assert!(!first.has_module("bx_outline_subject"));
        assert!(!second.has_module("bx_outline_subject.Item"));
        assert!(second.has_module("bx_outline_subject"));
        Ok(())
    }

    #[test]
    fn grouped_use_keeps_all_dependencies_and_the_last_same_named_leaf()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        fs::create_dir(dir.path().join("pkg"))?;
        fs::write(dir.path().join("pkg/a.x"), "pub struct Item { x: u16 }")?;
        fs::write(dir.path().join("pkg/b.x"), "pub struct Item { x: u8 }")?;
        fs::write(
            dir.path().join("facade.x"),
            "#![feature(type_inference_v1)]\n#![feature(use_syntax)]\nuse pkg::{a::Item, b::Item}; pub type Pick = Item;",
        )?;
        let uses = crate::parse_module("use facade;")?.uses;
        let graph =
            OutlineGraph::load_uses(&uses, xlsynth::default_dslx_stdlib_path(), &[dir.path()])?;
        assert!(graph.has_module("pkg.a"));
        assert!(graph.has_module("pkg.b"));
        assert_eq!(graph.named("facade", "Pick"), graph.named("pkg.b", "Item"));
        assert_ne!(graph.named("facade", "Pick"), graph.named("pkg.a", "Item"));
        Ok(())
    }

    #[test]
    fn import_cycles_are_diagnosed_before_installation() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("bx_outline_cycle_a.x"),
            "import bx_outline_cycle_b;",
        )?;
        fs::write(
            dir.path().join("bx_outline_cycle_b.x"),
            "import bx_outline_cycle_a;",
        )?;
        let uses = crate::parse_module("use bx_outline_cycle_a;")?.uses;
        let Err(error) =
            OutlineGraph::load_uses(&uses, xlsynth::default_dslx_stdlib_path(), &[dir.path()])
        else {
            panic!("expected cyclic import error")
        };
        assert_eq!(error, "cyclic DSLX import of 'bx_outline_cycle_a'");
        Ok(())
    }
}
