// SPDX-License-Identifier: Apache-2.0

//! Load BX libraries in dependency order and resolve names in their declaring
//! module.

mod lookup;
mod names;
mod reachable;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use crate::ImportOptions;
use crate::ast::{FunctionParametricRef, Module, UsePath};
use crate::check::MAX_ELABORATION_DEPTH;
use crate::import_structs::DslxSourceLoader;
use crate::parser;
use lookup::{BxUse, UseLookup};

pub(crate) use lookup::validate_use_path;
pub(crate) use names::{UnlinkedBlockScope, local_callee_name, mark_local_callee};

/// The source module and spelling of a hygienically qualified BX name.
pub(crate) fn library_item(name: &str) -> Option<(&str, &str)> {
    name.strip_prefix("$bx$")?.split_once('$')
}

fn qualified(subject: Option<&str>, name: &str) -> String {
    subject.map_or_else(|| name.to_owned(), |s| format!("$bx${s}${name}"))
}

pub(crate) fn zero_binding(owner: &str) -> String {
    qualified(library_item(owner).map(|(subject, _)| subject), "zero!")
}

pub(crate) const BUILTIN_MAP: &str = "$builtin$map";

/// A public declaration and the information required to parse its call syntax.
#[derive(Clone)]
struct Export {
    name: String,
    parametrics: Vec<ParametricKind>,
    is_enum: bool,
    is_struct: bool,
}

/// Callable arguments have the visibility of calls. Value arguments have
/// the visibility of expressions at the same site.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParametricKind {
    Value,
    Callback,
}

impl Export {
    /// Enum members inherit the visibility of their declaring type.
    fn reference(&self, member: Option<&str>) -> Option<String> {
        match member {
            None => Some(self.name.clone()),
            Some(member) if self.is_enum => Some(format!("{}::{member}", self.name)),
            Some(_) => None,
        }
    }
}

/// Public members of one physical BX file, shared by all incoming imports.
struct Library {
    subject: String,
    exports: BTreeMap<String, Export>,
    /// Longest BX import path starting here, including this library.
    depth: usize,
}

enum Binding {
    Module(Rc<Library>),
    Member(Export),
    Dslx(String),
}

/// Bindings available in one file before its declarations are linked together.
struct Names {
    subject: Option<String>,
    definitions: BTreeSet<String>,
    enums: BTreeSet<String>,
    structs: BTreeSet<String>,
    templates: BTreeMap<String, Vec<ParametricKind>>,
    uses: BTreeMap<String, Binding>,
    positions: BTreeMap<String, usize>,
}

impl Names {
    fn new(
        module: &Module,
        subject: Option<&str>,
        uses: BTreeMap<String, Binding>,
    ) -> Result<Self, String> {
        let mut definitions = BTreeSet::new();
        for name in module
            .structs
            .iter()
            .map(|d| &d.name)
            .chain(module.enums.iter().map(|d| &d.name))
            .chain(module.functions.iter().map(|d| &d.name))
            .chain(module.constants.iter().map(|d| &d.name))
            .chain(module.blocks.iter().map(|d| &d.name))
            .chain(module.externs.iter().map(|d| &d.name))
            .chain(module.extern_blocks.iter().map(|d| &d.name))
        {
            if name.contains('$') {
                return Err(format!("invalid source declaration name '{name}'"));
            }
            if uses.contains_key(name) || !definitions.insert(name.clone()) {
                return Err(format!("duplicate name '{name}'"));
            }
        }
        for f in &module.externs {
            if let Some((head, _)) = f.name.split_once("::")
                && uses.contains_key(head)
            {
                return Err(format!(
                    "extern fn '{}' conflicts with use binding '{head}'",
                    f.name
                ));
            }
        }
        let mut templates = BTreeMap::new();
        for f in &module.functions {
            if !f.value_parametrics.is_empty() || !f.callback_params.is_empty() {
                templates.insert(
                    f.name.clone(),
                    f.parametric_bindings()
                        .map(|p| match p {
                            FunctionParametricRef::Value(_) => ParametricKind::Value,
                            FunctionParametricRef::Callback(_) => ParametricKind::Callback,
                        })
                        .collect(),
                );
            }
        }
        for b in &module.blocks {
            if !b.callback_params.is_empty() {
                templates.insert(
                    b.name.clone(),
                    vec![ParametricKind::Callback; b.callback_params.len()],
                );
            }
        }
        Ok(Self {
            subject: subject.map(str::to_owned),
            definitions,
            enums: module.enums.iter().map(|d| d.name.clone()).collect(),
            structs: module.structs.iter().map(|d| d.name.clone()).collect(),
            templates,
            uses,
            positions: module
                .declaration_order
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect(),
        })
    }

    fn declared(&self, name: &str) -> String {
        qualified(self.subject.as_deref(), name)
    }

    /// Look up the formals under the spelling used in this source file.
    fn parametrics(&self, name: &str) -> &[ParametricKind] {
        if let Some((head, member)) = name.split_once("::")
            && let Some(Binding::Module(library)) = self.uses.get(head)
            && let Some(export) = library.exports.get(member)
        {
            &export.parametrics
        } else if let Some(Binding::Member(export)) = self.uses.get(name) {
            &export.parametrics
        } else {
            self.templates.get(name).map(Vec::as_slice).unwrap_or(&[])
        }
    }

    /// Resolve one free reference before it can see another file's bindings.
    fn resolve(&self, name: &str, at: Option<usize>) -> Result<String, String> {
        self.resolve_with_root_fallback(name, at, name)
    }

    /// Preserve the caller's interpretation of a name with no visible root
    /// binding.
    fn resolve_with_root_fallback(
        &self,
        name: &str,
        at: Option<usize>,
        fallback: &str,
    ) -> Result<String, String> {
        if name.contains('$') {
            return Err(format!("invalid source reference '{name}'"));
        }
        let (head, member) = name
            .split_once("::")
            .map_or((name, None), |(h, m)| (h, Some(m)));
        let visible = at
            .zip(self.positions.get(head))
            .is_none_or(|(at, &pos)| pos <= at);
        let builtin = name == "map" || crate::check::is_call_builtin(name);
        if let Some(binding) = self.uses.get(head) {
            if !visible {
                // A later use does not hide an earlier builtin reference.
                if builtin {
                    return Ok(format!("{}{name}", crate::check::BUILTIN_PREFIX));
                }
                return Err(format!(
                    "use binding '{head}' is declared later in the module"
                ));
            }
            return match (binding, member) {
                (Binding::Module(lib), Some(m)) => {
                    let (name, member) = m
                        .split_once("::")
                        .map_or((m, None), |(name, member)| (name, Some(member)));
                    lib.exports
                        .get(name)
                        .and_then(|e| e.reference(member))
                        .ok_or_else(|| {
                            format!("unknown or private BX member '{}::{m}'", lib.subject)
                        })
                }
                (Binding::Module(_), None) => Err(format!(
                    "use binding '{head}' names a BX module and requires a member"
                )),
                (Binding::Member(e), member) => e.reference(member).ok_or_else(|| {
                    format!("use binding '{head}' is a BX member and cannot qualify '{name}'")
                }),
                (Binding::Dslx(alias), Some(m)) => Ok(format!("{alias}::{m}")),
                (Binding::Dslx(alias), None) => Ok(alias.clone()),
            };
        }
        if self.definitions.contains(name) && (visible || !builtin) {
            return Ok(self.declared(name));
        }
        if let Some(member) = member
            && self.enums.contains(head)
        {
            return Ok(format!("{}::{member}", self.declared(head)));
        }
        if let Some(subject) = &self.subject {
            if builtin {
                return Ok(format!("{}{name}", crate::check::BUILTIN_PREFIX));
            }
            if name == "zero!" {
                return Ok(self.declared(name));
            }
            return Err(format!("unknown name '{name}' in BX module '{subject}'"));
        }
        Ok(fallback.to_owned())
    }
}

/// The checked frontend consumes a single declaration graph. Each DSLX use
/// retains its original path as well as its binding in that graph.
#[derive(Default)]
pub(crate) struct LinkedSources {
    pub module: Module,
    pub dslx_bindings: Vec<String>,
    /// Filenames behind the logical module names used in checked diagnostics.
    pub module_paths: BTreeMap<String, PathBuf>,
    /// Direct BX member imports that shadow the zero intrinsic at a
    /// declaration.
    pub zero_uses: BTreeMap<String, String>,
}

/// BX declarations and the DSLX source snapshot used to parse their calls.
pub(crate) struct SourceImports<'a> {
    pub libraries: Option<LinkedSources>,
    pub native: DslxSourceLoader<'a>,
}

/// An imported file awaiting resolution of its uses before its body is parsed.
struct LibrarySource {
    target: BxUse,
    canonical_path: PathBuf,
    binding: String,
    text: String,
    uses: Vec<UsePath>,
}

/// The remaining uses and completed bindings of one active module. The root
/// has no incoming import or file identity.
struct PendingModule {
    source: Option<LibrarySource>,
    uses: std::vec::IntoIter<UsePath>,
    bindings: BTreeMap<String, Binding>,
    depth: usize,
}

impl PendingModule {
    fn new(source: Option<LibrarySource>, uses: Vec<UsePath>) -> Self {
        Self {
            source,
            uses: uses.into_iter(),
            bindings: BTreeMap::new(),
            depth: 1,
        }
    }

    /// Bind a completed child and include its entire subtree in the depth.
    fn bind_library(
        &mut self,
        leaf: String,
        target: &BxUse,
        library: Rc<Library>,
    ) -> Result<(), String> {
        self.depth = self.depth.max(1 + library.depth);
        let binding = if let Some(member) = &target.member {
            let export = library.exports.get(member).ok_or_else(|| {
                format!(
                    "unknown or private BX member '{}::{member}'",
                    target.subject
                )
            })?;
            Binding::Member(export.clone())
        } else {
            Binding::Module(library)
        };
        self.bindings.insert(leaf, binding);
        Ok(())
    }
}

impl BxUse {
    fn context(&self, error: String) -> String {
        format!(
            "BX import '{}' ({}): {error}",
            self.subject,
            self.path.display()
        )
    }
}

/// A source snapshot and traversal state for one check or compilation.
struct Graph<'a> {
    lookup: UseLookup<'a>,
    loaded: BTreeMap<PathBuf, Rc<Library>>,
    linked: LinkedSources,
}

impl<'a> Graph<'a> {
    fn new(options: &'a ImportOptions<'a>) -> Self {
        Self {
            lookup: UseLookup::new(options),
            loaded: BTreeMap::new(),
            linked: LinkedSources::default(),
        }
    }

    /// Load each library after its dependencies, in source use order.
    fn bindings(&mut self, uses: Vec<UsePath>) -> Result<BTreeMap<String, Binding>, String> {
        let mut pending = vec![PendingModule::new(None, uses)];
        self.walk_imports(&mut pending).map_err(|error| {
            pending
                .iter()
                .rev()
                .filter_map(|m| m.source.as_ref())
                .fold(error, |e, source| source.target.context(e))
        })
    }

    /// Keep suspended importers on the heap while parsing each dependency.
    fn walk_imports(
        &mut self,
        pending: &mut Vec<PendingModule>,
    ) -> Result<BTreeMap<String, Binding>, String> {
        loop {
            let active_depth = pending.len() - 1;
            let current = pending.last_mut().expect("root is pending");
            let Some(path) = current.uses.next() else {
                let done = pending.pop().expect("current module");
                let Some(source) = done.source else {
                    return Ok(done.bindings);
                };
                let library = self
                    .finish_library(&source, done.bindings, done.depth)
                    .map_err(|e| source.target.context(e))?;
                let library = Rc::new(library);
                self.loaded
                    .insert(source.canonical_path, Rc::clone(&library));
                pending.last_mut().expect("library importer").bind_library(
                    source.binding,
                    &source.target,
                    library,
                )?;
                continue;
            };
            let target = self.lookup.resolve(&path)?;
            let leaf = path.binding().expect("validated use").to_owned();
            if current.bindings.contains_key(&leaf) {
                return Err(format!("duplicate use binding '{leaf}'"));
            }
            let Some(target) = target else {
                let subject = current.source.as_ref().map(|s| s.target.subject.as_str());
                let alias = qualified(subject, &leaf);
                current.bindings.insert(leaf, Binding::Dslx(alias));
                continue;
            };
            let canonical_path = fs::canonicalize(&target.path)
                .map_err(|e| format!("BX import '{}': {e}", target.path.display()))?;
            let cached = self.loaded.get(&canonical_path);
            if cached.is_none()
                && let Some(start) = pending.iter().position(|m| {
                    m.source
                        .as_ref()
                        .is_some_and(|s| s.canonical_path == canonical_path)
                })
            {
                let cycle = pending[start..]
                    .iter()
                    .filter_map(|m| m.source.as_ref())
                    .map(|s| s.target.subject.as_str())
                    .chain([target.subject.as_str()])
                    .collect::<Vec<_>>()
                    .join(" -> ");
                return Err(format!("cyclic BX import: {cycle}"));
            }
            if active_depth + cached.map_or(1, |lib| lib.depth) > MAX_ELABORATION_DEPTH {
                return Err(format!(
                    "BX import depth exceeds {MAX_ELABORATION_DEPTH} at '{}'",
                    target.subject
                ));
            }
            if let Some(library) = cached {
                pending.last_mut().expect("library importer").bind_library(
                    leaf,
                    &target,
                    Rc::clone(library),
                )?;
                continue;
            }
            let text =
                fs::read_to_string(&canonical_path).map_err(|e| target.context(e.to_string()))?;
            let uses = parser::module_uses(&text).map_err(|e| target.context(e))?;
            pending.push(PendingModule::new(
                Some(LibrarySource {
                    target,
                    canonical_path,
                    binding: leaf,
                    text,
                    uses: uses.clone(),
                }),
                uses,
            ));
        }
    }

    /// Parse and link a file after all its imported template signatures exist.
    fn finish_library(
        &mut self,
        source: &LibrarySource,
        bindings: BTreeMap<String, Binding>,
        depth: usize,
    ) -> Result<Library, String> {
        let module = self.parse_with_bindings(&source.text, &source.uses, &bindings)?;
        let names = Names::new(&module, Some(&source.target.subject), bindings)?;
        let exports = module
            .structs
            .iter()
            .map(|d| (&d.name, d.public))
            .chain(module.enums.iter().map(|d| (&d.name, d.public)))
            .chain(module.constants.iter().map(|d| (&d.name, d.public)))
            .chain(module.functions.iter().map(|d| (&d.name, d.public)))
            .chain(module.blocks.iter().map(|d| (&d.name, d.public)))
            .filter(|(_, public)| *public)
            .map(|(name, _)| {
                (
                    name.clone(),
                    Export {
                        name: names.declared(name),
                        parametrics: names.parametrics(name).to_vec(),
                        is_enum: names.enums.contains(name),
                        is_struct: names.structs.contains(name),
                    },
                )
            })
            .collect();
        self.append(&module, &names)?;
        self.linked
            .module_paths
            .insert(source.target.subject.clone(), source.target.path.clone());
        Ok(Library {
            subject: source.target.subject.clone(),
            exports,
            depth,
        })
    }

    /// Append definitions in dependency order while preserving order within a
    /// file.
    fn append(&mut self, source: &Module, names: &Names) -> Result<(), String> {
        let mut m = names::qualify(source, names)?;
        if let Some(Binding::Member(export)) = names.uses.get("zero!") {
            let use_pos = names.positions.get("zero!").copied();
            for (owner, block) in source
                .functions
                .iter()
                .map(|d| (&d.name, false))
                .chain(source.constants.iter().map(|d| (&d.name, false)))
                .chain(source.enums.iter().map(|d| (&d.name, false)))
                .chain(source.structs.iter().map(|d| (&d.name, false)))
                .chain(source.externs.iter().map(|d| (&d.name, false)))
                .chain(source.extern_blocks.iter().map(|d| (&d.name, false)))
                .chain(source.blocks.iter().map(|d| (&d.name, true)))
            {
                if block && !export.is_enum
                    || use_pos
                        .zip(names.positions.get(owner))
                        .is_none_or(|(u, &d)| u <= d)
                {
                    self.linked
                        .zero_uses
                        .insert(names.declared(owner), export.name.clone());
                }
            }
        }
        let linked = &mut self.linked.module;
        for path in &source.uses {
            if let Some(Binding::Dslx(alias)) =
                names.uses.get(path.binding().expect("validated use"))
            {
                linked.uses.push(path.clone());
                self.linked.dslx_bindings.push(alias.clone());
            }
        }
        for name in &source.declaration_order {
            match names.uses.get(name) {
                Some(Binding::Dslx(alias)) => linked.declaration_order.push(alias.clone()),
                Some(_) => { /* BX use visibility was checked in its source module. */ }
                None => linked.declaration_order.push(names.declared(name)),
            }
        }
        if source.declaration_order.is_empty() {
            linked.declaration_order.extend(
                m.enums
                    .iter()
                    .map(|d| d.name.clone())
                    .chain(m.externs.iter().map(|d| d.name.clone()))
                    .chain(m.constants.iter().map(|d| d.name.clone()))
                    .chain(m.functions.iter().map(|d| d.name.clone())),
            );
        }
        linked.structs.append(&mut m.structs);
        linked.enums.append(&mut m.enums);
        linked.constants.append(&mut m.constants);
        linked.externs.append(&mut m.externs);
        linked.functions.append(&mut m.functions);
        linked.extern_blocks.append(&mut m.extern_blocks);
        linked.blocks.append(&mut m.blocks);
        Ok(())
    }

    /// Imported template names disambiguate a comma inside angle brackets
    /// from a comma separating two comparisons.
    fn parse_with_bindings(
        &mut self,
        source: &str,
        uses: &[UsePath],
        bindings: &BTreeMap<String, Binding>,
    ) -> Result<Module, String> {
        let mut names = BTreeSet::new();
        if source.contains('<') {
            for path in uses {
                let head = path.binding().expect("validated use");
                match &bindings[head] {
                    Binding::Member(e) if !e.parametrics.is_empty() => {
                        names.insert(head.to_owned());
                    }
                    Binding::Module(lib) => {
                        names.extend(
                            lib.exports
                                .iter()
                                .filter(|(_, e)| !e.parametrics.is_empty())
                                .map(|(n, _)| format!("{head}::{n}")),
                        );
                    }
                    Binding::Dslx(_) => self.lookup.native.template_names(path, &mut names)?,
                    Binding::Member(_) => { /* Other members retain expression syntax. */ }
                }
            }
        }
        parser::parse_with_templates(source, &names)
    }

    fn finish(self) -> SourceImports<'a> {
        SourceImports {
            libraries: (!self.loaded.is_empty()).then_some(self.linked),
            native: self.lookup.native,
        }
    }
}

/// Resolve a parsed module's BX libraries, leaving pure DSLX import graphs on
/// the existing checker path.
pub(crate) fn link<'a>(
    module: &Module,
    options: &'a ImportOptions<'a>,
) -> Result<SourceImports<'a>, String> {
    let mut graph = Graph::new(options);
    let bindings = graph.bindings(module.uses.clone())?;
    if !graph.loaded.is_empty() {
        let names = Names::new(module, None, bindings)?;
        graph.append(module, &names)?;
    }
    Ok(graph.finish())
}

impl Graph<'_> {
    /// Parse the caller using the template signatures from this import
    /// snapshot.
    fn parse_root(&mut self, source: &str) -> Result<(Module, BTreeMap<String, Binding>), String> {
        if !source.contains("use") {
            return Ok((parser::parse_module(source)?, BTreeMap::new()));
        }
        let uses = parser::module_uses(source)?;
        let bindings = self.bindings(uses.clone())?;
        let module = self.parse_with_bindings(source, &uses, &bindings)?;
        Ok((module, bindings))
    }
}

/// Resolve imported template signatures before parsing the caller's
/// expressions.
pub(crate) fn parse_source(source: &str, options: &ImportOptions<'_>) -> Result<Module, String> {
    Graph::new(options)
        .parse_root(source)
        .map(|(module, _)| module)
}

/// Parse and resolve the caller against the same BX library snapshot.
pub(crate) fn read<'a>(
    source: &str,
    options: &'a ImportOptions<'a>,
) -> Result<(Module, SourceImports<'a>), String> {
    let mut graph = Graph::new(options);
    let (module, bindings) = graph.parse_root(source)?;
    if !graph.loaded.is_empty() {
        graph.append(&module, &Names::new(&module, None, bindings)?)?;
    }
    Ok((module, graph.finish()))
}

/// Render internal module-qualified symbols using BX source spelling.
pub(crate) fn diagnostic(message: String) -> String {
    if !message.contains("$bx$") {
        return message;
    }
    let mut parts = message.split("$bx$");
    let mut out = parts.next().unwrap_or_default().to_owned();
    for part in parts {
        if let Some((subject, rest)) = part.split_once('$') {
            out.push_str(&subject.replace('.', "::"));
            out.push_str("::");
            out.push_str(rest);
        } else {
            out.push_str("$bx$");
            out.push_str(part);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fs;
    use std::thread;

    use tempfile::tempdir;

    use super::{ImportOptions, MAX_ELABORATION_DEPTH, parse_source, read};
    use crate::ast::{BinaryOp, Expr, Type};

    #[test]
    fn linked_type_dimensions_follow_their_declaring_modules() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(dir.path().join("sizes.bx"), "pub const N = u32:3;")?;
        fs::write(
            dir.path().join("lib.bx"),
            r#"
use sizes::N;
const W = u32:8;
pub struct Bus { lanes: uN[W][N] }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let (_, imports) = read(
            "use lib; use sizes; block b(clk, x: lib::Bus) -> u8[sizes::N] { x.lanes }",
            &options,
        )?;
        let linked = imports.libraries.expect("two BX library imports").module;
        let Type::ParametricArray { element, size } = &linked.structs[0].fields[0].ty else {
            panic!("expected the library array dimension");
        };
        let Type::ParametricBits { width, .. } = &**element else {
            panic!("expected the library element width");
        };
        assert_eq!(width.as_ref(), &Expr::Name("$bx$lib$W".into()));
        assert_eq!(size.as_ref(), &Expr::Name("$bx$sizes$N".into()));
        let Type::ParametricArray {
            size: root_size, ..
        } = &linked.blocks[0].result
        else {
            panic!("expected the root result dimension");
        };
        assert_eq!(root_size, size);
        let error = read("struct Bus { lanes: u8[sizes::N] } use sizes;", &options)
            .err()
            .expect("the later import is out of scope");
        assert!(
            error.contains("use binding 'sizes' is declared later"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn library_function_type_formals_shadow_declarations_and_skip_explicit_slots()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("lib.bx"),
            r#"
struct T { field: u8 }
pub fn apply<F: fn(T) -> T, T: type, N: u32>(xs: T[N]) -> T {
    let x: T = xs[u32:0];
    F(x)
}
pub fn identity<T>(x: T) -> T { x }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let (_, imports) = read(
            r#"
use lib;
use lib::apply;
fn id(x:u8)->u8{x}
fn root(x:u8)->(u8,u8,u8) { (lib::apply<id,1>([x]), apply<id,1>([x]), lib::identity(x)) }
"#,
            &options,
        )?;
        let linked = imports.libraries.expect("BX library import").module;
        let apply = &linked.functions[0];
        let t = Type::Named("T".into());
        assert_eq!(apply.type_params, ["T"]);
        assert_eq!(apply.result, t);
        assert_eq!(
            apply.callback_params[0].params.as_slice(),
            std::slice::from_ref(&t)
        );
        assert_eq!(apply.callback_params[0].result, t);
        assert_eq!(apply.value_parametrics[0].index, 1);
        let Expr::Scope { bindings, .. } = &apply.body else {
            panic!("expected local let");
        };
        assert_eq!(bindings[0].ty.as_ref(), Some(&t));
        let Expr::Tuple(calls) = &linked.functions[3].body else {
            panic!("expected root calls");
        };
        assert_eq!(calls[0], calls[1]);
        assert!(
            matches!(&calls[0], Expr::ParametricCall { name, parametrics, .. } if name == "$bx$lib$apply" && matches!(&parametrics[..], [Expr::Name(callback), Expr::Literal { .. }] if callback == "id"))
        );
        assert!(matches!(&calls[2], Expr::Call { name, .. } if name == "$bx$lib$identity"));
        Ok(())
    }

    #[test]
    fn native_function_hints_distinguish_calls_and_comparisons() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("math.x"),
            r#"
pub fn resize<W: u32, N: u32>(x: uN[W]) -> uN[N] { x as uN[N] }
pub const LIMIT = u8:7;
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let module = parse_source(
            r#"
use math;
use math::{resize, LIMIT};
fn direct(x:u8)->u12 { resize<8,12>(x) }
fn qualified(xs:u8[1])->u12[1] { map(xs, math::resize<8,12>) }
fn compare(x:u8, y:u8)->(bool,bool) { (math::LIMIT<x, y>(x)) }
fn compare_direct(x:u8, y:u8)->(bool,bool) { (LIMIT<x, y>(x)) }
fn shadow(resize:u8, x:u8, y:u8)->(bool,bool) { (resize<x, y>(x)) }
"#,
            &options,
        )?;
        assert!(
            matches!(&module.functions[0].body, Expr::ParametricCall { name, parametrics, .. } if name == "resize" && parametrics.len() == 2)
        );
        assert!(
            matches!(&module.functions[1].body, Expr::NamedMap { callback, parametrics, .. } if callback == "math::resize" && parametrics.len() == 2)
        );
        for function in &module.functions[2..] {
            assert!(
                matches!(&function.body, Expr::Tuple(pair) if matches!(&pair[..], [
                    Expr::Binary { op: BinaryOp::Lt, .. },
                    Expr::Binary { op: BinaryOp::Gt, .. }
                ])),
                "{}",
                function.name
            );
        }
        Ok(())
    }

    #[test]
    fn native_parametrics_in_bx_libraries_use_one_source_snapshot() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        let math = dir.path().join("math.x");
        fs::write(
            &math,
            "pub fn resize<W: u32, N: u32>(x: uN[W]) -> uN[N] { x as uN[N] }",
        )?;
        fs::write(
            dir.path().join("lib.bx"),
            r#"
use math::resize;
use math;
const W = u32:8;
const N = u32:12;
pub fn apply(x:u8)->(u12,u12[1]) { (resize<W,N>(x), map([x], math::resize<W,N>)) }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let (module, imports) = read(
            "use lib; fn f(x:u8)->(u12,u12[1]) { lib::apply(x) }",
            &options,
        )?;
        fs::remove_file(math)?;
        let checked = crate::check::check_sources(&module, imports, &options)?;
        let package = crate::lower_module(
            &checked,
            crate::empty_package("circuit"),
            crate::ResetBehavior::default(),
        )?;
        xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
        Ok(())
    }

    #[test]
    fn import_depth_limit_includes_cached_dependencies_on_a_worker_stack()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        for i in 0..=MAX_ELABORATION_DEPTH {
            let mut source = String::new();
            if i < MAX_ELABORATION_DEPTH {
                source = format!("use lib{}; ", i + 1);
            }
            source.push_str("pub fn f(x: u8) -> u8 { x }");
            fs::write(dir.path().join(format!("lib{i}.bx")), source)?;
        }
        thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(move || -> Result<(), String> {
                let options = ImportOptions {
                    additional_search_paths: vec![dir.path()],
                    ..Default::default()
                };
                let (_, linked) = read("use lib1;", &options)?;
                assert_eq!(
                    linked
                        .libraries
                        .expect("loaded libraries")
                        .module
                        .functions
                        .len(),
                    MAX_ELABORATION_DEPTH
                );
                for (source, failing_import) in [
                    ("use lib0;", format!("lib{MAX_ELABORATION_DEPTH}")),
                    ("use lib1; use lib0;", "lib1".to_owned()),
                ] {
                    let error = read(source, &options)
                        .err()
                        .expect("one extra import level");
                    assert!(
                        error.ends_with(&format!(
                            "BX import depth exceeds {MAX_ELABORATION_DEPTH} at '{failing_import}'"
                        )),
                        "{error}"
                    );
                }
                Ok(())
            })?
            .join()
            .expect("BX import worker")?;
        Ok(())
    }
}
