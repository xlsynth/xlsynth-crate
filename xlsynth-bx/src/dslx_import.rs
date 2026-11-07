// SPDX-License-Identifier: Apache-2.0

//! Resolve DSLX imports and preserve the concrete IR functions selected by BX.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use xlsynth::dslx::{self, ImportData, MatchableModuleMember, TypecheckedModule};
use xlsynth_pir::ir::{NodePayload, Package};
use xlsynth_pir::ir_parser;

use crate::ImportOptions;
use crate::ast::{StructDef, Type, UsePath};
use crate::check::MAX_TYPE_SIZE;
use crate::import_structs::ImportedStructs;
use crate::parser::{
    MAX_SYNTAX_DEPTH, is_dslx_member_name, is_identifier_word, is_use_identifier, parse_value_type,
};

/// An imported DSLX function specialized for one set of BX argument types.
#[derive(Clone, Debug)]
pub struct ImportedCall {
    /// Full dotted DSLX module name, independent of the BX use spelling.
    pub module: String,
    /// Unqualified DSLX function name in the imported module.
    pub function: String,
    /// Concrete argument types in call order.
    pub args: Vec<Type>,
    /// Concrete result type inferred by DSLX.
    pub result: Type,
}

/// Concrete signatures and IR bodies captured from the same DSLX import graph.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResolvedImports {
    pub(crate) calls: Vec<ImportedCall>,
    pub(crate) batches: Vec<ImportBatch>,
}

/// Reachable IR functions from one canonical DSLX module. Each root symbol
/// corresponds to a call ID in the same position.
#[derive(Clone, Debug)]
pub(crate) struct ImportBatch {
    pub(crate) package: Package,
    pub(crate) roots: Vec<String>,
    pub(crate) call_ids: Vec<usize>,
}

/// A use leaf names either a module or a member of its parent module.
enum UseBinding {
    Module(String),
    MemberOf(String),
}

/// Per-check import graph and memoized concrete function signatures.
pub(crate) struct ImportResolver {
    bindings: BTreeMap<String, UseBinding>,
    subjects: BTreeSet<String>,
    imported_modules: BTreeSet<String>,
    stdlib_path: PathBuf,
    imports: Option<ImportData>,
    structs: ImportedStructs,
    ids: HashMap<(String, String, Vec<Type>), usize>,
    calls: Vec<ImportedCall>,
}

impl ImportResolver {
    /// Typecheck every import up front, including unused modules, so all calls
    /// in this BX module use a single snapshot of the DSLX sources.
    pub(crate) fn new(
        declarations: &[UsePath],
        options: &ImportOptions<'_>,
    ) -> Result<Self, String> {
        let stdlib_path = options
            .dslx_stdlib_path
            .unwrap_or(xlsynth::default_dslx_stdlib_path());
        for path in
            std::iter::once(stdlib_path).chain(options.additional_search_paths.iter().copied())
        {
            let text = path
                .to_str()
                .ok_or_else(|| format!("DSLX import path must be UTF-8: {path:?}"))?;
            if text.as_bytes().contains(&0) {
                return Err(format!("DSLX import path contains a NUL byte: {path:?}"));
            }
        }
        let mut bindings = BTreeMap::new();
        let mut subjects = BTreeSet::new();
        let mut imported_modules = BTreeSet::new();
        let mut imports = None;
        for (index, declaration) in declarations.iter().enumerate() {
            let Some((leaf, parents)) = declaration.components.split_last() else {
                return Err("empty DSLX use path".into());
            };
            if declaration.components.len() > MAX_SYNTAX_DEPTH {
                return Err(format!(
                    "DSLX use path exceeds {MAX_SYNTAX_DEPTH} components"
                ));
            }
            if !declaration
                .components
                .iter()
                .all(|name| is_use_identifier(name))
            {
                return Err(format!(
                    "invalid DSLX use path {:?}",
                    declaration.components
                ));
            }
            if bindings.contains_key(leaf) {
                return Err(format!("duplicate use binding '{leaf}'"));
            }
            let data = imports.get_or_insert_with(|| {
                ImportData::new(Some(stdlib_path), &options.additional_search_paths)
            });
            let subject = declaration.components.join(".");
            let use_path = declaration.components.join("::");
            let module_source = format!("import {subject} as bx_module;");
            let probe_name = format!("0bx_use_{index}");
            // A failed speculative typecheck can invalidate native inference
            // state. Classify ambiguous paths outside the shared import graph.
            let is_module = parents.is_empty() || {
                let mut scratch =
                    ImportData::new(Some(stdlib_path), &options.additional_search_paths);
                check_probe(&module_source, &probe_name, &mut scratch).is_ok()
            };
            let (source, binding) = if is_module {
                // The pinned v2 frontend handles a module leaf via ordinary
                // import.
                subjects.insert(subject.clone());
                (module_source, UseBinding::Module(subject))
            } else {
                // Native use resolution still prefers an existing full module,
                // so a malformed module cannot fall through to a parent's
                // member.
                // A DSLX module cannot mix use and import syntax. Check the
                // member here, then inspect its owner's import graph below.
                check_probe(
                    &format!("#![feature(use_syntax)]\nuse {use_path};"),
                    &format!("0bx_member_{index}"),
                    data,
                )
                .map_err(|e| format!("checking use '{use_path}': {e}"))?;
                let parent = parents.join(".");
                subjects.insert(parent.clone());
                (
                    format!("import {parent} as bx_module;"),
                    UseBinding::MemberOf(parent),
                )
            };
            check_probe(&source, &probe_name, data)
                .map_err(|e| format!("checking use '{use_path}': {e}"))?
                .collect_import_subjects(&mut imported_modules);
            bindings.insert(leaf.clone(), binding);
        }
        Ok(Self {
            bindings,
            structs: ImportedStructs::new(&subjects, stdlib_path, &options.additional_search_paths),
            subjects,
            imported_modules,
            stdlib_path: stdlib_path.to_owned(),
            imports,
            ids: HashMap::new(),
            calls: Vec::new(),
        })
    }

    /// Map a qualified module use or an unqualified member use to its DSLX
    /// module and member. Other names belong to BX declarations.
    pub(crate) fn lookup(&self, source_name: &str) -> Result<Option<(String, String)>, String> {
        if let Some((head, member)) = source_name.split_once("::") {
            return match self.bindings.get(head) {
                Some(UseBinding::Module(subject)) => Ok(Some((subject.clone(), member.to_owned()))),
                Some(UseBinding::MemberOf(_)) => Err(format!(
                    "use binding '{head}' is a DSLX member and cannot qualify '{source_name}'"
                )),
                None => Ok(None),
            };
        }
        match self.bindings.get(source_name) {
            Some(UseBinding::MemberOf(subject)) => {
                Ok(Some((subject.clone(), source_name.to_owned())))
            }
            Some(UseBinding::Module(_)) => Err(format!(
                "use binding '{source_name}' names a DSLX module and requires a member"
            )),
            None => Ok(None),
        }
    }

    /// Resolve a source-level type reference using the declared import graph.
    pub(crate) fn resolve_named(
        &mut self,
        written: &str,
        subject: &str,
        member: &str,
    ) -> Result<Type, String> {
        if !self.subjects.contains(subject) || !is_dslx_member_name(member) {
            return Err(format!(
                "invalid imported type reference '{subject}::{member}'"
            ));
        }
        let imports = self.imports.as_mut().expect("declared DSLX import graph");
        self.structs
            .resolve_written(written, subject, member, imports)
            .map(Type::Named)
    }

    /// Struct layouts discovered through named references and inferred call
    /// results.
    pub(crate) fn struct_defs(&self) -> &[StructDef] {
        self.structs.definitions()
    }

    /// Ask the DSLX typechecker to specialize the function from concrete data
    /// arguments. The checked call supplies its inferred and default bindings.
    pub(crate) fn resolve(
        &mut self,
        module: &str,
        function: &str,
        mut args: Vec<Type>,
    ) -> Result<(usize, Type), String> {
        if !self.subjects.contains(module) {
            return Err(format!("undeclared DSLX import {module:?}"));
        }
        if !is_identifier_word(function) || !module.split('.').all(is_identifier_word) {
            return Err(format!(
                "cannot link '{module}::{function}': XLS IR function symbols require module and function names made of letters, digits, and underscores"
            ));
        }
        for arg in &mut args {
            normalize_import_type(arg, self.struct_defs())
                .map_err(|e| format!("{module}::{function}: {e}"))?;
        }
        let key = (module.to_owned(), function.to_owned(), args);
        if let Some(&id) = self.ids.get(&key) {
            return Ok((id, self.calls[id].result.clone()));
        }
        let id = self.calls.len();
        let mut probe_imports = ProbeImports::default();
        let (params, expr) =
            probe_call(&mut probe_imports, &self.structs, module, function, &key.2)?;
        let witness = TypeWitness {
            imports: probe_imports.declarations(),
            params,
            expr,
        };
        let source = format!(
            r#"#![feature(generics)]
{}
fn capture<T: type>(_value: T) {{}}
fn probe_{id}({}) {{ capture({}) }}
"#,
            witness.imports, witness.params, witness.expr,
        );
        let data = self
            .imports
            .as_mut()
            .expect("a declared subject has an import graph");
        let result = check_probe(&source, &format!("0bx_signature_{id}"), data)
            .and_then(|checked| {
                let capture = ProbeCapture::new(&checked)?;
                let ty = capture.param_type(0)?;
                if needs_struct_names(&ty)? {
                    let graph = checked
                        .get_type_info()
                        .build_function_call_graph(&checked.get_module())
                        .map_err(|e| e.to_string())?;
                    for i in 0..graph.callee_count(&capture.probe) {
                        if let Some(callee) = graph.get_callee(&capture.probe, i)
                            && !callee.is_same_definition(&capture.capture)
                        {
                            self.structs
                                .observe_module(&callee.get_body().get_owner_module(), &checked);
                            if let Some(annotation) = callee.get_return_type() {
                                self.structs.observe_annotation(annotation, 0);
                            }
                        }
                    }
                }
                self.structs.reflect(&ty, data, &witness)
            })
            .map_err(|e| format!("{module}::{function}: {e}"))?;
        self.calls.push(ImportedCall {
            module: key.0.clone(),
            function: key.1.clone(),
            args: key.2.clone(),
            result: result.clone(),
        });
        self.ids.insert(key, id);
        Ok((id, result))
    }

    /// Compile each imported module's requested specializations together.
    /// Separate packages keep DSLX names such as `a.b` and `a_b` distinct
    /// until BX links their generated IR symbols into the destination.
    pub(crate) fn finish(
        mut self,
        mut on_import: impl FnMut(TypecheckedModule),
        on_sources: impl FnOnce(ImportedStructs, BTreeSet<String>),
    ) -> Result<ResolvedImports, String> {
        let mut batches = Vec::new();
        if let Some(data) = self.imports.as_mut() {
            let mut groups = BTreeMap::<&str, Vec<usize>>::new();
            for (id, call) in self.calls.iter().enumerate() {
                groups.entry(&call.module).or_default().push(id);
            }
            for (index, (subject, ids)) in groups.into_iter().enumerate() {
                let (batch, checked) = compile_batch(
                    subject,
                    index,
                    &self.calls,
                    ids,
                    data,
                    &self.stdlib_path,
                    &self.structs,
                )
                .map_err(|e| format!("compiling DSLX import '{subject}': {e}"))?;
                batches.push(batch);
                on_import(checked);
            }
        }
        on_sources(self.structs, self.imported_modules);
        Ok(ResolvedImports {
            calls: self.calls,
            batches,
        })
    }
}

/// A probe module starts with a digit, which DSLX source cannot import and
/// which real module names cannot acquire through dot-to-underscore mangling.
pub(crate) fn check_probe(
    source: &str,
    name: &str,
    imports: &mut ImportData,
) -> Result<TypecheckedModule, String> {
    dslx::parse_and_typecheck(source, &format!("{name}.x"), name, imports)
        .map_err(|e| e.to_string())
}

/// Validate structural import arguments and canonicalize DSLX's bool/u1 alias.
fn normalize_import_type(ty: &mut Type, structs: &[StructDef]) -> Result<(), String> {
    match ty {
        Type::Bool => {
            *ty = Type::Bits {
                signed: false,
                width: 1,
            }
        }
        Type::Tuple(items) => items
            .iter_mut()
            .try_for_each(|item| normalize_import_type(item, structs))?,
        Type::Array { element, .. } => normalize_import_type(element, structs)?,
        Type::Named(name) if !structs.iter().any(|s| s.name == *name) => {
            return Err(format!(
                "unsupported nominal type '{name}' in imported function signature"
            ));
        }
        Type::Bits { .. } | Type::Named(_) => {
            // Bits and resolved DSLX structs already have their canonical type.
        }
    }
    Ok(())
}

/// Module aliases shared by every qualified type and call in a generated DSLX
/// probe.
#[derive(Default)]
struct ProbeImports {
    aliases: BTreeMap<String, usize>,
}

impl ProbeImports {
    fn qualify(&mut self, subject: &str, member: &str) -> String {
        let next = self.aliases.len();
        let index = self.aliases.entry(subject.to_owned()).or_insert(next);
        format!("bx_module_{index}::{member}")
    }

    fn declarations(&self) -> String {
        self.aliases
            .iter()
            .map(|(subject, index)| format!("import {subject} as bx_module_{index};\n"))
            .collect()
    }

    /// Render nominal arguments using aliases valid in the generated module.
    fn render_type(&mut self, ty: &Type, structs: &ImportedStructs) -> Result<String, String> {
        match ty {
            Type::Named(name) => {
                let (subject, member) = structs
                    .public_spelling(name)?
                    .split_once("::")
                    .ok_or_else(|| format!("missing DSLX import for struct '{name}'"))?;
                Ok(self.qualify(subject, member))
            }
            Type::Tuple(items) => {
                let members = items
                    .iter()
                    .map(|t| self.render_type(t, structs))
                    .collect::<Result<Vec<_>, _>>()?;
                let trailing = if members.len() == 1 { "," } else { "" };
                Ok(format!("({}{trailing})", members.join(", ")))
            }
            Type::Array { element, size } => {
                Ok(format!("{}[{size}]", self.render_type(element, structs)?))
            }
            Type::Bits { .. } | Type::Bool => Ok(ty.to_string()),
        }
    }
}

/// Build parameter declarations and a call expression usable in a DSLX probe.
fn probe_call(
    imports: &mut ProbeImports,
    structs: &ImportedStructs,
    subject: &str,
    function: &str,
    args: &[Type],
) -> Result<(String, String), String> {
    let function = imports.qualify(subject, function);
    let mut params = Vec::with_capacity(args.len());
    let mut arguments = Vec::with_capacity(args.len());
    for (i, ty) in args.iter().enumerate() {
        let name = format!("arg_{i}");
        params.push(format!("{name}: {}", imports.render_type(ty, structs)?));
        arguments.push(name);
    }
    Ok((
        params.join(", "),
        format!("{function}({})", arguments.join(", ")),
    ))
}

/// Only nominal results need the public spellings in the callee's module.
/// The pinned C API provides native struct/array accessors. Only opaque
/// tuples need their display inspected to detect nested nominal members.
fn needs_struct_names(ty: &dslx::Type) -> Result<bool, String> {
    let mut leaf = ty.clone();
    loop {
        if leaf.is_struct() {
            return Ok(true);
        }
        if leaf.is_enum() || leaf.is_bits_like().is_some() {
            return Ok(false);
        }
        if leaf.is_array() {
            leaf = leaf.get_array_element_type();
        } else {
            let text = leaf.to_string().map_err(|e| e.to_string())?;
            return Ok(text.starts_with('(') && parse_value_type(&text).is_err());
        }
    }
}

/// A DSLX expression whose resolved type is being reflected into BX.
pub(crate) struct TypeWitness {
    pub(crate) imports: String,
    pub(crate) params: String,
    pub(crate) expr: String,
}

impl TypeWitness {
    pub(crate) fn project(&self, suffix: &str) -> Self {
        Self {
            imports: self.imports.clone(),
            params: self.params.clone(),
            expr: format!("({}){suffix}", self.expr),
        }
    }

    /// libxls v0.61 exposes concrete struct and array members but only
    /// *written* tuple annotations. Ask DSLX to destructure the actual
    /// tuple and capture the type of each binding, including its original
    /// nominal declaration.
    pub(crate) fn tuple_members(
        &self,
        ty: &dslx::Type,
        probe_index: usize,
        imports: &mut ImportData,
    ) -> Result<Vec<dslx::Type>, String> {
        // Only arity comes from the display, and DSLX checks it again when
        // destructuring. A struct's displayed name never identifies a type.
        let text = ty.to_string().map_err(|e| e.to_string())?;
        let body = text
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| format!("expected a DSLX tuple, found '{text}'"))?;
        let mut depth = 0usize;
        let mut arity = usize::from(!body.is_empty());
        for c in body.chars() {
            match c {
                '(' | '{' | '[' => depth += 1,
                ')' | '}' | ']' => {
                    depth = depth
                        .checked_sub(1)
                        .ok_or("unbalanced DSLX tuple display")?
                }
                ',' if depth == 0 => arity += 1,
                _ => { /* Nested separators and other characters do not affect arity. */ }
            }
        }
        // Even zero-bit elements count toward BX's layout budget, as does
        // the enclosing tuple. Check before allocating the capture parameters.
        if arity >= MAX_TYPE_SIZE {
            return Err(format!(
                "DSLX tuple exceeds maximum concrete type size {MAX_TYPE_SIZE}"
            ));
        }
        if arity == 0 || depth != 0 {
            return Err(format!("expected a nonempty DSLX tuple, found '{text}'"));
        }
        let bindings: Vec<_> = (0..arity).map(|i| format!("v{i}")).collect();
        let types: Vec<_> = (0..arity).map(|i| format!("T{i}: type")).collect();
        let params: Vec<_> = (0..arity).map(|i| format!("_v{i}: T{i}")).collect();
        let args = bindings.join(", ");
        // A runtime index permits type-only projections through empty arrays.
        let source = format!(
            r#"#![feature(generics)]
{}
fn capture<{}>({}) {{}}
fn probe(_bx_array_index: u32, {}) {{ let ({args},) = {}; capture({args}) }}
"#,
            self.imports,
            types.join(", "),
            params.join(", "),
            self.params,
            self.expr,
        );
        let checked = check_probe(&source, &format!("0bx_tuple_{probe_index}"), imports)?;
        let capture = ProbeCapture::new(&checked)?;
        (0..arity).map(|index| capture.param_type(index)).collect()
    }
}

/// The generated probe and the specialization of its sole generic capture call.
struct ProbeCapture {
    probe: dslx::Function,
    capture: dslx::Function,
    context: dslx::TypeInfo,
}

impl ProbeCapture {
    fn new(checked: &TypecheckedModule) -> Result<Self, String> {
        let module = checked.get_module();
        let mut functions = (0..module.get_member_count()).filter_map(|i| {
            match module.get_member(i).to_matchable() {
                Some(MatchableModuleMember::Function(f)) => Some(f),
                _ => None,
            }
        });
        let capture = functions.next().ok_or("missing DSLX capture function")?;
        let probe = functions.next().ok_or("missing DSLX signature probe")?;
        let root = checked.get_type_info();
        if root
            .requires_implicit_token(&probe)
            .map_err(|e| e.to_string())?
        {
            return Err("function requires an implicit token and cannot be lifted into BX".into());
        }
        let context = root
            .get_unique_invocation_callee_data(&capture)
            .and_then(|entries| entries.get(0))
            .and_then(|entry| entry.derived_type_info())
            .ok_or("missing DSLX type information for the capture invocation")?;
        Ok(Self {
            probe,
            capture,
            context,
        })
    }

    fn param_type(&self, index: usize) -> Result<dslx::Type, String> {
        self.context
            .get_type_for_type_annotation(&self.capture.get_param(index).get_type_annotation())
            .ok_or_else(|| "could not resolve the captured DSLX type".to_string())
    }
}

/// Recover each specialization's exact IR symbol from its generated probe.
fn compile_batch(
    subject: &str,
    index: usize,
    calls: &[ImportedCall],
    call_ids: Vec<usize>,
    imports: &mut ImportData,
    stdlib_path: &Path,
    structs: &ImportedStructs,
) -> Result<(ImportBatch, TypecheckedModule), String> {
    let mut probe_imports = ProbeImports::default();
    let mut probes = String::new();
    for &id in &call_ids {
        let call = &calls[id];
        let (params, expr) = probe_call(
            &mut probe_imports,
            structs,
            subject,
            &call.function,
            &call.args,
        )?;
        probes.push_str(&format!("fn probe_{id}({params}) {{ let _ = {expr}; }}\n"));
    }
    let source = probe_imports.declarations() + &probes;
    let module = format!("0bx_batch_{index}");
    let checked = check_probe(&source, &module, imports)?;
    let mut ir = checked.to_ir_text().map_err(|e| e.to_string())?;
    // libxls accepts a numeric initial in a synthetic DSLX module name.
    // Give just the enclosing PIR package a legal initial; the generated
    // function symbols retain their collision-free prefix.
    let header = "package ";
    if !ir.starts_with(&format!("{header}{module}\n")) {
        return Err("unexpected DSLX probe package header".into());
    }
    ir.replace_range(header.len()..header.len() + 1, "_");
    let mut package = ir_parser::Parser::new(&ir)
        .parse_xls_package()
        .map_err(|e| format!("reading imported functions as PIR: {e}"))?;
    for path in package.file_table.id_to_path.values_mut() {
        if let Ok(relative) = Path::new(path).strip_prefix(stdlib_path) {
            let relative = relative
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            *path = format!("xls/dslx/stdlib/{relative}");
        }
    }
    let functions: HashMap<_, _> = package
        .members
        .iter()
        .map(|m| (m.graph().name.as_str(), m.graph()))
        .collect();
    let mut roots = Vec::with_capacity(call_ids.len());
    for &id in &call_ids {
        let call = &calls[id];
        let name = xlsynth::mangle_dslx_name(&module, &format!("probe_{id}"))
            .map_err(|e| e.to_string())?;
        let probe = functions
            .get(name.as_str())
            .ok_or_else(|| format!("missing compiled probe for {subject}::{}", call.function))?;
        let mut invokes = probe.nodes.iter().filter_map(|n| match &n.payload {
            NodePayload::Invoke { to_apply, .. } => Some(to_apply),
            _ => None,
        });
        let symbol = invokes
            .next()
            .ok_or_else(|| format!("missing invoke for {subject}::{}", call.function))?;
        if invokes.next().is_some() {
            return Err(format!(
                "ambiguous compiled probe for {subject}::{}",
                call.function
            ));
        }
        roots.push(symbol.clone());
    }
    crate::import_link::retain_reachable_functions(&mut package, &roots)?;
    Ok((
        ImportBatch {
            package,
            roots,
            call_ids,
        },
        checked,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn import_argument_types_use_dslx_widths_and_reject_nested_nominals() {
        let module =
            crate::parse_module("extern fn f(x: (uN[0], s64, uN[65], (Packet,)[2])) -> ();")
                .unwrap();
        let mut ty = module.externs[0].params[0].ty.clone();
        assert_eq!(ty.to_string(), "(uN[0], s64, uN[65], (Packet,)[2])");
        assert_eq!(
            normalize_import_type(&mut ty, &[]).unwrap_err(),
            "unsupported nominal type 'Packet' in imported function signature"
        );
    }

    #[test]
    fn invalid_use_components_from_a_programmatic_ast_are_rejected_before_loading_dslx() {
        for name in [
            "",
            ".m",
            "m..n",
            "foo::bar",
            "x\0y",
            "f(); } fn injected() {",
        ] {
            let declaration = UsePath {
                components: vec![name.to_owned()],
            };
            assert!(ImportResolver::new(&[declaration], &ImportOptions::default()).is_err());
        }
        let empty = UsePath {
            components: Vec::new(),
        };
        assert!(ImportResolver::new(&[empty], &ImportOptions::default()).is_err());
        let deep = UsePath {
            components: vec!["x".into(); MAX_SYNTAX_DEPTH + 1],
        };
        let Err(error) = ImportResolver::new(&[deep], &ImportOptions::default()) else {
            panic!("expected an excessive use path to fail");
        };
        assert_eq!(
            error,
            format!("DSLX use path exceeds {MAX_SYNTAX_DEPTH} components")
        );
    }

    #[test]
    fn a_module_shadows_a_same_named_member_even_when_the_module_is_invalid()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::create_dir(dir.path().join("fabric"))?;
        fs::write(
            dir.path().join("fabric.x"),
            "pub fn leaf() -> u8 { u8:1 } pub fn plain() -> u8 { u8:2 }",
        )?;
        let leaf = dir.path().join("fabric/leaf.x");
        fs::write(&leaf, "pub fn child() -> u8 { u8:3 }")?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let declarations = crate::parse_module("use fabric::{leaf, plain};")?.uses;
        let resolver = ImportResolver::new(&declarations, &options)?;
        assert_eq!(
            resolver.lookup("leaf::child")?,
            Some(("fabric.leaf".into(), "child".into())),
        );
        assert_eq!(
            resolver.lookup("plain")?,
            Some(("fabric".into(), "plain".into())),
        );
        assert!(resolver.lookup("leaf").is_err());
        assert!(resolver.lookup("plain::child").is_err());
        assert_eq!(resolver.lookup("local")?, None);

        fs::write(&leaf, "pub fn child() -> u8 { missing }")?;
        assert!(ImportResolver::new(&declarations, &options).is_err());
        for source in ["use missing;", "use fabric::missing;"] {
            let module = crate::parse_module(source)?;
            assert!(
                ImportResolver::new(&module.uses, &options).is_err(),
                "{source}"
            );
        }
        Ok(())
    }

    #[test]
    fn unused_member_uses_resolve_without_exposing_private_types_or_functions()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("device.x"),
            "struct Hidden { x: u8 } fn hidden(x: u8) -> u8 { x }",
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let declarations = "use device::{Hidden, hidden};";
        let unused =
            crate::parse_module(&format!("{declarations} block b(clk, x: u8) -> u8 {{ x }}"))?;
        crate::check_module_with_options(&unused, &options)?;
        for block in [
            "block b(clk, x: Hidden) -> u8 { x.x }",
            "block b(clk, x: u8) -> u8 { hidden(x) }",
        ] {
            let used = crate::parse_module(&format!("{declarations} {block}"))?;
            let error = crate::check_module_with_options(&used, &options).unwrap_err();
            assert!(error.contains("public"), "{error}");
        }
        Ok(())
    }

    #[test]
    fn use_bound_calls_diagnose_unlinkable_ir_names() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        for (subject, function) in [("logic'", "identity"), ("logic", "identity'")] {
            fs::write(
                dir.path().join(format!("{subject}.x")),
                format!("pub fn {function}(x: u8) -> u8 {{ x }}"),
            )?;
            let block = crate::parse_module(&format!(
                "use {subject}::{function}; block b(clk, x: u8) -> u8 {{ {function}(x) }}"
            ))?;
            assert_eq!(
                crate::check_module_with_options(&block, &options).unwrap_err(),
                format!(
                    "block 'b': result: cannot link '{subject}::{function}': XLS IR function symbols require module and function names made of letters, digits, and underscores"
                )
            );
        }
        Ok(())
    }

    #[test]
    fn checked_batch_retains_only_runtime_callees() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        // A user dependency can have the name of an ordinary generated module
        // and define a function with the name of the generated probe.
        fs::write(
            dir.path().join("bx_batch_0.x"),
            r#"#[extern_verilog("external_bound {fn} (.out({return}));")]
fn bound() -> u32 { u32:2 }
pub fn probe_0(x: u8) -> u8 {
    for (_i, acc): (u32, u8) in u32:0..bound() { acc + x }(x)
}"#,
        )?;
        fs::write(
            dir.path().join("wrapper.x"),
            "import bx_batch_0; pub fn apply(x: u8) -> u8 { bx_batch_0::probe_0(x) }",
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let module =
            crate::parse_module("use wrapper; block b(clk, x: u8) -> u8 { wrapper::apply(x) }")?;
        let checked = crate::check_module_with_options(&module, &options)?;
        let batch = &checked.imports.batches[0];
        assert_eq!(batch.roots, ["__wrapper__apply"]);
        // DSLX folds bound() into the trip count. The snapshot keeps apply,
        // the user's probe_0 and its loop body.
        assert!(batch.package.get_fn("__bx_batch_0__probe_0").is_some());
        assert!(batch.package.get_fn("__bx_batch_0__bound").is_none());
        assert_eq!(batch.package.members.len(), 3);
        xlsynth_pir::ir_verify::verify_package(&batch.package)?;
        Ok(())
    }
}
