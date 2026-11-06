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
use crate::parser::{MAX_SYNTAX_DEPTH, is_dslx_member_name, is_identifier_word, is_use_identifier};

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
    #[cfg(test)]
    type_witnesses: usize,
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
        let mut names = BTreeSet::new();
        for declaration in declarations {
            let Some(leaf) = declaration.components.last() else {
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
            if !names.insert(leaf) {
                return Err(format!("duplicate use binding '{leaf}'"));
            }
        }
        let (structs, imports) =
            ImportedStructs::load(declarations, stdlib_path, &options.additional_search_paths)?;
        let mut bindings = BTreeMap::new();
        let mut subjects = BTreeSet::new();
        for declaration in declarations {
            let (leaf, parents) = declaration.components.split_last().expect("validated use");
            let subject = declaration.components.join(".");
            let binding = if structs.has_module(&subject) {
                subjects.insert(subject.clone());
                UseBinding::Module(subject)
            } else {
                let parent = parents.join(".");
                subjects.insert(parent.clone());
                UseBinding::MemberOf(parent)
            };
            bindings.insert(leaf.clone(), binding);
        }
        Ok(Self {
            bindings,
            structs,
            subjects,
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

    /// Check visibility without typechecking an invocation in an empty fold.
    pub(crate) fn check_member_name(&self, subject: &str, member: &str) -> Result<(), String> {
        if self.structs.has_public_member(subject, member) {
            Ok(())
        } else {
            Err(format!(
                "unknown or private DSLX member '{subject}::{member}'"
            ))
        }
    }

    /// Resolve a source-level type reference using the declared import graph.
    pub(crate) fn resolve_named(&mut self, subject: &str, member: &str) -> Result<Type, String> {
        if !self.subjects.contains(subject) || !is_dslx_member_name(member) {
            return Err(format!(
                "invalid imported type reference '{subject}::{member}'"
            ));
        }
        let imports = self.imports.as_mut().expect("declared DSLX import graph");
        self.structs
            .resolve_named(subject, member, imports)
            .map_err(|e| self.structs.remap_diagnostic(e))
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
        let data = self
            .imports
            .as_mut()
            .expect("a declared subject has an import graph");
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
        let result = self
            .structs
            .check_probe(&source, &format!("0bx_signature_{id}"), data)
            .and_then(|checked| {
                let capture = ProbeCapture::new(&checked)?;
                let ty = capture.param_type(0)?;
                self.structs
                    .reflect_result(module, function, &ty, data, &witness)
            })
            .map_err(|e| format!("{module}::{function}: {}", self.structs.remap_diagnostic(e)))?;
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
    pub(crate) fn finish(mut self) -> Result<ResolvedImports, String> {
        let mut batches = Vec::new();
        if let Some(data) = self.imports.as_mut() {
            let mut groups = BTreeMap::<&str, Vec<usize>>::new();
            for (id, call) in self.calls.iter().enumerate() {
                groups.entry(&call.module).or_default().push(id);
            }
            for (index, (subject, ids)) in groups.into_iter().enumerate() {
                let batch = compile_batch(
                    subject,
                    index,
                    &self.calls,
                    ids,
                    data,
                    &self.stdlib_path,
                    &self.structs,
                )
                .map_err(|e| {
                    format!(
                        "compiling DSLX import '{subject}': {}",
                        self.structs.remap_diagnostic(e)
                    )
                })?;
                batches.push(batch);
            }
        }
        Ok(ResolvedImports {
            calls: self.calls,
            batches,
            #[cfg(test)]
            type_witnesses: self.structs.type_witnesses(),
        })
    }
}

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
        structs: &ImportedStructs,
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
        let checked = structs.check_probe(&source, &format!("0bx_tuple_{probe_index}"), imports)?;
        let capture = ProbeCapture::new(&checked)?;
        (0..arity).map(|index| capture.param_type(index)).collect()
    }
}

/// The specialization of the generic capture call in a generated DSLX probe.
struct ProbeCapture {
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
        Ok(Self { capture, context })
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
) -> Result<ImportBatch, String> {
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
    let checked = structs.check_probe(&source, &module, imports)?;
    // XLS can embed filenames in generated symbols. Restore the logical
    // names before parsing: a temporary path can contain punctuation that
    // would otherwise turn a valid original identifier into invalid PIR.
    let mut ir = structs.restore_ir_names(&checked.to_ir_text().map_err(|e| e.to_string())?);
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
    // Relocation only inserts path components in import/use declarations.
    // Undo their column offsets before restoring the filenames themselves.
    for member in &mut package.members {
        for node in &mut member.graph_mut().nodes {
            for pos in node.pos.iter_mut().flatten() {
                if let Some(path) = package.file_table.id_to_path.get(&pos.fileno) {
                    pos.colno = structs.original_column(Path::new(path), pos.lineno, pos.colno);
                }
            }
        }
    }
    for path in package.file_table.id_to_path.values_mut() {
        let logical = structs.logical_path(Path::new(path));
        if let Ok(relative) = logical.strip_prefix(stdlib_path) {
            let relative = relative
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            *path = format!("xls/dslx/stdlib/{relative}");
        } else if logical != Path::new(path) {
            *path = logical.to_string_lossy().into_owned();
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
    Ok(ImportBatch {
        package,
        roots,
        call_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::fs;
    use tempfile::tempdir;
    use xlsynth_pir::IrValue;
    use xlsynth_pir::ir_eval::eval_fn_in_package_value;

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
    fn nested_struct_calls_reuse_dslx_provenance_in_either_declaration_order()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        for (module, source) in [
            ("b_function", "pub fn Header() -> u8 { u8:0 }"),
            ("c_other", "pub struct Header { tag: u8, channel: u8 }"),
            ("d_constant", "pub const Header = u8:0;"),
            ("z_protocol", "pub struct Header { tag: u8, channel: u8 }"),
            (
                "public_api",
                "import z_protocol; pub type Payload = z_protocol::Header;",
            ),
            (
                "a_factory",
                r#"
import b_function;
import c_other;
import d_constant;
import z_protocol;
pub struct Envelope { header: z_protocol::Header }
pub fn make(x: u8) -> Envelope {
    Envelope { header: z_protocol::Header { tag: x, channel: x + u8:1 } }
}
pub fn read(h: z_protocol::Header) -> u8 { h.channel }
"#,
            ),
        ] {
            fs::write(dir.path().join(format!("{module}.x")), source)?;
        }
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let uses = crate::parse_module("use public_api; use a_factory; use c_other;")?.uses;
        let byte = Type::Bits {
            signed: false,
            width: 8,
        };
        for alias_first in [true, false] {
            let mut resolver = ImportResolver::new(&uses, &options)?;
            if alias_first {
                resolver.resolve_named("public_api", "Payload")?;
            }
            let (_, envelope) = resolver.resolve("a_factory", "make", vec![byte.clone()])?;
            let Type::Named(name) = envelope else {
                panic!("expected Envelope")
            };
            let def = resolver
                .struct_defs()
                .iter()
                .find(|def| def.name == name)
                .ok_or("missing Envelope")?;
            let [field] = def.fields.as_slice() else {
                panic!("expected one Envelope field")
            };
            assert_eq!(field.name, "header");
            let header = field.ty.clone();
            // In the function-first order this call must have a public spelling
            // for the projected field before BX has seen the explicit alias.
            let (_, result) = resolver.resolve("a_factory", "read", vec![header.clone()])?;
            assert_eq!(result, byte);
            assert_eq!(header, resolver.resolve_named("public_api", "Payload")?);
            assert_ne!(header, resolver.resolve_named("c_other", "Header")?);
            let imports = resolver.finish()?;
            assert_eq!(imports.type_witnesses, 1, "alias_first={alias_first}");
            let [batch] = imports.batches.as_slice() else {
                panic!("expected one factory batch")
            };
            let make = batch
                .package
                .get_fn("__a_factory__make")
                .ok_or("missing make")?;
            let read = batch
                .package
                .get_fn("__a_factory__read")
                .ok_or("missing read")?;
            let made =
                eval_fn_in_package_value(&batch.package, make, &[IrValue::make_ubits(8, 41)?]);
            let read = eval_fn_in_package_value(&batch.package, read, &[made.get_element(0)?]);
            assert_eq!(read, IrValue::make_ubits(8, 42)?);
        }
        Ok(())
    }

    #[test]
    fn unrelated_struct_imports_work_with_a_public_declaration_named_self()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(dir.path().join("other.x"), "pub struct self { v: u8 }")?;
        fs::write(
            dir.path().join("wire.x"),
            r#"import other;
struct Hidden { v: u8 }
pub type self = Hidden;
pub type value_t = Hidden;
pub fn make(x: u8) -> Hidden { Hidden { v: x } }
pub fn read(value: Hidden) -> u8 { value.v }
"#,
        )?;
        let paths = [dir.path()];
        let mut native = ImportData::new(None, &paths);
        dslx::parse_and_typecheck(
            "import wire; fn probe(x: u8) -> u8 { let v: wire::value_t = wire::make(x); wire::read(v) }",
            "probe.x",
            "probe",
            &mut native,
        )?;
        let block = crate::parse_module(
            "use wire; block b(clk, x: u8) -> u8 { let v: wire::value_t = wire::make(x); wire::read(v) }",
        )?;
        let options = ImportOptions {
            additional_search_paths: paths.to_vec(),
            ..Default::default()
        };
        crate::check_module_with_options(&block, &options)?;
        Ok(())
    }

    #[test]
    fn public_alias_resolves_against_the_import_snapshot_after_its_private_struct_is_renamed()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        let wire = dir.path().join("wire.x");
        fs::write(
            &wire,
            r#"
struct Hidden { tag: u8 }
pub type Item = Hidden;
pub struct Envelope { item: Hidden }
pub fn read(value: Hidden) -> u8 { value.tag }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let uses = crate::parse_module("use wire;")?.uses;
        let mut resolver = ImportResolver::new(&uses, &options)?;
        // Source discovery now sees Item as a spelling for a different
        // declaration. The original graph still resolves it to Hidden.
        fs::write(
            wire,
            "struct Renamed { payload: u32 } pub type Item = Renamed;",
        )?;

        let Type::Named(name) = resolver.resolve_named("wire", "Envelope")? else {
            panic!("expected Envelope");
        };
        let envelope = resolver
            .struct_defs()
            .iter()
            .find(|def| def.name == name)
            .ok_or("missing Envelope layout")?;
        let [item] = envelope.fields.as_slice() else {
            panic!("expected one Envelope field");
        };
        resolver.resolve("wire", "read", vec![item.ty.clone()])?;
        let imports = resolver.finish()?;
        let [batch] = imports.batches.as_slice() else {
            panic!("expected one wire batch");
        };
        let read = batch.package.get_fn("__wire__read").ok_or("missing read")?;
        let value = IrValue::parse_typed("(bits[8]:42)")?;
        assert_eq!(
            eval_fn_in_package_value(&batch.package, read, &[value]),
            IrValue::make_ubits(8, 42)?
        );
        Ok(())
    }

    #[test]
    fn projecting_a_private_field_does_not_probe_an_unrelated_function_use()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        fs::write(dir.path().join("helper.x"), "pub fn read() -> u8 { u8:42 }")?;
        fs::write(
            dir.path().join("factory.x"),
            r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use helper::read;
struct Hidden { tag: u8 }
pub struct Frame { hidden: Hidden }
pub fn make() -> Frame { zero!<Frame>() }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let bx = crate::parse_module(
            "use factory; block sample(clk) -> u8 { factory::make().hidden.tag }",
        )?;
        let checked = crate::check_module_with_options(&bx, &options)?;
        assert_eq!(checked.imported_calls().len(), 1);
        Ok(())
    }

    #[test]
    fn use_leaf_keeps_its_alias_when_the_exported_type_changes() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        let wire = dir.path().join("wire.x");
        fs::write(
            dir.path().join("factory.x"),
            r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use wire::Item;
pub struct Envelope { header: Item }
pub fn make() -> Envelope { zero!<Envelope>() }
pub fn read(header: Item) -> u8 { header.channel + u8:42 }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let uses = crate::parse_module("use factory;")?.uses;
        for replacement in [
            "pub struct Item { other: u32 }",
            "pub type Item = u32;",
            "pub enum Item : u8 { ZERO = 0 }",
            // The pinned member API cannot enumerate this sum's name. The
            // generated read parameter must recover it from the checked use.
            "pub enum Item { None, Some(u8), }",
        ] {
            fs::write(
                &wire,
                "struct Header { channel: u8 } pub type Item = Header;",
            )?;
            let mut resolver = ImportResolver::new(&uses, &options)?;
            fs::write(&wire, replacement)?;

            resolver.resolve("factory", "make", vec![])?;
            let [_, envelope] = resolver.struct_defs() else {
                panic!("expected Header and Envelope layouts");
            };
            resolver.resolve("factory", "read", vec![envelope.fields[0].ty.clone()])?;
            let imports = resolver.finish()?;
            let package = &imports.batches[0].package;
            let make = package.get_fn("__factory__make").ok_or("missing make")?;
            let read = package.get_fn("__factory__read").ok_or("missing read")?;
            let envelope = eval_fn_in_package_value(package, make, &[]);
            assert_eq!(
                eval_fn_in_package_value(package, read, &[envelope.get_element(0)?]),
                IrValue::make_ubits(8, 42)?,
                "{replacement}"
            );
        }
        Ok(())
    }

    #[test]
    fn register_alias_names_an_opaque_dslx_use_leaf_in_either_order() -> Result<(), Box<dyn Error>>
    {
        let dir = tempdir()?;
        fs::write(
            dir.path().join("wire.x"),
            "pub struct Header { channel: u8 } pub type Item = Header;",
        )?;
        fs::write(
            dir.path().join("public_api.x"),
            "import wire; pub type Payload = wire::Header;",
        )?;
        fs::write(
            dir.path().join("factory.x"),
            r#"#![feature(type_inference_v1)]
#![feature(use_syntax)]
use wire::Item;
pub struct Envelope { header: Item }
pub fn make() -> Envelope { zero!<Envelope>() }
pub fn read(value: Item) -> u8 { value.channel }
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let reg = "reg packet: (Payload[1], u8);";
        let calls = "let frame = factory::make(); let channel = factory::read(frame.header);";
        for (first, second) in [(reg, calls), (calls, reg)] {
            let bx = format!(
                r#"use public_api::Payload;
use factory;
block sample(clk) -> u8 {{
    {first}
    {second}
    packet := ([frame.header], channel);
    packet.1
}}"#
            );
            let checked = crate::check_module_with_options(&crate::parse_module(&bx)?, &options)?;
            assert_eq!(checked.imported_calls().len(), 2);
            assert_eq!(checked.imports.type_witnesses, 1, "{bx}");
        }

        let consumer = format!("block consume(clk) -> u8 {{ {calls} channel }}");
        let storage = format!("block store(clk) -> u8 {{ {reg} packet := packet; packet.1 }}");
        for (first, second) in [(&consumer, &storage), (&storage, &consumer)] {
            let bx = format!("use public_api::Payload; use factory;\n{first}\n{second}");
            let checked = crate::check_module_with_options(&crate::parse_module(&bx)?, &options)?;
            assert_eq!(checked.imported_calls().len(), 2);
            assert_eq!(checked.imports.type_witnesses, 1, "{bx}");
        }

        for statements in [
            "let public_api = x; reg packet: public_api::Missing;",
            "reg packet: public_api::Missing; inst public_api: child;",
        ] {
            let bx = format!(
                r#"use public_api;
block child(clk) -> u8 {{ u8:0 }}
block sample(clk, x: u8) -> u8 {{ {statements} packet := x; x }}"#
            );
            let shadow = crate::parse_module(&bx)?;
            assert_eq!(
                crate::check_module_with_options(&shadow, &options).unwrap_err(),
                "block 'sample': 'public_api' names a local value here"
            );
        }

        let invalid = crate::parse_module(
            r#"use public_api;
block sample(clk) -> u8 {
    reg saved: public_api::Payload;
    missing := saved;
    reg bad: public_api::Missing;
    saved := saved;
    bad := bad;
    saved.channel
}"#,
        )?;
        assert_eq!(
            crate::check_module_with_options(&invalid, &options).unwrap_err(),
            "block 'sample': 'missing' is not a declared register"
        );
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
