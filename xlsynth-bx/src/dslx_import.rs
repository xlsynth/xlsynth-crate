// SPDX-License-Identifier: Apache-2.0

//! Resolve DSLX imports and preserve the concrete IR functions selected by BX.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use xlsynth::dslx::{self, ImportData, MatchableModuleMember, TypecheckedModule};
use xlsynth_pir::ir::{NodePayload, Package};
use xlsynth_pir::ir_parser;

use crate::ImportOptions;
use crate::ast::{Import, Type};
use crate::parser::{is_identifier, is_identifier_word, parse_value_type};

/// An imported DSLX function specialized for one set of BX argument types.
#[derive(Clone, Debug)]
pub struct ImportedCall {
    /// Full dotted import subject, independent of the local BX alias.
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

/// Functions compiled together from one canonical DSLX module. Each root
/// symbol corresponds to a call ID in the same position.
#[derive(Clone, Debug)]
pub(crate) struct ImportBatch {
    pub(crate) package: Package,
    pub(crate) roots: Vec<String>,
    pub(crate) call_ids: Vec<usize>,
}

/// Per-check import graph and memoized concrete function signatures.
pub(crate) struct ImportResolver {
    subjects: BTreeSet<String>,
    stdlib_path: PathBuf,
    imports: Option<ImportData>,
    ids: HashMap<(String, String, Vec<Type>), usize>,
    calls: Vec<ImportedCall>,
}

impl ImportResolver {
    /// Typecheck every import up front, including unused modules, so all calls
    /// in this BX module use a single snapshot of the DSLX sources.
    pub(crate) fn new(
        declarations: &[Import],
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
        let mut subjects = BTreeSet::new();
        for declaration in declarations {
            if !declaration.subject.split('.').all(is_identifier_word) {
                return Err(format!(
                    "invalid DSLX import subject {:?}",
                    declaration.subject
                ));
            }
            if !is_identifier(declaration.binding()) {
                return Err(format!(
                    "invalid BX import binding {:?}",
                    declaration.binding()
                ));
            }
            subjects.insert(declaration.subject.clone());
        }
        let mut imports = None;
        if !subjects.is_empty() {
            let mut data = ImportData::new(Some(stdlib_path), &options.additional_search_paths);
            let mut source = String::new();
            for (index, subject) in subjects.iter().enumerate() {
                source.push_str(&format!("import {subject} as bx_module_{index};\n"));
            }
            check_probe(&source, "0bx_declarations", &mut data)
                .map_err(|e| format!("checking DSLX imports: {e}"))?;
            imports = Some(data);
        }
        Ok(Self {
            subjects,
            stdlib_path: stdlib_path.to_owned(),
            imports,
            ids: HashMap::new(),
            calls: Vec::new(),
        })
    }

    /// Ask the DSLX typechecker to specialize the function from concrete data
    /// arguments. The checked call supplies its inferred and default bindings.
    pub(crate) fn resolve(
        &mut self,
        module: &str,
        function: &str,
        mut args: Vec<Type>,
    ) -> Result<(usize, Type), String> {
        for arg in &mut args {
            canonicalize_bool(arg);
        }
        if !self.subjects.contains(module) {
            return Err(format!("undeclared DSLX import {module:?}"));
        }
        if !is_identifier_word(function) {
            return Err(format!(
                "invalid function name {function:?} in DSLX import {module}"
            ));
        }
        let key = (module.to_owned(), function.to_owned(), args);
        if let Some(&id) = self.ids.get(&key) {
            return Ok((id, self.calls[id].result.clone()));
        }
        let id = self.calls.len();
        let probe = probe_definition(id, function, &key.2)
            .map_err(|e| format!("{module}::{function}: {e}"))?;
        let source = format!("import {module} as bx_module;\n{probe}");
        let data = self
            .imports
            .as_mut()
            .expect("a declared subject has an import graph");
        let result = check_probe(&source, &format!("0bx_signature_{id}"), data)
            .and_then(|checked| probe_result_type(&checked))
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
    pub(crate) fn finish(mut self) -> Result<ResolvedImports, String> {
        let mut batches = Vec::new();
        if let Some(data) = self.imports.as_mut() {
            let mut groups = BTreeMap::<&str, Vec<usize>>::new();
            for (id, call) in self.calls.iter().enumerate() {
                groups.entry(&call.module).or_default().push(id);
            }
            for (index, (subject, ids)) in groups.into_iter().enumerate() {
                let batch =
                    compile_batch(subject, index, &self.calls, ids, data, &self.stdlib_path)
                        .map_err(|e| format!("compiling DSLX import '{subject}': {e}"))?;
                batches.push(batch);
            }
        }
        Ok(ResolvedImports {
            calls: self.calls,
            batches,
        })
    }
}

/// A probe module starts with a digit, which DSLX source cannot import and
/// which real module names cannot acquire through dot-to-underscore mangling.
fn check_probe(
    source: &str,
    name: &str,
    imports: &mut ImportData,
) -> Result<TypecheckedModule, String> {
    dslx::parse_and_typecheck(source, &format!("{name}.x"), name, imports)
        .map_err(|e| e.to_string())
}

/// DSLX uses the same concrete type and specialization for bool and u1.
fn canonicalize_bool(ty: &mut Type) {
    match ty {
        Type::Bool => {
            *ty = Type::Bits {
                signed: false,
                width: 1,
            }
        }
        Type::Tuple(items) => items.iter_mut().for_each(canonicalize_bool),
        Type::Array { element, .. } => canonicalize_bool(element),
        Type::Bits { .. } | Type::Named(_) => {
            // These types have no bool components.
        }
    }
}

/// Serialize BX value types using DSLX's arbitrary-width bit type syntax.
/// Reject nominal types even inside a tuple or array.
fn dslx_type(ty: &Type) -> Result<String, String> {
    match ty {
        Type::Bits { signed, width } => {
            Ok(format!("{}N[{width}]", if *signed { "s" } else { "u" }))
        }
        Type::Bool => Ok("bool".into()),
        Type::Array { element, size } => Ok(format!("{}[{size}]", dslx_type(element)?)),
        Type::Tuple(members) => {
            let types = members
                .iter()
                .map(dslx_type)
                .collect::<Result<Vec<_>, _>>()?;
            let trailing = if types.len() == 1 { "," } else { "" };
            Ok(format!("({}{trailing})", types.join(", ")))
        }
        Type::Named(name) => Err(format!(
            "unsupported nominal type '{name}' in imported function signature"
        )),
    }
}

/// A unit-returning DSLX probe checks the call without needing to know its
/// result in advance. Its unoptimized IR contains the specialized invoke.
fn probe_definition(index: usize, function: &str, args: &[Type]) -> Result<String, String> {
    let mut params = Vec::with_capacity(args.len());
    let mut arguments = Vec::with_capacity(args.len());
    for (i, ty) in args.iter().enumerate() {
        let name = format!("arg_{i}");
        let ty = dslx_type(ty)?;
        params.push(format!("{name}: {ty}"));
        arguments.push(name);
    }
    Ok(format!(
        "fn probe_{index}({}) {{ let _ = bx_module::{function}({}); }}\n",
        params.join(", "),
        arguments.join(", "),
    ))
}

/// Recover the specialized return annotation from this specific invocation.
/// Imported modules may have other calls to the same function in their tests.
fn probe_result_type(checked: &TypecheckedModule) -> Result<Type, String> {
    let module = checked.get_module();
    let probe = (0..module.get_member_count())
        .find_map(|i| match module.get_member(i).to_matchable() {
            Some(MatchableModuleMember::Function(f)) => Some(f),
            _ => None,
        })
        .ok_or("could not find the generated DSLX probe")?;
    let root = checked.get_type_info();
    let graph = root
        .build_function_call_graph(&module)
        .map_err(|e| e.to_string())?;
    let callee = graph
        .get_callee(&probe, 0)
        .ok_or("call did not resolve to a DSLX function")?;
    if !callee.is_public() {
        return Err("function is private".into());
    }
    if root
        .requires_implicit_token(&probe)
        .map_err(|e| e.to_string())?
    {
        return Err("function requires an implicit token and cannot be lifted into BX".into());
    }
    let Some(annotation) = callee.get_return_type() else {
        // Omitting a DSLX return annotation declares a unit result.
        return Ok(Type::Tuple(Vec::new()));
    };
    let context = if callee.is_parametric() {
        let entries = root
            .get_all_invocation_callee_data(&callee)
            .ok_or("missing DSLX invocation data")?;
        (0..entries.len()).find_map(|index| {
            let entry = entries.get(index)?;
            let caller = root
                .get_root_invocation_data(&entry.invocation()?)?
                .caller()?;
            if caller.is_same_definition(&probe) {
                entry.derived_type_info()
            } else {
                None
            }
        })
    } else {
        checked.get_type_info_for_module(&callee.get_body().get_owner_module())
    }
    .ok_or("missing DSLX type information for the call")?;
    let resolved = context
        .get_type_for_type_annotation(&annotation)
        .ok_or("could not resolve the DSLX return type")?;
    let text = resolved.to_string().map_err(|e| e.to_string())?;
    parse_value_type(&text).map_err(|e| format!("unsupported result type '{text}': {e}"))
}

/// Recover each specialization's exact IR symbol from its generated probe.
fn compile_batch(
    subject: &str,
    index: usize,
    calls: &[ImportedCall],
    call_ids: Vec<usize>,
    imports: &mut ImportData,
    stdlib_path: &Path,
) -> Result<ImportBatch, String> {
    let mut source = format!("import {subject} as bx_module;\n");
    for &id in &call_ids {
        let call = &calls[id];
        source.push_str(&probe_definition(id, &call.function, &call.args)?);
    }
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

    #[test]
    fn invalid_import_names_from_a_programmatic_ast_are_rejected_before_loading_dslx() {
        for name in [
            "",
            ".m",
            "m..n",
            "foo::bar",
            "x\0y",
            "f(); } fn injected() {",
        ] {
            let import = Import {
                subject: name.to_owned(),
                alias: None,
            };
            assert!(ImportResolver::new(&[import], &ImportOptions::default()).is_err());
        }
        let import = Import {
            subject: "std".into(),
            alias: Some("bad.alias".into()),
        };
        assert!(ImportResolver::new(&[import], &ImportOptions::default()).is_err());
    }

    #[test]
    fn transitive_function_with_a_probe_name_survives_batch_compilation()
    -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        // A user dependency can have the name of an ordinary generated module
        // and define a function with the name of the generated probe.
        fs::write(
            dir.path().join("bx_batch_0.x"),
            "pub fn probe_0(x: u8) -> u8 { x + u8:1 }",
        )?;
        fs::write(
            dir.path().join("wrapper.x"),
            "import bx_batch_0; pub fn apply(x: u8) -> u8 { bx_batch_0::probe_0(x) }",
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        let import = Import {
            subject: "wrapper".into(),
            alias: None,
        };
        let mut resolver = ImportResolver::new(&[import], &options)?;
        resolver.resolve(
            "wrapper",
            "apply",
            vec![Type::Bits {
                signed: false,
                width: 8,
            }],
        )?;
        let imports = resolver.finish()?;
        let batch = &imports.batches[0];
        assert_eq!(batch.roots, ["__wrapper__apply"]);
        assert!(batch.package.get_fn("__bx_batch_0__probe_0").is_some());
        Ok(())
    }
}
