// SPDX-License-Identifier: Apache-2.0

//! Resolve concrete calls to DSLX std functions and compile their IR bodies.

use std::collections::HashMap;
use std::path::Path;

use xlsynth::dslx::{self, ImportData, MatchableModuleMember, TypecheckedModule};
use xlsynth_pir::ir::{NodePayload, Package};
use xlsynth_pir::ir_parser;

use crate::ast::Type;
use crate::parser::parse_value_type;

/// An imported std function specialized for one set of BX argument types.
#[derive(Clone, Debug)]
pub struct StdCall {
    /// Unqualified DSLX function name in the `std` module.
    pub function: String,
    /// Concrete argument types in call order.
    pub args: Vec<Type>,
    /// Concrete result type inferred by DSLX.
    pub result: Type,
}

/// One module's specializations, in first-use order, with a shared DSLX import
/// cache allocated on the first std call.
#[derive(Default)]
pub(crate) struct StdResolver {
    imports: Option<ImportData>,
    ids: HashMap<(String, Vec<Type>), usize>,
    calls: Vec<StdCall>,
}

impl StdResolver {
    /// Ask the DSLX typechecker to specialize the function from concrete data
    /// arguments. The checked call supplies its inferred and default bindings.
    pub(crate) fn resolve(
        &mut self,
        function: &str,
        mut args: Vec<Type>,
    ) -> Result<(usize, Type), String> {
        for arg in &mut args {
            canonicalize_bool(arg);
        }
        let key = (function.to_owned(), args);
        if let Some(&id) = self.ids.get(&key) {
            return Ok((id, self.calls[id].result.clone()));
        }
        let id = self.calls.len();
        let probe = probe_definition(0, function, &key.1)?;
        let source = format!("import std;\n{probe}");
        // ImportData retains successfully parsed modules by name.
        let name = format!("bx_std_signature_{id}");
        let checked = dslx::parse_and_typecheck(
            &source,
            &format!("{name}.x"),
            &name,
            self.imports.get_or_insert_with(ImportData::default),
        )
        .map_err(|e| format!("std::{function}: {e}"))?;
        let result = probe_result_type(&checked).map_err(|e| format!("std::{function}: {e}"))?;
        self.calls.push(StdCall {
            function: key.0.clone(),
            args: key.1.clone(),
            result: result.clone(),
        });
        self.ids.insert(key, id);
        Ok((id, result))
    }

    pub(crate) fn into_parts(self) -> (Vec<StdCall>, Option<ImportData>) {
        (self.calls, self.imports)
    }
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
            "unsupported nominal type '{name}' in std function signature"
        )),
    }
}

/// A unit-returning DSLX probe checks the call without needing to know its
/// result in advance. Its unoptimized IR contains the specialized invoke.
fn probe_definition(index: usize, function: &str, args: &[Type]) -> Result<String, String> {
    let mut chars = function.bytes();
    if !chars
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        || !chars.all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(format!("invalid DSLX std function name {function:?}"));
    }
    let mut params = Vec::with_capacity(args.len());
    let mut arguments = Vec::with_capacity(args.len());
    for (i, ty) in args.iter().enumerate() {
        let name = format!("arg_{i}");
        let ty = dslx_type(ty).map_err(|e| format!("std::{function}: {e}"))?;
        params.push(format!("{name}: {ty}"));
        arguments.push(name);
    }
    Ok(format!(
        "fn probe_{index}({}) {{ let _ = std::{function}({}); }}\n",
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

/// Compile all concrete calls together and recover each specialization's exact
/// symbol from its probe. The linker retains the functions reachable from them.
pub(crate) fn compile_calls(
    calls: &[StdCall],
    imports: Option<ImportData>,
) -> Result<(Package, Vec<String>), String> {
    const MODULE: &str = "bx_std_imports";
    if calls.is_empty() {
        return Ok((crate::empty_package(MODULE), Vec::new()));
    }
    let mut source = String::from("import std;\n");
    for (index, call) in calls.iter().enumerate() {
        source.push_str(&probe_definition(index, &call.function, &call.args)?);
    }
    let checked = dslx::parse_and_typecheck(
        &source,
        "bx_std_imports.x",
        MODULE,
        &mut imports.unwrap_or_default(),
    )
    .map_err(|e| format!("checking DSLX std calls: {e}"))?;
    let ir = checked
        .to_ir_text()
        .map_err(|e| format!("compiling DSLX std calls: {e}"))?;
    let mut package = ir_parser::Parser::new(&ir)
        .parse_xls_package()
        .map_err(|e| format!("reading DSLX std functions as PIR: {e}"))?;
    for path in package.file_table.id_to_path.values_mut() {
        if let Ok(relative) = Path::new(path).strip_prefix(xlsynth::default_dslx_stdlib_path()) {
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
    let mut symbols = Vec::with_capacity(calls.len());
    for (index, call) in calls.iter().enumerate() {
        let name = xlsynth::mangle_dslx_name(MODULE, &format!("probe_{index}"))
            .map_err(|e| e.to_string())?;
        let probe = functions
            .get(name.as_str())
            .ok_or_else(|| format!("missing compiled probe for std::{}", call.function))?;
        let mut invokes = probe.nodes.iter().filter_map(|n| match &n.payload {
            NodePayload::Invoke { to_apply, .. } => Some(to_apply),
            _ => None,
        });
        let symbol = invokes
            .next()
            .ok_or_else(|| format!("missing invoke for std::{}", call.function))?;
        if invokes.next().is_some() {
            return Err(format!(
                "ambiguous compiled probe for std::{}",
                call.function
            ));
        }
        symbols.push(symbol.clone());
    }
    Ok((package, symbols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_std_names_from_a_programmatic_ast_are_rejected_before_loading_dslx() {
        let mut resolver = StdResolver::default();
        for name in ["", "foo::bar", "x\0y", "f(); } fn injected() {"] {
            assert!(resolver.resolve(name, vec![]).is_err());
        }
        let nested = Type::Tuple(vec![Type::Array {
            element: Box::new(Type::Named("Other".into())),
            size: 2,
        }]);
        assert!(resolver.resolve("enumerate", vec![nested]).is_err());
        assert!(resolver.imports.is_none());
    }
}
