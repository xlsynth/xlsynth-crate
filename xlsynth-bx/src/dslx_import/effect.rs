// SPDX-License-Identifier: Apache-2.0

//! Explain an implicit-token rejection using the typechecker's call graph.

use std::collections::{BTreeSet, HashMap, hash_map::Entry};

use xlsynth::dslx::{Function, FunctionCallGraph, TypeInfo, TypecheckedModule};

use crate::effect_diagnostic::EffectDiagnostic;
use crate::import_structs::{
    EffectFunctionSource, EffectSources, ImportedStructs, is_token_builtin,
};

/// A verified call edge together with the callee's own module type context.
struct Callee {
    function: Function,
    type_info: TypeInfo,
    offset: Option<usize>,
    unconditional: bool,
}

/// Native effectful edges and the earliest source site with uncertain
/// ownership.
struct CheckedCalls {
    builtins: BTreeSet<String>,
    children: Vec<Callee>,
    opaque_offset: Option<usize>,
}

/// Pair the caller's typed edges with the earliest written call to each
/// declaration. A pure checked edge can rule out even an opaque call.
fn checked_calls(
    sources: &EffectSources<'_>,
    source: Option<&EffectFunctionSource<'_>>,
    graph: &FunctionCallGraph,
    function: &Function,
    type_info: &TypeInfo,
    native_name: &str,
) -> CheckedCalls {
    let mut builtins = BTreeSet::new();
    let mut children = Vec::new();
    let mut first_calls = HashMap::new();
    let mut opaque_calls = HashMap::new();
    for site in source.into_iter().flat_map(|s| &s.calls) {
        let key = (site.subject, site.name);
        first_calls.entry(key).or_insert(site);
        if site.opaque {
            opaque_calls.entry(key).or_insert(site.offset);
        }
    }
    for callee in (0..graph.callee_count(function)).filter_map(|i| graph.get_callee(function, i)) {
        let owner = callee.get_body().get_owner_module();
        let owner_name = owner.get_name();
        let name = callee.get_identifier();
        if owner_name == "<builtin_stubs>" {
            if is_token_builtin(&name) {
                builtins.insert(name);
            }
            continue;
        }
        let info = if owner_name == native_name {
            type_info.clone()
        } else if let Some(info) = type_info.get_imported_type_info(&owner) {
            info
        } else {
            continue;
        };
        let site = sources
            .subject(&owner)
            .and_then(|subject| first_calls.get(&(subject, name.as_str())).copied())
            .filter(|_| sources.is_named_definition(&callee));
        let Ok(requires_token) = info.requires_implicit_token(&callee) else {
            continue;
        };
        if requires_token {
            children.push(Callee {
                function: callee,
                type_info: info,
                offset: site.map(|s| s.offset),
                unconditional: site.is_some_and(|s| s.unconditional),
            });
        } else if let Some(site) = site {
            opaque_calls.remove(&(site.subject, site.name));
        }
    }
    let opaque_offset = source
        .and_then(|s| s.opaque_offset)
        .into_iter()
        .chain(opaque_calls.into_values())
        .min();
    children.sort_by_key(|callee| callee.offset);
    CheckedCalls {
        builtins,
        children,
        opaque_offset,
    }
}

/// Start at the exact function called by the signature probe. Converting a
/// rejected function to IR can abort in libxls even after it typechecks.
pub(super) fn diagnose(
    checked: &TypecheckedModule,
    probe: &Function,
    structs: &ImportedStructs,
    subject: &str,
    name: &str,
) -> Option<String> {
    let sources = structs.effect_sources();
    let probe_info = checked.get_type_info();
    let graph = probe_info
        .build_function_call_graph(&checked.get_module())
        .ok()?;
    let (function, type_info) = (0..graph.callee_count(probe)).find_map(|i| {
        let callee = graph.get_callee(probe, i)?;
        let owner = callee.get_body().get_owner_module();
        if callee.get_identifier() != name || sources.subject(&owner)? != subject {
            return None;
        }
        Some((callee, probe_info.get_imported_type_info(&owner)?))
    })?;
    let mut report = EffectDiagnostic {
        call_path: vec![format!("{subject}::{name}")],
        operation: None,
        locations: Vec::new(),
        const_assert_replacement: None,
    };
    find_effect(&sources, function, type_info, &mut report);
    Some(report.to_string())
}

/// Follow token-requiring edges in source evaluation order. The native graph
/// identifies definitions but does not expose the positions of invocations.
fn find_effect<'a>(
    sources: &EffectSources<'a>,
    mut function: Function,
    mut type_info: TypeInfo,
    report: &mut EffectDiagnostic<'a>,
) {
    let mut graphs: HashMap<String, FunctionCallGraph> = HashMap::new();
    let mut visited: Vec<Function> = Vec::new();
    let mut unconditional_path = true;
    while !visited
        .iter()
        .any(|seen| seen.is_same_definition(&function))
    {
        let module = function.get_body().get_owner_module();
        let native_name = module.get_name();
        let graph = match graphs.entry(native_name.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let Ok(graph) = type_info.build_function_call_graph(&module) else {
                    break;
                };
                entry.insert(graph)
            }
        };
        let source = sources.function(&function);
        if !visited.is_empty() {
            // Compiler-created functions such as a map lambda's `call` have
            // no named declaration in the enclosing user's module.
            if let Some(view) = &source {
                report
                    .call_path
                    .push(format!("{}::{}", view.subject, function.get_identifier()));
            } else if let Some(subject) = sources.subject(&module)
                && (0..graph.function_count()).any(|i| {
                    graph
                        .get_function(i)
                        .is_some_and(|f| f.is_same_definition(&function))
                })
            {
                report
                    .call_path
                    .push(format!("{subject}::{}", function.get_identifier()));
            }
        }
        let CheckedCalls {
            mut builtins,
            children,
            opaque_offset,
        } = checked_calls(
            sources,
            source.as_ref(),
            graph,
            &function,
            &type_info,
            &native_name,
        );
        let direct: Vec<_> = source
            .into_iter()
            .flat_map(|s| s.effects)
            .filter(|site| {
                // DSLX parses these as FormatMacro nodes; they have no
                // builtin invocation edge in the checked graph.
                matches!(
                    site.location.operation,
                    "trace_fmt!" | "vtrace_fmt!" | "assert_fmt!"
                ) || builtins.contains(site.location.operation)
            })
            .collect();
        let unlocated_builtins = builtins
            .iter()
            .filter(|name| {
                !direct
                    .iter()
                    .any(|site| site.location.operation == name.as_str())
            })
            .count();
        let choices = children.len() + direct.len() + unlocated_builtins;
        if choices > 1
            && (unlocated_builtins > 0 || children.iter().any(|child| child.offset.is_none()))
        {
            // The native graph verifies these effects but doesn't order its
            // edges. Keep the verified breadcrumb when their source is opaque.
            break;
        }
        let next_offset = children
            .first()
            .and_then(|callee| callee.offset)
            .unwrap_or(usize::MAX);
        let mut direct = direct.into_iter().peekable();
        let earliest = direct
            .peek()
            .map_or(next_offset, |site| site.location.range.end.min(next_offset));
        if opaque_offset.is_some_and(|offset| offset <= earliest) {
            // Some loop/lambda effects merge into the caller's native edge.
            // An earlier opaque occurrence can share the later call's identity.
            break;
        }
        if let Some(first) = direct.next_if(|site| site.location.range.end < next_offset) {
            let is_assertion = first.location.operation.starts_with("assert");
            report.operation = Some(first.location.operation.to_owned());
            if unconditional_path {
                report.const_assert_replacement = first.const_assert_replacement;
            }
            report.locations.push(first.location);
            if is_assertion {
                report.locations.extend(
                    direct
                        .filter(|site| site.location.operation.starts_with("assert"))
                        .map(|site| site.location),
                );
            }
            return;
        }
        if direct
            .peek()
            .is_some_and(|site| site.location.range.end == next_offset)
            || (children.len() >= 2 && children[0].offset == children[1].offset)
        {
            break;
        }
        let Some(next) = children.into_iter().next() else {
            if builtins.len() == 1 {
                report.operation = builtins.pop_first();
            }
            break;
        };
        visited.push(function);
        unconditional_path &= next.unconditional;
        function = next.function;
        type_info = next.type_info;
    }
}
