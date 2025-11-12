// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use log::{debug, info};
use xlsynth::dslx::{
    Function, FunctionCallGraph, FunctionSpecializationRequest, ImportData, InvocationRewriteRule,
    MatchableModuleMember, Module, ModuleMember, ModuleMemberKind, ParametricEnv, TypeInfo,
    TypecheckedModule, parse_and_typecheck,
};
use xlsynth::{
    DslxCallingConvention, XlsynthError, dslx_path_to_module_name, mangle_dslx_name_with_env,
};

use crate::dslx_utils::{parse_top_function_spec, resolve_parametric_env};

#[derive(Debug)]
pub struct SpecializedModule {
    pub source: String,
    pub top_name: String,
}

pub fn specialize_dslx_module(
    dslx_source: &str,
    source_path: &Path,
    top_function: &str,
    stdlib_path: Option<&Path>,
    additional_search_paths: &[PathBuf],
) -> Result<SpecializedModule, XlsynthError> {
    let parsed_top = parse_top_function_spec(top_function)?;
    let top_name = parsed_top.name.clone();

    let module_name = dslx_path_to_module_name(source_path)?;
    let additional_refs: Vec<&Path> = additional_search_paths
        .iter()
        .map(|p| p.as_path())
        .collect();
    let mut import_data = ImportData::new(stdlib_path, &additional_refs);
    info!("Parsing and typechecking module '{}'", module_name);
    let tm = parse_and_typecheck(
        dslx_source,
        source_path.to_str().unwrap(),
        module_name,
        &mut import_data,
    )?;
    let module = tm.get_module();
    let functions = collect_functions(&module);
    let top_fn = functions.get(&top_name).ok_or_else(|| {
        XlsynthError(format!(
            "Top function '{}' not found in module '{}'.",
            top_name, module_name
        ))
    })?;
    let top_env = resolve_parametric_env(top_fn, &parsed_top, /* require_explicit= */ true)?;

    let (tm, final_top) = specialize_typechecked_module(
        tm,
        &mut import_data,
        module_name,
        &top_name,
        top_env.as_ref(),
    )?;
    let module = tm.get_module();
    Ok(SpecializedModule {
        source: module.to_string(),
        top_name: final_top,
    })
}

fn specialize_typechecked_module(
    tm: TypecheckedModule,
    import_data: &mut ImportData,
    module_name: &str,
    top_function: &str,
    top_env: Option<&ParametricEnv>,
) -> Result<(TypecheckedModule, String), XlsynthError> {
    let tm = if top_env.is_some() {
        // When a parameter environment is provided we cannot rely on the initial
        // call graph reachability analysis: parametric callees may appear
        // unreachable until the requested specializations are installed. Skip the
        // early pruning step so we have the full module available for cloning.
        info!("Skipping initial prune because a top parameter environment was provided");
        tm
    } else {
        info!("Initial pruning of unreachable functions");
        prune_unreachable_functions(
            tm,
            import_data,
            &(module_name.to_string() + ".initial_prune"),
            top_function,
            &HashSet::new(),
        )?
    };

    info!(
        "Building call graph for module '{}', top '{}'.",
        module_name, top_function
    );

    let (tm_with_specializations, spec_name_map) =
        accumulate_specializations(tm, import_data, module_name, top_function, top_env)?;

    if spec_name_map.is_empty() {
        return Ok((tm_with_specializations, top_function.to_string()));
    }

    info!("Rewriting specialized calls");
    let tm_rewritten = rewrite_specialized_calls(
        tm_with_specializations,
        import_data,
        module_name,
        &spec_name_map,
    )?;

    let specialized_to_keep: HashSet<String> = spec_name_map.values().cloned().collect();

    let root_top = if let Some(env) = top_env {
        spec_name_map
            .iter()
            .find_map(|((name, candidate_env), spec_name)| {
                if name == top_function && candidate_env == env {
                    Some(spec_name.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| top_function.to_string())
    } else {
        top_function.to_string()
    };

    info!("Pruning unreachable functions");
    let tm_final = prune_unreachable_functions(
        tm_rewritten,
        import_data,
        module_name,
        &root_top,
        &specialized_to_keep,
    )?;

    Ok((tm_final, root_top))
}

fn accumulate_specializations(
    mut tm_current: TypecheckedModule,
    import_data: &mut ImportData,
    module_name: &str,
    top_function: &str,
    top_env: Option<&ParametricEnv>,
) -> Result<(TypecheckedModule, HashMap<(String, ParametricEnv), String>), XlsynthError> {
    let mut spec_name_map: HashMap<(String, ParametricEnv), String> = HashMap::new();
    let mut iteration: u32 = 0;
    let top_function_owned = top_function.to_string();
    let mut current_top_name = top_function_owned.clone();
    let mut top_env_owned = top_env.cloned();
    loop {
        iteration += 1;
        debug!("Iteration {} starting", iteration);

        let module = tm_current.get_module();
        let functions = collect_functions(&module);
        if !functions.contains_key(&current_top_name) {
            return Err(XlsynthError(format!(
                "Top function '{}' not found in module '{}'.",
                current_top_name, module_name
            )));
        }

        let type_info = tm_current.get_type_info();
        let call_graph = type_info.build_function_call_graph()?;
        let reachable = reachable_functions(&current_top_name, &call_graph)?;
        debug!(
            "Iteration {} reachable functions: {:?}",
            iteration, reachable
        );

        let mut specializations = collect_specializations(&type_info, &reachable, &functions);
        specializations.retain(|function_name, envs| {
            envs.retain(|env| !spec_name_map.contains_key(&(function_name.clone(), env.clone())));
            !envs.is_empty()
        });
        if let Some(env) = top_env {
            let already_done = spec_name_map
                .iter()
                .any(|((name, existing_env), _)| name == top_function && existing_env == env);
            if !already_done {
                specializations
                    .entry(top_function.to_string())
                    .or_insert_with(BTreeSet::new)
                    .insert(env.clone());
            }
        }

        if specializations.is_empty() {
            if spec_name_map.is_empty() {
                info!("No additional specializations required.");
            }
            break;
        }

        debug!("Building specialization plan");
        let spec_plan = build_specialization_plan(module_name, &specializations)?;
        let install_subject = format!("{}.specialize.{}", module_name, iteration);
        debug!("Applying specializations");
        let (new_tm, new_names) =
            apply_specializations(&tm_current, import_data, &spec_plan, &install_subject)?;
        debug!(
            "Module after specialization iteration {}:\n{}",
            iteration,
            new_tm.get_module().to_string()
        );
        spec_name_map.extend(new_names.into_iter());
        if let Some(env) = top_env_owned.as_ref() {
            if let Some(spec_name) = spec_name_map.get(&(top_function_owned.clone(), env.clone())) {
                current_top_name = spec_name.clone();
                // We only need to update once; avoid repeated lookups by clearing the env
                // so we do not spend additional work cloning the key.
                top_env_owned = None;
            }
        }
        tm_current = new_tm;
    }

    Ok((tm_current, spec_name_map))
}

fn rewrite_specialized_calls(
    tm: TypecheckedModule,
    import_data: &mut ImportData,
    module_name: &str,
    spec_name_map: &HashMap<(String, ParametricEnv), String>,
) -> Result<TypecheckedModule, XlsynthError> {
    let tm_current = tm;

    let module = tm_current.get_module();
    let functions = collect_functions(&module);
    let rewrite_entries = build_rewrite_plan(&functions, spec_name_map)?;
    let type_info = tm_current.get_type_info();

    let mut caller_seen: HashSet<String> = HashSet::new();
    let mut caller_accum: Vec<Function> = Vec::new();
    let mut rules_storage: Vec<(Function, Function, ParametricEnv)> = Vec::new();

    for spec_name in spec_name_map.values() {
        if let Some(spec_fn) = functions.get(spec_name) {
            if caller_seen.insert(spec_name.clone()) {
                caller_accum.push(spec_fn.clone());
            }
        }
    }

    for (entry_idx, entry) in rewrite_entries.iter().enumerate() {
        let Some(from_callee) = functions.get(&entry.from_callee_name) else {
            continue;
        };
        let Some(to_callee) = functions.get(&entry.to_callee_name) else {
            continue;
        };

        let caller_functions = collect_callers_matching_env(&type_info, from_callee, &entry.env);
        let mut caller_entries = caller_functions;

        let mut additional_callers: Vec<(String, Function, Option<ParametricEnv>)> = Vec::new();
        for (caller_name, _, caller_env) in caller_entries.iter() {
            if let Some(env) = caller_env {
                if let Some(spec_name) = spec_name_map.get(&(caller_name.clone(), env.clone())) {
                    if let Some(spec_fn) = functions.get(spec_name) {
                        additional_callers.push((spec_name.clone(), spec_fn.clone(), None));
                    }
                }
            }
        }
        caller_entries.extend(additional_callers.into_iter());

        if caller_entries.is_empty() {
            debug!(
                "Rewrite {}: {} -> {} skipped (no matching callers)",
                entry_idx, entry.from_callee_name, entry.to_callee_name
            );
            continue;
        }

        caller_entries.sort_by(|a, b| a.0.cmp(&b.0));
        let caller_names: Vec<String> = caller_entries
            .iter()
            .map(|(name, _, _)| name.clone())
            .collect();

        debug!(
            "Rewrite {}: {} -> {} callers {:?}",
            entry_idx, entry.from_callee_name, entry.to_callee_name, caller_names
        );

        for (name, function, _) in caller_entries {
            if caller_seen.insert(name.clone()) {
                caller_accum.push(function);
            }
        }

        rules_storage.push((from_callee.clone(), to_callee.clone(), entry.env.clone()));
    }

    if rules_storage.is_empty() {
        return Ok(tm_current);
    }

    let caller_refs: Vec<&Function> = caller_accum.iter().collect();
    let mut rules: Vec<InvocationRewriteRule> = Vec::with_capacity(rules_storage.len());
    for (from_callee, to_callee, env) in rules_storage.iter() {
        rules.push(InvocationRewriteRule {
            from_callee,
            to_callee,
            match_callee_env: Some(env.clone()),
            to_callee_env: None,
        });
    }

    debug!(
        "Rewrite batch: {} callers, {} rules",
        caller_refs.len(),
        rules.len()
    );

    let install_subject = format!("{}.rewrite", module_name);
    tm_current.replace_invocations_in_module(import_data, &caller_refs, &rules, &install_subject)
}

fn collect_functions(module: &Module) -> HashMap<String, Function> {
    let mut map = HashMap::new();
    for idx in 0..module.get_member_count() {
        let member = module.get_member(idx);
        if let Some(matchable) = member.to_matchable() {
            if let MatchableModuleMember::Function(function) = matchable {
                map.insert(function.get_identifier(), function);
            }
        }
    }
    map
}

fn collect_callers_matching_env(
    type_info: &TypeInfo,
    callee: &Function,
    target_env: &ParametricEnv,
) -> Vec<(String, Function, Option<ParametricEnv>)> {
    let mut callers = Vec::new();
    let mut seen = HashSet::new();
    if let Some(data_array) = type_info.get_all_invocation_callee_data(callee) {
        for idx in 0..data_array.len() {
            if let Some(data) = data_array.get(idx) {
                let Some(callee_env) = data.callee_bindings() else {
                    continue;
                };
                if callee_env != *target_env {
                    debug!(
                        "Skipping caller due to env mismatch: callee env {} target env {}",
                        callee_env.to_string(),
                        target_env.to_string()
                    );
                    continue;
                }
                let Some(invocation) = data.invocation() else {
                    continue;
                };
                let Some(invocation_data) = type_info.get_root_invocation_data(&invocation) else {
                    continue;
                };
                let Some(caller_fn) = invocation_data.caller() else {
                    continue;
                };
                let caller_name = caller_fn.get_identifier();
                let caller_env = data.caller_bindings();
                let key = (caller_name.clone(), caller_env.clone());
                if !seen.insert(key.clone()) {
                    continue;
                }
                callers.push((caller_name, caller_fn.clone(), caller_env));
            }
        }
    }
    callers
}

fn reachable_functions(
    top: &str,
    graph: &FunctionCallGraph,
) -> Result<HashSet<String>, XlsynthError> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    let mut top_function_handle: Option<Function> = None;
    for idx in 0..graph.function_count() {
        if let Some(function) = graph.get_function(idx) {
            if function.get_identifier() == top {
                top_function_handle = Some(function);
                break;
            }
        }
    }

    let Some(start_fn) = top_function_handle else {
        return Err(XlsynthError(format!(
            "Top function '{}' not present in call graph.",
            top
        )));
    };

    visited.insert(top.to_string());
    queue.push_back(start_fn);

    while let Some(current) = queue.pop_front() {
        let callee_count = graph.callee_count(&current);
        for idx in 0..callee_count {
            if let Some(callee) = graph.get_callee(&current, idx) {
                let callee_name = callee.get_identifier();
                if visited.insert(callee_name.clone()) {
                    queue.push_back(callee);
                }
            }
        }
    }

    Ok(visited)
}

fn collect_specializations(
    type_info: &TypeInfo,
    reachable: &HashSet<String>,
    functions: &HashMap<String, Function>,
) -> HashMap<String, BTreeSet<ParametricEnv>> {
    let mut plan: HashMap<String, BTreeSet<ParametricEnv>> = HashMap::new();
    for (callee_name, callee_fn) in functions {
        if !reachable.contains(callee_name) {
            continue;
        }
        if !callee_fn.is_parametric() {
            continue;
        }
        if let Some(data_array) = type_info.get_all_invocation_callee_data(callee_fn) {
            for idx in 0..data_array.len() {
                if let Some(data) = data_array.get(idx) {
                    let mut env_to_use = data.callee_bindings();
                    if env_to_use
                        .as_ref()
                        .map(|env| env.is_empty())
                        .unwrap_or(true)
                    {
                        if let Some(caller_env) = data.caller_bindings() {
                            if !caller_env.is_empty() {
                                env_to_use = Some(caller_env);
                            }
                        }
                    }
                    if let Some(env) = env_to_use {
                        if !env.is_empty() {
                            debug!(
                                "Recording specialization candidate for {} with env {}",
                                callee_name,
                                env.to_string()
                            );
                            plan.entry(callee_name.clone())
                                .or_insert_with(BTreeSet::new)
                                .insert(env);
                        }
                    }
                }
            }
        }
    }
    plan
}

struct SpecializationPlan {
    original_names: Vec<String>,
    specialized_names: Vec<String>,
    envs: Vec<ParametricEnv>,
}

fn specialization_suffix(
    module_name: &str,
    function_name: &str,
    env: &ParametricEnv,
) -> Result<String, XlsynthError> {
    if env.is_empty() {
        return Ok("specialized".to_string());
    }
    let mut binding_names: Vec<String> = env.bindings().into_iter().map(|(name, _)| name).collect();
    binding_names.sort();
    let key_refs: Vec<&str> = binding_names.iter().map(|s| s.as_str()).collect();
    let mangled = mangle_dslx_name_with_env(
        module_name,
        function_name,
        DslxCallingConvention::Typical,
        &key_refs,
        Some(env),
        None,
    )?;
    let parts = mangled
        .split("__")
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    Ok(parts.last().unwrap_or(&"specialized").to_string())
}

fn build_specialization_plan(
    module_name: &str,
    specializations: &HashMap<String, BTreeSet<ParametricEnv>>,
) -> Result<SpecializationPlan, XlsynthError> {
    let mut original_names = Vec::new();
    let mut specialized_names = Vec::new();
    let mut envs = Vec::new();
    let mut entries: Vec<(&String, &BTreeSet<ParametricEnv>)> = specializations.iter().collect();
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (function_name, env_set) in entries {
        for env in env_set {
            let suffix = specialization_suffix(module_name, function_name, env)?;
            let specialized_name = format!("{}_{}", function_name, suffix);
            original_names.push(function_name.clone());
            specialized_names.push(specialized_name);
            envs.push(env.clone());
        }
    }
    Ok(SpecializationPlan {
        original_names,
        specialized_names,
        envs,
    })
}

fn apply_specializations(
    tm: &TypecheckedModule,
    import_data: &mut ImportData,
    plan: &SpecializationPlan,
    install_subject: &str,
) -> Result<(TypecheckedModule, HashMap<(String, ParametricEnv), String>), XlsynthError> {
    if plan.original_names.is_empty() {
        return Ok((tm.clone(), HashMap::new()));
    }
    let mut current_tm = tm.clone();
    let mut name_map = HashMap::new();

    for i in 0..plan.original_names.len() {
        let request = FunctionSpecializationRequest {
            function_name: plan.original_names[i].as_str(),
            specialized_name: plan.specialized_names[i].as_str(),
            env: Some(plan.envs[i].clone()),
        };
        let unique_subject = format!("{}#{}", install_subject, i);
        match current_tm.insert_function_specializations(import_data, &[request], &unique_subject) {
            Ok(new_tm) => {
                name_map.insert(
                    (plan.original_names[i].clone(), plan.envs[i].clone()),
                    plan.specialized_names[i].clone(),
                );
                current_tm = new_tm;
            }
            Err(err) => {
                return Err(err);
            }
        }
    }

    Ok((current_tm, name_map))
}

struct RewriteEntry {
    from_callee_name: String,
    to_callee_name: String,
    env: ParametricEnv,
}

fn build_rewrite_plan(
    functions: &HashMap<String, Function>,
    spec_name_map: &HashMap<(String, ParametricEnv), String>,
) -> Result<Vec<RewriteEntry>, XlsynthError> {
    let mut entries = Vec::new();

    for ((original_name, env), specialized_name) in spec_name_map {
        let Some(_) = functions.get(original_name) else {
            continue;
        };
        let Some(_) = functions.get(specialized_name) else {
            continue;
        };

        entries.push(RewriteEntry {
            from_callee_name: original_name.clone(),
            to_callee_name: specialized_name.clone(),
            env: env.clone(),
        });
    }

    Ok(entries)
}

fn prune_unreachable_functions(
    tm: TypecheckedModule,
    import_data: &mut ImportData,
    module_name: &str,
    top_function: &str,
    extra_keep: &HashSet<String>,
) -> Result<TypecheckedModule, XlsynthError> {
    let module = tm.get_module();
    let functions = collect_functions(&module);
    let type_info = tm.get_type_info();
    let call_graph = type_info.build_function_call_graph()?;

    let mut reachable = reachable_functions(top_function, &call_graph)?;

    for keep in extra_keep {
        match reachable_functions(keep, &call_graph) {
            Ok(extra_reachable) => {
                reachable.extend(extra_reachable.into_iter());
            }
            Err(err) => {
                debug!(
                    "Skipping reachability expansion for '{}' due to: {}",
                    keep, err
                );
            }
        }
        reachable.insert(keep.clone());
    }

    debug!("Prune reachability set: {:?}", reachable);

    let mut to_remove_members: Vec<ModuleMember> = Vec::new();
    for (name, function) in &functions {
        if reachable.contains(name) {
            continue;
        }
        let member = ModuleMember::try_from(function)?;
        to_remove_members.push(member);
    }
    accumulate_test_and_quickcheck_members(&tm, &mut to_remove_members)?;
    if to_remove_members.is_empty() {
        return Ok(tm);
    }
    let refs: Vec<&ModuleMember> = to_remove_members.iter().collect();
    let install_subject = format!("{}.prune", module_name);
    tm.clone_ignoring_members(import_data, &refs, &install_subject)
}

fn accumulate_test_and_quickcheck_members(
    tm: &TypecheckedModule,
    members: &mut Vec<ModuleMember>,
) -> Result<(), XlsynthError> {
    let module = tm.get_module();
    for idx in 0..module.get_member_count() {
        let member = module.get_member(idx);
        match member.kind() {
            ModuleMemberKind::Quickcheck
            | ModuleMemberKind::TestFunction
            | ModuleMemberKind::TestProc => members.push(member),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn param_specialization_requires_multiple_phases() -> Result<(), XlsynthError> {
        let source = r#"
fn widen<N: u32>(x: bits[N]) -> bits[N] {
    x
}

fn intermediate<M: u32>(x: bits[M]) -> bits[M] {
    widen(x)
}

fn wrapper<K: u32>(x: bits[K]) -> bits[K] {
    intermediate(x)
}

pub fn top() -> bits[4] {
    let init = bits[4]:0b1010;
    wrapper(init)
}
"#;

        let specialized =
            specialize_dslx_module(source, Path::new("multiphase_test.x"), "top", None, &[])?;

        assert!(specialized.source.contains("fn wrapper_"));
        assert!(specialized.source.contains("fn intermediate_"));
        assert!(specialized.source.contains("fn widen_"));
        assert!(!specialized.source.contains("fn wrapper<K: u32>"));
        assert!(!specialized.source.contains("fn intermediate<M: u32>"));
        assert!(!specialized.source.contains("fn widen<N: u32>"));

        Ok(())
    }

    #[test]
    fn top_function_not_found_produces_error() {
        let source = r#"
fn identity<N: u32>(x: bits[N]) -> bits[N] { x }
"#;

        let err = specialize_dslx_module(
            source,
            Path::new("missing_top.x"),
            "does_not_exist",
            None,
            &[],
        )
        .expect_err("expected missing top error");
        let message = format!("{}", err);
        assert!(message.contains("Top function 'does_not_exist'"));
    }

    #[test]
    fn parametric_top_with_bindings_is_specialized() {
        let source = r#"
fn id<N: u32>(x: bits[N]) -> bits[N] { x }

fn helper<N: u32>(x: bits[N]) -> bits[N] { id(x) }

fn top<N: u32>(x: bits[N]) -> bits[N] { helper(x) }
"#;
        let specialized = specialize_dslx_module(
            source,
            Path::new("parametric_top_bindings.x"),
            "top<u32:32>",
            None,
            &[],
        )
        .expect("specialize with explicit bindings succeeds");
        assert!(
            specialized.source.contains("fn top_"),
            "Expected specialized top clone; module:\n{}",
            specialized.source
        );
        assert!(
            specialized.source.contains("fn helper_"),
            "Expected specialized helper clone; module:\n{}",
            specialized.source
        );
        assert!(
            !specialized.source.contains("fn top<N"),
            "Expected parametric top definition to be removed; module:\n{}",
            specialized.source
        );
        assert!(
            specialized.top_name.starts_with("top_"),
            "Expected specialized top name; got {}",
            specialized.top_name
        );
    }

    #[test]
    fn specialization_select_poly_regression() {
        let source = r#"
fn repeat<COUNT: u32, N: u32>(x: uN[N]) -> uN[N][COUNT] {
    uN[N][COUNT]:[x, ...]
}

fn select_poly<N: u32>(polys: uN[6][N], selector: u6) -> uN[6] {
    let repeated = repeat<N>(selector);
    let first = repeated[0];
    if first == selector { polys[0] } else { polys[1] }
}

pub fn top() -> uN[6] {
    let polys = uN[6][34]:[uN[6]:0, ...];
    select_poly(polys, u6:0)
}
"#;

        let specialized = specialize_dslx_module(
            source,
            Path::new("select_poly_regression.x"),
            "top",
            None,
            &[],
        )
        .expect("select_poly specialization succeeds");
        assert!(specialized.source.contains("fn repeat_"));
        assert!(specialized.source.contains("fn select_poly_"));
        assert!(!specialized.source.contains("fn repeat<"));
        assert!(!specialized.source.contains("fn select_poly<"));
    }

    #[test]
    fn specialization_fails_for_higher_order_parametric_helper() {
        let source = r#"
fn interval_matches<N: u32>(xy: (uN[N], uN[N])) -> bool {
    let (a, b) = xy;
    a == b
}

fn repeat<COUNT: u32, N: u32>(x: uN[N]) -> uN[N][COUNT] {
    uN[N][COUNT]:[x, ...]
}

fn select_poly<N: u32>(vals: uN[N][N], selector: uN[N]) -> bool {
    let rep = repeat<N>(selector);
    let mask = map(zip(vals, rep), interval_matches);
    mask[0]
}

pub fn top() -> bool {
    let vals = uN[6][6]:[uN[6]:0, ...];
    let selector = uN[6]:0;
    select_poly(vals, selector)
}
"#;

        let specialized = specialize_dslx_module(
            source,
            Path::new("higher_order_interval_matches.x"),
            "top",
            None,
            &[],
        )
        .expect("higher-order interval_matches specialization succeeds");
        assert!(specialized.source.contains("fn interval_matches_"));
        assert!(specialized.source.contains("fn repeat_"));
        assert!(specialized.source.contains("fn select_poly_"));
        assert!(specialized.source.contains("interval_matches_6"));
        assert!(!specialized.source.contains("fn interval_matches<"));
        assert!(!specialized.source.contains("fn repeat<"));
        assert!(!specialized.source.contains("fn select_poly<"));
        assert!(
            !specialized
                .source
                .contains("map(zip(vals, rep), interval_matches);")
        );
    }

    #[test]
    fn specialization_leaves_unrewritten_callers_for_discard_sign_bit() {
        let source = r#"
fn discard_sign_bit<N: u32, R: u32 = {N - u32:1}>(x: sN[N]) -> uN[R] {
    x as uN[R]
}

fn recip_normal_post(x: sN[24], flag: bool) -> uN[23] {
    if flag { uN[23]:0 } else { discard_sign_bit(x) }
}

fn rsqrt_normal_post(x: sN[24], is_even: bool) -> uN[23] {
    let raw = discard_sign_bit(x);
    if is_even { raw } else { raw }
}

fn select_poly<N: u32>(vals: sN[24][N], selector: u6) -> sN[24] {
    vals[u32:0]
}

pub fn top() -> (uN[23], uN[23]) {
    let vals = sN[24][34]:[sN[24]:0, ...];
    let val = select_poly(vals, u6:0);
    (
        recip_normal_post(val, false),
        rsqrt_normal_post(val, true),
    )
}
"#;

        let specialized = specialize_dslx_module(
            source,
            Path::new("discard_sign_bit_unrewritten.x"),
            "top",
            None,
            &[],
        )
        .expect("discard_sign_bit callers should be fully rewritten");
        assert!(specialized.source.contains("discard_sign_bit_24_23"));
        assert!(!specialized.source.contains("fn discard_sign_bit<N"));
        assert!(!specialized.source.contains("discard_sign_bit(val"));
    }

    #[test]
    fn specialization_rewrites_nested_call_chain() -> Result<(), XlsynthError> {
        let source = r#"
fn widen<N: u32>(x: bits[N]) -> bits[N] {
    x
}

fn intermediate<M: u32>(x: bits[M]) -> bits[M] {
    widen(x)
}

fn wrapper<K: u32>(x: bits[K]) -> bits[K] {
    intermediate(x)
}

pub fn top() -> bits[8] {
    let init = bits[8]:0b1100_0011;
    wrapper(init)
}
"#;

        let specialized =
            specialize_dslx_module(source, Path::new("chain_rewrite_test.x"), "top", None, &[])?;

        assert!(specialized.source.contains("fn wrapper_"));
        assert!(specialized.source.contains("fn intermediate_"));
        assert!(specialized.source.contains("fn widen_"));
        assert!(specialized.source.contains("widen_8("));
        assert!(specialized.source.contains("intermediate_8("));
        assert!(!specialized.source.contains("fn widen<N: u32>"));

        Ok(())
    }

    #[test]
    fn prune_removes_unreachable_functions() -> Result<(), XlsynthError> {
        let source = r#"
fn helper<N: u32>(x: uN[N]) -> uN[N] {
    x
}

fn unused(x: u32) -> u32 {
    helper(x)
}

pub fn top() -> u32 {
    helper(u32:42)
}
"#;

        let specialized =
            specialize_dslx_module(source, Path::new("prune_unused.x"), "top", None, &[])?;

        assert!(specialized.source.contains("fn helper_"));
        assert!(!specialized.source.contains("fn unused"));

        Ok(())
    }

    #[test]
    fn prune_removes_quickchecks_and_tests() -> Result<(), XlsynthError> {
        let source = r#"
fn helper_function(x: u32) -> u32 {
    x + u32:1
}

fn helper_secondary_function(x: u32) -> u32 {
    x + u32:1
}

#[quickcheck]
fn helper_property(x: u32) -> bool {
    helper_secondary_function(x) >= x
}

#[test]
fn helper_test() {
    let _ = helper_function(u32:5);
    ()
}

pub fn top() -> u32 {
    helper_function(u32:42)
}
"#;

        let specialized = specialize_dslx_module(
            source,
            Path::new("prune_quickcheck_test.x"),
            "top",
            None,
            &[],
        )?;

        assert!(specialized.source.contains("fn helper_function"));
        assert!(!specialized.source.contains("helper_property"));
        assert!(!specialized.source.contains("helper_secondary_function"));
        assert!(!specialized.source.contains("helper_test"));

        Ok(())
    }

    #[test]
    fn specialization_handles_multiple_environments() -> Result<(), XlsynthError> {
        let source = r#"
fn widen<N: u32>(x: bits[N]) -> bits[N] {
    x
}

fn wrapper<K: u32>(x: bits[K]) -> bits[K] {
    widen(x)
}

pub fn top() -> (bits[4], bits[6]) {
    let a = bits[4]:0b0011;
    let b = bits[6]:0b10_0101;
    (wrapper(a), wrapper(b))
}
"#;

        let specialized =
            specialize_dslx_module(source, Path::new("multi_env_test.x"), "top", None, &[])?;

        assert!(specialized.source.contains("fn wrapper_"));
        assert!(specialized.source.contains("fn wrapper_4"));
        assert!(specialized.source.contains("fn wrapper_6"));
        assert!(specialized.source.contains("fn widen_4"));
        assert!(specialized.source.contains("fn widen_6"));
        assert!(specialized.source.contains("wrapper_4("));
        assert!(specialized.source.contains("wrapper_6("));
        assert!(specialized.source.contains("widen_4("));
        assert!(specialized.source.contains("widen_6("));
        assert!(!specialized.source.contains("fn wrapper<K: u32>"));
        assert!(!specialized.source.contains("fn widen<N: u32>"));

        Ok(())
    }

    #[test]
    fn specialization_deduplicates_identical_environments() -> Result<(), XlsynthError> {
        fn count_occurrences(haystack: &str, needle: &str) -> usize {
            haystack.match_indices(needle).count()
        }

        let source = r#"
fn widen<N: u32>(x: bits[N]) -> bits[N] {
    x
}

fn wrapper<K: u32>(x: bits[K]) -> bits[K] {
    widen(x)
}

pub fn top() -> (bits[8], bits[8]) {
    let a = bits[8]:0xAB;
    let b = bits[8]:0xCD;
    (wrapper(a), wrapper(b))
}
"#;

        let specialized =
            specialize_dslx_module(source, Path::new("dedupe_env_test.x"), "top", None, &[])?;

        assert_eq!(count_occurrences(&specialized.source, "fn wrapper_8"), 1);
        assert_eq!(count_occurrences(&specialized.source, "fn widen_8"), 1);
        assert!(specialized.source.contains("wrapper_8("));
        assert!(specialized.source.contains("widen_8("));
        assert!(!specialized.source.contains("fn wrapper<K: u32>"));
        assert!(!specialized.source.contains("fn widen<N: u32>"));

        Ok(())
    }

    #[test]
    fn specialization_handles_simple_parametric_helper() -> Result<(), XlsynthError> {
        let source = r#"
fn to_bits<NB: u32>(x: bits[NB]) -> bits[NB] { x }

fn wrapper(x: bits[24]) -> bits[24] {
    to_bits(x)
}

pub fn top() -> bits[24] {
    wrapper(bits[24]:0)
}
"#;

        let specialized =
            specialize_dslx_module(source, Path::new("simple_param_helper.x"), "top", None, &[])?;

        assert!(specialized.source.contains("fn to_bits_"));
        assert!(!specialized.source.contains("fn to_bits<"));

        Ok(())
    }
}
