// SPDX-License-Identifier: Apache-2.0

use crate::common::{parse_uf_spec, resolve_type_inference_v2};
use crate::ir_equiv::outcome_from_report;
use crate::proofs::obligations::{LecObligation, LecSide, ObligationPayload, ProverObligation};
use crate::proofs::script::{
    execute_script, read_script_steps_from_json_path, read_script_steps_from_jsonl_path, OblTree,
    OblTreeConfig, ScriptStep,
};
use crate::toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig};
use std::path::{Path, PathBuf};
use xlsynth_prover::dslx_equiv::{run_dslx_equiv, DslxEquivRequest, DslxModule};
use xlsynth_prover::dslx_specializer::specialize_dslx_module;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism};
use xlsynth_prover::prover::SolverChoice;

const SUBCOMMAND: &str = "dslx-equiv";

pub fn handle_dslx_equiv(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_dslx_equiv");
    // Gather optional assertion-label filter regex for reuse.
    let assert_label_filter = matches.get_one::<String>("assert_label_filter").cloned();
    let lhs_file = matches.get_one::<String>("lhs_dslx_file").unwrap();
    let rhs_file = matches.get_one::<String>("rhs_dslx_file").unwrap();

    let tactic_json_path = matches.get_one::<String>("tactic_json").cloned();
    let tactic_jsonl_path = matches.get_one::<String>("tactic_jsonl").cloned();
    if tactic_json_path.is_some() && tactic_jsonl_path.is_some() {
        eprintln!("Error: --tactic_json and --tactic_jsonl cannot be used together");
        std::process::exit(1);
    }
    let tactic_script_json = tactic_json_path.clone();
    let tactic_script_jsonl = tactic_jsonl_path.clone();

    let mut lhs_top = matches
        .get_one::<String>("lhs_dslx_top")
        .map(|s| s.as_str());
    let mut rhs_top = matches
        .get_one::<String>("rhs_dslx_top")
        .map(|s| s.as_str());

    let top = matches.get_one::<String>("dslx_top").map(|s| s.as_str());

    if top.is_some() && (lhs_top.is_some() || rhs_top.is_some()) {
        eprintln!("Error: --dslx_top and --lhs_dslx_top/--rhs_dslx_top cannot be used together");
        std::process::exit(1);
    }
    if lhs_top.is_some() ^ rhs_top.is_some() {
        eprintln!("Error: --lhs_dslx_top and --rhs_dslx_top must be used together");
        std::process::exit(1);
    }
    if top.is_some() {
        lhs_top = top;
        rhs_top = top;
    }

    if lhs_top.is_none() || rhs_top.is_none() {
        eprintln!("Error: a top function must be specified via --dslx_top or both --lhs_dslx_top/--rhs_dslx_top");
        std::process::exit(1);
    }

    let lhs_path = std::path::Path::new(lhs_file);
    let rhs_path = std::path::Path::new(rhs_file);

    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path_ref = dslx_stdlib_path.as_deref().map(Path::new);
    let dslx_path = get_dslx_path(matches, config);
    let additional_search_paths: Vec<PathBuf> = dslx_path
        .as_deref()
        .map(|paths| {
            paths
                .split(';')
                .filter(|p| !p.is_empty())
                .map(std::path::PathBuf::from)
                .collect()
        })
        .unwrap_or_default();

    let enable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.enable_warnings.as_deref());
    let disable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.disable_warnings.as_deref());

    let type_inference_v2 = resolve_type_inference_v2(matches, config);

    let assertion_semantics = matches
        .get_one::<String>("assertion_semantics")
        .map(|s| s.parse().unwrap())
        .unwrap_or(AssertionSemantics::Same);

    let solver_choice: Option<SolverChoice> = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap());

    let flatten_aggregates = matches
        .get_one::<String>("flatten_aggregates")
        .map(|s| s == "true")
        .unwrap_or(false);
    let drop_params: Vec<String> = matches
        .get_one::<String>("drop_params")
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(Vec::new);

    let strategy = matches
        .get_one::<String>("parallelism_strategy")
        .map(|s| s.parse().unwrap())
        .unwrap_or(EquivParallelism::SingleThreaded);

    let lhs_fixed_implicit_activation = matches
        .get_one::<String>("lhs_fixed_implicit_activation")
        .map(|s| s.parse().unwrap())
        .unwrap_or(false);
    let rhs_fixed_implicit_activation = matches
        .get_one::<String>("rhs_fixed_implicit_activation")
        .map(|s| s.parse().unwrap())
        .unwrap_or(false);

    let output_json = matches.get_one::<String>("output_json");

    let assume_enum_in_bound = matches
        .get_one::<String>("assume-enum-in-bound")
        .map(|s| s == "true")
        .unwrap_or(true);

    let lhs_module_name = lhs_path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("valid LHS module name");
    let rhs_module_name = rhs_path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("valid RHS module name");

    // Parse uninterpreted function mappings (lhs/rhs).
    // Format: <func_name>:<uf_name>
    // Semantics: Functions mapped to the same <uf_name> are assumed equivalent
    // (treated as the same uninterpreted symbol) and assertions inside them are
    // ignored.
    let lhs_uf_map = parse_uf_spec(lhs_module_name, matches.get_many::<String>("lhs_uf"));
    let rhs_uf_map = parse_uf_spec(rhs_module_name, matches.get_many::<String>("rhs_uf"));
    let use_unoptimized_ir = !lhs_uf_map.is_empty() || !rhs_uf_map.is_empty();

    if tactic_script_json.is_some() || tactic_script_jsonl.is_some() {
        // Use tactic-based prover path.
        // Build root obligation from DSLX files.
        let lhs_top_str = lhs_top.unwrap();
        let rhs_top_str = rhs_top.unwrap();
        let lhs_pb = lhs_path.to_path_buf();
        let rhs_pb = rhs_path.to_path_buf();
        let mut lhs_side = LecSide::from_path(lhs_top_str, &lhs_pb);
        let mut rhs_side = LecSide::from_path(rhs_top_str, &rhs_pb);
        // UF mappings per side.
        lhs_side.uf_map = lhs_uf_map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        rhs_side.uf_map = rhs_uf_map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let root_ob = ProverObligation {
            selector_segment: String::new(),
            description: None,
            payload: ObligationPayload::Lec(LecObligation {
                lhs: lhs_side,
                rhs: rhs_side,
            }),
        };
        let dslx_paths_vec: Vec<std::path::PathBuf> = dslx_path
            .map(|s| {
                s.split(';')
                    .filter(|p| !p.is_empty())
                    .map(|p| std::path::PathBuf::from(p))
                    .collect()
            })
            .unwrap_or_default();
        let cfg = OblTreeConfig {
            dslx_stdlib_path: dslx_stdlib_path.map(|p| std::path::PathBuf::from(p)),
            dslx_paths: dslx_paths_vec,
            solver: solver_choice,
            timeout_ms: None,
        };
        let mut tree = OblTree::new(root_ob, cfg);

        // Read & parse tactic script from file (JSON array or JSONL).
        let steps: Vec<ScriptStep> = if let Some(path) = tactic_script_json {
            match read_script_steps_from_json_path(&path) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[{}] {}", SUBCOMMAND, e);
                    std::process::exit(2);
                }
            }
        } else if let Some(path) = tactic_script_jsonl {
            match read_script_steps_from_jsonl_path(&path) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[{}] {}", SUBCOMMAND, e);
                    std::process::exit(2);
                }
            }
        } else {
            unreachable!("tactic script presence already checked");
        };

        let start = std::time::Instant::now();
        let report = match execute_script(&mut tree, &steps) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[{}] execute_script error: {}", SUBCOMMAND, e);
                std::process::exit(2);
            }
        };
        let dur = start.elapsed();
        let success = report.failed.is_empty() && report.indefinite.is_empty();
        if let Some(path) = output_json {
            let json = serde_json::json!({
                "success": success,
                "report": report,
            });
            std::fs::write(path, serde_json::to_string(&json).unwrap()).unwrap();
        }
        if success {
            println!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
            println!("[{}] success: Solver proved equivalence", SUBCOMMAND);
            std::process::exit(0);
        } else {
            eprintln!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
            eprintln!("[{}] failure", SUBCOMMAND);
            std::process::exit(1);
        }
    } else {
        // Original direct prover path (now via library API).
        let lhs_contents = std::fs::read_to_string(lhs_path).unwrap_or_else(|e| {
            eprintln!("Failed to read DSLX file {}: {}", lhs_path.display(), e);
            std::process::exit(1);
        });
        let rhs_contents = std::fs::read_to_string(rhs_path).unwrap_or_else(|e| {
            eprintln!("Failed to read DSLX file {}: {}", rhs_path.display(), e);
            std::process::exit(1);
        });

        let lhs_top_spec = lhs_top.unwrap();
        let rhs_top_spec = rhs_top.unwrap();

        let lhs_specialized = match specialize_dslx_module(
            &lhs_contents,
            lhs_path,
            lhs_top_spec,
            dslx_stdlib_path_ref,
            &additional_search_paths,
        ) {
            Ok(output) => output,
            Err(err) => {
                eprintln!(
                    "[{}] DSLX specialization failed\n  path: {}\n  error: {}",
                    SUBCOMMAND,
                    lhs_path.display(),
                    err
                );
                std::process::exit(1);
            }
        };
        let rhs_specialized = match specialize_dslx_module(
            &rhs_contents,
            rhs_path,
            rhs_top_spec,
            dslx_stdlib_path_ref,
            &additional_search_paths,
        ) {
            Ok(output) => output,
            Err(err) => {
                eprintln!(
                    "[{}] DSLX specialization failed\n  path: {}\n  error: {}",
                    SUBCOMMAND,
                    rhs_path.display(),
                    err
                );
                std::process::exit(1);
            }
        };

        let lhs_source_owned = lhs_specialized.source;
        let rhs_source_owned = rhs_specialized.source;
        let lhs_top_resolved = lhs_specialized.top_name;
        let rhs_top_resolved = rhs_specialized.top_name;
        let lhs_top = lhs_top_resolved.as_str();
        let rhs_top = rhs_top_resolved.as_str();

        let tool_path = config
            .as_ref()
            .and_then(|c| c.tool_path.as_deref())
            .map(Path::new);

        let lhs_module = DslxModule::new(lhs_source_owned.as_str(), lhs_top)
            .with_path(Some(lhs_path))
            .with_uf_map(&lhs_uf_map)
            .with_fixed_implicit_activation(lhs_fixed_implicit_activation);
        let rhs_module = DslxModule::new(rhs_source_owned.as_str(), rhs_top)
            .with_path(Some(rhs_path))
            .with_uf_map(&rhs_uf_map)
            .with_fixed_implicit_activation(rhs_fixed_implicit_activation);

        let request = DslxEquivRequest::new(lhs_module, rhs_module)
            .with_drop_params(&drop_params)
            .with_flatten_aggregates(flatten_aggregates)
            .with_parallelism(strategy)
            .with_assertion_semantics(assertion_semantics)
            .with_assert_label_filter(assert_label_filter.as_deref())
            .with_solver(solver_choice)
            .with_assume_enum_in_bound(assume_enum_in_bound)
            .with_optimize(!use_unoptimized_ir)
            .with_tool_path(tool_path)
            .with_dslx_stdlib_path(dslx_stdlib_path_ref)
            .with_additional_search_paths(additional_search_paths)
            .with_enable_warnings(enable_warnings)
            .with_disable_warnings(disable_warnings)
            .with_type_inference_v2(type_inference_v2);

        let report = match run_dslx_equiv(&request) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("[{}] {}", SUBCOMMAND, err);
                std::process::exit(1);
            }
        };

        let outcome = outcome_from_report(report);

        if let Some(path) = output_json {
            std::fs::write(path, serde_json::to_string(&outcome).unwrap()).unwrap();
        }
        let dur = std::time::Duration::from_micros(outcome.time_micros as u64);
        if outcome.success {
            println!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
            println!("[{}] success: Solver proved equivalence", SUBCOMMAND);
            std::process::exit(0);
        } else {
            eprintln!("[{}] Time taken: {:?}", SUBCOMMAND, dur);
            if let Some(err) = outcome.error_str.as_ref() {
                eprintln!("[{}] failure: {}", SUBCOMMAND, err);
            } else {
                eprintln!("[{}] failure", SUBCOMMAND);
            }
            std::process::exit(1);
        }
    }
}
