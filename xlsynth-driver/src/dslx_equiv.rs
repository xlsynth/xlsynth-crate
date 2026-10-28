// SPDX-License-Identifier: Apache-2.0

use crate::common::parse_dslx_uf_spec;
use crate::ir_equiv::emit_equiv_outcome_and_exit;
use crate::proofs::script::TacticScript;
use crate::toolchain_config::{ToolchainConfig, get_dslx_path, get_dslx_stdlib_path, proof_driver};
use std::path::{Path, PathBuf};
use xlsynth_prover::dslx_equiv::{DslxEquivFile, DslxEquivFiles};
use xlsynth_prover::dslx_tactics::obligations::LecProofOptions;
use xlsynth_prover::prover::SolverChoice;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism};

const SUBCOMMAND: &str = "dslx-equiv";

pub fn handle_dslx_equiv(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_dslx_equiv");
    // Gather optional assertion-label filter regex for reuse.
    let assert_label_filter = matches.get_one::<String>("assert_label_filter").cloned();
    let lhs_file = matches.get_one::<String>("lhs_dslx_file").unwrap();
    let rhs_file = matches.get_one::<String>("rhs_dslx_file").unwrap();

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

    let (Some(lhs_top), Some(rhs_top)) = (lhs_top, rhs_top) else {
        eprintln!(
            "Error: a top function must be specified via --dslx_top or both --lhs_dslx_top/--rhs_dslx_top"
        );
        std::process::exit(1);
    };

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

    let assertion_semantics = matches
        .get_one::<String>("assertion_semantics")
        .map(|s| s.parse().unwrap())
        .unwrap_or(AssertionSemantics::Ignore);

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

    let request = DslxEquivFiles {
        lhs: DslxEquivFile {
            path: lhs_path,
            top: lhs_top,
            uf_map: parse_dslx_uf_spec(matches.get_many::<String>("lhs_uf")),
            fixed_implicit_activation: lhs_fixed_implicit_activation,
        },
        rhs: DslxEquivFile {
            path: rhs_path,
            top: rhs_top,
            uf_map: parse_dslx_uf_spec(matches.get_many::<String>("rhs_uf")),
            fixed_implicit_activation: rhs_fixed_implicit_activation,
        },
        options: LecProofOptions {
            assertion_semantics,
            assume_enum_in_bound,
            flatten_aggregates,
            drop_params,
            assert_label_filter,
            parallelism: strategy,
        },
        solver: solver_choice,
        dslx_stdlib_path: dslx_stdlib_path_ref,
        additional_search_paths,
    };
    if let Some(script) = TacticScript::from_matches(matches) {
        script.execute_and_exit(
            request.into_tactic_tree(),
            SUBCOMMAND,
            "Solver proved equivalence",
            output_json.map(String::as_str),
            proof_driver(matches),
        );
    }

    let tool_path = config
        .as_ref()
        .and_then(|c| c.tool_path.as_deref())
        .map(Path::new);
    let outcome = request.prove(tool_path, enable_warnings, disable_warnings);
    emit_equiv_outcome_and_exit(&outcome, SUBCOMMAND, output_json);
}
