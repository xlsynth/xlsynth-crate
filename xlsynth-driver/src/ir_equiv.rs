// SPDX-License-Identifier: Apache-2.0

use crate::toolchain_config::ToolchainConfig;
use serde::Serialize;
use std::path::Path;
use xlsynth_prover::ir_equiv::run_ir_equiv as prover_run_ir_equiv;
pub use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule};
use xlsynth_prover::prover::SolverChoice;
use xlsynth_prover::types::EquivParallelism;
use xlsynth_prover::types::{AssertionSemantics, EquivReport};

const SUBCOMMAND: &str = "ir-equiv";

#[derive(Debug, Serialize, Clone)]
pub struct EquivOutcome {
    pub time_micros: u128,
    pub success: bool,
    pub error_str: Option<String>,
}

/// Implements the "ir-equiv" subcommand.
pub fn handle_ir_equiv(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir_equiv");
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();

    let mut lhs_top = matches.get_one::<String>("lhs_ir_top").map(|s| s.as_str());
    let mut rhs_top = matches.get_one::<String>("rhs_ir_top").map(|s| s.as_str());

    let top = matches.get_one::<String>("ir_top").map(|s| s.as_str());

    if top.is_some() && (lhs_top.is_some() || rhs_top.is_some()) {
        eprintln!("Error: --ir_top and --lhs_ir_top/--rhs_ir_top cannot be used together");
        std::process::exit(1);
    }
    if lhs_top.is_some() ^ rhs_top.is_some() {
        eprintln!("Error: --lhs_ir_top and --rhs_ir_top must be used together");
        std::process::exit(1);
    }
    if top.is_some() {
        lhs_top = top;
        rhs_top = lhs_top;
    }

    let assertion_semantics = matches
        .get_one::<AssertionSemantics>("assertion_semantics")
        .unwrap_or(&AssertionSemantics::Same);

    let solver: Option<SolverChoice> = matches
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

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());

    let lhs_ir_text = std::fs::read_to_string(lhs).unwrap_or_else(|e| {
        eprintln!("Failed to read lhs IR file: {}", e);
        std::process::exit(1)
    });
    let rhs_ir_text = std::fs::read_to_string(rhs).unwrap_or_else(|e| {
        eprintln!("Failed to read rhs IR file: {}", e);
        std::process::exit(1)
    });

    let assert_label_filter = matches
        .get_one::<String>("assert_label_filter")
        .map(|s| s.as_str());

    let tool_path_ref = tool_path.map(Path::new);

    let request = IrEquivRequest::new(
        IrModule::new(&lhs_ir_text)
            .with_path(Some(Path::new(lhs)))
            .with_top(lhs_top)
            .with_fixed_implicit_activation(lhs_fixed_implicit_activation),
        IrModule::new(&rhs_ir_text)
            .with_path(Some(Path::new(rhs)))
            .with_top(rhs_top)
            .with_fixed_implicit_activation(rhs_fixed_implicit_activation),
    )
    .with_drop_params(&drop_params)
    .with_flatten_aggregates(flatten_aggregates)
    .with_parallelism(strategy)
    .with_assertion_semantics(*assertion_semantics)
    .with_assert_label_filter(assert_label_filter)
    .with_solver(solver)
    .with_tool_path(tool_path_ref);

    let outcome = dispatch_ir_equiv(&request, SUBCOMMAND);
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

pub fn dispatch_ir_equiv(request: &IrEquivRequest<'_>, subcommand: &str) -> EquivOutcome {
    log::info!(
        "dispatch_ir_equiv; solver_choice: {:?}, tool_path: {:?}",
        request.solver,
        request.tool_path
    );
    match prover_run_ir_equiv(request) {
        Ok(report) => outcome_from_report(report),
        Err(err) => {
            eprintln!("[{}] {}", subcommand, err);
            std::process::exit(1);
        }
    }
}

pub fn outcome_from_report(report: EquivReport) -> EquivOutcome {
    EquivOutcome {
        time_micros: report.duration.as_micros(),
        success: report.is_success(),
        error_str: report.error_str(),
    }
}
