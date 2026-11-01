// SPDX-License-Identifier: Apache-2.0

use crate::proofs::report_setup_error;
use crate::toolchain_config::ToolchainConfig;
use std::path::Path;
use std::time::Instant;
pub use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule};
use xlsynth_prover::proof_outcome::ProofSummary;
use xlsynth_prover::prover::SolverChoice;
use xlsynth_prover::prover::types::AssertionSemantics;
use xlsynth_prover::prover::types::EquivParallelism;

const SUBCOMMAND: &str = "ir-equiv";

/// Implements the "ir-equiv" subcommand.
pub fn handle_ir_equiv(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir_equiv");
    let start = Instant::now();
    let output_json = matches.get_one::<String>("output_json");
    let fail = |message: &str| -> ! {
        report_setup_error(start, output_json.map(String::as_str), SUBCOMMAND, message)
    };
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();

    let mut lhs_top = matches.get_one::<String>("lhs_ir_top").map(|s| s.as_str());
    let mut rhs_top = matches.get_one::<String>("rhs_ir_top").map(|s| s.as_str());

    let top = matches.get_one::<String>("ir_top").map(|s| s.as_str());

    if top.is_some() && (lhs_top.is_some() || rhs_top.is_some()) {
        fail("--ir_top and --lhs_ir_top/--rhs_ir_top cannot be used together");
    }
    if lhs_top.is_some() ^ rhs_top.is_some() {
        fail("--lhs_ir_top and --rhs_ir_top must be used together");
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
    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());

    let read_side = |side, path: &str| {
        std::fs::read_to_string(path)
            .unwrap_or_else(|err| fail(&format!("Failed to read {side} IR file: {err}")))
    };
    let lhs_ir_text = read_side("lhs", lhs);
    let rhs_ir_text = read_side("rhs", rhs);

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
    emit_equiv_outcome_and_exit(&outcome, SUBCOMMAND, output_json);
}

pub fn dispatch_ir_equiv(request: &IrEquivRequest<'_>, subcommand: &str) -> ProofSummary {
    log::info!(
        "[{subcommand}] dispatch_ir_equiv; solver_choice: {:?}, tool_path: {:?}",
        request.solver,
        request.tool_path
    );
    request.prove()
}

pub fn emit_equiv_outcome_and_exit(
    outcome: &ProofSummary,
    subcommand: &str,
    output_json: Option<&String>,
) -> ! {
    if let Some(path) = output_json {
        std::fs::write(path, serde_json::to_string(outcome).unwrap()).unwrap();
    }
    let dur = std::time::Duration::from_micros(outcome.time_micros as u64);
    if outcome.conclusion.status.is_proved() {
        println!("[{}] Time taken: {:?}", subcommand, dur);
        println!("[{}] success: Solver proved equivalence", subcommand);
    } else {
        eprintln!("[{}] Time taken: {:?}", subcommand, dur);
        if let Some(err) = outcome.error_str.as_ref() {
            eprintln!("[{}] failure: {}", subcommand, err);
        } else {
            eprintln!("[{}] failure", subcommand);
        }
    }
    std::process::exit(outcome.conclusion.status.exit_code());
}
