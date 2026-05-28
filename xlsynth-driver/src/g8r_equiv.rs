// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig_serdes::g8r::load_gate_fn_from_path;
use xlsynth_prover::prover::SolverChoice;

use std::path::Path;

use crate::gate_ir_equiv::prove_gate_fns_equiv_via_ir;
use crate::ir_equiv::emit_equiv_outcome_and_exit;
use crate::toolchain_config::ToolchainConfig;

pub fn handle_g8r_equiv(matches: &clap::ArgMatches, config: &Option<ToolchainConfig>) {
    let lhs_path = Path::new(matches.get_one::<String>("lhs_g8r_file").unwrap());
    let rhs_path = Path::new(matches.get_one::<String>("rhs_g8r_file").unwrap());
    let solver: Option<SolverChoice> = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap());
    let output_json = matches.get_one::<String>("output_json");

    let lhs = match load_gate_fn_from_path(lhs_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("g8r-equiv error: {}", e);
            std::process::exit(2);
        }
    };
    let rhs = match load_gate_fn_from_path(rhs_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("g8r-equiv error: {}", e);
            std::process::exit(2);
        }
    };

    let tool_path = config
        .as_ref()
        .and_then(|c| c.tool_path.as_deref())
        .map(Path::new);
    let outcome = match prove_gate_fns_equiv_via_ir(&lhs, &rhs, solver, tool_path, "g8r-equiv") {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!("g8r-equiv error: {}", e);
            std::process::exit(2);
        }
    };
    emit_equiv_outcome_and_exit(&outcome, "g8r-equiv", output_json);
}
