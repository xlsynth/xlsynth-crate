// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use crate::toolchain_config::ToolchainConfig;
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gate_fn_equiv_report;

fn load_aig_gate_fn(path: &Path) -> Result<GateFn, String> {
    load_aiger_auto_from_path(path, GateBuilderOptions::no_opt())
        .map(|res| res.gate_fn)
        .map_err(|e| format!("failed to load {}: {}", path.display(), e))
}

fn gate_fn_signature_summary(g: &GateFn) -> String {
    let inputs = g
        .inputs
        .iter()
        .map(|i| format!("{}:bits[{}]", i.name, i.get_bit_count()))
        .collect::<Vec<String>>()
        .join(", ");
    let outputs = g
        .outputs
        .iter()
        .map(|o| format!("{}:bits[{}]", o.name, o.get_bit_count()))
        .collect::<Vec<String>>()
        .join(", ");
    format!("inputs=[{}] outputs=[{}]", inputs, outputs)
}

fn gate_fn_signatures_match(lhs: &GateFn, rhs: &GateFn) -> bool {
    if lhs.inputs.len() != rhs.inputs.len() {
        return false;
    }
    if lhs.outputs.len() != rhs.outputs.len() {
        return false;
    }
    if lhs
        .inputs
        .iter()
        .zip(rhs.inputs.iter())
        .any(|(a, b)| a.get_bit_count() != b.get_bit_count())
    {
        return false;
    }
    if lhs
        .outputs
        .iter()
        .zip(rhs.outputs.iter())
        .any(|(a, b)| a.get_bit_count() != b.get_bit_count())
    {
        return false;
    }
    true
}

pub fn handle_aig_equiv(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let lhs_path = Path::new(matches.get_one::<String>("lhs_aig_file").unwrap());
    let rhs_path = Path::new(matches.get_one::<String>("rhs_aig_file").unwrap());

    let lhs = match load_aig_gate_fn(lhs_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("aig-equiv error: {}", e);
            std::process::exit(2);
        }
    };
    let rhs = match load_aig_gate_fn(rhs_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("aig-equiv error: {}", e);
            std::process::exit(2);
        }
    };

    if !gate_fn_signatures_match(&lhs, &rhs) {
        eprintln!(
            "aig-equiv error: gate function signatures do not match\nlhs: {}\nrhs: {}",
            gate_fn_signature_summary(&lhs),
            gate_fn_signature_summary(&rhs)
        );
        std::process::exit(2);
    }

    let report = gate_fn_equiv_report::prove_gate_fn_equiv_report(&lhs, &rhs);
    serde_json::to_writer(std::io::stdout(), &report).unwrap();
    println!();

    if report.results.values().any(|r| !r.is_equiv()) {
        std::process::exit(1);
    }
}
