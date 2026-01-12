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

    let report = gate_fn_equiv_report::prove_gate_fn_equiv_report(&lhs, &rhs);
    serde_json::to_writer(std::io::stdout(), &report).unwrap();
    println!();

    if report.results.values().any(|r| !r.is_equiv()) {
        std::process::exit(1);
    }
}
