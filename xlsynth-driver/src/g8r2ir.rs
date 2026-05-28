// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use xlsynth_g8r::aig_serdes::g8r::load_gate_fn_from_path;
use xlsynth_g8r::aig_serdes::gate2ir::gate_fn_to_xlsynth_ir;

use crate::toolchain_config::ToolchainConfig;

pub fn handle_g8r2ir(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let g8r_input_file = matches.get_one::<String>("g8r_input_file").unwrap();
    let g8r_path = Path::new(g8r_input_file);

    let gate_fn = match load_gate_fn_from_path(g8r_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("g8r2ir error: {}", e);
            std::process::exit(2);
        }
    };

    let flat_type = gate_fn.get_flat_type();
    let ir_pkg = match gate_fn_to_xlsynth_ir(&gate_fn, "gate", &flat_type) {
        Ok(pkg) => pkg,
        Err(e) => {
            eprintln!(
                "g8r2ir error: failed to convert GateFn to XLS IR for {}: {}",
                g8r_input_file, e
            );
            std::process::exit(2);
        }
    };

    println!("{}", ir_pkg.to_string());
}
