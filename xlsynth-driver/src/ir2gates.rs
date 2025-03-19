// SPDX-License-Identifier: Apache-2.0

//! Accepts input IR and then performs the g8r IR-to-gates mapping on it.

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use xlsynth_g8r::process_ir_path;

fn ir2gates(input_file: &std::path::Path, quiet: bool) {
    log::info!("ir2gates");
    let stats = process_ir_path::process_ir_path(input_file, false, false, quiet);
    if quiet {
        serde_json::to_writer(std::io::stdout(), &stats).unwrap();
        println!();
    }
}

pub fn handle_ir2gates(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let quiet = match matches.get_one::<String>("quiet").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };
    let input_path = std::path::Path::new(input_file);

    ir2gates(input_path, quiet);
}
