// SPDX-License-Identifier: Apache-2.0

//! Accepts input IR and then performs the g8r IR-to-gates mapping on it.

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use xlsynth_g8r::process_ir_path;

fn ir2gates(
    input_file: &std::path::Path,
    quiet: bool,
    fold: bool,
    hash: bool,
    fraig: bool,
    toggle_sample_count: usize,
    toggle_sample_seed: u64,
    compute_graph_logical_effort: bool,
    graph_logical_effort_beta1: f64,
    graph_logical_effort_beta2: f64,
) {
    log::info!("ir2gates");
    let options = process_ir_path::Options {
        check_equivalence: false,
        fold,
        hash,
        fraig,
        quiet,
        emit_netlist: false,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
    };
    let stats = process_ir_path::process_ir_path(input_file, &options);
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
    let fold = match matches.get_one::<String>("fold").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => true, // default for folding is true
    };
    let hash = match matches.get_one::<String>("hash").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => true, // default for hashing is true
    };
    let fraig = match matches.get_one::<String>("fraig").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => true, // default for fraig is true
    };
    let toggle_sample_count = matches
        .get_one::<String>("toggle_sample_count")
        .map(|s| s.parse::<usize>().unwrap_or(0))
        .unwrap_or(0);
    let toggle_sample_seed = matches
        .get_one::<String>("toggle_sample_seed")
        .map(|s| s.parse::<u64>().unwrap_or(0))
        .unwrap_or(0);
    let compute_graph_logical_effort = match matches
        .get_one::<String>("compute_graph_logical_effort")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => true, // default for compute_graph_logical_effort is true
    };
    let graph_logical_effort_beta1 = matches
        .get_one::<String>("graph_logical_effort_beta1")
        .map(|s| s.parse::<f64>().unwrap_or(1.0))
        .unwrap_or(1.0);
    let graph_logical_effort_beta2 = matches
        .get_one::<String>("graph_logical_effort_beta2")
        .map(|s| s.parse::<f64>().unwrap_or(0.0))
        .unwrap_or(0.0);
    let input_path = std::path::Path::new(input_file);

    ir2gates(
        input_path,
        quiet,
        fold,
        hash,
        fraig,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
    );
}
