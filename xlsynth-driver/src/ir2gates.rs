// SPDX-License-Identifier: Apache-2.0

//! Accepts input IR and then performs the g8r IR-to-gates mapping on it.

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use std::fs::File;
use std::io::Write;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_serdes::emit_netlist;
use xlsynth_g8r::process_ir_path;

pub fn handle_ir2gates(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let ir_top = matches.get_one::<String>("ir_top").map(|s| s.as_str());
    let quiet = match matches.get_one::<String>("quiet").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };
    let emit_independent_op_stats = match matches
        .get_one::<String>("emit-independent-op-stats")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };
    let output_json = matches.get_one::<String>("output_json");
    let prepared_ir_out = matches.get_one::<String>("prepared_ir_out");

    let input_path = std::path::Path::new(input_file);
    let options = crate::g8r_cli::build_process_ir_path_options_for_cli(
        matches,
        quiet,
        /* emit_netlist= */ false,
        emit_independent_op_stats,
        ir_top,
        prepared_ir_out.map(|s| std::path::Path::new(s)),
    );
    let stats = process_ir_path::process_ir_path_for_cli(input_path, &options);
    if quiet {
        serde_json::to_writer(std::io::stdout(), &stats).unwrap();
        println!();
    }
    if let Some(path) = output_json.map(|s| std::path::Path::new(s)) {
        let file = File::create(path)
            .unwrap_or_else(|e| panic!("Failed to create {}: {}", path.display(), e));
        serde_json::to_writer_pretty(file, &stats)
            .unwrap_or_else(|e| panic!("Failed to write JSON: {}", e));
    }
}

pub fn handle_ir2g8r(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let ir_top = matches.get_one::<String>("ir_top").map(|s| s.as_str());
    let bin_out = matches.get_one::<String>("bin_out");
    let aiger_out = matches.get_one::<String>("aiger_out");
    let stats_out = matches.get_one::<String>("stats_out");
    let netlist_out = matches.get_one::<String>("netlist_out");
    let input_path = std::path::Path::new(input_file);
    let options = crate::g8r_cli::build_process_ir_path_options_for_cli(
        matches, /* quiet= */ true, /* emit_netlist= */ false,
        /* emit_independent_op_stats= */ false, ir_top, /* prepared_ir_out= */ None,
    );
    let (gate_fn, stats) = process_ir_path::process_ir_path_with_gatefn(input_path, &options);
    // Always print the GateFn to stdout
    println!("{}", gate_fn.to_string());
    // If --bin-out is given, write the GateFn as bincode
    if let Some(bin_path) = bin_out {
        let bin = bincode::serialize(&gate_fn).expect("Failed to serialize GateFn");
        let mut f = File::create(bin_path).expect("Failed to create bin_out file");
        f.write_all(&bin).expect("Failed to write bin_out file");
    }
    // If --aiger-out is given, write the GateFn as ASCII AIGER ("aag").
    if let Some(aiger_path) = aiger_out {
        let is_binary_aig = std::path::Path::new(aiger_path)
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("aig"))
            .unwrap_or(false);
        if is_binary_aig {
            let bytes = match emit_aiger_binary(&gate_fn, true) {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!("Failed to emit binary AIGER: {}", e);
                    std::process::exit(1);
                }
            };
            let mut f = File::create(aiger_path).expect("Failed to create aiger_out file");
            f.write_all(&bytes).expect("Failed to write aiger_out file");
        } else {
            let aiger = match emit_aiger(&gate_fn, true) {
                Ok(aiger) => aiger,
                Err(e) => {
                    eprintln!("Failed to emit ASCII AIGER: {}", e);
                    std::process::exit(1);
                }
            };
            let mut f = File::create(aiger_path).expect("Failed to create aiger_out file");
            f.write_all(aiger.as_bytes())
                .expect("Failed to write aiger_out file");
        }
    }
    // If --stats-out is given, write the stats as JSON
    if let Some(stats_path) = stats_out {
        let json = serde_json::to_string_pretty(&stats).expect("Failed to serialize stats to JSON");
        let mut f = File::create(stats_path).expect("Failed to create stats_out file");
        f.write_all(json.as_bytes())
            .expect("Failed to write stats_out file");
    }
    // If --netlist-out is given, write the gate-level netlist (human-readable)
    if let Some(netlist_path) = netlist_out {
        let netlist =
            match emit_netlist::emit_netlist(&gate_fn.name, &gate_fn, false, false, false, None) {
                Ok(netlist) => netlist,
                Err(e) => {
                    eprintln!("Failed to emit netlist: {}", e);
                    std::process::exit(1);
                }
            };
        let mut f = File::create(netlist_path).expect("Failed to create netlist_out file");
        f.write_all(netlist.as_bytes())
            .expect("Failed to write netlist_out file");
    }
}
