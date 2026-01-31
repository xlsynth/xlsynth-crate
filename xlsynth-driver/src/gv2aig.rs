// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::get_summary_stats::AigStats;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::netlist::gv2aig::{convert_gv2aig_paths, Gv2AigOptions};

fn format_fanout_histogram(stats: &AigStats) -> String {
    let mut s = String::new();
    s.push('{');
    for (i, (fanout, count)) in stats.fanout_histogram.iter().enumerate() {
        if i != 0 {
            s.push(',');
        }
        s.push_str(&format!("{}:{}", fanout, count));
    }
    s.push('}');
    s
}

pub fn handle_gv2aig(matches: &clap::ArgMatches) {
    let netlist_path = matches.get_one::<String>("netlist").unwrap();
    let liberty_proto_path = matches.get_one::<String>("liberty_proto").unwrap();
    let aiger_out = matches.get_one::<String>("aiger_out").unwrap();

    let module_name: Option<String> = matches.get_one::<String>("module_name").cloned();

    let dff_cells_identity: HashSet<String> = matches
        .get_one::<String>("dff_cells")
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let dff_cell_formula: Option<String> = matches
        .get_one::<String>("dff_cell_formula")
        .map(|s| s.to_string());
    let dff_cell_invert_formula: Option<String> = matches
        .get_one::<String>("dff_cell_invert_formula")
        .map(|s| s.to_string());
    let collapse_sequential = matches
        .get_one::<bool>("collapse_sequential")
        .copied()
        .unwrap_or(true);

    let opts = Gv2AigOptions {
        module_name,
        dff_cells_identity,
        dff_cell_formula,
        dff_cell_invert_formula,
        collapse_sequential,
    };

    let gate_fn = match convert_gv2aig_paths(
        Path::new(netlist_path),
        Path::new(liberty_proto_path),
        &opts,
    ) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to convert netlist to AIG: {:#}", e);
            std::process::exit(1);
        }
    };

    let stats = get_aig_stats(&gate_fn);

    let is_binary_aig = Path::new(aiger_out)
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("aig"));

    if is_binary_aig {
        let bytes = match emit_aiger_binary(&gate_fn, true) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("Failed to emit binary AIGER: {}", e);
                std::process::exit(1);
            }
        };
        let mut f = match std::fs::File::create(aiger_out) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create aiger-out file '{}': {}", aiger_out, e);
                std::process::exit(1);
            }
        };
        if let Err(e) = f.write_all(&bytes) {
            eprintln!("Failed to write aiger-out file '{}': {}", aiger_out, e);
            std::process::exit(1);
        }
    } else {
        let aiger = match emit_aiger(&gate_fn, true) {
            Ok(aiger) => aiger,
            Err(e) => {
                eprintln!("Failed to emit ASCII AIGER: {}", e);
                std::process::exit(1);
            }
        };
        let mut f = match std::fs::File::create(aiger_out) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create aiger-out file '{}': {}", aiger_out, e);
                std::process::exit(1);
            }
        };
        if let Err(e) = f.write_all(aiger.as_bytes()) {
            eprintln!("Failed to write aiger-out file '{}': {}", aiger_out, e);
            std::process::exit(1);
        }
    }

    println!(
        "aig stats: and_nodes={} depth={} fanout_hist={}",
        stats.and_nodes,
        stats.max_depth,
        format_fanout_histogram(&stats)
    );
}
