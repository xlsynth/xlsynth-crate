// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::path::Path;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::get_summary_stats::AigStats;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::netlist::gv2aig::{Gv2AigOptions, convert_gv2aig_paths};
use xlsynth_g8r::netlist::ugv2aig::{Ugv2AigOptions, convert_ugv2aig_paths};

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

/// Converts GV with required Liberty cell definitions.
pub fn handle_gv2aig(matches: &clap::ArgMatches) {
    let netlist_path = matches.get_one::<String>("netlist").unwrap();
    let liberty_proto_path = matches.get_one::<String>("liberty_proto").unwrap();
    let aiger_out = matches.get_one::<String>("aiger_out").unwrap();

    let module_name: Option<String> = matches.get_one::<String>("module_name").cloned();

    let collapse_sequential = matches
        .get_one::<bool>("collapse_sequential")
        .copied()
        .unwrap_or(true);

    let opts = Gv2AigOptions {
        module_name,
        collapse_sequential,
        collapse_load_enable_feedback: matches.get_flag("collapse_load_enable_feedback"),
    };

    let gate_fn = match convert_gv2aig_paths(
        Path::new(netlist_path),
        Path::new(liberty_proto_path),
        &opts,
    ) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to convert GV to AIG: {:#}", e);
            std::process::exit(1);
        }
    };
    emit_aiger_output(&gate_fn, aiger_out);
}

/// Converts the supported combinational UGV subset.
pub fn handle_ugv2aig(matches: &clap::ArgMatches) {
    let netlist_path = matches.get_one::<String>("netlist").unwrap();
    let aiger_out = matches.get_one::<String>("aiger_out").unwrap();
    let opts = Ugv2AigOptions {
        module_name: matches.get_one::<String>("module_name").cloned(),
    };
    let gate_fn = match convert_ugv2aig_paths(Path::new(netlist_path), &opts) {
        Ok(gate_fn) => gate_fn,
        Err(error) => {
            eprintln!("Failed to convert UGV to AIG: {error:#}");
            std::process::exit(1);
        }
    };
    emit_aiger_output(&gate_fn, aiger_out);
}

/// Writes an AIGER result and its structural summary for either input format.
fn emit_aiger_output(gate_fn: &GateFn, aiger_out: &str) {
    let stats = get_aig_stats(gate_fn);

    let is_binary_aig = Path::new(aiger_out)
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("aig"));

    if is_binary_aig {
        let bytes = match emit_aiger_binary(gate_fn, true) {
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
        let aiger = match emit_aiger(gate_fn, true) {
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
