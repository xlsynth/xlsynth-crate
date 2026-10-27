// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::path::Path;

use clap::{Arg, ArgAction, ArgMatches, Command};
use xlsynth_g8r::aig::get_summary_stats::AigStats;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::{GateFn, SequentialGateFn};
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_serdes::g8r::{emit_g8r, encode_g8r_binary};
use xlsynth_g8r::netlist::gv2aig::{
    Gv2AigOptions, convert_gv2aig_paths_with_optional_boundary_request,
};
use xlsynth_g8r::netlist::gv2aig_boundaries::{
    Gv2AigBoundaryRequest, load_gv2aig_boundary_request,
};
use xlsynth_g8r::netlist::ugv2aig::{Ugv2AigOptions, convert_ugv2aig_paths};

use crate::common::write_stdout;

/// Defines the shared input, boundary, and collapsing options for GV
/// conversion.
fn gv_conversion_command(name: &'static str, about: &'static str) -> Command {
    Command::new(name)
        .about(about)
        .arg(
            Arg::new("netlist")
                .long("netlist")
                .help("Input GV cell netlist (.gv, .v, or gzip-compressed)")
                .required(true)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("liberty_proto")
                .long("liberty_proto")
                .help("Required Liberty proto (.proto, .textproto, or gzip-compressed); use ugv2aig for unmapped gate Verilog")
                .required(true)
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("module_name")
                .long("module_name")
                .value_name("MODULE")
                .help("Optional module name to select when netlist contains multiple modules")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("collapse_sequential")
                .long("collapse_sequential_cells")
                .alias("collapse_sequential")
                .value_name("BOOL")
                .default_value("false")
                .value_parser(clap::value_parser!(bool))
                .help("If true, collapse sequential state variables by substituting next_state.")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("collapse_load_enable_feedback")
                .long("collapse_load_enable_feedback")
                .help("Collapse recognized load-enable feedback loops, assuming prior Q is zero (disabled by default)")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("cone_boundary_file")
                .long("cone_boundary_file")
                .value_name("PATH")
                .help("Extract cones using compact source/sink selectors from a JSON file")
                .conflicts_with_all(["cone_sources", "cone_sinks"])
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("cone_sources")
                .long("cone_sources")
                .value_name("SELECTOR")
                .help("Add a compact source selector; repeat for multiple selectors")
                .requires("cone_sinks")
                .conflicts_with("cone_boundary_file")
                .action(ArgAction::Append),
        )
        .arg(
            Arg::new("cone_sinks")
                .long("cone_sinks")
                .value_name("SELECTOR")
                .help("Add a compact sink selector; repeat for multiple selectors")
                .conflicts_with("cone_boundary_file")
                .action(ArgAction::Append),
        )
}

/// Defines GV-to-AIGER conversion.
pub fn gv2aig_command() -> Command {
    gv_conversion_command(
        "gv2aig",
        "Converts GV cells and wiring to AIGER using Liberty",
    )
    .arg(
        Arg::new("aiger_out")
            .long("aiger-out")
            .value_name("PATH")
            .help("Path to write AIGER output; use .aag for ASCII or .aig for binary")
            .required(true)
            .action(ArgAction::Set),
    )
}

/// Defines GV-to-native-g8r conversion.
pub fn gv2g8r_command() -> Command {
    gv_conversion_command(
        "gv2g8r",
        "Converts GV cells and wiring to combinational native g8r text on stdout using Liberty",
    )
    .arg(
        Arg::new("bin_out")
            .long("bin-out")
            .value_name("PATH")
            .help("Optional path to also write native .g8rbin output")
            .action(ArgAction::Set),
    )
}

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

/// Applies the shared GV input, boundary, and collapsing options.
fn convert_gv(matches: &ArgMatches) -> Result<GateFn, anyhow::Error> {
    let netlist_path = matches.get_one::<String>("netlist").unwrap();
    let liberty_proto_path = matches.get_one::<String>("liberty_proto").unwrap();

    let module_name: Option<String> = matches.get_one::<String>("module_name").cloned();

    let collapse_sequential = matches
        .get_one::<bool>("collapse_sequential")
        .copied()
        .unwrap_or(false);

    let opts = Gv2AigOptions {
        module_name,
        collapse_sequential,
        collapse_load_enable_feedback: matches.get_flag("collapse_load_enable_feedback"),
    };

    let cone_request = if let Some(path) = matches.get_one::<String>("cone_boundary_file") {
        Some(load_gv2aig_boundary_request(Path::new(path)))
    } else {
        let sources = matches
            .get_many::<String>("cone_sources")
            .map(|values| values.cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let sinks = matches
            .get_many::<String>("cone_sinks")
            .map(|values| values.cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        if sources.is_empty() && sinks.is_empty() {
            None
        } else {
            Some(
                Gv2AigBoundaryRequest::from_selectors(&sources, &sinks).map_err(anyhow::Error::msg),
            )
        }
    };
    cone_request.transpose().and_then(|request| {
        convert_gv2aig_paths_with_optional_boundary_request(
            Path::new(netlist_path),
            Path::new(liberty_proto_path),
            &opts,
            request.as_ref(),
        )
        .map(|extraction| extraction.gate_fn)
    })
}

/// Converts GV with required Liberty cell definitions and writes AIGER.
pub fn handle_gv2aig(matches: &ArgMatches) {
    let gate_fn = match convert_gv(matches) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to convert GV to AIG: {:#}", e);
            std::process::exit(1);
        }
    };
    emit_aiger_output(&gate_fn, matches.get_one::<String>("aiger_out").unwrap());
}

/// Converts GV through the shared extractor and writes native g8r.
pub fn handle_gv2g8r(matches: &ArgMatches) -> Result<(), String> {
    let gate_fn = convert_gv(matches).map_err(|error| format!("{error:#}"))?;
    let design = SequentialGateFn::from_gate_fn(gate_fn);
    write_stdout(&emit_g8r(&design)?);
    if let Some(path) = matches.get_one::<String>("bin_out") {
        let bytes = encode_g8r_binary(&design)?;
        std::fs::write(path, bytes)
            .map_err(|error| format!("writing native g8r binary '{path}': {error}"))?;
    }
    Ok(())
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

    print_aig_stats(gate_fn);
}

/// Prints the structural summary for AIGER conversion.
fn print_aig_stats(gate_fn: &GateFn) {
    let stats = get_aig_stats(gate_fn);
    println!(
        "aig stats: and_nodes={} depth={} fanout_hist={}",
        stats.and_nodes,
        stats.max_depth,
        format_fanout_histogram(&stats)
    );
}
