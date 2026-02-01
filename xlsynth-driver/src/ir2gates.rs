// SPDX-License-Identifier: Apache-2.0

//! Accepts input IR and then performs the g8r IR-to-gates mapping on it.

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use std::fs::File;
use std::io::Write;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::emit_aiger_binary::emit_aiger_binary;
use xlsynth_g8r::aig_serdes::emit_netlist;
use xlsynth_g8r::cut_db::loader::CutDb;
use xlsynth_g8r::cut_db_cli_defaults::{
    CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI, CUT_DB_REWRITE_MAX_ITERATIONS_CLI,
};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::process_ir_path;

#[derive(Debug, Clone, Copy)]
struct CutDbRewriteBounds {
    max_cuts_per_node: usize,
    max_iterations: usize,
}

fn parse_adder_mapping(value: Option<&str>) -> AdderMapping {
    match value {
        Some("ripple-carry") => AdderMapping::RippleCarry,
        Some("brent-kung") => AdderMapping::BrentKung,
        Some("kogge-stone") => AdderMapping::KoggeStone,
        _ => AdderMapping::default(),
    }
}

fn parse_adder_mappings(matches: &ArgMatches) -> (AdderMapping, Option<AdderMapping>) {
    let adder_mapping = parse_adder_mapping(
        matches
            .get_one::<String>("adder_mapping")
            .map(|s| s.as_str()),
    );
    let mul_adder_mapping = match matches
        .get_one::<String>("mul_adder_mapping")
        .map(|s| s.as_str())
    {
        Some("ripple-carry") => Some(AdderMapping::RippleCarry),
        Some("brent-kung") => Some(AdderMapping::BrentKung),
        Some("kogge-stone") => Some(AdderMapping::KoggeStone),
        _ => None,
    };
    (adder_mapping, mul_adder_mapping)
}

fn ir2gates(
    input_file: &std::path::Path,
    ir_top: Option<&str>,
    quiet: bool,
    fold: bool,
    hash: bool,
    enable_rewrite_carry_out: bool,
    adder_mapping: AdderMapping,
    mul_adder_mapping: Option<AdderMapping>,
    fraig: bool,
    emit_independent_op_stats: bool,
    toggle_sample_count: usize,
    toggle_sample_seed: u64,
    compute_graph_logical_effort: bool,
    graph_logical_effort_beta1: f64,
    graph_logical_effort_beta2: f64,
    fraig_max_iterations: Option<usize>,
    fraig_sim_samples: Option<usize>,
    output_json: Option<&std::path::Path>,
    prepared_ir_out: Option<&std::path::Path>,
    cut_db: Option<std::sync::Arc<CutDb>>,
) {
    log::info!("ir2gates");
    let options = process_ir_path::Options {
        check_equivalence: false,
        fold,
        hash,
        enable_rewrite_carry_out,
        adder_mapping,
        mul_adder_mapping,
        fraig,
        emit_independent_op_stats,
        quiet,
        emit_netlist: false,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
        fraig_max_iterations,
        fraig_sim_samples,
        cut_db,
        // Keep CLI runtime predictable in tests/CI: small bounded rewrite.
        cut_db_rewrite_max_iterations: CUT_DB_REWRITE_MAX_ITERATIONS_CLI,
        cut_db_rewrite_max_cuts_per_node: CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI,
        prepared_ir_out: prepared_ir_out.map(|p| p.to_path_buf()),
        ir_top: ir_top.map(|s| s.to_string()),
    };
    let stats = process_ir_path::process_ir_path_for_cli(input_file, &options);
    if quiet {
        serde_json::to_writer(std::io::stdout(), &stats).unwrap();
        println!();
    }
    if let Some(path) = output_json {
        let file = File::create(path)
            .unwrap_or_else(|e| panic!("Failed to create {}: {}", path.display(), e));
        serde_json::to_writer_pretty(file, &stats)
            .unwrap_or_else(|e| panic!("Failed to write JSON: {}", e));
    }
}

pub fn handle_ir2gates(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let ir_top = matches.get_one::<String>("ir_top").map(|s| s.as_str());
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
    let (adder_mapping, mul_adder_mapping) = parse_adder_mappings(matches);
    let fraig = match matches.get_one::<String>("fraig").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => true, // default for fraig is true
    };
    let emit_independent_op_stats = match matches
        .get_one::<String>("emit-independent-op-stats")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false,
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
    let enable_rewrite_carry_out = match matches
        .get_one::<String>("enable-rewrite-carry-out")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false, // default is false
    };

    let graph_logical_effort_beta1 = matches
        .get_one::<String>("graph_logical_effort_beta1")
        .map(|s| s.parse::<f64>().unwrap_or(1.0))
        .unwrap_or(1.0);
    let graph_logical_effort_beta2 = matches
        .get_one::<String>("graph_logical_effort_beta2")
        .map(|s| s.parse::<f64>().unwrap_or(0.0))
        .unwrap_or(0.0);

    let fraig_max_iterations = match matches
        .get_one::<String>("fraig_max_iterations")
        .map(|s| s.parse::<usize>())
    {
        // The user can provide a max number of iterations.
        Some(Ok(n)) => Some(n),
        // If we can't parse the user-provided value, print an error and exit.
        Some(Err(_)) => {
            eprintln!(
                "Invalid --fraig-max-iterations: {:?}",
                matches.get_one::<String>("fraig_max_iterations").unwrap()
            );
            std::process::exit(1);
        }
        // If the user didn't provide a max number of iterations, don't use one.
        None => None,
    };
    let fraig_sim_samples_flag_value = matches.get_one::<String>("fraig_sim_samples");
    log::info!(
        "fraig_sim_samples_flag_value: {:?}",
        fraig_sim_samples_flag_value
    );
    let fraig_sim_samples_parsed = fraig_sim_samples_flag_value.map(|s| s.parse::<usize>());
    log::info!("fraig_sim_samples_parsed: {:?}", fraig_sim_samples_parsed);
    let fraig_sim_samples: Option<usize> = match fraig_sim_samples_parsed {
        // The user can provide a number of gatesim samples to use for fraig equivalence class
        // proposal.
        Some(Ok(n)) => Some(n),
        // If we can't parse the user-provided value, print an error and exit.
        Some(Err(_)) => {
            eprintln!(
                "Invalid --fraig-sim-samples: {:?}",
                matches.get_one::<String>("fraig_sim_samples").unwrap()
            );
            std::process::exit(1);
        }
        // If the user didn't provide a number of gatesim samples, use the default.
        None => None,
    };

    let output_json = matches.get_one::<String>("output_json");
    let prepared_ir_out = matches.get_one::<String>("prepared_ir_out");
    let cut_db = Some(CutDb::load_default());

    let input_path = std::path::Path::new(input_file);
    ir2gates(
        input_path,
        ir_top,
        quiet,
        fold,
        hash,
        enable_rewrite_carry_out,
        adder_mapping,
        mul_adder_mapping,
        fraig,
        emit_independent_op_stats,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
        fraig_max_iterations,
        fraig_sim_samples,
        output_json.map(|s| std::path::Path::new(s)),
        prepared_ir_out.map(|s| std::path::Path::new(s)),
        cut_db,
    );
}

fn ir_to_gatefn_with_stats(
    input_file: &std::path::Path,
    ir_top: Option<&str>,
    fold: bool,
    hash: bool,
    enable_rewrite_carry_out: bool,
    adder_mapping: AdderMapping,
    mul_adder_mapping: Option<AdderMapping>,
    fraig: bool,
    toggle_sample_count: usize,
    toggle_sample_seed: u64,
    compute_graph_logical_effort: bool,
    graph_logical_effort_beta1: f64,
    graph_logical_effort_beta2: f64,
    fraig_max_iterations: Option<usize>,
    fraig_sim_samples: Option<usize>,
    cut_db: Option<std::sync::Arc<xlsynth_g8r::cut_db::loader::CutDb>>,
    cut_db_rewrite_bounds: CutDbRewriteBounds,
) -> (
    xlsynth_g8r::aig::GateFn,
    process_ir_path::Ir2GatesSummaryStats,
) {
    let options = process_ir_path::Options {
        check_equivalence: false,
        fold,
        hash,
        enable_rewrite_carry_out,
        adder_mapping,
        mul_adder_mapping,
        fraig,
        emit_independent_op_stats: false,
        quiet: true,
        emit_netlist: false,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
        fraig_max_iterations,
        fraig_sim_samples,
        cut_db,
        cut_db_rewrite_max_iterations: cut_db_rewrite_bounds.max_iterations,
        cut_db_rewrite_max_cuts_per_node: cut_db_rewrite_bounds.max_cuts_per_node,
        prepared_ir_out: None,
        ir_top: ir_top.map(|s| s.to_string()),
    };
    process_ir_path::process_ir_path_with_gatefn(input_file, &options)
}

pub fn handle_ir2g8r(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let ir_top = matches.get_one::<String>("ir_top").map(|s| s.as_str());
    let fold = match matches.get_one::<String>("fold").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => true,
    };
    let hash = match matches.get_one::<String>("hash").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => true,
    };
    let fraig = match matches.get_one::<String>("fraig").map(|s| s.as_str()) {
        Some("true") => true,
        Some("false") => false,
        _ => true,
    };
    let enable_rewrite_carry_out = match matches
        .get_one::<String>("enable-rewrite-carry-out")
        .map(|s| s.as_str())
    {
        Some("true") => true,
        Some("false") => false,
        _ => false,
    };
    let (adder_mapping, mul_adder_mapping) = parse_adder_mappings(matches);
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
        _ => true,
    };
    let graph_logical_effort_beta1 = matches
        .get_one::<String>("graph_logical_effort_beta1")
        .map(|s| s.parse::<f64>().unwrap_or(1.0))
        .unwrap_or(1.0);
    let graph_logical_effort_beta2 = matches
        .get_one::<String>("graph_logical_effort_beta2")
        .map(|s| s.parse::<f64>().unwrap_or(0.0))
        .unwrap_or(0.0);
    let fraig_max_iterations = match matches
        .get_one::<String>("fraig_max_iterations")
        .map(|s| s.parse::<usize>())
    {
        Some(Ok(n)) => Some(n),
        Some(Err(_)) => {
            eprintln!(
                "Invalid --fraig-max-iterations: {:?}",
                matches.get_one::<String>("fraig_max_iterations").unwrap()
            );
            std::process::exit(1);
        }
        None => None,
    };
    let fraig_sim_samples_flag_value = matches.get_one::<String>("fraig_sim_samples");
    let fraig_sim_samples_parsed = fraig_sim_samples_flag_value.map(|s| s.parse::<usize>());
    let fraig_sim_samples: Option<usize> = match fraig_sim_samples_parsed {
        Some(Ok(n)) => Some(n),
        Some(Err(_)) => {
            eprintln!(
                "Invalid --fraig-sim-samples: {:?}",
                matches.get_one::<String>("fraig_sim_samples").unwrap()
            );
            std::process::exit(1);
        }
        None => None,
    };
    let bin_out = matches.get_one::<String>("bin_out");
    let aiger_out = matches.get_one::<String>("aiger_out");
    let stats_out = matches.get_one::<String>("stats_out");
    let netlist_out = matches.get_one::<String>("netlist_out");
    let cut_db = Some(CutDb::load_default());
    let cut_db_rewrite_bounds = CutDbRewriteBounds {
        max_cuts_per_node: CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI,
        max_iterations: CUT_DB_REWRITE_MAX_ITERATIONS_CLI,
    };
    let input_path = std::path::Path::new(input_file);
    let (gate_fn, stats) = ir_to_gatefn_with_stats(
        input_path,
        ir_top,
        fold,
        hash,
        enable_rewrite_carry_out,
        adder_mapping,
        mul_adder_mapping,
        fraig,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
        fraig_max_iterations,
        fraig_sim_samples,
        cut_db,
        cut_db_rewrite_bounds,
    );
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
