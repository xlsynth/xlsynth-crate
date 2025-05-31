// SPDX-License-Identifier: Apache-2.0

//! Accepts input IR and then performs the g8r IR-to-gates mapping on it.

use clap::ArgMatches;
use rand_xoshiro::rand_core::SeedableRng;

use crate::toolchain_config::ToolchainConfig;
use std::fs::File;
use std::io::Write;
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
    fraig_max_iterations: Option<usize>,
    fraig_sim_samples: Option<usize>,
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
        fraig_max_iterations,
        fraig_sim_samples,
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
        fraig_max_iterations,
        fraig_sim_samples,
    );
}

fn ir_to_gatefn_with_stats(
    input_file: &std::path::Path,
    fold: bool,
    hash: bool,
    fraig: bool,
    toggle_sample_count: usize,
    toggle_sample_seed: u64,
    compute_graph_logical_effort: bool,
    graph_logical_effort_beta1: f64,
    graph_logical_effort_beta2: f64,
    fraig_max_iterations: Option<usize>,
    fraig_sim_samples: Option<usize>,
) -> (
    xlsynth_g8r::gate::GateFn,
    process_ir_path::Ir2GatesSummaryStats,
) {
    let options = process_ir_path::Options {
        check_equivalence: false,
        fold,
        hash,
        fraig,
        quiet: true, // always quiet for stats-only
        emit_netlist: false,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
        fraig_max_iterations,
        fraig_sim_samples,
    };
    // This is a bit hacky: process_ir_path returns stats, but we want the GateFn.
    // So we re-run the core logic here, similar to process_ir_path, but return
    // GateFn. For now, call process_ir_path and re-gatify to get the GateFn.
    // TODO: Refactor process_ir_path to return GateFn and stats together.
    // For now, duplicate the logic as needed.
    //
    // Actually, process_ir_path already does the full pipeline, so we can copy its
    // logic here. But for now, call process_ir_path for stats, and re-gatify
    // for GateFn.
    //
    // Read the file into a string.
    let file_content = std::fs::read_to_string(&input_file)
        .unwrap_or_else(|err| panic!("Failed to read {}: {}", input_file.display(), err));
    let mut parser = xlsynth_g8r::xls_ir::ir_parser::Parser::new(&file_content);
    let ir_package = parser.parse_package().unwrap_or_else(|err| {
        eprintln!("Error encountered parsing XLS IR package: {:?}", err);
        std::process::exit(1);
    });
    let ir_top = match ir_package.get_top() {
        Some(ir_top) => ir_top,
        None => {
            eprintln!("No top module found in the IR package");
            std::process::exit(1);
        }
    };
    let gatify_output = xlsynth_g8r::ir2gate::gatify(
        &ir_top,
        xlsynth_g8r::ir2gate::GatifyOptions {
            fold,
            hash,
            check_equivalence: false,
        },
    )
    .unwrap();
    let mut gate_fn = gatify_output.gate_fn;
    // Apply fraig if requested
    if fraig {
        let iteration_bounds = if let Some(max_iterations) = fraig_max_iterations {
            xlsynth_g8r::fraig::IterationBounds::MaxIterations(max_iterations)
        } else {
            xlsynth_g8r::fraig::IterationBounds::ToConvergence
        };
        let sim_samples = match fraig_sim_samples {
            Some(n) => n,
            None => {
                let gate_count = gate_fn.gates.len();
                let scaled = (gate_count as f64 / 8.0).ceil() as usize;
                let result = ((scaled + 255) / 256) * 256;
                result
            }
        };
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);
        let fraig_result =
            xlsynth_g8r::fraig::fraig_optimize(&gate_fn, sim_samples, iteration_bounds, &mut rng);
        match fraig_result {
            Ok((optimized_fn, _did_converge, _iteration_stats)) => {
                gate_fn = optimized_fn;
            }
            Err(e) => {
                eprintln!("Fraig optimization failed: {}", e);
                std::process::exit(1);
            }
        }
    }
    // Get stats (re-run process_ir_path for now, since stats are not exposed
    // directly)
    let stats = process_ir_path::process_ir_path(input_file, &options);
    (gate_fn, stats)
}

pub fn handle_ir2g8r(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
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
    let stats_out = matches.get_one::<String>("stats_out");
    let input_path = std::path::Path::new(input_file);
    let (gate_fn, stats) = ir_to_gatefn_with_stats(
        input_path,
        fold,
        hash,
        fraig,
        toggle_sample_count,
        toggle_sample_seed,
        compute_graph_logical_effort,
        graph_logical_effort_beta1,
        graph_logical_effort_beta2,
        fraig_max_iterations,
        fraig_sim_samples,
    );
    // Always print the GateFn to stdout
    println!("{}", gate_fn.to_string());
    // If --bin-out is given, write the GateFn as bincode
    if let Some(bin_path) = bin_out {
        let bin = bincode::serialize(&gate_fn).expect("Failed to serialize GateFn");
        let mut f = File::create(bin_path).expect("Failed to create bin_out file");
        f.write_all(&bin).expect("Failed to write bin_out file");
    }
    // If --stats-out is given, write the stats as JSON
    if let Some(stats_path) = stats_out {
        let json = serde_json::to_string_pretty(&stats).expect("Failed to serialize stats to JSON");
        let mut f = File::create(stats_path).expect("Failed to create stats_out file");
        f.write_all(json.as_bytes())
            .expect("Failed to write stats_out file");
    }
}
