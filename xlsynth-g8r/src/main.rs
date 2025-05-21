// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use xlsynth_g8r::process_ir_path::{process_ir_path, Options};

/// Simple program to parse an XLS IR file and emit a Verilog netlist.
#[derive(Parser, Debug)]
struct Args {
    /// Whether to perform AIG folding optimizations
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    fold: bool,

    /// Whether to hash the AIG nodes.
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    hash: bool,

    /// Whether to run "fraiging" optimization.
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    fraig: bool,

    #[arg(long)]
    fraig_max_iterations: Option<usize>,

    #[arg(long)]
    fraig_sim_samples: Option<usize>,

    /// Whether to check equivalence between the IR and the gate function.
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    check_equivalence: bool,

    /// Whether to emit the netlist.
    #[arg(long, default_value_t = false)]
    #[arg(action = clap::ArgAction::Set)]
    emit_netlist: bool,

    /// Number of random input samples for toggle stats (0 disables)
    #[arg(long, default_value_t = 0)]
    toggle_sample_count: usize,

    /// Seed for random toggle stimulus (default 0)
    #[arg(long, default_value_t = 0)]
    toggle_sample_seed: u64,

    /// Whether to compute the graph logical effort worst case delay.
    #[arg(long, default_value_t = false)]
    #[arg(action = clap::ArgAction::Set)]
    compute_graph_logical_effort: bool,

    /// The beta1 value for the graph logical effort computation.
    #[arg(long, default_value_t = 1.0)]
    #[arg(action = clap::ArgAction::Set)]
    graph_logical_effort_beta1: f64,

    /// The beta2 value for the graph logical effort computation.
    #[arg(long, default_value_t = 0.0)]
    #[arg(action = clap::ArgAction::Set)]
    graph_logical_effort_beta2: f64,

    /// The path to the XLS IR file.
    input: String,
}

fn main() {
    let _ = env_logger::builder().try_init();
    let args = Args::parse();

    let options = Options {
        check_equivalence: args.check_equivalence,
        fold: args.fold,
        hash: args.hash,
        fraig: args.fraig,
        emit_netlist: args.emit_netlist,
        quiet: args.emit_netlist,
        toggle_sample_count: args.toggle_sample_count,
        toggle_sample_seed: args.toggle_sample_seed,
        compute_graph_logical_effort: args.compute_graph_logical_effort,
        graph_logical_effort_beta1: args.graph_logical_effort_beta1,
        graph_logical_effort_beta2: args.graph_logical_effort_beta2,
        fraig_max_iterations: args.fraig_max_iterations,
        fraig_sim_samples: args.fraig_sim_samples,
    };
    let input_path = std::path::Path::new(&args.input);
    process_ir_path(input_path, &options);
}
