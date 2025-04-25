// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use xlsynth_g8r::process_ir_path::{process_ir_path, Options};

/// Simple program to parse an XLS IR file and emit a Verilog netlist.
#[derive(Parser, Debug)]
struct Args {
    /// Whether to perform AIG folding optimizations
    #[arg(long, default_value = "true")]
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    fold: bool,

    /// Whether to hash the AIG nodes.
    #[arg(long, default_value = "true")]
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    hash: bool,

    /// Whether to run "fraiging" optimization.
    #[arg(long, default_value = "true")]
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    fraig: bool,

    /// Whether to check equivalence between the IR and the gate function.
    #[arg(long, default_value = "true")]
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
    };
    let input_path = std::path::Path::new(&args.input);
    process_ir_path(input_path, &options);
}
