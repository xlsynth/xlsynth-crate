// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use xlsynth_g8r::process_ir_path::process_ir_path;

/// Simple program to parse an XLS IR file and emit a Verilog netlist.
#[derive(Parser, Debug)]
struct Args {
    /// Whether to perform AIG folding optimizations
    #[arg(long, default_value = "true")]
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    fold: bool,

    /// Whether to check equivalence between the IR and the gate function.
    #[arg(long, default_value = "true")]
    #[arg(long, default_value_t = true)]
    #[arg(action = clap::ArgAction::Set)]
    check_equivalence: bool,

    /// The path to the XLS IR file.
    input: String,
}

fn main() {
    let _ = env_logger::builder().try_init();
    let args = Args::parse();

    let input_path = std::path::Path::new(&args.input);
    process_ir_path(input_path, args.check_equivalence, args.fold, false);
}
