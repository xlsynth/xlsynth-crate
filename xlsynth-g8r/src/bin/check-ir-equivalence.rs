// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "boolector")]

use clap::Parser;
use xlsynth_g8r::ir_equiv_boolector;
use xlsynth_g8r::ir_value_utils::ir_value_from_bits_with_type;
use xlsynth_g8r::xls_ir::ir::Package;
use xlsynth_g8r::xls_ir::ir_parser::Parser as IrParser;

/// Checks equivalence of two XLS IR functions by name.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the first IR file
    ir1: String,

    /// Path to the second IR file
    ir2: String,

    /// Name of the top function to check in both files
    #[arg(long)]
    top: String,
}

fn parse_package(path: &str) -> Package {
    let file_content =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    let mut parser = IrParser::new(&file_content);
    parser
        .parse_package()
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path, e))
}

fn main() {
    let _ = env_logger::builder().try_init();
    log::info!("Starting check-ir-equivalence");
    let args = Args::parse();

    let pkg1 = parse_package(&args.ir1);
    let pkg2 = parse_package(&args.ir2);

    let f1 = pkg1
        .get_fn(&args.top)
        .unwrap_or_else(|| panic!("function '{}' not found in {}", args.top, args.ir1));
    let f2 = pkg2
        .get_fn(&args.top)
        .unwrap_or_else(|| panic!("function '{}' not found in {}", args.top, args.ir2));

    match ir_equiv_boolector::check_equiv(f1, f2) {
        ir_equiv_boolector::EquivResult::Proved => {
            println!("Equivalence result: PROVED");
        }
        ir_equiv_boolector::EquivResult::Disproved(cex) => {
            println!("Equivalence result: DISPROVED");
            println!("Counterexample(s):");
            let values: Vec<_> = cex
                .iter()
                .zip(&f1.params)
                .map(|(bits, param)| ir_value_from_bits_with_type(bits, &param.ty))
                .collect();
            if values.len() == 1 {
                println!("  {}", values[0]);
            } else {
                println!("  {:?}", values);
            }
        }
    }
}
