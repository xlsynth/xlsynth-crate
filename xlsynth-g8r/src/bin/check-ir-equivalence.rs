// SPDX-License-Identifier: Apache-2.0

use clap::Parser;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::ir_equiv_boolector;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::ir_value_utils::ir_value_from_bits_with_type;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::xls_ir::ir::Package;

#[cfg(feature = "has-boolector")]
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

#[cfg(feature = "has-boolector")]
fn parse_package(path: &str) -> Package {
    let file_content =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    let mut parser = IrParser::new(&file_content);
    parser
        .parse_package()
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path, e))
}

#[cfg(feature = "has-boolector")]
fn main_has_boolector(args: Args) {
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

fn main() {
    let _ = env_logger::builder().try_init();
    log::info!("Starting check-ir-equivalence");

    #[cfg(feature = "has-boolector")]
    {
        let args = Args::parse();
        main_has_boolector(args);
    }

    #[cfg(not(feature = "has-boolector"))]
    {
        eprintln!("error: check-ir-equivalence requires --features=with-boolector-built or --features=with-boolector-system");
        std::process::exit(1);
    }
}
