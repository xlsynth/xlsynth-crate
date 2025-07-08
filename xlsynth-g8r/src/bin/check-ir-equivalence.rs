// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use std::time::Instant;
use xlsynth_g8r::check_equivalence::check_equivalence_with_top;
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

#[cfg(feature = "has-easy-smt")]
fn main_has_easy_smt(args: Args) {
    use xlsynth_g8r::equiv::easy_smt_backend::{EasySMTConfig, EasySMTSolver};
    use xlsynth_g8r::equiv::prove_equiv::{prove_ir_fn_equiv, EquivResult};

    let pkg1 = parse_package(&args.ir1);
    let pkg2 = parse_package(&args.ir2);

    let f1 = pkg1
        .get_fn(&args.top)
        .unwrap_or_else(|| panic!("function '{}' not found in {}", args.top, args.ir1));
    let f2 = pkg2
        .get_fn(&args.top)
        .unwrap_or_else(|| panic!("function '{}' not found in {}", args.top, args.ir2));

    // First run the Bitwuzla-based equivalence prover.
    let bitwuzla_start = Instant::now();
    let bitwuzla_result =
        prove_ir_fn_equiv::<EasySMTSolver>(&EasySMTConfig::bitwuzla(), f1, f2, false);
    let bitwuzla_elapsed = bitwuzla_start.elapsed();

    // Convert Bitwuzla result into a simple boolean for later comparison.
    let bitwuzla_equiv = matches!(&bitwuzla_result, EquivResult::Proved);

    match &bitwuzla_result {
        EquivResult::Proved => {
            println!(
                "Equivalence result (bitwuzla): PROVED (took {:?})",
                bitwuzla_elapsed
            );
        }
        EquivResult::Disproved {
            inputs: cex,
            outputs: (lhs_val, rhs_val),
        } => {
            println!(
                "Equivalence result (bitwuzla): DISPROVED (took {:?})",
                bitwuzla_elapsed
            );
            println!("Counterexample inputs:");
            let values: Vec<_> = cex.iter().cloned().collect();
            if values.len() == 1 {
                println!("  {}", values[0]);
            } else {
                println!("  {:?}", values);
            }
            // Report outputs for the counterexample
            println!("Output LHS: {}", lhs_val);
            println!("Output RHS: {}", rhs_val);
        }
    }

    // Now run the external `check_ir_equivalence_main` tool via the helper in
    // xlsynth_g8r.
    let pkg1_ir = pkg1.to_string();
    let pkg2_ir = pkg2.to_string();
    let external_start = Instant::now();
    let external_res = check_equivalence_with_top(&pkg1_ir, &pkg2_ir, Some(&args.top), false);
    let external_elapsed = external_start.elapsed();
    let ext_equiv_opt: Option<bool> = match &external_res {
        Ok(_) => {
            println!(
                "Equivalence result (external): PROVED (took {:?})",
                external_elapsed
            );
            Some(true)
        }
        Err(msg) => {
            if msg.to_lowercase().contains("not equivalent") {
                println!(
                    "Equivalence result (external): DISPROVED (took {:?})",
                    external_elapsed
                );
                Some(false)
            } else if msg.to_lowercase().contains("timedout")
                || msg.to_lowercase().contains("interrupted")
            {
                eprintln!(
                    "warning: external tool timed out or was interrupted after {:?}; skipping consistency check",
                    external_elapsed
                );
                None
            } else {
                eprintln!(
                    "warning: external tool error after {:?}: {msg}",
                    external_elapsed
                );
                None
            }
        }
    };

    // If we have a definitive answer from the external tool, ensure it matches
    // Bitwuzla.
    if let Some(ext_equiv) = ext_equiv_opt {
        if ext_equiv != bitwuzla_equiv {
            eprintln!("error: inconsistency detected between Bitwuzla prover and external tool");
            std::process::exit(2);
        }
    }

    // Exit with 0 when equivalent, 1 when non-equivalent.
    if bitwuzla_equiv {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}

fn main() {
    let _ = env_logger::builder().try_init();
    log::info!("Starting check-ir-equivalence");

    #[cfg(feature = "has-easy-smt")]
    {
        let args = Args::parse();
        main_has_easy_smt(args);
    }
}
