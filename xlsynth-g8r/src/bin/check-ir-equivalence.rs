// SPDX-License-Identifier: Apache-2.0

// Use some pragmas since when the configuration does not all have engines
// enabled we would get a bunch of warnings.
#![allow(unused)]

use clap::Parser;
use std::time::Instant;
use xlsynth_g8r::check_equivalence::check_equivalence_with_top;
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_parser::Parser as IrParser;

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
        .parse_and_validate_package()
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path, e))
}

#[cfg(feature = "has-bitwuzla")]
fn main_has_bitwuzla(args: Args) {
    use xlsynth_prover::bitwuzla_backend::{Bitwuzla, BitwuzlaOptions};
    use xlsynth_prover::prove_equiv::prove_ir_fn_equiv;
    use xlsynth_prover::types::{AssertionSemantics, EquivResult, ProverFn};

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
    let bitwuzla_result = prove_ir_fn_equiv::<Bitwuzla>(
        &BitwuzlaOptions::new(),
        &ProverFn::new(f1, Some(&pkg1)),
        &ProverFn::new(f2, Some(&pkg2)),
        AssertionSemantics::Same,
        false,
    );
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
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } => {
            println!(
                "Equivalence result (bitwuzla): DISPROVED (took {:?})",
                bitwuzla_elapsed
            );
            println!("failure: Solver found counterexample:");
            println!("  inputs LHS: {:?}", lhs_inputs);
            println!("  inputs RHS: {:?}", rhs_inputs);
            // Report outputs for the counterexample
            println!("  output LHS: {:?}", lhs_output);
            println!("  output RHS: {:?}", rhs_output);
        }
        EquivResult::Error(msg) => {
            eprintln!("error: {}", msg);
            std::process::exit(2);
        }
        _ => {
            eprintln!("error: unexpected equivalence result");
            std::process::exit(2);
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

    #[cfg(feature = "has-bitwuzla")]
    {
        let args = Args::parse();
        main_has_bitwuzla(args);
    }
}
