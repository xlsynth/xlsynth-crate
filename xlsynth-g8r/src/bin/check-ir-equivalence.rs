// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
#[cfg(feature = "has-boolector")]
use std::time::Instant;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::ir_equiv_boolector;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::xls_ir::ir::Package;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::xls_ir::ir_parser::Parser as IrParser;

#[cfg(feature = "has-boolector")]
use xlsynth_g8r::check_equivalence::check_equivalence_with_top;

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

    // First run the Boolector-based equivalence prover.
    let boolector_start = Instant::now();
    let boolector_result = ir_equiv_boolector::prove_ir_fn_equiv(f1, f2, false);
    let boolector_elapsed = boolector_start.elapsed();

    // Convert Boolector result into a simple boolean for later comparison.
    let boolector_equiv = matches!(&boolector_result, ir_equiv_boolector::EquivResult::Proved);

    match &boolector_result {
        ir_equiv_boolector::EquivResult::Proved => {
            println!(
                "Equivalence result (boolector): PROVED (took {:?})",
                boolector_elapsed
            );
        }
        ir_equiv_boolector::EquivResult::Disproved {
            inputs: cex,
            outputs: (lhs_val, rhs_val),
        } => {
            println!(
                "Equivalence result (boolector): DISPROVED (took {:?})",
                boolector_elapsed
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
    // Boolector.
    if let Some(ext_equiv) = ext_equiv_opt {
        if ext_equiv != boolector_equiv {
            eprintln!("error: inconsistency detected between Boolector prover and external tool");
            std::process::exit(2);
        }
    }

    // Exit with 0 when equivalent, 1 when non-equivalent.
    if boolector_equiv {
        std::process::exit(0);
    } else {
        std::process::exit(1);
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
        eprintln!(
            "error: check-ir-equivalence requires --features=with-boolector-built or --features=with-boolector-system"
        );
        std::process::exit(1);
    }
}
