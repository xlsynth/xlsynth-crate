// SPDX-License-Identifier: Apache-2.0
//! Stand-alone CLI to compare two serialized `.g8r` `GateFn`s for functional
//! equivalence.
//!
//! We invoke **three independent engines** and print the raw `{:?}` result for
//! each:
//!   • SAT/Z3 oracle – fast, semantic-level proof.
//!   • XLS IR equivalence checker – external `check_ir_equivalence_main`
//! binary.
//!   • Varisat structural SAT proof – also yields concrete counter-examples.
//!
//! Exit status:
//!   0 – All engines agree (either Equivalent or NOT Equivalent) – success.
//!   1 – Any disagreement between engines, inconclusive external checker, or
//!       I/O/parse failure – treated as error.

use std::fs;
use std::path::PathBuf;
use std::process::exit;

use anyhow::Context;
use clap::Parser;

use xlsynth_g8r::check_equivalence::{self, IrCheckResult};
use xlsynth_g8r::gate::AigRef;
use xlsynth_g8r::gate::GateFn;
use xlsynth_g8r::gate_sim::{self, Collect};
use xlsynth_g8r::mcmc_logic::oracle_equiv_sat;
use xlsynth_g8r::prove_gate_fn_equiv_varisat::{self, EquivResult};

/// Simple CLI to compare two GateFns.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// First `.g8r` file
    lhs: PathBuf,
    /// Second `.g8r` file
    rhs: PathBuf,
}

#[derive(Debug)]
enum SatResult {
    Equivalent,
    NotEquivalent,
}

fn load_gfn(p: &PathBuf) -> anyhow::Result<GateFn> {
    let bytes =
        fs::read(p).with_context(|| format!("failed to read GateFn file {}", p.display()))?;
    if p.extension().map(|e| e == "g8rbin").unwrap_or(false) {
        bincode::deserialize(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "failed to bincode-deserialize GateFn from {}: {}",
                p.display(),
                e
            )
        })
    } else {
        let txt = String::from_utf8(bytes)
            .map_err(|e| anyhow::anyhow!("failed to decode utf8 from {}: {}", p.display(), e))?;
        GateFn::from_str(&txt)
            .map_err(|e| anyhow::anyhow!("failed to parse GateFn from {}: {}", p.display(), e))
    }
}

fn main() {
    let cli = Cli::parse();

    let lhs = match load_gfn(&cli.lhs) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[equiv] error: {}", e);
            exit(1);
        }
    };
    let rhs = match load_gfn(&cli.rhs) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[equiv] error: {}", e);
            exit(1);
        }
    };

    // Checker 1: SAT/Z3 oracle (fast path)
    let sat_equiv_bool = oracle_equiv_sat(&lhs, &rhs);
    let sat_result = if sat_equiv_bool {
        SatResult::Equivalent
    } else {
        SatResult::NotEquivalent
    };
    println!("SAT/Z3 oracle result: {:?}", sat_result);

    // Checker 2: IR-based equivalence.
    let ir_status = check_equivalence::prove_same_gate_fn_via_ir_status(&lhs, &rhs);
    println!("IR checker result: {:?}", ir_status);
    let ir_equiv_opt: Option<bool> = match &ir_status {
        IrCheckResult::Equivalent => Some(true),
        IrCheckResult::NotEquivalent => Some(false),
        IrCheckResult::TimedOutOrInterrupted | IrCheckResult::OtherProcessError(_) => None,
    };

    // Checker 3: Varisat-based SAT prover (structural)
    let varisat_result = {
        let mut ctx = prove_gate_fn_equiv_varisat::Ctx::new();
        prove_gate_fn_equiv_varisat::prove_gate_fn_equiv(&lhs, &rhs, &mut ctx)
    };
    println!("Varisat checker result: {:?}", varisat_result);

    let varisat_equiv = match &varisat_result {
        EquivResult::Proved => true,
        EquivResult::Disproved(cex_inputs) => {
            println!(
                "Varisat checker produced counterexample ({} inputs):",
                cex_inputs.len()
            );
            for (idx, bits) in cex_inputs.iter().enumerate() {
                println!("  input{} = {}", idx, bits.to_string());
            }
            // Evaluate both GateFns on the counterexample so we can see the differing
            // outputs.
            let lhs_out = gate_sim::eval(&lhs, &cex_inputs, Collect::None);
            let rhs_out = gate_sim::eval(&rhs, &cex_inputs, Collect::None);
            println!(
                "  LHS outputs: {}",
                lhs_out
                    .outputs
                    .iter()
                    .map(|o| o.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!(
                "  RHS outputs: {}",
                rhs_out
                    .outputs
                    .iter()
                    .map(|o| o.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            // -- Extra debug: walk internal nodes to find first divergence.
            let lhs_all = gate_sim::eval(&lhs, &cex_inputs, Collect::All);
            let rhs_all = gate_sim::eval(&rhs, &cex_inputs, Collect::All);
            if let (Some(lhs_bits), Some(rhs_bits)) = (lhs_all.all_values, rhs_all.all_values) {
                for (idx, (l_val, r_val)) in lhs_bits.iter().zip(rhs_bits.iter()).enumerate() {
                    if l_val != r_val {
                        println!(
                            "  First differing internal node: %{} (lhs={}, rhs={})\n    LHS node: {:?}\n    RHS node: {:?}",
                            idx,
                            l_val,
                            r_val,
                            lhs.get(AigRef { id: idx }),
                            rhs.get(AigRef { id: idx })
                        );
                        break;
                    }
                }
            }
            false
        }
    };

    match (sat_result, ir_equiv_opt, varisat_equiv) {
        (SatResult::Equivalent, Some(true), true) => {
            println!("OK: all engines agree – equivalent.");
            exit(0);
        }
        (SatResult::NotEquivalent, Some(false), false) => {
            println!("OK: all engines agree – NOT equivalent.");
            exit(0);
        }
        (SatResult::Equivalent, None, true) => {
            eprintln!("ERROR: SAT & Varisat proved equivalence but IR checker was inconclusive.");
            exit(1);
        }
        _ => {
            eprintln!("ERROR: Disagreement between checkers.");
            exit(1);
        }
    }
}
