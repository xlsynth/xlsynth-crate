// SPDX-License-Identifier: Apache-2.0
//! Compare two `.g8r` GateFn descriptions for equivalence using all available
//! equivalence checkers (CaDiCaL, Varisat, and an XLS IR-based checker).
//!
//! Exit status:
//!   0 – all checkers agree the GateFns are equivalent
//!   1 – at least one checker reports non-equivalence or disagreement occurs
//!   2 – command-line / I/O / parse error

use std::path::PathBuf;
use std::process::exit;

use clap::Parser;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::g8r::load_gate_fn_from_path;
use xlsynth_g8r::mcmc_logic::oracle_equiv_sat;
use xlsynth_g8r::prove_gate_fn_equiv_sat::{self, EquivResult};

/// Simple CLI to compare two GateFns.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// First `.g8r` file
    lhs: PathBuf,
    /// Second `.g8r` file
    rhs: PathBuf,
}

fn load_gfn(p: &PathBuf) -> anyhow::Result<GateFn> {
    load_gate_fn_from_path(p).map_err(anyhow::Error::msg)
}

fn main() {
    let cli = Cli::parse();

    let lhs = match load_gfn(&cli.lhs) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[equiv] error: {}", e);
            exit(2);
        }
    };
    let rhs = match load_gfn(&cli.rhs) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[equiv] error: {}", e);
            exit(2);
        }
    };

    // Checker 1: CaDiCaL gate-level oracle (fast path)
    let sat_equiv = match oracle_equiv_sat(&lhs, &rhs) {
        Ok(equiv) => equiv,
        Err(e) => {
            eprintln!("CaDiCaL oracle error: {}", e);
            exit(2);
        }
    };
    println!("CaDiCaL oracle: {}", sat_equiv);

    // Checker 2: IR-based equivalence.
    let ir_equiv =
        match xlsynth_g8r::check_equivalence::prove_same_gate_fn_via_ir_via_toolchain(&lhs, &rhs) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("IR equivalence checker error: {}", e);
                false
            }
        };
    println!("IR checker: {}", ir_equiv);

    // Checker 3: Varisat-based SAT prover (structural)
    let varisat_equiv = {
        let mut ctx = prove_gate_fn_equiv_sat::VarisatCtx::new();
        matches!(
            prove_gate_fn_equiv_sat::prove_gate_fn_equiv_varisat(&lhs, &rhs, &mut ctx),
            EquivResult::Proved
        )
    };
    println!("Varisat checker: {}", varisat_equiv);

    if sat_equiv && ir_equiv && varisat_equiv {
        println!("All checkers agree – equivalent.");
        exit(0);
    } else {
        eprintln!("Disagreement or non-equivalence detected.");
        exit(1);
    }
}
