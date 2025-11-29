// SPDX-License-Identifier: Apache-2.0
//! Compare two `.g8r` GateFn descriptions for equivalence using all available
//! equivalence checkers (SAT/Z3 oracle + IR-based checker).
//!
//! Exit status:
//!   0 – all checkers agree the GateFns are equivalent
//!   1 – at least one checker reports non-equivalence or disagreement occurs
//!   2 – command-line / I/O / parse error

use std::fs;
use std::path::PathBuf;
use std::process::exit;

use anyhow::Context;
use clap::Parser;

use xlsynth_g8r::aig::GateFn;
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
        GateFn::try_from(txt.as_str())
            .map_err(|e| anyhow::anyhow!("failed to parse GateFn from {}: {}", p.display(), e))
    }
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

    // Checker 1: SAT/Z3 oracle (fast path)
    let sat_equiv = oracle_equiv_sat(&lhs, &rhs);
    println!("SAT/Z3 oracle: {}", sat_equiv);

    // Checker 2: IR-based equivalence.
    let ir_equiv = match xlsynth_g8r::check_equivalence::prove_same_gate_fn_via_ir(&lhs, &rhs) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("IR equivalence checker error: {}", e);
            false
        }
    };
    println!("IR checker: {}", ir_equiv);

    // Checker 3: Varisat-based SAT prover (structural)
    let varisat_equiv = {
        let mut ctx = prove_gate_fn_equiv_varisat::Ctx::new();
        matches!(
            prove_gate_fn_equiv_varisat::prove_gate_fn_equiv(&lhs, &rhs, &mut ctx),
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
