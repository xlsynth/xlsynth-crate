// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::check_equivalence::{self, IrCheckResult};
use xlsynth_g8r::prove_gate_fn_equiv_varisat::{self, EquivResult};
#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
use xlsynth_g8r::prove_gate_fn_equiv_z3;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::toolchain_config::ToolchainConfig;

#[derive(Serialize, Clone, PartialEq, Eq)]
#[serde(tag = "result", content = "counterexample")]
enum EngineResult {
    Equiv,
    NotEquiv(Option<String>),
}

impl EngineResult {
    fn is_equiv(&self) -> bool {
        matches!(self, EngineResult::Equiv)
    }
}

#[derive(Serialize)]
struct EquivReport {
    results: BTreeMap<String, EngineResult>,
    all_agree: bool,
}

fn load_gate_fn(path: &Path) -> Result<GateFn, String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    if path.extension().map(|e| e == "g8rbin").unwrap_or(false) {
        bincode::deserialize(&bytes).map_err(|e| {
            format!(
                "failed to deserialize GateFn from {}: {}",
                path.display(),
                e
            )
        })
    } else {
        let txt = String::from_utf8(bytes)
            .map_err(|e| format!("failed to decode utf8 from {}: {}", path.display(), e))?;
        GateFn::try_from(txt.as_str())
            .map_err(|e| format!("failed to parse GateFn from {}: {}", path.display(), e))
    }
}

pub fn handle_g8r_equiv(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let lhs_path = Path::new(matches.get_one::<String>("lhs_g8r_file").unwrap());
    let rhs_path = Path::new(matches.get_one::<String>("rhs_g8r_file").unwrap());

    let lhs = match load_gate_fn(lhs_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("g8r-equiv error: {}", e);
            std::process::exit(2);
        }
    };
    let rhs = match load_gate_fn(rhs_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("g8r-equiv error: {}", e);
            std::process::exit(2);
        }
    };

    let mut results: BTreeMap<String, EngineResult> = BTreeMap::new();

    #[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
    {
        let mut ctx = prove_gate_fn_equiv_z3::Ctx::new();
        let res = match prove_gate_fn_equiv_z3::prove_gate_fn_equiv(&lhs, &rhs, &mut ctx) {
            EquivResult::Proved => EngineResult::Equiv,
            EquivResult::Disproved(cex) => EngineResult::NotEquiv(Some(format!("{:?}", cex))),
        };
        results.insert("z3".to_string(), res);
    }

    let ir_checker = match check_equivalence::prove_same_gate_fn_via_ir_status(&lhs, &rhs) {
        IrCheckResult::Equivalent => EngineResult::Equiv,
        IrCheckResult::NotEquivalent => EngineResult::NotEquiv(None),
        IrCheckResult::TimedOutOrInterrupted => {
            EngineResult::NotEquiv(Some("TimedOutOrInterrupted".to_string()))
        }
        IrCheckResult::OtherProcessError(msg) => EngineResult::NotEquiv(Some(msg)),
    };
    results.insert("ir".to_string(), ir_checker.clone());

    let varisat = {
        let mut ctx = prove_gate_fn_equiv_varisat::Ctx::new();
        match prove_gate_fn_equiv_varisat::prove_gate_fn_equiv(&lhs, &rhs, &mut ctx) {
            EquivResult::Proved => EngineResult::Equiv,
            EquivResult::Disproved(cex) => EngineResult::NotEquiv(Some(format!("{:?}", cex))),
        }
    };
    results.insert("varisat".to_string(), varisat.clone());

    let all_agree = {
        let mut iter = results.values();
        if let Some(first) = iter.next() {
            iter.all(|r| r == first)
        } else {
            true
        }
    };
    let report = EquivReport {
        results: results.clone(),
        all_agree,
    };

    serde_json::to_writer(std::io::stdout(), &report).unwrap();
    println!();

    if results.values().any(|r| !r.is_equiv()) {
        std::process::exit(1);
    }
}
