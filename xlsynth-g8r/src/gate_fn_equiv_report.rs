// SPDX-License-Identifier: Apache-2.0

use crate::aig::GateFn;
use crate::check_equivalence::{self, IrCheckResult};
use crate::prove_gate_fn_equiv_varisat::{self, EquivResult};
#[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
use crate::prove_gate_fn_equiv_z3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "result", content = "counterexample")]
pub enum EngineResult {
    Equiv,
    NotEquiv(Option<String>),
}

impl EngineResult {
    pub fn is_equiv(&self) -> bool {
        matches!(self, EngineResult::Equiv)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct EquivReport {
    pub results: BTreeMap<String, EngineResult>,
    pub all_agree: bool,
}

pub fn prove_gate_fn_equiv_report(lhs: &GateFn, rhs: &GateFn) -> EquivReport {
    let mut results: BTreeMap<String, EngineResult> = BTreeMap::new();

    #[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
    {
        let mut ctx = prove_gate_fn_equiv_z3::Ctx::new();
        let res = match prove_gate_fn_equiv_z3::prove_gate_fn_equiv(lhs, rhs, &mut ctx) {
            EquivResult::Proved => EngineResult::Equiv,
            EquivResult::Disproved(cex) => EngineResult::NotEquiv(Some(format!("{:?}", cex))),
        };
        results.insert("z3".to_string(), res);
    }

    let ir_checker = match check_equivalence::prove_same_gate_fn_via_ir_status(lhs, rhs) {
        IrCheckResult::Equivalent => EngineResult::Equiv,
        IrCheckResult::NotEquivalent => EngineResult::NotEquiv(None),
        IrCheckResult::TimedOutOrInterrupted => {
            EngineResult::NotEquiv(Some("TimedOutOrInterrupted".to_string()))
        }
        IrCheckResult::OtherProcessError(msg) => EngineResult::NotEquiv(Some(msg)),
    };
    results.insert("ir".to_string(), ir_checker);

    let varisat = {
        let mut ctx = prove_gate_fn_equiv_varisat::Ctx::new();
        match prove_gate_fn_equiv_varisat::prove_gate_fn_equiv(lhs, rhs, &mut ctx) {
            EquivResult::Proved => EngineResult::Equiv,
            EquivResult::Disproved(cex) => EngineResult::NotEquiv(Some(format!("{:?}", cex))),
        }
    };
    results.insert("varisat".to_string(), varisat);

    let all_agree = {
        let mut iter = results.values();
        if let Some(first) = iter.next() {
            iter.all(|r| r == first)
        } else {
            true
        }
    };

    EquivReport { results, all_agree }
}
