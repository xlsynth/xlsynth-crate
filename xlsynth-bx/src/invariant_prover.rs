// SPDX-License-Identifier: Apache-2.0

//! Bounded checking and k-induction for clocked BX invariants.

use std::num::NonZeroUsize;

use xlsynth_pir::IrValue;
use xlsynth_pir::ir::Package;

use crate::BxInvariantObligation::{Base, LemmaStep, PropertyStep};
use crate::BxInvariantProof;

/// Selects a bounded prefix from reset or an unbounded proof by induction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvariantMode {
    Bmc,
    Induction,
}

/// Result of asking the solver whether the Boolean top function is universally
/// true.
pub enum InvariantQueryResult {
    Proved,
    Counterexample(Vec<IrValue>),
    Inconclusive(String),
}

/// An unbounded proof, bounded success, reachable failure, or an open
/// obligation.
#[derive(Debug, PartialEq, Eq)]
pub enum InvariantResult {
    Proved { depth: usize },
    Bounded { cycles: usize },
    Disproved { trace: String },
    Inconclusive { reason: String },
}

/// Checks initialized prefixes and establishes every helper before using it
/// to strengthen the main induction step. Only a base witness is reachable.
pub fn check_invariant(
    property: &BxInvariantProof,
    mode: InvariantMode,
    max_depth: NonZeroUsize,
    mut solve: impl FnMut(&Package) -> Result<InvariantQueryResult, String>,
) -> Result<InvariantResult, String> {
    let mut helpers_proved = !property.has_lemmas();
    let last = max_depth.get();
    let first = if mode == InvariantMode::Bmc { last } else { 1 };
    for depth in first..=last {
        for obligation in [Base, LemmaStep, PropertyStep] {
            if matches!(obligation, LemmaStep) && helpers_proved {
                continue;
            }
            let query = match property.query(obligation, depth) {
                Ok(query) => query,
                Err(reason) => return Ok(InvariantResult::Inconclusive { reason }),
            };
            match solve(&query)? {
                InvariantQueryResult::Proved => match obligation {
                    Base if mode == InvariantMode::Bmc => {
                        return Ok(InvariantResult::Bounded { cycles: depth });
                    }
                    Base => {
                        // The initialized prefix establishes the base for both
                        // steps.
                    }
                    LemmaStep => helpers_proved = true,
                    PropertyStep => return Ok(InvariantResult::Proved { depth }),
                },
                InvariantQueryResult::Counterexample(inputs) => {
                    if matches!(obligation, Base) {
                        let trace = property
                            .replay(depth, &inputs)
                            .map_err(|e| format!("counterexample replay failed: {e}"))?;
                        return Ok(InvariantResult::Disproved { trace });
                    }
                    // The step began in arbitrary state. Increase k and check
                    // the base again.
                    break;
                }
                InvariantQueryResult::Inconclusive(reason) => {
                    return Ok(InvariantResult::Inconclusive { reason });
                }
            }
        }
    }
    let open = if helpers_proved {
        "property"
    } else {
        "lemma group"
    };
    Ok(InvariantResult::Inconclusive {
        reason: format!("{open} induction step remains open at k={last}"),
    })
}
