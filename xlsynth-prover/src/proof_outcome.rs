// SPDX-License-Identifier: Apache-2.0

//! Typed proof conclusions shared by solver results and subprocess reports.

use serde::{Deserialize, Serialize};

use crate::prover::types::{BoolPropertyResult, EquivResult};

/// A proof conclusion, distinct from whether the prover executed successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProofStatus {
    Proved,
    Disproved,
    Inconclusive,
    ExecutionError,
}

impl ProofStatus {
    /// Returns whether the requested property was proved.
    pub fn is_proved(self) -> bool {
        self == Self::Proved
    }

    /// Returns the CLI exit code; JSON preserves the distinct failure kinds.
    pub fn exit_code(self) -> i32 {
        if self.is_proved() { 0 } else { 1 }
    }
}

impl From<&EquivResult> for ProofStatus {
    fn from(result: &EquivResult) -> Self {
        match result {
            EquivResult::Proved => Self::Proved,
            EquivResult::Disproved { .. } => Self::Disproved,
            EquivResult::Inconclusive(_) => Self::Inconclusive,
            // The legacy external-tool adapter puts arbitrary unsuccessful
            // process exits in this variant, including crashes. It does not
            // establish a counterexample and cannot justify a disproof.
            EquivResult::ToolchainDisproved(_) | EquivResult::Error(_) => Self::ExecutionError,
        }
    }
}

impl From<&BoolPropertyResult> for ProofStatus {
    fn from(result: &BoolPropertyResult) -> Self {
        match result {
            BoolPropertyResult::Proved => Self::Proved,
            BoolPropertyResult::Disproved { .. } => Self::Disproved,
            BoolPropertyResult::Inconclusive(_) => Self::Inconclusive,
            // This legacy variant also includes unsupported operations and
            // failed subprocess execution, not just counterexamples.
            BoolPropertyResult::ToolchainDisproved(_) | BoolPropertyResult::Error(_) => {
                Self::ExecutionError
            }
        }
    }
}

/// Combines a nonempty suite of required properties, preserving actual
/// disproofs.
///
/// A counterexample disproves the suite even if another property could not run.
/// Without a counterexample, execution errors dominate inconclusive results.
/// An empty suite is inconclusive because no requested property was checked.
pub fn aggregate_proof_statuses(statuses: impl IntoIterator<Item = ProofStatus>) -> ProofStatus {
    let mut aggregate = ProofStatus::Proved;
    let mut any = false;
    for status in statuses {
        any = true;
        aggregate = match (aggregate, status) {
            (ProofStatus::Disproved, _) | (_, ProofStatus::Disproved) => ProofStatus::Disproved,
            (ProofStatus::ExecutionError, _) | (_, ProofStatus::ExecutionError) => {
                ProofStatus::ExecutionError
            }
            (ProofStatus::Inconclusive, _) | (_, ProofStatus::Inconclusive) => {
                ProofStatus::Inconclusive
            }
            (ProofStatus::Proved, ProofStatus::Proved) => ProofStatus::Proved,
        };
    }
    if any {
        aggregate
    } else {
        ProofStatus::Inconclusive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::types::FnOutput;
    use xlsynth_pir::IrValue;

    #[test]
    fn equivalence_results_preserve_proof_status() {
        let output = FnOutput {
            value: IrValue::make_ubits(1, 0).unwrap(),
            assertion_violation: None,
        };
        for (result, expected) in [
            (EquivResult::Proved, ProofStatus::Proved),
            (
                EquivResult::Inconclusive("unknown".into()),
                ProofStatus::Inconclusive,
            ),
            (
                EquivResult::Disproved {
                    lhs_inputs: vec![],
                    rhs_inputs: vec![],
                    lhs_output: output.clone(),
                    rhs_output: output,
                },
                ProofStatus::Disproved,
            ),
            (
                EquivResult::ToolchainDisproved("tool crashed".into()),
                ProofStatus::ExecutionError,
            ),
            (
                EquivResult::Error("solver unavailable".into()),
                ProofStatus::ExecutionError,
            ),
        ] {
            assert_eq!(ProofStatus::from(&result), expected);
        }
    }

    #[test]
    fn boolean_results_preserve_proof_status() {
        for (result, expected) in [
            (BoolPropertyResult::Proved, ProofStatus::Proved),
            (
                BoolPropertyResult::Inconclusive("unknown".into()),
                ProofStatus::Inconclusive,
            ),
            (
                BoolPropertyResult::Disproved {
                    inputs: vec![],
                    output: FnOutput {
                        value: IrValue::make_ubits(1, 0).unwrap(),
                        assertion_violation: None,
                    },
                },
                ProofStatus::Disproved,
            ),
            (
                BoolPropertyResult::ToolchainDisproved("tool crashed".into()),
                ProofStatus::ExecutionError,
            ),
            (
                BoolPropertyResult::Error("solver unavailable".into()),
                ProofStatus::ExecutionError,
            ),
        ] {
            assert_eq!(ProofStatus::from(&result), expected);
        }
    }

    #[test]
    fn suite_status_is_order_independent_and_requires_every_proof() {
        let statuses = [
            ProofStatus::Proved,
            ProofStatus::Inconclusive,
            ProofStatus::ExecutionError,
            ProofStatus::Disproved,
        ];
        for (i, lhs) in statuses.iter().enumerate() {
            for (j, rhs) in statuses.iter().enumerate() {
                assert_eq!(aggregate_proof_statuses([*lhs, *rhs]), statuses[i.max(j)]);
            }
        }
        assert_eq!(aggregate_proof_statuses([]), ProofStatus::Inconclusive);
    }

    #[test]
    fn status_wire_names_and_exit_codes_are_explicit() {
        for (status, name, exit_code) in [
            (ProofStatus::Proved, "proved", 0),
            (ProofStatus::Disproved, "disproved", 1),
            (ProofStatus::Inconclusive, "inconclusive", 1),
            (ProofStatus::ExecutionError, "execution-error", 1),
        ] {
            let serialized = serde_json::to_value(status).unwrap();
            assert_eq!(serialized, serde_json::json!(name));
            assert_eq!(
                serde_json::from_value::<ProofStatus>(serialized).unwrap(),
                status
            );
            assert_eq!(status.exit_code(), exit_code);
            assert_eq!(status.is_proved(), exit_code == 0);
        }
    }
}
