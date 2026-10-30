// SPDX-License-Identifier: Apache-2.0

//! Typed proof conclusions shared by solver results and subprocess reports.

use serde::de::{self, MapAccess, Visitor, value::MapAccessDeserializer, value::StrDeserializer};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::time::{Duration, Instant};

use crate::proof_phases::PhaseCollector;
use crate::proof_timing::ProofTiming;
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

/// The common fields in every proof result exchanged with a subprocess.
/// Only `status` is stored: `success` is derived on write and checked on read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(into = "StatusFields")]
pub struct ProofResultHeader {
    pub status: ProofStatus,
}

/// Both fields are required in the wire protocol.
#[derive(Serialize, Deserialize)]
struct StatusFields {
    #[serde(deserialize_with = "deserialize_status_name")]
    status: ProofStatus,
    success: bool,
}

impl From<ProofStatus> for ProofResultHeader {
    fn from(status: ProofStatus) -> Self {
        Self { status }
    }
}

impl From<ProofResultHeader> for StatusFields {
    fn from(header: ProofResultHeader) -> Self {
        Self {
            status: header.status,
            success: header.status.is_proved(),
        }
    }
}

// Serde also accepts an externally tagged object for a unit enum. The worker
// protocol specifies a string, so check its type before parsing the variant.
fn deserialize_status_name<'de, D: Deserializer<'de>>(d: D) -> Result<ProofStatus, D::Error> {
    let name = String::deserialize(d)?;
    ProofStatus::deserialize(StrDeserializer::new(&name))
}

impl<'de> Deserialize<'de> for ProofResultHeader {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // A derived struct would accept a positional array as an alternative
        // to a named object. Only an object constitutes a worker report.
        struct ReportObject;
        impl<'de> Visitor<'de> for ReportObject {
            type Value = ProofResultHeader;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a proof result object with status and success fields")
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                let fields = StatusFields::deserialize(MapAccessDeserializer::new(map))?;
                if fields.success != fields.status.is_proved() {
                    return Err(de::Error::custom("proof result status contradicts success"));
                }
                Ok(fields.status.into())
            }
        }
        d.deserialize_map(ReportObject)
    }
}

/// A timed proof conclusion with an optional diagnostic, including setup
/// errors.
#[derive(Debug, Serialize, Clone)]
pub struct ProofSummary {
    pub time_micros: u128,
    pub timing: ProofTiming,
    #[serde(flatten)]
    pub conclusion: ProofResultHeader,
    pub error_str: Option<String>,
}

impl ProofSummary {
    /// Retains measured phases even when setup or the solver returns an error.
    pub(crate) fn run_with_phases<T: Into<Self>>(
        prove: impl FnOnce(&PhaseCollector) -> Result<T, String>,
    ) -> Self {
        let phases = PhaseCollector::default();
        let mut report = Self::run(|| prove(&phases));
        report.timing = report.timing.with_phases(phases.snapshot());
        report
    }

    /// Runs a proof invocation, including setup failures in the report.
    pub fn run<T: Into<Self>>(prove: impl FnOnce() -> Result<T, String>) -> Self {
        let start = Instant::now();
        match prove() {
            Ok(report) => {
                let mut report = report.into();
                report.timing.wall_micros = ProofTiming::elapsed(start).wall_micros;
                report
            }
            Err(error) => Self::execution_error(start.elapsed(), error),
        }
    }

    /// Reports an invocation failure without claiming a counterexample.
    pub fn execution_error(duration: Duration, error: String) -> Self {
        Self {
            time_micros: duration.as_micros(),
            timing: duration.into(),
            conclusion: ProofStatus::ExecutionError.into(),
            error_str: Some(error),
        }
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
