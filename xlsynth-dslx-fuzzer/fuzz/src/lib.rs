// SPDX-License-Identifier: Apache-2.0

pub mod campaign;
pub mod cli;
pub mod native;

// Keep the native coverage runtime linked for probes and integration tests as
// well as fuzz binaries; the instrumented DSO calls its exported callbacks.
use libfuzzer_sys as _;

use xlsynth_dslx_fuzzer::{Grammar, PreparedSample};
use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule, run_ir_equiv};
use xlsynth_prover::prover::types::{AssertionSemantics, EquivResult};
use xlsynth_prover::prover::{SolverChoice, SolverLimits};

/// Proves equality for all inputs, retaining counterexamples and bounded
/// unknowns.
pub fn prove(sample: &PreparedSample) -> Result<EquivResult, String> {
    prove_with_assertions(sample, AssertionSemantics::Ignore)
}

/// Includes the assertion failure condition in the counterexample query when
/// the selected grammar generates arm-local runtime assertions.
fn prove_with_assertions(
    sample: &PreparedSample,
    semantics: AssertionSemantics,
) -> Result<EquivResult, String> {
    let request = IrEquivRequest::new(
        IrModule::new(&sample.expected_ir).with_top(Some("main")),
        IrModule::new(&sample.converted_ir).with_top(Some(&sample.converted_top)),
    )
    .with_solver(Some(SolverChoice::Bitwuzla))
    .with_assertion_semantics(semantics)
    .with_solver_limits(SolverLimits {
        time_limit_per_ms: Some(10_000),
        memory_limit_mb: Some(1024),
    });
    Ok(run_ir_equiv(&request)?.result)
}

/// A concrete sample finding or a failure to run the checker reliably.
#[derive(Debug)]
pub enum CheckFailure {
    Disproved(String),
    ConverterRejected(String),
    InvalidOutput(String),
    StructuralGrowth(String),
    Checker(String),
}

impl From<String> for CheckFailure {
    fn from(message: String) -> Self {
        Self::Checker(message)
    }
}

impl CheckFailure {
    pub fn into_message(self) -> String {
        match self {
            Self::Disproved(message)
            | Self::ConverterRejected(message)
            | Self::InvalidOutput(message)
            | Self::StructuralGrowth(message)
            | Self::Checker(message) => message,
        }
    }
}

/// Returns proved or resource-limited outcomes and includes the full sample on
/// failure.
pub fn check_prepared(sample: &PreparedSample, grammar: Grammar) -> Result<EquivResult, String> {
    classify_prepared(sample, grammar).map_err(CheckFailure::into_message)
}

fn classify_prepared(
    sample: &PreparedSample,
    grammar: Grammar,
) -> Result<EquivResult, CheckFailure> {
    let semantics = if grammar.uses_v10_productions() {
        AssertionSemantics::Never
    } else {
        AssertionSemantics::Ignore
    };
    classify_with_assertions(sample, semantics)
}

fn classify_with_assertions(
    sample: &PreparedSample,
    semantics: AssertionSemantics,
) -> Result<EquivResult, CheckFailure> {
    match prove_with_assertions(sample, semantics) {
        Ok(result @ (EquivResult::Proved | EquivResult::Inconclusive(_))) => Ok(result),
        failure => {
            let message = format!("equivalence check failed: {failure:#?}\n{sample:#?}");
            match failure {
                Ok(EquivResult::Disproved { .. }) => Err(CheckFailure::Disproved(message)),
                _ => Err(CheckFailure::Checker(message)),
            }
        }
    }
}
