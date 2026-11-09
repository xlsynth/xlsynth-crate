// SPDX-License-Identifier: Apache-2.0

pub mod bx;
pub mod campaign;
pub mod cli;
pub mod native;

// Keep the native coverage runtime linked for probes and integration tests as
// well as fuzz binaries; the instrumented DSO calls its exported callbacks.
use libfuzzer_sys as _;

use xlsynth_dslx_fuzzer::{Grammar, PreparedSample};
use xlsynth_prover::ir_utils::parse_package_and_drop_params;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism, EquivResult, ProverFn};
use xlsynth_prover::prover::{SolverChoice, SolverLimits, prover_for_choice_with_limits};

/// Proves equality for all inputs, retaining counterexamples and bounded
/// unknowns.
pub fn prove(sample: &PreparedSample) -> Result<EquivResult, String> {
    prove_with_assertions(sample, AssertionSemantics::Ignore).map_err(CheckFailure::into_message)
}

/// Includes the assertion failure condition in the counterexample query when
/// the selected grammar generates arm-local runtime assertions.
fn prove_with_assertions(
    sample: &PreparedSample,
    semantics: AssertionSemantics,
) -> Result<EquivResult, CheckFailure> {
    let (expected_pkg, expected_fn) =
        parse_package_and_drop_params(&sample.expected_ir, Some("main"), &[])?;
    let (converted_pkg, converted_fn) =
        parse_package_and_drop_params(&sample.converted_ir, Some(&sample.converted_top), &[])?;
    // The public DSLX interface must survive lowering, even when distinct IR
    // return types have the same flattened bit width in the SMT comparison.
    let expected = expected_fn.get_type();
    let converted = converted_fn.get_type();
    if expected != converted {
        return Err(CheckFailure::InvalidOutput(format!(
            "Converted top signature differs from the generated reference: expected {expected:?}, got {converted:?}\n{sample:#?}"
        )));
    }
    let prover = prover_for_choice_with_limits(
        SolverChoice::Bitwuzla,
        None,
        SolverLimits {
            time_limit_per_ms: Some(10_000),
            memory_limit_mb: Some(1024),
        },
    );
    match prover.prove_ir_equiv(
        &ProverFn::new(&expected_fn, Some(&expected_pkg)),
        &ProverFn::new(&converted_fn, Some(&converted_pkg)),
        EquivParallelism::SingleThreaded,
        semantics,
        None,
        false,
    ) {
        EquivResult::Error(message) => Err(CheckFailure::Checker(message)),
        result => Ok(result),
    }
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
        Err(failure @ CheckFailure::InvalidOutput(_)) => Err(failure),
        failure => {
            let message = format!("equivalence check failed: {failure:#?}\n{sample:#?}");
            match failure {
                Ok(EquivResult::Disproved { .. }) => Err(CheckFailure::Disproved(message)),
                _ => Err(CheckFailure::Checker(message)),
            }
        }
    }
}
