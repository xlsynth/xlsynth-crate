// SPDX-License-Identifier: Apache-2.0

pub mod native;

// Keep the native coverage runtime linked for probes and integration tests as
// well as fuzz binaries; the instrumented DSO calls its exported callbacks.
use libfuzzer_sys as _;

use xlsynth_dslx_fuzzer::PreparedSample;
use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule, run_ir_equiv};
use xlsynth_prover::prover::types::EquivResult;
use xlsynth_prover::prover::{SolverChoice, SolverLimits};

/// Proves equality for all inputs, retaining counterexamples and bounded
/// unknowns.
pub fn prove(sample: &PreparedSample) -> Result<EquivResult, String> {
    let request = IrEquivRequest::new(
        IrModule::new(&sample.expected_ir).with_top(Some("main")),
        IrModule::new(&sample.converted_ir).with_top(Some(&sample.converted_top)),
    )
    .with_solver(Some(SolverChoice::Bitwuzla))
    .with_solver_limits(SolverLimits {
        time_limit_per_ms: Some(10_000),
        memory_limit_mb: Some(1024),
    });
    Ok(run_ir_equiv(&request)?.result)
}
