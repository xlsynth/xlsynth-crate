// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig_serdes::gate2ir;
use xlsynth_g8r_fuzz::{fuzz_solver_limits, generate_full_g8r_fuzz_case};
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule, run_ir_equiv};
use xlsynth_prover::prover::SolverChoice;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::types::EquivResult;

/// Proves source-vs-g8r equivalence, returning false for a bounded timeout.
#[cfg(feature = "has-bitwuzla")]
fn prove_equivalence(
    source_ir: &str,
    source_top: &str,
    gate_ir: &str,
    gate_top: &str,
) -> bool {
    let request = IrEquivRequest::new(
        IrModule::new(source_ir).with_top(Some(source_top)),
        IrModule::new(gate_ir).with_top(Some(gate_top)),
    )
    .with_solver(Some(SolverChoice::Bitwuzla))
    .with_solver_limits(fuzz_solver_limits());
    let report = run_ir_equiv(&request)
        .unwrap_or_else(|error| panic!("g8r formal equivalence failed to run: {error}"));
    match report.result {
        EquivResult::Proved => true,
        EquivResult::Inconclusive(message) => {
            log::debug!("g8r formal equivalence inconclusive: {message}");
            false
        }
        EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } => panic!(
            "g8r formal equivalence disproved:\nsource IR:\n{source_ir}\n\
             gate IR:\n{gate_ir}\nlhs_inputs={lhs_inputs:?}\nrhs_inputs={rhs_inputs:?}\n\
             lhs_output={lhs_output:?}\nrhs_output={rhs_output:?}"
        ),
        EquivResult::ToolchainDisproved(message) | EquivResult::Error(message) => panic!(
            "g8r formal equivalence failed:\nsource IR:\n{source_ir}\n\
             gate IR:\n{gate_ir}\nmessage={message}"
        ),
    }
}

#[cfg(not(feature = "has-bitwuzla"))]
fn prove_equivalence(
    _source_ir: &str,
    _source_top: &str,
    _gate_ir: &str,
    _gate_top: &str,
) -> bool {
    panic!(
        "fuzz_g8r_formal requires an in-process solver; build with \
         --features=with-bitwuzla-system (or with-bitwuzla-built)"
    )
}

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let case = generate_full_g8r_fuzz_case(data, "fuzz_g8r_formal")
        .unwrap_or_else(|error| panic!("{error}"));
    let source_fn = case
        .source_package
        .get_fn(&case.source_top)
        .expect("generated package should retain its top function");
    let exported_source_ir = emit_package_as_xls_ir_text(&case.source_package)
        .expect("generated PIR should export to standard XLS IR");
    let gate_ir = gate2ir::gate_fn_to_xlsynth_ir(
        &case.gate_fn,
        "fuzz_g8r_formal_gate",
        &source_fn.get_type(),
    )
    .expect("optimized GateFn should export to XLS IR")
    .to_string();
    if !prove_equivalence(
        &exported_source_ir,
        &case.source_top,
        &gate_ir,
        &case.gate_fn.name,
    ) {
        // A configured solver time or memory limit can make hard samples
        // inconclusive; those samples are not g8r correctness failures.
        return;
    }
});
