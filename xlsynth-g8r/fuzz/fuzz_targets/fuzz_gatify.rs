// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig_serdes::gate2ir;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOutput};
use xlsynth_g8r_fuzz::{fuzz_solver_limits, generate_gatify_random_pir_package};
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
use xlsynth_pir::ir;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule, run_ir_equiv};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::SolverChoice;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::types::EquivResult;

#[cfg(feature = "has-bitwuzla")]
fn prove_orig_vs_gate_equiv(
    variant_label: &str,
    orig_ir: &str,
    orig_top: &str,
    fn_type: &ir::FunctionType,
    gate_output: &GatifyOutput,
) -> bool {
    let gate_ir = gate2ir::gate_fn_to_xlsynth_ir(&gate_output.gate_fn, "gate_pkg", fn_type)
        .expect("gate_fn_to_xlsynth_ir should succeed")
        .to_string();
    let gate_top = gate_output.gate_fn.name.as_str();
    let request = IrEquivRequest::new(
        IrModule::new(orig_ir).with_top(Some(orig_top)),
        IrModule::new(&gate_ir).with_top(Some(gate_top)),
    )
    .with_solver(Some(SolverChoice::Bitwuzla))
    .with_solver_limits(fuzz_solver_limits());
    match run_ir_equiv(&request) {
        Ok(report) => {
            if let EquivResult::Inconclusive(msg) = &report.result {
                log::debug!("gatify {} equivalence inconclusive: {}", variant_label, msg);
                return false;
            }
            if !report.is_success() {
                let dump_stem = format!("/tmp/fuzz_gatify_last_{}", variant_label);
                let _ = std::fs::write(format!("{}_orig.ir", dump_stem), orig_ir);
                let _ = std::fs::write(format!("{}_gate.ir", dump_stem), &gate_ir);
                let err = report
                    .error_str()
                    .unwrap_or_else(|| "unknown equivalence failure".to_string());
                log::error!("gatify {} equivalence check failed: {}", variant_label, err);
                log::info!("Original IR:\n{}", orig_ir);
                log::info!("Gate IR:\n{}", gate_ir);
                panic!("gatify {} equivalence check failed: {}", variant_label, err);
            }
            true
        }
        Err(err) => {
            let dump_stem = format!("/tmp/fuzz_gatify_last_{}", variant_label);
            let _ = std::fs::write(format!("{}_orig.ir", dump_stem), orig_ir);
            let _ = std::fs::write(format!("{}_gate.ir", dump_stem), &gate_ir);
            log::error!("gatify equivalence check failed: {}", err);
            log::info!("Original IR:\n{}", orig_ir);
            log::info!("Gate IR:\n{}", gate_ir);
            panic!("gatify {} equivalence check failed: {}", variant_label, err);
        }
    }
}

#[cfg(not(feature = "has-bitwuzla"))]
fn prove_orig_vs_gate_equiv(
    _variant_label: &str,
    _orig_ir: &str,
    _orig_top: &str,
    _fn_type: &ir::FunctionType,
    _gate_output: &GatifyOutput,
) -> bool {
    panic!(
        "fuzz_gatify requires an in-process solver; \
         build with --features=with-bitwuzla-system (or with-bitwuzla-built)"
    );
}

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().try_init();

    let package = generate_gatify_random_pir_package(data, "fuzz_test");
    let parsed_fn = package
        .get_top_fn()
        .expect("generated PIR package should have a top function");
    let orig_ir = emit_package_as_xls_ir_text(&package)
        .expect("generated extension-bearing PIR should desugar to XLS IR");
    let orig_top = parsed_fn.name.as_str();
    let fn_type = parsed_fn.get_type();

    // Convert to gates with folding disabled to make less machinery under test.
    let gate_fn_no_fold = ir2gate::gatify(
        parsed_fn,
        ir2gate::GatifyOptions {
            fold: false,
            hash: false,
            ..ir2gate::GatifyOptions::all_opts_disabled()
        },
    );
    let gate_fn_no_fold = gate_fn_no_fold.expect("unfolded gatify should succeed");
    if !prove_orig_vs_gate_equiv("unfolded", &orig_ir, orig_top, &fn_type, &gate_fn_no_fold) {
        // Early-return rationale: a configured solver resource limit is an
        // expected inconclusive fuzz sample, not an equivalence failure.
        return;
    }

    log::info!("unfolded conversion succeeded, attempting folded version...");

    // Now check the folded version is also equivalent.
    let gate_fn_fold = ir2gate::gatify(parsed_fn, ir2gate::GatifyOptions::all_opts_disabled());
    let gate_fn_fold = gate_fn_fold.expect("folded gatify should succeed");
    if !prove_orig_vs_gate_equiv("folded", &orig_ir, orig_top, &fn_type, &gate_fn_fold) {
        // Early-return rationale: a configured solver resource limit is an
        // expected inconclusive fuzz sample, not an equivalence failure.
        return;
    }

    // If we got here the equivalence checks passed.
    // Note: because of transitivity we know that also the unopt version is
    // equivalent to the opt version.
});
