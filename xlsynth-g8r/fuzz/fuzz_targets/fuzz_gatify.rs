// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig_serdes::gate2ir;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOutput};
use xlsynth_g8r::ir2gate_utils;
use xlsynth_pir::ir;
use xlsynth_pir::ir_fuzz::{generate_ir_fn, FuzzSample};
use xlsynth_pir::ir_parser;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::ir_equiv::{run_ir_equiv, IrEquivRequest, IrModule};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::SolverChoice;

#[cfg(feature = "has-bitwuzla")]
fn prove_orig_vs_gate_equiv(
    variant_label: &str,
    orig_ir: &str,
    orig_top: &str,
    fn_type: &ir::FunctionType,
    gate_output: &GatifyOutput,
) {
    let gate_ir = gate2ir::gate_fn_to_xlsynth_ir(&gate_output.gate_fn, "gate_pkg", fn_type)
        .expect("gate_fn_to_xlsynth_ir should succeed")
        .to_string();
    let gate_top = gate_output.gate_fn.name.as_str();
    let request = IrEquivRequest::new(
        IrModule::new(orig_ir).with_top(Some(orig_top)),
        IrModule::new(&gate_ir).with_top(Some(gate_top)),
    )
    .with_solver(Some(SolverChoice::Bitwuzla));
    match run_ir_equiv(&request) {
        Ok(report) => {
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
) {
    panic!(
        "fuzz_gatify requires an in-process solver; \
         build with --features=with-bitwuzla-system (or with-bitwuzla-built)"
    );
}

fuzz_target!(|sample: FuzzSample| {
    // Skip empty operation lists
    if sample.ops.is_empty() {
        return;
    }

    let _ = env_logger::builder().try_init();

    // Generate IR function from fuzz input
    let mut package = xlsynth::IrPackage::new("fuzz_test").unwrap();
    if let Err(e) = generate_ir_fn(sample.ops, &mut package, None) {
        log::info!("Error generating IR function: {}", e);
        return;
    }

    let parsed_package =
        match ir_parser::Parser::new(&package.to_string()).parse_and_validate_package() {
            Ok(parsed_package) => parsed_package,
            Err(e) => {
                log::error!(
                    "Error parsing IR package: {}\npackage:\n{}",
                    e,
                    package.to_string()
                );
                return;
            }
        };
    let parsed_fn = parsed_package.get_top_fn().unwrap();
    let orig_ir = package.to_string();
    let orig_top = parsed_fn.name.as_str();
    let fn_type = parsed_fn.get_type();

    // Convert to gates with folding disabled to make less machinery under test.
    let gate_fn_no_fold = ir2gate::gatify(
        &parsed_fn,
        ir2gate::GatifyOptions {
            fold: false,
            hash: false,
            check_equivalence: false,
            adder_mapping: ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            array_index_lowering_strategy: Default::default(),
        },
    );
    let gate_fn_no_fold = gate_fn_no_fold.expect("unfolded gatify should succeed");
    prove_orig_vs_gate_equiv("unfolded", &orig_ir, orig_top, &fn_type, &gate_fn_no_fold);

    log::info!("unfolded conversion succeeded, attempting folded version...");

    // Now check the folded version is also equivalent.
    let gate_fn_fold = ir2gate::gatify(
        &parsed_fn,
        ir2gate::GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping: ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            array_index_lowering_strategy: Default::default(),
        },
    );
    let gate_fn_fold = gate_fn_fold.expect("folded gatify should succeed");
    prove_orig_vs_gate_equiv("folded", &orig_ir, orig_top, &fn_type, &gate_fn_fold);

    // If we got here the equivalence checks passed.
    // Note: because of transitivity we know that also the unopt version is
    // equivalent to the opt version.
});
