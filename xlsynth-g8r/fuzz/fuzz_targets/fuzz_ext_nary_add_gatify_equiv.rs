// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::time::Instant;

use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig_serdes::gate2ir;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOptions};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
use xlsynth_pir::ext_nary_add_fuzz::{
    EXT_NARY_ADD_FUZZ_FUNCTION_NAME, generate_ext_nary_add_fn_sample_without_zero_widths,
    render_ext_nary_add_fn_sample,
};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::EquivResult;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::ir_equiv::prove_ir_fn_equiv;
#[cfg(any(feature = "has-bitwuzla", feature = "has-boolector"))]
use xlsynth_prover::prover::types::{AssertionSemantics, ProverFn};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::solver::bitwuzla::{Bitwuzla, BitwuzlaOptions};
#[cfg(all(not(feature = "has-bitwuzla"), feature = "has-boolector"))]
use xlsynth_prover::prover::ir_equiv::prove_ir_fn_equiv;
#[cfg(all(not(feature = "has-bitwuzla"), feature = "has-boolector"))]
use xlsynth_prover::solver::boolector::{Boolector, BoolectorConfig};

#[cfg(feature = "has-bitwuzla")]
fn prove_exported_vs_gate_equiv(
    exported_fn: &ir::Fn,
    exported_pkg: &ir::Package,
    gate_fn: &ir::Fn,
    gate_pkg: &ir::Package,
) -> EquivResult {
    prove_ir_fn_equiv::<Bitwuzla>(
        &BitwuzlaOptions::new(),
        &ProverFn::new(exported_fn, Some(exported_pkg)),
        &ProverFn::new(gate_fn, Some(gate_pkg)),
        AssertionSemantics::Same,
        None,
        false,
    )
}

#[cfg(all(not(feature = "has-bitwuzla"), feature = "has-boolector"))]
fn prove_exported_vs_gate_equiv(
    exported_fn: &ir::Fn,
    exported_pkg: &ir::Package,
    gate_fn: &ir::Fn,
    gate_pkg: &ir::Package,
) -> EquivResult {
    prove_ir_fn_equiv::<Boolector>(
        &BoolectorConfig::new(),
        &ProverFn::new(exported_fn, Some(exported_pkg)),
        &ProverFn::new(gate_fn, Some(gate_pkg)),
        AssertionSemantics::Same,
        None,
        false,
    )
}

#[cfg(all(not(feature = "has-bitwuzla"), not(feature = "has-boolector")))]
fn prove_exported_vs_gate_equiv(
    _exported_fn: &ir::Fn,
    _exported_pkg: &ir::Package,
    _gate_fn: &ir::Fn,
    _gate_pkg: &ir::Package,
) -> EquivResult {
    panic!(
        "fuzz_ext_nary_add_gatify_equiv requires an in-process solver; \
         build with --features=with-bitwuzla-system, with-bitwuzla-built, \
         with-boolector-system, or with-boolector-built"
    )
}

fuzz_target!(|data: &[u8]| {
    let _ = env_logger::builder().is_test(true).try_init();

    let sample = generate_ext_nary_add_fn_sample_without_zero_widths(data);
    let ir_text = render_ext_nary_add_fn_sample(&sample);
    log::info!("generated ext_nary_add IR:\n{}", ir_text);
    let mut parser = Parser::new(&ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .unwrap_or_else(|e| panic!("generated ext_nary_add IR should parse:\n{ir_text}\nerror: {e}"));
    log::info!("parsed and validated ext_nary_add IR");
    let pir_fn = pkg.get_top_fn().expect("generated package should have a top fn");

    let gatify_start = Instant::now();
    let gatify_output = ir2gate::gatify(
        pir_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            array_index_lowering_strategy: Default::default(),
        },
    )
    .expect("gatify should succeed for generated ext_nary_add IR");
    log::info!("gatify completed in {:?}", gatify_start.elapsed());

    // The in-process prover works over PIR functions, so we prove equivalence
    // against the exported/desugared XLS IR form of the same package.
    let export_start = Instant::now();
    let exported_ir_text =
        emit_package_as_xls_ir_text(&pkg).expect("exporting generated ext_nary_add IR should work");
    log::info!(
        "emit_package_as_xls_ir_text completed in {:?} ({} bytes)",
        export_start.elapsed(),
        exported_ir_text.len()
    );
    let gate_ir_start = Instant::now();
    let gate_ir_text = gate2ir::gate_fn_to_xlsynth_ir(
        &gatify_output.gate_fn,
        EXT_NARY_ADD_FUZZ_FUNCTION_NAME,
        &pir_fn.get_type(),
    )
    .expect("gate_fn_to_xlsynth_ir should succeed")
    .to_string();
    log::info!(
        "gate_fn_to_xlsynth_ir completed in {:?} ({} bytes)",
        gate_ir_start.elapsed(),
        gate_ir_text.len()
    );

    let exported_pkg = Parser::new(&exported_ir_text)
        .parse_and_validate_package()
        .unwrap_or_else(|e| {
            panic!("exported ext_nary_add IR should parse:\n{exported_ir_text}\nerror: {e}")
        });
    let gate_pkg = Parser::new(&gate_ir_text)
        .parse_and_validate_package()
        .unwrap_or_else(|e| panic!("gate IR should parse:\n{gate_ir_text}\nerror: {e}"));
    let exported_fn = exported_pkg
        .get_top_fn()
        .expect("exported package should have a top fn");
    let gate_fn = gate_pkg
        .get_top_fn()
        .expect("gate package should have a top fn");

    let prove_start = Instant::now();
    let equiv_result =
        prove_exported_vs_gate_equiv(exported_fn, &exported_pkg, gate_fn, &gate_pkg);

    match equiv_result {
        EquivResult::Proved => {}
        EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } => panic!(
            "gatify equivalence disproved for generated ext_nary_add IR:\n{ir_text}\nexported:\n{exported_ir_text}\ngate:\n{gate_ir_text}\nlhs_inputs={lhs_inputs:?}\nrhs_inputs={rhs_inputs:?}\nlhs_output={lhs_output:?}\nrhs_output={rhs_output:?}"
        ),
        EquivResult::ToolchainDisproved(msg) | EquivResult::Error(msg) => panic!(
            "in-process formal equivalence failed for generated ext_nary_add IR:\n{ir_text}\nmessage: {msg}"
        ),
    }
    log::info!("in-process equivalence completed in {:?}", prove_start.elapsed());
});
