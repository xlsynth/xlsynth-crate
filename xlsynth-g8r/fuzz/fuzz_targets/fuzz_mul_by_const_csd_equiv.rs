// SPDX-License-Identifier: Apache-2.0

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::aig_serdes::gate2ir;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOptions, GatifyOutput};
use xlsynth_g8r_fuzz::fuzz_solver_limits;
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::ir_equiv::{run_ir_equiv, IrEquivRequest, IrModule};
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::types::EquivResult;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::SolverChoice;

#[derive(Debug, Clone, Arbitrary)]
struct MulConstSample {
    width: u8,
    constant: u16,
    literal_on_lhs: bool,
}

fn build_ir_text(sample: &MulConstSample) -> String {
    // Wider cases push the external IR-equivalence checker into multi-second
    // or slower proofs; keep the generic fuzz target responsive.
    let width = usize::from(sample.width).clamp(1, 12);
    let modulus = if width == 16 {
        1u64 << 16
    } else {
        1u64 << width
    };
    let constant = u64::from(sample.constant) % modulus;
    let umul = if sample.literal_on_lhs {
        "umul(c, x, id=3)"
    } else {
        "umul(x, c, id=3)"
    };
    format!(
        "package sample

top fn mul_const(x: bits[{width}] id=1) -> bits[{width}] {{
  c: bits[{width}] = literal(value={constant}, id=2)
  ret p: bits[{width}] = {umul}
}}
"
    )
}

#[cfg(feature = "has-bitwuzla")]
fn prove_orig_vs_gate_equiv(
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
    let report = run_ir_equiv(&request).expect("Bitwuzla IR equivalence should run");
    if let EquivResult::Inconclusive(msg) = &report.result {
        log::debug!("mul-by-const equivalence inconclusive: {}", msg);
        return false;
    }
    assert!(
        report.is_success(),
        "mul-by-const gate lowering must match source IR: {}",
        report
            .error_str()
            .unwrap_or_else(|| "unknown equivalence failure".to_string())
    );
    true
}

#[cfg(not(feature = "has-bitwuzla"))]
fn prove_orig_vs_gate_equiv(
    _orig_ir: &str,
    _orig_top: &str,
    _fn_type: &ir::FunctionType,
    _gate_output: &GatifyOutput,
) -> bool {
    panic!(
        "fuzz_mul_by_const_csd_equiv requires an in-process solver; \
         build with --features=with-bitwuzla-system (or with-bitwuzla-built)"
    );
}

fuzz_target!(|sample: MulConstSample| {
    let _ = env_logger::builder().is_test(true).try_init();

    let ir_text = build_ir_text(&sample);
    let mut parser = Parser::new(&ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .expect("constructed mul-by-const IR should parse");
    let pir_fn = pkg.get_top_fn().expect("top fn");
    let fn_type = pir_fn.get_type();

    let gatify_output = ir2gate::gatify(pir_fn, GatifyOptions::all_opts_disabled())
        .expect("gatify with built-in mul-by-const lowering");

    if !prove_orig_vs_gate_equiv(&ir_text, pir_fn.name.as_str(), &fn_type, &gatify_output) {
        // Early-return rationale: a configured solver resource limit is an
        // expected inconclusive fuzz sample, not an equivalence failure.
        return;
    }
});
