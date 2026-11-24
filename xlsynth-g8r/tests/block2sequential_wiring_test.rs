// SPDX-License-Identifier: Apache-2.0

//! Sequential gate lowering legalizes acyclic structural feedback when needed.

use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::block2sequential::block_ir_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_pir::IrBits;

const FEEDBACK: &str = r#"package gate_feedback
block leaf(d: bits[4], q: bits[4]) {
  d: bits[4] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(seed: bits[1], result: bits[4]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=4)
  low: bits[3] = bit_slice(q, start=0, width=3, id=5)
  driver: bits[4] = concat(low, seed, id=6)
  drive: () = instantiation_input(driver, instantiation=u, port_name=d, id=7)
  result: () = output_port(q, name=result, id=8)
}
"#;

#[test]
fn gate_lowering_preserves_the_function_of_structural_feedback() {
    let gate =
        block_ir_to_sequential_gate_fn(FEEDBACK, GatifyOptions::all_opts_disabled()).unwrap();
    assert!(gate.registers.is_empty());
    for seed in [false, true] {
        let result = gate_sim::eval(&gate.transition, &[IrBits::bool(seed)], Collect::None);
        let expected = IrBits::make_ubits(4, if seed { 15 } else { 0 }).unwrap();
        assert_eq!(result.outputs, vec![expected]);
    }
}

#[test]
fn gate_lowering_keeps_actual_feedback_an_error() {
    let source = FEEDBACK.replace("instantiation_input(driver,", "instantiation_input(q,");
    let error =
        block_ir_to_sequential_gate_fn(&source, GatifyOptions::all_opts_disabled()).unwrap_err();
    assert!(
        error.starts_with("block2sequential: block inlining failed:"),
        "unexpected error: {error}"
    );
    assert!(error.contains("cycle"), "unexpected error: {error}");
}

#[test]
fn gate_lowering_uses_behavioral_external_model_for_feedback() {
    let source = FEEDBACK.replace(
        "instantiation u(block=leaf, kind=block)",
        "instantiation u(block=leaf, module_name=external_leaf, parameters={WIDTH:\"4\"}, kind=behavioral_extern_block)",
    );
    let gate = block_ir_to_sequential_gate_fn(&source, GatifyOptions::all_opts_disabled()).unwrap();
    for seed in [false, true] {
        let result = gate_sim::eval(&gate.transition, &[IrBits::bool(seed)], Collect::None);
        assert_eq!(
            result.outputs,
            [IrBits::make_ubits(4, if seed { 15 } else { 0 }).unwrap()]
        );
    }
}
