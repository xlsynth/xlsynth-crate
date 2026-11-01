// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::gatify::ir2gate::GatifyOutput;
use xlsynth_g8r::ir2gates::{Ir2GatesOptions, ir2gates_from_ir_text};

#[test]
fn callers_can_construct_and_exhaustively_destructure_gatify_output() {
    let ir = "package api

top fn f(x: bits[1] id=1) -> bits[1] {
  ret y: bits[1] = not(x, id=2)
}
";
    let output =
        ir2gates_from_ir_text(ir, Some("f"), Ir2GatesOptions::all_opts_disabled()).unwrap();
    let GatifyOutput {
        gate_fn,
        lowering_map,
    } = output.gatify_output;
    let rebuilt = GatifyOutput {
        gate_fn,
        lowering_map,
    };
    assert_eq!(rebuilt.gate_fn.outputs.len(), 1);
    assert_eq!(rebuilt.gate_fn.outputs[0].get_bit_count(), 1);
}
