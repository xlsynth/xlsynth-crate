// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::sequential_cleanup::cleanup_sequential_gate_fn;
use xlsynth_g8r::aig_serdes::sequential_transition::rebind_sequential_transition;
use xlsynth_g8r::block2sequential::block_ir_to_sequential_gate_fn_with_controls;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;

#[test]
fn cleanup_and_transition_rebinding_require_effective_d() {
    let source = block_ir_to_sequential_gate_fn_with_controls(
        r#"package control
top block top(clk: clock, x: bits[1], en: bits[1], out: bits[1]) {
  reg state(bits[1])
  x: bits[1] = input_port(name=x, id=1)
  en: bits[1] = input_port(name=en, id=2)
  q: bits[1] = register_read(register=state, id=3)
  d: () = register_write(x, register=state, load_enable=en, id=4)
  out: () = output_port(q, name=out, id=5)
}
"#,
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap();
    let error = "register 'state' has explicit reset/enable controls; call legalize_register_controls() first";
    assert_eq!(cleanup_sequential_gate_fn(&source).unwrap_err(), error);
    assert_eq!(
        rebind_sequential_transition(&source, &source.transition).unwrap_err(),
        error
    );
    let legalized = source.legalize_register_controls().unwrap();
    assert!(cleanup_sequential_gate_fn(&legalized).is_ok());
    assert!(rebind_sequential_transition(&legalized, &legalized.transition).is_ok());
}
