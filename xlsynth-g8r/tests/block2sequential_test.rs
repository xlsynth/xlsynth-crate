// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrBits;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::block2sequential::block_ir_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;

fn lower(block_ir: &str) -> xlsynth_g8r::aig::SequentialGateFn {
    block_ir_to_sequential_gate_fn(block_ir, GatifyOptions::all_opts_disabled())
        .expect("block lowering should succeed")
}

#[test]
fn lowers_combinational_block_without_registers_or_clock() {
    let block_ir = r#"package combinational

top block top(a: bits[1], b: bits[1], y: bits[1], z: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  b: bits[1] = input_port(name=b, id=2)
  and.3: bits[1] = and(a, b, id=3)
  not.4: bits[1] = not(and.3, id=4)
  y: () = output_port(and.3, name=y, id=5)
  z: () = output_port(not.4, name=z, id=6)
}
"#;

    let sequential = lower(block_ir);

    assert!(sequential.registers.is_empty());
    assert!(sequential.clock.is_none());
    assert_eq!(
        sequential
            .transition
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(
        sequential
            .transition
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>(),
        vec!["y", "z"]
    );

    let result = gate_sim::eval(
        &sequential.transition,
        &[IrBits::bool(true), IrBits::bool(true)],
        Collect::None,
    );
    assert_eq!(
        result.outputs,
        vec![IrBits::bool(true), IrBits::bool(false)]
    );
}

#[test]
fn lowers_register_load_enable_and_synchronous_reset_into_effective_d() {
    let block_ir = r#"package pipeline

top block pipe(clk: clock, rst: bits[1], data: bits[8], le: bits[1], out: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=true)]
  reg state(bits[8], reset_value=3)
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[8] = input_port(name=data, id=2)
  le: bits[1] = input_port(name=le, id=3)
  state_q: bits[8] = register_read(register=state, id=4)
  add.5: bits[8] = add(state_q, data, id=5)
  state_d: () = register_write(add.5, register=state, load_enable=le, reset=rst, id=6)
  out: () = output_port(state_q, name=out, id=7)
}
"#;

    let sequential = lower(block_ir);

    let clock = sequential.clock.as_ref().expect("clock");
    assert_eq!(clock.name, "clk");
    assert_eq!(
        sequential
            .transition
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        vec!["rst", "data", "le", "state__q"]
    );
    assert_eq!(
        sequential
            .transition
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>(),
        vec!["out", "state__d"]
    );

    let register = &sequential.registers[0];
    assert_eq!(register.name, "state");
    assert_eq!(register.q.index(), 3);
    assert_eq!(register.d.index(), 1);

    let reset_result = gate_sim::eval(
        &sequential.transition,
        &[
            IrBits::bool(false),
            IrBits::make_ubits(8, 2).unwrap(),
            IrBits::bool(true),
            IrBits::make_ubits(8, 5).unwrap(),
        ],
        Collect::None,
    );
    assert_eq!(
        reset_result.outputs,
        vec![
            IrBits::make_ubits(8, 5).unwrap(),
            IrBits::make_ubits(8, 3).unwrap(),
        ]
    );

    let held_result = gate_sim::eval(
        &sequential.transition,
        &[
            IrBits::bool(true),
            IrBits::make_ubits(8, 2).unwrap(),
            IrBits::bool(false),
            IrBits::make_ubits(8, 5).unwrap(),
        ],
        Collect::None,
    );
    assert_eq!(
        held_result.outputs,
        vec![
            IrBits::make_ubits(8, 5).unwrap(),
            IrBits::make_ubits(8, 5).unwrap(),
        ]
    );

    let enabled_result = gate_sim::eval(
        &sequential.transition,
        &[
            IrBits::bool(true),
            IrBits::make_ubits(8, 2).unwrap(),
            IrBits::bool(true),
            IrBits::make_ubits(8, 5).unwrap(),
        ],
        Collect::None,
    );
    assert_eq!(
        enabled_result.outputs,
        vec![
            IrBits::make_ubits(8, 5).unwrap(),
            IrBits::make_ubits(8, 7).unwrap(),
        ]
    );
}

#[test]
fn lowers_registers_after_inlining_block_instantiations() {
    let block_ir = r#"package hierarchical

block stage(data: bits[1], out: bits[1]) {
  reg state(bits[1])
  data: bits[1] = input_port(name=data, id=1)
  state_q: bits[1] = register_read(register=state, id=2)
  state_d: () = register_write(data, register=state, id=3)
  out: () = output_port(state_q, name=out, id=4)
}

top block top(clk: clock, data: bits[1], out: bits[1]) {
  instantiation s0(block=stage, kind=block)
  data: bits[1] = input_port(name=data, id=11)
  s0_out: bits[1] = instantiation_output(instantiation=s0, port_name=out, id=12)
  s0_in: () = instantiation_input(data, instantiation=s0, port_name=data, id=13)
  out: () = output_port(s0_out, name=out, id=14)
}
"#;

    let sequential = lower(block_ir);

    assert_eq!(sequential.registers.len(), 1);
    assert_eq!(sequential.registers[0].name, "s0__state");
    assert_eq!(sequential.clock.as_ref().expect("clock").name, "clk");
    assert_eq!(
        sequential.transition.outputs[1].name,
        "s0__state__d".to_string()
    );
}

#[test]
fn lowers_inlined_register_reset_with_compatible_explicit_metadata() {
    let block_ir = r#"package hierarchical_reset

block stage(clk: clock, local_rst: bits[1], data: bits[1], out: bits[1]) {
  #![reset(port="local_rst", asynchronous=false, active_low=true)]
  reg state(bits[1], reset_value=0)
  local_rst: bits[1] = input_port(name=local_rst, id=1)
  data: bits[1] = input_port(name=data, id=2)
  state_q: bits[1] = register_read(register=state, id=3)
  state_d: () = register_write(data, register=state, reset=local_rst, id=4)
  out: () = output_port(state_q, name=out, id=5)
}

top block top(clk: clock, rst: bits[1], data: bits[1], out: bits[1]) {
  #![reset(port="rst", asynchronous=false, active_low=true)]
  instantiation s0(block=stage, kind=block)
  rst: bits[1] = input_port(name=rst, id=11)
  data: bits[1] = input_port(name=data, id=12)
  s0_out: bits[1] = instantiation_output(instantiation=s0, port_name=out, id=13)
  s0_rst: () = instantiation_input(rst, instantiation=s0, port_name=local_rst, id=14)
  s0_data: () = instantiation_input(data, instantiation=s0, port_name=data, id=15)
  out: () = output_port(s0_out, name=out, id=16)
}
"#;

    let sequential = lower(block_ir);

    assert_eq!(sequential.registers[0].d.index(), 1);
    assert_eq!(
        sequential
            .transition
            .outputs
            .iter()
            .map(|output| output.name.as_str())
            .collect::<Vec<_>>(),
        vec!["out", "s0__state__d"]
    );
}

#[test]
fn rejects_inlined_registered_block_with_incompatible_explicit_clock() {
    let block_ir = r#"package hierarchical_clock

block stage(child_clk: clock, data: bits[1], out: bits[1]) {
  reg state(bits[1])
  data: bits[1] = input_port(name=data, id=1)
  state_q: bits[1] = register_read(register=state, id=2)
  state_d: () = register_write(data, register=state, id=3)
  out: () = output_port(state_q, name=out, id=4)
}

top block top(clk: clock, data: bits[1], out: bits[1]) {
  instantiation s0(block=stage, kind=block)
  data: bits[1] = input_port(name=data, id=11)
  s0_out: bits[1] = instantiation_output(instantiation=s0, port_name=out, id=12)
  s0_data: () = instantiation_input(data, instantiation=s0, port_name=data, id=13)
  out: () = output_port(s0_out, name=out, id=14)
}
"#;

    let error =
        block_ir_to_sequential_gate_fn(block_ir, GatifyOptions::all_opts_disabled()).unwrap_err();
    assert_eq!(
        error,
        "block2sequential: registered instantiated block 'stage' at 'top.s0' declares clock 'child_clk' incompatible with top clock 'clk'"
    );
}

#[test]
fn rejects_inlined_reset_block_with_incompatible_reset_behavior() {
    let block_ir = r#"package hierarchical_reset_mismatch

block stage(clk: clock, local_rst: bits[1], data: bits[1], out: bits[1]) {
  #![reset(port="local_rst", asynchronous=false, active_low=false)]
  reg state(bits[1], reset_value=0)
  local_rst: bits[1] = input_port(name=local_rst, id=1)
  data: bits[1] = input_port(name=data, id=2)
  state_q: bits[1] = register_read(register=state, id=3)
  state_d: () = register_write(data, register=state, reset=local_rst, id=4)
  out: () = output_port(state_q, name=out, id=5)
}

top block top(clk: clock, rst: bits[1], data: bits[1], out: bits[1]) {
  #![reset(port="rst", asynchronous=false, active_low=true)]
  instantiation s0(block=stage, kind=block)
  rst: bits[1] = input_port(name=rst, id=11)
  data: bits[1] = input_port(name=data, id=12)
  s0_out: bits[1] = instantiation_output(instantiation=s0, port_name=out, id=13)
  s0_rst: () = instantiation_input(rst, instantiation=s0, port_name=local_rst, id=14)
  s0_data: () = instantiation_input(data, instantiation=s0, port_name=data, id=15)
  out: () = output_port(s0_out, name=out, id=16)
}
"#;

    let error =
        block_ir_to_sequential_gate_fn(block_ir, GatifyOptions::all_opts_disabled()).unwrap_err();
    assert_eq!(
        error,
        "block2sequential: reset-bearing instantiated block 'stage' at 'top.s0' declares reset behavior (asynchronous=false, active_low=false) incompatible with top reset behavior (asynchronous=false, active_low=true)"
    );
}

#[test]
fn rejects_asynchronous_register_reset() {
    let block_ir = r#"package asynchronous_reset

top block top(clk: clock, rst: bits[1], data: bits[1], out: bits[1]) {
  #![reset(port="rst", asynchronous=true, active_low=false)]
  reg state(bits[1], reset_value=0)
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[1] = input_port(name=data, id=2)
  state_q: bits[1] = register_read(register=state, id=3)
  state_d: () = register_write(data, register=state, reset=rst, id=4)
  out: () = output_port(state_q, name=out, id=5)
}
"#;

    let error =
        block_ir_to_sequential_gate_fn(block_ir, GatifyOptions::all_opts_disabled()).unwrap_err();
    assert_eq!(
        error,
        "block2sequential: asynchronous reset is not supported for register 'state'"
    );
}

#[test]
fn rejects_register_reset_without_block_reset_metadata() {
    let block_ir = r#"package invalid_reset

top block top(clk: clock, rst: bits[1], data: bits[1], out: bits[1]) {
  reg state(bits[1], reset_value=0)
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[1] = input_port(name=data, id=2)
  state_q: bits[1] = register_read(register=state, id=3)
  state_d: () = register_write(data, register=state, reset=rst, id=4)
  out: () = output_port(state_q, name=out, id=5)
}
"#;

    let error =
        block_ir_to_sequential_gate_fn(block_ir, GatifyOptions::all_opts_disabled()).unwrap_err();
    assert_eq!(
        error,
        "block2sequential: register 'state' has a reset write but the block has no reset metadata"
    );
}
