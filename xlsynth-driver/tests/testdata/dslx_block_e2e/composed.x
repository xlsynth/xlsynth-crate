// SPDX-License-Identifier: Apache-2.0

#[extern_verilog("assign {return} = {x} ^ 8'hA5;")]
fn ffi_scramble(x: u8) -> u8 { x }

fn add_bias(x: u8) -> u8 { x + u8:3 }

block custom_logic(
  output y: u8,
  input x: u8,
  input clk: clock,
  input rst: reset<active_high, sync>,
) {
  let biased = add_bias(x);
  assign y = biased;
}

proc Producer {
  output: chan<u8> out;
  config(output: chan<u8> out) { (output,) }
  init { u8:0 }
  next(state: u8) {
    let tok = send(join(), output, state);
    state + u8:1
  }
}

block proc_wrapper(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input ready: bool,
  output valid: bool,
  output data: u8,
) {
  // Clock and reset are structural connections; only data ports are bound.
  inst producer: Producer { _output_rdy: ready, }
  assign valid = producer._output_vld;
  assign data = producer._output;
}

pub block composed_top(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  input ready: bool,
  output transformed: u8,
  output stream_valid: bool,
  output stream_data: u8,
) {
  inst custom_i: custom_logic { x: x, }
  inst stream_i: proc_wrapper { ready: ready, }
  // `logic` is legal DSLX but a SystemVerilog keyword. The extern bridge must
  // bind the FFI operand through its own codegen-stable signal name.
  let logic = custom_i.y + u8:1;
  assign transformed = ffi_scramble(logic);
  assign stream_valid = stream_i.valid;
  assign stream_data = stream_i.data;
}

// This retained non-top user proves that --module_name updates hierarchical
// references as well as the selected module declaration.
block retained_parent(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input x: u8,
  input ready: bool,
  output transformed: u8,
  output stream_valid: bool,
  output stream_data: u8,
) {
  inst composed_i: composed_top { x: x, ready: ready, }
  assign transformed = composed_i.transformed;
  assign stream_valid = composed_i.stream_valid;
  assign stream_data = composed_i.stream_data;
}
