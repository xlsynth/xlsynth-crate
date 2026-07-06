// SPDX-License-Identifier: Apache-2.0

proc Producer {
  output: chan<u8> out;
  config(output: chan<u8> out) { (output,) }
  init { u8:0 }
  next(state: u8) {
    let tok = send(join(), output, state);
    state + u8:1
  }
}

pub block proc_active_low(
  input clk: clock,
  input rst_n: reset<active_low, sync>,
  input ready: bool,
  output valid: bool,
  output data: u8,
) {
  inst producer: Producer { _output_rdy: ready, }
  assign valid = producer._output_vld;
  assign data = producer._output;
}
