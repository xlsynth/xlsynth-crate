// SPDX-License-Identifier: Apache-2.0

pub block active_high_props(
  input clk: clock,
  input rst: reset<active_high, sync>,
  input predicate: bool,
  output y: bool,
) {
  assert!(predicate, "active-high assertion");
  cover!(predicate, "active-high cover");
  assign y = predicate;
}

pub block active_low_props(
  input clk: clock,
  input rst_n: reset<active_low, sync>,
  input predicate: bool,
  output y: bool,
) {
  assert!(predicate, "active-low assertion");
  cover!(predicate, "active-low cover");
  assign y = predicate;
}
