// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use crate::block2fn::{Block2FnOptions, block_ir_to_fn};
use crate::ir::NodePayload;
use crate::ir_query::{matches_node, parse_query};
use xlsynth::{IrBits, IrValue};

fn parse_bits(value: &str) -> IrBits {
    IrValue::parse_typed(value)
        .expect("parse literal")
        .to_bits()
        .expect("to_bits")
}

fn run_block2fn(
    block_ir: &str,
    tie_input_ports: &[(&str, &str)],
    drop_output_ports: &[&str],
) -> crate::ir::Fn {
    let mut tie_map = BTreeMap::new();
    for (name, value) in tie_input_ports.iter() {
        tie_map.insert((*name).to_string(), parse_bits(value));
    }
    let drop_set: BTreeSet<String> = drop_output_ports.iter().map(|s| s.to_string()).collect();
    let opts = Block2FnOptions {
        tie_input_ports: tie_map,
        drop_output_ports: drop_set,
    };
    block_ir_to_fn(block_ir, &opts)
        .expect("block2fn should succeed")
        .function
}

fn assert_return_matches(f: &crate::ir::Fn, query_text: &str) {
    let node_ref = f.ret_node_ref.expect("return present");
    let query =
        parse_query(query_text).unwrap_or_else(|e| panic!("invalid query '{}': {e}", query_text));
    if !matches_node(f, &query, node_ref) {
        panic!("return node did not match query '{}'", query_text);
    }
}

#[test]
fn e2e_trivial_passthrough_no_registers() {
    let block_ir = r#"package test

top block top(a: bits[8], out: bits[8]) {
  a: bits[8] = input_port(name=a, id=1)
  out: () = output_port(a, name=out, id=2)
}
"#;
    let f = run_block2fn(block_ir, &[], &[]);
    assert_eq!(f.params.len(), 1, "expected a single input param");
    assert_eq!(f.params[0].name, "a");
    assert_return_matches(&f, "get_param(name=\"a\")");
}

#[test]
fn e2e_tie_inputs_fold_add() {
    let block_ir = r#"package test

top block top(a: bits[4], b: bits[4], out: bits[4]) {
  a: bits[4] = input_port(name=a, id=1)
  b: bits[4] = input_port(name=b, id=2)
  myadd: bits[4] = add(a, b, id=3)
  out: () = output_port(myadd, name=out, id=4)
}
"#;
    let f = run_block2fn(block_ir, &[("a", "bits[4]:1"), ("b", "bits[4]:2")], &[]);
    assert_return_matches(&f, "literal(3)");
}

#[test]
fn e2e_block_inlining_single_instance() {
    let block_ir = r#"package test

block leaf(a: bits[1], y: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  not.2: bits[1] = not(a, id=2)
  y: () = output_port(not.2, name=y, id=3)
}

top block top(a: bits[1], y: bits[1]) {
  instantiation u0(block=leaf, kind=block)
  a: bits[1] = input_port(name=a, id=10)
  instantiation_output.11: bits[1] = instantiation_output(instantiation=u0, port_name=y, id=11)
  instantiation_input.12: () = instantiation_input(a, instantiation=u0, port_name=a, id=12)
  y: () = output_port(instantiation_output.11, name=y, id=13)
}
"#;
    let f = run_block2fn(block_ir, &[], &[]);
    assert_return_matches(&f, "not(get_param(name=\"a\"))");
}

#[test]
fn e2e_register_collapse_with_tied_input() {
    let block_ir = r#"package test

top block top(a: bits[1], out: bits[1]) {
  reg r(bits[1])
  a: bits[1] = input_port(name=a, id=1)
  r_q: bits[1] = register_read(register=r, id=2)
  r_d: () = register_write(a, register=r, id=3)
  out: () = output_port(r_q, name=out, id=4)
}
"#;
    let f = run_block2fn(block_ir, &[("a", "bits[1]:1")], &[]);
    assert_return_matches(&f, "literal(1)");
}

#[test]
fn e2e_drop_output_to_single() {
    let block_ir = r#"package test

top block top(a: bits[1], b: bits[1], out0: bits[1], out1: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  b: bits[1] = input_port(name=b, id=2)
  out0: () = output_port(a, name=out0, id=3)
  out1: () = output_port(b, name=out1, id=4)
}
"#;
    let f = run_block2fn(block_ir, &[], &["out1"]);
    assert_return_matches(&f, "get_param(name=\"a\")");
}

const NETLIST_LOAD_ENABLE_BLOCK: &str = r#"package test

block AND2(A: bits[1], B: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  B: bits[1] = input_port(name=B, id=2)
  myand: bits[1] = and(A, B, id=3)
  Y: () = output_port(and.3, name=Y, id=4)
}

block OR2(A: bits[1], B: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=11)
  B: bits[1] = input_port(name=B, id=12)
  myor: bits[1] = or(A, B, id=13)
  Y: () = output_port(or.13, name=Y, id=14)
}

block INV(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=21)
  not.22: bits[1] = not(A, id=22)
  Y: () = output_port(not.22, name=Y, id=23)
}

block DFF(CLK: bits[1], D: bits[1], Q: bits[1]) {
  reg Q_reg(bits[1])
  CLK: bits[1] = input_port(name=CLK, id=31)
  D: bits[1] = input_port(name=D, id=32)
  Q_q: bits[1] = register_read(register=Q_reg, id=33)
  Q_d: () = register_write(D, register=Q_reg, id=34)
  Q: () = output_port(Q_q, name=Q, id=35)
}

// Structure:
//   q_next <= (valid & data) | (!valid & q)
//   out = q_next
top block top(clk: bits[1], data: bits[1], valid: bits[1], out: bits[1]) {
  instantiation u_and_data(block=AND2, kind=block)
  instantiation u_and_hold(block=AND2, kind=block)
  instantiation u_or(block=OR2, kind=block)
  instantiation u_inv(block=INV, kind=block)
  instantiation u_dff(block=DFF, kind=block)

  clk: bits[1] = input_port(name=clk, id=101)
  data: bits[1] = input_port(name=data, id=102)
  valid: bits[1] = input_port(name=valid, id=103)

  inv_valid: bits[1] = instantiation_output(instantiation=u_inv, port_name=Y, id=110)
  instantiation_input.111: () = instantiation_input(valid, instantiation=u_inv, port_name=A, id=111)

  q: bits[1] = instantiation_output(instantiation=u_dff, port_name=Q, id=112)
  instantiation_input.113: () = instantiation_input(clk, instantiation=u_dff, port_name=CLK, id=113)

  and_data: bits[1] = instantiation_output(instantiation=u_and_data, port_name=Y, id=115)
  instantiation_input.116: () = instantiation_input(valid, instantiation=u_and_data, port_name=A, id=116)
  instantiation_input.117: () = instantiation_input(data, instantiation=u_and_data, port_name=B, id=117)

  and_hold: bits[1] = instantiation_output(instantiation=u_and_hold, port_name=Y, id=118)
  instantiation_input.119: () = instantiation_input(inv_valid, instantiation=u_and_hold, port_name=A, id=119)
  instantiation_input.120: () = instantiation_input(q, instantiation=u_and_hold, port_name=B, id=120)

  or_out: bits[1] = instantiation_output(instantiation=u_or, port_name=Y, id=121)
  instantiation_input.122: () = instantiation_input(and_data, instantiation=u_or, port_name=A, id=122)
  instantiation_input.123: () = instantiation_input(and_hold, instantiation=u_or, port_name=B, id=123)
  instantiation_input.114: () = instantiation_input(or_out, instantiation=u_dff, port_name=D, id=114)

  out: () = output_port(or_out, name=out, id=124)
}
"#;

#[test]
fn e2e_netlist_load_enable_feedback_elided() {
    let f = run_block2fn(NETLIST_LOAD_ENABLE_BLOCK, &[("valid", "bits[1]:1")], &[]);
    assert_eq!(f.params.len(), 2, "valid should be tied off");
    assert_eq!(f.params[0].name, "clk");
    assert_eq!(f.params[1].name, "data");
    for node in f.nodes.iter() {
        assert!(
            !matches!(
                node.payload,
                NodePayload::RegisterRead { .. } | NodePayload::RegisterWrite { .. }
            ),
            "register nodes should be removed when valid is constant"
        );
    }
}

#[test]
fn e2e_clock_header_converts_without_clock_flag() {
    let block_ir = r#"package test

top block top(clk: clock, data: bits[1], out: bits[1]) {
  reg r(bits[1])
  data: bits[1] = input_port(name=data, id=1)
  q: bits[1] = register_read(register=r, id=2)
  d: () = register_write(data, register=r, id=3)
  out: () = output_port(q, name=out, id=4)
}
"#;
    let f = run_block2fn(block_ir, &[], &[]);
    assert_eq!(f.params.len(), 1, "clock should not appear as data param");
    assert_eq!(f.params[0].name, "data");
    assert_return_matches(&f, "get_param(name=\"data\")");
}

#[test]
fn e2e_netlist_load_enable_feedback_cycle_errors() {
    let opts = Block2FnOptions {
        tie_input_ports: BTreeMap::new(),
        drop_output_ports: BTreeSet::new(),
    };
    let err = block_ir_to_fn(NETLIST_LOAD_ENABLE_BLOCK, &opts).expect_err("expected error");
    assert_eq!(
        err, "block2fn: cycle detected: u_and_hold__myand -> u_or__myor -> u_and_hold__myand",
        "unexpected error: {err}"
    );
}
