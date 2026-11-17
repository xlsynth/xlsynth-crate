// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use crate::block2fn::{
    Block2FnOptions, block_ir_to_fn, block_ir_to_fn_package, block_package_to_fn_package,
};
use crate::ir::NodePayload;
use crate::ir_parser::Parser;
use crate::ir_query::{matches_node, parse_query};
use crate::{IrBits, IrValue};

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
fn projected_top_preserves_calls_to_an_eponymous_function() {
    for name in ["flip", "step"] {
        let source = format!(
            r#"package helpers
fn flip(x: bits[1] id=1) -> bits[1] {{
  ret n: bits[1] = not(x, id=2)
}}
fn step(i: bits[2] id=3, x: bits[1] id=4) -> bits[1] {{
  ret n: bits[1] = invoke(x, to_apply=flip, id=5)
}}
fn {name}__callee(x: bits[1] id=6) -> bits[1] {{
  ret x: bits[1] = param(name=x, id=6)
}}
top block {name}(x: bits[1], y: bits[3]) {{
  x: bits[1] = input_port(name=x, id=7)
  n: bits[1] = invoke(x, to_apply=flip, id=8)
  a: bits[1][1] = array(x, id=9)
  mapped: bits[1][1] = map(a, to_apply=flip, id=10)
  zero: bits[1] = literal(value=0, id=11)
  first: bits[1] = array_index(mapped, indices=[zero], id=12)
  loop: bits[1] = counted_for(n, trip_count=1, stride=1, body=step, id=13)
  carried: bits[1] = invoke(loop, to_apply={name}__callee, id=14)
  word: bits[3] = concat(first, carried, n, id=15)
  y: () = output_port(word, name=y, id=16)
}}"#
        );
        let input = Parser::new(&source).parse_and_verify_package().unwrap();
        let original = input.to_string();
        let options = Block2FnOptions::default();
        let output = block_package_to_fn_package(&input, &options)
            .unwrap()
            .to_string();
        assert_eq!(input.to_string(), original);
        assert_eq!(
            block_ir_to_fn_package(&source, &options)
                .unwrap()
                .to_string(),
            output
        );

        let package = Parser::new(&output).parse_and_verify_package().unwrap();
        let entry = package.get_top_fn().unwrap();
        assert_eq!(entry.name, name);
        for (input, expected) in [(0, 0b101), (1, 0b010)] {
            let input = IrValue::make_ubits(1, input).unwrap();
            let value = crate::ir_eval::eval_fn_in_package_value(&package, entry, &[input]);
            assert_eq!(value, IrValue::make_ubits(3, expected).unwrap());
        }
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
    assert_eq!(f.get_param(0).param_name(), "a");
    assert_return_matches(&f, "param(name=\"a\")");
}

#[test]
fn empty_array_alias_chain_forwards_its_input_and_resolves_a_closed_loop() {
    let source = r#"package empty_ports
block leaf(d: bits[0][2], q: bits[0][2]) {
  d: bits[0][2] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block wrapper(data: bits[0][2], result: bits[0][2]) {
  instantiation a(block=leaf, kind=block)
  instantiation b(block=leaf, kind=block)
  data: bits[0][2] = input_port(name=data, id=3)
  qa: bits[0][2] = instantiation_output(instantiation=a, port_name=q, id=4)
  qb: bits[0][2] = instantiation_output(instantiation=b, port_name=q, id=5)
  da: () = instantiation_input(data, instantiation=a, port_name=d, id=6)
  db: () = instantiation_input(qa, instantiation=b, port_name=d, id=7)
  result: () = output_port(qb, name=result, id=8)
}"#;
    let direct = run_block2fn(source, &[], &[]);
    assert_return_matches(&direct, "param(name=\"data\")");

    let feedback = source.replace("instantiation_input(data,", "instantiation_input(qb,");
    let cyclic = run_block2fn(&feedback, &[], &[]);
    let zero = IrValue::make_ubits(0, 0).unwrap();
    let pair = IrValue::make_array(&[zero.clone(), zero]).unwrap();
    let crate::ir_eval::FnEvalResult::Success(result) =
        crate::ir_eval::eval_fn(&cyclic, std::slice::from_ref(&pair))
    else {
        panic!("failed to evaluate the closed empty-data loop");
    };
    assert_eq!(result.value, pair);
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
    assert_return_matches(&f, "not(param(name=\"a\"))");
}

#[test]
fn e2e_register_collapse_with_tied_input() {
    let block_ir = r#"package test

top block top(clk: clock, a: bits[1], out: bits[1]) {
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
    assert_return_matches(&f, "param(name=\"a\")");
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

block DFF(CLK: clock, D: bits[1], Q: bits[1]) {
  reg Q_reg(bits[1])
  D: bits[1] = input_port(name=D, id=32)
  Q_q: bits[1] = register_read(register=Q_reg, id=33)
  Q_d: () = register_write(D, register=Q_reg, id=34)
  Q: () = output_port(Q_q, name=Q, id=35)
}

// Structure:
//   q_next <= (valid & data) | (!valid & q)
//   out = q_next
top block top(clk: clock, data: bits[1], valid: bits[1], out: bits[1]) {
  instantiation u_and_data(block=AND2, kind=block)
  instantiation u_and_hold(block=AND2, kind=block)
  instantiation u_or(block=OR2, kind=block)
  instantiation u_inv(block=INV, kind=block)
  instantiation u_dff(block=DFF, kind=block)

  data: bits[1] = input_port(name=data, id=102)
  valid: bits[1] = input_port(name=valid, id=103)

  inv_valid: bits[1] = instantiation_output(instantiation=u_inv, port_name=Y, id=110)
  instantiation_input.111: () = instantiation_input(valid, instantiation=u_inv, port_name=A, id=111)

  q: bits[1] = instantiation_output(instantiation=u_dff, port_name=Q, id=112)

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
    assert_eq!(
        f.params.len(),
        1,
        "valid is tied off and the clock is not a data parameter"
    );
    assert_eq!(f.get_param(0).param_name(), "data");
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
    assert_eq!(f.get_param(0).param_name(), "data");
    assert_return_matches(&f, "param(name=\"data\")");
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

#[test]
fn tied_input_controls_feedback_during_function_projection() {
    let source = r#"package tied
block shift(d: bits[3], amount: bits[3], q: bits[3]) {
  d: bits[3] = input_port(name=d, id=1)
  amount: bits[3] = input_port(name=amount, id=2)
  shifted: bits[3] = shll(d, amount, id=3)
  q: () = output_port(shifted, name=q, id=4)
}
top block wrapper(amount: bits[3], result: bits[3]) {
  instantiation u(block=shift, kind=block)
  amount: bits[3] = input_port(name=amount, id=5)
  q: bits[3] = instantiation_output(instantiation=u, port_name=q, id=6)
  data: () = instantiation_input(q, instantiation=u, port_name=d, id=7)
  control: () = instantiation_input(amount, instantiation=u, port_name=amount, id=8)
  result: () = output_port(q, name=result, id=9)
}"#;
    for amount in [None, Some(0), Some(1)] {
        let mut options = Block2FnOptions::default();
        if let Some(amount) = amount {
            options.tie_input_ports.insert(
                "amount".into(),
                IrValue::make_ubits(3, amount).unwrap().to_bits().unwrap(),
            );
        }
        let projected = block_ir_to_fn(source, &options);
        if amount == Some(1) {
            assert_return_matches(&projected.unwrap().function, "literal(0)");
        } else {
            assert!(projected.unwrap_err().contains("cycle"));
        }
    }
}
