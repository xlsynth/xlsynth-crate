// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::check_equivalence::{
    IrCheckResult, prove_same_gate_fn_via_ir_status_via_toolchain, validate_same_fn_via_toolchain,
};
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, IrRegion, gatify_prepared_fn, gatify_region};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::NodeRef;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn fixing_mode_parity_removes_a_redundant_payload_choice() {
    for (fixed, parity_op) in [(3, "identity"), (7, "not")] {
        for relation in ["eq", "ne"] {
            for expose_routes in [false, true] {
                let (tested, reference) =
                    parity_router(fixed, relation, parity_op, "compact", expose_routes);
                let tested = checked_gates(&tested);
                let reference = checked_gates(&reference);
                assert_eq!(
                    get_aig_stats(&tested).and_nodes,
                    get_aig_stats(&reference).and_nodes,
                    "fixed={fixed} relation={relation} expose_routes={expose_routes}",
                );
                assert_eq!(
                    prove_same_gate_fn_via_ir_status_via_toolchain(&tested, &reference),
                    IrCheckResult::Equivalent,
                );
            }
        }
    }
}

#[test]
fn a_single_wrong_route_or_a_different_guard_prevents_bypass() {
    for (relation, routing, witness, want_selected, want_general) in [
        ("eq", "glitch", 23, 9, 2),
        ("ne", "unrelated", 7, 2, 9),
        ("ult", "compact", 2, 2, 9),
        ("slt", "compact", 18, 9, 2),
    ] {
        let (tested, general) = parity_router(3, relation, "identity", routing, false);
        let args = [
            IrBits::make_ubits(5, witness).unwrap(),
            IrBits::make_ubits(4, 2).unwrap(),
            IrBits::make_ubits(4, 9).unwrap(),
        ];
        for (body, want) in [(tested, want_selected), (general, want_general)] {
            assert_eq!(
                gate_sim::eval(&checked_gates(&body), &args, Collect::None).outputs,
                vec![IrBits::make_ubits(4, want).unwrap()],
                "relation={relation} routing={routing}",
            );
        }
    }
}

#[test]
fn a_guard_supplied_by_a_region_caller_remains_independent() {
    let f = Parser::new(
        r#"fn main(mode: bits[2] id=1, a: bits[2] id=2, b: bits[2] id=3) -> bits[2] {
  high: bits[1] = bit_slice(mode, start=1, width=1, id=4)
  one: bits[1] = literal(value=1, id=5)
  enable: bits[1] = eq(one, high, id=6)
  lo: bits[1] = bit_slice(mode, start=0, width=1, id=7)
  hi: bits[1] = bit_slice(mode, start=1, width=1, id=8)
  both: bits[1] = and(hi, lo, id=9)
  full: bits[2] = sel(both, cases=[a, b], id=10)
  reduced: bits[2] = sel(lo, cases=[a, b], id=11)
  ret selected: bits[2] = sel(enable, cases=[full, reduced], id=12)
}"#,
    )
    .parse_fn()
    .unwrap();
    let node = |id| NodeRef {
        index: f.nodes.iter().position(|node| node.text_id == id).unwrap(),
    };
    let region = IrRegion {
        function: &f,
        nodes: &[node(7), node(8), node(9), node(10), node(11), node(12)],
        inputs: &[node(1), node(2), node(3), node(6)],
        outputs: &[node(12)],
    };
    let gates = gatify_region(region, GateBuilderOptions::opt()).unwrap();
    let reference = Parser::new(
        r#"fn reference(input_0: bits[2] id=1, input_1: bits[2] id=2,
             input_2: bits[2] id=3, input_3: bits[1] id=4) -> bits[2] {
  hi: bits[1] = bit_slice(input_0, start=1, width=1, id=5)
  lo: bits[1] = bit_slice(input_0, start=0, width=1, id=6)
  both: bits[1] = and(hi, lo, id=7)
  full: bits[2] = sel(both, cases=[input_1, input_2], id=8)
  reduced: bits[2] = sel(lo, cases=[input_1, input_2], id=9)
  ret selected: bits[2] = sel(input_3, cases=[full, reduced], id=10)
}"#,
    )
    .parse_fn()
    .unwrap();
    validate_same_fn_via_toolchain(&reference, &gates).unwrap();
}

fn parity_router(
    fixed: usize,
    relation: &str,
    parity_op: &str,
    routing: &str,
    expose_routes: bool,
) -> (String, String) {
    let predicate = match relation {
        "eq" => "eq(field, fixed, id=7)",
        "ne" => "ne(fixed, field, id=7)",
        "ult" => "ult(mode, bound, id=7)",
        "slt" => "slt(mode, bound, id=7)",
        _ => unreachable!(),
    };
    let choices = if relation == "ne" {
        "short, full"
    } else {
        "full, short"
    };
    let (width, result) = if expose_routes {
        (12, "concat(full, selected, short, id=20)")
    } else {
        (4, "identity(selected, id=20)")
    };
    // Fixing the middle three bits leaves the bottom and top mode bits free.
    // Their parity selects the same payload when adjusted by the fixed parity.
    let selected = format!("sel(enable, cases=[{choices}], id=19)");
    let ir = format!(
        r#"fn main(mode: bits[5] id=1, a: bits[4] id=2, b: bits[4] id=3) -> bits[{width}] {{
  field: bits[3] = bit_slice(mode, start=1, width=3, id=4)
  fixed: bits[3] = literal(value={fixed}, id=5)
  bound: bits[5] = literal(value=13, id=6)
  enable: bits[1] = {predicate}
  odd: bits[1] = xor_reduce(mode, id=8)
  low: bits[1] = bit_slice(mode, start=0, width=1, id=9)
  high: bits[1] = bit_slice(mode, start=4, width=1, id=10)
  outside: bits[1] = xor(low, high, id=11)
  compact: bits[1] = {parity_op}(outside, id=12)
  corner: bits[1] = and(low, high, id=13)
  glitch: bits[1] = xor(compact, corner, id=14)
  unrelated: bits[1] = bit_slice(a, start=0, width=1, id=15)
  full: bits[4] = sel(odd, cases=[a, b], id=16)
  short: bits[4] = sel({routing}, cases=[a, b], id=17)
  selected: bits[4] = {selected}
  ret result: bits[{width}] = {result}
}}"#
    );
    let general = ir.replace(&selected, "identity(full, id=19)");
    (ir, general)
}

fn checked_gates(text: &str) -> GateFn {
    let f = Parser::new(text).parse_fn().unwrap();
    let mapped = gatify_prepared_fn(
        &f,
        GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap();
    let live = dce(&mapped.gate_fn);
    validate_same_fn_via_toolchain(&f, &live).unwrap();
    live
}
