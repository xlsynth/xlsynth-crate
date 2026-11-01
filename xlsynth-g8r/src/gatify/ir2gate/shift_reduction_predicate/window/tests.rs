// SPDX-License-Identifier: Apache-2.0

use super::{Cell, Plan, analyze};
use xlsynth_pir::ir_parser::Parser;

fn plans(ir: &str) -> Vec<Plan> {
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    analyze(package.get_top_fn().unwrap(), true).plans
}

#[test]
fn derived_selector_and_shift_share_the_same_control_domain() {
    let ir = "package p
      fn f(x: bits[7] id=1, k: bits[3] id=2) -> bits[1] {
        zero: bits[3] = literal(value=0, id=3)
        enabled: bits[1] = ne(k, zero, id=4)
        shifted: bits[7] = shrl(x, k, id=5)
        chosen: bits[7] = sel(enabled, cases=[x, shifted], id=6)
        observed: bits[3] = bit_slice(chosen, start=2, width=3, id=7)
        ret result: bits[1] = or_reduce(observed, id=8)
      }";
    let plans = plans(ir);
    assert_eq!(plans.len(), 1);
    let plan = &plans[0];
    assert_eq!(plan.cells.len(), 8);
    for (k, cell) in plan.cells.iter().enumerate() {
        for x in 0u64..128 {
            let result = match cell {
                Cell::Const(value) => *value,
                Cell::Any(spans) => spans
                    .iter()
                    .any(|(_, low, high)| ((x >> low) & ((1 << (high - low)) - 1)) != 0),
            };
            assert_eq!(result, ((x >> k) >> 2) & 7 != 0);
        }
    }
}

#[test]
fn priority_order_and_saturating_shift_counts() {
    for (amount, expected_high_span) in [("1", Some((5, 6))), ("0x10000000000000000", None)] {
        let ir = format!(
            "package p
          fn f(x: bits[6] id=1, k: bits[2] id=2) -> bits[1] {{
            lo: bits[2] = bit_slice(x, start=0, width=2, id=3)
            hi: bits[2] = bit_slice(x, start=4, width=2, id=4)
            zero: bits[2] = literal(value=0, id=5)
            amount: bits[129] = literal(value={amount}, id=6)
            shifted: bits[2] = shrl(hi, amount, id=7)
            choice: bits[2] = priority_sel(k, cases=[lo, shifted], default=zero, id=8)
            ret result: bits[1] = ne(zero, choice, id=9)
          }}"
        );
        let plans = plans(&ir);
        assert_eq!(plans.len(), 1);
        let cells = &plans[0].cells;
        assert_eq!(cells.len(), 4);
        assert_eq!(cells[0], Cell::Const(false));
        assert_eq!(cells[1], cells[3]);
        match (&cells[2], expected_high_span) {
            (Cell::Const(false), None) => (),
            (Cell::Any(spans), Some(expected)) => {
                assert_eq!(spans.len(), 1);
                assert_eq!((spans[0].1, spans[0].2), expected);
            }
            actual => panic!("unexpected data observation {actual:?}"),
        }
    }
}

#[test]
fn separately_returned_word_keeps_its_ordinary_mapping() {
    let ir = "package p
      fn f(x: bits[8] id=1, s: bits[1] id=2) -> (bits[1], bits[8]) {
        zero: bits[8] = literal(value=0, id=3)
        chosen: bits[8] = sel(s, cases=[x, zero], id=4)
        nz: bits[1] = ne(chosen, zero, id=5)
        ret observed: (bits[1], bits[8]) = tuple(nz, chosen, id=6)
      }";
    assert!(plans(ir).is_empty());
}

#[test]
fn source_guard_is_observed_on_the_entire_domain_and_may_escape_as_a_scalar() {
    let ir = "package p
      fn f(x: bits[5] id=1, y: bits[5] id=2, c: bits[3] id=3) -> (bits[1], bits[1]) {
        zero: bits[3] = literal(value=0, id=4)
        g: bits[1] = sgt(c, zero, id=5)
        chosen: bits[5] = priority_sel(g, cases=[y], default=x, id=6)
        z: bits[1] = or_reduce(chosen, id=7)
        ret result: (bits[1], bits[1]) = tuple(z, g, id=8)
      }";
    let ps = plans(ir);
    let [plan] = ps.as_slice() else {
        panic!("missing or duplicated observation")
    };
    assert_eq!(
        plan.guard.as_ref().unwrap().values,
        [false, true, true, true, false, false, false, false]
    );
    let escaped = ir
        .replace("-> (bits[1], bits[1])", "-> (bits[1], bits[5])")
        .replace(
            "result: (bits[1], bits[1]) = tuple(z, g",
            "result: (bits[1], bits[5]) = tuple(z, chosen",
        );
    assert!(plans(&escaped).is_empty());
}

#[test]
fn different_nested_word_guards_keep_the_complete_observation_plan() {
    let ir = "package p
      fn f(x: bits[5] id=1, y: bits[5] id=2, c: bits[3] id=3) -> bits[1] {
        zero: bits[3] = literal(value=0, id=4)
        g: bits[1] = sgt(c, zero, id=5)
        h: bits[1] = bit_slice(c, start=0, width=1, id=6)
        inner: bits[5] = sel(h, cases=[x,y], id=7)
        outer: bits[5] = sel(g, cases=[inner,y], id=8)
        ret observed: bits[1] = or_reduce(outer, id=9)
      }";
    let ps = plans(ir);
    let [plan] = ps.as_slice() else {
        panic!("ordinary finite-domain plan lost")
    };
    assert!(plan.guard.is_none());
}
