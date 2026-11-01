// SPDX-License-Identifier: Apache-2.0

use super::{Cell, FUNCTION_WORK, Plan, REGION, analyze};
use std::fmt::Write;
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
fn unary_wrappers_share_the_predicate_window() {
    let mut ir = String::from(
        r#"package p
top fn f(k: bits[1], x: bits[2]) -> bits[1] {
  zero: bits[2] = literal(value=0, id=3)
  word: bits[2] = sel(k, cases=[zero, x], id=4)
  pred0: bits[1] = or_reduce(word, id=5)
"#,
    );
    // Walking from every wrapper would exhaust the budget before reaching
    // pred0.
    let wrappers = REGION + FUNCTION_WORK.div_ceil(REGION);
    for i in 1..=wrappers {
        let ret = if i == wrappers { "ret " } else { "" };
        writeln!(
            ir,
            "  {ret}pred{i}: bits[1] = identity(pred{}, id={})",
            i - 1,
            i + 5
        )
        .unwrap();
    }
    ir.push('}');
    let package = Parser::new(&ir).parse_and_validate_package().unwrap();
    let f = package.get_top_fn().unwrap();
    let result = analyze(f, true);
    assert_eq!(result.plans.len(), 1);
    assert_eq!(f.get_node(result.plans[0].root).text_id, 5);
    assert!(result.visits < REGION);
}

#[test]
fn wide_default_literal_is_scanned_once_for_all_assignments() {
    let visits = [64usize, 65_536].map(|width| {
        let ir = format!(
            r#"package p
top fn f(k: bits[8], x: bits[{width}]) -> bits[1] {{
  lit: bits[{width}] = literal(value=1, id=3)
  word: bits[{width}] = sel(k, cases=[x], default=lit, id=4)
  ret pred: bits[1] = or_reduce(word, id=5)
}}"#
        );
        let package = Parser::new(&ir).parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();
        let result = analyze(f, true);
        let cells = &result.plans[0].cells;
        assert_eq!(cells.len(), 256);
        assert_eq!(cells[0], Cell::Any(vec![(f.params[1], 0, width)]));
        assert!(cells[1..].iter().all(|cell| cell == &Cell::Const(true)));
        result.visits
    });
    assert_eq!(visits[1] - visits[0], 65_536 / 64 - 1);
}

#[test]
fn repeated_operand_visits_exhaust_the_function_budget() {
    for op in ["or", "concat"] {
        let mut args = vec![if op == "or" { "x" } else { "empty" }; REGION];
        args[0] = "x";
        let args = args.join(", ");
        let ir = format!(
            r#"package p
top fn f(k: bits[8], x: bits[2]) -> bits[1] {{
  empty: bits[0] = literal(value=0, id=3)
  zero: bits[2] = literal(value=0, id=4)
  repeated: bits[2] = {op}({args}, id=5)
  word: bits[2] = sel(k, cases=[zero], default=repeated, id=6)
  ret pred: bits[1] = or_reduce(word, id=7)
}}"#
        );
        let package = Parser::new(&ir).parse_and_validate_package().unwrap();
        let result = analyze(package.get_top_fn().unwrap(), true);
        assert!(result.plans.is_empty(), "{op}");
        assert_eq!(result.reasons.0[super::Why::Work as usize], 1, "{op}");
    }
}

#[test]
fn wide_comparison_and_shift_literals_are_classified_once() {
    for shift in [false, true] {
        let visits = [64usize, 65_536].map(|width| {
            let ir = if shift {
                format!(
                    r#"package p
top fn f(k: bits[8], x: bits[2]) -> bits[1] {{
  amount: bits[{width}] = literal(value=1, id=3)
  default_word: bits[2] = shrl(x, amount, id=4)
  zero: bits[2] = literal(value=0, id=5)
  word: bits[2] = sel(k, cases=[zero], default=default_word, id=6)
  ret pred: bits[1] = or_reduce(word, id=7)
}}"#
                )
            } else {
                format!(
                    r#"package p
top fn f(k: bits[8], x: bits[{width}]) -> bits[1] {{
  zero: bits[{width}] = literal(value=0, id=3)
  nz: bits[1] = ne(x, zero, id=4)
  off: bits[1] = literal(value=0, id=5)
  ret pred: bits[1] = sel(k, cases=[off], default=nz, id=6)
}}"#
                )
            };
            let package = Parser::new(&ir).parse_and_validate_package().unwrap();
            let f = package.get_top_fn().unwrap();
            let result = analyze(f, true);
            let cells = &result.plans[0].cells;
            let (lo, hi) = if shift { (1, 2) } else { (0, width) };
            assert_eq!(cells.len(), 256);
            assert_eq!(cells[0], Cell::Const(false));
            let default = Cell::Any(vec![(f.params[1], lo, hi)]);
            assert!(cells[1..].iter().all(|c| c == &default));
            result.visits
        });
        assert_eq!(visits[1] - visits[0], 65_536 / 64 - 1, "shift={shift}");
    }
}

#[test]
fn literal_observation_masks_partial_limbs() {
    // Only bit 64 is set. Exercise both edge masks and a full interior limb.
    for (lo, hi) in [(0, 64), (1, 64), (1, 65), (64, 66), (65, 129), (0, 129)] {
        let width = hi - lo;
        let ir = format!(
            r#"package p
top fn f(k: bits[1], x: bits[129]) -> bits[1] {{
  lit: bits[129] = literal(value=0x10000000000000000, id=3)
  word: bits[129] = sel(k, cases=[x, lit], id=4)
  slice: bits[{width}] = bit_slice(word, start={lo}, width={width}, id=5)
  ret pred: bits[1] = or_reduce(slice, id=6)
}}"#
        );
        let plans = plans(&ir);
        assert_eq!(plans[0].cells[1], Cell::Const((lo..hi).contains(&64)));
    }
}
