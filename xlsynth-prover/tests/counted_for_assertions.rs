// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use std::collections::HashMap;
use test_case::test_case;
use xlsynth_pir::{ir_parser::Parser, ir_verify::verify_package};
use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule, run_ir_equiv};
use xlsynth_prover::prover::types::{AssertionSemantics, EquivResult};

#[test_case(0, 1, &[], true; "zero trips")]
#[test_case(1, 1, &[0], true; "one trip")]
#[test_case(3, 2, &[0, 2, 4], true; "each strided iteration")]
#[test_case(3, 2, &[], false; "loop asserts while the other side always succeeds")]
#[test_case(3, 2, &[0, 4], false; "an assertion only in the middle iteration")]
#[test_case(3, 2, &[0, 2, 3, 4], false; "an assertion on an unvisited index")]
fn counted_for_matches_explicit_calls(trips: usize, stride: usize, indices: &[usize], same: bool) {
    let rolled = format!(
        "  last: token = counted_for(t, trip_count={trips}, stride={stride}, body=body, invariant_args=[bad], id=20)\n"
    );
    let mut unrolled = String::new();
    let mut last = String::from("t");
    for (step, index) in indices.iter().enumerate() {
        let index_id = 100 + 2 * step;
        let call_id = index_id + 1;
        unrolled.push_str(&format!(
            "  i{step}: bits[3] = literal(value={index}, id={index_id})\n  t{step}: token = invoke(i{step}, {last}, bad, to_apply=body, id={call_id})\n"
        ));
        last = format!("t{step}");
    }
    let package = format!(
        r#"package counted_assertions

fn check(i: bits[3] id=1, t: token id=2, bad: bits[3] id=3) -> token {{
  ok: bits[1] = ne(i, bad, id=5)
  ret checked: token = assert(t, ok, message="visited forbidden index", label="index_check", id=6)
}}

fn body(i: bits[3] id=7, t: token id=8, bad: bits[3] id=9) -> token {{
  ret checked: token = invoke(i, t, bad, to_apply=check, id=12)
}}

fn rolled(t: token id=13, bad: bits[3] id=14) -> (token, bits[3]) {{
{rolled}  ret result: (token, bits[3]) = tuple(last, bad, id=21)
}}

fn unrolled(t: token id=22, bad: bits[3] id=23) -> (token, bits[3]) {{
{unrolled}  ret result: (token, bits[3]) = tuple({last}, bad, id=200)
}}
"#
    );
    let parsed = Parser::new(&package).parse_package().unwrap();
    verify_package(&parsed).unwrap();
    for (left, right) in [("rolled", "unrolled"), ("unrolled", "rolled")] {
        let request = IrEquivRequest::new(
            IrModule::new(&package).with_top(Some(left)),
            IrModule::new(&package).with_top(Some(right)),
        )
        .with_assertion_semantics(AssertionSemantics::Same);
        let result = run_ir_equiv(&request).unwrap().result;
        assert!(
            matches!(
                (same, &result),
                (true, EquivResult::Proved) | (false, EquivResult::Disproved { .. })
            ),
            "{left} vs {right}: {result:?}"
        );
    }
}

#[test_case(AssertionSemantics::Assume)]
#[test_case(AssertionSemantics::Same)]
#[test_case(AssertionSemantics::Ignore)]
fn counted_for_body_uf_exposes_output_difference(semantics: AssertionSemantics) {
    let text = r#"package counted_body_uf

fn body(i: bits[3] id=1, t: token id=2, bad: bits[3] id=3) -> token {
  ok: bits[1] = ne(i, bad, id=4)
  ret checked: token = assert(t, ok, message="visited forbidden index", label="index_check", id=5)
}

fn rolled(t: token id=13, bad: bits[3] id=14) -> (token, bits[1]) {
  last: token = counted_for(t, trip_count=1, stride=1, body=body, invariant_args=[bad], id=20)
  off: bits[1] = literal(value=0, id=21)
  ret result: (token, bits[1]) = tuple(last, off, id=22)
}

fn explicit(t: token id=23, bad: bits[3] id=24) -> (token, bits[1]) {
  zero: bits[3] = literal(value=0, id=25)
  last: token = invoke(zero, t, bad, to_apply=body, id=26)
  on: bits[1] = eq(bad, zero, id=27)
  ret result: (token, bits[1]) = tuple(last, on, id=28)
}
"#;
    verify_package(&Parser::new(text).parse_package().unwrap()).unwrap();
    let rolled = IrModule::new(text).with_top(Some("rolled"));
    let explicit = IrModule::new(text).with_top(Some("explicit"));
    let uf_map = HashMap::from([("body".to_string(), "abstract_body".to_string())]);
    for mapping in [HashMap::new(), uf_map] {
        let request = IrEquivRequest::new(
            rolled.clone().with_uf_map(&mapping),
            explicit.clone().with_uf_map(&mapping),
        )
        .with_assertion_semantics(semantics);
        let result = run_ir_equiv(&request).unwrap().result;
        // The outputs differ only at bad == 0, where the concrete body asserts.
        if mapping.is_empty() && semantics != AssertionSemantics::Ignore {
            assert_eq!(result, EquivResult::Proved);
        } else {
            let EquivResult::Disproved {
                lhs_inputs,
                lhs_output,
                rhs_output,
                ..
            } = result
            else {
                panic!("expected counterexample at bad == 0, got {result:?}");
            };
            assert!(lhs_inputs[1].value.bits_equals_u64_value(0));
            assert!(lhs_output.assertion_violation.is_none());
            assert!(rhs_output.assertion_violation.is_none());
            assert_ne!(lhs_output.value, rhs_output.value);
        }
    }
}

#[test]
fn counted_for_uses_uf_result_as_next_carry() {
    let text = r#"package counted_uf_carry

fn body(i: bits[3] id=1, carry: (token, bits[3]) id=2) -> (token, bits[3]) {
  ret same: (token, bits[3]) = identity(carry, id=3)
}

fn rolled(init: (token, bits[3]) id=4) -> (token, bits[3]) {
  ret last: (token, bits[3]) = counted_for(init, trip_count=2, stride=3, body=body, id=5)
}

fn explicit(init: (token, bits[3]) id=6) -> (token, bits[3]) {
  zero: bits[3] = literal(value=0, id=7)
  first: (token, bits[3]) = invoke(zero, init, to_apply=body, id=8)
  three: bits[3] = literal(value=3, id=9)
  ret last: (token, bits[3]) = invoke(three, first, to_apply=body, id=10)
}
"#;
    verify_package(&Parser::new(text).parse_package().unwrap()).unwrap();
    let uf_map = HashMap::from([("body".to_string(), "step".to_string())]);
    let rolled = IrModule::new(text)
        .with_top(Some("rolled"))
        .with_uf_map(&uf_map);
    let explicit = IrModule::new(text).with_top(Some("explicit"));
    let concrete = IrEquivRequest::new(rolled.clone(), explicit.clone());
    let result = run_ir_equiv(&concrete).unwrap().result;
    assert!(
        matches!(result, EquivResult::Disproved { .. }),
        "{result:?}"
    );

    let abstracted = IrEquivRequest::new(rolled, explicit.with_uf_map(&uf_map));
    assert_eq!(
        run_ir_equiv(&abstracted).unwrap().result,
        EquivResult::Proved
    );
}

#[test]
fn itok_prefix_on_loop_body_preserves_carry() {
    let text = r#"package prefixed_body

fn step(z: bits[1] id=1) -> bits[1] {
  ret result: bits[1] = identity(z, id=2)
}

fn __itokstep(i: bits[1] id=3, carry: bits[1] id=4, z: bits[1] id=5) -> bits[1] {
  ret result: bits[1] = identity(carry, id=6)
}

fn left(x: bits[1] id=7, y: bits[1] id=8, z: bits[1] id=9) -> (bits[1], bits[1]) {
  last: bits[1] = counted_for(x, trip_count=1, stride=1, body=__itokstep, invariant_args=[z], id=10)
  mapped: bits[1] = invoke(z, to_apply=step, id=11)
  ret pair: (bits[1], bits[1]) = tuple(last, mapped, id=12)
}

fn right(x: bits[1] id=13, y: bits[1] id=14, z: bits[1] id=15) -> (bits[1], bits[1]) {
  last: bits[1] = counted_for(y, trip_count=1, stride=1, body=__itokstep, invariant_args=[z], id=16)
  mapped: bits[1] = invoke(z, to_apply=step, id=17)
  ret pair: (bits[1], bits[1]) = tuple(last, mapped, id=18)
}
"#;
    verify_package(&Parser::new(text).parse_package().unwrap()).unwrap();
    let uf_map = HashMap::from([("step".to_string(), "F".to_string())]);
    let module = IrModule::new(text).with_uf_map(&uf_map);
    let request = IrEquivRequest::new(
        module.clone().with_top(Some("left")),
        module.with_top(Some("right")),
    );
    let result = run_ir_equiv(&request).unwrap().result;
    let EquivResult::Disproved {
        lhs_inputs,
        lhs_output,
        rhs_output,
        ..
    } = result
    else {
        panic!("expected different loop carries, got {result:?}");
    };
    assert_ne!(lhs_inputs[0].value, lhs_inputs[1].value);
    assert_eq!(
        lhs_output.value.get_element(0).unwrap(),
        lhs_inputs[0].value
    );
    assert_eq!(
        rhs_output.value.get_element(0).unwrap(),
        lhs_inputs[1].value
    );
    assert_eq!(
        lhs_output.value.get_element(1).unwrap(),
        rhs_output.value.get_element(1).unwrap()
    );
}
