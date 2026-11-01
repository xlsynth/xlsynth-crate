// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

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
