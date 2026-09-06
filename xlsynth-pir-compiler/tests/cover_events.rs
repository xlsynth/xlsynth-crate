// SPDX-License-Identifier: Apache-2.0

//! Cover events distinguish encountered false predicates from skipped sites.

use xlsynth_pir::IrValue;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::{ExecutionOptions, PirFunctionCompiler};

/// Checks exact cover-site membership and counts against both the contract and
/// PIR.
fn assert_cover_counts(ir: &str, args: &[IrValue], expected: &[(usize, &str, u64)]) {
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let FnEvalResult::Success(interpreted) = eval_fn_in_package(&package, function, args) else {
        panic!("cover-only package must evaluate without failures");
    };
    let compiled = PirFunctionCompiler::compile_package(&package).unwrap();
    let actual = compiled
        .run_ir_values_with_events(args, ExecutionOptions::collect_all())
        .unwrap();
    assert_eq!(actual.value, interpreted.value);
    assert!(actual.events.assertion_failures.is_empty());
    assert!(actual.events.assumption_failures.is_empty());
    assert!(actual.events.trace_messages.is_empty());
    let mut actual_covers: Vec<_> = actual
        .events
        .cover_counts
        .iter()
        .map(|cover| (cover.node_text_id, cover.label.as_str(), cover.count))
        .collect();
    let mut interpreted_covers: Vec<_> = interpreted
        .cover_counts
        .iter()
        .map(|cover| (cover.node_text_id, cover.label.as_str(), cover.count))
        .collect();
    actual_covers.sort_unstable();
    interpreted_covers.sort_unstable();
    assert_eq!(interpreted_covers, expected);
    assert_eq!(actual_covers, expected);
}

#[test]
fn zero_trip_token_loop_omits_unexecuted_cover() {
    assert_cover_counts(
        r#"package test

fn body(i: bits[1] id=1, token: token id=2) -> token {
  ret token: token = param(name=token, id=2)
  cv: () = cover(i, label="body_cover", id=3)
}

top fn f() -> token {
  token: token = after_all(id=4)
  ret result: token = counted_for(token, trip_count=0, stride=0, body=body, id=5)
}
"#,
        &[],
        &[],
    );
}

#[test]
fn loop_callee_covers_distinguish_skipped_false_and_true_predicates() {
    for trip_count in [0, 1, 3] {
        for predicate in [0, 1] {
            let ir = format!(
                r#"package test

fn observed(active: bits[1] id=1) -> bits[1] {{
  cv: () = cover(active, label="callee_cover", id=2)
  ret result: bits[1] = identity(active, id=3)
}}

fn body(i: bits[2] id=4, carry: bits[1] id=5) -> bits[1] {{
  ret result: bits[1] = invoke(carry, to_apply=observed, id=6)
}}

top fn f(active: bits[1] id=7) -> bits[1] {{
  ret result: bits[1] = counted_for(active, trip_count={trip_count}, stride=1, body=body, id=8)
}}
"#
            );
            let expected = if trip_count == 0 {
                Vec::new()
            } else {
                vec![(2, "callee_cover", trip_count * predicate)]
            };
            assert_cover_counts(
                &ir,
                &[IrValue::make_ubits(1, predicate).unwrap()],
                &expected,
            );
        }
    }
}

#[test]
fn nested_zero_trip_loop_omits_deep_callee_cover() {
    assert_cover_counts(
        r#"package test

fn observed(active: bits[1] id=1) -> bits[1] {
  cv: () = cover(active, label="callee_cover", id=2)
  ret result: bits[1] = identity(active, id=3)
}

fn inner(i: bits[1] id=4, carry: bits[1] id=5) -> bits[1] {
  ret result: bits[1] = invoke(carry, to_apply=observed, id=6)
}

fn outer(i: bits[2] id=7, carry: bits[1] id=8) -> bits[1] {
  ret result: bits[1] = counted_for(carry, trip_count=0, stride=0, body=inner, id=9)
}

top fn f(active: bits[1] id=10) -> bits[1] {
  ret result: bits[1] = counted_for(active, trip_count=3, stride=1, body=outer, id=11)
}
"#,
        &[IrValue::make_ubits(1, 1).unwrap()],
        &[],
    );
}

#[test]
fn skipped_loop_preserves_counts_from_the_same_callee_invoked_elsewhere() {
    for predicate in [0, 1] {
        assert_cover_counts(
            r#"package test

fn observed(active: bits[1] id=1) -> bits[1] {
  cv: () = cover(active, label="callee_cover", id=2)
  ret result: bits[1] = identity(active, id=3)
}

fn body(i: bits[1] id=4, carry: bits[1] id=5) -> bits[1] {
  ret result: bits[1] = invoke(carry, to_apply=observed, id=6)
}

top fn f(active: bits[1] id=7) -> bits[1] {
  called: bits[1] = invoke(active, to_apply=observed, id=8)
  ret result: bits[1] = counted_for(called, trip_count=0, stride=0, body=body, id=9)
}
"#,
            &[IrValue::make_ubits(1, predicate).unwrap()],
            &[(2, "callee_cover", predicate)],
        );
    }
}
