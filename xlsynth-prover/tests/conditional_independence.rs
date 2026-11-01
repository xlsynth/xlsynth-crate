// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use std::ops::Range;
use xlsynth_pir::{
    IrValue, ir,
    ir_eval::{FnEvalResult, eval_fn},
    ir_parser::Parser,
};
use xlsynth_prover::ir_operand_gate::PredicateReachability;
use xlsynth_prover::prover::conditional_independence::{
    BitSelection, IndependenceResult, InputBitSelection, prove_conditional_independence,
};
use xlsynth_prover::prover::{
    SolverLimits,
    types::{EquivResult, ProverFn},
};

fn bits(path: &[usize], range: Range<usize>) -> BitSelection {
    BitSelection {
        path: path.to_vec(),
        bits: range,
    }
}

fn vary(parameter: usize, path: &[usize], range: Range<usize>) -> InputBitSelection {
    InputBitSelection {
        parameter,
        selection: bits(path, range),
    }
}

/// Parses separately so function and guard may use unrelated parameter names.
fn function(text: &str) -> Result<ir::Fn, String> {
    let pkg = Parser::new(text)
        .parse_and_validate_package()
        .map_err(|e| e.to_string())?;
    Ok(pkg.get_top_fn().ok_or("no top")?.clone())
}

fn prove(
    f: &ir::Fn,
    guard: &ir::Fn,
    vary: &[InputBitSelection],
    observe: &[BitSelection],
) -> Result<IndependenceResult, String> {
    prove_conditional_independence(
        &ProverFn::new(f, None),
        &ProverFn::new(guard, None),
        vary,
        observe,
        SolverLimits::default(),
    )
}

/// Replays solver assignments through the independent PIR interpreter.
fn evaluate(f: &ir::Fn, args: &[IrValue]) -> IrValue {
    match eval_fn(f, args) {
        FnEvalResult::Success(result) => result.value,
        result => panic!("evaluation failed: {result:?}"),
    }
}

const IDENTITY: &str = r#"package test
top fn main(x: bits[8] id=1) -> bits[8] {
  ret result: bits[8] = identity(x, id=2)
}"#;
const TRUE: &str = r#"package test
top fn guard(renamed: bits[8] id=1) -> bits[1] {
  ret result: bits[1] = literal(value=1, id=2)
}"#;

#[test]
fn disabled_payload_is_independent_but_enabled_payload_is_not() {
    let f = function(
        r#"package test
top fn main(enable: bits[1] id=1, payload: bits[8] id=2) -> bits[8] {
  zero: bits[8] = literal(value=0, id=3)
  ret result: bits[8] = sel(enable, cases=[zero, payload], id=4)
}"#,
    )
    .unwrap();
    let disabled = function(
        r#"package test
top fn guard(e: bits[1] id=1, p: bits[8] id=2) -> bits[1] {
  ret result: bits[1] = not(e, id=3)
}"#,
    )
    .unwrap();
    let enabled = function(
        r#"package test
top fn guard(e: bits[1] id=1, p: bits[8] id=2) -> bits[1] {
  ret result: bits[1] = identity(e, id=3)
}"#,
    )
    .unwrap();
    let variation = [vary(1, &[], 0..8)];
    let observation = [bits(&[], 0..8)];
    let proof = prove(&f, &disabled, &variation, &observation).unwrap();
    assert_eq!(proof.result, EquivResult::Proved);
    assert_eq!(proof.guard_reachability, PredicateReachability::Reachable);
    let proof = prove(&f, &enabled, &variation, &observation).unwrap();
    let EquivResult::Disproved {
        lhs_inputs,
        rhs_inputs,
        lhs_output,
        rhs_output,
    } = proof.result
    else {
        panic!("expected counterexample");
    };
    let lhs: Vec<_> = lhs_inputs.iter().map(|p| p.value.clone()).collect();
    let rhs: Vec<_> = rhs_inputs.iter().map(|p| p.value.clone()).collect();
    assert_eq!(lhs[0], rhs[0]);
    assert!(evaluate(&enabled, &lhs).bits_equals_u64_value(1));
    assert_eq!(evaluate(&f, &lhs), lhs_output.value);
    assert_eq!(evaluate(&f, &rhs), rhs_output.value);
    assert_ne!(lhs_output.value, rhs_output.value);
}

#[test]
fn guard_applies_only_to_original_input() {
    let f = function(IDENTITY).unwrap();
    let guard = function(
        r#"package test
top fn guard(x: bits[8] id=1) -> bits[1] {
  zero: bits[8] = literal(value=0, id=2)
  ret result: bits[1] = eq(x, zero, id=3)
}"#,
    )
    .unwrap();
    let proof = prove(&f, &guard, &[vary(0, &[], 0..8)], &[bits(&[], 0..8)]).unwrap();
    let EquivResult::Disproved {
        lhs_inputs,
        rhs_inputs,
        ..
    } = proof.result
    else {
        panic!("requiring the guard on both copies would incorrectly prove this");
    };
    assert!(lhs_inputs[0].value.bits_equals_u64_value(0));
    assert!(!rhs_inputs[0].value.bits_equals_u64_value(0));
}

#[test]
fn partial_ranges_share_unselected_bits_and_overlaps_form_union() {
    let f = function(IDENTITY).unwrap();
    let guard = function(TRUE).unwrap();
    let variation = [vary(0, &[], 0..3), vary(0, &[], 2..4), vary(0, &[], 0..3)];
    assert_eq!(
        prove(&f, &guard, &variation, &[bits(&[], 4..8)])
            .unwrap()
            .result,
        EquivResult::Proved
    );
    let proof = prove(&f, &guard, &variation, &[bits(&[], 3..4)]).unwrap();
    let EquivResult::Disproved {
        lhs_inputs,
        rhs_inputs,
        ..
    } = proof.result
    else {
        panic!("expected counterexample");
    };
    let left = lhs_inputs[0].value.to_bits().unwrap();
    let right = rhs_inputs[0].value.to_bits().unwrap();
    for bit in 4..8 {
        assert_eq!(left.get_bit(bit).unwrap(), right.get_bit(bit).unwrap());
    }
    assert_ne!(left.get_bit(3).unwrap(), right.get_bit(3).unwrap());
}

#[test]
fn nested_tuple_array_paths_and_wide_bits() {
    let f = function(
        r#"package test
top fn main(x: (bits[65], (bits[4], bits[8])[2]) id=1) -> (bits[65], (bits[4], bits[8])[2]) {
  ret result: (bits[65], (bits[4], bits[8])[2]) = identity(x, id=2)
}"#,
    )
    .unwrap();
    let guard = function(
        r#"package test
top fn guard(x: (bits[65], (bits[4], bits[8])[2]) id=1) -> bits[1] {
  ret result: bits[1] = literal(value=1, id=2)
}"#,
    )
    .unwrap();
    for (path, range) in [
        (vec![0], 64..65),
        (vec![1, 0, 0], 1..3),
        (vec![1, 1, 1], 2..7),
    ] {
        let proof = prove(
            &f,
            &guard,
            &[vary(0, &path, range.clone())],
            &[bits(&path, range)],
        )
        .unwrap();
        let EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } = proof.result
        else {
            panic!("expected counterexample");
        };
        assert_eq!(
            evaluate(&f, &[lhs_inputs[0].value.clone()]),
            lhs_output.value
        );
        assert_eq!(
            evaluate(&f, &[rhs_inputs[0].value.clone()]),
            rhs_output.value
        );
        assert_ne!(lhs_output.value, rhs_output.value);
        // Check the selected logical leaf independently of SMT flattening.
        let mut left = lhs_inputs[0].value.clone();
        let mut right = rhs_inputs[0].value.clone();
        for index in &path {
            left = left.get_element(*index).unwrap();
            right = right.get_element(*index).unwrap();
        }
        assert_ne!(left, right);
    }
    let variation = [vary(0, &[1, 0, 0], 0..4)];
    for path in [vec![0], vec![1, 0, 1], vec![1, 1, 0], vec![1, 1, 1]] {
        assert_eq!(
            prove(&f, &guard, &variation, &[bits(&path, 0..4)])
                .unwrap()
                .result,
            EquivResult::Proved
        );
    }
}

#[test]
fn vacuity_and_empty_selections() {
    let f = function(IDENTITY).unwrap();
    let never = function(&TRUE.replace("value=1", "value=0")).unwrap();
    let proof = prove(&f, &never, &[vary(0, &[], 0..8)], &[bits(&[], 0..8)]).unwrap();
    assert_eq!(proof.result, EquivResult::Proved);
    assert_eq!(proof.guard_reachability, PredicateReachability::Unreachable);
    let guard = function(TRUE).unwrap();
    for (variation, observation) in [
        (vec![], vec![bits(&[], 0..8)]),
        (vec![vary(0, &[], 0..8)], vec![]),
        (vec![vary(0, &[], 8..8)], vec![bits(&[], 0..8)]),
        (vec![vary(0, &[], 0..8)], vec![bits(&[], 8..8)]),
    ] {
        let proof = prove(&f, &guard, &variation, &observation).unwrap();
        assert_eq!(proof.result, EquivResult::Proved);
        assert_eq!(proof.guard_reachability, PredicateReachability::Reachable);
    }
}

#[test]
fn zero_width_and_empty_aggregates() {
    for ty in ["bits[0]", "()", "bits[8][0]", "(token, bits[0])"] {
        let f = function(&format!("package test\ntop fn main(x: {ty} id=1) -> {ty} {{\n ret result: {ty} = identity(x, id=2)\n}}")).unwrap();
        let guard = function(&format!("package test\ntop fn guard(x: {ty} id=1) -> bits[1] {{\n ret result: bits[1] = literal(value=1, id=2)\n}}")).unwrap();
        assert_eq!(
            prove(&f, &guard, &[], &[]).unwrap().result,
            EquivResult::Proved
        );
        if ty == "bits[0]" {
            assert_eq!(
                prove(&f, &guard, &[vary(0, &[], 0..0)], &[bits(&[], 0..0)])
                    .unwrap()
                    .result,
                EquivResult::Proved
            );
        }
    }
    let f = function("package test\ntop fn main() -> () { ret result: () = tuple(id=1) }").unwrap();
    let guard = function(
        "package test\ntop fn guard() -> bits[1] { ret result: bits[1] = literal(value=1, id=1) }",
    )
    .unwrap();
    assert_eq!(
        prove(&f, &guard, &[], &[]).unwrap().result,
        EquivResult::Proved
    );
}

#[test]
fn malformed_requests_are_errors() {
    let f = function(IDENTITY).unwrap();
    let guard = function(TRUE).unwrap();
    for selection in [
        vary(1, &[], 0..1),
        vary(0, &[0], 0..1),
        vary(0, &[], 0..9),
        vary(0, &[], Range { start: 5, end: 4 }),
    ] {
        assert!(prove(&f, &guard, &[selection], &[]).is_err());
    }
    assert!(prove(&f, &guard, &[], &[bits(&[], 0..9)]).is_err());
    assert!(prove(&f, &f, &[], &[]).is_err());
    let mismatch = function(&TRUE.replace("bits[8]", "bits[7]")).unwrap();
    assert!(prove(&f, &mismatch, &[], &[]).is_err());
    let missing = function(
        "package test\ntop fn guard() -> bits[1] { ret result: bits[1] = literal(value=1, id=1) }",
    )
    .unwrap();
    assert!(prove(&f, &missing, &[], &[]).is_err());
    let aggregate = function("package test\ntop fn main(x: (bits[8], bits[8][2]) id=1) -> (bits[8], bits[8][2]) { ret r: (bits[8], bits[8][2]) = identity(x, id=2) }").unwrap();
    let aggregate_guard = function("package test\ntop fn guard(x: (bits[8], bits[8][2]) id=1) -> bits[1] { ret r: bits[1] = literal(value=1, id=2) }").unwrap();
    for path in [vec![], vec![2], vec![1], vec![1, 2]] {
        assert!(prove(&aggregate, &aggregate_guard, &[vary(0, &path, 0..1)], &[]).is_err());
    }
}

#[test]
fn invokes_reuse_package_translation() {
    let pkg = Parser::new(
        r#"package test
fn helper(x: bits[8] id=1) -> bits[8] { ret r: bits[8] = not(x, id=2) }
top fn main(x: bits[8] id=3) -> bits[8] { ret r: bits[8] = invoke(x, to_apply=helper, id=4) }
fn guard(x: bits[8] id=5) -> bits[1] { ret r: bits[1] = literal(value=1, id=6) }
"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let f = ProverFn::new(pkg.get_fn("main").unwrap(), Some(&pkg));
    let guard = ProverFn::new(pkg.get_fn("guard").unwrap(), Some(&pkg));
    let proof = prove_conditional_independence(
        &f,
        &guard,
        &[vary(0, &[], 0..8)],
        &[bits(&[], 0..8)],
        SolverLimits::default(),
    )
    .unwrap();
    assert!(matches!(proof.result, EquivResult::Disproved { .. }));
    let no_package = ProverFn::new(f.fn_ref, None);
    assert!(
        prove_conditional_independence(&no_package, &guard, &[], &[], SolverLimits::default())
            .is_err()
    );
    assert!(
        prove_conditional_independence(
            &f.clone().with_fixed_implicit_activation(true),
            &guard,
            &[],
            &[],
            SolverLimits::default()
        )
        .is_err()
    );
}

#[cfg(all(unix, feature = "has-easy-smt"))]
mod solver_status {
    use super::*;
    use std::cell::Cell;
    use xlsynth_prover::prover::Prover;
    use xlsynth_prover::solver::easy_smt::{EasySmtConfig, SolverFn};

    thread_local! {
        static CHECKS: Cell<usize> = const { Cell::new(0) };
    }

    /// The protocol sink acknowledges declarations; checks use the existing
    /// injectable check hook, avoiding timing-sensitive resource-limit tests.
    fn config() -> EasySmtConfig {
        EasySmtConfig {
            solver_path: "/bin/sh".into(),
            solver_args: vec!["-c".into(),
                "while IFS= read -r line; do case \"$line\" in *exit*) exit 0;; *) printf 'success\\n';; esac; done".into()],
            replay_file: None,
            solver_fn: SolverFn { check_fn: |_| Ok(easy_smt::Response::Unknown), ..SolverFn::default() },
        }
    }

    #[test]
    fn unknown_is_neither_a_proof_nor_unreachable() {
        let f = function(IDENTITY).unwrap();
        let guard = function(TRUE).unwrap();
        let f = ProverFn::new(&f, None);
        let guard = ProverFn::new(&guard, None);
        let mut config = config();
        let proof = config
            .prove_conditional_independence(&f, &guard, &[vary(0, &[], 0..8)], &[bits(&[], 0..8)])
            .unwrap();
        assert!(matches!(proof.result, EquivResult::Inconclusive(_)));
        assert_eq!(proof.guard_reachability, PredicateReachability::Unknown);

        CHECKS.set(0);
        config.solver_fn.check_fn = |_| {
            let check = CHECKS.get();
            CHECKS.set(check + 1);
            Ok(if check == 0 {
                easy_smt::Response::Unknown
            } else {
                easy_smt::Response::Unsat
            })
        };
        let proof = config
            .prove_conditional_independence(&f, &guard, &[], &[])
            .unwrap();
        assert_eq!(proof.result, EquivResult::Proved);
        assert_eq!(proof.guard_reachability, PredicateReachability::Unknown);

        config.solver_fn.check_fn = |_| Err(std::io::Error::other("injected solver failure"));
        assert!(
            config
                .prove_conditional_independence(&f, &guard, &[], &[])
                .is_err()
        );
    }
}

#[test]
fn assertions_do_not_constrain_inputs_and_tokens_are_zero_width() {
    let f = function(
        r#"package test
top fn main(tok: token id=1, x: bits[8] id=2) -> (token, bits[8]) {
  never: bits[1] = literal(value=0, id=3)
  failed: token = assert(tok, never, message="ignored", label="function", id=4)
  ret result: (token, bits[8]) = tuple(failed, x, id=5)
}"#,
    )
    .unwrap();
    let guard = function(
        r#"package test
top fn guard(tok: token id=1, x: bits[8] id=2) -> bits[1] {
  never: bits[1] = literal(value=0, id=3)
  failed: token = assert(tok, never, message="ignored", label="guard", id=4)
  always: bits[1] = literal(value=1, id=5)
  pair: (token, bits[1]) = tuple(failed, always, id=6)
  ret result: bits[1] = tuple_index(pair, index=1, id=7)
}"#,
    )
    .unwrap();
    let proof = prove(&f, &guard, &[vary(1, &[], 0..8)], &[bits(&[1], 0..8)]).unwrap();
    assert_eq!(proof.guard_reachability, PredicateReachability::Reachable);
    let EquivResult::Disproved {
        lhs_inputs,
        rhs_inputs,
        lhs_output,
        rhs_output,
    } = proof.result
    else {
        panic!("failing assertions must not turn this into a vacuous proof");
    };
    assert_eq!(lhs_inputs[0].value, IrValue::make_token());
    assert_eq!(rhs_inputs[0].value, IrValue::make_token());
    assert_eq!(
        lhs_output.value.get_element(1).unwrap(),
        lhs_inputs[1].value
    );
    assert_eq!(
        rhs_output.value.get_element(1).unwrap(),
        rhs_inputs[1].value
    );
    assert_ne!(lhs_output.value, rhs_output.value);
    assert!(lhs_output.assertion_violation.is_none());
    assert!(rhs_output.assertion_violation.is_none());
    assert!(prove(&f, &guard, &[vary(0, &[], 0..0)], &[]).is_err());
    assert!(prove(&f, &guard, &[], &[bits(&[0], 0..0)]).is_err());
}
