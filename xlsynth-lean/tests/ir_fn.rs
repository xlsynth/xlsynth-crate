// SPDX-License-Identifier: Apache-2.0

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::OnceLock;

use xlsynth_g8r::test_utils::interesting_ir_roundtrip_cases;
use xlsynth_lean::{
    LeanError, LeanEvaluator, LeanFunctionInputs, LeanModuleName, export_ir_function,
};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::random_inputs::{
    BitValuePattern, generate_argument_sets_from_seed, generate_pattern_arguments,
};

const EMPTY_MAP: &str = r#"package empty_map
fn widen(x: bits[4] id=1) -> bits[8] {
  ret result: bits[8] = sign_ext(x, new_bit_count=8, id=2)
}
top fn entry() -> bits[8][0] {
  empty: bits[4][0] = array(id=3)
  ret result: bits[8][0] = map(empty, to_apply=widen, id=4)
}"#;

const MAP: &str = include_str!("fixtures/word_map.ir");

#[test]
fn emits_typed_lean_for_array_map() {
    let pkg = Parser::new(MAP).parse_and_validate_package().unwrap();
    let module = LeanModuleName::new("Generated.Function").unwrap();
    let f = export_ir_function(&pkg, None, &module).unwrap();
    assert_eq!(f.lean_source(), include_str!("goldens/word_map.lean"));
}

#[test]
fn rejects_both_partial_products_even_in_a_callee() {
    for op in ["smulp", "umulp"] {
        let text = format!(
            r#"package p
fn child(x: bits[8] id=1) -> (bits[8], bits[8]) {{
  ret p: (bits[8], bits[8]) = {op}(x, x, id=2)
}}
top fn entry(x: bits[8] id=3) -> (bits[8], bits[8]) {{
  ret result: (bits[8], bits[8]) = invoke(x, to_apply=child, id=4)
}}"#
        );
        let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
        let module = LeanModuleName::new("Generated.Function").unwrap();
        let error = export_ir_function(&pkg, None, &module).unwrap_err();
        assert!(
            matches!(error, LeanError::UnsupportedIr { function, op: actual, id: 2, .. } if function == "child" && actual == op)
        );
    }
}

#[test]
fn rejects_unused_events_and_allows_selecting_an_unrelated_pure_function() {
    let text = r#"package p
fn clean(a: bits[1] id=1) -> bits[1] { ret x: bits[1] = not(a, id=2) }
top fn checked(a: bits[1] id=3) -> bits[1] {
  c: () = cover(a, label="hit", id=4)
  ret x: bits[1] = identity(a, id=5)
}"#;
    let pkg = Parser::new(text).parse_and_validate_package().unwrap();
    let module = LeanModuleName::new("Generated.Function").unwrap();
    assert!(
        matches!(export_ir_function(&pkg, None, &module), Err(LeanError::UnsupportedIr { op, id: 4, .. }) if op == "cover")
    );
    export_ir_function(&pkg, Some("clean"), &module).unwrap();
}

/// Shares the compiled Lean support modules across integration tests.
fn evaluator() -> &'static LeanEvaluator {
    static EVALUATOR: OnceLock<LeanEvaluator> = OnceLock::new();
    EVALUATOR.get_or_init(|| {
        let exe = std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN");
        LeanEvaluator::new(PathBuf::from(exe)).unwrap()
    })
}

/// Runs the real Lean module against the original, unmodified PIR function.
fn check_differential(ir: &str) {
    let pkg = Parser::new(ir).parse_and_validate_package().unwrap();
    let top = pkg.get_top_fn().unwrap();
    let f = export_ir_function(
        &pkg,
        None,
        &LeanModuleName::new("Generated.Function").unwrap(),
    )
    .unwrap();
    let mut inputs = generate_argument_sets_from_seed(top, 0x51cafe, 64);
    inputs.push(generate_pattern_arguments(top, BitValuePattern::Zero));
    inputs.push(generate_pattern_arguments(top, BitValuePattern::AllOnes));
    let results = evaluator()
        .eval_ir_function_batch(&f, &inputs)
        .unwrap_or_else(|e| panic!("{e}\n{}", f.lean_source()));
    for (args, actual) in inputs.iter().zip(results) {
        let FnEvalResult::Success(expected) = eval_fn_in_package(&pkg, top, args) else {
            panic!("PIR evaluation failed for {args:?}")
        };
        assert_eq!(actual, expected.value, "{ir}\n{args:?}");
    }
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn evaluates_mixed_signatures_and_repeated_module_names_together() {
    let module = LeanModuleName::new("Generated.Function").unwrap();
    let packages = [
        "package p\ntop fn entry(x: bits[8] id=1) -> bits[8] { ret n: bits[8] = neg(x, id=2) }",
        "package p\ntop fn entry(x: bits[8] id=1) -> bits[8] { ret n: bits[8] = not(x, id=2) }",
        EMPTY_MAP,
        MAP,
        MAP,
    ]
    .map(|ir| Parser::new(ir).parse_and_validate_package().unwrap());
    let functions = packages
        .each_ref()
        .map(|p| export_ir_function(p, None, &module).unwrap());
    let inputs = packages
        .iter()
        .zip([2, 2, 1, 3, 0])
        .map(|(p, count)| generate_argument_sets_from_seed(p.get_top_fn().unwrap(), 47, count))
        .collect::<Vec<_>>();
    let batches = functions
        .iter()
        .zip(&inputs)
        .map(|(function, input_cases)| LeanFunctionInputs {
            function,
            input_cases,
        })
        .collect::<Vec<_>>();
    let expected = packages
        .iter()
        .zip(&inputs)
        .map(|(p, cases)| {
            cases
                .iter()
                .map(|args| {
                    let FnEvalResult::Success(result) =
                        eval_fn_in_package(p, p.get_top_fn().unwrap(), args)
                    else {
                        panic!("PIR failed on {args:?}");
                    };
                    result.value
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(evaluator().eval_ir_functions(&batches).unwrap(), expected);
    assert!(evaluator().eval_ir_functions(&[]).unwrap().is_empty());
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn evaluates_word_operations_and_aggregate_interfaces() {
    check_differential(
        "package nullary\ntop fn empty() -> bits[0] { ret concat.1: bits[0] = concat(id=1) }",
    );
    for ir in [
        MAP,
        EMPTY_MAP,
        include_str!("fixtures/word_ops.ir"),
        include_str!("fixtures/word_ext.ir"),
    ] {
        check_differential(ir);
    }
    for case in interesting_ir_roundtrip_cases() {
        check_differential(case.ir_text);
    }
}

#[test]
#[ignore = "requires XLSYNTH_LEAN pointing to Lean 4.34.1"]
fn evaluates_a_thousand_shared_additions() {
    let mut ir = String::from("package depth\ntop fn main(n1: bits[8] id=1) -> bits[8] {\n");
    for i in 2..=1025 {
        let prefix = if i == 1025 { "ret " } else { "" };
        writeln!(ir, "  {prefix}n{i}: bits[8] = add(n{}, n1, id={i})", i - 1).unwrap();
    }
    ir.push_str("}\n");
    check_differential(&ir);
}
