// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "lean-tests")]
use std::{fmt::Write as _, path::PathBuf, sync::OnceLock};

#[cfg(feature = "lean-tests")]
use xlsynth_g8r::test_utils::interesting_ir_roundtrip_cases;
use xlsynth_lean::{LeanError, LeanModuleName, export_ir_function};
#[cfg(feature = "lean-tests")]
use xlsynth_lean::{LeanEvaluator, LeanFunctionInputs};
use xlsynth_pir::ir_parser::Parser;
#[cfg(feature = "lean-tests")]
use xlsynth_pir::{
    IrValue,
    ir_eval::{FnEvalResult, eval_fn_in_package},
    random_inputs::{
        BitValuePattern, generate_argument_sets_from_seed, generate_pattern_arguments,
    },
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
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    assert_eq!(
        f.write_to(root).unwrap(),
        root.join("Generated/Function.lean")
    );
    assert!(root.join("XlsynthLean/IR.lean").is_file());
    assert!(root.join("lean-toolchain").is_file());
    assert!(!root.join("XlsynthLean/Aig.lean").exists());
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
#[cfg(feature = "lean-tests")]
fn evaluator() -> &'static LeanEvaluator {
    static EVALUATOR: OnceLock<LeanEvaluator> = OnceLock::new();
    EVALUATOR.get_or_init(|| {
        let exe = std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN");
        LeanEvaluator::new(PathBuf::from(exe)).unwrap()
    })
}

/// Runs the real Lean module against the original, unmodified PIR function.
#[cfg(feature = "lean-tests")]
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
    assert_eq!(results.len(), inputs.len());
    for (args, actual) in inputs.iter().zip(results) {
        let FnEvalResult::Success(expected) = eval_fn_in_package(&pkg, top, args) else {
            panic!("PIR evaluation failed for {args:?}")
        };
        assert_eq!(actual, expected.value, "{ir}\n{args:?}");
    }
}

#[cfg(feature = "lean-tests")]
#[test]
fn evaluates_mixed_signatures_and_repeated_module_names_together() {
    let module = LeanModuleName::new("Generated.Function").unwrap();
    let packages = [
        "package p\ntop fn entry(x: bits[8] id=1) -> bits[8] { ret n: bits[8] = neg(x, id=2) }",
        "package p\ntop fn entry(x: bits[8] id=1) -> bits[8] { ret n: bits[8] = not(x, id=2) }",
        "package p\ntop fn entry() -> () { ret t: () = tuple(id=1) }",
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

#[cfg(feature = "lean-tests")]
#[test]
fn evaluates_word_operations_and_aggregate_interfaces() {
    check_differential(
        "package nullary\ntop fn empty() -> bits[0] { ret concat.1: bits[0] = concat(id=1) }",
    );
    for ir in [
        MAP,
        include_str!("fixtures/word_ops.ir"),
        include_str!("fixtures/word_ext.ir"),
    ] {
        check_differential(ir);
    }
    for case in interesting_ir_roundtrip_cases() {
        check_differential(case.ir_text);
    }
}

#[cfg(feature = "lean-tests")]
#[test]
fn imported_bitvec_module_uses_xls_signed_division() {
    let ir = r#"package division
top fn divide(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret result: bits[8] = sdiv(x, y, id=3)
}"#;
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let module = LeanModuleName::new("BitVec").unwrap();
    let function = export_ir_function(&package, None, &module).unwrap();
    let inputs = [[1, 0], [0xff, 0], [0xf4, 3]]
        .map(|args| args.map(|v| IrValue::make_ubits(8, v).unwrap()).to_vec());
    let actual = evaluator()
        .eval_ir_function_batch(&function, &inputs)
        .unwrap();
    assert_eq!(actual.len(), inputs.len());
    let top = package.get_top_fn().unwrap();
    for (args, value) in inputs.iter().zip(actual) {
        let FnEvalResult::Success(expected) = eval_fn_in_package(&package, top, args) else {
            panic!("PIR evaluation failed for {args:?}");
        };
        assert_eq!(value, expected.value, "signed division for {args:?}");
    }
}

#[cfg(feature = "lean-tests")]
#[test]
fn evaluates_a_thousand_shared_additions() {
    let mut ir = String::from("package depth\ntop fn main(n1: bits[8] id=1) -> bits[8] {\n");
    for i in 2..=1025 {
        let prefix = if i == 1025 { "ret " } else { "" };
        writeln!(ir, "  {prefix}n{i}: bits[8] = add(n{}, n1, id={i})", i - 1).unwrap();
    }
    ir.push_str("}\n");
    check_differential(&ir);
}

#[cfg(feature = "lean-tests")]
#[test]
fn evaluates_wide_nary_operations_and_arrays() {
    let operands: Vec<_> = (0..2048)
        .map(|i| if i % 5 < 2 { "x" } else { "y" })
        .collect();
    // Concat mixes widths 1 and 2, in bits or in array elements.
    let concat_len = operands.len() + operands.iter().filter(|&&name| name == "y").count();
    let operands = operands.join(", ");
    for (op, x_ty, y_ty, result_ty) in [
        (
            "concat",
            "bits[1]",
            "bits[2]",
            format!("bits[{concat_len}]"),
        ),
        (
            "array_concat",
            "bits[8][1]",
            "bits[8][2]",
            format!("bits[8][{concat_len}]"),
        ),
        ("nor", "bits[8]", "bits[8]", "bits[8]".into()),
        ("array", "bits[8]", "bits[8]", "bits[8][2048]".into()),
    ] {
        check_differential(&format!(
            "package wide\ntop fn entry(x: {x_ty} id=1, y: {y_ty} id=2) -> {result_ty} {{\n  ret result: {result_ty} = {op}({operands}, id=3)\n}}"
        ));
    }
    let values = (0..1025)
        .map(|i| (i % 256).to_string())
        .collect::<Vec<_>>()
        .join(", ");
    check_differential(&format!(
        "package wide_literal\ntop fn entry() -> bits[8][1025] {{\n  ret result: bits[8][1025] = literal(value=[{values}], id=1)\n}}"
    ));
}

#[cfg(feature = "lean-tests")]
#[test]
fn evaluates_deep_tuple_input_and_output() {
    let input_ty = ["bits[8]"; 786].join(", ");
    let output_ty = ["bits[8]"; 201].join(", ");
    let fields = (0..201)
        .map(|i| if i % 3 == 0 { "first" } else { "last" })
        .collect::<Vec<_>>()
        .join(", ");
    check_differential(&format!(
        r#"package deep_tuple
top fn entry(t: ({input_ty}) id=1) -> ({output_ty}) {{
  first: bits[8] = tuple_index(t, index=0, id=2)
  last: bits[8] = tuple_index(t, index=785, id=3)
  ret result: ({output_ty}) = tuple({fields}, id=4)
}}"#
    ));
}

#[cfg(feature = "lean-tests")]
#[test]
fn evaluates_array_of_computed_tuple_field() {
    check_differential(
        r#"package computed_element
top fn entry(a: (bits[2], bits[2], bits[2], bits[2])[2] id=1) -> bits[2][2] {
  x: bits[256] = literal(value=0xabcd, id=2)
  i: bits[8] = encode(x, id=3)
  t: (bits[2], bits[2], bits[2], bits[2]) = array_index(a, indices=[i], id=4)
  v: bits[2] = tuple_index(t, index=3, id=5)
  ret result: bits[2][2] = array(v, v, id=6)
}"#,
    );
}

#[test]
fn rejects_zero_length_arrays_in_exported_functions() {
    let module = LeanModuleName::new("Generated.Function").unwrap();
    for (text, function, op, id) in [
        (EMPTY_MAP, "entry", "array", 3),
        (
            r#"package nested_param
top fn entry(x: (bits[4], bits[8][0])[2] id=1) -> () {
  ret result: () = tuple(id=2)
}"#,
            "entry",
            "param",
            1,
        ),
        (
            r#"package empty_result
top fn entry() -> bits[8][0] {
  ret result: bits[8][0] = array(id=1)
}"#,
            "entry",
            "array",
            1,
        ),
    ] {
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        assert!(matches!(
            export_ir_function(&package, None, &module),
            Err(LeanError::UnsupportedIr { function: f, op: o, id: i, reason: "zero-length array" })
                if (f.as_str(), o.as_str(), i) == (function, op, id)
        ));
    }
}

#[test]
fn checks_unused_nodes_only_in_reachable_functions_for_zero_length_arrays() {
    let text = r#"package p
fn child() -> bits[8] {
  dead: bits[8][0] = array(id=1)
  ret result: bits[8] = literal(value=0, id=2)
}
top fn entry() -> bits[8] {
  ret result: bits[8] = invoke(to_apply=child, id=3)
}
fn clean(x: bits[0][2] id=4) -> () {
  ret result: () = tuple(id=5)
}"#;
    let package = Parser::new(text).parse_and_validate_package().unwrap();
    let module = LeanModuleName::new("Generated.Function").unwrap();
    assert!(matches!(
        export_ir_function(&package, None, &module),
        Err(LeanError::UnsupportedIr { function, id: 1, reason: "zero-length array", .. })
            if function == "child"
    ));
    export_ir_function(&package, Some("clean"), &module).unwrap();
}
