// SPDX-License-Identifier: Apache-2.0

//! Projected block functions retain their reachable function namespace.

use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::{Block2FnOptions, block_ir_to_fn_package};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_verify::verify_package;

#[test]
fn retains_transitive_callees_and_renames_a_function_named_like_the_top_block() {
    let source = r#"package calls
fn top__callee(x: bits[4] id=1) -> bits[4] {
  ret inverted: bits[4] = not(x, id=2)
}
fn top(x: bits[4] id=3) -> bits[4] {
  ret result: bits[4] = invoke(x, to_apply=top__callee, id=4)
}
top block top(value: bits[4], result: bits[4]) {
  value: bits[4] = input_port(name=value, id=5)
  called: bits[4] = invoke(value, to_apply=top, id=6)
  result: () = output_port(called, name=result, id=7)
}
"#;
    let package = block_ir_to_fn_package(source, &Block2FnOptions::default()).unwrap();
    verify_package(&package).unwrap();
    assert!(package.get_fn("top__callee").is_some());
    assert!(package.get_fn("top__callee_1").is_some());
    let function = package.get_top_fn().unwrap();
    assert_eq!(function.name, "top");
    for value in 0..16 {
        let input = IrValue::make_ubits(4, value).unwrap();
        let expected = IrValue::make_ubits(4, value ^ 15).unwrap();
        match eval_fn_in_package(&package, function, &[input]) {
            FnEvalResult::Success(output) => assert_eq!(output.value, expected),
            FnEvalResult::Failure(error) => panic!("projected function failed: {error:?}"),
        }
    }
}
