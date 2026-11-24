// SPDX-License-Identifier: Apache-2.0

//! State-transition lowering accepts acyclic structural feedback.

use xlsynth_pir::IrValue;
use xlsynth_pir::block_transition::lower_block_transition;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;

const FEEDBACK: &str = r#"package transition_feedback
block leaf(d: bits[4], q: bits[4]) {
  d: bits[4] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(seed: bits[1], result: bits[4]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=4)
  low: bits[3] = bit_slice(q, start=0, width=3, id=5)
  driver: bits[4] = concat(low, seed, id=6)
  drive: () = instantiation_input(driver, instantiation=u, port_name=d, id=7)
  result: () = output_port(q, name=result, id=8)
}
"#;

#[test]
fn transition_lowering_resolves_structural_feedback() {
    let package = Parser::new(FEEDBACK).parse_and_verify_package().unwrap();
    let original = package.to_string();
    let transition = lower_block_transition(&package, "root").unwrap();
    assert_eq!(package.to_string(), original);
    assert_eq!(transition.inputs.len(), 1);
    assert_eq!(transition.outputs.len(), 1);
    assert!(transition.registers.is_empty());
    for seed in 0..=1 {
        let input = IrValue::make_ubits(1, seed).unwrap();
        let output = IrValue::make_ubits(4, seed * 15).unwrap();
        match eval_fn(&transition.function, &[input]) {
            FnEvalResult::Success(result) => {
                assert_eq!(result.value, IrValue::make_tuple(&[output]))
            }
            FnEvalResult::Failure(error) => panic!("transition evaluation failed: {error:?}"),
        }
    }
}

#[test]
fn transition_lowering_rejects_actual_feedback() {
    let source = FEEDBACK.replace("instantiation_input(driver,", "instantiation_input(q,");
    let package = Parser::new(&source).parse_and_verify_package().unwrap();
    let error = lower_block_transition(&package, "root").unwrap_err();
    assert!(error.contains("cycle"), "unexpected error: {error}");
}

#[test]
fn transition_lowering_rejects_opaque_externs() {
    let source = r#"package extern_transition
fn foreign(arg: bits[4] id=1) -> bits[4] {
  ret arg: bits[4] = param(name=arg, id=1)
}
top block root(value: bits[4], result: bits[4]) {
  instantiation opaque(foreign_function=foreign, kind=extern)
  value: bits[4] = input_port(name=value, id=2)
  drive: () = instantiation_input(value, instantiation=opaque, port_name=arg, id=3)
  r: bits[4] = instantiation_output(instantiation=opaque, port_name=return, id=4)
  result: () = output_port(r, name=result, id=5)
}
"#;
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    let error = lower_block_transition(&package, "root").unwrap_err();
    assert!(error.contains("external"), "unexpected error: {error}");
}
