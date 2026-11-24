// SPDX-License-Identifier: Apache-2.0

//! Behavioral external blocks participate in proof lowering through their
//! model.

use xlsynth_pir::IrValue;
use xlsynth_pir::block_legalize::legalize_block_hierarchy;
use xlsynth_pir::block_simplify::simplify_block_hierarchy;
use xlsynth_pir::block_transition::lower_block_transition;
use xlsynth_pir::block2fn::{Block2FnOptions, block_package_to_fn};
use xlsynth_pir::ir::{NodePayload, Package, Unop};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;

const MODELED_FEEDBACK: &str = r#"package modeled_feedback
block model(d: bits[4], q: bits[4]) {
  d: bits[4] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(seed: bits[1], result: bits[4]) {
  instantiation u(block=model, module_name=external_leaf, parameters={WIDTH:"4"}, input_casts={d:"pkg::word_t"}, kind=behavioral_extern_block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=4)
  low: bits[3] = bit_slice(q, start=0, width=3, id=5)
  driver: bits[4] = concat(low, seed, id=6)
  drive: () = instantiation_input(driver, instantiation=u, port_name=d, id=7)
  result: () = output_port(q, name=result, id=8)
}
"#;

/// Parses package-form PIR, including external block extensions.
fn parse(source: &str) -> Package {
    Parser::new(source).parse_and_verify_package().unwrap()
}

/// Evaluates the expected successful PIR function.
fn evaluate(function: &xlsynth_pir::ir::Fn, args: &[IrValue]) -> IrValue {
    match eval_fn(function, args) {
        FnEvalResult::Success(result) => result.value,
        FnEvalResult::Failure(error) => panic!("function evaluation failed: {error:?}"),
    }
}

#[test]
fn modeled_external_feedback_uses_the_model_without_rewriting_rtl_metadata() {
    let package = parse(MODELED_FEEDBACK);
    let original = package.to_string();
    let explicit = legalize_block_hierarchy(&package, "root").unwrap();
    let automatic = block_package_to_fn(&package, &Block2FnOptions::default())
        .unwrap()
        .function;
    let transition = lower_block_transition(&package, "root").unwrap();

    assert!(explicit.get_top_block().unwrap().instantiations.is_empty());
    for seed in 0..=1 {
        let input = IrValue::make_ubits(1, seed).unwrap();
        let expected = IrValue::make_ubits(4, seed * 15).unwrap();
        assert_eq!(evaluate(&automatic, &[input.clone()]), expected);
        assert_eq!(
            evaluate(&transition.function, &[input]),
            IrValue::make_tuple(&[expected])
        );
    }
    assert_eq!(package.to_string(), original);
    assert!(original.contains("module_name=external_leaf"));
    assert!(original.contains("parameters={WIDTH:\"4\"}"));
    assert!(original.contains("input_casts={d:\"pkg::word_t\"}"));
}

#[test]
fn modeled_external_actual_feedback_is_rejected() {
    let source = MODELED_FEEDBACK.replace("instantiation_input(driver,", "instantiation_input(q,");
    let package = parse(&source);
    assert!(
        legalize_block_hierarchy(&package, "root")
            .unwrap_err()
            .contains("cycle")
    );
    assert!(
        block_package_to_fn(&package, &Block2FnOptions::default())
            .unwrap_err()
            .contains("cycle")
    );
}

#[test]
fn opaque_external_rejection_keeps_the_full_instance_path() {
    let package = parse(
        r#"package opaque_feedback
block child() {
  instantiation black_box(module_name=external_leaf, ports={}, kind=opaque_extern_block)
}
block independent(value: bits[1], result: bits[1]) {
  value: bits[1] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}
top block root() {
  instantiation u(block=child, kind=block)
}
"#,
    );
    legalize_block_hierarchy(&package, "independent").unwrap();
    let mut independent = package.clone();
    independent.set_top_block("independent").unwrap();
    block_package_to_fn(&independent, &Block2FnOptions::default()).unwrap();
    let expected = "opaque external block instance 'root.u.black_box' of module 'external_leaf' has no behavioral model";
    assert_eq!(
        legalize_block_hierarchy(&package, "root").unwrap_err(),
        expected
    );
    assert_eq!(
        lower_block_transition(&package, "root").unwrap_err(),
        expected
    );
    assert_eq!(
        block_package_to_fn(&package, &Block2FnOptions::default()).unwrap_err(),
        format!("block2fn: {expected}")
    );
}

#[test]
fn selected_hierarchy_simplifies_the_behavioral_model_without_altering_external_metadata() {
    let source = MODELED_FEEDBACK.replace(
        "  q: () = output_port(d, name=q, id=2)",
        "  zero: bits[4] = literal(value=0, id=9)\n  pass: bits[4] = or(d, zero, id=10)\n  q: () = output_port(pass, name=q, id=2)",
    );
    let package = parse(&source);
    let original = package.to_string();
    let simplified = simplify_block_hierarchy(&package, "root").unwrap();
    let model = simplified.get_block("model").unwrap();
    assert!(
        model
            .nodes
            .iter()
            .any(|node| matches!(node.payload, NodePayload::Unop(Unop::Identity, _)))
    );
    assert_eq!(
        simplified.get_block("root").unwrap().instantiations[0].kind,
        package.get_block("root").unwrap().instantiations[0].kind
    );
    assert_eq!(package.to_string(), original);
    block_package_to_fn(&simplified, &Block2FnOptions::default()).unwrap();
}
