// SPDX-License-Identifier: Apache-2.0

//! Optional legalization turns acyclic structural wiring into whole-node DAGs.

use xlsynth_pir::block_legalize::legalize_block_hierarchy;
use xlsynth_pir::block2fn::{Block2FnOptions, block_package_to_fn};
use xlsynth_pir::ir::{Package, Type};
use xlsynth_pir::ir_builder::{verify_block_combinational_cycles, verify_block_wiring_cycles};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_verify::verify_package;
use xlsynth_pir::{BlockBuilder, FnBuilder, IrValue};

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

/// Converts the legalized top into a function and evaluates its sole output.
fn evaluate(package: &Package, args: &[IrValue]) -> IrValue {
    let function = block_package_to_fn(package, &Block2FnOptions::default())
        .unwrap()
        .function;
    match eval_fn(&function, args) {
        FnEvalResult::Success(result) => result.value,
        FnEvalResult::Failure(failure) => panic!("unexpected evaluation failure: {failure:?}"),
    }
}

/// Builds q={q[width-2:0], seed}, an acyclic circuit with a whole-node cycle.
fn shifted_feedback(width: usize) -> Package {
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf.input_port("d", Type::Bits(width)).unwrap();
    leaf.output_port("q", d).unwrap();
    let mut package = leaf.build_package("feedback").unwrap();
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(1)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let lower = root.bit_slice(q, 0, width - 1).unwrap();
    let d = root.concat(&[lower, seed]).unwrap();
    root.instantiation_input(instance, "d", d).unwrap();
    root.output_port("result", q).unwrap();
    root.build_structural_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();
    package
}

#[test]
fn slice_concat_feedback_is_legalized_without_modifying_the_input() {
    let package = shifted_feedback(4);
    let original = package.to_string();
    let root = package.get_top_block().unwrap();
    assert!(verify_block_combinational_cycles(root, &package).is_err());
    verify_block_wiring_cycles(root, &package).unwrap();

    let legalized = legalize_block_hierarchy(&package, "root").unwrap();
    assert_eq!(package.to_string(), original);
    verify_package(&legalized).unwrap();
    assert!(legalized.get_top_block().unwrap().instantiations.is_empty());
    verify_block_combinational_cycles(legalized.get_top_block().unwrap(), &legalized).unwrap();
    for seed in 0..=1 {
        assert_eq!(evaluate(&legalized, &[bits(1, seed)]), bits(4, seed * 15));
    }
}

#[test]
fn aggregate_projection_feedback_produces_the_correct_function() {
    let pair = Type::Tuple(vec![Box::new(Type::Bits(3)), Box::new(Type::Bits(3))]);
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf.input_port("d", pair.clone()).unwrap();
    leaf.output_port("q", d).unwrap();
    let mut package = leaf.build_package("aggregate_feedback").unwrap();
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(3)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let first = root.tuple_index(q, 0).unwrap();
    let d = root.tuple(&[seed, first]).unwrap();
    root.instantiation_input(instance, "d", d).unwrap();
    root.output_port("result", q).unwrap();
    root.build_structural_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();

    let legalized = legalize_block_hierarchy(&package, "root").unwrap();
    for seed in 0..8 {
        let value = bits(3, seed);
        assert_eq!(
            evaluate(&legalized, &[value.clone()]),
            IrValue::make_tuple(&[value.clone(), value])
        );
    }
}

#[test]
fn fixed_array_index_feedback_produces_the_correct_function() {
    let array_type = Type::new_array(Type::Bits(3), 2);
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf.input_port("d", array_type).unwrap();
    leaf.output_port("q", d).unwrap();
    let mut package = leaf.build_package("array_feedback").unwrap();
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(3)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let zero = root.literal(bits(1, 0)).unwrap();
    let first = root.array_index(q, zero).unwrap();
    let d = root.array(Type::Bits(3), &[seed, first]).unwrap();
    root.instantiation_input(instance, "d", d).unwrap();
    root.output_port("result", q).unwrap();
    root.build_structural_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();

    let legalized = legalize_block_hierarchy(&package, "root").unwrap();
    for seed in 0..8 {
        let value = bits(3, seed);
        assert_eq!(
            evaluate(&legalized, &[value.clone()]),
            IrValue::make_array(&[value.clone(), value]).unwrap()
        );
    }
}

#[test]
fn actual_feedback_is_rejected() {
    let source = r#"package loop
block leaf(d: bits[1], q: bits[1]) {
  d: bits[1] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(result: bits[1]) {
  instantiation u(block=leaf, kind=block)
  q: bits[1] = instantiation_output(instantiation=u, port_name=q, id=3)
  drive: () = instantiation_input(q, instantiation=u, port_name=d, id=4)
  result: () = output_port(q, name=result, id=5)
}"#;
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    assert!(legalize_block_hierarchy(&package, "root").is_err());
}

#[test]
fn already_schedulable_blocks_are_unchanged() {
    let mut builder = BlockBuilder::new("root");
    let value = builder.input_port("value", Type::Bits(16)).unwrap();
    let inverted = builder.not(value).unwrap();
    builder.output_port("result", inverted).unwrap();
    let package = builder.build_package("plain").unwrap();
    let legalized = legalize_block_hierarchy(&package, "root").unwrap();
    assert_eq!(legalized.to_string(), package.to_string());
    assert_eq!(evaluate(&legalized, &[bits(16, 0xa5a5)]), bits(16, 0x5a5a));
}

#[test]
fn a_wide_schedulable_graph_is_not_expanded_per_bit() {
    let mut builder = BlockBuilder::new("root");
    let value = builder.input_port("value", Type::Bits(1_000_000)).unwrap();
    builder.output_port("result", value).unwrap();
    let package = builder.build_package("wide").unwrap();
    let legalized = legalize_block_hierarchy(&package, "root").unwrap();
    assert_eq!(legalized.to_string(), package.to_string());
}

#[test]
fn opaque_external_instances_are_rejected_explicitly() {
    let mut foreign = FnBuilder::new("foreign");
    let arg = foreign.param("arg", Type::Bits(8)).unwrap();
    let mut package = foreign.build_package(arg, "externs").unwrap();
    let mut root = BlockBuilder::new("root");
    let value = root.input_port("value", Type::Bits(8)).unwrap();
    let instance = root
        .instantiate_extern("external", package.get_fn("foreign").unwrap())
        .unwrap();
    root.instantiation_input(instance, "arg", value).unwrap();
    let result = root.instantiation_output(instance, "return").unwrap();
    root.output_port("result", result).unwrap();
    root.build_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();

    let error = legalize_block_hierarchy(&package, "root").unwrap_err();
    assert!(error.contains("extern"), "unexpected error: {error}");
}

/// A compact structural recurrence must fail before creating a quadratic proof
/// graph.
#[test]
fn excessive_intermediate_bit_width_is_rejected() {
    let package = shifted_feedback(6000);
    let error = legalize_block_hierarchy(&package, "root").unwrap_err();
    assert!(
        error.contains("proof-graph budget"),
        "unexpected error: {error}"
    );
}
