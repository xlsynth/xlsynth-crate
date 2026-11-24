// SPDX-License-Identifier: Apache-2.0

//! Directed structural-feedback cases exercise exact wiring projections.

use xlsynth_pir::block_legalize::legalize_block_hierarchy;
use xlsynth_pir::block2fn::{Block2FnOptions, block_package_to_fn};
use xlsynth_pir::ir::{NaryOp, NodePayload, Package, Type};
use xlsynth_pir::ir_builder::verify_block_wiring_cycles;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{BlockBuilder, IrValue};

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

/// Builds a package whose leaf forwards an arbitrary hardware type.
fn passthrough(ty: Type) -> Package {
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf.input_port("d", ty).unwrap();
    leaf.output_port("q", d).unwrap();
    leaf.build_package("feedback").unwrap()
}

/// Finalizes a structurally acyclic root and legalizes a separate package.
fn finish(mut package: Package, root: BlockBuilder) -> Package {
    root.build_structural_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();
    legalize_block_hierarchy(&package, "root").unwrap()
}

/// Evaluates the legalized block through the independent block2fn conversion.
fn evaluate(package: &Package, args: &[IrValue]) -> IrValue {
    let function = block_package_to_fn(package, &Block2FnOptions::default())
        .unwrap()
        .function;
    match eval_fn(&function, args) {
        FnEvalResult::Success(result) => result.value,
        FnEvalResult::Failure(error) => panic!("evaluation failed: {error:?}"),
    }
}

#[test]
fn reverse_projections_preserve_bit_order() {
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf.input_port("d", Type::Bits(4)).unwrap();
    let reversed = leaf.rev(d).unwrap();
    leaf.output_port("q", reversed).unwrap();
    let package = leaf.build_package("reverse").unwrap();
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(1)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let low = root.bit_slice(q, 0, 3).unwrap();
    let shifted = root.concat(&[low, seed]).unwrap();
    let reversed = root.rev(shifted).unwrap();
    root.instantiation_input(instance, "d", reversed).unwrap();
    root.output_port("result", q).unwrap();
    let legalized = finish(package, root);
    for seed in 0..2 {
        assert_eq!(evaluate(&legalized, &[bits(1, seed)]), bits(4, seed * 15));
    }
}

#[test]
fn signed_and_unsigned_extension_project_the_correct_high_bits() {
    for signed in [false, true] {
        let mut leaf = BlockBuilder::new("leaf");
        let d = leaf.input_port("d", Type::Bits(2)).unwrap();
        let extended = if signed {
            leaf.sign_extend(d, 4).unwrap()
        } else {
            leaf.zero_extend(d, 4).unwrap()
        };
        leaf.output_port("q", extended).unwrap();
        let package = leaf.build_package("extension").unwrap();
        let mut root = BlockBuilder::new("root");
        let seed = root.input_port("seed", Type::Bits(1)).unwrap();
        let instance = root
            .instantiate("u", package.get_block("leaf").unwrap())
            .unwrap();
        let q = root.instantiation_output(instance, "q").unwrap();
        let low = root.bit_slice(q, 0, 1).unwrap();
        let driver = root.concat(&[low, seed]).unwrap();
        root.instantiation_input(instance, "d", driver).unwrap();
        root.output_port("result", q).unwrap();
        let legalized = finish(package, root);
        for seed in 0..2 {
            let expected = if signed { seed * 15 } else { seed * 3 };
            assert_eq!(evaluate(&legalized, &[bits(1, seed)]), bits(4, expected));
        }
    }
}

#[test]
fn array_updates_replace_the_selected_feedback_elements() {
    let package = passthrough(Type::new_array(Type::Bits(2), 2));
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(2)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let zero = root.literal(bits(1, 0)).unwrap();
    let one = root.literal(bits(1, 1)).unwrap();
    let first = root.array_index(q, zero).unwrap();
    let blank = root
        .literal(IrValue::make_array(&[bits(2, 0), bits(2, 0)]).unwrap())
        .unwrap();
    let upper = root.array_update(blank, first, &[one]).unwrap();
    let driver = root.array_update(upper, seed, &[zero]).unwrap();
    root.instantiation_input(instance, "d", driver).unwrap();
    root.output_port("result", q).unwrap();
    let legalized = finish(package, root);
    for seed in 0..4 {
        let value = bits(2, seed);
        assert_eq!(
            evaluate(&legalized, &[value.clone()]),
            IrValue::make_array(&[value.clone(), value]).unwrap()
        );
    }
}

#[test]
fn array_slice_routes_selected_elements_across_a_feedback_boundary() {
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf
        .input_port("d", Type::new_array(Type::Bits(2), 3))
        .unwrap();
    let one = leaf.literal(bits(2, 1)).unwrap();
    let selected = leaf.array_slice(d, one, 2).unwrap();
    leaf.output_port("q", selected).unwrap();
    let package = leaf.build_package("array_slice").unwrap();
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(2)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let zero = root.literal(bits(1, 0)).unwrap();
    let first = root.array_index(q, zero).unwrap();
    let zero_element = root.literal(bits(2, 0)).unwrap();
    let driver = root
        .array(Type::Bits(2), &[zero_element, seed, first])
        .unwrap();
    root.instantiation_input(instance, "d", driver).unwrap();
    root.output_port("result", q).unwrap();
    let legalized = finish(package, root);
    for seed in 0..4 {
        let value = bits(2, seed);
        assert_eq!(
            evaluate(&legalized, &[value.clone()]),
            IrValue::make_array(&[value.clone(), value]).unwrap()
        );
    }
}

#[test]
fn out_of_bounds_array_update_ignores_its_feedback_value() {
    let package = passthrough(Type::new_array(Type::Bits(2), 2));
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(2)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let zero = root.literal(bits(1, 0)).unwrap();
    let first = root.array_index(q, zero).unwrap();
    let base = root.array(Type::Bits(2), &[seed, seed]).unwrap();
    let outside = root.literal(bits(2, 3)).unwrap();
    let driver = root.array_update(base, first, &[outside]).unwrap();
    root.instantiation_input(instance, "d", driver).unwrap();
    root.output_port("result", q).unwrap();
    let legalized = finish(package, root);
    for seed in 0..4 {
        let value = bits(2, seed);
        assert_eq!(
            evaluate(&legalized, &[value.clone()]),
            IrValue::make_array(&[value.clone(), value]).unwrap()
        );
    }
}

#[test]
fn array_slice_repeats_the_final_element_when_the_slice_extends_past_it() {
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf
        .input_port("d", Type::new_array(Type::Bits(2), 3))
        .unwrap();
    let start = leaf.literal(bits(2, 2)).unwrap();
    let selected = leaf.array_slice(d, start, 2).unwrap();
    leaf.output_port("q", selected).unwrap();
    let package = leaf.build_package("array_slice_bounds").unwrap();
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(2)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let zero = root.literal(bits(1, 0)).unwrap();
    let first = root.array_index(q, zero).unwrap();
    let driver = root.array(Type::Bits(2), &[first, seed, seed]).unwrap();
    root.instantiation_input(instance, "d", driver).unwrap();
    root.output_port("result", q).unwrap();
    let legalized = finish(package, root);
    for seed in 0..4 {
        let value = bits(2, seed);
        assert_eq!(
            evaluate(&legalized, &[value.clone()]),
            IrValue::make_array(&[value.clone(), value]).unwrap()
        );
    }
}

#[test]
fn zero_width_feedback_is_replaced_with_a_zero_width_value() {
    let package = passthrough(Type::Bits(0));
    let mut root = BlockBuilder::new("root");
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    root.instantiation_input(instance, "d", q).unwrap();
    root.output_port("result", q).unwrap();
    let legalized = finish(package, root);
    assert_eq!(evaluate(&legalized, &[]), bits(0, 0));
}

#[test]
fn compact_large_array_feedback_reports_its_expansion_budget() {
    let source = r#"package large_array
block leaf(d: bits[1][65537], q: bits[1][65537]) {
  d: bits[1][65537] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(seed: bits[1], result: bits[1][65537]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[1][65537] = instantiation_output(instantiation=u, port_name=q, id=4)
  zero: bits[17] = literal(value=0, id=5)
  driver: bits[1][65537] = array_update(q, seed, indices=[zero], id=6)
  drive: () = instantiation_input(driver, instantiation=u, port_name=d, id=7)
  result: () = output_port(q, name=result, id=8)
}"#;
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    let cycle_error =
        verify_block_wiring_cycles(package.get_top_block().unwrap(), &package).unwrap_err();
    assert!(
        cycle_error
            .to_string()
            .contains("would expand 65537 elements (limit 65536)")
    );
    let legalize_error = legalize_block_hierarchy(&package, "root").unwrap_err();
    assert!(legalize_error.contains("would expand 65537 elements (limit 65536)"));
}

#[test]
fn feedback_can_cross_two_sibling_instances() {
    let package = passthrough(Type::Bits(2));
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(1)).unwrap();
    let a = root
        .instantiate("a", package.get_block("leaf").unwrap())
        .unwrap();
    let b = root
        .instantiate("b", package.get_block("leaf").unwrap())
        .unwrap();
    let a_q = root.instantiation_output(a, "q").unwrap();
    let b_q = root.instantiation_output(b, "q").unwrap();
    let a_low = root.bit_slice(a_q, 0, 1).unwrap();
    let b_low = root.bit_slice(b_q, 0, 1).unwrap();
    let a_driver = root.concat(&[b_low, seed]).unwrap();
    let b_driver = root.concat(&[a_low, seed]).unwrap();
    root.instantiation_input(a, "d", a_driver).unwrap();
    root.instantiation_input(b, "d", b_driver).unwrap();
    let result = root.concat(&[a_q, b_q]).unwrap();
    root.output_port("result", result).unwrap();
    let legalized = finish(package, root);
    for seed in 0..2 {
        assert_eq!(evaluate(&legalized, &[bits(1, seed)]), bits(4, seed * 15));
    }
}

#[test]
fn feedback_can_cross_nested_instance_boundaries() {
    let mut package = passthrough(Type::Bits(3));
    let mut wrapper = BlockBuilder::new("wrapper");
    let d = wrapper.input_port("d", Type::Bits(3)).unwrap();
    let instance = wrapper
        .instantiate("inner", package.get_block("leaf").unwrap())
        .unwrap();
    wrapper.instantiation_input(instance, "d", d).unwrap();
    let q = wrapper.instantiation_output(instance, "q").unwrap();
    wrapper.output_port("q", q).unwrap();
    wrapper.build_into_package(&mut package).unwrap();
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(1)).unwrap();
    let instance = root
        .instantiate("outer", package.get_block("wrapper").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let low = root.bit_slice(q, 0, 2).unwrap();
    let driver = root.concat(&[low, seed]).unwrap();
    root.instantiation_input(instance, "d", driver).unwrap();
    root.output_port("result", q).unwrap();
    let legalized = finish(package, root);
    assert_eq!(evaluate(&legalized, &[bits(1, 1)]), bits(3, 7));
}

#[test]
fn multi_bit_gates_remain_whole_nodes() {
    let package = passthrough(Type::Bits(4));
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(2)).unwrap();
    let mask = root.input_port("mask", Type::Bits(2)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(instance, "q").unwrap();
    let low = root.bit_slice(q, 0, 2).unwrap();
    let gate = root.and(low, mask).unwrap();
    let driver = root.concat(&[gate, seed]).unwrap();
    root.instantiation_input(instance, "d", driver).unwrap();
    root.output_port("result", q).unwrap();
    let legalized = finish(package, root);
    let gates: Vec<_> = legalized
        .get_top_block()
        .unwrap()
        .nodes
        .iter()
        .filter(|node| matches!(node.payload, NodePayload::Nary(NaryOp::And, _)))
        .collect();
    assert_eq!(gates.len(), 1);
    assert_eq!(gates[0].ty, Type::Bits(2));
    for seed in 0..4 {
        for mask in 0..4 {
            assert_eq!(
                evaluate(&legalized, &[bits(2, seed), bits(2, mask)]),
                bits(4, ((seed & mask) << 2) | seed)
            );
        }
    }
}
