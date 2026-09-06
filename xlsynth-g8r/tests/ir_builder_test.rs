// SPDX-License-Identifier: Apache-2.0

//! Exercise function construction through both the PIR and XLS interpreters.

use std::collections::BTreeMap;

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_g8r::test_utils::interesting_ir_roundtrip_cases;
use xlsynth_pir::desugar_extensions::emit_package_as_xls_ir_text;
use xlsynth_pir::ir::{self, NodePayload, Type, Unop};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::libxls_bridge::{value_from_libxls, value_to_libxls};
use xlsynth_pir::random_inputs::generate_uniform_arguments_with_rng;
use xlsynth_pir::{FnBuilder, IrBits, IrValue, NaryAddOptions, NaryAddTerm, NormalizeLeftOptions};

/// Checks the emitted package, roundtrip stability, and independent execution.
fn check_with_xls(package: &ir::Package, samples: &[Vec<IrValue>]) {
    let text = package.to_string();
    let reparsed = Parser::new(&text).parse_and_validate_package().unwrap();
    assert_eq!(reparsed.to_string(), text);
    check_with_xls_text(package, &text, samples);
}

/// Compares PIR execution with independently parsed, upstream-compatible text.
fn check_with_xls_text(package: &ir::Package, text: &str, samples: &[Vec<IrValue>]) {
    let function = package.get_top_fn().unwrap();
    let xls_package = xlsynth::IrPackage::parse_ir(text, None)
        .unwrap_or_else(|error| panic!("XLS rejected builder output: {error}\n{text}"));
    xls_package.verify().unwrap();
    let xls_function = xls_package.get_function(&function.name).unwrap();
    for args in samples {
        let FnEvalResult::Success(pir_result) = eval_fn_in_package(package, function, args) else {
            panic!("unexpected PIR execution failure");
        };
        let xls_args = args
            .iter()
            .map(value_to_libxls)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let xls_result = xls_function.interpret(&xls_args).unwrap();
        assert_eq!(
            pir_result.value,
            value_from_libxls(&xls_result, &function.ret_ty).unwrap(),
            "interpreter disagreement for arguments {args:?}\n{text}"
        );
    }
}

/// Generates deterministic samples, including all-zero and all-one bitvectors.
fn bit_samples(width: usize, count: usize) -> Vec<Vec<IrValue>> {
    let zero = IrValue::make_ubits(width, 0).unwrap();
    let ones = IrValue::from_bits(&IrBits::from_lsb_is_0(&vec![true; width]));
    let high_bit = IrValue::from_bits(&IrBits::from_lsb_is_0(
        &(0..width).map(|bit| bit + 1 == width).collect::<Vec<_>>(),
    ));
    let mut result = Vec::new();
    for lhs in [&zero, &ones, &high_bit] {
        for rhs in [&zero, &ones, &high_bit] {
            result.push(vec![lhs.clone(), rhs.clone()]);
        }
    }
    let mut builder = FnBuilder::new("sample_types");
    let lhs = builder.param("lhs", Type::Bits(width)).unwrap();
    builder.param("rhs", Type::Bits(width)).unwrap();
    let function = builder.build(lhs).unwrap();
    let mut rng = StdRng::seed_from_u64(0x6275_696c_6465_7200 + width as u64);
    for _ in 0..count {
        result.push(generate_uniform_arguments_with_rng(&mut rng, &function));
    }
    result
}

#[test]
fn arithmetic_comparison_and_bit_operations_match_xls() {
    for width in [1, 2, 4, 7, 32, 63, 64, 65, 129] {
        let mut b = FnBuilder::new("operations");
        let x = b.param("x", Type::Bits(width)).unwrap();
        let y = b.param("y", Type::Bits(width)).unwrap();
        let values = [
            b.add(x, y).unwrap(),
            b.sub(x, y).unwrap(),
            b.umul(x, y).unwrap(),
            b.smul(x, y).unwrap(),
            b.udiv(x, y).unwrap(),
            b.sdiv(x, y).unwrap(),
            b.umod(x, y).unwrap(),
            b.smod(x, y).unwrap(),
            b.and(x, y).unwrap(),
            b.or(x, y).unwrap(),
            b.xor(x, y).unwrap(),
            b.nand(x, y).unwrap(),
            b.nor(x, y).unwrap(),
            b.eq(x, y).unwrap(),
            b.ne(x, y).unwrap(),
            b.ult(x, y).unwrap(),
            b.ule(x, y).unwrap(),
            b.ugt(x, y).unwrap(),
            b.uge(x, y).unwrap(),
            b.slt(x, y).unwrap(),
            b.sle(x, y).unwrap(),
            b.sgt(x, y).unwrap(),
            b.sge(x, y).unwrap(),
            b.shll(x, y).unwrap(),
            b.shrl(x, y).unwrap(),
            b.shra(x, y).unwrap(),
            b.neg(x).unwrap(),
            b.not(x).unwrap(),
            b.rev(x).unwrap(),
            b.and_reduce(x).unwrap(),
            b.or_reduce(x).unwrap(),
            b.xor_reduce(x).unwrap(),
            b.clz(x).unwrap(),
            b.ctz(x).unwrap(),
            b.sign_extend(x, width + 3).unwrap(),
            b.zero_extend(x, width + 3).unwrap(),
            b.one_hot(x, true).unwrap(),
            b.one_hot(x, false).unwrap(),
            b.encode(x).unwrap(),
        ];
        let result = b.tuple(&values).unwrap();
        let package = b.build_package(result, "operations").unwrap();
        check_with_xls(&package, &bit_samples(width, 16));
    }
}

#[test]
fn slices_and_updates_match_xls_at_boundaries() {
    for width in [1, 4, 65, 129] {
        let mut b = FnBuilder::new("slices");
        let x = b.param("x", Type::Bits(width)).unwrap();
        let start = b.param("start", Type::Bits(width)).unwrap();
        let values = [
            b.bit_slice(x, 0, width).unwrap(),
            b.bit_slice(x, width, 0).unwrap(),
            b.bit_slice(x, width - 1, 1).unwrap(),
            b.dynamic_bit_slice(x, start, width).unwrap(),
            b.bit_slice_update(x, start, x).unwrap(),
            b.concat(&[x, start]).unwrap(),
        ];
        let result = b.tuple(&values).unwrap();
        let package = b.build_package(result, "slices").unwrap();
        check_with_xls(&package, &bit_samples(width, 8));
    }
}

#[test]
fn aggregate_operations_and_selection_match_xls() {
    let mut b = FnBuilder::new("aggregates");
    let x = b.param("x", Type::Bits(65)).unwrap();
    let y = b.param("y", Type::Bits(65)).unwrap();
    let selector = b.param("selector", Type::Bits(3)).unwrap();
    let a = b.tuple(&[x, y]).unwrap();
    let c = b.tuple(&[y, x]).unwrap();
    let element_type = Type::Tuple(vec![Box::new(Type::Bits(65)), Box::new(Type::Bits(65))]);
    let array = b.array(element_type, &[a, c, a]).unwrap();
    let index = b.array_index(array, selector).unwrap();
    let updated = b.array_update(array, c, &[selector]).unwrap();
    let slice = b.array_slice(array, selector, 4).unwrap();
    let concatenated = b.array_concat(&[array, array]).unwrap();
    let one_hot = b.one_hot_select(selector, &[a, c, a]).unwrap();
    let priority = b.priority_select(selector, &[a, c, a], c).unwrap();
    let selected = b.select(selector, &[a, c, a], Some(c)).unwrap();
    let field = b.tuple_index(index, 1).unwrap();
    let result = b
        .tuple(&[
            index,
            updated,
            slice,
            concatenated,
            one_hot,
            priority,
            selected,
            field,
        ])
        .unwrap();
    let package = b.build_package(result, "aggregates").unwrap();
    let mut samples = Vec::new();
    for args in bit_samples(65, 4) {
        for selector in 0..8 {
            let mut args = args.clone();
            args.push(IrValue::make_ubits(3, selector).unwrap());
            samples.push(args);
        }
    }
    check_with_xls(&package, &samples);
}

#[test]
fn decoding_matches_xls_with_default_and_explicit_widths() {
    for width in 0..=4 {
        let mut b = FnBuilder::new("decoding");
        let x = b.param("x", Type::Bits(width)).unwrap();
        let default = b.decode(x, None).unwrap();
        let narrow = b.decode(x, Some(1)).unwrap();
        let bounded = b.decode(x, Some((1usize << width).min(7))).unwrap();
        let result = b.tuple(&[default, narrow, bounded]).unwrap();
        let package = b.build_package(result, "decoding").unwrap();
        let samples = (0..(1 << width))
            .map(|value| vec![IrValue::make_ubits(width, value).unwrap()])
            .collect::<Vec<_>>();
        check_with_xls(&package, &samples);
    }
}

#[test]
fn reconstruct_shared_signature_corpus_with_builder() {
    let mut rng = StdRng::seed_from_u64(0x6275_696c_6465_72);
    for case in interesting_ir_roundtrip_cases() {
        let original_package = Parser::new(case.ir_text)
            .parse_and_validate_package()
            .unwrap();
        let original = original_package.get_top_fn().unwrap();
        let mut b = FnBuilder::new(&original.name);
        let mut parameters = BTreeMap::new();
        for param in &original.params {
            parameters.insert(
                param.id.get_wrapped_id(),
                b.param(&param.name, param.ty.clone()).unwrap(),
            );
        }
        let mut nodes = BTreeMap::new();
        for (index, node) in original.nodes.iter().enumerate() {
            let value = match &node.payload {
                NodePayload::Nil => {
                    // PIR's sentinel is not an executable node.
                    continue;
                }
                NodePayload::GetParam(id) => parameters[&id.get_wrapped_id()],
                NodePayload::Literal(value) => b.literal(value.clone()).unwrap(),
                NodePayload::Unop(Unop::Identity, operand) => {
                    b.identity(nodes[&operand.index]).unwrap()
                }
                other => panic!("extend builder corpus reconstruction for {other:?}"),
            };
            nodes.insert(index, value);
        }
        let result = nodes[&original.ret_node_ref.unwrap().index];
        let package = b.build_package(result, "corpus").unwrap();
        let samples = (0..16)
            .map(|_| generate_uniform_arguments_with_rng(&mut rng, original))
            .collect::<Vec<_>>();
        check_with_xls(&package, &samples);
        for args in samples {
            assert_eq!(
                eval_fn(original, &args),
                eval_fn(package.get_top_fn().unwrap(), &args),
                "corpus case {}",
                case.name
            );
        }
    }
}

#[test]
fn independently_built_functions_share_an_xls_package_without_id_collisions() {
    let mut first = FnBuilder::new("first");
    let x = first.param("x", Type::Bits(65)).unwrap();
    let mut package = first.build_package(x, "multiple_functions").unwrap();

    let mut second = FnBuilder::new("second");
    let x = second.param("x", Type::Bits(65)).unwrap();
    let y = second.param("y", Type::Bits(65)).unwrap();
    let sum = second.add(x, y).unwrap();
    second.build_into_package(sum, &mut package).unwrap();

    assert_eq!(package.get_top_fn().unwrap().name, "first");
    check_with_xls(&package, &[vec![IrValue::all_ones_bits(65)]]);
    package.top = Some(("second".to_string(), ir::MemberType::Function));
    check_with_xls(&package, &bit_samples(65, 4));
}

#[test]
fn zero_width_construction_matches_xls() {
    let mut b = FnBuilder::new("zero_width");
    let x = b.param("x", Type::Bits(0)).unwrap();
    let results = [
        b.concat(&[]).unwrap(),
        b.bit_slice(x, 0, 0).unwrap(),
        b.zero_extend(x, 3).unwrap(),
        b.one_hot(x, true).unwrap(),
        b.one_hot(x, false).unwrap(),
        b.clz(x).unwrap(),
        b.ctz(x).unwrap(),
    ];
    let result = b.tuple(&results).unwrap();
    let package = b.build_package(result, "zero_width").unwrap();
    check_with_xls(&package, &[vec![IrValue::make_ubits(0, 0).unwrap()]]);
}

#[test]
fn multioperand_gating_and_partial_products_match_xls() {
    for width in [1, 4, 65, 129] {
        let mut b = FnBuilder::new("multioperand");
        let x = b.param("x", Type::Bits(width)).unwrap();
        let y = b.param("y", Type::Bits(width)).unwrap();
        let condition = b.ne(x, y).unwrap();
        let inverted = b.not(x).unwrap();
        let mut values = vec![
            b.and_all(&[x, y, inverted]).unwrap(),
            b.or_all(&[x, y, inverted]).unwrap(),
            b.xor_all(&[x, y, inverted]).unwrap(),
            b.nand_all(&[x, y, inverted]).unwrap(),
            b.nor_all(&[x, y, inverted]).unwrap(),
            b.gate(condition, x).unwrap(),
        ];
        for product_width in [1, width, width * 2] {
            for product in [
                b.umulp(x, y, product_width).unwrap(),
                b.smulp(x, y, product_width).unwrap(),
            ] {
                // Partial products are implementation-dependent; their sum is
                // the observable contract shared by PIR and XLS.
                let low = b.tuple_index(product, 0).unwrap();
                let high = b.tuple_index(product, 1).unwrap();
                values.push(b.add(low, high).unwrap());
            }
        }
        let result = b.tuple(&values).unwrap();
        let package = b.build_package(result, "multioperand").unwrap();
        check_with_xls(&package, &bit_samples(width, 8));
    }
}

#[test]
fn nested_calls_and_counted_loops_match_xls() {
    for width in [4, 65, 129] {
        let mut add = FnBuilder::new("add_pair");
        let x = add.param("x", Type::Bits(width)).unwrap();
        let y = add.param("y", Type::Bits(width)).unwrap();
        let result = add.add(x, y).unwrap();
        let mut package = add.build_package(result, "calls").unwrap();
        let carry_ty = Type::Tuple(vec![Box::new(Type::Bits(width)); 2]);
        let mut body = FnBuilder::new("loop_body");
        let index = body.param("index", Type::Bits(4)).unwrap();
        let carry = body.param("carry", carry_ty).unwrap();
        let invariant = body.param("invariant", Type::Bits(width)).unwrap();
        let first = body.tuple_index(carry, 0).unwrap();
        let second = body.tuple_index(carry, 1).unwrap();
        let sum = body
            .invoke(package.get_fn("add_pair").unwrap(), &[first, invariant])
            .unwrap();
        let index = body.zero_extend(index, width).unwrap();
        let second = body.add(second, index).unwrap();
        let result = body.tuple(&[sum, second]).unwrap();
        body.build_into_package(result, &mut package).unwrap();

        for (trip_count, stride) in [(0, 1), (1, 3), (4, 0), (4, 3)] {
            let name = format!("caller_{trip_count}_{stride}");
            let mut b = FnBuilder::new(&name);
            let x = b.param("x", Type::Bits(width)).unwrap();
            let y = b.param("y", Type::Bits(width)).unwrap();
            let init = b.tuple(&[x, y]).unwrap();
            let result = b
                .counted_for(
                    init,
                    trip_count,
                    stride,
                    package.get_fn("loop_body").unwrap(),
                    &[y],
                )
                .unwrap();
            b.build_into_package(result, &mut package).unwrap();
            package.top = Some((name, ir::MemberType::Function));
            check_with_xls(&package, &bit_samples(width, 4));
        }
    }
}

#[test]
fn all_builder_extensions_match_desugared_xls() {
    for width in [1, 4, 65, 129] {
        let mut b = FnBuilder::new("extensions");
        let x = b.param("x", Type::Bits(width)).unwrap();
        let y = b.param("y", Type::Bits(width)).unwrap();
        let carry = b.ne(x, y).unwrap();
        let mut values = vec![
            b.ext_carry_out(x, y, carry).unwrap(),
            b.ext_prio_encode(x, true).unwrap(),
            b.ext_prio_encode(x, false).unwrap(),
            b.ext_clz(x, 3, 9).unwrap(),
            b.ext_normalize_left(x, NormalizeLeftOptions::new(width))
                .unwrap(),
            b.ext_normalize_left(
                x,
                NormalizeLeftOptions {
                    normalized_bit_count: width + 3,
                    shift_offset: 2,
                    clz_bit_count: Some(9),
                },
            )
            .unwrap(),
            b.ext_mask_low(x, width + 1).unwrap(),
        ];
        for architecture in [
            None,
            Some(ir::ExtNaryAddArchitecture::RippleCarry),
            Some(ir::ExtNaryAddArchitecture::KoggeStone),
            Some(ir::ExtNaryAddArchitecture::BrentKung),
        ] {
            values.push(
                b.ext_nary_add(
                    &[
                        NaryAddTerm::signed(x),
                        NaryAddTerm::unsigned(y).negate(),
                        NaryAddTerm::signed(y),
                    ],
                    NaryAddOptions {
                        bit_count: width + 3,
                        architecture,
                    },
                )
                .unwrap(),
            );
        }
        let result = b.tuple(&values).unwrap();
        let package = b.build_package(result, "extensions").unwrap();
        let text = package.to_string();
        assert_eq!(
            Parser::new(&text)
                .parse_and_validate_package()
                .unwrap()
                .to_string(),
            text
        );
        let xls_text = emit_package_as_xls_ir_text(&package).unwrap();
        check_with_xls_text(&package, &xls_text, &bit_samples(width, 8));
    }
}

#[test]
fn function_effects_and_escaped_strings_are_accepted_by_xls() {
    let mut b = FnBuilder::new("effects");
    let condition = b.param("condition", Type::Bits(1)).unwrap();
    let x = b.param("x", Type::Bits(65)).unwrap();
    let token = b.after_all(&[]).unwrap();
    let token = b
        .assert(
            token,
            condition,
            "expected \"true\"\nwith \\ escaping",
            "assert_label",
        )
        .unwrap();
    let token = b
        .trace(token, condition, "value=0x{:x}\n\\\"", &[x], 2)
        .unwrap();
    let covered = b.cover(condition, "cover_label").unwrap();
    let result = b.tuple(&[token, covered, x]).unwrap();
    let package = b.build_package(result, "effects").unwrap();
    check_with_xls(
        &package,
        &[vec![IrValue::bool(true), IrValue::all_ones_bits(65)]],
    );
    let args = vec![IrValue::bool(false), IrValue::all_ones_bits(65)];
    assert!(!matches!(
        eval_fn(package.get_top_fn().unwrap(), &args),
        FnEvalResult::Success(_)
    ));
    let xls_package = xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
    let xls_args = args
        .iter()
        .map(value_to_libxls)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(
        xls_package
            .get_function("effects")
            .unwrap()
            .interpret(&xls_args)
            .is_err()
    );
}

#[test]
fn function_insertion_reserves_block_output_ids() {
    let original = Parser::new(
        r#"package mixed
block identity(x: bits[8], out: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  out: () = output_port(x, name=out, id=77)
}
"#,
    )
    .parse_and_validate_package()
    .unwrap();
    for output_id in [2, 77] {
        let mut package = original.clone();
        let ir::PackageMember::Block(block) = &mut package.members[0] else {
            panic!("expected a block");
        };
        let output = block.output_ports().next().unwrap();
        block.get_node_mut(output).text_id = output_id;
        let mut b = FnBuilder::new("function");
        let x = b.param("x", Type::Bits(65)).unwrap();
        let result = b.not(x).unwrap();
        b.build_into_package(result, &mut package).unwrap();
        package.top = Some(("function".to_string(), ir::MemberType::Function));
        check_with_xls(&package, &[vec![IrValue::all_ones_bits(65)]]);
    }
}
