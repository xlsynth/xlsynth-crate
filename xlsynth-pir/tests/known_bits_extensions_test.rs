// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_pir::ir::{self, Block, NodeGraph, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_eval::{EvalObserver, FnEvalResult, SelectEvent, eval_fn_with_observer};
use xlsynth_pir::ir_random::{RandomFnOptions, RngEntropy, StopPolicy, generate_fn};
use xlsynth_pir::known_bits::{KnownBitsAnalysis, KnownValue, analyze_block, analyze_fn};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;
use xlsynth_pir::{
    BlockBuilder, FnBuilder, IrBits, IrValue, NaryAddOptions, NaryAddTerm, NormalizeLeftOptions,
    RegisterWriteOptions, ResetBehavior,
};

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

fn named(graph: &NodeGraph, name: &str) -> NodeRef {
    NodeRef {
        index: graph
            .nodes
            .iter()
            .position(|node| node.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing node {name}")),
    }
}

/// Checks every computed value against facts, independently of the transfer
/// code.
struct ClaimObserver<'a, 'ir> {
    analysis: &'a KnownBitsAnalysis<'ir>,
    seen: Vec<bool>,
    require_exact: bool,
}

impl EvalObserver for ClaimObserver<'_, '_> {
    fn on_select(&mut self, _event: SelectEvent) {
        // Node-value callbacks already contain the concrete selected values.
    }

    fn on_node_value(&mut self, node_ref: NodeRef, node_text_id: usize, value: &IrValue) {
        let facts = self.analysis.get(node_ref).unwrap();
        assert!(
            facts.contains(value),
            "{} node id={node_text_id}: {facts:?} excludes {value}",
            self.analysis.graph().name,
        );
        if self.require_exact {
            assert_eq!(
                facts,
                &KnownValue::constant(value),
                "{} node id={node_text_id} should be fully known",
                self.analysis.graph().name,
            );
        }
        self.seen[node_ref.index] = true;
    }
}

/// Checks all claims against concrete execution and the same graph as a block.
fn check_claims(
    function: &ir::Fn,
    arguments: &[Vec<IrValue>],
    require_exact: bool,
) -> Vec<IrValue> {
    let analysis =
        analyze_fn(function).unwrap_or_else(|error| panic!("analysis failed: {error}\n{function}"));
    let block = Block::from_function(function.clone(), None).unwrap();
    let block_analysis = analyze_block(&block).unwrap();
    for (reference, facts) in analysis.iter() {
        assert_eq!(block_analysis.get(reference), Some(facts));
    }
    let mut outputs = Vec::with_capacity(arguments.len());
    for args in arguments {
        assert_eq!(args.len(), function.params.len());
        let mut observer = ClaimObserver {
            analysis: &analysis,
            seen: vec![false; function.nodes.len()],
            require_exact,
        };
        for (&param, value) in function.params.iter().zip(args) {
            observer.on_node_value(param, function.get_node(param).text_id, value);
        }
        let value = match eval_fn_with_observer(function, args, Some(&mut observer)) {
            FnEvalResult::Success(success) => success.value,
            FnEvalResult::Failure(failure) => {
                panic!("unexpected interpreter failure: {failure:?}\n{function}");
            }
        };
        assert!(
            analysis
                .get(function.ret_node_ref.unwrap())
                .unwrap()
                .contains(&value)
        );
        for (reference, _) in analysis.iter() {
            assert!(
                observer.seen[reference.index],
                "unobserved node {reference:?}"
            );
        }
        outputs.push(value);
    }
    outputs
}

#[test]
fn constant_extensions_are_exact_at_zero_and_wide_widths() {
    for width in [0, 1, 4, 65, 129] {
        let mut builder = FnBuilder::new("constant_extensions");
        let x = builder
            .literal(IrValue::from_bits(&IrBits::from_lsb_is_0(
                &(0..width).map(|i| i % 3 == 0).collect::<Vec<_>>(),
            )))
            .unwrap();
        let y = builder
            .literal(IrValue::from_bits(&IrBits::all_ones(width)))
            .unwrap();
        let carry_in = builder.literal(bits(1, 1)).unwrap();
        let mut large_count = vec![false; 129];
        large_count[128] = true;
        let large_count = builder
            .literal(IrValue::from_bits(&IrBits::from_lsb_is_0(&large_count)))
            .unwrap();
        let carry = builder.ext_carry_out(x, y, carry_in).unwrap();
        let lsb = builder.ext_prio_encode(x, true).unwrap();
        let msb = builder.ext_prio_encode(x, false).unwrap();
        let clz = builder.ext_clz(x, 3, 8).unwrap();
        let normalized = builder
            .ext_normalize_left(x, NormalizeLeftOptions::new(width + 3))
            .unwrap();
        let normalized_pair = builder
            .ext_normalize_left(
                x,
                NormalizeLeftOptions {
                    normalized_bit_count: width + 3,
                    shift_offset: 2,
                    clz_bit_count: Some(8),
                },
            )
            .unwrap();
        let mask = builder.ext_mask_low(large_count, width + 3).unwrap();
        let sum = builder
            .ext_nary_add(
                &[NaryAddTerm::unsigned(x), NaryAddTerm::signed(y).negate()],
                NaryAddOptions::new(width + 3),
            )
            .unwrap();
        let empty_sum = builder
            .ext_nary_add(&[], NaryAddOptions::new(width))
            .unwrap();
        let result = builder
            .tuple(&[
                carry,
                lsb,
                msb,
                clz,
                normalized,
                normalized_pair,
                mask,
                sum,
                empty_sum,
            ])
            .unwrap();
        let function = builder.build(result).unwrap();
        check_claims(&function, &[vec![]], true);
    }
}

#[test]
fn partial_extension_facts_hold_for_every_small_input() {
    let mut builder = FnBuilder::new("partial_extensions");
    let a = builder.param("a", Type::Bits(3)).unwrap();
    let b = builder.param("b", Type::Bits(3)).unwrap();
    let carry_in = builder.param("carry_in", Type::Bits(1)).unwrap();
    let a4 = builder.zero_extend(a, 4).unwrap();
    let b4 = builder.zero_extend(b, 4).unwrap();
    let carry_zero = builder.ext_carry_out(a4, b4, carry_in).unwrap();
    builder.set_name(carry_zero, "carry_zero").unwrap();
    let high = builder.literal(bits(4, 8)).unwrap();
    let a_high = builder.or(a4, high).unwrap();
    let b_high = builder.or(b4, high).unwrap();
    let carry_one = builder.ext_carry_out(a_high, b_high, carry_in).unwrap();
    builder.set_name(carry_one, "carry_one").unwrap();
    let sum = builder
        .ext_nary_add(
            &[NaryAddTerm::unsigned(a), NaryAddTerm::unsigned(b)],
            NaryAddOptions::new(8),
        )
        .unwrap();
    builder.set_name(sum, "sum").unwrap();

    let three = builder.literal(bits(4, 3)).unwrap();
    let four = builder.literal(bits(4, 4)).unwrap();
    let count = builder.and(a4, three).unwrap();
    let count = builder.or(count, four).unwrap();
    let mask = builder.ext_mask_low(count, 8).unwrap();
    builder.set_name(mask, "mask").unwrap();

    let a8 = builder.zero_extend(a, 8).unwrap();
    let eight = builder.literal(bits(8, 8)).unwrap();
    let shift = builder.literal(bits(3, 4)).unwrap();
    let upper = builder.shll(a8, shift).unwrap();
    let lsb_input = builder.or(upper, eight).unwrap();
    let lsb = builder.ext_prio_encode(lsb_input, true).unwrap();
    builder.set_name(lsb, "lsb").unwrap();
    let msb_input = builder.or(a8, eight).unwrap();
    let msb = builder.ext_prio_encode(msb_input, false).unwrap();
    builder.set_name(msb, "msb").unwrap();
    let clz = builder.ext_clz(msb_input, 2, 8).unwrap();
    builder.set_name(clz, "clz").unwrap();
    let normalized = builder
        .ext_normalize_left(
            a,
            NormalizeLeftOptions {
                normalized_bit_count: 8,
                shift_offset: 2,
                clz_bit_count: Some(8),
            },
        )
        .unwrap();
    builder.set_name(normalized, "normalized").unwrap();
    let result = builder
        .tuple(&[carry_zero, carry_one, sum, mask, lsb, msb, clz, normalized])
        .unwrap();
    let function = builder.build(result).unwrap();
    let analysis = analyze_fn(&function).unwrap();
    for (name, expected) in [
        ("carry_zero", "0"),
        ("carry_one", "1"),
        ("sum", "0000XXXX"),
        ("mask", "0XXX1111"),
        ("lsb", "0011"),
        ("msb", "0011"),
        ("clz", "00000110"),
    ] {
        assert_eq!(
            analysis
                .bits(named(&function, name))
                .unwrap()
                .to_ternary_string(),
            expected,
            "{name}"
        );
    }
    let normalized = named(&function, "normalized");
    let normalized_bits = analysis.leaf(normalized, &[0]).unwrap();
    for bit in 0..2 {
        assert!(normalized_bits.mask().get_bit(bit).unwrap());
        assert!(!normalized_bits.value().get_bit(bit).unwrap());
    }
    assert_eq!(
        analysis.leaf(normalized, &[1]).unwrap().to_ternary_string(),
        "000000XX"
    );
    let arguments = (0..8)
        .flat_map(|a| {
            (0..8).flat_map(move |b| (0..2).map(move |c| vec![bits(3, a), bits(3, b), bits(1, c)]))
        })
        .collect::<Vec<_>>();
    check_claims(&function, &arguments, false);
}

#[test]
fn extension_mask_precision_and_population_soundness_are_separate_properties() {
    let mut builder = FnBuilder::new("extension_population");
    let input = builder.param("input", Type::Bits(4)).unwrap();
    let clz = builder.ext_clz(input, 0, 3).unwrap();
    let priority = builder.ext_prio_encode(input, true).unwrap();
    let result = builder.tuple(&[clz, priority]).unwrap();
    let function = builder.build(result).unwrap();
    let arguments = (0..16)
        .map(|value| vec![bits(4, value)])
        .collect::<Vec<_>>();
    let outputs = check_claims(&function, &arguments, false);
    let expected = outputs
        .iter()
        .map(KnownValue::constant)
        .reduce(|a, b| a.join(&b).unwrap())
        .unwrap();
    let analysis = analyze_fn(&function).unwrap();
    for path in [[0], [1]] {
        let actual = analysis
            .leaf(function.ret_node_ref.unwrap(), &path)
            .unwrap();
        let expected = expected.leaf(&path).unwrap();
        assert_eq!(actual.mask(), expected.mask());
        assert_eq!(actual.value(), expected.value());
        // These transfers promise exact ternary masks for this domain, not
        // necessarily the strongest possible population interval as well.
        assert!(actual.min_ones() <= expected.min_ones());
        assert!(actual.max_ones() >= expected.max_ones());
    }
}

#[test]
fn priority_direction_zero_sentinel_and_clz_offset_wrapping_match_interpreter() {
    let mut builder = FnBuilder::new("priority_and_clz");
    let x = builder.param("x", Type::Bits(4)).unwrap();
    let lsb = builder.ext_prio_encode(x, true).unwrap();
    let msb = builder.ext_prio_encode(x, false).unwrap();
    let wrapped = builder.ext_clz(x, usize::MAX, 2).unwrap();
    let wide = builder.ext_clz(x, usize::MAX, 129).unwrap();
    let empty = builder.ext_clz(x, 1, 0).unwrap();
    let result = builder.tuple(&[lsb, msb, wrapped, wide, empty]).unwrap();
    let function = builder.build(result).unwrap();
    let arguments = (0..16).map(|x| vec![bits(4, x)]).collect::<Vec<_>>();
    let outputs = check_claims(&function, &arguments, false);
    assert_eq!(outputs[0].get_element(0).unwrap(), bits(3, 4));
    assert_eq!(outputs[0].get_element(1).unwrap(), bits(3, 4));
    assert_eq!(outputs[0].get_element(2).unwrap(), bits(2, 3));
    assert_eq!(outputs[10].get_element(0).unwrap(), bits(3, 1));
    assert_eq!(outputs[10].get_element(1).unwrap(), bits(3, 3));
    assert_eq!(outputs[0].get_element(4).unwrap(), bits(0, 0));
}

#[test]
fn nary_add_signed_negated_terms_resize_before_modular_addition() {
    for width in [0, 1, 2, 4, 8, 65, 129] {
        let mut builder = FnBuilder::new("nary_resize");
        let a = builder.param("a", Type::Bits(4)).unwrap();
        let b = builder.param("b", Type::Bits(3)).unwrap();
        let signed_negative = builder
            .ext_nary_add(
                &[NaryAddTerm::signed(a).negate(), NaryAddTerm::unsigned(b)],
                NaryAddOptions::new(width),
            )
            .unwrap();
        let unsigned_negative = builder
            .ext_nary_add(
                &[
                    NaryAddTerm::unsigned(a).negate(),
                    NaryAddTerm::signed(b).negate(),
                ],
                NaryAddOptions {
                    bit_count: width,
                    architecture: Some(ir::ExtNaryAddArchitecture::BrentKung),
                },
            )
            .unwrap();
        let single = builder
            .ext_nary_add(&[NaryAddTerm::signed(a)], NaryAddOptions::new(width))
            .unwrap();
        let empty = builder
            .ext_nary_add(&[], NaryAddOptions::new(width))
            .unwrap();
        let result = builder
            .tuple(&[signed_negative, unsigned_negative, single, empty])
            .unwrap();
        let function = builder.build(result).unwrap();
        let arguments = (0..16)
            .flat_map(|a| (0..8).map(move |b| vec![bits(4, a), bits(3, b)]))
            .collect::<Vec<_>>();
        let outputs = check_claims(&function, &arguments, false);
        for output in &outputs {
            assert_eq!(output.get_element(3).unwrap(), bits(width, 0));
        }
        if width == 8 {
            // a=15 represents -1 when signed, but +15 when unsigned.
            assert_eq!(outputs[15 * 8].get_element(0).unwrap(), bits(8, 1));
            assert_eq!(outputs[15 * 8].get_element(1).unwrap(), bits(8, 241));
        }
    }
}

#[test]
fn normalize_left_handles_saturating_offsets_and_locally_valid_truncation() {
    for width in [0, 2, 4, 8, 65] {
        for offset in [0, 1, usize::MAX - 2, usize::MAX] {
            let mut builder = FnBuilder::new("normalize_extremes");
            let x = builder.param("x", Type::Bits(4)).unwrap();
            let result = builder
                .ext_normalize_left(
                    x,
                    NormalizeLeftOptions {
                        normalized_bit_count: width.max(4),
                        shift_offset: 0,
                        clz_bit_count: Some(3),
                    },
                )
                .unwrap();
            let mut function = builder.build(result).unwrap();
            let result = function.ret_node_ref.unwrap();
            // Builders restrict these fields for portable desugaring, but
            // locally analyzed graphs use the concrete interpreter semantics.
            let NodePayload::ExtNormalizeLeft {
                normalized_bit_count,
                shift_offset,
                ..
            } = &mut function.get_node_mut(result).payload
            else {
                unreachable!();
            };
            *normalized_bit_count = width;
            *shift_offset = offset;
            let result_type = ir::ext_normalize_left_result_type(width, Some(3));
            function.get_node_mut(result).ty = result_type.clone();
            function.ret_ty = result_type;
            let arguments = (0..16).map(|x| vec![bits(4, x)]).collect::<Vec<_>>();
            let outputs = check_claims(&function, &arguments, false);
            if offset >= width {
                let analysis = analyze_fn(&function).unwrap();
                assert!(analysis.leaf(result, &[0]).unwrap().is_fully_known());
                assert!(analysis.leaf(result, &[0]).unwrap().value().is_zero());
                for output in outputs {
                    assert_eq!(output.get_element(0).unwrap(), bits(width, 0));
                }
            }
        }
    }
}

#[test]
fn wide_mask_counts_saturate_without_host_integer_truncation() {
    let mut builder = FnBuilder::new("wide_mask_counts");
    let count = builder.param("count", Type::Bits(129)).unwrap();
    let mut high_bits = vec![false; 129];
    high_bits[128] = true;
    let high = builder
        .literal(IrValue::from_bits(&IrBits::from_lsb_is_0(&high_bits)))
        .unwrap();
    let definitely_large = builder.or(count, high).unwrap();
    let saturated = builder.ext_mask_low(definitely_large, 129).unwrap();
    builder.set_name(saturated, "saturated").unwrap();
    let mask = builder.ext_mask_low(count, 129).unwrap();
    let empty = builder.ext_mask_low(count, 0).unwrap();
    let result = builder.tuple(&[saturated, mask, empty]).unwrap();
    let function = builder.build(result).unwrap();
    let analysis = analyze_fn(&function).unwrap();
    assert_eq!(
        analysis
            .bits(named(&function, "saturated"))
            .unwrap()
            .to_ternary_string(),
        "1".repeat(129)
    );
    let mut arguments = [0, 1, 64, 128, 129, u64::MAX]
        .into_iter()
        .map(|count| vec![bits(129, count)])
        .collect::<Vec<_>>();
    for high_bit in [64, 128] {
        let mut value = vec![false; 129];
        value[high_bit] = true;
        arguments.push(vec![IrValue::from_bits(&IrBits::from_lsb_is_0(&value))]);
    }
    arguments.push(vec![IrValue::from_bits(&IrBits::all_ones(129))]);
    let outputs = check_claims(&function, &arguments, false);
    for output in outputs.iter().skip(4) {
        assert_eq!(
            output.get_element(1).unwrap(),
            IrValue::from_bits(&IrBits::all_ones(129))
        );
    }
}

#[test]
fn extension_sources_use_arbitrary_register_state_not_reset_values() {
    let mut builder = BlockBuilder::new("state_extensions");
    builder.clock_port("clk").unwrap();
    let reset = builder.input_port("rst", Type::Bits(1)).unwrap();
    builder
        .set_reset(
            reset,
            ResetBehavior {
                asynchronous: true,
                active_low: false,
            },
        )
        .unwrap();
    let state = builder
        .register("state", Type::Bits(8), Some(bits(8, 7)))
        .unwrap();
    let current = builder.register_read(state).unwrap();
    builder.set_name(current, "current").unwrap();
    let clz = builder.ext_clz(current, 0, 8).unwrap();
    builder.set_name(clz, "clz").unwrap();
    builder
        .register_write(
            state,
            current,
            RegisterWriteOptions {
                load_enable: None,
                reset: Some(reset),
            },
        )
        .unwrap();
    builder.output_port("count", clz).unwrap();
    let block = builder.build().unwrap();
    let block_analysis = analyze_block(&block).unwrap();
    assert_eq!(
        block_analysis
            .bits(named(&block, "current"))
            .unwrap()
            .to_ternary_string(),
        "XXXXXXXX"
    );
    let block_clz = block_analysis.get(named(&block, "clz")).unwrap();
    assert_eq!(block_clz.as_bits().unwrap().to_ternary_string(), "0000XXXX");

    let mut builder = FnBuilder::new("current_state_reference");
    let current = builder.param("current", Type::Bits(8)).unwrap();
    let clz = builder.ext_clz(current, 0, 8).unwrap();
    let function = builder.build(clz).unwrap();
    let function_analysis = analyze_fn(&function).unwrap();
    assert_eq!(
        Some(block_clz),
        function_analysis.get(function.ret_node_ref.unwrap())
    );
    let arguments = (0..256)
        .map(|state| vec![bits(8, state)])
        .collect::<Vec<_>>();
    check_claims(&function, &arguments, false);
}

#[test]
fn generic_random_extension_graphs_have_sound_node_facts() {
    let mut seen_extensions = BTreeSet::new();
    for seed in 0..64 {
        let options = RandomFnOptions {
            max_params: 3,
            max_nodes: 28,
            max_bit_width: 16,
            max_div_mod_bit_width: Some(8),
            max_multiply_operand_bit_width: Some(8),
            max_type_depth: 2,
            max_aggregate_leaves: 8,
            max_array_length: 3,
            max_tuple_length: 3,
            allow_extension_ops: true,
            allow_zero_width_bits: true,
            allow_arbitrary_width_multiply: true,
            allow_gate: true,
            ..Default::default()
        };
        let mut entropy = RngEntropy::new(StdRng::seed_from_u64(seed));
        let function = generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(24))
            .unwrap()
            .function;
        for node in &function.nodes {
            let operation = node.payload.get_operator();
            if operation.starts_with("ext_") {
                seen_extensions.insert(operation.to_string());
            }
        }
        let arguments = generate_argument_sets_from_seed(&function, seed ^ 0x8877, 12);
        check_claims(&function, &arguments, false);
    }
    assert_eq!(
        seen_extensions,
        [
            "ext_carry_out",
            "ext_clz",
            "ext_mask_low",
            "ext_nary_add",
            "ext_normalize_left",
            "ext_prio_encode"
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    );
}
