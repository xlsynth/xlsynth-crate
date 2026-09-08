// SPDX-License-Identifier: Apache-2.0

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_pir::ir::{self, Block, NodeGraph, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_eval::{EvalObserver, FnEvalResult, SelectEvent, eval_fn_with_observer};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_random::{RandomFnOptions, RngEntropy, StopPolicy, generate_fn};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;
use xlsynth_pir::range_analysis::{
    IntervalSet, RangeAnalysis, RangeValue, analyze_block, analyze_fn,
};
use xlsynth_pir::{
    BlockBuilder, FnBuilder, IrBits, IrValue, NaryAddOptions, NaryAddTerm, NormalizeLeftOptions,
    RegisterWriteOptions, ResetBehavior,
};

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

fn ranges(width: usize, endpoints: &[(u64, u64)]) -> IntervalSet {
    IntervalSet::from_intervals(
        width,
        endpoints.iter().map(|&(lo, hi)| {
            (
                IrBits::make_ubits(width, lo).unwrap(),
                IrBits::make_ubits(width, hi).unwrap(),
            )
        }),
    )
    .unwrap()
}

#[test]
fn small_domain_transfers_refine_function_and_block_facts() {
    let mut builder = FnBuilder::new("small_domains");
    let start = builder.param("start", Type::Bits(2)).unwrap();
    let choice = builder.param("choice", Type::Bits(1)).unwrap();
    let update = builder.param("update", Type::Bits(1)).unwrap();
    let suffix = builder.literal(bits(7, 0)).unwrap();
    let sparse = builder.concat(&[choice, suffix]).unwrap();
    let base4 = builder.literal(bits(4, 0)).unwrap();
    let one = builder.literal(bits(1, 1)).unwrap();
    let wide_start = builder.zero_extend(start, 32).unwrap();
    let one_hot = builder.bit_slice_update(base4, wide_start, one).unwrap();
    let base2 = builder.literal(bits(2, 0)).unwrap();
    let optional = builder.bit_slice_update(base2, choice, update).unwrap();
    let result = builder.tuple(&[sparse, one_hot, optional]).unwrap();
    let function = builder.build(result).unwrap();
    let analysis = analyze_fn(&function).unwrap();
    let result = function.ret_node_ref.unwrap();
    assert_eq!(
        analysis.leaf(result, &[0]),
        Some(&ranges(8, &[(0, 0), (128, 128)]))
    );
    assert_eq!(
        analysis.leaf(result, &[1]),
        Some(&ranges(4, &[(1, 2), (4, 4), (8, 8)]))
    );
    assert_eq!(analysis.leaf(result, &[2]), Some(&ranges(2, &[(0, 2)])));
    let mut arguments = Vec::new();
    for start in 0..4 {
        for choice in 0..2 {
            for update in 0..2 {
                arguments.push(vec![bits(2, start), bits(1, choice), bits(1, update)]);
            }
        }
    }
    check_claims(&function, &arguments, false);
}

#[test]
fn cheap_precision_improvements_survive_graph_propagation() {
    let package = Parser::new(
        r#"package precision
top fn f(s: bits[2] id=1, x: bits[4] id=2, start: bits[129] id=3) -> (bits[4], bits[4], bits[8]) {
  three: bits[4] = literal(value=3, id=4)
  four: bits[4] = literal(value=4, id=5)
  five: bits[4] = literal(value=5, id=6)
  bounded: bits[4] = sel(s, cases=[three, four, five], default=five, id=7)
  inverted: bits[4] = not(bounded, id=8)
  eight: bits[4] = literal(value=8, id=9)
  remainder: bits[4] = umod(inverted, eight, id=10)
  wide: bits[8] = zero_ext(x, new_bit_count=8, id=11)
  sliced: bits[8] = dynamic_bit_slice(wide, start, width=8, id=12)
  ret result: (bits[4], bits[4], bits[8]) = tuple(inverted, remainder, sliced, id=13)
}
"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let function = package.get_top_fn().unwrap();
    let analysis = analyze_fn(function).unwrap();
    let result = function.ret_node_ref.unwrap();
    assert_eq!(analysis.leaf(result, &[0]), Some(&ranges(4, &[(10, 12)])));
    assert_eq!(analysis.leaf(result, &[1]), Some(&ranges(4, &[(2, 4)])));
    assert_eq!(analysis.leaf(result, &[2]), Some(&ranges(8, &[(0, 15)])));
    check_claims(
        function,
        &generate_argument_sets_from_seed(function, 123, 128),
        false,
    );
}

#[test]
fn array_routing_preserves_holes_and_out_of_bounds_noop_writes() {
    let package = Parser::new(
        r#"package arrays
top fn f(index: bits[129] id=1) -> (bits[4], bits[4][1], bits[4][2]) {
  a: bits[4][4] = literal(value=[1, 3, 7, 15], id=2)
  read: bits[4] = array_index(a, indices=[index], id=3)
  one: bits[4][1] = literal(value=[1], id=4)
  three: bits[4] = literal(value=3, id=5)
  written: bits[4][1] = array_update(one, three, indices=[index], id=6)
  sliced: bits[4][2] = array_slice(a, index, width=2, id=7)
  ret result: (bits[4], bits[4][1], bits[4][2]) = tuple(read, written, sliced, id=8)
}
"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let function = package.get_top_fn().unwrap();
    let analysis = analyze_fn(function).unwrap();
    let result = function.ret_node_ref.unwrap();
    assert_eq!(
        analysis.leaf(result, &[0]),
        Some(&ranges(4, &[(1, 1), (3, 3), (7, 7), (15, 15)]))
    );
    assert_eq!(
        analysis.leaf(result, &[1, 0]),
        Some(&ranges(4, &[(1, 1), (3, 3)]))
    );
    assert_eq!(
        analysis.leaf(result, &[2, 1]),
        Some(&ranges(4, &[(3, 3), (7, 7), (15, 15)]))
    );
    let mut inputs = (0..6).map(|x| vec![bits(129, x)]).collect::<Vec<_>>();
    inputs.push(vec![IrValue::from_bits(&IrBits::all_ones(129))]);
    check_claims(function, &inputs, false);
}

#[test]
fn all_node_claims_hold_for_random_graphs_and_function_block_views_match() {
    let mut informative = 0;
    for seed in 0..128 {
        let options = RandomFnOptions {
            max_params: 4,
            max_nodes: 48,
            max_bit_width: if seed % 2 == 0 { 8 } else { 129 },
            max_div_mod_bit_width: Some(8),
            max_multiply_operand_bit_width: Some(16),
            max_type_depth: 2,
            max_aggregate_leaves: 8,
            max_array_length: 4,
            max_tuple_length: 4,
            allow_zero_width_bits: true,
            allow_arbitrary_width_multiply: true,
            allow_gate: true,
            allow_extension_ops: true,
            allow_events: false,
            ..Default::default()
        };
        let mut entropy = RngEntropy::new(StdRng::seed_from_u64(seed));
        let function = generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(48))
            .unwrap()
            .function;
        let arguments = generate_argument_sets_from_seed(&function, seed ^ 0x654321, 16);
        check_claims(&function, &arguments, false);
        informative += analyze_fn(&function)
            .unwrap()
            .iter()
            .filter(|(node, facts)| {
                !matches!(
                    function.get_node(*node).payload,
                    NodePayload::Literal(_)
                        | NodePayload::Param
                        | NodePayload::InputPort { .. }
                        | NodePayload::RegisterRead { .. }
                ) && facts
                    .as_bits()
                    .is_some_and(|bits| bits.width() != 0 && !bits.is_full())
            })
            .count();
    }
    assert!(
        informative > 128,
        "random sweep must exercise informative claims for nonliteral computed bits"
    );
}

#[test]
fn block_current_state_is_unconstrained_and_sinks_have_unit_facts() {
    let mut builder = BlockBuilder::new("state");
    builder.clock_port("clk").unwrap();
    let reset = builder.input_port("rst", Type::Bits(1)).unwrap();
    builder
        .set_reset(
            reset,
            ResetBehavior {
                asynchronous: true,
                active_low: true,
            },
        )
        .unwrap();
    let state = builder
        .register("state", Type::Bits(8), Some(bits(8, 7)))
        .unwrap();
    let current = builder.register_read(state).unwrap();
    builder.set_name(current, "current").unwrap();
    let mask = builder.literal(bits(8, 15)).unwrap();
    let low = builder.and(current, mask).unwrap();
    builder.set_name(low, "low").unwrap();
    let write = builder
        .register_write(
            state,
            low,
            RegisterWriteOptions {
                load_enable: None,
                reset: Some(reset),
            },
        )
        .unwrap();
    builder.set_name(write, "write").unwrap();
    let nested_unit = builder.tuple(&[write]).unwrap();
    builder.output_port("unit", nested_unit).unwrap();
    let block = builder.build().unwrap();
    let analysis = analyze_block(&block).unwrap();
    assert!(analysis.bits(named(&block, "current")).unwrap().is_full());
    assert_eq!(
        analysis.bits(named(&block, "low")),
        Some(&ranges(8, &[(0, 15)]))
    );
    assert_eq!(
        analysis.get(named(&block, "write")),
        Some(&RangeValue::Tuple(vec![]))
    );
    let output = block.output_ports().next().unwrap();
    assert_eq!(
        analysis.get(block.output_value(output)),
        Some(&RangeValue::Tuple(vec![RangeValue::Tuple(vec![])]))
    );
    assert_eq!(analysis.get(output), Some(&RangeValue::Tuple(vec![])));
}

#[test]
fn invalid_graphs_and_missing_block_sentinel_return_errors() {
    let mut builder = FnBuilder::new("valid");
    let x = builder.param("x", Type::Bits(8)).unwrap();
    let n = builder.not(x).unwrap();
    let valid = builder.build(n).unwrap();
    let n = valid.ret_node_ref.unwrap();
    for invalid_operand in [n, NodeRef { index: 0 }, NodeRef { index: usize::MAX }] {
        let mut invalid = valid.clone();
        invalid.get_node_mut(n).payload = NodePayload::Unop(Unop::Not, invalid_operand);
        assert!(analyze_fn(&invalid).is_err());
    }
    let mut invalid = valid.clone();
    invalid.get_node_mut(n).text_id = invalid.get_param(0).text_id;
    assert!(analyze_fn(&invalid).is_err());
    let mut invalid = valid;
    invalid.ret_ty = Type::Bits(9);
    assert!(analyze_fn(&invalid).is_err());
    let mut block = BlockBuilder::new("empty").build().unwrap();
    assert_eq!(analyze_block(&block).unwrap().iter().count(), 0);
    block.nodes[0].ty = Type::Bits(8);
    block.nodes[0].payload = NodePayload::Literal(bits(8, 3));
    assert!(analyze_block(&block).is_err());
    block.nodes.clear();
    assert!(analyze_block(&block).is_err());
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
    analysis: &'a RangeAnalysis<'ir>,
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
                &RangeValue::constant(value),
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
