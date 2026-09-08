// SPDX-License-Identifier: Apache-2.0

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_pir::ir::{self, Block, NodeGraph, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_eval::{EvalObserver, FnEvalResult, SelectEvent, eval_fn_with_observer};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_random::{RandomFnOptions, RngEntropy, StopPolicy, generate_fn};
use xlsynth_pir::known_bits::{
    KnownBits, KnownBitsAnalysis, KnownValue, analyze_block, analyze_fn,
};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;
use xlsynth_pir::{BlockBuilder, FnBuilder, IrBits, IrValue, RegisterWriteOptions, ResetBehavior};

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

#[test]
fn population_bounds_are_checked_and_concrete_membership_uses_them() {
    for width in [0, 1, 64, 65, 129, 257] {
        let unknown = KnownBits::unknown(width);
        assert_eq!((unknown.min_ones(), unknown.max_ones()), (0, width));
        assert_eq!((unknown.min_zeros(), unknown.max_zeros()), (0, width));
        let value = IrBits::from_lsb_is_0(&(0..width).map(|bit| bit % 3 == 0).collect::<Vec<_>>());
        let ones = (0..width).filter(|bit| bit % 3 == 0).count();
        let constant = KnownBits::constant(&value);
        assert_eq!((constant.min_ones(), constant.max_ones()), (ones, ones));
        assert_eq!(
            (constant.min_zeros(), constant.max_zeros()),
            (width - ones, width - ones)
        );
        assert!(constant.contains(&value));
        assert_eq!(
            unknown.clone().with_popcount_bounds(0, 0).unwrap(),
            KnownBits::constant(&IrBits::zero(width))
        );
        assert_eq!(
            unknown.with_popcount_bounds(width, width).unwrap(),
            KnownBits::constant(&IrBits::all_ones(width))
        );
    }

    let sparse = KnownBits::unknown(4).with_popcount_bounds(1, 2).unwrap();
    assert_eq!(sparse.to_ternary_string(), "XXXX");
    assert_eq!((sparse.min_ones(), sparse.max_ones()), (1, 2));
    assert_eq!((sparse.min_zeros(), sparse.max_zeros()), (2, 3));
    for value in 0..16u64 {
        assert_eq!(
            sparse.contains(&IrBits::make_ubits(4, value).unwrap()),
            (1..=2).contains(&value.count_ones())
        );
    }
    assert!(!sparse.contains(&IrBits::make_ubits(5, 1).unwrap()));
    assert!(KnownBits::unknown(4).with_popcount_bounds(3, 2).is_err());
    assert!(KnownBits::unknown(4).with_popcount_bounds(0, 5).is_err());
    assert!(sparse.clone().with_popcount_bounds(0, 0).is_err());
    assert_eq!(sparse.clone().with_popcount_bounds(0, 4).unwrap(), sparse);

    let masked = KnownBits::from_mask_value(
        IrBits::make_ubits(4, 3).unwrap(),
        IrBits::make_ubits(4, 13).unwrap(),
    )
    .unwrap();
    assert_eq!(masked.to_ternary_string(), "XX01");
    assert_eq!((masked.min_ones(), masked.max_ones()), (1, 3));
    assert!(masked.clone().with_popcount_bounds(0, 0).is_err());
    assert!(masked.clone().with_popcount_bounds(4, 4).is_err());
    assert_eq!(
        masked.clone().with_popcount_bounds(0, 1).unwrap(),
        KnownBits::constant(&IrBits::make_ubits(4, 1).unwrap())
    );
    assert_eq!(
        masked.with_popcount_bounds(3, 4).unwrap(),
        KnownBits::constant(&IrBits::make_ubits(4, 13).unwrap())
    );
}

#[test]
fn one_hot_population_facts_propagate_without_a_known_bit_position() {
    let mut builder = FnBuilder::new("population_flow");
    let input = builder.param("input", Type::Bits(3)).unwrap();
    let decoded = builder.decode(input, Some(8)).unwrap();
    let one_hot = builder.one_hot(input, true).unwrap();
    let any = builder.or_reduce(decoded).unwrap();
    let parity = builder.xor_reduce(decoded).unwrap();
    let complemented = builder.not(decoded).unwrap();
    let result = builder
        .tuple(&[decoded, one_hot, any, parity, complemented])
        .unwrap();
    let function = builder.build(result).unwrap();
    let analysis = analyze_fn(&function).unwrap();
    let result = function.ret_node_ref.unwrap();
    for path in [[0], [1]] {
        let facts = analysis.leaf(result, &path).unwrap();
        assert_eq!((facts.min_ones(), facts.max_ones()), (1, 1));
        assert!(facts.mask().is_zero());
    }
    for path in [[2], [3]] {
        assert_eq!(
            analysis.leaf(result, &path).unwrap(),
            &KnownBits::constant(&IrBits::bool(true))
        );
    }
    let complemented = analysis.leaf(result, &[4]).unwrap();
    assert_eq!((complemented.min_ones(), complemented.max_ones()), (7, 7));
    assert_eq!((complemented.min_zeros(), complemented.max_zeros()), (1, 1));
    let arguments = (0..8).map(|value| vec![bits(3, value)]).collect::<Vec<_>>();
    check_concrete_inputs(&function, &arguments, 0);
    let block = Block::from_function(function.clone(), None).unwrap();
    let block_analysis = analyze_block(&block).unwrap();
    for (node, facts) in analysis.iter() {
        assert_eq!(block_analysis.get(node), Some(facts));
    }
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

/// Checks every concretely evaluated node, not only the function's return.
struct SoundnessObserver<'a, 'ir> {
    analysis: &'a KnownBitsAnalysis<'ir>,
    seen: Vec<bool>,
    seed: u64,
}

impl EvalObserver for SoundnessObserver<'_, '_> {
    fn on_select(&mut self, _event: SelectEvent) {
        // Concrete node values already include the selected result.
    }

    fn on_node_value(&mut self, node_ref: NodeRef, node_text_id: usize, value: &IrValue) {
        let knowledge = self.analysis.get(node_ref).unwrap();
        assert!(
            knowledge.contains(value),
            "seed={} node id={node_text_id}: {knowledge:?} excludes {value}",
            self.seed,
        );
        self.seen[node_ref.index] = true;
    }
}

/// Exercises all node claims under concrete inputs, including parameters.
fn check_concrete_inputs(function: &ir::Fn, arguments: &[Vec<IrValue>], seed: u64) {
    let analysis =
        analyze_fn(function).unwrap_or_else(|error| panic!("seed={seed}: {error}\n{function}"));
    for args in arguments {
        let mut observer = SoundnessObserver {
            analysis: &analysis,
            seen: vec![false; function.nodes.len()],
            seed,
        };
        for (&param, value) in function.params.iter().zip(args) {
            observer.on_node_value(param, function.get_node(param).text_id, value);
        }
        match eval_fn_with_observer(function, args, Some(&mut observer)) {
            FnEvalResult::Success(result) => {
                assert!(
                    analysis
                        .get(function.ret_node_ref.unwrap())
                        .unwrap()
                        .contains(&result.value)
                );
            }
            FnEvalResult::Failure(failure) => {
                panic!("seed={seed}: unexpected evaluation failure {failure:?}\n{function}");
            }
        }
        for (node_ref, _) in analysis.iter() {
            assert!(
                observer.seen[node_ref.index],
                "seed={seed}: concrete evaluator missed node id={}",
                function.get_node(node_ref).text_id,
            );
        }
    }
}

/// Counts scalar claims without confusing empty aggregates with unknown bits.
fn known_bit_count(value: &KnownValue) -> usize {
    match value {
        KnownValue::Bits(bits) => bits.known_bit_count(),
        KnownValue::Tuple(elements) | KnownValue::Array(elements) => {
            elements.iter().map(known_bit_count).sum()
        }
        KnownValue::Token => 0,
    }
}

#[test]
fn random_pure_graph_claims_hold_at_every_node_and_match_block_dataflow() {
    let mut total_claims = 0;
    for seed in 0..32 {
        let options = RandomFnOptions {
            max_params: 3,
            max_nodes: 28,
            max_bit_width: 8,
            max_type_depth: 2,
            max_aggregate_leaves: 8,
            max_array_length: 3,
            max_tuple_length: 3,
            allow_zero_width_bits: seed % 2 == 0,
            allow_arbitrary_width_multiply: true,
            allow_gate: true,
            ..Default::default()
        };
        let mut entropy = RngEntropy::new(StdRng::seed_from_u64(seed));
        let function = generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(24))
            .unwrap()
            .function;
        let arguments = generate_argument_sets_from_seed(&function, seed ^ 0x1234, 16);
        check_concrete_inputs(&function, &arguments, seed);

        let analysis = analyze_fn(&function).unwrap();
        total_claims += analysis
            .iter()
            .map(|(_, value)| known_bit_count(value))
            .sum::<usize>();
        let block = Block::from_function(function.clone(), None).unwrap();
        let block_analysis = analyze_block(&block).unwrap();
        for (node_ref, knowledge) in analysis.iter() {
            assert_eq!(block_analysis.get(node_ref), Some(knowledge), "seed={seed}");
        }
        let output = block.output_ports().next().unwrap();
        assert_eq!(
            block_analysis.get(block.output_value(output)),
            analysis.get(function.ret_node_ref.unwrap()),
        );
        assert_eq!(block_analysis.get(output), Some(&KnownValue::Tuple(vec![])));
    }
    assert!(total_claims > 0, "the sweep must check non-vacuous claims");
}

#[test]
fn tiny_arrays_keep_facts_with_wide_uncertain_indices() {
    let text = r#"package wide_indices
top fn sample(index: bits[129] id=1) -> (bits[4], bits[4], bits[4], bits[4][4], bits[4][4], bits[4][1], bits[4][2]) {
  a: bits[4][4] = literal(value=[1, 3, 7, 15], id=2)
  one: bits[129] = literal(value=1, id=3)
  high: bits[129] = literal(value=0x100000000000000000000000000000000, id=4)
  odd: bits[129] = or(index, one, id=5)
  oob: bits[129] = or(index, high, id=6)
  read: bits[4] = array_index(a, indices=[index], id=7)
  read_odd: bits[4] = array_index(a, indices=[odd], id=8)
  read_oob: bits[4] = array_index(a, indices=[oob], id=9)
  update: bits[4] = literal(value=15, id=10)
  written: bits[4][4] = array_update(a, update, indices=[index], id=11)
  written_oob: bits[4][4] = array_update(a, update, indices=[oob], id=12)
  single: bits[4][1] = literal(value=[1], id=13)
  three: bits[4] = literal(value=3, id=14)
  maybe_written: bits[4][1] = array_update(single, three, indices=[index], id=15)
  sliced: bits[4][2] = array_slice(a, index, width=2, id=16)
  ret result: (bits[4], bits[4], bits[4], bits[4][4], bits[4][4], bits[4][1], bits[4][2]) = tuple(read, read_odd, read_oob, written, written_oob, maybe_written, sliced, id=17)
}
"#;
    let package = Parser::new(text).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let analysis = analyze_fn(function).unwrap();
    let expected = [
        (vec![0], "XXX1"),
        (vec![1], "XX11"),
        (vec![2], "1111"),
        (vec![3, 0], "XXX1"),
        (vec![3, 1], "XX11"),
        (vec![3, 2], "X111"),
        (vec![3, 3], "1111"),
        (vec![4, 0], "0001"),
        (vec![4, 1], "0011"),
        (vec![4, 2], "0111"),
        (vec![4, 3], "1111"),
        // There is only one feasible in-bounds index, but OOB writes are no-ops.
        (vec![5, 0], "00X1"),
        (vec![6, 0], "XXX1"),
        (vec![6, 1], "XX11"),
    ];
    let result = analysis.get(function.ret_node_ref.unwrap()).unwrap();
    for (path, expected) in expected {
        assert_eq!(result.leaf(&path).unwrap().to_ternary_string(), expected);
    }
    let mut arguments = (0..16).map(|i| vec![bits(129, i)]).collect::<Vec<_>>();
    for high_bit in [64, 128] {
        let mut value = vec![false; 129];
        value[high_bit] = true;
        arguments.push(vec![IrValue::from_bits(&IrBits::from_lsb_is_0(&value))]);
    }
    check_concrete_inputs(function, &arguments, 0);
    let block = Block::from_function(function.clone(), None).unwrap();
    let block_analysis = analyze_block(&block).unwrap();
    for (reference, facts) in analysis.iter() {
        assert_eq!(block_analysis.get(reference), Some(facts));
    }
}

#[test]
fn aggregate_leaf_paths_tokens_zero_width_and_dead_nodes_are_preserved() {
    let text = r#"package aggregate_knowledge
top fn sample(x: bits[8] id=1, a: bits[4][2] id=2, tok: token id=3) -> (bits[8], (bits[4][2], bits[0]), token) {
  mask: bits[8] = literal(value=240, id=4)
  masked: bits[8] = and(x, mask, id=5)
  zero: bits[0] = literal(value=0, id=6)
  unused_unit: () = tuple(id=7)
  nested: (bits[4][2], bits[0]) = tuple(a, zero, id=8)
  ret result: (bits[8], (bits[4][2], bits[0]), token) = tuple(masked, nested, tok, id=9)
}
"#;
    let package = Parser::new(text).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let analysis = analyze_fn(function).unwrap();
    let result = function.ret_node_ref.unwrap();
    assert_eq!(
        analysis.leaf(result, &[0]).unwrap().to_ternary_string(),
        "XXXX0000"
    );
    for path in [&[1, 0, 0][..], &[1, 0, 1][..]] {
        assert_eq!(
            analysis.leaf(result, path).unwrap().to_ternary_string(),
            "XXXX"
        );
    }
    let empty_bits = analysis.leaf(result, &[1, 1]).unwrap();
    assert!(empty_bits.is_fully_known());
    assert_eq!(empty_bits.bit_count(), 0);
    assert_eq!(empty_bits.to_ternary_string(), "");
    assert_eq!(
        analysis.get(named(function, "tok")),
        Some(&KnownValue::Token)
    );
    assert_eq!(
        analysis.get(named(function, "unused_unit")),
        Some(&KnownValue::Tuple(vec![]))
    );
    assert!(analysis.leaf(result, &[2]).is_none());
    assert!(analysis.leaf(result, &[1, 0, 2]).is_none());
    assert!(analysis.get(NodeRef { index: usize::MAX }).is_none());
    check_concrete_inputs(
        function,
        &generate_argument_sets_from_seed(function, 7, 12),
        7,
    );
}

#[test]
fn typed_empty_arrays_and_dropped_slots_remain_distinct() {
    let mut builder = FnBuilder::new("empty_shapes");
    let dead = builder.literal(bits(8, 99)).unwrap();
    builder.set_name(dead, "dropped").unwrap();
    let array = builder.array(Type::Bits(129), &[]).unwrap();
    builder.set_name(array, "empty_array").unwrap();
    let unit = builder.tuple(&[]).unwrap();
    let empty_bits = builder.literal(bits(0, 0)).unwrap();
    let result = builder.tuple(&[array, unit, empty_bits]).unwrap();
    let mut function = builder.build(result).unwrap();
    let empty_array = named(&function, "empty_array");
    // Exercise the typed array operation as well as the builder's literal form.
    function.get_node_mut(empty_array).payload = NodePayload::Array(vec![]);
    check_concrete_inputs(&function, &[vec![]], 0);
    // Analysis also accepts holes left by rewrites before graph compaction.
    let dropped = named(&function, "dropped");
    function.get_node_mut(dropped).payload = NodePayload::Nil;
    let analysis = analyze_fn(&function).unwrap();
    assert_eq!(analysis.get(empty_array), Some(&KnownValue::Array(vec![])));
    assert!(analysis.get(NodeRef { index: 0 }).is_none());
    assert!(analysis.get(dropped).is_none());
    assert_eq!(analysis.iter().count(), function.nodes.len() - 2);
    let result = analysis.get(function.ret_node_ref.unwrap()).unwrap();
    assert!(result.matches_type(&function.ret_ty));
    assert!(result.contains(&IrValue::make_tuple(&[
        IrValue::make_array_typed(Type::Bits(129), &[]).unwrap(),
        IrValue::make_tuple(&[]),
        bits(0, 0),
    ])));
}

#[test]
fn arbitrary_width_masks_keep_the_highest_bit_and_all_interior_zeros() {
    for width in [1, 63, 64, 65, 127, 128, 129, 257] {
        let mut mask = vec![false; width];
        mask[0] = true;
        mask[width - 1] = true;
        let mask_value = IrValue::from_bits(&IrBits::from_lsb_is_0(&mask));
        let mut builder = FnBuilder::new("wide_mask");
        let x = builder.param("x", Type::Bits(width)).unwrap();
        let mask = builder.literal(mask_value.clone()).unwrap();
        let masked = builder.and(x, mask).unwrap();
        builder.set_name(masked, "masked").unwrap();
        let result = builder.tuple(&[masked, mask]).unwrap();
        let function = builder.build(result).unwrap();
        let analysis = analyze_fn(&function).unwrap();
        let result = function.ret_node_ref.unwrap();
        let expected = if width == 1 {
            "X".to_string()
        } else {
            format!("X{}X", "0".repeat(width - 2))
        };
        assert_eq!(
            analysis.leaf(result, &[0]).unwrap().to_ternary_string(),
            expected
        );
        assert_eq!(
            analysis.leaf(result, &[1]).unwrap().value(),
            mask_value.as_bits().unwrap()
        );
        assert!(analysis.leaf(result, &[1]).unwrap().is_fully_known());
        check_concrete_inputs(
            &function,
            &generate_argument_sets_from_seed(&function, width as u64, 12),
            width as u64,
        );
    }
}

#[test]
fn current_register_state_is_arbitrary_even_with_reset_and_no_output_ports() {
    for asynchronous in [false, true] {
        for active_low in [false, true] {
            let mut builder = BlockBuilder::new("state_knowledge");
            builder.clock_port("clk").unwrap();
            let reset = builder.input_port("rst", Type::Bits(1)).unwrap();
            builder
                .set_reset(
                    reset,
                    ResetBehavior {
                        asynchronous,
                        active_low,
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
            let unit_user = builder.tuple(&[write]).unwrap();
            builder.set_name(unit_user, "unit_user").unwrap();
            let block = builder.build().unwrap();
            assert_eq!(block.output_ports().count(), 0);
            let analysis = analyze_block(&block).unwrap();
            let current = analysis.bits(named(&block, "current")).unwrap();
            let low = analysis.bits(named(&block, "low")).unwrap();
            assert_eq!(current.to_ternary_string(), "XXXXXXXX");
            assert_eq!(low.to_ternary_string(), "0000XXXX");
            for value in 0..256 {
                assert!(current.contains(bits(8, value).as_bits().unwrap()));
                assert!(low.contains(bits(8, value & 15).as_bits().unwrap()));
            }
            assert_eq!(
                analysis.get(named(&block, "write")),
                Some(&KnownValue::Tuple(vec![]))
            );
            assert_eq!(
                analysis.get(named(&block, "unit_user")),
                Some(&KnownValue::Tuple(vec![KnownValue::Tuple(vec![])]))
            );
        }
    }
}

#[test]
fn hierarchy_outputs_are_opaque_even_when_the_child_is_constant() {
    let mut child = BlockBuilder::new("constant_child");
    let value = child.literal(bits(8, 42)).unwrap();
    child.output_port("value", value).unwrap();
    let mut package = child.build_package("hierarchy_knowledge").unwrap();
    let mut parent = BlockBuilder::new("parent");
    let instance = parent
        .instantiate("child", package.get_top_block().unwrap())
        .unwrap();
    let output = parent.instantiation_output(instance, "value").unwrap();
    parent.set_name(output, "opaque").unwrap();
    parent.output_port("value", output).unwrap();
    parent.build_into_package(&mut package).unwrap();
    let block = package.get_block("parent").unwrap();
    let analysis = analyze_block(block).unwrap();
    assert_eq!(
        analysis
            .bits(named(block, "opaque"))
            .unwrap()
            .to_ternary_string(),
        "XXXXXXXX"
    );
}

#[test]
fn empty_blocks_and_output_sink_users_have_real_unit_facts() {
    let block = BlockBuilder::new("empty").build().unwrap();
    let analysis = analyze_block(&block).unwrap();
    assert_eq!(analysis.iter().count(), 0);
    assert!(analysis.get(NodeRef { index: 0 }).is_none());

    let mut builder = BlockBuilder::new("units");
    let unit = builder.tuple(&[]).unwrap();
    let sink = builder.output_port("first", unit).unwrap();
    let nested_unit = builder.tuple(&[sink]).unwrap();
    builder.output_port("second", nested_unit).unwrap();
    let block = builder.build().unwrap();
    let analysis = analyze_block(&block).unwrap();
    let outputs: Vec<NodeRef> = block.output_ports().collect();
    assert_eq!(
        analysis.get(block.output_value(outputs[0])),
        Some(&KnownValue::Tuple(vec![]))
    );
    assert_eq!(
        analysis.get(block.output_value(outputs[1])),
        Some(&KnownValue::Tuple(vec![KnownValue::Tuple(vec![])]))
    );
    for output in outputs {
        assert_eq!(analysis.get(output), Some(&KnownValue::Tuple(vec![])));
    }
}

#[test]
fn block_analysis_rejects_an_empty_node_graph() {
    let mut block = BlockBuilder::new("missing_sentinel").build().unwrap();
    block.nodes.clear();
    assert_eq!(
        analyze_block(&block).unwrap_err().to_string(),
        "block graph must start with a Nil sentinel"
    );
}

#[test]
fn block_analysis_rejects_a_value_in_the_sentinel_slot() {
    let mut block = BlockBuilder::new("replaced_sentinel").build().unwrap();
    block.nodes[0].ty = Type::Bits(8);
    block.nodes[0].payload = NodePayload::Literal(bits(8, 42));
    assert_eq!(
        analyze_block(&block).unwrap_err().to_string(),
        "block graph must start with a Nil sentinel"
    );
}

#[test]
fn malformed_local_graphs_return_errors_instead_of_knowledge() {
    let mut builder = FnBuilder::new("valid");
    let x = builder.param("x", Type::Bits(8)).unwrap();
    let result = builder.not(x).unwrap();
    let valid = builder.build(result).unwrap();
    let return_node = valid.ret_node_ref.unwrap();
    let mut invalid = valid.clone();
    invalid.get_node_mut(return_node).payload =
        NodePayload::Unop(Unop::Not, NodeRef { index: usize::MAX });
    assert!(analyze_fn(&invalid).is_err());
    let mut invalid = valid.clone();
    invalid.get_node_mut(return_node).payload = NodePayload::Unop(Unop::Not, return_node);
    assert!(analyze_fn(&invalid).is_err());
    let mut invalid = valid.clone();
    invalid.get_node_mut(return_node).payload = NodePayload::Unop(Unop::Not, NodeRef { index: 0 });
    assert!(analyze_fn(&invalid).is_err());
    let mut invalid = valid.clone();
    invalid.get_node_mut(return_node).text_id = valid.get_param(0).text_id;
    assert!(analyze_fn(&invalid).is_err());
    let mut invalid = valid.clone();
    invalid.ret_ty = Type::Bits(9);
    assert!(analyze_fn(&invalid).is_err());
    let mut invalid = valid.clone();
    invalid.params[0] = NodeRef { index: usize::MAX };
    assert!(analyze_fn(&invalid).is_err());
    let mut invalid = valid.clone();
    invalid.nodes.clear();
    assert!(analyze_fn(&invalid).is_err());

    let mut block = Block::from_function(valid, None).unwrap();
    block.get_node_mut(return_node).payload = NodePayload::Unop(Unop::Not, return_node);
    assert!(analyze_block(&block).is_err());
}

#[test]
fn overflowing_one_hot_width_returns_an_analysis_error() {
    let mut builder = FnBuilder::new("one_hot_overflow");
    let input = builder.param("input", Type::Bits(1)).unwrap();
    let result = builder.one_hot(input, true).unwrap();
    let mut function = builder.build(result).unwrap();
    let input = function.params[0];
    let result = function.ret_node_ref.unwrap();
    function.get_node_mut(input).ty = Type::Bits(usize::MAX);
    function.get_node_mut(result).ty = Type::Bits(0);
    function.ret_ty = Type::Bits(0);
    assert!(analyze_fn(&function).is_err());
}

#[test]
fn overflowing_concat_width_returns_an_analysis_error() {
    let mut builder = FnBuilder::new("concat_overflow");
    let lhs = builder.param("lhs", Type::Bits(1)).unwrap();
    let rhs = builder.param("rhs", Type::Bits(1)).unwrap();
    let result = builder.concat(&[lhs, rhs]).unwrap();
    let mut function = builder.build(result).unwrap();
    let lhs = function.params[0];
    let result = function.ret_node_ref.unwrap();
    function.get_node_mut(lhs).ty = Type::Bits(usize::MAX);
    function.get_node_mut(result).ty = Type::Bits(0);
    function.ret_ty = Type::Bits(0);
    assert!(analyze_fn(&function).is_err());
}

#[test]
fn overflowing_array_concat_count_returns_an_analysis_error() {
    let mut builder = FnBuilder::new("array_concat_overflow");
    let ty = Type::new_array(Type::Bits(0), 1);
    let lhs = builder.param("lhs", ty.clone()).unwrap();
    let rhs = builder.param("rhs", ty).unwrap();
    let result = builder.array_concat(&[lhs, rhs]).unwrap();
    let mut function = builder.build(result).unwrap();
    let lhs = function.params[0];
    let result = function.ret_node_ref.unwrap();
    // Zero-bit elements keep the flat width at zero: the overflowing operation
    // is the element-count sum, not the aggregate-width calculation.
    function.get_node_mut(lhs).ty = Type::new_array(Type::Bits(0), usize::MAX);
    let result_type = Type::new_array(Type::Bits(0), 0);
    function.get_node_mut(result).ty = result_type.clone();
    function.ret_ty = result_type;
    assert!(analyze_fn(&function).is_err());
}

#[test]
fn overflowing_aggregate_flat_width_returns_an_analysis_error() {
    let mut builder = FnBuilder::new("aggregate_width_overflow");
    let input = builder
        .param("input", Type::new_array(Type::Bits(1), 2))
        .unwrap();
    let mut function = builder.build(input).unwrap();
    let result = function.ret_node_ref.unwrap();
    let result_type = Type::new_array(Type::Bits(usize::MAX), 2);
    function.get_node_mut(result).ty = result_type.clone();
    function.ret_ty = result_type;
    assert!(analyze_fn(&function).is_err());
}
