// SPDX-License-Identifier: Apache-2.0

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{self, Block, NodeGraph, NodeRef};
use xlsynth_pir::ir_eval::{EvalObserver, FnEvalResult, SelectEvent, eval_fn_with_observer};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_random::{RandomFnOptions, RngEntropy, StopPolicy, generate_fn};
use xlsynth_pir::known_bits::{KnownBits, KnownBitsAnalysis, analyze_block, analyze_fn};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;

fn value(width: usize, value: u64) -> IrValue {
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

fn assert_counts(facts: &KnownBits, minimum: usize, maximum: usize) {
    assert_eq!(facts.min_ones(), minimum, "{facts:?}");
    assert_eq!(facts.max_ones(), maximum, "{facts:?}");
    assert_eq!(facts.min_zeros(), facts.bit_count() - maximum);
    assert_eq!(facts.max_zeros(), facts.bit_count() - minimum);
}

/// Checks both packed bits and popcount constraints at every concrete node.
struct ContainsObserver<'a, 'ir> {
    analysis: &'a KnownBitsAnalysis<'ir>,
    seen: Vec<bool>,
}

impl EvalObserver for ContainsObserver<'_, '_> {
    fn on_select(&mut self, _event: SelectEvent) {
        // Node-value callbacks already report the actual selected result.
    }

    fn on_node_value(&mut self, node: NodeRef, text_id: usize, value: &IrValue) {
        let facts = self.analysis.get(node).unwrap();
        assert!(
            facts.contains(value),
            "node id={text_id}: {facts:?} excludes {value}"
        );
        self.seen[node.index] = true;
    }
}

/// Exercises complete graph facts and the equivalent block dataflow interface.
fn check_concrete_values(function: &ir::Fn, arguments: &[Vec<IrValue>]) {
    let analysis = analyze_fn(function).unwrap();
    for args in arguments {
        let mut observer = ContainsObserver {
            analysis: &analysis,
            seen: vec![false; function.nodes.len()],
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
            FnEvalResult::Failure(failure) => panic!("unexpected evaluation failure: {failure:?}"),
        }
        for (node, _) in analysis.iter() {
            assert!(
                observer.seen[node.index],
                "concrete evaluator missed node {node:?}"
            );
        }
    }

    let block = Block::from_function(function.clone(), None).unwrap();
    let block_analysis = analyze_block(&block).unwrap();
    for (node, facts) in analysis.iter() {
        assert_eq!(block_analysis.get(node), Some(facts));
    }
    let output = block.output_ports().next().unwrap();
    assert_eq!(
        block_analysis.get(block.output_value(output)),
        analysis.get(function.ret_node_ref.unwrap()),
    );
}

#[test]
fn routed_one_hot_counts_refine_selects_without_selector_definition_peeks() {
    let package = Parser::new(
        r#"package routed_one_hot
top fn sample(x: bits[2] id=1, raw: bits[3] id=2, index: bits[1] id=3) -> (bits[4], bits[4], bits[4], bits[4], bits[4], bits[4][1], bits[4]) {
  hot: bits[3] = one_hot(x, lsb_prio=true, id=4)
  copied: bits[3] = identity(hot, id=5)
  packed: (bits[3], bits[3]) = tuple(hot, copied, id=6)
  unpacked: bits[3] = tuple_index(packed, index=1, id=7)
  alternatives: bits[3][2] = array(copied, unpacked, id=8)
  routed: bits[3] = array_index(alternatives, indices=[index], id=9)
  extended: bits[5] = zero_ext(routed, new_bit_count=5, id=10)
  sliced: bits[3] = bit_slice(extended, start=0, width=3, id=11)
  zero: bits[4] = literal(value=0, id=12)
  one: bits[4] = literal(value=1, id=13)
  two: bits[4] = literal(value=2, id=14)
  four: bits[4] = literal(value=4, id=15)
  eight: bits[4] = literal(value=8, id=16)
  chosen: bits[4] = one_hot_sel(extended, cases=[one, two, four, eight, eight], id=17)
  prioritized: bits[4] = priority_sel(sliced, cases=[one, two, four], default=zero, id=18)
  raw_chosen: bits[4] = one_hot_sel(raw, cases=[one, two, four], id=19)
  raw_prioritized: bits[4] = priority_sel(raw, cases=[one, two, four], default=zero, id=20)
  indexed: bits[4] = sel(routed, cases=[zero, one, two, zero, four, zero, zero, zero], id=21)
  lookup: bits[4][8] = array(zero, one, two, zero, four, zero, zero, zero, id=22)
  read: bits[4] = array_index(lookup, indices=[routed], id=23)
  read_slice: bits[4][1] = array_slice(lookup, routed, width=1, id=24)
  forwarded: bits[4] = identity(chosen, id=25)
  next: bits[4] = priority_sel(forwarded, cases=[one, two, four, eight], default=zero, id=26)
  ret result: (bits[4], bits[4], bits[4], bits[4], bits[4], bits[4][1], bits[4]) = tuple(chosen, prioritized, raw_chosen, raw_prioritized, indexed, read_slice, next, id=27)
}
"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let function = package.get_top_fn().unwrap();
    let analysis = analyze_fn(function).unwrap();
    for name in ["hot", "copied", "unpacked", "routed", "extended", "sliced"] {
        assert_counts(analysis.bits(named(function, name)).unwrap(), 1, 1);
    }
    for name in [
        "chosen",
        "prioritized",
        "indexed",
        "read",
        "forwarded",
        "next",
    ] {
        let facts = analysis.bits(named(function, name)).unwrap();
        assert_counts(facts, 1, 1);
        assert!(!facts.contains(value(4, 0).as_bits().unwrap()));
        assert!(!facts.contains(value(4, 3).as_bits().unwrap()));
    }
    assert_counts(
        analysis.leaf(named(function, "packed"), &[0]).unwrap(),
        1,
        1,
    );
    assert_counts(
        analysis
            .leaf(named(function, "alternatives"), &[1])
            .unwrap(),
        1,
        1,
    );
    assert_counts(
        analysis.leaf(named(function, "read_slice"), &[0]).unwrap(),
        1,
        1,
    );
    assert_counts(analysis.bits(named(function, "raw")).unwrap(), 0, 3);
    assert_counts(analysis.bits(named(function, "raw_chosen")).unwrap(), 0, 3);
    assert_counts(
        analysis.bits(named(function, "raw_prioritized")).unwrap(),
        0,
        1,
    );

    let arguments = (0..4)
        .flat_map(|x| {
            (0..8).flat_map(move |raw| {
                (0..2).map(move |index| vec![value(2, x), value(3, raw), value(1, index)])
            })
        })
        .collect::<Vec<_>>();
    check_concrete_values(function, &arguments);
}

#[test]
fn decode_counts_distinguish_total_and_out_of_bounds_domains() {
    let package = Parser::new(
        r#"package decode_counts
top fn sample(index: bits[3] id=1) -> (bits[4], bits[4], bits[4], bits[3], bits[8]) {
  partial: bits[3] = decode(index, width=3, id=2)
  total: bits[8] = decode(index, width=8, id=3)
  partial_copy: bits[3] = identity(partial, id=4)
  total_copy: bits[8] = identity(total, id=5)
  zero: bits[4] = literal(value=0, id=6)
  seven: bits[4] = literal(value=7, id=7)
  maybe: bits[4] = one_hot_sel(partial_copy, cases=[seven, seven, seven], id=8)
  always: bits[4] = one_hot_sel(total_copy, cases=[seven, seven, seven, seven, seven, seven, seven, seven], id=9)
  prioritized: bits[4] = priority_sel(partial_copy, cases=[seven, seven, seven], default=zero, id=10)
  ret result: (bits[4], bits[4], bits[4], bits[3], bits[8]) = tuple(maybe, always, prioritized, partial, total, id=11)
}
"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let function = package.get_top_fn().unwrap();
    let analysis = analyze_fn(function).unwrap();
    assert_counts(analysis.bits(named(function, "partial")).unwrap(), 0, 1);
    assert_counts(analysis.bits(named(function, "total")).unwrap(), 1, 1);
    for name in ["maybe", "prioritized"] {
        let facts = analysis.bits(named(function, name)).unwrap();
        assert_counts(facts, 0, 3);
        assert!(facts.contains(value(4, 0).as_bits().unwrap()));
    }
    assert_eq!(
        analysis.bits(named(function, "always")).unwrap(),
        &KnownBits::constant(value(4, 7).as_bits().unwrap()),
    );
    check_concrete_values(
        function,
        &(0..8)
            .map(|index| vec![value(3, index)])
            .collect::<Vec<_>>(),
    );
}

#[test]
fn nonzero_multihot_selectors_keep_common_bits_without_one_hot_assumptions() {
    let package = Parser::new(
        r#"package multihot
top fn sample(x: bits[2] id=1, raw: bits[6] id=2) -> (bits[4], bits[4]) {
  hot: bits[3] = one_hot(x, lsb_prio=false, id=3)
  both: bits[6] = concat(hot, hot, id=4)
  routed: bits[6] = identity(both, id=5)
  one: bits[4] = literal(value=1, id=6)
  three: bits[4] = literal(value=3, id=7)
  five: bits[4] = literal(value=5, id=8)
  nonzero: bits[4] = one_hot_sel(routed, cases=[one, three, five, one, three, five], id=9)
  maybe_zero: bits[4] = one_hot_sel(raw, cases=[one, three, five, one, three, five], id=10)
  ret result: (bits[4], bits[4]) = tuple(nonzero, maybe_zero, id=11)
}
"#,
    )
    .parse_and_validate_package()
    .unwrap();
    let function = package.get_top_fn().unwrap();
    let analysis = analyze_fn(function).unwrap();
    assert_counts(analysis.bits(named(function, "routed")).unwrap(), 2, 2);
    assert_eq!(
        analysis
            .bits(named(function, "nonzero"))
            .unwrap()
            .to_ternary_string(),
        "0XX1"
    );
    assert_eq!(
        analysis
            .bits(named(function, "maybe_zero"))
            .unwrap()
            .to_ternary_string(),
        "0XXX"
    );
    let arguments = (0..4)
        .flat_map(|x| (0..64).map(move |raw| vec![value(2, x), value(6, raw)]))
        .collect::<Vec<_>>();
    check_concrete_values(function, &arguments);
}

#[test]
fn random_graph_concrete_values_satisfy_all_bit_and_popcount_facts() {
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
        let arguments =
            generate_argument_sets_from_seed(&function, seed ^ 0x706f_7063_6f75_6e74, 16);
        check_concrete_values(&function, &arguments);
    }
}
