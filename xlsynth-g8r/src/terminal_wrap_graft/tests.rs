// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::aig_sim::gate_sim::{Collect, eval};
use xlsynth_pir::IrBits;

#[test]
fn rematerialized_word_preserves_all_ordered_signed_scalar_uses() {
    let mut builder = GateBuilder::new("terminal".into(), GateBuilderOptions::opt());
    let input = builder.add_input("x".into(), 2);
    let carry = *builder.add_input("carry".into(), 1).get_lsb(0);
    let a = *input.get_lsb(0);
    let b = *input.get_lsb(1);
    let computed = builder.add_and_binary(a, b);
    let word = vec![a, computed, b.negate()];
    let ones = builder.add_and_binary(a, computed);
    let ones = builder.add_and_binary(ones, b.negate());
    let zeros = builder.add_and_binary(a.negate(), computed.negate());
    let zeros = builder.add_and_binary(zeros, b);
    let p = builder.add_mux2(carry, ones, zeros).negate();
    let ports = vec![p.negate(), builder.get_true(), p, p.negate(), p];
    builder.add_output("ret".into(), AigBitVector::from_lsb_is_index_0(&ports));
    let stock = builder.build();
    let ob = TerminalWrapCapture {
        predicate_pir_id: None,
        word_lsb: word,
        carry,
        equal: false,
        outputs: vec![vec![
            ReturnBit::Predicate { inverted: true },
            ReturnBit::Constant(true),
            ReturnBit::Predicate { inverted: false },
            ReturnBit::Predicate { inverted: true },
            ReturnBit::Predicate { inverted: false },
        ]],
    };
    let new = TerminalWrapPlan::new(&stock, ob)
        .unwrap()
        .apply(stock.clone())
        .unwrap();
    for x in 0..4 {
        for c in 0..2 {
            let args = [
                IrBits::make_ubits(2, x).unwrap(),
                IrBits::make_ubits(1, c).unwrap(),
            ];
            let word = (x & 1) | (((x & 1) & ((x >> 1) & 1)) << 1) | ((1 ^ ((x >> 1) & 1)) << 2);
            let p = u64::from(((word + c) & 7) != 0);
            let expected =
                IrBits::make_ubits(5, (1 ^ p) | 2 | (p << 2) | ((1 ^ p) << 3) | (p << 4)).unwrap();
            assert_eq!(eval(&new, &args, Collect::None).outputs, vec![expected]);
        }
    }
}

#[test]
fn distinct_equivalent_carry_cones_preserve_the_existing_graph() {
    let mut builder = GateBuilder::new("shared".into(), GateBuilderOptions::opt());
    let word = builder.add_input("w".into(), 3);
    let x = builder.add_input("x".into(), 3);
    let carry = builder.add_and_binary(*x.get_lsb(0), *x.get_lsb(1));
    let q = *x.get_lsb(2);
    let low = builder.add_and_binary(carry, q);
    let high = builder.add_and_binary(carry, q.negate());
    let alternate = builder.add_and_binary(low.negate(), high.negate()).negate();
    let w: Vec<_> = word.iter_lsb_to_msb().copied().collect();
    let all = builder.add_and_binary(w[0], w[1]);
    let all = builder.add_and_binary(all, w[2]);
    let none = builder.add_and_binary(w[0].negate(), w[1].negate());
    let none = builder.add_and_binary(none, w[2].negate());
    let low = builder.add_and_binary(none, carry.negate());
    let high = builder.add_and_binary(all, alternate);
    let p = builder.add_and_binary(low.negate(), high.negate()).negate();
    builder.add_output("ret".into(), AigBitVector::from_bit(p));
    let stock = builder.build();
    let ob = TerminalWrapCapture {
        predicate_pir_id: None,
        word_lsb: w,
        carry,
        equal: true,
        outputs: vec![vec![ReturnBit::Predicate { inverted: false }]],
    };
    let new = TerminalWrapPlan::new(&stock, ob)
        .unwrap()
        .apply(stock.clone())
        .unwrap();
    assert_eq!(new.gates, stock.gates);
    assert!(
        new.outputs[0]
            .bit_vector
            .iter_lsb_to_msb()
            .eq(stock.outputs[0].bit_vector.iter_lsb_to_msb())
    );
}

#[test]
fn hypothetical_bindings_preserve_prefix_provenance_and_track_new_gates() {
    use crate::aig_serdes::emit_aiger_binary::emit_aiger_binary;

    let run = |track: bool| {
        let mut b = GateBuilder::new("provenance".into(), GateBuilderOptions::opt());
        b.set_current_pir_node_id(track.then_some(1));
        let word = b.add_input("w".into(), 5);
        b.set_current_pir_node_id(track.then_some(2));
        let external = b.add_input("x".into(), 2);
        b.set_current_pir_node_id(track.then_some(3));
        let carry = b.add_and_binary(*external.get_lsb(0), *external.get_lsb(1));
        b.add_tag(carry.node, "real carry".into());
        let w: Vec<_> = word.iter_lsb_to_msb().copied().collect();
        b.set_current_pir_node_id(track.then_some(4));
        let all = w
            .iter()
            .copied()
            .reduce(|x, y| b.add_and_binary(x, y))
            .unwrap();
        let none = w
            .iter()
            .map(|x| x.negate())
            .reduce(|x, y| b.add_and_binary(x, y))
            .unwrap();
        let low = b.add_and_binary(none, carry.negate());
        let high = b.add_and_binary(all, carry);
        b.add_tag(high.node, "old endpoint".into());
        let p = b.add_and_binary(low.negate(), high.negate()).negate();
        b.set_current_pir_node_id(None);
        b.add_output("ret".into(), AigBitVector::from_bit(p));
        let stock = b.build();
        let observation = TerminalWrapCapture {
            predicate_pir_id: track.then_some(4),
            word_lsb: w,
            carry,
            equal: true,
            outputs: vec![vec![ReturnBit::Predicate { inverted: false }]],
        };
        let new = TerminalWrapPlan::new(&stock, observation)
            .unwrap()
            .apply(stock)
            .unwrap();
        assert!(!new.gates.iter().any(|g| {
            g.get_tags().is_some_and(|tags| {
                tags.iter().any(|s| s == "real carry") && tags.iter().any(|s| s == "old endpoint")
            })
        }));
        if track {
            assert!(
                new.get(new.outputs[0].bit_vector.get_lsb(0).node)
                    .get_pir_node_ids()
                    .contains(&4)
            );
            for gate in &new.gates {
                match gate {
                    AigNode::Literal { .. } => assert!(gate.get_pir_node_ids().is_empty()),
                    AigNode::Input { name, .. } => {
                        assert_eq!(gate.get_pir_node_ids(), [if name == "w" { 1 } else { 2 }])
                    }
                    AigNode::And2 { .. }
                        if gate
                            .get_tags()
                            .is_some_and(|tags| tags.iter().any(|s| s == "real carry")) =>
                    {
                        assert_eq!(gate.get_pir_node_ids(), [3]);
                    }
                    AigNode::And2 { .. } => assert!(gate.get_pir_node_ids().contains(&4)),
                }
            }
        } else {
            assert!(
                new.gates
                    .iter()
                    .all(|gate| gate.get_pir_node_ids().is_empty())
            );
        }
        emit_aiger_binary(&new, true).unwrap()
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn declined_replacement_preserves_stock_aliases_dead_gates_and_provenance() {
    fn graph(expand: bool, dead: bool) -> GateFn {
        let mut b = GateBuilder::new("aliases".into(), GateBuilderOptions::no_opt());
        let x = b.add_input("x".into(), 3);
        b.set_current_pir_node_id(Some(17));
        let p = b.add_and_binary(*x.get_lsb(0), *x.get_lsb(1));
        let z = *x.get_lsb(2);
        let p = if expand {
            let a = b.add_and_binary(p, z);
            let c = b.add_and_binary(p, z.negate());
            b.add_or_binary(a, c)
        } else {
            p
        };
        b.add_tag(p.node, "observed".into());
        b.add_output(
            "signed".into(),
            AigBitVector::from_lsb_is_index_0(&[p, p.negate(), p]),
        );
        b.add_output("again".into(), AigBitVector::from_bit(p.negate()));
        if dead {
            b.set_current_pir_node_id(Some(99));
            let mut spare = z;
            for _ in 0..9 {
                spare = b.add_and_binary(spare, p);
            }
            b.add_tag(spare.node, "unreachable".into());
        }
        b.build()
    }
    for expanded in [false, true] {
        let stock = graph(false, true);
        let proposed = graph(expanded, false);
        assert!(stock.gates.len() > proposed.gates.len());
        let expected = bincode::serialize(&stock).unwrap();
        let diagnostic = stock.to_string_including_dead_nodes();
        let original_storage = stock.gates.as_ptr();
        let result = choose_replacement(stock, proposed);
        assert_eq!(result.gates.as_ptr(), original_storage);
        assert_eq!(bincode::serialize(&result).unwrap(), expected);
        assert_eq!(result.to_string_including_dead_nodes(), diagnostic);
    }
}

#[test]
fn equal_area_can_reduce_depth_but_more_live_ands_are_declined() {
    fn reduction(balanced: bool, redundant: bool) -> GateFn {
        let mut b = GateBuilder::new("reduce".into(), GateBuilderOptions::no_opt());
        let x = b.add_input("x".into(), 8);
        let p = if balanced {
            let mut level: Vec<_> = x.iter_lsb_to_msb().copied().collect();
            while level.len() > 1 {
                level = level
                    .chunks_exact(2)
                    .map(|pair| b.add_and_binary(pair[0], pair[1]))
                    .collect();
            }
            level[0]
        } else {
            x.iter_lsb_to_msb()
                .copied()
                .reduce(|a, c| b.add_and_binary(a, c))
                .unwrap()
        };
        let p = if redundant {
            b.add_and_binary(p, *x.get_lsb(0))
        } else {
            p
        };
        b.add_output("ret".into(), AigBitVector::from_bit(p));
        b.build()
    }
    let stock = reduction(false, false);
    let balanced = reduction(true, false);
    let balanced_bytes = bincode::serialize(&dce(&balanced)).unwrap();
    let got = choose_replacement(stock.clone(), balanced);
    assert_eq!(bincode::serialize(&got).unwrap(), balanced_bytes);

    let small_but_late = reduction(false, false);
    let fast = reduction(true, true);
    let old_bytes = bincode::serialize(&fast).unwrap();
    let actual = choose_replacement(fast, small_but_late);
    assert_eq!(bincode::serialize(&actual).unwrap(), old_bytes);

    let bigger_but_shorter = reduction(true, true);
    let stock_bytes = bincode::serialize(&stock).unwrap();
    let got = choose_replacement(stock, bigger_but_shorter);
    assert_eq!(bincode::serialize(&got).unwrap(), stock_bytes);
}

#[test]
fn signed_cube_joins_the_latest_arriving_operand_last() {
    use crate::aig::get_summary_stats::get_output_cone_stats;
    use crate::aig::reassociation::and_cube_with_arrivals;
    let mut builder = GateBuilder::new("arrival".into(), GateBuilderOptions::no_opt());
    let x = builder.add_input("x".into(), 8);
    let bits: Vec<_> = x.iter_lsb_to_msb().copied().collect();
    let late = bits[..5]
        .iter()
        .copied()
        .reduce(|a, b| builder.add_and_binary(a, b))
        .unwrap();
    let mut depths = Vec::new();
    let leaf = bits[5];
    let unit = and_cube_with_arrivals(&mut builder, &mut depths, []);
    assert_eq!(unit, builder.get_true());
    let opposite = and_cube_with_arrivals(&mut builder, &mut depths, [leaf, leaf.negate()]);
    assert_eq!(opposite, builder.get_false());
    let repeated = and_cube_with_arrivals(
        &mut builder,
        &mut depths,
        [leaf.negate(), leaf.negate(), unit],
    );
    assert_eq!(repeated, leaf.negate());
    let result = and_cube_with_arrivals(
        &mut builder,
        &mut depths,
        [late.negate(), bits[5], bits[6], bits[7]],
    );
    builder.add_output("result".into(), AigBitVector::from_bit(result));
    assert_eq!(get_output_cone_stats(&builder.build()).root_depths, vec![5]);
}

#[test]
fn declined_endpoint_cut_does_not_color_a_reused_word_gate() {
    // The regular cube treats a&b as one leaf. Expanding that leaf recreates
    // the same signed high root: both build (a&b) & (d&carry). Only the
    // speculative arm constructs a&b on behalf of the predicate.
    let mut b = GateBuilder::new("tracked_wrap".into(), GateBuilderOptions::no_opt());
    let pins = b.add_input("x".into(), 5);
    let a = *pins.get_lsb(0);
    let bb = *pins.get_lsb(1);
    let d = *pins.get_lsb(2);
    let carry = *pins.get_lsb(3);
    let mask = *pins.get_lsb(4);
    let word_bit = b.add_and_binary(a, bb);
    b.add_tag(word_bit.node, "word[0]".into());
    b.add_pir_node_id(word_bit.node, 17);
    let high_pair = b.add_and_binary(d, carry);
    let high = b.add_and_binary(word_bit, high_pair);
    let low_pair = b.add_and_binary(d.negate(), carry.negate());
    let low = b.add_and_binary(word_bit.negate(), low_pair);
    let regular = b.add_or_binary(low, high);
    // A redundant Shannon expansion makes stock strictly larger than regular.
    let yes = b.add_and_binary(regular, mask);
    let no = b.add_and_binary(regular, mask.negate());
    let stock_root = b.add_or_binary(yes, no);
    b.add_pir_node_id(stock_root.node, 99);
    b.add_output("eq".into(), AigBitVector::from_bit(stock_root));
    b.add_output("ne".into(), AigBitVector::from_bit(stock_root.negate()));
    let stock = b.build();
    let observation = TerminalWrapCapture {
        predicate_pir_id: Some(99),
        word_lsb: vec![word_bit, d],
        carry,
        equal: true,
        outputs: vec![
            vec![ReturnBit::Predicate { inverted: false }],
            vec![ReturnBit::Predicate { inverted: true }],
        ],
    };
    let plan = TerminalWrapPlan::new(&stock, observation).unwrap();
    let word = &plan.rails.outputs[0].bit_vector;
    let cuts = endpoints::analyze(&plan.rails, word);
    assert!(matches!(&cuts[1], endpoints::Cut::Literals(atoms) if atoms.len() == 3));

    let gates = plan.apply(stock).unwrap();
    let bytes = crate::aig_serdes::g8r::encode_g8r_binary(
        &crate::aig::SequentialGateFn::from_gate_fn(gates),
    )
    .unwrap();
    let gates = crate::aig_serdes::g8r::decode_g8r_binary(&bytes)
        .unwrap()
        .try_into_gate_fn()
        .unwrap();
    let area = crate::aig::get_summary_stats::get_output_cone_stats(&gates).and_nodes;
    assert_eq!(
        area, 6,
        "the regular endpoint graph must win over the padded stock"
    );
    assert_eq!(
        (&gates.outputs[0].name, &gates.outputs[1].name),
        (&"eq".to_string(), &"ne".to_string())
    );
    let eq = *gates.outputs[0].bit_vector.get_lsb(0);
    assert_eq!(gates.outputs[1].bit_vector.get_lsb(0), &eq.negate());
    assert!(gates.get(eq.node).get_pir_node_ids().contains(&99));
    let mut words = gates.gates.iter().filter(|g| {
        g.get_tags()
            .is_some_and(|tags| tags.iter().any(|tag| tag == "word[0]"))
    });
    assert_eq!(words.next().unwrap().get_pir_node_ids(), &[17]);
    assert!(words.next().is_none());
}

#[test]
fn impossible_endpoints_keep_the_predicate_id_on_the_returned_constant() {
    let mut b = GateBuilder::new("constant_wrap".into(), GateBuilderOptions::no_opt());
    let pins = b.add_input("x".into(), 3);
    let a = *pins.get_lsb(0);
    let bb = *pins.get_lsb(1);
    let carry = *pins.get_lsb(2);
    let ab = b.add_and_binary(a, bb);
    let word = vec![a, bb, ab.negate()];
    // Neither uniform endpoint of (a, b, !(a&b)) can be reached.
    let ones = b.add_and_binary(ab, ab.negate());
    let nz = b.add_and_binary(a.negate(), bb.negate());
    let zero = b.add_and_binary(nz, ab);
    let high = b.add_and_binary(ones, carry);
    let low = b.add_and_binary(zero, carry.negate());
    let old = b.add_or_binary(high, low);
    b.add_pir_node_id(old.node, 99);
    b.add_output("ne".into(), AigBitVector::from_bit(old.negate()));
    b.add_output("eq".into(), AigBitVector::from_bit(old));
    let stock = b.build();
    let plan = TerminalWrapPlan::new(
        &stock,
        TerminalWrapCapture {
            predicate_pir_id: Some(99),
            word_lsb: word,
            carry,
            equal: false,
            outputs: vec![
                vec![ReturnBit::Predicate { inverted: false }],
                vec![ReturnBit::Predicate { inverted: true }],
            ],
        },
    )
    .unwrap();
    let cuts = endpoints::analyze(&plan.rails, &plan.rails.outputs[0].bit_vector);
    assert_eq!(
        cuts,
        [endpoints::Cut::Impossible, endpoints::Cut::Impossible]
    );
    let graph = plan.apply(stock).unwrap();
    let ne = *graph.outputs[0].bit_vector.get_lsb(0);
    let eq = *graph.outputs[1].bit_vector.get_lsb(0);
    assert_eq!(ne, eq.negate());
    assert!(matches!(graph.get(eq.node), AigNode::Literal { value, .. } if !(*value ^ eq.negated)));
    assert!(graph.get(eq.node).get_pir_node_ids().contains(&99));
}
