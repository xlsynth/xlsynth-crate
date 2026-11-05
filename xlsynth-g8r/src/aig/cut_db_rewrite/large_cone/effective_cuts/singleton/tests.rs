// SPDX-License-Identifier: Apache-2.0

use super::SingletonSignatures;
use crate::aig::dynamic_structural_hash::DynamicStructuralHash;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

fn builder() -> GateBuilder {
    GateBuilder::new("singleton".into(), GateBuilderOptions::no_opt())
}

fn input(g: &mut GateBuilder, name: &str) -> AigOperand {
    *g.add_input(name.into(), 1).get_lsb(0)
}

fn outputs(g: &mut GateBuilder, bits: &[AigOperand]) {
    g.add_output("out".into(), AigBitVector::from_lsb_is_index_0(bits));
}

#[test]
fn a_new_complementary_physical_gate_invalidates_a_previously_unique_root() {
    let mut g = builder();
    let a = input(&mut g, "a");
    let b = input(&mut g, "b");
    let root = g.add_and_binary(a, b);
    outputs(&mut g, &[root, a]);
    let mut state = DynamicStructuralHash::new(g.build()).unwrap();
    let mut signatures = SingletonSignatures::default();
    assert!(signatures.cannot_save(&state, root.node));

    let equivalent_complement = state.add_and(root.negate(), root.negate()).unwrap();
    assert_ne!(equivalent_complement.node, root.node);
    state.move_output_edge(0, 1, equivalent_complement).unwrap();
    assert!(!signatures.cannot_save(&state, root.node));

    state.move_output_edge(0, 1, a).unwrap();
    state.delete_node(equivalent_complement.node).unwrap();
    assert!(!state.is_live(equivalent_complement.node));
    // Within a round the removed alias still conservatively counts.
    assert!(!signatures.cannot_save(&state, root.node));
}

#[test]
fn constant_function_cannot_be_unique_against_both_constant_literals() {
    let mut g = builder();
    let a = input(&mut g, "a");
    let zero = g.add_and_binary(a, a.negate());
    outputs(&mut g, &[zero.negate()]);
    let state = DynamicStructuralHash::new(g.build()).unwrap();
    assert!(!SingletonSignatures::default().cannot_save(&state, zero.node));
}

#[test]
fn each_internal_fanin_needs_a_use_surviving_the_root() {
    for pin_child in [false, true] {
        let mut g = builder();
        let a = input(&mut g, "a");
        let b = input(&mut g, "b");
        let c = input(&mut g, "c");
        let child = g.add_and_binary(a, b);
        let root = g.add_and_binary(child, c);
        outputs(&mut g, &[root, if pin_child { child.negate() } else { a }]);
        let state = DynamicStructuralHash::new(g.build()).unwrap();
        assert_eq!(state.use_count(child.node), 1 + usize::from(pin_child));
        assert_eq!(
            SingletonSignatures::default().cannot_save(&state, root.node),
            pin_child
        );
    }

    let mut g = builder();
    let a = input(&mut g, "a");
    let b = input(&mut g, "b");
    let child = g.add_and_binary(a, b);
    let root = g.add_and_binary(child, child);
    outputs(&mut g, &[root]);
    let state = DynamicStructuralHash::new(g.build()).unwrap();
    assert_eq!(state.use_count(child.node), 2);
    assert_eq!(state.fanout_count(child.node), 1);
    assert!(!SingletonSignatures::default().cannot_save(&state, root.node));
}

#[test]
fn duplicate_keys_require_at_most_one_distinct_root_fanout() {
    let mut g = builder();
    let a = input(&mut g, "a");
    let b = input(&mut g, "b");
    let c = input(&mut g, "c");
    let root = g.add_and_binary(a, b);
    let repeated_use = g.add_and_binary(root, root.negate());
    let duplicate_a = g.add_and_binary(a, c);
    let duplicate_b = g.add_and_binary(c, a);
    outputs(&mut g, &[repeated_use, duplicate_a, duplicate_b, a]);
    let mut state = DynamicStructuralHash::new(g.build()).unwrap();
    let mut signatures = SingletonSignatures::default();

    assert!(!state.has_unique_and_keys());
    assert_eq!(state.use_count(root.node), 2);
    assert_eq!(state.fanout_count(root.node), 1);
    assert!(signatures.cannot_save(&state, root.node));
    let other_consumer = state.add_and(root, c).unwrap();
    state.move_output_edge(0, 3, other_consumer).unwrap();
    assert_eq!(state.fanout_count(root.node), 2);
    assert!(!signatures.cannot_save(&state, root.node));
}

#[test]
fn later_numbered_fanins_must_not_hide_a_complementary_alias() {
    let mut g = builder();
    let a = input(&mut g, "a");
    let b = input(&mut g, "b");
    let alias = g.add_and_binary(a, a);
    let root = g.add_and_binary(a, b);
    outputs(&mut g, &[alias, root]);
    let mut graph = g.build();
    let AigNode::And2 { a, b, .. } = &mut graph.gates[alias.node.id] else {
        unreachable!()
    };
    *a = root.negate();
    *b = root.negate();
    assert!(alias.node.id < root.node.id);
    let state = DynamicStructuralHash::new(graph).unwrap();
    assert!(!SingletonSignatures::default().cannot_save(&state, root.node));
}

#[test]
fn rare_nonconstant_function_matching_zero_word_remains_searchable() {
    let mut g = builder();
    let inputs: Vec<_> = (0..24)
        .map(|i| input(&mut g, &format!("rare{i}")))
        .collect();
    let mut conjunction = inputs[0];
    for op in &inputs[1..23] {
        conjunction = g.add_and_binary(conjunction, *op);
    }
    let root = g.add_and_binary(conjunction, inputs[23]);
    // Both true and false assignments exist. The direct output of the internal
    // child isolates the sampled collision as the sole failed precondition.
    outputs(&mut g, &[conjunction, root]);
    let state = DynamicStructuralHash::new(g.build()).unwrap();
    assert!(state.has_unique_and_keys());
    assert_eq!(state.use_count(conjunction.node), 2);
    let mut signatures = SingletonSignatures::default();
    assert!(!signatures.cannot_save(&state, root.node));
    assert_eq!(signatures.words[root.node.id], 0);
}
