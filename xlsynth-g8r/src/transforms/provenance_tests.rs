// SPDX-License-Identifier: Apache-2.0

use crate::aig::gate::{AigNode, AigRef, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::transforms::and_absorb::and_absorb_right_primitive;
use crate::transforms::balance_and_tree::{BalanceAndTreeTransform, UnbalanceAndTreeTransform};
use crate::transforms::double_negate::DoubleNegateTransform;
use crate::transforms::duplicate::UnduplicateGateTransform;
use crate::transforms::factor_shared_and::{FactorSharedAndTransform, UnfactorSharedAndTransform};
use crate::transforms::false_and::{insert_false_and_primitive, remove_false_and_primitive};
use crate::transforms::merge_equiv_leaves::MergeEquivLeavesTransform;
use crate::transforms::push_negation::PushNegationTransform;
use crate::transforms::redundant_and::{
    insert_redundant_and_primitive, remove_redundant_and_primitive,
};
use crate::transforms::rewire_operand::rewire_operand_primitive;
use crate::transforms::rotate_and::rotate_and_right_primitive;
use crate::transforms::split_fanout::{merge_fanout_primitive, split_fanout_primitive};
use crate::transforms::swap_operands::SwapOperandsTransform;
use crate::transforms::swap_outputs::SwapOutputBitsTransform;
use crate::transforms::toggle_operand_negation::ToggleOperandNegationTransform;
use crate::transforms::toggle_output::ToggleOutputBitTransform;
use crate::transforms::transform_trait::{Transform, TransformDirection, TransformLocation};
use crate::transforms::true_and::{insert_true_and_primitive, remove_true_and_primitive};

fn add_input(gb: &mut GateBuilder, name: &str, pir_node_id: u32) -> crate::aig::gate::AigOperand {
    gb.set_current_pir_node_id(Some(pir_node_id));
    let input = *gb.add_input(name.to_string(), 1).get_lsb(0);
    gb.set_current_pir_node_id(None);
    input
}

fn add_and(
    gb: &mut GateBuilder,
    lhs: crate::aig::gate::AigOperand,
    rhs: crate::aig::gate::AigOperand,
    pir_node_id: u32,
) -> crate::aig::gate::AigOperand {
    gb.set_current_pir_node_id(Some(pir_node_id));
    let and_op = gb.add_and_binary(lhs, rhs);
    gb.set_current_pir_node_id(None);
    and_op
}

fn assert_sorted_deduped(ids: &[u32]) {
    assert!(
        ids.windows(2).all(|pair| pair[0] < pair[1]),
        "provenance ids should be sorted/deduped: {:?}",
        ids
    );
}

fn assert_node_ids(g: &GateFn, node_ref: AigRef, expected: &[u32]) {
    let ids = g.gates[node_ref.id].get_pir_node_ids();
    assert_eq!(
        ids, expected,
        "unexpected provenance for node {:?}",
        node_ref
    );
    assert_sorted_deduped(ids);
}

#[test]
fn local_rewrite_transforms_preserve_existing_node_provenance() {
    let mut gb = GateBuilder::new("local_provenance".to_string(), GateBuilderOptions::no_opt());
    let i0 = add_input(&mut gb, "i0", 1);
    let i1 = add_input(&mut gb, "i1", 2);
    let inner = add_and(&mut gb, i0, i1, 10);
    let i2 = add_input(&mut gb, "i2", 3);
    let root = add_and(&mut gb, inner, i2, 11);
    gb.add_output("o".to_string(), root.into());
    gb.add_output("o_inner".to_string(), inner.into());

    let original = gb.build();

    let mut swapped = original.clone();
    SwapOperandsTransform::new()
        .apply(
            &mut swapped,
            &TransformLocation::Node(root.node),
            TransformDirection::Forward,
        )
        .unwrap();
    assert_node_ids(&swapped, root.node, &[11]);
    assert_node_ids(&swapped, inner.node, &[10]);

    let mut toggled = original.clone();
    ToggleOperandNegationTransform::new()
        .apply(
            &mut toggled,
            &TransformLocation::Operand(root.node, false),
            TransformDirection::Forward,
        )
        .unwrap();
    assert_node_ids(&toggled, root.node, &[11]);
    assert_node_ids(&toggled, inner.node, &[10]);

    let mut dbl = original.clone();
    DoubleNegateTransform::new()
        .apply(
            &mut dbl,
            &TransformLocation::Operand(root.node, false),
            TransformDirection::Forward,
        )
        .unwrap();
    assert_node_ids(&dbl, root.node, &[11]);
    assert_node_ids(&dbl, inner.node, &[10]);

    let mut out = original.clone();
    ToggleOutputBitTransform::new()
        .apply(
            &mut out,
            &TransformLocation::OutputPortBit {
                output_idx: 0,
                bit_idx: 0,
            },
            TransformDirection::Forward,
        )
        .unwrap();
    assert_node_ids(&out, root.node, &[11]);
    assert_node_ids(&out, inner.node, &[10]);

    let mut pushed = original.clone();
    PushNegationTransform::new()
        .apply(
            &mut pushed,
            &TransformLocation::Node(root.node),
            TransformDirection::Forward,
        )
        .unwrap();
    assert_node_ids(&pushed, root.node, &[11]);
    assert_node_ids(&pushed, inner.node, &[10]);

    let mut swapped_outputs = original.clone();
    SwapOutputBitsTransform::new()
        .apply(
            &mut swapped_outputs,
            &TransformLocation::SwapOutputBits {
                out_a_idx: 0,
                bit_a_idx: 0,
                out_b_idx: 1,
                bit_b_idx: 0,
            },
            TransformDirection::Forward,
        )
        .unwrap();
    assert_node_ids(&swapped_outputs, root.node, &[11]);
    assert_node_ids(&swapped_outputs, inner.node, &[10]);
}

#[test]
fn unduplicate_unions_killed_gate_provenance_into_survivor() {
    let mut gb = GateBuilder::new("undup".to_string(), GateBuilderOptions::no_opt());
    let i0 = add_input(&mut gb, "i0", 1);
    let i1 = add_input(&mut gb, "i1", 2);
    let first = add_and(&mut gb, i0, i1, 10);
    let second = add_and(&mut gb, i0, i1, 11);
    let top = add_and(&mut gb, first, second, 12);
    gb.add_output("o".to_string(), top.into());
    let mut g = gb.build();

    UnduplicateGateTransform::new()
        .apply(
            &mut g,
            &TransformLocation::Node(second.node),
            TransformDirection::Forward,
        )
        .unwrap();
    assert_node_ids(&g, first.node, &[10, 11]);
}

#[test]
fn merge_fanout_unions_duplicate_provenance_into_original() {
    let mut gb = GateBuilder::new("merge_fanout".to_string(), GateBuilderOptions::no_opt());
    let i0 = add_input(&mut gb, "i0", 1);
    let i1 = add_input(&mut gb, "i1", 2);
    let i2 = add_input(&mut gb, "i2", 3);
    let mid = add_and(&mut gb, i0, i1, 10);
    let use_a = add_and(&mut gb, mid, i2, 11);
    let use_b = add_and(&mut gb, mid, i2, 12);
    gb.add_output("o0".to_string(), use_a.into());
    gb.add_output("o1".to_string(), use_b.into());
    let mut g = gb.build();

    split_fanout_primitive(&mut g, mid.node, use_a.node).unwrap();
    let duplicate_ref = AigRef {
        id: g.gates.len() - 1,
    };
    g.gates[duplicate_ref.id].add_pir_node_id(77);
    merge_fanout_primitive(&mut g, mid.node, duplicate_ref).unwrap();

    assert_node_ids(&g, mid.node, &[10, 77]);
}

#[test]
fn wrapper_insert_remove_transforms_copy_and_union_provenance() {
    let mut gb = GateBuilder::new("true_and".to_string(), GateBuilderOptions::no_opt());
    let i0 = add_input(&mut gb, "i0", 1);
    gb.add_output("o".to_string(), i0.into());
    let mut g = gb.build();
    let true_ref = insert_true_and_primitive(&mut g, i0);
    assert_node_ids(&g, true_ref, &[1]);
    g.gates[true_ref.id].add_pir_node_id(50);
    remove_true_and_primitive(&mut g, true_ref).unwrap();
    assert_node_ids(&g, i0.node, &[1, 50]);

    let mut gb = GateBuilder::new("false_and".to_string(), GateBuilderOptions::no_opt());
    let i0 = add_input(&mut gb, "i0", 1);
    gb.add_output("o".to_string(), i0.into());
    let mut g = gb.build();
    let false_ref = insert_false_and_primitive(&mut g, i0.node).unwrap();
    assert_node_ids(&g, false_ref, &[1]);
    g.gates[false_ref.id].add_pir_node_id(51);
    remove_false_and_primitive(&mut g, false_ref).unwrap();
    assert_node_ids(&g, AigRef { id: 0 }, &[1, 51]);

    let mut gb = GateBuilder::new("redundant_and".to_string(), GateBuilderOptions::no_opt());
    let i0 = add_input(&mut gb, "i0", 1);
    gb.add_output("o".to_string(), i0.into());
    let mut g = gb.build();
    let redundant_ref = insert_redundant_and_primitive(&mut g, i0);
    assert_node_ids(&g, redundant_ref, &[1]);
    g.gates[redundant_ref.id].add_pir_node_id(52);
    remove_redundant_and_primitive(&mut g, redundant_ref).unwrap();
    assert_node_ids(&g, i0.node, &[1, 52]);
}

#[test]
fn and_absorb_unions_inner_provenance_into_outer() {
    let mut gb = GateBuilder::new("absorb".to_string(), GateBuilderOptions::no_opt());
    let a = add_input(&mut gb, "a", 1);
    let b = add_input(&mut gb, "b", 2);
    let inner = add_and(&mut gb, a, b, 10);
    let outer = add_and(&mut gb, inner, a, 11);
    gb.add_output("o".to_string(), outer.into());
    let mut g = gb.build();

    and_absorb_right_primitive(&mut g, outer.node).unwrap();
    assert_node_ids(&g, outer.node, &[10, 11]);
}

#[test]
fn merge_equiv_leaves_and_rewire_union_provenance() {
    let mut gb = GateBuilder::new("merge_leaves".to_string(), GateBuilderOptions::no_opt());
    let i0 = add_input(&mut gb, "i0", 1);
    let i1 = add_input(&mut gb, "i1", 2);
    let i2 = add_input(&mut gb, "i2", 3);
    let left = add_and(&mut gb, i0, i1, 10);
    let right = add_and(&mut gb, i0, i1, 11);
    let top = add_and(&mut gb, left, right, 12);
    gb.add_output("o".to_string(), top.into());
    let mut g = gb.build();

    MergeEquivLeavesTransform::new()
        .apply(
            &mut g,
            &TransformLocation::OperandTarget {
                parent: top.node,
                is_rhs: true,
                old_op: right,
            },
            TransformDirection::Forward,
        )
        .unwrap();
    assert_node_ids(&g, left.node, &[10, 11]);

    rewire_operand_primitive(&mut g, &top.node, true, &i2).unwrap();
    let top_ids = g.gates[top.node.id].get_pir_node_ids();
    assert!(top_ids.contains(&3) && top_ids.contains(&12));
    assert_sorted_deduped(top_ids);
}

#[test]
fn rotate_right_unions_provenance_across_touched_nodes() {
    let mut gb = GateBuilder::new("rotate".to_string(), GateBuilderOptions::no_opt());
    let a = add_input(&mut gb, "a", 1);
    let b = add_input(&mut gb, "b", 2);
    let c = add_input(&mut gb, "c", 3);
    let inner = add_and(&mut gb, a, b, 10);
    let outer = add_and(&mut gb, inner, c, 11);
    gb.add_output("o".to_string(), outer.into());
    let mut g = gb.build();

    rotate_and_right_primitive(&mut g, outer.node).unwrap();

    let inner_ids = g.gates[inner.node.id].get_pir_node_ids();
    let outer_ids = g.gates[outer.node.id].get_pir_node_ids();
    assert!(inner_ids.contains(&10) && inner_ids.contains(&11));
    assert!(outer_ids.contains(&10) && outer_ids.contains(&11));
    assert_sorted_deduped(inner_ids);
    assert_sorted_deduped(outer_ids);
}

#[test]
fn factor_and_unfactor_keep_non_empty_sorted_provenance() {
    let mut gb = GateBuilder::new("factor".to_string(), GateBuilderOptions::no_opt());
    let a = add_input(&mut gb, "a", 1);
    let b = add_input(&mut gb, "b", 2);
    let c = add_input(&mut gb, "c", 3);
    let left = add_and(&mut gb, a, b, 10);
    let right = add_and(&mut gb, a, c, 11);
    let outer = add_and(&mut gb, left, right, 12);
    gb.add_output("o".to_string(), outer.into());
    let mut g = gb.build();

    let factor = FactorSharedAndTransform::new();
    factor
        .apply(
            &mut g,
            &TransformLocation::Node(outer.node),
            TransformDirection::Forward,
        )
        .unwrap();
    assert!(!g.gates[left.node.id].get_pir_node_ids().is_empty());
    assert!(g.gates[left.node.id].get_pir_node_ids().contains(&10));
    assert!(g.gates[outer.node.id].get_pir_node_ids().contains(&12));
    assert_sorted_deduped(g.gates[left.node.id].get_pir_node_ids());
    assert_sorted_deduped(g.gates[outer.node.id].get_pir_node_ids());

    let before_new_gate_count = g.gates.len();
    let mut unfactor = UnfactorSharedAndTransform::new();
    let candidate = unfactor
        .find_candidates(&g, TransformDirection::Forward)
        .into_iter()
        .next()
        .expect("unfactor candidate should exist after factor");
    unfactor
        .apply(&mut g, &candidate, TransformDirection::Forward)
        .unwrap();
    let new_gate_ref = AigRef {
        id: before_new_gate_count,
    };
    assert!(!g.gates[new_gate_ref.id].get_pir_node_ids().is_empty());
    assert_sorted_deduped(g.gates[new_gate_ref.id].get_pir_node_ids());
}

#[test]
fn balance_and_unbalance_preserve_chain_provenance() {
    let mut gb = GateBuilder::new("balance".to_string(), GateBuilderOptions::no_opt());
    let i0 = add_input(&mut gb, "i0", 1);
    let i1 = add_input(&mut gb, "i1", 2);
    let i2 = add_input(&mut gb, "i2", 3);
    let i3 = add_input(&mut gb, "i3", 4);
    let n1 = add_and(&mut gb, i0, i1, 10);
    let n2 = add_and(&mut gb, n1, i2, 11);
    let n3 = add_and(&mut gb, n2, i3, 12);
    gb.add_output("o".to_string(), n3.into());
    let mut g = gb.build();

    BalanceAndTreeTransform::new()
        .apply(
            &mut g,
            &TransformLocation::Node(n3.node),
            TransformDirection::Forward,
        )
        .unwrap();
    for node_ref in [n1.node, n2.node, n3.node] {
        assert!(!g.gates[node_ref.id].get_pir_node_ids().is_empty());
        assert_sorted_deduped(g.gates[node_ref.id].get_pir_node_ids());
    }

    UnbalanceAndTreeTransform::new()
        .apply(
            &mut g,
            &TransformLocation::Node(n3.node),
            TransformDirection::Forward,
        )
        .unwrap();
    let root_ids = g.gates[n3.node.id].get_pir_node_ids();
    assert!(root_ids.contains(&12));
    assert_sorted_deduped(root_ids);
}

#[test]
fn only_literal_constant_false_carries_literal_variant_in_helper_graphs() {
    let mut gb = GateBuilder::new("literal_shape".to_string(), GateBuilderOptions::no_opt());
    let i0 = add_input(&mut gb, "i0", 1);
    gb.add_output("o".to_string(), i0.into());
    let g = gb.build();
    assert!(matches!(g.gates[0], AigNode::Literal { value: false, .. }));
}
