// SPDX-License-Identifier: Apache-2.0

//! Reassociates single-fanout AND supergates into shallower balanced trees.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

use crate::aig::dce::dce;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn, PirNodeIds};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::use_count::get_id_to_use_count;

#[derive(Debug)]
struct PlainAndSupergate {
    leaf_ops: Vec<AigOperand>,
    pir_node_ids: PirNodeIds,
    root_tags: Vec<String>,
}

/// Rebuilds live single-fanout AND supergates as depth-balanced trees.
///
/// The pass only flattens through non-negated AND edges. Complemented leaves
/// stay as ordinary operands, so De Morgan-encoded OR cones use the same
/// machinery without a separate OR representation.
pub fn reassociate_gatefn(orig_fn: &GateFn) -> GateFn {
    let use_counts = get_id_to_use_count(orig_fn);
    let mut builder = GateBuilder::new(
        orig_fn.name.clone(),
        GateBuilderOptions {
            fold: false,
            hash: true,
        },
    );
    let mut orig_to_new: Vec<Option<AigOperand>> = vec![None; orig_fn.gates.len()];
    let mut new_depths = vec![0usize];

    map_inputs(orig_fn, &mut builder, &mut orig_to_new);
    sync_depths(&builder, &mut new_depths);

    for orig_ref in orig_fn.post_order_refs() {
        if orig_to_new[orig_ref.id].is_some() {
            continue;
        }
        let orig_node = &orig_fn.gates[orig_ref.id];
        let new_op = match orig_node {
            AigNode::Input { .. } => unreachable!("inputs are mapped before post-order rebuild"),
            AigNode::Literal { value, .. } => {
                let op = if *value {
                    builder.get_true()
                } else {
                    builder.get_false()
                };
                builder.add_pir_node_ids(op.node, orig_node.get_pir_node_ids());
                op
            }
            AigNode::And2 { .. } => {
                let supergate = collect_plain_and_supergate(orig_fn, orig_ref, &use_counts);
                rebuild_plain_and_supergate(&mut builder, &mut new_depths, &orig_to_new, supergate)
            }
        };
        builder.add_pir_node_ids(new_op.node, orig_node.get_pir_node_ids());
        orig_to_new[orig_ref.id] = Some(new_op);
    }

    for orig_output in &orig_fn.outputs {
        let mut new_output_bits = Vec::with_capacity(orig_output.get_bit_count());
        for orig_bit in orig_output.bit_vector.iter_lsb_to_msb() {
            new_output_bits.push(remap_operand(&orig_to_new, *orig_bit));
        }
        builder.add_output(
            orig_output.name.clone(),
            AigBitVector::from_lsb_is_index_0(&new_output_bits),
        );
    }

    let rebuilt = builder.build();
    rebuilt.check_invariants_with_debug_assert();
    let result = dce(&rebuilt);
    result.check_invariants_with_debug_assert();
    result
}

fn map_inputs(orig_fn: &GateFn, builder: &mut GateBuilder, orig_to_new: &mut [Option<AigOperand>]) {
    for orig_input in &orig_fn.inputs {
        let new_input = builder.add_input(orig_input.name.clone(), orig_input.get_bit_count());
        for bit_index in 0..orig_input.get_bit_count() {
            let orig_bit = *orig_input.bit_vector.get_lsb(bit_index);
            let new_bit = *new_input.get_lsb(bit_index);
            builder.add_pir_node_ids(
                new_bit.node,
                orig_fn.gates[orig_bit.node.id].get_pir_node_ids(),
            );
            orig_to_new[orig_bit.node.id] = Some(new_bit);
        }
    }
}

fn collect_plain_and_supergate(
    g: &GateFn,
    root: AigRef,
    use_counts: &HashMap<AigRef, usize>,
) -> PlainAndSupergate {
    let mut leaf_ops = Vec::new();
    let mut pir_node_ids = PirNodeIds::new();
    let mut worklist = vec![root];

    while let Some(node_ref) = worklist.pop() {
        extend_pir_node_ids(&mut pir_node_ids, g.gates[node_ref.id].get_pir_node_ids());
        let AigNode::And2 { a, b, .. } = g.gates[node_ref.id] else {
            unreachable!("plain AND supergate roots are And2 nodes");
        };
        for operand in [b, a] {
            if can_flatten_plain_and_operand(g, operand, use_counts) {
                worklist.push(operand.node);
            } else {
                extend_pir_node_ids(
                    &mut pir_node_ids,
                    g.gates[operand.node.id].get_pir_node_ids(),
                );
                leaf_ops.push(operand);
            }
        }
    }

    PlainAndSupergate {
        leaf_ops,
        pir_node_ids,
        root_tags: g.gates[root.id]
            .get_tags()
            .map_or_else(Vec::new, |tags| tags.to_vec()),
    }
}

fn can_flatten_plain_and_operand(
    g: &GateFn,
    operand: AigOperand,
    use_counts: &HashMap<AigRef, usize>,
) -> bool {
    !operand.negated
        && matches!(g.gates[operand.node.id], AigNode::And2 { .. })
        && use_counts.get(&operand.node).copied().unwrap_or(0) == 1
}

fn rebuild_plain_and_supergate(
    builder: &mut GateBuilder,
    new_depths: &mut Vec<usize>,
    orig_to_new: &[Option<AigOperand>],
    supergate: PlainAndSupergate,
) -> AigOperand {
    let PlainAndSupergate {
        leaf_ops,
        pir_node_ids,
        root_tags,
    } = supergate;
    let normalized = normalize_leaf_ops(
        builder,
        leaf_ops
            .into_iter()
            .map(|operand| remap_operand(orig_to_new, operand)),
    );
    let output = match normalized {
        NormalizedLeaves::Constant(op) | NormalizedLeaves::Single(op) => op,
        NormalizedLeaves::Many(leaf_ops) => {
            build_balanced_and_tree(builder, new_depths, &leaf_ops, pir_node_ids.as_slice())
        }
    };
    builder.add_pir_node_ids(output.node, pir_node_ids.as_slice());
    add_tags(builder, output.node, &root_tags);
    output
}

enum NormalizedLeaves {
    Constant(AigOperand),
    Single(AigOperand),
    Many(Vec<AigOperand>),
}

fn normalize_leaf_ops(
    builder: &GateBuilder,
    leaf_ops: impl IntoIterator<Item = AigOperand>,
) -> NormalizedLeaves {
    let mut normalized = BTreeSet::new();
    for operand in leaf_ops {
        if builder.is_known_false(operand) {
            return NormalizedLeaves::Constant(builder.get_false());
        }
        if builder.is_known_true(operand) {
            continue;
        }
        if normalized.contains(&operand.negate()) {
            return NormalizedLeaves::Constant(builder.get_false());
        }
        normalized.insert(operand);
    }

    let mut normalized = normalized.into_iter().collect::<Vec<_>>();
    match normalized.len() {
        0 => NormalizedLeaves::Constant(builder.get_true()),
        1 => NormalizedLeaves::Single(normalized.pop().unwrap()),
        _ => NormalizedLeaves::Many(normalized),
    }
}

fn build_balanced_and_tree(
    builder: &mut GateBuilder,
    new_depths: &mut Vec<usize>,
    leaf_ops: &[AigOperand],
    pir_node_ids: &[u32],
) -> AigOperand {
    let mut heap = BinaryHeap::new();
    for operand in leaf_ops.iter().copied() {
        heap.push(Reverse((operand_depth(new_depths, operand), operand)));
    }

    while heap.len() > 1 {
        let Reverse((_, lhs)) = heap.pop().unwrap();
        let Reverse((_, rhs)) = heap.pop().unwrap();
        let output = builder.add_and_binary(lhs, rhs);
        builder.add_pir_node_ids(output.node, pir_node_ids);
        sync_depths(builder, new_depths);
        heap.push(Reverse((operand_depth(new_depths, output), output)));
    }

    heap.pop().unwrap().0.1
}

fn remap_operand(orig_to_new: &[Option<AigOperand>], operand: AigOperand) -> AigOperand {
    let mapped = orig_to_new[operand.node.id]
        .unwrap_or_else(|| panic!("missing rebuilt operand for {:?}", operand.node));
    if operand.negated {
        mapped.negate()
    } else {
        mapped
    }
}

fn sync_depths(builder: &GateBuilder, new_depths: &mut Vec<usize>) {
    while new_depths.len() < builder.gates.len() {
        let depth = match &builder.gates[new_depths.len()] {
            AigNode::Input { .. } | AigNode::Literal { .. } => 0,
            AigNode::And2 { a, b, .. } => {
                1 + operand_depth(new_depths, *a).max(operand_depth(new_depths, *b))
            }
        };
        new_depths.push(depth);
    }
}

fn operand_depth(new_depths: &[usize], operand: AigOperand) -> usize {
    new_depths[operand.node.id]
}

fn extend_pir_node_ids(dst: &mut PirNodeIds, src: &[u32]) {
    for pir_node_id in src {
        match dst.binary_search(pir_node_id) {
            Ok(_) => {}
            Err(index) => dst.insert(index, *pir_node_id),
        }
    }
}

fn add_tags(builder: &mut GateBuilder, aig_ref: AigRef, tags: &[String]) {
    for tag in tags {
        builder.add_tag(aig_ref, tag.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::get_summary_stats::get_aig_stats;
    use crate::mcmc_logic::oracle_equiv_sat;

    fn build_linear_and4() -> GateFn {
        let mut builder = GateBuilder::new("and4".to_string(), GateBuilderOptions::opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
        let d = *builder.add_input("d".to_string(), 1).get_lsb(0);
        let ab = builder.add_and_binary(a, b);
        let abc = builder.add_and_binary(ab, c);
        let abcd = builder.add_and_binary(abc, d);
        builder.add_output("o".to_string(), abcd.into());
        builder.build()
    }

    #[test]
    fn reassociate_balances_linear_and4() {
        let before = build_linear_and4();
        let after = reassociate_gatefn(&before);

        assert!(oracle_equiv_sat(&before, &after).unwrap());
        assert_eq!(get_aig_stats(&before).and_nodes, 3);
        assert_eq!(get_aig_stats(&after).and_nodes, 3);
        assert_eq!(get_aig_stats(&before).max_depth, 3);
        assert_eq!(get_aig_stats(&after).max_depth, 2);
    }

    #[test]
    fn reassociate_balances_demorgan_or4() {
        let mut builder = GateBuilder::new("or4".to_string(), GateBuilderOptions::opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
        let d = *builder.add_input("d".to_string(), 1).get_lsb(0);
        let ab = builder.add_or_binary(a, b);
        let abc = builder.add_or_binary(ab, c);
        let abcd = builder.add_or_binary(abc, d);
        builder.add_output("o".to_string(), abcd.into());
        let before = builder.build();

        let after = reassociate_gatefn(&before);

        assert!(oracle_equiv_sat(&before, &after).unwrap());
        assert_eq!(get_aig_stats(&before).and_nodes, 3);
        assert_eq!(get_aig_stats(&after).and_nodes, 3);
        assert_eq!(get_aig_stats(&before).max_depth, 3);
        assert_eq!(get_aig_stats(&after).max_depth, 2);
    }

    #[test]
    fn reassociate_dedupes_repeated_leaf() {
        let mut builder = GateBuilder::new("repeat".to_string(), GateBuilderOptions::no_opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let ab = builder.add_and_binary(a, b);
        let aba = builder.add_and_binary(ab, a);
        builder.add_output("o".to_string(), aba.into());
        let before = builder.build();

        let after = reassociate_gatefn(&before);

        assert!(oracle_equiv_sat(&before, &after).unwrap());
        assert_eq!(get_aig_stats(&before).and_nodes, 2);
        assert_eq!(get_aig_stats(&after).and_nodes, 1);
    }

    #[test]
    fn reassociate_preserves_root_tags() {
        let mut builder = GateBuilder::new("tags".to_string(), GateBuilderOptions::opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
        let ab = builder.add_and_binary(a, b);
        let abc = builder.add_and_binary(ab, c);
        builder.add_tag(abc.node, "root_tag".to_string());
        builder.add_output("o".to_string(), abc.into());
        let before = builder.build();

        let after = reassociate_gatefn(&before);

        assert!(oracle_equiv_sat(&before, &after).unwrap());
        let output = *after.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(
            after.gates[output.node.id].get_tags(),
            Some(&["root_tag".to_string()][..])
        );
    }

    #[test]
    fn reassociate_collapses_complementary_leaves_to_false() {
        let mut builder = GateBuilder::new("complement".to_string(), GateBuilderOptions::no_opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let ab = builder.add_and_binary(a, b);
        let contradiction = builder.add_and_binary(ab, a.negate());
        builder.add_output("o".to_string(), contradiction.into());
        let before = builder.build();

        let after = reassociate_gatefn(&before);

        assert!(oracle_equiv_sat(&before, &after).unwrap());
        assert_eq!(get_aig_stats(&after).and_nodes, 0);
        let output = *after.outputs[0].bit_vector.get_lsb(0);
        assert_eq!(output.node.id, 0);
        assert!(!output.negated);
    }
}
