// SPDX-License-Identifier: Apache-2.0

//! Reassociates single-fanout AND supergates into shallower balanced trees.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::time::Instant;

use crate::aig::dce::dce;
use crate::aig::gate::{
    AigBitVector, AigNode, AigOperand, AigRef, GateFn, PirNodeIds, PirNodeIdsInterner,
};
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
    let started = Instant::now();
    let use_counts = get_id_to_use_count(orig_fn);
    let post_order_refs = orig_fn.post_order_refs();
    let absorbed_nodes = find_absorbed_plain_and_nodes(orig_fn, &post_order_refs, &use_counts);
    let maximal_supergate_count = post_order_refs
        .iter()
        .filter(|node_ref| {
            matches!(orig_fn.gates[node_ref.id], AigNode::And2 { .. })
                && !absorbed_nodes[node_ref.id]
        })
        .count();
    let absorbed_node_count = absorbed_nodes.iter().filter(|absorbed| **absorbed).count();
    log::info!(
        "reassociation rebuilding {} maximal supergates and absorbing {} internal AND nodes",
        maximal_supergate_count,
        absorbed_node_count
    );
    let mut provenance_interner = PirNodeIdsInterner::default();
    let mut builder = GateBuilder::new(
        orig_fn.name.clone(),
        GateBuilderOptions {
            fold: false,
            hash: true,
        },
    );
    let mut orig_to_new: Vec<Option<AigOperand>> = vec![None; orig_fn.gates.len()];
    let mut new_depths = vec![0usize];

    map_inputs(
        orig_fn,
        &mut builder,
        &mut orig_to_new,
        &mut provenance_interner,
    );
    sync_depths(&builder, &mut new_depths);

    for orig_ref in post_order_refs {
        if orig_to_new[orig_ref.id].is_some() {
            continue;
        }
        if absorbed_nodes[orig_ref.id] {
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
                builder.add_interned_pir_node_ids(
                    op.node,
                    orig_node.get_pir_node_ids(),
                    &mut provenance_interner,
                );
                op
            }
            AigNode::And2 { .. } => {
                let supergate = collect_plain_and_supergate(
                    orig_fn,
                    orig_ref,
                    &use_counts,
                    &mut provenance_interner,
                );
                rebuild_plain_and_supergate(
                    &mut builder,
                    &mut new_depths,
                    &orig_to_new,
                    supergate,
                    &mut provenance_interner,
                )
            }
        };
        builder.add_interned_pir_node_ids(
            new_op.node,
            orig_node.get_pir_node_ids(),
            &mut provenance_interner,
        );
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
    log::info!(
        "reassociation complete: input_nodes={}, output_nodes={}, maximal_supergates={}, absorbed_internal_ands={}, seconds={:.6}",
        orig_fn.gates.len(),
        result.gates.len(),
        maximal_supergate_count,
        absorbed_node_count,
        started.elapsed().as_secs_f64()
    );
    result
}

/// Marks internal AND nodes that belong to a larger maximal supergate.
fn find_absorbed_plain_and_nodes(
    g: &GateFn,
    post_order_refs: &[AigRef],
    use_counts: &HashMap<AigRef, usize>,
) -> Vec<bool> {
    let mut absorbed = vec![false; g.gates.len()];
    for node_ref in post_order_refs {
        let AigNode::And2 { a, b, .. } = g.gates[node_ref.id] else {
            continue;
        };
        for operand in [a, b] {
            if can_flatten_plain_and_operand(g, operand, use_counts) {
                absorbed[operand.node.id] = true;
            }
        }
    }
    absorbed
}

fn map_inputs(
    orig_fn: &GateFn,
    builder: &mut GateBuilder,
    orig_to_new: &mut [Option<AigOperand>],
    provenance_interner: &mut PirNodeIdsInterner,
) {
    for orig_input in &orig_fn.inputs {
        let new_input = builder.add_input(orig_input.name.clone(), orig_input.get_bit_count());
        for bit_index in 0..orig_input.get_bit_count() {
            let orig_bit = *orig_input.bit_vector.get_lsb(bit_index);
            let new_bit = *new_input.get_lsb(bit_index);
            builder.add_interned_pir_node_ids(
                new_bit.node,
                orig_fn.gates[orig_bit.node.id].get_pir_node_ids(),
                provenance_interner,
            );
            orig_to_new[orig_bit.node.id] = Some(new_bit);
        }
    }
}

fn collect_plain_and_supergate(
    g: &GateFn,
    root: AigRef,
    use_counts: &HashMap<AigRef, usize>,
    provenance_interner: &mut PirNodeIdsInterner,
) -> PlainAndSupergate {
    let mut leaf_ops = Vec::new();
    let mut pir_node_ids = Vec::new();
    let mut worklist = vec![root];

    while let Some(node_ref) = worklist.pop() {
        pir_node_ids.extend_from_slice(g.gates[node_ref.id].get_pir_node_ids());
        let AigNode::And2 { a, b, .. } = g.gates[node_ref.id] else {
            unreachable!("plain AND supergate roots are And2 nodes");
        };
        for operand in [b, a] {
            if can_flatten_plain_and_operand(g, operand, use_counts) {
                worklist.push(operand.node);
            } else {
                pir_node_ids.extend_from_slice(g.gates[operand.node.id].get_pir_node_ids());
                leaf_ops.push(operand);
            }
        }
    }
    pir_node_ids.sort_unstable();
    pir_node_ids.dedup();

    PlainAndSupergate {
        leaf_ops,
        pir_node_ids: provenance_interner.intern_slice(&pir_node_ids),
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
    provenance_interner: &mut PirNodeIdsInterner,
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
        NormalizedLeaves::Many(leaf_ops) => build_balanced_and_tree(
            builder,
            new_depths,
            &leaf_ops,
            &pir_node_ids,
            provenance_interner,
        ),
    };
    builder.add_interned_pir_node_id_set(output.node, &pir_node_ids, provenance_interner);
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
    pir_node_ids: &PirNodeIds,
    provenance_interner: &mut PirNodeIdsInterner,
) -> AigOperand {
    let mut heap = BinaryHeap::new();
    for operand in leaf_ops.iter().copied() {
        heap.push(Reverse((operand_depth(new_depths, operand), operand)));
    }

    while heap.len() > 1 {
        let Reverse((_, lhs)) = heap.pop().unwrap();
        let Reverse((_, rhs)) = heap.pop().unwrap();
        let output = builder.add_and_binary(lhs, rhs);
        builder.add_interned_pir_node_id_set(output.node, pir_node_ids, provenance_interner);
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
    fn identifies_only_maximal_plain_and_supergate_roots() {
        let mut builder = GateBuilder::new("maximal_and".to_string(), GateBuilderOptions::opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
        let d = *builder.add_input("d".to_string(), 1).get_lsb(0);
        let ab = builder.add_and_binary(a, b);
        let abc = builder.add_and_binary(ab, c);
        let abcd = builder.add_and_binary(abc, d);
        builder.add_output("o".to_string(), abcd.into());
        let gate_fn = builder.build();

        let post_order_refs = gate_fn.post_order_refs();
        let use_counts = get_id_to_use_count(&gate_fn);
        let absorbed = find_absorbed_plain_and_nodes(&gate_fn, &post_order_refs, &use_counts);

        assert!(absorbed[ab.node.id]);
        assert!(absorbed[abc.node.id]);
        assert!(!absorbed[abcd.node.id]);
    }

    #[test]
    fn shared_fanout_and_is_a_supergate_boundary() {
        let mut builder = GateBuilder::new("shared_and".to_string(), GateBuilderOptions::opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
        let d = *builder.add_input("d".to_string(), 1).get_lsb(0);
        let ab = builder.add_and_binary(a, b);
        let abc = builder.add_and_binary(ab, c);
        let abd = builder.add_and_binary(ab, d);
        let root = builder.add_and_binary(abc, abd);
        builder.add_output("o".to_string(), root.into());
        let before = builder.build();

        let post_order_refs = before.post_order_refs();
        let use_counts = get_id_to_use_count(&before);
        let absorbed = find_absorbed_plain_and_nodes(&before, &post_order_refs, &use_counts);
        assert!(!absorbed[ab.node.id]);
        assert!(absorbed[abc.node.id]);
        assert!(absorbed[abd.node.id]);
        assert!(!absorbed[root.node.id]);

        let after = reassociate_gatefn(&before);
        assert!(oracle_equiv_sat(&before, &after).unwrap());
    }

    #[test]
    fn reassociate_interns_large_supergate_provenance_sets() {
        let mut builder =
            GateBuilder::new("and4_provenance".to_string(), GateBuilderOptions::opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
        let d = *builder.add_input("d".to_string(), 1).get_lsb(0);
        for (operand, pir_node_id) in [(a, 1), (b, 2), (c, 3), (d, 4)] {
            builder.add_pir_node_id(operand.node, pir_node_id);
        }
        let ab = builder.add_and_binary(a, b);
        let abc = builder.add_and_binary(ab, c);
        let abcd = builder.add_and_binary(abc, d);
        builder.add_output("o".to_string(), abcd.into());

        let after = reassociate_gatefn(&builder.build());
        let and_nodes = after
            .gates
            .iter()
            .filter(|node| matches!(node, AigNode::And2 { .. }))
            .collect::<Vec<_>>();
        assert_eq!(and_nodes.len(), 3);
        assert!(
            and_nodes
                .windows(2)
                .all(|pair| pair[0].shares_pir_node_id_storage_with(pair[1]))
        );
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
