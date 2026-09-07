// SPDX-License-Identifier: Apache-2.0

//! Reassociates single-fanout AND supergates into shallower balanced trees.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::time::Instant;

use crate::aig::dce::dce;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn, PirNodeIds};
use crate::aig::get_summary_stats::{AigConeStats, get_aig_cone_stats};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::use_count::get_id_to_use_count;

#[derive(Debug)]
struct PlainAndSupergate {
    leaf_ops: Vec<AigOperand>,
    pir_node_ids: PirNodeIds,
    root_tags: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AndTreePairingPolicy {
    EarliestArrival,
    PreferReuse,
}

#[derive(Debug)]
struct ReassociationResult {
    gate_fn: GateFn,
    used_alternate_reuse_pair: bool,
}

#[derive(Debug)]
struct FinalReassociationResult {
    gate_fn: GateFn,
    comparison_performed: bool,
    reuse_candidate_selected: bool,
}

/// Rebuilds live single-fanout AND supergates as depth-balanced trees.
///
/// The pass only flattens through non-negated AND edges. Complemented leaves
/// stay as ordinary operands, so De Morgan-encoded OR cones use the same
/// machinery without a separate OR representation.
pub fn reassociate_gatefn(orig_fn: &GateFn) -> GateFn {
    reassociate_gatefn_with_policy(orig_fn, AndTreePairingPolicy::EarliestArrival).gate_fn
}

/// Reassociates with reuse when it is an exact final-AIG Pareto improvement.
pub(crate) fn reassociate_gatefn_selecting_reuse(orig_fn: &GateFn) -> GateFn {
    let result = reassociate_gatefn_selecting_reuse_with_outcome(orig_fn);
    log::debug!(
        "final reassociation reuse outcome: comparison_performed={}, candidate_selected={}",
        result.comparison_performed,
        result.reuse_candidate_selected
    );
    result.gate_fn
}

fn reassociate_gatefn_selecting_reuse_with_outcome(orig_fn: &GateFn) -> FinalReassociationResult {
    let candidate = reassociate_gatefn_with_policy(orig_fn, AndTreePairingPolicy::PreferReuse);
    if !candidate.used_alternate_reuse_pair {
        return FinalReassociationResult {
            gate_fn: candidate.gate_fn,
            comparison_performed: false,
            reuse_candidate_selected: false,
        };
    }

    let baseline =
        reassociate_gatefn_with_policy(orig_fn, AndTreePairingPolicy::EarliestArrival).gate_fn;
    let baseline_stats = get_output_cone_stats(&baseline);
    let candidate_stats = get_output_cone_stats(&candidate.gate_fn);
    let accept_candidate =
        is_strict_non_regressing_pareto_improvement(&baseline_stats, &candidate_stats);
    log::debug!(
        "final reassociation reuse comparison: baseline_ands={}, candidate_ands={}, baseline_depths={:?}, candidate_depths={:?}, accepted={}",
        baseline_stats.and_nodes,
        candidate_stats.and_nodes,
        baseline_stats.root_depths,
        candidate_stats.root_depths,
        accept_candidate
    );
    FinalReassociationResult {
        gate_fn: if accept_candidate {
            candidate.gate_fn
        } else {
            baseline
        },
        comparison_performed: true,
        reuse_candidate_selected: accept_candidate,
    }
}

fn get_output_cone_stats(gate_fn: &GateFn) -> AigConeStats {
    let roots = gate_fn
        .outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb().copied())
        .collect::<Vec<_>>();
    get_aig_cone_stats(&gate_fn.gates, &roots)
}

fn is_strict_non_regressing_pareto_improvement(
    baseline: &AigConeStats,
    candidate: &AigConeStats,
) -> bool {
    assert_eq!(baseline.root_depths.len(), candidate.root_depths.len());
    let area_no_worse = candidate.and_nodes <= baseline.and_nodes;
    let depths_no_worse = candidate
        .root_depths
        .iter()
        .zip(&baseline.root_depths)
        .all(|(candidate, baseline)| candidate <= baseline);
    let strictly_better = candidate.and_nodes < baseline.and_nodes
        || candidate
            .root_depths
            .iter()
            .zip(&baseline.root_depths)
            .any(|(candidate, baseline)| candidate < baseline);
    area_no_worse && depths_no_worse && strictly_better
}

fn reassociate_gatefn_with_policy(
    orig_fn: &GateFn,
    pairing_policy: AndTreePairingPolicy,
) -> ReassociationResult {
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
    let mut builder = GateBuilder::new(
        orig_fn.name.clone(),
        GateBuilderOptions {
            fold: false,
            hash: true,
        },
    );
    let mut orig_to_new: Vec<Option<AigOperand>> = vec![None; orig_fn.gates.len()];
    let mut new_depths = vec![0usize];
    let mut used_alternate_reuse_pair = false;

    map_inputs(orig_fn, &mut builder, &mut orig_to_new);
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
                builder.add_pir_node_ids(op.node, orig_node.get_pir_node_ids());
                op
            }
            AigNode::And2 { .. } => {
                let supergate = collect_plain_and_supergate(orig_fn, orig_ref, &use_counts);
                rebuild_plain_and_supergate(
                    &mut builder,
                    &mut new_depths,
                    &orig_to_new,
                    supergate,
                    pairing_policy,
                    &mut used_alternate_reuse_pair,
                )
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
    log::info!(
        "reassociation complete: input_nodes={}, output_nodes={}, maximal_supergates={}, absorbed_internal_ands={}, seconds={:.6}",
        orig_fn.gates.len(),
        result.gates.len(),
        maximal_supergate_count,
        absorbed_node_count,
        started.elapsed().as_secs_f64()
    );
    ReassociationResult {
        gate_fn: result,
        used_alternate_reuse_pair,
    }
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
        pir_node_ids: PirNodeIds::from_vec(pir_node_ids),
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
    pairing_policy: AndTreePairingPolicy,
    used_alternate_reuse_pair: &mut bool,
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
            pairing_policy,
            used_alternate_reuse_pair,
        ),
    };
    builder.add_pir_node_ids(output.node, &pir_node_ids);
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

struct AndPairSelection {
    lhs: AigOperand,
    rhs: AigOperand,
    used_alternate_reuse_pair: bool,
}

// Bounds the quadratic search for an existing AND among arrival-equivalent
// choices. Large buckets retain deterministic behavior by considering their
// earliest operands only.
const MAX_REUSE_PAIR_SEARCH_OPERANDS: usize = 64;

fn pop_earliest_arrival_pair(
    heap: &mut BinaryHeap<Reverse<(usize, AigOperand)>>,
) -> AndPairSelection {
    let Reverse((_, lhs)) = heap.pop().expect("AND tree should have a first operand");
    let Reverse((_, rhs)) = heap.pop().expect("AND tree should have a second operand");
    AndPairSelection {
        lhs,
        rhs,
        used_alternate_reuse_pair: false,
    }
}

/// Removes an earliest-arrival pair, preferring an existing AND when tied.
fn pop_earliest_arrival_pair_prefer_reuse(
    builder: &GateBuilder,
    heap: &mut BinaryHeap<Reverse<(usize, AigOperand)>>,
) -> AndPairSelection {
    let Reverse(first) = heap.pop().expect("AND tree should have a first operand");
    let Reverse(second) = heap.pop().expect("AND tree should have a second operand");
    let first_depth = first.0;
    let second_depth = second.0;
    let mut eligible = vec![first, second];

    while eligible.len() < MAX_REUSE_PAIR_SEARCH_OPERANDS {
        let Some(Reverse((next_depth, _))) = heap.peek() else {
            break;
        };
        if *next_depth != second_depth {
            break;
        }
        eligible.push(heap.pop().unwrap().0);
    }

    let mut selected = None;
    for lhs_index in 0..eligible.len() {
        for rhs_index in lhs_index + 1..eligible.len() {
            // If the smallest arrival is unique, consuming it is part of the
            // earliest-arrival schedule. Otherwise any two operands in the
            // minimum-depth bucket are equivalent for depth.
            if first_depth < second_depth && lhs_index != 0 {
                continue;
            }
            let lhs = eligible[lhs_index];
            let rhs = eligible[rhs_index];
            let Some(existing) = builder.find_existing_and(lhs.1, rhs.1) else {
                continue;
            };
            let existing_depth = builder
                .aig_depth(existing)
                .expect("reassociation builder should cache AIG depths");
            let newly_built_depth = lhs.0.max(rhs.0) + 1;
            if existing_depth > newly_built_depth {
                continue;
            }
            let rank = (existing_depth, lhs_index, rhs_index);
            if selected.is_none_or(|(best_rank, _, _)| rank < best_rank) {
                selected = Some((rank, lhs_index, rhs_index));
            }
        }
    }

    let (lhs_index, rhs_index) = selected
        .map(|(_, lhs_index, rhs_index)| (lhs_index, rhs_index))
        .unwrap_or((0, 1));
    let lhs = eligible[lhs_index].1;
    let rhs = eligible[rhs_index].1;
    for (index, item) in eligible.into_iter().enumerate() {
        if index != lhs_index && index != rhs_index {
            heap.push(Reverse(item));
        }
    }
    AndPairSelection {
        lhs,
        rhs,
        used_alternate_reuse_pair: (lhs_index, rhs_index) != (0, 1),
    }
}

fn build_balanced_and_tree(
    builder: &mut GateBuilder,
    new_depths: &mut Vec<usize>,
    leaf_ops: &[AigOperand],
    pir_node_ids: &[u32],
    pairing_policy: AndTreePairingPolicy,
    used_alternate_reuse_pair: &mut bool,
) -> AigOperand {
    let mut heap = BinaryHeap::new();
    for operand in leaf_ops.iter().copied() {
        heap.push(Reverse((operand_depth(new_depths, operand), operand)));
    }

    while heap.len() > 1 {
        let selection = match pairing_policy {
            AndTreePairingPolicy::EarliestArrival => pop_earliest_arrival_pair(&mut heap),
            AndTreePairingPolicy::PreferReuse => {
                pop_earliest_arrival_pair_prefer_reuse(builder, &mut heap)
            }
        };
        *used_alternate_reuse_pair |= selection.used_alternate_reuse_pair;
        let output = builder.add_and_binary(selection.lhs, selection.rhs);
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
    fn reassociate_prefers_reusable_pair_without_depth_regression() {
        let mut builder = GateBuilder::new("reuse_pair".to_string(), GateBuilderOptions::opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
        let d = *builder.add_input("d".to_string(), 1).get_lsb(0);

        let shared = builder.add_and_binary(b, c);
        let ab = builder.add_and_binary(a, b);
        let cd = builder.add_and_binary(c, d);
        let root = builder.add_and_binary(ab, cd);

        // Output traversal runs in reverse output order, so place `shared`
        // last to make it available when the root supergate is rebuilt.
        builder.add_output("root".to_string(), root.into());
        builder.add_output("shared".to_string(), shared.into());
        let before = builder.build();

        let ordinary = reassociate_gatefn(&before);
        let outcome = reassociate_gatefn_selecting_reuse_with_outcome(&before);
        let repeated = reassociate_gatefn_selecting_reuse(&before);
        let after = outcome.gate_fn;

        assert!(oracle_equiv_sat(&before, &after).unwrap());
        assert!(outcome.comparison_performed);
        assert!(outcome.reuse_candidate_selected);
        assert_eq!(get_aig_stats(&before).and_nodes, 4);
        assert_eq!(get_aig_stats(&ordinary).and_nodes, 4);
        assert_eq!(get_aig_stats(&after).and_nodes, 3);
        assert_eq!(get_aig_stats(&before).max_depth, 2);
        assert_eq!(get_aig_stats(&ordinary).max_depth, 2);
        assert_eq!(get_aig_stats(&after).max_depth, 2);
        assert_eq!(after.gates, repeated.gates);
        assert_eq!(after.outputs.len(), repeated.outputs.len());
        for (after_output, repeated_output) in after.outputs.iter().zip(&repeated.outputs) {
            assert_eq!(after_output.name, repeated_output.name);
            assert!(
                after_output
                    .bit_vector
                    .iter_lsb_to_msb()
                    .eq(repeated_output.bit_vector.iter_lsb_to_msb())
            );
        }
    }

    #[test]
    fn reuse_preference_keeps_a_unique_earliest_operand() {
        let mut builder =
            GateBuilder::new("unique_earliest".to_string(), GateBuilderOptions::opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b_bits = builder.add_input("b".to_string(), 2);
        let c_bits = builder.add_input("c".to_string(), 2);
        let b = builder.add_and_binary(*b_bits.get_lsb(0), *b_bits.get_lsb(1));
        let c = builder.add_and_binary(*c_bits.get_lsb(0), *c_bits.get_lsb(1));

        // This reusable pair must not displace the uniquely earliest `a`.
        builder.add_and_binary(b, c);
        let mut heap = BinaryHeap::new();
        for operand in [a, b, c] {
            heap.push(Reverse((
                builder
                    .aig_depth(operand)
                    .expect("test builder should cache AIG depths"),
                operand,
            )));
        }

        let selection = pop_earliest_arrival_pair_prefer_reuse(&builder, &mut heap);
        assert!(selection.lhs == a || selection.rhs == a);
    }

    #[test]
    fn final_reassociation_guard_requires_a_strict_per_output_pareto_win() {
        let baseline = AigConeStats {
            and_nodes: 10,
            root_depths: vec![4, 6],
        };

        assert!(is_strict_non_regressing_pareto_improvement(
            &baseline,
            &AigConeStats {
                and_nodes: 9,
                root_depths: vec![4, 6],
            }
        ));
        assert!(is_strict_non_regressing_pareto_improvement(
            &baseline,
            &AigConeStats {
                and_nodes: 10,
                root_depths: vec![3, 6],
            }
        ));
        assert!(!is_strict_non_regressing_pareto_improvement(
            &baseline,
            &AigConeStats {
                and_nodes: 9,
                root_depths: vec![5, 5],
            }
        ));
        assert!(!is_strict_non_regressing_pareto_improvement(
            &baseline,
            &AigConeStats {
                and_nodes: 11,
                root_depths: vec![3, 5],
            }
        ));
        assert!(!is_strict_non_regressing_pareto_improvement(
            &baseline, &baseline
        ));
    }

    #[test]
    fn final_reassociation_skips_comparison_without_an_alternate_reuse_pair() {
        let before = build_linear_and4();
        let ordinary = reassociate_gatefn(&before);
        let outcome = reassociate_gatefn_selecting_reuse_with_outcome(&before);

        assert!(!outcome.comparison_performed);
        assert!(!outcome.reuse_candidate_selected);
        assert_eq!(outcome.gate_fn.gates, ordinary.gates);
        assert_eq!(outcome.gate_fn.outputs.len(), ordinary.outputs.len());
        for (actual, expected) in outcome.gate_fn.outputs.iter().zip(&ordinary.outputs) {
            assert_eq!(actual.name, expected.name);
            assert!(
                actual
                    .bit_vector
                    .iter_lsb_to_msb()
                    .eq(expected.bit_vector.iter_lsb_to_msb())
            );
        }
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
    fn reassociate_propagates_large_supergate_provenance_sets() {
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
                .iter()
                .all(|node| node.get_pir_node_ids() == [1, 2, 3, 4])
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
