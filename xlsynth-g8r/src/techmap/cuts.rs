// SPDX-License-Identifier: Apache-2.0

//! Choice normalization and NF-style bounded priority-cut enumeration.

use crate::aig::{AigNode, AigOperand, AigRef, ChoiceAig, GateFn};
use crate::techmap::liberty_index::LibertyCellIndex;
use crate::techmap::truth::{
    MAX_TRUTH_TABLE_INPUTS, complement_truth, minimize_support, remap_truth, variable_truth,
};
use anyhow::{Result, anyhow};
use std::collections::BTreeMap;

/// One equivalence class formed by ABC structural-choice sibling links.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ChoiceClass {
    pub canonical: AigRef,
    pub members: Vec<AigRef>,
}

/// Choice classes plus the all-zero phase used to normalize complemented
/// choices.
#[derive(Clone, Debug)]
pub(super) struct ChoiceAnalysis {
    pub classes: Vec<ChoiceClass>,
    pub class_by_node: Vec<usize>,
    pub phase_by_node: Vec<bool>,
}

impl ChoiceAnalysis {
    /// Returns the class and canonical-relative polarity for one AIG operand.
    pub fn state_for_operand(&self, operand: AigOperand) -> (usize, bool) {
        let class_id = self.class_by_node[operand.node.id];
        let canonical = self.classes[class_id].canonical;
        let polarity = operand.negated
            ^ self.phase_by_node[operand.node.id]
            ^ self.phase_by_node[canonical.id];
        (class_id, polarity)
    }

    /// Returns the class and canonical-relative polarity for a positive node.
    pub fn state_for_positive_node(&self, node: AigRef) -> (usize, bool) {
        self.state_for_operand(node.into())
    }
}

/// One cut, with truth expressed over leaves in their sorted order.
///
/// The extra fields are priority-cut estimates, not final mapping costs. They
/// serve the same purpose as ABC NF's useful/flow/delay cut ordering: keep a
/// small, mapping-relevant cut set before expensive Liberty matching starts.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Cut {
    pub leaves: Vec<AigRef>,
    pub truth: u64,
    pub useful: bool,
    pub flow: f64,
    pub delay: usize,
}

/// Builds deterministic choice classes and computes each node's all-zero phase.
pub(super) fn analyze_choices(choice_aig: &ChoiceAig) -> Result<ChoiceAnalysis> {
    let graph = choice_aig.graph();
    validate_topological(graph)?;
    let raw_classes = choice_aig.choice_classes();
    let mut classes = Vec::with_capacity(raw_classes.len());
    let mut class_by_node = vec![usize::MAX; graph.gates.len()];
    for (class_id, members) in raw_classes.into_iter().enumerate() {
        let canonical = members[0];
        for member in &members {
            class_by_node[member.id] = class_id;
        }
        classes.push(ChoiceClass { canonical, members });
    }
    if class_by_node.iter().any(|class_id| *class_id == usize::MAX) {
        return Err(anyhow!("choice analysis did not classify every AIG node"));
    }
    let phase_by_node = compute_zero_phase(graph);
    Ok(ChoiceAnalysis {
        classes,
        class_by_node,
        phase_by_node,
    })
}

/// Enumerates bounded structural cuts for an ordinary no-choice AIG.
///
/// This remains useful for focused cut tests. Production technology mapping
/// uses enumerate_choice_cuts so sibling alternatives propagate through parent
/// cut enumeration just as they do in ABC NF.
#[cfg(test)]
pub(super) fn enumerate_cuts(
    graph: &GateFn,
    max_cut_size: usize,
    max_cuts_per_node: usize,
) -> Result<Vec<Vec<Cut>>> {
    enumerate_cuts_impl(graph, None, None, None, max_cut_size, max_cuts_per_node)
}

/// Enumerates NF-style priority cuts while propagating sibling choice cuts.
pub(super) fn enumerate_choice_cuts(
    choice_aig: &ChoiceAig,
    analysis: &ChoiceAnalysis,
    cell_index: &LibertyCellIndex,
    max_cut_size: usize,
    max_cuts_per_node: usize,
) -> Result<Vec<Vec<Cut>>> {
    enumerate_cuts_impl(
        choice_aig.graph(),
        Some(choice_aig.sibling_links()),
        Some(analysis.phase_by_node.as_slice()),
        Some(cell_index),
        max_cut_size,
        max_cuts_per_node,
    )
}

fn enumerate_cuts_impl(
    graph: &GateFn,
    sibling_links: Option<&[Option<AigRef>]>,
    phase_by_node: Option<&[bool]>,
    cell_index: Option<&LibertyCellIndex>,
    max_cut_size: usize,
    max_cuts_per_node: usize,
) -> Result<Vec<Vec<Cut>>> {
    if max_cut_size == 0 || max_cut_size > MAX_TRUTH_TABLE_INPUTS {
        return Err(anyhow!(
            "max_cut_size must be in 1..={}, got {}",
            MAX_TRUTH_TABLE_INPUTS,
            max_cut_size
        ));
    }
    if max_cuts_per_node == 0 {
        return Err(anyhow!("max_cuts_per_node must be at least 1"));
    }
    validate_topological(graph)?;

    let fanout_estimates = structural_fanout_estimates(graph);
    let mut cuts_by_node: Vec<Vec<Cut>> = vec![Vec::new(); graph.gates.len()];
    for (node_id, node) in graph.gates.iter().enumerate() {
        let node_ref = AigRef { id: node_id };
        let mut generated = Vec::new();
        if let Some(sibling) = sibling_links
            .and_then(|links| links.get(node_id))
            .copied()
            .flatten()
        {
            let complement = phase_by_node
                .map(|phases| phases[node_id] ^ phases[sibling.id])
                .unwrap_or(false);
            for sibling_cut in propagatable_cuts(cuts_by_node[sibling.id].as_slice(), sibling) {
                let mut cut = sibling_cut.clone();
                if complement {
                    cut.truth = complement_truth(cut.truth, cut.leaves.len());
                }
                refresh_cut_priority(
                    &mut cut,
                    node_ref,
                    cuts_by_node.as_slice(),
                    fanout_estimates.as_slice(),
                    cell_index,
                );
                generated.push(cut);
            }
        }

        match node {
            AigNode::Input { .. } => {
                if generated.is_empty() {
                    cuts_by_node[node_id] = vec![trivial_cut(node_ref)];
                    continue;
                }
            }
            AigNode::Literal { value, .. } => {
                generated.push(Cut {
                    leaves: Vec::new(),
                    truth: u64::from(*value),
                    useful: true,
                    flow: 0.0,
                    delay: 0,
                });
            }
            AigNode::And2 { a, b, .. } => {
                for lhs in &cuts_by_node[a.node.id] {
                    for rhs in &cuts_by_node[b.node.id] {
                        let leaves = merge_leaves(lhs.leaves.as_slice(), rhs.leaves.as_slice());
                        if leaves.len() > max_cut_size {
                            continue;
                        }
                        let mut lhs_truth =
                            remap_truth(lhs.truth, lhs.leaves.as_slice(), leaves.as_slice());
                        let mut rhs_truth =
                            remap_truth(rhs.truth, rhs.leaves.as_slice(), leaves.as_slice());
                        if a.negated {
                            lhs_truth = complement_truth(lhs_truth, leaves.len());
                        }
                        if b.negated {
                            rhs_truth = complement_truth(rhs_truth, leaves.len());
                        }
                        let (truth, leaves) =
                            minimize_support(lhs_truth & rhs_truth, leaves.as_slice());
                        let mut cut = Cut {
                            leaves,
                            truth,
                            useful: false,
                            flow: 0.0,
                            delay: 0,
                        };
                        refresh_cut_priority(
                            &mut cut,
                            node_ref,
                            cuts_by_node.as_slice(),
                            fanout_estimates.as_slice(),
                            cell_index,
                        );
                        generated.push(cut);
                    }
                }
            }
        }

        let reserve_trivial = matches!(node, AigNode::Input { .. } | AigNode::And2 { .. });
        let limit = max_cuts_per_node.saturating_sub(usize::from(reserve_trivial));
        let mut cuts = deduplicate_and_trim(generated, limit);
        if reserve_trivial {
            cuts.push(trivial_cut(node_ref));
        }
        if cuts.is_empty() {
            cuts.push(trivial_cut(node_ref));
        }
        cuts_by_node[node_id] = cuts;
    }
    Ok(cuts_by_node)
}

/// Returns the number of enumerated cuts, including propagation-only trivial
/// cuts.
pub(super) fn cut_count(cuts_by_node: &[Vec<Cut>]) -> usize {
    cuts_by_node.iter().map(Vec::len).sum()
}

fn validate_topological(graph: &GateFn) -> Result<()> {
    for (node_id, node) in graph.gates.iter().enumerate() {
        let AigNode::And2 { a, b, .. } = node else {
            continue;
        };
        for operand in [a, b] {
            if operand.node.id >= node_id {
                return Err(anyhow!(
                    "technology mapping requires topological AIG storage: node {} depends on {}",
                    node_id,
                    operand.node.id
                ));
            }
        }
    }
    Ok(())
}

fn compute_zero_phase(graph: &GateFn) -> Vec<bool> {
    let mut phases = vec![false; graph.gates.len()];
    for (node_id, node) in graph.gates.iter().enumerate() {
        phases[node_id] = match node {
            AigNode::Input { .. } => false,
            AigNode::Literal { value, .. } => *value,
            AigNode::And2 { a, b, .. } => {
                operand_phase(*a, phases.as_slice()) & operand_phase(*b, phases.as_slice())
            }
        };
    }
    phases
}

fn operand_phase(operand: AigOperand, phases: &[bool]) -> bool {
    phases[operand.node.id] ^ operand.negated
}

fn trivial_cut(node: AigRef) -> Cut {
    Cut {
        leaves: vec![node],
        truth: variable_truth(1, 0),
        useful: false,
        flow: 0.0,
        delay: 0,
    }
}

fn propagatable_cuts(cuts: &[Cut], root: AigRef) -> impl Iterator<Item = &Cut> {
    let has_nontrivial = cuts.iter().any(|cut| !is_trivial_cut_for_node(cut, root));
    cuts.iter()
        .filter(move |cut| !has_nontrivial || !is_trivial_cut_for_node(cut, root))
}

fn is_trivial_cut_for_node(cut: &Cut, root: AigRef) -> bool {
    cut.leaves == [root] && cut.truth == variable_truth(1, 0)
}

fn structural_fanout_estimates(graph: &GateFn) -> Vec<usize> {
    let mut fanouts = vec![0usize; graph.gates.len()];
    for node in &graph.gates {
        let AigNode::And2 { a, b, .. } = node else {
            continue;
        };
        fanouts[a.node.id] += 1;
        fanouts[b.node.id] += 1;
    }
    for output in &graph.outputs {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            fanouts[operand.node.id] += 1;
        }
    }
    fanouts
}

fn refresh_cut_priority(
    cut: &mut Cut,
    root: AigRef,
    cuts_by_node: &[Vec<Cut>],
    fanout_estimates: &[usize],
    cell_index: Option<&LibertyCellIndex>,
) {
    cut.useful = cell_index.is_none_or(|index| {
        !index.matches(cut.leaves.len(), cut.truth).is_empty()
            || !index
                .matches(
                    cut.leaves.len(),
                    complement_truth(cut.truth, cut.leaves.len()),
                )
                .is_empty()
    });
    let mut flow = if cut.leaves.len() < 2 {
        0.0
    } else {
        cut.leaves.len() as f64
    };
    let mut delay = 0usize;
    for leaf in &cut.leaves {
        let leaf_best = cuts_by_node.get(leaf.id).and_then(|cuts| {
            cuts.iter()
                .find(|candidate| !is_trivial_cut_for_node(candidate, *leaf))
        });
        if let Some(leaf_best) = leaf_best {
            flow += leaf_best.flow;
            delay = delay.max(leaf_best.delay);
        }
    }
    let root_flow_refs = 2.0 * fanout_estimates[root.id].max(1) as f64;
    cut.flow = flow / root_flow_refs;
    cut.delay = delay + usize::from(cut.leaves.len() > 1);
}

fn merge_leaves(lhs: &[AigRef], rhs: &[AigRef]) -> Vec<AigRef> {
    let mut merged = Vec::with_capacity(lhs.len() + rhs.len());
    let (mut lhs_index, mut rhs_index) = (0usize, 0usize);
    while lhs_index < lhs.len() && rhs_index < rhs.len() {
        match lhs[lhs_index].cmp(&rhs[rhs_index]) {
            std::cmp::Ordering::Less => {
                merged.push(lhs[lhs_index]);
                lhs_index += 1;
            }
            std::cmp::Ordering::Equal => {
                merged.push(lhs[lhs_index]);
                lhs_index += 1;
                rhs_index += 1;
            }
            std::cmp::Ordering::Greater => {
                merged.push(rhs[rhs_index]);
                rhs_index += 1;
            }
        }
    }
    merged.extend_from_slice(&lhs[lhs_index..]);
    merged.extend_from_slice(&rhs[rhs_index..]);
    merged
}

fn deduplicate_and_trim(cuts: Vec<Cut>, limit: usize) -> Vec<Cut> {
    let mut by_leaves: BTreeMap<Vec<AigRef>, Cut> = BTreeMap::new();
    for cut in cuts {
        match by_leaves.get(&cut.leaves) {
            Some(existing) if cut_priority_order(existing, &cut).is_le() => {}
            _ => {
                by_leaves.insert(cut.leaves.clone(), cut);
            }
        }
    }
    let mut deduplicated: Vec<Cut> = by_leaves.into_values().collect();
    deduplicated.sort_by(cut_priority_order);
    let mut retained = Vec::new();
    for cut in deduplicated {
        if retained.iter().any(|existing: &Cut| {
            leaves_are_subset(existing.leaves.as_slice(), cut.leaves.as_slice())
        }) {
            continue;
        }
        retained.push(cut);
        if retained.len() == limit {
            break;
        }
    }
    retained
}

fn cut_priority_order(lhs: &Cut, rhs: &Cut) -> std::cmp::Ordering {
    rhs.useful
        .cmp(&lhs.useful)
        .then_with(|| lhs.flow.total_cmp(&rhs.flow))
        .then_with(|| lhs.delay.cmp(&rhs.delay))
        .then_with(|| lhs.leaves.len().cmp(&rhs.leaves.len()))
        .then_with(|| lhs.leaves.cmp(&rhs.leaves))
        .then_with(|| lhs.truth.cmp(&rhs.truth))
}

fn leaves_are_subset(lhs: &[AigRef], rhs: &[AigRef]) -> bool {
    let mut lhs_index = 0usize;
    let mut rhs_index = 0usize;
    while lhs_index < lhs.len() && rhs_index < rhs.len() {
        match lhs[lhs_index].cmp(&rhs[rhs_index]) {
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {
                lhs_index += 1;
                rhs_index += 1;
            }
            std::cmp::Ordering::Greater => rhs_index += 1,
        }
    }
    lhs_index == lhs.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{GateBuilder, GateBuilderOptions};

    #[test]
    fn and_cut_truth_includes_complemented_edges() {
        let mut builder = GateBuilder::new("cuts".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let root = builder.add_and_binary(a, b.negate());
        builder.add_output("o".to_string(), root.into());
        let graph = builder.build();

        let cuts = enumerate_cuts(&graph, 2, 8).unwrap();
        let root_cut = cuts[root.node.id]
            .iter()
            .find(|cut| cut.leaves == vec![a.node, b.node])
            .unwrap();

        assert_eq!(root_cut.truth, 0b0010);
    }

    #[test]
    fn choice_analysis_normalizes_relative_phase() {
        let mut builder = GateBuilder::new("choices".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let not_a = builder.add_and_binary(a.negate(), a.negate());
        builder.add_output("o".to_string(), not_a.into());
        let graph = builder.build();
        let mut siblings = vec![None; graph.gates.len()];
        siblings[not_a.node.id] = Some(a.node);
        let choice_aig = ChoiceAig::new(graph, siblings).unwrap();

        let analysis = analyze_choices(&choice_aig).unwrap();
        let (_, a_polarity) = analysis.state_for_positive_node(a.node);
        let (_, not_a_polarity) = analysis.state_for_positive_node(not_a.node);

        assert!(!a_polarity);
        assert!(not_a_polarity);
    }

    #[test]
    fn one_leaf_sibling_cut_propagates_to_parent() {
        let mut builder = GateBuilder::new(
            "choice_propagation".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let c: AigOperand = builder.add_input("c".to_string(), 1).try_into().unwrap();
        let choice = builder.add_and_binary(a, b);
        let parent = builder.add_and_binary(choice.into(), c);
        builder.add_output("o".to_string(), parent.into());
        let graph = builder.build();
        let mut siblings = vec![None; graph.gates.len()];
        siblings[choice.node.id] = Some(a.node);
        let choice_aig = ChoiceAig::new(graph, siblings).unwrap();
        let analysis = analyze_choices(&choice_aig).unwrap();

        let cuts = enumerate_cuts_impl(
            choice_aig.graph(),
            Some(choice_aig.sibling_links()),
            Some(analysis.phase_by_node.as_slice()),
            None,
            2,
            8,
        )
        .unwrap();

        assert!(
            cuts[parent.node.id]
                .iter()
                .any(|cut| cut.leaves == vec![a.node, c.node])
        );
    }
}
