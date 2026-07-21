// SPDX-License-Identifier: Apache-2.0

//! Choice normalization and bounded k-feasible AIG cut enumeration.

use crate::aig::{AigNode, AigOperand, AigRef, ChoiceAig, GateFn};
use crate::techmap::truth::{
    MAX_TRUTH_TABLE_INPUTS, complement_truth, remap_truth, variable_truth,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Cut {
    pub leaves: Vec<AigRef>,
    pub truth: u64,
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

/// Enumerates bounded structural cuts for every stored AIG node.
pub(super) fn enumerate_cuts(
    graph: &GateFn,
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

    let mut cuts_by_node: Vec<Vec<Cut>> = vec![Vec::new(); graph.gates.len()];
    for (node_id, node) in graph.gates.iter().enumerate() {
        let node_ref = AigRef { id: node_id };
        cuts_by_node[node_id] = match node {
            AigNode::Input { .. } => vec![trivial_cut(node_ref)],
            AigNode::Literal { value, .. } => vec![Cut {
                leaves: Vec::new(),
                truth: u64::from(*value),
            }],
            AigNode::And2 { a, b, .. } => {
                let mut merged = Vec::new();
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
                        merged.push(Cut {
                            leaves,
                            truth: lhs_truth & rhs_truth,
                        });
                    }
                }
                let mut cuts = deduplicate_and_trim(merged, max_cuts_per_node.saturating_sub(1));
                cuts.push(trivial_cut(node_ref));
                cuts
            }
        };
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
    }
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
        by_leaves.entry(cut.leaves.clone()).or_insert(cut);
    }
    let mut deduplicated: Vec<Cut> = by_leaves.into_values().collect();
    deduplicated.sort_by(|lhs, rhs| {
        lhs.leaves
            .len()
            .cmp(&rhs.leaves.len())
            .then_with(|| lhs.leaves.cmp(&rhs.leaves))
            .then_with(|| lhs.truth.cmp(&rhs.truth))
    });
    deduplicated.truncate(limit);
    deduplicated
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
}
