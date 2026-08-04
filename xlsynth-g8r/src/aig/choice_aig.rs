// SPDX-License-Identifier: Apache-2.0

//! Choice-preserving wrapper around the ordinary combinational AIG.
//!
//! ABC's GIA representation records structural choices as one optional
//! sibling link per object. Following the links from a node yields the other
//! structurally distinct implementations of the same Boolean function.

use crate::aig::{AigRef, GateFn};
use std::collections::BTreeMap;

/// A GateFn together with ABC-style structural-choice sibling chains.
///
/// graph.gates may intentionally contain nodes that are unreachable from the
/// primary outputs. Choice alternatives commonly live in those otherwise
/// dead cones, so consumers must follow sibling_next in addition to ordinary
/// AIG fanins when traversing the choice graph. Each choice class is one
/// nonbranching sibling chain, and only its head may have ordinary AIG or
/// primary-output fanout.
#[derive(Debug, Clone)]
pub struct ChoiceAig {
    /// Ordinary AIG storage, including any otherwise-dead choice cones.
    graph: GateFn,

    /// One optional next-sibling link per graph.gates entry.
    ///
    /// This mirrors ABC's pSibls: sibling_next[node.id] is the next
    /// alternative for node. Links always point to an earlier AIG node, and
    /// their targets cannot have ordinary graph fanout or another predecessor.
    sibling_next: Vec<Option<AigRef>>,
}

impl ChoiceAig {
    /// Constructs a choice AIG after validating canonical ABC sibling chains.
    pub fn new(graph: GateFn, sibling_next: Vec<Option<AigRef>>) -> Result<Self, String> {
        if sibling_next.len() != graph.gates.len() {
            return Err(format!(
                "choice sibling table has {} entries but graph has {} gates",
                sibling_next.len(),
                graph.gates.len()
            ));
        }
        let mut sibling_predecessors = vec![None; graph.gates.len()];
        for (node_id, sibling) in sibling_next.iter().enumerate() {
            let Some(sibling) = sibling else {
                continue;
            };
            if sibling.id >= graph.gates.len() {
                return Err(format!(
                    "choice sibling of node {} is out of bounds: {} >= {}",
                    node_id,
                    sibling.id,
                    graph.gates.len()
                ));
            }
            if sibling.id >= node_id {
                return Err(format!(
                    "choice sibling of node {} must be earlier than the node, got {}",
                    node_id, sibling.id
                ));
            }
            if let Some(previous) = sibling_predecessors[sibling.id].replace(node_id) {
                return Err(format!(
                    "choice sibling node {} has multiple chain predecessors: {} and {}; each choice class must have one canonical head",
                    sibling.id, previous, node_id
                ));
            }
        }

        let mut ordinary_fanouts = vec![0usize; graph.gates.len()];
        for node in &graph.gates {
            for operand in node.operands() {
                ordinary_fanouts[operand.node.id] += 1;
            }
        }
        for output in &graph.outputs {
            for operand in output.bit_vector.iter_lsb_to_msb() {
                ordinary_fanouts[operand.node.id] += 1;
            }
        }
        for (node_id, predecessor) in sibling_predecessors.iter().enumerate() {
            if predecessor.is_some() && ordinary_fanouts[node_id] != 0 {
                return Err(format!(
                    "choice sibling node {} has {} ordinary AIG or primary-output fanout{}; only the canonical chain head may have fanout",
                    node_id,
                    ordinary_fanouts[node_id],
                    if ordinary_fanouts[node_id] == 1 {
                        ""
                    } else {
                        "s"
                    }
                ));
            }
        }
        Ok(Self {
            graph,
            sibling_next,
        })
    }

    /// Wraps an ordinary AIG with no structural choices.
    pub fn without_choices(graph: GateFn) -> Self {
        let sibling_next = vec![None; graph.gates.len()];
        Self {
            graph,
            sibling_next,
        }
    }

    /// Returns the underlying ordinary AIG, including dead choice cones.
    pub fn graph(&self) -> &GateFn {
        &self.graph
    }

    /// Returns the ABC-style sibling table indexed by AIG node ID.
    pub fn sibling_links(&self) -> &[Option<AigRef>] {
        &self.sibling_next
    }

    /// Consumes the wrapper and returns its validated graph and sibling table.
    pub fn into_parts(self) -> (GateFn, Vec<Option<AigRef>>) {
        (self.graph, self.sibling_next)
    }

    /// Returns the next structural-choice sibling, if any.
    pub fn next_sibling(&self, node: AigRef) -> Option<AigRef> {
        self.sibling_next[node.id]
    }

    /// Iterates over node followed by every linked sibling.
    pub fn sibling_chain(&self, node: AigRef) -> impl Iterator<Item = AigRef> + '_ {
        std::iter::successors(Some(node), move |current| self.sibling_next[current.id])
    }

    /// Returns the number of stored sibling links.
    pub fn sibling_link_count(&self) -> usize {
        self.sibling_next
            .iter()
            .filter(|sibling| sibling.is_some())
            .count()
    }

    /// Returns every structural-choice equivalence class in deterministic
    /// order.
    ///
    /// ABC stores only one backwards sibling link per node. A consumer that
    /// starts from an older member therefore cannot discover later siblings by
    /// following links alone. This method closes the undirected sibling
    /// relation and returns each class sorted by AIG node ID; classes
    /// themselves are ordered by their earliest member.
    pub fn choice_classes(&self) -> Vec<Vec<AigRef>> {
        let mut parents: Vec<usize> = (0..self.graph.gates.len()).collect();
        for (node_id, sibling) in self.sibling_next.iter().enumerate() {
            let Some(sibling) = sibling else {
                continue;
            };
            union_roots(&mut parents, node_id, sibling.id);
        }

        let mut members_by_root: BTreeMap<usize, Vec<AigRef>> = BTreeMap::new();
        for node_id in 0..self.graph.gates.len() {
            let root = find_root(&mut parents, node_id);
            members_by_root
                .entry(root)
                .or_default()
                .push(AigRef { id: node_id });
        }

        let mut classes: Vec<Vec<AigRef>> = members_by_root.into_values().collect();
        classes.sort_by_key(|members| members[0].id);
        classes
    }
}

fn find_root(parents: &mut [usize], node: usize) -> usize {
    if parents[node] != node {
        parents[node] = find_root(parents, parents[node]);
    }
    parents[node]
}

fn union_roots(parents: &mut [usize], lhs: usize, rhs: usize) {
    let lhs_root = find_root(parents, lhs);
    let rhs_root = find_root(parents, rhs);
    if lhs_root == rhs_root {
        return;
    }
    let (earlier, later) = if lhs_root < rhs_root {
        (lhs_root, rhs_root)
    } else {
        (rhs_root, lhs_root)
    };
    parents[later] = earlier;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{AigOperand, GateBuilder, GateBuilderOptions};

    #[test]
    fn sibling_chain_follows_earlier_nodes() {
        let mut builder = GateBuilder::new("choices".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let first = builder.add_and_binary(a, b);
        let second = builder.add_and_binary(a, b);
        builder.add_output("o".to_string(), second.into());
        let graph = builder.build();
        let mut sibling_next = vec![None; graph.gates.len()];
        sibling_next[second.node.id] = Some(first.node);

        let choice_aig = ChoiceAig::new(graph, sibling_next).unwrap();
        let chain: Vec<AigRef> = choice_aig.sibling_chain(second.node).collect();

        assert_eq!(chain, vec![second.node, first.node]);
        assert_eq!(choice_aig.sibling_link_count(), 1);
    }

    #[test]
    fn rejects_non_earlier_sibling_link() {
        let mut builder = GateBuilder::new("choices".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        builder.add_output("o".to_string(), a.into());
        let graph = builder.build();
        let mut sibling_next = vec![None; graph.gates.len()];
        sibling_next[a.node.id] = Some(a.node);

        let err = ChoiceAig::new(graph, sibling_next).unwrap_err();

        assert!(err.contains("must be earlier"));
    }

    #[test]
    fn rejects_sibling_with_ordinary_aig_fanout() {
        let mut builder = GateBuilder::new("choices".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let alternative = builder.add_and_binary(a, b);
        let head = builder.add_and_binary(a, b);
        let consumer = builder.add_and_binary(alternative, a);
        builder.add_output("head".to_string(), head.into());
        builder.add_output("consumer".to_string(), consumer.into());
        let graph = builder.build();
        let mut sibling_next = vec![None; graph.gates.len()];
        sibling_next[head.node.id] = Some(alternative.node);

        let error = ChoiceAig::new(graph, sibling_next).unwrap_err();

        assert!(error.contains("ordinary AIG or primary-output fanout"));
    }

    #[test]
    fn rejects_sibling_driving_a_primary_output() {
        let mut builder = GateBuilder::new("choices".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let alternative = builder.add_and_binary(a, b);
        let head = builder.add_and_binary(a, b);
        builder.add_output("alternative".to_string(), alternative.into());
        builder.add_output("head".to_string(), head.into());
        let graph = builder.build();
        let mut sibling_next = vec![None; graph.gates.len()];
        sibling_next[head.node.id] = Some(alternative.node);

        let error = ChoiceAig::new(graph, sibling_next).unwrap_err();

        assert!(error.contains("ordinary AIG or primary-output fanout"));
    }

    #[test]
    fn rejects_branching_sibling_chains() {
        let mut builder = GateBuilder::new("choices".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let alternative = builder.add_and_binary(a, b);
        let first_head = builder.add_and_binary(a, b);
        let second_head = builder.add_and_binary(a, b);
        builder.add_output("first".to_string(), first_head.into());
        builder.add_output("second".to_string(), second_head.into());
        let graph = builder.build();
        let mut sibling_next = vec![None; graph.gates.len()];
        sibling_next[first_head.node.id] = Some(alternative.node);
        sibling_next[second_head.node.id] = Some(alternative.node);

        let error = ChoiceAig::new(graph, sibling_next).unwrap_err();

        assert!(error.contains("multiple chain predecessors"));
    }

    #[test]
    fn exposes_read_only_parts_and_can_return_ownership() {
        let mut builder = GateBuilder::new("choices".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        builder.add_output("o".to_string(), a.into());
        let graph = builder.build();
        let gate_count = graph.gates.len();
        let choice_aig = ChoiceAig::without_choices(graph);

        assert_eq!(choice_aig.graph().gates.len(), gate_count);
        assert_eq!(choice_aig.sibling_links(), vec![None; gate_count]);

        let (graph, sibling_next) = choice_aig.into_parts();
        assert_eq!(graph.gates.len(), gate_count);
        assert_eq!(sibling_next, vec![None; gate_count]);
    }

    #[test]
    fn choice_classes_close_backwards_sibling_links() {
        let mut builder = GateBuilder::new("choices".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let first = builder.add_and_binary(a, b);
        let second = builder.add_and_binary(a, b);
        let third = builder.add_and_binary(a, b);
        builder.add_output("o".to_string(), third.into());
        let graph = builder.build();
        let mut sibling_next = vec![None; graph.gates.len()];
        sibling_next[second.node.id] = Some(first.node);
        sibling_next[third.node.id] = Some(second.node);

        let choice_aig = ChoiceAig::new(graph, sibling_next).unwrap();
        let classes = choice_aig.choice_classes();
        let class = classes
            .iter()
            .find(|members| members.contains(&first.node))
            .unwrap();

        assert_eq!(class, &vec![first.node, second.node, third.node]);
    }
}
