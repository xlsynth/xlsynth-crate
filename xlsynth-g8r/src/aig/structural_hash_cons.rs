// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::aig::gate::{AigNode, AigOperand, AigRef};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct StructuralOperand {
    expression_id: usize,
    negated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum StructuralKey {
    Literal(bool),
    Input {
        name: String,
        lsb_index: usize,
    },
    And2 {
        lhs: StructuralOperand,
        rhs: StructuralOperand,
    },
}

#[derive(Clone, Copy, Debug)]
struct RefData {
    expression_id: usize,
    depth: usize,
}

#[derive(Clone, Copy, Debug)]
struct ExpressionData {
    min_depth: usize,
    best_ref: AigRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AndNeighbor {
    other: StructuralOperand,
    expression_id: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExistingAndPair {
    pub(crate) lhs_index: usize,
    pub(crate) rhs_index: usize,
    pub(crate) depth: usize,
}

/// Incrementally interns structural AIG expressions as a builder appends nodes.
#[derive(Clone)]
pub(crate) struct StructuralHashCons {
    key_to_expression_id: HashMap<StructuralKey, usize>,
    and_neighbors: Option<HashMap<StructuralOperand, Vec<AndNeighbor>>>,
    expression_data: Vec<ExpressionData>,
    ref_data: Vec<Option<RefData>>,
}

impl StructuralHashCons {
    pub(crate) fn new() -> Self {
        Self {
            key_to_expression_id: HashMap::new(),
            and_neighbors: None,
            expression_data: Vec::new(),
            ref_data: Vec::new(),
        }
    }

    fn structural_operand(&self, operand: AigOperand) -> StructuralOperand {
        let data = self
            .ref_data
            .get(operand.node.id)
            .and_then(Option::as_ref)
            .expect("hash-cons operand must have been registered");
        StructuralOperand {
            expression_id: data.expression_id,
            negated: operand.negated,
        }
    }

    fn and_key(&self, lhs: AigOperand, rhs: AigOperand) -> StructuralKey {
        let mut lhs = self.structural_operand(lhs);
        let mut rhs = self.structural_operand(rhs);
        if rhs < lhs {
            std::mem::swap(&mut lhs, &mut rhs);
        }
        StructuralKey::And2 { lhs, rhs }
    }

    fn register(&mut self, aig_ref: AigRef, depth: usize, key: StructuralKey) -> (usize, bool) {
        if self.ref_data.len() <= aig_ref.id {
            self.ref_data.resize(aig_ref.id + 1, None);
        }
        debug_assert!(self.ref_data[aig_ref.id].is_none());

        let (expression_id, is_new_expression) = match self.key_to_expression_id.get(&key).copied()
        {
            Some(expression_id) => {
                let data = &mut self.expression_data[expression_id];
                if depth < data.min_depth {
                    data.min_depth = depth;
                    data.best_ref = aig_ref;
                }
                (expression_id, false)
            }
            None => {
                let expression_id = self.expression_data.len();
                self.expression_data.push(ExpressionData {
                    min_depth: depth,
                    best_ref: aig_ref,
                });
                self.key_to_expression_id.insert(key, expression_id);
                (expression_id, true)
            }
        };
        self.ref_data[aig_ref.id] = Some(RefData {
            expression_id,
            depth,
        });
        (expression_id, is_new_expression)
    }

    pub(crate) fn register_literal(&mut self, aig_ref: AigRef, value: bool) {
        let _ = self.register(aig_ref, 0, StructuralKey::Literal(value));
    }

    pub(crate) fn register_input(&mut self, aig_ref: AigRef, name: &str, lsb_index: usize) {
        let _ = self.register(
            aig_ref,
            0,
            StructuralKey::Input {
                name: name.to_string(),
                lsb_index,
            },
        );
    }

    /// Returns the cached AIG depth for an already-registered operand.
    pub(crate) fn depth(&self, operand: AigOperand) -> usize {
        self.ref_data[operand.node.id]
            .expect("hash-cons operand must have been registered")
            .depth
    }

    pub(crate) fn find_and(&self, lhs: AigOperand, rhs: AigOperand) -> Option<AigRef> {
        let key = self.and_key(lhs, rhs);
        let expression_id = self.key_to_expression_id.get(&key)?;
        Some(self.expression_data[*expression_id].best_ref)
    }

    /// Returns existing ANDs whose two operands both occur in `operands`.
    ///
    /// Hash maps are used only for keyed lookup. Candidate order comes from
    /// `operands` and each adjacency vector's stable registration order.
    pub(crate) fn find_and_pairs(
        &mut self,
        nodes: &[AigNode],
        operands: &[AigOperand],
    ) -> Vec<ExistingAndPair> {
        self.ensure_and_neighbors(nodes);
        let structural_operands = operands
            .iter()
            .map(|operand| self.structural_operand(*operand))
            .collect::<Vec<_>>();
        let mut indices_by_operand: HashMap<StructuralOperand, Vec<usize>> = HashMap::new();
        for (index, operand) in structural_operands.iter().copied().enumerate() {
            indices_by_operand.entry(operand).or_default().push(index);
        }

        let mut result = Vec::new();
        let and_neighbors = self
            .and_neighbors
            .as_ref()
            .expect("AND adjacency should be initialized");
        for (lhs_index, lhs) in structural_operands.iter().enumerate() {
            let Some(neighbors) = and_neighbors.get(lhs) else {
                continue;
            };
            for neighbor in neighbors {
                let Some(rhs_indices) = indices_by_operand.get(&neighbor.other) else {
                    continue;
                };
                for rhs_index in rhs_indices.iter().copied() {
                    if lhs_index >= rhs_index {
                        continue;
                    }
                    result.push(ExistingAndPair {
                        lhs_index,
                        rhs_index,
                        depth: self.expression_data[neighbor.expression_id].min_depth,
                    });
                }
            }
        }
        result
    }

    fn ensure_and_neighbors(&mut self, nodes: &[AigNode]) {
        if self.and_neighbors.is_some() {
            return;
        }
        let mut and_neighbors = HashMap::new();
        let mut seen_expression = vec![false; self.expression_data.len()];
        // Node-ID order is registration order, so lazily constructing the
        // vectors this way gives the same stable order as incremental updates.
        for (node_id, node) in nodes.iter().enumerate() {
            let AigNode::And2 { a, b, .. } = node else {
                continue;
            };
            let Some(ref_data) = self.ref_data.get(node_id).and_then(Option::as_ref) else {
                continue;
            };
            let expression_id = ref_data.expression_id;
            if seen_expression[expression_id] {
                continue;
            }
            seen_expression[expression_id] = true;
            let key = self.and_key(*a, *b);
            let StructuralKey::And2 { lhs, rhs } = key else {
                unreachable!("and_key must construct an AND key");
            };
            debug_assert_eq!(
                self.key_to_expression_id[&StructuralKey::And2 { lhs, rhs }],
                expression_id
            );
            Self::add_and_neighbor(&mut and_neighbors, lhs, rhs, expression_id);
            if lhs != rhs {
                Self::add_and_neighbor(&mut and_neighbors, rhs, lhs, expression_id);
            }
        }
        self.and_neighbors = Some(and_neighbors);
    }

    fn add_and_neighbor(
        and_neighbors: &mut HashMap<StructuralOperand, Vec<AndNeighbor>>,
        operand: StructuralOperand,
        other: StructuralOperand,
        expression_id: usize,
    ) {
        and_neighbors.entry(operand).or_default().push(AndNeighbor {
            other,
            expression_id,
        });
    }

    fn remove_and_neighbor(
        and_neighbors: &mut HashMap<StructuralOperand, Vec<AndNeighbor>>,
        operand: StructuralOperand,
        other: StructuralOperand,
        expression_id: usize,
    ) {
        let remove_entry = {
            let neighbors = and_neighbors
                .get_mut(&operand)
                .expect("registered AND operand should have an adjacency entry");
            let index = neighbors
                .iter()
                .position(|neighbor| {
                    neighbor.other == other && neighbor.expression_id == expression_id
                })
                .expect("registered AND should have an adjacency edge");
            // Preserve registration order so rollback cannot perturb
            // deterministic candidate traversal.
            neighbors.remove(index);
            neighbors.is_empty()
        };
        if remove_entry {
            and_neighbors.remove(&operand);
        }
    }

    pub(crate) fn register_and(&mut self, aig_ref: AigRef, lhs: AigOperand, rhs: AigOperand) {
        let lhs_depth = self.ref_data[lhs.node.id]
            .expect("hash-cons lhs must have been registered")
            .depth;
        let rhs_depth = self.ref_data[rhs.node.id]
            .expect("hash-cons rhs must have been registered")
            .depth;
        let key = self.and_key(lhs, rhs);
        let StructuralKey::And2 { lhs, rhs } = key.clone() else {
            unreachable!("register_and must construct an AND key");
        };
        let (expression_id, is_new_expression) =
            self.register(aig_ref, std::cmp::max(lhs_depth, rhs_depth) + 1, key);
        if is_new_expression {
            if let Some(and_neighbors) = &mut self.and_neighbors {
                Self::add_and_neighbor(and_neighbors, lhs, rhs, expression_id);
                if lhs != rhs {
                    Self::add_and_neighbor(and_neighbors, rhs, lhs, expression_id);
                }
            }
        }
    }

    /// Removes registrations for append-only AND nodes at or above
    /// `gate_count`.
    pub(crate) fn truncate_to_gate_count(&mut self, gate_count: usize, nodes: &[AigNode]) {
        assert!(self.ref_data.len() <= nodes.len());
        for id in (gate_count.min(self.ref_data.len())..self.ref_data.len()).rev() {
            let Some(ref_data) = self.ref_data[id] else {
                // Folded-away trial nodes are deliberately not hash-consed.
                continue;
            };
            let AigNode::And2 { a, b, .. } = &nodes[id] else {
                panic!("append checkpoint can only roll back AND nodes");
            };
            let key = self.and_key(*a, *b);
            let StructuralKey::And2 { lhs, rhs } = key.clone() else {
                unreachable!("and_key must construct an AND key");
            };
            let removed = self
                .key_to_expression_id
                .remove(&key)
                .expect("trial AND key should be registered");
            if let Some(and_neighbors) = &mut self.and_neighbors {
                Self::remove_and_neighbor(and_neighbors, lhs, rhs, removed);
                if lhs != rhs {
                    Self::remove_and_neighbor(and_neighbors, rhs, lhs, removed);
                }
            }
            assert_eq!(removed, ref_data.expression_id);
            assert_eq!(ref_data.expression_id + 1, self.expression_data.len());
            self.expression_data.pop();
        }
        // `ref_data` can already be shorter than `gate_count` when the graph
        // prefix ends in folded-away, deliberately unregistered nodes.
        self.ref_data.truncate(gate_count);
    }
}
