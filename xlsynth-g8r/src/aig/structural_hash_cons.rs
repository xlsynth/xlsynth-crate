// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::aig::gate::{AigOperand, AigRef};

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

/// Incrementally interns structural AIG expressions as a builder appends nodes.
pub(crate) struct StructuralHashCons {
    key_to_expression_id: HashMap<StructuralKey, usize>,
    expression_data: Vec<ExpressionData>,
    ref_data: Vec<Option<RefData>>,
}

impl StructuralHashCons {
    pub(crate) fn new() -> Self {
        Self {
            key_to_expression_id: HashMap::new(),
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

    fn register(&mut self, aig_ref: AigRef, depth: usize, key: StructuralKey) {
        if self.ref_data.len() <= aig_ref.id {
            self.ref_data.resize(aig_ref.id + 1, None);
        }
        debug_assert!(self.ref_data[aig_ref.id].is_none());

        let expression_id = match self.key_to_expression_id.get(&key).copied() {
            Some(expression_id) => {
                let data = &mut self.expression_data[expression_id];
                if depth < data.min_depth {
                    data.min_depth = depth;
                    data.best_ref = aig_ref;
                }
                expression_id
            }
            None => {
                let expression_id = self.expression_data.len();
                self.expression_data.push(ExpressionData {
                    min_depth: depth,
                    best_ref: aig_ref,
                });
                self.key_to_expression_id.insert(key, expression_id);
                expression_id
            }
        };
        self.ref_data[aig_ref.id] = Some(RefData {
            expression_id,
            depth,
        });
    }

    pub(crate) fn register_literal(&mut self, aig_ref: AigRef, value: bool) {
        self.register(aig_ref, 0, StructuralKey::Literal(value));
    }

    pub(crate) fn register_input(&mut self, aig_ref: AigRef, name: &str, lsb_index: usize) {
        self.register(
            aig_ref,
            0,
            StructuralKey::Input {
                name: name.to_string(),
                lsb_index,
            },
        );
    }

    pub(crate) fn find_and(&self, lhs: AigOperand, rhs: AigOperand) -> Option<AigRef> {
        let key = self.and_key(lhs, rhs);
        let expression_id = self.key_to_expression_id.get(&key)?;
        Some(self.expression_data[*expression_id].best_ref)
    }

    pub(crate) fn register_and(&mut self, aig_ref: AigRef, lhs: AigOperand, rhs: AigOperand) {
        let lhs_depth = self.ref_data[lhs.node.id]
            .expect("hash-cons lhs must have been registered")
            .depth;
        let rhs_depth = self.ref_data[rhs.node.id]
            .expect("hash-cons rhs must have been registered")
            .depth;
        self.register(
            aig_ref,
            std::cmp::max(lhs_depth, rhs_depth) + 1,
            self.and_key(lhs, rhs),
        );
    }
}
