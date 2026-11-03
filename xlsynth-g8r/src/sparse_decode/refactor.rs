// SPDX-License-Identifier: Apache-2.0

use super::decoder_inputs;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef, GateFn};
use crate::aig::retained_rails::{CutIndex, RailBudget};
use crate::aig::topo::topo_sort_refs;
use crate::gate_builder::GateBuilder;

/// Observe the original edge phase on the rebuilt graph. A later divisor is
/// unavailable, which also prevents a substitution from introducing a cycle.
fn mapped_edge(mapped: &[Option<AigOperand>], e: AigOperand) -> Option<AigOperand> {
    let v = mapped[e.node.id]?;
    Some(if e.negated { v.negate() } else { v })
}

/// A snapshot of the original decoder's product rails and observation
/// boundaries.
struct Products<'a> {
    old: &'a GateFn,
    index: &'a CutIndex,
    key_bit: Vec<bool>,
    keep: Vec<bool>,
}

fn named(node: &AigNode) -> bool {
    node.get_tags().is_some_and(|tags| !tags.is_empty())
}

impl Products<'_> {
    /// Regroup k & (x & y) around an existing k & x. For k & (x | y),
    /// distribution costs an extra AND, so require a named k & x that already
    /// has to be implemented elsewhere.
    fn reuse(
        &self,
        g: &mut GateBuilder,
        mapped: &[Option<AigOperand>],
        key: AigOperand,
        inner: AigOperand,
    ) -> Option<AigOperand> {
        if !self.key_bit[key.node.id] || self.keep[inner.node.id] {
            return None;
        }
        let AigNode::And2 { a, b, .. } = *self.old.get(inner.node) else {
            return None;
        };
        let (a, b) = if inner.negated {
            (a.negate(), b.negate())
        } else {
            (a, b)
        };
        for (x, y) in [(a, b), (b, a)] {
            let Some(rail) = self.index.conjunction(key, x) else {
                continue;
            };
            if rail.negated
                || !matches!(self.old.get(rail.node), AigNode::And2 { .. })
                || inner.negated && !named(self.old.get(rail.node))
            {
                continue;
            }
            let Some(product) = mapped_edge(mapped, rail) else {
                continue;
            };
            let other = mapped_edge(mapped, y)?;
            return Some(if inner.negated {
                let term = g.add_and_binary(mapped_edge(mapped, key)?, other);
                g.add_or_binary(product, term)
            } else {
                g.add_and_binary(product, other)
            });
        }
        None
    }
}

/// A rebuilt decoder and the signed implementation of each mapped old node.
pub(super) struct Refactored {
    pub graph: GateFn,
    pub mapped: Vec<Option<AigOperand>>,
}

/// Try one local factoring sweep on the incumbent when a bank loses
/// observations.
pub(super) fn reuse_key_products(
    old: &GateFn,
    index: &CutIndex,
    offset: usize,
    width: usize,
    budget: &mut RailBudget,
) -> Option<Refactored> {
    // One visit and at most four conjunction lookups per incumbent gate.
    if !budget.take(5 * old.gates.len()) {
        return None;
    }
    let mut products = Products {
        old,
        index,
        key_bit: vec![false; old.gates.len()],
        keep: vec![false; old.gates.len()],
    };
    let (mut g, pins) = decoder_inputs(old);
    let mut mapped = vec![None; old.gates.len()];
    for (bit, (prev, pin)) in old
        .inputs
        .iter()
        .flat_map(|p| p.bit_vector.iter_lsb_to_msb())
        .zip(pins)
        .enumerate()
    {
        products.key_bit[prev.node.id] = (offset..offset + width).contains(&bit);
        mapped[prev.node.id] = Some(pin);
    }
    // Bypassing a gate cannot remove it while a port or a retained named AND
    // still consumes it. Keep named ANDs in their original phase and topology.
    for e in old
        .outputs
        .iter()
        .flat_map(|p| p.bit_vector.iter_lsb_to_msb())
    {
        products.keep[e.node.id] = true;
    }
    for (id, n) in old.gates.iter().enumerate() {
        if named(n) && index.signature(AigRef { id }.into()).is_some() {
            products.keep[id] = true;
            for child in n.args() {
                products.keep[child.id] = true;
            }
        }
    }
    let mut changed = false;
    for node in topo_sort_refs(&old.gates) {
        let n = old.get(node);
        let value = match *n {
            AigNode::Literal { value, .. } => {
                if value {
                    g.get_true()
                } else {
                    g.get_false()
                }
            }
            AigNode::And2 { a, b, .. } if index.signature(node.into()).is_some() => {
                let alternate = if named(n) {
                    None
                } else {
                    products
                        .reuse(&mut g, &mapped, a, b)
                        .or_else(|| products.reuse(&mut g, &mapped, b, a))
                };
                changed |= alternate.is_some();
                match alternate {
                    Some(value) => value,
                    None => g.add_and_binary(mapped_edge(&mapped, a)?, mapped_edge(&mapped, b)?),
                }
            }
            // Inputs are already mapped. Only live ANDs are needed.
            _ => continue,
        };
        mapped[node.id] = Some(value);
    }
    if !changed {
        return None;
    }
    for port in &old.outputs {
        let bits = port
            .bit_vector
            .iter_lsb_to_msb()
            .map(|e| mapped_edge(&mapped, *e))
            .collect::<Option<Vec<_>>>()?;
        g.add_output(port.name.clone(), AigBitVector::from_lsb_is_index_0(&bits));
    }
    Some(Refactored {
        graph: g.build(),
        mapped,
    })
}
