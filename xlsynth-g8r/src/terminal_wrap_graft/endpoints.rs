// SPDX-License-Identifier: Apache-2.0

use crate::aig::gate::{AigBitVector, AigNode, AigOperand, GateFn};
use crate::aig::topo::post_order_operands;
use std::collections::HashSet;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Cut {
    Impossible,
    Unchanged,
    /// Each signed operand must be true. An empty conjunction is true.
    Literals(Vec<AigOperand>),
}

/// Returns equivalent conjunctions for the all-zero and all-ones word, in that
/// order.
pub(super) fn analyze(graph: &GateFn, word: &AigBitVector) -> [Cut; 2] {
    let bits: Vec<_> = word.iter_lsb_to_msb().copied().collect();
    if bits
        .iter()
        .all(|bit| matches!(graph.get(bit.node), AigNode::Input { .. }))
    {
        let pins: HashSet<_> = bits.iter().copied().collect();
        return if bits.iter().any(|bit| pins.contains(&bit.negate())) {
            [Cut::Impossible, Cut::Impossible]
        } else {
            [Cut::Unchanged, Cut::Unchanged]
        };
    }
    let cone = WordCone::new(graph, &bits);
    [false, true].map(|ones| {
        let required: Vec<_> = bits
            .iter()
            .map(|&bit| if ones { bit } else { bit.negate() })
            .collect();
        let Some(facts) = cone.propagate(&required) else {
            return Cut::Impossible;
        };
        let atoms = cone.expand(&required, &facts);
        cone.check(&required, &facts, atoms)
    })
}

struct WordCone<'a> {
    graph: &'a GateFn,
    users: Vec<Vec<usize>>,
    depth: Vec<usize>,
    constants: Vec<AigOperand>,
}

impl<'a> WordCone<'a> {
    fn new(graph: &'a GateFn, word: &[AigOperand]) -> Self {
        let mut users = vec![Vec::new(); graph.gates.len()];
        let mut depth = vec![0; graph.gates.len()];
        let mut constants = Vec::new();
        let mut visited = vec![false; graph.gates.len()];
        for bit in post_order_operands(word, &graph.gates, false) {
            let id = bit.node.id;
            if std::mem::replace(&mut visited[id], true) {
                continue;
            }
            match graph.get(bit.node) {
                AigNode::And2 { a, b, .. } => {
                    depth[id] = 1 + depth[a.node.id].max(depth[b.node.id]);
                    users[a.node.id].push(id);
                    if b.node != a.node {
                        users[b.node.id].push(id);
                    }
                }
                AigNode::Literal { value, .. } => constants.push(AigOperand {
                    node: bit.node,
                    negated: !value,
                }),
                AigNode::Input { .. } => {}
            }
        }
        Self {
            graph,
            users,
            depth,
            constants,
        }
    }

    /// Unit consequences of y = a AND b under the given signed literals.
    /// None denotes contradiction. An unknown value establishes no feasibility
    /// claim.
    fn propagate(&self, assumptions: &[AigOperand]) -> Option<Vec<Option<bool>>> {
        let mut facts = vec![None; self.graph.gates.len()];
        let mut pending = self.constants.clone();
        pending.extend_from_slice(assumptions);
        while let Some(lit) = pending.pop() {
            let id = lit.node.id;
            if let Some(value) = facts[id] {
                if value == lit.negated {
                    return None;
                }
                continue;
            }
            facts[id] = Some(!lit.negated);
            for gate in std::iter::once(id).chain(self.users[id].iter().copied()) {
                let AigNode::And2 { a, b, .. } = &self.graph.gates[gate] else {
                    continue;
                };
                let av = signed_value(&facts, *a);
                let bv = signed_value(&facts, *b);
                let value = match (av, bv) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                };
                if let Some(value) = value {
                    pending.push(AigOperand {
                        node: crate::aig::AigRef { id: gate },
                        negated: !value,
                    });
                }
                match facts[gate] {
                    Some(true) => pending.extend([*a, *b]),
                    Some(false) if a == b || bv == Some(true) => pending.push(a.negate()),
                    Some(false) if av == Some(true) => pending.push(b.negate()),
                    _ => {}
                }
            }
        }
        Some(facts)
    }

    /// A false input forced by the *whole endpoint* suffices to prove a negated
    /// AND.
    fn expand(&self, required: &[AigOperand], facts: &[Option<bool>]) -> Vec<AigOperand> {
        let mut pending = required.to_vec();
        let mut visited = HashSet::new();
        let mut atoms = Vec::new();
        while let Some(lit) = pending.pop() {
            if !visited.insert(lit) {
                continue;
            }
            match self.graph.get(lit.node) {
                AigNode::Literal { .. } => {}
                AigNode::And2 { a, b, .. } if !lit.negated => pending.extend([*a, *b]),
                AigNode::And2 { a, b, .. } => {
                    let chosen = [a.negate(), b.negate()]
                        .into_iter()
                        .filter(|&input| signed_value(facts, input) == Some(true))
                        .min_by_key(|input| (self.depth[input.node.id], *input));
                    if let Some(input) = chosen {
                        pending.push(input);
                    } else {
                        atoms.push(lit);
                    }
                }
                AigNode::Input { .. } => atoms.push(lit),
            }
        }
        atoms
    }

    fn check(
        &self,
        required: &[AigOperand],
        facts: &[Option<bool>],
        atoms: Vec<AigOperand>,
    ) -> Cut {
        if !atoms.iter().all(|&a| signed_value(facts, a) == Some(true)) {
            return Cut::Unchanged;
        }
        let original: HashSet<_> = required.iter().copied().collect();
        if original.len() == atoms.len() && atoms.iter().all(|a| original.contains(a)) {
            return Cut::Unchanged;
        }
        let Some(reverse) = self.propagate(&atoms) else {
            return Cut::Unchanged;
        };
        if required
            .iter()
            .all(|&bit| signed_value(&reverse, bit) == Some(true))
        {
            Cut::Literals(atoms)
        } else {
            Cut::Unchanged
        }
    }
}

fn signed_value(facts: &[Option<bool>], bit: AigOperand) -> Option<bool> {
    facts[bit.node.id].map(|v| v ^ bit.negated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{GateBuilder, GateBuilderOptions};
    use crate::aig_sim::gate_sim::{Collect, eval};
    use xlsynth_pir::IrBits;

    #[test]
    fn positive_and_and_forced_negative_fanin_preserve_both_phases() {
        let mut builder = GateBuilder::new("signed".into(), GateBuilderOptions::opt());
        let x = builder.add_input("x".into(), 2);
        let (a, b) = (*x.get_lsb(0), *x.get_lsb(1));
        let ab = builder.add_and_binary(a, b);
        let w = AigBitVector::from_lsb_is_index_0(&[a.negate(), ab.negate(), a.negate()]);
        builder.add_output("w".into(), w.clone());
        let [zero, ones] = analyze(&builder.build(), &w);
        let Cut::Literals(zero) = zero else {
            panic!("expected both positive inputs")
        };
        assert_eq!(zero.len(), 2);
        assert!(zero.contains(&a) && zero.contains(&b));
        assert_eq!(ones, Cut::Literals(vec![a.negate()]));
    }

    #[test]
    fn signed_aliases_and_constant_endpoints_can_be_impossible() {
        let mut builder = GateBuilder::new("conflict".into(), GateBuilderOptions::opt());
        let x = builder.add_input("x".into(), 2);
        let (a, b) = (*x.get_lsb(0), *x.get_lsb(1));
        let ab = builder.add_and_binary(a, b);
        let false_bit = builder.get_false();
        builder.add_output("out".into(), AigBitVector::from_bit(ab));
        let graph = builder.build();
        let classify = |w: &[AigOperand]| analyze(&graph, &AigBitVector::from_lsb_is_index_0(w));
        assert_eq!(
            classify(&[a, a.negate()]),
            [Cut::Impossible, Cut::Impossible]
        );
        assert_eq!(
            classify(&[false_bit, false_bit]),
            [Cut::Literals(vec![]), Cut::Impossible]
        );
        assert_eq!(classify(&[a, a]), [Cut::Unchanged, Cut::Unchanged]);
        assert_eq!(classify(&[ab, b.negate()])[1], Cut::Impossible);
    }

    #[test]
    fn reverse_proof_rejects_an_insufficient_necessary_condition() {
        let mut builder = GateBuilder::new("reverse".into(), GateBuilderOptions::opt());
        let x = builder.add_input("x".into(), 2);
        let (a, b) = (*x.get_lsb(0), *x.get_lsb(1));
        let ab = builder.add_and_binary(a, b.negate());
        builder.add_output("out".into(), AigBitVector::from_bit(ab));
        let graph = builder.build();
        let cone = WordCone::new(&graph, &[ab]);
        let facts = cone.propagate(&[ab]).unwrap();
        assert_eq!(cone.check(&[ab], &facts, vec![a]), Cut::Unchanged);
        assert_eq!(cone.check(&[ab], &facts, vec![b]), Cut::Unchanged);
    }

    #[test]
    fn reconvergent_mux_endpoint_cuts_match_every_original_input() {
        let mut builder = GateBuilder::new("reconvergence".into(), GateBuilderOptions::opt());
        let x = builder.add_input("x".into(), 4);
        let (a, b, c, s) = (*x.get_lsb(0), *x.get_lsb(1), *x.get_lsb(2), *x.get_lsb(3));
        let mux = builder.add_mux2(s, a, b.negate());
        let par = builder.add_xor_binary(a, c);
        let together = builder.add_and_binary(mux, par);
        let w = AigBitVector::from_lsb_is_index_0(&[together.negate(), par, s, mux.negate()]);
        builder.add_output("w".into(), w.clone());
        let graph = builder.build();
        let cuts = analyze(&graph, &w);
        assert!(cuts.iter().any(|c| matches!(c, Cut::Literals(_))));
        for n in 0..16 {
            let result = eval(
                &graph,
                &[IrBits::make_ubits(4, n).unwrap()],
                Collect::AllWithInputs,
            );
            let vals = result.all_values.unwrap();
            let get = |lit: AigOperand| vals[lit.node.id] ^ lit.negated;
            for (i, cut) in cuts.iter().enumerate() {
                let expected = w.iter_lsb_to_msb().all(|bit| get(*bit) == (i == 1));
                let actual = match cut {
                    Cut::Impossible => false,
                    Cut::Unchanged => expected,
                    Cut::Literals(atoms) => atoms.iter().all(|a| get(*a)),
                };
                assert_eq!(actual, expected);
            }
        }
    }
}
