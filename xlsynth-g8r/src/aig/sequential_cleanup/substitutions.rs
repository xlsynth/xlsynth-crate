// SPDX-License-Identifier: Apache-2.0

//! Incremental congruence propagation after sequential state substitutions.

use std::collections::{BTreeMap, VecDeque};

use super::{StateAction, StateBit};
use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};

/// An equation to revisit when one of its operands changes representative.
#[derive(Clone, Copy)]
enum User {
    And(usize),
    Register(usize),
}

/// Signed equivalence classes plus the equations using each class.
struct Aliases {
    parents: Vec<AigOperand>,
    sizes: Vec<usize>,
    users: Vec<Vec<User>>,
    pending: VecDeque<User>,
    queued: Vec<bool>,
}

impl Aliases {
    fn new(node_count: usize, register_count: usize) -> Self {
        Self {
            parents: (0..node_count).map(|id| AigRef { id }.into()).collect(),
            sizes: vec![1; node_count],
            users: vec![Vec::new(); node_count],
            pending: VecDeque::new(),
            queued: vec![false; node_count + register_count],
        }
    }

    /// Resolves an operand with phase-preserving path compression.
    fn resolve(&mut self, operand: AigOperand) -> AigOperand {
        let parent = self.parents[operand.node.id];
        if parent.node == operand.node {
            return operand;
        }
        let root = self.resolve(parent);
        self.parents[operand.node.id] = root;
        AigOperand {
            negated: root.negated ^ operand.negated,
            ..root
        }
    }

    fn user_index(&self, user: User) -> usize {
        match user {
            User::And(id) => id,
            User::Register(index) => self.parents.len() + index,
        }
    }

    /// Coalesces repeated notifications while an equation is pending.
    fn schedule(&mut self, user: User) {
        let index = self.user_index(user);
        if !self.queued[index] {
            self.queued[index] = true;
            self.pending.push_back(user);
        }
    }

    fn pop(&mut self) -> Option<User> {
        let user = self.pending.pop_front()?;
        let index = self.user_index(user);
        self.queued[index] = false;
        Some(user)
    }

    /// Unions by size, except that the constant class always keeps node zero.
    /// Each dependency moves logarithmically often, plus at most one final move
    /// to the constant class.
    fn equate(&mut self, a: AigOperand, b: AigOperand) {
        let mut a = self.resolve(a);
        let mut b = self.resolve(b);
        if a.node == b.node {
            assert_eq!(a.negated, b.negated, "inconsistent sequential substitution");
            return;
        }
        // Becoming constant must change the representative of the nonconstant
        // class, even if it is larger, so all of its users are reconsidered.
        if b.node.id == 0
            || (a.node.id != 0
                && (self.sizes[a.node.id] < self.sizes[b.node.id]
                    || (self.sizes[a.node.id] == self.sizes[b.node.id] && a.node > b.node)))
        {
            std::mem::swap(&mut a, &mut b);
        }
        self.parents[b.node.id] = AigOperand {
            node: a.node,
            negated: a.negated ^ b.negated,
        };
        self.sizes[a.node.id] += self.sizes[b.node.id];
        let moved = std::mem::take(&mut self.users[b.node.id]);
        for &user in &moved {
            self.schedule(user);
        }
        self.users[a.node.id].extend(moved);
    }
}

/// A commutative AND signature, including input phases.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct AndKey(AigOperand, AigOperand);

impl AndKey {
    fn new(a: AigOperand, b: AigOperand) -> Self {
        Self(a.min(b), a.max(b))
    }
}

#[derive(Clone, Copy)]
struct RegisterMatch {
    index: usize,
    phase: bool,
}

/// Folds an affected AND under current aliases, without rewriting the graph.
fn fold_and(a: AigOperand, b: AigOperand, zero: AigOperand) -> Option<AigOperand> {
    if a == b {
        Some(a)
    } else if a == b.negate() || a == zero || b == zero {
        Some(zero)
    } else if a == zero.negate() {
        Some(b)
    } else if b == zero.negate() {
        Some(a)
    } else {
        None
    }
}

/// Batches duplicate/constant consequences of state substitutions. The returned
/// equation-visit count supports deterministic scaling tests.
pub(super) fn mark_duplicate_state_bits(
    graph: &GateFn,
    bits: &[StateBit],
    actions: &mut [StateAction],
) -> usize {
    if !actions.contains(&StateAction::Keep) {
        return 0;
    }
    let mut aliases = Aliases::new(graph.gates.len(), bits.len());
    let mut and_keys = vec![None; graph.gates.len()];
    let mut and_matches = BTreeMap::new();
    for (id, node) in graph.gates.iter().enumerate() {
        if let AigNode::And2 { a, b, .. } = node {
            aliases.users[a.node.id].push(User::And(id));
            aliases.users[b.node.id].push(User::And(id));
            let key = AndKey::new(*a, *b);
            and_keys[id] = Some(key);
            // Seed lookup only: unmodified cones are not normalized here.
            and_matches.entry(key).or_insert(id);
        }
    }
    for (index, bit) in bits.iter().enumerate() {
        if actions[index] == StateAction::Keep {
            aliases.users[bit.d.node.id].push(User::Register(index));
            aliases.schedule(User::Register(index));
        }
    }
    let zero: AigOperand = AigRef { id: 0 }.into();
    for (id, node) in graph.gates.iter().enumerate() {
        if let AigNode::Literal { value, .. } = node {
            aliases.equate(
                AigRef { id }.into(),
                AigOperand {
                    negated: *value,
                    ..zero
                },
            );
        }
    }
    for (index, bit) in bits.iter().enumerate() {
        if let StateAction::Constant(value) = actions[index] {
            aliases.equate(
                bit.q,
                AigOperand {
                    negated: value,
                    ..zero
                },
            );
        }
    }

    let mut register_keys = vec![None; bits.len()];
    let mut register_matches: BTreeMap<usize, RegisterMatch> = BTreeMap::new();
    let mut visits = 0;
    while let Some(user) = aliases.pop() {
        visits += 1;
        match user {
            User::And(id) => {
                if let Some(key) = and_keys[id].take()
                    && and_matches.get(&key) == Some(&id)
                {
                    and_matches.remove(&key);
                }
                let AigNode::And2 { a, b, .. } = graph.gates[id] else {
                    unreachable!("AND dependency references an AND");
                };
                let a = aliases.resolve(a);
                let b = aliases.resolve(b);
                let constant = aliases.resolve(zero);
                if let Some(value) = fold_and(a, b, constant) {
                    aliases.equate(AigRef { id }.into(), value);
                } else {
                    let key = AndKey::new(a, b);
                    if let Some(&other) = and_matches.get(&key) {
                        aliases.equate(AigRef { id }.into(), AigRef { id: other }.into());
                    } else {
                        and_matches.insert(key, id);
                    }
                    and_keys[id] = Some(key);
                }
            }
            User::Register(index) => {
                if let Some(key) = register_keys[index].take()
                    && register_matches.get(&key).is_some_and(|m| m.index == index)
                {
                    register_matches.remove(&key);
                }
                let d = aliases.resolve(bits[index].d);
                let constant = aliases.resolve(zero);
                if d.node == constant.node {
                    aliases.equate(bits[index].q, d);
                } else {
                    if let Some(other) = register_matches.get(&d.node.id) {
                        let q = bits[other.index].q;
                        aliases.equate(
                            bits[index].q,
                            AigOperand {
                                negated: q.negated ^ d.negated ^ other.phase,
                                ..q
                            },
                        );
                    } else {
                        register_matches.insert(
                            d.node.id,
                            RegisterMatch {
                                index,
                                phase: d.negated,
                            },
                        );
                    }
                    register_keys[index] = Some(d.node.id);
                }
            }
        }
    }

    // Pick the first original state bit as the physical representative, even
    // when union-by-size selected another node as its internal class root.
    let mut representatives: BTreeMap<usize, RegisterMatch> = BTreeMap::new();
    let constant = aliases.resolve(zero);
    for (index, bit) in bits.iter().enumerate() {
        if actions[index] == StateAction::Dead {
            continue;
        }
        let q = aliases.resolve(bit.q);
        actions[index] = if q.node == constant.node {
            StateAction::Constant(q.negated ^ constant.negated)
        } else if let Some(other) = representatives.get(&q.node.id) {
            StateAction::Duplicate {
                representative: other.index,
                negated: q.negated ^ other.phase,
            }
        } else {
            representatives.insert(
                q.node.id,
                RegisterMatch {
                    index,
                    phase: q.negated,
                },
            );
            StateAction::Keep
        };
    }
    visits
}
