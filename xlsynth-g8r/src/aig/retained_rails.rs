// SPDX-License-Identifier: Apache-2.0

//! Reidentify a live signed operand after rewriting the same function.
//! Witnesses use formal input coordinates, never gate IDs from the earlier
//! graph.

use std::collections::HashMap;

use super::gate::{AigNode, AigOperand, AigRef, GateFn};
mod sweep;
mod truth;
pub(crate) use sweep::coalesce_cut_boundary;
pub(crate) use truth::{CutBit, CutIndex, CutSignature};

/// Limit aggregate capture, index and candidate work across a function's
/// clients.
#[derive(Clone)]
pub(crate) struct RailBudget(usize);

impl Default for RailBudget {
    fn default() -> Self {
        Self(32 * 1024)
    }
}

impl RailBudget {
    pub(crate) fn take(&mut self, work: usize) -> bool {
        if work > self.0 {
            return false;
        }
        self.0 -= work;
        true
    }

    fn take_graph(&mut self, f: &GateFn) -> bool {
        if f.gates.len() > self.0 || f.outputs.len() > self.0 {
            return false;
        }
        let work = f.outputs.iter().fold(f.gates.len(), |work, port| {
            work.saturating_add(port.get_bit_count())
        });
        self.take(work)
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
struct Edge {
    class: usize,
    inverted: bool,
}

#[derive(Hash, Eq, PartialEq)]
enum Key {
    Literal(bool),
    Input(CutBit),
    And([Edge; 2]),
}

pub(crate) struct SignedCone {
    function: String,
    nodes: Vec<Key>,
    root: Edge,
}

impl SignedCone {
    pub(crate) fn capture(f: &GateFn, root: AigOperand, budget: &mut RailBudget) -> Option<Self> {
        // Walk only this operand. Reserve work before descending into a large
        // cone.
        if !budget.take(f.inputs.len()) {
            return None;
        }
        let formals = Formals::new(f)?;
        let mut source = HashMap::new();
        let mut nodes = Vec::new();
        let mut pending = vec![(root.node, false)];
        while let Some((node, ready)) = pending.pop() {
            if source.contains_key(&node) {
                continue;
            }
            if !budget.take(1) {
                return None;
            }
            if let AigNode::And2 { a, b, .. } = &f.gates.get(node.id)? {
                if !ready {
                    pending.extend([(node, true), (a.node, false), (b.node, false)]);
                    continue;
                }
            }
            let key = formals.key(f, node, |arg| source[&arg])?;
            source.insert(node, nodes.len());
            nodes.push(key);
        }
        Some(Self {
            function: f.name.clone(),
            root: Edge {
                class: source[&root.node],
                inverted: root.negated,
            },
            nodes,
        })
    }
}

struct Formals<'a>(HashMap<&'a str, &'a super::gate::Input>);

impl<'a> Formals<'a> {
    fn new(f: &'a GateFn) -> Option<Self> {
        let mut formals = HashMap::with_capacity(f.inputs.len());
        for p in &f.inputs {
            if formals.insert(p.name.as_str(), p).is_some() {
                return None;
            }
        }
        Some(Self(formals))
    }
    fn key(&self, f: &GateFn, node: AigRef, class: impl Fn(AigRef) -> usize) -> Option<Key> {
        match &f.gates[node.id] {
            AigNode::Literal { value, .. } => Some(Key::Literal(*value)),
            AigNode::Input {
                name, lsb_index, ..
            } => {
                let formal = self.0.get(name.as_str())?;
                let width = formal.get_bit_count();
                if *lsb_index >= width
                    || *formal.bit_vector.get_lsb(*lsb_index)
                        != (AigOperand {
                            node,
                            negated: false,
                        })
                {
                    return None;
                }
                Some(Key::Input(CutBit {
                    port: name.clone(),
                    width,
                    bit: *lsb_index,
                }))
            }
            AigNode::And2 { a, b, .. } => {
                let mut pair = [a, b].map(|e| Edge {
                    class: class(e.node),
                    inverted: e.negated,
                });
                pair.sort_unstable();
                Some(Key::And(pair))
            }
        }
    }
}

pub(crate) struct RailResolver<'a> {
    graph: &'a GateFn,
    structure: Option<Structure>,
    // Keep optional regrouping from exhausting existing structural and truth
    // lookups.
    conjunction_budget: RailBudget,
    cuts: HashMap<Vec<CutBit>, CutIndex>,
}

impl<'a> RailResolver<'a> {
    pub(crate) fn new(graph: &'a GateFn) -> Self {
        Self {
            graph,
            structure: None,
            conjunction_budget: RailBudget::default(),
            cuts: HashMap::new(),
        }
    }

    /// Resolve by truth table, exact structure, or regrouping of signed input
    /// literals.
    pub(crate) fn find(
        &mut self,
        cone: &SignedCone,
        budget: &mut RailBudget,
    ) -> Option<AigOperand> {
        if cone.function != self.graph.name || !budget.take(cone.nodes.len()) {
            return None;
        }
        if let Some((bits, rows)) = cone.cut_rows() {
            if !self.cuts.contains_key(&bits) {
                self.cuts
                    .insert(bits.clone(), CutIndex::build(self.graph, &bits, budget)?);
            }
            // The client still checks the depth of every output and the total
            // live area. Different copies of one Boolean function can differ in
            // both.
            return self.cuts[&bits].resolve(&CutSignature::from_rows(
                &cone.function,
                &bits,
                rows,
            )?);
        }
        if self.structure.is_none() {
            self.structure = Some(Structure::build(self.graph, budget)?);
        }
        let structure = self.structure.as_ref()?;
        structure
            .find(cone)
            .or_else(|| structure.find_conjunction(self.graph, cone, &mut self.conjunction_budget))
    }
}

struct Structure {
    classes: HashMap<Key, usize>,
    representatives: Vec<AigRef>,
}

impl Structure {
    fn build(f: &GateFn, budget: &mut RailBudget) -> Option<Self> {
        if !budget.take_graph(f) || !budget.take(f.inputs.len()) {
            return None;
        }
        let formals = Formals::new(f)?;
        let mut classes = HashMap::new();
        let mut representatives = Vec::new();
        let mut ids = HashMap::new();
        for edge in f.post_order_operands(false) {
            if ids.contains_key(&edge.node) {
                continue;
            }
            let key = formals.key(f, edge.node, |arg| ids[&arg])?;
            let class = *classes.entry(key).or_insert_with(|| {
                representatives.push(edge.node);
                representatives.len() - 1
            });
            if edge.node.id < representatives[class].id {
                representatives[class] = edge.node;
            }
            ids.insert(edge.node, class);
        }
        Some(Self {
            classes,
            representatives,
        })
    }

    fn find(&self, cone: &SignedCone) -> Option<AigOperand> {
        let mut found = Vec::with_capacity(cone.nodes.len());
        for node in &cone.nodes {
            let class = match node {
                Key::And(pair) => {
                    let mut remapped = pair.map(|e| Edge {
                        class: found[e.class],
                        inverted: e.inverted,
                    });
                    remapped.sort_unstable();
                    self.classes.get(&Key::And(remapped))
                }
                leaf => self.classes.get(leaf),
            }?;
            found.push(*class);
        }
        Some(AigOperand {
            node: self.representatives[found[cone.root.class]],
            negated: cone.root.inverted,
        })
    }
    /// Match an AND of signed primary inputs even if reassociation changed its
    /// internal gates. An inverted internal AND is a boundary. Repeated inputs
    /// are idempotent. Prefer the shallowest proven match, then the lowest ID.
    /// No candidate graph is constructed here.
    fn find_conjunction(
        &self,
        graph: &GateFn,
        cone: &SignedCone,
        budget: &mut RailBudget,
    ) -> Option<AigOperand> {
        use smallvec::SmallVec;

        if !budget.take(cone.nodes.len()) {
            return None;
        }
        let mut pins = HashMap::new();
        for node in &cone.nodes {
            let Key::And(children) = node else { continue };
            for child in children {
                match &cone.nodes[child.class] {
                    Key::And(_) if !child.inverted => {}
                    input @ Key::Input(_) => {
                        let pin = AigOperand {
                            node: self.representatives[*self.classes.get(input)?],
                            negated: child.inverted,
                        };
                        if pins.contains_key(&pin.negate()) {
                            return None;
                        }
                        let next = pins.len();
                        pins.entry(pin).or_insert(next);
                    }
                    _ => return None,
                }
            }
        }
        if pins.is_empty() || !budget.take_graph(graph) {
            return None;
        }
        let words = pins.len().div_ceil(u64::BITS as usize);
        if !budget.take(words.checked_mul(pins.len())?) {
            return None;
        }
        type Coverage = SmallVec<[u64; 1]>;
        let mut full = Coverage::from_elem(0, words);
        let mut coverage = HashMap::new();
        for (pin, index) in pins {
            let mut singleton = Coverage::from_elem(0, words);
            let bit = 1u64 << (index % 64);
            singleton[index / 64] = bit;
            full[index / 64] |= bit;
            coverage.insert(pin, (singleton, 0usize));
        }
        let mut best = None;
        for node in graph.post_order_refs() {
            let AigNode::And2 { a, b, .. } = &graph.gates[node.id] else {
                continue;
            };
            let (Some((left, left_depth)), Some((right, right_depth))) =
                (coverage.get(a), coverage.get(b))
            else {
                continue;
            };
            if !budget.take(words) {
                // A later merge cannot invalidate a match already proved.
                break;
            }
            let merged: Coverage = left.iter().zip(right).map(|(x, y)| x | y).collect();
            let depth = 1 + left_depth.max(right_depth);
            if merged == full && best.is_none_or(|old| (depth, node) < old) {
                best = Some((depth, node));
            }
            coverage.insert(
                AigOperand {
                    node,
                    negated: false,
                },
                (merged, depth),
            );
        }
        best.map(|(_, node)| AigOperand {
            node,
            negated: cone.root.inverted,
        })
    }
}

#[cfg(test)]
mod tests;
