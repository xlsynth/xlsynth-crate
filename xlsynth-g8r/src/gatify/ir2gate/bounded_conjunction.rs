// SPDX-License-Identifier: Apache-2.0

//! Bounded satisfiability of a conjunction of existing AIG operands.
//! A query is local to the structural graph and never constrains the builder.

use crate::aig::gate::{AigNode, AigOperand};
use crate::gate_builder::GateBuilder;
use std::collections::{HashMap, HashSet};

const MAX_CONE_NODES: usize = 4096;
const MAX_VISITS_PER_FUNCTION: usize = 1_000_000;
const MAX_SOLVER_VARIABLES: usize = 32_768;
const MAX_QUERIES_PER_FUNCTION: usize = 1024;
const SMALL_SUPPORT: usize = 7;
const CONFLICTS_PER_QUERY: i32 = 4;

#[derive(Default)]
pub(super) struct Facts {
    sat: Option<Interner>,
    pub(super) visits: usize,
    pub(super) queries: usize,
}

impl Facts {
    /// Refine an already sound upper bound. Adding more operands only
    /// restricts satisfying assignments, so prefix feasibility is monotone.
    pub(super) fn shortest_impossible_prefix(
        &mut self,
        gb: &GateBuilder,
        guards: &[AigOperand],
        bits: &[AigOperand],
        wire_floor: usize,
        mut best: usize,
    ) -> (usize, &'static str, &'static str, usize) {
        let mut retries = 0;
        let mut strategy = "bounds";
        let mut reason = "no-bound";
        if best > wire_floor && best - 1 + guards.len() > MAX_CONE_NODES {
            reason = "cone-limit";
        } else if best > wire_floor
            && self.visits < MAX_VISITS_PER_FUNCTION
            && self.queries < MAX_QUERIES_PER_FUNCTION
        {
            let mut roots = guards.to_vec();
            roots.extend_from_slice(&bits[..best - 1]);
            match Cone::new(&roots, gb, self) {
                Some(cone) => {
                    let encoded = if cone.inputs.len() > SMALL_SUPPORT {
                        self.sat.get_or_insert_with(Interner::new).encode(&cone, gb)
                    } else {
                        None
                    };
                    let mut decision = |k: usize| {
                        if self.queries >= MAX_QUERIES_PER_FUNCTION {
                            return Decision::Unknown;
                        }
                        let mut inputs = guards.to_vec();
                        inputs.extend_from_slice(&bits[..k]);
                        self.queries += 1;
                        if cone.inputs.len() <= SMALL_SUPPORT {
                            strategy = "truth";
                            cone.truth(&inputs, gb)
                        } else {
                            strategy = "sat";
                            let Some(lits) = encoded.as_ref() else {
                                return Decision::Unknown;
                            };
                            let solver = self.sat.as_mut().unwrap();
                            let first = solver.query(lits, &inputs, CONFLICTS_PER_QUERY);
                            if first == Decision::Unknown && self.queries < MAX_QUERIES_PER_FUNCTION
                            {
                                retries += 1;
                                self.queries += 1;
                                solver.query(lits, &inputs, 32)
                            } else {
                                first
                            }
                        }
                    };
                    match decision(best - 1) {
                        Decision::Impossible => {
                            best -= 1;
                            let mut low = wire_floor;
                            while low < best {
                                let mid = low + (best - low) / 2;
                                match decision(mid) {
                                    Decision::Impossible => best = mid,
                                    Decision::Possible => low = mid + 1,
                                    Decision::Unknown => break,
                                }
                            }
                            reason = "proved";
                        }
                        Decision::Possible => reason = "reachable",
                        Decision::Unknown => reason = "unknown",
                    }
                }
                None => reason = "cone-limit",
            }
        }
        (best, strategy, reason, retries)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Decision {
    Impossible,
    Possible,
    Unknown,
}

struct Cone {
    order: Vec<usize>,
    inputs: Vec<usize>,
}
impl Cone {
    fn new(roots: &[AigOperand], g: &GateBuilder, facts: &mut Facts) -> Option<Self> {
        let mut seen = HashSet::new();
        let mut todo: Vec<_> = roots.iter().map(|r| r.node.id).collect();
        let mut pins = vec![];
        while let Some(n) = todo.pop() {
            if !seen.insert(n) {
                continue;
            }
            facts.visits += 1;
            if seen.len() > MAX_CONE_NODES || facts.visits > MAX_VISITS_PER_FUNCTION {
                return None;
            }
            match g.gates[n] {
                AigNode::And2 { a, b, .. } => {
                    todo.push(a.node.id);
                    todo.push(b.node.id);
                }
                AigNode::Input { .. } => pins.push(n),
                AigNode::Literal { .. } => {}
            }
        }
        let mut order: Vec<_> = seen.into_iter().collect();
        order.sort_unstable();
        pins.sort_unstable();
        Some(Self {
            order,
            inputs: pins,
        })
    }

    fn truth(&self, required: &[AigOperand], g: &GateBuilder) -> Decision {
        // Lanes enumerate every assignment to the actual input pins. Repeated
        // lanes on smaller tables cannot turn a satisfiable conjunction into
        // false.
        for first in (0..(1usize << self.inputs.len())).step_by(64) {
            let mut values = HashMap::new();
            let literal = |r: AigOperand, values: &HashMap<usize, u64>| {
                let v = values[&r.node.id];
                if r.negated { !v } else { v }
            };
            for &n in &self.order {
                let value = match g.gates[n] {
                    AigNode::Literal { value, .. } => {
                        if value {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    AigNode::Input { .. } => {
                        let p = self.inputs.binary_search(&n).unwrap();
                        (0..64).fold(0, |v, i| v | ((((first + i) >> p) & 1) as u64) << i)
                    }
                    AigNode::And2 { a, b, .. } => literal(a, &values) & literal(b, &values),
                };
                values.insert(n, value);
            }
            if required
                .iter()
                .fold(u64::MAX, |v, &r| v & literal(r, &values))
                != 0
            {
                return Decision::Possible;
            }
        }
        Decision::Impossible
    }
}

#[derive(Hash, PartialEq, Eq)]
enum Key {
    Input(usize),
    And(i32, i32),
}

struct Interner {
    solver: cadical::Solver,
    vars: HashMap<Key, i32>,
}
impl Interner {
    fn new() -> Self {
        let mut solver = cadical::Solver::new();
        solver.add_clause([1]);
        Self {
            solver,
            vars: HashMap::new(),
        }
    }
    fn encode(&mut self, cone: &Cone, g: &GateBuilder) -> Option<HashMap<usize, i32>> {
        let mut local = HashMap::new();
        let signed = |r: AigOperand, v: &HashMap<usize, i32>| {
            let t = v[&r.node.id];
            if r.negated { -t } else { t }
        };
        for &n in &cone.order {
            let key = match g.gates[n] {
                AigNode::Literal { value, .. } => {
                    local.insert(n, if value { 1 } else { -1 });
                    continue;
                }
                AigNode::Input { .. } => Key::Input(n),
                AigNode::And2 { a, b, .. } => {
                    let (a, b) = (signed(a, &local), signed(b, &local));
                    Key::And(a.min(b), a.max(b))
                }
            };
            let var = if let Some(v) = self.vars.get(&key) {
                *v
            } else {
                if self.vars.len() >= MAX_SOLVER_VARIABLES {
                    return None;
                }
                let v = self.vars.len() as i32 + 2;
                if let Key::And(a, b) = key {
                    self.solver.add_clause([-v, a]);
                    self.solver.add_clause([-v, b]);
                    self.solver.add_clause([v, -a, -b]);
                }
                self.vars.insert(key, v);
                v
            };
            local.insert(n, var);
        }
        // Content-keyed ANDs permit gate-builder trial rollback and ID reuse:
        // cached equations only define fresh variables, never constrain a PI.
        Some(local)
    }
    fn query(
        &mut self,
        local: &HashMap<usize, i32>,
        required: &[AigOperand],
        limit: i32,
    ) -> Decision {
        let signed = |r: AigOperand| {
            let v = local[&r.node.id];
            if r.negated { -v } else { v }
        };
        if self.solver.set_limit("conflicts", limit).is_err() {
            return Decision::Unknown;
        }
        match self.solver.solve_with(required.iter().map(|r| signed(*r))) {
            Some(false) => Decision::Impossible,
            Some(true) => Decision::Possible,
            None => Decision::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::GateBuilderOptions;

    #[test]
    fn rolled_back_and_id_must_not_reuse_a_stale_sat_equation() {
        let mut gb = GateBuilder::new(
            "query".into(),
            GateBuilderOptions {
                fold: true,
                hash: true,
            },
        );
        let x = gb.add_input("x".into(), 10);
        let inputs: Vec<_> = x.iter_lsb_to_msb().copied().collect();
        let mut facts = Facts::default();
        let checkpoint = gb.begin_append_checkpoint();
        let before = gb.add_and_binary(inputs[0].negate(), inputs[9]);
        assert_eq!(
            facts
                .shortest_impossible_prefix(&gb, &[before], &inputs, 1, 10)
                .0,
            1
        );
        gb.rollback_append_checkpoint(checkpoint);
        let after = gb.add_and_binary(inputs[0], inputs[9]);
        assert_eq!(before.node, after.node);
        assert_eq!(
            facts
                .shortest_impossible_prefix(&gb, &[after], &inputs, 1, 10)
                .0,
            10
        );
    }
}
