// SPDX-License-Identifier: Apache-2.0

//! Graph-based logical effort worst-case delay estimation.

use crate::aig::topo::{postorder_for_aig_refs_node_only, topo_sort_refs};
use crate::aig::{AigNode, AigRef, GateFn};
use std::collections::HashMap;

#[derive(Copy, Clone, Debug)]
struct State {
    log_f: f64,
    n: usize,
    p: f64,
    prev: Option<(AigRef, f64, usize, f64)>,
}

fn state_delay(state: &State) -> f64 {
    if state.n == 0 {
        0.0
    } else {
        (state.n as f64) * ((state.log_f / (state.n as f64)).exp()) + state.p
    }
}

fn dominates(o: &State, c: &State) -> bool {
    let (of, on, op) = (o.log_f, o.n as f64, o.p);
    let (cf, cn, cp) = (c.log_f, c.n as f64, c.p);
    let left = on <= of && cn <= cf;
    let right = on >= of && cn >= cf;
    if left {
        return on <= cn && of >= cf && op >= cp;
    }
    if right {
        return on >= cn && of >= cf && op >= cp;
    }
    false
}

/// Computes the worst-case delay in a DAG using logical effort analysis.
///
/// - `dag` maps each node to a list of outgoing edges `(v, g, p)` where `g` is
///   the logical effort and `p` is the parasitic delay of the edge.
/// - `pin_load` is a function computing the load `h` for edge `(u, v)`.
///
/// Returns a tuple `(path, delay)` where `path` is the sequence of nodes
/// and `delay` is the worst-case delay value.
#[allow(non_snake_case)]
fn worst_case_delay<F>(
    dag: &HashMap<AigRef, Vec<(AigRef, f64, f64)>>,
    pin_load: F,
    gate_nodes: &[AigNode],
) -> (Vec<AigRef>, f64)
where
    F: Fn(AigRef, AigRef) -> f64,
{
    // global constants
    let g_max = dag
        .values()
        .flat_map(|edges| edges.iter().map(|&(_, g, _)| g))
        .fold(0.0_f64, |a, b| a.max(b));
    let p_max_global = dag
        .values()
        .flat_map(|edges| edges.iter().map(|&(_, _, p)| p))
        .fold(0.0_f64, |a, b| a.max(b));
    let mut h_max = 0.0_f64;
    for (&u, edges) in dag.iter() {
        for &(v, _, _) in edges {
            let h = pin_load(u, v);
            if h > h_max {
                h_max = h;
            }
        }
    }
    let log_gh_max = (g_max * h_max).ln();

    // Use topo_sort_refs for topological order
    let topo: Vec<AigRef> = topo_sort_refs(gate_nodes);

    // compute longest path R in reverse topological order
    let mut R: HashMap<AigRef, usize> = HashMap::new();
    for &u in topo.iter().rev() {
        let mut max_r = 0;
        if let Some(edges) = dag.get(&u) {
            for &(v, _, _) in edges {
                let rv = *R.get(&v).unwrap_or(&0);
                max_r = max_r.max(rv + 1);
            }
        }
        R.insert(u, max_r);
    }

    // frontier per node
    let mut S: HashMap<AigRef, Vec<State>> = HashMap::new();
    let mut best_delay = -1.0_f64;
    let mut best_state: Option<(AigRef, State)> = None;

    for &u in topo.iter() {
        // initialize the frontier
        if S.get(&u).map_or(true, |v| v.is_empty()) {
            S.insert(
                u,
                vec![State {
                    log_f: 0.0,
                    n: 0,
                    p: 0.0,
                    prev: None,
                }],
            );
        }
        // propagate to successors
        if let Some(edges) = dag.get(&u) {
            for &(v, g, p) in edges {
                let h = pin_load(u, v);
                let w = g.ln() + h.ln();
                let current_states = S.get(&u).unwrap().clone();
                for state in current_states {
                    let cand_log_f = state.log_f + w;
                    let cand_n = state.n + 1;
                    let cand_p = state.p + p;
                    let cand = State {
                        log_f: cand_log_f,
                        n: cand_n,
                        p: cand_p,
                        prev: Some((u, state.log_f, state.n, state.p)),
                    };
                    // global pruning
                    let r_left = *R.get(&v).unwrap_or(&0);
                    let n_max = cand_n as f64 + r_left as f64;
                    let log_f_max = cand_log_f + (r_left as f64) * log_gh_max;
                    let p_max = cand_p + (r_left as f64) * p_max_global;
                    let upper = n_max * ((log_f_max / n_max).exp()) + p_max;
                    if upper <= best_delay {
                        continue;
                    }
                    let out = S.entry(v).or_insert_with(Vec::new);
                    // local Pareto pruning
                    let mut keep = true;
                    for o in out.iter() {
                        if dominates(o, &cand)
                            || (o.n == cand_n && o.log_f >= cand_log_f && o.p >= cand_p)
                        {
                            keep = false;
                            break;
                        }
                    }
                    if !keep {
                        continue;
                    }
                    out.retain(|o| !dominates(&cand, o));
                    out.push(cand);
                    // if v is a sink, maybe update champion
                    let is_sink = dag.get(&v).map_or(true, |e| e.is_empty());
                    if is_sink {
                        let d = state_delay(&cand);
                        if d > best_delay {
                            best_delay = d;
                            best_state = Some((v, cand));
                        }
                    }
                }
            }
        }
    }

    // reconstruct path
    if let Some((mut node, mut state)) = best_state {
        let mut path: Vec<AigRef> = Vec::new();
        while {
            path.push(node);
            if let Some((prev_node, plog_f, p_n, p_p)) = state.prev {
                if let Some(states) = S.get(&prev_node) {
                    if let Some(&next_state) = states
                        .iter()
                        .find(|s| s.log_f == plog_f && s.n == p_n && s.p == p_p)
                    {
                        node = prev_node;
                        state = next_state;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        } {}
        path.reverse();
        (path, best_delay)
    } else {
        (Vec::new(), best_delay)
    }
}

/// Pre-compute an `h` function (effort) using fan-out and a quadratic model:
/// effort(u) = β₁ · f + β₂ · f², where f = fan-out of `u`.
///
/// * `beta1` defaults to 1.0
/// * `beta2` defaults to 0.0
/// Assumes all sinks have Cin = 1.0.
/// Returns a function to be used as `pin_load` in `worst_case_delay`.
pub fn eff_with_branch(
    dag: &HashMap<AigRef, Vec<(AigRef, f64, f64)>>,
    beta1: f64,
    beta2: f64,
) -> impl Fn(AigRef, AigRef) -> f64 + '_ {
    // 1. pre-compute fan-out for every node
    let mut tot_load: HashMap<AigRef, usize> = HashMap::new();
    for (&u, edges) in dag.iter() {
        tot_load.insert(u, edges.len());
    }

    // 2. capture β₁, β₂ and the table by value
    move |u, _v| {
        let f = *tot_load.get(&u).unwrap_or(&0) as f64;
        beta1 * f + beta2 * f.powi(2)
    }
}

/// Result of logical effort analysis for a GateFn.
pub struct LogicalEffortAnalysis {
    pub dag: HashMap<AigRef, Vec<(AigRef, f64, f64)>>,
    pub path: Vec<AigRef>,
    pub delay: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct GraphLogicalEffortOptions {
    pub beta1: f64,
    pub beta2: f64,
}

fn build_logical_effort_dag(gate_nodes: &[AigNode]) -> HashMap<AigRef, Vec<(AigRef, f64, f64)>> {
    let g_nand = 4.0 / 3.0;
    let p_nand = 2.0;
    let mut dag: HashMap<AigRef, Vec<(AigRef, f64, f64)>> = HashMap::new();
    for (i, node) in gate_nodes.iter().enumerate() {
        let u = AigRef { id: i };
        if let AigNode::And2 { a, b, .. } = node {
            dag.entry(a.node).or_default().push((u, g_nand, p_nand));
            dag.entry(b.node).or_default().push((u, g_nand, p_nand));
        }
    }
    dag
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DependencyStamp {
    node: AigRef,
    frontier_version: u64,
    fanout: usize,
}

#[derive(Clone, Debug)]
struct CachedFrontier {
    states: Vec<State>,
    dependencies: Vec<DependencyStamp>,
    version: u64,
}

/// Incremental graph logical-effort arrival cache for an append-only AIG.
///
/// New consumers can change an existing node's fanout load. Cached frontiers
/// therefore record both dependency versions and dependency fanouts; stale
/// descendants are recomputed lazily when a later query reaches them.
pub struct GraphLogicalEffortArrivalCache {
    options: GraphLogicalEffortOptions,
    processed_gate_count: usize,
    fanouts: Vec<usize>,
    frontiers: Vec<Option<CachedFrontier>>,
    next_frontier_version: u64,
}

impl GraphLogicalEffortArrivalCache {
    pub fn new(options: GraphLogicalEffortOptions) -> Self {
        Self {
            options,
            processed_gate_count: 0,
            fanouts: Vec::new(),
            frontiers: Vec::new(),
            next_frontier_version: 1,
        }
    }

    fn sync_graph(&mut self, gate_nodes: &[AigNode]) {
        assert!(
            gate_nodes.len() >= self.processed_gate_count,
            "graph logical-effort cache requires an append-only gate graph"
        );
        self.fanouts.resize(gate_nodes.len(), 0);
        self.frontiers.resize_with(gate_nodes.len(), || None);
        for node in &gate_nodes[self.processed_gate_count..] {
            if let AigNode::And2 { a, b, .. } = node {
                assert!(a.node.id < gate_nodes.len());
                assert!(b.node.id < gate_nodes.len());
                self.fanouts[a.node.id] += 1;
                self.fanouts[b.node.id] += 1;
            }
        }
        self.processed_gate_count = gate_nodes.len();
    }

    fn make_dependency_stamp(&self, node: AigRef) -> DependencyStamp {
        DependencyStamp {
            node,
            frontier_version: self.frontiers[node.id]
                .as_ref()
                .expect("dependency frontier should be available in topological order")
                .version,
            fanout: self.fanouts[node.id],
        }
    }

    fn extend_frontier(&self, out: &mut Vec<State>, dependency: AigRef) {
        let f = self.fanouts[dependency.id] as f64;
        let h = self.options.beta1 * f + self.options.beta2 * f.powi(2);
        let w = (4.0_f64 / 3.0).ln() + h.ln();
        let input_states = &self.frontiers[dependency.id]
            .as_ref()
            .expect("dependency frontier should be available")
            .states;
        for state in input_states {
            let cand = State {
                log_f: state.log_f + w,
                n: state.n + 1,
                p: state.p + 2.0,
                prev: None,
            };
            if out.iter().any(|other| {
                dominates(other, &cand)
                    || (other.n == cand.n && other.log_f >= cand.log_f && other.p >= cand.p)
            }) {
                continue;
            }
            out.retain(|other| !dominates(&cand, other));
            out.push(cand);
        }
    }

    /// Returns arrival times for all targets, sharing cached work across calls.
    pub fn arrival_times(&mut self, gate_nodes: &[AigNode], targets: &[AigRef]) -> Vec<f64> {
        if targets.is_empty() {
            return Vec::new();
        }
        self.sync_graph(gate_nodes);
        for target in targets {
            assert!(
                target.id < gate_nodes.len(),
                "graph logical-effort target {} is out of range for {} gates",
                target.id,
                gate_nodes.len()
            );
        }

        let topo =
            postorder_for_aig_refs_node_only(targets, gate_nodes, &HashMap::<AigRef, ()>::new());
        for node_ref in topo {
            let dependencies = match &gate_nodes[node_ref.id] {
                AigNode::Input { .. } | AigNode::Literal { .. } => Vec::new(),
                AigNode::And2 { a, b, .. } => vec![
                    self.make_dependency_stamp(a.node),
                    self.make_dependency_stamp(b.node),
                ],
            };
            if self.frontiers[node_ref.id]
                .as_ref()
                .is_some_and(|cached| cached.dependencies == dependencies)
            {
                continue;
            }

            let states = match &gate_nodes[node_ref.id] {
                AigNode::Input { .. } | AigNode::Literal { .. } => vec![State {
                    log_f: 0.0,
                    n: 0,
                    p: 0.0,
                    prev: None,
                }],
                AigNode::And2 { a, b, .. } => {
                    let mut states = Vec::new();
                    self.extend_frontier(&mut states, a.node);
                    self.extend_frontier(&mut states, b.node);
                    states
                }
            };
            let version = self.next_frontier_version;
            self.next_frontier_version += 1;
            self.frontiers[node_ref.id] = Some(CachedFrontier {
                states,
                dependencies,
                version,
            });
        }

        targets
            .iter()
            .map(|target| {
                self.frontiers[target.id]
                    .as_ref()
                    .expect("target frontier should be available")
                    .states
                    .iter()
                    .map(state_delay)
                    .fold(0.0_f64, f64::max)
            })
            .collect()
    }
}

/// Computes graph logical-effort arrival times for all targets in one pass.
pub fn graph_logical_effort_arrival_times(
    gate_nodes: &[AigNode],
    targets: &[AigRef],
    options: &GraphLogicalEffortOptions,
) -> Vec<f64> {
    GraphLogicalEffortArrivalCache::new(*options).arrival_times(gate_nodes, targets)
}

/// Analyzes a GateFn for logical effort using standard NAND2 parameters and
/// eff_with_branch. Returns the DAG, critical path, and delay.
pub fn analyze_graph_logical_effort(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
) -> LogicalEffortAnalysis {
    let dag = build_logical_effort_dag(&gate_fn.gates);
    let pin_load = eff_with_branch(&dag, options.beta1, options.beta2);
    let (path, delay) = worst_case_delay(&dag, pin_load, &gate_fn.gates);
    LogicalEffortAnalysis { dag, path, delay }
}

#[cfg(test)]
mod tests {
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    use super::*;

    #[test]
    fn test_arrival_times_for_multiple_targets_reflect_current_fanout() {
        let mut gb = GateBuilder::new("arrival_times".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let c = gb.add_input("c".to_string(), 1);
        let first = gb.add_nand_binary(*a.get_lsb(0), *b.get_lsb(0));
        let second = gb.add_nand_binary(first, *c.get_lsb(0));
        let options = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        };

        let mut cache = GraphLogicalEffortArrivalCache::new(options);
        let before = cache.arrival_times(&gb.gates, &[first.node, second.node]);
        assert_eq!(before.len(), 2);
        assert!(before[1] > before[0]);

        gb.add_output("result".to_string(), second.into());
        let gate_fn = GateFn {
            name: gb.name.clone(),
            inputs: gb.inputs.clone(),
            outputs: gb.outputs.clone(),
            gates: gb.gates.clone(),
        };
        let whole_graph = analyze_graph_logical_effort(&gate_fn, &options);
        assert!((before[1] - whole_graph.delay).abs() < 1e-9);

        // Appending a new consumer changes first's fanout load. The persistent
        // cache must lazily invalidate the affected descendant frontier.
        let _side_consumer = gb.add_nand_binary(first, *a.get_lsb(0));
        let after = cache.arrival_times(&gb.gates, &[first.node, second.node]);
        assert!(after[1] > before[1]);
    }

    #[test]
    fn test_nand_fanout_case() {
        let mut gb = GateBuilder::new("nand_fanout_case".to_string(), GateBuilderOptions::no_opt());
        // Inputs
        let a0 = gb.add_input("a0".to_string(), 1);
        let a1 = gb.add_input("a1".to_string(), 1);
        let b0 = gb.add_input("b0".to_string(), 1);
        let b1 = gb.add_input("b1".to_string(), 1);
        // First-level NANDs
        let n1 = gb.add_nand_binary(*a0.get_lsb(0), *a1.get_lsb(0));
        let n2 = gb.add_nand_binary(*b0.get_lsb(0), *b1.get_lsb(0));
        // Each first-level NAND drives four more NANDs
        let mut n1_sinks = vec![];
        let mut n2_sinks = vec![];
        for _i in 0..4 {
            let n1_sink = gb.add_nand_binary(n1, gb.get_true());
            n1_sinks.push(n1_sink);
            let n2_sink = gb.add_nand_binary(n2, gb.get_true());
            n2_sinks.push(n2_sink);
        }
        // Outputs (sinks)
        for (_i, &sink) in n1_sinks.iter().enumerate() {
            gb.add_output(format!("n1_{}", _i + 1), sink.into());
        }
        for (_i, &sink) in n2_sinks.iter().enumerate() {
            gb.add_output(format!("n2_{}", _i + 1), sink.into());
        }
        let gate_fn = gb.build();
        let options = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        };
        let analysis = analyze_graph_logical_effort(&gate_fn, &options);
        log::info!("critical path: {:?}", analysis.path);
        let expected = 12.666666666666663;
        let epsilon = 1e-6;
        assert!(
            (analysis.delay - expected).abs() < epsilon,
            "delay was {}",
            analysis.delay
        );
        let options = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 1.0,
        };
        let analysis = analyze_graph_logical_effort(&gate_fn, &options);
        log::info!("critical path: {:?}", analysis.path);
        let expected = 98.0;
        let epsilon = 1e-6;
        assert!(
            (analysis.delay - expected).abs() < epsilon,
            "delay was {}",
            analysis.delay
        );
    }

    #[test]
    fn test_nand_branch_case() {
        let mut gb = GateBuilder::new("nand_branch_case".to_string(), GateBuilderOptions::no_opt());
        // Inputs
        let a0 = gb.add_input("a0".to_string(), 1);
        let a1 = gb.add_input("a1".to_string(), 1);
        let a2 = gb.add_input("a2".to_string(), 1);
        let a3 = gb.add_input("a3".to_string(), 1);
        let b0 = gb.add_input("b0".to_string(), 1);
        let b1 = gb.add_input("b1".to_string(), 1);
        let b2 = gb.add_input("b2".to_string(), 1);
        let b3 = gb.add_input("b3".to_string(), 1);
        // Level-1 NANDs
        let n1 = gb.add_nand_binary(*a0.get_lsb(0), *a1.get_lsb(0));
        let n2 = gb.add_nand_binary(*a2.get_lsb(0), *a3.get_lsb(0));
        let n3 = gb.add_nand_binary(*b0.get_lsb(0), *b1.get_lsb(0));
        let n4 = gb.add_nand_binary(*b2.get_lsb(0), *b3.get_lsb(0));
        // Level-2 NANDs
        let n5 = gb.add_nand_binary(n1, n2);
        let n6 = gb.add_nand_binary(n3, n4);
        // Root NAND
        let n7 = gb.add_nand_binary(n5, n6);
        // Side buffer/inverter
        let inv1 = gb.add_not(n2);
        let buf1 = gb.add_nand_binary(n4, gb.get_true());
        // Sinks
        let m1 = gb.add_nand_binary(n5, gb.get_true());
        let m2 = gb.add_nand_binary(n5, gb.get_true());
        let m3 = gb.add_nand_binary(n6, gb.get_true());
        let v1 = gb.add_nand_binary(inv1, gb.get_true());
        let v2 = gb.add_nand_binary(buf1, gb.get_true());
        let y1 = gb.add_nand_binary(n7, gb.get_true());
        let y2 = gb.add_nand_binary(n7, gb.get_true());
        let y3 = gb.add_nand_binary(n7, gb.get_true());
        let y4 = gb.add_nand_binary(n7, gb.get_true());
        // Outputs (sinks)
        for (_i, &sink) in [m1, m2, m3, v1, v2, y1, y2, y3, y4].iter().enumerate() {
            gb.add_output(format!("sink_{}", _i + 1), sink.into());
        }
        let gate_fn = gb.build();
        let options = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        };
        let analysis = analyze_graph_logical_effort(&gate_fn, &options);
        log::info!("critical path: {:?}", analysis.path);
        let expected = 19.804607143470097;
        let epsilon = 1e-6;
        assert!(
            (analysis.delay - expected).abs() < epsilon,
            "delay was {}",
            analysis.delay
        );
    }
}
