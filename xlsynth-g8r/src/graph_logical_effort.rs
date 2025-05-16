// SPDX-License-Identifier: Apache-2.0

//! Graph-based logical effort worst-case delay estimation.

use crate::gate::{AigRef, GateFn};
use crate::topo::topo_sort_refs;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug)]
struct State {
    log_f: f64,
    n: usize,
    p: f64,
    prev: Option<(AigRef, f64, usize, f64)>,
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
    gate_nodes: &[crate::gate::AigNode],
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

    // dominance function
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
                        let d = (cand.n as f64) * ((cand.log_f / (cand.n as f64)).exp()) + cand.p;
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

pub struct GraphLogicalEffortOptions {
    pub beta1: f64,
    pub beta2: f64,
}

/// Analyzes a GateFn for logical effort using standard NAND2 parameters and
/// eff_with_branch. Returns the DAG, critical path, and delay.
pub fn analyze_graph_logical_effort(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
) -> LogicalEffortAnalysis {
    let g_nand = 4.0 / 3.0;
    let p_nand = 2.0;
    let mut dag: HashMap<AigRef, Vec<(AigRef, f64, f64)>> = HashMap::new();
    for (i, node) in gate_fn.gates.iter().enumerate() {
        let u = AigRef { id: i };
        match node {
            crate::gate::AigNode::And2 { a, b, .. } => {
                dag.entry(a.node).or_default().push((u, g_nand, p_nand));
                dag.entry(b.node).or_default().push((u, g_nand, p_nand));
            }
            _ => {}
        }
    }
    let pin_load = eff_with_branch(&dag, options.beta1, options.beta2);
    let (path, delay) = worst_case_delay(&dag, pin_load, &gate_fn.gates);
    LogicalEffortAnalysis { dag, path, delay }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_within,
        gate_builder::{GateBuilder, GateBuilderOptions},
        test_utils::{load_bf16_add_sample, load_bf16_mul_sample, Opt},
    };

    use super::*;

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

    #[test]
    fn test_graph_logical_effort_bf16_add() {
        let bf16_add = load_bf16_add_sample(Opt::Yes);
        let options = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        };
        let analysis = analyze_graph_logical_effort(&bf16_add.gate_fn, &options);
        let expected = 594.0;
        let epsilon = 10.0;
        assert!(
            (analysis.delay - expected).abs() < epsilon,
            "delay was {}",
            analysis.delay
        );
        let options = GraphLogicalEffortOptions {
            beta1: 1.5,
            beta2: 0.5,
        };
        let analysis = analyze_graph_logical_effort(&bf16_add.gate_fn, &options);
        let expected = 1298.0;
        let epsilon = 10.0;
        assert_within!(analysis.delay, expected, epsilon);
    }

    #[test]
    fn test_graph_logical_effort_bf16_mul() {
        let bf16_mul = load_bf16_mul_sample(Opt::Yes);
        let options = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        };
        let analysis = analyze_graph_logical_effort(&bf16_mul.gate_fn, &options);
        let expected = 480.0;
        let epsilon = 10.0;
        assert_within!(analysis.delay, expected, epsilon);
    }
}
