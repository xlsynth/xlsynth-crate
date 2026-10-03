// SPDX-License-Identifier: Apache-2.0

//! Graph-based logical effort worst-case delay estimation.

use crate::aig::topo::topo_sort_refs;
use crate::aig::{AigNode, AigRef, GateFn};
use std::collections::HashMap;

#[derive(Copy, Clone, Debug)]
struct State {
    log_f: f64,
    n: usize,
    p: f64,
    prev: Option<(AigRef, f64, usize, f64)>,
}

/// Counts visits to graph entries and frontier states, or leaves work
/// unlimited.
struct WorkBudget(Option<usize>);

impl WorkBudget {
    fn spend(&mut self, work: usize) -> Option<()> {
        if let Some(remaining) = &mut self.0 {
            *remaining = remaining.checked_sub(work)?;
        }
        Some(())
    }
}

/// Computes the worst-case delay in a DAG using logical effort analysis.
///
/// - `dag` maps each node to a list of outgoing edges `(v, g, p)` where `g` is
///   the logical effort and `p` is the parasitic delay of the edge.
/// - `pin_load` is a function computing the load `h` for edge `(u, v)`.
///
/// Returns a tuple `(path, delay)` where `path` is the sequence of nodes
/// and `delay` is the worst-case delay value, or `None` on budget exhaustion.
#[allow(non_snake_case)]
fn worst_case_delay<F>(
    dag: &HashMap<AigRef, Vec<(AigRef, f64, f64)>>,
    pin_load: F,
    gate_nodes: &[AigNode],
    budget: &mut WorkBudget,
) -> Option<(Vec<AigRef>, f64)>
where
    F: Fn(AigRef, AigRef) -> f64,
{
    // global constants
    let mut g_max = 0.0_f64;
    let mut p_max_global = 0.0_f64;
    let mut h_max = 0.0_f64;
    budget.spend(dag.len())?;
    for (&u, edges) in dag.iter() {
        budget.spend(edges.len())?;
        for &(v, g, p) in edges {
            g_max = g_max.max(g);
            p_max_global = p_max_global.max(p);
            let h = pin_load(u, v);
            if h > h_max {
                h_max = h;
            }
        }
    }
    let log_gh_max = (g_max * h_max).ln();

    // Prepay the node tables and traversal passes before topo_sort_refs
    // allocates. Each AIG node has at most two operands, so ten visits per
    // node bounds its linear setup and traversal work on an acyclic graph.
    for _ in 0..10 {
        budget.spend(gate_nodes.len())?;
    }
    let topo: Vec<AigRef> = topo_sort_refs(gate_nodes);

    // compute longest path R in reverse topological order
    let mut R: HashMap<AigRef, usize> = HashMap::new();
    for &u in topo.iter().rev() {
        budget.spend(1)?;
        let mut max_r = 0;
        if let Some(edges) = dag.get(&u) {
            budget.spend(edges.len())?;
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
        budget.spend(1)?;
        // initialize the frontier
        if S.get(&u).map_or(true, |v| v.is_empty()) {
            budget.spend(1)?;
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
                budget.spend(1)?;
                let h = pin_load(u, v);
                let w = g.ln() + h.ln();
                budget.spend(S.get(&u).unwrap().len())?;
                let current_states = S.get(&u).unwrap().clone();
                for state in current_states {
                    budget.spend(1)?;
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
                        budget.spend(1)?;
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
                    budget.spend(out.len())?;
                    out.retain(|o| !dominates(&cand, o));
                    budget.spend(1)?;
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
            budget.spend(1)?;
            path.push(node);
            if let Some((prev_node, plog_f, p_n, p_p)) = state.prev {
                if let Some(states) = S.get(&prev_node) {
                    budget.spend(states.len())?;
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
        budget.spend(path.len())?;
        path.reverse();
        Some((path, best_delay))
    } else {
        Some((Vec::new(), best_delay))
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

/// Analyzes a GateFn for logical effort using standard NAND2 parameters and
/// eff_with_branch. Returns the DAG, critical path, and delay.
pub fn analyze_graph_logical_effort(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
) -> LogicalEffortAnalysis {
    analyze_with_budget(gate_fn, options, &mut WorkBudget(None))
        .expect("unlimited logical effort analysis cannot exhaust its work budget")
}

/// Analyzes logical effort, returning `None` before exceeding the work budget.
///
/// Units count graph node/edge visits and frontier-state cloning, propagation,
/// comparisons, retention, and path reconstruction. Linear helper traversals
/// are conservatively prepaid before allocation. As with the unlimited API,
/// the input must be an acyclic graph with valid operand references.
pub fn analyze_graph_logical_effort_with_budget(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
    work_budget: usize,
) -> Option<LogicalEffortAnalysis> {
    analyze_with_budget(gate_fn, options, &mut WorkBudget(Some(work_budget)))
}

/// Shares the numerical analysis between bounded and unlimited callers.
fn analyze_with_budget(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
    budget: &mut WorkBudget,
) -> Option<LogicalEffortAnalysis> {
    budget.spend(gate_fn.gates.len())?;
    let g_nand = 4.0 / 3.0;
    let p_nand = 2.0;
    let mut dag: HashMap<AigRef, Vec<(AigRef, f64, f64)>> = HashMap::new();
    for (i, node) in gate_fn.gates.iter().enumerate() {
        let u = AigRef { id: i };
        match node {
            AigNode::And2 { a, b, .. } => {
                budget.spend(2)?;
                dag.entry(a.node).or_default().push((u, g_nand, p_nand));
                dag.entry(b.node).or_default().push((u, g_nand, p_nand));
            }
            _ => {
                // Inputs and literals have no incoming gate edges.
            }
        }
    }
    budget.spend(dag.len())?;
    let pin_load = eff_with_branch(&dag, options.beta1, options.beta2);
    let (path, delay) = worst_case_delay(&dag, pin_load, &gate_fn.gates, budget)?;
    Some(LogicalEffortAnalysis { dag, path, delay })
}

#[cfg(test)]
mod tests {
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

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
        let bounded = analyze_graph_logical_effort_with_budget(&gate_fn, &options, 10_000).unwrap();
        assert_eq!(bounded.dag, analysis.dag);
        assert_eq!(bounded.path, analysis.path);
        assert_eq!(bounded.delay, analysis.delay);
        assert!(analyze_graph_logical_effort_with_budget(&gate_fn, &options, 0).is_none());
    }

    #[test]
    fn bounded_analysis_declines_large_reconvergent_frontiers() {
        let mut gb = GateBuilder::new("reconvergent".to_string(), GateBuilderOptions::no_opt());
        let mut a = *gb.add_input("a".to_string(), 1).get_lsb(0);
        let mut b = *gb.add_input("b".to_string(), 1).get_lsb(0);
        for _ in 0..999 {
            let next = gb.add_and_binary(a, b);
            a = b;
            b = next;
        }
        gb.add_output("out".to_string(), b.into());
        let gate_fn = gb.build();
        let options = GraphLogicalEffortOptions {
            beta1: 2.0,
            beta2: 0.0,
        };
        // Enough for linear graph setup; the many nondominated paths exhaust
        // the budget during frontier propagation and comparison.
        let work_budget = 100_000;
        assert!(work_budget > 64 * gate_fn.gates.len());
        assert!(
            analyze_graph_logical_effort_with_budget(&gate_fn, &options, work_budget).is_none()
        );
    }
}
