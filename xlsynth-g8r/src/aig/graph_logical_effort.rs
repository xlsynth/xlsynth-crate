// SPDX-License-Identifier: Apache-2.0

//! Graph-based logical effort worst-case delay estimation.

use crate::aig::topo::topo_sort_refs;
use crate::aig::{AigNode, AigRef, GateFn};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::num::NonZeroUsize;

/// A nondominated source-to-node prefix, with a compact traceback link.
#[derive(Copy, Clone, Debug)]
struct State {
    log_f: f64,
    n: usize,
    p: f64,
    prev: Option<NonZeroUsize>,
}

/// Stable insertion order plus an index into each path length's Pareto set.
#[derive(Default)]
struct Frontier {
    states: Vec<Option<State>>,
    by_length: HashMap<usize, SmallVec<[usize; 1]>>,
}

impl Frontier {
    /// Only equal-length prefixes can dominate one another after any suffix.
    fn insert(&mut self, candidate: State, stats: &mut LogicalEffortStats) {
        let bucket = self.by_length.entry(candidate.n).or_default();
        for &index in bucket.iter() {
            let old = self.states[index].unwrap();
            stats.dominance_comparisons += 1;
            if old.log_f >= candidate.log_f && old.p >= candidate.p {
                return;
            }
        }
        bucket.retain(|index| {
            let index = *index;
            let old = self.states[index].unwrap();
            stats.dominance_comparisons += 1;
            if candidate.log_f >= old.log_f && candidate.p >= old.p {
                self.states[index] = None;
                false
            } else {
                true
            }
        });
        bucket.push(self.states.len());
        self.states.push(Some(candidate));
    }
}

/// Immutable traceback records outlive the numeric frontiers that created them.
struct Trace {
    node: AigRef,
    prev: Option<NonZeroUsize>,
}

/// Work and storage counters for profiling exact graph logical effort.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LogicalEffortStats {
    /// Prefix extensions considered, including those pruned by the global
    /// bound.
    pub candidates: usize,
    /// Pairwise comparisons within equal-length Pareto sets.
    pub dominance_comparisons: usize,
    /// Maximum simultaneously retained numeric slots, including replaced
    /// states. This counts initialized slots, not allocator capacity or
    /// HashMap storage.
    pub peak_frontier_slots: usize,
    /// Compact traceback records retained until the analysis finishes.
    pub traceback_records: usize,
}

/// The maximum, its concrete witness, and the work needed to find it.
struct DelayResult {
    path: Vec<AigRef>,
    delay: f64,
    stats: LogicalEffortStats,
}

/// Computes the worst delay while releasing numeric frontiers after
/// propagation.
///
/// Length indexing avoids comparing unrelated lengths. For the standard NAND2
/// model p = 2*n, so each bucket holds one state. The per-length Pareto set
/// also handles edges with different parasitics. Traceback storage can still be
/// quadratic in node count for deep reconvergent graphs. Releasing numeric
/// frontiers does not change that bound.
fn worst_case_delay<F>(
    dag: &HashMap<AigRef, Vec<(AigRef, f64, f64)>>,
    pin_load: F,
    gate_nodes: &[AigNode],
    with_traceback: bool,
) -> DelayResult
where
    F: Fn(AigRef, AigRef) -> f64,
{
    let g_max = dag
        .values()
        .flat_map(|edges| edges.iter().map(|&(_, g, _)| g))
        .fold(0.0_f64, f64::max);
    let p_max_global = dag
        .values()
        .flat_map(|edges| edges.iter().map(|&(_, _, p)| p))
        .fold(0.0_f64, f64::max);
    let mut h_max = 0.0_f64;
    for (&u, edges) in dag {
        for &(v, _, _) in edges {
            h_max = h_max.max(pin_load(u, v));
        }
    }
    let log_gh_max = (g_max * h_max).ln();
    let topo = topo_sort_refs(gate_nodes);
    let mut remaining = vec![0usize; gate_nodes.len()];
    for &u in topo.iter().rev() {
        if let Some(edges) = dag.get(&u) {
            for &(v, _, _) in edges {
                remaining[u.id] = remaining[u.id].max(remaining[v.id] + 1);
            }
        }
    }

    let mut frontiers: HashMap<AigRef, Frontier> = HashMap::new();
    let mut traces: Vec<Trace> = Vec::new();
    let mut stats = LogicalEffortStats::default();
    let mut frontier_slots = 0usize;
    let mut best_delay = -1.0_f64;
    let mut best_state: Option<(AigRef, State)> = None;
    for &u in &topo {
        let mut current = frontiers.remove(&u).unwrap_or_default();
        if !matches!(gate_nodes[u.id], AigNode::And2 { .. }) {
            current.insert(
                State {
                    log_f: 0.0,
                    n: 0,
                    p: 0.0,
                    prev: None,
                },
                &mut stats,
            );
            frontier_slots += 1;
        }
        // An internal AND with no retained states stays empty: it must never
        // become an artificial zero-length source after global pruning.
        let current_slots = current.states.len();
        stats.peak_frontier_slots = stats.peak_frontier_slots.max(frontier_slots);
        if let Some(edges) = dag.get(&u) {
            // A node is finalized before its successors are visited. Preserve
            // only its traceback links; numeric states and length indexes can
            // be dropped after this propagation. Iteration uses insertion
            // order, never HashMap order, including when
            // equal-scoring paths tie.
            if with_traceback {
                for state in current.states.iter_mut().flatten() {
                    traces.push(Trace {
                        node: u,
                        prev: state.prev,
                    });
                    state.prev = NonZeroUsize::new(traces.len());
                }
            }
            for &(v, g, p) in edges {
                let w = g.ln() + pin_load(u, v).ln();
                let is_sink = dag.get(&v).map_or(true, Vec::is_empty);
                for state in current.states.iter().flatten() {
                    stats.candidates += 1;
                    let candidate = State {
                        log_f: state.log_f + w,
                        n: state.n + 1,
                        p: state.p + p,
                        prev: state.prev,
                    };
                    let r_left = remaining[v.id] as f64;
                    let n_max = candidate.n as f64 + r_left;
                    let log_f_max = candidate.log_f + r_left * log_gh_max;
                    let p_max = candidate.p + r_left * p_max_global;
                    let upper = n_max * (log_f_max / n_max).exp() + p_max;
                    if upper <= best_delay {
                        continue;
                    }
                    let out = frontiers.entry(v).or_default();
                    let before = out.states.len();
                    out.insert(candidate, &mut stats);
                    frontier_slots += out.states.len() - before;
                    stats.peak_frontier_slots = stats.peak_frontier_slots.max(frontier_slots);
                    if is_sink {
                        let n = candidate.n as f64;
                        let delay = n * (candidate.log_f / n).exp() + candidate.p;
                        if delay > best_delay {
                            best_delay = delay;
                            if with_traceback {
                                best_state = Some((v, candidate));
                            }
                        }
                    }
                }
            }
        }
        frontier_slots -= current_slots;
    }
    stats.traceback_records = traces.len();
    let mut path = Vec::new();
    if let Some((node, state)) = best_state {
        path.push(node);
        let mut previous = state.prev;
        while let Some(index) = previous {
            let trace = &traces[index.get() - 1];
            path.push(trace.node);
            previous = trace.prev;
        }
        path.reverse();
    }
    DelayResult {
        path,
        delay: best_delay,
        stats,
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
    pub stats: LogicalEffortStats,
}

pub struct GraphLogicalEffortOptions {
    pub beta1: f64,
    pub beta2: f64,
}

/// Exact worst-case score and profiling counters, without path storage.
pub struct LogicalEffortScore {
    pub delay: f64,
    pub stats: LogicalEffortStats,
}

/// Analyzes a GateFn using standard NAND2 parameters and returns a worst path.
/// Retaining traceback can use O(nodes * depth) space on reconvergent graphs.
/// Use `score_graph_logical_effort` when only the delay is needed.
pub fn analyze_graph_logical_effort(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
) -> LogicalEffortAnalysis {
    analyze_graph_logical_effort_impl(gate_fn, options, /* with_traceback= */ true)
}

/// Computes the same exact score without retaining traceback records.
///
/// Numeric frontiers are released after propagation. This makes storage linear
/// for the single-sink Fibonacci DAG, but runtime is still quadratic, and a
/// general DAG may have many simultaneously live node/length states. No path
/// lengths are capped and no approximation is used. No-AND graphs return -1.
pub fn score_graph_logical_effort(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
) -> LogicalEffortScore {
    let analysis =
        analyze_graph_logical_effort_impl(gate_fn, options, /* with_traceback= */ false);
    LogicalEffortScore {
        delay: analysis.delay,
        stats: analysis.stats,
    }
}

/// Shares exact propagation between score-only and path-producing callers.
fn analyze_graph_logical_effort_impl(
    gate_fn: &GateFn,
    options: &GraphLogicalEffortOptions,
    with_traceback: bool,
) -> LogicalEffortAnalysis {
    let g_nand = 4.0 / 3.0;
    let p_nand = 2.0;
    let mut dag: HashMap<AigRef, Vec<(AigRef, f64, f64)>> = HashMap::new();
    for (i, node) in gate_fn.gates.iter().enumerate() {
        let u = AigRef { id: i };
        match node {
            AigNode::And2 { a, b, .. } => {
                dag.entry(a.node).or_default().push((u, g_nand, p_nand));
                dag.entry(b.node).or_default().push((u, g_nand, p_nand));
            }
            _ => {}
        }
    }
    let pin_load = eff_with_branch(&dag, options.beta1, options.beta2);
    let result = worst_case_delay(&dag, pin_load, &gate_fn.gates, with_traceback);
    LogicalEffortAnalysis {
        dag,
        path: result.path,
        delay: result.delay,
        stats: result.stats,
    }
}

#[cfg(test)]
mod tests {
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    use super::*;

    #[test]
    fn equal_length_pareto_frontier_keeps_effort_parasitic_tradeoffs() {
        let mut frontier = Frontier::default();
        let mut stats = LogicalEffortStats::default();
        for (log_f, p) in [(1.0, 5.0), (3.0, 1.0), (0.0, 0.0)] {
            frontier.insert(
                State {
                    log_f,
                    n: 2,
                    p,
                    prev: None,
                },
                &mut stats,
            );
        }
        assert_eq!(frontier.by_length[&2].len(), 2);
        let comparisons = stats.dominance_comparisons;
        frontier.insert(
            State {
                log_f: 100.0,
                n: 3,
                p: 100.0,
                prev: None,
            },
            &mut stats,
        );
        assert_eq!(stats.dominance_comparisons, comparisons);
        assert_eq!(frontier.by_length[&2].len(), 2);
        frontier.insert(
            State {
                log_f: 4.0,
                n: 2,
                p: 6.0,
                prev: None,
            },
            &mut stats,
        );
        assert_eq!(frontier.by_length[&2].len(), 1);
        assert_eq!(frontier.states.iter().flatten().count(), 2);
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
