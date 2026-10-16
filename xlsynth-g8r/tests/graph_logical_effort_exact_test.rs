// SPDX-License-Identifier: Apache-2.0

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::gate::{AigBitVector, AigNode, GateFn, Output};
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, analyze_graph_logical_effort, score_graph_logical_effort,
};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};

const LOAD_MODELS: [(f64, f64); 7] = [
    (1.0, 0.0),
    (1.0, 1.0),
    (0.5, 0.1),
    (0.1, 0.0),
    (0.0, 1.0),
    (0.0, 0.0),
    (-1.0, 1.0),
];

/// Counts input edges, including both pins of an AND with repeated operands.
fn fanouts(graph: &GateFn) -> Vec<usize> {
    let mut fanout = vec![0; graph.gates.len()];
    for node in &graph.gates {
        for arg in node.args() {
            fanout[arg.id] += 1;
        }
    }
    fanout
}

/// Enumerates every source-to-sink path using products, without pruning states.
fn enumerate_delay(graph: &GateFn, options: &GraphLogicalEffortOptions) -> f64 {
    let fanout = fanouts(graph);
    let mut paths = Vec::new();
    for (id, node) in graph.gates.iter().enumerate() {
        if fanout[id] == 0 && matches!(node, AigNode::And2 { .. }) {
            paths.push((id, 0usize, 1.0_f64));
        }
    }
    let mut best = -1.0_f64;
    while let Some((id, length, effort)) = paths.pop() {
        match &graph.gates[id] {
            AigNode::And2 { a, b, .. } => {
                for source in [a.node.id, b.node.id] {
                    let f = fanout[source] as f64;
                    let h = options.beta1 * f + options.beta2 * f * f;
                    paths.push((source, length + 1, effort * (4.0 / 3.0) * h));
                }
            }
            _ => {
                let n = length as f64;
                best = best.max(n * effort.powf(1.0 / n) + 2.0 * n);
            }
        }
    }
    best
}

/// Checks both the reported maximum and the concrete path witnessing it.
fn assert_exact_analysis(graph: &GateFn, options: &GraphLogicalEffortOptions) {
    let expected = enumerate_delay(graph, options);
    let actual = analyze_graph_logical_effort(graph, options);
    let score = score_graph_logical_effort(graph, options);
    assert_eq!(score.delay, actual.delay);
    assert_eq!(score.stats.traceback_records, 0);
    assert_eq!(score.stats.candidates, actual.stats.candidates);
    assert_eq!(
        score.stats.dominance_comparisons,
        actual.stats.dominance_comparisons
    );
    assert!(
        (actual.delay - expected).abs() < 1e-9,
        "reported={} enumerated={expected}",
        actual.delay
    );
    if expected == -1.0 {
        assert!(actual.path.is_empty());
        return;
    }
    assert!(actual.path.len() >= 2);
    assert!(!matches!(
        graph.gates[actual.path[0].id],
        AigNode::And2 { .. }
    ));
    let fanout = fanouts(graph);
    assert_eq!(fanout[actual.path.last().unwrap().id], 0);
    let mut effort = 1.0_f64;
    for edge in actual.path.windows(2) {
        assert!(graph.gates[edge[1].id].args().any(|arg| arg == edge[0]));
        let f = fanout[edge[0].id] as f64;
        effort *= (4.0 / 3.0) * (options.beta1 * f + options.beta2 * f * f);
    }
    let n = (actual.path.len() - 1) as f64;
    assert!((n * effort.powf(1.0 / n) + 2.0 * n - actual.delay).abs() < 1e-9);
}

/// A high-fanout suffix can make a shorter prefix become the worst path.
fn check_high_fanout_suffix(sink_count: usize, beta2: f64) {
    let mut gb = GateBuilder::new("fanout_suffix".into(), GateBuilderOptions::opt());
    let x = gb.add_input("x".into(), 3 + sink_count);
    let first = gb.add_and_binary(*x.get_lsb(1), *x.get_lsb(2));
    let shared = gb.add_and_binary(first, *x.get_lsb(0));
    let roots = (0..sink_count)
        .map(|i| gb.add_and_binary(shared, *x.get_lsb(3 + i)))
        .collect::<Vec<_>>();
    gb.add_output("result".into(), AigBitVector::from_lsb_is_index_0(&roots));
    let graph = gb.build();
    let options = GraphLogicalEffortOptions { beta1: 1.0, beta2 };
    let f = sink_count as f64;
    let expected =
        2.0 * (((4.0 / 3.0) * (1.0 + beta2)) * ((4.0 / 3.0) * (f + beta2 * f * f))).sqrt() + 4.0;
    assert!((enumerate_delay(&graph, &options) - expected).abs() < 1e-9);
    assert_exact_analysis(&graph, &options);
}

#[test]
fn shorter_path_can_win_after_linear_fanout_load() {
    check_high_fanout_suffix(32, 0.0);
}

#[test]
fn shorter_path_can_win_after_quadratic_fanout_load() {
    check_high_fanout_suffix(6, 1.0);
}

/// Pruned internal nodes must not be restarted as zero-length source paths.
#[test]
fn pruned_internal_node_does_not_become_a_new_source() {
    let mut gb = GateBuilder::new("pruned_source".into(), GateBuilderOptions::opt());
    let x = gb.add_input("x".into(), 70);
    // This branch establishes the incumbent before the deeper branch is
    // visited.
    let champion = gb.add_and_binary(*x.get_lsb(0), *x.get_lsb(1));
    let mut roots = (2..34)
        .map(|i| gb.add_and_binary(champion, *x.get_lsb(i)))
        .collect::<Vec<_>>();
    let a = gb.add_and_binary(*x.get_lsb(34), *x.get_lsb(35));
    let b = gb.add_and_binary(*x.get_lsb(36), *x.get_lsb(37));
    let shared = gb.add_and_binary(a, b);
    for i in 38..70 {
        roots.push(gb.add_and_binary(shared, *x.get_lsb(i)));
    }
    gb.add_output("result".into(), AigBitVector::from_lsb_is_index_0(&roots));
    let graph = gb.build();
    let options = GraphLogicalEffortOptions {
        beta1: 1.0,
        beta2: 0.0,
    };
    let expected = 2.0 * ((4.0_f64 / 3.0).powi(2) * 32.0).sqrt() + 4.0;
    assert!((enumerate_delay(&graph, &options) - expected).abs() < 1e-9);
    assert_exact_analysis(&graph, &options);
}

#[test]
fn logical_effort_matches_enumeration_on_small_dags() {
    let mut rng = StdRng::seed_from_u64(0x20261015);
    for sample in 0..256 {
        let mut gb = GateBuilder::new(
            "dag".into(),
            GateBuilderOptions {
                fold: sample % 2 == 0,
                hash: sample % 3 == 0,
            },
        );
        let mut available = gb
            .add_input("x".into(), 4 + sample % 4)
            .iter_lsb_to_msb()
            .copied()
            .collect::<Vec<_>>();
        available.push(gb.get_false());
        for _ in 0..1 + sample % 20 {
            let a = available[rng.gen_range(0..available.len())];
            let b = available[rng.gen_range(0..available.len())];
            let a = if rng.gen_bool(0.5) { gb.add_not(a) } else { a };
            available.push(gb.add_and_binary(a, b));
        }
        // Attach outputs before build(); each case selects its own roots below.
        gb.add_output(
            "result".into(),
            AigBitVector::from_lsb_is_index_0(&available),
        );
        let original = gb.build();
        for selection in 0..4 {
            let roots = (0..selection)
                .map(|_| available[rng.gen_range(0..available.len())])
                .collect::<Vec<_>>();
            let live = dce(&GateFn {
                outputs: vec![Output {
                    name: "result".into(),
                    bit_vector: AigBitVector::from_lsb_is_index_0(&roots),
                }],
                ..original.clone()
            });
            for (beta1, beta2) in LOAD_MODELS {
                assert_exact_analysis(&live, &GraphLogicalEffortOptions { beta1, beta2 });
            }
        }
    }
}

/// Enumerates all DAGs with two inputs, false, and up to three ordered ANDs.
#[test]
fn logical_effort_matches_enumeration_on_all_tiny_dags() {
    let mut pending = vec![Vec::<(usize, usize)>::new()];
    let mut comparisons = 0;
    while let Some(edges) = pending.pop() {
        let mut gb = GateBuilder::new("tiny_dag".into(), GateBuilderOptions::no_opt());
        let mut available = gb
            .add_input("x".into(), 2)
            .iter_lsb_to_msb()
            .copied()
            .collect::<Vec<_>>();
        available.push(gb.get_false());
        for &(a, b) in &edges {
            available.push(gb.add_and_binary(available[a], available[b]));
        }
        // Attach outputs before build(); each case selects its own roots below.
        gb.add_output(
            "result".into(),
            AigBitVector::from_lsb_is_index_0(&available),
        );
        let original = gb.build();
        for mask in 0..(1 << edges.len()) {
            let roots = (0..edges.len())
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| available[3 + i])
                .collect::<Vec<_>>();
            let live = dce(&GateFn {
                outputs: vec![Output {
                    name: "result".into(),
                    bit_vector: AigBitVector::from_lsb_is_index_0(&roots),
                }],
                ..original.clone()
            });
            for (beta1, beta2) in LOAD_MODELS {
                assert_exact_analysis(&live, &GraphLogicalEffortOptions { beta1, beta2 });
                comparisons += 1;
            }
        }
        if edges.len() < 3 {
            // Operand order and negation do not change the LE graph. Include
            // repeated operands, which contribute two loads to their source.
            for a in 0..available.len() {
                for b in a..available.len() {
                    let mut next = edges.clone();
                    next.push((a, b));
                    pending.push(next);
                }
            }
        }
    }
    assert_eq!(comparisons, 52_171);
}

/// Disables simplification so every step really adds the reconvergent AND.
fn fibonacci_graph(size: usize) -> GateFn {
    let mut gb = GateBuilder::new("fibonacci".into(), GateBuilderOptions::no_opt());
    let x = gb.add_input("x".into(), 2);
    let mut a = *x.get_lsb(0);
    let mut b = *x.get_lsb(1);
    for _ in 0..size {
        let next = gb.add_and_binary(a, b);
        a = b;
        b = next;
    }
    gb.add_output("result".into(), AigBitVector::from_lsb_is_index_0(&[b]));
    let graph = gb.build();
    assert_eq!(
        graph
            .gates
            .iter()
            .filter(|node| matches!(node, AigNode::And2 { .. }))
            .count(),
        size
    );
    graph
}

#[test]
fn reconvergent_chain_matches_path_enumeration() {
    for size in 1..=20 {
        let graph = fibonacci_graph(size);
        for (beta1, beta2) in LOAD_MODELS {
            assert_exact_analysis(&graph, &GraphLogicalEffortOptions { beta1, beta2 });
        }
    }
}

#[test]
fn reconvergent_chain_hulls_and_traceback_scale_linearly() {
    for size in [128, 512, 2048, 8192] {
        let graph = fibonacci_graph(size);
        for (beta1, beta2) in LOAD_MODELS {
            let options = GraphLogicalEffortOptions { beta1, beta2 };
            let result = analyze_graph_logical_effort(&graph, &options);
            let score = score_graph_logical_effort(&graph, &options);
            assert_eq!(result.delay, score.delay);
            assert_eq!(result.stats.candidates, score.stats.candidates);
            assert_eq!(score.stats.traceback_records, 0);
            assert!(result.stats.max_frontier_len <= 3);
            assert!(result.stats.candidates <= 4 * size);
            assert!(result.stats.total_frontier_states <= 2 * size + 3);
            assert!(result.stats.peak_frontier_slots <= 7);
            assert!(result.stats.traceback_records <= 2 * size + 3);
        }
    }
}

#[test]
fn finite_extreme_loads_and_mixed_sign_coefficients_preserve_witnesses() {
    let graph = fibonacci_graph(12);
    for beta1 in [f64::from_bits(1), 1e-300, 1e-100, 1e100, 1e300] {
        let options = GraphLogicalEffortOptions { beta1, beta2: 0.0 };
        let result = analyze_graph_logical_effort(&graph, &options);
        let score = score_graph_logical_effort(&graph, &options);
        assert_eq!(score.delay, result.delay);
        assert!(result.delay.is_finite());
        let counts = fanouts(&graph);
        let mut log_f = 0.0;
        for edge in result.path.windows(2) {
            assert!(graph.gates[edge[1].id].args().any(|arg| arg == edge[0]));
            log_f += (4.0_f64 / 3.0).ln() + (beta1 * counts[edge[0].id] as f64).ln();
        }
        let n = (result.path.len() - 1) as f64;
        let witness_delay = n * (log_f / n).exp() + 2.0 * n;
        assert!((witness_delay - result.delay).abs() <= 1e-11 * result.delay);
    }
    for (beta1, beta2) in [(2.0, -0.1), (-0.1, 1.0), (0.75, 0.0)] {
        assert_exact_analysis(&graph, &GraphLogicalEffortOptions { beta1, beta2 });
    }
}

#[test]
fn zero_effort_keeps_the_longest_path_and_its_witness() {
    for size in [1, 2, 12, 20] {
        let graph = fibonacci_graph(size);
        for (beta1, beta2) in [(0.0, 0.0), (2.0, -1.0), (-1.0, 1.0)] {
            assert_exact_analysis(&graph, &GraphLogicalEffortOptions { beta1, beta2 });
        }
        let result = analyze_graph_logical_effort(
            &graph,
            &GraphLogicalEffortOptions {
                beta1: 0.0,
                beta2: 0.0,
            },
        );
        assert_eq!(result.delay, 2.0 * size as f64);
        assert_eq!(result.path.len(), size + 1);
        assert_eq!(result.stats.max_frontier_len, 1);
    }
}

#[test]
fn overflowing_final_delay_still_has_a_witness() {
    let graph = fibonacci_graph(64);
    let result = analyze_graph_logical_effort(
        &graph,
        &GraphLogicalEffortOptions {
            beta1: 1e307,
            beta2: 0.0,
        },
    );
    assert_eq!(result.delay, f64::INFINITY);
    assert!(result.path.len() > 2);
    for edge in result.path.windows(2) {
        assert!(graph.gates[edge[1].id].args().any(|arg| arg == edge[0]));
    }
}

#[test]
#[should_panic(expected = "finite nonnegative pin loads")]
fn nonfinite_pin_load_is_rejected() {
    score_graph_logical_effort(
        &fibonacci_graph(1),
        &GraphLogicalEffortOptions {
            beta1: f64::INFINITY,
            beta2: 0.0,
        },
    );
}

#[test]
fn finite_and_zero_effort_paths_compete_by_delay() {
    for depth in [1, 8] {
        let mut gb = GateBuilder::new("mixed_zero".into(), GateBuilderOptions::no_opt());
        let x = gb.add_input("x".into(), depth + 3);
        // Both pins and both users keep the short path's loads positive for
        // beta=(-1,1); every edge in the long branch has a zero-load source.
        let short = gb.add_and_binary(*x.get_lsb(0), *x.get_lsb(0));
        let mut long = *x.get_lsb(1);
        for i in 0..depth {
            long = gb.add_and_binary(long, *x.get_lsb(i + 2));
        }
        let result = gb.add_and_binary(short, long);
        let side = gb.add_and_binary(short, *x.get_lsb(depth + 2));
        gb.add_output(
            "result".into(),
            AigBitVector::from_lsb_is_index_0(&[result, side]),
        );
        let graph = gb.build();
        let options = GraphLogicalEffortOptions {
            beta1: -1.0,
            beta2: 1.0,
        };
        assert_exact_analysis(&graph, &options);
        let analysis = analyze_graph_logical_effort(&graph, &options);
        if depth == 8 {
            assert_eq!(analysis.delay, 18.0);
            assert_eq!(analysis.path.len(), 10);
        } else {
            assert!((analysis.delay - 28.0 / 3.0).abs() < 1e-9);
            assert_eq!(analysis.path.len(), 3);
        }
    }
}
