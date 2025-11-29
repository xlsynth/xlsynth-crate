// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::aig::gate::{AigNode, GateFn};

/// Computes a histogram of fanout counts for all nodes in the given GateFn.
///
/// Primary outputs are not counted in the histogram, as their fanout is
/// conceptually unknown/unknowable (they may be used externally).
/// The special node AigRef 0 (the literal value zero) is also excluded from the
/// histogram. Only internal nodes and inputs are counted.
/// Returns a map from fanout count to the number of nodes with that fanout.
pub fn fanout_histogram(gate_fn: &GateFn) -> HashMap<usize, usize> {
    // Map from node id to fanout count
    let mut fanout_map: HashMap<usize, usize> = HashMap::new();
    for node in gate_fn.gates.iter() {
        match node {
            AigNode::And2 { a, b, .. } => {
                *fanout_map.entry(a.node.id).or_insert(0) += 1;
                *fanout_map.entry(b.node.id).or_insert(0) += 1;
            }
            _ => {}
        }
    }
    // Collect all node ids that are used as outputs (primary outputs)
    let mut output_node_ids = std::collections::HashSet::new();
    for output in &gate_fn.outputs {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            output_node_ids.insert(operand.node.id);
        }
    }
    // Now, build the histogram: map from fanout count to number of nodes with that
    // fanout, excluding primary outputs and AigRef 0 (literal zero)
    let mut histogram: HashMap<usize, usize> = HashMap::new();
    for id in 0..gate_fn.gates.len() {
        if id == 0 {
            continue; // skip AigRef 0 (literal zero)
        }
        if output_node_ids.contains(&id) {
            continue; // skip primary outputs
        }
        let fanout = *fanout_map.get(&id).unwrap_or(&0);
        *histogram.entry(fanout).or_insert(0) += 1;
    }
    histogram
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        setup_graph_with_more_redundancies, setup_graph_with_redundancies, setup_simple_graph,
    };

    #[test]
    fn test_fanout_histogram_simple_graph() {
        let g = setup_simple_graph();
        let hist = fanout_histogram(&g.g);
        // In the simple graph, only non-output, non-AigRef 0 nodes are counted:
        // i0: fanout 1
        // i1: fanout 2
        // i2: fanout 2
        // i3: fanout 1
        // a: fanout 1
        // b: fanout 1
        // (c and o are outputs and excluded)
        // So histogram: {1: 4, 2: 2}
        assert_eq!(hist.get(&1).copied().unwrap_or(0), 4);
        assert_eq!(hist.get(&2).copied().unwrap_or(0), 2);
        // No non-output node has fanout 0, so we do not assert on hist.get(&0)
    }

    #[test]
    fn test_fanout_histogram_graph_with_redundancies() {
        let g = setup_graph_with_redundancies();
        let hist = fanout_histogram(&g.g);
        // Check that the histogram is well-formed (all non-output, non-AigRef 0 nodes
        // counted)
        let output_node_ids: std::collections::HashSet<_> =
            g.g.outputs
                .iter()
                .flat_map(|output| output.bit_vector.iter_lsb_to_msb().map(|op| op.node.id))
                .collect();
        let total_nodes = g.g.gates.len();
        let non_output_non_zero_nodes = (1..total_nodes)
            .filter(|id| !output_node_ids.contains(id))
            .count();
        let sum: usize = hist.values().sum();
        assert_eq!(sum, non_output_non_zero_nodes);
    }

    #[test]
    fn test_fanout_histogram_graph_with_more_redundancies() {
        let g = setup_graph_with_more_redundancies();
        let hist = fanout_histogram(&g.g);
        // Check that the histogram is well-formed (all non-output, non-AigRef 0 nodes
        // counted)
        let output_node_ids: std::collections::HashSet<_> =
            g.g.outputs
                .iter()
                .flat_map(|output| output.bit_vector.iter_lsb_to_msb().map(|op| op.node.id))
                .collect();
        let total_nodes = g.g.gates.len();
        let non_output_non_zero_nodes = (1..total_nodes)
            .filter(|id| !output_node_ids.contains(id))
            .count();
        let sum: usize = hist.values().sum();
        assert_eq!(sum, non_output_non_zero_nodes);
    }
}
