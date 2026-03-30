// SPDX-License-Identifier: Apache-2.0

use crate::aig::fanout::fanout_histogram;
use crate::aig::gate::{self, AigNode};
use crate::aig::topo::topo_sort_refs;
use crate::use_count::get_id_to_use_count;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct SummaryStats {
    pub live_nodes: usize,
    pub deepest_path: usize,
    pub fanout_histogram: BTreeMap<usize, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AigStats {
    /// Count of live `AigNode::And2` nodes (i.e. the AIGER "A" count).
    pub and_nodes: usize,
    /// Maximum depth in terms of And-gate levels from primary inputs to any
    /// primary output (0 if all outputs are literals/inputs or there are no
    /// outputs).
    pub max_depth: usize,
    /// Histogram of fanout counts (use-counts), excluding literal nodes.
    ///
    /// Notes:
    /// - Primary outputs are counted as a "use", so nodes that only feed a
    ///   primary output will typically contribute to the `1` bin.
    /// - This is computed over *live* nodes (reachable from primary outputs).
    pub fanout_histogram: BTreeMap<usize, usize>,
}

// Add structured return type for gate depth info.
#[derive(Debug)]
pub struct GateDepthStats {
    pub depth_to_count: HashMap<usize, usize>,
    pub deepest_path: Vec<gate::AigRef>,
    pub ref_to_depth: HashMap<gate::AigRef, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelCriticalPathAnds {
    pub and_nodes: BTreeSet<gate::AigRef>,
    pub depth_aig_nodes: usize,
}

/// Returns the live `And2` nodes that lie on at least one max-level path from a
/// primary input to a primary output.
pub fn get_level_critical_path_ands(
    gate_fn: &gate::GateFn,
    live_nodes: &[gate::AigRef],
) -> LevelCriticalPathAnds {
    let depth_stats = get_gate_depth(gate_fn, live_nodes);
    let Some(max_depth) = gate_fn
        .outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb())
        .filter_map(|operand| depth_stats.ref_to_depth.get(&operand.node).copied())
        .max()
    else {
        return LevelCriticalPathAnds {
            and_nodes: BTreeSet::new(),
            depth_aig_nodes: 0,
        };
    };

    if max_depth == 0 {
        return LevelCriticalPathAnds {
            and_nodes: BTreeSet::new(),
            depth_aig_nodes: 0,
        };
    }

    let mut critical_and_nodes = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut worklist: Vec<gate::AigRef> = gate_fn
        .outputs
        .iter()
        .flat_map(|output| output.bit_vector.iter_lsb_to_msb())
        .filter_map(|operand| {
            depth_stats
                .ref_to_depth
                .get(&operand.node)
                .copied()
                .filter(|depth| *depth == max_depth)
                .map(|_| operand.node)
        })
        .collect();

    while let Some(node_ref) = worklist.pop() {
        if !visited.insert(node_ref) {
            continue;
        }

        let Some(node_depth) = depth_stats.ref_to_depth.get(&node_ref).copied() else {
            continue;
        };

        let AigNode::And2 { a, b, .. } = &gate_fn.gates[node_ref.id] else {
            continue;
        };
        critical_and_nodes.insert(node_ref);

        for child in [a.node, b.node] {
            let Some(child_depth) = depth_stats.ref_to_depth.get(&child).copied() else {
                continue;
            };
            if child_depth + 1 == node_depth {
                worklist.push(child);
            }
        }
    }

    LevelCriticalPathAnds {
        and_nodes: critical_and_nodes,
        depth_aig_nodes: max_depth,
    }
}

/// Returns the live `And2` nodes that lie on at least one max-level path from a
/// primary input to a primary output.
pub fn get_level_critical_path_and_nodes(
    gate_fn: &gate::GateFn,
    live_nodes: &[gate::AigRef],
) -> BTreeSet<gate::AigRef> {
    get_level_critical_path_ands(gate_fn, live_nodes).and_nodes
}

/// Returns:
/// * a mapping that shows {depth: count} where the count is in number of gates.
/// * the deepest path in the gate DAG
pub fn get_gate_depth(gate_fn: &gate::GateFn, live_nodes: &[gate::AigRef]) -> GateDepthStats {
    let mut depths: HashMap<gate::AigRef, usize> = HashMap::new();
    for input in gate_fn.inputs.iter() {
        for operand in input.bit_vector.iter_lsb_to_msb() {
            depths.insert(operand.node, 0);
        }
    }

    for gate_ref in gate_fn.post_order_refs() {
        let gate: &AigNode = &gate_fn.gates[gate_ref.id];
        match gate {
            &AigNode::Input { .. } => {
                continue;
            }
            &AigNode::Literal { .. } => {
                assert!(gate_ref.id < 2);
                depths.insert(gate_ref, 0);
            }
            &AigNode::And2 { a, b, .. } => {
                depths.insert(
                    gate_ref,
                    1 + std::cmp::max(depths.get(&a.node).unwrap(), depths.get(&b.node).unwrap()),
                );
            }
        }
    }

    // We do this in a worklist / topological fashion to avoid deep recursion and
    // potential stack overflows when the AIG has very long paths (e.g. >100k).
    let topo_order = topo_sort_refs(&gate_fn.gates);
    for node_ref in topo_order {
        if depths.contains_key(&node_ref) {
            continue;
        }
        let depth = match &gate_fn.gates[node_ref.id] {
            AigNode::Input { .. } | AigNode::Literal { .. } => 0,
            AigNode::And2 { a, b, .. } => {
                // We expect childrens' depths to be present as topo order ensures
                // they come earlier.
                1 + std::cmp::max(
                    *depths.get(&a.node).expect("child depth missing (a)"),
                    *depths.get(&b.node).expect("child depth missing (b)"),
                )
            }
        };
        depths.insert(node_ref, depth);
    }

    // Filter to just the nodes that are outputs to determine the deepest primary
    // output.
    let mut deepest_primary_output: Option<(gate::AigRef, usize)> = None;
    for output in gate_fn.outputs.iter() {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            if let Some(depth) = depths.get(&operand.node) {
                if deepest_primary_output.is_none()
                    || *depth > deepest_primary_output.as_ref().unwrap().1
                {
                    deepest_primary_output = Some((operand.node, *depth));
                }
            }
        }
    }

    if deepest_primary_output.is_none() {
        // If there are no outputs for this function, its summary stats are trivial.
        return GateDepthStats {
            depth_to_count: HashMap::new(),
            deepest_path: vec![],
            ref_to_depth: depths,
        };
    }

    log::trace!("Deepest primary output: {:?}", deepest_primary_output);

    let mut deepest_path = vec![];
    // Get the GateRef having the largest depth.
    let deepest_gate_ref = deepest_primary_output.unwrap().0;
    let mut current_gate_ref = Some(deepest_gate_ref);
    while let Some(gate_ref) = current_gate_ref {
        deepest_path.push(gate_ref);
        // Get whichever arg of this gate has the largest depth.
        let gate: &AigNode = &gate_fn.gates[gate_ref.id];
        if matches!(gate, AigNode::Input { .. } | AigNode::Literal { .. }) {
            break;
        }
        let args: Vec<gate::AigRef> = gate.get_args();
        assert!(!args.is_empty(), "gate {:?} should have args", gate);
        let max_arg_depth = args
            .iter()
            .map(|arg| depths.get(arg).unwrap())
            .max()
            .unwrap();
        current_gate_ref = args
            .iter()
            .find(|arg| depths.get(arg).unwrap() == max_arg_depth)
            .map(|arg| *arg);
    }

    let mut depth_to_count: HashMap<usize, usize> = HashMap::new();
    for node in live_nodes {
        if let Some(depth) = depths.get(node) {
            *depth_to_count.entry(*depth).or_insert(0) += 1;
        }
    }
    GateDepthStats {
        depth_to_count,
        deepest_path,
        ref_to_depth: depths,
    }
}

pub fn get_summary_stats(gate_fn: &gate::GateFn) -> SummaryStats {
    let id_to_use_count: HashMap<gate::AigRef, usize> = get_id_to_use_count(&gate_fn);
    let live_nodes: Vec<gate::AigRef> = id_to_use_count.keys().cloned().collect();

    let stats = get_gate_depth(&gate_fn, &live_nodes);

    let hist = fanout_histogram(gate_fn);
    let hist_sorted: BTreeMap<usize, usize> = hist.into_iter().collect();
    let summary_stats = SummaryStats {
        live_nodes: live_nodes.len(),
        deepest_path: stats.deepest_path.len(),
        fanout_histogram: hist_sorted,
    };
    summary_stats
}

pub fn get_aig_stats(gate_fn: &gate::GateFn) -> AigStats {
    let id_to_use_count: HashMap<gate::AigRef, usize> = get_id_to_use_count(gate_fn);
    let live_nodes: Vec<gate::AigRef> = id_to_use_count.keys().cloned().collect();

    let and_nodes = live_nodes
        .iter()
        .filter(|node_ref| matches!(gate_fn.gates[node_ref.id], AigNode::And2 { .. }))
        .count();

    let depth_stats = get_gate_depth(gate_fn, &live_nodes);
    let mut max_depth: usize = 0;
    for output in gate_fn.outputs.iter() {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            if let Some(depth) = depth_stats.ref_to_depth.get(&operand.node) {
                max_depth = std::cmp::max(max_depth, *depth);
            }
        }
    }

    let mut fanout_histogram: BTreeMap<usize, usize> = BTreeMap::new();
    for node_ref in live_nodes.iter() {
        // Skip literal nodes (AigRef 0/1).
        if matches!(gate_fn.gates[node_ref.id], AigNode::Literal { .. }) {
            continue;
        }
        let fanout = *id_to_use_count
            .get(node_ref)
            .expect("live node missing from use-count map");
        *fanout_histogram.entry(fanout).or_insert(0) += 1;
    }

    AigStats {
        and_nodes,
        max_depth,
        fanout_histogram,
    }
}

#[cfg(test)]
mod tests {
    use super::{get_level_critical_path_and_nodes, get_level_critical_path_ands};
    use crate::aig::gate;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::use_count::get_id_to_use_count;

    #[test]
    fn test_get_level_critical_path_and_nodes_returns_union_of_deepest_paths() {
        let mut builder = GateBuilder::new("f".to_string(), GateBuilderOptions::no_opt());
        let a = builder.add_input("a".to_string(), 1);
        let b = builder.add_input("b".to_string(), 1);

        let shared = builder.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        let left = builder.add_and_binary(shared, *a.get_lsb(0));
        let right = builder.add_and_binary(shared, *b.get_lsb(0));
        let shallow = builder.add_and_binary(*a.get_lsb(0), *a.get_lsb(0));

        builder.add_output("o0".to_string(), left.into());
        builder.add_output("o1".to_string(), right.into());
        builder.add_output("o2".to_string(), shallow.into());
        let gate_fn = builder.build();

        let live_nodes: Vec<gate::AigRef> = get_id_to_use_count(&gate_fn).keys().copied().collect();
        let got = get_level_critical_path_and_nodes(&gate_fn, &live_nodes);
        let got_with_depth = get_level_critical_path_ands(&gate_fn, &live_nodes);

        assert_eq!(
            got.into_iter()
                .map(|node_ref| node_ref.id)
                .collect::<Vec<_>>(),
            vec![shared.node.id, left.node.id, right.node.id]
        );
        assert_eq!(got_with_depth.depth_aig_nodes, 2);
        assert_eq!(got_with_depth.and_nodes.len(), 3);
    }
}
