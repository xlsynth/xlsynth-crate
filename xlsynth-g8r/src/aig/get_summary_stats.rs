// SPDX-License-Identifier: Apache-2.0

use crate::aig::fanout::fanout_histogram;
use crate::aig::gate::{self, AigNode};
use crate::aig::topo::topo_sort_refs;
use crate::use_count::get_id_to_use_count;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct SummaryStats {
    pub live_nodes: usize,
    pub deepest_path: usize,
    pub fanout_histogram: BTreeMap<usize, usize>,
}

// Add structured return type for gate depth info.
#[derive(Debug)]
pub struct GateDepthStats {
    pub depth_to_count: HashMap<usize, usize>,
    pub deepest_path: Vec<gate::AigRef>,
    pub ref_to_depth: HashMap<gate::AigRef, usize>,
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
            &AigNode::Literal(_) => {
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
            AigNode::Input { .. } | AigNode::Literal(_) => 0,
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
