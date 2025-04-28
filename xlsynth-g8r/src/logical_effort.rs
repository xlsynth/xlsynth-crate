// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigRef, GateFn};
use crate::get_summary_stats::get_gate_depth;
use crate::use_count::get_id_to_use_count;
use std::collections::HashMap;

fn get_fanout_map(gate_fn: &GateFn) -> HashMap<AigRef, usize> {
    let mut fanout_map: HashMap<AigRef, usize> = HashMap::new();
    for node in gate_fn.gates.iter() {
        match node {
            AigNode::And2 { a, b, .. } => {
                *fanout_map.entry(a.node).or_insert(0) += 1;
                *fanout_map.entry(b.node).or_insert(0) += 1;
            }
            _ => {}
        }
    }
    fanout_map.insert(AigRef { id: 0 }, 0);
    fanout_map
}

/// We allow non-snake-case names in this function just to match the conventions
/// of logical effort equations which use capital letters for paths and lower
/// case letters for individual gates.
#[allow(non_snake_case)]
pub fn compute_logical_effort_min_delay(gate_fn: &GateFn) -> f64 {
    let id_to_use_count: HashMap<AigRef, usize> = get_id_to_use_count(gate_fn);
    let live_nodes: Vec<AigRef> = id_to_use_count.keys().cloned().collect();
    let depth_stats = get_gate_depth(gate_fn, &live_nodes);
    let path = depth_stats.deepest_path;
    // Path logical effort.
    let N: f64 = path.len() as f64;
    let G: f64 = (4.0f64 / 3.0f64).powf(N);
    // Path branch effort.
    let fanout_map: HashMap<AigRef, usize> = get_fanout_map(gate_fn);
    let B: f64 = path
        .iter()
        .map(|node| *fanout_map.get(node).unwrap_or(&1) as f64)
        .product::<f64>();
    // Path electrical effort we assume the input capacitance matches the output
    // capacitance.
    let H: f64 = 1.0;
    let F: f64 = G * B * H;
    let P: f64 = 2.0 * N;
    let min_delay: f64 = N * F.powf(1.0 / N) + P;
    min_delay
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{load_bf16_add_sample, load_bf16_mul_sample, Opt};

    use super::*;

    #[test]
    fn test_compute_logical_effort_min_delay_bf16_add() {
        let bf16_add = load_bf16_add_sample(Opt::Yes);
        let min_delay = compute_logical_effort_min_delay(&bf16_add.gate_fn);
        assert!(
            (min_delay - 568.07).abs() < 0.01,
            "min_delay: {}",
            min_delay
        );
    }

    #[test]
    fn test_compute_logical_effort_min_delay_bf16_mul() {
        let bf16_mul = load_bf16_mul_sample(Opt::Yes);
        let min_delay = compute_logical_effort_min_delay(&bf16_mul.gate_fn);
        assert!(
            (min_delay - 491.62).abs() < 0.01,
            "min_delay: {}",
            min_delay
        );
    }
}
