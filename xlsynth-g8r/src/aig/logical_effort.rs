// SPDX-License-Identifier: Apache-2.0

use crate::aig::get_summary_stats::get_gate_depth;
use crate::aig::{AigNode, AigRef, GateFn};
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

    // Make sure literal node does not have a meaningful fanout.
    fanout_map.remove(&AigRef { id: 0 });

    fanout_map
}

pub struct Options {
    pub input_pin_capacitance: f64,
    pub output_pin_capacitance: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            input_pin_capacitance: 1.0,
            output_pin_capacitance: 1.0,
        }
    }
}
/// We allow non-snake-case names in this function just to match the conventions
/// of logical effort equations which use capital letters for paths and lower
/// case letters for individual gates.
#[allow(non_snake_case)]
pub fn compute_logical_effort_min_delay(gate_fn: &GateFn, options: &Options) -> f64 {
    let id_to_use_count: HashMap<AigRef, usize> = get_id_to_use_count(gate_fn);
    let live_nodes: Vec<AigRef> = id_to_use_count.keys().cloned().collect();
    let depth_stats = get_gate_depth(gate_fn, &live_nodes);
    let path: Vec<AigRef> = depth_stats.deepest_path;
    // Filter out input pins from the path.
    let path: Vec<AigRef> = path
        .into_iter()
        .filter(|node| !matches!(gate_fn.gates[node.id], AigNode::Input { .. }))
        .collect();

    // Path logical effort.
    let N: f64 = path.len() as f64;
    let G: f64 = (4.0f64 / 3.0f64).powf(N);
    // Path branch effort.
    let fanout_map: HashMap<AigRef, usize> = get_fanout_map(gate_fn);
    let mut B: f64 = 1.0;
    for node in path.iter() {
        let fanout = fanout_map.get(&node).unwrap_or(&1);
        log::trace!("node: {:?} fanout: {}", node, fanout);
        B *= *fanout as f64;
    }
    // Path electrical effort we assume the input capacitance matches the output
    // capacitance.
    let H: f64 = options.output_pin_capacitance / options.input_pin_capacitance;
    let F: f64 = G * B * H;
    let P: f64 = 2.0 * N;
    let min_delay: f64 = N * F.powf(1.0 / N) + P;
    log::trace!("path: {:?}", path);
    log::trace!("N: {} G: {} B: {} H: {} F: {} P: {}", N, G, B, H, F, P);
    min_delay
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_within,
        gate_builder::{GateBuilder, GateBuilderOptions},
        test_utils::{Opt, load_bf16_add_sample, load_bf16_mul_sample},
    };

    use super::*;

    /// Sample courtesy https://my.eng.utah.edu/~cs6710/slides/cs6710-log-effx2.pdf
    #[test]
    fn test_three_layer_and_graph() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut gb = GateBuilder::new(
            "test_three_layer_and_graph".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let i0 = gb.add_input("i0".to_string(), 1);
        let i1 = gb.add_input("i1".to_string(), 1);
        let i2 = gb.add_input("i2".to_string(), 1);
        let a0 = gb.add_and_vec(&i0, &i1);
        let b0 = gb.add_and_vec(&a0, &i2);
        let b1 = gb.add_and_vec(&a0, &i2);
        let c0 = gb.add_and_vec(&b0, &i2);
        let c1 = gb.add_and_vec(&b0, &i2);
        let c2 = gb.add_and_vec(&b0, &i2);
        log::info!("a0: {:?}", a0);
        log::info!("b0: {:?}", b0);
        log::info!("b1: {:?}", b1);
        log::info!("c0: {:?}", c0);
        log::info!("c1: {:?}", c1);
        log::info!("c2: {:?}", c2);
        gb.add_output("c0".to_string(), c0.into());
        gb.add_output("c1".to_string(), c1.into());
        gb.add_output("c2".to_string(), c2.into());
        let gate_fn = gb.build();
        let options = Options {
            input_pin_capacitance: 1.0,
            output_pin_capacitance: 4.5,
        };
        let min_delay = compute_logical_effort_min_delay(&gate_fn, &options);
        assert!((min_delay - 18.0).abs() < 0.01, "min_delay: {}", min_delay);
    }
}
