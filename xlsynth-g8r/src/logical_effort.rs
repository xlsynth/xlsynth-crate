// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigNode, AigRef, GateFn};
use crate::get_summary_stats::get_gate_depth;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellKind {
    Inverter,
    Nand,
}

/// Returns the logical effort for a given cell kind and input count.
pub fn get_logical_effort(kind: CellKind, input_count: usize) -> f64 {
    match kind {
        CellKind::Inverter => {
            assert_eq!(input_count, 1);
            1.0
        }
        CellKind::Nand => {
            assert!(input_count >= 1);
            (input_count as f64 + 2.0) / 3.0
        }
    }
}

/// Returns the parasitic delay for a given cell kind and input count.
pub fn get_parasitic_delay(kind: CellKind, input_count: usize) -> f64 {
    match kind {
        CellKind::Inverter => {
            assert_eq!(input_count, 1);
            1.0
        }
        CellKind::Nand => {
            assert!(input_count >= 1);
            input_count as f64
        }
    }
}

/// Computes the critical path delay for the given GateFn using logical effort.
/// Every And2 node is treated as a NAND for logical effort purposes.
pub fn compute_critical_path_delay(gate_fn: &GateFn) -> f64 {
    // Build a fanout map: AigRef -> fanout count
    let mut fanout_map: HashMap<AigRef, usize> = HashMap::new();
    for (_id, node) in gate_fn.gates.iter().enumerate() {
        if let AigNode::And2 { a, b, .. } = node {
            *fanout_map.entry(a.node).or_insert(0) += 1;
            *fanout_map.entry(b.node).or_insert(0) += 1;
        }
    }
    // Get the deepest path in the AIG
    let depth_stats = get_gate_depth(
        gate_fn,
        &(0..gate_fn.gates.len())
            .map(|id| AigRef { id })
            .collect::<Vec<_>>(),
    );
    let path = &depth_stats.deepest_path;
    if path.is_empty() {
        return 0.0;
    }
    let mut delay = 0.0;
    // Sum delay for every And2 node in the path, using fanout as electrical effort
    // (h)
    for &node_ref in path.iter().rev() {
        let node = &gate_fn.gates[node_ref.id];
        if let AigNode::And2 { .. } = node {
            let g = get_logical_effort(CellKind::Nand, 2);
            let p = get_parasitic_delay(CellKind::Nand, 2);
            // Use fanout as electrical effort (h), default to 1 if not found/0
            let h = fanout_map.get(&node_ref).copied().unwrap_or(1).max(1) as f64;
            delay += g * h + p;
        }
    }
    delay
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn test_path_delay_nand_chain() {
        // Build a 3-stage NAND chain: ai -> y -> z -> bo, all with i_aux as the other
        // input
        let mut gb = GateBuilder::new("test".to_string(), GateBuilderOptions::no_opt());
        let ai = gb.add_input("ai".to_string(), 1);
        let i_aux = gb.add_input("i_aux".to_string(), 1);

        let a0 = ai.get_lsb(0);
        let i_aux0 = i_aux.get_lsb(0);

        let y = gb.add_nand_binary(*a0, *i_aux0);
        let z = gb.add_nand_binary(y, *i_aux0);
        let bo = gb.add_nand_binary(z, *i_aux0);

        gb.add_output("bo".to_string(), bo.into());
        let gate_fn = gb.build();

        let delay = compute_critical_path_delay(&gate_fn);
        assert!((delay - 10.0).abs() < 1e-3, "delay: {}", delay);
    }

    #[test]
    fn test_path_delay_with_fanout() {
        // Build a circuit:
        // a = NAND(ai, i_aux)
        // b1 = NAND(a, i_aux)
        // b2 = NAND(a, i_aux)
        // output = NAND(b1, i_aux)
        // a drives both b1 and b2 (fanout = 2)
        // The critical path is a -> b1 -> output
        use crate::gate_builder::{GateBuilder, GateBuilderOptions};
        let mut gb = GateBuilder::new("fanout_test".to_string(), GateBuilderOptions::no_opt());
        let ai = gb.add_input("ai".to_string(), 1);
        let i_aux = gb.add_input("i_aux".to_string(), 1);
        let a0 = ai.get_lsb(0);
        let i_aux0 = i_aux.get_lsb(0);

        let a = gb.add_nand_binary(*a0, *i_aux0);
        let b1 = gb.add_nand_binary(a, *i_aux0);
        let _b2 = gb.add_nand_binary(a, *i_aux0);
        let output = gb.add_nand_binary(b1, *i_aux0);

        gb.add_output("out".to_string(), output.into());
        let gate_fn = gb.build();

        let delay = compute_critical_path_delay(&gate_fn);
        // Expectation:
        // The critical path is a (fanout=2) -> b1 (fanout=1) -> output (fanout=1)
        // For each NAND: g = 1.333..., p = 2
        // a: h = 2, delay = 1.333...*2 + 2 = 4.666...
        // b1: h = 1, delay = 1.333...*1 + 2 = 3.333...
        // output: h = 1, delay = 1.333...*1 + 2 = 3.333...
        // Total: 4.666... + 3.333... + 3.333... = 11.333...
        assert!((delay - 11.333).abs() < 1e-3, "delay: {}", delay);
    }
}
