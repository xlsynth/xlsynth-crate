// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use crate::aig::gate::{self, AigNode, AigOperand};
use crate::aig_serdes::load_aiger::finish_loaded_gate_builder;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};

/// Presents the live graph to order-sensitive mappers in binary AIGER order,
/// preserving port widths and names without formatting or parsing a file.
pub(crate) fn canonical_mapping_graph(graph: &gate::GateFn) -> Result<gate::GateFn, String> {
    let layout = CanonicalAigerLayout::new(graph);
    let mut gb = GateBuilder::new(graph.name.clone(), GateBuilderOptions::no_opt());
    let mut vars = Vec::with_capacity(layout.max_var_index as usize + 1);
    vars.push(gb.get_false());
    for input in &graph.inputs {
        let bits = gb.add_input(input.name.clone(), input.get_bit_count());
        vars.extend(bits.iter_lsb_to_msb().copied());
    }
    let operand = |vars: &[AigOperand], lit: u32| {
        let mut op = vars[(lit >> 1) as usize];
        op.negated ^= lit & 1 != 0;
        op
    };
    for id in &layout.and_node_ids {
        let (_, a, b) = layout.sorted_and_literals(graph, *id)?;
        vars.push(gb.add_and_binary(operand(&vars, a), operand(&vars, b)));
    }
    for output in &graph.outputs {
        let bits = output
            .bit_vector
            .iter_lsb_to_msb()
            .map(|op| {
                layout
                    .operand_to_literal(graph, *op)
                    .map(|lit| operand(&vars, lit))
            })
            .collect::<Result<Vec<_>, _>>()?;
        gb.add_output(
            output.name.clone(),
            gate::AigBitVector::from_lsb_is_index_0(&bits),
        );
    }
    Ok(finish_loaded_gate_builder(gb))
}

/// Canonical variable layout shared by ASCII and binary AIGER emission.
pub(super) struct CanonicalAigerLayout {
    pub input_count: u32,
    pub and_node_ids: Vec<usize>,
    pub and_count: u32,
    pub max_var_index: u32,
    pub output_count: u32,
    var_map: HashMap<usize, u32>,
}

impl CanonicalAigerLayout {
    /// Builds the stable AIGER numbering used by both output dialects.
    pub fn new(gate_fn: &gate::GateFn) -> Self {
        let mut input_node_ids: Vec<usize> = Vec::new();
        for input in &gate_fn.inputs {
            for bit in input.bit_vector.iter_lsb_to_msb() {
                debug_assert!(matches!(gate_fn.get(bit.node), AigNode::Input { .. }));
                input_node_ids.push(bit.node.id);
            }
        }
        let input_count = input_node_ids.len() as u32;

        // AIGER ANDs must appear after their dependencies. Number the live
        // graph from the outputs so ASCII and binary emission expose the same
        // graph presentation to downstream heuristic tools.
        let mut and_node_ids: Vec<usize> = Vec::new();
        let mut seen_and_node_ids: HashSet<usize> = HashSet::new();
        for op in gate_fn.post_order_operands(true) {
            if matches!(gate_fn.get(op.node), AigNode::And2 { .. })
                && seen_and_node_ids.insert(op.node.id)
            {
                and_node_ids.push(op.node.id);
            }
        }
        let and_count = and_node_ids.len() as u32;

        let mut var_map: HashMap<usize, u32> = HashMap::new();
        for (i, node_id) in input_node_ids.iter().enumerate() {
            var_map.insert(*node_id, (i as u32) + 1);
        }
        for (i, node_id) in and_node_ids.iter().enumerate() {
            var_map.insert(*node_id, input_count + (i as u32) + 1);
        }

        let max_var_index = input_count + and_count;
        let output_count = gate_fn
            .outputs
            .iter()
            .map(|o| o.bit_vector.get_bit_count())
            .sum::<usize>() as u32;

        Self {
            input_count,
            and_node_ids,
            and_count,
            max_var_index,
            output_count,
            var_map,
        }
    }

    /// Converts an AIG operand into the canonical AIGER literal.
    pub fn operand_to_literal(
        &self,
        gate_fn: &gate::GateFn,
        op: AigOperand,
    ) -> Result<u32, String> {
        if let AigNode::Literal { value, .. } = gate_fn.get(op.node) {
            let base = if *value { 1u32 } else { 0u32 };
            return Ok(base ^ (op.negated as u32));
        }

        let var = *self
            .var_map
            .get(&op.node.id)
            .ok_or_else(|| format!("missing var mapping for node id {}", op.node.id))?;
        Ok((var << 1) ^ (op.negated as u32))
    }

    /// Returns one canonical AND definition with descending RHS literals.
    pub fn sorted_and_literals(
        &self,
        gate_fn: &gate::GateFn,
        node_id: usize,
    ) -> Result<(u32, u32, u32), String> {
        let lhs_var = *self
            .var_map
            .get(&node_id)
            .ok_or_else(|| format!("missing var mapping for AND node id {}", node_id))?;
        let lhs_lit = lhs_var << 1;
        let (mut rhs0, mut rhs1) = match &gate_fn.gates[node_id] {
            AigNode::And2 { a, b, .. } => (
                self.operand_to_literal(gate_fn, *a)?,
                self.operand_to_literal(gate_fn, *b)?,
            ),
            _ => return Err("internal error: and_node_ids contained non-AND node".to_string()),
        };
        if rhs1 > rhs0 {
            std::mem::swap(&mut rhs0, &mut rhs1);
        }
        Ok((lhs_lit, rhs0, rhs1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_serdes::{
        emit_aiger_binary::emit_aiger_binary, load_aiger_binary::load_aiger_binary,
    };

    #[test]
    fn mapping_view_matches_serialized_scalar_transition_order() -> Result<(), String> {
        let mut gb = GateBuilder::new("core".into(), GateBuilderOptions::no_opt());
        let a = *gb.add_input("i0".into(), 1).get_lsb(0);
        let b = *gb.add_input("i1".into(), 1).get_lsb(0);
        let _dead = gb.add_and_binary(a, b);
        let x = gb.add_and_binary(a.negate(), b);
        let y = gb.add_and_binary(b.negate(), a);
        let z = gb.add_and_binary(y, x);
        gb.add_output("o0".into(), z.negate().into());
        gb.add_output("o1".into(), x.into());
        let g = gb.build();
        let reference =
            load_aiger_binary(&emit_aiger_binary(&g, true)?, GateBuilderOptions::no_opt())?.gate_fn;
        let got = canonical_mapping_graph(&g)?;
        assert_eq!(got.gates, reference.gates);
        assert_eq!(
            got.outputs
                .iter()
                .flat_map(|o| o.bit_vector.iter_lsb_to_msb())
                .collect::<Vec<_>>(),
            reference
                .outputs
                .iter()
                .flat_map(|o| o.bit_vector.iter_lsb_to_msb())
                .collect::<Vec<_>>()
        );
        Ok(())
    }
}
