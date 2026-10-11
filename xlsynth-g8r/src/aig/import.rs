// SPDX-License-Identifier: Apache-2.0

//! Import combinational graphs while preserving node provenance and tags.

use crate::aig::{AigBitVector, AigNode, AigOperand, GateFn};
use crate::gate_builder::GateBuilder;

/// Imports an AIG with explicit input bindings in dependency order.
pub(crate) fn import_gate_fn(
    source: &GateFn,
    input_bindings: &[AigBitVector],
    builder: &mut GateBuilder,
) -> Result<Vec<AigBitVector>, String> {
    if source.inputs.len() != input_bindings.len() {
        return Err(format!(
            "graph '{}' expects {} input port(s), but {} bindings were supplied",
            source.name,
            source.inputs.len(),
            input_bindings.len()
        ));
    }
    let mut imported_nodes = vec![None; source.gates.len()];
    for (input, binding) in source.inputs.iter().zip(input_bindings) {
        if input.get_bit_count() != binding.get_bit_count() {
            return Err(format!(
                "graph '{}' input '{}' expects {} bit(s), but its binding has {}",
                source.name,
                input.name,
                input.get_bit_count(),
                binding.get_bit_count()
            ));
        }
        for (source_bit, binding_bit) in input
            .bit_vector
            .iter_lsb_to_msb()
            .zip(binding.iter_lsb_to_msb())
        {
            if source_bit.negated {
                return Err(format!(
                    "graph '{}' input '{}' contains a negated input operand",
                    source.name, input.name
                ));
            }
            builder.add_pir_node_ids(
                binding_bit.node,
                source.gates[source_bit.node.id].get_pir_node_ids(),
            );
            imported_nodes[source_bit.node.id] = Some(*binding_bit);
        }
    }
    for operand in source.post_order_operands(false) {
        if imported_nodes[operand.node.id].is_some() {
            continue;
        }
        let source_node = &source.gates[operand.node.id];
        let imported = match source_node {
            AigNode::Input { name, .. } => {
                return Err(format!(
                    "graph '{}' references unbound input node '{}'",
                    source.name, name
                ));
            }
            AigNode::Literal { value, .. } => {
                if *value {
                    builder.get_true()
                } else {
                    builder.get_false()
                }
            }
            AigNode::And2 { a, b, tags, .. } => {
                let imported_a = mapped_operand(source, *a, &imported_nodes)?;
                let imported_b = mapped_operand(source, *b, &imported_nodes)?;
                let imported = builder.add_and_binary(imported_a, imported_b);
                if let Some(tags) = tags {
                    for tag in tags {
                        builder.add_tag(imported.node, tag.clone());
                    }
                }
                imported
            }
        };
        builder.add_pir_node_ids(imported.node, source_node.get_pir_node_ids());
        imported_nodes[operand.node.id] = Some(imported);
    }
    source
        .outputs
        .iter()
        .map(|output| {
            let imported_bits = output
                .bit_vector
                .iter_lsb_to_msb()
                .map(|bit| mapped_operand(source, *bit, &imported_nodes))
                .collect::<Result<Vec<AigOperand>, String>>()?;
            Ok(AigBitVector::from_lsb_is_index_0(&imported_bits))
        })
        .collect()
}

/// Restores the edge polarity after looking up an imported node.
fn mapped_operand(
    source: &GateFn,
    operand: AigOperand,
    imported_nodes: &[Option<AigOperand>],
) -> Result<AigOperand, String> {
    let base = imported_nodes[operand.node.id].ok_or_else(|| {
        format!(
            "graph '{}' does not have an imported value for AIG node {}",
            source.name, operand.node.id
        )
    })?;
    if operand.negated {
        Ok(base.negate())
    } else {
        Ok(base)
    }
}
