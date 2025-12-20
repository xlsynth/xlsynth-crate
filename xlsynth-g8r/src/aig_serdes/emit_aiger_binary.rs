// SPDX-License-Identifier: Apache-2.0
//
// --- AIGER (binary) emitter.
//
// Emits a `GateFn` into the compact binary AIGER "aig" format, suitable for
// consumption by tools that accept AIGER as a general interchange format.
//
// Notes:
// - We currently support purely combinational designs (L == 0).
// - The output uses a deterministic remapping of nodes:
//   - var 0 is constant-false
//   - vars 1..=I are the input bits in the same stable order as the GateFn
//     input bit-vectors (lsb->msb, in input list order)
//   - AND nodes are then assigned in a post-order traversal from outputs
// - We emit outputs and (optional) symbol table as ASCII, followed by the
//   binary delta-encoded AND section per the AIGER spec.

use crate::aig::gate::{self, AigNode, AigOperand};
use std::collections::HashMap;
use std::fmt::Write as _;

fn encode_u32_as_aiger_varint(mut x: u32, out: &mut Vec<u8>) {
    while x & !0x7f != 0 {
        out.push(((x & 0x7f) as u8) | 0x80);
        x >>= 7;
    }
    out.push((x & 0x7f) as u8);
}

fn is_const_literal(gf: &gate::GateFn, op: AigOperand) -> Option<bool> {
    match gf.get(op.node) {
        AigNode::Literal(v) => Some(*v),
        _ => None,
    }
}

fn operand_to_literal_with_var_map(
    gf: &gate::GateFn,
    var_map: &HashMap<usize, u32>,
    op: AigOperand,
) -> Result<u32, String> {
    if let Some(val) = is_const_literal(gf, op) {
        let base = if val { 1u32 } else { 0u32 };
        return Ok(base ^ (op.negated as u32));
    }
    let var = *var_map
        .get(&op.node.id)
        .ok_or_else(|| format!("missing var mapping for node id {}", op.node.id))?;
    Ok((var << 1) ^ (op.negated as u32))
}

/// Emits the supplied `GateFn` into binary AIGER ("aig") format.
///
/// Returns the serialized bytes on success.
pub fn emit_aiger_binary(gate_fn: &gate::GateFn, include_symbols: bool) -> Result<Vec<u8>, String> {
    // We only handle purely combinational circuits today.
    let latch_count = 0u32;

    // Stable input-bit ordering.
    let mut input_node_ids: Vec<usize> = Vec::new();
    for input in &gate_fn.inputs {
        for bit in input.bit_vector.iter_lsb_to_msb() {
            input_node_ids.push(bit.node.id);
        }
    }
    let input_count = input_node_ids.len() as u32;

    // Determine AND nodes in a deterministic, dependency-respecting order by
    // using a post-order traversal from outputs.
    //
    // Note: `post_order_operands()` deduplicates by *operand* (node + negation),
    // but we need a node-only ordering for AIGER encoding. So we explicitly
    // deduplicate by node id while preserving first-seen order.
    let mut and_node_ids: Vec<usize> = Vec::new();
    let mut seen_and_node_ids: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for op in gate_fn.post_order_operands(true) {
        if matches!(gate_fn.get(op.node), AigNode::And2 { .. })
            && seen_and_node_ids.insert(op.node.id)
        {
            and_node_ids.push(op.node.id);
        }
    }
    let and_count = and_node_ids.len() as u32;

    // Build var mapping: constant false is var 0; inputs are 1..=I; AND nodes
    // are then I+1..=I+A in the order selected above.
    let mut var_map: HashMap<usize, u32> = HashMap::new();
    for (i, node_id) in input_node_ids.iter().enumerate() {
        var_map.insert(*node_id, (i as u32) + 1);
    }
    for (i, node_id) in and_node_ids.iter().enumerate() {
        var_map.insert(*node_id, input_count + (i as u32) + 1);
    }

    // M is maximum variable index referenced. With our remapping, that's I + A.
    let max_var_index = input_count + and_count;

    let output_count = gate_fn
        .outputs
        .iter()
        .map(|o| o.bit_vector.get_bit_count())
        .sum::<usize>() as u32;

    let mut bytes: Vec<u8> = Vec::new();

    // Header.
    let header = format!(
        "aig {} {} {} {} {}\n",
        max_var_index, input_count, latch_count, output_count, and_count
    );
    bytes.extend_from_slice(header.as_bytes());

    // Outputs are emitted as ASCII literal lines, even in the binary AIGER
    // variant (per the AIGER spec / ABC expectations).
    for output in &gate_fn.outputs {
        for bit in output.bit_vector.iter_lsb_to_msb() {
            let lit = operand_to_literal_with_var_map(gate_fn, &var_map, *bit)?;
            let line = format!("{lit}\n");
            bytes.extend_from_slice(line.as_bytes());
        }
    }

    // AND section: binary delta encoding. Each AND is encoded as two varints:
    //   delta0 = lhs - rhs0
    //   delta1 = rhs0 - rhs1
    // with rhs0 >= rhs1 and both < lhs.
    for node_id in &and_node_ids {
        let lhs_var = *var_map
            .get(node_id)
            .ok_or_else(|| format!("missing var mapping for AND node id {}", node_id))?;
        let lhs_lit = lhs_var << 1;
        let (mut rhs0, mut rhs1) = match &gate_fn.gates[*node_id] {
            AigNode::And2 { a, b, .. } => (
                operand_to_literal_with_var_map(gate_fn, &var_map, *a)?,
                operand_to_literal_with_var_map(gate_fn, &var_map, *b)?,
            ),
            _ => return Err("internal error: and_node_ids contained non-AND node".to_string()),
        };
        if rhs1 > rhs0 {
            std::mem::swap(&mut rhs0, &mut rhs1);
        }
        debug_assert!(rhs0 < lhs_lit);
        debug_assert!(rhs1 <= rhs0);
        let delta0 = lhs_lit
            .checked_sub(rhs0)
            .ok_or_else(|| format!("invalid AIGER encoding: lhs {lhs_lit} < rhs0 {rhs0}"))?;
        let delta1 = rhs0
            .checked_sub(rhs1)
            .ok_or_else(|| format!("invalid AIGER encoding: rhs0 {rhs0} < rhs1 {rhs1}"))?;
        encode_u32_as_aiger_varint(delta0, &mut bytes);
        encode_u32_as_aiger_varint(delta1, &mut bytes);
    }

    if include_symbols {
        // Symbol table is ASCII, written after the binary AND section.
        let mut sym = String::new();
        let mut inp_idx = 0usize;
        for input in &gate_fn.inputs {
            let bit_cnt = input.bit_vector.get_bit_count();
            for bit_i in 0..bit_cnt {
                let name = if bit_cnt == 1 {
                    input.name.clone()
                } else {
                    format!("{}_{}", input.name, bit_i)
                };
                writeln!(&mut sym, "i{} {}", inp_idx, name).unwrap();
                inp_idx += 1;
            }
        }
        let mut out_idx = 0usize;
        for output in &gate_fn.outputs {
            let bit_cnt = output.bit_vector.get_bit_count();
            for bit_i in 0..bit_cnt {
                let name = if bit_cnt == 1 {
                    output.name.clone()
                } else {
                    format!("{}_{}", output.name, bit_i)
                };
                writeln!(&mut sym, "o{} {}", out_idx, name).unwrap();
                out_idx += 1;
            }
        }
        bytes.extend_from_slice(sym.as_bytes());
    }

    // Comment section (ASCII).
    bytes.extend_from_slice(b"c\n");
    bytes.extend_from_slice(b"generated by xlsynth-g8r emit_aiger_binary\n");

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    #[test]
    fn test_emit_aiger_binary_simple_and() {
        // Build: o = i0 & i1
        let mut gb = GateBuilder::new("and_fn".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1);
        let i1 = gb.add_input("i1".to_string(), 1);
        let and_out = gb.add_and_binary(*i0.get_lsb(0), *i1.get_lsb(0));
        gb.add_output("o".to_string(), gate::AigBitVector::from_bit(and_out));
        let gf = gb.build();

        let bytes = emit_aiger_binary(&gf, false).unwrap();

        // Expect:
        //   header: aig 3 2 0 1 1\n
        //   output: 6\n
        //   and section:
        //     lhs=6, rhs0=4, rhs1=2 -> delta0=2, delta1=2 => bytes 0x02 0x02
        //   comment
        let mut expected: Vec<u8> = Vec::new();
        expected.extend_from_slice(b"aig 3 2 0 1 1\n");
        expected.extend_from_slice(b"6\n");
        expected.extend_from_slice(&[0x02, 0x02]);
        expected.extend_from_slice(b"c\n");
        expected.extend_from_slice(b"generated by xlsynth-g8r emit_aiger_binary\n");

        assert_eq!(bytes, expected);
    }
}
