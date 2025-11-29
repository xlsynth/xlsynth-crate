// SPDX-License-Identifier: Apache-2.0

// --- AIGER (ASCII) emitter.
// Allows exporting a `GateFn` into the standard "aag" format so that
// downstream tools such as the ABC synthesis/verification suite can ingest it.
//
// This intentionally mirrors the public surface of `emit_netlist.rs` albeit
// with a far simpler implementation: we only support purely combinational
// designs (no flops/latches) and do not yet emit the compact binary `aig`
// variant – only the human-readable `aag` dialect.
//
// The mapping from `GateFn` to AIGER is largely mechanical:
//
//   • Gate index 0 is the dedicated constant-false node that `GateBuilder`
//     always creates.  In AIGER the constant-false literal is 0 and constant
//     true is literal 1 (the complement of literal 0).
//   • Each `Input` bit becomes an AIGER input literal (non-negated).
//   • Each `AigNode::And2` becomes an AIGER AND definition line.
//   • Output literals are produced directly, taking into account operand
//     negation.
//
// No additional optimization or topological re-ordering is required; AIGER
// tools tolerate arbitrary ordering as long as the header counts are
// consistent and every AND left-hand-side literal is unique.
//
// INVARIANTS asserted throughout help catch accidental inconsistencies early
// while still compiling with `--release` (they become no-ops outside of debug
// builds).

use crate::aig::gate::{self, AigNode, AigOperand};
use std::fmt::Write as _; // for write! macro on String // Needed for AigBitVector in tests

/// Converts the given operand into an AIGER literal.
///
/// A literal is `2 * var + negated`, where `var` is the variable index.  The
/// special variable 0 is the dedicated constant-false, therefore literal 0 is
/// constant false and literal 1 is constant true.
fn operand_to_literal(gf: &gate::GateFn, op: AigOperand) -> u32 {
    match &gf.gates[op.node.id] {
        AigNode::Literal(val) => {
            let base = if *val { 1 } else { 0 }; // constant true/false
            base ^ (op.negated as u32)
        }
        _ => {
            let mut lit = (op.node.id as u32) << 1;
            if op.negated {
                lit ^= 1;
            }
            lit
        }
    }
}

/// Emits the supplied `GateFn` into ASCII AIGER ("aag") format.
///
/// Returns the textual representation on success.
pub fn emit_aiger(gate_fn: &gate::GateFn, include_symbols: bool) -> Result<String, String> {
    // We only handle purely combinational circuits today.
    // Future work: support flops by mapping them onto AIGER latches.
    let latch_count = 0u32;

    // Count input bits and collect their gate IDs.
    let mut input_ids: Vec<usize> = Vec::new();
    for input in &gate_fn.inputs {
        for bit in input.bit_vector.iter_lsb_to_msb() {
            // Ensure the referenced node is indeed an Input.
            debug_assert!(matches!(gate_fn.get(bit.node), AigNode::Input { .. }));
            input_ids.push(bit.node.id);
        }
    }

    // Gather AND nodes (their IDs), skipping index 0 (constant) and any literals.
    let mut and_ids: Vec<usize> = Vec::new();
    for (idx, node) in gate_fn.gates.iter().enumerate() {
        if matches!(node, AigNode::And2 { .. }) {
            and_ids.push(idx);
        }
    }

    let max_var_index = input_ids
        .iter()
        .chain(and_ids.iter())
        .copied()
        .max()
        .unwrap_or(0) as u32; // `0` handles degenerate 0-gate case.

    let output_count = gate_fn
        .outputs
        .iter()
        .map(|o| o.bit_vector.get_bit_count())
        .sum::<usize>() as u32;

    let header = format!(
        "aag {} {} {} {} {}\n",
        max_var_index,   // M – maximum variable index referenced
        input_ids.len(), // I – #inputs
        latch_count,     // L – #latches (none)
        output_count,    // O – #outputs
        and_ids.len()    // A – #AND gates
    );

    let mut out = String::with_capacity(header.len() + 64 * (input_ids.len() + and_ids.len()));
    out.push_str(&header);

    // --- Inputs (one per line) ------------------------------------------------
    for id in &input_ids {
        write!(&mut out, "{}\n", (*id as u32) << 1).unwrap();
    }

    // --- Latches (none) -------------------------------------------------------
    // Nothing to emit.

    // --- Outputs --------------------------------------------------------------
    for output in &gate_fn.outputs {
        for bit in output.bit_vector.iter_lsb_to_msb() {
            let lit = operand_to_literal(gate_fn, *bit);
            write!(&mut out, "{}\n", lit).unwrap();
        }
    }

    // --- AND gates ------------------------------------------------------------
    // Emit in ascending gate ID order to keep things deterministic.
    for id in &and_ids {
        if let AigNode::And2 { a, b, .. } = &gate_fn.gates[*id] {
            let lhs = (*id as u32) << 1;
            let rhs0 = operand_to_literal(gate_fn, *a);
            let rhs1 = operand_to_literal(gate_fn, *b);
            write!(&mut out, "{} {} {}\n", lhs, rhs0, rhs1).unwrap();
        } else {
            unreachable!("and_ids filtered to AND nodes only");
        }
    }

    if include_symbols {
        // --- Symbol table -----------------------------------------------------
        // Inputs first.
        let mut inp_idx = 0usize;
        for input in &gate_fn.inputs {
            let bit_cnt = input.bit_vector.get_bit_count();
            for bit_i in 0..bit_cnt {
                let name = if bit_cnt == 1 {
                    input.name.clone()
                } else {
                    format!("{}_{}", input.name, bit_i)
                };
                write!(&mut out, "i{} {}\n", inp_idx, name).unwrap();
                inp_idx += 1;
            }
        }

        // Outputs.
        let mut out_idx = 0usize;
        for output in &gate_fn.outputs {
            let bit_cnt = output.bit_vector.get_bit_count();
            for bit_i in 0..bit_cnt {
                let name = if bit_cnt == 1 {
                    output.name.clone()
                } else {
                    format!("{}_{}", output.name, bit_i)
                };
                write!(&mut out, "o{} {}\n", out_idx, name).unwrap();
                out_idx += 1;
            }
        }
    }

    // --- Comment section ------------------------------------------------------
    out.push_str("c\n");
    writeln!(out, "generated by xlsynth-g8r emit_aiger").unwrap();

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use pretty_assertions::assert_eq;

    #[test]
    fn test_emit_aiger_simple_and() {
        // Build: o = i0 & i1
        let mut gb = GateBuilder::new("and_fn".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1);
        let i1 = gb.add_input("i1".to_string(), 1);
        let and_out = gb.add_and_binary(*i0.get_lsb(0), *i1.get_lsb(0));
        gb.add_output("o".to_string(), gate::AigBitVector::from_bit(and_out));
        let gf = gb.build();

        let aiger = emit_aiger(&gf, true).unwrap();

        // Expected structure:
        //   - 2 inputs (ids 1,2)
        //   - 1 AND gate (id 3)
        //   - 1 output
        // Header: aag 3 2 0 1 1
        // Inputs: 2,4
        // Output: 6 (literal of id 3)
        // AND:    6 2 4
        // Symbols: i0, i1, o

        let expected = "aag 3 2 0 1 1\n2\n4\n6\n6 2 4\ni0 i0\ni1 i1\no0 o\nc\ngenerated by xlsynth-g8r emit_aiger\n";

        // Because symbol table ordering is deterministic, we can compare.
        assert_eq!(aiger, expected);
    }
}
