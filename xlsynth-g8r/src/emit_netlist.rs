// SPDX-License-Identifier: Apache-2.0

//! Uses the xlsynth VAST APIs to build a netlist verilog representation of the
//! given gates.

use crate::gate;
use std::collections::HashMap;
use xlsynth::vast;

#[allow(dead_code)]
pub fn emit_netlist(name: &str, gate_fn: &gate::GateFn) -> String {
    let mut file = vast::VastFile::new(vast::VastFileType::Verilog);
    let mut module = file.add_module(name);
    let bit_type = file.make_bit_vector_type(1, false);

    let mut gate_ref_to_vast: HashMap<gate::AigRef, vast::LogicRef> = HashMap::new();

    // Add all the inputs to the module.
    for input in gate_fn.inputs.iter() {
        for (i, bit) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            let name = format!("{}_{}", input.name, i);
            let logic_ref = module.add_input(name.as_str(), &bit_type);
            gate_ref_to_vast.insert(bit.node, logic_ref);
        }
    }

    let mut output_operand_to_logic_ref = HashMap::new();

    // Add all the outputs to the module.
    for output in gate_fn.outputs.iter() {
        for (i, bit) in output.bit_vector.iter_lsb_to_msb().enumerate() {
            let name = format!("{}_{}", output.name, i);
            let logic_ref = module.add_output(name.as_str(), &bit_type);
            output_operand_to_logic_ref.insert(bit, logic_ref.clone());
        }
    }

    let bit_type = file.make_bit_vector_type(1, false);

    for (i, gate) in gate_fn.gates.iter().enumerate() {
        let gate_ref = gate::AigRef { id: i };
        match gate {
            gate::AigNode::Input { .. } => {
                // These are emitted as part of the module signature's input/output list.
                continue;
            }
            gate::AigNode::Literal(value) => {
                let gate_name = format!("G{}", i);
                let this_wire = module.add_wire(gate_name.as_str(), &bit_type);
                let value_str = if *value { "bits[1]:1" } else { "bits[1]:0" };
                let literal_gate = file
                    .make_literal(value_str, &xlsynth::ir_value::IrFormatPreference::Binary)
                    .unwrap();
                module.add_member_continuous_assignment(
                    file.make_continuous_assignment(&this_wire.to_expr(), &literal_gate),
                );
                gate_ref_to_vast.insert(gate_ref, this_wire);
            }
            gate::AigNode::And2 { a, b, tags: _tags } => {
                let gate_name = format!("G{}", i);
                let this_wire = module.add_wire(gate_name.as_str(), &bit_type);
                let a_expr = gate_ref_to_vast[&a.node].to_expr();
                let b_expr = gate_ref_to_vast[&b.node].to_expr();
                let rhs = file.make_bitwise_and(&a_expr, &b_expr);
                let assignment = file.make_continuous_assignment(&this_wire.to_expr(), &rhs);
                // TODO(cdleary): 2025-03-20 Emit tags as comments above the assignment.
                module.add_member_continuous_assignment(assignment);
                gate_ref_to_vast.insert(gate_ref, this_wire);
            }
        }
    }

    for output in gate_fn.outputs.iter() {
        for bit in output.bit_vector.iter_lsb_to_msb() {
            // Get the gate under `bit`, potentially invert it in assigning it to the
            // output.
            let gate_ref = bit.node;
            let gate_wire = gate_ref_to_vast.get(&gate_ref).unwrap();
            let output_expr = output_operand_to_logic_ref[&bit].to_expr();
            let rhs_expr = if bit.negated {
                file.make_not(&gate_wire.to_expr())
            } else {
                gate_wire.to_expr()
            };
            let assignment = file.make_continuous_assignment(&output_expr, &rhs_expr);
            module.add_member_continuous_assignment(assignment);
        }
    }

    file.emit()
}

#[cfg(test)]
mod tests {
    use crate::gate_builder::GateBuilder;

    use super::*;

    use pretty_assertions::assert_eq;

    #[test]
    fn test_emit_inverter() {
        let mut g8_builder = GateBuilder::new("my_inverter".to_string(), false);
        let i_ref = g8_builder.add_input("i".to_string(), 1);
        assert_eq!(i_ref.get_bit_count(), 1);
        let gate_ref = g8_builder.add_not(*i_ref.get_lsb(0));
        g8_builder.add_output("o".to_string(), gate::AigBitVector::from_bit(gate_ref));

        // Now emit the netlist for this built gate network.
        let netlist = emit_netlist("my_inverter", &g8_builder.build());
        assert_eq!(
            netlist,
            "module my_inverter(
  input wire i_0,
  output wire o_0
);
  wire G0;
  assign G0 = 1'b0;
  assign o_0 = ~i_0;
endmodule
"
        );
    }

    #[test]
    fn test_emit_and_gate() {
        let mut g8_builder = GateBuilder::new("my_and_gate".to_string(), false);
        let i_ref = g8_builder.add_input("i".to_string(), 1);
        let j_ref = g8_builder.add_input("j".to_string(), 1);
        let gate_ref = g8_builder.add_and_binary(*i_ref.get_lsb(0), *j_ref.get_lsb(0));
        g8_builder.add_output("o".to_string(), gate::AigBitVector::from_bit(gate_ref));

        // Now emit the netlist for this built gate network.
        let netlist = emit_netlist("my_and_gate", &g8_builder.build());
        assert_eq!(
            netlist,
            "module my_and_gate(
  input wire i_0,
  input wire j_0,
  output wire o_0
);
  wire G0;
  assign G0 = 1'b0;
  wire G3;
  assign G3 = i_0 & j_0;
  assign o_0 = G3;
endmodule
"
        );
    }
}
