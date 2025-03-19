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

    // Add all the outputs to the module.
    for output in gate_fn.outputs.iter() {
        for (i, bit) in output.bit_vector.iter_lsb_to_msb().enumerate() {
            let name = format!("{}_{}", output.name, i);
            let logic_ref = module.add_output(name.as_str(), &bit_type);
            gate_ref_to_vast.insert(bit.node, logic_ref);
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
                let gate_name = format!("gate_{}", i);
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
            _ => todo!("Unhandled gate for emission: {:?}", gate),
        }
    }

    file.emit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore]
    #[test]
    fn test_emit_inverter() {
        let mut g8_builder = gate::GateBuilder::new("my_inverter".to_string(), false);
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
  wire gate_0;
  assign gate_0 = 1'b0;
  wire gate_1;
  assign gate_1 = 1'b1;
  wire gate_3;
  assign gate_3 = ~i_0;
endmodule
"
        );
    }
}
