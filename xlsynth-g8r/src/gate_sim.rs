// SPDX-License-Identifier: Apache-2.0

#![allow(unused)]

use std::{collections::HashMap, iter::zip};

use bitvec::vec::BitVec;
use xlsynth::IrBits;

use crate::gate::{AigNode, AigOperand, GateFn};

pub struct GateSimResult {
    pub outputs: Vec<IrBits>,
    pub tagged_values: Option<HashMap<String, bool>>,
    pub all_values: Option<BitVec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Collect {
    None,
    Tagged,
    All,
}

pub fn eval(gate_fn: &GateFn, inputs: &[IrBits], collect: Collect) -> GateSimResult {
    assert_eq!(inputs.len(), gate_fn.inputs.len());

    let mut env: HashMap<AigOperand, bool> = HashMap::new();
    let mut tagged_values = if collect == Collect::Tagged {
        Some(HashMap::new())
    } else {
        None
    };
    let mut all_values = if collect == Collect::All {
        Some(BitVec::repeat(false, gate_fn.gates.len()))
    } else {
        None
    };

    // Seed the env with the input operands.
    for (input, gate_fn_input) in zip(inputs, gate_fn.inputs.iter()) {
        for (bit_index, operand) in gate_fn_input.bit_vector.iter_lsb_to_msb().enumerate() {
            let bit_value = input.get_bit(bit_index).unwrap();
            env.insert(*operand, bit_value);
        }
    }

    // Traverse the AIG nodes in post-order. Post-order traversal ensures that
    // when we visit a node, its children (inputs) have already been visited
    // and their values computed and stored in the environment.
    for operand in gate_fn.post_order_operands(true) {
        // Calculate the final boolean value for this specific operand
        let final_value: bool = match gate_fn.get(operand.node) {
            AigNode::Input { .. } => {
                // Get the base value seeded for the non-negated input operand
                let base_operand = AigOperand {
                    node: operand.node,
                    negated: false,
                };
                let base_value = *env.get(&base_operand).expect(&format!(
                    "Input base value should be seeded: {:?}",
                    base_operand
                ));
                // Apply negation only if this operand requires it
                if operand.negated {
                    !base_value
                } else {
                    base_value
                }
            }
            AigNode::Literal(value) => {
                // Apply negation if the operand using the literal is negated
                if operand.negated {
                    !*value
                } else {
                    *value
                }
            }
            AigNode::And2 { a, b, tags } => {
                // Get the final values already computed for the input operands
                let a_value = *env.get(a).expect(&format!(
                    "Input operand 'a' value not found for AND node: {:?}",
                    a
                ));
                let b_value = *env.get(b).expect(&format!(
                    "Input operand 'b' value not found for AND node: {:?}",
                    b
                ));
                // Compute the AND result
                let and_result = a_value && b_value;
                if let Some(tags) = tags
                    && collect == Collect::Tagged
                {
                    for tag in tags.iter() {
                        let map: &mut HashMap<String, bool> = tagged_values.as_mut().unwrap();
                        map.insert(tag.clone(), and_result);
                    }
                }
                // Apply negation if the operand using the AND result is negated
                if operand.negated {
                    !and_result
                } else {
                    and_result
                }
            }
        };
        if let Some(ref mut all_values) = all_values {
            all_values.set(operand.node.id, final_value);
        }
        // Store the correctly computed final value for this operand
        env.insert(operand, final_value);
    }

    let mut outputs: Vec<IrBits> = Vec::new();
    for output in gate_fn.outputs.iter() {
        // Collect all the bits for the output value and then unflatten them into form.
        let mut bits: Vec<bool> = Vec::new();
        for bit in output.bit_vector.iter_lsb_to_msb() {
            bits.push(*env.get(bit).unwrap());
        }
        outputs.push(crate::ir_value_utils::ir_bits_from_lsb_is_0(&bits));
    }
    GateSimResult {
        outputs,
        tagged_values,
        all_values,
    }
}

#[cfg(test)]
mod tests {
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    use super::*;

    #[test]
    fn test_simple_bitwise_and() {
        let mut gb = GateBuilder::new("simple_bitwise_and".to_string(), GateBuilderOptions::opt());
        let input_a = gb.add_input("a".to_string(), 4);
        let input_b = gb.add_input("b".to_string(), 4);
        let and_node = gb.add_and_vec(&input_a, &input_b);
        gb.add_output("out".to_string(), and_node);
        let gate_fn = gb.build();

        // We push the whole truth table through in one gate sim.
        let inputs = vec![
            IrBits::make_ubits(4, 0b0011).unwrap(),
            IrBits::make_ubits(4, 0b0101).unwrap(),
        ];
        let result = eval(&gate_fn, &inputs, Collect::None);
        assert_eq!(result.outputs, vec![IrBits::make_ubits(4, 0b0001).unwrap()]);
    }

    #[test]
    fn test_simple_bitwise_not() {
        let mut gb = GateBuilder::new("simple_bitwise_not".to_string(), GateBuilderOptions::opt());
        let input_a = gb.add_input("a".to_string(), 4);
        let not_node = gb.add_not_vec(&input_a);
        gb.add_output("out".to_string(), not_node);
        let gate_fn = gb.build();

        // Test NOT(0b1010) -> 0b0101
        let inputs = vec![IrBits::make_ubits(4, 0b1010).unwrap()];
        let result = eval(&gate_fn, &inputs, Collect::None);
        assert_eq!(result.outputs, vec![IrBits::make_ubits(4, 0b0101).unwrap()]);
    }

    #[test]
    fn test_simple_bitwise_nand() {
        let mut gb = GateBuilder::new("simple_bitwise_nand".to_string(), GateBuilderOptions::opt());
        let input_a = gb.add_input("a".to_string(), 4);
        let input_b = gb.add_input("b".to_string(), 4);
        // NAND is NOT(AND(a, b))
        let and_node = gb.add_and_vec(&input_a, &input_b);
        let nand_node = gb.add_not_vec(&and_node);
        gb.add_output("out".to_string(), nand_node);
        let gate_fn = gb.build();

        // Test NAND(0b0011, 0b0101) -> NOT(0b0001) -> 0b1110
        let inputs = vec![
            IrBits::make_ubits(4, 0b0011).unwrap(),
            IrBits::make_ubits(4, 0b0101).unwrap(),
        ];
        let result = eval(&gate_fn, &inputs, Collect::None);
        assert_eq!(result.outputs, vec![IrBits::make_ubits(4, 0b1110).unwrap()]);
    }

    #[test]
    fn test_simple_bitwise_or() {
        let mut gb = GateBuilder::new("simple_bitwise_or".to_string(), GateBuilderOptions::opt());
        let input_a = gb.add_input("a".to_string(), 4);
        let input_b = gb.add_input("b".to_string(), 4);
        let or_node = gb.add_or_vec(&input_a, &input_b);
        gb.add_output("out".to_string(), or_node);
        let gate_fn = gb.build();

        // Test OR(0b0011, 0b0101) -> 0b0111
        let inputs = vec![
            IrBits::make_ubits(4, 0b0011).unwrap(),
            IrBits::make_ubits(4, 0b0101).unwrap(),
        ];
        let result = eval(&gate_fn, &inputs, Collect::None);
        assert_eq!(result.outputs, vec![IrBits::make_ubits(4, 0b0111).unwrap()]);
    }

    #[test]
    fn test_simple_bitwise_xor() {
        let mut gb = GateBuilder::new(
            "simple_bitwise_xor".to_string(),
            GateBuilderOptions {
                fold: true,
                hash: false,
            },
        );
        let input_a = gb.add_input("a".to_string(), 4);
        let input_b = gb.add_input("b".to_string(), 4);
        let xor_node = gb.add_xor_vec(&input_a, &input_b);
        gb.add_output("out".to_string(), xor_node);
        let gate_fn = gb.build();

        // Test XOR(0b0011, 0b0101) -> 0b0110
        let inputs = vec![
            IrBits::make_ubits(4, 0b0011).unwrap(),
            IrBits::make_ubits(4, 0b0101).unwrap(),
        ];
        let result = eval(&gate_fn, &inputs, Collect::None);
        assert_eq!(result.outputs, vec![IrBits::make_ubits(4, 0b0110).unwrap()]);
    }
}
