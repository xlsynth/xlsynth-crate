// SPDX-License-Identifier: Apache-2.0

#![allow(unused)]

use std::{collections::HashMap, iter::zip};

use xlsynth::IrBits;

use crate::gate::{AigNode, AigOperand, GateFn};

/// Converts a `&[bool]` slice into an IR `Bits` value.
///
/// ```
/// use xlsynth::ir_value::IrFormatPreference;
/// use xlsynth::IrBits;
/// use xlsynth_g8r::gate_sim::ir_bits_from_lsb_is_0;
///
/// let bools = vec![true, false, true, false]; // LSB is bools[0]
/// let ir_bits: IrBits = ir_bits_from_lsb_is_0(&bools);
/// assert_eq!(ir_bits.to_string_fmt(IrFormatPreference::Binary, false), "0b101");
/// assert_eq!(ir_bits.get_bit_count(), 4);
/// assert_eq!(ir_bits.get_bit(0).unwrap(), true); // LSB
/// assert_eq!(ir_bits.get_bit(1).unwrap(), false);
/// assert_eq!(ir_bits.get_bit(2).unwrap(), true);
/// assert_eq!(ir_bits.get_bit(3).unwrap(), false); // MSB
/// ```
pub fn ir_bits_from_lsb_is_0(bits: &[bool]) -> IrBits {
    assert!(bits.len() <= 64);
    let mut u64_value = 0;
    for (i, bit) in bits.iter().enumerate() {
        if *bit {
            u64_value |= 1 << i;
        }
    }
    IrBits::make_ubits(bits.len(), u64_value).unwrap()
}

pub struct GateSimResult {
    pub outputs: Vec<IrBits>,
    pub tagged_values: HashMap<String, bool>,
}

pub fn eval(gate_fn: &GateFn, inputs: &[IrBits], collect_tags: bool) -> GateSimResult {
    assert_eq!(inputs.len(), gate_fn.inputs.len());

    let mut env: HashMap<AigOperand, bool> = HashMap::new();
    let mut tagged_values = HashMap::new();

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
    for operand in gate_fn.post_order(true) {
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
                    && collect_tags
                {
                    for tag in tags.iter() {
                        tagged_values.insert(tag.clone(), and_result);
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
        outputs.push(ir_bits_from_lsb_is_0(&bits));
    }
    GateSimResult {
        outputs,
        tagged_values,
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
        let result = eval(&gate_fn, &inputs, false);
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
        let result = eval(&gate_fn, &inputs, false);
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
        let result = eval(&gate_fn, &inputs, false);
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
        let result = eval(&gate_fn, &inputs, false);
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
        let result = eval(&gate_fn, &inputs, false);
        assert_eq!(result.outputs, vec![IrBits::make_ubits(4, 0b0110).unwrap()]);
    }
}
