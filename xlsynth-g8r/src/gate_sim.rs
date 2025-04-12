// SPDX-License-Identifier: Apache-2.0

#![allow(unused)]

use std::{collections::HashMap, iter::zip};

use xlsynth::IrBits;

use crate::gate::{AigNode, AigOperand, GateFn};

/// Creates an IrBits value from a slice where index 0 of the slice is the
/// least significant bit.
fn ir_bits_from_lsb_is_0(bits: &[bool]) -> IrBits {
    assert!(bits.len() <= 64);
    let mut u64_value = 0;
    for (i, bit) in bits.iter().enumerate() {
        if *bit {
            u64_value |= 1 << i;
        }
    }
    IrBits::make_ubits(bits.len(), u64_value).unwrap()
}

#[derive(Debug)]
struct GateSimResult {
    outputs: Vec<IrBits>,
    tagged_values: HashMap<String, bool>,
}

fn eval(gate_fn: &GateFn, inputs: &[IrBits], collect_tags: bool) -> GateSimResult {
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

    for operand in gate_fn.post_order(true).into_iter().rev() {
        let value_no_negate: bool = match gate_fn.get(operand.node) {
            AigNode::Input { .. } => {
                if operand.negated {
                    let non_negated = AigOperand {
                        node: operand.node,
                        negated: false,
                    };
                    !env.get(&non_negated).unwrap()
                } else {
                    *env.get(&operand).unwrap()
                }
            }
            AigNode::Literal(value) => *value,
            AigNode::And2 { a, b, tags } => {
                let a_value = env.get(a).unwrap();
                let b_value = env.get(b).unwrap();
                let result = *a_value && *b_value;
                if let Some(tags) = tags
                    && collect_tags
                {
                    for tag in tags.iter() {
                        tagged_values.insert(tag.clone(), result);
                    }
                }
                result
            }
        };
        let value = if operand.negated {
            !value_no_negate
        } else {
            value_no_negate
        };
        env.insert(operand, value);
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
    use crate::gate_builder::GateBuilder;

    use super::*;

    #[test]
    fn test_simple_bitwise_and() {
        let mut gb = GateBuilder::new("simple_bitwise_and".to_string(), true);
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
}
