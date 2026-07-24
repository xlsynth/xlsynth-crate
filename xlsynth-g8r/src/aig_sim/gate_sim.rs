// SPDX-License-Identifier: Apache-2.0

#![allow(unused)]

use std::collections::HashMap;

use bitvec::vec::BitVec;
use xlsynth::IrBits;

use crate::aig::topo::postorder_for_aig_refs_node_only;
use crate::aig::{AigNode, AigOperand, AigRef, GateFn};

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
    AllWithInputs,
}

#[derive(Debug, Clone, Copy)]
struct DenseOperand {
    node_id: usize,
    negated: bool,
}

impl From<AigOperand> for DenseOperand {
    fn from(operand: AigOperand) -> Self {
        Self {
            node_id: operand.node.id,
            negated: operand.negated,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct InputBitBinding {
    input_index: usize,
    bit_index: usize,
    operand: DenseOperand,
}

#[derive(Debug, Clone, Copy)]
struct AndStep {
    output_node_id: usize,
    a: DenseOperand,
    b: DenseOperand,
}

#[derive(Debug, Clone, Copy)]
struct TaggedNode<'a> {
    node_id: usize,
    tags: &'a [String],
}

/// Reusable dense evaluator for scalar AIG simulation.
///
/// This prepares one output-reachable node order up front and stores one
/// canonical, non-negated value per AIG node. Repeated callers can reuse the
/// prepared bindings and value storage; the free [`eval`] function uses the
/// same implementation for one-shot calls.
pub struct PreparedGateSim<'a> {
    input_count: usize,
    input_bindings: Vec<InputBitBinding>,
    and_steps: Vec<AndStep>,
    output_bindings: Vec<Vec<DenseOperand>>,
    all_node_ids: Vec<usize>,
    all_with_input_node_ids: Vec<usize>,
    tagged_nodes: Vec<TaggedNode<'a>>,
    values: Vec<u8>,
}

impl<'a> PreparedGateSim<'a> {
    /// Prepares the output-reachable cone of `gate_fn` for repeated evaluation.
    pub fn new(gate_fn: &'a GateFn) -> Self {
        let output_refs = gate_fn
            .outputs
            .iter()
            .flat_map(|output| {
                output
                    .bit_vector
                    .iter_lsb_to_msb()
                    .map(|operand| operand.node)
            })
            .collect::<Vec<AigRef>>();
        let empty_cache: HashMap<AigRef, ()> = HashMap::new();
        let postorder =
            postorder_for_aig_refs_node_only(&output_refs, &gate_fn.gates, &empty_cache);
        let all_with_input_node_ids = postorder.iter().map(|node| node.id).collect::<Vec<_>>();
        let mut live_nodes = vec![false; gate_fn.gates.len()];
        for node in &postorder {
            live_nodes[node.id] = true;
        }

        let mut input_bindings = Vec::new();
        let mut seeded_input_nodes = vec![false; gate_fn.gates.len()];
        for (input_index, input) in gate_fn.inputs.iter().enumerate() {
            for (bit_index, operand) in input.bit_vector.iter_lsb_to_msb().enumerate() {
                if live_nodes[operand.node.id] {
                    input_bindings.push(InputBitBinding {
                        input_index,
                        bit_index,
                        operand: (*operand).into(),
                    });
                    seeded_input_nodes[operand.node.id] = true;
                }
            }
        }

        let mut values = vec![0u8; gate_fn.gates.len()];
        let mut and_steps = Vec::new();
        let mut all_node_ids = Vec::new();
        let mut tagged_nodes = Vec::new();
        for node_ref in postorder {
            match gate_fn.get(node_ref) {
                AigNode::Input { .. } => {
                    assert!(
                        seeded_input_nodes[node_ref.id],
                        "reachable input node %{} is not bound to a GateFn input",
                        node_ref.id
                    );
                }
                AigNode::Literal { value, .. } => {
                    values[node_ref.id] = u8::from(*value);
                    all_node_ids.push(node_ref.id);
                }
                AigNode::And2 { a, b, tags, .. } => {
                    and_steps.push(AndStep {
                        output_node_id: node_ref.id,
                        a: (*a).into(),
                        b: (*b).into(),
                    });
                    all_node_ids.push(node_ref.id);
                    if let Some(tags) = tags {
                        tagged_nodes.push(TaggedNode {
                            node_id: node_ref.id,
                            tags: tags.as_slice(),
                        });
                    }
                }
            }
        }

        let output_bindings = gate_fn
            .outputs
            .iter()
            .map(|output| {
                output
                    .bit_vector
                    .iter_lsb_to_msb()
                    .map(|operand| (*operand).into())
                    .collect()
            })
            .collect();

        Self {
            input_count: gate_fn.inputs.len(),
            input_bindings,
            and_steps,
            output_bindings,
            all_node_ids,
            all_with_input_node_ids,
            tagged_nodes,
            values,
        }
    }

    fn evaluate_values(&mut self, inputs: &[IrBits]) {
        assert_eq!(inputs.len(), self.input_count);
        for binding in &self.input_bindings {
            let bit_value = inputs[binding.input_index]
                .get_bit(binding.bit_index)
                .unwrap();
            self.values[binding.operand.node_id] =
                u8::from(bit_value) ^ u8::from(binding.operand.negated);
        }

        for step in &self.and_steps {
            let a_value = self.values[step.a.node_id] ^ u8::from(step.a.negated);
            let b_value = self.values[step.b.node_id] ^ u8::from(step.b.negated);
            self.values[step.output_node_id] = a_value & b_value;
        }
    }

    fn collect_outputs(&self) -> Vec<IrBits> {
        self.output_bindings
            .iter()
            .map(|bindings| {
                let bits_lsb_first = bindings
                    .iter()
                    .map(|binding| (self.values[binding.node_id] ^ u8::from(binding.negated)) != 0)
                    .collect::<Vec<bool>>();
                IrBits::from_lsb_is_0(&bits_lsb_first)
            })
            .collect()
    }

    fn collect_tagged_values(&self) -> HashMap<String, bool> {
        let mut tagged_values = HashMap::new();
        for tagged_node in &self.tagged_nodes {
            let value = self.values[tagged_node.node_id] != 0;
            for tag in tagged_node.tags {
                tagged_values.insert(tag.clone(), value);
            }
        }
        tagged_values
    }

    fn collect_all_values(&self, node_ids: &[usize]) -> BitVec {
        let mut all_values = BitVec::repeat(false, self.values.len());
        for &node_id in node_ids {
            all_values.set(node_id, self.values[node_id] != 0);
        }
        all_values
    }

    /// Evaluates one input sample and returns outputs in `GateFn` order.
    pub fn eval_outputs(&mut self, inputs: &[IrBits]) -> Vec<IrBits> {
        self.evaluate_values(inputs);
        self.collect_outputs()
    }

    /// Evaluates one input sample and optionally collects internal values.
    pub fn eval(&mut self, inputs: &[IrBits], collect: Collect) -> GateSimResult {
        self.evaluate_values(inputs);
        let outputs = self.collect_outputs();
        let tagged_values = match collect {
            Collect::Tagged => Some(self.collect_tagged_values()),
            _ => None,
        };
        let all_values = match collect {
            Collect::All => Some(self.collect_all_values(&self.all_node_ids)),
            Collect::AllWithInputs => Some(self.collect_all_values(&self.all_with_input_node_ids)),
            _ => None,
        };
        GateSimResult {
            outputs,
            tagged_values,
            all_values,
        }
    }
}

/// Evaluates one scalar AIG sample using the dense prepared evaluator.
pub fn eval(gate_fn: &GateFn, inputs: &[IrBits], collect: Collect) -> GateSimResult {
    let mut prepared = PreparedGateSim::new(gate_fn);
    prepared.eval(inputs, collect)
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

    #[test]
    fn test_prepared_gate_sim_reuses_state_across_repeated_samples() {
        let mut gb = GateBuilder::new(
            "prepared_repeated".to_string(),
            GateBuilderOptions {
                fold: true,
                hash: false,
            },
        );
        let input_a = gb.add_input("a".to_string(), 4);
        let input_b = gb.add_input("b".to_string(), 4);
        let xor = gb.add_xor_vec(&input_a, &input_b);
        let gated = gb.add_and_vec(&xor, &input_a);
        let literal = gb.add_literal(&IrBits::make_ubits(4, 0b0101).unwrap());
        let masked = gb.add_and_vec(&gated, &literal);
        let inverted = gb.add_not_vec(&gated);
        let _dead = gb.add_or_vec(&input_a, &input_b);
        gb.add_output("gated".to_string(), gated);
        gb.add_output("masked".to_string(), masked);
        gb.add_output("inverted".to_string(), inverted);
        gb.add_output("direct".to_string(), input_b);
        let gate_fn = gb.build();
        let samples = vec![
            vec![
                IrBits::make_ubits(4, 0b0000).unwrap(),
                IrBits::make_ubits(4, 0b1111).unwrap(),
            ],
            vec![
                IrBits::make_ubits(4, 0b0011).unwrap(),
                IrBits::make_ubits(4, 0b0101).unwrap(),
            ],
            vec![
                IrBits::make_ubits(4, 0b1010).unwrap(),
                IrBits::make_ubits(4, 0b0110).unwrap(),
            ],
        ];
        let expected_samples = vec![
            vec![
                IrBits::make_ubits(4, 0b0000).unwrap(),
                IrBits::make_ubits(4, 0b0000).unwrap(),
                IrBits::make_ubits(4, 0b1111).unwrap(),
                IrBits::make_ubits(4, 0b1111).unwrap(),
            ],
            vec![
                IrBits::make_ubits(4, 0b0010).unwrap(),
                IrBits::make_ubits(4, 0b0000).unwrap(),
                IrBits::make_ubits(4, 0b1101).unwrap(),
                IrBits::make_ubits(4, 0b0101).unwrap(),
            ],
            vec![
                IrBits::make_ubits(4, 0b1000).unwrap(),
                IrBits::make_ubits(4, 0b0000).unwrap(),
                IrBits::make_ubits(4, 0b0111).unwrap(),
                IrBits::make_ubits(4, 0b0110).unwrap(),
            ],
        ];
        let mut prepared = PreparedGateSim::new(&gate_fn);

        for (inputs, expected) in samples.into_iter().zip(expected_samples) {
            assert_eq!(prepared.eval_outputs(&inputs), expected);
            assert_eq!(eval(&gate_fn, &inputs, Collect::None).outputs, expected);
        }
    }

    #[test]
    fn test_prepared_gate_sim_handles_no_outputs() {
        let gate_fn = GateFn {
            name: "prepared_no_outputs".to_string(),
            inputs: vec![],
            outputs: vec![],
            gates: vec![],
        };
        let mut prepared = PreparedGateSim::new(&gate_fn);

        assert!(prepared.eval_outputs(&[]).is_empty());
        assert!(eval(&gate_fn, &[], Collect::None).outputs.is_empty());
    }

    #[test]
    fn test_dense_eval_collects_canonical_values_tags_and_inputs() {
        let mut gb = GateBuilder::new("dense_collect".to_string(), GateBuilderOptions::opt());
        let input_a = gb.add_input("a".to_string(), 1);
        let input_b = gb.add_input("b".to_string(), 1);
        let and = gb.add_and_vec(&input_a, &input_b);
        let a_node_id = input_a.iter_lsb_to_msb().next().unwrap().node.id;
        let b_node_id = input_b.iter_lsb_to_msb().next().unwrap().node.id;
        let and_node_id = and.iter_lsb_to_msb().next().unwrap().node.id;
        gb.add_tag(AigRef { id: and_node_id }, "and_value".to_string());
        let not_and = gb.add_not_vec(&and);
        gb.add_output("out".to_string(), not_and);
        let gate_fn = gb.build();
        let inputs = vec![
            IrBits::make_ubits(1, 1).unwrap(),
            IrBits::make_ubits(1, 1).unwrap(),
        ];

        let all = eval(&gate_fn, &inputs, Collect::All).all_values.unwrap();
        assert!(!all[a_node_id]);
        assert!(!all[b_node_id]);
        assert!(all[and_node_id]);

        let all_with_inputs = eval(&gate_fn, &inputs, Collect::AllWithInputs)
            .all_values
            .unwrap();
        assert!(all_with_inputs[a_node_id]);
        assert!(all_with_inputs[b_node_id]);
        assert!(all_with_inputs[and_node_id]);

        let tagged = eval(&gate_fn, &inputs, Collect::Tagged)
            .tagged_values
            .unwrap();
        assert_eq!(tagged.get("and_value"), Some(&true));
    }
}
