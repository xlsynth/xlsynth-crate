// SPDX-License-Identifier: Apache-2.0

//! Shared PIR block evaluation helpers for random-block fuzz targets.

use std::collections::BTreeMap;

use xlsynth_pir::IrBits;
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::{Block, Fn, Node, NodeGraph, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_eval::{self, EvalObserver, FnEvalResult, SelectEvent};
use xlsynth_pir::ir_utils::remap_payload_with;
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;

#[derive(Debug, Clone, Copy)]
struct RegisterWriteRefs {
    arg: NodeRef,
    load_enable: Option<NodeRef>,
    reset: Option<NodeRef>,
}

pub struct ObservedBlockCycle {
    pub outputs: Vec<IrValue>,
    pub next_state: Vec<IrValue>,
    pub node_values: Vec<Option<IrValue>>,
}

struct NodeValues(Vec<Option<IrValue>>);

struct CycleEvalFn {
    function: Fn,
    block_to_function: Vec<NodeRef>,
}

impl EvalObserver for NodeValues {
    fn on_select(&mut self, _event: SelectEvent) {
        // Consumers classify events from node values and original block
        // metadata.
    }

    fn on_node_value(&mut self, node_ref: NodeRef, _id: usize, value: &IrValue) {
        self.0[node_ref.index] = Some(value.clone());
    }
}

/// Evaluates a cycle once, retaining operand values for behavioral coverage.
/// Register Q is substituted with explicit state; reset/enable priority follows
/// the block semantics independently of either RTL or gate lowering.
pub fn evaluate_block_cycle_observed(
    block: &Block,
    inputs: &[IrValue],
    state: &[IrValue],
) -> ObservedBlockCycle {
    let projection = cycle_eval_fn(block, state);
    let cycle_fn = &projection.function;
    let mut observer = NodeValues(vec![None; cycle_fn.nodes.len()]);
    // EvalObserver reports computed values, not parameter and unit nodes.
    for (index, node) in cycle_fn.nodes.iter().enumerate() {
        match &node.payload {
            NodePayload::Param => {
                let position = cycle_fn.param_index(NodeRef { index }).unwrap();
                observer.0[index] = Some(inputs[position].clone());
            }
            NodePayload::Nil => observer.0[index] = Some(IrValue::make_tuple(&[])),
            _ => { /* Computed values arrive through the observer. */ }
        }
    }
    let result = ir_eval::eval_fn_with_observer(cycle_fn, inputs, Some(&mut observer));
    assert!(
        matches!(result, FnEvalResult::Success(_)),
        "block evaluation failed: {result:?}\n{block}"
    );
    let value = |node: NodeRef| {
        observer.0[projection.block_to_function[node.index].index]
            .as_ref()
            .expect("all block nodes evaluated")
    };
    let outputs = block_output_refs(block)
        .iter()
        .map(|node| value(*node).clone())
        .collect();
    let writes = collect_register_writes(block);
    let next_state = block
        .registers
        .iter()
        .enumerate()
        .map(|(index, register)| {
            let Some(write) = writes.get(&register.name) else {
                // A register with no write holds its state.
                return state[index].clone();
            };
            let mut next = value(write.arg).clone();
            if write
                .load_enable
                .is_some_and(|enable| !bool_value(value(enable)))
            {
                next = state[index].clone();
            }
            if let (Some(reset), Some(reset_value), Some(reset_metadata)) = (
                write.reset,
                register.reset_value.as_ref(),
                block.reset.as_ref(),
            ) && bool_value(value(reset)) ^ reset_metadata.active_low
            {
                next = reset_value.clone();
            }
            next
        })
        .collect();
    let node_values = projection
        .block_to_function
        .iter()
        .map(|node| observer.0[node.index].clone())
        .collect();
    ObservedBlockCycle {
        outputs,
        next_state,
        node_values,
    }
}

/// Returns visible block output values in port declaration order.
pub fn block_output_refs(block: &Block) -> Vec<NodeRef> {
    block
        .output_ports()
        .map(|port| block.output_value(port))
        .collect()
}

/// Returns visible block output types in port declaration order.
pub fn block_output_types(block: &Block) -> Vec<&Type> {
    block
        .output_ports()
        .map(|port| block.port_type(port))
        .collect()
}

/// Evaluates visible outputs of one combinational block sample.
pub fn evaluate_block_outputs(block: &Block, inputs: &[IrValue], ir_text: &str) -> Vec<IrValue> {
    let mut function = combinational_block_to_fn(block).unwrap_or_else(|error| {
        panic!("combinational block projection failed:\n{ir_text}\n{error}")
    });
    let return_ref = function
        .ret_node_ref
        .expect("projection has a return value");
    let result = eval_ref(&mut function, return_ref, inputs, ir_text);
    if block.output_ports().count() == 1 {
        vec![result]
    } else {
        result
            .get_elements()
            .expect("multiple block outputs are returned as a tuple")
    }
}

/// Evaluates visible outputs and committed next state for one block cycle.
pub fn evaluate_block_cycle(
    block: &Block,
    inputs: &[IrValue],
    state: &[IrValue],
    ir_text: &str,
) -> (Vec<IrValue>, Vec<IrValue>) {
    let output_refs = block_output_refs(block);
    let writes = collect_register_writes(block);
    let CycleEvalFn {
        function: mut cycle_fn,
        block_to_function,
    } = cycle_eval_fn(block, state);
    if output_refs.is_empty() {
        // Keep outputless samples in the PIR evaluator portion of the property.
        let ret_ref = cycle_fn
            .ret_node_ref
            .expect("generated block should have a return node");
        let _ = eval_ref(&mut cycle_fn, ret_ref, inputs, ir_text);
    }
    let outputs = output_refs
        .into_iter()
        .map(|output_ref| {
            eval_ref(
                &mut cycle_fn,
                block_to_function[output_ref.index],
                inputs,
                ir_text,
            )
        })
        .collect();
    let mut next_state = Vec::with_capacity(block.registers.len());

    for (register_index, register) in block.registers.iter().enumerate() {
        let Some(write) = writes.get(&register.name) else {
            next_state.push(state[register_index].clone());
            continue;
        };
        let mut next_value = eval_ref(
            &mut cycle_fn,
            block_to_function[write.arg.index],
            inputs,
            ir_text,
        );
        if let Some(load_enable_ref) = write.load_enable
            && !bool_value(&eval_ref(
                &mut cycle_fn,
                block_to_function[load_enable_ref.index],
                inputs,
                ir_text,
            ))
        {
            next_value = state[register_index].clone();
        }
        if let (Some(reset_ref), Some(reset_value), Some(reset_metadata)) = (
            write.reset,
            register.reset_value.as_ref(),
            block.reset.as_ref(),
        ) {
            let reset_signal = bool_value(&eval_ref(
                &mut cycle_fn,
                block_to_function[reset_ref.index],
                inputs,
                ir_text,
            ));
            let reset_asserted = if reset_metadata.active_low {
                !reset_signal
            } else {
                reset_signal
            };
            if reset_asserted {
                next_value = reset_value.clone();
            }
        }
        next_state.push(next_value);
    }

    (outputs, next_state)
}

/// Flattens one generated PIR value into the packed LSB-first AIG convention.
pub fn flatten_value(value: &IrValue, ty: &Type) -> IrBits {
    let mut bits = Vec::with_capacity(ty.bit_count());
    flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut bits)
        .expect("generated value should match its PIR type");
    IrBits::from_lsb_is_0(&bits)
}

fn collect_register_writes(block: &Block) -> BTreeMap<String, RegisterWriteRefs> {
    block
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::RegisterWrite {
                arg,
                register,
                load_enable,
                reset,
            } => Some((
                register.clone(),
                RegisterWriteRefs {
                    arg: *arg,
                    load_enable: *load_enable,
                    reset: *reset,
                },
            )),
            _ => None,
        })
        .collect()
}

/// Projects ports/state into a temporary function while preserving an observer
/// map back to the block, whose ports need not occupy a function-style prefix.
fn cycle_eval_fn(block: &Block, state: &[IrValue]) -> CycleEvalFn {
    let state_by_register: BTreeMap<&str, &IrValue> = block
        .registers
        .iter()
        .zip(state)
        .map(|(register, value)| (register.name.as_str(), value))
        .collect();
    let mut order = vec![NodeRef { index: 0 }];
    order.extend(block.input_ports());
    order.extend(block.node_refs().into_iter().filter(|node| {
        node.index != 0 && !matches!(block.get_node(*node).payload, NodePayload::InputPort { .. })
    }));
    let mut block_to_function = vec![NodeRef { index: 0 }; block.nodes.len()];
    for (index, source) in order.iter().enumerate() {
        block_to_function[source.index] = NodeRef { index };
    }
    let nodes = order
        .iter()
        .map(|source| {
            let node = block.get_node(*source);
            Node {
                payload: remap_payload_with(&node.payload, |(_, dependency)| {
                    block_to_function[dependency.index]
                }),
                ..node.clone()
            }
        })
        .collect();
    let graph = NodeGraph {
        name: block.name.clone(),
        nodes,
        outer_attrs: block.outer_attrs.clone(),
        inner_attrs: block.inner_attrs.clone(),
    };
    let mut result = Fn {
        graph,
        params: block.input_ports().collect(),
        ret_ty: Type::nil(),
        ret_node_ref: None,
    };
    for node in &mut result.nodes {
        match &node.payload {
            NodePayload::InputPort { name, .. } => {
                node.name = Some(name.clone());
                node.payload = NodePayload::Param;
            }
            NodePayload::OutputPort { .. } => {
                node.payload = NodePayload::Tuple(Vec::new());
            }
            NodePayload::RegisterRead { register } => {
                node.payload = NodePayload::Literal(
                    (*state_by_register
                        .get(register.as_str())
                        .expect("generated register read should have state"))
                    .clone(),
                );
            }
            NodePayload::RegisterWrite { .. } => {
                node.ty = Type::nil();
                node.payload = NodePayload::Nil;
            }
            _ => {
                // Ordinary combinational nodes retain their original indices.
            }
        }
    }
    let ret = NodeRef {
        index: result.nodes.len(),
    };
    let text_id = result
        .nodes
        .iter()
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0)
        + 1;
    result.nodes.push(Node {
        text_id,
        name: None,
        ty: Type::nil(),
        payload: NodePayload::Tuple(Vec::new()),
        pos: None,
    });
    result.ret_node_ref = Some(ret);
    CycleEvalFn {
        function: result,
        block_to_function,
    }
}

fn eval_ref(cycle_fn: &mut Fn, node_ref: NodeRef, inputs: &[IrValue], ir_text: &str) -> IrValue {
    cycle_fn.ret_ty = cycle_fn.get_node(node_ref).ty.clone();
    cycle_fn.ret_node_ref = Some(node_ref);
    match ir_eval::eval_fn(cycle_fn, inputs) {
        FnEvalResult::Success(success) => success.value,
        failure @ FnEvalResult::Failure(_) => {
            panic!("block PIR evaluation failed:\nIR:\n{ir_text}\nresult={failure:?}")
        }
    }
}

fn bool_value(value: &IrValue) -> bool {
    value
        .to_bool()
        .expect("generated reset/load-enable value should be bits[1]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn cycle_oracle_remaps_header_order_and_preserves_observer_indices() {
        let source = r#"block registered(b: bits[8], out: bits[8], clk: clock, a: bits[8]) {
  reg value(bits[8])
  q: bits[8] = register_read(register=value, id=1)
  a: bits[8] = input_port(name=a, id=2)
  b: bits[8] = input_port(name=b, id=3)
  sum: bits[8] = add(a, b, id=4)
  update: () = register_write(sum, register=value, id=5)
  out: () = output_port(q, name=out, id=6)
}"#;
        let block = Parser::new(source).parse_block().unwrap();
        let inputs = [
            IrValue::make_ubits(8, 9).unwrap(),
            IrValue::make_ubits(8, 3).unwrap(),
        ];
        let state = [IrValue::make_ubits(8, 16).unwrap()];
        let observed = evaluate_block_cycle_observed(&block, &inputs, &state);
        let (outputs, next_state) = evaluate_block_cycle(&block, &inputs, &state, source);
        assert_eq!(observed.outputs, outputs);
        assert_eq!(outputs, state);
        assert_eq!(observed.next_state, next_state);
        assert_eq!(next_state, [IrValue::make_ubits(8, 12).unwrap()]);
        assert_eq!(
            observed.node_values[block.get_input_port("a").unwrap().index],
            Some(inputs[1].clone())
        );
        assert_eq!(
            observed.node_values[block.get_input_port("b").unwrap().index],
            Some(inputs[0].clone())
        );
    }

    #[test]
    fn combinational_oracle_keeps_one_tuple_output_whole() {
        let source = r#"block aggregate(data: (bits[8], bits[3]), result: (bits[8], bits[3])) {
  data: (bits[8], bits[3]) = input_port(name=data, id=1)
  result: () = output_port(data, name=result, id=2)
}"#;
        let block = Parser::new(source).parse_block().unwrap();
        let inputs = [IrValue::make_tuple(&[
            IrValue::make_ubits(8, 27).unwrap(),
            IrValue::make_ubits(3, 5).unwrap(),
        ])];
        assert_eq!(evaluate_block_outputs(&block, &inputs, source), inputs);
    }
}
