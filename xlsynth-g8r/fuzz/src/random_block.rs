// SPDX-License-Identifier: Apache-2.0

//! Shared PIR block evaluation helpers for random-block fuzz targets.

use std::collections::BTreeMap;

use xlsynth::{IrBits, IrValue};
use xlsynth_pir::ir::{BlockMetadata, Fn, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_eval::{self, FnEvalResult};
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;

#[derive(Debug, Clone, Copy)]
struct RegisterWriteRefs {
    arg: NodeRef,
    load_enable: Option<NodeRef>,
    reset: Option<NodeRef>,
}

/// Returns visible block output node references in metadata order.
pub fn block_output_refs(block: &Fn, metadata: &BlockMetadata) -> Vec<NodeRef> {
    let ret_ref = block
        .ret_node_ref
        .expect("generated block should have a return node");
    match metadata.output_names.len() {
        0 => Vec::new(),
        1 => vec![ret_ref],
        _ => {
            let NodePayload::Tuple(outputs) = &block.get_node(ret_ref).payload else {
                panic!("generated multi-output block should return a tuple");
            };
            outputs.clone()
        }
    }
}

/// Returns visible block output types in metadata order.
pub fn block_output_types<'a>(block: &'a Fn, metadata: &BlockMetadata) -> Vec<&'a Type> {
    match metadata.output_names.len() {
        0 => Vec::new(),
        1 => vec![&block.ret_ty],
        _ => {
            let Type::Tuple(types) = &block.ret_ty else {
                panic!("generated multi-output block should return a tuple type");
            };
            types.iter().map(|ty| &**ty).collect()
        }
    }
}

/// Evaluates visible outputs of one combinational block sample.
pub fn evaluate_block_outputs(
    block: &Fn,
    metadata: &BlockMetadata,
    inputs: &[IrValue],
    ir_text: &str,
) -> Vec<IrValue> {
    let output_refs = block_output_refs(block, metadata);
    if output_refs.is_empty() {
        // Keep outputless samples in the PIR evaluator portion of the property.
        let ret_ref = block
            .ret_node_ref
            .expect("generated block should have a return node");
        let mut eval_fn = block.clone();
        let _ = eval_ref(&mut eval_fn, ret_ref, inputs, ir_text);
    }
    output_refs
        .into_iter()
        .map(|output_ref| {
            let mut eval_fn = block.clone();
            eval_ref(&mut eval_fn, output_ref, inputs, ir_text)
        })
        .collect()
}

/// Evaluates visible outputs and committed next state for one block cycle.
pub fn evaluate_block_cycle(
    block: &Fn,
    metadata: &BlockMetadata,
    inputs: &[IrValue],
    state: &[IrValue],
    ir_text: &str,
) -> (Vec<IrValue>, Vec<IrValue>) {
    let output_refs = block_output_refs(block, metadata);
    let writes = collect_register_writes(block);
    let mut cycle_fn = cycle_eval_fn(block, metadata, state);
    if output_refs.is_empty() {
        // Keep outputless samples in the PIR evaluator portion of the property.
        let ret_ref = block
            .ret_node_ref
            .expect("generated block should have a return node");
        let _ = eval_ref(&mut cycle_fn, ret_ref, inputs, ir_text);
    }
    let outputs = output_refs
        .into_iter()
        .map(|output_ref| eval_ref(&mut cycle_fn, output_ref, inputs, ir_text))
        .collect();
    let mut next_state = Vec::with_capacity(metadata.registers.len());

    for (register_index, register) in metadata.registers.iter().enumerate() {
        let Some(write) = writes.get(&register.name) else {
            next_state.push(state[register_index].clone());
            continue;
        };
        let mut next_value = eval_ref(&mut cycle_fn, write.arg, inputs, ir_text);
        if let Some(load_enable_ref) = write.load_enable
            && !bool_value(&eval_ref(
                &mut cycle_fn,
                load_enable_ref,
                inputs,
                ir_text,
            ))
        {
            next_value = state[register_index].clone();
        }
        if let (Some(reset_ref), Some(reset_value), Some(reset_metadata)) =
            (write.reset, register.reset_value.as_ref(), metadata.reset.as_ref())
        {
            let reset_signal = bool_value(&eval_ref(&mut cycle_fn, reset_ref, inputs, ir_text));
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

fn collect_register_writes(block: &Fn) -> BTreeMap<String, RegisterWriteRefs> {
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

fn cycle_eval_fn(block: &Fn, metadata: &BlockMetadata, state: &[IrValue]) -> Fn {
    let state_by_register: BTreeMap<&str, &IrValue> = metadata
        .registers
        .iter()
        .zip(state)
        .map(|(register, value)| (register.name.as_str(), value))
        .collect();
    let mut result = block.clone();
    for node in &mut result.nodes {
        match &node.payload {
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
            _ => {}
        }
    }
    result
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
