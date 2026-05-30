// SPDX-License-Identifier: Apache-2.0

//! Conversion from a sequential gate-level representation into XLS block IR.

use std::collections::HashMap;

use xlsynth_pir::ir::{
    self, BlockMetadata, FileTable, MemberType, Node, NodePayload, NodeRef, Package, PackageMember,
    Register, Type,
};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_verify;

use crate::aig::SequentialGateFn;
use crate::aig_serdes::gate2ir::gate_fn_to_xlsynth_ir;

/// Lifts a `SequentialGateFn` into a package whose top member is an XLS block.
///
/// The transition logic is shared with GateFn-to-function lifting. Register
/// next-state behavior is already encoded in the transition D outputs, so
/// register writes emitted here are unconditional. An `initial_value` has no
/// faithful XLS block equivalent without inventing reset behavior.
pub fn sequential_gate_fn_to_pir_block_package(
    design: &SequentialGateFn,
    package_name: &str,
) -> Result<Package, String> {
    design
        .validate()
        .map_err(|e| format!("sequential2ir: invalid SequentialGateFn: {e}"))?;
    if let Some(register) = design
        .registers
        .iter()
        .find(|register| register.initial_value.is_some())
    {
        return Err(format!(
            "sequential2ir: register '{}' has an initial value, which cannot be represented as XLS block reset behavior",
            register.name
        ));
    }

    let lifted_ir = gate_fn_to_xlsynth_ir(
        &design.transition,
        package_name,
        &design.transition.get_flat_type(),
    )
    .map_err(|e| format!("sequential2ir: failed to lift transition GateFn: {e}"))?;
    let mut parser = Parser::new(&lifted_ir.to_string());
    let lifted_package = parser
        .parse_and_validate_package()
        .map_err(|e| format!("sequential2ir: failed to parse lifted transition IR: {e}"))?;
    let mut block = lifted_package
        .get_top_fn()
        .ok_or_else(|| "sequential2ir: lifted transition package has no top function".to_string())?
        .clone();
    block.name = design.name.clone();

    let original_params = block.params.clone();
    let transition_return_tuple = (design.transition.outputs.len() > 1)
        .then_some(block.ret_node_ref)
        .flatten();
    let transition_outputs =
        split_transition_return_values(&block, design.transition.outputs.len())?;
    replace_register_q_params(&mut block, design, &original_params)?;
    block.params = design
        .inputs
        .iter()
        .map(|id| original_params[id.index()].clone())
        .collect();

    let mut next_text_id = block
        .nodes
        .iter()
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0)
        + 1;
    for register in &design.registers {
        let d_ref = transition_outputs[register.d.index()];
        block.nodes.push(Node {
            text_id: next_text_id,
            name: Some(format!("{}_d", register.name)),
            ty: Type::nil(),
            payload: NodePayload::RegisterWrite {
                arg: d_ref,
                register: register.name.clone(),
                load_enable: None,
                reset: None,
            },
            pos: None,
        });
        next_text_id += 1;
    }

    let external_outputs = design
        .outputs
        .iter()
        .map(|id| transition_outputs[id.index()])
        .collect::<Vec<NodeRef>>();
    if let Some(tuple_ref) = transition_return_tuple {
        block.nodes[tuple_ref.index].payload = NodePayload::Nil;
        block.nodes[tuple_ref.index].ty = Type::nil();
    }
    set_block_return_value(&mut block, &external_outputs, &mut next_text_id);

    let output_names = design
        .outputs
        .iter()
        .map(|id| design.transition.outputs[id.index()].name.clone())
        .collect::<Vec<String>>();
    let output_port_ids = output_names
        .iter()
        .map(|name| {
            let id = next_text_id;
            next_text_id += 1;
            (name.clone(), id)
        })
        .collect::<HashMap<String, usize>>();
    let metadata = BlockMetadata {
        clock_port_name: design.clock.as_ref().map(|clock| clock.name.clone()),
        input_port_ids: block
            .params
            .iter()
            .map(|param| (param.name.clone(), param.id.get_wrapped_id()))
            .collect(),
        output_port_ids,
        output_names,
        reset: None,
        registers: design
            .registers
            .iter()
            .map(|register| Register {
                name: register.name.clone(),
                ty: Type::Bits(design.transition.inputs[register.q.index()].get_bit_count()),
                reset_value: None,
            })
            .collect(),
        instantiations: vec![],
    };
    let package = Package {
        name: package_name.to_string(),
        file_table: FileTable::new(),
        members: vec![PackageMember::Block {
            func: block,
            metadata,
        }],
        top: Some((design.name.clone(), MemberType::Block)),
    };
    ir_verify::verify_package(&package)
        .map_err(|e| format!("sequential2ir: generated invalid block package: {e}"))?;
    Ok(package)
}

fn split_transition_return_values(
    transition: &ir::Fn,
    output_count: usize,
) -> Result<Vec<NodeRef>, String> {
    match output_count {
        0 => Ok(vec![]),
        1 => transition
            .ret_node_ref
            .map(|ret| vec![ret])
            .ok_or_else(|| "sequential2ir: lifted transition has no return value".to_string()),
        _ => {
            let ret = transition.ret_node_ref.ok_or_else(|| {
                "sequential2ir: lifted transition has no tuple return value".to_string()
            })?;
            let NodePayload::Tuple(outputs) = &transition.nodes[ret.index].payload else {
                return Err(
                    "sequential2ir: lifted transition with multiple outputs does not return a tuple"
                        .to_string(),
                );
            };
            if outputs.len() != output_count {
                return Err(format!(
                    "sequential2ir: lifted transition returns {} values but expected {}",
                    outputs.len(),
                    output_count
                ));
            }
            Ok(outputs.clone())
        }
    }
}

fn replace_register_q_params(
    block: &mut ir::Fn,
    design: &SequentialGateFn,
    original_params: &[ir::Param],
) -> Result<(), String> {
    for register in &design.registers {
        let param = original_params.get(register.q.index()).ok_or_else(|| {
            format!(
                "sequential2ir: register '{}' Q references missing transition parameter {}",
                register.name,
                register.q.index()
            )
        })?;
        let node = block
            .nodes
            .iter_mut()
            .find(|node| node.payload == NodePayload::GetParam(param.id))
            .ok_or_else(|| {
                format!(
                    "sequential2ir: register '{}' Q parameter '{}' has no corresponding node",
                    register.name, param.name
                )
            })?;
        node.payload = NodePayload::RegisterRead {
            register: register.name.clone(),
        };
    }
    Ok(())
}

fn set_block_return_value(block: &mut ir::Fn, outputs: &[NodeRef], next_text_id: &mut usize) {
    match outputs {
        [] => {
            block.ret_ty = Type::nil();
            block.ret_node_ref = None;
        }
        [output] => {
            block.ret_ty = block.nodes[output.index].ty.clone();
            block.ret_node_ref = Some(*output);
        }
        _ => {
            let ty = Type::Tuple(
                outputs
                    .iter()
                    .map(|output| Box::new(block.nodes[output.index].ty.clone()))
                    .collect(),
            );
            let tuple_ref = NodeRef {
                index: block.nodes.len(),
            };
            block.nodes.push(Node {
                text_id: *next_text_id,
                name: None,
                ty: ty.clone(),
                payload: NodePayload::Tuple(outputs.to_vec()),
                pos: None,
            });
            *next_text_id += 1;
            block.ret_ty = ty;
            block.ret_node_ref = Some(tuple_ref);
        }
    }
}
