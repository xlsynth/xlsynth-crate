// SPDX-License-Identifier: Apache-2.0

//! Lowering from XLS block IR into a state-preserving sequential gate function.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth::IrBits;
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::dce::remove_dead_nodes;
use xlsynth_pir::ir::{self, BlockMetadata, MemberType, NodePayload, NodeRef, PackageMember, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils::{get_topological, operands, remap_payload_with};
use xlsynth_pir::ir_validate;
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;

use crate::aig::{
    ClockPort, GateFn, Output, RegisterBinding, ResetSpec, SequentialGateFn, TransitionInputId,
    TransitionOutputId,
};
use crate::gatify::ir2gate::{GatifyOptions, gatify};

#[derive(Debug, Clone)]
struct TransitionEndpoint {
    name: String,
    node_ref: NodeRef,
    ty: Type,
}

#[derive(Debug, Clone, Copy)]
struct RegisterWriteRefs {
    arg: NodeRef,
    load_enable: Option<NodeRef>,
    reset: Option<NodeRef>,
}

#[derive(Debug)]
struct PendingRegisterBinding {
    name: String,
    q: TransitionInputId,
    d_output_index: usize,
    load_enable_output_index: Option<usize>,
    reset: Option<PendingResetSpec>,
}

#[derive(Debug)]
struct PendingResetSpec {
    signal_output_index: usize,
    asynchronous: bool,
    active_low: bool,
    value: IrBits,
}

/// Parses an XLS IR package and lowers its selected block into a sequential
/// gate function.
pub fn block_ir_to_sequential_gate_fn(
    block_ir_text: &str,
    gatify_options: GatifyOptions,
) -> Result<SequentialGateFn, String> {
    let mut parser = Parser::new(block_ir_text);
    let package = parser
        .parse_and_validate_package()
        .map_err(|e| format!("parse block IR: {e}"))?;
    block_package_to_sequential_gate_fn(&package, gatify_options)
}

/// Lowers the selected block in a PIR package, inlining block instantiations
/// first.
pub fn block_package_to_sequential_gate_fn(
    package: &ir::Package,
    gatify_options: GatifyOptions,
) -> Result<SequentialGateFn, String> {
    ir_validate::validate_package(package)
        .map_err(|e| format!("block2sequential: input package validation failed: {e}"))?;

    let top_name = selected_block_name(package)?;
    validate_hierarchical_sequential_metadata(package, &top_name)?;
    let mut inlined_package = package.clone();
    inlined_package.top = Some((top_name, MemberType::Block));
    inline_all_blocks_in_package(&mut inlined_package)
        .map_err(|e| format!("block2sequential: block inlining failed: {e}"))?;
    ir_validate::validate_package(&inlined_package)
        .map_err(|e| format!("block2sequential: inlined package validation failed: {e}"))?;

    let PackageMember::Block { func, metadata } = inlined_package
        .get_top_block()
        .ok_or_else(|| "block2sequential: package has no selected block".to_string())?
    else {
        return Err("block2sequential: selected member is not a block".to_string());
    };
    lower_block_to_sequential_gate_fn(func, metadata, gatify_options)
}

/// Lowers an already flattened/inlined block into a sequential gate function.
pub fn lower_block_to_sequential_gate_fn(
    block: &ir::Fn,
    metadata: &BlockMetadata,
    gatify_options: GatifyOptions,
) -> Result<SequentialGateFn, String> {
    if !metadata.instantiations.is_empty() {
        return Err(
            "block2sequential: block contains instantiations; lower a package so they can be inlined"
                .to_string(),
        );
    }

    let validation_package = ir::Package {
        name: "block2sequential_validate".to_string(),
        file_table: ir::FileTable::new(),
        members: vec![PackageMember::Block {
            func: block.clone(),
            metadata: metadata.clone(),
        }],
        top: Some((block.name.clone(), MemberType::Block)),
    };
    ir_validate::validate_package(&validation_package)
        .map_err(|e| format!("block2sequential: block validation failed: {e}"))?;

    let (transition_fn, endpoints, pending_registers, external_input_count, clock) =
        build_transition_function(block, metadata)?;
    let mut transition = gatify(&transition_fn, gatify_options)
        .map_err(|e| format!("block2sequential: gatify transition function failed: {e}"))?
        .gate_fn;
    split_transition_outputs(&mut transition, &endpoints)?;

    let registers = pending_registers
        .into_iter()
        .map(|register| RegisterBinding {
            name: register.name,
            q: register.q,
            d: TransitionOutputId::new(register.d_output_index),
            load_enable: register
                .load_enable_output_index
                .map(TransitionOutputId::new),
            reset: register.reset.map(|reset| ResetSpec {
                signal: TransitionOutputId::new(reset.signal_output_index),
                asynchronous: reset.asynchronous,
                active_low: reset.active_low,
                value: reset.value,
            }),
            initial_value: None,
        })
        .collect();

    SequentialGateFn::new(
        block.name.clone(),
        transition,
        (0..external_input_count)
            .map(TransitionInputId::new)
            .collect(),
        (0..metadata.output_names.len())
            .map(TransitionOutputId::new)
            .collect(),
        clock,
        registers,
    )
    .map_err(|e| format!("block2sequential: invalid lowered sequential function: {e}"))
}

fn selected_block_name(package: &ir::Package) -> Result<String, String> {
    match &package.top {
        Some((name, MemberType::Block)) => Ok(name.clone()),
        Some((name, MemberType::Function)) => Err(format!(
            "block2sequential: package top '{}' is a function, not a block",
            name
        )),
        None => {
            let block_names: Vec<String> = package
                .members
                .iter()
                .filter_map(|member| match member {
                    PackageMember::Block { func, .. } => Some(func.name.clone()),
                    PackageMember::Function(_) => None,
                })
                .collect();
            match block_names.as_slice() {
                [] => Err("block2sequential: package has no blocks".to_string()),
                [name] => Ok(name.clone()),
                _ => Err(
                    "block2sequential: package has multiple blocks and no top block selection"
                        .to_string(),
                ),
            }
        }
    }
}

/// Rejects explicit child sequential behavior that would be lost during block
/// inlining.
fn validate_hierarchical_sequential_metadata(
    package: &ir::Package,
    top_name: &str,
) -> Result<(), String> {
    let (_, top_metadata) = get_block(package, top_name)?;
    let mut has_registers_memo = BTreeMap::new();
    let mut has_reset_writes_memo = BTreeMap::new();
    validate_instantiated_sequential_metadata(
        package,
        top_name,
        top_name,
        top_metadata,
        &mut has_registers_memo,
        &mut has_reset_writes_memo,
    )
}

fn validate_instantiated_sequential_metadata(
    package: &ir::Package,
    block_name: &str,
    path: &str,
    top_metadata: &BlockMetadata,
    has_registers_memo: &mut BTreeMap<String, bool>,
    has_reset_writes_memo: &mut BTreeMap<String, bool>,
) -> Result<(), String> {
    let (_, metadata) = get_block(package, block_name)?;
    for instantiation in &metadata.instantiations {
        let (_, child_metadata) = get_block(package, &instantiation.block)?;
        let child_path = format!("{path}.{}", instantiation.name);

        if subtree_has_registers(package, &instantiation.block, has_registers_memo)? {
            if let Some(child_clock) = child_metadata.clock_port_name.as_deref() {
                match top_metadata.clock_port_name.as_deref() {
                    Some(top_clock) if child_clock == top_clock => {}
                    Some(top_clock) => {
                        return Err(format!(
                            "block2sequential: registered instantiated block '{}' at '{}' declares clock '{}' incompatible with top clock '{}'",
                            instantiation.block, child_path, child_clock, top_clock
                        ));
                    }
                    None => {
                        return Err(format!(
                            "block2sequential: registered instantiated block '{}' at '{}' declares clock '{}' but the top block has no clock",
                            instantiation.block, child_path, child_clock
                        ));
                    }
                }
            }
        }

        if subtree_has_reset_writes(package, &instantiation.block, has_reset_writes_memo)? {
            if let Some(child_reset) = child_metadata.reset.as_ref() {
                match top_metadata.reset.as_ref() {
                    Some(top_reset)
                        if child_reset.asynchronous == top_reset.asynchronous
                            && child_reset.active_low == top_reset.active_low => {}
                    Some(top_reset) => {
                        return Err(format!(
                            "block2sequential: reset-bearing instantiated block '{}' at '{}' declares reset behavior (asynchronous={}, active_low={}) incompatible with top reset behavior (asynchronous={}, active_low={})",
                            instantiation.block,
                            child_path,
                            child_reset.asynchronous,
                            child_reset.active_low,
                            top_reset.asynchronous,
                            top_reset.active_low
                        ));
                    }
                    None => {
                        return Err(format!(
                            "block2sequential: reset-bearing instantiated block '{}' at '{}' declares reset behavior but the top block has no reset metadata",
                            instantiation.block, child_path
                        ));
                    }
                }
            }
        }

        validate_instantiated_sequential_metadata(
            package,
            &instantiation.block,
            &child_path,
            top_metadata,
            has_registers_memo,
            has_reset_writes_memo,
        )?;
    }
    Ok(())
}

fn subtree_has_registers(
    package: &ir::Package,
    block_name: &str,
    memo: &mut BTreeMap<String, bool>,
) -> Result<bool, String> {
    if let Some(has_registers) = memo.get(block_name) {
        return Ok(*has_registers);
    }
    let (_, metadata) = get_block(package, block_name)?;
    let mut has_registers = !metadata.registers.is_empty();
    for instantiation in &metadata.instantiations {
        has_registers |= subtree_has_registers(package, &instantiation.block, memo)?;
    }
    memo.insert(block_name.to_string(), has_registers);
    Ok(has_registers)
}

fn subtree_has_reset_writes(
    package: &ir::Package,
    block_name: &str,
    memo: &mut BTreeMap<String, bool>,
) -> Result<bool, String> {
    if let Some(has_reset_writes) = memo.get(block_name) {
        return Ok(*has_reset_writes);
    }
    let (block, metadata) = get_block(package, block_name)?;
    let mut has_reset_writes = block.nodes.iter().any(|node| {
        matches!(
            &node.payload,
            NodePayload::RegisterWrite { reset: Some(_), .. }
        )
    });
    for instantiation in &metadata.instantiations {
        has_reset_writes |= subtree_has_reset_writes(package, &instantiation.block, memo)?;
    }
    memo.insert(block_name.to_string(), has_reset_writes);
    Ok(has_reset_writes)
}

fn get_block<'a>(
    package: &'a ir::Package,
    block_name: &str,
) -> Result<(&'a ir::Fn, &'a BlockMetadata), String> {
    match package.get_block(block_name) {
        Some(PackageMember::Block { func, metadata }) => Ok((func, metadata)),
        _ => Err(format!(
            "block2sequential: referenced block '{}' is unavailable",
            block_name
        )),
    }
}

fn build_transition_function(
    block: &ir::Fn,
    metadata: &BlockMetadata,
) -> Result<
    (
        ir::Fn,
        Vec<TransitionEndpoint>,
        Vec<PendingRegisterBinding>,
        usize,
        Option<ClockPort>,
    ),
    String,
> {
    block
        .check_pir_layout_invariants()
        .map_err(|e| format!("block2sequential: invalid block layout: {e}"))?;

    let original_outputs = collect_block_outputs(block, metadata)?;
    let register_writes = collect_register_writes(block)?;
    let external_input_count = block.params.len();
    let mut used_param_names: BTreeSet<String> = block
        .params
        .iter()
        .map(|param| param.name.clone())
        .collect();
    let mut used_output_names: BTreeSet<String> = BTreeSet::new();
    let mut max_text_id = block
        .nodes
        .iter()
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0);

    let mut transition = ir::Fn {
        name: format!("{}__transition", block.name),
        params: block.params.clone(),
        ret_ty: Type::nil(),
        nodes: vec![block.nodes[0].clone()],
        ret_node_ref: None,
        outer_attrs: vec![],
        inner_attrs: vec![],
    };
    let mut old_to_new: Vec<Option<NodeRef>> = vec![None; block.nodes.len()];
    old_to_new[0] = Some(NodeRef { index: 0 });
    for (index, param) in block.params.iter().enumerate() {
        let old_ref = NodeRef { index: index + 1 };
        let new_ref = NodeRef {
            index: transition.nodes.len(),
        };
        transition.nodes.push(block.nodes[old_ref.index].clone());
        old_to_new[old_ref.index] = Some(new_ref);
        debug_assert_eq!(
            transition.nodes[new_ref.index].payload,
            NodePayload::GetParam(param.id)
        );
    }

    let mut register_q_refs: BTreeMap<String, (NodeRef, TransitionInputId)> = BTreeMap::new();
    for register in &metadata.registers {
        max_text_id += 1;
        let q_name = unique_name(&format!("{}__q", register.name), &mut used_param_names);
        let param_id = ir::ParamId::new(max_text_id);
        let q_ref = NodeRef {
            index: transition.nodes.len(),
        };
        transition.params.push(ir::Param {
            name: q_name.clone(),
            ty: register.ty.clone(),
            id: param_id,
        });
        transition.nodes.push(ir::Node {
            text_id: max_text_id,
            name: Some(q_name),
            ty: register.ty.clone(),
            payload: NodePayload::GetParam(param_id),
            pos: None,
        });
        register_q_refs.insert(
            register.name.clone(),
            (q_ref, TransitionInputId::new(transition.params.len() - 1)),
        );
    }

    for old_ref in get_topological(block) {
        if old_ref.index == 0 || (1..=block.params.len()).contains(&old_ref.index) {
            continue;
        }
        let node = block.get_node(old_ref);
        match &node.payload {
            NodePayload::RegisterRead { register } => {
                let (q_ref, _) = register_q_refs.get(register).ok_or_else(|| {
                    format!(
                        "block2sequential: register_read references unknown register '{}'",
                        register
                    )
                })?;
                old_to_new[old_ref.index] = Some(*q_ref);
            }
            NodePayload::RegisterWrite { .. } => {
                // Register writes become named transition outputs below.
            }
            NodePayload::InstantiationInput { .. } | NodePayload::InstantiationOutput { .. } => {
                return Err(
                    "block2sequential: instantiation nodes remain after block inlining".to_string(),
                );
            }
            NodePayload::Nil => {
                // Dead nodes have no transition-function meaning.
            }
            _ => {
                for dependency in operands(&node.payload) {
                    if old_to_new[dependency.index].is_none() {
                        return Err(format!(
                            "block2sequential: node {} depends on an unsupported sequential side-effect node",
                            node.text_id
                        ));
                    }
                }
                let payload = remap_payload_with(&node.payload, |(_, dependency)| {
                    old_to_new[dependency.index]
                        .expect("dependencies were checked before payload remapping")
                });
                let new_ref = NodeRef {
                    index: transition.nodes.len(),
                };
                transition.nodes.push(ir::Node {
                    payload,
                    ..node.clone()
                });
                old_to_new[old_ref.index] = Some(new_ref);
            }
        }
    }

    let mut endpoints: Vec<TransitionEndpoint> = Vec::new();
    for output in original_outputs {
        let mapped_ref = old_to_new[output.node_ref.index].ok_or_else(|| {
            format!(
                "block2sequential: external output '{}' refers to an unsupported node",
                output.name
            )
        })?;
        if !used_output_names.insert(output.name.clone()) {
            return Err(format!(
                "block2sequential: duplicate external output name '{}'",
                output.name
            ));
        }
        endpoints.push(TransitionEndpoint {
            node_ref: mapped_ref,
            ..output
        });
    }

    let reset_metadata = metadata.reset.as_ref();
    let mut pending_registers = Vec::with_capacity(metadata.registers.len());
    for register in &metadata.registers {
        let (q_ref, q_input) = *register_q_refs
            .get(&register.name)
            .expect("Q parameter was created for each register");
        let write = register_writes.get(&register.name).copied();
        let d_ref = match write {
            Some(write) => remapped_ref(
                &old_to_new,
                write.arg,
                &format!("register '{}' D", register.name),
            )?,
            None => q_ref,
        };
        let d_output_index = push_endpoint(
            &mut endpoints,
            &mut used_output_names,
            format!("{}__d", register.name),
            d_ref,
            register.ty.clone(),
        );

        let load_enable_output_index = match write.and_then(|write| write.load_enable) {
            Some(load_enable) => {
                let load_enable_ref = remapped_ref(
                    &old_to_new,
                    load_enable,
                    &format!("register '{}' load enable", register.name),
                )?;
                Some(push_endpoint(
                    &mut endpoints,
                    &mut used_output_names,
                    format!("{}__load_enable", register.name),
                    load_enable_ref,
                    Type::Bits(1),
                ))
            }
            None => None,
        };

        let reset = match (
            write.and_then(|write| write.reset),
            register.reset_value.as_ref(),
        ) {
            (Some(reset_ref), Some(reset_value)) => {
                let reset_metadata = reset_metadata.ok_or_else(|| {
                    format!(
                        "block2sequential: register '{}' has a reset write but the block has no reset metadata",
                        register.name
                    )
                })?;
                let reset_ref = remapped_ref(
                    &old_to_new,
                    reset_ref,
                    &format!("register '{}' reset", register.name),
                )?;
                let signal_output_index = push_endpoint(
                    &mut endpoints,
                    &mut used_output_names,
                    format!("{}__reset", register.name),
                    reset_ref,
                    Type::Bits(1),
                );
                Some(PendingResetSpec {
                    signal_output_index,
                    asynchronous: reset_metadata.asynchronous,
                    active_low: reset_metadata.active_low,
                    value: flatten_value(reset_value, &register.ty)?,
                })
            }
            (Some(_), None) => {
                return Err(format!(
                    "block2sequential: register '{}' has a reset write but no reset value",
                    register.name
                ));
            }
            (None, Some(_)) => {
                return Err(format!(
                    "block2sequential: register '{}' has a reset value but no reset write",
                    register.name
                ));
            }
            (None, None) => None,
        };

        pending_registers.push(PendingRegisterBinding {
            name: register.name.clone(),
            q: q_input,
            d_output_index,
            load_enable_output_index,
            reset,
        });
    }

    set_transition_return(&mut transition, &endpoints, &mut max_text_id);
    let transition = remove_dead_nodes(&transition);
    let clock = metadata
        .clock_port_name
        .as_ref()
        .map(|name| ClockPort { name: name.clone() });
    Ok((
        transition,
        endpoints,
        pending_registers,
        external_input_count,
        clock,
    ))
}

fn collect_block_outputs(
    block: &ir::Fn,
    metadata: &BlockMetadata,
) -> Result<Vec<TransitionEndpoint>, String> {
    let ret_ref = block
        .ret_node_ref
        .ok_or_else(|| "block2sequential: block has no return node".to_string())?;
    if metadata.output_names.len() == 1 {
        return Ok(vec![TransitionEndpoint {
            name: metadata.output_names[0].clone(),
            node_ref: ret_ref,
            ty: block.ret_ty.clone(),
        }]);
    }
    let NodePayload::Tuple(elements) = &block.get_node(ret_ref).payload else {
        return Err(
            "block2sequential: multiple block outputs require a tuple return node".to_string(),
        );
    };
    let Type::Tuple(types) = &block.ret_ty else {
        return Err("block2sequential: multiple block outputs require a tuple type".to_string());
    };
    if elements.len() != metadata.output_names.len() || types.len() != elements.len() {
        return Err("block2sequential: block output arity mismatch".to_string());
    }
    Ok(metadata
        .output_names
        .iter()
        .zip(elements)
        .zip(types)
        .map(|((name, node_ref), ty)| TransitionEndpoint {
            name: name.clone(),
            node_ref: *node_ref,
            ty: (**ty).clone(),
        })
        .collect())
}

fn collect_register_writes(block: &ir::Fn) -> Result<BTreeMap<String, RegisterWriteRefs>, String> {
    let mut writes = BTreeMap::new();
    for node in &block.nodes {
        if let NodePayload::RegisterWrite {
            arg,
            register,
            load_enable,
            reset,
        } = &node.payload
        {
            let write = RegisterWriteRefs {
                arg: *arg,
                load_enable: *load_enable,
                reset: *reset,
            };
            if writes.insert(register.clone(), write).is_some() {
                return Err(format!(
                    "block2sequential: multiple register writes for '{}'",
                    register
                ));
            }
        }
    }
    Ok(writes)
}

fn remapped_ref(
    old_to_new: &[Option<NodeRef>],
    old_ref: NodeRef,
    context: &str,
) -> Result<NodeRef, String> {
    old_to_new[old_ref.index]
        .ok_or_else(|| format!("block2sequential: {context} refers to an unsupported node"))
}

fn unique_name(base: &str, used: &mut BTreeSet<String>) -> String {
    if used.insert(base.to_string()) {
        return base.to_string();
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}__{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix sequence must provide a unique name")
}

fn push_endpoint(
    endpoints: &mut Vec<TransitionEndpoint>,
    used_names: &mut BTreeSet<String>,
    preferred_name: String,
    node_ref: NodeRef,
    ty: Type,
) -> usize {
    let name = unique_name(&preferred_name, used_names);
    let index = endpoints.len();
    endpoints.push(TransitionEndpoint { name, node_ref, ty });
    index
}

fn flatten_value(value: &xlsynth::IrValue, ty: &Type) -> Result<IrBits, String> {
    let mut flat_bits = Vec::with_capacity(ty.bit_count());
    flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut flat_bits)
        .map_err(|e| format!("block2sequential: flatten reset value failed: {e}"))?;
    Ok(IrBits::from_lsb_is_0(&flat_bits))
}

fn set_transition_return(
    transition: &mut ir::Fn,
    endpoints: &[TransitionEndpoint],
    max_text_id: &mut usize,
) {
    assert!(
        !endpoints.is_empty(),
        "parsed blocks must have at least one output endpoint"
    );
    if endpoints.len() == 1 {
        transition.ret_ty = endpoints[0].ty.clone();
        transition.ret_node_ref = Some(endpoints[0].node_ref);
        return;
    }

    *max_text_id += 1;
    let ret_ty = Type::Tuple(
        endpoints
            .iter()
            .map(|endpoint| Box::new(endpoint.ty.clone()))
            .collect(),
    );
    let ret_ref = NodeRef {
        index: transition.nodes.len(),
    };
    transition.nodes.push(ir::Node {
        text_id: *max_text_id,
        name: Some("transition_outputs".to_string()),
        ty: ret_ty.clone(),
        payload: NodePayload::Tuple(endpoints.iter().map(|endpoint| endpoint.node_ref).collect()),
        pos: None,
    });
    transition.ret_ty = ret_ty;
    transition.ret_node_ref = Some(ret_ref);
}

fn split_transition_outputs(
    transition: &mut GateFn,
    endpoints: &[TransitionEndpoint],
) -> Result<(), String> {
    if transition.outputs.len() != 1 {
        return Err(format!(
            "block2sequential: gatify produced {} transition outputs, expected one flat output",
            transition.outputs.len()
        ));
    }
    let flat_output = transition.outputs.remove(0).bit_vector;
    let expected_width: usize = endpoints
        .iter()
        .map(|endpoint| endpoint.ty.bit_count())
        .sum();
    if flat_output.get_bit_count() != expected_width {
        return Err(format!(
            "block2sequential: gatify transition output has width {}, expected {}",
            flat_output.get_bit_count(),
            expected_width
        ));
    }

    let mut lsb_start = expected_width;
    for endpoint in endpoints {
        let width = endpoint.ty.bit_count();
        lsb_start -= width;
        transition.outputs.push(Output {
            name: endpoint.name.clone(),
            bit_vector: flat_output.get_lsb_slice(lsb_start, width),
        });
    }
    Ok(())
}
