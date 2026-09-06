// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::block_inline::inline_all_blocks_in_package;
use crate::dce::remove_dead_nodes;
use crate::ir::{self, Block, BlockPort, MemberType, NodePayload, NodeRef, PackageMember, Type};
use crate::ir_eval::eval_pure_if_supported;
use crate::ir_parser::Parser;
use crate::ir_utils::{
    compact_and_toposort_in_place, compute_users, get_topological_nodes, operands,
    remap_payload_with, verify_no_cycle,
};
use crate::{IrBits, IrValue};

#[derive(Debug, Clone, Default)]
pub struct Block2FnOptions {
    pub tie_input_ports: BTreeMap<String, IrBits>,
    pub drop_output_ports: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub struct Block2FnResult {
    pub package_name: String,
    pub function: ir::Fn,
}

// Converts the top block of the package into a function. Optionally ties off
// inputs and drops outputs.
pub fn block_package_to_fn(
    pkg: &ir::Package,
    options: &Block2FnOptions,
) -> Result<Block2FnResult, String> {
    let mut pkg = pkg.clone();
    inline_all_blocks_in_package(&mut pkg)?;
    let package_name = pkg.name.clone();
    let top_block = pkg
        .get_top_block()
        .ok_or_else(|| "block2fn: package has no block members".to_string())?;
    let mut block = top_block.clone();
    if !block.instantiations.is_empty() {
        return Err(
            "block2fn: block contains instantiations; run block_inlining first".to_string(),
        );
    }
    tie_input_ports(&mut block, options)?;
    drop_output_ports(&mut block, options)?;
    let output_count = block.output_ports().count();
    if output_count != 1 {
        return Err(format!(
            "block2fn: expected exactly one output port after dropping outputs; got {output_count}"
        ));
    }
    let mut f = project_block_interface(&block)?;
    simplify_and_const_prop(&mut f)?;

    collapse_registers(&mut f)?;

    verify_no_cycle(&f).map_err(|e| format!("block2fn: {e}"))?;
    let mut f = remove_dead_nodes(&f);
    compact_and_toposort_in_place(&mut f).map_err(|e| format!("block2fn: compact failed: {e}"))?;

    Ok(Block2FnResult {
        package_name,
        function: f,
    })
}

pub fn block_ir_to_fn(
    block_ir_text: &str,
    options: &Block2FnOptions,
) -> Result<Block2FnResult, String> {
    let mut parser = Parser::new(block_ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .map_err(|e| format!("parse block IR: {e}"))?;
    block_package_to_fn(&pkg, options)
}

pub fn block_ir_to_fn_package(
    block_ir_text: &str,
    options: &Block2FnOptions,
) -> Result<ir::Package, String> {
    let result = block_ir_to_fn(block_ir_text, options)?;
    Ok(ir::Package {
        name: result.package_name,
        file_table: ir::FileTable::new(),
        members: vec![PackageMember::Function(result.function.clone())],
        top: Some((result.function.name.clone(), MemberType::Function)),
    })
}

/// Projects a combinational block's ordered interface into a function
/// signature.
///
/// One output returns its driving value directly; zero or multiple outputs
/// return a tuple in port order. Stateful blocks require a separate lowering.
pub fn combinational_block_to_fn(block: &Block) -> Result<ir::Fn, String> {
    require_combinational_block(block)?;
    project_block_interface(block)
}

/// Rejects stateful interfaces before using the explicit combinational adapter.
fn require_combinational_block(block: &Block) -> Result<(), String> {
    if !block.registers.is_empty()
        || !block.instantiations.is_empty()
        || block.nodes.iter().any(|node| {
            matches!(
                node.payload,
                NodePayload::RegisterRead { .. }
                    | NodePayload::RegisterWrite { .. }
                    | NodePayload::InstantiationInput { .. }
                    | NodePayload::InstantiationOutput { .. }
            )
        })
    {
        return Err(
            "combinational block projection does not support registers or instantiations"
                .to_string(),
        );
    }
    Ok(())
}

/// Replaces combinational logic while retaining the template's physical
/// interface.
///
/// Input and output port IDs, names, declaration order, clock, source
/// positions, and SystemVerilog annotations are preserved. New logic nodes
/// whose IDs or names collide with retained ports are assigned fresh IDs or
/// names without changing graph references.
pub fn replace_combinational_block_logic(
    template: &Block,
    mut function: ir::Fn,
) -> Result<Block, String> {
    require_combinational_block(template)?;
    let output_types = template
        .output_ports()
        .map(|port| Box::new(template.port_type(port).clone()))
        .collect::<Vec<_>>();
    let return_type = if output_types.len() == 1 {
        (*output_types[0]).clone()
    } else {
        Type::Tuple(output_types)
    };
    if function.params.len() != template.input_ports().count()
        || function
            .params
            .iter()
            .zip(template.input_ports())
            .any(|(actual, port)| {
                actual.name != template.port_name(port) || &actual.ty != template.port_type(port)
            })
        || function.ret_ty != return_type
    {
        return Err(
            "replacement function must preserve the block's ordered input and output types"
                .to_string(),
        );
    }
    rename_logic_conflicting_with_ports(&mut function, template)?;
    let output_names = template
        .output_ports()
        .map(|port| template.port_name(port).to_string())
        .collect::<Vec<_>>();
    let mut block = Block::from_function(function, Some(&output_names))?;
    let mut port_pairs = Vec::new();
    let mut ports = Vec::with_capacity(template.ports.len());
    for port in &template.ports {
        match port {
            BlockPort::Input(old) => {
                let new = block
                    .get_input_port(template.port_name(*old))
                    .ok_or("replacement is missing an input port")?;
                port_pairs.push((*old, new));
                ports.push(BlockPort::Input(new));
            }
            BlockPort::Output(old) => {
                let new = block
                    .get_output_port(template.port_name(*old))
                    .ok_or("replacement is missing an output port")?;
                port_pairs.push((*old, new));
                ports.push(BlockPort::Output(new));
            }
            BlockPort::Clock(name) => ports.push(BlockPort::Clock(name.clone())),
        }
    }
    let reserved_ids = port_pairs
        .iter()
        .map(|(old, _)| template.get_node(*old).text_id)
        .collect::<BTreeSet<_>>();
    let port_indices = port_pairs
        .iter()
        .map(|(_, new)| new.index)
        .collect::<BTreeSet<_>>();
    let mut last_id = block
        .nodes
        .iter()
        .chain(&template.nodes)
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0);
    for (index, node) in block.nodes.iter_mut().enumerate() {
        if !port_indices.contains(&index)
            && !matches!(node.payload, NodePayload::Nil)
            && reserved_ids.contains(&node.text_id)
        {
            last_id = last_id
                .checked_add(1)
                .ok_or("replacement node IDs overflow usize")?;
            if node.name.as_deref()
                == Some(format!("{}.{}", node.payload.get_operator(), node.text_id).as_str())
            {
                node.name = None;
            }
            node.text_id = last_id;
        }
    }
    for (old, new) in &port_pairs {
        let original = template.get_node(*old);
        let replacement = block.get_node_mut(*new);
        replacement.text_id = original.text_id;
        replacement.name = original.name.clone();
        replacement.pos = original.pos.clone();
        match (&original.payload, &mut replacement.payload) {
            (
                NodePayload::InputPort { sv_type, .. },
                NodePayload::InputPort {
                    sv_type: new_sv_type,
                    ..
                },
            )
            | (
                NodePayload::OutputPort { sv_type, .. },
                NodePayload::OutputPort {
                    sv_type: new_sv_type,
                    ..
                },
            ) => {
                *new_sv_type = sv_type.clone();
            }
            _ => return Err("replacement port direction changed".to_string()),
        }
    }
    block.ports = ports;
    block.reset = template
        .reset
        .as_ref()
        .map(|reset| {
            let port = port_pairs
                .iter()
                .find(|(old, _)| *old == reset.port)
                .map(|(_, new)| *new)
                .ok_or("template reset does not reference an input port")?;
            Ok::<_, String>(ir::BlockReset {
                port,
                asynchronous: reset.asynchronous,
                active_low: reset.active_low,
            })
        })
        .transpose()?;
    block.name = template.name.clone();
    block.outer_attrs = template.outer_attrs.clone();
    block.inner_attrs = template.inner_attrs.clone();
    Ok(block)
}

/// Reserves the physical interface's semantic and emitted names before
/// constructing outputs or restoring aliases onto replacement input nodes.
fn rename_logic_conflicting_with_ports(
    function: &mut ir::Fn,
    template: &Block,
) -> Result<(), String> {
    let mut reserved_names = BTreeSet::new();
    for port in &template.ports {
        match port {
            BlockPort::Input(port) | BlockPort::Output(port) => {
                reserved_names.insert(template.port_name(*port).to_string());
                let node = template.get_node(*port);
                reserved_names.insert(node.name.clone().unwrap_or_else(|| {
                    format!("{}.{}", node.payload.get_operator(), node.text_id)
                }));
            }
            BlockPort::Clock(name) => {
                reserved_names.insert(name.clone());
            }
        }
    }
    let mut used_names = reserved_names.clone();
    used_names.extend(function.nodes.iter().filter_map(|node| node.name.clone()));
    for node in &mut function.nodes {
        if matches!(node.payload, NodePayload::Nil | NodePayload::GetParam(_)) {
            // Parameters retain the names required by the function signature;
            // they will become the matching ports, not competing logic nodes.
            continue;
        }
        let Some(name) = &node.name else { continue };
        if !reserved_names.contains(name) {
            continue;
        }
        let mut suffix = 1usize;
        loop {
            let candidate = format!("{name}_{suffix}");
            if used_names.insert(candidate.clone()) {
                node.name = Some(candidate);
                break;
            }
            suffix = suffix
                .checked_add(1)
                .ok_or("replacement node-name suffix overflows usize")?;
        }
    }
    Ok(())
}

/// Converts only the interface; the historical register-collapse flow follows
/// this with explicit state elimination before returning a function.
fn project_block_interface(block: &Block) -> Result<ir::Fn, String> {
    let outputs = block
        .output_ports()
        .map(|port| block.output_value(port))
        .collect::<Vec<_>>();
    let mut graph = block.graph.clone();
    let users = compute_users(block);
    let mut params = Vec::new();
    for port in block.input_ports() {
        let node = graph.get_node_mut(port);
        let name = block.port_name(port).to_string();
        let id = ir::ParamId::new(node.text_id);
        params.push(ir::Param {
            name: name.clone(),
            ty: node.ty.clone(),
            id,
        });
        node.name = Some(name);
        node.payload = NodePayload::GetParam(id);
    }
    for port in block.output_ports() {
        // Only materialize a sink's unit value if another node observes it;
        // otherwise no-op ECO roundtrips would accumulate dead placeholders.
        graph.get_node_mut(port).payload =
            if users.get(&port).is_some_and(|users| !users.is_empty()) {
                NodePayload::Tuple(Vec::new())
            } else {
                NodePayload::Nil
            };
        graph.get_node_mut(port).name = None;
    }
    let result = if outputs.len() == 1 {
        outputs[0]
    } else {
        let text_id = graph
            .nodes
            .iter()
            .map(|node| node.text_id)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or("block2fn: node IDs overflow usize")?;
        let ty = Type::Tuple(
            outputs
                .iter()
                .map(|node| Box::new(graph.get_node_ty(*node).clone()))
                .collect(),
        );
        let result = NodeRef {
            index: graph.nodes.len(),
        };
        graph.nodes.push(ir::Node {
            text_id,
            name: None,
            ty,
            payload: NodePayload::Tuple(outputs),
            pos: None,
        });
        result
    };
    let ret_ty = graph.get_node_ty(result).clone();
    let mut function = ir::Fn {
        graph,
        params,
        ret_ty,
        ret_node_ref: Some(result),
    };
    reorder_params_and_compact(&mut function)?;
    Ok(function)
}

/// Replaces tied input ports with literals while retaining the other ports.
fn tie_input_ports(block: &mut Block, options: &Block2FnOptions) -> Result<(), String> {
    let mut replacements = Vec::new();
    for (name, bits) in &options.tie_input_ports {
        let port = block
            .get_input_port(name)
            .ok_or_else(|| format!("block2fn: unknown input port '{name}'"))?;
        let value = parse_literal_for_type(bits, block.port_type(port))?;
        replacements.push((port, value));
    }
    for (port, value) in replacements {
        block.get_node_mut(port).payload = NodePayload::Literal(IrValue::from_bits(&value));
        block.get_node_mut(port).name = None;
        block.ports.retain(|entry| *entry != BlockPort::Input(port));
        if block.reset.as_ref().is_some_and(|reset| reset.port == port) {
            block.reset = None;
        }
    }
    Ok(())
}

/// Removes selected output sinks without changing the remaining output order.
fn drop_output_ports(block: &mut Block, options: &Block2FnOptions) -> Result<(), String> {
    let mut dropped = Vec::new();
    for name in &options.drop_output_ports {
        let port = block
            .get_output_port(name)
            .ok_or_else(|| format!("block2fn: unknown output ports: {name}"))?;
        dropped.push(port);
    }
    if !dropped.is_empty() && dropped.len() == block.output_ports().count() {
        return Err("block2fn: all outputs dropped; at least one output required".to_string());
    }
    for port in dropped {
        block.get_node_mut(port).payload = NodePayload::Tuple(Vec::new());
        block.get_node_mut(port).name = None;
        block
            .ports
            .retain(|entry| *entry != BlockPort::Output(port));
    }
    Ok(())
}

fn simplify_and_const_prop(f: &mut ir::Fn) -> Result<(), String> {
    let reg_write_args = collect_register_write_args(f)?;
    let mut const_nodes: HashMap<usize, IrValue> = HashMap::new();
    for (idx, node) in f.nodes.iter().enumerate() {
        if let NodePayload::Literal(value) = &node.payload {
            const_nodes.insert(idx, value.clone());
        }
    }

    loop {
        let mut newly_constant: Vec<(usize, IrValue)> = Vec::new();
        for (idx, node) in f.nodes.iter().enumerate() {
            if const_nodes.contains_key(&idx) {
                continue;
            }
            let new_value = match &node.payload {
                NodePayload::Nary(ir::NaryOp::And, operands) => {
                    simplify_and(node, operands, &const_nodes)?
                }
                NodePayload::Nary(ir::NaryOp::Or, operands) => {
                    simplify_or(node, operands, &const_nodes)?
                }
                NodePayload::RegisterRead { register } => {
                    simplify_register_read(register, &reg_write_args, &const_nodes)?
                }
                _ => None,
            };
            if let Some(value) = new_value {
                newly_constant.push((idx, value));
                continue;
            }

            let deps = operands(&node.payload);
            if deps.is_empty() {
                continue;
            }
            if !deps.iter().all(|nr| const_nodes.contains_key(&nr.index)) {
                continue;
            }
            let operand_values: Vec<&IrValue> = deps
                .iter()
                .map(|nr| {
                    const_nodes
                        .get(&nr.index)
                        .expect("dependency value must be available")
                })
                .collect();
            if let Some(value) = eval_pure_if_supported(node, &operand_values) {
                newly_constant.push((idx, value));
            }
        }
        if newly_constant.is_empty() {
            break;
        }
        for (idx, value) in newly_constant {
            let node_ref = NodeRef { index: idx };
            let ty = f.nodes[idx].ty.clone();
            crate::ir_utils::replace_node_payload(
                f,
                node_ref,
                NodePayload::Literal(value.clone()),
                Some(ty),
            )
            .map_err(|e| format!("block2fn: const-prop failed: {e}"))?;
            const_nodes.insert(idx, value);
        }
    }
    Ok(())
}

fn simplify_and(
    node: &ir::Node,
    operands: &[NodeRef],
    const_nodes: &HashMap<usize, IrValue>,
) -> Result<Option<IrValue>, String> {
    if !matches!(node.ty, Type::Bits(_)) {
        return Ok(None);
    }
    let bit_count = node.ty.bit_count();
    for nr in operands {
        if let Some(value) = const_nodes.get(&nr.index)
            && ir_value_is_all_zeros(value, bit_count)?
        {
            return Ok(Some(ir_value_zero(bit_count)?));
        }
    }
    Ok(None)
}

fn simplify_or(
    node: &ir::Node,
    operands: &[NodeRef],
    const_nodes: &HashMap<usize, IrValue>,
) -> Result<Option<IrValue>, String> {
    if !matches!(node.ty, Type::Bits(_)) {
        return Ok(None);
    }
    let bit_count = node.ty.bit_count();
    for nr in operands {
        if let Some(value) = const_nodes.get(&nr.index)
            && ir_value_is_all_ones(value, bit_count)?
        {
            return Ok(Some(ir_value_ones(bit_count)?));
        }
    }
    Ok(None)
}

fn simplify_register_read(
    register: &str,
    reg_write_args: &HashMap<String, NodeRef>,
    const_nodes: &HashMap<usize, IrValue>,
) -> Result<Option<IrValue>, String> {
    if let Some(value) = reg_write_args
        .get(register)
        .and_then(|arg_ref| const_nodes.get(&arg_ref.index))
    {
        return Ok(Some(value.clone()));
    }
    Ok(None)
}

fn collapse_registers(f: &mut ir::Fn) -> Result<(), String> {
    let reg_write_args = collect_register_write_args(f)?;
    let mut reads: Vec<(NodeRef, String, NodeRef)> = Vec::new();
    for (idx, node) in f.nodes.iter().enumerate() {
        if let NodePayload::RegisterRead { register } = &node.payload {
            let arg_ref = reg_write_args.get(register).ok_or_else(|| {
                format!(
                    "block2fn: register_read '{}' has no matching write",
                    register
                )
            })?;
            reads.push((NodeRef { index: idx }, register.clone(), *arg_ref));
        }
    }
    for (read_ref, register, arg_ref) in reads {
        crate::ir_utils::replace_node_with_ref(f, read_ref, arg_ref)
            .map_err(|e| format!("block2fn: collapse register '{}': {e}", register))?;
    }
    Ok(())
}

fn collect_register_write_args(f: &ir::Fn) -> Result<HashMap<String, NodeRef>, String> {
    let mut reg_write_args: HashMap<String, NodeRef> = HashMap::new();
    for node in f.nodes.iter() {
        if let NodePayload::RegisterWrite { arg, register, .. } = &node.payload
            && reg_write_args.insert(register.clone(), *arg).is_some()
        {
            return Err(format!(
                "block2fn: multiple register_write nodes for '{}'",
                register
            ));
        }
    }
    Ok(reg_write_args)
}

fn reorder_params_and_compact(f: &mut ir::Fn) -> Result<(), String> {
    let mut param_nodes: HashMap<ir::ParamId, usize> = HashMap::new();
    for (idx, node) in f.nodes.iter().enumerate() {
        if let NodePayload::GetParam(pid) = node.payload {
            param_nodes.insert(pid, idx);
        }
    }

    let mut kept_order: Vec<NodeRef> = Vec::new();
    kept_order.push(NodeRef { index: 0 });
    for param in f.params.iter() {
        let idx = *param_nodes
            .get(&param.id)
            .ok_or_else(|| format!("block2fn: missing GetParam for '{}'", param.name))?;
        kept_order.push(NodeRef { index: idx });
    }

    // Note: we may be calling this while PIR layout invariants are temporarily
    // violated (e.g. after dropping/reordering params but before compaction).
    // Use the nodes-only topo routine to avoid debug assertions on `Fn`.
    let topo_all = get_topological_nodes(&f.nodes);
    let mut already_kept = vec![false; f.nodes.len()];
    for nr in kept_order.iter().copied() {
        already_kept[nr.index] = true;
    }
    for nr in topo_all.into_iter() {
        if already_kept[nr.index] {
            continue;
        }
        if matches!(f.get_node(nr).payload, NodePayload::Nil) {
            continue;
        }
        kept_order.push(nr);
        already_kept[nr.index] = true;
    }

    let old_len = f.nodes.len();
    let mut old_to_new: Vec<Option<usize>> = vec![None; old_len];
    for (new_idx, nr) in kept_order.iter().enumerate() {
        old_to_new[nr.index] = Some(new_idx);
    }

    let mut new_nodes: Vec<ir::Node> = Vec::with_capacity(kept_order.len());
    for nr in kept_order.iter().copied() {
        let src = f.get_node(nr).clone();
        let remapped_payload = remap_payload_with(&src.payload, |(_, dep): (usize, NodeRef)| {
            let Some(new_index) = old_to_new.get(dep.index).and_then(|x| *x) else {
                panic!("block2fn: dependency {} removed during reorder", dep.index);
            };
            NodeRef { index: new_index }
        });
        new_nodes.push(ir::Node {
            payload: remapped_payload,
            ..src
        });
    }

    if let Some(old_ret) = f.ret_node_ref {
        let mapped = old_to_new[old_ret.index].ok_or_else(|| {
            format!(
                "block2fn: return node {} removed during reorder",
                old_ret.index
            )
        })?;
        f.ret_node_ref = Some(NodeRef { index: mapped });
    }
    f.nodes = new_nodes;
    Ok(())
}

fn parse_literal_for_type(literal: &IrBits, ty: &Type) -> Result<IrBits, String> {
    let Type::Bits(width) = ty else {
        return Err(format!(
            "block2fn: tie-input only supports bits type; got {}",
            ty
        ));
    };
    if literal.get_bit_count() != *width {
        return Err(format!(
            "block2fn: literal has bit width {}, expected {}",
            literal.get_bit_count(),
            width
        ));
    }
    Ok(literal.clone())
}

fn ir_value_is_all_zeros(value: &IrValue, bit_count: usize) -> Result<bool, String> {
    let bits = value
        .to_bits()
        .map_err(|e| format!("block2fn: to_bits failed: {e}"))?;
    if bits.get_bit_count() != bit_count {
        return Ok(false);
    }
    for i in 0..bit_count {
        if bits
            .get_bit(i)
            .map_err(|e| format!("block2fn: get_bit failed: {e}"))?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn ir_value_is_all_ones(value: &IrValue, bit_count: usize) -> Result<bool, String> {
    let bits = value
        .to_bits()
        .map_err(|e| format!("block2fn: to_bits failed: {e}"))?;
    if bits.get_bit_count() != bit_count {
        return Ok(false);
    }
    for i in 0..bit_count {
        if !bits
            .get_bit(i)
            .map_err(|e| format!("block2fn: get_bit failed: {e}"))?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn ir_value_zero(bit_count: usize) -> Result<IrValue, String> {
    IrValue::make_ubits(bit_count, 0).map_err(|e| format!("block2fn: zero literal failed: {e}"))
}

fn ir_value_ones(bit_count: usize) -> Result<IrValue, String> {
    Ok(IrValue::from_bits(&IrBits::all_ones(bit_count)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;
    use crate::ir_query::{find_matching_nodes, matches_node, parse_query};

    #[test]
    fn no_op_eco_roundtrip_preserves_ordinary_blocks_exactly() {
        let golden = include_str!("../tests/goldens/block2fn/no_op_eco.ir");
        let mut package = Parser::new(golden).parse_and_validate_package().unwrap();
        for _ in 0..2 {
            for name in ["single", "multiple"] {
                let template = package.get_block(name).unwrap();
                let function = combinational_block_to_fn(template).unwrap();
                crate::ir_verify::verify_function(&function).unwrap();
                let replacement = replace_combinational_block_logic(template, function).unwrap();
                package.replace_block(name, replacement).unwrap();
            }
            crate::ir_verify::verify_package(&package).unwrap();
            assert_eq!(package.to_string(), golden);
        }
    }

    #[test]
    fn replacement_avoids_port_aliases_semantic_names_and_occupied_suffixes() {
        let mut package = Parser::new(
            r#"package aliases

top block sample(a: bits[8], out: bits[8], port_alias_2: clock) {
  port_alias: bits[8] = input_port(name=a, id=10)
  output_alias: () = output_port(port_alias, name=out, id=20)
}
"#,
        )
        .parse_and_validate_package()
        .unwrap();
        let function = Parser::new(
            r#"fn replacement(a: bits[8] id=1) -> bits[8] {
  port_alias: bits[8] = not(a, id=2)
  port_alias_1: bits[8] = identity(port_alias, id=3)
  output_alias: bits[8] = not(port_alias_1, id=4)
  output_alias_1: bits[8] = identity(output_alias, id=5)
  out: bits[8] = identity(output_alias_1, id=6)
  ret add.7: bits[8] = add(out, a, id=7)
}"#,
        )
        .parse_fn()
        .unwrap();
        crate::ir_verify::verify_function(&function).unwrap();
        let template = package.get_top_block().unwrap();
        let replacement = replace_combinational_block_logic(template, function.clone()).unwrap();
        let repeated = replace_combinational_block_logic(template, function).unwrap();
        assert_eq!(replacement.to_string(), repeated.to_string());
        package.replace_block("sample", replacement).unwrap();
        crate::ir_verify::verify_package(&package).unwrap();
        let golden = include_str!("../tests/goldens/block2fn/replacement_port_name_collisions.ir");
        assert_eq!(package.to_string(), golden);
        let reparsed = Parser::new(golden).parse_and_validate_package().unwrap();
        assert_eq!(reparsed.to_string(), golden);
        let projected = combinational_block_to_fn(reparsed.get_top_block().unwrap()).unwrap();
        crate::ir_verify::verify_function(&projected).unwrap();
        for value in [0, 1, 0xa5, 0xff] {
            let input = IrValue::make_ubits(8, value).unwrap();
            let crate::ir_eval::FnEvalResult::Success(result) =
                crate::ir_eval::eval_fn(&projected, &[input])
            else {
                panic!("replacement evaluation failed")
            };
            assert_eq!(
                result.value,
                IrValue::make_ubits(8, (value * 2) & 0xff).unwrap()
            );
        }
    }

    #[test]
    fn no_op_eco_roundtrip_preserves_observed_output_sink_unit_values() {
        let mut package = Parser::new(
            r#"package observed_sink

top block sample(x: bits[8], y: bits[8], z: (())) {
  x: bits[8] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
  observed: (()) = tuple(y, id=3)
  z: () = output_port(observed, name=z, id=4)
}
"#,
        )
        .parse_and_validate_package()
        .unwrap();
        for _ in 0..2 {
            let template = package.get_top_block().unwrap();
            let function = combinational_block_to_fn(template).unwrap();
            crate::ir_verify::verify_function(&function).unwrap();
            let replacement = replace_combinational_block_logic(template, function).unwrap();
            for name in ["y", "z"] {
                assert_eq!(
                    template
                        .get_node(template.get_output_port(name).unwrap())
                        .text_id,
                    replacement
                        .get_node(replacement.get_output_port(name).unwrap())
                        .text_id
                );
            }
            package.replace_block("sample", replacement).unwrap();
            crate::ir_verify::verify_package(&package).unwrap();
            let function = combinational_block_to_fn(package.get_top_block().unwrap()).unwrap();
            let input = IrValue::make_ubits(8, 42).unwrap();
            let crate::ir_eval::FnEvalResult::Success(result) =
                crate::ir_eval::eval_fn(&function, &[input.clone()])
            else {
                panic!("roundtrip evaluation failed")
            };
            assert_eq!(
                result.value,
                IrValue::make_tuple(&[input, IrValue::make_tuple(&[IrValue::make_tuple(&[])])])
            );
        }
    }

    #[test]
    fn combinational_projection_preserves_wide_aggregate_and_empty_interfaces() {
        for output_count in 0..=2 {
            let mut block = Block::new("wide");
            let input = block
                .add_input_port(
                    "data",
                    Type::Tuple(vec![
                        Box::new(Type::Bits(129)),
                        Box::new(Type::Array(ir::ArrayTypeData {
                            element_type: Box::new(Type::Bits(3)),
                            element_count: 2,
                        })),
                    ]),
                )
                .unwrap();
            for index in 0..output_count {
                block
                    .add_output_port(&format!("out{index}"), input)
                    .unwrap();
            }
            let function = combinational_block_to_fn(&block).unwrap();
            crate::ir_verify::verify_function(&function).unwrap();
            let input_value =
                IrValue::parse_typed("(bits[129]:0x10000000000000000, [bits[3]:1, bits[3]:7])")
                    .unwrap();
            let crate::ir_eval::FnEvalResult::Success(result) =
                crate::ir_eval::eval_fn(&function, &[input_value.clone()])
            else {
                panic!("projection evaluation failed")
            };
            let expected = if output_count == 1 {
                input_value.clone()
            } else {
                IrValue::make_tuple(&vec![input_value.clone(); output_count])
            };
            assert_eq!(result.value, expected);
            assert!(!function.nodes.iter().any(|node| matches!(
                node.payload,
                NodePayload::InputPort { .. } | NodePayload::OutputPort { .. }
            )));
        }
    }

    #[test]
    fn replacement_retains_port_identity_and_annotations_even_when_ids_collide() {
        let mut template = Block::new("top");
        let input = template.add_input_port("x", Type::Bits(129)).unwrap();
        let output = template.add_output_port("out", input).unwrap();
        template.add_clock_port("clk").unwrap();
        template.ports = vec![
            BlockPort::Output(output),
            BlockPort::Clock("clk".to_string()),
            BlockPort::Input(input),
        ];
        if let NodePayload::InputPort { sv_type, .. } = &mut template.get_node_mut(input).payload {
            *sv_type = Some("data_t".to_string());
        }
        if let NodePayload::OutputPort { sv_type, .. } = &mut template.get_node_mut(output).payload
        {
            *sv_type = Some("result_t".to_string());
        }
        let function = Parser::new(
            r#"fn replacement(x: bits[129] id=1) -> bits[129] {
  ret not.2: bits[129] = not(x, id=2)
}"#,
        )
        .parse_fn()
        .unwrap();
        let replacement = replace_combinational_block_logic(&template, function).unwrap();
        assert_eq!(replacement.name, "top");
        for (original, actual) in template.ports.iter().zip(&replacement.ports) {
            match (original, actual) {
                (BlockPort::Clock(lhs), BlockPort::Clock(rhs)) => assert_eq!(lhs, rhs),
                (BlockPort::Input(lhs), BlockPort::Input(rhs))
                | (BlockPort::Output(lhs), BlockPort::Output(rhs)) => {
                    assert_eq!(
                        template.get_node(*lhs).text_id,
                        replacement.get_node(*rhs).text_id
                    );
                    assert_eq!(template.port_name(*lhs), replacement.port_name(*rhs));
                    assert_eq!(template.port_type(*lhs), replacement.port_type(*rhs));
                }
                _ => panic!("port order or direction changed"),
            }
        }
        let replacement_input = replacement.get_input_port("x").unwrap();
        assert_eq!(
            template.get_node(input).payload,
            replacement.get_node(replacement_input).payload
        );
        let replacement_output = replacement.get_output_port("out").unwrap();
        assert!(matches!(&replacement.get_node(replacement_output).payload,
            NodePayload::OutputPort { sv_type: Some(sv_type), .. } if sv_type == "result_t"));
        let package = ir::Package {
            name: "p".to_string(),
            file_table: ir::FileTable::new(),
            members: vec![PackageMember::Block(replacement.clone())],
            top: Some(("top".to_string(), MemberType::Block)),
        };
        crate::ir_verify::verify_package(&package).unwrap();
        let projected = combinational_block_to_fn(&replacement).unwrap();
        let crate::ir_eval::FnEvalResult::Success(result) =
            crate::ir_eval::eval_fn(&projected, &[IrValue::from_bits(&IrBits::zero(129))])
        else {
            panic!("replacement evaluation failed")
        };
        assert_eq!(result.value, IrValue::from_bits(&IrBits::all_ones(129)));
    }

    #[test]
    fn combinational_projection_rejects_stateful_blocks() {
        let mut block = Block::new("state");
        block.registers.push(ir::Register {
            name: "r".to_string(),
            ty: Type::Bits(1),
            reset_value: None,
        });
        assert_eq!(
            combinational_block_to_fn(&block).unwrap_err(),
            "combinational block projection does not support registers or instantiations"
        );
    }

    fn bits1(value: &str) -> IrBits {
        IrValue::parse_typed(&format!("bits[1]:{value}"))
            .expect("parse literal")
            .to_bits()
            .expect("to_bits")
    }

    fn output_node_ref(block: &Block, name: &str) -> NodeRef {
        block.output_value(block.get_output_port(name).expect("output port"))
    }

    fn has_output_port(block: &Block, name: &str) -> bool {
        block.get_output_port(name).is_some()
    }

    fn assert_output_matches(block: &Block, name: &str, query_text: &str) {
        let node_ref = output_node_ref(block, name);
        let query = parse_query(query_text).expect("valid query");
        assert!(
            matches_node(block, &query, node_ref),
            "output '{name}' did not match '{query_text}'"
        );
    }

    fn assert_return_matches(function: &ir::Fn, query_text: &str) {
        let query = parse_query(query_text).expect("valid query");
        assert!(matches_node(
            function,
            &query,
            function.ret_node_ref.unwrap()
        ));
    }

    fn assert_no_matches(f: &ir::NodeGraph, query_text: &str) {
        let query = parse_query(query_text)
            .unwrap_or_else(|e| panic!("invalid query '{}': {e}", query_text));
        let matches = find_matching_nodes(f, &query);
        assert!(
            matches.is_empty(),
            "expected no matches for query '{}'",
            query_text
        );
    }

    const TWO_PARAM_BLOCK: &str = r#"package test

top block top(a: bits[1], b: bits[1], out0: bits[1], out1: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  b: bits[1] = input_port(name=b, id=2)
  out0: () = output_port(a, name=out0, id=3)
  out1: () = output_port(b, name=out1, id=4)
}
"#;

    const THREE_OUTPUT_BLOCK: &str = r#"package test

top block top(out0: bits[1], out1: bits[1], out2: bits[1]) {
  out0_lit: bits[1] = literal(value=0, id=1)
  out1_lit: bits[1] = literal(value=1, id=2)
  out2_lit: bits[1] = literal(value=0, id=3)
  out0: () = output_port(out0_lit, name=out0, id=4)
  out1: () = output_port(out1_lit, name=out1, id=5)
  out2: () = output_port(out2_lit, name=out2, id=6)
}
"#;

    fn run_block2fn(
        block_ir: &str,
        tie_input_ports: &[(&str, &str)],
        drop_output_ports: &[&str],
    ) -> ir::Fn {
        let mut tie_map = BTreeMap::new();
        for (name, value) in tie_input_ports.iter() {
            let bits = bits1(value);
            tie_map.insert((*name).to_string(), bits);
        }
        let drop_set: BTreeSet<String> = drop_output_ports.iter().map(|s| s.to_string()).collect();
        let opts = Block2FnOptions {
            tie_input_ports: tie_map,
            drop_output_ports: drop_set,
        };
        let result = block_ir_to_fn(block_ir, &opts).expect("block2fn should succeed");
        result.function
    }

    fn parse_top_block(block_ir: &str) -> Block {
        Parser::new(block_ir)
            .parse_and_validate_package()
            .unwrap()
            .get_block("top")
            .unwrap()
            .clone()
    }

    #[test]
    fn tie_input_ports_in_isolation_one_input() {
        let mut f = parse_top_block(TWO_PARAM_BLOCK);

        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::from([("a".to_string(), bits1("0"))]),
            drop_output_ports: BTreeSet::new(),
        };
        tie_input_ports(&mut f, &opts).expect("tie_input_ports succeeds");

        assert_output_matches(&f, "out0", "literal(0)");
        assert_output_matches(&f, "out1", "input_port(name=\"b\")");
    }

    #[test]
    fn tie_input_ports_in_isolation_two_inputs() {
        let mut f = parse_top_block(TWO_PARAM_BLOCK);

        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::from([
                ("a".to_string(), bits1("0")),
                ("b".to_string(), bits1("1")),
            ]),
            drop_output_ports: BTreeSet::new(),
        };
        tie_input_ports(&mut f, &opts).expect("tie_input_ports succeeds");

        assert_output_matches(&f, "out0", "literal(0)");
        assert_output_matches(&f, "out1", "literal(1)");
    }

    #[test]
    fn drop_output_ports_in_isolation_to_single_output() {
        let mut f = parse_top_block(THREE_OUTPUT_BLOCK);
        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::new(),
            drop_output_ports: BTreeSet::from(["out0".to_string(), "out2".to_string()]),
        };
        drop_output_ports(&mut f, &opts).expect("drop_output_ports succeeds");

        assert!(has_output_port(&f, "out1"));
        assert!(!has_output_port(&f, "out0"));
        assert!(!has_output_port(&f, "out2"));
        assert_eq!(f.output_ports().count(), 1, "dropped outputs removed");

        assert_eq!(
            f.port_type(f.get_output_port("out1").unwrap()),
            &Type::Bits(1)
        );
        assert_output_matches(&f, "out1", "literal(1)");
    }

    #[test]
    fn block2fn_errors_if_output_count_not_one() {
        let block_ir = r#"package test

top block top(a: bits[1], b: bits[1], out0: bits[1], out1: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  b: bits[1] = input_port(name=b, id=2)
  out0: () = output_port(a, name=out0, id=3)
  out1: () = output_port(b, name=out1, id=4)
}
"#;
        let opts = Block2FnOptions::default();
        let err = block_ir_to_fn(block_ir, &opts).unwrap_err();
        assert!(
            err.contains("expected exactly one output port"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tie_inputs_then_collapse_registers() {
        let block_ir = r#"package test

top block top(clk: clock, a: bits[1], out: bits[1]) {
  reg r(bits[1])
  a: bits[1] = input_port(name=a, id=1)
  r_q: bits[1] = register_read(register=r, id=2)
  and.3: bits[1] = and(a, r_q, id=3)
  r_d: () = register_write(and.3, register=r, id=4)
  out: () = output_port(and.3, name=out, id=5)
}
"#;
        let f = run_block2fn(block_ir, &[("a", "0")], &[]);
        assert_eq!(f.params.len(), 0, "tied inputs should be removed");
        let f_str = f.to_string();
        assert!(
            f_str.contains("literal(value=0, id="),
            "expected constant propagation to introduce literal"
        );
        assert!(f_str.contains("ret "), "expected function to have return");
    }

    #[test]
    fn tie_input_port_replaces_uses_with_literal() {
        let block_ir = r#"package test

top block top(a: bits[1], out: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  out: () = output_port(a, name=out, id=2)
}
"#;
        let f = run_block2fn(block_ir, &[("a", "1")], &[]);
        assert_eq!(f.params.len(), 0, "tied input should be removed");
        assert_no_matches(&f, "get_param(name=\"a\")");
        assert_return_matches(&f, "literal(1)");
    }

    #[test]
    fn tie_two_input_ports_replaces_both_uses() {
        let block_ir = r#"package test

top block top(a: bits[1], b: bits[1], out: (bits[1], bits[1])) {
  a: bits[1] = input_port(name=a, id=1)
  b: bits[1] = input_port(name=b, id=2)
  tuple.3: (bits[1], bits[1]) = tuple(a, b, id=3)
  out: () = output_port(tuple.3, name=out, id=4)
}
"#;
        let mut f = parse_top_block(block_ir);
        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::from([
                ("a".to_string(), bits1("0")),
                ("b".to_string(), bits1("1")),
            ]),
            drop_output_ports: BTreeSet::new(),
        };
        tie_input_ports(&mut f, &opts).expect("tie_input_ports succeeds");
        assert_eq!(f.input_ports().count(), 0, "tied inputs should be removed");
        assert_no_matches(&f, "get_param(name=\"a\")");
        assert_no_matches(&f, "get_param(name=\"b\")");
        assert_output_matches(&f, "out", "tuple(literal(0), literal(1))");
    }

    #[test]
    fn drop_output_ports_updates_return() {
        let block_ir = r#"package test

top block top(a: bits[1], b: bits[1], out0: bits[1], out1: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  b: bits[1] = input_port(name=b, id=2)
  out0: () = output_port(a, name=out0, id=3)
  out1: () = output_port(b, name=out1, id=4)
}
"#;
        let mut f = parse_top_block(block_ir);
        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::new(),
            drop_output_ports: BTreeSet::from(["out1".to_string()]),
        };
        drop_output_ports(&mut f, &opts).expect("drop_output_ports succeeds");
        assert!(has_output_port(&f, "out0"));
        assert!(!has_output_port(&f, "out1"));
        assert_output_matches(&f, "out0", "input_port(name=\"a\")");
    }
}
