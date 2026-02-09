// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::block_inline::inline_all_blocks_in_package;
use crate::dce::remove_dead_nodes;
use crate::ir::{self, BlockMetadata, MemberType, NodePayload, NodeRef, PackageMember, Type};
use crate::ir_eval::eval_pure_if_supported;
use crate::ir_parser::Parser;
use crate::ir_utils::{
    compact_and_toposort_in_place, get_topological_nodes, operands, remap_payload_with,
    verify_no_cycle,
};
use xlsynth::{IrBits, IrValue};

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
    let (mut f, mut metadata) = match top_block {
        PackageMember::Block { func, metadata } => (func.clone(), metadata.clone()),
        _ => {
            return Err("block2fn: top member is not a block".to_string());
        }
    };

    if !metadata.instantiations.is_empty() {
        return Err(
            "block2fn: block contains instantiations; run block_inlining first".to_string(),
        );
    }

    tie_input_ports(&mut f, &mut metadata, options)?;
    strip_clock_port(&mut f, &mut metadata)?;
    drop_output_ports(&mut f, &mut metadata, options)?;

    if metadata.output_names.len() != 1 {
        return Err(format!(
            "block2fn: expected exactly one output port after dropping outputs; got {}",
            metadata.output_names.len()
        ));
    }

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

fn tie_input_ports(
    f: &mut ir::Fn,
    metadata: &mut BlockMetadata,
    options: &Block2FnOptions,
) -> Result<(), String> {
    if options.tie_input_ports.is_empty() {
        return Ok(());
    }

    let mut literal_by_port: HashMap<String, IrBits> = HashMap::new();
    for (name, literal) in options.tie_input_ports.iter() {
        let param = f
            .params
            .iter()
            .find(|p| p.name == *name)
            .ok_or_else(|| format!("block2fn: unknown input port '{}'", name))?;
        let value = parse_literal_for_type(literal, &param.ty)?;
        literal_by_port.insert(name.clone(), value);
    }

    let mut max_id = f.nodes.iter().map(|n| n.text_id).max().unwrap_or(0);
    let mut tied_param_ids: HashSet<ir::ParamId> = HashSet::new();
    let params = f.params.clone();
    for param in params.iter() {
        let Some(literal) = literal_by_port.get(&param.name) else {
            continue;
        };
        let param_node_index = find_get_param_node_index(f, param.id)
            .ok_or_else(|| format!("block2fn: missing GetParam for '{}'", param.name))?;
        max_id += 1;
        let literal_node = ir::Node {
            text_id: max_id,
            name: Some(format!("{}_tied", param.name)),
            ty: param.ty.clone(),
            payload: NodePayload::Literal(IrValue::from_bits(literal)),
            pos: None,
        };
        let literal_ref = NodeRef {
            index: f.nodes.len(),
        };
        f.nodes.push(literal_node);
        crate::ir_utils::replace_node_with_ref(
            f,
            NodeRef {
                index: param_node_index,
            },
            literal_ref,
        )
        .map_err(|e| format!("block2fn: tie input '{}': {e}", param.name))?;
        tied_param_ids.insert(param.id);
        metadata.input_port_ids.remove(&param.name);
    }

    f.params.retain(|p| !tied_param_ids.contains(&p.id));
    reorder_params_and_compact(f)?;
    Ok(())
}

fn strip_clock_port(f: &mut ir::Fn, metadata: &mut BlockMetadata) -> Result<(), String> {
    let Some(clock_name) = metadata.clock_port_name.clone() else {
        return Ok(());
    };

    let Some(param_id) = f.params.iter().find(|p| p.name == clock_name).map(|p| p.id) else {
        metadata.clock_port_name = None;
        return Ok(());
    };

    let Some(param_node_index) = find_get_param_node_index(f, param_id) else {
        metadata.clock_port_name = None;
        return Ok(());
    };
    let param_ref = NodeRef {
        index: param_node_index,
    };

    let uses: Vec<ir::NodeRef> = f
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| {
            if idx == param_node_index {
                return None;
            }
            let deps = operands(&node.payload);
            if deps.iter().any(|nr| *nr == param_ref) {
                Some(NodeRef { index: idx })
            } else {
                None
            }
        })
        .collect();

    if !uses.is_empty() {
        let mut text_ids: Vec<usize> = uses.iter().map(|nr| f.nodes[nr.index].text_id).collect();
        text_ids.sort_unstable();
        return Err(format!(
            "block2fn: clock port '{}' is used by non-register nodes (ids: {})",
            clock_name,
            text_ids
                .into_iter()
                .map(|id| id.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        ));
    }

    f.nodes[param_node_index].payload = NodePayload::Nil;
    f.params.retain(|p| p.id != param_id);
    metadata.input_port_ids.remove(&clock_name);
    metadata.clock_port_name = None;
    reorder_params_and_compact(f)?;
    Ok(())
}

fn drop_output_ports(
    f: &mut ir::Fn,
    metadata: &mut BlockMetadata,
    options: &Block2FnOptions,
) -> Result<(), String> {
    if options.drop_output_ports.is_empty() {
        return Ok(());
    }

    // Move the original outputs out so we can build the retained output list by
    // *moving* strings (no cloning). We'll restore/update `metadata` on all
    // paths.
    let original_outputs = std::mem::take(&mut metadata.output_names);
    let unknown_outputs: Vec<String> = options
        .drop_output_ports
        .iter()
        .filter(|name| original_outputs.iter().all(|o| o != *name))
        .cloned()
        .collect();
    if !unknown_outputs.is_empty() {
        metadata.output_names = original_outputs;
        return Err(format!(
            "block2fn: unknown output ports: {}",
            unknown_outputs.join(", ")
        ));
    }

    let retained_count = original_outputs
        .iter()
        .filter(|name| !options.drop_output_ports.contains(*name))
        .count();
    if retained_count == 0 {
        metadata.output_names = original_outputs;
        return Err("block2fn: all outputs dropped; at least one output required".to_string());
    }

    let ret_ref = match f.ret_node_ref {
        Some(r) => r,
        None => {
            metadata.output_names = original_outputs;
            return Err("block2fn: function missing return node".to_string());
        }
    };

    if original_outputs.len() == 1 {
        // Single output block: nothing to do unless it was dropped, which is
        // already handled above.
        metadata.output_names = original_outputs;
        let retained_set: HashSet<&str> =
            metadata.output_names.iter().map(|s| s.as_str()).collect();
        metadata
            .output_port_ids
            .retain(|name, _| retained_set.contains(name.as_str()));
        return Ok(());
    }

    let elems = match &f.nodes[ret_ref.index].payload {
        NodePayload::Tuple(elems) => elems,
        _ => {
            metadata.output_names = original_outputs;
            return Err(
                "block2fn: expected tuple return when dropping outputs from multi-output block"
                    .to_string(),
            );
        }
    };
    if elems.len() != original_outputs.len() {
        metadata.output_names = original_outputs;
        return Err("block2fn: output tuple arity does not match output names".to_string());
    }
    let mut kept_elems: Vec<NodeRef> = Vec::new();
    let mut kept_types: Vec<Type> = Vec::new();
    for (idx, name) in original_outputs.iter().enumerate() {
        if !options.drop_output_ports.contains(name) {
            let nr = elems[idx];
            kept_elems.push(nr);
            kept_types.push(f.nodes[nr.index].ty.clone());
        }
    }

    if kept_elems.len() == 1 {
        f.ret_ty = kept_types.remove(0);
        f.ret_node_ref = Some(kept_elems[0]);
        f.nodes[ret_ref.index].payload = NodePayload::Nil;

        let retained_outputs: Vec<String> = original_outputs
            .into_iter()
            .filter(|name| !options.drop_output_ports.contains(name))
            .collect();
        metadata.output_names = retained_outputs;
        let retained_set: HashSet<&str> =
            metadata.output_names.iter().map(|s| s.as_str()).collect();
        metadata
            .output_port_ids
            .retain(|name, _| retained_set.contains(name.as_str()));
        return Ok(());
    }

    let new_ret_ty = Type::Tuple(kept_types.into_iter().map(Box::new).collect());
    let new_payload = NodePayload::Tuple(kept_elems);
    if let Err(e) =
        crate::ir_utils::replace_node_payload(f, ret_ref, new_payload, Some(new_ret_ty.clone()))
    {
        metadata.output_names = original_outputs;
        return Err(format!("block2fn: update tuple return failed: {e}"));
    }
    f.ret_ty = new_ret_ty;

    let retained_outputs: Vec<String> = original_outputs
        .into_iter()
        .filter(|name| !options.drop_output_ports.contains(name))
        .collect();
    metadata.output_names = retained_outputs;
    let retained_set: HashSet<&str> = metadata.output_names.iter().map(|s| s.as_str()).collect();
    metadata
        .output_port_ids
        .retain(|name, _| retained_set.contains(name.as_str()));
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
            let mut env: HashMap<ir::NodeRef, IrValue> = HashMap::with_capacity(deps.len());
            for nr in deps.iter().copied() {
                let Some(value) = const_nodes.get(&nr.index) else {
                    continue;
                };
                env.insert(nr, value.clone());
            }
            if let Some(value) = eval_pure_if_supported(node, &env) {
                newly_constant.push((idx, value));
            }
        }
        if newly_constant.is_empty() {
            break;
        }
        for (idx, value) in newly_constant {
            let node_ref = NodeRef { index: idx };
            crate::ir_utils::replace_node_payload(
                f,
                node_ref,
                NodePayload::Literal(value.clone()),
                Some(f.nodes[idx].ty.clone()),
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

fn find_get_param_node_index(f: &ir::Fn, pid: ir::ParamId) -> Option<usize> {
    f.nodes
        .iter()
        .position(|n| matches!(n.payload, NodePayload::GetParam(id) if id == pid))
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
    if bit_count == 0 {
        return Ok(IrValue::make_ubits(0, 0).expect("bits[0] should be valid"));
    }
    IrValue::parse_typed(&format!("bits[{bit_count}]:0"))
        .map_err(|e| format!("block2fn: zero literal failed: {e}"))
}

fn ir_value_ones(bit_count: usize) -> Result<IrValue, String> {
    if bit_count == 0 {
        return Ok(IrValue::make_ubits(0, 0).expect("bits[0] should be valid"));
    }
    let ones = "1".repeat(bit_count);
    IrValue::parse_typed(&format!("bits[{bit_count}]:0b{ones}"))
        .map_err(|e| format!("block2fn: ones literal failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;
    use crate::ir_query::{find_matching_nodes, matches_node, parse_query};

    fn bits1(value: &str) -> IrBits {
        IrValue::parse_typed(&format!("bits[1]:{value}"))
            .expect("parse literal")
            .to_bits()
            .expect("to_bits")
    }

    fn output_node_ref(f: &ir::Fn, metadata: &BlockMetadata, name: &str) -> NodeRef {
        let idx = metadata
            .output_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("output '{}' not found", name));
        let ret_ref = f.ret_node_ref.expect("expected return node");
        if metadata.output_names.len() == 1 {
            return ret_ref;
        }
        let NodePayload::Tuple(elems) = &f.nodes[ret_ref.index].payload else {
            panic!("expected tuple return for multi-output block");
        };
        elems[idx]
    }

    fn has_output_port(metadata: &BlockMetadata, name: &str) -> bool {
        metadata.output_names.iter().any(|n| n == name)
            && metadata.output_port_ids.contains_key(name)
    }

    fn assert_output_matches(f: &ir::Fn, metadata: &BlockMetadata, name: &str, query_text: &str) {
        let node_ref = output_node_ref(f, metadata, name);
        let query = parse_query(query_text)
            .unwrap_or_else(|e| panic!("invalid query '{}': {e}", query_text));
        if !matches_node(f, &query, node_ref) {
            panic!("output '{}' did not match query '{}'", name, query_text);
        }
    }

    fn assert_no_matches(f: &ir::Fn, query_text: &str) {
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

    fn parse_top_block(block_ir: &str) -> (ir::Fn, BlockMetadata) {
        let mut parser = Parser::new(block_ir);
        let pkg = parser
            .parse_and_validate_package()
            .expect("parse package should succeed");
        let PackageMember::Block { func, metadata } = pkg.get_block("top").unwrap() else {
            panic!("expected block");
        };
        (func.clone(), metadata.clone())
    }

    #[test]
    fn tie_input_ports_in_isolation_one_input() {
        let (mut f, mut metadata) = parse_top_block(TWO_PARAM_BLOCK);
        f.check_pir_layout_invariants()
            .expect("precondition: PIR layout invariants");

        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::from([("a".to_string(), bits1("0"))]),
            drop_output_ports: BTreeSet::new(),
        };
        tie_input_ports(&mut f, &mut metadata, &opts).expect("tie_input_ports succeeds");
        f.check_pir_layout_invariants()
            .expect("postcondition: PIR layout invariants");

        assert_output_matches(&f, &metadata, "out0", "literal(0)");
        assert_output_matches(&f, &metadata, "out1", "get_param(name=\"b\")");
    }

    #[test]
    fn tie_input_ports_in_isolation_two_inputs() {
        let (mut f, mut metadata) = parse_top_block(TWO_PARAM_BLOCK);
        f.check_pir_layout_invariants()
            .expect("precondition: PIR layout invariants");

        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::from([
                ("a".to_string(), bits1("0")),
                ("b".to_string(), bits1("1")),
            ]),
            drop_output_ports: BTreeSet::new(),
        };
        tie_input_ports(&mut f, &mut metadata, &opts).expect("tie_input_ports succeeds");
        f.check_pir_layout_invariants()
            .expect("postcondition: PIR layout invariants");

        assert_output_matches(&f, &metadata, "out0", "literal(0)");
        assert_output_matches(&f, &metadata, "out1", "literal(1)");
    }

    #[test]
    fn drop_output_ports_in_isolation_to_single_output() {
        let (mut f, mut metadata) = parse_top_block(THREE_OUTPUT_BLOCK);
        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::new(),
            drop_output_ports: BTreeSet::from(["out0".to_string(), "out2".to_string()]),
        };
        drop_output_ports(&mut f, &mut metadata, &opts).expect("drop_output_ports succeeds");

        assert!(has_output_port(&metadata, "out1"));
        assert!(!has_output_port(&metadata, "out0"));
        assert!(!has_output_port(&metadata, "out2"));
        assert_eq!(metadata.output_port_ids.len(), 1, "dropped outputs removed");

        assert_eq!(f.ret_ty, Type::Bits(1), "single-output return type");
        assert_output_matches(&f, &metadata, "out1", "literal(1)");
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

top block top(a: bits[1], out: bits[1]) {
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
        let (_orig_f, metadata) = parse_top_block(block_ir);
        assert_no_matches(&f, "get_param(name=\"a\")");
        assert_output_matches(&f, &metadata, "out", "literal(1)");
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
        let (mut f, mut metadata) = parse_top_block(block_ir);
        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::from([
                ("a".to_string(), bits1("0")),
                ("b".to_string(), bits1("1")),
            ]),
            drop_output_ports: BTreeSet::new(),
        };
        tie_input_ports(&mut f, &mut metadata, &opts).expect("tie_input_ports succeeds");
        assert_eq!(f.params.len(), 0, "tied inputs should be removed");
        assert_no_matches(&f, "get_param(name=\"a\")");
        assert_no_matches(&f, "get_param(name=\"b\")");
        assert_output_matches(&f, &metadata, "out", "tuple(literal(0), literal(1))");
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
        let (mut f, mut metadata) = parse_top_block(block_ir);
        let opts = Block2FnOptions {
            tie_input_ports: BTreeMap::new(),
            drop_output_ports: BTreeSet::from(["out1".to_string()]),
        };
        drop_output_ports(&mut f, &mut metadata, &opts).expect("drop_output_ports succeeds");
        assert!(has_output_port(&metadata, "out0"));
        assert!(!has_output_port(&metadata, "out1"));
        assert_output_matches(&f, &metadata, "out0", "get_param(name=\"a\")");
    }
}
