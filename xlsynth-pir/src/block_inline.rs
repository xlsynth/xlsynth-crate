// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};

use crate::ir::{self, BlockMetadata, MemberType, NodePayload, NodeRef, PackageMember, Register};
use crate::ir_utils::{compact_and_toposort_in_place, get_topological, remap_payload_with};

// Inlines all block instantiations in the package. All blocks are removed after
// inlining except for the top block.
pub fn inline_all_blocks_in_package(pkg: &mut ir::Package) -> Result<(), String> {
    loop {
        let mut progress = false;
        let mut has_instantiations = false;
        let block_names: Vec<String> = pkg
            .members
            .iter()
            .filter_map(|m| match m {
                PackageMember::Block { func, .. } => Some(func.name.clone()),
                _ => None,
            })
            .collect();
        for name in block_names {
            let mut block = match pkg.get_block(&name) {
                Some(PackageMember::Block { func, metadata }) => (func.clone(), metadata.clone()),
                _ => continue,
            };
            if block.1.instantiations.is_empty() {
                continue;
            }
            has_instantiations = true;
            let inlined = inline_block_instantiations(pkg, &mut block)?;
            if inlined {
                pkg.replace_block(
                    &name,
                    PackageMember::Block {
                        func: block.0,
                        metadata: block.1,
                    },
                )?;
                progress = true;
            }
        }
        if !has_instantiations {
            break;
        }
        if !progress {
            return Err(
                "block_inline: could not inline remaining instantiations (cycle?)".to_string(),
            );
        }
    }
    prune_to_single_top_block(pkg)?;
    Ok(())
}

fn inline_block_instantiations(
    pkg: &ir::Package,
    block: &mut (ir::Fn, BlockMetadata),
) -> Result<bool, String> {
    let mut inlined_any = false;
    loop {
        let instantiations = block.1.instantiations.clone();
        let mut did_inline = false;
        for inst in instantiations {
            let callee = match pkg.get_block(&inst.block) {
                Some(PackageMember::Block { func, metadata }) => (func.clone(), metadata.clone()),
                _ => {
                    return Err(format!(
                        "block_inline: instantiation '{}' references missing block '{}'",
                        inst.name, inst.block
                    ));
                }
            };
            if !callee.1.instantiations.is_empty() {
                continue;
            }
            inline_single_instantiation(&mut block.0, &mut block.1, &inst.name, &callee)?;
            did_inline = true;
            inlined_any = true;
            break;
        }
        if !did_inline {
            break;
        }
    }
    Ok(inlined_any)
}

fn inline_single_instantiation(
    caller_fn: &mut ir::Fn,
    caller_meta: &mut BlockMetadata,
    inst_name: &str,
    callee: &(ir::Fn, BlockMetadata),
) -> Result<(), String> {
    let (callee_fn, callee_meta) = callee;
    let input_map = collect_instantiation_inputs(caller_fn, inst_name)?;
    let output_map = collect_callee_outputs(callee_fn, callee_meta)?;

    let mut register_name_map: HashMap<String, String> = HashMap::new();
    let mut used_register_names: HashMap<String, usize> = caller_meta
        .registers
        .iter()
        .map(|r| (r.name.clone(), 1))
        .collect();
    for reg in callee_meta.registers.iter() {
        let base_name = format!("{}__{}", inst_name, reg.name);
        let unique_name = uniquify_register_name(&base_name, &mut used_register_names);
        register_name_map.insert(reg.name.clone(), unique_name.clone());
        caller_meta.registers.push(Register {
            name: unique_name,
            ty: reg.ty.clone(),
            reset_value: reg.reset_value.clone(),
        });
    }

    let mut max_text_id = caller_fn.nodes.iter().map(|n| n.text_id).max().unwrap_or(0);
    let mut used_node_names: HashMap<String, usize> = caller_fn
        .nodes
        .iter()
        .filter_map(|n| n.name.clone())
        .map(|name| (name, 1))
        .collect();
    let mut mapping: HashMap<usize, NodeRef> = HashMap::new();
    let param_name_by_id: HashMap<ir::ParamId, String> = callee_fn
        .params
        .iter()
        .map(|p| (p.id, p.name.clone()))
        .collect();

    let topo = get_topological(callee_fn);
    for nr in topo {
        if nr.index == 0 {
            continue;
        }
        let node = callee_fn.get_node(nr);
        match &node.payload {
            NodePayload::GetParam(pid) => {
                let name = param_name_by_id
                    .get(pid)
                    .ok_or_else(|| "block_inline: missing param name".to_string())?;
                let arg = input_map.get(name).ok_or_else(|| {
                    format!(
                        "block_inline: instantiation '{}' missing input '{}'",
                        inst_name, name
                    )
                })?;
                mapping.insert(nr.index, *arg);
            }
            NodePayload::InstantiationInput { .. } | NodePayload::InstantiationOutput { .. } => {
                return Err(format!(
                    "block_inline: callee '{}' still has instantiation nodes",
                    callee_fn.name
                ));
            }
            _ => {
                let new_payload =
                    remap_payload_with(&node.payload, |(_, dep): (usize, NodeRef)| {
                        mapping.get(&dep.index).copied().unwrap_or_else(|| {
                            panic!("block_inline: missing mapping for {}", dep.index)
                        })
                    });
                let new_payload = rewrite_register_payload(new_payload, &register_name_map);
                max_text_id += 1;
                let new_node = ir::Node {
                    text_id: max_text_id,
                    name: node.name.as_ref().map(|n| {
                        let base = format!("{}__{}", inst_name, n);
                        uniquify_node_name(&base, &mut used_node_names)
                    }),
                    ty: node.ty.clone(),
                    payload: new_payload,
                    pos: None,
                };
                let new_ref = NodeRef {
                    index: caller_fn.nodes.len(),
                };
                caller_fn.nodes.push(new_node);
                mapping.insert(nr.index, new_ref);
            }
        }
    }

    let mut output_replacements: Vec<(NodeRef, NodeRef)> = Vec::new();
    let mut input_nodes_to_nil: Vec<NodeRef> = Vec::new();
    for (idx, node) in caller_fn.nodes.iter().enumerate() {
        match &node.payload {
            NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } if instantiation == inst_name => {
                let callee_out = output_map.get(port_name).ok_or_else(|| {
                    format!(
                        "block_inline: instantiation '{}' missing output '{}'",
                        inst_name, port_name
                    )
                })?;
                let mapped = mapping.get(&callee_out.index).copied().ok_or_else(|| {
                    format!("block_inline: missing mapping for output '{}'", port_name)
                })?;
                output_replacements.push((NodeRef { index: idx }, mapped));
            }
            NodePayload::InstantiationInput { instantiation, .. } if instantiation == inst_name => {
                input_nodes_to_nil.push(NodeRef { index: idx });
            }
            _ => {}
        }
    }

    for (target, replacement) in output_replacements {
        crate::ir_utils::replace_node_with_ref(caller_fn, target, replacement)
            .map_err(|e| format!("block_inline: replace output failed: {e}"))?;
    }
    for target in input_nodes_to_nil {
        caller_fn.nodes[target.index].payload = NodePayload::Nil;
    }

    caller_meta
        .instantiations
        .retain(|inst| inst.name != inst_name);

    compact_and_toposort_in_place(caller_fn)
        .map_err(|e| format!("block_inline: compaction failed: {e}"))?;
    Ok(())
}

fn collect_instantiation_inputs(
    caller_fn: &ir::Fn,
    inst_name: &str,
) -> Result<BTreeMap<String, NodeRef>, String> {
    let mut inputs = BTreeMap::new();
    for node in caller_fn.nodes.iter() {
        if let NodePayload::InstantiationInput {
            instantiation,
            port_name,
            arg,
        } = &node.payload
        {
            if instantiation == inst_name {
                if inputs.insert(port_name.clone(), *arg).is_some() {
                    return Err(format!(
                        "block_inline: duplicate input '{}' for instantiation '{}'",
                        port_name, inst_name
                    ));
                }
            }
        }
    }
    Ok(inputs)
}

fn collect_callee_outputs(
    callee_fn: &ir::Fn,
    callee_meta: &BlockMetadata,
) -> Result<BTreeMap<String, NodeRef>, String> {
    let ret_ref = callee_fn
        .ret_node_ref
        .ok_or_else(|| "block_inline: callee has no return node".to_string())?;
    let mut outputs = BTreeMap::new();
    if callee_meta.output_names.len() == 1 {
        outputs.insert(callee_meta.output_names[0].clone(), ret_ref);
        return Ok(outputs);
    }
    let NodePayload::Tuple(elems) = &callee_fn.nodes[ret_ref.index].payload else {
        return Err("block_inline: expected tuple return for multi-output block".to_string());
    };
    if elems.len() != callee_meta.output_names.len() {
        return Err("block_inline: output arity mismatch".to_string());
    }
    for (name, elem) in callee_meta.output_names.iter().zip(elems.iter()) {
        outputs.insert(name.clone(), *elem);
    }
    Ok(outputs)
}

fn rewrite_register_payload(
    payload: NodePayload,
    reg_map: &HashMap<String, String>,
) -> NodePayload {
    match payload {
        NodePayload::RegisterRead { register } => NodePayload::RegisterRead {
            register: reg_map.get(&register).cloned().unwrap_or(register),
        },
        NodePayload::RegisterWrite {
            arg,
            register,
            load_enable,
            reset,
        } => NodePayload::RegisterWrite {
            arg,
            register: reg_map.get(&register).cloned().unwrap_or(register),
            load_enable,
            reset,
        },
        _ => payload,
    }
}

fn uniquify_register_name(base: &str, used: &mut HashMap<String, usize>) -> String {
    if !used.contains_key(base) {
        used.insert(base.to_string(), 1);
        return base.to_string();
    }
    let mut counter = *used.get(base).unwrap_or(&1);
    loop {
        let candidate = format!("{}__{}", base, counter);
        if !used.contains_key(&candidate) {
            used.insert(base.to_string(), counter + 1);
            used.insert(candidate.clone(), 1);
            return candidate;
        }
        counter += 1;
    }
}

fn uniquify_node_name(base: &str, used: &mut HashMap<String, usize>) -> String {
    if !used.contains_key(base) {
        used.insert(base.to_string(), 1);
        return base.to_string();
    }
    let mut counter = *used.get(base).unwrap_or(&1);
    loop {
        let candidate = format!("{}__{}", base, counter);
        if !used.contains_key(&candidate) {
            used.insert(base.to_string(), counter + 1);
            used.insert(candidate.clone(), 1);
            return candidate;
        }
        counter += 1;
    }
}

fn prune_to_single_top_block(pkg: &mut ir::Package) -> Result<(), String> {
    let top_name = match &pkg.top {
        Some((name, MemberType::Block)) => name.clone(),
        Some((_name, MemberType::Function)) => {
            return Err("block_inline: package top is a function".to_string());
        }
        None => {
            let first = pkg.members.iter().find_map(|m| match m {
                PackageMember::Block { func, .. } => Some(func.name.clone()),
                _ => None,
            });
            first.ok_or_else(|| "block_inline: no block members to keep".to_string())?
        }
    };
    pkg.members.retain(|m| match m {
        PackageMember::Block { func, .. } => func.name == top_name,
        _ => false,
    });
    pkg.top = Some((top_name, MemberType::Block));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;
    use crate::ir_query::{matches_node, parse_query};

    fn parse_pkg(text: &str) -> ir::Package {
        let mut parser = Parser::new(text);
        parser
            .parse_and_validate_package()
            .expect("parse package should succeed")
    }

    fn assert_no_instantiation_nodes(f: &ir::Fn) {
        for node in f.nodes.iter() {
            match node.payload {
                NodePayload::InstantiationInput { .. }
                | NodePayload::InstantiationOutput { .. } => {
                    panic!("unexpected instantiation node: {:?}", node.payload);
                }
                _ => {}
            }
        }
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

    fn assert_output_matches(f: &ir::Fn, metadata: &BlockMetadata, name: &str, query_text: &str) {
        let node_ref = output_node_ref(f, metadata, name);
        let query = parse_query(query_text)
            .unwrap_or_else(|e| panic!("invalid query '{}': {e}", query_text));
        if !matches_node(f, &query, node_ref) {
            panic!("output '{}' did not match query '{}'", name, query_text);
        }
    }

    #[test]
    fn inline_no_instantiations_is_noop() {
        let pkg_text = r#"package test

top block top(a: bits[1], y: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  y: () = output_port(a, name=y, id=2)
}
"#;
        let mut pkg = parse_pkg(pkg_text);
        inline_all_blocks_in_package(&mut pkg).expect("inline should succeed");
        assert_eq!(
            pkg.members.len(),
            1,
            "expected a single block after inlining"
        );
        let PackageMember::Block { func, metadata } = pkg.get_block("top").unwrap() else {
            panic!("expected block");
        };
        assert!(metadata.instantiations.is_empty());
        assert_no_instantiation_nodes(func);
        assert_output_matches(func, metadata, "y", "get_param(name=\"a\")");
    }

    #[test]
    fn inline_single_instantiation() {
        let pkg_text = r#"package test

block leaf(a: bits[1], y: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  not.2: bits[1] = not(a, id=2)
  y: () = output_port(not.2, name=y, id=3)
}

top block top(a: bits[1], y: bits[1]) {
  instantiation u0(block=leaf, kind=block)
  a: bits[1] = input_port(name=a, id=11)
  instantiation_output.12: bits[1] = instantiation_output(instantiation=u0, port_name=y, id=12)
  instantiation_input.13: () = instantiation_input(a, instantiation=u0, port_name=a, id=13)
  y: () = output_port(instantiation_output.12, name=y, id=14)
}
"#;
        let mut pkg = parse_pkg(pkg_text);
        inline_all_blocks_in_package(&mut pkg).expect("inline should succeed");
        assert_eq!(
            pkg.members.len(),
            1,
            "expected a single block after inlining"
        );
        let PackageMember::Block { func, metadata } = pkg.get_block("top").unwrap() else {
            panic!("expected block");
        };
        assert!(metadata.instantiations.is_empty());
        assert_no_instantiation_nodes(func);
        assert_output_matches(func, metadata, "y", "not(get_param(name=\"a\"))");
    }

    #[test]
    fn inline_double_instantiation() {
        let pkg_text = r#"package test

block leaf(a: bits[1], y: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  not.2: bits[1] = not(a, id=2)
  y: () = output_port(not.2, name=y, id=3)
}

top block top(a: bits[1], y0: bits[1], y1: bits[1]) {
  instantiation u0(block=leaf, kind=block)
  instantiation u1(block=leaf, kind=block)
  a: bits[1] = input_port(name=a, id=11)
  instantiation_output.12: bits[1] = instantiation_output(instantiation=u0, port_name=y, id=12)
  instantiation_output.13: bits[1] = instantiation_output(instantiation=u1, port_name=y, id=13)
  instantiation_input.14: () = instantiation_input(a, instantiation=u0, port_name=a, id=14)
  instantiation_input.15: () = instantiation_input(a, instantiation=u1, port_name=a, id=15)
  y0: () = output_port(instantiation_output.12, name=y0, id=16)
  y1: () = output_port(instantiation_output.13, name=y1, id=17)
}
"#;
        let mut pkg = parse_pkg(pkg_text);
        inline_all_blocks_in_package(&mut pkg).expect("inline should succeed");
        assert_eq!(
            pkg.members.len(),
            1,
            "expected a single block after inlining"
        );
        let PackageMember::Block { func, metadata } = pkg.get_block("top").unwrap() else {
            panic!("expected block");
        };
        assert!(metadata.instantiations.is_empty());
        assert_no_instantiation_nodes(func);
        assert_output_matches(func, metadata, "y0", "not(get_param(name=\"a\"))");
        assert_output_matches(func, metadata, "y1", "not(get_param(name=\"a\"))");
    }

    #[test]
    fn inline_nested_instantiations() {
        let pkg_text = r#"package test

block leaf(a: bits[1], y: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  not.2: bits[1] = not(a, id=2)
  y: () = output_port(not.2, name=y, id=3)
}

block mid(a: bits[1], y: bits[1]) {
  instantiation u0(block=leaf, kind=block)
  a: bits[1] = input_port(name=a, id=11)
  instantiation_output.12: bits[1] = instantiation_output(instantiation=u0, port_name=y, id=12)
  instantiation_input.13: () = instantiation_input(a, instantiation=u0, port_name=a, id=13)
  y: () = output_port(instantiation_output.12, name=y, id=14)
}

top block top(a: bits[1], y: bits[1]) {
  instantiation u0(block=mid, kind=block)
  a: bits[1] = input_port(name=a, id=21)
  instantiation_output.22: bits[1] = instantiation_output(instantiation=u0, port_name=y, id=22)
  instantiation_input.23: () = instantiation_input(a, instantiation=u0, port_name=a, id=23)
  y: () = output_port(instantiation_output.22, name=y, id=24)
}
"#;
        let mut pkg = parse_pkg(pkg_text);
        inline_all_blocks_in_package(&mut pkg).expect("inline should succeed");
        assert_eq!(
            pkg.members.len(),
            1,
            "expected a single block after inlining"
        );
        for member in pkg.members.iter() {
            let PackageMember::Block { func, metadata } = member else {
                continue;
            };
            assert!(metadata.instantiations.is_empty());
            assert_no_instantiation_nodes(func);
        }
        let PackageMember::Block { func, metadata } = pkg.get_block("top").unwrap() else {
            panic!("expected block");
        };
        assert_output_matches(func, metadata, "y", "not(get_param(name=\"a\"))");
    }

    #[test]
    fn inline_block_with_register() {
        let pkg_text = r#"package test

block leaf(a: bits[1], y: bits[1]) {
  reg r(bits[1])
  a: bits[1] = input_port(name=a, id=1)
  r_q: bits[1] = register_read(register=r, id=2)
  and.3: bits[1] = and(a, r_q, id=3)
  r_d: () = register_write(and.3, register=r, id=4)
  y: () = output_port(and.3, name=y, id=5)
}

top block top(a: bits[1], y: bits[1]) {
  instantiation u0(block=leaf, kind=block)
  a: bits[1] = input_port(name=a, id=11)
  instantiation_output.12: bits[1] = instantiation_output(instantiation=u0, port_name=y, id=12)
  instantiation_input.13: () = instantiation_input(a, instantiation=u0, port_name=a, id=13)
  y: () = output_port(instantiation_output.12, name=y, id=14)
}
"#;
        let mut pkg = parse_pkg(pkg_text);
        inline_all_blocks_in_package(&mut pkg).expect("inline should succeed");
        let PackageMember::Block { func, metadata } = pkg.get_block("top").unwrap() else {
            panic!("expected block");
        };
        assert!(metadata.instantiations.is_empty());
        assert_no_instantiation_nodes(func);
        assert!(
            metadata.registers.iter().any(|r| r.name == "u0__r"),
            "expected inlined register name"
        );
        assert_output_matches(
            func,
            metadata,
            "y",
            "and(get_param(name=\"a\"), register_read(register=\"u0__r\"))",
        );
    }

    #[test]
    fn inline_register_name_collision_is_uniquified() {
        let pkg_text = r#"package test

block leaf(a: bits[1], y: bits[1]) {
  reg r(bits[1])
  a: bits[1] = input_port(name=a, id=1)
  r_q: bits[1] = register_read(register=r, id=2)
  and.3: bits[1] = and(a, r_q, id=3)
  r_d: () = register_write(and.3, register=r, id=4)
  y: () = output_port(and.3, name=y, id=5)
}

top block top(a: bits[1], y: bits[1]) {
  reg u0__r(bits[1])
  instantiation u0(block=leaf, kind=block)
  a: bits[1] = input_port(name=a, id=11)
  instantiation_output.12: bits[1] = instantiation_output(instantiation=u0, port_name=y, id=12)
  instantiation_input.13: () = instantiation_input(a, instantiation=u0, port_name=a, id=13)
  y: () = output_port(instantiation_output.12, name=y, id=14)
}
"#;
        let mut pkg = parse_pkg(pkg_text);
        inline_all_blocks_in_package(&mut pkg).expect("inline should succeed");
        let PackageMember::Block { metadata, .. } = pkg.get_block("top").unwrap() else {
            panic!("expected block");
        };
        assert!(metadata.registers.iter().any(|r| r.name == "u0__r"));
        assert!(
            metadata
                .registers
                .iter()
                .any(|r| r.name.starts_with("u0__r__")),
            "expected uniquified inlined register name"
        );
    }
}
