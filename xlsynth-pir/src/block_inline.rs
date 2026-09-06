// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};

use crate::ir::{self, Block, MemberType, NodePayload, NodeRef, PackageMember, Register};
use crate::ir_utils::{get_topological, remap_payload_with};

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
                PackageMember::Block(block) => Some(block.name.clone()),
                _ => None,
            })
            .collect();
        for name in block_names {
            let mut block = match pkg.get_block(&name) {
                Some(block) => block.clone(),
                _ => continue,
            };
            if block.instantiations.is_empty() {
                continue;
            }
            has_instantiations = true;
            let inlined = inline_block_instantiations(pkg, &mut block)?;
            if inlined {
                pkg.replace_block(&name, block)?;
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

fn inline_block_instantiations(pkg: &ir::Package, block: &mut Block) -> Result<bool, String> {
    let mut inlined_any = false;
    loop {
        let instantiations = block.instantiations.clone();
        let mut did_inline = false;
        for inst in instantiations {
            let callee = match pkg.get_block(&inst.block) {
                Some(block) => block.clone(),
                _ => {
                    return Err(format!(
                        "block_inline: instantiation '{}' references missing block '{}'",
                        inst.name, inst.block
                    ));
                }
            };
            if !callee.instantiations.is_empty() {
                continue;
            }
            let package_max_id = pkg
                .members
                .iter()
                .flat_map(|member| &member.graph().nodes)
                .map(|node| node.text_id)
                .max()
                .unwrap_or(0);
            inline_single_instantiation(block, &inst.name, &callee, package_max_id)?;
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
    caller: &mut Block,
    inst_name: &str,
    callee: &Block,
    package_max_id: usize,
) -> Result<(), String> {
    let input_map = collect_instantiation_inputs(caller, inst_name)?;
    let output_map = collect_callee_outputs(callee)?;
    if !callee.registers.is_empty() && caller.clock_port_name().is_none() {
        return Err(format!(
            "block_inline: instantiation '{inst_name}' has registers but the parent has no clock port"
        ));
    }
    let reset = merge_inlined_reset(caller, callee, &input_map, inst_name)?;

    let mut register_name_map: HashMap<String, String> = HashMap::new();
    let mut used_register_names: HashMap<String, usize> = caller
        .registers
        .iter()
        .map(|r| (r.name.clone(), 1))
        .collect();
    for reg in callee.registers.iter() {
        let base_name = format!("{}__{}", inst_name, reg.name);
        let unique_name = uniquify_register_name(&base_name, &mut used_register_names);
        register_name_map.insert(reg.name.clone(), unique_name.clone());
        caller.registers.push(Register {
            name: unique_name,
            ty: reg.ty.clone(),
            reset_value: reg.reset_value.clone(),
        });
    }

    let mut max_text_id = caller
        .nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .max(package_max_id);
    let mut used_node_names: HashMap<String, usize> = caller
        .nodes
        .iter()
        .filter_map(|n| n.name.clone())
        .map(|name| (name, 1))
        .collect();
    let mut mapping: HashMap<usize, NodeRef> = HashMap::new();
    let topo = get_topological(callee);
    for nr in topo {
        if nr.index == 0 {
            continue;
        }
        let node = callee.get_node(nr);
        match &node.payload {
            NodePayload::InputPort { name, .. } => {
                let arg = input_map.get(name).ok_or_else(|| {
                    format!(
                        "block_inline: instantiation '{}' missing input '{}'",
                        inst_name, name
                    )
                })?;
                mapping.insert(nr.index, *arg);
            }
            NodePayload::OutputPort { .. } => {
                // Output sinks are unit-valued and may have ordinary unit
                // users. The external output wiring still uses
                // their data operands.
                max_text_id = max_text_id
                    .checked_add(1)
                    .ok_or("block_inline: node IDs overflow usize")?;
                let new_ref = NodeRef {
                    index: caller.nodes.len(),
                };
                caller.nodes.push(ir::Node {
                    text_id: max_text_id,
                    name: None,
                    ty: ir::Type::nil(),
                    payload: NodePayload::Tuple(Vec::new()),
                    pos: node.pos.clone(),
                });
                mapping.insert(nr.index, new_ref);
            }
            NodePayload::InstantiationInput { .. } | NodePayload::InstantiationOutput { .. } => {
                return Err(format!(
                    "block_inline: callee '{}' still has instantiation nodes",
                    callee.name
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
                max_text_id = max_text_id
                    .checked_add(1)
                    .ok_or("block_inline: node IDs overflow usize")?;
                let new_node = ir::Node {
                    text_id: max_text_id,
                    name: node.name.as_ref().map(|n| {
                        let base = format!("{}__{}", inst_name, n);
                        uniquify_node_name(&base, &mut used_node_names)
                    }),
                    ty: node.ty.clone(),
                    payload: new_payload,
                    pos: node.pos.clone(),
                };
                let new_ref = NodeRef {
                    index: caller.nodes.len(),
                };
                caller.nodes.push(new_node);
                mapping.insert(nr.index, new_ref);
            }
        }
    }

    let mut output_replacements: Vec<(NodeRef, NodeRef)> = Vec::new();
    let mut input_sinks: Vec<NodeRef> = Vec::new();
    for (idx, node) in caller.nodes.iter().enumerate() {
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
                input_sinks.push(NodeRef { index: idx });
            }
            _ => {}
        }
    }

    for (target, replacement) in output_replacements {
        crate::ir_utils::replace_graph_node_with_ref(caller, target, replacement)
            .map_err(|e| format!("block_inline: replace output failed: {e}"))?;
    }
    for target in input_sinks {
        // The connection is gone, but ordinary users still observe its unit
        // value.
        caller.nodes[target.index].payload = NodePayload::Tuple(Vec::new());
        caller.nodes[target.index].name = None;
    }

    caller.instantiations.retain(|inst| inst.name != inst_name);
    caller.reset = reset;

    caller
        .compact_and_toposort()
        .map_err(|e| format!("block_inline: compaction failed: {e}"))?;
    Ok(())
}

/// Carries a child's one reset domain into the parent without changing wiring.
fn merge_inlined_reset(
    caller: &Block,
    callee: &Block,
    inputs: &BTreeMap<String, NodeRef>,
    instance: &str,
) -> Result<Option<ir::BlockReset>, String> {
    let Some(child_reset) = &callee.reset else {
        return Ok(caller.reset.clone());
    };
    let child_name = callee.port_name(child_reset.port);
    let port = inputs.get(child_name).copied().ok_or_else(|| {
        format!("block_inline: instantiation '{instance}' is missing reset input '{child_name}'")
    })?;
    if !caller.input_ports().any(|input| input == port) {
        return Err(format!(
            "block_inline: instantiation '{instance}' reset must connect directly to a parent input port"
        ));
    }
    let mapped = ir::BlockReset {
        port,
        asynchronous: child_reset.asynchronous,
        active_low: child_reset.active_low,
    };
    if let Some(parent_reset) = &caller.reset {
        if parent_reset != &mapped {
            return Err(format!(
                "block_inline: instantiation '{instance}' has an incompatible reset port, polarity, or timing"
            ));
        }
    }
    Ok(Some(mapped))
}

fn collect_instantiation_inputs(
    caller: &Block,
    inst_name: &str,
) -> Result<BTreeMap<String, NodeRef>, String> {
    let mut inputs = BTreeMap::new();
    for node in caller.nodes.iter() {
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

fn collect_callee_outputs(callee: &Block) -> Result<BTreeMap<String, NodeRef>, String> {
    let mut outputs = BTreeMap::new();
    for port in callee.output_ports() {
        let name = callee.port_name(port).to_string();
        if outputs
            .insert(name.clone(), callee.output_value(port))
            .is_some()
        {
            return Err(format!("block_inline: duplicate output '{name}'"));
        }
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
                PackageMember::Block(block) => Some(block.name.clone()),
                _ => None,
            });
            first.ok_or_else(|| "block_inline: no block members to keep".to_string())?
        }
    };
    pkg.members.retain(|m| match m {
        PackageMember::Block(block) => block.name == top_name,
        PackageMember::Function(_) => true,
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

    fn assert_no_instantiation_nodes(f: &Block) {
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

    #[test]
    fn inlining_preserves_unit_users_of_output_sinks() {
        let mut package = parse_pkg(
            r#"package output_users

block leaf(x: bits[8], out: bits[8], unit: (())) {
  x: bits[8] = input_port(name=x, id=1)
  out: () = output_port(x, name=out, id=2)
  value: (()) = tuple(out, id=3)
  unit: () = output_port(value, name=unit, id=4)
}

top block caller(x: bits[8], out: bits[8], unit: (())) {
  instantiation child(block=leaf, kind=block)
  x: bits[8] = input_port(name=x, id=11)
  data: bits[8] = instantiation_output(instantiation=child, port_name=out, id=12)
  aggregate: (()) = instantiation_output(instantiation=child, port_name=unit, id=13)
  write: () = instantiation_input(x, instantiation=child, port_name=x, id=14)
  out: () = output_port(data, name=out, id=15)
  unit: () = output_port(aggregate, name=unit, id=16)
}
"#,
        );
        inline_all_blocks_in_package(&mut package).unwrap();
        crate::ir_verify::verify_package(&package).unwrap();
        let function =
            crate::block2fn::combinational_block_to_fn(package.get_top_block().unwrap()).unwrap();
        let argument = crate::IrValue::make_ubits(8, 42).unwrap();
        let crate::ir_eval::FnEvalResult::Success(result) =
            crate::ir_eval::eval_fn(&function, &[argument.clone()])
        else {
            panic!("inlined block evaluation failed")
        };
        assert_eq!(
            result.value,
            crate::IrValue::make_tuple(&[
                argument,
                crate::IrValue::make_tuple(&[crate::IrValue::make_tuple(&[])])
            ])
        );
    }

    #[test]
    fn inlining_preserves_unit_users_of_instantiation_input_sinks() {
        let mut package = parse_pkg(
            r#"package unit_connections

block child(x: bits[1], y: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}

top block parent(x: bits[1], z: ((), bits[1])) {
  instantiation u(block=child, kind=block)
  x: bits[1] = input_port(name=x, id=3)
  connect: () = instantiation_input(x, instantiation=u, port_name=x, id=4)
  result: bits[1] = instantiation_output(instantiation=u, port_name=y, id=5)
  pair: ((), bits[1]) = tuple(connect, result, id=6)
  z: () = output_port(pair, name=z, id=7)
}
"#,
        );
        inline_all_blocks_in_package(&mut package).unwrap();
        crate::ir_verify::verify_package(&package).unwrap();
        let function =
            crate::block2fn::combinational_block_to_fn(package.get_top_block().unwrap()).unwrap();
        let argument = crate::IrValue::make_ubits(1, 1).unwrap();
        let crate::ir_eval::FnEvalResult::Success(result) =
            crate::ir_eval::eval_fn(&function, &[argument.clone()])
        else {
            panic!("inlined block evaluation failed")
        };
        assert_eq!(
            result.value,
            crate::IrValue::make_tuple(&[crate::IrValue::make_tuple(&[]), argument])
        );
    }

    #[test]
    fn inlining_retains_function_callees_and_avoids_their_node_ids() {
        let mut package = parse_pkg(
            r#"package calls

fn invert(x: bits[1] id=15) -> bits[1] {
  ret result: bits[1] = not(x, id=16)
}

block leaf(x: bits[1], y: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  value: bits[1] = invoke(x, to_apply=invert, id=2)
  y: () = output_port(value, name=y, id=3)
}

top block parent(x: bits[1], y: bits[1]) {
  instantiation child(block=leaf, kind=block)
  x: bits[1] = input_port(name=x, id=11)
  value: bits[1] = instantiation_output(instantiation=child, port_name=y, id=12)
  write: () = instantiation_input(x, instantiation=child, port_name=x, id=13)
  y: () = output_port(value, name=y, id=14)
}
"#,
        );
        inline_all_blocks_in_package(&mut package).unwrap();
        crate::ir_verify::verify_package(&package).unwrap();
        assert!(package.get_fn("invert").is_some());
        assert!(package.get_block("leaf").is_none());
        assert_eq!(package.members.len(), 2);
        let function =
            crate::block2fn::combinational_block_to_fn(package.get_top_block().unwrap()).unwrap();
        let crate::ir_eval::FnEvalResult::Success(result) = crate::ir_eval::eval_fn_in_package(
            &package,
            &function,
            &[crate::IrValue::make_ubits(1, 0).unwrap()],
        ) else {
            panic!("inlined call evaluation failed")
        };
        assert_eq!(result.value, crate::IrValue::make_ubits(1, 1).unwrap());
    }

    const RESET_HIERARCHY: &str = r#"package reset_hierarchy

block leaf(clk: clock, rst: bits[1], data: bits[8], result: bits[8]) {
  #![reset(port="rst", asynchronous=true, active_low=true)]
  reg state(bits[8], reset_value=7)
  rst: bits[1] = input_port(name=rst, id=1)
  data: bits[8] = input_port(name=data, id=2)
  current: bits[8] = register_read(register=state, id=3)
  write: () = register_write(data, register=state, reset=rst, id=4)
  result: () = output_port(current, name=result, id=5)
}

top block caller(clk: clock, reset_n: bits[1], other: bits[1], data: bits[8], result: bits[8]) {
  instantiation child(block=leaf, kind=block)
  reset_n: bits[1] = input_port(name=reset_n, id=11)
  other: bits[1] = input_port(name=other, id=12)
  data: bits[8] = input_port(name=data, id=13)
  reset_write: () = instantiation_input(reset_n, instantiation=child, port_name=rst, id=14)
  data_write: () = instantiation_input(data, instantiation=child, port_name=data, id=15)
  value: bits[8] = instantiation_output(instantiation=child, port_name=result, id=16)
  result: () = output_port(value, name=result, id=17)
}
"#;

    #[test]
    fn inlining_maps_child_reset_to_parent_input_and_preserves_behavior() {
        let mut package = parse_pkg(RESET_HIERARCHY);
        assert!(package.get_top_block().unwrap().reset.is_none());
        inline_all_blocks_in_package(&mut package).unwrap();
        crate::ir_verify::verify_package(&package).unwrap();
        let block = package.get_top_block().unwrap();
        assert_eq!(
            block.reset,
            Some(ir::BlockReset {
                port: block.get_input_port("reset_n").unwrap(),
                asynchronous: true,
                active_low: true,
            })
        );
        assert_eq!(block.clock_port_name(), Some("clk"));
        assert_eq!(
            block.registers[0].reset_value,
            Some(crate::IrValue::make_ubits(8, 7).unwrap())
        );
    }

    #[test]
    fn inlining_rejects_incompatible_reset_domains_and_derived_reset_ports() {
        for (name, asynchronous, active_low) in [
            ("other", true, true),
            ("reset_n", false, true),
            ("reset_n", true, false),
        ] {
            let mut package = parse_pkg(RESET_HIERARCHY);
            let PackageMember::Block(block) = package.members.last_mut().unwrap() else {
                unreachable!()
            };
            block.reset = Some(ir::BlockReset {
                port: block.get_input_port(name).unwrap(),
                asynchronous,
                active_low,
            });
            crate::ir_verify::verify_package(&package).unwrap();
            assert_eq!(
                inline_all_blocks_in_package(&mut package).unwrap_err(),
                "block_inline: instantiation 'child' has an incompatible reset port, polarity, or timing"
            );
        }
        let derived = RESET_HIERARCHY.replace(
            "  reset_write: () = instantiation_input(reset_n,",
            "  inverted: bits[1] = not(reset_n, id=18)\n  reset_write: () = instantiation_input(inverted,",
        );
        let mut package = parse_pkg(&derived);
        assert_eq!(
            inline_all_blocks_in_package(&mut package).unwrap_err(),
            "block_inline: instantiation 'child' reset must connect directly to a parent input port"
        );
    }

    fn output_node_ref(block: &Block, name: &str) -> NodeRef {
        block.output_value(block.get_output_port(name).expect("output port"))
    }

    fn assert_output_matches(f: &Block, name: &str, query_text: &str) {
        let node_ref = output_node_ref(f, name);
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
        let block = pkg.get_block("top").unwrap();
        assert!(block.instantiations.is_empty());
        assert_no_instantiation_nodes(block);
        assert_output_matches(block, "y", "input_port(name=\"a\")");
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
        let block = pkg.get_block("top").unwrap();
        assert!(block.instantiations.is_empty());
        assert_no_instantiation_nodes(block);
        assert_output_matches(block, "y", "not(input_port(name=\"a\"))");
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
        let block = pkg.get_block("top").unwrap();
        assert!(block.instantiations.is_empty());
        assert_no_instantiation_nodes(block);
        assert_output_matches(block, "y0", "not(input_port(name=\"a\"))");
        assert_output_matches(block, "y1", "not(input_port(name=\"a\"))");
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
            let PackageMember::Block(block) = member else {
                continue;
            };
            assert!(block.instantiations.is_empty());
            assert_no_instantiation_nodes(block);
        }
        let block = pkg.get_block("top").unwrap();
        assert_output_matches(block, "y", "not(input_port(name=\"a\"))");
    }

    #[test]
    fn inline_block_with_register() {
        let pkg_text = r#"package test

block leaf(clk: clock, a: bits[1], y: bits[1]) {
  reg r(bits[1])
  a: bits[1] = input_port(name=a, id=1)
  r_q: bits[1] = register_read(register=r, id=2)
  and.3: bits[1] = and(a, r_q, id=3)
  r_d: () = register_write(and.3, register=r, id=4)
  y: () = output_port(and.3, name=y, id=5)
}

top block top(clk: clock, a: bits[1], y: bits[1]) {
  instantiation u0(block=leaf, kind=block)
  a: bits[1] = input_port(name=a, id=11)
  instantiation_output.12: bits[1] = instantiation_output(instantiation=u0, port_name=y, id=12)
  instantiation_input.13: () = instantiation_input(a, instantiation=u0, port_name=a, id=13)
  y: () = output_port(instantiation_output.12, name=y, id=14)
}
"#;
        let mut pkg = parse_pkg(pkg_text);
        inline_all_blocks_in_package(&mut pkg).expect("inline should succeed");
        let block = pkg.get_block("top").unwrap();
        assert!(block.instantiations.is_empty());
        assert_no_instantiation_nodes(block);
        assert!(
            block.registers.iter().any(|r| r.name == "u0__r"),
            "expected inlined register name"
        );
        assert_output_matches(
            block,
            "y",
            "and(input_port(name=\"a\"), register_read(register=\"u0__r\"))",
        );
    }

    #[test]
    fn inline_register_name_collision_is_uniquified() {
        let pkg_text = r#"package test

block leaf(clk: clock, a: bits[1], y: bits[1]) {
  reg r(bits[1])
  a: bits[1] = input_port(name=a, id=1)
  r_q: bits[1] = register_read(register=r, id=2)
  and.3: bits[1] = and(a, r_q, id=3)
  r_d: () = register_write(and.3, register=r, id=4)
  y: () = output_port(and.3, name=y, id=5)
}

top block top(clk: clock, a: bits[1], y: bits[1]) {
  reg u0__r(bits[1])
  instantiation u0(block=leaf, kind=block)
  a: bits[1] = input_port(name=a, id=11)
  instantiation_output.12: bits[1] = instantiation_output(instantiation=u0, port_name=y, id=12)
  instantiation_input.13: () = instantiation_input(a, instantiation=u0, port_name=a, id=13)
  y: () = output_port(instantiation_output.12, name=y, id=14)
  existing_q: bits[1] = register_read(register=u0__r, id=15)
  existing_d: () = register_write(a, register=u0__r, id=16)
}
"#;
        let mut pkg = parse_pkg(pkg_text);
        inline_all_blocks_in_package(&mut pkg).expect("inline should succeed");
        let block = pkg.get_block("top").unwrap();
        assert!(block.registers.iter().any(|r| r.name == "u0__r"));
        assert!(
            block
                .registers
                .iter()
                .any(|r| r.name.starts_with("u0__r__")),
            "expected uniquified inlined register name"
        );
    }
}
