// SPDX-License-Identifier: Apache-2.0

mod aggregates;
mod slices;
mod wiring;

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use crate::ir::{self, Block, InstantiationKind, MemberType, NodePayload, PackageMember};

/// Blocks and functions occupy separate namespaces in a PIR package.
fn member_key(member: &PackageMember) -> (&str, MemberType) {
    match member {
        PackageMember::Block(block) => (&block.name, MemberType::Block),
        PackageMember::Function(function) => (&function.name, MemberType::Function),
    }
}

/// Flattens one block hierarchy, retaining its transitively called functions.
/// Registers from a child instance acquire an instance-prefixed name. External
/// instances have no block body to inline and are rejected.
pub fn flatten_block_hierarchy(package: &ir::Package, top: &str) -> Result<ir::Package, String> {
    if package.get_block(top).is_none() {
        return Err(format!("block '{top}' not found"));
    }
    let members: HashMap<_, _> = package
        .members
        .iter()
        .map(|member| (member_key(member), member))
        .collect();
    let mut reachable = HashSet::new();
    let mut pending = vec![(top, MemberType::Block)];
    while let Some(key) = pending.pop() {
        if !reachable.insert(key.clone()) {
            continue;
        }
        let member = members
            .get(&key)
            .ok_or_else(|| format!("missing block hierarchy dependency '{}'", key.0))?;
        if let PackageMember::Block(block) = member {
            for instance in &block.instantiations {
                if instance.kind != InstantiationKind::Block {
                    return Err(format!(
                        "cannot inline external instance '{}'",
                        instance.name
                    ));
                }
                pending.push((&instance.block, MemberType::Block));
            }
        }
        pending.extend(
            member
                .graph()
                .nodes
                .iter()
                .filter_map(|node| match &node.payload {
                    NodePayload::Invoke { to_apply, .. } | NodePayload::Map { to_apply, .. } => {
                        Some((to_apply.as_str(), MemberType::Function))
                    }
                    NodePayload::CountedFor { body, .. } => {
                        Some((body.as_str(), MemberType::Function))
                    }
                    _ => None,
                }),
        );
    }
    let mut hierarchy = ir::Package {
        name: package.name.clone(),
        file_table: package.file_table.clone(),
        members: package
            .members
            .iter()
            .filter(|member| reachable.contains(&member_key(member)))
            .cloned()
            .collect(),
        top: Some((top.to_owned(), MemberType::Block)),
    };
    crate::ir_verify::verify_package(&hierarchy).map_err(|e| e.to_string())?;
    inline_all_blocks_in_package(&mut hierarchy)?;
    Ok(hierarchy)
}

/// Wires each instance into its parent and removes all blocks except the top.
/// Whole-value feedback may become acyclic once the enclosing instances bind
/// their inputs. Keep intermediate definitions as wiring graphs until then.
pub fn inline_all_blocks_in_package(pkg: &mut ir::Package) -> Result<(), String> {
    inline_blocks_with_top_bindings(pkg, |_| Ok(()))
}

/// Applies the selected root's external bindings after wiring all instances and
/// before deciding whether any combinational feedback remains.
pub(crate) fn inline_blocks_with_top_bindings(
    pkg: &mut ir::Package,
    bind_top: impl Fn(&mut Block) -> Result<(), String>,
) -> Result<(), String> {
    let blocks: Vec<_> = pkg
        .members
        .iter()
        .filter_map(|m| match m {
            PackageMember::Block(block) => Some(block),
            PackageMember::Function(_) => None,
        })
        .collect();
    if blocks.iter().all(|block| block.instantiations.is_empty()) {
        if let Some(top) = pkg.get_top_block_mut() {
            bind_top(top)?;
        }
        return prune_to_single_top_block(pkg);
    }
    let indices: HashMap<_, _> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name.as_str(), i))
        .collect();
    let mut parents = vec![Vec::new(); blocks.len()];
    let mut pending = Vec::with_capacity(blocks.len());
    let mut ready = Vec::new();
    for (i, block) in blocks.iter().enumerate() {
        crate::ir_verify::verify_graph_operand_indices_in_bounds(block)?;
        for instance in &block.instantiations {
            if instance.kind != InstantiationKind::Block {
                return Err(format!(
                    "cannot inline external instance '{}'",
                    instance.name
                ));
            }
            let child = indices.get(instance.block.as_str()).ok_or_else(|| {
                format!(
                    "block_inline: instantiation '{}' references missing block '{}'",
                    instance.name, instance.block
                )
            })?;
            parents[*child].push(i);
        }
        pending.push(block.instantiations.len());
        if block.instantiations.is_empty() {
            ready.push(i);
        }
    }

    let top = pkg
        .get_top_block()
        .ok_or("block_inline: missing top block")?
        .name
        .clone();
    let top_index = indices[top.as_str()];
    let mut last_id = crate::ir_rebase_ids::package_max_emitted_node_id(pkg);
    let mut bodies: Vec<_> = parents
        .iter()
        .map(|uses| CachedBody {
            expanded: None,
            remaining_uses: uses.len(),
        })
        .collect();
    // The selected top is also a result when an unrelated root instantiates it.
    bodies[top_index].remaining_uses += 1;
    let mut finished = 0;
    while let Some(i) = ready.pop() {
        let source = blocks[i];
        if !source.instantiations.is_empty() {
            let mut block = source.clone();
            let mut wiring = wiring::Wiring::new(source)?;
            for instance in &source.instantiations {
                let child = indices[instance.block.as_str()];
                let body = bodies[child].for_instance(blocks[child]);
                wiring.copy_instance(&mut block, instance, &body, &mut last_id)?;
            }
            wiring.finish(&mut block)?;
            bodies[i].expanded = Some(block);
        }
        finished += 1;
        for &parent in &parents[i] {
            pending[parent] -= 1;
            if pending[parent] == 0 {
                ready.push(parent);
            }
        }
    }
    if finished != blocks.len() {
        return Err("block_inline: recursive block instantiation".to_string());
    }

    bind_top(
        bodies[top_index]
            .expanded
            .get_or_insert_with(|| blocks[top_index].clone()),
    )?;
    let mut calls = crate::block_invoke_inline::FeedbackInvokes::new(pkg);
    // A disconnected hierarchy has its own root and must still be checked.
    for (i, body) in bodies.iter_mut().enumerate() {
        if (parents[i].is_empty() || i == top_index)
            && let Some(block) = &mut body.expanded
            && let Err(error) = block.compact_and_toposort()
        {
            let mut original = Cow::Borrowed(blocks[i]);
            if i == top_index {
                bind_top(original.to_mut())?;
            }
            crate::ir_builder::verify_block_combinational_cycles(&original, pkg)
                .map_err(|e| e.to_string())?;
            if !slices::resolve_slices(block, pkg, &mut calls, last_id)? {
                return Err(format!("block_inline: compaction failed: {error}"));
            }
            block
                .compact_and_toposort()
                .map_err(|e| format!("block_inline: compaction failed: {e}"))?;
        }
    }
    if let Some(block) = bodies[top_index].expanded.take() {
        pkg.replace_block(&top, block)?;
    }
    prune_to_single_top_block(pkg)
}

/// Expressions staged against a single graph snapshot. References into the
/// staged nodes become ordinary node indices when the batch is appended.
struct PendingNodes {
    nodes: Vec<ir::Node>,
    last_id: usize,
}

impl PendingNodes {
    fn new(last_id: usize) -> Self {
        Self {
            nodes: Vec::new(),
            last_id,
        }
    }

    /// Allocates a package-wide fresh ID and copies the original source
    /// position.
    fn push(
        &mut self,
        graph: &ir::NodeGraph,
        source: ir::NodeRef,
        ty: ir::Type,
        payload: NodePayload,
    ) -> Result<ir::NodeRef, String> {
        self.last_id = self
            .last_id
            .checked_add(1)
            .ok_or("block_inline: node IDs overflow usize")?;
        let index = graph.nodes.len() + self.nodes.len();
        self.nodes.push(ir::Node {
            text_id: self.last_id,
            name: None,
            pos: graph.get_node(source).pos.clone(),
            ty,
            payload,
        });
        Ok(ir::NodeRef { index })
    }
}

/// Shares an assembled body across its parents and releases it after its last
/// instance. Leaf bodies are borrowed directly from the input package.
struct CachedBody {
    expanded: Option<Block>,
    remaining_uses: usize,
}

impl CachedBody {
    /// Borrows a shared definition or moves the final cached copy to its
    /// consumer.
    fn for_instance<'a>(&'a mut self, source: &'a Block) -> Cow<'a, Block> {
        self.remaining_uses -= 1;
        if self.remaining_uses == 0
            && let Some(last) = self.expanded.take()
        {
            return Cow::Owned(last);
        }
        Cow::Borrowed(self.expanded.as_ref().unwrap_or(source))
    }
}

/// Reserves a name in a module or function, appending a numeric suffix on
/// collision.
pub(crate) fn uniquify_node_name(base: &str, used: &mut HashMap<String, usize>) -> String {
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
    use crate::ir::NodeRef;
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
    fn expanded_definition_is_shared_until_the_last_instance() {
        let source = Block::new("source");
        let mut cache = CachedBody {
            expanded: Some(Block::new("expanded")),
            remaining_uses: 2,
        };
        assert!(
            matches!(cache.for_instance(&source), Cow::Borrowed(block) if block.name == "expanded")
        );
        assert!(
            matches!(cache.for_instance(&source), Cow::Owned(block) if block.name == "expanded")
        );
        assert!(cache.expanded.is_none());
    }

    #[test]
    fn shifted_feedback_preserves_bitwise_operations() {
        let mut design = parse_pkg(
            r#"package shifted
block leaf(d: bits[4], q: bits[4]) {
  d: bits[4] = input_port(name=d, id=1)
  mask: bits[4] = literal(value=5, id=2)
  masked: bits[4] = xor(d, mask, id=3)
  inverted: bits[4] = not(masked, id=4)
  q: () = output_port(inverted, name=q, id=5)
}
top block wrapper(seed: bits[1], result: bits[4]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=6)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=7)
  lo: bits[3] = bit_slice(q, start=0, width=3, id=8)
  packed_d: bits[4] = concat(lo, seed, id=9)
  drive: () = instantiation_input(packed_d, instantiation=u, port_name=d, id=10)
  result: () = output_port(q, name=result, id=11)
}
"#,
        );
        // Native clients may explicitly store canonical names. These must
        // follow both the copied child's fresh IDs and the projected operator.
        for member in &mut design.members {
            if let PackageMember::Block(block) = member {
                for node in &mut block.nodes {
                    if matches!(node.text_id, 3 | 8 | 10) {
                        node.name =
                            Some(format!("{}.{}", node.payload.get_operator(), node.text_id));
                    }
                }
            }
        }
        crate::ir_verify::verify_package(&design).unwrap();
        let flat = flatten_block_hierarchy(&design, "wrapper").unwrap();
        crate::ir_verify::verify_package(&flat).unwrap();
        let top = flat.get_top_block().unwrap();
        for name in ["packed_d", "u__inverted"] {
            assert!(
                top.nodes
                    .iter()
                    .any(|node| node.name.as_deref() == Some(name))
            );
        }
        let f = crate::block2fn::combinational_block_to_fn(flat.get_top_block().unwrap()).unwrap();
        // At each lane the previous bit is inverted exactly when its mask bit
        // is zero.
        for (seed, expected) in [(0, 0b0110), (1, 0b1001)] {
            let arg = crate::IrValue::make_ubits(1, seed).unwrap();
            let actual = crate::ir_eval::eval_fn_in_package_value(&flat, &f, &[arg]);
            assert_eq!(actual, crate::IrValue::make_ubits(4, expected).unwrap());
        }
    }

    #[test]
    fn passthrough_outputs_can_drive_other_inputs_of_the_same_instance() {
        let source = r#"package chain
block leaf(x: bits[1], y: bits[1], a: bits[1], b: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  y: bits[1] = input_port(name=y, id=2)
  a: () = output_port(x, name=a, id=3)
  b: () = output_port(y, name=b, id=4)
}
top block wrapper(seed: bits[1], result: bits[1]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=5)
  a: bits[1] = instantiation_output(instantiation=u, port_name=a, id=6)
  b: bits[1] = instantiation_output(instantiation=u, port_name=b, id=7)
  first: () = instantiation_input(seed, instantiation=u, port_name=x, id=8)
  second: () = instantiation_input(a, instantiation=u, port_name=y, id=9)
  result: () = output_port(b, name=result, id=10)
}
"#;
        let flat = flatten_block_hierarchy(&parse_pkg(source), "wrapper").unwrap();
        crate::ir_verify::verify_package(&flat).unwrap();
        let f = crate::block2fn::combinational_block_to_fn(flat.get_top_block().unwrap()).unwrap();
        for seed in 0..2 {
            let arg = crate::IrValue::make_ubits(1, seed).unwrap();
            assert_eq!(
                crate::ir_eval::eval_fn_in_package_value(&flat, &f, std::slice::from_ref(&arg)),
                arg
            );
        }
        // Closing the chain or wiring the first output straight back to itself
        // is a real loop.
        for feedback in ["a", "b"] {
            let cyclic = parse_pkg(&source.replace(
                "instantiation_input(seed,",
                &format!("instantiation_input({feedback},"),
            ));
            assert!(
                flatten_block_hierarchy(&cyclic, "wrapper")
                    .unwrap_err()
                    .contains("cycle")
            );
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
    fn sibling_reset_passthrough_is_independent_of_instance_order() {
        let passthrough = r#"block forward(a: bits[1], b: bits[1]) {
  a: bits[1] = input_port(name=a, id=21)
  b: () = output_port(a, name=b, id=22)
}

"#;
        let storage = "  instantiation child(block=leaf, kind=block)";
        let route = "  instantiation wire_rst(block=forward, kind=block)";
        for instances in [format!("{storage}\n{route}"), format!("{route}\n{storage}")] {
            let text = RESET_HIERARCHY
                .replace("block leaf(", &format!("{passthrough}block leaf("))
                .replace(storage, &instances)
                .replace("  reset_write: () = instantiation_input(reset_n,", r#"  source: () = instantiation_input(reset_n, instantiation=wire_rst, port_name=a, id=23)
  routed: bits[1] = instantiation_output(instantiation=wire_rst, port_name=b, id=24)
  reset_write: () = instantiation_input(routed,"#);
            let mut package = parse_pkg(&text);
            inline_all_blocks_in_package(&mut package).unwrap();
            crate::ir_verify::verify_package(&package).unwrap();
            let top = package.get_top_block().unwrap();
            assert_eq!(
                top.reset.as_ref().unwrap().port,
                top.get_input_port("reset_n").unwrap()
            );
        }
    }

    #[test]
    fn invalid_child_operand_returns_error_before_changing_the_package() {
        let mut package = parse_pkg(RESET_HIERARCHY);
        let mut child = package.get_block("leaf").unwrap().clone();
        let output = child.get_output_port("result").unwrap();
        let NodePayload::OutputPort { arg, .. } = &mut child.get_node_mut(output).payload else {
            unreachable!()
        };
        *arg = NodeRef { index: usize::MAX };
        package.replace_block("leaf", child).unwrap();
        let original = format!("{package:?}");
        let error = inline_all_blocks_in_package(&mut package).unwrap_err();
        assert!(error.contains("out of bounds"), "{error}");
        assert_eq!(format!("{package:?}"), original);
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
