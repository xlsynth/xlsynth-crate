// SPDX-License-Identifier: Apache-2.0

//! Exact combinational-dependency analysis across block instantiations.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use xlsynth_pir::ir::{BlockMetadata, Fn, InstantiationKind, NodePayload, NodeRef};
use xlsynth_pir::ir_utils::operands;

use crate::BlockCodegenError;

type OutputDependencies = BTreeMap<String, BTreeSet<String>>;

/// Rejects combinational cycles without treating registered child paths as
/// feedback.
pub(crate) fn verify_hierarchy(
    ordered_blocks: &[(&Fn, &BlockMetadata)],
) -> Result<(), BlockCodegenError> {
    let mut dependencies = BTreeMap::<&str, OutputDependencies>::new();

    for &(func, metadata) in ordered_blocks {
        let mut successors = vec![BTreeSet::new(); func.nodes.len()];
        let mut incoming_counts = vec![0usize; func.nodes.len()];
        let mut instance_inputs = BTreeMap::<(&str, &str), usize>::new();
        let mut instance_outputs = Vec::<(&str, &str, usize)>::new();

        for (index, node) in func.nodes.iter().enumerate() {
            for operand in operands(&node.payload) {
                add_edge(&mut successors, &mut incoming_counts, operand.index, index);
            }

            match &node.payload {
                NodePayload::InstantiationInput {
                    instantiation,
                    port_name,
                    ..
                } => {
                    instance_inputs.insert((instantiation.as_str(), port_name.as_str()), index);
                }
                NodePayload::InstantiationOutput {
                    instantiation,
                    port_name,
                } => {
                    instance_outputs.push((instantiation.as_str(), port_name.as_str(), index));
                }
                _ => {
                    // Explicit operands completely describe ordinary node
                    // dependencies.
                }
            }
        }

        for (instance_name, output_name, output_node) in instance_outputs {
            let instance = metadata
                .instantiations
                .iter()
                .find(|instance| instance.name == instance_name)
                .ok_or_else(|| {
                    BlockCodegenError::InvalidBlock(format!(
                        "block `{}` references undeclared instance `{instance_name}`",
                        func.name
                    ))
                })?;
            if instance.kind == InstantiationKind::Extern {
                // XLS treats foreign instances as opaque during elaborated
                // combinational-cycle verification.
                continue;
            }

            let child_dependencies =
                dependencies.get(instance.block.as_str()).ok_or_else(|| {
                    BlockCodegenError::InvalidBlock(format!(
                        "block `{}` instantiates child `{}` before its dependencies are available",
                        func.name, instance.block
                    ))
                })?;
            let output_dependencies = child_dependencies.get(output_name).ok_or_else(|| {
                BlockCodegenError::InvalidBlock(format!(
                    "instance `{instance_name}` of block `{}` has no output `{output_name}`",
                    instance.block
                ))
            })?;

            for input_name in output_dependencies {
                let input_node = instance_inputs
                    .get(&(instance_name, input_name.as_str()))
                    .copied()
                    .ok_or_else(|| {
                        BlockCodegenError::InvalidBlock(format!(
                            "instance `{instance_name}` of block `{}` has no connected input \
                             `{input_name}`",
                            instance.block
                        ))
                    })?;
                add_edge(
                    &mut successors,
                    &mut incoming_counts,
                    input_node,
                    output_node,
                );
            }
        }

        let parameter_indices = func
            .params
            .iter()
            .enumerate()
            .map(|(index, parameter)| (parameter.id.get_wrapped_id(), index))
            .collect::<BTreeMap<_, _>>();
        let mut node_dependencies = vec![BTreeSet::<usize>::new(); func.nodes.len()];
        let mut ready = incoming_counts
            .iter()
            .enumerate()
            .filter_map(|(index, &count)| (count == 0).then_some(index))
            .collect::<VecDeque<_>>();
        let mut visited = 0usize;

        while let Some(index) = ready.pop_front() {
            visited += 1;
            if let NodePayload::GetParam(parameter_id) = &func.nodes[index].payload {
                if let Some(&parameter_index) =
                    parameter_indices.get(&parameter_id.get_wrapped_id())
                {
                    node_dependencies[index].insert(parameter_index);
                }
            }

            let inherited = std::mem::take(&mut node_dependencies[index]);
            for &successor in &successors[index] {
                node_dependencies[successor].extend(inherited.iter().copied());
                incoming_counts[successor] -= 1;
                if incoming_counts[successor] == 0 {
                    ready.push_back(successor);
                }
            }
            node_dependencies[index] = inherited;
        }

        if visited != func.nodes.len() {
            let index = incoming_counts
                .iter()
                .enumerate()
                .find(|&(index, &count)| {
                    count != 0
                        && matches!(
                            func.nodes[index].payload,
                            NodePayload::InstantiationInput { .. }
                                | NodePayload::InstantiationOutput { .. }
                        )
                })
                .or_else(|| {
                    incoming_counts
                        .iter()
                        .enumerate()
                        .find(|&(_, &count)| count != 0)
                })
                .map(|(index, _)| index)
                .expect("an unvisited node has a nonzero incoming count");
            let node = &func.nodes[index];
            let node_name = node
                .name
                .clone()
                .unwrap_or_else(|| format!("{}.{}", node.payload.get_operator(), node.text_id));
            return Err(BlockCodegenError::InvalidBlock(format!(
                "block `{}` contains a combinational cycle through its block hierarchy \
                 at node `{node_name}`",
                func.name
            )));
        }

        let mut outputs = OutputDependencies::new();
        for (name, node) in metadata
            .output_names
            .iter()
            .zip(output_nodes(func, metadata)?)
        {
            outputs.insert(
                name.clone(),
                node_dependencies[node.index]
                    .iter()
                    .map(|&index| func.params[index].name.clone())
                    .collect(),
            );
        }
        dependencies.insert(func.name.as_str(), outputs);
    }

    Ok(())
}

/// Adds one unique zero-delay edge and updates its destination's indegree.
fn add_edge(
    successors: &mut [BTreeSet<usize>],
    incoming_counts: &mut [usize],
    source: usize,
    destination: usize,
) {
    if successors[source].insert(destination) {
        incoming_counts[destination] += 1;
    }
}

/// Resolves source output nodes in their declared block-port order.
fn output_nodes(func: &Fn, metadata: &BlockMetadata) -> Result<Vec<NodeRef>, BlockCodegenError> {
    if metadata.output_names.is_empty() {
        return Ok(Vec::new());
    }

    let returned = func.ret_node_ref.ok_or_else(|| {
        BlockCodegenError::InvalidBlock(format!(
            "block `{}` declares outputs but has no return node",
            func.name
        ))
    })?;
    if metadata.output_names.len() == 1 {
        return Ok(vec![returned]);
    }

    match &func.get_node(returned).payload {
        NodePayload::Tuple(nodes) if nodes.len() == metadata.output_names.len() => {
            Ok(nodes.clone())
        }
        _ => Err(BlockCodegenError::InvalidBlock(format!(
            "block `{}` has {} outputs but its return node is not a matching tuple",
            func.name,
            metadata.output_names.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use xlsynth_pir::ir::{BlockMetadata, Fn, Package, PackageMember};
    use xlsynth_pir::ir_parser::Parser;

    use super::verify_hierarchy;
    use crate::BlockCodegenError;

    /// Parses public synthetic IR and preserves dependency-before-parent order.
    fn parse_blocks(source: &str) -> Package {
        Parser::new(source)
            .parse_and_verify_package()
            .unwrap_or_else(|error| panic!("invalid public test package:\n{source}\n{error}"))
    }

    /// Collects package blocks in their already-verified declaration order.
    fn ordered_blocks(package: &Package) -> Vec<(&Fn, &BlockMetadata)> {
        package
            .members
            .iter()
            .filter_map(|member| match member {
                PackageMember::Block { func, metadata } => Some((func, metadata)),
                PackageMember::Function(_) => None,
            })
            .collect()
    }

    #[test]
    fn direct_child_feedback_is_rejected() {
        let package = parse_blocks(
            r#"package public_direct_cycle

block passthrough(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

top block feedback(result: bits[8]) {
  instantiation component(block=passthrough, kind=block)
  fed: bits[8] = instantiation_output(instantiation=component, port_name=result, id=3)
  connected: () = instantiation_input(fed, instantiation=component, port_name=value, id=4)
  result: () = output_port(fed, name=result, id=5)
}
"#,
        );

        assert!(matches!(
            verify_hierarchy(&ordered_blocks(&package)),
            Err(BlockCodegenError::InvalidBlock(message))
                if message.contains("block `feedback` contains a combinational cycle")
        ));
    }

    #[test]
    fn nested_child_feedback_is_rejected_transitively() {
        let package = parse_blocks(
            r#"package public_nested_cycle

block inner(value: bits[8], result: bits[8]) {
  value: bits[8] = input_port(name=value, id=1)
  result: () = output_port(value, name=result, id=2)
}

block middle(value: bits[8], result: bits[8]) {
  instantiation nested(block=inner, kind=block)
  value: bits[8] = input_port(name=value, id=3)
  connected: () = instantiation_input(value, instantiation=nested, port_name=value, id=4)
  received: bits[8] = instantiation_output(instantiation=nested, port_name=result, id=5)
  result: () = output_port(received, name=result, id=6)
}

top block feedback(result: bits[8]) {
  instantiation component(block=middle, kind=block)
  fed: bits[8] = instantiation_output(instantiation=component, port_name=result, id=7)
  connected: () = instantiation_input(fed, instantiation=component, port_name=value, id=8)
  result: () = output_port(fed, name=result, id=9)
}
"#,
        );

        assert!(matches!(
            verify_hierarchy(&ordered_blocks(&package)),
            Err(BlockCodegenError::InvalidBlock(message))
                if message.contains("block `feedback` contains a combinational cycle")
        ));
    }

    #[test]
    fn registered_child_legally_breaks_the_feedback_path() {
        let package = parse_blocks(
            r#"package public_registered_feedback

block delayed(clk: clock, value: bits[8], result: bits[8]) {
  reg state(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  stored: bits[8] = register_read(register=state, id=2)
  update: () = register_write(value, register=state, id=3)
  result: () = output_port(stored, name=result, id=4)
}

top block feedback(clk: clock, result: bits[8]) {
  instantiation component(block=delayed, kind=block)
  fed: bits[8] = instantiation_output(instantiation=component, port_name=result, id=5)
  connected: () = instantiation_input(fed, instantiation=component, port_name=value, id=6)
  result: () = output_port(fed, name=result, id=7)
}
"#,
        );

        assert_eq!(verify_hierarchy(&ordered_blocks(&package)), Ok(()));
    }

    #[test]
    fn unrelated_child_input_and_output_do_not_create_false_feedback() {
        let package = parse_blocks(
            r#"package public_independent_ports

block independent(left: bits[8], right: bits[8], first: bits[8], second: bits[8]) {
  left: bits[8] = input_port(name=left, id=1)
  right: bits[8] = input_port(name=right, id=2)
  first: () = output_port(left, name=first, id=3)
  second: () = output_port(right, name=second, id=4)
}

top block connected(value: bits[8], result: bits[8]) {
  instantiation component(block=independent, kind=block)
  value: bits[8] = input_port(name=value, id=5)
  first_value: bits[8] = instantiation_output(instantiation=component, port_name=first, id=6)
  second_value: bits[8] = instantiation_output(instantiation=component, port_name=second, id=7)
  connected_left: () = instantiation_input(second_value, instantiation=component, port_name=left, id=8)
  connected_right: () = instantiation_input(value, instantiation=component, port_name=right, id=9)
  result: () = output_port(first_value, name=result, id=10)
}
"#,
        );

        assert_eq!(verify_hierarchy(&ordered_blocks(&package)), Ok(()));
    }

    #[test]
    fn registered_output_breaks_feedback_even_with_a_combinational_sibling() {
        let package = parse_blocks(
            r#"package public_mixed_output_feedback

block mixed(clk: clock, value: bits[8], direct: bits[8], delayed: bits[8]) {
  reg state(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  stored: bits[8] = register_read(register=state, id=2)
  update: () = register_write(value, register=state, id=3)
  direct: () = output_port(value, name=direct, id=4)
  delayed: () = output_port(stored, name=delayed, id=5)
}

top block feedback(clk: clock, direct: bits[8], delayed: bits[8]) {
  instantiation component(block=mixed, kind=block)
  direct_value: bits[8] = instantiation_output(instantiation=component, port_name=direct, id=6)
  delayed_value: bits[8] = instantiation_output(instantiation=component, port_name=delayed, id=7)
  connected: () = instantiation_input(delayed_value, instantiation=component, port_name=value, id=8)
  direct: () = output_port(direct_value, name=direct, id=9)
  delayed: () = output_port(delayed_value, name=delayed, id=10)
}
"#,
        );

        assert_eq!(verify_hierarchy(&ordered_blocks(&package)), Ok(()));
    }

    #[test]
    fn combinational_output_remains_cyclic_with_a_registered_sibling() {
        let package = parse_blocks(
            r#"package public_mixed_output_cycle

block mixed(clk: clock, value: bits[8], direct: bits[8], delayed: bits[8]) {
  reg state(bits[8])
  value: bits[8] = input_port(name=value, id=1)
  stored: bits[8] = register_read(register=state, id=2)
  update: () = register_write(value, register=state, id=3)
  direct: () = output_port(value, name=direct, id=4)
  delayed: () = output_port(stored, name=delayed, id=5)
}

top block feedback(clk: clock, direct: bits[8], delayed: bits[8]) {
  instantiation component(block=mixed, kind=block)
  direct_value: bits[8] = instantiation_output(instantiation=component, port_name=direct, id=6)
  delayed_value: bits[8] = instantiation_output(instantiation=component, port_name=delayed, id=7)
  connected: () = instantiation_input(direct_value, instantiation=component, port_name=value, id=8)
  direct: () = output_port(direct_value, name=direct, id=9)
  delayed: () = output_port(delayed_value, name=delayed, id=10)
}
"#,
        );

        assert!(matches!(
            verify_hierarchy(&ordered_blocks(&package)),
            Err(BlockCodegenError::InvalidBlock(message))
                if message.contains("block `feedback` contains a combinational cycle")
        ));
    }

    #[test]
    fn forwarded_clock_reset_and_zero_width_ports_preserve_registered_feedback() {
        let package = parse_blocks(
            r#"package public_empty_reset_feedback

block delayed(clk: clock, rst: bits[1], empty: bits[0], value: bits[8], empty_result: bits[0], result: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg state(bits[8], reset_value=0)
  rst: bits[1] = input_port(name=rst, id=1)
  empty: bits[0] = input_port(name=empty, id=2)
  value: bits[8] = input_port(name=value, id=3)
  stored: bits[8] = register_read(register=state, id=4)
  update: () = register_write(value, register=state, reset=rst, id=5)
  empty_result: () = output_port(empty, name=empty_result, id=6)
  result: () = output_port(stored, name=result, id=7)
}

top block feedback(clk: clock, rst: bits[1], empty: bits[0], empty_result: bits[0], result: bits[8]) {
  instantiation component(block=delayed, kind=block)
  rst: bits[1] = input_port(name=rst, id=8)
  empty: bits[0] = input_port(name=empty, id=9)
  fed: bits[8] = instantiation_output(instantiation=component, port_name=result, id=10)
  empty_value: bits[0] = instantiation_output(instantiation=component, port_name=empty_result, id=11)
  reset_connected: () = instantiation_input(rst, instantiation=component, port_name=rst, id=12)
  empty_connected: () = instantiation_input(empty, instantiation=component, port_name=empty, id=13)
  value_connected: () = instantiation_input(fed, instantiation=component, port_name=value, id=14)
  empty_result: () = output_port(empty_value, name=empty_result, id=15)
  result: () = output_port(fed, name=result, id=16)
}
"#,
        );

        assert_eq!(verify_hierarchy(&ordered_blocks(&package)), Ok(()));
    }

    #[test]
    fn foreign_instances_remain_opaque_to_cycle_verification() {
        let package = parse_blocks(
            r#"package public_external_feedback

#[ffi_proto("""code_template: "external_cell {fn} (.value({value}), .result({return}));"
""")]
fn external_identity(value: bits[8] id=1) -> bits[8] {
  ret value: bits[8] = param(name=value, id=1)
}

top block feedback(result: bits[8]) {
  instantiation component(foreign_function=external_identity, kind=extern)
  fed: bits[8] = instantiation_output(instantiation=component, port_name=return, id=2)
  connected: () = instantiation_input(fed, instantiation=component, port_name=value, id=3)
  result: () = output_port(fed, name=result, id=4)
}
"#,
        );

        assert_eq!(verify_hierarchy(&ordered_blocks(&package)), Ok(()));
    }
}
