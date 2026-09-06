// SPDX-License-Identifier: Apache-2.0

//! Hierarchical combinational dependencies without cloning or expanding blocks.

use std::collections::{BTreeMap, BTreeSet};

use super::BuilderError;
use crate::ir::{Block, BlockPort, InstantiationKind, NodePayload, NodeRef, Package};
use crate::ir_utils::operands;

type InputDependencies = BTreeSet<String>;
type OutputDependencies = BTreeMap<String, InputDependencies>;

#[derive(Default)]
struct Hierarchy {
    summaries: BTreeMap<String, OutputDependencies>,
    active: BTreeSet<String>,
}

/// Checks implicit instance connections as well as explicit graph edges.
///
/// Each distinct child graph is visited once and summarized by the input ports
/// reaching each output. Registers cut dependencies, so sequential feedback is
/// permitted. Extern outputs conservatively depend on every wired input: the
/// foreign function's IR body may be a stub, not the instantiated RTL's logic.
pub(super) fn verify_block_combinational_cycles(
    block: &Block,
    package: &Package,
) -> Result<(), BuilderError> {
    Hierarchy::default().summarize(block, package)
}

impl Hierarchy {
    /// Memoizes each child's interface dependencies, rejecting recursive
    /// hierarchy.
    fn summarize(&mut self, block: &Block, package: &Package) -> Result<(), BuilderError> {
        if self.summaries.contains_key(&block.name) {
            return Ok(());
        }
        if !self.active.insert(block.name.clone()) {
            return Err(invalid(format!(
                "recursive block instantiation involving '{}'",
                block.name
            )));
        }
        for instance in &block.instantiations {
            if instance.kind == InstantiationKind::Block {
                let child = package
                    .get_block(&instance.block)
                    .ok_or_else(|| invalid(format!("missing child block '{}'", instance.block)))?;
                if child.clock_port_name().is_some() && block.clock_port_name().is_none() {
                    return Err(invalid(format!(
                        "clocked child '{}' requires a clock on parent block '{}'",
                        child.name, block.name
                    )));
                }
                // Validate even children with no visible outputs: their
                // internal combinational logic can still
                // contain an invalid cycle.
                self.summarize(child, package)?;
            }
        }
        let mut input_drivers: BTreeMap<&str, BTreeMap<&str, NodeRef>> = BTreeMap::new();
        for node in &block.nodes {
            if let NodePayload::InstantiationInput {
                instantiation,
                port_name,
                arg,
            } = &node.payload
            {
                input_drivers
                    .entry(instantiation)
                    .or_default()
                    .insert(port_name, *arg);
            }
        }
        let definitions: BTreeMap<_, _> = block
            .instantiations
            .iter()
            .map(|instance| (instance.name.as_str(), instance))
            .collect();
        let mut dependencies = Vec::with_capacity(block.nodes.len());
        for node in &block.nodes {
            let mut edges = operands(&node.payload);
            if let NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } = &node.payload
            {
                let definition = definitions.get(instantiation.as_str()).ok_or_else(|| {
                    invalid(format!(
                        "unknown instance '{instantiation}' in block '{}'",
                        block.name
                    ))
                })?;
                let connected = input_drivers.get(instantiation.as_str());
                match definition.kind {
                    InstantiationKind::Block => {
                        let summary = &self.summaries[&definition.block];
                        let inputs = summary.get(port_name).ok_or_else(|| {
                            invalid(format!(
                                "unknown output '{port_name}' on child '{}'",
                                definition.block
                            ))
                        })?;
                        for input in inputs {
                            let driver = connected
                                .and_then(|ports| ports.get(input.as_str()))
                                .ok_or_else(|| {
                                    invalid(format!(
                                        "unconnected input '{input}' on instance '{instantiation}'"
                                    ))
                                })?;
                            edges.push(*driver);
                        }
                    }
                    InstantiationKind::Extern => {
                        if let Some(connected) = connected {
                            edges.extend(connected.values().copied());
                        }
                    }
                }
            }
            dependencies.push(edges);
        }
        let reached_inputs = graph_input_dependencies(block, &dependencies)?;
        let mut summary = BTreeMap::new();
        for port in &block.ports {
            if let BlockPort::Output(reference) = port {
                let node = block
                    .nodes
                    .get(reference.index)
                    .ok_or_else(|| invalid("block output references a missing node"))?;
                let NodePayload::OutputPort { name, .. } = &node.payload else {
                    return Err(invalid("block output does not reference an output_port"));
                };
                summary.insert(name.clone(), reached_inputs[reference.index].clone());
            }
        }
        self.active.remove(&block.name);
        self.summaries.insert(block.name.clone(), summary);
        Ok(())
    }
}

/// Computes reachability with iterative DFS, checking even unobserved nodes.
fn graph_input_dependencies(
    block: &Block,
    dependencies: &[Vec<NodeRef>],
) -> Result<Vec<InputDependencies>, BuilderError> {
    let mut colors = vec![0u8; block.nodes.len()];
    let mut reached = vec![BTreeSet::new(); block.nodes.len()];
    for root in 0..block.nodes.len() {
        if colors[root] == 2 {
            continue;
        }
        colors[root] = 1;
        let mut pending = vec![(root, 0usize)];
        while let Some((node, next_edge)) = pending.last_mut() {
            if let Some(dependency) = dependencies[*node].get(*next_edge) {
                *next_edge += 1;
                let index = dependency.index;
                let color = colors.get(index).ok_or_else(|| {
                    invalid(format!(
                        "block '{}' references missing node {index}",
                        block.name
                    ))
                })?;
                match color {
                    1 => {
                        return Err(invalid(format!(
                            "combinational cycle in block '{}' through node {}",
                            block.name, block.nodes[index].text_id
                        )));
                    }
                    0 => {
                        colors[index] = 1;
                        pending.push((index, 0));
                    }
                    _ => {
                        // A completed dependency already has its input summary.
                    }
                }
            } else {
                let (node, _) = pending.pop().expect("DFS frame exists");
                let mut inputs = BTreeSet::new();
                if let NodePayload::InputPort { name, .. } = &block.nodes[node].payload {
                    inputs.insert(name.clone());
                }
                for dependency in &dependencies[node] {
                    inputs.extend(reached[dependency.index].iter().cloned());
                }
                reached[node] = inputs;
                colors[node] = 2;
            }
        }
    }
    Ok(reached)
}

fn invalid(reason: impl Into<String>) -> BuilderError {
    BuilderError::InvalidOperation(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    fn check(source: &str) -> Result<(), BuilderError> {
        let package = Parser::new(source).parse_and_verify_package().unwrap();
        verify_block_combinational_cycles(package.get_top_block().unwrap(), &package)
    }

    #[test]
    fn rejects_feedback_through_a_combinational_child() {
        let source = r#"package cycle
block child(x: bits[8], y: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}
top block parent(out: bits[8]) {
  instantiation child_instance(block=child, kind=block)
  value: bits[8] = instantiation_output(instantiation=child_instance, port_name=y, id=3)
  connect: () = instantiation_input(value, instantiation=child_instance, port_name=x, id=4)
  out: () = output_port(value, name=out, id=5)
}
"#;
        assert!(
            matches!(check(source), Err(BuilderError::InvalidOperation(message)) if message.contains("combinational cycle"))
        );
    }

    #[test]
    fn allows_register_broken_feedback_and_constant_output_feedback() {
        let registered = r#"package sequential
block child(clk: clock, x: bits[8], y: bits[8]) {
  reg state(bits[8])
  x: bits[8] = input_port(name=x, id=1)
  q: bits[8] = register_read(register=state, id=2)
  update: () = register_write(x, register=state, id=3)
  y: () = output_port(q, name=y, id=4)
}
top block parent(clk: clock, out: bits[8]) {
  instantiation child_instance(block=child, kind=block)
  value: bits[8] = instantiation_output(instantiation=child_instance, port_name=y, id=5)
  connect: () = instantiation_input(value, instantiation=child_instance, port_name=x, id=6)
  out: () = output_port(value, name=out, id=7)
}
"#;
        check(registered).unwrap();
        let constant = r#"package independent
block child(x: bits[8], y: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  zero: bits[8] = literal(value=0, id=2)
  y: () = output_port(zero, name=y, id=3)
}
top block parent(out: bits[8]) {
  instantiation child_instance(block=child, kind=block)
  value: bits[8] = instantiation_output(instantiation=child_instance, port_name=y, id=4)
  connect: () = instantiation_input(value, instantiation=child_instance, port_name=x, id=5)
  out: () = output_port(value, name=out, id=6)
}
"#;
        check(constant).unwrap();
    }

    #[test]
    fn rejects_cross_instance_feedback_through_nested_hierarchy() {
        let source = r#"package nested
block leaf(x: bits[8], y: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}
block child(x: bits[8], y: bits[8]) {
  instantiation inner(block=leaf, kind=block)
  x: bits[8] = input_port(name=x, id=3)
  connect: () = instantiation_input(x, instantiation=inner, port_name=x, id=4)
  value: bits[8] = instantiation_output(instantiation=inner, port_name=y, id=5)
  y: () = output_port(value, name=y, id=6)
}
top block parent(out: bits[8]) {
  instantiation first(block=child, kind=block)
  instantiation second(block=child, kind=block)
  left: bits[8] = instantiation_output(instantiation=first, port_name=y, id=7)
  right: bits[8] = instantiation_output(instantiation=second, port_name=y, id=8)
  first_input: () = instantiation_input(right, instantiation=first, port_name=x, id=9)
  second_input: () = instantiation_input(left, instantiation=second, port_name=x, id=10)
  out: () = output_port(left, name=out, id=11)
}
"#;
        assert!(
            matches!(check(source), Err(BuilderError::InvalidOperation(message)) if message.contains("combinational cycle"))
        );
    }

    #[test]
    fn rejects_hidden_cycles_in_children_without_outputs() {
        let source = r#"package hidden
block leaf(x: bits[8], y: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  y: () = output_port(x, name=y, id=2)
}
block child() {
  instantiation inner(block=leaf, kind=block)
  value: bits[8] = instantiation_output(instantiation=inner, port_name=y, id=3)
  connect: () = instantiation_input(value, instantiation=inner, port_name=x, id=4)
}
top block parent() {
  instantiation hidden_instance(block=child, kind=block)
}
"#;
        assert!(
            matches!(check(source), Err(BuilderError::InvalidOperation(message)) if message.contains("combinational cycle"))
        );
    }

    #[test]
    fn rejects_missing_clocks_inside_existing_nested_hierarchy() {
        let source = r#"package nested_clock
block leaf(tick: clock, y: bits[8]) {
  zero: bits[8] = literal(value=0, id=1)
  y: () = output_port(zero, name=y, id=2)
}
block middle(y: bits[8]) {
  instantiation inner(block=leaf, kind=block)
  value: bits[8] = instantiation_output(instantiation=inner, port_name=y, id=3)
  y: () = output_port(value, name=y, id=4)
}
top block parent(clk: clock, out: bits[8]) {
  instantiation middle_instance(block=middle, kind=block)
  value: bits[8] = instantiation_output(instantiation=middle_instance, port_name=y, id=5)
  out: () = output_port(value, name=out, id=6)
}
"#;
        assert_eq!(
            check(source),
            Err(invalid(
                "clocked child 'leaf' requires a clock on parent block 'middle'"
            ))
        );
    }

    #[test]
    fn extern_feedback_does_not_trust_a_constant_stub_body() {
        let source = r#"package foreign_cycle
fn foreign(x: bits[8] id=1) -> bits[8] {
  ret stub: bits[8] = literal(value=0, id=2)
}
top block parent(out: bits[8]) {
  instantiation external(foreign_function=foreign, kind=extern)
  value: bits[8] = instantiation_output(instantiation=external, port_name=return, id=3)
  connect: () = instantiation_input(value, instantiation=external, port_name=x, id=4)
  out: () = output_port(value, name=out, id=5)
}
"#;
        assert!(
            matches!(check(source), Err(BuilderError::InvalidOperation(message)) if message.contains("combinational cycle"))
        );
    }
}
