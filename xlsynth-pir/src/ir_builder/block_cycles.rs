// SPDX-License-Identifier: Apache-2.0

//! Hierarchical combinational dependencies without cloning or expanding blocks.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    BuilderError,
    port_ranges::{Direction, Graph, Link, Network, Span, Summary},
};
use crate::ir::{Block, BlockPort, InstantiationKind, NaryOp, NodePayload, NodeRef, Package, Unop};
use crate::ir_utils::operands;

/// Caches boundary dependencies for each verified block in an unchanged
/// package.
#[derive(Default)]
pub(crate) struct BlockHierarchy {
    summaries: BTreeMap<String, Summary>,
    active: BTreeSet<String>,
}

/// Checks implicit instance connections as well as explicit graph edges.
///
/// Each distinct child block is checked once and summarized by the input ports
/// reaching each output. Registers cut dependencies, so sequential feedback is
/// permitted. Inout reads are sources and drives are sinks. Electrical
/// connections through resolved nets do not create dataflow edges.
/// Traversal stops at extern boundaries because the foreign
/// function body can be a placeholder for separately supplied RTL.
pub(super) fn verify_block_combinational_cycles(
    block: &Block,
    package: &Package,
) -> Result<(), BuilderError> {
    // Only children need interface summaries; nobody consumes the root's.
    BlockHierarchy::default()
        .verify_graph(block, package)
        .map(|_| ())
}

impl BlockHierarchy {
    /// Memoizes only each child's interface dependencies, not interior nodes.
    pub(crate) fn summarize(
        &mut self,
        block: &Block,
        package: &Package,
    ) -> Result<(), BuilderError> {
        if self.summaries.contains_key(&block.name) {
            return Ok(());
        }
        let graph = self.verify_graph(block, package)?;
        let summary = graph.summarize();
        self.summaries.insert(block.name.clone(), summary);
        Ok(())
    }

    /// Checks the hierarchy and returns cycle-free, instance-aware graph edges.
    fn verify_graph(&mut self, block: &Block, package: &Package) -> Result<Graph, BuilderError> {
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
        let children: BTreeMap<_, _> = block
            .instantiations
            .iter()
            .filter(|i| i.kind == InstantiationKind::Block)
            .map(|i| (i.name.as_str(), &self.summaries[&i.block]))
            .collect();
        let graph = build_graph(block, &children)?;
        graph.check(block)?;
        self.active.remove(&block.name);
        Ok(graph)
    }
}

/// Construct the local range graph using only cached child interfaces.
fn build_graph(block: &Block, children: &BTreeMap<&str, &Summary>) -> Result<Graph, BuilderError> {
    let mut graph = Network::default();
    let mut pins = BTreeMap::<(&str, &str), Span<usize>>::new();
    let mut nodes = Vec::with_capacity(block.nodes.len());
    for (index, node) in block.nodes.iter().enumerate() {
        if let Some(bad) = operands(&node.payload)
            .into_iter()
            .find(|v| v.index >= block.nodes.len())
        {
            return Err(invalid(format!(
                "block '{}' references missing node {}",
                block.name, bad.index
            )));
        }
        let ty = match &node.payload {
            NodePayload::OutputPort { arg, .. } | NodePayload::InstantiationInput { arg, .. } => {
                block.get_node_ty(*arg)
            }
            _ => &node.ty,
        };
        // Unit-valued operations still occupy a dependency vertex.
        let width = ty
            .checked_bit_count()
            .ok_or_else(|| invalid("signal width overflow"))?
            .max(1);
        nodes.push(graph.signal(width, NodeRef { index }));
    }
    for port in &block.ports {
        let (r, dir) = match port {
            BlockPort::Input(r) => (*r, Direction::Input),
            BlockPort::Output(r) => (*r, Direction::Output),
            BlockPort::Inout(_) | BlockPort::Clock(_) => continue,
        };
        let signal = *nodes
            .get(r.index)
            .ok_or_else(|| invalid("invalid interface node"))?;
        let name = match &block.get_node(r).payload {
            NodePayload::InputPort { name, .. } | NodePayload::OutputPort { name, .. } => name,
            _ => return Err(invalid("invalid interface node")),
        };
        graph.ports.insert(name.clone(), (signal, dir));
    }
    for (index, node) in block.nodes.iter().enumerate() {
        let sink = nodes[index];
        match &node.payload {
            NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } => {
                pins.insert((instantiation, port_name), sink);
            }
            NodePayload::InstantiationInput {
                instantiation,
                port_name,
                arg,
            } => {
                pins.insert((instantiation, port_name), sink);
                graph.connect(sink, Link::Aligned, nodes[arg.index])?;
            }
            NodePayload::OutputPort { arg, .. } | NodePayload::Unop(Unop::Identity, arg) => {
                graph.connect(sink, Link::Aligned, nodes[arg.index])?;
            }
            NodePayload::BitSlice { arg, start, width } => {
                let source = nodes[arg.index];
                let available = source.width.saturating_sub(*start).min(*width);
                if available > 0 {
                    graph.connect(
                        sink.part(0, available),
                        Link::Aligned,
                        source.part(*start, available),
                    )?;
                }
            }
            NodePayload::Nary(NaryOp::Concat, args) => {
                let mut start = 0usize;
                for a in args.iter().rev() {
                    let width = block
                        .get_node_ty(*a)
                        .checked_bit_count()
                        .ok_or_else(|| invalid("concat width overflow"))?;
                    if width > 0 {
                        graph.connect(sink.part(start, width), Link::Aligned, nodes[a.index])?;
                    }
                    start = start
                        .checked_add(width)
                        .ok_or_else(|| invalid("concat width overflow"))?;
                }
            }
            _ => {
                // Inout reads and bindings have no SSA operands. A drive
                // consumes data and enable, and creates no edge to a read.
                for dep in operands(&node.payload) {
                    graph.connect(sink, Link::All, nodes[dep.index])?;
                }
            }
        }
    }
    for (&instance, child) in children {
        for (to, kind, from) in child.iter() {
            let Some(out) = pins.get(&(instance, &to.signal)) else {
                // An unused ordinary child output contributes no parent edges.
                continue;
            };
            let input = pins.get(&(instance, &from.signal)).ok_or_else(|| {
                invalid(format!(
                    "unconnected input '{}' on instance '{instance}'",
                    from.signal
                ))
            })?;
            graph.connect(
                out.part(to.start, to.width),
                *kind,
                input.part(from.start, from.width),
            )?;
        }
    }
    Ok(graph.finish())
}

pub(super) fn invalid(reason: impl Into<String>) -> BuilderError {
    BuilderError::InvalidOperation(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Type;
    use crate::ir_parser::Parser;
    use crate::{BlockBuilder, BlockCycleConnection, BlockCycleStep};

    fn check(source: &str) -> Result<(), BuilderError> {
        let package = Parser::new(source).parse_and_verify_package().unwrap();
        verify_block_combinational_cycles(package.get_top_block().unwrap(), &package)
    }

    #[test]
    fn summaries_keep_shared_and_independent_output_cones_separate() {
        let source = r#"package shared_cones
top block child(x: bits[8], y: bits[8], z: bits[8], left: bits[8], right: bits[8], direct: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  y: bits[8] = input_port(name=y, id=2)
  z: bits[8] = input_port(name=z, id=3)
  shared: bits[8] = xor(x, y, id=4)
  diamond: bits[8] = and(shared, x, id=5)
  combined: bits[8] = xor(shared, z, id=6)
  left: () = output_port(diamond, name=left, id=7)
  right: () = output_port(combined, name=right, id=8)
  direct: () = output_port(z, name=direct, id=9)
}
"#;
        let package = Parser::new(source).parse_and_verify_package().unwrap();
        let mut hierarchy = BlockHierarchy::default();
        hierarchy
            .summarize(package.get_top_block().unwrap(), &package)
            .unwrap();
        let expected = BTreeMap::from([
            ("left".into(), BTreeSet::from(["x".into(), "y".into()])),
            (
                "right".into(),
                BTreeSet::from(["x".into(), "y".into(), "z".into()]),
            ),
            ("direct".into(), BTreeSet::from(["z".into()])),
        ]);
        let mut actual = BTreeMap::<String, BTreeSet<String>>::new();
        for (out, _, input) in &hierarchy.summaries["child"] {
            actual
                .entry(out.signal.clone())
                .or_default()
                .insert(input.signal.clone());
        }
        assert_eq!(actual, expected);

        let mut root_check = BlockHierarchy::default();
        root_check
            .verify_graph(package.get_top_block().unwrap(), &package)
            .unwrap();
        assert!(root_check.summaries.is_empty());
    }

    #[test]
    fn summaries_handle_many_disjoint_output_cones() {
        let mut builder = BlockBuilder::new("disjoint");
        let mut expected = BTreeMap::new();
        for index in 0..4096 {
            let input_name = format!("input_{index}");
            let output_name = format!("output_{index}");
            let input = builder.input_port(&input_name, Type::Bits(1)).unwrap();
            builder.output_port(&output_name, input).unwrap();
            expected.insert(output_name, BTreeSet::from([input_name]));
        }
        let package = builder.build_package("disjoint_cones").unwrap();
        let mut hierarchy = BlockHierarchy::default();
        hierarchy
            .summarize(package.get_top_block().unwrap(), &package)
            .unwrap();
        let mut actual = BTreeMap::<String, BTreeSet<String>>::new();
        for (out, _, input) in &hierarchy.summaries["disjoint"] {
            actual
                .entry(out.signal.clone())
                .or_default()
                .insert(input.signal.clone());
        }
        assert_eq!(actual, expected);
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
        assert_eq!(
            check(source),
            Err(BuilderError::CombinationalCycle {
                block: "parent".into(),
                steps: vec![
                    BlockCycleStep {
                        node_id: 3,
                        connection: Some(BlockCycleConnection {
                            instance: "child_instance".into(),
                            input_port: "x".into(),
                            output_port: "y".into(),
                            input_node_id: 4,
                        }),
                    },
                    BlockCycleStep {
                        node_id: 4,
                        connection: None,
                    },
                ],
            })
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
        assert!(matches!(
            check(source),
            Err(BuilderError::CombinationalCycle { .. })
        ));
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
        assert!(matches!(
            check(source),
            Err(BuilderError::CombinationalCycle { .. })
        ));
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
    fn extern_feedback_stops_at_boundary_even_with_passthrough_body() {
        let source = r#"package foreign_cycle
fn foreign(x: bits[8] id=1) -> bits[8] {
  ret x: bits[8] = param(name=x, id=1)
}
top block parent(out: bits[8]) {
  instantiation external(foreign_function=foreign, kind=extern)
  value: bits[8] = instantiation_output(instantiation=external, port_name=return, id=3)
  connect: () = instantiation_input(value, instantiation=external, port_name=x, id=4)
  out: () = output_port(value, name=out, id=5)
}
"#;
        check(source).unwrap();
    }
}
