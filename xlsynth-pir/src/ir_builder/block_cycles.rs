// SPDX-License-Identifier: Apache-2.0

//! Hierarchical combinational dependencies without cloning or expanding blocks.

use std::collections::{BTreeMap, BTreeSet};

use super::{BlockCycleConnection, BlockCycleStep, BuilderError};
use crate::ir::{
    Block, BlockPort, Instantiation, InstantiationKind, NodePayload, NodeRef, Package,
};
use crate::ir_utils::operands;

type InputDependencies = BTreeSet<String>;
type OutputDependencies = BTreeMap<String, InputDependencies>;

/// Boundary relations of one block definition. All aliases of an electrical
/// net share a representative port in the dependency summary.
#[derive(Default)]
struct BlockSummary {
    outputs: OutputDependencies,
    port_classes: BTreeMap<String, String>,
}

/// A connected set of local net declarations, joined through child aliases.
#[derive(Default)]
struct NetClass<'a> {
    public_name: Option<&'a str>,
    witness: Option<NodeRef>,
}

/// Extends SSA dependencies with a vertex per electrically distinct net.
struct DependencyGraph<'a> {
    edges: Vec<Vec<NodeRef>>,
    local_nets: BTreeMap<&'a str, NodeRef>,
    public_nets: BTreeMap<&'a str, NodeRef>,
    classes: Vec<NetClass<'a>>,
}

impl<'a> DependencyGraph<'a> {
    /// Collapses passive aliases before adding directed driver and read edges.
    fn new(
        block: &'a Block,
        definitions: &BTreeMap<&str, &Instantiation>,
        summaries: &BTreeMap<String, BlockSummary>,
    ) -> Result<Self, BuilderError> {
        let mut edges = Vec::with_capacity(block.nodes.len() + block.inout_nets.len());
        for node in &block.nodes {
            let deps = operands(&node.payload);
            // The builder can call this before its structural verifier. An
            // out-of-bounds SSA operand must never alias an added net vertex.
            if let Some(bad) = deps.iter().find(|dep| dep.index >= block.nodes.len()) {
                return Err(invalid(format!(
                    "block '{}' references missing node {}",
                    block.name, bad.index
                )));
            }
            edges.push(deps);
        }

        let names: BTreeMap<_, _> = block
            .inout_nets
            .iter()
            .enumerate()
            .map(|(i, net)| (net.name.as_str(), i))
            .collect();
        let mut ties = vec![Vec::new(); block.inout_nets.len()];
        let mut first_bindings = BTreeMap::new();
        for node in &block.nodes {
            let NodePayload::ExtInstantiationInout {
                instantiation,
                port_name,
                net,
            } = &node.payload
            else {
                continue;
            };
            let index = *names
                .get(net.as_str())
                .ok_or_else(|| invalid(format!("undeclared inout net '{net}'")))?;
            let canonical = definitions
                .get(instantiation.as_str())
                .filter(|instance| instance.kind == InstantiationKind::Block)
                .and_then(|instance| summaries.get(&instance.block))
                .and_then(|child| child.port_classes.get(port_name))
                .ok_or_else(|| {
                    invalid(format!(
                        "no block instance inout named {instantiation}.{port_name}"
                    ))
                })?;
            let first = *first_bindings
                .entry((instantiation, canonical))
                .or_insert(index);
            if first != index {
                ties[first].push(index);
                ties[index].push(first);
            }
        }

        // Each component includes transitively joined aliases from different
        // instances. Each declared net and each tie is visited once.
        let mut vertices = vec![None; block.inout_nets.len()];
        let mut classes = Vec::new();
        let mut pending = Vec::new();
        for start in 0..vertices.len() {
            if vertices[start].is_some() {
                continue;
            }
            let vertex = NodeRef { index: edges.len() };
            edges.push(Vec::new());
            classes.push(NetClass::default());
            vertices[start] = Some(vertex);
            pending.push(start);
            while let Some(index) = pending.pop() {
                for &other in &ties[index] {
                    if vertices[other].is_none() {
                        vertices[other] = Some(vertex);
                        pending.push(other);
                    }
                }
            }
        }
        let mut local_nets = BTreeMap::new();
        let mut public_nets = BTreeMap::new();
        for (net, vertex) in block.inout_nets.iter().zip(vertices) {
            let vertex = vertex.expect("visited each net component");
            local_nets.insert(net.name.as_str(), vertex);
            for name in &net.ports {
                public_nets.insert(name.as_str(), vertex);
            }
        }
        for name in block.inout_ports() {
            let vertex = public_nets
                .get(name)
                .ok_or_else(|| invalid(format!("inout port '{name}' has no declared net")))?;
            classes[vertex.index - block.nodes.len()]
                .public_name
                .get_or_insert(name);
        }
        Ok(Self {
            edges,
            local_nets,
            public_nets,
            classes,
        })
    }

    /// Records a concrete IR node to identify a resolved net in a cycle.
    fn net(
        &mut self,
        name: &str,
        node: NodeRef,
        node_count: usize,
    ) -> Result<NodeRef, BuilderError> {
        let vertex = *self
            .local_nets
            .get(name)
            .ok_or_else(|| invalid(format!("undeclared inout net '{name}'")))?;
        self.classes[vertex.index - node_count]
            .witness
            .get_or_insert(node);
        Ok(vertex)
    }
}

/// Caches boundary dependencies for each verified block in an unchanged
/// package.
#[derive(Default)]
pub(crate) struct BlockHierarchy {
    summaries: BTreeMap<String, BlockSummary>,
    active: BTreeSet<String>,
}

/// Checks implicit instance connections as well as explicit graph edges.
///
/// Each distinct child block is checked once and summarized by the input ports
/// reaching each output. Registers cut dependencies, so sequential feedback is
/// permitted. Traversal stops at extern boundaries because the foreign
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
        let summary = graph_output_dependencies(block, &graph)?;
        self.summaries.insert(block.name.clone(), summary);
        Ok(())
    }

    /// Checks the hierarchy and returns cycle-free, instance-aware graph edges.
    fn verify_graph<'a>(
        &mut self,
        block: &'a Block,
        package: &Package,
    ) -> Result<DependencyGraph<'a>, BuilderError> {
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
        let definitions: BTreeMap<_, _> = block
            .instantiations
            .iter()
            .map(|inst| (inst.name.as_str(), inst))
            .collect();
        let mut graph = DependencyGraph::new(block, &definitions, &self.summaries)?;
        let mut input_connections: BTreeMap<&str, BTreeMap<&str, NodeRef>> = BTreeMap::new();
        let mut output_connections = Vec::new();
        for (index, node) in block.nodes.iter().enumerate() {
            let reference = NodeRef { index };
            match &node.payload {
                NodePayload::ExtInoutRead { net } => {
                    let net = graph.net(net, reference, block.nodes.len())?;
                    graph.edges[index].push(net);
                }
                NodePayload::ExtInoutDrive { net, .. } => {
                    let net = graph.net(net, reference, block.nodes.len())?;
                    graph.edges[net.index].push(reference);
                }
                NodePayload::InstantiationInput {
                    instantiation,
                    port_name,
                    ..
                } => {
                    input_connections
                        .entry(instantiation)
                        .or_default()
                        .insert(port_name, NodeRef { index });
                }
                NodePayload::InstantiationOutput {
                    instantiation,
                    port_name,
                } => {
                    output_connections.push((instantiation, port_name, index));
                }
                NodePayload::ExtInstantiationInout {
                    instantiation,
                    port_name,
                    net,
                } => {
                    let net = graph.net(net, reference, block.nodes.len())?;
                    // Graph construction has resolved every binding's instance
                    // and public alias, including during builder prechecks.
                    let child = &self.summaries[&definitions[instantiation.as_str()].block];
                    if child.port_classes[port_name] == *port_name {
                        // Each child net has exactly one boundary
                        // representative. Passive aliases already share a
                        // vertex in this graph.
                        input_connections
                            .entry(instantiation)
                            .or_default()
                            .insert(port_name, net);
                        output_connections.push((instantiation, port_name, net.index));
                    }
                }
                _ => { /* Other nodes have no implicit instance connections. */ }
            }
        }
        for (instantiation, port_name, index) in output_connections {
            let edges = &mut graph.edges[index];
            let definition = definitions.get(instantiation.as_str()).ok_or_else(|| {
                invalid(format!(
                    "unknown instance '{instantiation}' in block '{}'",
                    block.name
                ))
            })?;
            let connected = input_connections.get(instantiation.as_str());
            match definition.kind {
                InstantiationKind::Block => {
                    let summary = &self.summaries[&definition.block];
                    let inputs = summary.outputs.get(port_name).ok_or_else(|| {
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
                    // The extern's internal connectivity is opaque.
                }
            }
        }
        verify_graph_acyclic(block, &graph)?;
        self.active.remove(&block.name);
        Ok(graph)
    }
}

/// Checks even unobserved nodes in O(nodes + edges) time and O(nodes) space.
fn verify_graph_acyclic(block: &Block, graph: &DependencyGraph) -> Result<(), BuilderError> {
    let mut colors = vec![0u8; graph.edges.len()];
    for root in 0..graph.edges.len() {
        if colors[root] == 2 {
            continue;
        }
        colors[root] = 1;
        let mut pending = vec![(root, 0usize)];
        while let Some((node, next_edge)) = pending.last_mut() {
            if let Some(dependency) = graph.edges[*node].get(*next_edge) {
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
                        let start = pending
                            .iter()
                            .position(|(node, _)| *node == index)
                            .expect("active node is on the DFS path");
                        return Err(cycle_error(block, graph, &pending[start..]));
                    }
                    0 => {
                        colors[index] = 1;
                        pending.push((index, 0));
                    }
                    _ => {
                        // This dependency's subgraph has already been checked.
                    }
                }
            } else {
                let (node, _) = pending.pop().expect("DFS frame exists");
                colors[node] = 2;
            }
        }
    }
    Ok(())
}

/// Converts the cyclic segment of the DFS path into block-local text IDs.
fn cycle_error(block: &Block, graph: &DependencyGraph, path: &[(usize, usize)]) -> BuilderError {
    let witness = |index: usize| {
        block.nodes.get(index).unwrap_or_else(|| {
            let origin = graph.classes[index - block.nodes.len()]
                .witness
                .expect("a net on a directed cycle has an access or a binding");
            block.get_node(origin)
        })
    };
    let steps = path
        .iter()
        .zip(path.iter().cycle().skip(1))
        .map(|(&(index, _), &(next, _))| {
            let node = witness(index);
            let dependency = witness(next);
            let connection = match (&node.payload, &dependency.payload) {
                (
                    NodePayload::InstantiationOutput {
                        instantiation,
                        port_name: output_port,
                    },
                    NodePayload::InstantiationInput {
                        port_name: input_port,
                        ..
                    },
                ) => Some(BlockCycleConnection {
                    instance: instantiation.clone(),
                    input_port: input_port.clone(),
                    output_port: output_port.clone(),
                    input_node_id: dependency.text_id,
                }),
                _ => None,
            };
            BlockCycleStep {
                node_id: node.text_id,
                connection,
            }
        })
        .collect();
    BuilderError::CombinationalCycle {
        block: block.name.clone(),
        steps,
    }
}

/// Traces each output cone in a checked graph, retaining only interface sets.
///
/// Each output visits a reachable node/edge at most once. The scratch storage
/// is O(nodes), independent of transitive dependency sizes at interior nodes;
/// only the output-to-input relation itself can require quadratic storage.
fn graph_output_dependencies(
    block: &Block,
    graph: &DependencyGraph,
) -> Result<BlockSummary, BuilderError> {
    // Tag visits by output instead of clearing O(nodes) storage for each cone.
    let mut visited = vec![None; graph.edges.len()];
    let mut pending = Vec::new();
    let mut summary = BlockSummary::default();
    for port in &block.ports {
        let (name, root) = match port {
            BlockPort::Output(reference) => {
                let Some(NodePayload::OutputPort { name, .. }) =
                    block.nodes.get(reference.index).map(|n| &n.payload)
                else {
                    return Err(invalid("block output references an invalid node"));
                };
                (name, *reference)
            }
            BlockPort::Inout(name) => {
                let root = graph.public_nets[name.as_str()];
                let representative = graph.classes[root.index - block.nodes.len()]
                    .public_name
                    .expect("public net has a representative");
                summary
                    .port_classes
                    .insert(name.clone(), representative.to_string());
                if representative != name {
                    continue;
                }
                (name, root)
            }
            BlockPort::Clock(_) | BlockPort::Input(_) => continue,
        };
        let inputs = summary.outputs.entry(name.clone()).or_default();
        visited[root.index] = Some(root);
        pending.push(root);
        while let Some(vertex) = pending.pop() {
            if let Some(node) = block.nodes.get(vertex.index) {
                if let NodePayload::InputPort { name, .. } = &node.payload {
                    inputs.insert(name.clone());
                }
            } else if let Some(name) = graph.classes[vertex.index - block.nodes.len()].public_name
                && vertex != root
            {
                // Other public nets can be externally driven. The root net
                // summarizes its own drivers, independent of its alias names.
                inputs.insert(name.to_string());
            }
            for dependency in &graph.edges[vertex.index] {
                if visited[dependency.index] != Some(root) {
                    visited[dependency.index] = Some(root);
                    pending.push(*dependency);
                }
            }
        }
    }
    Ok(summary)
}

fn invalid(reason: impl Into<String>) -> BuilderError {
    BuilderError::InvalidOperation(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlockBuilder;
    use crate::ir::Type;
    use crate::ir_parser::Parser;

    fn check(source: &str) -> Result<(), BuilderError> {
        let package = Parser::new(source).parse_and_verify_package().unwrap();
        verify_block_combinational_cycles(package.get_top_block().unwrap(), &package)
    }

    #[test]
    fn builder_cycle_precheck_handles_unverified_inout_bindings() {
        let source = r#"package precheck
block child(a: bits[1], b: bits[1], sensed: bits[1]) {
  ext_inout_net wire(bits[1], ports=[a, b])
  read: bits[1] = ext_inout_read(net=wire, id=1)
  sensed: () = output_port(read, name=sensed, id=2)
}
top block parent() {
  ext_inout_net local(bits[1])
  instantiation c(block=child, kind=block)
  first: () = ext_instantiation_inout(instantiation=c, port_name=a, net=local, id=3)
  second: () = ext_instantiation_inout(instantiation=c, port_name=b, net=local, id=4)
  sense: bits[1] = instantiation_output(instantiation=c, port_name=sensed, id=5)
}
"#;
        for (from, to) in [
            ("net=local, id=3", "net=absent, id=3"),
            (
                "instantiation=c, port_name=b",
                "instantiation=absent, port_name=b",
            ),
            ("port_name=b, net=", "port_name=absent, net="),
            (
                "first: () = ext_instantiation_inout(instantiation=c, port_name=a, net=local, id=3)",
                "",
            ),
        ] {
            let malformed = source.replace(from, to);
            let package = Parser::new(&malformed).parse_package().unwrap();
            assert!(
                verify_block_combinational_cycles(package.get_top_block().unwrap(), &package)
                    .is_err(),
                "{malformed}"
            );
        }
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
        assert_eq!(hierarchy.summaries["child"].outputs, expected);

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
        assert_eq!(hierarchy.summaries["disjoint"].outputs, expected);
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
