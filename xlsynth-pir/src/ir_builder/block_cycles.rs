// SPDX-License-Identifier: Apache-2.0

//! Hierarchical combinational dependencies without cloning or expanding blocks.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

mod slices;

use super::{BlockCycleConnection, BlockCycleStep, BuilderError};
use crate::block_feedback::InstanceConnections;
use crate::ir::{Block, InstantiationKind, NodeGraph, NodePayload, NodeRef, Package};
use crate::ir_utils::operands;

type InputDependencies = BTreeSet<String>;
type OutputDependencies = BTreeMap<String, InputDependencies>;

/// Local feedback candidates and the child instances that contain others.
#[derive(Default)]
struct Candidates {
    roots: Vec<usize>,
    children: Vec<usize>,
}

/// Definition-level whole-port summaries and candidate paths from the root.
#[derive(Default)]
struct BlockSummary {
    outputs: OutputDependencies,
    port_classes: BTreeMap<String, String>,
}

/// One local net or a set of local nets joined through child ports.
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
            let missing_pin = || {
                invalid(format!(
                    "no block instance inout named {instantiation}.{port_name}"
                ))
            };
            let instance = definitions
                .get(instantiation.as_str())
                .ok_or_else(missing_pin)?;
            if matches!(instance.kind, InstantiationKind::OpaqueExternBlock { .. }) {
                // No passive aliases are known across the opaque boundary.
                continue;
            }
            let canonical = instance
                .referenced_block()
                .and_then(|name| summaries.get(name))
                .and_then(|child| child.port_classes.get(port_name))
                .ok_or_else(missing_pin)?;
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
        let local_nets: BTreeMap<_, _> = block
            .inout_nets
            .iter()
            .zip(vertices)
            .map(|(net, vertex)| {
                (
                    net.name.as_str(),
                    vertex.expect("visited each net component"),
                )
            })
            .collect();
        let mut public_nets = BTreeMap::new();
        for port in block.inout_ports() {
            let Some(NodePayload::ExtInoutPort { name, net, .. }) =
                block.nodes.get(port.index).map(|node| &node.payload)
            else {
                return Err(invalid("block inout port references an invalid node"));
            };
            let vertex = *local_nets
                .get(net.as_str())
                .ok_or_else(|| invalid(format!("undeclared inout net '{net}'")))?;
            public_nets.insert(name.as_str(), vertex);
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
    candidates: BTreeMap<String, Candidates>,
}

/// Checks implicit instance connections as well as explicit graph edges.
///
/// Each distinct child block is summarized by the input ports reaching each
/// output. Registers and opaque externs cut same-cycle dependencies.
/// Suspected cycles follow selected bits and aggregate elements through block
/// ports and local calls under the literal bindings supplied by the root.
/// Operations with runtime offsets conservatively depend on their operands.
///
/// The block can be under construction. Its children and local callees must
/// belong to a verified package, and its ports and nodes must be well typed.
pub fn verify_block_combinational_cycles(
    block: &Block,
    package: &Package,
) -> Result<(), BuilderError> {
    let mut hierarchy = Hierarchy::default();
    hierarchy.verify_graph(block, package)?;
    if !hierarchy.candidates.is_empty() {
        slices::Checker::new(package).check(block, &hierarchy.candidates)?;
    }
    Ok(())
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
        let dependencies = self.verify_graph(block, package)?;
        let summary = graph_output_dependencies(block, &dependencies);
        self.summaries.insert(block.name.clone(), summary);
        Ok(())
    }

    /// Checks the hierarchy and returns its conservative whole-node
    /// dependencies.
    fn verify_graph(
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
        let mut candidates = Candidates::default();
        for (index, instance) in block.instantiations.iter().enumerate() {
            if instance.kind == InstantiationKind::Block {
                let child = package
                    .get_block(child_name)
                    .ok_or_else(|| invalid(format!("missing child block '{child_name}'")))?;
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
                if self.candidates.contains_key(&child.name) {
                    candidates.children.push(index);
                }
            }
        }
        let connections = InstanceConnections::new(block);
        let mut dependencies = Vec::with_capacity(block.nodes.len());
        for node in &block.nodes {
            let mut edges = operands(&node.payload);
            if let NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } = &node.payload
            {
                let definition = connections
                    .definitions
                    .get(instantiation.as_str())
                    .ok_or_else(|| {
                        invalid(format!(
                            "unknown instance '{instantiation}' in block '{}'",
                            block.name
                        ))
                    })?;
                // Follow the input sink node so a cycle can name the
                // connection. Its explicit operand already
                // points to the driving value.
                let connected = connections.inputs.get(instantiation.as_str());
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
                            let connection = connected
                                .and_then(|ports| ports.get(input.as_str()))
                                .ok_or_else(|| {
                                    invalid(format!(
                                        "unconnected input '{input}' on instance '{instantiation}'"
                                    ))
                                })?;
                            edges.push(*connection);
                        }
                    }
                    InstantiationKind::Extern => {
                        // The extern interface doesn't tell us which inputs can
                        // affect this output combinationally, so traversal
                        // stops here.
                    }
                }
                _ => { /* Other nodes have no implicit instance connections. */ }
            }
        }
        candidates.roots = feedback_roots(block, &dependencies)?;
        if !candidates.roots.is_empty() || !candidates.children.is_empty() {
            self.candidates.insert(block.name.clone(), candidates);
        }
        self.active.remove(&block.name);
        Ok(graph)
    }
}

/// Every whole-value cycle contains a DFS back edge. Start range refinement
/// at each distinct target, including feedback inside disconnected logic.
fn feedback_roots(
    block: &Block,
    dependencies: &[Vec<NodeRef>],
) -> Result<Vec<usize>, BuilderError> {
    let missing = |i| {
        invalid(format!(
            "block '{}' references missing node {i}",
            block.name
        ))
    };
    let mut roots = BTreeSet::new();
    crate::block_feedback::walk_dependencies(
        dependencies.len(),
        |i| &dependencies[i],
        missing,
        |path| {
            roots.insert(path[0].node);
            Ok(())
        },
        |_| {},
    )?;
    // An explicit operand cycle is invalid PIR even when some other path
    // between the same nodes crosses an instance. In a topologically stored
    // block every explicit edge already points to an earlier node.
    let explicit = |i: usize| -> &[NodeRef] {
        if matches!(
            block.nodes[i].payload,
            NodePayload::InstantiationOutput { .. }
        ) {
            &[]
        } else {
            &dependencies[i]
        }
    };
    if !roots.is_empty()
        && (0..dependencies.len()).any(|i| explicit(i).iter().any(|arg| arg.index >= i))
    {
        crate::block_feedback::walk_dependencies(
            dependencies.len(),
            explicit,
            missing,
            |path| {
                Err(cycle_error(
                    block,
                    &path.iter().map(|f| f.node).collect::<Vec<_>>(),
                ))
            },
            |_| {},
        )?;
    }
    Ok(roots.into_iter().collect())
}

/// Converts the cyclic segment of the DFS path into block-local text IDs.
fn cycle_error(block: &NodeGraph, path: &[usize]) -> BuilderError {
    let steps = path
        .iter()
        .zip(path.iter().cycle().skip(1))
        .map(|(&index, &next)| {
            let node = &block.nodes[index];
            let dependency = &block.nodes[next];
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

/// Collects the reachable input/output pairs without expanding any data bits.
/// Give the smaller interface the first pass. If many groups traverse shared
/// logic, let the opposite direction advance too, resuming at group boundaries.
fn graph_output_dependencies(block: &Block, dependencies: &[Vec<NodeRef>]) -> OutputDependencies {
    let inputs: Vec<_> = block.input_ports().collect();
    let outputs: Vec<_> = block.output_ports().collect();
    if inputs.is_empty() || outputs.is_empty() {
        return outputs
            .into_iter()
            .map(|port| (block.port_name(port).to_owned(), BTreeSet::new()))
            .collect();
    }
    // Index only the interface pairs we actually find. Build the named sets
    // once, after the traversal, without allocating names in its inner loop.
    let mut drivers = vec![Vec::new(); outputs.len()];
    let mut output_index = vec![usize::MAX; dependencies.len()];
    for (i, output) in outputs.iter().enumerate() {
        output_index[output.index] = i;
    }

    let mut from_inputs = inputs.len() < outputs.len();
    let mut input_groups = inputs.chunks(u64::BITS as usize);
    let mut output_groups = outputs.chunks(u64::BITS as usize);
    let mut users = None;
    let mut search = PortCones::new(dependencies.len());
    let quantum = dependencies.len() + dependencies.iter().map(Vec::len).sum::<usize>();
    while input_groups.len() != 0 && output_groups.len() != 0 {
        let (groups, edges) = if from_inputs {
            let edges = users.get_or_insert_with(|| {
                let mut users = vec![Vec::new(); dependencies.len()];
                for (index, args) in dependencies.iter().enumerate() {
                    for arg in args {
                        users[arg.index].push(NodeRef { index });
                    }
                }
                users
            });
            (&mut input_groups, edges.as_slice())
        } else {
            (&mut output_groups, dependencies)
        };
        search.advance(groups, edges, quantum, |dest, sources, ports| {
            if from_inputs {
                let row = output_index[dest.index];
                if row != usize::MAX {
                    drivers[row].extend(reached_ports(sources, ports));
                }
            } else if matches!(block.get_node(dest).payload, NodePayload::InputPort { .. }) {
                for output in reached_ports(sources, ports) {
                    drivers[output_index[output.index]].push(dest);
                }
            }
        });
        from_inputs = !from_inputs;
    }
    outputs
        .into_iter()
        .zip(drivers)
        .map(|(output, inputs)| {
            (
                block.port_name(output).to_owned(),
                inputs
                    .into_iter()
                    .map(|n| block.port_name(n).to_owned())
                    .collect(),
            )
        })
        .collect()
}

/// Enumerates the starting ports whose paths reached a common destination.
fn reached_ports(mut mask: u64, ports: &[NodeRef]) -> impl Iterator<Item = NodeRef> {
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let index = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        Some(ports[index])
    })
}

/// Each bit identifies one starting port in the current group. Propagating
/// newly reached ports together coalesces shared paths. Propagating each bit
/// at most once per edge also handles cycles in the whole-node graph.
struct PortCones {
    // Per node: all sources seen, and sources still to propagate.
    masks: Vec<(u64, u64)>,
    completed_ports: Vec<bool>,
    ready: VecDeque<NodeRef>,
    touched: Vec<NodeRef>,
}

impl PortCones {
    fn new(nodes: usize) -> Self {
        Self {
            masks: vec![(0, 0); nodes],
            completed_ports: vec![false; nodes],
            ready: VecDeque::new(),
            touched: Vec::new(),
        }
    }

    /// Switches at a group boundary once the work reaches a full graph scan.
    /// Only nodes reached by the group need their scratch masks cleared.
    fn advance(
        &mut self,
        groups: &mut std::slice::Chunks<'_, NodeRef>,
        edges: &[Vec<NodeRef>],
        mut budget: usize,
        mut visit: impl FnMut(NodeRef, u64, &[NodeRef]),
    ) {
        while let Some(ports) = groups.next() {
            for (bit, &node) in ports.iter().enumerate() {
                self.masks[node.index] = (1 << bit, 1 << bit);
                self.ready.push_back(node);
                self.touched.push(node);
            }
            while let Some(node) = self.ready.pop_front() {
                let delta = std::mem::take(&mut self.masks[node.index].1);
                budget = budget.saturating_sub(1 + edges[node.index].len());
                for &next in &edges[node.index] {
                    let (seen, pending) = &mut self.masks[next.index];
                    let new = delta & !*seen;
                    if new == 0 {
                        continue;
                    }
                    if *seen == 0 {
                        self.touched.push(next);
                    }
                    if *pending == 0 {
                        self.ready.push_back(next);
                    }
                    *seen |= new;
                    *pending |= new;
                }
            }
            for dest in self.touched.drain(..) {
                let (sources, _) = std::mem::take(&mut self.masks[dest.index]);
                // The opposite traversal has already emitted every pair for
                // a completed destination. It still participates in routing.
                if !self.completed_ports[dest.index] {
                    visit(dest, sources, ports);
                }
            }
            for port in ports {
                self.completed_ports[port.index] = true;
            }
            if budget == 0 {
                break;
            }
        }
    }
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
    fn whole_port_summary_preserves_independent_shared_cones() {
        // Each independent group spans more than one batch of ports.
        for (input_count, output_count) in [(65, 66), (66, 65)] {
            let mut child = BlockBuilder::new("child");
            let mut expected = OutputDependencies::new();
            let mut groups = Vec::new();
            let mut all_inputs = BTreeSet::new();
            for group in ["a", "b"] {
                let mut ports = Vec::new();
                let mut names = BTreeSet::new();
                for i in 0..input_count {
                    let name = format!("{group}_in_{i}");
                    ports.push(child.input_port(&name, Type::Bits(1)).unwrap());
                    names.insert(name);
                }
                let packed = child.concat(&ports).unwrap();
                let mut shared = child.xor_reduce(packed).unwrap();
                for _ in 0..128 {
                    shared = child.not(shared).unwrap();
                }
                for i in 0..output_count {
                    let name = format!("{group}_out_{i}");
                    child.output_port(&name, shared).unwrap();
                    expected.insert(name, names.clone());
                }
                groups.push(shared);
                all_inputs.extend(names);
            }
            let both = child.concat(&groups).unwrap();
            child.output_port("both", both).unwrap();
            expected.insert("both".into(), all_inputs);
            // This pair comes after both shared cones in either port order.
            let tail = child.input_port("tail", Type::Bits(1)).unwrap();
            child.output_port("last", tail).unwrap();
            expected.insert("last".into(), BTreeSet::from(["tail".into()]));

            let package = child.build_package("cones").unwrap();
            let block = package.get_top_block().unwrap();
            let edges: Vec<_> = block
                .nodes
                .iter()
                .map(|node| operands(&node.payload))
                .collect();
            assert_eq!(graph_output_dependencies(block, &edges), expected);
        }
    }

    #[test]
    fn child_feedback_uses_only_the_bound_control_field() {
        let source = r#"package bound_field
block shifter(data: bits[3], amount: bits[2], result: bits[3]) {
  data: bits[3] = input_port(name=data, id=1)
  amount: bits[2] = input_port(name=amount, id=2)
  shifted: bits[3] = shll(data, amount, id=3)
  result: () = output_port(shifted, name=result, id=4)
}
block feedback(config: bits[132], result: bits[3]) {
  instantiation shift(block=shifter, kind=block)
  config: bits[132] = input_port(name=config, id=5)
  offset: bits[2] = bit_slice(config, start=128, width=2, id=6)
  q: bits[3] = instantiation_output(instantiation=shift, port_name=result, id=7)
  drive_data: () = instantiation_input(q, instantiation=shift, port_name=data, id=8)
  drive_amount: () = instantiation_input(offset, instantiation=shift, port_name=amount, id=9)
  result: () = output_port(q, name=result, id=10)
}
top block configured(high: bits[2], low: bits[128], result: bits[3]) {
  instantiation inner(block=feedback, kind=block)
  high: bits[2] = input_port(name=high, id=11)
  low: bits[128] = input_port(name=low, id=12)
  amount: bits[2] = literal(value=1, id=13)
  config: bits[132] = concat(high, amount, low, id=14)
  drive: () = instantiation_input(config, instantiation=inner, port_name=config, id=15)
  q: bits[3] = instantiation_output(instantiation=inner, port_name=result, id=16)
  result: () = output_port(q, name=result, id=17)
}
"#;
        check(source).unwrap();
        for cyclic in [
            source.replace("value=1", "value=0"),
            source.replace("high, amount, low", "high, high, low"),
        ] {
            assert!(matches!(
                check(&cyclic),
                Err(BuilderError::CombinationalCycle { .. })
            ));
        }
    }

    #[test]
    fn one_feedback_cone_binds_all_independent_controls() {
        let source = r#"package split_cone
block combine(data: bits[3], a: bits[1], b: bits[1], q: bits[3]) {
  data: bits[3] = input_port(name=data, id=1)
  a: bits[1] = input_port(name=a, id=2)
  b: bits[1] = input_port(name=b, id=3)
  left: bits[3] = shll(data, a, id=4)
  right: bits[3] = shll(data, b, id=5)
  both: bits[3] = or(left, right, id=6)
  q: () = output_port(both, name=q, id=7)
}
top block root(q: bits[3]) {
  instantiation u(block=combine, kind=block)
  value: bits[3] = instantiation_output(instantiation=u, port_name=q, id=8)
  one: bits[1] = literal(value=1, id=9)
  zero: bits[1] = literal(value=0, id=10)
  left: () = instantiation_input(one, instantiation=u, port_name=a, id=11)
  right: () = instantiation_input(one, instantiation=u, port_name=b, id=12)
  feedback: () = instantiation_input(value, instantiation=u, port_name=data, id=13)
  q: () = output_port(value, name=q, id=14)
}"#;
        check(source).unwrap();
        for port in ["a", "b"] {
            let binding = format!("one, instantiation=u, port_name={port}");
            let cyclic = source.replace(&binding, &binding.replace("one", "zero"));
            assert!(matches!(
                check(&cyclic),
                Err(BuilderError::CombinationalCycle { .. })
            ));
        }
    }

    #[test]
    fn deep_local_calls_preserve_the_cycle_verdict_on_a_small_stack() {
        use std::fmt::Write;

        let depth = 2048;
        let mut source = String::from("package deep_calls\n");
        for level in 0..depth {
            let arg_id = 2 * level + 1;
            let expr = if level == 0 {
                format!("identity(arg, id={})", arg_id + 1)
            } else {
                format!("invoke(arg, to_apply=f{}, id={})", level - 1, arg_id + 1)
            };
            writeln!(
                source,
                "fn f{level}(arg: bits[3] id={arg_id}) -> bits[3] {{ ret y: bits[3] = {expr} }}"
            )
            .unwrap();
        }
        let last = depth - 1;
        writeln!(
            source,
            r#"block child(d: bits[3], q: bits[3]) {{
  d: bits[3] = input_port(name=d, id=5001)
  y: bits[3] = invoke(d, to_apply=f{last}, id=5002)
  q: () = output_port(y, name=q, id=5003)
}}
top block root(q: bits[3]) {{
  instantiation u(block=child, kind=block)
  value: bits[3] = instantiation_output(instantiation=u, port_name=q, id=5004)
  amount: bits[1] = literal(value=1, id=5005)
  shifted: bits[3] = shll(value, amount, id=5006)
  feedback: () = instantiation_input(shifted, instantiation=u, port_name=d, id=5007)
  q: () = output_port(value, name=q, id=5008)
}}"#
        )
        .unwrap();

        for (amount, valid) in [(1, true), (0, false)] {
            let source = source.replace("literal(value=1", &format!("literal(value={amount}"));
            let package = Parser::new(&source).parse_and_verify_package().unwrap();
            std::thread::Builder::new()
                .stack_size(512 * 1024)
                .spawn(move || {
                    let result = verify_block_combinational_cycles(
                        package.get_top_block().unwrap(),
                        &package,
                    );
                    if valid {
                        result.unwrap();
                    } else {
                        assert!(matches!(
                            result,
                            Err(BuilderError::CombinationalCycle { .. })
                        ));
                    }
                })
                .unwrap()
                .join()
                .unwrap();
        }
    }

    #[test]
    fn summaries_keep_shared_and_independent_output_cones_separate() {
        let source = r#"package shared_cones
top block child(x: bits[8], y: bits[8], z: bits[8], left: bits[8], right: bits[8], direct: bits[8], fixed: bits[8]) {
  x: bits[8] = input_port(name=x, id=1)
  y: bits[8] = input_port(name=y, id=2)
  z: bits[8] = input_port(name=z, id=3)
  shared: bits[8] = xor(x, y, id=4)
  diamond: bits[8] = and(shared, x, id=5)
  combined: bits[8] = xor(shared, z, id=6)
  left: () = output_port(diamond, name=left, id=7)
  right: () = output_port(combined, name=right, id=8)
  direct: () = output_port(z, name=direct, id=9)
  zero: bits[8] = literal(value=0, id=10)
  fixed: () = output_port(zero, name=fixed, id=11)
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
            ("fixed".into(), BTreeSet::new()),
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
    fn explicit_operand_cycle_remains_invalid_inside_a_refinable_component() {
        let source = r#"package mixed
block pass(d: bits[1], q: bits[1]) {
  d: bits[1] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block wrapper(seed: bits[1], result: bits[3]) {
  instantiation u(block=pass, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  word: bits[3] = zero_ext(seed, new_bit_count=3, id=4)
  q: bits[1] = instantiation_output(instantiation=u, port_name=q, id=5)
  lo: bits[1] = bit_slice(word, start=0, width=1, id=6)
  drive: () = instantiation_input(lo, instantiation=u, port_name=d, id=7)
  result: () = output_port(word, name=result, id=8)
}"#;
        let package = Parser::new(source).parse_and_verify_package().unwrap();
        let mut root = package.get_top_block().unwrap().clone();
        let word = crate::ir_utils::find_node_by_name(&root, "word").unwrap();
        let args = ["q", "lo", "seed"]
            .map(|name| crate::ir_utils::find_node_by_name(&root, name).unwrap());
        root.get_node_mut(word).payload =
            NodePayload::Nary(crate::ir::NaryOp::Concat, args.to_vec());
        // The low bit always comes from seed. The explicit word -> lo -> word
        // reference cycle still violates the PIR graph invariant.
        let Err(BuilderError::CombinationalCycle { steps, .. }) =
            verify_block_combinational_cycles(&root, &package)
        else {
            panic!("expected an explicit operand cycle");
        };
        assert_eq!(steps.iter().map(|s| s.node_id).collect::<Vec<_>>(), [4, 6]);
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
