// SPDX-License-Identifier: Apache-2.0

//! Register-delimited stage reconstruction for feed-forward block layouts.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, VecDeque};

use xlsynth_pir::ir::{Block, InstantiationKind, NodeGraph, NodePayload, NodeRef};
use xlsynth_pir::ir_utils::{get_topological, operands};

use crate::BlockCodegenError;

/// Groups the register reads, combinational nodes, and writes in one stage.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Stage {
    pub(crate) register_reads: Vec<NodeRef>,
    pub(crate) combinational_nodes: Vec<NodeRef>,
    pub(crate) register_writes: Vec<NodeRef>,
}

/// Describes a child output's earliest stage and the register depth from each
/// input that reaches it.
struct OutputTiming<'a> {
    minimum_stage: isize,
    input_delays: BTreeMap<&'a str, isize>,
}

/// Shares one timing summary among output ports driven by the same value.
pub(crate) struct BlockTiming<'a> {
    ports: BTreeMap<&'a str, NodeRef>,
    values: HashMap<NodeRef, OutputTiming<'a>>,
}

/// Maps each previously emitted child block to the timing of its output ports.
pub(crate) type ChildTimings<'a> = BTreeMap<&'a str, BlockTiming<'a>>;

/// Reconstructs one block's stages in child-before-parent order.
pub(crate) fn reconstruct_stages<'a>(
    block: &'a Block,
    child_timings: &mut ChildTimings<'a>,
    summarize_for_parent: bool,
) -> Result<Vec<Stage>, BlockCodegenError> {
    let node_count = block.nodes.len();
    let mut users: Vec<Vec<(usize, isize)>> = vec![Vec::new(); node_count];
    let mut stages = vec![0isize; node_count];
    let mut register_reads: BTreeMap<&str, usize> = BTreeMap::new();
    let mut instantiation_inputs: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut instantiation_outputs: BTreeMap<&str, Vec<(usize, &str)>> = BTreeMap::new();

    for (index, node) in block.nodes.iter().enumerate() {
        match &node.payload {
            NodePayload::RegisterRead { register } => {
                register_reads.insert(register.as_str(), index);
            }
            NodePayload::InstantiationInput {
                instantiation,
                port_name,
                ..
            } => {
                instantiation_inputs
                    .entry(instantiation.as_str())
                    .or_default()
                    .insert(port_name.as_str(), index);
            }
            NodePayload::InstantiationOutput {
                instantiation,
                port_name,
            } => {
                instantiation_outputs
                    .entry(instantiation.as_str())
                    .or_default()
                    .push((index, port_name));
            }
            _ => {
                // Ordinary operations contribute only their explicit operands.
            }
        }
        for operand in operands(&node.payload) {
            users[operand.index].push((index, 0));
        }
    }

    for instance in &block.instantiations {
        let Some(outputs) = instantiation_outputs.get(instance.name.as_str()) else {
            continue;
        };
        let inputs = instantiation_inputs.get(instance.name.as_str());
        for &(output, port_name) in outputs {
            if instance.kind == InstantiationKind::Block {
                let child = &child_timings[instance.block.as_str()];
                let timing = &child.values[&child.ports[port_name]];
                stages[output] = timing.minimum_stage;
                for (&name, &delay) in &timing.input_delays {
                    let input = inputs.expect("verified child inputs are connected")[name];
                    users[input].push((output, delay));
                }
            } else if let Some(inputs) = inputs {
                // The implementation of an external function is opaque.
                for &input in inputs.values() {
                    users[input].push((output, 0));
                }
            }
        }
    }

    // Registers impose exact stage differences in addition to the minimum
    // stage differences imposed by consumers.
    let mut graph = users.clone();
    for (index, node) in block.nodes.iter().enumerate() {
        if let NodePayload::RegisterWrite { register, .. } = &node.payload {
            let Some(&read_index) = register_reads.get(register.as_str()) else {
                return Err(BlockCodegenError::InvalidBlock(format!(
                    "register `{register}` in block `{}` has no register_read",
                    block.name
                )));
            };
            graph[index].push((read_index, 1));
            graph[read_index].push((index, -1));
        }
    }

    let maximum_edge = graph
        .iter()
        .flat_map(|edges| edges.iter().map(|(_, distance)| *distance))
        .max()
        .unwrap_or(1)
        .max(1) as usize;
    let mut max_stage = stages.iter().copied().max().unwrap_or(0) as usize;
    let maximum_valid_stage = max_stage.saturating_add(node_count.saturating_mul(maximum_edge));
    let mut worklist: VecDeque<usize> = (0..node_count).collect();
    while let Some(source) = worklist.pop_front() {
        if stages[source] >= maximum_valid_stage as isize {
            return Err(BlockCodegenError::NotPipeline(format!(
                "block `{}` cannot use layout=pipeline: register feedback or uneven \
                 register layering prevents feed-forward stage reconstruction; use layout=none",
                block.name
            )));
        }
        for &(target, distance) in &graph[source] {
            let minimum = stages[source] + distance;
            if stages[target] < minimum {
                stages[target] = minimum;
                max_stage = max_stage.max(minimum as usize);
                worklist.push_back(target);
            }
        }
    }

    // Summarize the earliest stages, before moving combinational operations
    // towards their users for emission.
    if summarize_for_parent {
        child_timings.insert(
            block.name.as_str(),
            summarize_outputs(block, &graph, &stages),
        );
    }
    let topological = get_topological(block);
    let constant_only = constant_only_nodes(block, &topological);
    for node_ref in topological.iter().rev().copied() {
        if users[node_ref.index].is_empty()
            || matches!(
                block.get_node(node_ref).payload,
                NodePayload::RegisterRead { .. } | NodePayload::RegisterWrite { .. }
            )
        {
            continue;
        }
        stages[node_ref.index] = users[node_ref.index]
            .iter()
            .map(|&(user, distance)| stages[user] - distance)
            .min()
            .expect("nonempty user set");
    }

    let node_stages = stages
        .iter()
        .map(|stage| *stage as usize)
        .collect::<Vec<_>>();
    let mut result = vec![Stage::default(); max_stage + 1];
    for node_ref in topological {
        let stage_index = node_stages[node_ref.index];
        let stage = &mut result[stage_index];
        match &block.get_node(node_ref).payload {
            NodePayload::Nil | NodePayload::InputPort { .. } | NodePayload::OutputPort { .. } => {
                // Inputs are represented by module ports, outside stage
                // sections.
            }
            NodePayload::RegisterRead { .. } => stage.register_reads.push(node_ref),
            NodePayload::RegisterWrite { arg, register, .. } => {
                let read_index = register_reads[register.as_str()];
                // Constant-only data may be shared by consumers in different
                // stages; its emission location does not give it a cycle age.
                if node_stages[read_index] != stage_index + 1
                    || node_stages[arg.index] > stage_index
                    || (node_stages[arg.index] < stage_index && !constant_only[arg.index])
                {
                    return Err(BlockCodegenError::NotPipeline(format!(
                        "block `{}` cannot use layout=pipeline: register `{register}` \
                         crosses incompatible pipeline stages; use layout=none",
                        block.name
                    )));
                }
                stage.register_writes.push(node_ref);
            }
            _ => stage.combinational_nodes.push(node_ref),
        }
    }

    Ok(result)
}

/// Finds the longest forward register path from each reachable input to each
/// output, including paths through nested instances.
fn summarize_outputs<'a>(
    block: &'a Block,
    graph: &[Vec<(usize, isize)>],
    earliest: &[isize],
) -> BlockTiming<'a> {
    let mut predecessors = vec![Vec::new(); graph.len()];
    for (source, edges) in graph.iter().enumerate() {
        for &(target, delay) in edges {
            if delay >= 0 {
                // Reverse register constraints align stages but carry no data.
                let slack = earliest[target] - earliest[source] - delay;
                predecessors[target].push((source, slack));
            }
        }
    }

    let ports: BTreeMap<_, _> = block
        .output_ports()
        .map(|port| (block.port_name(port), block.output_value(port)))
        .collect();
    let mut timings: HashMap<_, _> = ports
        .values()
        .map(|&value| {
            let timing = OutputTiming {
                minimum_stage: earliest[value.index],
                input_delays: BTreeMap::new(),
            };
            (value, timing)
        })
        .collect();
    let output_values: Vec<_> = timings.keys().copied().collect();

    // Restrict forward traversals to the union of output cones, including when
    // an input has many unrelated users. Only inputs in these cones matter.
    let mut reaches_output = vec![false; graph.len()];
    let mut pending: Vec<_> = output_values.iter().map(|value| value.index).collect();
    while let Some(node) = pending.pop() {
        if reaches_output[node] {
            continue;
        }
        reaches_output[node] = true;
        pending.extend(predecessors[node].iter().map(|&(source, _)| source));
    }
    let inputs: Vec<_> = block
        .input_ports()
        .filter(|input| reaches_output[input.index])
        .collect();

    // Across any path, latency equals the difference in earliest stages minus
    // the sum of its edge slacks. Minimize slack from the smaller set of roots.
    let record = |timing: &mut OutputTiming<'a>, input: NodeRef, slack: isize| {
        let delay = timing.minimum_stage - earliest[input.index] - slack;
        timing.input_delays.insert(block.port_name(input), delay);
    };
    if inputs.len() < output_values.len() {
        let mut successors = vec![Vec::new(); graph.len()];
        for (target, edges) in predecessors.into_iter().enumerate() {
            if reaches_output[target] {
                for (source, slack) in edges {
                    successors[source].push((target, slack));
                }
            }
        }
        visit_least_slack_paths(&successors, &inputs, |input, node, slack| {
            if let Some(timing) = timings.get_mut(&node) {
                record(timing, input, slack);
            }
        });
    } else {
        visit_least_slack_paths(&predecessors, &output_values, |output, node, slack| {
            if matches!(block.get_node(node).payload, NodePayload::InputPort { .. }) {
                record(timings.get_mut(&output).expect("output root"), node, slack);
            }
        });
    }

    BlockTiming {
        ports,
        values: timings,
    }
}

/// Visits each reachable node once per root, in order of minimum total slack.
fn visit_least_slack_paths(
    edges: &[Vec<(usize, isize)>],
    roots: &[NodeRef],
    mut visit: impl FnMut(NodeRef, NodeRef, isize),
) {
    let mut visited_from = vec![None; edges.len()];
    let mut pending = BinaryHeap::new();
    for &root in roots {
        pending.push(Reverse((0, root.index)));
        while let Some(Reverse((slack, index))) = pending.pop() {
            if visited_from[index] == Some(root) {
                continue;
            }
            visited_from[index] = Some(root);
            visit(root, NodeRef { index }, slack);
            for &(neighbor, edge_slack) in &edges[index] {
                pending.push(Reverse((slack + edge_slack, neighbor)));
            }
        }
    }
}

/// Recognizes pure constant cones without evaluating or rewriting the graph.
fn constant_only_nodes(graph: &NodeGraph, topological: &[NodeRef]) -> Vec<bool> {
    let mut constant_only = vec![false; graph.nodes.len()];
    for &node_ref in topological {
        let payload = &graph.get_node(node_ref).payload;
        constant_only[node_ref.index] = match payload {
            NodePayload::Literal(_)
            | NodePayload::Tuple(_)
            | NodePayload::Array(_)
            | NodePayload::ArrayConcat(_)
            | NodePayload::ArraySlice { .. }
            | NodePayload::TupleIndex { .. }
            | NodePayload::Binop(..)
            | NodePayload::Unop(..)
            | NodePayload::SignExt { .. }
            | NodePayload::ZeroExt { .. }
            | NodePayload::ArrayUpdate { .. }
            | NodePayload::ArrayIndex { .. }
            | NodePayload::DynamicBitSlice { .. }
            | NodePayload::BitSlice { .. }
            | NodePayload::BitSliceUpdate { .. }
            | NodePayload::ExtCarryOut { .. }
            | NodePayload::ExtPrioEncode { .. }
            | NodePayload::ExtClz { .. }
            | NodePayload::ExtNormalizeLeft { .. }
            | NodePayload::ExtMaskLow { .. }
            | NodePayload::ExtNaryAdd { .. }
            | NodePayload::Nary(..)
            | NodePayload::PrioritySel { .. }
            | NodePayload::OneHotSel { .. }
            | NodePayload::OneHot { .. }
            | NodePayload::Sel { .. }
            | NodePayload::Decode { .. }
            | NodePayload::Encode { .. } => operands(payload)
                .iter()
                .all(|operand| constant_only[operand.index]),
            NodePayload::Nil
            | NodePayload::Param
            | NodePayload::InputPort { .. }
            | NodePayload::OutputPort { .. }
            | NodePayload::RegisterRead { .. }
            | NodePayload::RegisterWrite { .. }
            | NodePayload::InstantiationInput { .. }
            | NodePayload::InstantiationOutput { .. }
            | NodePayload::Assert { .. }
            | NodePayload::Trace { .. }
            | NodePayload::Cover { .. }
            | NodePayload::AfterAll(_)
            | NodePayload::Invoke { .. }
            | NodePayload::Map { .. }
            | NodePayload::CountedFor { .. } => false,
        };
    }
    constant_only
}
