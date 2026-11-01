// SPDX-License-Identifier: Apache-2.0

//! Register-delimited stage reconstruction for feed-forward block layouts.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, VecDeque};

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
pub(crate) struct OutputTiming<'a> {
    minimum_stage: isize,
    input_delays: BTreeMap<&'a str, isize>,
}

/// Maps each previously emitted child block to the timing of its output ports.
pub(crate) type ChildTimings<'a> = BTreeMap<&'a str, BTreeMap<&'a str, OutputTiming<'a>>>;

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
                let timing = &child_timings[instance.block.as_str()][port_name];
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
) -> BTreeMap<&'a str, OutputTiming<'a>> {
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

    let mut outputs = BTreeMap::new();
    let mut visited_from = vec![None; graph.len()];
    let mut pending = BinaryHeap::new();
    for port in block.output_ports() {
        let output = block.output_value(port).index;
        let minimum_stage = earliest[output];
        let mut input_delays = BTreeMap::new();
        // Across any path, latency is the difference in earliest stages minus
        // the sum of its nonnegative edge slacks. Visiting least slack first
        // gives the longest path to each input in one traversal of the cone.
        pending.push(Reverse((0, output)));
        while let Some(Reverse((slack, index))) = pending.pop() {
            if visited_from[index] == Some(port) {
                continue;
            }
            visited_from[index] = Some(port);
            if let NodePayload::InputPort { name, .. } = &block.nodes[index].payload {
                input_delays.insert(name.as_str(), minimum_stage - earliest[index] - slack);
            }
            for &(source, edge_slack) in &predecessors[index] {
                pending.push(Reverse((slack + edge_slack, source)));
            }
        }
        outputs.insert(
            block.port_name(port),
            OutputTiming {
                minimum_stage,
                input_delays,
            },
        );
    }
    outputs
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
