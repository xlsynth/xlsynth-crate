// SPDX-License-Identifier: Apache-2.0

//! Register-delimited stage reconstruction for feed-forward block layouts.

use std::collections::{BTreeMap, VecDeque};

use xlsynth_pir::ir::{
    BlockMetadata, Fn, InstantiationKind, NodePayload, NodeRef, Package, PackageMember,
};
use xlsynth_pir::ir_utils::{get_topological, operands};

use crate::BlockCodegenError;

/// Groups the register reads, combinational nodes, and writes in one stage.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Stage {
    pub(crate) register_reads: Vec<NodeRef>,
    pub(crate) combinational_nodes: Vec<NodeRef>,
    pub(crate) register_writes: Vec<NodeRef>,
}

/// Assigns every representable node to its reconstructed pipeline stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StageLayout {
    pub(crate) stages: Vec<Stage>,
    pub(crate) node_stages: Vec<usize>,
}

/// Reconstructs feed-forward stages without changing the block's registers.
pub(crate) fn reconstruct_stages(
    package: &Package,
    func: &Fn,
    metadata: &BlockMetadata,
) -> Result<StageLayout, BlockCodegenError> {
    if func.nodes.is_empty() {
        return Ok(StageLayout {
            stages: vec![Stage::default()],
            node_stages: Vec::new(),
        });
    }

    let node_count = func.nodes.len();
    let synthetic_output_tuple = if metadata.output_names.len() != 1 {
        func.ret_node_ref
    } else {
        None
    };
    let mut graph: Vec<Vec<(usize, isize)>> = vec![Vec::new(); node_count];
    let mut users: Vec<Vec<(usize, isize)>> = vec![Vec::new(); node_count];
    let mut register_reads: BTreeMap<&str, usize> = BTreeMap::new();
    let mut instantiation_inputs: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let mut instantiation_outputs: BTreeMap<&str, Vec<(usize, &str)>> = BTreeMap::new();

    for (index, node) in func.nodes.iter().enumerate() {
        if synthetic_output_tuple == Some(NodeRef { index }) {
            continue;
        }
        match &node.payload {
            NodePayload::RegisterRead { register } => {
                register_reads.insert(register.as_str(), index);
            }
            NodePayload::InstantiationInput { instantiation, .. } => {
                instantiation_inputs
                    .entry(instantiation.as_str())
                    .or_default()
                    .push(index);
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
            graph[operand.index].push((index, 0));
            users[operand.index].push((index, 0));
        }
    }

    for instance in &metadata.instantiations {
        let Some(inputs) = instantiation_inputs.get(instance.name.as_str()) else {
            continue;
        };
        let Some(outputs) = instantiation_outputs.get(instance.name.as_str()) else {
            continue;
        };
        let output_delays = if instance.kind == InstantiationKind::Block {
            let (child, child_metadata) = package
                .members
                .iter()
                .find_map(|member| match member {
                    PackageMember::Block { func, metadata } if func.name == instance.block => {
                        Some((func, metadata))
                    }
                    _ => None,
                })
                .ok_or_else(|| {
                    BlockCodegenError::InvalidBlock(format!(
                        "instantiated block `{}` does not exist in the package",
                        instance.block
                    ))
                })?;
            let child_layout = reconstruct_stages(package, child, child_metadata)?;
            let output_nodes = if child_metadata.output_names.len() == 1 {
                child.ret_node_ref.into_iter().collect::<Vec<_>>()
            } else if let Some(ret) = child.ret_node_ref {
                match &child.get_node(ret).payload {
                    NodePayload::Tuple(nodes) => nodes.clone(),
                    _ => Vec::new(),
                }
            } else {
                Vec::new()
            };
            child_metadata
                .output_names
                .iter()
                .zip(output_nodes)
                .map(|(name, node)| (name.as_str(), child_layout.node_stages[node.index]))
                .collect::<BTreeMap<_, _>>()
        } else {
            BTreeMap::new()
        };
        for &(output, port_name) in outputs {
            let delay = output_delays.get(port_name).copied().unwrap_or(0) as isize;
            for &input in inputs {
                graph[input].push((output, delay));
                users[input].push((output, delay));
            }
        }
    }

    for (index, node) in func.nodes.iter().enumerate() {
        if let NodePayload::RegisterWrite { register, .. } = &node.payload {
            let Some(&read_index) = register_reads.get(register.as_str()) else {
                return Err(BlockCodegenError::InvalidBlock(format!(
                    "register `{register}` in block `{}` has no register_read",
                    func.name
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
    let maximum_valid_stage = node_count.saturating_mul(maximum_edge);
    let mut stages = vec![0isize; node_count];
    let mut worklist: VecDeque<usize> = (0..node_count).collect();
    let mut max_stage = 0usize;
    while let Some(source) = worklist.pop_front() {
        if stages[source] >= maximum_valid_stage as isize {
            return Err(BlockCodegenError::NotPipeline(format!(
                "block `{}` cannot use layout=pipeline: register feedback or uneven \
                 register layering prevents feed-forward stage reconstruction; use layout=none",
                func.name
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

    let topological = get_topological(func);
    let constant_only = constant_only_nodes(func, &topological);
    for node_ref in topological.iter().rev().copied() {
        if synthetic_output_tuple == Some(node_ref)
            || users[node_ref.index].is_empty()
            || matches!(
                func.get_node(node_ref).payload,
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
        if synthetic_output_tuple == Some(node_ref) {
            continue;
        }
        let stage_index = node_stages[node_ref.index];
        let stage = &mut result[stage_index];
        match &func.get_node(node_ref).payload {
            NodePayload::Nil | NodePayload::GetParam(_) => {
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
                        func.name
                    )));
                }
                stage.register_writes.push(node_ref);
            }
            _ => stage.combinational_nodes.push(node_ref),
        }
    }

    Ok(StageLayout {
        stages: result,
        node_stages,
    })
}

/// Recognizes pure constant cones without evaluating or rewriting the graph.
fn constant_only_nodes(func: &Fn, topological: &[NodeRef]) -> Vec<bool> {
    let mut constant_only = vec![false; func.nodes.len()];
    for &node_ref in topological {
        let payload = &func.get_node(node_ref).payload;
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
            | NodePayload::GetParam(_)
            | NodePayload::RegisterRead { .. }
            | NodePayload::RegisterWrite { .. }
            | NodePayload::InstantiationInput { .. }
            | NodePayload::InstantiationOutput { .. }
            | NodePayload::Assert { .. }
            | NodePayload::Trace { .. }
            | NodePayload::Cover { .. }
            | NodePayload::AfterAll(_)
            | NodePayload::Invoke { .. }
            | NodePayload::CountedFor { .. } => false,
        };
    }
    constant_only
}
