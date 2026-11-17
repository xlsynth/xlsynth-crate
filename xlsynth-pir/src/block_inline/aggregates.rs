// SPDX-License-Identifier: Apache-2.0

//! Materializes aggregate reads that expose independent elements of a cycle.

use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use super::PendingNodes;
use crate::IrValue;
use crate::block_feedback::controls::ControlValues;
use crate::block_feedback::projection::{
    AggregateProjection, aggregate_selection, element_type, project_aggregate,
};
use crate::ir::{Block, NaryOp, NodeGraph, NodePayload, NodeRef, Package, Type, Unop};
use crate::ir_value_utils::zero_ir_value_for_type;

/// Whole-value operations that can hide independent aggregate elements.
pub(super) fn can_project(payload: &NodePayload) -> bool {
    matches!(
        payload,
        NodePayload::ArrayConcat(_)
            | NodePayload::ArraySlice { .. }
            | NodePayload::ArrayUpdate { .. }
            | NodePayload::ArrayIndex { .. }
            | NodePayload::TupleIndex { .. }
            | NodePayload::OneHotSel { .. }
    )
}

/// Rebuilds only selected values still in a cycle after direct references and
/// bit ranges have been refined. All reads use the same immutable graph.
pub(super) fn project(
    block: &mut Block,
    package: &Package,
    cyclic: &[bool],
    max_id: &mut usize,
) -> Result<bool, String> {
    let mut offsets = ControlValues::new(package);
    offsets.register_block(block);
    let mut reads = HashMap::new();
    // Reuse reads inside the component. Querying indices elsewhere could
    // evaluate an unrelated consumer's control logic.
    for (index, node) in block.nodes.iter().enumerate() {
        if !cyclic[index] {
            continue;
        }
        let node_ref = NodeRef { index };
        if let NodePayload::ArrayIndex { indices, .. } = &node.payload
            && indices.iter().any(|i| cyclic[i.index])
        {
            // A replacement can remove feedback through a static index driver.
            continue;
        }
        if let Some((source, path)) =
            aggregate_selection(block, node_ref, |arg| offsets.root_value(block, arg))
        {
            reads
                .entry(Field { node: source, path })
                .or_insert(node_ref);
        }
    }
    let mut planner = Planner {
        graph: block,
        cyclic,
        offsets,
        reads,
        resolved: HashMap::new(),
        literals: HashMap::new(),
        pending: PendingNodes::new(*max_id),
    };
    let mut replacements = Vec::new();
    for (index, &cyclic) in cyclic.iter().enumerate() {
        if !cyclic || !can_project(&block.nodes[index].payload) {
            continue;
        }
        let root = Field {
            node: NodeRef { index },
            path: SmallVec::new(),
        };
        let result = planner.resolve(root)?;
        if result.index != index {
            replacements.push((index, result));
        }
    }
    if replacements.is_empty() {
        return Ok(false);
    }
    *max_id = planner.pending.last_id;
    let added = planner.pending.nodes;
    block.nodes.extend(added);
    for (index, arg) in replacements {
        block.nodes[index].replace_payload(NodePayload::Unop(Unop::Identity, arg));
    }
    Ok(true)
}

/// One aggregate element, optionally an entire array or tuple.
#[derive(Clone, Eq, PartialEq, Hash)]
struct Field {
    node: NodeRef,
    path: SmallVec<[usize; 2]>,
}

impl Field {
    /// Selects another level of a nested array or tuple.
    fn child(&self, index: usize) -> Self {
        let mut child = self.clone();
        child.path.push(index);
        child
    }
}

/// Postorder work for a field, its forwarded source, or its constituent
/// elements.
enum Work {
    Read(Field),
    Forward(Field, Field),
    Elements(Field, usize),
    Or(Field, Vec<Field>),
}

/// Caches the expression materialized for each selected element of a node.
struct Planner<'a> {
    graph: &'a NodeGraph,
    cyclic: &'a [bool],
    offsets: ControlValues<'a>,
    /// Existing and newly built primitive reads from otherwise opaque values.
    reads: HashMap<Field, NodeRef>,
    resolved: HashMap<Field, NodeRef>,
    literals: HashMap<usize, NodeRef>,
    pending: PendingNodes,
}

impl Planner<'_> {
    /// Resolves dependencies before their users. A repeated read retains its
    /// original graph dependency for subsequent bit or call refinement.
    fn resolve(&mut self, root: Field) -> Result<NodeRef, String> {
        let mut stack = vec![Work::Read(root.clone())];
        let mut active = HashSet::new();
        while let Some(work) = stack.pop() {
            let (field, result) = match work {
                Work::Read(field) => {
                    if self.resolved.contains_key(&field) {
                        continue;
                    }
                    if self.cyclic[field.node.index] && active.insert(field.clone()) {
                        let step = project_aggregate(self.graph, field.node, &field.path, |arg| {
                            self.offsets.root_value(self.graph, arg)
                        });
                        match step {
                            Some(AggregateProjection::Forward { node, path }) => {
                                let source = Field { node, path };
                                stack.push(Work::Forward(field, source.clone()));
                                stack.push(Work::Read(source));
                                continue;
                            }
                            Some(AggregateProjection::Or(arms)) => {
                                let ty =
                                    element_type(self.graph.get_node_ty(field.node), &field.path)
                                        .expect("verified select field");
                                if arms.is_empty() {
                                    let result = self.pending.push(
                                        self.graph,
                                        field.node,
                                        ty.clone(),
                                        NodePayload::Literal(zero_ir_value_for_type(ty)),
                                    )?;
                                    active.remove(&field);
                                    self.resolved.insert(field, result);
                                    continue;
                                }
                                // Aggregate one-hot select ORs corresponding
                                // leaves. Splitting here keeps independent
                                // fields independent.
                                let count = match ty {
                                    Type::Bits(_) => {
                                        let sources: Vec<_> = arms
                                            .into_iter()
                                            .map(|node| Field {
                                                node,
                                                path: field.path.clone(),
                                            })
                                            .collect();
                                        stack.push(Work::Or(field, sources.clone()));
                                        stack.extend(sources.into_iter().rev().map(Work::Read));
                                        continue;
                                    }
                                    Type::Array(dim) => dim.element_count,
                                    Type::Tuple(fields) => fields.len(),
                                    Type::Token => {
                                        unreachable!("one-hot selects cannot contain tokens")
                                    }
                                };
                                stack.push(Work::Elements(field.clone(), count));
                                stack.extend((0..count).rev().map(|i| Work::Read(field.child(i))));
                                continue;
                            }
                            Some(AggregateProjection::Elements(count)) => {
                                stack.push(Work::Elements(field.clone(), count));
                                stack.extend((0..count).rev().map(|i| Work::Read(field.child(i))));
                                continue;
                            }
                            None => {} // The original node supplies this value.
                        }
                    }
                    let result = self.read(&field)?;
                    (field, result)
                }
                Work::Forward(field, source) => (field, self.resolved[&source]),
                Work::Elements(field, count) => {
                    let elements = (0..count).map(|i| self.resolved[&field.child(i)]).collect();
                    let ty = element_type(self.graph.get_node_ty(field.node), &field.path)
                        .expect("valid aggregate selection");
                    let payload = match ty {
                        Type::Tuple(_) => NodePayload::Tuple(elements),
                        Type::Array(_) => NodePayload::Array(elements),
                        _ => unreachable!("only aggregates split into elements"),
                    };
                    let result = self
                        .pending
                        .push(self.graph, field.node, ty.clone(), payload)?;
                    (field, result)
                }
                Work::Or(field, sources) => {
                    let args = sources.iter().map(|arg| self.resolved[arg]).collect();
                    let ty = element_type(self.graph.get_node_ty(field.node), &field.path)
                        .expect("verified select field");
                    let result = self.pending.push(
                        self.graph,
                        field.node,
                        ty.clone(),
                        NodePayload::Nary(NaryOp::Or, args),
                    )?;
                    (field, result)
                }
            };
            active.remove(&field);
            self.resolved.insert(field, result);
        }
        Ok(self.resolved[&root])
    }

    /// Emits in-bounds reads from a source whose implementation is opaque to
    /// routing. Reuses nodes already present in the graph when possible.
    fn read(&mut self, field: &Field) -> Result<NodeRef, String> {
        if field.path.is_empty() {
            return Ok(field.node);
        }
        if let Some(&node) = self.reads.get(field) {
            return Ok(node);
        }
        let mut key = Field {
            node: field.node,
            path: SmallVec::new(),
        };
        let mut node = field.node;
        let mut ty = self.graph.get_node_ty(node);
        for &index in &field.path {
            key.path.push(index);
            let next_ty = element_type(ty, &[index]).expect("valid aggregate selection");
            if let Some(&existing) = self.reads.get(&key) {
                node = existing;
            } else {
                let payload = match ty {
                    Type::Tuple(_) => NodePayload::TupleIndex { tuple: node, index },
                    Type::Array(_) => {
                        let literal = if let Some(&literal) = self.literals.get(&index) {
                            literal
                        } else {
                            let width = (usize::BITS - index.leading_zeros()).max(1) as usize;
                            let value = IrValue::make_ubits(width, index as u64)
                                .map_err(|e| e.to_string())?;
                            let literal = self.pending.push(
                                self.graph,
                                field.node,
                                Type::Bits(width),
                                NodePayload::Literal(value),
                            )?;
                            self.literals.insert(index, literal);
                            literal
                        };
                        NodePayload::ArrayIndex {
                            array: node,
                            indices: vec![literal],
                            assumed_in_bounds: false,
                        }
                    }
                    _ => unreachable!("aggregate index applied to a scalar"),
                };
                node = self
                    .pending
                    .push(self.graph, field.node, next_ty.clone(), payload)?;
                self.reads.insert(key.clone(), node);
            }
            ty = next_ty;
        }
        Ok(node)
    }
}
