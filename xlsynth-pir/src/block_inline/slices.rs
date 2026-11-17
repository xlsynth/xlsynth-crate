// SPDX-License-Identifier: Apache-2.0

//! Materialize static slices when flattening turns an instance connection into
//! a circular reference between whole values.

use std::collections::{HashMap, HashSet};

use super::PendingNodes;
use crate::block_feedback::controls::ControlValues;
use crate::block_feedback::projection::{
    BitProjection, BitRange, is_empty_data, selected_aggregate_operand,
};
use crate::ir::{Block, Node, NodePayload, NodeRef, Package, Type, Unop, node_textual_id};
use crate::ir_utils::{operands, remap_payload_with};

/// Refines native wiring before opening a local call. A call may only consume
/// feedback even when its output appears in the same whole-value component.
pub(super) fn resolve_slices(
    block: &mut Block,
    package: &Package,
    calls: &mut crate::block_invoke_inline::FeedbackInvokes,
    mut max_id: usize,
) -> Result<bool, String> {
    let mut changed = false;
    loop {
        let mut cyclic = cyclic_nodes(block)?;
        if cut_empty_operands(block, &cyclic, &mut max_id)? {
            changed = true;
            cyclic = cyclic_nodes(block)?;
        }
        let has_calls = cyclic.iter().enumerate().any(|(i, &cycle)| {
            cycle && matches!(block.nodes[i].payload, NodePayload::Invoke { .. })
        });
        let has_aggregates = cyclic
            .iter()
            .enumerate()
            .any(|(i, &cycle)| cycle && super::aggregates::can_project(&block.nodes[i].payload));
        if project_ranges(block, package, &cyclic, &mut max_id)? {
            changed = true;
            if has_calls || has_aggregates {
                cyclic = cyclic_nodes(block)?;
            }
        }
        if has_aggregates && super::aggregates::project(block, package, &cyclic, &mut max_id)? {
            changed = true;
            continue;
        }
        if !has_calls || !calls.expand(block, &cyclic, &mut max_id)? {
            return Ok(changed);
        }
        changed = true;
    }
}

/// Finds feedback in the current wiring snapshot. Rewriting a range or opening
/// a call can remove edges and change membership of its entire component.
fn cyclic_nodes(block: &Block) -> Result<Vec<bool>, String> {
    let dependencies = block
        .nodes
        .iter()
        .map(|node| operands(&node.payload))
        .collect::<Vec<_>>();
    crate::block_feedback::cyclic_nodes(&dependencies)
}

/// Plans exact expressions against one immutable graph and installs them
/// together, preserving the original named nodes and their users.
fn project_ranges(
    block: &mut Block,
    package: &Package,
    cyclic: &[bool],
    max_id: &mut usize,
) -> Result<bool, String> {
    if !cyclic.iter().any(|&member| member) {
        return Ok(false);
    }
    let mut offsets = ControlValues::new(package);
    offsets.register_block(block);
    let mut resolver = Resolver {
        cyclic,
        resolved: HashMap::new(),
        pending: PendingNodes::new(*max_id),
    };
    let mut replacements = Vec::new();
    for index in 0..block.nodes.len() {
        if !resolver.cyclic[index] {
            continue;
        }
        let node = NodeRef { index };
        if let Some(selected) =
            selected_aggregate_operand(block, node, |control| offsets.root_value(block, control))
        {
            replacements.push((index, NodePayload::Unop(Unop::Identity, selected)));
            continue;
        }
        let Type::Bits(width) = *block.get_node_ty(node) else {
            continue;
        };
        if width == 0 {
            continue;
        }
        // An operation can select lanes implicitly, even when the two ports
        // in the feedback path are connected at their full width.
        let whole = BitRange {
            node,
            lo: 0,
            hi: width,
        };
        if let Some(projection) = whole.project(block, |control| offsets.root_value(block, control))
        {
            for &part in projection.inputs() {
                resolver.resolve(block, part, &mut offsets)?;
            }
            replacements.push((index, resolver.payload(block, whole, projection)?));
        }
    }
    let changed = !replacements.is_empty();
    *max_id = resolver.pending.last_id;
    block.nodes.extend(resolver.pending.nodes);
    for (index, payload) in replacements {
        // Preserve any name, source position, and users of the original node.
        block.nodes[index].replace_payload(payload);
    }
    Ok(changed)
}

/// A nonempty port can still depend on an empty intermediate value. Redirect
/// uses of that value while retaining producers such as calls and block sinks.
fn cut_empty_operands(
    block: &mut Block,
    cyclic: &[bool],
    max_id: &mut usize,
) -> Result<bool, String> {
    let mut literals = HashMap::new();
    for (index, &participates) in cyclic.iter().enumerate() {
        if !participates {
            continue;
        }
        let args = operands(&block.nodes[index].payload);
        if !args
            .iter()
            .any(|arg| cyclic[arg.index] && is_empty_data(block.get_node_ty(*arg)))
        {
            continue;
        }
        for arg in args {
            let ty = block.get_node_ty(arg);
            if !cyclic[arg.index] || !is_empty_data(ty) || literals.contains_key(ty) {
                continue;
            }
            let ty = ty.clone();
            let payload = NodePayload::Literal(crate::ir_value_utils::zero_ir_value_for_type(&ty));
            *max_id = max_id
                .checked_add(1)
                .ok_or("block_inline: node IDs overflow usize")?;
            literals.insert(
                ty.clone(),
                NodeRef {
                    index: block.nodes.len(),
                },
            );
            block.nodes.push(Node {
                text_id: *max_id,
                name: None,
                pos: None,
                ty,
                payload,
            });
        }
        let payload = remap_payload_with(&block.nodes[index].payload, |(_, arg)| {
            if cyclic[arg.index] {
                literals.get(block.get_node_ty(arg)).copied().unwrap_or(arg)
            } else {
                arg
            }
        });
        block.nodes[index].replace_payload(payload);
    }
    Ok(!literals.is_empty())
}

/// Shares projected expressions between slices of the same original nodes.
struct Resolver<'a> {
    cyclic: &'a [bool],
    pending: PendingNodes,
    resolved: HashMap<BitRange, NodeRef>,
}

impl Resolver<'_> {
    /// Builds operand expressions before their users. The explicit stack also
    /// handles wiring that shifts by one bit through a very wide bus.
    fn resolve<'a>(
        &mut self,
        block: &'a Block,
        root: BitRange,
        offsets: &mut ControlValues<'a>,
    ) -> Result<(), String> {
        let mut stack = vec![(root, None::<BitProjection>)];
        let mut active = HashSet::new();
        while let Some((bits, projection)) = stack.pop() {
            if self.resolved.contains_key(&bits) {
                continue;
            }
            let payload = if let Some(projection) = projection {
                active.remove(&bits);
                if let BitProjection::Forward(part) = projection {
                    self.resolved.insert(bits, self.resolved[&part]);
                    continue;
                }
                self.payload(block, bits, projection)?
            } else if bits.lo == 0 && block.get_node_ty(bits.node) == &Type::Bits(bits.hi) {
                self.resolved.insert(bits, bits.node);
                continue;
            } else if self.cyclic[bits.node.index]
                && let Some(projection) =
                    bits.project(block, |control| offsets.root_value(block, control))
            {
                if !active.insert(bits) {
                    return Err(format!(
                        "block_inline: combinational cycle through bits [{}:{}] of {} in block '{}'",
                        bits.hi - 1,
                        bits.lo,
                        node_textual_id(block, bits.node),
                        block.name
                    ));
                }
                let base = stack.len();
                for &part in projection.inputs() {
                    stack.push((part, None));
                }
                stack.insert(base, (bits, Some(projection)));
                continue;
            } else {
                NodePayload::BitSlice {
                    arg: bits.node,
                    start: bits.lo,
                    width: bits.hi - bits.lo,
                }
            };
            let node =
                self.pending
                    .push(block, bits.node, Type::Bits(bits.hi - bits.lo), payload)?;
            self.resolved.insert(bits, node);
        }
        Ok(())
    }

    /// Rebuilds the operation after all of its selected inputs are available.
    fn payload(
        &mut self,
        block: &Block,
        bits: BitRange,
        projection: BitProjection,
    ) -> Result<NodePayload, String> {
        projection.into_payload(
            bits.hi - bits.lo,
            |part| self.resolved[&part],
            |width| {
                self.pending.push(
                    block,
                    bits.node,
                    Type::Bits(width),
                    NodePayload::Literal(crate::IrValue::Bits(crate::IrBits::zero(width))),
                )
            },
        )
    }
}
