// SPDX-License-Identifier: Apache-2.0

//! Local, hierarchy-preserving simplification for Block IR packages.

use std::borrow::Cow;
use std::collections::{HashSet, VecDeque};

use crate::block_invoke_inline::BlockInvokeInliner;
use crate::ir::{
    Binop, Block, NaryOp, Node, NodePayload, NodeRef, Package, PackageMember, Type, Unop,
};
use crate::ir_eval::eval_pure_if_supported;
use crate::ir_rebase_ids::package_max_emitted_node_id;
use crate::ir_utils::operands;
use crate::ir_value_utils::zero_ir_value_for_type;
use crate::ir_verify::verify_package;
use crate::{IrBits, IrValue};

/// Simplifies local constants and lowers statically controlled wiring in every
/// block, after expanding native function invokes. Block instance boundaries
/// are not crossed, foreign invokes stay opaque, and the hierarchy is
/// unchanged. Constant materialization is bounded to keep compact aggregate
/// operations compact.
///
/// Block and port names, instantiations, and original node IDs, names, and
/// source positions are preserved. Additional wiring nodes receive fresh
/// package-wide IDs. The input is unchanged if simplification fails.
pub fn simplify_block_package(package: &Package) -> Result<Package, String> {
    simplify_blocks(package, None)
}

/// Simplifies only the chosen block and reachable block definitions, leaving
/// unrelated members unchanged. Instances and shared definitions stay intact.
pub fn simplify_block_hierarchy(package: &Package, top: &str) -> Result<Package, String> {
    let mut reachable = HashSet::new();
    let mut pending = vec![top];
    while let Some(name) = pending.pop() {
        if !reachable.insert(name) {
            continue;
        }
        let block = package
            .get_block(name)
            .ok_or_else(|| format!("block '{name}' not found"))?;
        pending.extend(
            block
                .instantiations
                .iter()
                .filter_map(|instance| instance.referenced_block()),
        );
    }
    simplify_blocks(package, Some(&reachable))
}

/// Expands and simplifies a bounded set of block definitions in two phases.
fn simplify_blocks(package: &Package, selected: Option<&HashSet<&str>>) -> Result<Package, String> {
    let mut inliner =
        BlockInvokeInliner::new(package).map_err(|err| format!("invalid input package: {err}"))?;
    let mut result = package.clone();
    for member in &mut result.members {
        if let PackageMember::Block(block) = member
            && selected.is_none_or(|names| names.contains(block.name.as_str()))
            && let Cow::Owned(expanded) =
                inliner.inline_native_calls(&block.name, MAX_INLINED_NODES_PER_BLOCK)?
        {
            *block = expanded;
        }
    }
    // Expansion allocates IDs across all blocks. Wiring helpers must begin
    // above that entire range, rather than competing with the inliner.
    let mut last_id = package_max_emitted_node_id(&result);
    for member in &mut result.members {
        if let PackageMember::Block(block) = member
            && selected.is_none_or(|names| names.contains(block.name.as_str()))
        {
            simplify_block(block, &mut last_id)?;
        }
    }
    verify_package(&result).map_err(|err| format!("invalid simplified package: {err}"))?;
    Ok(result)
}

/// Rewrites original nodes in topological order, then orders any new helpers.
fn simplify_block(block: &mut Block, last_id: &mut usize) -> Result<(), String> {
    let original_len = block.nodes.len();
    for index in 1..original_len {
        simplify_node(block, index, last_id)?;
    }
    if block.nodes.len() != original_len {
        block
            .compact_and_toposort()
            .map_err(|err| format!("block {}: {err}", block.name))?;
    }
    Ok(())
}

/// Simplifies a temporary, potentially cyclic flattened proof graph to a local
/// fixed point without ordering it. Fresh helpers share the caller's package ID
/// allocator. Failure may leave the temporary graph partly simplified.
pub(crate) fn simplify_unordered_block(
    block: &mut Block,
    last_id: &mut usize,
) -> Result<(), String> {
    let initial_len = block.nodes.len();
    let limit = initial_len.saturating_mul(16).saturating_add(1_024);
    let mut users = vec![Vec::<usize>::new(); initial_len];
    for (index, node) in block.nodes.iter().enumerate().skip(1) {
        for arg in operands(&node.payload) {
            users
                .get_mut(arg.index)
                .ok_or_else(|| format!("block '{}': missing node {}", block.name, arg.index))?
                .push(index);
        }
    }
    let edge_limit = users
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .saturating_add(initial_len)
        .saturating_mul(64)
        .saturating_add(1_024);
    let mut queued = vec![true; initial_len];
    let mut queue: VecDeque<usize> = (1..initial_len).collect();
    let mut pending = vec![0usize; initial_len];
    let mut processed = vec![0usize; initial_len];
    let mut epoch = 0usize;
    let mut edges = 0usize;
    let mut visits = 0usize;
    while let Some(index) = queue.pop_front() {
        queued[index] = false;
        let incoming = std::mem::take(&mut pending[index]);
        processed[index] = processed[index].max(incoming);
        visits += 1;
        if visits > limit {
            return Err(format!(
                "block '{}': unordered simplification exceeds its convergence budget ({limit} node visits)",
                block.name
            ));
        }
        let before_len = block.nodes.len();
        let before_args = operands(&block.nodes[index].payload);
        let changed = simplify_node(block, index, last_id)?;
        if block.nodes.len().saturating_sub(initial_len) > MAX_INLINED_NODES_PER_BLOCK {
            return Err(format!(
                "block '{}': unordered simplification exceeds its helper budget ({MAX_INLINED_NODES_PER_BLOCK} nodes)",
                block.name
            ));
        }
        users.resize_with(block.nodes.len(), Vec::new);
        queued.resize(block.nodes.len(), false);
        pending.resize(block.nodes.len(), 0);
        processed.resize(block.nodes.len(), 0);
        for helper in before_len..block.nodes.len() {
            for arg in operands(&block.nodes[helper].payload) {
                users
                    .get_mut(arg.index)
                    .ok_or_else(|| format!("block '{}': missing node {}", block.name, arg.index))?
                    .push(helper);
            }
            enqueue(&mut queue, &mut queued, helper);
        }
        if changed {
            let before_args: HashSet<_> = before_args.into_iter().collect();
            for arg in operands(&block.nodes[index].payload) {
                if !before_args.contains(&arg) {
                    users
                        .get_mut(arg.index)
                        .ok_or_else(|| {
                            format!("block '{}': missing node {}", block.name, arg.index)
                        })?
                        .push(index);
                }
            }
            enqueue(&mut queue, &mut queued, index);
            epoch = epoch
                .checked_add(1)
                .ok_or("simplification epoch overflow")?;
            processed[index] = epoch;
        }
        // A wiring node can expose a newly constant slice without changing
        // its own payload. Epochs coalesce those notifications and stop waves
        // from circling indefinitely through structural feedback.
        let notification = if changed {
            epoch
        } else if transparent_wiring(&block.nodes[index].payload) {
            incoming
        } else {
            0
        };
        if notification != 0 {
            for &user in &users[index] {
                edges += 1;
                if edges > edge_limit {
                    return Err(format!(
                        "block '{}': unordered simplification exceeds its propagation budget ({edge_limit} edges)",
                        block.name
                    ));
                }
                if notification > processed[user] && notification > pending[user] {
                    pending[user] = notification;
                    enqueue(&mut queue, &mut queued, user);
                }
            }
        }
    }
    Ok(())
}

/// Identifies the wiring operations traversed recursively by `constant_slice`.
fn transparent_wiring(payload: &NodePayload) -> bool {
    matches!(
        payload,
        NodePayload::Unop(Unop::Identity | Unop::Reverse, _)
            | NodePayload::BitSlice { .. }
            | NodePayload::Nary(NaryOp::Concat, _)
            | NodePayload::ZeroExt { .. }
            | NodePayload::SignExt { .. }
    )
}

/// Schedules a node once until its current work item is consumed.
fn enqueue(queue: &mut VecDeque<usize>, queued: &mut [bool], index: usize) {
    if !queued[index] {
        queued[index] = true;
        queue.push_back(index);
    }
}

/// Applies one local, side-effect-free rewrite, preserving the original node.
fn simplify_node(block: &mut Block, index: usize, last_id: &mut usize) -> Result<bool, String> {
    let node = block.nodes[index].clone();
    if node.ty.contains_token() {
        return Ok(false);
    }
    if let Some(value) = fold_literals(block, &node) {
        let replacement = NodePayload::Literal(value);
        if replacement != node.payload {
            block.nodes[index].payload = replacement;
            return Ok(true);
        }
        return Ok(false);
    }
    let replacement = match &node.payload {
        NodePayload::TupleIndex { tuple, index } if !block.get_node(*tuple).ty.contains_token() => {
            match &block.get_node(*tuple).payload {
                NodePayload::Tuple(fields) => {
                    fields.get(*index).map(|field| identity(block, *field))
                }
                _ => None,
            }
        }
        NodePayload::ArrayIndex {
            array,
            indices,
            assumed_in_bounds,
        } if indices.len() == 1 => match &block.get_node(*array).payload {
            NodePayload::Array(elements) if !elements.is_empty() => {
                literal_offset(block, indices[0]).and_then(|index| {
                    if *assumed_in_bounds && index >= elements.len() {
                        None
                    } else {
                        Some(identity(block, elements[index.min(elements.len() - 1)]))
                    }
                })
            }
            _ => None,
        },
        NodePayload::Nary(op, args) if *op != NaryOp::Concat => {
            simplify_gate(block, *op, args, &node.ty)
        }
        NodePayload::Binop(Binop::Gate, predicate, value) => literal_bits(block, *predicate)
            .and_then(|bits| {
                if bits.is_zero() {
                    materializable(&node.ty)
                        .then(|| NodePayload::Literal(zero_ir_value_for_type(&node.ty)))
                } else {
                    Some(identity(block, *value))
                }
            }),
        NodePayload::Binop(op @ (Binop::Shll | Binop::Shrl | Binop::Shra), arg, control) => {
            if let Some(amount) = literal_offset(block, *control) {
                Some(lower_shift(block, &node, *op, *arg, amount, last_id)?)
            } else {
                None
            }
        }
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => literal_offset(block, *selector)
            .and_then(|offset| cases.get(offset).copied().or(*default))
            .map(|selected| identity(block, selected)),
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => literal_bits(block, *selector).and_then(|bits| {
            cases
                .iter()
                .enumerate()
                .find_map(|(i, case)| {
                    (i < bits.get_bit_count() && bits.get_bit(i).unwrap_or(false)).then_some(*case)
                })
                .or(*default)
                .map(|selected| identity(block, selected))
        }),
        NodePayload::OneHotSel { selector, cases } => {
            simplify_one_hot_sel(block, &node, *selector, cases, last_id)?
        }
        NodePayload::DynamicBitSlice { arg, start, width } => {
            if let Some(offset) = literal_offset(block, *start) {
                Some(lower_dynamic_slice(
                    block, &node, *arg, offset, *width, last_id,
                )?)
            } else {
                None
            }
        }
        NodePayload::BitSliceUpdate {
            arg,
            start,
            update_value,
        } => {
            if let Some(offset) = literal_offset(block, *start) {
                Some(lower_slice_update(
                    block,
                    &node,
                    *arg,
                    offset,
                    *update_value,
                    last_id,
                )?)
            } else {
                None
            }
        }
        _ => None,
    };
    if let Some(payload) = replacement
        && payload != node.payload
    {
        block.nodes[index].payload = payload;
        return Ok(true);
    }
    Ok(false)
}

const MAX_INLINED_NODES_PER_BLOCK: usize = 100_000;
const MAX_WIRING_PROJECTION_TASKS: usize = 256;
const MAX_CONSTANT_BITS: usize = 4096;
const MAX_CONSTANT_VALUE_NODES: usize = 256;

/// Bounds both flattened bits and actual aggregate value nodes before folding.
fn materializable(ty: &Type) -> bool {
    ty.checked_bit_count()
        .is_some_and(|bits| bits <= MAX_CONSTANT_BITS)
        && materialized_nodes(ty).is_some_and(|nodes| nodes <= MAX_CONSTANT_VALUE_NODES)
}

/// Counts the concrete value nodes an aggregate literal would allocate.
fn materialized_nodes(ty: &Type) -> Option<usize> {
    match ty {
        Type::Bits(_) | Type::Token => Some(1),
        Type::Tuple(fields) => fields.iter().try_fold(1usize, |count, field| {
            count.checked_add(materialized_nodes(field)?)
        }),
        Type::Array(array) => materialized_nodes(&array.element_type)?
            .checked_mul(array.element_count)?
            .checked_add(1),
    }
}

/// A static wiring fragment or padding to inspect in LSB-to-MSB order.
enum ConstantSliceTask {
    Slice(NodeRef, usize, usize),
    Zero(usize),
    RepeatLast(usize),
    ReverseLast(usize),
}

/// Extracts only statically known slice bits through wiring, never through
/// gates.
fn constant_slice(block: &Block, arg: NodeRef, start: usize, width: usize) -> Option<IrBits> {
    if width > MAX_CONSTANT_BITS {
        return None;
    }
    let mut tasks: smallvec::SmallVec<[ConstantSliceTask; 8]> =
        smallvec::smallvec![ConstantSliceTask::Slice(arg, start, width)];
    let mut output = Vec::new();
    let mut remaining = MAX_WIRING_PROJECTION_TASKS;
    while let Some(task) = tasks.pop() {
        remaining = remaining.checked_sub(1)?;
        match task {
            ConstantSliceTask::Zero(count) => output.extend(std::iter::repeat_n(false, count)),
            ConstantSliceTask::RepeatLast(count) => {
                let bit = *output.last()?;
                output.extend(std::iter::repeat_n(bit, count));
            }
            ConstantSliceTask::ReverseLast(count) => {
                let begin = output.len().checked_sub(count)?;
                output[begin..].reverse();
            }
            ConstantSliceTask::Slice(arg, start, width) => {
                if width == 0 {
                    continue;
                }
                let node = block.get_node(arg);
                let Type::Bits(node_width) = node.ty else {
                    return None;
                };
                let end = start.checked_add(width)?;
                if end > node_width {
                    return None;
                }
                match &node.payload {
                    NodePayload::Literal(IrValue::Bits(bits)) => {
                        output.extend(
                            (start..end).map(|i| bits.get_bit(i).expect("slice bit in bounds")),
                        );
                    }
                    NodePayload::Unop(Unop::Identity, inner) => {
                        tasks.push(ConstantSliceTask::Slice(*inner, start, width))
                    }
                    NodePayload::Unop(Unop::Reverse, inner) => {
                        tasks.push(ConstantSliceTask::ReverseLast(width));
                        tasks.push(ConstantSliceTask::Slice(*inner, node_width - end, width));
                    }
                    NodePayload::BitSlice {
                        arg: inner,
                        start: base,
                        ..
                    } => {
                        tasks.push(ConstantSliceTask::Slice(
                            *inner,
                            base.checked_add(start)?,
                            width,
                        ));
                    }
                    NodePayload::Nary(NaryOp::Concat, parts) => {
                        let mut offset = 0usize;
                        let mut pieces = smallvec::SmallVec::<[ConstantSliceTask; 4]>::new();
                        for part in parts.iter().rev() {
                            let Type::Bits(part_width) = block.get_node(*part).ty else {
                                return None;
                            };
                            let part_end = offset.checked_add(part_width)?;
                            let lo = start.max(offset);
                            let hi = end.min(part_end);
                            if lo < hi {
                                pieces.push(ConstantSliceTask::Slice(*part, lo - offset, hi - lo));
                            }
                            offset = part_end;
                            if offset >= end {
                                break;
                            }
                        }
                        tasks.extend(pieces.into_iter().rev());
                    }
                    NodePayload::ZeroExt { arg: inner, .. }
                    | NodePayload::SignExt { arg: inner, .. } => {
                        let Type::Bits(inner_width) = block.get_node(*inner).ty else {
                            return None;
                        };
                        let copied = if start < inner_width {
                            width.min(inner_width - start)
                        } else {
                            0
                        };
                        let padding = width - copied;
                        if padding != 0 {
                            if matches!(node.payload, NodePayload::SignExt { .. })
                                && inner_width != 0
                            {
                                tasks.push(ConstantSliceTask::RepeatLast(padding - 1));
                                tasks.push(ConstantSliceTask::Slice(*inner, inner_width - 1, 1));
                            } else {
                                tasks.push(ConstantSliceTask::Zero(padding));
                            }
                        }
                        if copied != 0 {
                            tasks.push(ConstantSliceTask::Slice(*inner, start, copied));
                        }
                    }
                    _ => return None,
                }
            }
        }
    }
    (output.len() == width).then(|| IrBits::from_lsb_is_0(&output))
}

/// Evaluates only side-effect-free operations with literal operands.
fn fold_literals(block: &Block, node: &Node) -> Option<IrValue> {
    if !materializable(&node.ty) {
        return None;
    }
    match &node.payload {
        NodePayload::BitSlice { arg, start, width } => {
            return constant_slice(block, *arg, *start, *width).map(IrValue::Bits);
        }
        NodePayload::Decode { arg, width } => {
            let index = literal_offset(block, *arg)?;
            return Some(IrValue::Bits(IrBits::from_lsb_fn(*width, |i| i == index)));
        }
        NodePayload::Encode { arg } => {
            let Type::Bits(width) = node.ty else {
                return None;
            };
            let bits = literal_bits(block, *arg)?;
            let mut encoded = 0usize;
            for i in 0..bits.get_bit_count() {
                if bits.get_bit(i).expect("bit index in bounds") {
                    encoded |= i;
                }
            }
            return Some(IrValue::Bits(IrBits::from_lsb_fn(width, |i| {
                i < usize::BITS as usize && encoded & (1usize << i) != 0
            })));
        }
        _ => {}
    }
    let pure = matches!(
        node.payload,
        NodePayload::Binop(..)
            | NodePayload::Unop(..)
            | NodePayload::Tuple(_)
            | NodePayload::TupleIndex { .. }
            | NodePayload::Array(_)
            | NodePayload::ArrayConcat(_)
            | NodePayload::ArraySlice { .. }
            | NodePayload::ArrayIndex {
                assumed_in_bounds: false,
                ..
            }
            | NodePayload::ArrayUpdate {
                assumed_in_bounds: false,
                ..
            }
            | NodePayload::DynamicBitSlice { .. }
            | NodePayload::BitSlice { .. }
            | NodePayload::BitSliceUpdate { .. }
            | NodePayload::ZeroExt { .. }
            | NodePayload::SignExt { .. }
            | NodePayload::Nary(..)
            | NodePayload::Sel { .. }
            | NodePayload::PrioritySel { .. }
            | NodePayload::OneHotSel { .. }
            | NodePayload::OneHot { .. }
            | NodePayload::ExtCarryOut { .. }
            | NodePayload::ExtPrioEncode { .. }
            | NodePayload::ExtClz { .. }
            | NodePayload::ExtNormalizeLeft { .. }
            | NodePayload::ExtMaskLow { .. }
            | NodePayload::ExtNaryAdd { .. }
    );
    if !pure {
        return None;
    }
    let args = operands(&node.payload);
    if args.is_empty() {
        return None;
    }
    let values: Option<Vec<&IrValue>> = args
        .iter()
        .map(|arg| {
            let operand = block.get_node(*arg);
            if operand.ty.contains_token() {
                return None;
            }
            match &operand.payload {
                NodePayload::Literal(value) => Some(value),
                _ => None,
            }
        })
        .collect();
    eval_pure_if_supported(node, &values?)
}

/// Simplifies neutral and absorbing constants without splitting gates into
/// bits.
fn simplify_gate(block: &Block, op: NaryOp, args: &[NodeRef], ty: &Type) -> Option<NodePayload> {
    let Type::Bits(width) = ty else {
        return None;
    };
    if !materializable(ty) || !args.iter().any(|arg| literal_bits(block, *arg).is_some()) {
        return None;
    }
    let mut kept = Vec::with_capacity(args.len());
    for arg in args {
        if let Some(bits) = literal_bits(block, *arg) {
            let zero = bits.is_zero();
            let ones = op != NaryOp::Xor && is_all_ones(bits);
            match op {
                NaryOp::And | NaryOp::Nand if zero => {
                    return Some(bits_literal(*width, op == NaryOp::Nand));
                }
                NaryOp::Or | NaryOp::Nor if ones => {
                    return Some(bits_literal(*width, op == NaryOp::Or));
                }
                NaryOp::And | NaryOp::Nand if ones => continue,
                NaryOp::Or | NaryOp::Nor | NaryOp::Xor if zero => continue,
                _ => {}
            }
        }
        kept.push(*arg);
    }
    if kept.len() == args.len() {
        return None;
    }
    Some(match kept.len() {
        0 => bits_literal(*width, matches!(op, NaryOp::And | NaryOp::Nor)),
        1 if matches!(op, NaryOp::Nand | NaryOp::Nor) => NodePayload::Unop(Unop::Not, kept[0]),
        1 => identity(block, kept[0]),
        _ => NodePayload::Nary(op, kept),
    })
}

/// Replaces a constant one-hot selector with only its active OR inputs.
fn simplify_one_hot_sel(
    block: &mut Block,
    node: &Node,
    selector: NodeRef,
    cases: &[NodeRef],
    last_id: &mut usize,
) -> Result<Option<NodePayload>, String> {
    let Some(bits) = literal_bits(block, selector) else {
        return Ok(None);
    };
    let selected: Vec<NodeRef> = cases
        .iter()
        .enumerate()
        .filter_map(|(i, case)| {
            (i < bits.get_bit_count() && bits.get_bit(i).expect("selector bit in bounds"))
                .then_some(*case)
        })
        .collect();
    Ok(match selected.as_slice() {
        [] => {
            materializable(&node.ty).then(|| NodePayload::Literal(zero_ir_value_for_type(&node.ty)))
        }
        [case] => Some(identity(block, *case)),
        _ if matches!(node.ty, Type::Bits(_)) => Some(NodePayload::Nary(NaryOp::Or, selected)),
        _ if selected.len() < cases.len() => {
            // Whole aggregate values have no OR opcode. Retain the aggregate
            // one-hot OR while removing its inactive arms; do not split it into
            // field or bit gates.
            let width = selected.len();
            let control = append_fill(block, node, width, true, last_id)?;
            Some(NodePayload::OneHotSel {
                selector: control,
                cases: selected,
            })
        }
        _ => None,
    })
}

/// Lowers a fixed shift to slices, extensions, and concatenation.
fn lower_shift(
    block: &mut Block,
    node: &Node,
    op: Binop,
    arg: NodeRef,
    amount: usize,
    last_id: &mut usize,
) -> Result<NodePayload, String> {
    let Type::Bits(width) = node.ty else {
        return Err("shift result is not bits".to_string());
    };
    if width == 0 || amount == 0 {
        return Ok(identity(block, arg));
    }
    if amount >= width {
        return if op == Binop::Shra {
            let sign = append_slice(block, node, arg, width - 1, 1, last_id)?;
            Ok(NodePayload::SignExt {
                arg: sign,
                new_bit_count: width,
            })
        } else {
            fill_payload(block, node, width, false, last_id)
        };
    }
    let kept_width = width - amount;
    match op {
        Binop::Shll => {
            let kept = append_slice(block, node, arg, 0, kept_width, last_id)?;
            let zero = append_fill(block, node, amount, false, last_id)?;
            Ok(NodePayload::Nary(NaryOp::Concat, vec![kept, zero]))
        }
        Binop::Shrl | Binop::Shra => {
            let kept = append_slice(block, node, arg, amount, kept_width, last_id)?;
            Ok(if op == Binop::Shra {
                NodePayload::SignExt {
                    arg: kept,
                    new_bit_count: width,
                }
            } else {
                NodePayload::ZeroExt {
                    arg: kept,
                    new_bit_count: width,
                }
            })
        }
        _ => unreachable!(),
    }
}

/// Lowers a fixed dynamic slice, preserving XLS zero-padding semantics.
fn lower_dynamic_slice(
    block: &mut Block,
    node: &Node,
    arg: NodeRef,
    offset: usize,
    width: usize,
    last_id: &mut usize,
) -> Result<NodePayload, String> {
    let Type::Bits(arg_width) = block.get_node(arg).ty else {
        return Err("slice input is not bits".to_string());
    };
    if width == 0 || offset >= arg_width {
        return fill_payload(block, node, width, false, last_id);
    }
    let available = width.min(arg_width - offset);
    if available == width {
        return Ok(NodePayload::BitSlice {
            arg,
            start: offset,
            width,
        });
    }
    let kept = append_slice(block, node, arg, offset, available, last_id)?;
    Ok(NodePayload::ZeroExt {
        arg: kept,
        new_bit_count: width,
    })
}

/// Lowers a fixed slice update to concatenated slices, clipping at the MSB.
fn lower_slice_update(
    block: &mut Block,
    node: &Node,
    arg: NodeRef,
    offset: usize,
    update: NodeRef,
    last_id: &mut usize,
) -> Result<NodePayload, String> {
    let Type::Bits(width) = node.ty else {
        return Err("slice-update result is not bits".to_string());
    };
    let Type::Bits(update_width) = block.get_node(update).ty else {
        return Err("slice-update value is not bits".to_string());
    };
    if offset >= width || update_width == 0 {
        return Ok(identity(block, arg));
    }
    let copied = update_width.min(width - offset);
    let high_start = offset + copied;
    let mut parts = Vec::with_capacity(3);
    if high_start < width {
        parts.push(append_slice(
            block,
            node,
            arg,
            high_start,
            width - high_start,
            last_id,
        )?);
    }
    parts.push(if copied == update_width {
        update
    } else {
        append_slice(block, node, update, 0, copied, last_id)?
    });
    if offset != 0 {
        parts.push(append_slice(block, node, arg, 0, offset, last_id)?);
    }
    Ok(if parts.len() == 1 {
        identity(block, parts[0])
    } else {
        NodePayload::Nary(NaryOp::Concat, parts)
    })
}

/// Represents a constant fill compactly when its literal exceeds the budget.
fn fill_payload(
    block: &mut Block,
    node: &Node,
    width: usize,
    ones: bool,
    last_id: &mut usize,
) -> Result<NodePayload, String> {
    if width <= MAX_CONSTANT_BITS {
        return Ok(bits_literal(width, ones));
    }
    let bit = append_node(block, node, Type::Bits(1), bits_literal(1, ones), last_id)?;
    Ok(if ones {
        NodePayload::SignExt {
            arg: bit,
            new_bit_count: width,
        }
    } else {
        NodePayload::ZeroExt {
            arg: bit,
            new_bit_count: width,
        }
    })
}

/// Appends a constant fill without creating a width-sized literal.
fn append_fill(
    block: &mut Block,
    node: &Node,
    width: usize,
    ones: bool,
    last_id: &mut usize,
) -> Result<NodeRef, String> {
    let payload = fill_payload(block, node, width, ones, last_id)?;
    append_node(block, node, Type::Bits(width), payload, last_id)
}

/// Adds a static slice helper while retaining the original node's source
/// location.
fn append_slice(
    block: &mut Block,
    node: &Node,
    arg: NodeRef,
    start: usize,
    width: usize,
    last_id: &mut usize,
) -> Result<NodeRef, String> {
    append_node(
        block,
        node,
        Type::Bits(width),
        NodePayload::BitSlice { arg, start, width },
        last_id,
    )
}

/// Allocates a new graph node with an unused textual ID.
fn append_node(
    block: &mut Block,
    node: &Node,
    ty: Type,
    payload: NodePayload,
    last_id: &mut usize,
) -> Result<NodeRef, String> {
    *last_id = last_id
        .checked_add(1)
        .ok_or_else(|| "cannot allocate a fresh node ID".to_string())?;
    let result = NodeRef {
        index: block.nodes.len(),
    };
    block.nodes.push(Node {
        text_id: *last_id,
        name: None,
        ty,
        payload,
        pos: node.pos.clone(),
    });
    Ok(result)
}

/// Returns the literal bit vector at a reference, if any.
fn literal_bits(block: &Block, arg: NodeRef) -> Option<&IrBits> {
    match &block.get_node(arg).payload {
        NodePayload::Literal(IrValue::Bits(bits)) => Some(bits),
        _ => None,
    }
}

/// Converts an arbitrary-width constant to a saturating host index.
fn literal_offset(block: &Block, arg: NodeRef) -> Option<usize> {
    literal_bits(block, arg).map(|bits| {
        bits.to_u64()
            .ok()
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(usize::MAX)
    })
}

/// Preserves a named node while forwarding an operand or its known literal.
fn identity(block: &Block, arg: NodeRef) -> NodePayload {
    match &block.get_node(arg).payload {
        NodePayload::Literal(value) if materializable(&block.get_node(arg).ty) => {
            NodePayload::Literal(value.clone())
        }
        _ => NodePayload::Unop(Unop::Identity, arg),
    }
}

/// Checks an all-ones constant directly in its limbs without allocating a mask.
fn is_all_ones(bits: &IrBits) -> bool {
    let width = bits.get_bit_count();
    let remainder = width % 64;
    bits.limbs().iter().enumerate().all(|(index, limb)| {
        let expected = if remainder != 0 && index + 1 == bits.limbs().len() {
            (1u64 << remainder) - 1
        } else {
            u64::MAX
        };
        *limb == expected
    })
}

/// Constructs a width-correct all-zero or all-one literal.
fn bits_literal(width: usize, ones: bool) -> NodePayload {
    debug_assert!(
        width <= MAX_CONSTANT_BITS,
        "constant literal exceeds materialization budget"
    );
    NodePayload::Literal(IrValue::Bits(if ones {
        IrBits::all_ones(width)
    } else {
        IrBits::zero(width)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_eval::{BlockEvaluator, RegisterInit};
    use crate::ir_parser::Parser;

    /// Parses and verifies an independent Block IR test package.
    fn parse(text: &str) -> Package {
        let package = Parser::new(text).parse_package().unwrap();
        verify_package(&package).unwrap();
        package
    }

    /// Returns a node by its explicitly assigned name.
    fn named<'a>(package: &'a Package, block: &str, name: &str) -> &'a Node {
        package
            .get_block(block)
            .unwrap()
            .nodes
            .iter()
            .find(|node| node.name.as_deref() == Some(name))
            .unwrap()
    }

    /// Compares every output for all combinations of two four-bit inputs.
    fn assert_equivalent(original: &Package, simplified: &Package) {
        let before = BlockEvaluator::prepare(original, "top").unwrap();
        let after = BlockEvaluator::prepare(simplified, "top").unwrap();
        let old_state = before.initial_state(RegisterInit::AllZeros).unwrap();
        let new_state = after.initial_state(RegisterInit::AllZeros).unwrap();
        for a in 0..16 {
            for b in 0..16 {
                let input = [
                    IrValue::make_ubits(4, a).unwrap(),
                    IrValue::make_ubits(4, b).unwrap(),
                ];
                assert_eq!(
                    before.step(&old_state, &input).unwrap().outputs,
                    after.step(&new_state, &input).unwrap().outputs
                );
            }
        }
    }

    #[test]
    fn local_constants_muxes_and_gates_preserve_names() {
        let original = parse(
            r#"package p

top block top(a: bits[4], b: bits[4], o0: bits[4], o1: bits[4], o2: bits[4], o3: bits[4]) {
  a: bits[4] = input_port(name=a, id=1)
  b: bits[4] = input_port(name=b, id=2)
  zero: bits[4] = literal(value=0, id=3)
  one: bits[4] = literal(value=1, id=4)
  calculated: bits[4] = add(zero, one, id=5)
  keep_a: bits[4] = or(a, zero, id=6)
  absorbed: bits[4] = and(b, zero, id=7)
  chosen: bits[4] = sel(calculated, cases=[b, keep_a], default=zero, id=8)
  both: bits[2] = literal(value=3, id=9)
  merged: bits[4] = one_hot_sel(both, cases=[a, b], id=10)
  o0: () = output_port(keep_a, name=o0, id=11)
  o1: () = output_port(absorbed, name=o1, id=12)
  o2: () = output_port(chosen, name=o2, id=13)
  o3: () = output_port(merged, name=o3, id=14)
}
"#,
        );
        let before_text = original.to_string();
        let simplified = simplify_block_package(&original).unwrap();
        assert_eq!(original.to_string(), before_text);
        assert!(matches!(
            named(&simplified, "top", "calculated").payload,
            NodePayload::Literal(_)
        ));
        assert!(matches!(
            named(&simplified, "top", "keep_a").payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
        assert!(matches!(
            named(&simplified, "top", "absorbed").payload,
            NodePayload::Literal(_)
        ));
        assert!(matches!(
            named(&simplified, "top", "chosen").payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
        assert!(matches!(
            named(&simplified, "top", "merged").payload,
            NodePayload::Nary(NaryOp::Or, _)
        ));
        assert_eq!(named(&simplified, "top", "chosen").text_id, 8);
        assert_equivalent(&original, &simplified);
    }

    #[test]
    fn fixed_shifts_and_dynamic_slices_lower_to_wiring() {
        for (op, amount, amount_width) in [
            ("shll", "0", 4),
            ("shll", "1", 4),
            ("shrl", "3", 4),
            ("shra", "1", 4),
            ("shra", "4", 4),
            ("shll", "0", 128),
            ("shrl", "1", 128),
            ("shll", "0x10000000000000000", 128),
        ] {
            let original = parse(&format!(
                r#"package p

top block top(a: bits[4], b: bits[4], shifted: bits[4], sliced: bits[4], updated: bits[4]) {{
  a: bits[4] = input_port(name=a, id=1)
  b: bits[4] = input_port(name=b, id=2)
  amount: bits[{amount_width}] = literal(value={amount}, id=3)
  shift: bits[4] = {op}(a, amount, id=4)
  slice: bits[4] = dynamic_bit_slice(a, amount, width=4, id=5)
  update: bits[4] = bit_slice_update(a, amount, b, id=6)
  shifted: () = output_port(shift, name=shifted, id=7)
  sliced: () = output_port(slice, name=sliced, id=8)
  updated: () = output_port(update, name=updated, id=9)
}}
"#
            ));
            let simplified = simplify_block_package(&original).unwrap();
            let block = simplified.get_block("top").unwrap();
            assert!(!block.nodes.iter().any(|node| matches!(
                node.payload,
                NodePayload::Binop(Binop::Shll | Binop::Shrl | Binop::Shra, ..)
                    | NodePayload::DynamicBitSlice { .. }
                    | NodePayload::BitSliceUpdate { .. }
            )));
            assert_equivalent(&original, &simplified);
        }
    }

    #[test]
    fn preserves_shared_child_definition_and_unknown_control() {
        let original = parse(
            r#"package p

block leaf(a: bits[4], b: bits[4], s: bits[1], y: bits[4]) {
  a: bits[4] = input_port(name=a, id=1)
  b: bits[4] = input_port(name=b, id=2)
  s: bits[1] = input_port(name=s, id=3)
  pick: bits[4] = sel(s, cases=[a, b], id=4)
  y: () = output_port(pick, name=y, id=5)
}

top block top(a: bits[4], b: bits[4], y0: bits[4], y1: bits[4]) {
  instantiation u0(block=leaf, kind=block)
  instantiation u1(block=leaf, kind=block)
  a: bits[4] = input_port(name=a, id=10)
  b: bits[4] = input_port(name=b, id=11)
  c0: bits[1] = literal(value=0, id=12)
  c1: bits[1] = literal(value=1, id=13)
  instantiation_input.14: () = instantiation_input(a, instantiation=u0, port_name=a, id=14)
  instantiation_input.15: () = instantiation_input(b, instantiation=u0, port_name=b, id=15)
  instantiation_input.16: () = instantiation_input(c0, instantiation=u0, port_name=s, id=16)
  instantiation_input.17: () = instantiation_input(a, instantiation=u1, port_name=a, id=17)
  instantiation_input.18: () = instantiation_input(b, instantiation=u1, port_name=b, id=18)
  instantiation_input.19: () = instantiation_input(c1, instantiation=u1, port_name=s, id=19)
  result0: bits[4] = instantiation_output(instantiation=u0, port_name=y, id=20)
  result1: bits[4] = instantiation_output(instantiation=u1, port_name=y, id=21)
  y0: () = output_port(result0, name=y0, id=22)
  y1: () = output_port(result1, name=y1, id=23)
}
"#,
        );
        let simplified = simplify_block_package(&original).unwrap();
        assert_eq!(simplified.members.len(), original.members.len());
        assert_eq!(simplified.get_block("top").unwrap().instantiations.len(), 2);
        assert!(matches!(
            named(&simplified, "leaf", "pick").payload,
            NodePayload::Sel { .. }
        ));
        assert_equivalent(&original, &simplified);
    }

    #[test]
    fn zero_width_and_observable_operations_are_preserved() {
        let original = parse(
            r#"package p

top block top(a: bits[0], zero_width: bits[0], checked: bits[4]) {
  a: bits[0] = input_port(name=a, id=1)
  offset: bits[3] = literal(value=0, id=2)
  shifted: bits[0] = shra(a, offset, id=3)
  array: bits[4][2] = literal(value=[1, 2], id=4)
  oob: bits[3] = literal(value=7, id=5)
  access: bits[4] = array_index(array, indices=[oob], assumed_in_bounds=true, id=6)
  token_start: token = after_all(id=7)
  bad: bits[1] = literal(value=0, id=8)
  assertion: token = assert(token_start, bad, message="bad", id=9)
  zero_width: () = output_port(shifted, name=zero_width, id=10)
  checked: () = output_port(access, name=checked, id=11)
}
"#,
        );
        let simplified = simplify_block_package(&original).unwrap();
        assert!(matches!(
            named(&simplified, "top", "shifted").payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
        assert!(matches!(
            named(&simplified, "top", "access").payload,
            NodePayload::ArrayIndex {
                assumed_in_bounds: true,
                ..
            }
        ));
        assert!(matches!(
            named(&simplified, "top", "assertion").payload,
            NodePayload::Assert { .. }
        ));
        for package in [&original, &simplified] {
            let evaluator = BlockEvaluator::prepare(package, "top").unwrap();
            let state = evaluator.initial_state(RegisterInit::AllZeros).unwrap();
            assert!(
                evaluator
                    .step(&state, &[IrValue::make_ubits(0, 0).unwrap()])
                    .is_err()
            );
        }
    }

    #[test]
    fn constant_priority_and_decoder_controls_propagate() {
        let original = parse(
            r#"package p

top block top(a: bits[4], b: bits[4], y: bits[4]) {
  a: bits[4] = input_port(name=a, id=1)
  b: bits[4] = input_port(name=b, id=2)
  start: bits[2] = literal(value=1, id=3)
  decoded: bits[2] = decode(start, width=2, id=4)
  pick: bits[4] = priority_sel(decoded, cases=[a, b], default=a, id=5)
  y: () = output_port(pick, name=y, id=6)
}
"#,
        );
        let simplified = simplify_block_package(&original).unwrap();
        assert!(matches!(
            named(&simplified, "top", "decoded").payload,
            NodePayload::Literal(_)
        ));
        assert!(matches!(
            named(&simplified, "top", "pick").payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
        assert_equivalent(&original, &simplified);
    }

    #[test]
    fn wide_selectors_use_their_value_instead_of_their_declared_width() {
        for selector in ["0", "1", "0x10000000000000000"] {
            let original = parse(&format!(
                r#"package p

top block top(a: bits[4], b: bits[4], y: bits[4]) {{
  a: bits[4] = input_port(name=a, id=1)
  b: bits[4] = input_port(name=b, id=2)
  selector: bits[128] = literal(value={selector}, id=3)
  pick: bits[4] = sel(selector, cases=[a, b], default=a, id=4)
  y: () = output_port(pick, name=y, id=5)
}}
"#
            ));
            let simplified = simplify_block_package(&original).unwrap();
            assert!(matches!(
                named(&simplified, "top", "pick").payload,
                NodePayload::Unop(Unop::Identity, _)
            ));
            assert_equivalent(&original, &simplified);
        }
    }

    #[test]
    fn aggregate_one_hot_select_prunes_inactive_arms_without_splitting_gates() {
        let original = parse(
            r#"package p

top block top(a: bits[4], b: bits[4], y: (bits[4], bits[4])) {
  a: bits[4] = input_port(name=a, id=1)
  b: bits[4] = input_port(name=b, id=2)
  ab: (bits[4], bits[4]) = tuple(a, b, id=3)
  ba: (bits[4], bits[4]) = tuple(b, a, id=4)
  control: bits[3] = literal(value=3, id=5)
  merge: (bits[4], bits[4]) = one_hot_sel(control, cases=[ab, ba, ab], id=6)
  y: () = output_port(merge, name=y, id=7)
}
"#,
        );
        let simplified = simplify_block_package(&original).unwrap();
        let NodePayload::OneHotSel { selector, cases } =
            &named(&simplified, "top", "merge").payload
        else {
            panic!("expected aggregate one-hot OR");
        };
        assert_eq!(cases.len(), 2);
        assert_eq!(
            literal_bits(simplified.get_block("top").unwrap(), *selector),
            Some(&IrBits::all_ones(2))
        );
        assert_equivalent(&original, &simplified);
    }

    #[test]
    fn native_calls_expand_before_local_wiring_simplification() {
        let original = parse(
            r#"package p

fn control(x: bits[4] id=1) -> bits[4] {
  ret amount: bits[4] = literal(value=1, id=2)
}

fn shift_right(x: bits[4] id=3) -> bits[4] {
  amount: bits[4] = literal(value=2, id=4)
  ret shifted: bits[4] = shrl(x, amount, id=5)
}

block leaf(a: bits[4], y0: bits[4], y1: bits[4]) {
  a: bits[4] = input_port(name=a, id=6)
  amount: bits[4] = invoke(a, to_apply=control, id=7)
  shifted: bits[4] = shll(a, amount, id=8)
  called: bits[4] = invoke(a, to_apply=shift_right, id=9)
  y0: () = output_port(shifted, name=y0, id=10)
  y1: () = output_port(called, name=y1, id=11)
}

top block top(a: bits[4], b: bits[4], y0: bits[4], y1: bits[4]) {
  instantiation u0(block=leaf, kind=block)
  a: bits[4] = input_port(name=a, id=12)
  b: bits[4] = input_port(name=b, id=13)
  instantiation_input.14: () = instantiation_input(a, instantiation=u0, port_name=a, id=14)
  r0: bits[4] = instantiation_output(instantiation=u0, port_name=y0, id=15)
  r1: bits[4] = instantiation_output(instantiation=u0, port_name=y1, id=16)
  y0: () = output_port(r0, name=y0, id=17)
  y1: () = output_port(r1, name=y1, id=18)
}
"#,
        );
        let simplified = simplify_block_package(&original).unwrap();
        let leaf = simplified.get_block("leaf").unwrap();
        assert!(!leaf.nodes.iter().any(|node| matches!(
            node.payload,
            NodePayload::Invoke { .. }
                | NodePayload::Binop(Binop::Shll | Binop::Shrl | Binop::Shra, ..)
        )));
        assert_eq!(named(&simplified, "leaf", "shifted").text_id, 8);
        assert_eq!(named(&simplified, "leaf", "called").text_id, 9);
        assert_eq!(simplified.members.len(), original.members.len());
        assert_eq!(simplified.get_block("top").unwrap().instantiations.len(), 1);
        assert_equivalent(&original, &simplified);
    }

    #[test]
    fn foreign_invokes_remain_opaque_even_in_unrelated_or_expanded_blocks() {
        let text = r#"package p

#[ffi_proto("""code_template: "foreign_cell {fn} (.x({x}), .y({return}));"
""")]
fn foreign(x: bits[4] id=1) -> bits[4] {
  ret x: bits[4] = param(name=x, id=1)
}

fn native(x: bits[4] id=2) -> bits[4] {
  ret delegated: bits[4] = invoke(x, to_apply=foreign, id=3)
}

block unrelated(a: bits[4], y: bits[4]) {
  a: bits[4] = input_port(name=a, id=4)
  delegated: bits[4] = invoke(a, to_apply=foreign, id=5)
  y: () = output_port(delegated, name=y, id=6)
}

top block top(a: bits[4], y: bits[4]) {
  a: bits[4] = input_port(name=a, id=7)
  delegated: bits[4] = invoke(a, to_apply=native, id=8)
  y: () = output_port(delegated, name=y, id=9)
}
"#;
        let original = Parser::new(text)
            .parse_package_preserving_ffi_invokes()
            .unwrap();
        let simplified = simplify_block_package(&original).unwrap();
        for name in ["unrelated", "top"] {
            assert!(simplified.get_block(name).unwrap().nodes.iter().any(|node| matches!(&node.payload, NodePayload::Invoke { to_apply, .. } if to_apply == "foreign")));
        }
    }

    #[test]
    fn compact_aggregate_operations_stay_compact_while_static_scalar_controls_fold() {
        let original = parse(
            r#"package p

top block top(a: bits[4], y: bits[4], huge: bits[4][1000]) {
  a: bits[4] = input_port(name=a, id=1)
  one: bits[1] = literal(value=1, id=2)
  packed: bits[5] = concat(a, one, id=3)
  control: bits[1] = bit_slice(packed, start=0, width=1, id=4)
  shifted: bits[4] = shll(a, control, id=5)
  array: bits[4][1] = literal(value=[1], id=6)
  zero: bits[1] = literal(value=0, id=7)
  repeated: bits[4][1000] = array_slice(array, zero, width=1000, id=8)
  y: () = output_port(shifted, name=y, id=9)
  huge: () = output_port(repeated, name=huge, id=10)
}
"#,
        );
        let simplified = simplify_block_package(&original).unwrap();
        assert!(matches!(
            named(&simplified, "top", "control").payload,
            NodePayload::Literal(_)
        ));
        assert!(matches!(
            named(&simplified, "top", "shifted").payload,
            NodePayload::Nary(NaryOp::Concat, _)
        ));
        assert!(matches!(
            named(&simplified, "top", "repeated").payload,
            NodePayload::ArraySlice { .. }
        ));
        assert!(simplified.get_block("top").unwrap().nodes.len() < 20);
    }

    #[test]
    fn wide_static_fills_use_compact_extensions() {
        let original = parse(
            r#"package p

top block top(a: bits[8192], y0: bits[8192], y1: bits[8192], y2: bits[8192]) {
  a: bits[8192] = input_port(name=a, id=1)
  padding: bits[14] = literal(value=5000, id=2)
  beyond: bits[14] = literal(value=9000, id=3)
  padded: bits[8192] = shll(a, padding, id=4)
  zeroed: bits[8192] = shrl(a, beyond, id=5)
  sliced: bits[8192] = dynamic_bit_slice(a, beyond, width=8192, id=6)
  y0: () = output_port(padded, name=y0, id=7)
  y1: () = output_port(zeroed, name=y1, id=8)
  y2: () = output_port(sliced, name=y2, id=9)
}
"#,
        );
        let simplified = simplify_block_package(&original).unwrap();
        let block = simplified.get_block("top").unwrap();
        assert!(matches!(
            named(&simplified, "top", "padded").payload,
            NodePayload::Nary(NaryOp::Concat, _)
        ));
        assert!(matches!(
            named(&simplified, "top", "zeroed").payload,
            NodePayload::ZeroExt {
                new_bit_count: 8192,
                ..
            }
        ));
        assert!(matches!(
            named(&simplified, "top", "sliced").payload,
            NodePayload::ZeroExt {
                new_bit_count: 8192,
                ..
            }
        ));
        assert!(block.nodes.iter().all(|node| !matches!(&node.payload, NodePayload::Literal(IrValue::Bits(bits)) if bits.get_bit_count() > MAX_CONSTANT_BITS)));
        assert!(block.nodes.len() < 20);
    }

    #[test]
    fn selected_hierarchy_does_not_expand_unrelated_native_calls() {
        let original = parse(
            r#"package p

fn native(x: bits[4] id=1) -> bits[4] {
  ret one: bits[4] = literal(value=1, id=2)
}

block unrelated(a: bits[4], y: bits[4]) {
  a: bits[4] = input_port(name=a, id=3)
  call: bits[4] = invoke(a, to_apply=native, id=4)
  y: () = output_port(call, name=y, id=5)
}

top block top(a: bits[4], y: bits[4]) {
  a: bits[4] = input_port(name=a, id=6)
  call: bits[4] = invoke(a, to_apply=native, id=7)
  y: () = output_port(call, name=y, id=8)
}
"#,
        );
        let simplified = simplify_block_hierarchy(&original, "top").unwrap();
        assert!(matches!(
            named(&simplified, "unrelated", "call").payload,
            NodePayload::Invoke { .. }
        ));
        assert!(matches!(
            named(&simplified, "top", "call").payload,
            NodePayload::Literal(_)
        ));
    }

    #[test]
    fn constructor_projections_preserve_checked_bounds_and_tokens() {
        let original = parse(
            r#"package p

top block top(a: bits[1], b: bits[1], good: bits[1], bad: bits[1], ordered: bits[1]) {
  a: bits[1] = input_port(name=a, id=1)
  b: bits[1] = input_port(name=b, id=2)
  values: bits[1][2] = array(a, b, id=3)
  one: bits[2] = literal(value=1, id=4)
  three: bits[2] = literal(value=3, id=5)
  checked_good: bits[1] = array_index(values, indices=[one], assumed_in_bounds=true, id=6)
  checked_bad: bits[1] = array_index(values, indices=[three], assumed_in_bounds=true, id=7)
  token_start: token = after_all(id=8)
  pair: (token, bits[1]) = tuple(token_start, a, id=9)
  token_ordered: bits[1] = tuple_index(pair, index=1, id=10)
  good: () = output_port(checked_good, name=good, id=11)
  bad: () = output_port(checked_bad, name=bad, id=12)
  ordered: () = output_port(token_ordered, name=ordered, id=13)
}
"#,
        );
        let simplified = simplify_block_package(&original).unwrap();
        assert!(matches!(
            named(&simplified, "top", "checked_good").payload,
            NodePayload::Unop(Unop::Identity, _)
        ));
        assert!(matches!(
            named(&simplified, "top", "checked_bad").payload,
            NodePayload::ArrayIndex {
                assumed_in_bounds: true,
                ..
            }
        ));
        assert!(matches!(
            named(&simplified, "top", "token_ordered").payload,
            NodePayload::TupleIndex { .. }
        ));
    }

    #[test]
    fn unordered_worklist_propagates_through_unchanged_wiring_feedback() {
        let mut package = parse(
            r#"package unordered

top block sample(x: bits[1], y: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  zero: bits[1] = literal(value=0, id=2)
  initial: bits[2] = concat(x, zero, id=3)
  chosen: bits[1] = bit_slice(initial, start=0, width=1, id=4)
  future: bits[1] = not(zero, id=5)
  mixed: bits[2] = concat(chosen, future, id=6)
  y: () = output_port(chosen, name=y, id=7)
}
"#,
        );
        let block = package.get_block_mut("sample").unwrap();
        let chosen = NodeRef {
            index: block
                .nodes
                .iter()
                .position(|n| n.name.as_deref() == Some("chosen"))
                .unwrap(),
        };
        let mixed = NodeRef {
            index: block
                .nodes
                .iter()
                .position(|n| n.name.as_deref() == Some("mixed"))
                .unwrap(),
        };
        block.nodes[chosen.index].payload = NodePayload::BitSlice {
            arg: mixed,
            start: 0,
            width: 1,
        };
        let mut id = 7;
        simplify_unordered_block(block, &mut id).unwrap();
        assert_eq!(
            block.nodes[chosen.index].payload,
            NodePayload::Literal(IrValue::bool(true))
        );
    }

    #[test]
    fn unordered_worklist_coalesces_long_wiring_chains() {
        let mut package = parse(
            r#"package unordered

top block sample(y: bits[1]) {
  zero: bits[1] = literal(value=0, id=1)
  y: () = output_port(zero, name=y, id=2)
}
"#,
        );
        let block = package.get_block_mut("sample").unwrap();
        let mut arg = NodeRef { index: 1 };
        let template = block.get_node(arg).clone();
        let mut id = 2;
        for _ in 0..512 {
            arg = append_node(
                block,
                &template,
                Type::Bits(1),
                NodePayload::Unop(Unop::Identity, arg),
                &mut id,
            )
            .unwrap();
        }
        simplify_unordered_block(block, &mut id).unwrap();
        assert_eq!(
            block.get_node(arg).payload,
            NodePayload::Literal(IrValue::bool(false))
        );
    }

    #[test]
    fn unordered_worklist_revisits_forward_dependencies_without_sorting() {
        let mut package = parse(
            r#"package unordered
top block sample(y: bits[1]) {
  zero: bits[1] = literal(value=0, id=1)
  forward: bits[1] = identity(zero, id=2)
  future: bits[1] = not(zero, id=3)
  y: () = output_port(forward, name=y, id=4)
}
"#,
        );
        let block = package.get_block_mut("sample").unwrap();
        let forward = NodeRef {
            index: block
                .nodes
                .iter()
                .position(|n| n.name.as_deref() == Some("forward"))
                .unwrap(),
        };
        let future = NodeRef {
            index: block
                .nodes
                .iter()
                .position(|n| n.name.as_deref() == Some("future"))
                .unwrap(),
        };
        block.nodes[forward.index].payload = NodePayload::Unop(Unop::Identity, future);
        let before = block.nodes.len();
        let mut id = 4;
        simplify_unordered_block(block, &mut id).unwrap();
        assert_eq!(block.nodes.len(), before);
        assert!(forward.index < future.index);
        assert_eq!(
            block.nodes[forward.index].payload,
            NodePayload::Literal(IrValue::bool(true))
        );
        assert!(matches!(
            block.nodes.last().unwrap().payload,
            NodePayload::OutputPort { .. }
        ));
    }
}
