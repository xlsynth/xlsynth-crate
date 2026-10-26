// SPDX-License-Identifier: Apache-2.0

//! Shared PIR inspection and node construction for preparation rewrites.
//!
//! The cached appender, scanning node insertion, and literal-reuse helpers
//! retain distinct behavior. Use counts include all stored consumers.

use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{self, Binop, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_match::{self, MatchCtx};
use xlsynth_pir::ir_utils;

/// Returns per-node use counts for the provided function.
pub(super) fn get_use_counts(f: &ir::Fn) -> Vec<usize> {
    let mut use_counts = vec![0usize; f.nodes.len()];
    for node in &f.nodes {
        for operand in ir_utils::operands(&node.payload) {
            use_counts[operand.index] += 1;
        }
    }
    if let Some(ret) = f.ret_node_ref {
        use_counts[ret.index] += 1;
    }
    use_counts
}

pub(super) fn nil_out_node(f: &mut ir::Fn, node_ref: ir::NodeRef) {
    let node = &mut f.nodes[node_ref.index];
    node.payload = NodePayload::Nil;
    node.ty = Type::nil();
}

pub(super) fn next_text_id(f: &ir::Fn) -> usize {
    f.nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

/// Matches a nonzero/zero test, returning its word and whether to invert it.
pub(super) fn match_zero_test(f: &ir::Fn, root: NodeRef) -> Option<(NodeRef, bool)> {
    match f.get_node(root).payload {
        NodePayload::Unop(Unop::OrReduce, arg) => Some((arg, false)),
        NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), lhs, rhs) => {
            let zero = [lhs, rhs].into_iter().find(|&n|
                matches!(&f.get_node(n).payload, NodePayload::Literal(v) if v.bits_equals_u64_value(0)))?;
            let ctx = MatchCtx::new(f);
            let bindings = ctx.commutative_binop_pair(
                root,
                op,
                ir_match::any("value"),
                ir_match::exact(zero),
            )?;
            Some((bindings.get_node("value")?, op == Binop::Eq))
        }
        _ => None,
    }
}

/// Appends a bounded replacement without rescanning the function for text IDs.
pub(super) struct NodeAppender<'a> {
    pub(super) f: &'a mut ir::Fn,
    pub(super) next_id: usize,
}

impl NodeAppender<'_> {
    pub(super) fn push(&mut self, width: usize, payload: NodePayload) -> NodeRef {
        let result = NodeRef {
            index: self.f.nodes.len(),
        };
        self.f.nodes.push(ir::Node {
            text_id: self.next_id,
            name: None,
            ty: Type::Bits(width),
            payload,
            pos: None,
        });
        self.next_id += 1;
        result
    }

    pub(super) fn literal(&mut self, width: usize, value: usize) -> NodeRef {
        self.push(
            width,
            NodePayload::Literal(
                IrValue::make_ubits(width, value as u64)
                    .expect("bounded literal fits its IR width"),
            ),
        )
    }
}

pub(super) fn push_node(f: &mut ir::Fn, ty: Type, payload: NodePayload) -> NodeRef {
    let text_id = next_text_id(f);
    let idx = f.nodes.len();
    f.nodes.push(ir::Node {
        text_id,
        name: None,
        ty,
        payload,
        pos: None,
    });
    NodeRef { index: idx }
}

pub(super) fn get_or_insert_bits1_literal(f: &mut ir::Fn, value: bool) -> NodeRef {
    // Prefer reusing an existing bits[1] literal to keep node count stable.
    for (idx, n) in f.nodes.iter().enumerate() {
        if let NodePayload::Literal(lit) = &n.payload {
            if n.ty.bit_count() == 1 && lit.to_bool().ok() == Some(value) {
                return NodeRef { index: idx };
            }
        }
    }
    let lit = IrValue::make_ubits(1, if value { 1 } else { 0 }).expect("bits[1] literal");
    push_node(f, Type::Bits(1), NodePayload::Literal(lit))
}

pub(super) fn get_or_insert_ubits_literal(f: &mut ir::Fn, bit_count: usize, value: u64) -> NodeRef {
    for (idx, n) in f.nodes.iter().enumerate() {
        if let NodePayload::Literal(lit) = &n.payload {
            if n.ty.bit_count() == bit_count && lit.to_u64().ok() == Some(value) {
                return NodeRef { index: idx };
            }
        }
    }
    let lit = IrValue::make_ubits(bit_count, value).expect("ubits literal");
    push_node(f, Type::Bits(bit_count), NodePayload::Literal(lit))
}
