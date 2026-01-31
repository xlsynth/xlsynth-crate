// SPDX-License-Identifier: Apache-2.0
//
//! Desugars PIR extension ops (e.g. `ext_carry_out`) into upstream-compatible
//! PIR / XLS IR basis operations.
//!
//! ## Design invariant
//! This module is a *semantic projection* from the extended PIR opcode set onto
//! the canonical XLS IR opcode basis. It must be deterministic and
//! semantics-preserving; it is **not** where QoR strategies belong.
//!
//! Gate-level QoR strategies belong in `xlsynth-g8r` (gatify), where different
//! circuits can be chosen for the same semantics.

use crate::ir::{Binop, Fn, Node, NodePayload, NodeRef, Package, PackageMember, Type, Unop};
use crate::ir_utils::compact_and_toposort_in_place;
use crate::math::ceil_log2;

#[derive(Debug, Clone)]
pub struct DesugarError {
    msg: String,
}

impl DesugarError {
    fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl std::fmt::Display for DesugarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DesugarError: {}", self.msg)
    }
}

impl std::error::Error for DesugarError {}

fn next_text_id(f: &Fn) -> usize {
    f.nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn expect_bits_width(f: &Fn, r: NodeRef, ctx: &str) -> Result<usize, DesugarError> {
    let ty = f.get_node_ty(r);
    match ty {
        Type::Bits(w) => Ok(*w),
        _ => Err(DesugarError::new(format!(
            "{}: expected bits operand, got {}",
            ctx, ty
        ))),
    }
}

fn push_node(f: &mut Fn, ty: Type, payload: NodePayload) -> NodeRef {
    let text_id = next_text_id(f);
    let new_index = f.nodes.len();
    f.nodes.push(Node {
        text_id,
        name: None,
        ty,
        payload,
        pos: None,
    });
    NodeRef { index: new_index }
}

fn desugar_ext_carry_out_in_fn(f: &mut Fn) -> Result<bool, DesugarError> {
    let mut changed = false;

    // Snapshot length so we only visit original nodes; desugaring appends nodes.
    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        let NodePayload::ExtCarryOut { lhs, rhs, c_in } = payload else {
            continue;
        };
        changed = true;

        let w = expect_bits_width(f, lhs, "ext_carry_out.lhs")?;
        let rhs_w = expect_bits_width(f, rhs, "ext_carry_out.rhs")?;
        if w != rhs_w {
            return Err(DesugarError::new(format!(
                "ext_carry_out: lhs width {} != rhs width {}",
                w, rhs_w
            )));
        }
        let c_in_w = expect_bits_width(f, c_in, "ext_carry_out.c_in")?;
        if c_in_w != 1 {
            return Err(DesugarError::new(format!(
                "ext_carry_out: c_in must be bits[1], got bits[{}]",
                c_in_w
            )));
        }

        // Canonical desugaring:
        //   carry = bit_slice(
        //     add(add(zero_ext(lhs,w+1), zero_ext(rhs,w+1)), zero_ext(c_in,w+1)),
        //     start=w, width=1)
        let w1 = w.saturating_add(1);
        let lhs_ext = push_node(
            f,
            Type::Bits(w1),
            NodePayload::ZeroExt {
                arg: lhs,
                new_bit_count: w1,
            },
        );
        let rhs_ext = push_node(
            f,
            Type::Bits(w1),
            NodePayload::ZeroExt {
                arg: rhs,
                new_bit_count: w1,
            },
        );
        let sum_w1 = push_node(
            f,
            Type::Bits(w1),
            NodePayload::Binop(Binop::Add, lhs_ext, rhs_ext),
        );
        let c_in_ext = push_node(
            f,
            Type::Bits(w1),
            NodePayload::ZeroExt {
                arg: c_in,
                new_bit_count: w1,
            },
        );
        let sum_w1_ci = push_node(
            f,
            Type::Bits(w1),
            NodePayload::Binop(Binop::Add, sum_w1, c_in_ext),
        );
        let lowered_carry_out = push_node(
            f,
            Type::Bits(1),
            NodePayload::BitSlice {
                arg: sum_w1_ci,
                start: w,
                width: 1,
            },
        );

        // Overwrite the ext node in-place; compaction/toposort will place deps
        // before this node.
        let node = f.get_node_mut(nr);
        node.ty = Type::Bits(1);
        node.payload = NodePayload::Unop(Unop::Identity, lowered_carry_out);
    }

    Ok(changed)
}

fn desugar_ext_prio_encode_in_fn(f: &mut Fn) -> Result<bool, DesugarError> {
    let mut changed = false;

    // Snapshot length so we only visit original nodes; desugaring appends nodes.
    let original_len = f.nodes.len();
    for idx in 0..original_len {
        let nr = NodeRef { index: idx };
        let payload = f.get_node(nr).payload.clone();
        let NodePayload::ExtPrioEncode { arg, lsb_prio } = payload else {
            continue;
        };
        changed = true;

        let n = expect_bits_width(f, arg, "ext_prio_encode.arg")?;
        let one_hot_w = n.saturating_add(1);
        let out_w = ceil_log2(one_hot_w);

        let one_hot = push_node(
            f,
            Type::Bits(one_hot_w),
            NodePayload::OneHot { arg, lsb_prio },
        );
        let encoded = push_node(f, Type::Bits(out_w), NodePayload::Encode { arg: one_hot });

        // Overwrite the ext node in-place; compaction/toposort will place deps
        // before this node.
        let node = f.get_node_mut(nr);
        node.ty = Type::Bits(out_w);
        node.payload = NodePayload::Unop(Unop::Identity, encoded);
    }

    Ok(changed)
}

/// Desugars extension ops within `f` into upstream-compatible PIR operations.
///
/// This function also normalizes the node list into a valid topological order.
pub fn desugar_extensions_in_fn(f: &mut Fn) -> Result<(), DesugarError> {
    let _changed = desugar_ext_carry_out_in_fn(f)? | desugar_ext_prio_encode_in_fn(f)?;
    compact_and_toposort_in_place(f).map_err(DesugarError::new)?;
    Ok(())
}

/// Desugars extension ops within `pkg` into upstream-compatible PIR operations.
pub fn desugar_extensions_in_package(pkg: &mut Package) -> Result<(), DesugarError> {
    for member in pkg.members.iter_mut() {
        match member {
            PackageMember::Function(f) => desugar_extensions_in_fn(f)?,
            PackageMember::Block { func, .. } => desugar_extensions_in_fn(func)?,
        }
    }
    Ok(())
}

/// Emits upstream-compatible XLS IR text for `pkg` by desugaring extensions
/// first.
pub fn emit_package_as_xls_ir_text(pkg: &Package) -> Result<String, DesugarError> {
    let mut desugared = pkg.clone();
    desugar_extensions_in_package(&mut desugared)?;
    Ok(desugared.to_string())
}
