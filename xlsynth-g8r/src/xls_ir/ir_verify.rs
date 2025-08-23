// SPDX-License-Identifier: Apache-2.0

//! Lightweight IR validators for common invariants useful in debugging emitted
//! IR.

use std::collections::HashSet;

use crate::xls_ir::ir::{Fn, NodePayload};

/// Verifies that all node text IDs within a function are unique.
pub fn verify_fn_unique_node_ids(f: &Fn) -> Result<(), String> {
    let mut seen: HashSet<usize> = HashSet::new();
    for (idx, n) in f.nodes.iter().enumerate() {
        if !seen.insert(n.text_id) {
            return Err(format!(
                "duplicate node id={} found at node index {} in function '{}'",
                n.text_id, idx, f.name
            ));
        }
    }
    Ok(())
}

/// Verifies that all NodeRef indices referenced by payloads are within bounds.
pub fn verify_fn_operand_indices_in_bounds(f: &Fn) -> Result<(), String> {
    let n = f.nodes.len();
    let check = |nr: crate::xls_ir::ir::NodeRef, ctx: &str| -> Result<(), String> {
        if nr.index >= n {
            return Err(format!(
                "operand index {} out of bounds in {}; function '{}' has {} nodes",
                nr.index, ctx, f.name, n
            ));
        }
        Ok(())
    };
    for (i, node) in f.nodes.iter().enumerate() {
        match &node.payload {
            NodePayload::Nil | NodePayload::GetParam(_) | NodePayload::Literal(_) => {}
            NodePayload::Tuple(nodes)
            | NodePayload::Array(nodes)
            | NodePayload::AfterAll(nodes)
            | NodePayload::Nary(_, nodes) => {
                for r in nodes.iter() {
                    check(*r, &format!("node {} payload list", i))?;
                }
            }
            NodePayload::TupleIndex { tuple, .. }
            | NodePayload::Unop(_, tuple)
            | NodePayload::Decode { arg: tuple, .. }
            | NodePayload::Encode { arg: tuple }
            | NodePayload::OneHot { arg: tuple, .. }
            | NodePayload::BitSlice { arg: tuple, .. } => {
                check(*tuple, &format!("node {} single", i))?;
            }
            NodePayload::Binop(_, a, b) => {
                check(*a, &format!("node {} binop.lhs", i))?;
                check(*b, &format!("node {} binop.rhs", i))?;
            }
            NodePayload::SignExt { arg, .. } | NodePayload::ZeroExt { arg, .. } => {
                check(*arg, &format!("node {} ext.arg", i))?;
            }
            NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                ..
            } => {
                check(*array, &format!("node {} array_update.array", i))?;
                check(*value, &format!("node {} array_update.value", i))?;
                for r in indices.iter() {
                    check(*r, &format!("node {} array_update.index", i))?;
                }
            }
            NodePayload::ArrayIndex { array, indices, .. } => {
                check(*array, &format!("node {} array_index.array", i))?;
                for r in indices.iter() {
                    check(*r, &format!("node {} array_index.index", i))?;
                }
            }
            NodePayload::DynamicBitSlice { arg, start, .. } => {
                check(*arg, &format!("node {} dbs.arg", i))?;
                check(*start, &format!("node {} dbs.start", i))?;
            }
            NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                check(*arg, &format!("node {} bsu.arg", i))?;
                check(*start, &format!("node {} bsu.start", i))?;
                check(*update_value, &format!("node {} bsu.update_value", i))?;
            }
            NodePayload::Assert {
                token, activate, ..
            } => {
                check(*token, &format!("node {} assert.token", i))?;
                check(*activate, &format!("node {} assert.activate", i))?;
            }
            NodePayload::Trace {
                token,
                activated,
                operands,
                ..
            } => {
                check(*token, &format!("node {} trace.token", i))?;
                check(*activated, &format!("node {} trace.activated", i))?;
                for r in operands.iter() {
                    check(*r, &format!("node {} trace.operand", i))?;
                }
            }
            NodePayload::Invoke { operands, .. } => {
                for r in operands.iter() {
                    check(*r, &format!("node {} invoke.operand", i))?;
                }
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            }
            | NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                check(*selector, &format!("node {} sel.selector", i))?;
                for r in cases.iter() {
                    check(*r, &format!("node {} sel.case", i))?;
                }
                if let Some(d) = default {
                    check(*d, &format!("node {} sel.default", i))?;
                }
            }
            NodePayload::OneHotSel { selector, cases } => {
                check(*selector, &format!("node {} one_hot_sel.selector", i))?;
                for r in cases.iter() {
                    check(*r, &format!("node {} one_hot_sel.case", i))?;
                }
            }
            NodePayload::CountedFor {
                init,
                invariant_args,
                ..
            } => {
                check(*init, &format!("node {} counted_for.init", i))?;
                for r in invariant_args.iter() {
                    check(*r, &format!("node {} counted_for.invariant", i))?;
                }
            }
            NodePayload::Cover { predicate, .. } => {
                check(*predicate, &format!("node {} cover.predicate", i))?;
            }
        }
    }
    Ok(())
}
