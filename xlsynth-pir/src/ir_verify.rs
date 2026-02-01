// SPDX-License-Identifier: Apache-2.0

//! Lightweight IR validators for common invariants useful in debugging emitted
//! IR.

use std::collections::HashSet;

use crate::ir::{self, Fn, NodePayload, Package, PackageMember, Type};
use crate::ir_deduce::{deduce_result_type, deduce_result_type_with};
use crate::ir_utils::operands;

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
    let check = |nr: ir::NodeRef, ctx: &str| -> Result<(), String> {
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
            NodePayload::ArraySlice { array, start, .. } => {
                check(*array, &format!("node {} array_slice.array", i))?;
                check(*start, &format!("node {} array_slice.start", i))?;
            }
            NodePayload::Binop(_, a, b) => {
                check(*a, &format!("node {} binop.lhs", i))?;
                check(*b, &format!("node {} binop.rhs", i))?;
            }
            NodePayload::ExtCarryOut { lhs, rhs, c_in } => {
                check(*lhs, &format!("node {} ext_carry_out.lhs", i))?;
                check(*rhs, &format!("node {} ext_carry_out.rhs", i))?;
                check(*c_in, &format!("node {} ext_carry_out.c_in", i))?;
            }
            NodePayload::ExtPrioEncode { arg, lsb_prio: _ } => {
                check(*arg, &format!("node {} ext_prio_encode.arg", i))?;
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
            NodePayload::RegisterRead { .. } => {}
            NodePayload::RegisterWrite {
                arg,
                load_enable,
                reset,
                ..
            } => {
                check(*arg, &format!("node {} register_write.arg", i))?;
                if let Some(le) = load_enable {
                    check(*le, &format!("node {} register_write.load_enable", i))?;
                }
                if let Some(rst) = reset {
                    check(*rst, &format!("node {} register_write.reset", i))?;
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

/// Verifies that for all nodes we can deduce, the node's `ty` matches the
/// type anticipated by the deduction routine.
pub fn verify_fn_types_agree_with_deduction(f: &Fn) -> Result<(), String> {
    for (i, node) in f.nodes.iter().enumerate() {
        // Gather operand types in operand order.
        let op_refs = operands(&node.payload);
        let mut op_types: Vec<Type> = Vec::with_capacity(op_refs.len());
        for nr in op_refs.iter() {
            op_types.push(f.get_node(*nr).ty.clone());
        }
        // Ask the deducer what the result type should be for this payload.
        match deduce_result_type(&node.payload, &op_types).map_err(|e| e.to_string())? {
            Some(deduced) => {
                if deduced != node.ty {
                    return Err(format!(
                        "type mismatch for node {} ({}): deduced {} vs actual {}",
                        i,
                        node.payload.get_operator(),
                        deduced,
                        node.ty
                    ));
                }
            }
            None => {
                // Deduction not implemented for this payload; skip.
            }
        }
    }
    Ok(())
}

/// Verifies that all node text IDs are unique across the entire package
/// (functions and blocks), matching upstream XLS expectations.
pub fn verify_package_unique_node_ids(pkg: &Package) -> Result<(), String> {
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for m in pkg.members.iter() {
        let f: &Fn = match m {
            PackageMember::Function(f) => f,
            PackageMember::Block { func, .. } => func,
        };
        for (idx, n) in f.nodes.iter().enumerate() {
            if !seen.insert(n.text_id) {
                return Err(format!(
                    "duplicate node id={} found at node index {} in function '{}'",
                    n.text_id, idx, f.name
                ));
            }
        }
    }
    Ok(())
}

/// Like `verify_fn_types_agree_with_deduction`, but allows providing a
/// resolver for callee return types so that `invoke` result types can be
/// checked when package context is available.
pub fn verify_fn_types_agree_with_deduction_in_pkg(f: &Fn, pkg: &Package) -> Result<(), String> {
    for (i, node) in f.nodes.iter().enumerate() {
        let op_refs = operands(&node.payload);
        let mut op_types: Vec<Type> = Vec::with_capacity(op_refs.len());
        for nr in op_refs.iter() {
            op_types.push(f.get_node(*nr).ty.clone());
        }
        let resolver =
            |name: &str| -> Option<Type> { pkg.get_fn_type(name).map(|ft| ft.return_type) };
        match deduce_result_type_with(&node.payload, &op_types, resolver)
            .map_err(|e| e.to_string())?
        {
            Some(deduced) => {
                if deduced != node.ty {
                    return Err(format!(
                        "type mismatch for node {} ({}): deduced {} vs actual {}",
                        i,
                        node.payload.get_operator(),
                        deduced,
                        node.ty
                    ));
                }
            }
            None => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_parser::Parser;

    #[test]
    fn type_mismatch_on_add_is_flagged() {
        let ir = r#"
fn foo(x: bits[8] id=1) -> bits[16] {
  add.2: bits[16] = add(x, x, id=2)
  ret identity.3: bits[16] = identity(add.2, id=3)
}
"#;
        let mut parser = Parser::new(ir);
        let f = parser.parse_fn().expect("parse fn");
        // add node is declared bits[16] but deduction expects bits[8]; should error.
        assert!(verify_fn_types_agree_with_deduction(&f).is_err());
    }

    #[test]
    fn type_mismatch_on_invoke_is_flagged_with_pkg_context() {
        let ir = r#"
package test

fn callee(x: bits[1] id=1) -> (bits[1], bits[1]) {
  ret tuple.3: (bits[1], bits[1]) = tuple(x, x, id=3)
}

fn foo(x: bits[1] id=1) -> bits[1] {
  invoke.2: bits[1] = invoke(x, to_apply=callee, id=2)
  ret identity.3: bits[1] = identity(invoke.2, id=3)
}
"#;
        let mut parser = Parser::new(ir);
        let pkg = parser.parse_package().expect("parse package");
        let foo_fn = pkg
            .members
            .iter()
            .find_map(|m| match m {
                PackageMember::Function(f) if f.name == "foo" => Some(f.clone()),
                _ => None,
            })
            .expect("find foo");
        assert!(verify_fn_types_agree_with_deduction_in_pkg(&foo_fn, &pkg).is_err());
    }
}
