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

use crate::ir::{Binop, Fn, FnInPkgMut, NodePayload, NodeRef, Package, Type, Unop};
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

fn push_node(f: &mut FnInPkgMut<'_>, ty: Type, payload: NodePayload) -> NodeRef {
    f.push_node(ty, payload)
}

fn desugar_ext_carry_out_in_fn(f: &mut FnInPkgMut<'_>) -> Result<bool, DesugarError> {
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

fn desugar_ext_prio_encode_in_fn(f: &mut FnInPkgMut<'_>) -> Result<bool, DesugarError> {
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

/// Desugars extension ops within a package-backed function mutation context.
///
/// This function also normalizes the node list into a valid topological order.
pub fn desugar_extensions_in_fn(f: &mut FnInPkgMut<'_>) -> Result<(), DesugarError> {
    let _changed = desugar_ext_carry_out_in_fn(f)? | desugar_ext_prio_encode_in_fn(f)?;
    compact_and_toposort_in_place(f.function_mut()).map_err(DesugarError::new)?;
    Ok(())
}

/// Desugars extension ops within `pkg` into upstream-compatible PIR operations.
pub fn desugar_extensions_in_package(pkg: &mut Package) -> Result<(), DesugarError> {
    for member_index in 0..pkg.members.len() {
        let mut fn_in_pkg = pkg
            .fn_in_pkg_mut(member_index)
            .expect("desugar_extensions_in_package: member index should be valid");
        desugar_extensions_in_fn(&mut fn_in_pkg)?;
    }
    pkg.sync_next_text_id();
    Ok(())
}

/// Emits upstream-compatible XLS IR text for `pkg` by desugaring extensions
/// first.
pub fn emit_package_as_xls_ir_text(pkg: &Package) -> Result<String, DesugarError> {
    let mut desugared = pkg.clone();
    desugar_extensions_in_package(&mut desugared)?;
    Ok(desugared.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{FileTable, MemberType, PackageMember};
    use crate::ir_parser;
    use crate::ir_validate;

    fn parse_sample_package() -> Package {
        let ir_text = r#"package sample

top fn f(x: bits[4] id=1, y: bits[4] id=2, cin: bits[1] id=3) -> (bits[1], bits[2]) {
  carry: bits[1] = ext_carry_out(x, y, cin, id=4)
  enc_in: bits[2] = bit_slice(x, start=0, width=2, id=5)
  enc: bits[2] = ext_prio_encode(enc_in, lsb_prio=true, id=6)
  ret out: (bits[1], bits[2]) = tuple(carry, enc, id=7)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        parser.parse_and_validate_package().unwrap()
    }

    fn fn_has_extension_ops(f: &Fn) -> bool {
        f.nodes.iter().any(|n| n.payload.is_extension_op())
    }

    #[test]
    fn desugar_extensions_in_fn_advances_package_next_text_id_and_validates() {
        let mut pkg = parse_sample_package();
        let before_next = pkg.peek_next_text_id();
        {
            let mut f = pkg.get_top_fn_in_pkg_mut().unwrap();
            desugar_extensions_in_fn(&mut f).unwrap();
            assert!(
                !fn_has_extension_ops(f.function()),
                "package-backed desugar should eliminate extension ops"
            );
        }
        assert!(
            pkg.peek_next_text_id() > before_next,
            "package-backed desugar should allocate fresh text ids"
        );
        assert_eq!(pkg.peek_next_text_id(), pkg.recompute_next_unused_text_id());
        ir_validate::validate_package(&pkg).unwrap();
    }

    #[test]
    fn explicit_synthetic_package_desugar_matches_package_backed_desugar() {
        let pkg = parse_sample_package();
        let mut standalone = pkg.get_top_fn().unwrap().clone();
        let mut standalone_pkg = Package::new(
            "standalone".to_string(),
            FileTable::new(),
            vec![PackageMember::Function(standalone)],
            Some(("f".to_string(), MemberType::Function)),
        );
        {
            let mut f = standalone_pkg.get_top_fn_in_pkg_mut().unwrap();
            desugar_extensions_in_fn(&mut f).unwrap();
        }
        standalone = standalone_pkg.get_top_fn().unwrap().clone();
        assert!(
            !fn_has_extension_ops(&standalone),
            "explicit synthetic-package desugar should eliminate extension ops"
        );

        let mut pkg_backed = pkg.clone();
        {
            let mut f = pkg_backed.get_top_fn_in_pkg_mut().unwrap();
            desugar_extensions_in_fn(&mut f).unwrap();
        }
        let expected = pkg_backed.get_top_fn().unwrap().clone();
        assert_eq!(standalone.to_string(), expected.to_string());
        ir_validate::validate_package(&standalone_pkg).unwrap();
    }
}
