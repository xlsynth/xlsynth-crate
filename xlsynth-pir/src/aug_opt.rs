// SPDX-License-Identifier: Apache-2.0

//! A small, local "augmented optimizer" that runs lightweight PIR rewrites in a
//! co-recursive loop with the upstream XLS optimizer (`xlsynth::optimize_ir`).
//!
//! Design goals:
//! - **Basis ops only**: rewrites must not introduce PIR extension ops, because
//!   libxls optimization does not understand them.
//! - **Bounded effort**: intended as a fast front-end; keep rounds small.
//! - **Deterministic**: stable iteration order and stable outputs.

use crate::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use crate::ir_parser;
use crate::ir_utils;
use xlsynth::IrValue;

#[derive(Debug, Clone, Copy)]
pub struct AugOptOptions {
    pub enable: bool,
    pub rounds: usize,
}

impl Default for AugOptOptions {
    fn default() -> Self {
        Self {
            enable: false,
            rounds: 1,
        }
    }
}

pub fn run_aug_opt_over_ir_text(
    ir_text: &str,
    top: Option<&str>,
    options: AugOptOptions,
) -> Result<String, String> {
    if !options.enable {
        return Ok(ir_text.to_string());
    }

    // Basis-only contract: aug-opt does not support PIR extension ops.
    // Fail fast with a clear error before libxls parsing.
    verify_no_extension_ops_in_ir_text(ir_text)?;

    // Start by letting libxls do its normal canonicalization.
    let mut cur_pkg = xlsynth::IrPackage::parse_ir(ir_text, None)
        .map_err(|e| format!("aug_opt: xlsynth parse_ir failed: {e}"))?;
    let top_name = top
        .ok_or_else(|| "aug_opt: top is required".to_string())?
        .to_string();
    cur_pkg
        .set_top_by_name(&top_name)
        .map_err(|e| format!("aug_opt: set_top_by_name('{top_name}') failed: {e}"))?;

    // One initial opt pass before the co-recursive rounds.
    cur_pkg = xlsynth::optimize_ir(&cur_pkg, &top_name)
        .map_err(|e| format!("aug_opt: optimize_ir initial failed: {e}"))?;

    for _round in 0..options.rounds {
        let cur_text = cur_pkg.to_string();

        // Parse with PIR, apply basis-only rewrites to the top function.
        let mut pir_parser = ir_parser::Parser::new(&cur_text);
        let mut pir_pkg = pir_parser
            .parse_and_validate_package()
            .map_err(|e| format!("aug_opt: PIR parse/validate failed: {e}"))?;

        let top_fn = pir_pkg
            .get_fn(&top_name)
            .ok_or_else(|| format!("aug_opt: PIR package missing top fn '{top_name}'"))?
            .clone();

        let rewritten_top = apply_basis_rewrites_to_fn(&top_fn);

        // Swap the rewritten top back into the PIR package.
        for member in pir_pkg.members.iter_mut() {
            match member {
                ir::PackageMember::Function(f) if f.name == top_name => {
                    *f = rewritten_top.clone();
                }
                ir::PackageMember::Block { func, .. } if func.name == top_name => {
                    *func = rewritten_top.clone();
                }
                _ => {}
            }
        }

        // Verify we did not introduce extension ops (basis-only contract).
        verify_no_extension_ops_in_package(&pir_pkg)?;

        // Emit basis XLS IR text and hand it back to libxls.
        let lowered_text = pir_pkg.to_string();

        let mut next_pkg = xlsynth::IrPackage::parse_ir(&lowered_text, None)
            .map_err(|e| format!("aug_opt: xlsynth parse_ir (post-rewrite) failed: {e}"))?;
        next_pkg
            .set_top_by_name(&top_name)
            .map_err(|e| format!("aug_opt: set_top_by_name('{top_name}') failed: {e}"))?;
        cur_pkg = xlsynth::optimize_ir(&next_pkg, &top_name)
            .map_err(|e| format!("aug_opt: optimize_ir post-rewrite failed: {e}"))?;
    }

    Ok(cur_pkg.to_string())
}

fn verify_no_extension_ops_in_ir_text(ir_text: &str) -> Result<(), String> {
    let mut pir_parser = ir_parser::Parser::new(ir_text);
    let pir_pkg = pir_parser
        .parse_and_validate_package()
        .map_err(|e| format!("aug_opt: PIR parse/validate failed: {e}"))?;
    verify_no_extension_ops_in_package(&pir_pkg)
}

fn verify_no_extension_ops_in_package(pkg: &ir::Package) -> Result<(), String> {
    for member in &pkg.members {
        match member {
            ir::PackageMember::Function(f) => verify_no_extension_ops_in_fn(f)?,
            ir::PackageMember::Block { func, .. } => verify_no_extension_ops_in_fn(func)?,
        }
    }
    Ok(())
}

fn verify_no_extension_ops_in_fn(f: &ir::Fn) -> Result<(), String> {
    for node in &f.nodes {
        if node.payload.is_extension_op() {
            return Err(format!(
                "aug_opt: basis-only: found extension op in fn '{}': text_id={} payload={:?}",
                f.name, node.text_id, node.payload
            ));
        }
    }
    Ok(())
}

fn apply_basis_rewrites_to_fn(f: &ir::Fn) -> ir::Fn {
    let mut cloned = f.clone();
    let _rewrites = rewrite_guarded_sel_ne_literal1_nor(&mut cloned);
    let _rewrites = rewrite_lsb_of_shll_via_shift_is_zero(&mut cloned);
    let _rewrites = rewrite_and_nor_self_to_zero(&mut cloned);
    // Ensure textual IR is defs-before-uses by reordering body nodes into a
    // topological order (while preserving PIR layout invariants). This makes
    // it safe for rewrites to append new nodes.
    ir_utils::compact_and_toposort_in_place(&mut cloned)
        .expect("aug_opt: compact_and_toposort_in_place failed");
    cloned
}

fn next_text_id(f: &ir::Fn) -> usize {
    f.nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn push_node(f: &mut ir::Fn, ty: Type, payload: NodePayload) -> NodeRef {
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

fn push_ubits_literal(f: &mut ir::Fn, w: usize, v: u64) -> NodeRef {
    let ty = Type::Bits(w);
    let lit = IrValue::make_ubits(w, v).expect("ubits literal construction should succeed");
    push_node(f, ty, NodePayload::Literal(lit))
}

fn is_ubits_literal_1_of_width(f: &ir::Fn, nr: NodeRef, w: usize) -> bool {
    let node = f.get_node(nr);
    if node.ty != Type::Bits(w) {
        return false;
    }
    let NodePayload::Literal(v) = &node.payload else {
        return false;
    };
    v.to_u64().ok() == Some(1)
}

/// Rewrite:
///
/// `and(..., a, ..., nor(..., a, ...), ...)` → `literal(0)`
///
/// In particular, this covers:
/// - `and(a, nor(a, b))`
/// - `and(a, nor(b, a))`
///
/// Valid for any bit width: \(a \,\&\, \neg(a \lor b) = 0\).
fn rewrite_and_nor_self_to_zero(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for and_index in 0..f.nodes.len() {
        let NodePayload::Nary(NaryOp::And, operands) = f.nodes[and_index].payload.clone() else {
            continue;
        };
        if operands.len() < 2 {
            continue;
        }

        let w = f.nodes[and_index].ty.bit_count();
        if w == 0 {
            continue;
        }

        let mut matched = false;
        for &a in &operands {
            if *f.get_node_ty(a) != Type::Bits(w) {
                continue;
            }
            for &other in &operands {
                if other.index == a.index {
                    continue;
                }
                let other_payload = f.get_node(other).payload.clone();

                // Case 1: `nor(..., a, ...)` (support n-ary NOR as well).
                if let NodePayload::Nary(NaryOp::Nor, nor_ops) = &other_payload {
                    if nor_ops.len() >= 2 && nor_ops.iter().any(|&x| x.index == a.index) {
                        matched = true;
                        break;
                    }
                }

                // Case 2: `not(or(..., a, ...))` (common canonicalization of NOR).
                if let NodePayload::Unop(Unop::Not, inner) = &other_payload {
                    if let NodePayload::Nary(NaryOp::Or, or_ops) = &f.get_node(*inner).payload {
                        if or_ops.len() >= 2 && or_ops.iter().any(|&x| x.index == a.index) {
                            matched = true;
                            break;
                        }
                    }
                }

                if matched {
                    matched = true;
                    break;
                }
            }
            if matched {
                break;
            }
        }

        if !matched {
            continue;
        }

        // Rewrite the AND node in-place to a zero literal of matching width.
        let lit0 = IrValue::make_ubits(w, 0).expect("zero literal should always fit");
        f.nodes[and_index].payload = NodePayload::Literal(lit0);
        rewrites += 1;
    }

    rewrites
}

/// Rewrite:
///
/// `bit_slice(shll(x, s), start=0, width=1)`
///   →
/// `and(bit_slice(x, start=0, width=1), eq(s, literal(0)))`
///
/// Intuition: the LSB of a left-shifted value is preserved iff the shift amount
/// is zero; otherwise it is forced to zero.
fn rewrite_lsb_of_shll_via_shift_is_zero(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for slice_index in 0..f.nodes.len() {
        let NodePayload::BitSlice { arg, start, width } = f.nodes[slice_index].payload.clone()
        else {
            continue;
        };
        if start != 0 || width != 1 {
            continue;
        }
        if f.nodes[slice_index].ty != Type::Bits(1) {
            continue;
        }

        let (x, s) = match f.get_node(arg).payload.clone() {
            NodePayload::Binop(Binop::Shll, x, s) => (x, s),
            _ => continue,
        };
        let w_x = f.get_node_ty(x).bit_count();
        if w_x == 0 {
            continue;
        }

        // Build: bit_slice(x, start=0, width=1)
        let slice_x = push_node(
            f,
            Type::Bits(1),
            NodePayload::BitSlice {
                arg: x,
                start: 0,
                width: 1,
            },
        );

        // Build: eq(s, 0) with a literal of the same width as s.
        let w_s = f.get_node_ty(s).bit_count();
        if w_s == 0 {
            continue;
        }
        let lit0 = push_ubits_literal(f, w_s, 0);
        let eq_s0 = push_node(f, Type::Bits(1), NodePayload::Binop(Binop::Eq, s, lit0));

        // Rewrite the bit_slice node in-place.
        f.nodes[slice_index].payload = NodePayload::Nary(NaryOp::And, vec![slice_x, eq_s0]);
        f.nodes[slice_index].ty = Type::Bits(1);

        rewrites += 1;
    }

    rewrites
}

/// Rewrite:
///
/// `nor(s, ne(sel(selector=s, cases=[x, y]), literal(1)))`
///   →
/// `and(not(s), eq(x, literal(1)))`
///
/// Preconditions (kept intentionally narrow for the first cut):
/// - `s: bits[1]`
/// - `sel` has exactly 2 cases and no default
/// - comparison is against literal(1) of the same width as the select result
fn rewrite_guarded_sel_ne_literal1_nor(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for nor_index in 0..f.nodes.len() {
        let NodePayload::Nary(NaryOp::Nor, operands) = f.nodes[nor_index].payload.clone() else {
            continue;
        };
        if operands.len() != 2 {
            continue;
        }

        // Try both operand orders: (s, ne(...)) or (ne(...), s).
        let candidates = [(operands[0], operands[1]), (operands[1], operands[0])];
        let mut matched: Option<(NodeRef, NodeRef, NodeRef, NodeRef, NodeRef)> = None;

        for (s, ne_nr) in candidates {
            if *f.get_node_ty(s) != Type::Bits(1) {
                continue;
            }
            let NodePayload::Binop(Binop::Ne, sel_nr, lit_nr) = &f.get_node(ne_nr).payload else {
                continue;
            };
            let NodePayload::Sel {
                selector,
                cases,
                default,
            } = &f.get_node(*sel_nr).payload
            else {
                continue;
            };
            if default.is_some() || cases.len() != 2 {
                continue;
            }
            if *selector != s {
                continue;
            }
            let w = f.get_node_ty(*sel_nr).bit_count();
            if w == 0 || *f.get_node_ty(cases[0]) != Type::Bits(w) {
                continue;
            }
            if !is_ubits_literal_1_of_width(f, *lit_nr, w) {
                continue;
            }

            matched = Some((s, ne_nr, *sel_nr, *lit_nr, cases[0]));
            break;
        }

        let Some((s, _ne_nr, _sel_nr, lit1_nr, x_case0)) = matched else {
            continue;
        };

        // Build: not(s)
        let not_s = push_node(f, Type::Bits(1), NodePayload::Unop(Unop::Not, s));

        // Build: eq(x, literal(1)) using the matched literal node to keep graphs
        // stable.
        let eq_x = push_node(
            f,
            Type::Bits(1),
            NodePayload::Binop(Binop::Eq, x_case0, lit1_nr),
        );

        // Rewrite the original nor node in-place to preserve indices.
        f.nodes[nor_index].payload = NodePayload::Nary(NaryOp::And, vec![not_s, eq_x]);
        f.nodes[nor_index].ty = Type::Bits(1);

        rewrites += 1;
    }

    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::quickcheck_ir_text_fn_equivalence_ubits_le64;

    #[test]
    fn aug_opt_rewrites_guarded_sel_ne1_nor_and_opt_dces_sel() {
        let ir_text = r#"package bool_cone

top fn cone(leaf_22: bits[1] id=1, leaf_36: bits[8] id=2, leaf_37: bits[8] id=3) -> bits[1] {
  sel.4: bits[8] = sel(leaf_22, cases=[leaf_36, leaf_37], id=4)
  literal.5: bits[8] = literal(value=1, id=5)
  ne.6: bits[1] = ne(sel.4, literal.5, id=6)
  ret nor.7: bits[1] = nor(leaf_22, ne.6, id=7)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
            },
        )
        .expect("aug opt");

        // After aug-opt + libxls opt, the top function should not contain a Sel.
        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");
        let has_sel = f
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::Sel { .. }));
        assert!(!has_sel, "expected sel to be DCE'd; got:\n{}", out_text);

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[1, 8, 8],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_rewrites_lsb_of_shll_to_shift_is_zero() {
        let ir_text = r#"package shll_lsb

top fn f(x: bits[8] id=1, s: bits[4] id=2) -> bits[1] {
  shll.3: bits[8] = shll(x, s, id=3)
  ret bit_slice.4: bits[1] = bit_slice(shll.3, start=0, width=1, id=4)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("f"),
            AugOptOptions {
                enable: true,
                rounds: 1,
            },
        )
        .expect("aug opt");

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "f",
            &[8, 4],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_rewrites_and_nor_self_to_zero_both_orders() {
        let ir_text_0 = r#"package and_nor_self

top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  nor.3: bits[8] = nor(a, b, id=3)
  ret and.4: bits[8] = and(a, nor.3, id=4)
}
"#;

        // Directly test the local rewrite (no libxls involvement) by rewriting
        // the parsed PIR function and checking the return node becomes literal(0).
        let mut p0 = ir_parser::Parser::new(ir_text_0);
        let pkg0 = p0.parse_and_validate_package().expect("parse/validate");
        let mut f0 = pkg0.get_fn("f").unwrap().clone();
        let rewrites0 = rewrite_and_nor_self_to_zero(&mut f0);
        assert_eq!(rewrites0, 1);
        let ret = f0.ret_node_ref.expect("ret");
        let NodePayload::Literal(v) = &f0.get_node(ret).payload else {
            panic!(
                "expected ret payload to be Literal(0), got {:?}",
                f0.get_node(ret).payload
            );
        };
        assert_eq!(v.to_u64().ok(), Some(0));

        let ir_text_1 = r#"package and_nor_self

top fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  nor.3: bits[8] = nor(b, a, id=3)
  ret and.4: bits[8] = and(a, nor.3, id=4)
}
"#;
        let mut p1 = ir_parser::Parser::new(ir_text_1);
        let pkg1 = p1.parse_and_validate_package().expect("parse/validate");
        let mut f1 = pkg1.get_fn("f").unwrap().clone();
        let rewrites1 = rewrite_and_nor_self_to_zero(&mut f1);
        assert_eq!(rewrites1, 1);
        let ret = f1.ret_node_ref.expect("ret");
        let NodePayload::Literal(v) = &f1.get_node(ret).payload else {
            panic!(
                "expected ret payload to be Literal(0), got {:?}",
                f1.get_node(ret).payload
            );
        };
        assert_eq!(v.to_u64().ok(), Some(0));
    }

    #[test]
    fn aug_opt_rewrites_and_nor_self_to_zero_nary_and_not_or_forms() {
        // N-ary NOR case.
        let ir_text_nary_nor = r#"package and_nor_self

top fn f(a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3) -> bits[1] {
  nor.4: bits[1] = nor(b, a, c, id=4)
  ret and.5: bits[1] = and(a, nor.4, id=5)
}
"#;
        let mut p0 = ir_parser::Parser::new(ir_text_nary_nor);
        let pkg0 = p0.parse_and_validate_package().expect("parse/validate");
        let mut f0 = pkg0.get_fn("f").unwrap().clone();
        let rewrites0 = rewrite_and_nor_self_to_zero(&mut f0);
        assert_eq!(rewrites0, 1);
        let ret = f0.ret_node_ref.expect("ret");
        let NodePayload::Literal(v) = &f0.get_node(ret).payload else {
            panic!(
                "expected ret payload to be Literal(0), got {:?}",
                f0.get_node(ret).payload
            );
        };
        assert_eq!(v.to_u64().ok(), Some(0));

        // not(or(...)) canonicalization form.
        let ir_text_not_or = r#"package and_nor_self

top fn f(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  or.3: bits[1] = or(a, b, id=3)
  not.4: bits[1] = not(or.3, id=4)
  ret and.5: bits[1] = and(a, not.4, id=5)
}
"#;
        let mut p1 = ir_parser::Parser::new(ir_text_not_or);
        let pkg1 = p1.parse_and_validate_package().expect("parse/validate");
        let mut f1 = pkg1.get_fn("f").unwrap().clone();
        let rewrites1 = rewrite_and_nor_self_to_zero(&mut f1);
        assert_eq!(rewrites1, 1);
        let ret = f1.ret_node_ref.expect("ret");
        let NodePayload::Literal(v) = &f1.get_node(ret).payload else {
            panic!(
                "expected ret payload to be Literal(0), got {:?}",
                f1.get_node(ret).payload
            );
        };
        assert_eq!(v.to_u64().ok(), Some(0));
    }
}
