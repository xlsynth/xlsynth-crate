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
    use crate::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth::IrValue;

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

        // Sanity-check equivalence on a small, representative sample set.
        let mut p0 = ir_parser::Parser::new(ir_text);
        let pkg0 = p0.parse_and_validate_package().expect("parse/validate");
        let f0 = pkg0.get_fn("cone").unwrap();
        let f1 = f;

        let ys: [u64; 4] = [0, 1, 2, 255];
        for a in 0u64..=1u64 {
            for x in 0u64..=255u64 {
                for y in ys {
                    let args = [
                        IrValue::make_ubits(1, a).unwrap(),
                        IrValue::make_ubits(8, x).unwrap(),
                        IrValue::make_ubits(8, y).unwrap(),
                    ];
                    let got0 = match eval_fn(f0, &args) {
                        FnEvalResult::Success(s) => s.value.to_bool().unwrap(),
                        FnEvalResult::Failure(e) => panic!("unexpected eval failure: {:?}", e),
                    };
                    let got1 = match eval_fn(f1, &args) {
                        FnEvalResult::Success(s) => s.value.to_bool().unwrap(),
                        FnEvalResult::Failure(e) => panic!("unexpected eval failure: {:?}", e),
                    };
                    assert_eq!(got0, got1, "mismatch at a={} x={} y={}", a, x, y);
                }
            }
        }
    }
}
