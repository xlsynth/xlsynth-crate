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
    /// When true, run an initial libxls `optimize_ir` pass before the
    /// co-recursive rounds.
    pub run_xlsynth_opt_before: bool,
    /// When true, run libxls `optimize_ir` after the PIR rewrite in each round.
    pub run_xlsynth_opt_after: bool,
}

#[derive(Debug, Clone)]
pub struct AugOptRunResult {
    pub output_text: String,
    pub total_rewrites: usize,
}

impl AugOptRunResult {
    pub fn rewrote(&self) -> bool {
        self.total_rewrites > 0
    }
}

impl Default for AugOptOptions {
    fn default() -> Self {
        Self {
            enable: false,
            rounds: 1,
            run_xlsynth_opt_before: true,
            run_xlsynth_opt_after: true,
        }
    }
}

pub fn run_aug_opt_over_ir_text(
    ir_text: &str,
    top: Option<&str>,
    options: AugOptOptions,
) -> Result<String, String> {
    run_aug_opt_over_ir_text_with_stats(ir_text, top, options).map(|result| result.output_text)
}

pub fn run_aug_opt_over_ir_text_with_stats(
    ir_text: &str,
    top: Option<&str>,
    options: AugOptOptions,
) -> Result<AugOptRunResult, String> {
    if !options.enable {
        return Ok(AugOptRunResult {
            output_text: ir_text.to_string(),
            total_rewrites: 0,
        });
    }

    // Basis-only contract: aug-opt does not support PIR extension ops.
    // Fail fast with a clear error before libxls parsing.
    verify_no_extension_ops_in_ir_text(ir_text)?;

    let top_name = top
        .ok_or_else(|| "aug_opt: top is required".to_string())?
        .to_string();

    // Fast path: run PIR basis rewrites without libxls at all. This is useful
    // for debugging whether a PIR pattern matcher is firing.
    if !options.run_xlsynth_opt_before && !options.run_xlsynth_opt_after {
        let mut pir_parser = ir_parser::Parser::new(ir_text);
        let mut pir_pkg = pir_parser
            .parse_and_validate_package()
            .map_err(|e| format!("aug_opt: PIR parse/validate failed: {e}"))?;

        let top_fn = pir_pkg
            .get_fn(&top_name)
            .ok_or_else(|| format!("aug_opt: PIR package missing top fn '{top_name}'"))?
            .clone();
        let (rewritten_top, rewrites) = apply_basis_rewrites_to_fn(&top_fn);

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
        verify_no_extension_ops_in_package(&pir_pkg)?;

        return Ok(AugOptRunResult {
            output_text: pir_pkg.to_string(),
            total_rewrites: rewrites,
        });
    }

    // Start by letting libxls do its normal canonicalization.
    let mut cur_pkg = xlsynth::IrPackage::parse_ir(ir_text, None)
        .map_err(|e| format!("aug_opt: xlsynth parse_ir failed: {e}"))?;
    cur_pkg
        .set_top_by_name(&top_name)
        .map_err(|e| format!("aug_opt: set_top_by_name('{top_name}') failed: {e}"))?;

    if options.run_xlsynth_opt_before {
        // One initial opt pass before the co-recursive rounds.
        cur_pkg = xlsynth::optimize_ir(&cur_pkg, &top_name)
            .map_err(|e| format!("aug_opt: optimize_ir initial failed: {e}"))?;
    }

    let mut total_rewrites = 0usize;
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

        let (rewritten_top, rewrites_in_round) = apply_basis_rewrites_to_fn(&top_fn);
        total_rewrites = total_rewrites.saturating_add(rewrites_in_round);

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
        if options.run_xlsynth_opt_after {
            cur_pkg = xlsynth::optimize_ir(&next_pkg, &top_name)
                .map_err(|e| format!("aug_opt: optimize_ir post-rewrite failed: {e}"))?;
        } else {
            cur_pkg = next_pkg;
        }
    }

    // Final PIR rewrite pass: after libxls has canonicalized, apply basis-only
    // rewrites one more time. This lets us match patterns that are only exposed
    // after optimization (e.g. zero-tests over a mux between `x` and `neg(x)`).
    //
    // We intentionally do not run another libxls pass after this; the goal is
    // to preserve the rewritten shape for downstream tools like g8r.
    if options.run_xlsynth_opt_after {
        let cur_text = cur_pkg.to_string();
        let mut pir_parser = ir_parser::Parser::new(&cur_text);
        let mut pir_pkg = pir_parser
            .parse_and_validate_package()
            .map_err(|e| format!("aug_opt: PIR parse/validate failed (final): {e}"))?;

        let top_fn = pir_pkg
            .get_fn(&top_name)
            .ok_or_else(|| format!("aug_opt: PIR package missing top fn '{top_name}' (final)"))?
            .clone();

        let (rewritten_top, rewrites_in_final) = apply_basis_rewrites_to_fn(&top_fn);
        if rewrites_in_final > 0 {
            total_rewrites = total_rewrites.saturating_add(rewrites_in_final);

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
            verify_no_extension_ops_in_package(&pir_pkg)?;

            let lowered_text = pir_pkg.to_string();
            let mut next_pkg = xlsynth::IrPackage::parse_ir(&lowered_text, None)
                .map_err(|e| format!("aug_opt: xlsynth parse_ir (final) failed: {e}"))?;
            next_pkg
                .set_top_by_name(&top_name)
                .map_err(|e| format!("aug_opt: set_top_by_name('{top_name}') failed: {e}"))?;
            cur_pkg = next_pkg;
        }
    }

    Ok(AugOptRunResult {
        output_text: cur_pkg.to_string(),
        total_rewrites,
    })
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

fn apply_basis_rewrites_to_fn(f: &ir::Fn) -> (ir::Fn, usize) {
    let mut cloned = f.clone();
    let mut rewrites = 0usize;
    rewrites = rewrites.saturating_add(rewrite_and_of_nors_to_flat_nor(&mut cloned));
    rewrites = rewrites.saturating_add(rewrite_nor_of_contiguous_bit_slices_to_not_or_reduce(
        &mut cloned,
    ));
    rewrites = rewrites.saturating_add(rewrite_encode_one_hot_reverse_ult_k_to_or_reduce_top_bits(
        &mut cloned,
    ));
    rewrites = rewrites.saturating_add(
        rewrite_or_reduce_of_encode_one_hot_reverse_slice_ge2_to_nor_top2bits(&mut cloned),
    );
    rewrites = rewrites.saturating_add(
        rewrite_not_or_reduce_of_encode_one_hot_lsb_prio_slice_ge2_to_or_reduce_low2bits(
            &mut cloned,
        ),
    );
    rewrites = rewrites.saturating_add(
        rewrite_or_reduce_of_encode_one_hot_lsb_prio_slice_ge2_to_nor_low2bits(&mut cloned),
    );
    rewrites = rewrites.saturating_add(rewrite_or_reduce_of_select_between_x_and_neg_to_or_reduce(
        &mut cloned,
    ));
    rewrites = rewrites.saturating_add(rewrite_or_reduce_of_neg_to_or_reduce(&mut cloned));
    rewrites = rewrites.saturating_add(rewrite_guarded_sel_ne_literal1_nor(&mut cloned));
    rewrites = rewrites.saturating_add(rewrite_lsb_of_shll_via_shift_is_zero(&mut cloned));
    // Ensure textual IR is defs-before-uses by reordering body nodes into a
    // topological order (while preserving PIR layout invariants). This makes
    // it safe for rewrites to append new nodes.
    ir_utils::compact_and_toposort_in_place(&mut cloned)
        .expect("aug_opt: compact_and_toposort_in_place failed");
    (cloned, rewrites)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ContiguousBitSliceSpan {
    arg: NodeRef,
    start_min: usize,
    width: usize,
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
/// `and(nor(a, b, ...), nor(c, d, ...), ...)`
///   →
/// `nor(a, b, ..., c, d, ...)`
///
/// This is a DeMorgan canonicalization for boolean cones that helps enable
/// subsequent rewrites (e.g. recognizing `nor` of contiguous bit slices).
fn rewrite_and_of_nors_to_flat_nor(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for and_index in 0..f.nodes.len() {
        let NodePayload::Nary(NaryOp::And, and_operands) = f.nodes[and_index].payload.clone()
        else {
            continue;
        };
        if f.nodes[and_index].ty != Type::Bits(1) {
            continue;
        }
        if and_operands.is_empty() {
            continue;
        }

        let mut flat_nor_operands: Vec<NodeRef> = Vec::new();
        let mut ok = true;
        for and_operand in and_operands {
            if *f.get_node_ty(and_operand) != Type::Bits(1) {
                ok = false;
                break;
            }
            let NodePayload::Nary(NaryOp::Nor, nor_operands) =
                f.get_node(and_operand).payload.clone()
            else {
                ok = false;
                break;
            };
            if nor_operands.is_empty() {
                ok = false;
                break;
            }
            flat_nor_operands.extend(nor_operands);
        }
        if !ok || flat_nor_operands.is_empty() {
            continue;
        }

        // Rewrite the original and node in-place to preserve indices.
        f.nodes[and_index].payload = NodePayload::Nary(NaryOp::Nor, flat_nor_operands);
        f.nodes[and_index].ty = Type::Bits(1);
        rewrites += 1;
    }

    rewrites
}

fn match_contiguous_1bit_bit_slices(
    f: &ir::Fn,
    operands: &[NodeRef],
) -> Option<ContiguousBitSliceSpan> {
    if operands.is_empty() {
        return None;
    }

    let (arg0, start0) = match &f.get_node(operands[0]).payload {
        NodePayload::BitSlice {
            arg,
            start,
            width: 1,
        } => (*arg, *start),
        _ => return None,
    };

    let mut starts: Vec<usize> = Vec::with_capacity(operands.len());
    starts.push(start0);
    for &operand in operands.iter().skip(1) {
        let NodePayload::BitSlice { arg, start, width } = &f.get_node(operand).payload else {
            return None;
        };
        if *width != 1 || *arg != arg0 {
            return None;
        }
        starts.push(*start);
    }

    starts.sort_unstable();
    let start_min = *starts.first().expect("non-empty");
    let start_max = *starts.last().expect("non-empty");
    let width = operands.len();
    if width == 0 {
        return None;
    }
    if start_max.saturating_sub(start_min).saturating_add(1) != width {
        return None;
    }
    for (i, s) in starts.iter().enumerate() {
        if *s != start_min.saturating_add(i) {
            return None;
        }
    }

    Some(ContiguousBitSliceSpan {
        arg: arg0,
        start_min,
        width,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SelectBetweenXAndNeg {
    x: NodeRef,
    selector: NodeRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EncodeOneHotReverse {
    x: NodeRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EncodeOneHotDirect {
    x: NodeRef,
}

fn match_encode_one_hot_reverse_lsb_prio_true(
    f: &ir::Fn,
    candidate: NodeRef,
) -> Option<EncodeOneHotReverse> {
    if f.get_node_ty(candidate).bit_count() == 0 {
        return None;
    }

    let NodePayload::Encode { arg: one_hot } = f.get_node(candidate).payload else {
        return None;
    };
    let NodePayload::OneHot { arg: rev, lsb_prio } = f.get_node(one_hot).payload else {
        return None;
    };
    if !lsb_prio {
        return None;
    }
    let NodePayload::Unop(Unop::Reverse, x) = f.get_node(rev).payload else {
        return None;
    };
    if f.get_node_ty(x).bit_count() == 0 {
        return None;
    }

    Some(EncodeOneHotReverse { x })
}

fn match_encode_one_hot_lsb_prio_true(
    f: &ir::Fn,
    candidate: NodeRef,
) -> Option<EncodeOneHotDirect> {
    if f.get_node_ty(candidate).bit_count() == 0 {
        return None;
    }

    let NodePayload::Encode { arg: one_hot } = f.get_node(candidate).payload else {
        return None;
    };
    let NodePayload::OneHot { arg: x, lsb_prio } = f.get_node(one_hot).payload else {
        return None;
    };
    if !lsb_prio {
        return None;
    }
    if f.get_node_ty(x).bit_count() == 0 {
        return None;
    }

    Some(EncodeOneHotDirect { x })
}

fn match_select_between_x_and_neg(f: &ir::Fn, candidate: NodeRef) -> Option<SelectBetweenXAndNeg> {
    if f.get_node_ty(candidate).bit_count() == 0 {
        return None;
    }
    match &f.get_node(candidate).payload {
        NodePayload::Sel {
            selector,
            cases,
            default,
        } => {
            if default.is_some() || cases.len() != 2 {
                return None;
            }
            if *f.get_node_ty(*selector) != Type::Bits(1) {
                return None;
            }
            let a = cases[0];
            let b = cases[1];
            if *f.get_node_ty(a) != *f.get_node_ty(candidate)
                || *f.get_node_ty(b) != *f.get_node_ty(candidate)
            {
                return None;
            }
            // Match either (x, neg(x)) or (neg(x), x).
            if let NodePayload::Unop(Unop::Neg, inner) = f.get_node(a).payload {
                if inner == b {
                    return Some(SelectBetweenXAndNeg {
                        x: b,
                        selector: *selector,
                    });
                }
            }
            if let NodePayload::Unop(Unop::Neg, inner) = f.get_node(b).payload {
                if inner == a {
                    return Some(SelectBetweenXAndNeg {
                        x: a,
                        selector: *selector,
                    });
                }
            }
            None
        }
        NodePayload::PrioritySel {
            selector,
            cases,
            default,
        } => {
            // Only handle the simple 1-case form: priority_sel(s, cases=[...], default=...)
            // which behaves like a mux when `s: bits[1]`.
            if cases.len() != 1 {
                return None;
            }
            let Some(def) = *default else {
                return None;
            };
            if *f.get_node_ty(*selector) != Type::Bits(1) {
                return None;
            }
            let c0 = cases[0];
            if *f.get_node_ty(c0) != *f.get_node_ty(candidate)
                || *f.get_node_ty(def) != *f.get_node_ty(candidate)
            {
                return None;
            }
            // Match either cases=[neg(x)], default=x OR cases=[x], default=neg(x).
            if let NodePayload::Unop(Unop::Neg, inner) = f.get_node(c0).payload {
                if inner == def {
                    return Some(SelectBetweenXAndNeg {
                        x: def,
                        selector: *selector,
                    });
                }
            }
            if let NodePayload::Unop(Unop::Neg, inner) = f.get_node(def).payload {
                if inner == c0 {
                    return Some(SelectBetweenXAndNeg {
                        x: c0,
                        selector: *selector,
                    });
                }
            }
            None
        }
        _ => None,
    }
}

/// Rewrite:
///
/// `nor(bit_slice(x, start=s, width=1), bit_slice(x, start=s+1, width=1), ...)`
///   →
/// `not(or_reduce(bit_slice(x, start=s, width=W)))`
///
/// Preconditions (kept intentionally narrow):
/// - output is `bits[1]`
/// - all operands are 1-bit slices from the same `arg`
/// - the slice starts form a contiguous span
fn rewrite_nor_of_contiguous_bit_slices_to_not_or_reduce(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for nor_index in 0..f.nodes.len() {
        let NodePayload::Nary(NaryOp::Nor, operands) = f.nodes[nor_index].payload.clone() else {
            continue;
        };
        if f.nodes[nor_index].ty != Type::Bits(1) {
            continue;
        }

        let Some(span) = match_contiguous_1bit_bit_slices(f, &operands) else {
            continue;
        };
        if span.width == 0 {
            continue;
        }

        let wide_slice = push_node(
            f,
            Type::Bits(span.width),
            NodePayload::BitSlice {
                arg: span.arg,
                start: span.start_min,
                width: span.width,
            },
        );
        let or_reduced = push_node(
            f,
            Type::Bits(1),
            NodePayload::Unop(Unop::OrReduce, wide_slice),
        );

        // Rewrite the original nor node in-place to preserve indices.
        f.nodes[nor_index].payload = NodePayload::Unop(Unop::Not, or_reduced);
        f.nodes[nor_index].ty = Type::Bits(1);

        rewrites += 1;
    }

    rewrites
}

/// Rewrite:
///
/// `or_reduce(neg(x))` → `or_reduce(x)`
///
/// Intuition: `or_reduce(x)` is a nonzero test (i.e. `x != 0`).
/// Two's-complement negation preserves whether a value is zero: `neg(x) == 0`
/// iff `x == 0`.
fn rewrite_or_reduce_of_neg_to_or_reduce(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for node_index in 0..f.nodes.len() {
        let NodePayload::Unop(Unop::OrReduce, arg) = f.nodes[node_index].payload.clone() else {
            continue;
        };
        if f.nodes[node_index].ty != Type::Bits(1) {
            continue;
        }
        let NodePayload::Unop(Unop::Neg, inner) = f.get_node(arg).payload.clone() else {
            continue;
        };
        if f.get_node_ty(inner).bit_count() == 0 {
            continue;
        }

        f.nodes[node_index].payload = NodePayload::Unop(Unop::OrReduce, inner);
        f.nodes[node_index].ty = Type::Bits(1);
        rewrites += 1;
    }

    rewrites
}

/// Rewrite:
///
/// `or_reduce(sel(s, cases=[x, neg(x)]))` → `or_reduce(x)`
/// `or_reduce(priority_sel(s, cases=[neg(x)], default=x))` → `or_reduce(x)`
///
/// Intuition: `or_reduce(v)` is a nonzero test (`v != 0`). Negation preserves
/// whether a value is zero, so selecting between `x` and `neg(x)` does not
/// change the result.
fn rewrite_or_reduce_of_select_between_x_and_neg_to_or_reduce(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for node_index in 0..f.nodes.len() {
        let NodePayload::Unop(Unop::OrReduce, arg) = f.nodes[node_index].payload.clone() else {
            continue;
        };
        if f.nodes[node_index].ty != Type::Bits(1) {
            continue;
        }

        let Some(m) = match_select_between_x_and_neg(f, arg) else {
            continue;
        };
        // No need to check selector polarity: `or_reduce(x) == or_reduce(neg(x))`.
        let _ = m.selector;

        f.nodes[node_index].payload = NodePayload::Unop(Unop::OrReduce, m.x);
        f.nodes[node_index].ty = Type::Bits(1);
        rewrites += 1;
    }

    rewrites
}

/// Rewrite:
///
/// `ult(encode(one_hot(reverse(x), lsb_prio=true)), K)`
///   →
/// `or_reduce(bit_slice(x, start=W-K, width=K))`
///
/// Here `W` is the bit width of `x`, and `K` is a literal integer constant
/// with `1 <= K <= W`.
///
/// Intuition: `encode(one_hot(reverse(x), lsb_prio=true))` is the distance from
/// the MSB down to the first `1` bit in `x` (with an all-zero sentinel). Being
/// `< K` is equivalent to “there exists a `1` in the top `K` bits of `x`”, i.e.
/// `or_reduce(x[W-1 : W-K])`.
fn rewrite_encode_one_hot_reverse_ult_k_to_or_reduce_top_bits(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for node_index in 0..f.nodes.len() {
        let NodePayload::Binop(Binop::Ult, lhs, rhs) = f.nodes[node_index].payload.clone() else {
            continue;
        };
        if f.nodes[node_index].ty != Type::Bits(1) {
            continue;
        }

        let Some(m) = match_encode_one_hot_reverse_lsb_prio_true(f, lhs) else {
            continue;
        };

        let NodePayload::Literal(k_lit) = &f.get_node(rhs).payload else {
            continue;
        };
        let Some(k_u64) = k_lit.to_u64().ok() else {
            continue;
        };
        let Some(k) = usize::try_from(k_u64).ok() else {
            continue;
        };

        let w = f.get_node_ty(m.x).bit_count();
        if w == 0 || k == 0 || k > w {
            continue;
        }

        // Ensure the compare is well-typed: K literal width must match encode width.
        if *f.get_node_ty(rhs) != *f.get_node_ty(lhs) {
            continue;
        }

        let top_slice = push_node(
            f,
            Type::Bits(k),
            NodePayload::BitSlice {
                arg: m.x,
                start: w - k,
                width: k,
            },
        );
        f.nodes[node_index].payload = NodePayload::Unop(Unop::OrReduce, top_slice);
        f.nodes[node_index].ty = Type::Bits(1);
        rewrites += 1;
    }

    rewrites
}

/// Rewrite:
///
/// `or_reduce(bit_slice(encode(one_hot(reverse(x), lsb_prio=true)), start=1,
/// width=E-1))`   →
/// `nor(bit_slice(x, start=W-1, width=1), bit_slice(x, start=W-2, width=1))`
///
/// where:
/// - `W` is the width of `x` (and must satisfy `W >= 2`)
/// - `E` is the width of the `encode(...)` result
///
/// Intuition: `encode(one_hot(reverse(x), lsb_prio=true))` yields the index of
/// the most-significant 1 bit in `x`, with a sentinel value for the all-zero
/// case. Slicing off bit 0 (i.e. `>> 1`) and `or_reduce` is testing whether
/// that index is >= 2, which is equivalent to “the top two bits of `x` are both
/// zero”.
///
/// This is a common consumer form in our k3 cones corpus (as opposed to an
/// explicit `< K` compare).
fn rewrite_or_reduce_of_encode_one_hot_reverse_slice_ge2_to_nor_top2bits(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for node_index in 0..f.nodes.len() {
        let NodePayload::Unop(Unop::OrReduce, slice_nr) = f.nodes[node_index].payload.clone()
        else {
            continue;
        };
        if f.nodes[node_index].ty != Type::Bits(1) {
            continue;
        }

        let NodePayload::BitSlice { arg, start, width } = f.get_node(slice_nr).payload else {
            continue;
        };
        if start != 1 {
            continue;
        }

        let enc_w = f.get_node_ty(arg).bit_count();
        if enc_w == 0 {
            continue;
        }
        // Only handle the full "drop the LSB" slice so this condition is exactly
        // `encode(...) >= 2`.
        if width != enc_w.saturating_sub(1) {
            continue;
        }

        let Some(m) = match_encode_one_hot_reverse_lsb_prio_true(f, arg) else {
            continue;
        };
        let w = f.get_node_ty(m.x).bit_count();
        if w < 2 {
            continue;
        }

        let msb = push_node(
            f,
            Type::Bits(1),
            NodePayload::BitSlice {
                arg: m.x,
                start: w - 1,
                width: 1,
            },
        );
        let second_msb = push_node(
            f,
            Type::Bits(1),
            NodePayload::BitSlice {
                arg: m.x,
                start: w - 2,
                width: 1,
            },
        );
        f.nodes[node_index].payload = NodePayload::Nary(NaryOp::Nor, vec![msb, second_msb]);
        f.nodes[node_index].ty = Type::Bits(1);
        rewrites += 1;
    }

    rewrites
}

/// Rewrite:
///
/// `not(or_reduce(bit_slice(encode(one_hot(x, lsb_prio=true)), start=1,
/// width=E-1)))`   →
/// `or_reduce(bit_slice(x, start=0, width=2))`
///
/// where:
/// - `x` has width `W >= 2`
/// - the encode result has width `E` and we match the full "drop LSB" slice
///
/// Intuition: `encode(one_hot(x, lsb_prio=true))` yields the index of the
/// least- significant 1 bit in `x` (with a sentinel for all-zero). Dropping bit
/// 0 and `or_reduce` tests whether that index is >= 2. Negating it yields
/// “index < 2”, which is equivalent to `x[0] | x[1]` (the least-significant 1
/// is at bit 0 or 1).
fn rewrite_not_or_reduce_of_encode_one_hot_lsb_prio_slice_ge2_to_or_reduce_low2bits(
    f: &mut ir::Fn,
) -> usize {
    let mut rewrites = 0usize;

    for node_index in 0..f.nodes.len() {
        let NodePayload::Unop(Unop::Not, inner) = f.nodes[node_index].payload.clone() else {
            continue;
        };
        if f.nodes[node_index].ty != Type::Bits(1) {
            continue;
        }

        let NodePayload::Unop(Unop::OrReduce, slice_nr) = f.get_node(inner).payload else {
            continue;
        };
        let NodePayload::BitSlice { arg, start, width } = f.get_node(slice_nr).payload else {
            continue;
        };
        if start != 1 {
            continue;
        }

        let enc_w = f.get_node_ty(arg).bit_count();
        if enc_w == 0 || width != enc_w.saturating_sub(1) {
            continue;
        }

        let Some(m) = match_encode_one_hot_lsb_prio_true(f, arg) else {
            continue;
        };
        let w = f.get_node_ty(m.x).bit_count();
        if w < 2 {
            continue;
        }

        let low2 = push_node(
            f,
            Type::Bits(2),
            NodePayload::BitSlice {
                arg: m.x,
                start: 0,
                width: 2,
            },
        );

        // Rewrite `not(or_reduce(...))` in-place to `or_reduce(low2)`.
        f.nodes[node_index].payload = NodePayload::Unop(Unop::OrReduce, low2);
        f.nodes[node_index].ty = Type::Bits(1);
        rewrites += 1;
    }

    rewrites
}

/// Rewrite:
///
/// `or_reduce(bit_slice(encode(one_hot(x, lsb_prio=true)), start=1,
/// width=E-1))`   →
/// `nor(bit_slice(x, start=0, width=1), bit_slice(x, start=1, width=1))`
///
/// where:
/// - `x` has width `W >= 2`
/// - the encode result has width `E` and we match the full "drop LSB" slice
///
/// Intuition: `encode(one_hot(x, lsb_prio=true))` yields the index of the
/// least- significant 1 bit in `x`, with a sentinel value for the all-zero
/// case. Slicing off bit 0 and `or_reduce` tests whether that index is >= 2,
/// which holds iff the low two bits of `x` are both 0 (including the all-zero
/// sentinel case).
fn rewrite_or_reduce_of_encode_one_hot_lsb_prio_slice_ge2_to_nor_low2bits(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for node_index in 0..f.nodes.len() {
        let NodePayload::Unop(Unop::OrReduce, slice_nr) = f.nodes[node_index].payload.clone()
        else {
            continue;
        };
        if f.nodes[node_index].ty != Type::Bits(1) {
            continue;
        }

        let NodePayload::BitSlice { arg, start, width } = f.get_node(slice_nr).payload else {
            continue;
        };
        if start != 1 {
            continue;
        }

        let enc_w = f.get_node_ty(arg).bit_count();
        if enc_w == 0 || width != enc_w.saturating_sub(1) {
            continue;
        }

        let Some(m) = match_encode_one_hot_lsb_prio_true(f, arg) else {
            continue;
        };
        let w = f.get_node_ty(m.x).bit_count();
        if w < 2 {
            continue;
        }

        let lsb0 = push_node(
            f,
            Type::Bits(1),
            NodePayload::BitSlice {
                arg: m.x,
                start: 0,
                width: 1,
            },
        );
        let lsb1 = push_node(
            f,
            Type::Bits(1),
            NodePayload::BitSlice {
                arg: m.x,
                start: 1,
                width: 1,
            },
        );
        f.nodes[node_index].payload = NodePayload::Nary(NaryOp::Nor, vec![lsb0, lsb1]);
        f.nodes[node_index].ty = Type::Bits(1);
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
    use crate::fuzz_utils::arbitrary_irbits;
    use crate::ir_eval::{FnEvalResult, eval_fn};
    use crate::test_utils::quickcheck_ir_text_fn_equivalence_ubits_le64;
    use rand_pcg::Pcg64Mcg;
    use xlsynth::{IrBits, IrValue};

    fn quickcheck_ir_text_fn_equivalence_anywidth(
        ir_text_0: &str,
        ir_text_1: &str,
        fn_name: &str,
        param_widths: &[usize],
        random_samples: usize,
    ) {
        let mut p0 = ir_parser::Parser::new(ir_text_0);
        let pkg0 = p0.parse_and_validate_package().expect("parse/validate lhs");
        let f0 = pkg0.get_fn(fn_name).expect("lhs missing function");

        let mut p1 = ir_parser::Parser::new(ir_text_1);
        let pkg1 = p1.parse_and_validate_package().expect("parse/validate rhs");
        let f1 = pkg1.get_fn(fn_name).expect("rhs missing function");

        let mut rng = Pcg64Mcg::new(0);

        let run_case = |args: &[IrValue]| {
            let got0 = match eval_fn(f0, args) {
                FnEvalResult::Success(s) => s.value.clone(),
                FnEvalResult::Failure(e) => panic!("unexpected eval failure (lhs): {:?}", e),
            };
            let got1 = match eval_fn(f1, args) {
                FnEvalResult::Success(s) => s.value.clone(),
                FnEvalResult::Failure(e) => panic!("unexpected eval failure (rhs): {:?}", e),
            };
            assert_eq!(got0, got1, "mismatch on args={args:?}");
        };

        // Deterministic edge case: all zeros.
        {
            let zeros: Vec<IrValue> = param_widths
                .iter()
                .map(|&w| IrValue::from_bits(&IrBits::make_ubits(w, 0).unwrap()))
                .collect();
            run_case(&zeros);
        }

        // Deterministic pseudo-random sampling.
        for _ in 0..random_samples {
            let mut args: Vec<IrValue> = Vec::with_capacity(param_widths.len());
            for &w in param_widths {
                let bits = arbitrary_irbits(&mut rng, w);
                args.push(IrValue::from_bits(&bits));
            }
            run_case(&args);
        }
    }

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
                run_xlsynth_opt_before: true,
                run_xlsynth_opt_after: true,
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
                run_xlsynth_opt_before: true,
                run_xlsynth_opt_after: true,
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

    /// Verifies that `aug_opt` recognizes the `and(of nor(of 1-bit slices))`
    /// pattern and canonicalizes it into `not(or_reduce(bit_slice(...)))`,
    /// enabling cheaper lowering in g8r.
    #[test]
    fn aug_opt_rewrites_and_of_nors_of_contiguous_slices_into_not_or_reduce() {
        let ir_text = r#"package bool_cone2

top fn cone(leaf_40: bits[27] id=1, leaf_52: bits[26] id=2, leaf_57: bits[1] id=3) -> bits[1] {

  bit_slice.4: bits[25] = bit_slice(leaf_52, start=0, width=25, id=4)

  bit_slice.5: bits[2] = bit_slice(leaf_40, start=1, width=2, id=5)

  concat.6: bits[28] = concat(bit_slice.4, bit_slice.5, leaf_57, id=6)

  bit_slice.7: bits[1] = bit_slice(leaf_52, start=25, width=1, id=7)

  neg.8: bits[28] = neg(concat.6, id=8)

  priority_sel.9: bits[28] = priority_sel(bit_slice.7, cases=[neg.8], default=concat.6, id=9)

  bit_slice.10: bits[1] = bit_slice(priority_sel.9, start=3, width=1, id=10)

  bit_slice.11: bits[1] = bit_slice(priority_sel.9, start=2, width=1, id=11)

  bit_slice.12: bits[1] = bit_slice(priority_sel.9, start=1, width=1, id=12)

  bit_slice.13: bits[1] = bit_slice(priority_sel.9, start=0, width=1, id=13)

  nor.15: bits[1] = nor(bit_slice.10, bit_slice.11, id=15)

  nor.14: bits[1] = nor(bit_slice.12, bit_slice.13, id=14)

  ret and.16: bits[1] = and(nor.15, nor.14, id=16)

}
"#;

        // Check the local PIR rewrite shape directly.
        let mut p = ir_parser::Parser::new(ir_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let top_fn = pkg.get_fn("cone").expect("top fn");
        let (rewritten, rewrites) = apply_basis_rewrites_to_fn(top_fn);
        assert!(rewrites > 0, "expected at least one rewrite");

        let mut found = false;
        for node in &rewritten.nodes {
            let NodePayload::Unop(Unop::Not, not_arg) = &node.payload else {
                continue;
            };
            let NodePayload::Unop(Unop::OrReduce, red_arg) = &rewritten.get_node(*not_arg).payload
            else {
                continue;
            };
            let NodePayload::BitSlice { start, width, .. } = &rewritten.get_node(*red_arg).payload
            else {
                continue;
            };
            if *start == 0 && *width == 4 {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "expected not(or_reduce(bit_slice(start=0,width=4))); got:\n{rewritten}"
        );

        // And verify end-to-end equivalence through aug_opt's libxls sandwich.
        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                run_xlsynth_opt_before: true,
                run_xlsynth_opt_after: true,
            },
        )
        .expect("aug opt");
        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[27, 26, 1],
            /* random_samples= */ 2000,
        );
    }

    /// Verifies the generic invariant `or_reduce(neg(x)) == or_reduce(x)`.
    #[test]
    fn aug_opt_rewrites_or_reduce_of_neg_to_or_reduce() {
        let ir_text = r#"package or_reduce_neg

top fn f(x: bits[8] id=1) -> bits[1] {
  neg.2: bits[8] = neg(x, id=2)
  ret or_reduce.3: bits[1] = or_reduce(neg.2, id=3)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("f"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                run_xlsynth_opt_before: true,
                run_xlsynth_opt_after: true,
            },
        )
        .expect("aug opt");

        // Assert the output contains an or_reduce directly of x (no neg operand).
        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("f").expect("top fn");
        let mut saw_or_reduce_of_x = false;
        for node in &f.nodes {
            let NodePayload::Unop(Unop::OrReduce, arg) = &node.payload else {
                continue;
            };
            if matches!(f.get_node(*arg).payload, NodePayload::GetParam(_)) {
                saw_or_reduce_of_x = true;
                break;
            }
        }
        assert!(
            saw_or_reduce_of_x,
            "expected or_reduce(x) after rewrite; got:\n{}",
            out_text
        );

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "f",
            &[8],
            /* random_samples= */ 2000,
        );
    }

    /// Verifies the invariant `or_reduce(sel_between(x, neg(x))) ==
    /// or_reduce(x)`.
    #[test]
    fn aug_opt_rewrites_or_reduce_of_select_between_x_and_neg_to_or_reduce() {
        let ir_text = r#"package or_reduce_sel_neg

top fn f(x: bits[8] id=1, s: bits[1] id=2) -> bits[1] {
  neg.3: bits[8] = neg(x, id=3)
  sel.4: bits[8] = sel(s, cases=[x, neg.3], id=4)
  ret or_reduce.5: bits[1] = or_reduce(sel.4, id=5)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("f"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                run_xlsynth_opt_before: true,
                run_xlsynth_opt_after: true,
            },
        )
        .expect("aug opt");

        // Assert the output contains an or_reduce directly of x (no sel operand).
        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("f").expect("top fn");
        let mut saw_or_reduce_of_x = false;
        for node in &f.nodes {
            let NodePayload::Unop(Unop::OrReduce, arg) = &node.payload else {
                continue;
            };
            if matches!(f.get_node(*arg).payload, NodePayload::GetParam(_)) {
                saw_or_reduce_of_x = true;
                break;
            }
        }
        assert!(
            saw_or_reduce_of_x,
            "expected or_reduce(x) after rewrite; got:\n{}",
            out_text
        );

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "f",
            &[8, 1],
            /* random_samples= */ 2000,
        );
    }

    /// Verifies the full cone regression: a zero-test on the low bits of a
    /// conditional-negation (`priority_sel` selecting between `x` and `neg(x)`)
    /// simplifies to a zero-test on `x` alone.
    ///
    /// This is a key canonicalization for g8r: after aug-opt, the AIGER/ABC
    /// `and/lev` drops dramatically for this cone.
    #[test]
    fn aug_opt_regression_k3_cone_conditional_negation_zero_test() {
        let ir_text = r#"package bool_cone

top fn cone(leaf_40: bits[27] id=1, leaf_52: bits[26] id=2, leaf_57: bits[1] id=3) -> bits[1] {
  bit_slice.4: bits[25] = bit_slice(leaf_52, start=0, width=25, id=4)
  bit_slice.5: bits[2] = bit_slice(leaf_40, start=1, width=2, id=5)
  concat.6: bits[28] = concat(bit_slice.4, bit_slice.5, leaf_57, id=6)
  bit_slice.7: bits[1] = bit_slice(leaf_52, start=25, width=1, id=7)
  neg.8: bits[28] = neg(concat.6, id=8)
  priority_sel.9: bits[28] = priority_sel(bit_slice.7, cases=[neg.8], default=concat.6, id=9)
  bit_slice.10: bits[1] = bit_slice(priority_sel.9, start=3, width=1, id=10)
  bit_slice.11: bits[1] = bit_slice(priority_sel.9, start=2, width=1, id=11)
  bit_slice.12: bits[1] = bit_slice(priority_sel.9, start=1, width=1, id=12)
  bit_slice.13: bits[1] = bit_slice(priority_sel.9, start=0, width=1, id=13)
  nor.15: bits[1] = nor(bit_slice.10, bit_slice.11, id=15)
  nor.14: bits[1] = nor(bit_slice.12, bit_slice.13, id=14)
  ret and.16: bits[1] = and(nor.15, nor.14, id=16)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                run_xlsynth_opt_before: true,
                run_xlsynth_opt_after: true,
            },
        )
        .expect("aug opt");

        // Assert that the output performs `or_reduce` on the concat-derived value
        // (i.e. the conditional negation was eliminated under the nonzero test),
        // and does NOT reduce over the priority_sel result.
        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");

        let mut saw_or_reduce_of_concat = false;
        let mut saw_or_reduce_of_priority_sel = false;
        for node in &f.nodes {
            let NodePayload::Unop(Unop::OrReduce, arg) = &node.payload else {
                continue;
            };
            match &f.get_node(*arg).payload {
                NodePayload::Nary(NaryOp::Concat, _) => {
                    saw_or_reduce_of_concat = true;
                }
                NodePayload::PrioritySel { .. } => {
                    saw_or_reduce_of_priority_sel = true;
                }
                _ => {}
            }
        }
        assert!(
            saw_or_reduce_of_concat,
            "expected or_reduce(concat(...)) after aug_opt; got:\n{}",
            out_text
        );
        assert!(
            !saw_or_reduce_of_priority_sel,
            "expected conditional negation to be eliminated under or_reduce; got:\n{}",
            out_text
        );

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[27, 26, 1],
            /* random_samples= */ 2000,
        );
    }

    /// Verifies:
    /// `encode(one_hot(reverse(x), lsb_prio=true)) < K`
    ///   ↔
    /// `or_reduce(x[W-1 : W-K])`.
    #[test]
    fn aug_opt_rewrites_encode_one_hot_reverse_ult_k_to_or_reduce_top_bits() {
        let ir_text = r#"package enc_onehot_rev

top fn f(x: bits[8] id=1) -> bits[1] {
  reverse.2: bits[8] = reverse(x, id=2)
  one_hot.3: bits[9] = one_hot(reverse.2, lsb_prio=true, id=3)
  encode.4: bits[4] = encode(one_hot.3, id=4)
  literal.5: bits[4] = literal(value=3, id=5)
  ret ult.6: bits[1] = ult(encode.4, literal.5, id=6)
}
"#;

        // Use the PIR-only path so the output preserves the rewrite shape for
        // structural assertions.
        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("f"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                run_xlsynth_opt_before: false,
                run_xlsynth_opt_after: false,
            },
        )
        .expect("aug opt");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("f").expect("top fn");

        let mut saw = false;
        for node in &f.nodes {
            let NodePayload::Unop(Unop::OrReduce, arg) = &node.payload else {
                continue;
            };
            let NodePayload::BitSlice {
                arg: slice_arg,
                start,
                width,
            } = &f.get_node(*arg).payload
            else {
                continue;
            };
            if matches!(f.get_node(*slice_arg).payload, NodePayload::GetParam(_))
                && *start == 5
                && *width == 3
            {
                saw = true;
                break;
            }
        }
        assert!(
            saw,
            "expected or_reduce(bit_slice(x,start=5,width=3)); got:\n{out_text}"
        );

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "f",
            &[8],
            /* random_samples= */ 2000,
        );
    }

    /// Verifies:
    /// `or_reduce(bit_slice(encode(one_hot(reverse(x), lsb_prio=true)),
    /// start=1, width=E-1))`   ↔
    /// `nor(x[W-1], x[W-2])`.
    #[test]
    fn aug_opt_rewrites_encode_one_hot_reverse_encode_ge2_to_nor_top2bits() {
        let ir_text = r#"package enc_onehot_rev_ge2

top fn f(x: bits[8] id=1) -> bits[1] {
  reverse.2: bits[8] = reverse(x, id=2)
  one_hot.3: bits[9] = one_hot(reverse.2, lsb_prio=true, id=3)
  encode.4: bits[4] = encode(one_hot.3, id=4)
  bit_slice.5: bits[3] = bit_slice(encode.4, start=1, width=3, id=5)
  ret or_reduce.6: bits[1] = or_reduce(bit_slice.5, id=6)
}
"#;

        // Use the PIR-only path so the output preserves the rewrite shape for
        // structural assertions.
        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("f"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                run_xlsynth_opt_before: false,
                run_xlsynth_opt_after: false,
            },
        )
        .expect("aug opt");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("f").expect("top fn");

        let mut saw = false;
        for node in &f.nodes {
            let NodePayload::Nary(NaryOp::Nor, operands) = &node.payload else {
                continue;
            };
            if operands.len() != 2 {
                continue;
            }
            let mut starts: Vec<usize> = Vec::new();
            for &op in operands {
                let NodePayload::BitSlice { arg, start, width } = &f.get_node(op).payload else {
                    starts.clear();
                    break;
                };
                if *width != 1 {
                    starts.clear();
                    break;
                }
                if !matches!(f.get_node(*arg).payload, NodePayload::GetParam(_)) {
                    starts.clear();
                    break;
                }
                starts.push(*start);
            }
            starts.sort_unstable();
            if starts == vec![6, 7] {
                saw = true;
                break;
            }
        }
        assert!(
            saw,
            "expected nor(bit_slice(x,6,1), bit_slice(x,7,1)); got:\n{out_text}"
        );

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "f",
            &[8],
            /* random_samples= */ 2000,
        );
    }

    /// Regression for the lsb-priority one_hot/encode cone:
    /// this should simplify to `or_reduce(x[0:2])`.
    #[test]
    fn aug_opt_regression_one_hot_encode_lsb_prio_not_or_reduce_slice_ge2() {
        let ir_text = r#"package float64

top fn cone(leaf_137: bits[161] id=1) -> bits[1] {
  one_hot.2: bits[162] = one_hot(leaf_137, lsb_prio=true, id=2)
  encode.3: bits[8] = encode(one_hot.2, id=3)
  bit_slice.4: bits[7] = bit_slice(encode.3, start=1, width=7, id=4)
  or_reduce.5: bits[1] = or_reduce(bit_slice.4, id=5)
  ret not.6: bits[1] = not(or_reduce.5, id=6)
}
"#;

        // Use PIR-only mode so the output preserves the rewrite shape.
        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                run_xlsynth_opt_before: false,
                run_xlsynth_opt_after: false,
            },
        )
        .expect("aug opt");

        // Structural assertion: expect an `or_reduce(bit_slice(leaf_137, start=0,
        // width=2))`.
        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");
        let mut saw = false;
        for node in &f.nodes {
            let NodePayload::Unop(Unop::OrReduce, arg) = &node.payload else {
                continue;
            };
            let NodePayload::BitSlice {
                arg: slice_arg,
                start,
                width,
            } = &f.get_node(*arg).payload
            else {
                continue;
            };
            if matches!(f.get_node(*slice_arg).payload, NodePayload::GetParam(_))
                && *start == 0
                && *width == 2
            {
                saw = true;
                break;
            }
        }
        assert!(
            saw,
            "expected or_reduce(bit_slice(x,start=0,width=2)); got:\n{out_text}"
        );

        // Semantics check with an any-width evaluator.
        quickcheck_ir_text_fn_equivalence_anywidth(
            ir_text,
            &out_text,
            "cone",
            &[161],
            /* random_samples= */ 2000,
        );
    }

    /// Regression for the lsb-priority one_hot/encode cone (without the final
    /// `not`): this should simplify to `nor(x[0], x[1])`.
    #[test]
    fn aug_opt_regression_one_hot_encode_lsb_prio_or_reduce_slice_ge2() {
        let ir_text = r#"package bool_cone

top fn cone(leaf_137: bits[161] id=1) -> bits[1] {
  one_hot.2: bits[162] = one_hot(leaf_137, lsb_prio=true, id=2)
  encode.3: bits[8] = encode(one_hot.2, id=3)
  bit_slice.4: bits[7] = bit_slice(encode.3, start=1, width=7, id=4)
  ret or_reduce.5: bits[1] = or_reduce(bit_slice.4, id=5)
}
"#;

        // Use PIR-only mode so the output preserves the rewrite shape.
        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                run_xlsynth_opt_before: false,
                run_xlsynth_opt_after: false,
            },
        )
        .expect("aug opt");

        // Structural assertion: expect `nor(bit_slice(x,0,1), bit_slice(x,1,1))`.
        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");
        let mut saw = false;
        for node in &f.nodes {
            let NodePayload::Nary(NaryOp::Nor, operands) = &node.payload else {
                continue;
            };
            if operands.len() != 2 {
                continue;
            }
            let mut starts: Vec<usize> = Vec::new();
            for &op in operands {
                let NodePayload::BitSlice { arg, start, width } = &f.get_node(op).payload else {
                    starts.clear();
                    break;
                };
                if *width != 1 {
                    starts.clear();
                    break;
                }
                if !matches!(f.get_node(*arg).payload, NodePayload::GetParam(_)) {
                    starts.clear();
                    break;
                }
                starts.push(*start);
            }
            starts.sort_unstable();
            if starts == vec![0, 1] {
                saw = true;
                break;
            }
        }
        assert!(
            saw,
            "expected nor(bit_slice(x,0,1), bit_slice(x,1,1)); got:\n{out_text}"
        );

        quickcheck_ir_text_fn_equivalence_anywidth(
            ir_text,
            &out_text,
            "cone",
            &[161],
            /* random_samples= */ 2000,
        );
    }
}
