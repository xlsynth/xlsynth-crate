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
use crate::ir_range_info::IrRangeInfo;
use crate::ir_utils;
use xlsynth::IrValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AugOptMode {
    /// The default "opt sandwich":
    /// libxls opt -> PIR rewrites -> libxls opt.
    Sandwich,
    /// Apply PIR rewrites only (no libxls optimization passes).
    PirOnly,
}

#[derive(Debug, Clone, Copy)]
pub struct AugOptOptions {
    pub enable: bool,
    pub rounds: usize,
    pub mode: AugOptMode,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AugOptRewriteStats {
    pub guarded_sel_ne1_nor: usize,
    pub lsb_of_shll: usize,
    pub pow2_msb_compare_with_eq_tiebreak: usize,
    pub eq_priority_sel_to_selector_predicate: usize,
    pub eq_add_zero_to_eq_rhs_sub: usize,
    pub ne_shrl_slice_known_one_shift_nonzero: usize,
    pub predicate_hoist_across_select: usize,
}

impl AugOptRewriteStats {
    pub fn total(&self) -> usize {
        self.guarded_sel_ne1_nor
            .saturating_add(self.lsb_of_shll)
            .saturating_add(self.pow2_msb_compare_with_eq_tiebreak)
            .saturating_add(self.eq_priority_sel_to_selector_predicate)
            .saturating_add(self.eq_add_zero_to_eq_rhs_sub)
            .saturating_add(self.ne_shrl_slice_known_one_shift_nonzero)
            .saturating_add(self.predicate_hoist_across_select)
    }

    fn saturating_add_assign(&mut self, other: AugOptRewriteStats) {
        self.guarded_sel_ne1_nor = self
            .guarded_sel_ne1_nor
            .saturating_add(other.guarded_sel_ne1_nor);
        self.lsb_of_shll = self.lsb_of_shll.saturating_add(other.lsb_of_shll);
        self.pow2_msb_compare_with_eq_tiebreak = self
            .pow2_msb_compare_with_eq_tiebreak
            .saturating_add(other.pow2_msb_compare_with_eq_tiebreak);
        self.eq_priority_sel_to_selector_predicate = self
            .eq_priority_sel_to_selector_predicate
            .saturating_add(other.eq_priority_sel_to_selector_predicate);
        self.eq_add_zero_to_eq_rhs_sub = self
            .eq_add_zero_to_eq_rhs_sub
            .saturating_add(other.eq_add_zero_to_eq_rhs_sub);
        self.ne_shrl_slice_known_one_shift_nonzero = self
            .ne_shrl_slice_known_one_shift_nonzero
            .saturating_add(other.ne_shrl_slice_known_one_shift_nonzero);
        self.predicate_hoist_across_select = self
            .predicate_hoist_across_select
            .saturating_add(other.predicate_hoist_across_select);
    }
}

#[derive(Debug, Clone)]
pub struct AugOptRunResult {
    pub output_text: String,
    pub total_rewrites: usize,
    pub rewrite_stats: AugOptRewriteStats,
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
            mode: AugOptMode::Sandwich,
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
            rewrite_stats: AugOptRewriteStats::default(),
        });
    }

    // Basis-only contract: aug-opt does not support PIR extension ops.
    // Fail fast with a clear error before libxls parsing.
    verify_no_extension_ops_in_ir_text(ir_text)?;

    let top_name = top
        .ok_or_else(|| "aug_opt: top is required".to_string())?
        .to_string();
    match options.mode {
        AugOptMode::Sandwich => {
            // Start by letting libxls do its normal canonicalization.
            let mut cur_pkg = xlsynth::IrPackage::parse_ir(ir_text, None)
                .map_err(|e| format!("aug_opt: xlsynth parse_ir failed: {e}"))?;
            cur_pkg
                .set_top_by_name(&top_name)
                .map_err(|e| format!("aug_opt: set_top_by_name('{top_name}') failed: {e}"))?;

            // One initial opt pass before the co-recursive rounds.
            cur_pkg = xlsynth::optimize_ir(&cur_pkg, &top_name)
                .map_err(|e| format!("aug_opt: optimize_ir initial failed: {e}"))?;

            let mut rewrite_stats = AugOptRewriteStats::default();
            for _round in 0..options.rounds {
                let cur_text = cur_pkg.to_string();
                let (lowered_text, rewrites_in_round) =
                    apply_pir_rewrites_to_ir_text(&cur_text, &top_name)?;
                rewrite_stats.saturating_add_assign(rewrites_in_round);

                let mut next_pkg = xlsynth::IrPackage::parse_ir(&lowered_text, None)
                    .map_err(|e| format!("aug_opt: xlsynth parse_ir (post-rewrite) failed: {e}"))?;
                next_pkg
                    .set_top_by_name(&top_name)
                    .map_err(|e| format!("aug_opt: set_top_by_name('{top_name}') failed: {e}"))?;
                cur_pkg = xlsynth::optimize_ir(&next_pkg, &top_name)
                    .map_err(|e| format!("aug_opt: optimize_ir post-rewrite failed: {e}"))?;
            }

            Ok(AugOptRunResult {
                output_text: cur_pkg.to_string(),
                total_rewrites: rewrite_stats.total(),
                rewrite_stats,
            })
        }
        AugOptMode::PirOnly => {
            let mut cur_text = ir_text.to_string();
            let mut rewrite_stats = AugOptRewriteStats::default();

            // Even with zero rounds, we still want to validate the requested top
            // and ensure the emitted text marks it as top.
            if options.rounds == 0 {
                let mut pir_parser = ir_parser::Parser::new(&cur_text);
                let mut pir_pkg = pir_parser
                    .parse_and_validate_package()
                    .map_err(|e| format!("aug_opt: PIR parse/validate failed: {e}"))?;
                pir_pkg
                    .get_fn(&top_name)
                    .ok_or_else(|| format!("aug_opt: PIR package missing top fn '{top_name}'"))?;
                pir_pkg.set_top_fn(&top_name).map_err(|e| {
                    format!("aug_opt: internal error: set_top_fn('{top_name}') failed: {e}")
                })?;
                return Ok(AugOptRunResult {
                    output_text: pir_pkg.to_string(),
                    total_rewrites: 0,
                    rewrite_stats: AugOptRewriteStats::default(),
                });
            }

            for _round in 0..options.rounds {
                let (next_text, rewrites_in_round) =
                    apply_pir_rewrites_to_ir_text(&cur_text, &top_name)?;
                rewrite_stats.saturating_add_assign(rewrites_in_round);
                cur_text = next_text;
            }
            Ok(AugOptRunResult {
                output_text: cur_text,
                total_rewrites: rewrite_stats.total(),
                rewrite_stats,
            })
        }
    }
}

fn apply_pir_rewrites_to_ir_text(
    ir_text: &str,
    top_name: &str,
) -> Result<(String, AugOptRewriteStats), String> {
    // Parse with PIR, apply basis-only rewrites to the top function.
    let mut pir_parser = ir_parser::Parser::new(ir_text);
    let mut pir_pkg = pir_parser
        .parse_and_validate_package()
        .map_err(|e| format!("aug_opt: PIR parse/validate failed: {e}"))?;

    let top_fn = pir_pkg
        .get_fn(top_name)
        .ok_or_else(|| format!("aug_opt: PIR package missing top fn '{top_name}'"))?
        .clone();

    let mut xls_pkg = xlsynth::IrPackage::parse_ir(ir_text, None)
        .map_err(|e| format!("aug_opt: xlsynth parse_ir (analysis) failed: {e}"))?;
    xls_pkg
        .set_top_by_name(top_name)
        .map_err(|e| format!("aug_opt: set_top_by_name('{top_name}') failed: {e}"))?;
    let analysis = xls_pkg
        .create_ir_analysis()
        .map_err(|e| format!("aug_opt: create_ir_analysis failed: {e}"))?;
    let range_info = IrRangeInfo::build_from_analysis(&analysis, &top_fn)
        .map_err(|e| format!("aug_opt: building IrRangeInfo failed: {e}"))?;

    let (rewritten_top, rewrites_in_round) =
        apply_basis_rewrites_to_fn(&top_fn, Some(range_info.as_ref()));

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

    // Preserve the caller-supplied top in emitted IR text (especially for
    // aug-opt-only mode, where downstream tools may rely on the `top` marker).
    pir_pkg
        .set_top_fn(top_name)
        .map_err(|e| format!("aug_opt: internal error: set_top_fn('{top_name}') failed: {e}"))?;

    // Verify we did not introduce extension ops (basis-only contract).
    verify_no_extension_ops_in_package(&pir_pkg)?;

    Ok((pir_pkg.to_string(), rewrites_in_round))
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

fn apply_basis_rewrites_to_fn(
    f: &ir::Fn,
    range_info: Option<&IrRangeInfo>,
) -> (ir::Fn, AugOptRewriteStats) {
    let mut cloned = f.clone();
    let mut stats = AugOptRewriteStats::default();
    stats.guarded_sel_ne1_nor = rewrite_guarded_sel_ne_literal1_nor(&mut cloned);
    stats.lsb_of_shll = rewrite_lsb_of_shll_via_shift_is_zero(&mut cloned);
    stats.pow2_msb_compare_with_eq_tiebreak =
        rewrite_pow2_msb_compare_with_eq_tiebreak(&mut cloned);
    stats.eq_priority_sel_to_selector_predicate =
        rewrite_eq_priority_sel_to_selector_predicate(&mut cloned, range_info);
    stats.eq_add_zero_to_eq_rhs_sub = rewrite_eq_add_zero_to_eq_rhs_sub(&mut cloned);
    // Hoist predicate reductions/comparisons across sel/priority_sel first so
    // downstream rewrites can see through the selector.
    stats.predicate_hoist_across_select = rewrite_predicate_hoist_across_select(&mut cloned);
    stats.ne_shrl_slice_known_one_shift_nonzero =
        rewrite_ne_shrl_slice_known_one_shift_nonzero(&mut cloned, range_info);
    // Ensure textual IR is defs-before-uses by reordering body nodes into a
    // topological order (while preserving PIR layout invariants). This makes
    // it safe for rewrites to append new nodes.
    ir_utils::compact_and_toposort_in_place(&mut cloned)
        .expect("aug_opt: compact_and_toposort_in_place failed");
    (cloned, stats)
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

fn literal_one_hot_index(v: &IrValue) -> Option<usize> {
    let Ok(bits) = v.to_bits() else {
        return None;
    };
    let w = bits.get_bit_count();
    if w == 0 {
        return None;
    }
    let mut found: Option<usize> = None;
    for i in 0..w {
        if bits.get_bit(i).unwrap_or(false) {
            if found.is_some() {
                return None;
            }
            found = Some(i);
        }
    }
    found
}

fn literals_equal_bits(a: &IrValue, b: &IrValue) -> bool {
    match (a.to_bits(), b.to_bits()) {
        (Ok(ab), Ok(bb)) => ab == bb,
        _ => false,
    }
}

fn eq_against_literal(f: &ir::Fn, nr: NodeRef) -> Option<(NodeRef, NodeRef, IrValue)> {
    let NodePayload::Binop(Binop::Eq, a, b) = f.get_node(nr).payload.clone() else {
        return None;
    };
    let candidates = [(a, b), (b, a)];
    for (maybe_x, maybe_lit) in candidates {
        let NodePayload::Literal(lit_v) = f.get_node(maybe_lit).payload.clone() else {
            continue;
        };
        return Some((maybe_x, maybe_lit, lit_v));
    }
    None
}

fn ugt_against_literal_rhs(f: &ir::Fn, nr: NodeRef) -> Option<(NodeRef, NodeRef, IrValue)> {
    let NodePayload::Binop(Binop::Ugt, x, lit_nr) = f.get_node(nr).payload.clone() else {
        return None;
    };
    let NodePayload::Literal(lit_v) = f.get_node(lit_nr).payload.clone() else {
        return None;
    };
    Some((x, lit_nr, lit_v))
}

/// Rewrite a specific power-of-two compare shape:
///
/// `or(ugt(x, 2^(w-1)), and(eq(x, 2^(w-1)), hi))`
///   →
/// `and(msb(x), or(or_reduce(x[0..w-1)), hi))`
///
/// Notes:
/// - We intentionally require the power-of-two to be the MSB of `x` (i.e.
///   `2^(w-1)`) to keep the logic simple and obviously correct.
/// - This targets the common "truncated compare with tie-breaker bit" pattern.
fn rewrite_pow2_msb_compare_with_eq_tiebreak(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for or_index in 0..f.nodes.len() {
        let NodePayload::Nary(NaryOp::Or, operands) = f.nodes[or_index].payload.clone() else {
            continue;
        };
        if operands.len() != 2 || f.nodes[or_index].ty != Type::Bits(1) {
            continue;
        }

        // Try both operand orders: (ugt, and) or (and, ugt).
        let candidates = [(operands[0], operands[1]), (operands[1], operands[0])];
        let mut matched: Option<(NodeRef, NodeRef, NodeRef, IrValue)> = None; // (x, hi, lit_nr, lit_v)

        for (maybe_ugt, maybe_and) in candidates {
            let Some((x_ugt, _lit_ugt_nr, lit_ugt_v)) = ugt_against_literal_rhs(f, maybe_ugt)
            else {
                continue;
            };

            let NodePayload::Nary(NaryOp::And, and_ops) = f.get_node(maybe_and).payload.clone()
            else {
                continue;
            };
            if and_ops.len() != 2 {
                continue;
            }

            // The AND must be (eq(x, lit), hi) in either order.
            let and_candidates = [(and_ops[0], and_ops[1]), (and_ops[1], and_ops[0])];
            for (maybe_eq, maybe_hi) in and_candidates {
                let Some((x_eq, lit_eq_nr, lit_eq_v)) = eq_against_literal(f, maybe_eq) else {
                    continue;
                };
                if x_eq != x_ugt || !literals_equal_bits(&lit_eq_v, &lit_ugt_v) {
                    continue;
                }
                if f.get_node_ty(maybe_hi) != &Type::Bits(1) {
                    continue;
                }

                matched = Some((x_ugt, maybe_hi, lit_eq_nr, lit_eq_v));
                break;
            }
            if matched.is_some() {
                break;
            }
        }

        let Some((x, hi, _lit_nr, lit_v)) = matched else {
            continue;
        };

        let w = f.get_node_ty(x).bit_count();
        if w < 2 || f.get_node_ty(x) != &Type::Bits(w) {
            continue;
        }

        // Require the literal to be exactly the MSB power-of-two: 2^(w-1).
        let Some(one_hot_idx) = literal_one_hot_index(&lit_v) else {
            continue;
        };
        if one_hot_idx != w.saturating_sub(1) || f.get_node_ty(x) != &Type::Bits(w) {
            continue;
        }

        // msb = x[w-1]
        let msb = push_node(
            f,
            Type::Bits(1),
            NodePayload::BitSlice {
                arg: x,
                start: w - 1,
                width: 1,
            },
        );

        // lo = x[0..w-1)
        let lo = push_node(
            f,
            Type::Bits(w - 1),
            NodePayload::BitSlice {
                arg: x,
                start: 0,
                width: w - 1,
            },
        );

        // lo_nz = or_reduce(lo)
        let lo_nz = push_node(f, Type::Bits(1), NodePayload::Unop(Unop::OrReduce, lo));

        // rhs = lo_nz | hi
        let rhs = push_node(
            f,
            Type::Bits(1),
            NodePayload::Nary(NaryOp::Or, vec![lo_nz, hi]),
        );

        // out = msb & rhs
        f.nodes[or_index].payload = NodePayload::Nary(NaryOp::And, vec![msb, rhs]);
        f.nodes[or_index].ty = Type::Bits(1);
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

fn proves_node_ne_bits_literal(
    range_info: &IrRangeInfo,
    node_text_id: usize,
    lit: &IrValue,
) -> bool {
    let Ok(lit_bits) = lit.to_bits() else {
        return false;
    };
    let Some(info) = range_info.get(node_text_id) else {
        return false;
    };

    if let Some(intervals) = info.intervals.as_ref() {
        let lit_is_possible = intervals
            .iter()
            .any(|it| it.lo.ule(&lit_bits) && lit_bits.ule(&it.hi));
        return !lit_is_possible;
    }

    if let Some(k) = info.known_bits.as_ref() {
        let w = k.mask.get_bit_count();
        if w != lit_bits.get_bit_count() {
            return false;
        }
        for i in 0..w {
            let is_known = k.mask.get_bit(i).unwrap_or(false);
            if !is_known {
                continue;
            }
            let kb = k.value.get_bit(i).unwrap_or(false);
            let lb = lit_bits.get_bit(i).unwrap_or(false);
            if kb != lb {
                return true;
            }
        }
    }

    false
}

/// Rewrites:
///
/// `eq(priority_sel(sel, cases=[c0..cN-1], default=d), literal(L))`
///   →
/// `or(and(selected_0(sel), eq(c0, L)), ..., and(selected_{N-1}(sel),
/// eq(c{N-1}, L)))`
///
/// When analysis proves `d != L`, the default term can be omitted safely.
fn rewrite_eq_priority_sel_to_selector_predicate(
    f: &mut ir::Fn,
    range_info: Option<&IrRangeInfo>,
) -> usize {
    let Some(range_info) = range_info else {
        return 0;
    };

    let mut rewrites = 0usize;

    for eq_index in 0..f.nodes.len() {
        let NodePayload::Binop(Binop::Eq, a, b) = f.nodes[eq_index].payload.clone() else {
            continue;
        };

        // Try both operand orders: eq(priority_sel(...), literal) or eq(literal,
        // priority_sel(...)).
        let candidates = [(a, b), (b, a)];
        let mut matched: Option<(NodeRef, NodeRef, NodeRef, Vec<NodeRef>, NodeRef, IrValue)> = None;

        for (maybe_ps, maybe_lit) in candidates {
            let NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } = f.get_node(maybe_ps).payload.clone()
            else {
                continue;
            };
            let Some(default_nr) = default else {
                continue;
            };
            let NodePayload::Literal(lit_v) = f.get_node(maybe_lit).payload.clone() else {
                continue;
            };

            matched = Some((maybe_ps, selector, default_nr, cases, maybe_lit, lit_v));
            break;
        }

        let Some((_ps_nr, selector, default_nr, cases, lit_nr, lit_v)) = matched else {
            continue;
        };

        let selector_w = f.get_node_ty(selector).bit_count();
        if selector_w == 0 || selector_w != cases.len() {
            continue;
        }

        // Require analysis to prove the default cannot equal the literal.
        let default_text_id = f.get_node(default_nr).text_id;
        if !proves_node_ne_bits_literal(range_info, default_text_id, &lit_v) {
            continue;
        }

        let mut terms: Vec<NodeRef> = Vec::new();
        for i in 0..cases.len() {
            // selected_i(sel) = bit_i(sel) & !or_reduce(sel[0..i])
            let bit_i: NodeRef =
                if selector_w == 1 && i == 0 && *f.get_node_ty(selector) == Type::Bits(1) {
                    selector
                } else {
                    push_node(
                        f,
                        Type::Bits(1),
                        NodePayload::BitSlice {
                            arg: selector,
                            start: i,
                            width: 1,
                        },
                    )
                };

            let selected_i: NodeRef = if i == 0 {
                bit_i
            } else {
                let lower_bits = push_node(
                    f,
                    Type::Bits(i),
                    NodePayload::BitSlice {
                        arg: selector,
                        start: 0,
                        width: i,
                    },
                );
                let lower_any = push_node(
                    f,
                    Type::Bits(1),
                    NodePayload::Unop(Unop::OrReduce, lower_bits),
                );
                let not_lower_any =
                    push_node(f, Type::Bits(1), NodePayload::Unop(Unop::Not, lower_any));
                push_node(
                    f,
                    Type::Bits(1),
                    NodePayload::Nary(NaryOp::And, vec![bit_i, not_lower_any]),
                )
            };

            let term = if cases[i] == lit_nr {
                selected_i
            } else {
                let eq_case = push_node(
                    f,
                    Type::Bits(1),
                    NodePayload::Binop(Binop::Eq, cases[i], lit_nr),
                );
                push_node(
                    f,
                    Type::Bits(1),
                    NodePayload::Nary(NaryOp::And, vec![selected_i, eq_case]),
                )
            };

            terms.push(term);
        }

        if terms.is_empty() {
            continue;
        }

        f.nodes[eq_index].payload = if terms.len() == 1 {
            NodePayload::Unop(Unop::Identity, terms[0])
        } else {
            NodePayload::Nary(NaryOp::Or, terms)
        };
        f.nodes[eq_index].ty = Type::Bits(1);
        rewrites += 1;
    }

    rewrites
}

fn is_bits_literal_zero(v: &IrValue) -> bool {
    let Ok(bits) = v.to_bits() else {
        return false;
    };
    for i in 0..bits.get_bit_count() {
        if bits.get_bit(i).unwrap_or(false) {
            return false;
        }
    }
    true
}

fn is_get_param_node(f: &ir::Fn, nr: NodeRef) -> bool {
    matches!(f.get_node(nr).payload, NodePayload::GetParam(_))
}

#[derive(Debug, Clone, Copy)]
enum PredicateShape {
    Reduce(Unop),
    CompareConst {
        binop: Binop,
        const_nr: NodeRef,
        const_on_lhs: bool,
        value_width: usize,
    },
}

fn is_reduce_predicate(unop: Unop) -> bool {
    matches!(unop, Unop::OrReduce | Unop::AndReduce | Unop::XorReduce)
}

fn is_compare_predicate(binop: Binop) -> bool {
    matches!(
        binop,
        Binop::Eq
            | Binop::Ne
            | Binop::Uge
            | Binop::Ugt
            | Binop::Ult
            | Binop::Ule
            | Binop::Sgt
            | Binop::Sge
            | Binop::Slt
            | Binop::Sle
    )
}

fn match_predicate_shape(f: &ir::Fn, nr: NodeRef) -> Option<(PredicateShape, NodeRef)> {
    let node = f.get_node(nr);
    if node.ty != Type::Bits(1) {
        return None;
    }
    match node.payload.clone() {
        NodePayload::Unop(unop, arg) if is_reduce_predicate(unop) => {
            if f.get_node_ty(arg).bit_count() == 0 {
                return None;
            }
            Some((PredicateShape::Reduce(unop), arg))
        }
        NodePayload::Binop(binop, a, b) if is_compare_predicate(binop) => {
            let (value_nr, const_nr, const_on_lhs) = match f.get_node(a).payload {
                NodePayload::Literal(_) => (b, a, true),
                _ => match f.get_node(b).payload {
                    NodePayload::Literal(_) => (a, b, false),
                    _ => return None,
                },
            };
            let value_ty = f.get_node_ty(value_nr);
            let const_ty = f.get_node_ty(const_nr);
            let value_w = value_ty.bit_count();
            if value_w == 0 || *const_ty != Type::Bits(value_w) {
                return None;
            }
            Some((
                PredicateShape::CompareConst {
                    binop,
                    const_nr,
                    const_on_lhs,
                    value_width: value_w,
                },
                value_nr,
            ))
        }
        _ => None,
    }
}

fn build_predicate_on_value(f: &mut ir::Fn, shape: PredicateShape, value_nr: NodeRef) -> NodeRef {
    match shape {
        PredicateShape::Reduce(unop) => {
            push_node(f, Type::Bits(1), NodePayload::Unop(unop, value_nr))
        }
        PredicateShape::CompareConst {
            binop,
            const_nr,
            const_on_lhs,
            value_width,
        } => {
            debug_assert_eq!(f.get_node_ty(value_nr), &Type::Bits(value_width));
            if const_on_lhs {
                push_node(
                    f,
                    Type::Bits(1),
                    NodePayload::Binop(binop, const_nr, value_nr),
                )
            } else {
                push_node(
                    f,
                    Type::Bits(1),
                    NodePayload::Binop(binop, value_nr, const_nr),
                )
            }
        }
    }
}

fn count_users(f: &ir::Fn, target: NodeRef) -> usize {
    f.nodes
        .iter()
        .filter(|n| ir_utils::operands(&n.payload).contains(&target))
        .count()
}

/// Hoist predicate reductions/comparisons across selection:
///
/// - `pred(sel(s, cases=[a,b]))` → `sel(s, cases=[pred(a), pred(b)])`
/// - `pred(priority_sel(sel, cases, default))` → `priority_sel(sel,
///   cases=map(pred), default=pred(default))`
///
/// This keeps the result as a selector of predicates, which is generally
/// friendlier to downstream lowering than SOP expansion.
fn rewrite_predicate_hoist_across_select(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for idx in 0..f.nodes.len() {
        let nr = NodeRef { index: idx };
        let Some((shape, value_nr)) = match_predicate_shape(f, nr) else {
            continue;
        };
        // For compare-to-constant predicates, only hoist when the selected value
        // has a single user. This bounds compare replication in membership-like
        // shapes (e.g. OR-of-many EQs on the same selected value).
        if matches!(shape, PredicateShape::CompareConst { .. }) && count_users(f, value_nr) != 1 {
            continue;
        }

        match f.get_node(value_nr).payload.clone() {
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                // Keep this narrow and predictable: 2-way boolean selects only.
                if f.get_node_ty(selector) != &Type::Bits(1) || cases.len() != 2 {
                    continue;
                }
                let case_ty = f.get_node_ty(cases[0]).clone();
                if cases.iter().any(|c| f.get_node_ty(*c) != &case_ty) {
                    continue;
                }
                if let Some(d) = default {
                    if f.get_node_ty(d) != &case_ty {
                        continue;
                    }
                }

                let pred_cases: Vec<NodeRef> = cases
                    .iter()
                    .map(|c| build_predicate_on_value(f, shape, *c))
                    .collect();
                let pred_default = default.map(|d| build_predicate_on_value(f, shape, d));

                f.nodes[idx].payload = NodePayload::Sel {
                    selector,
                    cases: pred_cases,
                    default: pred_default,
                };
                f.nodes[idx].ty = Type::Bits(1);
                rewrites += 1;
            }
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                let Some(default_nr) = default else {
                    continue;
                };
                let case_ty = f.get_node_ty(default_nr).clone();
                if cases.iter().any(|c| f.get_node_ty(*c) != &case_ty) {
                    continue;
                }

                let pred_cases: Vec<NodeRef> = cases
                    .iter()
                    .map(|c| build_predicate_on_value(f, shape, *c))
                    .collect();
                let pred_default = build_predicate_on_value(f, shape, default_nr);

                f.nodes[idx].payload = NodePayload::PrioritySel {
                    selector,
                    cases: pred_cases,
                    default: Some(pred_default),
                };
                f.nodes[idx].ty = Type::Bits(1);
                rewrites += 1;
            }
            _ => {}
        }
    }

    rewrites
}

fn known_one_positions(range_info: &IrRangeInfo, text_id: usize) -> Vec<usize> {
    let Some(info) = range_info.get(text_id) else {
        return Vec::new();
    };
    let Some(k) = info.known_bits.as_ref() else {
        return Vec::new();
    };
    let w = k.mask.get_bit_count();
    let mut out = Vec::new();
    for i in 0..w {
        let is_known = k.mask.get_bit(i).unwrap_or(false);
        if !is_known {
            continue;
        }
        let is_one = k.value.get_bit(i).unwrap_or(false);
        if is_one {
            out.push(i);
        }
    }
    out
}

fn unsigned_bounds_u64(range_info: &IrRangeInfo, text_id: usize) -> Option<(u64, u64)> {
    let info = range_info.get(text_id)?;
    let min_bits = info.unsigned_min.as_ref()?;
    let max_bits = info.unsigned_max.as_ref()?;
    let min_u = min_bits.to_u64().ok()?;
    let max_u = max_bits.to_u64().ok()?;
    Some((min_u, max_u))
}

fn const_fits_in_width(v: u64, width: usize) -> bool {
    if width == 0 {
        return v == 0;
    }
    if width >= 64 {
        return true;
    }
    v < (1u64 << width)
}

/// Given a set of inclusive intervals, returns true iff they cover every point
/// in [lo, hi] (inclusive).
fn intervals_cover_range(mut intervals: Vec<(usize, usize)>, lo: usize, hi: usize) -> bool {
    if lo > hi {
        return true;
    }
    if intervals.is_empty() {
        return false;
    }
    intervals.sort_by_key(|(l, _)| *l);
    let mut cur = lo;
    for (l, h) in intervals {
        if h < cur {
            continue;
        }
        if l > cur {
            return false;
        }
        cur = cur.max(h.saturating_add(1));
        if cur > hi {
            return true;
        }
    }
    cur > hi
}

/// Rewrite (range-info guarded):
///
/// `ne(bit_slice(shrl(x, s), start=0, width=W), 0)`
///   →
/// `or(and(s in [1, hi]), and(eq(s, 0), ne(bit_slice(x, 0, W), 0)))`
///
/// where `hi = min(s_max, x_width-1)` and known-bits proves that every shift
/// amount in [1, hi] must land at least one provably-one bit into the low-W
/// slice.
///
/// This avoids materializing a large dynamic shifter cone when the
/// nonzero-after-shift predicate can be discharged cheaply by known bits and
/// shift bounds.
fn rewrite_ne_shrl_slice_known_one_shift_nonzero(
    f: &mut ir::Fn,
    range_info: Option<&IrRangeInfo>,
) -> usize {
    let Some(range_info) = range_info else {
        return 0;
    };
    let mut rewrites = 0usize;

    for ne_index in 0..f.nodes.len() {
        let NodePayload::Binop(Binop::Ne, a, b) = f.nodes[ne_index].payload.clone() else {
            continue;
        };

        // Match ne(slice, 0) in either operand order.
        let candidates = [(a, b), (b, a)];
        let mut matched: Option<(NodeRef, NodeRef, usize, usize)> = None;
        // (x, s, W, shrl_width)

        for (maybe_slice, maybe_lit0) in candidates {
            let NodePayload::BitSlice { arg, start, width } =
                f.get_node(maybe_slice).payload.clone()
            else {
                continue;
            };
            if start != 0 || width == 0 {
                continue;
            }
            let NodePayload::Binop(Binop::Shrl, x, s) = f.get_node(arg).payload.clone() else {
                continue;
            };
            let NodePayload::Literal(lit_v) = f.get_node(maybe_lit0).payload.clone() else {
                continue;
            };
            if !is_bits_literal_zero(&lit_v) {
                continue;
            }
            let slice_ty = f.get_node_ty(maybe_slice);
            if *slice_ty != Type::Bits(width) || f.get_node_ty(maybe_lit0) != slice_ty {
                continue;
            }
            matched = Some((x, s, width, f.get_node_ty(arg).bit_count()));
            break;
        }

        let Some((x, s, w, shrl_width)) = matched else {
            continue;
        };
        if shrl_width == 0 || shrl_width < w {
            continue;
        }

        let s_width = f.get_node_ty(s).bit_count();
        if s_width == 0 || s_width > 64 {
            continue;
        }

        let x_text_id = f.get_node(x).text_id;
        let s_text_id = f.get_node(s).text_id;
        let Some((s_min, s_max)) = unsigned_bounds_u64(range_info, s_text_id) else {
            continue;
        };

        // We only reason about shifts that could still affect the low-W slice.
        let hi_req = s_max.min((shrl_width.saturating_sub(1)) as u64);
        if hi_req == 0 {
            // Only shift=0 is possible; no win here.
            continue;
        }

        let lo_req = s_min.max(1);
        if lo_req > hi_req {
            continue;
        }

        // Gather known-one bit positions in x that are above the low-W slice.
        let known_ones = known_one_positions(range_info, x_text_id);
        let upper_known_ones: Vec<usize> = known_ones
            .into_iter()
            .filter(|k| *k >= w && *k < shrl_width)
            .collect();
        if upper_known_ones.is_empty() {
            continue;
        }

        // For a known-one bit at position k, the low-W slice is guaranteed
        // nonzero when s is in [k-W+1, k].
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        for k in upper_known_ones {
            let lb = k.saturating_sub(w.saturating_sub(1));
            let ub = k;
            intervals.push((lb, ub));
        }

        let lo_req_usize = usize::try_from(lo_req).ok();
        let hi_req_usize = usize::try_from(hi_req).ok();
        let (Some(lo_req_u), Some(hi_req_u)) = (lo_req_usize, hi_req_usize) else {
            continue;
        };

        // Conservative proof: assume s can take any value in [lo_req, hi_req].
        // Only rewrite if known ones cover the full required interval.
        if !intervals_cover_range(intervals, lo_req_u, hi_req_u) {
            continue;
        }

        // Ensure constants fit the shift amount type.
        if !const_fits_in_width(lo_req, s_width) || !const_fits_in_width(hi_req, s_width) {
            continue;
        }

        let lo_lit = push_ubits_literal(f, s_width, lo_req);
        let hi_lit = push_ubits_literal(f, s_width, hi_req);
        let zero_lit = push_ubits_literal(f, s_width, 0);

        // s in [lo_req, hi_req]
        let s_ge_lo = push_node(f, Type::Bits(1), NodePayload::Binop(Binop::Uge, s, lo_lit));
        let s_le_hi = push_node(f, Type::Bits(1), NodePayload::Binop(Binop::Ule, s, hi_lit));
        let s_in_interval = push_node(
            f,
            Type::Bits(1),
            NodePayload::Nary(NaryOp::And, vec![s_ge_lo, s_le_hi]),
        );

        // s == 0 && ne(bit_slice(x, 0, W), 0)
        let s_eq_zero = push_node(f, Type::Bits(1), NodePayload::Binop(Binop::Eq, s, zero_lit));
        let low_slice = push_node(
            f,
            Type::Bits(w),
            NodePayload::BitSlice {
                arg: x,
                start: 0,
                width: w,
            },
        );
        let low_zero = push_ubits_literal(f, w, 0);
        let low_ne_zero = push_node(
            f,
            Type::Bits(1),
            NodePayload::Binop(Binop::Ne, low_slice, low_zero),
        );
        let s0_and_low_nonzero = push_node(
            f,
            Type::Bits(1),
            NodePayload::Nary(NaryOp::And, vec![s_eq_zero, low_ne_zero]),
        );

        f.nodes[ne_index].payload =
            NodePayload::Nary(NaryOp::Or, vec![s_in_interval, s0_and_low_nonzero]);
        f.nodes[ne_index].ty = Type::Bits(1);
        rewrites += 1;
    }

    rewrites
}

/// Rewrite:
///
/// `eq(add(x, y), literal(0))`
///   →
/// `eq(y, sub(literal(0), x))`
///
/// This is a semantics-preserving normalization in fixed-width modular
/// arithmetic. It can unlock further simplifications when `x` is range-bounded
/// (e.g. a small set of possible constants) and libxls does not solve the
/// modular equation directly.
fn rewrite_eq_add_zero_to_eq_rhs_sub(f: &mut ir::Fn) -> usize {
    let mut rewrites = 0usize;

    for eq_index in 0..f.nodes.len() {
        let NodePayload::Binop(Binop::Eq, a, b) = f.nodes[eq_index].payload.clone() else {
            continue;
        };

        // Try both orders: eq(add(..), 0) or eq(0, add(..)).
        let candidates = [(a, b), (b, a)];
        let mut matched: Option<(NodeRef, NodeRef, NodeRef)> = None; // (x, y, lit0)

        for (maybe_add, maybe_lit0) in candidates {
            let NodePayload::Binop(Binop::Add, x, y) = f.get_node(maybe_add).payload.clone() else {
                continue;
            };
            let NodePayload::Literal(lit_v) = f.get_node(maybe_lit0).payload.clone() else {
                continue;
            };

            let w = f.get_node_ty(maybe_add).bit_count();
            if w == 0 || f.get_node_ty(maybe_lit0) != &Type::Bits(w) {
                continue;
            }
            if !is_bits_literal_zero(&lit_v) {
                continue;
            }

            matched = Some((x, y, maybe_lit0));
            break;
        }

        let Some((x, y, lit0)) = matched else {
            continue;
        };

        // Choose which operand to "solve for" deterministically: prefer a param
        // node when available.
        let (solve_for, other) = match (is_get_param_node(f, x), is_get_param_node(f, y)) {
            (true, false) => (x, y),
            (false, true) => (y, x),
            _ => (y, x),
        };

        let w = f.get_node_ty(other).bit_count();
        if w == 0
            || f.get_node_ty(lit0) != &Type::Bits(w)
            || f.get_node_ty(solve_for) != &Type::Bits(w)
        {
            continue;
        }

        let rhs_sub = push_node(
            f,
            Type::Bits(w),
            NodePayload::Binop(Binop::Sub, lit0, other),
        );

        f.nodes[eq_index].payload = NodePayload::Binop(Binop::Eq, solve_for, rhs_sub);
        f.nodes[eq_index].ty = Type::Bits(1);
        rewrites += 1;
    }

    rewrites
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::quickcheck_ir_text_fn_equivalence_ubits_le64;

    fn resolve_identity<'a>(f: &'a ir::Fn, mut nr: NodeRef) -> &'a ir::Node {
        loop {
            let n = f.get_node(nr);
            match n.payload {
                NodePayload::Unop(Unop::Identity, inner) => {
                    nr = inner;
                }
                _ => return n,
            }
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
                mode: AugOptMode::Sandwich,
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
                mode: AugOptMode::Sandwich,
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
    fn aug_opt_hoists_or_reduce_across_sel() {
        let ir_text = r#"package sel_reduce

top fn cone(s: bits[1] id=1, a: bits[8] id=2, b: bits[8] id=3) -> bits[1] {
  sel.4: bits[8] = sel(s, cases=[a, b], id=4)
  ret or_reduce.5: bits[1] = or_reduce(sel.4, id=5)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");

        let ret_nr = f.ret_node_ref.expect("ret node");
        let ret_node = resolve_identity(f, ret_nr);
        match &ret_node.payload {
            NodePayload::Sel {
                selector, cases, ..
            } => {
                assert_eq!(f.get_node_ty(*selector), &Type::Bits(1));
                assert_eq!(cases.len(), 2, "expected 2-case sel; got:\n{}", out_text);
                for c in cases {
                    let cn = f.get_node(*c);
                    assert!(
                        matches!(cn.payload, NodePayload::Unop(Unop::OrReduce, _)),
                        "expected sel case to be or_reduce(..); got {:?}\noutput:\n{}",
                        cn.payload,
                        out_text
                    );
                }
            }
            other => panic!(
                "expected ret to be sel(or_reduce(..)); got {:?}\noutput:\n{}",
                other, out_text
            ),
        }

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[1, 8, 8],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_hoists_const_compare_across_priority_sel() {
        let ir_text = r#"package prio_cmp

top fn cone(sel: bits[2] id=1, a: bits[6] id=2, b: bits[6] id=3, d: bits[6] id=4) -> bits[1] {
  priority_sel.5: bits[6] = priority_sel(sel, cases=[a, b], default=d, id=5)
  literal.6: bits[6] = literal(value=21, id=6)
  ret eq.7: bits[1] = eq(priority_sel.5, literal.6, id=7)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");

        let ret_nr = f.ret_node_ref.expect("ret node");
        let ret_node = resolve_identity(f, ret_nr);
        match &ret_node.payload {
            NodePayload::PrioritySel { cases, default, .. } => {
                assert_eq!(cases.len(), 2);
                let default = default.expect("priority_sel should retain default");
                for c in cases.iter().chain([default].iter()) {
                    let cn = resolve_identity(f, *c);
                    assert!(
                        matches!(cn.payload, NodePayload::Binop(Binop::Eq, _, _)),
                        "expected priority_sel arm to be eq(..); got {:?}\noutput:\n{}",
                        cn.payload,
                        out_text
                    );
                }
            }
            other => panic!(
                "expected ret to be priority_sel(eq(..)); got {:?}\noutput:\n{}",
                other, out_text
            ),
        }

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[2, 6, 6, 6],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_does_not_hoist_compare_when_selected_value_has_multiple_users() {
        let ir_text = r#"package no_hoist_multi_user

top fn cone(s: bits[1] id=1, a: bits[3] id=2, b: bits[3] id=3) -> bits[1] {
  sel.4: bits[3] = sel(s, cases=[a, b], id=4)
  literal.5: bits[3] = literal(value=1, id=5)
  literal.6: bits[3] = literal(value=2, id=6)
  eq.7: bits[1] = eq(sel.4, literal.5, id=7)
  eq.8: bits[1] = eq(sel.4, literal.6, id=8)
  ret or.9: bits[1] = or(eq.7, eq.8, id=9)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");

        let mut saw_eq_over_sel = 0usize;
        for node in &f.nodes {
            if let NodePayload::Binop(Binop::Eq, lhs, rhs) = node.payload {
                let lhs_is_sel = matches!(f.get_node(lhs).payload, NodePayload::Sel { .. });
                let rhs_is_sel = matches!(f.get_node(rhs).payload, NodePayload::Sel { .. });
                if lhs_is_sel || rhs_is_sel {
                    saw_eq_over_sel += 1;
                }
            }
        }
        assert!(
            saw_eq_over_sel >= 2,
            "expected compares to remain over selected value when it has multiple users;\noutput:\n{}",
            out_text
        );

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[1, 3, 3],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_aug_opt_only_rewrites_without_liblxls_dce() {
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
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt aug-opt-only");

        // In aug-opt-only mode, we still expect the Nor rewrite to have fired...
        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");
        let has_nor = f
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::Nary(NaryOp::Nor, _)));
        assert!(!has_nor, "expected nor to be rewritten; got:\n{}", out_text);

        // ...but we do not expect libxls DCE to have removed the Sel.
        let has_sel = f
            .nodes
            .iter()
            .any(|n| matches!(n.payload, NodePayload::Sel { .. }));
        assert!(
            has_sel,
            "expected sel to remain without libxls opt; got:\n{}",
            out_text
        );

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[1, 8, 8],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_aug_opt_only_preserves_requested_top_in_output() {
        let ir_text = r#"package top_swap

top fn a(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}

fn b(y: bits[1] id=10) -> bits[1] {
  ret identity.11: bits[1] = identity(y, id=11)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("b"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt aug-opt-only");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        match &pkg.top {
            Some((name, ir::MemberType::Function)) => assert_eq!(name, "b"),
            other => panic!("expected top fn 'b', got {:?}", other),
        }
    }

    #[test]
    fn aug_opt_rewrites_eq_priority_sel_to_selector_predicate_when_default_cannot_match() {
        let ir_text = r#"package prio_eq

top fn cone(sel: bits[1] id=1, x: bits[5] id=2) -> bits[1] {
  literal.3: bits[6] = literal(value=0, id=3)
  literal.4: bits[1] = literal(value=1, id=4)
  concat.5: bits[6] = concat(literal.4, x, id=5)
  priority_sel.6: bits[6] = priority_sel(sel, cases=[literal.3], default=concat.5, id=6)
  ret eq.7: bits[1] = eq(priority_sel.6, literal.3, id=7)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");

        // In PirOnly mode we do not run libxls optimization after rewriting, so we
        // do not expect dead nodes to be removed. Instead, assert that the return
        // no longer depends on the priority_sel and is rewritten to the selector.
        let ret_nr = f.ret_node_ref.expect("ret node");
        let ret_node = f.get_node(ret_nr);
        assert_eq!(ret_node.ty, Type::Bits(1));
        match &ret_node.payload {
            NodePayload::Unop(Unop::Identity, op) => {
                // Param nodes are at indices 1..=params.len() in signature order.
                assert_eq!(*op, NodeRef { index: 1 });
            }
            other => panic!("unexpected ret payload: {:?}\noutput:\n{}", other, out_text),
        }

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[1, 5],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_rewrites_eq_add_zero_to_eq_rhs_sub_when_addend_is_small_set() {
        let ir_text = r#"package add_eq_zero_smallset

top fn cone(leaf_52: bits[2] id=1, y: bits[5] id=2) -> bits[1] {
  literal.3: bits[3] = literal(value=5, id=3)
  not.4: bits[2] = not(leaf_52, id=4)
  concat.5: bits[5] = concat(literal.3, not.4, id=5)
  add.6: bits[5] = add(concat.5, y, id=6)
  literal.7: bits[5] = literal(value=0, id=7)
  ret eq.8: bits[1] = eq(add.6, literal.7, id=8)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");

        let ret_nr = f.ret_node_ref.expect("ret node");
        let ret_node = f.get_node(ret_nr);
        assert_eq!(ret_node.ty, Type::Bits(1));
        let NodePayload::Binop(Binop::Eq, lhs, rhs) = ret_node.payload else {
            panic!(
                "expected ret to be eq(..); got {:?}\noutput:\n{}",
                ret_node.payload, out_text
            );
        };

        // We should solve for the `y` parameter (second param => node index 2).
        assert_eq!(lhs, NodeRef { index: 2 });
        let rhs_node = f.get_node(rhs);
        let NodePayload::Binop(Binop::Sub, sub_lhs, sub_rhs) = rhs_node.payload else {
            panic!(
                "expected rhs to be sub(..); got {:?}\noutput:\n{}",
                rhs_node.payload, out_text
            );
        };

        // Ensure we reused the existing literal-0 node.
        assert_eq!(f.get_node(sub_lhs).text_id, 7);
        // Ensure we subtracted the small-set addend.
        assert_eq!(f.get_node(sub_rhs).text_id, 5);

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[2, 5],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_rewrites_pow2_msb_compare_with_eq_tiebreak() {
        let ir_text = r#"package pow2_cmp

top fn main(a: bits[10] id=1, b: bits[10] id=2) -> bits[1] {
  smul.17: bits[20] = smul(a, b, id=17)
  bit_slice.8: bits[9] = bit_slice(smul.17, start=0, width=9, id=8)
  literal.9: bits[9] = literal(value=256, id=9)
  eq.10: bits[1] = eq(bit_slice.8, literal.9, id=10)
  bit_slice.11: bits[1] = bit_slice(smul.17, start=9, width=1, id=11)
  ugt.12: bits[1] = ugt(bit_slice.8, literal.9, id=12)
  and.13: bits[1] = and(eq.10, bit_slice.11, id=13)
  ret or.14: bits[1] = or(ugt.12, and.13, id=14)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("main"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("main").expect("top fn");

        let ret_nr = f.ret_node_ref.expect("ret node");
        let ret_node = f.get_node(ret_nr);
        assert_eq!(ret_node.ty, Type::Bits(1));
        match &ret_node.payload {
            NodePayload::Nary(NaryOp::And, ops) => {
                assert_eq!(ops.len(), 2, "expected 2-input and; got:\n{}", out_text);
            }
            other => panic!(
                "expected ret to be an and(..) after rewrite; got {:?}\noutput:\n{}",
                other, out_text
            ),
        }

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "main",
            &[10, 10],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_rewrites_ne_shrl_slice_with_upper_known_ones_and_bounded_shift() {
        // Known-one bits are injected above the low-W slice. When the shift
        // amount is in [1, x_width-1], those known ones must land in-range.
        let ir_text = r#"package shrl_known_one

top fn cone(x: bits[8] id=1, a: bits[7] id=2) -> bits[1] {
  literal.3: bits[2] = literal(value=3, id=3)
  concat.4: bits[10] = concat(literal.3, x, id=4)
  literal.5: bits[7] = literal(value=15, id=5)
  and.6: bits[7] = and(a, literal.5, id=6)
  shrl.7: bits[10] = shrl(concat.4, and.6, id=7)
  bit_slice.8: bits[8] = bit_slice(shrl.7, start=0, width=8, id=8)
  literal.9: bits[8] = literal(value=0, id=9)
  ret ne.10: bits[1] = ne(bit_slice.8, literal.9, id=10)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("cone"),
            AugOptOptions {
                enable: true,
                rounds: 1,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        let f = pkg.get_fn("cone").expect("top fn");
        let ret_nr = f.ret_node_ref.expect("ret node");
        let ret_node = f.get_node(ret_nr);
        assert_eq!(ret_node.ty, Type::Bits(1));
        // The return should no longer be a direct ne(bit_slice(shrl(...)), 0).
        assert!(
            !matches!(ret_node.payload, NodePayload::Binop(Binop::Ne, _, _)),
            "expected ret to be predicate-based; got {:?}\noutput:\n{}",
            ret_node.payload,
            out_text
        );

        quickcheck_ir_text_fn_equivalence_ubits_le64(
            ir_text,
            &out_text,
            "cone",
            &[8, 7],
            /* random_samples= */ 2000,
        );
    }

    #[test]
    fn aug_opt_aug_opt_only_rounds_zero_sets_requested_top_in_output() {
        let ir_text = r#"package top_swap_zero_rounds

top fn a(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}

fn b(y: bits[1] id=10) -> bits[1] {
  ret identity.11: bits[1] = identity(y, id=11)
}
"#;

        let out_text = run_aug_opt_over_ir_text(
            ir_text,
            Some("b"),
            AugOptOptions {
                enable: true,
                rounds: 0,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect("aug opt aug-opt-only rounds=0");

        let mut p = ir_parser::Parser::new(&out_text);
        let pkg = p.parse_and_validate_package().expect("parse/validate");
        match &pkg.top {
            Some((name, ir::MemberType::Function)) => assert_eq!(name, "b"),
            other => panic!("expected top fn 'b', got {:?}", other),
        }
    }

    #[test]
    fn aug_opt_aug_opt_only_rounds_zero_invalid_top_is_error() {
        let ir_text = r#"package top_swap_zero_rounds_invalid_top

top fn a(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}
"#;

        let err = run_aug_opt_over_ir_text(
            ir_text,
            Some("nope"),
            AugOptOptions {
                enable: true,
                rounds: 0,
                mode: AugOptMode::PirOnly,
            },
        )
        .expect_err("expected invalid top to error");
        assert_eq!(err, "aug_opt: PIR package missing top fn 'nope'");
    }
}
