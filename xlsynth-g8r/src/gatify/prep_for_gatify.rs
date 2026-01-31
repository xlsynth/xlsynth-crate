// SPDX-License-Identifier: Apache-2.0

//! Preparatory PIR "feng shui" before gatification.
//!
//! This module exists to rewrite PIR into shapes that are **cheaper and more
//! structurally regular** for `gatify`, without changing semantics.
//!
//! Key properties:
//! - **May introduce PIR extension ops** (e.g. `ext_carry_out`) when they are a
//!   more direct representation of the intended semantics for gateification.
//! - **May delete/rewrite internal nodes** (including DCE-like cleanup) but
//!   keeps the **function signature** unchanged.
//! - Preserves **node indices** (no compaction/reindexing); dead nodes are
//!   marked as `Nil` to maintain PIR layout invariants.
//!
//! Note: `xlsynth`/libxls analysis does not understand extension ops; callers
//! that need analysis should project to the XLS basis ops separately (see
//! `xlsynth_pir::desugar_extensions`).

use xlsynth::IrValue;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir_utils;
use xlsynth_pir::math::ceil_log2;

#[derive(Debug, Clone, Copy, Default)]
pub struct PrepForGatifyOptions {
    /// When true, rewrite carry-out idioms like
    /// `bit_slice(add(...), start=msb, width=1)` into `ext_carry_out`.
    ///
    /// Default is false because the rewrite is not always profitable.
    pub enable_rewrite_carry_out: bool,

    /// When true, rewrite the idiom `encode(one_hot(x))` (for power-of-two `x`
    /// width) into the extension op `ext_prio_encode(x, lsb_prio=...)` so
    /// gatification can use a specialized priority-encoder lowering.
    ///
    /// Default is false because this is a QoR strategy choice and we want to be
    /// able to compare old vs new circuits easily.
    pub enable_rewrite_prio_encode: bool,
}

/// Returns per-node use counts for the provided function.
fn get_use_counts(f: &ir::Fn) -> Vec<usize> {
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

/// Convert `or(or_reduce(a), or_reduce(b), ...)` into
/// `or_reduce(concat(a, b, ...))` when each of the `or_reduce` nodes has a
/// single use.
fn combine_or_reduces(f: &mut ir::Fn) {
    let use_counts = get_use_counts(f);
    for node_index in 0..f.nodes.len() {
        let payload = f.nodes[node_index].payload.clone();
        let NodePayload::Nary(NaryOp::Or, operands) = payload else {
            continue;
        };
        if operands.len() < 2 {
            continue;
        }

        let mut concat_inputs = Vec::with_capacity(operands.len());
        let mut reductions: Vec<ir::NodeRef> = Vec::with_capacity(operands.len());

        for operand in operands {
            let Some(ir::NodePayload::Unop(Unop::OrReduce, arg)) =
                f.nodes.get(operand.index).map(|n| &n.payload)
            else {
                reductions.clear();
                break;
            };
            if use_counts[operand.index] != 1 {
                reductions.clear();
                break;
            }
            reductions.push(operand);
            concat_inputs.push(*arg);
        }

        if reductions.len() < 2 {
            continue;
        }

        let concat_width: usize = concat_inputs
            .iter()
            .map(|nr| f.nodes[nr.index].ty.bit_count())
            .sum();
        if concat_width == 0 {
            continue;
        }

        let concat_ref = reductions[0];
        {
            let concat_node = &mut f.nodes[concat_ref.index];
            concat_node.payload = NodePayload::Nary(NaryOp::Concat, concat_inputs);
            concat_node.ty = ir::Type::Bits(concat_width);
        }

        {
            let node = &mut f.nodes[node_index];
            node.payload = NodePayload::Unop(Unop::OrReduce, concat_ref);
        }

        for reduction_ref in reductions.into_iter().skip(1) {
            let unused_node = &mut f.nodes[reduction_ref.index];
            unused_node.payload = NodePayload::Nil;
            unused_node.ty = ir::Type::nil();
        }
    }
}

/// Rewrites `encode(one_hot(x, lsb_prio=...))` into `ext_prio_encode(x,
/// lsb_prio=...)` when:
/// - `x` has power-of-two bit width
/// - the `one_hot` node has a single user (the `encode`)
///
/// This preserves the sentinel behavior of `encode(one_hot(...))` where `x==0`
/// yields the index `N` (for `x: bits[N]`).
fn rewrite_encode_one_hot_to_ext_prio_encode(f: &mut ir::Fn) -> usize {
    let use_counts = get_use_counts(f);
    let mut rewrites: usize = 0;

    // Snapshot length so we only visit original nodes.
    let original_len = f.nodes.len();
    for node_index in 0..original_len {
        let payload = f.nodes[node_index].payload.clone();
        let NodePayload::Encode { arg: one_hot } = payload else {
            continue;
        };
        if one_hot.index >= f.nodes.len() {
            continue;
        }
        if use_counts[one_hot.index] != 1 {
            continue;
        }

        let one_hot_payload = f.nodes[one_hot.index].payload.clone();
        let NodePayload::OneHot { arg, lsb_prio } = one_hot_payload else {
            continue;
        };
        let n = f.nodes[arg.index].ty.bit_count();
        if n == 0 || !n.is_power_of_two() {
            continue;
        }

        // Ensure the node result type matches the encode(one_hot) idiom shape:
        // one_hot makes width N+1, encode returns ceil_log2(N+1).
        let expected_out_w = ceil_log2(n.saturating_add(1));
        if f.nodes[node_index].ty.bit_count() != expected_out_w {
            continue;
        }

        let node_ref = ir::NodeRef { index: node_index };
        ir_utils::replace_node_payload(
            f,
            node_ref,
            NodePayload::ExtPrioEncode { arg, lsb_prio },
            Some(Type::Bits(expected_out_w)),
        )
        .expect("prep_for_gatify: ext_prio_encode payload replacement failed");

        rewrites += 1;
    }

    rewrites
}

fn nil_out_node(f: &mut ir::Fn, node_ref: ir::NodeRef) {
    let node = &mut f.nodes[node_ref.index];
    node.payload = NodePayload::Nil;
    node.ty = Type::nil();
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

fn get_or_insert_bits1_literal(f: &mut ir::Fn, value: bool) -> NodeRef {
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

fn is_ubits_literal_0_or_1_of_width(f: &ir::Fn, nr: NodeRef, w: usize) -> Option<bool> {
    let node = f.get_node(nr);
    if node.ty.bit_count() != w {
        return None;
    }
    let NodePayload::Literal(v) = &node.payload else {
        return None;
    };
    let u = v.to_u64().ok()?;
    match u {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

/// Returns `Some(low_bits)` iff `nr` is known to have a zero MSB and its low
/// `w` bits correspond to an underlying `bits[w]` value.
///
/// This is a *guard* used for carry-out rewrites: we must not “truncate” an
/// unconstrained `bits[w+1]` value, as that changes semantics.
fn msb_is_provably_zero(
    range_info: Option<&IrRangeInfo>,
    text_id: usize,
    msb_index: usize,
) -> bool {
    let ri = match range_info {
        Some(ri) => ri,
        None => return false,
    };
    let info = match ri.get(text_id) {
        Some(i) => i,
        None => return false,
    };

    if let Some(k) = info.known_bits.as_ref() {
        let known = k.mask.get_bit(msb_index).unwrap_or(false);
        if known {
            let bit_is_one = k.value.get_bit(msb_index).unwrap_or(false);
            return !bit_is_one;
        }
    }

    if let Some(max_bits) = info.unsigned_max.as_ref() {
        if max_bits.get_bit_count() > msb_index {
            return !max_bits.get_bit(msb_index).unwrap_or(false);
        }
    }

    false
}

fn get_zero_extended_low_bits(
    f: &mut ir::Fn,
    nr: NodeRef,
    w: usize,
    range_info: Option<&IrRangeInfo>,
) -> Option<NodeRef> {
    let node_w = f.get_node(nr).ty.bit_count();
    if node_w == w {
        return Some(nr);
    }
    if node_w != w.saturating_add(1) || w == 0 {
        return None;
    }
    let msb_index = w;
    let text_id = f.get_node(nr).text_id;

    // Finally, if analysis can prove the MSB is always 0, it is safe to take
    // the low `w` bits for carry-out computation.
    if msb_is_provably_zero(range_info, text_id, msb_index) {
        return Some(push_node(
            f,
            Type::Bits(w),
            NodePayload::BitSlice {
                arg: nr,
                start: 0,
                width: w,
            },
        ));
    }

    None
}

fn mark_dead_nodes_as_nil(f: &mut ir::Fn) {
    if f.nodes.is_empty() {
        return;
    }
    let Some(ret) = f.ret_node_ref else {
        return;
    };

    let mut live: Vec<bool> = vec![false; f.nodes.len()];
    let mut stack: Vec<ir::NodeRef> = vec![ret];
    while let Some(nr) = stack.pop() {
        if live[nr.index] {
            continue;
        }
        live[nr.index] = true;
        for dep in ir_utils::operands(&f.nodes[nr.index].payload) {
            if !live[dep.index] {
                stack.push(dep);
            }
        }
    }

    // Preserve reserved nil and param nodes (PIR layout invariants).
    let param_count = f.params.len();
    if !f.nodes.is_empty() {
        live[0] = true;
    }
    for i in 0..param_count {
        let idx = i + 1;
        if idx < live.len() {
            live[idx] = true;
        }
    }

    for i in 0..f.nodes.len() {
        if !live[i] {
            nil_out_node(f, ir::NodeRef { index: i });
        }
    }
}

/// Rewrites the canonical carry-out idiom:
/// `bit_slice(add(add(zero_ext(lhs), zero_ext(rhs)), zero_ext(c_in)), start=w,
/// width=1)` into `ext_carry_out(lhs, rhs, c_in)` when the add-chain nodes have
/// a single use (the slice) so we can safely DCE the chain.
fn rewrite_add_slice_carry_out_to_ext_carry_out(
    f: &mut ir::Fn,
    range_info: Option<&IrRangeInfo>,
) -> usize {
    let use_counts = get_use_counts(f);
    let mut rewrites: usize = 0;

    // Snapshot length so we only visit original nodes; rewrites may append nodes.
    let original_len = f.nodes.len();
    for node_index in 0..original_len {
        let payload = f.nodes[node_index].payload.clone();
        let NodePayload::BitSlice { arg, start, width } = payload else {
            continue;
        };
        if width != 1 {
            continue;
        }

        // The slice must be taking the MSB of the add result.
        let add_w = f.nodes[arg.index].ty.bit_count();
        if add_w == 0 {
            continue;
        }
        let w = add_w - 1;
        if start != w {
            continue;
        }

        // The add feeding the slice must be single-use, so we can drop it.
        if use_counts[arg.index] != 1 {
            continue;
        }

        // Supported idioms (c_in opportunistic):
        // (a) bit_slice(add(add(zero_ext(x), zero_ext(y)), zero_ext(c_in)), msb)
        // (b) bit_slice(add(add(x, y), literal(1)), msb)
        // (c) bit_slice(add(x, y), msb)   => ext_carry_out(x, y, literal(0))
        let arg_payload = f.nodes[arg.index].payload.clone();

        // First, try the existing canonical zero_ext chain (a).
        if let NodePayload::Binop(Binop::Add, sum_w1, c_in_ext) = arg_payload.clone() {
            if use_counts[sum_w1.index] == 1 && use_counts[c_in_ext.index] == 1 {
                let sum_payload = f.nodes[sum_w1.index].payload.clone();
                if let NodePayload::Binop(Binop::Add, lhs_ext, rhs_ext) = sum_payload {
                    if use_counts[lhs_ext.index] == 1 && use_counts[rhs_ext.index] == 1 {
                        let lhs_ext_payload = f.nodes[lhs_ext.index].payload.clone();
                        let rhs_ext_payload = f.nodes[rhs_ext.index].payload.clone();
                        let c_in_ext_payload = f.nodes[c_in_ext.index].payload.clone();
                        if let (
                            NodePayload::ZeroExt {
                                arg: lhs,
                                new_bit_count: lhs_w1,
                            },
                            NodePayload::ZeroExt {
                                arg: rhs,
                                new_bit_count: rhs_w1,
                            },
                            NodePayload::ZeroExt {
                                arg: c_in,
                                new_bit_count: c_in_w1,
                            },
                        ) = (lhs_ext_payload, rhs_ext_payload, c_in_ext_payload)
                        {
                            if lhs_w1 == add_w
                                && rhs_w1 == add_w
                                && c_in_w1 == add_w
                                && f.nodes[lhs.index].ty.bit_count() == w
                                && f.nodes[rhs.index].ty.bit_count() == w
                                && f.nodes[c_in.index].ty.bit_count() == 1
                            {
                                // Rewrite the `bit_slice` node into `ext_carry_out`.
                                //
                                // Note: this is an in-place rewrite (so it preserves ordering),
                                // and we separately DCE the now-dead add chain (guarded by
                                // single-use checks above).
                                let bit_slice_nr = ir::NodeRef { index: node_index };
                                ir_utils::replace_node_payload(
                                    f,
                                    bit_slice_nr,
                                    NodePayload::ExtCarryOut { lhs, rhs, c_in },
                                    Some(Type::Bits(1)),
                                )
                                .expect(
                                    "prep_for_gatify: ext_carry_out payload replacement failed",
                                );

                                // Nil out the now-dead chain (safe due to single-use checks).
                                nil_out_node(f, arg);
                                nil_out_node(f, sum_w1);
                                nil_out_node(f, lhs_ext);
                                nil_out_node(f, rhs_ext);
                                nil_out_node(f, c_in_ext);

                                rewrites += 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Otherwise handle (b) and (c).
        let mut carry_in: bool = false;
        let mut base_add: NodeRef = arg;
        let mut inner_add_to_nil: Option<NodeRef> = None;

        // Detect (b): add(add(x,y), literal(0/1))
        if let NodePayload::Binop(Binop::Add, a, b) = arg_payload {
            if let Some(ci) = is_ubits_literal_0_or_1_of_width(f, a, add_w) {
                carry_in = ci;
                base_add = b;
                inner_add_to_nil = Some(base_add);
            } else if let Some(ci) = is_ubits_literal_0_or_1_of_width(f, b, add_w) {
                carry_in = ci;
                base_add = a;
                inner_add_to_nil = Some(base_add);
            }
        }

        // Inner add must be single-use if present, so we can drop it.
        if inner_add_to_nil.is_some() && use_counts[base_add.index] != 1 {
            continue;
        }

        let base_payload = f.nodes[base_add.index].payload.clone();
        let NodePayload::Binop(Binop::Add, x, y) = base_payload else {
            continue;
        };
        let Some(lhs_low) = get_zero_extended_low_bits(f, x, w, range_info) else {
            continue;
        };
        let Some(rhs_low) = get_zero_extended_low_bits(f, y, w, range_info) else {
            continue;
        };
        let c_in_lit = get_or_insert_bits1_literal(f, carry_in);

        let bit_slice_nr = ir::NodeRef { index: node_index };

        // We may have appended helper nodes (e.g. low-bit slices) while
        // computing `lhs_low`/`rhs_low`. To preserve the SSA / textual ordering
        // invariant (operands must be defined before use) without reindexing,
        // we materialize the `ExtCarryOut` node *after* helpers and then use the
        // node replacement helpers to redirect the return to it.
        //
        // This rewrite is intended for cone-style patterns where the carry-out
        // slice is the (single-use) return value (possibly wrapped in an
        // `identity`).
        let Some(ret_nr) = f.ret_node_ref else {
            continue;
        };

        // Require that the slice is return-only (or return-only via identity),
        // since we will delete the old slice and add nodes.
        let (ret_target, bit_slice_to_nil): (ir::NodeRef, Option<ir::NodeRef>) =
            if ret_nr == bit_slice_nr {
                (bit_slice_nr, None)
            } else {
                match f.get_node(ret_nr).payload.clone() {
                    NodePayload::Unop(Unop::Identity, arg) if arg == bit_slice_nr => {
                        (ret_nr, Some(bit_slice_nr))
                    }
                    _ => continue,
                }
            };
        if use_counts[bit_slice_nr.index] != 1 {
            continue;
        }
        if use_counts[ret_target.index] != 1 {
            continue;
        }

        let ext_nr = push_node(
            f,
            Type::Bits(1),
            NodePayload::ExtCarryOut {
                lhs: lhs_low,
                rhs: rhs_low,
                c_in: c_in_lit,
            },
        );

        // Redirect the return (and any users) to the new node, then drop the old
        // return node(s).
        ir_utils::replace_node_with_ref(f, ret_target, ext_nr)
            .expect("prep_for_gatify: redirecting return to ext_carry_out failed");
        nil_out_node(f, ret_target);
        if let Some(bs) = bit_slice_to_nil {
            nil_out_node(f, bs);
        }

        nil_out_node(f, arg);
        if let Some(inner) = inner_add_to_nil {
            nil_out_node(f, inner);
        }

        rewrites += 1;
    }

    rewrites
}

/// Runs lightweight PIR rewrites that make gatification cleaner.
///
/// This pass:
/// - may rewrite into extension ops (e.g. `ExtCarryOut`) to expose intent for
///   cheaper gateification
/// - may mark dead nodes `Nil`
/// - does **not** reindex/compact nodes and does **not** change the function
///   signature
pub fn prep_for_gatify(
    f: &ir::Fn,
    range_info: Option<&IrRangeInfo>,
    options: PrepForGatifyOptions,
) -> ir::Fn {
    let mut cloned = f.clone();
    combine_or_reduces(&mut cloned);
    if options.enable_rewrite_carry_out {
        let _rewrites = rewrite_add_slice_carry_out_to_ext_carry_out(&mut cloned, range_info);
    }
    if options.enable_rewrite_prio_encode {
        let _rewrites = rewrite_encode_one_hot_to_ext_prio_encode(&mut cloned);
    }
    mark_dead_nodes_as_nil(&mut cloned);
    cloned
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_pir::ir_range_info::IrRangeInfo;

    #[test]
    fn or_reduces_with_single_use_are_combined() {
        let ir_text = "package sample
fn f(x: bits[2], y: bits[3]) -> bits[1] {
  x: bits[2] = param(name=x, id=1)
  y: bits[3] = param(name=y, id=2)
  x_any: bits[1] = or_reduce(x, id=3)
  y_any: bits[1] = or_reduce(y, id=4)
  ret combined: bits[1] = or(x_any, y_any, id=5)
}
";
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();

        let optimized = prep_for_gatify(f, None, PrepForGatifyOptions::default());

        let expected = r#"fn f(x: bits[2] id=1, y: bits[3] id=2) -> bits[1] {
  x_any: bits[5] = concat(x, y, id=3)
  ret combined: bits[1] = or_reduce(x_any, id=5)
}"#;
        assert_eq!(optimized.to_string(), expected);
    }

    #[test]
    fn carry_out_bit_slice_of_add_is_rewritten_to_ext_carry_out_with_cin0() {
        // Safe rewrite shape:
        //   add: bits[9] = add(zero_ext(x: bits[8]), zero_ext(y: bits[8]))
        //   ret: bits[1] = bit_slice(add, start=8, width=1)
        //
        // Here the MSBs of the add operands are *provably zero*, so the MSB of
        // the 9-bit sum is the carry-out of the underlying 8-bit add.
        let ir_text = "package bool_cone

top fn cone(x: bits[8] id=1, y: bits[8] id=2) -> bits[1] {
  zx: bits[9] = zero_ext(x, new_bit_count=9, id=3)
  zy: bits[9] = zero_ext(y, new_bit_count=9, id=4)
  add.5: bits[9] = add(zx, zy, id=5)
  ret bit_slice.6: bits[1] = bit_slice(add.5, start=8, width=1, id=6)
}
";
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_top_fn().unwrap();

        let mut xlsynth_pkg = xlsynth::IrPackage::parse_ir(ir_text, None).unwrap();
        xlsynth_pkg.set_top_by_name("cone").unwrap();
        let analysis = xlsynth_pkg.create_ir_analysis().unwrap();
        let range_info = IrRangeInfo::build_from_analysis(&analysis, f).unwrap();

        let optimized = prep_for_gatify(
            f,
            Some(range_info.as_ref()),
            PrepForGatifyOptions {
                enable_rewrite_carry_out: true,
                ..PrepForGatifyOptions::default()
            },
        );
        let optimized_text = optimized.to_string();
        assert!(
            optimized_text.contains("ext_carry_out(") && !optimized_text.contains("add("),
            "expected carry-out rewrite to eliminate add and introduce ext_carry_out; got:\n{}",
            optimized_text
        );

        // Equivalence for all inputs.
        for x in 0u64..=255u64 {
            for y in 0u64..=255u64 {
                let args = [
                    IrValue::make_ubits(8, x).unwrap(),
                    IrValue::make_ubits(8, y).unwrap(),
                ];
                let got_orig = match eval_fn(f, &args) {
                    FnEvalResult::Success(s) => s.value.to_bool().unwrap(),
                    FnEvalResult::Failure(f) => panic!("unexpected eval failure: {:?}", f),
                };
                let got_opt = match eval_fn(&optimized, &args) {
                    FnEvalResult::Success(s) => s.value.to_bool().unwrap(),
                    FnEvalResult::Failure(f) => panic!("unexpected eval failure: {:?}", f),
                };
                assert_eq!(got_orig, got_opt, "mismatch at x={} y={}", x, y);
            }
        }
    }

    #[test]
    fn carry_out_rewrite_can_use_range_info_to_prove_msb_zero() {
        // Here `umod(..., 256)` constrains the 9-bit values to <256, so their
        // MSB (bit 8) is provably 0. This makes it safe to rewrite the MSB
        // slice of the 9-bit sum into ExtCarryOut over the low 8 bits.
        let ir_text = "package sample

top fn cone(p0: bits[9] id=1, p1: bits[9] id=2) -> bits[1] {
  m: bits[9] = literal(value=256, id=3)
  x: bits[9] = umod(p0, m, id=4)
  y: bits[9] = umod(p1, m, id=5)
  add.6: bits[9] = add(x, y, id=6)
  ret bit_slice.7: bits[1] = bit_slice(add.6, start=8, width=1, id=7)
}
";
        let mut pir_parser = Parser::new(ir_text);
        let pir_pkg = pir_parser.parse_and_validate_package().unwrap();
        let pir_fn = pir_pkg.get_top_fn().unwrap();

        // Build range info from libxls analysis on the same text (no extensions).
        let mut xlsynth_pkg = xlsynth::IrPackage::parse_ir(ir_text, None).unwrap();
        xlsynth_pkg.set_top_by_name("cone").unwrap();
        let analysis = xlsynth_pkg.create_ir_analysis().unwrap();
        let range_info = IrRangeInfo::build_from_analysis(&analysis, pir_fn).unwrap();

        let optimized = prep_for_gatify(
            pir_fn,
            Some(range_info.as_ref()),
            PrepForGatifyOptions {
                enable_rewrite_carry_out: true,
                ..PrepForGatifyOptions::default()
            },
        );
        let optimized_text = optimized.to_string();
        assert!(
            optimized_text.contains("ext_carry_out(") && !optimized_text.contains("add("),
            "expected carry-out rewrite to introduce ext_carry_out; got:\n{}",
            optimized_text
        );
    }
}
