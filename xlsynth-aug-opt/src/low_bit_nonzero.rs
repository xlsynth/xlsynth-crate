// SPDX-License-Identifier: Apache-2.0

//! Test low bits for nonzero before an optional two's-complement negation.
//!
//! For `x: u32`, `or_reduce((p ? -x : x)[0:8])` equals
//! `or_reduce(x[0:8])`: negation modulo 2^8 preserves zero. A slice
//! starting above bit zero, or a set of slices with holes, does not have this
//! property. Whole-function costing accounts for retained numeric consumers.

use std::collections::BTreeSet;

use crate::ir_cost::IrCost;
use xlsynth_pir::ir::{self, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_match::MatchCtx;
use xlsynth_pir::ir_utils;

/// One scalar reduction of a contiguous low prefix of a common operand.
struct LowPrefix {
    arg: NodeRef,
    width: usize,
}

/// Finds a contiguous low prefix through scalar ORs and reduction/slice leaves.
fn low_prefix(f: &ir::Fn, root: NodeRef) -> Option<LowPrefix> {
    let ctx = MatchCtx::new(f);
    if !matches!(
        f.get_node(root).payload,
        NodePayload::Nary(NaryOp::Or, _) | NodePayload::Unop(Unop::OrReduce, _)
    ) {
        return None;
    }
    if ctx.bits_width(root) != Some(1) {
        return None;
    }
    // OR is idempotent, so each shared node need only be visited once.
    // Multiplicity-preserving associative flattening can expand a small shared
    // DAG exponentially; this traversal is bounded by reachable nodes/edges.
    let mut pending = vec![root];
    let mut visited = BTreeSet::new();
    let mut source = None;
    let mut intervals = Vec::new();
    while let Some(leaf) = pending.pop() {
        if !visited.insert(leaf.index) {
            continue;
        }
        if let NodePayload::Nary(NaryOp::Or, operands) = &f.get_node(leaf).payload {
            pending.extend(operands.iter().copied());
            continue;
        }
        let operand = match f.get_node(leaf).payload {
            NodePayload::Unop(Unop::OrReduce, arg) => arg,
            NodePayload::BitSlice { width: 1, .. } => leaf,
            _ => return None,
        };
        let (arg, start, width) = match f.get_node(operand).payload {
            NodePayload::BitSlice { arg, start, width } => (arg, start, width),
            _ => (operand, 0, ctx.bits_width(operand)?),
        };
        if source.is_some_and(|previous| previous != arg) {
            return None;
        }
        source = Some(arg);
        intervals.push((start, start.checked_add(width)?));
    }
    intervals.sort_unstable();
    let mut end = 0;
    for (start, limit) in intervals {
        if start > end {
            return None;
        }
        end = end.max(limit);
    }
    (end > 0).then_some(LowPrefix {
        arg: source?,
        width: end,
    })
}

/// Returns the input of negation or either ordering of select(x, -x).
fn before_negation(f: &ir::Fn, node: NodeRef) -> Option<NodeRef> {
    match &f.get_node(node).payload {
        NodePayload::Unop(Unop::Neg, arg) => Some(*arg),
        NodePayload::Sel {
            selector,
            cases,
            default: None,
        } if cases.len() == 2 && MatchCtx::new(f).bits_width(*selector) == Some(1) => {
            for (plain, negated) in [(cases[0], cases[1]), (cases[1], cases[0])] {
                if matches!(
                    f.get_node(negated).payload,
                    NodePayload::Unop(Unop::Neg, arg) if arg == plain
                ) {
                    return Some(plain);
                }
            }
            None
        }
        _ => None,
    }
}

/// One scalar predicate and its negation-independent low prefix.
struct Rewrite {
    root: NodeRef,
    arg: NodeRef,
    width: usize,
}

/// Builds a basis-IR candidate while leaving the original function untouched.
fn candidate(f: &ir::Fn) -> Option<(ir::Fn, usize)> {
    let dead = xlsynth_pir::dce::get_dead_nodes(f);
    let mut live = vec![true; f.nodes.len()];
    for node in dead {
        live[node.index] = false;
    }
    let mut matches = Vec::new();
    for (index, live) in live.into_iter().enumerate() {
        let root = NodeRef { index };
        if !live {
            continue;
        }
        let Some(prefix) = low_prefix(f, root) else {
            continue;
        };
        if let Some(arg) = before_negation(f, prefix.arg) {
            matches.push(Rewrite {
                root,
                arg,
                width: prefix.width,
            });
        }
    }
    if matches.is_empty() {
        return None;
    }
    let mut result = f.clone();
    let mut next_id = f.nodes.iter().map(|node| node.text_id).max().unwrap_or(0);
    for &Rewrite { root, arg, width } in &matches {
        let reduced = if MatchCtx::new(f).bits_width(arg) == Some(width) {
            arg
        } else {
            next_id = next_id.checked_add(1)?;
            let slice = NodeRef {
                index: result.nodes.len(),
            };
            result.nodes.push(ir::Node {
                text_id: next_id,
                name: None,
                ty: Type::Bits(width),
                payload: NodePayload::BitSlice {
                    arg,
                    start: 0,
                    width,
                },
                pos: None,
            });
            slice
        };
        result.nodes[root.index].payload = NodePayload::Unop(Unop::OrReduce, reduced);
    }
    ir_utils::compact_and_toposort_in_place(&mut result).ok()?;
    Some((result, matches.len()))
}

/// Installs the rewrite only when whole-function area/delay strictly improve.
///
/// Ties, invalid costs, evaluation errors, and required ID allocation overflow
/// preserve the input. Numeric sharing is permitted and included in the cost.
pub fn rewrite_with_evaluator(
    f: &mut ir::Fn,
    evaluator: &mut impl FnMut(&ir::Fn) -> Result<IrCost, String>,
) -> usize {
    let Some((rewritten, count)) = candidate(f) else {
        return 0;
    };
    let (Ok(old), Ok(new)) = (evaluator(f), evaluator(&rewritten)) else {
        return 0;
    };
    if !new.is_pareto_improvement_on(old) {
        return 0;
    }
    *f = rewritten;
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    fn sample(shape: &str) -> ir::Fn {
        let tail = match shape {
            "empty" => {
                "  lo: bits[0] = bit_slice(v, start=0, width=0, id=5)\n  ret out: bits[1] = or_reduce(lo, id=6)"
            }
            "offset" => {
                "  lo: bits[3] = bit_slice(v, start=1, width=3, id=5)\n  ret out: bits[1] = or_reduce(lo, id=6)"
            }
            "gap" => {
                "  a: bits[1] = bit_slice(v, start=0, width=1, id=5)\n  b: bits[1] = bit_slice(v, start=2, width=1, id=6)\n  ret out: bits[1] = or(a, b, id=7)"
            }
            "decomposed" => {
                "  a: bits[1] = bit_slice(v, start=0, width=1, id=5)\n  b: bits[2] = bit_slice(v, start=1, width=2, id=6)\n  br: bits[1] = or_reduce(b, id=7)\n  nested: bits[1] = or(br, a, id=8)\n  ret out: bits[1] = or(a, nested, id=9)"
            }
            _ => {
                "  lo: bits[3] = bit_slice(v, start=0, width=3, id=5)\n  ret out: bits[1] = or_reduce(lo, id=6)"
            }
        };
        let text = format!(
            "package test\n\ntop fn main(x: bits[8] id=1, p: bits[1] id=2) -> bits[1] {{\n  n: bits[8] = neg(x, id=3)\n  v: bits[8] = sel(p, cases=[x, n], id=4)\n{tail}\n}}"
        );
        Parser::new(&text)
            .parse_and_validate_package()
            .unwrap()
            .get_top_fn()
            .unwrap()
            .clone()
    }

    #[test]
    fn rejects_offset_and_gapped_slices_but_accepts_overlapping_contiguous_slices() {
        for shape in ["empty", "offset", "gap"] {
            assert!(candidate(&sample(shape)).is_none(), "{shape}");
        }
        for shape in ["single", "decomposed"] {
            let (rewritten, count) = candidate(&sample(shape)).unwrap();
            assert!(count > 0);
            xlsynth_pir::ir_verify::verify_function(&rewritten).unwrap();
            let rewritten = xlsynth_pir::dce::remove_dead_nodes(&rewritten);
            assert!(
                !rewritten
                    .nodes
                    .iter()
                    .any(|node| matches!(node.payload, NodePayload::Unop(Unop::Neg, _)))
            );
        }
    }

    #[test]
    fn ties_errors_invalid_costs_and_tradeoffs_preserve_the_input() {
        let original = sample("single");
        for cost in [
            Ok(IrCost {
                area: 100,
                delay: 100.0,
            }),
            Ok(IrCost {
                area: 99,
                delay: 101.0,
            }),
            Ok(IrCost {
                area: 101,
                delay: 99.0,
            }),
            Ok(IrCost {
                area: 99,
                delay: f64::NAN,
            }),
            Err("unavailable cost".to_string()),
        ] {
            let mut rewritten = original.clone();
            let mut first = true;
            assert_eq!(
                rewrite_with_evaluator(&mut rewritten, &mut |_| {
                    if std::mem::take(&mut first) {
                        Ok(IrCost {
                            area: 100,
                            delay: 100.0,
                        })
                    } else {
                        cost.clone()
                    }
                }),
                0
            );
            assert_eq!(rewritten.to_string(), original.to_string());
        }
    }

    #[test]
    fn exhausted_ids_do_not_leak_partial_rewrites() {
        let mut f = sample("single");
        f.nodes.last_mut().unwrap().text_id = usize::MAX;
        let before = f.to_string();
        assert_eq!(
            rewrite_with_evaluator(&mut f, &mut |_| panic!(
                "must not cost an incomplete candidate"
            )),
            0
        );
        assert_eq!(f.to_string(), before);
    }

    #[test]
    fn shared_or_dag_does_not_expand_operand_multiplicity() {
        let mut f = sample("single");
        let mut root = f.ret_node_ref.unwrap();
        for text_id in 7..71 {
            let next = NodeRef {
                index: f.nodes.len(),
            };
            f.nodes.push(ir::Node {
                text_id,
                name: None,
                ty: Type::Bits(1),
                payload: NodePayload::Nary(NaryOp::Or, vec![root, root]),
                pos: None,
            });
            root = next;
        }
        f.ret_node_ref = Some(root);
        let (rewritten, count) = candidate(&f).unwrap();
        assert_eq!(count, 65);
        xlsynth_pir::ir_verify::verify_function(&rewritten).unwrap();
        assert!(rewritten.nodes.len() <= f.nodes.len() + count);
    }

    #[test]
    fn offset_and_gapped_masks_have_concrete_counterexamples() {
        // Eight-bit -1 has bit 1 set, while +1 does not.
        let offset_x = 1u8;
        assert_eq!((offset_x >> 1) & 7, 0);
        assert_ne!((offset_x.wrapping_neg() >> 1) & 7, 0);
        // Observing bits 0 and 2 misses +2 but sees bit 2 of -2.
        let gapped_x = 2u8;
        assert_eq!(gapped_x & 5, 0);
        assert_ne!(gapped_x.wrapping_neg() & 5, 0);
    }

    #[test]
    fn disabled_and_zero_round_pir_pipelines_skip_the_rewrite() {
        let source = format!("package test\n\ntop {}\n", sample("single"));
        for (enable, rounds) in [(false, 1), (true, 0)] {
            let result = crate::run_aug_opt_over_ir_text_with_stats(
                &source,
                Some("main"),
                crate::AugOptOptions {
                    enable,
                    rounds,
                    mode: crate::AugOptMode::PirOnly,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(result.rewrite_stats.low_bit_nonzero_simplified, 0);
            assert_eq!(result.output_text, source);
        }
    }

    #[test]
    fn dead_predicates_do_not_count_as_rewrites() {
        let mut f = sample("single");
        f.ret_node_ref = Some(f.params[1]);
        assert!(candidate(&f).is_none());
    }
}
