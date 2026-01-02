// SPDX-License-Identifier: Apache-2.0

//! Preparatory IR optimizations that make gatification cleaner.
//! Currently this focuses on simplifying reductions that XLS has
//! expanded for other optimizer passes.

use xlsynth_pir::ir::{self, NaryOp, NodePayload, Unop};
use xlsynth_pir::ir_utils;

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

/// Run lightweight IR optimizations that make gatification cleaner without
/// changing node indices or the function signature.
pub fn prep_for_gatify(f: &ir::Fn) -> ir::Fn {
    let mut cloned = f.clone();
    combine_or_reduces(&mut cloned);
    cloned
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

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

        let optimized = prep_for_gatify(f);

        let expected = r#"fn f(x: bits[2] id=1, y: bits[3] id=2) -> bits[1] {
  x_any: bits[5] = concat(x, y, id=3)
  ret combined: bits[1] = or_reduce(x_any, id=5)
}"#;
        assert_eq!(optimized.to_string(), expected);
    }
}
