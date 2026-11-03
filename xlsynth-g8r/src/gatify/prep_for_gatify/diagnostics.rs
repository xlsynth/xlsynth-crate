// SPDX-License-Identifier: Apache-2.0

//! Opt-in observations of the actual preparation pass sequence.

use std::collections::BTreeMap;
use std::time::Instant;
use xlsynth_pir::ir::{self, NodePayload};
use xlsynth_pir::ir_utils;

const TARGET: &str = "xlsynth_g8r::gatify::prep_for_gatify";

/// Runs a pass without collecting diagnostics unless its log target is enabled.
pub(super) fn run(
    f: &mut ir::Fn,
    pass: &str,
    round: usize,
    apply: impl FnOnce(&mut ir::Fn) -> usize,
) -> usize {
    if !log::log_enabled!(target: TARGET, log::Level::Debug) {
        return apply(f);
    }
    run_observed(f, pass, round, apply)
}

/// Records pass-local work; enclosing fixed-point timings include child
/// logging.
fn run_observed(
    f: &mut ir::Fn,
    pass: &str,
    round: usize,
    apply: impl FnOnce(&mut ir::Fn) -> usize,
) -> usize {
    let before = Quality::capture(f);
    let started = Instant::now();
    let rewrites = apply(f);
    let elapsed = started.elapsed();
    let after = Quality::capture(f);
    log::debug!(target: TARGET,
        "prep_pass function={} pass={} round={} rewrites={} elapsed_us={} before={:?} after={:?}",
        f.name, pass, round, rewrites, elapsed.as_micros(), before, after);
    log::trace!(target: TARGET,
        "prep_ir function={} pass={} round={}\n{}", f.name, pass, round, f);
    rewrites
}

/// Records an input checkpoint without treating it as a transformation.
pub(super) fn snapshot(f: &ir::Fn, pass: &str, round: usize) {
    if log::log_enabled!(target: TARGET, log::Level::Debug) {
        log::debug!(target: TARGET,
            "prep_snapshot function={} pass={} round={} quality={:?}",
            f.name, pass, round, Quality::capture(f));
    }
    log::trace!(target: TARGET,
        "prep_ir function={} pass={} round={}\n{}", f.name, pass, round, f);
}

/// Distinguishes stored IR from output-reachable work without changing
/// analyses.
#[derive(Debug, PartialEq, Eq)]
struct Quality {
    stored_slots: usize,
    stored_ops: BTreeMap<String, usize>,
    live_nodes: usize,
    live_bits: usize,
    live_operand_edges: usize,
    max_live_operand_uses: usize,
    live_literal_bits: usize,
    live_ops: BTreeMap<String, usize>,
}

impl Quality {
    /// Measures only output reachability; unused signature parameters stay
    /// stored.
    fn capture(f: &ir::Fn) -> Self {
        let mut live = vec![false; f.nodes.len()];
        let mut uses = vec![0usize; f.nodes.len()];
        let mut pending = f.ret_node_ref.into_iter().collect::<Vec<_>>();
        let mut result = Self {
            stored_slots: ir_utils::fn_node_count(f),
            stored_ops: ir_utils::op_histogram(f),
            live_nodes: 0,
            live_bits: 0,
            live_operand_edges: 0,
            max_live_operand_uses: 0,
            live_literal_bits: 0,
            live_ops: BTreeMap::new(),
        };
        while let Some(node_ref) = pending.pop() {
            if live[node_ref.index] {
                continue;
            }
            live[node_ref.index] = true;
            let node = f.get_node(node_ref);
            if node.payload == NodePayload::Nil {
                continue;
            }
            result.live_nodes += 1;
            result.live_bits = result.live_bits.saturating_add(node.ty.bit_count());
            if matches!(node.payload, NodePayload::Literal(_)) {
                result.live_literal_bits =
                    result.live_literal_bits.saturating_add(node.ty.bit_count());
            }
            if node.payload != NodePayload::Param {
                *result
                    .live_ops
                    .entry(node.payload.get_operator().to_string())
                    .or_default() += 1;
            }
            for operand in ir_utils::operands(&node.payload) {
                uses[operand.index] += 1;
                result.live_operand_edges += 1;
                result.max_live_operand_uses =
                    result.max_live_operand_uses.max(uses[operand.index]);
                pending.push(operand);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::{Quality, run_observed};
    use crate::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
    use crate::gatify::prep_for_gatify::rewrite_nary_adds_to_fixed_point;
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_pir::ir_utils;

    #[test]
    fn quality_distinguishes_stored_and_live_nodes() {
        let f = Parser::new(
            r#"fn main(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  unused: bits[8] = literal(value=7, id=3)
  ret result: bits[8] = add(a, b, id=4)
}"#,
        )
        .parse_fn()
        .unwrap();
        let before = f.to_string();
        let quality = Quality::capture(&f);
        assert_eq!(quality.stored_slots, 4);
        assert_eq!(quality.live_nodes, 3);
        assert_eq!(quality.live_bits, 24);
        assert_eq!(quality.live_operand_edges, 2);
        assert_eq!(quality.max_live_operand_uses, 1);
        assert_eq!(quality.live_literal_bits, 0);
        assert_eq!(quality.stored_ops["literal"], 1);
        assert_eq!(quality.live_ops.len(), 1);
        assert_eq!(quality.live_ops["add"], 1);
        assert_eq!(f.to_string(), before);
    }

    #[test]
    fn observing_pass_preserves_final_ir_and_aig() {
        let mut plain = Parser::new(
            r#"fn main(a: bits[8] id=1, b: bits[8] id=2, c: bits[8] id=3) -> bits[8] {
  partial: bits[8] = add(a, b, id=4)
  ret result: bits[8] = add(partial, c, id=5)
}"#,
        )
        .parse_fn()
        .unwrap();
        let mut observed = plain.clone();
        let plain_count = rewrite_nary_adds_to_fixed_point(&mut plain);
        let observed_count = run_observed(
            &mut observed,
            "nary_add.fixed_point",
            0,
            rewrite_nary_adds_to_fixed_point,
        );
        assert!(plain_count > 0);
        assert_eq!(observed_count, plain_count);
        assert_eq!(observed.to_string(), plain.to_string());
        ir_utils::compact_and_toposort_in_place(&mut plain).unwrap();
        ir_utils::compact_and_toposort_in_place(&mut observed).unwrap();
        let options = GatifyOptions {
            fold: true,
            hash: true,
            ..GatifyOptions::all_opts_disabled()
        };
        let plain_gates = gatify_prepared_fn(&plain, options.clone()).unwrap().gate_fn;
        let observed_gates = gatify_prepared_fn(&observed, options).unwrap().gate_fn;
        assert_eq!(
            observed_gates.to_string_including_dead_nodes(),
            plain_gates.to_string_including_dead_nodes()
        );
    }
}
