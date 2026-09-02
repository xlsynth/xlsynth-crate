// SPDX-License-Identifier: Apache-2.0

//! Input placement for the linear-XOR/majority full-adder lowering.

use super::{FullAdderOutput, GateBuilder};
use crate::aig::gate::{AigNode, AigOperand};
use crate::aig::get_summary_stats::get_aig_cone_stats;

struct TrialStats {
    depths: [usize; 2],
    suffix_ands: usize,
    prefix_roots: Vec<AigOperand>,
}

impl TrialStats {
    /// Measures the small appended cone, stopping at the common builder prefix.
    fn new(gb: &GateBuilder, output: FullAdderOutput, prefix: usize) -> Self {
        let roots = [output.sum, output.carry];
        let depths = roots.map(|root| gb.aig_depth(root).expect("hashed builder has depths"));
        let mut visited = vec![false; gb.gates.len() - prefix];
        let mut pending = roots.to_vec();
        let mut prefix_roots = Vec::new();
        let mut suffix_ands = 0;
        while let Some(root) = pending.pop() {
            if root.node.id < prefix {
                prefix_roots.push(AigOperand {
                    node: root.node,
                    negated: false,
                });
                continue;
            }
            let seen = &mut visited[root.node.id - prefix];
            if *seen {
                continue;
            }
            *seen = true;
            if let AigNode::And2 { a, b, .. } = gb.gates[root.node.id] {
                suffix_ands += 1;
                pending.extend([a, b]);
            }
        }
        prefix_roots.sort_by_key(|root| root.node.id);
        prefix_roots.dedup();
        Self {
            depths,
            suffix_ands,
            prefix_roots,
        }
    }

    /// Cancels a shared boundary, or counts differing prefix cones exactly.
    fn and_delta(
        &self,
        baseline: &Self,
        gb: &GateBuilder,
        baseline_prefix_ands: &mut Option<usize>,
    ) -> i64 {
        let prefix_delta = if self.prefix_roots == baseline.prefix_roots {
            0
        } else {
            let baseline_ands = *baseline_prefix_ands.get_or_insert_with(|| {
                get_aig_cone_stats(&gb.gates, &baseline.prefix_roots).and_nodes
            });
            get_aig_cone_stats(&gb.gates, &self.prefix_roots).and_nodes as i64
                - baseline_ands as i64
        };
        prefix_delta + self.suffix_ands as i64 - baseline.suffix_ands as i64
    }
}

impl GateBuilder {
    /// Tries each final-XOR input from the same prefix and retains a local
    /// Pareto improvement. This does not guarantee post-cleanup or mapped QoR.
    pub(super) fn add_full_adder_with_input_placement(
        &mut self,
        a: AigOperand,
        b: AigOperand,
        c: AigOperand,
    ) -> FullAdderOutput {
        if self.hash_cons.is_none() || self.active_append_checkpoint.is_some() {
            // No cached arrivals, or a caller owns the non-nestable checkpoint.
            return self.add_full_adder_in_order(a, b, c);
        }
        let prefix = self.gates.len();
        let checkpoint = self.begin_append_checkpoint();
        let baseline = self.add_full_adder_in_order(a, b, c);
        let baseline_stats = TrialStats::new(self, baseline, prefix);
        self.rollback_append_checkpoint(checkpoint);

        let mut selected = [a, b, c];
        let rank = |depths: [usize; 2], and_delta| {
            (depths[0].max(depths[1]), depths[0] + depths[1], and_delta)
        };
        let mut best_rank = rank(baseline_stats.depths, 0);
        let mut baseline_prefix_ands = None;
        for order in [[b, c, a], [a, c, b]] {
            if order == [a, b, c] {
                continue;
            }
            let checkpoint = self.begin_append_checkpoint();
            let candidate = self.add_full_adder_in_order(order[0], order[1], order[2]);
            let stats = TrialStats::new(self, candidate, prefix);
            if stats
                .depths
                .iter()
                .zip(&baseline_stats.depths)
                .all(|(c, b)| c <= b)
            {
                let and_delta = stats.and_delta(&baseline_stats, self, &mut baseline_prefix_ands);
                let candidate_rank = rank(stats.depths, and_delta);
                if and_delta <= 0 && candidate_rank < best_rank {
                    selected = order;
                    best_rank = candidate_rank;
                }
            }
            self.rollback_append_checkpoint(checkpoint);
        }
        self.add_full_adder_in_order(selected[0], selected[1], selected[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::gate::{AigBitVector, GateFn};
    use crate::gate_builder::GateBuilderOptions;
    use crate::mcmc_logic::oracle_equiv_sat;

    /// Gives each adder input its own controllable arrival and polarity.
    fn inputs(gb: &mut GateBuilder, delays: [usize; 3], negations: usize) -> [AigOperand; 3] {
        std::array::from_fn(|i| {
            let bits = gb.add_input(format!("in{i}"), delays[i] + 1);
            let mut result = *bits.get_lsb(0);
            for j in 1..=delays[i] {
                result = gb.add_and_binary(result, *bits.get_lsb(j));
            }
            if negations & (1 << i) != 0 {
                result.negate()
            } else {
                result
            }
        })
    }

    fn finish(mut gb: GateBuilder, output: FullAdderOutput) -> GateFn {
        gb.add_output(
            "out".into(),
            AigBitVector::from_lsb_is_index_0(&[output.sum, output.carry]),
        );
        gb.build()
    }

    #[test]
    fn full_adder_input_placement_improves_late_first_input() {
        let mut gb = GateBuilder::new("fa".into(), GateBuilderOptions::opt());
        let [a, b, c] = inputs(&mut gb, [5, 1, 0], 0);
        let mut linear = gb.clone();
        let reference = linear.add_full_adder_in_order(a, b, c);
        let baseline = get_aig_cone_stats(&linear.gates, &[reference.sum, reference.carry]);
        let selected = gb.add_full_adder(a, b, c);
        let result = get_aig_cone_stats(&gb.gates, &[selected.sum, selected.carry]);
        assert!(result.root_depths[0] < baseline.root_depths[0]);
        assert!(result.root_depths[1] <= baseline.root_depths[1]);
        assert!(result.and_nodes <= baseline.and_nodes);
        assert!(oracle_equiv_sat(&finish(linear, reference), &finish(gb, selected)).unwrap());
    }

    #[test]
    fn full_adder_input_placement_preserves_equivalence_and_local_pareto_guard() {
        let mut changed = 0;
        let mut differing_boundaries = 0;
        for fold in [false, true] {
            for delays in [
                [0, 2, 5],
                [0, 5, 2],
                [2, 0, 5],
                [2, 5, 0],
                [5, 0, 2],
                [5, 2, 0],
            ] {
                for negations in 0..8 {
                    for shared_order in 0..4 {
                        let mut gb =
                            GateBuilder::new("fa".into(), GateBuilderOptions { fold, hash: true });
                        let [a, b, c] = inputs(&mut gb, delays, negations);
                        let orders = [[a, b, c], [b, c, a], [a, c, b]];
                        if shared_order < 3 {
                            let order = orders[shared_order];
                            gb.add_full_adder_in_order(order[0], order[1], order[2]);
                        }
                        let prefix = gb.gates.len();
                        let mut linear = gb.clone();
                        let reference = linear.add_full_adder_in_order(a, b, c);
                        let baseline =
                            get_aig_cone_stats(&linear.gates, &[reference.sum, reference.carry]);
                        let base_boundary = TrialStats::new(&linear, reference, prefix);
                        for order in orders {
                            let mut trial = gb.clone();
                            let output =
                                trial.add_full_adder_in_order(order[0], order[1], order[2]);
                            let stats = TrialStats::new(&trial, output, prefix);
                            differing_boundaries +=
                                usize::from(stats.prefix_roots != base_boundary.prefix_roots);
                            let exact =
                                get_aig_cone_stats(&trial.gates, &[output.sum, output.carry]);
                            assert_eq!(
                                stats.and_delta(&base_boundary, &trial, &mut None),
                                exact.and_nodes as i64 - baseline.and_nodes as i64
                            );
                        }
                        let selected = gb.add_full_adder(a, b, c);
                        let actual = get_aig_cone_stats(&gb.gates, &[selected.sum, selected.carry]);
                        assert!(actual.and_nodes <= baseline.and_nodes);
                        assert!(
                            actual
                                .root_depths
                                .iter()
                                .zip(&baseline.root_depths)
                                .all(|(a, b)| a <= b)
                        );
                        changed += usize::from(actual != baseline);
                        assert!(
                            oracle_equiv_sat(&finish(linear, reference), &finish(gb, selected))
                                .unwrap()
                        );
                    }
                }
            }
        }
        assert!(changed > 0);
        assert!(differing_boundaries > 0);
    }

    #[test]
    fn full_adder_input_placement_handles_constants_aliases_and_complements() {
        let mut prefix = GateBuilder::new("fa".into(), GateBuilderOptions::opt());
        let [a, b, c] = inputs(&mut prefix, [0, 1, 3], 0);
        let choices = [
            prefix.get_false(),
            prefix.get_true(),
            a,
            a.negate(),
            b,
            b.negate(),
            c,
        ];
        for x in choices {
            for y in choices {
                for z in choices {
                    let mut gb = prefix.clone();
                    let mut linear = prefix.clone();
                    let reference = linear.add_full_adder_in_order(x, y, z);
                    let selected = gb.add_full_adder(x, y, z);
                    let expected =
                        get_aig_cone_stats(&linear.gates, &[reference.sum, reference.carry]);
                    let actual = get_aig_cone_stats(&gb.gates, &[selected.sum, selected.carry]);
                    assert!(actual.and_nodes <= expected.and_nodes);
                    assert!(
                        actual
                            .root_depths
                            .iter()
                            .zip(&expected.root_depths)
                            .all(|(a, b)| a <= b)
                    );
                    assert!(
                        oracle_equiv_sat(&finish(linear, reference), &finish(gb, selected))
                            .unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn full_adder_input_placement_preserves_fallback_and_prefix_metadata() {
        for hash in [false, true] {
            for nested in [false, true] {
                let mut gb = GateBuilder::new("fa".into(), GateBuilderOptions { fold: true, hash });
                let [a, b, c] = inputs(&mut gb, [0, 0, 0], 0);
                gb.set_current_pir_node_id(Some(7));
                gb.add_full_adder_in_order(a, b, c);
                gb.set_current_pir_node_id(Some(19));
                let mut linear = gb.clone();
                let reference = linear.add_full_adder_in_order(a, b, c);
                let checkpoint = nested.then(|| gb.begin_append_checkpoint());
                let selected = gb.add_full_adder(a, b, c);
                if let Some(checkpoint) = checkpoint {
                    gb.commit_append_checkpoint(checkpoint);
                }
                assert_eq!(selected, reference);
                assert_eq!(gb.gates.len(), linear.gates.len());
                for (actual, expected) in gb.gates.iter().zip(&linear.gates) {
                    assert_eq!(actual.get_pir_node_ids(), expected.get_pir_node_ids());
                }
                assert_eq!(
                    finish(gb, selected).to_string(),
                    finish(linear, reference).to_string()
                );
            }
        }
    }
}
