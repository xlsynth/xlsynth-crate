// SPDX-License-Identifier: Apache-2.0

//! Reversible alternatives measured over the union of a bounded gate region.

use std::collections::HashSet;

use super::GateBuilder;
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, AigRef};

impl GateBuilder {
    /// Tries an append-only alternative to already-emitted output roots.
    ///
    /// The AND count stops below `prefix_end` and at `boundaries`. AIG depth
    /// includes the complete fanin. The closure may append only AND gates and
    /// mutate provenance through builder APIs. No inputs or outputs may be
    /// added. Hash-disabled regions and regions with zero ANDs or depth skip
    /// the trial. An alternative must use no extra ANDs and deepen no output.
    /// Either the AND count or the maximum output depth must strictly decrease.
    ///
    /// Sharing outside the cut and later optimization can change these counts.
    pub(crate) fn try_region_alternative(
        &mut self,
        ordinary: AigBitVector,
        prefix_end: usize,
        boundaries: &[AigOperand],
        emit: impl FnOnce(&mut GateBuilder) -> AigBitVector,
    ) -> (AigBitVector, Option<GateRegionTrial>) {
        assert!(prefix_end <= self.gates.len());
        let Some(before_depth) = ordinary.iter_lsb_to_msb().try_fold(0usize, |depth, bit| {
            self.aig_depth(*bit).map(|bit_depth| depth.max(bit_depth))
        }) else {
            return (ordinary, None);
        };
        let boundaries: HashSet<_> = boundaries.iter().map(|bit| bit.node).collect();
        let before = GateRegionCost {
            and_nodes: suffix_and_count(self, &ordinary, prefix_end, &boundaries),
            depth: before_depth,
        };
        if before.and_nodes == 0 || before.depth == 0 {
            return (ordinary, None);
        }
        let stored_before = self.gates.len();
        let checkpoint = self.begin_append_checkpoint();
        let alternative = emit(self);
        assert_eq!(alternative.get_bit_count(), ordinary.get_bit_count());
        let after = GateRegionCost {
            and_nodes: suffix_and_count(self, &alternative, prefix_end, &boundaries),
            depth: alternative.iter_lsb_to_msb().fold(0, |depth, bit| {
                depth.max(self.aig_depth(*bit).expect("AIG depth remains available"))
            }),
        };
        let delayed_outputs = alternative
            .iter_lsb_to_msb()
            .zip(ordinary.iter_lsb_to_msb())
            .filter(|(new, old)| {
                self.aig_depth(**new).expect("AIG depth remains available")
                    > self
                        .aig_depth(**old)
                        .expect("ordinary AIG depth remains available")
            })
            .count();
        let accepted = after.and_nodes <= before.and_nodes
            && delayed_outputs == 0
            && (after.and_nodes < before.and_nodes || after.depth < before.depth);
        let trial = GateRegionTrial {
            before,
            after,
            appended: self.gates.len() - stored_before,
            delayed_outputs,
            accepted,
        };
        log::debug!(
            "gate_region_trial outputs={} frontier={} before_cut_ands={} before_depth={} after_cut_ands={} after_depth={} appended={} delayed_outputs={} accepted={}",
            ordinary.get_bit_count(),
            prefix_end,
            trial.before.and_nodes,
            trial.before.depth,
            trial.after.and_nodes,
            trial.after.depth,
            trial.appended,
            trial.delayed_outputs,
            trial.accepted
        );
        if trial.accepted {
            self.commit_append_checkpoint(checkpoint);
            (alternative, Some(trial))
        } else {
            self.rollback_append_checkpoint(checkpoint);
            (ordinary, Some(trial))
        }
    }
}

/// Reachable suffix AND count and maximum full-fanin AIG depth over all roots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GateRegionCost {
    pub and_nodes: usize,
    pub depth: usize,
}

/// Measured comparison for an emitted alternative, including rejected work.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GateRegionTrial {
    pub before: GateRegionCost,
    pub after: GateRegionCost,
    pub appended: usize,
    /// Number of output bits whose AIG depth increased.
    pub delayed_outputs: usize,
    pub accepted: bool,
}

/// Counts each reachable AND once across all roots without visiting the prefix.
fn suffix_and_count(
    gb: &GateBuilder,
    roots: &AigBitVector,
    prefix_end: usize,
    boundaries: &HashSet<AigRef>,
) -> usize {
    let mut pending: Vec<_> = roots.iter_lsb_to_msb().map(|bit| bit.node).collect();
    let mut seen = HashSet::new();
    let mut ands = 0;
    while let Some(node) = pending.pop() {
        if node.id < prefix_end || boundaries.contains(&node) || !seen.insert(node) {
            continue;
        }
        if let AigNode::And2 { a, b, .. } = gb.gates[node.id] {
            ands += 1;
            pending.extend([a.node, b.node]);
        }
    }
    ands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::GateBuilderOptions;

    #[test]
    fn shared_subtrees_are_counted_once_and_rejected_trials_restore_hashing() {
        let mut gb = GateBuilder::new("shared".to_string(), GateBuilderOptions::opt());
        let inputs = gb.add_input("inputs".to_string(), 4);
        let [a, b, c, d] = std::array::from_fn(|i| *inputs.get_lsb(i));
        let frontier = gb.gates.len();
        let common = gb.add_or_binary(a, b);
        let first = gb.add_or_binary(common, c);
        let second = gb.add_or_binary(common, d);
        let ordinary = AigBitVector::from_lsb_is_index_0(&[first, second]);
        gb.add_tag(common.node, "shared".to_string());
        gb.set_current_pir_node_id(Some(99));
        let before = gb.gates.clone();
        let (result, trial) = gb.try_region_alternative(ordinary.clone(), frontier, &[], |gb| {
            let reused = gb.add_or_binary(a, b);
            assert_eq!(reused, common);
            gb.add_tag(reused.node, "trial".to_string());
            let ac = gb.add_or_binary(a, c);
            let ad = gb.add_or_binary(a, d);
            let first = gb.add_or_binary(ac, b);
            let second = gb.add_or_binary(ad, b);
            AigBitVector::from_lsb_is_index_0(&[first, second])
        });
        let trial = trial.unwrap();
        assert_eq!(trial.before.and_nodes, 3);
        assert_eq!(trial.after.and_nodes, 4);
        assert!(!trial.accepted);
        assert!(trial.appended > 0);
        assert_eq!(gb.gates, before);
        assert_eq!(
            result.iter_lsb_to_msb().collect::<Vec<_>>(),
            ordinary.iter_lsb_to_msb().collect::<Vec<_>>()
        );
        let rebuilt = gb.add_or_binary(a, c);
        assert_eq!(rebuilt.node.id, before.len());
        assert_eq!(gb.aig_depth(rebuilt), Some(1));
    }

    #[test]
    fn a_strictly_smaller_factorization_is_retained() {
        let mut gb = GateBuilder::new("factor".to_string(), GateBuilderOptions::opt());
        let inputs = gb.add_input("inputs".to_string(), 3);
        let [a, b, x] = std::array::from_fn(|i| *inputs.get_lsb(i));
        let frontier = gb.gates.len();
        let ax = gb.add_and_binary(a, x);
        let bx = gb.add_and_binary(b, x);
        let ordinary = gb.add_or_binary(ax, bx);
        let (_, trial) = gb.try_region_alternative(ordinary.into(), frontier, &[], |gb| {
            let either = gb.add_or_binary(a, b);
            gb.add_and_binary(either, x).into()
        });
        let trial = trial.unwrap();
        assert_eq!(trial.before.and_nodes, 3);
        assert_eq!(trial.after.and_nodes, 2);
        assert_eq!(trial.after.depth, trial.before.depth);
        assert!(trial.accepted);
    }

    #[test]
    fn aggregate_savings_cannot_hide_a_delayed_output() {
        let mut gb = GateBuilder::new(
            "arrival_frontier".to_string(),
            GateBuilderOptions {
                fold: false,
                hash: true,
            },
        );
        let inputs = gb.add_input("inputs".to_string(), 3);
        let [a, b, c] = std::array::from_fn(|i| *inputs.get_lsb(i));
        let frontier = gb.gates.len();
        let bb = gb.add_and_binary(b, b);
        let cc = gb.add_and_binary(c, c);
        let deep = gb.add_and_binary(bb, cc);
        let ordinary = AigBitVector::from_lsb_is_index_0(&[a, deep]);
        let before = gb.gates.clone();
        let (_, trial) = gb.try_region_alternative(ordinary, frontier, &[], |gb| {
            let delayed = gb.add_and_binary(a, a);
            let factored = gb.add_and_binary(b, c);
            AigBitVector::from_lsb_is_index_0(&[delayed, factored])
        });
        let trial = trial.unwrap();
        assert!(trial.after.and_nodes < trial.before.and_nodes);
        assert!(trial.after.depth < trial.before.depth);
        assert_eq!(trial.delayed_outputs, 1);
        assert!(!trial.accepted);
        assert_eq!(gb.gates, before);
    }

    #[test]
    fn zero_cost_regions_skip_the_trial() {
        let mut gb = GateBuilder::new("zero".to_string(), GateBuilderOptions::opt());
        let inputs = gb.add_input("inputs".to_string(), 2);
        let a = *inputs.get_lsb(0);
        let ab = gb.add_and_binary(a, *inputs.get_lsb(1));
        let frontier = gb.gates.len();
        for root in [a, ab] {
            let (result, trial) = gb.try_region_alternative(root.into(), frontier, &[], |_| {
                panic!("a region with no suffix ANDs must skip the trial")
            });
            assert!(trial.is_none());
            assert_eq!(*result.get_lsb(0), root);
        }
    }

    #[test]
    fn equal_area_requires_a_strictly_smaller_maximum_output_depth() {
        let mut gb = GateBuilder::new("balance".to_string(), GateBuilderOptions::opt());
        let inputs = gb.add_input("inputs".to_string(), 5);
        let [a, b, c, d, e] = std::array::from_fn(|i| *inputs.get_lsb(i));
        let frontier = gb.gates.len();
        let ab = gb.add_and_binary(a, b);
        let abc = gb.add_and_binary(ab, c);
        let chain = gb.add_and_binary(abc, d);
        let tail = gb.add_and_binary(chain, e);
        // Retain the ordinary roots when balancing only a noncritical output.
        // The tail is a boundary for the region count but retains its full
        // depth.
        let ordinary = AigBitVector::from_lsb_is_index_0(&[chain, tail]);
        let (result, trial) = gb.try_region_alternative(ordinary, frontier, &[tail], |gb| {
            let cd = gb.add_and_binary(c, d);
            let balanced = gb.add_and_binary(ab, cd);
            AigBitVector::from_lsb_is_index_0(&[balanced, tail])
        });
        let trial = trial.unwrap();
        assert_eq!(trial.before, trial.after);
        assert_eq!(trial.delayed_outputs, 0);
        assert!(!trial.accepted);
        assert_eq!(*result.get_lsb(0), chain);

        // With only the first output, the same rewrite reduces maximum depth.
        let (_, trial) = gb.try_region_alternative(chain.into(), frontier, &[], |gb| {
            let cd = gb.add_and_binary(c, d);
            gb.add_and_binary(ab, cd).into()
        });
        let trial = trial.unwrap();
        assert_eq!(trial.before.and_nodes, trial.after.and_nodes);
        assert_eq!((trial.before.depth, trial.after.depth), (3, 2));
        assert!(trial.accepted);
    }

    #[test]
    fn no_arrival_cache_does_not_execute_the_trial() {
        let mut gb = GateBuilder::new("fallback".to_string(), GateBuilderOptions::no_opt());
        let inputs = gb.add_input("inputs".to_string(), 2);
        let frontier = gb.gates.len();
        let ordinary = gb.add_and_binary(*inputs.get_lsb(0), *inputs.get_lsb(1));
        let (result, trial) = gb.try_region_alternative(ordinary.into(), frontier, &[], |_| {
            panic!("hash-disabled mapping must retain ordinary lowering")
        });
        assert!(trial.is_none());
        assert_eq!(*result.get_lsb(0), ordinary);
    }
}
