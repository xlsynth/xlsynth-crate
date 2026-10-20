// SPDX-License-Identifier: Apache-2.0

//! Exact optimizer reductions indexed by the core transaction's affected IDs.
//! These records contain reporting values only, never graph or timing state.

use super::*;

/// Component-wise maxima over a contiguous range of instance records.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Maxima {
    slew: f64,
    load: f64,
    fanout: usize,
}
impl Maxima {
    fn merge(self, other: Self) -> Self {
        Self {
            slew: self.slew.max(other.slew),
            load: self.load.max(other.load),
            fanout: self.fanout.max(other.fanout),
        }
    }
    fn from_contribution(value: Option<Contribution>) -> Self {
        let value = value.unwrap_or_default();
        Self {
            slew: value.slew,
            load: value.load,
            fanout: value.fanout,
        }
    }
}

/// One instance's output-net contributions; counts are load-table, slew-table,
/// maximum-capacitance and maximum-transition violations, in that order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Contribution {
    pub slew: f64,
    pub load: f64,
    pub fanout: usize,
    pub counts: [usize; 4],
}

/// An array reduction tree avoids ordered-map allocation for changing maxima.
/// Leaves are instance IDs; internal nodes hold only reporting reductions.
#[derive(Default)]
pub(super) struct MetricsCache {
    instances: Vec<Option<Contribution>>,
    maxima: Vec<Maxima>,
    counts: [usize; 4],
}

impl MetricsCache {
    /// Updates only an affected contribution and returns its prior record.
    pub(super) fn set(&mut self, i: usize, next: Option<Contribution>) -> Option<Contribution> {
        if self.instances.len() <= i {
            self.instances.resize(i + 1, None);
        }
        let old = std::mem::replace(&mut self.instances[i], next);
        if old == next {
            return old;
        }
        self.update_maxima(i, next);
        for k in 0..4 {
            self.counts[k] -= old.map_or(0, |v| v.counts[k]);
            self.counts[k] += next.map_or(0, |v| v.counts[k]);
        }
        old
    }

    /// Repairs only the changed leaf's ancestors, stopping once a reduction
    /// is unchanged. Capacity growth rebuilds scalar rows once; speculative
    /// insertions may retain that capacity after their values are rolled back.
    fn update_maxima(&mut self, i: usize, next: Option<Contribution>) {
        let capacity = self.maxima.len() / 2;
        if capacity < self.instances.len() {
            let capacity = self.instances.len().next_power_of_two();
            self.maxima = vec![Maxima::default(); capacity * 2];
            for (j, &value) in self.instances.iter().enumerate() {
                self.maxima[capacity + j] = Maxima::from_contribution(value);
            }
            for n in (1..capacity).rev() {
                self.maxima[n] = self.maxima[n * 2].merge(self.maxima[n * 2 + 1]);
            }
            return;
        }
        let mut n = capacity + i;
        self.maxima[n] = Maxima::from_contribution(next);
        while n > 1 {
            n /= 2;
            let value = self.maxima[n * 2].merge(self.maxima[n * 2 + 1]);
            if self.maxima[n] == value {
                break;
            }
            self.maxima[n] = value;
        }
    }

    /// Materializes exact reductions; no graph-wide traversal is needed.
    pub(super) fn report(&self, sta: &IncrementalSta<'_>, area: f64) -> FanoutMetrics {
        let [load_table, slew_table, capacitance, transition] = self.counts;
        let maxima = self.maxima.get(1).copied().unwrap_or_default();
        FanoutMetrics {
            area,
            cell_count: sta.instance_count(),
            worst_delay: sta.score().input_to_output,
            max_slew: maxima.slew,
            max_load: maxima.load,
            max_fanout: maxima.fanout,
            output_load_above_table_max: load_table,
            input_slew_above_table_max: slew_table,
            max_capacitance_violations: capacitance,
            max_transition_violations: transition,
            timing_queries: sta
                .combinational_query_diagnostics()
                .expect("combinational optimizer"),
        }
    }

    /// Temporarily updates scalar reductions, then restores them before the
    /// caller's oracle can fail. The core journal owns all graph rollback.
    pub(super) fn preview(
        &mut self,
        changes: &[(usize, Option<Contribution>)],
        sta: &IncrementalSta<'_>,
        area: f64,
    ) -> FanoutMetrics {
        let old_len = self.instances.len();
        let journal: Vec<_> = changes
            .iter()
            .map(|&(i, next)| (i, self.set(i, next)))
            .collect();
        let result = self.report(sta, area);
        for (i, old) in journal.into_iter().rev() {
            self.set(i, old);
        }
        self.instances.truncate(old_len);
        result
    }
}
