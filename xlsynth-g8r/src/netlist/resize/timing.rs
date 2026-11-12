// SPDX-License-Identifier: Apache-2.0

//! Resize search ordering and acceptance over the standalone timing API.

use super::*;
use crate::netlist::sta::incremental::{
    ElectricalViolations, TimingEdit, TimingEndpoint as Endpoint, TimingMutation, TrialOptions,
};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashSet};

const ELECTRICAL_COMPARISON_TOLERANCE: f64 = 1e-9;
const NEAR_CRITICAL_ENDPOINT_LIMIT: usize = 32;
const NEAR_CRITICAL_WINDOW_FRACTION: f64 = 0.05;

/// Stable total ordering for timing scores and search priorities.
#[derive(Clone, Copy, Debug)]
pub(super) struct RankedIndex(f64, usize);
impl PartialEq for RankedIndex {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for RankedIndex {}
impl PartialOrd for RankedIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankedIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// Lexicographic progress below the worst delay: first reduce the number of
/// equally critical endpoints, then improve a bounded near-critical sum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TimingTieBreak {
    worst_count: usize,
    pub(super) near_sum: f64,
}

impl TimingTieBreak {
    /// Rank progress on tied paths directly, including nonpositive arrivals.
    pub fn improvement_rank(self, old: Self) -> Option<(u8, f64)> {
        if self.worst_count < old.worst_count {
            Some((1, self.worst_count as f64))
        } else if self.worst_count == old.worst_count
            && self.near_sum < old.near_sum - TIMING_COMPARISON_TOLERANCE
        {
            Some((2, self.near_sum))
        } else {
            None
        }
    }

    /// Returns a dimensionless progress ratio for area-efficiency ranking.
    /// A nonpositive baseline cannot justify paying additional area.
    pub fn relative_to(self, old: Self) -> f64 {
        if self.worst_count < old.worst_count {
            self.worst_count as f64 / old.worst_count as f64
        } else if old.near_sum > 0.0 {
            self.near_sum / old.near_sum
        } else {
            1.0
        }
    }
}

/// Resizing policy adapter; the timing engine owns all mutations and rollback.
pub(super) trait ResizeTimingPolicy {
    fn objective(&self) -> usize;
    fn secondary(&self, objective: usize) -> TimingTieBreak;
    fn critical_instances(
        &self,
        objective: usize,
        limit: usize,
        eligible: impl Fn(usize) -> bool,
    ) -> Vec<usize>;
    fn fanin_priorities(&self, i: usize, mode: usize, deficit: f64) -> SmallVec<[RankedIndex; 4]>;
    fn recovery_headroom(&self, limits: BoundaryDelays) -> Vec<f64>;
    fn trial(
        &mut self,
        instance: usize,
        cell: usize,
        work_limit: usize,
        limits: Option<BoundaryDelays>,
        accept: impl FnOnce(&Self) -> bool,
    ) -> Result<TrialOutcome>;
}

impl ResizeTimingPolicy for IncrementalSta<'_> {
    /// Selects register-to-register timing when reachable. Otherwise the
    /// longest currently reachable path determines the boundary.
    fn objective(&self) -> usize {
        if self.endpoint_maximum_count(3) > 0 {
            3
        } else {
            let delays = self.score().values();
            (0..3)
                .filter(|i| self.endpoint_maximum_count(*i) > 0)
                .max_by(|a, b| delays[*a].total_cmp(&delays[*b]))
                .unwrap_or(0)
        }
    }

    /// Breaks delay ties without losing progress on large critical plateaus.
    fn secondary(&self, objective: usize) -> TimingTieBreak {
        // Without a register path, ties can span boundary classes. A trial
        // can lower one class until a different class supplies the maximum.
        let worst_count = if objective == 3 {
            self.endpoint_maximum_count(3)
        } else {
            let delays = self.score().values();
            let worst = delays[self.objective()];
            (0..3)
                .filter(|&i| delays[i] == worst)
                .map(|i| self.endpoint_maximum_count(i))
                .sum()
        };
        TimingTieBreak {
            worst_count,
            near_sum: self
                .endpoint_arrivals(objective)
                .rev()
                .take(NEAR_CRITICAL_ENDPOINT_LIMIT)
                .map(|(_, arrival)| arrival)
                .sum(),
        }
    }

    /// Visits near-critical fanin by cumulative arrival deficit, once per
    /// round. This is only a search-order heuristic: asymmetric pin arcs
    /// and load/slew effects are accounted for by accurate candidate
    /// evaluation afterward. Ineligible cells are traversed but do not
    /// consume the limit of selectable candidates.
    fn critical_instances(
        &self,
        objective: usize,
        limit: usize,
        eligible: impl Fn(usize) -> bool,
    ) -> Vec<usize> {
        let mut queue = BTreeSet::new();
        let worst = self.score().values()[objective];
        for (endpoint, arrival) in self
            .endpoint_arrivals(objective)
            .rev()
            .take(NEAR_CRITICAL_ENDPOINT_LIMIT)
            .filter(|(_, arrival)| *arrival >= worst - worst.abs() * NEAR_CRITICAL_WINDOW_FRACTION)
        {
            match endpoint {
                Endpoint::Output(bit) => {
                    if let Some(i) = self.driver_for_bit(bit) {
                        queue.insert(RankedIndex(worst - arrival, i));
                    }
                }
                Endpoint::Capture(i) => {
                    queue.insert(RankedIndex(worst - arrival, i));
                    queue.extend(self.fanin_priorities(i, objective / 2, worst - arrival));
                }
            }
        }
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        while let Some(RankedIndex(deficit, i)) = queue.pop_first() {
            if !seen.insert(i) {
                continue;
            }
            if eligible(i) {
                result.push(i);
                if result.len() >= limit {
                    break;
                }
            }
            // A launch register's D cone belongs to another timing path.
            // Capture-register fanin is seeded separately above.
            if !self.is_register(i) {
                queue.extend(self.fanin_priorities(i, objective / 2, deficit));
            }
        }
        result
    }

    /// Ranks data-input drivers by arrival deficit in one launch class.
    fn fanin_priorities(&self, i: usize, mode: usize, deficit: f64) -> SmallVec<[RankedIndex; 4]> {
        let inputs = self.data_input_arrivals(i, mode);
        let latest = inputs
            .iter()
            .map(|(_, arrival)| *arrival)
            .fold(0.0, f64::max);
        inputs
            .into_iter()
            .map(|(driver, arrival)| RankedIndex(deficit + (latest - arrival).max(0.0), driver))
            .collect()
    }

    /// Estimates per-cell headroom once per recovery round, across all timing
    /// boundaries. This is an ordering heuristic, not a timing constraint:
    /// exact trials still account for pin arcs, slew and upstream load changes.
    fn recovery_headroom(&self, limits: BoundaryDelays) -> Vec<f64> {
        let mut result = vec![f64::INFINITY; self.instance_count()];
        for mode in 0..self.launch_class_count() {
            let mut margins = vec![f64::INFINITY; self.instance_count()];
            for (endpoint, values) in self.endpoint_values() {
                let Some(arrival) = values[mode] else {
                    continue;
                };
                let capture = usize::from(matches!(endpoint, Endpoint::Capture(_)));
                let margin = (limits.values()[mode * 2 + capture] - arrival).max(0.0);
                match endpoint {
                    Endpoint::Output(bit) => {
                        if let Some(i) = self.driver_for_bit(bit) {
                            margins[i] = margins[i].min(margin);
                        }
                    }
                    Endpoint::Capture(i) => {
                        margins[i] = margins[i].min(margin);
                        for RankedIndex(value, driver) in self.fanin_priorities(i, mode, margin) {
                            margins[driver] = margins[driver].min(value);
                        }
                    }
                }
            }
            for i in self.topological_instances().iter().rev().copied() {
                if !self.is_register(i) && margins[i].is_finite() {
                    for RankedIndex(value, driver) in self.fanin_priorities(i, mode, margins[i]) {
                        margins[driver] = margins[driver].min(value);
                    }
                }
            }
            for (result, margin) in result.iter_mut().zip(margins) {
                *result = result.min(margin);
            }
        }
        result
    }

    /// Preserves phase-wide budgets and rejects worsened electrical violations.
    fn trial(
        &mut self,
        instance: usize,
        cell: usize,
        work_limit: usize,
        limits: Option<BoundaryDelays>,
        accept: impl FnOnce(&Self) -> bool,
    ) -> Result<TrialOutcome> {
        let edit = TimingEdit {
            new_bits: 0,
            mutations: vec![TimingMutation::Resize { instance, cell }],
        };
        let options = TrialOptions {
            cell_budget: work_limit.saturating_sub(self.propagated_instances()),
            limits,
        };
        self.trial_edit(&edit, options, |timing, affected| {
            for &instance in affected {
                let current = timing.electrical_violations(instance);
                let previous = timing.cached_electrical_violations(instance);
                for (index, new) in current.iter().enumerate() {
                    let old = previous.get(index).copied().unwrap_or_default();
                    if electrical_worsens(*new, old) {
                        return Ok(false);
                    }
                }
            }
            Ok(accept(timing))
        })
        .map(|work| work.outcome)
    }
}

/// Electrical constraints cannot be traded for a timing or area improvement.
pub(super) fn electrical_worsens(new: ElectricalViolations, old: ElectricalViolations) -> bool {
    new.capacitance > old.capacitance + ELECTRICAL_COMPARISON_TOLERANCE
        || new.transition > old.transition + ELECTRICAL_COMPARISON_TOLERANCE
        || new.fanout > old.fanout + ELECTRICAL_COMPARISON_TOLERANCE
}
