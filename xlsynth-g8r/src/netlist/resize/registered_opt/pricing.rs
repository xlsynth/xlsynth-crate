// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use super::{BoundaryDelays, ElectricalGuard, Score, preserves_endpoints};
use crate::liberty_model::Library;
use crate::netlist::sta::incremental::{ElectricalViolations, IncrementalSta, TimingEndpoint};
use anyhow::Result;

/// Endpoint identities and reachability from inputs and registers at the start
/// of a round.
pub(super) struct EndpointShape(Vec<(TimingEndpoint, [bool; 2])>);

impl EndpointShape {
    pub(super) fn new(sta: &IncrementalSta<'_>) -> Self {
        Self(Self::entries(sta).collect())
    }

    fn entries<'s>(
        sta: &'s IncrementalSta<'_>,
    ) -> impl Iterator<Item = (TimingEndpoint, [bool; 2])> + 's {
        sta.endpoint_values()
            .map(|(endpoint, arrivals)| (endpoint, arrivals.map(|a| a.is_some())))
    }

    fn matches(&self, trial: &IncrementalSta<'_>) -> bool {
        Self::entries(trial).eq(self.0.iter().copied())
    }

    /// Compare two simultaneously available timing states without copying their
    /// endpoints.
    pub(super) fn preserved_between(old: &IncrementalSta<'_>, new: &IncrementalSta<'_>) -> bool {
        Self::entries(old).eq(Self::entries(new))
    }
}

/// Decision for one wiring candidate. A retry asks for additional buffering
/// on the listed output pins.
#[derive(Debug, PartialEq)]
pub(super) enum Decision {
    Skip,
    Retry {
        rank: (u8, f64),
        outputs: Vec<(usize, String)>,
    },
    Keep {
        rank: (u8, f64),
        score: Score,
    },
}

/// Current circuit score and limits used to evaluate a wiring candidate.
pub(super) struct TopologyPricing {
    pub(super) score: Score,
    pub(super) initial_delay: BoundaryDelays,
    pub(super) weight: f64,
    pub(super) area_limit: f64,
    pub(super) guard: ElectricalGuard,
    pub(super) output_budget: usize,
    pub(super) best_rank: Option<(u8, f64)>,
}

impl TopologyPricing {
    /// Compare independently built timing states for the current circuit and
    /// the proposed circuit.
    pub(super) fn full(
        &self,
        old: &IncrementalSta<'_>,
        trial: &IncrementalSta<'_>,
        library: &Library,
        objective: usize,
    ) -> Result<Decision> {
        let next = Score::measure(trial, library, objective)?;
        Ok(self.decide(
            next,
            || preserves_endpoints(old, trial),
            || {
                let violations =
                    (0..trial.instance_count()).map(|i| (i, trial.cached_electrical_violations(i)));
                self.electrical_progress(old, trial, old.instance_count(), violations)
            },
        ))
    }

    /// Evaluate a temporary edit before its electrical results enter the cache.
    pub(super) fn incremental(
        &self,
        trial: &IncrementalSta<'_>,
        affected: &BTreeSet<usize>,
        incumbent_count: usize,
        endpoints: &EndpointShape,
        library: &Library,
        objective: usize,
    ) -> Result<Decision> {
        // Keeping existing masters makes incumbent area the exact prefix of the
        // full sum. Fold additions in instance order, including its rounding.
        let area = trial.cell_indices()[incumbent_count..]
            .iter()
            .fold(self.score.area, |area, &cell| {
                area + library.cells[cell].area
            });
        let next = Score::measure_with_area(trial, objective, area)?;
        Ok(self.decide(
            next,
            || endpoints.matches(trial),
            || {
                // Existing masters are unchanged and their cached pin
                // violations still describe the incumbent until the transaction
                // commits.
                let violations = affected
                    .iter()
                    .map(|&i| (i, trial.electrical_violations(i)));
                self.electrical_progress(trial, trial, incumbent_count, violations)
            },
        ))
    }

    /// Compare electrical violations in stable instance and pin order. Any
    /// newly inserted pin starts with zero allowed excess. Worsened existing
    /// pins may request a repair retry within the caller's driver budget.
    fn electrical_progress<Pins: AsRef<[ElectricalViolations]>>(
        &self,
        incumbent: &IncrementalSta<'_>,
        trial: &IncrementalSta<'_>,
        incumbent_count: usize,
        violations: impl Iterator<Item = (usize, Pins)>,
    ) -> Option<(bool, Vec<(usize, String)>)> {
        let mut improved = false;
        let mut outputs = Vec::new();
        for (instance, pins) in violations {
            let before = if instance < incumbent_count {
                incumbent.cached_electrical_violations(instance)
            } else {
                &[]
            };
            for (pin, &raw) in pins.as_ref().iter().enumerate() {
                let previous = before.get(pin).map_or_else(Default::default, |&value| {
                    self.guard
                        .excess(incumbent, incumbent.cell_indices()[instance], pin, value)
                });
                let next = self
                    .guard
                    .excess(trial, trial.cell_indices()[instance], pin, raw);
                if self.guard.worsens(next, previous) {
                    if self.output_budget == 0 || instance >= incumbent_count {
                        return None;
                    }
                    // One net's transition can exceed limits at its driver and
                    // at many receivers. Nominate that driver once, using trial
                    // connectivity for both full and incremental pricing. The
                    // expanded trial must pass every pin check with no retries.
                    let (driver, name) = trial.electrical_driver(instance, pin)?;
                    if driver >= incumbent_count {
                        return None;
                    }
                    if !outputs.iter().any(|(d, n)| *d == driver && n == name) {
                        if outputs.len() == self.output_budget {
                            return None;
                        }
                        outputs.push((driver, name.to_owned()));
                    }
                }
                improved |= self.guard.worsens(previous, next);
            }
        }
        Some((improved, outputs))
    }

    /// Preserve the rank and veto order while avoiding expensive checks on an
    /// already losing rank.
    fn decide(
        &self,
        next: Score,
        endpoints_preserved: impl FnOnce() -> bool,
        electrical_progress: impl FnOnce() -> Option<(bool, Vec<(usize, String)>)>,
    ) -> Decision {
        let Some(rank) = next.rank(self.score, self.weight) else {
            return Decision::Skip;
        };
        if !next.boundaries.within(self.initial_delay)
            || !self.best_rank.is_none_or(|best| rank < best)
            || !endpoints_preserved()
        {
            return Decision::Skip;
        }
        let Some((improved, outputs)) = electrical_progress() else {
            return Decision::Skip;
        };
        if next.area > self.area_limit && !improved {
            return Decision::Skip;
        }
        if outputs.is_empty() {
            Decision::Keep { rank, score: next }
        } else {
            Decision::Retry { rank, outputs }
        }
    }
}
