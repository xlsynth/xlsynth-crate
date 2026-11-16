// SPDX-License-Identifier: Apache-2.0

//! Find simultaneous repairs on competing registered timing paths. A tie can
//! require far more edited roots than the number of affordable STA trials.

use super::coordination::{editable_sites, select_endpoint_roots};
use super::edit::Branch;
use super::graph::{Buffer, Root};
use super::plan;
use crate::netlist::sta::StaOptions;
use crate::netlist::sta::incremental::urgency::PinTimingUrgency;
use crate::netlist::sta::incremental::{IncrementalSta, TIMING_COMPARISON_TOLERANCE};
use anyhow::Result;
use std::collections::HashMap;

/// Buildable buffer trees for each timing window, sorted by the original
/// driver ID.
pub(super) type ConeWindows = [Vec<(usize, Vec<Branch>)>; 2];

/// Choose candidate drivers and build their trees before trial edits mutate
/// STA state. Keep the order of buffer choices, widths, and timing windows.
pub(super) fn plan_windows(
    cones: Option<TimingCones<'_, '_>>,
    roots: &[Root],
    buffers: &[&Buffer],
    widths: &[usize],
    timing: StaOptions,
) -> HashMap<(usize, usize), ConeWindows> {
    let mut plans = HashMap::new();
    let Some(cones) = cones else {
        return plans;
    };
    for buffer in buffers {
        for &width in widths {
            // Cache both successful and unbuildable roots. The two windows can
            // visit the same sink and must use the same geometry.
            let mut trees = HashMap::new();
            let groups = cones.groups(|id| {
                trees
                    .entry(id)
                    .or_insert_with(|| plan::tree(&roots[id], buffer, width, timing))
                    .is_some()
            });
            let windows = groups.map(|group| {
                group
                    .into_iter()
                    .map(|id| {
                        (
                            id,
                            trees[&id]
                                .as_ref()
                                .expect("selected buildable tree")
                                .clone(),
                        )
                    })
                    .collect()
            });
            plans.insert((buffer.cell, width), windows);
        }
    }
    plans
}

/// Timing and editable pin identities shared by each buffer choice and width
/// on this registered snapshot. No arrival or required time is recomputed for
/// a change in tree geometry.
pub(super) struct TimingCones<'a, 'lib> {
    sta: &'a IncrementalSta<'lib>,
    owners: HashMap<(usize, &'a str), usize>,
    urgency: PinTimingUrgency,
    windows: [f64; 2],
    boundary: usize,
}

impl<'a, 'lib> TimingCones<'a, 'lib> {
    pub(super) fn prepare(
        sta: &'a IncrementalSta<'lib>,
        roots: &[Root],
        eligible: &[usize],
        boundary: usize,
    ) -> Result<Option<Self>> {
        // Joint timing windows target register-to-register paths. Other
        // boundaries use the ordinary fanout candidates.
        if boundary != 3 || eligible.len() < 2 {
            return Ok(None);
        }
        let Some((_, worst)) = sta.endpoint_arrivals(boundary).next_back() else {
            return Ok(None);
        };
        let urgency = sta.pin_timing_urgency(boundary)?;
        // The tight window captures competing paths at the current delay limit.
        // The 5% window also considers nearby paths that can become the next
        // limit. Trying one forest per distinct window bounds the number of STA
        // trials.
        let tight = TIMING_COMPARISON_TOLERANCE;
        let widest = tight.max(0.05 * worst);
        let sites = eligible.iter().flat_map(|&id| {
            roots[id]
                .sinks
                .iter()
                .filter(|s| {
                    urgency
                        .sink_slack(s.instance, s.connection)
                        .is_some_and(|v| v <= widest)
                })
                .map(move |s| {
                    let instance = sta.instance(s.instance).expect("live root sink instance");
                    (id, s.instance, instance.pin_order[s.connection].as_str())
                })
        });
        Ok(editable_sites(sites).map(|owners| Self {
            sta,
            owners,
            urgency,
            windows: [tight, widest],
            boundary,
        }))
    }

    /// At each timing window, collect the closest feasible drivers across
    /// competing paths. An unbuildable downstream tree leaves its predecessors
    /// reachable. Group width is independent of the number of STA trials.
    pub(super) fn groups(&self, mut feasible: impl FnMut(usize) -> bool) -> [Vec<usize>; 2] {
        let mut groups = self.windows.map(|window| {
            select_endpoint_roots(
                self.sta,
                self.boundary,
                &self.owners,
                window,
                |sink, pin| {
                    let inst = self.sta.instance(sink).expect("live timing instance");
                    inst.pin_order
                        .iter()
                        .position(|s| s == pin)
                        .and_then(|index| self.urgency.sink_slack(sink, index))
                        .is_some_and(|slack| slack <= window)
                },
                &mut feasible,
            )
        });
        if groups[1] == groups[0] {
            groups[1].clear();
        }
        groups
    }
}

#[cfg(test)]
mod tests;
