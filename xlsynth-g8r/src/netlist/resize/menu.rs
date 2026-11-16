// SPDX-License-Identifier: Apache-2.0

//! Shortlists compatible cells using a cheap loaded-path estimate.
//! The existing incremental STA still decides every actual replacement.

use std::collections::HashMap;

use crate::liberty::io_defaults::resolve_reference_buffer;
use crate::liberty_model::{Library, PinDirection};
use crate::netlist::fanout_opt::pin_cap;
use crate::netlist::sta::incremental::IncrementalSta;
use crate::netlist::sta::{PreparedStaLibrary, fixed_pin_delays};

/// How to select pin-compatible, smaller cells for each area-recovery trial.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecoveryCellMenu {
    /// First cells in increasing (area, name) order.
    #[default]
    Area,
    /// Use loaded delay to break minimum-cost ties when shortening a list to
    /// at least two trials. A single trial takes the first minimum by name
    /// and avoids estimating delays for the family.
    ElectricalMinArea,
}

/// How timing search nominates cells for its bounded, exact timing trials.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimingCellMenu {
    /// Try the nearest physical cell area first, breaking ties by name.
    #[default]
    NearestArea,
    /// Try the fastest predicted path first, then the cheapest cell in each
    /// remaining equal-rank delay band, in increasing band order.
    DelayBandsFastestFirst,
}

/// Per-phase estimates at a fixed transition. Changes in the downstream load
/// select a distinct cache entry. The timing engine still prices each trial.
#[derive(Default)]
pub(super) struct LoadedPaths {
    delays: HashMap<(usize, u64), Option<f64>>,
    ranked_by_delay: Vec<(usize, f64)>,
    driver_delays: HashMap<u64, Option<f64>>,
    reference_buffer: Option<Option<usize>>,
}

impl LoadedPaths {
    /// Estimate the slowest arc through the candidate and its nominal driver.
    fn cost(
        &mut self,
        cell: usize,
        load: f64,
        transition: f64,
        lib: &Library,
        prepared: &PreparedStaLibrary<'_>,
    ) -> Option<f64> {
        let reference = *self.reference_buffer.get_or_insert_with(|| {
            let name = resolve_reference_buffer(lib).ok()?;
            lib.cells.iter().position(|cell| cell.name == name)
        });
        *self
            .delays
            .entry((cell, load.to_bits()))
            .or_insert_with(|| {
                loaded_path_delay(
                    lib,
                    prepared,
                    cell,
                    transition,
                    load,
                    reference,
                    &mut self.driver_delays,
                )
            })
    }

    /// Select eligible nominees and order them for bounded incremental STA
    /// trials.
    pub(super) fn timing_shortlist(
        &mut self,
        candidates: &mut Vec<usize>,
        width: usize,
        sta: &IncrementalSta<'_>,
        instance: usize,
        options: &super::ResizeOptions,
        lib: &Library,
    ) {
        if options.timing_cell_menu == TimingCellMenu::DelayBandsFastestFirst
            && self.delay_band_shortlist(candidates, width, sta, instance, options, lib)
        {
            return;
        }
        let current_area = lib.cells[sta.cell_indices()[instance]].area;
        candidates.sort_by(|&a, &b| {
            (lib.cells[a].area - current_area)
                .abs()
                .total_cmp(&(lib.cells[b].area - current_area).abs())
                .then_with(|| lib.cells[a].name.cmp(&lib.cells[b].name))
        });
    }

    /// Anchor the fastest cell, then take the minimum-area cell in each
    /// equal-rank band of the remaining candidates. Missing estimates rank
    /// last. Without any finite estimate, preserve the full eligible list.
    fn delay_band_shortlist(
        &mut self,
        candidates: &mut Vec<usize>,
        width: usize,
        sta: &IncrementalSta<'_>,
        instance: usize,
        options: &super::ResizeOptions,
        lib: &Library,
    ) -> bool {
        if width == 0 || candidates.is_empty() {
            return false;
        }
        let Some(load) = sta.single_output_load(instance) else {
            return false;
        };
        let transition = options.timing.primary_input_transition;
        self.ranked_by_delay.clear();
        for &cell in candidates.iter() {
            let delay = self
                .cost(cell, load, transition, lib, sta.prepared_library())
                .filter(|delay| delay.is_finite())
                .unwrap_or(f64::INFINITY);
            self.ranked_by_delay.push((cell, delay));
        }
        let ranked = &mut self.ranked_by_delay;
        ranked.sort_by(|&(a, ad), &(b, bd)| {
            ad.total_cmp(&bd)
                .then_with(|| lib.cells[a].area.total_cmp(&lib.cells[b].area))
                .then_with(|| lib.cells[a].name.cmp(&lib.cells[b].name))
        });
        if !ranked[0].1.is_finite() {
            return false;
        }
        candidates.clear();
        candidates.push(ranked[0].0);
        // Use one slot for the fastest cell. Spend the rest on cheaper cells at
        // different points in the delay ranking so a cluster of nearly
        // identical fast cells cannot consume every exact timing trial.
        let bands = width.min(ranked.len()) - 1;
        let remaining = &ranked[1..];
        for band in 0..bands {
            let start = band * remaining.len() / bands;
            let end = (band + 1) * remaining.len() / bands;
            let &(cell, _) = remaining[start..end]
                .iter()
                .min_by(|&&(a, ad), &&(b, bd)| {
                    lib.cells[a]
                        .area
                        .total_cmp(&lib.cells[b].area)
                        .then_with(|| ad.total_cmp(&bd))
                        .then_with(|| lib.cells[a].name.cmp(&lib.cells[b].name))
                })
                .expect("each delay band contains at least one cell");
            candidates.push(cell);
        }
        true
    }
}

/// All shortlist data belongs to one recovery phase with fixed library,
/// transition, menu and width. Only the load on an instance can change.
#[derive(Default)]
pub(super) struct RecoveryMenus {
    indices: HashMap<(usize, u64), Vec<usize>>,
    paths: LoadedPaths,
}

impl RecoveryMenus {
    pub(super) fn shortlist<'a>(
        &'a mut self,
        families: &'a super::CellFamilies,
        current: usize,
        sta: &IncrementalSta<'_>,
        instance: usize,
        options: &super::ResizeOptions,
        lib: &Library,
    ) -> &'a [usize] {
        let width = options
            .recovery_cell_limit
            .unwrap_or(options.max_cell_candidates);
        let sorted = families.alternatives(current);
        let end = sorted.partition_point(|&c| families.costs[c] < families.costs[current]);
        let eligible = &sorted[..end];
        if eligible.len() <= width
            || width <= 1
            || options.recovery_cell_menu == RecoveryCellMenu::Area
        {
            return eligible;
        }

        // A sequential or multi-output cell needs another timing model. Keep
        // these families in normal area order for every policy.
        let Some(actual_load) = sta.single_output_load(instance) else {
            return eligible;
        };
        let load = load_bucket(actual_load);
        let key = (current, load.to_bits());
        self.indices.entry(key).or_insert_with(|| {
            let prepared = sta.prepared_library();
            let transition = options.timing.primary_input_transition;
            let mut fastest = |cells: &[usize]| {
                let mut best = cells[0];
                let mut best_cost = f64::INFINITY;
                for &cell in cells {
                    // Missing estimates and equal-cost ties preserve area/name
                    // order.
                    if let Some(value) = self.paths.cost(cell, load, transition, lib, prepared)
                        && value < best_cost
                    {
                        best = cell;
                        best_cost = value;
                    }
                }
                best
            };
            let mut result = vec![eligible[0]];
            // Nominate the fastest loaded candidate in each band by area.
            for band in 0..width - 1 {
                let start = 1 + band * (eligible.len() - 1) / (width - 1);
                let end = 1 + (band + 1) * (eligible.len() - 1) / (width - 1);
                result.push(fastest(&eligible[start..end]));
            }
            let minimum_cost = families.costs[eligible[0]];
            let end = eligible.partition_point(|&c| families.costs[c] == minimum_cost);
            if end > 1 {
                // Try a minimum-cost cell first. If several cells have the same
                // area or leakage cost, choose the fastest estimated one so a
                // limited trial budget has a better chance of finding a legal
                // replacement.
                let first = fastest(&eligible[..end]);
                // An existing nominee trades places with the first one so
                // the budget still buys the same number of distinct trials.
                if let Some(position) = result.iter().position(|&c| c == first) {
                    result.swap(0, position);
                } else {
                    result[0] = first;
                }
            }
            result
        })
    }
}

/// Use a representative positive load within a factor of sqrt(2) of the
/// current worst edge. This bounds cache growth on designs with many fanouts.
fn load_bucket(load: f64) -> f64 {
    if load == 0.0 {
        return 0.0;
    }
    let representative = load.log2().round().exp2();
    if representative.is_finite() {
        representative
    } else {
        load
    }
}

/// Worst input-to-output path through an equivalent candidate and a common
/// reference driver driving that candidate's pin capacitance. If the Liberty
/// lacks a reference buffer use the candidate delay alone. These estimates
/// rank trials only. Timing at actual upstream cells is recomputed by STA.
fn loaded_path_delay(
    lib: &Library,
    prepared: &PreparedStaLibrary<'_>,
    cell: usize,
    transition: f64,
    load: f64,
    reference: Option<usize>,
    driver_delays: &mut HashMap<u64, Option<f64>>,
) -> Option<f64> {
    let candidate = &lib.cells[cell];
    if candidate
        .pins
        .iter()
        .filter(|p| p.direction == PinDirection::Output as i32)
        .count()
        != 1
    {
        return None;
    }
    fixed_pin_delays(prepared, cell, transition, load)
        .ok()?
        .into_iter()
        .map(|(pin_name, delay)| {
            let driver = if let Some(buffer) = reference {
                let pin = candidate
                    .pins
                    .iter()
                    .find(|p| lib.resolve_string(&p.name) == pin_name)?;
                let [rise, fall] = pin_cap(pin);
                let cap = rise.max(fall);
                let upstream = driver_delays.entry(cap.to_bits()).or_insert_with(|| {
                    fixed_pin_delays(prepared, buffer, transition, cap)
                        .ok()?
                        .first()
                        .map(|(_, d)| *d)
                });
                (*upstream)?
            } else {
                0.0
            };
            Some(delay + driver)
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max_by(f64::total_cmp)
}
