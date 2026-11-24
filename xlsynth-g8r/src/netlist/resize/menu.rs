// SPDX-License-Identifier: Apache-2.0

//! Shortlists compatible cells using a cheap loaded-path estimate.
//! The existing incremental STA still decides every actual replacement.

use std::collections::HashMap;

use crate::liberty_model::{Library, PinDirection};
use crate::netlist::sta::incremental::IncrementalSta;
use crate::netlist::sta::{PreparedStaLibrary, fixed_pin_delays};

/// How to select pin-compatible, smaller cells for each area-recovery trial.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecoveryCellMenu {
    /// First cells in increasing (area, name) order.
    #[default]
    Area,
    /// Shortlist the fastest cell at each low cost. Reserve the last
    /// trial for the fastest remaining cell. One trial takes the first
    /// minimum by name without estimating delay.
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
}

impl LoadedPaths {
    /// Estimate the slowest arc through the candidate at its output load.
    fn cost(
        &mut self,
        cell: usize,
        load: f64,
        transition: f64,
        lib: &Library,
        prepared: &PreparedStaLibrary<'_>,
    ) -> Option<f64> {
        *self
            .delays
            .entry((cell, load.to_bits()))
            .or_insert_with(|| loaded_path_delay(lib, prepared, cell, transition, load))
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
        area_sorted: &'a [usize],
        current: usize,
        sta: &IncrementalSta<'_>,
        instance: usize,
        options: &super::ResizeOptions,
        lib: &Library,
    ) -> &'a [usize] {
        let width = options
            .recovery_cell_limit
            .unwrap_or(options.max_cell_candidates);
        let end = area_sorted.partition_point(|&c| lib.cells[c].area < lib.cells[current].area);
        let eligible = &area_sorted[..end];
        if eligible.len() <= width
            || width <= 1
            || options.recovery_cell_menu == RecoveryCellMenu::Area
        {
            return eligible;
        }

        // A sequential or multi-output cell needs another timing model. Keep
        // these families in normal area order for every policy.
        let Some(load) = sta.single_output_load(instance) else {
            return eligible;
        };
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
            // Give each cheap group a trial. Keep the final slot for a
            // faster cell even when it lies outside those groups.
            let mut rest = eligible;
            let mut choices = Vec::with_capacity(width);
            let cost = |cell: usize| lib.cells[cell].area;
            while !rest.is_empty() && choices.len() + 1 < width {
                let end = rest.partition_point(|&cell| cost(cell) == cost(rest[0]));
                choices.push(fastest(&rest[..end]));
                rest = &rest[end..];
            }
            if !rest.is_empty() {
                choices.push(fastest(rest));
            }
            choices
        })
    }
}

/// Slowest input-to-output arc at the given load and input transition.
/// Incremental STA evaluates the actual upstream delay for each trial.
fn loaded_path_delay(
    lib: &Library,
    prepared: &PreparedStaLibrary<'_>,
    cell: usize,
    transition: f64,
    load: f64,
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
        .map(|(_, delay)| delay)
        .max_by(f64::total_cmp)
}
