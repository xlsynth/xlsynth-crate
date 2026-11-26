// SPDX-License-Identifier: Apache-2.0

//! Bounded compatible-cell nomination. Estimates order trials; incremental STA
//! decides whether each replacement is legal and worthwhile.

use std::collections::HashMap;

use super::{AREA_COMPARISON_TOLERANCE, CellFamilies, ResizeOptions};
use crate::liberty_model::{Library, PinDirection};
use crate::netlist::sta::incremental::IncrementalSta;
use crate::netlist::sta::{PreparedStaLibrary, fixed_pin_delays};

/// How to select pin-compatible, smaller cells for each area-recovery trial.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecoveryCellMenu {
    /// First cells in increasing (area, name) order.
    #[default]
    Area,
    /// The fastest cell in each cheap area group, then the fastest remaining
    /// cell. A single trial takes the minimum-area cell by name.
    CheapestGroupsThenFastest,
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

/// One phase's fixed library and input transition, with delay estimates keyed
/// by candidate and the exact output load. Borrow STA only for each query.
struct CellDelayEstimates<'a> {
    library: &'a Library,
    transition: f64,
    delays: HashMap<(usize, u64), Option<f64>>,
}

impl<'a> CellDelayEstimates<'a> {
    fn new(library: &'a Library, transition: f64) -> Self {
        Self {
            library,
            transition,
            delays: HashMap::new(),
        }
    }

    /// Estimate the worst input-to-output arc at the current load. The caller
    /// must use STA prepared from this selector's library.
    fn estimated_delay(
        &mut self,
        cell: usize,
        load: f64,
        prepared: &PreparedStaLibrary<'_>,
    ) -> Option<f64> {
        *self
            .delays
            .entry((cell, load.to_bits()))
            .or_insert_with(|| {
                loaded_path_delay(self.library, prepared, cell, self.transition, load)
            })
    }
}

/// Nominates compatible, distinct, eligible cells in deterministic trial order.
/// Library, families, transition, menus, and limits are fixed for the phase.
/// Returned indices belong to the same library as the STA instance.
pub(super) struct CandidateSelector<'a> {
    library: &'a Library,
    families: &'a CellFamilies,
    options: &'a ResizeOptions,
    estimates: CellDelayEstimates<'a>,
    ranked_by_delay: Vec<(usize, f64)>,
    recovery_menus: HashMap<(usize, u64), Vec<usize>>,
    choices: Vec<usize>,
}

impl<'a> CandidateSelector<'a> {
    pub(super) fn new<'lib: 'a>(
        prepared: &PreparedStaLibrary<'lib>,
        families: &'a CellFamilies,
        options: &'a ResizeOptions,
    ) -> Self {
        let library = prepared.library();
        Self {
            library,
            families,
            options,
            estimates: CellDelayEstimates::new(library, options.timing.primary_input_transition),
            ranked_by_delay: Vec::new(),
            recovery_menus: HashMap::new(),
            choices: Vec::new(),
        }
    }

    /// Select timing candidates after applying the pass-wide area allowance.
    pub(super) fn timing_candidates(
        &mut self,
        instance: usize,
        sta: &IncrementalSta<'_>,
        area: f64,
        area_limit: f64,
    ) -> &[usize] {
        debug_assert!(std::ptr::eq(self.library, sta.prepared_library().library()));
        let current = sta.cell_indices()[instance];
        self.choices.clear();
        self.choices.extend(
            self.families
                .alternatives(current)
                .iter()
                .copied()
                .filter(|&cell| {
                    cell != current
                        && area - self.library.cells[current].area + self.library.cells[cell].area
                            <= area_limit + AREA_COMPARISON_TOLERANCE
                }),
        );
        let width = self.options.max_cell_candidates;
        if self.options.timing_cell_menu != TimingCellMenu::DelayBandsFastestFirst
            || !self.delay_band_shortlist(instance, sta, width)
        {
            let current_area = self.library.cells[current].area;
            self.choices.sort_by(|&a, &b| {
                (self.library.cells[a].area - current_area)
                    .abs()
                    .total_cmp(&(self.library.cells[b].area - current_area).abs())
                    .then_with(|| self.library.cells[a].name.cmp(&self.library.cells[b].name))
            });
        }
        self.choices.truncate(width);
        &self.choices
    }

    /// Select smaller recovery candidates, preserving area order when the full
    /// family fits, only one trial is allowed, or delay estimates are
    /// unavailable.
    pub(super) fn recovery_candidates(
        &mut self,
        instance: usize,
        sta: &IncrementalSta<'_>,
        area: f64,
        area_limit: f64,
    ) -> &[usize] {
        debug_assert!(std::ptr::eq(self.library, sta.prepared_library().library()));
        let current = sta.cell_indices()[instance];
        let width = self
            .options
            .recovery_cell_limit
            .unwrap_or(self.options.max_cell_candidates);
        let family = self.families.alternatives(current);
        let end = family.partition_point(|&cell| {
            self.library.cells[cell].area < self.library.cells[current].area
        });
        let eligible = &family[..end];
        let shortlist = if eligible.len() <= width
            || width <= 1
            || self.options.recovery_cell_menu == RecoveryCellMenu::Area
        {
            eligible
        } else if let Some(load) = sta.single_output_load(instance) {
            let estimates = &mut self.estimates;
            let lib = self.library;
            self.recovery_menus
                .entry((current, load.to_bits()))
                .or_insert_with(|| {
                    let mut fastest = |cells: &[usize]| {
                        let mut best = cells[0];
                        let mut best_delay = f64::INFINITY;
                        for &cell in cells {
                            // Missing estimates and equal-delay ties preserve
                            // area/name order.
                            if let Some(delay) =
                                estimates.estimated_delay(cell, load, sta.prepared_library())
                                && delay < best_delay
                            {
                                best = cell;
                                best_delay = delay;
                            }
                        }
                        best
                    };
                    let mut rest = eligible;
                    let mut choices = Vec::with_capacity(width);
                    while !rest.is_empty() && choices.len() + 1 < width {
                        let end = rest.partition_point(|&cell| {
                            lib.cells[cell].area == lib.cells[rest[0]].area
                        });
                        choices.push(fastest(&rest[..end]));
                        rest = &rest[end..];
                    }
                    if !rest.is_empty() {
                        choices.push(fastest(rest));
                    }
                    choices
                })
                .as_slice()
        } else {
            // Sequential and multi-output candidates use ordinary area order.
            eligible
        };
        self.choices.clear();
        self.choices.extend(
            shortlist
                .iter()
                .copied()
                .filter(|&cell| {
                    cell != current
                        && self.library.cells[cell].area < self.library.cells[current].area
                        && area - self.library.cells[current].area + self.library.cells[cell].area
                            <= area_limit + AREA_COMPARISON_TOLERANCE
                })
                .take(width),
        );
        &self.choices
    }

    /// Anchor the fastest cell and select the cheapest representative of each
    /// remaining equal-rank delay band. With no finite estimate, report failure
    /// so the caller can use its nearest-area fallback.
    fn delay_band_shortlist(
        &mut self,
        instance: usize,
        sta: &IncrementalSta<'_>,
        width: usize,
    ) -> bool {
        if width == 0 || self.choices.is_empty() {
            return false;
        }
        let Some(load) = sta.single_output_load(instance) else {
            return false;
        };
        self.ranked_by_delay.clear();
        for &cell in &self.choices {
            let delay = self
                .estimates
                .estimated_delay(cell, load, sta.prepared_library())
                .filter(|delay| delay.is_finite())
                .unwrap_or(f64::INFINITY);
            self.ranked_by_delay.push((cell, delay));
        }
        let lib = self.library;
        let ranked = &mut self.ranked_by_delay;
        ranked.sort_by(|&(a, ad), &(b, bd)| {
            ad.total_cmp(&bd)
                .then_with(|| lib.cells[a].area.total_cmp(&lib.cells[b].area))
                .then_with(|| lib.cells[a].name.cmp(&lib.cells[b].name))
        });
        if !ranked[0].1.is_finite() {
            return false;
        }
        self.choices.clear();
        self.choices.push(ranked[0].0);
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
            self.choices.push(cell);
        }
        true
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
