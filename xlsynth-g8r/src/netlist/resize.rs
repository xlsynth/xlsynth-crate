// SPDX-License-Identifier: Apache-2.0

//! Bounded, topology-preserving timing improvement followed by area recovery.
//! Uses the same Liberty model as `gv-stats`; no placement, hold or clock-tree
//! timing is implied. Buffer insertion, pin swapping and logic remapping are
//! deliberately separate transformations.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, anyhow};
use serde::Serialize;
use string_interner::{StringInterner, backend::StringBackend, symbol::SymbolU32};

use crate::liberty::Library;
use crate::liberty::cell_formula::parse_formula;
use crate::liberty_model::{Cell, PinDirection};
use crate::netlist::parse::{Net, NetlistModule};
use crate::netlist::sta::StaOptions;
use crate::netlist::sta::incremental::IncrementalSta;
pub use crate::netlist::sta::incremental::{
    BoundaryDelays, TrialTolerance, measure_boundary_delays,
};

/// Deterministic search limits. The area allowance is relative to the input,
/// never compounded across accepted moves. All timing limits use Liberty units.
#[derive(Clone, Debug)]
pub struct ResizeOptions {
    pub timing: StaOptions,
    pub max_area_growth: f64,
    pub max_upsize_rounds: usize,
    pub max_downsize_rounds: usize,
    pub max_candidate_sites: usize,
    pub max_cell_candidates: usize,
    pub max_trials: usize,
    pub max_propagated_instances: usize,
    pub resize_registers: bool,
    /// Approximate candidate screening only; commits and final verification
    /// always use accurate timing. Both thresholds default to zero (disabled).
    pub trial_tolerance: TrialTolerance,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            timing: StaOptions::default(),
            max_area_growth: 0.02,
            max_upsize_rounds: 4,
            max_downsize_rounds: 2,
            max_candidate_sites: 128,
            max_cell_candidates: 8,
            max_trials: 1024,
            max_propagated_instances: 100_000,
            resize_registers: true,
            trial_tolerance: TrialTolerance::default(),
        }
    }
}

/// Phase-level measurements exclude wall time so serialized results are stable.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ResizePhaseStats {
    pub trials: usize,
    pub accepted: usize,
    pub register_changes: usize,
    pub propagated_instances: usize,
    pub screened_out: usize,
    pub exact_trials: usize,
    pub prediction_rejections: usize,
}

/// Verified before/after QoR and bounded search work; original netlist is kept
/// unchanged if preparation or final independent timing verification fails.
#[derive(Clone, Debug, Serialize)]
pub struct ResizeReport {
    pub initial_area: f64,
    pub upsized_area: f64,
    pub final_area: f64,
    pub initial_delay: BoundaryDelays,
    pub upsized_delay: BoundaryDelays,
    pub final_delay: BoundaryDelays,
    pub upsize: ResizePhaseStats,
    pub downsize: ResizePhaseStats,
    pub budget_exhausted: bool,
}

/// Complete pin-compatible Boolean families, conservative about state/control
/// semantics. Missed equivalent functions are safe; false equivalences are not.
struct CellFamilies {
    alternatives: Vec<Vec<usize>>,
}

impl CellFamilies {
    /// Builds deterministic families once for the immutable library snapshot.
    fn new(library: &Library) -> Self {
        let mut groups: BTreeMap<Vec<String>, Vec<usize>> = BTreeMap::new();
        for (i, cell) in library.cells.iter().enumerate() {
            if let Some(signature) = cell_signature(cell, library) {
                groups.entry(signature).or_default().push(i);
            }
        }
        let mut alternatives = vec![Vec::new(); library.cells.len()];
        for mut group in groups.into_values() {
            group.sort_by(|a, b| {
                library.cells[*a]
                    .area
                    .total_cmp(&library.cells[*b].area)
                    .then(library.cells[*a].name.cmp(&library.cells[*b].name))
            });
            let selectable: Vec<_> = group
                .iter()
                .copied()
                .filter(|i| library.cells[*i].dont_use != Some(true))
                .collect();
            for i in group {
                alternatives[i] = selectable.clone();
            }
        }
        Self { alternatives }
    }
}

/// Enumerates bounded truth tables under identical pin names and state
/// metadata.
fn cell_signature(cell: &Cell, library: &Library) -> Option<Vec<String>> {
    if cell.validate_functional_semantics().is_err()
        || cell.clock_gate.is_some()
        || !cell.area.is_finite()
        || cell.area < 0.0
    {
        return None;
    }
    let mut pins: Vec<_> = cell.pins.iter().collect();
    pins.sort_by_key(|pin| library.resolve_string(&pin.name));
    let mut variables: Vec<_> = pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Input as i32)
        .map(|pin| library.resolve_string(&pin.name).to_string())
        .collect();
    for seq in &cell.sequential {
        variables.push(seq.state_var.clone());
        if let Some(name) = &seq.complementary_state_var {
            variables.push(name.clone());
        }
    }
    variables.sort();
    variables.dedup();
    // Covers ordinary cells and scan FFs without exponential library setup.
    if variables.len() > 10 {
        return None;
    }
    let mut signature = vec![format!("{:?}", cell.sequential)];
    let mut assignment: HashMap<String, bool> =
        variables.iter().cloned().map(|v| (v, false)).collect();
    for pin in pins {
        signature.push(format!(
            "{}:{}:{}",
            library.resolve_string(&pin.name),
            pin.direction,
            pin.is_clocking_pin
        ));
        if pin.direction != PinDirection::Output as i32 {
            continue;
        }
        let formula = parse_formula(library.resolve_string(&pin.function)).ok()?;
        let mut truth = String::with_capacity(1 << variables.len());
        for value in 0..1usize << variables.len() {
            for (bit, name) in variables.iter().enumerate() {
                assignment.insert(name.clone(), value & (1 << bit) != 0);
            }
            truth.push(if formula.evaluate_partial(&assignment)? {
                '1'
            } else {
                '0'
            });
        }
        signature.push(truth);
    }
    Some(signature)
}

/// Resizes compatible combinational cells and FFs without changing any wires.
/// Upsizing never worsens a boundary class; downsizing preserves the timing
/// checkpoint reached by upsizing. No hold/physical timing guarantee is made.
pub fn resize_netlist(
    module: &mut NetlistModule,
    nets: &[Net],
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &ResizeOptions,
) -> Result<ResizeReport> {
    if !options.max_area_growth.is_finite() || options.max_area_growth < 0.0 {
        return Err(anyhow!("max_area_growth must be finite and nonnegative"));
    }
    if [
        options.trial_tolerance.arrival,
        options.trial_tolerance.slew,
    ]
    .into_iter()
    .any(|v| !v.is_finite() || v < 0.0)
    {
        return Err(anyhow!("trial tolerances must be finite and nonnegative"));
    }
    if options.max_candidate_sites == 0 || options.max_cell_candidates == 0 {
        return Err(anyhow!("candidate limits must be positive"));
    }
    let families = CellFamilies::new(library);
    let mut sta = IncrementalSta::new(module, nets, interner, library, options.timing)?;
    let initial_delay = sta.score();
    let initial_area: f64 = sta.cells.iter().map(|i| library.cells[*i].area).sum();
    if !initial_area.is_finite() || sta.cells.iter().any(|i| library.cells[*i].area < 0.0) {
        return Err(anyhow!("cell areas must be finite and nonnegative"));
    }
    let area_limit = initial_area * (1.0 + options.max_area_growth);
    if !area_limit.is_finite() {
        return Err(anyhow!("area allowance overflows"));
    }
    let mut area = initial_area;
    let mut stats = [ResizePhaseStats::default(), ResizePhaseStats::default()];
    let mut upsized_delay = initial_delay;
    let mut upsized_area = initial_area;
    let objective = sta.objective();
    let mut trials = 0;
    let mut budget_exhausted = false;
    for (phase, phase_stats) in stats.iter_mut().enumerate() {
        let start_work = sta.propagated;
        // Reserve half the trials/work for the final area phase. The total
        // allowance remains shared; unused timing work is available to
        // recovery.
        let phase_trials = if phase == 0 && options.max_downsize_rounds > 0 {
            options.max_trials / 2
        } else {
            options.max_trials
        };
        let phase_work = if phase == 0 && options.max_downsize_rounds > 0 {
            options.max_propagated_instances / 2
        } else {
            options.max_propagated_instances
        };
        sta.budget_exhausted = false;
        let rounds = if phase == 0 {
            options.max_upsize_rounds
        } else {
            options.max_downsize_rounds
        };
        for _ in 0..rounds {
            let mut sites = if phase == 0 {
                sta.critical_instances(objective, options.max_candidate_sites)
            } else {
                let mut sites: Vec<_> = (0..sta.cells.len()).collect();
                sites.sort_by(|a, b| {
                    library.cells[sta.cells[*b]]
                        .area
                        .total_cmp(&library.cells[sta.cells[*a]].area)
                        .then(a.cmp(b))
                });
                sites
            };
            sites.retain(|i| options.resize_registers || !sta.is_register(*i));
            let mut accepted = 0;
            for i in sites {
                if trials >= phase_trials || sta.budget_exhausted {
                    break;
                }
                let current = sta.cells[i];
                let mut candidates = families.alternatives[current].clone();
                candidates.retain(|c| {
                    *c != current
                        && (phase == 0 || library.cells[*c].area < library.cells[current].area)
                });
                // Nearby strengths first in timing search; largest savings
                // first in recovery. Names break ties
                // independently of library order.
                if phase == 0 {
                    candidates.sort_by(|a, b| {
                        (library.cells[*a].area - library.cells[current].area)
                            .abs()
                            .total_cmp(
                                &(library.cells[*b].area - library.cells[current].area).abs(),
                            )
                            .then(library.cells[*a].name.cmp(&library.cells[*b].name))
                    });
                }
                for cell in candidates.into_iter().take(options.max_cell_candidates) {
                    if trials >= phase_trials || sta.budget_exhausted {
                        break;
                    }
                    let new_area =
                        area - library.cells[sta.cells[i]].area + library.cells[cell].area;
                    if new_area > area_limit + 1e-12 {
                        continue;
                    }
                    let old_score = sta.score();
                    let old_secondary = sta.secondary(objective);
                    trials += 1;
                    phase_stats.trials += 1;
                    let acceptable = |timing: &IncrementalSta<'_>| {
                        let score = timing.score();
                        if phase == 1 {
                            return score.within(upsized_delay);
                        }
                        score.within(initial_delay)
                            && score.within(old_score)
                            && (score.values()[objective] < old_score.values()[objective] - 1e-9
                                || (score.values()[objective] <= old_score.values()[objective]
                                    && timing.secondary(objective) < old_secondary - 1e-9))
                    };
                    if options.trial_tolerance.enabled() {
                        let mut promising = false;
                        sta.trial_with_tolerance(
                            i,
                            cell,
                            phase_work,
                            options.trial_tolerance,
                            |timing| {
                                promising = acceptable(timing);
                                false
                            },
                        )?;
                        if !promising || sta.budget_exhausted {
                            phase_stats.screened_out += 1;
                            continue;
                        }
                    }
                    phase_stats.exact_trials += 1;
                    let committed = sta.trial(i, cell, phase_work, acceptable)?;
                    if options.trial_tolerance.enabled() && !committed {
                        phase_stats.prediction_rejections += 1;
                    }
                    if committed {
                        phase_stats.accepted += 1;
                        phase_stats.register_changes += usize::from(sta.is_register(i));
                        accepted += 1;
                        area = new_area;
                        break;
                    }
                }
            }
            if accepted == 0 || trials >= phase_trials || sta.budget_exhausted {
                break;
            }
        }
        phase_stats.propagated_instances = sta.propagated - start_work;
        budget_exhausted |= sta.budget_exhausted || (rounds > 0 && trials >= phase_trials);
        if phase == 0 {
            upsized_delay = sta.score();
            upsized_area = area;
        }
    }
    let incremental_delay = sta.score();
    let cells = sta.cells.clone();
    drop(sta);
    let mut candidate = module.clone();
    for (instance, cell) in candidate.instances.iter_mut().zip(cells) {
        instance.type_name = interner.get_or_intern(&library.cells[cell].name);
    }
    let final_delay = measure_boundary_delays(&candidate, nets, interner, library, options.timing)?;
    verify_timing(incremental_delay, final_delay)?;
    if !final_delay.within(upsized_delay) {
        return Err(anyhow!(
            "final resizing violates the fixed timing checkpoint"
        ));
    }
    *module = candidate;
    let [upsize, downsize] = stats;
    Ok(ResizeReport {
        initial_area,
        upsized_area,
        final_area: area,
        initial_delay,
        upsized_delay,
        final_delay,
        upsize,
        downsize,
        budget_exhausted,
    })
}

/// Compares independent timing paths without turning rounding tolerance into a
/// cumulative optimization allowance.
fn verify_timing(incremental: BoundaryDelays, full: BoundaryDelays) -> Result<()> {
    for (a, b) in incremental.values().into_iter().zip(full.values()) {
        if (a - b).abs() > 1e-7 {
            return Err(anyhow!("incremental/full STA mismatch: {a} versus {b}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
