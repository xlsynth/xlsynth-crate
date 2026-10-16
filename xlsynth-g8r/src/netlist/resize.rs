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
use crate::liberty_model::{Cell, PinDirection, Sequential};
use crate::netlist::parse::{Net, NetlistModule};
use crate::netlist::sta::StaOptions;
pub use crate::netlist::sta::incremental::{BoundaryDelays, measure_boundary_delays};
use crate::netlist::sta::incremental::{
    IncrementalSta, STA_VERIFICATION_TOLERANCE, TIMING_COMPARISON_TOLERANCE, TimingTieBreak,
    TrialOutcome,
};

const AREA_COMPARISON_TOLERANCE: f64 = 1e-12;
const TIMING_PHASE_BUDGET_DIVISOR: usize = 3;
const MAX_SIGNATURE_VARIABLES: usize = 10;

/// Deterministic search limits. The area allowance is relative to the input,
/// never compounded across accepted moves. All timing limits use Liberty units.
#[derive(Clone, Debug)]
pub struct ResizeOptions {
    pub timing: StaOptions,
    pub max_area_growth: f64,
    /// Timing-search rounds; substitutions may increase or decrease cell area.
    pub max_upsize_rounds: usize,
    /// Area-recovery rounds against the timing-search checkpoint.
    pub max_downsize_rounds: usize,
    /// Selectable sites per timing-search round; recovery uses the work limits.
    pub max_candidate_sites: usize,
    pub max_cell_candidates: usize,
    pub max_trials: usize,
    pub max_propagated_instances: usize,
    pub resize_registers: bool,
}

/// Why a phase stopped. Work limits are cumulative, but the timing phase may
/// stop at its reserved share while leaving work available for area recovery.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResizeStopReason {
    #[default]
    Disabled,
    NoImprovement,
    RoundLimit,
    TrialLimit,
    PropagationLimit,
}

impl ResizeStopReason {
    fn is_work_limit(self) -> bool {
        matches!(self, Self::TrialLimit | Self::PropagationLimit)
    }
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            timing: StaOptions::default(),
            max_area_growth: 0.02,
            max_upsize_rounds: 4,
            max_downsize_rounds: 2,
            max_candidate_sites: 64,
            max_cell_candidates: 8,
            max_trials: 1536,
            max_propagated_instances: 150_000,
            resize_registers: true,
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
    pub stop_reason: ResizeStopReason,
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
    /// A phase stopped on work and reached an overall trial/propagation limit.
    /// A timing-phase reservation alone does not exhaust the overall budget.
    pub budget_exhausted: bool,
}

/// Exact sequential metadata, deliberately not Boolean-normalized: equivalent
/// syntax may miss a family, but incompatible state/control semantics never
/// join.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SequentialSignature {
    state_var: String,
    next_state: String,
    clock_expr: String,
    kind: i32,
    clear_expr: String,
    preset_expr: String,
    complementary_state_var: Option<String>,
}

impl From<&Sequential> for SequentialSignature {
    fn from(sequential: &Sequential) -> Self {
        // Exhaustive destructuring makes new sequential metadata require an
        // explicit compatibility decision here, rather than silently ignoring
        // it.
        let Sequential {
            state_var,
            next_state,
            clock_expr,
            kind,
            clear_expr,
            preset_expr,
            complementary_state_var,
        } = sequential;
        Self {
            state_var: state_var.clone(),
            next_state: next_state.clone(),
            clock_expr: clock_expr.clone(),
            kind: *kind,
            clear_expr: clear_expr.clone(),
            preset_expr: preset_expr.clone(),
            complementary_state_var: complementary_state_var.clone(),
        }
    }
}

/// One name-compatible pin, with output truth bits packed in assignment order.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PinSignature {
    name: String,
    direction: i32,
    clocking: bool,
    truth: Option<Vec<u64>>,
}

/// Complete compatibility key independent of debug formatting and pin order.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CellSignature {
    sequential: Vec<SequentialSignature>,
    variables: Vec<String>,
    pins: Vec<PinSignature>,
}

/// Complete pin-compatible Boolean families, conservative about state/control
/// semantics. Missed equivalent functions are safe; false equivalences are not.
struct CellFamilies {
    family_by_cell: Vec<Option<usize>>,
    alternatives: Vec<Vec<usize>>,
}

impl CellFamilies {
    /// Builds deterministic families once for the immutable library snapshot.
    fn new(library: &Library) -> Self {
        let mut groups: BTreeMap<CellSignature, Vec<usize>> = BTreeMap::new();
        for (i, cell) in library.cells.iter().enumerate() {
            if let Some(signature) = cell_signature(cell, library) {
                groups.entry(signature).or_default().push(i);
            }
        }
        let mut family_by_cell = vec![None; library.cells.len()];
        let mut alternatives = Vec::with_capacity(groups.len());
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
                family_by_cell[i] = Some(alternatives.len());
            }
            alternatives.push(selectable);
        }
        Self {
            family_by_cell,
            alternatives,
        }
    }

    /// Returns a shared area/name-sorted list, excluding `dont_use`
    /// alternatives.
    fn alternatives(&self, cell: usize) -> &[usize] {
        self.family_by_cell[cell].map_or(&[], |family| &self.alternatives[family])
    }
}

/// Enumerates bounded truth tables under identical pin names and state
/// metadata.
fn cell_signature(cell: &Cell, library: &Library) -> Option<CellSignature> {
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
    if variables.len() > MAX_SIGNATURE_VARIABLES {
        return None;
    }
    let mut signature = Vec::with_capacity(pins.len());
    let mut assignment: HashMap<String, bool> =
        variables.iter().cloned().map(|v| (v, false)).collect();
    for pin in pins {
        let truth = if pin.direction == PinDirection::Output as i32 {
            let formula = parse_formula(library.resolve_string(&pin.function)).ok()?;
            let assignments = 1usize << variables.len();
            let mut truth = vec![0u64; assignments.div_ceil(u64::BITS as usize)];
            for value in 0..assignments {
                for (bit, name) in variables.iter().enumerate() {
                    *assignment
                        .get_mut(name)
                        .expect("signature variable binding") = value & (1 << bit) != 0;
                }
                if formula.evaluate_partial(&assignment)? {
                    truth[value / u64::BITS as usize] |= 1 << (value % u64::BITS as usize);
                }
            }
            Some(truth)
        } else {
            None
        };
        signature.push(PinSignature {
            name: library.resolve_string(&pin.name).to_owned(),
            direction: pin.direction,
            clocking: pin.is_clocking_pin,
            truth,
        });
    }
    Some(CellSignature {
        sequential: cell
            .sequential
            .iter()
            .map(SequentialSignature::from)
            .collect(),
        variables,
        pins: signature,
    })
}

/// The two search policies share transactional trials but have distinct
/// candidate ordering, acceptance rules, and timing checkpoints.
#[derive(Clone, Copy)]
enum SearchPhase {
    Timing,
    AreaRecovery,
}

/// Cumulative work ceilings; unused timing work remains available to recovery.
struct WorkLimits {
    trials: usize,
    propagated_instances: usize,
}

impl SearchPhase {
    fn rounds(self, options: &ResizeOptions) -> usize {
        match self {
            Self::Timing => options.max_upsize_rounds,
            Self::AreaRecovery => options.max_downsize_rounds,
        }
    }

    fn work_limits(self, options: &ResizeOptions) -> WorkLimits {
        // Reserve two thirds for recovery only when that phase is enabled.
        let divisor = match self {
            Self::Timing if options.max_downsize_rounds > 0 => TIMING_PHASE_BUDGET_DIVISOR,
            _ => 1,
        };
        WorkLimits {
            trials: options.max_trials / divisor,
            propagated_instances: options.max_propagated_instances / divisor,
        }
    }
}

/// Search policy and bookkeeping, independent of the STA transaction state.
struct ResizeSearch<'a> {
    library: &'a Library,
    options: &'a ResizeOptions,
    families: CellFamilies,
    objective: usize,
    initial_delay: BoundaryDelays,
    timing_checkpoint: BoundaryDelays,
    area: f64,
    area_limit: f64,
    trials: usize,
}

impl ResizeSearch<'_> {
    /// Selects timing sites or ranks recoverable area once per unchanged round.
    fn candidate_sites(&self, phase: SearchPhase, sta: &IncrementalSta<'_>) -> Vec<usize> {
        let mut sites = match phase {
            SearchPhase::Timing => {
                sta.critical_instances(self.objective, self.options.max_candidate_sites, |i| {
                    let current = sta.cells[i];
                    (self.options.resize_registers || !sta.is_register(i))
                        && self.families.alternatives(current).iter().any(|c| {
                            *c != current
                                && self.area - self.library.cells[current].area
                                    + self.library.cells[*c].area
                                    <= self.area_limit + AREA_COMPARISON_TOLERANCE
                        })
                })
            }
            SearchPhase::AreaRecovery => {
                let priorities = sta
                    .recovery_headroom(self.timing_checkpoint)
                    .into_iter()
                    .zip(&sta.cells)
                    .map(|(headroom, cell)| {
                        let area = self.library.cells[*cell].area;
                        let savings = self
                            .families
                            .alternatives(*cell)
                            .first()
                            .map_or(0.0, |smallest| area - self.library.cells[*smallest].area);
                        if savings > 0.0 {
                            headroom * savings
                        } else {
                            -1.0
                        }
                    })
                    .collect::<Vec<_>>();
                // Zero-headroom candidates remain eligible: a cheaper cell
                // can have identical or better timing. No cheaper cell means
                // there is no useful recovery work at this site.
                let mut sites: Vec<_> = (0..sta.cells.len())
                    .filter(|i| priorities[*i] >= 0.0)
                    .collect();
                sites.sort_by(|a, b| {
                    priorities[*b]
                        .total_cmp(&priorities[*a])
                        .then_with(|| {
                            self.library.cells[sta.cells[*b]]
                                .area
                                .total_cmp(&self.library.cells[sta.cells[*a]].area)
                        })
                        .then(a.cmp(b))
                });
                sites
            }
        };
        sites.retain(|i| self.options.resize_registers || !sta.is_register(*i));
        sites
    }

    /// Applies the phase's acceptance policy to an exact, rollback-safe trial.
    fn trial(
        &self,
        phase: SearchPhase,
        sta: &mut IncrementalSta<'_>,
        instance: usize,
        cell: usize,
        work_limit: usize,
        new_area: f64,
    ) -> Result<TrialOutcome> {
        match phase {
            SearchPhase::Timing => {
                let previous = sta.score();
                let secondary = sta.secondary(self.objective);
                sta.trial(instance, cell, work_limit, Some(previous), |timing| {
                    self.accept_timing(timing, previous, secondary, new_area)
                })
            }
            SearchPhase::AreaRecovery => sta.trial(
                instance,
                cell,
                work_limit,
                Some(self.timing_checkpoint),
                |timing| timing.score().within(self.timing_checkpoint),
            ),
        }
    }

    /// Prices area growth by area-delay-squared merit; tied worst paths use
    /// near-critical progress. Hard limits retain all four boundary classes.
    fn accept_timing(
        &self,
        timing: &IncrementalSta<'_>,
        previous: BoundaryDelays,
        previous_secondary: TimingTieBreak,
        new_area: f64,
    ) -> bool {
        let score = timing.score();
        let secondary = timing.secondary(self.objective);
        let old_delay = previous.values()[self.objective];
        let new_delay = score.values()[self.objective];
        let delay_ratio = if old_delay <= 0.0 {
            // Negative characterized delays are supported by STA, but cannot
            // define a positive merit ratio that justifies additional area.
            1.0
        } else if new_delay < old_delay - TIMING_COMPARISON_TOLERANCE {
            new_delay / old_delay
        } else {
            secondary.relative_to(previous_secondary)
        };
        let worthwhile = new_area <= self.area || delay_ratio < (self.area / new_area).sqrt();
        score.within(self.initial_delay)
            && score.within(previous)
            && worthwhile
            && (new_delay < old_delay - TIMING_COMPARISON_TOLERANCE
                || (new_delay <= old_delay && secondary.improves(previous_secondary)))
    }

    /// Runs one bounded phase without changing candidate or tie-breaking order.
    fn run_phase(
        &mut self,
        phase: SearchPhase,
        sta: &mut IncrementalSta<'_>,
    ) -> Result<ResizePhaseStats> {
        let mut stats = ResizePhaseStats::default();
        let rounds = phase.rounds(self.options);
        if rounds == 0 {
            return Ok(stats);
        }
        stats.stop_reason = ResizeStopReason::RoundLimit;
        let limits = phase.work_limits(self.options);
        let start_work = sta.propagated_instances();
        let mut timing_candidates = Vec::new();
        'rounds: for _ in 0..rounds {
            let sites = self.candidate_sites(phase, sta);
            let mut accepted = 0;
            for i in sites {
                if self.trials >= limits.trials {
                    stats.stop_reason = ResizeStopReason::TrialLimit;
                    break 'rounds;
                }
                let current = sta.cells[i];
                let area = self.area;
                let area_limit = self.area_limit;
                let library = self.library;
                let allowed = |cell: &usize| {
                    *cell != current
                        && match phase {
                            SearchPhase::Timing => true,
                            SearchPhase::AreaRecovery => {
                                library.cells[*cell].area < library.cells[current].area
                            }
                        }
                        && area - library.cells[current].area + library.cells[*cell].area
                            <= area_limit + AREA_COMPARISON_TOLERANCE
                };
                let alternatives = self.families.alternatives(current);
                let candidates = match phase {
                    SearchPhase::Timing => {
                        timing_candidates.clear();
                        timing_candidates.extend(alternatives.iter().copied().filter(allowed));
                        timing_candidates.sort_by(|a, b| {
                            (library.cells[*a].area - library.cells[current].area)
                                .abs()
                                .total_cmp(
                                    &(library.cells[*b].area - library.cells[current].area).abs(),
                                )
                                .then(library.cells[*a].name.cmp(&library.cells[*b].name))
                        });
                        timing_candidates.as_slice()
                    }
                    // Families are already ordered by largest savings first.
                    SearchPhase::AreaRecovery => alternatives,
                };
                for cell in candidates
                    .iter()
                    .copied()
                    .filter(allowed)
                    .take(self.options.max_cell_candidates)
                {
                    if self.trials >= limits.trials {
                        stats.stop_reason = ResizeStopReason::TrialLimit;
                        break 'rounds;
                    }
                    let new_area =
                        self.area - library.cells[sta.cells[i]].area + library.cells[cell].area;
                    self.trials += 1;
                    stats.trials += 1;
                    match self.trial(phase, sta, i, cell, limits.propagated_instances, new_area)? {
                        TrialOutcome::Accepted => {
                            stats.accepted += 1;
                            stats.register_changes += usize::from(sta.is_register(i));
                            accepted += 1;
                            self.area = new_area;
                            break;
                        }
                        TrialOutcome::Rejected => {
                            // The transaction restored the incumbent; try the
                            // next strength.
                        }
                        TrialOutcome::PropagationLimit => {
                            stats.stop_reason = ResizeStopReason::PropagationLimit;
                            break 'rounds;
                        }
                    }
                }
            }
            if self.trials >= limits.trials {
                stats.stop_reason = ResizeStopReason::TrialLimit;
                break;
            }
            if accepted == 0 {
                stats.stop_reason = ResizeStopReason::NoImprovement;
                break;
            }
        }
        stats.propagated_instances = sta.propagated_instances() - start_work;
        Ok(stats)
    }
}

/// Resizes compatible combinational cells and FFs without changing any wires.
/// The timing phase prioritizes near-critical paths and prices area-growing
/// moves by an area-delay-squared merit, without worsening any boundary class.
/// Recovery orders candidates by potential area savings and estimated timing
/// headroom, but accepts them only against the exact timing checkpoint. No
/// hold/physical timing guarantee is made.
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
    let mut search = ResizeSearch {
        library,
        options,
        families,
        objective: sta.objective(),
        initial_delay,
        timing_checkpoint: initial_delay,
        area: initial_area,
        area_limit,
        trials: 0,
    };
    let upsize = search.run_phase(SearchPhase::Timing, &mut sta)?;
    let upsized_delay = sta.score();
    search.timing_checkpoint = upsized_delay;
    // Re-anchor at the phase boundary to avoid incremental cancellation error.
    let upsized_area = sta.cells.iter().map(|i| library.cells[*i].area).sum();
    search.area = upsized_area;
    let downsize = search.run_phase(SearchPhase::AreaRecovery, &mut sta)?;
    let budget_exhausted = (upsize.stop_reason.is_work_limit()
        || downsize.stop_reason.is_work_limit())
        && (search.trials >= options.max_trials
            || sta.propagated_instances() >= options.max_propagated_instances);
    let incremental_delay = sta.score();
    let final_area: f64 = sta.cells.iter().map(|i| library.cells[*i].area).sum();
    if !final_area.is_finite() || final_area > area_limit + AREA_COMPARISON_TOLERANCE {
        return Err(anyhow!("final resizing exceeds the area allowance"));
    }
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
    Ok(ResizeReport {
        initial_area,
        upsized_area,
        final_area,
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
        if (a - b).abs() > STA_VERIFICATION_TOLERANCE {
            return Err(anyhow!("incremental/full STA mismatch: {a} versus {b}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
