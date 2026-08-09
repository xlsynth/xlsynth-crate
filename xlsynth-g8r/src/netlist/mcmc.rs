// SPDX-License-Identifier: Apache-2.0

//! Function-preserving MCMC over mapped covers, cell sizes, and buffer trees.

mod remap;

use crate::liberty_model::{Library, PinDirection};
use crate::netlist::cell_catalog::{CatalogCell, CellCatalog};
use crate::netlist::normalized::{BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistInstance, NetlistModule, PortDirection};
use crate::netlist::report::{
    NetlistReport, build_area_report, build_netlist_report_with_primary_input_arrivals,
};
use crate::netlist::sta::{ScopedTimingTableEnvelopeCache, StaOptions};
use crate::netlist::timing_buffer::BufferTimingConstraints;
use crate::netlist::timing_resize::{SearchIncrementalSta, SearchTimingScore};
use anyhow::{Context, Result, anyhow, bail};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use remap::{RemapLibrary, RemapRequest, RemapShape, propose_equivalent_remap};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};
use xlsynth_mcmc::multichain::{
    ChainRole, ChainStrategy, SegmentOutcome, SegmentRunParams, SegmentRunner, run_multichain,
};
use xlsynth_mcmc::{MIN_TEMPERATURE_RATIO, McmcStats, metropolis_accept};

const OBJECTIVE_EPSILON: f64 = 1e-9;
const SECONDARY_OBJECTIVE_WEIGHT: f64 = 1e-6;
const DEFAULT_CRITICAL_PATHS: usize = 16;
const CRITICAL_REFRESH_INTERVAL: u64 = 16;
const MAX_REMAP_TRUTH_INPUTS: usize = 6;

/// Whether mapped-netlist exploration prioritizes delay or constrained area.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetlistMcmcObjective {
    Delay,
    Area,
}

/// Search bounds and exact timing assumptions for mapped-netlist exploration.
#[derive(Clone, Debug, PartialEq)]
pub struct NetlistMcmcOptions {
    pub objective: NetlistMcmcObjective,
    pub iterations: u64,
    /// Optional wall-clock search budget; overrides the iteration ceiling.
    pub time_limit_seconds: Option<u64>,
    pub threads: usize,
    pub seed: u64,
    pub initial_temperature: f64,
    pub checkpoint_iterations: u64,
    pub sta_options: StaOptions,
    pub timing_constraints: BufferTimingConstraints,
    /// Absolute delay ceiling for area recovery; defaults to initial delay.
    pub delay_limit: Option<f64>,
    /// Maximum relative area growth permitted during delay optimization.
    pub max_area_growth: Option<f64>,
    pub enable_sizing: bool,
    pub enable_pin_swaps: bool,
    pub enable_buffer_moves: bool,
    /// Enables truth-table-proven alternate combinational cell coverings.
    pub enable_remap: bool,
    /// Maximum distinct external data inputs of one remapped logic cone.
    pub max_remap_leaves: usize,
    /// Exact incremental sizing trials used to settle each remapped cone.
    pub remap_relax_evaluations: usize,
    pub buffer_primary_inputs: bool,
    pub max_buffer_fanout: usize,
    pub critical_window: f64,
}

impl Default for NetlistMcmcOptions {
    fn default() -> Self {
        Self {
            objective: NetlistMcmcObjective::Delay,
            iterations: 1_000,
            time_limit_seconds: None,
            threads: 1,
            seed: 0,
            initial_temperature: 0.02,
            checkpoint_iterations: 128,
            sta_options: StaOptions::default(),
            timing_constraints: BufferTimingConstraints::default(),
            delay_limit: None,
            max_area_growth: None,
            enable_sizing: true,
            enable_pin_swaps: true,
            enable_buffer_moves: true,
            enable_remap: true,
            max_remap_leaves: MAX_REMAP_TRUTH_INPUTS,
            remap_relax_evaluations: 16,
            buffer_primary_inputs: false,
            max_buffer_fanout: 12,
            critical_window: 0.10,
        }
    }
}

/// Function-preserving mapped-netlist proposal families.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NetlistMcmcMoveKind {
    ResizeCell,
    SwapInputPins,
    ResizeConnectedCells,
    InsertBuffer,
    RemoveBuffer,
    MoveBufferSink,
    SplitBuffer,
    MergeBuffers,
    ReparentBuffer,
    InsertBufferAndResizeDriver,
    RemoveBufferAndResizeDriver,
    CollapseCone,
    ExpandCone,
    RemapCone,
}

impl NetlistMcmcMoveKind {
    /// Provides stable machine-readable move names without depending on Debug.
    fn as_str(self) -> &'static str {
        match self {
            Self::ResizeCell => "resize_cell",
            Self::SwapInputPins => "swap_input_pins",
            Self::ResizeConnectedCells => "resize_connected_cells",
            Self::InsertBuffer => "insert_buffer",
            Self::RemoveBuffer => "remove_buffer",
            Self::MoveBufferSink => "move_buffer_sink",
            Self::SplitBuffer => "split_buffer",
            Self::MergeBuffers => "merge_buffers",
            Self::ReparentBuffer => "reparent_buffer",
            Self::InsertBufferAndResizeDriver => "insert_buffer_and_resize_driver",
            Self::RemoveBufferAndResizeDriver => "remove_buffer_and_resize_driver",
            Self::CollapseCone => "collapse_cone",
            Self::ExpandCone => "expand_cone",
            Self::RemapCone => "remap_cone",
        }
    }
}

/// One accepted move on the returned best state's deterministic search path.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NetlistMcmcMove {
    pub kind: NetlistMcmcMoveKind,
    pub instances: Vec<String>,
    pub area_before: f64,
    pub area_after: f64,
    pub delay_before: f64,
    pub delay_after: f64,
}

/// Exact before/after QoR and search diagnostics for a mapped MCMC run.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct NetlistMcmcStats {
    pub initial_area: f64,
    pub initial_delay: f64,
    pub final_area: f64,
    pub final_delay: f64,
    pub iterations: u64,
    pub elapsed_seconds: f64,
    pub threads: usize,
    pub attempted_moves: usize,
    pub accepted_moves: usize,
    pub rejected_candidates: usize,
    pub rejected_timing: usize,
    pub rejected_metropolis: usize,
    pub incremental_timing_evaluations: usize,
    pub complete_timing_evaluations: usize,
    pub recomputed_instances: usize,
    pub accepted_edits_by_kind: BTreeMap<String, usize>,
    pub best_path: Vec<NetlistMcmcMove>,
}

#[derive(Clone)]
struct SearchState {
    module: NetlistModule,
    nets: Vec<Net>,
    interner: StringInterner<StringBackend<SymbolU32>>,
    history: Vec<NetlistMcmcMove>,
}

#[derive(Clone, Copy, Debug)]
struct SearchCost {
    area: f64,
    delay: f64,
    energy: f64,
}

#[derive(Default)]
struct SearchCounters {
    attempts: AtomicUsize,
    accepted: AtomicUsize,
    candidate_failures: AtomicUsize,
    timing_failures: AtomicUsize,
    metropolis_rejections: AtomicUsize,
    incremental_evaluations: AtomicUsize,
    complete_evaluations: AtomicUsize,
    recomputed_instances: AtomicUsize,
    accepted_by_kind: Mutex<BTreeMap<String, usize>>,
}

struct SearchRunner {
    library: Arc<Library>,
    remap_library: Option<Arc<RemapLibrary>>,
    options: NetlistMcmcOptions,
    baseline_area: f64,
    baseline_delay: f64,
    baseline_electrical_violations: usize,
    started_at: Instant,
    deadline: Option<Instant>,
    counters: Arc<SearchCounters>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PinLocation {
    instance_index: usize,
    connection_index: usize,
}

#[derive(Clone, Debug, Default)]
struct NetFanout {
    driver: Option<PinLocation>,
    sinks: Vec<PinLocation>,
    primary_input: bool,
    primary_output: bool,
    protected_clock: bool,
    protected_assign: bool,
}

#[derive(Clone, Copy, Debug)]
struct BufferInstance {
    instance_index: usize,
    input_bit: usize,
    output_bit: usize,
    input_location: PinLocation,
}

/// One normalized, single-output combinational Liberty implementation.
#[derive(Clone, Debug)]
struct LogicInstance {
    cell_index: usize,
    input_bits: Vec<usize>,
    output_bit: usize,
}

struct Connectivity {
    fanouts: Vec<NetFanout>,
    buffers: Vec<BufferInstance>,
    logic: Vec<Option<LogicInstance>>,
}

enum Proposal<'a> {
    Resize {
        instance_index: usize,
        replacement_index: usize,
        score: SearchTimingScore,
        area: f64,
    },
    PinSwap {
        instance_index: usize,
        first_input: usize,
        second_input: usize,
        score: SearchTimingScore,
    },
    Topology {
        state: SearchState,
        timing: Box<SearchIncrementalSta<'a>>,
        cost: SearchCost,
        instances: Vec<String>,
    },
}

/// Explores equivalent cell assignments and buffer trees around a mapped
/// netlist.
pub fn optimize_mapped_netlist_mcmc(
    module: &mut NetlistModule,
    nets: &mut Vec<Net>,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: Arc<Library>,
    options: &NetlistMcmcOptions,
) -> Result<NetlistMcmcStats> {
    validate_options(options)?;
    if (options.enable_buffer_moves || options.enable_remap)
        && module.net_index_range.end != nets.len()
    {
        bail!(
            "mapped MCMC topology changes require the selected module to own the end of the net table"
        );
    }

    let initial_report = build_netlist_report_with_primary_input_arrivals(
        module,
        nets.as_slice(),
        interner,
        library.as_ref(),
        options.sta_options,
        &options.timing_constraints.primary_input_arrivals,
    )
    .context("computing initial exact mapped-netlist MCMC timing")?;
    let initial_delay = report_objective_delay(&initial_report);
    let initial_area = initial_report.cell_area;
    let initial_timing = SearchIncrementalSta::new(
        module,
        nets.as_slice(),
        interner,
        library.as_ref(),
        options.sta_options,
        &options.timing_constraints,
    )
    .context("building initial register-aware MCMC timing graph")?;
    if !initial_timing.score().constraints_satisfied {
        bail!("initial mapped netlist does not satisfy the requested timing constraints");
    }
    let initial_electrical_violations = initial_timing.electrical_violations();
    drop(initial_timing);

    let remap_library = if options.enable_remap {
        let catalog = CellCatalog::new(library.as_ref())?;
        Some(Arc::new(RemapLibrary::new(
            &catalog,
            options.max_remap_leaves,
        )))
    } else {
        None
    };

    let started_at = Instant::now();
    let deadline = options
        .time_limit_seconds
        .map(|seconds| started_at + Duration::from_secs(seconds));
    let counters = Arc::new(SearchCounters::default());
    let runner = Arc::new(SearchRunner {
        library: library.clone(),
        remap_library,
        options: options.clone(),
        baseline_area: initial_area,
        baseline_delay: initial_delay,
        baseline_electrical_violations: initial_electrical_violations,
        started_at,
        deadline,
        counters: counters.clone(),
    });
    let start_state = SearchState {
        module: module.clone(),
        nets: nets.clone(),
        interner: interner.clone(),
        history: Vec::new(),
    };

    if options.iterations == 0 && options.time_limit_seconds.is_none() {
        return Ok(NetlistMcmcStats {
            initial_area,
            final_area: initial_area,
            initial_delay,
            final_delay: initial_delay,
            iterations: 0,
            threads: options.threads,
            ..NetlistMcmcStats::default()
        });
    }

    let planned_iterations = if options.time_limit_seconds.is_some() {
        u64::MAX / 2
    } else {
        options.iterations
    };
    let (best, _, _) = run_multichain(
        start_state,
        planned_iterations,
        options.seed,
        options.threads,
        if options.threads > 1 {
            ChainStrategy::ExploreExploit
        } else {
            ChainStrategy::Independent
        },
        options.checkpoint_iterations,
        runner,
        |cost: &SearchCost| cost.energy.to_bits(),
        search_state_tiebreak,
        |local: &SearchCost, global: &SearchCost| local.energy > global.energy + 0.01,
        |state: &SearchState, _, _| state.clone(),
    )?;

    *module = best.module;
    *nets = best.nets;
    *interner = best.interner;
    let final_report = build_netlist_report_with_primary_input_arrivals(
        module,
        nets.as_slice(),
        interner,
        library.as_ref(),
        options.sta_options,
        &options.timing_constraints.primary_input_arrivals,
    )
    .context("independently verifying final mapped-netlist MCMC area and timing")?;

    let accepted_edits_by_kind = counters
        .accepted_by_kind
        .lock()
        .map_err(|_| anyhow!("mapped MCMC move statistics were poisoned"))?
        .clone();
    Ok(NetlistMcmcStats {
        initial_area,
        initial_delay,
        final_area: final_report.cell_area,
        final_delay: report_objective_delay(&final_report),
        iterations: options.iterations,
        elapsed_seconds: started_at.elapsed().as_secs_f64(),
        threads: options.threads,
        attempted_moves: counters.attempts.load(Ordering::Relaxed),
        accepted_moves: counters.accepted.load(Ordering::Relaxed),
        rejected_candidates: counters.candidate_failures.load(Ordering::Relaxed),
        rejected_timing: counters.timing_failures.load(Ordering::Relaxed),
        rejected_metropolis: counters.metropolis_rejections.load(Ordering::Relaxed),
        incremental_timing_evaluations: counters.incremental_evaluations.load(Ordering::Relaxed),
        complete_timing_evaluations: counters.complete_evaluations.load(Ordering::Relaxed),
        recomputed_instances: counters.recomputed_instances.load(Ordering::Relaxed),
        accepted_edits_by_kind,
        best_path: best.history,
    })
}

impl SegmentRunner<SearchState, SearchCost, NetlistMcmcMoveKind> for SearchRunner {
    type Error = anyhow::Error;

    /// Stops all chains without discarding their current best mapped netlists.
    fn should_stop(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Runs one deterministic annealed chain segment against exact Liberty STA.
    fn run_segment(
        &self,
        start_state: SearchState,
        params: SegmentRunParams,
    ) -> Result<SegmentOutcome<SearchState, SearchCost, NetlistMcmcMoveKind>> {
        let _timing_envelopes = ScopedTimingTableEnvelopeCache::new(self.library.as_ref());
        let catalog = CellCatalog::new(self.library.as_ref())?;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(params.seed);
        let mut state = start_state;
        let mut timing = self.build_timing(&state)?;
        let area = build_area_report(&state.module, &state.interner, self.library.as_ref())?.area;
        let mut cost = self.cost(area, timing.score());
        let mut best_state = state.clone();
        let mut best_cost = cost;
        let mut stats = McmcStats::default();
        let mut critical = timing
            .critical_instances(DEFAULT_CRITICAL_PATHS, self.options.critical_window)
            .unwrap_or_default();

        for local_iteration in 0..params.segment_iters {
            if self.should_stop() {
                break;
            }
            let iteration = params.iter_offset + local_iteration;
            if local_iteration > 0 && local_iteration % CRITICAL_REFRESH_INTERVAL == 0 {
                critical = timing
                    .critical_instances(DEFAULT_CRITICAL_PATHS, self.options.critical_window)
                    .unwrap_or_default();
            }
            let Some(kind) = self.pick_move(&mut rng) else {
                continue;
            };
            self.counters.attempts.fetch_add(1, Ordering::Relaxed);
            let proposal = match self.propose(
                kind,
                &state,
                &mut timing,
                &catalog,
                critical.as_slice(),
                cost,
                &mut rng,
            ) {
                Ok(Some(proposal)) => proposal,
                Ok(None) | Err(_) => {
                    stats.rejected_candidate_fail += 1;
                    self.counters
                        .candidate_failures
                        .fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };

            let candidate_cost = match &proposal {
                Proposal::Resize { score, area, .. } => self.cost(*area, *score),
                Proposal::PinSwap { score, .. } => self.cost(cost.area, *score),
                Proposal::Topology { cost, .. } => *cost,
            };
            if !self.feasible(candidate_cost) {
                stats.rejected_apply_fail += 1;
                self.counters
                    .timing_failures
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let progress = if let Some(seconds) = self.options.time_limit_seconds {
                self.started_at.elapsed().as_secs_f64() / seconds as f64
            } else {
                iteration as f64 / params.total_iters.max(1) as f64
            };
            let role_scale = if params.role == ChainRole::Explorer {
                2.0
            } else {
                1.0
            };
            let temperature = self.options.initial_temperature
                * role_scale
                * (1.0 - progress).max(MIN_TEMPERATURE_RATIO);
            if !metropolis_accept(cost.energy, candidate_cost.energy, temperature, &mut rng) {
                stats.rejected_metro += 1;
                self.counters
                    .metropolis_rejections
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let instances = self.commit(proposal, &mut state, &mut timing)?;
            state.history.push(NetlistMcmcMove {
                kind,
                instances,
                area_before: cost.area,
                area_after: candidate_cost.area,
                delay_before: cost.delay,
                delay_after: candidate_cost.delay,
            });
            cost = candidate_cost;
            stats.accepted_overall += 1;
            *stats.accepted_edits_by_kind.entry(kind).or_insert(0) += 1;
            self.counters.accepted.fetch_add(1, Ordering::Relaxed);
            *self
                .counters
                .accepted_by_kind
                .lock()
                .map_err(|_| anyhow!("mapped MCMC move statistics were poisoned"))?
                .entry(kind.as_str().to_string())
                .or_insert(0) += 1;

            if cost.energy + OBJECTIVE_EPSILON < best_cost.energy {
                best_state = state.clone();
                best_cost = cost;
            }
            if !matches!(
                kind,
                NetlistMcmcMoveKind::ResizeCell | NetlistMcmcMoveKind::SwapInputPins
            ) {
                critical = timing
                    .critical_instances(DEFAULT_CRITICAL_PATHS, self.options.critical_window)
                    .unwrap_or_default();
            }
        }

        Ok(SegmentOutcome {
            end_state: state,
            end_cost: cost,
            best_state,
            best_cost,
            stats,
        })
    }
}

impl SearchRunner {
    /// Rebuilds exact sequential/combinational timing after a topology change.
    fn build_timing<'a>(&'a self, state: &SearchState) -> Result<SearchIncrementalSta<'a>> {
        self.counters
            .complete_evaluations
            .fetch_add(1, Ordering::Relaxed);
        SearchIncrementalSta::new(
            &state.module,
            state.nets.as_slice(),
            &state.interner,
            self.library.as_ref(),
            self.options.sta_options,
            &self.options.timing_constraints,
        )
    }

    /// Normalizes primary and secondary QoR metrics to the original netlist.
    fn cost(&self, area: f64, timing: SearchTimingScore) -> SearchCost {
        let delay = timing.objective_delay();
        let area_ratio = area / self.baseline_area.max(OBJECTIVE_EPSILON);
        let delay_ratio = delay / self.baseline_delay.max(OBJECTIVE_EPSILON);
        let energy = match self.options.objective {
            NetlistMcmcObjective::Delay => delay_ratio + SECONDARY_OBJECTIVE_WEIGHT * area_ratio,
            NetlistMcmcObjective::Area => area_ratio + SECONDARY_OBJECTIVE_WEIGHT * delay_ratio,
        };
        SearchCost {
            area,
            delay,
            energy,
        }
    }

    /// Enforces the selected timing ceiling or optional area-growth budget.
    fn feasible(&self, cost: SearchCost) -> bool {
        if !cost.area.is_finite() || !cost.delay.is_finite() || !cost.energy.is_finite() {
            return false;
        }
        match self.options.objective {
            NetlistMcmcObjective::Area => {
                cost.delay
                    <= self.options.delay_limit.unwrap_or(self.baseline_delay) + OBJECTIVE_EPSILON
            }
            NetlistMcmcObjective::Delay => self.options.max_area_growth.is_none_or(|growth| {
                cost.area <= self.baseline_area * (1.0 + growth) + OBJECTIVE_EPSILON
            }),
        }
    }

    /// Mixes cheap incremental edits with less frequent buffer-tree changes.
    fn pick_move(&self, rng: &mut Xoshiro256PlusPlus) -> Option<NetlistMcmcMoveKind> {
        let mut weighted = Vec::<(NetlistMcmcMoveKind, usize)>::new();
        if self.options.enable_sizing {
            weighted.push((NetlistMcmcMoveKind::ResizeCell, 36));
            weighted.push((NetlistMcmcMoveKind::ResizeConnectedCells, 12));
        }
        if self.options.enable_pin_swaps {
            weighted.push((NetlistMcmcMoveKind::SwapInputPins, 8));
        }
        if self.options.enable_buffer_moves {
            let area = self.options.objective == NetlistMcmcObjective::Area;
            weighted.push((NetlistMcmcMoveKind::InsertBuffer, if area { 4 } else { 12 }));
            weighted.push((NetlistMcmcMoveKind::RemoveBuffer, if area { 16 } else { 6 }));
            weighted.push((NetlistMcmcMoveKind::MoveBufferSink, 10));
            weighted.push((NetlistMcmcMoveKind::SplitBuffer, if area { 3 } else { 6 }));
            weighted.push((NetlistMcmcMoveKind::MergeBuffers, if area { 10 } else { 4 }));
            weighted.push((NetlistMcmcMoveKind::ReparentBuffer, 4));
            if self.options.enable_sizing {
                weighted.push((NetlistMcmcMoveKind::InsertBufferAndResizeDriver, 5));
                weighted.push((NetlistMcmcMoveKind::RemoveBufferAndResizeDriver, 5));
            }
        }
        if self.options.enable_remap {
            let area = self.options.objective == NetlistMcmcObjective::Area;
            weighted.push((
                NetlistMcmcMoveKind::CollapseCone,
                if area { 16 } else { 10 },
            ));
            weighted.push((NetlistMcmcMoveKind::ExpandCone, if area { 4 } else { 8 }));
            weighted.push((NetlistMcmcMoveKind::RemapCone, 10));
        }
        let total = weighted.iter().map(|(_, weight)| *weight).sum::<usize>();
        if total == 0 {
            return None;
        }
        let mut selection = rng.gen_range(0..total);
        for (kind, weight) in weighted {
            if selection < weight {
                return Some(kind);
            }
            selection -= weight;
        }
        None
    }

    /// Proposes one reversible incremental or independently timed graph edit.
    #[allow(clippy::too_many_arguments)]
    fn propose<'a>(
        &'a self,
        kind: NetlistMcmcMoveKind,
        state: &SearchState,
        timing: &mut SearchIncrementalSta<'a>,
        catalog: &CellCatalog,
        critical: &[usize],
        current: SearchCost,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<Option<Proposal<'a>>> {
        match kind {
            NetlistMcmcMoveKind::ResizeCell => {
                let Some(instance) = choose_instance(state, timing, critical, rng) else {
                    return Ok(None);
                };
                let alternatives = timing.size_alternatives(instance);
                let Some(&replacement) = alternatives.choose(rng) else {
                    return Ok(None);
                };
                self.counters
                    .incremental_evaluations
                    .fetch_add(1, Ordering::Relaxed);
                let score = timing.evaluate_resize(instance, replacement, false)?;
                self.counters
                    .recomputed_instances
                    .fetch_add(score.recomputed_instances, Ordering::Relaxed);
                if !score.constraints_satisfied {
                    return Ok(None);
                }
                let old = timing
                    .current_cell_index(instance)
                    .ok_or_else(|| anyhow!("MCMC resize instance is missing"))?;
                let area = current.area - self.library.cells[old].area
                    + self.library.cells[replacement].area;
                Ok(Some(Proposal::Resize {
                    instance_index: instance,
                    replacement_index: replacement,
                    score,
                    area,
                }))
            }
            NetlistMcmcMoveKind::SwapInputPins => {
                let Some(instance) = choose_instance(state, timing, critical, rng) else {
                    return Ok(None);
                };
                let pairs = timing.symmetric_input_pairs(instance);
                let Some(&(first, second)) = pairs.choose(rng) else {
                    return Ok(None);
                };
                self.counters
                    .incremental_evaluations
                    .fetch_add(1, Ordering::Relaxed);
                let score = timing.evaluate_pin_swap(instance, first, second, false)?;
                self.counters
                    .recomputed_instances
                    .fetch_add(score.recomputed_instances, Ordering::Relaxed);
                if !score.constraints_satisfied {
                    return Ok(None);
                }
                Ok(Some(Proposal::PinSwap {
                    instance_index: instance,
                    first_input: first,
                    second_input: second,
                    score,
                }))
            }
            _ => self.propose_topology(kind, state, timing, catalog, critical, rng),
        }
    }

    /// Builds and exactly validates one coupled sizing, mapping, or buffer
    /// trial.
    fn propose_topology<'a>(
        &'a self,
        kind: NetlistMcmcMoveKind,
        current: &SearchState,
        current_timing: &SearchIncrementalSta<'a>,
        catalog: &CellCatalog,
        critical: &[usize],
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<Option<Proposal<'a>>> {
        let connectivity = build_connectivity(current, self.library.as_ref(), catalog)?;
        let mut candidate = current.clone();
        let instances = match kind {
            NetlistMcmcMoveKind::ResizeConnectedCells => resize_connected_cells(
                &mut candidate,
                current_timing,
                self.library.as_ref(),
                critical,
                rng,
            ),
            NetlistMcmcMoveKind::InsertBuffer => insert_buffer(
                &mut candidate,
                &connectivity,
                catalog,
                self.library.as_ref(),
                critical,
                &self.options,
                false,
                current_timing,
                rng,
            ),
            NetlistMcmcMoveKind::InsertBufferAndResizeDriver => insert_buffer(
                &mut candidate,
                &connectivity,
                catalog,
                self.library.as_ref(),
                critical,
                &self.options,
                true,
                current_timing,
                rng,
            ),
            NetlistMcmcMoveKind::RemoveBuffer => remove_buffer(
                &mut candidate,
                &connectivity,
                current_timing,
                self.library.as_ref(),
                false,
                rng,
            ),
            NetlistMcmcMoveKind::RemoveBufferAndResizeDriver => remove_buffer(
                &mut candidate,
                &connectivity,
                current_timing,
                self.library.as_ref(),
                true,
                rng,
            ),
            NetlistMcmcMoveKind::MoveBufferSink => {
                move_buffer_sink(&mut candidate, &connectivity, rng)
            }
            NetlistMcmcMoveKind::SplitBuffer => split_buffer(
                &mut candidate,
                &connectivity,
                catalog,
                self.library.as_ref(),
                critical,
                &self.options,
                rng,
            ),
            NetlistMcmcMoveKind::MergeBuffers => merge_buffers(&mut candidate, &connectivity, rng),
            NetlistMcmcMoveKind::ReparentBuffer => {
                reparent_buffer(&mut candidate, &connectivity, rng)
            }
            NetlistMcmcMoveKind::CollapseCone
            | NetlistMcmcMoveKind::ExpandCone
            | NetlistMcmcMoveKind::RemapCone => {
                let Some(remap_library) = self.remap_library.as_ref() else {
                    return Ok(None);
                };
                let shape = match kind {
                    NetlistMcmcMoveKind::CollapseCone => RemapShape::Collapse,
                    NetlistMcmcMoveKind::ExpandCone => RemapShape::Expand,
                    NetlistMcmcMoveKind::RemapCone => RemapShape::Recover,
                    _ => unreachable!("only remapping moves reach remap dispatch"),
                };
                propose_equivalent_remap(
                    &mut candidate,
                    &connectivity,
                    catalog,
                    self.library.as_ref(),
                    remap_library,
                    current_timing,
                    RemapRequest {
                        shape,
                        objective: self.options.objective,
                        max_leaves: self.options.max_remap_leaves,
                        critical,
                    },
                    rng,
                )
            }
            NetlistMcmcMoveKind::ResizeCell | NetlistMcmcMoveKind::SwapInputPins => {
                bail!("incremental proposal reached topology search")
            }
        }?;
        let Some(mut instances) = instances else {
            return Ok(None);
        };
        let candidate_connectivity =
            build_connectivity(&candidate, self.library.as_ref(), catalog)?;
        if candidate_connectivity
            .fanouts
            .iter()
            .enumerate()
            .any(|(bit, fanout)| {
                fanout.sinks.len() > self.options.max_buffer_fanout
                    && fanout.sinks.len()
                        > connectivity
                            .fanouts
                            .get(bit)
                            .map_or(0, |previous| previous.sinks.len())
            })
        {
            self.counters
                .timing_failures
                .fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let mut candidate_timing = self.build_timing(&candidate)?;
        let mut area = build_area_report(
            &candidate.module,
            &candidate.interner,
            self.library.as_ref(),
        )?
        .area;
        if matches!(
            kind,
            NetlistMcmcMoveKind::CollapseCone
                | NetlistMcmcMoveKind::ExpandCone
                | NetlistMcmcMoveKind::RemapCone
        ) && self.options.enable_sizing
            && self.options.remap_relax_evaluations > 0
        {
            self.relax_remapped_neighborhood(
                &mut candidate,
                &mut candidate_timing,
                &mut instances,
                &mut area,
                rng,
            )?;
        }
        let score = candidate_timing.score();
        if !score.constraints_satisfied
            || candidate_timing.electrical_violations() > self.baseline_electrical_violations
        {
            self.counters
                .timing_failures
                .fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        Ok(Some(Proposal::Topology {
            state: candidate,
            timing: Box::new(candidate_timing),
            cost: self.cost(area, score),
            instances,
        }))
    }

    /// Repairs a changed cover and its immediate electrical neighborhood.
    fn relax_remapped_neighborhood(
        &self,
        state: &mut SearchState,
        timing: &mut SearchIncrementalSta<'_>,
        instances: &mut Vec<String>,
        area: &mut f64,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<()> {
        let mut focus = BTreeSet::new();
        for name in instances.iter() {
            let Some(index) = state.module.instances.iter().position(|instance| {
                state.interner.resolve(instance.instance_name) == Some(name.as_str())
            }) else {
                continue;
            };
            focus.insert(index);
            focus.extend(timing.neighboring_instances(index));
        }
        if focus.is_empty() {
            return Ok(());
        }

        let mut focus = focus.into_iter().collect::<Vec<_>>();
        let mut remaining = self.options.remap_relax_evaluations;
        while remaining > 0 && !self.should_stop() {
            focus.shuffle(rng);
            let current_score = timing.score();
            let current_rank =
                self.relaxation_energy(self.cost(*area, current_score), current_score);
            let mut best = None;
            let mut trials_this_round = 0;
            for &instance in &focus {
                if remaining == 0 || trials_this_round >= 8 || self.should_stop() {
                    break;
                }
                let Some(previous) = timing.current_cell_index(instance) else {
                    continue;
                };
                let mut alternatives = timing.size_alternatives(instance);
                alternatives.shuffle(rng);
                alternatives.sort_by(|lhs, rhs| {
                    let comparison = self.library.cells[*lhs]
                        .area
                        .total_cmp(&self.library.cells[*rhs].area);
                    if self.options.objective == NetlistMcmcObjective::Area {
                        comparison
                    } else {
                        comparison.reverse()
                    }
                });
                for replacement in alternatives.into_iter().take(2) {
                    if remaining == 0 || trials_this_round >= 8 || self.should_stop() {
                        break;
                    }
                    remaining -= 1;
                    trials_this_round += 1;
                    self.counters
                        .incremental_evaluations
                        .fetch_add(1, Ordering::Relaxed);
                    let Ok(score) = timing.evaluate_resize(instance, replacement, false) else {
                        continue;
                    };
                    self.counters
                        .recomputed_instances
                        .fetch_add(score.recomputed_instances, Ordering::Relaxed);
                    let trial_area = *area - self.library.cells[previous].area
                        + self.library.cells[replacement].area;
                    let rank = self.relaxation_energy(self.cost(trial_area, score), score);
                    if rank + OBJECTIVE_EPSILON < current_rank
                        && best.as_ref().is_none_or(
                            |(_, _, _, best_rank): &(usize, usize, f64, f64)| {
                                rank + OBJECTIVE_EPSILON < *best_rank
                            },
                        )
                    {
                        best = Some((instance, replacement, trial_area, rank));
                    }
                }
            }
            let Some((instance, replacement, next_area, _)) = best else {
                break;
            };
            let score = timing.evaluate_resize(instance, replacement, true)?;
            self.counters
                .recomputed_instances
                .fetch_add(score.recomputed_instances, Ordering::Relaxed);
            state.module.instances[instance].type_name = state
                .interner
                .get_or_intern(self.library.cells[replacement].name.as_str());
            let name = instance_name(state, instance)?;
            if !instances.contains(&name) {
                instances.push(name);
            }
            *area = next_area;
        }
        Ok(())
    }

    /// Guides local repair toward feasibility before comparing final QoR.
    fn relaxation_energy(&self, cost: SearchCost, score: SearchTimingScore) -> f64 {
        let mut energy = cost.energy;
        if self.options.objective == NetlistMcmcObjective::Area {
            let limit = self.options.delay_limit.unwrap_or(self.baseline_delay);
            energy +=
                8.0 * (cost.delay - limit).max(0.0) / self.baseline_delay.max(OBJECTIVE_EPSILON);
        }
        if let Some(growth) = self.options.max_area_growth {
            let limit = self.baseline_area * (1.0 + growth);
            energy +=
                8.0 * (cost.area - limit).max(0.0) / self.baseline_area.max(OBJECTIVE_EPSILON);
        }
        if !score.constraints_satisfied {
            energy += 8.0;
        }
        energy
    }

    /// Commits a validated proposal while keeping raw and incremental state
    /// synchronized.
    fn commit<'a>(
        &'a self,
        proposal: Proposal<'a>,
        state: &mut SearchState,
        timing: &mut SearchIncrementalSta<'a>,
    ) -> Result<Vec<String>> {
        match proposal {
            Proposal::Resize {
                instance_index,
                replacement_index,
                ..
            } => {
                let name = instance_name(state, instance_index)?;
                let committed = timing.evaluate_resize(instance_index, replacement_index, true)?;
                self.counters
                    .recomputed_instances
                    .fetch_add(committed.recomputed_instances, Ordering::Relaxed);
                state.module.instances[instance_index].type_name = state
                    .interner
                    .get_or_intern(self.library.cells[replacement_index].name.as_str());
                Ok(vec![name])
            }
            Proposal::PinSwap {
                instance_index,
                first_input,
                second_input,
                ..
            } => {
                let name = instance_name(state, instance_index)?;
                let first = timing
                    .input_pin_name(instance_index, first_input)
                    .ok_or_else(|| anyhow!("first MCMC swap pin is missing"))?
                    .to_string();
                let second = timing
                    .input_pin_name(instance_index, second_input)
                    .ok_or_else(|| anyhow!("second MCMC swap pin is missing"))?
                    .to_string();
                let committed =
                    timing.evaluate_pin_swap(instance_index, first_input, second_input, true)?;
                self.counters
                    .recomputed_instances
                    .fetch_add(committed.recomputed_instances, Ordering::Relaxed);
                swap_instance_connections(state, instance_index, first.as_str(), second.as_str())?;
                Ok(vec![name])
            }
            Proposal::Topology {
                state: candidate,
                timing: candidate_timing,
                instances,
                ..
            } => {
                *state = candidate;
                *timing = *candidate_timing;
                Ok(instances)
            }
        }
    }
}

/// Validates finite objective parameters before allocating search chains.
fn validate_options(options: &NetlistMcmcOptions) -> Result<()> {
    if options.threads == 0 || options.checkpoint_iterations == 0 || options.max_buffer_fanout == 0
    {
        bail!("mapped MCMC threads, checkpoint iterations, and buffer fanout must be positive");
    }
    if options.time_limit_seconds == Some(0) {
        bail!("mapped MCMC wall-clock limit must be positive");
    }
    if !options.initial_temperature.is_finite() || options.initial_temperature <= 0.0 {
        bail!("mapped MCMC initial temperature must be finite and positive");
    }
    if !options.critical_window.is_finite() || !(0.0..=1.0).contains(&options.critical_window) {
        bail!("mapped MCMC critical window must be between zero and one");
    }
    if options.enable_remap && !(2..=MAX_REMAP_TRUTH_INPUTS).contains(&options.max_remap_leaves) {
        bail!("mapped MCMC remapping requires between two and six boundary inputs");
    }
    if options
        .delay_limit
        .is_some_and(|limit| !limit.is_finite() || limit < 0.0)
        || options
            .max_area_growth
            .is_some_and(|growth| !growth.is_finite() || growth < 0.0)
    {
        bail!("mapped MCMC delay and area limits must be finite and nonnegative");
    }
    if !options.sta_options.primary_input_transition.is_finite()
        || options.sta_options.primary_input_transition < 0.0
        || !options.sta_options.module_output_load.is_finite()
        || options.sta_options.module_output_load < 0.0
    {
        bail!("mapped MCMC STA assumptions must be finite and nonnegative");
    }
    Ok(())
}

/// Selects register-to-register timing when physical capture paths exist.
fn report_objective_delay(report: &NetlistReport) -> f64 {
    report
        .max_register_to_register_delay
        .or(report.max_input_to_register_delay)
        .or(report.max_register_to_output_delay)
        .or(report.max_delay)
        .unwrap_or(0.0)
}

/// Produces a stable lightweight tie-breaker for parallel chain reduction.
fn search_state_tiebreak(state: &SearchState) -> String {
    state
        .module
        .instances
        .iter()
        .map(|instance| {
            let name = state.interner.resolve(instance.instance_name).unwrap_or("");
            let cell = state.interner.resolve(instance.type_name).unwrap_or("");
            format!("{name}:{cell}")
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// Prefers actual critical cells while retaining unbiased whole-design
/// coverage.
fn choose_instance(
    state: &SearchState,
    timing: &SearchIncrementalSta<'_>,
    critical: &[usize],
    rng: &mut Xoshiro256PlusPlus,
) -> Option<usize> {
    if state.module.instances.is_empty() {
        return None;
    }
    if !critical.is_empty() && rng.gen_bool(0.7) {
        let candidates = critical
            .iter()
            .copied()
            .filter(|index| timing.current_cell_index(*index).is_some())
            .collect::<Vec<_>>();
        if let Some(index) = candidates.choose(rng) {
            return Some(*index);
        }
    }
    Some(rng.gen_range(0..state.module.instances.len()))
}

/// Normalizes actual Liberty pin directions and protects clock/external wiring.
fn build_connectivity(
    state: &SearchState,
    library: &Library,
    catalog: &CellCatalog,
) -> Result<Connectivity> {
    let normalized =
        NormalizedNetlistModule::new(&state.module, state.nets.as_slice(), &state.interner)?;
    let mut fanouts = vec![NetFanout::default(); normalized.bit_count()];
    for port in &normalized.ports {
        for bit in &port.bits {
            match port.direction {
                PortDirection::Input => fanouts[*bit].primary_input = true,
                PortDirection::Output => fanouts[*bit].primary_output = true,
                PortDirection::Inout => {
                    fanouts[*bit].protected_assign = true;
                }
            }
        }
    }
    for assign in &normalized.assigns {
        for bit in &assign.lhs_bits {
            fanouts[*bit].protected_assign = true;
        }
        for expression in &assign.rhs_bits {
            let mut bits = Vec::new();
            expression.collect_source_bits(&mut bits);
            for bit in bits {
                fanouts[bit].protected_assign = true;
            }
        }
    }

    let cells = library
        .cells
        .iter()
        .map(|cell| (cell.name.as_str(), cell))
        .collect::<HashMap<_, _>>();
    let mut buffers = Vec::new();
    let mut logic = vec![None; state.module.instances.len()];
    for normalized_instance in &normalized.instances {
        let index = normalized_instance.raw_index.0;
        let name = state
            .interner
            .resolve(normalized_instance.type_name)
            .ok_or_else(|| anyhow!("cannot resolve mapped MCMC cell type"))?;
        let cell = cells
            .get(name)
            .ok_or_else(|| anyhow!("mapped MCMC references unknown Liberty cell '{name}'"))?;
        let mut buffer_input = None;
        let mut buffer_input_location = None;
        let mut buffer_output = None;
        for (connection_index, connection) in normalized_instance.connections.iter().enumerate() {
            if connection.bits.len() != 1 {
                continue;
            }
            let BitSource::Bit(bit) = connection.bits[0] else {
                continue;
            };
            let pin_name = state
                .interner
                .resolve(connection.port)
                .ok_or_else(|| anyhow!("cannot resolve mapped MCMC instance pin"))?;
            let pin = cell
                .pins
                .iter()
                .find(|pin| library.resolve_string(&pin.name) == pin_name)
                .ok_or_else(|| anyhow!("cell '{name}' has no pin '{pin_name}'"))?;
            let location = PinLocation {
                instance_index: index,
                connection_index,
            };
            if pin.direction == PinDirection::Input as i32 {
                fanouts[bit].sinks.push(location);
                if pin.is_clocking_pin {
                    fanouts[bit].protected_clock = true;
                } else {
                    buffer_input = Some(bit);
                    buffer_input_location = Some(location);
                }
            } else if pin.direction == PinDirection::Output as i32 {
                if fanouts[bit].driver.replace(location).is_some() {
                    bail!("mapped MCMC encountered a multiply driven data net");
                }
                buffer_output = Some(bit);
            }
        }
        let catalog_cell = catalog.by_name(name);
        if let Some(classified) = catalog_cell {
            let input_bits = classified
                .family
                .input_names
                .iter()
                .map(|input_name| {
                    normalized_instance
                        .connections
                        .iter()
                        .find(|connection| {
                            state.interner.resolve(connection.port) == Some(input_name.as_str())
                        })
                        .and_then(|connection| match connection.bits.as_slice() {
                            [BitSource::Bit(bit)] => Some(*bit),
                            _ => None,
                        })
                })
                .collect::<Option<Vec<_>>>();
            let output_bit = normalized_instance
                .connections
                .iter()
                .find(|connection| {
                    state.interner.resolve(connection.port)
                        == Some(classified.family.output_name.as_str())
                })
                .and_then(|connection| match connection.bits.as_slice() {
                    [BitSource::Bit(bit)] => Some(*bit),
                    _ => None,
                });
            if let (Some(input_bits), Some(output_bit)) = (input_bits, output_bit) {
                logic[index] = Some(LogicInstance {
                    cell_index: classified.cell_index,
                    input_bits,
                    output_bit,
                });
            }
        }
        if catalog_cell.is_some_and(CatalogCell::is_buffer)
            && let (Some(input_bit), Some(output_bit), Some(input_location)) =
                (buffer_input, buffer_output, buffer_input_location)
            && input_bit != output_bit
        {
            buffers.push(BufferInstance {
                instance_index: index,
                input_bit,
                output_bit,
                input_location,
            });
        }
    }
    Ok(Connectivity {
        fanouts,
        buffers,
        logic,
    })
}

/// Applies several exact-function size changes as one jointly timed proposal.
fn resize_connected_cells(
    state: &mut SearchState,
    timing: &SearchIncrementalSta<'_>,
    library: &Library,
    critical: &[usize],
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Option<Vec<String>>> {
    let Some(anchor) = choose_instance(state, timing, critical, rng) else {
        return Ok(None);
    };
    let mut selected = vec![anchor];
    let mut neighbors = timing.neighboring_instances(anchor);
    neighbors.shuffle(rng);
    selected.extend(neighbors.into_iter().take(rng.gen_range(1..=3)));
    if selected.len() < 2 {
        return Ok(None);
    }
    let mut changed = Vec::new();
    for index in selected {
        let alternatives = timing.size_alternatives(index);
        let Some(&replacement) = alternatives.choose(rng) else {
            continue;
        };
        changed.push(instance_name(state, index)?);
        state.module.instances[index].type_name = state
            .interner
            .get_or_intern(library.cells[replacement].name.as_str());
    }
    Ok((changed.len() >= 2).then_some(changed))
}

/// Isolates selected noncritical sinks behind one exact-identity Liberty cell.
#[allow(clippy::too_many_arguments)]
fn insert_buffer(
    state: &mut SearchState,
    connectivity: &Connectivity,
    catalog: &CellCatalog,
    library: &Library,
    critical: &[usize],
    options: &NetlistMcmcOptions,
    resize_driver: bool,
    timing: &SearchIncrementalSta<'_>,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Option<Vec<String>>> {
    let candidates = connectivity
        .fanouts
        .iter()
        .enumerate()
        .filter(|(_, fanout)| {
            !fanout.protected_clock
                && fanout.sinks.len() >= 2
                && (fanout.driver.is_some()
                    || fanout.primary_input && options.buffer_primary_inputs)
        })
        .collect::<Vec<_>>();
    let critical = critical.iter().copied().collect::<BTreeSet<_>>();
    let near_critical = candidates
        .iter()
        .copied()
        .filter(|(_, fanout)| {
            fanout
                .driver
                .is_some_and(|driver| critical.contains(&driver.instance_index))
                || fanout
                    .sinks
                    .iter()
                    .any(|sink| critical.contains(&sink.instance_index))
        })
        .collect::<Vec<_>>();
    let selected = if !near_critical.is_empty() && rng.gen_bool(0.75) {
        near_critical.choose(rng).copied()
    } else {
        candidates.choose(rng).copied()
    };
    let Some((root, fanout)) = selected else {
        return Ok(None);
    };
    let mut sinks = fanout.sinks.clone();
    sinks.sort_by_key(|sink| {
        (
            !critical.contains(&sink.instance_index),
            sink.instance_index,
            sink.connection_index,
        )
    });
    if sinks.len() < 2 {
        return Ok(None);
    }
    let mut moved = sinks[1..].to_vec();
    moved.shuffle(rng);
    let count = rng.gen_range(1..=moved.len().min(options.max_buffer_fanout));
    moved.truncate(count);
    let Some(buffer) = choose_buffer(catalog, rng) else {
        return Ok(None);
    };
    let source = bit_reference(state, connectivity, root)?;
    let output = append_wire(state);
    for sink in moved {
        state.module.instances[sink.instance_index].connections[sink.connection_index].1 =
            NetRef::Simple(output);
    }
    let index = append_buffer(state, library, buffer, source, NetRef::Simple(output))?;
    let mut names = vec![instance_name(state, index)?];
    if resize_driver {
        let Some(driver) = fanout.driver else {
            return Ok(None);
        };
        let alternatives = timing.size_alternatives(driver.instance_index);
        let Some(&replacement) = alternatives.choose(rng) else {
            return Ok(None);
        };
        names.push(instance_name(state, driver.instance_index)?);
        state.module.instances[driver.instance_index].type_name = state
            .interner
            .get_or_intern(library.cells[replacement].name.as_str());
    }
    Ok(Some(names))
}

/// Bypasses a true identity buffer without touching visible or assigned nets.
fn remove_buffer(
    state: &mut SearchState,
    connectivity: &Connectivity,
    timing: &SearchIncrementalSta<'_>,
    library: &Library,
    resize_driver: bool,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Option<Vec<String>>> {
    let candidates = connectivity
        .buffers
        .iter()
        .filter(|buffer| buffer_output_is_internal(connectivity, **buffer))
        .collect::<Vec<_>>();
    let Some(&buffer) = candidates.choose(rng) else {
        return Ok(None);
    };
    let mut names = vec![instance_name(state, buffer.instance_index)?];
    if resize_driver {
        let Some(driver) = connectivity.fanouts[buffer.input_bit].driver else {
            return Ok(None);
        };
        let alternatives = timing.size_alternatives(driver.instance_index);
        let Some(&replacement) = alternatives.choose(rng) else {
            return Ok(None);
        };
        names.push(instance_name(state, driver.instance_index)?);
        state.module.instances[driver.instance_index].type_name = state
            .interner
            .get_or_intern(library.cells[replacement].name.as_str());
    }
    let input = bit_reference(state, connectivity, buffer.input_bit)?;
    for sink in &connectivity.fanouts[buffer.output_bit].sinks {
        state.module.instances[sink.instance_index].connections[sink.connection_index].1 =
            input.clone();
    }
    state.module.instances.remove(buffer.instance_index);
    Ok(Some(names))
}

/// Moves a data sink between an existing buffer and its immediate parent.
fn move_buffer_sink(
    state: &mut SearchState,
    connectivity: &Connectivity,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Option<Vec<String>>> {
    let candidates = connectivity
        .buffers
        .iter()
        .filter(|buffer| {
            !connectivity.fanouts[buffer.input_bit].protected_clock
                && !connectivity.fanouts[buffer.output_bit].protected_clock
        })
        .collect::<Vec<_>>();
    let Some(&buffer) = candidates.choose(rng) else {
        return Ok(None);
    };
    let parent = &connectivity.fanouts[buffer.input_bit];
    let child = &connectivity.fanouts[buffer.output_bit];
    let move_out = rng.gen_bool(0.5);
    let (sinks, destination) = if move_out && child.sinks.len() >= 2 {
        (&child.sinks, buffer.input_bit)
    } else {
        (&parent.sinks, buffer.output_bit)
    };
    let candidates = sinks
        .iter()
        .copied()
        .filter(|sink| sink.instance_index != buffer.instance_index)
        .collect::<Vec<_>>();
    let Some(&sink) = candidates.choose(rng) else {
        return Ok(None);
    };
    let name = instance_name(state, sink.instance_index)?;
    let reference = bit_reference(state, connectivity, destination)?;
    state.module.instances[sink.instance_index].connections[sink.connection_index].1 = reference;
    Ok(Some(vec![name]))
}

/// Splits one buffer's sink group into two sibling identity-buffer branches.
#[allow(clippy::too_many_arguments)]
fn split_buffer(
    state: &mut SearchState,
    connectivity: &Connectivity,
    catalog: &CellCatalog,
    library: &Library,
    critical: &[usize],
    options: &NetlistMcmcOptions,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Option<Vec<String>>> {
    let candidates = connectivity
        .buffers
        .iter()
        .filter(|buffer| {
            !connectivity.fanouts[buffer.input_bit].protected_clock
                && connectivity.fanouts[buffer.output_bit].sinks.len() >= 2
        })
        .collect::<Vec<_>>();
    let Some(&original) = candidates.choose(rng) else {
        return Ok(None);
    };
    let Some(replacement) = choose_buffer(catalog, rng) else {
        return Ok(None);
    };
    let critical = critical.iter().copied().collect::<BTreeSet<_>>();
    let mut sinks = connectivity.fanouts[original.output_bit].sinks.clone();
    sinks.sort_by_key(|sink| {
        (
            !critical.contains(&sink.instance_index),
            sink.instance_index,
        )
    });
    let mut selected = sinks[1..].to_vec();
    selected.shuffle(rng);
    selected.truncate(rng.gen_range(1..=selected.len().min(options.max_buffer_fanout)));
    let input = bit_reference(state, connectivity, original.input_bit)?;
    let output = append_wire(state);
    for sink in selected {
        state.module.instances[sink.instance_index].connections[sink.connection_index].1 =
            NetRef::Simple(output);
    }
    let index = append_buffer(state, library, replacement, input, NetRef::Simple(output))?;
    Ok(Some(vec![
        instance_name(state, original.instance_index)?,
        instance_name(state, index)?,
    ]))
}

/// Merges two identity buffers driven by the same exact parent signal.
fn merge_buffers(
    state: &mut SearchState,
    connectivity: &Connectivity,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Option<Vec<String>>> {
    let pairs = sibling_buffer_pairs(connectivity)
        .into_iter()
        .filter(|(_, remove)| buffer_output_is_internal(connectivity, *remove))
        .collect::<Vec<_>>();
    let Some(&(keep, remove)) = pairs.choose(rng) else {
        return Ok(None);
    };
    let names = vec![
        instance_name(state, keep.instance_index)?,
        instance_name(state, remove.instance_index)?,
    ];
    let destination = bit_reference(state, connectivity, keep.output_bit)?;
    for sink in &connectivity.fanouts[remove.output_bit].sinks {
        state.module.instances[sink.instance_index].connections[sink.connection_index].1 =
            destination.clone();
    }
    state.module.instances.remove(remove.instance_index);
    Ok(Some(names))
}

/// Promotes a child buffer or demotes one sibling beneath another sibling.
fn reparent_buffer(
    state: &mut SearchState,
    connectivity: &Connectivity,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<Option<Vec<String>>> {
    if rng.gen_bool(0.5) {
        let nested = connectivity
            .buffers
            .iter()
            .filter_map(|child| {
                connectivity
                    .buffers
                    .iter()
                    .find(|parent| parent.output_bit == child.input_bit)
                    .map(|parent| (*child, *parent))
            })
            .collect::<Vec<_>>();
        if let Some(&(child, parent)) = nested.choose(rng) {
            let location = child.input_location;
            let reference = bit_reference(state, connectivity, parent.input_bit)?;
            state.module.instances[location.instance_index].connections
                [location.connection_index]
                .1 = reference;
            return Ok(Some(vec![instance_name(state, child.instance_index)?]));
        }
    }
    let pairs = sibling_buffer_pairs(connectivity);
    let Some(&(parent, child)) = pairs.choose(rng) else {
        return Ok(None);
    };
    let location = child.input_location;
    let reference = bit_reference(state, connectivity, parent.output_bit)?;
    state.module.instances[location.instance_index].connections[location.connection_index].1 =
        reference;
    Ok(Some(vec![instance_name(state, child.instance_index)?]))
}

/// Lists deterministic same-parent buffer pairs independently of instance
/// names.
fn sibling_buffer_pairs(connectivity: &Connectivity) -> Vec<(BufferInstance, BufferInstance)> {
    let mut pairs = Vec::new();
    for (index, first) in connectivity.buffers.iter().copied().enumerate() {
        if connectivity.fanouts[first.input_bit].protected_clock {
            continue;
        }
        for second in connectivity.buffers.iter().copied().skip(index + 1) {
            if first.input_bit == second.input_bit {
                pairs.push((first, second));
                pairs.push((second, first));
            }
        }
    }
    pairs
}

/// Prevents bypassing module outputs, aliases, or protected clock signals.
fn buffer_output_is_internal(connectivity: &Connectivity, buffer: BufferInstance) -> bool {
    let output = &connectivity.fanouts[buffer.output_bit];
    !output.primary_output && !output.protected_assign && !output.protected_clock
}

/// Selects any characterized exact-identity cell, including imported buffers.
fn choose_buffer<'a>(
    catalog: &'a CellCatalog,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<&'a CatalogCell> {
    let cells = catalog.buffers().collect::<Vec<_>>();
    cells.choose(rng).copied()
}

/// Resolves a data bit through its actual existing driver or sink connection.
fn bit_reference(state: &SearchState, connectivity: &Connectivity, bit: usize) -> Result<NetRef> {
    let fanout = connectivity
        .fanouts
        .get(bit)
        .ok_or_else(|| anyhow!("mapped MCMC references an out-of-range data bit"))?;
    let location = fanout
        .driver
        .or_else(|| fanout.sinks.first().copied())
        .ok_or_else(|| anyhow!("mapped MCMC data bit has no usable connection"))?;
    Ok(
        state.module.instances[location.instance_index].connections[location.connection_index]
            .1
            .clone(),
    )
}

/// Appends a collision-free scalar internal wire without flattening module
/// ports.
fn append_wire(state: &mut SearchState) -> NetIndex {
    for suffix in 0usize.. {
        let name = format!("n_mcmc_buf_{suffix}");
        if state.interner.get(name.as_str()).is_some() {
            continue;
        }
        let index = NetIndex(state.nets.len());
        state.nets.push(Net {
            name: state.interner.get_or_intern(name.as_str()),
            width: None,
        });
        state.module.wires.push(index);
        state.module.net_index_range.end = state.nets.len();
        return index;
    }
    unreachable!("a collision-free mapped MCMC wire name must exist")
}

/// Emits one real Liberty identity buffer with deterministically ordered pins.
fn append_buffer(
    state: &mut SearchState,
    library: &Library,
    buffer: &CatalogCell,
    input: NetRef,
    output: NetRef,
) -> Result<usize> {
    let instance_name = (0usize..)
        .map(|suffix| format!("u_mcmc_buf_{suffix}"))
        .find(|name| state.interner.get(name.as_str()).is_none())
        .ok_or_else(|| anyhow!("cannot allocate mapped MCMC buffer instance"))?;
    let cell = &library.cells[buffer.cell_index];
    let input_name = library.resolve_string(&cell.pins[buffer.input_pin_indices[0]].name);
    let output_name = library.resolve_string(&cell.pins[buffer.output_pin_index].name);
    let mut connections = vec![
        (state.interner.get_or_intern(input_name), input),
        (state.interner.get_or_intern(output_name), output),
    ];
    connections.sort_by_key(|(pin, _)| state.interner.resolve(*pin).unwrap_or("").to_string());
    let index = state.module.instances.len();
    state.module.instances.push(NetlistInstance {
        type_name: state.interner.get_or_intern(buffer.name.as_str()),
        instance_name: state.interner.get_or_intern(instance_name.as_str()),
        connections,
        inst_lineno: 1,
        inst_colno: 1,
    });
    Ok(index)
}

/// Exchanges physical connections for two already-proven symmetric input pins.
fn swap_instance_connections(
    state: &mut SearchState,
    instance: usize,
    first_name: &str,
    second_name: &str,
) -> Result<()> {
    let connections = &state.module.instances[instance].connections;
    let first = connections
        .iter()
        .position(|(pin, _)| state.interner.resolve(*pin) == Some(first_name))
        .ok_or_else(|| anyhow!("first MCMC pin-swap connection is missing"))?;
    let second = connections
        .iter()
        .position(|(pin, _)| state.interner.resolve(*pin) == Some(second_name))
        .ok_or_else(|| anyhow!("second MCMC pin-swap connection is missing"))?;
    let first_reference = state.module.instances[instance].connections[first]
        .1
        .clone();
    state.module.instances[instance].connections[first].1 = state.module.instances[instance]
        .connections[second]
        .1
        .clone();
    state.module.instances[instance].connections[second].1 = first_reference;
    Ok(())
}

/// Resolves physical instance names for stable user-facing optimization traces.
fn instance_name(state: &SearchState, index: usize) -> Result<String> {
    let instance = state
        .module
        .instances
        .get(index)
        .ok_or_else(|| anyhow!("mapped MCMC instance index is out of range"))?;
    state
        .interner
        .resolve(instance.instance_name)
        .map(str::to_string)
        .ok_or_else(|| anyhow!("cannot resolve mapped MCMC instance name"))
}

#[cfg(test)]
mod tests {
    use super::{NetlistMcmcObjective, NetlistMcmcOptions, optimize_mapped_netlist_mcmc};
    use crate::liberty_model::{Library, LibraryBuilder};
    use crate::netlist::cell_catalog::test_utils::{parse_module, sizing_library, timed_cell};
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::gatefn_from_netlist::{LabeledNetlistAig, project_labeled_netlist_aig};
    use crate::netlist::parse::{Net, NetlistModule};
    use crate::netlist::report::build_netlist_report;
    use crate::netlist::timing_buffer::tests::{
        high_fanout_register_source, registered_timing_library,
    };
    use std::sync::Arc;
    use string_interner::symbol::SymbolU32;
    use string_interner::{StringInterner, backend::StringBackend};
    use xlsynth::IrBits;

    /// Supplies exact AND/OR/complex alternatives with complete Liberty timing.
    fn remapping_library(complex_delay: f64, complex_area: f64) -> Library {
        let mut builder = LibraryBuilder::from_library(sizing_library());
        let or = timed_cell(
            &mut builder,
            "OR2",
            &["A", "B"],
            "A + B",
            1.0,
            2.0,
            0.1,
            1.6,
        );
        let complex = timed_cell(
            &mut builder,
            "AO21",
            &["A", "B", "C"],
            "(A * B) + C",
            complex_area,
            complex_delay,
            0.1,
            1.6,
        );
        let inverter = timed_cell(&mut builder, "INV", &["A"], "!A", 0.5, 1.0, 0.1, 1.6);
        builder.cells.extend([or, complex, inverter]);
        builder.finish()
    }

    /// Exhaustively evaluates every scalar input combination after remapping.
    fn assert_same_scalar_truth(
        original: &LabeledNetlistAig,
        module: &NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        library: &Library,
    ) {
        let remapped = project_labeled_netlist_aig(module, nets, interner, library)
            .expect("project the remapped combinational netlist");
        assert_eq!(original.gate_fn.inputs.len(), remapped.gate_fn.inputs.len());
        for assignment in 0..(1usize << original.gate_fn.inputs.len()) {
            let inputs = original
                .gate_fn
                .inputs
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    IrBits::make_ubits(1, ((assignment >> index) & 1) as u64)
                        .expect("construct one scalar input")
                })
                .collect::<Vec<_>>();
            assert_eq!(
                original
                    .evaluate_bits(inputs.as_slice())
                    .expect("evaluate the original netlist"),
                remapped
                    .evaluate_bits(inputs.as_slice())
                    .expect("evaluate the remapped netlist"),
                "mapped cone changed its function on input assignment {assignment}"
            );
        }
    }

    #[test]
    fn incrementally_upsizes_a_slow_critical_gate() {
        let library = Arc::new(sizing_library());
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 critical (.A(a), .B(b), .Y(y));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                iterations: 64,
                enable_buffer_moves: false,
                enable_pin_swaps: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("incrementally optimize one critical gate");

        assert!(stats.final_delay < stats.initial_delay);
        assert_eq!(
            interner.resolve(module.instances[0].type_name),
            Some("AND2_FAST")
        );
        assert!(stats.incremental_timing_evaluations > 0);
        assert!(stats.accepted_edits_by_kind.contains_key("resize_cell"));
    }

    #[test]
    fn recovers_slack_cell_area_without_worsening_timing() {
        let library = Arc::new(sizing_library());
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, b, critical, slack);
  input a, b;
  output critical, slack;
  AND2 critical_gate (.A(a), .B(b), .Y(critical));
  BUF_FAST slack_gate (.A(a), .Y(slack));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                objective: NetlistMcmcObjective::Area,
                iterations: 128,
                enable_buffer_moves: false,
                enable_pin_swaps: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("recover area from a noncritical oversized buffer");

        assert!(stats.final_area < stats.initial_area);
        assert!(stats.final_delay <= stats.initial_delay + 1e-9);
    }

    #[test]
    fn honors_explicit_zero_area_growth_during_delay_search() {
        let library = Arc::new(sizing_library());
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 critical (.A(a), .B(b), .Y(y));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                iterations: 64,
                max_area_growth: Some(0.0),
                enable_buffer_moves: false,
                enable_pin_swaps: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("reject a faster gate when it exceeds the requested area budget");

        assert_eq!(stats.final_area, stats.initial_area);
        assert_eq!(stats.final_delay, stats.initial_delay);
        assert_eq!(
            interner.resolve(module.instances[0].type_name),
            Some("AND2")
        );
    }

    #[test]
    fn removes_imported_buffers_without_instance_name_assumptions() {
        let library = Arc::new(sizing_library());
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, y);
  input a;
  output y;
  wire middle;
  BUF abc_identity_217 (.A(a), .Y(middle));
  BUF_FAST visible_driver (.A(middle), .Y(y));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                objective: NetlistMcmcObjective::Area,
                iterations: 96,
                enable_sizing: false,
                enable_pin_swaps: false,
                enable_remap: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("bypass a functionally classified imported buffer");

        assert!(stats.final_area < stats.initial_area);
        assert!(stats.final_delay <= stats.initial_delay);
        assert_eq!(module.instances.len(), 1);
        assert!(stats.accepted_edits_by_kind.contains_key("remove_buffer"));
    }

    #[test]
    fn opportunistically_buffers_critical_register_fanout() {
        let library = Arc::new(registered_timing_library());
        let (mut module, mut nets, mut interner) = parse_module(high_fanout_register_source());

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library.clone(),
            &NetlistMcmcOptions {
                iterations: 256,
                enable_sizing: false,
                enable_pin_swaps: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("explore timing-improving register fanout isolation");
        let report = build_netlist_report(
            &module,
            nets.as_slice(),
            &interner,
            library.as_ref(),
            Default::default(),
        )
        .expect("independently verify registered capture timing");

        assert!(stats.final_delay < stats.initial_delay);
        assert_eq!(
            stats.final_delay,
            report.max_register_to_register_delay.unwrap()
        );
        assert!(module.instances.len() > 9);
        assert!(stats.complete_timing_evaluations > 1);
    }

    #[test]
    fn buffers_timing_critical_fanout_below_its_electrical_limit() {
        let library = Arc::new(registered_timing_library());
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(clk, a, y0, y1, y2, y3);
  input clk, a;
  output y0, y1, y2, y3;
  wire root;
  DFF launch (.CLK(clk), .D(a), .Q(root));
  DFF sink0 (.CLK(clk), .D(root), .Q(y0));
  DFF sink1 (.CLK(clk), .D(root), .Q(y1));
  DFF sink2 (.CLK(clk), .D(root), .Q(y2));
  DFF sink3 (.CLK(clk), .D(root), .Q(y3));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                iterations: 256,
                enable_sizing: false,
                enable_pin_swaps: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("isolate noncritical register loads even without an electrical violation");

        assert!(stats.final_delay < stats.initial_delay);
        assert!(stats.accepted_edits_by_kind.contains_key("insert_buffer"));
    }

    #[test]
    fn resizes_equivalent_flip_flops_without_changing_clock_connections() {
        let mut builder = LibraryBuilder::from_library(registered_timing_library());
        let mut smaller = builder
            .cells
            .iter()
            .find(|cell| cell.name == "DFF")
            .expect("find original physical flip-flop")
            .clone();
        smaller.name = "DFF_SMALL".to_string();
        smaller.area = 2.0;
        builder.cells.push(smaller);
        let library = Arc::new(builder.finish());
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(clk, a, y);
  input clk, a;
  output y;
  wire stage;
  DFF launch (.CLK(clk), .D(a), .Q(stage));
  DFF capture (.CLK(clk), .D(stage), .Q(y));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                objective: NetlistMcmcObjective::Area,
                iterations: 96,
                enable_buffer_moves: false,
                enable_pin_swaps: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("recover area from pin- and state-equivalent physical registers");

        assert!(stats.final_area < stats.initial_area);
        assert!((stats.final_delay - stats.initial_delay).abs() < 1e-9);
        assert!(
            module
                .instances
                .iter()
                .any(|instance| { interner.resolve(instance.type_name) == Some("DFF_SMALL") })
        );
        let text = emit_module_as_netlist_text(&module, nets.as_slice(), &interner)
            .expect("render optimized physical-register netlist");
        assert_eq!(text.matches(".CLK(clk)").count(), 2);
    }

    #[test]
    fn parallel_chains_produce_deterministic_best_netlists() {
        let library = Arc::new(sizing_library());
        let source = r#"
module top(a, b, critical, slack);
  input a, b;
  output critical, slack;
  AND2 first (.A(a), .B(b), .Y(critical));
  BUF_FAST second (.A(a), .Y(slack));
endmodule
"#;
        let run = || {
            let (mut module, mut nets, mut interner) = parse_module(source);
            let stats = optimize_mapped_netlist_mcmc(
                &mut module,
                &mut nets,
                &mut interner,
                library.clone(),
                &NetlistMcmcOptions {
                    iterations: 48,
                    threads: 2,
                    checkpoint_iterations: 16,
                    seed: 29,
                    ..NetlistMcmcOptions::default()
                },
            )
            .expect("run deterministic parallel mapped MCMC");
            let text = emit_module_as_netlist_text(&module, nets.as_slice(), &interner)
                .expect("render deterministic mapped result");
            (text, stats.final_area, stats.final_delay)
        };

        assert_eq!(run(), run());
    }

    #[test]
    fn wall_clock_budget_gracefully_returns_the_best_parallel_state() {
        let library = Arc::new(sizing_library());
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 critical (.A(a), .B(b), .Y(y));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                iterations: 1,
                time_limit_seconds: Some(1),
                threads: 2,
                checkpoint_iterations: 16,
                enable_buffer_moves: false,
                enable_pin_swaps: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("finish parallel exploration when the shared wall-clock budget expires");

        assert!(stats.elapsed_seconds >= 0.9);
        assert!(stats.elapsed_seconds < 5.0);
        assert!(stats.attempted_moves > 1);
        assert!(stats.final_delay < stats.initial_delay);
    }

    #[test]
    fn collapses_a_fanout_free_cone_into_an_exact_complex_cell() {
        let library = Arc::new(remapping_library(3.0, 1.25));
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, b, c, y);
  input a, b, c;
  output y;
  wire product;
  AND2 first (.A(a), .B(b), .Y(product));
  OR2 second (.A(product), .B(c), .Y(y));
endmodule
"#,
        );
        let original = project_labeled_netlist_aig(&module, &nets, &interner, library.as_ref())
            .expect("project the original fanout-free cone");

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library.clone(),
            &NetlistMcmcOptions {
                objective: NetlistMcmcObjective::Area,
                iterations: 256,
                enable_sizing: false,
                enable_pin_swaps: false,
                enable_buffer_moves: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("replace a closed AND/OR cone with one exact Liberty complex gate");

        assert_eq!(module.instances.len(), 1);
        assert_eq!(
            interner.resolve(module.instances[0].type_name),
            Some("AO21")
        );
        assert!(stats.final_area < stats.initial_area);
        assert!(stats.final_delay <= stats.initial_delay);
        assert!(stats.accepted_edits_by_kind.contains_key("collapse_cone"));
        assert_same_scalar_truth(&original, &module, &nets, &interner, library.as_ref());
    }

    #[test]
    fn expands_a_slow_complex_cell_into_a_faster_equivalent_cover() {
        let library = Arc::new(remapping_library(12.0, 1.25));
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, b, c, y);
  input a, b, c;
  output y;
  AO21 complex (.A(a), .B(b), .C(c), .Y(y));
endmodule
"#,
        );
        let original = project_labeled_netlist_aig(&module, &nets, &interner, library.as_ref())
            .expect("project the original complex gate");

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library.clone(),
            &NetlistMcmcOptions {
                iterations: 256,
                enable_sizing: false,
                enable_pin_swaps: false,
                enable_buffer_moves: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("expand a slow complex function into faster exact combinational gates");

        assert!(module.instances.len() >= 2);
        assert!(stats.final_delay < stats.initial_delay);
        assert!(stats.accepted_edits_by_kind.contains_key("expand_cone"));
        assert_same_scalar_truth(&original, &module, &nets, &interner, library.as_ref());
    }

    #[test]
    fn relaxes_complex_cell_sizing_while_recovering_area() {
        let mut builder = LibraryBuilder::from_library(remapping_library(12.0, 0.75));
        let fast_complex = timed_cell(
            &mut builder,
            "AO21_FAST",
            &["A", "B", "C"],
            "(A * B) + C",
            1.5,
            3.0,
            0.2,
            1.6,
        );
        builder.cells.push(fast_complex);
        let library = Arc::new(builder.finish());
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, b, c, y);
  input a, b, c;
  output y;
  wire product;
  AND2 first (.A(a), .B(b), .Y(product));
  OR2 second (.A(product), .B(c), .Y(y));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                objective: NetlistMcmcObjective::Area,
                iterations: 256,
                enable_pin_swaps: false,
                enable_buffer_moves: false,
                remap_relax_evaluations: 16,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("jointly recover mapped area and settle the replacement drive strength");

        assert_eq!(module.instances.len(), 1);
        assert_eq!(
            interner.resolve(module.instances[0].type_name),
            Some("AO21_FAST")
        );
        assert!(stats.final_area < stats.initial_area);
        assert!(stats.final_delay <= stats.initial_delay);
        assert!(stats.incremental_timing_evaluations > 0);
    }

    #[test]
    fn rejects_unsupported_remap_truth_table_widths() {
        for max_remap_leaves in [1, 7] {
            let library = Arc::new(sizing_library());
            let (mut module, mut nets, mut interner) = parse_module(
                r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 gate (.A(a), .B(b), .Y(y));
endmodule
"#,
            );

            let error = optimize_mapped_netlist_mcmc(
                &mut module,
                &mut nets,
                &mut interner,
                library,
                &NetlistMcmcOptions {
                    max_remap_leaves,
                    ..NetlistMcmcOptions::default()
                },
            )
            .expect_err("reject truth tables outside the supported input range");

            assert_eq!(
                error.to_string(),
                "mapped MCMC remapping requires between two and six boundary inputs"
            );
        }
    }

    #[test]
    fn preserves_shared_logic_when_replacing_a_mapped_cone() {
        let library = Arc::new(remapping_library(3.0, 1.25));
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, b, c, y, shared_output);
  input a, b, c;
  output y, shared_output;
  wire shared;
  AND2 shared_gate (.A(a), .B(b), .Y(shared));
  OR2 first_use (.A(shared), .B(c), .Y(y));
  BUF second_use (.A(shared), .Y(shared_output));
endmodule
"#,
        );

        optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                objective: NetlistMcmcObjective::Area,
                iterations: 256,
                enable_sizing: false,
                enable_pin_swaps: false,
                enable_buffer_moves: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("preserve an externally shared fanin when replacing local covers");

        assert!(
            module.instances.iter().any(|instance| {
                interner.resolve(instance.instance_name) == Some("shared_gate")
            })
        );
    }

    #[test]
    fn remaps_registered_data_without_crossing_clock_or_state_boundaries() {
        let mut builder = LibraryBuilder::from_library(registered_timing_library());
        let or = timed_cell(
            &mut builder,
            "OR2",
            &["A", "B"],
            "A + B",
            1.0,
            2.0,
            0.1,
            1.6,
        );
        let complex = timed_cell(
            &mut builder,
            "AO21",
            &["A", "B", "C"],
            "(A * B) + C",
            1.25,
            2.0,
            0.1,
            1.6,
        );
        builder.cells.extend([or, complex]);
        let library = Arc::new(builder.finish());
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(clk, a, b, c, y);
  input clk, a, b, c;
  output y;
  wire registered_a, registered_b, registered_c, product, combined;
  DFF launch_a (.CLK(clk), .D(a), .Q(registered_a));
  DFF launch_b (.CLK(clk), .D(b), .Q(registered_b));
  DFF launch_c (.CLK(clk), .D(c), .Q(registered_c));
  AND2 first (.A(registered_a), .B(registered_b), .Y(product));
  OR2 second (.A(product), .B(registered_c), .Y(combined));
  DFF capture (.CLK(clk), .D(combined), .Q(y));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist_mcmc(
            &mut module,
            &mut nets,
            &mut interner,
            library,
            &NetlistMcmcOptions {
                iterations: 384,
                enable_sizing: false,
                enable_pin_swaps: false,
                enable_buffer_moves: false,
                ..NetlistMcmcOptions::default()
            },
        )
        .expect("replace logic between physical launch and capture registers");

        assert!(stats.final_delay < stats.initial_delay);
        assert_eq!(
            module
                .instances
                .iter()
                .filter(|instance| interner.resolve(instance.type_name) == Some("DFF"))
                .count(),
            4
        );
        let text = emit_module_as_netlist_text(&module, nets.as_slice(), &interner)
            .expect("render remapped registered netlist");
        assert_eq!(text.matches(".CLK(clk)").count(), 4);
    }
}
