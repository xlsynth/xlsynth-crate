// SPDX-License-Identifier: Apache-2.0

//! Incremental exact-Liberty sizing across physical register boundaries.

use crate::liberty::cell_formula::parse_formula;
use crate::liberty_model::{Cell, Library, PinDirection};
use crate::netlist::cell_catalog::CellCatalog;
use crate::netlist::normalized::{BitExpr, BitIndex, BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{Net, NetRef, NetlistModule, PortDirection, PortId};
use crate::netlist::report::{NetlistReport, build_netlist_report_with_primary_input_arrivals};
use crate::netlist::resize::{
    PinSwapStep, ResizeOptions, ResizeStats, ResizeStep, validate_options,
};
use crate::netlist::sequential_liberty::get_gv_eval_sequential_cell_spec;
use crate::netlist::sta::{
    CombinationalOutputLoad, EdgeTiming, SignalTiming, StaOptions, TimingEdge, TimingPredecessor,
    TimingQueryDiagnosticCounts, TracedCombinationalTiming,
    analyze_combinational_max_arrival_with_primary_input_arrivals,
    analyze_register_boundary_max_arrival_with_primary_input_arrivals,
    effective_input_capacitance_for_mapping,
    evaluate_combinational_cell_output_timing_with_predecessors,
    evaluate_sequential_cell_capture_timing_with_predecessor,
    evaluate_sequential_cell_output_timing, is_sequential_boundary_cell,
};
use crate::netlist::timing_buffer::BufferTimingConstraints;
use crate::netlist::utils::validate_constant_output_assignments;
use anyhow::{Context, Result, anyhow, bail};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Matches ABC's squared gain when sizing an already buffered netlist.
const MAX_ELECTRICAL_EFFORT: f64 = 9.0;
/// Avoid rejecting equivalent table computations on roundoff alone.
const TIMING_VERIFICATION_EPSILON: f64 = 1e-7;
/// Limit exhaustive FF state-function classification to practical cell sizes.
const MAX_SEQUENTIAL_FUNCTION_INPUTS: usize = 8;
/// Start at ABC's one-percent near-critical arrival window.
const INITIAL_CRITICAL_WINDOW: f64 = 0.01;
/// Double the slack window when a critical round yields no useful moves.
const MAX_CRITICAL_WINDOW: f64 = 1.0;
/// Select an independent batch comparable to ABC's ten-percent sizing queue.
const SIZING_BATCH_DIVISOR: usize = 10;
/// Observe enough tied endpoints to distinguish improvements in wide datapaths.
const MAX_SECONDARY_ENDPOINTS: usize = 32;
/// Preserve alternate endpoint cones without freezing genuinely slack paths.
const SECONDARY_CRITICAL_WINDOW: f64 = 0.05;
/// Give widespread electrical repairs broader coverage than one timing round.
const ELECTRICAL_SIZING_BUDGET_MULTIPLIER: usize = 8;
/// Allow a coordinated sizing wave to cross a small, recoverable delay ridge.
const MAX_EXPLORATORY_TIMING_REGRESSION: f64 = 0.01;
/// Skip whole-cone sizing waves when ordinary exact trials cover the design.
const MIN_COORDINATED_SIZING_CONE: usize = 16;
/// Avoid replacing a small gate with an unrealistically abrupt input-load jump.
const MAX_COORDINATED_INPUT_GROWTH: f64 = 2.0;
/// Bound exploratory sizing waves that have not improved the saved solution.
const MAX_COORDINATED_STAGNANT_WAVES: usize = 8;
/// Whole-cone retiming is cheap enough to explore several waves per exact
/// round.
const COORDINATED_SIZING_WAVE_MULTIPLIER: usize = 4;
/// Scan beyond the exact-trial budget when substitutions are ranked cheaply.
const COORDINATED_CANDIDATE_MULTIPLIER: usize = 4;

/// One pin- and state-equivalent family of positively clocked Liberty FFs.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RegisterFamilyKey {
    clock_pin: String,
    input_pins: Vec<String>,
    output_functions: Vec<(String, u8)>,
    next_state_truth: Vec<bool>,
}

/// One safely interchangeable, fully described physical flip-flop variant.
#[derive(Clone, Debug)]
struct RegisterCatalogCell {
    cell_index: usize,
    name: String,
    family: RegisterFamilyKey,
    area: f64,
}

/// Functionally indexed flip-flop families, independent of Liberty names.
#[derive(Default)]
struct RegisterCellCatalog {
    cells: Vec<RegisterCatalogCell>,
    by_name: HashMap<String, usize>,
    families: BTreeMap<RegisterFamilyKey, Vec<usize>>,
}

impl RegisterCellCatalog {
    /// Indexes only supported FFs with exactly interchangeable interfaces.
    fn new(library: &Library) -> Result<Self> {
        let mut result = Self::default();
        for (cell_index, cell) in library.cells.iter().enumerate() {
            let Some(candidate) = classify_register_cell(library, cell_index, cell)? else {
                continue;
            };
            let index = result.cells.len();
            if result
                .by_name
                .insert(candidate.name.clone(), index)
                .is_some()
            {
                bail!(
                    "Liberty defines usable flip-flop '{}' twice",
                    candidate.name
                );
            }
            result
                .families
                .entry(candidate.family.clone())
                .or_default()
                .push(index);
            result.cells.push(candidate);
        }
        for members in result.families.values_mut() {
            members.sort_by(|lhs, rhs| {
                result.cells[*lhs]
                    .area
                    .total_cmp(&result.cells[*rhs].area)
                    .then_with(|| result.cells[*lhs].name.cmp(&result.cells[*rhs].name))
            });
        }
        Ok(result)
    }

    /// Resolves the safely classified variant of one physical FF instance.
    fn by_name(&self, name: &str) -> Option<&RegisterCatalogCell> {
        self.by_name.get(name).map(|index| &self.cells[*index])
    }

    /// Iterates only exact-state, exact-clock, and exact-pin FF variants.
    fn family(&self, cell: &RegisterCatalogCell) -> impl Iterator<Item = &RegisterCatalogCell> {
        self.families
            .get(&cell.family)
            .into_iter()
            .flatten()
            .map(|index| &self.cells[*index])
    }
}

/// Classifies a positive-edge FF without depending on drive-strength spelling.
fn classify_register_cell(
    library: &Library,
    cell_index: usize,
    cell: &Cell,
) -> Result<Option<RegisterCatalogCell>> {
    if cell.sequential.is_empty()
        || cell.clock_gate.is_some()
        || cell.dont_use == Some(true)
        || !cell.area.is_finite()
        || cell.area < 0.0
    {
        return Ok(None);
    }
    let spec = match get_gv_eval_sequential_cell_spec(cell, library) {
        Ok(Some(spec)) if !spec.clock.is_negated => spec,
        Ok(Some(_)) | Ok(None) | Err(_) => return Ok(None),
    };

    let mut input_pins = cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Input as i32 && !pin.is_clocking_pin)
        .map(|pin| library.resolve_string(&pin.name).to_string())
        .collect::<Vec<_>>();
    input_pins.sort();
    if input_pins.is_empty()
        || input_pins.len() > MAX_SEQUENTIAL_FUNCTION_INPUTS
        || input_pins.windows(2).any(|pins| pins[0] == pins[1])
    {
        return Ok(None);
    }

    let allowed = input_pins
        .iter()
        .cloned()
        .chain(std::iter::once(spec.state_var.clone()))
        .chain(spec.complementary_state_var.iter().cloned())
        .collect::<BTreeSet<_>>();
    if spec
        .next_state
        .inputs()
        .into_iter()
        .any(|input| !allowed.contains(&input))
    {
        return Ok(None);
    }

    let mut next_state_truth = Vec::with_capacity(1 << (input_pins.len() + 1));
    for assignment in 0..(1usize << (input_pins.len() + 1)) {
        let state = (assignment & 1) != 0;
        let mut values = HashMap::with_capacity(input_pins.len() + 2);
        values.insert(spec.state_var.clone(), state);
        if let Some(complement) = &spec.complementary_state_var {
            values.insert(complement.clone(), !state);
        }
        for (index, input) in input_pins.iter().enumerate() {
            values.insert(input.clone(), ((assignment >> (index + 1)) & 1) != 0);
        }
        let Some(next_state) = spec.next_state.evaluate_partial(&values) else {
            return Ok(None);
        };
        next_state_truth.push(next_state);
    }

    let mut output_functions = Vec::new();
    for pin in cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Output as i32)
    {
        let formula = library.resolve_string(&pin.function);
        let Ok(term) = parse_formula(formula) else {
            return Ok(None);
        };
        if term.inputs().into_iter().any(|input| {
            input != spec.state_var
                && spec
                    .complementary_state_var
                    .as_ref()
                    .is_none_or(|complement| input != *complement)
        }) {
            return Ok(None);
        }
        let mut signature = 0u8;
        for (index, state) in [false, true].into_iter().enumerate() {
            let mut values = HashMap::with_capacity(2);
            values.insert(spec.state_var.clone(), state);
            if let Some(complement) = &spec.complementary_state_var {
                values.insert(complement.clone(), !state);
            }
            let Some(value) = term.evaluate_partial(&values) else {
                return Ok(None);
            };
            signature |= u8::from(value) << index;
        }
        output_functions.push((library.resolve_string(&pin.name).to_string(), signature));
    }
    output_functions.sort();
    if output_functions.is_empty()
        || output_functions
            .windows(2)
            .any(|outputs| outputs[0].0 == outputs[1].0)
    {
        return Ok(None);
    }

    Ok(Some(RegisterCatalogCell {
        cell_index,
        name: cell.name.clone(),
        family: RegisterFamilyKey {
            clock_pin: spec.clock.pin_name,
            input_pins,
            output_functions,
            next_state_truth,
        },
        area: cell.area,
    }))
}

/// Independent timing envelopes for primary-input and register launches.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct BoundarySignalTiming {
    primary_input: Option<SignalTiming>,
    register: Option<SignalTiming>,
}

impl BoundarySignalTiming {
    /// Selects the physically independent primary-input or register launch.
    fn for_launch(self, register_launch: bool) -> Option<SignalTiming> {
        if register_launch {
            self.register
        } else {
            self.primary_input
        }
    }
}

/// Identifies an actual rise/fall transition on one normalized net bit.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BitEdge {
    bit: BitIndex,
    edge: TimingEdge,
}

/// Retains the true Liberty input predecessor for both output transitions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EdgePredecessors {
    rise: Option<BitEdge>,
    fall: Option<BitEdge>,
}

impl EdgePredecessors {
    /// Retrieves the predecessor for one rise/fall output transition.
    fn for_edge(self, edge: TimingEdge) -> Option<BitEdge> {
        match edge {
            TimingEdge::Rise => self.rise,
            TimingEdge::Fall => self.fall,
        }
    }
}

/// Keeps independent timing predecessors for data- and FF-launched paths.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct BoundaryPredecessors {
    primary_input: EdgePredecessors,
    register: EdgePredecessors,
}

impl BoundaryPredecessors {
    /// Selects the predecessor belonging to the actual endpoint launch class.
    fn for_launch(self, register_launch: bool) -> EdgePredecessors {
        if register_launch {
            self.register
        } else {
            self.primary_input
        }
    }
}

/// Identifies the exact setup-winning transition at a register capture.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CapturePredecessors {
    primary_input: Option<BitEdge>,
    register: Option<BitEdge>,
}

/// One Liberty-related input contribution to both edges of an output bit.
#[derive(Clone, Copy, Debug, Default)]
struct WindowArc {
    rise: Option<(f64, BitEdge)>,
    fall: Option<(f64, BitEdge)>,
}

impl WindowArc {
    /// Retrieves the actual input contribution to a selected output edge.
    fn for_edge(self, edge: TimingEdge) -> Option<(f64, BitEdge)> {
        match edge {
            TimingEdge::Rise => self.rise,
            TimingEdge::Fall => self.fall,
        }
    }
}

/// One endpoint in the four independently constrained timing classes.
#[derive(Clone, Copy, Debug)]
struct CriticalEndpoint {
    class: usize,
    arrival: f64,
    transition: BitEdge,
    register_launch: bool,
    capture: Option<usize>,
    available_slack: f64,
}

/// A distinct cell in one or more exact-arc near-critical timing cones.
#[derive(Clone, Copy, Debug)]
struct CriticalInstance {
    instance_index: usize,
    path_count: usize,
    first_path: usize,
    slack: f64,
}

#[derive(Clone, Debug)]
struct SizingInput {
    name: String,
    bit: Option<BitIndex>,
    clock: bool,
}

#[derive(Clone, Debug)]
struct SizingOutput {
    name: String,
    pin_index: usize,
    bit: BitIndex,
}

#[derive(Clone, Debug)]
struct SizingInstance {
    cell_index: usize,
    inputs: Vec<SizingInput>,
    outputs: Vec<SizingOutput>,
    known_pin_values: HashMap<String, bool>,
    sequential: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RegisterCaptureScore {
    primary_input: Option<f64>,
    register: Option<f64>,
}

#[derive(Clone, Debug)]
struct OutputEndpoint {
    name: String,
    bit: BitIndex,
}

/// Register-first objective shared by exact and incremental timing reports.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct BoundaryTimingScore {
    register_to_register: Option<f64>,
    input_to_register: Option<f64>,
    register_to_output: Option<f64>,
    input_to_output: Option<f64>,
}

impl BoundaryTimingScore {
    /// Builds the same four timing classes reported by `gv-stats`.
    fn from_report(report: &NetlistReport) -> Self {
        Self {
            register_to_register: report.max_register_to_register_delay,
            input_to_register: report.max_input_to_register_delay,
            register_to_output: report.max_register_to_output_delay,
            input_to_output: report.max_delay,
        }
    }

    /// Prioritizes register capture, then output paths, deterministically.
    fn improvement_over(self, previous: Self, epsilon: f64) -> Option<f64> {
        for (candidate, current) in self.values().into_iter().zip(previous.values()) {
            let delta = current - candidate;
            if delta.abs() > epsilon {
                return (delta > 0.0).then_some(delta);
            }
        }
        None
    }

    /// Ensures area recovery does not worsen any achieved timing class.
    fn no_worse_than(self, limit: Self, epsilon: f64) -> bool {
        self.values()
            .into_iter()
            .zip(limit.values())
            .all(|(candidate, bound)| candidate <= bound + epsilon)
    }

    /// Returns the independently measured worst endpoint delay.
    fn worst_delay(self) -> f64 {
        self.values().into_iter().fold(0.0, f64::max)
    }

    fn values(self) -> [f64; 4] {
        [
            self.register_to_register.unwrap_or(0.0),
            self.input_to_register.unwrap_or(0.0),
            self.register_to_output.unwrap_or(0.0),
            self.input_to_output.unwrap_or(0.0),
        ]
    }
}

#[derive(Clone, Debug)]
struct SizingTrial {
    score: BoundaryTimingScore,
    secondary_delay: f64,
    recomputed_instances: usize,
    clock_load: f64,
    constraints_satisfied: bool,
    local_improvement: f64,
}

/// One function-preserving incremental cell-assignment change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SizingMoveKind {
    Resize {
        cell_index: usize,
    },
    SwapPins {
        first_input: usize,
        second_input: usize,
    },
}

#[derive(Clone, Debug)]
struct SizingMove {
    instance_index: usize,
    kind: SizingMoveKind,
    score: BoundaryTimingScore,
    ranking: f64,
    area: f64,
}

/// Fairly scheduled exact trials for one timing-relevant physical instance.
#[derive(Clone, Debug)]
struct TimingMoveQueue {
    instance_index: usize,
    moves: VecDeque<SizingMoveKind>,
}

/// Actual-load local timing estimate for ordering equivalent Liberty sizes.
#[derive(Clone, Debug)]
struct EstimatedCellAlternative {
    cell_index: usize,
    arrival: f64,
    input_load: f64,
}

/// Original raw connections saved lazily for instances whose pins move.
#[derive(Clone, Debug, Default)]
struct PinAssignmentHistory {
    original_connections: BTreeMap<usize, Vec<(PortId, NetRef)>>,
    visited_assignments: BTreeSet<PinAssignmentKey>,
}

/// One deterministic normalized source attached to a characterized input pin.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PinBindingKey {
    bit: Option<BitIndex>,
    constant: Option<bool>,
}

/// A complete cell-and-pin assignment used to prevent inverse-swap cycles.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PinAssignmentKey {
    instance_index: usize,
    cell_index: usize,
    inputs: Vec<PinBindingKey>,
}

impl PinAssignmentKey {
    /// Captures each sorted input's exact normalized net or known constant.
    fn current(timing: &IncrementalRegisteredSta<'_>, instance_index: usize) -> Self {
        let instance = &timing.instances[instance_index];
        Self {
            instance_index,
            cell_index: instance.cell_index,
            inputs: instance
                .inputs
                .iter()
                .map(|input| PinBindingKey {
                    bit: input.bit,
                    constant: instance.known_pin_values.get(&input.name).copied(),
                })
                .collect(),
        }
    }

    /// Returns the complete assignment produced by one legal input exchange.
    fn swapped(
        timing: &IncrementalRegisteredSta<'_>,
        instance_index: usize,
        first_input: usize,
        second_input: usize,
    ) -> Self {
        let mut result = Self::current(timing, instance_index);
        result.inputs.swap(first_input, second_input);
        result
    }
}

/// Retains the best independently realizable cell assignment from sizing.
#[derive(Clone, Debug)]
struct BestSizingState {
    cell_indices: Vec<usize>,
    pin_connections: BTreeMap<usize, Vec<(PortId, NetRef)>>,
    score: BoundaryTimingScore,
    secondary_delay: f64,
    area: f64,
    replacements: usize,
    pin_swap_steps: usize,
    upsizes: usize,
    downsizes: usize,
    register_upsizes: usize,
    register_downsizes: usize,
    pin_swaps: usize,
}

impl BestSizingState {
    /// Captures the complete current solution rather than only a local move.
    fn capture(
        module: &NetlistModule,
        timing: &IncrementalRegisteredSta<'_>,
        score: BoundaryTimingScore,
        area: f64,
        stats: &ResizeStats,
        pin_history: &PinAssignmentHistory,
    ) -> Self {
        Self {
            cell_indices: timing
                .instances
                .iter()
                .map(|instance| instance.cell_index)
                .collect(),
            pin_connections: pin_history
                .original_connections
                .iter()
                .filter_map(|(index, original)| {
                    let current = &module.instances[*index].connections;
                    (current != original).then(|| (*index, current.clone()))
                })
                .collect(),
            score,
            secondary_delay: timing.secondary_delay(),
            area,
            replacements: stats.replacements.len(),
            pin_swap_steps: stats.pin_swap_steps.len(),
            upsizes: stats.upsizes,
            downsizes: stats.downsizes,
            register_upsizes: stats.register_upsizes,
            register_downsizes: stats.register_downsizes,
            pin_swaps: stats.pin_swaps,
        }
    }

    /// Prefers actual endpoint delay, using total cell area only as a
    /// tie-break.
    fn update_if_better(
        &mut self,
        module: &NetlistModule,
        timing: &IncrementalRegisteredSta<'_>,
        score: BoundaryTimingScore,
        area: f64,
        stats: &ResizeStats,
        options: &ResizeOptions,
        pin_history: &PinAssignmentHistory,
    ) {
        let same_timing = score.no_worse_than(self.score, options.improvement_epsilon)
            && self.score.no_worse_than(score, options.improvement_epsilon);
        let same_area = (area - self.area).abs() <= options.area_epsilon;
        if score
            .improvement_over(self.score, options.improvement_epsilon)
            .is_some()
            || (same_timing && area + options.area_epsilon < self.area)
            || (same_timing
                && same_area
                && timing.secondary_delay() + options.improvement_epsilon < self.secondary_delay)
        {
            *self = Self::capture(module, timing, score, area, stats, pin_history);
        }
    }
}

/// Sizes exact-family gates, buffers, and flip-flops across registered stages.
pub fn resize_timing_aware_netlist(
    module: &mut NetlistModule,
    nets: &[Net],
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &ResizeOptions,
    constraints: &BufferTimingConstraints,
) -> Result<ResizeStats> {
    validate_options(options)?;
    validate_constant_output_assignments(module, nets)
        .context("validating register-aware sizing output assignments")?;
    let original_report = build_netlist_report_with_primary_input_arrivals(
        module,
        nets,
        interner,
        library,
        options.sta_options,
        &constraints.primary_input_arrivals,
    )
    .context("computing initial exact register-aware sizing report")?;
    let catalog = CellCatalog::new(library)?;
    let registers = RegisterCellCatalog::new(library)?;
    let mut timing = IncrementalRegisteredSta::new(
        module,
        nets,
        interner,
        library,
        options.sta_options,
        constraints,
    )?;
    let mut score = timing.score();
    verify_exact_score(score, &original_report, options.improvement_epsilon)?;
    let initial_clock_load = timing.clock_load;
    let has_registers = timing.has_registers();
    let mut area = original_report.cell_area;
    let mut stats = ResizeStats {
        initial_delay: score.worst_delay(),
        final_delay: score.worst_delay(),
        initial_area: area,
        final_area: area,
        initial_clock_load: has_registers.then_some(initial_clock_load),
        final_clock_load: has_registers.then_some(initial_clock_load),
        ..ResizeStats::default()
    };

    let mut pin_history = PinAssignmentHistory::default();
    if options.max_iterations > 0 {
        apply_electrical_sizing(
            module,
            interner,
            library,
            &catalog,
            &registers,
            options,
            &mut timing,
            &mut pin_history,
            &mut score,
            &mut area,
            &mut stats,
        )?;
    }

    let mut best_solution =
        BestSizingState::capture(module, &timing, score, area, &stats, &pin_history);
    let mut achieved = score;
    for _ in 0..options.max_outer_iterations {
        stats.outer_iterations += 1;
        let round_start_score = score;
        let round_start_area = area;
        let round_start_secondary = timing.secondary_delay();

        if options.max_iterations > 0 {
            apply_coordinated_timing_waves(
                module,
                nets,
                interner,
                library,
                &catalog,
                &registers,
                options,
                constraints,
                &mut timing,
                &mut pin_history,
                &mut score,
                &mut area,
                &mut stats,
                &mut best_solution,
            )?;
            optimize_timing_moves(
                module,
                interner,
                library,
                &catalog,
                &registers,
                options,
                &mut timing,
                &mut pin_history,
                &mut score,
                &mut area,
                &mut stats,
                &mut best_solution,
                options.max_iterations,
                false,
            )?;
            restore_best_sizing_state(
                module,
                nets,
                interner,
                library,
                options,
                constraints,
                &best_solution,
                &pin_history,
                &mut timing,
                &mut stats,
                &mut score,
                &mut area,
            )?;
        }

        achieved = score;
        recover_timing_protected_area(
            module,
            interner,
            library,
            &catalog,
            &registers,
            options,
            &mut timing,
            &mut pin_history,
            &mut score,
            &mut area,
            &mut stats,
            achieved,
        )?;
        best_solution.update_if_better(module, &timing, score, area, &stats, options, &pin_history);

        let improves_timing = score
            .improvement_over(round_start_score, options.improvement_epsilon)
            .is_some();
        let recovers_area = area + options.area_epsilon < round_start_area;
        let improves_pin_assignment = score
            .no_worse_than(round_start_score, options.improvement_epsilon)
            && (area - round_start_area).abs() <= options.area_epsilon
            && timing.secondary_delay() + options.improvement_epsilon < round_start_secondary;
        if !improves_timing && !recovers_area && !improves_pin_assignment {
            break;
        }
    }

    if options.max_iterations > 0 {
        let final_pin_swaps = optimize_timing_moves(
            module,
            interner,
            library,
            &catalog,
            &registers,
            options,
            &mut timing,
            &mut pin_history,
            &mut score,
            &mut area,
            &mut stats,
            &mut best_solution,
            1,
            true,
        )?;
        if final_pin_swaps > 0 {
            restore_best_sizing_state(
                module,
                nets,
                interner,
                library,
                options,
                constraints,
                &best_solution,
                &pin_history,
                &mut timing,
                &mut stats,
                &mut score,
                &mut area,
            )?;
            achieved = score;
            recover_timing_protected_area(
                module,
                interner,
                library,
                &catalog,
                &registers,
                options,
                &mut timing,
                &mut pin_history,
                &mut score,
                &mut area,
                &mut stats,
                achieved,
            )?;
        }
    }

    let final_report = build_netlist_report_with_primary_input_arrivals(
        module,
        nets,
        interner,
        library,
        options.sta_options,
        &constraints.primary_input_arrivals,
    )
    .context("independently verifying register-aware sizing")?;
    verify_exact_score(score, &final_report, options.improvement_epsilon)?;
    if !BoundaryTimingScore::from_report(&final_report)
        .no_worse_than(achieved, options.improvement_epsilon)
    {
        bail!("register-aware area recovery increased the achieved exact path delay");
    }
    stats.final_delay = score.worst_delay();
    stats.final_area = final_report.cell_area;
    stats.final_clock_load = has_registers.then_some(timing.clock_load);
    Ok(stats)
}

/// Checks the incremental endpoint score against a complete Liberty STA.
fn verify_exact_score(
    score: BoundaryTimingScore,
    report: &NetlistReport,
    epsilon: f64,
) -> Result<()> {
    let exact = BoundaryTimingScore::from_report(report);
    let tolerance = epsilon.max(TIMING_VERIFICATION_EPSILON);
    for (index, (incremental, independent)) in
        score.values().into_iter().zip(exact.values()).enumerate()
    {
        if (incremental - independent).abs() > tolerance {
            bail!(
                "incremental registered sizing timing class {index} disagrees with full Liberty STA: {incremental} vs {independent}"
            );
        }
    }
    Ok(())
}

/// Orders full endpoint timing first, then local delay and deterministic area.
fn compare_delay_moves(
    lhs: &SizingMove,
    rhs: &SizingMove,
    library: &Library,
    options: &ResizeOptions,
) -> std::cmp::Ordering {
    for (left, right) in lhs.score.values().into_iter().zip(rhs.score.values()) {
        if (left - right).abs() > options.improvement_epsilon {
            return left.total_cmp(&right);
        }
    }
    rhs.ranking
        .total_cmp(&lhs.ranking)
        .then_with(|| lhs.area.total_cmp(&rhs.area))
        .then_with(|| lhs.instance_index.cmp(&rhs.instance_index))
        .then_with(|| compare_move_kinds(lhs.kind, rhs.kind, library))
}

/// Breaks otherwise tied moves in favor of a deterministic zero-area swap.
fn compare_move_kinds(
    lhs: SizingMoveKind,
    rhs: SizingMoveKind,
    library: &Library,
) -> std::cmp::Ordering {
    match (lhs, rhs) {
        (
            SizingMoveKind::SwapPins {
                first_input: lhs_first,
                second_input: lhs_second,
            },
            SizingMoveKind::SwapPins {
                first_input: rhs_first,
                second_input: rhs_second,
            },
        ) => lhs_first
            .cmp(&rhs_first)
            .then_with(|| lhs_second.cmp(&rhs_second)),
        (SizingMoveKind::SwapPins { .. }, SizingMoveKind::Resize { .. }) => {
            std::cmp::Ordering::Less
        }
        (SizingMoveKind::Resize { .. }, SizingMoveKind::SwapPins { .. }) => {
            std::cmp::Ordering::Greater
        }
        (
            SizingMoveKind::Resize {
                cell_index: lhs_index,
            },
            SizingMoveKind::Resize {
                cell_index: rhs_index,
            },
        ) => library.cells[lhs_index]
            .name
            .cmp(&library.cells[rhs_index].name),
    }
}

/// Orders timing-safe area recovery by physical area savings first.
fn compare_area_moves(
    lhs: &SizingMove,
    rhs: &SizingMove,
    library: &Library,
    options: &ResizeOptions,
) -> std::cmp::Ordering {
    rhs.ranking
        .total_cmp(&lhs.ranking)
        .then_with(|| compare_delay_moves(lhs, rhs, library, options))
}

/// Restores the best complete assignment before timing-safe area recovery.
#[allow(clippy::too_many_arguments)]
fn restore_best_sizing_state<'a>(
    module: &mut NetlistModule,
    nets: &[Net],
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &'a Library,
    options: &ResizeOptions,
    constraints: &BufferTimingConstraints,
    best: &BestSizingState,
    pin_history: &PinAssignmentHistory,
    timing: &mut IncrementalRegisteredSta<'a>,
    stats: &mut ResizeStats,
    score: &mut BoundaryTimingScore,
    area: &mut f64,
) -> Result<()> {
    let cells_match = timing
        .instances
        .iter()
        .map(|instance| instance.cell_index)
        .eq(best.cell_indices.iter().copied());
    let pins_match = pin_history
        .original_connections
        .iter()
        .all(|(index, original)| {
            let expected = best.pin_connections.get(index).unwrap_or(original);
            module.instances[*index].connections == *expected
        });
    if cells_match && pins_match {
        return Ok(());
    }

    for (instance, cell_index) in module.instances.iter_mut().zip(&best.cell_indices) {
        instance.type_name = interner.get_or_intern(library.cells[*cell_index].name.as_str());
    }
    for (index, original) in &pin_history.original_connections {
        module.instances[*index].connections = best
            .pin_connections
            .get(index)
            .cloned()
            .unwrap_or_else(|| original.clone());
    }
    *timing = IncrementalRegisteredSta::new(
        module,
        nets,
        interner,
        library,
        options.sta_options,
        constraints,
    )
    .context("restoring the best complete Liberty sizing solution")?;
    *score = timing.score();
    if !score.no_worse_than(best.score, TIMING_VERIFICATION_EPSILON)
        || !best
            .score
            .no_worse_than(*score, TIMING_VERIFICATION_EPSILON)
    {
        bail!("restored best sizing solution does not reproduce its exact endpoint timing");
    }
    stats.replacements.truncate(best.replacements);
    stats.pin_swap_steps.truncate(best.pin_swap_steps);
    stats.upsizes = best.upsizes;
    stats.downsizes = best.downsizes;
    stats.register_upsizes = best.register_upsizes;
    stats.register_downsizes = best.register_downsizes;
    stats.pin_swaps = best.pin_swaps;
    stats.final_clock_load = timing.has_registers().then_some(timing.clock_load);
    *area = best.area;
    Ok(())
}

/// Lists safely interchangeable combinational or positive-edge FF variants.
fn size_alternatives(
    timing: &IncrementalRegisteredSta<'_>,
    instance_index: usize,
    library: &Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
) -> Vec<usize> {
    let name = &library.cells[timing.instances[instance_index].cell_index].name;
    if timing.instances[instance_index].sequential {
        registers
            .by_name(name)
            .into_iter()
            .flat_map(|cell| registers.family(cell))
            .map(|cell| cell.cell_index)
            .collect()
    } else {
        catalog
            .by_name(name)
            .into_iter()
            .flat_map(|cell| catalog.family(cell))
            .map(|cell| cell.cell_index)
            .collect()
    }
}

/// Orders equivalent cells using their real current Liberty operating point.
fn ordered_size_alternatives(
    timing: &mut IncrementalRegisteredSta<'_>,
    instance_index: usize,
    library: &Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
) -> Vec<EstimatedCellAlternative> {
    let current_index = timing.instances[instance_index].cell_index;
    let mut candidates = size_alternatives(timing, instance_index, library, catalog, registers)
        .into_iter()
        .filter(|index| *index != current_index)
        .filter_map(|cell_index| {
            let arrival = match timing.estimate_replacement_arrival(instance_index, cell_index) {
                Ok(arrival) if arrival.is_finite() => arrival,
                Ok(_) | Err(_) => return None,
            };
            let input_load =
                average_functional_input_capacitance(library, &library.cells[cell_index]).ok()?;
            Some(EstimatedCellAlternative {
                cell_index,
                arrival,
                input_load,
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|lhs, rhs| {
        library.cells[lhs.cell_index]
            .area
            .total_cmp(&library.cells[rhs.cell_index].area)
            .then_with(|| lhs.arrival.total_cmp(&rhs.arrival))
            .then_with(|| lhs.input_load.total_cmp(&rhs.input_load))
            .then_with(|| {
                library.cells[lhs.cell_index]
                    .name
                    .cmp(&library.cells[rhs.cell_index].name)
            })
    });
    candidates
}

/// Retrieves the characterized rise/fall load of one named functional pin.
fn characterized_input_load(
    library: &Library,
    cell_index: usize,
    pin_name: &str,
) -> Option<CombinationalOutputLoad> {
    let cell = library.cells.get(cell_index)?;
    let pin = cell
        .pins
        .iter()
        .find(|pin| library.resolve_string(&pin.name) == pin_name)?;
    effective_input_capacitance_for_mapping(
        pin,
        &format!("load-sensitive sizing input '{}.{pin_name}'", cell.name),
    )
    .ok()
}

/// Estimates how much a sibling can unload an actual critical driver net.
fn critical_driver_load_relief(
    timing: &IncrementalRegisteredSta<'_>,
    instance_index: usize,
    critical_bits: &BTreeSet<BitIndex>,
    library: &Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
) -> f64 {
    let instance = &timing.instances[instance_index];
    let current = &library.cells[instance.cell_index];
    let classified = (!instance.sequential)
        .then(|| catalog.by_name(current.name.as_str()))
        .flatten();
    let alternatives = size_alternatives(timing, instance_index, library, catalog, registers);
    let mut relief = 0.0_f64;
    for input in instance.inputs.iter().filter(|input| !input.clock) {
        if input.bit.is_none_or(|bit| !critical_bits.contains(&bit)) {
            continue;
        }
        let Some(current_load) =
            characterized_input_load(library, instance.cell_index, input.name.as_str())
        else {
            continue;
        };
        for alternative in &alternatives {
            let Some(candidate) =
                characterized_input_load(library, *alternative, input.name.as_str())
            else {
                continue;
            };
            relief = relief
                .max(current_load.rise - candidate.rise)
                .max(current_load.fall - candidate.fall);
        }
        if let Some(classified) = classified {
            for pair in &classified.symmetric_input_pairs {
                let first = &classified.family.input_names[pair.first_input];
                let second = &classified.family.input_names[pair.second_input];
                let other = if input.name == *first {
                    second
                } else if input.name == *second {
                    first
                } else {
                    continue;
                };
                let Some(candidate) = characterized_input_load(library, instance.cell_index, other)
                else {
                    continue;
                };
                relief = relief
                    .max(current_load.rise - candidate.rise)
                    .max(current_load.fall - candidate.fall);
            }
        }
    }
    relief.max(0.0)
}

/// Interleaves near-critical gates with the siblings loading critical nets.
fn prioritized_timing_instances(
    timing: &IncrementalRegisteredSta<'_>,
    critical: &[CriticalInstance],
    library: &Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
    options: &ResizeOptions,
) -> Vec<usize> {
    let critical_set = critical
        .iter()
        .map(|candidate| candidate.instance_index)
        .collect::<BTreeSet<_>>();
    let mut load_sensitive = BTreeMap::<usize, (f64, usize)>::new();
    for (rank, candidate) in critical.iter().enumerate() {
        let driver = candidate.instance_index;
        let bits = timing.instances[driver]
            .outputs
            .iter()
            .map(|output| output.bit)
            .collect::<BTreeSet<_>>();
        if bits.is_empty() {
            continue;
        }
        let mut sinks = timing.successors[driver].clone();
        for bit in &bits {
            sinks.extend(timing.capture_consumers[*bit].iter().copied());
        }
        sinks.sort_unstable();
        sinks.dedup();
        for sink in sinks {
            if critical_set.contains(&sink) {
                continue;
            }
            let relief =
                critical_driver_load_relief(timing, sink, &bits, library, catalog, registers);
            if relief <= options.improvement_epsilon {
                continue;
            }
            load_sensitive
                .entry(sink)
                .and_modify(|existing| {
                    existing.0 = existing.0.max(relief);
                    existing.1 = existing.1.min(rank);
                })
                .or_insert((relief, rank));
        }
    }
    let mut siblings = load_sensitive.into_iter().collect::<Vec<_>>();
    siblings.sort_by(|(lhs_index, lhs), (rhs_index, rhs)| {
        rhs.0
            .total_cmp(&lhs.0)
            .then_with(|| lhs.1.cmp(&rhs.1))
            .then_with(|| lhs_index.cmp(rhs_index))
    });

    let mut result = Vec::new();
    let mut critical_indices = critical.iter().map(|candidate| candidate.instance_index);
    let mut sibling_indices = siblings.into_iter().map(|(index, _)| index);
    while result.len() < options.max_evaluations_per_iteration {
        let mut added = false;
        for _ in 0..3 {
            if result.len() == options.max_evaluations_per_iteration {
                break;
            }
            if let Some(index) = critical_indices.next() {
                result.push(index);
                added = true;
            }
        }
        if result.len() < options.max_evaluations_per_iteration
            && let Some(index) = sibling_indices.next()
        {
            result.push(index);
            added = true;
        }
        if !added {
            break;
        }
    }
    result
}

/// Gives swaps and each drive-strength direction fair per-instance coverage.
fn timing_move_queues(
    timing: &mut IncrementalRegisteredSta<'_>,
    critical: &[CriticalInstance],
    library: &Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
    options: &ResizeOptions,
    pin_history: &mut PinAssignmentHistory,
    pin_only: bool,
) -> Vec<TimingMoveQueue> {
    let indices =
        prioritized_timing_instances(timing, critical, library, catalog, registers, options);
    let mut result = Vec::new();
    for instance_index in indices
        .into_iter()
        .take(options.max_evaluations_per_iteration)
    {
        let instance = &timing.instances[instance_index];
        let current_index = instance.cell_index;
        let current_area = library.cells[current_index].area;
        let mut groups = [
            VecDeque::new(),
            VecDeque::new(),
            VecDeque::new(),
            VecDeque::new(),
        ];

        if !instance.sequential
            && let Some(classified) = catalog.by_name(library.cells[current_index].name.as_str())
        {
            let current_assignment = PinAssignmentKey::current(timing, instance_index);
            pin_history
                .visited_assignments
                .insert(current_assignment.clone());
            for pair in &classified.symmetric_input_pairs {
                let first_name = &classified.family.input_names[pair.first_input];
                let second_name = &classified.family.input_names[pair.second_input];
                let Some(first_input) = instance
                    .inputs
                    .iter()
                    .position(|input| input.name == *first_name)
                else {
                    continue;
                };
                let Some(second_input) = instance
                    .inputs
                    .iter()
                    .position(|input| input.name == *second_name)
                else {
                    continue;
                };
                let swapped =
                    PinAssignmentKey::swapped(timing, instance_index, first_input, second_input);
                if swapped == current_assignment
                    || pin_history.visited_assignments.contains(&swapped)
                {
                    continue;
                }
                groups[1].push_back(SizingMoveKind::SwapPins {
                    first_input,
                    second_input,
                });
            }
        }

        if !pin_only {
            for candidate in
                ordered_size_alternatives(timing, instance_index, library, catalog, registers)
            {
                let alternative_area = library.cells[candidate.cell_index].area;
                let group = if (alternative_area - current_area).abs() <= options.area_epsilon {
                    2
                } else if alternative_area > current_area {
                    0
                } else {
                    3
                };
                groups[group].push_back(SizingMoveKind::Resize {
                    cell_index: candidate.cell_index,
                });
            }
        }

        let mut moves = VecDeque::new();
        if !pin_only {
            for group_index in [2, 0, 3] {
                while moves.len() < options.max_cell_candidates_per_instance {
                    let Some(kind) = groups[group_index].pop_front() else {
                        break;
                    };
                    moves.push_back(kind);
                }
            }
        }
        let mut pin_candidates = 0usize;
        while pin_candidates < options.max_cell_candidates_per_instance {
            let Some(kind) = groups[1].pop_front() else {
                break;
            };
            moves.push_back(kind);
            pin_candidates += 1;
        }
        if !moves.is_empty() {
            result.push(TimingMoveQueue {
                instance_index,
                moves,
            });
        }
    }
    result
}

/// Records exact-trial work independently from accepted optimization moves.
fn record_optimization_evaluation(stats: &mut ResizeStats, kind: SizingMoveKind) {
    stats.evaluations += 1;
    if matches!(kind, SizingMoveKind::SwapPins { .. }) {
        stats.pin_swap_evaluations += 1;
    }
}

/// Accepts a real endpoint improvement or bounded, nonregressing local work.
fn timing_trial_is_acceptable(
    trial: &SizingTrial,
    previous: BoundaryTimingScore,
    best: BoundaryTimingScore,
    options: &ResizeOptions,
    allow_exploration: bool,
) -> bool {
    if !trial.constraints_satisfied {
        return false;
    }
    let improves_timing = trial
        .score
        .improvement_over(previous, options.improvement_epsilon)
        .is_some();
    let improves_local_timing = trial.local_improvement > options.improvement_epsilon;
    improves_timing
        || (trial
            .score
            .no_worse_than(previous, options.improvement_epsilon)
            && improves_local_timing)
        || (allow_exploration
            && improves_local_timing
            && trial.score.no_worse_than(
                best,
                options
                    .improvement_epsilon
                    .max(best.worst_delay() * MAX_EXPLORATORY_TIMING_REGRESSION),
            ))
}

/// Times and ranks one legal move without mutating the accepted solution.
#[allow(clippy::too_many_arguments)]
fn evaluate_timing_candidate(
    timing: &mut IncrementalRegisteredSta<'_>,
    instance_index: usize,
    kind: SizingMoveKind,
    library: &Library,
    catalog: &CellCatalog,
    options: &ResizeOptions,
    stats: &mut ResizeStats,
    score: BoundaryTimingScore,
    best: BoundaryTimingScore,
    area: f64,
) -> Option<SizingMove> {
    record_optimization_evaluation(stats, kind);
    let trial = match timing.evaluate_optimization_move(instance_index, kind, catalog, false) {
        Ok(trial) => trial,
        Err(error) => {
            stats.failed_evaluations += 1;
            log::debug!("rejecting incremental cell optimization trial: {error:#}");
            return None;
        }
    };
    stats.recomputed_instances += trial.recomputed_instances;
    if !timing_trial_is_acceptable(
        &trial,
        score,
        best,
        options,
        matches!(kind, SizingMoveKind::Resize { .. }),
    ) {
        return None;
    }
    let current = &library.cells[timing.instances[instance_index].cell_index];
    let new_area = match kind {
        SizingMoveKind::Resize { cell_index } => {
            area - current.area + library.cells[cell_index].area
        }
        SizingMoveKind::SwapPins { .. } => area,
    };
    Some(SizingMove {
        instance_index,
        kind,
        score: trial.score,
        ranking: trial
            .score
            .improvement_over(score, options.improvement_epsilon)
            .unwrap_or(trial.local_improvement),
        area: new_area,
    })
}

/// Preserves the best exact candidate found for each physical instance.
fn retain_timing_candidate(
    best_moves: &mut BTreeMap<usize, SizingMove>,
    proposed: SizingMove,
    library: &Library,
    options: &ResizeOptions,
) {
    let preferred = best_moves
        .get(&proposed.instance_index)
        .is_none_or(|existing| {
            compare_delay_moves(&proposed, existing, library, options) == std::cmp::Ordering::Less
        });
    if preferred {
        best_moves.insert(proposed.instance_index, proposed);
    }
}

/// Applies inexpensive broad sizing waves before exact per-cell refinement.
#[allow(clippy::too_many_arguments)]
fn apply_coordinated_timing_waves<'a>(
    module: &mut NetlistModule,
    nets: &[Net],
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &'a Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
    options: &ResizeOptions,
    constraints: &BufferTimingConstraints,
    timing: &mut IncrementalRegisteredSta<'a>,
    pin_history: &mut PinAssignmentHistory,
    score: &mut BoundaryTimingScore,
    area: &mut f64,
    stats: &mut ResizeStats,
    best_solution: &mut BestSizingState,
) -> Result<()> {
    let mut visited_sizes = timing
        .instances
        .iter()
        .enumerate()
        .map(|(index, instance)| (index, instance.cell_index))
        .collect::<BTreeSet<_>>();
    let mut stagnant_waves = 0usize;
    for _ in 0..options
        .max_iterations
        .saturating_mul(COORDINATED_SIZING_WAVE_MULTIPLIER)
    {
        let critical = timing
            .critical_window_instances(options.max_candidate_paths, INITIAL_CRITICAL_WINDOW)?;
        if critical.len() < MIN_COORDINATED_SIZING_CONE {
            break;
        }
        let critical_indices = critical
            .iter()
            .map(|candidate| candidate.instance_index)
            .collect::<BTreeSet<_>>();
        let mut prioritized =
            prioritized_timing_instances(timing, &critical, library, catalog, registers, options);
        let mut prioritized_indices = prioritized.iter().copied().collect::<BTreeSet<_>>();
        let maximum_candidates = options
            .max_evaluations_per_iteration
            .saturating_mul(COORDINATED_CANDIDATE_MULTIPLIER);
        for candidate in &critical {
            if prioritized.len() >= maximum_candidates {
                break;
            }
            if prioritized_indices.insert(candidate.instance_index) {
                prioritized.push(candidate.instance_index);
            }
        }
        let mut proposals = Vec::new();
        for instance_index in prioritized {
            let current_index = timing.instances[instance_index].cell_index;
            let current = &library.cells[current_index];
            let Ok(current_arrival) =
                timing.estimate_replacement_arrival(instance_index, current_index)
            else {
                continue;
            };
            let Ok(current_input) = average_functional_input_capacitance(library, current) else {
                continue;
            };
            let on_critical_path = critical_indices.contains(&instance_index);
            let mut preferred = None::<(EstimatedCellAlternative, f64)>;
            for candidate in
                ordered_size_alternatives(timing, instance_index, library, catalog, registers)
            {
                if visited_sizes.contains(&(instance_index, candidate.cell_index)) {
                    continue;
                }
                let candidate_cell = &library.cells[candidate.cell_index];
                let ranking = if on_critical_path {
                    if candidate_cell.area + options.area_epsilon < current.area
                        || candidate.arrival + options.improvement_epsilon >= current_arrival
                        || candidate.input_load
                            > current_input * MAX_COORDINATED_INPUT_GROWTH
                                + TIMING_VERIFICATION_EPSILON
                    {
                        continue;
                    }
                    (current_arrival - candidate.arrival)
                        / (1.0 + (candidate.input_load - current_input).max(0.0))
                } else {
                    if candidate.input_load + TIMING_VERIFICATION_EPSILON >= current_input {
                        continue;
                    }
                    current_input - candidate.input_load
                };
                if preferred
                    .as_ref()
                    .is_none_or(|(existing, existing_ranking)| {
                        ranking > *existing_ranking + options.improvement_epsilon
                            || ((ranking - *existing_ranking).abs() <= options.improvement_epsilon
                                && candidate_cell.area
                                    < library.cells[existing.cell_index].area
                                        - options.area_epsilon)
                    })
                {
                    preferred = Some((candidate, ranking));
                }
            }
            let Some((candidate, ranking)) = preferred else {
                continue;
            };
            proposals.push(SizingMove {
                instance_index,
                kind: SizingMoveKind::Resize {
                    cell_index: candidate.cell_index,
                },
                score: *score,
                ranking,
                area: *area - current.area + library.cells[candidate.cell_index].area,
            });
        }
        if proposals.is_empty() {
            break;
        }
        proposals.sort_by(|lhs, rhs| {
            rhs.ranking
                .total_cmp(&lhs.ranking)
                .then_with(|| lhs.instance_index.cmp(&rhs.instance_index))
        });
        let batch_limit = critical
            .len()
            .div_ceil(SIZING_BATCH_DIVISOR)
            .min(options.max_evaluations_per_iteration)
            .max(1);
        let mut selected = timing.independent_move_batch(proposals, batch_limit);
        let mut accepted = false;
        while !selected.is_empty() {
            let previous_score = *score;
            let previous_area = *area;
            let mut changes = Vec::with_capacity(selected.len());
            for candidate in &selected {
                let SizingMoveKind::Resize { cell_index } = candidate.kind else {
                    // Coordinated waves intentionally never change pin assignments.
                    continue;
                };
                let old_cell_index = timing.instances[candidate.instance_index].cell_index;
                module.instances[candidate.instance_index].type_name =
                    interner.get_or_intern(library.cells[cell_index].name.as_str());
                changes.push((candidate.instance_index, old_cell_index, cell_index));
            }
            stats.evaluations += 1;
            let trial = IncrementalRegisteredSta::new(
                module,
                nets,
                interner,
                library,
                options.sta_options,
                constraints,
            );
            let accepted_trial = trial.ok().filter(|candidate| {
                let candidate_score = candidate.score();
                candidate.satisfies_constraints(candidate_score)
                    && candidate_score.no_worse_than(
                        best_solution.score,
                        options.improvement_epsilon.max(
                            best_solution.score.worst_delay() * MAX_EXPLORATORY_TIMING_REGRESSION,
                        ),
                    )
            });
            let Some(candidate_timing) = accepted_trial else {
                stats.failed_evaluations += 1;
                for (instance_index, old_cell_index, _) in &changes {
                    module.instances[*instance_index].type_name =
                        interner.get_or_intern(library.cells[*old_cell_index].name.as_str());
                }
                selected.truncate(selected.len() / 2);
                continue;
            };
            let candidate_score = candidate_timing.score();
            stats.recomputed_instances += candidate_timing.instances.len();
            *timing = candidate_timing;
            *score = candidate_score;
            for (instance_index, old_cell_index, new_cell_index) in changes {
                let old_cell = &library.cells[old_cell_index];
                let new_cell = &library.cells[new_cell_index];
                *area += new_cell.area - old_cell.area;
                let instance = interner
                    .resolve(module.instances[instance_index].instance_name)
                    .ok_or_else(|| anyhow!("cannot resolve coordinated sizing instance"))?
                    .to_string();
                stats.replacements.push(ResizeStep {
                    instance,
                    old_cell: old_cell.name.clone(),
                    new_cell: new_cell.name.clone(),
                    delay_before: previous_score.worst_delay(),
                    delay_after: candidate_score.worst_delay(),
                    area_before: previous_area,
                    area_after: *area,
                });
                if new_cell.area > old_cell.area + options.area_epsilon {
                    stats.upsizes += 1;
                    stats.register_upsizes +=
                        usize::from(timing.instances[instance_index].sequential);
                } else if new_cell.area + options.area_epsilon < old_cell.area {
                    stats.downsizes += 1;
                    stats.register_downsizes +=
                        usize::from(timing.instances[instance_index].sequential);
                }
                visited_sizes.insert((instance_index, new_cell_index));
            }
            stats.final_clock_load = timing.has_registers().then_some(timing.clock_load);
            let previous_best = best_solution.score;
            best_solution.update_if_better(
                module,
                timing,
                *score,
                *area,
                stats,
                options,
                pin_history,
            );
            if best_solution
                .score
                .improvement_over(previous_best, options.improvement_epsilon)
                .is_some()
            {
                stagnant_waves = 0;
            } else {
                stagnant_waves += 1;
            }
            accepted = true;
            break;
        }
        if !accepted || stagnant_waves >= MAX_COORDINATED_STAGNANT_WAVES {
            break;
        }
    }
    Ok(())
}

/// Alternates exact pin and cell trials fairly across the critical window.
#[allow(clippy::too_many_arguments)]
fn optimize_timing_moves(
    module: &mut NetlistModule,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
    options: &ResizeOptions,
    timing: &mut IncrementalRegisteredSta<'_>,
    pin_history: &mut PinAssignmentHistory,
    score: &mut BoundaryTimingScore,
    area: &mut f64,
    stats: &mut ResizeStats,
    best_solution: &mut BestSizingState,
    max_rounds: usize,
    pin_only: bool,
) -> Result<usize> {
    let mut critical_window = INITIAL_CRITICAL_WINDOW;
    let mut sizing_rounds = 0usize;
    let mut total_committed = 0usize;
    let mut visited_sizes = timing
        .instances
        .iter()
        .enumerate()
        .map(|(index, instance)| (index, instance.cell_index))
        .collect::<BTreeSet<_>>();
    while sizing_rounds < max_rounds {
        let critical =
            timing.critical_window_instances(options.max_candidate_paths, critical_window)?;
        let queues = timing_move_queues(
            timing,
            &critical,
            library,
            catalog,
            registers,
            options,
            pin_history,
            pin_only,
        );
        let mut best_moves = BTreeMap::<usize, SizingMove>::new();
        if !pin_only {
            let mut resize_queues = queues
                .iter()
                .filter_map(|queue| {
                    let moves = queue
                        .moves
                        .iter()
                        .copied()
                        .filter(|kind| matches!(kind, SizingMoveKind::Resize { .. }))
                        .collect::<VecDeque<_>>();
                    (!moves.is_empty()).then_some(TimingMoveQueue {
                        instance_index: queue.instance_index,
                        moves,
                    })
                })
                .collect::<Vec<_>>();
            let mut evaluations = 0usize;
            while evaluations < options.max_evaluations_per_iteration {
                let mut attempted = false;
                for queue in &mut resize_queues {
                    if evaluations == options.max_evaluations_per_iteration {
                        break;
                    }
                    let kind = loop {
                        let Some(kind) = queue.moves.pop_front() else {
                            break None;
                        };
                        let SizingMoveKind::Resize { cell_index } = kind else {
                            // Sizing queues contain only physical substitutions.
                            continue;
                        };
                        if !visited_sizes.contains(&(queue.instance_index, cell_index)) {
                            break Some(kind);
                        }
                    };
                    let Some(kind) = kind else {
                        continue;
                    };
                    attempted = true;
                    evaluations += 1;
                    if let Some(candidate) = evaluate_timing_candidate(
                        timing,
                        queue.instance_index,
                        kind,
                        library,
                        catalog,
                        options,
                        stats,
                        *score,
                        best_solution.score,
                        *area,
                    ) {
                        retain_timing_candidate(&mut best_moves, candidate, library, options);
                    }
                }
                if !attempted {
                    break;
                }
            }
        }

        let swap_budget = if pin_only {
            options.max_evaluations_per_iteration
        } else {
            options.max_evaluations_per_iteration.div_ceil(4).max(1)
        };
        let mut pin_queues = queues
            .iter()
            .filter_map(|queue| {
                let moves = queue
                    .moves
                    .iter()
                    .copied()
                    .filter(|kind| matches!(kind, SizingMoveKind::SwapPins { .. }))
                    .collect::<VecDeque<_>>();
                (!moves.is_empty()).then_some(TimingMoveQueue {
                    instance_index: queue.instance_index,
                    moves,
                })
            })
            .collect::<Vec<_>>();
        let mut pin_evaluations = 0usize;
        while pin_evaluations < swap_budget {
            let mut attempted = false;
            for queue in &mut pin_queues {
                if pin_evaluations == swap_budget {
                    break;
                }
                let Some(kind) = queue.moves.pop_front() else {
                    continue;
                };
                attempted = true;
                pin_evaluations += 1;
                if let Some(candidate) = evaluate_timing_candidate(
                    timing,
                    queue.instance_index,
                    kind,
                    library,
                    catalog,
                    options,
                    stats,
                    *score,
                    best_solution.score,
                    *area,
                ) {
                    retain_timing_candidate(&mut best_moves, candidate, library, options);
                }
            }
            if !attempted {
                break;
            }
        }

        let mut candidates = best_moves.into_values().collect::<Vec<_>>();
        if candidates.is_empty() {
            if pin_only || critical_window >= MAX_CRITICAL_WINDOW {
                break;
            }
            critical_window = (critical_window * 2.0).min(MAX_CRITICAL_WINDOW);
            continue;
        }
        candidates.sort_by(|lhs, rhs| compare_delay_moves(lhs, rhs, library, options));
        let batch_limit = critical.len().div_ceil(SIZING_BATCH_DIVISOR).max(1);
        let selected = timing.independent_move_batch(candidates, batch_limit);
        let mut committed = 0usize;
        let round_score = *score;
        for mut candidate in selected {
            if let SizingMoveKind::SwapPins {
                first_input,
                second_input,
            } = candidate.kind
            {
                let target = PinAssignmentKey::swapped(
                    timing,
                    candidate.instance_index,
                    first_input,
                    second_input,
                );
                if pin_history.visited_assignments.contains(&target) {
                    continue;
                }
            }
            record_optimization_evaluation(stats, candidate.kind);
            let trial = match timing.evaluate_optimization_move(
                candidate.instance_index,
                candidate.kind,
                catalog,
                false,
            ) {
                Ok(trial) => trial,
                Err(error) => {
                    stats.failed_evaluations += 1;
                    log::debug!("rejecting revalidated cell optimization: {error:#}");
                    continue;
                }
            };
            stats.recomputed_instances += trial.recomputed_instances;
            if !timing_trial_is_acceptable(
                &trial,
                *score,
                best_solution.score,
                options,
                matches!(candidate.kind, SizingMoveKind::Resize { .. }),
            ) {
                continue;
            }
            let current = &library.cells[timing.instances[candidate.instance_index].cell_index];
            candidate.score = trial.score;
            candidate.area = match candidate.kind {
                SizingMoveKind::Resize { cell_index } => {
                    *area - current.area + library.cells[cell_index].area
                }
                SizingMoveKind::SwapPins { .. } => *area,
            };
            let visited_size = match candidate.kind {
                SizingMoveKind::Resize { cell_index } => {
                    Some((candidate.instance_index, cell_index))
                }
                SizingMoveKind::SwapPins { .. } => None,
            };
            commit_registered_move(
                module,
                interner,
                library,
                catalog,
                timing,
                pin_history,
                stats,
                score,
                area,
                candidate,
            )?;
            if let Some(visited_size) = visited_size {
                visited_sizes.insert(visited_size);
            }
            committed += 1;
            total_committed += 1;
            best_solution.update_if_better(
                module,
                timing,
                *score,
                *area,
                stats,
                options,
                pin_history,
            );
        }
        if committed == 0 {
            if pin_only || critical_window >= MAX_CRITICAL_WINDOW {
                break;
            }
            critical_window = (critical_window * 2.0).min(MAX_CRITICAL_WINDOW);
            continue;
        }
        sizing_rounds += 1;
        critical_window = if score
            .improvement_over(round_score, options.improvement_epsilon)
            .is_some()
        {
            INITIAL_CRITICAL_WINDOW
        } else {
            (critical_window * 2.0).min(MAX_CRITICAL_WINDOW)
        };
    }
    Ok(total_committed)
}

/// Recovers gate, buffer, and FF area without worsening any timing boundary.
#[allow(clippy::too_many_arguments)]
fn recover_timing_protected_area(
    module: &mut NetlistModule,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
    options: &ResizeOptions,
    timing: &mut IncrementalRegisteredSta<'_>,
    pin_history: &mut PinAssignmentHistory,
    score: &mut BoundaryTimingScore,
    area: &mut f64,
    stats: &mut ResizeStats,
    achieved: BoundaryTimingScore,
) -> Result<usize> {
    let mut total_committed = 0usize;
    let achieved_secondary = timing.secondary_delay();
    for _ in 0..options.max_area_iterations {
        let recoverable = area_recovery_instances(timing, library, catalog, registers, options);
        let batch_limit = recoverable.len().div_ceil(SIZING_BATCH_DIVISOR).max(1);
        let mut queues = recoverable
            .into_iter()
            .take(options.max_evaluations_per_iteration)
            .filter_map(|instance_index| {
                let current = &library.cells[timing.instances[instance_index].cell_index];
                let current_area = current.area;
                let current_input = average_functional_input_capacitance(library, current).ok()?;
                let required_input = timing.instances[instance_index]
                    .outputs
                    .iter()
                    .map(|output| max_load(timing.loads[output.bit]))
                    .fold(0.0_f64, f64::max)
                    / MAX_ELECTRICAL_EFFORT;
                let moves =
                    ordered_size_alternatives(timing, instance_index, library, catalog, registers)
                        .into_iter()
                        .filter(|candidate| {
                            library.cells[candidate.cell_index].area + options.area_epsilon
                                < current_area
                                && candidate.input_load + TIMING_VERIFICATION_EPSILON
                                    >= current_input.min(required_input)
                        })
                        .take(options.max_cell_candidates_per_instance)
                        .map(|candidate| SizingMoveKind::Resize {
                            cell_index: candidate.cell_index,
                        })
                        .collect::<VecDeque<_>>();
                (!moves.is_empty()).then_some(TimingMoveQueue {
                    instance_index,
                    moves,
                })
            })
            .collect::<Vec<_>>();
        let mut best_moves = BTreeMap::<usize, SizingMove>::new();
        let mut evaluations = 0usize;
        while evaluations < options.max_evaluations_per_iteration {
            let mut attempted = false;
            for queue in &mut queues {
                if evaluations == options.max_evaluations_per_iteration {
                    break;
                }
                let Some(kind) = queue.moves.pop_front() else {
                    continue;
                };
                attempted = true;
                evaluations += 1;
                record_optimization_evaluation(stats, kind);
                let trial = match timing.evaluate_optimization_move(
                    queue.instance_index,
                    kind,
                    catalog,
                    false,
                ) {
                    Ok(trial) => trial,
                    Err(error) => {
                        stats.failed_evaluations += 1;
                        log::debug!("rejecting timing-protected area recovery: {error:#}");
                        continue;
                    }
                };
                stats.recomputed_instances += trial.recomputed_instances;
                if !trial
                    .score
                    .no_worse_than(achieved, options.improvement_epsilon)
                    || trial.secondary_delay > achieved_secondary + options.improvement_epsilon
                    || !trial.constraints_satisfied
                {
                    continue;
                }
                let SizingMoveKind::Resize { cell_index } = kind else {
                    // Area recovery only enumerates smaller cell substitutions.
                    continue;
                };
                let current = &library.cells[timing.instances[queue.instance_index].cell_index];
                let candidate = &library.cells[cell_index];
                let proposed = SizingMove {
                    instance_index: queue.instance_index,
                    kind,
                    score: trial.score,
                    ranking: current.area - candidate.area,
                    area: *area - current.area + candidate.area,
                };
                let preferred = best_moves
                    .get(&queue.instance_index)
                    .is_none_or(|existing| {
                        compare_area_moves(&proposed, existing, library, options)
                            == std::cmp::Ordering::Less
                    });
                if preferred {
                    best_moves.insert(queue.instance_index, proposed);
                }
            }
            if !attempted {
                break;
            }
        }
        let mut candidates = best_moves.into_values().collect::<Vec<_>>();
        if candidates.is_empty() {
            break;
        }
        candidates.sort_by(|lhs, rhs| compare_area_moves(lhs, rhs, library, options));
        let selected = timing.independent_move_batch(candidates, batch_limit);
        let mut committed = 0usize;
        for mut candidate in selected {
            record_optimization_evaluation(stats, candidate.kind);
            let trial = match timing.evaluate_optimization_move(
                candidate.instance_index,
                candidate.kind,
                catalog,
                false,
            ) {
                Ok(trial) => trial,
                Err(error) => {
                    stats.failed_evaluations += 1;
                    log::debug!("rejecting revalidated area recovery: {error:#}");
                    continue;
                }
            };
            stats.recomputed_instances += trial.recomputed_instances;
            if !trial
                .score
                .no_worse_than(achieved, options.improvement_epsilon)
                || trial.secondary_delay > achieved_secondary + options.improvement_epsilon
                || !trial.constraints_satisfied
            {
                continue;
            }
            let SizingMoveKind::Resize { cell_index } = candidate.kind else {
                // Timing-safe recovery never commits a zero-area pin exchange.
                continue;
            };
            let current = &library.cells[timing.instances[candidate.instance_index].cell_index];
            candidate.score = trial.score;
            candidate.area = *area - current.area + library.cells[cell_index].area;
            commit_registered_move(
                module,
                interner,
                library,
                catalog,
                timing,
                pin_history,
                stats,
                score,
                area,
                candidate,
            )?;
            committed += 1;
            total_committed += 1;
        }
        if committed == 0 {
            break;
        }
    }
    Ok(total_committed)
}

/// Prioritizes high-savings, noncritical cells during timing-safe downsizing.
fn area_recovery_instances(
    timing: &IncrementalRegisteredSta<'_>,
    library: &Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
    options: &ResizeOptions,
) -> Vec<usize> {
    let mut candidates = Vec::new();
    for (index, instance) in timing.instances.iter().enumerate() {
        let current = &library.cells[instance.cell_index];
        let saving = size_alternatives(timing, index, library, catalog, registers)
            .into_iter()
            .map(|cell| (current.area - library.cells[cell].area).max(0.0))
            .fold(0.0, f64::max);
        if saving > options.area_epsilon {
            candidates.push((index, saving));
        }
    }
    candidates.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
    candidates.into_iter().map(|(index, _)| index).collect()
}

/// Commits one validated gate, buffer, or physical flip-flop replacement.
#[allow(clippy::too_many_arguments)]
fn commit_registered_move(
    module: &mut NetlistModule,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    catalog: &CellCatalog,
    timing: &mut IncrementalRegisteredSta<'_>,
    pin_history: &mut PinAssignmentHistory,
    stats: &mut ResizeStats,
    score: &mut BoundaryTimingScore,
    area: &mut f64,
    selected: SizingMove,
) -> Result<()> {
    let instance = &timing.instances[selected.instance_index];
    let old_cell = &library.cells[instance.cell_index];
    let new_cell_index = match selected.kind {
        SizingMoveKind::Resize { cell_index } => cell_index,
        SizingMoveKind::SwapPins { .. } => instance.cell_index,
    };
    let new_cell = &library.cells[new_cell_index];
    let previous_delay = score.worst_delay();
    let previous_area = *area;
    let is_register = instance.sequential;
    let is_upsize = new_cell.area > old_cell.area;
    let instance_name = interner
        .resolve(module.instances[selected.instance_index].instance_name)
        .ok_or_else(|| anyhow!("cannot resolve register-aware sized instance"))?
        .to_string();
    let swap_connections = match selected.kind {
        SizingMoveKind::Resize { .. } => None,
        SizingMoveKind::SwapPins {
            first_input,
            second_input,
        } => {
            let first_pin = instance
                .inputs
                .get(first_input)
                .ok_or_else(|| anyhow!("committed first swap input is out of bounds"))?
                .name
                .clone();
            let second_pin = instance
                .inputs
                .get(second_input)
                .ok_or_else(|| anyhow!("committed second swap input is out of bounds"))?
                .name
                .clone();
            let connections = &module.instances[selected.instance_index].connections;
            let first_connection = connections
                .iter()
                .position(|(port, _)| interner.resolve(*port) == Some(first_pin.as_str()))
                .ok_or_else(|| anyhow!("missing committed pin-swap connection '{first_pin}'"))?;
            let second_connection = connections
                .iter()
                .position(|(port, _)| interner.resolve(*port) == Some(second_pin.as_str()))
                .ok_or_else(|| anyhow!("missing committed pin-swap connection '{second_pin}'"))?;
            pin_history
                .original_connections
                .entry(selected.instance_index)
                .or_insert_with(|| connections.clone());
            pin_history
                .visited_assignments
                .insert(PinAssignmentKey::current(timing, selected.instance_index));
            Some((first_connection, second_connection, first_pin, second_pin))
        }
    };
    let committed =
        timing.evaluate_optimization_move(selected.instance_index, selected.kind, catalog, true)?;
    stats.recomputed_instances += committed.recomputed_instances;
    if let Some((first, second, first_pin, second_pin)) = swap_connections {
        let connections = &mut module.instances[selected.instance_index].connections;
        let first_reference = connections[first].1.clone();
        connections[first].1 = connections[second].1.clone();
        connections[second].1 = first_reference;
        pin_history
            .visited_assignments
            .insert(PinAssignmentKey::current(timing, selected.instance_index));
        stats.pin_swaps += 1;
        stats.pin_swap_steps.push(PinSwapStep {
            instance: instance_name,
            cell: old_cell.name.clone(),
            first_pin,
            second_pin,
            delay_before: previous_delay,
            delay_after: committed.score.worst_delay(),
        });
    } else {
        module.instances[selected.instance_index].type_name =
            interner.get_or_intern(new_cell.name.as_str());
        stats.replacements.push(ResizeStep {
            instance: instance_name,
            old_cell: old_cell.name.clone(),
            new_cell: new_cell.name.clone(),
            delay_before: previous_delay,
            delay_after: committed.score.worst_delay(),
            area_before: previous_area,
            area_after: selected.area,
        });
    }
    *score = committed.score;
    *area = selected.area;
    if matches!(selected.kind, SizingMoveKind::Resize { .. }) && is_upsize {
        stats.upsizes += 1;
        stats.register_upsizes += usize::from(is_register);
    } else if matches!(selected.kind, SizingMoveKind::Resize { .. })
        && new_cell.area < old_cell.area
    {
        stats.downsizes += 1;
        stats.register_downsizes += usize::from(is_register);
    }
    stats.final_clock_load = timing.has_registers().then_some(committed.clock_load);
    Ok(())
}

/// Repairs severe driver effort before narrower critical-path optimization.
#[allow(clippy::too_many_arguments)]
fn apply_electrical_sizing(
    module: &mut NetlistModule,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    catalog: &CellCatalog,
    registers: &RegisterCellCatalog,
    options: &ResizeOptions,
    timing: &mut IncrementalRegisteredSta<'_>,
    pin_history: &mut PinAssignmentHistory,
    score: &mut BoundaryTimingScore,
    area: &mut f64,
    stats: &mut ResizeStats,
) -> Result<()> {
    let mut order = timing
        .critical_window_instances(options.max_candidate_paths, INITIAL_CRITICAL_WINDOW)?
        .into_iter()
        .map(|candidate| candidate.instance_index)
        .collect::<Vec<_>>();
    let mut seen = order.iter().copied().collect::<BTreeSet<_>>();
    for index in timing.topological_order.iter().rev().copied() {
        if seen.insert(index) {
            order.push(index);
        }
    }
    let mut candidates = Vec::new();
    for (priority, instance_index) in order.into_iter().enumerate() {
        let instance = &timing.instances[instance_index];
        let current = &library.cells[instance.cell_index];
        let Some(output) = instance.outputs.first() else {
            continue;
        };
        let output_load = max_load(timing.loads[output.bit]);
        let current_input = average_functional_input_capacitance(library, current)?;
        let effort_ratio = if current_input > TIMING_VERIFICATION_EPSILON {
            output_load / (current_input * MAX_ELECTRICAL_EFFORT)
        } else {
            0.0
        };
        let output_pin = current.pins.iter().find(|pin| {
            pin.direction == PinDirection::Output as i32
                && library.resolve_string(&pin.name) == output.name
        });
        let characterized_ratio = output_pin
            .and_then(|pin| pin.max_capacitance)
            .filter(|limit| limit.is_finite() && *limit > 0.0)
            .map(|limit| output_load / limit)
            .unwrap_or(0.0);
        let severity = effort_ratio.max(characterized_ratio);
        if severity > 1.0 + TIMING_VERIFICATION_EPSILON {
            candidates.push((
                instance_index,
                characterized_ratio > 1.0,
                severity,
                priority,
            ));
        }
    }
    candidates.sort_by(|lhs, rhs| {
        rhs.1
            .cmp(&lhs.1)
            .then_with(|| rhs.2.total_cmp(&lhs.2))
            .then_with(|| lhs.3.cmp(&rhs.3))
            .then_with(|| lhs.0.cmp(&rhs.0))
    });

    let max_evaluations = options
        .max_evaluations_per_iteration
        .saturating_mul(ELECTRICAL_SIZING_BUDGET_MULTIPLIER);
    let mut evaluations = 0usize;
    for (instance_index, _, _, _) in candidates {
        if evaluations >= max_evaluations {
            break;
        }
        let instance = &timing.instances[instance_index];
        let current = &library.cells[instance.cell_index];
        let Some(output) = instance.outputs.first() else {
            continue;
        };
        let required_input = max_load(timing.loads[output.bit]) / MAX_ELECTRICAL_EFFORT;
        let preferred = size_alternatives(timing, instance_index, library, catalog, registers)
            .into_iter()
            .find(|cell_index| {
                let cell = &library.cells[*cell_index];
                let output_pin = cell.pins.iter().find(|pin| {
                    pin.direction == PinDirection::Output as i32
                        && library.resolve_string(&pin.name) == output.name
                });
                let output_is_legal = output_pin
                    .and_then(|pin| pin.max_capacitance)
                    .is_none_or(|limit| max_load(timing.loads[output.bit]) <= limit + 1e-9);
                output_is_legal
                    && average_functional_input_capacitance(library, cell)
                        .is_ok_and(|capacitance| capacitance + 1e-9 >= required_input)
            });
        let Some(cell_index) = preferred else {
            continue;
        };
        if cell_index == instance.cell_index {
            continue;
        }
        evaluations += 1;
        stats.evaluations += 1;
        let trial = match timing.evaluate_cell_substitution(instance_index, cell_index, false) {
            Ok(trial) => trial,
            Err(error) => {
                stats.failed_evaluations += 1;
                log::debug!("rejecting electrical register-aware resize: {error:#}");
                continue;
            }
        };
        stats.recomputed_instances += trial.recomputed_instances;
        let candidate = &library.cells[cell_index];
        let improves_delay = trial
            .score
            .improvement_over(*score, options.improvement_epsilon)
            .is_some();
        let preserves_delay = trial
            .score
            .no_worse_than(*score, options.improvement_epsilon);
        let improves_local_timing = trial.local_improvement > options.improvement_epsilon;
        if !trial.constraints_satisfied
            || !(improves_delay
                || (preserves_delay
                    && (improves_local_timing
                        || candidate.area + options.area_epsilon < current.area)))
        {
            continue;
        }
        commit_registered_move(
            module,
            interner,
            library,
            catalog,
            timing,
            pin_history,
            stats,
            score,
            area,
            SizingMove {
                instance_index,
                kind: SizingMoveKind::Resize { cell_index },
                score: trial.score,
                ranking: 0.0,
                area: *area - current.area + candidate.area,
            },
        )?;
    }
    Ok(())
}

/// Measures the actual characterized data input, not a drive-name suffix.
fn average_functional_input_capacitance(library: &Library, cell: &Cell) -> Result<f64> {
    let mut total = 0.0;
    let mut count = 0usize;
    for pin in cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Input as i32 && !pin.is_clocking_pin)
    {
        total += max_load(effective_input_capacitance_for_mapping(
            pin,
            &format!(
                "registered sizing cell '{}.{}'",
                cell.name,
                library.resolve_string(&pin.name)
            ),
        )?);
        count += 1;
    }
    Ok(if count == 0 {
        0.0
    } else {
        total / count as f64
    })
}

/// Incremental per-bit STA for both sides of synchronous register boundaries.
struct IncrementalRegisteredSta<'a> {
    library: &'a Library,
    instances: Vec<SizingInstance>,
    drivers: Vec<Option<usize>>,
    loads: Vec<CombinationalOutputLoad>,
    bit_timing: Vec<BoundarySignalTiming>,
    bit_predecessors: Vec<BoundaryPredecessors>,
    successors: Vec<Vec<usize>>,
    capture_consumers: Vec<Vec<usize>>,
    captures: Vec<RegisterCaptureScore>,
    capture_predecessors: Vec<CapturePredecessors>,
    outputs: Vec<OutputEndpoint>,
    topological_order: Vec<usize>,
    topological_positions: Vec<usize>,
    has_registers: bool,
    clock_load: f64,
    constraints: BufferTimingConstraints,
    diagnostics: TimingQueryDiagnosticCounts,
}

impl<'a> IncrementalRegisteredSta<'a> {
    /// Builds canonical bit connectivity and seeds each launch independently.
    fn new(
        module: &NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        library: &'a Library,
        options: StaOptions,
        constraints: &BufferTimingConstraints,
    ) -> Result<Self> {
        let normalized = NormalizedNetlistModule::new(module, nets, interner)
            .context("normalizing register-aware sizing connectivity")?;
        let by_name = library
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| (cell.name.as_str(), index))
            .collect::<HashMap<_, _>>();
        let mut drivers = vec![None; normalized.bit_count()];
        let mut loads = vec![CombinationalOutputLoad::default(); normalized.bit_count()];
        let mut instances = Vec::with_capacity(normalized.instances.len());
        let mut registers = Vec::new();
        let mut clock_load = 0.0;

        for (instance_index, instance) in normalized.instances.iter().enumerate() {
            let name = interner
                .resolve(instance.type_name)
                .ok_or_else(|| anyhow!("cannot resolve registered sizing cell"))?;
            let cell_index = *by_name
                .get(name)
                .ok_or_else(|| anyhow!("registered sizing references unknown cell '{name}'"))?;
            let cell = &library.cells[cell_index];
            let sequential = is_sequential_boundary_cell(cell);
            if sequential {
                registers.push(instance_index);
            }
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            let mut known_pin_values = HashMap::new();
            for connection in &instance.connections {
                let pin_name = interner
                    .resolve(connection.port)
                    .ok_or_else(|| anyhow!("cannot resolve registered sizing pin"))?;
                let (pin_index, pin) = cell
                    .pins
                    .iter()
                    .enumerate()
                    .find(|(_, pin)| library.resolve_string(&pin.name) == pin_name)
                    .ok_or_else(|| anyhow!("cell '{name}' has no pin '{pin_name}'"))?;
                if connection.bits.len() > 1 {
                    bail!("registered sizing requires one-bit Liberty pin '{name}.{pin_name}'");
                }
                match (pin.direction, connection.bits.first().copied()) {
                    (direction, Some(BitSource::Bit(bit)))
                        if direction == PinDirection::Input as i32 =>
                    {
                        let capacitance = effective_input_capacitance_for_mapping(
                            pin,
                            &format!("registered sizing input '{name}.{pin_name}'"),
                        )?;
                        loads[bit].rise += capacitance.rise;
                        loads[bit].fall += capacitance.fall;
                        if pin.is_clocking_pin {
                            clock_load += max_load(capacitance);
                        }
                        inputs.push(SizingInput {
                            name: pin_name.to_string(),
                            bit: Some(bit),
                            clock: pin.is_clocking_pin,
                        });
                    }
                    (direction, Some(BitSource::Literal(value)))
                        if direction == PinDirection::Input as i32 =>
                    {
                        known_pin_values.insert(pin_name.to_string(), value);
                        inputs.push(SizingInput {
                            name: pin_name.to_string(),
                            bit: None,
                            clock: pin.is_clocking_pin,
                        });
                    }
                    (direction, Some(BitSource::Bit(bit)))
                        if direction == PinDirection::Output as i32 =>
                    {
                        if drivers[bit].replace(instance_index).is_some() {
                            bail!("register-aware sizing requires one driver per net bit");
                        }
                        outputs.push(SizingOutput {
                            name: pin_name.to_string(),
                            pin_index,
                            bit,
                        });
                    }
                    _ => {
                        // Unconnected pins and unknown literals carry no timed
                        // bit.
                    }
                }
            }
            inputs.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
            outputs.sort_by(|lhs, rhs| lhs.name.cmp(&rhs.name));
            instances.push(SizingInstance {
                cell_index,
                inputs,
                outputs,
                known_pin_values,
                sequential,
            });
        }

        let constant_bits = normalized
            .assigns
            .iter()
            .flat_map(|assign| assign.lhs_bits.iter().zip(&assign.rhs_bits))
            .filter_map(|(bit, expression)| {
                matches!(expression, BitExpr::Source(BitSource::Literal(_))).then_some(*bit)
            })
            .collect::<BTreeSet<_>>();
        let mut outputs = Vec::new();
        for port in &normalized.ports {
            if port.direction != PortDirection::Output {
                continue;
            }
            let name = interner
                .resolve(port.name)
                .ok_or_else(|| anyhow!("cannot resolve registered sizing output"))?;
            let net = module
                .find_net_index(port.name, nets)
                .ok_or_else(|| anyhow!("registered sizing output '{name}' has no net"))?;
            for (offset, bit) in port.bits.iter().copied().enumerate() {
                if constant_bits.contains(&bit) {
                    continue;
                }
                loads[bit].rise += options.module_output_load;
                loads[bit].fall += options.module_output_load;
                let bit_number = nets[net.0]
                    .bit_number(offset)
                    .ok_or_else(|| anyhow!("invalid registered sizing output bit '{name}'"))?;
                outputs.push(OutputEndpoint {
                    name: if nets[net.0].width_bits() == 1 {
                        name.to_string()
                    } else {
                        format!("{name}_{bit_number}")
                    },
                    bit,
                });
            }
        }

        let primary = if registers.is_empty() {
            analyze_combinational_max_arrival_with_primary_input_arrivals(
                module,
                nets,
                interner,
                library,
                options,
                &constraints.primary_input_arrivals,
            )?
        } else {
            analyze_register_boundary_max_arrival_with_primary_input_arrivals(
                module,
                nets,
                interner,
                library,
                options,
                true,
                &[],
                &constraints.primary_input_arrivals,
            )?
        };
        let register = if registers.is_empty() {
            None
        } else {
            Some(
                analyze_register_boundary_max_arrival_with_primary_input_arrivals(
                    module,
                    nets,
                    interner,
                    library,
                    options,
                    false,
                    registers.as_slice(),
                    &constraints.primary_input_arrivals,
                )?,
            )
        };

        let bit_timing = (0..normalized.bit_count())
            .map(|bit| BoundarySignalTiming {
                primary_input: primary.timing_for_bit(bit),
                register: register
                    .as_ref()
                    .and_then(|report| report.timing_for_bit(bit)),
            })
            .collect::<Vec<_>>();
        let captures = (0..instances.len())
            .map(|index| RegisterCaptureScore {
                primary_input: primary
                    .register_input_arrivals
                    .get(index)
                    .copied()
                    .flatten(),
                register: register
                    .as_ref()
                    .and_then(|report| report.register_input_arrivals.get(index))
                    .copied()
                    .flatten(),
            })
            .collect::<Vec<_>>();

        let mut successors = vec![Vec::new(); instances.len()];
        let mut capture_consumers = vec![Vec::new(); normalized.bit_count()];
        let mut indegrees = vec![0usize; instances.len()];
        for (index, instance) in instances.iter().enumerate() {
            if instance.sequential {
                for input in instance.inputs.iter().filter(|input| !input.clock) {
                    if let Some(bit) = input.bit {
                        capture_consumers[bit].push(index);
                    }
                }
                continue;
            }
            let mut predecessors = BTreeSet::new();
            for input in instance.inputs.iter().filter(|input| !input.clock) {
                if let Some(bit) = input.bit
                    && let Some(driver) = drivers[bit]
                    && driver != index
                {
                    predecessors.insert(driver);
                }
            }
            indegrees[index] = predecessors.len();
            for predecessor in predecessors {
                successors[predecessor].push(index);
            }
        }
        for successor in &mut successors {
            successor.sort_unstable();
        }
        for consumers in &mut capture_consumers {
            consumers.sort_unstable();
            consumers.dedup();
        }
        let mut ready = indegrees
            .iter()
            .enumerate()
            .filter_map(|(index, indegree)| (*indegree == 0).then_some(index))
            .collect::<BTreeSet<_>>();
        let mut topological_order = Vec::with_capacity(instances.len());
        while let Some(index) = ready.pop_first() {
            topological_order.push(index);
            for successor in &successors[index] {
                indegrees[*successor] -= 1;
                if indegrees[*successor] == 0 {
                    ready.insert(*successor);
                }
            }
        }
        if topological_order.len() != instances.len() {
            bail!("register-aware incremental sizing detected a combinational cycle");
        }
        let mut topological_positions = vec![0usize; instances.len()];
        for (position, instance) in topological_order.iter().copied().enumerate() {
            topological_positions[instance] = position;
        }

        let has_registers = !registers.is_empty();
        let mut state = Self {
            library,
            instances,
            drivers,
            loads,
            bit_timing,
            bit_predecessors: vec![BoundaryPredecessors::default(); normalized.bit_count()],
            successors,
            capture_consumers,
            captures,
            capture_predecessors: vec![CapturePredecessors::default(); normalized.instances.len()],
            outputs,
            topological_order,
            topological_positions,
            has_registers,
            clock_load,
            constraints: constraints.clone(),
            diagnostics: TimingQueryDiagnosticCounts::default(),
        };
        for position in 0..state.topological_order.len() {
            let instance = state.topological_order[position];
            state.recompute_instance(instance)?;
        }
        for instance in registers {
            state.recompute_capture(instance)?;
        }
        Ok(state)
    }

    /// Returns whether this timing graph contains physical register endpoints.
    fn has_registers(&self) -> bool {
        self.has_registers
    }

    /// Computes the exact register/input launch endpoint objective.
    fn score(&self) -> BoundaryTimingScore {
        BoundaryTimingScore {
            register_to_register: self
                .captures
                .iter()
                .filter_map(|capture| capture.register)
                .reduce(f64::max),
            input_to_register: self
                .captures
                .iter()
                .filter_map(|capture| capture.primary_input)
                .reduce(f64::max),
            register_to_output: self
                .outputs
                .iter()
                .filter_map(|output| self.bit_timing[output.bit].register)
                .map(signal_arrival)
                .reduce(f64::max),
            input_to_output: self
                .outputs
                .iter()
                .filter_map(|output| self.bit_timing[output.bit].primary_input)
                .map(signal_arrival)
                .reduce(f64::max),
        }
    }

    /// Ranks only a small, bounded set of near-critical physical endpoints.
    fn secondary_delay(&self) -> f64 {
        let mut arrivals = Vec::with_capacity(self.captures.len() * 2 + self.outputs.len() * 4);
        for capture in &self.captures {
            arrivals.extend(
                [capture.register, capture.primary_input]
                    .into_iter()
                    .flatten(),
            );
        }
        for output in &self.outputs {
            let timing = self.bit_timing[output.bit];
            for signal in [timing.register, timing.primary_input]
                .into_iter()
                .flatten()
            {
                arrivals.push(signal.rise.arrival);
                arrivals.push(signal.fall.arrival);
            }
        }
        arrivals.sort_unstable_by(|left, right| right.total_cmp(left));
        let threshold = self.score().worst_delay() * (1.0 - SECONDARY_CRITICAL_WINDOW);
        arrivals
            .into_iter()
            .filter(|arrival| *arrival + TIMING_VERIFICATION_EPSILON >= threshold)
            .take(MAX_SECONDARY_ENDPOINTS)
            .sum()
    }

    /// Estimates a replacement using the actual current slew and output load.
    fn estimate_replacement_arrival(
        &mut self,
        instance_index: usize,
        replacement_index: usize,
    ) -> Result<f64> {
        let instance = self
            .instances
            .get(instance_index)
            .ok_or_else(|| anyhow!("estimated sizing instance is out of bounds"))?;
        let cell = self
            .library
            .cells
            .get(replacement_index)
            .ok_or_else(|| anyhow!("estimated sizing replacement is out of bounds"))?;
        let mut arrival = 0.0_f64;
        for output in &instance.outputs {
            let pin = cell
                .pins
                .iter()
                .find(|pin| {
                    pin.direction == PinDirection::Output as i32
                        && self.library.resolve_string(&pin.name) == output.name
                })
                .ok_or_else(|| anyhow!("estimated replacement changes output '{}'", output.name))?;
            if pin.max_capacitance.is_some_and(|limit| {
                max_load(self.loads[output.bit]) > limit + TIMING_VERIFICATION_EPSILON
            }) {
                bail!(
                    "estimated replacement '{}.{}' cannot drive its actual output load",
                    cell.name,
                    output.name
                );
            }
            if instance.sequential {
                let timing = evaluate_sequential_cell_output_timing(
                    self.library,
                    cell.name.as_str(),
                    pin,
                    self.loads[output.bit],
                    &instance.known_pin_values,
                    &mut self.diagnostics,
                )?;
                arrival = arrival.max(signal_arrival(timing));
                continue;
            }
            for register_launch in [false, true] {
                let mut input_timings = Vec::new();
                for input in instance.inputs.iter().filter(|input| !input.clock) {
                    if let Some(bit) = input.bit {
                        if let Some(timing) = self.bit_timing[bit].for_launch(register_launch) {
                            input_timings.push((input.name.as_str(), timing));
                        }
                    } else if !self.has_registers
                        && !register_launch
                        && instance.known_pin_values.contains_key(&input.name)
                    {
                        input_timings.push((input.name.as_str(), literal_signal_timing()));
                    }
                }
                if input_timings.is_empty() {
                    continue;
                }
                let timing = evaluate_combinational_cell_output_timing_with_predecessors(
                    self.library,
                    cell.name.as_str(),
                    pin,
                    input_timings.as_slice(),
                    self.loads[output.bit],
                    &instance.known_pin_values,
                    &mut self.diagnostics,
                )?;
                arrival = arrival.max(signal_arrival(timing.timing));
            }
        }
        if instance.sequential {
            for input in instance.inputs.iter().filter(|input| !input.clock) {
                let Some(bit) = input.bit else {
                    continue;
                };
                let pin = cell
                    .pins
                    .iter()
                    .find(|pin| self.library.resolve_string(&pin.name) == input.name)
                    .ok_or_else(|| {
                        anyhow!("estimated register replacement changes '{}'", input.name)
                    })?;
                for signal in [
                    self.bit_timing[bit].primary_input,
                    self.bit_timing[bit].register,
                ]
                .into_iter()
                .flatten()
                {
                    if let Some(capture) = evaluate_sequential_cell_capture_timing_with_predecessor(
                        self.library,
                        cell.name.as_str(),
                        pin,
                        signal,
                        &instance.known_pin_values,
                        &mut self.diagnostics,
                    )? {
                        arrival = arrival.max(capture.arrival);
                    }
                }
            }
        }
        Ok(arrival)
    }

    /// Preserves clock periods and individual external output deadlines.
    fn satisfies_constraints(&self, score: BoundaryTimingScore) -> bool {
        if let Some(period) = self.constraints.clock_period
            && [score.input_to_register, score.register_to_register]
                .into_iter()
                .flatten()
                .any(|arrival| arrival > period + TIMING_VERIFICATION_EPSILON)
        {
            return false;
        }
        for output in &self.outputs {
            let Some(required) = self.constraints.primary_output_required.get(&output.name) else {
                continue;
            };
            let timing = self.bit_timing[output.bit];
            if [timing.primary_input, timing.register]
                .into_iter()
                .flatten()
                .map(signal_arrival)
                .any(|arrival| arrival > *required + TIMING_VERIFICATION_EPSILON)
            {
                return false;
            }
        }
        true
    }

    /// Traces all exact-Liberty arcs inside an adaptive near-critical window.
    fn critical_window_instances(
        &mut self,
        max_paths: usize,
        window_fraction: f64,
    ) -> Result<Vec<CriticalInstance>> {
        let score = self.score();
        let class_maxima = score.values();
        let mut endpoints = Vec::new();
        for (index, capture) in self.captures.iter().enumerate() {
            let predecessors = self.capture_predecessors[index];
            for (class, arrival, transition, register_launch) in [
                (0, capture.register, predecessors.register, true),
                (1, capture.primary_input, predecessors.primary_input, false),
            ] {
                let (Some(arrival), Some(transition)) = (arrival, transition) else {
                    continue;
                };
                let available_slack =
                    class_maxima[class].abs() * window_fraction - (class_maxima[class] - arrival);
                if available_slack + TIMING_VERIFICATION_EPSILON >= 0.0 {
                    endpoints.push(CriticalEndpoint {
                        class,
                        arrival,
                        transition,
                        register_launch,
                        capture: Some(index),
                        available_slack: available_slack.max(0.0),
                    });
                }
            }
        }
        for output in &self.outputs {
            let timing = self.bit_timing[output.bit];
            for (class, signal, register_launch) in
                [(2, timing.register, true), (3, timing.primary_input, false)]
            {
                let Some(signal) = signal else {
                    continue;
                };
                for (edge, arrival) in [
                    (TimingEdge::Rise, signal.rise.arrival),
                    (TimingEdge::Fall, signal.fall.arrival),
                ] {
                    let available_slack = class_maxima[class].abs() * window_fraction
                        - (class_maxima[class] - arrival);
                    if available_slack + TIMING_VERIFICATION_EPSILON >= 0.0 {
                        endpoints.push(CriticalEndpoint {
                            class,
                            arrival,
                            transition: BitEdge {
                                bit: output.bit,
                                edge,
                            },
                            register_launch,
                            capture: None,
                            available_slack: available_slack.max(0.0),
                        });
                    }
                }
            }
        }
        endpoints.sort_by(|lhs, rhs| {
            lhs.class
                .cmp(&rhs.class)
                .then_with(|| rhs.arrival.total_cmp(&lhs.arrival))
                .then_with(|| lhs.capture.cmp(&rhs.capture))
                .then_with(|| lhs.transition.cmp(&rhs.transition))
        });

        let mut ranks = BTreeMap::<usize, CriticalInstance>::new();
        let mut arc_cache = HashMap::<(BitIndex, bool), Vec<WindowArc>>::new();
        for (path_index, endpoint) in endpoints.into_iter().take(max_paths).enumerate() {
            if let Some(capture) = endpoint.capture {
                record_critical_instance(&mut ranks, capture, path_index, endpoint.available_slack);
            }
            let mut queue = VecDeque::from([(endpoint.transition, endpoint.available_slack)]);
            let mut visited = BTreeMap::<BitEdge, f64>::new();
            while let Some((transition, available_slack)) = queue.pop_front() {
                if visited.get(&transition).is_some_and(|previous| {
                    *previous + TIMING_VERIFICATION_EPSILON >= available_slack
                }) {
                    continue;
                }
                visited.insert(transition, available_slack);
                let Some(driver) = self.drivers[transition.bit] else {
                    continue;
                };
                record_critical_instance(&mut ranks, driver, path_index, available_slack);
                if self.instances[driver].sequential {
                    continue;
                }
                let Some(signal) =
                    self.bit_timing[transition.bit].for_launch(endpoint.register_launch)
                else {
                    continue;
                };
                let output_arrival = timing_edge_arrival(signal, transition.edge);
                let key = (transition.bit, endpoint.register_launch);
                let arcs = if let Some(cached) = arc_cache.get(&key) {
                    cached.clone()
                } else {
                    let computed =
                        self.input_window_arcs(driver, transition.bit, endpoint.register_launch)?;
                    arc_cache.insert(key, computed.clone());
                    computed
                };
                for arc in arcs {
                    let Some((input_arrival, input_transition)) = arc.for_edge(transition.edge)
                    else {
                        continue;
                    };
                    let path_slack = (output_arrival - input_arrival).max(0.0);
                    if path_slack <= available_slack + TIMING_VERIFICATION_EPSILON {
                        queue
                            .push_back((input_transition, (available_slack - path_slack).max(0.0)));
                    }
                }
                if let Some(predecessor) = self.bit_predecessors[transition.bit]
                    .for_launch(endpoint.register_launch)
                    .for_edge(transition.edge)
                {
                    queue.push_back((predecessor, available_slack));
                }
            }
        }

        let mut result = ranks.into_values().collect::<Vec<_>>();
        result.sort_by(|lhs, rhs| {
            rhs.path_count
                .cmp(&lhs.path_count)
                .then_with(|| lhs.slack.total_cmp(&rhs.slack))
                .then_with(|| lhs.first_path.cmp(&rhs.first_path))
                .then_with(|| lhs.instance_index.cmp(&rhs.instance_index))
        });
        Ok(result)
    }

    /// Evaluates each related pin at its actual slew, polarity, and net load.
    fn input_window_arcs(
        &mut self,
        instance_index: usize,
        output_bit: BitIndex,
        register_launch: bool,
    ) -> Result<Vec<WindowArc>> {
        let instance = &self.instances[instance_index];
        let cell = &self.library.cells[instance.cell_index];
        let Some(output) = instance
            .outputs
            .iter()
            .find(|output| output.bit == output_bit)
        else {
            return Ok(Vec::new());
        };
        let output_pin = &cell.pins[output.pin_index];
        let mut arcs = Vec::new();
        for input in instance.inputs.iter().filter(|input| !input.clock) {
            let Some(bit) = input.bit else {
                continue;
            };
            let Some(signal) = self.bit_timing[bit].for_launch(register_launch) else {
                continue;
            };
            if !output_pin.timing_arcs.iter().any(|arc| {
                self.library
                    .resolve_string(&arc.related_pin)
                    .split_whitespace()
                    .any(|related| related == input.name)
            }) {
                continue;
            }
            let candidate = evaluate_combinational_cell_output_timing_with_predecessors(
                self.library,
                cell.name.as_str(),
                output_pin,
                &[(input.name.as_str(), signal)],
                self.loads[output_bit],
                &instance.known_pin_values,
                &mut self.diagnostics,
            )?;
            arcs.push(WindowArc {
                rise: candidate.rise_predecessor.map(|predecessor| {
                    (
                        candidate.timing.rise.arrival,
                        BitEdge {
                            bit,
                            edge: predecessor.input_edge,
                        },
                    )
                }),
                fall: candidate.fall_predecessor.map(|predecessor| {
                    (
                        candidate.timing.fall.arrival,
                        BitEdge {
                            bit,
                            edge: predecessor.input_edge,
                        },
                    )
                }),
            });
        }
        Ok(arcs)
    }

    /// Selects high-ranked replacements whose immediate timing cones do not
    /// overlap.
    fn independent_move_batch(
        &self,
        candidates: Vec<SizingMove>,
        maximum: usize,
    ) -> Vec<SizingMove> {
        let mut blocked = vec![false; self.instances.len()];
        let mut selected = Vec::new();
        for candidate in candidates {
            if selected.len() == maximum {
                break;
            }
            let index = candidate.instance_index;
            if blocked[index] {
                continue;
            }
            blocked[index] = true;
            for successor in &self.successors[index] {
                blocked[*successor] = true;
            }
            for input in &self.instances[index].inputs {
                let Some(bit) = input.bit else {
                    continue;
                };
                let Some(driver) = self.drivers[bit] else {
                    continue;
                };
                blocked[driver] = true;
                for sibling in &self.successors[driver] {
                    blocked[*sibling] = true;
                }
            }
            for output in &self.instances[index].outputs {
                for capture in &self.capture_consumers[output.bit] {
                    blocked[*capture] = true;
                }
            }
            selected.push(candidate);
        }
        selected
    }

    /// Evaluates either supported exact-function optimization move.
    fn evaluate_optimization_move(
        &mut self,
        instance_index: usize,
        kind: SizingMoveKind,
        catalog: &CellCatalog,
        commit: bool,
    ) -> Result<SizingTrial> {
        match kind {
            SizingMoveKind::Resize { cell_index } => {
                self.evaluate_cell_substitution(instance_index, cell_index, commit)
            }
            SizingMoveKind::SwapPins {
                first_input,
                second_input,
            } => self.evaluate_pin_swap(instance_index, first_input, second_input, catalog, commit),
        }
    }

    /// Evaluates a reversible same-pin cell substitution.
    fn evaluate_cell_substitution(
        &mut self,
        instance_index: usize,
        replacement_index: usize,
        commit: bool,
    ) -> Result<SizingTrial> {
        let mut replacement = self
            .instances
            .get(instance_index)
            .cloned()
            .ok_or_else(|| anyhow!("registered sizing instance is out of bounds"))?;
        replacement.cell_index = replacement_index;
        self.evaluate_instance_change(instance_index, replacement, commit)
    }

    /// Evaluates a Boolean-safe exchange of characterized combinational pins.
    fn evaluate_pin_swap(
        &mut self,
        instance_index: usize,
        first_input: usize,
        second_input: usize,
        catalog: &CellCatalog,
        commit: bool,
    ) -> Result<SizingTrial> {
        let original = self
            .instances
            .get(instance_index)
            .ok_or_else(|| anyhow!("pin-swap instance is out of bounds"))?;
        if original.sequential {
            bail!("physical register pins cannot be swapped");
        }
        if first_input == second_input {
            bail!("a pin swap requires two distinct input pins");
        }
        let first = original
            .inputs
            .get(first_input)
            .ok_or_else(|| anyhow!("first pin-swap input is out of bounds"))?;
        let second = original
            .inputs
            .get(second_input)
            .ok_or_else(|| anyhow!("second pin-swap input is out of bounds"))?;
        if first.clock || second.clock {
            bail!("clock and sequential control pins cannot be swapped");
        }
        let cell = &self.library.cells[original.cell_index];
        let classified = catalog.by_name(cell.name.as_str()).ok_or_else(|| {
            anyhow!(
                "pin-swap cell '{}' has no Boolean classification",
                cell.name
            )
        })?;
        let is_symmetric = classified.symmetric_input_pairs.iter().any(|pair| {
            let left = &classified.family.input_names[pair.first_input];
            let right = &classified.family.input_names[pair.second_input];
            (first.name == *left && second.name == *right)
                || (first.name == *right && second.name == *left)
        });
        if !is_symmetric {
            bail!(
                "cell '{}.{}' and '{}.{}' are not Boolean-symmetric",
                cell.name,
                first.name,
                cell.name,
                second.name
            );
        }

        let mut replacement = original.clone();
        let first_name = replacement.inputs[first_input].name.clone();
        let second_name = replacement.inputs[second_input].name.clone();
        let first_bit = replacement.inputs[first_input].bit;
        replacement.inputs[first_input].bit = replacement.inputs[second_input].bit;
        replacement.inputs[second_input].bit = first_bit;
        let first_constant = replacement.known_pin_values.remove(&first_name);
        let second_constant = replacement.known_pin_values.remove(&second_name);
        if let Some(value) = second_constant {
            replacement.known_pin_values.insert(first_name, value);
        }
        if let Some(value) = first_constant {
            replacement.known_pin_values.insert(second_name, value);
        }
        self.evaluate_instance_change(instance_index, replacement, commit)
    }

    /// Propagates one reversible assignment through both affected timing cones.
    fn evaluate_instance_change(
        &mut self,
        instance_index: usize,
        mut replacement: SizingInstance,
        commit: bool,
    ) -> Result<SizingTrial> {
        let original = self
            .instances
            .get(instance_index)
            .cloned()
            .ok_or_else(|| anyhow!("registered sizing instance is out of bounds"))?;
        let old_cell = &self.library.cells[original.cell_index];
        let new_cell = self
            .library
            .cells
            .get(replacement.cell_index)
            .ok_or_else(|| anyhow!("registered sizing replacement is out of bounds"))?;
        if original.sequential != replacement.sequential
            || original.inputs.len() != replacement.inputs.len()
            || original.outputs.len() != replacement.outputs.len()
        {
            bail!("registered sizing changes the physical cell interface");
        }
        let original_output_timing = original
            .outputs
            .iter()
            .map(|output| (output.bit, self.bit_timing[output.bit]))
            .collect::<Vec<_>>();
        let original_capture = original.sequential.then_some(self.captures[instance_index]);

        let mut changed_loads =
            BTreeMap::<BitIndex, (CombinationalOutputLoad, CombinationalOutputLoad)>::new();
        let mut clock_delta = 0.0;
        for (old_input, new_input) in original.inputs.iter().zip(&replacement.inputs) {
            if old_input.name != new_input.name || old_input.clock != new_input.clock {
                bail!("registered sizing changes input pin '{}'", old_input.name);
            }
            let old_pin = old_cell
                .pins
                .iter()
                .find(|pin| self.library.resolve_string(&pin.name) == old_input.name)
                .ok_or_else(|| anyhow!("original cell is missing input '{}'", old_input.name))?;
            let new_pin = new_cell
                .pins
                .iter()
                .find(|pin| self.library.resolve_string(&pin.name) == new_input.name)
                .ok_or_else(|| anyhow!("replacement changes input pin '{}'", new_input.name))?;
            if old_pin.direction != new_pin.direction
                || old_pin.is_clocking_pin != new_pin.is_clocking_pin
                || old_pin.is_clocking_pin != old_input.clock
            {
                bail!(
                    "registered sizing changes the direction or clock role of '{}'",
                    old_input.name
                );
            }
            if let Some(bit) = old_input.bit {
                let capacitance = effective_input_capacitance_for_mapping(
                    old_pin,
                    &format!(
                        "original sized input '{}.{}'",
                        old_cell.name, old_input.name
                    ),
                )?;
                let entry = changed_loads
                    .entry(bit)
                    .or_insert((self.loads[bit], CombinationalOutputLoad::default()));
                entry.1.rise -= capacitance.rise;
                entry.1.fall -= capacitance.fall;
                if old_input.clock {
                    clock_delta -= max_load(capacitance);
                }
            }
            if let Some(bit) = new_input.bit {
                let capacitance = effective_input_capacitance_for_mapping(
                    new_pin,
                    &format!(
                        "replacement sized input '{}.{}'",
                        new_cell.name, new_input.name
                    ),
                )?;
                let entry = changed_loads
                    .entry(bit)
                    .or_insert((self.loads[bit], CombinationalOutputLoad::default()));
                entry.1.rise += capacitance.rise;
                entry.1.fall += capacitance.fall;
                if new_input.clock {
                    clock_delta += max_load(capacitance);
                }
            }
        }

        for output in &mut replacement.outputs {
            let (pin_index, pin) = new_cell
                .pins
                .iter()
                .enumerate()
                .find(|(_, pin)| {
                    pin.direction == PinDirection::Output as i32
                        && self.library.resolve_string(&pin.name) == output.name
                })
                .ok_or_else(|| anyhow!("replacement changes output pin '{}'", output.name))?;
            if pin.max_capacitance.is_some_and(|limit| {
                max_load(self.loads[output.bit]) > limit + TIMING_VERIFICATION_EPSILON
            }) {
                bail!(
                    "replacement '{}.{}' cannot drive its actual output load",
                    new_cell.name,
                    output.name
                );
            }
            output.pin_index = pin_index;
        }

        let mut original_driver_timing = BTreeMap::new();
        let original_clock_load = self.clock_load;
        let mut dirty = BTreeSet::new();
        dirty.insert((self.topological_positions[instance_index], instance_index));
        let mut dirty_captures = BTreeSet::new();
        if original.sequential {
            dirty_captures.insert(instance_index);
        }
        for (bit, (_, delta)) in &changed_loads {
            if (delta.rise != 0.0 || delta.fall != 0.0)
                && let Some(driver) = self.drivers[*bit]
            {
                dirty.insert((self.topological_positions[driver], driver));
                for output in &self.instances[driver].outputs {
                    original_driver_timing
                        .entry(output.bit)
                        .or_insert(self.bit_timing[output.bit]);
                }
            }
        }

        self.instances[instance_index] = replacement;
        self.clock_load += clock_delta;
        for (bit, (previous, delta)) in &changed_loads {
            self.loads[*bit].rise = previous.rise + delta.rise;
            self.loads[*bit].fall = previous.fall + delta.fall;
            if self.loads[*bit].rise < -TIMING_VERIFICATION_EPSILON
                || self.loads[*bit].fall < -TIMING_VERIFICATION_EPSILON
            {
                self.restore_trial(
                    instance_index,
                    original,
                    &changed_loads,
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    original_clock_load,
                );
                bail!("registered sizing produced a negative net load");
            }
            self.loads[*bit].rise = self.loads[*bit].rise.max(0.0);
            self.loads[*bit].fall = self.loads[*bit].fall.max(0.0);
        }

        let mut saved_timings =
            BTreeMap::<BitIndex, (BoundarySignalTiming, BoundaryPredecessors)>::new();
        let mut saved_captures =
            BTreeMap::<usize, (RegisterCaptureScore, CapturePredecessors)>::new();
        let result = (|| {
            let mut recomputed = 0usize;
            while let Some((_, index)) = dirty.pop_first() {
                let old_outputs = self.instances[index]
                    .outputs
                    .iter()
                    .map(|output| {
                        (
                            output.bit,
                            self.bit_timing[output.bit],
                            self.bit_predecessors[output.bit],
                        )
                    })
                    .collect::<Vec<_>>();
                for (bit, timing, predecessors) in &old_outputs {
                    saved_timings
                        .entry(*bit)
                        .or_insert((*timing, *predecessors));
                }
                self.recompute_instance(index)?;
                recomputed += 1;
                for (bit, previous, _) in old_outputs {
                    if self.bit_timing[bit] != previous {
                        for successor in &self.successors[index] {
                            dirty.insert((self.topological_positions[*successor], *successor));
                        }
                        for capture in &self.capture_consumers[bit] {
                            dirty_captures.insert(*capture);
                        }
                    }
                }
            }
            for capture in dirty_captures {
                saved_captures
                    .entry(capture)
                    .or_insert((self.captures[capture], self.capture_predecessors[capture]));
                self.recompute_capture(capture)?;
                recomputed += 1;
            }
            let score = self.score();
            let local_improvement = original_output_timing
                .iter()
                .map(|(bit, previous)| (*bit, *previous))
                .chain(
                    original_driver_timing
                        .iter()
                        .map(|(bit, previous)| (*bit, *previous)),
                )
                .map(|(bit, previous)| boundary_timing_improvement(previous, self.bit_timing[bit]))
                .chain(original_capture.into_iter().map(|previous| {
                    capture_timing_improvement(previous, self.captures[instance_index])
                }))
                .fold(0.0_f64, f64::max);
            Ok(SizingTrial {
                score,
                secondary_delay: self.secondary_delay(),
                recomputed_instances: recomputed,
                clock_load: self.clock_load,
                constraints_satisfied: self.satisfies_constraints(score),
                local_improvement,
            })
        })();

        if result.is_err() || !commit {
            self.restore_trial(
                instance_index,
                original,
                &changed_loads,
                &saved_timings,
                &saved_captures,
                original_clock_load,
            );
        }
        result
    }

    /// Restores all timing, loading, capture, and clock state after a trial.
    fn restore_trial(
        &mut self,
        instance_index: usize,
        original: SizingInstance,
        changed_loads: &BTreeMap<BitIndex, (CombinationalOutputLoad, CombinationalOutputLoad)>,
        saved_timings: &BTreeMap<BitIndex, (BoundarySignalTiming, BoundaryPredecessors)>,
        saved_captures: &BTreeMap<usize, (RegisterCaptureScore, CapturePredecessors)>,
        original_clock_load: f64,
    ) {
        self.instances[instance_index] = original;
        for (bit, (previous, _)) in changed_loads {
            self.loads[*bit] = *previous;
        }
        for (bit, (previous, predecessors)) in saved_timings {
            self.bit_timing[*bit] = *previous;
            self.bit_predecessors[*bit] = *predecessors;
        }
        for (index, (previous, predecessors)) in saved_captures {
            self.captures[*index] = *previous;
            self.capture_predecessors[*index] = *predecessors;
        }
        self.clock_load = original_clock_load;
    }

    /// Recomputes an exact Liberty combinational or FF launch output.
    fn recompute_instance(&mut self, instance_index: usize) -> Result<()> {
        let instance = &self.instances[instance_index];
        let cell = &self.library.cells[instance.cell_index];
        for output in &instance.outputs {
            let pin = &cell.pins[output.pin_index];
            if instance.sequential {
                let timing = evaluate_sequential_cell_output_timing(
                    self.library,
                    cell.name.as_str(),
                    pin,
                    self.loads[output.bit],
                    &instance.known_pin_values,
                    &mut self.diagnostics,
                )?;
                self.bit_timing[output.bit] = BoundarySignalTiming {
                    primary_input: None,
                    register: Some(timing),
                };
                self.bit_predecessors[output.bit] = BoundaryPredecessors::default();
                continue;
            }
            let mut primary = Vec::new();
            let mut primary_bits = Vec::new();
            let mut register = Vec::new();
            let mut register_bits = Vec::new();
            for input in instance.inputs.iter().filter(|input| !input.clock) {
                if let Some(bit) = input.bit {
                    let timing = self.bit_timing[bit];
                    if let Some(timing) = timing.primary_input {
                        primary.push((input.name.as_str(), timing));
                        primary_bits.push(Some(bit));
                    }
                    if let Some(timing) = timing.register {
                        register.push((input.name.as_str(), timing));
                        register_bits.push(Some(bit));
                    }
                } else if !self.has_registers && instance.known_pin_values.contains_key(&input.name)
                {
                    primary.push((input.name.as_str(), literal_signal_timing()));
                    primary_bits.push(None);
                }
            }
            let primary = if primary.is_empty() {
                None
            } else {
                Some(evaluate_combinational_cell_output_timing_with_predecessors(
                    self.library,
                    cell.name.as_str(),
                    pin,
                    primary.as_slice(),
                    self.loads[output.bit],
                    &instance.known_pin_values,
                    &mut self.diagnostics,
                )?)
            };
            let register = if register.is_empty() {
                None
            } else {
                Some(evaluate_combinational_cell_output_timing_with_predecessors(
                    self.library,
                    cell.name.as_str(),
                    pin,
                    register.as_slice(),
                    self.loads[output.bit],
                    &instance.known_pin_values,
                    &mut self.diagnostics,
                )?)
            };
            self.bit_timing[output.bit] = BoundarySignalTiming {
                primary_input: primary.map(|result| result.timing),
                register: register.map(|result| result.timing),
            };
            self.bit_predecessors[output.bit] = BoundaryPredecessors {
                primary_input: traced_edge_predecessors(primary, primary_bits.as_slice()),
                register: traced_edge_predecessors(register, register_bits.as_slice()),
            };
        }
        Ok(())
    }

    /// Updates both setup-constrained path classes of a capture register.
    fn recompute_capture(&mut self, instance_index: usize) -> Result<()> {
        let instance = &self.instances[instance_index];
        if !instance.sequential {
            return Ok(());
        }
        let cell = &self.library.cells[instance.cell_index];
        let mut capture = RegisterCaptureScore::default();
        let mut predecessors = CapturePredecessors::default();
        for input in instance.inputs.iter().filter(|input| !input.clock) {
            let Some(bit) = input.bit else {
                continue;
            };
            let pin = cell
                .pins
                .iter()
                .find(|pin| self.library.resolve_string(&pin.name) == input.name)
                .ok_or_else(|| anyhow!("register capture lost pin '{}'", input.name))?;
            let timing = self.bit_timing[bit];
            for (signal, register_launch) in
                [(timing.primary_input, false), (timing.register, true)]
            {
                let Some(signal) = signal else {
                    continue;
                };
                let Some(result) = evaluate_sequential_cell_capture_timing_with_predecessor(
                    self.library,
                    cell.name.as_str(),
                    pin,
                    signal,
                    &instance.known_pin_values,
                    &mut self.diagnostics,
                )?
                else {
                    continue;
                };
                let (destination, predecessor) = if register_launch {
                    (&mut capture.register, &mut predecessors.register)
                } else {
                    (&mut capture.primary_input, &mut predecessors.primary_input)
                };
                if destination.is_none_or(|current| result.arrival > current) {
                    *destination = Some(result.arrival);
                    *predecessor = Some(BitEdge {
                        bit,
                        edge: result.input_edge,
                    });
                }
            }
        }
        self.captures[instance_index] = capture;
        self.capture_predecessors[instance_index] = predecessors;
        Ok(())
    }
}

/// Updates deterministic endpoint/path/slack priorities for a critical cell.
fn record_critical_instance(
    ranks: &mut BTreeMap<usize, CriticalInstance>,
    instance_index: usize,
    path_index: usize,
    slack: f64,
) {
    ranks
        .entry(instance_index)
        .and_modify(|current| {
            current.path_count += 1;
            current.slack = current.slack.min(slack);
        })
        .or_insert(CriticalInstance {
            instance_index,
            path_count: 1,
            first_path: path_index,
            slack,
        });
}

/// Converts exact Liberty predecessor indices to stable normalized net bits.
fn traced_edge_predecessors(
    timing: Option<TracedCombinationalTiming>,
    input_bits: &[Option<BitIndex>],
) -> EdgePredecessors {
    let Some(timing) = timing else {
        return EdgePredecessors::default();
    };
    let map = |predecessor: Option<TimingPredecessor>| {
        predecessor.and_then(|predecessor| {
            input_bits
                .get(predecessor.input_index)
                .copied()
                .flatten()
                .map(|bit| BitEdge {
                    bit,
                    edge: predecessor.input_edge,
                })
        })
    };
    EdgePredecessors {
        rise: map(timing.rise_predecessor),
        fall: map(timing.fall_predecessor),
    }
}

/// Matches the zero-arrival, zero-slew literal used by full combinational STA.
fn literal_signal_timing() -> SignalTiming {
    SignalTiming {
        rise: EdgeTiming {
            arrival: 0.0,
            transition: 0.0,
        },
        fall: EdgeTiming {
            arrival: 0.0,
            transition: 0.0,
        },
    }
}

/// Returns the exact arrival of the requested physical Liberty transition.
fn timing_edge_arrival(timing: SignalTiming, edge: TimingEdge) -> f64 {
    match edge {
        TimingEdge::Rise => timing.rise.arrival,
        TimingEdge::Fall => timing.fall.arrival,
    }
}

/// Measures physical local improvement without hiding a tied global endpoint.
fn boundary_timing_improvement(
    previous: BoundarySignalTiming,
    current: BoundarySignalTiming,
) -> f64 {
    [
        (previous.primary_input, current.primary_input),
        (previous.register, current.register),
    ]
    .into_iter()
    .filter_map(|(previous, current)| Some((previous?, current?)))
    .flat_map(|(previous, current)| {
        [
            previous.rise.arrival - current.rise.arrival,
            previous.fall.arrival - current.fall.arrival,
        ]
    })
    .fold(0.0_f64, f64::max)
}

/// Measures true setup-arc improvement at a resized physical capture FF.
fn capture_timing_improvement(
    previous: RegisterCaptureScore,
    current: RegisterCaptureScore,
) -> f64 {
    [
        (previous.primary_input, current.primary_input),
        (previous.register, current.register),
    ]
    .into_iter()
    .filter_map(|(previous, current)| Some(previous? - current?))
    .fold(0.0_f64, f64::max)
}

/// Returns the conservative arrival across both Liberty timing edges.
fn signal_arrival(timing: SignalTiming) -> f64 {
    timing.rise.arrival.max(timing.fall.arrival)
}

/// Returns the conservative actual capacitance across rise and fall.
fn max_load(load: CombinationalOutputLoad) -> f64 {
    load.rise.max(load.fall)
}

#[cfg(test)]
mod tests {
    use super::{
        BoundaryTimingScore, IncrementalRegisteredSta, RegisterCellCatalog, SizingTrial,
        prioritized_timing_instances, resize_timing_aware_netlist, timing_trial_is_acceptable,
    };
    use crate::liberty_model::{Library, LibraryBuilder};
    use crate::liberty_proto::TimingTableKind;
    use crate::netlist::cell_catalog::CellCatalog;
    use crate::netlist::cell_catalog::test_utils::{parse_module, sizing_library, timed_cell};
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::report::{
        build_netlist_report, build_netlist_report_with_primary_input_arrivals,
    };
    use crate::netlist::resize::ResizeOptions;
    use crate::netlist::sta::{StaOptions, analyze_register_boundary_max_arrival};
    use crate::netlist::timing_buffer::BufferTimingConstraints;
    use crate::netlist::timing_buffer::tests::{
        high_fanout_register_source, registered_timing_library,
    };
    use std::collections::BTreeMap;

    /// Adds an exact-state, same-pin FF with stronger clock-to-Q drive.
    fn registered_sizing_library() -> Library {
        let mut builder = LibraryBuilder::from_library(registered_timing_library());
        let mut fast = builder
            .cells
            .iter()
            .find(|cell| cell.name == "DFF")
            .expect("find the synthetic reference flip-flop")
            .clone();
        fast.name = "DFF_FAST".to_string();
        fast.area = 5.0;

        let mut clock_tables = Vec::new();
        for (kind, values) in [
            (TimingTableKind::CellRise, vec![0.25, 6.25]),
            (TimingTableKind::CellFall, vec![0.25, 6.25]),
            (TimingTableKind::RiseTransition, vec![0.05, 0.4]),
            (TimingTableKind::FallTransition, vec![0.05, 0.4]),
        ] {
            clock_tables.push(
                builder
                    .add_timing_table_f64(kind, 1, vec![], vec![], vec![], values, vec![2], "")
                    .expect("construct stronger clock-to-Q timing"),
            );
        }
        let clock_arc = builder
            .add_timing_arc("CLK", "non_unate", "rising_edge", "", clock_tables)
            .expect("construct stronger flip-flop launch arc");
        let mut setup_tables = Vec::new();
        for kind in [
            TimingTableKind::RiseConstraint,
            TimingTableKind::FallConstraint,
        ] {
            setup_tables.push(
                builder
                    .add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![0.1], vec![], "")
                    .expect("construct stronger flip-flop setup timing"),
            );
        }
        let setup_arc = builder
            .add_timing_arc("CLK", "", "setup_rising", "", setup_tables)
            .expect("construct stronger flip-flop setup arc");
        for pin in &mut fast.pins {
            let name = builder.resolve_string(&pin.name);
            if name == "Q" {
                pin.max_capacitance = Some(1.6);
                pin.timing_arcs = vec![clock_arc.clone()];
            } else if name == "D" {
                pin.capacitance = Some(0.13);
                pin.timing_arcs = vec![setup_arc.clone()];
            } else if name == "CLK" {
                pin.capacitance = Some(0.0101);
            }
        }
        builder.cells.push(fast);
        builder.finish()
    }

    /// Gives the A input a slower Liberty arc than a later-arriving B input.
    fn asymmetric_sizing_library() -> Library {
        let mut builder = LibraryBuilder::from_library(sizing_library());
        let tables = [
            (TimingTableKind::CellRise, 12.0),
            (TimingTableKind::CellFall, 12.0),
            (TimingTableKind::RiseTransition, 0.1),
            (TimingTableKind::FallTransition, 0.1),
        ]
        .into_iter()
        .map(|(kind, value)| {
            builder
                .add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![value], vec![], "")
                .expect("construct asymmetric synthetic pin timing")
        })
        .collect::<Vec<_>>();
        let slow_arc = builder
            .add_timing_arc("A", "positive_unate", "combinational", "", tables)
            .expect("construct the slower A-to-Y timing arc");
        let cell_index = builder
            .cells
            .iter()
            .position(|cell| cell.name == "AND2")
            .expect("find the synthetic AND cell");
        let pin_index = builder.cells[cell_index]
            .pins
            .iter()
            .position(|pin| builder.resolve_string(&pin.name) == "Y")
            .expect("find the synthetic AND output pin");
        let mut arcs = builder.cells[cell_index].pins[pin_index]
            .timing_arcs
            .clone();
        let arc_index = arcs
            .iter()
            .position(|arc| builder.resolve_string(&arc.related_pin) == "A")
            .expect("find the original A-to-Y arc");
        arcs[arc_index] = slow_arc;
        builder.cells[cell_index].pins[pin_index].timing_arcs = arcs;
        builder.finish()
    }

    /// Removes resizing alternatives so a characterized pin swap is isolated.
    fn pin_swap_only_library() -> Library {
        let mut builder = LibraryBuilder::from_library(asymmetric_sizing_library());
        builder.cells.retain(|cell| cell.name != "AND2_FAST");
        builder.finish()
    }

    /// Makes an off-path AND input load dominate a physical launch flip-flop.
    fn asymmetric_register_load_library() -> Library {
        let mut builder = LibraryBuilder::from_library(registered_timing_library());
        builder.cells.retain(|cell| cell.name != "AND2_FAST");
        let index = builder
            .cells
            .iter()
            .position(|cell| cell.name == "AND2")
            .expect("find the off-path symmetric AND");
        let mut and = builder.cells[index].clone();
        for pin in &mut and.pins {
            match builder.resolve_string(&pin.name) {
                "A" => {
                    pin.capacitance = Some(0.35);
                    pin.rise_capacitance = Some(0.30);
                    pin.fall_capacitance = Some(0.45);
                }
                "B" => {
                    pin.capacitance = Some(0.03);
                    pin.rise_capacitance = Some(0.02);
                    pin.fall_capacitance = Some(0.04);
                }
                _ => {
                    // The output timing and maximum drive stay unchanged.
                }
            }
        }
        builder.cells[index] = and;
        builder.finish()
    }

    /// Returns a pipeline with an otherwise dead, heavily loading sibling.
    fn asymmetric_register_load_source() -> &'static str {
        r#"
module top(clk, a, b, y);
  input clk, a, b;
  output y;
  wire root, unused;
  DFF launch (.CLK(clk), .D(a), .Q(root));
  DFF capture (.CLK(clk), .D(root), .Q(y));
  AND2 offpath (.A(root), .B(b), .Y(unused));
endmodule
"#
    }

    #[test]
    fn critical_window_follows_actual_pin_delay_instead_of_latest_input() {
        let source = r#"
module top(a, b, y);
  input a, b;
  output y;
  wire slow_pin, late_input;
  BUF slow_pin_driver (.A(a), .Y(slow_pin));
  BUF late_input_driver (.A(b), .Y(late_input));
  AND2 logic (.A(slow_pin), .B(late_input), .Y(y));
endmodule
"#;
        let library = asymmetric_sizing_library();
        let (module, nets, interner) = parse_module(source);
        let constraints = BufferTimingConstraints {
            primary_input_arrivals: BTreeMap::from([("b".to_string(), 4.0)]),
            ..BufferTimingConstraints::default()
        };
        let mut timing = IncrementalRegisteredSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &constraints,
        )
        .expect("construct exact-pin-delay incremental timing");
        let critical = timing
            .critical_window_instances(16, super::INITIAL_CRITICAL_WINDOW)
            .expect("trace the actual characterized critical pin")
            .into_iter()
            .map(|candidate| candidate.instance_index)
            .collect::<Vec<_>>();

        assert!(critical.contains(&0), "missing the actual slow-pin driver");
        assert!(critical.contains(&2), "missing the critical AND gate");
        assert!(
            !critical.contains(&1),
            "traced a later arrival with a noncritical Liberty pin delay"
        );
    }

    #[test]
    fn swaps_the_late_signal_onto_the_faster_characterized_input() {
        let source = r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 logic (.A(a), .B(b), .Y(y));
endmodule
"#;
        let library = pin_swap_only_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let original_connections = module.instances[0].connections.clone();
        let constraints = BufferTimingConstraints {
            primary_input_arrivals: BTreeMap::from([("a".to_string(), 4.0)]),
            ..BufferTimingConstraints::default()
        };

        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_outer_iterations: 1,
                max_iterations: 1,
                max_area_iterations: 0,
                ..ResizeOptions::default()
            },
            &constraints,
        )
        .expect("swap the late signal onto the faster real Liberty pin");

        assert_eq!(stats.pin_swaps, 1);
        assert_eq!(stats.pin_swap_steps.len(), 1);
        assert!(stats.pin_swap_evaluations >= 2);
        assert!((stats.initial_area - stats.final_area).abs() < 1e-9);
        assert!(stats.final_delay < stats.initial_delay);
        assert_eq!(stats.pin_swap_steps[0].first_pin, "A");
        assert_eq!(stats.pin_swap_steps[0].second_pin, "B");
        let original_a = original_connections
            .iter()
            .find(|(port, _)| interner.resolve(*port) == Some("A"))
            .expect("find the original A binding");
        let original_b = original_connections
            .iter()
            .find(|(port, _)| interner.resolve(*port) == Some("B"))
            .expect("find the original B binding");
        let final_a = module.instances[0]
            .connections
            .iter()
            .find(|(port, _)| interner.resolve(*port) == Some("A"))
            .expect("preserve the A connection");
        let final_b = module.instances[0]
            .connections
            .iter()
            .find(|(port, _)| interner.resolve(*port) == Some("B"))
            .expect("preserve the B connection");
        assert_eq!(final_a.1, original_b.1);
        assert_eq!(final_b.1, original_a.1);
    }

    #[test]
    fn retains_complete_pin_assignments_across_tied_critical_outputs() {
        let source = r#"
module top(a, b, y0, y1);
  input a, b;
  output y0, y1;
  AND2 first (.A(a), .B(b), .Y(y0));
  AND2 second (.A(a), .B(b), .Y(y1));
endmodule
"#;
        let library = pin_swap_only_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let constraints = BufferTimingConstraints {
            primary_input_arrivals: BTreeMap::from([("a".to_string(), 4.0)]),
            ..BufferTimingConstraints::default()
        };

        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_outer_iterations: 1,
                max_iterations: 1,
                max_area_iterations: 0,
                ..ResizeOptions::default()
            },
            &constraints,
        )
        .expect("retain the first tied pin move and finish the second output");
        let report = build_netlist_report_with_primary_input_arrivals(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &constraints.primary_input_arrivals,
        )
        .expect("independently verify the complete retained pin assignment");

        assert_eq!(stats.pin_swaps, 2);
        assert!(stats.final_delay < stats.initial_delay);
        assert_eq!(stats.final_delay, report.max_delay.unwrap());
        assert!((stats.final_area - stats.initial_area).abs() < 1e-9);
    }

    #[test]
    fn pin_swapping_is_deterministic() {
        let source = r#"
module top(a, b, y);
  input a, b;
  output y;
  AND2 logic (.A(a), .B(b), .Y(y));
endmodule
"#;
        let library = pin_swap_only_library();
        let constraints = BufferTimingConstraints {
            primary_input_arrivals: BTreeMap::from([("a".to_string(), 4.0)]),
            ..BufferTimingConstraints::default()
        };
        let options = ResizeOptions::default();
        let (mut first, first_nets, mut first_interner) = parse_module(source);
        let (mut second, second_nets, mut second_interner) = parse_module(source);

        let first_stats = resize_timing_aware_netlist(
            &mut first,
            &first_nets,
            &mut first_interner,
            &library,
            &options,
            &constraints,
        )
        .expect("optimize the first deterministic pin assignment");
        let second_stats = resize_timing_aware_netlist(
            &mut second,
            &second_nets,
            &mut second_interner,
            &library,
            &options,
            &constraints,
        )
        .expect("repeat the deterministic pin assignment");

        assert!(first_stats.pin_swaps > 0);
        assert_eq!(first_stats, second_stats);
        assert_eq!(
            emit_module_as_netlist_text(&first, &first_nets, &first_interner).unwrap(),
            emit_module_as_netlist_text(&second, &second_nets, &second_interner).unwrap()
        );
    }

    #[test]
    fn rejected_pin_swap_restores_loads_timing_and_predecessors() {
        let library = asymmetric_register_load_library();
        let (module, nets, interner) = parse_module(asymmetric_register_load_source());
        let catalog = CellCatalog::new(&library).expect("classify asymmetric pin loads");
        let mut timing = IncrementalRegisteredSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("construct load-sensitive registered incremental timing");
        let before_score = timing.score();
        let before_loads = timing.loads.clone();
        let before_timings = timing.bit_timing.clone();
        let before_bit_predecessors = timing.bit_predecessors.clone();
        let before_captures = timing.captures.clone();
        let before_capture_predecessors = timing.capture_predecessors.clone();
        let before_clock_load = timing.clock_load;
        let before_known = timing.instances[2].known_pin_values.clone();

        let trial = timing
            .evaluate_pin_swap(2, 0, 1, &catalog, false)
            .expect("evaluate both real input-load changes reversibly");

        assert!(trial.score.register_to_register < before_score.register_to_register);
        assert_eq!(timing.score(), before_score);
        assert_eq!(timing.loads, before_loads);
        assert_eq!(timing.bit_timing, before_timings);
        assert_eq!(timing.bit_predecessors, before_bit_predecessors);
        assert_eq!(timing.captures, before_captures);
        assert_eq!(timing.capture_predecessors, before_capture_predecessors);
        assert_eq!(timing.clock_load, before_clock_load);
        assert_eq!(timing.instances[2].known_pin_values, before_known);
    }

    #[test]
    fn prioritizes_the_off_path_sink_loading_a_critical_register() {
        let library = asymmetric_register_load_library();
        let (module, nets, interner) = parse_module(asymmetric_register_load_source());
        let catalog = CellCatalog::new(&library).expect("classify load-sensitive gates");
        let registers = RegisterCellCatalog::new(&library).expect("classify physical registers");
        let mut timing = IncrementalRegisteredSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("construct registered load-sensitive critical paths");
        let options = ResizeOptions::default();
        let critical = timing
            .critical_window_instances(options.max_candidate_paths, super::INITIAL_CRITICAL_WINDOW)
            .expect("trace the actual register-to-register path");

        assert!(
            critical
                .iter()
                .all(|candidate| candidate.instance_index != 2),
            "the unobserved sibling must not already be on a timed endpoint path"
        );
        let prioritized = prioritized_timing_instances(
            &timing, &critical, &library, &catalog, &registers, &options,
        );
        assert!(
            prioritized.contains(&2),
            "include the off-path pin capacitance loading the critical launch"
        );
    }

    #[test]
    fn resizes_an_off_path_sibling_to_unload_a_critical_register() {
        let source = r#"
module top(clk, a, b, y);
  input clk, a, b;
  output y;
  wire root, unused;
  DFF launch (.CLK(clk), .D(a), .Q(root));
  DFF capture (.CLK(clk), .D(root), .Q(y));
  AND2_FAST offpath (.A(root), .B(b), .Y(unused));
endmodule
"#;
        let library = registered_timing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let before =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("time the register loaded by an off-path gate");

        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_outer_iterations: 1,
                max_iterations: 1,
                max_area_iterations: 0,
                ..ResizeOptions::default()
            },
            &BufferTimingConstraints::default(),
        )
        .expect("resize the otherwise untimed loading sibling");
        let after =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("independently verify the unloaded register path");

        assert_eq!(
            interner.resolve(module.instances[2].type_name),
            Some("AND2")
        );
        assert!(stats.downsizes > 0);
        assert!(stats.final_area < stats.initial_area);
        assert!(
            after.max_register_to_register_delay.unwrap()
                < before.max_register_to_register_delay.unwrap()
        );
    }

    #[test]
    fn accepts_only_bounded_locally_improving_exploratory_sizes() {
        let best = BoundaryTimingScore {
            input_to_output: Some(100.0),
            ..BoundaryTimingScore::default()
        };
        let options = ResizeOptions::default();
        let mut trial = SizingTrial {
            score: BoundaryTimingScore {
                input_to_output: Some(100.5),
                ..BoundaryTimingScore::default()
            },
            secondary_delay: 100.5,
            recomputed_instances: 1,
            clock_load: 0.0,
            constraints_satisfied: true,
            local_improvement: 2.0,
        };

        assert!(!timing_trial_is_acceptable(
            &trial, best, best, &options, false
        ));
        assert!(timing_trial_is_acceptable(
            &trial, best, best, &options, true
        ));

        trial.score.input_to_output = Some(101.5);
        assert!(!timing_trial_is_acceptable(
            &trial, best, best, &options, true
        ));

        trial.score.input_to_output = Some(100.5);
        trial.local_improvement = 0.0;
        assert!(!timing_trial_is_acceptable(
            &trial, best, best, &options, true
        ));
    }

    #[test]
    fn area_recovery_preserves_near_critical_electrical_drive() {
        let mut builder = LibraryBuilder::from_library(sizing_library());
        let weak = builder
            .cells
            .iter()
            .position(|cell| cell.name == "BUF")
            .expect("find the weaker synthetic buffer");
        let output = builder.cells[weak]
            .pins
            .iter()
            .position(|pin| builder.resolve_string(&pin.name) == "Y")
            .expect("find the weaker buffer output");
        builder.cells[weak].pins[output].max_capacitance = Some(2.0);
        let library = builder.finish();
        let source = r#"
module top(a, b, y0, y1);
  input a, b;
  output y0, y1;
  wire root;
  AND2 critical (.A(a), .B(b), .Y(root));
  BUF_FAST critical_output (.A(root), .Y(y0));
  BUF_FAST near_output (.A(a), .Y(y1));
endmodule
"#;
        let (mut module, nets, mut interner) = parse_module(source);
        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                sta_options: StaOptions {
                    module_output_load: 1.0,
                    ..StaOptions::default()
                },
                max_iterations: 0,
                ..ResizeOptions::default()
            },
            &BufferTimingConstraints::default(),
        )
        .expect("avoid creating an electrically underdriven near-critical output");

        assert_eq!(
            interner.resolve(module.instances[2].type_name),
            Some("BUF_FAST")
        );
        assert_eq!(stats.final_area, stats.initial_area);
        assert_eq!(stats.final_delay, stats.initial_delay);
    }

    #[test]
    fn swaps_an_off_path_pin_to_improve_real_register_to_register_delay() {
        let library = asymmetric_register_load_library();
        let (mut module, nets, mut interner) = parse_module(asymmetric_register_load_source());
        let before =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("independently time the original loaded physical register");

        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_outer_iterations: 2,
                max_area_iterations: 0,
                ..ResizeOptions::default()
            },
            &BufferTimingConstraints::default(),
        )
        .expect("unload the critical register with a zero-area sibling pin swap");
        let after =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("independently verify swapped register-boundary timing");

        assert!(stats.pin_swaps > 0);
        assert!((stats.initial_area - stats.final_area).abs() < 1e-9);
        assert!(
            after.max_register_to_register_delay.unwrap()
                < before.max_register_to_register_delay.unwrap()
        );
        assert!(
            (stats.final_delay - BoundaryTimingScore::from_report(&after).worst_delay()).abs()
                < 1e-9
        );
        assert_eq!(stats.final_clock_load, stats.initial_clock_load);
    }

    #[test]
    fn rejects_functionally_distinct_complex_gate_pins() {
        let mut builder = LibraryBuilder::new();
        let aoi = timed_cell(
            &mut builder,
            "AOI21",
            &["A1", "A2", "B"],
            "!(A1 * A2 + B)",
            1.0,
            1.0,
            0.1,
            1.0,
        );
        builder.cells.push(aoi);
        let library = builder.finish();
        let source = r#"
module top(a, b, c, y);
  input a, b, c;
  output y;
  AOI21 logic (.A1(a), .A2(b), .B(c), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_module(source);
        let catalog = CellCatalog::new(&library).expect("classify asymmetric AOI pin roles");
        let mut timing = IncrementalRegisteredSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("construct complex-gate incremental timing");

        timing
            .evaluate_pin_swap(0, 0, 1, &catalog, false)
            .expect("the two AND inputs are Boolean-symmetric");
        assert!(
            timing.evaluate_pin_swap(0, 0, 2, &catalog, false).is_err(),
            "the distinct OR input cannot be exchanged with an AND input"
        );
    }

    #[test]
    fn swaps_known_constants_without_losing_pin_values() {
        let library = pin_swap_only_library();
        let source = r#"
module top(a, y);
  input a;
  output y;
  AND2 logic (.A(a), .B(1'b1), .Y(y));
endmodule
"#;
        let (mut module, nets, mut interner) = parse_module(source);
        let constraints = BufferTimingConstraints {
            primary_input_arrivals: BTreeMap::from([("a".to_string(), 20.0)]),
            ..BufferTimingConstraints::default()
        };
        let result = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_outer_iterations: 1,
                max_iterations: 1,
                max_area_iterations: 0,
                ..ResizeOptions::default()
            },
            &constraints,
        );
        let stats = result.unwrap_or_else(|error| {
            let emitted = emit_module_as_netlist_text(&module, &nets, &interner)
                .expect("emit the partially optimized constant-pin module");
            panic!(
                "move a known constant without corrupting exact Liberty conditions: \
                 {error:#}; resulting netlist:\n{emitted}"
            );
        });
        let after = build_netlist_report_with_primary_input_arrivals(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &constraints.primary_input_arrivals,
        )
        .expect("independently time the swapped constant-pin netlist");

        assert_eq!(stats.pin_swaps, 1);
        assert!(stats.final_delay < stats.initial_delay);
        assert_eq!(
            stats.final_delay,
            after.max_delay.expect("retain the combinational output")
        );
    }

    #[test]
    fn avoids_no_op_swaps_of_repeated_sources() {
        let library = pin_swap_only_library();
        let source = r#"
module top(a, y);
  input a;
  output y;
  AND2 logic (.A(a), .B(a), .Y(y));
endmodule
"#;
        let (mut module, nets, mut interner) = parse_module(source);
        let original = module.clone();
        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("ignore a pin exchange with an identical normalized source");

        assert_eq!(stats.pin_swaps, 0);
        assert_eq!(stats.pin_swap_evaluations, 0);
        assert_eq!(stats.outer_iterations, 1);
        assert_eq!(module, original);
    }

    #[test]
    fn never_swaps_physical_flip_flop_pins() {
        let library = registered_sizing_library();
        let (module, nets, interner) = parse_module(high_fanout_register_source());
        let catalog = CellCatalog::new(&library).expect("classify combinational gate pins");
        let mut timing = IncrementalRegisteredSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("construct exact physical register timing");

        assert!(
            timing.evaluate_pin_swap(0, 0, 1, &catalog, false).is_err(),
            "clock, data, reset, and enable pins are never interchangeable"
        );
    }

    #[test]
    fn one_adaptive_round_sizes_multiple_independent_critical_paths() {
        let source = r#"
module top(a, b, near, out);
  input a, b, near;
  output [11:0] out;
  AND2 critical0 (.A(a), .B(b), .Y(out[0]));
  AND2 critical1 (.A(a), .B(b), .Y(out[1]));
  BUF_FAST near0 (.A(near), .Y(out[2]));
  BUF_FAST near1 (.A(near), .Y(out[3]));
  BUF_FAST near2 (.A(near), .Y(out[4]));
  BUF_FAST near3 (.A(near), .Y(out[5]));
  BUF_FAST near4 (.A(near), .Y(out[6]));
  BUF_FAST near5 (.A(near), .Y(out[7]));
  BUF_FAST near6 (.A(near), .Y(out[8]));
  BUF_FAST near7 (.A(near), .Y(out[9]));
  BUF_FAST near8 (.A(near), .Y(out[10]));
  BUF_FAST near9 (.A(near), .Y(out[11]));
endmodule
"#;
        let library = sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let constraints = BufferTimingConstraints {
            primary_input_arrivals: BTreeMap::from([("near".to_string(), 3.95)]),
            ..BufferTimingConstraints::default()
        };
        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_iterations: 1,
                max_area_iterations: 0,
                ..ResizeOptions::default()
            },
            &constraints,
        )
        .expect("batch both independent equal-delay critical paths");

        assert_eq!(stats.upsizes, 2);
        assert!(stats.final_delay < stats.initial_delay);
        assert_eq!(
            interner.resolve(module.instances[0].type_name),
            Some("AND2_FAST")
        );
        assert_eq!(
            interner.resolve(module.instances[1].type_name),
            Some("AND2_FAST")
        );
        assert_eq!(stats.initial_clock_load, None);
        assert_eq!(stats.final_clock_load, None);
    }

    #[test]
    fn electrical_presizing_repairs_tied_outputs_without_a_global_first_move() {
        let source = r#"
module top(a, b, y0, y1);
  input a, b;
  output y0, y1;
  AND2 critical0 (.A(a), .B(b), .Y(y0));
  AND2 critical1 (.A(a), .B(b), .Y(y1));
endmodule
"#;
        let library = sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);

        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                sta_options: StaOptions {
                    module_output_load: 1.2,
                    ..StaOptions::default()
                },
                max_outer_iterations: 1,
                max_iterations: 1,
                max_area_iterations: 0,
                max_evaluations_per_iteration: 1,
                ..ResizeOptions::default()
            },
            &BufferTimingConstraints::default(),
        )
        .expect("repair both electrically overloaded tied output drivers");

        assert_eq!(stats.upsizes, 2);
        assert!(stats.final_delay < stats.initial_delay);
        assert!(
            module
                .instances
                .iter()
                .all(|instance| { interner.resolve(instance.type_name) == Some("AND2_FAST") })
        );
    }

    #[test]
    fn restores_best_complete_solution_after_an_unfinished_tied_path() {
        let source = r#"
module top(a, b, y0, y1);
  input a, b;
  output y0, y1;
  AND2 critical0 (.A(a), .B(b), .Y(y0));
  AND2 critical1 (.A(a), .B(b), .Y(y1));
endmodule
"#;
        let library = sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_iterations: 1,
                max_area_iterations: 0,
                ..ResizeOptions::default()
            },
            &BufferTimingConstraints::default(),
        )
        .expect("retain the best complete solution when a sizing batch cannot finish");

        assert_eq!(stats.upsizes, 0);
        assert!(stats.replacements.is_empty());
        assert_eq!(stats.final_delay, stats.initial_delay);
        assert_eq!(stats.final_area, stats.initial_area);
        assert!(
            module
                .instances
                .iter()
                .all(|instance| { interner.resolve(instance.type_name) == Some("AND2") })
        );
    }

    #[test]
    fn indexes_exact_state_and_pin_compatible_flip_flop_sizes() {
        let library = registered_sizing_library();
        let catalog = RegisterCellCatalog::new(&library)
            .expect("classify exact-state synthetic flip-flop families");
        let initial = catalog.by_name("DFF").expect("index the original FF");
        let family = catalog
            .family(initial)
            .map(|cell| cell.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(family, ["DFF", "DFF_FAST"]);
    }

    #[test]
    fn rejects_flip_flop_variants_with_different_state_behavior() {
        let mut builder = LibraryBuilder::from_library(registered_sizing_library());
        let mut inverted = builder
            .cells
            .iter()
            .find(|cell| cell.name == "DFF_FAST")
            .expect("find the stronger synthetic FF")
            .clone();
        inverted.name = "DFF_INVERTED".to_string();
        inverted.sequential[0].next_state = "!D".to_string();
        builder.cells.push(inverted);
        let library = builder.finish();
        let catalog = RegisterCellCatalog::new(&library).expect("classify incompatible FF");
        let initial = catalog.by_name("DFF").expect("index the original FF");
        let family = catalog
            .family(initial)
            .map(|cell| cell.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(family, ["DFF", "DFF_FAST"]);
    }

    #[test]
    fn rejects_negative_edge_and_asynchronously_cleared_flip_flops() {
        let mut builder = LibraryBuilder::from_library(registered_sizing_library());
        let reference = builder
            .cells
            .iter()
            .find(|cell| cell.name == "DFF")
            .expect("find the original synthetic FF")
            .clone();
        let mut negative_edge = reference.clone();
        negative_edge.name = "DFF_NEGATIVE".to_string();
        negative_edge.sequential[0].clock_expr = "!CLK".to_string();
        builder.cells.push(negative_edge);
        let mut asynchronous = reference;
        asynchronous.name = "DFF_CLEAR".to_string();
        asynchronous.sequential[0].clear_expr = "D".to_string();
        builder.cells.push(asynchronous);
        let library = builder.finish();
        let catalog = RegisterCellCatalog::new(&library)
            .expect("skip unsupported negative-edge and asynchronous FFs");

        assert!(catalog.by_name("DFF_NEGATIVE").is_none());
        assert!(catalog.by_name("DFF_CLEAR").is_none());
        assert_eq!(
            catalog
                .family(catalog.by_name("DFF").expect("retain the supported FF"))
                .map(|cell| cell.name.as_str())
                .collect::<Vec<_>>(),
            ["DFF", "DFF_FAST"]
        );
    }

    #[test]
    fn incremental_register_timing_matches_independent_full_sta() {
        let library = registered_sizing_library();
        let (module, nets, interner) = parse_module(high_fanout_register_source());
        let constraints = BufferTimingConstraints::default();
        let report =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("independently time high-fanout physical registers");
        let incremental = IncrementalRegisteredSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &constraints,
        )
        .expect("construct register-aware incremental timing");

        assert_eq!(
            incremental.score(),
            BoundaryTimingScore::from_report(&report)
        );
    }

    #[test]
    fn upsizes_a_critical_physical_flip_flop_without_limiting_clock_load() {
        let mut library = registered_sizing_library();
        let fast_cell = library
            .cells
            .iter()
            .position(|cell| cell.name == "DFF_FAST")
            .expect("find the stronger synthetic flip-flop");
        let clock_pin = library.cells[fast_cell]
            .pins
            .iter()
            .position(|pin| library.resolve_string(&pin.name) == "CLK")
            .expect("find the stronger flip-flop clock input");
        library.cells[fast_cell].pins[clock_pin].capacitance = Some(0.03);
        let (mut module, nets, mut interner) = parse_module(high_fanout_register_source());
        let before =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("time original loaded flip-flop");

        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("upsize exact-state critical launch flip-flop");
        let after =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("independently time the resized physical registers");

        assert!(stats.register_upsizes > 0);
        assert!(stats.upsizes >= stats.register_upsizes);
        assert!(
            after.max_register_to_register_delay.unwrap()
                < before.max_register_to_register_delay.unwrap()
        );
        assert!(
            module
                .instances
                .iter()
                .any(|instance| interner.resolve(instance.type_name) == Some("DFF_FAST"))
        );
        assert_eq!(stats.final_area, after.cell_area);
        assert!(stats.final_clock_load.unwrap() > stats.initial_clock_load.unwrap() * 1.05);
    }

    #[test]
    fn rejected_flip_flop_trial_restores_loads_setup_and_clock_timing() {
        let library = registered_sizing_library();
        let (module, nets, interner) = parse_module(high_fanout_register_source());
        let mut timing = IncrementalRegisteredSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("build incremental FF sizing state");
        let before_score = timing.score();
        let before_loads = timing.loads.clone();
        let before_bit_predecessors = timing.bit_predecessors.clone();
        let before_capture_predecessors = timing.capture_predecessors.clone();
        let before_clock_load = timing.clock_load;
        let fast_index = library
            .cells
            .iter()
            .position(|cell| cell.name == "DFF_FAST")
            .expect("find the stronger synthetic FF");

        let trial = timing
            .evaluate_cell_substitution(0, fast_index, false)
            .expect("evaluate a reversible physical flip-flop replacement");

        assert!(trial.score.register_to_register < before_score.register_to_register);
        assert_eq!(timing.score(), before_score);
        assert_eq!(timing.loads, before_loads);
        assert_eq!(timing.bit_predecessors, before_bit_predecessors);
        assert_eq!(timing.capture_predecessors, before_capture_predecessors);
        assert_eq!(timing.clock_load, before_clock_load);
        assert_eq!(library.cells[timing.instances[0].cell_index].name, "DFF");
    }

    #[test]
    fn recovers_area_by_downsizing_a_noncritical_flip_flop() {
        let source = r#"
module top(clk, a, out);
  input clk;
  input a;
  output [9:0] out;
  wire critical;
  DFF launch (.CLK(clk), .D(a), .Q(critical));
  DFF capture0 (.CLK(clk), .D(critical), .Q(out[0]));
  DFF capture1 (.CLK(clk), .D(critical), .Q(out[1]));
  DFF capture2 (.CLK(clk), .D(critical), .Q(out[2]));
  DFF capture3 (.CLK(clk), .D(critical), .Q(out[3]));
  DFF capture4 (.CLK(clk), .D(critical), .Q(out[4]));
  DFF capture5 (.CLK(clk), .D(critical), .Q(out[5]));
  DFF capture6 (.CLK(clk), .D(critical), .Q(out[6]));
  DFF capture7 (.CLK(clk), .D(critical), .Q(out[7]));
  DFF_FAST noncritical (.CLK(clk), .D(a), .Q(out[8]));
  DFF input_anchor (.CLK(clk), .D(a), .Q(out[9]));
endmodule
"#;
        let library = registered_sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let before =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("time the oversized noncritical flip-flop");

        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions {
                max_iterations: 0,
                ..ResizeOptions::default()
            },
            &BufferTimingConstraints::default(),
        )
        .expect("recover noncritical flip-flop area without worsening registered timing");
        let after =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("independently time the recovered registered netlist");
        let noncritical = module
            .instances
            .iter()
            .find(|instance| interner.resolve(instance.instance_name) == Some("noncritical"))
            .expect("preserve the noncritical physical register");

        assert!(
            stats.register_downsizes > 0,
            "expected noncritical flip-flop area recovery; stats={stats:?}; noncritical={:?}",
            interner.resolve(noncritical.type_name)
        );
        assert!(stats.downsizes >= stats.register_downsizes);
        assert_eq!(interner.resolve(noncritical.type_name), Some("DFF"));
        assert!(
            after.max_register_to_register_delay.unwrap()
                <= before.max_register_to_register_delay.unwrap()
        );
        assert_eq!(stats.final_area, after.cell_area);
    }

    #[test]
    fn evaluates_packed_output_constraints_before_rolling_back_a_trial() {
        let source = r#"
module top(clk, a, out);
  input clk;
  input a;
  output [8:0] out;
  DFF launch (.CLK(clk), .D(a), .Q(out[8]));
  DFF capture0 (.CLK(clk), .D(out[8]), .Q(out[0]));
  DFF capture1 (.CLK(clk), .D(out[8]), .Q(out[1]));
  DFF capture2 (.CLK(clk), .D(out[8]), .Q(out[2]));
  DFF capture3 (.CLK(clk), .D(out[8]), .Q(out[3]));
  DFF capture4 (.CLK(clk), .D(out[8]), .Q(out[4]));
  DFF capture5 (.CLK(clk), .D(out[8]), .Q(out[5]));
  DFF capture6 (.CLK(clk), .D(out[8]), .Q(out[6]));
  DFF capture7 (.CLK(clk), .D(out[8]), .Q(out[7]));
endmodule
"#;
        let library = registered_sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        let constraints = BufferTimingConstraints {
            primary_output_required: BTreeMap::from([("out_8".to_string(), 20.0)]),
            ..BufferTimingConstraints::default()
        };
        let mut timing = IncrementalRegisteredSta::new(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            &constraints,
        )
        .expect("build packed-output registered timing");
        let fast_index = library
            .cells
            .iter()
            .position(|cell| cell.name == "DFF_FAST")
            .expect("find the stronger synthetic FF");

        assert!(!timing.satisfies_constraints(timing.score()));
        let trial = timing
            .evaluate_cell_substitution(0, fast_index, false)
            .expect("evaluate the exact constrained launch replacement");
        assert!(trial.constraints_satisfied);
        assert!(!timing.satisfies_constraints(timing.score()));

        let stats = resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions::default(),
            &constraints,
        )
        .expect("meet the real packed-output deadline by sizing the launch FF");
        let launch_registers = (0..module.instances.len()).collect::<Vec<_>>();
        let report = analyze_register_boundary_max_arrival(
            &module,
            &nets,
            &interner,
            &library,
            StaOptions::default(),
            false,
            &launch_registers,
        )
        .expect("independently verify the resized packed-output deadline");
        let output = report
            .timing_for_output_bit("out_8")
            .expect("report the constrained packed-output bit");

        assert!(stats.register_upsizes > 0);
        assert!(output.rise.arrival.max(output.fall.arrival) <= 20.0);
    }

    #[test]
    fn register_sizing_preserves_packed_ports_and_constant_outputs() {
        let source = r#"
module top(clk, a, out);
  input clk;
  input a;
  output [8:0] out;
  wire root;
  assign out[8] = 1'b0;
  DFF launch (.CLK(clk), .D(a), .Q(root));
  DFF capture0 (.CLK(clk), .D(root), .Q(out[0]));
  DFF capture1 (.CLK(clk), .D(root), .Q(out[1]));
  DFF capture2 (.CLK(clk), .D(root), .Q(out[2]));
  DFF capture3 (.CLK(clk), .D(root), .Q(out[3]));
  DFF capture4 (.CLK(clk), .D(root), .Q(out[4]));
  DFF capture5 (.CLK(clk), .D(root), .Q(out[5]));
  DFF capture6 (.CLK(clk), .D(root), .Q(out[6]));
  DFF capture7 (.CLK(clk), .D(root), .Q(out[7]));
endmodule
"#;
        let library = registered_sizing_library();
        let (mut module, nets, mut interner) = parse_module(source);
        resize_timing_aware_netlist(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &ResizeOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("size physical FFs without changing packed outputs");
        let output = emit_module_as_netlist_text(&module, &nets, &interner)
            .expect("render sized packed-output module");

        assert!(output.contains("output [8:0] out;"));
        assert!(output.contains("assign out[8] = 1'b0;"));
    }

    #[test]
    fn register_sizing_is_deterministic() {
        let library = registered_sizing_library();
        let render = || {
            let (mut module, nets, mut interner) = parse_module(high_fanout_register_source());
            resize_timing_aware_netlist(
                &mut module,
                &nets,
                &mut interner,
                &library,
                &ResizeOptions::default(),
                &BufferTimingConstraints::default(),
            )
            .expect("deterministically size synchronous registers");
            emit_module_as_netlist_text(&module, &nets, &interner)
                .expect("render deterministically resized module")
        };

        assert_eq!(render(), render());
    }
}
