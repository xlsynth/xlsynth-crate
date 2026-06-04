// SPDX-License-Identifier: Apache-2.0

//! Basic static timing analysis (STA) for parsed gate-level netlists.
//!
//! This module implements a limited-scope max-arrival analysis:
//! - Uses Liberty combinational timing arcs (`cell_rise`, `cell_fall`,
//!   `rise_transition`, `fall_transition`).
//! - Propagates rise/fall arrival and transition values bit-by-bit, then
//!   aggregates report timing back to parsed nets.
//! - Assumes fixed transition at primary-input sources.
//! - Uses summed input-pin capacitance on each net plus a fixed module-output
//!   load to query timing tables; zero output load is evaluated at each table's
//!   minimum characterized load coordinate.
//! - In register-boundary analysis, models flip-flop clock-to-output arcs and
//!   setup checks using an ideal clock edge evaluated at each table's minimum
//!   characterized clock transition. Hold, skew, and physical clock delivery
//!   are outside this model.
//! - In combinational-only analysis, rejects sequential output pins.
//! - Uses Liberty `when` predicates when known scalar constants make an arc
//!   provably false; arcs with true or unknown predicates remain possible.
//! - Treats arc-less output pins with constant Liberty functions as zero-delay
//!   sources, covering tie-high and tie-low cells without cell-name matching.
//! - Supports acyclic combinational timing dependencies between output pins of
//!   the same cell, including dependencies through unconnected outputs.
//! - Repairs non-monotone delay/slew tables during lookup with a conservative
//!   coordinatewise upper envelope, because later frontier reduction relies on
//!   larger transition/load queries not producing smaller delay/slew values.
//! - Clamps below-minimum coordinates; for above-maximum delay/slew queries,
//!   extrapolates one varying axis or holds multiple varying axes at their
//!   characterized upper boundary. Setup queries remain clamped.
//! - Accepts literal, alias, slice, and concat continuous assigns as zero-delay
//!   bit sources so mapped netlists emitted by tools such as Yosys/ABC can
//!   preserve constants and wire aliases without needing synthetic cells.
//!
//! At a high level, the analysis is:
//! 1. Index each instance's cell/pin connectivity, recording one driver and all
//!    loads for each net.
//! 2. Build a combinational dependency graph between instances and compute the
//!    edge-specific capacitive load on each net.
//! 3. Seed primary inputs with zero arrival and the configured source
//!    transition; reject undriven non-input nets that feed logic or module
//!    outputs.
//! 4. Visit instances in topological order. For each output timing arc:
//!    - choose the relevant input edge candidates from the arc sense
//!      (`positive_unate`, `negative_unate`, or `non_unate`);
//!    - query Liberty delay and transition tables using input transition and
//!      output load;
//!    - produce output `(arrival, transition)` candidates and retain the
//!      non-dominated frontier while combining related pins/arcs. Keeping that
//!      frontier preserves cases where the latest-arriving candidate and the
//!      highest-transition candidate differ; pruning dominated candidates only
//!      removes candidates that are worse in both dimensions under the intended
//!      monotone timing model.
//! 5. Before storing an output-net result, collapse each rise/fall candidate
//!    set to its conservative envelope: maximum arrival and maximum transition.
//! 6. Report the maximum rise/fall arrival observed at module outputs.

use crate::liberty::LibraryWithTimingData;
use crate::liberty::cell_formula::parse_formula;
use crate::liberty::timing_table::TimingTableArrayView;
use crate::liberty_proto::{LuTableTemplate, Pin, PinDirection, TimingArc, TimingTable};
use crate::netlist::normalized::{BitExpr, BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{Net, NetIndex, NetlistModule, PortDirection};
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeTiming {
    pub arrival: f64,
    pub transition: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignalTiming {
    pub rise: EdgeTiming,
    pub fall: EdgeTiming,
}

/// Component delays making up one launched register-to-capture-register path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct RegisterPathDelayBreakdown {
    pub clock_to_output_delay: f64,
    pub combinational_delay: f64,
    pub setup_delay: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EdgeTimingCandidate {
    timing: EdgeTiming,
    register_path_breakdown: Option<RegisterPathDelayBreakdown>,
}

impl EdgeTimingCandidate {
    fn from_timing(timing: EdgeTiming) -> Self {
        Self {
            timing,
            register_path_breakdown: None,
        }
    }

    fn from_register_launch(timing: EdgeTiming) -> Self {
        Self {
            register_path_breakdown: Some(RegisterPathDelayBreakdown {
                clock_to_output_delay: timing.arrival,
                ..Default::default()
            }),
            timing,
        }
    }

    fn from_primary_input_launch(timing: EdgeTiming) -> Self {
        Self {
            timing,
            register_path_breakdown: Some(RegisterPathDelayBreakdown::default()),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct EdgeTimingSet {
    values: Vec<EdgeTimingCandidate>,
}

impl EdgeTimingSet {
    fn from_single(edge: EdgeTiming) -> Self {
        let mut set = Self::default();
        set.insert(EdgeTimingCandidate::from_timing(edge));
        set
    }

    fn from_primary_input_launch(edge: EdgeTiming) -> Self {
        let mut set = Self::default();
        set.insert(EdgeTimingCandidate::from_primary_input_launch(edge));
        set
    }

    fn insert(&mut self, candidate: EdgeTimingCandidate) {
        if self
            .values
            .iter()
            .any(|existing| existing.timing == candidate.timing)
        {
            return;
        }

        if self
            .values
            .iter()
            .any(|existing| edge_timing_dominates(existing.timing, candidate.timing))
        {
            return;
        }
        self.values
            .retain(|existing| !edge_timing_dominates(candidate.timing, existing.timing));

        self.values.push(candidate);
        self.values.sort_by(|lhs, rhs| {
            lhs.timing
                .arrival
                .partial_cmp(&rhs.timing.arrival)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    lhs.timing
                        .transition
                        .partial_cmp(&rhs.timing.transition)
                        .unwrap_or(Ordering::Equal)
                })
        });
    }

    fn extend_from(&mut self, rhs: &Self) {
        for edge in &rhs.values {
            self.insert(*edge);
        }
    }

    fn max_arrival_candidate(&self) -> Option<EdgeTimingCandidate> {
        self.values
            .iter()
            .copied()
            .reduce(choose_worse_edge_timing_candidate_by_arrival)
    }

    fn max_arrival_edge(&self) -> Option<EdgeTiming> {
        self.max_arrival_candidate()
            .map(|candidate| candidate.timing)
    }

    fn iter(&self) -> impl Iterator<Item = EdgeTimingCandidate> + '_ {
        self.values.iter().copied()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct SignalTimingSet {
    rise: EdgeTimingSet,
    fall: EdgeTimingSet,
}

impl SignalTimingSet {
    fn from_single(signal: SignalTiming) -> Self {
        Self {
            rise: EdgeTimingSet::from_single(signal.rise),
            fall: EdgeTimingSet::from_single(signal.fall),
        }
    }

    fn from_primary_input_launch(signal: SignalTiming) -> Self {
        Self {
            rise: EdgeTimingSet::from_primary_input_launch(signal.rise),
            fall: EdgeTimingSet::from_primary_input_launch(signal.fall),
        }
    }

    fn merge(mut self, rhs: &Self) -> Self {
        self.rise.extend_from(&rhs.rise);
        self.fall.extend_from(&rhs.fall);
        self
    }

    fn as_report_signal_timing(&self) -> Option<SignalTiming> {
        Some(SignalTiming {
            rise: self.rise.max_arrival_edge()?,
            fall: self.fall.max_arrival_edge()?,
        })
    }

    fn worst_arrival_candidate(&self) -> Option<EdgeTimingCandidate> {
        Some(choose_worse_edge_timing_candidate_by_arrival(
            self.rise.max_arrival_candidate()?,
            self.fall.max_arrival_candidate()?,
        ))
    }
}

fn edge_timing_dominates(lhs: EdgeTiming, rhs: EdgeTiming) -> bool {
    (lhs.arrival >= rhs.arrival && lhs.transition >= rhs.transition)
        && (lhs.arrival > rhs.arrival || lhs.transition > rhs.transition)
}

fn parse_trace_env_flag(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sta_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("XLSYNTH_G8R_STA_TRACE")
            .map(|value| parse_trace_env_flag(value.as_str()))
            .unwrap_or(false)
    })
}

fn sta_trace(msg: impl FnOnce() -> String) {
    if sta_trace_enabled() {
        log::info!(target: "xlsynth_g8r::netlist::sta_trace", "{}", msg());
    }
}

fn format_edge_timing(edge: EdgeTiming) -> String {
    format!("(arr={:.6},tr={:.6})", edge.arrival, edge.transition)
}

fn format_edge_timing_set(set: &EdgeTimingSet) -> String {
    let parts: Vec<String> = set
        .values
        .iter()
        .map(|candidate| format_edge_timing(candidate.timing))
        .collect();
    format!("[{}]", parts.join(", "))
}

fn format_optional_edge_timing(edge: Option<EdgeTiming>) -> String {
    edge.map(format_edge_timing)
        .unwrap_or_else(|| "<none>".to_string())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaOptions {
    /// Transition applied to primary-input source nets.
    pub primary_input_transition: f64,
    /// Extra load added to nets attached to module outputs.
    pub module_output_load: f64,
}

impl Default for StaOptions {
    fn default() -> Self {
        Self {
            primary_input_transition: 0.01,
            module_output_load: 0.0,
        }
    }
}

/// Counts timing-table coordinate queries evaluated outside characterized
/// bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TimingQueryDiagnosticCounts {
    pub delay_slew_below_min_clamp_count: usize,
    pub delay_slew_single_above_max_extrapolation_count: usize,
    pub delay_slew_multiple_above_max_clamp_count: usize,
    pub setup_below_min_clamp_count: usize,
    pub setup_above_max_clamp_count: usize,
}

impl std::ops::AddAssign for TimingQueryDiagnosticCounts {
    fn add_assign(&mut self, rhs: Self) {
        self.delay_slew_below_min_clamp_count += rhs.delay_slew_below_min_clamp_count;
        self.delay_slew_single_above_max_extrapolation_count +=
            rhs.delay_slew_single_above_max_extrapolation_count;
        self.delay_slew_multiple_above_max_clamp_count +=
            rhs.delay_slew_multiple_above_max_clamp_count;
        self.setup_below_min_clamp_count += rhs.setup_below_min_clamp_count;
        self.setup_above_max_clamp_count += rhs.setup_above_max_clamp_count;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StaReport {
    pub net_timing: Vec<Option<SignalTiming>>,
    pub worst_output_arrival: f64,
    pub worst_output_breakdown: Option<RegisterPathDelayBreakdown>,
    pub worst_register_input_arrival: f64,
    pub worst_register_input_breakdown: Option<RegisterPathDelayBreakdown>,
    pub register_input_arrivals: Vec<Option<f64>>,
    pub register_input_breakdowns: Vec<Option<RegisterPathDelayBreakdown>>,
    pub timing_query_diagnostic_counts: TimingQueryDiagnosticCounts,
    pub cell_levels: usize,
}

impl StaReport {
    pub fn timing_for_net(&self, net: NetIndex) -> Option<SignalTiming> {
        self.net_timing.get(net.0).copied().flatten()
    }
}

struct StaLibraryIndex<'a> {
    library: &'a crate::liberty_proto::Library,
    cell_by_name: HashMap<String, usize>,
    pin_by_cell: Vec<HashMap<String, usize>>,
}

impl<'a> StaLibraryIndex<'a> {
    fn new(library: &'a crate::liberty_proto::Library) -> Result<Self> {
        let mut cell_by_name = HashMap::new();
        let mut pin_by_cell = Vec::with_capacity(library.cells.len());
        for (cell_idx, cell) in library.cells.iter().enumerate() {
            if cell_by_name.insert(cell.name.clone(), cell_idx).is_some() {
                return Err(anyhow!(
                    "library defines cell '{}' more than once; duplicate cell names are unsupported in basic STA",
                    cell.name
                ));
            }
            let mut pin_map = HashMap::new();
            for (pin_idx, pin) in cell.pins.iter().enumerate() {
                if pin_map.insert(pin.name.clone(), pin_idx).is_some() {
                    return Err(anyhow!(
                        "library cell '{}' defines pin '{}' more than once; duplicate pin names are unsupported in basic STA",
                        cell.name,
                        pin.name
                    ));
                }
            }
            pin_by_cell.push(pin_map);
        }
        Ok(Self {
            library,
            cell_by_name,
            pin_by_cell,
        })
    }

    fn cell_index(&self, cell_name: &str) -> Option<usize> {
        self.cell_by_name.get(cell_name).copied()
    }

    fn pin(&self, cell_idx: usize, pin_name: &str) -> Option<&Pin> {
        let pin_idx = *self.pin_by_cell.get(cell_idx)?.get(pin_name)?;
        self.library.cells.get(cell_idx)?.pins.get(pin_idx)
    }

    fn pin_index(&self, cell_idx: usize, pin_name: &str) -> Option<usize> {
        self.pin_by_cell.get(cell_idx)?.get(pin_name).copied()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EdgeLoadCapacitance {
    rise: f64,
    fall: f64,
}

#[derive(Clone, Debug)]
struct NetEndpoint {
    inst_idx: usize,
    pin_name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BitAssignSource {
    Literal(bool),
    Unknown,
    Alias(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedTimingSource {
    Literal(bool),
    Unknown,
    Bit(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimingSourceVisitState {
    Unvisited,
    Visiting,
    Resolved(ResolvedTimingSource),
}

#[derive(Clone, Debug)]
struct UnsupportedAssignSource {
    lhs_bits: Vec<usize>,
    rendered_lhs: String,
}

#[derive(Clone, Debug)]
struct AssignSourceAnalysis {
    bit_sources: Vec<Option<BitAssignSource>>,
    unsupported_sources: Vec<UnsupportedAssignSource>,
}

type PinBitSource = BitSource;

pub fn analyze_combinational_max_arrival(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &LibraryWithTimingData,
    options: StaOptions,
) -> Result<StaReport> {
    analyze_combinational_max_arrival_proto(module, nets, interner, library.as_proto(), options)
}

/// Analyzes combinational segments launched by primary inputs and/or selected
/// sequential instances, treating sequential inputs as capture boundaries.
pub fn analyze_register_boundary_max_arrival(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &LibraryWithTimingData,
    options: StaOptions,
    launch_primary_inputs: bool,
    launch_register_instances: &[usize],
) -> Result<StaReport> {
    let launch_register_instances: HashSet<usize> =
        launch_register_instances.iter().copied().collect();
    analyze_max_arrival_proto_with_mode(
        module,
        nets,
        interner,
        library.as_proto(),
        options,
        StaAnalysisMode::RegisterBoundaries {
            launch_primary_inputs,
            launch_register_instances: &launch_register_instances,
        },
    )
}

/// Returns whether a cell should form a timing boundary for register-aware
/// segment analysis.
pub(crate) fn is_sequential_boundary_cell(cell: &crate::liberty_proto::Cell) -> bool {
    !cell.sequential.is_empty()
        || cell.pins.iter().any(|pin| {
            pin.direction == PinDirection::Output as i32
                && pin
                    .timing_arcs
                    .iter()
                    .any(|arc| matches!(arc.timing_type.as_str(), "rising_edge" | "falling_edge"))
        })
}

enum StaAnalysisMode<'a> {
    CombinationalOnly,
    RegisterBoundaries {
        launch_primary_inputs: bool,
        launch_register_instances: &'a HashSet<usize>,
    },
}

impl StaAnalysisMode<'_> {
    fn uses_register_boundaries(&self) -> bool {
        matches!(self, Self::RegisterBoundaries { .. })
    }

    fn launches_primary_inputs(&self) -> bool {
        match self {
            Self::CombinationalOnly => true,
            Self::RegisterBoundaries {
                launch_primary_inputs,
                ..
            } => *launch_primary_inputs,
        }
    }

    fn launches_register(&self, inst_idx: usize) -> bool {
        match self {
            Self::CombinationalOnly => false,
            Self::RegisterBoundaries {
                launch_register_instances,
                ..
            } => launch_register_instances.contains(&inst_idx),
        }
    }
}

fn analyze_combinational_max_arrival_proto(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &crate::liberty_proto::Library,
    options: StaOptions,
) -> Result<StaReport> {
    analyze_max_arrival_proto_with_mode(
        module,
        nets,
        interner,
        library,
        options,
        StaAnalysisMode::CombinationalOnly,
    )
}

fn analyze_max_arrival_proto_with_mode(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &crate::liberty_proto::Library,
    options: StaOptions,
    analysis_mode: StaAnalysisMode<'_>,
) -> Result<StaReport> {
    if !options.primary_input_transition.is_finite() {
        return Err(anyhow!(
            "primary_input_transition must be finite; got {}",
            options.primary_input_transition
        ));
    }
    if options.primary_input_transition < 0.0 {
        return Err(anyhow!(
            "primary_input_transition must be non-negative; got {}",
            options.primary_input_transition
        ));
    }
    if !options.module_output_load.is_finite() {
        return Err(anyhow!(
            "module_output_load must be finite; got {}",
            options.module_output_load
        ));
    }
    if options.module_output_load < 0.0 {
        return Err(anyhow!(
            "module_output_load must be non-negative; got {}",
            options.module_output_load
        ));
    }

    let lib = StaLibraryIndex::new(library)?;
    let normalized = NormalizedNetlistModule::new(module, nets, interner)?;
    let instance_count = normalized.instances.len();
    let bit_count = normalized.bit_count();
    let assign_analysis = analyze_assign_sources(&normalized, nets, interner)?;
    let assign_sources = assign_analysis.bit_sources;
    let resolved_timing_sources =
        resolve_timing_sources(assign_sources.as_slice(), &normalized, nets, interner)?;

    let mut instance_cell_indices: Vec<usize> = Vec::with_capacity(instance_count);
    let mut instance_cell_names: Vec<String> = Vec::with_capacity(instance_count);
    let mut instance_pin_sources: Vec<HashMap<String, Vec<PinBitSource>>> =
        Vec::with_capacity(instance_count);
    let mut instance_known_pin_values: Vec<HashMap<String, bool>> =
        Vec::with_capacity(instance_count);
    let mut instance_timing_related_input_pins: Vec<HashSet<String>> =
        Vec::with_capacity(instance_count);
    let mut instance_output_pin_orders: Vec<Vec<usize>> = Vec::with_capacity(instance_count);
    let mut instance_is_sequential: Vec<bool> = Vec::with_capacity(instance_count);

    let mut bit_drivers: Vec<Vec<NetEndpoint>> = vec![Vec::new(); bit_count];
    let mut bit_loads: Vec<Vec<NetEndpoint>> = vec![Vec::new(); bit_count];

    for (inst_idx, inst) in normalized.instances.iter().enumerate() {
        let cell_name = resolve_symbol(interner, inst.type_name, "cell type")?;
        let cell_idx = lib.cell_index(cell_name.as_str()).ok_or_else(|| {
            anyhow!(
                "instance '{}' references unknown cell '{}'",
                resolve_symbol(interner, inst.instance_name, "instance name")
                    .unwrap_or_else(|_| "<unknown>".to_string()),
                cell_name
            )
        })?;
        instance_cell_indices.push(cell_idx);
        instance_cell_names.push(cell_name.clone());
        let is_sequential = is_sequential_boundary_cell(&lib.library.cells[cell_idx]);
        instance_is_sequential.push(is_sequential);
        let mut timing_related_input_pins = HashSet::new();
        if !(analysis_mode.uses_register_boundaries() && is_sequential) {
            for output_pin in lib.library.cells[cell_idx]
                .pins
                .iter()
                .filter(|pin| pin.direction == PinDirection::Output as i32)
            {
                for arc in output_pin.timing_arcs.iter().filter(|arc| {
                    StaTimingType::from_raw(arc.timing_type.as_str()).is_combinational()
                }) {
                    for related_pin_name in split_related_pin_names(arc.related_pin.as_str()) {
                        let related_pin = lib.pin(cell_idx, related_pin_name).ok_or_else(|| {
                            anyhow!(
                                "cell '{}' output pin '{}' has timing arc with unknown related pin '{}'",
                                cell_name,
                                output_pin.name,
                                related_pin_name
                            )
                        })?;
                        if related_pin.direction == PinDirection::Input as i32 {
                            timing_related_input_pins.insert(related_pin_name.to_string());
                        } else if related_pin.direction != PinDirection::Output as i32 {
                            return Err(anyhow!(
                                "cell '{}' output pin '{}' has timing arc related pin '{}' with unsupported direction value {}",
                                cell_name,
                                output_pin.name,
                                related_pin_name,
                                related_pin.direction
                            ));
                        }
                    }
                }
            }
        }

        let mut pin_sources: HashMap<String, Vec<PinBitSource>> = HashMap::new();
        let mut known_pin_values = HashMap::new();
        for connection in &inst.connections {
            let pin_name = resolve_symbol(interner, connection.port, "pin name")?;
            if pin_sources.contains_key(pin_name.as_str()) {
                return Err(anyhow!(
                    "instance '{}' of '{}' connects pin '{}' more than once; duplicate pin bindings are unsupported in basic STA",
                    resolve_symbol(interner, inst.instance_name, "instance name")
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                    cell_name,
                    pin_name
                ));
            }
            let pin = lib.pin(cell_idx, pin_name.as_str()).ok_or_else(|| {
                anyhow!(
                    "instance '{}' of '{}' references unknown pin '{}'",
                    resolve_symbol(interner, inst.instance_name, "instance name")
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                    cell_name,
                    pin_name
                )
            })?;
            let pin_bit_sources = connection.bits.clone();
            let pin_bit_sources = if pin.direction == PinDirection::Input as i32 {
                pin_bit_sources
                    .into_iter()
                    .map(|source| {
                        canonicalize_pin_bit_source(source, resolved_timing_sources.as_slice())
                    })
                    .collect()
            } else {
                pin_bit_sources
            };
            if pin_bit_sources.len() > 1 {
                return Err(anyhow!(
                    "instance '{}' pin '{}.{}' connects {} bits; basic STA requires scalar cell pin connections after bit expansion",
                    resolve_symbol(interner, inst.instance_name, "instance name")
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                    cell_name,
                    pin_name,
                    pin_bit_sources.len()
                ));
            }
            if pin.direction == PinDirection::Output as i32
                && !matches!(pin_bit_sources.as_slice(), [PinBitSource::Bit(_)])
            {
                return Err(anyhow!(
                    "instance '{}' output pin '{}.{}' uses unsupported literal or unconnected binding",
                    resolve_symbol(interner, inst.instance_name, "instance name")
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                    cell_name,
                    pin_name
                ));
            }
            for source in &pin_bit_sources {
                match pin.direction {
                    d if d == PinDirection::Output as i32 => match source {
                        PinBitSource::Bit(bit_idx) => bit_drivers[*bit_idx].push(NetEndpoint {
                            inst_idx,
                            pin_name: pin_name.clone(),
                        }),
                        PinBitSource::Literal(_) | PinBitSource::Unknown => unreachable!(),
                    },
                    d if d == PinDirection::Input as i32 => match source {
                        PinBitSource::Bit(bit_idx) => bit_loads[*bit_idx].push(NetEndpoint {
                            inst_idx,
                            pin_name: pin_name.clone(),
                        }),
                        PinBitSource::Literal(value) => {
                            known_pin_values.insert(pin_name.clone(), *value);
                        }
                        PinBitSource::Unknown => {}
                    },
                    _ => {
                        return Err(anyhow!(
                            "instance '{}' pin '{}.{}' has unsupported direction value {}",
                            resolve_symbol(interner, inst.instance_name, "instance name")
                                .unwrap_or_else(|_| "<unknown>".to_string()),
                            cell_name,
                            pin_name,
                            pin.direction
                        ));
                    }
                }
            }
            pin_sources.insert(pin_name, pin_bit_sources);
        }
        let output_pin_order = if analysis_mode.uses_register_boundaries() && is_sequential {
            lib.library.cells[cell_idx]
                .pins
                .iter()
                .enumerate()
                .filter(|(_, pin)| {
                    pin.direction == PinDirection::Output as i32
                        && pin_sources.contains_key(pin.name.as_str())
                })
                .map(|(pin_idx, _)| pin_idx)
                .collect()
        } else {
            combinational_output_pin_evaluation_order(&lib, cell_idx, &pin_sources)?
        };
        instance_pin_sources.push(pin_sources);
        instance_known_pin_values.push(known_pin_values);
        instance_timing_related_input_pins.push(timing_related_input_pins);
        instance_output_pin_orders.push(output_pin_order);
    }

    let mut module_output_bits: Vec<usize> = Vec::new();
    let mut has_module_output = vec![false; bit_count];
    let mut is_module_input = vec![false; bit_count];
    for port in &normalized.ports {
        let port_name = resolve_symbol(interner, port.name, "port name")
            .unwrap_or_else(|_| "<unknown>".to_string());
        match port.direction {
            PortDirection::Input => {
                for bit_idx in &port.bits {
                    is_module_input[*bit_idx] = true;
                }
            }
            PortDirection::Output => {
                for bit_idx in &port.bits {
                    if !has_module_output[*bit_idx] {
                        has_module_output[*bit_idx] = true;
                        module_output_bits.push(*bit_idx);
                    }
                }
            }
            PortDirection::Inout => {
                return Err(anyhow!(
                    "module port '{}' is inout; basic STA only supports input and output ports",
                    port_name
                ));
            }
        }
    }
    module_output_bits.sort_unstable();
    let mut has_resolved_module_output = vec![false; bit_count];
    for bit_idx in &module_output_bits {
        if let ResolvedTimingSource::Bit(source_bit_idx) = resolved_timing_sources[*bit_idx] {
            has_resolved_module_output[source_bit_idx] = true;
        }
    }

    reject_live_unsupported_assigns(
        assign_analysis.unsupported_sources.as_slice(),
        assign_sources.as_slice(),
        bit_loads.as_slice(),
        has_module_output.as_slice(),
        &normalized,
        nets,
        interner,
    )?;

    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); instance_count];
    let mut indegree: Vec<usize> = vec![0; instance_count];
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
    for (bit_idx, drivers) in bit_drivers.iter().enumerate() {
        let has_assign_source = assign_sources[bit_idx].is_some();
        if is_module_input[bit_idx] && (!drivers.is_empty() || has_assign_source) {
            let bit_name = normalized.render_bit(bit_idx, nets, interner);
            return Err(anyhow!(
                "module input bit '{}' also has an internal driver; basic STA does not support multiply driven primary inputs",
                bit_name
            ));
        }
        if drivers.len() > 1 || (!drivers.is_empty() && has_assign_source) {
            let bit_name = normalized.render_bit(bit_idx, nets, interner);
            return Err(anyhow!(
                "net bit '{}' has {} drivers; wired multi-driver nets are unsupported in basic STA",
                bit_name,
                drivers.len() + usize::from(has_assign_source)
            ));
        }
    }

    for (source_bit_idx, loads) in bit_loads.iter().enumerate() {
        let Some(driver) = bit_drivers[source_bit_idx].first() else {
            continue;
        };
        for load in loads {
            if !instance_timing_related_input_pins[load.inst_idx].contains(&load.pin_name) {
                continue;
            }
            if load.inst_idx == driver.inst_idx {
                continue;
            }
            if seen_edges.insert((driver.inst_idx, load.inst_idx)) {
                successors[driver.inst_idx].push(load.inst_idx);
                indegree[load.inst_idx] += 1;
            }
        }
    }

    let mut bit_load_capacitance = vec![EdgeLoadCapacitance::default(); bit_count];
    for (load_source_bit_idx, loads) in bit_loads.iter().enumerate() {
        let mut cap = EdgeLoadCapacitance::default();
        for load in loads {
            let cell_idx = instance_cell_indices[load.inst_idx];
            let pin = lib.pin(cell_idx, load.pin_name.as_str()).ok_or_else(|| {
                anyhow!(
                    "could not resolve load pin '{}.{}' while computing output load",
                    instance_cell_names[load.inst_idx],
                    load.pin_name
                )
            })?;
            let pin_cap = effective_input_capacitance_by_edge(
                pin,
                &format!(
                    "load pin '{}.{}'",
                    instance_cell_names[load.inst_idx], load.pin_name
                ),
            )?;
            cap.rise += pin_cap.rise;
            cap.fall += pin_cap.fall;
        }
        bit_load_capacitance[load_source_bit_idx].rise += cap.rise;
        bit_load_capacitance[load_source_bit_idx].fall += cap.fall;
    }
    for bit_idx in &module_output_bits {
        let ResolvedTimingSource::Bit(output_source_bit_idx) = resolved_timing_sources[*bit_idx]
        else {
            continue;
        };
        bit_load_capacitance[output_source_bit_idx].rise += options.module_output_load;
        bit_load_capacitance[output_source_bit_idx].fall += options.module_output_load;
    }

    let source_timing = SignalTiming {
        rise: EdgeTiming {
            arrival: 0.0,
            transition: options.primary_input_transition,
        },
        fall: EdgeTiming {
            arrival: 0.0,
            transition: options.primary_input_transition,
        },
    };
    let source_timing_set = if analysis_mode.uses_register_boundaries() {
        SignalTimingSet::from_primary_input_launch(source_timing)
    } else {
        SignalTimingSet::from_single(source_timing)
    };
    let literal_source_timing_set = SignalTimingSet::from_single(SignalTiming {
        rise: EdgeTiming {
            arrival: 0.0,
            transition: 0.0,
        },
        fall: EdgeTiming {
            arrival: 0.0,
            transition: 0.0,
        },
    });
    let mut bit_timing_sets: Vec<Option<SignalTimingSet>> = vec![None; bit_count];

    for (bit_idx, drivers) in bit_drivers.iter().enumerate() {
        if !drivers.is_empty() {
            continue;
        }
        match resolved_timing_sources[bit_idx] {
            ResolvedTimingSource::Literal(_) | ResolvedTimingSource::Unknown => continue,
            ResolvedTimingSource::Bit(source_bit_idx) if source_bit_idx != bit_idx => continue,
            ResolvedTimingSource::Bit(_) => {}
        }
        if is_module_input[bit_idx] && analysis_mode.launches_primary_inputs() {
            bit_timing_sets[bit_idx] = Some(source_timing_set.clone());
            continue;
        }
        if (!bit_loads[bit_idx].is_empty() || has_resolved_module_output[bit_idx])
            && !analysis_mode.uses_register_boundaries()
        {
            return Err(anyhow!(
                "net bit '{}' is undriven and is not a module input; basic STA does not support floating source nets",
                normalized.render_bit(bit_idx, nets, interner)
            ));
        }
    }

    let mut queue = VecDeque::new();
    let mut instance_levels = vec![1usize; instance_count];
    let mut validated_timing_tables: HashSet<*const TimingTable> = HashSet::new();
    let mut timing_query_diagnostic_counts = TimingQueryDiagnosticCounts::default();
    for (idx, deg) in indegree.iter().enumerate() {
        if *deg == 0 {
            queue.push_back(idx);
        }
    }

    let mut processed = 0usize;
    while let Some(inst_idx) = queue.pop_front() {
        processed += 1;

        let cell_idx = instance_cell_indices[inst_idx];
        let cell_name = &instance_cell_names[inst_idx];
        let inst_pin_map = &instance_pin_sources[inst_idx];
        let known_pin_values = &instance_known_pin_values[inst_idx];
        let instance = &normalized.instances[inst_idx];
        let instance_name = resolve_symbol(interner, instance.instance_name, "instance name")
            .unwrap_or_else(|_| "<unknown>".to_string());

        let mut local_output_timing_sets: HashMap<String, SignalTimingSet> = HashMap::new();
        for pin_idx in &instance_output_pin_orders[inst_idx] {
            let pin = &lib.library.cells[cell_idx].pins[*pin_idx];
            let output_sources = inst_pin_map.get(pin.name.as_str());
            if analysis_mode.uses_register_boundaries() && instance_is_sequential[inst_idx] {
                if analysis_mode.launches_register(inst_idx) {
                    let output_bit_idx = match output_sources.and_then(|sources| sources.first()) {
                        Some(PinBitSource::Bit(output_bit_idx)) => *output_bit_idx,
                        Some(PinBitSource::Literal(_) | PinBitSource::Unknown) | None => {
                            unreachable!("connected sequential output pins have net bit sources")
                        }
                    };
                    let output_load = bit_load_capacitance[output_bit_idx];
                    bit_timing_sets[output_bit_idx] = Some(evaluate_register_launch_output_set(
                        lib.library,
                        cell_name,
                        pin,
                        known_pin_values,
                        output_load,
                        &mut validated_timing_tables,
                        &mut timing_query_diagnostic_counts,
                        &format!(
                            "{}.{} (instance '{}') clock-to-output",
                            cell_name, pin.name, instance_name
                        ),
                    )?);
                }
                continue;
            }
            if let Some(unsupported_arc) = pin
                .timing_arcs
                .iter()
                .find(|arc| !StaTimingType::from_raw(arc.timing_type.as_str()).is_combinational())
            {
                return Err(anyhow!(
                    "basic STA only supports combinational output pins; instance '{}' output pin '{}.{}' has unsupported timing type '{}'",
                    instance_name,
                    cell_name,
                    pin.name,
                    unsupported_arc.timing_type
                ));
            }
            let combinational_arcs: Vec<&TimingArc> = pin
                .timing_arcs
                .iter()
                .filter(|arc| StaTimingType::from_raw(arc.timing_type.as_str()).is_combinational())
                .collect();
            let output_bit_idx = match output_sources.and_then(|sources| sources.first()) {
                Some(PinBitSource::Bit(output_bit_idx)) => Some(*output_bit_idx),
                Some(PinBitSource::Literal(_) | PinBitSource::Unknown) => {
                    unreachable!("connected output pins have net bit sources")
                }
                None => None,
            };
            let output_net_name = output_bit_idx
                .map(|output_bit_idx| normalized.render_bit(output_bit_idx, nets, interner))
                .unwrap_or_else(|| format!("<unconnected {}.{}>", instance_name, pin.name));
            if combinational_arcs.is_empty() {
                if let Some(constant_value) = constant_output_function_value(cell_name, pin)? {
                    sta_trace(|| {
                        format!(
                            "inst={} cell={} out_pin={} out_net={} constant_source={}",
                            instance_name, cell_name, pin.name, output_net_name, constant_value,
                        )
                    });
                    store_output_timing_set(
                        &mut local_output_timing_sets,
                        &mut bit_timing_sets,
                        pin.name.as_str(),
                        output_bit_idx,
                        literal_source_timing_set.clone(),
                    );
                    continue;
                }
                return Err(anyhow!(
                    "basic STA only supports combinational or constant-source output pins; instance '{}' output pin '{}.{}' has no combinational timing arcs and no constant function",
                    instance_name,
                    cell_name,
                    pin.name
                ));
            }
            validate_timing_tables_once(
                lib.library,
                cell_name,
                pin.name.as_str(),
                combinational_arcs.as_slice(),
                &mut validated_timing_tables,
            )?;
            let output_load = output_bit_idx
                .map(|output_bit_idx| bit_load_capacitance[output_bit_idx])
                .unwrap_or_default();
            let mut accumulated: Option<SignalTimingSet> = None;

            for arc in &combinational_arcs {
                let arc_context = format!(
                    "cell '{}' output pin '{}' timing arc related_pin '{}'",
                    cell_name, pin.name, arc.related_pin
                );
                if !arc_when_may_apply(arc, known_pin_values, arc_context.as_str())? {
                    sta_trace(|| {
                        format!(
                            "inst={} cell={} out_pin={} when={} skipped=true",
                            instance_name, cell_name, pin.name, arc.when,
                        )
                    });
                    continue;
                }
                for related_pin_name in split_related_pin_names(arc.related_pin.as_str()) {
                    let related_pin = lib.pin(cell_idx, related_pin_name).ok_or_else(|| {
                            anyhow!(
                                "cell '{}' output pin '{}' has timing arc with unknown related pin '{}'",
                                cell_name,
                                pin.name,
                                related_pin_name
                            )
                        })?;
                    let mut related_timing_sets: Vec<(&SignalTimingSet, String)> = Vec::new();
                    if related_pin.direction == PinDirection::Output as i32 {
                        let Some(timing) = local_output_timing_sets.get(related_pin_name) else {
                            if analysis_mode.uses_register_boundaries() {
                                continue;
                            }
                            return Err(anyhow!(
                                "missing intra-cell source timing for output pin '{}.{}' feeding '{}.{}' in instance '{}'",
                                cell_name,
                                related_pin_name,
                                cell_name,
                                pin.name,
                                instance_name
                            ));
                        };
                        let related_net_name = inst_pin_map
                            .get(related_pin_name)
                            .and_then(|sources| sources.first())
                            .and_then(|source| match source {
                                PinBitSource::Bit(related_bit_idx) => {
                                    Some(normalized.render_bit(*related_bit_idx, nets, interner))
                                }
                                PinBitSource::Literal(_) | PinBitSource::Unknown => None,
                            })
                            .unwrap_or_else(|| {
                                format!("<unconnected {}.{}>", instance_name, related_pin_name)
                            });
                        related_timing_sets.push((timing, related_net_name));
                    } else {
                        let related_sources =
                                inst_pin_map.get(related_pin_name).ok_or_else(|| {
                                    anyhow!(
                                        "instance '{}' output pin '{}.{}' requires timing-related input pin '{}' to be connected",
                                        instance_name,
                                        cell_name,
                                        pin.name,
                                        related_pin_name
                                    )
                                })?;
                        if related_sources.is_empty() {
                            return Err(anyhow!(
                                "instance '{}' output pin '{}.{}' has unconnected timing-related input pin '{}'",
                                instance_name,
                                cell_name,
                                pin.name,
                                related_pin_name
                            ));
                        }
                        for related_source in related_sources {
                            let (input_timing_set, related_net_name) = match related_source {
                                PinBitSource::Bit(related_bit_idx) => {
                                    let Some(timing) = bit_timing_sets[*related_bit_idx].as_ref()
                                    else {
                                        if analysis_mode.uses_register_boundaries() {
                                            continue;
                                        }
                                        return Err(anyhow!(
                                            "missing source timing for net bit '{}' feeding '{}.{}' (related pin '{}')",
                                            normalized.render_bit(*related_bit_idx, nets, interner),
                                            cell_name,
                                            pin.name,
                                            related_pin_name
                                        ));
                                    };
                                    (
                                        timing,
                                        normalized.render_bit(*related_bit_idx, nets, interner),
                                    )
                                }
                                PinBitSource::Literal(_) | PinBitSource::Unknown
                                    if analysis_mode.uses_register_boundaries() =>
                                {
                                    continue;
                                }
                                PinBitSource::Literal(value) => (
                                    &literal_source_timing_set,
                                    if *value { "1'b1" } else { "1'b0" }.to_string(),
                                ),
                                PinBitSource::Unknown => {
                                    (&literal_source_timing_set, "1'bx".to_string())
                                }
                            };
                            related_timing_sets.push((input_timing_set, related_net_name));
                        }
                    }
                    for (input_timing_set, related_net_name) in related_timing_sets {
                        let context = format!(
                            "{}.{} (instance '{}') related_pin '{}'",
                            cell_name, pin.name, instance_name, related_pin_name
                        );
                        let candidate = evaluate_arc_set(
                            lib.library,
                            arc,
                            input_timing_set,
                            output_load,
                            &mut timing_query_diagnostic_counts,
                            &context,
                        )?;
                        sta_trace(|| {
                            format!(
                                "inst={} cell={} out_pin={} out_net={} related_pin={} related_net={} when={} sense={} type={} rise_candidates={} fall_candidates={} rise_pick={} fall_pick={}",
                                instance_name,
                                cell_name,
                                pin.name,
                                output_net_name,
                                related_pin_name,
                                related_net_name,
                                if arc.when.is_empty() {
                                    "<empty>"
                                } else {
                                    arc.when.as_str()
                                },
                                arc.timing_sense,
                                arc.timing_type,
                                format_edge_timing_set(&candidate.rise),
                                format_edge_timing_set(&candidate.fall),
                                format_optional_edge_timing(candidate.rise.max_arrival_edge()),
                                format_optional_edge_timing(candidate.fall.max_arrival_edge()),
                            )
                        });
                        accumulated = Some(match accumulated {
                            Some(prev) => prev.merge(&candidate),
                            None => candidate,
                        });
                    }
                }
            }

            let Some(mut out_timing_set) = accumulated else {
                if analysis_mode.uses_register_boundaries() {
                    continue;
                }
                return Err(anyhow!(
                    "no usable combinational timing arcs for instance '{}' pin '{}.{}'",
                    instance_name,
                    cell_name,
                    pin.name
                ));
            };
            // Use a conservative per-edge envelope: max arrival and max
            // transition may come from different source candidates.
            collapse_signal_timing_set_to_envelope(&mut out_timing_set);
            sta_trace(|| {
                format!(
                    "inst={} cell={} out_pin={} out_net={} envelope_rise={} envelope_fall={} envelope_rise_pick={} envelope_fall_pick={}",
                    instance_name,
                    cell_name,
                    pin.name,
                    output_net_name,
                    format_edge_timing_set(&out_timing_set.rise),
                    format_edge_timing_set(&out_timing_set.fall),
                    format_optional_edge_timing(out_timing_set.rise.max_arrival_edge()),
                    format_optional_edge_timing(out_timing_set.fall.max_arrival_edge()),
                )
            });

            store_output_timing_set(
                &mut local_output_timing_sets,
                &mut bit_timing_sets,
                pin.name.as_str(),
                output_bit_idx,
                out_timing_set,
            );
        }

        for succ in &successors[inst_idx] {
            instance_levels[*succ] = instance_levels[*succ].max(instance_levels[inst_idx] + 1);
            indegree[*succ] = indegree[*succ].saturating_sub(1);
            if indegree[*succ] == 0 {
                queue.push_back(*succ);
            }
        }
    }

    if processed != instance_count {
        return Err(anyhow!(
            "combinational cycle detected or unresolved dependencies: processed {} of {} instances",
            processed,
            instance_count
        ));
    }

    let mut worst_output: Option<EdgeTimingCandidate> = None;
    let mut cell_levels = 0usize;
    for bit_idx in &module_output_bits {
        let timing_source = resolved_timing_sources[*bit_idx];
        let timing_set = if analysis_mode.uses_register_boundaries()
            && matches!(
                timing_source,
                ResolvedTimingSource::Literal(_) | ResolvedTimingSource::Unknown
            ) {
            None
        } else {
            timing_set_for_resolved_source(
                timing_source,
                bit_timing_sets.as_slice(),
                &literal_source_timing_set,
            )
        };
        let Some(timing_set) = timing_set else {
            if analysis_mode.uses_register_boundaries() {
                continue;
            }
            return Err(anyhow!(
                "missing timing result for module output net bit '{}'",
                normalized.render_bit(*bit_idx, nets, interner)
            ));
        };
        let output_timing = timing_set.worst_arrival_candidate().ok_or_else(|| {
            anyhow!(
                "missing edge timing candidates for module output net bit '{}'",
                normalized.render_bit(*bit_idx, nets, interner)
            )
        })?;
        worst_output = Some(match worst_output {
            Some(current) => choose_worse_edge_timing_candidate_by_arrival(current, output_timing),
            None => output_timing,
        });
        if let ResolvedTimingSource::Bit(source_bit_idx) = timing_source {
            if let Some(driver) = bit_drivers[source_bit_idx].first() {
                cell_levels = cell_levels.max(instance_levels[driver.inst_idx]);
            }
        }
    }

    let mut register_input_timings: Vec<Option<RegisterCaptureTimingCandidate>> =
        vec![None; instance_count];
    let mut worst_register_input: Option<RegisterCaptureTimingCandidate> = None;
    if analysis_mode.uses_register_boundaries() {
        for inst_idx in 0..instance_count {
            if !instance_is_sequential[inst_idx] {
                continue;
            }
            let cell_idx = instance_cell_indices[inst_idx];
            let pin_sources = &instance_pin_sources[inst_idx];
            for pin in lib.library.cells[cell_idx]
                .pins
                .iter()
                .filter(|pin| pin.direction == PinDirection::Input as i32 && !pin.is_clocking_pin)
            {
                let setup_arcs: Vec<&TimingArc> = pin
                    .timing_arcs
                    .iter()
                    .filter(|arc| StaTimingType::from_raw(arc.timing_type.as_str()).is_setup())
                    .collect();
                if setup_arcs.is_empty() {
                    continue;
                }
                validate_constraint_tables_once(
                    lib.library,
                    &instance_cell_names[inst_idx],
                    pin.name.as_str(),
                    setup_arcs.as_slice(),
                    &mut validated_timing_tables,
                )?;
                let Some(sources) = pin_sources.get(pin.name.as_str()) else {
                    continue;
                };
                for source in sources {
                    let PinBitSource::Bit(bit_idx) = source else {
                        continue;
                    };
                    let Some(timing_set) = bit_timing_sets[*bit_idx].as_ref() else {
                        continue;
                    };
                    let Some(capture_timing) = evaluate_register_setup_capture_arrival(
                        lib.library,
                        setup_arcs.as_slice(),
                        &instance_known_pin_values[inst_idx],
                        timing_set,
                        &mut timing_query_diagnostic_counts,
                        &format!(
                            "{}.{} (instance '{}') setup",
                            instance_cell_names[inst_idx],
                            pin.name,
                            resolve_symbol(
                                interner,
                                normalized.instances[inst_idx].instance_name,
                                "instance name"
                            )
                            .unwrap_or_else(|_| "<unknown>".to_string())
                        ),
                    )?
                    else {
                        continue;
                    };
                    register_input_timings[inst_idx] =
                        Some(match register_input_timings[inst_idx] {
                            Some(current) => {
                                choose_worse_register_capture_timing(current, capture_timing)
                            }
                            None => capture_timing,
                        });
                    worst_register_input = Some(match worst_register_input {
                        Some(current) => {
                            choose_worse_register_capture_timing(current, capture_timing)
                        }
                        None => capture_timing,
                    });
                }
            }
        }
    }

    let net_timing = aggregate_bit_timing_by_net(
        bit_timing_sets.as_slice(),
        resolved_timing_sources.as_slice(),
        &literal_source_timing_set,
        &normalized,
        nets.len(),
    );
    let register_input_arrivals = register_input_timings
        .iter()
        .map(|timing| timing.map(|timing| timing.arrival))
        .collect();
    let register_input_breakdowns = register_input_timings
        .iter()
        .map(|timing| timing.and_then(|timing| timing.register_path_breakdown))
        .collect();

    Ok(StaReport {
        net_timing,
        worst_output_arrival: worst_output
            .map(|timing| timing.timing.arrival)
            .unwrap_or(0.0),
        worst_output_breakdown: worst_output.and_then(|timing| timing.register_path_breakdown),
        worst_register_input_arrival: worst_register_input
            .map(|timing| timing.arrival)
            .unwrap_or(0.0),
        worst_register_input_breakdown: worst_register_input
            .and_then(|timing| timing.register_path_breakdown),
        register_input_arrivals,
        register_input_breakdowns,
        timing_query_diagnostic_counts,
        cell_levels,
    })
}
fn effective_input_capacitance_by_edge(pin: &Pin, context: &str) -> Result<EdgeLoadCapacitance> {
    // max_capacitance is a design-rule limit, not nominal pin capacitance.
    // Prefer edge-specific capacitance when provided; fall back to nominal.
    let capacitance = EdgeLoadCapacitance {
        rise: pin
            .rise_capacitance
            .or(pin.capacitance)
            .or(pin.fall_capacitance)
            .unwrap_or(0.0),
        fall: pin
            .fall_capacitance
            .or(pin.capacitance)
            .or(pin.rise_capacitance)
            .unwrap_or(0.0),
    };
    validate_non_negative_finite(capacitance.rise, "rise capacitance", context)?;
    validate_non_negative_finite(capacitance.fall, "fall capacitance", context)?;
    Ok(capacitance)
}

#[cfg(test)]
fn effective_input_capacitance(pin: &Pin) -> f64 {
    // `max_capacitance` is a design-rule limit, not nominal pin capacitance.
    // Net load should be built from sink-pin capacitance, not that limit.
    if let Some(cap) = pin.capacitance {
        return cap;
    }
    [pin.rise_capacitance, pin.fall_capacitance]
        .into_iter()
        .flatten()
        .fold(0.0, f64::max)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaTimingType {
    Combinational,
    CombinationalRise,
    CombinationalFall,
    RisingEdge,
    FallingEdge,
    SetupRising,
    SetupFalling,
    Other,
}

impl StaTimingType {
    fn from_raw(raw: &str) -> Self {
        match raw {
            "" | "combinational" => Self::Combinational,
            "combinational_rise" => Self::CombinationalRise,
            "combinational_fall" => Self::CombinationalFall,
            "rising_edge" => Self::RisingEdge,
            "falling_edge" => Self::FallingEdge,
            "setup_rising" => Self::SetupRising,
            "setup_falling" => Self::SetupFalling,
            _ => Self::Other,
        }
    }

    fn is_combinational(self) -> bool {
        matches!(
            self,
            Self::Combinational | Self::CombinationalRise | Self::CombinationalFall
        )
    }

    fn is_clock_to_output(self) -> bool {
        matches!(self, Self::RisingEdge | Self::FallingEdge)
    }

    fn is_setup(self) -> bool {
        matches!(self, Self::SetupRising | Self::SetupFalling)
    }

    fn produces_rise(self) -> bool {
        !matches!(self, Self::CombinationalFall)
    }

    fn produces_fall(self) -> bool {
        !matches!(self, Self::CombinationalRise)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaTimingSense {
    Unspecified,
    PositiveUnate,
    NegativeUnate,
    NonUnate,
    Other,
}

impl StaTimingSense {
    fn from_raw(raw: &str) -> Self {
        match raw {
            "" => Self::Unspecified,
            "positive_unate" => Self::PositiveUnate,
            "negative_unate" => Self::NegativeUnate,
            "non_unate" => Self::NonUnate,
            _ => Self::Other,
        }
    }

    fn may_use_either_input_edge(self) -> bool {
        matches!(self, Self::Unspecified | Self::NonUnate)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaTimingTableKind {
    CellRise,
    CellFall,
    RiseTransition,
    FallTransition,
    RiseConstraint,
    FallConstraint,
}

impl StaTimingTableKind {
    fn as_raw(self) -> &'static str {
        match self {
            Self::CellRise => "cell_rise",
            Self::CellFall => "cell_fall",
            Self::RiseTransition => "rise_transition",
            Self::FallTransition => "fall_transition",
            Self::RiseConstraint => "rise_constraint",
            Self::FallConstraint => "fall_constraint",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LibertyTableKind {
    CellRise,
    CellFall,
    RiseTransition,
    FallTransition,
    RiseConstraint,
    FallConstraint,
    RisePower,
    FallPower,
    Other,
}

impl LibertyTableKind {
    fn from_raw(raw: &str) -> Self {
        match raw {
            "cell_rise" => Self::CellRise,
            "cell_fall" => Self::CellFall,
            "rise_transition" => Self::RiseTransition,
            "fall_transition" => Self::FallTransition,
            "rise_constraint" => Self::RiseConstraint,
            "fall_constraint" => Self::FallConstraint,
            "rise_power" => Self::RisePower,
            "fall_power" => Self::FallPower,
            _ => Self::Other,
        }
    }

    fn template_kind(self) -> Option<LuTableTemplateKind> {
        match self {
            Self::CellRise
            | Self::CellFall
            | Self::RiseTransition
            | Self::FallTransition
            | Self::RiseConstraint
            | Self::FallConstraint => Some(LuTableTemplateKind::Timing),
            Self::RisePower | Self::FallPower => Some(LuTableTemplateKind::Power),
            Self::Other => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LuTableTemplateKind {
    Timing,
    Power,
    Other,
}

impl LuTableTemplateKind {
    fn from_raw(raw: &str) -> Self {
        match raw {
            "lu_table_template" => Self::Timing,
            "power_lut_template" => Self::Power,
            _ => Self::Other,
        }
    }

    fn as_raw(self) -> &'static str {
        match self {
            Self::Timing => "lu_table_template",
            Self::Power => "power_lut_template",
            Self::Other => "<unsupported>",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AxisVariable {
    Unspecified,
    InputTransition,
    OutputLoad,
    ConstrainedPinTransition,
    RelatedPinTransition,
    Other,
}

impl AxisVariable {
    fn from_raw(raw: &str) -> Self {
        match raw.trim() {
            "" => Self::Unspecified,
            "input_net_transition" | "input_transition_time" => Self::InputTransition,
            "total_output_net_capacitance" => Self::OutputLoad,
            "constrained_pin_transition" => Self::ConstrainedPinTransition,
            "related_pin_transition" => Self::RelatedPinTransition,
            _ => Self::Other,
        }
    }
}

fn split_related_pin_names(related_pin: &str) -> impl Iterator<Item = &str> {
    related_pin.split_whitespace()
}

/// Returns the Boolean value of an output function when it is constant without
/// any input bindings.
fn constant_output_function_value(cell_name: &str, pin: &Pin) -> Result<Option<bool>> {
    if pin.function.trim().is_empty() {
        return Ok(None);
    }
    let function = parse_formula(pin.function.as_str()).map_err(|e| {
        anyhow!(
            "cell '{}' output pin '{}' has invalid function='{}': {}",
            cell_name,
            pin.name,
            pin.function,
            e
        )
    })?;
    Ok(function.evaluate_partial(&HashMap::new()))
}

/// Stores one instance-output timing set for local and net-level propagation.
fn store_output_timing_set(
    local_output_timing_sets: &mut HashMap<String, SignalTimingSet>,
    bit_timing_sets: &mut [Option<SignalTimingSet>],
    pin_name: &str,
    output_bit_idx: Option<usize>,
    out_timing_set: SignalTimingSet,
) {
    local_output_timing_sets.insert(pin_name.to_string(), out_timing_set.clone());
    if let Some(output_bit_idx) = output_bit_idx {
        bit_timing_sets[output_bit_idx] = Some(match bit_timing_sets[output_bit_idx].as_ref() {
            Some(prev) => prev.clone().merge(&out_timing_set),
            None => out_timing_set,
        });
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum OutputPinVisitState {
    #[default]
    Unvisited,
    Visiting,
    Done,
}

/// Orders connected outputs and their output-pin dependencies for local STA.
fn combinational_output_pin_evaluation_order(
    lib: &StaLibraryIndex<'_>,
    cell_idx: usize,
    pin_sources: &HashMap<String, Vec<PinBitSource>>,
) -> Result<Vec<usize>> {
    let cell = &lib.library.cells[cell_idx];
    let mut visit_states = vec![OutputPinVisitState::Unvisited; cell.pins.len()];
    let mut order = Vec::new();
    for (pin_idx, pin) in cell.pins.iter().enumerate() {
        if pin.direction == PinDirection::Output as i32
            && pin_sources.contains_key(pin.name.as_str())
        {
            visit_combinational_output_pin(lib, cell_idx, pin_idx, &mut visit_states, &mut order)?;
        }
    }
    Ok(order)
}

/// Adds one output pin after recursively ordering its output-pin dependencies.
fn visit_combinational_output_pin(
    lib: &StaLibraryIndex<'_>,
    cell_idx: usize,
    pin_idx: usize,
    visit_states: &mut [OutputPinVisitState],
    order: &mut Vec<usize>,
) -> Result<()> {
    let cell = &lib.library.cells[cell_idx];
    let pin = &cell.pins[pin_idx];
    match visit_states[pin_idx] {
        OutputPinVisitState::Visiting => {
            return Err(anyhow!(
                "cell '{}' has a combinational output-pin dependency cycle involving '{}'; basic STA only supports acyclic output-related timing arcs",
                cell.name,
                pin.name
            ));
        }
        OutputPinVisitState::Done => return Ok(()),
        OutputPinVisitState::Unvisited => {}
    }
    visit_states[pin_idx] = OutputPinVisitState::Visiting;
    for arc in pin
        .timing_arcs
        .iter()
        .filter(|arc| StaTimingType::from_raw(arc.timing_type.as_str()).is_combinational())
    {
        for related_pin_name in split_related_pin_names(arc.related_pin.as_str()) {
            let related_pin_idx = lib.pin_index(cell_idx, related_pin_name).ok_or_else(|| {
                anyhow!(
                    "cell '{}' output pin '{}' has timing arc with unknown related pin '{}'",
                    cell.name,
                    pin.name,
                    related_pin_name
                )
            })?;
            let related_pin = &cell.pins[related_pin_idx];
            if related_pin.direction == PinDirection::Output as i32 {
                visit_combinational_output_pin(
                    lib,
                    cell_idx,
                    related_pin_idx,
                    visit_states,
                    order,
                )?;
            }
        }
    }
    visit_states[pin_idx] = OutputPinVisitState::Done;
    order.push(pin_idx);
    Ok(())
}

/// Validates that one cell output pin is fully usable by the limited-scope STA
/// engine for the requested related input pins.
pub fn validate_output_pin_for_basic_sta(
    library: &crate::liberty_proto::Library,
    cell_name: &str,
    pin: &Pin,
    required_related_pins: &[String],
) -> Result<()> {
    if pin.direction != PinDirection::Output as i32 {
        return Err(anyhow!(
            "cell '{}' pin '{}' is not an output pin",
            cell_name,
            pin.name
        ));
    }
    if pin.timing_arcs.is_empty() && constant_output_function_value(cell_name, pin)?.is_none() {
        return Err(anyhow!(
            "cell '{}' output pin '{}' has no timing arcs and no constant function",
            cell_name,
            pin.name
        ));
    }
    if let Some(unsupported_arc) = pin
        .timing_arcs
        .iter()
        .find(|arc| !StaTimingType::from_raw(arc.timing_type.as_str()).is_combinational())
    {
        return Err(anyhow!(
            "cell '{}' output pin '{}' has unsupported timing type '{}'",
            cell_name,
            pin.name,
            unsupported_arc.timing_type
        ));
    }
    let combinational_arcs: Vec<&TimingArc> = pin.timing_arcs.iter().collect();
    let mut validated_tables = HashSet::new();
    validate_timing_tables_once(
        library,
        cell_name,
        pin.name.as_str(),
        combinational_arcs.as_slice(),
        &mut validated_tables,
    )?;

    for arc in &combinational_arcs {
        let timing_type = StaTimingType::from_raw(arc.timing_type.as_str());
        let context = format!(
            "cell '{}' output pin '{}' related_pin '{}'",
            cell_name, pin.name, arc.related_pin
        );
        if timing_type.produces_rise() {
            find_unique_table(arc, StaTimingTableKind::CellRise, context.as_str())?;
            find_unique_table(arc, StaTimingTableKind::RiseTransition, context.as_str())?;
        }
        if timing_type.produces_fall() {
            find_unique_table(arc, StaTimingTableKind::CellFall, context.as_str())?;
            find_unique_table(arc, StaTimingTableKind::FallTransition, context.as_str())?;
        }
    }

    for required_pin in required_related_pins {
        let mut has_rise = false;
        let mut has_fall = false;
        for arc in &combinational_arcs {
            if !split_related_pin_names(arc.related_pin.as_str())
                .any(|related_pin| related_pin == required_pin)
            {
                continue;
            }
            let timing_type = StaTimingType::from_raw(arc.timing_type.as_str());
            has_rise |= timing_type.produces_rise();
            has_fall |= timing_type.produces_fall();
        }
        if !has_rise || !has_fall {
            return Err(anyhow!(
                "cell '{}' output pin '{}' lacks complete rise/fall combinational timing coverage for functional input '{}'",
                cell_name,
                pin.name,
                required_pin
            ));
        }
    }
    Ok(())
}

/// Classifies continuous assigns into live-STA-supported bit sources and
/// unsupported sources that may still be harmless if they are unused.
fn analyze_assign_sources(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<AssignSourceAnalysis> {
    let mut sources = vec![None; normalized.bit_count()];
    let mut unsupported_sources = Vec::new();
    for assign in &normalized.assigns {
        if let Some(rhs_sources) = supported_assign_bit_sources(assign.rhs_bits.as_slice()) {
            for (lhs_bit_idx, rhs_source) in assign.lhs_bits.iter().copied().zip(rhs_sources) {
                if sources[lhs_bit_idx].is_some() {
                    let bit_name = normalized.render_bit(lhs_bit_idx, nets, interner);
                    return Err(anyhow!(
                        "net bit '{}' has multiple continuous assign drivers; wired multi-driver nets are unsupported in basic STA",
                        bit_name
                    ));
                }
                sources[lhs_bit_idx] = Some(rhs_source);
            }
        } else {
            unsupported_sources.push(UnsupportedAssignSource {
                lhs_bits: assign.lhs_bits.clone(),
                rendered_lhs: assign.rendered_lhs.clone(),
            });
        }
    }
    Ok(AssignSourceAnalysis {
        bit_sources: sources,
        unsupported_sources,
    })
}

/// Returns supported zero-delay assign bit sources, or `None` for unsupported
/// expression shapes.
fn supported_assign_bit_sources(rhs_bits: &[BitExpr]) -> Option<Vec<BitAssignSource>> {
    rhs_bits
        .iter()
        .map(|bit| match bit {
            BitExpr::Source(BitSource::Bit(bit_idx)) => Some(BitAssignSource::Alias(*bit_idx)),
            BitExpr::Source(BitSource::Literal(value)) => Some(BitAssignSource::Literal(*value)),
            BitExpr::Source(BitSource::Unknown) => Some(BitAssignSource::Unknown),
            BitExpr::Not(_) | BitExpr::And(_, _) | BitExpr::Or(_, _) | BitExpr::Xor(_, _) => None,
        })
        .collect()
}

/// Rejects unsupported continuous assigns only when they drive timing-live
/// bits.
fn reject_live_unsupported_assigns(
    unsupported_sources: &[UnsupportedAssignSource],
    assign_sources: &[Option<BitAssignSource>],
    bit_loads: &[Vec<NetEndpoint>],
    has_module_output: &[bool],
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<()> {
    if unsupported_sources.is_empty() {
        return Ok(());
    }
    let mut live = vec![false; normalized.bit_count()];
    for bit_idx in 0..normalized.bit_count() {
        live[bit_idx] = !bit_loads[bit_idx].is_empty() || has_module_output[bit_idx];
    }
    let mut changed = true;
    while changed {
        changed = false;
        for (bit_idx, source) in assign_sources.iter().enumerate() {
            if !live[bit_idx] {
                continue;
            }
            let Some(BitAssignSource::Alias(source_bit_idx)) = source else {
                continue;
            };
            if !live[*source_bit_idx] {
                live[*source_bit_idx] = true;
                changed = true;
            }
        }
    }
    for unsupported in unsupported_sources {
        if unsupported.lhs_bits.iter().any(|bit_idx| live[*bit_idx]) {
            let live_names: Vec<String> = unsupported
                .lhs_bits
                .iter()
                .filter(|bit_idx| live[**bit_idx])
                .map(|bit_idx| normalized.render_bit(*bit_idx, nets, interner))
                .collect();
            return Err(anyhow!(
                "basic STA only supports literal, alias, slice, or concat continuous assigns for live bits; unsupported assign to '{}' drives live bit(s): {}",
                unsupported.rendered_lhs,
                live_names.join(", ")
            ));
        }
    }
    Ok(())
}

fn canonicalize_pin_bit_source(
    source: PinBitSource,
    resolved_timing_sources: &[ResolvedTimingSource],
) -> PinBitSource {
    match source {
        PinBitSource::Bit(bit_idx) => match resolved_timing_sources[bit_idx] {
            ResolvedTimingSource::Literal(value) => PinBitSource::Literal(value),
            ResolvedTimingSource::Unknown => PinBitSource::Unknown,
            ResolvedTimingSource::Bit(source_bit_idx) => PinBitSource::Bit(source_bit_idx),
        },
        PinBitSource::Literal(_) | PinBitSource::Unknown => source,
    }
}

/// Resolves zero-delay continuous-assign aliases once for all timing lookups.
fn resolve_timing_sources(
    assign_sources: &[Option<BitAssignSource>],
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<Vec<ResolvedTimingSource>> {
    let mut visit_states = vec![TimingSourceVisitState::Unvisited; assign_sources.len()];
    for start in 0..assign_sources.len() {
        if matches!(visit_states[start], TimingSourceVisitState::Resolved(_)) {
            continue;
        }
        let mut current = start;
        let mut path = Vec::new();
        let resolved = loop {
            match visit_states[current] {
                TimingSourceVisitState::Resolved(source) => break source,
                TimingSourceVisitState::Visiting => {
                    return Err(anyhow!(
                        "continuous assign aliases contain a cycle involving net bit '{}'",
                        normalized.render_bit(current, nets, interner)
                    ));
                }
                TimingSourceVisitState::Unvisited => {
                    visit_states[current] = TimingSourceVisitState::Visiting;
                    path.push(current);
                    match assign_sources[current] {
                        Some(BitAssignSource::Literal(value)) => {
                            break ResolvedTimingSource::Literal(value);
                        }
                        Some(BitAssignSource::Unknown) => break ResolvedTimingSource::Unknown,
                        Some(BitAssignSource::Alias(next)) => current = next,
                        None => break ResolvedTimingSource::Bit(current),
                    }
                }
            }
        };
        for bit_idx in path {
            visit_states[bit_idx] = TimingSourceVisitState::Resolved(resolved);
        }
    }
    Ok(visit_states
        .into_iter()
        .map(|state| match state {
            TimingSourceVisitState::Resolved(source) => source,
            TimingSourceVisitState::Unvisited | TimingSourceVisitState::Visiting => {
                unreachable!("every timing source path should be resolved")
            }
        })
        .collect())
}

fn timing_set_for_resolved_source<'a>(
    source: ResolvedTimingSource,
    bit_timing_sets: &'a [Option<SignalTimingSet>],
    literal_source_timing_set: &'a SignalTimingSet,
) -> Option<&'a SignalTimingSet> {
    match source {
        ResolvedTimingSource::Literal(_) | ResolvedTimingSource::Unknown => {
            Some(literal_source_timing_set)
        }
        ResolvedTimingSource::Bit(bit_idx) => bit_timing_sets[bit_idx].as_ref(),
    }
}

fn aggregate_bit_timing_by_net(
    bit_timing_sets: &[Option<SignalTimingSet>],
    resolved_timing_sources: &[ResolvedTimingSource],
    literal_source_timing_set: &SignalTimingSet,
    normalized: &NormalizedNetlistModule<'_>,
    nets_len: usize,
) -> Vec<Option<SignalTiming>> {
    let mut aggregate_sets: Vec<Option<SignalTimingSet>> = vec![None; nets_len];
    for (bit_idx, source) in resolved_timing_sources.iter().copied().enumerate() {
        let Some(timing_set) =
            timing_set_for_resolved_source(source, bit_timing_sets, literal_source_timing_set)
        else {
            continue;
        };
        let net_idx = normalized.bit(bit_idx).net.0;
        aggregate_sets[net_idx] = Some(match aggregate_sets[net_idx].take() {
            Some(prev) => prev.merge(timing_set),
            None => timing_set.clone(),
        });
    }
    aggregate_sets
        .into_iter()
        .map(|timing_set| timing_set.and_then(|set| set.as_report_signal_timing()))
        .collect()
}

fn arc_when_may_apply(
    arc: &TimingArc,
    known_pin_values: &HashMap<String, bool>,
    context: &str,
) -> Result<bool> {
    if arc.when.is_empty() {
        return Ok(true);
    }
    let when = parse_formula(arc.when.as_str())
        .map_err(|e| anyhow!("{context}: could not parse when='{}': {}", arc.when, e))?;
    Ok(when.evaluate_partial(known_pin_values) != Some(false))
}

#[cfg(test)]
fn evaluate_arc(
    library: &crate::liberty_proto::Library,
    arc: &TimingArc,
    input_timing: SignalTiming,
    output_load: f64,
    context: &str,
) -> Result<SignalTiming> {
    let mut timing_query_diagnostic_counts = TimingQueryDiagnosticCounts::default();
    let set = evaluate_arc_set(
        library,
        arc,
        &SignalTimingSet::from_single(input_timing),
        EdgeLoadCapacitance {
            rise: output_load,
            fall: output_load,
        },
        &mut timing_query_diagnostic_counts,
        context,
    )?;
    set.as_report_signal_timing().ok_or_else(|| {
        anyhow!("{context}: no output timing candidates were produced after arc evaluation")
    })
}

fn evaluate_arc_set(
    library: &crate::liberty_proto::Library,
    arc: &TimingArc,
    input_timing: &SignalTimingSet,
    output_load: EdgeLoadCapacitance,
    timing_query_diagnostic_counts: &mut TimingQueryDiagnosticCounts,
    context: &str,
) -> Result<SignalTimingSet> {
    let timing_type = StaTimingType::from_raw(arc.timing_type.as_str());
    let timing_sense = StaTimingSense::from_raw(arc.timing_sense.as_str());
    let all_inputs = if timing_sense.may_use_either_input_edge() {
        let mut combined = input_timing.rise.clone();
        combined.extend_from(&input_timing.fall);
        Some(combined)
    } else {
        None
    };
    let source_edges = |output_edge_is_rise: bool| -> Result<&EdgeTimingSet> {
        match (timing_sense, output_edge_is_rise) {
            (StaTimingSense::PositiveUnate, true) => Ok(&input_timing.rise),
            (StaTimingSense::PositiveUnate, false) => Ok(&input_timing.fall),
            (StaTimingSense::NegativeUnate, true) => Ok(&input_timing.fall),
            (StaTimingSense::NegativeUnate, false) => Ok(&input_timing.rise),
            (StaTimingSense::Unspecified | StaTimingSense::NonUnate, _) => Ok(all_inputs
                .as_ref()
                .expect("non-unate input set should be built")),
            _ => Err(anyhow!(
                "{context}: unsupported timing_sense '{}'",
                arc.timing_sense
            )),
        }
    };

    let mut output = SignalTimingSet::default();
    if timing_type.produces_rise() {
        let cell_rise = find_unique_table(arc, StaTimingTableKind::CellRise, context)?;
        let rise_transition = find_unique_table(arc, StaTimingTableKind::RiseTransition, context)?;
        output.rise = evaluate_output_edge_set(
            library,
            cell_rise,
            rise_transition,
            source_edges(true)?,
            output_load.rise,
            timing_query_diagnostic_counts,
            context,
            StaTimingTableKind::CellRise,
            StaTimingTableKind::RiseTransition,
        )?;
    }
    if timing_type.produces_fall() {
        let cell_fall = find_unique_table(arc, StaTimingTableKind::CellFall, context)?;
        let fall_transition = find_unique_table(arc, StaTimingTableKind::FallTransition, context)?;
        output.fall = evaluate_output_edge_set(
            library,
            cell_fall,
            fall_transition,
            source_edges(false)?,
            output_load.fall,
            timing_query_diagnostic_counts,
            context,
            StaTimingTableKind::CellFall,
            StaTimingTableKind::FallTransition,
        )?;
    }
    Ok(output)
}

fn evaluate_output_edge_set(
    library: &crate::liberty_proto::Library,
    delay_table: &TimingTable,
    slew_table: &TimingTable,
    source_edges: &EdgeTimingSet,
    output_load: f64,
    timing_query_diagnostic_counts: &mut TimingQueryDiagnosticCounts,
    context: &str,
    delay_kind: StaTimingTableKind,
    slew_kind: StaTimingTableKind,
) -> Result<EdgeTimingSet> {
    let mut outputs = EdgeTimingSet::default();

    for source_edge in source_edges.iter() {
        let delay = evaluate_table_with_diagnostics(
            library,
            delay_table,
            source_edge.timing.transition,
            output_load,
            timing_query_diagnostic_counts,
            &format!("{context} {}", delay_kind.as_raw()),
        )?;
        // Characterized transition values may contain small negative artifacts;
        // physical transition is bounded below by zero.
        let transition = evaluate_table_with_diagnostics(
            library,
            slew_table,
            source_edge.timing.transition,
            output_load,
            timing_query_diagnostic_counts,
            &format!("{context} {}", slew_kind.as_raw()),
        )?
        .max(0.0);
        let arrival = source_edge.timing.arrival + delay;
        if !arrival.is_finite() {
            return Err(anyhow!(
                "{context}: propagated arrival must be finite; got {} + {} = {}",
                source_edge.timing.arrival,
                delay,
                arrival
            ));
        }
        outputs.insert(EdgeTimingCandidate {
            timing: EdgeTiming {
                arrival,
                transition,
            },
            register_path_breakdown: source_edge.register_path_breakdown.map(|mut breakdown| {
                breakdown.combinational_delay += delay;
                breakdown
            }),
        });
    }

    if outputs.values.is_empty() {
        return Err(anyhow!(
            "{context}: no source edge candidates for '{}'/'{}' evaluation",
            delay_kind.as_raw(),
            slew_kind.as_raw()
        ));
    }

    Ok(outputs)
}

/// Evaluates a register output launched by an ideal clock edge.
fn evaluate_register_launch_output_set(
    library: &crate::liberty_proto::Library,
    cell_name: &str,
    pin: &Pin,
    known_pin_values: &HashMap<String, bool>,
    output_load: EdgeLoadCapacitance,
    validated_tables: &mut HashSet<*const TimingTable>,
    timing_query_diagnostic_counts: &mut TimingQueryDiagnosticCounts,
    context: &str,
) -> Result<SignalTimingSet> {
    let launch_arcs: Vec<&TimingArc> = pin
        .timing_arcs
        .iter()
        .filter(|arc| StaTimingType::from_raw(arc.timing_type.as_str()).is_clock_to_output())
        .collect();
    if launch_arcs.is_empty() {
        return Err(anyhow!(
            "{context}: sequential output pin has no rising_edge or falling_edge timing arc"
        ));
    }
    validate_timing_tables_once(
        library,
        cell_name,
        pin.name.as_str(),
        launch_arcs.as_slice(),
        validated_tables,
    )?;

    let mut output = SignalTimingSet::default();
    for arc in launch_arcs {
        if !arc_when_may_apply(arc, known_pin_values, context)? {
            continue;
        }
        if let Some((delay, transition)) = find_optional_delay_slew_tables(
            arc,
            StaTimingTableKind::CellRise,
            StaTimingTableKind::RiseTransition,
            context,
        )? {
            let query = TimingTableQuery::ideal_clock_to_output(output_load.rise);
            let arrival = evaluate_table_with_query_and_diagnostics(
                library,
                delay,
                query,
                timing_query_diagnostic_counts,
                &format!("{context} {}", StaTimingTableKind::CellRise.as_raw()),
            )?;
            output
                .rise
                .insert(EdgeTimingCandidate::from_register_launch(EdgeTiming {
                    arrival,
                    transition: evaluate_table_with_query_and_diagnostics(
                        library,
                        transition,
                        query,
                        timing_query_diagnostic_counts,
                        &format!("{context} {}", StaTimingTableKind::RiseTransition.as_raw()),
                    )?
                    .max(0.0),
                }));
        }
        if let Some((delay, transition)) = find_optional_delay_slew_tables(
            arc,
            StaTimingTableKind::CellFall,
            StaTimingTableKind::FallTransition,
            context,
        )? {
            let query = TimingTableQuery::ideal_clock_to_output(output_load.fall);
            let arrival = evaluate_table_with_query_and_diagnostics(
                library,
                delay,
                query,
                timing_query_diagnostic_counts,
                &format!("{context} {}", StaTimingTableKind::CellFall.as_raw()),
            )?;
            output
                .fall
                .insert(EdgeTimingCandidate::from_register_launch(EdgeTiming {
                    arrival,
                    transition: evaluate_table_with_query_and_diagnostics(
                        library,
                        transition,
                        query,
                        timing_query_diagnostic_counts,
                        &format!("{context} {}", StaTimingTableKind::FallTransition.as_raw()),
                    )?
                    .max(0.0),
                }));
        }
    }
    if output.rise.values.is_empty() || output.fall.values.is_empty() {
        return Err(anyhow!(
            "{context}: clock-to-output arcs do not provide complete rise/fall delay and transition coverage"
        ));
    }
    collapse_signal_timing_set_to_envelope(&mut output);
    Ok(output)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RegisterCaptureTimingCandidate {
    arrival: f64,
    register_path_breakdown: Option<RegisterPathDelayBreakdown>,
}

/// Returns the worst data arrival plus setup requirement at one capture pin.
fn evaluate_register_setup_capture_arrival(
    library: &crate::liberty_proto::Library,
    setup_arcs: &[&TimingArc],
    known_pin_values: &HashMap<String, bool>,
    data_timing: &SignalTimingSet,
    timing_query_diagnostic_counts: &mut TimingQueryDiagnosticCounts,
    context: &str,
) -> Result<Option<RegisterCaptureTimingCandidate>> {
    let mut worst_timing: Option<RegisterCaptureTimingCandidate> = None;
    for arc in setup_arcs {
        if !arc_when_may_apply(arc, known_pin_values, context)? {
            continue;
        }
        if let Some(table) =
            find_optional_unique_table(arc, StaTimingTableKind::RiseConstraint, context)?
        {
            for data_edge in data_timing.rise.iter() {
                let setup = evaluate_table_with_query_and_diagnostics(
                    library,
                    table,
                    TimingTableQuery::ideal_clock_setup(data_edge.timing.transition),
                    timing_query_diagnostic_counts,
                    &format!("{context} {}", StaTimingTableKind::RiseConstraint.as_raw()),
                )?;
                let capture_arrival = data_edge.timing.arrival + setup;
                if !capture_arrival.is_finite() {
                    return Err(anyhow!(
                        "{context}: setup-adjusted arrival must be finite; got {} + {} = {}",
                        data_edge.timing.arrival,
                        setup,
                        capture_arrival
                    ));
                }
                let candidate = register_capture_timing_candidate(data_edge, setup);
                worst_timing = Some(match worst_timing {
                    Some(current) => choose_worse_register_capture_timing(current, candidate),
                    None => candidate,
                });
            }
        }
        if let Some(table) =
            find_optional_unique_table(arc, StaTimingTableKind::FallConstraint, context)?
        {
            for data_edge in data_timing.fall.iter() {
                let setup = evaluate_table_with_query_and_diagnostics(
                    library,
                    table,
                    TimingTableQuery::ideal_clock_setup(data_edge.timing.transition),
                    timing_query_diagnostic_counts,
                    &format!("{context} {}", StaTimingTableKind::FallConstraint.as_raw()),
                )?;
                let capture_arrival = data_edge.timing.arrival + setup;
                if !capture_arrival.is_finite() {
                    return Err(anyhow!(
                        "{context}: setup-adjusted arrival must be finite; got {} + {} = {}",
                        data_edge.timing.arrival,
                        setup,
                        capture_arrival
                    ));
                }
                let candidate = register_capture_timing_candidate(data_edge, setup);
                worst_timing = Some(match worst_timing {
                    Some(current) => choose_worse_register_capture_timing(current, candidate),
                    None => candidate,
                });
            }
        }
    }
    Ok(worst_timing)
}

fn register_capture_timing_candidate(
    data_edge: EdgeTimingCandidate,
    setup: f64,
) -> RegisterCaptureTimingCandidate {
    RegisterCaptureTimingCandidate {
        arrival: data_edge.timing.arrival + setup,
        register_path_breakdown: data_edge.register_path_breakdown.map(|mut breakdown| {
            breakdown.setup_delay += setup;
            breakdown
        }),
    }
}

fn choose_worse_register_capture_timing(
    lhs: RegisterCaptureTimingCandidate,
    rhs: RegisterCaptureTimingCandidate,
) -> RegisterCaptureTimingCandidate {
    if rhs.arrival > lhs.arrival { rhs } else { lhs }
}

fn collapse_signal_timing_set_to_envelope(signal: &mut SignalTimingSet) {
    collapse_edge_timing_set_to_envelope(&mut signal.rise);
    collapse_edge_timing_set_to_envelope(&mut signal.fall);
}

fn collapse_edge_timing_set_to_envelope(set: &mut EdgeTimingSet) {
    let max_arrival_candidate = set.max_arrival_candidate();
    let max_transition = set
        .values
        .iter()
        .map(|edge| edge.timing.transition)
        .reduce(f64::max);
    if let (Some(mut candidate), Some(transition)) = (max_arrival_candidate, max_transition) {
        candidate.timing.transition = transition;
        set.values.clear();
        set.values.push(candidate);
    }
}

fn choose_worse_edge_timing_candidate_by_arrival(
    lhs: EdgeTimingCandidate,
    rhs: EdgeTimingCandidate,
) -> EdgeTimingCandidate {
    if lhs.timing.arrival > rhs.timing.arrival {
        lhs
    } else if rhs.timing.arrival > lhs.timing.arrival {
        rhs
    } else if rhs.timing.transition > lhs.timing.transition {
        rhs
    } else {
        lhs
    }
}

fn find_optional_unique_table<'a>(
    arc: &'a TimingArc,
    kind: StaTimingTableKind,
    context: &str,
) -> Result<Option<&'a TimingTable>> {
    let mut matches = arc
        .tables
        .iter()
        .filter(|table| table.kind == kind.as_raw());
    let first = matches.next();
    if matches.next().is_some() {
        return Err(anyhow!(
            "{context}: multiple '{}' timing tables are unsupported in basic STA",
            kind.as_raw()
        ));
    }
    Ok(first)
}

fn find_unique_table<'a>(
    arc: &'a TimingArc,
    kind: StaTimingTableKind,
    context: &str,
) -> Result<&'a TimingTable> {
    find_optional_unique_table(arc, kind, context)?
        .ok_or_else(|| anyhow!("{context}: missing '{}' timing table", kind.as_raw()))
}

fn find_optional_delay_slew_tables<'a>(
    arc: &'a TimingArc,
    delay_kind: StaTimingTableKind,
    slew_kind: StaTimingTableKind,
    context: &str,
) -> Result<Option<(&'a TimingTable, &'a TimingTable)>> {
    let delay = find_optional_unique_table(arc, delay_kind, context)?;
    let slew = find_optional_unique_table(arc, slew_kind, context)?;
    match (delay, slew) {
        (Some(delay), Some(slew)) => Ok(Some((delay, slew))),
        (None, None) => Ok(None),
        _ => Err(anyhow!(
            "{context}: '{}' and '{}' timing tables must both be present for an output edge",
            delay_kind.as_raw(),
            slew_kind.as_raw()
        )),
    }
}

fn expected_template_kind_for_timing_table(table: &TimingTable) -> Result<LuTableTemplateKind> {
    LibertyTableKind::from_raw(table.kind.as_str())
        .template_kind()
        .ok_or_else(|| anyhow!("unsupported Liberty table kind '{}'", table.kind))
}

struct TimingTableLayout<'a> {
    axes: [&'a [f64]; 3],
    variables: [&'a str; 3],
}

/// Resolves template-backed table axes and variables without evaluating a
/// query.
fn timing_table_layout<'a>(
    library: &'a crate::liberty_proto::Library,
    table: &'a TimingTable,
    context: &str,
) -> Result<TimingTableLayout<'a>> {
    let template: Option<&LuTableTemplate> = if table.template_id == 0 {
        None
    } else {
        let idx = (table.template_id - 1) as usize;
        let tmpl = library.lu_table_templates.get(idx).ok_or_else(|| {
            anyhow!(
                "{context}: template_id {} out of range ({} templates)",
                table.template_id,
                library.lu_table_templates.len()
            )
        })?;
        let expected_kind = expected_template_kind_for_timing_table(table)
            .map_err(|e| anyhow!("{context}: {e}"))?;
        let actual_kind = LuTableTemplateKind::from_raw(tmpl.kind.as_str());
        if actual_kind != expected_kind {
            return Err(anyhow!(
                "{context}: template_id {} kind mismatch; got '{}' expected '{}'",
                table.template_id,
                tmpl.kind,
                expected_kind.as_raw()
            ));
        }
        Some(tmpl)
    };

    Ok(TimingTableLayout {
        axes: [
            effective_axis(&table.index_1, template.map(|t| t.index_1.as_slice())),
            effective_axis(&table.index_2, template.map(|t| t.index_2.as_slice())),
            effective_axis(&table.index_3, template.map(|t| t.index_3.as_slice())),
        ],
        variables: [
            template.map(|t| t.variable_1.as_str()).unwrap_or(""),
            template.map(|t| t.variable_2.as_str()).unwrap_or(""),
            template.map(|t| t.variable_3.as_str()).unwrap_or(""),
        ],
    })
}

/// Validates table shape, axes, and numeric payload without imposing delay
/// monotonicity requirements.
fn validate_timing_table_structure(
    library: &crate::liberty_proto::Library,
    table: &TimingTable,
    context: &str,
) -> Result<()> {
    let array = TimingTableArrayView::from_timing_table(table)
        .map_err(|e| anyhow!("{context}: invalid timing table payload: {e}"))?;
    for value in &table.values {
        if !value.is_finite() {
            return Err(anyhow!(
                "{context}: timing table contains non-finite value {}",
                value
            ));
        }
    }
    if array.rank() > 3 {
        return Err(anyhow!(
            "{context}: rank-{} timing table is unsupported in basic STA",
            array.rank()
        ));
    }
    let layout = timing_table_layout(library, table, context)?;
    validate_effective_axes(table, layout.axes, context)?;
    for (axis_idx, axis) in layout.axes.iter().take(array.rank()).enumerate() {
        if axis.is_empty() {
            return Err(anyhow!(
                "{context}: missing axis data for rank-{} table on axis {}",
                array.rank(),
                axis_idx + 1
            ));
        }
        if let Some(value) = axis.iter().copied().find(|value| !value.is_finite()) {
            return Err(anyhow!(
                "{context}: axis {} contains non-finite value {}",
                axis_idx + 1,
                value
            ));
        }
        if !is_strictly_increasing(axis) {
            return Err(anyhow!(
                "{context}: axis {} is not strictly increasing",
                axis_idx + 1
            ));
        }
    }
    Ok(())
}

/// Validates query-invariant delay/slew table structure once before STA
/// propagation. Non-monotone values are repaired conservatively during lookup.
fn validate_timing_table_payload(
    library: &crate::liberty_proto::Library,
    table: &TimingTable,
    context: &str,
) -> Result<()> {
    validate_timing_table_structure(library, table, context)
}

/// Validates each queried delay/slew table at most once during an STA run.
fn validate_timing_tables_once(
    library: &crate::liberty_proto::Library,
    cell_name: &str,
    pin_name: &str,
    arcs: &[&TimingArc],
    validated_tables: &mut HashSet<*const TimingTable>,
) -> Result<()> {
    for arc in arcs {
        for table in &arc.tables {
            if !matches!(
                LibertyTableKind::from_raw(table.kind.as_str()),
                LibertyTableKind::CellRise
                    | LibertyTableKind::CellFall
                    | LibertyTableKind::RiseTransition
                    | LibertyTableKind::FallTransition
            ) {
                continue;
            }
            if !validated_tables.insert(table as *const TimingTable) {
                continue;
            }
            validate_timing_table_payload(
                library,
                table,
                &format!(
                    "cell '{}' pin '{}' related_pin '{}' table '{}'",
                    cell_name, pin_name, arc.related_pin, table.kind
                ),
            )?;
        }
    }
    Ok(())
}

/// Validates setup constraint tables without requiring a monotone surface.
///
/// Setup surfaces commonly decrease as related clock transition increases, so
/// the monotonicity rule used for combinational delay/slew propagation does not
/// apply.
fn validate_constraint_tables_once(
    library: &crate::liberty_proto::Library,
    cell_name: &str,
    pin_name: &str,
    arcs: &[&TimingArc],
    validated_tables: &mut HashSet<*const TimingTable>,
) -> Result<()> {
    for arc in arcs {
        for table in &arc.tables {
            if !matches!(
                LibertyTableKind::from_raw(table.kind.as_str()),
                LibertyTableKind::RiseConstraint | LibertyTableKind::FallConstraint
            ) {
                continue;
            }
            if !validated_tables.insert(table as *const TimingTable) {
                continue;
            }
            validate_timing_table_structure(
                library,
                table,
                &format!(
                    "cell '{}' pin '{}' related_pin '{}' table '{}'",
                    cell_name, pin_name, arc.related_pin, table.kind
                ),
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MinimumCharacterizedAxis {
    None,
    InputTransition,
    RelatedPinTransition,
}

#[derive(Clone, Copy, Debug)]
struct TimingTableQuery {
    input_transition: f64,
    output_load: f64,
    constrained_pin_transition: f64,
    related_pin_transition: f64,
    /// For an ideal clock, select the fastest slew characterized by each table.
    minimum_characterized_axis: MinimumCharacterizedAxis,
}

impl TimingTableQuery {
    fn combinational(input_transition: f64, output_load: f64) -> Self {
        Self {
            input_transition,
            output_load,
            constrained_pin_transition: input_transition,
            related_pin_transition: input_transition,
            minimum_characterized_axis: MinimumCharacterizedAxis::None,
        }
    }

    fn ideal_clock_to_output(output_load: f64) -> Self {
        Self {
            input_transition: 0.0,
            output_load,
            constrained_pin_transition: 0.0,
            related_pin_transition: 0.0,
            minimum_characterized_axis: MinimumCharacterizedAxis::InputTransition,
        }
    }

    fn ideal_clock_setup(data_transition: f64) -> Self {
        Self {
            input_transition: data_transition,
            output_load: 0.0,
            constrained_pin_transition: data_transition,
            related_pin_transition: 0.0,
            minimum_characterized_axis: MinimumCharacterizedAxis::RelatedPinTransition,
        }
    }
}

#[cfg(test)]
fn evaluate_table(
    library: &crate::liberty_proto::Library,
    table: &TimingTable,
    input_transition: f64,
    output_load: f64,
    context: &str,
) -> Result<f64> {
    let mut timing_query_diagnostic_counts = TimingQueryDiagnosticCounts::default();
    evaluate_table_with_diagnostics(
        library,
        table,
        input_transition,
        output_load,
        &mut timing_query_diagnostic_counts,
        context,
    )
}

fn evaluate_table_with_diagnostics(
    library: &crate::liberty_proto::Library,
    table: &TimingTable,
    input_transition: f64,
    output_load: f64,
    timing_query_diagnostic_counts: &mut TimingQueryDiagnosticCounts,
    context: &str,
) -> Result<f64> {
    evaluate_table_with_query_and_diagnostics(
        library,
        table,
        TimingTableQuery::combinational(input_transition, output_load),
        timing_query_diagnostic_counts,
        context,
    )
}

fn evaluate_table_with_query_and_diagnostics(
    library: &crate::liberty_proto::Library,
    table: &TimingTable,
    query: TimingTableQuery,
    timing_query_diagnostic_counts: &mut TimingQueryDiagnosticCounts,
    context: &str,
) -> Result<f64> {
    validate_non_negative_finite(query.input_transition, "input transition query", context)?;
    validate_non_negative_finite(query.output_load, "output load query", context)?;
    validate_non_negative_finite(
        query.constrained_pin_transition,
        "constrained pin transition query",
        context,
    )?;
    validate_non_negative_finite(
        query.related_pin_transition,
        "related pin transition query",
        context,
    )?;
    let array = TimingTableArrayView::from_timing_table(table)
        .map_err(|e| anyhow!("{context}: invalid timing table payload: {e}"))?;
    let layout = timing_table_layout(library, table, context)?;
    let rank = array.rank();
    if rank == 0 {
        return array
            .get(&[])
            .ok_or_else(|| anyhow!("{context}: scalar timing table had no value"));
    }
    let mut bounds: Vec<(usize, usize, f64)> = Vec::with_capacity(rank);
    let mut axis_queries: Vec<f64> = Vec::with_capacity(rank);
    let is_setup = matches!(
        LibertyTableKind::from_raw(table.kind.as_str()),
        LibertyTableKind::RiseConstraint | LibertyTableKind::FallConstraint
    );
    let mut above_max_axis_count = 0usize;

    for axis_idx in 0..rank {
        let axis = layout.axes[axis_idx];
        let axis_variable = AxisVariable::from_raw(layout.variables[axis_idx]);
        let raw_axis_query =
            axis_query_value_with_query(layout.variables[axis_idx], axis_idx, query, context)?;
        let axis_lo = axis[0];
        let axis_hi = axis[axis.len() - 1];
        let selects_minimum_characterized_coordinate = match query.minimum_characterized_axis {
            MinimumCharacterizedAxis::None => false,
            MinimumCharacterizedAxis::InputTransition => {
                axis_variable == AxisVariable::InputTransition
                    || (axis_variable == AxisVariable::Unspecified && axis_idx == 0)
            }
            MinimumCharacterizedAxis::RelatedPinTransition => {
                axis_variable == AxisVariable::RelatedPinTransition
            }
        } || (query.output_load == 0.0
            && (axis_variable == AxisVariable::OutputLoad
                || (axis_variable == AxisVariable::Unspecified && axis_idx == 1)));
        let axis_query = if selects_minimum_characterized_coordinate {
            axis_lo
        } else {
            raw_axis_query
        };
        if axis_query < axis_lo {
            if is_setup {
                timing_query_diagnostic_counts.setup_below_min_clamp_count += 1;
            } else {
                timing_query_diagnostic_counts.delay_slew_below_min_clamp_count += 1;
            }
        } else if axis_query > axis_hi {
            above_max_axis_count += 1;
        }
        axis_queries.push(axis_query);
    }

    let extrapolate_single_delay_slew_axis = !is_setup && above_max_axis_count == 1;
    if is_setup {
        timing_query_diagnostic_counts.setup_above_max_clamp_count += above_max_axis_count;
    } else if above_max_axis_count == 1 {
        timing_query_diagnostic_counts.delay_slew_single_above_max_extrapolation_count += 1;
    } else if above_max_axis_count > 1 {
        timing_query_diagnostic_counts.delay_slew_multiple_above_max_clamp_count += 1;
    }

    for (axis_idx, raw_axis_query) in axis_queries.into_iter().enumerate() {
        let axis = layout.axes[axis_idx];
        let axis_lo = axis[0];
        let axis_hi = axis[axis.len() - 1];
        let mut axis_query = raw_axis_query;
        if axis_query < axis_lo {
            sta_trace(|| {
                format!(
                    "table_query_clamped_below_min context='{}' axis={} var='{}' query={:.6} axis_lo={:.6} axis_hi={:.6}",
                    context,
                    axis_idx + 1,
                    layout.variables[axis_idx],
                    raw_axis_query,
                    axis_lo,
                    axis_hi,
                )
            });
        } else if axis_query > axis_hi {
            let action = if is_setup {
                "clamped_above_max_setup"
            } else if extrapolate_single_delay_slew_axis {
                "extrapolated_above_max_single_axis"
            } else {
                "clamped_above_max_multiple_axes"
            };
            sta_trace(|| {
                format!(
                    "table_query_{} context='{}' axis={} var='{}' query={:.6} axis_lo={:.6} axis_hi={:.6}",
                    action,
                    context,
                    axis_idx + 1,
                    layout.variables[axis_idx],
                    raw_axis_query,
                    axis_lo,
                    axis_hi,
                )
            });
        }
        if axis_query < axis_lo {
            axis_query = axis_query.max(axis_lo);
        }
        if !extrapolate_single_delay_slew_axis || axis_query <= axis_hi {
            axis_query = axis_query.min(axis_hi);
        }
        bounds.push(bracket_axis(axis, axis_query));
    }

    let mut indices = vec![0usize; rank];
    let mut varying_axes: Vec<(usize, usize, usize, f64)> = Vec::with_capacity(rank);
    for (axis_idx, (lo, hi, t)) in bounds.iter().copied().enumerate() {
        if lo == hi {
            indices[axis_idx] = lo;
        } else {
            varying_axes.push((axis_idx, lo, hi, t));
        }
    }

    let corner_count = 1usize << varying_axes.len();
    let mut result = 0.0;
    for corner in 0..corner_count {
        let mut weight = 1.0;
        for (bit_idx, (axis_idx, lo, hi, t)) in varying_axes.iter().enumerate() {
            if ((corner >> bit_idx) & 1) == 1 {
                indices[*axis_idx] = *hi;
                weight *= *t;
            } else {
                indices[*axis_idx] = *lo;
                weight *= 1.0 - *t;
            }
        }
        let value = evaluate_table_corner_value(&array, table, indices.as_slice(), context)?;
        result += weight * value;
    }
    if !result.is_finite() {
        return Err(anyhow!(
            "{context}: timing table evaluation produced non-finite result {}",
            result
        ));
    }
    Ok(result)
}

fn uses_monotone_upper_envelope(table: &TimingTable) -> bool {
    matches!(
        LibertyTableKind::from_raw(table.kind.as_str()),
        LibertyTableKind::CellRise
            | LibertyTableKind::CellFall
            | LibertyTableKind::RiseTransition
            | LibertyTableKind::FallTransition
    )
}

/// Evaluates one characterized point after conservatively repairing delay/slew
/// values: a point is at least as large as any predecessor along its axes.
fn evaluate_table_corner_value(
    array: &TimingTableArrayView<'_>,
    table: &TimingTable,
    indices: &[usize],
    context: &str,
) -> Result<f64> {
    if !uses_monotone_upper_envelope(table) || indices.is_empty() {
        return array
            .get(indices)
            .ok_or_else(|| anyhow!("{context}: could not index timing table at {:?}", indices));
    }

    let mut cursor = vec![0usize; indices.len()];
    let mut maximum = f64::NEG_INFINITY;
    loop {
        let value = array
            .get(cursor.as_slice())
            .ok_or_else(|| anyhow!("{context}: could not index timing table at {:?}", cursor))?;
        maximum = maximum.max(value);

        let mut advanced = false;
        for axis_idx in (0..cursor.len()).rev() {
            if cursor[axis_idx] < indices[axis_idx] {
                cursor[axis_idx] += 1;
                for trailing in cursor.iter_mut().skip(axis_idx + 1) {
                    *trailing = 0;
                }
                advanced = true;
                break;
            }
        }
        if !advanced {
            break;
        }
    }
    Ok(maximum)
}

fn validate_non_negative_finite(value: f64, what: &str, context: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(anyhow!("{context}: {what} must be finite; got {}", value));
    }
    if value < 0.0 {
        return Err(anyhow!(
            "{context}: {what} must be non-negative; got {}",
            value
        ));
    }
    Ok(())
}

fn axis_rank(index_1: &[f64], index_2: &[f64], index_3: &[f64]) -> usize {
    if !index_3.is_empty() {
        3
    } else if !index_2.is_empty() {
        2
    } else if !index_1.is_empty() {
        1
    } else {
        0
    }
}

fn axes_are_contiguous(index_1: &[f64], index_2: &[f64], index_3: &[f64]) -> bool {
    if !index_3.is_empty() && (index_1.is_empty() || index_2.is_empty()) {
        return false;
    }
    if !index_2.is_empty() && index_1.is_empty() {
        return false;
    }
    true
}

fn validate_effective_axes(table: &TimingTable, axes: [&[f64]; 3], context: &str) -> Result<()> {
    if !axes_are_contiguous(axes[0], axes[1], axes[2]) {
        return Err(anyhow!(
            "{context}: timing table has non-contiguous effective axes (index_1={}, index_2={}, index_3={})",
            axes[0].len(),
            axes[1].len(),
            axes[2].len()
        ));
    }
    let expected_rank = axis_rank(axes[0], axes[1], axes[2]);
    if table.dimensions.len() != expected_rank {
        return Err(anyhow!(
            "{context}: timing table dimension rank {} does not match effective axis rank {}",
            table.dimensions.len(),
            expected_rank
        ));
    }
    for (axis_idx, axis) in axes.iter().take(expected_rank).enumerate() {
        let dimension = table.dimensions[axis_idx] as usize;
        if dimension != axis.len() {
            return Err(anyhow!(
                "{context}: timing table axis {} dimension {} does not match effective axis length {}",
                axis_idx + 1,
                dimension,
                axis.len()
            ));
        }
    }
    Ok(())
}

fn effective_axis<'a>(table_axis: &'a [f64], template_axis: Option<&'a [f64]>) -> &'a [f64] {
    if table_axis.is_empty() {
        template_axis.unwrap_or(&[])
    } else {
        table_axis
    }
}

#[cfg(test)]
fn axis_query_value(
    variable_name: &str,
    axis_idx: usize,
    input_transition: f64,
    output_load: f64,
    context: &str,
) -> Result<f64> {
    axis_query_value_with_query(
        variable_name,
        axis_idx,
        TimingTableQuery::combinational(input_transition, output_load),
        context,
    )
}

fn axis_query_value_with_query(
    variable_name: &str,
    axis_idx: usize,
    query: TimingTableQuery,
    context: &str,
) -> Result<f64> {
    match AxisVariable::from_raw(variable_name) {
        AxisVariable::Unspecified => match axis_idx {
            0 => Ok(query.input_transition),
            1 => Ok(query.output_load),
            _ => Err(anyhow!(
                "{context}: missing variable name for axis {}; cannot infer query value",
                axis_idx + 1
            )),
        },
        AxisVariable::InputTransition => Ok(query.input_transition),
        AxisVariable::OutputLoad => Ok(query.output_load),
        AxisVariable::ConstrainedPinTransition => Ok(query.constrained_pin_transition),
        AxisVariable::RelatedPinTransition => Ok(query.related_pin_transition),
        AxisVariable::Other => Err(anyhow!(
            "{context}: unsupported axis variable '{}' for basic STA",
            variable_name
        )),
    }
}

fn is_strictly_increasing(values: &[f64]) -> bool {
    values
        .windows(2)
        .all(|w| w.first().copied().unwrap_or(0.0) < w.get(1).copied().unwrap_or(0.0))
}

fn bracket_axis(axis: &[f64], query: f64) -> (usize, usize, f64) {
    debug_assert!(!axis.is_empty());
    if axis.len() == 1 {
        return (0, 0, 0.0);
    }

    let last = axis.len() - 1;
    if query <= axis[0] {
        let lo = 0usize;
        let hi = 1usize;
        let denom = axis[hi] - axis[lo];
        if denom.abs() < f64::EPSILON {
            return (0, 0, 0.0);
        }
        return (lo, hi, (query - axis[lo]) / denom);
    }
    if query >= axis[last] {
        let lo = last - 1;
        let hi = last;
        let denom = axis[hi] - axis[lo];
        if denom.abs() < f64::EPSILON {
            return (last, last, 0.0);
        }
        return (lo, hi, (query - axis[lo]) / denom);
    }

    for idx in 0..last {
        let lo = axis[idx];
        let hi = axis[idx + 1];
        if query >= lo && query <= hi {
            let denom = hi - lo;
            if denom.abs() < f64::EPSILON {
                return (idx, idx, 0.0);
            }
            let t = (query - lo) / denom;
            return (idx, idx + 1, t);
        }
    }

    (last, last, 0.0)
}

fn resolve_symbol(
    interner: &StringInterner<StringBackend<SymbolU32>>,
    sym: SymbolU32,
    what: &str,
) -> Result<String> {
    interner
        .resolve(sym)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("could not resolve {} symbol {:?}", what, sym))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_proto::{Cell, Pin, TimingArc, TimingTable};
    use crate::netlist::bench_synth_netlist;
    use crate::netlist::parse::{Parser, TokenScanner};

    fn parse_single_module(
        src: &str,
    ) -> (
        NetlistModule,
        Vec<Net>,
        StringInterner<StringBackend<SymbolU32>>,
    ) {
        let bytes = src.as_bytes().to_vec();
        let lines: Vec<String> = src.lines().map(ToString::to_string).collect();
        let lookup = move |lineno: u32| lines.get((lineno - 1) as usize).cloned();
        let scanner = TokenScanner::with_line_lookup(std::io::Cursor::new(bytes), Box::new(lookup));
        let mut parser = Parser::new(scanner);
        let mut modules = parser.parse_file().expect("parse should succeed");
        assert_eq!(modules.len(), 1);
        (modules.remove(0), parser.nets, parser.interner)
    }

    fn find_net_index(
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        name: &str,
    ) -> NetIndex {
        nets.iter()
            .enumerate()
            .find_map(|(idx, n)| {
                if interner.resolve(n.name) == Some(name) {
                    Some(NetIndex(idx))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("missing net '{}'", name))
    }

    fn scalar_table(kind: &str, value: f64) -> TimingTable {
        TimingTable {
            kind: kind.to_string(),
            values: vec![value],
            dimensions: vec![],
            ..Default::default()
        }
    }

    #[test]
    fn axis_query_value_accepts_only_explicit_supported_variable_names() {
        assert_eq!(
            axis_query_value("input_net_transition", 0, 0.2, 0.7, "axis").unwrap(),
            0.2
        );
        assert_eq!(
            axis_query_value("input_transition_time", 0, 0.2, 0.7, "axis").unwrap(),
            0.2
        );
        assert_eq!(
            axis_query_value("total_output_net_capacitance", 1, 0.2, 0.7, "axis").unwrap(),
            0.7
        );
        assert!(
            axis_query_value("made_up_transition", 0, 0.2, 0.7, "axis")
                .unwrap_err()
                .to_string()
                .contains("unsupported axis variable")
        );
        assert!(
            axis_query_value("made_up_capacitance", 1, 0.2, 0.7, "axis")
                .unwrap_err()
                .to_string()
                .contains("unsupported axis variable")
        );
    }

    #[test]
    fn ideal_clock_queries_select_minimum_characterized_slew_without_diagnostic() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![
                LuTableTemplate {
                    kind: "lu_table_template".to_string(),
                    name: "delay".to_string(),
                    variable_1: "input_net_transition".to_string(),
                    index_1: vec![5.0, 10.0],
                    ..Default::default()
                },
                LuTableTemplate {
                    kind: "lu_table_template".to_string(),
                    name: "constraint".to_string(),
                    variable_1: "constrained_pin_transition".to_string(),
                    variable_2: "related_pin_transition".to_string(),
                    index_1: vec![1.0, 2.0],
                    index_2: vec![5.0, 10.0],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let c2q = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2],
            values: vec![7.0, 9.0],
            ..Default::default()
        };
        let setup = TimingTable {
            kind: "rise_constraint".to_string(),
            template_id: 2,
            dimensions: vec![2, 2],
            values: vec![3.0, 1.0, 4.0, 2.0],
            ..Default::default()
        };
        let mut counts = TimingQueryDiagnosticCounts::default();

        assert_close(
            evaluate_table_with_query_and_diagnostics(
                &lib,
                &c2q,
                TimingTableQuery::ideal_clock_to_output(0.0),
                &mut counts,
                "clock_to_q",
            )
            .expect("clock-to-output table evaluation"),
            7.0,
        );
        validate_timing_table_structure(&lib, &setup, "setup")
            .expect("non-monotone setup table should be structurally valid");
        assert_close(
            evaluate_table_with_query_and_diagnostics(
                &lib,
                &setup,
                TimingTableQuery::ideal_clock_setup(1.0),
                &mut counts,
                "setup",
            )
            .expect("setup table evaluation"),
            3.0,
        );
        assert_close(
            evaluate_table_with_query_and_diagnostics(
                &lib,
                &setup,
                TimingTableQuery {
                    input_transition: 0.0,
                    output_load: 0.0,
                    constrained_pin_transition: 1.0,
                    related_pin_transition: 10.0,
                    minimum_characterized_axis: MinimumCharacterizedAxis::None,
                },
                &mut counts,
                "raw_setup_surface",
            )
            .expect("setup table evaluation"),
            1.0,
        );
        assert_eq!(counts, TimingQueryDiagnosticCounts::default());
    }

    #[test]
    fn template_kind_lookup_rejects_unknown_power_named_table_kinds() {
        let table = scalar_table("made_up_power", 1.0);
        assert!(
            expected_template_kind_for_timing_table(&table)
                .unwrap_err()
                .to_string()
                .contains("unsupported Liberty table kind")
        );
    }

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= 1e-9,
            "expected {} ~= {} (|diff|={})",
            lhs,
            rhs,
            (lhs - rhs).abs()
        );
    }

    fn scalar_inv_library() -> crate::liberty_proto::Library {
        crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "negative_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 2.0),
                                scalar_table("cell_fall", 3.0),
                                scalar_table("rise_transition", 0.2),
                                scalar_table("fall_transition", 0.3),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn sta_accepts_scalar_literal_output_tie_offs() {
        let src = r#"
module top (y);
  output y;
  wire y;
  assign y = 1'b0;
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("scalar literal output tie-off should be supported");
        assert_eq!(report.worst_output_arrival, 0.0);
        assert_eq!(report.cell_levels, 0);
    }

    #[test]
    fn sta_accepts_constant_function_tie_cells() {
        let src = r#"
module top (y_hi, y_lo, z);
  output y_hi;
  output y_lo;
  output z;
  wire y_hi;
  wire y_lo;
  wire z;
  TIEHI tie_hi (.H(y_hi));
  TIELO tie_lo (.L(y_lo));
  INV inv (.A(y_hi), .Y(z));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let mut lib = scalar_inv_library();
        lib.cells.extend([
            Cell {
                name: "TIEHI".to_string(),
                pins: vec![Pin {
                    direction: PinDirection::Output as i32,
                    function: "1".to_string(),
                    name: "H".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            },
            Cell {
                name: "TIELO".to_string(),
                pins: vec![Pin {
                    direction: PinDirection::Output as i32,
                    function: "0".to_string(),
                    name: "L".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            },
        ]);

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("constant-function tie cells should be supported");
        assert_close(report.worst_output_arrival, 3.0);
        for name in ["y_hi", "y_lo"] {
            let timing = report
                .timing_for_net(find_net_index(&nets, &interner, name))
                .unwrap_or_else(|| panic!("timing for {}", name));
            assert_close(timing.rise.arrival, 0.0);
            assert_close(timing.fall.arrival, 0.0);
        }
        let z_timing = report
            .timing_for_net(find_net_index(&nets, &interner, "z"))
            .expect("timing for z");
        assert_close(z_timing.rise.arrival, 2.0);
        assert_close(z_timing.fall.arrival, 3.0);
    }

    #[test]
    fn sta_rejects_arcless_nonconstant_output_pin() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  BUF u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "BUF".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        function: "A".to_string(),
                        name: "Y".to_string(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("arc-less nonconstant outputs should still be rejected");
        assert!(
            error
                .to_string()
                .contains("has no combinational timing arcs and no constant function")
        );
    }

    #[test]
    fn sta_accepts_scalar_alias_output_assigns() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n;
  INV u0 (.A(a), .Y(n));
  assign y = n;
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("scalar alias output assign should be supported");
        let n_idx = find_net_index(&nets, &interner, "n");
        let y_idx = find_net_index(&nets, &interner, "y");
        assert_eq!(report.timing_for_net(y_idx), report.timing_for_net(n_idx));
        assert_close(report.worst_output_arrival, 3.0);
        assert_eq!(report.cell_levels, 1);
    }

    #[test]
    fn sta_accepts_scalar_alias_chains_to_instance_inputs() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  wire n1;
  INV u0 (.A(a), .Y(n0));
  assign n1 = n0;
  INV u1 (.A(n1), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("scalar alias should carry timing into downstream instances");
        assert_close(report.worst_output_arrival, 5.0);
        assert_eq!(report.cell_levels, 2);
    }

    #[test]
    fn sta_rejects_non_alias_continuous_assigns() {
        let src = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  assign y = a & b;
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("non-alias continuous assigns should be rejected");
        assert!(
            error
                .to_string()
                .contains("unsupported assign to 'y' drives live bit")
        );
    }

    #[test]
    fn sta_ignores_unused_concat_continuous_assigns() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n;
  wire [1:0] unused;
  INV u0 (.A(a), .Y(n));
  INV u1 (.A(n), .Y(y));
  assign unused = {1'b0, a};
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("unused concat assign should not affect scalar STA");
        assert_close(report.worst_output_arrival, 5.0);
        assert_eq!(report.cell_levels, 2);
    }

    #[test]
    fn sta_accepts_live_concat_continuous_assigns() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire live;
  assign live = {a};
  INV u0 (.A(live), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("live concat assign should be modeled as bit wiring");
        assert_close(report.worst_output_arrival, 3.0);
        assert_eq!(report.cell_levels, 1);
    }

    #[test]
    fn sta_accepts_live_concat_lhs_continuous_assigns() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire [1:0] live;
  assign {live[1], live[0]} = {1'b0, a};
  INV u0 (.A(live[0]), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("live concat lhs assign should be modeled as bit wiring");
        assert_close(report.worst_output_arrival, 3.0);
        assert_eq!(report.cell_levels, 1);
    }

    #[test]
    fn sta_accepts_live_unknown_literal_continuous_assigns() {
        let src = r#"
module top (y);
  output y;
  wire y;
  wire live;
  assign live = 1'hx;
  INV u0 (.A(live), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("live unknown literal assign should be a zero-delay timing source");
        assert_close(report.worst_output_arrival, 3.0);
        assert_eq!(report.cell_levels, 1);
    }

    #[test]
    fn sta_accepts_live_concat_through_alias_assigns() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire live_source;
  wire live_alias;
  assign live_source = {a};
  assign live_alias = live_source;
  INV u0 (.A(live_alias), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("live concat assign should be modeled through aliases");
        assert_close(report.worst_output_arrival, 3.0);
        assert_eq!(report.cell_levels, 1);
    }

    #[test]
    fn sta_rejects_non_finite_options() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let transition_error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions {
                primary_input_transition: f64::NAN,
                module_output_load: 0.0,
            },
        )
        .expect_err("NaN primary-input transition should be rejected");
        assert!(
            transition_error
                .to_string()
                .contains("primary_input_transition must be finite")
        );

        let load_error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions {
                primary_input_transition: 0.01,
                module_output_load: f64::INFINITY,
            },
        )
        .expect_err("infinite module-output load should be rejected");
        assert!(
            load_error
                .to_string()
                .contains("module_output_load must be finite")
        );
    }

    #[test]
    fn sta_rejects_negative_options() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let transition_error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions {
                primary_input_transition: -0.1,
                module_output_load: 0.0,
            },
        )
        .expect_err("negative primary-input transition should be rejected");
        assert!(
            transition_error
                .to_string()
                .contains("primary_input_transition must be non-negative")
        );

        let load_error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions {
                primary_input_transition: 0.01,
                module_output_load: -0.1,
            },
        )
        .expect_err("negative module-output load should be rejected");
        assert!(
            load_error
                .to_string()
                .contains("module_output_load must be non-negative")
        );
    }

    #[test]
    fn sta_accepts_vector_bit_select_connectivity() {
        let src = r#"
module top (a, y);
  input [1:0] a;
  output [1:0] y;
  wire [1:0] a;
  wire [1:0] y;
  INV u0 (.A(a[0]), .Y(y[0]));
  INV u1 (.A(a[1]), .Y(y[1]));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("bit-select connectivity should be accepted");
        assert_close(report.worst_output_arrival, 3.0);
        assert_eq!(report.cell_levels, 1);
    }

    #[test]
    fn sta_rejects_whole_vector_simple_connectivity() {
        let src = r#"
module top (a, y);
  input [1:0] a;
  output [1:0] y;
  wire [1:0] a;
  wire [1:0] y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("whole-vector simple connectivity should be rejected");
        assert!(error.to_string().contains("connects 2 bits"));
    }

    #[test]
    fn sta_accepts_implicit_scalar_connectivity() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(n));
  INV u1 (.A(n), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("implicit scalar wire should be accepted");
        assert_close(report.worst_output_arrival, 5.0);
    }

    #[test]
    fn sta_rejects_floating_internal_source_nets() {
        let src = r#"
module top (y);
  output y;
  wire y;
  wire floating_internal;
  INV u0 (.A(floating_internal), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("floating internal source nets should be rejected");
        assert!(error.to_string().contains("undriven"));
    }

    #[test]
    fn sta_rejects_missing_timing_related_input_pins() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  NAND2 u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "B".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                            TimingArc {
                                related_pin: "B".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("missing timing-related input pins should be rejected");
        assert!(
            error
                .to_string()
                .contains("requires timing-related input pin 'B' to be connected")
        );
    }

    #[test]
    fn sta_accepts_literal_tied_timing_inputs() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  NAND2 u0 (.A(a), .B(1'b0), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "B".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                            TimingArc {
                                related_pin: "B".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 4.0),
                                    scalar_table("cell_fall", 5.0),
                                    scalar_table("rise_transition", 0.4),
                                    scalar_table("fall_transition", 0.5),
                                ],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("literal-tied timing inputs should use a zero-transition literal source");
        assert_close(report.worst_output_arrival, 5.0);
    }

    #[test]
    fn sta_propagates_output_related_timing_arcs_before_library_pin_order() {
        let src = r#"
module top (a, y0, y1);
  input a;
  output y0;
  output y1;
  wire a;
  wire y0;
  wire y1;
  DUALOUT u0 (.A(a), .Y0(y0), .Y1(y1));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "DUALOUT".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y1".to_string(),
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 3.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.1),
                                    scalar_table("fall_transition", 0.1),
                                ],
                                ..Default::default()
                            },
                            TimingArc {
                                related_pin: "Y0".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 1.0),
                                    scalar_table("cell_fall", 1.0),
                                    scalar_table("rise_transition", 0.1),
                                    scalar_table("fall_transition", 0.1),
                                ],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y0".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 1.0),
                                scalar_table("cell_fall", 1.0),
                                scalar_table("rise_transition", 0.1),
                                scalar_table("fall_transition", 0.1),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("acyclic output-related arcs should be propagated");
        let y0_idx = find_net_index(&nets, &interner, "y0");
        let y1_idx = find_net_index(&nets, &interner, "y1");
        assert_close(
            report
                .timing_for_net(y0_idx)
                .expect("timing for y0")
                .rise
                .arrival,
            1.0,
        );
        assert_close(
            report
                .timing_for_net(y1_idx)
                .expect("timing for y1")
                .rise
                .arrival,
            3.0,
        );
        assert_close(report.worst_output_arrival, 3.0);
    }

    #[test]
    fn sta_propagates_output_related_timing_arcs_through_unconnected_outputs() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  DUALOUT u0 (.A(a), .Y1(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "DUALOUT".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y1".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "Y0".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 1.0),
                                scalar_table("cell_fall", 1.0),
                                scalar_table("rise_transition", 0.1),
                                scalar_table("fall_transition", 0.1),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y0".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 1.0),
                                scalar_table("cell_fall", 1.0),
                                scalar_table("rise_transition", 0.1),
                                scalar_table("fall_transition", 0.1),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("unconnected output dependencies should use local timing");
        assert_close(report.worst_output_arrival, 2.0);
    }

    #[test]
    fn sta_rejects_output_related_timing_arc_cycles() {
        let src = r#"
module top (y);
  output y;
  wire y;
  DUALOUT u0 (.Y0(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "DUALOUT".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y0".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "Y1".to_string(),
                            timing_type: "combinational".to_string(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y1".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "Y0".to_string(),
                            timing_type: "combinational".to_string(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("cyclic output-related arcs should be rejected");
        assert!(
            error
                .to_string()
                .contains("combinational output-pin dependency cycle")
        );
    }

    #[test]
    fn sta_rejects_primary_inputs_with_instance_drivers() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(y), .Y(a));
  INV u1 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("primary inputs with instance drivers should be rejected");
        assert!(error.to_string().contains("also has an internal driver"));
    }

    #[test]
    fn sta_rejects_duplicate_instance_pin_bindings() {
        let src = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  INV u0 (.A(a), .A(b), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("duplicate instance pin bindings should be rejected");
        assert!(
            error
                .to_string()
                .contains("connects pin 'A' more than once")
        );
    }

    #[test]
    fn sta_rejects_output_pins_bound_to_literals() {
        let src = r#"
module top (a);
  input a;
  wire a;
  INV u0 (.A(a), .Y(1'b0));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("literal output pin bindings should be rejected");
        assert!(
            error
                .to_string()
                .contains("uses unsupported literal or unconnected binding")
        );
    }

    #[test]
    fn sta_rejects_inout_ports() {
        let src = r#"
module top (io);
  inout io;
  wire io;
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("inout ports should be rejected");
        assert!(error.to_string().contains("is inout"));
    }

    #[test]
    fn sta_combines_one_sided_combinational_rise_and_fall_arcs() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational_rise".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("rise_transition", 0.2),
                                ],
                                ..Default::default()
                            },
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational_fall".to_string(),
                                tables: vec![
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.1,
                module_output_load: 0.0,
            },
        )
        .expect("one-sided combinational arcs should combine");
        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 2.0);
        assert_close(y_timing.fall.arrival, 3.0);
        assert_close(y_timing.rise.transition, 0.2);
        assert_close(y_timing.fall.transition, 0.3);
        assert_close(report.worst_output_arrival, 3.0);
    }

    #[test]
    fn sta_reports_negative_worst_output_arrival_without_zero_clamp() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", -2.0),
                                scalar_table("cell_fall", -1.0),
                                scalar_table("rise_transition", 0.2),
                                scalar_table("fall_transition", 0.3),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("sta should succeed");
        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, -2.0);
        assert_close(y_timing.fall.arrival, -1.0);
        assert_close(report.worst_output_arrival, -1.0);
    }

    #[test]
    fn sta_ignores_non_timing_input_pins_when_building_dependencies() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  PASS_WITH_EN u0 (.A(a), .EN(y), .Y(n0));
  INV u1 (.A(n0), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![
                Cell {
                    name: "PASS_WITH_EN".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "EN".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "INV".to_string(),
                    pins: scalar_inv_library().cells[0].pins.clone(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("non-timing EN feedback should not create a timing cycle");
        assert_close(report.worst_output_arrival, 5.0);
    }

    #[test]
    fn sta_splits_multi_pin_related_pin_names() {
        let src = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  NAND2 u0 (.A(a), .B(b), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "B".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A B".to_string(),
                            timing_sense: "negative_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 2.0),
                                scalar_table("cell_fall", 3.0),
                                scalar_table("rise_transition", 0.2),
                                scalar_table("fall_transition", 0.3),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("multi-pin related_pin names should be expanded");
        assert_close(report.worst_output_arrival, 3.0);
    }

    #[test]
    fn sta_rejects_non_combinational_output_pins() {
        let src = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  wire d;
  wire clk;
  wire q;
  DFF u0 (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "DFF".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "D".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "CLK".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Q".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "CLK".to_string(),
                            timing_type: "rising_edge".to_string(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("sequential output pin should be rejected");
        assert!(
            error
                .to_string()
                .contains("basic STA only supports combinational output pins")
        );
    }

    #[test]
    fn sta_accepts_unknown_conditional_timing_arcs_as_possible() {
        let src = r#"
module top (a, en, y);
  input a;
  input en;
  output y;
  wire a;
  wire en;
  wire y;
  BUF_EN u0 (.A(a), .EN(en), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "BUF_EN".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "EN".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            when: "EN".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 2.0),
                                scalar_table("cell_fall", 3.0),
                                scalar_table("rise_transition", 0.2),
                                scalar_table("fall_transition", 0.3),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("unknown conditional timing arcs should remain possible");
        assert_eq!(report.worst_output_arrival, 3.0);
    }

    #[test]
    fn sta_rejects_mixed_output_timing_types() {
        let src = r#"
module top (a, clk, y);
  input a;
  input clk;
  output y;
  wire a;
  wire clk;
  wire y;
  MIXED u0 (.A(a), .CLK(clk), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "MIXED".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "CLK".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                            TimingArc {
                                related_pin: "CLK".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "rising_edge".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 4.0),
                                    scalar_table("cell_fall", 5.0),
                                    scalar_table("rise_transition", 0.4),
                                    scalar_table("fall_transition", 0.5),
                                ],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("mixed output timing types should be rejected");
        assert!(
            error
                .to_string()
                .contains("has unsupported timing type 'rising_edge'")
        );
    }

    #[test]
    fn effective_input_capacitance_ignores_max_capacitance_constraint() {
        let with_nominal = Pin {
            capacitance: Some(0.6),
            rise_capacitance: Some(0.7),
            fall_capacitance: Some(0.5),
            max_capacitance: Some(46.1),
            ..Default::default()
        };
        assert_close(effective_input_capacitance(&with_nominal), 0.6);

        let no_nominal = Pin {
            rise_capacitance: Some(0.6),
            fall_capacitance: Some(0.5),
            max_capacitance: Some(46.1),
            ..Default::default()
        };
        assert_close(effective_input_capacitance(&no_nominal), 0.6);

        let only_max = Pin {
            max_capacitance: Some(46.1),
            ..Default::default()
        };
        assert_close(effective_input_capacitance(&only_max), 0.0);
    }

    #[test]
    fn effective_input_capacitance_by_edge_rejects_negative_values() {
        let negative = Pin {
            rise_capacitance: Some(-0.1),
            fall_capacitance: Some(0.2),
            ..Default::default()
        };
        let error = effective_input_capacitance_by_edge(&negative, "pin")
            .expect_err("negative capacitance should be rejected");
        assert!(
            error
                .to_string()
                .contains("rise capacitance must be non-negative")
        );
    }

    #[test]
    fn sta_uses_nominal_input_capacitance_for_net_load() {
        let src = r#"
module top (a, n, y);
  input a;
  output n;
  output y;
  wire n;
  wire y;
  INV u0 (.A(a), .Y(n));
  INV u1 (.A(n), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2, 2],
            values,
            ..Default::default()
        };
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_2d".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![0.0, 1.0],
                index_2: vec![0.0, 10.0],
                ..Default::default()
            }],
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        capacitance: Some(1.0),
                        max_capacitance: Some(1000.0),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                table("cell_rise", vec![0.0, 10.0, 0.0, 10.0]),
                                table("cell_fall", vec![0.0, 10.0, 0.0, 10.0]),
                                table("rise_transition", vec![0.0, 0.0, 0.0, 0.0]),
                                table("fall_transition", vec![0.0, 0.0, 0.0, 0.0]),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.0,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let n_idx = find_net_index(&nets, &interner, "n");
        let y_idx = find_net_index(&nets, &interner, "y");
        let n_timing = report.timing_for_net(n_idx).expect("timing for n");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");

        assert_close(n_timing.rise.arrival, 1.0);
        assert_close(n_timing.fall.arrival, 1.0);
        assert_close(y_timing.rise.arrival, 1.0);
        assert_close(y_timing.fall.arrival, 1.0);
    }

    #[test]
    fn sta_inv_chain_propagates_arrival_and_transition() {
        let src = bench_synth_netlist::make_chain_netlist(2);
        let (module, nets, interner) = parse_single_module(src.as_str());
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 2.0),
                                scalar_table("cell_fall", 3.0),
                                scalar_table("rise_transition", 0.4),
                                scalar_table("fall_transition", 0.5),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.2,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 4.0);
        assert_close(y_timing.fall.arrival, 6.0);
        assert_close(y_timing.rise.transition, 0.4);
        assert_close(y_timing.fall.transition, 0.5);
        assert_close(report.worst_output_arrival, 6.0);
    }

    #[test]
    fn sta_negative_unate_uses_opposite_input_edge() {
        let src = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  wire n0;
  INV u0 (.A(a), .Y(n0));
  NAND2 u1 (.A(n0), .B(b), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let inv_arc = TimingArc {
            related_pin: "A".to_string(),
            timing_sense: "positive_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                scalar_table("cell_rise", 2.0),
                scalar_table("cell_fall", 7.0),
                scalar_table("rise_transition", 0.2),
                scalar_table("fall_transition", 0.7),
            ],
            ..Default::default()
        };
        let nand_arc_a = TimingArc {
            related_pin: "A".to_string(),
            timing_sense: "negative_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                scalar_table("cell_rise", 100.0),
                scalar_table("cell_fall", 10.0),
                scalar_table("rise_transition", 1.0),
                scalar_table("fall_transition", 1.0),
            ],
            ..Default::default()
        };
        let nand_arc_b = TimingArc {
            related_pin: "B".to_string(),
            timing_sense: "negative_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                scalar_table("cell_rise", 1.0),
                scalar_table("cell_fall", 1.0),
                scalar_table("rise_transition", 0.5),
                scalar_table("fall_transition", 0.5),
            ],
            ..Default::default()
        };
        let lib = crate::liberty_proto::Library {
            cells: vec![
                Cell {
                    name: "INV".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![inv_arc],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "NAND2".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "B".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![nand_arc_a, nand_arc_b],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.1,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 107.0);
        assert_close(y_timing.fall.arrival, 12.0);
        assert_close(report.worst_output_arrival, 107.0);
    }

    #[test]
    fn sta_non_unate_keeps_arrival_and_transition_correlated_per_source_edge() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1d_transition".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![1.0, 5.0],
                ..Default::default()
            }],
            ..Default::default()
        };

        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2],
            values,
            ..Default::default()
        };

        let arc = TimingArc {
            related_pin: "A".to_string(),
            timing_sense: "non_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                table("cell_rise", vec![10.0, 50.0]),
                table("cell_fall", vec![4.0, 20.0]),
                table("rise_transition", vec![2.0, 8.0]),
                table("fall_transition", vec![3.0, 7.0]),
            ],
            ..Default::default()
        };

        let input = SignalTiming {
            rise: EdgeTiming {
                arrival: 200.0,
                transition: 1.0,
            },
            fall: EdgeTiming {
                arrival: 100.0,
                transition: 5.0,
            },
        };

        let output = evaluate_arc(&lib, &arc, input, 0.0, "non_unate_test").expect("evaluate arc");

        // Rise edge should come from rise input edge (200 + 10), not mixed with
        // fall-edge transition.
        assert_close(output.rise.arrival, 210.0);
        assert_close(output.rise.transition, 2.0);

        // Fall edge should also come from rise input edge (200 + 4).
        assert_close(output.fall.arrival, 204.0);
        assert_close(output.fall.transition, 3.0);
    }
    #[test]
    fn sta_composed_logic_keeps_non_dominated_input_candidates() {
        let src = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  wire n0;
  MERGE u0 (.A(a), .B(b), .Y(n0));
  LOADSENS u1 (.A(n0), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);

        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2],
            values,
            ..Default::default()
        };

        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1d_transition".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![1.0, 5.0],
                ..Default::default()
            }],
            cells: vec![
                Cell {
                    name: "MERGE".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "B".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![
                                TimingArc {
                                    related_pin: "A".to_string(),
                                    timing_sense: "positive_unate".to_string(),
                                    timing_type: "combinational".to_string(),
                                    tables: vec![
                                        scalar_table("cell_rise", 10.0),
                                        scalar_table("cell_fall", 10.0),
                                        scalar_table("rise_transition", 1.0),
                                        scalar_table("fall_transition", 1.0),
                                    ],
                                    ..Default::default()
                                },
                                TimingArc {
                                    related_pin: "B".to_string(),
                                    timing_sense: "positive_unate".to_string(),
                                    timing_type: "combinational".to_string(),
                                    tables: vec![
                                        scalar_table("cell_rise", 8.0),
                                        scalar_table("cell_fall", 8.0),
                                        scalar_table("rise_transition", 5.0),
                                        scalar_table("fall_transition", 5.0),
                                    ],
                                    ..Default::default()
                                },
                            ],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "LOADSENS".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    table("cell_rise", vec![10.0, 50.0]),
                                    table("cell_fall", vec![10.0, 50.0]),
                                    table("rise_transition", vec![1.0, 5.0]),
                                    table("fall_transition", vec![1.0, 5.0]),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.0,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 60.0);
        assert_close(y_timing.fall.arrival, 60.0);
        assert_close(report.worst_output_arrival, 60.0);
    }

    #[test]
    fn sta_composed_logic_handles_mixed_sense_arcs_across_stages() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  wire n1;
  SKEW u0 (.A(a), .Y(n0));
  DUAL u1 (.A(n0), .Y(n1));
  LOADSENS u2 (.A(n1), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);

        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2],
            values,
            ..Default::default()
        };

        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1d_transition".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![1.0, 5.0],
                ..Default::default()
            }],
            cells: vec![
                Cell {
                    name: "SKEW".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 40.0),
                                    scalar_table("cell_fall", 0.0),
                                    scalar_table("rise_transition", 1.0),
                                    scalar_table("fall_transition", 5.0),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "DUAL".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![
                                TimingArc {
                                    related_pin: "A".to_string(),
                                    timing_sense: "positive_unate".to_string(),
                                    timing_type: "combinational".to_string(),
                                    tables: vec![
                                        scalar_table("cell_rise", 10.0),
                                        scalar_table("cell_fall", 10.0),
                                        scalar_table("rise_transition", 1.0),
                                        scalar_table("fall_transition", 1.0),
                                    ],
                                    ..Default::default()
                                },
                                TimingArc {
                                    related_pin: "A".to_string(),
                                    timing_sense: "negative_unate".to_string(),
                                    timing_type: "combinational".to_string(),
                                    tables: vec![
                                        scalar_table("cell_rise", 20.0),
                                        scalar_table("cell_fall", 20.0),
                                        scalar_table("rise_transition", 5.0),
                                        scalar_table("fall_transition", 5.0),
                                    ],
                                    ..Default::default()
                                },
                            ],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "LOADSENS".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    table("cell_rise", vec![10.0, 50.0]),
                                    table("cell_fall", vec![10.0, 50.0]),
                                    table("rise_transition", vec![1.0, 5.0]),
                                    table("fall_transition", vec![1.0, 5.0]),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.0,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");

        assert_close(y_timing.rise.arrival, 100.0);
        assert_close(y_timing.fall.arrival, 110.0);
        assert_close(report.worst_output_arrival, 110.0);
    }

    #[test]
    fn sta_interpolates_two_dimensional_tables() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_2d".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![0.1, 0.3],
                index_2: vec![1.0, 3.0],
                ..Default::default()
            }],
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                TimingTable {
                                    kind: "cell_rise".to_string(),
                                    template_id: 1,
                                    dimensions: vec![2, 2],
                                    values: vec![10.0, 20.0, 30.0, 40.0],
                                    ..Default::default()
                                },
                                TimingTable {
                                    kind: "cell_fall".to_string(),
                                    template_id: 1,
                                    dimensions: vec![2, 2],
                                    values: vec![5.0, 7.0, 9.0, 11.0],
                                    ..Default::default()
                                },
                                TimingTable {
                                    kind: "rise_transition".to_string(),
                                    template_id: 1,
                                    dimensions: vec![2, 2],
                                    values: vec![1.0, 2.0, 3.0, 4.0],
                                    ..Default::default()
                                },
                                TimingTable {
                                    kind: "fall_transition".to_string(),
                                    template_id: 1,
                                    dimensions: vec![2, 2],
                                    values: vec![2.0, 4.0, 6.0, 8.0],
                                    ..Default::default()
                                },
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.2,
                module_output_load: 2.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 25.0);
        assert_close(y_timing.fall.arrival, 8.0);
        assert_close(y_timing.rise.transition, 2.5);
        assert_close(y_timing.fall.transition, 5.0);
        assert_close(report.worst_output_arrival, 25.0);
    }

    #[test]
    fn sta_skips_conditional_timing_arcs_when_known_false() {
        let src = r#"
module top (a, en, y);
  input a;
  output y;
  wire a;
  wire en;
  wire y;
  assign en = 1'b0;
  BUF_EN u0 (.A(a), .EN(en), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let arc = |when: &str, cell_rise: f64, cell_fall: f64| TimingArc {
            related_pin: "A".to_string(),
            timing_sense: "positive_unate".to_string(),
            timing_type: "combinational".to_string(),
            when: when.to_string(),
            tables: vec![
                scalar_table("cell_rise", cell_rise),
                scalar_table("cell_fall", cell_fall),
                scalar_table("rise_transition", 0.0),
                scalar_table("fall_transition", 0.0),
            ],
            ..Default::default()
        };
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "BUF_EN".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "EN".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![arc("EN", 10.0, 10.0), arc("!EN", 2.0, 3.0)],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("sta should skip conditional arcs that are provably false");

        assert_close(report.worst_output_arrival, 3.0);
    }

    #[test]
    fn sta_uses_edge_specific_input_capacitance_for_net_load() {
        let src = r#"
module top (a, n);
  input a;
  output n;
  wire a;
  wire n;
  DRV u0 (.A(a), .Y(n));
  SINK u1 (.A(n));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);

        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2],
            values,
            ..Default::default()
        };

        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1d_load".to_string(),
                variable_1: "total_output_net_capacitance".to_string(),
                index_1: vec![1.0, 10.0],
                ..Default::default()
            }],
            cells: vec![
                Cell {
                    name: "DRV".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    table("cell_rise", vec![10.0, 20.0]),
                                    table("cell_fall", vec![30.0, 40.0]),
                                    table("rise_transition", vec![0.0, 0.0]),
                                    table("fall_transition", vec![0.0, 0.0]),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "SINK".to_string(),
                    pins: vec![Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        capacitance: Some(9.0),
                        rise_capacitance: Some(1.0),
                        fall_capacitance: Some(10.0),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.0,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let n_idx = find_net_index(&nets, &interner, "n");
        let n_timing = report.timing_for_net(n_idx).expect("timing for n");

        // Rising output uses rise_capacitance (1.0); falling output uses
        // fall_capacitance (10.0), even when nominal capacitance is present.
        assert_close(n_timing.rise.arrival, 10.0);
        assert_close(n_timing.fall.arrival, 40.0);
    }

    #[test]
    fn evaluate_table_does_not_duplicate_weight_on_singleton_axes() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_singleton_x_axis".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![0.0],
                index_2: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![1, 2],
            values: vec![1.0, 2.0],
            ..Default::default()
        };

        let low = evaluate_table(&lib, &table, 7.0, 0.0, "singleton_axis_low").expect("table eval");
        assert_close(low, 1.0);

        let mid = evaluate_table(&lib, &table, 7.0, 0.5, "singleton_axis_mid").expect("table eval");
        assert_close(mid, 1.5);
    }

    #[test]
    fn evaluate_table_clamps_below_axis_range() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_2d_extrap".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![5.0, 10.0],
                index_2: vec![0.72, 1.44],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2, 2],
            values: vec![6.90715, 9.84125, 8.69936, 11.6159],
            ..Default::default()
        };

        let clamped = evaluate_table(&lib, &table, 0.0, 0.619_928, "clamped").expect("table eval");
        assert_close(clamped, 6.90715);
    }

    #[test]
    fn evaluate_table_counts_clamp_categories() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![
                LuTableTemplate {
                    kind: "lu_table_template".to_string(),
                    name: "tmpl_2d_delay_diagnostics".to_string(),
                    variable_1: "input_net_transition".to_string(),
                    variable_2: "total_output_net_capacitance".to_string(),
                    index_1: vec![5.0, 10.0],
                    index_2: vec![0.72, 1.44],
                    ..Default::default()
                },
                LuTableTemplate {
                    kind: "lu_table_template".to_string(),
                    name: "tmpl_2d_setup_diagnostics".to_string(),
                    variable_1: "constrained_pin_transition".to_string(),
                    variable_2: "related_pin_transition".to_string(),
                    index_1: vec![5.0, 10.0],
                    index_2: vec![5.0, 10.0],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let delay_table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2, 2],
            values: vec![1.0, 2.0, 3.0, 4.0],
            ..Default::default()
        };
        let setup_table = TimingTable {
            kind: "rise_constraint".to_string(),
            template_id: 2,
            dimensions: vec![2, 2],
            values: vec![1.0, 2.0, 3.0, 4.0],
            ..Default::default()
        };
        let mut counts = TimingQueryDiagnosticCounts::default();

        assert_close(
            evaluate_table_with_query_and_diagnostics(
                &lib,
                &delay_table,
                TimingTableQuery::combinational(5.0, 0.0),
                &mut counts,
                "delay_minimum_characterized_zero_load",
            )
            .expect("delay/slew zero-load table evaluation"),
            1.0,
        );
        evaluate_table_with_query_and_diagnostics(
            &lib,
            &delay_table,
            TimingTableQuery::combinational(0.0, 1.0),
            &mut counts,
            "delay_below_min_data",
        )
        .expect("delay/slew below-minimum transition table evaluation");
        evaluate_table_with_query_and_diagnostics(
            &lib,
            &delay_table,
            TimingTableQuery::combinational(7.0, 0.1),
            &mut counts,
            "delay_below_min_positive_load",
        )
        .expect("delay/slew below-minimum positive-load table evaluation");
        evaluate_table_with_query_and_diagnostics(
            &lib,
            &delay_table,
            TimingTableQuery::ideal_clock_to_output(1.0),
            &mut counts,
            "delay_minimum_characterized_clock",
        )
        .expect("delay/slew ideal-clock minimum-characterized table evaluation");
        let extrapolated = evaluate_table_with_query_and_diagnostics(
            &lib,
            &delay_table,
            TimingTableQuery::combinational(7.0, 2.0),
            &mut counts,
            "delay_single_above_max_load",
        )
        .expect("delay/slew single-axis above-maximum table evaluation");
        assert!(extrapolated > 3.0);
        assert_close(
            evaluate_table_with_query_and_diagnostics(
                &lib,
                &delay_table,
                TimingTableQuery::combinational(11.0, 2.0),
                &mut counts,
                "delay_multiple_above_max",
            )
            .expect("delay/slew multi-axis above-maximum table evaluation"),
            4.0,
        );
        evaluate_table_with_query_and_diagnostics(
            &lib,
            &setup_table,
            TimingTableQuery::ideal_clock_setup(7.0),
            &mut counts,
            "setup_minimum_characterized_clock",
        )
        .expect("setup ideal-clock minimum-characterized table evaluation");
        evaluate_table_with_query_and_diagnostics(
            &lib,
            &setup_table,
            TimingTableQuery {
                input_transition: 0.0,
                output_load: 0.0,
                constrained_pin_transition: 0.0,
                related_pin_transition: 7.0,
                minimum_characterized_axis: MinimumCharacterizedAxis::None,
            },
            &mut counts,
            "setup_below_min",
        )
        .expect("setup below-minimum table evaluation");
        assert_close(
            evaluate_table_with_query_and_diagnostics(
                &lib,
                &setup_table,
                TimingTableQuery {
                    input_transition: 0.0,
                    output_load: 0.0,
                    constrained_pin_transition: 11.0,
                    related_pin_transition: 5.0,
                    minimum_characterized_axis: MinimumCharacterizedAxis::None,
                },
                &mut counts,
                "setup_above_max",
            )
            .expect("setup above-maximum table evaluation"),
            3.0,
        );

        assert_eq!(
            counts,
            TimingQueryDiagnosticCounts {
                delay_slew_below_min_clamp_count: 2,
                delay_slew_single_above_max_extrapolation_count: 1,
                delay_slew_multiple_above_max_clamp_count: 1,
                setup_below_min_clamp_count: 1,
                setup_above_max_clamp_count: 1,
            }
        );
    }

    #[test]
    fn evaluate_table_rejects_axis_dimension_mismatch() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_bad_extent".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![3],
            values: vec![1.0, 2.0, 3.0],
            ..Default::default()
        };

        let error = validate_timing_table_payload(&lib, &table, "bad_extent")
            .expect_err("axis extent mismatch should be rejected");
        assert!(error.to_string().contains("axis 1 dimension 3"));
    }

    #[test]
    fn evaluate_table_rejects_axis_rank_mismatch() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_bad_rank".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![0.0, 1.0],
                index_2: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2],
            values: vec![1.0, 2.0],
            ..Default::default()
        };

        let error = validate_timing_table_payload(&lib, &table, "bad_rank")
            .expect_err("axis rank mismatch should be rejected");
        assert!(error.to_string().contains("dimension rank 1"));
    }

    #[test]
    fn evaluate_table_rejects_non_finite_axes_and_values() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_non_finite_axis".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![0.0, f64::INFINITY],
                ..Default::default()
            }],
            ..Default::default()
        };
        let axis_table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2],
            values: vec![1.0, 2.0],
            ..Default::default()
        };
        let axis_error = validate_timing_table_payload(&lib, &axis_table, "bad_axis")
            .expect_err("non-finite axes should be rejected");
        assert!(
            axis_error
                .to_string()
                .contains("axis 1 contains non-finite")
        );

        let value_table = TimingTable {
            kind: "cell_rise".to_string(),
            values: vec![f64::NAN],
            ..Default::default()
        };
        let value_error = validate_timing_table_payload(
            &crate::liberty_proto::Library::default(),
            &value_table,
            "bad_value",
        )
        .expect_err("non-finite table values should be rejected");
        assert!(
            value_error
                .to_string()
                .contains("timing table contains non-finite value")
        );
    }

    #[test]
    fn evaluate_table_rejects_duplicate_axis_points() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_duplicate_axis".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![0.0, 0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![3],
            values: vec![1.0, 2.0, 3.0],
            ..Default::default()
        };

        let error = validate_timing_table_payload(&lib, &table, "duplicate_axis")
            .expect_err("duplicate axis points should be rejected");
        assert!(
            error
                .to_string()
                .contains("axis 1 is not strictly increasing")
        );
    }

    #[test]
    fn evaluate_table_repairs_non_monotone_delay_values_with_upper_envelope() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_non_monotone".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2],
            values: vec![2.0, 1.0],
            ..Default::default()
        };

        validate_timing_table_payload(&lib, &table, "non_monotone")
            .expect("non-monotone delay tables should be structurally valid");
        assert_close(
            evaluate_table(&lib, &table, 1.0, 0.0, "non_monotone").expect("table eval"),
            2.0,
        );
    }

    #[test]
    fn evaluate_table_repairs_two_dimensional_delay_values_coordinatewise() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_non_monotone_2d".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![0.0, 1.0],
                index_2: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2, 2],
            values: vec![1.0, 4.0, 3.0, 2.0],
            ..Default::default()
        };

        assert_close(
            evaluate_table(&lib, &table, 1.0, 1.0, "non_monotone_2d").expect("table eval"),
            4.0,
        );
    }

    #[test]
    fn evaluate_table_repairs_small_characterization_noise() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_characterization_noise".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2],
            values: vec![28.4535, 28.2959],
            ..Default::default()
        };

        assert_close(
            evaluate_table(&lib, &table, 1.0, 0.0, "characterization_noise").expect("table eval"),
            28.4535,
        );
    }

    #[test]
    fn evaluate_output_edge_set_clamps_negative_transition_results() {
        let input_timing = EdgeTimingSet::from_single(EdgeTiming {
            arrival: 0.0,
            transition: 0.1,
        });
        let delay_table = scalar_table("cell_rise", 1.0);
        let slew_table = scalar_table("rise_transition", -0.1);
        let mut timing_query_diagnostic_counts = TimingQueryDiagnosticCounts::default();

        let output = evaluate_output_edge_set(
            &crate::liberty_proto::Library::default(),
            &delay_table,
            &slew_table,
            &input_timing,
            0.0,
            &mut timing_query_diagnostic_counts,
            "negative_transition",
            StaTimingTableKind::CellRise,
            StaTimingTableKind::RiseTransition,
        )
        .expect("negative transition outputs should clamp to physical zero");
        assert_eq!(
            output,
            EdgeTimingSet::from_single(EdgeTiming {
                arrival: 1.0,
                transition: 0.0,
            })
        );
    }

    #[test]
    fn evaluate_output_edge_set_rejects_non_finite_arrival_results() {
        let input_timing = EdgeTimingSet::from_single(EdgeTiming {
            arrival: f64::MAX,
            transition: 0.1,
        });
        let delay_table = scalar_table("cell_rise", f64::MAX);
        let slew_table = scalar_table("rise_transition", 0.1);
        let mut timing_query_diagnostic_counts = TimingQueryDiagnosticCounts::default();

        let error = evaluate_output_edge_set(
            &crate::liberty_proto::Library::default(),
            &delay_table,
            &slew_table,
            &input_timing,
            0.0,
            &mut timing_query_diagnostic_counts,
            "overflow_arrival",
            StaTimingTableKind::CellRise,
            StaTimingTableKind::RiseTransition,
        )
        .expect_err("overflowed arrival outputs should be rejected");
        assert!(
            error
                .to_string()
                .contains("propagated arrival must be finite")
        );
    }

    #[test]
    fn sta_rejects_duplicate_library_cells_and_pins() {
        let duplicate_cells = crate::liberty_proto::Library {
            cells: vec![
                Cell {
                    name: "INV".to_string(),
                    ..Default::default()
                },
                Cell {
                    name: "INV".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let cell_error = match StaLibraryIndex::new(&duplicate_cells) {
            Ok(_) => panic!("duplicate cells should fail"),
            Err(error) => error,
        };
        assert!(
            cell_error
                .to_string()
                .contains("defines cell 'INV' more than once")
        );

        let duplicate_pins = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        name: "A".to_string(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let pin_error = match StaLibraryIndex::new(&duplicate_pins) {
            Ok(_) => panic!("duplicate pins should fail"),
            Err(error) => error,
        };
        assert!(
            pin_error
                .to_string()
                .contains("defines pin 'A' more than once")
        );
    }

    #[test]
    fn sta_ignores_invalid_timing_tables_on_unused_cells() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let mut lib = scalar_inv_library();
        lib.cells.push(Cell {
            name: "UNUSED".to_string(),
            pins: vec![Pin {
                direction: PinDirection::Output as i32,
                name: "Y".to_string(),
                timing_arcs: vec![TimingArc {
                    related_pin: "A".to_string(),
                    timing_sense: "positive_unate".to_string(),
                    timing_type: "combinational".to_string(),
                    tables: vec![scalar_table("cell_rise", f64::NAN)],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        });

        analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("unused malformed tables should not poison an analyzable module");
    }
}
