// SPDX-License-Identifier: Apache-2.0

//! Transactional timing for cell substitution and bounded topology edits. Full
//! and incremental analyses share the output, Liberty-table and setup kernels.

use super::{
    CoverageConstantPropagation, EdgeLoadCapacitance, EdgeTiming, InstanceCoverageInputs,
    InstanceTimingInputs, NetEndpoint, PinBitSource, PreparedStaLibrary, ResolvedTimingSource,
    SignalTiming, SignalTimingSet, StaOptions, StaTimingType, TimingEvaluationContext,
    TimingQueryDiagnosticCounts, analyze_assign_sources, canonicalize_pin_bit_source,
    combinational_output_pin_evaluation_order, effective_input_capacitance_by_edge,
    evaluate_instance_outputs, evaluate_register_setup_capture_arrival,
    is_sequential_boundary_cell, resolve_symbol, resolve_timing_sources,
    validate_constraint_tables_once, validate_instance_functional_timing_coverage,
};
use crate::liberty::Library;
use crate::liberty_model::{PinDirection, TimingArc};
use crate::netlist::normalized::{BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{Net, NetlistModule, PortDirection};
use anyhow::{Result, anyhow};
use serde::Serialize;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use string_interner::{StringInterner, backend::StringBackend, symbol::SymbolU32};

pub(crate) const TIMING_COMPARISON_TOLERANCE: f64 = 1e-9;
pub(crate) const STA_VERIFICATION_TOLERANCE: f64 = 1e-7;

/// Transaction result; rejected and work-limited trials both restore the
/// incumbent, but only the latter stops the caller's current search phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrialOutcome {
    Accepted,
    #[default]
    Rejected,
    PropagationLimit,
}

/// Measures all four launch/capture classes independently with full `gv-stats`
/// STA, avoiding a separate signoff timing implementation in the optimizer.
pub fn measure_boundary_delays(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: StaOptions,
) -> Result<BoundaryDelays> {
    let prepared = PreparedStaLibrary::new(library)?;
    measure_boundary_delays_with_library(module, nets, interner, &prepared, options)
}

/// Measures launch classes using one reusable prepared library.
pub(crate) fn measure_boundary_delays_with_library(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    prepared: &PreparedStaLibrary<'_>,
    options: StaOptions,
) -> Result<BoundaryDelays> {
    let library = prepared.library;
    let by_name: HashMap<_, _> = library
        .cells
        .iter()
        .map(|cell| (cell.name.as_str(), cell))
        .collect();
    let mut registers = Vec::new();
    for (i, instance) in module.instances.iter().enumerate() {
        let name = interner
            .resolve(instance.type_name)
            .ok_or_else(|| anyhow!("unknown cell symbol"))?;
        let cell = by_name
            .get(name)
            .ok_or_else(|| anyhow!("unknown cell {name}"))?;
        if is_sequential_boundary_cell(cell) {
            registers.push(i);
        }
    }
    if registers.is_empty() {
        let report = prepared.analyze_combinational_max_arrival(module, nets, interner, options)?;
        return Ok(BoundaryDelays {
            input_to_output: report.worst_output_arrival,
            ..BoundaryDelays::default()
        });
    }
    let input = prepared.analyze_register_boundary_max_arrival(
        module,
        nets,
        interner,
        options,
        true,
        &[],
    )?;
    let register = prepared.analyze_register_boundary_max_arrival(
        module, nets, interner, options, false, &registers,
    )?;
    Ok(BoundaryDelays {
        input_to_output: input.worst_output_arrival,
        input_to_register: input.worst_register_input_arrival,
        register_to_output: register.worst_output_arrival,
        register_to_register: register.worst_register_input_arrival,
    })
}

/// Timing objectives, ordered PI->PO, PI->register, register->PO,
/// register->register.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct BoundaryDelays {
    pub input_to_output: f64,
    pub input_to_register: f64,
    pub register_to_output: f64,
    pub register_to_register: f64,
}

impl BoundaryDelays {
    pub(crate) fn values(self) -> [f64; 4] {
        [
            self.input_to_output,
            self.input_to_register,
            self.register_to_output,
            self.register_to_register,
        ]
    }

    pub(crate) fn within(self, limits: Self) -> bool {
        self.values()
            .into_iter()
            .zip(limits.values())
            .all(|(x, limit)| x <= limit + TIMING_COMPARISON_TOLERANCE)
    }
}

/// Stable total ordering for timing scores and search priorities.
#[derive(Clone, Copy, Debug)]
struct RankedIndex(f64, usize);
impl PartialEq for RankedIndex {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for RankedIndex {}
impl PartialOrd for RankedIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankedIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// Ordered endpoint arrivals with an incrementally maintained count of ties at
/// the maximum. A bounded endpoint sum alone cannot see progress when more
/// endpoints share the maximum than fit in that sum.
#[derive(Clone, Default)]
struct EndpointScores {
    values: BTreeSet<RankedIndex>,
    worst_count: usize,
}

impl EndpointScores {
    fn worst(&self) -> Option<f64> {
        self.values.last().map(|key| key.0)
    }

    /// Updates a single endpoint. Only removing the last current maximum
    /// requires counting the newly exposed maximum bucket; other updates are
    /// logarithmic and do not scan unaffected endpoints.
    fn replace(&mut self, endpoint: usize, old: Option<f64>, new: Option<f64>) {
        if old.map(f64::to_bits) == new.map(f64::to_bits) {
            return;
        }
        let previous = self.worst();
        if let Some(old) = old {
            let removed = self.values.remove(&RankedIndex(old, endpoint));
            debug_assert!(removed);
        }
        if let Some(new) = new {
            self.values.insert(RankedIndex(new, endpoint));
        }
        let current = self.worst();
        if current == previous {
            self.worst_count += usize::from(new.is_some() && new == current);
            self.worst_count -= usize::from(old.is_some() && old == previous);
        } else {
            self.worst_count = match (previous, current) {
                (_, None) => 0,
                (Some(before), Some(after)) if after < before => self
                    .values
                    .iter()
                    .rev()
                    .take_while(|key| key.0 == after)
                    .count(),
                _ => 1,
            };
        }
    }
}

/// Cached connectivity independent of the selected drive strength.
#[derive(Clone)]
struct Instance {
    name: String,
    pin_order: Vec<String>,
    pins: HashMap<String, Vec<PinBitSource>>,
    known: HashMap<String, bool>,
    outputs: Vec<usize>,
    sequential: bool,
}

/// A physical output or a complete register capture boundary.
#[derive(Clone, Copy)]
pub enum TimingEndpoint {
    Output(usize),
    Capture(usize),
}

/// Sparse undo journal. Rejected and budget-interrupted trials restore exactly.
#[derive(Default)]
struct Undo {
    bits: Vec<(usize, [Option<SignalTimingSet>; 2])>,
    endpoints: Vec<(usize, [Option<f64>; 2])>,
    counts: Vec<(usize, TimingQueryDiagnosticCounts)>,
}

/// Connected-cell dependencies are cyclic even when finer timing-pin
/// dependencies might admit full STA. Callers may choose a full-STA fallback.
#[derive(Debug)]
pub struct UnsupportedTopology;
impl std::fmt::Display for UnsupportedTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("unsupported cyclic cell dependencies")
    }
}
impl std::error::Error for UnsupportedTopology {}

/// Componentwise violation amounts for one pin, all zero when unconstrained.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ElectricalViolations {
    pub capacitance: f64,
    pub transition: f64,
    pub fanout: f64,
}

/// Read-only normalized instance bindings. Iterate pin_order for stable
/// ordering.
pub struct TimingInstanceView<'a> {
    pub name: &'a str,
    pub cell: usize,
    pub pins: &'a HashMap<String, Vec<BitSource>>,
    pub pin_order: &'a [String],
    pub outputs: &'a [usize],
    pub sequential: bool,
}

/// Owned timing graph with shared Liberty preparation and transactional edits.
pub struct IncrementalSta<'lib> {
    lib: Arc<PreparedStaLibrary<'lib>>,
    bit_names: Vec<String>,
    bit_sources: Vec<BitSource>,
    primary_inputs: BTreeSet<usize>,
    instance_names: HashSet<String>,
    timing_constants: Vec<Option<bool>>,
    coverage_values: Vec<Option<bool>>,
    instances: Vec<Instance>,
    cells: Vec<usize>,
    orders: Vec<Vec<usize>>,
    drivers: Vec<Option<usize>>,
    loads: Vec<Vec<(usize, String)>>,
    capacitances: Vec<EdgeLoadCapacitance>,
    external_load_counts: Vec<usize>,
    module_output_load: f64,
    wire_load: super::WireLoadModel,
    consumers: Vec<Vec<usize>>,
    positions: Vec<usize>,
    topological: Vec<usize>,
    traversal_dirty: bool,
    timing: [Vec<Option<SignalTimingSet>>; 2],
    endpoints: Vec<TimingEndpoint>,
    endpoint_values: Vec<[Option<f64>; 2]>,
    endpoints_by_bit: Vec<Vec<usize>>,
    capture_endpoint: Vec<Option<usize>>,
    scores: [EndpointScores; 4],
    electrical: Vec<Vec<ElectricalViolations>>,
    evaluation: TimingEvaluationContext<'lib>,
    query_counts: Vec<TimingQueryDiagnosticCounts>,
    query_totals: TimingQueryDiagnosticCounts,
    coverage_known: Vec<HashMap<String, bool>>,
    registered: bool,
    literal: SignalTimingSet,
    propagated: usize,
}

impl<'lib> IncrementalSta<'lib> {
    /// Retains prepared Liberty metadata for the final full timing check.
    pub(crate) fn into_prepared_library(self) -> Arc<PreparedStaLibrary<'lib>> {
        self.lib
    }

    /// Exposes the retained session to coverage-reuse regression tests.
    #[cfg(test)]
    pub(super) fn prepared_library(&self) -> &PreparedStaLibrary<'lib> {
        &self.lib
    }

    /// Prepares stable connectivity and initializes both independent launch
    /// classes.
    pub fn new(
        module: &NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        library: &'lib Library,
        options: StaOptions,
    ) -> Result<Self> {
        Self::new_mode(
            module,
            nets,
            interner,
            Arc::new(PreparedStaLibrary::new(library)?),
            options,
            true,
            None,
        )
    }

    /// Constructs an explicitly combinational session. Unused output arcs do
    /// not turn otherwise combinational instances into register boundaries.
    pub fn new_combinational(
        module: &NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        library: &'lib Library,
        options: StaOptions,
    ) -> Result<Self> {
        Self::new_mode(
            module,
            nets,
            interner,
            Arc::new(PreparedStaLibrary::new(library)?),
            options,
            false,
            None,
        )
    }

    /// Reuses the caller's prepared library and successful full combinational
    /// analysis of these exact inputs. Only internal clients that just
    /// performed that analysis use this entry point; public constructors
    /// validate normally. The caller also compares every initialized net
    /// with this report.
    pub(crate) fn from_validated_combinational(
        module: &NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        prepared: Arc<PreparedStaLibrary<'lib>>,
        options: StaOptions,
        report: &super::StaReport,
    ) -> Result<Self> {
        Self::new_mode(
            module,
            nets,
            interner,
            prepared,
            options,
            false,
            Some(BoundaryDelays {
                input_to_output: report.worst_output_arrival,
                ..BoundaryDelays::default()
            }),
        )
    }

    fn new_mode(
        module: &NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        lib: Arc<PreparedStaLibrary<'lib>>,
        options: StaOptions,
        register_boundaries: bool,
        validated: Option<BoundaryDelays>,
    ) -> Result<Self> {
        options.wire_load.validate()?;
        let library = lib.library;
        let normalized = NormalizedNetlistModule::new(module, nets, interner)?;
        let assign = analyze_assign_sources(&normalized, nets, interner)?;
        let resolved = resolve_timing_sources(&assign.bit_sources, &normalized, nets, interner)?;
        let n = normalized.bit_count();
        let mut instances = Vec::new();
        let mut cells = Vec::new();
        let mut drivers = vec![None; n];
        let mut loads = vec![Vec::new(); n];
        let mut orders = Vec::new();
        for (idx, inst) in normalized.instances.iter().enumerate() {
            let name = resolve_symbol(interner, inst.type_name, "cell")?;
            let cell_idx = lib
                .cell_index(&name)
                .ok_or_else(|| anyhow!("unknown cell {name}"))?;
            let cell = &library.cells[cell_idx];
            lib.validate_functional_semantics(cell_idx)
                .map_err(anyhow::Error::msg)?;
            let mut pins = HashMap::new();
            let mut pin_order = Vec::new();
            let mut known = HashMap::new();
            let mut outputs = Vec::new();
            for conn in &inst.connections {
                let name = resolve_symbol(interner, conn.port, "pin")?;
                let pin = lib
                    .pin(cell_idx, &name)
                    .ok_or_else(|| anyhow!("unknown pin {name}"))?;
                let sources: Vec<_> = conn
                    .bits
                    .iter()
                    .map(|source| {
                        if pin.direction == PinDirection::Input as i32 {
                            canonicalize_pin_bit_source(*source, &resolved)
                        } else {
                            *source
                        }
                    })
                    .collect();
                if sources.len() > 1 || pins.contains_key(&name) {
                    return Err(anyhow!(
                        "incremental STA requires unique scalar pin bindings"
                    ));
                }
                if let Some(source) = sources.first() {
                    match (pin.direction, *source) {
                        (d, BitSource::Bit(bit)) if d == PinDirection::Output as i32 => {
                            if drivers[bit].replace(idx).is_some() {
                                return Err(anyhow!("multiple drivers"));
                            }
                            outputs.push(bit);
                        }
                        (d, BitSource::Bit(bit)) if d == PinDirection::Input as i32 => {
                            loads[bit].push((idx, name.clone()))
                        }
                        (_, BitSource::Literal(value)) => {
                            known.insert(name.clone(), value);
                        }
                        _ => { /* The full validation below rejects unsupported bindings. */ }
                    }
                }
                pin_order.push(name.clone());
                pins.insert(name, sources);
            }
            let sequential = register_boundaries && is_sequential_boundary_cell(cell);
            let order = if sequential {
                cell.pins
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        p.direction == PinDirection::Output as i32
                            && pins.contains_key(library.resolve_string(&p.name))
                    })
                    .map(|(i, _)| i)
                    .collect()
            } else {
                combinational_output_pin_evaluation_order(&lib, cell_idx, &pins)?
            };
            cells.push(cell_idx);
            orders.push(order);
            instances.push(Instance {
                name: resolve_symbol(interner, inst.instance_name, "instance name")?,
                pin_order,
                pins,
                known,
                outputs,
                sequential,
            });
        }
        let registered = instances.iter().any(|inst| inst.sequential);
        // Full STA is the validation authority, including undriven nets,
        // aliases, conditional arcs and unsupported assignments.
        // Preparation is one-time.
        let reference = if let Some(validated) = validated {
            validated
        } else if register_boundaries {
            measure_boundary_delays_with_library(module, nets, interner, &lib, options)?
        } else {
            BoundaryDelays {
                input_to_output: lib
                    .analyze_combinational_max_arrival(module, nets, interner, options)?
                    .worst_output_arrival,
                input_to_register: 0.0,
                register_to_output: 0.0,
                register_to_register: 0.0,
            }
        };
        let mut constants = vec![None; n];
        for (i, inst) in instances.iter().enumerate() {
            for (name, sources) in &inst.pins {
                let pin = lib.pin(cells[i], name).expect("validated pin");
                if pin.direction == PinDirection::Output as i32
                    && pin.timing_arcs.is_empty()
                    && let [BitSource::Bit(bit)] = sources.as_slice()
                {
                    constants[*bit] = lib.timing_constant(
                        cells[i],
                        lib.pin_index(cells[i], name).expect("validated pin"),
                    )?;
                }
            }
        }
        let mut consumers = vec![Vec::new(); n];
        let mut successors = vec![BTreeSet::new(); instances.len()];
        let mut indegree = vec![0; instances.len()];
        for (i, inst) in instances.iter_mut().enumerate() {
            for (name, sources) in &inst.pins {
                let pin = lib.pin(cells[i], name).expect("validated pin");
                if pin.direction != PinDirection::Input as i32 {
                    continue;
                }
                if let [BitSource::Bit(bit)] = sources.as_slice() {
                    if let Some(value) = constants[*bit] {
                        inst.known.insert(name.clone(), value);
                    }
                    // Clock pins still impose electrical limits. Only a
                    // sequential cell cuts arrival propagation; a clock pin
                    // annotation on a combinational cell does not.
                    consumers[*bit].push(i);
                    if !inst.sequential
                        && let Some(driver) = drivers[*bit]
                        && driver != i
                        && successors[driver].insert(i)
                    {
                        indegree[i] += 1;
                    }
                }
            }
        }
        let mut ready: BTreeSet<_> = indegree
            .iter()
            .enumerate()
            .filter(|(_, d)| **d == 0)
            .map(|(i, _)| i)
            .collect();
        let mut topological = Vec::new();
        while let Some(i) = ready.pop_first() {
            topological.push(i);
            for j in &successors[i] {
                indegree[*j] -= 1;
                if indegree[*j] == 0 {
                    ready.insert(*j);
                }
            }
        }
        if topological.len() != instances.len() {
            return Err(UnsupportedTopology.into());
        }
        let mut positions = vec![0; instances.len()];
        let rank_gap = topology::RANK_GAP.min(usize::MAX / (instances.len() + 1));
        for (pos, i) in topological.iter().enumerate() {
            positions[*i] = (pos + 1) * rank_gap;
        }
        let mut endpoints = Vec::new();
        let mut endpoints_by_bit = vec![Vec::new(); n];
        let mut capture_endpoint = vec![None; instances.len()];
        let mut external_load_counts = vec![0; n];
        let mut seen_outputs = HashSet::new();
        let mut timing: [Vec<Option<SignalTimingSet>>; 2] = [vec![None; n], vec![None; n]];
        let source = SignalTiming {
            rise: EdgeTiming {
                arrival: 0.0,
                transition: options.primary_input_transition,
            },
            fall: EdgeTiming {
                arrival: 0.0,
                transition: options.primary_input_transition,
            },
        };
        let literal = SignalTimingSet::from_single(SignalTiming {
            rise: EdgeTiming {
                arrival: 0.0,
                transition: 0.0,
            },
            fall: EdgeTiming {
                arrival: 0.0,
                transition: 0.0,
            },
        });
        for port in &normalized.ports {
            for bit in &port.bits {
                if port.direction == PortDirection::Input {
                    timing[0][*bit] = Some(if registered {
                        SignalTimingSet::from_primary_input_launch(source)
                    } else {
                        SignalTimingSet::from_single(source)
                    });
                } else if seen_outputs.insert(*bit)
                    && let ResolvedTimingSource::Bit(root) = resolved[*bit]
                {
                    external_load_counts[root] += 1;
                    endpoints_by_bit[root].push(endpoints.len());
                    endpoints.push(TimingEndpoint::Output(root));
                }
            }
        }
        for (i, inst) in instances.iter().enumerate() {
            if inst.sequential {
                capture_endpoint[i] = Some(endpoints.len());
                for (name, sources) in &inst.pins {
                    let pin = lib.pin(cells[i], name).expect("validated pin");
                    // Q timing does not affect this register's setup checks.
                    // Resizing the register itself marks its capture endpoint
                    // explicitly in trial(), regardless of changed inputs.
                    if pin.direction != PinDirection::Input as i32 || pin.is_clocking_pin {
                        continue;
                    }
                    for source in sources {
                        if let BitSource::Bit(bit) = source {
                            endpoints_by_bit[*bit].push(endpoints.len());
                        }
                    }
                }
                endpoints.push(TimingEndpoint::Capture(i));
            }
        }
        let endpoint_values = vec![[None; 2]; endpoints.len()];
        let electrical = vec![Vec::new(); instances.len()];
        let query_counts = vec![TimingQueryDiagnosticCounts::default(); instances.len()];
        let mut sta = Self {
            lib,
            bit_names: (0..n)
                .map(|bit| normalized.render_bit(bit, nets, interner))
                .collect(),
            bit_sources: (0..n)
                .map(|bit| canonicalize_pin_bit_source(BitSource::Bit(bit), &resolved))
                .collect(),
            primary_inputs: normalized
                .ports
                .iter()
                .filter(|p| p.direction == PortDirection::Input)
                .flat_map(|p| p.bits.iter().copied())
                .collect(),
            instance_names: instances.iter().map(|inst| inst.name.clone()).collect(),
            timing_constants: constants,
            coverage_values: vec![None; n],
            instances,
            cells,
            orders,
            drivers,
            loads,
            capacitances: vec![EdgeLoadCapacitance::default(); n],
            external_load_counts,
            module_output_load: options.module_output_load,
            wire_load: options.wire_load,
            consumers,
            positions,
            topological,
            traversal_dirty: false,
            timing,
            endpoints,
            endpoint_values,
            endpoints_by_bit,
            capture_endpoint,
            scores: std::array::from_fn(|_| EndpointScores::default()),
            electrical,
            evaluation: TimingEvaluationContext::new(library),
            query_counts,
            query_totals: TimingQueryDiagnosticCounts::default(),
            coverage_known: Vec::new(),
            registered,
            literal,
            propagated: 0,
        };
        for bit in 0..n {
            sta.capacitances[bit] = sta.compute_load(bit)?;
        }
        for pos in 0..sta.topological.len() {
            sta.recompute(sta.topological[pos])?;
        }
        sta.coverage_known = sta
            .coverage_constants(usize::MAX)
            .0
            .expect("unlimited coverage propagation");
        sta.refresh_coverage_values();
        for endpoint in 0..sta.endpoints.len() {
            let values = sta.compute_endpoint(endpoint)?;
            sta.set_endpoint(endpoint, values);
        }
        for i in 0..sta.instances.len() {
            sta.electrical[i] = sta.electrical_violations(i);
        }
        for (computed, expected) in sta.score().values().into_iter().zip(reference.values()) {
            if (computed - expected).abs() > STA_VERIFICATION_TOLERANCE {
                return Err(anyhow!(
                    "incremental/full STA preparation mismatch: {computed} versus {expected}"
                ));
            }
        }
        Ok(sta)
    }

    /// Computes the same coverage-only constants as full STA for the current
    /// cells.
    fn coverage_constants(&self, limit: usize) -> (Option<Vec<HashMap<String, bool>>>, usize) {
        // A full pass must visit every instance at least once. Avoid cloning
        // the connectivity when the remaining trial budget cannot cover it.
        if limit < self.instances.len() {
            return (None, 0);
        }
        let pin_sources: Vec<_> = self.instances.iter().map(|inst| &inst.pins).collect();
        let bit_loads: Vec<Vec<NetEndpoint>> = self
            .loads
            .iter()
            .map(|loads| {
                loads
                    .iter()
                    .map(|(inst_idx, pin_name)| NetEndpoint {
                        inst_idx: *inst_idx,
                        pin_name: pin_name.clone(),
                    })
                    .collect()
            })
            .collect();
        CoverageConstantPropagation {
            lib: &self.lib,
            cell_indices: &self.cells,
            pin_sources: &pin_sources,
            output_pin_orders: &self.orders,
            bit_loads: &bit_loads,
        }
        .run_limited(limit)
    }

    /// Detects whether a replacement changes any connected inferred output.
    fn changes_coverage_constants(&mut self, i: usize, old: usize, new: usize) -> bool {
        let lib = &self.lib;
        let known = &self.coverage_known[i];
        let old_cell = &lib.library.cells[old];
        let old_supported = lib.supports_boolean_analysis(old);
        let new_supported = lib.supports_boolean_analysis(new);
        if !old_supported && !new_supported {
            return false;
        }
        // The resizer's dependency order ignores a direct self-edge. Such a
        // pin may contain a constant inferred from this very output, so it
        // cannot be used to prove the replacement has the same fixed point.
        let self_feedback = self.instances[i].pins.iter().any(|(name, sources)| {
            lib.pin(old, name)
                .is_some_and(|pin| pin.direction == PinDirection::Input as i32)
                && sources.iter().any(|source| {
                    matches!(source, PinBitSource::Bit(bit) if self.drivers[*bit] == Some(i))
                })
        });
        let value = |cell_idx: usize, pin_idx: usize, supported: bool| {
            if !supported {
                return None;
            }
            lib.constant_function(cell_idx, pin_idx)
                .and_then(|term| term.evaluate_partial(known))
        };
        for (old_pin_idx, old_pin) in old_cell.pins.iter().enumerate() {
            let name = lib.library.resolve_string(&old_pin.name);
            if old_pin.direction != PinDirection::Output as i32
                || !self.instances[i].pins.contains_key(name)
            {
                continue;
            }
            let Some(new_pin_idx) = lib.pin_index(new, name) else {
                return true;
            };
            let new_pin = &lib.library.cells[new].pins[new_pin_idx];
            if old_supported == new_supported
                && old_pin.function == new_pin.function
                && lib.output_has_unsupported_arcs(old, old_pin_idx)
                    == lib.output_has_unsupported_arcs(new, new_pin_idx)
            {
                continue;
            }
            if self_feedback {
                return true;
            }
            if value(old, old_pin_idx, old_supported) != value(new, new_pin_idx, new_supported) {
                return true;
            }
        }
        false
    }

    /// Checks the fixed-pin Resize contract for constants used by `when` arcs.
    /// Combinational Replace edits can instead update these bindings.
    fn changes_timing_constants(&self, i: usize, old: usize, new: usize) -> Result<bool> {
        let old_cell = &self.lib.library.cells[old];
        for (old_pin_idx, old_pin) in old_cell.pins.iter().enumerate() {
            let name = self.lib.library.resolve_string(&old_pin.name);
            if old_pin.direction != PinDirection::Output as i32
                || !self.instances[i].pins.contains_key(name)
            {
                continue;
            }
            let Some(new_pin_idx) = self.lib.pin_index(new, name) else {
                return Ok(true);
            };
            if self.lib.timing_constant(old, old_pin_idx)?
                != self.lib.timing_constant(new, new_pin_idx)?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Re-sums just one edited net to avoid accumulating subtraction roundoff.
    fn compute_load(&self, bit: usize) -> Result<EdgeLoadCapacitance> {
        let mut cap = EdgeLoadCapacitance::default();
        for (inst, name) in &self.loads[bit] {
            let pin = self
                .lib
                .pin(self.cells[*inst], name)
                .expect("compatible pin");
            let c = effective_input_capacitance_by_edge(pin, name)?;
            cap.rise += c.rise;
            cap.fall += c.fall;
        }
        // Match full STA's original sink order, then each physical output
        // port load, so trial/report arithmetic also agrees bit for bit.
        for _ in 0..self.external_load_counts[bit] {
            cap.rise += self.module_output_load;
            cap.fall += self.module_output_load;
        }
        if self.drivers[bit].is_some() {
            self.wire_load.add_to(
                &mut cap,
                self.loads[bit].len(),
                self.external_load_counts[bit] != 0,
            )?;
        }
        Ok(cap)
    }

    /// Updates the two independent launch classes through the shared STA
    /// kernel.
    fn recompute(&mut self, i: usize) -> Result<()> {
        self.evaluation.diagnostics = TimingQueryDiagnosticCounts::default();
        let inst = &self.instances[i];
        for mode in 0..if self.registered { 2 } else { 1 } {
            for bit in &inst.outputs {
                self.timing[mode][*bit] = None;
            }
            evaluate_instance_outputs(
                InstanceTimingInputs {
                    lib: &self.lib,
                    cell_idx: self.cells[i],
                    inst_pin_map: &inst.pins,
                    known_pin_values: &inst.known,
                    output_pin_order: &self.orders[i],
                    instance_name: &inst.name,
                    render_bit: &|bit| self.bit_names[bit].clone(),
                    register_boundaries: self.registered,
                    is_sequential: inst.sequential,
                    launch_register: mode == 1,
                },
                &self.capacitances,
                &mut self.timing[mode],
                &self.literal,
                &mut self.evaluation,
            )?;
        }
        self.set_query_counts(i, self.evaluation.diagnostics);
        Ok(())
    }

    /// Maintains exact query totals through the same per-instance journal used
    /// for timing, avoiding a full reduction on every candidate.
    fn set_query_counts(&mut self, i: usize, next: TimingQueryDiagnosticCounts) {
        let old = self.query_counts[i];
        self.query_totals.delay_slew_below_min_clamp_count =
            self.query_totals.delay_slew_below_min_clamp_count
                - old.delay_slew_below_min_clamp_count
                + next.delay_slew_below_min_clamp_count;
        self.query_totals
            .delay_slew_single_above_max_extrapolation_count = self
            .query_totals
            .delay_slew_single_above_max_extrapolation_count
            - old.delay_slew_single_above_max_extrapolation_count
            + next.delay_slew_single_above_max_extrapolation_count;
        self.query_totals.delay_slew_multiple_above_max_clamp_count =
            self.query_totals.delay_slew_multiple_above_max_clamp_count
                - old.delay_slew_multiple_above_max_clamp_count
                + next.delay_slew_multiple_above_max_clamp_count;
        self.query_totals.setup_below_min_clamp_count =
            self.query_totals.setup_below_min_clamp_count - old.setup_below_min_clamp_count
                + next.setup_below_min_clamp_count;
        self.query_totals.setup_above_max_clamp_count =
            self.query_totals.setup_above_max_clamp_count - old.setup_above_max_clamp_count
                + next.setup_above_max_clamp_count;
        self.query_counts[i] = next;
    }

    /// Evaluates one changed physical output or setup-adjusted capture
    /// boundary.
    fn compute_endpoint(&mut self, endpoint: usize) -> Result<[Option<f64>; 2]> {
        let mut values = [None; 2];
        for (mode, value) in values
            .iter_mut()
            .enumerate()
            .take(if self.registered { 2 } else { 1 })
        {
            *value = match self.endpoints[endpoint] {
                TimingEndpoint::Output(bit) => self.timing[mode][bit]
                    .as_ref()
                    .and_then(SignalTimingSet::worst_arrival_candidate)
                    .map(|t| t.timing.arrival),
                TimingEndpoint::Capture(i) => {
                    let mut worst: Option<f64> = None;
                    let cell = &self.lib.library.cells[self.cells[i]];
                    for pin in cell
                        .pins
                        .iter()
                        .filter(|p| p.direction == PinDirection::Input as i32 && !p.is_clocking_pin)
                    {
                        let name = self.lib.library.resolve_string(&pin.name);
                        let arcs: SmallVec<[&TimingArc; 4]> = pin
                            .timing_arcs
                            .iter()
                            .filter(|arc| StaTimingType::from_arc(arc, self.lib.library).is_setup())
                            .collect();
                        if arcs.is_empty() {
                            continue;
                        }
                        validate_constraint_tables_once(
                            self.lib.library,
                            &cell.name,
                            name,
                            &arcs,
                            &mut self.evaluation,
                        )?;
                        if let Some(sources) = self.instances[i].pins.get(name) {
                            for source in sources {
                                if let BitSource::Bit(bit) = source
                                    && let Some(timing) = &self.timing[mode][*bit]
                                    && let Some(capture) = evaluate_register_setup_capture_arrival(
                                        self.lib.library,
                                        &arcs,
                                        &self.instances[i].known,
                                        timing,
                                        &mut self.evaluation,
                                        name,
                                    )?
                                {
                                    worst = Some(
                                        worst.map_or(capture.arrival, |v| v.max(capture.arrival)),
                                    );
                                }
                            }
                        }
                    }
                    worst
                }
            };
        }
        Ok(values)
    }

    /// Maintains ordered endpoint scores without rescanning unaffected
    /// endpoints.
    fn set_endpoint(&mut self, endpoint: usize, values: [Option<f64>; 2]) {
        let capture = usize::from(matches!(
            self.endpoints[endpoint],
            TimingEndpoint::Capture(_)
        ));
        for (mode, value) in values.iter().enumerate() {
            let scores = &mut self.scores[mode * 2 + capture];
            scores.replace(endpoint, self.endpoint_values[endpoint][mode], *value);
        }
        self.endpoint_values[endpoint] = values;
    }

    pub fn score(&self) -> BoundaryDelays {
        let v: [f64; 4] = std::array::from_fn(|i| self.scores[i].worst().unwrap_or(0.0));
        BoundaryDelays {
            input_to_output: v[0],
            input_to_register: v[1],
            register_to_output: v[2],
            register_to_register: v[3],
        }
    }

    pub fn is_register(&self, i: usize) -> bool {
        self.instances[i].sequential
    }

    /// Cumulative timing and Boolean-coverage cell visits, including rejected
    /// and interrupted trials.
    pub fn propagated_instances(&self) -> usize {
        self.propagated
    }

    /// Stable current cell selections, indexed by instance ID.
    pub fn cell_indices(&self) -> &[usize] {
        &self.cells
    }

    /// Read-only normalized bindings for one stable instance ID.
    pub fn instance(&self, i: usize) -> Option<TimingInstanceView<'_>> {
        let inst = self.instances.get(i)?;
        Some(TimingInstanceView {
            name: &inst.name,
            cell: self.cells[i],
            pins: &inst.pins,
            pin_order: &inst.pin_order,
            outputs: &inst.outputs,
            sequential: inst.sequential,
        })
    }

    /// Driver of a canonical bit, if driven by an instance.
    pub fn driver_for_bit(&self, bit: usize) -> Option<usize> {
        self.drivers.get(bit).copied().flatten()
    }

    /// Input-pin loads in deterministic capacitance-summation order.
    pub fn loads_for_bit(&self, bit: usize) -> &[(usize, String)] {
        &self.loads[bit]
    }

    /// Number of physical module outputs loading a canonical bit.
    pub fn output_load_count(&self, bit: usize) -> usize {
        self.external_load_counts[bit]
    }

    /// Whether a stable instance name is already present.
    pub fn contains_instance_name(&self, name: &str) -> bool {
        self.instance_names.contains(name)
    }

    /// Library pin index for a current instance binding.
    pub fn pin_index(&self, instance: usize, name: &str) -> Option<usize> {
        self.cells
            .get(instance)
            .and_then(|&cell| self.lib.pin_index(cell, name))
    }

    /// Output pin driving a canonical bit from this instance.
    pub fn output_pin_index(&self, instance: usize, bit: usize) -> Option<usize> {
        let inst = self.instances.get(instance)?;
        let outputs = &self.orders[instance];
        // Ordinary scalar cells have one output. Reuse its prepared pin index
        // instead of resolving every input-pin name during metric reduction.
        if inst.outputs.as_slice() == [bit] && outputs.len() == 1 {
            return Some(outputs[0]);
        }
        outputs.iter().copied().find(|&pin| {
            let name = self
                .lib
                .library
                .resolve_string(&self.lib.library.cells[self.cells[instance]].pins[pin].name);
            inst.pins
                .get(name)
                .is_some_and(|sources| sources.as_slice() == [BitSource::Bit(bit)])
        })
    }

    /// Number of independently propagated launch classes (PI, then registers).
    pub fn launch_class_count(&self) -> usize {
        if self.registered { 2 } else { 1 }
    }

    /// Current instances in dependency order, including outstanding edits.
    /// During a topology trial the order is materialized on demand; otherwise
    /// this borrows the committed traversal without copying.
    pub fn topological_instances(&self) -> std::borrow::Cow<'_, [usize]> {
        if self.traversal_dirty {
            let mut order: Vec<_> = (0..self.instance_count()).collect();
            order.sort_by_key(|&i| (self.positions[i], i));
            std::borrow::Cow::Owned(order)
        } else {
            std::borrow::Cow::Borrowed(&self.topological)
        }
    }

    /// Ordered endpoint arrivals for PI->PO, PI->register, register->PO or
    /// register->register (indices 0 through 3).
    pub fn endpoint_arrivals(
        &self,
        boundary: usize,
    ) -> impl DoubleEndedIterator<Item = (TimingEndpoint, f64)> + '_ {
        self.scores[boundary]
            .values
            .iter()
            .map(|key| (self.endpoints[key.1], key.0))
    }

    /// Number of endpoints tied at this boundary's maximum.
    pub fn endpoint_maximum_count(&self, boundary: usize) -> usize {
        self.scores[boundary].worst_count
    }

    /// Endpoints and their separate PI/register launch arrivals.
    pub fn endpoint_values(&self) -> impl Iterator<Item = (TimingEndpoint, [Option<f64>; 2])> + '_ {
        self.endpoints
            .iter()
            .copied()
            .zip(self.endpoint_values.iter().copied())
    }

    /// Data-input driver arrivals in a launch class; excludes register clock
    /// pins while retaining clock-marked combinational dependencies.
    pub fn data_input_arrivals(&self, i: usize, mode: usize) -> SmallVec<[(usize, f64); 4]> {
        let mut inputs = SmallVec::new();
        for (name, sources) in &self.instances[i].pins {
            let pin = self.lib.pin(self.cells[i], name).expect("validated pin");
            if pin.direction != PinDirection::Input as i32
                || (self.instances[i].sequential && pin.is_clocking_pin)
            {
                continue;
            }
            for source in sources {
                if let BitSource::Bit(bit) = source
                    && let Some(driver) = self.drivers[*bit]
                    && let Some(arrival) = self.timing[mode][*bit]
                        .as_ref()
                        .and_then(SignalTimingSet::worst_arrival_candidate)
                {
                    inputs.push((driver, arrival.timing.arrival));
                }
            }
        }
        inputs
    }

    /// Incumbent electrical violations, ordered by pin name. During a trial
    /// this remains the pre-trial baseline; unconstrained cells have no
    /// entries.
    pub fn cached_electrical_violations(&self, i: usize) -> &[ElectricalViolations] {
        &self.electrical[i]
    }

    /// Reports current componentwise violations in stable pin-name order.
    /// Clients choose their acceptance policy. Unconstrained cells have no
    /// entries; otherwise each cell pin has one entry, including zero amounts.
    pub fn electrical_violations(&self, i: usize) -> Vec<ElectricalViolations> {
        let cell = &self.lib.library.cells[self.cells[i]];
        let pins = self.lib.electrical_pin_order(self.cells[i]);
        let mut values = Vec::with_capacity(pins.len());
        for &pin_index in pins {
            let pin = &cell.pins[pin_index];
            if pin.max_capacitance.is_none()
                && pin.max_transition.is_none()
                && pin.max_fanout.is_none()
            {
                values.push(ElectricalViolations::default());
                continue;
            }
            let sources = self.instances[i]
                .pins
                .get(self.lib.library.resolve_string(&pin.name));
            let bit = sources.and_then(|s| s.first()).and_then(|s| {
                if let BitSource::Bit(bit) = s {
                    Some(*bit)
                } else {
                    None
                }
            });
            let mut cap = 0.0_f64;
            let mut transition = 0.0_f64;
            let mut fanout = 0.0_f64;
            if let Some(bit) = bit {
                if pin.direction == PinDirection::Output as i32 {
                    cap = self.capacitances[bit].rise.max(self.capacitances[bit].fall);
                    if pin.max_fanout.is_some() {
                        fanout = self.loads[bit]
                            .iter()
                            .map(|(load, name)| {
                                self.lib
                                    .pin(self.cells[*load], name)
                                    .expect("validated pin")
                                    .fanout_load
                                    .unwrap_or(1.0)
                            })
                            .sum();
                    }
                }
                if pin.max_transition.is_some() {
                    for timing in &self.timing {
                        if let Some(t) = &timing[bit] {
                            for edge in t.rise.iter().chain(t.fall.iter()) {
                                transition = transition.max(edge.timing.transition);
                            }
                        }
                    }
                }
            }
            values.push(ElectricalViolations {
                capacitance: pin
                    .max_capacitance
                    .map_or(0.0, |limit| (cap - limit).max(0.0)),
                transition: pin
                    .max_transition
                    .map_or(0.0, |limit| (transition - limit).max(0.0)),
                fanout: pin
                    .max_fanout
                    .map_or(0.0, |limit| (fanout - limit).max(0.0)),
            });
        }
        values
    }

    /// Propagates a seeded cone through the same kernels for both edit APIs.
    fn propagate_edits(
        &mut self,
        mut queue: BTreeSet<(usize, usize)>,
        undo: &mut Undo,
        affected: &mut BTreeSet<usize>,
        endpoints: &mut BTreeSet<usize>,
        budget: usize,
        limits: Option<BoundaryDelays>,
    ) -> Result<Option<TrialOutcome>> {
        while let Some((_, inst)) = queue.pop_first() {
            if self.propagated >= budget {
                return Ok(Some(TrialOutcome::PropagationLimit));
            }
            self.propagated += 1;
            let start = undo.bits.len();
            for bit in &self.instances[inst].outputs {
                undo.bits.push((
                    *bit,
                    [self.timing[0][*bit].clone(), self.timing[1][*bit].clone()],
                ));
            }
            undo.counts.push((inst, self.query_counts[inst]));
            self.recompute(inst)?;
            affected.insert(inst);
            for (bit, old) in &undo.bits[start..] {
                if old[0] == self.timing[0][*bit] && old[1] == self.timing[1][*bit] {
                    continue;
                }
                if let Some(limits) = limits
                    && self.endpoints_by_bit[*bit].iter().any(|endpoint| {
                        matches!(self.endpoints[*endpoint], TimingEndpoint::Output(_))
                    })
                {
                    // Topological propagation has settled this output:
                    // later work cannot repair its timing. Capture setup
                    // constraints can depend on other pending inputs and
                    // remain part of the complete endpoint update below.
                    for (mode, timing) in self.timing.iter().enumerate() {
                        if let Some(arrival) = timing[*bit]
                            .as_ref()
                            .and_then(SignalTimingSet::worst_arrival_candidate)
                            && arrival.timing.arrival
                                > limits.values()[mode * 2] + TIMING_COMPARISON_TOLERANCE
                        {
                            return Ok(Some(TrialOutcome::Rejected));
                        }
                    }
                }
                endpoints.extend(self.endpoints_by_bit[*bit].iter().copied());
                for consumer in &self.consumers[*bit] {
                    affected.insert(*consumer);
                    if !self.instances[*consumer].sequential {
                        queue.insert((self.positions[*consumer], *consumer));
                    }
                }
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn endpoint_maximum_counts_match_rescanning_through_replacement_and_rollback() {
        let mut scores = EndpointScores::default();
        let mut values = [None; 64];
        let candidates = [
            None,
            Some(-2.0),
            Some(-0.0),
            Some(0.0),
            Some(1.0),
            Some(2.0),
        ];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc017);
        for _ in 0..10_000 {
            let endpoint = rng.gen_range(0..values.len());
            let old = values[endpoint];
            let new = candidates[rng.gen_range(0..candidates.len())];
            scores.replace(endpoint, old, new);
            values[endpoint] = new;
            let expected = values.iter().copied().flatten().max_by(f64::total_cmp);
            assert_eq!(scores.worst().map(f64::to_bits), expected.map(f64::to_bits));
            assert_eq!(
                scores.worst_count,
                values
                    .iter()
                    .filter(|v| v.is_some() && **v == expected)
                    .count()
            );
            if rng.gen_bool(0.5) {
                scores.replace(endpoint, new, old);
                values[endpoint] = old;
                let expected = values.iter().copied().flatten().max_by(f64::total_cmp);
                assert_eq!(scores.worst().map(f64::to_bits), expected.map(f64::to_bits));
                assert_eq!(
                    scores.worst_count,
                    values
                        .iter()
                        .filter(|v| v.is_some() && **v == expected)
                        .count()
                );
            }
        }
        for (endpoint, old) in values.into_iter().enumerate() {
            scores.replace(endpoint, old, None);
        }
        assert_eq!(scores.worst(), None);
        assert_eq!(scores.worst_count, 0);
    }
}

mod topology;
pub use topology::{TimingEdit, TimingInstance, TimingMutation, TrialOptions, TrialResult};
