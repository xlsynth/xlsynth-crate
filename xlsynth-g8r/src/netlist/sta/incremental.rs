// SPDX-License-Identifier: Apache-2.0

//! Transactional, fixed-topology timing for cell substitution. Full and
//! incremental analyses share the output, Liberty-table and setup kernels.

use super::{
    EdgeLoadCapacitance, EdgeTiming, InstanceTimingInputs, PinBitSource, ResolvedTimingSource,
    SignalTiming, SignalTimingSet, StaLibraryIndex, StaOptions, StaTimingType,
    TimingEvaluationContext, analyze_assign_sources, analyze_combinational_max_arrival,
    analyze_register_boundary_max_arrival, canonicalize_pin_bit_source,
    combinational_output_pin_evaluation_order, constant_output_function_value,
    effective_input_capacitance_by_edge, evaluate_instance_outputs,
    evaluate_register_setup_capture_arrival, is_sequential_boundary_cell, resolve_symbol,
    resolve_timing_sources, validate_constraint_tables_once,
};
use crate::liberty::Library;
use crate::liberty_model::{PinDirection, TimingTable};
use crate::netlist::normalized::{BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{Net, NetlistModule, PortDirection};
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use string_interner::{StringInterner, backend::StringBackend, symbol::SymbolU32};

/// Measures all four launch/capture classes independently with full `gv-stats`
/// STA, avoiding a separate signoff timing implementation in the optimizer.
pub fn measure_boundary_delays(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: StaOptions,
) -> Result<BoundaryDelays> {
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
        let report = analyze_combinational_max_arrival(module, nets, interner, library, options)?;
        return Ok(BoundaryDelays {
            input_to_output: report.worst_output_arrival,
            ..BoundaryDelays::default()
        });
    }
    let input =
        analyze_register_boundary_max_arrival(module, nets, interner, library, options, true, &[])?;
    let register = analyze_register_boundary_max_arrival(
        module, nets, interner, library, options, false, &registers,
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
            .all(|(x, limit)| x <= limit + 1e-9)
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
#[derive(Default)]
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

/// Lexicographic progress below the worst delay: first reduce the number of
/// equally critical endpoints, then improve a bounded near-critical sum.
#[derive(Clone, Copy)]
pub(crate) struct TimingTieBreak {
    worst_count: usize,
    near_sum: f64,
}

impl TimingTieBreak {
    pub fn improves(self, old: Self) -> bool {
        self.worst_count < old.worst_count
            || (self.worst_count == old.worst_count && self.near_sum < old.near_sum - 1e-9)
    }

    /// Returns a dimensionless progress ratio for area-efficiency ranking.
    /// A nonpositive baseline cannot justify paying additional area.
    pub fn relative_to(self, old: Self) -> f64 {
        if self.worst_count < old.worst_count {
            self.worst_count as f64 / old.worst_count as f64
        } else if old.near_sum > 0.0 {
            self.near_sum / old.near_sum
        } else {
            1.0
        }
    }
}

/// Cached connectivity independent of the selected drive strength.
struct Instance {
    pins: HashMap<String, Vec<PinBitSource>>,
    known: HashMap<String, bool>,
    outputs: Vec<usize>,
    sequential: bool,
}

/// A physical output or a complete register capture boundary.
#[derive(Clone, Copy)]
enum Endpoint {
    Output(usize),
    Capture(usize),
}

/// Sparse undo journal. Rejected and budget-interrupted trials restore exactly.
#[derive(Default)]
struct Undo {
    bits: Vec<(usize, [Option<SignalTimingSet>; 2])>,
    loads: Vec<(usize, EdgeLoadCapacitance)>,
    endpoints: Vec<(usize, [Option<f64>; 2])>,
}

/// Mutable timing state; the borrowed module's connectivity cannot change.
pub(crate) struct IncrementalSta<'a> {
    lib: StaLibraryIndex<'a>,
    normalized: NormalizedNetlistModule<'a>,
    nets: &'a [Net],
    interner: &'a StringInterner<StringBackend<SymbolU32>>,
    instances: Vec<Instance>,
    pub cells: Vec<usize>,
    orders: Vec<Vec<usize>>,
    drivers: Vec<Option<usize>>,
    loads: Vec<Vec<(usize, String)>>,
    capacitances: Vec<EdgeLoadCapacitance>,
    external_loads: Vec<f64>,
    consumers: Vec<Vec<usize>>,
    positions: Vec<usize>,
    topological: Vec<usize>,
    timing: [Vec<Option<SignalTimingSet>>; 2],
    endpoints: Vec<Endpoint>,
    endpoint_values: Vec<[Option<f64>; 2]>,
    endpoints_by_bit: Vec<Vec<usize>>,
    capture_endpoint: Vec<Option<usize>>,
    scores: [EndpointScores; 4],
    electrical: Vec<Vec<f64>>,
    validated: HashSet<*const TimingTable>,
    evaluation: TimingEvaluationContext<'a>,
    registered: bool,
    literal: SignalTimingSet,
    pub propagated: usize,
    pub budget_exhausted: bool,
}

impl<'a> IncrementalSta<'a> {
    /// Prepares stable connectivity and initializes both independent launch
    /// classes.
    pub fn new(
        module: &'a NetlistModule,
        nets: &'a [Net],
        interner: &'a StringInterner<StringBackend<SymbolU32>>,
        library: &'a Library,
        options: StaOptions,
    ) -> Result<Self> {
        let lib = StaLibraryIndex::new(library)?;
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
            cell.validate_functional_semantics()
                .map_err(|e| anyhow!(e))?;
            let mut pins = HashMap::new();
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
                pins.insert(name, sources);
            }
            let sequential = is_sequential_boundary_cell(cell);
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
        let reference = measure_boundary_delays(module, nets, interner, library, options)?;
        let mut constants = vec![None; n];
        for (i, inst) in instances.iter().enumerate() {
            for (name, sources) in &inst.pins {
                let pin = lib.pin(cells[i], name).expect("validated pin");
                if pin.direction == PinDirection::Output as i32
                    && pin.timing_arcs.is_empty()
                    && let [BitSource::Bit(bit)] = sources.as_slice()
                {
                    constants[*bit] = constant_output_function_value(
                        library,
                        &library.cells[cells[i]].name,
                        pin,
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
                    if !pin.is_clocking_pin {
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
            return Err(anyhow!("unsupported cyclic cell dependencies"));
        }
        let mut positions = vec![0; instances.len()];
        for (pos, i) in topological.iter().enumerate() {
            positions[*i] = pos;
        }
        let mut endpoints = Vec::new();
        let mut endpoints_by_bit = vec![Vec::new(); n];
        let mut capture_endpoint = vec![None; instances.len()];
        let mut external_loads = vec![0.0; n];
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
                    external_loads[root] += options.module_output_load;
                    endpoints_by_bit[root].push(endpoints.len());
                    endpoints.push(Endpoint::Output(root));
                }
            }
        }
        for (i, inst) in instances.iter().enumerate() {
            if inst.sequential {
                capture_endpoint[i] = Some(endpoints.len());
                for (name, sources) in &inst.pins {
                    if lib
                        .pin(cells[i], name)
                        .expect("validated pin")
                        .is_clocking_pin
                    {
                        continue;
                    }
                    for source in sources {
                        if let BitSource::Bit(bit) = source {
                            endpoints_by_bit[*bit].push(endpoints.len());
                        }
                    }
                }
                endpoints.push(Endpoint::Capture(i));
            }
        }
        let endpoint_values = vec![[None; 2]; endpoints.len()];
        let electrical = vec![Vec::new(); instances.len()];
        let mut sta = Self {
            lib,
            normalized,
            nets,
            interner,
            instances,
            cells,
            orders,
            drivers,
            loads,
            capacitances: vec![EdgeLoadCapacitance::default(); n],
            external_loads,
            consumers,
            positions,
            topological,
            timing,
            endpoints,
            endpoint_values,
            endpoints_by_bit,
            capture_endpoint,
            scores: std::array::from_fn(|_| EndpointScores::default()),
            electrical,
            validated: HashSet::new(),
            evaluation: TimingEvaluationContext::new(library),
            registered,
            literal,
            propagated: 0,
            budget_exhausted: false,
        };
        for bit in 0..n {
            sta.capacitances[bit] = sta.compute_load(bit)?;
        }
        for pos in 0..sta.topological.len() {
            sta.recompute(sta.topological[pos])?;
        }
        for endpoint in 0..sta.endpoints.len() {
            let values = sta.compute_endpoint(endpoint)?;
            sta.set_endpoint(endpoint, values);
        }
        for i in 0..sta.instances.len() {
            sta.electrical[i] = sta.electrical_violations(i);
        }
        for (computed, expected) in sta.score().values().into_iter().zip(reference.values()) {
            if (computed - expected).abs() > 1e-7 {
                return Err(anyhow!(
                    "incremental/full STA preparation mismatch: {computed} versus {expected}"
                ));
            }
        }
        Ok(sta)
    }

    /// Re-sums just one edited net to avoid accumulating subtraction roundoff.
    fn compute_load(&self, bit: usize) -> Result<EdgeLoadCapacitance> {
        let mut cap = EdgeLoadCapacitance {
            rise: self.external_loads[bit],
            fall: self.external_loads[bit],
        };
        for (inst, name) in &self.loads[bit] {
            let pin = self
                .lib
                .pin(self.cells[*inst], name)
                .expect("compatible pin");
            let c = effective_input_capacitance_by_edge(pin, name)?;
            cap.rise += c.rise;
            cap.fall += c.fall;
        }
        Ok(cap)
    }

    /// Updates the two independent launch classes through the shared STA
    /// kernel.
    fn recompute(&mut self, i: usize) -> Result<()> {
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
                    normalized: &self.normalized,
                    nets: self.nets,
                    interner: self.interner,
                    inst_idx: i,
                    register_boundaries: self.registered,
                    is_sequential: inst.sequential,
                    launch_register: mode == 1,
                },
                &self.capacitances,
                &mut self.timing[mode],
                &self.literal,
                &mut self.validated,
                &mut self.evaluation,
            )?;
        }
        Ok(())
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
                Endpoint::Output(bit) => self.timing[mode][bit]
                    .as_ref()
                    .and_then(SignalTimingSet::worst_arrival_candidate)
                    .map(|t| t.timing.arrival),
                Endpoint::Capture(i) => {
                    let mut worst: Option<f64> = None;
                    let cell = &self.lib.library.cells[self.cells[i]];
                    for pin in cell
                        .pins
                        .iter()
                        .filter(|p| p.direction == PinDirection::Input as i32 && !p.is_clocking_pin)
                    {
                        let name = self.lib.library.resolve_string(&pin.name);
                        let arcs: Vec<_> = pin
                            .timing_arcs
                            .iter()
                            .filter(|arc| {
                                StaTimingType::from_raw(arc.timing_type_str(self.lib.library))
                                    .is_setup()
                            })
                            .collect();
                        if arcs.is_empty() {
                            continue;
                        }
                        validate_constraint_tables_once(
                            self.lib.library,
                            &cell.name,
                            name,
                            &arcs,
                            &mut self.validated,
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
        let capture = usize::from(matches!(self.endpoints[endpoint], Endpoint::Capture(_)));
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

    /// Prioritizes register-to-register timing when present, otherwise worst
    /// IO.
    pub fn objective(&self) -> usize {
        if self.scores[3].worst().is_some() {
            3
        } else {
            (0..4)
                .filter(|i| self.scores[*i].worst().is_some())
                .max_by(|a, b| self.score().values()[*a].total_cmp(&self.score().values()[*b]))
                .unwrap_or(0)
        }
    }

    /// Breaks delay ties without losing progress on large critical plateaus.
    pub fn secondary(&self, objective: usize) -> TimingTieBreak {
        let scores = &self.scores[objective];
        TimingTieBreak {
            worst_count: scores.worst_count,
            near_sum: scores.values.iter().rev().take(32).map(|key| key.0).sum(),
        }
    }

    pub fn is_register(&self, i: usize) -> bool {
        self.instances[i].sequential
    }

    /// Visits near-critical fanin by cumulative arrival deficit, once per
    /// round. This is only a search-order heuristic: asymmetric pin arcs
    /// and load/slew effects are accounted for by accurate candidate
    /// evaluation afterward.
    pub fn critical_instances(&self, objective: usize, limit: usize) -> Vec<usize> {
        let mut queue = BTreeSet::new();
        let worst = self.score().values()[objective];
        for key in self.scores[objective]
            .values
            .iter()
            .rev()
            .take(32)
            .filter(|k| k.0 >= worst - worst.abs() * 0.05)
        {
            match self.endpoints[key.1] {
                Endpoint::Output(bit) => {
                    if let Some(i) = self.drivers[bit] {
                        queue.insert(RankedIndex(worst - key.0, i));
                    }
                }
                Endpoint::Capture(i) => {
                    queue.insert(RankedIndex(worst - key.0, i));
                    queue.extend(self.fanin_priorities(i, objective / 2, worst - key.0));
                }
            }
        }
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        while let Some(RankedIndex(deficit, i)) = queue.pop_first() {
            if !seen.insert(i) {
                continue;
            }
            result.push(i);
            if result.len() >= limit {
                break;
            }
            // A launch register's D cone belongs to another timing path.
            // Capture-register fanin is seeded separately above.
            if !self.instances[i].sequential {
                queue.extend(self.fanin_priorities(i, objective / 2, deficit));
            }
        }
        result
    }

    /// Ranks data-input drivers by arrival deficit in one launch class.
    fn fanin_priorities(&self, i: usize, mode: usize, deficit: f64) -> Vec<RankedIndex> {
        let mut inputs = Vec::new();
        for (pin, sources) in &self.instances[i].pins {
            let pin = self.lib.pin(self.cells[i], pin).expect("validated pin");
            if pin.direction != PinDirection::Input as i32 || pin.is_clocking_pin {
                continue;
            }
            for source in sources {
                if let BitSource::Bit(bit) = source
                    && let Some(driver) = self.drivers[*bit]
                    && let Some(arrival) = self.timing[mode][*bit]
                        .as_ref()
                        .and_then(SignalTimingSet::worst_arrival_candidate)
                {
                    inputs.push((arrival.timing.arrival, driver));
                }
            }
        }
        let latest = inputs
            .iter()
            .map(|(arrival, _)| *arrival)
            .fold(0.0, f64::max);
        inputs
            .into_iter()
            .map(|(arrival, driver)| RankedIndex(deficit + (latest - arrival).max(0.0), driver))
            .collect()
    }

    /// Estimates per-cell headroom once per recovery round, across all timing
    /// boundaries. This is an ordering heuristic, not a timing constraint:
    /// exact trials still account for pin arcs, slew and upstream load changes.
    pub fn recovery_headroom(&self, limits: BoundaryDelays) -> Vec<f64> {
        let mut result = vec![f64::INFINITY; self.instances.len()];
        for mode in 0..if self.registered { 2 } else { 1 } {
            let mut margins = vec![f64::INFINITY; self.instances.len()];
            for (endpoint, values) in self.endpoints.iter().zip(&self.endpoint_values) {
                let Some(arrival) = values[mode] else {
                    continue;
                };
                let capture = usize::from(matches!(endpoint, Endpoint::Capture(_)));
                let margin = (limits.values()[mode * 2 + capture] - arrival).max(0.0);
                match *endpoint {
                    Endpoint::Output(bit) => {
                        if let Some(i) = self.drivers[bit] {
                            margins[i] = margins[i].min(margin);
                        }
                    }
                    Endpoint::Capture(i) => {
                        margins[i] = margins[i].min(margin);
                        for RankedIndex(value, driver) in self.fanin_priorities(i, mode, margin) {
                            margins[driver] = margins[driver].min(value);
                        }
                    }
                }
            }
            for i in self.topological.iter().rev().copied() {
                if !self.instances[i].sequential && margins[i].is_finite() {
                    for RankedIndex(value, driver) in self.fanin_priorities(i, mode, margins[i]) {
                        margins[driver] = margins[driver].min(value);
                    }
                }
            }
            for (result, margin) in result.iter_mut().zip(margins) {
                *result = result.min(margin);
            }
        }
        result
    }

    /// Reports componentwise violations in stable pin-name order. Existing
    /// violations may remain but cannot be worsened by an accepted
    /// substitution.
    fn electrical_violations(&self, i: usize) -> Vec<f64> {
        let cell = &self.lib.library.cells[self.cells[i]];
        let mut pins: Vec<_> = cell.pins.iter().collect();
        pins.sort_by_key(|p| self.lib.library.resolve_string(&p.name));
        let mut values = Vec::new();
        for pin in pins {
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
                for timing in &self.timing {
                    if let Some(t) = &timing[bit] {
                        for edge in t.rise.iter().chain(t.fall.iter()) {
                            transition = transition.max(edge.timing.transition);
                        }
                    }
                }
            }
            values.extend([
                pin.max_capacitance
                    .map_or(0.0, |limit| (cap - limit).max(0.0)),
                pin.max_transition
                    .map_or(0.0, |limit| (transition - limit).max(0.0)),
                pin.max_fanout
                    .map_or(0.0, |limit| (fanout - limit).max(0.0)),
            ]);
        }
        values
    }

    /// Trials one prevalidated equivalent cell; only the acceptance closure can
    /// commit it. Errors, rejection and budget exhaustion restore the
    /// incumbent.
    pub fn trial(
        &mut self,
        i: usize,
        cell: usize,
        budget: usize,
        accept: impl FnOnce(&Self) -> bool,
    ) -> Result<bool> {
        let old_cell = self.cells[i];
        let old_order = self.orders[i].clone();
        let mut undo = Undo::default();
        self.cells[i] = cell;
        let result = (|| -> Result<bool> {
            self.orders[i] = if self.instances[i].sequential {
                self.lib.library.cells[cell]
                    .pins
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        p.direction == PinDirection::Output as i32
                            && self.instances[i]
                                .pins
                                .contains_key(self.lib.library.resolve_string(&p.name))
                    })
                    .map(|(idx, _)| idx)
                    .collect()
            } else {
                combinational_output_pin_evaluation_order(&self.lib, cell, &self.instances[i].pins)?
            };
            let mut changed_loads = BTreeSet::new();
            for (name, sources) in &self.instances[i].pins {
                if self.lib.pin(cell, name).expect("compatible pin").direction
                    == PinDirection::Input as i32
                {
                    for source in sources {
                        if let BitSource::Bit(bit) = source {
                            changed_loads.insert(*bit);
                        }
                    }
                }
            }
            let mut queue = BTreeSet::from([(self.positions[i], i)]);
            let mut affected = BTreeSet::from([i]);
            for bit in changed_loads {
                // Fanout weights can change even when pin capacitance does not.
                if let Some(driver) = self.drivers[bit] {
                    affected.insert(driver);
                }
                let load = self.compute_load(bit)?;
                if load != self.capacitances[bit] {
                    undo.loads.push((bit, self.capacitances[bit]));
                    self.capacitances[bit] = load;
                    if let Some(driver) = self.drivers[bit] {
                        queue.insert((self.positions[driver], driver));
                        affected.insert(driver);
                    }
                }
            }
            let mut endpoints = BTreeSet::new();
            if let Some(endpoint) = self.capture_endpoint[i] {
                endpoints.insert(endpoint);
            }
            while let Some((_, inst)) = queue.pop_first() {
                if self.propagated >= budget {
                    self.budget_exhausted = true;
                    return Ok(false);
                }
                self.propagated += 1;
                let start = undo.bits.len();
                for bit in &self.instances[inst].outputs {
                    undo.bits.push((
                        *bit,
                        [self.timing[0][*bit].clone(), self.timing[1][*bit].clone()],
                    ));
                }
                self.recompute(inst)?;
                affected.insert(inst);
                for (bit, old) in &undo.bits[start..] {
                    if old[0] == self.timing[0][*bit] && old[1] == self.timing[1][*bit] {
                        continue;
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
            for endpoint in endpoints {
                undo.endpoints
                    .push((endpoint, self.endpoint_values[endpoint]));
                let value = self.compute_endpoint(endpoint)?;
                self.set_endpoint(endpoint, value);
            }
            let mut violations = Vec::new();
            for inst in affected {
                let new = self.electrical_violations(inst);
                if new
                    .iter()
                    .zip(&self.electrical[inst])
                    .any(|(a, b)| *a > *b + 1e-9)
                {
                    return Ok(false);
                }
                violations.push((inst, new));
            }
            if !accept(self) {
                return Ok(false);
            }
            for (inst, new) in violations {
                self.electrical[inst] = new;
            }
            Ok(true)
        })();
        if !matches!(result, Ok(true)) {
            self.cells[i] = old_cell;
            self.orders[i] = old_order;
            for (bit, values) in undo.bits.into_iter().rev() {
                [self.timing[0][bit], self.timing[1][bit]] = values;
            }
            for (bit, load) in undo.loads {
                self.capacitances[bit] = load;
            }
            for (endpoint, value) in undo.endpoints {
                self.set_endpoint(endpoint, value);
            }
        }
        result
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
