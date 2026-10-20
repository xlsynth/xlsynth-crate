// SPDX-License-Identifier: Apache-2.0

//! Bounded experimental post-mapping fanout optimization using cached
//! incremental STA.
//!
//! Scalar combinational GV only. Output aliases and literal inputs are
//! preserved; external output loads stay on the original driver. Every trial is
//! evaluated with propagated changes to upstream input loading. No
//! physical wire model or Boolean remapping is introduced.

use crate::liberty::cell_formula::parse_formula;
use crate::liberty_model::{Library, Pin, PinDirection};
use crate::netlist::io::resolve_symbol;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistAssignKind, NetlistInstance, PortDirection,
};
use crate::netlist::report::build_area_report;
use crate::netlist::sta::{StaOptions, StaReport, TimingQueryDiagnosticCounts};
use anyhow::{Result, anyhow, ensure};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

/// Search bounds and a dimensionless area penalty; times/loads use Liberty
/// units.
#[derive(Clone, Debug, Serialize)]
pub struct FanoutOptimizationOptions {
    pub full_sta: bool,
    pub verify_incremental: bool,
    pub max_rounds: usize,
    pub drivers_per_round: usize,
    pub fanout_trigger: usize,
    pub area_growth_limit: f64,
    pub area_weight: f64,
    pub primary_input_transition: f64,
    pub module_output_load: f64,
}
impl Default for FanoutOptimizationOptions {
    fn default() -> Self {
        Self {
            full_sta: false,
            verify_incremental: false,
            max_rounds: 8,
            drivers_per_round: 3,
            fanout_trigger: 16,
            area_growth_limit: 0.05,
            area_weight: 0.5,
            primary_input_transition: StaOptions::default().primary_input_transition,
            module_output_load: 1.0,
        }
    }
}

/// Library identity and exact Boolean signature, with lexical named input pins.
#[derive(Clone, Debug)]
struct CellChoice {
    index: usize,
    name: String,
    inputs: Vec<String>,
    output: String,
    truth: u64,
    area: f64,
    can_instantiate: bool,
}

/// One rewirable scalar input-pin load.
#[derive(Clone, Debug)]
struct Sink {
    instance: usize,
    connection: usize,
    capacitance: f64,
}

/// Electrical information for one uniquely driven scalar net.
#[derive(Clone, Debug)]
struct Driver {
    instance: usize,
    output_connection: usize,
    net: usize,
    sinks: Vec<Sink>,
    load: [f64; 2],
    output_ports: usize,
    fixed_output_load: f64,
    slew_bound: f64,
}

/// Canonical scalar connectivity; directed aliases resolve to their ultimate
/// source.
struct Graph {
    drivers: Vec<Driver>,
    roots: Vec<Option<usize>>,
    cells: Vec<usize>,
}

/// Timing, physical area, and electrical range diagnostics for a trial.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FanoutMetrics {
    pub area: f64,
    pub cell_count: usize,
    pub worst_delay: f64,
    pub max_slew: f64,
    pub max_load: f64,
    pub max_fanout: usize,
    pub output_load_above_table_max: usize,
    pub input_slew_above_table_max: usize,
    pub max_capacitance_violations: usize,
    pub max_transition_violations: usize,
    pub timing_queries: TimingQueryDiagnosticCounts,
}

/// Objective values for a measured trial; rich diagnostics are reported only
/// for the baseline and accepted edits, not speculatively maintained.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FanoutTrialMetrics {
    pub area: f64,
    pub cell_count: usize,
    pub worst_delay: f64,
}
impl From<&FanoutMetrics> for FanoutTrialMetrics {
    fn from(m: &FanoutMetrics) -> Self {
        Self {
            area: m.area,
            cell_count: m.cell_count,
            worst_delay: m.worst_delay,
        }
    }
}

/// A bounded edit; buffer levels count added stages from the original driver.
#[derive(Clone, Debug, Serialize)]
pub enum FanoutEdit {
    Upsize {
        cell: String,
    },
    Buffer {
        cell: String,
        groups: usize,
        levels: usize,
    },
    Replicate {
        groups: usize,
    },
}

/// One tried edit, including rejected upstream-loading and area-budget
/// alternatives.
#[derive(Clone, Debug, Serialize)]
pub struct FanoutTrial {
    pub round: usize,
    pub instance: String,
    pub kind: FanoutEdit,
    pub metrics: Option<FanoutTrialMetrics>,
    pub score: Option<f64>,
    pub reason: String,
}

/// One selected optimization and the signal context used to make the decision.
#[derive(Clone, Debug, Serialize)]
pub struct FanoutDecision {
    pub round: usize,
    pub instance: String,
    pub kind: FanoutEdit,
    pub input_timing: Vec<(String, [f64; 4])>,
    pub sink_count: usize,
    pub output_ports: usize,
    pub old_load: [f64; 2],
    pub before: FanoutMetrics,
    pub after: FanoutMetrics,
}

/// Deterministic decisions and trial outcomes, plus explicitly nondeterministic
/// runtime.
#[derive(Clone, Debug, Serialize)]
pub struct FanoutOptimizationReport {
    pub incremental: Option<IncrementalStats>,
    pub incremental_fallback_reason: Option<String>,
    pub options: FanoutOptimizationOptions,
    pub before: FanoutMetrics,
    pub after: FanoutMetrics,
    pub decisions: Vec<FanoutDecision>,
    pub trials: Vec<FanoutTrial>,
    pub sta_evaluations: usize,
    pub sta_seconds: f64,
    pub optimization_seconds: f64,
}

/// Computes edge-specific input capacitance with the same fallback as
/// repository STA.
fn pin_cap(pin: &Pin) -> [f64; 2] {
    [
        pin.rise_capacitance
            .or(pin.capacitance)
            .or(pin.fall_capacitance)
            .unwrap_or(0.0),
        pin.fall_capacitance
            .or(pin.capacitance)
            .or(pin.rise_capacitance)
            .unwrap_or(0.0),
    ]
}

/// Compiles exact signatures without assuming that cell names imply
/// equivalence.
#[cfg(test)]
fn choices(library: &Library) -> Result<Vec<CellChoice>> {
    choices_prepared(
        library,
        &crate::netlist::sta::PreparedStaLibrary::new(library)?,
    )
}

/// Screens cells using the timing preparation retained by the optimization.
fn choices_prepared(
    library: &Library,
    timing: &crate::netlist::sta::PreparedStaLibrary<'_>,
) -> Result<Vec<CellChoice>> {
    let mut out = Vec::new();
    for (index, cell) in library.cells.iter().enumerate() {
        if cell.dont_use == Some(true)
            || !cell.sequential.is_empty()
            || cell.clock_gate.is_some()
            || cell.validate_functional_semantics().is_err()
            || cell.pins.iter().any(|p| p.is_clocking_pin)
        {
            continue;
        }
        let outputs: Vec<_> = cell
            .pins
            .iter()
            .filter(|p| p.direction == PinDirection::Output as i32)
            .collect();
        let mut inputs: Vec<_> = cell
            .pins
            .iter()
            .filter(|p| p.direction == PinDirection::Input as i32)
            .map(|p| library.resolve_string(&p.name).to_string())
            .collect();
        inputs.sort();
        if outputs.len() != 1 || inputs.is_empty() || inputs.len() > 5 {
            continue;
        }
        let output = outputs[0];
        let Ok(formula) = parse_formula(library.resolve_string(&output.function)) else {
            continue;
        };
        if formula.inputs().is_empty() {
            continue;
        }
        if !cell.area.is_finite() || cell.area <= 0.0 {
            continue;
        }
        if timing
            .validate_independent_combinational_cell(index)
            .is_err()
        {
            continue;
        }
        let mut truth = 0u64;
        let mut bound = true;
        for assignment in 0..(1usize << inputs.len()) {
            let bindings: HashMap<_, _> = inputs
                .iter()
                .enumerate()
                .map(|(i, p)| (p.clone(), assignment & (1 << i) != 0))
                .collect();
            match formula.evaluate_partial(&bindings) {
                Some(true) => truth |= 1 << assignment,
                Some(false) => {} // False assignments leave this truth-table bit clear.
                None => {
                    bound = false;
                    break;
                }
            }
        }
        if !bound {
            continue;
        }
        // Constant functions and cells without a complete timing interface are
        // not replacement/replication candidates; the existing netlist stays
        // intact.
        if truth == 0 || truth == (1u64 << (1usize << inputs.len())) - 1 {
            continue;
        }
        let covered: BTreeSet<_> = output
            .timing_arcs
            .iter()
            .flat_map(|a| library.resolve_string(&a.related_pin).split_whitespace())
            .collect();
        if inputs.iter().any(|p| !covered.contains(p.as_str())) {
            continue;
        }
        out.push(CellChoice {
            index,
            name: cell.name.clone(),
            inputs,
            output: library.resolve_string(&output.name).into(),
            truth,
            area: cell.area,
            can_instantiate: cell.use_for_size_only != Some(true)
                && cell.validate_mapping_policy_coverage().is_ok(),
        });
    }
    out.sort_by(|a, b| a.area.total_cmp(&b.area).then(a.name.cmp(&b.name)));
    Ok(out)
}

/// Resolves an alias chain, rejecting cycles instead of silently losing loads.
fn root(index: usize, aliases: &BTreeMap<usize, Option<usize>>) -> Result<Option<usize>> {
    let mut current = index;
    let mut seen = BTreeSet::new();
    while let Some(next) = aliases.get(&current) {
        ensure!(seen.insert(current), "cyclic scalar alias");
        match next {
            Some(n) => current = *n,
            None => return Ok(None),
        }
    }
    Ok(Some(current))
}

/// Reads the scalar source of a connection; unsupported widths fail closed.
fn source(reference: &NetRef, roots: &[Option<usize>]) -> Result<Option<usize>> {
    match reference {
        NetRef::Simple(n) => Ok(roots[n.0]),
        NetRef::Literal(v) if v.get_bit_count() == 1 => Ok(None),
        NetRef::Unconnected => Ok(None),
        _ => Err(anyhow!(
            "fanout optimization supports scalar simple nets and known scalar literals only"
        )),
    }
}

/// Returns the tightest characterized maximum for an input-slew or output-load
/// axis.
fn table_bound(library: &Library, pin: &Pin, related: Option<&str>, load_axis: bool) -> f64 {
    let mut bound = f64::INFINITY;
    for arc in &pin.timing_arcs {
        if related.is_some_and(|p| {
            !library
                .resolve_string(&arc.related_pin)
                .split_whitespace()
                .any(|r| r == p)
        }) {
            continue;
        }
        for table in &arc.tables {
            let shape = library.timing_table_shape(table);
            let template = shape
                .template_id
                .checked_sub(1)
                .and_then(|i| library.lu_table_templates.get(i as usize));
            let explicit = library.timing_table_axes(table);
            for axis in 0..2 {
                let (name, fallback) = match (template, axis) {
                    (Some(t), 0) => (t.variable_1_str(library), t.index_1.as_slice()),
                    (Some(t), _) => (t.variable_2_str(library), t.index_2.as_slice()),
                    _ => ("", &[][..]),
                };
                let is_load =
                    name == "total_output_net_capacitance" || (name.is_empty() && axis == 1);
                let is_slew = matches!(name, "input_net_transition" | "input_transition_time")
                    || (name.is_empty() && axis == 0);
                if (load_axis && is_load) || (!load_axis && is_slew) {
                    let values = if explicit[axis].is_empty() {
                        fallback
                    } else {
                        explicit[axis]
                    };
                    if let Some(v) = values.last() {
                        bound = bound.min(*v);
                    }
                }
            }
        }
    }
    bound
}

/// Indexes each sink once, aggregating alias-connected external loads
/// separately.
fn graph(
    netlist: &MappedNetlist,
    library: &Library,
    options: &FanoutOptimizationOptions,
) -> Result<Graph> {
    ensure!(
        netlist.module.net_index_range == (0..netlist.nets.len()),
        "fanout optimization requires an owned single module"
    );
    ensure!(
        netlist.nets.iter().all(|n| n.width.is_none()),
        "fanout optimization requires scalar nets"
    );
    let byname: BTreeMap<_, _> = library
        .cells
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.as_str(), (i, c)))
        .collect();
    let mut cells = Vec::with_capacity(netlist.module.instances.len());
    let mut aliases = BTreeMap::new();
    for a in &netlist.module.assigns {
        ensure!(
            a.kind == NetlistAssignKind::Continuous,
            "tran aliases are outside optimization scope"
        );
        let NetRef::Simple(lhs) = a.lhs else {
            return Err(anyhow!("non-scalar alias lhs"));
        };
        let rhs = match &a.rhs {
            AssignExpr::Leaf(NetRef::Simple(n)) => Some(n.0),
            AssignExpr::Leaf(NetRef::Literal(v)) if v.get_bit_count() == 1 => None,
            _ => {
                return Err(anyhow!(
                    "fanout optimization requires scalar wire/literal aliases"
                ));
            }
        };
        ensure!(
            aliases.insert(lhs.0, rhs).is_none(),
            "multiple alias drivers"
        );
    }
    let roots = (0..netlist.nets.len())
        .map(|i| root(i, &aliases))
        .collect::<Result<Vec<_>>>()?;
    let mut sinks: BTreeMap<usize, Vec<Sink>> = BTreeMap::new();
    let mut loads: BTreeMap<usize, [f64; 2]> = BTreeMap::new();
    let mut slew_bounds: BTreeMap<usize, f64> = BTreeMap::new();
    let mut drivers = Vec::new();
    let mut seen_drivers = BTreeSet::new();
    for (i, inst) in netlist.module.instances.iter().enumerate() {
        let name = resolve_symbol(&netlist.interner, inst.type_name, "cell")?;
        let &(cell_index, cell) = byname
            .get(name.as_str())
            .ok_or_else(|| anyhow!("unknown cell {name}"))?;
        cells.push(cell_index);
        ensure!(
            cell.sequential.is_empty() && cell.clock_gate.is_none(),
            "fanout optimization requires combinational cells"
        );
        cell.validate_functional_semantics()
            .map_err(anyhow::Error::msg)?;
        for (j, (pn, r)) in inst.connections.iter().enumerate() {
            let pname = resolve_symbol(&netlist.interner, *pn, "pin")?;
            let p = cell
                .pins
                .iter()
                .find(|p| library.resolve_string(&p.name) == pname)
                .ok_or_else(|| anyhow!("unknown pin {pname}"))?;
            let Some(n) = source(r, &roots)? else {
                continue;
            };
            if p.direction == PinDirection::Input as i32 {
                let cap = pin_cap(p);
                sinks.entry(n).or_default().push(Sink {
                    instance: i,
                    connection: j,
                    capacitance: cap[0].max(cap[1]),
                });
                let load = loads.entry(n).or_insert([0.0; 2]);
                load[0] += cap[0];
                load[1] += cap[1];
                let bound = cell
                    .pins
                    .iter()
                    .filter(|p| p.direction == PinDirection::Output as i32)
                    .map(|out| table_bound(library, out, Some(&pname), false))
                    .fold(f64::INFINITY, f64::min);
                let entry = slew_bounds.entry(n).or_insert(f64::INFINITY);
                *entry = entry.min(bound);
            } else if p.direction == PinDirection::Output as i32 {
                ensure!(seen_drivers.insert(n), "multiple cell drivers");
                drivers.push(Driver {
                    instance: i,
                    output_connection: j,
                    net: n,
                    sinks: vec![],
                    load: [0.0; 2],
                    output_ports: 0,
                    fixed_output_load: 0.0,
                    slew_bound: f64::INFINITY,
                });
            }
        }
    }
    let mut net_by_name = HashMap::with_capacity(netlist.nets.len());
    for (i, n) in netlist.nets.iter().enumerate() {
        net_by_name.entry(n.name).or_insert(NetIndex(i));
    }
    let mut ports = BTreeMap::<usize, usize>::new();
    for p in &netlist.module.ports {
        ensure!(
            p.direction != PortDirection::Inout,
            "inout ports unsupported"
        );
        if p.direction == PortDirection::Output {
            let n = net_by_name
                .get(&p.name)
                .copied()
                .ok_or_else(|| anyhow!("missing output net"))?;
            if let Some(n) = roots[n.0] {
                *ports.entry(n).or_default() += 1;
                let load = loads.entry(n).or_insert([0.0; 2]);
                load[0] += options.module_output_load;
                load[1] += options.module_output_load;
            }
        }
    }
    for d in &mut drivers {
        d.sinks = sinks.remove(&d.net).unwrap_or_default();
        d.load = loads.get(&d.net).copied().unwrap_or_default();
        d.output_ports = ports.get(&d.net).copied().unwrap_or_default();
        d.fixed_output_load = d.output_ports as f64 * options.module_output_load;
        d.slew_bound = slew_bounds.get(&d.net).copied().unwrap_or(f64::INFINITY);
    }
    Ok(Graph {
        drivers,
        roots,
        cells,
    })
}

/// Best complete candidate retained while exploring one bounded search round.
struct TrialWinner {
    score: f64,
    metrics: FanoutTrialMetrics,
    plan: edit::EditPlan,
    driver: Driver,
    kind: FanoutEdit,
    trial: usize,
}

/// An unusable candidate timing context, distinct from an oracle or graph bug.
#[derive(Debug)]
struct CandidateTimingError(anyhow::Error);

impl std::fmt::Display for CandidateTimingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#}", self.0)
    }
}

impl std::error::Error for CandidateTimingError {}

/// Evaluates complete timing, so no candidate freezes changed upstream source
/// timing.
#[cfg(test)]
fn measure(
    netlist: &MappedNetlist,
    library: &Library,
    options: &FanoutOptimizationOptions,
) -> Result<(FanoutMetrics, StaReport, Graph)> {
    let prepared = crate::netlist::sta::PreparedStaLibrary::new(library)?;
    measure_prepared(netlist, library, options, &prepared)
}

fn measure_prepared(
    netlist: &MappedNetlist,
    library: &Library,
    options: &FanoutOptimizationOptions,
    prepared: &crate::netlist::sta::PreparedStaLibrary<'_>,
) -> Result<(FanoutMetrics, StaReport, Graph)> {
    let sta = prepared
        .analyze_combinational_max_arrival(
            &netlist.module,
            &netlist.nets,
            &netlist.interner,
            StaOptions {
                primary_input_transition: options.primary_input_transition,
                module_output_load: options.module_output_load,
            },
        )
        .map_err(CandidateTimingError)?;
    let g = graph(netlist, library, options)?;
    let m = report_metrics(netlist, library, &sta, &g)?;
    Ok((m, sta, g))
}

/// Computes rich electrical diagnostics at accepted search boundaries.
fn report_metrics(
    netlist: &MappedNetlist,
    library: &Library,
    sta: &StaReport,
    g: &Graph,
) -> Result<FanoutMetrics> {
    let area = build_area_report(&netlist.module, &netlist.interner, library)?;
    let mut m = FanoutMetrics {
        area: area.area,
        cell_count: area.cell_count,
        worst_delay: sta.worst_output_arrival,
        max_slew: 0.0,
        max_load: 0.0,
        max_fanout: 0,
        output_load_above_table_max: 0,
        input_slew_above_table_max: 0,
        max_capacitance_violations: 0,
        max_transition_violations: 0,
        timing_queries: sta.timing_query_diagnostic_counts,
    };
    for d in &g.drivers {
        let inst = &netlist.module.instances[d.instance];
        let name = resolve_symbol(&netlist.interner, inst.type_name, "cell")?;
        let cell = library
            .cells
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| anyhow!("unknown cell"))?;
        let pname = resolve_symbol(
            &netlist.interner,
            inst.connections[d.output_connection].0,
            "pin",
        )?;
        let pin = cell
            .pins
            .iter()
            .find(|p| library.resolve_string(&p.name) == pname)
            .ok_or_else(|| anyhow!("unknown pin"))?;
        let timing = sta
            .timing_for_net(NetIndex(d.net))
            .ok_or_else(|| anyhow!("missing driver timing"))?;
        let slew = timing.rise.transition.max(timing.fall.transition);
        let load = d.load[0].max(d.load[1]);
        m.max_slew = m.max_slew.max(slew);
        m.max_load = m.max_load.max(load);
        m.max_fanout = m.max_fanout.max(d.sinks.len());
        m.output_load_above_table_max += usize::from(load > table_bound(library, pin, None, true));
        m.max_capacitance_violations += usize::from(pin.max_capacitance.is_some_and(|v| load > v));
        m.max_transition_violations += usize::from(pin.max_transition.is_some_and(|v| slew > v));
        for sink in &d.sinks {
            let si = &netlist.module.instances[sink.instance];
            let sn = resolve_symbol(&netlist.interner, si.type_name, "sink cell")?;
            let sc = library
                .cells
                .iter()
                .find(|c| c.name == sn)
                .ok_or_else(|| anyhow!("unknown sink"))?;
            let sp = resolve_symbol(
                &netlist.interner,
                si.connections[sink.connection].0,
                "sink pin",
            )?;
            let bound = sc
                .pins
                .iter()
                .filter(|p| p.direction == PinDirection::Output as i32)
                .map(|p| table_bound(library, p, Some(&sp), false))
                .fold(f64::INFINITY, f64::min);
            m.input_slew_above_table_max += usize::from(slew > bound);
        }
    }
    Ok(m)
}

/// Partitions large capacitances first, with deterministic instance/pin tie
/// breaking.
fn partition(sinks: &[Sink], groups: usize, initial_load: f64) -> Vec<Vec<Sink>> {
    let mut ordered = sinks.to_vec();
    ordered.sort_by(|a, b| {
        b.capacitance
            .total_cmp(&a.capacitance)
            .then(a.instance.cmp(&b.instance))
            .then(a.connection.cmp(&b.connection))
    });
    let mut result = vec![vec![]; groups];
    let mut load = vec![0.0; groups];
    load[0] = initial_load;
    for sink in ordered {
        let k = (0..groups)
            .min_by(|&a, &b| load[a].total_cmp(&load[b]).then(a.cmp(&b)))
            .unwrap();
        load[k] += sink.capacitance;
        result[k].push(sink);
    }
    result
}

/// Allocates a collision-free symbol in the shared namespace, including user
/// names.
fn fresh_symbol(netlist: &mut MappedNetlist) -> crate::netlist::parse::PortId {
    let mut index = netlist.interner.len();
    loop {
        let name = format!("fanout_opt_{index}");
        if netlist.interner.get(&name).is_none() {
            return netlist.interner.get_or_intern(name);
        }
        index += 1;
    }
}

#[cfg(test)]
fn apply(
    netlist: &mut MappedNetlist,
    library: &Library,
    driver: &Driver,
    kind: &FanoutEdit,
    choices: &[CellChoice],
) -> Result<()> {
    let g = graph(netlist, library, &FanoutOptimizationOptions::default())?;
    edit::EditPlan::new(netlist, &g, driver, kind, choices)?.apply(netlist, library)
}

mod edit;

/// Selects a small spread of library strengths from an area-ordered list.
fn spread<T: Clone>(items: &[T]) -> Vec<T> {
    if items.is_empty() {
        return vec![];
    }
    let indices = BTreeSet::from([0, items.len() / 2, items.len() - 1]);
    indices.into_iter().map(|i| items[i].clone()).collect()
}

/// Enumerates at most 24 alternatives using equivalent pin functions and
/// positive buffers.
fn candidates(driver: &Driver, old: &CellChoice, choices: &[CellChoice]) -> Vec<FanoutEdit> {
    let mut out = Vec::new();
    let equivalents = choices
        .iter()
        .filter(|c| c.inputs == old.inputs && c.truth == old.truth && c.area > old.area)
        .collect::<Vec<_>>();
    for c in spread(&equivalents) {
        out.push(FanoutEdit::Upsize {
            cell: c.name.clone(),
        });
    }
    let buffers = choices
        .iter()
        .filter(|c| c.can_instantiate && c.inputs.len() == 1 && c.truth == 2)
        .collect::<Vec<_>>();
    for c in spread(&buffers) {
        for groups in [1, 2, 4, 8] {
            if groups <= driver.sinks.len() {
                out.push(FanoutEdit::Buffer {
                    cell: c.name.clone(),
                    groups,
                    levels: 1,
                });
                if groups >= 4 {
                    out.push(FanoutEdit::Buffer {
                        cell: c.name.clone(),
                        groups,
                        levels: 2,
                    });
                }
            }
        }
    }
    for groups in [2, 4, 8] {
        if old.can_instantiate && groups <= driver.sinks.len() {
            out.push(FanoutEdit::Replicate { groups });
        }
    }
    out
}

/// Captures actual input arrival/slew at an accepted edit, including previous
/// repairs.
fn input_context(
    netlist: &MappedNetlist,
    driver: &Driver,
    g: &Graph,
    sta: &StaReport,
) -> Result<Vec<(String, [f64; 4])>> {
    let mut out = Vec::new();
    for (j, (pin, r)) in netlist.module.instances[driver.instance]
        .connections
        .iter()
        .enumerate()
    {
        if j == driver.output_connection {
            continue;
        }
        let name = resolve_symbol(&netlist.interner, *pin, "pin")?;
        if let Some(n) = source(r, &g.roots)? {
            if let Some(t) = sta.timing_for_net(NetIndex(n)) {
                out.push((
                    name,
                    [
                        t.rise.arrival,
                        t.fall.arrival,
                        t.rise.transition,
                        t.fall.transition,
                    ],
                ));
            }
        }
    }
    Ok(out)
}

/// Incremental costs and affected-cone sizes; oracle mode is measured
/// separately.
#[derive(Clone, Debug, Default, Serialize)]
pub struct IncrementalStats {
    pub cache_seconds: f64,
    pub oracle_seconds: f64,
    pub oracle_checks: usize,
    pub accepted_checks: usize,
    pub candidates: Vec<incremental::Work>,
}

mod incremental;

/// Improves modeled worst output delay using bounded buffering, replication
/// and equivalent-cell upsizing. Original connectivity need not be electrically
/// illegal: high fanout and table-range excursions only prioritize candidates.
///
/// Every accepted edit reduces both global delay and delay/area score within
/// the original area budget. Incremental STA includes upstream load changes;
/// full STA and per-candidate oracle modes are available for validation.
/// Supports owned flat scalar combinational modules and directed wire aliases.
pub fn optimize_fanout(
    netlist: &mut MappedNetlist,
    library: &Library,
    options: FanoutOptimizationOptions,
) -> Result<FanoutOptimizationReport> {
    ensure!(
        !(options.full_sta && options.verify_incremental),
        "full STA and incremental oracle modes are mutually exclusive"
    );
    ensure!(
        options.max_rounds <= 32 && (1..=8).contains(&options.drivers_per_round),
        "search bounds exceeded"
    );
    ensure!(
        options.fanout_trigger > 0
            && options.area_growth_limit.is_finite()
            && options.area_growth_limit >= 0.0
            && options.area_weight.is_finite()
            && options.area_weight >= 0.0,
        "invalid optimization limits"
    );
    let start = Instant::now();
    let prepared = std::sync::Arc::new(crate::netlist::sta::PreparedStaLibrary::new(library)?);
    let cells = choices_prepared(library, &prepared)?;
    let ts = Instant::now();
    let (mut current, mut sta, mut g) = measure_prepared(netlist, library, &options, &prepared)?;
    let mut sta_seconds = ts.elapsed().as_secs_f64();
    let mut cache = None;
    let mut use_full_sta = options.full_sta;
    let mut incremental_fallback_reason = None;
    let mut incremental = IncrementalStats::default();
    let mut sta_evaluations = 1;
    let before = current.clone();
    let mut trials = vec![];
    let mut decisions = vec![];
    let score = |m: &FanoutTrialMetrics| {
        m.worst_delay / before.worst_delay.max(1e-12)
            + options.area_weight * (m.area / before.area.max(1e-12) - 1.0)
    };
    for round in 0..options.max_rounds {
        let mut targets = Vec::new();
        for d in &g.drivers {
            let name = resolve_symbol(
                &netlist.interner,
                netlist.module.instances[d.instance].type_name,
                "cell",
            )?;
            let Some(c) = cells.iter().find(|c| c.name == name) else {
                continue;
            };
            let t = sta
                .timing_for_net(NetIndex(d.net))
                .ok_or_else(|| anyhow!("missing driver timing"))?;
            let p = library.cells[c.index]
                .pins
                .iter()
                .find(|p| library.resolve_string(&p.name) == c.output)
                .ok_or_else(|| anyhow!("missing pin"))?;
            let load = d.load[0].max(d.load[1]);
            let slew = t.rise.transition.max(t.fall.transition);
            let overloaded = load > p.max_capacitance.unwrap_or(f64::INFINITY)
                || load > table_bound(library, p, None, true)
                || slew > d.slew_bound
                || slew > p.max_transition.unwrap_or(f64::INFINITY);
            if d.sinks.len() >= options.fanout_trigger || overloaded {
                targets.push((slew, d.instance, d.clone(), c.clone()));
            }
        }
        targets.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        targets.truncate(options.drivers_per_round);
        if targets.is_empty() {
            break;
        }
        if !use_full_sta && cache.is_none() {
            let start = Instant::now();
            match incremental::Engine::new_prepared(
                netlist,
                library,
                &options,
                &sta,
                prepared.clone(),
            ) {
                Ok(engine) => cache = Some(engine),
                Err(error)
                    if error
                        .downcast_ref::<incremental::UnsupportedTopology>()
                        .is_some() =>
                {
                    use_full_sta = true;
                    incremental_fallback_reason = Some(error.to_string());
                }
                Err(error) => return Err(error),
            }
            incremental.cache_seconds = start.elapsed().as_secs_f64();
        }
        let mut best: Option<TrialWinner> = None;
        for (_, _, driver, old) in targets {
            for kind in candidates(&driver, &old, &cells) {
                let instance = resolve_symbol(
                    &netlist.interner,
                    netlist.module.instances[driver.instance].instance_name,
                    "instance",
                )?;
                let plan = edit::EditPlan::new(netlist, &g, &driver, &kind, &cells)?;
                let mut full_candidate = None;
                let area = if let Some(engine) = cache.as_ref() {
                    engine.candidate_area(&plan)
                } else {
                    let mut candidate = netlist.clone();
                    plan.apply(&mut candidate, library)?;
                    let area =
                        build_area_report(&candidate.module, &candidate.interner, library)?.area;
                    full_candidate = Some(candidate);
                    area
                };
                if area > before.area * (1.0 + options.area_growth_limit) + 1e-10 {
                    trials.push(FanoutTrial {
                        round,
                        instance,
                        kind,
                        metrics: None,
                        score: None,
                        reason: "area budget".into(),
                    });
                    continue;
                }
                let ts = Instant::now();
                let measured = if let Some(engine) = cache.as_mut() {
                    engine.trial(&plan, &g, area, |engine, metrics| {
                        if options.verify_incremental {
                            let start = Instant::now();
                            let mut candidate = netlist.clone();
                            plan.apply(&mut candidate, library)?;
                            let (full_metrics, full_sta, full_g) =
                                measure_prepared(&candidate, library, &options, &prepared)
                                    .map_err(|error| {
                                        anyhow!("full STA oracle failed: {error:#}")
                                    })?;
                            engine.check(&full_sta)?;
                            engine.check_loads(&full_g)?;
                            ensure!(
                                *metrics == FanoutTrialMetrics::from(&full_metrics),
                                "oracle optimization metrics mismatch: {:?} vs {:?}",
                                metrics,
                                full_metrics
                            );
                            incremental.oracle_checks += 1;
                            incremental.oracle_seconds += start.elapsed().as_secs_f64();
                        }
                        Ok(())
                    })
                } else {
                    let candidate = full_candidate.as_ref().expect("full STA candidate exists");
                    prepared
                        .analyze_combinational_max_arrival(
                            &candidate.module,
                            &candidate.nets,
                            &candidate.interner,
                            StaOptions {
                                primary_input_transition: options.primary_input_transition,
                                module_output_load: options.module_output_load,
                            },
                        )
                        .map_err(CandidateTimingError)
                        .map(|sta| {
                            (
                                FanoutTrialMetrics {
                                    area,
                                    cell_count: candidate.module.instances.len(),
                                    worst_delay: sta.worst_output_arrival,
                                },
                                incremental::Work::default(),
                            )
                        })
                        .map_err(anyhow::Error::from)
                };
                sta_seconds += ts.elapsed().as_secs_f64();
                sta_evaluations += 1;
                // Candidate coverage/table failures are ordinary rejections.
                // Oracle and topology failures remain fatal.
                let (metrics, work) = match measured {
                    Ok(value) => value,
                    Err(error) if error.downcast_ref::<CandidateTimingError>().is_some() => {
                        trials.push(FanoutTrial {
                            round,
                            instance,
                            kind,
                            metrics: None,
                            score: None,
                            reason: format!("STA rejected: {error}"),
                        });
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                if !use_full_sta {
                    incremental.candidates.push(work);
                }
                let value = score(&metrics);
                let eligible = metrics.worst_delay < current.worst_delay - 1e-9
                    && value < score(&FanoutTrialMetrics::from(&current)) - 1e-12;
                let wins = eligible
                    && best.as_ref().is_none_or(|b| {
                        value < b.score - 1e-12
                            || ((value - b.score).abs() <= 1e-12 && metrics.area < b.metrics.area)
                    });
                let trial = trials.len();
                trials.push(FanoutTrial {
                    round,
                    instance,
                    kind: kind.clone(),
                    metrics: Some(metrics.clone()),
                    score: Some(value),
                    reason: if eligible {
                        "eligible, not selected"
                    } else {
                        "no global delay/area-score improvement"
                    }
                    .into(),
                });
                if wins {
                    best = Some(TrialWinner {
                        score: value,
                        metrics,
                        plan,
                        driver: driver.clone(),
                        kind,
                        trial,
                    });
                }
            }
        }
        let Some(TrialWinner {
            metrics,
            plan,
            driver,
            kind,
            trial,
            ..
        }) = best
        else {
            break;
        };
        trials[trial].reason = "selected".into();
        let input_timing = input_context(netlist, &driver, &g, &sta)?;
        if let Some(engine) = cache.as_mut() {
            engine.commit(&plan, &g)?;
        }
        plan.apply(netlist, library)?;
        let accepted = if let Some(engine) = cache.as_ref() {
            sta = engine.report();
            g = graph(netlist, library, &options)?;
            report_metrics(netlist, library, &sta, &g)?
        } else {
            let ts = Instant::now();
            let (accepted, accepted_sta, accepted_graph) =
                measure_prepared(netlist, library, &options, &prepared)?;
            sta_seconds += ts.elapsed().as_secs_f64();
            sta_evaluations += 1;
            sta = accepted_sta;
            g = accepted_graph;
            accepted
        };
        ensure!(
            FanoutTrialMetrics::from(&accepted) == metrics,
            "accepted objective differs from candidate"
        );
        if options.verify_incremental
            && let Some(engine) = cache.as_ref()
        {
            let start = Instant::now();
            let (full, full_sta, full_graph) =
                measure_prepared(netlist, library, &options, &prepared)?;
            engine.check(&full_sta)?;
            engine.check_loads(&full_graph)?;
            ensure!(
                accepted == full,
                "accepted optimization differs from full STA"
            );
            incremental.accepted_checks += 1;
            incremental.oracle_seconds += start.elapsed().as_secs_f64();
        }
        decisions.push(FanoutDecision {
            round,
            instance: trials[trial].instance.clone(),
            kind,
            input_timing,
            sink_count: driver.sinks.len(),
            output_ports: driver.output_ports,
            old_load: driver.load,
            before: current,
            after: accepted.clone(),
        });
        current = accepted;
    }
    drop(cache);
    Ok(FanoutOptimizationReport {
        incremental: (!use_full_sta).then_some(incremental),
        incremental_fallback_reason,
        options,
        before,
        after: current,
        decisions,
        trials,
        sta_evaluations,
        sta_seconds,
        optimization_seconds: start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests;
