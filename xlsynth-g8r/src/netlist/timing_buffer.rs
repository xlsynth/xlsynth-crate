// SPDX-License-Identifier: Apache-2.0

//! Exact-Liberty, criticality-aware buffering of registered mapped netlists.

use crate::liberty_model::{Library, PinDirection};
use crate::netlist::buffer::{BufferOptions, BufferStats};
use crate::netlist::cell_catalog::{CatalogCell, CellCatalog};
use crate::netlist::normalized::{BitExpr, BitIndex, BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistInstance, NetlistModule, PortDirection};
use crate::netlist::report::{NetlistReport, build_netlist_report_with_primary_input_arrivals};
use crate::netlist::sta::{
    CombinationalOutputLoad, SignalTiming, StaOptions, StaReport, TimingQueryDiagnosticCounts,
    analyze_combinational_max_arrival_with_primary_input_arrivals,
    analyze_register_boundary_max_arrival_with_primary_input_arrivals,
    effective_input_capacitance_for_mapping, evaluate_combinational_cell_output_timing,
    is_sequential_boundary_cell,
};
use crate::netlist::utils::validate_constant_output_assignments;
use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// ABC's default buffer gain is three, giving a nine-to-one gate effort.
const MAX_ELECTRICAL_EFFORT: f64 = 9.0;
/// Bound full-STA work while allowing individually useful rejected batches.
const MAX_TIMING_EVALUATIONS: usize = 64;
/// Evaluate several independent net edits with one exact timing analysis.
const MAX_TIMING_BATCH_SIZE: usize = 8;
/// Keep insignificant floating-point differences from changing a cover.
const TIMING_EPSILON: f64 = 1e-9;

/// External timing requirements preserved during registered buffer insertion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BufferTimingConstraints {
    /// Launch arrivals of flattened external primary-input bits.
    pub primary_input_arrivals: BTreeMap<String, f64>,
    /// Required arrival times of flattened external primary-output bits.
    pub primary_output_required: BTreeMap<String, f64>,
    /// Optional register-capture deadline, in Liberty time units.
    pub clock_period: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimingSinkTarget {
    InstancePin {
        instance_index: usize,
        connection_index: usize,
    },
    ModuleOutput {
        net: NetIndex,
        bit: u32,
    },
}

#[derive(Clone, Debug)]
struct TimingSink {
    target: TimingSinkTarget,
    load: CombinationalOutputLoad,
    criticality: f64,
}

#[derive(Clone, Copy, Debug)]
struct TimingDriver {
    instance_index: usize,
    connection_index: usize,
    cell_index: usize,
    pin_index: usize,
}

#[derive(Clone, Debug, Default)]
struct TimingFanout {
    sinks: Vec<TimingSink>,
    driver: Option<TimingDriver>,
    is_primary_input: bool,
    protected_clock: bool,
    constant: bool,
}

#[derive(Clone, Debug, Default)]
struct TimingInstance {
    inputs: Vec<(BitIndex, usize)>,
    outputs: Vec<BitIndex>,
    sequential: bool,
}

#[derive(Clone, Debug)]
struct TimingGraph {
    fanouts: Vec<TimingFanout>,
    bits: Vec<(NetIndex, u32)>,
    instances: Vec<TimingInstance>,
    departures: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct TimingSinkGroup {
    sinks: Vec<TimingSink>,
    load: CombinationalOutputLoad,
    has_module_output: bool,
}

#[derive(Clone, Debug)]
struct TimingSnapshot {
    report: NetlistReport,
    combined: StaReport,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
struct BatchStats {
    buffered_nets: usize,
    buffers_inserted: usize,
    area_added: f64,
}

#[derive(Default)]
struct BufferNames {
    wire: usize,
    instance: usize,
}

/// Inserts electrically bounded, exact-timing-validated register-aware buffers.
pub fn insert_timing_aware_buffers(
    module: &mut NetlistModule,
    nets: &mut Vec<Net>,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
    sta_options: StaOptions,
    constraints: &BufferTimingConstraints,
) -> Result<BufferStats> {
    validate_timing_options(options, sta_options, constraints)?;
    if module.net_index_range.end != nets.len() {
        bail!("timing-driven buffering requires the module to own the end of the net table");
    }

    validate_constant_output_assignments(module, nets)
        .context("validating registered buffer output assignments")?;
    let catalog = CellCatalog::new(library)?;
    let mut snapshot =
        analyze_timing_snapshot(module, nets, interner, library, sta_options, constraints)?;
    let initial_graph = build_timing_graph(
        module,
        nets,
        interner,
        library,
        options,
        constraints,
        &snapshot,
    )?;
    let mut stats = BufferStats {
        max_fanout_before: eligible_fanouts(&initial_graph, options)
            .map(|fanout| fanout.sinks.len())
            .max()
            .unwrap_or(0),
        max_load_before: eligible_fanouts(&initial_graph, options)
            .map(|fanout| max_load(sum_sink_load(&fanout.sinks)))
            .fold(0.0, f64::max),
        initial_worst_delay: Some(worst_path_delay(&snapshot.report)),
        final_worst_delay: Some(worst_path_delay(&snapshot.report)),
        timing_evaluations: 1,
        ..BufferStats::default()
    };

    let mut roots = overloaded_roots(&initial_graph, library, &catalog, options, &snapshot)?;
    if roots.is_empty() {
        stats.max_fanout_after = stats.max_fanout_before;
        stats.max_load_after = stats.max_load_before;
        return Ok(stats);
    }
    if catalog.buffers().next().is_none() {
        bail!("timing-driven buffering found an overloaded net but Liberty has no usable buffer");
    }

    let hard_roots = roots
        .iter()
        .copied()
        .filter(|root| {
            let fanout = &initial_graph.fanouts[*root];
            fanout.sinks.len() > options.max_fanout
                || exceeds_characterized_output_limit(fanout, library, options)
        })
        .collect::<Vec<_>>();
    if !hard_roots.is_empty() {
        let mut trial_module = module.clone();
        let mut trial_nets = nets.clone();
        let mut trial_interner = interner.clone();
        let mut names = BufferNames::default();
        let mut batch_stats = BatchStats::default();
        for root in hard_roots {
            apply_buffer_tree(
                &mut trial_module,
                &mut trial_nets,
                &mut trial_interner,
                library,
                &catalog,
                options,
                &snapshot,
                &initial_graph,
                root,
                &mut names,
                &mut batch_stats,
            )?;
        }
        if batch_stats.buffers_inserted > 0 {
            let candidate = analyze_timing_snapshot(
                &trial_module,
                &trial_nets,
                &trial_interner,
                library,
                sta_options,
                constraints,
            );
            stats.timing_evaluations += 1;
            match candidate {
                Ok(candidate) if preserves_constraints(&candidate, constraints) => {
                    *module = trial_module;
                    *nets = trial_nets;
                    *interner = trial_interner;
                    stats.buffered_nets += batch_stats.buffered_nets;
                    stats.buffers_inserted += batch_stats.buffers_inserted;
                    stats.area_added += batch_stats.area_added;
                    stats.final_worst_delay = Some(worst_path_delay(&candidate.report));
                    snapshot = candidate;
                    let graph = build_timing_graph(
                        module,
                        nets,
                        interner,
                        library,
                        options,
                        constraints,
                        &snapshot,
                    )?;
                    roots = overloaded_roots(&graph, library, &catalog, options, &snapshot)?;
                }
                Ok(_) => {
                    stats.rejected_timing_batches += 1;
                }
                Err(error) => {
                    stats.rejected_timing_batches += 1;
                    log::debug!("rejecting mandatory electrical buffer repair: {error:#}");
                }
            }
        }
    }

    let mut batches = VecDeque::new();
    while !roots.is_empty() {
        let take = roots.len().min(MAX_TIMING_BATCH_SIZE);
        batches.push_back(roots.drain(..take).collect::<Vec<_>>());
    }

    while let Some(batch) = batches.pop_front() {
        if stats.timing_evaluations >= MAX_TIMING_EVALUATIONS {
            break;
        }

        let graph = build_timing_graph(
            module,
            nets,
            interner,
            library,
            options,
            constraints,
            &snapshot,
        )?;
        let mut trial_module = module.clone();
        let mut trial_nets = nets.clone();
        let mut trial_interner = interner.clone();
        let mut names = BufferNames::default();
        let mut batch_stats = BatchStats::default();
        let mut fixes_hard_violation = false;

        for root in &batch {
            let Some(fanout) = graph.fanouts.get(*root) else {
                continue;
            };
            if !fanout_eligible(fanout, options)
                || !fanout_is_overloaded(fanout, library, &catalog, options)?
            {
                continue;
            }
            fixes_hard_violation |= fanout.sinks.len() > options.max_fanout
                || exceeds_characterized_output_limit(fanout, library, options);
            apply_buffer_tree(
                &mut trial_module,
                &mut trial_nets,
                &mut trial_interner,
                library,
                &catalog,
                options,
                &snapshot,
                &graph,
                *root,
                &mut names,
                &mut batch_stats,
            )?;
        }

        if batch_stats.buffers_inserted == 0 {
            continue;
        }

        let trial = analyze_timing_snapshot(
            &trial_module,
            &trial_nets,
            &trial_interner,
            library,
            sta_options,
            constraints,
        );
        stats.timing_evaluations += 1;
        let accepted = match trial {
            Ok(candidate) if preserves_constraints(&candidate, constraints) => {
                if timing_improves(&candidate.report, &snapshot.report) || fixes_hard_violation {
                    *module = trial_module;
                    *nets = trial_nets;
                    *interner = trial_interner;
                    stats.buffered_nets += batch_stats.buffered_nets;
                    stats.buffers_inserted += batch_stats.buffers_inserted;
                    stats.area_added += batch_stats.area_added;
                    stats.final_worst_delay = Some(worst_path_delay(&candidate.report));
                    snapshot = candidate;
                    true
                } else {
                    false
                }
            }
            Ok(_) => false,
            Err(error) => {
                log::debug!("rejecting timing-driven buffer batch: {error:#}");
                false
            }
        };

        if !accepted {
            stats.rejected_timing_batches += 1;
            if batch.len() > 1 {
                let midpoint = batch.len() / 2;
                batches.push_back(batch[..midpoint].to_vec());
                batches.push_back(batch[midpoint..].to_vec());
            }
        }
    }

    let final_graph = build_timing_graph(
        module,
        nets,
        interner,
        library,
        options,
        constraints,
        &snapshot,
    )?;
    stats.max_fanout_after = eligible_fanouts(&final_graph, options)
        .map(|fanout| fanout.sinks.len())
        .max()
        .unwrap_or(0);
    stats.max_load_after = eligible_fanouts(&final_graph, options)
        .map(|fanout| max_load(sum_sink_load(&fanout.sinks)))
        .fold(0.0, f64::max);
    stats.unresolved_overloaded_nets = eligible_fanouts(&final_graph, options)
        .map(|fanout| fanout_is_overloaded(fanout, library, &catalog, options))
        .try_fold(0usize, |count, overloaded| {
            overloaded.map(|overloaded| count + usize::from(overloaded))
        })?;
    Ok(stats)
}

/// Rejects invalid electrical and endpoint constraints before editing a net.
fn validate_timing_options(
    options: &BufferOptions,
    sta_options: StaOptions,
    constraints: &BufferTimingConstraints,
) -> Result<()> {
    if options.max_fanout < 2 {
        bail!("buffer max_fanout must be at least 2");
    }
    if !options.module_output_load.is_finite() || options.module_output_load < 0.0 {
        bail!("buffer module_output_load must be non-negative and finite");
    }
    if options
        .target_load
        .is_some_and(|load| !load.is_finite() || load <= 0.0)
    {
        bail!("buffer target_load must be positive and finite");
    }
    if !sta_options.primary_input_transition.is_finite()
        || sta_options.primary_input_transition < 0.0
        || !sta_options.module_output_load.is_finite()
        || sta_options.module_output_load < 0.0
    {
        bail!("buffer STA options must be non-negative and finite");
    }
    if constraints
        .clock_period
        .is_some_and(|period| !period.is_finite() || period <= 0.0)
    {
        bail!("buffer clock period must be finite and strictly positive");
    }
    Ok(())
}

/// Independently times every input, output, and register-capture endpoint.
fn analyze_timing_snapshot(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: StaOptions,
    constraints: &BufferTimingConstraints,
) -> Result<TimingSnapshot> {
    let report = build_netlist_report_with_primary_input_arrivals(
        module,
        nets,
        interner,
        library,
        options,
        &constraints.primary_input_arrivals,
    )
    .context("computing exact timing-driven buffer endpoint report")?;
    let cells: HashMap<&str, usize> = library
        .cells
        .iter()
        .enumerate()
        .map(|(index, cell)| (cell.name.as_str(), index))
        .collect();
    let registers = module
        .instances
        .iter()
        .enumerate()
        .filter_map(|(index, instance)| {
            interner
                .resolve(instance.type_name)
                .and_then(|name| cells.get(name).copied())
                .is_some_and(|cell| is_sequential_boundary_cell(&library.cells[cell]))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let combined = if registers.is_empty() {
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
            &registers,
            &constraints.primary_input_arrivals,
        )?
    };
    Ok(TimingSnapshot { report, combined })
}

/// Builds canonical per-bit fanouts and exact-timing downstream priorities.
fn build_timing_graph(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
    constraints: &BufferTimingConstraints,
    snapshot: &TimingSnapshot,
) -> Result<TimingGraph> {
    validate_constant_output_assignments(module, nets)?;
    let normalized = NormalizedNetlistModule::new(module, nets, interner)
        .context("normalizing timing-driven buffer connectivity")?;
    let cells: HashMap<&str, usize> = library
        .cells
        .iter()
        .enumerate()
        .map(|(index, cell)| (cell.name.as_str(), index))
        .collect();
    let mut fanouts = vec![TimingFanout::default(); normalized.bit_count()];
    let bits = (0..normalized.bit_count())
        .map(|index| {
            let bit = normalized.bit(normalized.canonical_bit(index));
            (bit.net, bit.bit_number)
        })
        .collect::<Vec<_>>();
    let mut instances = vec![TimingInstance::default(); normalized.instances.len()];

    for assign in &normalized.assigns {
        for (bit, expression) in assign.lhs_bits.iter().zip(&assign.rhs_bits) {
            if matches!(expression, BitExpr::Source(BitSource::Literal(_))) {
                fanouts[*bit].constant = true;
            }
        }
    }

    for port in &normalized.ports {
        let port_net = module.find_net_index(port.name, nets).ok_or_else(|| {
            anyhow!("timing-driven buffering cannot resolve a normalized module port")
        })?;
        for (offset, &bit) in port.bits.iter().enumerate() {
            match port.direction {
                PortDirection::Input => fanouts[bit].is_primary_input = true,
                PortDirection::Output => {
                    let bit_number = nets[port_net.0].bit_number(offset).ok_or_else(|| {
                        anyhow!("timing-driven buffering found an invalid packed output bit")
                    })?;
                    fanouts[bit].sinks.push(TimingSink {
                        target: TimingSinkTarget::ModuleOutput {
                            net: port_net,
                            bit: bit_number,
                        },
                        load: CombinationalOutputLoad {
                            rise: options.module_output_load,
                            fall: options.module_output_load,
                        },
                        criticality: 0.0,
                    });
                }
                PortDirection::Inout => fanouts[bit].protected_clock = true,
            }
        }
    }

    for (instance_index, instance) in normalized.instances.iter().enumerate() {
        let name = interner
            .resolve(instance.type_name)
            .ok_or_else(|| anyhow!("cannot resolve a timing-driven buffer cell"))?;
        let cell_index = cells
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("mapped instance references unknown Liberty cell '{name}'"))?;
        let cell = &library.cells[cell_index];
        instances[instance_index].sequential = is_sequential_boundary_cell(cell);

        for (connection_index, connection) in instance.connections.iter().enumerate() {
            let pin_name = interner
                .resolve(connection.port)
                .ok_or_else(|| anyhow!("cannot resolve a timing-driven buffer cell pin"))?;
            let (pin_index, pin) = cell
                .pins
                .iter()
                .enumerate()
                .find(|(_, pin)| library.resolve_string(&pin.name) == pin_name)
                .ok_or_else(|| anyhow!("cell '{name}' has no pin '{pin_name}'"))?;

            if connection.bits.len() > 1 {
                bail!(
                    "timing-driven buffering requires one-bit Liberty pin bindings; '{}.{}' has {} bits",
                    name,
                    pin_name,
                    connection.bits.len()
                );
            }
            let Some(BitSource::Bit(bit)) = connection.bits.first().copied() else {
                continue;
            };
            if pin.direction == PinDirection::Input as i32 {
                if pin.is_clocking_pin {
                    fanouts[bit].protected_clock = true;
                }
                let load = effective_input_capacitance_for_mapping(
                    pin,
                    &format!("timing-driven buffer sink '{name}.{pin_name}'"),
                )?;
                fanouts[bit].sinks.push(TimingSink {
                    target: TimingSinkTarget::InstancePin {
                        instance_index,
                        connection_index,
                    },
                    load,
                    criticality: 0.0,
                });
                if !pin.is_clocking_pin {
                    instances[instance_index]
                        .inputs
                        .push((bit, connection_index));
                }
            } else if pin.direction == PinDirection::Output as i32 {
                if fanouts[bit]
                    .driver
                    .replace(TimingDriver {
                        instance_index,
                        connection_index,
                        cell_index,
                        pin_index,
                    })
                    .is_some()
                {
                    bail!("timing-driven buffering requires one driver per canonical net bit");
                }
                instances[instance_index].outputs.push(bit);
            }
        }
    }

    let mut graph = TimingGraph {
        fanouts,
        bits,
        instances,
        departures: vec![0.0; normalized.bit_count()],
    };
    compute_downstream_criticality(&mut graph, module, nets, interner, constraints, snapshot)?;
    Ok(graph)
}

/// Propagates actual endpoint arrival and setup urgency toward source nets.
fn compute_downstream_criticality(
    graph: &mut TimingGraph,
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    constraints: &BufferTimingConstraints,
    snapshot: &TimingSnapshot,
) -> Result<()> {
    for (bit_index, fanout) in graph.fanouts.iter().enumerate() {
        for sink in &fanout.sinks {
            match sink.target {
                TimingSinkTarget::InstancePin { instance_index, .. }
                    if graph.instances[instance_index].sequential =>
                {
                    if let Some(arrival) = snapshot
                        .combined
                        .register_input_arrivals
                        .get(instance_index)
                        .copied()
                        .flatten()
                    {
                        let source_arrival = signal_arrival(snapshot, graph.bits[bit_index].0);
                        graph.departures[bit_index] =
                            graph.departures[bit_index].max((arrival - source_arrival).max(0.0));
                    }
                }
                TimingSinkTarget::ModuleOutput { net, bit } => {
                    let name = flattened_output_name(net, bit, module, nets, interner)?;
                    if let Some(required) = constraints.primary_output_required.get(&name) {
                        graph.departures[bit_index] = graph.departures[bit_index]
                            .max((worst_path_delay(&snapshot.report) - required).max(0.0));
                    }
                }
                TimingSinkTarget::InstancePin { .. } => {}
            }
        }
    }

    let mut successors = vec![Vec::<usize>::new(); graph.instances.len()];
    let mut indegrees = vec![0usize; graph.instances.len()];
    for (instance_index, instance) in graph.instances.iter().enumerate() {
        if instance.sequential {
            continue;
        }
        let mut predecessors = BTreeSet::new();
        for &(bit, _) in &instance.inputs {
            if let Some(driver) = graph.fanouts[bit].driver
                && driver.instance_index != instance_index
                && !graph.instances[driver.instance_index].sequential
            {
                predecessors.insert(driver.instance_index);
            }
        }
        indegrees[instance_index] = predecessors.len();
        for predecessor in predecessors {
            successors[predecessor].push(instance_index);
        }
    }
    let mut ready: BTreeSet<usize> = graph
        .instances
        .iter()
        .enumerate()
        .filter_map(|(index, instance)| {
            (!instance.sequential && indegrees[index] == 0).then_some(index)
        })
        .collect();
    let mut topological = Vec::new();
    while let Some(index) = ready.pop_first() {
        topological.push(index);
        for &successor in &successors[index] {
            indegrees[successor] -= 1;
            if indegrees[successor] == 0 {
                ready.insert(successor);
            }
        }
    }

    for instance_index in topological.into_iter().rev() {
        let instance = &graph.instances[instance_index];
        let output_departure = instance
            .outputs
            .iter()
            .map(|&bit| graph.departures[bit])
            .fold(0.0, f64::max);
        let output_arrival = instance
            .outputs
            .iter()
            .map(|&bit| signal_arrival(snapshot, graph.bits[bit].0))
            .fold(0.0, f64::max);
        for &(bit, _) in &instance.inputs {
            let input_arrival = signal_arrival(snapshot, graph.bits[bit].0);
            let edge_delay = (output_arrival - input_arrival).max(0.0);
            graph.departures[bit] = graph.departures[bit].max(output_departure + edge_delay);
        }
    }

    for bit_index in 0..graph.fanouts.len() {
        let source_arrival = signal_arrival(snapshot, graph.bits[bit_index].0);
        for sink in &mut graph.fanouts[bit_index].sinks {
            sink.criticality = match sink.target {
                TimingSinkTarget::InstancePin { instance_index, .. }
                    if graph.instances[instance_index].sequential =>
                {
                    graph.departures[bit_index]
                }
                TimingSinkTarget::InstancePin { instance_index, .. } => graph.instances
                    [instance_index]
                    .outputs
                    .iter()
                    .map(|&output| {
                        graph.departures[output]
                            + (signal_arrival(snapshot, graph.bits[output].0) - source_arrival)
                                .max(0.0)
                    })
                    .fold(0.0, f64::max),
                TimingSinkTarget::ModuleOutput { .. } => graph.departures[bit_index],
            };
        }
    }
    Ok(())
}

/// Ranks hard electrical repairs before severity and exact sink criticality.
fn overloaded_roots(
    graph: &TimingGraph,
    library: &Library,
    catalog: &CellCatalog,
    options: &BufferOptions,
    snapshot: &TimingSnapshot,
) -> Result<Vec<usize>> {
    let mut roots = Vec::new();
    for (index, fanout) in graph.fanouts.iter().enumerate() {
        if !fanout_eligible(fanout, options)
            || !fanout_is_overloaded(fanout, library, catalog, options)?
        {
            continue;
        }
        let priority = signal_arrival(snapshot, graph.bits[index].0)
            + fanout
                .sinks
                .iter()
                .map(|sink| sink.criticality)
                .fold(0.0, f64::max);
        let fanout_ratio = fanout.sinks.len() as f64 / options.max_fanout as f64;
        let load_ratio = characterized_load_limit(fanout, library, catalog, options)?
            .filter(|limit| *limit > 0.0)
            .map(|limit| max_load(sum_sink_load(&fanout.sinks)) / limit)
            .unwrap_or(0.0);
        let hard = fanout.sinks.len() > options.max_fanout
            || exceeds_characterized_output_limit(fanout, library, options);
        roots.push((index, hard, fanout_ratio.max(load_ratio), priority));
    }
    roots.sort_by(|lhs, rhs| {
        rhs.1
            .cmp(&lhs.1)
            .then_with(|| rhs.2.total_cmp(&lhs.2))
            .then_with(|| rhs.3.total_cmp(&lhs.3))
            .then_with(|| lhs.0.cmp(&rhs.0))
    });
    Ok(roots.into_iter().map(|(index, _, _, _)| index).collect())
}

/// Returns only legal, driven data roots; clocks and constants are protected.
fn fanout_eligible(fanout: &TimingFanout, options: &BufferOptions) -> bool {
    !fanout.protected_clock
        && !fanout.constant
        && !fanout.sinks.is_empty()
        && (fanout.driver.is_some() || (fanout.is_primary_input && options.buffer_primary_inputs))
        && (!fanout.is_primary_input || options.buffer_primary_inputs)
}

/// Iterates eligible data fanouts without allocating temporary root lists.
fn eligible_fanouts<'a>(
    graph: &'a TimingGraph,
    options: &'a BufferOptions,
) -> impl Iterator<Item = &'a TimingFanout> + 'a {
    graph
        .fanouts
        .iter()
        .filter(move |fanout| fanout_eligible(fanout, options))
}

/// Tests the real driver capacitance and ABC-style electrical effort.
fn fanout_is_overloaded(
    fanout: &TimingFanout,
    library: &Library,
    catalog: &CellCatalog,
    options: &BufferOptions,
) -> Result<bool> {
    if fanout.sinks.len() > options.max_fanout {
        return Ok(true);
    }
    let load = max_load(sum_sink_load(&fanout.sinks));
    if let Some(limit) = characterized_load_limit(fanout, library, catalog, options)? {
        Ok(load > limit + TIMING_EPSILON)
    } else {
        Ok(false)
    }
}

/// Uses explicit targets, Liberty max capacitance, and bounded gate effort.
fn characterized_load_limit(
    fanout: &TimingFanout,
    library: &Library,
    catalog: &CellCatalog,
    options: &BufferOptions,
) -> Result<Option<f64>> {
    if let Some(limit) = options.target_load {
        return Ok(Some(limit));
    }
    let Some(driver) = fanout.driver else {
        return Ok(catalog
            .buffers()
            .next()
            .map(|buffer| max_load(buffer.input_capacitances[0]) * MAX_ELECTRICAL_EFFORT));
    };
    let cell = &library.cells[driver.cell_index];
    let pin = &cell.pins[driver.pin_index];
    let mut input_capacitance = 0.0;
    let mut input_count = 0usize;
    for input in cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Input as i32 && !pin.is_clocking_pin)
    {
        let capacitance = effective_input_capacitance_for_mapping(
            input,
            &format!(
                "timing-driven buffer driver '{}.{}'",
                cell.name,
                library.resolve_string(&input.name)
            ),
        )?;
        input_capacitance += max_load(capacitance);
        input_count += 1;
    }
    let effort = (input_count > 0 && input_capacitance > 0.0)
        .then_some(input_capacitance / input_count as f64 * MAX_ELECTRICAL_EFFORT);
    let electrical = pin
        .max_capacitance
        .filter(|limit| limit.is_finite() && *limit > 0.0);
    Ok(match (electrical, effort) {
        (Some(electrical), Some(effort)) => Some(electrical.min(effort)),
        (Some(electrical), None) => Some(electrical),
        (None, Some(effort)) => Some(effort),
        (None, None) => None,
    })
}

/// Distinguishes hard Liberty/user electrical bounds from soft effort targets.
fn exceeds_characterized_output_limit(
    fanout: &TimingFanout,
    library: &Library,
    options: &BufferOptions,
) -> bool {
    let limit = options.target_load.or_else(|| {
        fanout
            .driver
            .and_then(|driver| {
                library.cells[driver.cell_index].pins[driver.pin_index].max_capacitance
            })
            .filter(|limit| limit.is_finite() && *limit > 0.0)
    });
    limit.is_some_and(|limit| max_load(sum_sink_load(&fanout.sinks)) > limit + TIMING_EPSILON)
}

/// Preserves exact capture and externally supplied output deadlines.
fn preserves_constraints(snapshot: &TimingSnapshot, constraints: &BufferTimingConstraints) -> bool {
    if let Some(period) = constraints.clock_period {
        for arrival in [
            snapshot.report.max_input_to_register_delay,
            snapshot.report.max_register_to_register_delay,
        ]
        .into_iter()
        .flatten()
        {
            if arrival > period + TIMING_EPSILON {
                return false;
            }
        }
    }
    for (name, required) in &constraints.primary_output_required {
        if let Some(timing) = snapshot.combined.timing_for_output_bit(name)
            && timing.rise.arrival.max(timing.fall.arrival) > required + TIMING_EPSILON
        {
            return false;
        }
    }
    true
}

/// Gives register capture first priority, then output and input timing.
fn timing_improves(candidate: &NetlistReport, current: &NetlistReport) -> bool {
    let candidate_scores = [
        candidate.max_register_to_register_delay.unwrap_or(0.0),
        candidate.max_input_to_register_delay.unwrap_or(0.0),
        candidate.max_register_to_output_delay.unwrap_or(0.0),
        candidate.max_delay.unwrap_or(0.0),
    ];
    let current_scores = [
        current.max_register_to_register_delay.unwrap_or(0.0),
        current.max_input_to_register_delay.unwrap_or(0.0),
        current.max_register_to_output_delay.unwrap_or(0.0),
        current.max_delay.unwrap_or(0.0),
    ];
    for (candidate, current) in candidate_scores.into_iter().zip(current_scores) {
        if (candidate - current).abs() <= TIMING_EPSILON {
            continue;
        }
        return candidate < current;
    }
    false
}

/// Returns the complete worst launched or captured endpoint arrival.
fn worst_path_delay(report: &NetlistReport) -> f64 {
    [
        report.max_delay,
        report.max_input_to_register_delay,
        report.max_register_to_register_delay,
        report.max_register_to_output_delay,
    ]
    .into_iter()
    .flatten()
    .fold(0.0, f64::max)
}

/// Returns the exact rise/fall arrival already established by complete STA.
fn signal_arrival(snapshot: &TimingSnapshot, net: NetIndex) -> f64 {
    snapshot
        .combined
        .timing_for_net(net)
        .map(|timing| timing.rise.arrival.max(timing.fall.arrival))
        .unwrap_or(0.0)
}

/// Splits one overloaded net while keeping its critical sink nearest the root.
#[allow(clippy::too_many_arguments)]
fn apply_buffer_tree(
    module: &mut NetlistModule,
    nets: &mut Vec<Net>,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    catalog: &CellCatalog,
    options: &BufferOptions,
    snapshot: &TimingSnapshot,
    graph: &TimingGraph,
    root: usize,
    names: &mut BufferNames,
    stats: &mut BatchStats,
) -> Result<()> {
    let fanout = &graph.fanouts[root];
    let (root_net, root_bit) = graph.bits[root];
    let mut source = reference_for_bit(root_net, root_bit, nets)?;
    let mut ordered = fanout.sinks.clone();
    ordered.sort_by(|lhs, rhs| {
        rhs.criticality
            .total_cmp(&lhs.criticality)
            .then_with(|| max_load(rhs.load).total_cmp(&max_load(lhs.load)))
            .then_with(|| timing_sink_order(lhs.target, rhs.target))
    });

    let visible_output = ordered
        .iter()
        .any(|sink| matches!(sink.target, TimingSinkTarget::ModuleOutput { .. }));
    if visible_output && let Some(driver) = fanout.driver {
        let internal = append_wire(module, nets, interner, names);
        module.instances[driver.instance_index].connections[driver.connection_index].1 =
            NetRef::Simple(internal);
        source = NetRef::Simple(internal);
    }

    let mut direct = Vec::new();
    if fanout.driver.is_some()
        && let Some(index) = ordered
            .iter()
            .position(|sink| matches!(sink.target, TimingSinkTarget::InstancePin { .. }))
    {
        let sink = ordered.remove(index);
        if visible_output {
            reconnect_sink(module, sink.target, source.clone());
        }
        direct.push(sink);
    }

    if fanout.driver.is_none() {
        let mut remaining = Vec::with_capacity(ordered.len());
        for sink in ordered {
            if matches!(sink.target, TimingSinkTarget::ModuleOutput { .. }) {
                direct.push(sink);
            } else {
                remaining.push(sink);
            }
        }
        ordered = remaining;
    }

    if ordered.is_empty() {
        return Ok(());
    }

    let source_timing = snapshot.combined.timing_for_net(root_net);
    let root_limit = characterized_load_limit(fanout, library, catalog, options)?;
    let mut level = ordered;
    let mut inserted = false;

    loop {
        if inserted {
            let total_count = direct.len() + level.len();
            let total_load = sum_sink_load(&direct);
            let level_load = sum_sink_load(&level);
            let root_load = CombinationalOutputLoad {
                rise: total_load.rise + level_load.rise,
                fall: total_load.fall + level_load.fall,
            };
            if total_count <= options.max_fanout
                && root_limit.is_none_or(|limit| max_load(root_load) <= limit + TIMING_EPSILON)
            {
                break;
            }
        }

        let previous_count = level.len();
        let previous_load = max_load(sum_sink_load(&level));
        let groups = partition_timing_sinks(level, options.max_fanout, root_limit);
        let mut next = Vec::with_capacity(groups.len());
        for group in groups {
            let buffer = select_timing_buffer(catalog, library, source_timing, &group)?;
            let output = group
                .sinks
                .iter()
                .find_map(|sink| match sink.target {
                    TimingSinkTarget::ModuleOutput { net, bit } => {
                        Some(reference_for_bit(net, bit, nets))
                    }
                    TimingSinkTarget::InstancePin { .. } => None,
                })
                .transpose()?
                .unwrap_or_else(|| NetRef::Simple(append_wire(module, nets, interner, names)));
            for sink in &group.sinks {
                if matches!(sink.target, TimingSinkTarget::InstancePin { .. }) {
                    reconnect_sink(module, sink.target, output.clone());
                }
            }
            let (instance_index, connection_index) = append_timing_buffer(
                module,
                interner,
                library,
                buffer,
                source.clone(),
                output,
                names,
            )?;
            next.push(TimingSink {
                target: TimingSinkTarget::InstancePin {
                    instance_index,
                    connection_index,
                },
                load: buffer.input_capacitances[0],
                criticality: group
                    .sinks
                    .iter()
                    .map(|sink| sink.criticality)
                    .fold(0.0, f64::max),
            });
            stats.buffers_inserted += 1;
            stats.area_added += buffer.area;
        }

        inserted = true;
        let next_load = max_load(sum_sink_load(&next));
        if next.len() >= previous_count && next_load >= previous_load - TIMING_EPSILON {
            break;
        }
        level = next;
    }

    if inserted {
        stats.buffered_nets += 1;
    }
    Ok(())
}

/// Groups critical sinks deterministically without joining two visible outputs.
fn partition_timing_sinks(
    sinks: Vec<TimingSink>,
    max_fanout: usize,
    target_load: Option<f64>,
) -> Vec<TimingSinkGroup> {
    let mut groups = Vec::<TimingSinkGroup>::new();
    for sink in sinks {
        let is_output = matches!(sink.target, TimingSinkTarget::ModuleOutput { .. });
        let eligible = groups
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                group.sinks.len() < max_fanout
                    && !(is_output && group.has_module_output)
                    && target_load.is_none_or(|limit| {
                        group.load.rise + sink.load.rise <= limit + TIMING_EPSILON
                            && group.load.fall + sink.load.fall <= limit + TIMING_EPSILON
                    })
            })
            .min_by(|(lhs_index, lhs), (rhs_index, rhs)| {
                max_load(lhs.load)
                    .total_cmp(&max_load(rhs.load))
                    .then_with(|| lhs_index.cmp(rhs_index))
            })
            .map(|(index, _)| index);
        let group_index = eligible.unwrap_or_else(|| {
            groups.push(TimingSinkGroup::default());
            groups.len() - 1
        });
        let group = &mut groups[group_index];
        group.load.rise += sink.load.rise;
        group.load.fall += sink.load.fall;
        group.has_module_output |= is_output;
        group.sinks.push(sink);
    }
    groups
}

/// Chooses the smallest buffer meeting ABC-style effort and real Liberty
/// timing.
fn select_timing_buffer<'a>(
    catalog: &'a CellCatalog,
    library: &Library,
    source_timing: Option<SignalTiming>,
    group: &TimingSinkGroup,
) -> Result<&'a CatalogCell> {
    let load = max_load(group.load);
    let required_input_capacitance = load / MAX_ELECTRICAL_EFFORT;
    let mut best: Option<(&CatalogCell, f64)> = None;
    let mut fallback: Option<(&CatalogCell, f64)> = None;
    for buffer in catalog.buffers() {
        if buffer
            .output_max_capacitance
            .is_some_and(|limit| load > limit + TIMING_EPSILON)
        {
            continue;
        }
        let delay = if let Some(timing) = source_timing {
            let cell = &library.cells[buffer.cell_index];
            let input = library.resolve_string(&cell.pins[buffer.input_pin_indices[0]].name);
            let output = &cell.pins[buffer.output_pin_index];
            let mut diagnostics = TimingQueryDiagnosticCounts::default();
            let result = evaluate_combinational_cell_output_timing(
                library,
                &cell.name,
                output,
                &[(input, timing)],
                group.load,
                &HashMap::new(),
                &mut diagnostics,
            )?;
            result.rise.arrival.max(result.fall.arrival)
        } else {
            buffer.nominal_delay
        };
        let input_capacitance = max_load(buffer.input_capacitances[0]);
        if input_capacitance + TIMING_EPSILON >= required_input_capacitance {
            let replace = best.is_none_or(|(current, current_delay)| {
                buffer
                    .area
                    .total_cmp(&current.area)
                    .then_with(|| delay.total_cmp(&current_delay))
                    .then_with(|| buffer.name.cmp(&current.name))
                    == Ordering::Less
            });
            if replace {
                best = Some((buffer, delay));
            }
        } else {
            let replace = fallback.is_none_or(|(current, current_delay)| {
                max_load(current.input_capacitances[0])
                    .total_cmp(&input_capacitance)
                    .then_with(|| buffer.area.total_cmp(&current.area))
                    .then_with(|| delay.total_cmp(&current_delay))
                    .then_with(|| buffer.name.cmp(&current.name))
                    == Ordering::Less
            });
            if replace {
                fallback = Some((buffer, delay));
            }
        }
    }
    best.or(fallback).map(|(buffer, _)| buffer).ok_or_else(|| {
        anyhow!("Liberty has no usable timing-characterized identity buffer for load {load}")
    })
}

/// Creates a collision-free internal scalar net for a buffer tree.
fn append_wire(
    module: &mut NetlistModule,
    nets: &mut Vec<Net>,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    names: &mut BufferNames,
) -> NetIndex {
    loop {
        let name = format!("n_buf_{}", names.wire);
        names.wire += 1;
        if interner.get(name.as_str()).is_some() {
            continue;
        }
        let index = NetIndex(nets.len());
        nets.push(Net {
            name: interner.get_or_intern(name.as_str()),
            width: None,
        });
        module.wires.push(index);
        module.net_index_range.end = nets.len();
        return index;
    }
}

/// Appends one characterized Liberty identity without flattening packed ports.
#[allow(clippy::too_many_arguments)]
fn append_timing_buffer(
    module: &mut NetlistModule,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    buffer: &CatalogCell,
    input: NetRef,
    output: NetRef,
    names: &mut BufferNames,
) -> Result<(usize, usize)> {
    let instance_name = loop {
        let candidate = format!("u_buf_{}", names.instance);
        names.instance += 1;
        if interner.get(candidate.as_str()).is_none() {
            break interner.get_or_intern(candidate.as_str());
        }
    };
    let cell = &library.cells[buffer.cell_index];
    let input_name = library.resolve_string(&cell.pins[buffer.input_pin_indices[0]].name);
    let output_name = library.resolve_string(&cell.pins[buffer.output_pin_index].name);
    let mut connections = vec![
        (interner.get_or_intern(input_name), input),
        (interner.get_or_intern(output_name), output),
    ];
    connections.sort_by(|lhs, rhs| {
        interner
            .resolve(lhs.0)
            .unwrap_or("")
            .cmp(interner.resolve(rhs.0).unwrap_or(""))
    });
    let connection_index = connections
        .iter()
        .position(|(pin, _)| interner.resolve(*pin) == Some(input_name))
        .ok_or_else(|| anyhow!("timing-driven buffer input connection is missing"))?;
    let instance_index = module.instances.len();
    module.instances.push(NetlistInstance {
        type_name: interner.get_or_intern(&buffer.name),
        instance_name,
        connections,
        inst_lineno: 1,
        inst_colno: 1,
    });
    Ok((instance_index, connection_index))
}

/// Reconnects a one-bit cell sink without changing a visible module port.
fn reconnect_sink(module: &mut NetlistModule, target: TimingSinkTarget, reference: NetRef) {
    if let TimingSinkTarget::InstancePin {
        instance_index,
        connection_index,
    } = target
    {
        module.instances[instance_index].connections[connection_index].1 = reference;
    }
}

/// Reconstructs a scalar or packed-bit connection from canonical connectivity.
fn reference_for_bit(net: NetIndex, bit: u32, nets: &[Net]) -> Result<NetRef> {
    let entry = nets
        .get(net.0)
        .ok_or_else(|| anyhow!("timing-driven buffer references an out-of-range net"))?;
    if entry.bit_offset(bit).is_none() {
        bail!("timing-driven buffer references an out-of-range packed bit");
    }
    Ok(if entry.width_bits() == 1 {
        NetRef::Simple(net)
    } else {
        NetRef::BitSelect(net, bit)
    })
}

/// Resolves the timing-constraint spelling of a scalar or packed output bit.
fn flattened_output_name(
    net: NetIndex,
    bit: u32,
    _module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<String> {
    let entry = nets
        .get(net.0)
        .ok_or_else(|| anyhow!("timing-driven buffering cannot resolve an output bit"))?;
    let name = interner
        .resolve(entry.name)
        .ok_or_else(|| anyhow!("timing-driven buffering cannot resolve an output name"))?;
    Ok(if entry.width_bits() == 1 {
        name.to_string()
    } else {
        format!("{name}_{bit}")
    })
}

/// Provides deterministic tie-breaking between otherwise identical sinks.
fn timing_sink_order(lhs: TimingSinkTarget, rhs: TimingSinkTarget) -> Ordering {
    match (lhs, rhs) {
        (
            TimingSinkTarget::InstancePin {
                instance_index: lhs_instance,
                connection_index: lhs_connection,
            },
            TimingSinkTarget::InstancePin {
                instance_index: rhs_instance,
                connection_index: rhs_connection,
            },
        ) => (lhs_instance, lhs_connection).cmp(&(rhs_instance, rhs_connection)),
        (TimingSinkTarget::InstancePin { .. }, TimingSinkTarget::ModuleOutput { .. }) => {
            Ordering::Less
        }
        (TimingSinkTarget::ModuleOutput { .. }, TimingSinkTarget::InstancePin { .. }) => {
            Ordering::Greater
        }
        (
            TimingSinkTarget::ModuleOutput {
                net: lhs,
                bit: lhs_bit,
            },
            TimingSinkTarget::ModuleOutput {
                net: rhs,
                bit: rhs_bit,
            },
        ) => (lhs.0, lhs_bit).cmp(&(rhs.0, rhs_bit)),
    }
}

/// Sums real rise/fall sink capacitances without treating limits as loads.
fn sum_sink_load(sinks: &[TimingSink]) -> CombinationalOutputLoad {
    sinks
        .iter()
        .fold(CombinationalOutputLoad::default(), |mut total, sink| {
            total.rise += sink.load.rise;
            total.fall += sink.load.fall;
            total
        })
}

/// Returns the conservative scalar of independently modeled edge loads.
fn max_load(load: CombinationalOutputLoad) -> f64 {
    load.rise.max(load.fall)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{
        BufferTimingConstraints, TimingSinkGroup, insert_timing_aware_buffers, select_timing_buffer,
    };
    use crate::liberty_model::{
        Cell, Library, LibraryBuilder, LuTableTemplate, Pin, PinDirection, Sequential,
        SequentialKind, TimingArc, TimingTable,
    };
    use crate::liberty_proto::TimingTableKind;
    use crate::netlist::buffer::BufferOptions;
    use crate::netlist::cell_catalog::CellCatalog;
    use crate::netlist::cell_catalog::test_utils::{parse_module, sizing_library};
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::parse::NetRef;
    use crate::netlist::report::build_netlist_report;
    use crate::netlist::sta::{CombinationalOutputLoad, StaOptions};
    use std::collections::BTreeMap;

    /// Builds complete scalar setup tables for a test capture register.
    fn scalar_table(
        builder: &mut LibraryBuilder,
        kind: TimingTableKind,
        value: f64,
    ) -> TimingTable {
        builder
            .add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![value], vec![], "")
            .expect("construct scalar registered-buffer timing table")
    }

    /// Creates a real, load-sensitive FF beside functionally indexed buffers.
    pub(crate) fn registered_timing_library() -> Library {
        let mut builder = LibraryBuilder::from_library(sizing_library());
        builder.lu_table_templates.push(LuTableTemplate {
            kind: "lu_table_template".to_string().into(),
            name: "buffer_test_register_load".to_string(),
            variable_1: "total_output_net_capacitance".to_string().into(),
            index_1: vec![0.0, 1.0],
            ..LuTableTemplate::default()
        });
        let mut clock_tables = Vec::new();
        for (kind, values) in [
            (TimingTableKind::CellRise, vec![0.5, 40.5]),
            (TimingTableKind::CellFall, vec![0.5, 40.5]),
            (TimingTableKind::RiseTransition, vec![0.1, 1.0]),
            (TimingTableKind::FallTransition, vec![0.1, 1.0]),
        ] {
            clock_tables.push(
                builder
                    .add_timing_table_f64(kind, 1, vec![], vec![], vec![], values, vec![2], "")
                    .expect("construct load-sensitive register output table"),
            );
        }
        let clock_arc = builder
            .add_timing_arc("CLK", "non_unate", "rising_edge", "", clock_tables)
            .expect("construct load-sensitive register launch arc");
        let setup_tables = vec![
            scalar_table(&mut builder, TimingTableKind::RiseConstraint, 0.25),
            scalar_table(&mut builder, TimingTableKind::FallConstraint, 0.25),
        ];
        let setup_arc: TimingArc = builder
            .add_timing_arc("CLK", "", "setup_rising", "", setup_tables)
            .expect("construct registered-buffer setup arc");
        let d = builder.intern_string("D").expect("intern D");
        let clk = builder.intern_string("CLK").expect("intern CLK");
        let q = builder.intern_string("Q").expect("intern Q");
        let iq = builder.intern_string("IQ").expect("intern IQ");
        builder.cells.push(Cell {
            name: "DFF".to_string(),
            pins: vec![
                Pin {
                    name: d,
                    direction: PinDirection::Input as i32,
                    capacitance: Some(0.12),
                    timing_arcs: vec![setup_arc],
                    ..Pin::default()
                },
                Pin {
                    name: clk,
                    direction: PinDirection::Input as i32,
                    capacitance: Some(0.01),
                    is_clocking_pin: true,
                    ..Pin::default()
                },
                Pin {
                    name: q,
                    direction: PinDirection::Output as i32,
                    function: iq,
                    max_capacitance: Some(0.8),
                    timing_arcs: vec![clock_arc],
                    ..Pin::default()
                },
            ],
            area: 4.0,
            sequential: vec![Sequential {
                state_var: "IQ".to_string(),
                next_state: "D".to_string(),
                clock_expr: "CLK".to_string(),
                kind: SequentialKind::Ff as i32,
                ..Sequential::default()
            }],
            ..Cell::default()
        });
        builder.finish()
    }

    /// Returns an eight-capture pipeline with a genuinely overloaded FF Q.
    pub(crate) fn high_fanout_register_source() -> &'static str {
        r#"
module top(clk, a, out);
  input clk;
  input a;
  output [7:0] out;
  wire root;
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
"#
    }

    #[test]
    fn chooses_smallest_buffer_with_sufficient_electrical_effort() {
        let library = registered_timing_library();
        let catalog = CellCatalog::new(&library).expect("catalog timing-complete test buffers");
        let group = TimingSinkGroup {
            load: CombinationalOutputLoad {
                rise: 0.72,
                fall: 0.72,
            },
            ..TimingSinkGroup::default()
        };

        let buffer = select_timing_buffer(&catalog, &library, None, &group)
            .expect("select smallest electrically sufficient test buffer");

        assert_eq!(buffer.name, "BUF");
    }

    #[test]
    fn increases_buffer_strength_when_small_buffer_exceeds_liberty_load() {
        let library = registered_timing_library();
        let catalog = CellCatalog::new(&library).expect("catalog timing-complete test buffers");
        let group = TimingSinkGroup {
            load: CombinationalOutputLoad {
                rise: 0.9,
                fall: 0.9,
            },
            ..TimingSinkGroup::default()
        };

        let buffer = select_timing_buffer(&catalog, &library, None, &group)
            .expect("select stronger buffer required by real Liberty load");

        assert_eq!(buffer.name, "BUF_FAST");
    }

    #[test]
    fn preserves_fanout_and_load_stats_when_no_buffer_is_needed() {
        let source = r#"
module top(clk, a, out);
  input clk;
  input a;
  output out;
  wire root;
  DFF launch (.CLK(clk), .D(a), .Q(root));
  DFF capture (.CLK(clk), .D(root), .Q(out));
endmodule
"#;
        let library = registered_timing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("leave a legally driven register path unchanged");

        assert_eq!(stats.buffers_inserted, 0);
        assert_eq!(stats.max_fanout_before, 1);
        assert_eq!(stats.max_fanout_after, stats.max_fanout_before);
        assert_eq!(stats.max_load_after, stats.max_load_before);
        assert_eq!(stats.final_worst_delay, stats.initial_worst_delay);
    }

    #[test]
    fn buffers_flip_flop_outputs_using_exact_register_path_timing() {
        let library = registered_timing_library();
        let (mut module, mut nets, mut interner) = parse_module(high_fanout_register_source());
        let before =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("time original high-fanout register path");

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 3,
                ..BufferOptions::default()
            },
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("buffer overloaded register-launch output");
        let after =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("independently time buffered register path");

        assert!(stats.buffers_inserted > 0);
        assert_eq!(stats.max_fanout_before, 8);
        assert!(stats.max_fanout_after <= 3);
        assert!(
            after.max_register_to_register_delay.unwrap()
                < before.max_register_to_register_delay.unwrap(),
            "buffering must improve exact register-to-register delay"
        );
        assert_eq!(
            module
                .instances
                .iter()
                .filter(|instance| interner.resolve(instance.type_name) == Some("DFF"))
                .count(),
            9,
            "buffering must preserve all original physical registers"
        );
        assert!(stats.timing_evaluations >= 2);
    }

    #[test]
    fn repairs_every_overloaded_root_without_spending_one_sta_per_root() {
        const ROOT_COUNT: usize = 72;
        const SINKS_PER_ROOT: usize = 4;
        let mut source = format!(
            "module top(a, out);\n  input a;\n  output [{}:0] out;\n  wire [{}:0] root;\n",
            ROOT_COUNT * SINKS_PER_ROOT - 1,
            ROOT_COUNT - 1
        );
        for root in 0..ROOT_COUNT {
            source.push_str(&format!("  BUF driver{root} (.A(a), .Y(root[{root}]));\n"));
            for sink in 0..SINKS_PER_ROOT {
                let output = root * SINKS_PER_ROOT + sink;
                source.push_str(&format!(
                    "  BUF sink{output} (.A(root[{root}]), .Y(out[{output}]));\n"
                ));
            }
        }
        source.push_str("endmodule\n");
        let library = registered_timing_library();
        let (mut module, mut nets, mut interner) = parse_module(&source);

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 3,
                ..BufferOptions::default()
            },
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("repair all independent hard-overload roots in one fair batch");

        assert_eq!(stats.max_fanout_before, SINKS_PER_ROOT);
        assert!(stats.max_fanout_after <= 3);
        assert_eq!(stats.buffered_nets, ROOT_COUNT);
        assert_eq!(stats.buffers_inserted, ROOT_COUNT);
        assert_eq!(stats.timing_evaluations, 2);
        assert_eq!(stats.unresolved_overloaded_nets, 0);
    }

    #[test]
    fn hard_electrical_repair_never_violates_explicit_output_deadlines() {
        let source = r#"
module top(a, out);
  input a;
  output [3:0] out;
  wire root;
  BUF driver (.A(a), .Y(root));
  BUF sink0 (.A(root), .Y(out[0]));
  BUF sink1 (.A(root), .Y(out[1]));
  BUF sink2 (.A(root), .Y(out[2]));
  BUF sink3 (.A(root), .Y(out[3]));
endmodule
"#;
        let library = registered_timing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        let constraints = BufferTimingConstraints {
            primary_output_required: BTreeMap::from_iter(
                (0..4).map(|index| (format!("out_{index}"), 8.0)),
            ),
            ..BufferTimingConstraints::default()
        };

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 3,
                ..BufferOptions::default()
            },
            StaOptions::default(),
            &constraints,
        )
        .expect("retain valid explicit deadlines when buffering cannot preserve them");

        assert_eq!(stats.buffers_inserted, 0);
        assert_eq!(stats.max_fanout_after, 4);
        assert_eq!(stats.final_worst_delay, stats.initial_worst_delay);
    }

    #[test]
    fn protects_every_clock_pin_from_buffer_insertion() {
        let library = registered_timing_library();
        let (mut module, mut nets, mut interner) = parse_module(high_fanout_register_source());
        let clock = module
            .find_net_index(interner.get("clk").expect("resolve clk"), &nets)
            .expect("resolve clock net");
        insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 2,
                buffer_primary_inputs: true,
                ..BufferOptions::default()
            },
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("buffer register data without touching the clock");

        for instance in &module.instances {
            if interner
                .resolve(instance.type_name)
                .is_some_and(|name| name.starts_with("BUF"))
            {
                assert!(
                    instance
                        .connections
                        .iter()
                        .all(|(_, net)| !matches!(net, NetRef::Simple(net) if *net == clock)),
                    "a data buffer must never be inserted on the clock"
                );
            }
        }
    }

    #[test]
    fn preserves_packed_ports_and_zero_area_constant_bits() {
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
        let library = registered_timing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 3,
                ..BufferOptions::default()
            },
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("buffer logic without replacing packed constant outputs");

        assert!(stats.buffers_inserted > 0);
        assert_eq!(module.assigns.len(), 1);
        let rendered = emit_module_as_netlist_text(&module, &nets, &interner)
            .expect("emit buffered packed output");
        assert!(rendered.contains("output [8:0] out;"));
        assert!(rendered.contains("assign out[8] = 1'b0;"));
    }

    #[test]
    fn registered_buffer_output_is_deterministic() {
        let library = registered_timing_library();
        let render = || {
            let (mut module, mut nets, mut interner) = parse_module(high_fanout_register_source());
            insert_timing_aware_buffers(
                &mut module,
                &mut nets,
                &mut interner,
                &library,
                &BufferOptions {
                    max_fanout: 3,
                    ..BufferOptions::default()
                },
                StaOptions::default(),
                &BufferTimingConstraints::default(),
            )
            .expect("buffer test netlist");
            emit_module_as_netlist_text(&module, &nets, &interner)
                .expect("emit deterministic buffered design")
        };

        assert_eq!(render(), render());
    }

    #[test]
    fn rejects_overloaded_designs_without_usable_liberty_buffers() {
        let mut builder = LibraryBuilder::from_library(registered_timing_library());
        builder.cells.retain(|cell| !cell.name.starts_with("BUF"));
        let library = builder.finish();
        let (mut module, mut nets, mut interner) = parse_module(high_fanout_register_source());
        let error = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 3,
                ..BufferOptions::default()
            },
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect_err("an overloaded design requires an actual Liberty buffer");

        assert!(error.to_string().contains("no usable buffer"));
    }

    #[test]
    fn rejects_invalid_electrical_constraints_before_editing() {
        let library = registered_timing_library();
        let (mut module, mut nets, mut interner) = parse_module(high_fanout_register_source());
        let initial_instances = module.instances.len();
        let error = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 1,
                ..BufferOptions::default()
            },
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect_err("one-way buffer trees cannot terminate");

        assert!(error.to_string().contains("at least 2"));
        assert_eq!(module.instances.len(), initial_instances);
    }
}
