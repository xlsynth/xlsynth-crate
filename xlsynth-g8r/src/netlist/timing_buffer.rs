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
    boundary_timing_applies_to_module_output, boundary_timing_applies_to_primary_input,
    effective_input_capacitance_for_mapping, effective_representative_driver,
    evaluate_combinational_cell_output_timing, is_sequential_boundary_cell,
    resolved_module_output_load,
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
/// Keep spare stage capacity for resizing without rebuilding full trees.
const PREFERRED_BUFFER_STAGE_FANOUT: usize = 10;
/// Critical-path trials can profitably isolate loads below electrical limits.
const TIMING_DRIVEN_EFFORT_FRACTION: f64 = 0.5;
/// Bound speculative complete-STA work to actual near-critical paths.
const MIN_CRITICAL_PATH_FRACTION: f64 = 0.9;
/// Avoid exploring every fanout on large speculative timing rounds.
const MAX_OPPORTUNISTIC_ROOTS: usize = 16;
/// A timing-selected buffer must still reduce its parent driver's load.
const MIN_TIMING_BUFFER_EFFORT: f64 = 2.0;
/// Extremely slow directly exposed outputs can also delay their internal users.
const MIN_SHARED_OUTPUT_SLEW_FRACTION: f64 = 0.5;
/// Bound exact timing work spent recovering redundant sibling-buffer area.
const MAX_BUFFER_CONSOLIDATION_EVALUATIONS: usize = 16;
/// Avoid repeatedly rebuilding very large buffer trees during area recovery.
const MAX_BUFFER_CONSOLIDATION_MOVES: usize = 4;
/// Try only the fastest pin-compatible shared strengths for one sibling pair.
const MAX_BUFFER_CONSOLIDATION_STRENGTHS: usize = 3;
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
    fanout_load: f64,
    max_transition: Option<f64>,
    criticality: f64,
}

#[derive(Clone, Copy, Debug)]
struct TimingDriver {
    instance_index: usize,
    connection_index: usize,
    cell_index: usize,
    pin_index: usize,
}

/// Characterized output pin of either a real gate or a virtual input driver.
#[derive(Clone, Copy, Debug)]
struct TimingDriverPin {
    cell_index: usize,
    pin_index: usize,
}

#[derive(Clone, Debug, Default)]
struct TimingFanout {
    sinks: Vec<TimingSink>,
    driver: Option<TimingDriver>,
    representative_driver: Option<TimingDriverPin>,
    is_primary_input: bool,
    protected_clock: bool,
    constant: bool,
}

impl TimingFanout {
    /// Returns the real or scope-valid virtual Liberty output driving this net.
    fn electrical_driver(&self) -> Option<TimingDriverPin> {
        self.driver
            .map(|driver| TimingDriverPin {
                cell_index: driver.cell_index,
                pin_index: driver.pin_index,
            })
            .or(self.representative_driver)
    }
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
    fanout_load: f64,
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

/// Exact-timing diagnostics for accepted sibling-buffer consolidation.
#[derive(Clone, Debug, Default)]
pub(crate) struct BufferConsolidationStats {
    pub buffers_removed: usize,
    pub area_recovered: f64,
    pub timing_evaluations: usize,
    pub final_delay: f64,
    pub max_fanout_after: usize,
    pub max_load_after: f64,
    pub unresolved_overloaded_nets: usize,
}

/// One same-parent inserted-buffer pair and its legal shared strengths.
#[derive(Clone, Debug)]
struct BufferConsolidationCandidate {
    keep_instance: usize,
    remove_instance: usize,
    keep_output: usize,
    remove_output: usize,
    combined_area: f64,
    criticality: f64,
    replacements: Vec<BufferConsolidationReplacement>,
}

/// One pin-compatible shared-buffer drive strength and local timing estimate.
#[derive(Clone, Debug)]
struct BufferConsolidationReplacement {
    name: String,
    area: f64,
    predicted_delay: f64,
}

/// Best exact-timed replacement encountered for one sibling-buffer pair.
#[derive(Clone)]
struct AcceptedBufferConsolidation {
    module: NetlistModule,
    interner: StringInterner<StringBackend<SymbolU32>>,
    delay: f64,
    area_recovered: f64,
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
    insert_timing_aware_buffers_with_strategy(
        module,
        nets,
        interner,
        library,
        options,
        sta_options,
        constraints,
        false,
    )
}

/// Refreshes final electrical diagnostics using an already-computed STA report.
pub(crate) fn refresh_timing_buffer_diagnostics(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
    timing: &StaReport,
    stats: &mut BufferStats,
) -> Result<()> {
    let graph = build_electrical_timing_graph(module, nets, interner, library, options)?;
    let catalog = CellCatalog::new(library)?;
    stats.max_fanout_after = eligible_fanouts(&graph, options)
        .map(|fanout| fanout.sinks.len())
        .max()
        .unwrap_or(0);
    stats.max_load_after = eligible_fanouts(&graph, options)
        .map(|fanout| max_load(sum_sink_load(&fanout.sinks)))
        .fold(0.0, f64::max);
    stats.unresolved_overloaded_nets =
        count_unresolved_overloaded_nets(&graph, library, &catalog, options, timing)?;
    stats.final_worst_delay = Some(timing.worst_output_arrival);
    Ok(())
}

/// Explores critical fanouts inside an independently validated trial netlist.
pub(crate) fn insert_speculative_timing_aware_buffers(
    module: &mut NetlistModule,
    nets: &mut Vec<Net>,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
    sta_options: StaOptions,
    constraints: &BufferTimingConstraints,
) -> Result<BufferStats> {
    insert_timing_aware_buffers_with_strategy(
        module,
        nets,
        interner,
        library,
        options,
        sta_options,
        constraints,
        true,
    )
}

/// Detects a near-critical, slow output shared with an internal data sink.
pub(crate) fn has_slow_shared_primary_output(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
    sta_options: StaOptions,
) -> Result<bool> {
    let output_load = resolved_module_output_load(library, sta_options)?;
    if output_load.rise <= 0.0 && output_load.fall <= 0.0 {
        return Ok(false);
    }

    // Connectivity is much cheaper than Liberty STA; most small circuits have
    // no output also consumed internally and need no extra timing analysis.
    let normalized = NormalizedNetlistModule::new(module, nets, interner)
        .context("normalizing shared primary-output buffer candidates")?;
    let output_bits = normalized
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
        .flat_map(|port| port.bits.iter().copied())
        .collect::<BTreeSet<_>>();
    if output_bits.is_empty() {
        return Ok(false);
    }
    let mut binding_counts = vec![0usize; normalized.bit_count()];
    let mut shared_output = false;
    for instance in &normalized.instances {
        for connection in &instance.connections {
            for bit in &connection.bits {
                if let BitSource::Bit(bit) = bit
                    && output_bits.contains(bit)
                {
                    binding_counts[*bit] += 1;
                    shared_output |= binding_counts[*bit] >= 2;
                }
            }
        }
    }
    if !shared_output {
        return Ok(false);
    }

    let constraints = BufferTimingConstraints::default();
    let snapshot =
        analyze_timing_snapshot(module, nets, interner, library, sta_options, &constraints)?;
    let graph = build_timing_graph(
        module,
        nets,
        interner,
        library,
        options,
        &constraints,
        &snapshot,
    )?;
    if graph.instances.iter().any(|instance| instance.sequential) {
        return Ok(false);
    }
    let catalog = CellCatalog::new(library)?;
    let minimum_transition = worst_path_delay(&snapshot.report) * MIN_SHARED_OUTPUT_SLEW_FRACTION;

    for (root, fanout) in graph.fanouts.iter().enumerate() {
        if !fanout_eligible(fanout, options)
            || fanout.driver.is_none()
            || !fanout
                .sinks
                .iter()
                .any(|sink| matches!(sink.target, TimingSinkTarget::ModuleOutput { .. }))
            || !fanout
                .sinks
                .iter()
                .any(|sink| matches!(sink.target, TimingSinkTarget::InstancePin { .. }))
        {
            continue;
        }
        let Some(timing) = snapshot.combined.timing_for_net(graph.bits[root].0) else {
            continue;
        };
        if timing.rise.transition.max(timing.fall.transition) + TIMING_EPSILON < minimum_transition
        {
            continue;
        }
        if fanout_is_buffer_candidate(&graph, root, library, &catalog, options, &snapshot, true)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Separates conservative production buffering from speculative timing trials.
#[allow(clippy::too_many_arguments)]
fn insert_timing_aware_buffers_with_strategy(
    module: &mut NetlistModule,
    nets: &mut Vec<Net>,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
    sta_options: StaOptions,
    constraints: &BufferTimingConstraints,
    speculative: bool,
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

    let mut roots = overloaded_roots(
        &initial_graph,
        library,
        &catalog,
        options,
        &snapshot,
        speculative,
    )?;
    if roots.is_empty() {
        stats.max_fanout_after = stats.max_fanout_before;
        stats.max_load_after = stats.max_load_before;
        stats.unresolved_overloaded_nets = count_unresolved_overloaded_nets(
            &initial_graph,
            library,
            &catalog,
            options,
            &snapshot.combined,
        )?;
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
                || exceeds_weighted_fanout_limit(fanout, library)
                || exceeds_representative_transition_limit(
                    fanout,
                    library,
                    &catalog,
                    snapshot.combined.timing_for_bit(*root),
                )
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
                speculative,
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
                    roots = overloaded_roots(
                        &graph,
                        library,
                        &catalog,
                        options,
                        &snapshot,
                        speculative,
                    )?;
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
                || !fanout_is_buffer_candidate(
                    &graph,
                    *root,
                    library,
                    &catalog,
                    options,
                    &snapshot,
                    speculative,
                )?
            {
                continue;
            }
            fixes_hard_violation |= fanout.sinks.len() > options.max_fanout
                || exceeds_characterized_output_limit(fanout, library, options)
                || exceeds_weighted_fanout_limit(fanout, library)
                || exceeds_representative_transition_limit(
                    fanout,
                    library,
                    &catalog,
                    snapshot.combined.timing_for_bit(*root),
                );
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
                speculative,
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
    stats.unresolved_overloaded_nets = count_unresolved_overloaded_nets(
        &final_graph,
        library,
        &catalog,
        options,
        &snapshot.combined,
    )?;
    Ok(stats)
}

/// Merges equivalent sibling buffers only when exact final timing is preserved.
pub(crate) fn consolidate_timing_aware_buffers(
    module: &mut NetlistModule,
    nets: &[Net],
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
    sta_options: StaOptions,
    achieved_delay: f64,
) -> Result<BufferConsolidationStats> {
    let constraints = BufferTimingConstraints::default();
    validate_timing_options(options, sta_options, &constraints)?;
    let catalog = CellCatalog::new(library)?;
    let mut stats = BufferConsolidationStats {
        final_delay: achieved_delay,
        ..BufferConsolidationStats::default()
    };

    while stats.buffers_removed < MAX_BUFFER_CONSOLIDATION_MOVES
        && stats.timing_evaluations < MAX_BUFFER_CONSOLIDATION_EVALUATIONS
    {
        let snapshot =
            analyze_timing_snapshot(module, nets, interner, library, sta_options, &constraints)?;
        let graph = build_timing_graph(
            module,
            nets,
            interner,
            library,
            options,
            &constraints,
            &snapshot,
        )?;
        if graph.instances.iter().any(|instance| instance.sequential) {
            break;
        }
        let candidates = sibling_buffer_consolidations(
            module, interner, library, &catalog, options, &snapshot, &graph,
        )?;
        if candidates.is_empty() {
            break;
        }

        let mut accepted = None::<AcceptedBufferConsolidation>;
        for candidate in candidates {
            for replacement in candidate
                .replacements
                .iter()
                .take(MAX_BUFFER_CONSOLIDATION_STRENGTHS)
            {
                if stats.timing_evaluations == MAX_BUFFER_CONSOLIDATION_EVALUATIONS {
                    break;
                }
                let mut trial_module = module.clone();
                let mut trial_interner = interner.clone();
                let keep_output = reference_for_bit(
                    graph.bits[candidate.keep_output].0,
                    graph.bits[candidate.keep_output].1,
                    nets,
                )?;
                for sink in &graph.fanouts[candidate.remove_output].sinks {
                    reconnect_sink(&mut trial_module, sink.target, keep_output.clone());
                }
                trial_module.instances[candidate.keep_instance].type_name =
                    trial_interner.get_or_intern(replacement.name.as_str());
                trial_module.instances.remove(candidate.remove_instance);
                let timing = analyze_combinational_max_arrival_with_primary_input_arrivals(
                    &trial_module,
                    nets,
                    &trial_interner,
                    library,
                    sta_options,
                    &BTreeMap::new(),
                );
                stats.timing_evaluations += 1;
                let trial_delay = match timing {
                    Ok(report) => report.worst_output_arrival,
                    Err(error) => {
                        log::debug!("rejecting sibling buffer consolidation: {error:#}");
                        continue;
                    }
                };
                if trial_delay > stats.final_delay + TIMING_EPSILON {
                    continue;
                }
                let area_recovered = candidate.combined_area - replacement.area;
                let improves = accepted.as_ref().is_none_or(|best| {
                    trial_delay + TIMING_EPSILON < best.delay
                        || (trial_delay - best.delay).abs() <= TIMING_EPSILON
                            && area_recovered > best.area_recovered + TIMING_EPSILON
                });
                if improves {
                    accepted = Some(AcceptedBufferConsolidation {
                        module: trial_module,
                        interner: trial_interner,
                        delay: trial_delay,
                        area_recovered,
                    });
                }
            }
            if accepted.is_some()
                || stats.timing_evaluations == MAX_BUFFER_CONSOLIDATION_EVALUATIONS
            {
                break;
            }
        }

        let Some(best) = accepted else {
            break;
        };
        *module = best.module;
        *interner = best.interner;
        stats.buffers_removed += 1;
        stats.area_recovered += best.area_recovered;
        stats.final_delay = best.delay;
    }

    if stats.buffers_removed > 0 {
        let snapshot =
            analyze_timing_snapshot(module, nets, interner, library, sta_options, &constraints)?;
        let graph = build_timing_graph(
            module,
            nets,
            interner,
            library,
            options,
            &constraints,
            &snapshot,
        )?;
        stats.final_delay = worst_path_delay(&snapshot.report);
        stats.max_fanout_after = eligible_fanouts(&graph, options)
            .map(|fanout| fanout.sinks.len())
            .max()
            .unwrap_or(0);
        stats.max_load_after = eligible_fanouts(&graph, options)
            .map(|fanout| max_load(sum_sink_load(&fanout.sinks)))
            .fold(0.0, f64::max);
        stats.unresolved_overloaded_nets = count_unresolved_overloaded_nets(
            &graph,
            library,
            &catalog,
            options,
            &snapshot.combined,
        )?;
    }
    Ok(stats)
}

/// Enumerates deterministic, pin-compatible same-parent buffer replacements.
#[allow(clippy::too_many_arguments)]
fn sibling_buffer_consolidations(
    module: &NetlistModule,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    catalog: &CellCatalog,
    options: &BufferOptions,
    snapshot: &TimingSnapshot,
    graph: &TimingGraph,
) -> Result<Vec<BufferConsolidationCandidate>> {
    let mut candidates = Vec::new();
    for (root, fanout) in graph.fanouts.iter().enumerate() {
        if fanout.protected_clock || fanout.constant {
            continue;
        }
        let mut siblings = fanout
            .sinks
            .iter()
            .filter_map(|sink| {
                let TimingSinkTarget::InstancePin { instance_index, .. } = sink.target else {
                    return None;
                };
                let instance = &module.instances[instance_index];
                let name = interner.resolve(instance.instance_name)?;
                let cell = catalog.by_name(interner.resolve(instance.type_name)?)?;
                let output = graph.instances[instance_index]
                    .outputs
                    .as_slice()
                    .first()
                    .copied()?;
                if !name.starts_with("u_buf_")
                    || !cell.is_buffer()
                    || graph.instances[instance_index].outputs.len() != 1
                    || graph.fanouts[output].protected_clock
                    || graph.fanouts[output]
                        .sinks
                        .iter()
                        .any(|sink| matches!(sink.target, TimingSinkTarget::ModuleOutput { .. }))
                {
                    return None;
                }
                Some(instance_index)
            })
            .collect::<Vec<_>>();
        siblings.sort_unstable();
        siblings.dedup();

        for pair in siblings.windows(2) {
            let keep_instance = pair[0];
            let remove_instance = pair[1];
            let keep_output = graph.instances[keep_instance].outputs[0];
            let remove_output = graph.instances[remove_instance].outputs[0];
            let keep_fanout = &graph.fanouts[keep_output];
            let remove_fanout = &graph.fanouts[remove_output];
            if keep_fanout.sinks.len() + remove_fanout.sinks.len() > options.max_fanout {
                continue;
            }
            let keep_name = interner
                .resolve(module.instances[keep_instance].type_name)
                .ok_or_else(|| anyhow!("cannot resolve a sibling buffer type"))?;
            let remove_name = interner
                .resolve(module.instances[remove_instance].type_name)
                .ok_or_else(|| anyhow!("cannot resolve a sibling buffer type"))?;
            let keep_cell = catalog
                .by_name(keep_name)
                .ok_or_else(|| anyhow!("cannot classify a sibling buffer type"))?;
            let remove_cell = catalog
                .by_name(remove_name)
                .ok_or_else(|| anyhow!("cannot classify a sibling buffer type"))?;
            let combined_area = keep_cell.area + remove_cell.area;
            let keep_load = sum_sink_load(&keep_fanout.sinks);
            let remove_load = sum_sink_load(&remove_fanout.sinks);
            let combined_load = CombinationalOutputLoad {
                rise: keep_load.rise + remove_load.rise,
                fall: keep_load.fall + remove_load.fall,
            };
            let combined_fanout_load = sum_sink_fanout_load(&keep_fanout.sinks)
                + sum_sink_fanout_load(&remove_fanout.sinks);
            let sink_transition_limit = keep_fanout
                .sinks
                .iter()
                .chain(&remove_fanout.sinks)
                .filter_map(|sink| sink.max_transition)
                .reduce(f64::min);
            if options.target_load.is_some_and(|limit| {
                combined_load.rise > limit + TIMING_EPSILON
                    || combined_load.fall > limit + TIMING_EPSILON
            }) {
                continue;
            }
            let old_parent_load = CombinationalOutputLoad {
                rise: keep_cell.input_capacitances[0].rise + remove_cell.input_capacitances[0].rise,
                fall: keep_cell.input_capacitances[0].fall + remove_cell.input_capacitances[0].fall,
            };
            let old_parent_fanout_load = library.cells[keep_cell.cell_index].pins
                [keep_cell.input_pin_indices[0]]
                .fanout_load
                .unwrap_or(0.0)
                + library.cells[remove_cell.cell_index].pins[remove_cell.input_pin_indices[0]]
                    .fanout_load
                    .unwrap_or(0.0);
            let source_timing = snapshot.combined.timing_for_net(graph.bits[root].0);
            let mut replacements = Vec::new();
            for replacement in catalog.family(keep_cell) {
                if !replacement.is_buffer() {
                    continue;
                }
                let cell = &library.cells[replacement.cell_index];
                let output = &cell.pins[replacement.output_pin_index];
                let input = &cell.pins[replacement.input_pin_indices[0]];
                if replacement.area + TIMING_EPSILON >= combined_area
                    || replacement.input_capacitances[0].rise
                        > old_parent_load.rise + TIMING_EPSILON
                    || replacement.input_capacitances[0].fall
                        > old_parent_load.fall + TIMING_EPSILON
                    || replacement
                        .output_max_capacitance
                        .is_some_and(|limit| max_load(combined_load) > limit + TIMING_EPSILON)
                    || input.fanout_load.unwrap_or(0.0) > old_parent_fanout_load + TIMING_EPSILON
                    || output
                        .max_fanout
                        .is_some_and(|limit| combined_fanout_load > limit + TIMING_EPSILON)
                {
                    continue;
                }
                let predicted_delay = if let Some(timing) = source_timing {
                    let input = library.resolve_string(&input.name);
                    let mut diagnostics = TimingQueryDiagnosticCounts::default();
                    let timing = evaluate_combinational_cell_output_timing(
                        library,
                        &cell.name,
                        output,
                        &[(input, timing)],
                        combined_load,
                        &HashMap::new(),
                        &mut diagnostics,
                    )?;
                    let transition_limit = match (output.max_transition, sink_transition_limit) {
                        (Some(output), Some(sink)) => Some(output.min(sink)),
                        (Some(output), None) => Some(output),
                        (None, sink) => sink,
                    };
                    if transition_limit.is_some_and(|limit| {
                        timing.rise.transition.max(timing.fall.transition) > limit + TIMING_EPSILON
                    }) {
                        continue;
                    }
                    timing.rise.arrival.max(timing.fall.arrival)
                } else {
                    replacement.nominal_delay
                };
                replacements.push(BufferConsolidationReplacement {
                    name: replacement.name.clone(),
                    area: replacement.area,
                    predicted_delay,
                });
            }
            replacements.sort_by(|lhs, rhs| {
                lhs.predicted_delay
                    .total_cmp(&rhs.predicted_delay)
                    .then_with(|| lhs.area.total_cmp(&rhs.area))
                    .then_with(|| lhs.name.cmp(&rhs.name))
            });
            if replacements.is_empty() {
                continue;
            }
            candidates.push(BufferConsolidationCandidate {
                keep_instance,
                remove_instance,
                keep_output,
                remove_output,
                combined_area,
                criticality: graph.departures[keep_output].max(graph.departures[remove_output]),
                replacements,
            });
        }
    }
    candidates.sort_by(|lhs, rhs| {
        rhs.criticality
            .total_cmp(&lhs.criticality)
            .then_with(|| lhs.keep_instance.cmp(&rhs.keep_instance))
            .then_with(|| lhs.remove_instance.cmp(&rhs.remove_instance))
    });
    Ok(candidates)
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
    let mut graph = build_electrical_timing_graph(module, nets, interner, library, options)?;
    compute_downstream_criticality(&mut graph, module, nets, interner, constraints, snapshot)?;
    Ok(graph)
}

/// Builds exact per-bit electrical connectivity without another STA pass.
fn build_electrical_timing_graph(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
) -> Result<TimingGraph> {
    validate_constant_output_assignments(module, nets)?;
    let output_load = resolved_module_output_load(
        library,
        StaOptions {
            module_output_load: options.module_output_load,
            ..StaOptions::default()
        },
    )?;
    let normalized = NormalizedNetlistModule::new(module, nets, interner)
        .context("normalizing timing-driven buffer connectivity")?;
    let cells: HashMap<&str, usize> = library
        .cells
        .iter()
        .enumerate()
        .map(|(index, cell)| (cell.name.as_str(), index))
        .collect();
    let representative_driver = effective_representative_driver(library)?
        .map(|driver| -> Result<TimingDriverPin> {
            let cell_index = cells
                .get(driver.cell.name.as_str())
                .copied()
                .ok_or_else(|| {
                    anyhow!(
                        "representative input driver '{}' is absent from the timing library",
                        driver.cell.name
                    )
                })?;
            let pin_index = library.cells[cell_index]
                .pins
                .iter()
                .position(|pin| std::ptr::eq(pin, driver.output_pin))
                .ok_or_else(|| {
                    anyhow!(
                        "representative input driver '{}' has no characterized output pin",
                        driver.cell.name
                    )
                })?;
            Ok(TimingDriverPin {
                cell_index,
                pin_index,
            })
        })
        .transpose()?;
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
                PortDirection::Input => {
                    fanouts[bit].is_primary_input = true;
                    if let Some(driver) = representative_driver {
                        let port_name = interner.resolve(port.name).ok_or_else(|| {
                            anyhow!("timing-driven buffering cannot resolve an input port name")
                        })?;
                        let input_name = if port.bits.len() == 1 {
                            port_name.to_string()
                        } else {
                            let bit_number =
                                nets[port_net.0].bit_number(offset).ok_or_else(|| {
                                    anyhow!(
                                        "timing-driven buffering found an invalid packed input bit"
                                    )
                                })?;
                            format!("{port_name}_{bit_number}")
                        };
                        if boundary_timing_applies_to_primary_input(input_name.as_str()) {
                            fanouts[bit].representative_driver = Some(driver);
                        }
                    }
                }
                PortDirection::Output => {
                    let bit_number = nets[port_net.0].bit_number(offset).ok_or_else(|| {
                        anyhow!("timing-driven buffering found an invalid packed output bit")
                    })?;
                    let output_name =
                        flattened_output_name(port_net, bit_number, module, nets, interner)?;
                    let load = if boundary_timing_applies_to_module_output(output_name.as_str()) {
                        output_load
                    } else {
                        CombinationalOutputLoad::default()
                    };
                    fanouts[bit].sinks.push(TimingSink {
                        target: TimingSinkTarget::ModuleOutput {
                            net: port_net,
                            bit: bit_number,
                        },
                        load,
                        fanout_load: 0.0,
                        max_transition: None,
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
                    fanout_load: pin.fanout_load.unwrap_or(0.0),
                    max_transition: pin.max_transition,
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

    Ok(TimingGraph {
        fanouts,
        bits,
        instances,
        departures: vec![0.0; normalized.bit_count()],
    })
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
    speculative: bool,
) -> Result<Vec<usize>> {
    let mut roots = Vec::new();
    for (index, fanout) in graph.fanouts.iter().enumerate() {
        if !fanout_eligible(fanout, options)
            || !fanout_is_buffer_candidate(
                graph,
                index,
                library,
                catalog,
                options,
                snapshot,
                speculative,
            )?
        {
            continue;
        }
        let overloaded = fanout_is_overloaded(fanout, library, catalog, options)?
            || exceeds_representative_transition_limit(
                fanout,
                library,
                catalog,
                snapshot.combined.timing_for_bit(index),
            );
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
            || exceeds_characterized_output_limit(fanout, library, options)
            || exceeds_weighted_fanout_limit(fanout, library)
            || exceeds_representative_transition_limit(
                fanout,
                library,
                catalog,
                snapshot.combined.timing_for_bit(index),
            );
        roots.push((
            index,
            hard,
            fanout_ratio.max(load_ratio),
            priority,
            overloaded,
        ));
    }
    roots.sort_by(|lhs, rhs| {
        rhs.1
            .cmp(&lhs.1)
            .then_with(|| {
                if speculative {
                    rhs.3.total_cmp(&lhs.3)
                } else {
                    rhs.2.total_cmp(&lhs.2)
                }
            })
            .then_with(|| {
                if speculative {
                    rhs.2.total_cmp(&lhs.2)
                } else {
                    rhs.3.total_cmp(&lhs.3)
                }
            })
            .then_with(|| lhs.0.cmp(&rhs.0))
    });
    let mut opportunistic_roots = 0;
    Ok(roots
        .into_iter()
        .filter_map(|(index, _, _, _, overloaded)| {
            if !overloaded {
                if opportunistic_roots == MAX_OPPORTUNISTIC_ROOTS {
                    return None;
                }
                opportunistic_roots += 1;
            }
            Some(index)
        })
        .collect())
}

/// Returns only legal, driven data roots; clocks and constants are protected.
fn fanout_eligible(fanout: &TimingFanout, options: &BufferOptions) -> bool {
    let driven_primary_input = fanout.is_primary_input
        && (fanout.representative_driver.is_some() || options.buffer_primary_inputs);
    !fanout.protected_clock
        && !fanout.constant
        && !fanout.sinks.is_empty()
        && (fanout.driver.is_some() || driven_primary_input)
        && (!fanout.is_primary_input || driven_primary_input)
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

/// Counts remaining count, capacitive, weighted-fanout, and slew violations.
fn count_unresolved_overloaded_nets(
    graph: &TimingGraph,
    library: &Library,
    catalog: &CellCatalog,
    options: &BufferOptions,
    timing: &StaReport,
) -> Result<usize> {
    graph
        .fanouts
        .iter()
        .enumerate()
        .filter(|(_, fanout)| fanout_eligible(fanout, options))
        .try_fold(0usize, |count, (root, fanout)| -> Result<usize> {
            let overloaded = fanout_is_overloaded(fanout, library, catalog, options)?
                || exceeds_transition_limit(fanout, library, timing.timing_for_bit(root));
            Ok(count + usize::from(overloaded))
        })
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
    if exceeds_weighted_fanout_limit(fanout, library) {
        return Ok(true);
    }
    let load = max_load(sum_sink_load(&fanout.sinks));
    if let Some(limit) = characterized_load_limit(fanout, library, catalog, options)? {
        Ok(load > limit + TIMING_EPSILON)
    } else {
        Ok(false)
    }
}

/// Exposes heavily loaded critical fanouts only to speculative trial passes.
#[allow(clippy::too_many_arguments)]
fn fanout_is_buffer_candidate(
    graph: &TimingGraph,
    root: usize,
    library: &Library,
    catalog: &CellCatalog,
    options: &BufferOptions,
    snapshot: &TimingSnapshot,
    speculative: bool,
) -> Result<bool> {
    let fanout = &graph.fanouts[root];
    if fanout.representative_driver.is_some()
        && fanout.sinks.len() == 1
        && hard_output_load_limit(fanout, library, options)
            .is_some_and(|limit| max_load(sum_sink_load(&fanout.sinks)) > limit + TIMING_EPSILON)
        && !representative_driver_load_can_be_reduced(fanout, catalog)
    {
        return Ok(false);
    }
    if fanout_is_overloaded(fanout, library, catalog, options)?
        || exceeds_representative_transition_limit(
            fanout,
            library,
            catalog,
            snapshot.combined.timing_for_bit(root),
        )
    {
        return Ok(true);
    }
    if !speculative {
        return Ok(false);
    }

    let drives_visible_output = fanout
        .sinks
        .iter()
        .any(|sink| matches!(sink.target, TimingSinkTarget::ModuleOutput { .. }));
    if fanout.electrical_driver().is_none() || (fanout.sinks.len() < 3 && !drives_visible_output) {
        return Ok(false);
    }

    let worst_delay = worst_path_delay(&snapshot.report);
    let root_criticality = signal_arrival(snapshot, graph.bits[root].0)
        + fanout
            .sinks
            .iter()
            .map(|sink| sink.criticality)
            .fold(0.0, f64::max);
    if root_criticality + TIMING_EPSILON < worst_delay * MIN_CRITICAL_PATH_FRACTION {
        return Ok(false);
    }

    Ok(
        characterized_load_limit(fanout, library, catalog, options)?.is_some_and(|limit| {
            max_load(sum_sink_load(&fanout.sinks))
                > limit * TIMING_DRIVEN_EFFORT_FRACTION + TIMING_EPSILON
        }),
    )
}

/// Uses explicit targets, Liberty max capacitance, and bounded gate effort.
fn characterized_load_limit(
    fanout: &TimingFanout,
    library: &Library,
    catalog: &CellCatalog,
    options: &BufferOptions,
) -> Result<Option<f64>> {
    if options.target_load.is_some() {
        return Ok(hard_output_load_limit(fanout, library, options));
    }
    let Some(driver) = fanout.electrical_driver() else {
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

/// Intersects an explicit load target with the driver's hard Liberty limit.
fn hard_output_load_limit(
    fanout: &TimingFanout,
    library: &Library,
    options: &BufferOptions,
) -> Option<f64> {
    let characterized = fanout
        .electrical_driver()
        .and_then(|driver| library.cells[driver.cell_index].pins[driver.pin_index].max_capacitance)
        .filter(|limit| limit.is_finite() && *limit > 0.0);
    match (options.target_load, characterized) {
        (Some(target), Some(characterized)) => Some(target.min(characterized)),
        (Some(target), None) => Some(target),
        (None, Some(characterized)) => Some(characterized),
        (None, None) => None,
    }
}

/// Distinguishes hard Liberty/user electrical bounds from soft effort targets.
fn exceeds_characterized_output_limit(
    fanout: &TimingFanout,
    library: &Library,
    options: &BufferOptions,
) -> bool {
    let limit = hard_output_load_limit(fanout, library, options);
    limit.is_some_and(|limit| max_load(sum_sink_load(&fanout.sinks)) > limit + TIMING_EPSILON)
        || exceeds_weighted_fanout_limit(fanout, library)
}

/// Compares real Liberty sink weights against their driver's fanout budget.
fn exceeds_weighted_fanout_limit(fanout: &TimingFanout, library: &Library) -> bool {
    let Some(driver) = fanout.electrical_driver() else {
        return false;
    };
    let Some(limit) = library.cells[driver.cell_index].pins[driver.pin_index].max_fanout else {
        return false;
    };
    limit.is_finite()
        && limit >= 0.0
        && sum_sink_fanout_load(&fanout.sinks) > limit + TIMING_EPSILON
}

/// Checks the strictest legal output/sink slew on one actual timed net.
fn exceeds_transition_limit(
    fanout: &TimingFanout,
    library: &Library,
    timing: Option<SignalTiming>,
) -> bool {
    let Some(timing) = timing else {
        return false;
    };
    let output_limit = fanout
        .electrical_driver()
        .and_then(|driver| library.cells[driver.cell_index].pins[driver.pin_index].max_transition);
    let limit = fanout
        .sinks
        .iter()
        .filter_map(|sink| sink.max_transition)
        .chain(output_limit)
        .filter(|limit| limit.is_finite() && *limit >= 0.0)
        .reduce(f64::min);
    limit.is_some_and(|limit| {
        timing.rise.transition.max(timing.fall.transition) > limit + TIMING_EPSILON
    })
}

/// Repairs illegal virtual-driver slew without changing legacy real-gate
/// policy.
fn exceeds_representative_transition_limit(
    fanout: &TimingFanout,
    library: &Library,
    catalog: &CellCatalog,
    timing: Option<SignalTiming>,
) -> bool {
    if fanout.representative_driver.is_none() || !exceeds_transition_limit(fanout, library, timing)
    {
        return false;
    }
    representative_driver_load_can_be_reduced(fanout, catalog)
}

/// Determines whether a real buffer can lower an external driver's sink load.
fn representative_driver_load_can_be_reduced(fanout: &TimingFanout, catalog: &CellCatalog) -> bool {
    let movable = fanout
        .sinks
        .iter()
        .filter(|sink| matches!(sink.target, TimingSinkTarget::InstancePin { .. }))
        .fold(CombinationalOutputLoad::default(), |mut load, sink| {
            load.rise += sink.load.rise;
            load.fall += sink.load.fall;
            load
        });
    catalog.buffers().any(|buffer| {
        let input = buffer.input_capacitances[0];
        input.rise <= movable.rise + TIMING_EPSILON
            && input.fall <= movable.fall + TIMING_EPSILON
            && (input.rise + TIMING_EPSILON < movable.rise
                || input.fall + TIMING_EPSILON < movable.fall)
    })
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
    speculative: bool,
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
            let weighted_load = sum_sink_fanout_load(&direct) + sum_sink_fanout_load(&level);
            let weighted_limit = fanout.electrical_driver().and_then(|driver| {
                library.cells[driver.cell_index].pins[driver.pin_index].max_fanout
            });
            if total_count <= options.max_fanout
                && root_limit.is_none_or(|limit| max_load(root_load) <= limit + TIMING_EPSILON)
                && weighted_limit.is_none_or(|limit| weighted_load <= limit + TIMING_EPSILON)
            {
                break;
            }
        }

        let previous_count = level.len();
        let previous_load = max_load(sum_sink_load(&level));
        // The root's effort and Liberty limits constrain only its own direct
        // fanout. Each child group is driven by a newly selected buffer, whose
        // electrical limits can be substantially larger than the root's.
        let groups = partition_timing_sinks(level, catalog, library, options);
        let mut next = Vec::with_capacity(groups.len());
        for group in groups {
            let buffer =
                select_timing_buffer(catalog, library, source_timing, &group, speculative)?;
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
                fanout_load: library.cells[buffer.cell_index].pins[buffer.input_pin_indices[0]]
                    .fanout_load
                    .unwrap_or(0.0),
                max_transition: library.cells[buffer.cell_index].pins[buffer.input_pin_indices[0]]
                    .max_transition,
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

/// Packs sinks using each prospective buffer's own hard electrical limits.
fn partition_timing_sinks(
    sinks: Vec<TimingSink>,
    catalog: &CellCatalog,
    library: &Library,
    options: &BufferOptions,
) -> Vec<TimingSinkGroup> {
    let mut groups = Vec::<TimingSinkGroup>::new();
    for sink in sinks {
        let is_output = matches!(sink.target, TimingSinkTarget::ModuleOutput { .. });
        let eligible = groups
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                if group.sinks.len() >= options.max_fanout.min(PREFERRED_BUFFER_STAGE_FANOUT)
                    || is_output && group.has_module_output
                {
                    return false;
                }
                let load = CombinationalOutputLoad {
                    rise: group.load.rise + sink.load.rise,
                    fall: group.load.fall + sink.load.fall,
                };
                if options
                    .target_load
                    .is_some_and(|limit| max_load(load) > limit + TIMING_EPSILON)
                {
                    return false;
                }
                let fanout_load = group.fanout_load + sink.fanout_load;
                catalog
                    .buffers()
                    .any(|buffer| timing_buffer_can_drive(buffer, library, load, fanout_load))
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
        group.fanout_load += sink.fanout_load;
        group.has_module_output |= is_output;
        group.sinks.push(sink);
    }
    groups
}

/// Checks hard Liberty output limits independently of soft electrical effort.
fn timing_buffer_can_drive(
    buffer: &CatalogCell,
    library: &Library,
    load: CombinationalOutputLoad,
    fanout_load: f64,
) -> bool {
    let output = &library.cells[buffer.cell_index].pins[buffer.output_pin_index];
    buffer
        .output_max_capacitance
        .is_none_or(|limit| max_load(load) <= limit + TIMING_EPSILON)
        && output
            .max_fanout
            .is_none_or(|limit| fanout_load <= limit + TIMING_EPSILON)
}

/// Chooses the smallest buffer meeting ABC-style effort and real Liberty
/// timing.
fn select_timing_buffer<'a>(
    catalog: &'a CellCatalog,
    library: &Library,
    source_timing: Option<SignalTiming>,
    group: &TimingSinkGroup,
    speculative: bool,
) -> Result<&'a CatalogCell> {
    let load = max_load(group.load);
    let required_input_capacitance = load / MAX_ELECTRICAL_EFFORT;
    let mut best: Option<(&CatalogCell, f64)> = None;
    let mut timing_best: Option<(&CatalogCell, f64)> = None;
    let mut fallback: Option<(&CatalogCell, f64)> = None;
    for buffer in catalog.buffers() {
        let cell = &library.cells[buffer.cell_index];
        let output = &cell.pins[buffer.output_pin_index];
        if !timing_buffer_can_drive(buffer, library, group.load, group.fanout_load) {
            continue;
        }
        let delay = if let Some(timing) = source_timing {
            let input = library.resolve_string(&cell.pins[buffer.input_pin_indices[0]].name);
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
            if speculative
                && source_timing.is_some()
                && input_capacitance * MIN_TIMING_BUFFER_EFFORT <= load + TIMING_EPSILON
            {
                let replace = timing_best.is_none_or(|(current, current_delay)| {
                    delay
                        .total_cmp(&current_delay)
                        .then_with(|| {
                            input_capacitance.total_cmp(&max_load(current.input_capacitances[0]))
                        })
                        .then_with(|| buffer.area.total_cmp(&current.area))
                        .then_with(|| buffer.name.cmp(&current.name))
                        == Ordering::Less
                });
                if replace {
                    timing_best = Some((buffer, delay));
                }
            }
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
    timing_best
        .or(best)
        .or(fallback)
        .map(|(buffer, _)| buffer)
        .ok_or_else(|| {
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

/// Sums dimensionless Liberty fanout weights independently of capacitance.
fn sum_sink_fanout_load(sinks: &[TimingSink]) -> f64 {
    sinks.iter().map(|sink| sink.fanout_load).sum()
}

/// Returns the conservative scalar of independently modeled edge loads.
fn max_load(load: CombinationalOutputLoad) -> f64 {
    load.rise.max(load.fall)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{
        BufferTimingConstraints, TimingDriver, TimingFanout, TimingGraph, TimingSink,
        TimingSinkGroup, TimingSinkTarget, build_electrical_timing_graph,
        consolidate_timing_aware_buffers, exceeds_transition_limit, fanout_eligible,
        fanout_is_overloaded, has_slow_shared_primary_output,
        insert_speculative_timing_aware_buffers, insert_timing_aware_buffers, max_load,
        partition_timing_sinks, select_timing_buffer, sum_sink_fanout_load, sum_sink_load,
    };
    use crate::liberty_model::{
        Cell, Library, LibraryBuilder, LuTableTemplate, Pin, PinDirection, Sequential,
        SequentialKind, TimingArc, TimingTable,
    };
    use crate::liberty_proto::{BoundaryTimingDefaults, TimingTableKind};
    use crate::netlist::buffer::BufferOptions;
    use crate::netlist::cell_catalog::CellCatalog;
    use crate::netlist::cell_catalog::test_utils::{parse_module, sizing_library};
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::parse::NetRef;
    use crate::netlist::report::build_netlist_report;
    use crate::netlist::sta::{
        CombinationalOutputLoad, EdgeTiming, ScopedBoundaryTimingDefaultsSuppression, SignalTiming,
        StaOptions, resolved_module_output_load,
    };
    use std::collections::{BTreeMap, BTreeSet};

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

    /// Models a legal but slow complex output also consumed internally.
    pub(crate) fn slow_shared_output_library() -> Library {
        let mut builder = LibraryBuilder::from_library(sizing_library());
        builder.lu_table_templates.push(LuTableTemplate {
            kind: "lu_table_template".to_string().into(),
            name: "shared_output_load".to_string(),
            variable_1: "total_output_net_capacitance".to_string().into(),
            index_1: vec![0.0, 1.0],
            ..LuTableTemplate::default()
        });

        let mut arcs = Vec::new();
        for input in ["A", "B", "C"] {
            let mut tables = Vec::new();
            for (kind, values) in [
                (TimingTableKind::CellRise, vec![0.5, 10.5]),
                (TimingTableKind::CellFall, vec![0.5, 10.5]),
                (TimingTableKind::RiseTransition, vec![0.1, 10.1]),
                (TimingTableKind::FallTransition, vec![0.1, 10.1]),
            ] {
                tables.push(
                    builder
                        .add_timing_table_f64(kind, 1, vec![], vec![], vec![], values, vec![2], "")
                        .expect("construct load-sensitive shared-output timing table"),
                );
            }
            arcs.push(
                builder
                    .add_timing_arc(input, "positive_unate", "combinational", "", tables)
                    .expect("construct shared-output combinational timing arc"),
            );
        }

        let mut pins = ["A", "B", "C"]
            .into_iter()
            .map(|input| Pin {
                name: builder
                    .intern_string(input)
                    .expect("intern weak logic input"),
                direction: PinDirection::Input as i32,
                capacitance: Some(0.1),
                ..Pin::default()
            })
            .collect::<Vec<_>>();
        pins.push(Pin {
            name: builder
                .intern_string("Y")
                .expect("intern weak logic output"),
            direction: PinDirection::Output as i32,
            function: builder
                .intern_string("A * B * C")
                .expect("intern weak logic function"),
            max_capacitance: Some(1.6),
            timing_arcs: arcs,
            ..Pin::default()
        });
        builder.cells.push(Cell {
            name: "WEAK_AND3".to_string(),
            pins,
            area: 1.0,
            ..Cell::default()
        });
        builder.finish()
    }

    /// Shares one directly exposed logic driver with a critical internal sink.
    pub(crate) fn slow_shared_output_source() -> &'static str {
        r#"
module top(a, b, c, direct, path);
  input a, b, c;
  output direct, path;
  WEAK_AND3 weak (.A(a), .B(b), .C(c), .Y(direct));
  BUF consumer (.A(direct), .Y(path));
endmodule
"#
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

    /// Creates a load-sensitive external driver beside ordinary legal buffers.
    fn representative_input_library() -> Library {
        let mut builder = LibraryBuilder::from_library(sizing_library());
        builder.lu_table_templates.push(LuTableTemplate {
            kind: "lu_table_template".to_string().into(),
            name: "representative_input_load".to_string(),
            variable_1: "total_output_net_capacitance".to_string().into(),
            index_1: vec![0.0, 1.0],
            ..LuTableTemplate::default()
        });
        let template_index = u32::try_from(builder.lu_table_templates.len())
            .expect("test template index fits in a Liberty timing reference");
        let tables = [
            (TimingTableKind::CellRise, vec![0.0, 50.0]),
            (TimingTableKind::CellFall, vec![0.0, 50.0]),
            (TimingTableKind::RiseTransition, vec![0.01, 1.01]),
            (TimingTableKind::FallTransition, vec![0.01, 1.01]),
        ]
        .into_iter()
        .map(|(kind, values)| {
            builder
                .add_timing_table_f64(
                    kind,
                    template_index,
                    vec![],
                    vec![],
                    vec![],
                    values,
                    vec![2],
                    "",
                )
                .expect("construct load-sensitive representative input timing")
        })
        .collect();
        let arc = builder
            .add_timing_arc("A", "positive_unate", "combinational", "", tables)
            .expect("characterize the virtual primary-input driver");
        let mut driver = builder
            .cells
            .iter()
            .find(|cell| cell.name == "BUF")
            .cloned()
            .expect("clone the ordinary characterized test buffer");
        driver.name = "DRIVER".to_string();
        driver.area = 1000.0;
        let output = driver
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Output as i32)
            .expect("find the virtual-driver output");
        output.max_capacitance = Some(1.0);
        output.timing_arcs = vec![arc];
        builder.cells.push(driver);

        let mut library = builder.finish();
        library.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "DRIVER".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 1,
        });
        library
    }

    /// Builds a scalar primary input with independently exposed buffer sinks.
    fn high_fanout_primary_input_source(sinks: usize) -> String {
        assert!(sinks > 0);
        let mut source = format!(
            "module top(a, out);\n  input a;\n  output [{}:0] out;\n",
            sinks - 1
        );
        for sink in 0..sinks {
            source.push_str(&format!("  BUF sink{sink} (.A(a), .Y(out[{sink}]));\n"));
        }
        source.push_str("endmodule\n");
        source
    }

    /// Builds a weak internal root with independently observable buffer sinks.
    fn weak_root_buffer_source(sinks: usize) -> String {
        assert!(sinks > 0);
        let mut source = format!(
            "module top(a, b, out);\n  input a, b;\n  output [{}:0] out;\n  wire root;\n  AND2 driver (.A(a), .B(b), .Y(root));\n",
            sinks - 1
        );
        for sink in 0..sinks {
            source.push_str(&format!("  BUF sink{sink} (.A(root), .Y(out[{sink}]));\n"));
        }
        source.push_str("endmodule\n");
        source
    }

    /// Mutably resolves the output limits of the synthetic external driver.
    fn representative_driver_output(library: &mut Library) -> &mut Pin {
        library
            .cells
            .iter_mut()
            .find(|cell| cell.name == "DRIVER")
            .expect("find the synthetic representative driver")
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Output as i32)
            .expect("find the synthetic representative-driver output")
    }

    /// Resolves the electrically constrained virtual driver of scalar input A.
    fn representative_primary_input_fanout(graph: &TimingGraph) -> &TimingFanout {
        graph
            .fanouts
            .iter()
            .find(|fanout| fanout.is_primary_input && fanout.representative_driver.is_some())
            .expect("find the characterized representative primary-input driver")
    }

    #[test]
    fn automatically_buffers_representative_driven_primary_inputs() {
        let library = representative_input_library();
        let (mut module, mut nets, mut interner) =
            parse_module(&high_fanout_primary_input_source(4));
        let before =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("time the overloaded representative external driver");

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
        .expect("automatically repair the characterized primary-input fanout");
        let after =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("independently time the repaired representative-driver load");

        assert!(stats.buffers_inserted > 0);
        assert_eq!(stats.max_fanout_before, 4);
        assert!(stats.max_fanout_after <= 3);
        assert_eq!(stats.unresolved_overloaded_nets, 0);
        assert!(after.max_delay.unwrap() < before.max_delay.unwrap());
    }

    #[test]
    fn representative_driver_max_capacitance_triggers_primary_input_buffering() {
        let mut library = representative_input_library();
        representative_driver_output(&mut library).max_capacitance = Some(0.25);
        let (mut module, mut nets, mut interner) =
            parse_module(&high_fanout_primary_input_source(4));

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("repair the virtual output capacitance below the generic fanout bound");

        let graph = build_electrical_timing_graph(
            &module,
            &nets,
            &interner,
            &library,
            &BufferOptions::default(),
        )
        .expect("inspect the repaired representative driver separately from child buffers");
        let root = representative_primary_input_fanout(&graph);
        assert_eq!(stats.max_fanout_before, 4);
        assert!(stats.buffers_inserted > 0);
        assert!(max_load(sum_sink_load(&root.sinks)) <= 0.25 + 1e-9);
        assert_eq!(stats.unresolved_overloaded_nets, 0);
    }

    #[test]
    fn explicit_load_target_cannot_override_representative_driver_max_capacitance() {
        let mut library = representative_input_library();
        representative_driver_output(&mut library).max_capacitance = Some(0.25);
        let (mut module, mut nets, mut interner) =
            parse_module(&high_fanout_primary_input_source(4));

        let options = BufferOptions {
            target_load: Some(1.0),
            ..BufferOptions::default()
        };
        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &options,
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("preserve the hard Liberty capacitance under a looser explicit target");

        let graph = build_electrical_timing_graph(&module, &nets, &interner, &library, &options)
            .expect("inspect the actual representative-driver load");
        let root = representative_primary_input_fanout(&graph);
        assert!(stats.buffers_inserted > 0);
        assert!(max_load(sum_sink_load(&root.sinks)) <= 0.25 + 1e-9);
        assert_eq!(stats.unresolved_overloaded_nets, 0);
    }

    #[test]
    fn unrepairable_representative_driver_capacitance_does_not_add_redundant_buffers() {
        let mut library = representative_input_library();
        representative_driver_output(&mut library).max_capacitance = Some(0.05);
        let (mut module, mut nets, mut interner) =
            parse_module(&high_fanout_primary_input_source(1));

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("report an impossible single-sink capacitance without worsening the circuit");

        assert_eq!(stats.buffers_inserted, 0);
        assert_eq!(stats.unresolved_overloaded_nets, 1);
    }

    #[test]
    fn representative_driver_max_fanout_triggers_primary_input_buffering() {
        let mut library = representative_input_library();
        representative_driver_output(&mut library).max_fanout = Some(2.5);
        for cell in &mut library.cells {
            if matches!(cell.name.as_str(), "BUF" | "BUF_FAST") {
                cell.pins
                    .iter_mut()
                    .find(|pin| pin.direction == PinDirection::Input as i32)
                    .expect("find the weighted actual buffer input")
                    .fanout_load = Some(1.0);
            }
        }
        let (mut module, mut nets, mut interner) =
            parse_module(&high_fanout_primary_input_source(4));

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("repair the representative driver's stricter weighted-fanout limit");

        let graph = build_electrical_timing_graph(
            &module,
            &nets,
            &interner,
            &library,
            &BufferOptions::default(),
        )
        .expect("inspect the representative driver's own weighted fanout");
        let root = representative_primary_input_fanout(&graph);
        assert_eq!(stats.max_fanout_before, 4);
        assert!(stats.buffers_inserted > 0);
        assert!(sum_sink_fanout_load(&root.sinks) <= 2.5 + 1e-9);
        assert_eq!(stats.unresolved_overloaded_nets, 0);
    }

    #[test]
    fn representative_driver_max_transition_triggers_primary_input_buffering() {
        let mut library = representative_input_library();
        representative_driver_output(&mut library).max_transition = Some(0.25);
        let (mut module, mut nets, mut interner) =
            parse_module(&high_fanout_primary_input_source(4));

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("repair a representative-driver slew violation below generic fanout bounds");

        assert_eq!(stats.max_fanout_before, 4);
        assert!(stats.buffers_inserted > 0);
        assert_eq!(stats.unresolved_overloaded_nets, 0);
        assert!(stats.final_worst_delay.unwrap() < stats.initial_worst_delay.unwrap());
    }

    #[test]
    fn unrepairable_representative_driver_slew_does_not_add_redundant_buffers() {
        let mut library = representative_input_library();
        representative_driver_output(&mut library).max_transition = Some(0.05);
        let (mut module, mut nets, mut interner) =
            parse_module(&high_fanout_primary_input_source(1));

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("report an unrepairable single-sink source slew without redundant buffers");

        assert_eq!(stats.buffers_inserted, 0);
        assert_eq!(stats.unresolved_overloaded_nets, 1);
    }

    #[test]
    fn ideal_primary_inputs_still_require_explicit_buffering_opt_in() {
        let library = sizing_library();
        let source = high_fanout_primary_input_source(4);
        let options = BufferOptions {
            max_fanout: 3,
            ..BufferOptions::default()
        };
        let (mut ideal_module, mut ideal_nets, mut ideal_interner) = parse_module(&source);
        let ideal = insert_timing_aware_buffers(
            &mut ideal_module,
            &mut ideal_nets,
            &mut ideal_interner,
            &library,
            &options,
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("preserve the legacy ideal-input buffering default");
        assert_eq!(ideal.buffers_inserted, 0);

        let (mut module, mut nets, mut interner) = parse_module(&source);
        let explicit = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                buffer_primary_inputs: true,
                ..options
            },
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("preserve explicit buffering of legacy ideal primary inputs");
        assert!(explicit.buffers_inserted > 0);
    }

    #[test]
    fn selective_physical_scope_excludes_packed_synthetic_register_inputs() {
        let library = representative_input_library();
        let source = r#"
module top(a, q, out);
  input [3:2] a;
  input [3:2] q;
  output [7:0] out;
  BUF physical0 (.A(a[3]), .Y(out[0]));
  BUF physical1 (.A(a[3]), .Y(out[1]));
  BUF physical2 (.A(a[3]), .Y(out[2]));
  BUF physical3 (.A(a[3]), .Y(out[3]));
  BUF synthetic0 (.A(q[3]), .Y(out[4]));
  BUF synthetic1 (.A(q[3]), .Y(out[5]));
  BUF synthetic2 (.A(q[3]), .Y(out[6]));
  BUF synthetic3 (.A(q[3]), .Y(out[7]));
endmodule
"#;
        let (mut module, mut nets, mut interner) = parse_module(source);
        let options = BufferOptions {
            max_fanout: 3,
            ..BufferOptions::default()
        };
        let _physical = ScopedBoundaryTimingDefaultsSuppression::for_physical_ports(
            BTreeSet::from(["a_3".to_string()]),
            (0..8).map(|index| format!("out_{index}")).collect(),
        );
        let graph = build_electrical_timing_graph(&module, &nets, &interner, &library, &options)
            .expect("identify the actual packed external bit and exclude synthetic Q");
        let a = module
            .find_net_index(interner.get("a").expect("resolve a"), &nets)
            .expect("find packed physical input");
        let q = module
            .find_net_index(interner.get("q").expect("resolve q"), &nets)
            .expect("find packed synthetic input");
        let physical = graph
            .bits
            .iter()
            .position(|&(net, bit)| net == a && bit == 3)
            .expect("find a[3]");
        let synthetic = graph
            .bits
            .iter()
            .position(|&(net, bit)| net == q && bit == 3)
            .expect("find q[3]");
        assert!(graph.fanouts[physical].representative_driver.is_some());
        assert!(graph.fanouts[synthetic].representative_driver.is_none());
        assert!(fanout_eligible(&graph.fanouts[physical], &options));
        assert!(!fanout_eligible(&graph.fanouts[synthetic], &options));

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &options,
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("buffer only the genuinely driven packed physical input");
        assert_eq!(stats.buffered_nets, 1);
        let after = build_electrical_timing_graph(&module, &nets, &interner, &library, &options)
            .expect("inspect the buffered physical and untouched synthetic bits");
        let physical = after
            .bits
            .iter()
            .position(|&(net, bit)| net == a && bit == 3)
            .expect("find buffered a[3]");
        let synthetic = after
            .bits
            .iter()
            .position(|&(net, bit)| net == q && bit == 3)
            .expect("find unbuffered q[3]");
        assert!(after.fanouts[physical].sinks.len() <= options.max_fanout);
        assert_eq!(after.fanouts[synthetic].sinks.len(), 4);
    }

    #[test]
    fn selective_physical_scope_excludes_packed_synthetic_register_outputs() {
        let library = representative_input_library();
        let source = r#"
module top(a, y, d);
  input a;
  output [3:2] y;
  output [3:2] d;
  BUF physical0 (.A(a), .Y(y[3]));
  BUF physical1 (.A(a), .Y(y[2]));
  BUF synthetic0 (.A(a), .Y(d[3]));
  BUF synthetic1 (.A(a), .Y(d[2]));
endmodule
"#;
        let (module, nets, interner) = parse_module(source);
        let physical_net = module
            .find_net_index(interner.get("y").expect("resolve y"), &nets)
            .expect("find packed physical outputs");
        let synthetic_net = module
            .find_net_index(interner.get("d").expect("resolve d"), &nets)
            .expect("find packed synthetic register D outputs");
        let _physical = ScopedBoundaryTimingDefaultsSuppression::for_physical_ports(
            BTreeSet::from(["a".to_string()]),
            BTreeSet::from(["y_3".to_string(), "y_2".to_string()]),
        );

        for module_output_load in [0.0, 0.75] {
            let options = BufferOptions {
                module_output_load,
                ..BufferOptions::default()
            };
            let expected = resolved_module_output_load(
                &library,
                StaOptions {
                    module_output_load,
                    ..StaOptions::default()
                },
            )
            .expect("resolve explicit or representative physical-output loading");
            let graph =
                build_electrical_timing_graph(&module, &nets, &interner, &library, &options)
                    .expect("distinguish physical outputs from packed synthetic D endpoints");
            for bit in [3, 2] {
                let physical = graph
                    .bits
                    .iter()
                    .position(|&(net, candidate)| net == physical_net && candidate == bit)
                    .expect("find packed physical output");
                let synthetic = graph
                    .bits
                    .iter()
                    .position(|&(net, candidate)| net == synthetic_net && candidate == bit)
                    .expect("find packed synthetic register output");
                assert_eq!(graph.fanouts[physical].sinks.len(), 1);
                assert_eq!(graph.fanouts[synthetic].sinks.len(), 1);
                assert_eq!(graph.fanouts[physical].sinks[0].load, expected);
                assert_eq!(
                    graph.fanouts[synthetic].sinks[0].load,
                    CombinationalOutputLoad::default()
                );
            }
        }
    }

    #[test]
    fn fully_suppressed_boundary_scope_preserves_ideal_input_behavior() {
        let library = representative_input_library();
        let (mut module, mut nets, mut interner) =
            parse_module(&high_fanout_primary_input_source(4));
        let _suppressed = ScopedBoundaryTimingDefaultsSuppression::new();

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
        .expect("leave implicitly ideal synthetic boundaries untouched");

        assert_eq!(stats.buffers_inserted, 0);
    }

    #[test]
    fn fully_suppressed_boundary_scope_excludes_explicit_module_output_load() {
        let library = representative_input_library();
        let (module, nets, interner) = parse_module(&high_fanout_primary_input_source(1));
        let _suppressed = ScopedBoundaryTimingDefaultsSuppression::new();
        let graph = build_electrical_timing_graph(
            &module,
            &nets,
            &interner,
            &library,
            &BufferOptions {
                module_output_load: 0.75,
                ..BufferOptions::default()
            },
        )
        .expect("keep explicitly specified loads off suppressed synthetic output boundaries");

        let output = module
            .find_net_index(interner.get("out").expect("resolve out"), &nets)
            .expect("find the suppressed output");
        let bit = graph
            .bits
            .iter()
            .position(|&(net, _)| net == output)
            .expect("find the suppressed output bit");
        assert_eq!(graph.fanouts[bit].sinks.len(), 1);
        assert_eq!(
            graph.fanouts[bit].sinks[0].load,
            CombinationalOutputLoad::default()
        );
    }

    #[test]
    fn representative_driver_never_enables_primary_clock_buffering() {
        let mut library = registered_timing_library();
        library.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "BUF".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 1,
        });
        let (mut module, mut nets, mut interner) = parse_module(high_fanout_register_source());
        let options = BufferOptions {
            max_fanout: 3,
            ..BufferOptions::default()
        };
        let clock = module
            .find_net_index(interner.get("clk").expect("resolve clk"), &nets)
            .expect("resolve clock net");
        let graph = build_electrical_timing_graph(&module, &nets, &interner, &library, &options)
            .expect("classify the representative-driven clock and register data fanouts");
        let clock_bit = graph
            .bits
            .iter()
            .position(|&(net, _)| net == clock)
            .expect("find the primary clock bit");
        assert!(graph.fanouts[clock_bit].representative_driver.is_some());
        assert!(graph.fanouts[clock_bit].protected_clock);
        assert!(!fanout_eligible(&graph.fanouts[clock_bit], &options));

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &options,
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("repair physical register data without ever buffering the clock");
        assert!(stats.buffers_inserted > 0);
        for instance in &module.instances {
            if interner
                .resolve(instance.instance_name)
                .is_some_and(|name| name.starts_with("u_buf_"))
            {
                assert!(
                    instance
                        .connections
                        .iter()
                        .all(|(_, net)| !matches!(net, NetRef::Simple(net) if *net == clock)),
                    "a representative driver must never allow clock buffering"
                );
            }
        }
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

        let buffer = select_timing_buffer(&catalog, &library, None, &group, false)
            .expect("select smallest electrically sufficient test buffer");

        assert_eq!(buffer.name, "BUF");
    }

    #[test]
    fn packs_buffer_groups_using_their_own_capacitance_and_weighted_fanout() {
        let mut library = sizing_library();
        let driver = library
            .cells
            .iter()
            .position(|cell| cell.name == "AND2")
            .expect("find the electrically weak root driver");
        let output = library.cells[driver]
            .pins
            .iter()
            .position(|pin| pin.direction == PinDirection::Output as i32)
            .expect("find the weak root output");
        library.cells[driver].pins[output].max_capacitance = Some(0.35);
        library.cells[driver].pins[output].max_fanout = Some(2.0);
        for name in ["BUF", "BUF_FAST"] {
            let buffer = library
                .cells
                .iter_mut()
                .find(|cell| cell.name == name)
                .expect("find the characterized child buffer");
            buffer
                .pins
                .iter_mut()
                .find(|pin| pin.direction == PinDirection::Input as i32)
                .expect("find the child buffer input")
                .fanout_load = Some(1.0);
            buffer
                .pins
                .iter_mut()
                .find(|pin| pin.direction == PinDirection::Output as i32)
                .expect("find the child buffer output")
                .max_fanout = Some(12.0);
        }
        let (mut module, mut nets, mut interner) = parse_module(&weak_root_buffer_source(11));

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("repair a weak root with one electrically stronger child buffer");

        assert_eq!(stats.buffers_inserted, 1);
        assert_eq!(stats.unresolved_overloaded_nets, 0);
        let inserted = module
            .instances
            .iter()
            .find(|instance| {
                interner
                    .resolve(instance.instance_name)
                    .is_some_and(|name| name.starts_with("u_buf_"))
            })
            .expect("find the sole inserted child stage");
        assert_eq!(interner.resolve(inserted.type_name), Some("BUF_FAST"));
        assert_eq!(stats.max_fanout_after, 10);
    }

    #[test]
    fn child_buffer_grouping_preserves_explicit_per_stage_load_targets() {
        let library = sizing_library();
        let catalog = CellCatalog::new(&library).expect("classify legal child buffers");
        let sinks = (0..6)
            .map(|index| TimingSink {
                target: TimingSinkTarget::InstancePin {
                    instance_index: index,
                    connection_index: 0,
                },
                load: CombinationalOutputLoad {
                    rise: 0.1,
                    fall: 0.1,
                },
                fanout_load: 0.0,
                max_transition: None,
                criticality: 0.0,
            })
            .collect();

        let groups = partition_timing_sinks(
            sinks,
            &catalog,
            &library,
            &BufferOptions {
                target_load: Some(0.25),
                ..BufferOptions::default()
            },
        );

        assert_eq!(groups.len(), 3);
        assert!(groups.iter().all(|group| group.sinks.len() == 2));
        assert!(groups.iter().all(|group| group.load.rise <= 0.25));
    }

    #[test]
    fn child_buffer_grouping_keeps_timing_headroom_below_the_hard_fanout_limit() {
        let library = sizing_library();
        let catalog = CellCatalog::new(&library).expect("classify legal child buffers");
        let sinks = (0..21)
            .map(|index| TimingSink {
                target: TimingSinkTarget::InstancePin {
                    instance_index: index,
                    connection_index: 0,
                },
                load: CombinationalOutputLoad {
                    rise: 0.1,
                    fall: 0.1,
                },
                fanout_load: 0.0,
                max_transition: None,
                criticality: 0.0,
            })
            .collect();

        let groups = partition_timing_sinks(sinks, &catalog, &library, &BufferOptions::default());

        assert_eq!(groups.len(), 3);
        assert!(groups.iter().all(|group| group.sinks.len() <= 10));
        assert_eq!(
            groups.iter().map(|group| group.sinks.len()).sum::<usize>(),
            21
        );
    }

    #[test]
    fn zero_weight_fanout_does_not_turn_liberty_limit_into_sink_count() {
        let mut library = sizing_library();
        let driver_cell = library
            .cells
            .iter()
            .position(|cell| cell.name == "BUF")
            .expect("test library contains the driver");
        let driver_pin = library.cells[driver_cell]
            .pins
            .iter()
            .position(|pin| pin.direction == PinDirection::Output as i32)
            .expect("driver contains an output");
        library.cells[driver_cell].pins[driver_pin].max_fanout = Some(1.5);
        let catalog = CellCatalog::new(&library).expect("classify test library");
        let mut fanout = TimingFanout {
            driver: Some(TimingDriver {
                instance_index: 0,
                connection_index: 0,
                cell_index: driver_cell,
                pin_index: driver_pin,
            }),
            sinks: (0..3)
                .map(|index| TimingSink {
                    target: TimingSinkTarget::InstancePin {
                        instance_index: index + 1,
                        connection_index: 0,
                    },
                    load: CombinationalOutputLoad::default(),
                    fanout_load: 0.0,
                    max_transition: None,
                    criticality: 0.0,
                })
                .collect(),
            ..TimingFanout::default()
        };

        assert!(
            !fanout_is_overloaded(&fanout, &library, &catalog, &BufferOptions::default())
                .expect("evaluate zero-weight fanout")
        );

        for sink in &mut fanout.sinks {
            sink.fanout_load = 0.75;
        }
        assert!(
            fanout_is_overloaded(&fanout, &library, &catalog, &BufferOptions::default())
                .expect("evaluate fractional weighted fanout")
        );
    }

    #[test]
    fn checks_strictest_driver_and_sink_transition_limits() {
        let mut library = sizing_library();
        let driver_cell = library
            .cells
            .iter()
            .position(|cell| cell.name == "BUF")
            .expect("test library contains the driver");
        let driver_pin = library.cells[driver_cell]
            .pins
            .iter()
            .position(|pin| pin.direction == PinDirection::Output as i32)
            .expect("driver contains an output");
        library.cells[driver_cell].pins[driver_pin].max_transition = Some(0.3);
        let mut fanout = TimingFanout {
            driver: Some(TimingDriver {
                instance_index: 0,
                connection_index: 0,
                cell_index: driver_cell,
                pin_index: driver_pin,
            }),
            sinks: vec![TimingSink {
                target: TimingSinkTarget::InstancePin {
                    instance_index: 1,
                    connection_index: 0,
                },
                load: CombinationalOutputLoad::default(),
                fanout_load: 0.0,
                max_transition: Some(0.15),
                criticality: 0.0,
            }],
            ..TimingFanout::default()
        };
        let timing = SignalTiming {
            rise: EdgeTiming {
                arrival: 0.0,
                transition: 0.2,
            },
            fall: EdgeTiming {
                arrival: 0.0,
                transition: 0.1,
            },
        };

        assert!(exceeds_transition_limit(&fanout, &library, Some(timing)));
        fanout.sinks[0].max_transition = Some(0.25);
        assert!(!exceeds_transition_limit(&fanout, &library, Some(timing)));
        library.cells[driver_cell].pins[driver_pin].max_transition = Some(0.18);
        assert!(exceeds_transition_limit(&fanout, &library, Some(timing)));
    }

    #[test]
    fn reports_actual_unresolved_transition_without_creating_a_slew_only_buffer_root() {
        let mut library = sizing_library();
        let driver = library
            .cells
            .iter()
            .position(|cell| cell.name == "BUF")
            .expect("find slew-limited output driver");
        let output = library.cells[driver]
            .pins
            .iter()
            .position(|pin| pin.direction == PinDirection::Output as i32)
            .expect("find slew-limited output pin");
        library.cells[driver].pins[output].max_transition = Some(0.05);
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, y);
  input a;
  output y;
  BUF driver (.A(a), .Y(y));
endmodule
"#,
        );

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("diagnose an actual final slew violation without speculative topology changes");

        assert_eq!(stats.buffers_inserted, 0);
        assert_eq!(stats.unresolved_overloaded_nets, 1);
    }

    #[test]
    fn reports_transition_violations_for_exact_packed_output_bits() {
        let mut builder = LibraryBuilder::from_library(sizing_library());
        let buffer_index = builder
            .cells
            .iter()
            .position(|cell| cell.name == "BUF")
            .expect("find packed-output buffer");
        let output_index = builder.cells[buffer_index]
            .pins
            .iter()
            .position(|pin| pin.direction == PinDirection::Output as i32)
            .expect("find packed-output buffer pin");
        builder.cells[buffer_index].pins[output_index].max_transition = Some(0.2);

        let mut slow = builder.cells[buffer_index].clone();
        slow.name = "BUF_SLOW".to_string();
        slow.pins[output_index].max_transition = Some(0.3);
        let tables = [
            (TimingTableKind::CellRise, 5.0),
            (TimingTableKind::CellFall, 5.0),
            (TimingTableKind::RiseTransition, 0.4),
            (TimingTableKind::FallTransition, 0.4),
        ]
        .into_iter()
        .map(|(kind, value)| scalar_table(&mut builder, kind, value))
        .collect();
        slow.pins[output_index].timing_arcs = vec![
            builder
                .add_timing_arc("A", "positive_unate", "combinational", "", tables)
                .expect("characterize slower packed-output bit"),
        ];
        builder.cells.push(slow);
        let library = builder.finish();
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, y);
  input [1:0] a;
  output [1:0] y;
  BUF fast (.A(a[0]), .Y(y[0]));
  BUF_SLOW slow (.A(a[1]), .Y(y[1]));
endmodule
"#,
        );

        let stats = insert_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("evaluate each packed output with its own final Liberty slew");

        assert_eq!(stats.buffers_inserted, 0);
        assert_eq!(stats.unresolved_overloaded_nets, 1);
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

        let buffer = select_timing_buffer(&catalog, &library, None, &group, false)
            .expect("select stronger buffer required by real Liberty load");

        assert_eq!(buffer.name, "BUF_FAST");
    }

    #[test]
    fn retains_provisional_buffer_when_overloaded_source_slew_exceeds_every_limit() {
        let mut library = sizing_library();
        for (name, transition_limit) in [("BUF", 0.01), ("BUF_FAST", 0.02)] {
            let cell = library
                .cells
                .iter()
                .position(|cell| cell.name == name)
                .expect("find timing-characterized provisional buffer");
            let output = library.cells[cell]
                .pins
                .iter()
                .position(|pin| pin.direction == PinDirection::Output as i32)
                .expect("find provisional-buffer output");
            library.cells[cell].pins[output].max_transition = Some(transition_limit);
        }
        let catalog = CellCatalog::new(&library).expect("catalog provisional buffer variants");
        let group = TimingSinkGroup {
            load: CombinationalOutputLoad {
                rise: 0.72,
                fall: 0.72,
            },
            ..TimingSinkGroup::default()
        };
        let overloaded_source = SignalTiming {
            rise: EdgeTiming {
                arrival: 5.0,
                transition: 1.0,
            },
            fall: EdgeTiming {
                arrival: 5.0,
                transition: 1.0,
            },
        };

        let selected =
            select_timing_buffer(&catalog, &library, Some(overloaded_source), &group, false)
                .expect("unloading the original source requires retaining a provisional buffer");

        assert_eq!(selected.name, "BUF");

        let speculative =
            select_timing_buffer(&catalog, &library, Some(overloaded_source), &group, true)
                .expect("preserve the original timing-ranked speculative choice");
        assert_eq!(speculative.name, "BUF_FAST");
    }

    #[test]
    fn overloaded_source_repair_preserves_legacy_ranking_even_when_stronger_buffer_looks_legal() {
        let mut library = sizing_library();
        for (name, transition_limit) in [("BUF", 0.05), ("BUF_FAST", 0.2)] {
            let cell = library
                .cells
                .iter()
                .position(|cell| cell.name == name)
                .expect("find overloaded-source repair buffer");
            let output = library.cells[cell]
                .pins
                .iter()
                .position(|pin| pin.direction == PinDirection::Output as i32)
                .expect("find overloaded-source repair output");
            library.cells[cell].pins[output].max_transition = Some(transition_limit);
        }
        let catalog = CellCatalog::new(&library).expect("catalog overloaded-source repair cells");
        let group = TimingSinkGroup {
            load: CombinationalOutputLoad {
                rise: 0.72,
                fall: 0.72,
            },
            ..TimingSinkGroup::default()
        };
        let overloaded_source = SignalTiming {
            rise: EdgeTiming {
                arrival: 5.0,
                transition: 1.0,
            },
            fall: EdgeTiming {
                arrival: 5.0,
                transition: 1.0,
            },
        };

        let repaired =
            select_timing_buffer(&catalog, &library, Some(overloaded_source), &group, false)
                .expect("retain the original economical repair tree");

        assert_eq!(repaired.name, "BUF");
    }

    #[test]
    fn speculative_buffer_strength_prioritizes_exact_delay_without_overloading_driver() {
        let library = registered_timing_library();
        let catalog = CellCatalog::new(&library).expect("catalog timing-complete test buffers");
        let group = TimingSinkGroup {
            load: CombinationalOutputLoad {
                rise: 0.72,
                fall: 0.72,
            },
            ..TimingSinkGroup::default()
        };
        let source_timing = SignalTiming {
            rise: EdgeTiming {
                arrival: 5.0,
                transition: 0.1,
            },
            fall: EdgeTiming {
                arrival: 5.0,
                transition: 0.1,
            },
        };

        let conservative =
            select_timing_buffer(&catalog, &library, Some(source_timing), &group, false)
                .expect("preserve the economical production buffer choice");
        let speculative =
            select_timing_buffer(&catalog, &library, Some(source_timing), &group, true)
                .expect("choose the exactly faster load-relieving speculative buffer");

        assert_eq!(conservative.name, "BUF");
        assert_eq!(speculative.name, "BUF_FAST");
    }

    #[test]
    fn speculative_pass_buffers_critical_register_fanout_below_hard_limit() {
        let source = r#"
module top(clk, a, out);
  input clk;
  input a;
  output [3:0] out;
  wire root;
  DFF launch (.CLK(clk), .D(a), .Q(root));
  DFF capture0 (.CLK(clk), .D(root), .Q(out[0]));
  DFF capture1 (.CLK(clk), .D(root), .Q(out[1]));
  DFF capture2 (.CLK(clk), .D(root), .Q(out[2]));
  DFF capture3 (.CLK(clk), .D(root), .Q(out[3]));
endmodule
"#;
        let library = registered_timing_library();
        let (mut conservative_module, mut conservative_nets, mut conservative_interner) =
            parse_module(source);
        let conservative = insert_timing_aware_buffers(
            &mut conservative_module,
            &mut conservative_nets,
            &mut conservative_interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("preserve conservative production buffering for a legal fanout");

        let (mut module, mut nets, mut interner) = parse_module(source);
        let speculative = insert_speculative_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            &BufferTimingConstraints::default(),
        )
        .expect("speculatively isolate a legally driven critical register fanout");

        assert_eq!(conservative.buffers_inserted, 0);
        assert_eq!(speculative.max_fanout_before, 4);
        assert!(speculative.buffers_inserted > 0);
        assert!(speculative.final_worst_delay.unwrap() < speculative.initial_worst_delay.unwrap());
    }

    #[test]
    fn speculative_pass_buffers_loaded_critical_output_below_hard_limit() {
        let source = r#"
module top(clk, a, out);
  input clk;
  input a;
  output out;
  DFF launch (.CLK(clk), .D(a), .Q(out));
endmodule
"#;
        let library = registered_timing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        let sta_options = StaOptions {
            module_output_load: 0.5,
            ..StaOptions::default()
        };

        let stats = insert_speculative_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                module_output_load: sta_options.module_output_load,
                ..BufferOptions::default()
            },
            sta_options,
            &BufferTimingConstraints::default(),
        )
        .expect("speculatively improve a heavily loaded critical module output");

        assert_eq!(stats.max_fanout_before, 1);
        assert_eq!(stats.buffers_inserted, 1);
        assert!(stats.final_worst_delay.unwrap() < stats.initial_worst_delay.unwrap());
    }

    #[test]
    fn detects_only_slow_outputs_shared_with_internal_data_sinks() {
        let library = slow_shared_output_library();
        let (module, nets, interner) = parse_module(slow_shared_output_source());
        let sta_options = StaOptions {
            module_output_load: 0.6,
            ..StaOptions::default()
        };
        let options = BufferOptions {
            module_output_load: sta_options.module_output_load,
            ..BufferOptions::default()
        };

        assert!(
            has_slow_shared_primary_output(
                &module,
                &nets,
                &interner,
                &library,
                &options,
                sta_options,
            )
            .expect("recognize a slow output shared with a critical internal sink")
        );

        let (unshared_module, unshared_nets, unshared_interner) = parse_module(
            r#"
module top(a, b, c, direct, path);
  input a, b, c;
  output direct, path;
  WEAK_AND3 weak (.A(a), .B(b), .C(c), .Y(direct));
  BUF consumer (.A(a), .Y(path));
endmodule
"#,
        );
        assert!(
            !has_slow_shared_primary_output(
                &unshared_module,
                &unshared_nets,
                &unshared_interner,
                &library,
                &options,
                sta_options,
            )
            .expect("skip an output with no internal data consumers")
        );
    }

    #[test]
    fn speculative_pass_isolates_slow_output_from_critical_internal_sink() {
        let library = slow_shared_output_library();
        let (mut module, mut nets, mut interner) = parse_module(slow_shared_output_source());
        module.instances[1].type_name = interner.get_or_intern("BUF_FAST");
        let sta_options = StaOptions {
            module_output_load: 0.6,
            ..StaOptions::default()
        };
        let options = BufferOptions {
            module_output_load: sta_options.module_output_load,
            ..BufferOptions::default()
        };

        let stats = insert_speculative_timing_aware_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &options,
            sta_options,
            &BufferTimingConstraints::default(),
        )
        .expect("isolate the external load without buffering the critical internal sink");

        assert_eq!(stats.max_fanout_before, 2);
        assert_eq!(stats.buffers_inserted, 1);
        assert!(stats.final_worst_delay.unwrap() < stats.initial_worst_delay.unwrap());
        let direct = module
            .find_net_index(
                interner.get("direct").expect("resolve direct output"),
                &nets,
            )
            .expect("find direct output net");
        let weak_output = module.instances[0]
            .connections
            .iter()
            .find(|(pin, _)| interner.resolve(*pin) == Some("Y"))
            .expect("find weak logic output")
            .1
            .clone();
        let consumer_input = module.instances[1]
            .connections
            .iter()
            .find(|(pin, _)| interner.resolve(*pin) == Some("A"))
            .expect("find critical internal input")
            .1
            .clone();
        assert_eq!(weak_output, consumer_input);
        assert_ne!(weak_output, NetRef::Simple(direct));
    }

    #[test]
    fn consolidates_inserted_sibling_buffers_with_a_faster_cheaper_shared_driver() {
        let source = r#"
module top(a, out);
  input a;
  output [3:0] out;
  wire root;
  wire branch0;
  wire branch1;
  BUF_FAST driver (.A(a), .Y(root));
  BUF u_buf_0 (.A(root), .Y(branch0));
  BUF u_buf_1 (.A(root), .Y(branch1));
  BUF sink0 (.A(branch0), .Y(out[0]));
  BUF sink1 (.A(branch0), .Y(out[1]));
  BUF sink2 (.A(branch1), .Y(out[2]));
  BUF sink3 (.A(branch1), .Y(out[3]));
endmodule
"#;
        let mut builder = LibraryBuilder::from_library(sizing_library());
        builder
            .cells
            .iter_mut()
            .find(|cell| cell.name == "BUF_FAST")
            .expect("find the pin-compatible stronger shared buffer")
            .area = 1.5;
        let library = builder.finish();
        let (mut module, nets, mut interner) = parse_module(source);
        let before =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("time the independent original buffer branches");

        let stats = consolidate_timing_aware_buffers(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
            StaOptions::default(),
            before.max_delay.unwrap(),
        )
        .expect("replace sibling identity buffers with one stronger shared buffer");
        let after =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("independently verify the consolidated buffer tree");

        assert_eq!(stats.buffers_removed, 1);
        assert_eq!(stats.area_recovered, 0.5);
        assert!(after.cell_area < before.cell_area);
        assert!(after.max_delay.unwrap() < before.max_delay.unwrap());
        assert_eq!(stats.final_delay, after.max_delay.unwrap());
    }

    #[test]
    fn sibling_buffer_consolidation_preserves_explicit_per_stage_load_limits() {
        let source = r#"
module top(a, out);
  input a;
  output [3:0] out;
  wire root;
  wire branch0;
  wire branch1;
  BUF_FAST driver (.A(a), .Y(root));
  BUF u_buf_0 (.A(root), .Y(branch0));
  BUF u_buf_1 (.A(root), .Y(branch1));
  BUF sink0 (.A(branch0), .Y(out[0]));
  BUF sink1 (.A(branch0), .Y(out[1]));
  BUF sink2 (.A(branch1), .Y(out[2]));
  BUF sink3 (.A(branch1), .Y(out[3]));
endmodule
"#;
        let mut builder = LibraryBuilder::from_library(sizing_library());
        builder
            .cells
            .iter_mut()
            .find(|cell| cell.name == "BUF_FAST")
            .expect("find the pin-compatible stronger shared buffer")
            .area = 1.5;
        let library = builder.finish();
        let (mut module, nets, mut interner) = parse_module(source);
        let original_instance_count = module.instances.len();
        let before =
            build_netlist_report(&module, &nets, &interner, &library, StaOptions::default())
                .expect("time independent electrically valid sibling branches");

        let stats = consolidate_timing_aware_buffers(
            &mut module,
            &nets,
            &mut interner,
            &library,
            &BufferOptions {
                target_load: Some(0.3),
                ..BufferOptions::default()
            },
            StaOptions::default(),
            before.max_delay.unwrap(),
        )
        .expect("preserve the explicit per-stage load bound");

        assert_eq!(stats.buffers_removed, 0);
        assert_eq!(module.instances.len(), original_instance_count);
        assert_eq!(stats.final_delay, before.max_delay.unwrap());
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
