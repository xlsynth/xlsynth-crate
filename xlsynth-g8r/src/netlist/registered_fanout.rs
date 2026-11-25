// SPDX-License-Identifier: Apache-2.0

//! Buffer overloaded register data outputs in a single trial.
//! Assign sinks round-robin and split leaves to meet electrical limits.
//!
//! The combinational mapper sees stage inputs as ideal sources. After assembly
//! the actual launch cell, downstream pin loads and capture setup arcs are
//! known. This pass keeps register masters, port connections and logic
//! functions fixed. It measures proposed trees with the same Liberty library
//! and wire model used for the original registered netlist. It does not
//! estimate routed delay or hold.

mod buffers;
mod edit;
mod plan;
#[cfg(test)]
mod tests;

use self::buffers::InsertableBufferCell;
use crate::liberty_model::Library;
use crate::netlist::editability::{EditableOutput, analyze_editable_outputs};
use crate::netlist::electrical::RiseFall;
use crate::netlist::io::ParsedNetlist;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::parse::NetRef;
use crate::netlist::register_mapping::AssembledRegisters;
use crate::netlist::report::build_area_report;
use crate::netlist::sta::{PreparedStaLibrary, StaOptions, StaReport, TimingQueryDiagnosticCounts};
use anyhow::{Context, Result, ensure};
use serde::Serialize;

// Empirically chosen tradeoff between delay and buffer area.
// Load limits may require fewer sinks per buffer.
const DEFAULT_BRANCH_WIDTH: usize = 12;

/// Timing and area use the supplied Liberty units. Delay includes launch and
/// setup.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RegisteredFanoutScore {
    pub area: f64,
    pub register_to_register_delay: f64,
    pub register_to_output_delay: f64,
    pub capture_arrival_sum: f64,
    pub queries: TimingQueryDiagnosticCounts,
}

/// A measured batch uses one buffer master and a branching limit for every
/// tree.
#[derive(Debug, Serialize)]
pub struct RegisteredFanoutTrial {
    pub buffer: String,
    pub branch_width: usize,
    pub roots: Vec<String>,
    pub buffers_added: usize,
    pub area_added: f64,
    pub score: Option<RegisteredFanoutScore>,
    pub disposition: String,
}

/// Measured before/after electrical state in Liberty units. An accepted edit
/// may improve a violation without eliminating it.
#[derive(Debug, Serialize)]
pub struct RegisteredFanoutElectrical {
    #[serde(rename = "movable_pins")]
    pub data_sink_count: usize,
    /// Capacitance for a rising transition, then for a falling transition.
    pub load: [f64; 2],
    pub fanout_load: f64,
    pub transition: Option<f64>,
}

/// Constant driver identity and limits, with both electrical measurements in
/// one record. `after` is present only for a driver changed by the winner.
#[derive(Debug, Serialize)]
pub struct RegisteredFanoutDriver {
    pub instance: String,
    pub pin: String,
    pub master: String,
    pub protected_pins: usize,
    pub output_bits: usize,
    pub capacitance_limit: Option<f64>,
    pub fanout_limit: Option<f64>,
    pub transition_limit: Option<f64>,
    pub before: RegisteredFanoutElectrical,
    pub after: Option<RegisteredFanoutElectrical>,
}

/// Outcome of the repair attempt, including gaps in electrical or timing data.
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisteredFanoutStatus {
    #[default]
    Unchanged,
    Improved,
    Unsupported,
    /// The proposed trees could not be timed. The original netlist is intact.
    PartialSearch,
}

/// `accepted_trial` indexes `trials`. When absent the netlist is unchanged.
#[derive(Debug, Default, Serialize)]
pub struct RegisteredFanoutReport {
    pub candidate_roots: usize,
    pub shared_outputs: usize,
    pub overloaded: Vec<RegisteredFanoutDriver>,
    pub initial: Option<RegisteredFanoutScore>,
    pub accepted_trial: Option<usize>,
    pub trials: Vec<RegisteredFanoutTrial>,
    pub outcome: String,
    pub status: RegisteredFanoutStatus,
    pub unpriced_trials: usize,
}

/// Repair register Q fanout in one assembled mapped module. The original is a
/// candidate. Missing timing coverage is reported explicitly. Malformed inputs
/// and invalid generated edits return errors with the original preserved. Input
/// and output transition/load use Liberty units.
pub fn repair_registered_fanout(
    netlist: &mut ParsedNetlist,
    library: &Library,
    timing: StaOptions,
    assembled: Option<AssembledRegisters<'_>>,
) -> Result<RegisteredFanoutReport> {
    let mut report = RegisteredFanoutReport::default();
    if netlist.modules.len() != 1 {
        report.status = RegisteredFanoutStatus::Unsupported;
        report.outcome = "requires one assembled module".into();
        return Ok(report);
    }
    let mut original = MappedNetlist {
        module: netlist.modules.pop().unwrap(),
        nets: std::mem::take(&mut netlist.nets),
        interner: std::mem::take(&mut netlist.interner),
    };
    // Trial netlists own their added wires and symbols. Only a selected winner
    // replaces the original, including on a failed library or timing query.
    let result = try_buffering(&mut original, library, timing, assembled, &mut report);
    netlist.modules.push(original.module);
    netlist.nets = original.nets;
    netlist.interner = original.interner;
    let reason = result?;
    if report.outcome.is_empty() {
        report.outcome = reason.into();
    }
    if report.unpriced_trials > 0 {
        report.status = RegisteredFanoutStatus::PartialSearch;
    } else if report.accepted_trial.is_some() {
        report.status = RegisteredFanoutStatus::Improved;
    }
    Ok(report)
}

/// Time one group of feasible trees and apply it only if accepted.
fn try_buffering(
    original: &mut MappedNetlist,
    library: &Library,
    timing: StaOptions,
    assembled: Option<AssembledRegisters<'_>>,
    report: &mut RegisteredFanoutReport,
) -> Result<&'static str> {
    let graph = analyze_editable_outputs(original, library, timing.module_output_load, assembled)?;
    let mut roots = graph.roots;
    // This focused pass keeps the entire generated control cone unchanged.
    roots.retain(|r| !r.in_control_fanin);
    report.shared_outputs = roots.iter().filter(|r| r.sinks.len() >= 2).count();
    if !roots.iter().any(|r| {
        r.sinks.len() >= 2
            && (r.electrical.limits.transition.is_some() || {
                let loads = driver_load_capacitance(r, timing);
                value_limit_ratio(loads.rise.max(loads.fall), driver_output_load_ceiling(r))
                    > CAPACITY_TOLERANCE
                    || value_limit_ratio(fanout_load(r), r.electrical.limits.fanout)
                        > CAPACITY_TOLERANCE
            })
    }) {
        return Ok(
            "no shared register output outside a known capacitance or fanout limit or requiring a slew check",
        );
    }
    let launches = graph.launches;
    let sta = PreparedStaLibrary::new(library)?;
    let analyze = |net: &MappedNetlist, drivers: &[EditableOutput]| {
        let observed: Vec<_> = drivers.iter().map(|root| root.source_bit).collect();
        sta.analyze_register_boundary_with_slews(
            &net.module,
            &net.nets,
            &net.interner,
            timing,
            false,
            &launches,
            &observed,
        )
    };
    let (base_sta, slews) = match analyze(original, &roots) {
        Ok(result) => result,
        Err(error) => {
            report.status = RegisteredFanoutStatus::Unsupported;
            report.outcome = format!("could not time the original register boundaries: {error:#}");
            return Ok("unpriced original");
        }
    };
    for (root, slew) in roots.iter_mut().zip(&slews) {
        root.electrical.transition = *slew;
    }
    if base_sta.worst_register_input_breakdown.is_none() {
        return Ok("no timed register-to-register path");
    }
    let area = build_area_report(&original.module, &original.interner, library)?.area;
    let initial = score(&base_sta, area)?;
    report.initial = Some(initial);
    // Negative setup constraints are legal. At nonpositive delay the
    // area-delay product ceases to reward smaller circuits.
    if initial.register_to_register_delay <= 0.0 {
        return Ok("nonpositive original register delay");
    }
    roots.retain(|root| {
        root.electrical.transition.is_some()
            && root.sinks.len() >= 2
            && value_limit_ratio_max(root, timing) > CAPACITY_TOLERANCE
    });
    if roots.is_empty() {
        return Ok("all candidate register outputs within known limits");
    }
    report.candidate_roots = roots.len();
    report.overloaded = roots
        .iter()
        .map(|root| describe_driver(root, library, timing))
        .collect();
    let Some(buffer) = &buffers::highest_gain_buffer(library, &sta) else {
        report.status = RegisteredFanoutStatus::Unsupported;
        return Ok("no characterized identity buffer");
    };
    // Keep indices into report rows and trial slews when skipping a root.
    let plans: Vec<_> = (0..roots.len())
        .filter_map(|i| plan::build_buffer_tree(&roots, i, buffer, DEFAULT_BRANCH_WIDTH, timing))
        .collect();
    if plans.is_empty() {
        return Ok("original retained");
    }
    let mut trial = original.clone();
    edit::apply_buffer_trees(&mut trial, library, buffer, &roots, &plans)?;
    let buffers_added = plans.iter().map(|tree| tree.buffers.len()).sum();
    let mut row = RegisteredFanoutTrial {
        buffer: library.cells[buffer.cell_index].name.clone(),
        branch_width: DEFAULT_BRANCH_WIDTH,
        roots: plans
            .iter()
            .map(|tree| {
                let root = &roots[tree.driver_index];
                format!("{}.{}", root.instance_name, root.pin_name)
            })
            .collect(),
        buffers_added,
        area_added: buffers_added as f64 * library.cells[buffer.cell_index].area,
        score: None,
        disposition: String::new(),
    };
    let (next, slews) = match analyze(&trial, &roots) {
        Ok(result) => result,
        Err(error) => {
            row.disposition = format!("unpriced trial: {error:#}");
            report.unpriced_trials += 1;
            report.trials.push(row);
            return Ok("original retained");
        }
    };
    let measured = score(&next, area + row.area_added)
        .context("invalid score from registered buffer trial")?;
    row.score = Some(measured);
    let a = initial.queries;
    let b = measured.queries;
    let rejection = if measured.register_to_register_delay <= 0.0 {
        Some("nonpositive trial register delay")
    } else if !new_buffer_slews_ok(&trial, original.module.instances.len(), buffer, &next) {
        Some("inserted buffer exceeds maximum transition")
    } else if b.delay_slew_single_above_max_extrapolation_count
        > a.delay_slew_single_above_max_extrapolation_count
        || b.delay_slew_multiple_above_max_clamp_count > a.delay_slew_multiple_above_max_clamp_count
        || b.setup_above_max_clamp_count > a.setup_above_max_clamp_count
    {
        Some("increased out-of-range timing queries")
    } else if plans.iter().any(|tree| slews[tree.driver_index].is_none()) {
        Some("missing registered output slew")
    } else if improves(&base_sta, &next, initial, measured) {
        None
    } else {
        Some("timing or area-delay did not improve")
    };
    row.disposition = rejection.unwrap_or("improves").to_owned();
    report.trials.push(row);
    if rejection.is_some() {
        return Ok("original retained");
    }
    report.accepted_trial = Some(0);
    record_buffered_drivers(report, &roots, &plans, buffer, timing, &slews);
    *original = trial;
    Ok("improved")
}

/// Describe an overloaded driver before reconnecting its data input pins.
fn describe_driver(
    root: &EditableOutput,
    library: &Library,
    timing: StaOptions,
) -> RegisteredFanoutDriver {
    let load = driver_load_capacitance(root, timing);
    RegisteredFanoutDriver {
        instance: root.instance_name.clone(),
        pin: root.pin_name.clone(),
        master: library.cells[root.cell].name.clone(),
        protected_pins: root.fixed_sink_count,
        output_bits: root.electrical.load.output_ports,
        capacitance_limit: driver_output_load_ceiling(root),
        fanout_limit: root.electrical.limits.fanout,
        transition_limit: root.electrical.limits.transition,
        before: RegisteredFanoutElectrical {
            data_sink_count: root.sinks.len(),
            load: [load.rise, load.fall],
            fanout_load: fanout_load(root),
            transition: root.electrical.transition,
        },
        after: None,
    }
}

/// Record the load and slew of each driver whose tree was accepted.
fn record_buffered_drivers(
    report: &mut RegisteredFanoutReport,
    roots: &[EditableOutput],
    plans: &[plan::BufferTreePlan],
    buffer: &InsertableBufferCell,
    timing: StaOptions,
    slews: &[Option<f64>],
) {
    for tree in plans {
        let root = &roots[tree.driver_index];
        let top = tree
            .buffers
            .iter()
            .filter(|b| b.driving_buffer_index.is_none())
            .count();
        let wire = timing.wire_load.constant_capacitance
            + timing.wire_load.capacitance_per_sink * (top + root.fixed_sink_count) as f64;
        let load = RiseFall {
            rise: root.fixed_load.rise + wire + buffer.input_capacitance.rise * top as f64,
            fall: root.fixed_load.fall + wire + buffer.input_capacitance.fall * top as f64,
        };
        report.overloaded[tree.driver_index].after = Some(RegisteredFanoutElectrical {
            data_sink_count: top,
            load: [load.rise, load.fall],
            fanout_load: root.fixed_fanout_load + top as f64 * buffer.input_fanout_load,
            transition: slews[tree.driver_index],
        });
    }
}

/// Driver load including input pins, output ports, and wire capacitance.
fn driver_load_capacitance(root: &EditableOutput, timing: StaOptions) -> RiseFall<f64> {
    let wire = timing.wire_load.constant_capacitance
        + timing.wire_load.capacitance_per_sink * root.electrical.load.pins as f64;
    let cap = root.electrical.load.capacitance;
    RiseFall {
        rise: cap.rise + wire,
        fall: cap.fall + wire,
    }
}

/// Return the slower transition on a scalar buffer output.
fn output_slew(timing: &StaReport, net: &NetRef) -> Option<f64> {
    // A bus summary can report the transition of a different bit.
    let NetRef::Simple(index) = net else {
        return None;
    };
    let edges = timing.timing_for_net(*index)?;
    Some(edges.rise.transition.max(edges.fall.transition))
}

// Shared relative margin for finding overloaded roots and packing their
// sinks. A zero limit stays exact regardless of library units.
const CAPACITY_TOLERANCE: f64 = 1.0 + 1e-9;

/// Returns `value / limit`, zero for a missing limit, and infinity
/// when a positive value exceeds a zero limit.
fn value_limit_ratio(value: f64, bound: Option<f64>) -> f64 {
    match bound {
        Some(0.0) if value > 0.0 => f64::INFINITY,
        Some(limit) if limit > 0.0 => value / limit,
        _ => 0.0,
    }
}

/// Total fanout load in Liberty units, including fixed connections.
fn fanout_load(r: &EditableOutput) -> f64 {
    r.electrical.load.weighted_fanout
}

/// Respect both the pin's capacitance limit and the timing table range.
fn driver_output_load_ceiling(root: &EditableOutput) -> Option<f64> {
    buffers::capacitance_ceiling(
        root.electrical.limits.capacitance,
        root.electrical.characterization.max_load,
    )
}

/// Returns the largest value/limit ratio across capacitance,
/// output transition time, and weighted fanout.
///
/// A ratio above 1 exceeds at least one bound.
/// Capacitance also considers the timing-table range.
fn value_limit_ratio_max(root: &EditableOutput, timing: StaOptions) -> f64 {
    let load = driver_load_capacitance(root, timing);
    value_limit_ratio(load.rise.max(load.fall), driver_output_load_ceiling(root))
        .max(value_limit_ratio(
            root.electrical.transition.unwrap_or(0.0),
            root.electrical.limits.transition,
        ))
        .max(value_limit_ratio(
            fanout_load(root),
            root.electrical.limits.fanout,
        ))
}

/// Check each inserted buffer output against its Liberty transition limit.
fn new_buffer_slews_ok(
    net: &MappedNetlist,
    first: usize,
    buffer: &InsertableBufferCell,
    sta: &StaReport,
) -> bool {
    let Some(limit) = buffer.max_transition else {
        return true;
    };
    net.module.instances[first..].iter().all(|instance| {
        instance.connections.iter().any(|(pin, signal)| {
            net.interner.resolve(*pin) == Some(buffer.output_pin_name.as_str())
                && output_slew(sta, signal).is_some_and(|slew| slew.is_finite() && slew <= limit)
        })
    })
}

/// Collect registered path delays, area and timing table diagnostics.
fn score(sta: &StaReport, area: f64) -> Result<RegisteredFanoutScore> {
    let s = RegisteredFanoutScore {
        area,
        register_to_register_delay: sta.worst_register_input_arrival,
        register_to_output_delay: sta.worst_output_arrival,
        capture_arrival_sum: sta.register_input_arrivals.iter().flatten().sum(),
        queries: sta.timing_query_diagnostic_counts,
    };
    ensure!(
        sta.worst_register_input_breakdown.is_some()
            && s.area.is_finite()
            && s.area > 0.0
            && s.register_to_register_delay.is_finite()
            && s.capture_arrival_sum.is_finite()
            && s.register_to_output_delay.is_finite(),
        "invalid registered score"
    );
    Ok(s)
}

/// Require a lower area-delay product on a faster register path. On a
/// delay tie, compare area times the sum of capture arrivals instead.
fn improves(
    original_sta: &StaReport,
    candidate_sta: &StaReport,
    original: RegisteredFanoutScore,
    candidate: RegisteredFanoutScore,
) -> bool {
    let better = |a: f64, b: f64| a < b - 1e-9 * b.abs().max(1e-9);
    if better(
        original.register_to_register_delay,
        candidate.register_to_register_delay,
    ) || better(
        original.register_to_output_delay,
        candidate.register_to_output_delay,
    ) {
        return false;
    }
    let cost = candidate.area * candidate.register_to_register_delay;
    if better(
        candidate.register_to_register_delay,
        original.register_to_register_delay,
    ) {
        return better(cost, original.area * original.register_to_register_delay);
    }
    // Equal worst endpoints can hide an independently repaired launch. Admit a
    // tie only when no previously timed capture regresses and the sum of their
    // arrival times improves enough to pay for the additional buffer area.
    let tail_cost = candidate.area * candidate.capture_arrival_sum;
    original.capture_arrival_sum > 0.0
        && candidate.capture_arrival_sum > 0.0
        && better(tail_cost, original.area * original.capture_arrival_sum)
        && original_sta
            .register_input_arrivals
            .iter()
            .zip(&candidate_sta.register_input_arrivals)
            .all(|(old, new)| match (old, new) {
                (Some(a), Some(b)) => !better(*a, *b),
                (None, None) => true,
                _ => false,
            })
}
