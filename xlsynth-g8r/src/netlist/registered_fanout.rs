// SPDX-License-Identifier: Apache-2.0

//! Capacitance-balanced buffer trees for overloaded register data outputs.
//!
//! The combinational mapper sees stage inputs as ideal sources. After assembly
//! the actual launch cell, downstream pin loads and capture setup arcs are
//! known. This pass keeps register masters, port connections and logic
//! functions fixed. It measures proposed trees with the same Liberty library
//! and wire model used for the original registered netlist. It does not
//! estimate routed delay or hold.

mod buffers;
mod candidates;
mod edit;
mod plan;
#[cfg(test)]
mod tests;

use self::buffers::Buffer;
use crate::liberty_model::Library;
use crate::netlist::editability::{EditableOutput as Root, analyze_editable_outputs};
use crate::netlist::io::ParsedNetlist;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::parse::NetRef;
use crate::netlist::register_mapping::AssembledRegisters;
use crate::netlist::report::build_area_report;
use crate::netlist::sta::{PreparedStaLibrary, StaOptions, StaReport};
use anyhow::{Context, Result, ensure};
use serde::Serialize;

/// Timing and area use the supplied Liberty units. Delay includes launch and
/// setup.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RegisteredFanoutScore {
    pub area: f64,
    pub register_to_register_delay: f64,
    pub register_to_output_delay: f64,
    pub capture_arrival_sum: f64,
    pub queries: crate::netlist::sta::TimingQueryDiagnosticCounts,
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
    pub movable_pins: usize,
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

/// Distinguishes an exhaustive bounded search from missing electrical evidence.
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisteredFanoutStatus {
    #[default]
    Unchanged,
    Improved,
    Unsupported,
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
    let result = search(&mut original, library, timing, assembled, &mut report);
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

fn search(
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
                let loads = root_load(r, timing);
                overload_ratio(loads[0].max(loads[1]), root_cap_ceiling(r)) > CAPACITY_TOLERANCE
                    || overload_ratio(fanout(r), r.electrical.limits.fanout) > CAPACITY_TOLERANCE
            })
    }) {
        return Ok(
            "no shared register output outside a known capacitance or fanout limit or requiring a slew check",
        );
    }
    let observed: Vec<_> = roots.iter().map(|r| r.source_bit).collect();
    let launches = graph.launches;
    let sta = PreparedStaLibrary::new(library)?;
    let analyze = |net: &MappedNetlist, sta: &PreparedStaLibrary| {
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
    let (base_sta, slews) = match analyze(original, &sta) {
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
    let mut ranked: Vec<_> = roots
        .iter()
        .enumerate()
        .filter_map(|(index, root)| {
            let overload = severity(root, timing);
            (slews[index].is_some() && root.sinks.len() >= 2 && overload > CAPACITY_TOLERANCE)
                .then_some((index, overload))
        })
        .collect();
    ranked.sort_unstable_by(|(i, a), (j, b)| b.total_cmp(a).then(i.cmp(j)));
    report.candidate_roots = ranked.len();
    report.overloaded = ranked
        .iter()
        .map(|&(id, _)| {
            let root = &roots[id];
            RegisteredFanoutDriver {
                instance: root.instance_name.clone(),
                pin: root.pin_name.clone(),
                master: library.cells[root.cell].name.clone(),
                protected_pins: root.fixed_sink_count,
                output_bits: root.electrical.load.output_ports,
                capacitance_limit: root_cap_ceiling(root),
                fanout_limit: root.electrical.limits.fanout,
                transition_limit: root.electrical.limits.transition,
                before: RegisteredFanoutElectrical {
                    movable_pins: root.sinks.len(),
                    load: root_load(root, timing),
                    fanout_load: fanout(root),
                    transition: root.electrical.transition,
                },
                after: None,
            }
        })
        .collect();
    if ranked.is_empty() {
        return Ok("all candidate register outputs within known limits");
    }
    let buffers = buffers::buffers(library, &sta);
    let choices = buffer_choices(&buffers, library);
    if choices.is_empty() {
        report.status = RegisteredFanoutStatus::Unsupported;
        return Ok("no characterized identity buffer");
    }
    let mut best = None;
    let mut best_rank = (2_u8, f64::INFINITY);
    candidates::visit_topology_candidates(
        original,
        library,
        timing,
        &roots,
        &ranked,
        &choices,
        |trial, proposal| {
            let buffers_added = proposal
                .plans
                .iter()
                .map(|(_, branches)| branches.len())
                .sum();
            let mut row = RegisteredFanoutTrial {
                buffer: library.cells[proposal.buffer.cell].name.clone(),
                branch_width: proposal.width,
                roots: proposal
                    .plans
                    .iter()
                    .map(|(i, _)| format!("{}.{}", roots[*i].instance_name, roots[*i].pin_name))
                    .collect(),
                buffers_added,
                area_added: buffers_added as f64 * library.cells[proposal.buffer.cell].area,
                score: None,
                disposition: String::new(),
            };
            match analyze(&trial, &sta) {
                Err(error) => {
                    row.disposition = format!("unpriced trial: {error:#}");
                    report.unpriced_trials += 1;
                }
                Ok((next, slews)) => {
                    let measured = score(&next, area + row.area_added)
                        .context("invalid score from registered buffer trial")?;
                    row.score = Some(measured);
                    let a = initial.queries;
                    let b = measured.queries;
                    let (rank, disposition) = if measured.register_to_register_delay <= 0.0 {
                        (None, "nonpositive trial register delay")
                    } else if !new_buffer_slews_ok(
                        &trial,
                        original.module.instances.len(),
                        proposal.buffer,
                        &next,
                    ) {
                        (None, "inserted buffer exceeds maximum transition")
                    } else if b.delay_slew_single_above_max_extrapolation_count
                        > a.delay_slew_single_above_max_extrapolation_count
                        || b.delay_slew_multiple_above_max_clamp_count
                            > a.delay_slew_multiple_above_max_clamp_count
                        || b.setup_above_max_clamp_count > a.setup_above_max_clamp_count
                    {
                        (None, "increased out-of-range timing queries")
                    } else if proposal.plans.iter().any(|(i, _)| slews[*i].is_none()) {
                        (None, "missing registered output slew")
                    } else {
                        let rank = rank(&base_sta, &next, initial, measured);
                        (
                            rank,
                            if rank.is_some() {
                                "improves"
                            } else {
                                "timing or area-delay did not improve"
                            },
                        )
                    };
                    row.disposition = disposition.into();
                    if let Some(rank) = rank
                        && rank < best_rank
                    {
                        best_rank = rank;
                        best = Some((proposal.buffer.clone(), proposal.plans.to_vec(), slews));
                        report.accepted_trial = Some(report.trials.len());
                    }
                }
            }
            report.trials.push(row);
            Ok(())
        },
    )?;
    if let Some((buffer, plans, slews)) = best {
        let mut changed = plans.iter().peekable();
        for (driver, &(id, _)) in report.overloaded.iter_mut().zip(&ranked) {
            if let Some((_, branches)) = changed.next_if(|(index, _)| *index == id) {
                let root = &roots[id];
                let top = branches.iter().filter(|b| b.parent.is_none()).count();
                let wire = timing.wire_load.constant_capacitance
                    + timing.wire_load.capacitance_per_sink * (top + root.fixed_sink_count) as f64;
                driver.after = Some(RegisteredFanoutElectrical {
                    movable_pins: top,
                    load: std::array::from_fn(|edge| {
                        root.fixed_load[edge] + wire + buffer.input_cap[edge] * top as f64
                    }),
                    fanout_load: root.fixed_fanout_load + top as f64 * buffer.input_fanout_load,
                    transition: slews[id],
                });
            }
        }
        let mut winner = original.clone();
        edit::apply_forest(&mut winner, library, &buffer, &roots, &plans)?;
        *original = winner;
        Ok("improved")
    } else {
        Ok("original retained")
    }
}

fn buffer_choices<'a>(buffers: &'a [Buffer], library: &Library) -> Vec<&'a Buffer> {
    let mut legal: Vec<_> = buffers
        .iter()
        .filter(|b| {
            b.max_capacitance.is_some_and(|c| c.is_finite() && c > 0.0)
                && b.input_cap.iter().all(|&c| c.is_finite() && c > 0.0)
                && library.cells[b.cell].area.is_finite()
                && library.cells[b.cell].area > 0.0
        })
        .collect();
    // Try the characterized reference master and a high electrical-gain option.
    // Prefer small area, then lexical master name for equal-gain choices.
    legal.sort_by(|a, b| {
        let gain = |c: &Buffer| c.max_capacitance.unwrap() / c.input_cap[0].max(c.input_cap[1]);
        gain(b)
            .total_cmp(&gain(a))
            .then(
                library.cells[a.cell]
                    .area
                    .total_cmp(&library.cells[b.cell].area),
            )
            .then(library.cells[a.cell].name.cmp(&library.cells[b.cell].name))
    });
    let mut selected = Vec::new();
    if let Ok(name) = crate::liberty::io_defaults::resolve_reference_buffer(library)
        && let Some(b) = legal.iter().find(|b| library.cells[b.cell].name == name)
    {
        selected.push(*b);
    }
    if let Some(b) = legal
        .into_iter()
        .find(|b| selected.first().is_none_or(|r| r.cell != b.cell))
    {
        selected.push(b);
    }
    selected
}

fn root_load(root: &Root, timing: StaOptions) -> [f64; 2] {
    // fixed_load includes unrewirable sink pins and external output port loads.
    let wire = timing.wire_load.constant_capacitance
        + timing.wire_load.capacitance_per_sink * root.electrical.load.pins as f64;
    root.electrical.load.capacitance.map(|cap| cap + wire)
}

fn output_slew(timing: &StaReport, net: &NetRef) -> Option<f64> {
    // STA's bus summary combines bits. Do not assign a sibling bit's slew to a
    // Q.
    let NetRef::Simple(index) = net else {
        return None;
    };
    let edges = timing.timing_for_net(*index)?;
    Some(edges.rise.transition.max(edges.fall.transition))
}

// Shared relative margin for finding overloaded roots and packing their
// sinks. A zero limit stays exact regardless of library units.
const CAPACITY_TOLERANCE: f64 = 1.0 + 1e-9;

fn overload_ratio(value: f64, bound: Option<f64>) -> f64 {
    match bound {
        Some(0.0) if value > 0.0 => f64::INFINITY,
        Some(limit) if limit > 0.0 => value / limit,
        _ => 0.0,
    }
}
fn fanout(r: &Root) -> f64 {
    r.electrical.load.weighted_fanout
}
fn root_cap_ceiling(root: &Root) -> Option<f64> {
    buffers::capacitance_ceiling(
        root.electrical.limits.capacitance,
        root.electrical.characterization.max_load,
    )
}
fn severity(root: &Root, timing: StaOptions) -> f64 {
    let [rise, fall] = root_load(root, timing);
    overload_ratio(rise.max(fall), root_cap_ceiling(root))
        .max(overload_ratio(
            root.electrical.transition.unwrap_or(0.0),
            root.electrical.limits.transition,
        ))
        .max(overload_ratio(fanout(root), root.electrical.limits.fanout))
}
fn new_buffer_slews_ok(
    net: &MappedNetlist,
    first: usize,
    buffer: &Buffer,
    sta: &StaReport,
) -> bool {
    let Some(limit) = buffer.max_transition else {
        return true;
    };
    net.module.instances[first..].iter().all(|instance| {
        instance.connections.iter().any(|(pin, signal)| {
            net.interner.resolve(*pin) == Some(buffer.output.as_str())
                && output_slew(sta, signal).is_some_and(|slew| slew.is_finite() && slew <= limit)
        })
    })
}

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

fn rank(
    original_sta: &StaReport,
    candidate_sta: &StaReport,
    original: RegisteredFanoutScore,
    candidate: RegisteredFanoutScore,
) -> Option<(u8, f64)> {
    let better = |a: f64, b: f64| a < b - 1e-9 * b.abs().max(1e-9);
    if better(
        original.register_to_register_delay,
        candidate.register_to_register_delay,
    ) || better(
        original.register_to_output_delay,
        candidate.register_to_output_delay,
    ) {
        return None;
    }
    let cost = candidate.area * candidate.register_to_register_delay;
    if better(
        candidate.register_to_register_delay,
        original.register_to_register_delay,
    ) {
        return better(cost, original.area * original.register_to_register_delay)
            .then_some((0, cost));
    }
    // Equal worst endpoints can hide an independently repaired launch. Admit a
    // tie only when no previously timed capture regresses and the sum of their
    // arrival times improves enough to pay for the additional buffer area.
    let tail_cost = candidate.area * candidate.capture_arrival_sum;
    if original.capture_arrival_sum > 0.0
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
    {
        Some((1, tail_cost))
    } else {
        None
    }
}
