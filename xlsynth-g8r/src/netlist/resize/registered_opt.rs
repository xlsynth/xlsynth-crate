// SPDX-License-Identifier: Apache-2.0

//! Optimize an assembled gate cover with its physical registers attached.
//! Compare buffering and cell substitutions from the same
//! starting circuit, using timing that includes launch and capture arcs.

use std::sync::Arc;

use super::timing::{ElectricalGuard, ResizeTimingPolicy, TimingTieBreak};
use super::{
    BoundaryDelays, CellFamilies, ResizeOptions, ResizeSearch, SearchPhase, verify_timing,
};
use crate::liberty_model::Library;
use crate::netlist::fanout_opt::FanoutOptimizationOptions;
use crate::netlist::io::ParsedNetlist;
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::registered_fanout::{
    ForestRetry, OwnedBuffers, TopologyTrial, visit_topology_candidates,
};
use crate::netlist::sta::incremental::{
    IncrementalSta, TIMING_COMPARISON_TOLERANCE, TimingEndpoint,
    measure_boundary_delays_with_library,
};
use crate::netlist::sta::{PreparedStaLibrary, StaOptions};
use anyhow::{Context, Result, ensure};
use serde::Serialize;

/// Selects additional topology proposals for one registered netlist. Both
/// policies use identical electrical, whole-stage timing and area acceptance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RegisteredCandidateSelection {
    /// Original bounded severity/fanout order.
    #[default]
    Legacy,
    /// Also evaluate joint repairs on critical and near-critical
    /// register-to-register paths.
    TimingCones,
}

/// Timing model and optional buffering and cell-sizing policies for optimizing
/// a netlist with its boundary registers attached.
#[derive(Clone, Debug)]
pub struct RegisteredOptimizationOptions {
    /// Shared launch, load and wire model for all candidate edits.
    pub timing: StaOptions,
    pub fanout: Option<FanoutOptimizationOptions>,
    pub resize: Option<ResizeOptions>,
}

/// Area penalty and ceiling for fanout edits, relative to the assembled cover.
#[derive(Clone, Copy)]
pub(crate) struct FanoutAreaPolicy {
    pub(crate) weight: f64,
    pub(crate) limit: f64,
}

impl FanoutAreaPolicy {
    pub(crate) const UNBOUNDED: Self = Self {
        weight: 0.5,
        limit: f64::INFINITY,
    };
}

impl RegisteredOptimizationOptions {
    pub(crate) fn fanout_area_policy(&self, initial_area: f64) -> FanoutAreaPolicy {
        match &self.fanout {
            Some(fanout) => FanoutAreaPolicy {
                weight: fanout.area_weight,
                limit: initial_area * (1.0 + fanout.area_growth_limit),
            },
            None => FanoutAreaPolicy::UNBOUNDED,
        }
    }
}

/// Area and boundary timing before and after optimization, with trial counts
/// and the edit selected in each search round.
#[derive(Debug, Serialize)]
pub struct RegisteredOptimizationReport {
    pub initial_area: f64,
    pub initial_delay: BoundaryDelays,
    pub final_area: f64,
    pub final_delay: BoundaryDelays,
    /// Timing of the registered stage, or worst available boundary when it has
    /// no register path.
    pub objective_delay: f64,
    pub timing_evaluations: usize,
    pub recovery_evaluations: usize,
    pub topology_trials: usize,
    pub unpriced_topologies: usize,
    pub rounds: Vec<RegisteredOptimizationRound>,
}

/// Trial counts include rejected proposals. Committed edits describe the
/// winning candidate.
#[derive(Debug, Serialize)]
pub struct RegisteredOptimizationRound {
    pub topology_trials: usize,
    pub electrical_retry_trials: usize,
    /// Largest number of drivers evaluated together in each timing window,
    /// first critical and then near-critical.
    pub timing_cone_root_counts: [usize; 2],
    pub timing_cone_trials: usize,
    pub accepted_timing_cone: bool,
    /// Window and original group size of the winner, including an electrical
    /// retry.
    pub accepted_timing_cone_window: Option<usize>,
    pub accepted_timing_cone_roots: Option<usize>,
    pub timing_trials: usize,
    pub recovery_trials: usize,
    pub renewed_recovery_windows: usize,
    pub accepted: &'static str,
    pub added_cells: usize,
    pub changed_cell_masters: usize,
    pub area: f64,
    pub delay: f64,
}

/// Optimize one covered and assembled module. All four boundary delays are
/// guarded. The input is replaced only after an independent full STA verifies
/// the winner. There is no inferred clock constraint or estimate of routed
/// interconnect.
#[cfg(test)]
pub(crate) fn optimize_registered_netlist(
    netlist: &mut ParsedNetlist,
    library: &Library,
    options: &RegisteredOptimizationOptions,
) -> Result<RegisteredOptimizationReport> {
    optimize_registered_netlist_tracked(netlist, library, options, None)
}

/// Run the same search, carrying data-buffer ownership across selected edits.
/// The ledger and netlist are committed together after full timing
/// verification.
#[cfg(test)]
pub(crate) fn optimize_registered_netlist_tracked(
    netlist: &mut ParsedNetlist,
    library: &Library,
    options: &RegisteredOptimizationOptions,
    ownership: Option<&mut OwnedBuffers>,
) -> Result<RegisteredOptimizationReport> {
    optimize_registered_stage(
        netlist,
        library,
        options,
        ElectricalGuard::Absolute,
        ownership,
        RegisteredCandidateSelection::Legacy,
    )
}

/// Run topology proposal selection on the same snapshot and under the same
/// whole-stage acceptance checks as ordinary sizing and recovery.
pub(crate) fn optimize_registered_stage(
    netlist: &mut ParsedNetlist,
    library: &Library,
    options: &RegisteredOptimizationOptions,
    electrical_guard: ElectricalGuard,
    ownership: Option<&mut OwnedBuffers>,
    selection: RegisteredCandidateSelection,
) -> Result<RegisteredOptimizationReport> {
    let mut current_owned = ownership.as_deref().cloned();
    let [module] = netlist.modules.as_slice() else {
        anyhow::bail!("registered electrical optimization requires one assembled module");
    };
    let mut current = MappedNetlist {
        module: module.clone(),
        nets: netlist.nets.clone(),
        interner: netlist.interner.clone(),
    };
    let prepared = Arc::new(PreparedStaLibrary::new(library)?);
    let analyze = |n: &MappedNetlist| {
        IncrementalSta::with_registered_library(
            &n.module,
            &n.nets,
            &n.interner,
            Arc::clone(&prepared),
            options.timing,
        )
    };
    let mut sta = analyze(&current)?;
    let mut objective = sta.objective();
    let mut score = Score::measure(&sta, library, objective)?;
    let FanoutAreaPolicy {
        weight,
        limit: topology_area_limit,
    } = options.fanout_area_policy(score.area);
    ensure!(
        weight.is_finite() && weight >= 0.0,
        "area weight must be finite and nonnegative"
    );
    let mut topology_rounds = 0;
    if let Some(f) = &options.fanout {
        ensure!(
            f.area_growth_limit.is_finite() && f.area_growth_limit >= 0.0,
            "invalid fanout area growth limit"
        );
        topology_rounds = f.max_rounds;
    }
    let mut sizing = options.resize.clone().unwrap_or_default();
    if options.resize.is_none() {
        sizing.max_upsize_rounds = 0;
        sizing.max_downsize_rounds = 0;
    }
    ensure!(
        sizing.max_area_growth.is_finite() && sizing.max_area_growth >= 0.0,
        "invalid resize area growth limit"
    );
    let base_window = sizing
        .work_limits(sta.instance_count())
        .recovery_evaluations;
    let automatic = sizing.recovery_evaluation_budget.is_none();
    let mut limits = sizing.registered_limits(sta.instance_count());
    ensure!(
        limits.timing_candidate_sites > 0 && sizing.max_cell_candidates > 0,
        "candidate limits must be positive"
    );
    let families = options.resize.as_ref().map(|_| CellFamilies::new(library));
    let mut timing_rounds = sizing.max_upsize_rounds;
    let mut recovery_rounds = sizing.max_downsize_rounds;
    let mut sizing_area_limit = score.area * (1.0 + sizing.max_area_growth);
    let mut report = RegisteredOptimizationReport {
        initial_area: score.area,
        final_area: score.area,
        initial_delay: score.boundaries,
        final_delay: score.boundaries,
        objective_delay: score.delay,
        timing_evaluations: 0,
        recovery_evaluations: 0,
        topology_trials: 0,
        unpriced_topologies: 0,
        rounds: Vec::new(),
    };
    // Compare wiring and sizing from the same starting circuit in each round.
    // The wiring move buffers or copies a driver. The sizing move swaps cells.
    // Then recover area on the winning circuit under its new timing limits.
    while topology_rounds > 0 || timing_rounds > 0 || recovery_rounds > 0 {
        if objective != sta.objective() {
            objective = sta.objective();
            score = Score::measure(&sta, library, objective)?;
        }
        let mut topology = None;
        let mut best_rank = None;
        let mut round = RegisteredOptimizationRound {
            topology_trials: 0,
            electrical_retry_trials: 0,
            timing_cone_root_counts: [0; 2],
            timing_cone_trials: 0,
            accepted_timing_cone: false,
            accepted_timing_cone_window: None,
            accepted_timing_cone_roots: None,
            timing_trials: 0,
            recovery_trials: 0,
            renewed_recovery_windows: 0,
            accepted: "none",
            added_cells: 0,
            changed_cell_masters: 0,
            area: score.area,
            delay: score.delay,
        };
        if let Some(fanout) = &options.fanout
            && topology_rounds > 0
        {
            topology_rounds -= 1;
            visit_topology_candidates(
                &current,
                library,
                &prepared,
                options.timing,
                fanout,
                (selection == RegisteredCandidateSelection::TimingCones)
                    .then_some((&sta, objective)),
                |candidate, kind, edits| {
                    if let TopologyTrial::TimingCone(cone) = kind {
                        round.timing_cone_trials += 1;
                        round.timing_cone_root_counts[cone.window] =
                            round.timing_cone_root_counts[cone.window].max(cone.roots);
                    }
                    round.topology_trials += 1;
                    round.electrical_retry_trials +=
                        usize::from(matches!(kind, TopologyTrial::ElectricalRetry(_)));
                    let trial = match analyze(&candidate) {
                        Ok(trial) => trial,
                        Err(error) => {
                            log::debug!("unpriced topology: {error:#}");
                            report.unpriced_topologies += 1;
                            return Ok(None);
                        }
                    };
                    let next = Score::measure(&trial, library, objective)?;
                    if let Some(rank) = next.rank(score, weight)
                        && next.boundaries.within(report.initial_delay)
                        && best_rank.is_none_or(|best| rank < best)
                        && preserves_endpoints(&sta, &trial)
                        && let Some((improved, worse_outputs)) = electrical_progress(
                            &sta,
                            &trial,
                            electrical_guard,
                            if matches!(kind, TopologyTrial::Forest | TopologyTrial::TimingCone(_))
                            {
                                fanout.drivers_per_round
                            } else {
                                0
                            },
                        )
                        && (next.area <= topology_area_limit || improved)
                    {
                        if !worse_outputs.is_empty() {
                            return Ok(Some(ForestRetry {
                                rank,
                                outputs: worse_outputs,
                            }));
                        }
                        let mut candidate_owned = current_owned.clone();
                        if let Some(ref mut owned) = candidate_owned {
                            edits.record(&candidate, current.module.instances.len(), owned);
                        }
                        topology = Some((candidate, trial, next, candidate_owned, kind));
                        best_rank = Some(rank);
                    }
                    Ok(None)
                },
            )?;
        }
        report.topology_trials += round.topology_trials;
        // Compare cell swaps and wiring edits from the same starting circuit.
        // Spend the area-recovery budget only on the selected circuit; trying
        // to recover area on candidates that lose would waste the budget.
        let mut timing_changes = 0;
        let mut next = score;
        let mut search = families.as_ref().map(|families| ResizeSearch {
            electrical_guard,
            library,
            options: &sizing,
            families,
            area_weight: weight,
            initial_delay: report.initial_delay,
            timing_checkpoint: score.boundaries,
            area: score.area,
            area_limit: sizing_area_limit,
            work_limits: limits,
        });
        if let Some(search) = &mut search
            && timing_rounds > 0
            && limits.timing_evaluations > 0
        {
            let stats = search.run_phase(SearchPhase::Timing, &mut sta, 1);
            limits.timing_evaluations -= stats.propagated_instances;
            report.timing_evaluations += stats.propagated_instances;
            round.timing_trials = stats.trials;
            timing_changes = stats.accepted;
            if timing_changes > 0 {
                next = Score::measure(&sta, library, objective)?;
            }
        }
        if timing_changes > 0
            && next
                .rank(score, weight)
                .is_some_and(|rank| best_rank.is_none_or(|best| rank <= best))
        {
            round.accepted = "cell_substitutions";
            timing_rounds -= 1;
            score = next;
        } else if let Some((candidate, timing, accepted, candidate_owned, kind)) = topology {
            sizing_area_limit += accepted.area - score.area;
            round.added_cells = candidate.module.instances.len() - current.module.instances.len();
            round.accepted = "topology";
            let cone = kind.timing_cone();
            round.accepted_timing_cone = cone.is_some();
            round.accepted_timing_cone_window = cone.map(|c| c.window);
            round.accepted_timing_cone_roots = cone.map(|c| c.roots);
            current = candidate;
            current_owned = candidate_owned;
            sta = timing;
            score = accepted;
        } else if timing_changes > 0 {
            sta = analyze(&current)?;
        }
        let timing_winner = round.accepted == "cell_substitutions";
        let mut recovered = 0;
        let stable = round.accepted == "none";
        // Preserve one recovery opportunity for the last selected topology.
        // A productive sweep there can finish scanning its pending sites.
        let allowance = if automatic && !stable {
            base_window.min(limits.recovery_evaluations.saturating_sub(base_window))
        } else {
            limits.recovery_evaluations
        };
        if let Some(search) = &mut search
            && recovery_rounds > 0
            && allowance > 0
            && (!automatic || stable || recovery_rounds > 1)
        {
            search.timing_checkpoint = score.boundaries;
            search.area = score.area;
            search.area_limit = sizing_area_limit;
            search.work_limits.recovery_evaluations = allowance;
            let stats = search.run_phase_with_windows(
                SearchPhase::AreaRecovery,
                &mut sta,
                1,
                (automatic && stable).then_some(base_window),
            );
            round.renewed_recovery_windows = stats.renewed_windows;
            limits.recovery_evaluations -= stats.propagated_instances;
            report.recovery_evaluations += stats.propagated_instances;
            round.recovery_trials = stats.trials;
            recovered = stats.accepted;
            if recovered > 0 {
                recovery_rounds -= 1;
                score = Score::measure(&sta, library, objective)?;
                if round.accepted == "none" {
                    round.accepted = "cell_substitutions";
                }
            }
        }
        if timing_winner || recovered > 0 {
            for (inst, &cell) in current.module.instances.iter_mut().zip(sta.cell_indices()) {
                let master = current.interner.get_or_intern(&library.cells[cell].name);
                round.changed_cell_masters += usize::from(master != inst.type_name);
                inst.type_name = master;
            }
        }
        if round.accepted == "none" {
            report.rounds.push(round);
            break;
        }
        round.area = score.area;
        round.delay = score.delay;
        report.rounds.push(round);
    }
    let full = measure_boundary_delays_with_library(
        &current.module,
        &current.nets,
        &current.interner,
        &prepared,
        options.timing,
    )
    .context("checking all timing boundaries of the assembled winner")?;
    verify_timing(score.boundaries, full)?;
    ensure!(
        full.within(report.initial_delay),
        "assembled winner regresses a timing boundary"
    );
    report.final_area = score.area;
    report.final_delay = full;
    report.objective_delay = full.values()[sta.objective()];
    *netlist = current.into();
    if let (Some(dst), Some(src)) = (ownership, current_owned) {
        *dst = src;
    }
    Ok(report)
}

#[derive(Clone, Copy)]
pub(crate) struct Score {
    boundaries: BoundaryDelays,
    pub(crate) delay: f64,
    secondary: TimingTieBreak,
    pub(crate) area: f64,
}

impl Score {
    pub(crate) fn measure(
        sta: &IncrementalSta<'_>,
        library: &Library,
        objective: usize,
    ) -> Result<Self> {
        let area: f64 = sta
            .cell_indices()
            .iter()
            .map(|&cell| library.cells[cell].area)
            .sum();
        let boundaries = sta.score();
        // Compare near-critical sums on the same class throughout a round.
        // Its maximum can fall below another boundary's delay during a trial.
        let delay = boundaries.values()[sta.objective()];
        let secondary = sta.secondary(objective);
        ensure!(
            area.is_finite()
                && area >= 0.0
                && secondary.near_sum.is_finite()
                && boundaries.values().iter().all(|x| x.is_finite()),
            "nonfinite assembled timing or invalid area"
        );
        Ok(Self {
            boundaries,
            delay,
            secondary,
            area,
        })
    }

    /// Rank candidates using delay with an area penalty. If worst delay is
    /// unchanged, also reward reducing the number of tied critical endpoints or
    /// improving near-critical paths enough to justify the added area.
    pub(crate) fn rank(self, old: Self, weight: f64) -> Option<(u8, f64)> {
        if !self.boundaries.within(old.boundaries) {
            return None;
        }
        let area_cost = if self.area == 0.0 && old.area == 0.0 {
            1.0
        } else if old.area == 0.0 {
            return None;
        } else {
            (self.area / old.area).powf(weight)
        };
        if self.delay > old.delay + TIMING_COMPARISON_TOLERANCE {
            return None;
        }
        if old.delay > 0.0 {
            let cost = self.delay / old.delay * area_cost;
            if cost < 1.0 - 1e-12 {
                return Some((0, cost));
            }
        } else if self.delay < old.delay - TIMING_COMPARISON_TOLERANCE && self.area <= old.area {
            return Some((0, self.delay));
        }
        if self.delay <= old.delay && self.area < old.area {
            return Some((0, self.area / old.area));
        }
        if (self.delay - old.delay).abs() <= TIMING_COMPARISON_TOLERANCE {
            if old.secondary.near_sum <= 0.0 {
                if self.delay <= old.delay && self.area <= old.area {
                    return self.secondary.improvement_rank(old.secondary);
                }
                return None;
            }
            let cost = self.secondary.relative_to(old.secondary) * area_cost;
            if cost < 1.0 - 1e-12 {
                return Some((1, cost));
            }
        }
        None
    }
}

/// The frontier consists of original IO bits and register data endpoints.
/// Register cells and ports retain their indices in an append-only topology
/// edit.
pub(crate) fn preserves_endpoints(old: &IncrementalSta<'_>, new: &IncrementalSta<'_>) -> bool {
    let key = |(endpoint, arrivals): (TimingEndpoint, [Option<f64>; 2])| {
        let id = match endpoint {
            TimingEndpoint::Output(i) => (false, i),
            TimingEndpoint::Capture(i) => (true, i),
        };
        (id, arrivals.map(|a| a.is_some()))
    };
    old.endpoint_values()
        .map(key)
        .eq(new.endpoint_values().map(key))
}

/// Existing pins may shed violations; a newly inserted pin must satisfy each
/// characterized limit. Comparisons use the same pin order on unchanged
/// masters. A positive output budget collects eligible retry sites without
/// accepting the trial. The retry itself has a zero budget.
fn electrical_progress(
    old: &IncrementalSta<'_>,
    new: &IncrementalSta<'_>,
    guard: ElectricalGuard,
    output_budget: usize,
) -> Option<(bool, Vec<(usize, String)>)> {
    let mut improved = false;
    let mut worse_outputs = Vec::new();
    for instance in 0..new.instance_count() {
        let before = (instance < old.instance_count())
            .then(|| old.cached_electrical_violations(instance))
            .unwrap_or(&[]);
        for (pin, &next) in new
            .cached_electrical_violations(instance)
            .iter()
            .enumerate()
        {
            let previous = before.get(pin).map_or_else(Default::default, |&v| {
                guard.excess(old, old.cell_indices()[instance], pin, v)
            });
            let next = guard.excess(new, new.cell_indices()[instance], pin, next);
            if guard.worsens(next, previous) {
                if output_budget == 0 || instance >= old.instance_count() {
                    return None;
                }
                // One net's transition can exceed limits at its driver and
                // at many receivers. Nominate that driver once, using trial
                // connectivity. The expanded trial must pass every pin check
                // with no retries.
                let (driver, name) = new.electrical_driver(instance, pin)?;
                if driver >= old.instance_count() {
                    return None;
                }
                if !worse_outputs.iter().any(|(d, n)| *d == driver && n == name) {
                    if worse_outputs.len() == output_budget {
                        return None;
                    }
                    worse_outputs.push((driver, name.to_owned()));
                }
            }
            improved |= guard.worsens(previous, next);
        }
    }
    Some((improved, worse_outputs))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::io::read_gv_from_str;
    use crate::netlist::resize::measure_boundary_delays;
    use std::fmt::Write;

    #[test]
    fn retry_charges_one_driver_for_multiple_receiver_slew_violations() -> Result<()> {
        let mut lib = super::super::tests::library()?;
        let ff = lib.cells.iter_mut().find(|c| c.name == "DFF1").unwrap();
        for pin in &mut ff.pins {
            pin.max_transition = Some(2.0);
        }

        // Balancing the loads from 4+2 to 3+3 improves the worst path.
        // Launch b's Q and both original receivers' D pins cross the slew
        // limit.
        let analyze = |sinks: &str| {
            let mut gv = String::from(
                "module t(input clk, input d, output [5:0] y); wire a,b;
                 DFF1 launch_a(.CLK(clk),.D(d),.Q(a));
                 DFF1 launch_b(.CLK(clk),.D(d),.Q(b));",
            );
            for (i, source) in sinks.chars().enumerate() {
                writeln!(gv, "DFF1 c{i}(.CLK(clk),.D({source}),.Q(y[{i}]));")?;
            }
            gv.push_str("endmodule");
            let net = read_gv_from_str(&gv)?;
            IncrementalSta::new(
                &net.modules[0],
                &net.nets,
                &net.interner,
                &lib,
                StaOptions::default(),
            )
        };
        let old = analyze("aaaabb")?;
        let trial = analyze("aaabbb")?;
        assert!(trial.score().register_to_register < old.score().register_to_register);
        let progress =
            |budget| electrical_progress(&old, &trial, ElectricalGuard::Relative, budget);
        let (_, outputs) = progress(1).expect("expected a retry");
        assert_eq!(outputs, [(1, "Q".into())]);
        assert!(progress(0).is_none());
        Ok(())
    }

    // A loaded launch Q and a separate short path exercise both parts of a
    // round. Compare topology with sizing, then downsize the short path on the
    // winner.
    #[test]
    fn registered_search_compares_trees_and_sizes_on_the_same_stage() -> Result<()> {
        let mut lib = super::super::tests::library()?;
        // Make the launch genuinely overload in the library while a small
        // buffer tree can drive the captures. Keep multiple strengths on the
        // data gates.
        for cell in &mut lib.cells {
            for pin in &mut cell.pins {
                if pin.direction == crate::liberty_proto::PinDirection::Output as i32 {
                    pin.max_capacitance = Some(if cell.name.starts_with("DFF") {
                        0.8
                    } else {
                        3.2
                    });
                } else if cell.name == "BUF4" {
                    pin.capacitance = Some(0.08);
                }
            }
        }
        let mut gv = "module t(input clk, input d, input other, output short_q, output [31:0] y); wire spare_d,short_d,q; DFF1 spare_ff(.CLK(clk),.D(other),.Q(spare_d)); INV4 cheap_on_short_path(.A(spare_d),.Y(short_d)); DFF1 end_short(.CLK(clk),.D(short_d),.Q(short_q)); DFF1 launch(.CLK(clk),.D(d),.Q(q));".to_string();
        for i in 0..32 {
            writeln!(
                gv,
                "wire n{i}; INV1 inv{i}(.A(q),.Y(n{i})); DFF1 cap{i}(.CLK(clk),.D(n{i}),.Q(y[{i}]));"
            )?;
        }
        gv.push_str("endmodule");
        let opts = RegisteredOptimizationOptions {
            timing: StaOptions::default(),
            fanout: Some(FanoutOptimizationOptions::default()),
            resize: Some(ResizeOptions {
                max_area_growth: 0.15,
                resize_registers: false,
                ..Default::default()
            }),
        };
        let mut net = read_gv_from_str(&gv)?;
        let r = optimize_registered_netlist(&mut net, &lib, &opts)?;
        assert!(
            r.rounds
                .iter()
                .any(|r| r.topology_trials > 0 && r.timing_trials > 0)
        );
        assert!(
            r.rounds
                .iter()
                .any(|r| r.accepted == "topology" && r.changed_cell_masters > 0)
        );
        let short = net.modules[0]
            .instances
            .iter()
            .find(|inst| net.interner.resolve(inst.instance_name) == Some("cheap_on_short_path"))
            .unwrap();
        assert_eq!(net.interner.resolve(short.type_name), Some("INV1"));
        assert!(r.objective_delay < r.initial_delay.register_to_register);
        assert_eq!(
            r.final_delay,
            measure_boundary_delays(&net.modules[0], &net.nets, &net.interner, &lib, opts.timing)?
        );
        assert_eq!(
            net.modules[0]
                .instances
                .iter()
                .filter(|i| net.interner.resolve(i.type_name) == Some("DFF1"))
                .count(),
            35
        );
        Ok(())
    }

    // Zero sizing budgets with topology disabled must leave both the gate text
    // and the measured score intact.
    #[test]
    fn exhausted_budgets_keep_the_emitted_stage_and_its_score() -> Result<()> {
        use crate::netlist::emit::emit_module_as_netlist_text;
        let lib = super::super::tests::library()?;
        let mut net = read_gv_from_str(
            "module t(input clk, input d, output y); wire q,n; DFF1 launch(.CLK(clk),.D(d),.Q(q)); INV4 inverter(.A(q),.Y(n)); DFF1 capture(.CLK(clk),.D(n),.Q(y)); endmodule",
        )?;
        let before = emit_module_as_netlist_text(&net.modules[0], &net.nets, &net.interner)?;
        let opts = RegisteredOptimizationOptions {
            timing: StaOptions::default(),
            fanout: None,
            resize: Some(ResizeOptions {
                timing_evaluation_budget: Some(0),
                recovery_evaluation_budget: Some(0),
                ..Default::default()
            }),
        };
        let r = optimize_registered_netlist(&mut net, &lib, &opts)?;
        assert_eq!((r.timing_evaluations, r.recovery_evaluations), (0, 0));
        assert_eq!(r.initial_area, r.final_area);
        assert_eq!(r.initial_delay, r.final_delay);
        assert_eq!(
            before,
            emit_module_as_netlist_text(&net.modules[0], &net.nets, &net.interner)?
        );
        Ok(())
    }

    #[test]
    fn sizing_follows_the_longest_reachable_boundary_between_rounds() -> Result<()> {
        let mut lib = super::super::tests::library()?;
        for cell in lib.cells.iter_mut().filter(|c| c.name.starts_with("BUF")) {
            cell.area = 1.0;
            cell.dont_use = Some(cell.name == "BUF4");
            for pin in &mut cell.pins {
                pin.capacitance = Some(0.0);
            }
        }
        let run = |rounds| {
            let mut net = read_gv_from_str(
                r#"module m(input clk, input a, output x, output y, output z);
                   wire n, q;
                   BUF1 first(.A(a), .Y(n));
                   BUF1 second(.A(n), .Y(x));
                   DFF1 ff(.CLK(clk), .D(a), .Q(q));
                   BUF1 slow(.A(q), .Y(y));
                   BUF2 fast(.A(q), .Y(z));
                   endmodule"#,
            )?;
            optimize_registered_netlist(
                &mut net,
                &lib,
                &RegisteredOptimizationOptions {
                    timing: StaOptions {
                        primary_input_transition: 0.0,
                        ..Default::default()
                    },
                    fanout: None,
                    resize: Some(ResizeOptions {
                        max_upsize_rounds: rounds,
                        max_downsize_rounds: 0,
                        resize_registers: false,
                        ..Default::default()
                    }),
                },
            )
        };
        // Initially x and y tie. Sizing y makes it tie with z below x.
        // The next sweep must follow x to reduce the stage delay.
        let first = run(1)?;
        assert_eq!(
            first.initial_delay.register_to_output,
            first.initial_delay.input_to_output
        );
        assert!(first.final_delay.input_to_output > first.final_delay.register_to_output);
        assert_eq!(first.objective_delay, first.final_delay.input_to_output);
        let second = run(2)?;
        assert!(second.objective_delay < first.objective_delay);
        assert_eq!(
            second.objective_delay,
            second
                .final_delay
                .input_to_output
                .max(second.final_delay.register_to_output)
        );
        Ok(())
    }
}
