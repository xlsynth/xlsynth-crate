// SPDX-License-Identifier: Apache-2.0

//! Shared NF policy traversal and boundary-aware complete-cover comparison.

use super::cover::{CoverPlan, SolutionChoice, SourceKind};
use super::electrical::{self, ElectricalBoundary, NfTimingCalibration};
use super::{
    MappedNetlist, PreparedTechMapLibrary, TechMapOptions, TechMapStats, TechMapTimingConstraints,
    cuts, finish_prepared_choice_cover, liberty_index, nf,
};
use crate::aig::ChoiceAig;
use anyhow::{Result, anyhow, bail};
use serde::Serialize;

/// Controls complete NF policy comparisons, independently of AIG choices.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NfCoverSearch {
    /// Map exactly one fastest-child cover for both registered and
    /// combinational designs; never enter implicit large-cover portfolios.
    #[default]
    Single,
    /// Preserve the established single registered cover and large combinational
    /// search.
    Automatic,
    /// Compare area-child and fastest-child covers after all requested
    /// processing.
    FastAndArea,
    /// Also try structural area-priority cuts, retaining the original cover
    /// unless a complete alternative preserves every area/timing bound.
    GuardedArea,
    /// Calibrate a second cover at the incumbent's internal slew/load medians.
    CalibratedTiming,
    /// Also propagate candidate slews and refresh live-cover loads between
    /// rounds.
    LoadSlew,
}

/// Exact final physical-netlist metrics for one successfully evaluated policy.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NfCoverMetrics {
    pub area: f64,
    pub cells: usize,
    pub worst_delay: f64,
    pub input_to_register: Option<f64>,
    pub register_to_register: Option<f64>,
    pub register_to_output: Option<f64>,
    pub clock_period: Option<f64>,
    pub worst_register_slack: Option<f64>,
}

impl NfCoverMetrics {
    /// Copies the same final registered timing classes reported by gv-stats.
    fn from_stats(stats: &TechMapStats) -> Self {
        Self {
            area: stats.selected_area,
            cells: stats.selected_instance_count,
            worst_delay: stats.worst_estimated_output_arrival,
            input_to_register: stats.worst_input_to_register_arrival,
            register_to_register: stats.worst_register_to_register_arrival,
            register_to_output: stats.worst_register_to_output_arrival,
            clock_period: stats.clock_period,
            worst_register_slack: stats.worst_register_slack,
        }
    }

    /// Rejects non-finite metrics instead of silently ordering invalid timing.
    fn validate(&self) -> Result<()> {
        if !self.area.is_finite()
            || self.area < 0.0
            || !self.worst_delay.is_finite()
            || self.worst_delay < 0.0
            || [
                self.input_to_register,
                self.register_to_register,
                self.register_to_output,
                self.clock_period,
                self.worst_register_slack,
            ]
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            bail!("NF cover has non-finite or negative area/delay metrics");
        }
        Ok(())
    }
}

/// One attempted policy, including failures that did not discard another cover.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NfCoverCandidateStats {
    pub policy: String,
    pub metrics: Option<NfCoverMetrics>,
    pub error: Option<String>,
}

/// Deterministic audit trail for a complete guarded policy search.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NfCoverSearchReport {
    /// Metrics are measured after register restoration and requested physical
    /// passes.
    pub stage: String,
    pub selected_policy: String,
    pub selection_reason: String,
    pub candidates: Vec<NfCoverCandidateStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_calibration: Option<NfTimingCalibration>,
    /// A cheap screen may skip the alternative's complete physical
    /// optimization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alternative_skipped: Option<String>,
    /// Maximum permitted delay regression in percent; zero is strict selection.
    pub max_delay_regression_percent: f64,
    /// Optional cooperative wall-time budget; absent for deterministic runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alternative_timeout_seconds: Option<f64>,
}

/// One internal covering policy and its Liberty root tie-breaking variant.
#[derive(Clone, Copy)]
pub(super) struct NfCoverSpec {
    pub policy: nf::NfCoverPolicy,
    pub stable_roots: bool,
}

/// Visits policies with shared choice analysis and lazily prepared root
/// variants.
pub(super) fn visit_nf_covers(
    choice_aig: &ChoiceAig,
    prepared: &PreparedTechMapLibrary<'_>,
    analysis: &cuts::ChoiceAnalysis,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
    policies: &[NfCoverSpec],
    mut visit: impl FnMut(NfCoverSpec, &PreparedTechMapLibrary<'_>, Result<nf::NfCover>) -> Result<()>,
) -> Result<()> {
    let alternate = policies
        .iter()
        .any(|spec| spec.stable_roots)
        .then(|| {
            liberty_index::LibertyCellIndex::build_nf_stable_roots(
                prepared.library,
                prepared.max_cut_size,
            )
            .map(|cell_index| PreparedTechMapLibrary {
                library: prepared.library,
                cell_index,
                max_cut_size: prepared.max_cut_size,
            })
        })
        .transpose()?;
    for &spec in policies {
        let library = if spec.stable_roots {
            alternate
                .as_ref()
                .expect("requested stable roots were prepared")
        } else {
            prepared
        };
        let cover = nf::build_cover_plan_with_policy(
            choice_aig,
            analysis,
            library.library,
            &library.cell_index,
            options,
            constraints,
            spec.policy,
        );
        visit(spec, library, cover)?;
    }
    Ok(())
}

/// Compares fully optimized NF covers using caller-owned timing boundaries.
pub(super) fn map_nf_cover_search(
    choice_aig: &ChoiceAig,
    prepared: &PreparedTechMapLibrary<'_>,
    analysis: &cuts::ChoiceAnalysis,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
    boundary: Option<&ElectricalBoundary>,
    mut evaluate: impl FnMut(&PreparedTechMapLibrary<'_>, nf::NfCover) -> Result<MappedNetlist>,
) -> Result<MappedNetlist> {
    if matches!(
        options.nf_cover_search,
        NfCoverSearch::CalibratedTiming | NfCoverSearch::LoadSlew
    ) {
        return map_electrical_cover_search(
            choice_aig,
            prepared,
            analysis,
            constraints,
            options,
            boundary.unwrap_or(&ElectricalBoundary::default()),
            evaluate,
        );
    }
    let policies = match options.nf_cover_search {
        NfCoverSearch::Single => vec![],
        NfCoverSearch::Automatic => bail!("explicit NF cover search requires a search policy"),
        NfCoverSearch::FastAndArea => vec![nf::NfCoverPolicy::AreaChildren],
        NfCoverSearch::GuardedArea => vec![
            nf::NfCoverPolicy::AreaChildren,
            nf::NfCoverPolicy::StructuralAreaCuts,
        ],
        NfCoverSearch::CalibratedTiming | NfCoverSearch::LoadSlew => unreachable!("handled above"),
    }
    .into_iter()
    .chain([nf::NfCoverPolicy::Standard])
    .map(|policy| NfCoverSpec {
        policy,
        stable_roots: false,
    })
    .collect::<Vec<_>>();
    let mut results = Vec::new();
    let mut reports = Vec::new();
    visit_nf_covers(
        choice_aig,
        prepared,
        analysis,
        constraints,
        options,
        &policies,
        |spec, prepared, cover| {
            let policy = match spec.policy {
                nf::NfCoverPolicy::Standard => "fastest-children",
                nf::NfCoverPolicy::AreaChildren => "area-children",
                nf::NfCoverPolicy::StructuralAreaCuts => "area-priority-cuts",
                _ => unreachable!("guarded search enumerates only its named policies"),
            };
            let result = (|| {
                let mapped = evaluate(prepared, cover?)?;
                NfCoverMetrics::from_stats(&mapped.stats).validate()?;
                Ok::<_, anyhow::Error>(mapped)
            })();
            reports.push(NfCoverCandidateStats {
                policy: policy.to_string(),
                metrics: result
                    .as_ref()
                    .ok()
                    .map(|mapped| NfCoverMetrics::from_stats(&mapped.stats)),
                error: result.as_ref().err().map(|error| format!("{error:#}")),
            });
            results.push(result.ok());
            Ok(())
        },
    )?;
    let (selected, reason) = select_cover(&reports, options.nf_cover_max_delay_regression)?;
    let mut mapped = results[selected].take().expect("selected cover succeeded");
    mapped.stats.nf_cover_search = Some(NfCoverSearchReport {
        stage: "final-netlist-after-requested-optimization".to_string(),
        selected_policy: reports[selected].policy.clone(),
        selection_reason: reason.to_string(),
        candidates: reports,
        timing_calibration: None,
        alternative_skipped: None,
        max_delay_regression_percent: options.nf_cover_max_delay_regression * 100.0,
        alternative_timeout_seconds: options
            .nf_cover_timeout
            .map(|timeout| timeout.as_secs_f64()),
    });
    Ok(mapped)
}

/// Runs at most two physical optimizations, preserving a known-good incumbent.
#[allow(clippy::too_many_arguments)]
fn map_electrical_cover_search(
    graph: &ChoiceAig,
    prepared: &PreparedTechMapLibrary<'_>,
    analysis: &cuts::ChoiceAnalysis,
    constraints: &TechMapTimingConstraints,
    options: &TechMapOptions,
    boundary: &ElectricalBoundary,
    mut evaluate: impl FnMut(&PreparedTechMapLibrary<'_>, nf::NfCover) -> Result<MappedNetlist>,
) -> Result<MappedNetlist> {
    let incumbent = nf::build_cover_plan(
        graph,
        analysis,
        prepared.library,
        &prepared.cell_index,
        options,
        constraints,
    )?;
    let raw_area = cover_area(&incumbent.plan);
    let incumbent_identity = CoverIdentity::new(&incumbent.plan);
    let calibration = electrical::calibrate_cover(
        &incumbent.plan,
        graph,
        prepared.library,
        options,
        constraints,
        boundary,
    );
    let mut mapped = evaluate(prepared, incumbent)?;
    let baseline = NfCoverMetrics::from_stats(&mapped.stats);
    baseline.validate()?;
    let mut reports = vec![NfCoverCandidateStats {
        policy: "fastest-children".into(),
        metrics: Some(baseline.clone()),
        error: None,
    }];
    let policy = if options.nf_cover_search == NfCoverSearch::LoadSlew {
        "load-slew"
    } else {
        "calibrated-timing"
    };
    let mut selected_policy = "fastest-children";
    let mut skipped = None;
    let timing_calibration = calibration.as_ref().ok().copied();
    let alternative = (|| -> Result<Option<MappedNetlist>> {
        let _budget =
            crate::optimization_budget::ScopedOptimizationBudget::new(options.nf_cover_timeout);
        crate::optimization_budget::check()?;
        let calibration = calibration?;
        let cover = nf::build_electrical_cover(
            graph,
            analysis,
            prepared.library,
            &prepared.cell_index,
            options,
            constraints,
            calibration,
            boundary,
            options.nf_cover_search == NfCoverSearch::LoadSlew,
        )?;
        crate::optimization_budget::check()?;
        if CoverIdentity::new(&cover.plan) == incumbent_identity {
            skipped = Some("alternative has identical cells, pin connections, and outputs".into());
            return Ok(None);
        }
        if cover_area(&cover.plan) > raw_area + 1e-9 {
            skipped = Some("alternative pre-optimization area exceeds incumbent cover area".into());
            return Ok(None);
        }
        let mut alternative = evaluate(prepared, cover)?;
        crate::optimization_budget::check()?;
        alternative.stats.representative_input_transition = Some(calibration.input_transition);
        NfCoverMetrics::from_stats(&alternative.stats).validate()?;
        Ok(Some(alternative))
    })();
    reports.push(NfCoverCandidateStats {
        policy: policy.into(),
        metrics: alternative
            .as_ref()
            .ok()
            .and_then(|mapped| mapped.as_ref())
            .map(|mapped| NfCoverMetrics::from_stats(&mapped.stats)),
        error: alternative.as_ref().err().map(|error| format!("{error:#}")),
    });
    if let Ok(Some(alternative)) = alternative {
        if improves_with_delay_budget(
            &NfCoverMetrics::from_stats(&alternative.stats),
            &baseline,
            options.nf_cover_max_delay_regression,
        ) {
            mapped = alternative;
            selected_policy = policy;
        }
    }
    mapped.stats.nf_cover_search = Some(NfCoverSearchReport {
        stage: "final-netlist-after-requested-optimization".into(),
        selected_policy: selected_policy.into(),
        selection_reason: if selected_policy == policy {
            if options.nf_cover_max_delay_regression == 0.0 {
                "alternative-cover-pareto-improves"
            } else {
                "alternative-cover-within-delay-budget"
            }
        } else {
            "retain-fastest-child-baseline"
        }
        .into(),
        candidates: reports,
        timing_calibration,
        alternative_skipped: skipped,
        max_delay_regression_percent: options.nf_cover_max_delay_regression * 100.0,
        alternative_timeout_seconds: options
            .nf_cover_timeout
            .map(|timeout| timeout.as_secs_f64()),
    });
    Ok(mapped)
}

/// Exact reconstruction identity, ignoring heuristic scores and timing models.
#[derive(PartialEq, Eq)]
struct CoverIdentity {
    cells: Vec<CoverIdentityNode>,
    outputs: Vec<usize>,
}

#[derive(PartialEq, Eq)]
enum CoverIdentityNode {
    Input(usize),
    Literal(bool),
    Cell {
        cell: usize,
        output: usize,
        pins: Vec<String>,
        inputs: Vec<usize>,
    },
}

impl CoverIdentity {
    fn new(plan: &CoverPlan) -> Self {
        Self {
            cells: plan
                .solutions
                .iter()
                .map(|solution| match &solution.choice {
                    SolutionChoice::Source(SourceKind::Input(node)) => {
                        CoverIdentityNode::Input(node.id)
                    }
                    SolutionChoice::Source(SourceKind::Literal(value)) => {
                        CoverIdentityNode::Literal(*value)
                    }
                    SolutionChoice::Cell { binding, inputs } => CoverIdentityNode::Cell {
                        cell: binding.cell_index,
                        output: binding.output_pin_index,
                        pins: binding.input_pin_names.clone(),
                        inputs: inputs.iter().map(|input| input.0).collect(),
                    },
                })
                .collect(),
            outputs: plan
                .output_solutions
                .iter()
                .map(|output| output.0)
                .collect(),
        }
    }
}

/// Sums only selected cells, without charging source aliases or dead choices.
fn cover_area(plan: &CoverPlan) -> f64 {
    plan.solutions
        .iter()
        .map(|solution| match &solution.choice {
            SolutionChoice::Cell { binding, .. } => binding.area,
            SolutionChoice::Source(_) => 0.0,
        })
        .sum()
}

/// Emits one NF cover without changing its policy or boundary context.
pub(super) fn finish_nf_cover(
    choice_aig: &ChoiceAig,
    prepared: &PreparedTechMapLibrary<'_>,
    analysis: &cuts::ChoiceAnalysis,
    cover: nf::NfCover,
    options: &TechMapOptions,
) -> Result<MappedNetlist> {
    finish_prepared_choice_cover(
        choice_aig,
        prepared,
        analysis,
        cover.plan,
        cover.enumerated_cut_count,
        cover.representative_output_load,
        options,
    )
}

/// Chooses the smallest alternative that preserves all baseline bounds.
fn select_cover(
    candidates: &[NfCoverCandidateStats],
    delay_budget: f64,
) -> Result<(usize, &'static str)> {
    let baseline_index = candidates
        .len()
        .checked_sub(1)
        .ok_or_else(|| anyhow!("NF cover search has no candidates"))?;
    let baseline = candidates[baseline_index].metrics.as_ref();
    let best = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| candidate.metrics.as_ref().map(|metrics| (index, metrics)))
        .filter(|(index, candidate)| {
            *index != baseline_index
                && baseline.is_none_or(|baseline| {
                    improves_with_delay_budget(candidate, baseline, delay_budget)
                })
        })
        .min_by(|(left_index, left), (right_index, right)| {
            left.area
                .total_cmp(&right.area)
                .then_with(|| left.worst_delay.total_cmp(&right.worst_delay))
                .then_with(|| left_index.cmp(right_index))
        });
    match (best, baseline) {
        (Some((index, _)), None) => Ok((index, "fastest-child-policy-failed")),
        (Some((index, _)), Some(_)) => Ok((
            index,
            if delay_budget == 0.0 {
                "alternative-cover-pareto-improves"
            } else {
                "alternative-cover-within-delay-budget"
            },
        )),
        (None, Some(_)) => Ok((baseline_index, "retain-fastest-child-baseline")),
        (None, None) => Err(anyhow!(
            "all NF cover policies failed: {}",
            candidates
                .iter()
                .map(|candidate| format!(
                    "{}: {}",
                    candidate.policy,
                    candidate.error.as_deref().unwrap_or("missing metrics")
                ))
                .collect::<Vec<_>>()
                .join("; ")
        )),
    }
}

/// Allows bounded timing tradeoffs only for smaller covers with legal
/// deadlines.
fn improves_with_delay_budget(
    candidate: &NfCoverMetrics,
    baseline: &NfCoverMetrics,
    budget: f64,
) -> bool {
    const EPSILON: f64 = 1e-9;
    if dominates(candidate, baseline) {
        return true;
    }
    if budget <= 0.0
        || !budget.is_finite()
        || candidate.area + EPSILON >= baseline.area
        || candidate.clock_period != baseline.clock_period
    {
        return false;
    }
    let within = |candidate: f64, baseline: f64| candidate <= baseline * (1.0 + budget) + EPSILON;
    if !within(candidate.worst_delay, baseline.worst_delay) {
        return false;
    }
    for (candidate, baseline) in [
        (candidate.input_to_register, baseline.input_to_register),
        (
            candidate.register_to_register,
            baseline.register_to_register,
        ),
        (candidate.register_to_output, baseline.register_to_output),
    ] {
        match (candidate, baseline) {
            (Some(candidate), Some(baseline)) if within(candidate, baseline) => {
                // Both retain their timing-class bound.
            }
            (None, None) => {
                // Neither netlist has this timing class.
            }
            _ => return false,
        }
    }
    match (
        candidate.worst_register_slack,
        baseline.worst_register_slack,
    ) {
        (Some(slack), Some(_)) => slack >= -EPSILON,
        (None, None) => true,
        _ => false,
    }
}

/// Requires no area or timing-class regression, with deterministic baseline
/// ties.
fn dominates(candidate: &NfCoverMetrics, baseline: &NfCoverMetrics) -> bool {
    const EPSILON: f64 = 1e-9;
    let mut improved = candidate.area + EPSILON < baseline.area
        || candidate.worst_delay + EPSILON < baseline.worst_delay;
    if candidate.area > baseline.area + EPSILON
        || candidate.worst_delay > baseline.worst_delay + EPSILON
        || candidate.clock_period != baseline.clock_period
    {
        return false;
    }
    for (candidate, baseline) in [
        (candidate.input_to_register, baseline.input_to_register),
        (
            candidate.register_to_register,
            baseline.register_to_register,
        ),
        (candidate.register_to_output, baseline.register_to_output),
    ] {
        match (candidate, baseline) {
            (Some(candidate), Some(baseline)) => {
                if candidate > baseline + EPSILON {
                    return false;
                }
                improved |= candidate + EPSILON < baseline;
            }
            (None, None) => { /* Neither netlist has this timing class. */ }
            _ => return false,
        }
    }
    match (
        candidate.worst_register_slack,
        baseline.worst_register_slack,
    ) {
        (Some(candidate), Some(baseline)) => {
            if candidate + EPSILON < baseline {
                return false;
            }
        }
        (None, None) => { /* Neither netlist has a constrained register path. */ }
        _ => return false,
    }
    improved
}

#[cfg(test)]
mod tests {
    use super::super::electrical::tests::{electrical_graph, electrical_library};
    use crate::aig::{GateBuilder, GateBuilderOptions};

    #[test]
    fn single_cover_default_evaluates_only_the_fastest_child_cover() {
        let graph = electrical_graph();
        let library = electrical_library();
        let prepared = PreparedTechMapLibrary::new(&library, 6).unwrap();
        let analysis = cuts::analyze_choices(&graph).unwrap();
        let options = TechMapOptions::default();
        assert_eq!(options.nf_cover_search, NfCoverSearch::Single);
        let mut calls = 0;
        let mapped = map_nf_cover_search(
            &graph,
            &prepared,
            &analysis,
            &TechMapTimingConstraints::default(),
            &options,
            None,
            |prepared, cover| {
                calls += 1;
                finish_nf_cover(&graph, prepared, &analysis, cover, &options)
            },
        )
        .unwrap();
        assert_eq!(calls, 1);
        let report = mapped.stats.nf_cover_search.unwrap();
        assert_eq!(report.candidates.len(), 1);
        assert_eq!(report.selected_policy, "fastest-children");
        assert!(report.candidates[0].metrics.is_some());
    }

    #[test]
    fn expired_alternative_budget_retains_a_complete_incumbent() {
        let graph = electrical_graph();
        let library = electrical_library();
        let prepared = PreparedTechMapLibrary::new(&library, 6).unwrap();
        let analysis = cuts::analyze_choices(&graph).unwrap();
        let options = TechMapOptions {
            nf_cover_search: NfCoverSearch::LoadSlew,
            nf_cover_timeout: Some(std::time::Duration::ZERO),
            ..Default::default()
        };
        let mut calls = 0;
        let mapped = map_nf_cover_search(
            &graph,
            &prepared,
            &analysis,
            &TechMapTimingConstraints::default(),
            &options,
            None,
            |prepared, cover| {
                calls += 1;
                assert!(
                    crate::optimization_budget::check().is_ok(),
                    "incumbent is unbudgeted"
                );
                finish_nf_cover(&graph, prepared, &analysis, cover, &options)
            },
        )
        .unwrap();
        assert_eq!(calls, 1);
        let report = mapped.stats.nf_cover_search.unwrap();
        assert_eq!(report.selected_policy, "fastest-children");
        assert!(report.candidates[0].metrics.is_some());
        assert_eq!(
            report.candidates[1].error.as_deref(),
            Some("alternative cover exceeded its optimization time budget")
        );
        assert_eq!(report.alternative_timeout_seconds, Some(0.0));
        assert!(crate::optimization_budget::check().is_ok());
    }

    #[test]
    fn identity_screen_compares_wiring_and_outputs_but_not_timing_estimates() {
        let graph = electrical_graph();
        let library = electrical_library();
        let prepared = PreparedTechMapLibrary::new(&library, 6).unwrap();
        let analysis = cuts::analyze_choices(&graph).unwrap();
        let cover = nf::build_cover_plan(
            &graph,
            &analysis,
            &library,
            &prepared.cell_index,
            &TechMapOptions::default(),
            &TechMapTimingConstraints::default(),
        )
        .unwrap();
        let mut changed = cover.plan.clone();
        changed.output_arrivals.fill(1e9);
        assert!(CoverIdentity::new(&cover.plan) == CoverIdentity::new(&changed));
        changed.output_solutions.clear();
        assert!(CoverIdentity::new(&cover.plan) != CoverIdentity::new(&changed));
    }

    #[test]
    fn two_percent_budget_requires_area_savings_and_protects_every_timing_class() {
        let baseline = metrics();
        let mut candidate = baseline.clone();
        candidate.area = 83.2;
        candidate.worst_delay = 10.11;
        candidate.register_to_register = Some(6.066);
        candidate.worst_register_slack = Some(1.89);
        assert!(!improves_with_delay_budget(&candidate, &baseline, 0.0));
        assert!(improves_with_delay_budget(&candidate, &baseline, 0.02));
        candidate.area = baseline.area;
        assert!(!improves_with_delay_budget(&candidate, &baseline, 0.02));
        candidate.area = 83.2;
        candidate.register_to_register = Some(6.121);
        assert!(!improves_with_delay_budget(&candidate, &baseline, 0.02));
        candidate.register_to_register = Some(6.12);
        assert!(improves_with_delay_budget(&candidate, &baseline, 0.02));
        candidate.register_to_output = Some(4.081);
        assert!(!improves_with_delay_budget(&candidate, &baseline, 0.02));
        candidate.register_to_output = baseline.register_to_output;
        candidate.worst_register_slack = Some(-0.001);
        assert!(!improves_with_delay_budget(&candidate, &baseline, 0.02));
        candidate.worst_register_slack = Some(1.89);
        candidate.input_to_register = None;
        assert!(!improves_with_delay_budget(&candidate, &baseline, 0.02));
    }

    #[test]
    fn delay_allowance_rejects_invalid_or_silently_unused_options() {
        for budget in [-0.1, f64::NAN, f64::INFINITY, 1.01] {
            assert!(
                super::super::assert_supported_timing_model(&TechMapOptions {
                    nf_cover_search: NfCoverSearch::LoadSlew,
                    nf_cover_max_delay_regression: budget,
                    ..Default::default()
                })
                .is_err()
            );
        }
        assert!(
            super::super::assert_supported_timing_model(&TechMapOptions {
                nf_cover_max_delay_regression: 0.02,
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn empty_internal_calibration_retains_the_incumbent_without_an_extra_evaluation() {
        let mut builder = GateBuilder::new("identity".into(), GateBuilderOptions::no_opt());
        let input = builder.add_input("a".into(), 1);
        builder.add_output("o".into(), input);
        let graph = ChoiceAig::without_choices(builder.build());
        let library = electrical_library();
        let prepared = PreparedTechMapLibrary::new(&library, 6).unwrap();
        let analysis = cuts::analyze_choices(&graph).unwrap();
        let options = TechMapOptions {
            nf_cover_search: NfCoverSearch::LoadSlew,
            ..Default::default()
        };
        let mut calls = 0;
        let result = map_nf_cover_search(
            &graph,
            &prepared,
            &analysis,
            &TechMapTimingConstraints::default(),
            &options,
            None,
            |prepared, cover| {
                calls += 1;
                finish_nf_cover(&graph, prepared, &analysis, cover, &options)
            },
        )
        .unwrap();
        assert_eq!(calls, 1);
        let report = result.stats.nf_cover_search.unwrap();
        assert_eq!(report.selected_policy, "fastest-children");
        assert_eq!(report.timing_calibration, None);
        assert_eq!(
            report.candidates[1].error.as_deref(),
            Some("selected cover has no loaded cells to calibrate")
        );
    }
    use super::*;

    #[test]
    fn calibrated_search_rejects_worse_final_metrics_and_preserves_failed_alternatives() {
        let graph = electrical_graph();
        let library = electrical_library();
        let prepared = PreparedTechMapLibrary::new(&library, 6).unwrap();
        let analysis = cuts::analyze_choices(&graph).unwrap();
        for mode in [NfCoverSearch::CalibratedTiming, NfCoverSearch::LoadSlew] {
            for failure in [false, true] {
                let options = TechMapOptions {
                    nf_cover_search: mode,
                    ..Default::default()
                };
                let mut calls = 0;
                let mapped = map_nf_cover_search(
                    &graph,
                    &prepared,
                    &analysis,
                    &TechMapTimingConstraints::default(),
                    &options,
                    None,
                    |prepared, cover| {
                        calls += 1;
                        if calls == 2 && failure {
                            bail!("candidate physical optimization failed");
                        }
                        let mut mapped =
                            finish_nf_cover(&graph, prepared, &analysis, cover, &options)?;
                        mapped.stats.selected_area = if calls == 1 { 100.0 } else { 90.0 };
                        mapped.stats.worst_estimated_output_arrival = 10.0;
                        mapped.stats.worst_register_to_register_arrival =
                            Some(if calls == 1 { 6.0 } else { 7.0 });
                        Ok(mapped)
                    },
                )
                .unwrap();
                assert_eq!(mapped.stats.selected_area, 100.0);
                let report = mapped.stats.nf_cover_search.unwrap();
                assert_eq!(report.selected_policy, "fastest-children");
                assert!(report.timing_calibration.is_some());
                if report.alternative_skipped.is_some() {
                    assert_eq!(calls, 1, "identical covers skip the physical callback");
                    assert_eq!(
                        report.alternative_skipped.as_deref(),
                        Some("alternative has identical cells, pin connections, and outputs")
                    );
                    assert_eq!(report.candidates[1].metrics, None);
                    assert_eq!(report.candidates[1].error, None);
                } else {
                    assert_eq!(
                        calls, 2,
                        "one baseline and at most one complete alternative"
                    );
                    assert_eq!(report.candidates[1].error.is_some(), failure);
                }
            }
        }
    }

    fn metrics() -> NfCoverMetrics {
        NfCoverMetrics {
            area: 100.0,
            cells: 50,
            worst_delay: 10.0,
            input_to_register: Some(10.0),
            register_to_register: Some(6.0),
            register_to_output: Some(4.0),
            clock_period: Some(12.0),
            worst_register_slack: Some(2.0),
        }
    }

    #[test]
    fn comparison_protects_each_registered_timing_class() {
        let baseline = metrics();
        let mut candidate = baseline.clone();
        candidate.area = 90.0;
        assert!(dominates(&candidate, &baseline));
        candidate.register_to_register = Some(7.0);
        assert!(
            !dominates(&candidate, &baseline),
            "a slower register path cannot hide behind a longer input path"
        );
        candidate.register_to_register = None;
        assert!(
            !dominates(&candidate, &baseline),
            "a missing timing class is not a zero delay"
        );
        assert!(!dominates(&baseline, &baseline), "ties keep the baseline");
        candidate = baseline.clone();
        candidate.area = 101.0;
        candidate.worst_delay = 9.0;
        assert!(
            !dominates(&candidate, &baseline),
            "faster but larger is not Pareto-improving"
        );
    }

    #[test]
    fn failed_policies_do_not_discard_a_successful_cover() {
        let mut candidates = vec![
            NfCoverCandidateStats {
                policy: "area-children".to_string(),
                metrics: None,
                error: Some("infeasible capture timing".to_string()),
            },
            NfCoverCandidateStats {
                policy: "fastest-children".to_string(),
                metrics: Some(metrics()),
                error: None,
            },
        ];
        assert_eq!(
            select_cover(&candidates, 0.0).unwrap(),
            (1, "retain-fastest-child-baseline")
        );
        candidates[0].metrics = candidates[1].metrics.take();
        candidates[0].error = None;
        candidates[1].error = Some("infeasible output requirement".to_string());
        assert_eq!(
            select_cover(&candidates, 0.0).unwrap(),
            (0, "fastest-child-policy-failed")
        );
        candidates[0].metrics = None;
        assert!(select_cover(&candidates, 0.0).is_err());
    }

    #[test]
    fn rejects_nonfinite_candidate_metrics() {
        let mut candidate = metrics();
        candidate.register_to_register = Some(f64::NAN);
        assert!(candidate.validate().is_err());
    }

    #[test]
    fn guarded_search_selects_smallest_baseline_preserving_alternative() {
        let baseline = metrics();
        let mut area_child = baseline.clone();
        area_child.area = 95.0;
        area_child.worst_delay = 9.0;
        let mut area_cuts = baseline.clone();
        area_cuts.area = 80.0;
        let mut candidates = [area_child, area_cuts, baseline]
            .into_iter()
            .enumerate()
            .map(|(index, metrics)| NfCoverCandidateStats {
                policy: format!("policy-{index}"),
                metrics: Some(metrics),
                error: None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            select_cover(&candidates, 0.0).unwrap(),
            (1, "alternative-cover-pareto-improves")
        );
        candidates[1].metrics.as_mut().unwrap().register_to_register = Some(6.5);
        assert_eq!(select_cover(&candidates, 0.0).unwrap().0, 0);
        candidates[0].metrics = None;
        assert_eq!(select_cover(&candidates, 0.0).unwrap().0, 2);
        candidates[2].metrics = None;
        assert_eq!(
            select_cover(&candidates, 0.0).unwrap(),
            (1, "fastest-child-policy-failed")
        );
        assert!(select_cover(&[], 0.0).is_err());
    }

    #[test]
    fn guarded_search_preserves_clock_constraints_and_timing_class_presence() {
        let baseline = metrics();
        let mut candidate = baseline.clone();
        candidate.area = 90.0;
        candidate.clock_period = Some(13.0);
        assert!(!dominates(&candidate, &baseline));
        candidate.clock_period = baseline.clock_period;
        candidate.worst_register_slack = Some(1.0);
        assert!(!dominates(&candidate, &baseline));
        candidate.worst_register_slack = None;
        assert!(!dominates(&candidate, &baseline));
    }
}
