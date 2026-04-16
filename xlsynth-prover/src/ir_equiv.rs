// SPDX-License-Identifier: Apache-2.0

//! Library helpers for IR equivalence flows shared by multiple driver commands.

use crate::ir_utils;
use crate::prover::types::{
    AssertionSemantics, EquivParallelism, EquivReport, EquivResult, ParamDomains, ProverFn,
};
use crate::prover::{SolverChoice, prover_for_choice};
use crate::solver::{AtomicSolverInterrupt, SolverInterruptHandle};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use xlsynth_pir::edit_distance::{
    CandidateRankingOptions, RankedFnCandidate, rank_fn_candidates_by_similarity,
};
use xlsynth_pir::ir;

/// Description of a single IR module participating in equivalence.
#[derive(Clone)]
pub struct IrModule<'a> {
    pub source: &'a str,
    pub path: Option<&'a Path>,
    pub top: Option<&'a str>,
    pub param_domains: Option<&'a ParamDomains>,
    pub uf_map: Cow<'a, HashMap<String, String>>,
    pub fixed_implicit_activation: bool,
}

impl<'a> IrModule<'a> {
    /// Creates a new IR module description with default settings.
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            path: None,
            top: None,
            param_domains: None,
            uf_map: Cow::Owned(HashMap::new()),
            fixed_implicit_activation: false,
        }
    }

    /// Sets the top-level function name (if any) to compare.
    pub fn with_top(mut self, top: Option<&'a str>) -> Self {
        self.top = top;
        self
    }

    /// Supplies optional parameter domains for this module.
    pub fn with_param_domains(mut self, domains: Option<&'a ParamDomains>) -> Self {
        self.param_domains = domains;
        self
    }

    /// Sets the UF mapping to use when proving equivalence.
    pub fn with_uf_map(mut self, uf_map: &'a HashMap<String, String>) -> Self {
        self.uf_map = Cow::Borrowed(uf_map);
        self
    }

    /// Provides an owned UF mapping (used mostly in tests).
    pub fn with_owned_uf_map(mut self, uf_map: HashMap<String, String>) -> Self {
        self.uf_map = Cow::Owned(uf_map);
        self
    }

    /// Records the filesystem path the IR was loaded from (if any).
    pub fn with_path(mut self, path: Option<&'a Path>) -> Self {
        self.path = path;
        self
    }

    /// Specifies whether the implicit activation input should be fixed.
    pub fn with_fixed_implicit_activation(mut self, fixed: bool) -> Self {
        self.fixed_implicit_activation = fixed;
        self
    }
}

/// Request describing an IR equivalence proof.
#[derive(Clone)]
pub struct IrEquivRequest<'a> {
    pub lhs: IrModule<'a>,
    pub rhs: IrModule<'a>,
    pub drop_params: &'a [String],
    pub flatten_aggregates: bool,
    pub parallelism: EquivParallelism,
    pub assertion_semantics: AssertionSemantics,
    pub assert_label_filter: Option<&'a str>,
    pub solver: Option<SolverChoice>,
    pub tool_path: Option<&'a Path>,
}

impl<'a> IrEquivRequest<'a> {
    /// Creates a new request with default comparison settings.
    pub fn new(lhs: IrModule<'a>, rhs: IrModule<'a>) -> Self {
        Self {
            lhs,
            rhs,
            drop_params: &[],
            flatten_aggregates: false,
            parallelism: EquivParallelism::SingleThreaded,
            assertion_semantics: AssertionSemantics::Ignore,
            assert_label_filter: None,
            solver: None,
            tool_path: None,
        }
    }

    pub fn with_drop_params(mut self, drop_params: &'a [String]) -> Self {
        self.drop_params = drop_params;
        self
    }

    pub fn with_flatten_aggregates(mut self, flatten: bool) -> Self {
        self.flatten_aggregates = flatten;
        self
    }

    pub fn with_parallelism(mut self, parallelism: EquivParallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    pub fn with_assertion_semantics(mut self, semantics: AssertionSemantics) -> Self {
        self.assertion_semantics = semantics;
        self
    }

    pub fn with_assert_label_filter(mut self, filter: Option<&'a str>) -> Self {
        self.assert_label_filter = filter;
        self
    }

    pub fn with_solver(mut self, solver: Option<SolverChoice>) -> Self {
        self.solver = solver;
        self
    }

    pub fn with_tool_path(mut self, tool_path: Option<&'a Path>) -> Self {
        self.tool_path = tool_path;
        self
    }
}

/// Member of a candidate equivalence class.
#[derive(Clone)]
pub struct EquivClassMember<'a> {
    pub id: Cow<'a, str>,
    pub module: IrModule<'a>,
}

impl<'a> EquivClassMember<'a> {
    /// Creates a new equivalence-class member description.
    pub fn new<S: Into<Cow<'a, str>>>(id: S, module: IrModule<'a>) -> Self {
        Self {
            id: id.into(),
            module,
        }
    }
}

/// Controls how aggressively the equivalence-class API shortlists class
/// members before formal proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EquivClassShortlistOptions {
    /// When the class has more than `max_parallel_proofs` members, compute edit
    /// distance for at most `prefilter_multiplier * max_parallel_proofs`
    /// structurally closest members.
    pub prefilter_multiplier: usize,
}

impl Default for EquivClassShortlistOptions {
    fn default() -> Self {
        Self {
            prefilter_multiplier: 4,
        }
    }
}

impl EquivClassShortlistOptions {
    fn effective_prefilter_limit(self, total_members: usize, max_parallel_proofs: usize) -> usize {
        if total_members <= max_parallel_proofs {
            total_members
        } else {
            std::cmp::min(
                total_members,
                self.prefilter_multiplier
                    .max(1)
                    .saturating_mul(max_parallel_proofs),
            )
        }
    }
}

/// Request describing a candidate-vs-class membership proof.
#[derive(Clone)]
pub struct EquivClassRequest<'a> {
    pub candidate: IrModule<'a>,
    pub members: Vec<EquivClassMember<'a>>,
    pub drop_params: &'a [String],
    pub flatten_aggregates: bool,
    pub parallelism: EquivParallelism,
    pub assertion_semantics: AssertionSemantics,
    pub assert_label_filter: Option<&'a str>,
    pub solver: Option<SolverChoice>,
    pub tool_path: Option<&'a Path>,
    pub max_parallel_proofs: usize,
    pub shortlist: EquivClassShortlistOptions,
}

impl<'a> EquivClassRequest<'a> {
    /// Creates a new equivalence-class membership request.
    pub fn new(candidate: IrModule<'a>, members: Vec<EquivClassMember<'a>>) -> Self {
        Self {
            candidate,
            members,
            drop_params: &[],
            flatten_aggregates: false,
            parallelism: EquivParallelism::SingleThreaded,
            assertion_semantics: AssertionSemantics::Ignore,
            assert_label_filter: None,
            solver: None,
            tool_path: None,
            max_parallel_proofs: default_max_parallel_proofs(),
            shortlist: EquivClassShortlistOptions::default(),
        }
    }

    pub fn with_drop_params(mut self, drop_params: &'a [String]) -> Self {
        self.drop_params = drop_params;
        self
    }

    pub fn with_flatten_aggregates(mut self, flatten: bool) -> Self {
        self.flatten_aggregates = flatten;
        self
    }

    pub fn with_parallelism(mut self, parallelism: EquivParallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    pub fn with_assertion_semantics(mut self, semantics: AssertionSemantics) -> Self {
        self.assertion_semantics = semantics;
        self
    }

    pub fn with_assert_label_filter(mut self, filter: Option<&'a str>) -> Self {
        self.assert_label_filter = filter;
        self
    }

    pub fn with_solver(mut self, solver: Option<SolverChoice>) -> Self {
        self.solver = solver;
        self
    }

    pub fn with_tool_path(mut self, tool_path: Option<&'a Path>) -> Self {
        self.tool_path = tool_path;
        self
    }

    pub fn with_max_parallel_proofs(mut self, max_parallel_proofs: usize) -> Self {
        self.max_parallel_proofs = max_parallel_proofs;
        self
    }

    pub fn with_shortlist_options(mut self, shortlist: EquivClassShortlistOptions) -> Self {
        self.shortlist = shortlist;
        self
    }
}

/// Ranking metadata for one shortlisted class member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquivClassRankingEntry {
    pub shortlist_rank: usize,
    pub member_index: usize,
    pub member_id: String,
    pub shared_structural_hashes: usize,
    pub candidate_structural_hashes: usize,
    pub member_structural_hashes: usize,
    pub edit_distance: u64,
}

/// Completed proof attempt against one class member.
#[derive(Debug, Clone)]
pub struct EquivClassProofEntry {
    pub member_index: usize,
    pub member_id: String,
    pub ranking: EquivClassRankingEntry,
    pub report: EquivReport,
}

/// Metadata for a successful class-membership match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquivClassMatch {
    pub member_index: usize,
    pub member_id: String,
    pub shortlist_rank: usize,
}

/// Reason the class-membership API did not find an equivalent member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquivClassNoMatchReason {
    NoMembers,
    ExhaustedShortlist,
}

/// Outcome of an equivalence-class membership proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquivClassResult {
    Matched(EquivClassMatch),
    NoMatch { reason: EquivClassNoMatchReason },
    InvariantViolation { message: String },
    Error { message: String },
}

/// Detailed report for a class-membership proof run.
#[derive(Debug, Clone)]
pub struct EquivClassReport {
    pub duration: Duration,
    pub total_members: usize,
    pub max_parallel_proofs: usize,
    pub shortlisted_members: Vec<EquivClassRankingEntry>,
    pub completed_proofs: Vec<EquivClassProofEntry>,
    pub excluded_by_shortlist_count: usize,
    pub unstarted_shortlist_count: usize,
    pub result: EquivClassResult,
}

struct PreparedIrModule {
    pkg: ir::Package,
    func: ir::Fn,
    param_domains: Option<ParamDomains>,
    uf_map: HashMap<String, String>,
    fixed_implicit_activation: bool,
}

impl PreparedIrModule {
    fn from_ir_module(module: &IrModule<'_>, drop_params: &[String]) -> Result<Self, String> {
        let (pkg, func) =
            ir_utils::parse_package_and_drop_params(module.source, module.top, drop_params)?;
        Ok(Self {
            pkg,
            func,
            param_domains: module.param_domains.cloned(),
            uf_map: module.uf_map.clone().into_owned(),
            fixed_implicit_activation: module.fixed_implicit_activation,
        })
    }

    fn as_prover_fn(&self) -> ProverFn<'_> {
        ProverFn::new(&self.func, Some(&self.pkg))
            .with_fixed_implicit_activation(self.fixed_implicit_activation)
            .with_domains(self.param_domains.clone())
            .with_uf_map(self.uf_map.clone())
    }
}

struct PreparedEquivClassMember {
    id: String,
    module: PreparedIrModule,
}

#[derive(Clone, Copy)]
struct EquivRunConfig<'a> {
    flatten_aggregates: bool,
    parallelism: EquivParallelism,
    assertion_semantics: AssertionSemantics,
    assert_label_filter: Option<&'a str>,
    solver: Option<SolverChoice>,
    tool_path: Option<&'a Path>,
}

fn default_max_parallel_proofs() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn normalize_parallelism(requested: usize) -> usize {
    requested.max(1)
}

fn total_free_param_width(module: &PreparedIrModule) -> usize {
    let skip_params = if module.fixed_implicit_activation {
        2
    } else {
        0
    };
    module
        .func
        .params
        .iter()
        .skip(skip_params)
        .map(|param| param.ty.bit_count())
        .sum()
}

fn validate_equiv_compatibility(
    lhs: &PreparedIrModule,
    rhs: &PreparedIrModule,
    allow_flatten: bool,
) -> Result<(), String> {
    if allow_flatten {
        let lhs_param_width = total_free_param_width(lhs);
        let rhs_param_width = total_free_param_width(rhs);
        if lhs_param_width != rhs_param_width {
            return Err(format!(
                "flattened input widths differ: lhs={} rhs={}",
                lhs_param_width, rhs_param_width
            ));
        }
        let lhs_return_width = lhs.func.ret_ty.bit_count();
        let rhs_return_width = rhs.func.ret_ty.bit_count();
        if lhs_return_width != rhs_return_width {
            return Err(format!(
                "return widths differ: lhs={} rhs={}",
                lhs_return_width, rhs_return_width
            ));
        }
        Ok(())
    } else if lhs.func.get_type() != rhs.func.get_type() {
        Err(format!(
            "function signatures differ: lhs={:?} rhs={:?}",
            lhs.func.get_type(),
            rhs.func.get_type()
        ))
    } else {
        Ok(())
    }
}

fn run_prepared_ir_equiv(
    config: EquivRunConfig<'_>,
    lhs: &PreparedIrModule,
    rhs: &PreparedIrModule,
    interrupt: Option<SolverInterruptHandle>,
) -> EquivReport {
    let start_time = Instant::now();
    let result = match validate_equiv_compatibility(lhs, rhs, config.flatten_aggregates) {
        Ok(()) => {
            let choice = config.solver.unwrap_or(SolverChoice::Auto);
            let prover = prover_for_choice(choice, config.tool_path);
            let lhs_side = lhs.as_prover_fn();
            let rhs_side = rhs.as_prover_fn();
            prover.prove_ir_equiv_with_interrupt(
                &lhs_side,
                &rhs_side,
                config.parallelism,
                config.assertion_semantics,
                config.assert_label_filter,
                config.flatten_aggregates,
                interrupt,
            )
        }
        Err(msg) => EquivResult::Error(msg),
    };
    EquivReport {
        duration: start_time.elapsed(),
        result,
    }
}

fn build_ranked_shortlist(
    candidate: &PreparedIrModule,
    members: &[PreparedEquivClassMember],
    max_parallel_proofs: usize,
    shortlist: EquivClassShortlistOptions,
) -> Vec<EquivClassRankingEntry> {
    if members.is_empty() {
        return Vec::new();
    }
    let member_refs: Vec<&ir::Fn> = members.iter().map(|member| &member.module.func).collect();
    let ranked: Vec<RankedFnCandidate> = rank_fn_candidates_by_similarity(
        &candidate.func,
        &member_refs,
        CandidateRankingOptions {
            prefilter_limit: shortlist
                .effective_prefilter_limit(members.len(), max_parallel_proofs),
        },
    );
    ranked
        .into_iter()
        .enumerate()
        .map(
            |(shortlist_rank, candidate_ranking)| EquivClassRankingEntry {
                shortlist_rank,
                member_index: candidate_ranking.index,
                member_id: members[candidate_ranking.index].id.clone(),
                shared_structural_hashes: candidate_ranking.shared_structural_hashes,
                candidate_structural_hashes: candidate_ranking.query_structural_hashes,
                member_structural_hashes: candidate_ranking.candidate_structural_hashes,
                edit_distance: candidate_ranking.edit_distance,
            },
        )
        .collect()
}

fn describe_result_kind(result: &EquivResult) -> String {
    match result {
        EquivResult::Proved => "proved".to_string(),
        EquivResult::Disproved { .. } => "disproved".to_string(),
        EquivResult::Interrupted => "interrupted".to_string(),
        EquivResult::ToolchainDisproved(msg) => {
            format!("toolchain-disproved: {}", msg)
        }
        EquivResult::Error(msg) => format!("error: {}", msg),
    }
}

fn is_conclusive_equiv_result(result: &EquivResult) -> bool {
    matches!(
        result,
        EquivResult::Proved | EquivResult::Disproved { .. } | EquivResult::ToolchainDisproved(_)
    )
}

fn resolve_equiv_class_result(
    shortlisted_members: &[EquivClassRankingEntry],
    completed_proofs: &[EquivClassProofEntry],
) -> EquivClassResult {
    let proved_members: Vec<&EquivClassProofEntry> = completed_proofs
        .iter()
        .filter(|entry| matches!(&entry.report.result, EquivResult::Proved))
        .collect();
    if !proved_members.is_empty() {
        let conflicting: Vec<String> = completed_proofs
            .iter()
            .filter(|entry| {
                is_conclusive_equiv_result(&entry.report.result)
                    && !matches!(&entry.report.result, EquivResult::Proved)
            })
            .map(|entry| {
                format!(
                    "{} ({})",
                    entry.member_id,
                    describe_result_kind(&entry.report.result)
                )
            })
            .collect();
        if !conflicting.is_empty() {
            return EquivClassResult::InvariantViolation {
                message: format!(
                    "concurrently completed proofs disagreed after a winning proof: {}",
                    conflicting.join(", ")
                ),
            };
        }
        if let Some(first_error) =
            completed_proofs
                .iter()
                .find_map(|entry| match &entry.report.result {
                    EquivResult::Error(msg) => Some((entry.member_id.clone(), msg.clone())),
                    _ => None,
                })
        {
            return EquivClassResult::Error {
                message: format!(
                    "proof against member '{}' failed: {}",
                    first_error.0, first_error.1
                ),
            };
        }
        if let Some(winner) = shortlisted_members.iter().find(|ranking| {
            proved_members
                .iter()
                .any(|entry| entry.member_index == ranking.member_index)
        }) {
            return EquivClassResult::Matched(EquivClassMatch {
                member_index: winner.member_index,
                member_id: winner.member_id.clone(),
                shortlist_rank: winner.shortlist_rank,
            });
        }
    }
    if let Some(first_error) =
        completed_proofs
            .iter()
            .find_map(|entry| match &entry.report.result {
                EquivResult::Error(msg) => Some((entry.member_id.clone(), msg.clone())),
                _ => None,
            })
    {
        return EquivClassResult::Error {
            message: format!(
                "proof against member '{}' failed: {}",
                first_error.0, first_error.1
            ),
        };
    }
    if shortlisted_members.is_empty() {
        EquivClassResult::NoMatch {
            reason: EquivClassNoMatchReason::NoMembers,
        }
    } else {
        EquivClassResult::NoMatch {
            reason: EquivClassNoMatchReason::ExhaustedShortlist,
        }
    }
}

/// Dispatches an IR equivalence comparison using the requested solver.
pub fn run_ir_equiv(request: &IrEquivRequest<'_>) -> Result<EquivReport, String> {
    let lhs = PreparedIrModule::from_ir_module(&request.lhs, request.drop_params)?;
    let rhs = PreparedIrModule::from_ir_module(&request.rhs, request.drop_params)?;
    let report = run_prepared_ir_equiv(
        EquivRunConfig {
            flatten_aggregates: request.flatten_aggregates,
            parallelism: request.parallelism,
            assertion_semantics: request.assertion_semantics,
            assert_label_filter: request.assert_label_filter,
            solver: request.solver,
            tool_path: request.tool_path,
        },
        &lhs,
        &rhs,
        None,
    );
    match &report.result {
        EquivResult::Error(msg) => Err(msg.clone()),
        _ => Ok(report),
    }
}

/// Proves whether the candidate belongs to an existing equivalence class.
///
/// The current implementation uses in-process worker threads with cooperative
/// early stop plus solver-level interruption where supported by the backend.
pub fn run_ir_equiv_class_membership(
    request: &EquivClassRequest<'_>,
) -> Result<EquivClassReport, String> {
    let start_time = Instant::now();
    let max_parallel_proofs = normalize_parallelism(request.max_parallel_proofs);
    let candidate = PreparedIrModule::from_ir_module(&request.candidate, request.drop_params)?;
    let prepared_members: Vec<PreparedEquivClassMember> = request
        .members
        .iter()
        .map(|member| {
            Ok(PreparedEquivClassMember {
                id: member.id.clone().into_owned(),
                module: PreparedIrModule::from_ir_module(&member.module, request.drop_params)?,
            })
        })
        .collect::<Result<Vec<PreparedEquivClassMember>, String>>()?;
    let shortlisted_members = build_ranked_shortlist(
        &candidate,
        &prepared_members,
        max_parallel_proofs,
        request.shortlist,
    );

    let completed_proofs: Vec<EquivClassProofEntry> = if shortlisted_members.is_empty() {
        Vec::new()
    } else {
        let worker_count = std::cmp::min(shortlisted_members.len(), max_parallel_proofs);
        let next_shortlist_index = Arc::new(AtomicUsize::new(0));
        let stop_requested = AtomicSolverInterrupt::new();
        let completed_proofs = Arc::new(Mutex::new(Vec::<EquivClassProofEntry>::new()));
        std::thread::scope(|scope| {
            for _ in 0..worker_count {
                let next_shortlist_index_cl = Arc::clone(&next_shortlist_index);
                let stop_requested_cl = stop_requested.clone();
                let completed_proofs_cl = Arc::clone(&completed_proofs);
                let candidate_ref = &candidate;
                let prepared_members_ref = &prepared_members;
                let shortlisted_members_ref = &shortlisted_members;
                let run_config = EquivRunConfig {
                    flatten_aggregates: request.flatten_aggregates,
                    parallelism: request.parallelism,
                    assertion_semantics: request.assertion_semantics,
                    assert_label_filter: request.assert_label_filter,
                    solver: request.solver,
                    tool_path: request.tool_path,
                };
                scope.spawn(move || {
                    loop {
                        if stop_requested_cl.is_interrupted() {
                            break;
                        }
                        let shortlist_index =
                            next_shortlist_index_cl.fetch_add(1, Ordering::SeqCst);
                        if shortlist_index >= shortlisted_members_ref.len() {
                            break;
                        }
                        let ranking = shortlisted_members_ref[shortlist_index].clone();
                        let member = &prepared_members_ref[ranking.member_index];
                        let report = run_prepared_ir_equiv(
                            run_config,
                            candidate_ref,
                            &member.module,
                            Some(stop_requested_cl.handle()),
                        );
                        let proved = matches!(&report.result, EquivResult::Proved);
                        completed_proofs_cl
                            .lock()
                            .expect("proof result collection lock")
                            .push(EquivClassProofEntry {
                                member_index: ranking.member_index,
                                member_id: member.id.clone(),
                                ranking,
                                report,
                            });
                        if proved {
                            stop_requested_cl.interrupt();
                        }
                    }
                });
            }
        });
        let mut completed = Arc::try_unwrap(completed_proofs)
            .expect("no other proof collectors remain")
            .into_inner()
            .expect("proof result collection lock available");
        completed.sort_by(|lhs, rhs| {
            lhs.ranking
                .shortlist_rank
                .cmp(&rhs.ranking.shortlist_rank)
                .then_with(|| lhs.member_index.cmp(&rhs.member_index))
        });
        completed
    };

    let excluded_by_shortlist_count = prepared_members
        .len()
        .saturating_sub(shortlisted_members.len());
    let unstarted_shortlist_count = shortlisted_members
        .len()
        .saturating_sub(completed_proofs.len());
    let result = resolve_equiv_class_result(&shortlisted_members, &completed_proofs);
    Ok(EquivClassReport {
        duration: start_time.elapsed(),
        total_members: prepared_members.len(),
        max_parallel_proofs,
        shortlisted_members,
        completed_proofs,
        excluded_by_shortlist_count,
        unstarted_shortlist_count,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        EquivClassProofEntry, EquivClassRankingEntry, EquivClassResult, EquivReport, EquivResult,
        resolve_equiv_class_result,
    };

    fn make_ranking(
        shortlist_rank: usize,
        member_index: usize,
        member_id: &str,
    ) -> EquivClassRankingEntry {
        EquivClassRankingEntry {
            shortlist_rank,
            member_index,
            member_id: member_id.to_string(),
            shared_structural_hashes: 0,
            candidate_structural_hashes: 0,
            member_structural_hashes: 0,
            edit_distance: shortlist_rank as u64,
        }
    }

    fn make_completed_proof(
        ranking: EquivClassRankingEntry,
        result: EquivResult,
    ) -> EquivClassProofEntry {
        EquivClassProofEntry {
            member_index: ranking.member_index,
            member_id: ranking.member_id.clone(),
            ranking,
            report: EquivReport {
                duration: std::time::Duration::default(),
                result,
            },
        }
    }

    #[test]
    fn test_resolve_equiv_class_result_reports_invariant_violation_on_conflict() {
        let shortlist = vec![make_ranking(0, 0, "winner"), make_ranking(1, 1, "conflict")];
        let completed = vec![
            make_completed_proof(shortlist[0].clone(), EquivResult::Proved),
            make_completed_proof(
                shortlist[1].clone(),
                EquivResult::ToolchainDisproved("counterexample".to_string()),
            ),
        ];

        match resolve_equiv_class_result(&shortlist, &completed) {
            EquivClassResult::InvariantViolation { message } => {
                assert!(message.contains("conflict"));
            }
            other => panic!("expected invariant violation, got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_equiv_class_result_ignores_interrupted_losers_after_winner() {
        let shortlist = vec![
            make_ranking(0, 0, "winner"),
            make_ranking(1, 1, "cancelled"),
        ];
        let completed = vec![
            make_completed_proof(shortlist[0].clone(), EquivResult::Proved),
            make_completed_proof(shortlist[1].clone(), EquivResult::Interrupted),
        ];

        assert_eq!(
            resolve_equiv_class_result(&shortlist, &completed),
            EquivClassResult::Matched(super::EquivClassMatch {
                member_index: 0,
                member_id: "winner".to_string(),
                shortlist_rank: 0,
            })
        );
    }
}
