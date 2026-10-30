// SPDX-License-Identifier: Apache-2.0

//! Script-driven obligation tree execution with selectors and tactics.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use crate::dslx_tactics::obligations::ProverObligation;
use crate::dslx_tactics::tactics::{IsTactic, Tactic};
use crate::proof_outcome::{ProofResultHeader, ProofStatus};
use crate::proof_plan::report::{
    IndefiniteReason, ProverReport, ProverReportNode, TaskOutcome, TaskTiming,
};
use crate::proof_timing::{ProofTiming, TimingObject, optional_measurement};
use crate::prover::SolverChoice;
use serde::{Deserialize, Deserializer, Serialize};

/// A path from the root to an obligation in the decomposition tree.
pub type Selector = Vec<String>;

fn root_selector() -> Selector {
    vec!["root".to_string()]
}
fn join_selector(parent: &Selector, seg: &str) -> Selector {
    let mut v = parent.clone();
    v.push(seg.to_string());
    v
}
fn selector_to_path(sel: &Selector) -> String {
    sel.join("/")
}

/// The most recent proof state of an obligation node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeStatus {
    Open,
    /// The last attempt did not decide the obligation; a missing result has no
    /// reason.
    Pending(Option<IndefiniteReason>),
    Solved,
    Failed,
}

/// An obligation or its decomposition, retaining the original for rollback.
#[derive(Debug)]
pub enum NodeKind {
    Leaf(ProverObligation),
    Internal {
        tactic: Tactic,
        children: Vec<OblNode>,
        original: ProverObligation,
    },
}

/// An addressable obligation and its current proof state.
#[derive(Debug)]
pub struct OblNode {
    pub segment: String,
    pub selector: Selector,
    pub kind: NodeKind,
    pub status: NodeStatus,
    /// A queued proof request, consumed when its runner report is accepted.
    pub solve: bool,
}

impl OblNode {
    /// Creates an unproved child obligation beneath the given parent selector.
    pub fn leaf(seg: String, parent: &Selector, ob: ProverObligation) -> Self {
        Self {
            segment: seg.clone(),
            selector: join_selector(parent, &seg),
            kind: NodeKind::Leaf(ob),
            status: NodeStatus::Open,
            solve: false,
        }
    }
}

/// Solver and DSLX configuration passed unchanged to the obligation runner.
#[derive(Debug)]
pub struct OblTreeConfig {
    pub dslx_stdlib_path: Option<PathBuf>,
    pub dslx_paths: Vec<PathBuf>,
    pub solver: Option<SolverChoice>,
    pub timeout_ms: Option<u64>,
}

/// The evolving decomposition tree and configuration for its proofs.
#[derive(Debug)]
pub struct OblTree {
    pub root: OblNode,
    pub config: OblTreeConfig,
}

impl OblTree {
    /// Creates an unproved root obligation with a canonical root selector.
    pub fn new(mut root_obligation: ProverObligation, config: OblTreeConfig) -> Self {
        // Ensure root obligation has selector segment = "root"
        root_obligation.selector_segment = "root".to_string();
        let root = OblNode {
            segment: "root".to_string(),
            selector: root_selector(),
            kind: NodeKind::Leaf(root_obligation),
            status: NodeStatus::Open,
            solve: false,
        };
        Self { root, config }
    }

    /// Finds a mutable node by its full selector in the surviving tree.
    fn find_mut<'a>(node: &'a mut OblNode, sel: &Selector) -> Option<&'a mut OblNode> {
        if &node.selector == sel {
            return Some(node);
        }
        if let NodeKind::Internal { children, .. } = &mut node.kind {
            for c in children.iter_mut() {
                if let Some(found) = Self::find_mut(c, sel) {
                    return Some(found);
                }
            }
        }
        None
    }

    // Note: parent lookup is performed by computing the parent selector in the
    // executor.
}

/// A DSLX decomposition or proof action addressed to a single obligation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Command {
    Apply(Tactic),
    Solve,
}

/// Script syntax shared by proof domains with their own command sets.
pub mod syntax {
    use super::Selector;
    use serde::{Deserialize, Serialize};

    /// One serialized command and the obligation it addresses.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct ScriptStep<C> {
        pub selector: Selector,
        pub command: C,
    }
}

/// One serialized action in a DSLX tactic script.
pub type ScriptStep = syntax::ScriptStep<Command>;

/// Replaces a leaf with a tactic's children while retaining its original proof.
fn apply_tactic(
    node: &mut OblNode,
    parent_selector: &Selector,
    tactic: Tactic,
) -> Result<(), String> {
    let base = match &node.kind {
        NodeKind::Leaf(ob) => ob.clone(),
        _ => return Err("tactic requires a leaf".to_string()),
    };
    let children_obs = tactic.apply(&base)?;
    let mut children: Vec<OblNode> = Vec::new();
    for (idx, ob) in children_obs.into_iter().enumerate() {
        if ob.selector_segment.is_empty() {
            return Err(format!(
                "tactic produced child {} with empty selector segment",
                idx + 1
            ));
        }
        let seg = ob.selector_segment.clone();
        children.push(OblNode::leaf(seg, parent_selector, ob));
    }
    node.kind = NodeKind::Internal {
        tactic,
        children,
        original: base,
    };
    Ok(())
}

/// Collects selected leaves in tree order for one obligation-runner batch.
fn collect_solve_targets(node: &OblNode, out: &mut Vec<(Selector, ProverObligation)>) {
    match &node.kind {
        NodeKind::Leaf(ob) => {
            if node.solve {
                out.push((node.selector.clone(), ob.clone()));
            }
        }
        NodeKind::Internal { children, .. } => {
            for c in children.iter() {
                collect_solve_targets(c, out);
            }
        }
    }
}

/// Records unproved leaves and outstanding execution errors in the surviving
/// tree.
fn collect_unresolved(node: &OblNode, report: &mut ScriptReport) {
    match &node.kind {
        NodeKind::Leaf(_) => {
            if node.status != NodeStatus::Solved {
                report.unresolved.push(node.selector.clone());
            }
            if let NodeStatus::Pending(Some(reason)) = node.status {
                report.has_execution_error |= reason.is_execution_error();
            }
        }
        NodeKind::Internal { children, .. } => {
            for child in children {
                collect_unresolved(child, report);
            }
        }
    }
}

/// The proof states a runner can report for an attempted obligation.
#[derive(Clone, Copy, Debug)]
enum TaskSimple {
    Success,
    Failure,
    Indefinite(IndefiniteReason),
}

/// One validated runner result, borrowing captured diagnostics from its report.
struct ReportedTask<'a> {
    status: Option<TaskSimple>,
    stdout: Option<&'a str>,
    stderr: Option<&'a str>,
    timing: &'a TaskTiming,
}

/// Correlates results by selected task ID, rejecting ambiguous or foreign IDs.
fn collect_outcomes<'a>(
    report: &'a ProverReportNode,
    expected: &BTreeSet<&str>,
) -> Result<HashMap<&'a str, ReportedTask<'a>>, String> {
    /// Validates the subtree and returns whether a selected task reported an
    /// execution error.
    fn visit<'a>(
        report: &'a ProverReportNode,
        expected: &BTreeSet<&str>,
        out: &mut HashMap<&'a str, ReportedTask<'a>>,
    ) -> Result<bool, String> {
        match report {
            ProverReportNode::Task {
                task_id,
                outcome,
                stdout,
                stderr,
                timing,
                ..
            } => {
                let id = task_id
                    .as_deref()
                    .ok_or("proof runner returned a task without an ID")?;
                if !expected.contains(id) {
                    return Err(format!("proof runner returned unexpected task ID: {id}"));
                }
                let status = outcome.as_ref().map(|outcome| match outcome {
                    TaskOutcome::Normal { success: true } => TaskSimple::Success,
                    TaskOutcome::Normal { success: false } => TaskSimple::Failure,
                    TaskOutcome::Indefinite { reason } => TaskSimple::Indefinite(*reason),
                });
                let task = ReportedTask {
                    status,
                    stdout: stdout.as_deref(),
                    stderr: stderr.as_deref(),
                    timing,
                };
                if out.insert(id, task).is_some() {
                    return Err(format!("proof runner returned duplicate task ID: {id}"));
                }
                Ok(outcome.is_some_and(|o| o.proof_status() == ProofStatus::ExecutionError))
            }
            ProverReportNode::Group { tasks, outcome, .. } => {
                let mut has_task_error = false;
                for task in tasks {
                    has_task_error |= visit(task, expected, out)?;
                }
                if !has_task_error
                    && outcome.is_some_and(|o| o.proof_status() == ProofStatus::ExecutionError)
                {
                    return Err(
                        "proof runner group error cannot be assigned to a selected task".into(),
                    );
                }
                Ok(has_task_error)
            }
        }
    }
    let mut out = HashMap::new();
    visit(report, expected, &mut out)?;
    Ok(out)
}

/// Updates a surviving node and records attempt history, rolling back failures.
fn update_from_outcome(
    tree: &mut OblTree,
    sel: Selector,
    task: Option<&ReportedTask<'_>>,
    report: &mut ScriptReport,
) {
    if let Some(task) = task {
        report.attempts.push(AttemptTiming {
            selector: sel.clone(),
            timing: task.timing.clone(),
        });
    }
    let status = task.and_then(|task| task.status);
    if let Some(node) = OblTree::find_mut(&mut tree.root, &sel) {
        node.solve = false;
        node.status = match status {
            Some(TaskSimple::Success) => NodeStatus::Solved,
            Some(TaskSimple::Failure) => NodeStatus::Failed,
            Some(TaskSimple::Indefinite(reason)) => NodeStatus::Pending(Some(reason)),
            None => NodeStatus::Pending(None),
        };
    }
    match status {
        Some(TaskSimple::Success) => {
            report.solved.push(sel);
        }
        Some(TaskSimple::Failure) => {
            if let Some(task) = task {
                report.failed.push(TaskLogEntry {
                    reason: None,
                    selector: sel.clone(),
                    stdout: task.stdout.map(str::to_owned),
                    stderr: task.stderr.map(str::to_owned),
                });
            }
            if sel.len() > 1 {
                let parent_sel: Selector = sel[..sel.len() - 1].to_vec();
                if let Some(parent) = OblTree::find_mut(&mut tree.root, &parent_sel) {
                    if let NodeKind::Internal { original, .. } = &parent.kind {
                        let original_ob = original.clone();
                        parent.kind = NodeKind::Leaf(original_ob);
                        parent.status = NodeStatus::Open;
                        parent.solve = false;
                    }
                    report.rolled_back.push(RollbackEntry {
                        selector: parent_sel,
                        reason_selector: sel,
                    });
                }
            }
        }
        Some(TaskSimple::Indefinite(reason)) => {
            if let Some(task) = task {
                report.indefinite.push(TaskLogEntry {
                    reason: Some(reason),
                    selector: sel,
                    stdout: task.stdout.map(str::to_owned),
                    stderr: task.stderr.map(str::to_owned),
                });
            }
        }
        None => {
            // A missing outcome leaves the node pending without a conclusion.
        }
    }
}

/// Captured runner output for an unsuccessful or inconclusive proof attempt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskLogEntry {
    pub selector: Selector,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    /// Why an attempt remained unresolved; absent for actual disproofs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<IndefiniteReason>,
}

/// A restored parent obligation and the failed child that caused its rollback.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollbackEntry {
    pub selector: Selector,
    pub reason_selector: Selector,
}

/// The timing a runner reported for one selected obligation in this invocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptTiming {
    pub selector: Selector,
    #[serde(
        default,
        deserialize_with = "TaskTiming::deserialize_or_empty",
        skip_serializing_if = "TaskTiming::is_empty"
    )]
    pub timing: TaskTiming,
}

/// Reads valid attempts in order, ignoring malformed entries or a non-list.
fn deserialize_attempts<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<AttemptTiming>, D::Error> {
    let attempts: Vec<TimingObject<AttemptTiming>> = optional_measurement(d)?.unwrap_or_default();
    Ok(attempts
        .into_iter()
        .filter_map(TimingObject::into_option)
        .collect())
}

/// Attempt outcomes and unresolved obligations after all tactic rollbacks.
/// Malformed values of optional telemetry fields do not invalidate the proof
/// conclusion. Duplicate report fields remain ambiguous and are rejected.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScriptReport {
    /// Wall time of this invocation, from applying steps through resolving the
    /// final tree. ScriptExecution also exposes this measurement at the top
    /// level so schedulers can read it like any other worker's timing.
    #[serde(
        default,
        deserialize_with = "ProofTiming::deserialize_optional",
        skip_serializing_if = "Option::is_none"
    )]
    pub timing: Option<ProofTiming>,
    /// Reported tasks in selected tree order, including discarded children.
    #[serde(default, deserialize_with = "deserialize_attempts")]
    pub attempts: Vec<AttemptTiming>,
    pub solved: Vec<Selector>,
    pub indefinite: Vec<TaskLogEntry>,
    pub failed: Vec<TaskLogEntry>,
    pub rolled_back: Vec<RollbackEntry>,
    /// Final leaves that are not proved, including unselected and reopened
    /// leaves.
    ///
    /// Required during deserialization. Legacy reports without this field are
    /// intentionally unsupported: absence means completion is unknown, while an
    /// empty list asserts that no unproved leaves remain. Do not default it.
    pub unresolved: Vec<Selector>,
    /// Whether the surviving root leaf is disproved, including prior calls.
    /// Failed decomposition children are not counterexamples to the root.
    pub root_disproved: bool,
    /// Whether a surviving leaf's latest attempt ended in an execution error.
    pub has_execution_error: bool,
}

/// Final tactic report together with the subprocess output for the selected
/// leaves.
pub struct ScriptExecution {
    pub report: ScriptReport,
    /// Absent when the script did not select any leaves to solve.
    pub task_report: Option<ProverReport>,
}

// Child JSON carries the conclusion about the root, not the last solver batch.
impl Serialize for ScriptExecution {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct TacticOutcome<'a> {
            #[serde(flatten)]
            conclusion: ProofResultHeader,
            #[serde(skip_serializing_if = "Option::is_none")]
            timing: Option<&'a ProofTiming>,
            report: &'a ScriptReport,
        }
        TacticOutcome {
            conclusion: self.report.proof_status().into(),
            timing: self.report.timing.as_ref(),
            report: &self.report,
        }
        .serialize(serializer)
    }
}

impl ScriptReport {
    /// Classifies the original obligation, distinguishing failed decompositions
    /// from counterexamples to the root itself.
    pub fn proof_status(&self) -> ProofStatus {
        if self.is_complete() {
            ProofStatus::Proved
        } else if self.root_disproved {
            ProofStatus::Disproved
        } else if self.has_execution_error {
            ProofStatus::ExecutionError
        } else {
            ProofStatus::Inconclusive
        }
    }

    /// Whether the entire proof is discharged without failed or indefinite
    /// tasks.
    pub fn is_complete(&self) -> bool {
        !self.root_disproved
            && !self.has_execution_error
            && self.unresolved.is_empty()
            && self.failed.is_empty()
            && self.indefinite.is_empty()
    }
}

/// Parses a tactic script from a string as a JSON array of ScriptStep objects.
pub fn parse_script_steps_from_json_str(input: &str) -> Result<Vec<ScriptStep>, String> {
    serde_json::from_str::<Vec<ScriptStep>>(input)
        .map_err(|e| format!("tactic script: expected JSON array of ScriptStep: {}", e))
}

/// Parses a tactic script from a string as JSONL (one ScriptStep JSON object
/// per non-empty, non-comment line).
pub fn parse_script_steps_from_jsonl_str(input: &str) -> Result<Vec<ScriptStep>, String> {
    let mut steps: Vec<ScriptStep> = Vec::new();
    for (lineno, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        match serde_json::from_str::<ScriptStep>(line) {
            Ok(step) => steps.push(step),
            Err(e) => {
                return Err(format!(
                    "tactic script (jsonl) parse error on line {}: {}",
                    lineno + 1,
                    e
                ));
            }
        }
    }
    Ok(steps)
}

/// Executes selected proofs and reports completion against the surviving tree.
///
/// The runner receives one batch of `(task ID, obligation)` pairs and the
/// tree's configuration. It must return each result under its supplied task ID;
/// missing results remain pending. Duplicate, absent, or unexpected result IDs
/// are errors. A group reporting an execution, protocol, or process error must
/// include a selected task with an execution error in its subtree. A runner
/// that fails before it can associate the error with a task should return
/// `Err`. The runner is not called when no leaves are selected. Applying script
/// steps precedes running proofs, so an error does not undo successfully
/// applied steps.
///
/// Accepting a report consumes all selected Solve requests, including those
/// with missing or inconclusive results. A later call must explicitly Solve a
/// leaf again to retry it. Runner errors and invalid reports leave requests
/// queued, without committing outcomes.
pub fn execute_script_with_runner(
    tree: &mut OblTree,
    steps: &[ScriptStep],
    runner: impl FnOnce(&[(String, ProverObligation)], &OblTreeConfig) -> Result<ProverReport, String>,
) -> Result<ScriptReport, String> {
    run_script(tree, steps, runner).map(|execution| execution.report)
}

/// Retains the accepted batch so callers can display diagnostics after the
/// engine finishes.
pub(super) fn run_script(
    tree: &mut OblTree,
    steps: &[ScriptStep],
    runner: impl FnOnce(&[(String, ProverObligation)], &OblTreeConfig) -> Result<ProverReport, String>,
) -> Result<ScriptExecution, String> {
    let start = Instant::now();
    // 1) Apply all tactics except Solve; mark Solve flags only.
    for step in steps {
        match &step.command {
            Command::Solve => {
                let node = OblTree::find_mut(&mut tree.root, &step.selector).ok_or_else(|| {
                    format!("selector not found: {}", selector_to_path(&step.selector))
                })?;
                if !matches!(node.kind, NodeKind::Leaf(_)) {
                    return Err("solve requires a leaf".to_string());
                }
                node.solve = true;
            }
            Command::Apply(tactic) => {
                let node = OblTree::find_mut(&mut tree.root, &step.selector).ok_or_else(|| {
                    format!("selector not found: {}", selector_to_path(&step.selector))
                })?;
                apply_tactic(node, &step.selector, tactic.clone())?;
            }
        }
    }

    // 2) Gather solve leaves and build obligations with selector paths.
    let mut targets: Vec<(Selector, ProverObligation)> = Vec::new();
    collect_solve_targets(&tree.root, &mut targets);
    let mut obligations_with_ids: Vec<(String, ProverObligation)> = Vec::new();
    let mut selected_ids = BTreeSet::new();
    for (sel, ob) in targets.iter() {
        let path = selector_to_path(sel);
        if !selected_ids.insert(path.clone()) {
            return Err(format!("duplicate selected task ID: {path}"));
        }
        obligations_with_ids.push((path, ob.clone()));
    }

    log::debug!("obligations_to_solve={}", obligations_with_ids.len());

    let mut completion = ScriptReport::default();
    let task_report = if obligations_with_ids.is_empty() {
        None
    } else {
        let report = runner(&obligations_with_ids, &tree.config)?;
        let expected = selected_ids.iter().map(String::as_str).collect();
        // Validate the complete report before committing any reported outcomes.
        let outcomes = collect_outcomes(&report.plan, &expected)?;
        for (sel, _) in targets {
            let path = selector_to_path(&sel);
            update_from_outcome(tree, sel, outcomes.get(path.as_str()), &mut completion);
        }
        Some(report)
    };

    // Outcomes for discarded children remain useful attempt history, but only
    // the surviving tree determines which obligations still need proof.
    collect_unresolved(&tree.root, &mut completion);
    completion.root_disproved =
        matches!(tree.root.kind, NodeKind::Leaf(_)) && tree.root.status == NodeStatus::Failed;
    completion.timing = Some(ProofTiming::elapsed(start));
    Ok(ScriptExecution {
        report: completion,
        task_report,
    })
}

#[cfg(test)]
mod tests {
    use super::Command::Solve;
    use super::*;
    use crate::dslx_equiv::{DslxEquivRequest, DslxModule, run_dslx_equiv};
    use crate::dslx_tactics::obligations::{
        FileWithHistory, LecObligation, LecSide, ObligationPayload, ProverObligation, SourceFile,
    };
    #[cfg(feature = "has-bitwuzla")]
    use crate::dslx_tactics::tactics::cosliced::{CoslicedTactic, NamedSlice};
    use crate::dslx_tactics::tactics::focus::{FocusPair, FocusTactic};
    use crate::proof_outcome::aggregate_proof_statuses;
    use crate::proof_plan::config::GroupKind;
    use crate::proof_plan::report::IndefiniteReason;
    use std::path::Path;
    use std::time::Duration;
    use test_case::test_case;

    #[test]
    fn dslx_commands_and_steps_infer_without_type_annotations() {
        let command = Command::Solve;
        assert_eq!(serde_json::to_string(&command).unwrap(), r#""Solve""#);
        let step = ScriptStep {
            selector: root_selector(),
            command: serde_json::from_str(r#""Solve""#).unwrap(),
        };
        assert_eq!(
            serde_json::to_string(&step).unwrap(),
            r#"{"selector":["root"],"command":"Solve"}"#
        );
        assert!(matches!(step.command, Solve));
    }

    /// Runs selected equivalence obligations directly through the library API.
    fn run_test_obligations(
        obligations: &[(String, ProverObligation)],
        config: &OblTreeConfig,
    ) -> Result<ProverReport, String> {
        let mut tasks = Vec::new();
        let mut statuses = Vec::new();
        for (id, obligation) in obligations {
            let ObligationPayload::Lec(lec) = &obligation.payload else {
                return Err("test runner requires equivalence obligations".to_string());
            };
            let lhs_uf = lec
                .lhs
                .uf_map
                .iter()
                .map(|(name, alias)| {
                    xlsynth::mangle_dslx_name("lhs", name)
                        .map(|name| (name, alias.clone()))
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<HashMap<_, _>, String>>()?;
            let rhs_uf = lec
                .rhs
                .uf_map
                .iter()
                .map(|(name, alias)| {
                    xlsynth::mangle_dslx_name("rhs", name)
                        .map(|name| (name, alias.clone()))
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<HashMap<_, _>, String>>()?;
            let lhs = DslxModule::new(&lec.lhs.file.text, &lec.lhs.top_func)
                .with_path(Some(Path::new("lhs.x")))
                .with_fixed_implicit_activation(lec.lhs.fixed_implicit_activation)
                .with_owned_uf_map(lhs_uf);
            let rhs = DslxModule::new(&lec.rhs.file.text, &lec.rhs.top_func)
                .with_path(Some(Path::new("rhs.x")))
                .with_fixed_implicit_activation(lec.rhs.fixed_implicit_activation)
                .with_owned_uf_map(rhs_uf);
            let request = DslxEquivRequest::new(lhs, rhs)
                .with_assertion_semantics(lec.options.assertion_semantics)
                .with_assume_enum_in_bound(lec.options.assume_enum_in_bound)
                .with_assert_label_filter(lec.options.assert_label_filter.as_deref())
                .with_drop_params(&lec.options.drop_params)
                .with_flatten_aggregates(lec.options.flatten_aggregates)
                .with_parallelism(lec.options.parallelism)
                .with_solver(config.solver.clone())
                .with_dslx_stdlib_path(config.dslx_stdlib_path.as_deref())
                .with_additional_search_paths(config.dslx_paths.clone());
            let proof = run_dslx_equiv(&request)?;
            let status = ProofStatus::from(&proof.result);
            statuses.push(status);
            tasks.push(ProverReportNode::Task {
                timing: Default::default(),
                task_id: Some(id.clone()),
                cmdline: None,
                outcome: Some(status.into()),
                stdout: None,
                stderr: proof.error_str(),
            });
        }
        let status = aggregate_proof_statuses(statuses);
        Ok(ProverReport {
            wall_micros: None,
            plan: ProverReportNode::Group {
                kind: GroupKind::All,
                outcome: Some(status.into()),
                tasks,
            },
        })
    }

    /// Executes a test script using the in-process equivalence runner.
    fn execute_script(tree: &mut OblTree, steps: &[ScriptStep]) -> Result<ScriptReport, String> {
        execute_script_with_runner(tree, steps, run_test_obligations)
    }

    /// Returns success for every supplied obligation in fake-runner tests.
    fn accept_all_obligations(
        obligations: &[(String, ProverObligation)],
        _config: &OblTreeConfig,
    ) -> Result<ProverReport, String> {
        Ok(ProverReport {
            wall_micros: None,
            plan: ProverReportNode::Group {
                kind: GroupKind::All,
                outcome: Some(TaskOutcome::Normal { success: true }),
                tasks: obligations
                    .iter()
                    .map(|(id, _)| ProverReportNode::Task {
                        timing: Default::default(),
                        task_id: Some(id.clone()),
                        cmdline: None,
                        outcome: Some(TaskOutcome::Normal { success: true }),
                        stdout: None,
                        stderr: None,
                    })
                    .collect(),
            },
        })
    }

    fn sel(parts: &[&str]) -> Selector {
        parts.iter().map(|s| s.to_string()).collect()
    }

    fn base_obligation() -> ProverObligation {
        let dslx = "pub fn f(x: u8) -> u8 { x + u8:3 }".to_string();
        let lhs = LecSide {
            fixed_implicit_activation: false,
            top_func: "f".to_string(),
            uf_map: std::collections::BTreeMap::new(),
            file: FileWithHistory {
                base_source: SourceFile::Text(dslx.clone()),
                edits: Vec::new(),
                text: dslx.clone(),
            },
        };
        let rhs = LecSide {
            fixed_implicit_activation: false,
            top_func: "f".to_string(),
            uf_map: std::collections::BTreeMap::new(),
            file: FileWithHistory {
                base_source: SourceFile::Text(dslx.clone()),
                edits: Vec::new(),
                text: dslx,
            },
        };
        ProverObligation {
            selector_segment: "root".to_string(),
            description: None,
            payload: ObligationPayload::Lec(LecObligation {
                lhs,
                rhs,
                options: Default::default(),
            }),
        }
    }

    fn base_config() -> OblTreeConfig {
        OblTreeConfig {
            dslx_stdlib_path: None,
            dslx_paths: vec![],
            solver: None,
            timeout_ms: None,
        }
    }

    /// Creates a main-function equivalence tree with Bitwuzla as its solver.
    #[cfg(feature = "has-bitwuzla")]
    fn bitwuzla_lec_tree(lhs: &str, rhs: &str) -> OblTree {
        let obligation = ProverObligation {
            selector_segment: "root".to_string(),
            description: None,
            payload: ObligationPayload::Lec(LecObligation {
                lhs: LecSide::from_text("main", lhs),
                rhs: LecSide::from_text("main", rhs),
                options: Default::default(),
            }),
        };
        let mut config = base_config();
        config.solver = Some(SolverChoice::Bitwuzla);
        OblTree::new(obligation, config)
    }

    #[cfg(feature = "has-bitwuzla")]
    #[test_case("pair_1", "skeleton"; "pair_then_skeleton")]
    #[test_case("skeleton", "pair_1"; "skeleton_then_pair")]
    fn focus_requires_all_leaves_across_executions(first: &str, remaining: &str) {
        let mut tree = bitwuzla_lec_tree(
            r#"
fn helper(x: u8) -> u8 { x + u8:1 }
fn main(x: u8) -> u8 { helper(x) }
"#,
            r#"
fn helper(x: u8) -> u8 { u8:1 + x }
fn main(x: u8) -> u8 { helper(x) }
"#,
        );
        let empty = execute_script(&mut tree, &[]).unwrap();
        assert!(!empty.is_complete());
        assert_eq!(empty.unresolved, vec![root_selector()]);

        let applied = execute_script(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Apply(Tactic::Focus(FocusTactic {
                    pairs: vec![FocusPair {
                        lhs: "helper".to_string(),
                        rhs: "helper".to_string(),
                    }],
                })),
            }],
        )
        .unwrap();
        assert!(!applied.is_complete());
        assert_eq!(
            applied.unresolved,
            vec![sel(&["root", "pair_1"]), sel(&["root", "skeleton"])]
        );

        let partial = execute_script(
            &mut tree,
            &[ScriptStep {
                selector: sel(&["root", first]),
                command: Command::Solve,
            }],
        )
        .unwrap();
        assert_eq!(partial.solved, vec![sel(&["root", first])]);
        assert!(partial.failed.is_empty());
        assert!(partial.indefinite.is_empty());
        assert!(!partial.is_complete());
        assert_eq!(partial.unresolved, vec![sel(&["root", remaining])]);

        let complete = execute_script_with_runner(
            &mut tree,
            &[ScriptStep {
                selector: sel(&["root", remaining]),
                command: Command::Solve,
            }],
            |obligations, config| {
                assert_eq!(
                    obligations
                        .iter()
                        .map(|(id, _)| id.as_str())
                        .collect::<Vec<_>>(),
                    vec![format!("root/{remaining}")]
                );
                run_test_obligations(obligations, config)
            },
        )
        .unwrap();
        assert!(complete.is_complete(), "incomplete proof: {complete:?}");
        assert_eq!(complete.unresolved, Vec::<Selector>::new());
    }

    #[test]
    #[cfg(feature = "has-bitwuzla")]
    fn successful_focus_skeleton_does_not_prove_inequivalent_helpers() {
        let mut tree = bitwuzla_lec_tree(
            r#"
fn helper(x: u8) -> u8 { x + u8:1 }
fn main(x: u8) -> u8 { helper(x) }
"#,
            r#"
fn helper(x: u8) -> u8 { x + u8:2 }
fn main(x: u8) -> u8 { helper(x) }
"#,
        );
        let partial = execute_script(
            &mut tree,
            &[
                ScriptStep {
                    selector: root_selector(),
                    command: Command::Apply(Tactic::Focus(FocusTactic {
                        pairs: vec![FocusPair {
                            lhs: "helper".to_string(),
                            rhs: "helper".to_string(),
                        }],
                    })),
                },
                ScriptStep {
                    selector: sel(&["root", "skeleton"]),
                    command: Command::Solve,
                },
            ],
        )
        .unwrap();
        assert_eq!(partial.solved, vec![sel(&["root", "skeleton"])]);
        assert!(partial.failed.is_empty());
        assert!(partial.indefinite.is_empty());
        assert!(!partial.is_complete());
        assert_eq!(partial.unresolved, vec![sel(&["root", "pair_1"])]);

        let failed = execute_script(
            &mut tree,
            &[ScriptStep {
                selector: sel(&["root", "pair_1"]),
                command: Command::Solve,
            }],
        )
        .unwrap();
        assert!(!failed.is_complete());
        assert_eq!(failed.unresolved, vec![root_selector()]);
        assert_eq!(failed.failed.len(), 1);
        assert_eq!(failed.failed[0].selector, sel(&["root", "pair_1"]));
        assert_eq!(failed.rolled_back.len(), 1);
        assert_eq!(failed.rolled_back[0].selector, root_selector());
        assert_eq!(
            failed.rolled_back[0].reason_selector,
            sel(&["root", "pair_1"])
        );
    }

    #[cfg(feature = "has-bitwuzla")]
    #[test_case(false; "distinct_helpers_remain_inequivalent")]
    #[test_case(true; "matching_helpers_remain_equivalent")]
    fn nested_focus_preserves_distinct_helper_assumptions(equivalent: bool) {
        let helpers = r#"
fn a(x: u8) -> u8 { x }
fn b(x: u8) -> u8 { !a(x) }
"#;
        let lhs_result = if equivalent {
            "b(x) - a(x)"
        } else {
            "a(x) - b(x)"
        };
        let lhs = format!("{helpers}\nfn main(x: u8) -> u8 {{ {lhs_result} }}");
        let rhs = format!("{helpers}\nfn main(x: u8) -> u8 {{ b(x) - a(x) }}");

        let mut direct_tree = bitwuzla_lec_tree(&lhs, &rhs);
        let direct = execute_script(
            &mut direct_tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Solve,
            }],
        )
        .unwrap();
        assert_eq!(direct.is_complete(), equivalent);
        assert_eq!(direct.failed.len(), usize::from(!equivalent));

        // Both helper pairs are individually equivalent across the two sides,
        // but a and b must retain distinct UF assumptions in nested skeletons.
        let mut nested_tree = bitwuzla_lec_tree(&lhs, &rhs);
        let nested = execute_script(
            &mut nested_tree,
            &[
                ScriptStep {
                    selector: root_selector(),
                    command: Command::Apply(Tactic::Focus(FocusTactic {
                        pairs: vec![FocusPair {
                            lhs: "a".to_string(),
                            rhs: "a".to_string(),
                        }],
                    })),
                },
                ScriptStep {
                    selector: sel(&["root", "skeleton"]),
                    command: Command::Apply(Tactic::Focus(FocusTactic {
                        pairs: vec![FocusPair {
                            lhs: "b".to_string(),
                            rhs: "b".to_string(),
                        }],
                    })),
                },
                ScriptStep {
                    selector: sel(&["root", "pair_1"]),
                    command: Command::Solve,
                },
                ScriptStep {
                    selector: sel(&["root", "skeleton", "pair_1"]),
                    command: Command::Solve,
                },
                ScriptStep {
                    selector: sel(&["root", "skeleton", "skeleton"]),
                    command: Command::Solve,
                },
            ],
        )
        .unwrap();
        assert_eq!(nested.is_complete(), equivalent, "{nested:?}");
        assert_eq!(nested.failed.len(), usize::from(!equivalent));
        assert_eq!(
            nested.unresolved,
            if equivalent {
                vec![]
            } else {
                vec![sel(&["root", "skeleton"])]
            }
        );
    }

    #[test_case(TaskOutcome::Normal { success: true }; "discarded_sibling_proved")]
    #[test_case(TaskOutcome::Indefinite { reason: IndefiniteReason::ExecutionError }; "discarded_sibling_execution_error")]
    #[test_case(TaskOutcome::Indefinite { reason: IndefiniteReason::GroupCriteriaAlreadyMet }; "discarded_sibling_canceled")]
    fn rollback_classifies_the_restored_root(sibling_outcome: TaskOutcome) {
        let mut tree = OblTree::new(base_obligation(), base_config());
        execute_script(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Apply(Tactic::Focus(FocusTactic {
                    pairs: vec![FocusPair {
                        lhs: "f".to_string(),
                        rhs: "f".to_string(),
                    }],
                })),
            }],
        )
        .unwrap();
        let pair_timing = TaskTiming {
            queued_micros: Some(3),
            worker_wall_micros: Some(29),
            worker_timing: Some(Duration::from_micros(17).into()),
        };
        let sibling_timing = TaskTiming {
            queued_micros: Some(12),
            ..Default::default()
        };
        let report = ProverReport {
            wall_micros: None,
            plan: ProverReportNode::Group {
                kind: GroupKind::All,
                outcome: Some(TaskOutcome::Normal { success: false }),
                // Return the sibling first, as a concurrent runner may do.
                tasks: [
                    ("root/skeleton", sibling_outcome, sibling_timing.clone()),
                    (
                        "root/pair_1",
                        ProofStatus::Disproved.into(),
                        pair_timing.clone(),
                    ),
                ]
                .into_iter()
                .map(|(path, outcome, timing)| ProverReportNode::Task {
                    timing,
                    task_id: Some(path.to_string()),
                    cmdline: None,
                    outcome: Some(outcome),
                    stdout: None,
                    stderr: None,
                })
                .collect(),
            },
        };
        let attempted = execute_script_with_runner(
            &mut tree,
            &[
                ScriptStep {
                    selector: sel(&["root", "pair_1"]),
                    command: Command::Solve,
                },
                ScriptStep {
                    selector: sel(&["root", "skeleton"]),
                    command: Command::Solve,
                },
            ],
            |_, _| Ok(report),
        )
        .unwrap();
        assert_eq!(
            attempted.attempts,
            [
                AttemptTiming {
                    selector: sel(&["root", "pair_1"]),
                    timing: pair_timing
                },
                AttemptTiming {
                    selector: sel(&["root", "skeleton"]),
                    timing: sibling_timing
                },
            ]
        );
        if let TaskOutcome::Indefinite { reason } = sibling_outcome {
            assert_eq!(attempted.indefinite.len(), 1);
            assert_eq!(attempted.indefinite[0].reason, Some(reason));
        } else {
            assert_eq!(attempted.solved, vec![sel(&["root", "skeleton"])]);
        }
        assert_eq!(attempted.proof_status(), ProofStatus::Inconclusive);
        assert_eq!(attempted.rolled_back.len(), 1);
        assert_eq!(attempted.rolled_back[0].selector, root_selector());

        // A later report has no failed attempts, but the restored root still
        // needs proof, regardless of the discarded sibling's result.
        let completion = execute_script(&mut tree, &[]).unwrap();
        assert!(completion.attempts.is_empty());
        assert!(completion.failed.is_empty());
        assert!(!completion.is_complete());
        assert_eq!(completion.proof_status(), ProofStatus::Inconclusive);
        assert_eq!(completion.unresolved, vec![root_selector()]);
    }

    #[test]
    fn report_deserialization_requires_explicit_completion_information() {
        // An old empty-script report records no attempts, even though its root
        // still needs proof. It must not decode as a completed proof.
        let mut encoded = serde_json::json!({
            "solved": [],
            "indefinite": [],
            "failed": [],
            "rolled_back": []
        });
        let error = serde_json::from_value::<ScriptReport>(encoded.clone()).unwrap_err();
        assert_eq!(error.to_string(), "missing field `unresolved`");

        encoded["unresolved"] = serde_json::json!([["root"]]);
        let error = serde_json::from_value::<ScriptReport>(encoded.clone()).unwrap_err();
        assert_eq!(error.to_string(), "missing field `root_disproved`");
        encoded["root_disproved"] = serde_json::json!(false);
        let error = serde_json::from_value::<ScriptReport>(encoded.clone()).unwrap_err();
        assert_eq!(error.to_string(), "missing field `has_execution_error`");
        encoded["has_execution_error"] = serde_json::json!(false);
        let pending = serde_json::from_value::<ScriptReport>(encoded.clone()).unwrap();
        assert!(!pending.is_complete());
        assert_eq!(pending.unresolved, vec![root_selector()]);

        encoded["unresolved"] = serde_json::json!([]);
        let complete = serde_json::from_value::<ScriptReport>(encoded).unwrap();
        assert!(complete.is_complete());
        assert!(complete.timing.is_none());
    }

    #[test]
    fn standalone_report_roundtrip_preserves_timing() {
        let encoded = serde_json::json!({
            "timing": { "wall_micros": 53 },
            "attempts": [{
                "selector": ["root"],
                "timing": {
                    "queued_micros": 3, "worker_wall_micros": 29,
                    "worker_timing": { "wall_micros": 17 }
                }
            }],
            "solved": [["root"]], "failed": [], "indefinite": [],
            "rolled_back": [], "unresolved": [],
            "root_disproved": false, "has_execution_error": false
        });
        let report: ScriptReport = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(report.timing, Some(Duration::from_micros(53).into()));
        assert_eq!(serde_json::to_value(report).unwrap(), encoded);
    }

    #[test]
    fn standalone_report_timing_survives_buffered_deserialization() {
        /// A consumer's result envelope with one variant per report kind.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum SavedResult {
            Tactic(ScriptReport),
        }

        /// A consumer that places the tactic fields alongside its own fields.
        #[derive(Deserialize)]
        struct SavedRun {
            #[serde(flatten)]
            report: ScriptReport,
        }

        let encoded = r#"{
            "timing":{"wall_micros":8},
            "attempts":[{"selector":["root"],"timing":{"queued_micros":1,
                "worker_wall_micros":3,"worker_timing":{"wall_micros":2}}}],
            "solved":[["root"]],"failed":[],"indefinite":[],"rolled_back":[],
            "unresolved":[],"root_disproved":false,"has_execution_error":false
        }"#;
        let SavedResult::Tactic(untagged) = serde_json::from_str(encoded).unwrap();
        let flattened: SavedRun = serde_json::from_str(encoded).unwrap();
        let expected: serde_json::Value = serde_json::from_str(encoded).unwrap();
        for report in [untagged, flattened.report] {
            assert_eq!(report.proof_status(), ProofStatus::Proved);
            assert_eq!(serde_json::to_value(report).unwrap(), expected);
        }
    }

    #[test]
    fn standalone_report_reads_conclusion_with_malformed_telemetry() {
        let mut encoded = serde_json::json!({
            "timing": { "wall_micros": -1 },
            "attempts": "unavailable",
            "solved": [], "failed": [], "indefinite": [],
            "rolled_back": [], "unresolved": [["root"]],
            "root_disproved": true, "has_execution_error": false
        });
        let report: ScriptReport = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(report.proof_status(), ProofStatus::Disproved);
        assert_eq!(report.unresolved, vec![root_selector()]);
        assert!(report.timing.is_none());
        assert!(report.attempts.is_empty());

        encoded["root_disproved"] = serde_json::json!("true");
        assert!(serde_json::from_value::<ScriptReport>(encoded).is_err());
    }

    #[test]
    fn standalone_report_keeps_valid_attempt_measurements() {
        let encoded = serde_json::json!({
            "timing": { "wall_micros": 19, "phases": [7] },
            "attempts": [
                { "selector": ["root", "a"], "timing": {
                    "queued_micros": 1,
                    "worker_timing": { "wall_micros": 7, "phases": { "solver_check_count": -1 } }
                }},
                { "selector": 17 },
                { "selector": ["root", "b"], "timing": {
                    "queued_micros": "unknown", "worker_wall_micros": 2,
                    "worker_timing": { "wall_micros": -1 }
                }}
            ],
            "solved": [["root"]], "failed": [], "indefinite": [],
            "rolled_back": [], "unresolved": [],
            "root_disproved": false, "has_execution_error": false
        });
        let report: ScriptReport = serde_json::from_value(encoded).unwrap();
        assert_eq!(report.proof_status(), ProofStatus::Proved);
        assert_eq!(
            serde_json::to_value(report.timing).unwrap(),
            serde_json::json!({"wall_micros":19})
        );
        assert_eq!(
            serde_json::to_value(report.attempts).unwrap(),
            serde_json::json!([
                { "selector": ["root", "a"], "timing": {
                    "queued_micros": 1, "worker_timing": { "wall_micros": 7 }
                }},
                { "selector": ["root", "b"], "timing": { "worker_wall_micros": 2 } }
            ])
        );
    }

    #[test]
    fn missing_task_outcome_keeps_leaf_unresolved() {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let report = ProverReport {
            wall_micros: None,
            plan: ProverReportNode::Task {
                timing: Default::default(),
                task_id: Some("root".to_string()),
                cmdline: None,
                outcome: None,
                stdout: None,
                stderr: None,
            },
        };
        let completion = execute_script_with_runner(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Solve,
            }],
            |_, _| Ok(report),
        )
        .unwrap();
        assert_eq!(tree.root.status, NodeStatus::Pending(None));
        assert!(completion.failed.is_empty());
        assert!(completion.indefinite.is_empty());
        assert!(!completion.is_complete());
        assert_eq!(completion.unresolved, vec![root_selector()]);
    }

    #[test]
    fn empty_script_does_not_invoke_runner() {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let completion = execute_script_with_runner(&mut tree, &[], |_, _| {
            panic!("an empty selection must not invoke the proof runner")
        })
        .unwrap();
        let timing = completion.timing.as_ref().unwrap();
        assert_eq!(
            serde_json::to_value(&completion).unwrap(),
            serde_json::json!({
                "timing": timing,
                "attempts": [],
                "solved": [],
                "failed": [],
                "indefinite": [],
                "rolled_back": [],
                "unresolved": [["root"]],
                "root_disproved": false, "has_execution_error": false
            })
        );
    }

    #[test_case(&[Some("root"), Some("root")], "proof runner returned duplicate task ID: root"; "duplicate")]
    #[test_case(&[Some("root"), Some("foreign")], "proof runner returned unexpected task ID: foreign"; "unexpected")]
    #[test_case(&[Some("root"), None], "proof runner returned a task without an ID"; "absent")]
    fn malformed_runner_ids_do_not_commit_outcomes(ids: &[Option<&str>], expected: &str) {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let error = execute_script_with_runner(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Solve,
            }],
            |_, _| {
                Ok(ProverReport {
                    wall_micros: None,
                    plan: ProverReportNode::Group {
                        kind: GroupKind::All,
                        outcome: Some(TaskOutcome::Normal { success: true }),
                        tasks: ids
                            .iter()
                            .enumerate()
                            .map(|(index, id)| {
                                ProverReportNode::Group {
                                    kind: GroupKind::All,
                                    outcome: Some(TaskOutcome::Normal { success: true }),
                                    tasks: vec![ProverReportNode::Task {
                                        timing: Default::default(),
                                        task_id: id.map(str::to_owned),
                                        cmdline: None,
                                        // Reserve IDs even without an outcome.
                                        outcome: (index == 0)
                                            .then_some(TaskOutcome::Normal { success: true }),
                                        stdout: None,
                                        stderr: None,
                                    }],
                                }
                            })
                            .collect(),
                    },
                })
            },
        )
        .unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(tree.root.status, NodeStatus::Open);
        let retried = execute_script_with_runner(&mut tree, &[], |obligations, config| {
            assert_eq!(obligations.len(), 1);
            assert_eq!(obligations[0].0, "root");
            accept_all_obligations(obligations, config)
        })
        .unwrap();
        assert!(retried.is_complete());
        assert_eq!(retried.solved, vec![root_selector()]);
    }

    #[test]
    fn unassigned_group_error_leaves_selected_task_queued_for_retry() {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let error = execute_script_with_runner(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Solve,
            }],
            |_, _| {
                Ok(ProverReport {
                    wall_micros: None,
                    plan: ProverReportNode::Group {
                        kind: GroupKind::All,
                        outcome: Some(ProofStatus::ExecutionError.into()),
                        tasks: vec![ProverReportNode::Task {
                            timing: Default::default(),
                            task_id: Some("root".into()),
                            outcome: None,
                            cmdline: None,
                            stdout: None,
                            stderr: None,
                        }],
                    },
                })
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            "proof runner group error cannot be assigned to a selected task"
        );
        assert_eq!(tree.root.status, NodeStatus::Open);
        let retried = execute_script_with_runner(&mut tree, &[], accept_all_obligations).unwrap();
        assert_eq!(retried.solved, vec![root_selector()]);
        assert!(retried.is_complete());
    }

    #[test_case(Some(TaskOutcome::Normal { success: true }), ProofStatus::Proved; "success")]
    #[test_case(Some(TaskOutcome::Normal { success: false }), ProofStatus::Disproved; "failure")]
    #[test_case(Some(TaskOutcome::Indefinite { reason: IndefiniteReason::Timeout }), ProofStatus::Inconclusive; "timeout")]
    #[test_case(Some(TaskOutcome::Indefinite { reason: IndefiniteReason::ExecutionError }), ProofStatus::ExecutionError; "execution_error")]
    #[test_case(Some(TaskOutcome::Indefinite { reason: IndefiniteReason::ProtocolError }), ProofStatus::ExecutionError; "protocol_error")]
    #[test_case(Some(TaskOutcome::Indefinite { reason: IndefiniteReason::ProcessFailure }), ProofStatus::ExecutionError; "process_failure")]
    #[test_case(None, ProofStatus::Inconclusive; "missing")]
    fn accepted_results_consume_selections_until_explicit_solve(
        outcome: Option<TaskOutcome>,
        expected_status: ProofStatus,
    ) {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let solve = [ScriptStep {
            selector: root_selector(),
            command: Command::Solve,
        }];
        let first = execute_script_with_runner(&mut tree, &solve, |_, _| {
            Ok(ProverReport {
                wall_micros: None,
                plan: ProverReportNode::Group {
                    kind: GroupKind::All,
                    outcome,
                    tasks: outcome
                        .into_iter()
                        .map(|outcome| ProverReportNode::Task {
                            timing: Default::default(),
                            task_id: Some("root".to_string()),
                            cmdline: None,
                            outcome: Some(outcome),
                            stdout: None,
                            stderr: None,
                        })
                        .collect(),
                },
            })
        })
        .unwrap();
        assert_eq!(first.proof_status(), expected_status);

        let unchanged = execute_script_with_runner(&mut tree, &[], |_, _| {
            panic!("accepted requests must not be resubmitted by an empty followup")
        })
        .unwrap();
        assert_eq!(unchanged.is_complete(), first.is_complete());
        assert_eq!(unchanged.proof_status(), first.proof_status());
        let roundtrip: ScriptReport =
            serde_json::from_value(serde_json::to_value(&unchanged).unwrap()).unwrap();
        assert_eq!(roundtrip.proof_status(), first.proof_status());
        assert_eq!(unchanged.unresolved, first.unresolved);
        assert!(unchanged.solved.is_empty());
        assert!(unchanged.failed.is_empty());
        assert!(unchanged.indefinite.is_empty());
        assert!(unchanged.attempts.is_empty());

        let mut reran = false;
        let retried = execute_script_with_runner(&mut tree, &solve, |obligations, config| {
            reran = true;
            assert_eq!(obligations.len(), 1);
            assert_eq!(obligations[0].0, "root");
            accept_all_obligations(obligations, config)
        })
        .unwrap();
        assert!(reran);
        assert!(retried.is_complete());
        assert_eq!(retried.solved, vec![root_selector()]);
    }

    #[test]
    fn runner_error_preserves_selection_for_retry() {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let error = execute_script_with_runner(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Solve,
            }],
            |_, _| Err("runner unavailable".to_string()),
        )
        .unwrap_err();
        assert_eq!(error, "runner unavailable");
        let retried = execute_script_with_runner(&mut tree, &[], |obligations, config| {
            assert_eq!(obligations.len(), 1);
            assert_eq!(obligations[0].0, "root");
            accept_all_obligations(obligations, config)
        })
        .unwrap();
        assert!(retried.is_complete());
        assert_eq!(retried.solved, vec![root_selector()]);
    }

    #[test]
    fn missing_task_result_keeps_leaf_pending_despite_batch_success() {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let execution = run_script(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Solve,
            }],
            |obligations, _| {
                assert_eq!(obligations.len(), 1);
                assert_eq!(obligations[0].0, "root");
                Ok(ProverReport {
                    wall_micros: None,
                    plan: ProverReportNode::Group {
                        kind: GroupKind::All,
                        outcome: Some(TaskOutcome::Normal { success: true }),
                        tasks: vec![],
                    },
                })
            },
        )
        .unwrap();
        assert_eq!(tree.root.status, NodeStatus::Pending(None));
        assert_eq!(
            execution.task_report.as_ref().unwrap().plan.proof_status(),
            ProofStatus::Proved
        );
        let timing = execution.report.timing.as_ref().unwrap();
        assert_eq!(
            serde_json::to_value(&execution).unwrap(),
            serde_json::json!({
                "status": "inconclusive", "success": false,
                "timing": { "wall_micros": timing.wall_micros },
                "report": {
                    "timing": { "wall_micros": timing.wall_micros },
                    "attempts": [],
                    "unresolved": [["root"]], "root_disproved": false, "has_execution_error": false,
                    "solved": [], "indefinite": [], "failed": [], "rolled_back": []
                }
            })
        );
    }

    #[test_case(false; "failed")]
    #[test_case(true; "indefinite")]
    fn unsuccessful_task_captures_diagnostics(indefinite: bool) {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let completion = execute_script_with_runner(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Solve,
            }],
            |_, _| {
                Ok(ProverReport {
                    wall_micros: None,
                    plan: ProverReportNode::Task {
                        timing: Default::default(),
                        task_id: Some("root".to_string()),
                        cmdline: None,
                        outcome: Some(if indefinite {
                            TaskOutcome::Indefinite {
                                reason: IndefiniteReason::Timeout,
                            }
                        } else {
                            TaskOutcome::Normal { success: false }
                        }),
                        stdout: Some("proof output".to_string()),
                        stderr: Some("proof diagnostic".to_string()),
                    },
                })
            },
        )
        .unwrap();
        assert!(!completion.is_complete());
        let mut log = serde_json::json!([{
            "selector": ["root"],
            "stdout": "proof output",
            "stderr": "proof diagnostic"
        }]);
        if indefinite {
            log[0]["reason"] = serde_json::json!("Timeout");
        }
        let timing = completion.timing.as_ref().unwrap();
        assert_eq!(
            serde_json::to_value(&completion).unwrap(),
            serde_json::json!({
                "timing": timing,
                "attempts": [{"selector": ["root"]}],
                "solved": [],
                "failed": if indefinite { serde_json::json!([]) } else { log.clone() },
                "indefinite": if indefinite { log } else { serde_json::json!([]) },
                "rolled_back": [],
                "unresolved": [["root"]],
                "root_disproved": !indefinite, "has_execution_error": false
            })
        );
    }

    #[test]
    fn repeated_completion_reports_follow_new_children_of_solved_leaf() {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let initial = execute_script_with_runner(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Solve,
            }],
            accept_all_obligations,
        )
        .unwrap();
        assert!(initial.is_complete());
        for _ in 0..2 {
            let completion = execute_script_with_runner(&mut tree, &[], |_, _| {
                panic!("completed leaves must not be resubmitted")
            })
            .unwrap();
            assert!(completion.is_complete());
            assert!(completion.solved.is_empty());
        }

        let completion = execute_script(
            &mut tree,
            &[ScriptStep {
                selector: root_selector(),
                command: Command::Apply(Tactic::Focus(FocusTactic { pairs: vec![] })),
            }],
        )
        .unwrap();
        assert!(!completion.is_complete());
        assert_eq!(completion.unresolved, vec![sel(&["root", "skeleton"])]);

        let child = OblTree::find_mut(&mut tree.root, &sel(&["root", "skeleton"])).unwrap();
        child.status = NodeStatus::Solved;
        assert!(execute_script(&mut tree, &[]).unwrap().is_complete());
    }

    #[test]
    #[cfg(feature = "has-bitwuzla")]
    fn cosliced_then_solve_all_succeeds() {
        let mut config = base_config();
        config.solver = Some(SolverChoice::Bitwuzla);
        let mut tree = OblTree::new(base_obligation(), config);
        let s1 = vec![
            NamedSlice {
                func_name: "f_slice_1".to_string(),
                code: SourceFile::Text("pub fn f_slice_1(x: u8) -> u8 { x + u8:1 }".to_string()),
            },
            NamedSlice {
                func_name: "f_slice_2".to_string(),
                code: SourceFile::Text("pub fn f_slice_2(x: u8) -> u8 { x + u8:2 }".to_string()),
            },
        ];
        let s2 = vec![
            NamedSlice {
                func_name: "f_slice_1".to_string(),
                code: SourceFile::Text("pub fn f_slice_1(x: u8) -> u8 { x + u8:1 }".to_string()),
            },
            NamedSlice {
                func_name: "f_slice_2".to_string(),
                code: SourceFile::Text("pub fn f_slice_2(x: u8) -> u8 { x + u8:2 }".to_string()),
            },
        ];
        let c1 = NamedSlice {
            func_name: "f_composed".to_string(),
            code: SourceFile::Text(
                "pub fn f_composed(x: u8) -> u8 { f_slice_2(f_slice_1(x)) }".to_string(),
            ),
        };
        let c2 = NamedSlice {
            func_name: "f_composed".to_string(),
            code: SourceFile::Text(
                "pub fn f_composed(x: u8) -> u8 { f_slice_2(f_slice_1(x)) }".to_string(),
            ),
        };
        let mut steps = Vec::new();
        steps.push(ScriptStep {
            selector: root_selector(),
            command: Command::Apply(Tactic::Cosliced(CoslicedTactic {
                lhs_slices: s1,
                rhs_slices: s2,
                lhs_composed: c1,
                rhs_composed: c2,
            })),
        });
        // Solve all leaves produced by cosliced (including skeleton)
        for seg in ["lhs_self", "rhs_self", "slice_1", "slice_2", "skeleton"] {
            steps.push(ScriptStep {
                selector: sel(&["root", seg]).clone(),
                command: Command::Solve,
            });
        }

        let report = execute_script(&mut tree, &steps).expect("execute_script");
        assert!(report.is_complete(), "incomplete proof: {report:?}");
        // Gather statuses from the tree.
        /// Records proof statuses without treating open internal nodes as
        /// leaves.
        fn walk(
            n: &OblNode,
            solved: &mut Vec<Selector>,
            failed: &mut Vec<Selector>,
            pending: &mut Vec<Selector>,
        ) {
            match n.status {
                NodeStatus::Solved => solved.push(n.selector.clone()),
                NodeStatus::Failed => failed.push(n.selector.clone()),
                NodeStatus::Pending(_) => pending.push(n.selector.clone()),
                NodeStatus::Open => {
                    // Internal nodes retain their original status after their
                    // children are proved; only leaf completion matters.
                }
            }
            if let NodeKind::Internal { children, .. } = &n.kind {
                for c in children {
                    walk(c, solved, failed, pending);
                }
            }
        }
        let mut solved = Vec::new();
        let mut failed = Vec::new();
        let mut pending = Vec::new();
        walk(&tree.root, &mut solved, &mut failed, &mut pending);
        assert!(failed.is_empty(), "some tasks failed: {:?}", failed);
        assert!(pending.is_empty(), "some tasks pending: {:?}", pending);
        let solved_paths: std::collections::HashSet<String> =
            solved.iter().map(|s| selector_to_path(s)).collect();
        for seg in ["lhs_self", "rhs_self", "slice_1", "slice_2", "skeleton"] {
            assert!(
                solved_paths.contains(&format!("root/{}", seg)),
                "missing solved {}",
                seg
            );
        }
    }
}
