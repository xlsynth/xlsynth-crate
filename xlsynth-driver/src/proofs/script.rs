// SPDX-License-Identifier: Apache-2.0

//! Script-driven obligation tree execution with selectors and tactics.

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

use crate::proofs::obligations::ProverObligation;
use crate::proofs::plans::{ObligationPlan, build_plan_from_obligations};
use crate::proofs::tactics::{IsTactic, Tactic};
use crate::prover::{ProverReport, ProverReportNode, TaskOutcome, run_prover_plan};
use serde::{Deserialize, Serialize};
use xlsynth_prover::prover::SolverChoice;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeStatus {
    Open,
    Pending,
    Solved,
    Failed,
}

#[derive(Debug)]
pub enum NodeKind {
    Leaf(ProverObligation),
    Internal {
        #[allow(dead_code)]
        tactic: Tactic,
        children: Vec<OblNode>,
        original: ProverObligation,
    },
}

#[derive(Debug)]
pub struct OblNode {
    #[allow(dead_code)]
    pub segment: String,
    pub selector: Selector,
    pub kind: NodeKind,
    pub status: NodeStatus,
    pub solve: bool,
}

impl OblNode {
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

#[derive(Debug)]
pub struct OblTreeConfig {
    pub dslx_stdlib_path: Option<PathBuf>,
    pub dslx_paths: Vec<PathBuf>,
    pub solver: Option<SolverChoice>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug)]
pub struct OblTree {
    pub root: OblNode,
    pub config: OblTreeConfig,
}

impl OblTree {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Command {
    Apply(Tactic),
    Solve,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScriptStep {
    pub selector: Selector,
    pub command: Command,
}

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

/// Collects unproved leaves from the surviving tree in deterministic tree
/// order.
fn collect_unresolved(node: &OblNode, out: &mut Vec<Selector>) {
    match &node.kind {
        NodeKind::Leaf(_) => {
            if node.status != NodeStatus::Solved {
                out.push(node.selector.clone());
            }
        }
        NodeKind::Internal { children, .. } => {
            for child in children {
                collect_unresolved(child, out);
            }
        }
    }
}

#[derive(Clone, Debug)]
enum TaskSimple {
    Success,
    Failure,
    Indefinite,
}

fn collect_outcomes(report: &ProverReportNode) -> HashMap<String, TaskSimple> {
    let mut out: HashMap<String, TaskSimple> = HashMap::new();
    match report {
        ProverReportNode::Task {
            task_id, outcome, ..
        } => {
            if let (Some(tid), Some(outcome_ref)) = (task_id, outcome) {
                let simple = match outcome_ref {
                    TaskOutcome::Normal { success: true } => TaskSimple::Success,
                    TaskOutcome::Normal { success: false } => TaskSimple::Failure,
                    TaskOutcome::Indefinite { .. } => TaskSimple::Indefinite,
                };
                out.insert(tid.clone(), simple);
            }
        }
        ProverReportNode::Group { tasks, .. } => {
            for t in tasks {
                out.extend(collect_outcomes(t));
            }
        }
    }
    out
}

fn update_from_outcome(
    tree: &mut OblTree,
    report: &ProverReport,
    sel: Selector,
    path: &str,
    status: Option<TaskSimple>,
    solved: &mut Vec<Selector>,
    failed_logs: &mut Vec<TaskLogEntry>,
    indefinite_logs: &mut Vec<TaskLogEntry>,
    rolled_back: &mut Vec<RollbackEntry>,
) {
    match status {
        Some(TaskSimple::Success) => {
            if let Some(n) = OblTree::find_mut(&mut tree.root, &sel) {
                n.status = NodeStatus::Solved;
            }
            let _ = report.plan.find_task_node(path);
            solved.push(sel);
        }
        Some(TaskSimple::Failure) => {
            if let Some(n) = OblTree::find_mut(&mut tree.root, &sel) {
                n.status = NodeStatus::Failed;
            }
            if let Some(task_node) = report.plan.find_task_node(path) {
                if let ProverReportNode::Task {
                    task_id,
                    cmdline,
                    stdout,
                    stderr,
                    ..
                } = task_node
                {
                    println!(
                        "FAILED TASK id={:?}\ncmd={}\nstdout=\n{}\nstderr=\n{}\n",
                        task_id,
                        cmdline.as_deref().unwrap_or(""),
                        stdout.as_deref().unwrap_or(""),
                        stderr.as_deref().unwrap_or("")
                    );
                    failed_logs.push(TaskLogEntry {
                        selector: sel.clone(),
                        stdout: stdout.clone(),
                        stderr: stderr.clone(),
                    });
                }
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
                    rolled_back.push(RollbackEntry {
                        selector: parent_sel,
                        reason_selector: sel,
                    });
                }
            }
        }
        Some(TaskSimple::Indefinite) => {
            if let Some(n) = OblTree::find_mut(&mut tree.root, &sel) {
                n.status = NodeStatus::Pending;
            }
            if let Some(task_node) = report.plan.find_task_node(path) {
                if let ProverReportNode::Task { stdout, stderr, .. } = task_node {
                    indefinite_logs.push(TaskLogEntry {
                        selector: sel,
                        stdout: stdout.clone(),
                        stderr: stderr.clone(),
                    });
                }
            }
        }
        None => {
            if let Some(n) = OblTree::find_mut(&mut tree.root, &sel) {
                n.status = NodeStatus::Pending;
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskLogEntry {
    pub selector: Selector,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollbackEntry {
    pub selector: Selector,
    pub reason_selector: Selector,
}

/// Attempt outcomes and unresolved obligations after all tactic rollbacks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScriptReport {
    pub solved: Vec<Selector>,
    pub indefinite: Vec<TaskLogEntry>,
    pub failed: Vec<TaskLogEntry>,
    pub rolled_back: Vec<RollbackEntry>,
    /// Final leaves that are not proved, including unselected and reopened
    /// leaves.
    pub unresolved: Vec<Selector>,
}

impl ScriptReport {
    /// Whether the entire proof is discharged without failed or indefinite
    /// tasks.
    pub fn is_complete(&self) -> bool {
        self.unresolved.is_empty() && self.failed.is_empty() && self.indefinite.is_empty()
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

/// Reads the raw script text from a path ("-" for stdin) with contextualized
/// errors.
fn read_script_text_from_path(path: &str, kind: &str) -> Result<String, String> {
    if path == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read tactic script ({} ) from stdin: {}", kind, e))?;
        Ok(buf)
    } else {
        std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read tactic script ({}) {}: {}", kind, path, e))
    }
}

/// Reads and parses a tactic script from a path ("-" for stdin) as JSON array.
pub fn read_script_steps_from_json_path(path: &str) -> Result<Vec<ScriptStep>, String> {
    let text = read_script_text_from_path(path, "json")?;
    parse_script_steps_from_json_str(&text)
}

/// Reads and parses a tactic script from a path ("-" for stdin) as JSONL.
pub fn read_script_steps_from_jsonl_path(path: &str) -> Result<Vec<ScriptStep>, String> {
    let text = read_script_text_from_path(path, "jsonl")?;
    parse_script_steps_from_jsonl_str(&text)
}

/// Executes selected proofs and reports completion against the final obligation
/// tree.
pub fn execute_script(tree: &mut OblTree, steps: &[ScriptStep]) -> Result<ScriptReport, String> {
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
    for (sel, ob) in targets.iter() {
        let path = selector_to_path(sel);
        obligations_with_ids.push((path, ob.clone()));
    }

    log::debug!("obligations_to_solve={}", obligations_with_ids.len());

    if obligations_with_ids.is_empty() {
        let mut unresolved = Vec::new();
        collect_unresolved(&tree.root, &mut unresolved);
        return Ok(ScriptReport {
            solved: vec![],
            indefinite: vec![],
            failed: vec![],
            rolled_back: vec![],
            unresolved,
        });
    }

    // No special-case for skeleton-only; always execute the plan.

    // 3) Build plan and run.
    let stdlib = tree.config.dslx_stdlib_path.as_deref();
    let paths: Vec<&std::path::Path> = tree.config.dslx_paths.iter().map(|p| p.as_path()).collect();
    let plan_bundle: ObligationPlan = build_plan_from_obligations(
        &obligations_with_ids,
        stdlib,
        &paths,
        tree.config.solver.clone(),
        tree.config.timeout_ms,
    )?;
    let (plan, tempdir) = plan_bundle.into_parts();
    let _keep_tempdir = tempdir;
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let report: ProverReport = run_prover_plan(plan, cores).map_err(|e| e.to_string())?;

    // 4) Map outcomes by task_id path.
    let outcomes: HashMap<String, TaskSimple> = collect_outcomes(&report.plan);

    // 5) Update node statuses and perform rollbacks.
    let mut solved: Vec<Selector> = Vec::new();
    let mut failed_logs: Vec<TaskLogEntry> = Vec::new();
    let mut indefinite_logs: Vec<TaskLogEntry> = Vec::new();
    let mut rolled_back: Vec<RollbackEntry> = Vec::new();

    for (sel, _) in targets.into_iter() {
        let path = selector_to_path(&sel);
        let status = outcomes.get(&path).cloned();
        update_from_outcome(
            tree,
            &report,
            sel,
            &path,
            status,
            &mut solved,
            &mut failed_logs,
            &mut indefinite_logs,
            &mut rolled_back,
        );
    }

    // Outcomes for discarded children remain useful attempt history, but only
    // the surviving tree determines which obligations still need proof.
    let mut unresolved = Vec::new();
    collect_unresolved(&tree.root, &mut unresolved);
    Ok(ScriptReport {
        solved,
        indefinite: indefinite_logs,
        failed: failed_logs,
        rolled_back,
        unresolved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::obligations::{
        FileWithHistory, LecObligation, LecSide, ObligationPayload, ProverObligation, SourceFile,
    };
    #[cfg(feature = "has-bitwuzla")]
    use crate::proofs::tactics::cosliced::{CoslicedTactic, NamedSlice};
    use crate::proofs::tactics::focus::{FocusPair, FocusTactic};
    use crate::prover_config::GroupKind;
    #[cfg(feature = "has-bitwuzla")]
    use test_case::test_case;

    fn sel(parts: &[&str]) -> Selector {
        parts.iter().map(|s| s.to_string()).collect()
    }

    fn base_obligation() -> ProverObligation {
        let dslx = "pub fn f(x: u8) -> u8 { x + u8:3 }".to_string();
        let lhs = LecSide {
            top_func: "f".to_string(),
            uf_map: std::collections::BTreeMap::new(),
            file: FileWithHistory {
                base_source: SourceFile::Text(dslx.clone()),
                edits: Vec::new(),
                text: dslx.clone(),
            },
        };
        let rhs = LecSide {
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
            payload: ObligationPayload::Lec(LecObligation { lhs, rhs }),
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

        let complete = execute_script(
            &mut tree,
            &[ScriptStep {
                selector: sel(&["root", remaining]),
                command: Command::Solve,
            }],
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

    #[test]
    fn rollback_stays_incomplete_after_discarded_sibling_succeeds() {
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
        let report = ProverReport {
            success: false,
            plan: ProverReportNode::Group {
                kind: GroupKind::All,
                outcome: Some(TaskOutcome::Normal { success: false }),
                tasks: [("root/pair_1", false), ("root/skeleton", true)]
                    .into_iter()
                    .map(|(path, success)| ProverReportNode::Task {
                        task_id: Some(path.to_string()),
                        cmdline: None,
                        outcome: Some(TaskOutcome::Normal { success }),
                        stdout: None,
                        stderr: None,
                    })
                    .collect(),
            },
        };
        let outcomes = collect_outcomes(&report.plan);
        let mut solved = Vec::new();
        let mut failed = Vec::new();
        let mut indefinite = Vec::new();
        let mut rolled_back = Vec::new();
        for segment in ["pair_1", "skeleton"] {
            let path = format!("root/{segment}");
            update_from_outcome(
                &mut tree,
                &report,
                sel(&["root", segment]),
                &path,
                outcomes.get(&path).cloned(),
                &mut solved,
                &mut failed,
                &mut indefinite,
                &mut rolled_back,
            );
        }
        assert_eq!(solved, vec![sel(&["root", "skeleton"])]);
        assert_eq!(rolled_back.len(), 1);
        assert_eq!(rolled_back[0].selector, root_selector());

        // A later report has no failed attempts, but the restored root still
        // needs proof; the discarded sibling's success cannot discharge it.
        let completion = execute_script(&mut tree, &[]).unwrap();
        assert!(completion.failed.is_empty());
        assert!(!completion.is_complete());
        assert_eq!(completion.unresolved, vec![root_selector()]);
    }

    #[test]
    fn missing_task_outcome_keeps_leaf_unresolved() {
        let mut tree = OblTree::new(base_obligation(), base_config());
        let report = ProverReport {
            success: false,
            plan: ProverReportNode::Task {
                task_id: Some("root".to_string()),
                cmdline: None,
                outcome: None,
                stdout: None,
                stderr: None,
            },
        };
        let outcomes = collect_outcomes(&report.plan);
        update_from_outcome(
            &mut tree,
            &report,
            root_selector(),
            "root",
            outcomes.get("root").cloned(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
        assert_eq!(tree.root.status, NodeStatus::Pending);
        let completion = execute_script(&mut tree, &[]).unwrap();
        assert!(completion.failed.is_empty());
        assert!(completion.indefinite.is_empty());
        assert!(!completion.is_complete());
        assert_eq!(completion.unresolved, vec![root_selector()]);
    }

    #[test]
    fn repeated_completion_reports_follow_new_children_of_solved_leaf() {
        let mut tree = OblTree::new(base_obligation(), base_config());
        tree.root.status = NodeStatus::Solved;
        for _ in 0..2 {
            let completion = execute_script(&mut tree, &[]).unwrap();
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
        // Gather statuses from the tree
        fn walk(
            n: &OblNode,
            solved: &mut Vec<Selector>,
            failed: &mut Vec<Selector>,
            pending: &mut Vec<Selector>,
        ) {
            match n.status {
                NodeStatus::Solved => solved.push(n.selector.clone()),
                NodeStatus::Failed => failed.push(n.selector.clone()),
                NodeStatus::Pending => pending.push(n.selector.clone()),
                _ => {}
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
            if !solved_paths.contains(&format!("root/{}", seg)) {
                println!("missing solved {}", seg);
            }
            assert!(
                solved_paths.contains(&format!("root/{}", seg)),
                "missing solved {}",
                seg
            );
        }
    }
}
