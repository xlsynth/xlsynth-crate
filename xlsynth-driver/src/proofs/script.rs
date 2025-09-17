// SPDX-License-Identifier: Apache-2.0

//! Script-driven obligation tree execution with selectors and tactics.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::PathBuf;

use crate::proofs::obligations::LecObligation;
use crate::proofs::plans::{build_plan_from_obligations, ObligationPlan};
#[cfg(test)]
use crate::proofs::tactics::cosliced::{CoslicedTactic, NamedSlice};
use crate::proofs::tactics::{IsTactic, Tactic};
use crate::prover::{run_prover_plan, ProverReport, ProverReportNode, TaskOutcome};
use crate::solver_choice::SolverChoice;
use serde::{Deserialize, Serialize};

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
    Leaf(LecObligation),
    Internal {
        tactic: Tactic,
        children: Vec<OblNode>,
        original: LecObligation,
    },
}

#[derive(Debug)]
pub struct OblNode {
    pub segment: String,
    pub selector: Selector,
    pub kind: NodeKind,
    pub status: NodeStatus,
    pub solve: bool,
}

impl OblNode {
    pub fn leaf(seg: String, parent: &Selector, ob: LecObligation) -> Self {
        Self {
            segment: seg.clone(),
            selector: join_selector(parent, &seg),
            kind: NodeKind::Leaf(ob),
            status: NodeStatus::Open,
            solve: false,
        }
    }
}

pub struct OblTreeConfig {
    pub dslx_stdlib_path: Option<PathBuf>,
    pub dslx_paths: Vec<PathBuf>,
    pub solver: Option<SolverChoice>,
    pub timeout_ms: Option<u64>,
}

pub struct OblTree {
    pub root: OblNode,
    pub config: OblTreeConfig,
}

impl OblTree {
    pub fn new(root_obligation: LecObligation, config: OblTreeConfig) -> Self {
        let mut ob = root_obligation;
        // Ensure root obligation has selector segment = "root"
        ob.selector_segment = "root".to_string();
        let root = OblNode {
            segment: "root".to_string(),
            selector: root_selector(),
            kind: NodeKind::Leaf(ob),
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
    let mut segs: HashSet<String> = HashSet::new();
    let mut children: Vec<OblNode> = Vec::new();
    for (idx, ob) in children_obs.into_iter().enumerate() {
        let parent_last = parent_selector
            .last()
            .cloned()
            .unwrap_or_else(|| "".to_string());
        let mut seg = if ob.selector_segment.is_empty() || ob.selector_segment == parent_last {
            "skeleton".to_string()
        } else {
            ob.selector_segment.clone()
        };
        if seg == parent_last || seg.is_empty() {
            seg = format!("child_{}", idx + 1);
        }
        if !segs.insert(seg.clone()) {
            return Err(format!("duplicate child selector segment: {}", seg));
        }
        children.push(OblNode::leaf(seg, parent_selector, ob));
    }
    node.kind = NodeKind::Internal {
        tactic,
        children,
        original: base,
    };
    Ok(())
}

fn collect_solve_targets(node: &OblNode, out: &mut Vec<(Selector, LecObligation)>) {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScriptReport {
    pub solved: Vec<Selector>,
    pub indefinite: Vec<TaskLogEntry>,
    pub failed: Vec<TaskLogEntry>,
    pub rolled_back: Vec<RollbackEntry>,
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
    let mut targets: Vec<(Selector, LecObligation)> = Vec::new();
    collect_solve_targets(&tree.root, &mut targets);
    let obligations_with_ids: Vec<(String, LecObligation)> = targets
        .iter()
        .map(|(sel, ob)| (selector_to_path(sel), ob.clone()))
        .collect();

    log::debug!("obligations_to_solve={}", obligations_with_ids.len());

    if obligations_with_ids.is_empty() {
        return Ok(ScriptReport {
            solved: vec![],
            indefinite: vec![],
            failed: vec![],
            rolled_back: vec![],
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

    for (sel, _ob) in targets.into_iter() {
        let path = selector_to_path(&sel);
        match outcomes.get(&path) {
            Some(TaskSimple::Success) => {
                if let Some(n) = OblTree::find_mut(&mut tree.root, &sel) {
                    n.status = NodeStatus::Solved;
                }
                let _ = report.plan.find_task_node(&path);
                solved.push(sel.clone());
            }
            Some(TaskSimple::Failure) => {
                if let Some(n) = OblTree::find_mut(&mut tree.root, &sel) {
                    n.status = NodeStatus::Failed;
                }
                // Print failure details if available
                let tid_path = selector_to_path(&sel);
                if let Some(task_node) = report.plan.find_task_node(&tid_path) {
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

                // Rollback parent tactic
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
                            reason_selector: sel.clone(),
                        });
                    }
                }
            }
            Some(TaskSimple::Indefinite) => {
                if let Some(n) = OblTree::find_mut(&mut tree.root, &sel) {
                    n.status = NodeStatus::Pending;
                }
                if let Some(task_node) = report.plan.find_task_node(&path) {
                    if let ProverReportNode::Task { stdout, stderr, .. } = task_node {
                        indefinite_logs.push(TaskLogEntry {
                            selector: sel.clone(),
                            stdout: stdout.clone(),
                            stderr: stderr.clone(),
                        });
                    }
                }
            }
            None => {
                // No outcome available; mark pending conservatively.
                if let Some(n) = OblTree::find_mut(&mut tree.root, &sel) {
                    n.status = NodeStatus::Pending;
                }
            }
        }
    }

    Ok(ScriptReport {
        solved,
        indefinite: indefinite_logs,
        failed: failed_logs,
        rolled_back,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::obligations::SourceFile;

    fn sel(parts: &[&str]) -> Selector {
        parts.iter().map(|s| s.to_string()).collect()
    }

    fn base_obligation() -> LecObligation {
        use crate::proofs::obligations::{FileWithHistory, LecSide, SourceFile};
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
        LecObligation {
            selector_segment: "root".to_string(),
            lhs,
            rhs,
            description: None,
        }
    }

    fn base_config() -> OblTreeConfig {
        let mut cfg = OblTreeConfig {
            dslx_stdlib_path: None,
            dslx_paths: vec![],
            solver: None,
            timeout_ms: Some(5000),
        };
        #[cfg(any(feature = "with-bitwuzla-built", feature = "with-bitwuzla-system"))]
        {
            cfg.solver = Some(crate::solver_choice::SolverChoice::Bitwuzla);
        }
        cfg
    }

    #[cfg(any(feature = "with-bitwuzla-built", feature = "with-bitwuzla-system"))]
    #[test]
    fn cosliced_then_solve_all_succeeds() {
        let mut tree = OblTree::new(base_obligation(), base_config());
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

        execute_script(&mut tree, &steps).expect("execute_script");
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
