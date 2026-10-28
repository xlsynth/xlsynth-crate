// SPDX-License-Identifier: Apache-2.0

//! Process results must establish a logical outcome before resolving a proof.

#![cfg(unix)]

use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use tempfile::TempDir;
use test_case::test_case;
use xlsynth_prover::dslx_tactics::obligations::{
    LecObligation, LecSide, ObligationPayload, ProverObligation,
};
use xlsynth_prover::dslx_tactics::script::{
    Command, NodeKind, NodeStatus, OblTree, OblTreeConfig, ScriptStep, execute_script_with_runner,
};
use xlsynth_prover::dslx_tactics::tactics::Tactic;
use xlsynth_prover::dslx_tactics::tactics::focus::{FocusPair, FocusTactic};
use xlsynth_prover::proof_outcome::ProofStatus;
use xlsynth_prover::proof_plan::config::{GroupKind, IrEquivConfig, ProverPlan, ProverTask};
use xlsynth_prover::proof_plan::report::{IndefiniteReason, ProverReportNode, TaskOutcome};
use xlsynth_prover::proof_plan::scheduler::run_prover_plan_with_driver;

/// An executable fixture whose lifetime includes all of its scheduled tasks.
struct Backend {
    _directory: TempDir,
    path: PathBuf,
}

impl Backend {
    /// Gives each task deterministic JSON, diagnostics, and process
    /// termination.
    fn new(cases: &[(&str, Option<&str>, &str)]) -> io::Result<Self> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("backend.sh");
        let mut script = String::from(
            r#"#!/bin/sh
set -eu
mode="$2"
while [ "$#" -gt 0 ]; do
    if [ "$1" = --output_json ]; then
        shift
        output="$1"
        break
    fi
    shift
done
printf '%s\n' "$mode stdout"
printf '%s\n' "$mode stderr" >&2
case "$mode" in
"#,
        );
        for (mode, json, termination) in cases {
            script.push_str(&format!("{mode})\n"));
            if let Some(json) = json {
                script.push_str(&format!(
                    "cat > \"$output\" <<'PROOF_RESULT'\n{json}\nPROOF_RESULT\n"
                ));
            }
            script.push_str(termination);
            script.push_str("\n;;\n");
        }
        script.push_str("esac\n");
        fs::write(&path, script)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
        Ok(Self {
            _directory: directory,
            path,
        })
    }
}

/// Uses the otherwise unused input filename to select the fixture behavior.
fn task(mode: &str, timeout_ms: Option<u64>) -> ProverPlan {
    ProverPlan::Task {
        task: ProverTask::IrEquiv {
            config: IrEquivConfig {
                lhs_ir_file: PathBuf::from(mode),
                rhs_ir_file: PathBuf::from("unused.ir"),
                ..Default::default()
            },
        },
        timeout_ms,
        task_id: Some(mode.to_string()),
    }
}

/// Extracts the completed outcome while checking that the node is a leaf task.
fn task_outcome(node: &ProverReportNode) -> Option<TaskOutcome> {
    let ProverReportNode::Task { outcome, .. } = node else {
        panic!("expected a task report: {node:?}");
    };
    *outcome
}

#[test_case(r#"{"status":"proved","success":true}"#, "exit 0", true; "proof")]
#[test_case(r#"{"details":{"nested":[1,2]},"success":true,"status":"proved"}"#, "exit 0", true; "proof with diagnostics")]
#[test_case(r#"{"status":"disproved","success":false}"#, "exit 1", false; "counterexample")]
fn accepts_consistent_logical_results(json: &str, termination: &str, success: bool) {
    let backend = Backend::new(&[("result", Some(json), termination)]).unwrap();
    let report = run_prover_plan_with_driver(task("result", None), 1, &backend.path).unwrap();
    assert_eq!(report.plan.proof_status().is_proved(), success);
    assert_eq!(
        task_outcome(&report.plan),
        Some(TaskOutcome::Normal { success })
    );
}

#[test_case(Some(r#"{"status":"proved","success":true}"#), "exit 23", IndefiniteReason::ProcessFailure; "proof followed by process error")]
#[test_case(Some(r#"{"status":"proved","success":true}"#), "kill -TERM \"$$\"", IndefiniteReason::ProcessFailure; "proof followed by signal")]
#[test_case(Some(r#"{"status":"proved","success":true}"#), "exit 1", IndefiniteReason::ProcessFailure; "proof with disproof exit")]
#[test_case(Some(r#"{"status":"disproved","success":false}"#), "exit 0", IndefiniteReason::ProcessFailure; "disproof with proof exit")]
#[test_case(Some(r#"{"status":"disproved","success":false}"#), "exit 23", IndefiniteReason::ProcessFailure; "disproof followed by process error")]
#[test_case(Some(r#"{"status":"proved","success":false}"#), "exit 0", IndefiniteReason::ProtocolError; "proof contradicts success")]
#[test_case(Some(r#"{"status":"disproved","success":true}"#), "exit 1", IndefiniteReason::ProtocolError; "disproof contradicts success")]
#[test_case(Some(r#"{"status":"inconclusive","success":false}"#), "exit 1", IndefiniteReason::Inconclusive; "solver inconclusive")]
#[test_case(Some(r#"{"status":"execution-error","success":false}"#), "exit 1", IndefiniteReason::ExecutionError; "solver execution error")]
#[test_case(Some(r#"{"status":"inconclusive","success":true}"#), "exit 0", IndefiniteReason::ProtocolError; "inconclusive contradicts success")]
#[test_case(Some(r#"{"status":"execution-error","success":true}"#), "exit 0", IndefiniteReason::ProtocolError; "execution error contradicts success")]
#[test_case(Some(r#"{"status":"unrecognized","success":true}"#), "exit 0", IndefiniteReason::ProtocolError; "unknown status")]
#[test_case(Some(r#"{"status":true,"success":true}"#), "exit 0", IndefiniteReason::ProtocolError; "nonstring status")]
#[test_case(Some(r#"["proved",true]"#), "exit 0", IndefiniteReason::ProtocolError; "positional result array")]
#[test_case(Some(r#"{"status":{"proved":null},"success":true}"#), "exit 0", IndefiniteReason::ProtocolError; "tagged status object")]
#[test_case(Some(r#"{"status":"disproved","status":"proved","success":true}"#), "exit 0", IndefiniteReason::ProtocolError; "duplicate status")]
#[test_case(Some(r#"{"status":"proved","success":false,"success":true}"#), "exit 0", IndefiniteReason::ProtocolError; "duplicate success")]
#[test_case(Some(r#"{"success":true}"#), "exit 0", IndefiniteReason::ProtocolError; "absent status")]
#[test_case(Some(r#"{"status":"proved"}"#), "exit 0", IndefiniteReason::ProtocolError; "absent success")]
#[test_case(Some(r#"{"status":"proved","success":"true"}"#), "exit 0", IndefiniteReason::ProtocolError; "nonboolean success")]
#[test_case(Some("{broken json"), "exit 0", IndefiniteReason::ProtocolError; "malformed json")]
#[test_case(None, "exit 0", IndefiniteReason::ProtocolError; "missing json after normal exit")]
#[test_case(None, "exit 23", IndefiniteReason::ProcessFailure; "missing json after process error")]
fn nonlogical_process_results_remain_indefinite(
    json: Option<&str>,
    termination: &str,
    reason: IndefiniteReason,
) {
    let backend = Backend::new(&[("result", json, termination)]).unwrap();
    let report = run_prover_plan_with_driver(task("result", None), 1, &backend.path).unwrap();
    assert!(!report.plan.proof_status().is_proved());
    assert_eq!(
        task_outcome(&report.plan),
        Some(TaskOutcome::Indefinite { reason })
    );
    let ProverReportNode::Task { stdout, stderr, .. } = report.plan else {
        panic!("expected a task report");
    };
    assert_eq!(stdout.as_deref(), Some("result stdout\n"));
    // Scheduler diagnostics may follow the exact output captured from the task.
    assert!(stderr.unwrap().starts_with("result stderr\n"));
}

#[test_case(false, "proved", true; "cancel after first proof")]
#[test_case(true, "proved", true; "finish all after first proof")]
#[test_case(false, "disproved", false; "cancel after first counterexample")]
#[test_case(true, "disproved", false; "finish all after first counterexample")]
fn first_group_skips_nonlogical_results(keep_running: bool, status: &str, success: bool) {
    let logical_json = format!(r#"{{"status":"{status}","success":{success}}}"#);
    let backend = Backend::new(&[
        ("malformed", Some("{broken json"), "exit 0"),
        (
            "error",
            Some(r#"{"status":"execution-error","success":false}"#),
            "exit 1",
        ),
        (
            "inconclusive",
            Some(r#"{"status":"inconclusive","success":false}"#),
            "exit 1",
        ),
        (
            "crashed_after_proof",
            Some(r#"{"status":"proved","success":true}"#),
            "exit 23",
        ),
        (
            "logical",
            Some(&logical_json),
            if success { "exit 0" } else { "exit 1" },
        ),
    ])
    .unwrap();
    let plan = ProverPlan::Group {
        kind: GroupKind::First,
        tasks: [
            "malformed",
            "error",
            "inconclusive",
            "crashed_after_proof",
            "logical",
        ]
        .map(|mode| task(mode, None))
        .into(),
        keep_running_till_finish: keep_running,
    };
    // One core makes the error finish before the logical result without sleeps.
    let report = run_prover_plan_with_driver(plan, 1, &backend.path).unwrap();
    assert_eq!(report.plan.proof_status().is_proved(), success);
    let ProverReportNode::Group { outcome, .. } = &report.plan else {
        panic!("expected a group report");
    };
    assert_eq!(*outcome, Some(TaskOutcome::Normal { success }));
    let logical = report.plan.find_task_node("logical").unwrap();
    assert_eq!(task_outcome(logical), Some(TaskOutcome::Normal { success }));
    for mode in ["malformed", "error", "inconclusive", "crashed_after_proof"] {
        assert!(matches!(
            task_outcome(report.plan.find_task_node(mode).unwrap()),
            Some(TaskOutcome::Indefinite { .. })
        ));
    }
}

#[test_case(GroupKind::First, false; "first cancel on resolution")]
#[test_case(GroupKind::First, true; "first finish all")]
#[test_case(GroupKind::All, false; "all cancel on resolution")]
#[test_case(GroupKind::All, true; "all finish all")]
#[test_case(GroupKind::Any, false; "any cancel on resolution")]
#[test_case(GroupKind::Any, true; "any finish all")]
fn group_without_logical_results_is_indefinite(kind: GroupKind, keep_running: bool) {
    let backend = Backend::new(&[
        ("malformed", Some("{broken json"), "exit 0"),
        (
            "error",
            Some(r#"{"status":"execution-error","success":false}"#),
            "exit 1",
        ),
    ])
    .unwrap();
    let plan = ProverPlan::Group {
        kind,
        tasks: vec![task("malformed", None), task("error", None)],
        keep_running_till_finish: keep_running,
    };
    let report = run_prover_plan_with_driver(plan, 1, &backend.path).unwrap();
    assert!(!report.plan.proof_status().is_proved());
    let ProverReportNode::Group { outcome, .. } = report.plan else {
        panic!("expected a group report");
    };
    assert!(matches!(outcome, Some(TaskOutcome::Indefinite { .. })));
}

#[test]
fn timeout_preserves_captured_diagnostics() {
    // The shell writes diagnostics before waiting for the scheduler to kill it.
    let backend = Backend::new(&[("timeout", None, "while :; do :; done")]).unwrap();
    let report = run_prover_plan_with_driver(task("timeout", Some(500)), 1, &backend.path).unwrap();
    assert!(!report.plan.proof_status().is_proved());
    assert_eq!(
        task_outcome(&report.plan),
        Some(TaskOutcome::Indefinite {
            reason: IndefiniteReason::Timeout,
        })
    );
    let ProverReportNode::Task { stdout, stderr, .. } = report.plan else {
        panic!("expected a task report");
    };
    assert_eq!(stdout.as_deref(), Some("timeout stdout\n"));
    assert_eq!(stderr.as_deref(), Some("timeout stderr\n"));
}

#[test_case(false; "missing driver")]
#[test_case(true; "nonexecutable driver")]
fn launch_failure_is_an_execution_error(exists: bool) {
    let directory = tempfile::tempdir().unwrap();
    let driver = directory.path().join("backend.sh");
    if exists {
        fs::write(&driver, "#!/bin/sh\nexit 0\n").unwrap();
        fs::set_permissions(&driver, fs::Permissions::from_mode(0o644)).unwrap();
    }
    let report = run_prover_plan_with_driver(task("result", None), 1, &driver).unwrap();
    assert!(!report.plan.proof_status().is_proved());
    assert_eq!(
        task_outcome(&report.plan),
        Some(TaskOutcome::Indefinite {
            reason: IndefiniteReason::ExecutionError,
        })
    );
    let ProverReportNode::Task { stderr, .. } = report.plan else {
        panic!("expected a task report");
    };
    assert!(!stderr.expect("launch error diagnostic").is_empty());
}

#[test_case(Some("{broken json"), "exit 0", Some(IndefiniteReason::ProtocolError); "malformed output")]
#[test_case(Some(r#"{"status":"execution-error","success":false}"#), "exit 1", Some(IndefiniteReason::ExecutionError); "execution error")]
#[test_case(Some(r#"{"status":"inconclusive","success":false}"#), "exit 1", Some(IndefiniteReason::Inconclusive); "inconclusive proof")]
#[test_case(Some(r#"{"status":"proved","success":true}"#), "exit 23", Some(IndefiniteReason::ProcessFailure); "crash after proof")]
#[test_case(Some(r#"{"status":"proved","success":true}"#), "kill -TERM \"$$\"", Some(IndefiniteReason::ProcessFailure); "signal after proof")]
#[test_case(Some(r#"{"status":"disproved","success":false}"#), "exit 1", None; "counterexample control")]
fn process_outcomes_only_roll_back_actual_child_disproofs(
    json: Option<&str>,
    termination: &str,
    expected_reason: Option<IndefiniteReason>,
) {
    let backend = Backend::new(&[
        ("result", json, termination),
        (
            "proved",
            Some(r#"{"status":"proved","success":true}"#),
            "exit 0",
        ),
    ])
    .unwrap();
    let source = "fn helper(x: u8) -> u8 { x } fn main(x: u8) -> u8 { helper(x) }";
    let mut tree = OblTree::new(
        ProverObligation {
            selector_segment: "root".to_string(),
            description: None,
            payload: ObligationPayload::Lec(LecObligation {
                lhs: LecSide::from_text("main", source),
                rhs: LecSide::from_text("main", source),
                options: Default::default(),
            }),
        },
        OblTreeConfig {
            dslx_stdlib_path: None,
            dslx_paths: vec![],
            solver: None,
            timeout_ms: None,
        },
    );
    let root = vec!["root".to_string()];
    let pair = vec!["root".to_string(), "pair_1".to_string()];
    let skeleton = vec!["root".to_string(), "skeleton".to_string()];
    let steps = [
        ScriptStep {
            selector: root.clone(),
            command: Command::Apply(Tactic::Focus(FocusTactic {
                pairs: vec![FocusPair {
                    lhs: "helper".to_string(),
                    rhs: "helper".to_string(),
                }],
            })),
        },
        ScriptStep {
            selector: pair.clone(),
            command: Command::Solve,
        },
        ScriptStep {
            selector: skeleton.clone(),
            command: Command::Solve,
        },
    ];
    let report = execute_script_with_runner(&mut tree, &steps, |obligations, _config| {
        assert_eq!(
            obligations
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            vec!["root/pair_1", "root/skeleton"]
        );
        let tasks = obligations
            .iter()
            .enumerate()
            .map(|(index, (id, _))| ProverPlan::Task {
                task: ProverTask::IrEquiv {
                    config: IrEquivConfig {
                        lhs_ir_file: PathBuf::from(if index == 0 { "result" } else { "proved" }),
                        rhs_ir_file: PathBuf::from("unused.ir"),
                        ..Default::default()
                    },
                },
                timeout_ms: None,
                task_id: Some(id.clone()),
            })
            .collect();
        let plan = ProverPlan::Group {
            kind: GroupKind::All,
            tasks,
            keep_running_till_finish: true,
        };
        run_prover_plan_with_driver(plan, 1, &backend.path).map_err(|error| error.to_string())
    })
    .unwrap();
    assert!(!report.is_complete());
    if let Some(reason) = expected_reason {
        assert!(report.failed.is_empty());
        assert!(report.rolled_back.is_empty());
        assert_eq!(report.solved, vec![skeleton]);
        assert_eq!(report.unresolved, vec![pair.clone()]);
        assert_eq!(report.indefinite.len(), 1);
        assert_eq!(report.indefinite[0].selector, pair);
        assert_eq!(report.indefinite[0].reason, Some(reason));
        assert_eq!(
            report.indefinite[0].stdout.as_deref(),
            Some("result stdout\n")
        );
        assert!(
            report.indefinite[0]
                .stderr
                .as_deref()
                .unwrap()
                .starts_with("result stderr\n")
        );
        assert_eq!(
            report.proof_status(),
            TaskOutcome::Indefinite { reason }.proof_status()
        );
        let NodeKind::Internal { children, .. } = &tree.root.kind else {
            panic!("infrastructure failures must retain the decomposition");
        };
        assert_eq!(children[0].status, NodeStatus::Pending);
        assert!(!children[0].solve);
        assert_eq!(children[1].status, NodeStatus::Solved);
    } else {
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].selector, pair);
        assert!(report.indefinite.is_empty());
        assert_eq!(report.rolled_back.len(), 1);
        assert_eq!(report.rolled_back[0].selector, root.clone());
        assert_eq!(report.unresolved, vec![root]);
        assert_eq!(report.proof_status(), ProofStatus::Inconclusive);
        assert!(matches!(tree.root.kind, NodeKind::Leaf(_)));
    }
}
