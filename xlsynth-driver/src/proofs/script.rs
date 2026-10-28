// SPDX-License-Identifier: Apache-2.0

//! CLI input and diagnostic presentation for the library tactic engine.

use std::io::{self, Read, Write};

use xlsynth_prover::dslx_tactics::process::run_obligations;
pub use xlsynth_prover::dslx_tactics::script::*;
use xlsynth_prover::proof_plan::config::resolve_driver_exe;
use xlsynth_prover::proof_plan::report::{IndefiniteReason, ProverReportNode, TaskOutcome};

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

/// Displays unsuccessful proof attempts in plan order, omitting canceled tasks.
fn write_task_diagnostics(node: &ProverReportNode, writer: &mut impl Write) -> io::Result<()> {
    match node {
        ProverReportNode::Task {
            task_id,
            cmdline,
            outcome: Some(outcome),
            stdout,
            stderr,
        } => {
            match outcome {
                TaskOutcome::Normal { success: false } => {
                    writeln!(writer, "FAILED TASK id={task_id:?}")?;
                }
                TaskOutcome::Indefinite {
                    reason:
                        reason @ (IndefiniteReason::ExecutionError
                        | IndefiniteReason::ProtocolError
                        | IndefiniteReason::ProcessFailure),
                } => {
                    writeln!(writer, "ERROR TASK id={task_id:?} reason={reason:?}")?;
                }
                TaskOutcome::Indefinite {
                    reason: reason @ (IndefiniteReason::Inconclusive | IndefiniteReason::Timeout),
                } => {
                    writeln!(writer, "INCONCLUSIVE TASK id={task_id:?} reason={reason:?}")?;
                }
                TaskOutcome::Normal { success: true }
                | TaskOutcome::Indefinite {
                    reason:
                        IndefiniteReason::GroupCriteriaAlreadyMet
                        | IndefiniteReason::IndefiniteChildren
                        | IndefiniteReason::Cleanup,
                } => {
                    // Successful or canceled tasks need no failure diagnostic.
                    return Ok(());
                }
            }
            writeln!(
                writer,
                "cmd={}\nstdout=\n{}\nstderr=\n{}\n",
                cmdline.as_deref().unwrap_or(""),
                stdout.as_deref().unwrap_or(""),
                stderr.as_deref().unwrap_or("")
            )?;
        }
        ProverReportNode::Group { tasks, .. } => {
            for task in tasks {
                write_task_diagnostics(task, writer)?;
            }
        }
        ProverReportNode::Task { .. } => {
            // Unstarted tasks have no proof attempt to diagnose.
        }
    }
    Ok(())
}

/// Calls the library engine and presents captured task failures for the CLI.
pub fn execute_script(tree: &mut OblTree, steps: &[ScriptStep]) -> Result<ScriptReport, String> {
    let driver_exe = resolve_driver_exe();
    execute_script_with_runner(tree, steps, |obligations, config| {
        let report = run_obligations(obligations, config, &driver_exe)?;
        write_task_diagnostics(&report.plan, &mut io::stdout().lock())
            .map_err(|error| format!("failed to write proof diagnostics: {error}"))?;
        Ok(report)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_prover::proof_plan::config::GroupKind;
    use xlsynth_test_helpers::compare_golden_text;

    #[test]
    fn tactic_diagnostics_preserve_errors_and_omit_canceled_tasks() {
        let mut tasks = vec![ProverReportNode::Task {
            task_id: Some("counterexample".into()),
            cmdline: Some("driver prove-counterexample".into()),
            outcome: Some(TaskOutcome::Normal { success: false }),
            stdout: Some("counterexample output".into()),
            stderr: Some("counterexample diagnostic".into()),
        }];
        for reason in [
            IndefiniteReason::ExecutionError,
            IndefiniteReason::ProtocolError,
            IndefiniteReason::ProcessFailure,
            IndefiniteReason::Inconclusive,
            IndefiniteReason::Timeout,
            IndefiniteReason::GroupCriteriaAlreadyMet,
            IndefiniteReason::IndefiniteChildren,
            IndefiniteReason::Cleanup,
        ] {
            tasks.push(ProverReportNode::Task {
                task_id: Some(format!("{reason:?}")),
                cmdline: Some("driver prove-property".into()),
                outcome: Some(TaskOutcome::Indefinite { reason }),
                stdout: Some("captured output".into()),
                stderr: Some("captured diagnostic".into()),
            });
        }
        for outcome in [None, Some(TaskOutcome::Normal { success: true })] {
            tasks.push(ProverReportNode::Task {
                task_id: Some("quiet".into()),
                cmdline: None,
                outcome,
                stdout: Some("must not display".into()),
                stderr: Some("must not display".into()),
            });
        }
        let report = ProverReportNode::Group {
            kind: GroupKind::All,
            outcome: Some(TaskOutcome::Indefinite {
                reason: IndefiniteReason::IndefiniteChildren,
            }),
            tasks,
        };
        let mut output = Vec::new();
        write_task_diagnostics(&report, &mut output).unwrap();
        compare_golden_text(
            &String::from_utf8(output).unwrap(),
            "src/testdata/tactic_task_diagnostics.golden.txt",
        );
    }
}
