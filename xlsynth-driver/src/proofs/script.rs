// SPDX-License-Identifier: Apache-2.0

//! CLI input and diagnostic presentation for the library tactic engine.

use std::io::{self, Read, Write};
use std::time::Instant;

use xlsynth_prover::dslx_tactics::process::execute_script;
use xlsynth_prover::dslx_tactics::script::{
    OblTree, ScriptStep, parse_script_steps_from_json_str, parse_script_steps_from_jsonl_str,
};
use xlsynth_prover::proof_outcome::{ProofStatus, ProofSummary};
use xlsynth_prover::proof_plan::config::ProofDriver;
use xlsynth_prover::proof_plan::report::{IndefiniteReason, ProverReportNode, TaskOutcome};

/// Reads the raw script text from a path ("-" for stdin) with contextualized
/// errors.
fn read_script_text_from_path(path: &str, kind: &str) -> Result<String, String> {
    if path == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read tactic script ({kind}) from stdin: {e}"))?;
        Ok(buf)
    } else {
        std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read tactic script ({}) {}: {}", kind, path, e))
    }
}

/// A tactic script supplied to a proof command as JSON or JSONL.
pub(crate) enum TacticScript<'a> {
    Json(&'a str),
    JsonLines(&'a str),
}

impl<'a> TacticScript<'a> {
    /// Clap enforces mutual exclusion of the two formats.
    pub(crate) fn from_matches(matches: &'a clap::ArgMatches) -> Option<Self> {
        matches
            .get_one::<String>("tactic_json")
            .map(|path| Self::Json(path))
            .or_else(|| {
                matches
                    .get_one::<String>("tactic_jsonl")
                    .map(|path| Self::JsonLines(path))
            })
    }

    fn read(&self) -> Result<Vec<ScriptStep>, String> {
        match self {
            Self::Json(path) => {
                parse_script_steps_from_json_str(&read_script_text_from_path(path, "json")?)
            }
            Self::JsonLines(path) => {
                parse_script_steps_from_jsonl_str(&read_script_text_from_path(path, "jsonl")?)
            }
        }
    }

    /// Runs the library's process backend and presents the root conclusion and
    /// task diagnostics.
    pub(crate) fn execute_and_exit(
        self,
        command_start: Instant,
        tree: Result<OblTree, String>,
        subcommand: &str,
        success_message: &str,
        output_json: Option<&str>,
        driver: ProofDriver,
    ) -> ! {
        let script_start = Instant::now();
        let fail = |error: String| -> ! {
            if let Some(path) = output_json {
                let failed_at = Instant::now();
                // The legacy duration starts after preparing the obligation
                // tree.
                let outcome = ProofSummary {
                    timing: (failed_at - command_start).into(),
                    ..ProofSummary::execution_error(failed_at - script_start, error.clone())
                };
                std::fs::write(path, serde_json::to_vec(&outcome).unwrap()).unwrap();
            }
            eprintln!("[{subcommand}] {error}");
            std::process::exit(ProofStatus::ExecutionError.exit_code())
        };
        let mut tree = tree.unwrap_or_else(|error| fail(error));
        let steps = self.read().unwrap_or_else(|error| fail(error));
        let execution = execute_script(&mut tree, &steps, driver)
            .unwrap_or_else(|error| fail(format!("execute_script error: {error}")));
        let duration = script_start.elapsed();
        if let Some(batch) = &execution.task_report {
            write_task_diagnostics(&batch.plan, &mut io::stdout().lock()).unwrap_or_else(|error| {
                fail(format!("failed to write proof diagnostics: {error}"))
            });
        }
        if let Some(path) = output_json {
            std::fs::write(path, serde_json::to_vec(&execution).unwrap()).unwrap();
        }
        let status = execution.report.proof_status();
        if status.is_proved() {
            println!("[{subcommand}] Time taken: {duration:?}");
            println!("[{subcommand}] success: {success_message}");
        } else {
            eprintln!("[{subcommand}] Time taken: {duration:?}");
            eprintln!("[{subcommand}] failure");
            for selector in &execution.report.unresolved {
                eprintln!(
                    "[{subcommand}] unresolved obligation: {}",
                    selector.join("/")
                );
            }
        }
        std::process::exit(status.exit_code())
    }
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
            ..
        } => {
            match outcome {
                TaskOutcome::Normal { success: false } => {
                    writeln!(writer, "FAILED TASK id={task_id:?}")?;
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
                TaskOutcome::Indefinite { reason } => {
                    let label = if outcome.proof_status() == ProofStatus::ExecutionError {
                        "ERROR"
                    } else {
                        "INCONCLUSIVE"
                    };
                    writeln!(writer, "{label} TASK id={task_id:?} reason={reason:?}")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_prover::proof_plan::config::GroupKind;
    use xlsynth_test_helpers::compare_golden_text;

    #[test]
    fn tactic_diagnostics_preserve_errors_and_omit_canceled_tasks() {
        let mut tasks = vec![ProverReportNode::Task {
            timing: Default::default(),
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
                timing: Default::default(),
                task_id: Some(format!("{reason:?}")),
                cmdline: Some("driver prove-property".into()),
                outcome: Some(TaskOutcome::Indefinite { reason }),
                stdout: Some("captured output".into()),
                stderr: Some("captured diagnostic".into()),
            });
        }
        for outcome in [None, Some(TaskOutcome::Normal { success: true })] {
            tasks.push(ProverReportNode::Task {
                timing: Default::default(),
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
