// SPDX-License-Identifier: Apache-2.0

//! CLI input and diagnostic presentation for the library tactic engine.

use std::io::Read;

use xlsynth_prover::dslx_tactics::process::run_obligations;
pub use xlsynth_prover::dslx_tactics::script::*;
use xlsynth_prover::proof_plan::config::resolve_driver_exe;
use xlsynth_prover::proof_plan::report::{ProverReportNode, TaskOutcome};

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

/// Displays captured diagnostics for failed tasks in plan order.
fn print_failed_tasks(node: &ProverReportNode) {
    match node {
        ProverReportNode::Task {
            task_id,
            cmdline,
            outcome: Some(TaskOutcome::Normal { success: false }),
            stdout,
            stderr,
        } => {
            println!(
                "FAILED TASK id={:?}\ncmd={}\nstdout=\n{}\nstderr=\n{}\n",
                task_id,
                cmdline.as_deref().unwrap_or(""),
                stdout.as_deref().unwrap_or(""),
                stderr.as_deref().unwrap_or("")
            );
        }
        ProverReportNode::Group { tasks, .. } => {
            for task in tasks {
                print_failed_tasks(task);
            }
        }
        ProverReportNode::Task { .. } => {
            // Successful, indefinite, and unstarted tasks have no failure log.
        }
    }
}

/// Calls the library engine and presents captured task failures for the CLI.
pub fn execute_script(tree: &mut OblTree, steps: &[ScriptStep]) -> Result<ScriptReport, String> {
    let driver_exe = resolve_driver_exe();
    execute_script_with_runner(tree, steps, |obligations, config| {
        let report = run_obligations(obligations, config, &driver_exe)?;
        print_failed_tasks(&report.plan);
        Ok(report)
    })
}
