// SPDX-License-Identifier: Apache-2.0

//! CLI presentation and compatibility exports for the library proof-plan
//! engine.

use std::io::Read;

use crate::prover_config::ProverPlan;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use log::{info, warn};

// These compatibility exports are also consumed outside the CLI binary.
#[allow(unused_imports)]
pub use xlsynth_prover::proof_plan::report::{
    IndefiniteReason, ProverReport, ProverReportNode, TaskOutcome,
};
#[allow(unused_imports)]
pub use xlsynth_prover::proof_plan::scheduler::{
    Scheduler, run_prover_plan, run_prover_plan_with_driver,
};

/// Best-effort: write a minimal error JSON if `--output_json` was requested.
/// Shape:
/// { "success": false, "error": { "message": <msg>, "exception": <exc>,
/// "stage": <stage> } }
fn write_error_json(
    output_json_path: &Option<String>,
    message: &str,
    exception: &str,
    stage: &str,
) {
    if let Some(path) = output_json_path {
        let obj = serde_json::json!({
            "success": false,
            "error": {
                "message": message,
                "exception": exception,
                "stage": stage,
            }
        });
        if let Err(write_err) = std::fs::write(path, obj.to_string()) {
            warn!(
                "prover: failed writing output_json error to {}: {}",
                path, write_err
            );
        }
    }
}

/// Helper to both emit JSON (when requested) and print CLI error then exit.
fn fail_and_exit(
    output_json_path: &Option<String>,
    stage: &str,
    message: &str,
    exception: &str,
    extra_details: &[(&str, &str)],
) -> ! {
    write_error_json(output_json_path, message, exception, stage);
    // Assemble CLI details
    let mut details: Vec<(&str, &str)> = Vec::with_capacity(2 + extra_details.len());
    details.push(("stage", stage));
    details.push(("exception", exception));
    details.extend_from_slice(extra_details);
    report_cli_error_and_exit(message, Some("prover"), details)
}

/// Implements the `prover` sub-command.
/// CLI behavior:
/// - Installs signal handlers that set a global cancel flag.
/// - Reads the plan JSON from a file or stdin.
/// - Parses and executes the plan with the requested concurrency.
/// Process exits with status 0 on success, 1 on failure.
pub fn handle_prover(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let cores: usize = matches
        .get_one::<String>("cores")
        .map(|s| s.parse::<usize>().unwrap_or(1))
        .unwrap_or(1)
        .max(1);
    let plan_path = matches
        .get_one::<String>("plan_json_file")
        .expect("plan_json_file arg missing");
    let output_json_path = matches
        .get_one::<String>("output_json")
        .map(|s| s.to_string());

    info!(
        "prover: starting with cores={} plan_source= {}",
        cores,
        if plan_path == "-" { "stdin" } else { plan_path }
    );

    let plan_json = if plan_path == "-" {
        let mut buf = String::new();
        if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
            fail_and_exit(
                &output_json_path,
                "read-plan",
                "Failed to read plan from stdin",
                &e.to_string(),
                &[],
            );
        }
        buf
    } else {
        match std::fs::read_to_string(plan_path) {
            Ok(s) => s,
            Err(e) => {
                fail_and_exit(
                    &output_json_path,
                    "read-plan",
                    "Failed to read plan JSON file",
                    &e.to_string(),
                    &[("path", plan_path)],
                );
            }
        }
    };

    let plan: ProverPlan = match serde_json::from_str(&plan_json) {
        Ok(p) => p,
        Err(e) => {
            fail_and_exit(
                &output_json_path,
                "parse-plan",
                "Failed to parse ProverPlan JSON",
                &e.to_string(),
                &[],
            );
        }
    };

    match run_prover_plan(plan, cores) {
        Ok(report) => {
            if let Some(path) = &output_json_path {
                // Write full report including plan tree and captured outputs;
                // do not print per-task outputs.
                let s = serde_json::to_string(&report).unwrap();
                if let Err(e) = std::fs::write(path, s) {
                    warn!("prover: failed writing output_json to {}: {}", path, e);
                }
            } else {
                // No JSON requested; print all task outputs together at the end
                // in DFS order.
                fn print_task_outputs(node: &ProverReportNode) {
                    match node {
                        ProverReportNode::Task {
                            cmdline,
                            stdout,
                            stderr,
                            ..
                        } => {
                            println!("{}", "-".repeat(80));
                            println!("{}", cmdline.as_deref().unwrap_or(""));
                            let out = stdout.as_deref().unwrap_or("");
                            let err = stderr.as_deref().unwrap_or("");
                            if out.is_empty() {
                                println!(">>>> stdout:");
                            } else {
                                println!(">>>> stdout:\n{}", out);
                            }
                            if err.is_empty() {
                                println!(">>>> stderr:");
                            } else {
                                println!(">>>> stderr:\n{}", err);
                            }
                            println!("{}", "-".repeat(80));
                        }
                        ProverReportNode::Group { tasks, .. } => {
                            for t in tasks {
                                print_task_outputs(t);
                            }
                        }
                    }
                }
                print_task_outputs(&report.plan);
            }
            if report.success {
                println!("Overall: success");
                std::process::exit(0)
            } else {
                println!("Overall: failure");
                std::process::exit(1)
            }
        }
        Err(e) => {
            let exc = e.to_string();
            // Best-effort JSON + CLI error with context, then exit.
            fail_and_exit(&output_json_path, "run", "prover run failed", &exc, &[]);
        }
    }
}
