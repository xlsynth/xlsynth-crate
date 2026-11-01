// SPDX-License-Identifier: Apache-2.0

//! Subprocess backend for tactic scripts using an explicitly selected driver.

use std::path::Path;

use super::obligations::ProverObligation;
use super::plans::build_plan_from_obligations;
use super::script::{OblTree, OblTreeConfig, ScriptExecution, ScriptStep, run_script};
use crate::proof_plan::config::ProofDriver;
use crate::proof_plan::report::ProverReport;
use crate::proof_plan::scheduler::run_prover_plan_with_driver;

/// Runs selected obligations and retains their temporary source files until all
/// subprocesses finish. Task output is captured in the returned report.
pub fn run_obligations(
    obligations: &[(String, ProverObligation)],
    config: &OblTreeConfig,
    driver: impl Into<ProofDriver>,
) -> Result<ProverReport, String> {
    let paths: Vec<&Path> = config.dslx_paths.iter().map(|p| p.as_path()).collect();
    let bundle = build_plan_from_obligations(
        obligations,
        config.dslx_stdlib_path.as_deref(),
        &paths,
        config.solver.clone(),
        config.timeout_ms,
    )?;
    let (plan, _sources) = bundle.into_parts();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    run_prover_plan_with_driver(plan, cores, driver).map_err(|e| e.to_string())
}

/// Executes a tactic script and returns its conclusion and captured task
/// diagnostics.
pub fn execute_script(
    tree: &mut OblTree,
    steps: &[ScriptStep],
    driver: impl Into<ProofDriver>,
) -> Result<ScriptExecution, String> {
    run_script(tree, steps, |obligations, config| {
        run_obligations(obligations, config, driver)
    })
}
