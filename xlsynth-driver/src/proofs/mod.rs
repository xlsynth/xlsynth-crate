// SPDX-License-Identifier: Apache-2.0

pub mod script;

use std::time::Instant;
use xlsynth_prover::proof_outcome::ProofSummary;

/// Emits the worker protocol even if CLI request construction fails.
pub(crate) fn report_setup_error(
    start: Instant,
    output_json: Option<&str>,
    subcommand: &str,
    message: &str,
) -> ! {
    if let Some(path) = output_json {
        let summary = ProofSummary::execution_error(start.elapsed(), message.to_owned());
        if let Err(error) = std::fs::write(path, serde_json::to_vec(&summary).unwrap()) {
            eprintln!("[{subcommand}] failed to write result to {path}: {error}");
        }
    }
    crate::report_cli_error::report_cli_error_and_exit(message, Some(subcommand), vec![])
}
