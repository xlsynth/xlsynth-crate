// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Serialize;
use xlsynth_prover::dslx_assertions::{DslxAssertionsRequest, run_dslx_assertions};
use xlsynth_prover::proof_outcome::ProofStatus;
use xlsynth_prover::prover::SolverChoice;
use xlsynth_prover::prover::types::{BoolPropertyResult, FnInput, FnOutput};

use crate::common::{parse_bool_flag_or, parse_uf_spec};
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::{ToolchainConfig, get_dslx_path, get_dslx_stdlib_path};

const SUBCOMMAND: &str = "dslx-fn-prove-assertions";
const IMPLICIT_TOKEN_RUNTIME_PARAM_COUNT: usize = 2;

#[derive(Debug, Serialize)]
struct SerializedInput {
    name: String,
    value: String,
}

#[derive(Debug, Serialize)]
struct SerializedOutput {
    value: String,
    assertion_label: Option<String>,
    assertion_message: Option<String>,
}

#[derive(Debug, Serialize)]
struct SerializedCounterexample {
    inputs: Vec<SerializedInput>,
    output: SerializedOutput,
}

/// CLI representation of an assertion proof with any counterexample or error.
#[derive(Debug, Serialize)]
struct AssertionsOutcome {
    status: ProofStatus,
    success: bool,
    time_micros: u128,
    counterexample: Option<SerializedCounterexample>,
    error_str: Option<String>,
}

fn collect_additional_search_paths(dslx_path: Option<&str>) -> Vec<PathBuf> {
    dslx_path
        .map(|s| {
            s.split(';')
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default()
}

fn serialize_inputs(inputs: &[FnInput]) -> Vec<SerializedInput> {
    inputs
        .iter()
        // The synthetic property function always uses the standard
        // implicit-token ABI: param 0 is token, param 1 is activation.
        .skip(IMPLICIT_TOKEN_RUNTIME_PARAM_COUNT)
        .map(|input| SerializedInput {
            name: input.name.clone(),
            value: format!("{:?}", input.value),
        })
        .collect()
}

fn serialize_output(output: &FnOutput) -> SerializedOutput {
    let (assertion_label, assertion_message) = output
        .assertion_violation
        .as_ref()
        .map(|vio| (Some(vio.label.clone()), Some(vio.message.clone())))
        .unwrap_or((None, None));
    SerializedOutput {
        value: format!("{:?}", output.value),
        assertion_label,
        assertion_message,
    }
}

fn write_json(path: &str, outcome: &AssertionsOutcome) {
    std::fs::write(path, serde_json::to_string_pretty(outcome).unwrap()).unwrap_or_else(|err| {
        report_cli_error_and_exit(
            &format!("Failed to write JSON output: {}", err),
            Some(SUBCOMMAND),
            vec![("path", path)],
        );
    });
}

fn emit_outcome_and_exit(outcome: AssertionsOutcome, output_json: Option<&String>) -> ! {
    if let Some(path) = output_json {
        write_json(path, &outcome);
    }

    let duration = std::time::Duration::from_micros(outcome.time_micros as u64);
    if outcome.success {
        println!("[{}] Time taken: {:?}", SUBCOMMAND, duration);
        println!("[{}] success: All selected assertions proved", SUBCOMMAND);
        std::process::exit(outcome.status.exit_code());
    }

    eprintln!("[{}] Time taken: {:?}", SUBCOMMAND, duration);
    if let Some(counterexample) = outcome.counterexample.as_ref() {
        eprintln!("[{}] failure: Found assertion counterexample", SUBCOMMAND);
        for input in &counterexample.inputs {
            eprintln!("  {} = {}", input.name, input.value);
        }
        if let Some(label) = counterexample.output.assertion_label.as_ref() {
            let message = counterexample
                .output
                .assertion_message
                .as_deref()
                .unwrap_or("");
            eprintln!("  violated assertion label '{}': {}", label, message);
        }
    } else if let Some(error) = outcome.error_str.as_ref() {
        eprintln!("[{}] failure: {}", SUBCOMMAND, error);
    } else {
        eprintln!("[{}] failure", SUBCOMMAND);
    }
    std::process::exit(outcome.status.exit_code());
}

/// Keeps execution errors and inconclusive results separate from
/// counterexamples.
fn outcome_from_result(result: BoolPropertyResult, duration: Duration) -> AssertionsOutcome {
    let status = ProofStatus::from(&result);
    let (counterexample, error_str) = match result {
        BoolPropertyResult::Proved => (None, None),
        BoolPropertyResult::Disproved { inputs, output } => (
            Some(SerializedCounterexample {
                inputs: serialize_inputs(&inputs),
                output: serialize_output(&output),
            }),
            None,
        ),
        BoolPropertyResult::Inconclusive(msg)
        | BoolPropertyResult::ToolchainDisproved(msg)
        | BoolPropertyResult::Error(msg) => (None, Some(msg)),
    };
    AssertionsOutcome {
        status,
        success: status.is_proved(),
        time_micros: duration.as_micros(),
        counterexample,
        error_str,
    }
}

/// Implements the `dslx-fn-prove-assertions` sub-command.
pub fn handle_dslx_fn_prove_assertions(
    matches: &clap::ArgMatches,
    config: &Option<ToolchainConfig>,
) {
    let input_file = matches
        .get_one::<String>("dslx_input_file")
        .expect("dslx_input_file arg missing");
    let dslx_top = matches
        .get_one::<String>("dslx_top")
        .expect("dslx_top arg missing");
    let input_path = Path::new(input_file);

    let solver = matches
        .get_one::<String>("solver")
        .map(|s| s.parse().unwrap())
        .unwrap_or(SolverChoice::Bitwuzla);
    if solver == SolverChoice::Toolchain {
        report_cli_error_and_exit(
            "Solver 'toolchain' is not supported for this command",
            Some(SUBCOMMAND),
            vec![("solver", "toolchain")],
        );
    }

    let source = std::fs::read_to_string(input_path).unwrap_or_else(|err| {
        report_cli_error_and_exit(
            &format!("Failed to read DSLX file: {}", err),
            Some(SUBCOMMAND),
            vec![("file", input_file.as_str())],
        );
    });

    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path_buf = dslx_stdlib_path.as_deref().map(PathBuf::from);
    let dslx_path = get_dslx_path(matches, config);
    let additional_search_paths = collect_additional_search_paths(dslx_path.as_deref());
    let assert_label_filter = matches
        .get_one::<String>("assert_label_filter")
        .map(String::as_str);
    let assume_enum_in_bound = parse_bool_flag_or(matches, "assume-enum-in-bound", true);
    let output_json = matches.get_one::<String>("output_json");

    let module_name = input_path.file_stem().unwrap().to_str().unwrap();
    let uf_map: HashMap<String, String> =
        parse_uf_spec(module_name, matches.get_many::<String>("uf"));

    let request = DslxAssertionsRequest {
        source: &source,
        path: input_path,
        top: dslx_top,
        solver: Some(solver),
        dslx_stdlib_path: dslx_stdlib_path_buf.as_deref(),
        additional_search_paths,
        assert_label_filter,
        assume_enum_in_bound,
        uf_map,
    };

    let start = Instant::now();
    let outcome = match run_dslx_assertions(&request) {
        Ok(report) => outcome_from_result(report.result, report.duration),
        Err(err) => outcome_from_result(BoolPropertyResult::Error(err), start.elapsed()),
    };

    emit_outcome_and_exit(outcome, output_json);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assertion_json_does_not_conflate_errors_with_counterexamples() {
        for (result, status, error) in [
            (BoolPropertyResult::Proved, "proved", None),
            (
                BoolPropertyResult::Inconclusive("unknown".into()),
                "inconclusive",
                Some("unknown"),
            ),
            (
                BoolPropertyResult::Error("solver failed".into()),
                "execution-error",
                Some("solver failed"),
            ),
            (
                BoolPropertyResult::ToolchainDisproved("tool failed".into()),
                "execution-error",
                Some("tool failed"),
            ),
        ] {
            let outcome = outcome_from_result(result, Duration::ZERO);
            assert_eq!(
                serde_json::to_value(outcome).unwrap(),
                serde_json::json!({
                    "time_micros": 0,
                    "status": status,
                    "success": status == "proved",
                    "counterexample": null,
                    "error_str": error,
                }),
            );
        }
    }
}
