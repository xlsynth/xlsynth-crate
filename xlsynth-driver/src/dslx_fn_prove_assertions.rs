// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use std::time::Instant;

use xlsynth_prover::dslx_assertions::{
    AssertionUfMappings, AssertionsOutcome, DslxAssertionsRequest,
};

use crate::common::{parse_bool_flag_or, parse_dslx_uf_spec};
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::{ToolchainConfig, get_dslx_search_paths, get_dslx_stdlib_path};

const SUBCOMMAND: &str = "dslx-fn-prove-assertions";

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

    let duration = std::time::Duration::from_micros(outcome.summary.time_micros as u64);
    if outcome.summary.conclusion.status.is_proved() {
        println!("[{}] Time taken: {:?}", SUBCOMMAND, duration);
        println!("[{}] success: All selected assertions proved", SUBCOMMAND);
        std::process::exit(outcome.summary.conclusion.status.exit_code());
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
    } else if let Some(error) = outcome.summary.error_str.as_ref() {
        eprintln!("[{}] failure: {}", SUBCOMMAND, error);
    } else {
        eprintln!("[{}] failure", SUBCOMMAND);
    }
    std::process::exit(outcome.summary.conclusion.status.exit_code());
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
        .map(|s| s.parse().unwrap());
    let output_json = matches.get_one::<String>("output_json");
    let start = Instant::now();
    let dslx_uf_map =
        parse_dslx_uf_spec(matches.get_many::<String>("uf")).unwrap_or_else(|message| {
            emit_outcome_and_exit(
                AssertionsOutcome::execution_error(start.elapsed(), message),
                output_json,
            )
        });
    let source = std::fs::read_to_string(input_path).unwrap_or_else(|err| {
        let message = format!("Failed to read DSLX file: {err}");
        if let Some(path) = output_json {
            write_json(
                path,
                &AssertionsOutcome::execution_error(start.elapsed(), message.clone()),
            );
        }
        report_cli_error_and_exit(&message, Some(SUBCOMMAND), vec![("file", input_file)]);
    });

    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path_buf = dslx_stdlib_path.as_deref().map(PathBuf::from);
    let additional_search_paths = get_dslx_search_paths(matches, config);
    let assert_label_filter = matches
        .get_one::<String>("assert_label_filter")
        .map(String::as_str);
    let assume_enum_in_bound = parse_bool_flag_or(matches, "assume-enum-in-bound", true);

    let request = DslxAssertionsRequest {
        source: &source,
        path: input_path,
        top: dslx_top,
        solver,
        dslx_stdlib_path: dslx_stdlib_path_buf.as_deref(),
        additional_search_paths,
        assert_label_filter,
        assume_enum_in_bound,
        uf_map: AssertionUfMappings::LocalDslx(dslx_uf_map),
    };

    emit_outcome_and_exit(request.prove(), output_json);
}
