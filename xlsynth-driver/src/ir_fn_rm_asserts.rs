// SPDX-License-Identifier: Apache-2.0

use crate::common::write_stdout;
use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;
use xlsynth_pir::ir_fn_rm_asserts::remove_asserts_from_package;
use xlsynth_pir::ir_parser;

/// Handles the `ir-fn-rm-asserts` driver subcommand.
pub fn handle_ir_fn_rm_asserts(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");
    let target_fn = matches.get_one::<String>("ir_top").map(String::as_str);

    let file_content = std::fs::read_to_string(input_file).unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format!("Failed to read {input_file}: {e}"),
            Some("ir-fn-rm-asserts"),
            vec![],
        )
    });

    let mut parser = ir_parser::Parser::new(&file_content);
    let pkg = parser.parse_and_validate_package().unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format!("Failed to parse/validate IR package: {e}"),
            Some("ir-fn-rm-asserts"),
            vec![],
        )
    });

    let outcome = remove_asserts_from_package(&pkg, target_fn).unwrap_or_else(|e| {
        report_cli_error_and_exit(&e.to_string(), Some("ir-fn-rm-asserts"), vec![])
    });
    write_stdout(&outcome.rewritten_package.to_string());
}
