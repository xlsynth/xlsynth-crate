// SPDX-License-Identifier: Apache-2.0

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;

const SUBCOMMAND: &str = "xls-ir-fn-to-z3-smtlib";

/// Emits Z3 SMT-LIB text for a selected XLS IR function.
pub fn handle_ir_fn_to_z3_smtlib(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let input_file = matches
        .get_one::<String>("ir_input_file")
        .expect("ir_input_file is required");

    let mut package = xlsynth::IrPackage::parse_ir_from_path(std::path::Path::new(input_file))
        .unwrap_or_else(|e| {
            report_cli_error_and_exit(
                &format!("Failed to parse IR package: {}", e),
                Some(SUBCOMMAND),
                vec![("ir_input_file", input_file)],
            )
        });

    if let Some(ir_top) = matches.get_one::<String>("ir_top") {
        package.set_top_by_name(ir_top).unwrap_or_else(|e| {
            report_cli_error_and_exit(
                &format!("Failed to set --top: {}", e),
                Some(SUBCOMMAND),
                vec![("ir_top", ir_top)],
            )
        });
    }

    let ir_top = matches
        .get_one::<String>("ir_top")
        .expect("--top is required by clap");
    let function = package.get_function(ir_top).unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format!("Failed to resolve function from IR package: {}", e),
            Some(SUBCOMMAND),
            vec![("ir_top", ir_top)],
        )
    });

    let smtlib = function.to_z3_smtlib().unwrap_or_else(|e| {
        report_cli_error_and_exit(
            &format!("Failed to convert function to Z3 SMT-LIB: {}", e),
            Some(SUBCOMMAND),
            vec![("ir_top", ir_top)],
        )
    });

    println!("{}", smtlib);
}
