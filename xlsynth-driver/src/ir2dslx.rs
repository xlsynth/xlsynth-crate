// SPDX-License-Identifier: Apache-2.0

use crate::toolchain_config::ToolchainConfig;
use clap::ArgMatches;

pub fn handle_ir2dslx(matches: &ArgMatches, _config: &Option<ToolchainConfig>) {
    let ir_path = std::path::Path::new(matches.get_one::<String>("ir_input_file").unwrap());
    let top_opt = matches.get_one::<String>("ir_top").map(|s| s.as_str());
    let pkg = match xlsynth_g8r::xls_ir::ir_parser::parse_path_to_package(ir_path) {
        Ok(p) => p,
        Err(e) => {
            crate::report_cli_error::report_cli_error_and_exit(
                "Failed to parse IR",
                Some(&format!("{}", e)),
                vec![("ir_input_file", &ir_path.display().to_string())],
            );
            unreachable!();
        }
    };
    let func = if let Some(top) = top_opt {
        pkg.get_fn(top).unwrap_or_else(|| {
            crate::report_cli_error::report_cli_error_and_exit(
                "Top function not found in package",
                None,
                vec![
                    ("top", top),
                    ("ir_input_file", &ir_path.display().to_string()),
                ],
            );
            unreachable!()
        })
    } else {
        pkg.get_top().unwrap_or_else(|| {
            crate::report_cli_error::report_cli_error_and_exit(
                "No function found in package",
                None,
                vec![("ir_input_file", &ir_path.display().to_string())],
            );
            unreachable!()
        })
    };

    match xlsynth_g8r::xls_ir::ir2dslx::emit_fn_as_dslx(func) {
        Ok(text) => {
            println!("{}", text);
        }
        Err(e) => {
            crate::report_cli_error::report_cli_error_and_exit(
                "IR→DSLX emission failed",
                Some(&e),
                vec![("ir_input_file", &ir_path.display().to_string())],
            );
        }
    }
}
