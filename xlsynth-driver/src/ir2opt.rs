// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth::IrPackage;

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_opt_main;

fn ir2opt_with_toolchain(input_file: &std::path::Path, top: &str, tool_path: &str) {
    log::info!("ir2opt");
    let output = run_opt_main(input_file, Some(top), tool_path);
    println!("{}", output);
}

fn ir2opt_with_runtime(ir_package: &IrPackage, top: &str) {
    log::info!("ir2opt");
    let optimized_ir = xlsynth::optimize_ir(ir_package, top).unwrap();
    println!("{}", optimized_ir.to_string());
}

fn read_ir_text(input_path: &std::path::Path, input_file: &str) -> String {
    std::fs::read_to_string(input_path).unwrap_or_else(|err| {
        let err_msg = err.to_string();
        report_cli_error_and_exit(
            "failed to read IR input",
            Some("ir2opt"),
            vec![("path", input_file), ("error", err_msg.as_str())],
        )
    })
}

pub fn handle_ir2opt(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir2opt");
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());
    let provided_top = matches.get_one::<String>("ir_top").cloned();
    match tool_path {
        Some(tool_path) => {
            let top = if let Some(top) = provided_top {
                Some(top)
            } else {
                let ir_text = read_ir_text(input_path, input_file);
                let filename = input_path.file_name().and_then(|name| name.to_str());
                let ir_package = IrPackage::parse_ir(&ir_text, filename).unwrap();
                ir_package.top_name().unwrap_or_else(|err| {
                    let err_msg = err.to_string();
                    report_cli_error_and_exit(
                        "failed to resolve top name",
                        Some("ir2opt"),
                        vec![("path", input_file), ("error", err_msg.as_str())],
                    )
                })
            };
            let top = top.unwrap_or_else(|| {
                report_cli_error_and_exit(
                    "no top specified and no top found in IR",
                    Some("ir2opt"),
                    vec![("path", input_file)],
                )
            });
            ir2opt_with_toolchain(input_path, &top, tool_path);
        }
        None => {
            let ir_text = read_ir_text(input_path, input_file);
            let filename = input_path.file_name().and_then(|name| name.to_str());
            let ir_package = IrPackage::parse_ir(&ir_text, filename).unwrap();
            let top = provided_top
                .or_else(|| {
                    ir_package.top_name().unwrap_or_else(|err| {
                        let err_msg = err.to_string();
                        report_cli_error_and_exit(
                            "failed to resolve top name",
                            Some("ir2opt"),
                            vec![("path", input_file), ("error", err_msg.as_str())],
                        )
                    })
                })
                .unwrap_or_else(|| {
                    report_cli_error_and_exit(
                        "no top specified and no top found in IR",
                        Some("ir2opt"),
                        vec![("path", input_file)],
                    )
                });
            ir2opt_with_runtime(&ir_package, &top);
        }
    }
}
