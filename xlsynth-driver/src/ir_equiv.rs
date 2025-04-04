// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::report_cli_error::report_cli_error_and_exit;
use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_check_ir_equivalence_main;

const SUBCOMMAND: &str = "ir-equiv";

fn ir_equiv(
    lhs: &std::path::Path,
    rhs: &std::path::Path,
    top: Option<&str>,
    config: &Option<ToolchainConfig>,
) {
    log::info!("ir-equiv");
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_check_ir_equivalence_main(lhs, rhs, top, tool_path);
        match output {
            Ok(stdout) => {
                println!("success: {}", stdout);
            }
            Err(output) => {
                // Note: the details of the counterexample come on stdout, not stderr.
                let mut message = String::from_utf8_lossy(&output.stdout);
                if message.is_empty() {
                    // Try for stderr if stdout was empty, e.g. in case of other kinds of errors.
                    message = String::from_utf8_lossy(&output.stderr);
                }
                report_cli_error_and_exit(
                    &format!("failure: {}", message),
                    Some(SUBCOMMAND),
                    vec![
                        ("lhs_ir_file", lhs.to_str().unwrap()),
                        ("rhs_ir_file", rhs.to_str().unwrap()),
                        (
                            "stdout",
                            &format!("{:?}", String::from_utf8_lossy(&output.stdout)),
                        ),
                        (
                            "stderr",
                            &format!("{:?}", String::from_utf8_lossy(&output.stderr)),
                        ),
                    ],
                );
            }
        }
    } else {
        report_cli_error_and_exit(
            "a tool path was not provided (e.g. via a xlsynth-toolchain.toml file)",
            Some(SUBCOMMAND),
            vec![("toolchain-config", format!("{:?}", config).as_str())],
        );
    }
}

pub fn handle_ir_equiv(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir_equiv");
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();
    let top = if let Some(top) = matches.get_one::<String>("ir_top") {
        Some(top.as_str())
    } else {
        None
    };
    let lhs_path = std::path::Path::new(lhs);
    let rhs_path = std::path::Path::new(rhs);

    ir_equiv(lhs_path, rhs_path, top, config);
}
