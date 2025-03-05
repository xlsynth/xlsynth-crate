// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_check_ir_equivalence_main;

fn irequiv(
    lhs: &std::path::Path,
    rhs: &std::path::Path,
    top: Option<&str>,
    config: &Option<ToolchainConfig>,
) {
    log::info!("irequiv");
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_check_ir_equivalence_main(lhs, rhs, top, tool_path);
        match output {
            Ok(stdout) => {
                println!("success: {}", stdout);
            }
            Err(output) => {
                eprintln!("failure: {}", String::from_utf8_lossy(&output.stdout));
            }
        }
    } else {
        todo!(
            "irequiv subcommand using runtime APIs; no tool path found in config: {:?}",
            *config
        );
    }
}

pub fn handle_irequiv(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_irequiv");
    let lhs = matches.get_one::<String>("lhs_ir_file").unwrap();
    let rhs = matches.get_one::<String>("rhs_ir_file").unwrap();
    let top = if let Some(top) = matches.get_one::<String>("ir_top") {
        Some(top.as_str())
    } else {
        None
    };
    let lhs_path = std::path::Path::new(lhs);
    let rhs_path = std::path::Path::new(rhs);

    irequiv(lhs_path, rhs_path, top, config);
}
