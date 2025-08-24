// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use crate::tools::tool_path_or_exit;
use std::process;
use std::process::Command;

fn run_delay_info_main(
    input_file: &std::path::Path,
    top: Option<&str>,
    delay_model: &str,
    tool_path: &str,
) -> String {
    log::info!("run_delay_info_main");
    let delay_info_path = tool_path_or_exit(tool_path, "delay_info_main", "Delay info");
    let mut command = Command::new(delay_info_path);
    command.arg(input_file);
    command.arg("--delay_model").arg(delay_model);
    if top.is_some() {
        command.arg("--top").arg(top.unwrap());
    }

    let output = command.output().expect("delay_info_main should succeed");

    if !output.status.success() {
        eprintln!("Delay info failed with status: {}", output.status);
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    String::from_utf8_lossy(&output.stdout).to_string()
}

fn ir2delayinfo(
    input_file: &std::path::Path,
    top: &str,
    delay_model: &str,
    config: &Option<ToolchainConfig>,
) {
    log::info!("ir2delayinfo");
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_delay_info_main(input_file, Some(top), delay_model, tool_path);
        println!("{}", output);
    } else {
        todo!("ir2delayinfo subcommand using runtime APIs")
    }
}

pub fn handle_ir2delayinfo(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir2delayinfo");
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let top = matches.get_one::<String>("ir_top").unwrap();
    let input_path = std::path::Path::new(input_file);
    let delay_model = matches.get_one::<String>("DELAY_MODEL").unwrap();

    ir2delayinfo(input_path, top, delay_model, config);
}
