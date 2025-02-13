// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_opt_main;

fn ir2opt(input_file: &std::path::Path, top: &str, config: &Option<ToolchainConfig>) {
    log::info!("ir2opt");
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_opt_main(input_file, Some(top), tool_path);
        println!("{}", output);
    } else {
        todo!("ir2opt subcommand using runtime APIs")
    }
}

pub fn handle_ir2opt(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir2opt");
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let top = matches.get_one::<String>("ir_top").unwrap();
    let input_path = std::path::Path::new(input_file);

    ir2opt(input_path, top, config);
}
