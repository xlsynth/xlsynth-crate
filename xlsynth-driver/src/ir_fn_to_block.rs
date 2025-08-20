// SPDX-License-Identifier: Apache-2.0

//! IR function-to-block IR conversion using external `codegen_main`.

use clap::ArgMatches;

use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_codegen_block_ir_to_string;

pub fn handle_ir_fn_to_block(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    let top = matches
        .get_one::<String>("ir_top")
        .expect("--top must be specified");

    // This subcommand currently only supports execution via external toolchain.
    let tool_path = match config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        Some(p) => p,
        None => {
            eprintln!("ir-fn-to-block requires a toolchain configuration with a `tool_path` entry");
            std::process::exit(1);
        }
    };

    // Temporary directory for artifacts.
    let temp_dir = tempfile::TempDir::new().unwrap();
    let block_ir_path = temp_dir.path().join("output.block.ir");

    let block_ir = run_codegen_block_ir_to_string(input_path, top, tool_path, &block_ir_path);

    // Note: If we later add a --keep_temps flag, we could `temp_dir.keep()` here.
    println!("{}", block_ir);
}
