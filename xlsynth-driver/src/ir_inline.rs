// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use std::path::PathBuf;
use xlsynth_pir::{run_ir_inline_over_ir_text, IrInlineBackend, IrInlineOptions};

use crate::toolchain_config::ToolchainConfig;

pub fn handle_ir_inline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let top = matches.get_one::<String>("ir_top").map(|s| s.as_str());
    let unroll = matches
        .get_one::<String>("unroll")
        .map(|s| s == "true")
        .unwrap_or(true);

    let tool_path = match config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        Some(tool_path) => tool_path,
        None => {
            eprintln!(
                "error: ir-inline currently requires --toolchain (external tool path); runtime backend is not implemented yet"
            );
            std::process::exit(2);
        }
    };

    let input_text = std::fs::read_to_string(input_file).unwrap_or_else(|e| {
        eprintln!("error: ir-inline: failed to read {}: {}", input_file, e);
        std::process::exit(1);
    });

    let output_text = run_ir_inline_over_ir_text(
        &input_text,
        top,
        IrInlineOptions { unroll },
        IrInlineBackend::Toolchain(PathBuf::from(tool_path)),
    )
    .unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });

    println!("{output_text}");
}
