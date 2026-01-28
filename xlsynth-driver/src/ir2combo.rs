// SPDX-License-Identifier: Apache-2.0

//! IR to combinational SystemVerilog code-gen command implementation.
//! This mirrors `ir2pipeline` but requests the `combinational` generator
//! in `codegen_main` instead of the pipeline generator.

use clap::ArgMatches;
use std::process;

use crate::common::{extract_codegen_flags, CodegenFlags};
use crate::toolchain_config::ToolchainConfig;
use crate::tools::{run_codegen_combinational, run_opt_main};
use xlsynth_pir::{run_aug_opt_over_ir_text, AugOptOptions};

/// Entry point invoked from `main.rs` when the `ir2combo` subcommand is used.
pub fn handle_ir2combo(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    let delay_model = matches
        .get_one::<String>("DELAY_MODEL")
        .expect("--delay_model must be specified");

    let codegen_flags = extract_codegen_flags(matches, config.as_ref());

    let keep_temps = matches.get_one::<String>("keep_temps").map(|s| s == "true");

    let optimize = matches
        .get_one::<String>("opt")
        .map(|s| s == "true")
        .unwrap_or(false);

    let aug_opt = matches
        .get_one::<String>("aug_opt")
        .map(|s| s == "true")
        .unwrap_or(false);

    let ir_top_opt = matches.get_one::<String>("ir_top");

    ir2combo(
        input_path,
        delay_model,
        &codegen_flags,
        optimize,
        aug_opt,
        ir_top_opt.map(|s| s.as_str()),
        &keep_temps,
        config,
    );
}

fn ir2combo(
    input_file: &std::path::Path,
    delay_model: &str,
    codegen_flags: &CodegenFlags,
    optimize: bool,
    aug_opt: bool,
    ir_top: Option<&str>,
    keep_temps: &Option<bool>,
    config: &Option<ToolchainConfig>,
) {
    log::info!("ir2combo");
    if aug_opt && !optimize {
        eprintln!("error: ir2combo: --aug-opt=true requires --opt=true");
        process::exit(2);
    }

    // We only support tool-path execution for now.
    let tool_path = match config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        Some(p) => p,
        None => {
            eprintln!("ir2combo requires a toolchain configuration with a `tool_path` entry");
            process::exit(1);
        }
    };

    // Temporary directory for artifacts.
    let temp_dir = tempfile::TempDir::new().unwrap();

    // If requested, optimize first.
    let ir_for_codegen_path: std::path::PathBuf = if optimize {
        let top_name = ir_top.expect("--opt requires --top to be specified");
        let opt_ir = if aug_opt {
            let input_text =
                std::fs::read_to_string(input_file).expect("IR input file should be readable");
            run_aug_opt_over_ir_text(
                &input_text,
                Some(top_name),
                AugOptOptions {
                    enable: true,
                    rounds: 1,
                    ..Default::default()
                },
            )
            .expect("aug_opt should succeed")
        } else {
            run_opt_main(input_file, Some(top_name), tool_path)
        };
        let opt_ir_path = temp_dir.path().join("opt.ir");
        std::fs::write(&opt_ir_path, &opt_ir).unwrap();
        opt_ir_path
    } else {
        input_file.to_path_buf()
    };

    let sv = run_codegen_combinational(&ir_for_codegen_path, delay_model, codegen_flags, tool_path);

    let sv_path = temp_dir.path().join("output.sv");
    std::fs::write(&sv_path, &sv).unwrap();

    if keep_temps.is_some() {
        let temp_dir_path = temp_dir.keep();
        eprintln!(
            "Combinational code generation successful. Output written to: {}",
            temp_dir_path.to_str().unwrap()
        );
    }

    println!("{}", sv);
}
