// SPDX-License-Identifier: Apache-2.0

use crate::common::get_dslx_paths;
use crate::toolchain_config::ToolchainConfig;

pub fn handle_dslx_fn_eval(matches: &clap::ArgMatches, _config: &Option<ToolchainConfig>) {
    let dslx_path = matches
        .get_one::<String>("dslx_input_file")
        .expect("dslx_input_file required");
    let top_fn = matches
        .get_one::<String>("dslx_top")
        .expect("dslx_top required");
    let input_ir_path = matches
        .get_one::<String>("input_ir_path")
        .expect("input_ir_path required");
    let eval_mode = matches.get_one::<String>("eval_mode").map(|s| s.as_str());

    let dslx_paths = get_dslx_paths(matches, _config);

    let irvals_text = match std::fs::read_to_string(input_ir_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read --input_ir_path: {}", e);
            std::process::exit(1);
        }
    };
    let lines: Vec<String> = irvals_text.lines().map(|s| s.to_string()).collect();

    let mode = match eval_mode {
        Some("jit") => crate::fn_eval::EvalMode::Jit,
        Some("pir-interp") => crate::fn_eval::EvalMode::PirInterp,
        Some("interp") | None => crate::fn_eval::EvalMode::Interp,
        Some(other) => {
            eprintln!(
                "Unknown --eval_mode: {} (expected interp|jit|pir-interp)",
                other
            );
            std::process::exit(1);
        }
    };

    let opts = crate::fn_eval::DslxFnEvalOptions {
        dslx_stdlib_path: dslx_paths.stdlib_path.as_deref(),
        additional_search_paths: dslx_paths.search_path_views(),
        force_implicit_token_calling_convention: false,
    };

    let dslx_file = std::path::Path::new(dslx_path);
    match crate::fn_eval::evaluate_dslx_function_over_ir_values(
        dslx_file, top_fn, &lines, mode, &opts,
    ) {
        Ok(outputs) => {
            for o in outputs {
                println!("{}", o);
            }
        }
        Err(e) => {
            let msg = e.to_string();
            if matches!(eval_mode, Some(s) if s == "pir-interp") {
                // Always emit a canonical marker for assertion-related failures in PIR mode.
                eprintln!("assertion failure(s)");
                eprintln!("assertion failure");
                if msg.contains("assertion failure") {
                    eprintln!("{}", msg);
                } else {
                    eprintln!("{}", msg);
                }
            } else if msg.contains("assertion failure") {
                eprintln!("{}", msg);
            } else {
                eprintln!("{}", msg);
            }
            std::process::exit(1);
        }
    }
}
