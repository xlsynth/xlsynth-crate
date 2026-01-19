// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth::IrPackage;
use xlsynth_pir::{run_aug_opt_over_ir_text, AugOptOptions};

use crate::toolchain_config::ToolchainConfig;
use crate::tools::run_opt_main;

fn ir2opt(
    input_file: &std::path::Path,
    top: &str,
    config: &Option<ToolchainConfig>,
    aug_opt: bool,
) {
    log::info!("ir2opt");

    if aug_opt {
        if config
            .as_ref()
            .and_then(|c| c.tool_path.as_deref())
            .is_some()
        {
            eprintln!("error: ir2opt: --aug-opt=true is not supported with --toolchain (external tool path) yet");
            std::process::exit(2);
        }
        let input_text = std::fs::read_to_string(input_file).unwrap();
        let out = run_aug_opt_over_ir_text(
            &input_text,
            Some(top),
            AugOptOptions {
                enable: true,
                rounds: 1,
                run_xlsynth_opt_before: true,
                run_xlsynth_opt_after: true,
            },
        )
        .unwrap();
        println!("{out}");
        return;
    }

    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        let output = run_opt_main(input_file, Some(top), tool_path);
        println!("{}", output);
    } else {
        let ir_package = IrPackage::parse_ir_from_path(input_file).unwrap();
        let optimized_ir = xlsynth::optimize_ir(&ir_package, top).unwrap();
        println!("{}", optimized_ir.to_string());
    }
}

pub fn handle_ir2opt(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_ir2opt");
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let top = matches.get_one::<String>("ir_top").unwrap();
    let input_path = std::path::Path::new(input_file);

    let aug_opt = matches
        .get_one::<String>("aug_opt")
        .is_some_and(|s| s == "true");

    ir2opt(input_path, top, config, aug_opt);
}
