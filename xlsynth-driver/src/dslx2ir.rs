// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth::DslxConvertOptions;

use crate::{
    toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig},
    tools::run_ir_converter_main,
};

fn dslx2ir(
    input_file: &std::path::Path,
    dslx_top: Option<&str>,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    tool_path: Option<&str>,
    enable_warnings: Option<&[String]>,
    disable_warnings: Option<&[String]>,
) {
    log::info!("dslx2ir");
    if let Some(tool_path) = tool_path {
        let output = run_ir_converter_main(
            input_file,
            dslx_top,
            dslx_stdlib_path,
            dslx_path,
            tool_path,
            enable_warnings,
            disable_warnings,
        );
        println!("{}", output);
    } else {
        let dslx_contents = std::fs::read_to_string(input_file).expect("file read successful");
        let dslx_stdlib_path: Option<&std::path::Path> =
            dslx_stdlib_path.map(|s| std::path::Path::new(s));
        let additional_search_paths: Vec<&std::path::Path> = dslx_path
            .map(|s| s.split(';').map(|p| std::path::Path::new(p)).collect())
            .unwrap_or_default();
        let result = xlsynth::convert_dslx_to_ir_text(
            &dslx_contents,
            input_file,
            &DslxConvertOptions {
                dslx_stdlib_path,
                additional_search_paths,
                enable_warnings,
                disable_warnings,
            },
        )
        .expect("successful conversion");
        for warning in result.warnings {
            log::warn!(
                "DSLX warning for {}: {}",
                input_file.to_str().unwrap(),
                warning
            );
        }
        println!("{}", result.ir);
    }
}

pub fn handle_dslx2ir(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_dslx2ir");
    let input_file = matches.get_one::<String>("dslx_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    let top = if let Some(top) = matches.get_one::<String>("dslx_top") {
        Some(top.to_string())
    } else {
        None
    };
    let top = top.as_deref();
    let dslx_stdlib_path = get_dslx_stdlib_path(matches, config);
    let dslx_stdlib_path = dslx_stdlib_path.as_deref();

    let dslx_path = get_dslx_path(matches, config);
    let dslx_path = dslx_path.as_deref();

    let tool_path = config.as_ref().and_then(|c| c.tool_path.as_deref());

    let enable_warnings = config.as_ref().and_then(|c| c.enable_warnings.as_deref());
    let disable_warnings = config.as_ref().and_then(|c| c.disable_warnings.as_deref());

    // Stub function for DSLX to IR conversion
    dslx2ir(
        input_path,
        top,
        dslx_stdlib_path,
        dslx_path,
        tool_path,
        enable_warnings,
        disable_warnings,
    );
}
