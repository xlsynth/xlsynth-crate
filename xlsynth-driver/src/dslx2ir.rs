// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;
use xlsynth::{DslxConvertOptions, IrPackage};

use crate::{
    common::{parse_bool_flag_or, resolve_type_inference_v2},
    toolchain_config::{get_dslx_path, get_dslx_stdlib_path, ToolchainConfig},
    tools::{run_ir_converter_main, run_opt_main},
};

fn dslx2ir(
    input_file: &std::path::Path,
    dslx_top: Option<&str>,
    dslx_stdlib_path: Option<&str>,
    dslx_path: Option<&str>,
    tool_path: Option<&str>,
    enable_warnings: Option<&[String]>,
    disable_warnings: Option<&[String]>,
    type_inference_v2: Option<bool>,
    opt: bool,
    convert_tests: bool,
) {
    log::info!("dslx2ir");

    let dslx_module_name = input_file.file_stem().unwrap().to_str().unwrap();

    if let Some(tool_path) = tool_path {
        let mut output = run_ir_converter_main(
            input_file,
            dslx_top,
            dslx_stdlib_path,
            dslx_path,
            tool_path,
            enable_warnings,
            disable_warnings,
            type_inference_v2,
            convert_tests,
        );
        if opt {
            // Write the output of conversion to a temp file and then pass that.
            let temp_file = tempfile::NamedTempFile::new().unwrap();
            let temp_file_path = temp_file.path();
            std::fs::write(temp_file_path, output).unwrap();
            let ir_top = xlsynth::mangle_dslx_name(dslx_module_name, dslx_top.unwrap()).unwrap();
            output = run_opt_main(temp_file_path, Some(&ir_top), tool_path);
        }
        println!("{}", output);
    } else {
        if type_inference_v2 == Some(true) {
            eprintln!("error: --type_inference_v2 is only supported when using --toolchain (external tool path)");
            std::process::exit(1);
        }
        let dslx_contents = std::fs::read_to_string(input_file).expect(&format!(
            "file read should succeed for path {:?}",
            input_file
        ));
        let dslx_stdlib_path: Option<&std::path::Path> =
            dslx_stdlib_path.map(|s| std::path::Path::new(s));
        let additional_search_paths: Vec<&std::path::Path> = dslx_path
            .map(|s| s.split(';').map(|p| std::path::Path::new(p)).collect())
            .unwrap_or_default();
        let result: xlsynth::DslxToIrTextResult = xlsynth::convert_dslx_to_ir_text(
            &dslx_contents,
            input_file,
            &DslxConvertOptions {
                dslx_stdlib_path,
                additional_search_paths,
                enable_warnings,
                disable_warnings,
                force_implicit_token_calling_convention: false,
            },
        )
        .expect("DSLX to IR conversion failed");
        for warning in result.warnings {
            log::warn!(
                "DSLX warning for {}: {}",
                input_file.to_str().unwrap(),
                warning
            );
        }

        let result_text: String = if opt {
            let ir_top = xlsynth::mangle_dslx_name(dslx_module_name, dslx_top.unwrap()).unwrap();
            let ir_package = IrPackage::parse_ir(&result.ir, Some(&ir_top)).unwrap();
            let optimized_ir_package = xlsynth::optimize_ir(&ir_package, &ir_top).unwrap();
            optimized_ir_package.to_string()
        } else {
            result.ir
        };

        println!("{}", result_text);
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

    let enable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.enable_warnings.as_deref());
    let disable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.disable_warnings.as_deref());

    let opt = parse_bool_flag_or(matches, "opt", false);

    let convert_tests = parse_bool_flag_or(
        matches,
        "convert_tests",
        crate::flag_defaults::IR_CONVERTER_CONVERT_TESTS,
    );

    if convert_tests && top.is_some() {
        crate::report_cli_error::report_cli_error_and_exit(
            "`--convert_tests=true` cannot be combined with `--dslx_top` (upstream ir_converter_main ignores tests when a top is specified). Remove `--dslx_top` or set `--convert_tests=false`.",
            Some("dslx2ir"),
            vec![
                ("dslx_top", top.unwrap()),
                ("convert_tests", "true"),
            ],
        );
    }

    let type_inference_v2 = resolve_type_inference_v2(matches, config);

    dslx2ir(
        input_path,
        top,
        dslx_stdlib_path,
        dslx_path,
        tool_path,
        enable_warnings,
        disable_warnings,
        type_inference_v2,
        opt,
        convert_tests,
    );
}
