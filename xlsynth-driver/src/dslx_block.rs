// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use clap::ArgMatches;
use xlsynth_block_lang::{
    BlockCompileOptions, CombinationalOptimization, ParametricBinding, apply_xls53_extern_codegen,
    compile_block_module, prepare_xls53_extern_codegen, rename_package_block,
    reorder_system_verilog_package_ports,
};
use xlsynth_pir::ir::MemberType;

use crate::common::{CodegenFlags, write_stdout};
use crate::toolchain_config::{ToolchainConfig, get_dslx_path, get_dslx_stdlib_path};
use crate::tools::run_block_to_verilog_non_pipeline;

fn compile(
    matches: &ArgMatches,
    config: &Option<ToolchainConfig>,
) -> xlsynth_block_lang::BlockCompileOutput {
    let input = PathBuf::from(
        matches
            .get_one::<String>("dslx_input_file")
            .expect("clap requires dslx_input_file"),
    );
    let source = std::fs::read_to_string(&input).unwrap_or_else(|error| {
        eprintln!(
            "dslx block error: could not read {}: {error}",
            input.display()
        );
        std::process::exit(1);
    });
    let parametric_bindings = matches
        .get_many::<String>("param")
        .into_iter()
        .flatten()
        .map(|text| parse_parametric_binding(text))
        .collect::<Vec<_>>();
    let combinational_optimization = match matches
        .get_one::<String>("comb_opt")
        .map(String::as_str)
        .unwrap_or("free")
    {
        "free" => CombinationalOptimization::Free,
        "preserve-names" => CombinationalOptimization::PreserveNames,
        "preserve-names-and-functions" => CombinationalOptimization::PreserveNamesAndFunctions,
        value => unreachable!("clap checked comb-opt value: {value}"),
    };
    let dslx_path = get_dslx_path(matches, config);
    let dslx_config = config.as_ref().and_then(|config| config.dslx.as_ref());
    let options = BlockCompileOptions {
        top: matches.get_one::<String>("dslx_top").cloned(),
        parametric_bindings,
        combinational_optimization,
        dslx_stdlib_path: get_dslx_stdlib_path(matches, config).map(PathBuf::from),
        additional_search_paths: dslx_path
            .as_deref()
            .map(|paths| paths.split(';').map(PathBuf::from).collect())
            .unwrap_or_default(),
        enable_warnings: dslx_config.and_then(|dslx| dslx.enable_warnings.clone()),
        disable_warnings: dslx_config.and_then(|dslx| dslx.disable_warnings.clone()),
        tool_path: config
            .as_ref()
            .and_then(|toolchain| toolchain.tool_path.as_ref())
            .map(PathBuf::from),
        ..BlockCompileOptions::default()
    };
    let result = compile_block_module(&source, &input, &options).unwrap_or_else(|error| {
        eprintln!("dslx block error: {error}");
        std::process::exit(1);
    });
    let warnings_as_errors = matches
        .get_one::<String>("warnings_as_errors")
        .map(|value| value == "true")
        .or_else(|| dslx_config.and_then(|dslx| dslx.warnings_as_errors))
        .unwrap_or(false);
    for warning in &result.warnings {
        eprintln!("dslx block warning: {warning}");
    }
    if warnings_as_errors && !result.warnings.is_empty() {
        eprintln!("dslx block error: warnings found with warnings-as-errors enabled");
        std::process::exit(1);
    }
    result
}

fn parse_parametric_binding(text: &str) -> ParametricBinding {
    let Some((name, value)) = text.split_once('=') else {
        eprintln!("dslx block error: --param must be NAME=DSLX_CONST_EXPR, got '{text}'");
        std::process::exit(2);
    };
    if name.is_empty() || value.is_empty() {
        eprintln!("dslx block error: --param must be NAME=DSLX_CONST_EXPR, got '{text}'");
        std::process::exit(2);
    }
    ParametricBinding {
        name: name.to_string(),
        value: value.to_string(),
    }
}

fn extract_block_codegen_flags(
    matches: &ArgMatches,
    config: Option<&ToolchainConfig>,
) -> Result<CodegenFlags, String> {
    let codegen = config.and_then(|config| config.codegen.as_ref());
    if codegen.and_then(|codegen| codegen.use_system_verilog) == Some(false) {
        return Err(
            "dslx-block2sv requires SystemVerilog; remove use_system_verilog=false from the toolchain config"
                .to_string(),
        );
    }
    if codegen.and_then(|codegen| codegen.add_invariant_assertions) == Some(true) {
        return Err(
            "dslx-block2sv does not accept codegen-added invariant assertions; author assertions in the block"
                .to_string(),
        );
    }
    Ok(CodegenFlags {
        input_valid_signal: None,
        output_valid_signal: None,
        use_system_verilog: Some(true),
        flop_inputs: None,
        flop_outputs: None,
        add_idle_output: None,
        add_invariant_assertions: None,
        module_name: matches.get_one::<String>("module_name").cloned(),
        array_index_bounds_checking: matches
            .get_one::<String>("array_index_bounds_checking")
            .map(|value| value == "true")
            .or_else(|| codegen.and_then(|codegen| codegen.array_index_bounds_checking))
            .or(Some(
                crate::flag_defaults::CODEGEN_ARRAY_INDEX_BOUNDS_CHECKING,
            )),
        separate_lines: matches
            .get_one::<String>("separate_lines")
            .map(|value| value == "true"),
        reset: None,
        reset_active_low: None,
        reset_asynchronous: None,
        reset_data_path: None,
        gate_format: None,
        assert_format: None,
        output_schedule_path: None,
        output_verilog_line_map_path: None,
        output_block_ir_path: None,
        output_residual_data_path: None,
        reference_residual_data_path: None,
    })
}

pub fn handle_dslx_block2ir(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    write_stdout(&compile(matches, config).ir_text);
}

pub fn handle_dslx_block2sv(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let result = compile(matches, config);
    let (source_top_name, MemberType::Block) = result
        .package
        .top
        .as_ref()
        .expect("block compiler always selects a top block")
    else {
        unreachable!("block compiler emitted a non-block top");
    };
    let flags = extract_block_codegen_flags(matches, config.as_ref()).unwrap_or_else(|error| {
        eprintln!("dslx block error: {error}");
        std::process::exit(2);
    });
    let mut codegen_package = result.package.clone();
    if let Some(requested) = flags.module_name.as_deref() {
        rename_package_block(&mut codegen_package, source_top_name, requested).unwrap_or_else(
            |error| {
                eprintln!("dslx block error: could not apply --module_name: {error}");
                std::process::exit(1);
            },
        );
    }
    let tool_path = config
        .as_ref()
        .and_then(|toolchain| toolchain.tool_path.as_deref())
        .unwrap_or_else(|| {
            eprintln!("dslx block error: dslx-block2sv requires --toolchain with tool_path");
            std::process::exit(2);
        });
    let temporary = tempfile::NamedTempFile::new().unwrap_or_else(|error| {
        eprintln!("dslx block error: could not create temporary Block IR file: {error}");
        std::process::exit(1);
    });
    let extern_codegen = prepare_xls53_extern_codegen(&codegen_package).unwrap_or_else(|error| {
        eprintln!("dslx block error: could not prepare Verilog FFI codegen: {error}");
        std::process::exit(1);
    });
    std::fs::write(temporary.path(), &extern_codegen.ir_text).unwrap_or_else(|error| {
        eprintln!("dslx block error: could not write temporary Block IR: {error}");
        std::process::exit(1);
    });
    let mut xls_flags = flags.clone();
    xls_flags.module_name = None;
    let generated_system_verilog =
        run_block_to_verilog_non_pipeline(Path::new(temporary.path()), &xls_flags, tool_path);
    let system_verilog =
        reorder_system_verilog_package_ports(&generated_system_verilog, &codegen_package)
            .unwrap_or_else(|error| {
                eprintln!("dslx block error: could not preserve source port order: {error}");
                std::process::exit(1);
            });
    let system_verilog = apply_xls53_extern_codegen(&system_verilog, &extern_codegen)
        .unwrap_or_else(|error| {
            eprintln!("dslx block error: could not restore Verilog FFI code: {error}");
            std::process::exit(1);
        });
    write_stdout(&system_verilog);
}
