// SPDX-License-Identifier: Apache-2.0

use std::process;

use clap::ArgMatches;

use crate::common::{
    codegen_flags_to_textproto, extract_codegen_flags, extract_pipeline_spec, CodegenFlags,
    PipelineSpec, DEFAULT_WARNINGS_AS_ERRORS,
};
use crate::toolchain_config::ToolchainConfig;
use crate::tools::{run_codegen_pipeline, run_ir_converter_main, run_opt_main};

/// Converts the DSLX source in `input_file` using the top level entry point
/// named `top` into a Verilog pipeline and prints that to stdout.
fn dslx2pipeline(
    input_file: &std::path::Path,
    dslx_top: &str,
    pipeline_spec: &PipelineSpec,
    codegen_flags: &CodegenFlags,
    delay_model: &str,
    keep_temps: &Option<bool>,
    config: &Option<ToolchainConfig>,
) {
    let module_name = xlsynth::dslx_path_to_module_name(input_file).unwrap();
    let ir_top = xlsynth::mangle_dslx_name(module_name, dslx_top).unwrap();
    log::info!("dslx2pipeline; dslx_top: {}; ir_top: {}", dslx_top, ir_top);

    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        log::info!("dslx2pipeline using tool path: {}", tool_path);
        let temp_dir = tempfile::TempDir::new().unwrap();

        let dslx_stdlib_path = config.as_ref().and_then(|c| c.dslx_stdlib_path.as_deref());
        let dslx_path_slice = config.as_ref().and_then(|c| c.dslx_path.as_deref());
        let dslx_path = dslx_path_slice.map(|s| s.join(":"));
        let dslx_path_ref = dslx_path.as_ref().map(|s| s.as_str());

        let enable_warnings = config.as_ref().and_then(|c| c.enable_warnings.as_deref());
        let disable_warnings = config.as_ref().and_then(|c| c.disable_warnings.as_deref());
        let unopt_ir = run_ir_converter_main(
            input_file,
            Some(dslx_top),
            dslx_stdlib_path,
            dslx_path_ref,
            tool_path,
            enable_warnings,
            disable_warnings,
        );
        let unopt_ir_path = temp_dir.path().join("unopt.ir");
        std::fs::write(&unopt_ir_path, unopt_ir).unwrap();

        let opt_ir = run_opt_main(&unopt_ir_path, Some(&ir_top), tool_path);
        let opt_ir_path = temp_dir.path().join("opt.ir");
        std::fs::write(&opt_ir_path, opt_ir).unwrap();

        let sv = run_codegen_pipeline(
            &opt_ir_path,
            delay_model,
            pipeline_spec,
            codegen_flags,
            tool_path,
        );
        let sv_path = temp_dir.path().join("output.sv");
        std::fs::write(&sv_path, &sv).unwrap();

        if let Some(_) = keep_temps {
            eprintln!(
                "Pipeline generation successful. Output written to: {}",
                temp_dir.into_path().to_str().unwrap()
            );
        }
        println!("{}", sv);
    } else {
        log::info!("dslx2pipeline using runtime APIs");
        let dslx = std::fs::read_to_string(input_file).unwrap();

        let dslx_path = config.as_ref().and_then(|c| c.dslx_path.as_deref());
        let dslx_path_vec = match dslx_path {
            Some(entries) => entries
                .iter()
                .map(|p| std::path::Path::new(p))
                .collect::<Vec<_>>(),
            None => vec![],
        };
        let dslx_stdlib_path = config.as_ref().and_then(|c| c.dslx_stdlib_path.as_deref());
        let enable_warnings = config.as_ref().and_then(|c| c.enable_warnings.as_deref());
        let disable_warnings = config.as_ref().and_then(|c| c.disable_warnings.as_deref());
        let warnings_as_errors = config
            .as_ref()
            .and_then(|c| c.warnings_as_errors)
            .unwrap_or(DEFAULT_WARNINGS_AS_ERRORS);
        let convert_options = xlsynth::DslxConvertOptions {
            dslx_stdlib_path: dslx_stdlib_path.map(|p| std::path::Path::new(p)),
            additional_search_paths: dslx_path_vec,
            enable_warnings: enable_warnings,
            disable_warnings: disable_warnings,
        };
        let convert_result: xlsynth::DslxToIrPackageResult =
            xlsynth::convert_dslx_to_ir(&dslx, input_file, &convert_options)
                .expect("successful conversion");
        if warnings_as_errors && !convert_result.warnings.is_empty() {
            for warning in convert_result.warnings {
                eprintln!(
                    "DSLX warning for {}: {}",
                    input_file.to_str().unwrap(),
                    warning
                );
            }
            eprintln!("DSLX warnings found with warnings-as-errors enabled; exiting.");
            process::exit(1);
        }

        for warning in convert_result.warnings {
            log::warn!(
                "DSLX warning for {}: {}",
                input_file.to_str().unwrap(),
                warning
            );
        }

        let opt_ir = xlsynth::optimize_ir(&convert_result.ir, &ir_top).unwrap();

        let mut sched_opt_lines = vec![format!("delay_model: \"{}\"", delay_model)];
        match pipeline_spec {
            PipelineSpec::Stages(stages) => {
                sched_opt_lines.push(format!("pipeline_stages: {}", stages))
            }
            PipelineSpec::ClockPeriodPs(clock_period_ps) => {
                sched_opt_lines.push(format!("clock_period_ps: {}", clock_period_ps))
            }
        }
        let scheduling_options_flags_proto = sched_opt_lines.join("\n");
        let codegen_flags_proto = format!(
            "register_merge_strategy: STRATEGY_IDENTITY_ONLY\ngenerator: GENERATOR_KIND_PIPELINE\n{}",
            codegen_flags_to_textproto(codegen_flags)
        );
        let codegen_result = xlsynth::schedule_and_codegen(
            &opt_ir,
            &scheduling_options_flags_proto,
            &codegen_flags_proto,
        )
        .unwrap();
        let sv = codegen_result.get_verilog_text().unwrap();
        println!("{}", sv);
    }
}

pub fn handle_dslx2pipeline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    log::info!("handle_dslx2pipeline");
    let input_file = matches.get_one::<String>("dslx_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    let top = matches.get_one::<String>("dslx_top").unwrap();
    let pipeline_spec = extract_pipeline_spec(matches);
    let delay_model = matches.get_one::<String>("DELAY_MODEL").unwrap();
    let keep_temps = matches.get_one::<String>("keep_temps").map(|s| s == "true");
    let codegen_flags = extract_codegen_flags(matches);

    // Stub function for DSLX to SV conversion
    dslx2pipeline(
        input_path,
        top,
        &pipeline_spec,
        &codegen_flags,
        delay_model,
        &keep_temps,
        config,
    );
}
