// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::common::{
    extract_codegen_flags, extract_pipeline_spec, pipeline_codegen_flags_proto,
    scheduling_options_proto, CodegenFlags, PipelineSpec,
};
use crate::toolchain_config::ToolchainConfig;
use crate::tools::{run_codegen_pipeline, run_opt_main};

pub fn handle_ir2pipeline(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    let input_file = matches.get_one::<String>("ir_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    let delay_model = matches.get_one::<String>("DELAY_MODEL").unwrap();

    // See which of pipeline_stages or clock_period_ps we're using.
    let pipeline_spec = extract_pipeline_spec(matches);

    let codegen_flags = extract_codegen_flags(matches, config.as_ref());

    let keep_temps = matches.get_one::<String>("keep_temps").map(|s| s == "true");

    let optimize = matches
        .get_one::<String>("opt")
        .map(|s| s == "true")
        .unwrap_or(false);

    let ir_top_opt = matches.get_one::<String>("ir_top");

    ir2pipeline(
        input_path,
        delay_model,
        &pipeline_spec,
        &codegen_flags,
        optimize,
        ir_top_opt.map(|s| s.as_str()),
        &keep_temps,
        config,
    );
}

/// To convert an IR file to a pipeline we run the codegen_main command and give
/// it a number of pipeline stages.
fn ir2pipeline(
    input_file: &std::path::Path,
    delay_model: &str,
    pipeline_spec: &PipelineSpec,
    codegen_flags: &CodegenFlags,
    optimize: bool,
    ir_top: Option<&str>,
    keep_temps: &Option<bool>,
    config: &Option<ToolchainConfig>,
) {
    log::info!("ir2pipeline");
    if let Some(tool_path) = config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        // Temporary directory for any artifacts we create (optimized IR and generated
        // SV).
        let temp_dir = tempfile::TempDir::new().unwrap();

        // Determine which IR file we hand to codegen.
        let ir_for_codegen_path: std::path::PathBuf = if optimize {
            // Ensure top was provided.
            let top_name = ir_top.expect("--opt requires --top to be specified");
            // Optimize the incoming IR first.
            let opt_ir = run_opt_main(input_file, Some(top_name), tool_path);
            let opt_ir_path = temp_dir.path().join("opt.ir");
            std::fs::write(&opt_ir_path, &opt_ir).unwrap();
            opt_ir_path
        } else {
            // Just use the input file directly (no optimisation step).
            input_file.to_path_buf()
        };

        // Run the codegen pipeline.
        let sv = run_codegen_pipeline(
            &ir_for_codegen_path,
            delay_model,
            pipeline_spec,
            codegen_flags,
            tool_path,
        );

        // Persist the SV output in the temp dir for symmetry with dslx2pipeline.
        let sv_path = temp_dir.path().join("output.sv");
        std::fs::write(&sv_path, &sv).unwrap();

        // Keep temporary files if requested.
        if let Some(_) = keep_temps {
            let temp_dir_path = temp_dir.keep();
            eprintln!(
                "Pipeline generation successful. Output written to: {}",
                temp_dir_path.to_str().unwrap()
            );
        }

        println!("{}", sv);
    } else {
        // -- Runtime API implementation (no external toolchain path) --

        // Read the IR text from the provided file.
        let ir_text =
            std::fs::read_to_string(input_file).expect("IR input file should be readable");

        // Parse the IR package.
        let mut ir_package =
            xlsynth::IrPackage::parse_ir(&ir_text, input_file.file_name().and_then(|s| s.to_str()))
                .expect("IR parsing should succeed");

        // Optionally optimize the IR package.
        if optimize {
            let top_name = ir_top.expect("--opt requires --top to be specified");
            ir_package = xlsynth::optimize_ir(&ir_package, top_name)
                .expect("IR optimization should succeed");
        }

        let scheduling_options_flags_proto = scheduling_options_proto(delay_model, pipeline_spec);
        let codegen_flags_proto = pipeline_codegen_flags_proto(codegen_flags);

        let codegen_result = xlsynth::schedule_and_codegen(
            &ir_package,
            &scheduling_options_flags_proto,
            &codegen_flags_proto,
        )
        .expect("schedule and codegen should succeed");

        let sv = codegen_result.get_verilog_text().unwrap();
        println!("{}", sv);
    }
}
