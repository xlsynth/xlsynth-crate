// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::common::{
    extract_codegen_flags, extract_pipeline_spec, parse_bool_flag, resolve_type_inference_v2,
    CodegenFlags, PipelineSpec,
};
use crate::toolchain_config::ToolchainConfig;
use crate::tools::{run_codegen_pipeline, run_ir_converter_main, run_opt_main};

fn dslx2pipeline_eco(
    input_file: &std::path::Path,
    dslx_top: &str,
    pipeline_spec: &PipelineSpec,
    codegen_flags: &CodegenFlags,
    delay_model: &str,
    keep_temps: &Option<bool>,
    type_inference_v2: Option<bool>,
    output_unopt_ir: &Option<&std::path::Path>,
    output_opt_ir: &Option<&std::path::Path>,
    config: &Option<ToolchainConfig>,
    baseline_unopt_ir_path: &std::path::Path,
    edits_debug_out: &Option<&std::path::Path>,
) {
    log::info!("dslx2pipeline_eco; config: {:?}", config);
    let module_name = xlsynth::dslx_path_to_module_name(input_file).unwrap();
    let ir_top = xlsynth::mangle_dslx_name(module_name, dslx_top).unwrap();

    let tool_path = match config.as_ref().and_then(|c| c.tool_path.as_deref()) {
        Some(p) => p,
        None => {
            eprintln!("error: --toolchain is required for this command");
            std::process::exit(1);
        }
    };

    log::info!("dslx2pipeline using tool path: {}", tool_path);
    let temp_dir = tempfile::TempDir::new().unwrap();

    let dslx_stdlib_path = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.dslx_stdlib_path.as_deref());
    let dslx_path_slice = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.dslx_path.as_deref());
    let dslx_path = dslx_path_slice.map(|s| s.join(":"));
    let dslx_path_ref = dslx_path.as_ref().map(|s| s.as_str());

    let enable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.enable_warnings.as_deref());
    let disable_warnings = config
        .as_ref()
        .and_then(|c| c.dslx.as_ref()?.disable_warnings.as_deref());
    let unopt_ir = run_ir_converter_main(
        input_file,
        Some(dslx_top),
        dslx_stdlib_path,
        dslx_path_ref,
        tool_path,
        enable_warnings,
        disable_warnings,
        type_inference_v2,
        /* convert_tests= */ false,
    );
    let unopt_ir_path = temp_dir.path().join("unopt.ir");
    std::fs::write(&unopt_ir_path, unopt_ir).unwrap();

    // Run the baseline unopt IR and new unopt IR through the optimizer
    let baseline_opt_ir = run_opt_main(&baseline_unopt_ir_path, Some(&ir_top), tool_path);
    let baseline_opt_ir_path = temp_dir.path().join("baseline_opt.ir");
    std::fs::write(&baseline_opt_ir_path, baseline_opt_ir).unwrap();

    // Run the baseline opt IR through codegen to get the block IR and the residual
    // metadata.
    let baseline_block_ir_path = temp_dir.path().join("baseline.block.ir");
    let baseline_residual_data_path = temp_dir.path().join("residual_data.pb");
    let mut baseline_codegen_flags = codegen_flags.clone();
    baseline_codegen_flags.set_output_block_ir_path(&baseline_block_ir_path);
    baseline_codegen_flags.set_output_residual_data_path(&baseline_residual_data_path);
    let baseline_sv = run_codegen_pipeline(
        &baseline_opt_ir_path,
        delay_model,
        pipeline_spec,
        &baseline_codegen_flags,
        tool_path,
    );
    let baseline_sv_path = temp_dir.path().join("baseline_sv.sv");
    std::fs::write(&baseline_sv_path, &baseline_sv).unwrap();

    let opt_ir = run_opt_main(&unopt_ir_path, Some(&ir_top), tool_path);
    let opt_ir_path = temp_dir.path().join("opt.ir");
    std::fs::write(&opt_ir_path, opt_ir.clone()).unwrap();

    if let Some(path) = output_unopt_ir {
        std::fs::write(path, &std::fs::read_to_string(&unopt_ir_path).unwrap())
            .expect("write output_unopt_ir");
    }
    if let Some(path) = output_opt_ir {
        std::fs::write(path, &std::fs::read_to_string(&opt_ir_path).unwrap())
            .expect("write output_opt_ir");
    }

    // Run the new opt IR through codegen to get the new block IR.
    let mut new_codegen_flags = codegen_flags.clone();
    let new_block_ir_path = temp_dir.path().join("new.block.ir");
    new_codegen_flags.set_output_block_ir_path(&new_block_ir_path);
    new_codegen_flags.set_reference_residual_data_path(&baseline_residual_data_path);
    let sv = run_codegen_pipeline(
        &opt_ir_path,
        delay_model,
        pipeline_spec,
        &new_codegen_flags,
        tool_path,
    );
    let sv_path = temp_dir.path().join("output.sv");
    std::fs::write(&sv_path, &sv).unwrap();

    // compute the edit distance between the baseline and new block IRs.

    // Run block_to_verilog to get the patched SV.

    if let Some(_) = keep_temps {
        let temp_dir_path = temp_dir.keep();
        eprintln!(
            "Pipeline generation successful. Output written to: {}",
            temp_dir_path.to_str().unwrap()
        );
    }
    println!("{}", sv);
}

pub fn handle_dslx2pipeline_eco(matches: &ArgMatches, config: &Option<ToolchainConfig>) {
    // Inputs: same as dslx2pipeline + baseline_unopt_ir
    let input_file = matches.get_one::<String>("dslx_input_file").unwrap();
    let input_path = std::path::Path::new(input_file);
    let dslx_top = matches.get_one::<String>("dslx_top").unwrap();
    let delay_model = matches.get_one::<String>("DELAY_MODEL").unwrap();
    let pipeline_spec = extract_pipeline_spec(matches);
    let codegen_flags = extract_codegen_flags(matches, config.as_ref());
    let keep_temps = parse_bool_flag(matches, "keep_temps");
    let baseline_unopt_ir_path = std::path::Path::new(
        matches
            .get_one::<String>("baseline_unopt_ir")
            .expect("--baseline_unopt_ir is required"),
    );
    let edits_debug_out = matches.get_one::<String>("edits_debug_out");

    let output_unopt_ir: Option<std::path::PathBuf> = matches
        .get_one::<String>("output_unopt_ir")
        .map(|s| std::path::PathBuf::from(s));
    let output_opt_ir: Option<std::path::PathBuf> = matches
        .get_one::<String>("output_opt_ir")
        .map(|s| std::path::PathBuf::from(s));

    let type_inference_v2 = resolve_type_inference_v2(matches, config);

    dslx2pipeline_eco(
        input_path,
        dslx_top,
        &pipeline_spec,
        &codegen_flags,
        delay_model,
        &keep_temps,
        type_inference_v2,
        &output_unopt_ir.as_deref(),
        &output_opt_ir.as_deref(),
        config,
        baseline_unopt_ir_path,
        &edits_debug_out.as_deref().map(|s| std::path::Path::new(s)),
    );
}
