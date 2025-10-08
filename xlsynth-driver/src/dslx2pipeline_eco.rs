// SPDX-License-Identifier: Apache-2.0

use clap::ArgMatches;

use crate::common::{
    extract_codegen_flags, extract_pipeline_spec, parse_bool_flag, resolve_type_inference_v2,
    CodegenFlags, PipelineSpec,
};
use crate::toolchain_config::ToolchainConfig;
use crate::tools::{
    run_block_to_verilog, run_codegen_pipeline, run_ir_converter_main, run_opt_main,
};
use xlsynth_pir::greedy_matching_ged::GreedyMatchSelector;
use xlsynth_pir::ir::PackageMember;
use xlsynth_pir::matching_ged::{apply_block_edits, compute_block_edit};

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
    output_baseline_verilog_path: &Option<&std::path::Path>,
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
    let mut temp_dir = tempfile::Builder::new()
        .prefix("dslx2pipeline_eco.")
        .tempdir()
        .unwrap();
    if let Some(_) = keep_temps {
        temp_dir.disable_cleanup(true);
        eprintln!(
            "`dslx2pipeline_eco` working directory: {}",
            temp_dir.path().display()
        );
    }
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

    // Run the baseline unopt IR through the optimizer
    let baseline_opt_ir = run_opt_main(&baseline_unopt_ir_path, Some(&ir_top), tool_path);
    let baseline_opt_ir_path = temp_dir.path().join("baseline_opt.ir");
    std::fs::write(&baseline_opt_ir_path, baseline_opt_ir).unwrap();

    // Run the baseline opt IR through codegen to get the block IR and the residual
    // metadata.
    let baseline_block_ir_path = temp_dir.path().join("baseline.block.ir");
    let baseline_residual_data_path = temp_dir.path().join("residual_data.pb");
    let baseline_codegen_flags = CodegenFlags {
        output_block_ir_path: Some(baseline_block_ir_path.to_string_lossy().into_owned()),
        output_residual_data_path: Some(baseline_residual_data_path.to_string_lossy().into_owned()),
        ..codegen_flags.clone()
    };
    let baseline_sv = run_codegen_pipeline(
        &baseline_opt_ir_path,
        delay_model,
        pipeline_spec,
        &baseline_codegen_flags,
        tool_path,
    );
    let baseline_sv_path = temp_dir.path().join("baseline_sv.sv");
    std::fs::write(&baseline_sv_path, &baseline_sv).unwrap();
    if let Some(path) = output_baseline_verilog_path {
        // Add a newline to the end of the file to match the output of dslx2pipeline
        // which uses println.
        std::fs::write(path, format!("{}\n", baseline_sv))
            .expect("write output_baseline_verilog_path");
    }

    // Run the new unopt IR through the optimizer
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
    let new_block_ir_path = temp_dir.path().join("new.block.ir");
    let new_codegen_flags = CodegenFlags {
        output_block_ir_path: Some(new_block_ir_path.to_string_lossy().into_owned()),
        ..codegen_flags.clone()
    };
    let sv = run_codegen_pipeline(
        &opt_ir_path,
        delay_model,
        pipeline_spec,
        &new_codegen_flags,
        tool_path,
    );
    let sv_path = temp_dir.path().join("output.sv");
    std::fs::write(&sv_path, &sv).unwrap();

    // Parse the baseline and new block IRs
    let mut baseline_block_ir =
        xlsynth_pir::ir_parser::parse_path_to_package(&baseline_block_ir_path).unwrap();
    let baseline_block = baseline_block_ir.get_block(ir_top.as_str()).unwrap();
    let new_block_ir = xlsynth_pir::ir_parser::parse_path_to_package(&new_block_ir_path).unwrap();
    let new_block = new_block_ir.get_block(ir_top.as_str()).unwrap();

    // Compute the edit.
    let (old_name, old_fn, new_fn) = match (baseline_block, new_block) {
        (PackageMember::Block { func: o, .. }, PackageMember::Block { func: n, .. }) => {
            (o.name.clone(), o, n)
        }
        _ => unreachable!("get_top_block should return a Block"),
    };
    let mut selector = GreedyMatchSelector::new(old_fn, new_fn);
    let edits = compute_block_edit(baseline_block, new_block, &mut selector).unwrap();
    let patched_block = apply_block_edits(baseline_block, &edits).unwrap();

    let edits_path = temp_dir.path().join("edits.txt");
    std::fs::write(&edits_path, format!("{:#?}\n", edits)).unwrap();
    if let Some(path) = edits_debug_out {
        std::fs::write(path, format!("{:#?}\n", edits)).unwrap();
    }

    // Set the patched block to top, and write out the patched block IR.
    let patched_block_name = match &patched_block {
        PackageMember::Block { func, .. } => func.name.as_str(),
        _ => unreachable!("patched_block should be a Block"),
    };
    baseline_block_ir.set_top_block(patched_block_name).unwrap();
    let patched_block_ir_path = temp_dir.path().join("patched.block.ir");
    baseline_block_ir
        .replace_block(&old_name, patched_block)
        .unwrap();
    std::fs::write(&patched_block_ir_path, baseline_block_ir.to_string()).unwrap();

    // Call block_to_verilog to generate the patched SV. Using esidual data only
    // works with combinational generator so check that the options are for a
    // single-statge pipeline with no IO flops.
    if codegen_flags.flop_inputs == Some(true) || codegen_flags.flop_outputs == Some(true) {
        if *pipeline_spec == PipelineSpec::Stages(1) {
            eprintln!(
            "`dslx2pipeline_eco` only works with combinational blocks (single-stage, no IO flops)"
        );
            std::process::exit(1);
        }
    }
    let block_to_verilog_flags = CodegenFlags {
        reference_residual_data_path: Some(
            baseline_residual_data_path.to_string_lossy().into_owned(),
        ),

        ..codegen_flags.clone()
    };
    let patched_sv =
        run_block_to_verilog(&patched_block_ir_path, &block_to_verilog_flags, tool_path);
    let patched_sv_path = temp_dir.path().join("patched.sv");
    std::fs::write(&patched_sv_path, &patched_sv).unwrap();

    println!("{}", patched_sv);
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
    let output_baseline_verilog_path: Option<std::path::PathBuf> = matches
        .get_one::<String>("output_baseline_verilog_path")
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
        &output_baseline_verilog_path.as_deref(),
    );
}
