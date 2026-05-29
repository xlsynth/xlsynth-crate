// SPDX-License-Identifier: Apache-2.0

//! DSLX pipeline-stage lowering into the native sequential g8r representation.

use std::path::Path;

use xlsynth::{DslxConvertOptions, convert_dslx_to_ir, dslx, optimize_ir};

use super::{
    check_for_parametric_stages, check_implicit_stage_numbering, discover_stage_names,
    verify_stage_port_widths, verify_stage_signatures,
};
use crate::aig::{
    ClockPort, SequentialGateFn, SequentialPipelineOptions, stitch_gate_fns_into_pipeline,
};
use crate::process_ir_path::{CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts};

/// Options for turning DSLX stage functions into one sequential g8r design.
#[derive(Debug)]
pub struct StitchG8rPipelineOptions<'a> {
    pub explicit_stages: Option<Vec<String>>,
    pub stdlib_path: Option<&'a Path>,
    pub search_paths: Vec<&'a Path>,
    pub flop_inputs: bool,
    pub flop_outputs: bool,
    pub input_valid_signal: Option<&'a str>,
    pub output_valid_signal: Option<&'a str>,
    pub reset_signal: Option<&'a str>,
    pub reset_active_low: bool,
    pub output_design_name: &'a str,
    pub clock_name: &'a str,
}

/// Lowers DSLX stage functions and stitches them into a sequential g8r design.
pub fn stitch_g8r_pipeline(
    dslx_text: &str,
    path: &Path,
    top: &str,
    options: &StitchG8rPipelineOptions<'_>,
    lowering_options: &CanonicalG8rOptions,
) -> Result<SequentialGateFn, xlsynth::XlsynthError> {
    let explicit_stages = options.explicit_stages.as_deref();
    let module_name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| xlsynth::XlsynthError("invalid path".into()))?;
    let mut import_data = dslx::ImportData::new(options.stdlib_path, &options.search_paths);
    let typechecked_module = dslx::parse_and_typecheck(
        dslx_text,
        path.to_str().unwrap(),
        module_name,
        &mut import_data,
    )?;
    check_for_parametric_stages(&typechecked_module, top, explicit_stages)?;
    if explicit_stages.is_none() {
        check_implicit_stage_numbering(&typechecked_module, top)?;
    }

    let convert_options = DslxConvertOptions {
        dslx_stdlib_path: options.stdlib_path,
        additional_search_paths: options.search_paths.clone(),
        enable_warnings: None,
        disable_warnings: None,
        force_implicit_token_calling_convention: false,
    };
    let conversion = convert_dslx_to_ir(dslx_text, path, &convert_options)?;
    let ir = conversion.ir;
    let stages = discover_stage_names(&ir, path, top, explicit_stages)?;
    verify_stage_signatures(&ir, &stages)?;
    verify_stage_port_widths(&ir, &stages)?;

    let mut stage_gate_fns = Vec::with_capacity(stages.len());
    for (stage_name, stage_mangled) in &stages {
        let optimized_ir = optimize_ir(&ir, stage_mangled)?;
        let lowering = canonical_ir_text_to_g8r_lowering_artifacts(
            &optimized_ir.to_string(),
            Some(stage_mangled),
            lowering_options,
        )
        .map_err(|error| {
            xlsynth::XlsynthError(format!(
                "failed to lower pipeline stage '{stage_name}' to g8r: {error}"
            ))
        })?;
        stage_gate_fns.push(lowering.gate_fn);
    }

    stitch_gate_fns_into_pipeline(
        &stage_gate_fns,
        &SequentialPipelineOptions {
            name: options.output_design_name.to_string(),
            clock: ClockPort {
                name: options.clock_name.to_string(),
            },
            flop_inputs: options.flop_inputs,
            flop_outputs: options.flop_outputs,
            input_valid_signal: options.input_valid_signal.map(str::to_string),
            output_valid_signal: options.output_valid_signal.map(str::to_string),
            reset_signal: options.reset_signal.map(str::to_string),
            reset_active_low: options.reset_active_low,
        },
    )
    .map_err(xlsynth::XlsynthError)
}
