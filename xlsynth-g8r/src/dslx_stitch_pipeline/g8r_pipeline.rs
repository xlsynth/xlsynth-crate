// SPDX-License-Identifier: Apache-2.0

//! DSLX pipeline-stage lowering into the native sequential g8r representation.

use std::path::Path;

use xlsynth::optimize_ir;

use super::{ResolvedStitchPipeline, spec::resolve_stages};
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
    let resolved = resolve_stages(
        dslx_text,
        path,
        top,
        options.explicit_stages.as_deref(),
        options.stdlib_path,
        &options.search_paths,
        SequentialPipelineOptions {
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
    )?;
    resolved.lower_to_g8r(lowering_options)
}

impl ResolvedStitchPipeline {
    /// Lowers the resolved stages and stitches their combinational gate
    /// functions into a clocked design.
    pub fn lower_to_g8r(
        &self,
        lowering_options: &CanonicalG8rOptions,
    ) -> Result<SequentialGateFn, xlsynth::XlsynthError> {
        let mut stage_gate_fns = Vec::with_capacity(self.stages.len());
        for (stage_name, stage_mangled) in &self.stages {
            let optimized_ir = optimize_ir(&self.ir, stage_mangled)?;
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

        stitch_gate_fns_into_pipeline(&stage_gate_fns, &self.boundary)
            .map_err(xlsynth::XlsynthError)
    }
}
