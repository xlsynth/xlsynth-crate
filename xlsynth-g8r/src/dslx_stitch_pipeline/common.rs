// SPDX-License-Identifier: Apache-2.0

use crate::verilog_version::VerilogVersion;

/// Immutable configuration passed around while stitching a pipeline.
#[derive(Clone)]
pub(crate) struct PipelineCfg<'a> {
    pub(crate) ir: &'a xlsynth::ir_package::IrPackage,
    pub(crate) verilog_version: VerilogVersion,
}

/// One port in a stage module (flattened).
#[derive(Debug, Clone)]
pub(crate) struct Port {
    pub(crate) name: String,
    pub(crate) is_input: bool,
    pub(crate) width: u32,
}

/// Information derived for each stage that the wrapper needs.
#[derive(Debug, Clone)]
pub(crate) struct StageInfo {
    pub(crate) sv_text: String,
    pub(crate) ports: Vec<Port>,
    #[allow(dead_code)]
    pub(crate) output_width: u32,
}
