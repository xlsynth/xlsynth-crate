// SPDX-License-Identifier: Apache-2.0

//! Packed data connections and register placement for a DSLX pipeline wrapper.

use std::ops::Range;

use xlsynth::XlsynthError;
use xlsynth::ir_package::IrPackage;

use crate::aig::SequentialPipelineOptions;

/// One destination port and its LSB-first slice of a packed stage result.
pub(super) struct BoundaryPort {
    pub(super) name: String,
    pub(super) bits: Range<usize>,
}

/// Data ports at a boundary, in declaration order, and their register layer.
pub(super) struct PipelineBoundary {
    pub(super) ports: Vec<BoundaryPort>,
    pub(super) register_layer: Option<usize>,
}

impl PipelineBoundary {
    fn new(
        ports: Vec<(String, usize)>,
        register_layer: Option<usize>,
    ) -> Result<Self, XlsynthError> {
        let mut offset = 0usize;
        let mut packed = Vec::with_capacity(ports.len());
        for (name, width) in ports {
            // VAST represents packed widths and slice indices as signed i64.
            let end = offset
                .checked_add(width)
                .filter(|end| i64::try_from(*end).is_ok())
                .ok_or_else(|| {
                    XlsynthError(format!(
                        "packed pipeline boundary overflows at port `{name}`"
                    ))
                })?;
            packed.push(BoundaryPort {
                name,
                bits: offset..end,
            });
            offset = end;
        }
        Ok(Self {
            ports: packed,
            register_layer,
        })
    }

    pub(super) fn width(&self) -> usize {
        self.ports.last().map_or(0, |port| port.bits.end)
    }
}

/// The boundary before each stage and the boundary driving the wrapper output.
pub(super) struct PipelineLayout {
    pub(super) stage_inputs: Vec<PipelineBoundary>,
    pub(super) output: PipelineBoundary,
}

impl PipelineLayout {
    /// Resolves the port widths from the same function signatures used to
    /// typecheck the pipeline.
    pub(super) fn from_ir(
        ir: &IrPackage,
        stages: &[(String, String)],
        options: &SequentialPipelineOptions,
    ) -> Result<Self, XlsynthError> {
        let mut inputs = Vec::with_capacity(stages.len());
        let mut output_width = 0;
        for (name, mangled) in stages {
            let ports = stage_ports(ir, name, mangled)?;
            inputs.push(ports.inputs);
            output_width = ports.output_width;
        }
        Self::new(
            inputs,
            ("out".into(), output_width),
            options.flop_inputs,
            options.flop_outputs,
            options.input_valid_signal.is_some(),
        )
    }

    /// Numbers register layers in pipeline order. A requested input layer
    /// reserves its number even when it contains no registers.
    pub(super) fn new(
        inputs: Vec<Vec<(String, usize)>>,
        output: (String, usize),
        flop_inputs: bool,
        flop_outputs: bool,
        has_valid: bool,
    ) -> Result<Self, XlsynthError> {
        if inputs.is_empty() {
            return Err(XlsynthError("no pipeline stages found".into()));
        }
        let mut stage_inputs = Vec::with_capacity(inputs.len());
        let mut next_layer = 0;
        for (stage, ports) in inputs.into_iter().enumerate() {
            let layer = if stage > 0 || flop_inputs {
                let number = next_layer;
                next_layer += 1;
                // An empty input layer has no state but still reserves p0.
                (!ports.is_empty() || has_valid).then_some(number)
            } else {
                None
            };
            stage_inputs.push(PipelineBoundary::new(ports, layer)?);
        }
        let output = PipelineBoundary::new(vec![output], flop_outputs.then_some(next_layer))?;
        Ok(Self {
            stage_inputs,
            output,
        })
    }

    /// The next stage's arguments, or the external output for the last stage.
    pub(super) fn after_stage(&self, index: usize) -> &PipelineBoundary {
        self.stage_inputs.get(index + 1).unwrap_or(&self.output)
    }

    pub(super) fn latency(&self) -> usize {
        self.stage_inputs
            .iter()
            .chain([&self.output])
            .filter(|boundary| boundary.register_layer.is_some())
            .count()
    }
}

/// Flat widths for the data interface of a single DSLX stage.
struct StagePorts {
    inputs: Vec<(String, usize)>,
    output_width: usize,
}

/// Reads the ports of a stage and verifies that each can be exposed as RTL.
fn stage_ports(ir: &IrPackage, name: &str, mangled: &str) -> Result<StagePorts, XlsynthError> {
    let function = ir.get_function(mangled)?;
    let signature = function.get_type()?;
    let mut inputs = Vec::new();
    for i in 0..signature.param_count() {
        let param_name = function.param_name(i)?;
        let ty = signature.param_type(i)?;
        let width = ty.get_flat_bit_count();
        if width == 0 {
            return Err(XlsynthError(format!(
                "stage parameter `{name}.{param_name}` has zero-width type `{ty}`; stitch-pipeline cannot emit it as a Verilog port"
            )));
        }
        let width = usize::try_from(width).map_err(|_| {
            XlsynthError(format!(
                "stage parameter `{name}.{param_name}` is too wide for this host"
            ))
        })?;
        inputs.push((param_name, width));
    }
    let ty = signature.return_type();
    let width = ty.get_flat_bit_count();
    if width == 0 {
        return Err(XlsynthError(format!(
            "stage `{name}` has zero-width return type `{ty}`; stitch-pipeline cannot emit it as Verilog output port `out`"
        )));
    }
    Ok(StagePorts {
        inputs,
        output_width: usize::try_from(width).map_err(|_| {
            XlsynthError(format!(
                "stage `{name}` has a return type too wide for this host"
            ))
        })?,
    })
}

/// Checks the data interfaces before constructing a constant-flop proof.
pub(super) fn verify_stage_port_widths(
    ir: &IrPackage,
    stages: &[(String, String)],
) -> Result<(), XlsynthError> {
    for (name, mangled) in stages {
        stage_ports(ir, name, mangled)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_layer_reserves_its_number_without_adding_latency() {
        for has_valid in [false, true] {
            let layout = PipelineLayout::new(
                vec![vec![], vec![("lo".into(), 3), ("hi".into(), 5)]],
                ("out".into(), 8),
                /* flop_inputs= */ true,
                /* flop_outputs= */ true,
                has_valid,
            )
            .unwrap();
            assert_eq!(
                layout.stage_inputs[0].register_layer,
                has_valid.then_some(0)
            );
            let middle = layout.after_stage(0);
            assert_eq!(middle.register_layer, Some(1));
            assert_eq!(middle.ports[0].bits, 0..3);
            assert_eq!(middle.ports[1].bits, 3..8);
            assert_eq!(layout.after_stage(1).register_layer, Some(2));
            assert_eq!(layout.latency(), 2 + usize::from(has_valid));
        }
    }
}
