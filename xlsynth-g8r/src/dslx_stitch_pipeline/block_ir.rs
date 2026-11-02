// SPDX-License-Identifier: Apache-2.0

//! Constructs the stitched wrapper as an XLS block invoking the stage
//! functions.

use std::ops::Range;

use xlsynth_pir::ir::{self, Type};
use xlsynth_pir::{
    BValue, BlockBuilder, BuilderError, IrValue, RegisterWriteOptions, ResetBehavior,
};

use super::layout::{PipelineBoundary, PipelineLayout};
use super::names::{bank_reg_name, flop_reg_name, stage_output_wire_name, valid_reg_name};
use crate::aig::SequentialPipelineOptions;

/// Appends the wrapper to its stage package and selects it as the package top.
pub(super) fn build_pipeline_block(
    package: &mut ir::Package,
    stages: &[(String, String)],
    boundary: &SequentialPipelineOptions,
    layout: &PipelineLayout,
) -> Result<(), String> {
    let functions = stages
        .iter()
        .map(|(name, mangled)| {
            package.get_fn(mangled).ok_or_else(|| {
                format!("missing IR function '{mangled}' for pipeline stage '{name}'")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (first, rest) = functions
        .split_first()
        .ok_or_else(|| "cannot build a block for an empty pipeline".to_string())?;
    pipeline_builder(first, rest, boundary, layout)
        .and_then(|builder| builder.build_into_package(package))
        .map_err(|error| {
            format!(
                "could not build pipeline block '{}': {error}",
                boundary.name
            )
        })?;
    package.set_top_block(&boundary.name)
}

/// Connects each stage to the preceding register layer in the declared order.
fn pipeline_builder(
    first: &ir::Fn,
    rest: &[&ir::Fn],
    boundary: &SequentialPipelineOptions,
    layout: &PipelineLayout,
) -> Result<BlockBuilder, BuilderError> {
    let mut builder = BlockBuilder::new(&boundary.name);
    builder.clock_port(&boundary.clock.name)?;
    let reset = if let Some(name) = &boundary.reset_signal {
        let port = builder.input_port(name, Type::Bits(1))?;
        builder.set_reset(
            port,
            ResetBehavior {
                asynchronous: false,
                active_low: boundary.reset_active_low,
            },
        )?;
        Some(port)
    } else {
        None
    };

    let inputs = &layout.stage_inputs[0];
    let mut arguments = Vec::with_capacity(inputs.ports.len());
    for (param, port) in first.param_nodes().zip(&inputs.ports) {
        let width = port.bits.len();
        let packed = builder.input_port(&port.name, Type::Bits(width))?;
        arguments.push(unpack_for_storage(
            &mut builder,
            packed,
            &param.ty,
            0..width,
        )?);
    }
    let valid = boundary
        .input_valid_signal
        .as_ref()
        .map(|name| builder.input_port(name, Type::Bits(1)))
        .transpose()?;
    let mut control = PipelineControl { valid, reset };
    control.register_boundary(&mut builder, inputs, &mut arguments)?;

    let mut result = invoke_stage(&mut builder, first, arguments, 0)?;
    for (i, stage) in rest.iter().enumerate() {
        let inputs = layout.after_stage(i);
        // A token-free unary argument can cross the register layer whole.
        // Multiple arguments split the packed result starting at the LSB.
        let mut arguments = if !inputs.banks.is_empty() {
            let ty = builder.get_type(result)?.clone();
            let packed = pack_value(&mut builder, result, &ty)?;
            let registered = control.register_banks(&mut builder, inputs, packed)?;
            stage
                .param_nodes()
                .zip(&inputs.ports)
                .map(|(p, port)| {
                    unpack_for_storage(&mut builder, registered, &p.ty, port.bits.clone())
                })
                .collect::<Result<Vec<_>, _>>()?
        } else if inputs.ports.len() == 1 && !builder.get_type(result)?.contains_token() {
            vec![result]
        } else {
            let ty = builder.get_type(result)?.clone();
            let packed = pack_value(&mut builder, result, &ty)?;
            stage
                .param_nodes()
                .zip(&inputs.ports)
                .map(|(p, port)| unpack_for_storage(&mut builder, packed, &p.ty, port.bits.clone()))
                .collect::<Result<Vec<_>, _>>()?
        };
        if inputs.banks.is_empty() {
            control.register_boundary(&mut builder, inputs, &mut arguments)?;
        }
        result = invoke_stage(&mut builder, stage, arguments, i + 1)?;
    }

    let ty = builder.get_type(result)?.clone();
    let mut output = pack_value(&mut builder, result, &ty)?;
    control.register_boundary(
        &mut builder,
        &layout.output,
        std::slice::from_mut(&mut output),
    )?;
    builder.output_port(&layout.output.ports[0].name, output)?;
    if let (Some(name), Some(valid)) = (&boundary.output_valid_signal, control.valid) {
        builder.output_port(name, valid)?;
    }
    Ok(builder)
}

/// Leaves token-bearing arguments packed because XLS cannot trace a token
/// through a register read.
fn unpack_for_storage(
    builder: &mut BlockBuilder,
    packed: BValue,
    ty: &Type,
    bits: Range<usize>,
) -> Result<BValue, BuilderError> {
    let mut offset = bits.start;
    if ty.contains_token() {
        unpack_value(builder, packed, &Type::Bits(bits.len()), &mut offset)
    } else {
        unpack_value(builder, packed, ty, &mut offset)
    }
}

/// Restores token-bearing arguments from their data bits before the stage call.
fn invoke_stage(
    builder: &mut BlockBuilder,
    stage: &ir::Fn,
    mut arguments: Vec<BValue>,
    stage_index: usize,
) -> Result<BValue, BuilderError> {
    for (argument, param) in arguments.iter_mut().zip(stage.param_nodes()) {
        if param.ty.contains_token() {
            let mut offset = 0;
            *argument = unpack_value(builder, *argument, &param.ty, &mut offset)?;
        }
    }
    let result = builder.invoke(stage, &arguments)?;
    builder.set_name(result, &stage_output_wire_name(stage_index))?;
    Ok(result)
}

/// Carries the block's reset port and the valid signal at the current boundary.
struct PipelineControl {
    valid: Option<BValue>,
    reset: Option<BValue>,
}

impl PipelineControl {
    /// Registers packed fields under their producer's predicates, then restores
    /// the packed value for the next stage's normal port slicing.
    fn register_banks(
        &mut self,
        builder: &mut BlockBuilder,
        boundary: &PipelineBoundary,
        packed: BValue,
    ) -> Result<BValue, BuilderError> {
        let layer = boundary
            .register_layer
            .expect("banks have a register layer");
        let mut parts = Vec::new();
        for segment in boundary.bank_segments().into_iter().rev() {
            if !segment.store {
                parts.push(builder.literal(IrValue::bool(false))?);
                continue;
            }
            let width = segment.bits.len();
            let data = builder.bit_slice(packed, segment.bits.start, width)?;
            let mut load = self.valid;
            if let Some(bit) = segment.enable {
                let enabled = builder.bit_slice(packed, bit, 1)?;
                load = Some(match load {
                    Some(valid) => builder.and(valid, enabled)?,
                    None => enabled,
                });
            }
            let reg = builder.register(
                &bank_reg_name(layer, segment.bits.start),
                Type::Bits(width),
                None,
            )?;
            builder.register_write(
                reg,
                data,
                RegisterWriteOptions {
                    load_enable: load,
                    reset: None,
                },
            )?;
            parts.push(builder.register_read(reg)?);
        }
        self.register_valid(builder, layer)?;
        builder.concat(&parts)
    }

    /// Registers data on valid cycles and advances valid on every clock edge.
    fn register_boundary(
        &mut self,
        builder: &mut BlockBuilder,
        boundary: &PipelineBoundary,
        data: &mut [BValue],
    ) -> Result<(), BuilderError> {
        let Some(layer) = boundary.register_layer else {
            return Ok(());
        };
        for (port, value) in boundary.ports.iter().zip(data) {
            let register = builder.register(
                &flop_reg_name(layer, &port.name),
                builder.get_type(*value)?.clone(),
                None,
            )?;
            builder.register_write(
                register,
                *value,
                RegisterWriteOptions {
                    load_enable: self.valid,
                    reset: None,
                },
            )?;
            *value = builder.register_read(register)?;
        }
        self.register_valid(builder, layer)
    }

    /// Advances the transaction-valid bit alongside the data at this boundary.
    fn register_valid(
        &mut self,
        builder: &mut BlockBuilder,
        layer: usize,
    ) -> Result<(), BuilderError> {
        if let Some(valid) = self.valid {
            let register = builder.register(
                &valid_reg_name(layer),
                Type::Bits(1),
                self.reset.map(|_| IrValue::bool(false)),
            )?;
            builder.register_write(
                register,
                valid,
                RegisterWriteOptions {
                    load_enable: None,
                    reset: self.reset,
                },
            )?;
            self.valid = Some(builder.register_read(register)?);
        }
        Ok(())
    }
}

/// Packs a stage value with tuple field zero at the MSB and array element zero
/// at the LSB.
fn pack_value(
    builder: &mut BlockBuilder,
    value: BValue,
    ty: &Type,
) -> Result<BValue, BuilderError> {
    match ty {
        Type::Bits(_) => Ok(value),
        Type::Tuple(fields) => {
            let mut parts = Vec::with_capacity(fields.len());
            for (i, field_ty) in fields.iter().enumerate() {
                let field = builder.tuple_index(value, i)?;
                parts.push(pack_value(builder, field, field_ty)?);
            }
            builder.concat(&parts)
        }
        Type::Array(array) => {
            if array.element_count == 0 || array.element_type.bit_count() == 0 {
                return builder.concat(&[]);
            }
            if array.element_type.contains_token() {
                return Err(BuilderError::InvalidOperation(format!(
                    "cannot pack stage result containing {ty}: XLS array_index cannot extract an element containing a token"
                )));
            }
            let mut parts = Vec::with_capacity(array.element_count);
            for i in (0..array.element_count).rev() {
                let index = builder.literal(IrValue::u64(i as u64))?;
                let element = builder.array_index(value, index)?;
                parts.push(pack_value(builder, element, &array.element_type)?);
            }
            builder.concat(&parts)
        }
        Type::Token => builder.concat(&[]),
    }
}

/// Reconstructs one typed argument from the packed bits starting at the LSB
/// offset.
fn unpack_value(
    builder: &mut BlockBuilder,
    packed: BValue,
    ty: &Type,
    offset: &mut usize,
) -> Result<BValue, BuilderError> {
    match ty {
        Type::Bits(width) => {
            let start = *offset;
            *offset += width;
            if start == 0 && builder.get_type(packed)? == ty {
                Ok(packed)
            } else {
                builder.bit_slice(packed, start, *width)
            }
        }
        Type::Tuple(fields) => {
            let mut elements = Vec::with_capacity(fields.len());
            for field_ty in fields.iter().rev() {
                elements.push(unpack_value(builder, packed, field_ty, offset)?);
            }
            elements.reverse();
            builder.tuple(&elements)
        }
        Type::Array(array) => {
            let mut elements = Vec::with_capacity(array.element_count);
            for _ in 0..array.element_count {
                elements.push(unpack_value(builder, packed, &array.element_type, offset)?);
            }
            builder.array((*array.element_type).clone(), &elements)
        }
        Type::Token => builder.literal(IrValue::make_token()),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use xlsynth_pir::ir::Type;

    use crate::dslx_stitch_pipeline::StitchPipelineSpec;

    #[test]
    fn token_tuple_arguments_cross_registers_as_data_bits() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("echo.x"),
            r#"fn first(x: (token, u4)) -> (token, u4) { x }
fn second(x: (token, u4)) -> (token, u4) { x }"#,
        )
        .unwrap();
        let manifest = directory.path().join("echo.toml");
        fs::write(
            &manifest,
            r#"dslx_input_file = "echo.x"
stages = ["first", "second"]
output_module_name = "pipe"
flop_outputs = false
input_valid_signal = "present"
output_valid_signal = "out_valid"
"#,
        )
        .unwrap();
        let mut spec = StitchPipelineSpec::load(&manifest).unwrap();
        for flop_inputs in [false, true] {
            spec.flop_inputs = flop_inputs;
            let package = spec.resolve().unwrap().emit_block_ir().unwrap();
            let block = package.get_top_block().unwrap();
            let types: Vec<&Type> = block
                .registers
                .iter()
                .map(|register| &register.ty)
                .collect();
            let layer_types = [&Type::Bits(4), &Type::Bits(1)];
            assert_eq!(types, layer_types.repeat(1 + usize::from(flop_inputs)));
            let native = xlsynth::IrPackage::parse_ir(&package.to_string(), None).unwrap();
            xlsynth::optimize_ir(&native, "pipe").unwrap();
        }
    }
}
