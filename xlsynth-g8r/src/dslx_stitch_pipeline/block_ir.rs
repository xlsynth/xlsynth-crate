// SPDX-License-Identifier: Apache-2.0

//! Constructs the stitched wrapper as an XLS block invoking the stage
//! functions.

use xlsynth_pir::ir::{self, Type};
use xlsynth_pir::{
    BValue, BlockBuilder, BuilderError, IrValue, RegisterWriteOptions, ResetBehavior,
};

use super::build_pipeline::{flop_reg_name, stage_output_wire_name, valid_reg_name};
use crate::aig::SequentialPipelineOptions;

/// Appends the wrapper to its stage package and selects it as the package top.
pub(super) fn build_pipeline_block(
    package: &mut ir::Package,
    stages: &[(String, String)],
    boundary: &SequentialPipelineOptions,
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
    pipeline_builder(first, rest, boundary)
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

    let mut arguments = Vec::with_capacity(first.params.len());
    for param in first.param_nodes() {
        let port = builder.input_port(param.param_name(), Type::Bits(param.ty.bit_count()))?;
        let mut offset = 0;
        arguments.push(unpack_value(&mut builder, port, &param.ty, &mut offset)?);
    }
    let valid = boundary
        .input_valid_signal
        .as_ref()
        .map(|name| builder.input_port(name, Type::Bits(1)))
        .transpose()?;
    let mut layers = RegisterLayers {
        next: 0,
        valid,
        reset,
    };
    if boundary.flop_inputs {
        layers.insert(
            &mut builder,
            first
                .param_nodes()
                .map(|p| p.param_name())
                .zip(&mut arguments),
        )?;
    }

    let mut result = builder.invoke(first, &arguments)?;
    builder.set_name(result, &stage_output_wire_name(0))?;
    for (i, stage) in rest.iter().enumerate() {
        // A unary stage takes the preceding result whole, including its type.
        // With multiple parameters the wrapper instead slices the packed result
        // starting at the LSB, so a parameter can span tuple fields.
        let mut arguments = if stage.params.len() == 1 {
            vec![result]
        } else {
            let ty = builder.get_type(result)?.clone();
            let packed = pack_value(&mut builder, result, &ty)?;
            let mut offset = 0;
            stage
                .param_nodes()
                .map(|p| unpack_value(&mut builder, packed, &p.ty, &mut offset))
                .collect::<Result<Vec<_>, _>>()?
        };
        layers.insert(
            &mut builder,
            stage
                .param_nodes()
                .map(|p| p.param_name())
                .zip(&mut arguments),
        )?;
        result = builder.invoke(stage, &arguments)?;
        builder.set_name(result, &stage_output_wire_name(i + 1))?;
    }

    let ty = builder.get_type(result)?.clone();
    let mut output = pack_value(&mut builder, result, &ty)?;
    if boundary.flop_outputs {
        layers.insert(&mut builder, [("out", &mut output)])?;
    }
    builder.output_port("out", output)?;
    if let (Some(name), Some(valid)) = (&boundary.output_valid_signal, layers.valid) {
        builder.output_port(name, valid)?;
    }
    Ok(builder)
}

/// Tracks register numbering and the valid signal at the current pipeline
/// boundary.
struct RegisterLayers {
    next: u32,
    valid: Option<BValue>,
    reset: Option<BValue>,
}

impl RegisterLayers {
    /// Registers data on valid cycles and advances valid on every clock edge.
    fn insert<'a>(
        &mut self,
        builder: &mut BlockBuilder,
        data: impl IntoIterator<Item = (&'a str, &'a mut BValue)>,
    ) -> Result<(), BuilderError> {
        for (name, value) in data {
            let register = builder.register(
                &flop_reg_name(self.next, name),
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
        if let Some(valid) = self.valid {
            let register = builder.register(
                &valid_reg_name(self.next),
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
        self.next += 1;
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
