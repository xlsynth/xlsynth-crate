// SPDX-License-Identifier: Apache-2.0

use std::{mem, ptr};

use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{
    BlockCycleEvents, BlockEvalError, BlockRegisterSource, BlockState, BlockValueKind, RegisterInit,
};

use super::{CompiledBlock, CompiledBlockError};
use crate::{
    CompilerError, ExecutionContext, ExecutionOptions, NativeTupleFieldLayout, NativeValueLayout,
    NativeValueStorage, ScalarLayout, read_ir_value_from_native, write_ir_value_to_native,
};

/// Binds a caller's complete input order to one compiled block's data ports.
pub struct BlockInputBinding<'block> {
    block: &'block CompiledBlock,
    indices: Vec<usize>,
}

/// Identifies a single output on one compiled block, shared by its instances.
pub struct BlockOutputPort<'block> {
    block: &'block CompiledBlock,
    index: usize,
}

/// Native storage for independently clocking one instance of a compiled block.
///
/// Inputs must be supplied on every edge, including reset. On a successful
/// tick, outputs sample the state before the edge. A failed tick preserves both
/// registers and the outputs from the last successful tick. Output accessors
/// produce owned values that remain valid across later ticks and restores.
pub struct PreparedBlockInstance<'block> {
    block: &'block CompiledBlock,
    inputs: Vec<NativeValueStorage>,
    registers: Vec<NativeValueStorage>,
    next_registers: Vec<NativeValueStorage>,
    resets: Vec<Option<NativeValueStorage>>,
    samples: NativeValueStorage,
    last_samples: NativeValueStorage,
    fields: &'block [NativeTupleFieldLayout],
    scratch: Vec<u64>,
    pointers: Vec<*const u8>,
    context: ExecutionContext<'block>,
    has_outputs: bool,
}

impl CompiledBlock {
    /// Creates a reusable native instance collecting all traces and coverage.
    pub fn prepare_instance(
        &self,
        init: RegisterInit,
    ) -> Result<PreparedBlockInstance<'_>, CompiledBlockError> {
        self.prepare_instance_with_options(init, ExecutionOptions::collect_all())
    }

    /// Creates a reusable native instance with explicit observation options.
    pub fn prepare_instance_with_options(
        &self,
        init: RegisterInit,
        options: ExecutionOptions,
    ) -> Result<PreparedBlockInstance<'_>, CompiledBlockError> {
        let state = self.plan.initial_state(init)?;
        let compiler = &self.compiler;
        let (inputs, regs) = compiler.param_layouts().split_at(self.plan.inputs().len());
        let registers = regs
            .iter()
            .zip(state.values())
            .map(|(layout, value)| NativeValueStorage::from_ir_value(layout, value))
            .collect::<Result<_, _>>()?;
        let resets = regs
            .iter()
            .zip(self.plan.reset_values())
            .map(|(layout, value)| {
                value
                    .map(|v| NativeValueStorage::from_ir_value(layout, v))
                    .transpose()
            })
            .collect::<Result<_, _>>()?;
        let NativeValueLayout::Tuple { fields, .. } = compiler.result_layout() else {
            return Err(
                CompilerError::InvalidFunction("block cycle must return a tuple".into()).into(),
            );
        };
        Ok(PreparedBlockInstance {
            block: self,
            inputs: inputs.iter().map(NativeValueStorage::zeroed).collect(),
            registers,
            next_registers: regs.iter().map(NativeValueStorage::zeroed).collect(),
            resets,
            samples: NativeValueStorage::zeroed(compiler.result_layout()),
            last_samples: NativeValueStorage::zeroed(compiler.result_layout()),
            fields,
            scratch: vec![0; compiler.scratch_byte_count().div_ceil(size_of::<u64>())],
            pointers: Vec::with_capacity(compiler.param_layouts().len()),
            context: ExecutionContext::new_in_execution_order(compiler.metadata(), options),
            has_outputs: false,
        })
    }

    /// Binds one name per data input. Subsequent ticks take values in this
    /// order.
    pub fn bind_inputs(&self, names: &[&str]) -> Result<BlockInputBinding<'_>, CompiledBlockError> {
        let inputs = self.plan.inputs();
        let mut indices = Vec::with_capacity(names.len());
        let mut seen = vec![false; inputs.len()];
        for name in names {
            let index = inputs
                .iter()
                .position(|port| port.name == *name)
                .ok_or_else(|| {
                    CompilerError::InvalidArgument(format!("unknown block input '{name}'"))
                })?;
            if mem::replace(&mut seen[index], true) {
                return Err(CompilerError::InvalidArgument(format!(
                    "duplicate block input '{name}'"
                ))
                .into());
            }
            indices.push(index);
        }
        if let Some(index) = seen.iter().position(|seen| !seen) {
            return Err(CompilerError::InvalidArgument(format!(
                "missing block input '{}'",
                inputs[index].name
            ))
            .into());
        }
        Ok(BlockInputBinding {
            block: self,
            indices,
        })
    }

    /// Binds a named output once for use with any instance of this compiled
    /// block.
    pub fn bind_output(&self, name: &str) -> Result<BlockOutputPort<'_>, CompiledBlockError> {
        let index = self
            .plan
            .outputs()
            .iter()
            .position(|port| port.name == name)
            .ok_or_else(|| {
                CompilerError::InvalidArgument(format!("unknown block output '{name}'"))
            })?;
        Ok(BlockOutputPort { block: self, index })
    }
}

impl PreparedBlockInstance<'_> {
    /// Advances one edge with inputs in the block's declared order.
    pub fn tick(&mut self, inputs: &[IrValue]) -> Result<BlockCycleEvents, CompiledBlockError> {
        self.tick_in_order(inputs, 0..self.inputs.len())
    }

    /// Advances one edge with inputs in the order established by the binding.
    pub fn tick_bound(
        &mut self,
        binding: &BlockInputBinding<'_>,
        inputs: &[IrValue],
    ) -> Result<BlockCycleEvents, CompiledBlockError> {
        if !ptr::eq(self.block, binding.block) {
            return Err(CompilerError::InvalidArgument(
                "input binding belongs to another compiled block".into(),
            )
            .into());
        }
        self.tick_in_order(inputs, binding.indices.iter().copied())
    }

    /// Stages input values, executes into private samples, then commits
    /// registers and outputs.
    fn tick_in_order(
        &mut self,
        inputs: &[IrValue],
        order: impl Iterator<Item = usize>,
    ) -> Result<BlockCycleEvents, CompiledBlockError> {
        if inputs.len() != self.inputs.len() {
            return Err(BlockEvalError::ValueCount {
                kind: BlockValueKind::Input,
                expected: self.inputs.len(),
                actual: inputs.len(),
            }
            .into());
        }
        let compiler = &self.block.compiler;
        for (value, index) in inputs.iter().zip(order) {
            self.block.plan.inputs()[index].validate(BlockValueKind::Input, value)?;
            // SAFETY: the storage was sized and aligned for this exact layout,
            // and the port validation checks the complete PIR type.
            unsafe {
                write_ir_value_to_native(
                    self.inputs[index].as_mut_ptr(),
                    &compiler.param_layouts()[index],
                    value,
                )?;
            }
        }
        self.pointers.clear();
        self.pointers.extend(
            self.inputs
                .iter()
                .chain(&self.registers)
                .map(NativeValueStorage::as_ptr),
        );
        self.context.clear();
        // SAFETY: argument storage matches the input/register layouts, sample
        // storage matches the result layout, and the u64 scratch slab is
        // aligned for all supported native types. Their allocations are
        // disjoint and live for the entire call.
        unsafe {
            compiler.run_native_with_scratch_and_context(
                &self.pointers,
                self.samples.as_mut_ptr(),
                self.scratch.as_mut_ptr().cast(),
                self.scratch.len() * size_of::<u64>(),
                &mut self.context,
            )?;
        }
        let events = self.block.decode_events(self.context.result())?;
        self.block.plan.resolve_writes(
            |index| {
                let field = &self.fields[index];
                debug_assert_eq!(
                    field.layout.as_scalar(),
                    Some(ScalarLayout {
                        bit_count: 1,
                        byte_count: 1
                    })
                );
                // SAFETY: all sampled enables and resets are verified bits[1]
                // fields of the immutable native result tuple.
                unsafe { self.samples.as_ptr().add(field.offset).read() & 1 != 0 }
            },
            |reg, source| {
                let destination = &mut self.next_registers[reg];
                match source {
                    BlockRegisterSource::Hold => destination
                        .words
                        .copy_from_slice(&self.registers[reg].words),
                    BlockRegisterSource::Reset(_) => {
                        let reset = self.resets[reg]
                            .as_ref()
                            .expect("verified reset has prepared storage");
                        destination.words.copy_from_slice(&reset.words);
                    }
                    BlockRegisterSource::Sample(index) => {
                        let field = &self.fields[index];
                        // SAFETY: preparation verifies write data has the
                        // register's type. The sampled field and independently
                        // allocated next-register storage cannot overlap.
                        unsafe {
                            ptr::copy_nonoverlapping(
                                self.samples.as_ptr().add(field.offset),
                                destination.as_mut_ptr(),
                                field.layout.byte_count(),
                            );
                        }
                    }
                }
            },
        )?;
        mem::swap(&mut self.registers, &mut self.next_registers);
        mem::swap(&mut self.samples, &mut self.last_samples);
        self.has_outputs = true;
        Ok(events)
    }

    /// Reads an owned, typed output from the last successful edge.
    pub fn output(&self, index: usize) -> Result<IrValue, CompiledBlockError> {
        if index >= self.block.plan.outputs().len() {
            return Err(CompilerError::InvalidArgument(format!(
                "invalid block output index {index}"
            ))
            .into());
        }
        if !self.has_outputs {
            return Err(CompilerError::InvalidArgument(
                "block outputs require a successful tick".into(),
            )
            .into());
        }
        let field = &self.fields[index];
        // SAFETY: output fields prefix the native result tuple, whose storage
        // remains intact until the next successful tick or restore.
        Ok(unsafe {
            read_ir_value_from_native(self.last_samples.as_ptr().add(field.offset), &field.layout)?
        })
    }

    /// Reads an owned output through a port previously bound by this block.
    pub fn read_output(&self, port: &BlockOutputPort<'_>) -> Result<IrValue, CompiledBlockError> {
        if !ptr::eq(self.block, port.block) {
            return Err(CompilerError::InvalidArgument(
                "output port belongs to another compiled block".into(),
            )
            .into());
        }
        self.output(port.index)
    }

    /// Takes an independent typed snapshot of the committed registers.
    pub fn snapshot(&self) -> Result<BlockState, CompiledBlockError> {
        let layouts = &self.block.compiler.param_layouts()[self.inputs.len()..];
        let values = self
            .registers
            .iter()
            .zip(layouts)
            .map(|(storage, layout)| storage.to_ir_value(layout))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self
            .block
            .plan
            .initial_state(RegisterInit::Supplied(values))?)
    }

    /// Restores registers from this compiled block and clears sampled outputs.
    pub fn restore(&mut self, state: &BlockState) -> Result<(), CompiledBlockError> {
        self.block.plan.validate_state(state)?;
        let layouts = &self.block.compiler.param_layouts()[self.inputs.len()..];
        for ((storage, layout), value) in self
            .next_registers
            .iter_mut()
            .zip(layouts)
            .zip(state.values())
        {
            // SAFETY: state ownership guarantees an exact register layout,
            // including arity and individual value types. This bank remains
            // private until every register has been staged.
            unsafe { write_ir_value_to_native(storage.as_mut_ptr(), layout, value)? };
        }
        mem::swap(&mut self.registers, &mut self.next_registers);
        self.has_outputs = false;
        Ok(())
    }
}
