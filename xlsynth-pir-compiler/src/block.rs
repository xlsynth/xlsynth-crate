// SPDX-License-Identifier: Apache-2.0

//! Compile a synchronous PIR block once and clock independent simulations.
//!
//! [`CompiledBlock::step`] samples the ordered data inputs and old registers.
//! It returns pre-edge outputs and a new register state. Initialization is
//! explicit, and reset pins are ordinary required inputs. A failed step leaves
//! the original state usable. Traces and covers are collected by default.
//!
//! [`PreparedBlockInstance`] reuses native storage across edges. Both paths
//! support assertion, trace, cover and bounds-assumption events. Per-node
//! observers are available through the PIR interpreter.

mod prepared;
pub use prepared::{BlockInputBinding, BlockOutputPort, PreparedBlockInstance};

use thiserror::Error;
use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{
    BlockCycleEvents, BlockCycleResult, BlockEvalError, BlockEvalField, BlockState, PreparedBlock,
    RegisterInit,
};
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_eval;
use xlsynth_pir::ir_inline::package_with_reachable_callees;
use xlsynth_pir::ir_verify::verify_package;

use crate::{
    AssumptionFailureKind, CompilerError, ExecutionContext, ExecutionOptions, ExecutionResult,
    NodeSchedule, PirFunctionCompiler,
};

/// An error preparing or evaluating a block, or compiling or calling native
/// code.
#[derive(Debug, Error)]
pub enum CompiledBlockError {
    #[error(transparent)]
    Evaluation(#[from] BlockEvalError),
    #[error(transparent)]
    Compiler(#[from] CompilerError),
}

/// A validated cycle plan and its reachable functions, ready for JIT
/// compilation.
pub struct BlockCompilation {
    plan: PreparedBlock,
    package: Package,
}

impl BlockCompilation {
    /// Projects the named block and packages the functions called during a
    /// clock.
    pub fn prepare(package: &Package, top: &str) -> Result<Self, CompiledBlockError> {
        let plan = PreparedBlock::prepare(package, top)?;
        let package = package_with_reachable_callees(package, plan.function())
            .map_err(CompilerError::InvalidFunction)?;
        verify_package(&package).map_err(|e| CompilerError::InvalidFunction(e.to_string()))?;
        Ok(Self { plan, package })
    }

    /// Generates executable code for the previously prepared block and its
    /// callees.
    pub fn compile(self) -> Result<CompiledBlock, CompiledBlockError> {
        let compiler = PirFunctionCompiler::compile_functions(
            Some(&self.package),
            self.plan.function(),
            NodeSchedule::Interpreter,
        )?;
        Ok(CompiledBlock {
            plan: self.plan,
            compiler,
        })
    }
}

/// Native code and interface shared by independent simulations of one block.
pub struct CompiledBlock {
    plan: PreparedBlock,
    compiler: PirFunctionCompiler,
}

impl CompiledBlock {
    /// Prepares and compiles the selected block, including its modeled
    /// instances.
    pub fn compile(package: &Package, top: &str) -> Result<Self, CompiledBlockError> {
        BlockCompilation::prepare(package, top)?.compile()
    }

    /// Data inputs in port order, including wired reset pins. The clock is
    /// implicit.
    pub fn inputs(&self) -> &[BlockEvalField] {
        self.plan.inputs()
    }

    /// Sampled outputs in port order.
    pub fn outputs(&self) -> &[BlockEvalField] {
        self.plan.outputs()
    }

    /// Registers in declaration order, followed by child instances in preorder.
    pub fn registers(&self) -> &[BlockEvalField] {
        self.plan.registers()
    }

    /// Sets the register values before the first clock, independently of reset.
    pub fn initial_state(&self, init: RegisterInit) -> Result<BlockState, CompiledBlockError> {
        Ok(self.plan.initial_state(init)?)
    }

    /// Executes one edge, collecting all traces and covers.
    pub fn step(
        &self,
        state: &BlockState,
        inputs: &[IrValue],
    ) -> Result<BlockCycleResult, CompiledBlockError> {
        self.step_with_options(state, inputs, ExecutionOptions::collect_all())
    }

    /// Executes one edge with configurable observations. Assertion and bounds
    /// failures are always checked before register writes are resolved.
    pub fn step_with_options(
        &self,
        state: &BlockState,
        inputs: &[IrValue],
        options: ExecutionOptions,
    ) -> Result<BlockCycleResult, CompiledBlockError> {
        self.plan.validate_state(state)?;
        self.plan.validate_inputs(inputs)?;
        let args = inputs.iter().chain(state.values());
        let mut context =
            ExecutionContext::new_in_execution_order(self.compiler.metadata(), options);
        let result = self
            .compiler
            .run_ir_values_with_context(args, &mut context)?;
        let events = self.decode_events(result.events)?;
        Ok(self.plan.finish_cycle(state, result.value, events)?)
    }

    /// Translates native diagnostics into the shared PIR block result.
    fn decode_events(
        &self,
        events: ExecutionResult,
    ) -> Result<BlockCycleEvents, CompiledBlockError> {
        let assertions = events
            .assertion_failures
            .into_iter()
            .map(|a| ir_eval::AssertionFailure {
                message: a.message,
                label: a.label,
            })
            .collect();
        let assumptions = events
            .assumption_failures
            .into_iter()
            .map(|a| ir_eval::AssumptionFailure {
                node_text_id: a.node_text_id,
                kind: match a.kind {
                    AssumptionFailureKind::ArrayIndexOutOfBounds => {
                        ir_eval::AssumptionFailureKind::ArrayIndexOutOfBounds
                    }
                    AssumptionFailureKind::ArrayUpdateOutOfBounds => {
                        ir_eval::AssumptionFailureKind::ArrayUpdateOutOfBounds
                    }
                },
            })
            .collect();
        let traces = events
            .trace_messages
            .into_iter()
            .map(|t| ir_eval::TraceMessage {
                message: t.message,
                verbosity: t.verbosity,
            })
            .collect();
        let covers = events
            .cover_counts
            .into_iter()
            .map(|c| ir_eval::CoverCount {
                node_text_id: c.node_text_id,
                label: c.label,
                count: c.count,
            })
            .collect();
        Ok(self
            .plan
            .check_events(assertions, assumptions, traces, covers)?)
    }
}
