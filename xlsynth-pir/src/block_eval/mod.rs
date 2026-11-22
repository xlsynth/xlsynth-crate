// SPDX-License-Identifier: Apache-2.0

//! Cycle-by-cycle interpretation of combinational and synchronous PIR blocks.
//!
//! Prepare a block once and select its initial register values explicitly.
//! Each step samples the data inputs and old register values, produces the
//! pre-edge outputs, and returns the register state for the following cycle.
//!
//! ```
//! use xlsynth_pir::{IrValue, block_eval::{BlockEvaluator, RegisterInit}, ir_parser::Parser};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ir = r#"package example
//! top block delay(clk: clock, x: bits[8], y: bits[8]) {
//!   reg q(bits[8])
//!   x: bits[8] = input_port(name=x, id=1)
//!   old: bits[8] = register_read(register=q, id=2)
//!   w: () = register_write(x, register=q, id=3)
//!   y: () = output_port(old, name=y, id=4)
//! }"#;
//! let package = Parser::new(ir).parse_package().expect("valid example IR");
//! let eval = BlockEvaluator::prepare(&package, "delay")?;
//! let state = eval.initial_state(RegisterInit::AllZeros)?;
//! let sample = IrValue::make_ubits(8, 17)?;
//! let cycle = eval.step(&state, std::slice::from_ref(&sample))?;
//! assert_eq!(cycle.outputs, [IrValue::make_ubits(8, 0)?]);
//! assert_eq!(cycle.next_state.values(), &[sample]);
//! # Ok(())
//! # }
//! ```

mod observe;
mod plan;
mod prepare;
#[doc(hidden)]
pub mod test_fixtures;

pub use plan::{BlockCycleEvents, BlockRegisterSource, PreparedBlock};
use std::fmt;
use std::sync::Arc;

use self::observe::BlockObserver;
use crate::IrValue;
use crate::ir::{Package, Type};
use crate::ir_eval::{
    AssertionFailure, AssumptionFailureKind, EvalObserver, FnEvalResult, TraceMessage,
    eval_fn_in_package_with_observer,
};
use crate::ir_verify::VerifyError;

/// A data port or qualified register in the evaluator's ordered interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockEvalField {
    pub name: String,
    pub ty: Type,
}

/// Selects the value of every register before the first clock edge.
#[derive(Debug, Clone)]
pub enum RegisterInit {
    AllZeros,
    /// Generate uniformly random bits in register layout order.
    Random {
        seed: u64,
    },
    /// Provide one exactly typed value per entry in
    /// `BlockEvaluator::registers`.
    Supplied(Vec<IrValue>),
}

/// Register values belonging to one prepared evaluator. Clone to fork a run.
#[derive(Debug, Clone)]
pub struct BlockState {
    owner: Arc<()>,
    values: Vec<IrValue>,
}

impl BlockState {
    /// Values in the evaluator's register order, suitable for checkpointing.
    pub fn values(&self) -> &[IrValue] {
        &self.values
    }
}

/// A block instance and original IR text ID, or a static site in a called
/// function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockEvalSite {
    /// Empty for top-level block nodes and package function nodes.
    pub instance: String,
    pub node_text_id: usize,
}

/// Cover hits accumulated at one site during a single cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockCoverCount {
    pub site: BlockEvalSite,
    pub label: String,
    pub count: u64,
}

/// One failed bounds assumption encountered during the cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockAssumptionFailure {
    pub site: BlockEvalSite,
    pub kind: AssumptionFailureKind,
}

/// Outputs before the edge, updated register values, and events from one step.
#[derive(Debug, Clone)]
pub struct BlockCycleResult {
    pub outputs: Vec<IrValue>,
    pub next_state: BlockState,
    pub trace_messages: Vec<TraceMessage>,
    pub cover_counts: Vec<BlockCoverCount>,
}

/// Runtime contract failures and other diagnostics from the same evaluation.
#[derive(Debug, Clone)]
pub struct BlockRuntimeFailure {
    pub assertion_failures: Vec<AssertionFailure>,
    pub assumption_failures: Vec<BlockAssumptionFailure>,
    pub trace_messages: Vec<TraceMessage>,
    pub cover_counts: Vec<BlockCoverCount>,
}

/// Identifies the ordered values supplied at the public API boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockValueKind {
    Input,
    Register,
}

impl fmt::Display for BlockValueKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Input => "input",
            Self::Register => "register",
        })
    }
}

/// Preparation, interface or evaluation failure. A failed step leaves its input
/// state intact.
#[derive(Debug)]
pub enum BlockEvalError {
    InvalidPackage(Box<VerifyError>),
    Preparation(String),
    UnknownInstance(String),
    ValueCount {
        kind: BlockValueKind,
        expected: usize,
        actual: usize,
    },
    ValueType {
        kind: BlockValueKind,
        name: String,
        expected: Type,
        actual: Type,
    },
    ForeignState,
    /// The evaluated value has a different type than the prepared function
    /// returns.
    CycleResultType {
        expected: Type,
        actual: Type,
    },
    ConflictingWrites {
        register: String,
        first: usize,
        second: usize,
    },
    Runtime(Box<BlockRuntimeFailure>),
}

impl fmt::Display for BlockEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPackage(err) => fmt::Display::fmt(err, f),
            Self::Preparation(message) => f.write_str(message),
            Self::UnknownInstance(path) => write!(f, "unknown block instance '{path}'"),
            Self::ValueCount {
                kind,
                expected,
                actual,
            } => write!(f, "expected {expected} {kind} values, got {actual}"),
            Self::ValueType {
                kind,
                name,
                expected,
                actual,
            } => write!(f, "{kind} '{name}': expected {expected}, got {actual}"),
            Self::CycleResultType { expected, actual } => write!(
                f,
                "cycle function expected result of type {expected}, got {actual}"
            ),
            Self::ForeignState => {
                f.write_str("register state belongs to a different block evaluator")
            }
            Self::ConflictingWrites {
                register,
                first,
                second,
            } => write!(
                f,
                "register '{register}' has multiple enabled writes (IR node IDs {first} and {second})"
            ),
            Self::Runtime(failure) => {
                write!(f, "block runtime failure")?;
                for assertion in &failure.assertion_failures {
                    write!(
                        f,
                        "\n  assertion '{}': {}",
                        assertion.label, assertion.message
                    )?;
                }
                for assumption in &failure.assumption_failures {
                    write!(f, "\n  {:?} at ", assumption.kind)?;
                    if !assumption.site.instance.is_empty() {
                        write!(f, "{} ", assumption.site.instance)?;
                    }
                    write!(f, "node {}", assumption.site.node_text_id)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for BlockEvalError {}

/// An interpreter for one prepared block and its package functions.
pub struct BlockEvaluator<'p> {
    package: &'p Package,
    plan: PreparedBlock,
}

impl<'p> BlockEvaluator<'p> {
    /// Validate and prepare the selected block and its reachable instances.
    pub fn prepare(package: &'p Package, top: &str) -> Result<Self, BlockEvalError> {
        let plan = PreparedBlock::prepare(package, top)?;
        Ok(Self { package, plan })
    }

    /// Data inputs in port order, including reset pins. The clock is implicit.
    pub fn inputs(&self) -> &[BlockEvalField] {
        self.plan.inputs()
    }

    /// Outputs in port order.
    pub fn outputs(&self) -> &[BlockEvalField] {
        self.plan.outputs()
    }

    /// Register layout in declaration order, followed by child instances in
    /// preorder.
    pub fn registers(&self) -> &[BlockEvalField] {
        self.plan.registers()
    }

    /// Initialize registers independently of their synchronous reset values.
    pub fn initial_state(&self, init: RegisterInit) -> Result<BlockState, BlockEvalError> {
        self.plan.initial_state(init)
    }

    /// Sample one cycle and commit all enabled register writes simultaneously.
    pub fn step(
        &self,
        state: &BlockState,
        inputs: &[IrValue],
    ) -> Result<BlockCycleResult, BlockEvalError> {
        self.step_impl(state, inputs, None)
    }

    /// Step while observing nodes of the top block using their original node
    /// refs and IDs.
    pub fn step_with_observer(
        &self,
        state: &BlockState,
        inputs: &[IrValue],
        observer: &mut dyn EvalObserver,
    ) -> Result<BlockCycleResult, BlockEvalError> {
        self.step_with_instance_observer(state, inputs, "", observer)
    }

    /// Observe one reachable instance by its qualified path. An empty path
    /// selects the top.
    pub fn step_with_instance_observer(
        &self,
        state: &BlockState,
        inputs: &[IrValue],
        instance: &str,
        observer: &mut dyn EvalObserver,
    ) -> Result<BlockCycleResult, BlockEvalError> {
        if !instance.is_empty()
            && !self
                .plan
                .prepared
                .origins
                .iter()
                .flatten()
                .any(|o| &*o.instance == instance)
        {
            return Err(BlockEvalError::UnknownInstance(instance.to_owned()));
        }
        let mut observer = BlockObserver {
            instance,
            prepared: &self.plan.prepared,
            inner: observer,
        };
        self.step_impl(state, inputs, Some(&mut observer))
    }

    /// Evaluate the prepared cycle function and resolve its sampled register
    /// writes.
    fn step_impl(
        &self,
        state: &BlockState,
        inputs: &[IrValue],
        observer: Option<&mut dyn EvalObserver>,
    ) -> Result<BlockCycleResult, BlockEvalError> {
        self.plan.validate_state(state)?;
        self.plan.validate_inputs(inputs)?;
        let args: Vec<_> = inputs.iter().chain(state.values()).cloned().collect();
        let result =
            eval_fn_in_package_with_observer(self.package, self.plan.function(), &args, observer);
        let (samples, events) = match result {
            FnEvalResult::Success(success) => (
                success.value,
                self.plan.check_events(
                    Vec::new(),
                    Vec::new(),
                    success.trace_messages,
                    success.cover_counts,
                )?,
            ),
            FnEvalResult::Failure(failure) => (
                failure.value,
                self.plan.check_events(
                    failure.assertion_failures,
                    failure.assumption_failures,
                    failure.trace_messages,
                    failure.cover_counts,
                )?,
            ),
        };
        self.plan.finish_cycle(state, samples, events)
    }
}
