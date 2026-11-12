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
mod prepare;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use self::observe::BlockObserver;
use self::prepare::{Prepared, RegisterUpdate};
use crate::IrValue;
use crate::ir::{NodePayload, Package, Type};
use crate::ir_eval::{
    AssertionFailure, AssumptionFailureKind, EvalObserver, FnEvalResult, TraceMessage,
    eval_fn_in_package_with_observer,
};
use crate::ir_value_utils::zero_ir_value_for_type;
use crate::ir_verify::{VerifyError, verify_package};
use crate::random_inputs::generate_uniform_value_with_rng;

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

/// A reusable execution plan and register layout for one elaborated block.
pub struct BlockEvaluator<'p> {
    package: &'p Package,
    prepared: Prepared,
    owner: Arc<()>,
    origin_by_id: HashMap<usize, usize>,
}

impl<'p> BlockEvaluator<'p> {
    /// Validate the package and prepare the selected block and its block
    /// instances.
    pub fn prepare(package: &'p Package, top: &str) -> Result<Self, BlockEvalError> {
        verify_package(package).map_err(|err| BlockEvalError::InvalidPackage(Box::new(err)))?;
        let prepared = prepare::prepare(package, top).map_err(BlockEvalError::Preparation)?;
        let origin_by_id = prepared
            .function
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                matches!(
                    node.payload,
                    NodePayload::Cover { .. }
                        | NodePayload::ArrayIndex {
                            assumed_in_bounds: true,
                            ..
                        }
                        | NodePayload::ArrayUpdate {
                            assumed_in_bounds: true,
                            ..
                        }
                )
            })
            .map(|(i, node)| (node.text_id, i))
            .collect();
        Ok(Self {
            package,
            prepared,
            owner: Arc::new(()),
            origin_by_id,
        })
    }

    /// Data inputs in port order, including reset pins. The clock is implicit.
    pub fn inputs(&self) -> &[BlockEvalField] {
        &self.prepared.inputs
    }

    /// Outputs in port order.
    pub fn outputs(&self) -> &[BlockEvalField] {
        &self.prepared.outputs
    }

    /// Register layout in declaration order, followed by child instances in
    /// preorder.
    pub fn registers(&self) -> &[BlockEvalField] {
        &self.prepared.registers
    }

    /// Create register state independent of each register's synchronous reset
    /// value.
    pub fn initial_state(&self, init: RegisterInit) -> Result<BlockState, BlockEvalError> {
        let values = match init {
            RegisterInit::AllZeros => self
                .registers()
                .iter()
                .map(|r| zero_ir_value_for_type(&r.ty))
                .collect(),
            RegisterInit::Random { seed } => {
                let mut rng = Pcg64Mcg::seed_from_u64(seed);
                self.registers()
                    .iter()
                    .map(|r| generate_uniform_value_with_rng(&mut rng, &r.ty))
                    .collect()
            }
            RegisterInit::Supplied(values) => {
                validate_values(BlockValueKind::Register, self.registers(), &values)?;
                values
            }
        };
        Ok(BlockState {
            owner: self.owner.clone(),
            values,
        })
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
            prepared: &self.prepared,
            inner: observer,
        };
        self.step_impl(state, inputs, Some(&mut observer))
    }

    /// Resolve a projected event back to its instance, retaining package IDs of
    /// called functions.
    fn site(&self, id: usize) -> BlockEvalSite {
        match self
            .origin_by_id
            .get(&id)
            .and_then(|i| self.prepared.origins[*i].as_ref())
        {
            Some(origin) => BlockEvalSite {
                instance: origin.instance.to_string(),
                node_text_id: origin.text_id,
            },
            None => BlockEvalSite {
                instance: String::new(),
                node_text_id: id,
            },
        }
    }

    fn cover_counts(&self, counts: Vec<crate::ir_eval::CoverCount>) -> Vec<BlockCoverCount> {
        counts
            .into_iter()
            .map(|c| BlockCoverCount {
                site: self.site(c.node_text_id),
                label: c.label,
                count: c.count,
            })
            .collect()
    }

    /// Evaluate the cycle's dataflow once, then resolve writes from those
    /// sampled values.
    fn step_impl(
        &self,
        state: &BlockState,
        inputs: &[IrValue],
        observer: Option<&mut dyn EvalObserver>,
    ) -> Result<BlockCycleResult, BlockEvalError> {
        if !Arc::ptr_eq(&self.owner, &state.owner) {
            return Err(BlockEvalError::ForeignState);
        }
        validate_values(BlockValueKind::Input, self.inputs(), inputs)?;
        let args: Vec<_> = inputs.iter().chain(&state.values).cloned().collect();
        let eval = eval_fn_in_package_with_observer(
            self.package,
            &self.prepared.function,
            &args,
            observer,
        );
        let success = match eval {
            FnEvalResult::Failure(failure) => {
                return Err(BlockEvalError::Runtime(Box::new(BlockRuntimeFailure {
                    assertion_failures: failure.assertion_failures,
                    assumption_failures: failure
                        .assumption_failures
                        .into_iter()
                        .map(|a| BlockAssumptionFailure {
                            site: self.site(a.node_text_id),
                            kind: a.kind,
                        })
                        .collect(),
                    trace_messages: failure.trace_messages,
                    cover_counts: self.cover_counts(failure.cover_counts),
                })));
            }
            FnEvalResult::Success(success) => success,
        };
        let IrValue::Tuple(results) = success.value else {
            unreachable!("prepared block function returns a tuple")
        };
        let mut values = state.values.clone();
        for ((value, reg), update) in values
            .iter_mut()
            .zip(self.registers())
            .zip(&self.prepared.updates)
        {
            if let Some(next) = next_register_value(&reg.name, update, &results)? {
                value.clone_from(next);
            }
        }
        Ok(BlockCycleResult {
            outputs: results[..self.outputs().len()].to_vec(),
            next_state: BlockState {
                owner: self.owner.clone(),
                values,
            },
            trace_messages: success.trace_messages,
            cover_counts: self.cover_counts(success.cover_counts),
        })
    }
}

/// Reset takes priority only when wired to this register; otherwise one enabled
/// write may win.
fn next_register_value<'a>(
    name: &str,
    update: &'a RegisterUpdate,
    samples: &'a [IrValue],
) -> Result<Option<&'a IrValue>, BlockEvalError> {
    if let Some((index, active_low, reset_value)) = &update.reset {
        if samples[*index]
            .to_bool()
            .expect("verified one-bit reset operand")
            != *active_low
        {
            return Ok(Some(reset_value));
        }
    }
    let mut enabled = None;
    for write in &update.writes {
        if write.enable.is_none_or(|index| {
            samples[index]
                .to_bool()
                .expect("verified one-bit load enable")
        }) {
            if let Some((first, _)) = enabled {
                return Err(BlockEvalError::ConflictingWrites {
                    register: name.to_owned(),
                    first,
                    second: write.node_id,
                });
            }
            enabled = Some((write.node_id, &samples[write.data]));
        }
    }
    Ok(enabled.map(|(_, data)| data))
}

/// Check both arity and exact structural type at the public boundary.
fn validate_values(
    kind: BlockValueKind,
    fields: &[BlockEvalField],
    values: &[IrValue],
) -> Result<(), BlockEvalError> {
    if fields.len() != values.len() {
        return Err(BlockEvalError::ValueCount {
            kind,
            expected: fields.len(),
            actual: values.len(),
        });
    }
    for (field, value) in fields.iter().zip(values) {
        let actual = value.type_();
        if actual != field.ty {
            return Err(BlockEvalError::ValueType {
                kind,
                name: field.name.clone(),
                expected: field.ty.clone(),
                actual,
            });
        }
    }
    Ok(())
}
