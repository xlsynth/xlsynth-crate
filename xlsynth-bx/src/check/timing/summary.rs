// SPDX-License-Identifier: Apache-2.0

//! An unannotated output keeps the timing operations on the paths that feed it.
//! For a registered `x` added to `r`, its summary advances input 0, then joins
//! that stage with input 1. Substituting the caller's stages checks both the
//! adder's input requirement and the resulting output stage.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use super::{Cycle, Join};
use crate::check::MAX_TYPE_SIZE;

/// The declared input stages shared by every instance of a fixed signature.
pub(super) type FixedInputs = Rc<[(String, u64)]>;

/// One constraint in a child output's timing summary.
#[derive(Clone)]
pub(super) enum Rule {
    Join(Join),
    Offset {
        name: Rc<str>,
        inputs: FixedInputs,
    },
    Translate {
        name: Rc<str>,
        port: String,
        local: u64,
    },
    RequireOffset,
}

impl Rule {
    /// Evaluate after all of this rule's operands have concrete timing
    /// information.
    fn evaluate(&self, args: &[Cycle]) -> Result<Cycle, String> {
        match self {
            Self::Join(join) => join.apply(args[0], args[1]),
            Self::Offset { name, inputs } => {
                let mut offset = None;
                let mut unknown = false;
                for ((port, local), actual) in inputs.iter().zip(args) {
                    match actual {
                        Cycle::At(stage) => {
                            let local = i128::from(*local);
                            if let Some(offset) = offset
                                && local + offset != *stage
                            {
                                return Err(format!(
                                    "instance '{name}' input '{port}': expected parent cycle {}, got {stage}",
                                    local + offset
                                ));
                            }
                            offset = Some(*stage - local);
                        }
                        Cycle::Unknown => unknown = true,
                        Cycle::Independent => { /* A constant fits any stage. */ }
                        Cycle::Input(_) | Cycle::Expr(_) => unreachable!("symbolic operand"),
                    }
                }
                Ok(if unknown {
                    Cycle::Unknown
                } else {
                    offset.map_or(Cycle::Unknown, Cycle::At)
                })
            }
            Self::Translate { name, port, local } => {
                let Cycle::At(offset) = args[0] else {
                    return Ok(Cycle::Unknown);
                };
                let stage = offset + i128::from(*local);
                u64::try_from(stage).map_err(|_| {
                    format!("instance '{name}.{port}': translated cycle {stage} is out of range")
                })?;
                Ok(Cycle::At(stage))
            }
            Self::RequireOffset => Ok(if matches!(args[0], Cycle::At(_)) {
                args[1]
            } else {
                Cycle::Unknown
            }),
        }
    }
}

/// Operands refer only to inputs or earlier instructions.
#[derive(Clone)]
struct Instruction {
    rule: Rule,
    args: Box<[Cycle]>,
}

/// Record timing operations when an input stage is symbolic, and evaluate them
/// immediately when the parent has concrete stages.
#[derive(Default)]
pub(super) struct Builder {
    instructions: Vec<Instruction>,
    work: usize,
}

impl Builder {
    /// Bound the work of expanding nested instances and exporting their ports.
    fn charge(&mut self, amount: usize) -> Result<(), String> {
        self.work = self.work.saturating_add(amount);
        if self.work > MAX_TYPE_SIZE {
            return Err("child timing summary exceeds the elaboration limit".into());
        }
        Ok(())
    }

    /// A concrete caller checks the rule now. A symbolic caller saves it.
    pub(super) fn apply(&mut self, rule: Rule, args: &[Cycle]) -> Result<Cycle, String> {
        if !args
            .iter()
            .any(|arg| matches!(arg, Cycle::Input(_) | Cycle::Expr(_)))
        {
            return rule.evaluate(args);
        }
        if let Rule::Join(join) = &rule
            && !matches!(join, Join::Register(_))
        {
            match args {
                [left, right] if left == right => return Ok(*left),
                [Cycle::Independent, stage] | [stage, Cycle::Independent] => return Ok(*stage),
                _ => { /* Both operand stages must be checked at the call. */ }
            }
        }
        self.charge(1 + args.len())?;
        let index = self.instructions.len();
        self.instructions.push(Instruction {
            rule,
            args: args.into(),
        });
        Ok(Cycle::Expr(index))
    }

    /// Keep just the operands and instructions reachable from one physical
    /// port.
    pub(super) fn summary(&mut self, result: Cycle) -> Result<Summary, String> {
        let mut needed = BTreeSet::new();
        let mut inputs = BTreeSet::new();
        let mut pending = vec![result];
        while let Some(value) = pending.pop() {
            match value {
                Cycle::Expr(i) if needed.insert(i) => {
                    self.charge(1 + self.instructions[i].args.len())?;
                    pending.extend(self.instructions[i].args.iter().copied())
                }
                Cycle::Input(i) => {
                    inputs.insert(i);
                }
                _ => { /* A concrete stage or an instruction already visited. */ }
            }
        }
        let ids: BTreeMap<_, _> = needed.iter().copied().zip(0..).collect();
        let inputs: Vec<_> = inputs.into_iter().collect();
        let remap = |value| match value {
            Cycle::Expr(i) => Cycle::Expr(ids[&i]),
            Cycle::Input(i) => Cycle::Input(inputs.binary_search(&i).expect("reachable input")),
            other => other,
        };
        let instructions = needed
            .into_iter()
            .map(|i| {
                let op = &self.instructions[i];
                Instruction {
                    rule: op.rule.clone(),
                    args: op.args.iter().copied().map(remap).collect(),
                }
            })
            .collect();
        let result = remap(result);
        Ok(Summary {
            inputs,
            instructions,
            result,
        })
    }
}

/// A reusable relation between a single child output and its contributing
/// inputs.
pub(super) struct Summary {
    pub inputs: Vec<usize>,
    instructions: Vec<Instruction>,
    result: Cycle,
}

impl Summary {
    /// Substitute caller inputs. A symbolic caller gets a composed relation.
    pub(super) fn instantiate(
        &self,
        inputs: &[Cycle],
        builder: &mut Builder,
    ) -> Result<Cycle, String> {
        let resolve = |value, values: &[Cycle]| match value {
            Cycle::Expr(i) => values[i],
            Cycle::Input(i) => inputs[i],
            other => other,
        };
        let mut values = Vec::with_capacity(self.instructions.len());
        let mut args = Vec::new();
        for op in &self.instructions {
            builder.charge(1 + op.args.len())?;
            args.clear();
            args.extend(op.args.iter().map(|v| resolve(*v, &values)));
            values.push(builder.apply(op.rule.clone(), &args)?);
        }
        Ok(resolve(self.result, &values))
    }
}
