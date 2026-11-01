// SPDX-License-Identifier: Apache-2.0
//! Evaluate a closed scalar expression with at most one bits-valued input.
use super::eval_pure_if_supported;
use crate::IrValue;
use crate::ir::{self, NodePayload as P, NodeRef};
use crate::ir_utils::operands;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Prices unique nodes and operand occurrences separately. A repeated operand
/// is one value, but each occurrence is still read by the primitive kernel.
pub struct Limits {
    pub nodes: usize,
    pub edges: usize,
    pub value_bits: usize,
    pub input_bits: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Budget,
    Work,
    Effect,
    InputDomain,
    Unsupported,
    Cycle,
}

struct Step<'a> {
    node: &'a ir::Node,
    inputs: SmallVec<[usize; 3]>,
}

/// A local evaluation order over the original nodes. Other parameters, users
/// of shared intermediates, and the storage order of the host function do not
/// affect the expression's value.
pub struct BitsCone<'a> {
    steps: Vec<Step<'a>>,
    input: Option<(NodeRef, usize)>,
    edges: usize,
}

impl<'a> BitsCone<'a> {
    /// Charge discovery work on both success and rejection, so a function with
    /// many individually ineligible roots cannot defeat the caller's total work
    /// bound.
    pub fn compile(
        f: &'a ir::Fn,
        root: NodeRef,
        limits: Limits,
        remaining_work: &mut usize,
    ) -> Result<Self, Error> {
        let mut plan = Self {
            steps: Vec::new(),
            input: None,
            edges: 0,
        };
        // Some(slot) is complete; None is on the current DFS path.
        let mut slots = HashMap::<NodeRef, Option<usize>>::new();
        let mut stack = vec![(root, false)];
        while let Some((r, finish)) = stack.pop() {
            let node = f.nodes.get(r.index).ok_or(Error::Unsupported)?;
            if finish {
                let inputs = operands(&node.payload)
                    .iter()
                    .map(|key| slots[key].expect("postorder has evaluated all operands"))
                    .collect();
                slots.insert(r, Some(plan.steps.len()));
                plan.steps.push(Step { node, inputs });
                continue;
            }
            if let Some(slot) = slots.get(&r) {
                if slot.is_none() {
                    return Err(Error::Cycle);
                }
                continue;
            }
            *remaining_work = remaining_work.checked_sub(1).ok_or(Error::Work)?;
            if slots.len() >= limits.nodes {
                return Err(Error::Budget);
            }
            slots.insert(r, None);
            if matches!(
                node.payload,
                P::Assert { .. }
                    | P::Trace { .. }
                    | P::Cover { .. }
                    | P::RegisterRead { .. }
                    | P::RegisterWrite { .. }
            ) {
                return Err(Error::Effect);
            }
            let ir::Type::Bits(width) = f.get_node_ty(r) else {
                return Err(Error::Unsupported);
            };
            if *width > limits.value_bits {
                return Err(Error::Budget);
            }
            match &node.payload {
                P::Param => {
                    if *width > limits.input_bits || plan.input.replace((r, *width)).is_some() {
                        return Err(Error::InputDomain);
                    }
                }
                P::Nary(_, args) if args.len() > limits.edges - plan.edges => {
                    return Err(Error::Budget);
                }
                P::ExtNaryAdd { terms, .. } if terms.len() > limits.edges - plan.edges => {
                    return Err(Error::Budget);
                }
                P::Literal(_)
                | P::Binop(..)
                | P::Unop(..)
                | P::Nary(..)
                | P::BitSlice { .. }
                | P::SignExt { .. }
                | P::ZeroExt { .. }
                | P::ExtNaryAdd { .. } => (),
                _ => return Err(Error::Unsupported),
            }
            let deps = operands(&node.payload);
            if deps.len() > limits.edges - plan.edges {
                return Err(Error::Budget);
            }
            *remaining_work = remaining_work.checked_sub(deps.len()).ok_or(Error::Work)?;
            plan.edges += deps.len();
            stack.push((r, true));
            stack.extend(deps.into_iter().rev().map(|dep| (dep, false)));
        }
        Ok(plan)
    }

    pub fn input(&self) -> Option<(NodeRef, usize)> {
        self.input
    }

    /// Number of node evaluations plus operand reads for one input assignment.
    pub fn work(&self) -> usize {
        self.steps.len() + self.edges
    }

    pub fn evaluate(&self, input: Option<&IrValue>) -> Result<IrValue, Error> {
        if input.map(|v| v.as_bits().ok().map(|b| b.get_bit_count()))
            != self.input.map(|(_, width)| Some(width))
        {
            return Err(Error::InputDomain);
        }
        let mut values = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let value = if matches!(step.node.payload, P::Param) {
                input.expect("checked source binding").clone()
            } else {
                let args: SmallVec<[&IrValue; 8]> =
                    step.inputs.iter().map(|&i| &values[i]).collect();
                eval_pure_if_supported(step.node, &args).ok_or(Error::Unsupported)?
            };
            values.push(value);
        }
        Ok(values.pop().expect("a compiled plan includes its root"))
    }
}

#[cfg(test)]
mod tests;
