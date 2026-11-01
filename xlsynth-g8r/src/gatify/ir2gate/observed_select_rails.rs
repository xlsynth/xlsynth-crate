// SPDX-License-Identifier: Apache-2.0
//! Reuse equivalent signed rails at the boundary of an observed word select.

use super::word_observations::WordObservations;
use super::{GateEnv, UseCounts};
use crate::aig::retained_rails::{CutBit, RailBudget, coalesce_cut_boundary};
use crate::aig::{AigNode, AigOperand, GateFn};
use std::collections::BTreeSet;
use xlsynth_pir::ir::{self, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_utils;

pub(crate) struct WordObservation {
    bits: Vec<CutBit>,
}

pub(super) fn capture(
    f: &ir::Fn,
    env: &GateEnv,
    graph: &GateFn,
    budget: &mut RailBudget,
) -> Vec<WordObservation> {
    let mut plans = Vec::<WordObservation>::new();
    let UseCounts::Function(counts) = &env.use_counts else {
        return plans;
    };
    if !budget.take(f.nodes.len()) {
        return plans;
    }
    let mut users = None;
    let mut edges = 0;
    for (i, node) in f.nodes.iter().enumerate() {
        let NodePayload::Sel {
            selector,
            cases,
            default: None,
        } = &node.payload
        else {
            continue;
        };
        let [left, right] = cases.as_slice() else {
            continue;
        };
        if f.get_node_ty(*selector) != &Type::Bits(1)
            || !matches!(&node.ty, Type::Bits(w) if *w > 1)
        {
            continue;
        }
        let data = match (&f.get_node(*left).payload, &f.get_node(*right).payload) {
            (NodePayload::Literal(_), NodePayload::Literal(_)) => continue,
            (NodePayload::Literal(_), _) => *right,
            (_, NodePayload::Literal(_)) => *left,
            _ => continue,
        };
        if users.is_none() {
            edges = counts.iter().copied().sum();
            if !budget.take(f.nodes.len().saturating_add(edges)) {
                break;
            }
            users = Some(ir_utils::compute_users(f));
        }
        // The query can visit each use and describe a word-sized comparison at
        // each node. Reserve extra pins for n-ary operands before copying them.
        let query_work = edges.saturating_add(f.nodes.len().saturating_mul(node.ty.bit_count()));
        if !budget.take(query_work) {
            continue;
        }
        let Some(observations) =
            WordObservations::find(f, NodeRef { index: i }, users.as_ref().unwrap(), |pins| {
                budget.take(pins)
            })
        else {
            continue;
        };
        if !observations
            .boundary_scalars()
            .any(|s| s.cube.pins.len() > 1)
        {
            continue;
        }
        let (Ok(enable), Ok(data)) = (env.get_bit_vector(*selector), env.get_bit_vector(data))
        else {
            continue;
        };
        let pins: BTreeSet<_> = observations
            .boundary_scalars()
            .flat_map(|s| s.cube.pins.iter().map(|&(bit, _)| bit))
            .collect();
        let operands = std::iter::once(*enable.get_lsb(0))
            .chain(pins.into_iter().map(|bit| *data.get_lsb(bit)));
        let Some(bits) = input_cut(graph, operands, budget) else {
            continue;
        };
        if !plans.iter().any(|plan| plan.bits == bits) {
            plans.push(WordObservation { bits });
        }
    }
    plans
}

fn input_cut(
    graph: &GateFn,
    operands: impl Iterator<Item = AigOperand>,
    budget: &mut RailBudget,
) -> Option<Vec<CutBit>> {
    let mut bits = Vec::new();
    for operand in operands {
        match &graph.gates[operand.node.id] {
            AigNode::Literal { .. } => continue,
            AigNode::Input {
                name, lsb_index, ..
            } => {
                if !budget.take(graph.inputs.len()) {
                    return None;
                }
                let port = graph.inputs.iter().find(|port| &port.name == name)?;
                let bit = CutBit {
                    port: name.clone(),
                    width: port.get_bit_count(),
                    bit: *lsb_index,
                };
                if !bits.contains(&bit) {
                    bits.push(bit);
                    if bits.len() > 6 {
                        return None;
                    }
                }
            }
            AigNode::And2 { .. } => return None,
        }
    }
    bits.sort_unstable_by(|a, b| (&a.port, a.bit).cmp(&(&b.port, b.bit)));
    (!bits.is_empty()).then_some(bits)
}

pub(crate) fn apply(
    mut stock: GateFn,
    observations: &[WordObservation],
    budget: &mut RailBudget,
) -> GateFn {
    for observation in observations {
        if let Some(smaller) = coalesce_cut_boundary(&stock, &observation.bits, budget) {
            stock = smaller;
        }
    }
    stock
}

#[cfg(test)]
mod tests;
