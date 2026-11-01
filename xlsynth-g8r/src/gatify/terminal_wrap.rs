// SPDX-License-Identifier: Apache-2.0

//! Passive, source-qualified coordinates in one immutable pre-optimization gate
//! function.

use crate::aig::gate::{AigBitVector, AigNode, AigOperand, GateFn};
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils;
use xlsynth_pir::known_bits::{LowWordQuery, boolean_under_low_word_extremes};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReturnBit {
    Constant(bool),
    Predicate { inverted: bool },
}

/// Operands belong to the gate function which supplied the lowering map.
/// Each group in `outputs` follows its real output from least to most
/// significant bit.
#[derive(Debug)]
pub(crate) struct TerminalWrapCapture {
    pub(crate) predicate_pir_id: Option<u32>,
    pub(crate) word_lsb: Vec<AigOperand>,
    pub(crate) carry: AigOperand,
    pub(crate) equal: bool,
    pub(crate) outputs: Vec<Vec<ReturnBit>>,
}

#[derive(PartialEq)]
pub(super) struct TerminalWrapPermit {
    root_id: usize,
    sum_id: usize,
    word: Definition,
    carry: Definition,
    width: usize,
    equal: bool,
}

impl TerminalWrapPermit {
    /// At the ordinary-add site, every nonconstant path to a return must cross
    /// one observation.
    pub(super) fn at_source(f: &ir::Fn) -> Option<Self> {
        let r = terminal_observation(f, Stage::Source)?;
        // Preserve known word/carry sharing. The modular identity itself holds
        // for every carry.
        match boolean_under_low_word_extremes(f, r.word, r.width, r.carry) {
            LowWordQuery::Disjoint => {}
            LowWordQuery::Constrained(facts)
                if facts.unconditional.is_some()
                    || (!facts.gains_known_bit() && !facts.any_infeasible()) => {}
            _ => return None,
        }
        Some(Self::key(f, &r))
    }

    pub(super) fn capture<'a>(
        self,
        prepared: &ir::Fn,
        gates: &GateFn,
        lowered: impl Fn(NodeRef) -> Option<&'a AigBitVector>,
    ) -> Option<TerminalWrapCapture> {
        let r = terminal_observation(prepared, Stage::Prepared)?;
        if Self::key(prepared, &r) != self {
            return None;
        }
        let word = lowered(low_source(prepared, r.word, r.width))?;
        let carry = lowered(r.carry)?;
        let predicate = lowered(r.root)?;
        if word.get_bit_count() < r.width
            || carry.get_bit_count() != 1
            || predicate.get_bit_count() != 1
        {
            return None;
        }
        let predicate = *predicate.get_lsb(0);
        let outputs = gates
            .outputs
            .iter()
            .map(|out| {
                out.bit_vector
                    .iter_lsb_to_msb()
                    .map(|&op| {
                        if let AigNode::Literal { value, .. } = gates.gates.get(op.node.id)? {
                            Some(ReturnBit::Constant(*value ^ op.negated))
                        } else if op.node == predicate.node {
                            Some(ReturnBit::Predicate {
                                inverted: op.negated != predicate.negated,
                            })
                        } else {
                            None
                        }
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?;
        if !outputs
            .iter()
            .flatten()
            .any(|bit| matches!(bit, ReturnBit::Predicate { .. }))
        {
            return None;
        }
        let predicate_pir_id = u32::try_from(self.root_id)
            .ok()
            .filter(|id| gates.get(predicate.node).get_pir_node_ids().contains(id));
        Some(TerminalWrapCapture {
            predicate_pir_id,
            word_lsb: word.iter_lsb_to_msb().take(r.width).copied().collect(),
            carry: *carry.get_lsb(0),
            equal: r.equal,
            outputs,
        })
    }

    fn key(f: &ir::Fn, r: &Observation) -> Self {
        Self {
            root_id: f.get_node(r.root).text_id,
            sum_id: f.get_node(r.sum).text_id,
            word: Definition::of(f, low_source(f, r.word, r.width)),
            carry: Definition::of(f, r.carry),
            equal: r.equal,
            width: r.width,
        }
    }
}

/// An immediate typed definition in stable text-ID coordinates, independent of
/// compaction.
#[derive(PartialEq)]
struct Definition {
    text_id: usize,
    ty: Type,
    payload: NodePayload,
}

impl Definition {
    fn of(f: &ir::Fn, nr: NodeRef) -> Self {
        let n = f.get_node(nr);
        Self {
            text_id: n.text_id,
            ty: n.ty.clone(),
            payload: ir_utils::remap_payload_with(&n.payload, |(_, op)| NodeRef {
                index: f.get_node(op).text_id,
            }),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Source,
    Prepared,
}

struct Observation {
    root: NodeRef,
    sum: NodeRef,
    word: NodeRef,
    carry: NodeRef,
    width: usize,
    equal: bool,
}

fn terminal_observation(f: &ir::Fn, stage: Stage) -> Option<Observation> {
    let mut pending = vec![f.ret_node_ref?];
    let mut visited = vec![false; f.nodes.len()];
    let mut found = None;
    while let Some(n) = pending.pop() {
        if std::mem::replace(&mut visited[n.index], true) {
            continue;
        }
        if let Some(observation) = recognize(f, n, stage) {
            if found.replace(observation).is_some() {
                return None;
            }
            continue;
        }
        let payload = &f.get_node(n).payload;
        if !matches!(payload, NodePayload::Literal(_)) {
            let inputs = ir_utils::operands(payload);
            if inputs.is_empty() {
                return None;
            }
            pending.extend(inputs);
        }
    }
    found
}

fn recognize(f: &ir::Fn, root: NodeRef, stage: Stage) -> Option<Observation> {
    let zero = |n: NodeRef| matches!(&f.get_node(n).payload, NodePayload::Literal(v) if v.bits_equals_u64_value(0));
    let (value, equal) = match f.get_node(root).payload {
        NodePayload::Unop(Unop::OrReduce, arg) => (arg, false),
        NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), a, b) if zero(a) || zero(b) => {
            (if zero(a) { b } else { a }, op == Binop::Eq)
        }
        _ => return None,
    };
    let Type::Bits(width) = f.get_node(value).ty else {
        return None;
    };
    if width < 2 {
        return None;
    }
    let sum = match f.get_node(value).payload {
        NodePayload::BitSlice { arg, start: 0, .. } => arg,
        _ => value,
    };
    let (word, carry) = match &f.get_node(sum).payload {
        NodePayload::Binop(Binop::Add, a, b) => {
            [(*a, *b), (*b, *a)].into_iter().find_map(|(word, inc)| {
                let carry = match &f.get_node(inc).payload {
                    NodePayload::ZeroExt { arg, .. } => *arg,
                    NodePayload::Nary(NaryOp::Concat, args) => {
                        let (low, high) = args.split_last()?;
                        if high.is_empty() || !high.iter().all(|&n| zero(n)) {
                            return None;
                        }
                        *low
                    }
                    _ => return None,
                };
                (f.get_node(carry).ty == Type::Bits(1)).then_some((word, carry))
            })?
        }
        NodePayload::ExtNaryAdd { terms, arch: None } if stage == Stage::Prepared => {
            let [a, b] = terms.as_slice() else {
                return None;
            };
            if a.signed || b.signed || a.negated || b.negated {
                return None;
            }
            [(a.operand, b.operand), (b.operand, a.operand)]
                .into_iter()
                .find(|&(_, carry)| f.get_node(carry).ty == Type::Bits(1))?
        }
        _ => return None,
    };
    if !matches!(f.get_node(word).ty, Type::Bits(w) if w >= width) {
        return None;
    }
    Some(Observation {
        root,
        sum,
        word,
        carry,
        width,
        equal,
    })
}

/// Drop only wiring entirely above the observed low window. The surviving node
/// must have a complete lowered value; a partially demanded wider value cannot
/// supply this witness.
fn low_source(f: &ir::Fn, mut word: NodeRef, width: usize) -> NodeRef {
    loop {
        let next = match &f.get_node(word).payload {
            NodePayload::ZeroExt { arg, .. } | NodePayload::Unop(Unop::Identity, arg) => *arg,
            NodePayload::Nary(NaryOp::Concat, args) if !args.is_empty() => args[args.len() - 1],
            _ => break,
        };
        if !matches!(f.get_node(next).ty, Type::Bits(w) if w >= width) {
            break;
        }
        word = next;
    }
    word
}

#[cfg(test)]
mod tests;
