// SPDX-License-Identifier: Apache-2.0

//! Group repeated priority payloads before masking wide words.

use std::collections::HashSet;

use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{self, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir_utils;

pub(super) fn rewrite(f: &mut ir::Fn, ranges: Option<&IrRangeInfo>) -> usize {
    let mut rewrites = 0;
    let mut next_id = None;
    for index in 0..f.nodes.len() {
        let node = &f.nodes[index];
        let NodePayload::PrioritySel {
            selector,
            cases,
            default: Some(default),
        } = &node.payload
        else {
            continue;
        };
        let known = |n: NodeRef| {
            ranges
                .and_then(|r| r.get(f.get_node(n).text_id))
                .and_then(|r| r.known_bits.as_ref())
        };
        let width = node.ty.bit_count();
        if width <= 3 || cases.len() <= 1 || known(*selector).is_some_and(|k| k.max_ones() <= 1) {
            continue;
        }
        if !matches!(&f.get_node(*default).payload, NodePayload::Literal(value) if is_zero(value)) {
            continue;
        }
        let mut groups = Vec::new();
        let mut seen = HashSet::new();
        let mut repeated_data = false;
        for &arm in cases.iter().chain([default]) {
            if !seen.insert(arm) {
                if arm != *default && !matches!(f.get_node(arm).payload, NodePayload::Literal(_)) {
                    repeated_data |= !known(arm).is_some_and(|k| k.is_fully_known());
                }
            } else {
                groups.push(arm);
            }
        }
        // One scalar priority per group takes O(arms × groups) work. Keep it
        // within the O(arms × width) cost of the original word masks.
        if !repeated_data || groups.len() - 1 > width {
            continue;
        }
        let (selector, default, cases, ty) = (*selector, *default, cases.clone(), node.ty.clone());
        let tag = tag_choices(f, &cases, &groups, &seen, default);
        let id = next_id.get_or_insert_with(|| super::next_text_id(f));
        let mut emit = |ty, payload| {
            let n = NodeRef {
                index: f.nodes.len(),
            };
            f.nodes.push(ir::Node {
                text_id: *id,
                name: None,
                ty,
                payload,
                pos: None,
            });
            *id += 1;
            n
        };
        let bits = [false, true].map(|bit| {
            emit(
                Type::Bits(1),
                NodePayload::Literal(IrValue::make_ubits(1, bit as u64).unwrap()),
            )
        });
        let priority = |cases, otherwise| NodePayload::PrioritySel {
            selector,
            cases,
            default: Some(otherwise),
        };
        let result = if let Some((choices, leaves)) = tag {
            // A scalar valid bit preserves all priority, including zero arms.
            // A data decision only needs the arms consistent with earlier tags.
            let validity = choices
                .iter()
                .map(|c| {
                    let test = c.classify(|word| Some(word != default)).unwrap();
                    test.emit(&mut emit, bits)
                })
                .collect();
            let valid = emit(Type::Bits(1), priority(validity, bits[0]));
            let mut value = *leaves.last().expect("a repeated data group");
            for i in (0..leaves.len() - 1).rev() {
                let mut arms = choices.iter().enumerate().rev().filter_map(|(slot, c)| {
                    let test =
                        c.classify(|w| leaves[i..].contains(&w).then_some(w == leaves[i]))?;
                    Some((slot, test))
                });
                // With a matching winner guaranteed, the last eligible arm
                // is the default regardless of its selector bit.
                let mut which = arms.next().unwrap().1.emit(&mut emit, bits);
                for (slot, test) in arms {
                    let arm = test.emit(&mut emit, bits);
                    if arm != which {
                        let active = emit(
                            Type::Bits(1),
                            NodePayload::BitSlice {
                                arg: selector,
                                start: slot,
                                width: 1,
                            },
                        );
                        which = emit(Type::Bits(1), mux(active, which, arm));
                    }
                }
                value = emit(ty.clone(), mux(which, value, leaves[i]));
            }
            mux(valid, default, value)
        } else {
            let mut controls = Vec::with_capacity(groups.len());
            for &group in &groups {
                let arms = cases
                    .iter()
                    .map(|&arm| bits[usize::from(arm == group)])
                    .collect();
                controls.push(emit(
                    Type::Bits(1),
                    priority(arms, bits[usize::from(default == group)]),
                ));
            }
            controls.reverse();
            let selector = emit(
                Type::Bits(controls.len()),
                NodePayload::Nary(NaryOp::Concat, controls),
            );
            NodePayload::OneHotSel {
                selector,
                cases: groups,
            }
        };
        ir_utils::replace_node_payload(f, NodeRef { index }, result, None)
            .expect("grouped priority has the original payload type");
        rewrites += 1;
    }
    rewrites
}

fn mux(selector: NodeRef, on_false: NodeRef, on_true: NodeRef) -> NodePayload {
    NodePayload::Sel {
        selector,
        cases: vec![on_false, on_true],
        default: None,
    }
}

/// A shared inner mux can pay for a second data decision: crossing the word
/// boundary exchanges a word mux for scalar decisions. Bound expansion to a
/// pair of decisions; larger sets use shared one-hot payload membership.
fn tag_choices(
    f: &ir::Fn,
    cases: &[NodeRef],
    groups: &[NodeRef],
    seen: &HashSet<NodeRef>,
    zero: NodeRef,
) -> Option<(Vec<Choice>, Vec<NodeRef>)> {
    let words: Vec<_> = groups.iter().copied().filter(|&g| g != zero).collect();
    let shared = |g: NodeRef| match &f.get_node(g).payload {
        NodePayload::Sel {
            selector,
            cases,
            default: None,
        } if cases.len() == 2 && cases.iter().any(|v| seen.contains(v)) => {
            Some(Choice::Mux(*selector, cases[0], cases[1]))
        }
        _ => None,
    };
    if words.iter().any(|&g| shared(g).is_some()) {
        let choices: Vec<_> = cases
            .iter()
            .map(|&g| shared(g).unwrap_or(Choice::Word(g)))
            .collect();
        let mut leaves = Vec::new();
        for choice in &choices {
            let (a, b) = match *choice {
                Choice::Word(w) => (w, w),
                Choice::Mux(_, a, b) => (a, b),
            };
            for w in [a, b] {
                if w != zero && !leaves.contains(&w) {
                    leaves.push(w);
                }
            }
        }
        if !leaves.is_empty() && leaves.len() <= 3 {
            return Some((choices, leaves));
        }
    }
    let [a, b] = words.as_slice() else {
        return None;
    };
    // A pre-masked word contributes only when its own enable is true.
    // Fold that enable into validity rather than masking every data bit.
    let choices = cases
        .iter()
        .map(|&g| match word_enable(f, g) {
            Some((data, enable)) => Choice::Mux(enable, zero, data),
            None => Choice::Word(g),
        })
        .collect();
    let a = word_enable(f, *a).map_or(*a, |(data, _)| data);
    let b = word_enable(f, *b).map_or(*b, |(data, _)| data);
    let leaves = if a == b { vec![a] } else { vec![b, a] };
    Some((choices, leaves))
}

#[derive(Clone, Copy)]
enum Choice {
    Word(NodeRef),
    Mux(NodeRef, NodeRef, NodeRef),
}
enum Test {
    Constant(bool),
    Selector(NodeRef, bool),
}

impl Choice {
    /// None means the arm is impossible in the current data partition. If
    /// only one branch remains possible, its local selector is immaterial.
    fn classify(&self, test: impl Fn(NodeRef) -> Option<bool>) -> Option<Test> {
        match *self {
            Self::Word(w) => test(w).map(Test::Constant),
            Self::Mux(selector, a, b) => match (test(a), test(b)) {
                (None, None) => None,
                (Some(v), None) | (None, Some(v)) => Some(Test::Constant(v)),
                (Some(a), Some(b)) if a == b => Some(Test::Constant(a)),
                (Some(a), Some(_)) => Some(Test::Selector(selector, a)),
            },
        }
    }
}

impl Test {
    fn emit(
        self,
        emit: &mut impl FnMut(Type, NodePayload) -> NodeRef,
        bits: [NodeRef; 2],
    ) -> NodeRef {
        match self {
            Self::Constant(v) => bits[usize::from(v)],
            Self::Selector(s, false) => s,
            Self::Selector(s, true) => emit(Type::Bits(1), NodePayload::Unop(Unop::Not, s)),
        }
    }
}

fn is_zero(value: &IrValue) -> bool {
    match value {
        IrValue::Bits(bits) => bits.is_zero(),
        IrValue::Tuple(xs) => xs.iter().all(is_zero),
        IrValue::Array(xs) => xs.elements().iter().all(is_zero),
        IrValue::Token => true,
    }
}

/// Remove a scalar enable only if the remaining word already exists in the
/// graph. This preserves shared interior expressions without rebuilding them.
fn word_enable(f: &ir::Fn, value: NodeRef) -> Option<(NodeRef, NodeRef)> {
    let NodePayload::Nary(NaryOp::And, args) = &f.get_node(value).payload else {
        return None;
    };
    let [a, b] = args.as_slice() else { return None };
    [(*a, *b), (*b, *a)].into_iter().find_map(|(word, mask)| {
        let NodePayload::SignExt { arg, .. } = f.get_node(mask).payload else {
            return None;
        };
        (f.get_node(arg).ty == Type::Bits(1)).then_some((word, arg))
    })
}
