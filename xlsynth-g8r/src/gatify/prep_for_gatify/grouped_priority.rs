// SPDX-License-Identifier: Apache-2.0

//! Group repeated priority payloads before masking wide words.

use super::GateCostOptions;
use std::collections::{HashMap, HashSet};

use xlsynth_pir::IrValue;
use xlsynth_pir::ir::{self, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir_utils;

pub(super) fn rewrite(
    f: &mut ir::Fn,
    ranges: Option<&IrRangeInfo>,
    preserve_intermediates: bool,
    gate_cost: GateCostOptions,
) -> usize {
    let mut rewrites = 0;
    let mut fresh_ids = None;
    let mut literal_is_zero = cached_by_field(is_zero);
    let mut field_width = cached_by_field(Type::checked_bit_count);
    // A tag rewrite can expose a data leaf of an inner mux. Exclude these
    // new word users from ownership decisions made with the cached user map.
    let mut masks: Option<HashMap<NodeRef, WordMask>> = None;
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
        if cases.len() <= 1 {
            continue;
        }
        let Some(width) = node.ty.checked_bit_count() else {
            continue;
        };
        if width <= 3 || known(*selector).is_some_and(|k| k.max_ones() <= 1) {
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
        // Bound the number of payload masks by the data width.
        if !repeated_data || groups.len() - 1 > width {
            continue;
        }
        let (selector, default, ty) = (*selector, *default, &node.ty);
        let tag = tag_choices(f, cases, &groups, &seen, default);
        // PIR permits tokens in sel and priority_sel, but not in one_hot_sel.
        if tag.is_none() && ty.contains_token() {
            continue;
        }
        // Use the lowest unused IDs so a sparse high input ID retains its
        // headroom for later passes that allocate above the function maximum.
        let ids = fresh_ids.get_or_insert_with(|| {
            let input_ids: HashSet<_> = f.nodes.iter().map(|n| n.text_id).collect();
            (1..=usize::MAX).filter(move |id| !input_ids.contains(id))
        });
        let first_helper = f.nodes.len();
        let mut helpers = Vec::new();
        let mut emit = |ty, payload| {
            let n = NodeRef {
                index: first_helper + helpers.len(),
            };
            helpers.push(ir::Node {
                text_id: ids
                    .next()
                    .expect("a Vec<Node> cannot exhaust usize text IDs"),
                name: None,
                ty,
                payload,
                pos: None,
            });
            n
        };
        let mut target = NodeRef { index };
        let result = if let Some((choices, leaves)) = tag {
            if let Some(masks) = &mut masks {
                for leaf in &leaves {
                    masks.remove(leaf);
                }
            }
            let bits = [false, true].map(|bit| {
                emit(
                    Type::Bits(1),
                    NodePayload::Literal(IrValue::make_ubits(1, bit as u64).unwrap()),
                )
            });
            // A scalar valid bit preserves all priority, including zero arms.
            // A data decision only needs the arms consistent with earlier tags.
            let validity = choices
                .iter()
                .map(|c| {
                    let test = c.classify(|word| Some(word != default)).unwrap();
                    test.emit(&mut emit, bits)
                })
                .collect();
            let valid = emit(
                Type::Bits(1),
                NodePayload::PrioritySel {
                    selector,
                    cases: validity,
                    default: Some(bits[0]),
                },
            );
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
            groups.retain(|&group| group != default);
            // Tracked consumers can observe the unmasked word and its PIR ID.
            let scalar = if !preserve_intermediates && cases.len() <= u64::BITS as usize {
                let mask = masks
                    .get_or_insert_with(|| exclusive_masks(f))
                    .remove(&target);
                scalar_memberships(
                    f,
                    cases,
                    &groups,
                    default,
                    mask.as_ref().map(|_| width),
                    gate_cost.reassociation,
                )
                .map(|slots| (mask, slots))
            } else {
                None
            };
            if let Some((mask, slots)) = scalar {
                if let Some(mask) = &mask {
                    target = mask.root;
                }
                let controls = scalar_grants(&mut emit, selector, mask, slots);
                select_groups(&mut emit, controls, groups)
            } else {
                let controls = shared_grants(&mut emit, selector, cases, default, &groups);
                // Reassociation rebuilds gates with structural hashing.
                let share_masks = gate_cost.hash || gate_cost.reassociation;
                let fields = (!preserve_intermediates)
                    .then(|| {
                        repeated_fields(f, &groups, cases.len(), share_masks, &mut field_width)
                    })
                    .flatten();
                if let Some(words) = fields {
                    select_fields(
                        f,
                        &mut emit,
                        &controls,
                        &words,
                        groups[0],
                        &mut literal_is_zero,
                    )
                } else {
                    select_groups(&mut emit, controls, groups)
                }
            }
        };
        f.nodes.extend(helpers);
        // An explicit default name encodes the old opcode. User names still
        // identify the same value after the replacement.
        super::clear_opcode_name(f.get_node_mut(target));
        ir_utils::replace_node_payload(f, target, result, None)
            .expect("grouped priority has the original payload type");
        rewrites += 1;
    }
    rewrites
}

/// A scalar word mask that owns the only observation of the priority result.
struct WordMask {
    root: NodeRef,
    enable: NodeRef,
    inverted: bool,
}

/// Collect the sole consuming mask of each priority word in this snapshot.
fn exclusive_masks(f: &ir::Fn) -> HashMap<NodeRef, WordMask> {
    let uses = super::get_use_counts(f); // Includes the function return.
    let owns = |word: NodeRef| {
        uses[word.index] == 1 && matches!(f.get_node(word).payload, NodePayload::PrioritySel { .. })
    };
    let zero = |n: NodeRef| matches!(&f.get_node(n).payload, NodePayload::Literal(v) if is_zero(v));
    let mut masks = HashMap::new();
    for (index, node) in f.nodes.iter().enumerate() {
        let root = NodeRef { index };
        let (word, enable, inverted) = match &node.payload {
            NodePayload::Sel {
                selector,
                cases,
                default: None,
            } if f.get_node(*selector).ty == Type::Bits(1) => match cases.as_slice() {
                &[a, b] if owns(b) && zero(a) => (b, *selector, false),
                &[a, b] if owns(a) && zero(b) => (a, *selector, true),
                _ => continue,
            },
            _ => {
                let Some((word, enable)) = word_enable(f, root) else {
                    continue;
                };
                if !owns(word) {
                    continue;
                }
                (word, enable, false)
            }
        };
        masks.insert(
            word,
            WordMask {
                root,
                enable,
                inverted,
            },
        );
    }
    masks
}

/// Compare scalar decisions with a prefix update per slot, a grant per
/// nonzero request and one OR per additional request for the same payload.
/// A single bitmap per group bounds even the widest word's scalar work by
/// 64 visits per slot. Larger selectors use the linear shared prefix scan.
fn scalar_memberships(
    f: &ir::Fn,
    cases: &[NodeRef],
    groups: &[NodeRef],
    zero: NodeRef,
    masked_width: Option<usize>,
    reassociate_gates: bool,
) -> Option<Vec<u64>> {
    // Masking a literal payload folds to wiring. Keep sharing the decoder
    // when data width would overstate the work of masking those payloads.
    if groups
        .iter()
        .any(|&g| matches!(f.get_node(g).payload, NodePayload::Literal(_)))
    {
        return None;
    }
    // Each transferred enable costs an AND per payload, even if padding
    // prevents us from crediting the removed word-wide mask.
    let enable_gates = masked_width.map_or(0, |_| groups.len());
    let masked_width = masked_width.filter(|_| {
        groups.iter().any(|&g| match &f.get_node(g).payload {
            NodePayload::SignExt { .. } | NodePayload::ZeroExt { .. } => false,
            NodePayload::Nary(NaryOp::And | NaryOp::Or | NaryOp::Concat, args) => {
                // Bound the work even when many selectors share a payload.
                args.len() == 2
                    && args
                        .iter()
                        .all(|&n| !matches!(f.get_node(n).payload, NodePayload::Literal(_)))
            }
            _ => true,
        })
    });
    let mut by_group = HashMap::<NodeRef, u64>::new();
    for (i, &g) in cases.iter().enumerate() {
        if g != zero {
            *by_group.entry(g).or_default() |= 1u64 << i;
        }
    }
    let slots: Vec<_> = groups.iter().map(|g| by_group[g]).collect();
    let steps: usize = slots.iter().map(|bits| bits.ilog2() as usize).sum();
    let requests: usize = slots.iter().map(|v| v.count_ones() as usize).sum();
    let merges = requests - groups.len();
    // Only slots through the last nonzero payload feed a live shared grant.
    let last = slots.iter().max()?.ilog2() as usize;
    let max_members = slots.iter().map(|v| v.count_ones()).max()?;
    let shared_depth =
        (last + 1).next_power_of_two().ilog2() + 1 + max_members.next_power_of_two().ilog2();
    if slots.iter().any(|&v| {
        let last_hit = v.ilog2();
        let depth = if reassociate_gates && masked_width.is_some() {
            // Moving a mask exposes each grant cone to reassociation. Only
            // consecutive decisions of the same kind can then balance.
            let below_last = (1u64 << last_hit) - 1;
            let transitions = (v ^ (v << 1)) & below_last & !1;
            transitions.count_ones() + u32::from(below_last != 0)
        } else {
            // scalar_grants emits a serial binary decision per lower slot.
            last_hit
        };
        depth > shared_depth
    }) {
        return None;
    }
    let budget = (last + requests + merges).saturating_add(masked_width.unwrap_or(0));
    (steps + enable_gates <= budget).then_some(slots)
}

/// Build the scalar grant only through its last winning slot. Lower-priority
/// requests cannot affect it. Share both phases of each selector bit.
fn scalar_grants(
    emit: &mut impl FnMut(Type, NodePayload) -> NodeRef,
    selector: NodeRef,
    mask: Option<WordMask>,
    slots: Vec<u64>,
) -> Vec<NodeRef> {
    let end = slots.iter().max().expect("a repeated data group").ilog2() as usize;
    let requests: Vec<_> = (0..=end)
        .map(|start| {
            let bit = emit(
                Type::Bits(1),
                NodePayload::BitSlice {
                    arg: selector,
                    start,
                    width: 1,
                },
            );
            [bit, emit(Type::Bits(1), NodePayload::Unop(Unop::Not, bit))]
        })
        .collect();
    let enable = mask.map(|mask| {
        if mask.inverted {
            emit(Type::Bits(1), NodePayload::Unop(Unop::Not, mask.enable))
        } else {
            mask.enable
        }
    });
    slots
        .into_iter()
        .rev()
        .map(|membership| {
            let last = membership.ilog2() as usize;
            let mut win = requests[last][0];
            for i in (0..last).rev() {
                let (op, request) = if membership & (1 << i) != 0 {
                    (NaryOp::Or, requests[i][0])
                } else {
                    (NaryOp::And, requests[i][1])
                };
                win = emit(Type::Bits(1), NodePayload::Nary(op, vec![request, win]));
            }
            enable.map_or(win, |en| {
                emit(Type::Bits(1), NodePayload::Nary(NaryOp::And, vec![win, en]))
            })
        })
        .collect()
}

/// Resolves priority once, then masks each distinct payload with the OR of its
/// winning slots. Zero slots still participate in priority but supply no data.
fn shared_grants(
    emit: &mut impl FnMut(Type, NodePayload) -> NodeRef,
    selector: NodeRef,
    cases: &[NodeRef],
    zero: NodeRef,
    groups: &[NodeRef],
) -> Vec<NodeRef> {
    let grants = emit(
        Type::Bits(cases.len() + 1),
        NodePayload::OneHot {
            arg: selector,
            lsb_prio: true,
        },
    );
    let mut by_group: HashMap<NodeRef, Vec<NodeRef>> = HashMap::new();
    for (slot, &group) in cases.iter().enumerate() {
        if group != zero {
            let grant = emit(
                Type::Bits(1),
                NodePayload::BitSlice {
                    arg: grants,
                    start: slot,
                    width: 1,
                },
            );
            by_group.entry(group).or_default().push(grant);
        }
    }
    merged_grants(emit, groups, by_group)
}

/// Merge each distinct value's grants, in the bit order expected by concat.
fn merged_grants(
    emit: &mut impl FnMut(Type, NodePayload) -> NodeRef,
    values: &[NodeRef],
    mut by_value: HashMap<NodeRef, Vec<NodeRef>>,
) -> Vec<NodeRef> {
    values
        .iter()
        .rev()
        .map(|v| {
            let grants = by_value.remove(v).expect("each value has a grant");
            if grants.len() == 1 {
                grants[0]
            } else {
                emit(Type::Bits(1), NodePayload::Nary(NaryOp::Or, grants))
            }
        })
        .collect()
}

/// Select one payload per grant. Controls arrive in concat (MSB-first) order.
fn select_groups(
    emit: &mut impl FnMut(Type, NodePayload) -> NodeRef,
    controls: Vec<NodeRef>,
    groups: Vec<NodeRef>,
) -> NodePayload {
    let selector = emit(
        Type::Bits(groups.len()),
        NodePayload::Nary(NaryOp::Concat, controls),
    );
    NodePayload::OneHotSel {
        selector,
        cases: groups,
    }
}

/// Factor an aggregate field only when repeated data masks repay the field
/// references visited. Bound all visits by the slot, control and payload
/// references already needed for the whole-word priority route.
fn repeated_fields<'a>(
    f: &'a ir::Fn,
    groups: &[NodeRef],
    slots: usize,
    share_masks: bool,
    field_width: &mut impl FnMut(NodeRef, &Type) -> Option<usize>,
) -> Option<Vec<&'a [NodeRef]>> {
    let fields = |g| match &f.get_node(g).payload {
        NodePayload::Array(xs) | NodePayload::Tuple(xs) => Some(xs.as_slice()),
        _ => None,
    };
    let visits = fields(groups[0])?.len().checked_mul(groups.len())?;
    if visits > slots.saturating_add(groups.len().saturating_mul(2)) {
        return None;
    }
    let words = groups
        .iter()
        .map(|&g| fields(g))
        .collect::<Option<Vec<_>>>()?;
    let mut seen = HashSet::new();
    let mut saved = 0usize;
    let mut cost = visits;
    let mut masked_fields = HashSet::new();
    for word in &words {
        masked_fields.clear();
        for (column, &field) in word.iter().enumerate() {
            let node = f.get_node(field);
            // Constants already mask to wires. Repeated nonliteral fields can
            // replace a mask per data bit with one OR of their grants.
            if matches!(node.payload, NodePayload::Literal(_)) {
                continue;
            }
            if !seen.insert((column, field)) {
                saved = saved.saturating_add(field_width(field, &node.ty)?.saturating_sub(1));
            }
            // Within a word, all occurrences use the same data and grant.
            // Structural hashing can already share this entire data mask.
            if share_masks && !masked_fields.insert(field) {
                cost = cost.saturating_add(field_width(field, &node.ty)?);
            }
        }
    }
    (saved > cost).then_some(words)
}

/// Compute a field property on first use. Grouping preserves existing node
/// types and literal values across rewrites.
fn cached_by_field<T, R: Copy>(mut compute: impl FnMut(&T) -> R) -> impl FnMut(NodeRef, &T) -> R {
    let mut known = HashMap::new();
    move |field, value| *known.entry(field).or_insert_with(|| compute(value))
}

/// Route each field once per distinct nonzero value using the common grants.
fn select_fields(
    f: &ir::Fn,
    emit: &mut impl FnMut(Type, NodePayload) -> NodeRef,
    controls: &[NodeRef],
    words: &[&[NodeRef]],
    first: NodeRef,
    literal_is_zero: &mut impl FnMut(NodeRef, &IrValue) -> bool,
) -> NodePayload {
    let mut fields = Vec::new();
    for column in 0..words[0].len() {
        let mut values = Vec::new();
        let mut grants: HashMap<NodeRef, Vec<NodeRef>> = HashMap::new();
        for (&control, word) in controls.iter().rev().zip(words) {
            let field = word[column];
            if let NodePayload::Literal(v) = &f.get_node(field).payload
                && literal_is_zero(field, v)
            {
                continue;
            }
            let slots = grants.entry(field).or_default();
            if slots.is_empty() {
                values.push(field);
            }
            slots.push(control);
        }
        let ty = &f.get_node(words[0][column]).ty;
        let selected = if values.is_empty() {
            words[0][column]
        } else {
            let bits = merged_grants(emit, &values, grants);
            let payload = select_groups(emit, bits, values);
            emit(ty.clone(), payload)
        };
        fields.push(selected);
    }
    if matches!(f.get_node(first).ty, Type::Tuple(_)) {
        NodePayload::Tuple(fields)
    } else {
        NodePayload::Array(fields)
    }
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
        let choices = cases.iter().map(|&g| shared(g).unwrap_or(Choice::Word(g)));
        if let Some(tag) = collect_shared_choices(choices, zero) {
            return Some(tag);
        }
    }
    // With one data word, validity alone selects it.
    if words.len() == 1 {
        return Some((cases.iter().copied().map(Choice::Word).collect(), words));
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

/// Collects shared mux arms while they fit in at most two data decisions.
fn collect_shared_choices(
    choices: impl Iterator<Item = Choice>,
    zero: NodeRef,
) -> Option<(Vec<Choice>, Vec<NodeRef>)> {
    let mut arms = Vec::new();
    let mut leaves = Vec::new();
    for choice in choices {
        let (a, b) = match choice {
            Choice::Word(w) => (w, w),
            Choice::Mux(_, a, b) => (a, b),
        };
        for w in [a, b] {
            if w != zero && !leaves.contains(&w) {
                if leaves.len() == 3 {
                    return None;
                }
                leaves.push(w);
            }
        }
        arms.push(choice);
    }
    (!leaves.is_empty()).then_some((arms, leaves))
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

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn fourth_shared_mux_leaf_stops_arm_collection() {
        let [zero, enable, a, b, c, d] = std::array::from_fn(|index| NodeRef { index });
        let choices = [
            Choice::Mux(enable, a, b),
            Choice::Word(a),
            Choice::Word(b),
            Choice::Word(c),
            Choice::Word(d),
        ]
        .into_iter()
        .chain(std::iter::once_with(|| {
            panic!("visited an arm after the fourth leaf")
        }));
        assert!(collect_shared_choices(choices, zero).is_none());
    }

    #[test]
    fn routed_literals_are_classified_once_per_node() {
        let zero = IrValue::make_ubits(4096, 0).unwrap();
        let one = IrValue::make_ubits(4096, 1).unwrap();
        let mut tests = 0;
        let mut classify = cached_by_field(|value| {
            tests += 1;
            is_zero(value)
        });
        for _ in 0..65 {
            assert!(classify(NodeRef { index: 0 }, &zero));
            assert!(!classify(NodeRef { index: 1 }, &one));
        }
        drop(classify);
        assert_eq!(tests, 2);
    }

    #[test]
    fn repeated_field_costs_share_checked_widths() {
        let text = r#"package widths
top fn f(x: bits[12] id=1, y: bits[12] id=2, z: bits[12] id=3, w: bits[12] id=4) -> bits[12][6] {
  a: bits[12][6] = array(x, x, x, x, y, y, id=5)
  ret b: bits[12][6] = array(x, x, x, x, z, w, id=6)
}"#;
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();
        let groups = ["a", "b"].map(|name| ir_utils::find_node_by_name(f, name).unwrap());
        // Four repeated x columns save 44 gates over 12 field visits.
        // Sharing masks charges six repeated x's and one y (84 gates).
        // Charging just once per field would incorrectly allow routing.
        for (share_masks, expected_calls) in [(false, 1), (true, 2)] {
            let mut calls = 0;
            let mut width = cached_by_field(|ty: &Type| {
                calls += 1;
                ty.checked_bit_count()
            });
            assert_eq!(
                repeated_fields(f, &groups, 12, share_masks, &mut width).is_some(),
                !share_masks
            );
            drop(width);
            assert_eq!(calls, expected_calls);
        }
    }
}
