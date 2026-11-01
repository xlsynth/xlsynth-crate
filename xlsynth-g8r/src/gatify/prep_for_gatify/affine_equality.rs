// SPDX-License-Identifier: Apache-2.0

//! Pull a masked equality through a modular translation or reflection.

use std::collections::{BTreeMap, HashMap, HashSet};

use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils::{self, Users};
use xlsynth_pir::{IrBits, IrValue};

pub(super) fn rewrite(f: &mut ir::Fn) -> usize {
    let mut observations = BTreeMap::<usize, Vec<Observation>>::new();
    for (index, node) in f.nodes.iter().enumerate() {
        let NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), a, b) = node.payload else {
            continue;
        };
        let (view, literal) = match (&f.get_node(a).payload, &f.get_node(b).payload) {
            (NodePayload::Literal(v), _) => (b, v),
            (_, NodePayload::Literal(v)) => (a, v),
            _ => continue,
        };
        if let Some((word, observed)) = literal
            .to_bits()
            .ok()
            .and_then(|v| Observation::of_view(f, NodeRef { index }, op == Binop::Eq, view, &v))
        {
            observations.entry(word.index).or_default().push(observed);
        }
    }
    if observations.is_empty() {
        return 0;
    }
    let users = ir_utils::compute_users(f);
    let mut roots = HashMap::new();
    let mut plans = Vec::new();
    for (index, atoms) in observations {
        let word = NodeRef { index };
        let width = f.get_node(word).ty.bit_count();
        // A single whole-word comparison belongs to arithmetic simplification.
        // This pass combines separate observations or removes unobserved work.
        if atoms.len() == 1 && atoms[0].mask == IrBits::all_ones(width) {
            continue;
        }
        if !word_is_exclusively_observed(f, &users, word, &atoms) {
            continue;
        }
        if atoms.len() > 1 {
            let mut owners = atoms
                .iter()
                .map(|a| conjunction_root(f, &users, (a.comparison, a.equal), &mut roots));
            let first = owners.next().unwrap();
            if owners.any(|owner| owner != first) {
                continue;
            }
        }
        let affine = Affine::of(f, word).expect("observed affine word");
        let mut mask = IrBits::zero(width);
        let mut value = IrBits::zero(width);
        let mut conflict = false;
        for a in &atoms {
            conflict |= a.conflict || !mask.and(&a.mask).and(&value.xor(&a.value)).is_zero();
            mask = mask.or(&a.mask);
            value = value.or(&a.value);
        }
        let inverse = if conflict {
            None
        } else {
            let Some(value) = affine.invert_cube(&mask, &value) else {
                continue;
            };
            let input_width = f.get_node(affine.input).ty.bit_count();
            if input_width < width && !mask.shrl(input_width as i64).is_zero() {
                // Constraining the extended field requires reasoning about the
                // extension.
                continue;
            }
            Some(value)
        };
        // Cut rewriting can share the original arithmetic with another output
        // even when no other IR node observes the arithmetic result.
        if shares_input_demand(f, &users, word, affine.input, &mask) {
            continue;
        }
        plans.push((atoms, inverse.map(|value| (affine.input, mask, value))));
    }

    let count = plans.len();
    for (atoms, inverse) in plans {
        let first = &atoms[0];
        let payload = if let Some((input, mask, value)) = inverse {
            let width = (0..mask.get_bit_count())
                .rfind(|&i| mask.get_bit(i).unwrap())
                .unwrap()
                + 1;
            let mask = mask.width_slice(0, width as i64);
            let value = value.width_slice(0, width as i64);
            let ty = Type::Bits(width);
            let mut observed = input;
            if f.get_node(input).ty != ty {
                observed = ir_utils::push_node(
                    f,
                    ty.clone(),
                    NodePayload::BitSlice {
                        arg: input,
                        start: 0,
                        width,
                    },
                );
            }
            if mask != IrBits::all_ones(width) {
                let m = ir_utils::push_node(
                    f,
                    ty.clone(),
                    NodePayload::Literal(IrValue::from_bits(&mask)),
                );
                observed = ir_utils::push_node(
                    f,
                    ty.clone(),
                    NodePayload::Nary(NaryOp::And, vec![observed, m]),
                );
            }
            let literal =
                ir_utils::push_node(f, ty, NodePayload::Literal(IrValue::from_bits(&value)));
            NodePayload::Binop(
                if first.equal { Binop::Eq } else { Binop::Ne },
                observed,
                literal,
            )
        } else {
            NodePayload::Literal(IrValue::from_bits(&IrBits::bool(!first.equal)))
        };
        f.nodes[first.comparison.index].payload = payload;
        // The group is observed only as a conjunction of these truth
        // assignments.
        for a in &atoms[1..] {
            f.nodes[a.comparison.index].payload =
                NodePayload::Literal(IrValue::from_bits(&IrBits::bool(a.equal)));
        }
    }
    if count > 0 {
        super::mark_dead_nodes_as_nil(f);
    }
    count
}

struct Affine {
    input: NodeRef,
    negative: bool,
    offset: IrBits,
}

impl Affine {
    fn of(f: &ir::Fn, word: NodeRef) -> Option<Self> {
        let width = f.get_node(word).ty.bit_count();
        let mut input = None;
        let mut offset = None;
        let mut term = |operand: NodeRef, signed: bool, negative: bool| -> Option<()> {
            if let NodePayload::Literal(ref literal) = f.get_node(operand).payload {
                let bits = literal.to_bits().ok()?;
                let mut extended = bits.width_slice(0, width as i64);
                if signed && bits.get_bit_count() < width && bits.get_bit_count() > 0 && bits.msb()
                {
                    extended =
                        extended.or(&IrBits::all_ones(width).shll(bits.get_bit_count() as i64));
                }
                let accumulated = offset.get_or_insert_with(|| IrBits::zero(width));
                *accumulated = if negative {
                    accumulated.sub(&extended)
                } else {
                    accumulated.add(&extended)
                };
            } else if input.replace((operand, negative)).is_some() {
                return None;
            }
            Some(())
        };
        match f.get_node(word).payload {
            NodePayload::ExtNaryAdd {
                ref terms,
                arch: None,
            } => {
                if matches!(terms.as_slice(), [t] if !t.negated) {
                    return None;
                }
                for &ExtNaryAddTerm {
                    operand,
                    signed,
                    negated,
                } in terms
                {
                    term(operand, signed, negated)?;
                }
            }
            NodePayload::Binop(op @ (Binop::Add | Binop::Sub), left, right) => {
                term(left, false, false)?;
                term(right, false, op == Binop::Sub)?;
            }
            NodePayload::Unop(Unop::Neg, x) => term(x, false, true)?,
            _ => return None,
        }
        let (input, negative) = input?;
        Some(Self {
            input,
            negative,
            offset: offset.unwrap_or_else(|| IrBits::zero(width)),
        })
    }

    fn invert_cube(&self, mask: &IrBits, value: &IrBits) -> Option<IrBits> {
        let inverse = |y: &IrBits| {
            if self.negative {
                self.offset.sub(y)
            } else {
                y.sub(&self.offset)
            }
        };
        let lo = inverse(value);
        let hi = inverse(&value.or(&mask.not()));
        // At a fixed bit the incoming carry or borrow is monotone in the low
        // word. These endpoints cover both extrema. When all fixed bits stay
        // fixed, bijectivity gives equal cardinality and hence an exact cube.
        lo.xor(&hi).and(mask).is_zero().then(|| lo.and(mask))
    }
}

struct Observation {
    comparison: NodeRef,
    equal: bool,
    views: Vec<NodeRef>,
    mask: IrBits,
    value: IrBits,
    conflict: bool,
}

impl Observation {
    fn of_view(
        f: &ir::Fn,
        comparison: NodeRef,
        equal: bool,
        view: NodeRef,
        literal: &IrBits,
    ) -> Option<(NodeRef, Self)> {
        let mut pending = vec![(view, 0usize, literal.get_bit_count(), 0usize)];
        let mut source = None;
        let mut bits: Option<(IrBits, IrBits)> = None;
        let mut views = Vec::new();
        let mut conflict = false;
        while let Some((node, start, length, target)) = pending.pop() {
            if length == 0 {
                continue;
            }
            match &f.get_node(node).payload {
                NodePayload::BitSlice {
                    arg, start: low, ..
                } => {
                    views.push(node);
                    pending.push((*arg, start + low, length, target));
                }
                NodePayload::Nary(NaryOp::Concat, args) => {
                    views.push(node);
                    let mut base = 0;
                    for &arg in args.iter().rev() {
                        let end = base + f.get_node(arg).ty.bit_count();
                        let lo = base.max(start);
                        let hi = end.min(start + length);
                        if hi > lo {
                            pending.push((arg, lo - base, hi - lo, target + lo - start));
                        }
                        base = end;
                    }
                }
                _ => {
                    if let Some(other) = source {
                        if other != node {
                            return None;
                        }
                    } else {
                        Affine::of(f, node)?;
                        source = Some(node);
                    }
                    let width = f.get_node(node).ty.bit_count() as i64;
                    let (mask, value) = bits.get_or_insert_with(|| {
                        (IrBits::zero(width as usize), IrBits::zero(width as usize))
                    });
                    let field_mask = IrBits::all_ones(length)
                        .width_slice(0, width)
                        .shll(start as i64);
                    let field = literal
                        .width_slice(target as i64, length as i64)
                        .width_slice(0, width)
                        .shll(start as i64);
                    conflict |= !mask.and(&field_mask).and(&value.xor(&field)).is_zero();
                    *mask = mask.or(&field_mask);
                    *value = value.or(&field);
                }
            }
        }
        let (mask, value) = bits?;
        Some((
            source?,
            Self {
                comparison,
                equal,
                views,
                mask,
                value,
                conflict,
            },
        ))
    }
}

fn word_is_exclusively_observed(
    f: &ir::Fn,
    users: &Users,
    word: NodeRef,
    atoms: &[Observation],
) -> bool {
    let data: HashSet<_> = std::iter::once(word)
        .chain(atoms.iter().flat_map(|a| a.views.iter().copied()))
        .collect();
    let comparisons: HashSet<_> = atoms.iter().map(|a| a.comparison).collect();
    data.iter().all(|n| {
        f.ret_node_ref != Some(*n)
            && users
                .get(n)
                .unwrap()
                .iter()
                .all(|u| data.contains(u) || comparisons.contains(u))
    })
}

/// A partial sum can depend on every input bit below its highest observed bit.
fn shares_input_demand(
    f: &ir::Fn,
    users: &Users,
    word: NodeRef,
    input: NodeRef,
    observed: &IrBits,
) -> bool {
    if f.get_node(input).payload == NodePayload::Param
        && users.get(&input).is_some_and(|uses| uses == [word])
    {
        return false;
    }
    let Some(root) = f.ret_node_ref else {
        return true;
    };
    let input_width = f.get_node(input).ty.bit_count();
    let Some(highest) = (0..observed.get_bit_count()).rfind(|&i| observed.get_bit(i).unwrap())
    else {
        return true;
    };
    let low = IrBits::all_ones((highest + 1).min(input_width)).width_slice(0, input_width as i64);
    let source = input_support(f, input, low, word);
    let outside = input_support(
        f,
        root,
        IrBits::all_ones(f.get_node(root).ty.bit_count()),
        word,
    );
    source.iter().any(|(param, bits)| {
        outside
            .get(param)
            .is_some_and(|other| !bits.and(other).is_zero())
    })
}

/// Wiring preserves exact bit demand. Other operators conservatively demand
/// their full operands. Cutting the affine word leaves its independent context.
fn input_support(
    f: &ir::Fn,
    root: NodeRef,
    mask: IrBits,
    cut: NodeRef,
) -> HashMap<NodeRef, IrBits> {
    let mut pending = vec![(root, mask)];
    let mut visited = HashMap::<NodeRef, IrBits>::new();
    let mut support = HashMap::new();
    while let Some((node, mut mask)) = pending.pop() {
        if node == cut || mask.is_zero() {
            continue;
        }
        if let Some(prev) = visited.get(&node) {
            mask = mask.and(&prev.not());
            if mask.is_zero() {
                continue;
            }
            visited.insert(node, prev.or(&mask));
        } else {
            visited.insert(node, mask.clone());
        }
        match &f.get_node(node).payload {
            NodePayload::Literal(_) | NodePayload::Nil => {}
            NodePayload::Param | NodePayload::InputPort { .. } => {
                support.insert(node, visited[&node].clone());
            }
            NodePayload::Unop(Unop::Identity | Unop::Not, x) => pending.push((*x, mask)),
            NodePayload::BitSlice { arg, start, .. } => {
                let width = f.get_node(*arg).ty.bit_count() as i64;
                pending.push((*arg, mask.width_slice(0, width).shll(*start as i64)));
            }
            NodePayload::TupleIndex { tuple, index } => {
                let Type::Tuple(fields) = &f.get_node(*tuple).ty else {
                    unreachable!("validated tuple index")
                };
                // XLS packs the last tuple field at the least significant end.
                let low: usize = fields[index + 1..]
                    .iter()
                    .map(|field| field.bit_count())
                    .sum();
                let width = f.get_node(*tuple).ty.bit_count() as i64;
                pending.push((*tuple, mask.width_slice(0, width).shll(low as i64)));
            }
            NodePayload::Tuple(parts) | NodePayload::Nary(NaryOp::Concat, parts) => {
                let mut low = 0;
                for &part in parts.iter().rev() {
                    let width = f.get_node(part).ty.bit_count() as i64;
                    pending.push((part, mask.width_slice(low, width)));
                    low += width;
                }
            }
            _ => {
                for arg in ir_utils::operands(&f.get_node(node).payload) {
                    pending.push((arg, IrBits::all_ones(f.get_node(arg).ty.bit_count())));
                }
            }
        }
    }
    support
}

type Truth = (NodeRef, bool);

/// A true AND or a false OR requires every child. Stop at the first use that
/// can observe this condition separately. Cache both polarities independently.
fn conjunction_root(
    f: &ir::Fn,
    users: &Users,
    leaf: Truth,
    cache: &mut HashMap<Truth, Truth>,
) -> Truth {
    let mut path = Vec::new();
    let mut at = leaf;
    let root = loop {
        if let Some(&old) = cache.get(&at) {
            break old;
        }
        path.push(at);
        let (node, value) = at;
        let [user] = users.get(&node).unwrap() else {
            break at;
        };
        if f.ret_node_ref == Some(node) {
            break at;
        }
        let needed = match f.get_node(*user).payload {
            NodePayload::Unop(Unop::Not, _) => !value,
            NodePayload::Nary(NaryOp::And, _) if value => true,
            NodePayload::Nary(NaryOp::Nand, _) if value => false,
            NodePayload::Nary(NaryOp::Or, _) if !value => false,
            NodePayload::Nary(NaryOp::Nor, _) if !value => true,
            _ => break at,
        };
        at = (*user, needed);
    };
    for at in path {
        cache.insert(at, root);
    }
    root
}
