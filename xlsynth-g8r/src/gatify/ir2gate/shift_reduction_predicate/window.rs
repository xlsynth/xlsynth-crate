// SPDX-License-Identifier: Apache-2.0
//! Conservative, finite control domains for observing the OR of selected data
//! bits.
use std::collections::{HashMap, HashSet};
use xlsynth_pir::{
    IrValue,
    ir::{self, Binop, NaryOp, NodePayload as P, NodeRef as R, Type, Unop},
    ir_eval::bounded_bits_cone::{BitsCone, Error as ConeError, Limits},
    ir_utils::{get_topological, operands},
};

const REGION: usize = 192;
const SPANS: usize = 24;
const CONTROL_BITS: usize = 8;
const CONTROL_OPS: usize = 48;
const FUNCTION_WORK: usize = 32_768;
const FUNCTION_VALUATIONS: usize = 2_048;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cell {
    Const(bool),
    Any(Vec<(R, usize, usize)>),
}
#[derive(Debug)]
pub struct Plan {
    pub root: R,
    pub inverted: bool,
    pub source: Vec<R>,
    pub named: bool,
    pub cells: Vec<Cell>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
enum Why {
    OverlappingRoot,
    Seed,
    Shared,
    Dependencies,
    Unsupported,
    Invariant,
    Work,
    ControlCone,
    ControlValuations,
    Fragments,
}
/// Counts are allocation-free. Labels are produced only when a diagnostic is
/// formatted.
#[derive(Default)]
pub struct Reasons([usize; 10]);

impl Reasons {
    fn count(&mut self, why: Why) {
        self.0[why as usize] += 1;
    }
}

impl std::fmt::Debug for Reasons {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for reason in [
            Why::OverlappingRoot,
            Why::Seed,
            Why::Shared,
            Why::Dependencies,
            Why::Unsupported,
            Why::Invariant,
            Why::Work,
            Why::ControlCone,
            Why::ControlValuations,
            Why::Fragments,
        ] {
            let count = self.0[reason as usize];
            if count != 0 {
                map.entry(&format!("{reason:?}"), &count);
            }
        }
        map.finish()
    }
}

type Res<T> = Result<T, Why>;
#[derive(Default)]
pub struct Results {
    pub plans: Vec<Plan>,
    pub reasons: Reasons,
    pub visits: usize,
    pub valuations: usize,
}

pub fn analyze(f: &ir::Fn, allow_source: bool) -> Results {
    let mut out = Results::default();
    let Some(ret) = f.ret_node_ref else {
        return out;
    };
    // A selection is necessary in both domains; only named controls require a
    // shift.
    let mut shift = false;
    let mut select = false;
    for n in &f.nodes {
        shift |= matches!(n.payload, P::Binop(Binop::Shrl | Binop::Shll, _, _));
        select |= matches!(n.payload, P::Sel { .. } | P::PrioritySel { .. });
    }
    if !select || (!allow_source && !shift) {
        out.reasons.count(Why::Seed);
        return out;
    }
    let order = get_topological(f);
    let mut live = vec![false; f.nodes.len()];
    let mut uses = vec![0; f.nodes.len()];
    live[ret.index] = true;
    for &r in order.iter().rev() {
        if live[r.index] {
            for a in operands(&f.get_node(r).payload) {
                live[a.index] = true;
                uses[a.index] += 1
            }
        }
    }
    let mut flags = vec![(false, false, false); f.nodes.len()];
    let mut roots = Vec::new();
    for &r in &order {
        if !live[r.index] {
            continue;
        }
        let p = &f.get_node(r).payload;
        let args = operands(p);
        let shifted = matches!(p, P::Binop(Binop::Shrl | Binop::Shll, _, _))
            || args.iter().any(|x| flags[x.index].0);
        let selected = matches!(p, P::Sel { .. } | P::PrioritySel { .. })
            || args.iter().any(|x| flags[x.index].1);
        let observing = width(f, r) == Some(1)
            && (super::observed(f, r).is_some_and(|(x, _)| width(f, x).is_some_and(|w| w > 1))
                || (matches!(
                    p,
                    P::Unop(Unop::Not | Unop::Identity, _)
                        | P::Nary(NaryOp::Nor | NaryOp::Or, _)
                        | P::Sel { .. }
                ) && args.iter().any(|x| flags[x.index].2)));
        flags[r.index] = (shifted, selected, observing);
        if selected && observing && (allow_source || shifted) {
            roots.push(r)
        }
    }
    if roots.is_empty() {
        out.reasons.count(Why::Seed);
        return out;
    }
    let mut budget = Budget::default();
    let mut covered = HashSet::new();
    for r in roots.into_iter().rev() {
        if covered.contains(&r) {
            continue;
        }
        if budget.work == 0 || budget.valuations == 0 {
            out.reasons.count(Why::Work);
            break;
        }
        let result = if shift {
            attempt(f, r, &uses, ret, Domain::Named([None, None]), &mut budget)
        } else {
            Err(Why::Seed)
        };
        let result = match result {
            Err(Why::Seed | Why::Unsupported | Why::Dependencies | Why::Invariant)
                if allow_source =>
            {
                attempt(f, r, &uses, ret, Domain::Source(None), &mut budget)
            }
            other => other,
        };
        match result {
            Ok((plan, inside)) if inside.is_disjoint(&covered) => {
                covered.extend(inside);
                out.plans.push(plan)
            }
            Ok(_) => {
                out.reasons.count(Why::OverlappingRoot);
            }
            Err(reason) => {
                out.reasons.count(reason);
            }
        }
    }
    out.visits = FUNCTION_WORK - budget.work;
    out.valuations = FUNCTION_VALUATIONS - budget.valuations;
    out
}

struct Budget {
    work: usize,
    valuations: usize,
    cache: HashMap<R, Res<(Option<R>, Vec<usize>)>>,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            work: FUNCTION_WORK,
            valuations: FUNCTION_VALUATIONS,
            cache: HashMap::new(),
        }
    }
}
impl Budget {
    fn charge(&mut self, num: usize) -> Res<()> {
        self.work = self.work.checked_sub(num).ok_or(Why::Work)?;
        Ok(())
    }
}

#[derive(Debug)]
enum Domain {
    Named([Option<R>; 2]),
    Source(Option<R>),
}
#[derive(Clone, Copy)]
enum Role {
    Shift,
    Select,
}
fn width(f: &ir::Fn, r: R) -> Option<usize> {
    match f.get_node_ty(r) {
        Type::Bits(w) => Some(*w),
        _ => None,
    }
}

impl Domain {
    fn len(&self, f: &ir::Fn) -> usize {
        match self {
            Self::Named(_) => 4,
            Self::Source(None) => 1,
            Self::Source(Some(x)) => 1usize << width(f, *x).unwrap(),
        }
    }
    fn bind(&mut self, f: &ir::Fn, mut r: R, role: Role, k: usize, b: &mut Budget) -> Res<usize> {
        for _ in 0..REGION {
            b.charge(1)?;
            match &f.get_node(r).payload {
                P::Literal(v) => return count(v),
                P::ZeroExt { arg, .. } | P::Unop(Unop::Identity, arg) => r = *arg,
                _ => break,
            }
        }
        match self {
            Self::Named(slots) => {
                if width(f, r) != Some(1) {
                    return Err(Why::Dependencies);
                }
                let i = match role {
                    Role::Shift => 0,
                    Role::Select => 1,
                };
                if slots[i].replace(r).is_some_and(|p| p != r) {
                    return Err(Why::Dependencies);
                }
                Ok((k >> i) & 1)
            }
            Self::Source(source) => {
                if !b.cache.contains_key(&r) {
                    let expr = control_table(f, r, b);
                    b.cache.insert(r, expr);
                }
                let (leaf, values) = b.cache.get(&r).unwrap().as_ref().map_err(|e| *e)?;
                if let Some(x) = *leaf {
                    if source.replace(x).is_some_and(|p| p != x) {
                        return Err(Why::Dependencies);
                    }
                    values.get(k).copied().ok_or(Why::Dependencies)
                } else {
                    Ok(values[0])
                }
            }
        }
    }
}

/// Saturate by numerical magnitude. Wide encodings of zero and small distances
/// remain exact; callers bound any select by its actual arm count.
fn count(v: &IrValue) -> Res<usize> {
    let bits = v.as_bits().map_err(|_| Why::Unsupported)?;
    // PIR checks numerical fit, including for encodings wider than 64 bits.
    Ok(usize::try_from(bits.to_u64().unwrap_or(u64::MAX)).unwrap_or(usize::MAX))
}

fn control_table(f: &ir::Fn, r: R, b: &mut Budget) -> Res<(Option<R>, Vec<usize>)> {
    let limits = Limits {
        nodes: CONTROL_OPS,
        edges: REGION,
        input_bits: CONTROL_BITS,
        value_bits: 256,
    };
    let cone = BitsCone::compile(f, r, limits, &mut b.work).map_err(|err| match err {
        ConeError::Work => Why::Work,
        ConeError::Budget => Why::ControlCone,
        ConeError::InputDomain => Why::Dependencies,
        ConeError::Unsupported | ConeError::Effect | ConeError::Cycle => Why::Unsupported,
    })?;
    let input = cone.input();
    let nstates = input.map_or(1, |(_, width)| 1usize << width);
    b.valuations = b
        .valuations
        .checked_sub(nstates)
        .ok_or(Why::ControlValuations)?;
    b.charge(nstates * cone.work())?;
    let mut table = Vec::with_capacity(nstates);
    for k in 0..nstates {
        let value = input
            .map(|(_, width)| IrValue::make_ubits(width, k as u64))
            .transpose()
            .map_err(|_| Why::Unsupported)?;
        table.push(count(
            &cone
                .evaluate(value.as_ref())
                .map_err(|_| Why::Unsupported)?,
        )?);
    }
    Ok((input.map(|(node, _)| node), table))
}
fn direct_shift_or_literal_choice(f: &ir::Fn, word: R) -> bool {
    let is_shift = |r| {
        matches!(
            f.get_node(r).payload,
            P::Binop(Binop::Shll | Binop::Shrl, _, _)
        )
    };
    let is_literal = |r| matches!(f.get_node(r).payload, P::Literal(_));
    if let P::Sel {
        cases,
        default: None,
        ..
    } = &f.get_node(word).payload
        && let &[lhs, rhs] = cases.as_slice()
    {
        (is_shift(lhs) && is_literal(rhs)) || (is_literal(lhs) && is_shift(rhs))
    } else {
        is_shift(word)
    }
}

fn attempt(
    f: &ir::Fn,
    root: R,
    uses: &[usize],
    ret: R,
    mut domain: Domain,
    b: &mut Budget,
) -> Res<(Plan, HashSet<R>)> {
    let named = matches!(domain, Domain::Named(_));
    let mut start = root;
    let mut inverted = false;
    let mut inside = HashSet::new();
    loop {
        if inside.len() >= REGION {
            return Err(Why::Work);
        }
        inside.insert(start);
        match f.get_node(start).payload {
            P::Unop(op @ (Unop::Not | Unop::Identity), a) => {
                inverted ^= op == Unop::Not;
                start = a;
            }
            _ => break,
        }
    }
    let nor = matches!(f.get_node(start).payload, P::Nary(NaryOp::Nor, _));
    inverted ^= nor;
    if let Some((word, flip)) = super::observed(f, start) {
        // The ordinary shift-reduction rule owns these forms when profitable.
        if direct_shift_or_literal_choice(f, word) {
            return Err(Why::Seed);
        }
        start = word;
        inverted ^= flip
    }
    let mut cells = Vec::new();
    let mut leaves = HashSet::new();
    let mut controls = HashSet::new();
    while cells.len() < domain.len(f) {
        let k = cells.len();
        let mut stack = vec![(start, 0, width(f, start).ok_or(Why::Unsupported)?)];
        let mut seen = HashSet::new();
        let mut spans = Vec::new();
        let mut asserted = false;
        while let Some((n, lo, hi)) = stack.pop() {
            if lo >= hi || !seen.insert((n, lo, hi)) {
                continue;
            }
            b.charge(1)?;
            if seen.len() > REGION || stack.len() > REGION {
                return Err(Why::Work);
            }
            let w = width(f, n).ok_or(Why::Unsupported)?;
            if hi > w {
                return Err(Why::Unsupported);
            }
            if let Some((x, flip)) = super::observed(f, n) {
                if flip {
                    return Err(Why::Unsupported);
                }
                inside.insert(n);
                stack.push((x, 0, width(f, x).ok_or(Why::Unsupported)?));
                continue;
            }
            match &f.get_node(n).payload {
                P::Param | P::TupleIndex { .. } => {
                    leaves.insert(n);
                    spans.push((n, lo, hi));
                    continue;
                }
                P::Literal(v) => {
                    if !v.bits_equals_u64_value(0) {
                        if named {
                            return Err(Why::Unsupported);
                        }
                        let bits = v.to_bits().map_err(|_| Why::Unsupported)?;
                        asserted |= !bits.width_slice(lo as i64, (hi - lo) as i64).is_zero();
                    }
                    continue;
                }
                P::Unop(Unop::Identity, x) => stack.push((*x, lo, hi)),
                P::ZeroExt { arg, .. } => {
                    stack.push((*arg, lo, hi.min(width(f, *arg).ok_or(Why::Unsupported)?)))
                }
                P::BitSlice { arg, start, .. } => stack.push((
                    *arg,
                    start.checked_add(lo).ok_or(Why::Unsupported)?,
                    start.checked_add(hi).ok_or(Why::Unsupported)?,
                )),
                P::Nary(NaryOp::Concat, args) if args.len() <= REGION => {
                    let mut off = 0usize;
                    for &a in args.iter().rev() {
                        let end = off
                            .checked_add(width(f, a).ok_or(Why::Unsupported)?)
                            .ok_or(Why::Unsupported)?;
                        if lo < end && off < hi {
                            stack.push((a, lo.saturating_sub(off), hi.min(end) - off))
                        }
                        off = end;
                        if off >= hi {
                            break;
                        }
                    }
                }
                P::Nary(op, args)
                    if args.len() <= REGION
                        && (*op == NaryOp::Or || (*op == NaryOp::Nor && n == start && nor)) =>
                {
                    stack.extend(args.iter().map(|&a| (a, lo, hi)))
                }
                P::Sel {
                    selector,
                    cases,
                    default,
                } if !cases.is_empty() && cases.len() <= REGION => {
                    if named
                        && (width(f, *selector) != Some(1) || default.is_some() || cases.len() != 2)
                    {
                        return Err(Why::Unsupported);
                    }
                    let v = domain.bind(f, *selector, Role::Select, k, b)?;
                    controls.insert(*selector);
                    stack.push((
                        *cases.get(v).or(default.as_ref()).ok_or(Why::Unsupported)?,
                        lo,
                        hi,
                    ));
                }
                P::PrioritySel {
                    selector,
                    cases,
                    default,
                } if !named
                    && cases.len() <= 64
                    && width(f, *selector).is_some_and(|w| w <= 64) =>
                {
                    let v = domain.bind(f, *selector, Role::Select, k, b)?;
                    controls.insert(*selector);
                    let arm = if v == 0 {
                        default.as_ref()
                    } else {
                        cases.get(v.trailing_zeros() as usize)
                    };
                    stack.push((*arm.ok_or(Why::Unsupported)?, lo, hi));
                }
                P::Binop(op @ (Binop::Shrl | Binop::Shll), data, amount) => {
                    let v = domain.bind(f, *amount, Role::Shift, k, b)?;
                    controls.insert(*amount);
                    let (l, h) = if *op == Binop::Shrl {
                        (lo.saturating_add(v), hi.saturating_add(v).min(w))
                    } else {
                        (lo.saturating_sub(v), hi.saturating_sub(v))
                    };
                    stack.push((*data, l, h));
                }
                _ => return Err(Why::Unsupported),
            }
            inside.insert(n);
        }
        if named && spans.is_empty() {
            return Err(Why::Invariant);
        }
        let cell = if asserted {
            Cell::Const(true)
        } else if spans.is_empty() {
            Cell::Const(false)
        } else {
            spans.sort_by_key(|(r, lo, _)| (r.index, *lo));
            let mut merged: Vec<(R, usize, usize)> = Vec::new();
            for (r, l, h) in spans {
                if let Some((prev, _, end)) = merged.last_mut()
                    && *prev == r
                    && l <= *end
                {
                    *end = (*end).max(h)
                } else {
                    merged.push((r, l, h))
                }
            }
            if merged.len() > SPANS {
                return Err(Why::Fragments);
            }
            Cell::Any(merged)
        };
        cells.push(cell)
    }
    let source = match domain {
        Domain::Named([Some(a), Some(c)]) if a != c && leaves.len() == 1 => vec![a, c],
        Domain::Source(Some(x)) => vec![x],
        _ => return Err(Why::Dependencies),
    };
    if cells.iter().all(|c| c == &cells[0]) || leaves.is_empty() {
        return Err(Why::Invariant);
    }
    // The below-frontier data and control cones may stay live. Replacing a
    // predicate must dispose of every internal shift/wire it would duplicate.
    let mut peeled = HashSet::new();
    let mut counts = HashMap::<R, usize>::new();
    let mut pending = vec![root];
    while let Some(n) = pending.pop() {
        if !peeled.insert(n) {
            continue;
        }
        b.charge(1)?;
        if peeled.len() > REGION || counts.len() > REGION {
            return Err(Why::Work);
        }
        for a in operands(&f.get_node(n).payload) {
            if leaves.contains(&a)
                || controls.contains(&a)
                || source.contains(&a)
                || a == ret
                || matches!(f.get_node(a).payload, P::Param | P::Literal(_))
            {
                continue;
            }
            let c = counts.entry(a).or_default();
            *c += 1;
            if *c == uses[a.index] {
                pending.push(a)
            }
        }
    }
    if !inside.is_subset(&peeled) {
        return Err(Why::Shared);
    }
    Ok((
        Plan {
            root,
            source,
            named,
            cells,
            inverted,
        },
        inside,
    ))
}

#[cfg(test)]
mod tests;
