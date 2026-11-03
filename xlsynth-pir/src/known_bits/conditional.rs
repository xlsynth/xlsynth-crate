// SPDX-License-Identifier: Apache-2.0

//! Local Boolean queries with the low bits of a scalar word fixed uniformly.

use super::{KnownBits, KnownValue, eval};
use crate::{
    IrBits,
    ir::{self, NaryOp, NodePayload, NodeRef, Type, Unop},
    ir_utils,
};
use std::collections::{HashMap, HashSet};

/// A known bit holds under the premise. Infeasible means the analysis proved a
/// contradiction; an unknown bit makes no claim about the target or the
/// feasibility of the premise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditionalBit {
    Known(bool),
    Unknown,
    Infeasible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LowWordBooleanFacts {
    pub unconditional: Option<bool>,
    pub at_zero: ConditionalBit,
    pub at_ones: ConditionalBit,
}

impl LowWordBooleanFacts {
    pub fn any_infeasible(&self) -> bool {
        self.at_zero == ConditionalBit::Infeasible || self.at_ones == ConditionalBit::Infeasible
    }

    pub fn gains_known_bit(&self) -> bool {
        self.unconditional.is_none()
            && (matches!(self.at_zero, ConditionalBit::Known(_))
                || matches!(self.at_ones, ConditionalBit::Known(_)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryLimit {
    Work,
    Unsupported,
    UnmodeledInverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LowWordQuery {
    /// No variable source of the low word occurs in the target's dependency
    /// cone.
    Disjoint,
    /// A completed conditional transfer. Unknown endpoints do not establish
    /// reachability.
    Constrained(LowWordBooleanFacts),
    /// The query did not complete. No independence or conditional fact was
    /// established.
    Inconclusive(QueryLimit),
}

/// Queries a Boolean in a valid function under each uniform value of
/// `word[width-1:0]`. High bits retain their ordinary facts. No fact escapes
/// its query or modifies the function.
pub fn boolean_under_low_word_extremes(
    f: &ir::Fn,
    word: NodeRef,
    width: usize,
    target: NodeRef,
) -> LowWordQuery {
    query(f, word, width, target).unwrap_or_else(LowWordQuery::Inconclusive)
}

fn query(
    f: &ir::Fn,
    word: NodeRef,
    width: usize,
    target: NodeRef,
) -> Result<LowWordQuery, QueryLimit> {
    let Type::Bits(word_width) = f.nodes.get(word.index).ok_or(QueryLimit::Unsupported)?.ty else {
        return Err(QueryLimit::Unsupported);
    };
    if width == 0
        || width > word_width
        || f.nodes.get(target.index).ok_or(QueryLimit::Unsupported)?.ty != Type::Bits(1)
    {
        return Err(QueryLimit::Unsupported);
    }
    if word != target
        && matches!(f.get_node(word).payload, NodePayload::Param)
        && matches!(f.get_node(target).payload, NodePayload::Param)
    {
        return Ok(LowWordQuery::Disjoint);
    }
    if word_width > MAX_BITS {
        return Err(QueryLimit::Work);
    }
    let mut work = MAX_WORK;
    let mut givens = HashMap::<NodeRef, Given>::new();
    let mut impossible = [false; 2];
    for bit in 0..width {
        match source_bit(f, word, bit, &mut work)? {
            SourceBit::Literal(value) => impossible[usize::from(!value)] = true,
            SourceBit::Source(n, i, inverted) => {
                let bytes = f.get_node(n).ty.bit_count().div_ceil(8);
                if bytes > MAX_BITS.div_ceil(8) {
                    return Err(QueryLimit::Work);
                }
                let given = givens.entry(n).or_insert_with(|| Given {
                    mask: vec![0; bytes],
                    zero: vec![0; bytes],
                });
                let b = 1u8 << (i % 8);
                let byte = i / 8;
                if given.mask[byte] & b != 0 && (given.zero[byte] & b != 0) != inverted {
                    impossible = [true; 2];
                }
                given.mask[byte] |= b;
                if inverted {
                    given.zero[byte] |= b;
                }
            }
        }
    }
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    collect(f, target, &mut visited, &mut order, &mut work)?;
    let routed_reaches_target = givens.keys().any(|n| visited.contains(n));
    if !routed_reaches_target && !visited.contains(&word) {
        // A target can read an input of a computed word without consuming its
        // output. Prove disjoint primary support through the actual low-bit
        // origins first.
        if givens
            .keys()
            .all(|n| matches!(f.get_node(*n).payload, NodePayload::Param))
        {
            return Ok(LowWordQuery::Disjoint);
        }
        let target_params: HashSet<_> = order
            .iter()
            .copied()
            .filter(|&n| {
                let p = &f.get_node(n).payload;
                !matches!(p, NodePayload::Literal(_) | NodePayload::Nil)
                    && ir_utils::operands(p).is_empty()
            })
            .collect();
        for &n in givens.keys() {
            collect(f, n, &mut visited, &mut order, &mut work)?;
        }
        let mut shared = target_params;
        for &nr in &order {
            if ir_utils::operands(&f.get_node(nr).payload)
                .iter()
                .any(|r| shared.contains(r))
            {
                shared.insert(nr);
            }
        }
        if !givens.keys().any(|n| shared.contains(n)) {
            return Ok(LowWordQuery::Disjoint);
        }
        return Err(QueryLimit::UnmodeledInverse);
    }
    let primary_givens = givens
        .keys()
        .all(|&n| matches!(f.get_node(n).payload, NodePayload::Param));
    // Bind the original word and every justified upstream signal on a modeled
    // premise path.
    let bytes = word_width.div_ceil(8);
    let mut direct = Given {
        mask: vec![0; bytes],
        zero: vec![0; bytes],
    };
    direct.mask[..width / 8].fill(0xff);
    if width % 8 != 0 {
        direct.mask[width / 8] = (1u8 << (width % 8)) - 1;
    }
    givens.entry(word).or_insert(direct);
    for &n in givens.keys() {
        collect(f, n, &mut visited, &mut order, &mut work)?;
    }
    let result = evaluate(f, target, &order, &givens, impossible).ok_or(QueryLimit::Unsupported)?;
    if result.unconditional.is_some() || result.gains_known_bit() || result.any_infeasible() {
        return Ok(LowWordQuery::Constrained(result));
    }

    // A shared node can be read solely through its unconditioned high bits.
    // Require a forward path from a given bit before interpreting two unknown
    // endpoint transfers.
    let mut pending = vec![(target, 0)];
    let mut seen = HashSet::new();
    let mut complete = true;
    while let Some((n, i)) = pending.pop() {
        let SourceBit::Source(n, i, _) = source_bit(f, n, i, &mut work)? else {
            continue;
        };
        if givens
            .get(&n)
            .is_some_and(|g| g.mask[i / 8] & (1 << (i % 8)) != 0)
        {
            return Ok(LowWordQuery::Constrained(result));
        }
        if !seen.insert((n, i)) {
            continue;
        }
        let node = f.get_node(n);
        if matches!(node.payload, NodePayload::Param) {
            continue;
        }
        // A scalar transfer consumes its operands as whole values. A partial
        // demand on an opaque vector does not identify which source bits reach
        // that demand.
        if node.ty != Type::Bits(1) {
            complete = false;
            continue;
        }
        for arg in ir_utils::operands(&node.payload) {
            let Type::Bits(w) = f.get_node(arg).ty else {
                return Err(QueryLimit::Unsupported);
            };
            work = work.checked_sub(w).ok_or(QueryLimit::Work)?;
            pending.extend((0..w).map(|bit| (arg, bit)));
        }
    }
    if complete && primary_givens {
        Ok(LowWordQuery::Disjoint)
    } else {
        Err(QueryLimit::UnmodeledInverse)
    }
}

fn evaluate(
    f: &ir::Fn,
    target: NodeRef,
    order: &[NodeRef],
    givens: &HashMap<NodeRef, Given>,
    mut impossible: [bool; 2],
) -> Option<LowWordBooleanFacts> {
    let mut base = HashMap::new();
    let mut seeds = HashMap::new();
    for &nr in order {
        let node = f.get_node(nr);
        let ordinary = eval::evaluate_with(node, &|r| base.get(&r)).ok()?;
        if let Some(given) = givens.get(&nr) {
            let g = given.facts(node.ty.bit_count())?;
            for (k, seed) in g.iter().enumerate() {
                if refine(ordinary.as_bits()?, seed).is_none() {
                    impossible[k] = true;
                }
            }
            seeds.insert(nr, g);
        }
        base.insert(nr, ordinary);
    }
    let unconditional = known_bit(base.get(&target)?.as_bits()?);
    let mut results = [ConditionalBit::Unknown; 2];
    for k in 0..2 {
        if impossible[k] {
            results[k] = ConditionalBit::Infeasible;
            continue;
        }
        // Store only changed facts. Independent operands continue to use the
        // unconditional table.
        let mut changes = HashMap::new();
        for &nr in order {
            let node = f.get_node(nr);
            let affected = ir_utils::operands(&node.payload)
                .iter()
                .any(|u| changes.contains_key(u));
            let seed = seeds.get(&nr).map(|s| &s[k]);
            if !affected && seed.is_none() {
                continue;
            }
            let computed = if affected {
                Some(
                    eval::evaluate_with(node, &|r| changes.get(&r).or_else(|| base.get(&r)))
                        .ok()?,
                )
            } else {
                None
            };
            let value = computed.as_ref().unwrap_or(&base[&nr]);
            let result = if let Some(given) = seed {
                let Some(bits) = refine(value.as_bits()?, given) else {
                    results[k] = ConditionalBit::Infeasible;
                    break;
                };
                KnownValue::Bits(bits)
            } else {
                computed?
            };
            if result != base[&nr] {
                changes.insert(nr, result);
            }
        }
        if results[k] != ConditionalBit::Infeasible {
            let known = changes.get(&target).unwrap_or(&base[&target]).as_bits()?;
            results[k] = known_bit(known).map_or(ConditionalBit::Unknown, ConditionalBit::Known);
        }
    }
    Some(LowWordBooleanFacts {
        unconditional,
        at_zero: results[0],
        at_ones: results[1],
    })
}

// Bound local graph, shape, and routing work independently of function size.
const MAX_NODES: usize = 256;
const MAX_BITS: usize = 2048;
const MAX_WORK: usize = 32768;

struct Given {
    mask: Vec<u8>,
    zero: Vec<u8>,
}
impl Given {
    fn facts(&self, width: usize) -> Option<[KnownBits; 2]> {
        let mask = IrBits::from_le_bytes(width, &self.mask).ok()?;
        let zero = IrBits::from_le_bytes(width, &self.zero).ok()?;
        let one = mask.and(&zero.not());
        Some([
            KnownBits::from_mask_value(mask.clone(), zero).ok()?,
            KnownBits::from_mask_value(mask, one).ok()?,
        ])
    }
}
fn known_bit(v: &KnownBits) -> Option<bool> {
    v.mask()
        .get_bit(0)
        .ok()?
        .then(|| v.value().get_bit(0).expect("same-width fact"))
}
fn refine(value: &KnownBits, given: &KnownBits) -> Option<KnownBits> {
    if !value
        .mask()
        .and(given.mask())
        .and(&value.value().xor(given.value()))
        .is_zero()
    {
        return None;
    }
    KnownBits::from_mask_value(
        value.mask().or(given.mask()),
        value.value().or(given.value()),
    )
    .ok()?
    .with_popcount_bounds(value.min_ones(), value.max_ones())
    .ok()
}

enum SourceBit {
    Literal(bool),
    Source(NodeRef, usize, bool),
}
fn source_bit(
    f: &ir::Fn,
    mut n: NodeRef,
    mut i: usize,
    work: &mut usize,
) -> Result<SourceBit, QueryLimit> {
    let mut inverted = false;
    loop {
        *work = work.checked_sub(1).ok_or(QueryLimit::Work)?;
        let node = f.nodes.get(n.index).ok_or(QueryLimit::Unsupported)?;
        let Type::Bits(width) = node.ty else {
            return Err(QueryLimit::Unsupported);
        };
        if width > MAX_BITS {
            return Err(QueryLimit::Work);
        }
        if i >= width {
            return Err(QueryLimit::Unsupported);
        }
        match &node.payload {
            NodePayload::Literal(v) => {
                return Ok(SourceBit::Literal(
                    v.to_bits()
                        .map_err(|_| QueryLimit::Unsupported)?
                        .get_bit(i)
                        .map_err(|_| QueryLimit::Unsupported)?
                        ^ inverted,
                ));
            }
            NodePayload::Unop(op @ (Unop::Identity | Unop::Not | Unop::Reverse), arg) => {
                match op {
                    Unop::Not => inverted = !inverted,
                    Unop::Reverse => i = width - 1 - i,
                    _ => {}
                }
                n = *arg;
            }
            NodePayload::BitSlice { arg, start, .. } => {
                i = i.checked_add(*start).ok_or(QueryLimit::Unsupported)?;
                n = *arg;
            }
            NodePayload::ZeroExt { arg, .. } | NodePayload::SignExt { arg, .. } => {
                let Type::Bits(w) = f.nodes.get(arg.index).ok_or(QueryLimit::Unsupported)?.ty
                else {
                    return Err(QueryLimit::Unsupported);
                };
                if i >= w && matches!(node.payload, NodePayload::ZeroExt { .. }) {
                    return Ok(SourceBit::Literal(inverted));
                }
                i = i.min(w.checked_sub(1).ok_or(QueryLimit::Unsupported)?);
                n = *arg;
            }
            NodePayload::Nary(NaryOp::Concat, parts) => {
                let mut found = None;
                for p in parts.iter().rev() {
                    *work = work.checked_sub(1).ok_or(QueryLimit::Work)?;
                    let Type::Bits(w) = f.nodes.get(p.index).ok_or(QueryLimit::Unsupported)?.ty
                    else {
                        return Err(QueryLimit::Unsupported);
                    };
                    if i < w {
                        found = Some(*p);
                        break;
                    }
                    i -= w;
                }
                n = found.ok_or(QueryLimit::Unsupported)?;
            }
            _ => return Ok(SourceBit::Source(n, i, inverted)),
        }
    }
}

fn collect(
    f: &ir::Fn,
    n: NodeRef,
    visited: &mut HashSet<NodeRef>,
    order: &mut Vec<NodeRef>,
    work: &mut usize,
) -> Result<(), QueryLimit> {
    if !visited.insert(n) {
        return Ok(());
    }
    if visited.len() > MAX_NODES {
        return Err(QueryLimit::Work);
    }
    let node = f.nodes.get(n.index).ok_or(QueryLimit::Unsupported)?;
    // This query accepts scalar transfer cones. Aggregate/dynamic container
    // shapes can require work unrelated to the bounded number of scalar result
    // bits.
    let Type::Bits(bits) = node.ty else {
        return Err(QueryLimit::Unsupported);
    };
    if bits > MAX_BITS {
        return Err(QueryLimit::Work);
    }
    let arity = match &node.payload {
        NodePayload::Nary(_, v) | NodePayload::Invoke { operands: v, .. } => v.len(),
        NodePayload::Sel { cases, .. }
        | NodePayload::PrioritySel { cases, .. }
        | NodePayload::OneHotSel { cases, .. } => cases.len(),
        NodePayload::ExtNaryAdd { terms, .. } => terms.len(),
        NodePayload::ArrayIndex { indices, .. } | NodePayload::ArrayUpdate { indices, .. } => {
            indices.len()
        }
        NodePayload::CountedFor { invariant_args, .. } => invariant_args.len(),
        _ => 0,
    };
    *work = work
        .checked_sub(1 + bits + arity.min(MAX_WORK))
        .ok_or(QueryLimit::Work)?;
    for r in ir_utils::operands(&node.payload) {
        collect(f, r, visited, order, work)?;
    }
    order.push(n);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IrValue, ir_eval, ir_parser::Parser};

    #[test]
    fn conditional_results_hold_for_every_concrete_input() {
        use ConditionalBit::{Infeasible, Known, Unknown};
        // Each word is six bits. Some views repeat or complement the same
        // original signal.
        for (w, c, width, expected) in [
            ("identity(x)", "and(bit, flag)", 4, (Known(false), Unknown)),
            ("reverse(x)", "or(high, flag)", 3, (Unknown, Known(true))),
            (
                "concat(pair, pair, pair)",
                "xor_reduce(pair)",
                6,
                (Known(false), Known(false)),
            ),
            (
                "concat(pair, inverse_pair, pair)",
                "or_reduce(pair)",
                6,
                (Infeasible, Infeasible),
            ),
            (
                "concat(hi, pair, pad)",
                "or_reduce(pair)",
                4,
                (Known(false), Infeasible),
            ),
            (
                "sign_ext(hi, new_bit_count=6)",
                "or_reduce(hi)",
                6,
                (Known(false), Known(true)),
            ),
            (
                "identity(x)",
                "eq(sum, one)",
                4,
                (Known(true), Known(false)),
            ),
            ("identity(x)", "or(bit, nbit)", 4, (Unknown, Known(true))),
        ] {
            let text = format!(
                "package c
top fn check(x: bits[6] id=1, flag: bits[1] id=2) -> (bits[6], bits[1], bits[1]) {{
  pair: bits[2] = bit_slice(x, start=0, width=2, id=3)
  inverse_pair: bits[2] = not(pair, id=4)
  hi: bits[2] = bit_slice(x, start=4, width=2, id=5)
  pad: bits[2] = literal(value=0, id=6)
  word: bits[6] = {}, id=7)
  bit: bits[1] = bit_slice(x, start=2, width=1, id=8)
  high: bits[1] = bit_slice(x, start=4, width=1, id=9)
  low: bits[4] = bit_slice(x, start=0, width=4, id=10)
  one: bits[4] = literal(value=1, id=11)
  sum: bits[4] = add(low, one, id=12)
  nbit: bits[1] = not(flag, id=13)
  target: bits[1] = {}, id=14)
  ret out: (bits[6], bits[1], bits[1]) = tuple(word, target, target, id=15)
}}",
                w.strip_suffix(')').unwrap(),
                c.strip_suffix(')').unwrap()
            );
            let pkg = Parser::new(&text).parse_and_validate_package().unwrap();
            let f = pkg.get_top_fn().unwrap();
            let word = id(f, 7);
            let target = id(f, 14);
            let LowWordQuery::Constrained(facts) =
                boolean_under_low_word_extremes(f, word, width, target)
            else {
                panic!("expected facts");
            };
            assert_eq!((facts.at_zero, facts.at_ones), expected, "{w} {c}");
            let mut seen = [[false; 2]; 2];
            for x in 0..64 {
                for flag in 0..2 {
                    let args = [
                        IrValue::make_ubits(6, x).unwrap(),
                        IrValue::make_ubits(1, flag).unwrap(),
                    ];
                    let ir_eval::FnEvalResult::Success(out) = ir_eval::eval_fn(f, &args) else {
                        panic!("valid function");
                    };
                    let value = out.value.get_element(0).unwrap().to_u64().unwrap();
                    let result = out.value.get_element(1).unwrap().to_u64().unwrap() as usize;
                    let mask = (1u64 << width) - 1;
                    for (b, low) in [0, mask].into_iter().enumerate() {
                        if value & mask == low {
                            seen[b][result] = true;
                        }
                    }
                }
            }
            for (b, prediction) in [facts.at_zero, facts.at_ones].iter().enumerate() {
                match prediction {
                    Infeasible => assert_eq!(seen[b], [false; 2]),
                    Known(v) => assert!(!seen[b][usize::from(!v)]),
                    Unknown => {}
                }
            }
            assert_eq!(
                facts.gains_known_bit(),
                expected.0 != Infeasible
                    && (matches!(expected.0, Known(_)) || matches!(expected.1, Known(_)))
            );
            let unconditional = super::super::analyze_fn(f).unwrap();
            assert_eq!(
                facts.unconditional,
                known_bit(unconditional.bits(target).unwrap())
            );
        }
    }

    #[test]
    fn partial_high_byte_and_independent_targets_remain_free() {
        let text = "package c
top fn f(x: bits[73] id=1, flag: bits[1] id=2) -> bits[1] {
  below: bits[1] = bit_slice(x, start=64, width=1, id=3)
  above: bits[1] = bit_slice(x, start=65, width=1, id=4)
  zero: bits[1] = literal(value=0, id=5)
  always: bits[1] = and(below, zero, id=6)
  ret predicate: bits[1] = and(below, above, flag, id=7)
}";
        let p = Parser::new(text).parse_and_validate_package().unwrap();
        let f = p.get_top_fn().unwrap();
        let w = id(f, 1);
        let target = id(f, 7);
        let LowWordQuery::Constrained(a) = boolean_under_low_word_extremes(f, w, 65, target) else {
            panic!("expected facts");
        };
        assert_eq!(
            (a.at_zero, a.at_ones),
            (ConditionalBit::Known(false), ConditionalBit::Unknown)
        );
        let LowWordQuery::Constrained(low) = boolean_under_low_word_extremes(f, w, 65, id(f, 3))
        else {
            panic!("expected facts");
        };
        assert_eq!(low.at_ones, ConditionalBit::Known(true));
        assert_eq!(
            boolean_under_low_word_extremes(f, w, 65, id(f, 4)),
            LowWordQuery::Disjoint
        );
        let LowWordQuery::Constrained(fixed) = boolean_under_low_word_extremes(f, w, 65, id(f, 6))
        else {
            panic!("expected facts");
        };
        assert_eq!(fixed.unconditional, Some(false));
        assert!(!fixed.gains_known_bit());
        assert_eq!(
            boolean_under_low_word_extremes(f, w, 65, id(f, 2)),
            LowWordQuery::Disjoint
        );
        assert_eq!(
            boolean_under_low_word_extremes(f, w, 0, target),
            LowWordQuery::Inconclusive(QueryLimit::Unsupported)
        );
        assert_eq!(
            boolean_under_low_word_extremes(f, w, 74, target),
            LowWordQuery::Inconclusive(QueryLimit::Unsupported)
        );
        assert_eq!(
            boolean_under_low_word_extremes(f, w, 65, w),
            LowWordQuery::Inconclusive(QueryLimit::Unsupported)
        );
        assert_eq!(
            boolean_under_low_word_extremes(f, NodeRef { index: usize::MAX }, 65, target),
            LowWordQuery::Inconclusive(QueryLimit::Unsupported)
        );
    }

    #[test]
    fn budget_is_local_to_the_requested_cone() {
        let mut text = String::from(
            "package q\ntop fn f(x: bits[8] id=1, v: bits[1] id=2) -> bits[1] {\n  bit: bits[1] = bit_slice(x, start=0, width=1, id=3)\n",
        );
        let mut prev = "bit".to_string();
        for idx in 4..320 {
            let n = format!("n{idx}");
            text.push_str(&format!("  {n}: bits[1] = xor({prev}, v, id={idx})\n"));
            prev = n;
        }
        text.push_str(&format!(
            "  ret many: bits[1] = identity({prev}, id=320)\n}}"
        ));
        let p = Parser::new(&text).parse_and_validate_package().unwrap();
        let f = p.get_top_fn().unwrap();
        assert!(
            matches!(boolean_under_low_word_extremes(f,id(f,1),4,id(f,3)),LowWordQuery::Constrained(v) if v.gains_known_bit())
        );
        assert_eq!(
            boolean_under_low_word_extremes(f, id(f, 1), 4, id(f, 320)),
            LowWordQuery::Inconclusive(QueryLimit::Work)
        );
    }

    #[test]
    fn distinguishes_unknown_disjoint_and_unmodeled_upstream_support() {
        let text = "package q
top fn f(x: bits[8] id=1, y: bits[8] id=2, g: bits[1] id=3) -> bits[1] {
  word: bits[8] = add(x, y, id=4)
  xbit: bits[1] = bit_slice(x, start=0, width=1, id=5)
  first: bits[1] = bit_slice(word, start=0, width=1, id=6)
  ret uncertain: bits[1] = xor(first, g, id=7)
}";
        let p = Parser::new(text).parse_and_validate_package().unwrap();
        let f = p.get_top_fn().unwrap();
        let word = id(f, 4);
        assert_eq!(
            boolean_under_low_word_extremes(f, word, 5, id(f, 5)),
            LowWordQuery::Inconclusive(QueryLimit::UnmodeledInverse)
        );
        assert_eq!(
            boolean_under_low_word_extremes(f, word, 5, id(f, 3)),
            LowWordQuery::Disjoint
        );
        let LowWordQuery::Constrained(v) = boolean_under_low_word_extremes(f, word, 5, id(f, 7))
        else {
            panic!("word is on the carry path");
        };
        assert_eq!(
            (v.at_zero, v.at_ones),
            (ConditionalBit::Unknown, ConditionalBit::Unknown)
        );
        assert!(!v.gains_known_bit());

        let big=Parser::new("package q\ntop fn f(x: bits[65536] id=1, c: bits[1] id=2) -> bits[1] {\n  ret low: bits[1] = bit_slice(x, start=1, width=1, id=3)\n}").parse_and_validate_package().unwrap();
        let f = big.get_top_fn().unwrap();
        assert_eq!(
            boolean_under_low_word_extremes(f, id(f, 1), 65535, id(f, 2)),
            LowWordQuery::Disjoint
        );
        assert_eq!(
            boolean_under_low_word_extremes(f, id(f, 1), 65535, id(f, 3)),
            LowWordQuery::Inconclusive(QueryLimit::Work)
        );
    }

    #[test]
    fn reading_only_high_bits_of_an_opaque_word_does_not_model_its_low_bits() {
        let text = "package q
top fn f(x: bits[8] id=1) -> bits[1] {
  two: bits[8] = literal(value=2, id=2)
  word: bits[8] = add(x, two, id=3)
  hi: bits[1] = bit_slice(word, start=7, width=1, id=4)
  par: bits[1] = bit_slice(x, start=0, width=1, id=5)
  ret carry: bits[1] = and(hi, par, id=6)
}";
        let p = Parser::new(text).parse_and_validate_package().unwrap();
        let f = p.get_top_fn().unwrap();
        assert_eq!(
            boolean_under_low_word_extremes(f, id(f, 3), 4, id(f, 6)),
            LowWordQuery::Inconclusive(QueryLimit::UnmodeledInverse)
        );
        for x in 0..256 {
            if (x + 2) % 16 == 0 {
                let args = [IrValue::make_ubits(8, x).unwrap()];
                let ir_eval::FnEvalResult::Success(out) = ir_eval::eval_fn(f, &args) else {
                    panic!("valid function");
                };
                assert_eq!(out.value.to_u64().unwrap(), 0);
            }
        }
    }

    fn id(f: &ir::Fn, text_id: usize) -> NodeRef {
        NodeRef {
            index: f.nodes.iter().position(|v| v.text_id == text_id).unwrap(),
        }
    }
}
