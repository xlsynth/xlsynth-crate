// SPDX-License-Identifier: Apache-2.0

//! Pull a masked equality through a modular translation or reflection.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::aig::gate::PirNodeIds;
use crate::gatify::wiring;
use bitvec::prelude::{BitVec, Lsb0};
use xlsynth_pir::ir::{self, Binop, ExtNaryAddTerm, NaryOp, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils::{self, Users};
use xlsynth_pir::{IrBits, IrValue};

/// Original operations available for attributing fresh comparison gates.
pub(super) struct Origins<'a> {
    pub original: &'a ir::Fn,
    pub rewrites: &'a HashMap<usize, usize>,
    pub comparisons: &'a mut HashMap<usize, PirNodeIds>,
}

impl Origins<'_> {
    /// Attribute the comparator to the operation where all its atoms rejoin.
    fn record_merge(&mut self, f: &ir::Fn, join: NodeRef, atoms: &[Truth]) {
        // Prep only appends nodes before final compaction. If a previous pass
        // created the join, use its annotation or an original comparison.
        let ids = std::iter::once(join)
            .chain(atoms.iter().map(|&(node, _)| node))
            .find_map(|node| {
                let text_id = f.get_node(node).text_id;
                let source = self
                    .rewrites
                    .get(&text_id)
                    .copied()
                    .or_else(|| self.original.nodes.get(node.index).map(|op| op.text_id));
                if let Some(source) = source {
                    let id = u32::try_from(source).expect("node id too large for u32");
                    Some(PirNodeIds::from_slice(&[id]))
                } else {
                    self.comparisons.get(&text_id).cloned()
                }
            });
        if let Some(ids) = ids {
            self.comparisons.insert(f.get_node(atoms[0].0).text_id, ids);
        }
    }
}

/// Pulls exclusively conjoined observations back to the input of an affine
/// word.
pub(super) fn rewrite(f: &mut ir::Fn, origins: Option<Origins<'_>>) -> usize {
    let mut observations = BTreeMap::<usize, Vec<Observation>>::new();
    let mut search = ViewSearch::new(f);
    for index in 0..f.nodes.len() {
        if let Some((word, observed)) = search.observation(f, NodeRef { index }) {
            observations.entry(word.index).or_default().push(observed);
        }
    }
    if observations.is_empty() {
        return 0;
    }
    let users = ir_utils::compute_users(f);
    // All cut queries share one graph-sized allowance for edges and mask work.
    // A private parameter needs no search, even after the allowance runs out.
    let mut support_work = users.len() + users.iter().map(|(_, uses)| uses.len()).sum::<usize>();
    let mut roots = HashMap::new();
    let mut plans = Vec::new();
    for (index, atoms) in observations {
        let word = NodeRef { index };
        let width = f.get_node(word).ty.bit_count();
        // A single whole-word comparison belongs to arithmetic simplification.
        // This pass combines separate observations or removes unobserved work.
        if atoms.len() == 1 && matches!(atoms[0].fields.as_slice(), [field] if field.width == width)
        {
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
        let affine = search.sources[&word].as_ref().unwrap();
        let ObservedCube {
            mask,
            value,
            conflict,
        } = ObservedCube::merge(width, &atoms);
        // A concatenated or overlapping view can also cover the whole word.
        if atoms.len() == 1 && mask == IrBits::all_ones(width) {
            continue;
        }
        let inverse = if conflict {
            None
        } else {
            let Some(value) = affine.invert_cube(f, &mask, &value) else {
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
        if shares_input_demand(f, &users, word, affine.input, &mask, &mut support_work) {
            continue;
        }
        plans.push(Plan {
            atoms: atoms.into_iter().map(|a| (a.comparison, a.equal)).collect(),
            inverse: inverse.map(|value| Preimage {
                input: affine.input,
                mask,
                value,
            }),
        });
    }

    let count = plans.len();
    if count == 0 {
        return 0;
    }
    let mut next_id = super::next_text_id(f);
    // At most one replacement ID per original node and four helpers per plan.
    if next_id.checked_add(f.nodes.len() + 4 * count).is_none() {
        return 0;
    }
    refresh_partial_conjunction_ids(f, &users, &plans, &mut next_id, origins);
    for Plan { atoms, inverse } in plans {
        let (first, equal) = atoms[0];
        let payload = if let Some(Preimage { input, mask, value }) = inverse {
            let width = (0..mask.get_bit_count())
                .rfind(|&i| mask.get_bit(i).unwrap())
                .unwrap()
                + 1;
            let mask = mask.width_slice(0, width as i64);
            let value = value.width_slice(0, width as i64);
            let ty = Type::Bits(width);
            let needs_slice = f.get_node(input).ty != ty;
            let mut emit = |payload| {
                let index = f.nodes.len();
                f.nodes.push(ir::Node {
                    text_id: next_id,
                    name: None,
                    ty: ty.clone(),
                    payload,
                    pos: None,
                });
                next_id += 1;
                NodeRef { index }
            };
            let mut observed = input;
            if needs_slice {
                observed = emit(NodePayload::BitSlice {
                    arg: input,
                    start: 0,
                    width,
                });
            }
            if mask != IrBits::all_ones(width) {
                let m = emit(NodePayload::Literal(IrValue::from_bits(&mask)));
                observed = emit(NodePayload::Nary(NaryOp::And, vec![observed, m]));
            }
            let literal = emit(NodePayload::Literal(IrValue::from_bits(&value)));
            NodePayload::Binop(if equal { Binop::Eq } else { Binop::Ne }, observed, literal)
        } else {
            NodePayload::Literal(IrValue::bool(!equal))
        };
        f.nodes[first.index].payload = payload;
        // The group is observed only as a conjunction of these truth
        // assignments.
        for &(comparison, equal) in &atoms[1..] {
            f.nodes[comparison.index].payload = NodePayload::Literal(IrValue::bool(equal));
        }
    }
    super::mark_dead_nodes_as_nil(f);
    count
}

/// Comparisons to replace together and their exact preimage, if consistent.
struct Plan {
    atoms: Vec<Truth>,
    inverse: Option<Preimage>,
}

/// The input bits and value tested by a pulled-back equality.
struct Preimage {
    input: NodeRef,
    mask: IrBits,
    value: IrBits,
}

/// A node containing only part of a merged group may change value. Once all
/// its leaves meet, the conjunction (including any guards) is equivalent again.
fn refresh_partial_conjunction_ids(
    f: &mut ir::Fn,
    users: &Users,
    plans: &[Plan],
    next_id: &mut usize,
    mut origins: Option<Origins<'_>>,
) {
    let mut pending = HashMap::<NodeRef, HashMap<usize, usize>>::new();
    for (group, plan) in plans.iter().enumerate().filter(|(_, p)| p.atoms.len() > 1) {
        for &(comparison, _) in &plan.atoms {
            pending.entry(comparison).or_default().insert(group, 1);
        }
    }
    if pending.is_empty() {
        return;
    }
    for node in ir_utils::get_topological(f) {
        let Some(mut counts) = pending.remove(&node) else {
            continue;
        };
        f.nodes[node.index].text_id = *next_id;
        f.nodes[node.index].name = None;
        *next_id += 1;
        let [parent] = users.get(&node).unwrap() else {
            unreachable!("partial conjunctions have one user up to their common root")
        };
        // Groups can interleave in one Boolean tree. Move the smaller map so a
        // chain of gates does not repeatedly copy all pending groups.
        let merged = pending.entry(*parent).or_default();
        if counts.len() > merged.len() {
            std::mem::swap(merged, &mut counts);
        }
        for (group, count) in counts {
            let total = merged.entry(group).or_default();
            *total += count;
            if *total == plans[group].atoms.len() {
                merged.remove(&group);
                if let Some(origins) = &mut origins {
                    origins.record_merge(f, *parent, &plans[group].atoms);
                }
            }
        }
        if merged.is_empty() {
            pending.remove(parent);
        }
    }
}

/// A modular translation or reflection, retaining its constant terms until the
/// observations qualify for inversion.
struct Affine {
    input: NodeRef,
    negative: bool,
    constants: Vec<ExtNaryAddTerm>,
}

impl Affine {
    fn of(f: &ir::Fn, word: NodeRef) -> Option<Self> {
        let mut input = None;
        let mut constants = Vec::new();
        let mut term = |operand: NodeRef, signed: bool, negative: bool| -> Option<()> {
            if matches!(f.get_node(operand).payload, NodePayload::Literal(_)) {
                constants.push(ExtNaryAddTerm {
                    operand,
                    signed,
                    negated: negative,
                });
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
            constants,
        })
    }

    fn invert_cube(&self, f: &ir::Fn, mask: &IrBits, value: &IrBits) -> Option<IrBits> {
        let width = mask.get_bit_count();
        let mut offset = IrBits::zero(width);
        for term in &self.constants {
            let NodePayload::Literal(literal) = &f.get_node(term.operand).payload else {
                unreachable!("cached constant term");
            };
            let bits = literal.as_bits().ok()?;
            let mut extended = bits.width_slice(0, width as i64);
            if term.signed && bits.get_bit_count() < width && bits.get_bit_count() > 0 && bits.msb()
            {
                extended = extended.or(&IrBits::all_ones(width).shll(bits.get_bit_count() as i64));
            }
            offset = if term.negated {
                offset.sub(&extended)
            } else {
                offset.add(&extended)
            };
        }
        let inverse = |y: &IrBits| {
            if self.negative {
                offset.sub(y)
            } else {
                y.sub(&offset)
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

/// One contiguous part of a comparison literal wired to the affine word.
struct Field {
    start: usize,
    literal_start: usize,
    width: usize,
}

/// An equality or inequality observing a possibly rearranged slice of a word.
struct Observation<'a> {
    comparison: NodeRef,
    equal: bool,
    views: Vec<NodeRef>,
    fields: Vec<Field>,
    literal: &'a IrBits,
}

/// The combined bit constraints of a group with a common observation point.
struct ObservedCube {
    mask: IrBits,
    value: IrBits,
    conflict: bool,
}

impl ObservedCube {
    /// Write only the observed fields, checking overlaps within and across
    /// views.
    fn merge(width: usize, atoms: &[Observation]) -> Self {
        let mut mask = BitVec::<u8, Lsb0>::repeat(false, width);
        let mut value = BitVec::<u8, Lsb0>::repeat(false, width);
        let mut conflict = false;
        for atom in atoms {
            for field in &atom.fields {
                for offset in 0..field.width {
                    let index = field.start + offset;
                    let bit = atom.literal.get_bit(field.literal_start + offset).unwrap();
                    conflict |= mask[index] && value[index] != bit;
                    mask.set(index, true);
                    value.set(index, bit);
                }
            }
        }
        Self {
            mask: IrBits::from_le_bytes(width, mask.as_raw_slice()).unwrap(),
            value: IrBits::from_le_bytes(width, value.as_raw_slice()).unwrap(),
            conflict,
        }
    }
}

/// Discovery state shared by all comparisons on this unchanged graph.
struct ViewSearch {
    expanded: HashSet<NodeRef>,
    repeat_work: usize,
    sources: HashMap<NodeRef, Option<Affine>>,
}

impl ViewSearch {
    fn new(f: &ir::Fn) -> Self {
        Self {
            expanded: HashSet::new(),
            repeat_work: f.nodes.len(),
            sources: HashMap::new(),
        }
    }

    /// Each view can expand its own edges once. Bound duplicate paths across
    /// all observations while leaving independent views free to expand.
    fn expand(&mut self, node: NodeRef, edges: usize) -> Option<()> {
        if !self.expanded.insert(node) {
            charge(&mut self.repeat_work, edges.max(1))?;
        }
        Some(())
    }

    /// Records the wiring to the comparison literal without widening any bits.
    fn observation<'a>(
        &mut self,
        f: &'a ir::Fn,
        comparison: NodeRef,
    ) -> Option<(NodeRef, Observation<'a>)> {
        let NodePayload::Binop(op @ (Binop::Eq | Binop::Ne), a, b) = f.get_node(comparison).payload
        else {
            return None;
        };
        let (view, literal) = match (&f.get_node(a).payload, &f.get_node(b).payload) {
            (NodePayload::Literal(v), _) => (b, v.as_bits().ok()?),
            (_, NodePayload::Literal(v)) => (a, v.as_bits().ok()?),
            _ => return None,
        };
        let mut pending = vec![(view, 0usize, literal.get_bit_count(), 0usize)];
        let mut source = None;
        let mut fields = Vec::new();
        let mut views = Vec::new();
        while let Some((node, start, length, target)) = pending.pop() {
            if length == 0 {
                continue;
            }
            match &f.get_node(node).payload {
                NodePayload::BitSlice {
                    arg, start: low, ..
                } => {
                    self.expand(node, 1)?;
                    views.push(node);
                    pending.push((*arg, start + low, length, target));
                }
                NodePayload::Nary(NaryOp::Concat, args) => {
                    self.expand(node, args.len())?;
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
                    if source.replace(node).is_some_and(|other| other != node) {
                        return None;
                    }
                    self.sources
                        .entry(node)
                        .or_insert_with(|| Affine::of(f, node))
                        .as_ref()?;
                    fields.push(Field {
                        start,
                        literal_start: target,
                        width: length,
                    });
                }
            }
        }
        Some((
            source?,
            Observation {
                comparison,
                equal: op == Binop::Eq,
                views,
                fields,
                literal,
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

/// Reserves work before expanding a path or constructing a support mask.
fn charge(work: &mut usize, amount: usize) -> Option<()> {
    let ok = *work >= amount;
    *work = work.saturating_sub(amount);
    ok.then_some(())
}

/// Each mask visit costs at least one edge, plus work for additional limbs.
fn mask_work(width: usize) -> usize {
    width.div_ceil(u64::BITS as usize).max(1)
}

/// A partial sum can depend on every input bit below its highest observed bit.
fn shares_input_demand(
    f: &ir::Fn,
    users: &Users,
    word: NodeRef,
    input: NodeRef,
    observed: &IrBits,
    work: &mut usize,
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
    if charge(work, mask_work(input_width.max(observed.get_bit_count()))).is_none() {
        return true;
    }
    let Some(highest) = (0..observed.get_bit_count()).rfind(|&i| observed.get_bit(i).unwrap())
    else {
        return true;
    };
    let low = IrBits::all_ones((highest + 1).min(input_width)).width_slice(0, input_width as i64);
    let mut source = HashMap::new();
    if visit_input_support(f, input, low, word, work, |param, bits| {
        source.insert(param, bits.clone());
        true
    })
    .is_none()
    {
        return true;
    }
    if source.is_empty() {
        return false;
    }
    let output_width = f.get_node(root).ty.bit_count();
    if charge(work, mask_work(output_width)).is_none() {
        return true;
    }
    visit_input_support(
        f,
        root,
        IrBits::all_ones(output_width),
        word,
        work,
        |param, bits| {
            !source
                .get(&param)
                .is_some_and(|other| !bits.and(other).is_zero())
        },
    )
    .is_none()
}

/// Wiring preserves exact bit demand. Other operators conservatively demand
/// their full operands. Stop on overlap or when the shared search is exhausted.
fn visit_input_support(
    f: &ir::Fn,
    root: NodeRef,
    mask: IrBits,
    cut: NodeRef,
    work: &mut usize,
    mut visit: impl FnMut(NodeRef, &IrBits) -> bool,
) -> Option<()> {
    let mut pending = vec![(root, mask)];
    let mut visited = HashMap::<NodeRef, IrBits>::new();
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
        let payload = &f.get_node(node).payload;
        if let NodePayload::TupleIndex { tuple, .. } = payload {
            let Type::Tuple(fields) = f.get_node_ty(*tuple) else {
                unreachable!("validated tuple index")
            };
            // Computing the field offset can scan the tuple.
            charge(work, fields.len())?;
        }
        if let Some((arg, start)) = wiring::slice_source(f, payload) {
            let width = f.get_node_ty(arg).bit_count();
            charge(work, mask_work(width))?;
            if mask.get_bit_count() != width {
                mask = mask.width_slice(0, width as i64);
            }
            if start != 0 {
                mask = mask.shll(start as i64);
            }
            pending.push((arg, mask));
            continue;
        }
        if let Some(parts) = wiring::concat_sources(f, payload) {
            for (arg, range) in parts {
                charge(work, mask_work(range.len()))?;
                pending.push((
                    arg,
                    mask.width_slice(range.start as i64, range.len() as i64),
                ));
            }
            continue;
        }
        match payload {
            NodePayload::Literal(_) | NodePayload::Nil => {
                // Neither depends on an input.
            }
            NodePayload::Param | NodePayload::InputPort { .. } => {
                if !visit(node, &visited[&node]) {
                    return None;
                }
            }
            _ => {
                for arg in ir_utils::operands(payload) {
                    let width = f.get_node_ty(arg).bit_count();
                    charge(work, mask_work(width))?;
                    pending.push((arg, IrBits::all_ones(width)));
                }
            }
        }
    }
    Some(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;
    use xlsynth_pir::{BuilderError, FnBuilder};

    #[test_case(false; "flat flag bank")]
    #[test_case(true; "shared input returned alongside flag bank")]
    fn shared_input_search_is_bounded_and_private_inputs_still_rewrite(
        return_shared: bool,
    ) -> Result<(), BuilderError> {
        let mut b = FnBuilder::new("bank");
        let shared = b.param("shared", Type::Bits(32))?;
        let mut inputs = vec![shared; 128];
        let private_count = 128;
        for i in 0..private_count {
            inputs.push(b.param(&format!("x_{i}"), Type::Bits(32))?);
        }
        let zero = b.literal(IrValue::make_ubits(16, 0).unwrap())?;
        let mut flags = Vec::new();
        for (i, input) in inputs.into_iter().enumerate() {
            let offset = b.literal(IrValue::u32(i as u32 + 1))?;
            let word = b.add(input, offset)?;
            let lo = b.bit_slice(word, 0, 16)?;
            flags.push(b.eq(lo, zero)?);
        }
        let bank = b.concat(&flags)?;
        let result = if return_shared {
            b.concat(&[shared, bank])?
        } else {
            bank
        };
        let mut f = b.build(result)?;
        let users = ir_utils::compute_users(&f);
        let mut work = f.nodes.len();
        let mask = IrBits::make_ubits(32, 0xffff).unwrap();
        for (index, n) in f.nodes.iter().enumerate() {
            if let NodePayload::Binop(Binop::Add, input, _) = n.payload {
                assert_eq!(
                    shares_input_demand(&f, &users, NodeRef { index }, input, &mask, &mut work),
                    input == f.params[0]
                );
            }
        }
        assert_eq!(work, 0);

        // Each successful single-comparison plan keeps its ID and appends a
        // slice and a literal above every original ID, including gaps.
        let original_len = f.nodes.len();
        let max_id = original_len * 2;
        let root = f.ret_node_ref.unwrap();
        f.nodes[root.index].text_id = max_id;
        assert_eq!(rewrite(&mut f, None), private_count);
        let helper_ids = f.nodes[original_len..].iter().map(|n| n.text_id);
        assert!(helper_ids.eq(max_id + 1..=max_id + 2 * private_count));
        Ok(())
    }

    #[test]
    fn shared_slice_prefixes_stop_at_an_ineligible_source_or_the_repeat_budget()
    -> Result<(), BuilderError> {
        for affine in [false, true] {
            let mut b = FnBuilder::new("shared_views");
            let x = b.param("x", Type::Bits(8))?;
            let rhs = if affine {
                b.literal(IrValue::make_ubits(8, 1).unwrap())?
            } else {
                b.param("y", Type::Bits(8))?
            };
            let word = b.add(x, rhs)?;
            let mut bit = b.bit_slice(word, 0, 1)?;
            let mut parts = Vec::new();
            for _ in 0..128 {
                bit = b.bit_slice(bit, 0, 1)?;
                parts.push(bit);
            }
            let view = b.concat(&parts)?;
            let target = b.literal(IrValue::make_ubits(parts.len(), 0).unwrap())?;
            let cmp = b.eq(view, target)?;
            let f = b.build(cmp)?;
            let mut search = ViewSearch::new(&f);
            assert!(search.observation(&f, f.ret_node_ref.unwrap()).is_none());
            assert_eq!(search.repeat_work, if affine { 0 } else { f.nodes.len() });
        }
        Ok(())
    }

    #[test]
    fn independent_bit_observations_of_a_wide_word_are_rejected() -> Result<(), BuilderError> {
        let width = 65_536;
        let mut b = FnBuilder::new("independent");
        let x = b.param("x", Type::Bits(width))?;
        let one = b.literal(IrValue::make_ubits(width, 1).unwrap())?;
        let word = b.add(x, one)?;
        let on = b.literal(IrValue::bool(true))?;
        let mut flags = Vec::new();
        for bit in 0..width {
            let slice = b.bit_slice(word, bit, 1)?;
            flags.push(b.eq(slice, on)?);
        }
        let result = b.concat(&flags)?;
        let mut f = b.build(result)?;
        assert_eq!(rewrite(&mut f, None), 0);
        Ok(())
    }
}
