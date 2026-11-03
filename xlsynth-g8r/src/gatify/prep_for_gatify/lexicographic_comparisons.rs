// SPDX-License-Identifier: Apache-2.0

//! Coalesces unsigned lexicographic predicates using carry-only comparisons.

use std::collections::{HashMap, HashSet};
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_match::{self, MatchCtx};
use xlsynth_pir::ir_utils;

/// Rewrites maximal comparison regions without duplicating shared predicates.
pub(super) fn rewrite(
    f: &mut ir::Fn,
    mut helper_origins: Option<&mut HashMap<usize, usize>>,
) -> usize {
    let order = ir_utils::get_topological(f);
    let users = ir_utils::compute_users(f);
    let graph_work = f.nodes.len().saturating_add(
        f.nodes
            .iter()
            .map(|n| ir_utils::operands(&n.payload).len())
            .sum::<usize>(),
    );
    let uses = output_uses(f, &order);
    let mut analysis = Analysis::new(f, graph_work);
    let mut predicates = PredicateIndex::new(f.nodes.len());
    for &node in &order {
        if uses[node.index] != OutputUse::Unused {
            predicates.record(f, node, uses[node.index]);
            analysis.analyze(node);
        }
    }
    let mut covered = HashSet::new();
    let mut replacements = Vec::new();
    // Count the input graph and the persistent word plans traversed below.
    let mut validation_work = graph_work.saturating_add(analysis.words.len());
    let mut shared = 0;
    'candidates: for &root in order.iter().rev() {
        if covered.contains(&root)
            || !matches!(f.get_node(root).payload, NodePayload::Nary(NaryOp::Or, _))
        {
            continue;
        }
        let Some(plan) = analysis.comparisons[root.index] else {
            continue;
        };
        let Some(parts) = analysis.word_parts(plan.word, &mut validation_work) else {
            break;
        };
        let frontier: HashSet<_> = parts.iter().flat_map(|part| [part.lhs, part.rhs]).collect();
        // Concatenation hides literal thresholds from the native comparator.
        // A single-part rewrite still emits a native comparison.
        if parts.len() > 1
            && frontier
                .iter()
                .any(|&operand| matches!(f.get_node(operand).payload, NodePayload::Literal(_)))
        {
            continue;
        }
        let Some(region_nodes) = analysis.region_nodes(root, &mut validation_work) else {
            break;
        };
        let region: HashSet<_> = region_nodes.iter().copied().collect();
        let mut region_predicates = HashMap::new();
        let mut has_external_user = false;
        for node in region_nodes {
            if node == root {
                continue;
            }
            if let Some(key) = predicates.key(&f.get_node(node).payload) {
                *region_predicates.entry(key).or_insert(0usize) += 1;
            }
            // A predicate consumed both as a guard and as a limb will still
            // be needed to construct the replacement word.
            if frontier.contains(&node) || f.ret_node_ref == Some(node) {
                has_external_user = true;
                break;
            }
            for user in users.get(&node).unwrap_or(&[]) {
                let Some(remaining) = validation_work.checked_sub(1) else {
                    break 'candidates;
                };
                validation_work = remaining;
                if !region.contains(user) {
                    has_external_user = true;
                    break;
                }
            }
            if has_external_user {
                break;
            }
        }
        // Other live PIR nodes can request the same structurally hashed
        // predicate even when every user of this particular node is local.
        if has_external_user
            || region_predicates
                .iter()
                .any(|(key, count)| predicates.live[key].count > *count)
        {
            shared += 1;
            continue;
        }
        covered.extend(region);
        let whole = predicates.whole_word(&parts, plan.strict);
        if whole.is_some_and(|p| p.returned.is_none()) {
            // An indirect consumer may fold away. Preserve the native region,
            // including its subchains, so any existing gates can still share.
            shared += 1;
            continue;
        }
        replacements.push(Replacement {
            root,
            strict: plan.strict,
            parts,
            whole: whole.and_then(|p| p.returned),
        });
    }
    log::debug!(target: "xlsynth_g8r::gatify::prep_for_gatify", "lexicographic_comparisons accepted={} shared={} matching_work_left={} validation_work_left={}", replacements.len(), shared, analysis.remaining_work, validation_work);
    drop(analysis);
    let Some(mut next_id) = f
        .nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
    else {
        return 0;
    };
    let mut count = 0;
    for Replacement {
        root,
        strict,
        parts,
        whole,
    } in replacements
    {
        let Some(width) = parts
            .iter()
            .try_fold(0usize, |width, part| width.checked_add(part.width))
        else {
            continue;
        };
        let replacement = if let [part] = parts.as_slice() {
            NodePayload::Binop(
                if strict { Binop::Ugt } else { Binop::Uge },
                part.lhs,
                part.rhs,
            )
        } else if let Some(node) = whole {
            let existing_strict = matches!(
                f.get_node(node).payload,
                NodePayload::Binop(Binop::Ugt | Binop::Ult, _, _)
            );
            // A one-input OR keeps the matched OR's provenance on the shared
            // gate. NOR does the same when the returned predicate is its dual.
            let op = if strict == existing_strict {
                NaryOp::Or
            } else {
                NaryOp::Nor
            };
            NodePayload::Nary(op, vec![node])
        } else {
            let Some(after_ids) = next_id.checked_add(5) else {
                break;
            };
            let origin = f.get_node(root).text_id;
            let mut append = |ty, payload| {
                let node = NodeRef {
                    index: f.nodes.len(),
                };
                f.nodes.push(ir::Node {
                    text_id: next_id,
                    name: None,
                    ty,
                    payload,
                    pos: None,
                });
                // The helpers implement the comparison, but their values
                // (in particular the complemented carry) need distinct IDs.
                if let Some(origins) = helper_origins.as_deref_mut() {
                    origins.insert(next_id, origin);
                }
                next_id += 1;
                node
            };
            let left = append(
                Type::Bits(width),
                NodePayload::Nary(NaryOp::Concat, parts.iter().map(|part| part.lhs).collect()),
            );
            let right = append(
                Type::Bits(width),
                NodePayload::Nary(NaryOp::Concat, parts.iter().map(|part| part.rhs).collect()),
            );
            // A > B is the complement of the carry from B + ~A + 1.
            // A >= B uses the same circuit with carry-in zero.
            let complement = append(Type::Bits(width), NodePayload::Unop(ir::Unop::Not, left));
            let literal = xlsynth_pir::IrValue::make_ubits(1, u64::from(strict))
                .expect("carry input fits one bit");
            let carry_in = append(Type::Bits(1), NodePayload::Literal(literal));
            let carry = append(
                Type::Bits(1),
                NodePayload::ExtCarryOut {
                    lhs: right,
                    rhs: complement,
                    c_in: carry_in,
                },
            );
            debug_assert_eq!(next_id, after_ids);
            NodePayload::Unop(ir::Unop::Not, carry)
        };
        ir_utils::replace_node_payload(f, root, replacement, None)
            .expect("lexicographic replacement preserves the Boolean result type");
        count += 1;
    }
    count
}

/// A validated chain and the returned whole-word predicate it can share, if
/// any.
struct Replacement {
    root: NodeRef,
    strict: bool,
    parts: Vec<Part>,
    whole: Option<NodeRef>,
}

/// Whether a node's full value is wired to the output or only used to compute
/// it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OutputUse {
    Unused,
    Dependency,
    Returned,
}

/// Propagates output uses once, distinguishing tuple construction from
/// projection.
fn output_uses(f: &ir::Fn, order: &[NodeRef]) -> Vec<OutputUse> {
    let mut uses = vec![OutputUse::Unused; f.nodes.len()];
    if let Some(ret) = f.ret_node_ref {
        uses[ret.index] = OutputUse::Returned;
    }
    for &node in order.iter().rev() {
        let usage = uses[node.index];
        if usage == OutputUse::Unused {
            continue;
        }
        let payload = &f.get_node(node).payload;
        if let NodePayload::TupleIndex { tuple, index } = payload
            && let NodePayload::Tuple(fields) = &f.get_node(*tuple).payload
        {
            let field = fields[*index];
            uses[field.index] = uses[field.index].max(usage);
            continue;
        }
        let operand_use = match payload {
            NodePayload::Tuple(_)
            | NodePayload::Array(_)
            | NodePayload::Nary(NaryOp::Concat, _)
            | NodePayload::Unop(ir::Unop::Identity | ir::Unop::Not, _) => usage,
            _ => OutputUse::Dependency,
        };
        for operand in ir_utils::operands(payload) {
            uses[operand.index] = uses[operand.index].max(operand_use);
        }
    }
    uses
}

/// Native predicates that share a gate cone, including complemented outputs.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Predicate {
    GreaterThan(NodeRef, NodeRef),
    Equal(NodeRef, NodeRef),
}

/// The number of live native predicates sharing a gate and an exposed result.
#[derive(Default)]
struct LivePredicate {
    returned: Option<NodeRef>,
    count: usize,
}

/// Canonical wire operands and their native predicates in the input graph.
struct PredicateIndex {
    operands: Vec<NodeRef>,
    concats: HashMap<Vec<NodeRef>, NodeRef>,
    live: HashMap<Predicate, LivePredicate>,
}

impl PredicateIndex {
    fn new(node_count: usize) -> Self {
        Self {
            operands: (0..node_count).map(|index| NodeRef { index }).collect(),
            concats: HashMap::new(),
            live: HashMap::new(),
        }
    }

    /// Visits live nodes in topological order so every identity chain and
    /// repeated concatenation has a representative before its uses.
    fn record(&mut self, f: &ir::Fn, node: NodeRef, usage: OutputUse) {
        let payload = &f.get_node(node).payload;
        match payload {
            NodePayload::Unop(ir::Unop::Identity, arg) => {
                self.operands[node.index] = self.operands[arg.index];
            }
            NodePayload::Nary(NaryOp::Concat, args) => {
                let parts = args.iter().map(|arg| self.operands[arg.index]).collect();
                self.operands[node.index] = *self.concats.entry(parts).or_insert(node);
            }
            _ => {
                if let Some(key) = self.key(payload) {
                    let entry = self.live.entry(key).or_default();
                    entry.count += 1;
                    if usage == OutputUse::Returned {
                        entry.returned.get_or_insert(node);
                    }
                }
            }
        }
    }

    fn key(&self, payload: &NodePayload) -> Option<Predicate> {
        let NodePayload::Binop(op, lhs, rhs) = *payload else {
            return None;
        };
        let (lhs, rhs) = (self.operands[lhs.index], self.operands[rhs.index]);
        match op {
            // A <= B lowers to !(A > B). Swapping the operands gives the
            // corresponding < and >= forms of the same gate.
            Binop::Ugt | Binop::Ule => Some(Predicate::GreaterThan(lhs, rhs)),
            Binop::Ult | Binop::Uge => Some(Predicate::GreaterThan(rhs, lhs)),
            Binop::Eq | Binop::Ne => {
                let (a, b) = unordered_pair(lhs, rhs);
                Some(Predicate::Equal(a, b))
            }
            _ => None,
        }
    }

    /// Looks up native comparisons of the concatenated limbs, including their
    /// duals.
    fn whole_word(&self, parts: &[Part], strict: bool) -> Option<&LivePredicate> {
        if parts.len() < 2 || self.concats.is_empty() {
            return None;
        }
        let (left, right): (Vec<_>, Vec<_>) = parts
            .iter()
            .map(|p| (self.operands[p.lhs.index], self.operands[p.rhs.index]))
            .unzip();
        let lhs = *self.concats.get(&left)?;
        let rhs = *self.concats.get(&right)?;
        let key = if strict {
            Predicate::GreaterThan(lhs, rhs)
        } else {
            Predicate::GreaterThan(rhs, lhs)
        };
        self.live.get(&key)
    }
}

fn unordered_pair(lhs: NodeRef, rhs: NodeRef) -> (NodeRef, NodeRef) {
    if lhs.index <= rhs.index {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    }
}

/// One limb oriented so that the reconstructed relation is greater-than.
#[derive(Clone, Copy)]
struct Part {
    lhs: NodeRef,
    rhs: NodeRef,
    width: usize,
}

/// Persistent concatenation plans avoid copying every prefix of a long chain.
enum Word {
    Part(Part),
    Concat(usize, usize),
}

/// Equality conjunctions retain their DAG instead of copying prefix sets.
enum Equalities {
    Pair(NodeRef, NodeRef),
    All(Vec<usize>),
}

/// A word comparison with either a strict or inclusive final limb.
#[derive(Clone, Copy)]
struct Comparison {
    word: usize,
    strict: bool,
}

/// A conjunction of equalities, optionally guarding a lower word comparison.
#[derive(Clone, Copy)]
struct Guard {
    equalities: usize,
    lower: Option<Comparison>,
}

/// One immutable graph analysis with bounded local work and persistent
/// summaries.
struct Analysis<'a> {
    ctx: MatchCtx<'a>,
    comparisons: Vec<Option<Comparison>>,
    guards: Vec<Option<Guard>>,
    words: Vec<Word>,
    equalities: Vec<Equalities>,
    remaining_work: usize,
}

impl<'a> Analysis<'a> {
    fn new(f: &'a ir::Fn, remaining_work: usize) -> Self {
        Self {
            ctx: MatchCtx::new(f),
            comparisons: vec![None; f.nodes.len()],
            guards: vec![None; f.nodes.len()],
            words: Vec::new(),
            equalities: Vec::new(),
            remaining_work,
        }
    }

    fn analyze(&mut self, node: NodeRef) {
        if self.ctx.bits_width(node) != Some(1) {
            return;
        }
        match &self.ctx.f.get_node(node).payload {
            NodePayload::Binop(
                op @ (Binop::Ugt | Binop::Uge | Binop::Ult | Binop::Ule),
                lhs,
                rhs,
            ) => {
                let Some(width) = self.ctx.bits_width(*lhs) else {
                    return;
                };
                if width == 0 || self.ctx.bits_width(*rhs) != Some(width) {
                    return;
                }
                let (lhs, rhs) = if matches!(op, Binop::Ult | Binop::Ule) {
                    (*rhs, *lhs)
                } else {
                    (*lhs, *rhs)
                };
                let word = self.words.len();
                self.words.push(Word::Part(Part { lhs, rhs, width }));
                self.comparisons[node.index] = Some(Comparison {
                    word,
                    strict: matches!(op, Binop::Ugt | Binop::Ult),
                });
            }
            NodePayload::Binop(Binop::Eq, _, _) => {
                let Some(matched) = self.ctx.commutative_binop_pair(
                    node,
                    Binop::Eq,
                    ir_match::any("lhs"),
                    ir_match::any("rhs"),
                ) else {
                    return;
                };
                let (Some(lhs), Some(rhs)) = (matched.get_node("lhs"), matched.get_node("rhs"))
                else {
                    return;
                };
                if !self
                    .ctx
                    .bits_width(lhs)
                    .is_some_and(|w| w > 0 && self.ctx.bits_width(rhs) == Some(w))
                {
                    return;
                }
                let equalities = self.equalities.len();
                self.equalities.push(Equalities::Pair(lhs, rhs));
                self.guards[node.index] = Some(Guard {
                    equalities,
                    lower: None,
                });
            }
            NodePayload::Nary(NaryOp::And, args) => {
                let mut prefixes = Vec::new();
                let mut lower = None;
                for &arg in args {
                    if let Some(guard) = self.guards[arg.index] {
                        prefixes.push(guard.equalities);
                        if let Some(value) = guard.lower {
                            if lower.replace(value).is_some() {
                                return;
                            }
                        }
                    } else if let Some(value) = self.comparisons[arg.index] {
                        if lower.replace(value).is_some() {
                            return;
                        }
                    } else {
                        return;
                    }
                }
                let equalities = self.equalities.len();
                self.equalities.push(Equalities::All(prefixes));
                self.guards[node.index] = Some(Guard { equalities, lower });
            }
            NodePayload::Nary(NaryOp::Or, _) => {
                self.comparisons[node.index] = self.analyze_or(node);
            }
            _ => {
                // Other Boolean operations cannot start or extend a chain.
            }
        }
    }

    /// Reconstructs the word from its unguarded comparison and successively
    /// longer equality guards, independent of the order of the OR operands.
    fn analyze_or(&mut self, root: NodeRef) -> Option<Comparison> {
        let mut pending = vec![root];
        let mut high = None;
        let mut guards = Vec::new();
        self.remaining_work = self.remaining_work.checked_sub(1)?;
        while let Some(node) = pending.pop() {
            if let Some(plan) = self.comparisons[node.index] {
                // A merge consumes one guard and never a second unguarded word.
                if high.replace(plan).is_some() {
                    return None;
                }
            } else if let Some(guard) = self.guards[node.index] {
                guards.push(guard);
            } else if let NodePayload::Nary(NaryOp::Or, args) = &self.ctx.f.get_node(node).payload {
                // Prepay both the variable-fanout copy and its later visits.
                self.remaining_work = self.remaining_work.checked_sub(args.len())?;
                pending.extend(args);
            } else {
                return None;
            }
        }
        let mut high = high?;
        let mut by_length = HashMap::new();
        for guard in guards {
            let pairs = self.equality_pairs(guard.equalities)?;
            by_length
                .entry(pairs.len())
                .or_insert_with(Vec::new)
                .push((pairs, guard));
        }

        let mut expected = HashSet::new();
        let mut added_word = high.word;
        while !by_length.is_empty() {
            if !high.strict {
                return None;
            }
            self.add_word_pairs(added_word, &mut expected)?;
            let tails = by_length.remove(&expected.len())?;
            // The prefix set only grows. Every guard of this size must match
            // it now, even if some comparisons repeat a previously seen limb.
            let tied = tails.len() > 1;
            let mut next = None;
            for (pairs, guard) in tails {
                if pairs != expected {
                    return None;
                }
                if tied
                    && let Some(lower) = guard.lower
                    && lower.strict
                {
                    let mut lower_pairs = HashSet::new();
                    self.add_word_pairs(lower.word, &mut lower_pairs)?;
                    if lower_pairs.is_subset(&expected) {
                        // Under this guard the lower operands are equal, so
                        // their strict comparison contributes false to the OR.
                        continue;
                    }
                }
                if next.replace(guard).is_some() {
                    return None;
                }
            }
            let Some(guard) = next else {
                return by_length.is_empty().then_some(high);
            };
            let Some(lower) = guard.lower else {
                high.strict = false;
                return by_length.is_empty().then_some(high);
            };
            let word = self.words.len();
            self.words.push(Word::Concat(high.word, lower.word));
            high = Comparison {
                word,
                strict: lower.strict,
            };
            added_word = lower.word;
        }
        Some(high)
    }

    /// Extends an equality prefix by visiting only the newly appended word.
    fn add_word_pairs(
        &mut self,
        word: usize,
        pairs: &mut HashSet<(NodeRef, NodeRef)>,
    ) -> Option<()> {
        let mut pending = vec![word];
        while let Some(id) = pending.pop() {
            self.remaining_work = self.remaining_work.checked_sub(1)?;
            match self.words[id] {
                Word::Part(part) => {
                    pairs.insert(unordered_pair(part.lhs, part.rhs));
                }
                Word::Concat(high, low) => {
                    pending.push(low);
                    pending.push(high);
                }
            }
        }
        Some(())
    }

    /// Collects the distinct operand pairs required by one guard.
    fn equality_pairs(&mut self, equalities: usize) -> Option<HashSet<(NodeRef, NodeRef)>> {
        let mut actual = HashSet::new();
        let mut visited = HashSet::new();
        let mut pending = vec![equalities];
        self.remaining_work = self.remaining_work.checked_sub(1)?;
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            match &self.equalities[id] {
                Equalities::Pair(lhs, rhs) => {
                    actual.insert(unordered_pair(*lhs, *rhs));
                }
                Equalities::All(children) => {
                    self.remaining_work = self.remaining_work.checked_sub(children.len())?;
                    pending.extend(children.iter().copied());
                }
            }
        }
        Some(actual)
    }

    /// Follows the matched AND/OR edges up to and including predicate leaves.
    /// A leaf's operands are word data, even if another path tests one of
    /// them as a predicate in its own right.
    fn region_nodes(&self, root: NodeRef, work: &mut usize) -> Option<Vec<NodeRef>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut pending = vec![root];
        *work = work.checked_sub(1)?;
        while let Some(node) = pending.pop() {
            if !visited.insert(node) {
                continue;
            }
            result.push(node);
            if let NodePayload::Nary(NaryOp::And | NaryOp::Or, args) =
                &self.ctx.f.get_node(node).payload
            {
                *work = work.checked_sub(args.len())?;
                pending.extend(args);
            }
        }
        Some(result)
    }

    fn word_parts(&self, word: usize, work: &mut usize) -> Option<Vec<Part>> {
        let mut result = Vec::new();
        let mut pending = vec![word];
        while let Some(id) = pending.pop() {
            *work = work.checked_sub(1)?;
            match self.words[id] {
                Word::Part(part) => result.push(part),
                Word::Concat(high, low) => {
                    pending.push(low);
                    pending.push(high);
                }
            }
        }
        Some(result)
    }
}
