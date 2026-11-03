// SPDX-License-Identifier: Apache-2.0

//! Coalesces unsigned lexicographic predicates using carry-only comparisons.

use std::collections::HashSet;
use xlsynth_pir::ir::{self, Binop, NaryOp, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_match::{self, MatchCtx};
use xlsynth_pir::ir_utils;

/// Rewrites maximal comparison regions without duplicating shared predicates.
pub(super) fn rewrite(f: &mut ir::Fn) -> usize {
    let order = ir_utils::get_topological(f);
    let users = ir_utils::compute_users(f);
    let graph_work = f.nodes.len().saturating_add(
        f.nodes
            .iter()
            .map(|n| ir_utils::operands(&n.payload).len())
            .sum::<usize>(),
    );
    let mut live = vec![false; f.nodes.len()];
    let mut pending: Vec<_> = f.ret_node_ref.into_iter().collect();
    while let Some(node) = pending.pop() {
        if !live[node.index] {
            live[node.index] = true;
            pending.extend(ir_utils::operands(&f.get_node(node).payload));
        }
    }
    let mut analysis = Analysis::new(f, graph_work);
    for &node in &order {
        if live[node.index] {
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
        let Some(region_nodes) = analysis.region_nodes(root, &frontier, &mut validation_work)
        else {
            break;
        };
        let region: HashSet<_> = region_nodes.iter().copied().collect();
        let mut has_external_user = false;
        for node in region_nodes {
            if node == root {
                continue;
            }
            if f.ret_node_ref == Some(node) {
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
        if has_external_user {
            shared += 1;
            continue;
        }
        covered.extend(region);
        replacements.push((root, plan.strict, parts));
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
    for (root, strict, parts) in replacements {
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
        } else {
            let Some(after_ids) = next_id.checked_add(5) else {
                break;
            };
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

#[derive(Clone, Copy)]
struct Comparison {
    word: usize,
    strict: bool,
}

#[derive(Clone, Copy)]
struct Guard {
    equalities: usize,
    lower: Option<Comparison>,
}

#[derive(Clone, Copy)]
enum Term {
    Comparison(Comparison),
    Guard(Guard),
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
            _ => {}
        }
    }

    fn analyze_or(&mut self, root: NodeRef) -> Option<Comparison> {
        let mut pending = vec![root];
        let mut terms = Vec::new();
        self.remaining_work = self.remaining_work.checked_sub(1)?;
        while let Some(node) = pending.pop() {
            if let Some(plan) = self.comparisons[node.index] {
                terms.push(Term::Comparison(plan));
            } else if let Some(guard) = self.guards[node.index] {
                terms.push(Term::Guard(guard));
            } else if let NodePayload::Nary(NaryOp::Or, args) = &self.ctx.f.get_node(node).payload {
                // Prepay both the variable-fanout copy and its later visits.
                self.remaining_work = self.remaining_work.checked_sub(args.len())?;
                pending.extend(args.iter().rev().copied());
            } else {
                return None;
            }
        }
        while terms.len() > 1 {
            let mut found = None;
            'pairs: for (hi, term) in terms.iter().enumerate() {
                self.remaining_work = self.remaining_work.checked_sub(1)?;
                let Term::Comparison(high) = *term else {
                    continue;
                };
                if !high.strict {
                    continue;
                }
                for (lo, term) in terms.iter().enumerate() {
                    self.remaining_work = self.remaining_work.checked_sub(1)?;
                    let Term::Guard(tail) = *term else { continue };
                    if hi == lo || !self.prefix_matches(high.word, tail.equalities)? {
                        continue;
                    }
                    let (word, strict) = if let Some(lower) = tail.lower {
                        let word = self.words.len();
                        self.words.push(Word::Concat(high.word, lower.word));
                        (word, lower.strict)
                    } else {
                        (high.word, false)
                    };
                    found = Some((hi, lo, Comparison { word, strict }));
                    break 'pairs;
                }
            }
            let (hi, lo, merged) = found?;
            // OR operands are unordered; removing them must not shift a long
            // guard prefix on every merge.
            terms.swap_remove(hi.max(lo));
            terms.swap_remove(hi.min(lo));
            terms.push(Term::Comparison(merged));
        }
        let [Term::Comparison(result)] = terms.as_slice() else {
            return None;
        };
        Some(*result)
    }

    fn prefix_matches(&mut self, word: usize, equalities: usize) -> Option<bool> {
        let mut expected = HashSet::new();
        let mut pending = vec![word];
        while let Some(id) = pending.pop() {
            self.remaining_work = self.remaining_work.checked_sub(1)?;
            match self.words[id] {
                Word::Part(part) => {
                    expected.insert(if part.lhs.index < part.rhs.index {
                        (part.lhs, part.rhs)
                    } else {
                        (part.rhs, part.lhs)
                    });
                }
                Word::Concat(high, low) => {
                    pending.push(low);
                    pending.push(high);
                }
            }
        }
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
                    actual.insert(if lhs.index < rhs.index {
                        (*lhs, *rhs)
                    } else {
                        (*rhs, *lhs)
                    });
                }
                Equalities::All(children) => {
                    self.remaining_work = self.remaining_work.checked_sub(children.len())?;
                    pending.extend(children.iter().copied());
                }
            }
        }
        Some(actual == expected)
    }

    /// Traverses only the matched Boolean cone, stopping at its word operands.
    /// The existing PIR graph is the region representation; no copied region
    /// DAG or whole-function traversal is needed for an individual candidate.
    fn region_nodes(
        &self,
        root: NodeRef,
        frontier: &HashSet<NodeRef>,
        work: &mut usize,
    ) -> Option<Vec<NodeRef>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut pending = vec![root];
        *work = work.checked_sub(1)?;
        while let Some(node) = pending.pop() {
            if frontier.contains(&node) || !visited.insert(node) {
                continue;
            }
            result.push(node);
            let operands = ir_utils::operands(&self.ctx.f.get_node(node).payload);
            *work = work.checked_sub(operands.len())?;
            pending.extend(operands);
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
