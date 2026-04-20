// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use super::macro_utils as mu;
use super::*;

const MAX_ARM_CONE_NODES: usize = 64;
const MAX_CANDIDATES: usize = 2000;

/// Rewires one-bit operands inside guarded predicate cones.
#[derive(Debug)]
pub struct GuardedPredicateRewireTransform;

impl GuardedPredicateRewireTransform {
    fn operand_pairs(payload: &NodePayload) -> Vec<(usize, NodeRef)> {
        let mut pairs = Vec::new();
        let _ = remap_payload_with(payload, |(slot, dep)| {
            pairs.push((slot, dep));
            dep
        });
        pairs
    }

    fn compute_fanout_cone(
        users_map: &std::collections::HashMap<NodeRef, HashSet<NodeRef>>,
        root: NodeRef,
    ) -> HashSet<NodeRef> {
        let mut visited = HashSet::new();
        let mut work = VecDeque::new();
        visited.insert(root);
        work.push_back(root);
        while let Some(cur) = work.pop_front() {
            if let Some(users) = users_map.get(&cur) {
                for user in users {
                    if visited.insert(*user) {
                        work.push_back(*user);
                    }
                }
            }
        }
        visited
    }

    fn dependency_cone(f: &IrFn, root: NodeRef) -> Option<Vec<NodeRef>> {
        let mut seen = HashSet::new();
        let mut stack = vec![root];
        while let Some(cur) = stack.pop() {
            if !seen.insert(cur) {
                continue;
            }
            if seen.len() > MAX_ARM_CONE_NODES {
                return None;
            }
            for (_, dep) in Self::operand_pairs(&f.get_node(cur).payload) {
                stack.push(dep);
            }
        }
        let mut out = seen.into_iter().collect::<Vec<_>>();
        out.sort_by_key(|r| r.index);
        Some(out)
    }

    fn existing_not_of(f: &IrFn, arg: NodeRef) -> Vec<NodeRef> {
        let mut out = Vec::new();
        for nr in f.node_refs() {
            if matches!(f.get_node(nr).payload, NodePayload::Unop(Unop::Not, a) if a == arg)
                && mu::is_u1(f, nr)
            {
                out.push(nr);
            }
        }
        out.sort_by_key(|r| r.index);
        out
    }

    fn existing_selector_bit(f: &IrFn, selector: NodeRef, bit_index: usize) -> Option<NodeRef> {
        for nr in f.node_refs() {
            let Some((sel, index)) = mu::selector_bit(f, nr) else {
                continue;
            };
            if sel == selector && index == bit_index && mu::is_u1(f, nr) {
                return Some(nr);
            }
        }
        None
    }

    fn is_compare(op: Binop) -> bool {
        matches!(
            op,
            Binop::Eq
                | Binop::Ne
                | Binop::Uge
                | Binop::Ugt
                | Binop::Ult
                | Binop::Ule
                | Binop::Sgt
                | Binop::Sge
                | Binop::Slt
                | Binop::Sle
        )
    }

    fn is_reduce(op: Unop) -> bool {
        matches!(op, Unop::OrReduce | Unop::AndReduce | Unop::XorReduce)
    }

    fn is_bool_nary(op: NaryOp) -> bool {
        matches!(
            op,
            NaryOp::And | NaryOp::Or | NaryOp::Xor | NaryOp::Nand | NaryOp::Nor
        )
    }

    fn is_predicate_node(f: &IrFn, nr: NodeRef) -> bool {
        if !mu::is_u1(f, nr) {
            return false;
        }
        match f.get_node(nr).payload {
            NodePayload::Binop(op, _, _) => Self::is_compare(op),
            NodePayload::Unop(Unop::Not, _) => true,
            NodePayload::Unop(op, _) => Self::is_reduce(op),
            NodePayload::Nary(op, _) => Self::is_bool_nary(op),
            NodePayload::Sel { .. }
            | NodePayload::PrioritySel { .. }
            | NodePayload::OneHotSel { .. }
            | NodePayload::Literal(_) => true,
            _ => false,
        }
    }

    fn is_rewire_target(f: &IrFn, nr: NodeRef) -> bool {
        match f.get_node(nr).payload {
            NodePayload::Unop(Unop::Not, _) => mu::is_u1(f, nr),
            NodePayload::Nary(op, _) => mu::is_u1(f, nr) && Self::is_bool_nary(op),
            NodePayload::Sel { .. } | NodePayload::PrioritySel { .. } => mu::is_u1(f, nr),
            NodePayload::Binop(op, _, _) => mu::is_u1(f, nr) && Self::is_compare(op),
            _ => false,
        }
    }

    fn one_bit_operand_slots(f: &IrFn, nr: NodeRef) -> Vec<(usize, NodeRef)> {
        if !Self::is_rewire_target(f, nr) {
            return Vec::new();
        }
        let mut slots = Self::operand_pairs(&f.get_node(nr).payload)
            .into_iter()
            .filter(|(_, dep)| mu::is_u1(f, *dep))
            .collect::<Vec<_>>();
        slots.sort_by_key(|(slot, dep)| (*slot, dep.index));
        slots
    }

    fn push_unique(pool: &mut Vec<NodeRef>, nr: NodeRef) {
        if !pool.contains(&nr) {
            pool.push(nr);
        }
    }

    fn replacement_pool(
        f: &IrFn,
        cone: &[NodeRef],
        arm_guard: Option<NodeRef>,
        guard_base: NodeRef,
    ) -> Vec<NodeRef> {
        let mut pool = Vec::new();
        if let Some(guard) = arm_guard {
            if mu::is_u1(f, guard) {
                Self::push_unique(&mut pool, guard);
            }
        }
        for not_guard in Self::existing_not_of(f, guard_base) {
            Self::push_unique(&mut pool, not_guard);
        }
        for nr in f.node_refs() {
            if mu::is_u1(f, nr) && matches!(f.get_node(nr).payload, NodePayload::Literal(_)) {
                Self::push_unique(&mut pool, nr);
            }
        }
        for nr in cone {
            if Self::is_predicate_node(f, *nr) {
                Self::push_unique(&mut pool, *nr);
            }
        }
        pool.sort_by_key(|r| r.index);
        pool
    }

    fn append_arm_candidates(
        f: &IrFn,
        users_map: &std::collections::HashMap<NodeRef, HashSet<NodeRef>>,
        arm_root: NodeRef,
        arm_guard: Option<NodeRef>,
        guard_base: NodeRef,
        out: &mut Vec<TransformCandidate>,
    ) {
        if out.len() >= MAX_CANDIDATES {
            return;
        }
        let Some(cone) = Self::dependency_cone(f, arm_root) else {
            return;
        };
        let pool = Self::replacement_pool(f, &cone, arm_guard, guard_base);
        if pool.is_empty() {
            return;
        }
        for target in cone {
            if out.len() >= MAX_CANDIDATES {
                return;
            }
            let fanout_cone = Self::compute_fanout_cone(users_map, target);
            for (slot, old_dep) in Self::one_bit_operand_slots(f, target) {
                if out.len() >= MAX_CANDIDATES {
                    return;
                }
                for new_operand in &pool {
                    if out.len() >= MAX_CANDIDATES {
                        return;
                    }
                    if *new_operand == target
                        || *new_operand == old_dep
                        || fanout_cone.contains(new_operand)
                    {
                        continue;
                    }
                    out.push(TransformCandidate {
                        location: TransformLocation::RewireOperand {
                            node: target,
                            operand_slot: slot,
                            new_operand: *new_operand,
                        },
                        always_equivalent: false,
                    });
                }
            }
        }
    }
}

impl PirTransform for GuardedPredicateRewireTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::GuardedPredicateRewire
    }

    fn can_emit_always_equivalent_candidates(&self) -> bool {
        false
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformCandidate> {
        let users_map = compute_users(f);
        let mut out = Vec::new();
        for nr in f.node_refs() {
            if out.len() >= MAX_CANDIDATES {
                break;
            }
            match &f.get_node(nr).payload {
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if default.is_some() || cases.len() != 2 || !mu::is_u1(f, *selector) {
                        continue;
                    }
                    let not_selector = Self::existing_not_of(f, *selector).into_iter().next();
                    Self::append_arm_candidates(
                        f,
                        &users_map,
                        cases[0],
                        not_selector,
                        *selector,
                        &mut out,
                    );
                    Self::append_arm_candidates(
                        f,
                        &users_map,
                        cases[1],
                        Some(*selector),
                        *selector,
                        &mut out,
                    );
                }
                NodePayload::PrioritySel {
                    selector,
                    cases,
                    default: _,
                } => {
                    let Some(selector_width) = mu::bits_width(f, *selector) else {
                        continue;
                    };
                    for (case_index, case) in cases.iter().enumerate() {
                        if case_index >= selector_width {
                            break;
                        }
                        let Some(selector_bit) =
                            Self::existing_selector_bit(f, *selector, case_index)
                        else {
                            continue;
                        };
                        Self::append_arm_candidates(
                            f,
                            &users_map,
                            *case,
                            Some(selector_bit),
                            selector_bit,
                            &mut out,
                        );
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (node, operand_slot, new_operand) = match loc {
            TransformLocation::RewireOperand {
                node,
                operand_slot,
                new_operand,
            } => (*node, *operand_slot, *new_operand),
            TransformLocation::Node(_) => {
                return Err("GuardedPredicateRewireTransform: expected rewire location".to_string());
            }
        };
        if node.index >= f.nodes.len() || new_operand.index >= f.nodes.len() {
            return Err("GuardedPredicateRewireTransform: node ref out of bounds".to_string());
        }
        if node == new_operand {
            return Err("GuardedPredicateRewireTransform: cannot self-depend".to_string());
        }
        if !mu::is_u1(f, new_operand) {
            return Err("GuardedPredicateRewireTransform: replacement must be bits[1]".to_string());
        }
        let users_map = compute_users(f);
        let fanout_cone = Self::compute_fanout_cone(&users_map, node);
        if fanout_cone.contains(&new_operand) {
            return Err(
                "GuardedPredicateRewireTransform: rewire would introduce a cycle".to_string(),
            );
        }
        let old_payload = f.get_node(node).payload.clone();
        let mut found_slot = false;
        let new_payload = remap_payload_with(&old_payload, |(slot, dep)| {
            if slot == operand_slot {
                found_slot = true;
                new_operand
            } else {
                dep
            }
        });
        if !found_slot {
            return Err("GuardedPredicateRewireTransform: operand slot not found".to_string());
        }
        f.get_node_mut(node).payload = new_payload;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser;

    fn parse_fn(ir_text: &str) -> IrFn {
        ir_parser::Parser::new(ir_text).parse_fn().unwrap()
    }

    #[test]
    fn emits_oracle_backed_sel_arm_candidates() {
        let f = parse_fn(
            r#"fn t(p: bits[1] id=1, q: bits[1] id=2, x: bits[8] id=3) -> bits[1] {
  notp: bits[1] = not(p, id=4)
  zero: bits[8] = literal(value=0, id=5)
  eqx: bits[1] = eq(x, zero, id=6)
  arm0: bits[1] = and(q, notp, id=7)
  arm1: bits[1] = or(q, eqx, id=8)
  ret out: bits[1] = sel(p, cases=[arm0, arm1], id=9)
}"#,
        );
        let mut t = GuardedPredicateRewireTransform;
        let candidates = t.find_candidates(&f);
        assert!(!candidates.is_empty());
        assert!(candidates.iter().all(|c| !c.always_equivalent));
        assert!(!t.can_emit_always_equivalent_candidates());
    }

    #[test]
    fn applies_rewire_candidate() {
        let mut f = parse_fn(
            r#"fn t(p: bits[1] id=1, q: bits[1] id=2, x: bits[8] id=3) -> bits[1] {
  notp: bits[1] = not(p, id=4)
  zero: bits[8] = literal(value=0, id=5)
  eqx: bits[1] = eq(x, zero, id=6)
  arm0: bits[1] = and(q, notp, id=7)
  arm1: bits[1] = or(q, eqx, id=8)
  ret out: bits[1] = sel(p, cases=[arm0, arm1], id=9)
}"#,
        );
        let mut t = GuardedPredicateRewireTransform;
        let candidate = t.find_candidates(&f).into_iter().next().unwrap();
        t.apply(&mut f, &candidate.location).unwrap();
    }

    #[test]
    fn emits_priority_sel_candidates_only_with_selector_slice() {
        let f = parse_fn(
            r#"fn t(sel: bits[2] id=1, q: bits[1] id=2, r: bits[1] id=3) -> bits[1] {
  bit0: bits[1] = bit_slice(sel, start=0, width=1, id=4)
  c0: bits[1] = and(q, bit0, id=5)
  c1: bits[1] = or(q, r, id=6)
  zero: bits[1] = literal(value=0, id=7)
  ret out: bits[1] = priority_sel(sel, cases=[c0, c1], default=zero, id=8)
}"#,
        );
        let mut t = GuardedPredicateRewireTransform;
        assert!(!t.find_candidates(&f).is_empty());

        let no_slice = parse_fn(
            r#"fn t(sel: bits[2] id=1, q: bits[1] id=2, r: bits[1] id=3) -> bits[1] {
  c0: bits[1] = and(q, r, id=4)
  c1: bits[1] = or(q, r, id=5)
  zero: bits[1] = literal(value=0, id=6)
  ret out: bits[1] = priority_sel(sel, cases=[c0, c1], default=zero, id=7)
}"#,
        );
        assert!(t.find_candidates(&no_slice).is_empty());
    }

    #[test]
    fn rejects_node_location() {
        let mut f = parse_fn(
            r#"fn t(p: bits[1] id=1) -> bits[1] {
  ret out: bits[1] = not(p, id=2)
}"#,
        );
        let target = f.ret_node_ref.unwrap();
        let err = GuardedPredicateRewireTransform
            .apply(&mut f, &TransformLocation::Node(target))
            .unwrap_err();
        assert!(err.contains("expected rewire location"));
    }
}
