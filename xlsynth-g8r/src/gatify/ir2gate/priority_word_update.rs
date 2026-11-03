// SPDX-License-Identifier: Apache-2.0

//! Share the grant for a word transition across a priority arbiter. Each
//! request can hold the default word or choose the same alternative word. The
//! changed word must already expose each bit as the base bit XOR a signed
//! toggle rail.

use super::{GateEnv, gatify_priority_sel};
use crate::aig::gate::{AigBitVector, AigNode, AigOperand, GateFn};
use crate::aig::get_summary_stats::{AigConeStats, get_output_cone_stats};
use crate::gate_builder::GateBuilder;
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use xlsynth_pir::ir::{Binop, Fn, NaryOp, NodePayload, NodeRef, Unop};

pub(super) fn lower(
    f: &Fn,
    root: NodeRef,
    env: &mut GateEnv,
    gb: &mut GateBuilder,
    ordinary: AigBitVector,
) -> AigBitVector {
    // Only a whole-function mapping can inspect the definitions of operands.
    if !matches!(env.word_recipients.mode, Mode::Discover | Mode::Rewrite)
        || !gb.options.fold
        || !gb.options.hash
    {
        return ordinary;
    }
    let NodePayload::PrioritySel {
        selector,
        cases,
        default: Some(default),
    } = &f.get_node(root).payload
    else {
        return ordinary;
    };
    if ordinary.get_bit_count() <= 1
        || !cases
            .iter()
            .any(|&n| matches!(f.get_node(n).payload, NodePayload::Sel { .. }))
    {
        return ordinary;
    }
    let Some(plan) = requests(f, *default, cases, &mut env.word_recipients.guards) else {
        return ordinary;
    };
    let member = env
        .word_recipients
        .trial
        .as_ref()
        .and_then(|t| t.members.get(&root))
        .copied();
    let transition = env.word_recipients.anchors.contains(&root);
    let inner_owned = plan
        .inner_uses
        .iter()
        .all(|(n, uses)| env.use_count(*n) == *uses);
    let possible_joint = env.use_count(*selector) > 1;
    if !inner_owned && !possible_joint {
        return ordinary;
    }
    let Some(local) = plan
        .arms
        .iter()
        .map(|arm| arm.operand(env, gb))
        .collect::<Option<Vec<_>>>()
    else {
        return ordinary;
    };
    let Ok(priority) = env.get_bit_vector(*selector) else {
        return ordinary;
    };
    if let Some(group) = member.filter(|_| !transition) {
        let (Ok(hold), Ok(take)) = (
            env.get_bit_vector(*default),
            env.get_bit_vector(plan.alternative),
        ) else {
            return ordinary;
        };
        let grant = env
            .word_recipients
            .trial
            .as_mut()
            .unwrap()
            .grant(group, priority, &local, gb);
        return gb.add_mux2_vec(&grant, &take, &hold);
    }
    // The arbiter may use either polarity. A retained endpoint can serve as the
    // base, but replacing a computed endpoint must own every one of its uses.
    for (base, changed, invert) in [
        (*default, plan.alternative, false),
        (plan.alternative, *default, true),
    ] {
        let owned = inner_owned && env.use_count(changed) == plan.internal_uses[&changed];
        if (!owned && !possible_joint)
            || !matches!(
                f.get_node(changed).payload,
                NodePayload::Binop(Binop::Add | Binop::Sub, ..)
                    | NodePayload::ExtNaryAdd { arch: None, .. }
                    | NodePayload::Nary(NaryOp::Xor, _)
                    | NodePayload::Unop(Unop::Not, _)
            )
        {
            continue;
        }
        let (Ok(base), Ok(changed)) = (env.get_bit_vector(base), env.get_bit_vector(changed))
        else {
            continue;
        };
        let Some(toggles) = base
            .iter_lsb_to_msb()
            .zip(changed.iter_lsb_to_msb())
            .map(|(a, b)| xor_factor(gb, *a, *b))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if let Some(group) = member {
            let grant = env
                .word_recipients
                .trial
                .as_mut()
                .unwrap()
                .grant(group, priority, &local, gb);
            // A separately used changed word or inner selection remains live.
            // Select that word directly so its computed value stays shareable.
            if !owned {
                let take = if invert { grant.negate() } else { grant };
                return gb.add_mux2_vec(&take, &changed, &base);
            }
            return apply_toggle(gb, grant, invert, &base, &toggles);
        }
        // An isolated trial cannot charge a common request to all its
        // recipients.
        let joint = possible_joint && env.word_recipients.group(f, root).is_some();
        if !owned && !joint {
            continue;
        }
        if env.word_recipients.mode == Mode::Discover {
            env.word_recipients.mode = Mode::Found;
            return ordinary;
        }
        if joint {
            if env.word_recipients.trial.is_none() {
                env.word_recipients.anchors.insert(root);
                log::debug!(
                    "joint priority grant witness: root={} owned={owned}",
                    f.get_node(root).text_id
                );
            }
            return ordinary;
        }
        let boundary: Vec<_> = base
            .iter_lsb_to_msb()
            .chain(&toggles)
            .chain(priority.iter_lsb_to_msb())
            .chain(&local)
            .copied()
            .collect();
        let (result, trial) = gb.try_region_alternative(ordinary, 0, &boundary, |gb| {
            let grant = lower_grant(gb, priority, &local);
            apply_toggle(gb, grant, invert, &base, &toggles)
        });
        if trial.is_some_and(|trial| trial.accepted) {
            env.word_recipients.applied_isolated = true;
        }
        log::debug!(
            "priority word update: root={} trial={:?}",
            f.get_node(root).text_id,
            trial
        );
        return result;
    }
    ordinary
}

fn lower_grant(gb: &mut GateBuilder, priority: AigBitVector, local: &[AigOperand]) -> AigOperand {
    let bits: Vec<AigBitVector> = local.iter().map(|r| (*r).into()).collect();
    *gatify_priority_sel(gb, 1, priority, &bits, Some(gb.get_false().into()), None).get_lsb(0)
}

fn apply_toggle(
    gb: &mut GateBuilder,
    grant: AigOperand,
    invert: bool,
    base: &AigBitVector,
    toggles: &[AigOperand],
) -> AigBitVector {
    let enable = if invert { grant.negate() } else { grant };
    let result: Vec<_> = base
        .iter_lsb_to_msb()
        .zip(toggles)
        .map(|(bit, toggle)| {
            let active = gb.add_and_binary(enable, *toggle);
            gb.add_xor_binary(*bit, active)
        })
        .collect();
    AigBitVector::from_lsb_is_index_0(&result)
}

/// A trial has its own builder. Gate operands must never cross between maps.
struct JointTrial {
    members: HashMap<NodeRef, NodeRef>,
    grants: HashMap<NodeRef, AigOperand>,
    emitted: usize,
}

impl JointTrial {
    fn grant(
        &mut self,
        group: NodeRef,
        priority: AigBitVector,
        local: &[AigOperand],
        gb: &mut GateBuilder,
    ) -> AigOperand {
        self.emitted += 1;
        let grant = *self
            .grants
            .entry(group)
            .or_insert_with(|| lower_grant(gb, priority, local));
        gb.union_current_pir_node_id_into_operand(grant)
    }
}

/// Ordinary mappings stop searching after the first eligible transition.
#[derive(Default, PartialEq, Eq)]
pub(super) enum Mode {
    #[default]
    Disabled,
    Discover,
    Found,
    Rewrite,
}

/// All word recipients of the same complete, signed request are keyed by the
/// first root in source order. Discover groups only after a real toggle proof.
#[derive(Default)]
pub(super) struct Recipients {
    pub(super) mode: Mode,
    pub(super) applied_isolated: bool,
    guards: NormalizedGuards,
    groups: OnceCell<HashMap<NodeRef, NodeRef>>,
    anchors: HashSet<NodeRef>,
    trial: Option<JointTrial>,
}

impl Recipients {
    pub(super) fn take_trial(
        &mut self,
        demanded: Option<&HashMap<NodeRef, Vec<bool>>>,
    ) -> Option<Self> {
        if self.anchors.is_empty() {
            return None;
        }
        let demanded = demanded?;
        let live = |r: &NodeRef| demanded.get(r).is_some_and(|bits| bits.contains(&true));
        let groups = self.groups.get()?;
        let wanted: HashSet<_> = self
            .anchors
            .iter()
            .filter(|r| live(r))
            .map(|r| groups[r])
            .collect();
        let mut counts = HashMap::<NodeRef, usize>::new();
        let mut members: HashMap<_, _> = groups
            .iter()
            .filter(|(r, g)| live(r) && wanted.contains(*g))
            .map(|(&r, &g)| {
                *counts.entry(g).or_default() += 1;
                (r, g)
            })
            .collect();
        members.retain(|_, g| counts[&*g] > 1);
        if members.is_empty() {
            return None;
        }
        log::debug!(
            "joint priority grants: attempting {} recipients",
            members.len()
        );
        let mut next = std::mem::take(self);
        next.trial = Some(JointTrial {
            members,
            grants: HashMap::new(),
            emitted: 0,
        });
        Some(next)
    }

    pub(super) fn complete(&self) -> bool {
        self.trial.as_ref().is_some_and(|t| {
            // The normal topological walk visits each root once.
            log::debug!(
                "joint priority grants: mapped {} of {} recipients in {} groups",
                t.emitted,
                t.members.len(),
                t.grants.len()
            );
            t.emitted == t.members.len()
        })
    }

    fn group(&mut self, f: &Fn, root: NodeRef) -> Option<NodeRef> {
        self.groups
            .get_or_init(|| {
                let mut first = HashMap::new();
                let mut groups = HashMap::new();
                for (i, n) in f.nodes.iter().enumerate() {
                    let NodePayload::PrioritySel {
                        selector,
                        cases,
                        default: Some(default),
                    } = &n.payload
                    else {
                        continue;
                    };
                    if !matches!(n.ty, xlsynth_pir::ir::Type::Bits(w) if w > 1) {
                        continue;
                    }
                    if let Some(plan) = requests(f, *default, cases, &mut self.guards) {
                        let root = NodeRef { index: i };
                        let leader = *first.entry((*selector, plan.arms)).or_insert(root);
                        if root != leader {
                            groups.insert(leader, leader);
                            groups.insert(root, leader);
                        }
                    }
                }
                groups
            })
            .get(&root)
            .copied()
    }
}

/// Charge every replacement for the complete live cone, including independent
/// consumers that share gates only after structural hashing.
pub(super) fn improves_mapping(baseline: &GateFn, candidate: &GateFn) -> bool {
    let start = log::log_enabled!(log::Level::Debug).then(std::time::Instant::now);
    let before = get_output_cone_stats(baseline);
    let after = get_output_cone_stats(candidate);
    let take = improves(&before, &after);
    log::debug!(
        "priority word updates: accepted={take} compare={:?} live ANDs {} -> {}",
        start.map(|t| t.elapsed()),
        before.and_nodes,
        after.and_nodes
    );
    take
}

fn improves(before: &AigConeStats, after: &AigConeStats) -> bool {
    after.root_depths.len() == before.root_depths.len()
        && after.and_nodes <= before.and_nodes
        && after
            .root_depths
            .iter()
            .zip(&before.root_depths)
            .all(|(a, b)| a <= b)
        && (after.and_nodes < before.and_nodes
            || after.root_depths.iter().max() < before.root_depths.iter().max())
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Request {
    Constant(bool),
    Guard { node: NodeRef, positive: bool },
}

impl Request {
    fn operand(self, env: &GateEnv, gb: &GateBuilder) -> Option<AigOperand> {
        match self {
            Self::Constant(false) => Some(gb.get_false()),
            Self::Constant(true) => Some(gb.get_true()),
            Self::Guard { node, positive } => {
                let wire = *env.get_bit_vector(node).ok()?.get_lsb(0);
                Some(if positive { wire } else { wire.negate() })
            }
        }
    }
}

/// Scalar NOT chains resolved against the unchanged prepared function. Shared
/// prefixes are cached for subsequent arms, roots, and recipient discovery.
#[derive(Default)]
struct NormalizedGuards {
    by_node: HashMap<NodeRef, (NodeRef, bool)>,
    #[cfg(test)]
    not_edges_visited: usize,
}

impl NormalizedGuards {
    /// Reversing the select arms and inverting its guard cancel each other.
    fn request(&mut self, f: &Fn, mut node: NodeRef, take_on_true: bool) -> Request {
        let mut path = Vec::new();
        let (base, mut positive) = loop {
            if let Some(&guard) = self.by_node.get(&node) {
                break guard;
            }
            let NodePayload::Unop(Unop::Not, parent) = f.get_node(node).payload else {
                break (node, true);
            };
            #[cfg(test)]
            {
                self.not_edges_visited += 1;
            }
            path.push(node);
            node = parent;
        };
        for node in path.into_iter().rev() {
            positive = !positive;
            self.by_node.insert(node, (base, positive));
        }
        Request::Guard {
            node: base,
            positive: positive == take_on_true,
        }
    }
}

/// The common alternative word, each arm's request, and its PIR ownership.
struct Plan {
    alternative: NodeRef,
    arms: Vec<Request>,
    internal_uses: HashMap<NodeRef, usize>,
    inner_uses: HashMap<NodeRef, usize>,
}

fn requests(
    f: &Fn,
    default: NodeRef,
    cases: &[NodeRef],
    guards: &mut NormalizedGuards,
) -> Option<Plan> {
    let mut alternative = None;
    let mut arms = Vec::with_capacity(cases.len());
    let mut inner_uses = HashMap::new();
    let mut internal_uses = HashMap::from([(default, 1)]);
    for &case in cases {
        *internal_uses.entry(case).or_default() += 1;
        if case == default {
            arms.push(Request::Constant(false));
            continue;
        }
        let (other, request) = if let NodePayload::Sel {
            selector,
            cases,
            default: None,
        } = &f.get_node(case).payload
        {
            let [off, on] = cases.as_slice() else {
                return None;
            };
            if !matches!(f.get_node(*selector).ty, xlsynth_pir::ir::Type::Bits(1))
                || (*off == default) == (*on == default)
            {
                return None;
            }
            let uses = inner_uses.entry(case).or_insert(0);
            if *uses == 0 {
                *internal_uses.entry(*off).or_default() += 1;
                *internal_uses.entry(*on).or_default() += 1;
            }
            *uses += 1;
            let take_on_true = *off == default;
            let other = if take_on_true { *on } else { *off };
            (other, guards.request(f, *selector, take_on_true))
        } else {
            (case, Request::Constant(true))
        };
        if alternative.replace(other).is_some_and(|old| old != other) {
            return None;
        }
        arms.push(request);
    }
    if inner_uses.is_empty() {
        return None;
    }
    Some(Plan {
        alternative: alternative?,
        arms,
        internal_uses,
        inner_uses,
    })
}

/// Recover the signed coefficient t in changed = base XOR t, including the
/// two-product XOR/XNOR encoding with either child order or operand polarity.
fn xor_factor(gb: &GateBuilder, base: AigOperand, changed: AigOperand) -> Option<AigOperand> {
    if base.node == changed.node {
        return Some(if base == changed {
            gb.get_false()
        } else {
            gb.get_true()
        });
    }
    let AigNode::And2 { a, b, .. } = gb.gates[changed.node.id] else {
        return None;
    };
    if !a.negated || !b.negated {
        return None;
    }
    let AigNode::And2 { a: x, b: y, .. } = gb.gates[a.node.id] else {
        return None;
    };
    let AigNode::And2 { a: u, b: v, .. } = gb.gates[b.node.id] else {
        return None;
    };
    if !((u == x.negate() && v == y.negate()) || (v == x.negate() && u == y.negate())) {
        return None;
    }
    let (pin, other) = if x.node == base.node {
        (x, y)
    } else if y.node == base.node {
        (y, x)
    } else {
        return None;
    };
    // At base=0 exactly one of the two products is enabled. Complementing
    // the sum-of-products output also complements its recovered toggle.
    let at_zero = if pin == base { other.negate() } else { other };
    Some(if changed.negated {
        at_zero
    } else {
        at_zero.negate()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::GateBuilderOptions;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn compare_every_output_before_replacing_the_mapping() {
        let stats = |and_nodes, depths: &[usize]| AigConeStats {
            and_nodes,
            root_depths: depths.to_vec(),
        };
        let stock = stats(20, &[8, 2, 3]);
        assert!(improves(&stock, &stats(19, &[8, 2, 3])));
        assert!(improves(&stock, &stats(20, &[7, 2, 3])));
        assert!(!improves(&stock, &stats(18, &[7, 3, 2])));
        assert!(!improves(&stock, &stats(21, &[7, 1, 2])));
        assert!(!improves(&stock, &stock));
    }

    #[test]
    fn recipients_use_the_full_signed_arm_sequence_independently_of_word_width() {
        for (third_guard, repeated) in [("p", true), ("np", false)] {
            let ir = format!("fn multi(x: bits[6] id=1, y: bits[6] id=2, a: bits[3] id=3, b: bits[3] id=4, p: bits[1] id=5, order: bits[2] id=6) -> (bits[6], bits[3]) {{
                np: bits[1] = not(p, id=7)
                xword: bits[6] = sel(p, cases=[x,y], id=8)
                yword: bits[3] = sel({third_guard}, cases=[a,b], id=9)
                xr: bits[6] = priority_sel(order, cases=[xword, x], default=x, id=10)
                yr: bits[3] = priority_sel(order, cases=[yword, a], default=a, id=11)
                ret out: (bits[6], bits[3]) = tuple(xr, yr, id=12)
            }}");
            let f = Parser::new(&ir).parse_fn().unwrap();
            let mut recipients = Recipients::default();
            let matches: Vec<_> = f
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| matches!(node.payload, NodePayload::PrioritySel { .. }))
                .map(|(index, _)| recipients.group(&f, NodeRef { index }).is_some())
                .collect();
            assert_eq!(matches, [repeated, repeated]);
        }
    }

    #[test]
    fn repeated_arms_visit_each_guard_inversion_once() {
        let f = Parser::new(r#"fn choose(p: bits[1] id=1, x: bits[2] id=2, y: bits[2] id=3, s: bits[6] id=4) -> bits[2] {
            n1: bits[1] = not(p, id=5)
            n2: bits[1] = not(n1, id=6)
            n3: bits[1] = not(n2, id=7)
            a: bits[2] = sel(n3, cases=[x, y], id=8)
            b: bits[2] = sel(n2, cases=[y, x], id=9)
            ret word: bits[2] = priority_sel(s, cases=[a, b, a, b, a, b], default=x, id=10)
        }"#).parse_fn().unwrap();
        let NodePayload::PrioritySel { cases, .. } = &f.get_node(f.ret_node_ref.unwrap()).payload
        else {
            unreachable!()
        };
        let mut guards = NormalizedGuards::default();
        let plan = requests(&f, f.params[1], cases, &mut guards).unwrap();
        let take_on_false = Request::Guard {
            node: f.params[0],
            positive: false,
        };
        assert_eq!(plan.arms.len(), cases.len());
        assert!(plan.arms.iter().all(|&arm| arm == take_on_false));
        assert_eq!(guards.not_edges_visited, 3);
    }

    #[test]
    fn recovers_all_xor_polarities_without_constructing_gates() {
        let mut gb = GateBuilder::new("test".into(), GateBuilderOptions::opt());
        let inputs = gb.add_input("v".into(), 2);
        let a = *inputs.get_lsb(0);
        let b = *inputs.get_lsb(1);
        for x in [a, a.negate()] {
            for y in [b, b.negate()] {
                let word = gb.add_xor_binary(y, x);
                assert_eq!(xor_factor(&gb, x, word), Some(y));
                assert_eq!(xor_factor(&gb, x, word.negate()), Some(y.negate()));
                assert_eq!(xor_factor(&gb, y, word), Some(x));
            }
        }
        let conjunction = gb.add_and_binary(a, b);
        assert_eq!(xor_factor(&gb, a, conjunction), None);
        assert_eq!(xor_factor(&gb, a, b), None);
        assert_eq!(xor_factor(&gb, a, a), Some(gb.get_false()));
        assert_eq!(xor_factor(&gb, a, a.negate()), Some(gb.get_true()));
    }
}
