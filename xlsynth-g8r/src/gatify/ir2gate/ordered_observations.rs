// SPDX-License-Identifier: Apache-2.0
//! Distributes a monotone scalar observation over an ordered two-way selection.
use super::word_observations::{Cube, WordObservations};
use super::*;
use ir::{Binop as B, NodePayload as Op, NodeRef, Type, Unop};

#[derive(Default)]
pub(super) struct OrderedObservations {
    pub(super) receivers: HashMap<NodeRef, Offer>,
}

/// A scalar predicate that can be distributed over the selected word's arms.
pub(super) struct Offer {
    word: NodeRef,
    arms: [NodeRef; 2],
    use_or: bool,
    predicate: Predicate,
}

enum Predicate {
    Cube(Cube),
    Threshold(B, NodeRef),
}

impl OrderedObservations {
    /// Claim each scalar only when the word and its selector have no escaping
    /// uses.
    pub(super) fn analyze(f: &ir::Fn, topological: &[NodeRef]) -> Self {
        let mut plan = Self::default();
        let mut users = None;
        let mut scalar_outputs = None;
        let mut regions = 0;
        'regions: for (index, node) in f.nodes.iter().enumerate() {
            let n = NodeRef { index };
            let (selector, a, b) = match &node.payload {
                Op::Sel {
                    selector,
                    cases,
                    default: None,
                } if cases.len() == 2 => (*selector, cases[0], cases[1]),
                Op::PrioritySel {
                    selector,
                    cases,
                    default: Some(d),
                } if cases.len() == 1 => (*selector, *d, cases[0]),
                _ => continue,
            };
            let Type::Bits(width @ 2..=128) = node.ty else {
                continue;
            };
            if a == b || f.get_node_ty(selector) != &Type::Bits(1) {
                continue;
            }
            let Some((maximum, signed)) = ordered_arms(f, selector, a, b, width) else {
                continue;
            };
            // The shared observer caps each walk at 128 nodes. Also bound its
            // invocation count so repeated overlapping regions stay local.
            if regions == 32 {
                break;
            }
            regions += 1;
            let users = users.get_or_insert_with(|| ir_utils::compute_users(f));
            let sole_use = |value, user| {
                f.ret_node_ref != Some(value) && users.get(&value) == Some(&[user][..])
            };
            if !sole_use(selector, n)
                || matches!(f.get_node(selector).payload, Op::BitSlice { arg, .. } if !sole_use(arg, selector))
            {
                continue;
            }
            let mut offers = Vec::new();
            let observations = WordObservations::find_with_direct_observer(f, n, users, |user| {
                if let Op::Binop(op, l, r) = f.get_node(user).payload
                    && order(op).is_some()
                    && let Some(cmp) = normalize_cmp_literal_rhs(f, op, l, r)
                    && let Some((upward, cmp_signed)) = order(cmp.binop)
                    && cmp_signed == signed
                    && cmp.lhs == n
                {
                    offers.push((user, upward, Predicate::Threshold(cmp.binop, cmp.rhs)));
                    true
                } else {
                    false
                }
            });
            let Some(observations) = observations else {
                continue;
            };
            if !observations.escapes.is_empty() {
                continue;
            }
            for observation in observations.scalars {
                let cube = observation.cube;
                let Some(upward) = prefix_direction(&cube, width, signed) else {
                    continue 'regions;
                };
                offers.push((observation.value, upward, Predicate::Cube(cube)));
            }
            if offers.is_empty() {
                continue;
            }
            let scalar_outputs =
                scalar_outputs.get_or_insert_with(|| closed_scalar_uses(f, topological));
            if !offers
                .iter()
                .all(|(receiver, _, _)| scalar_outputs[receiver.index])
            {
                continue;
            }
            for (receiver, upward, predicate) in offers {
                plan.receivers.entry(receiver).or_insert(Offer {
                    word: n,
                    arms: [a, b],
                    use_or: maximum == upward,
                    predicate,
                });
            }
        }
        plan
    }
}

/// A distributed predicate can feed scalar logic and output wiring. If any
/// use enters word computation, preserve the joint word and its observers.
fn closed_scalar_uses(f: &ir::Fn, topological: &[NodeRef]) -> Vec<bool> {
    let mut closed = vec![true; f.nodes.len()];
    for &node in topological.iter().rev() {
        let user = f.get_node(node);
        let operands = ir_utils::operands(&user.payload);
        let wiring = matches!(
            user.payload,
            Op::Nary(ir::NaryOp::Concat, _)
                | Op::Tuple(_)
                | Op::TupleIndex { .. }
                | Op::BitSlice { .. }
                | Op::ZeroExt { .. }
                | Op::SignExt { .. }
                | Op::Unop(Unop::Not | Unop::Reverse, _)
        );
        if !closed[node.index]
            || !(wiring
                || user.ty == Type::Bits(1)
                    && operands
                        .iter()
                        .all(|&arg| f.get_node_ty(arg) == &Type::Bits(1)))
        {
            for operand in operands {
                closed[operand.index] = false;
            }
        }
    }
    closed
}

impl Offer {
    pub(super) fn lower(
        &self,
        f: &ir::Fn,
        n: NodeRef,
        gb: &mut GateBuilder,
        env: &GateEnv,
    ) -> Option<AigOperand> {
        // If folding already selected an arm, keep the ordinary predicate on
        // that word. Distributing it could make the other arm live again.
        let word = &env.ir_to_g8[&self.word];
        if self.arms.iter().any(|arm| &env.ir_to_g8[arm] == word) {
            return None;
        }
        let [a, b] = self.arms.map(|arm| match &self.predicate {
            Predicate::Threshold(op, literal) => gatify_cmp_with_optional_literal_rhs(
                f,
                gb,
                env,
                f.get_node(n).text_id,
                *op,
                arm,
                *literal,
            ),
            Predicate::Cube(c) => {
                let bits = env
                    .get_bit_vector(arm)
                    .expect("ordered arm precedes observer");
                let pins: Vec<_> = c
                    .pins
                    .iter()
                    .map(|&(i, high)| {
                        let rail = *bits.get_lsb(i);
                        if high { rail } else { rail.negate() }
                    })
                    .collect();
                let hit = gb.add_and_nary(&pins, ReductionKind::Tree);
                if c.inverted { hit.negate() } else { hit }
            }
        });
        Some(if self.use_or {
            gb.add_or_binary(a, b)
        } else {
            gb.add_and_binary(a, b)
        })
    }
}

// Monotone cubes are contiguous leading bits, all high or all low in the
// biased order key. Signed order complements just the sign bit of that key.
fn prefix_direction(cube: &Cube, width: usize, signed: bool) -> Option<bool> {
    let first = width.checked_sub(cube.pins.len())?;
    let mut direction = None;
    for (offset, &(pin, value)) in cube.pins.iter().enumerate() {
        let high = value ^ (signed && pin == width - 1);
        if pin != first + offset || direction.is_some_and(|known| known != high) {
            return None;
        }
        direction = Some(high);
    }
    direction.map(|up| up ^ cube.inverted)
}

fn order(op: B) -> Option<(bool, bool)> {
    match op {
        B::Ugt | B::Uge => Some((true, false)),
        B::Ult | B::Ule => Some((false, false)),
        B::Sgt | B::Sge => Some((true, true)),
        B::Slt | B::Sle => Some((false, true)),
        _ => None,
    }
}

fn ordered_arms(
    f: &ir::Fn,
    selector: NodeRef,
    a: NodeRef,
    b: NodeRef,
    width: usize,
) -> Option<(bool, bool)> {
    let (greater, lesser, signed) = match &f.get_node(selector).payload {
        Op::Binop(op, l, r) => {
            let (forward, signed) = order(*op)?;
            if forward {
                (*l, *r, signed)
            } else {
                (*r, *l, signed)
            }
        }
        op => {
            let (l, r) = match op {
                // x + !y + carry_in selects x > y except at equality,
                // where either arm is the same word for both carry-in values.
                Op::ExtCarryOut { lhs, rhs, .. } => (*lhs, *rhs),
                Op::BitSlice {
                    arg,
                    start,
                    width: 1,
                } if *start == width && f.get_node_ty(*arg) == &Type::Bits(width + 1) => {
                    let Op::ExtNaryAdd { terms, .. } = &f.get_node(*arg).payload else {
                        return None;
                    };
                    let [l, r] = terms.as_slice() else {
                        return None;
                    };
                    if l.signed || r.signed || l.negated || r.negated {
                        return None;
                    }
                    (l.operand, r.operand)
                }
                _ => return None,
            };
            if f.get_node_ty(l) != &Type::Bits(width) || f.get_node_ty(r) != &Type::Bits(width) {
                return None;
            }
            return [(l, r), (r, l)].into_iter().find_map(|(x, ny)| {
                let Op::Unop(Unop::Not, y) = f.get_node(ny).payload else {
                    return None;
                };
                if (b, a) == (x, y) {
                    Some((true, false))
                } else if (a, b) == (x, y) {
                    Some((false, false))
                } else {
                    None
                }
            });
        }
    };
    if (b, a) == (greater, lesser) {
        Some((true, signed))
    } else if (a, b) == (greater, lesser) {
        Some((false, signed))
    } else {
        None
    }
}

#[cfg(test)]
mod scalar_use_tests {
    use super::{OrderedObservations, Type, ir_utils};
    use xlsynth_pir::ir_parser::Parser;
    use xlsynth_pir::{FnBuilder, IrValue};

    fn offers(tail: &str, return_width: usize) -> usize {
        let source = format!(
            r#"package pred_use
top fn f(a: bits[4] id=1, b: bits[4] id=2, flag: bits[1] id=3) -> bits[{return_width}] {{
  c: bits[1] = uge(a, b, id=4)
  hi: bits[4] = sel(c, cases=[b, a], id=5)
  k: bits[4] = literal(value=5, id=6)
  threshold: bits[1] = uge(hi, k, id=7)
  last: bits[4] = literal(value=15, id=8)
  endpoint: bits[1] = eq(hi, last, id=9)
  {tail}
}}"#
        );
        let package = Parser::new(&source).parse_and_validate_package().unwrap();
        let f = package.get_fn("f").unwrap();
        OrderedObservations::analyze(f, &ir_utils::get_topological(f))
            .receivers
            .len()
    }

    #[test]
    fn packed_boolean_consumers_admit_both_observers() {
        let tail = "x: bits[1] = xor(endpoint, flag, id=10)
  ret out: bits[2] = concat(threshold, x, id=11)";
        assert_eq!(offers(tail, 2), 2);
    }

    #[test]
    fn packing_the_word_with_its_predicates_rejects_distribution() {
        let tail = "ret out: bits[6] = concat(hi, threshold, endpoint, id=10)";
        assert_eq!(offers(tail, 6), 0);
    }

    #[test]
    fn direct_thresholds_on_one_word_can_exceed_the_cube_walk_limit() {
        let mut builder = FnBuilder::new("thresholds");
        let a = builder.param("a", Type::Bits(16)).unwrap();
        let b = builder.param("b", Type::Bits(16)).unwrap();
        let selector = builder.uge(a, b).unwrap();
        let maximum = builder.select(selector, &[b, a], None).unwrap();
        let mut thresholds = Vec::new();
        for k in 0..10_000 {
            let literal = builder
                .literal(IrValue::make_ubits(16, k).unwrap())
                .unwrap();
            thresholds.push(builder.ugt(maximum, literal).unwrap());
        }
        let output = builder.concat(&thresholds).unwrap();
        let f = builder.build(output).unwrap();
        let plan = OrderedObservations::analyze(&f, &ir_utils::get_topological(&f));
        assert_eq!(plan.receivers.len(), thresholds.len());
    }

    #[test]
    fn any_word_consumer_preserves_the_whole_observer_group() {
        let tail = "x: bits[1] = xor(endpoint, flag, id=10)
  word: bits[4] = sign_ext(x, new_bit_count=4, id=11)
  sum: bits[4] = add(word, b, id=12)
  ret out: bits[5] = concat(threshold, sum, id=13)";
        assert_eq!(offers(tail, 5), 0);
    }
}
