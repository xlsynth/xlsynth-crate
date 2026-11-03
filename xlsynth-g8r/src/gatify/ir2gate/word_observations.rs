// SPDX-License-Identifier: Apache-2.0
//! Describes which predicates and full values may leave a word's observation
//! region.
use super::{ir, literal_bits_if_bits_node};
use std::collections::{HashMap, VecDeque};
use xlsynth_pir::ir::{Binop, NaryOp, NodePayload as Op, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils::Users;

pub(super) type Pin = (usize, bool);

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Cube {
    pub(super) pins: Vec<Pin>,
    pub(super) inverted: bool,
}

#[derive(Clone)]
enum Expr {
    View {
        start: usize,
        width: usize,
        inverted: bool,
    },
    Test(Cube),
}

impl Expr {
    fn predicate(&self) -> Option<Cube> {
        match self {
            Self::View {
                start,
                width: 1,
                inverted,
            } => Some(Cube {
                pins: vec![(*start, !inverted)],
                inverted: false,
            }),
            Self::Test(cube) => Some(cube.clone()),
            _ => None,
        }
    }
}

pub(super) struct ScalarObservation {
    pub(super) value: NodeRef,
    pub(super) cube: Cube,
}

pub(super) struct WordObservations {
    pub(super) scalars: Vec<ScalarObservation>,
    pub(super) escapes: Vec<NodeRef>,
}

impl WordObservations {
    pub(super) fn find(f: &ir::Fn, root: NodeRef, users: &Users) -> Option<Self> {
        // Each comparison can describe up to a whole record. Bound the number
        // visited so unsuccessful searches in long Boolean chains stay local.
        const MAX_REGION_NODES: usize = 128;
        let mut exprs = HashMap::from([(
            root,
            Expr::View {
                start: 0,
                width: f.get_node_ty(root).bit_count(),
                inverted: false,
            },
        )]);
        let mut pending = VecDeque::from([root]);
        while let Some(n) = pending.pop_front() {
            for &u in users.get(&n).unwrap_or_default() {
                if !exprs.contains_key(&u) {
                    if let Some(expr) = expression(f, u, &exprs) {
                        if exprs.len() == MAX_REGION_NODES {
                            return None;
                        }
                        exprs.insert(u, expr);
                        pending.push_back(u);
                    }
                }
            }
        }
        let mut scalars = Vec::new();
        let mut escapes = Vec::new();
        for (&n, expr) in &exprs {
            let escapes_region = Some(n) == f.ret_node_ref
                || users
                    .get(&n)
                    .unwrap_or_default()
                    .iter()
                    .any(|u| !exprs.contains_key(u));
            if !escapes_region {
                continue;
            }
            if f.get_node_ty(n) == &Type::Bits(1) {
                let cube = expr.predicate()?;
                scalars.push(ScalarObservation { value: n, cube });
            } else {
                escapes.push(n);
            }
        }
        scalars.sort_unstable_by_key(|s| s.value.index);
        Some(Self { scalars, escapes })
    }
}

fn expression(f: &ir::Fn, n: NodeRef, exprs: &HashMap<NodeRef, Expr>) -> Option<Expr> {
    let view = |n: NodeRef| match exprs.get(&n)? {
        Expr::View {
            start,
            width,
            inverted,
        } => Some((*start, *width, *inverted)),
        _ => None,
    };
    let node = f.get_node(n);
    let test = |pins: Vec<Pin>, inverted| Some(Expr::Test(Cube { pins, inverted }));
    match &node.payload {
        Op::Unop(Unop::Identity, a) => exprs.get(a).cloned(),
        Op::Unop(Unop::Not, a) => match exprs.get(a)? {
            Expr::View {
                start,
                width,
                inverted,
            } => Some(Expr::View {
                start: *start,
                width: *width,
                inverted: !inverted,
            }),
            Expr::Test(c) => test(c.pins.clone(), !c.inverted),
        },
        Op::BitSlice {
            arg,
            start: offset,
            width,
        } => {
            let (start, _, inverted) = view(*arg)?;
            Some(Expr::View {
                start: start + offset,
                width: *width,
                inverted,
            })
        }
        Op::TupleIndex { tuple, index } => {
            let (start, _, inverted) = view(*tuple)?;
            let field = f
                .get_node_ty(*tuple)
                .tuple_get_flat_bit_slice_for_index(*index)
                .ok()?;
            Some(Expr::View {
                start: start + field.start,
                width: field.limit - field.start,
                inverted,
            })
        }
        Op::Binop(op @ (Binop::Eq | Binop::Ne), a, b) => {
            let (source, literal) = if matches!(f.get_node(*b).payload, Op::Literal(_)) {
                (*a, *b)
            } else {
                (*b, *a)
            };
            let (start, width, inverted) = view(source)?;
            let value = literal_bits_if_bits_node(f, literal)?;
            let pins = (0..width)
                .map(|bit| (start + bit, value.get_bit(bit).unwrap() ^ inverted))
                .collect();
            test(pins, *op == Binop::Ne)
        }
        Op::Unop(op @ (Unop::AndReduce | Unop::OrReduce), a) => {
            let (start, width, inverted) = view(*a)?;
            let ones = *op == Unop::AndReduce;
            test(
                (0..width).map(|i| (start + i, ones ^ inverted)).collect(),
                !ones,
            )
        }
        Op::Nary(op @ (NaryOp::And | NaryOp::Nand | NaryOp::Or | NaryOp::Nor), args)
            if node.ty == Type::Bits(1) =>
        {
            let disjunction = matches!(op, NaryOp::Or | NaryOp::Nor);
            let mut pins = Vec::new();
            for a in args {
                let c = exprs.get(a)?.predicate()?;
                if c.inverted == disjunction {
                    pins.extend(c.pins);
                } else if let [(bit, value)] = c.pins[..] {
                    pins.push((bit, !value));
                } else {
                    return None;
                }
            }
            pins.sort_unstable();
            pins.dedup();
            if pins.windows(2).any(|p| p[0].0 == p[1].0) {
                return None;
            }
            test(pins, matches!(op, NaryOp::Or | NaryOp::Nand))
        }
        _ => None,
    }
}
