// SPDX-License-Identifier: Apache-2.0
//! Describes which predicates and full values may leave a word's observation
//! region.
use super::{ir, literal_bits_if_bits_node};
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use xlsynth_pir::ir::{Binop, NaryOp, NodePayload as Op, NodeRef, Type, Unop};
use xlsynth_pir::ir_utils::Users;

pub(super) type Pin = (usize, bool);

/// A conjunction of signed word bits, optionally complemented.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Cube {
    pub(super) pins: Rc<[Pin]>,
    pub(super) inverted: bool,
}

/// A contiguous view of the root word or a predicate computed from it.
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
                pins: [(*start, !inverted)].into(),
                inverted: false,
            }),
            Self::Test(cube) => Some(cube.clone()),
            _ => None,
        }
    }
}

/// A recognized scalar, including producers used only by other predicates.
pub(super) struct ScalarObservation {
    pub(super) value: NodeRef,
    pub(super) cube: Cube,
    escapes: bool,
}

/// Scalar predicates and non-scalar escapes in node-index order.
pub(super) struct WordObservations {
    pub(super) scalars: Vec<ScalarObservation>,
    pub(super) escapes: Vec<NodeRef>,
}

impl WordObservations {
    /// Predicates returned by the function or consumed outside the region.
    pub(super) fn boundary_scalars(&self) -> impl Iterator<Item = &ScalarObservation> {
        self.scalars.iter().filter(|s| s.escapes)
    }

    /// Follows word views and scalar tests, reserving extra pin copies when
    /// several predicates feed the same Boolean operation.
    pub(super) fn find(
        f: &ir::Fn,
        root: NodeRef,
        users: &Users,
        mut take_pins: impl FnMut(usize) -> bool,
    ) -> Option<Self> {
        // Each comparison can describe up to a whole record. Bound the number
        // visited so unsuccessful searches in long Boolean chains stay local.
        const MAX_REGION_NODES: usize = 128;
        // Bound user scans, operand scans and extra n-ary pins per root.
        // An unrecognized use or a repeated operand need not add a node.
        let mut work_left: usize = 32 * 1024;
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
            let consumers = users.get(&n).unwrap_or_default();
            work_left = work_left.checked_sub(consumers.len())?;
            for &u in consumers {
                if !exprs.contains_key(&u)
                    && let Some(expr) = expression(f, u, &exprs, &mut work_left, &mut take_pins)
                {
                    if exprs.len() == MAX_REGION_NODES {
                        return None;
                    }
                    exprs.insert(u, expr);
                    pending.push_back(u);
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
            if f.get_node_ty(n) == &Type::Bits(1) {
                scalars.push(ScalarObservation {
                    value: n,
                    cube: expr.predicate()?,
                    escapes: escapes_region,
                });
            } else if escapes_region {
                escapes.push(n);
            }
        }
        scalars.sort_unstable_by_key(|s| s.value.index);
        escapes.sort_unstable_by_key(|n| n.index);
        Some(Self { scalars, escapes })
    }
}

fn expression(
    f: &ir::Fn,
    n: NodeRef,
    exprs: &HashMap<NodeRef, Expr>,
    work_left: &mut usize,
    take_pins: &mut impl FnMut(usize) -> bool,
) -> Option<Expr> {
    let view = |n: NodeRef| match exprs.get(&n)? {
        Expr::View {
            start,
            width,
            inverted,
        } => Some((*start, *width, *inverted)),
        _ => None,
    };
    let node = f.get_node(n);
    let test = |pins: Vec<Pin>, inverted| {
        Some(Expr::Test(Cube {
            pins: pins.into(),
            inverted,
        }))
    };
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
            Expr::Test(c) => Some(Expr::Test(Cube {
                pins: c.pins.clone(),
                inverted: !c.inverted,
            })),
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
            // AND and OR are idempotent. Resolve every distinct operand before
            // reserving or expanding pins: a later operand may be independent
            // of the word, or may only become ready on a later visit.
            *work_left = work_left.checked_sub(args.len())?;
            let mut seen = HashSet::new();
            let mut count: usize = 0;
            for a in args {
                if seen.insert(a) {
                    let len = match exprs.get(a)? {
                        Expr::View { width: 1, .. } => 1,
                        Expr::Test(c) if c.inverted == disjunction || c.pins.len() == 1 => {
                            c.pins.len()
                        }
                        _ => return None,
                    };
                    count = count.checked_add(len)?;
                }
            }
            let remaining = work_left.checked_sub(count)?;
            if !take_pins(count) {
                return None;
            }
            *work_left = remaining;
            let mut pins = Vec::with_capacity(count);
            for a in seen {
                let c = exprs[a].predicate()?;
                if c.inverted == disjunction {
                    pins.extend_from_slice(&c.pins);
                } else {
                    let (bit, value) = c.pins[0];
                    pins.push((bit, !value));
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
