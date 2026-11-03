// SPDX-License-Identifier: Apache-2.0

//! Map a small clipped signed difference by testing intervals in the unshifted
//! word.

use super::{GateEnv, gatify_uge_via_bit_tests, gatify_ule_via_bit_tests};
use crate::aig::gate::{AigBitVector, AigOperand, AigRef};
use crate::gate_builder::{GateBuilder, ReductionKind};
use xlsynth_pir::ir::{Binop, Fn, NaryOp, NodePayload, NodeRef, Type};

/// Limit compile-time expansion across offset assignments and unclipped
/// distances.
const MAX_ENTRIES: u64 = 32;

pub(super) fn lower(
    f: &Fn,
    root: NodeRef,
    env: &GateEnv,
    gb: &mut GateBuilder,
) -> Option<AigBitVector> {
    if !gb.options.fold || !gb.options.hash {
        return None;
    }
    let plan = Plan::recognize(f, root, env)?;
    let (Ok(data), Ok(offset)) = (
        env.get_bit_vector(plan.data),
        env.get_bit_vector(plan.offset),
    ) else {
        return None;
    };
    let zero = gb.get_false().node;
    let mut controls = Vec::<AigRef>::new();
    for bit in offset.iter_lsb_to_msb() {
        if bit.node != zero && !controls.contains(&bit.node) {
            controls.push(bit.node);
            if (1u64 << controls.len()) * plan.cap > MAX_ENTRIES {
                return None;
            }
        }
    }
    let width = data.get_bit_count();
    let mask = u64::MAX >> (u64::BITS as usize - width);
    let out_width = f.get_node(root).ty.bit_count();
    log::debug!(
        "clipped signed distance: root={} data={} bias={} cap={} width={} offset_atoms={}",
        f.get_node(root).text_id,
        f.get_node(plan.data).text_id,
        plan.bias,
        plan.cap,
        width,
        controls.len()
    );
    let mut words = Vec::with_capacity(1 << controls.len());
    for assignment in 0..1usize << controls.len() {
        let mut value = 0u64;
        for (i, bit) in offset.iter_lsb_to_msb().enumerate() {
            let atom = controls
                .iter()
                .position(|v| *v == bit.node)
                .is_some_and(|j| assignment & (1 << j) != 0);
            value |= u64::from(atom ^ bit.negated) << i;
        }
        let center = value.wrapping_sub(plan.bias) & mask;
        words.push(emit_interval(gb, &data, center, plan.cap, out_width, mask));
    }
    for control in controls {
        words = words
            .chunks_exact(2)
            .map(|ab| {
                gb.add_mux2_vec(
                    &AigOperand {
                        node: control,
                        negated: false,
                    },
                    &ab[1],
                    &ab[0],
                )
            })
            .collect();
    }
    words.pop()
}

struct Plan {
    offset: NodeRef,
    data: NodeRef,
    bias: u64,
    cap: u64,
}

impl Plan {
    fn recognize(f: &Fn, root: NodeRef, env: &GateEnv) -> Option<Self> {
        let NodePayload::Sel {
            selector,
            cases,
            default: None,
        } = &f.get_node(root).payload
        else {
            return None;
        };
        let [full, low] = cases.as_slice() else {
            return None;
        };
        let NodePayload::Binop(Binop::Slt, positive, bound) = f.get_node(*selector).payload else {
            return None;
        };
        let cap = literal(f, *full)?;
        let Type::Bits(w) = f.get_node(positive).ty else {
            return None;
        };
        if !(2..=64).contains(&w)
            || cap == 0
            || cap > MAX_ENTRIES
            || cap >= 1u64 << (w - 1)
            || literal(f, bound)? != cap
        {
            return None;
        }
        let NodePayload::BitSlice { arg, start: 0, .. } = f.get_node(*low).payload else {
            return None;
        };
        if arg != positive {
            return None;
        }
        let NodePayload::Nary(NaryOp::And, operands) = &f.get_node(positive).payload else {
            return None;
        };
        let [a, b] = operands.as_slice() else {
            return None;
        };
        // Materializing a second clipped-distance cone cannot remove a
        // full-word producer when the value, mask, or decision is independently
        // observable.
        if env.use_count(positive) != 2
            || !env.has_single_use(*selector)
            || !env.has_single_use(*low)
        {
            return None;
        }
        let value = [(*a, *b), (*b, *a)].into_iter().find_map(|(value, mask)| {
            let NodePayload::SignExt { arg, .. } = f.get_node(mask).payload else {
                return None;
            };
            let NodePayload::Binop(Binop::Sgt, lhs, zero) = f.get_node(arg).payload else {
                return None;
            };
            (lhs == value
                && literal(f, zero) == Some(0)
                && env.use_count(value) == 2
                && env.has_single_use(mask)
                && env.has_single_use(arg))
            .then_some(value)
        })?;
        let NodePayload::ExtNaryAdd { terms, arch: None } = &f.get_node(value).payload else {
            return None;
        };
        let [a, b] = terms.as_slice() else {
            return None;
        };
        let (offset, neg) = if !a.negated && b.negated {
            (a.operand, b.operand)
        } else if a.negated && !b.negated {
            (b.operand, a.operand)
        } else {
            return None;
        };
        if f.get_node(offset).ty != Type::Bits(w) || f.get_node(neg).ty != Type::Bits(w) {
            return None;
        }
        let mut data = neg;
        let mut bias = 0;
        // Keep a prescribed adder in the path. An ordinary bias may be absorbed
        // into the interval endpoints even when other consumers still use the
        // sum.
        if let NodePayload::ExtNaryAdd { terms, arch: None } = &f.get_node(neg).payload {
            if let [a, b] = terms.as_slice() {
                if let Some((v, k)) = [(a, b), (b, a)].into_iter().find_map(|(v, k)| {
                    (!v.negated
                        && !k.negated
                        && f.get_node(v.operand).ty == Type::Bits(w)
                        && f.get_node(k.operand).ty == Type::Bits(w))
                    .then(|| literal(f, k.operand).map(|n| (v.operand, n)))
                    .flatten()
                }) {
                    data = v;
                    bias = k;
                }
            }
        }
        Some(Self {
            offset,
            data,
            bias,
            cap,
        })
    }
}

fn literal(f: &Fn, n: NodeRef) -> Option<u64> {
    let NodePayload::Literal(v) = &f.get_node(n).payload else {
        return None;
    };
    v.to_u64().ok()
}

/// Each output bit is high on alternating runs of positive distances. The
/// final run extends to the largest signed distance when saturation sets it.
fn emit_interval(
    gb: &mut GateBuilder,
    data: &AigBitVector,
    center: u64,
    cap: u64,
    out_width: usize,
    mask: u64,
) -> AigBitVector {
    let width = data.get_bit_count();
    let signed_max = (1u64 << (width - 1)) - 1;
    let mut bits = Vec::with_capacity(out_width);
    for bit in 0..out_width {
        if bit >= u64::BITS as usize || 1u64 << bit > cap {
            bits.push(gb.get_false());
            continue;
        }
        let stride = 1u64 << bit;
        let mut terms = Vec::new();
        let mut first = stride;
        while first <= cap {
            let last = (first + stride - 1).min(cap);
            let hi = center.wrapping_sub(first) & mask;
            if last == cap {
                let lo = center.wrapping_sub(signed_max) & mask;
                let lower = gb.add_literal(&xlsynth_pir::IrBits::make_ubits(width, lo).unwrap());
                let upper = gb.add_literal(&xlsynth_pir::IrBits::make_ubits(width, hi).unwrap());
                let at_least = gatify_uge_via_bit_tests(gb, 0, data, &lower);
                let at_most = gatify_ule_via_bit_tests(gb, 0, data, &upper);
                terms.push(if lo <= hi {
                    gb.add_and_binary(at_least, at_most)
                } else {
                    gb.add_or_binary(at_least, at_most)
                });
            } else {
                // Cover the finite interval with aligned binary prefixes. All
                // low bits underneath a prefix are unobserved.
                let mut lo = center.wrapping_sub(last) & mask;
                let mut count = last - first + 1;
                while count != 0 {
                    let free = lo.trailing_zeros().min(count.ilog2());
                    let prefix = data
                        .iter_lsb_to_msb()
                        .enumerate()
                        .skip(free as usize)
                        .map(|(i, v)| {
                            if lo & (1u64 << i) == 0 {
                                v.negate()
                            } else {
                                *v
                            }
                        })
                        .collect::<Vec<_>>();
                    terms.push(gb.add_and_nary(&prefix, ReductionKind::Tree));
                    let size = 1u64 << free;
                    count -= size;
                    lo = lo.wrapping_add(size) & mask;
                }
            }
            first += 2 * stride;
        }
        bits.push(gb.add_or_nary(&terms, ReductionKind::Tree));
    }
    AigBitVector::from_lsb_is_index_0(&bits)
}
