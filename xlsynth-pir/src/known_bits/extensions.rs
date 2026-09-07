// SPDX-License-Identifier: Apache-2.0

//! Direct transfers for extension operations, without constructing a lowered
//! graph.

use super::{KnownBits, KnownValue, bits};
use crate::IrBits;
use crate::ir::{self, Binop, NodePayload, Type, Unop};

/// Bounds conditional normalization work while retaining linear-time facts for
/// larger domains. One fixed shift is always evaluated, regardless of width.
const NORMALIZE_BIT_BUDGET: usize = 65_536;

fn bit_at(value: &KnownBits, index: usize) -> Option<bool> {
    value
        .mask()
        .get_bit(index)
        .unwrap()
        .then(|| value.value().get_bit(index).unwrap())
}

/// Constructs canonical facts without allocating a three-state vector.
fn from_bit_fn(width: usize, bit: impl Fn(usize) -> Option<bool>) -> KnownBits {
    KnownBits::from_mask_value(
        IrBits::from_lsb_fn(width, |i| bit(i).is_some()),
        IrBits::from_lsb_fn(width, |i| bit(i).unwrap_or(false)),
    )
    .expect("mask and value have the same width")
}

/// Visits feasible first-set indices in priority order, including the all-zero
/// sentinel when possible. No assignments to unrelated input bits are expanded.
fn visit_first_set(arg: &KnownBits, lsb_prio: bool, mut visit: impl FnMut(usize)) {
    for step in 0..arg.bit_count() {
        let index = if lsb_prio {
            step
        } else {
            arg.bit_count() - 1 - step
        };
        match bit_at(arg, index) {
            Some(false) => {
                // A known-zero bit cannot be the first set bit.
            }
            None => visit(index),
            Some(true) => {
                visit(index);
                return;
            }
        }
    }
    visit(arg.bit_count());
}

/// Visits exactly the feasible leading-zero counts, including zero-input N.
fn visit_clz(arg: &KnownBits, mut visit: impl FnMut(usize)) {
    visit_first_set(arg, false, |index| {
        visit(if index == arg.bit_count() {
            index
        } else {
            arg.bit_count() - 1 - index
        });
    });
}

/// Joins machine-sized counts plus offsets without dropping their overflow bit.
/// Both inputs are usize, so their exact sum needs at most usize::BITS + 1
/// bits.
struct CountFacts {
    ones: usize,
    zeros: usize,
    carry_one: bool,
    carry_zero: bool,
}

impl CountFacts {
    fn new() -> Self {
        Self {
            ones: usize::MAX,
            zeros: usize::MAX,
            carry_one: true,
            carry_zero: true,
        }
    }

    fn observe(&mut self, count: usize, offset: usize) {
        let (low, carry) = count.overflowing_add(offset);
        self.ones &= low;
        self.zeros &= !low;
        self.carry_one &= carry;
        self.carry_zero &= !carry;
    }

    fn finish(self, width: usize) -> KnownBits {
        from_bit_fn(width, |i| {
            if i < usize::BITS as usize {
                (((self.ones | self.zeros) >> i) & 1 != 0).then(|| (self.ones >> i) & 1 != 0)
            } else if i == usize::BITS as usize {
                (self.carry_one || self.carry_zero).then_some(self.carry_one)
            } else {
                Some(false)
            }
        })
    }
}

fn clz(arg: &KnownBits, offset: usize, width: usize) -> KnownBits {
    let mut result = CountFacts::new();
    visit_clz(arg, |count| result.observe(count, offset));
    result.finish(width)
}

fn priority_encode(arg: &KnownBits, lsb_prio: bool, width: usize) -> KnownBits {
    let mut result = CountFacts::new();
    visit_first_set(arg, lsb_prio, |index| result.observe(index, 0));
    result.finish(width)
}

/// Each carry depends on independent current-column bits and the incoming
/// carry.
fn carry_out(lhs: &KnownBits, rhs: &KnownBits, carry: &KnownBits) -> KnownBits {
    let mut carry = bit_at(carry, 0);
    for i in 0..lhs.bit_count() {
        carry = bits::full_adder(bit_at(lhs, i), bit_at(rhs, i), carry).1;
    }
    from_bit_fn(1, |_| carry)
}

/// Saturated extrema answer every threshold needed by a low-bit mask.
fn mask_low(count: &KnownBits, width: usize) -> KnownBits {
    let mut minimum = 0usize;
    let mut maximum = 0usize;
    for i in (0..count.bit_count()).rev() {
        let bit = bit_at(count, i);
        minimum = minimum
            .saturating_mul(2)
            .saturating_add(usize::from(bit == Some(true)))
            .min(width);
        maximum = maximum
            .saturating_mul(2)
            .saturating_add(usize::from(bit != Some(false)))
            .min(width);
    }
    from_bit_fn(width, |i| {
        if i < minimum {
            Some(true)
        } else if i >= maximum {
            Some(false)
        } else {
            None
        }
    })
}

/// Conditions the input on one feasible CLZ before applying its fixed shift.
fn normalize_given_clz(
    arg: &KnownBits,
    count: usize,
    offset: usize,
    width: usize,
    condition: bool,
) -> KnownBits {
    let shift = count.saturating_add(offset);
    if count == arg.bit_count() || shift >= width || i64::try_from(shift).is_err() {
        return KnownBits::constant(&IrBits::zero(width));
    }
    if !condition {
        return bits::shift_fixed(arg, shift, width, false, false);
    }
    let first_one = arg.bit_count() - 1 - count;
    let conditional = from_bit_fn(arg.bit_count(), |i| {
        if i > first_one {
            Some(false)
        } else if i == first_one {
            Some(true)
        } else {
            bit_at(arg, i)
        }
    });
    bits::shift_fixed(&conditional, shift, width, false, false)
}

/// Retains common low/high zeros and the normalized leading one in linear time.
fn normalize_large_domain(
    arg: &KnownBits,
    min_count: usize,
    zero_input_possible: bool,
    offset: usize,
    width: usize,
) -> KnownBits {
    let minimum_shift = min_count.saturating_add(offset);
    if minimum_shift >= width || i64::try_from(minimum_shift).is_err() {
        return KnownBits::constant(&IrBits::zero(width));
    }
    let trailing_zeros = (0..arg.bit_count())
        .take_while(|&i| bit_at(arg, i) == Some(false))
        .count();
    let low_zeros = minimum_shift.saturating_add(trailing_zeros);
    let leading_one = arg
        .bit_count()
        .checked_sub(1)
        .and_then(|i| i.checked_add(offset));
    from_bit_fn(width, |i| {
        if i < low_zeros || leading_one.is_some_and(|position| i > position) {
            Some(false)
        } else if !zero_input_possible && leading_one == Some(i) && i64::try_from(i).is_ok() {
            Some(true)
        } else {
            None
        }
    })
}

/// Keeps CLZ precise even when a large conditional normalized-value join is
/// capped.
fn normalize_left(
    arg: &KnownBits,
    offset: usize,
    width: usize,
    clz_width: Option<usize>,
) -> KnownValue {
    let candidate_limit = (NORMALIZE_BIT_BUDGET / width.max(arg.bit_count()).max(1)).max(1);
    let mut candidates = Vec::new();
    let mut capped = false;
    let mut count_facts = CountFacts::new();
    let mut min_count = usize::MAX;
    let mut zero_input_possible = false;
    visit_clz(arg, |count| {
        count_facts.observe(count, 0);
        min_count = min_count.min(count);
        zero_input_possible |= count == arg.bit_count();
        if candidates.len() < candidate_limit {
            candidates.push(count);
        } else {
            capped = true;
        }
    });
    let normalized = if capped {
        normalize_large_domain(arg, min_count, zero_input_possible, offset, width)
    } else {
        let condition = candidates.len() > 1;
        let mut result: Option<KnownBits> = None;
        for count in candidates {
            let next = normalize_given_clz(arg, count, offset, width, condition);
            result = Some(match result {
                Some(previous) => previous.join(&next),
                None => next,
            });
        }
        result.expect("every input domain has a feasible leading-zero count")
    };
    match clz_width {
        Some(width) => KnownValue::Tuple(vec![
            KnownValue::Bits(normalized),
            KnownValue::Bits(count_facts.finish(width)),
        ]),
        None => KnownValue::Bits(normalized),
    }
}

/// Evaluates a validated extension using operands in ir_utils::operands order.
pub(super) fn evaluate(node: &ir::Node, operands: &[&KnownBits]) -> KnownValue {
    let width = || match node.ty {
        Type::Bits(width) => width,
        _ => unreachable!("extension has a validated bits result"),
    };
    KnownValue::Bits(match &node.payload {
        NodePayload::ExtCarryOut { .. } => carry_out(operands[0], operands[1], operands[2]),
        NodePayload::ExtPrioEncode { lsb_prio, .. } => {
            priority_encode(operands[0], *lsb_prio, width())
        }
        NodePayload::ExtClz {
            offset,
            new_bit_count,
            ..
        } => clz(operands[0], *offset, *new_bit_count),
        NodePayload::ExtNormalizeLeft {
            shift_offset,
            normalized_bit_count,
            clz_bit_count,
            ..
        } => {
            return normalize_left(
                operands[0],
                *shift_offset,
                *normalized_bit_count,
                *clz_bit_count,
            );
        }
        NodePayload::ExtMaskLow { .. } => mask_low(operands[0], width()),
        NodePayload::ExtNaryAdd { terms, arch: _ } => {
            // Architecture changes lowering, not the modular arithmetic
            // contract.
            let mut result = KnownBits::constant(&IrBits::zero(width()));
            for (term, operand) in terms.iter().zip(operands) {
                let mut value = bits::resize(operand, width(), term.signed);
                if term.negated {
                    value = bits::unop(Unop::Neg, &value);
                }
                result = bits::binop(Binop::Add, &result, &value, width());
            }
            result
        }
        _ => unreachable!("extension transfer called on a non-extension node"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IrValue;
    use crate::ir::{ExtNaryAddArchitecture, ExtNaryAddTerm, NodeRef};

    fn domain(width: usize) -> Vec<KnownBits> {
        (0..3usize.pow(width as u32))
            .map(|mut index| {
                let bits = (0..width)
                    .map(|_| {
                        let bit = match index % 3 {
                            0 => Some(false),
                            1 => Some(true),
                            _ => None,
                        };
                        index /= 3;
                        bit
                    })
                    .collect::<Vec<_>>();
                KnownBits::from_lsb_bits(&bits)
            })
            .collect()
    }

    fn concrete(value: &KnownBits) -> Vec<usize> {
        (0..1usize << value.bit_count())
            .filter(|&candidate| {
                value.contains(&IrBits::make_ubits(value.bit_count(), candidate as u64).unwrap())
            })
            .collect()
    }

    fn constant_usize(value: usize, width: usize) -> IrBits {
        IrBits::from_lsb_is_0(
            &(0..width)
                .map(|i| i < usize::BITS as usize && (value >> i) & 1 != 0)
                .collect::<Vec<_>>(),
        )
    }

    fn exact_join(values: impl IntoIterator<Item = IrBits>) -> KnownBits {
        values
            .into_iter()
            .map(|value| KnownBits::constant(&value))
            .reduce(|a, b| a.join(&b))
            .unwrap()
    }

    fn concrete_clz(value: usize, width: usize) -> usize {
        (0..width)
            .take_while(|&i| value & (1 << (width - 1 - i)) == 0)
            .count()
    }

    fn node(payload: NodePayload, ty: Type) -> ir::Node {
        ir::Node {
            text_id: 1,
            name: None,
            ty,
            payload,
            pos: None,
        }
    }

    #[test]
    fn carry_out_matches_exact_small_domains_including_zero_width() {
        for width in 0..=3 {
            for lhs in domain(width) {
                for rhs in domain(width) {
                    for carry in domain(1) {
                        let mut results = Vec::new();
                        for a in concrete(&lhs) {
                            for b in concrete(&rhs) {
                                for c in concrete(&carry) {
                                    results.push(IrBits::bool((a + b + c) >> width != 0));
                                }
                            }
                        }
                        let expected = exact_join(results);
                        assert_eq!(carry_out(&lhs, &rhs, &carry), expected);
                    }
                }
            }
        }
        let ones = KnownBits::constant(&IrBits::all_ones(129));
        let zero = KnownBits::constant(&IrBits::zero(129));
        let one_bit = KnownBits::constant(&IrBits::bool(true));
        assert_eq!(carry_out(&ones, &zero, &one_bit), one_bit);
    }

    #[test]
    fn count_encoders_match_exact_small_domains_and_wrapping_offsets() {
        for width in 0..=4 {
            for arg in domain(width) {
                for out_width in [0, 1, 3, 65] {
                    for offset in [0, 1, 3, usize::MAX] {
                        let expected = exact_join(concrete(&arg).into_iter().map(|value| {
                            constant_usize(concrete_clz(value, width), out_width)
                                .add(&constant_usize(offset, out_width))
                        }));
                        assert_eq!(clz(&arg, offset, out_width), expected);
                    }
                    for lsb_prio in [false, true] {
                        let expected = exact_join(concrete(&arg).into_iter().map(|value| {
                            let index = if value == 0 {
                                width
                            } else if lsb_prio {
                                value.trailing_zeros() as usize
                            } else {
                                usize::BITS as usize - 1 - value.leading_zeros() as usize
                            };
                            constant_usize(index, out_width)
                        }));
                        assert_eq!(priority_encode(&arg, lsb_prio, out_width), expected);
                    }
                }
            }
        }
        let zero = KnownBits::constant(&IrBits::zero(129));
        let expected = constant_usize(129, 129).add(&constant_usize(usize::MAX, 129));
        assert_eq!(clz(&zero, usize::MAX, 129), KnownBits::constant(&expected));
        assert_eq!(priority_encode(&zero, true, 8), clz(&zero, 0, 8));
    }

    #[test]
    fn mask_low_matches_exact_small_domains_and_wide_saturation() {
        for count_width in 0..=4 {
            for count in domain(count_width) {
                for width in 0..=7 {
                    let expected = exact_join(concrete(&count).into_iter().map(|count| {
                        IrBits::from_lsb_is_0(&(0..width).map(|i| i < count).collect::<Vec<_>>())
                    }));
                    assert_eq!(mask_low(&count, width), expected);
                }
            }
        }
        let large = from_bit_fn(129, |i| if i == 128 { Some(true) } else { None });
        assert_eq!(
            mask_low(&large, 257),
            KnownBits::constant(&IrBits::all_ones(257))
        );
        let small = from_bit_fn(129, |i| if i < 2 { None } else { Some(false) });
        assert_eq!(mask_low(&small, 5).to_ternary_string(), "00XXX");
    }

    #[test]
    fn normalization_matches_exact_small_domains_including_truncation() {
        for input_width in 0..=4 {
            for arg in domain(input_width) {
                for width in 0..=input_width + 2 {
                    for offset in [0, 1, 3, usize::MAX] {
                        let expected = exact_join(concrete(&arg).into_iter().map(|value| {
                            let shift = concrete_clz(value, input_width).saturating_add(offset);
                            let normalized = if shift >= width { 0 } else { value << shift };
                            constant_usize(normalized, width)
                        }));
                        for clz_width in [None, Some(0), Some(3), Some(65)] {
                            let actual = normalize_left(&arg, offset, width, clz_width);
                            let normalized = match clz_width {
                                None => actual.as_bits().unwrap(),
                                Some(clz_width) => {
                                    assert_eq!(
                                        actual.leaf(&[1]).unwrap(),
                                        &clz(&arg, 0, clz_width)
                                    );
                                    actual.leaf(&[0]).unwrap()
                                }
                            };
                            assert_eq!(*normalized, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn large_normalization_retains_clz_leading_one_and_guaranteed_zeros() {
        let arg = from_bit_fn(513, |i| {
            if i < 3 {
                Some(false)
            } else if i == 3 {
                Some(true)
            } else {
                None
            }
        });
        let actual = normalize_left(&arg, 2, 520, Some(65));
        let normalized = actual.leaf(&[0]).unwrap();
        for i in 0..5 {
            assert_eq!(bit_at(normalized, i), Some(false));
        }
        assert_eq!(bit_at(normalized, 514), Some(true));
        for i in 515..520 {
            assert_eq!(bit_at(normalized, i), Some(false));
        }
        assert_eq!(actual.leaf(&[1]).unwrap(), &clz(&arg, 0, 65));
        for first_one in 3..513 {
            let value = IrBits::from_lsb_fn(513, |i| i == first_one || i == 3);
            let shift = 512 - first_one + 2;
            let expected = IrBits::from_lsb_fn(520, |i| {
                i.checked_sub(shift)
                    .is_some_and(|i| i < 513 && value.get_bit(i).unwrap())
            });
            assert!(normalized.contains(&expected));
        }
        let truncated = normalize_left(&arg, 2, 5, None);
        assert_eq!(
            truncated,
            KnownValue::Bits(KnownBits::constant(&IrBits::zero(5)))
        );
        // Clipping the normalized leading one does not imply the entire
        // shifted value is zero: lower set bits can remain in bounds.
        let clipped = normalize_left(&arg, 1, 513, None);
        let surviving_low_bit = IrBits::from_lsb_fn(513, |i| i == 4);
        assert!(clipped.contains(&IrValue::from_bits(&surviving_low_bit)));
        assert_ne!(
            clipped,
            KnownValue::Bits(KnownBits::constant(&IrBits::zero(513)))
        );
        let oversized = normalize_left(&arg, usize::MAX, 520, Some(129));
        assert_eq!(
            oversized.leaf(&[0]).unwrap(),
            &KnownBits::constant(&IrBits::zero(520))
        );
        assert_eq!(oversized.leaf(&[1]).unwrap(), &clz(&arg, 0, 129));

        // A single feasible count is exact even above the enumeration budget.
        let fixed = from_bit_fn(NORMALIZE_BIT_BUDGET + 1, |i| {
            if i == NORMALIZE_BIT_BUDGET {
                Some(true)
            } else {
                None
            }
        });
        assert_eq!(
            normalize_left(&fixed, 0, fixed.bit_count(), None),
            KnownValue::Bits(fixed)
        );
        let fixed = KnownBits::constant(&IrBits::from_lsb_fn(129, |i| i == 64));
        let normalized = normalize_left(&fixed, 3, 257, Some(129));
        assert_eq!(
            normalized.leaf(&[0]).unwrap(),
            &KnownBits::constant(&IrBits::from_lsb_fn(257, |i| i == 131))
        );
        assert_eq!(
            normalized.leaf(&[1]).unwrap(),
            &KnownBits::constant(&constant_usize(64, 129))
        );
    }

    #[test]
    fn nary_add_resizes_before_negation_and_ignores_lowering_architecture() {
        for a_width in 0..=2 {
            for b_width in 0..=2 {
                for a in domain(a_width) {
                    for b in domain(b_width) {
                        for flags in 0..16 {
                            let terms = vec![
                                ExtNaryAddTerm {
                                    operand: NodeRef { index: 0 },
                                    signed: flags & 1 != 0,
                                    negated: flags & 2 != 0,
                                },
                                ExtNaryAddTerm {
                                    operand: NodeRef { index: 1 },
                                    signed: flags & 4 != 0,
                                    negated: flags & 8 != 0,
                                },
                            ];
                            for width in 0..=4 {
                                let node = node(
                                    NodePayload::ExtNaryAdd {
                                        terms: terms.clone(),
                                        arch: None,
                                    },
                                    Type::Bits(width),
                                );
                                let result = evaluate(&node, &[&a, &b]);
                                for av in concrete(&a) {
                                    for bv in concrete(&b) {
                                        let mut sum = 0i64;
                                        for ((value, in_width), term) in
                                            [(av, a_width), (bv, b_width)].into_iter().zip(&terms)
                                        {
                                            let signed = if term.signed
                                                && in_width > 0
                                                && value & (1 << (in_width - 1)) != 0
                                            {
                                                value as i64 - (1i64 << in_width)
                                            } else {
                                                value as i64
                                            };
                                            sum += if term.negated { -signed } else { signed };
                                        }
                                        let expected = IrBits::make_ubits(
                                            width,
                                            sum as u64 & ((1u64 << width) - 1),
                                        )
                                        .unwrap();
                                        assert!(result.contains(&IrValue::from_bits(&expected)));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let arg = KnownBits::constant(&IrBits::make_ubits(1, 1).unwrap());
        for arch in [
            None,
            Some(ExtNaryAddArchitecture::RippleCarry),
            Some(ExtNaryAddArchitecture::KoggeStone),
            Some(ExtNaryAddArchitecture::BrentKung),
        ] {
            let terms = vec![ExtNaryAddTerm {
                operand: NodeRef { index: 0 },
                signed: true,
                negated: true,
            }];
            let node = node(NodePayload::ExtNaryAdd { terms, arch }, Type::Bits(129));
            assert_eq!(
                evaluate(&node, &[&arg]),
                KnownValue::Bits(KnownBits::constant(&IrBits::make_ubits(129, 1).unwrap()))
            );
        }
        let empty = node(
            NodePayload::ExtNaryAdd {
                terms: vec![],
                arch: None,
            },
            Type::Bits(129),
        );
        assert_eq!(
            evaluate(&empty, &[]),
            KnownValue::Bits(KnownBits::constant(&IrBits::zero(129)))
        );
    }
}
