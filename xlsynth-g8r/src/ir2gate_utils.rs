// SPDX-License-Identifier: Apache-2.0

//! These are helper routines for the process of mapping XLS IR operations to
//! gates.
//!
//! * gatify_add_ripple_carry: instantiates a ripple-carry adder
//! * gatify_barrel_shifter: instantiates a barrel shifter (logarithmic stages)

use crate::aig::gate::{self, AigBitVector, AigOperand};
use crate::gate_builder::GateBuilder;
use crate::gate_builder::ReductionKind;
use crate::prefix_scan_utils::{prefix_scan_exclusive, prefix_scan_inclusive};

pub use crate::prefix_scan_utils::PrefixScanStrategy;

/// Selects the adder implementation to use when lowering addition operations.
#[derive(Debug, Clone, Copy)]
pub enum AdderMapping {
    RippleCarry,
    BrentKung,
    KoggeStone,
}

impl Default for AdderMapping {
    fn default() -> Self {
        AdderMapping::BrentKung
    }
}

impl std::fmt::Display for AdderMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdderMapping::RippleCarry => write!(f, "ripple-carry"),
            AdderMapping::BrentKung => write!(f, "brent-kung"),
            AdderMapping::KoggeStone => write!(f, "kogge-stone"),
        }
    }
}

/// Emits a carry-select adder for the given inputs.
///
/// A carry-select adder specializes groups of bits on whether the carry-in to
/// the group is zero or one. Then you just incur the mux propagation time
/// across the groups if they are matched in their production delay times.
#[allow(dead_code)]
pub fn gatify_add_carry_select(
    lhs: &AigBitVector,
    rhs: &AigBitVector,
    group_partitions: &[usize],
    mut c_in: gate::AigOperand,
    tag_prefix: &str,
    g8_builder: &mut GateBuilder,
) -> (gate::AigOperand, gate::AigBitVector) {
    assert_eq!(lhs.get_bit_count(), rhs.get_bit_count());
    assert_eq!(group_partitions.iter().sum::<usize>(), lhs.get_bit_count());
    assert!(group_partitions.iter().all(|&x| x > 0));
    let mut results: Vec<gate::AigOperand> = Vec::new();
    for i in 0..group_partitions.len() {
        let start = group_partitions[..i].iter().sum::<usize>();
        let group_size = group_partitions[i];
        // Emit the specialization that assumes c_in = 0.
        let (c_out_given_0, sum_given_0) = gatify_add_ripple_carry(
            &lhs.get_lsb_slice(start, group_size),
            &rhs.get_lsb_slice(start, group_size),
            g8_builder.get_false(),
            None,
            g8_builder,
        );
        // Emit the specialization that assumes c_in = 1.
        let (c_out_given_1, sum_given_1) = gatify_add_ripple_carry(
            &lhs.get_lsb_slice(start, group_size),
            &rhs.get_lsb_slice(start, group_size),
            g8_builder.get_true(),
            None,
            g8_builder,
        );
        // Emit the mux to select between the two.
        let group_c_out = g8_builder.add_mux2(c_in, c_out_given_1, c_out_given_0);
        let group_sum = g8_builder.add_mux2_vec(&c_in, &sum_given_1, &sum_given_0);
        results.extend(group_sum.iter_lsb_to_msb());
        c_in = group_c_out;
    }

    for (i, bit) in results.iter().enumerate() {
        g8_builder.add_tag(
            bit.node,
            format!(
                "{}_carry_select_adder_{}_count_bit_{}",
                tag_prefix,
                lhs.get_bit_count(),
                lhs.get_bit_count() - i - 1
            ),
        );
    }

    (c_in, AigBitVector::from_lsb_is_index_0(&results))
}

// Returns `(carry_out, result_gates)` where `result_gates` is the same size as
// the input bit-vectors.
pub fn gatify_add_ripple_carry(
    lhs: &AigBitVector,
    rhs: &AigBitVector,
    mut c_in: gate::AigOperand,
    tag_prefix: Option<&str>,
    g8_builder: &mut GateBuilder,
) -> (gate::AigOperand, AigBitVector) {
    assert_eq!(lhs.get_bit_count(), rhs.get_bit_count());
    let mut gates = Vec::new();
    for i in 0..lhs.get_bit_count() {
        // The truth table for a adder bit is:
        //
        //  a b c | sum cout
        // --------
        //  0 0 0 | 0   0
        //  0 0 1 | 1   0
        //  0 1 0 | 1   0
        //  0 1 1 | 0   1
        //  1 0 0 | 1   0
        //  1 0 1 | 0   1
        //  1 1 0 | 0   1
        //  1 1 1 | 1   1
        //
        // sum = a ^ b ^ c_in
        // cout = (a & b) | (b & c_in) | (a & c_in)
        let lhs_i = lhs.get_lsb(i);
        let rhs_i = rhs.get_lsb(i);
        let sum = g8_builder.add_xor_nary(&[*lhs_i, *rhs_i, c_in], ReductionKind::Linear);
        let c_out_0 = g8_builder.add_and_binary(*lhs_i, *rhs_i);
        let c_out_1 = g8_builder.add_and_binary(*rhs_i, c_in);
        let c_out_2 = g8_builder.add_and_binary(*lhs_i, c_in);
        let cout = g8_builder.add_or_nary(&[c_out_0, c_out_1, c_out_2], ReductionKind::Linear);
        if let Some(tag_prefix) = tag_prefix {
            g8_builder.add_tag(
                cout.node,
                format!(
                    "{}_ripple_carry_adder_{}_count_cout_{}",
                    tag_prefix,
                    lhs.get_bit_count(),
                    i
                ),
            );
        }
        gates.push(sum);
        c_in = cout;
    }
    assert_eq!(gates.len(), lhs.get_bit_count());
    if let Some(tag_prefix) = tag_prefix {
        g8_builder.add_tag(c_in.node, format!("{}_ripple_carry_out", tag_prefix));
        for (i, gate) in gates.iter().enumerate() {
            g8_builder.add_tag(
                gate.node,
                format!(
                    "{}_ripple_carry_adder_{}_count_bit_{}",
                    tag_prefix,
                    lhs.get_bit_count(),
                    lhs.get_bit_count() - i - 1
                ),
            );
        }
    }
    (c_in, AigBitVector::from_lsb_is_index_0(&gates))
}

pub fn gatify_add_kogge_stone(
    lhs: &AigBitVector,
    rhs: &AigBitVector,
    c_in: gate::AigOperand,
    tag_prefix: Option<&str>,
    g8_builder: &mut GateBuilder,
) -> (gate::AigOperand, AigBitVector) {
    assert_eq!(lhs.get_bit_count(), rhs.get_bit_count());
    let bits = lhs.get_bit_count();
    let xor_bits: Vec<AigOperand> = (0..bits)
        .map(|i| g8_builder.add_xor_binary(*lhs.get_lsb(i), *rhs.get_lsb(i)))
        .collect();
    let pg_inputs: Vec<PrefixPg> = (0..bits)
        .map(|i| PrefixPg {
            p: xor_bits[i],
            g: g8_builder.add_and_binary(*lhs.get_lsb(i), *rhs.get_lsb(i)),
        })
        .collect();
    let identity = PrefixPg {
        p: g8_builder.get_true(),
        g: g8_builder.get_false(),
    };
    let pg_scan = prefix_scan_inclusive(
        g8_builder,
        &pg_inputs,
        PrefixScanStrategy::KoggeStone,
        identity,
        combine_prefix_pg,
    );
    let mut carries = Vec::with_capacity(bits + 1);
    carries.push(c_in);
    for i in 0..bits {
        let and = g8_builder.add_and_binary(pg_scan[i].p, c_in);
        let carry = g8_builder.add_or_binary(pg_scan[i].g, and);
        carries.push(carry);
    }
    let mut sum = Vec::with_capacity(bits);
    for i in 0..bits {
        let tmp = xor_bits[i];
        let s = g8_builder.add_xor_binary(tmp, carries[i]);
        if let Some(prefix) = tag_prefix {
            g8_builder.add_tag(
                s.node,
                format!(
                    "{}_kogge_stone_adder_{}_count_bit_{}",
                    prefix,
                    bits,
                    bits - i - 1
                ),
            );
        }
        sum.push(s);
    }
    let c_out = carries[bits];
    if let Some(prefix) = tag_prefix {
        g8_builder.add_tag(c_out.node, format!("{}_kogge_stone_out", prefix));
    }
    (c_out, AigBitVector::from_lsb_is_index_0(&sum))
}

pub fn gatify_add_brent_kung(
    lhs: &AigBitVector,
    rhs: &AigBitVector,
    c_in: gate::AigOperand,
    tag_prefix: Option<&str>,
    g8_builder: &mut GateBuilder,
) -> (gate::AigOperand, AigBitVector) {
    assert_eq!(lhs.get_bit_count(), rhs.get_bit_count());
    let bits = lhs.get_bit_count();
    let xor_bits: Vec<AigOperand> = (0..bits)
        .map(|i| g8_builder.add_xor_binary(*lhs.get_lsb(i), *rhs.get_lsb(i)))
        .collect();
    let pg_inputs: Vec<PrefixPg> = (0..bits)
        .map(|i| PrefixPg {
            p: xor_bits[i],
            g: g8_builder.add_and_binary(*lhs.get_lsb(i), *rhs.get_lsb(i)),
        })
        .collect();
    let identity = PrefixPg {
        p: g8_builder.get_true(),
        g: g8_builder.get_false(),
    };
    let pg_scan = prefix_scan_inclusive(
        g8_builder,
        &pg_inputs,
        PrefixScanStrategy::BrentKung,
        identity,
        combine_prefix_pg,
    );
    let mut carries = Vec::with_capacity(bits + 1);
    carries.push(c_in);
    for i in 0..bits {
        let and = g8_builder.add_and_binary(pg_scan[i].p, c_in);
        let carry = g8_builder.add_or_binary(pg_scan[i].g, and);
        carries.push(carry);
    }
    let mut sum = Vec::with_capacity(bits);
    for i in 0..bits {
        let tmp = xor_bits[i];
        let s = g8_builder.add_xor_binary(tmp, carries[i]);
        if let Some(prefix) = tag_prefix {
            g8_builder.add_tag(
                s.node,
                format!(
                    "{}_brent_kung_adder_{}_count_bit_{}",
                    prefix,
                    bits,
                    bits - i - 1
                ),
            );
        }
        sum.push(s);
    }
    let c_out = carries[bits];
    if let Some(prefix) = tag_prefix {
        g8_builder.add_tag(c_out.node, format!("{}_brent_kung_out", prefix));
    }
    (c_out, AigBitVector::from_lsb_is_index_0(&sum))
}

#[derive(Debug, Clone)]
pub struct ArrayAddResult {
    pub sum: AigBitVector,
    pub carry_out: AigOperand,
}

fn add_with_mapping(
    adder_mapping: AdderMapping,
    lhs: &AigBitVector,
    rhs: &AigBitVector,
    c_in: AigOperand,
    gb: &mut GateBuilder,
) -> (AigOperand, AigBitVector) {
    match adder_mapping {
        AdderMapping::RippleCarry => gatify_add_ripple_carry(lhs, rhs, c_in, None, gb),
        AdderMapping::BrentKung => gatify_add_brent_kung(lhs, rhs, c_in, None, gb),
        AdderMapping::KoggeStone => gatify_add_kogge_stone(lhs, rhs, c_in, None, gb),
    }
}

fn widen_with_zero_msb(bit_vector: &AigBitVector, gb: &mut GateBuilder) -> AigBitVector {
    let mut operands: Vec<AigOperand> = bit_vector.iter_lsb_to_msb().cloned().collect();
    operands.push(gb.get_false());
    AigBitVector::from_lsb_is_index_0(&operands)
}

/// Returns `(sum, carry)` where both are the same width as the inputs.
///
/// `carry` is already shifted left by one (i.e. `carry[0] = 0` and `carry[i+1]`
/// is the carry bit resulting from bit position `i`).
fn compress_3_to_2(
    gb: &mut GateBuilder,
    a: &AigBitVector,
    b: &AigBitVector,
    c: &AigBitVector,
) -> (AigBitVector, AigBitVector) {
    assert_eq!(a.get_bit_count(), b.get_bit_count());
    assert_eq!(a.get_bit_count(), c.get_bit_count());
    let bit_count = a.get_bit_count();
    assert!(bit_count > 0, "cannot compress 0-bit vectors");

    let mut sum_bits: Vec<AigOperand> = Vec::with_capacity(bit_count);
    let mut carry_bits: Vec<AigOperand> = Vec::with_capacity(bit_count);
    carry_bits.push(gb.get_false());
    for i in 0..bit_count {
        let fa = gb.add_full_adder(*a.get_lsb(i), *b.get_lsb(i), *c.get_lsb(i));
        sum_bits.push(fa.sum);
        if i + 1 < bit_count {
            carry_bits.push(fa.carry);
        }
    }
    assert_eq!(sum_bits.len(), bit_count);
    assert_eq!(carry_bits.len(), bit_count);
    (
        AigBitVector::from_lsb_is_index_0(&sum_bits),
        AigBitVector::from_lsb_is_index_0(&carry_bits),
    )
}

fn reduce_operands_to_two(
    gb: &mut GateBuilder,
    mut operands: Vec<AigBitVector>,
) -> (AigBitVector, AigBitVector) {
    assert!(!operands.is_empty(), "expected at least one operand");
    let bit_count = operands[0].get_bit_count();
    assert!(bit_count > 0, "cannot reduce 0-bit vectors");
    for op in operands.iter() {
        assert_eq!(op.get_bit_count(), bit_count, "operand width mismatch");
    }

    if operands.len() == 1 {
        return (operands[0].clone(), AigBitVector::zeros(bit_count));
    }

    while operands.len() > 2 {
        let mut next: Vec<AigBitVector> = Vec::new();
        for chunk in operands.chunks(3) {
            match chunk {
                [a, b, c] => {
                    let (sum, carry) = compress_3_to_2(gb, a, b, c);
                    next.push(sum);
                    next.push(carry);
                }
                [a, b] => {
                    next.push(a.clone());
                    next.push(b.clone());
                }
                [a] => {
                    next.push(a.clone());
                }
                _ => unreachable!("chunks(3) gives 1..=3 items"),
            }
        }
        operands = next;
    }
    assert_eq!(operands.len(), 2);
    (operands[0].clone(), operands[1].clone())
}

/// Adds an array of `bits[N]` vectors, optionally with a 1-bit carry-in.
///
/// Returns `sum: bits[N]` and `carry_out`, where `carry_out` is the bit `N` of
/// the full sum (i.e. the carry out of the most significant bit).
pub fn array_add_with_carry_out(
    gb: &mut GateBuilder,
    operands: &[AigBitVector],
    carry_in: Option<AigOperand>,
    adder_mapping: AdderMapping,
) -> ArrayAddResult {
    assert!(
        !operands.is_empty(),
        "array_add expects at least one operand"
    );
    let bit_count = operands[0].get_bit_count();
    assert!(bit_count > 0, "array_add does not support 0-bit operands");
    for op in operands.iter() {
        assert_eq!(op.get_bit_count(), bit_count, "operand width mismatch");
    }

    // We only care about:
    // - the low `bit_count` result bits, and
    // - the 1-bit carry-out at position `bit_count`.
    // So we compute everything modulo 2^(bit_count+1).
    let ext_width = bit_count + 1;
    let ext_ops: Vec<AigBitVector> = operands
        .iter()
        .map(|op| widen_with_zero_msb(op, gb))
        .collect();
    for op in ext_ops.iter() {
        assert_eq!(op.get_bit_count(), ext_width);
    }

    let (a, b) = reduce_operands_to_two(gb, ext_ops);
    let c_in = carry_in.unwrap_or_else(|| gb.get_false());
    let (_ignored, sum_ext) = add_with_mapping(adder_mapping, &a, &b, c_in, gb);
    assert_eq!(sum_ext.get_bit_count(), ext_width);

    let sum = sum_ext.get_lsb_slice(0, bit_count);
    let carry_out = *sum_ext.get_lsb(bit_count);
    ArrayAddResult { sum, carry_out }
}

/// Convenience wrapper that returns only the `bits[N]` sum.
pub fn array_add(
    gb: &mut GateBuilder,
    operands: Vec<AigBitVector>,
    carry_in: Option<AigOperand>,
) -> AigBitVector {
    array_add_with_carry_out(gb, &operands, carry_in, AdderMapping::default()).sum
}

#[derive(Debug, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
}

// Implements a stage-based barrel shifter (with logarithmic stages) using 2:1
// muxes.
fn gatify_barrel_shifter_internal(
    arg_gates: &AigBitVector,
    amount_gates: &AigBitVector,
    direction: Direction,
    tag_prefix: &str,
    g8_builder: &mut GateBuilder,
) -> AigBitVector {
    // Number of bits in the input vector.
    let bit_count = arg_gates.get_bit_count();
    // Start with the input vector.
    let mut current: Vec<AigOperand> = arg_gates.iter_lsb_to_msb().cloned().collect::<Vec<_>>();
    // Each bit in the shift amount (assumed little-endian) represents a stage.
    for stage in 0..amount_gates.get_bit_count() {
        let shift = 1 << stage; // 2^stage shift amount for this stage.
        let control = amount_gates.get_lsb(stage);
        let mut next_stage = Vec::with_capacity(bit_count);
        match direction {
            Direction::Right => {
                // For a right shift:
                // If control == 1 then output bit at position j becomes current[j + shift] (if
                // within range), or false otherwise; if control == 0 then
                // remain unchanged.
                for j in 0..bit_count {
                    let candidate = if j + shift < bit_count {
                        current[j + shift]
                    } else {
                        // Out of range: use the false literal.
                        g8_builder.get_false()
                    };
                    // Mux: if control is true choose candidate (shifted), else keep current[j].
                    let mux = g8_builder.add_mux2(*control, candidate, current[j]);
                    next_stage.push(mux);
                }
            }
            Direction::Left => {
                // For a left shift:
                // If control == 1 then output bit at position j becomes current[j - shift] (if
                // possible), or false if j < shift; if control == 0 then remain
                // unchanged.
                for j in 0..bit_count {
                    let candidate = if j >= shift {
                        current[j - shift]
                    } else {
                        g8_builder.get_false()
                    };
                    // Mux: if control is true choose candidate (shifted), else keep current[j].
                    let mux = g8_builder.add_mux2(*control, candidate, current[j]);
                    next_stage.push(mux);
                }
            }
        }
        current = next_stage;
    }
    for gate in current.iter() {
        let direction_str = if direction == Direction::Left {
            "left"
        } else {
            "right"
        };
        g8_builder.add_tag(
            gate.node,
            format!(
                "{}_barrel_shifter_{}_{}_count_output",
                tag_prefix, direction_str, bit_count
            ),
        )
    }
    AigBitVector::from_lsb_is_index_0(&current)
}

pub fn gatify_barrel_shifter(
    arg_gates: &AigBitVector,
    amount_gates: &AigBitVector,
    direction: Direction,
    tag_prefix: &str,
    gb: &mut GateBuilder,
) -> AigBitVector {
    let natural_amount_bits = (arg_gates.get_bit_count() as f64).log2().ceil() as usize;
    let amount_can_be_oversize = amount_gates.get_bit_count() > natural_amount_bits;
    // The "amount" value is limited in what it can realistically cause to happen by
    // the number of bits in the arg value. If the number is larger than that,
    // we can just give back zero.
    if amount_can_be_oversize {
        let gate::Split { msbs, lsbs } =
            amount_gates.get_lsb_partition(natural_amount_bits as usize);
        assert!(
            msbs.get_bit_count() != 0,
            "since amount can be oversize high bits should be non-empty"
        );
        let overlarge = gb.add_nez(&msbs, ReductionKind::Tree);
        let normal = gatify_barrel_shifter_internal(arg_gates, &lsbs, direction, tag_prefix, gb);
        let zero = AigBitVector::zeros(arg_gates.get_bit_count());
        gb.add_mux2_vec(&overlarge, &zero, &normal)
    } else {
        gatify_barrel_shifter_internal(arg_gates, amount_gates, direction, tag_prefix, gb)
    }
}

/// Emits a one-hot-select gate pattern into the gate builder.
///
/// `selector_bits` should have one bit for every case, N bits total.
/// `cases` should have N bit vectors.
///
/// This does `bitwise_or_reduce([masked_case for case in cases])` where
/// `masked_case` is `case & selector_bits[i]`.
pub fn gatify_one_hot_select(
    gb: &mut GateBuilder,
    selector_bits: &AigBitVector,
    cases: &[AigBitVector],
) -> AigBitVector {
    assert_eq!(selector_bits.get_bit_count(), cases.len());
    let case_bit_count = cases[0].get_bit_count();
    for case in cases.iter() {
        assert_eq!(
            case.get_bit_count(),
            case_bit_count,
            "all cases must have the same bit count"
        );
    }
    let mut masked: Vec<AigBitVector> = Vec::new();
    for i in 0..selector_bits.get_bit_count() {
        let case = cases[i].clone();
        let selector_bit = selector_bits.get_lsb(i);
        let replicated = gb.replicate(*selector_bit, case_bit_count);
        masked.push(gb.add_and_vec(&replicated, &case));
    }
    // Or-reduce the cases bitwise.
    let result = gb.add_or_vec_nary(&masked, ReductionKind::Tree);
    result
}

pub fn gatify_one_hot(gb: &mut GateBuilder, bits: &AigBitVector, lsb_prio: bool) -> AigBitVector {
    gatify_one_hot_with_nonzero_flag(gb, bits, lsb_prio, /* value_cannot_be_zero= */ false)
}

pub fn gatify_one_hot_for_depth(
    gb: &mut GateBuilder,
    bits: &AigBitVector,
    lsb_prio: bool,
) -> AigBitVector {
    gatify_one_hot_with_nonzero_flag_for_depth(
        gb, bits, lsb_prio, /* value_cannot_be_zero= */ false,
    )
}

pub fn gatify_one_hot_for_area(
    gb: &mut GateBuilder,
    bits: &AigBitVector,
    lsb_prio: bool,
) -> AigBitVector {
    gatify_one_hot_with_nonzero_flag_for_area(
        gb, bits, lsb_prio, /* value_cannot_be_zero= */ false,
    )
}

pub fn gatify_one_hot_with_nonzero_flag(
    gb: &mut GateBuilder,
    bits: &AigBitVector,
    lsb_prio: bool,
    value_cannot_be_zero: bool,
) -> AigBitVector {
    gatify_one_hot_with_nonzero_flag_prefix_strategy(
        gb,
        bits,
        lsb_prio,
        value_cannot_be_zero,
        PrefixScanStrategy::BrentKung,
    )
}

pub fn gatify_one_hot_with_nonzero_flag_prefix_strategy(
    gb: &mut GateBuilder,
    bits: &AigBitVector,
    lsb_prio: bool,
    value_cannot_be_zero: bool,
    prefix_strategy: PrefixScanStrategy,
) -> AigBitVector {
    let mut ordered_bits: Vec<AigOperand> = Vec::with_capacity(bits.get_bit_count());
    for i in 0..bits.get_bit_count() {
        let this_input_bit = if lsb_prio {
            bits.get_lsb(i)
        } else {
            bits.get_msb(i)
        };
        ordered_bits.push(*this_input_bit);
    }

    let mut inverted: Vec<AigOperand> = Vec::with_capacity(bits.get_bit_count());
    for bit in ordered_bits.iter() {
        inverted.push(gb.add_not(*bit));
    }

    let identity = gb.get_true();
    let inclusive = prefix_scan_inclusive(
        gb,
        &inverted,
        prefix_strategy,
        identity,
        |builder, lhs, rhs| builder.add_and_binary(lhs, rhs),
    );
    let exclusive = prefix_scan_exclusive(&inclusive, identity);

    let mut gates: Vec<AigOperand> = Vec::with_capacity(bits.get_bit_count() + 1);
    for (i, bit) in ordered_bits.iter().enumerate() {
        let no_prior_bit = exclusive[i];
        let this_output_bit = if no_prior_bit == gb.get_true() {
            *bit
        } else {
            gb.add_and_binary(*bit, no_prior_bit)
        };
        gates.push(this_output_bit);
    }

    if !lsb_prio {
        gates.reverse();
    }

    if value_cannot_be_zero {
        // If the input value is provably nonzero, then "none of the input bits were
        // set" is provably false. Emit it as a literal zero.
        gates.push(gb.get_false());
    } else {
        let no_prior_bit = if inclusive.is_empty() {
            gb.get_true()
        } else {
            *inclusive
                .last()
                .expect("inclusive scan should not be empty")
        };
        gates.push(no_prior_bit);
    }

    AigBitVector::from_lsb_is_index_0(&gates)
}

pub fn gatify_one_hot_with_nonzero_flag_for_area(
    gb: &mut GateBuilder,
    bits: &AigBitVector,
    lsb_prio: bool,
    value_cannot_be_zero: bool,
) -> AigBitVector {
    gatify_one_hot_with_nonzero_flag_prefix_strategy(
        gb,
        bits,
        lsb_prio,
        value_cannot_be_zero,
        PrefixScanStrategy::Linear,
    )
}

pub fn gatify_one_hot_with_nonzero_flag_for_depth(
    gb: &mut GateBuilder,
    bits: &AigBitVector,
    lsb_prio: bool,
    value_cannot_be_zero: bool,
) -> AigBitVector {
    gatify_one_hot_with_nonzero_flag_prefix_strategy(
        gb,
        bits,
        lsb_prio,
        value_cannot_be_zero,
        PrefixScanStrategy::KoggeStone,
    )
}

#[derive(Clone, Copy)]
struct PrefixPg {
    p: AigOperand,
    g: AigOperand,
}

fn combine_prefix_pg(gb: &mut GateBuilder, lhs: PrefixPg, rhs: PrefixPg) -> PrefixPg {
    let and = gb.add_and_binary(rhs.p, lhs.g);
    let g = gb.add_or_binary(rhs.g, and);
    let p = gb.add_and_binary(rhs.p, lhs.p);
    PrefixPg { p, g }
}

#[cfg(test)]
mod tests {
    use crate::{
        aig::gate::AigBitVector,
        aig_serdes::ir2gate::{gatify_ule_via_adder, gatify_ule_via_bit_tests},
        aig_sim::gate_sim,
        check_equivalence,
        gate_builder::GateBuilderOptions,
    };

    use super::*;

    use test_case::test_case;
    use xlsynth::IrBits;

    fn eval_array_add_case(
        bit_count: usize,
        operand_values: &[u64],
        carry_in: Option<bool>,
    ) -> (u64, bool) {
        let mut gb = GateBuilder::new("array_add_test".to_string(), GateBuilderOptions::no_opt());
        let mut operands = Vec::new();
        for (i, _) in operand_values.iter().enumerate() {
            operands.push(gb.add_input(format!("op_{}", i), bit_count));
        }
        let carry_in_op = carry_in.map(|_| gb.add_input("carry_in".to_string(), 1));
        let carry_in_bit = carry_in_op.as_ref().map(|v| *v.get_lsb(0));

        let res =
            array_add_with_carry_out(&mut gb, &operands, carry_in_bit, AdderMapping::default());
        gb.add_output("sum".to_string(), res.sum.clone());
        gb.add_output(
            "carry_out".to_string(),
            AigBitVector::from_bit(res.carry_out),
        );
        let gate_fn = gb.build();

        let mut inputs = Vec::new();
        for &v in operand_values {
            inputs.push(IrBits::make_ubits(bit_count, v).unwrap());
        }
        if let Some(ci) = carry_in {
            inputs.push(IrBits::bool(ci));
        }
        let got = gate_sim::eval(&gate_fn, &inputs, gate_sim::Collect::None);
        assert_eq!(got.outputs.len(), 2);
        let got_sum = got.outputs[0].to_u64().unwrap();
        let got_carry_out = got.outputs[1].get_bit(0).unwrap();
        (got_sum, got_carry_out)
    }

    #[test]
    fn test_array_add_exhaustive_3bit_3ops_no_carry_in() {
        let bit_count = 3usize;
        for a in 0u64..(1u64 << bit_count) {
            for b in 0u64..(1u64 << bit_count) {
                for c in 0u64..(1u64 << bit_count) {
                    let (got_sum, got_c_out) = eval_array_add_case(bit_count, &[a, b, c], None);
                    let total = a + b + c;
                    let want_sum = total & ((1u64 << bit_count) - 1);
                    let want_c_out = ((total >> bit_count) & 1) != 0;
                    assert_eq!(
                        (got_sum, got_c_out),
                        (want_sum, want_c_out),
                        "a={} b={} c={}",
                        a,
                        b,
                        c
                    );
                }
            }
        }
    }

    #[test]
    fn test_array_add_exhaustive_3bit_3ops_with_carry_in() {
        let bit_count = 3usize;
        for a in 0u64..(1u64 << bit_count) {
            for b in 0u64..(1u64 << bit_count) {
                for c in 0u64..(1u64 << bit_count) {
                    for carry_in in [false, true] {
                        let (got_sum, got_c_out) =
                            eval_array_add_case(bit_count, &[a, b, c], Some(carry_in));
                        let total = a + b + c + u64::from(carry_in);
                        let want_sum = total & ((1u64 << bit_count) - 1);
                        let want_c_out = ((total >> bit_count) & 1) != 0;
                        assert_eq!(
                            (got_sum, got_c_out),
                            (want_sum, want_c_out),
                            "a={} b={} c={} carry_in={}",
                            a,
                            b,
                            c,
                            carry_in
                        );
                    }
                }
            }
        }
    }

    fn make_ripple_carry(bits: usize) -> gate::GateFn {
        let mut ripple_builder =
            GateBuilder::new("ripple_carry".to_string(), GateBuilderOptions::no_opt());
        let lhs = ripple_builder.add_input("lhs".to_string(), bits);
        let rhs = ripple_builder.add_input("rhs".to_string(), bits);
        let c_in = ripple_builder
            .add_input("c_in".to_string(), 1)
            .get_lsb(0)
            .clone();
        let (c_out, results) =
            gatify_add_ripple_carry(&lhs, &rhs, c_in, Some("ripple_carry"), &mut ripple_builder);
        ripple_builder.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
        ripple_builder.add_output("results".to_string(), results);
        ripple_builder.build()
    }

    fn make_carry_select(bits: usize, partitions: &[usize]) -> gate::GateFn {
        let mut carry_select_builder =
            GateBuilder::new("carry_select".to_string(), GateBuilderOptions::no_opt());
        let lhs = carry_select_builder.add_input("lhs".to_string(), bits);
        let rhs = carry_select_builder.add_input("rhs".to_string(), bits);
        let c_in = carry_select_builder
            .add_input("c_in".to_string(), 1)
            .get_lsb(0)
            .clone();
        let (c_out, results) = gatify_add_carry_select(
            &lhs,
            &rhs,
            partitions,
            c_in.into(),
            "carry_select",
            &mut carry_select_builder,
        );
        carry_select_builder.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
        carry_select_builder.add_output("results".to_string(), results);
        carry_select_builder.build()
    }

    #[test]
    fn test_gatify_one_hot_nonzero_sets_is_zero_bit_literal_false() {
        let mut builder =
            GateBuilder::new("one_hot_nonzero".to_string(), GateBuilderOptions::no_opt());
        let arg: AigBitVector = builder.add_input("arg".to_string(), 8);

        let one_hot = gatify_one_hot_with_nonzero_flag(
            &mut builder,
            &arg,
            /* lsb_prio= */ true,
            /* value_cannot_be_zero= */ true,
        );

        // The output is N+1 bits; the final bit indicates "input was zero".
        // When the input is provably nonzero, this bit must be a literal 0.
        assert_eq!(one_hot.get_bit_count(), 9);
        assert_eq!(*one_hot.get_lsb(8), builder.get_false());
    }

    fn make_kogge_stone(bits: usize) -> gate::GateFn {
        let mut builder = GateBuilder::new("kogge_stone".to_string(), GateBuilderOptions::no_opt());
        let lhs = builder.add_input("lhs".to_string(), bits);
        let rhs = builder.add_input("rhs".to_string(), bits);
        let c_in = builder.add_input("c_in".to_string(), 1).get_lsb(0).clone();
        let (c_out, results) = gatify_add_kogge_stone(&lhs, &rhs, c_in, Some("ks"), &mut builder);
        builder.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
        builder.add_output("results".to_string(), results);
        builder.build()
    }

    fn make_brent_kung(bits: usize) -> gate::GateFn {
        let mut builder = GateBuilder::new("brent_kung".to_string(), GateBuilderOptions::no_opt());
        let lhs = builder.add_input("lhs".to_string(), bits);
        let rhs = builder.add_input("rhs".to_string(), bits);
        let c_in = builder.add_input("c_in".to_string(), 1).get_lsb(0).clone();
        let (c_out, results) = gatify_add_brent_kung(&lhs, &rhs, c_in, Some("bk"), &mut builder);
        builder.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
        builder.add_output("results".to_string(), results);
        builder.build()
    }

    #[test_case(1, &[1]; "1-bit 1-partition")]
    #[test_case(2, &[1,1]; "2-bit 2-partition")]
    #[test_case(3, &[1, 2]; "3-bit 2-partition")]
    #[test_case(4, &[2, 2]; "4-bit 2-partition")]
    #[test_case(4, &[1, 1, 2]; "4-bit 3-partition")]
    #[test_case(8, &[2, 3, 3]; "8-bit 3-partition")]
    #[test_case(8, &[4, 4]; "8-bit 2-partition")]
    fn test_gatify_add_carry_select(bits: usize, carry_select_partitions: &[usize]) {
        let ripple = make_ripple_carry(bits);
        let carry = make_carry_select(bits, carry_select_partitions);
        check_equivalence::prove_same_gate_fn_via_ir(&ripple, &carry)
            .expect("carry select and ripple carry should be equivalent");
    }

    #[test]
    fn test_gatify_add_kogge_stone_1_to_16() {
        for bits in 1..=16 {
            let ripple = make_ripple_carry(bits);
            let ks = make_kogge_stone(bits);
            check_equivalence::prove_same_gate_fn_via_ir(&ripple, &ks).expect(&format!(
                "kogge stone and ripple carry should be equivalent for {} bits",
                bits
            ));
        }
    }

    #[test]
    fn test_gatify_add_brent_kung_1_to_16() {
        for bits in 1..=16 {
            let ripple = make_ripple_carry(bits);
            let bk = make_brent_kung(bits);
            check_equivalence::prove_same_gate_fn_via_ir(&ripple, &bk).expect(&format!(
                "brent kung and ripple carry should be equivalent for {} bits",
                bits
            ));
        }
    }

    #[test_case(1)]
    #[test_case(2)]
    #[test_case(3)]
    #[test_case(4)]
    #[test_case(5)]
    #[test_case(6)]
    #[test_case(7)]
    #[test_case(8)]
    fn test_gatify_ule(bits: usize) {
        let via_adder = {
            let mut builder =
                GateBuilder::new("ule_via_adder".to_string(), GateBuilderOptions::no_opt());
            let lhs = builder.add_input("lhs".to_string(), bits);
            let rhs = builder.add_input("rhs".to_string(), bits);
            let result = gatify_ule_via_adder(&mut builder, 3, &lhs, &rhs);
            builder.add_output("results".to_string(), result.into());
            builder.build()
        };
        let via_bit_tests = {
            let mut builder = GateBuilder::new(
                "ule_via_bit_tests".to_string(),
                GateBuilderOptions::no_opt(),
            );
            let lhs = builder.add_input("lhs".to_string(), bits);
            let rhs = builder.add_input("rhs".to_string(), bits);
            let result = gatify_ule_via_bit_tests(&mut builder, 3, &lhs, &rhs);
            builder.add_output("results".to_string(), result.into());
            builder.build()
        };
        check_equivalence::prove_same_gate_fn_via_ir(&via_adder, &via_bit_tests)
            .expect("ule via adder and ule via bit tests should be equivalent");
    }
}
