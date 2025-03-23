// SPDX-License-Identifier: Apache-2.0

//! These are helper routines for the process of mapping XLS IR operations to
//! gates.
//!
//! * gatify_add_ripple_carry: instantiates a ripple-carry adder
//! * gatify_barrel_shifter: instantiates a barrel shifter (logarithmic stages)

use crate::gate::{self, AigBitVector, AigOperand, GateBuilder, ReductionKind};

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
    g8_builder: &mut gate::GateBuilder,
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
    g8_builder: &mut gate::GateBuilder,
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
    let mut gates = Vec::new();

    // Implementation note: instead of chaining all the "no prior bit" computations
    // linearly through, we do a tree reduction for each bit.

    let mut prior_bits_inverted = Vec::new();

    for i in 0..bits.get_bit_count() {
        let this_input_bit = if lsb_prio {
            bits.get_lsb(i)
        } else {
            bits.get_msb(i)
        };
        let no_prior_bit = if prior_bits_inverted.is_empty() {
            gb.get_true()
        } else {
            gb.add_and_nary(&prior_bits_inverted, ReductionKind::Tree)
        };
        let this_output_bit = gb.add_and_binary(*this_input_bit, no_prior_bit);
        gates.push(this_output_bit);

        prior_bits_inverted.push(gb.add_not(*this_input_bit));
    }
    if !lsb_prio {
        gates.reverse();
    }
    let no_prior_bit = gb.add_and_nary(&prior_bits_inverted, ReductionKind::Tree);
    gates.push(no_prior_bit);
    AigBitVector::from_lsb_is_index_0(&gates)
}

#[cfg(test)]
mod tests {
    use crate::{
        check_equivalence,
        gate::AigBitVector,
        ir2gate::{gatify_ule_via_adder, gatify_ule_via_bit_tests},
    };

    use super::*;

    use test_case::test_case;

    fn make_ripple_carry(bits: usize) -> gate::GateFn {
        let mut ripple_builder = gate::GateBuilder::new("ripple_carry".to_string(), false);
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
        let mut carry_select_builder = gate::GateBuilder::new("carry_select".to_string(), false);
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
        check_equivalence::validate_same_gate_fn(&ripple, &carry)
            .expect("carry select and ripple carry should be equivalent");
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
            let mut builder = GateBuilder::new("ule_via_adder".to_string(), false);
            let lhs = builder.add_input("lhs".to_string(), bits);
            let rhs = builder.add_input("rhs".to_string(), bits);
            let result = gatify_ule_via_adder(&mut builder, 3, &lhs, &rhs);
            builder.add_output("results".to_string(), result.into());
            builder.build()
        };
        let via_bit_tests = {
            let mut builder = GateBuilder::new("ule_via_bit_tests".to_string(), false);
            let lhs = builder.add_input("lhs".to_string(), bits);
            let rhs = builder.add_input("rhs".to_string(), bits);
            let result = gatify_ule_via_bit_tests(&mut builder, 3, &lhs, &rhs);
            builder.add_output("results".to_string(), result.into());
            builder.build()
        };
        check_equivalence::validate_same_gate_fn(&via_adder, &via_bit_tests)
            .expect("ule via adder and ule via bit tests should be equivalent");
    }
}
