// SPDX-License-Identifier: Apache-2.0

//! Arithmetic lowering helpers for `ir2gate`.

use crate::aig::gate::AigBitVector;
use crate::gate_builder::GateBuilder;
use crate::gatify::ir2gate::{
    GateEnv, gatify_add_with_mapping, gatify_sext_or_truncate, gatify_zext_or_truncate,
    get_pow2_minus1_k, literal_bits_if_bits_node,
};
use crate::ir2gate_utils::{AdderMapping, array_add_with_carry_out};
use xlsynth_pir::ir;

/// Resizes a literal term to the `ExtNaryAdd` output width.
fn resize_literal_bits_for_ext_nary_add(
    bits: &xlsynth::IrBits,
    output_width: usize,
    signed: bool,
) -> xlsynth::IrBits {
    match bits.get_bit_count().cmp(&output_width) {
        std::cmp::Ordering::Less => {
            let fill_bit = signed && bits.get_bit_count() != 0 && bits.msb();
            let mut resized_bits = Vec::with_capacity(output_width);
            for i in 0..bits.get_bit_count() {
                resized_bits.push(
                    bits.get_bit(i)
                        .expect("literal bit index should be in bounds during resize"),
                );
            }
            resized_bits.resize(output_width, fill_bit);
            xlsynth::IrBits::from_lsb_is_0(&resized_bits)
        }
        std::cmp::Ordering::Equal => bits.clone(),
        std::cmp::Ordering::Greater => bits.width_slice(0, output_width as i64),
    }
}

/// Adds one literal contribution into the `ExtNaryAdd` constant accumulator.
fn accumulate_ext_nary_add_literal(
    literal_sum: &mut xlsynth::IrBits,
    term_bits: &xlsynth::IrBits,
    output_width: usize,
    signed: bool,
    negated: bool,
) {
    let resized = resize_literal_bits_for_ext_nary_add(term_bits, output_width, signed);
    let contribution = if negated { resized.negate() } else { resized };
    *literal_sum = literal_sum.add(&contribution);
}

/// Returns whether `bits` is exactly the unsigned value 1.
fn is_one(bits: &xlsynth::IrBits) -> bool {
    if bits.get_bit_count() == 0 {
        return false;
    }
    if !bits
        .get_bit(0)
        .expect("literal bit 0 should be in bounds for non-empty bits")
    {
        return false;
    }
    for i in 1..bits.get_bit_count() {
        if bits
            .get_bit(i)
            .expect("literal bit index should be in bounds during one-check")
        {
            return false;
        }
    }
    true
}

fn normalize_add_literal_rhs(
    f: &ir::Fn,
    a: ir::NodeRef,
    b: ir::NodeRef,
) -> Option<(ir::NodeRef, xlsynth::IrBits)> {
    let a_lit = literal_bits_if_bits_node(f, a);
    let b_lit = literal_bits_if_bits_node(f, b);

    match (a_lit, b_lit) {
        (None, Some(rhs_bits)) => Some((a, rhs_bits)),
        (Some(rhs_bits), None) => Some((b, rhs_bits)),
        // If both are literals, we expect folding to have handled this.
        (Some(_), Some(_)) => None,
        (None, None) => None,
    }
}

fn gatify_add_const_pow2_minus1(
    gb: &mut GateBuilder,
    lhs_bits: &AigBitVector,
    k: usize,
) -> AigBitVector {
    let bit_count = lhs_bits.get_bit_count();
    assert!(k > 0 && k <= bit_count);

    let mut sum = Vec::with_capacity(bit_count);
    // For low bits where rhs_i=1, carry recurrence is c_{i+1} = lhs_i | c_i,
    // and sum_i = !(lhs_i ^ c_i).
    let mut carry = gb.get_false();
    for i in 0..k {
        let lhs_i = *lhs_bits.get_lsb(i);
        let lhs_xor_carry = gb.add_xor_binary(lhs_i, carry);
        let sum_i = gb.add_not(lhs_xor_carry);
        sum.push(sum_i);
        carry = gb.add_or_binary(lhs_i, carry);
    }

    // For upper bits where rhs_i=0, this is an increment-by-carry chain:
    // sum_i = lhs_i ^ carry and c_{i+1} = lhs_i & carry.
    for i in k..bit_count {
        let lhs_i = *lhs_bits.get_lsb(i);
        let sum_i = gb.add_xor_binary(lhs_i, carry);
        sum.push(sum_i);
        carry = gb.add_and_binary(lhs_i, carry);
    }

    AigBitVector::from_lsb_is_index_0(&sum)
}

/// Adds one literal to an already-lowered dynamic sum.
///
/// Preserves the plain `Add` specialization for `(1<<k)-1` constants.
fn gatify_add_literal_to_dynamic_sum(
    gb: &mut GateBuilder,
    sum_bits: &AigBitVector,
    literal_bits: &xlsynth::IrBits,
    adder_mapping: AdderMapping,
) -> AigBitVector {
    assert_eq!(sum_bits.get_bit_count(), literal_bits.get_bit_count());
    if let Some(k) = get_pow2_minus1_k(literal_bits) {
        if k == 0 {
            return sum_bits.clone();
        }
        if k <= sum_bits.get_bit_count() {
            return gatify_add_const_pow2_minus1(gb, sum_bits, k);
        }
    }

    let literal_vec = gb.add_literal(literal_bits);
    let (_c_out, sum) = gatify_add_with_mapping(
        adder_mapping,
        sum_bits,
        &literal_vec,
        gb.get_false(),
        None,
        gb,
    );
    sum
}

/// Lowers a plain `add` node.
pub(super) fn gatify_add_binop(
    f: &ir::Fn,
    env: &GateEnv,
    text_id: usize,
    a: ir::NodeRef,
    b: ir::NodeRef,
    adder_mapping: AdderMapping,
    g8_builder: &mut GateBuilder,
) -> AigBitVector {
    if let Some((lhs, rhs_bits)) = normalize_add_literal_rhs(f, a, b) {
        if let Some(k) = get_pow2_minus1_k(&rhs_bits) {
            let lhs_gate_refs = env
                .get_bit_vector(lhs)
                .expect("add lhs should be present for literal-rhs rewrite");
            assert_eq!(lhs_gate_refs.get_bit_count(), rhs_bits.get_bit_count());
            return if k == 0 {
                lhs_gate_refs
            } else {
                gatify_add_const_pow2_minus1(g8_builder, &lhs_gate_refs, k)
            };
        }
    }

    let a_gate_refs = env.get_bit_vector(a).expect("add lhs should be present");
    let b_gate_refs = env.get_bit_vector(b).expect("add rhs should be present");
    assert_eq!(a_gate_refs.get_bit_count(), b_gate_refs.get_bit_count());
    let add_tag = format!("add_{}", text_id);
    let (_c_out, gates) = gatify_add_with_mapping(
        adder_mapping,
        &a_gate_refs,
        &b_gate_refs,
        g8_builder.get_false(),
        Some(&add_tag),
        g8_builder,
    );
    assert_eq!(gates.get_bit_count(), a_gate_refs.get_bit_count());
    gates
}

/// Lowers a plain `sub` node.
pub(super) fn gatify_sub_binop(
    env: &GateEnv,
    text_id: usize,
    output_bit_count: usize,
    a: ir::NodeRef,
    b: ir::NodeRef,
    adder_mapping: AdderMapping,
    g8_builder: &mut GateBuilder,
) -> AigBitVector {
    let a_gate_refs = env.get_bit_vector(a).expect("sub lhs should be present");
    let b_gate_refs = env.get_bit_vector(b).expect("sub rhs should be present");
    assert_eq!(a_gate_refs.get_bit_count(), b_gate_refs.get_bit_count());
    let b_complement = g8_builder.add_not_vec(&b_gate_refs);
    let sub_tag = format!("sub_{}", text_id);
    let (_c_out, gates) = gatify_add_with_mapping(
        adder_mapping,
        &a_gate_refs,
        &b_complement,
        g8_builder.get_true(),
        Some(&sub_tag),
        g8_builder,
    );
    assert_eq!(gates.get_bit_count(), output_bit_count);
    for (i, gate) in gates.iter_lsb_to_msb().enumerate() {
        g8_builder.add_tag(gate.node, format!("sub_{}_output_bit_{}", text_id, i));
    }
    gates
}

/// Lowers an `ext_nary_add` node.
pub(super) fn gatify_ext_nary_add(
    f: &ir::Fn,
    env: &GateEnv,
    text_id: usize,
    terms: &[ir::ExtNaryAddTerm],
    arch: Option<ir::ExtNaryAddArchitecture>,
    output_width: usize,
    default_adder_mapping: AdderMapping,
    g8_builder: &mut GateBuilder,
) -> AigBitVector {
    if output_width == 0 {
        return AigBitVector::zeros(0);
    }
    if terms.is_empty() {
        return AigBitVector::zeros(output_width);
    }

    let one_bits = xlsynth::IrBits::make_ubits(output_width, 1)
        .expect("bits[output_width]:1 should construct");
    let mut literal_sum = xlsynth::IrBits::zero(output_width);
    let mut lowered_terms: Vec<AigBitVector> = Vec::with_capacity(terms.len());
    for term in terms {
        if let Some(literal_bits) = literal_bits_if_bits_node(f, term.operand) {
            accumulate_ext_nary_add_literal(
                &mut literal_sum,
                &literal_bits,
                output_width,
                term.signed,
                term.negated,
            );
            continue;
        }

        let bits = env
            .get_bit_vector(term.operand)
            .expect("ext_nary_add operand should be present");
        let resized = if term.signed {
            gatify_sext_or_truncate(g8_builder, text_id, output_width, &bits)
        } else {
            gatify_zext_or_truncate(output_width, &bits)
        };
        if term.negated {
            lowered_terms.push(g8_builder.add_not_vec(&resized));
            literal_sum = literal_sum.add(&one_bits);
        } else {
            lowered_terms.push(resized);
        }
    }
    let carry_in = if is_one(&literal_sum) && lowered_terms.len() >= 2 {
        literal_sum = xlsynth::IrBits::zero(output_width);
        Some(g8_builder.get_true())
    } else {
        None
    };
    let selected_adder_mapping = arch
        .map(AdderMapping::from)
        .unwrap_or(default_adder_mapping);
    let selected_adder_mapping_name = match selected_adder_mapping {
        AdderMapping::RippleCarry => "ripple_carry",
        AdderMapping::BrentKung => "brent_kung",
        AdderMapping::KoggeStone => "kogge_stone",
    };
    let sum = if lowered_terms.is_empty() {
        g8_builder.add_literal(&literal_sum)
    } else if lowered_terms.len() == 1 && carry_in.is_none() {
        gatify_add_literal_to_dynamic_sum(
            g8_builder,
            &lowered_terms[0],
            &literal_sum,
            selected_adder_mapping,
        )
    } else if get_pow2_minus1_k(&literal_sum).is_some() {
        let dynamic_sum =
            array_add_with_carry_out(g8_builder, &lowered_terms, carry_in, selected_adder_mapping)
                .sum;
        gatify_add_literal_to_dynamic_sum(
            g8_builder,
            &dynamic_sum,
            &literal_sum,
            selected_adder_mapping,
        )
    } else {
        if !literal_sum.is_zero() {
            lowered_terms.push(g8_builder.add_literal(&literal_sum));
        }
        array_add_with_carry_out(g8_builder, &lowered_terms, carry_in, selected_adder_mapping).sum
    };
    for (i, gate) in sum.iter_lsb_to_msb().enumerate() {
        g8_builder.add_tag(
            gate.node,
            format!(
                "ext_nary_add_{}_{}_output_bit_{}",
                text_id, selected_adder_mapping_name, i
            ),
        );
    }
    sum
}
