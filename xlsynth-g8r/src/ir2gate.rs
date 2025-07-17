// SPDX-License-Identifier: Apache-2.0

//! Functionality for converting an IR function into a gate function via
//! `gatify`.

use crate::check_equivalence;
use crate::gate::{AigBitVector, AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::xls_ir::ir::{self, ParamId, StartAndLimit};
use crate::xls_ir::ir_utils;
use std::collections::HashMap;

use crate::ir2gate_utils::{
    AdderMapping, Direction, gatify_add_brent_kung, gatify_add_kogge_stone,
    gatify_add_ripple_carry, gatify_barrel_shifter, gatify_one_hot, gatify_one_hot_select,
};

use crate::gate_builder::ReductionKind;

#[derive(Debug)]
enum GateOrVec {
    Gate(AigOperand),
    BitVector(AigBitVector),
}

struct GateEnv {
    ir_to_g8: HashMap<ir::NodeRef, GateOrVec>,
}

impl GateEnv {
    fn new() -> Self {
        Self {
            ir_to_g8: HashMap::new(),
        }
    }

    pub fn contains(&self, ir_node_ref: ir::NodeRef) -> bool {
        self.ir_to_g8.contains_key(&ir_node_ref)
    }

    pub fn add(&mut self, ir_node_ref: ir::NodeRef, gate_or_vec: GateOrVec) {
        log::debug!(
            "add; ir_node_ref: {:?}; gate_or_vec: {:?}",
            ir_node_ref,
            gate_or_vec
        );
        match self.ir_to_g8.insert(ir_node_ref, gate_or_vec) {
            Some(_) => {
                panic!("Duplicate gate reference for IR node {:?}", ir_node_ref);
            }
            None => {}
        }
    }

    pub fn get_bit_vector(&self, ir_node_ref: ir::NodeRef) -> Result<AigBitVector, String> {
        match self.ir_to_g8.get(&ir_node_ref) {
            Some(GateOrVec::BitVector(bv)) => Ok(bv.clone()),
            Some(GateOrVec::Gate(gate_ref)) => Ok(AigBitVector::from_bit(*gate_ref)),
            None => Err(format!(
                "No gate data present for IR node {:?}",
                ir_node_ref
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Signedness {
    Unsigned,
    Signed,
}

fn gatify_priority_sel(
    gb: &mut GateBuilder,
    output_bit_count: usize,
    selector_bits: AigBitVector,
    cases: &[AigBitVector],
    default_bits: Option<AigBitVector>,
) -> AigBitVector {
    assert_eq!(
        selector_bits.get_bit_count(),
        cases.len(),
        "priority select selector bit width {} does not match number of cases {}",
        selector_bits.get_bit_count(),
        cases.len()
    );

    let mut masked_cases = vec![];
    // As we process cases we track whether any prior case had been selected.
    let mut any_prior_selected = gb.get_false();
    for (i, case_bits) in cases.iter().enumerate() {
        assert_eq!(
            case_bits.get_bit_count(),
            output_bit_count,
            "all cases of the priority select must have the same bit count which is the same as the output bit count"
        );
        let this_wants_selected = selector_bits.get_lsb(i).clone();
        let no_prior_selected = gb.add_not(any_prior_selected);
        let this_selected = gb.add_and_binary(this_wants_selected, no_prior_selected);
        any_prior_selected = gb.add_or_binary(any_prior_selected, this_selected);

        let mask = gb.replicate(this_selected, output_bit_count);
        let masked = gb.add_and_vec(&mask, &case_bits);
        masked_cases.push(masked);
    }

    if let Some(default_bits) = default_bits {
        let no_prior_selected = gb.add_not(any_prior_selected);
        let mask = gb.replicate(no_prior_selected, output_bit_count);
        let masked = gb.add_and_vec(&mask, &default_bits);
        masked_cases.push(masked);
    }
    gb.add_or_vec_nary(&masked_cases, ReductionKind::Tree)
}

fn gatify_array_index(
    gb: &mut GateBuilder,
    array_ty: &ir::ArrayTypeData,
    array_bits: &AigBitVector,
    index_bits: &AigBitVector,
    assumed_in_bounds: bool,
) -> AigBitVector {
    let element_bit_count = array_ty.element_type.bit_count();

    if assumed_in_bounds {
        let index_decoded = gatify_decode(gb, array_ty.element_count, index_bits);
        let mut cases = Vec::new();
        for i in (0..array_ty.element_count).rev() {
            let case_bits = array_bits.get_lsb_slice(i * element_bit_count, element_bit_count);
            cases.push(case_bits);
        }
        return gatify_one_hot_select(gb, &index_decoded, &cases);
    }

    let array_element_count = array_ty.element_count;
    let index_decoded = gatify_decode(gb, array_element_count, index_bits);
    let oob = gb.add_ez(&index_decoded, ReductionKind::Tree);
    let one_hot_selector = AigBitVector::concat(oob.into(), index_decoded);

    // An array index selection is effectively a one hot selection of the elements
    // into a single element result.
    let mut cases = Vec::new();
    for i in (0..array_element_count).rev() {
        let case_bits = array_bits.get_lsb_slice(i * element_bit_count, element_bit_count);
        cases.push(case_bits);
    }
    cases.push(cases.last().unwrap().clone());
    let result = gatify_one_hot_select(gb, &one_hot_selector, &cases);
    result
}

fn gatify_array_update(
    gb: &mut GateBuilder,
    array_ty: &ir::ArrayTypeData,
    array_bits: &AigBitVector,
    value_bits: &AigBitVector,
    index_bits: &[AigBitVector],
) -> AigBitVector {
    assert!(!index_bits.is_empty());

    let element_bit_count = array_ty.element_type.bit_count();
    let index_decoded = gatify_decode(gb, array_ty.element_count, &index_bits[0]);
    let mut updated_elems: Vec<AigBitVector> = Vec::new();

    for i in 0..array_ty.element_count {
        let orig_elem = array_bits.get_lsb_slice(i * element_bit_count, element_bit_count);
        let updated_value = if index_bits.len() == 1 {
            value_bits.clone()
        } else {
            let next_ty = match array_ty.element_type.as_ref() {
                ir::Type::Array(ty) => ty,
                other => panic!(
                    "Expected array type for multidimensional array update, got {:?}",
                    other
                ),
            };
            gatify_array_update(gb, next_ty, &orig_elem, value_bits, &index_bits[1..])
        };
        let selector = index_decoded.get_lsb(i);
        let updated = gb.add_mux2_vec(selector, &updated_value, &orig_elem);
        updated_elems.push(updated);
    }

    let mut lsb_to_msb = Vec::new();
    for elem_bits in updated_elems.into_iter().rev() {
        lsb_to_msb.extend(elem_bits.iter_lsb_to_msb().cloned());
    }
    AigBitVector::from_lsb_is_index_0(&lsb_to_msb)
}

fn gatify_sel(
    gb: &mut GateBuilder,
    selector_bits: &AigBitVector,
    cases: &[AigBitVector],
    default_bits: Option<AigBitVector>,
) -> AigBitVector {
    let case_count = cases.len();
    let index_decoded = gatify_decode(gb, case_count, selector_bits);

    let mut ohs_cases: Vec<AigBitVector> = Vec::new();
    for case in cases {
        ohs_cases.push(case.clone());
    }

    if let Some(default_bits) = default_bits {
        // This is the scenario where the select has an OOB case.
        ohs_cases.push(default_bits.clone());
        let oob = gb.add_ez(&index_decoded, ReductionKind::Tree);
        let one_hot_selector = AigBitVector::concat(oob.into(), index_decoded);
        gatify_one_hot_select(gb, &one_hot_selector, &ohs_cases)
    } else {
        // This is the scenario where there is no OOB case so we can just OHS using the
        // decoded value.
        gatify_one_hot_select(gb, &index_decoded, &ohs_cases)
    }
}

fn gatify_mul(
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    output_bit_count: usize,
    signedness: Signedness,
    gb: &mut GateBuilder,
) -> AigBitVector {
    match signedness {
        Signedness::Unsigned => gatify_umul(lhs_bits, rhs_bits, output_bit_count, gb),
        Signedness::Signed => {
            // Pre-sign-extend operands to the final output width.
            let lhs_ext = if lhs_bits.get_bit_count() < output_bit_count {
                gatify_sign_ext(gb, 0, output_bit_count, lhs_bits)
            } else {
                lhs_bits.clone()
            };
            let rhs_ext = if rhs_bits.get_bit_count() < output_bit_count {
                gatify_sign_ext(gb, 0, output_bit_count, rhs_bits)
            } else {
                rhs_bits.clone()
            };
            gatify_umul(&lhs_ext, &rhs_ext, output_bit_count, gb)
        }
    }
}

fn gatify_concat(args: &[AigBitVector]) -> AigBitVector {
    let mut bits = Vec::new();
    for arg in args.iter().rev() {
        bits.extend(arg.iter_lsb_to_msb().cloned());
    }
    AigBitVector::from_lsb_is_index_0(&bits)
}

fn gatify_zero_ext(new_bit_count: usize, arg_bits: &AigBitVector) -> AigBitVector {
    let zero_count = new_bit_count - arg_bits.get_bit_count();
    let zeros = AigBitVector::zeros(zero_count);
    AigBitVector::concat(zeros, arg_bits.clone())
}

fn gatify_umul(
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    output_bit_count: usize,
    gb: &mut GateBuilder,
) -> AigBitVector {
    let lhs_bit_count = lhs_bits.get_bit_count();
    let rhs_bit_count = rhs_bits.get_bit_count();

    if lhs_bit_count == 0 || rhs_bit_count == 0 {
        assert_eq!(
            output_bit_count, 0,
            "output_bit_count must be 0 if lhs_bits or rhs_bits have no bits"
        );
        return AigBitVector::zeros(0);
    }

    let mut partial_products = Vec::new();

    // For each bit in the multiplier (rhs), generate a scaled partial product
    for (i, rhs_bit) in rhs_bits.iter_lsb_to_msb().enumerate() {
        let mut row = Vec::new();
        for lhs_bit in lhs_bits.iter_lsb_to_msb() {
            let pp = gb.add_and_binary(*lhs_bit, *rhs_bit);
            row.push(pp);
        }

        // Shift the partial product left by i positions (prepend i false bits)
        let mut shifted = vec![gb.get_false(); i];
        shifted.extend(row);

        // Ensure the partial product has the correct width
        while shifted.len() < output_bit_count {
            shifted.push(gb.get_false());
        }
        while shifted.len() > output_bit_count {
            shifted.pop();
        }

        partial_products.push(AigBitVector::from_lsb_is_index_0(&shifted));
    }

    // Sum all partial products using ripple-carry addition
    let mut result = partial_products[0].clone();
    for pp in partial_products.iter().skip(1) {
        let (_carry, sum) = gatify_add_ripple_carry(&result, pp, gb.get_false(), None, gb);
        result = sum;
    }
    result
}

fn gatify_twos_complement(bits: &AigBitVector, gb: &mut GateBuilder) -> AigBitVector {
    let inverted = gb.add_not_vec(bits);
    let one = gb.add_literal(&xlsynth::IrBits::make_ubits(bits.get_bit_count(), 1).unwrap());
    let (_carry, sum) = gatify_add_ripple_carry(&inverted, &one, gb.get_false(), None, gb);
    sum
}

fn gatify_abs(bits: &AigBitVector, gb: &mut GateBuilder) -> AigBitVector {
    let sign = bits.get_msb(0);
    let negated = gatify_twos_complement(bits, gb);
    gb.add_mux2_vec(sign, &negated, bits)
}

fn gatify_udiv(
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    gb: &mut GateBuilder,
) -> AigBitVector {
    assert_eq!(lhs_bits.get_bit_count(), rhs_bits.get_bit_count());
    let bit_count = lhs_bits.get_bit_count();

    let mut remainder = AigBitVector::zeros(bit_count);
    let mut quotient_bits = Vec::with_capacity(bit_count);

    for dividend_bit in lhs_bits.iter_msb_to_lsb() {
        let mut shifted_ops = Vec::with_capacity(bit_count);
        shifted_ops.push(*dividend_bit);
        for i in 0..bit_count - 1 {
            shifted_ops.push(*remainder.get_lsb(i));
        }
        let shifted = AigBitVector::from_lsb_is_index_0(&shifted_ops);

        let ge = gatify_uge_via_bit_tests(gb, 0, &shifted, rhs_bits);
        let rhs_comp = gb.add_not_vec(rhs_bits);
        let (_c, diff) = gatify_add_ripple_carry(&shifted, &rhs_comp, gb.get_true(), None, gb);
        remainder = gb.add_mux2_vec(&ge, &diff, &shifted);
        quotient_bits.push(ge);
    }

    quotient_bits.reverse();
    let quotient = AigBitVector::from_lsb_is_index_0(&quotient_bits);

    let divisor_zero = gb.add_ez(rhs_bits, ReductionKind::Tree);
    let ones = gb.replicate(gb.get_true(), bit_count);
    gb.add_mux2_vec(&divisor_zero, &ones, &quotient)
}

fn gatify_div(
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    signedness: Signedness,
    gb: &mut GateBuilder,
) -> AigBitVector {
    match signedness {
        Signedness::Unsigned => gatify_udiv(lhs_bits, rhs_bits, gb),
        Signedness::Signed => {
            let lhs_abs = gatify_abs(lhs_bits, gb);
            let rhs_abs = gatify_abs(rhs_bits, gb);
            let unsigned = gatify_udiv(&lhs_abs, &rhs_abs, gb);
            let sign_a = lhs_bits.get_msb(0);
            let sign_b = rhs_bits.get_msb(0);
            let result_neg = gb.add_xor_binary(*sign_a, *sign_b);
            let negated = gatify_twos_complement(&unsigned, gb);
            let signed_result = gb.add_mux2_vec(&result_neg, &negated, &unsigned);
            let divisor_zero = gb.add_ez(rhs_bits, ReductionKind::Tree);
            let ones = gb.replicate(gb.get_true(), lhs_bits.get_bit_count());
            gb.add_mux2_vec(&divisor_zero, &ones, &signed_result)
        }
    }
}

fn gatify_umod(
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    gb: &mut GateBuilder,
) -> AigBitVector {
    assert_eq!(lhs_bits.get_bit_count(), rhs_bits.get_bit_count());
    let bit_count = lhs_bits.get_bit_count();

    let mut remainder = AigBitVector::zeros(bit_count);

    for dividend_bit in lhs_bits.iter_msb_to_lsb() {
        let mut shifted_ops = Vec::with_capacity(bit_count);
        shifted_ops.push(*dividend_bit);
        for i in 0..bit_count - 1 {
            shifted_ops.push(*remainder.get_lsb(i));
        }
        let shifted = AigBitVector::from_lsb_is_index_0(&shifted_ops);

        let ge = gatify_uge_via_bit_tests(gb, 0, &shifted, rhs_bits);
        let rhs_comp = gb.add_not_vec(rhs_bits);
        let (_c, diff) = gatify_add_ripple_carry(&shifted, &rhs_comp, gb.get_true(), None, gb);
        remainder = gb.add_mux2_vec(&ge, &diff, &shifted);
    }

    let divisor_zero = gb.add_ez(rhs_bits, ReductionKind::Tree);
    let zeros = AigBitVector::zeros(bit_count);
    gb.add_mux2_vec(&divisor_zero, &zeros, &remainder)
}

fn gatify_mod(
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    signedness: Signedness,
    gb: &mut GateBuilder,
) -> AigBitVector {
    match signedness {
        Signedness::Unsigned => gatify_umod(lhs_bits, rhs_bits, gb),
        Signedness::Signed => {
            let lhs_abs = gatify_abs(lhs_bits, gb);
            let rhs_abs = gatify_abs(rhs_bits, gb);
            let unsigned = gatify_umod(&lhs_abs, &rhs_abs, gb);
            let sign_a = lhs_bits.get_msb(0);
            let negated = gatify_twos_complement(&unsigned, gb);
            let signed_result = gb.add_mux2_vec(sign_a, &negated, &unsigned);
            let divisor_zero = gb.add_ez(rhs_bits, ReductionKind::Tree);
            let zeros = AigBitVector::zeros(lhs_bits.get_bit_count());
            gb.add_mux2_vec(&divisor_zero, &zeros, &signed_result)
        }
    }
}

#[allow(dead_code)]
fn gatify_ugt_via_adder(
    gb: &mut GateBuilder,
    text_id: usize,
    a_bits: &AigBitVector,
    b_bits: &AigBitVector,
) -> AigOperand {
    // ugt(a, b) iff a - b is > 0
    let b_complement = gb.add_not_vec(&b_bits);
    let (carry_out, sub_result) = gatify_add_ripple_carry(
        &a_bits,
        &b_complement,
        gb.get_true(),
        Some(&format!("ugt_{}", text_id)),
        gb,
    );
    let sub_result_is_zero = gb.add_ez(&sub_result, ReductionKind::Tree);
    let sub_result_is_nonzero = gb.add_not(sub_result_is_zero);
    // The carry_out represents that `a >= b`.
    let sub_result_is_positive = carry_out;
    gb.add_and_binary(sub_result_is_positive, sub_result_is_nonzero)
}

#[allow(dead_code)]
fn gatify_uge_via_adder(
    gb: &mut GateBuilder,
    text_id: usize,
    a_bits: &AigBitVector,
    b_bits: &AigBitVector,
) -> AigOperand {
    let b_complement = gb.add_not_vec(&b_bits);
    let (carry_out, _sub_result) = gatify_add_ripple_carry(
        &a_bits,
        &b_complement,
        gb.get_true(),
        Some(&format!("uge_{}", text_id)),
        gb,
    );
    // The carry_out represents that `a >= b`.
    carry_out
}

#[allow(dead_code)]
fn gatify_ult_via_adder(
    gb: &mut GateBuilder,
    text_id: usize,
    a_bits: &AigBitVector,
    b_bits: &AigBitVector,
) -> AigOperand {
    let b_inverted = gb.add_not_vec(&b_bits);
    let (c_out, _sub_result) = gatify_add_ripple_carry(
        &a_bits,
        &b_inverted,
        gb.get_true(),
        Some(&format!("ult_{}", text_id)),
        gb,
    );
    gb.add_not(c_out)
}

pub fn gatify_ule_via_adder(
    gb: &mut GateBuilder,
    text_id: usize,
    a_bits: &AigBitVector,
    b_bits: &AigBitVector,
) -> AigOperand {
    let b_inverted = gb.add_not_vec(&b_bits);
    let (c_out, _sub_result) = gatify_add_ripple_carry(
        &a_bits,
        &b_inverted,
        gb.get_true(),
        Some(&format!("ule_{}", text_id)),
        gb,
    );
    // Note: there's a choice here of whether to test after subtraction or equality
    // before subtraction.
    let is_lt = gb.add_not(c_out);
    let is_eq = gb.add_eq_vec(&a_bits, &b_bits, ReductionKind::Tree);
    gb.add_or_binary(is_lt, is_eq)
}

/// This is the generalization of unsigned comparisons via bit tests.
///
/// The basic premise is that all comparisons care about:
/// * Whether prior bits are all the same
/// * Then do some test on the next bit in that scenario, given the two
///   operands.
/// * Those all get or'd together to form the final result.
/// * And maybe the final result also wants to know if the whole vector was
///   equal.
pub fn gatify_ucmp_via_bit_tests<F>(
    gb: &mut GateBuilder,
    _text_id: usize,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    or_eq: bool,
    handle_bit: &F,
) -> AigOperand
where
    F: Fn(&mut GateBuilder, AigOperand, AigOperand) -> AigOperand,
{
    assert_eq!(lhs_bits.get_bit_count(), rhs_bits.get_bit_count());
    let input_bit_count = lhs_bits.get_bit_count();
    let eq_bits = gb.add_xnor_vec(lhs_bits, rhs_bits);
    let mut bit_tests = Vec::new();
    for msb_i in 0..input_bit_count {
        let eq_bits_slice = eq_bits.get_msbs(msb_i);
        let prior_bits_equal = if eq_bits_slice.is_empty() {
            assert_eq!(msb_i, 0);
            gb.get_true()
        } else {
            gb.add_and_reduce(&eq_bits_slice, ReductionKind::Tree)
        };
        let lhs_bit = lhs_bits.get_msb(msb_i);
        let rhs_bit = rhs_bits.get_msb(msb_i);
        let this_bit_test = handle_bit(gb, *lhs_bit, *rhs_bit);
        let bit = gb.add_and_binary(this_bit_test, prior_bits_equal);
        bit_tests.push(bit);
    }
    let result = gb.add_or_nary(&bit_tests, ReductionKind::Tree);
    if or_eq {
        let eq_bits = gb.add_and_reduce(&eq_bits, ReductionKind::Tree);
        gb.add_or_binary(result, eq_bits)
    } else {
        result
    }
}

pub fn gatify_ult_via_bit_tests(
    gb: &mut GateBuilder,
    _text_id: usize,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
) -> AigOperand {
    gatify_ucmp_via_bit_tests(
        gb,
        _text_id,
        lhs_bits,
        rhs_bits,
        false,
        &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
            let lhs_bit_unset = gb.add_not(lhs_bit);
            let rhs_bit_set = rhs_bit;
            let rhs_larger_this_bit = gb.add_and_binary(lhs_bit_unset, rhs_bit_set);
            rhs_larger_this_bit
        },
    )
}

/// This lowers a unsigned `lhs <= rhs` operator by testing bits in sequence (in
/// lieu of using an adder like `gatify_ule_via_adder` above).
pub fn gatify_ule_via_bit_tests(
    gb: &mut GateBuilder,
    _text_id: usize,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
) -> AigOperand {
    gatify_ucmp_via_bit_tests(
        gb,
        _text_id,
        lhs_bits,
        rhs_bits,
        true,
        &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
            let lhs_bit_unset = gb.add_not(lhs_bit);
            let rhs_bit_set = rhs_bit;
            let rhs_larger_this_bit = gb.add_and_binary(lhs_bit_unset, rhs_bit_set);
            rhs_larger_this_bit
        },
    )
}

pub fn gatify_ugt_via_bit_tests(
    gb: &mut GateBuilder,
    _text_id: usize,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
) -> AigOperand {
    assert!(
        lhs_bits.get_bit_count() > 0 && rhs_bits.get_bit_count() > 0,
        "cannot compare 0-bit operands; lhs_bits: {:?} rhs_bits: {:?}",
        lhs_bits,
        rhs_bits
    );
    gatify_ucmp_via_bit_tests(
        gb,
        _text_id,
        lhs_bits,
        rhs_bits,
        false,
        &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
            let lhs_bit_set = lhs_bit;
            let rhs_bit_unset = gb.add_not(rhs_bit);
            let lhs_larger_this_bit = gb.add_and_binary(lhs_bit_set, rhs_bit_unset);
            lhs_larger_this_bit
        },
    )
}

pub fn gatify_uge_via_bit_tests(
    gb: &mut GateBuilder,
    _text_id: usize,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
) -> AigOperand {
    gatify_ucmp_via_bit_tests(
        gb,
        _text_id,
        lhs_bits,
        rhs_bits,
        true,
        &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
            let lhs_bit_set = lhs_bit;
            let rhs_bit_unset = gb.add_not(rhs_bit);
            let lhs_larger_this_bit = gb.add_and_binary(lhs_bit_set, rhs_bit_unset);
            lhs_larger_this_bit
        },
    )
}

pub enum CmpKind {
    Lt,
    Gt,
}

pub fn gatify_scmp(
    gb: &mut GateBuilder,
    text_id: usize,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    adder_mapping: AdderMapping,
    cmp_kind: CmpKind,
    or_eq: bool,
) -> AigOperand {
    assert_eq!(
        lhs_bits.get_bit_count(),
        rhs_bits.get_bit_count(),
        "scmp requires equal-width bit vectors"
    );
    assert!(
        lhs_bits.get_bit_count() > 0,
        "scmp requires non-zero-width bit vectors"
    );
    let bit_count = lhs_bits.get_bit_count();
    if bit_count == 1 {
        // Special-case 1-bit: In two's complement, 0 represents 0 and 1 represents -1.
        // Thus, for a 1-bit comparison:
        //   a < b is true if a = 1 and b = 0.
        //   a > b is true if a = 0 and b = 1.
        let a = *lhs_bits.get_lsb(0);
        let b = *rhs_bits.get_lsb(0);
        match cmp_kind {
            CmpKind::Lt => {
                let b_complement = gb.add_not(b);
                let slt = gb.add_and_binary(a, b_complement);
                if or_eq {
                    let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
                    gb.add_or_binary(slt, eq)
                } else {
                    slt
                }
            }
            CmpKind::Gt => {
                let a_complement = gb.add_not(a);
                let sgt = gb.add_and_binary(a_complement, b);
                if or_eq {
                    let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
                    gb.add_or_binary(sgt, eq)
                } else {
                    sgt
                }
            }
        }
    } else {
        // For multi-bit signed comparisons, compute diff = a - b = a + (not b) + 1.
        let b_complement = gb.add_not_vec(rhs_bits);
        let (_carry, diff) = match adder_mapping {
            AdderMapping::RippleCarry => gatify_add_ripple_carry(
                lhs_bits,
                &b_complement,
                gb.get_true(),
                Some(&format!("scmp_{}", text_id)),
                gb,
            ),
            AdderMapping::BrentKung => gatify_add_brent_kung(
                lhs_bits,
                &b_complement,
                gb.get_true(),
                Some(&format!("scmp_{}", text_id)),
                gb,
            ),
            AdderMapping::KoggeStone => gatify_add_kogge_stone(
                lhs_bits,
                &b_complement,
                gb.get_true(),
                Some(&format!("scmp_{}", text_id)),
                gb,
            ),
        };
        let a_msb = lhs_bits.get_msb(0);
        let b_msb = rhs_bits.get_msb(0);
        let diff_msb = diff.get_msb(0);

        // Compute sign_diff = a_msb XOR b_msb.
        let sign_diff = gb.add_xor_binary(*a_msb, *b_msb);
        // When the signs differ, a < b if a_msb is 1.
        // When the signs are the same, a < b if diff's msb is 1.
        let term1 = gb.add_and_binary(sign_diff, *a_msb);
        let not_sign_diff = gb.add_not(sign_diff);
        let term2 = gb.add_and_binary(not_sign_diff, *diff_msb);
        let lt = gb.add_or_binary(term1, term2);
        let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
        match cmp_kind {
            CmpKind::Lt => {
                if or_eq {
                    gb.add_or_binary(lt, eq)
                } else {
                    lt
                }
            }
            CmpKind::Gt => {
                let lt_or_eq = gb.add_or_binary(lt, eq);
                let gt = gb.add_not(lt_or_eq);
                if or_eq { gb.add_or_binary(gt, eq) } else { gt }
            }
        }
    }
}

fn gatify_sign_ext(
    gb: &mut GateBuilder,
    text_id: usize,
    new_bit_count: usize,
    arg_bits: &AigBitVector,
) -> AigBitVector {
    let msb = arg_bits.get_msb(0);
    gb.add_tag(msb.node, format!("sign_ext_{}_msb", text_id));
    let input_bit_count = arg_bits.get_bit_count();
    assert!(new_bit_count >= input_bit_count);
    let replicated_msb = gb.replicate(*msb, new_bit_count - input_bit_count);

    // Concatenate the replicated msb with the gates[1..]
    let result = AigBitVector::concat(replicated_msb, arg_bits.clone());

    assert_eq!(result.get_bit_count(), new_bit_count);
    result
}

/// The `decode` operation tests whether the input matches a particular value,
/// and if so sets that corresponding bit in the output to be 1.
///
/// Since the input bits can only take on at most one value at a time,
/// this output vector is at most one hot. If the width of the output
/// is the full `output_bits = 2^input_bits`, then the output is
/// guaranteed to be a one-hot bit vector.
#[allow(dead_code)]
fn gatify_decode_naive(
    gb: &mut GateBuilder,
    output_width: usize,
    input_bits: &AigBitVector,
) -> AigBitVector {
    let input_bit_count = input_bits.get_bit_count();
    log::info!(
        "gatify_decode; input_bit_count: {}; width: {}",
        input_bit_count,
        output_width
    );
    let mut bits = Vec::new();
    for i in 0..output_width {
        let literal_bits =
            gb.add_literal(&xlsynth::IrBits::make_ubits(input_bit_count, i as u64).unwrap());
        let is_selected = gb.add_eq_vec(&input_bits, &literal_bits, ReductionKind::Tree);
        bits.push(is_selected);
    }
    AigBitVector::from_lsb_is_index_0(&bits)
}

#[allow(dead_code)]
fn gatify_encode_naive(
    gb: &mut GateBuilder,
    output_bit_count: usize,
    arg_bits: &AigBitVector,
) -> AigBitVector {
    let input_bit_count = arg_bits.get_bit_count();
    let mut to_or_reduce = Vec::new();
    for i in 0..input_bit_count {
        let gate_i_set = arg_bits.get_lsb(i);
        let gate_i_mask = gb.replicate(*gate_i_set, output_bit_count);
        let on_selected =
            gb.add_literal(&xlsynth::IrBits::make_ubits(output_bit_count, i as u64).unwrap());
        let masked = gb.add_and_vec(&gate_i_mask, &on_selected);
        to_or_reduce.push(masked);
    }
    let to_or_reduce_slices: Vec<AigBitVector> = to_or_reduce.iter().map(|v| v.clone()).collect();
    let or_reduced = gb.add_or_vec_nary(&to_or_reduce_slices, ReductionKind::Tree);
    assert_eq!(or_reduced.get_bit_count(), output_bit_count);
    or_reduced
}

fn gatify_encode(
    gb: &mut GateBuilder,
    output_bit_count: usize,
    arg_bits: &AigBitVector,
) -> AigBitVector {
    let input_bit_count = arg_bits.get_bit_count();
    let mut encoded_output = vec![gb.get_false(); output_bit_count];
    for j in 0..output_bit_count {
        // Input bits that contribute to the jth output bit.
        let mut or_candidates: Vec<AigOperand> = Vec::new();

        // For each input bit, check if the j-th bit of its index is `1`. That means
        // it contributes to the jth output bit.
        for i in 0..input_bit_count {
            // Shifting by an amount >= width of usize is UB in Rust. If that
            // happens we know the result would be zero, so we can avoid the
            // shift entirely.
            let bit_set = if j < usize::BITS as usize {
                ((i >> j) & 1) == 1
            } else {
                false
            };

            if bit_set {
                or_candidates.push(arg_bits.get_lsb(i).clone());
            }
        }

        // Now we compute the j-th output bit by or-reducing the candidates.
        encoded_output[j] = if or_candidates.is_empty() {
            gb.get_false()
        } else {
            gb.add_or_nary(&or_candidates, ReductionKind::Tree)
        };
    }
    AigBitVector::from_lsb_is_index_0(&encoded_output)
}

fn gatify_decode(
    gb: &mut GateBuilder,
    output_bit_count: usize,
    arg_bits: &AigBitVector,
) -> AigBitVector {
    let input_count = arg_bits.get_bit_count();

    let mut outputs = Vec::with_capacity(output_bit_count);
    log::debug!(
        "gatify_decode: input_count: {}, output_bit_count: {}",
        input_count,
        output_bit_count
    );

    // For every possible output value...
    for i in 0..output_bit_count {
        let mut terms = Vec::with_capacity(input_count);
        // For each bit in the input, choose whether to use the bit directly or its
        // inversion.
        for j in 0..input_count {
            let bit = arg_bits.get_lsb(j);
            // Avoid shifting by more than the width of `usize`.
            let bit_set = if j < usize::BITS as usize {
                ((i >> j) & 1) == 1
            } else {
                false
            };

            if bit_set {
                terms.push(bit.clone());
            } else {
                terms.push(gb.add_not(*bit));
            }
        }
        // AND the terms together to generate the one-hot output.
        let out = gb.add_and_nary(&terms, ReductionKind::Tree);
        outputs.push(out);
    }
    AigBitVector::from_lsb_is_index_0(&outputs)
}

// Refactored gatify_shra function
fn gatify_shra(
    gb: &mut GateBuilder,
    arg_bits: &AigBitVector,
    amount_bits: &AigBitVector,
    text_id: usize,
) -> AigBitVector {
    let input_bit_count = arg_bits.get_bit_count();

    // The shift amount is given in its own bit width
    let shift_amount_width = amount_bits.get_bit_count();

    // Compute the required number of bits for a valid shift amount
    let required_shift_bits = if input_bit_count > 1 {
        (input_bit_count as f64).log2().ceil() as usize
    } else {
        1
    };

    // Get the MSB (sign bit) of the input
    let msb = arg_bits.get_msb(0);

    // First perform a logical right shift
    let logical_shift = gatify_barrel_shifter(
        &arg_bits,
        &amount_bits,
        Direction::Right,
        &format!("shra_logical_{}", text_id),
        gb,
    );

    // Computes the conditional mux bits used in both narrow and wide cases.
    fn compute_shifted_bits(
        gb: &mut GateBuilder,
        msb: &AigOperand,
        logical_shift: &AigBitVector,
        input_bit_count: usize,
        decode_len: usize,
        decoded_amount: &AigBitVector,
    ) -> Vec<AigOperand> {
        let mut bits = Vec::with_capacity(input_bit_count);
        for j in 0..input_bit_count {
            let start_n = if input_bit_count > j {
                input_bit_count - j
            } else {
                0
            };
            let mut cond_terms = Vec::new();
            for n in start_n..decode_len {
                cond_terms.push(decoded_amount.get_lsb(n).clone());
            }
            let condition = if cond_terms.is_empty() {
                gb.get_false()
            } else if cond_terms.len() == 1 {
                cond_terms[0]
            } else {
                gb.add_or_nary(&cond_terms, ReductionKind::Tree)
            };
            let res_bit = gb.add_mux2(condition, *msb, *logical_shift.get_lsb(j));
            bits.push(res_bit);
        }
        bits
    }

    if shift_amount_width <= required_shift_bits {
        // Narrow case: use the full amount_bits
        let decode_len = 1 << shift_amount_width;
        let decoded_amount = gatify_decode(gb, decode_len, &amount_bits);
        let result_bits = compute_shifted_bits(
            gb,
            msb,
            &logical_shift,
            input_bit_count,
            decode_len,
            &decoded_amount,
        );
        return AigBitVector::from_lsb_is_index_0(&result_bits);
    }

    // Applies the out-of-bounds mux on each bit
    fn apply_out_of_bounds_mux(
        gb: &mut GateBuilder,
        input_bit_count: usize,
        out_of_bounds: &AigOperand,
        mask: &AigBitVector,
        in_bound_result: &AigBitVector,
    ) -> Vec<AigOperand> {
        let mut bits = Vec::with_capacity(input_bit_count);
        for j in 0..input_bit_count {
            let final_bit = gb.add_mux2(
                out_of_bounds.clone(),
                *mask.get_lsb(j),
                *in_bound_result.get_lsb(j),
            );
            bits.push(final_bit);
        }
        bits
    }

    // Wide case: use only the lower effective bits
    let effective_shift_width = required_shift_bits;
    let effective_decode_len = 1 << effective_shift_width;

    // Extract the lower bits of amount_bits (effective shift amount)
    let mut effective_amount_bits_vec = Vec::with_capacity(effective_shift_width);
    for n in 0..effective_shift_width {
        effective_amount_bits_vec.push(amount_bits.get_lsb(n).clone());
    }
    let effective_amount_bits = AigBitVector::from_lsb_is_index_0(&effective_amount_bits_vec);

    // Compute out-of-bound condition from the higher bits
    let mut high_amt_terms = Vec::new();
    for n in effective_shift_width..shift_amount_width {
        high_amt_terms.push(amount_bits.get_lsb(n).clone());
    }
    let out_of_bounds = if high_amt_terms.len() == 1 {
        high_amt_terms[0].clone()
    } else {
        gb.add_or_nary(&high_amt_terms, ReductionKind::Tree)
    };

    let decoded_amount = gatify_decode(gb, effective_decode_len, &effective_amount_bits);
    let in_bound_bits = compute_shifted_bits(
        gb,
        msb,
        &logical_shift,
        input_bit_count,
        effective_decode_len,
        &decoded_amount,
    );
    let in_bound_result = AigBitVector::from_lsb_is_index_0(&in_bound_bits);

    // Create mask: replicate msb over the entire bit vector
    let mask = AigBitVector::from_lsb_is_index_0(&vec![*msb; input_bit_count]);

    // Final result: if out_of_bounds then select mask, else use in_bound_result
    let final_bits =
        apply_out_of_bounds_mux(gb, input_bit_count, &out_of_bounds, &mask, &in_bound_result);
    AigBitVector::from_lsb_is_index_0(&final_bits)
}

fn flatten_literal_to_bits(
    literal: &xlsynth::IrValue,
    ty: &ir::Type,
    g8_builder: &mut GateBuilder,
) -> AigBitVector {
    match ty {
        ir::Type::Bits(_) => {
            let bits = literal.to_bits().unwrap();
            g8_builder.add_literal(&bits)
        }
        ir::Type::Array(array_ty) => {
            let elements = literal.get_elements().unwrap();
            let mut bit_vectors = Vec::new();
            // Reverse order to match array flattening in node lowering (LSB = last element)
            for elem in elements.iter().rev() {
                let elem_bits = flatten_literal_to_bits(elem, &array_ty.element_type, g8_builder);
                bit_vectors.push(elem_bits);
            }
            // Concatenate all element bit vectors (LSb to MSb order)
            let mut lsb_to_msb = Vec::new();
            for bv in bit_vectors {
                lsb_to_msb.extend(bv.iter_lsb_to_msb().cloned());
            }
            AigBitVector::from_lsb_is_index_0(&lsb_to_msb)
        }
        ir::Type::Tuple(types) => {
            let elements = literal.get_elements().unwrap();
            let mut bit_vectors = Vec::new();
            // Reverse order to match tuple flattening in node lowering
            for (elem, elem_ty) in elements.iter().rev().zip(types.iter().rev()) {
                let elem_bits = flatten_literal_to_bits(elem, elem_ty, g8_builder);
                bit_vectors.push(elem_bits);
            }
            let mut lsb_to_msb = Vec::new();
            for bv in bit_vectors {
                lsb_to_msb.extend(bv.iter_lsb_to_msb().cloned());
            }
            AigBitVector::from_lsb_is_index_0(&lsb_to_msb)
        }
        ir::Type::Token => {
            // Tokens are zero bits, so return an empty bit vector
            AigBitVector::zeros(0)
        }
    }
}

/// Converts the contents of the given IR function to our "g8" representation
/// which has gates and vectors of gates.
fn gatify_internal(
    f: &ir::Fn,
    g8_builder: &mut GateBuilder,
    env: &mut GateEnv,
    options: &GatifyOptions,
) {
    log::info!("gatify_internal; f.name: {}", f.name);
    log::debug!("gatify; f:\n{}", f.to_string());

    // Precompute a map from parameter text_id to its NodeRef in f.nodes.
    let mut param_id_to_node_ref: HashMap<ParamId, ir::NodeRef> = HashMap::new();

    // First we place all the inputs into the G8 structure and environment.
    for (i, param) in f.params.iter().enumerate() {
        let param_ref = ir::NodeRef { index: i + 1 };
        assert!(f.nodes[i + 1].payload == ir::NodePayload::GetParam(param.id));
        log::debug!("Gatifying param {:?}", param);
        let gate_ref_vec = g8_builder.add_input(param.name.clone(), param.ty.bit_count());
        env.add(param_ref, GateOrVec::BitVector(gate_ref_vec));
        // Map ParamId to its NodeRef
        param_id_to_node_ref.insert(param.id, param_ref);
    }

    for node_ref in ir_utils::get_topological(f) {
        let node = &f.get_node(node_ref);
        let payload = &node.payload;
        log::debug!(
            "Gatifying node {:?} type: {:?} payload: {:?}",
            node_ref,
            node.ty,
            payload
        );
        match payload {
            ir::NodePayload::GetParam(param_id) => {
                if env.contains(node_ref) {
                    continue; // Already inserted above.
                }
                // Look up the original parameter node_ref by its ParamId
                let pr = param_id_to_node_ref
                    .get(param_id)
                    .expect("ParamId not found in mapping");
                let entry = env.get_bit_vector(*pr).unwrap();
                env.add(node_ref, GateOrVec::BitVector(entry));
            }
            ir::NodePayload::ArrayIndex {
                array,
                indices,
                assumed_in_bounds,
            } => {
                assert!(
                    !indices.is_empty(),
                    "Array index must have at least one index"
                );

                let mut array_ty = match f.get_node_ty(*array) {
                    ir::Type::Array(array_ty_data) => array_ty_data,
                    other => panic!("Expected array type for array_index, got {:?}", other),
                };
                let mut array_bits = env.get_bit_vector(*array).unwrap();

                for (i, index_node) in indices.iter().enumerate() {
                    let index_bits = env.get_bit_vector(*index_node).unwrap();
                    array_bits = gatify_array_index(
                        g8_builder,
                        array_ty,
                        &array_bits,
                        &index_bits,
                        *assumed_in_bounds,
                    );
                    if i + 1 < indices.len() {
                        array_ty = match array_ty.element_type.as_ref() {
                            ir::Type::Array(next_ty) => next_ty,
                            other => panic!(
                                "Expected array type for index dimension {}, got {:?}",
                                i + 1,
                                other
                            ),
                        };
                    }
                }

                env.add(node_ref, GateOrVec::BitVector(array_bits));
            }
            ir::NodePayload::ArrayUpdate {
                array,
                value,
                indices,
                assumed_in_bounds: _,
            } => {
                assert!(
                    !indices.is_empty(),
                    "Array update must have at least one index",
                );
                let array_ty = match f.get_node_ty(*array) {
                    ir::Type::Array(array_ty_data) => array_ty_data,
                    other => panic!("Expected array type for array_update, got {:?}", other),
                };

                let array_bits = env.get_bit_vector(*array).unwrap();
                let value_bits = env.get_bit_vector(*value).unwrap();
                let index_bits: Vec<AigBitVector> = indices
                    .iter()
                    .map(|i| env.get_bit_vector(*i).unwrap())
                    .collect();

                let result_bits = gatify_array_update(
                    g8_builder,
                    array_ty,
                    &array_bits,
                    &value_bits,
                    &index_bits,
                );

                env.add(node_ref, GateOrVec::BitVector(result_bits));
            }
            ir::NodePayload::Array(members) => {
                // Similar to Tuple: flatten all members into a bit vector.
                // Arrays are conceptually homogeneous, but in bit flattening, we just
                // concatenate all elements.
                let mut lsb_to_msb = Vec::new();
                for member in members.iter().rev() {
                    let member_bits = env
                        .get_bit_vector(*member)
                        .expect("array element should be present");
                    lsb_to_msb.extend(member_bits.iter_lsb_to_msb().cloned());
                }
                let bit_vector = AigBitVector::from_lsb_is_index_0(&lsb_to_msb);
                env.add(node_ref, GateOrVec::BitVector(bit_vector));
            }
            ir::NodePayload::TupleIndex { tuple, index } => {
                // We have to figure out what bit range the index indicates from the original
                // tuple's flat bits.
                let tuple_bits = env.get_bit_vector(*tuple).unwrap();
                let tuple_ty = f.get_node_ty(*tuple);
                let StartAndLimit { start, limit } =
                    tuple_ty.tuple_get_flat_bit_slice_for_index(*index).unwrap();
                let member_bits = tuple_bits
                    .iter_lsb_to_msb()
                    .skip(start)
                    .take(limit - start)
                    .cloned()
                    .collect::<Vec<_>>();
                let bit_vector = AigBitVector::from_lsb_is_index_0(&member_bits);
                env.add(node_ref, GateOrVec::BitVector(bit_vector));
            }
            ir::NodePayload::Tuple(args) => {
                // Tuples, similar to arrays, need to answer the question: "which member is
                // least significant when flattened?"
                //
                // When we perform: `tuple(a, b, c)` does `a` go in the lower bits or does `c`?
                //
                // For concat we have `concat(a, b, c)` place `a` in the upper bits such that
                // the result is: `a_msb, ..., a_lsb, b_msb, ..., b_lsb, c_msb,
                // ..., c_lsb`. So we take the same approach for tuples -- we
                // iterate the arguments in reverse order to make sure c's lsb
                // comes first and build from lsb to msb.

                let mut lsb_to_msb = Vec::new();
                for arg in args.iter().rev() {
                    let arg_gates = env
                        .get_bit_vector(*arg)
                        .expect("tuple arg should be present");
                    lsb_to_msb.extend(arg_gates.iter_lsb_to_msb().cloned());
                }
                let bit_vector = AigBitVector::from_lsb_is_index_0(&lsb_to_msb);
                env.add(node_ref, GateOrVec::BitVector(bit_vector));
            }
            ir::NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                // Note: sel is basically an array index into the cases where we pick default if
                // the selector value is OOB.
                let selector_bits = env
                    .get_bit_vector(*selector)
                    .expect("selector should be present");
                let cases: Vec<AigBitVector> = cases
                    .iter()
                    .map(|c| env.get_bit_vector(*c).expect("case should be present"))
                    .collect::<Vec<AigBitVector>>();
                let default_bits =
                    default.map(|d| env.get_bit_vector(d).expect("default should be present"));

                let gates = gatify_sel(g8_builder, &selector_bits, &cases, default_bits);

                // Tag the result bits
                for (i, gate) in gates.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(gate.node, format!("sel_{}_output_bit_{}", node.text_id, i));
                }
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Literal(literal) => {
                let bit_vector = flatten_literal_to_bits(literal, &node.ty, g8_builder);
                env.add(node_ref, GateOrVec::BitVector(bit_vector));
            }

            // -- unary operations
            ir::NodePayload::Unop(ir::Unop::Not, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let gates: AigBitVector = g8_builder.add_not_vec(&arg_gates);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Unop(ir::Unop::Neg, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let not_arg = g8_builder.add_not_vec(&arg_gates);
                let zero = g8_builder.add_literal(
                    &xlsynth::IrBits::make_ubits(arg_gates.get_bit_count(), 0).unwrap(),
                );
                let (_, result) = gatify_add_ripple_carry(
                    &not_arg,
                    &zero,
                    g8_builder.get_true(),
                    Some(&format!("neg_{}", node.text_id)),
                    g8_builder,
                );
                env.add(node_ref, GateOrVec::BitVector(result));
            }
            ir::NodePayload::Unop(ir::Unop::Identity, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                env.add(node_ref, GateOrVec::BitVector(arg_gates));
            }
            ir::NodePayload::Unop(ir::Unop::Reverse, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let result_gates: Vec<AigOperand> = arg_gates
                    .iter_lsb_to_msb()
                    .rev()
                    .cloned()
                    .collect::<Vec<_>>();
                env.add(
                    node_ref,
                    GateOrVec::BitVector(AigBitVector::from_lsb_is_index_0(&result_gates)),
                );
            }

            // -- bitwise reductions
            ir::NodePayload::Unop(ir::Unop::OrReduce, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let gate: AigOperand = g8_builder.add_or_reduce(&arg_gates, ReductionKind::Tree);
                g8_builder.add_tag(gate.node, format!("or_reduce_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Unop(ir::Unop::AndReduce, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let gate: AigOperand = g8_builder.add_and_reduce(&arg_gates, ReductionKind::Tree);
                g8_builder.add_tag(gate.node, format!("and_reduce_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Unop(ir::Unop::XorReduce, arg) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("unop arg should be present");
                let gate: AigOperand = g8_builder.add_xor_reduce(&arg_gates, ReductionKind::Tree);
                g8_builder.add_tag(gate.node, format!("xor_reduce_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }

            // -- binary operations
            ir::NodePayload::Binop(ir::Binop::Eq, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("eq lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("eq rhs should be present");
                assert_eq!(a_bits.get_bit_count(), b_bits.get_bit_count());
                let gate: AigOperand = g8_builder.add_eq_vec(&a_bits, &b_bits, ReductionKind::Tree);
                g8_builder.add_tag(gate.node, format!("eq_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Ne, a, b) => {
                let a_gate_refs = env.get_bit_vector(*a).expect("ne lhs should be present");
                let b_gate_refs = env.get_bit_vector(*b).expect("ne rhs should be present");
                log::debug!(
                    "ne lhs bits[{}] rhs bits[{}]",
                    a_gate_refs.get_bit_count(),
                    b_gate_refs.get_bit_count()
                );
                let gate: AigOperand =
                    g8_builder.add_ne_vec(&a_gate_refs, &b_gate_refs, ReductionKind::Tree);
                g8_builder.add_tag(gate.node, format!("ne_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Ult, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("ult lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("ult rhs should be present");
                let gate: AigOperand =
                    gatify_ult_via_bit_tests(g8_builder, node.text_id, &a_bits, &b_bits);
                g8_builder.add_tag(gate.node, format!("ult_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Ugt, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("ugt lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("ugt rhs should be present");
                let gate: AigOperand =
                    gatify_ugt_via_bit_tests(g8_builder, node.text_id, &a_bits, &b_bits);
                g8_builder.add_tag(gate.node, format!("ugt_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Uge, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("uge lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("uge rhs should be present");
                let gate: AigOperand =
                    gatify_uge_via_bit_tests(g8_builder, node.text_id, &a_bits, &b_bits);
                g8_builder.add_tag(gate.node, format!("uge_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Ule, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("ule lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("ule rhs should be present");
                let gate = gatify_ule_via_bit_tests(g8_builder, node.text_id, &a_bits, &b_bits);
                g8_builder.add_tag(gate.node, format!("ule_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }

            // signed comparisons
            ir::NodePayload::Binop(ir::Binop::Sgt, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("sgt lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("sgt rhs should be present");
                let gate = gatify_scmp(
                    g8_builder,
                    node.text_id,
                    &a_bits,
                    &b_bits,
                    options.adder_mapping,
                    CmpKind::Gt,
                    false,
                );
                g8_builder.add_tag(gate.node, format!("sgt_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Sge, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("sge lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("sge rhs should be present");
                let gate = gatify_scmp(
                    g8_builder,
                    node.text_id,
                    &a_bits,
                    &b_bits,
                    options.adder_mapping,
                    CmpKind::Gt,
                    true,
                );
                g8_builder.add_tag(gate.node, format!("sge_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Slt, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("slt lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("slt rhs should be present");
                let gate = gatify_scmp(
                    g8_builder,
                    node.text_id,
                    &a_bits,
                    &b_bits,
                    options.adder_mapping,
                    CmpKind::Lt,
                    false,
                );
                g8_builder.add_tag(gate.node, format!("slt_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }
            ir::NodePayload::Binop(ir::Binop::Sle, a, b) => {
                let a_bits = env.get_bit_vector(*a).expect("sle lhs should be present");
                let b_bits = env.get_bit_vector(*b).expect("sle rhs should be present");
                let gate = gatify_scmp(
                    g8_builder,
                    node.text_id,
                    &a_bits,
                    &b_bits,
                    options.adder_mapping,
                    CmpKind::Lt,
                    true,
                );
                g8_builder.add_tag(gate.node, format!("sle_{}", node.text_id));
                env.add(node_ref, GateOrVec::Gate(gate));
            }

            // -- nary operations
            ir::NodePayload::Nary(ir::NaryOp::And, args) => {
                let arg_gates: Vec<AigBitVector> = args
                    .iter()
                    .map(|arg| env.get_bit_vector(*arg).expect("and arg should be present"))
                    .collect();
                let gates: AigBitVector =
                    g8_builder.add_and_vec_nary(arg_gates.as_slice(), ReductionKind::Tree);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Nary(ir::NaryOp::Nor, args) => {
                let arg_gates: Vec<AigBitVector> = args
                    .iter()
                    .map(|arg| env.get_bit_vector(*arg).expect("nor arg should be present"))
                    .collect();
                let gates: AigBitVector =
                    g8_builder.add_or_vec_nary(&arg_gates, ReductionKind::Tree);
                let gates = g8_builder.add_not_vec(&gates);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Nary(ir::NaryOp::Or, args) => {
                let arg_gates: Vec<AigBitVector> = args
                    .iter()
                    .map(|arg| env.get_bit_vector(*arg).expect("or arg should be present"))
                    .collect();
                let gates: AigBitVector =
                    g8_builder.add_or_vec_nary(&arg_gates, ReductionKind::Tree);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Nary(ir::NaryOp::Xor, args) => {
                let arg_gates: Vec<AigBitVector> = args
                    .iter()
                    .map(|arg| env.get_bit_vector(*arg).expect("xor arg should be present"))
                    .collect();
                let gates: AigBitVector =
                    g8_builder.add_xor_vec_nary(&arg_gates, ReductionKind::Tree);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Nary(ir::NaryOp::Nand, args) => {
                let arg_gates: Vec<AigBitVector> = args
                    .iter()
                    .map(|arg| {
                        env.get_bit_vector(*arg)
                            .expect("nand arg should be present")
                    })
                    .collect();
                let gates: AigBitVector =
                    g8_builder.add_and_vec_nary(&arg_gates, ReductionKind::Tree);
                let gates = g8_builder.add_not_vec(&gates);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Nary(ir::NaryOp::Concat, args) => {
                let arg_bits: Vec<AigBitVector> = args
                    .iter()
                    .map(|arg| {
                        env.get_bit_vector(*arg)
                            .expect("concat arg should be present")
                    })
                    .collect();

                let bits = gatify_concat(&arg_bits);

                let output_bit_count = node.ty.bit_count();
                assert_eq!(bits.get_bit_count(), output_bit_count);
                for (i, bit) in bits.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        bit.node,
                        format!("concat_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(bits));
            }
            ir::NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                let output_bit_count = node.ty.bit_count();
                let selector_bits = env
                    .get_bit_vector(*selector)
                    .expect("selector should be present");
                let cases: Vec<AigBitVector> = cases
                    .iter()
                    .map(|c| env.get_bit_vector(*c).expect("case should be present"))
                    .collect::<Vec<AigBitVector>>();
                let default_bits =
                    default.map(|d| env.get_bit_vector(d).expect("default should be present"));

                let gates = gatify_priority_sel(
                    g8_builder,
                    output_bit_count,
                    selector_bits,
                    cases.as_slice(),
                    default_bits,
                );
                // Tag the result.
                for (i, gate) in gates.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        gate.node,
                        format!("priority_sel_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::OneHotSel { selector, cases } => {
                let selector_bits = env
                    .get_bit_vector(*selector)
                    .expect("selector should be present");
                let cases: Vec<AigBitVector> = cases
                    .iter()
                    .map(|c| env.get_bit_vector(*c).expect("case should be present"))
                    .collect::<Vec<AigBitVector>>();
                let gates = gatify_one_hot_select(g8_builder, &selector_bits, &cases);
                // Tag the result.
                for (i, gate) in gates.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        gate.node,
                        format!("one_hot_select_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Binop(ir::Binop::Add, a, b) => {
                let a_gate_refs = env.get_bit_vector(*a).expect("add lhs should be present");
                let b_gate_refs = env.get_bit_vector(*b).expect("add rhs should be present");
                assert_eq!(a_gate_refs.get_bit_count(), b_gate_refs.get_bit_count());
                let (_c_out, gates) = match options.adder_mapping {
                    crate::ir2gate_utils::AdderMapping::RippleCarry => gatify_add_ripple_carry(
                        &a_gate_refs,
                        &b_gate_refs,
                        g8_builder.get_false(),
                        Some(&format!("add_{}", node.text_id)),
                        g8_builder,
                    ),
                    crate::ir2gate_utils::AdderMapping::BrentKung => gatify_add_brent_kung(
                        &a_gate_refs,
                        &b_gate_refs,
                        g8_builder.get_false(),
                        Some(&format!("add_{}", node.text_id)),
                        g8_builder,
                    ),
                    crate::ir2gate_utils::AdderMapping::KoggeStone => gatify_add_kogge_stone(
                        &a_gate_refs,
                        &b_gate_refs,
                        g8_builder.get_false(),
                        Some(&format!("add_{}", node.text_id)),
                        g8_builder,
                    ),
                };
                assert_eq!(gates.get_bit_count(), a_gate_refs.get_bit_count());
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Binop(ir::Binop::Sub, a, b) => {
                let a_gate_refs = env.get_bit_vector(*a).expect("sub lhs should be present");
                let b_gate_refs = env.get_bit_vector(*b).expect("sub rhs should be present");
                assert_eq!(a_gate_refs.get_bit_count(), b_gate_refs.get_bit_count());
                let b_complement = g8_builder.add_not_vec(&b_gate_refs);
                let (_c_out, gates) = match options.adder_mapping {
                    crate::ir2gate_utils::AdderMapping::RippleCarry => gatify_add_ripple_carry(
                        &a_gate_refs,
                        &b_complement,
                        g8_builder.get_true(),
                        Some(&format!("sub_{}", node.text_id)),
                        g8_builder,
                    ),
                    crate::ir2gate_utils::AdderMapping::BrentKung => gatify_add_brent_kung(
                        &a_gate_refs,
                        &b_complement,
                        g8_builder.get_true(),
                        Some(&format!("sub_{}", node.text_id)),
                        g8_builder,
                    ),
                    crate::ir2gate_utils::AdderMapping::KoggeStone => gatify_add_kogge_stone(
                        &a_gate_refs,
                        &b_complement,
                        g8_builder.get_true(),
                        Some(&format!("sub_{}", node.text_id)),
                        g8_builder,
                    ),
                };
                let output_bit_count = node.ty.bit_count();
                assert_eq!(gates.get_bit_count(), output_bit_count);
                for (i, gate) in gates.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(gate.node, format!("sub_{}_output_bit_{}", node.text_id, i));
                }
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::BitSlice { arg, start, width } => {
                let value_gates = env
                    .get_bit_vector(*arg)
                    .expect("bit_slice value should be present");
                let slice_gates = value_gates.get_lsb_slice(*start, *width);
                assert_eq!(slice_gates.get_bit_count(), *width);
                env.add(node_ref, GateOrVec::BitVector(slice_gates));
            }
            ir::NodePayload::ZeroExt { arg, new_bit_count } => {
                let arg_bits = env
                    .get_bit_vector(*arg)
                    .expect("zero_ext value should be present");
                let result_bits = gatify_zero_ext(*new_bit_count, &arg_bits);
                env.add(node_ref, GateOrVec::BitVector(result_bits));
            }
            ir::NodePayload::SignExt { arg, new_bit_count } => {
                let arg_bits = env
                    .get_bit_vector(*arg)
                    .expect("sign_ext value should be present");
                let result_bits =
                    gatify_sign_ext(g8_builder, node.text_id, *new_bit_count, &arg_bits);
                env.add(node_ref, GateOrVec::BitVector(result_bits));
            }
            ir::NodePayload::Decode { arg, width } => {
                assert_eq!(*width, node.ty.bit_count());
                let input_bits = env
                    .get_bit_vector(*arg)
                    .expect("decode arg should be present");
                let bits = gatify_decode(g8_builder, *width, &input_bits);
                assert_eq!(bits.get_bit_count(), *width);
                for (i, bit) in bits.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        bit.node,
                        format!("decode_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(bits));
            }
            ir::NodePayload::Encode { arg } => {
                log::debug!("gatifying encode; ty: {}", node.ty);
                let arg_bits = env
                    .get_bit_vector(*arg)
                    .expect("encode arg should be present");
                let result_bits = gatify_encode(g8_builder, node.ty.bit_count(), &arg_bits);
                assert_eq!(result_bits.get_bit_count(), node.ty.bit_count());
                for (i, gate) in result_bits.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        gate.node,
                        format!("encode_{}_output_bit_{}", node.text_id, i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(result_bits));
            }
            ir::NodePayload::Binop(ir::Binop::Shrl, arg, amount) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("shrl arg should be present");
                let amount_gates = env
                    .get_bit_vector(*amount)
                    .expect("shrl amount should be present");
                let result_gates = gatify_barrel_shifter(
                    &arg_gates,
                    &amount_gates,
                    Direction::Right,
                    &format!("shrl_{}", node.text_id),
                    g8_builder,
                );
                env.add(node_ref, GateOrVec::BitVector(result_gates));
            }
            ir::NodePayload::Binop(ir::Binop::Shra, arg, amount) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("shra arg should be present");
                let amount_gates = env
                    .get_bit_vector(*amount)
                    .expect("shra amount should be present");

                let result = gatify_shra(g8_builder, &arg_gates, &amount_gates, node.text_id);
                env.add(node_ref, GateOrVec::BitVector(result));
            }
            ir::NodePayload::Binop(ir::Binop::Shll, arg, amount) => {
                let arg_gates = env
                    .get_bit_vector(*arg)
                    .expect("shll arg should be present");
                let amount_gates = env
                    .get_bit_vector(*amount)
                    .expect("shll amount should be present");
                let result_gates = gatify_barrel_shifter(
                    &arg_gates,
                    &amount_gates,
                    Direction::Left,
                    &format!("shll_{}", node.text_id),
                    g8_builder,
                );
                env.add(node_ref, GateOrVec::BitVector(result_gates));
            }
            ir::NodePayload::OneHot { arg, lsb_prio } => {
                let bits = env
                    .get_bit_vector(*arg)
                    .expect("one_hot arg should be present");
                let bit_vector = gatify_one_hot(g8_builder, &bits, *lsb_prio);
                for (lsb_i, gate) in bit_vector.iter_lsb_to_msb().enumerate() {
                    g8_builder.add_tag(
                        gate.node,
                        format!("one_hot_{}_output_bit_{}", node.text_id, lsb_i),
                    );
                }
                env.add(node_ref, GateOrVec::BitVector(bit_vector));
            }
            ir::NodePayload::Binop(ir::Binop::Umul | ir::Binop::Smul, lhs, rhs) => {
                let output_bit_count = node.ty.bit_count();
                let lhs_bits = env.get_bit_vector(*lhs).expect("mul lhs should be present");
                let rhs_bits = env.get_bit_vector(*rhs).expect("mul rhs should be present");
                let signedness =
                    if matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Smul, ..)) {
                        Signedness::Signed
                    } else {
                        Signedness::Unsigned
                    };
                let gates = gatify_mul(
                    &lhs_bits,
                    &rhs_bits,
                    output_bit_count,
                    signedness,
                    g8_builder,
                );
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Binop(ir::Binop::Udiv | ir::Binop::Sdiv, lhs, rhs) => {
                let lhs_bits = env.get_bit_vector(*lhs).expect("div lhs should be present");
                let rhs_bits = env.get_bit_vector(*rhs).expect("div rhs should be present");
                let signedness =
                    if matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Sdiv, ..)) {
                        Signedness::Signed
                    } else {
                        Signedness::Unsigned
                    };
                let gates = gatify_div(&lhs_bits, &rhs_bits, signedness, g8_builder);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Binop(ir::Binop::Umod | ir::Binop::Smod, lhs, rhs) => {
                let lhs_bits = env.get_bit_vector(*lhs).expect("mod lhs should be present");
                let rhs_bits = env.get_bit_vector(*rhs).expect("mod rhs should be present");
                let signedness =
                    if matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Smod, ..)) {
                        Signedness::Signed
                    } else {
                        Signedness::Unsigned
                    };
                let gates = gatify_mod(&lhs_bits, &rhs_bits, signedness, g8_builder);
                env.add(node_ref, GateOrVec::BitVector(gates));
            }
            ir::NodePayload::Assert { .. }
            | ir::NodePayload::AfterAll(..)
            | ir::NodePayload::Trace { .. }
            | ir::NodePayload::Nil => {
                // These IR nodes manipulate tokens or have no semantic
                // representation in gates. Map them to a zero-width bit
                // vector so any subsequent references (e.g. via tuples) have
                // a placeholder value.
                env.add(node_ref, GateOrVec::BitVector(AigBitVector::zeros(0)));
            }
            ir::NodePayload::DynamicBitSlice { arg, start, width } => {
                let arg_bits = env
                    .get_bit_vector(*arg)
                    .expect("DynamicBitSlice arg should be present");
                let start_bits = env
                    .get_bit_vector(*start)
                    .expect("DynamicBitSlice start should be present");
                let shifted_bits = gatify_barrel_shifter(
                    &arg_bits,
                    &start_bits,
                    Direction::Right,
                    &format!("dynamic_bit_slice_shift_{}", node.text_id),
                    g8_builder,
                );
                // After shifting right by 'start', the desired bits are the LSBs.
                let result_bits = shifted_bits.get_lsb_slice(0, *width);
                assert_eq!(result_bits.get_bit_count(), *width);
                env.add(node_ref, GateOrVec::BitVector(result_bits));
            }
            ir::NodePayload::BitSliceUpdate {
                arg,
                start,
                update_value,
            } => {
                let arg_bits = env
                    .get_bit_vector(*arg)
                    .expect("BitSliceUpdate arg should be present");
                let start_bits = env
                    .get_bit_vector(*start)
                    .expect("BitSliceUpdate start should be present");
                let update_bits = env
                    .get_bit_vector(*update_value)
                    .expect("BitSliceUpdate value should be present");

                let arg_width = arg_bits.get_bit_count();
                let update_width = update_bits.get_bit_count();

                // Effective number of bits that can be written into `arg_bits`.
                // Any portion of `update_bits` that would extend beyond the
                // destination width is silently truncated (mirrors XLS
                // semantics).
                let effective_update_width = std::cmp::min(update_width, arg_width);

                // -----------------------------------------------------------------
                // Build the write mask (ones shifted by `start`).
                // -----------------------------------------------------------------
                let ones_effective =
                    g8_builder.replicate(g8_builder.get_true(), effective_update_width);

                // Extend the mask to the full destination width by prepending zeros
                // (MSBs) so the LSbs align with the slice we intend to write.
                let zeros_high_count = arg_width - effective_update_width;
                let ones_ext = if zeros_high_count == 0 {
                    ones_effective.clone()
                } else {
                    let zeros = AigBitVector::zeros(zeros_high_count);
                    AigBitVector::concat(zeros, ones_effective)
                };

                let mask = gatify_barrel_shifter(
                    &ones_ext,
                    &start_bits,
                    Direction::Left,
                    &format!("bit_slice_update_mask_{}", node.text_id),
                    g8_builder,
                );

                // -----------------------------------------------------------------
                // Prepare the update value (truncate or zero-extend to dest width),
                // then shift it into position.
                // -----------------------------------------------------------------
                let update_trim = if update_width > effective_update_width {
                    // Take only the LSbs that fit into the destination.
                    update_bits.get_lsb_slice(0, effective_update_width)
                } else {
                    update_bits.clone()
                };

                let update_ext = if zeros_high_count == 0 {
                    update_trim.clone()
                } else {
                    let zeros = AigBitVector::zeros(zeros_high_count);
                    AigBitVector::concat(zeros, update_trim)
                };

                let update_shifted = gatify_barrel_shifter(
                    &update_ext,
                    &start_bits,
                    Direction::Left,
                    &format!("bit_slice_update_value_{}", node.text_id),
                    g8_builder,
                );

                // -----------------------------------------------------------------
                // Combine original value with the updated slice.
                // -----------------------------------------------------------------
                let mask_not = g8_builder.add_not_vec(&mask);
                let cleared = g8_builder.add_and_vec(&arg_bits, &mask_not);
                let inserted = g8_builder.add_and_vec(&update_shifted, &mask);
                let result_bits = g8_builder.add_or_vec(&cleared, &inserted);

                env.add(node_ref, GateOrVec::BitVector(result_bits));
            }
            _ => {
                todo!("Unsupported node payload {:?}", payload);
            }
        }
    }
    // Resolve the outputs and place them into the builder.
    let ret_node_ref = match f.ret_node_ref {
        Some(ret_node_ref) => ret_node_ref,
        None => {
            return;
        }
    };
    let gate_refs = env
        .get_bit_vector(ret_node_ref)
        .expect("return node should be present");
    assert_eq!(
        gate_refs.get_bit_count(),
        f.ret_ty.bit_count(),
        "return node bit count mismatch; expected: {} via return type {}, got: {}",
        f.ret_ty.bit_count(),
        f.ret_ty,
        gate_refs.get_bit_count()
    );
    g8_builder.add_output("output_value".to_string(), gate_refs);
}

pub struct GatifyOptions {
    pub fold: bool,
    pub hash: bool,
    pub check_equivalence: bool,
    pub adder_mapping: crate::ir2gate_utils::AdderMapping,
}

// Type alias for the lowering map
pub type IrToGateMap = HashMap<ir::NodeRef, AigBitVector>;

/// Holds the output of the `gatify` process.
#[derive(Debug)]
pub struct GatifyOutput {
    pub gate_fn: GateFn,
    pub lowering_map: IrToGateMap,
}

pub fn gatify(f: &ir::Fn, options: GatifyOptions) -> Result<GatifyOutput, String> {
    let mut g8_builder = GateBuilder::new(
        f.name.clone(),
        GateBuilderOptions {
            fold: options.fold,
            hash: options.hash,
        },
    );
    let mut env = GateEnv::new();
    gatify_internal(f, &mut g8_builder, &mut env, &options);
    let gate_fn = g8_builder.build();
    log::debug!(
        "converted IR function to gate function:\n{}",
        gate_fn.to_string()
    );

    // Convert the internal GateEnv map to the public IrToGateMap
    let mut lowering_map: IrToGateMap = HashMap::new();
    for (node_ref, gate_or_vec) in env.ir_to_g8.into_iter() {
        let bit_vector = match gate_or_vec {
            GateOrVec::BitVector(bv) => bv,
            GateOrVec::Gate(gate_ref) => AigBitVector::from_bit(gate_ref),
        };
        lowering_map.insert(node_ref, bit_vector);
    }

    // If we're told we should do so, we check equivalence between the original IR
    // function and the gate function that we converted it to.
    if options.check_equivalence {
        log::info!("checking equivalence of IR function and gate function...");
        check_equivalence::validate_same_fn(f, &gate_fn)?;
    }
    // Construct and return the GatifyOutput struct
    Ok(GatifyOutput {
        gate_fn,
        lowering_map,
    })
}

#[cfg(test)]
mod tests {
    use crate::xls_ir::ir;
    use crate::{ir2gate::GatifyOptions, ir2gate::gatify, xls_ir::ir_parser};

    #[test]
    fn test_gatify_array_literal_flattening() {
        let ir_text = "fn f() -> bits[8][5] {\n  ret literal.1: bits[8][5] = literal(value=[1, 2, 3, 4, 5], id=1)\n}";
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_fn = parser.parse_fn().unwrap();

        let gatify_output = crate::ir2gate::gatify(
            &ir_fn,
            crate::ir2gate::GatifyOptions {
                fold: false,
                hash: false,
                check_equivalence: false,
                adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
            },
        )
        .unwrap();

        // The output should be a bit vector of 5 * 8 = 40 bits
        let output_vec = &gatify_output.gate_fn.outputs[0].bit_vector;
        assert_eq!(
            output_vec.get_bit_count(),
            40,
            "Expected 40 bits for bits[8][5] array literal"
        );
    }

    #[test]
    fn test_lowering_map_not_add() {
        let ir_text = "package sample
fn f(a: bits[8], b: bits[8]) -> bits[8] {
  not.3: bits[8] = not(a, id=3)
  ret add.4: bits[8] = add(not.3, b, id=4)
}
";
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_package().unwrap();
        let ir_fn = ir_package.get_top().unwrap();

        let gatify_output = gatify(
            &ir_fn,
            GatifyOptions {
                fold: true,               // Folding shouldn't affect this test
                check_equivalence: false, // Not needed for this map check
                hash: true,
                adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
            },
        )
        .unwrap();

        let lowering_map = gatify_output.lowering_map;

        // Find the 'Not' node and its input parameter reference within the IR function.
        let mut not_node_ref: Option<ir::NodeRef> = None;
        let mut param_ref: Option<ir::NodeRef> = None;

        for (i, node) in ir_fn.nodes.iter().enumerate() {
            if let ir::NodePayload::Unop(ir::Unop::Not, operand) = &node.payload {
                not_node_ref = Some(ir::NodeRef { index: i });
                param_ref = Some(*operand);
                break; // Assuming only one 'Not' operation in this simple test
                // case
            }
        }

        let not_ref = not_node_ref.expect("Could not find 'Not' node in IR function");
        let param_ref = param_ref.expect("Could not find input parameter for 'Not' node");

        // Retrieve AIG vectors from the map using the identified node references
        let param_vec = lowering_map
            .get(&param_ref)
            .expect("Lowering for parameter node not found");
        let not_vec = lowering_map
            .get(&not_ref)
            .expect("Lowering for 'Not' node not found");

        assert_eq!(param_vec.get_bit_count(), 8, "Param should be 8 bits");
        assert_eq!(not_vec.get_bit_count(), 8, "Node 'not' should be 8 bits");

        // Verify that each bit in not_vec is the negation of the corresponding bit in
        // param_vec
        for i in 0..8 {
            let param_op = param_vec.get_lsb(i);
            let not_op = not_vec.get_lsb(i);

            // The not operation should ideally result in operands pointing to the same
            // underlying node but with opposite negation flags.
            assert_eq!(param_op.node, not_op.node, "Bit {} nodes should match", i);
            assert_ne!(
                param_op.negated, not_op.negated,
                "Bit {} negation flags should differ",
                i
            );
        }
    }
}
