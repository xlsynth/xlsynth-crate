// SPDX-License-Identifier: Apache-2.0

//! Functionality for converting an IR function into a gate function via
//! `gatify`.

use crate::aig::gate::{AigBitVector, AigOperand, GateFn};
use crate::check_equivalence;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use std::collections::HashMap;
use std::sync::Arc;
use xlsynth_pir::ir::{self, ParamId, StartAndLimit};
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir_utils;
use xlsynth_pir::ir_validate;

use crate::ir2gate_utils::{
    AdderMapping, Direction, array_add_with_carry_out, gatify_add_brent_kung,
    gatify_add_kogge_stone, gatify_add_ripple_carry, gatify_barrel_shifter, gatify_one_hot,
    gatify_one_hot_select, gatify_one_hot_with_nonzero_flag, gatify_prio_encode,
};

use crate::gate_builder::ReductionKind;

#[derive(Debug)]
enum GateOrVec {
    Gate(AigOperand),
    BitVector(AigBitVector),
}

fn get_known_zero_bit_indices_for_selector(
    options: &GatifyOptions,
    f: &ir::Fn,
    selector_node_ref: ir::NodeRef,
) -> Vec<usize> {
    let range_info = match options.range_info.as_ref() {
        Some(ri) => ri,
        None => return vec![],
    };
    let selector_text_id = f.get_node(selector_node_ref).text_id;
    let info = match range_info.get(selector_text_id) {
        Some(i) => i,
        None => return vec![],
    };
    let known = match info.known_bits.as_ref() {
        Some(k) => k,
        None => return vec![],
    };

    let mask = &known.mask;
    let value = &known.value;
    let bit_count = mask.get_bit_count();
    assert_eq!(
        bit_count,
        value.get_bit_count(),
        "known mask/value bit count mismatch for selector text_id={}",
        selector_text_id
    );

    let mut known_zero_bits: Vec<usize> = Vec::new();
    for i in 0..bit_count {
        let is_known = mask.get_bit(i).unwrap_or(false);
        if !is_known {
            continue;
        }
        let is_one = value.get_bit(i).unwrap_or(false);
        if !is_one {
            known_zero_bits.push(i);
        }
    }
    known_zero_bits
}

fn get_impossible_in_bounds_sel_case_indices(
    options: &GatifyOptions,
    f: &ir::Fn,
    selector_node_ref: ir::NodeRef,
    cases_len: usize,
) -> Vec<usize> {
    let range_info = match options.range_info.as_ref() {
        Some(ri) => ri,
        None => return vec![],
    };
    let selector_text_id = f.get_node(selector_node_ref).text_id;
    let info = match range_info.get(selector_text_id) {
        Some(i) => i,
        None => return vec![],
    };
    let intervals = match info.intervals.as_ref() {
        Some(v) => v,
        None => return vec![],
    };

    let selector_ty = f.get_node_ty(selector_node_ref);
    let width = selector_ty.bit_count();

    let mut impossible: Vec<usize> = Vec::new();
    for i in 0..cases_len {
        let value_bits = xlsynth::IrBits::make_ubits(width, i as u64).unwrap();
        let mut possible = false;
        for it in intervals {
            if it.lo.ule(&value_bits) && value_bits.ule(&it.hi) {
                possible = true;
                break;
            }
        }
        if !possible {
            impossible.push(i);
        }
    }
    impossible
}

fn maybe_warn_shift_amount_truncatable(
    range_info: Option<&Arc<IrRangeInfo>>,
    amount_text_id: usize,
    shift_bound: usize,
    amount_bits: &AigBitVector,
) {
    let range_info = match range_info {
        Some(ri) => ri,
        None => return,
    };
    if shift_bound == 0 {
        return;
    }
    if !range_info.proves_ult(amount_text_id, shift_bound) {
        return;
    }
    let max_effective_bits =
        match range_info.effective_amount_bits_for_ult(amount_text_id, shift_bound) {
            Some(v) => v,
            None => return,
        };
    let required_bits = if shift_bound == 1 {
        0
    } else {
        xlsynth_pir::math::ceil_log2(shift_bound)
    };
    let effective_bits = std::cmp::min(required_bits, max_effective_bits);
    if effective_bits < amount_bits.get_bit_count() {
        log::warn!(
            "shift amount text_id={} is truncatable in consumer: shift_bound={} amount_bits={} effective_bits={} (max_effective_bits={} required_bits={}); expected upstream optimizer to slice",
            amount_text_id,
            shift_bound,
            amount_bits.get_bit_count(),
            effective_bits,
            max_effective_bits,
            required_bits
        );
    }
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

fn gatify_add_with_mapping(
    adder_mapping: AdderMapping,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    c_in: AigOperand,
    tag: Option<&str>,
    gb: &mut GateBuilder,
) -> (AigOperand, AigBitVector) {
    match adder_mapping {
        AdderMapping::RippleCarry => gatify_add_ripple_carry(lhs_bits, rhs_bits, c_in, tag, gb),
        AdderMapping::BrentKung => gatify_add_brent_kung(lhs_bits, rhs_bits, c_in, tag, gb),
        AdderMapping::KoggeStone => gatify_add_kogge_stone(lhs_bits, rhs_bits, c_in, tag, gb),
    }
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
    for case_bits in cases.iter() {
        assert_eq!(
            case_bits.get_bit_count(),
            output_bit_count,
            "all cases of the priority select must have the same bit count which is the same as the output bit count"
        );
    }

    // Binary mux form we just emit as a single binary mux.
    if cases.len() == 1 && default_bits.is_some() {
        assert_eq!(selector_bits.get_bit_count(), 1);
        let selector = selector_bits.get_lsb(0);
        // Note: if all the selector bits are zero we use the default bit value.
        return gb.add_mux2_vec(
            selector,
            /* on_true= */ &cases[0],
            /* on_false= */ &default_bits.unwrap(),
        );
    }

    // For small output widths and a present default, a mux-chain version tends
    // to have a better depth profile than the masking+OR form, and can also win
    // on size at W=1 (see the table sweep unit test).
    if output_bit_count <= 3 && default_bits.is_some() && !cases.is_empty() {
        return gatify_priority_sel_mux_chain(gb, selector_bits, cases, default_bits.unwrap());
    }

    gatify_priority_sel_masking(gb, output_bit_count, selector_bits, cases, default_bits)
}

fn gatify_priority_sel_mux_chain(
    gb: &mut GateBuilder,
    selector_bits: AigBitVector,
    cases: &[AigBitVector],
    default_bits: AigBitVector,
) -> AigBitVector {
    // For selector bits s0..s{n-1} (LSB-first priority) and cases c0..c{n-1}:
    //   result = mux(s0, c0, mux(s1, c1, ... mux(s{n-1}, c{n-1}, default)...))
    let mut acc = default_bits;
    for i in (0..cases.len()).rev() {
        let s_i = selector_bits.get_lsb(i);
        acc = gb.add_mux2_vec(s_i, &cases[i], &acc);
    }
    acc
}

fn gatify_priority_sel_masking(
    gb: &mut GateBuilder,
    output_bit_count: usize,
    selector_bits: AigBitVector,
    cases: &[AigBitVector],
    default_bits: Option<AigBitVector>,
) -> AigBitVector {
    let mut masked_cases = vec![];
    // As we process cases we track whether any prior case had been selected.
    let mut any_prior_selected = gb.get_false();
    for (i, case_bits) in cases.iter().enumerate() {
        let this_wants_selected = selector_bits.get_lsb(i).clone();
        let no_prior_selected = gb.add_not(any_prior_selected);
        let this_selected = gb.add_and_binary(this_wants_selected, no_prior_selected);
        any_prior_selected = gb.add_or_binary(any_prior_selected, this_selected);

        let mask = gb.replicate(this_selected, output_bit_count);
        let masked = gb.add_and_vec(&mask, case_bits);
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

fn gatify_array_slice(
    gb: &mut GateBuilder,
    array_ty: &ir::ArrayTypeData,
    array_bits: &AigBitVector,
    start_bits: &AigBitVector,
    assumed_start_in_bounds: bool,
    width: usize,
    text_id: usize,
    mul_adder_mapping: AdderMapping,
) -> AigBitVector {
    let e_bits = array_ty.element_type.bit_count();
    let n_elems = array_ty.element_count;

    // Clamp start to the last valid index to ensure OOB semantics replicate the
    // last element, even when the start index is larger than
    // (n_elems - 1 + width - 1).
    let start_w = start_bits.get_bit_count();
    let clamped_start_bits = if assumed_start_in_bounds {
        start_bits.clone()
    } else {
        let last_idx_bits = gb.add_literal(
            &xlsynth::IrBits::make_ubits(start_w, (n_elems.saturating_sub(1)) as u64).unwrap(),
        );
        let start_le_last = gatify_ule_via_bit_tests(gb, text_id, start_bits, &last_idx_bits);
        gb.add_mux2_vec(&start_le_last, start_bits, &last_idx_bits)
    };

    // 1) Build a padding prefix of (width-1) copies of the last element to emulate
    // XLS out-of-bounds semantics (select last element when OOB).
    let last_elem = array_bits.get_lsb_slice((n_elems - 1) * e_bits, e_bits);
    let mut pad = AigBitVector::zeros(0);
    if width > 0 {
        for _ in 0..(width - 1) {
            pad = AigBitVector::concat(last_elem.clone(), pad);
        }
    }

    // 2) Concatenate pad || array_bits to form the extended sequence.
    let extended = AigBitVector::concat(pad, array_bits.clone());

    // 3) Compute start_scaled = clamped_start * e_bits, with sufficient width to
    //    hold the product.
    let mut tmp = if e_bits > 0 { e_bits - 1 } else { 0 };
    let mut extra = 0usize;
    while tmp > 0 {
        extra += 1;
        tmp >>= 1;
    }
    let start_scaled_w = start_w + extra;
    let e_const =
        gb.add_literal(&xlsynth::IrBits::make_ubits(start_scaled_w, e_bits as u64).unwrap());
    let start_ext = if start_scaled_w > start_w {
        let zeros = AigBitVector::zeros(start_scaled_w - start_w);
        AigBitVector::concat(zeros, clamped_start_bits.clone())
    } else {
        clamped_start_bits.clone()
    };
    let e_ext = if start_scaled_w > start_w {
        let zeros = AigBitVector::zeros(start_scaled_w - start_w);
        AigBitVector::concat(zeros, e_const)
    } else {
        e_const
    };
    let start_scaled = gatify_umul(&start_ext, &e_ext, start_scaled_w, mul_adder_mapping, gb);

    // 4) Shift right by start_scaled and take low (width * e_bits) bits.
    let shifted = gatify_barrel_shifter(
        &extended,
        &start_scaled,
        Direction::Right,
        &format!("array_slice_shift_{}", text_id),
        gb,
    );
    let out_width_bits = width * e_bits;
    shifted.get_lsb_slice(0, out_width_bits)
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

    if case_count == 2 && default_bits.is_none() {
        assert_eq!(selector_bits.get_bit_count(), 1);
        let selector = selector_bits.get_lsb(0);
        return gb.add_mux2_vec(
            selector, /* on_true= */ &cases[1], /* on_false= */ &cases[0],
        );
    }

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
    mul_adder_mapping: AdderMapping,
    gb: &mut GateBuilder,
) -> AigBitVector {
    match signedness {
        Signedness::Unsigned => {
            gatify_umul(lhs_bits, rhs_bits, output_bit_count, mul_adder_mapping, gb)
        }
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
            gatify_umul(&lhs_ext, &rhs_ext, output_bit_count, mul_adder_mapping, gb)
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
    mul_adder_mapping: AdderMapping,
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

    // Canonicalize operand order so gate count / depth is insensitive to the
    // caller's operand order. We prefer using the narrower operand as the
    // multiplier (the number of partial-product rows) since it generally leads
    // to smaller and shallower circuits.
    let (multiplicand_bits, multiplier_bits) = if rhs_bit_count <= lhs_bit_count {
        (lhs_bits, rhs_bits)
    } else {
        (rhs_bits, lhs_bits)
    };

    let mut partial_products = Vec::new();

    // For each bit in the multiplier, generate a scaled partial product.
    for (i, mul_bit) in multiplier_bits.iter_lsb_to_msb().enumerate() {
        let mut row = Vec::new();
        for mcand_bit in multiplicand_bits.iter_lsb_to_msb() {
            let pp = gb.add_and_binary(*mcand_bit, *mul_bit);
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
    array_add_with_carry_out(gb, &partial_products, None, mul_adder_mapping).sum
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

fn gatify_ult_and_eq_via_bit_tests(
    gb: &mut GateBuilder,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
) -> (AigOperand, AigOperand) {
    assert_eq!(lhs_bits.get_bit_count(), rhs_bits.get_bit_count());
    let input_bit_count = lhs_bits.get_bit_count();
    assert!(input_bit_count > 0);

    // Compute the XNOR bits once so both lt and eq can share them.
    let eq_bits = gb.add_xnor_vec(lhs_bits, rhs_bits);
    let eq = gb.add_and_reduce(&eq_bits, ReductionKind::Tree);

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

        // rhs larger at this bit: (!lhs_bit) & rhs_bit.
        let lhs_bit_unset = gb.add_not(*lhs_bit);
        let rhs_bit_set = *rhs_bit;
        let rhs_larger_this_bit = gb.add_and_binary(lhs_bit_unset, rhs_bit_set);

        let bit = gb.add_and_binary(rhs_larger_this_bit, prior_bits_equal);
        bit_tests.push(bit);
    }
    let lt = gb.add_or_nary(&bit_tests, ReductionKind::Tree);
    (lt, eq)
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

fn literal_bits_if_bits_node(f: &ir::Fn, node_ref: ir::NodeRef) -> Option<xlsynth::IrBits> {
    match &f.get_node(node_ref).payload {
        ir::NodePayload::Literal(literal) => literal.to_bits().ok(),
        _ => None,
    }
}

fn is_all_zeros(bits: &xlsynth::IrBits) -> bool {
    for i in 0..bits.get_bit_count() {
        if bits.get_bit(i).unwrap() {
            return false;
        }
    }
    true
}

fn is_all_ones(bits: &xlsynth::IrBits) -> bool {
    for i in 0..bits.get_bit_count() {
        if !bits.get_bit(i).unwrap() {
            return false;
        }
    }
    true
}

fn get_pow2_lsb_index(bits: &xlsynth::IrBits) -> Option<usize> {
    // Recognizes non-zero values with exactly one bit set.
    let mut found: Option<usize> = None;
    for i in 0..bits.get_bit_count() {
        let bit = bits.get_bit(i).unwrap();
        if bit {
            if found.is_some() {
                return None;
            }
            found = Some(i);
        }
    }
    found
}

fn get_pow2_minus1_k(bits: &xlsynth::IrBits) -> Option<usize> {
    // Recognizes values of the form (1<<k)-1, i.e. k low bits are 1 and the rest
    // are 0. k=0 => 0, k=bit_count => all ones.
    let bit_count = bits.get_bit_count();
    let mut k = 0usize;
    while k < bit_count && bits.get_bit(k).unwrap() {
        k += 1;
    }
    for i in k..bit_count {
        if bits.get_bit(i).unwrap() {
            return None;
        }
    }
    Some(k)
}

fn get_neg_pow2_k(bits: &xlsynth::IrBits) -> Option<usize> {
    // Recognizes values of the form -2^k in two's complement, i.e. high bits are
    // 1 and the low k bits are 0. k=0 => all ones, k=bit_count-1 => int_min.
    let bit_count = bits.get_bit_count();
    assert!(bit_count > 0);
    if !bits.get_bit(bit_count - 1).unwrap() {
        return None;
    }
    let mut k = 0usize;
    while k < bit_count && !bits.get_bit(k).unwrap() {
        k += 1;
    }
    if k == 0 || k == bit_count - 1 {
        return None;
    }
    for i in k..bit_count {
        if !bits.get_bit(i).unwrap() {
            return None;
        }
    }
    Some(k)
}

fn simplify_ugt_all_ones_above_k(
    gb: &mut GateBuilder,
    lhs_bits: &AigBitVector,
    bit_count: usize,
    k: usize,
) -> AigOperand {
    assert!(k > 0 && k < bit_count);
    let upper = lhs_bits.get_lsb_slice(k, bit_count - k);
    let upper_is_all_ones = gb.add_and_reduce(&upper, ReductionKind::Tree);
    let low = lhs_bits.get_lsb_slice(0, k);
    let low_is_nonzero = gb.add_or_reduce(&low, ReductionKind::Tree);
    gb.add_and_binary(upper_is_all_ones, low_is_nonzero)
}

fn is_int_min(bits: &xlsynth::IrBits) -> bool {
    let bit_count = bits.get_bit_count();
    assert!(bit_count > 0);
    if !bits.get_bit(bit_count - 1).unwrap() {
        return false;
    }
    for i in 0..(bit_count - 1) {
        if bits.get_bit(i).unwrap() {
            return false;
        }
    }
    true
}

fn is_int_max(bits: &xlsynth::IrBits) -> bool {
    let bit_count = bits.get_bit_count();
    assert!(bit_count > 0);
    if bits.get_bit(bit_count - 1).unwrap() {
        return false;
    }
    for i in 0..(bit_count - 1) {
        if !bits.get_bit(i).unwrap() {
            return false;
        }
    }
    true
}

fn is_non_negative_signed(bits: &xlsynth::IrBits) -> bool {
    let bit_count = bits.get_bit_count();
    assert!(bit_count > 0);
    !bits.get_bit(bit_count - 1).unwrap()
}

fn commute_cmp_binop(binop: ir::Binop) -> Option<ir::Binop> {
    match binop {
        ir::Binop::Eq => Some(ir::Binop::Eq),
        ir::Binop::Ne => Some(ir::Binop::Ne),

        ir::Binop::Ult => Some(ir::Binop::Ugt),
        ir::Binop::Ule => Some(ir::Binop::Uge),
        ir::Binop::Ugt => Some(ir::Binop::Ult),
        ir::Binop::Uge => Some(ir::Binop::Ule),

        ir::Binop::Slt => Some(ir::Binop::Sgt),
        ir::Binop::Sle => Some(ir::Binop::Sge),
        ir::Binop::Sgt => Some(ir::Binop::Slt),
        ir::Binop::Sge => Some(ir::Binop::Sle),

        _ => None,
    }
}

struct NormalizedCmpLiteralRhs {
    binop: ir::Binop,
    lhs: ir::NodeRef,
    rhs: ir::NodeRef, // literal node
    rhs_bits: xlsynth::IrBits,
}

fn normalize_cmp_literal_rhs(
    f: &ir::Fn,
    binop: ir::Binop,
    a: ir::NodeRef,
    b: ir::NodeRef,
) -> Option<NormalizedCmpLiteralRhs> {
    let b_lit = literal_bits_if_bits_node(f, b);
    let a_lit = literal_bits_if_bits_node(f, a);

    match (a_lit, b_lit) {
        (None, Some(rhs_bits)) => Some(NormalizedCmpLiteralRhs {
            binop,
            lhs: a,
            rhs: b,
            rhs_bits,
        }),
        (Some(rhs_bits), None) => {
            let binop = commute_cmp_binop(binop)?;
            Some(NormalizedCmpLiteralRhs {
                binop,
                lhs: b,
                rhs: a,
                rhs_bits,
            })
        }
        // If both are literals, we expect folding to have handled it already.
        (Some(_), Some(_)) => None,
        (None, None) => None,
    }
}

fn try_simplify_cmp_literal_rhs(
    gb: &mut GateBuilder,
    binop: ir::Binop,
    lhs_bits: &AigBitVector,
    rhs_bits_vec: &AigBitVector,
    rhs_bits: &xlsynth::IrBits,
) -> Option<AigOperand> {
    assert_eq!(lhs_bits.get_bit_count(), rhs_bits_vec.get_bit_count());
    if lhs_bits.get_bit_count() == 0 {
        return None;
    }
    let bit_count = lhs_bits.get_bit_count();
    assert_eq!(rhs_bits.get_bit_count(), bit_count);

    let rhs_is_zero = is_all_zeros(rhs_bits);
    let rhs_is_all_ones = is_all_ones(rhs_bits);

    match binop {
        // unsigned comparisons
        ir::Binop::Ult => {
            if rhs_is_zero {
                Some(gb.get_false())
            } else if rhs_is_all_ones {
                Some(gb.add_ne_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree))
            } else if let Some(k) = get_pow2_lsb_index(rhs_bits) {
                let slice = lhs_bits.get_lsb_slice(k, bit_count - k);
                Some(gb.add_ez(&slice, ReductionKind::Tree))
            } else if let Some(k) = get_pow2_minus1_k(rhs_bits) {
                // x < (1<<k)-1  iff  upper_bits == 0  AND  low_bits != all_ones
                //
                // When k==0 => rhs==0, handled above. When k==bit_count => rhs==all_ones,
                // handled above.
                assert!(k > 0 && k < bit_count);
                let upper = lhs_bits.get_lsb_slice(k, bit_count - k);
                let upper_is_zero = gb.add_ez(&upper, ReductionKind::Tree);
                let low = lhs_bits.get_lsb_slice(0, k);
                let low_is_all_ones = gb.add_and_reduce(&low, ReductionKind::Tree);
                let low_is_not_all_ones = gb.add_not(low_is_all_ones);
                Some(gb.add_and_binary(upper_is_zero, low_is_not_all_ones))
            } else {
                None
            }
        }
        ir::Binop::Ule => {
            if rhs_is_zero {
                Some(gb.add_eq_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree))
            } else if rhs_is_all_ones {
                Some(gb.get_true())
            } else if let Some(k) = get_pow2_lsb_index(rhs_bits) {
                assert!(k < bit_count);
                // For small k, the (upper==0) & (!bit_k | low==0) form is cheaper than building
                // a full equality check. For larger k, this form adds extra
                // reduction depth, so we use (lt | eq) instead.
                if k <= 4 {
                    // x <= (1<<k) iff upper bits above k are zero and (bit_k is 0 or low bits are
                    // zero).
                    //
                    // This captures the fact that with upper bits = 0, values are in [0, 2^(k+1)-1]
                    // and the only disallowed case is bit_k=1 with any lower bit set.
                    let upper = lhs_bits.get_lsb_slice(k + 1, bit_count.saturating_sub(k + 1));
                    let upper_is_zero = if upper.get_bit_count() == 0 {
                        gb.get_true()
                    } else {
                        gb.add_ez(&upper, ReductionKind::Tree)
                    };
                    let bit_k = *lhs_bits.get_lsb(k);
                    let low = lhs_bits.get_lsb_slice(0, k);
                    let low_is_zero = if low.get_bit_count() == 0 {
                        gb.get_true()
                    } else {
                        gb.add_ez(&low, ReductionKind::Tree)
                    };
                    let not_bit_k = gb.add_not(bit_k);
                    let cond = gb.add_or_binary(not_bit_k, low_is_zero);
                    Some(gb.add_and_binary(upper_is_zero, cond))
                } else {
                    // x <= (1<<k)  iff  (x < (1<<k)) OR (x == (1<<k))
                    let lt = {
                        let slice = lhs_bits.get_lsb_slice(k, bit_count - k);
                        gb.add_ez(&slice, ReductionKind::Tree)
                    };
                    let eq = gb.add_eq_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree);
                    Some(gb.add_or_binary(lt, eq))
                }
            } else if let Some(k) = get_pow2_minus1_k(rhs_bits) {
                let slice = lhs_bits.get_lsb_slice(k, bit_count - k);
                Some(gb.add_ez(&slice, ReductionKind::Tree))
            } else {
                None
            }
        }
        ir::Binop::Ugt => {
            if rhs_is_zero {
                Some(gb.add_ne_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree))
            } else if rhs_is_all_ones {
                Some(gb.get_false())
            } else if let Some(k) = get_pow2_lsb_index(rhs_bits) {
                // x > (1<<k) iff (upper bits above k are non-zero) OR (bit_k is 1 AND low bits
                // are non-zero).
                assert!(k < bit_count);
                let upper = lhs_bits.get_lsb_slice(k + 1, bit_count.saturating_sub(k + 1));
                let upper_is_nonzero = if upper.get_bit_count() == 0 {
                    gb.get_false()
                } else {
                    gb.add_or_reduce(&upper, ReductionKind::Tree)
                };

                let bit_k = *lhs_bits.get_lsb(k);
                let low = lhs_bits.get_lsb_slice(0, k);
                let low_is_nonzero = if low.get_bit_count() == 0 {
                    gb.get_false()
                } else {
                    gb.add_or_reduce(&low, ReductionKind::Tree)
                };
                let term2 = gb.add_and_binary(bit_k, low_is_nonzero);
                Some(gb.add_or_binary(upper_is_nonzero, term2))
            } else if let Some(k) = get_pow2_minus1_k(rhs_bits) {
                let slice = lhs_bits.get_lsb_slice(k, bit_count - k);
                let lt_pow2 = gb.add_ez(&slice, ReductionKind::Tree);
                Some(gb.add_not(lt_pow2))
            } else {
                None
            }
        }
        ir::Binop::Uge => {
            if rhs_is_zero {
                Some(gb.get_true())
            } else if rhs_is_all_ones {
                Some(gb.add_eq_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree))
            } else if let Some(k) = get_pow2_lsb_index(rhs_bits) {
                let slice = lhs_bits.get_lsb_slice(k, bit_count - k);
                let lt_pow2 = gb.add_ez(&slice, ReductionKind::Tree);
                Some(gb.add_not(lt_pow2))
            } else if let Some(k) = get_pow2_minus1_k(rhs_bits) {
                // x >= (1<<k)-1  iff  (upper_bits != 0) OR (low_bits == all_ones)
                //
                // When k==0 => rhs==0, handled above. When k==bit_count => rhs==all_ones,
                // handled above.
                assert!(k > 0 && k < bit_count);
                let upper = lhs_bits.get_lsb_slice(k, bit_count - k);
                let upper_is_zero = gb.add_ez(&upper, ReductionKind::Tree);
                let upper_is_nonzero = gb.add_not(upper_is_zero);
                let low = lhs_bits.get_lsb_slice(0, k);
                let low_is_all_ones = gb.add_and_reduce(&low, ReductionKind::Tree);
                Some(gb.add_or_binary(upper_is_nonzero, low_is_all_ones))
            } else {
                None
            }
        }

        // signed comparisons
        ir::Binop::Slt => {
            let msb = *lhs_bits.get_msb(0);
            if rhs_is_zero {
                Some(msb)
            } else if rhs_is_all_ones {
                let eq = gb.add_eq_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree);
                let ne = gb.add_not(eq);
                Some(gb.add_and_binary(msb, ne))
            } else if is_int_min(rhs_bits) {
                Some(gb.get_false())
            } else if is_int_max(rhs_bits) {
                Some(gb.add_ne_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree))
            } else if is_non_negative_signed(rhs_bits) {
                let u = try_simplify_cmp_literal_rhs(
                    gb,
                    ir::Binop::Ult,
                    lhs_bits,
                    rhs_bits_vec,
                    rhs_bits,
                )
                .unwrap_or_else(|| {
                    gatify_ucmp_fallback(gb, 0, ir::Binop::Ult, lhs_bits, rhs_bits_vec)
                });
                // If rhs is non-negative, then for negative lhs (msb==1) the unsigned
                // comparison is necessarily false, so we can drop the nonneg
                // guard.
                Some(gb.add_or_binary(msb, u))
            } else {
                None
            }
        }
        ir::Binop::Sle => {
            let msb = *lhs_bits.get_msb(0);
            if rhs_is_zero {
                let eq = gb.add_eq_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree);
                Some(gb.add_or_binary(msb, eq))
            } else if rhs_is_all_ones {
                Some(msb)
            } else if is_int_min(rhs_bits) {
                Some(gb.add_eq_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree))
            } else if is_int_max(rhs_bits) {
                Some(gb.get_true())
            } else if is_non_negative_signed(rhs_bits) {
                let u = try_simplify_cmp_literal_rhs(
                    gb,
                    ir::Binop::Ule,
                    lhs_bits,
                    rhs_bits_vec,
                    rhs_bits,
                )
                .unwrap_or_else(|| {
                    gatify_ucmp_fallback(gb, 0, ir::Binop::Ule, lhs_bits, rhs_bits_vec)
                });
                // If rhs is non-negative, then for negative lhs (msb==1) the unsigned
                // comparison is necessarily false, so we can drop the nonneg
                // guard.
                Some(gb.add_or_binary(msb, u))
            } else {
                None
            }
        }
        ir::Binop::Sgt => {
            let msb = *lhs_bits.get_msb(0);
            if rhs_is_zero {
                let eq0 = gb.add_eq_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree);
                let ne0 = gb.add_not(eq0);
                let nonneg = gb.add_not(msb);
                Some(gb.add_and_binary(nonneg, ne0))
            } else if rhs_is_all_ones {
                Some(gb.add_not(msb))
            } else if is_int_min(rhs_bits) {
                Some(gb.add_ne_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree))
            } else if is_int_max(rhs_bits) {
                Some(gb.get_false())
            } else if let Some(k) = get_neg_pow2_k(rhs_bits) {
                let nonneg = gb.add_not(msb);
                let u = simplify_ugt_all_ones_above_k(gb, lhs_bits, bit_count, k);
                Some(gb.add_or_binary(nonneg, u))
            } else if is_non_negative_signed(rhs_bits) {
                let nonneg = gb.add_not(msb);
                let u = try_simplify_cmp_literal_rhs(
                    gb,
                    ir::Binop::Ugt,
                    lhs_bits,
                    rhs_bits_vec,
                    rhs_bits,
                )
                .unwrap_or_else(|| {
                    gatify_ucmp_fallback(gb, 0, ir::Binop::Ugt, lhs_bits, rhs_bits_vec)
                });
                Some(gb.add_and_binary(nonneg, u))
            } else {
                None
            }
        }
        ir::Binop::Sge => {
            let msb = *lhs_bits.get_msb(0);
            if rhs_is_zero {
                Some(gb.add_not(msb))
            } else if rhs_is_all_ones {
                let eq = gb.add_eq_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree);
                let nonneg = gb.add_not(msb);
                Some(gb.add_or_binary(nonneg, eq))
            } else if is_int_min(rhs_bits) {
                Some(gb.get_true())
            } else if is_int_max(rhs_bits) {
                Some(gb.add_eq_vec(lhs_bits, rhs_bits_vec, ReductionKind::Tree))
            } else if is_non_negative_signed(rhs_bits) {
                let nonneg = gb.add_not(msb);
                let u = try_simplify_cmp_literal_rhs(
                    gb,
                    ir::Binop::Uge,
                    lhs_bits,
                    rhs_bits_vec,
                    rhs_bits,
                )
                .unwrap_or_else(|| {
                    gatify_ucmp_fallback(gb, 0, ir::Binop::Uge, lhs_bits, rhs_bits_vec)
                });
                Some(gb.add_and_binary(nonneg, u))
            } else {
                None
            }
        }

        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
struct UcmpConstResult {
    lt: AigOperand,
    gt: AigOperand,
    eq: AigOperand,
}

fn gatify_eq_literal_rhs(
    gb: &mut GateBuilder,
    lhs_bits: &AigBitVector,
    rhs_bits: &xlsynth::IrBits,
) -> AigOperand {
    let bit_count = lhs_bits.get_bit_count();
    assert_eq!(
        rhs_bits.get_bit_count(),
        bit_count,
        "eq literal rhs width mismatch"
    );
    assert!(bit_count > 0, "eq requires non-zero width");

    let mut terms: Vec<AigOperand> = Vec::with_capacity(bit_count);
    for i in 0..bit_count {
        let lhs = *lhs_bits.get_lsb(i);
        let rhs = rhs_bits.get_bit(i).unwrap();
        terms.push(if rhs { lhs } else { gb.add_not(lhs) });
    }
    gb.add_and_nary(&terms, ReductionKind::Tree)
}

/// Builds `lt`/`gt` for `lhs_bits` compared to a constant RHS using a run-based
/// factorization over the constant bits (LSB-first).
///
/// This avoids building N separate "prefix equality" terms that must then be
/// OR'd together; instead it produces a nested OR/AND form whose AIG AND-count
/// is close to what ABC tends to find for threshold predicates.
fn gatify_ucmp_const_threshold(
    gb: &mut GateBuilder,
    lhs_bits: &AigBitVector,
    rhs_bits: &xlsynth::IrBits,
) -> UcmpConstResult {
    let bit_count = lhs_bits.get_bit_count();
    assert_eq!(
        rhs_bits.get_bit_count(),
        bit_count,
        "ucmp literal rhs width mismatch"
    );
    assert!(bit_count > 0, "ucmp requires non-zero width");

    // Equality is simply the conjunction of the per-bit equality literals.
    let eq = gatify_eq_literal_rhs(gb, lhs_bits, rhs_bits);

    // Build `gt` and `lt` with LSB-first recurrence, compressed over runs of
    // identical RHS bits to allow balanced trees within runs.
    let mut gt = gb.get_false();
    let mut lt = gb.get_false();

    let mut i = 0usize;
    while i < bit_count {
        let rhs_bit = rhs_bits.get_bit(i).unwrap();
        let mut j = i + 1;
        while j < bit_count && rhs_bits.get_bit(j).unwrap() == rhs_bit {
            j += 1;
        }
        // Run is bits [i, j).
        let run_len = j - i;
        let mut lhs_run: Vec<AigOperand> = Vec::with_capacity(run_len);
        let mut not_lhs_run: Vec<AigOperand> = Vec::with_capacity(run_len);
        for k in i..j {
            let lhs = *lhs_bits.get_lsb(k);
            lhs_run.push(lhs);
            not_lhs_run.push(gb.add_not(lhs));
        }

        if rhs_bit {
            // For RHS=1 bits:
            // - gt recurrence: gt = (AND over lhs_run) & gt
            // - lt recurrence: lt = (OR over !lhs_run) | lt
            let run_and = gb.add_and_nary(&lhs_run, ReductionKind::Tree);
            gt = gb.add_and_binary(run_and, gt);

            let run_or = gb.add_or_nary(&not_lhs_run, ReductionKind::Tree);
            lt = gb.add_or_binary(run_or, lt);
        } else {
            // For RHS=0 bits:
            // - gt recurrence: gt = (OR over lhs_run) | gt
            // - lt recurrence: lt = (AND over !lhs_run) & lt
            let run_or = gb.add_or_nary(&lhs_run, ReductionKind::Tree);
            gt = gb.add_or_binary(run_or, gt);

            let run_and = gb.add_and_nary(&not_lhs_run, ReductionKind::Tree);
            lt = gb.add_and_binary(run_and, lt);
        }

        i = j;
    }

    UcmpConstResult { lt, gt, eq }
}

fn try_gatify_ucmp_literal_rhs_threshold(
    gb: &mut GateBuilder,
    binop: ir::Binop,
    lhs_bits: &AigBitVector,
    rhs_bits: &xlsynth::IrBits,
) -> Option<AigOperand> {
    match binop {
        ir::Binop::Ult | ir::Binop::Ule | ir::Binop::Ugt | ir::Binop::Uge => {}
        _ => return None,
    }
    let s = gatify_ucmp_const_threshold(gb, lhs_bits, rhs_bits);
    Some(match binop {
        ir::Binop::Ult => s.lt,
        ir::Binop::Ugt => s.gt,
        ir::Binop::Ule => gb.add_or_binary(s.lt, s.eq),
        ir::Binop::Uge => gb.add_or_binary(s.gt, s.eq),
        _ => unreachable!(),
    })
}

fn gatify_ucmp_fallback(
    gb: &mut GateBuilder,
    text_id: usize,
    binop: ir::Binop,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
) -> AigOperand {
    match binop {
        ir::Binop::Ult => gatify_ult_via_bit_tests(gb, text_id, lhs_bits, rhs_bits),
        ir::Binop::Ule => gatify_ule_via_bit_tests(gb, text_id, lhs_bits, rhs_bits),
        ir::Binop::Ugt => gatify_ugt_via_bit_tests(gb, text_id, lhs_bits, rhs_bits),
        ir::Binop::Uge => gatify_uge_via_bit_tests(gb, text_id, lhs_bits, rhs_bits),
        other => panic!("unexpected ucmp binop: {:?}", other),
    }
}

fn gatify_scmp_fallback(
    gb: &mut GateBuilder,
    text_id: usize,
    binop: ir::Binop,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
) -> AigOperand {
    match binop {
        ir::Binop::Slt => {
            gatify_scmp_via_bit_tests(gb, text_id, lhs_bits, rhs_bits, CmpKind::Lt, false)
        }
        ir::Binop::Sle => {
            gatify_scmp_via_bit_tests(gb, text_id, lhs_bits, rhs_bits, CmpKind::Lt, true)
        }
        ir::Binop::Sgt => {
            gatify_scmp_via_bit_tests(gb, text_id, lhs_bits, rhs_bits, CmpKind::Gt, false)
        }
        ir::Binop::Sge => {
            gatify_scmp_via_bit_tests(gb, text_id, lhs_bits, rhs_bits, CmpKind::Gt, true)
        }
        other => panic!("unexpected scmp binop: {:?}", other),
    }
}

fn gatify_cmp_with_optional_literal_rhs(
    f: &ir::Fn,
    gb: &mut GateBuilder,
    env: &GateEnv,
    text_id: usize,
    binop: ir::Binop,
    a: ir::NodeRef,
    b: ir::NodeRef,
) -> AigOperand {
    if let Some(n) = normalize_cmp_literal_rhs(f, binop, a, b) {
        let lhs_bits = env
            .get_bit_vector(n.lhs)
            .expect("cmp lhs should be present (normalized)");
        let rhs_bits_vec = env
            .get_bit_vector(n.rhs)
            .expect("cmp rhs should be present (normalized)");
        if let Some(gate) =
            try_simplify_cmp_literal_rhs(gb, n.binop, &lhs_bits, &rhs_bits_vec, &n.rhs_bits)
        {
            return gate;
        }
        if let Some(gate) =
            try_gatify_ucmp_literal_rhs_threshold(gb, n.binop, &lhs_bits, &n.rhs_bits)
        {
            return gate;
        }
        match n.binop {
            ir::Binop::Ult | ir::Binop::Ule | ir::Binop::Ugt | ir::Binop::Uge => {
                return gatify_ucmp_fallback(gb, text_id, n.binop, &lhs_bits, &rhs_bits_vec);
            }
            ir::Binop::Slt | ir::Binop::Sle | ir::Binop::Sgt | ir::Binop::Sge => {
                return gatify_scmp_fallback(gb, text_id, n.binop, &lhs_bits, &rhs_bits_vec);
            }
            other => panic!("unexpected normalized cmp binop: {:?}", other),
        }
    }

    let lhs_bits = env.get_bit_vector(a).expect("cmp lhs should be present");
    let rhs_bits = env.get_bit_vector(b).expect("cmp rhs should be present");
    match binop {
        ir::Binop::Ult | ir::Binop::Ule | ir::Binop::Ugt | ir::Binop::Uge => {
            gatify_ucmp_fallback(gb, text_id, binop, &lhs_bits, &rhs_bits)
        }
        ir::Binop::Slt | ir::Binop::Sle | ir::Binop::Sgt | ir::Binop::Sge => {
            gatify_scmp_fallback(gb, text_id, binop, &lhs_bits, &rhs_bits)
        }
        other => panic!("unexpected cmp binop: {:?}", other),
    }
}

pub fn gatify_scmp_via_bit_tests(
    gb: &mut GateBuilder,
    _text_id: usize,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
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
        // Signed comparisons:
        // - If signs differ, a < b iff a is negative.
        // - If signs are the same, signed order matches unsigned order.
        let a_msb = lhs_bits.get_msb(0);
        let b_msb = rhs_bits.get_msb(0);
        let sign_diff = gb.add_xor_binary(*a_msb, *b_msb);

        let (ult, eq) = gatify_ult_and_eq_via_bit_tests(gb, lhs_bits, rhs_bits);
        let term1 = gb.add_and_binary(sign_diff, *a_msb);
        let not_sign_diff = gb.add_not(sign_diff);
        let term2 = gb.add_and_binary(not_sign_diff, ult);
        let lt = gb.add_or_binary(term1, term2);
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
        let scmp_tag = format!("scmp_{}", text_id);
        let (_carry, diff) = gatify_add_with_mapping(
            adder_mapping,
            lhs_bits,
            &b_complement,
            gb.get_true(),
            Some(&scmp_tag),
            gb,
        );
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

fn gatify_node(
    f: &ir::Fn,
    node_ref: ir::NodeRef,
    node: &ir::Node,
    g8_builder: &mut GateBuilder,
    env: &mut GateEnv,
    options: &GatifyOptions,
    param_id_to_node_ref: &HashMap<ParamId, ir::NodeRef>,
) -> Result<(), String> {
    let payload = &node.payload;
    match payload {
        ir::NodePayload::GetParam(param_id) => {
            if env.contains(node_ref) {
                return Ok(()); // Already inserted above.
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
                let proven_in_bounds = options.range_info.as_ref().is_some_and(|ri| {
                    ri.proves_ult(f.get_node(*index_node).text_id, array_ty.element_count)
                });
                array_bits = gatify_array_index(
                    g8_builder,
                    array_ty,
                    &array_bits,
                    &index_bits,
                    *assumed_in_bounds || proven_in_bounds,
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
        ir::NodePayload::ArraySlice {
            array,
            start,
            width,
        } => {
            let array_ty = match f.get_node_ty(*array) {
                ir::Type::Array(array_ty_data) => array_ty_data,
                other => panic!("Expected array type for array_slice, got {:?}", other),
            };
            let array_bits = env.get_bit_vector(*array).unwrap();
            let start_bits = env.get_bit_vector(*start).unwrap();
            let proven_start_in_bounds = options.range_info.as_ref().is_some_and(|ri| {
                ri.proves_ult(f.get_node(*start).text_id, array_ty.element_count)
            });
            let mul_adder_mapping = options.mul_adder_mapping.unwrap_or(options.adder_mapping);
            let result = gatify_array_slice(
                g8_builder,
                array_ty,
                &array_bits,
                &start_bits,
                proven_start_in_bounds,
                *width,
                node.text_id,
                mul_adder_mapping,
            );
            env.add(node_ref, GateOrVec::BitVector(result));
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

            let result_bits =
                gatify_array_update(g8_builder, array_ty, &array_bits, &value_bits, &index_bits);

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
            let impossible_case_indices =
                get_impossible_in_bounds_sel_case_indices(options, f, *selector, cases.len());
            if !impossible_case_indices.is_empty() {
                let max_show = 16usize;
                let shown: Vec<usize> = impossible_case_indices
                    .iter()
                    .copied()
                    .take(max_show)
                    .collect();
                let suffix = if impossible_case_indices.len() > max_show {
                    format!(" (showing first {max_show})")
                } else {
                    String::new()
                };
                log::warn!(
                    "sel text_id={} has selector text_id={} with {} unreachable in-bounds case(s) out of {}: {:?}{}",
                    node.text_id,
                    f.get_node(*selector).text_id,
                    impossible_case_indices.len(),
                    cases.len(),
                    shown,
                    suffix
                );
            }
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
            let zero = g8_builder
                .add_literal(&xlsynth::IrBits::make_ubits(arg_gates.get_bit_count(), 0).unwrap());
            let neg_tag = format!("neg_{}", node.text_id);
            let (_, result) = gatify_add_with_mapping(
                options.adder_mapping,
                &not_arg,
                &zero,
                g8_builder.get_true(),
                Some(&neg_tag),
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

        // -- extension operations
        ir::NodePayload::ExtCarryOut { lhs, rhs, c_in } => {
            let lhs_bits = env.get_bit_vector(*lhs).expect("lhs should be present");
            let rhs_bits = env.get_bit_vector(*rhs).expect("rhs should be present");
            assert_eq!(
                lhs_bits.get_bit_count(),
                rhs_bits.get_bit_count(),
                "ExtCarryOut requires equal-width operands"
            );
            let c_in_bits = env.get_bit_vector(*c_in).expect("c_in should be present");
            let c_in_bit: AigOperand = c_in_bits
                .clone()
                .try_into()
                .expect("ExtCarryOut c_in should be bits[1]");

            let w = lhs_bits.get_bit_count();
            let mut carry: AigOperand = c_in_bit;
            for i in 0..w {
                let a_i = *lhs_bits.get_lsb(i);
                let b_i = *rhs_bits.get_lsb(i);
                let g_i = g8_builder.add_and_binary(a_i, b_i);
                let p_i = g8_builder.add_or_binary(a_i, b_i);
                let p_and_c = g8_builder.add_and_binary(p_i, carry);
                carry = g8_builder.add_or_binary(g_i, p_and_c);
            }

            env.add(node_ref, GateOrVec::Gate(carry));
        }
        ir::NodePayload::ExtPrioEncode { arg, lsb_prio } => {
            let arg_bits = env
                .get_bit_vector(*arg)
                .expect("ext_prio_encode arg should be present");
            let (any, idx_bits) = gatify_prio_encode(g8_builder, &arg_bits, *lsb_prio);
            let sentinel_bit = g8_builder.add_not(any);

            let expected_out_w = idx_bits.get_bit_count().saturating_add(1);
            assert_eq!(
                node.ty.bit_count(),
                expected_out_w,
                "ExtPrioEncode output width mismatch; expected {} got {}",
                expected_out_w,
                node.ty.bit_count()
            );

            let mut out: Vec<AigOperand> = Vec::with_capacity(expected_out_w);
            for bit in idx_bits.iter_lsb_to_msb() {
                out.push(*bit);
            }
            out.push(sentinel_bit);

            let out_bits = AigBitVector::from_lsb_is_index_0(&out);
            for (i, gate) in out_bits.iter_lsb_to_msb().enumerate() {
                g8_builder.add_tag(
                    gate.node,
                    format!("ext_prio_encode_{}_output_bit_{}", node.text_id, i),
                );
            }
            env.add(node_ref, GateOrVec::BitVector(out_bits));
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
        ir::NodePayload::Binop(
            binop @ (ir::Binop::Ult | ir::Binop::Ule | ir::Binop::Ugt | ir::Binop::Uge),
            a,
            b,
        ) => {
            let gate = gatify_cmp_with_optional_literal_rhs(
                f,
                g8_builder,
                env,
                node.text_id,
                *binop,
                *a,
                *b,
            );
            g8_builder.add_tag(
                gate.node,
                format!("{}_{}", ir::binop_to_operator(*binop), node.text_id),
            );
            env.add(node_ref, GateOrVec::Gate(gate));
        }

        ir::NodePayload::Binop(
            binop @ (ir::Binop::Slt | ir::Binop::Sle | ir::Binop::Sgt | ir::Binop::Sge),
            a,
            b,
        ) => {
            let gate = gatify_cmp_with_optional_literal_rhs(
                f,
                g8_builder,
                env,
                node.text_id,
                *binop,
                *a,
                *b,
            );
            g8_builder.add_tag(
                gate.node,
                format!("{}_{}", ir::binop_to_operator(*binop), node.text_id),
            );
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
            let gates: AigBitVector = g8_builder.add_or_vec_nary(&arg_gates, ReductionKind::Tree);
            let gates = g8_builder.add_not_vec(&gates);
            env.add(node_ref, GateOrVec::BitVector(gates));
        }
        ir::NodePayload::Nary(ir::NaryOp::Or, args) => {
            let arg_gates: Vec<AigBitVector> = args
                .iter()
                .map(|arg| env.get_bit_vector(*arg).expect("or arg should be present"))
                .collect();
            let gates: AigBitVector = g8_builder.add_or_vec_nary(&arg_gates, ReductionKind::Tree);
            env.add(node_ref, GateOrVec::BitVector(gates));
        }
        ir::NodePayload::Nary(ir::NaryOp::Xor, args) => {
            let arg_gates: Vec<AigBitVector> = args
                .iter()
                .map(|arg| env.get_bit_vector(*arg).expect("xor arg should be present"))
                .collect();
            let gates: AigBitVector = g8_builder.add_xor_vec_nary(&arg_gates, ReductionKind::Tree);
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
            let gates: AigBitVector = g8_builder.add_and_vec_nary(&arg_gates, ReductionKind::Tree);
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
            let known_zero_bits = get_known_zero_bit_indices_for_selector(options, f, *selector);
            if !known_zero_bits.is_empty() {
                log::warn!(
                    "priority_sel text_id={} has selector text_id={} with {} provably-zero bit(s): {:?}",
                    node.text_id,
                    f.get_node(*selector).text_id,
                    known_zero_bits.len(),
                    known_zero_bits
                );
            }
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
            let known_zero_bits = get_known_zero_bit_indices_for_selector(options, f, *selector);
            if !known_zero_bits.is_empty() {
                log::warn!(
                    "one_hot_sel text_id={} has selector text_id={} with {} provably-zero bit(s): {:?}",
                    node.text_id,
                    f.get_node(*selector).text_id,
                    known_zero_bits.len(),
                    known_zero_bits
                );
            }
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
            let add_tag = format!("add_{}", node.text_id);
            let (_c_out, gates) = gatify_add_with_mapping(
                options.adder_mapping,
                &a_gate_refs,
                &b_gate_refs,
                g8_builder.get_false(),
                Some(&add_tag),
                g8_builder,
            );
            assert_eq!(gates.get_bit_count(), a_gate_refs.get_bit_count());
            env.add(node_ref, GateOrVec::BitVector(gates));
        }
        ir::NodePayload::Binop(ir::Binop::Sub, a, b) => {
            let a_gate_refs = env.get_bit_vector(*a).expect("sub lhs should be present");
            let b_gate_refs = env.get_bit_vector(*b).expect("sub rhs should be present");
            assert_eq!(a_gate_refs.get_bit_count(), b_gate_refs.get_bit_count());
            let b_complement = g8_builder.add_not_vec(&b_gate_refs);
            let sub_tag = format!("sub_{}", node.text_id);
            let (_c_out, gates) = gatify_add_with_mapping(
                options.adder_mapping,
                &a_gate_refs,
                &b_complement,
                g8_builder.get_true(),
                Some(&sub_tag),
                g8_builder,
            );
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
            let result_bits = gatify_sign_ext(g8_builder, node.text_id, *new_bit_count, &arg_bits);
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
            maybe_warn_shift_amount_truncatable(
                options.range_info.as_ref(),
                f.get_node(*amount).text_id,
                arg_gates.get_bit_count(),
                &amount_gates,
            );
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

            maybe_warn_shift_amount_truncatable(
                options.range_info.as_ref(),
                f.get_node(*amount).text_id,
                arg_gates.get_bit_count(),
                &amount_gates,
            );
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
            maybe_warn_shift_amount_truncatable(
                options.range_info.as_ref(),
                f.get_node(*amount).text_id,
                arg_gates.get_bit_count(),
                &amount_gates,
            );
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
            let proven_nonzero = options
                .range_info
                .as_ref()
                .is_some_and(|ri| ri.proves_nonzero(f.get_node(*arg).text_id));
            let bit_vector = if proven_nonzero {
                gatify_one_hot_with_nonzero_flag(g8_builder, &bits, *lsb_prio, true)
            } else {
                gatify_one_hot(g8_builder, &bits, *lsb_prio)
            };
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
            let signedness = if matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Smul, ..))
            {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            let mul_adder_mapping = options.mul_adder_mapping.unwrap_or(options.adder_mapping);
            let gates = gatify_mul(
                &lhs_bits,
                &rhs_bits,
                output_bit_count,
                signedness,
                mul_adder_mapping,
                g8_builder,
            );
            env.add(node_ref, GateOrVec::BitVector(gates));
        }
        ir::NodePayload::Binop(ir::Binop::Udiv | ir::Binop::Sdiv, lhs, rhs) => {
            let lhs_bits = env.get_bit_vector(*lhs).expect("div lhs should be present");
            let rhs_bits = env.get_bit_vector(*rhs).expect("div rhs should be present");
            let signedness = if matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Sdiv, ..))
            {
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
            let signedness = if matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Smod, ..))
            {
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
            let msg = format!("Unsupported node payload {:?}", payload);
            return Err(msg);
        }
    }
    Ok(())
}

/// Converts the contents of the given IR function to our "g8" representation
/// which has gates and vectors of gates.
fn gatify_internal(
    f: &ir::Fn,
    g8_builder: &mut GateBuilder,
    env: &mut GateEnv,
    options: &GatifyOptions,
) -> Result<(), String> {
    log::debug!("gatify_internal; f.name: {}", f.name);
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
        log::debug!(
            "Gatifying node {:?} type: {:?} payload: {:?}",
            node_ref,
            node.ty,
            node.payload
        );
        gatify_node(
            f,
            node_ref,
            node,
            g8_builder,
            env,
            options,
            &param_id_to_node_ref,
        )?;
    }
    // Resolve the outputs and place them into the builder.
    let ret_node_ref = match f.ret_node_ref {
        Some(ret_node_ref) => ret_node_ref,
        None => {
            return Ok(());
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
    Ok(())
}

#[derive(Clone)]
pub struct GatifyOptions {
    pub fold: bool,
    pub hash: bool,
    pub check_equivalence: bool,
    pub adder_mapping: crate::ir2gate_utils::AdderMapping,
    pub mul_adder_mapping: Option<crate::ir2gate_utils::AdderMapping>,
    pub range_info: Option<Arc<IrRangeInfo>>,
    pub enable_rewrite_carry_out: bool,
    pub enable_rewrite_prio_encode: bool,
}

// Type alias for the lowering map
pub type IrToGateMap = HashMap<ir::NodeRef, AigBitVector>;

/// Holds the output of the `gatify` process.
#[derive(Debug)]
pub struct GatifyOutput {
    pub gate_fn: GateFn,
    pub lowering_map: IrToGateMap,
}

fn validate_fn_for_gatify(f: &ir::Fn) -> Result<(), String> {
    // Validate the function in a minimal package context.
    //
    // This catches structural issues (operand bounds/order, return node/type,
    // text-id uniqueness within the function, etc.) before and after
    // `prep_for_gatify`.
    let pkg = ir::Package {
        name: "gatify_validate".to_string(),
        file_table: ir::FileTable::new(),
        members: vec![ir::PackageMember::Function(f.clone())],
        top: Some((f.name.clone(), ir::MemberType::Function)),
    };
    ir_validate::validate_package(&pkg).map_err(|e| e.to_string())
}

pub fn gatify(orig_fn: &ir::Fn, options: GatifyOptions) -> Result<GatifyOutput, String> {
    validate_fn_for_gatify(orig_fn)
        .map_err(|e| format!("PIR validation failed before prep_for_gatify: {e}"))?;

    // `prep_for_gatify` may introduce many new nodes (e.g. lowering ext ops), so
    // any `NodeRef { index }` values produced during gatification generally
    // refer to the *prepared* function's node vector. Most consumers want to
    // interpret the lowering map in terms of the original function, so we
    // remap prepared nodes back to original nodes via stable `text_id`.
    let mut orig_ref_by_text_id: HashMap<usize, ir::NodeRef> = HashMap::new();
    for (idx, n) in orig_fn.nodes.iter().enumerate() {
        orig_ref_by_text_id.insert(n.text_id, ir::NodeRef { index: idx });
    }

    let prepared_fn = prep_for_gatify(
        orig_fn,
        options.range_info.as_deref(),
        PrepForGatifyOptions {
            enable_rewrite_carry_out: options.enable_rewrite_carry_out,
            enable_rewrite_prio_encode: options.enable_rewrite_prio_encode,
        },
    );
    validate_fn_for_gatify(&prepared_fn)
        .map_err(|e| format!("PIR validation failed after prep_for_gatify: {e}"))?;

    let f = &prepared_fn;
    let mut g8_builder = GateBuilder::new(
        f.name.clone(),
        GateBuilderOptions {
            fold: options.fold,
            hash: options.hash,
        },
    );
    let mut env = GateEnv::new();
    gatify_internal(f, &mut g8_builder, &mut env, &options)?;
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
        let prepared_text_id = f.get_node(node_ref).text_id;
        let Some(orig_node_ref) = orig_ref_by_text_id.get(&prepared_text_id).copied() else {
            // Not a sample failure: many helper nodes (introduced during prep/lowering)
            // do not exist in the original function, so we don't expose them here.
            continue;
        };
        lowering_map.insert(orig_node_ref, bit_vector);
    }

    // If we're told we should do so, we check equivalence between the original IR
    // function and the gate function that we converted it to.
    if options.check_equivalence {
        log::info!("checking equivalence of IR function and gate function...");
        check_equivalence::validate_same_fn(orig_fn, &gate_fn)?;
    }
    // Construct and return the GatifyOutput struct
    Ok(GatifyOutput {
        gate_fn,
        lowering_map,
    })
}

pub fn gatify_node_as_fn(
    f: &ir::Fn,
    node_ref: ir::NodeRef,
    options: &GatifyOptions,
) -> Result<GateFn, String> {
    let target_text_id = f.get_node(node_ref).text_id;
    validate_fn_for_gatify(f)
        .map_err(|e| format!("PIR validation failed before prep_for_gatify: {e}"))?;
    let prepared_fn = prep_for_gatify(
        f,
        options.range_info.as_deref(),
        PrepForGatifyOptions {
            enable_rewrite_carry_out: options.enable_rewrite_carry_out,
            enable_rewrite_prio_encode: options.enable_rewrite_prio_encode,
        },
    );
    validate_fn_for_gatify(&prepared_fn)
        .map_err(|e| format!("PIR validation failed after prep_for_gatify: {e}"))?;
    let f = &prepared_fn;
    let prepared_node_ref: ir::NodeRef = f
        .nodes
        .iter()
        .enumerate()
        .find_map(|(idx, n)| {
            if n.text_id == target_text_id {
                Some(ir::NodeRef { index: idx })
            } else {
                None
            }
        })
        .ok_or_else(|| {
            format!(
                "gatify_node_as_fn: could not find node with original text_id={} after prep",
                target_text_id
            )
        })?;
    let node = f.get_node(prepared_node_ref);
    let mut g8_builder = GateBuilder::new(
        format!("{}_node_{}", f.name, node.text_id),
        GateBuilderOptions {
            fold: options.fold,
            hash: options.hash,
        },
    );
    let mut env = GateEnv::new();

    // Precompute a map from parameter id to its NodeRef in f.nodes. This is used
    // when lowering GetParam nodes.
    let mut param_id_to_node_ref: HashMap<ParamId, ir::NodeRef> = HashMap::new();
    for (i, param) in f.params.iter().enumerate() {
        let param_ref = ir::NodeRef { index: i + 1 };
        assert!(
            f.nodes[i + 1].payload == ir::NodePayload::GetParam(param.id),
            "expected param node at index {}",
            i + 1
        );
        param_id_to_node_ref.insert(param.id, param_ref);
    }

    // Seed the direct operands of this node as independent GateFn inputs.
    let operands: Vec<ir::NodeRef> = ir_utils::operands(&node.payload);
    for (i, operand_ref) in operands.iter().enumerate() {
        if env.contains(*operand_ref) {
            continue;
        }
        let operand_ty = f.get_node_ty(*operand_ref);
        let width = operand_ty.bit_count();
        let input_bits = g8_builder.add_input(format!("op{}_n{}", i, operand_ref.index), width);
        env.add(*operand_ref, GateOrVec::BitVector(input_bits));
    }

    // Lower the node into env/builder and emit it as the single output.
    match &node.payload {
        ir::NodePayload::GetParam(param_id) => {
            let param = f
                .params
                .iter()
                .find(|p| p.id == *param_id)
                .ok_or_else(|| format!("GetParam refers to missing ParamId {:?}", param_id))?;
            let bits = g8_builder.add_input(param.name.clone(), param.ty.bit_count());
            g8_builder.add_output("output_value".to_string(), bits);
            return Ok(g8_builder.build());
        }
        ir::NodePayload::Literal(literal) => {
            let bits = flatten_literal_to_bits(literal, &node.ty, &mut g8_builder);
            g8_builder.add_output("output_value".to_string(), bits);
            return Ok(g8_builder.build());
        }
        ir::NodePayload::Nil
        | ir::NodePayload::Assert { .. }
        | ir::NodePayload::AfterAll(..)
        | ir::NodePayload::Trace { .. } => {
            g8_builder.add_output(
                "output_value".to_string(),
                AigBitVector::zeros(node.ty.bit_count()),
            );
            return Ok(g8_builder.build());
        }
        _ => {}
    }

    gatify_node(
        f,
        node_ref,
        node,
        &mut g8_builder,
        &mut env,
        options,
        &param_id_to_node_ref,
    )?;
    let output_bits = env.get_bit_vector(node_ref)?;
    g8_builder.add_output("output_value".to_string(), output_bits);
    Ok(g8_builder.build())
}

#[cfg(test)]
mod tests {
    use crate::aig::get_summary_stats::{SummaryStats, get_summary_stats};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::gatify::ir2gate::{GatifyOptions, gatify};
    use crate::ir2gate_utils::AdderMapping;
    use xlsynth_pir::ir;
    use xlsynth_pir::ir_parser;

    #[test]
    fn test_gatify_array_literal_flattening() {
        let ir_text = "fn f() -> bits[8][5] {\n  ret literal.1: bits[8][5] = literal(value=[1, 2, 3, 4, 5], id=1)\n}";
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_fn = parser.parse_fn().unwrap();

        let gatify_output = gatify(
            &ir_fn,
            GatifyOptions {
                fold: false,
                hash: false,
                check_equivalence: false,
                adder_mapping: AdderMapping::default(),
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
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
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();

        let gatify_output = gatify(
            &ir_fn,
            GatifyOptions {
                fold: true,               // Folding shouldn't affect this test
                check_equivalence: false, // Not needed for this map check
                hash: true,
                adder_mapping: AdderMapping::default(),
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
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

    fn get_1b_priority_sel_stats_for_impl(
        operand_count: usize,
        use_mux_chain: bool,
    ) -> SummaryStats {
        let mut gb = GateBuilder::new(
            format!(
                "prio_sel_{}_{}",
                operand_count,
                if use_mux_chain { "mux" } else { "mask" }
            ),
            GateBuilderOptions::opt(),
        );
        let selector_bits = gb.add_input("sel".to_string(), operand_count);
        let mut cases = Vec::with_capacity(operand_count);
        for i in 0..operand_count {
            cases.push(gb.add_input(format!("a{}", i), 1));
        }
        let default_bits = gb.add_input("default_value".to_string(), 1);

        let result = if use_mux_chain {
            super::gatify_priority_sel_mux_chain(&mut gb, selector_bits, &cases, default_bits)
        } else {
            super::gatify_priority_sel_masking(
                &mut gb,
                /* output_bit_count= */ 1,
                selector_bits,
                &cases,
                Some(default_bits),
            )
        };
        gb.add_output("result".to_string(), result);
        let gate_fn = gb.build();
        get_summary_stats(&gate_fn)
    }

    /// TDD-ish “microbenchmark sweep” test: record the baseline W=1
    /// priority_sel cost (mask+OR implementation) and assert the mux-chain
    /// specialization is strictly cheaper.
    #[test]
    fn test_priority_sel_1b_mux_chain_is_cheaper_than_masking_sweep() {
        #[rustfmt::skip]
        let want_masking: &[(usize, usize)] = &[
            (2, 12),
            (3, 18),
            (4, 24),
            (5, 30),
        ];
        #[rustfmt::skip]
        let want_mux_chain: &[(usize, usize)] = &[
            (2, 11),
            (3, 16),
            (4, 21),
            (5, 26),
        ];

        for &(operand_count, want_live_nodes) in want_masking {
            let got =
                get_1b_priority_sel_stats_for_impl(operand_count, /* use_mux_chain= */ false);
            assert_eq!(
                got.live_nodes, want_live_nodes,
                "masking impl live_nodes mismatch for operand_count={}",
                operand_count
            );
        }

        for &(operand_count, want_live_nodes) in want_mux_chain {
            let got =
                get_1b_priority_sel_stats_for_impl(operand_count, /* use_mux_chain= */ true);
            assert_eq!(
                got.live_nodes, want_live_nodes,
                "mux-chain impl live_nodes mismatch for operand_count={}",
                operand_count
            );
        }

        for operand_count in 2usize..=5 {
            let masking =
                get_1b_priority_sel_stats_for_impl(operand_count, /* use_mux_chain= */ false);
            let mux =
                get_1b_priority_sel_stats_for_impl(operand_count, /* use_mux_chain= */ true);
            assert!(
                mux.live_nodes < masking.live_nodes,
                "expected mux-chain to be cheaper for operand_count={}; masking live_nodes={}, mux live_nodes={}",
                operand_count,
                masking.live_nodes,
                mux.live_nodes
            );
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct PrioSelSweepRow {
        operand_count: usize,
        masking_live_nodes: usize,
        masking_deepest_path: usize,
        mux_chain_live_nodes: usize,
        mux_chain_deepest_path: usize,
    }

    fn get_priority_sel_stats_for_impl(
        output_bit_count: usize,
        operand_count: usize,
        use_mux_chain: bool,
    ) -> SummaryStats {
        let mut gb = GateBuilder::new(
            format!(
                "prio_sel_w{}_n{}_{}",
                output_bit_count,
                operand_count,
                if use_mux_chain { "mux" } else { "mask" }
            ),
            GateBuilderOptions::opt(),
        );
        let selector_bits = gb.add_input("sel".to_string(), operand_count);
        let mut cases = Vec::with_capacity(operand_count);
        for i in 0..operand_count {
            cases.push(gb.add_input(format!("a{}", i), output_bit_count));
        }
        let default_bits = gb.add_input("default_value".to_string(), output_bit_count);

        let result = if use_mux_chain {
            // Hypothetical mux-chain lowering for comparison across widths:
            // mux(s0, c0, mux(s1, c1, ... default))
            let mut acc = default_bits;
            for i in (0..cases.len()).rev() {
                let s_i = selector_bits.get_lsb(i);
                acc = gb.add_mux2_vec(s_i, &cases[i], &acc);
            }
            acc
        } else {
            super::gatify_priority_sel_masking(
                &mut gb,
                output_bit_count,
                selector_bits,
                &cases,
                Some(default_bits),
            )
        };
        gb.add_output("result".to_string(), result);
        let gate_fn = gb.build();
        get_summary_stats(&gate_fn)
    }

    /// “Table sweep” test: captures masking vs mux-chain AIG sizes as output
    /// width increases (W=1..4) for small operand counts (2..5).
    ///
    /// This reflects the crossover behavior we care about when deciding whether
    /// a mux-chain specialization is worthwhile beyond W=1.
    #[test]
    fn test_priority_sel_masking_vs_mux_chain_table_sweep_w1_to_w4() {
        #[rustfmt::skip]
        const WANT_W1: &[PrioSelSweepRow] = &[
            PrioSelSweepRow { operand_count: 2, masking_live_nodes: 12, masking_deepest_path: 6, mux_chain_live_nodes: 11, mux_chain_deepest_path: 5 },
            PrioSelSweepRow { operand_count: 3, masking_live_nodes: 18, masking_deepest_path: 8, mux_chain_live_nodes: 16, mux_chain_deepest_path: 7 },
            PrioSelSweepRow { operand_count: 4, masking_live_nodes: 24, masking_deepest_path: 11, mux_chain_live_nodes: 21, mux_chain_deepest_path: 9 },
            PrioSelSweepRow { operand_count: 5, masking_live_nodes: 30, masking_deepest_path: 13, mux_chain_live_nodes: 26, mux_chain_deepest_path: 11 },
        ];
        #[rustfmt::skip]
        const WANT_W2: &[PrioSelSweepRow] = &[
            PrioSelSweepRow { operand_count: 2, masking_live_nodes: 20, masking_deepest_path: 6, mux_chain_live_nodes: 20, mux_chain_deepest_path: 5 },
            PrioSelSweepRow { operand_count: 3, masking_live_nodes: 29, masking_deepest_path: 8, mux_chain_live_nodes: 29, mux_chain_deepest_path: 7 },
            PrioSelSweepRow { operand_count: 4, masking_live_nodes: 38, masking_deepest_path: 11, mux_chain_live_nodes: 38, mux_chain_deepest_path: 9 },
            PrioSelSweepRow { operand_count: 5, masking_live_nodes: 47, masking_deepest_path: 13, mux_chain_live_nodes: 47, mux_chain_deepest_path: 11 },
        ];
        #[rustfmt::skip]
        const WANT_W3: &[PrioSelSweepRow] = &[
            PrioSelSweepRow { operand_count: 2, masking_live_nodes: 28, masking_deepest_path: 6, mux_chain_live_nodes: 29, mux_chain_deepest_path: 5 },
            PrioSelSweepRow { operand_count: 3, masking_live_nodes: 40, masking_deepest_path: 8, mux_chain_live_nodes: 42, mux_chain_deepest_path: 7 },
            PrioSelSweepRow { operand_count: 4, masking_live_nodes: 52, masking_deepest_path: 11, mux_chain_live_nodes: 55, mux_chain_deepest_path: 9 },
            PrioSelSweepRow { operand_count: 5, masking_live_nodes: 64, masking_deepest_path: 13, mux_chain_live_nodes: 68, mux_chain_deepest_path: 11 },
        ];
        #[rustfmt::skip]
        const WANT_W4: &[PrioSelSweepRow] = &[
            PrioSelSweepRow { operand_count: 2, masking_live_nodes: 36, masking_deepest_path: 6, mux_chain_live_nodes: 38, mux_chain_deepest_path: 5 },
            PrioSelSweepRow { operand_count: 3, masking_live_nodes: 51, masking_deepest_path: 8, mux_chain_live_nodes: 55, mux_chain_deepest_path: 7 },
            PrioSelSweepRow { operand_count: 4, masking_live_nodes: 66, masking_deepest_path: 11, mux_chain_live_nodes: 72, mux_chain_deepest_path: 9 },
            PrioSelSweepRow { operand_count: 5, masking_live_nodes: 81, masking_deepest_path: 13, mux_chain_live_nodes: 89, mux_chain_deepest_path: 11 },
        ];
        #[rustfmt::skip]
        const WANT: &[(usize, &[PrioSelSweepRow])] = &[
            (1, WANT_W1),
            (2, WANT_W2),
            (3, WANT_W3),
            (4, WANT_W4),
        ];

        let mut computed: Vec<(usize, Vec<PrioSelSweepRow>)> = Vec::new();
        for w in 1usize..=4 {
            let mut rows: Vec<PrioSelSweepRow> = Vec::new();
            for operand_count in 2usize..=5 {
                let masking = get_priority_sel_stats_for_impl(
                    w,
                    operand_count,
                    /* use_mux_chain= */ false,
                );
                let mux = get_priority_sel_stats_for_impl(
                    w,
                    operand_count,
                    /* use_mux_chain= */ true,
                );
                rows.push(PrioSelSweepRow {
                    operand_count,
                    masking_live_nodes: masking.live_nodes,
                    masking_deepest_path: masking.deepest_path,
                    mux_chain_live_nodes: mux.live_nodes,
                    mux_chain_deepest_path: mux.deepest_path,
                });
            }
            computed.push((w, rows));
        }

        for &(w, want_rows) in WANT {
            let got_rows = computed
                .iter()
                .find(|(got_w, _)| *got_w == w)
                .unwrap()
                .1
                .as_slice();
            assert_eq!(
                got_rows.len(),
                want_rows.len(),
                "row count mismatch for W={}",
                w
            );
            for (got, want) in got_rows.iter().zip(want_rows.iter()) {
                assert_eq!(
                    got, want,
                    "priority_sel sweep mismatch for W={}, operand_count={}; computed tables: {:?}",
                    w, want.operand_count, computed
                );
            }
        }
    }
}
