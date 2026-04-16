// SPDX-License-Identifier: Apache-2.0

//! Functionality for converting an IR function into a gate function via
//! `gatify`.

use crate::aig::gate::{AigBitVector, AigOperand, GateFn, Split};
use crate::check_equivalence;
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
use crate::gatify::ir2gate_arithmetic::{gatify_add_binop, gatify_ext_nary_add, gatify_sub_binop};
use crate::gatify::mul_by_const_csd::{SignedDigitSign, decompose_umul_const_terms};
use crate::gatify::prep_for_gatify::{PrepForGatifyOptions, prep_for_gatify};
use std::collections::HashMap;
use std::sync::Arc;
use xlsynth_pir::ir::{self, ParamId, StartAndLimit};
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir_utils;
use xlsynth_pir::ir_validate;

use crate::ir2gate_utils::{
    AdderMapping, Direction, array_add_with_carry_out, gatify_add_brent_kung,
    gatify_add_kogge_stone, gatify_add_ripple_carry, gatify_barrel_shifter,
    gatify_indexed_select_mux_tree_exact, gatify_indexed_select_mux_tree_pad_last_if_type_fits,
    gatify_one_hot, gatify_one_hot_select, gatify_one_hot_with_nonzero_flag, gatify_prio_encode,
};

use crate::gate_builder::ReductionKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayIndexLoweringStrategy {
    Auto,
    ForceOobOneHot,
    ForceNearPow2MuxTree,
}

impl Default for ArrayIndexLoweringStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

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

pub(super) struct GateEnv {
    ir_to_g8: HashMap<ir::NodeRef, GateOrVec>,
}

impl GateEnv {
    fn new() -> Self {
        Self {
            ir_to_g8: HashMap::new(),
        }
    }

    fn contains(&self, ir_node_ref: ir::NodeRef) -> bool {
        self.ir_to_g8.contains_key(&ir_node_ref)
    }

    fn add(&mut self, ir_node_ref: ir::NodeRef, gate_or_vec: GateOrVec) {
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

    fn apply_known_bits_if_any(
        &mut self,
        ir_node_ref: ir::NodeRef,
        _f: &ir::Fn,
        node: &ir::Node,
        options: &GatifyOptions,
        g8_builder: &mut GateBuilder,
    ) {
        let range_info = match options.range_info.as_ref() {
            Some(ri) => ri,
            None => return,
        };
        let info = match range_info.get(node.text_id) {
            Some(i) => i,
            None => return,
        };
        let known = match info.known_bits.as_ref() {
            Some(k) => k,
            None => return,
        };
        let mask = &known.mask;
        let value = &known.value;

        let Some(entry) = self.ir_to_g8.get_mut(&ir_node_ref) else {
            return;
        };
        let bit_count = match entry {
            GateOrVec::Gate(_) => 1,
            GateOrVec::BitVector(bv) => bv.get_bit_count(),
        };
        if bit_count == 0 || mask.get_bit_count() != bit_count {
            return;
        }

        let mut any_known = false;
        let mut apply_masked_bits = |i: usize| -> Option<AigOperand> {
            let is_known = mask.get_bit(i).unwrap_or(false);
            if !is_known {
                return None;
            }
            any_known = true;
            let is_one = value.get_bit(i).unwrap_or(false);
            Some(if is_one {
                g8_builder.get_true()
            } else {
                g8_builder.get_false()
            })
        };

        match entry {
            GateOrVec::Gate(gate) => {
                let Some(replacement) = apply_masked_bits(0) else {
                    return;
                };
                *gate = replacement;
            }
            GateOrVec::BitVector(bits) => {
                let mut updated: Vec<AigOperand> = Vec::with_capacity(bit_count);
                for i in 0..bit_count {
                    if let Some(replacement) = apply_masked_bits(i) {
                        updated.push(replacement);
                    } else {
                        updated.push(*bits.get_lsb(i));
                    }
                }
                if any_known {
                    *bits = AigBitVector::from_lsb_is_index_0(&updated);
                }
            }
        }
    }

    pub(super) fn get_bit_vector(&self, ir_node_ref: ir::NodeRef) -> Result<AigBitVector, String> {
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

pub(super) fn gatify_add_with_mapping(
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
    strategy: ArrayIndexLoweringStrategy,
) -> AigBitVector {
    if assumed_in_bounds {
        return gatify_array_index_exact(gb, array_ty, array_bits, index_bits);
    }

    match strategy {
        ArrayIndexLoweringStrategy::Auto => {
            if let Some(result) = gatify_array_index_near_pow2_mux_tree_if_profitable(
                gb, array_ty, array_bits, index_bits,
            ) {
                return result;
            }
        }
        ArrayIndexLoweringStrategy::ForceOobOneHot => {
            // Intentionally fall through to the existing OOB one-hot lowering
            // below.
        }
        ArrayIndexLoweringStrategy::ForceNearPow2MuxTree => {
            if let Some(result) = gatify_array_index_near_pow2_mux_tree_if_type_fits(
                gb, array_ty, array_bits, index_bits,
            ) {
                return result;
            }
        }
    }

    gatify_array_index_oob_one_hot(gb, array_ty, array_bits, index_bits)
}

/// Lowers an in-bounds array index with an exact one-hot selection.
fn gatify_array_index_exact(
    gb: &mut GateBuilder,
    array_ty: &ir::ArrayTypeData,
    array_bits: &AigBitVector,
    index_bits: &AigBitVector,
) -> AigBitVector {
    let element_bit_count = array_ty.element_type.bit_count();
    let index_decoded = gatify_decode(gb, array_ty.element_count, index_bits);
    let mut cases = Vec::new();
    for i in 0..array_ty.element_count {
        let case_bits = array_bits.get_lsb_slice(i * element_bit_count, element_bit_count);
        cases.push(case_bits);
    }
    gatify_one_hot_select(gb, &index_decoded, &cases)
}

/// Lowers a clamped array index with the existing decode-plus-one-hot path.
fn gatify_array_index_oob_one_hot(
    gb: &mut GateBuilder,
    array_ty: &ir::ArrayTypeData,
    array_bits: &AigBitVector,
    index_bits: &AigBitVector,
) -> AigBitVector {
    let element_bit_count = array_ty.element_type.bit_count();
    let array_element_count = array_ty.element_count;
    let index_decoded = gatify_decode(gb, array_element_count, index_bits);
    let oob = gb.add_ez(&index_decoded, ReductionKind::Tree);
    let one_hot_selector = AigBitVector::concat(oob.into(), index_decoded);

    // An array index selection is effectively a one hot selection of the elements
    // into a single element result.
    let mut cases = Vec::new();
    for i in 0..array_element_count {
        let case_bits = array_bits.get_lsb_slice(i * element_bit_count, element_bit_count);
        cases.push(case_bits);
    }
    cases.push(cases.last().unwrap().clone());
    gatify_one_hot_select(gb, &one_hot_selector, &cases)
}

/// Lowers a non-power-of-two clamped array index with a pad-last mux tree when
/// the index type fits.
fn gatify_array_index_near_pow2_mux_tree_if_type_fits(
    gb: &mut GateBuilder,
    array_ty: &ir::ArrayTypeData,
    array_bits: &AigBitVector,
    index_bits: &AigBitVector,
) -> Option<AigBitVector> {
    let array_element_count = array_ty.element_count;
    let padded_case_count = array_element_count.next_power_of_two();
    if padded_case_count == array_element_count {
        return None;
    }
    let required_index_bits = padded_case_count.trailing_zeros() as usize;
    if index_bits.get_bit_count() > required_index_bits {
        return None;
    }

    let mut cases = Vec::with_capacity(array_element_count);
    for i in 0..array_element_count {
        let element_bit_count = array_ty.element_type.bit_count();
        let case_bits = array_bits.get_lsb_slice(i * element_bit_count, element_bit_count);
        cases.push(case_bits);
    }
    gatify_indexed_select_mux_tree_pad_last_if_type_fits(gb, index_bits, &cases).ok()
}

/// Uses the pad-last mux-tree lowering only in the measured profitable region.
fn gatify_array_index_near_pow2_mux_tree_if_profitable(
    gb: &mut GateBuilder,
    array_ty: &ir::ArrayTypeData,
    array_bits: &AigBitVector,
    index_bits: &AigBitVector,
) -> Option<AigBitVector> {
    const MAX_PADDED_COUNT: usize = 32;
    const MAX_EXTRA_ELEMS: usize = 8;
    const MAX_ELEMENT_BIT_COUNT: usize = 2;

    let element_bit_count = array_ty.element_type.bit_count();
    let array_element_count = array_ty.element_count;
    let padded_case_count = array_element_count.next_power_of_two();
    if element_bit_count == 0
        || element_bit_count > MAX_ELEMENT_BIT_COUNT
        || padded_case_count == array_element_count
        || padded_case_count > MAX_PADDED_COUNT
        || padded_case_count - array_element_count > MAX_EXTRA_ELEMS
    {
        return None;
    }

    gatify_array_index_near_pow2_mux_tree_if_type_fits(gb, array_ty, array_bits, index_bits)
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
    if let Some(result) =
        gatify_array_slice_element_mux_if_profitable(gb, array_ty, array_bits, start_bits, width)
    {
        return result;
    }

    gatify_array_slice_bit_shift(
        gb,
        array_ty,
        array_bits,
        start_bits,
        assumed_start_in_bounds,
        width,
        text_id,
        mul_adder_mapping,
    )
}

/// Builds the flat output bits for one concrete `array_slice` start value.
fn gatify_array_slice_window_case(
    array_bits: &AigBitVector,
    n_elems: usize,
    e_bits: usize,
    start: usize,
    width: usize,
) -> AigBitVector {
    debug_assert!(n_elems > 0);
    let last_idx = n_elems - 1;
    let last_elem = array_bits.get_lsb_slice(last_idx * e_bits, e_bits);
    let mut pad = AigBitVector::zeros(0);
    if width > 0 {
        for _ in 0..(width - 1) {
            pad = AigBitVector::concat(last_elem.clone(), pad);
        }
    }
    let extended = AigBitVector::concat(pad, array_bits.clone());
    let clamped_start = start.min(last_idx);
    extended.get_lsb_slice(clamped_start * e_bits, width * e_bits)
}

/// Lowers small awkward-width `array_slice`s as an exact mux over element
/// windows.
fn gatify_array_slice_element_mux_if_profitable(
    gb: &mut GateBuilder,
    array_ty: &ir::ArrayTypeData,
    array_bits: &AigBitVector,
    start_bits: &AigBitVector,
    width: usize,
) -> Option<AigBitVector> {
    const MAX_CASE_COUNT: usize = 64;
    const MAX_SLICE_WIDTH: usize = 4;
    const MAX_OUTPUT_BITS: usize = 256;

    let e_bits = array_ty.element_type.bit_count();
    let n_elems = array_ty.element_count;
    let start_w = start_bits.get_bit_count();
    let output_bits = width.checked_mul(e_bits)?;

    if n_elems == 0 {
        return None;
    }
    if output_bits == 0 {
        return Some(AigBitVector::zeros(0));
    }

    let case_count = 1usize.checked_shl(start_w as u32)?;
    if case_count > MAX_CASE_COUNT || width > MAX_SLICE_WIDTH || output_bits > MAX_OUTPUT_BITS {
        return None;
    }

    // The bit-shift lowering must synthesize `start * e_bits`; this is especially
    // expensive when `e_bits` is not a power of two. Single-bit elements also
    // consistently benefit from selecting element windows directly.
    if e_bits != 1 && e_bits.is_power_of_two() {
        return None;
    }

    let cases: Vec<AigBitVector> = (0..case_count)
        .map(|start| gatify_array_slice_window_case(array_bits, n_elems, e_bits, start, width))
        .collect();
    gatify_indexed_select_mux_tree_exact(gb, start_bits, &cases).ok()
}

/// Lowers `array_slice` by scaling the start index and shifting the flat array.
fn gatify_array_slice_bit_shift(
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
    let last_idx = n_elems.saturating_sub(1);
    let last_idx_w = std::cmp::max(
        1,
        (usize::BITS as usize) - last_idx.leading_zeros() as usize,
    );
    let clamped_start_bits = if assumed_start_in_bounds {
        start_bits.clone()
    } else if start_w < last_idx_w {
        // `start_bits` cannot represent `last_idx` (or any larger index), so an
        // out-of-bounds start is impossible from the type alone.
        //
        // Example: for 8 elements, `last_idx = 7` (needs 3 bits). If `start_w = 2`,
        // representable starts are only 0..=3, all in-bounds. Clamping would be a
        // semantic no-op, so keep `start_bits` as-is and avoid creating an
        // out-of-range literal like bits[2]:7.
        start_bits.clone()
    } else {
        let last_idx_bits =
            gb.add_literal(&xlsynth::IrBits::make_ubits(start_w, last_idx as u64).unwrap());
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
    for elem_bits in updated_elems {
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

pub(super) fn gatify_zext_or_truncate(
    new_bit_count: usize,
    arg_bits: &AigBitVector,
) -> AigBitVector {
    match arg_bits.get_bit_count().cmp(&new_bit_count) {
        std::cmp::Ordering::Less => gatify_zero_ext(new_bit_count, arg_bits),
        std::cmp::Ordering::Equal => arg_bits.clone(),
        std::cmp::Ordering::Greater => arg_bits.get_lsb_slice(0, new_bit_count),
    }
}

pub(super) fn gatify_sext_or_truncate(
    gb: &mut GateBuilder,
    text_id: usize,
    new_bit_count: usize,
    arg_bits: &AigBitVector,
) -> AigBitVector {
    match arg_bits.get_bit_count().cmp(&new_bit_count) {
        std::cmp::Ordering::Less => {
            if arg_bits.get_bit_count() == 0 {
                AigBitVector::zeros(new_bit_count)
            } else {
                gatify_sign_ext(gb, text_id, new_bit_count, arg_bits)
            }
        }
        std::cmp::Ordering::Equal => arg_bits.clone(),
        std::cmp::Ordering::Greater => arg_bits.get_lsb_slice(0, new_bit_count),
    }
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

fn shll_const_and_resize(
    arg_bits: &AigBitVector,
    shift: usize,
    output_bit_count: usize,
    gb: &mut GateBuilder,
) -> AigBitVector {
    if output_bit_count == 0 {
        return AigBitVector::zeros(0);
    }
    let mut shifted = vec![gb.get_false(); shift];
    shifted.extend(arg_bits.iter_lsb_to_msb().cloned());
    if shifted.len() > output_bit_count {
        shifted.truncate(output_bit_count);
    }
    while shifted.len() < output_bit_count {
        shifted.push(gb.get_false());
    }
    AigBitVector::from_lsb_is_index_0(&shifted)
}

fn usize_as_aig_bits(value: usize, bit_count: usize, gb: &mut GateBuilder) -> AigBitVector {
    let mut bits = Vec::with_capacity(bit_count);
    for i in 0..bit_count {
        let bit_is_set = if i < usize::BITS as usize {
            ((value >> i) & 1) != 0
        } else {
            false
        };
        bits.push(if bit_is_set {
            gb.get_true()
        } else {
            gb.get_false()
        });
    }
    AigBitVector::from_lsb_is_index_0(&bits)
}

fn gatify_umul_const_via_csd(
    multiplicand_bits: &AigBitVector,
    constant_bits: &xlsynth::IrBits,
    output_bit_count: usize,
    mul_adder_mapping: AdderMapping,
    gb: &mut GateBuilder,
) -> AigBitVector {
    if output_bit_count == 0 {
        return AigBitVector::zeros(0);
    }
    let terms = decompose_umul_const_terms(constant_bits, output_bit_count);
    if terms.is_empty() {
        return AigBitVector::zeros(output_bit_count);
    }

    let mut operands: Vec<AigBitVector> = Vec::with_capacity(terms.len() + 1);
    let mut neg_term_count = 0usize;
    for term in terms {
        let shifted = shll_const_and_resize(multiplicand_bits, term.shift, output_bit_count, gb);
        match term.sign {
            SignedDigitSign::Plus => operands.push(shifted),
            SignedDigitSign::Minus => {
                operands.push(gb.add_not_vec(&shifted));
                neg_term_count = neg_term_count.saturating_add(1);
            }
        }
    }
    if neg_term_count != 0 {
        operands.push(usize_as_aig_bits(neg_term_count, output_bit_count, gb));
    }
    array_add_with_carry_out(gb, &operands, None, mul_adder_mapping).sum
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
            let bit_count = lhs_bits.get_bit_count();
            let lhs_abs = gatify_abs(lhs_bits, gb);
            let rhs_abs = gatify_abs(rhs_bits, gb);
            let unsigned = gatify_udiv(&lhs_abs, &rhs_abs, gb);
            let sign_a = lhs_bits.get_msb(0);
            let sign_b = rhs_bits.get_msb(0);
            let result_neg = gb.add_xor_binary(*sign_a, *sign_b);
            let negated = gatify_twos_complement(&unsigned, gb);
            let signed_result = gb.add_mux2_vec(&result_neg, &negated, &unsigned);
            let divisor_zero = gb.add_ez(rhs_bits, ReductionKind::Tree);
            let mut signed_max_bits = vec![gb.get_true(); bit_count];
            if let Some(msb) = signed_max_bits.last_mut() {
                *msb = gb.get_false();
            }
            let signed_max = AigBitVector::from_lsb_is_index_0(&signed_max_bits);
            let mut signed_min_bits = vec![gb.get_false(); bit_count];
            if let Some(msb) = signed_min_bits.last_mut() {
                *msb = gb.get_true();
            }
            let signed_min = AigBitVector::from_lsb_is_index_0(&signed_min_bits);
            let zero_div_result = gb.add_mux2_vec(sign_a, &signed_min, &signed_max);
            gb.add_mux2_vec(&divisor_zero, &zero_div_result, &signed_result)
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

/// Returns `(cmp, eq)` for `lhs_bits[start..start+len]` vs the same RHS range.
///
/// `cmp` is true when the most-significant differing bit in the range satisfies
/// `handle_bit(lhs_bit, rhs_bit)`. For unsigned less-than, `handle_bit` is
/// `!lhs_bit & rhs_bit`; for greater-than, it is `lhs_bit & !rhs_bit`.
fn gatify_ucmp_and_eq_range_via_bit_tests<F>(
    gb: &mut GateBuilder,
    lhs_bits: &AigBitVector,
    rhs_bits: &AigBitVector,
    start: usize,
    len: usize,
    handle_bit: &F,
) -> (AigOperand, AigOperand)
where
    F: Fn(&mut GateBuilder, AigOperand, AigOperand) -> AigOperand,
{
    debug_assert!(len > 0);
    debug_assert!(start + len <= lhs_bits.get_bit_count());
    debug_assert!(start + len <= rhs_bits.get_bit_count());

    if len == 1 {
        let lhs_bit = *lhs_bits.get_lsb(start);
        let rhs_bit = *rhs_bits.get_lsb(start);
        let cmp = handle_bit(gb, lhs_bit, rhs_bit);
        let eq = gb.add_xnor(lhs_bit, rhs_bit);
        return (cmp, eq);
    }

    let low_len = len / 2;
    let high_len = len - low_len;
    let (low_cmp, low_eq) =
        gatify_ucmp_and_eq_range_via_bit_tests(gb, lhs_bits, rhs_bits, start, low_len, handle_bit);
    let (high_cmp, high_eq) = gatify_ucmp_and_eq_range_via_bit_tests(
        gb,
        lhs_bits,
        rhs_bits,
        start + low_len,
        high_len,
        handle_bit,
    );

    // Compare the more-significant half first. The less-significant half only
    // matters when the high halves are equal.
    let high_eq_and_low_cmp = gb.add_and_binary(high_eq, low_cmp);
    let cmp = gb.add_or_binary(high_cmp, high_eq_and_low_cmp);
    let eq = gb.add_and_binary(high_eq, low_eq);
    (cmp, eq)
}

/// This is the generalization of unsigned comparisons via bit tests.
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
    assert!(input_bit_count > 0, "ucmp requires non-zero-width operands");
    let (cmp, eq) = gatify_ucmp_and_eq_range_via_bit_tests(
        gb,
        lhs_bits,
        rhs_bits,
        0,
        input_bit_count,
        handle_bit,
    );
    if or_eq {
        gb.add_or_binary(cmp, eq)
    } else {
        cmp
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
    assert!(input_bit_count > 0, "ucmp requires non-zero-width operands");
    gatify_ucmp_and_eq_range_via_bit_tests(
        gb,
        lhs_bits,
        rhs_bits,
        0,
        input_bit_count,
        &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
            let lhs_bit_unset = gb.add_not(lhs_bit);
            gb.add_and_binary(lhs_bit_unset, rhs_bit)
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

pub(super) fn literal_bits_if_bits_node(
    f: &ir::Fn,
    node_ref: ir::NodeRef,
) -> Option<xlsynth::IrBits> {
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

pub(super) fn get_pow2_minus1_k(bits: &xlsynth::IrBits) -> Option<usize> {
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

fn normalize_umul_literal_rhs(
    f: &ir::Fn,
    a: ir::NodeRef,
    b: ir::NodeRef,
) -> Option<(ir::NodeRef, xlsynth::IrBits)> {
    let a_lit = literal_bits_if_bits_node(f, a);
    let b_lit = literal_bits_if_bits_node(f, b);
    match (a_lit, b_lit) {
        (None, Some(rhs_bits)) => Some((a, rhs_bits)),
        (Some(rhs_bits), None) => Some((b, rhs_bits)),
        (Some(_), Some(_)) => None,
        (None, None) => None,
    }
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
            } else {
                let u = gatify_ucmp_literal_rhs_best_effort(
                    gb,
                    ir::Binop::Ult,
                    lhs_bits,
                    rhs_bits_vec,
                    rhs_bits,
                );
                if is_non_negative_signed(rhs_bits) {
                    // rhs is non-negative: any negative lhs is strictly less.
                    Some(gb.add_or_binary(msb, u))
                } else {
                    // rhs is negative: lhs must also be negative, and then signed
                    // ordering matches unsigned ordering.
                    Some(gb.add_and_binary(msb, u))
                }
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
            } else {
                let u = gatify_ucmp_literal_rhs_best_effort(
                    gb,
                    ir::Binop::Ule,
                    lhs_bits,
                    rhs_bits_vec,
                    rhs_bits,
                );
                if is_non_negative_signed(rhs_bits) {
                    // rhs is non-negative: any negative lhs is <= rhs.
                    Some(gb.add_or_binary(msb, u))
                } else {
                    // rhs is negative: lhs must be negative, and then signed
                    // ordering matches unsigned ordering.
                    Some(gb.add_and_binary(msb, u))
                }
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
            } else {
                let nonneg = gb.add_not(msb);
                let u = gatify_ucmp_literal_rhs_best_effort(
                    gb,
                    ir::Binop::Ugt,
                    lhs_bits,
                    rhs_bits_vec,
                    rhs_bits,
                );
                if is_non_negative_signed(rhs_bits) {
                    // rhs is non-negative: lhs must be non-negative, and then signed
                    // ordering matches unsigned ordering.
                    Some(gb.add_and_binary(nonneg, u))
                } else {
                    // rhs is negative: any non-negative lhs is strictly greater.
                    Some(gb.add_or_binary(nonneg, u))
                }
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
            } else {
                let nonneg = gb.add_not(msb);
                let u = gatify_ucmp_literal_rhs_best_effort(
                    gb,
                    ir::Binop::Uge,
                    lhs_bits,
                    rhs_bits_vec,
                    rhs_bits,
                );
                if is_non_negative_signed(rhs_bits) {
                    // rhs is non-negative: lhs must be non-negative, and then signed
                    // ordering matches unsigned ordering.
                    Some(gb.add_and_binary(nonneg, u))
                } else {
                    // rhs is negative: any non-negative lhs is >= rhs.
                    Some(gb.add_or_binary(nonneg, u))
                }
            }
        }

        _ => None,
    }
}

fn gatify_ucmp_literal_rhs_best_effort(
    gb: &mut GateBuilder,
    binop: ir::Binop,
    lhs_bits: &AigBitVector,
    rhs_bits_vec: &AigBitVector,
    rhs_bits: &xlsynth::IrBits,
) -> AigOperand {
    assert!(
        matches!(
            binop,
            ir::Binop::Ult | ir::Binop::Ule | ir::Binop::Ugt | ir::Binop::Uge
        ),
        "expected unsigned compare binop; got {:?}",
        binop
    );
    if let Some(gate) = try_simplify_cmp_literal_rhs(gb, binop, lhs_bits, rhs_bits_vec, rhs_bits) {
        gate
    } else if let Some(gate) = try_gatify_ucmp_literal_rhs_threshold(gb, binop, lhs_bits, rhs_bits)
    {
        gate
    } else {
        gatify_ucmp_fallback(gb, 0, binop, lhs_bits, rhs_bits_vec)
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

/// Emits a staged arithmetic-right barrel shifter with sign fill.
fn gatify_arithmetic_right_barrel_shifter(
    arg_bits: &AigBitVector,
    amount_bits: &AigBitVector,
    tag_prefix: &str,
    gb: &mut GateBuilder,
) -> AigBitVector {
    let bit_count = arg_bits.get_bit_count();
    assert!(bit_count > 0, "shra requires non-zero-width operands");

    let sign = *arg_bits.get_msb(0);
    let mut current: Vec<AigOperand> = arg_bits.iter_lsb_to_msb().cloned().collect();
    for stage in 0..amount_bits.get_bit_count() {
        let shift = 1usize << stage;
        let control = amount_bits.get_lsb(stage);
        let mut next_stage = Vec::with_capacity(bit_count);
        for j in 0..bit_count {
            let candidate = if j + shift < bit_count {
                current[j + shift]
            } else {
                sign
            };
            next_stage.push(gb.add_mux2(*control, candidate, current[j]));
        }
        current = next_stage;
    }

    for gate in current.iter() {
        gb.add_tag(
            gate.node,
            format!("{tag_prefix}_arithmetic_right_barrel_shifter_{bit_count}_count_output"),
        );
    }

    AigBitVector::from_lsb_is_index_0(&current)
}

/// Emits an arithmetic right shift.
fn gatify_shra(
    gb: &mut GateBuilder,
    arg_bits: &AigBitVector,
    amount_bits: &AigBitVector,
    text_id: usize,
    range_info: Option<&Arc<IrRangeInfo>>,
    amount_text_id: usize,
) -> AigBitVector {
    let w = arg_bits.get_bit_count();
    assert!(w > 0, "shra requires non-zero-width operands");

    // Effective shift bits needed to represent [0..w-1]. When w<=1, shifting is a
    // no-op.
    let required_k = if w <= 1 {
        0
    } else {
        xlsynth_pir::math::ceil_log2(w)
    };

    // If range_info proves amount < w, there can be no out-of-bounds case and we
    // may ignore any provably-irrelevant high bits.
    let (k_for_shift, oob): (usize, AigOperand) = if range_info.is_some_and(|ri| {
        // `proves_ult(amount, w)` means the dynamic shift amount is always in-bounds
        // for this consumer.
        ri.proves_ult(amount_text_id, w)
    }) {
        let max_effective_bits = range_info
            .and_then(|ri| ri.effective_amount_bits_for_ult(amount_text_id, w))
            .expect("effective_amount_bits_for_ult should succeed when proves_ult is true");
        let k = std::cmp::min(
            required_k,
            std::cmp::min(max_effective_bits, amount_bits.get_bit_count()),
        );
        (k, gb.get_false())
    } else {
        let amount_w = amount_bits.get_bit_count();
        let k = std::cmp::min(required_k, amount_w);
        let Split {
            msbs: amt_hi,
            lsbs: _,
        } = amount_bits.get_lsb_partition(k);

        // `oob_hi` is true when any higher (beyond `k`) shift amount bit is set.
        let oob = if amt_hi.get_bit_count() == 0 {
            gb.get_false()
        } else {
            gb.add_nez(&amt_hi, ReductionKind::Tree)
        };

        (k, oob)
    };

    // Sign bit (MSB) of the input.
    let sign = *arg_bits.get_msb(0);

    let amt_lo = amount_bits.get_lsb_slice(0, k_for_shift);
    let arith = if !w.is_power_of_two() && k_for_shift == required_k {
        // Low shift amounts in [w, next_power_of_two(w)) naturally produce all sign
        // bits through the staged sign-fill shifter, so only high amount bits need a
        // separate out-of-bounds mux.
        gatify_arithmetic_right_barrel_shifter(
            arg_bits,
            &amt_lo,
            &format!("shra_ext_{}", text_id),
            gb,
        )
    } else {
        // For power-of-two widths or too-narrow amount fields, the former
        // sign-extend-then-logical-shift shape remains slightly smaller after AIG
        // folding because there is no low-field saturation case to remove.
        let sign_ext = gb.replicate(sign, w);
        let arg_ext = AigBitVector::concat(sign_ext, arg_bits.clone());
        let shifted = gatify_barrel_shifter(
            &arg_ext,
            &amt_lo,
            Direction::Right,
            &format!("shra_ext_{}", text_id),
            gb,
        );
        shifted.get_lsb_slice(0, w)
    };

    // Saturating/oob rule: shift >= w => all sign bits.
    if gb.is_known_false(oob) {
        arith
    } else {
        let all_sign = gb.replicate(sign, w);
        gb.add_mux2_vec(&oob, &all_sign, &arith)
    }
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
            // Array element 0 occupies the least-significant flat bits.
            for elem in elements.iter() {
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
                    options.array_index_lowering_strategy,
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
            for member in members.iter() {
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
            let in_w = arg_bits.get_bit_count();
            let expected_out_w = xlsynth_pir::math::ceil_log2(in_w.saturating_add(1));
            if node.ty.bit_count() != expected_out_w {
                return Err(format!(
                    "ExtPrioEncode output width mismatch; expected {} got {}",
                    expected_out_w,
                    node.ty.bit_count()
                ));
            }

            let (any, idx_bits) = gatify_prio_encode(g8_builder, &arg_bits, *lsb_prio)
                .map_err(|e| format!("ExtPrioEncode lowering failed: {e}"))?;
            let idx_w = xlsynth_pir::math::ceil_log2(in_w);
            if idx_bits.get_bit_count() != idx_w {
                return Err(format!(
                    "ExtPrioEncode internal width mismatch; expected {} got {}",
                    idx_w,
                    idx_bits.get_bit_count()
                ));
            }

            let mut out: Vec<AigOperand> = Vec::with_capacity(expected_out_w);
            for bit_i in 0..expected_out_w {
                let idx_bit = if bit_i < idx_w {
                    *idx_bits.get_lsb(bit_i)
                } else {
                    g8_builder.get_false()
                };
                let sentinel_bit = if bit_i < usize::BITS as usize && ((in_w >> bit_i) & 1) == 1 {
                    g8_builder.get_true()
                } else {
                    g8_builder.get_false()
                };
                out.push(g8_builder.add_mux2(any, idx_bit, sentinel_bit));
            }

            let out_bits = AigBitVector::from_lsb_is_index_0(&out);
            for (i, gate) in out_bits.iter_lsb_to_msb().enumerate() {
                g8_builder.add_tag(
                    gate.node,
                    format!("ext_prio_encode_{}_output_bit_{}", node.text_id, i),
                );
            }
            env.add(node_ref, GateOrVec::BitVector(out_bits));
        }
        ir::NodePayload::ExtClz { arg } => {
            let arg_bits = env
                .get_bit_vector(*arg)
                .expect("ext_clz arg should be present");
            let in_w = arg_bits.get_bit_count();
            let expected_out_w = xlsynth_pir::math::ceil_log2(in_w.saturating_add(1));
            if node.ty.bit_count() != expected_out_w {
                return Err(format!(
                    "ExtClz output width mismatch; expected {} got {}",
                    expected_out_w,
                    node.ty.bit_count()
                ));
            }

            let (any, count_bits) = crate::ir2gate_utils::gatify_clz(g8_builder, &arg_bits)
                .map_err(|e| format!("ExtClz lowering failed: {e}"))?;
            let count_w = xlsynth_pir::math::ceil_log2(in_w);
            if count_bits.get_bit_count() != count_w {
                return Err(format!(
                    "ExtClz internal width mismatch; expected {} got {}",
                    count_w,
                    count_bits.get_bit_count()
                ));
            }

            let mut out: Vec<AigOperand> = Vec::with_capacity(expected_out_w);
            for bit_i in 0..expected_out_w {
                let count_bit = if bit_i < count_w {
                    *count_bits.get_lsb(bit_i)
                } else {
                    g8_builder.get_false()
                };
                let sentinel_bit = if bit_i < usize::BITS as usize && ((in_w >> bit_i) & 1) == 1 {
                    g8_builder.get_true()
                } else {
                    g8_builder.get_false()
                };
                out.push(g8_builder.add_mux2(any, count_bit, sentinel_bit));
            }

            let out_bits = AigBitVector::from_lsb_is_index_0(&out);
            for (i, gate) in out_bits.iter_lsb_to_msb().enumerate() {
                g8_builder.add_tag(
                    gate.node,
                    format!("ext_clz_{}_output_bit_{}", node.text_id, i),
                );
            }
            env.add(node_ref, GateOrVec::BitVector(out_bits));
        }
        ir::NodePayload::ExtNaryAdd { terms, arch } => {
            let ir::Type::Bits(output_width) = node.ty else {
                return Err("ExtNaryAdd result must be bits-typed".to_string());
            };
            let gates = gatify_ext_nary_add(
                f,
                &env,
                node.text_id,
                terms,
                *arch,
                output_width,
                options.adder_mapping,
                g8_builder,
            );
            env.add(node_ref, GateOrVec::BitVector(gates));
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
            let gates = gatify_add_binop(
                f,
                &env,
                node.text_id,
                *a,
                *b,
                options.adder_mapping,
                g8_builder,
            );
            env.add(node_ref, GateOrVec::BitVector(gates));
        }
        ir::NodePayload::Binop(ir::Binop::Sub, a, b) => {
            let gates = gatify_sub_binop(
                &env,
                node.text_id,
                node.ty.bit_count(),
                *a,
                *b,
                options.adder_mapping,
                g8_builder,
            );
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
            let amount_text_id = f.get_node(*amount).text_id;

            maybe_warn_shift_amount_truncatable(
                options.range_info.as_ref(),
                amount_text_id,
                arg_gates.get_bit_count(),
                &amount_gates,
            );
            let result = gatify_shra(
                g8_builder,
                &arg_gates,
                &amount_gates,
                node.text_id,
                options.range_info.as_ref(),
                amount_text_id,
            );
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
            let signedness = if matches!(node.payload, ir::NodePayload::Binop(ir::Binop::Smul, ..))
            {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            let mul_adder_mapping = options.mul_adder_mapping.unwrap_or(options.adder_mapping);
            let gates = if matches!(signedness, Signedness::Unsigned) {
                if let Some((multiplicand, constant_bits)) =
                    normalize_umul_literal_rhs(f, *lhs, *rhs)
                {
                    let multiplicand_bits = env
                        .get_bit_vector(multiplicand)
                        .expect("mul multiplicand should be present");
                    gatify_umul_const_via_csd(
                        &multiplicand_bits,
                        &constant_bits,
                        output_bit_count,
                        mul_adder_mapping,
                        g8_builder,
                    )
                } else {
                    let lhs_bits = env.get_bit_vector(*lhs).expect("mul lhs should be present");
                    let rhs_bits = env.get_bit_vector(*rhs).expect("mul rhs should be present");
                    gatify_mul(
                        &lhs_bits,
                        &rhs_bits,
                        output_bit_count,
                        signedness,
                        mul_adder_mapping,
                        g8_builder,
                    )
                }
            } else {
                let lhs_bits = env.get_bit_vector(*lhs).expect("mul lhs should be present");
                let rhs_bits = env.get_bit_vector(*rhs).expect("mul rhs should be present");
                gatify_mul(
                    &lhs_bits,
                    &rhs_bits,
                    output_bit_count,
                    signedness,
                    mul_adder_mapping,
                    g8_builder,
                )
            };
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
            // `update_shifted` was zero-extended before shifting, so it is already
            // zero outside the write window selected by `mask`.
            let result_bits = g8_builder.add_or_vec(&cleared, &update_shifted);

            env.add(node_ref, GateOrVec::BitVector(result_bits));
        }
        _ => {
            let msg = format!("Unsupported node payload {:?}", payload);
            return Err(msg);
        }
    }
    env.apply_known_bits_if_any(node_ref, f, node, options, g8_builder);
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
        g8_builder.set_current_pir_node_id(Some(f.nodes[i + 1].text_id as u32));
        let gate_ref_vec = g8_builder.add_input(param.name.clone(), param.ty.bit_count());
        g8_builder.set_current_pir_node_id(None);
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
        g8_builder.set_current_pir_node_id(Some(
            u32::try_from(node.text_id).expect("node id too large for u32"),
        ));
        gatify_node(
            f,
            node_ref,
            node,
            g8_builder,
            env,
            options,
            &param_id_to_node_ref,
        )?;
        g8_builder.set_current_pir_node_id(None);
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
    pub enable_rewrite_nary_add: bool,
    pub array_index_lowering_strategy: ArrayIndexLoweringStrategy,
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

fn gatify_lower_prepared_fn(
    f: &ir::Fn,
    options: &GatifyOptions,
    orig_ref_by_text_id: Option<&HashMap<usize, ir::NodeRef>>,
    equiv_fn: &ir::Fn,
) -> Result<GatifyOutput, String> {
    let mut g8_builder = GateBuilder::new(
        f.name.clone(),
        GateBuilderOptions {
            fold: options.fold,
            hash: options.hash,
        },
    );
    let mut env = GateEnv::new();
    gatify_internal(f, &mut g8_builder, &mut env, options)?;
    let gate_fn = g8_builder.build();
    log::debug!(
        "converted IR function to gate function:\n{}",
        gate_fn.to_string()
    );

    let mut lowering_map: IrToGateMap = HashMap::new();
    for (node_ref, gate_or_vec) in env.ir_to_g8.into_iter() {
        let bit_vector = match gate_or_vec {
            GateOrVec::BitVector(bv) => bv,
            GateOrVec::Gate(gate_ref) => AigBitVector::from_bit(gate_ref),
        };
        if let Some(orig_ref_by_text_id) = orig_ref_by_text_id {
            let prepared_text_id = f.get_node(node_ref).text_id;
            let Some(orig_node_ref) = orig_ref_by_text_id.get(&prepared_text_id).copied() else {
                continue;
            };
            lowering_map.insert(orig_node_ref, bit_vector);
        } else {
            lowering_map.insert(node_ref, bit_vector);
        }
    }

    if options.check_equivalence {
        log::info!("checking equivalence of IR function and gate function...");
        check_equivalence::validate_same_fn(equiv_fn, &gate_fn)?;
    }
    Ok(GatifyOutput {
        gate_fn,
        lowering_map,
    })
}

/// Lowers an IR function that has already been prepared for gatification.
///
/// This skips `prep_for_gatify`; callers are responsible for running any
/// desired prep rewrites first.
pub fn gatify_prepared_fn(f: &ir::Fn, options: GatifyOptions) -> Result<GatifyOutput, String> {
    validate_fn_for_gatify(f)
        .map_err(|e| format!("PIR validation failed before gatify_prepared_fn: {e}"))?;
    gatify_lower_prepared_fn(f, &options, None, f)
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
            enable_rewrite_nary_add: options.enable_rewrite_nary_add,
            ..PrepForGatifyOptions::all_opts_enabled()
        },
    );
    validate_fn_for_gatify(&prepared_fn)
        .map_err(|e| format!("PIR validation failed after prep_for_gatify: {e}"))?;
    gatify_lower_prepared_fn(&prepared_fn, &options, Some(&orig_ref_by_text_id), orig_fn)
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
            enable_rewrite_nary_add: options.enable_rewrite_nary_add,
            ..PrepForGatifyOptions::all_opts_enabled()
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
    use crate::aig::gate::{AigNode, GateFn, Split};
    use crate::aig::get_summary_stats::{AigStats, SummaryStats, get_aig_stats, get_summary_stats};
    use crate::aig::{AigBitVector, AigOperand};
    use crate::aig_sim::gate_sim;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions, ReductionKind};
    use crate::gatify::ir2gate::{GatifyOptions, gatify};
    use crate::ir2gate_utils::{AdderMapping, Direction, gatify_barrel_shifter};
    use xlsynth::{IrBits, IrValue};
    use xlsynth_pir::ir;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth_pir::ir_parser;
    use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;

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
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
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
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
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

    #[test]
    fn test_gatify_seeds_pir_node_ids_on_inputs_and_lowered_ands() {
        let ir_text = r#"package sample
fn f(a: bits[2] id=1, b: bits[2] id=2) -> bits[2] {
  ret add.3: bits[2] = add(a, b, id=3)
}
"#;
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();

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
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .unwrap();

        let gate_fn = gatify_output.gate_fn;
        for input in &gate_fn.inputs {
            let expected_id = if input.name == "a" { 1 } else { 2 };
            for bit in input.bit_vector.iter_lsb_to_msb() {
                assert_eq!(
                    gate_fn.gates[bit.node.id].get_pir_node_ids(),
                    &[expected_id],
                    "input bit {} should carry PIR provenance id {}",
                    input.name,
                    expected_id
                );
            }
        }

        for node in &gate_fn.gates {
            if let AigNode::And2 { .. } = node {
                assert_eq!(
                    node.get_pir_node_ids(),
                    &[3],
                    "every lowered AND for this simple add should carry the add node text_id"
                );
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum CmpLowering {
        OldRepeatedPrefix,
        Public,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct CmpQorRow {
        binop: ir::Binop,
        width: usize,
        old_and_nodes: usize,
        old_depth: usize,
        public_and_nodes: usize,
        public_depth: usize,
    }

    fn gatify_ucmp_via_repeated_prefix_bit_tests_old<F>(
        gb: &mut GateBuilder,
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
        assert!(input_bit_count > 0, "ucmp requires non-zero-width operands");
        let eq_bits = gb.add_xnor_vec(lhs_bits, rhs_bits);
        let mut bit_tests = Vec::new();

        for msb_i in 0..input_bit_count {
            let eq_bits_slice = eq_bits.get_msbs(msb_i);
            let prior_bits_equal = if eq_bits_slice.is_empty() {
                gb.get_true()
            } else {
                gb.add_and_reduce(&eq_bits_slice, ReductionKind::Tree)
            };
            let lhs_bit = *lhs_bits.get_msb(msb_i);
            let rhs_bit = *rhs_bits.get_msb(msb_i);
            let bit_test = handle_bit(gb, lhs_bit, rhs_bit);
            bit_tests.push(gb.add_and_binary(prior_bits_equal, bit_test));
        }

        let cmp = gb.add_or_nary(&bit_tests, ReductionKind::Tree);
        if or_eq {
            let eq = gb.add_and_reduce(&eq_bits, ReductionKind::Tree);
            gb.add_or_binary(cmp, eq)
        } else {
            cmp
        }
    }

    fn gatify_ult_and_eq_via_repeated_prefix_bit_tests_old(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
    ) -> (AigOperand, AigOperand) {
        let ult = gatify_ucmp_via_repeated_prefix_bit_tests_old(
            gb,
            lhs_bits,
            rhs_bits,
            false,
            &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                let lhs_bit_unset = gb.add_not(lhs_bit);
                gb.add_and_binary(lhs_bit_unset, rhs_bit)
            },
        );
        let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
        (ult, eq)
    }

    fn gatify_scmp_via_repeated_prefix_bit_tests_old(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
        cmp_kind: super::CmpKind,
        or_eq: bool,
    ) -> AigOperand {
        assert_eq!(lhs_bits.get_bit_count(), rhs_bits.get_bit_count());
        assert!(lhs_bits.get_bit_count() > 0);
        let bit_count = lhs_bits.get_bit_count();
        if bit_count == 1 {
            let a = *lhs_bits.get_lsb(0);
            let b = *rhs_bits.get_lsb(0);
            return match cmp_kind {
                super::CmpKind::Lt => {
                    let b_complement = gb.add_not(b);
                    let slt = gb.add_and_binary(a, b_complement);
                    if or_eq {
                        let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
                        gb.add_or_binary(slt, eq)
                    } else {
                        slt
                    }
                }
                super::CmpKind::Gt => {
                    let a_complement = gb.add_not(a);
                    let sgt = gb.add_and_binary(a_complement, b);
                    if or_eq {
                        let eq = gb.add_eq_vec(lhs_bits, rhs_bits, ReductionKind::Tree);
                        gb.add_or_binary(sgt, eq)
                    } else {
                        sgt
                    }
                }
            };
        }

        let a_msb = lhs_bits.get_msb(0);
        let b_msb = rhs_bits.get_msb(0);
        let sign_diff = gb.add_xor_binary(*a_msb, *b_msb);

        let (ult, eq) = gatify_ult_and_eq_via_repeated_prefix_bit_tests_old(gb, lhs_bits, rhs_bits);
        let term1 = gb.add_and_binary(sign_diff, *a_msb);
        let not_sign_diff = gb.add_not(sign_diff);
        let term2 = gb.add_and_binary(not_sign_diff, ult);
        let lt = gb.add_or_binary(term1, term2);
        match cmp_kind {
            super::CmpKind::Lt => {
                if or_eq {
                    gb.add_or_binary(lt, eq)
                } else {
                    lt
                }
            }
            super::CmpKind::Gt => {
                let lt_or_eq = gb.add_or_binary(lt, eq);
                let gt = gb.add_not(lt_or_eq);
                if or_eq { gb.add_or_binary(gt, eq) } else { gt }
            }
        }
    }

    fn gatify_cmp_for_qor_test(
        gb: &mut GateBuilder,
        lhs_bits: &AigBitVector,
        rhs_bits: &AigBitVector,
        binop: ir::Binop,
        lowering: CmpLowering,
    ) -> AigOperand {
        match lowering {
            CmpLowering::Public => match binop {
                ir::Binop::Ult => super::gatify_ult_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Ule => super::gatify_ule_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Ugt => super::gatify_ugt_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Uge => super::gatify_uge_via_bit_tests(gb, 0, lhs_bits, rhs_bits),
                ir::Binop::Slt => super::gatify_scmp_via_bit_tests(
                    gb,
                    0,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    false,
                ),
                ir::Binop::Sle => super::gatify_scmp_via_bit_tests(
                    gb,
                    0,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    true,
                ),
                ir::Binop::Sgt => super::gatify_scmp_via_bit_tests(
                    gb,
                    0,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    false,
                ),
                ir::Binop::Sge => super::gatify_scmp_via_bit_tests(
                    gb,
                    0,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    true,
                ),
                other => panic!("unexpected cmp binop in QoR test: {other:?}"),
            },
            CmpLowering::OldRepeatedPrefix => match binop {
                ir::Binop::Ult => gatify_ucmp_via_repeated_prefix_bit_tests_old(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    false,
                    &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                        let lhs_bit_unset = gb.add_not(lhs_bit);
                        gb.add_and_binary(lhs_bit_unset, rhs_bit)
                    },
                ),
                ir::Binop::Ule => gatify_ucmp_via_repeated_prefix_bit_tests_old(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    true,
                    &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                        let lhs_bit_unset = gb.add_not(lhs_bit);
                        gb.add_and_binary(lhs_bit_unset, rhs_bit)
                    },
                ),
                ir::Binop::Ugt => gatify_ucmp_via_repeated_prefix_bit_tests_old(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    false,
                    &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                        let rhs_bit_unset = gb.add_not(rhs_bit);
                        gb.add_and_binary(lhs_bit, rhs_bit_unset)
                    },
                ),
                ir::Binop::Uge => gatify_ucmp_via_repeated_prefix_bit_tests_old(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    true,
                    &|gb: &mut GateBuilder, lhs_bit, rhs_bit| {
                        let rhs_bit_unset = gb.add_not(rhs_bit);
                        gb.add_and_binary(lhs_bit, rhs_bit_unset)
                    },
                ),
                ir::Binop::Slt => gatify_scmp_via_repeated_prefix_bit_tests_old(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    false,
                ),
                ir::Binop::Sle => gatify_scmp_via_repeated_prefix_bit_tests_old(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Lt,
                    true,
                ),
                ir::Binop::Sgt => gatify_scmp_via_repeated_prefix_bit_tests_old(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    false,
                ),
                ir::Binop::Sge => gatify_scmp_via_repeated_prefix_bit_tests_old(
                    gb,
                    lhs_bits,
                    rhs_bits,
                    super::CmpKind::Gt,
                    true,
                ),
                other => panic!("unexpected cmp binop in QoR test: {other:?}"),
            },
        }
    }

    fn build_cmp_gate_fn_for_qor_test(
        width: usize,
        binop: ir::Binop,
        lowering: CmpLowering,
    ) -> GateFn {
        let mut gb = GateBuilder::new(
            format!("cmp_{lowering:?}_{binop:?}_{width}b"),
            GateBuilderOptions::opt(),
        );
        let lhs_bits = gb.add_input("lhs".to_string(), width);
        let rhs_bits = gb.add_input("rhs".to_string(), width);
        let result = gatify_cmp_for_qor_test(&mut gb, &lhs_bits, &rhs_bits, binop, lowering);
        gb.add_output("result".to_string(), AigBitVector::from_bit(result));
        gb.build()
    }

    fn get_cmp_qor_test_stats(width: usize, binop: ir::Binop, lowering: CmpLowering) -> AigStats {
        get_aig_stats(&build_cmp_gate_fn_for_qor_test(width, binop, lowering))
    }

    fn validate_cmp_old_vs_public_simulation(width: usize, binop: ir::Binop) {
        let old_gate_fn =
            build_cmp_gate_fn_for_qor_test(width, binop, CmpLowering::OldRepeatedPrefix);
        let public_gate_fn = build_cmp_gate_fn_for_qor_test(width, binop, CmpLowering::Public);
        let value_count = 1usize << width;
        for lhs in 0..value_count {
            for rhs in 0..value_count {
                let lhs_bits = IrBits::make_ubits(width, lhs as u64).unwrap();
                let rhs_bits = IrBits::make_ubits(width, rhs as u64).unwrap();
                let old = gate_sim::eval(
                    &old_gate_fn,
                    &[lhs_bits.clone(), rhs_bits.clone()],
                    gate_sim::Collect::None,
                );
                let public = gate_sim::eval(
                    &public_gate_fn,
                    &[lhs_bits, rhs_bits],
                    gate_sim::Collect::None,
                );
                assert_eq!(
                    public.outputs, old.outputs,
                    "cmp mismatch for binop={binop:?} width={width} lhs={lhs} rhs={rhs}"
                );
            }
        }
    }

    fn gather_cmp_qor_rows() -> Vec<CmpQorRow> {
        let mut got = Vec::new();
        for binop in [
            ir::Binop::Ult,
            ir::Binop::Ule,
            ir::Binop::Ugt,
            ir::Binop::Uge,
            ir::Binop::Slt,
            ir::Binop::Sle,
            ir::Binop::Sgt,
            ir::Binop::Sge,
        ] {
            for width in [3usize, 4, 5, 8, 16, 32] {
                if width <= 5 {
                    validate_cmp_old_vs_public_simulation(width, binop);
                }
                let old = get_cmp_qor_test_stats(width, binop, CmpLowering::OldRepeatedPrefix);
                let public = get_cmp_qor_test_stats(width, binop, CmpLowering::Public);
                got.push(CmpQorRow {
                    binop,
                    width,
                    old_and_nodes: old.and_nodes,
                    old_depth: old.max_depth,
                    public_and_nodes: public.and_nodes,
                    public_depth: public.max_depth,
                });
            }
        }
        got
    }

    #[test]
    fn test_cmp_recursive_bit_tree_qor_and_equivalence_sweep() {
        let got = gather_cmp_qor_rows();

        for row in &got {
            assert!(
                row.public_and_nodes <= row.old_and_nodes,
                "expected recursive cmp lowering not to increase AND nodes: {:?}",
                row
            );
            assert!(
                row.public_depth <= row.old_depth,
                "expected recursive cmp lowering not to increase depth: {:?}",
                row
            );
            assert!(
                row.public_and_nodes < row.old_and_nodes || row.public_depth < row.old_depth,
                "expected recursive cmp lowering to improve this row: {:?}",
                row
            );
        }

        #[rustfmt::skip]
        let want: &[CmpQorRow] = &[
            CmpQorRow { binop: ir::Binop::Ult, width: 3, old_and_nodes: 12, old_depth: 6, public_and_nodes: 12, public_depth: 5 },
            CmpQorRow { binop: ir::Binop::Ult, width: 4, old_and_nodes: 18, old_depth: 7, public_and_nodes: 17, public_depth: 6 },
            CmpQorRow { binop: ir::Binop::Ult, width: 5, old_and_nodes: 25, old_depth: 8, public_and_nodes: 23, public_depth: 6 },
            CmpQorRow { binop: ir::Binop::Ult, width: 8, old_and_nodes: 47, old_depth: 9, public_and_nodes: 40, public_depth: 8 },
            CmpQorRow { binop: ir::Binop::Ult, width: 16, old_and_nodes: 120, old_depth: 11, public_and_nodes: 87, public_depth: 10 },
            CmpQorRow { binop: ir::Binop::Ult, width: 32, old_and_nodes: 299, old_depth: 13, public_and_nodes: 182, public_depth: 12 },
            CmpQorRow { binop: ir::Binop::Ule, width: 3, old_and_nodes: 16, old_depth: 7, public_and_nodes: 16, public_depth: 6 },
            CmpQorRow { binop: ir::Binop::Ule, width: 4, old_and_nodes: 23, old_depth: 8, public_and_nodes: 22, public_depth: 7 },
            CmpQorRow { binop: ir::Binop::Ule, width: 5, old_and_nodes: 30, old_depth: 9, public_and_nodes: 28, public_depth: 7 },
            CmpQorRow { binop: ir::Binop::Ule, width: 8, old_and_nodes: 53, old_depth: 10, public_and_nodes: 46, public_depth: 9 },
            CmpQorRow { binop: ir::Binop::Ule, width: 16, old_and_nodes: 127, old_depth: 12, public_and_nodes: 94, public_depth: 11 },
            CmpQorRow { binop: ir::Binop::Ule, width: 32, old_and_nodes: 307, old_depth: 14, public_and_nodes: 190, public_depth: 13 },
            CmpQorRow { binop: ir::Binop::Ugt, width: 3, old_and_nodes: 12, old_depth: 6, public_and_nodes: 12, public_depth: 5 },
            CmpQorRow { binop: ir::Binop::Ugt, width: 4, old_and_nodes: 18, old_depth: 7, public_and_nodes: 17, public_depth: 6 },
            CmpQorRow { binop: ir::Binop::Ugt, width: 5, old_and_nodes: 25, old_depth: 8, public_and_nodes: 23, public_depth: 6 },
            CmpQorRow { binop: ir::Binop::Ugt, width: 8, old_and_nodes: 47, old_depth: 9, public_and_nodes: 40, public_depth: 8 },
            CmpQorRow { binop: ir::Binop::Ugt, width: 16, old_and_nodes: 120, old_depth: 11, public_and_nodes: 87, public_depth: 10 },
            CmpQorRow { binop: ir::Binop::Ugt, width: 32, old_and_nodes: 299, old_depth: 13, public_and_nodes: 182, public_depth: 12 },
            CmpQorRow { binop: ir::Binop::Uge, width: 3, old_and_nodes: 16, old_depth: 7, public_and_nodes: 16, public_depth: 6 },
            CmpQorRow { binop: ir::Binop::Uge, width: 4, old_and_nodes: 23, old_depth: 8, public_and_nodes: 22, public_depth: 7 },
            CmpQorRow { binop: ir::Binop::Uge, width: 5, old_and_nodes: 30, old_depth: 9, public_and_nodes: 28, public_depth: 7 },
            CmpQorRow { binop: ir::Binop::Uge, width: 8, old_and_nodes: 53, old_depth: 10, public_and_nodes: 46, public_depth: 9 },
            CmpQorRow { binop: ir::Binop::Uge, width: 16, old_and_nodes: 127, old_depth: 12, public_and_nodes: 94, public_depth: 11 },
            CmpQorRow { binop: ir::Binop::Uge, width: 32, old_and_nodes: 307, old_depth: 14, public_and_nodes: 190, public_depth: 13 },
            CmpQorRow { binop: ir::Binop::Slt, width: 3, old_and_nodes: 15, old_depth: 8, public_and_nodes: 15, public_depth: 7 },
            CmpQorRow { binop: ir::Binop::Slt, width: 4, old_and_nodes: 21, old_depth: 9, public_and_nodes: 20, public_depth: 8 },
            CmpQorRow { binop: ir::Binop::Slt, width: 5, old_and_nodes: 28, old_depth: 10, public_and_nodes: 26, public_depth: 8 },
            CmpQorRow { binop: ir::Binop::Slt, width: 8, old_and_nodes: 50, old_depth: 11, public_and_nodes: 43, public_depth: 10 },
            CmpQorRow { binop: ir::Binop::Slt, width: 16, old_and_nodes: 123, old_depth: 13, public_and_nodes: 90, public_depth: 12 },
            CmpQorRow { binop: ir::Binop::Slt, width: 32, old_and_nodes: 302, old_depth: 15, public_and_nodes: 185, public_depth: 14 },
            CmpQorRow { binop: ir::Binop::Sle, width: 3, old_and_nodes: 19, old_depth: 9, public_and_nodes: 19, public_depth: 8 },
            CmpQorRow { binop: ir::Binop::Sle, width: 4, old_and_nodes: 26, old_depth: 10, public_and_nodes: 25, public_depth: 9 },
            CmpQorRow { binop: ir::Binop::Sle, width: 5, old_and_nodes: 33, old_depth: 11, public_and_nodes: 31, public_depth: 9 },
            CmpQorRow { binop: ir::Binop::Sle, width: 8, old_and_nodes: 56, old_depth: 12, public_and_nodes: 49, public_depth: 11 },
            CmpQorRow { binop: ir::Binop::Sle, width: 16, old_and_nodes: 130, old_depth: 14, public_and_nodes: 97, public_depth: 13 },
            CmpQorRow { binop: ir::Binop::Sle, width: 32, old_and_nodes: 310, old_depth: 16, public_and_nodes: 193, public_depth: 15 },
            CmpQorRow { binop: ir::Binop::Sgt, width: 3, old_and_nodes: 19, old_depth: 9, public_and_nodes: 19, public_depth: 8 },
            CmpQorRow { binop: ir::Binop::Sgt, width: 4, old_and_nodes: 26, old_depth: 10, public_and_nodes: 25, public_depth: 9 },
            CmpQorRow { binop: ir::Binop::Sgt, width: 5, old_and_nodes: 33, old_depth: 11, public_and_nodes: 31, public_depth: 9 },
            CmpQorRow { binop: ir::Binop::Sgt, width: 8, old_and_nodes: 56, old_depth: 12, public_and_nodes: 49, public_depth: 11 },
            CmpQorRow { binop: ir::Binop::Sgt, width: 16, old_and_nodes: 130, old_depth: 14, public_and_nodes: 97, public_depth: 13 },
            CmpQorRow { binop: ir::Binop::Sgt, width: 32, old_and_nodes: 310, old_depth: 16, public_and_nodes: 193, public_depth: 15 },
            CmpQorRow { binop: ir::Binop::Sge, width: 3, old_and_nodes: 20, old_depth: 10, public_and_nodes: 20, public_depth: 9 },
            CmpQorRow { binop: ir::Binop::Sge, width: 4, old_and_nodes: 27, old_depth: 11, public_and_nodes: 26, public_depth: 10 },
            CmpQorRow { binop: ir::Binop::Sge, width: 5, old_and_nodes: 34, old_depth: 12, public_and_nodes: 32, public_depth: 10 },
            CmpQorRow { binop: ir::Binop::Sge, width: 8, old_and_nodes: 57, old_depth: 13, public_and_nodes: 50, public_depth: 12 },
            CmpQorRow { binop: ir::Binop::Sge, width: 16, old_and_nodes: 131, old_depth: 15, public_and_nodes: 98, public_depth: 14 },
            CmpQorRow { binop: ir::Binop::Sge, width: 32, old_and_nodes: 311, old_depth: 17, public_and_nodes: 194, public_depth: 16 },
        ];
        assert_eq!(got.as_slice(), want);
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ShraQorRow {
        width: usize,
        amount_width: usize,
        old_and_nodes: usize,
        old_depth: usize,
        public_and_nodes: usize,
        public_depth: usize,
    }

    fn build_shra_ir_text(width: usize, amount_width: usize) -> String {
        format!(
            r#"package sample

top fn main(x: bits[{width}], amt: bits[{amount_width}]) -> bits[{width}] {{
  ret y: bits[{width}] = shra(x, amt, id=3)
}}
"#
        )
    }

    fn gatify_shra_old_sign_ext_shift(
        gb: &mut GateBuilder,
        arg_bits: &AigBitVector,
        amount_bits: &AigBitVector,
    ) -> AigBitVector {
        let w = arg_bits.get_bit_count();
        assert!(w > 0);

        let required_k = if w <= 1 {
            0
        } else {
            xlsynth_pir::math::ceil_log2(w)
        };
        let amount_w = amount_bits.get_bit_count();
        let k = std::cmp::min(required_k, amount_w);
        let Split {
            msbs: amt_hi,
            lsbs: amt_lo,
        } = amount_bits.get_lsb_partition(k);

        let oob_hi = if amt_hi.get_bit_count() == 0 {
            gb.get_false()
        } else {
            gb.add_nez(&amt_hi, ReductionKind::Tree)
        };
        let oob_lo = if k == 0 || w.is_power_of_two() || k != required_k {
            gb.get_false()
        } else {
            let w_bits =
                IrBits::make_ubits(k, w as u64).expect("width must fit in shra low amount bits");
            super::try_gatify_ucmp_literal_rhs_threshold(gb, ir::Binop::Uge, &amt_lo, &w_bits)
                .expect("Uge threshold compare should be supported")
        };
        let oob = gb.add_or_binary(oob_hi, oob_lo);

        let sign = *arg_bits.get_msb(0);
        let sign_ext = gb.replicate(sign, w);
        let arg_ext = AigBitVector::concat(sign_ext, arg_bits.clone());
        let shifted =
            gatify_barrel_shifter(&arg_ext, &amt_lo, Direction::Right, "shra_old_ext", gb);
        let arith = shifted.get_lsb_slice(0, w);

        if gb.is_known_false(oob) {
            arith
        } else {
            let all_sign = gb.replicate(sign, w);
            gb.add_mux2_vec(&oob, &all_sign, &arith)
        }
    }

    fn get_shra_old_stats(width: usize, amount_width: usize) -> AigStats {
        let mut gb = GateBuilder::new(
            format!("shra_old_w{width}_amt{amount_width}"),
            GateBuilderOptions::opt(),
        );
        let arg_bits = gb.add_input("x".to_string(), width);
        let amount_bits = gb.add_input("amt".to_string(), amount_width);
        let result = gatify_shra_old_sign_ext_shift(&mut gb, &arg_bits, &amount_bits);
        gb.add_output("result".to_string(), result);
        get_aig_stats(&gb.build())
    }

    fn shra_sample_bits(width: usize) -> Vec<IrValue> {
        let all_ones = bit_mask(width);
        let sign_bit = 1u64 << (width - 1);
        let max_positive = sign_bit - 1;
        let alternating = 0xaaaa_aaaa_aaaa_aaaau64 & all_ones;
        let mut values = vec![
            0,
            1,
            all_ones,
            sign_bit,
            sign_bit | 1,
            max_positive,
            alternating,
        ];
        values.sort_unstable();
        values.dedup();
        values
            .into_iter()
            .map(|value| IrValue::make_ubits(width, value).unwrap())
            .collect()
    }

    fn validate_shra_public_simulation(
        ir_fn: &ir::Fn,
        gate_fn: &GateFn,
        width: usize,
        amount_width: usize,
    ) {
        let x_samples = shra_sample_bits(width);
        let amount_count = 1usize << amount_width;
        for x_value in &x_samples {
            for amount in 0..amount_count {
                let amount_value = IrValue::make_ubits(amount_width, amount as u64).unwrap();
                let want = match eval_fn(ir_fn, &[x_value.clone(), amount_value.clone()]) {
                    FnEvalResult::Success(success) => success.value.to_bits().unwrap(),
                    FnEvalResult::Failure(failure) => {
                        panic!("shra source IR failed during simulation: {failure:?}")
                    }
                };
                let sim = gate_sim::eval(
                    gate_fn,
                    &[x_value.to_bits().unwrap(), amount_value.to_bits().unwrap()],
                    gate_sim::Collect::None,
                );
                let got = sim.outputs[0].clone();
                assert_eq!(
                    got, want,
                    "shra simulation mismatch for width={width} amount_width={amount_width} \
                     amount={amount} x={x_value}"
                );
            }
        }
    }

    fn get_shra_public_stats_and_validate(width: usize, amount_width: usize) -> AigStats {
        let ir_text = build_shra_ir_text(width, amount_width);
        let mut parser = ir_parser::Parser::new(&ir_text);
        let ir_package = parser.parse_and_validate_package().expect("parse package");
        let ir_fn = ir_package.get_top_fn().expect("top fn");
        let gatify_output = gatify(
            ir_fn,
            GatifyOptions {
                fold: true,
                hash: true,
                check_equivalence: false,
                adder_mapping: AdderMapping::BrentKung,
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .expect("gatify shra");
        validate_shra_public_simulation(ir_fn, &gatify_output.gate_fn, width, amount_width);
        get_aig_stats(&gatify_output.gate_fn)
    }

    fn gather_shra_qor_rows() -> Vec<ShraQorRow> {
        let mut got = Vec::new();
        for (width, amount_widths) in [
            (3usize, &[2usize, 3, 4][..]),
            (4usize, &[2usize, 3, 4][..]),
            (5usize, &[2usize, 3, 4, 5][..]),
            (6usize, &[2usize, 3, 4][..]),
            (7usize, &[2usize, 3, 4][..]),
            (8usize, &[3usize, 4][..]),
            (9usize, &[3usize, 4, 5][..]),
            (16usize, &[4usize, 5][..]),
            (32usize, &[5usize, 6][..]),
        ] {
            for amount_width in amount_widths {
                let old = get_shra_old_stats(width, *amount_width);
                let public = get_shra_public_stats_and_validate(width, *amount_width);
                got.push(ShraQorRow {
                    width,
                    amount_width: *amount_width,
                    old_and_nodes: old.and_nodes,
                    old_depth: old.max_depth,
                    public_and_nodes: public.and_nodes,
                    public_depth: public.max_depth,
                });
            }
        }
        got
    }

    #[test]
    fn test_shra_arithmetic_barrel_qor_and_equivalence_sweep() {
        let _ = env_logger::builder().is_test(true).try_init();

        let got = gather_shra_qor_rows();

        for row in &got {
            assert!(
                row.public_and_nodes <= row.old_and_nodes,
                "expected shra lowering not to increase AND nodes: {:?}",
                row
            );
            assert!(
                row.public_depth <= row.old_depth,
                "expected shra lowering not to increase depth: {:?}",
                row
            );
            if !row.width.is_power_of_two()
                && row.amount_width >= xlsynth_pir::math::ceil_log2(row.width)
            {
                assert!(
                    row.public_and_nodes < row.old_and_nodes || row.public_depth < row.old_depth,
                    "expected non-power-of-two shra row to improve: {:?}",
                    row
                );
            }
        }

        #[rustfmt::skip]
        let want: &[ShraQorRow] = &[
            ShraQorRow { width: 3, amount_width: 2, old_and_nodes: 23, old_depth: 6, public_and_nodes: 16, public_depth: 4 },
            ShraQorRow { width: 3, amount_width: 3, old_and_nodes: 24, old_depth: 6, public_and_nodes: 23, public_depth: 6 },
            ShraQorRow { width: 3, amount_width: 4, old_and_nodes: 25, old_depth: 6, public_and_nodes: 24, public_depth: 6 },
            ShraQorRow { width: 4, amount_width: 2, old_and_nodes: 21, old_depth: 4, public_and_nodes: 21, public_depth: 4 },
            ShraQorRow { width: 4, amount_width: 3, old_and_nodes: 30, old_depth: 6, public_and_nodes: 30, public_depth: 6 },
            ShraQorRow { width: 4, amount_width: 4, old_and_nodes: 31, old_depth: 6, public_and_nodes: 31, public_depth: 6 },
            ShraQorRow { width: 5, amount_width: 2, old_and_nodes: 27, old_depth: 4, public_and_nodes: 27, public_depth: 4 },
            ShraQorRow { width: 5, amount_width: 3, old_and_nodes: 57, old_depth: 8, public_and_nodes: 40, public_depth: 6 },
            ShraQorRow { width: 5, amount_width: 4, old_and_nodes: 58, old_depth: 8, public_and_nodes: 51, public_depth: 8 },
            ShraQorRow { width: 5, amount_width: 5, old_and_nodes: 59, old_depth: 8, public_and_nodes: 52, public_depth: 8 },
            ShraQorRow { width: 6, amount_width: 2, old_and_nodes: 33, old_depth: 4, public_and_nodes: 33, public_depth: 4 },
            ShraQorRow { width: 6, amount_width: 3, old_and_nodes: 67, old_depth: 8, public_and_nodes: 49, public_depth: 6 },
            ShraQorRow { width: 6, amount_width: 4, old_and_nodes: 68, old_depth: 8, public_and_nodes: 62, public_depth: 8 },
            ShraQorRow { width: 7, amount_width: 2, old_and_nodes: 39, old_depth: 4, public_and_nodes: 39, public_depth: 4 },
            ShraQorRow { width: 7, amount_width: 3, old_and_nodes: 73, old_depth: 8, public_and_nodes: 58, public_depth: 6 },
            ShraQorRow { width: 7, amount_width: 4, old_and_nodes: 74, old_depth: 8, public_and_nodes: 73, public_depth: 8 },
            ShraQorRow { width: 8, amount_width: 3, old_and_nodes: 65, old_depth: 6, public_and_nodes: 65, public_depth: 6 },
            ShraQorRow { width: 8, amount_width: 4, old_and_nodes: 82, old_depth: 8, public_and_nodes: 82, public_depth: 8 },
            ShraQorRow { width: 9, amount_width: 3, old_and_nodes: 74, old_depth: 6, public_and_nodes: 74, public_depth: 6 },
            ShraQorRow { width: 9, amount_width: 4, old_and_nodes: 135, old_depth: 10, public_and_nodes: 96, public_depth: 8 },
            ShraQorRow { width: 9, amount_width: 5, old_and_nodes: 136, old_depth: 10, public_and_nodes: 115, public_depth: 10 },
            ShraQorRow { width: 16, amount_width: 4, old_and_nodes: 177, old_depth: 8, public_and_nodes: 177, public_depth: 8 },
            ShraQorRow { width: 16, amount_width: 5, old_and_nodes: 210, old_depth: 10, public_and_nodes: 210, public_depth: 10 },
            ShraQorRow { width: 32, amount_width: 5, old_and_nodes: 449, old_depth: 10, public_and_nodes: 449, public_depth: 10 },
            ShraQorRow { width: 32, amount_width: 6, old_and_nodes: 514, old_depth: 12, public_and_nodes: 514, public_depth: 12 },
        ];

        assert_eq!(got.as_slice(), want);
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct BitSliceUpdateQorRow {
        arg_width: usize,
        update_width: usize,
        old_and_nodes: usize,
        old_depth: usize,
        public_and_nodes: usize,
        public_depth: usize,
    }

    fn build_bit_slice_update_ir_text(
        arg_width: usize,
        start_width: usize,
        update_width: usize,
    ) -> String {
        format!(
            r#"package sample

top fn main(x: bits[{arg_width}], start: bits[{start_width}], update: bits[{update_width}]) -> bits[{arg_width}] {{
  ret y: bits[{arg_width}] = bit_slice_update(x, start, update, id=4)
}}
"#
        )
    }

    fn bit_mask(width: usize) -> u64 {
        if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        }
    }

    fn bit_slice_update_sample_bits(width: usize) -> Vec<IrValue> {
        let all_ones = bit_mask(width);
        let alternating = 0xaaaa_aaaa_aaaa_aaaau64 & all_ones;
        let mut values = vec![0, all_ones, alternating];
        for bit_index in [0, width / 2, width - 1] {
            values.push(1u64 << bit_index);
        }
        values.sort_unstable();
        values.dedup();
        values
            .into_iter()
            .map(|value| IrValue::make_ubits(width, value).unwrap())
            .collect()
    }

    fn gatify_bit_slice_update_old_insert_and(
        gb: &mut GateBuilder,
        arg_bits: &AigBitVector,
        start_bits: &AigBitVector,
        update_bits: &AigBitVector,
    ) -> AigBitVector {
        let arg_width = arg_bits.get_bit_count();
        let update_width = update_bits.get_bit_count();
        let effective_update_width = std::cmp::min(update_width, arg_width);

        let ones_effective = gb.replicate(gb.get_true(), effective_update_width);
        let zeros_high_count = arg_width - effective_update_width;
        let ones_ext = if zeros_high_count == 0 {
            ones_effective.clone()
        } else {
            let zeros = AigBitVector::zeros(zeros_high_count);
            AigBitVector::concat(zeros, ones_effective)
        };
        let mask = gatify_barrel_shifter(
            &ones_ext,
            start_bits,
            Direction::Left,
            "bit_slice_update_old_mask",
            gb,
        );

        let update_trim = if update_width > effective_update_width {
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
            start_bits,
            Direction::Left,
            "bit_slice_update_old_value",
            gb,
        );

        let mask_not = gb.add_not_vec(&mask);
        let cleared = gb.add_and_vec(arg_bits, &mask_not);
        let inserted = gb.add_and_vec(&update_shifted, &mask);
        gb.add_or_vec(&cleared, &inserted)
    }

    fn get_bit_slice_update_old_stats(
        arg_width: usize,
        start_width: usize,
        update_width: usize,
    ) -> AigStats {
        let mut gb = GateBuilder::new(
            format!("bit_slice_update_old_w{arg_width}_u{update_width}"),
            GateBuilderOptions::opt(),
        );
        let arg_bits = gb.add_input("x".to_string(), arg_width);
        let start_bits = gb.add_input("start".to_string(), start_width);
        let update_bits = gb.add_input("update".to_string(), update_width);
        let result =
            gatify_bit_slice_update_old_insert_and(&mut gb, &arg_bits, &start_bits, &update_bits);
        gb.add_output("result".to_string(), result);
        get_aig_stats(&gb.build())
    }

    fn validate_bit_slice_update_public_simulation(
        ir_fn: &ir::Fn,
        gate_fn: &GateFn,
        arg_width: usize,
        start_width: usize,
        update_width: usize,
    ) {
        let x_samples = bit_slice_update_sample_bits(arg_width);
        let update_samples = bit_slice_update_sample_bits(update_width);
        let start_count = 1usize << start_width;

        for x_value in &x_samples {
            for update_value in &update_samples {
                for start in 0..start_count {
                    let start_value = IrValue::make_ubits(start_width, start as u64).unwrap();
                    let want = match eval_fn(
                        ir_fn,
                        &[x_value.clone(), start_value.clone(), update_value.clone()],
                    ) {
                        FnEvalResult::Success(success) => success.value.to_bits().unwrap(),
                        FnEvalResult::Failure(failure) => {
                            panic!(
                                "bit_slice_update source IR failed during simulation: {failure:?}"
                            )
                        }
                    };
                    let sim = gate_sim::eval(
                        gate_fn,
                        &[
                            x_value.to_bits().unwrap(),
                            start_value.to_bits().unwrap(),
                            update_value.to_bits().unwrap(),
                        ],
                        gate_sim::Collect::None,
                    );
                    let got = sim.outputs[0].clone();
                    assert_eq!(
                        got, want,
                        "bit_slice_update simulation mismatch for arg_width={arg_width} \
                         update_width={update_width} start={start} x={x_value} \
                         update={update_value}"
                    );
                }
            }
        }
    }

    fn get_bit_slice_update_public_stats_and_validate(
        arg_width: usize,
        start_width: usize,
        update_width: usize,
    ) -> AigStats {
        let ir_text = build_bit_slice_update_ir_text(arg_width, start_width, update_width);
        let mut parser = ir_parser::Parser::new(&ir_text);
        let ir_package = parser.parse_and_validate_package().expect("parse package");
        let ir_fn = ir_package.get_top_fn().expect("top fn");
        let gatify_output = gatify(
            ir_fn,
            GatifyOptions {
                fold: true,
                hash: true,
                check_equivalence: false,
                adder_mapping: AdderMapping::BrentKung,
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .expect("gatify bit_slice_update");
        validate_bit_slice_update_public_simulation(
            ir_fn,
            &gatify_output.gate_fn,
            arg_width,
            start_width,
            update_width,
        );
        get_aig_stats(&gatify_output.gate_fn)
    }

    fn gather_bit_slice_update_qor_rows() -> Vec<BitSliceUpdateQorRow> {
        let mut got = Vec::new();
        for (arg_width, update_widths) in [
            (8usize, &[1usize, 2, 3, 5, 8][..]),
            (16usize, &[1usize, 2, 3, 5, 8, 16][..]),
            (32usize, &[1usize, 2, 3, 5, 8, 16][..]),
        ] {
            let start_width = (arg_width - 1).ilog2() as usize + 1;
            for update_width in update_widths {
                let old = get_bit_slice_update_old_stats(arg_width, start_width, *update_width);
                let public = get_bit_slice_update_public_stats_and_validate(
                    arg_width,
                    start_width,
                    *update_width,
                );
                got.push(BitSliceUpdateQorRow {
                    arg_width,
                    update_width: *update_width,
                    old_and_nodes: old.and_nodes,
                    old_depth: old.max_depth,
                    public_and_nodes: public.and_nodes,
                    public_depth: public.max_depth,
                });
            }
        }
        got
    }

    #[test]
    fn test_bit_slice_update_qor_and_equivalence_sweep() {
        let _ = env_logger::builder().is_test(true).try_init();

        let got = gather_bit_slice_update_qor_rows();

        for row in &got {
            assert!(
                row.public_and_nodes < row.old_and_nodes,
                "expected bit_slice_update lowering to reduce AND nodes: {:?}",
                row
            );
            assert!(
                row.public_depth <= row.old_depth,
                "expected bit_slice_update lowering not to increase depth: {:?}",
                row
            );
        }

        #[rustfmt::skip]
        let want: &[BitSliceUpdateQorRow] = &[
            BitSliceUpdateQorRow { arg_width: 8, update_width: 1, old_and_nodes: 50, old_depth: 5, public_and_nodes: 42, public_depth: 4 },
            BitSliceUpdateQorRow { arg_width: 8, update_width: 2, old_and_nodes: 64, old_depth: 6, public_and_nodes: 56, public_depth: 5 },
            BitSliceUpdateQorRow { arg_width: 8, update_width: 3, old_and_nodes: 75, old_depth: 7, public_and_nodes: 67, public_depth: 6 },
            BitSliceUpdateQorRow { arg_width: 8, update_width: 5, old_and_nodes: 95, old_depth: 8, public_and_nodes: 87, public_depth: 7 },
            BitSliceUpdateQorRow { arg_width: 8, update_width: 8, old_and_nodes: 101, old_depth: 8, public_and_nodes: 93, public_depth: 7 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 1, old_and_nodes: 106, old_depth: 6, public_and_nodes: 90, public_depth: 5 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 2, old_and_nodes: 126, old_depth: 7, public_and_nodes: 110, public_depth: 6 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 3, old_and_nodes: 143, old_depth: 8, public_and_nodes: 127, public_depth: 7 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 5, old_and_nodes: 174, old_depth: 9, public_and_nodes: 158, public_depth: 8 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 8, old_and_nodes: 216, old_depth: 10, public_and_nodes: 200, public_depth: 9 },
            BitSliceUpdateQorRow { arg_width: 16, update_width: 16, old_and_nodes: 253, old_depth: 10, public_and_nodes: 237, public_depth: 9 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 1, old_and_nodes: 218, old_depth: 7, public_and_nodes: 186, public_depth: 6 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 2, old_and_nodes: 244, old_depth: 8, public_and_nodes: 212, public_depth: 7 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 3, old_and_nodes: 267, old_depth: 9, public_and_nodes: 235, public_depth: 8 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 5, old_and_nodes: 310, old_depth: 10, public_and_nodes: 278, public_depth: 9 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 8, old_and_nodes: 370, old_depth: 11, public_and_nodes: 338, public_depth: 10 },
            BitSliceUpdateQorRow { arg_width: 32, update_width: 16, old_and_nodes: 506, old_depth: 12, public_and_nodes: 474, public_depth: 11 },
        ];

        assert_eq!(got.as_slice(), want);
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ArraySliceQorRow {
        array_len: usize,
        element_width: usize,
        slice_width: usize,
        old_and_nodes: usize,
        old_depth: usize,
        elem_mux_and_nodes: usize,
        elem_mux_depth: usize,
        public_and_nodes: usize,
        public_depth: usize,
    }

    fn build_array_slice_ir_text(
        array_len: usize,
        element_width: usize,
        start_width: usize,
        slice_width: usize,
    ) -> String {
        let return_width = element_width * slice_width;
        let mut text = format!(
            r#"package sample

top fn main(array: bits[{element_width}][{array_len}], start: bits[{start_width}]) -> bits[{return_width}] {{
  y: bits[{element_width}][{slice_width}] = array_slice(array, start, width={slice_width}, id=3)
"#
        );
        for i in 0..slice_width {
            text.push_str(&format!(
                "  idx{i}: bits[32] = literal(value={i}, id={})\n",
                10 + i
            ));
            text.push_str(&format!(
                "  elem{i}: bits[{element_width}] = array_index(y, indices=[idx{i}], id={})\n",
                20 + i
            ));
        }
        if slice_width == 1 {
            text.push_str(&format!(
                "  ret out: bits[{return_width}] = array_index(y, indices=[idx0], id=100)\n"
            ));
        } else {
            let operands = (0..slice_width)
                .rev()
                .map(|i| format!("elem{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            text.push_str(&format!(
                "  ret out: bits[{return_width}] = concat({operands}, id=100)\n"
            ));
        }
        text.push_str("}\n");
        text
    }

    fn make_array_slice_sample_array(
        array_len: usize,
        element_width: usize,
        elements: &[u64],
    ) -> IrValue {
        assert_eq!(elements.len(), array_len);
        let values = elements
            .iter()
            .map(|value| IrValue::make_ubits(element_width, *value).unwrap())
            .collect::<Vec<_>>();
        IrValue::make_array(&values).unwrap()
    }

    fn array_slice_sample_arrays(array_len: usize, element_width: usize) -> Vec<IrValue> {
        let all_ones = if element_width == 64 {
            u64::MAX
        } else {
            (1u64 << element_width) - 1
        };
        let high_bit = 1u64 << (element_width - 1);
        let mut samples = Vec::new();
        samples.push(make_array_slice_sample_array(
            array_len,
            element_width,
            &vec![0; array_len],
        ));
        samples.push(make_array_slice_sample_array(
            array_len,
            element_width,
            &vec![all_ones; array_len],
        ));

        for index in 0..array_len {
            for value in [1, high_bit, all_ones] {
                let mut elements = vec![0; array_len];
                elements[index] = value;
                samples.push(make_array_slice_sample_array(
                    array_len,
                    element_width,
                    &elements,
                ));
            }
        }

        let alternating = (0..array_len)
            .map(|i| if i % 2 == 0 { 0 } else { all_ones })
            .collect::<Vec<_>>();
        samples.push(make_array_slice_sample_array(
            array_len,
            element_width,
            &alternating,
        ));

        samples
    }

    fn validate_array_slice_public_simulation(
        ir_fn: &ir::Fn,
        gate_fn: &GateFn,
        array_len: usize,
        element_width: usize,
        start_width: usize,
    ) {
        let array_ty = ir::Type::Array(ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        });
        let start_count = 1usize << start_width;
        for array_value in array_slice_sample_arrays(array_len, element_width) {
            let mut array_bits = Vec::new();
            flatten_ir_value_to_lsb0_bits_for_type(&array_value, &array_ty, &mut array_bits)
                .unwrap();
            let gate_array = IrBits::from_lsb_is_0(&array_bits);
            for start in 0..start_count {
                let start_value = IrValue::make_ubits(start_width, start as u64).unwrap();
                let want = match eval_fn(ir_fn, &[array_value.clone(), start_value.clone()]) {
                    FnEvalResult::Success(success) => success.value.to_bits().unwrap(),
                    FnEvalResult::Failure(failure) => {
                        panic!("array_slice source IR failed during simulation: {failure:?}")
                    }
                };
                let gate_start = start_value.to_bits().unwrap();
                let sim = gate_sim::eval(
                    gate_fn,
                    &[gate_array.clone(), gate_start],
                    gate_sim::Collect::None,
                );
                let got = sim.outputs[0].clone();
                assert_eq!(
                    got, want,
                    "array_slice simulation mismatch for array_len={array_len} element_width={element_width} start={start} array={array_value}"
                );
            }
        }
    }

    fn get_array_slice_bit_shift_stats(
        array_len: usize,
        element_width: usize,
        start_width: usize,
        slice_width: usize,
    ) -> AigStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        };
        let mut gb = GateBuilder::new(
            format!("array_slice_bit_shift_n{array_len}_w{element_width}_s{slice_width}"),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("array".to_string(), array_len * element_width);
        let start_bits = gb.add_input("start".to_string(), start_width);
        let result = super::gatify_array_slice_bit_shift(
            &mut gb,
            &array_ty,
            &array_bits,
            &start_bits,
            false,
            slice_width,
            0,
            AdderMapping::BrentKung,
        );
        gb.add_output("result".to_string(), result);
        get_aig_stats(&gb.build())
    }

    fn get_array_slice_element_mux_stats(
        array_len: usize,
        element_width: usize,
        start_width: usize,
        slice_width: usize,
    ) -> AigStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        };
        let mut gb = GateBuilder::new(
            format!("array_slice_elem_mux_n{array_len}_w{element_width}_s{slice_width}"),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("array".to_string(), array_len * element_width);
        let start_bits = gb.add_input("start".to_string(), start_width);
        let result = super::gatify_array_slice_element_mux_if_profitable(
            &mut gb,
            &array_ty,
            &array_bits,
            &start_bits,
            slice_width,
        )
        .expect("array-slice element mux should apply for characterization case");
        gb.add_output("result".to_string(), result);
        get_aig_stats(&gb.build())
    }

    fn get_array_slice_public_stats_and_validate(
        array_len: usize,
        element_width: usize,
        start_width: usize,
        slice_width: usize,
    ) -> AigStats {
        let ir_text = build_array_slice_ir_text(array_len, element_width, start_width, slice_width);
        let mut parser = ir_parser::Parser::new(&ir_text);
        let ir_package = parser.parse_and_validate_package().expect("parse package");
        let ir_fn = ir_package.get_top_fn().expect("top fn");
        let gatify_output = gatify(
            ir_fn,
            GatifyOptions {
                fold: true,
                hash: true,
                check_equivalence: false,
                adder_mapping: AdderMapping::BrentKung,
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .expect("gatify array_slice");
        validate_array_slice_public_simulation(
            ir_fn,
            &gatify_output.gate_fn,
            array_len,
            element_width,
            start_width,
        );
        get_aig_stats(&gatify_output.gate_fn)
    }

    fn gather_array_slice_qor_rows() -> Vec<ArraySliceQorRow> {
        let mut got = Vec::new();
        for array_len in [5usize, 8, 16] {
            let start_width = (array_len - 1).ilog2() as usize + 1;
            for element_width in [1usize, 3, 5] {
                for slice_width in 1usize..=4 {
                    let old = get_array_slice_bit_shift_stats(
                        array_len,
                        element_width,
                        start_width,
                        slice_width,
                    );
                    let elem_mux = get_array_slice_element_mux_stats(
                        array_len,
                        element_width,
                        start_width,
                        slice_width,
                    );
                    let public = get_array_slice_public_stats_and_validate(
                        array_len,
                        element_width,
                        start_width,
                        slice_width,
                    );
                    got.push(ArraySliceQorRow {
                        array_len,
                        element_width,
                        slice_width,
                        old_and_nodes: old.and_nodes,
                        old_depth: old.max_depth,
                        elem_mux_and_nodes: elem_mux.and_nodes,
                        elem_mux_depth: elem_mux.max_depth,
                        public_and_nodes: public.and_nodes,
                        public_depth: public.max_depth,
                    });
                }
            }
        }
        got
    }

    #[test]
    fn test_array_slice_element_mux_qor_and_equivalence_sweep() {
        let _ = env_logger::builder().is_test(true).try_init();

        let got = gather_array_slice_qor_rows();

        for row in &got {
            assert!(
                row.elem_mux_and_nodes < row.old_and_nodes,
                "expected element-mux lowering to reduce AND nodes: {:?}",
                row
            );
            assert!(
                row.elem_mux_depth < row.old_depth,
                "expected element-mux lowering to reduce depth: {:?}",
                row
            );
            assert_eq!(
                (row.public_and_nodes, row.public_depth),
                (row.elem_mux_and_nodes, row.elem_mux_depth),
                "expected public array_slice lowering to use element-mux strategy: {:?}",
                row
            );
        }

        #[rustfmt::skip]
        let want: &[ArraySliceQorRow] = &[
            ArraySliceQorRow { array_len: 5, element_width: 1, slice_width: 1, old_and_nodes: 21, old_depth: 10, elem_mux_and_nodes: 18, elem_mux_depth: 6, public_and_nodes: 18, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 1, slice_width: 2, old_and_nodes: 35, old_depth: 10, elem_mux_and_nodes: 28, elem_mux_depth: 6, public_and_nodes: 28, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 1, slice_width: 3, old_and_nodes: 43, old_depth: 10, elem_mux_and_nodes: 32, elem_mux_depth: 6, public_and_nodes: 32, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 1, slice_width: 4, old_and_nodes: 49, old_depth: 10, elem_mux_and_nodes: 36, elem_mux_depth: 6, public_and_nodes: 36, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 3, slice_width: 1, old_and_nodes: 135, old_depth: 14, elem_mux_and_nodes: 54, elem_mux_depth: 6, public_and_nodes: 54, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 3, slice_width: 2, old_and_nodes: 188, old_depth: 15, elem_mux_and_nodes: 84, elem_mux_depth: 6, public_and_nodes: 84, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 3, slice_width: 3, old_and_nodes: 220, old_depth: 15, elem_mux_and_nodes: 96, elem_mux_depth: 6, public_and_nodes: 96, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 3, slice_width: 4, old_and_nodes: 240, old_depth: 15, elem_mux_and_nodes: 108, elem_mux_depth: 6, public_and_nodes: 108, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 5, slice_width: 1, old_and_nodes: 251, old_depth: 15, elem_mux_and_nodes: 90, elem_mux_depth: 6, public_and_nodes: 90, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 5, slice_width: 2, old_and_nodes: 339, old_depth: 15, elem_mux_and_nodes: 140, elem_mux_depth: 6, public_and_nodes: 140, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 5, slice_width: 3, old_and_nodes: 397, old_depth: 16, elem_mux_and_nodes: 160, elem_mux_depth: 6, public_and_nodes: 160, public_depth: 6 },
            ArraySliceQorRow { array_len: 5, element_width: 5, slice_width: 4, old_and_nodes: 441, old_depth: 16, elem_mux_and_nodes: 180, elem_mux_depth: 6, public_and_nodes: 180, public_depth: 6 },

            ArraySliceQorRow { array_len: 8, element_width: 1, slice_width: 1, old_and_nodes: 34, old_depth: 12, elem_mux_and_nodes: 21, elem_mux_depth: 6, public_and_nodes: 21, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 1, slice_width: 2, old_and_nodes: 54, old_depth: 12, elem_mux_and_nodes: 41, elem_mux_depth: 6, public_and_nodes: 41, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 1, slice_width: 3, old_and_nodes: 62, old_depth: 12, elem_mux_and_nodes: 49, elem_mux_depth: 6, public_and_nodes: 49, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 1, slice_width: 4, old_and_nodes: 70, old_depth: 12, elem_mux_and_nodes: 57, elem_mux_depth: 6, public_and_nodes: 57, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 3, slice_width: 1, old_and_nodes: 206, old_depth: 16, elem_mux_and_nodes: 63, elem_mux_depth: 6, public_and_nodes: 63, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 3, slice_width: 2, old_and_nodes: 285, old_depth: 16, elem_mux_and_nodes: 123, elem_mux_depth: 6, public_and_nodes: 123, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 3, slice_width: 3, old_and_nodes: 331, old_depth: 16, elem_mux_and_nodes: 147, elem_mux_depth: 6, public_and_nodes: 147, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 3, slice_width: 4, old_and_nodes: 357, old_depth: 16, elem_mux_and_nodes: 171, elem_mux_depth: 6, public_and_nodes: 171, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 5, slice_width: 1, old_and_nodes: 402, old_depth: 18, elem_mux_and_nodes: 105, elem_mux_depth: 6, public_and_nodes: 105, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 5, slice_width: 2, old_and_nodes: 538, old_depth: 18, elem_mux_and_nodes: 205, elem_mux_depth: 6, public_and_nodes: 205, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 5, slice_width: 3, old_and_nodes: 620, old_depth: 18, elem_mux_and_nodes: 245, elem_mux_depth: 6, public_and_nodes: 245, public_depth: 6 },
            ArraySliceQorRow { array_len: 8, element_width: 5, slice_width: 4, old_and_nodes: 677, old_depth: 18, elem_mux_and_nodes: 285, elem_mux_depth: 6, public_and_nodes: 285, public_depth: 6 },

            ArraySliceQorRow { array_len: 16, element_width: 1, slice_width: 1, old_and_nodes: 63, old_depth: 15, elem_mux_and_nodes: 45, elem_mux_depth: 8, public_and_nodes: 45, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 1, slice_width: 2, old_and_nodes: 107, old_depth: 15, elem_mux_and_nodes: 89, elem_mux_depth: 8, public_and_nodes: 89, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 1, slice_width: 3, old_and_nodes: 127, old_depth: 15, elem_mux_and_nodes: 109, elem_mux_depth: 8, public_and_nodes: 109, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 1, slice_width: 4, old_and_nodes: 147, old_depth: 15, elem_mux_and_nodes: 129, elem_mux_depth: 8, public_and_nodes: 129, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 3, slice_width: 1, old_and_nodes: 406, old_depth: 20, elem_mux_and_nodes: 135, elem_mux_depth: 8, public_and_nodes: 135, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 3, slice_width: 2, old_and_nodes: 560, old_depth: 20, elem_mux_and_nodes: 267, elem_mux_depth: 8, public_and_nodes: 267, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 3, slice_width: 3, old_and_nodes: 654, old_depth: 20, elem_mux_and_nodes: 327, elem_mux_depth: 8, public_and_nodes: 327, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 3, slice_width: 4, old_and_nodes: 712, old_depth: 20, elem_mux_and_nodes: 387, elem_mux_depth: 8, public_and_nodes: 387, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 5, slice_width: 1, old_and_nodes: 808, old_depth: 21, elem_mux_and_nodes: 225, elem_mux_depth: 8, public_and_nodes: 225, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 5, slice_width: 2, old_and_nodes: 1069, old_depth: 21, elem_mux_and_nodes: 445, elem_mux_depth: 8, public_and_nodes: 445, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 5, slice_width: 3, old_and_nodes: 1229, old_depth: 21, elem_mux_and_nodes: 545, elem_mux_depth: 8, public_and_nodes: 545, public_depth: 8 },
            ArraySliceQorRow { array_len: 16, element_width: 5, slice_width: 4, old_and_nodes: 1329, old_depth: 21, elem_mux_and_nodes: 645, elem_mux_depth: 8, public_and_nodes: 645, public_depth: 8 },
        ];

        assert_eq!(got.as_slice(), want);
    }

    #[test]
    fn test_gatify_array_slice_narrow_start_regression() {
        let ir_text = "package sample
top fn f(start: bits[2], a: bits[8]) -> bits[8][1] {
  array.4: bits[8][8] = array(a, a, a, a, a, a, a, a, id=4)
  ret array_slice.5: bits[8][1] = array_slice(array.4, start, width=1, id=5)
}
";
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();

        // Regression: this used to panic when clamping an out-of-bounds start index
        // because last_idx (7) was forced into start width (2 bits).
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
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .expect("gatify array_slice with narrow start should not panic");

        // bits[8][1] flatten to an 8-bit gate output.
        assert_eq!(
            gatify_output.gate_fn.outputs[0].bit_vector.get_bit_count(),
            8
        );

        // Optional end-to-end equivalence check against the source IR.
        // This test focuses on clamp behavior in the gate-level lowering itself.
        // End-to-end IR equivalence for this wider-start case is covered elsewhere.
        let _ = ir_fn;
    }

    #[test]
    fn test_gatify_array_slice_wide_start_clamps_oob_to_last_element() {
        let ir_text = "package sample
top fn f(start: bits[4], a: bits[8], b: bits[8]) -> bits[8][1] {
  array.4: bits[8][8] = array(a, a, a, a, a, a, b, b, id=4)
  ret array_slice.5: bits[8][1] = array_slice(array.4, start, width=1, id=5)
}
";
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();

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
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .expect("gatify array_slice with wide start should succeed");

        let eval = |start: u64, a: u64, b: u64| {
            let inputs = vec![
                xlsynth::IrBits::make_ubits(4, start).unwrap(),
                xlsynth::IrBits::make_ubits(8, a).unwrap(),
                xlsynth::IrBits::make_ubits(8, b).unwrap(),
            ];
            gate_sim::eval(&gatify_output.gate_fn, &inputs, gate_sim::Collect::None).outputs[0]
                .clone()
        };

        let a = 0x12;
        let b = 0x34;
        let at_7 = eval(7, a, b);
        assert_eq!(eval(8, a, b), at_7);
        assert_eq!(eval(15, a, b), at_7);

        let _ = ir_fn;
    }

    fn gatify_ir_text(ir_text: &str) -> GateFn {
        let mut parser = ir_parser::Parser::new(ir_text);
        let ir_package = parser.parse_and_validate_package().unwrap();
        let ir_fn = ir_package.get_top_fn().unwrap();
        gatify(
            &ir_fn,
            GatifyOptions {
                fold: true,
                hash: true,
                check_equivalence: false,
                adder_mapping: AdderMapping::default(),
                mul_adder_mapping: None,
                range_info: None,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_nary_add: false,
                array_index_lowering_strategy: Default::default(),
            },
        )
        .unwrap()
        .gate_fn
    }

    fn gate_eval_1bit(gate_fn: &GateFn, lhs: u64, lhs_width: usize) -> bool {
        let inputs = vec![xlsynth::IrBits::make_ubits(lhs_width, lhs).unwrap()];
        gate_sim::eval(gate_fn, &inputs, gate_sim::Collect::None).outputs[0]
            .get_bit(0)
            .unwrap()
    }

    fn bits_as_signed(value: u64, width: usize) -> i64 {
        assert!(width > 0 && width < 63);
        let sign_bit = 1u64 << (width - 1);
        if (value & sign_bit) == 0 {
            value as i64
        } else {
            (value as i64) - (1i64 << width)
        }
    }

    fn expected_signed_cmp(binop: ir::Binop, lhs: u64, rhs: u64, width: usize) -> bool {
        let lhs_signed = bits_as_signed(lhs, width);
        let rhs_signed = bits_as_signed(rhs, width);
        match binop {
            ir::Binop::Slt => lhs_signed < rhs_signed,
            ir::Binop::Sle => lhs_signed <= rhs_signed,
            ir::Binop::Sgt => lhs_signed > rhs_signed,
            ir::Binop::Sge => lhs_signed >= rhs_signed,
            _ => panic!(
                "unexpected binop for signed-compare proof test: {:?}",
                binop
            ),
        }
    }

    #[test]
    fn test_signed_literal_cmp_proof_matrix_rhs_and_lhs_literal() {
        let width = 5usize;
        let values = [9u64, 26u64];
        let cases: &[(ir::Binop, &str)] = &[
            (ir::Binop::Slt, "slt"),
            (ir::Binop::Sle, "sle"),
            (ir::Binop::Sgt, "sgt"),
            (ir::Binop::Sge, "sge"),
        ];

        for &(binop, op_name) in cases {
            for &rhs in &values {
                for literal_on_lhs in [false, true] {
                    let expr = if literal_on_lhs {
                        format!("{op_name}(k, x, id=11)")
                    } else {
                        format!("{op_name}(x, k, id=11)")
                    };
                    let ir_text = format!(
                        r#"package sample
top fn f(x: bits[{width}]) -> bits[1] {{
  k: bits[{width}] = literal(value={rhs}, id=10)
  ret out: bits[1] = {expr}
}}
"#
                    );
                    let gate_fn = gatify_ir_text(&ir_text);
                    for lhs in 0u64..(1u64 << width) {
                        let got = gate_eval_1bit(&gate_fn, lhs, width);
                        let want = if literal_on_lhs {
                            expected_signed_cmp(binop, rhs, lhs, width)
                        } else {
                            expected_signed_cmp(binop, lhs, rhs, width)
                        };
                        assert_eq!(
                            got, want,
                            "signed literal compare mismatch: op={} rhs={} literal_on_lhs={} lhs={}",
                            op_name, rhs, literal_on_lhs, lhs
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_signed_literal_cmp_cone_quality_matches_manual_decomposition() {
        let cone_ir = r#"package sample
top fn cone(leaf_7: bits[5], leaf_9: bits[33]) -> bits[1] {
  sign_ext.3431: bits[33] = sign_ext(leaf_7, new_bit_count=33, id=3431)
  add.3432: bits[33] = add(leaf_9, sign_ext.3431, id=3432)
  literal.3433: bits[33] = literal(value=8589934578, id=3433)
  ret slt.3434: bits[1] = slt(add.3432, literal.3433, id=3434)
}
"#;
        let decomp_ir = r#"package sample
top fn cone(leaf_7: bits[5], leaf_9: bits[33]) -> bits[1] {
  sign_ext.3431: bits[33] = sign_ext(leaf_7, new_bit_count=33, id=3431)
  add.3432: bits[33] = add(leaf_9, sign_ext.3431, id=3432)
  literal.3433: bits[33] = literal(value=8589934578, id=3433)
  bit_slice.3435: bits[1] = bit_slice(add.3432, start=32, width=1, id=3435)
  ult.3436: bits[1] = ult(add.3432, literal.3433, id=3436)
  ret and.3437: bits[1] = and(bit_slice.3435, ult.3436, id=3437)
}
"#;

        let cone_gate_fn = gatify_ir_text(cone_ir);
        let decomp_gate_fn = gatify_ir_text(decomp_ir);
        let cone_stats = get_summary_stats(&cone_gate_fn);
        let decomp_stats = get_summary_stats(&decomp_gate_fn);

        assert_eq!(
            cone_stats.live_nodes, decomp_stats.live_nodes,
            "literal signed-compare lowering should match manual decomposition node count"
        );
        assert_eq!(
            cone_stats.deepest_path, decomp_stats.deepest_path,
            "literal signed-compare lowering should match manual decomposition depth"
        );

        // Characterization guard for the original regression report:
        // before this lowering path, this cone was observed at roughly
        // 562 nodes / 33 levels in g8r output.
        assert!(
            cone_stats.live_nodes <= 434,
            "expected improved node count for regression cone; got {}",
            cone_stats.live_nodes
        );
        assert!(
            cone_stats.deepest_path <= 28,
            "expected improved depth for regression cone; got {}",
            cone_stats.deepest_path
        );
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

    fn get_tuple_field0_array_index_stats(
        array_len: usize,
        payload_width: usize,
        index_width: usize,
        strategy: super::ArrayIndexLoweringStrategy,
    ) -> SummaryStats {
        let tuple_ty = ir::Type::Tuple(vec![
            Box::new(ir::Type::Bits(1)),
            Box::new(ir::Type::Bits(payload_width)),
        ]);
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(tuple_ty.clone()),
            element_count: array_len,
        };
        let field0 = tuple_ty.tuple_get_flat_bit_slice_for_index(0).unwrap();

        let mut gb = GateBuilder::new(
            format!("tuple_field0_w{}_n{}", payload_width, array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), tuple_ty.bit_count() * array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let selected = super::gatify_array_index(
            &mut gb,
            &array_ty,
            &array_bits,
            &index_bits,
            false,
            strategy,
        );
        let result = selected.get_lsb_slice(field0.start, field0.limit - field0.start);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_1b_array_index_oob_one_hot_stats(array_len: usize, index_width: usize) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(1)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_1b_n{}_oob", array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result =
            super::gatify_array_index_oob_one_hot(&mut gb, &array_ty, &array_bits, &index_bits);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_1b_array_index_exact_stats(array_len: usize, index_width: usize) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(1)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_1b_n{}_exact", array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result = super::gatify_array_index_exact(&mut gb, &array_ty, &array_bits, &index_bits);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_1b_array_index_near_pow2_padded_stats(
        array_len: usize,
        index_width: usize,
    ) -> SummaryStats {
        const MAX_PADDED_COUNT: usize = 32;
        const MAX_EXTRA_ELEMS: usize = 8;
        let padded_count = array_len.next_power_of_two();
        assert!(
            padded_count > array_len
                && padded_count <= MAX_PADDED_COUNT
                && padded_count - array_len <= MAX_EXTRA_ELEMS
        );

        let mut gb = GateBuilder::new(
            format!("array_index_1b_n{}_padpow2", array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let padded_last_bits = gb.add_literal(
            &xlsynth::IrBits::make_ubits(index_width, (padded_count - 1) as u64).unwrap(),
        );
        let idx_le_padded_last =
            super::gatify_ule_via_bit_tests(&mut gb, 0, &index_bits, &padded_last_bits);
        let clamped_index = gb.add_mux2_vec(&idx_le_padded_last, &index_bits, &padded_last_bits);
        let last_elem = array_bits.get_lsb_slice(array_len - 1, 1);
        let mut padded_array_bits = array_bits.clone();
        for _ in array_len..padded_count {
            padded_array_bits = AigBitVector::concat(last_elem.clone(), padded_array_bits);
        }
        let padded_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(1)),
            element_count: padded_count,
        };
        let result = super::gatify_array_index_exact(
            &mut gb,
            &padded_ty,
            &padded_array_bits,
            &clamped_index,
        );
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_1b_array_index_public_stats(array_len: usize, index_width: usize) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(1)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_1b_n{}_public", array_len),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result = super::gatify_array_index(
            &mut gb,
            &array_ty,
            &array_bits,
            &index_bits,
            false,
            super::ArrayIndexLoweringStrategy::Auto,
        );
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_array_index_mux_tree_pad_last_stats(
        array_len: usize,
        element_width: usize,
        index_width: usize,
    ) -> SummaryStats {
        let mut gb = GateBuilder::new(
            format!("array_index_mux_tree_n{}_w{}", array_len, element_width),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), element_width * array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let mut cases = Vec::with_capacity(array_len);
        for i in 0..array_len {
            let case_bits = array_bits.get_lsb_slice(i * element_width, element_width);
            cases.push(case_bits);
        }
        let result = crate::ir2gate_utils::gatify_indexed_select_mux_tree_pad_last_if_type_fits(
            &mut gb,
            &index_bits,
            &cases,
        )
        .unwrap_or_else(|e| {
            panic!(
                "pad-last mux-tree helper should apply for array_len={} element_width={} index_width={}: {}",
                array_len, element_width, index_width, e
            )
        });
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_array_index_oob_one_hot_stats(
        array_len: usize,
        element_width: usize,
        index_width: usize,
    ) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_ohs_n{}_w{}", array_len, element_width),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), element_width * array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result =
            super::gatify_array_index_oob_one_hot(&mut gb, &array_ty, &array_bits, &index_bits);
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    fn get_array_index_public_stats(
        array_len: usize,
        element_width: usize,
        index_width: usize,
    ) -> SummaryStats {
        let array_ty = ir::ArrayTypeData {
            element_type: Box::new(ir::Type::Bits(element_width)),
            element_count: array_len,
        };

        let mut gb = GateBuilder::new(
            format!("array_index_public_n{}_w{}", array_len, element_width),
            GateBuilderOptions::opt(),
        );
        let array_bits = gb.add_input("arr".to_string(), element_width * array_len);
        let index_bits = gb.add_input("idx".to_string(), index_width);
        let result = super::gatify_array_index(
            &mut gb,
            &array_ty,
            &array_bits,
            &index_bits,
            false,
            super::ArrayIndexLoweringStrategy::Auto,
        );
        gb.add_output("result".to_string(), result);
        get_summary_stats(&gb.build())
    }

    #[test]
    fn test_tuple_index_array_index_field0_dce_stays_flat_across_dead_payload_width() {
        #[rustfmt::skip]
        const WANT: &[(usize, usize, usize)] = &[
            (8, 155, 15),
            (32, 155, 15),
            (128, 155, 15),
        ];

        for &(payload_width, want_live_nodes, want_deepest_path) in WANT {
            let got = get_tuple_field0_array_index_stats(
                /* array_len= */ 27,
                payload_width,
                /* index_width= */ 5,
                super::ArrayIndexLoweringStrategy::ForceOobOneHot,
            );
            assert_eq!(
                got.live_nodes, want_live_nodes,
                "tuple field0 array-index live_nodes mismatch for payload_width={}",
                payload_width
            );
            assert_eq!(
                got.deepest_path, want_deepest_path,
                "tuple field0 array-index depth mismatch for payload_width={}",
                payload_width
            );
        }
    }

    #[test]
    fn test_tuple_index_array_index_field0_width2_tuple_hits_public_mux_tree_strategy() {
        let got = get_tuple_field0_array_index_stats(
            /* array_len= */ 27,
            /* payload_width= */ 1,
            5,
            super::ArrayIndexLoweringStrategy::Auto,
        );
        assert_eq!(got.live_nodes, 118);
        assert_eq!(got.deepest_path, 11);
    }

    #[test]
    fn test_1b_array_index_near_pow2_padding_characterization() {
        let oob_27 = get_1b_array_index_oob_one_hot_stats(
            /* array_len= */ 27, /* index_width= */ 5,
        );
        let padded_27 = get_1b_array_index_near_pow2_padded_stats(
            /* array_len= */ 27, /* index_width= */ 5,
        );
        let public_27 =
            get_1b_array_index_public_stats(/* array_len= */ 27, /* index_width= */ 5);
        let exact_32 =
            get_1b_array_index_exact_stats(/* array_len= */ 32, /* index_width= */ 5);

        assert_eq!(oob_27.live_nodes, 155);
        assert_eq!(oob_27.deepest_path, 15);
        assert_eq!(padded_27.live_nodes, 166);
        assert_eq!(padded_27.deepest_path, 17);
        assert_eq!(public_27.live_nodes, 118);
        assert_eq!(public_27.deepest_path, 11);
        assert_eq!(exact_32.live_nodes, 148);
        assert_eq!(exact_32.deepest_path, 10);

        assert!(
            padded_27.live_nodes > oob_27.live_nodes,
            "expected naive padded near-pow2 clamping to increase live_nodes in g8r: {:?} vs {:?}",
            padded_27,
            oob_27
        );
        assert!(
            padded_27.deepest_path > oob_27.deepest_path,
            "expected naive padded near-pow2 clamping to increase depth in g8r: {:?} vs {:?}",
            padded_27,
            oob_27
        );
    }

    #[test]
    fn test_array_index_near_pow2_mux_tree_vs_one_hot_sweep_characterization() {
        let widths = [1usize, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32];
        for element_width in widths {
            let ohs = get_array_index_oob_one_hot_stats(
                /* array_len= */ 27,
                element_width,
                /* index_width= */ 5,
            );
            let mux = get_array_index_mux_tree_pad_last_stats(
                /* array_len= */ 27,
                element_width,
                /* index_width= */ 5,
            );
            let ohs_product = ohs.live_nodes * ohs.deepest_path;
            let mux_product = mux.live_nodes * mux.deepest_path;
            assert!(
                mux.deepest_path < ohs.deepest_path,
                "expected mux-tree to reduce depth for element_width={}: ohs={:?} mux={:?}",
                element_width,
                ohs,
                mux
            );
            assert!(
                mux_product < ohs_product,
                "expected mux-tree to reduce nodes*depth for element_width={}: ohs_product={} mux_product={}",
                element_width,
                ohs_product,
                mux_product
            );
            if element_width <= 2 {
                assert!(
                    mux.live_nodes < ohs.live_nodes,
                    "expected mux-tree to reduce live_nodes for element_width={}: ohs={:?} mux={:?}",
                    element_width,
                    ohs,
                    mux
                );
            } else {
                assert!(
                    mux.live_nodes > ohs.live_nodes,
                    "expected mux-tree to trade nodes for depth beyond width-2 crossover for element_width={}: ohs={:?} mux={:?}",
                    element_width,
                    ohs,
                    mux
                );
            }
        }
    }

    #[test]
    fn test_array_index_public_strategy_uses_mux_tree_only_in_measured_profitable_region() {
        for element_width in [1usize, 2] {
            let public = get_array_index_public_stats(/* array_len= */ 27, element_width, 5);
            let mux =
                get_array_index_mux_tree_pad_last_stats(/* array_len= */ 27, element_width, 5);
            assert_eq!(
                public, mux,
                "expected public strategy to use near-pow2 mux-tree for element_width={}",
                element_width
            );
        }

        for element_width in [3usize, 4, 8] {
            let public = get_array_index_public_stats(/* array_len= */ 27, element_width, 5);
            let ohs = get_array_index_oob_one_hot_stats(/* array_len= */ 27, element_width, 5);
            assert_eq!(
                public, ohs,
                "expected public strategy to keep OHS lowering for element_width={}",
                element_width
            );
        }
    }
}
