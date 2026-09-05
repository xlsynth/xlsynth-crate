// SPDX-License-Identifier: Apache-2.0

//! Arithmetic lowering helpers for `ir2gate`.

use crate::aig::gate::{AigBitVector, AigOperand};
use crate::gate_builder::GateBuilder;
use crate::gatify::ir2gate::{
    GateEnv, gatify_add_with_mapping, gatify_sext_or_truncate, gatify_zext_or_truncate,
    get_pow2_minus1_k, literal_bits_if_bits_node,
};
use crate::ir2gate_utils::{AdderMapping, array_add_with_carry_out};
use std::ops::Range;
use xlsynth_pir::ir;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtNaryAddFillKind {
    Zero,
    Sign,
}

#[derive(Clone, Debug)]
struct ExtNaryAddTermDimensions {
    term_bits: AigBitVector,
    weight_shift: usize,
    active_range: Range<usize>,
    fill_kind: ExtNaryAddFillKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExtNaryAddUnitCorrection {
    control: AigOperand,
    weight_shift: usize,
    is_decrement: bool,
}

#[derive(Clone, Debug)]
struct SelectedUnitDeltaMatch {
    base_bits: AigBitVector,
    changed_bits: AigBitVector,
    is_decrement: bool,
    invert_selector: bool,
    adder_mapping: AdderMapping,
}

#[derive(Clone, Debug)]
struct SelectedNegateMatch {
    base_bits: AigBitVector,
    selected_bits: AigBitVector,
    control: AigOperand,
    invert_control: bool,
}

/// Returns the minimum selector-arrival margin for a selected unit update.
fn selected_unit_delta_arrival_margin(output_width: usize) -> usize {
    2.max(xlsynth_pir::math::ceil_log2(output_width))
}

/// Returns the maximum cached AIG depth of a bit vector.
fn max_aig_depth(gb: &GateBuilder, bits: &AigBitVector) -> Option<usize> {
    bits.iter_lsb_to_msb().map(|bit| gb.aig_depth(*bit)).max()?
}

/// Matches a safely fusable `sel(p, [x, -x])` arithmetic term.
fn match_selected_negate_term(
    f: &ir::Fn,
    env: &GateEnv,
    term: &ir::ExtNaryAddTerm,
    output_width: usize,
    false_bit: AigOperand,
) -> Option<SelectedNegateMatch> {
    if !env.has_single_use(term.operand) {
        return None;
    }
    let ir::NodePayload::Sel {
        selector,
        cases,
        default: None,
    } = &f.get_node(term.operand).payload
    else {
        return None;
    };
    if cases.len() != 2 {
        return None;
    }
    let selector_bits = env.get_bit_vector(*selector).ok()?;
    if selector_bits.get_bit_count() != 1 {
        return None;
    }

    for (neg_index, base_index) in [(1usize, 0usize), (0, 1)] {
        let neg_node = cases[neg_index];
        if !env.has_single_use(neg_node) {
            continue;
        }
        let ir::NodePayload::Unop(ir::Unop::Neg, neg_arg) = f.get_node(neg_node).payload else {
            continue;
        };
        if neg_arg != cases[base_index] {
            continue;
        }
        let base_bits = env.get_bit_vector(cases[base_index]).ok()?;
        let base_width = base_bits.get_bit_count();
        if base_width == 0
            || base_width > output_width
            || (base_width < output_width && (!term.signed || *base_bits.get_msb(0) != false_bit))
        {
            continue;
        }
        let selected_bits = env.get_bit_vector(term.operand).ok()?;
        if selected_bits.get_bit_count() != base_width {
            continue;
        }
        return Some(SelectedNegateMatch {
            base_bits,
            selected_bits,
            control: *selector_bits.get_lsb(0),
            invert_control: neg_index == 0,
        });
    }
    None
}

/// Resizes a literal term to the `ExtNaryAdd` output width.
fn resize_literal_bits_for_ext_nary_add(
    bits: &xlsynth_pir::IrBits,
    output_width: usize,
    signed: bool,
) -> xlsynth_pir::IrBits {
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
            xlsynth_pir::IrBits::from_lsb_is_0(&resized_bits)
        }
        std::cmp::Ordering::Equal => bits.clone(),
        std::cmp::Ordering::Greater => bits.width_slice(0, output_width as i64),
    }
}

/// Adds one literal contribution into the `ExtNaryAdd` constant accumulator.
fn accumulate_ext_nary_add_literal(
    literal_sum: &mut xlsynth_pir::IrBits,
    term_bits: &xlsynth_pir::IrBits,
    output_width: usize,
    signed: bool,
    negated: bool,
) {
    let resized = resize_literal_bits_for_ext_nary_add(term_bits, output_width, signed);
    let contribution = if negated { resized.negate() } else { resized };
    *literal_sum = literal_sum.add(&contribution);
}

/// Returns whether `bits` is exactly the unsigned value 1.
fn is_one(bits: &xlsynth_pir::IrBits) -> bool {
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

/// Returns whether two lowered bit vectors contain the same AIG operands.
fn aig_bits_equal(lhs: &AigBitVector, rhs: &AigBitVector) -> bool {
    lhs.get_bit_count() == rhs.get_bit_count()
        && lhs
            .iter_lsb_to_msb()
            .zip(rhs.iter_lsb_to_msb())
            .all(|(lhs_bit, rhs_bit)| lhs_bit == rhs_bit)
}

/// Resizes a dynamic arithmetic term without creating or tagging AIG nodes.
fn resize_dynamic_term_for_match(
    bits: &AigBitVector,
    output_width: usize,
    signed: bool,
) -> AigBitVector {
    match bits.get_bit_count().cmp(&output_width) {
        std::cmp::Ordering::Less if signed && bits.get_bit_count() != 0 => {
            let sign = *bits.get_msb(0);
            let extension =
                AigBitVector::from_lsb_is_index_0(&vec![sign; output_width - bits.get_bit_count()]);
            AigBitVector::concat(extension, bits.clone())
        }
        std::cmp::Ordering::Less => gatify_zext_or_truncate(output_width, bits),
        std::cmp::Ordering::Equal => bits.clone(),
        std::cmp::Ordering::Greater => bits.get_lsb_slice(0, output_width),
    }
}

/// Matches the prepared or unprepared spelling of a width-preserving unit
/// delta.
fn match_unit_delta_from_base_bits(
    f: &ir::Fn,
    env: &GateEnv,
    changed: ir::NodeRef,
    base_bits: &AigBitVector,
    output_width: usize,
) -> Option<bool> {
    let (dynamic_operand, dynamic_signed, dynamic_negated, literal_contribution) =
        match &f.get_node(changed).payload {
            ir::NodePayload::Binop(ir::Binop::Add, lhs, rhs) => {
                let (dynamic_operand, literal_bits) = normalize_add_literal_rhs(f, *lhs, *rhs)?;
                (dynamic_operand, false, false, literal_bits)
            }
            ir::NodePayload::Binop(ir::Binop::Sub, lhs, rhs) => {
                let literal_bits = literal_bits_if_bits_node(f, *rhs)?;
                (*lhs, false, false, literal_bits.negate())
            }
            ir::NodePayload::ExtNaryAdd { terms, .. } if terms.len() == 2 => {
                let mut dynamic_term = None;
                let mut literal_sum = xlsynth_pir::IrBits::zero(output_width);
                for term in terms {
                    if let Some(literal_bits) = literal_bits_if_bits_node(f, term.operand) {
                        accumulate_ext_nary_add_literal(
                            &mut literal_sum,
                            &literal_bits,
                            output_width,
                            term.signed,
                            term.negated,
                        );
                    } else if dynamic_term.replace(*term).is_some() {
                        return None;
                    }
                }
                let dynamic_term = dynamic_term?;
                (
                    dynamic_term.operand,
                    dynamic_term.signed,
                    dynamic_term.negated,
                    literal_sum,
                )
            }
            _ => return None,
        };
    if dynamic_negated {
        return None;
    }

    let dynamic_bits = env.get_bit_vector(dynamic_operand).ok()?;
    let resized_dynamic_bits =
        resize_dynamic_term_for_match(&dynamic_bits, output_width, dynamic_signed);
    if !aig_bits_equal(&resized_dynamic_bits, base_bits) {
        return None;
    }

    if literal_contribution.equals_u64_value(1) {
        Some(false)
    } else if literal_contribution.negate().equals_u64_value(1) {
        Some(true)
    } else {
        None
    }
}

/// Finds `sel(p, cases=[x, x +/- 1])`, accounting for prep-induced resize
/// aliases.
fn match_selected_unit_delta(
    f: &ir::Fn,
    env: &GateEnv,
    cases: &[ir::NodeRef],
    output_width: usize,
    default_adder_mapping: AdderMapping,
) -> Option<SelectedUnitDeltaMatch> {
    if cases.len() != 2 || output_width == 0 {
        return None;
    }
    for (changed_index, base_index) in [(1usize, 0usize), (0, 1)] {
        let changed = cases[changed_index];
        if !env.has_single_use(changed) {
            continue;
        }
        if !matches!(
            &f.get_node(changed).payload,
            ir::NodePayload::Binop(ir::Binop::Add | ir::Binop::Sub, _, _)
                | ir::NodePayload::ExtNaryAdd { .. }
        ) {
            continue;
        }
        let base_bits = env.get_bit_vector(cases[base_index]).ok()?;
        if base_bits.get_bit_count() != output_width {
            continue;
        }
        let Some(is_decrement) =
            match_unit_delta_from_base_bits(f, env, changed, &base_bits, output_width)
        else {
            continue;
        };
        let changed_bits = env.get_bit_vector(changed).ok()?;
        if changed_bits.get_bit_count() != output_width {
            continue;
        }
        let adder_mapping = match &f.get_node(changed).payload {
            ir::NodePayload::ExtNaryAdd {
                arch: Some(arch), ..
            } => AdderMapping::from(*arch),
            _ => default_adder_mapping,
        };
        return Some(SelectedUnitDeltaMatch {
            base_bits,
            changed_bits,
            is_decrement,
            invert_selector: changed_index == 0,
            adder_mapping,
        });
    }
    None
}

/// Lowers an early selected unit update by injecting the predicate as carry-in.
pub(super) fn try_gatify_selected_unit_delta(
    f: &ir::Fn,
    env: &GateEnv,
    selector: ir::NodeRef,
    cases: &[ir::NodeRef],
    output_width: usize,
    adder_mapping: AdderMapping,
    gb: &mut GateBuilder,
) -> Option<AigBitVector> {
    let selector_bits = env.get_bit_vector(selector).ok()?;
    if selector_bits.get_bit_count() != 1 {
        return None;
    }
    let matched = match_selected_unit_delta(f, env, cases, output_width, adder_mapping)?;
    if matches!(matched.adder_mapping, AdderMapping::RippleCarry) {
        log::debug!(
            "selected-unit lowering: selector={} width={} adder={} choice=mux",
            f.get_node(selector).text_id,
            output_width,
            matched.adder_mapping
        );
        return None;
    }
    let selector_bit = *selector_bits.get_lsb(0);
    let selector_depth = gb.aig_depth(selector_bit)?;
    let changed_depth = matched
        .changed_bits
        .iter_lsb_to_msb()
        .map(|bit| gb.aig_depth(*bit))
        .max()??;

    // The carry-in must be early enough to propagate through the selected
    // update; require at least the prefix depth or the two-level mux cost.
    let margin = selected_unit_delta_arrival_margin(output_width);
    if selector_depth.saturating_add(margin) >= changed_depth {
        log::debug!(
            "selected-unit lowering: selector={} width={} selector_depth={} changed_depth={} margin={} choice=mux",
            f.get_node(selector).text_id,
            output_width,
            selector_depth,
            changed_depth,
            margin
        );
        return None;
    }
    log::debug!(
        "selected-unit lowering: selector={} width={} selector_depth={} changed_depth={} margin={} choice=carry-in",
        f.get_node(selector).text_id,
        output_width,
        selector_depth,
        changed_depth,
        margin
    );

    let control = if matched.invert_selector {
        gb.add_not(selector_bit)
    } else {
        selector_bit
    };
    let base_bits = if matched.is_decrement {
        gb.add_not_vec(&matched.base_bits)
    } else {
        matched.base_bits
    };
    let zeros = AigBitVector::zeros(output_width);
    let (_, sum) =
        gatify_add_with_mapping(matched.adder_mapping, &base_bits, &zeros, control, None, gb);
    if matched.is_decrement {
        Some(gb.add_not_vec(&sum))
    } else {
        Some(sum)
    }
}

fn normalize_add_literal_rhs(
    f: &ir::Fn,
    a: ir::NodeRef,
    b: ir::NodeRef,
) -> Option<(ir::NodeRef, xlsynth_pir::IrBits)> {
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
    assert!(k > 0 && k <= lhs_bits.get_bit_count());
    gatify_add_const_single_ones_run(gb, lhs_bits, 0..k, gb.get_false())
}

/// Lowers `lhs_bits + (1 << one_bit_position)` through `adder_mapping`.
fn gatify_add_const_pow2_with_mapping(
    gb: &mut GateBuilder,
    lhs_bits: &AigBitVector,
    one_bit_position: usize,
    adder_mapping: AdderMapping,
) -> AigBitVector {
    let bit_count = lhs_bits.get_bit_count();
    assert!(one_bit_position < bit_count);

    let mut sum_bits = Vec::with_capacity(bit_count);
    for i in 0..one_bit_position {
        sum_bits.push(*lhs_bits.get_lsb(i));
    }

    let upper_lhs_bits = lhs_bits.get_lsb_slice(one_bit_position, bit_count - one_bit_position);
    let upper_rhs_bits = AigBitVector::zeros(bit_count - one_bit_position);
    let (_, upper_sum_bits) = gatify_add_with_mapping(
        adder_mapping,
        &upper_lhs_bits,
        &upper_rhs_bits,
        gb.get_true(),
        None,
        gb,
    );
    sum_bits.extend(upper_sum_bits.iter_lsb_to_msb().copied());
    AigBitVector::from_lsb_is_index_0(&sum_bits)
}

/// Recognizes literals with one contiguous run of 1s and zeros elsewhere.
fn get_single_ones_run(bits: &xlsynth_pir::IrBits) -> Option<Range<usize>> {
    let bit_count = bits.get_bit_count();
    let mut run_start = 0usize;
    while run_start < bit_count
        && !bits
            .get_bit(run_start)
            .expect("literal bit index should be in bounds during run detection")
    {
        run_start += 1;
    }

    let mut run_end = run_start;
    while run_end < bit_count
        && bits
            .get_bit(run_end)
            .expect("literal bit index should be in bounds during run detection")
    {
        run_end += 1;
    }

    for i in run_end..bit_count {
        if bits
            .get_bit(i)
            .expect("literal bit index should be in bounds during run detection")
        {
            return None;
        }
    }

    Some(run_start..run_end)
}

/// Returns the one-hot bit position if `bits` is exactly `1 << k`.
fn get_single_one_bit_position(bits: &xlsynth_pir::IrBits) -> Option<usize> {
    let mut one_bit_position = None;
    for i in 0..bits.get_bit_count() {
        if !bits
            .get_bit(i)
            .expect("literal bit index should be in bounds during one-hot detection")
        {
            continue;
        }
        if one_bit_position.is_some() {
            return None;
        }
        one_bit_position = Some(i);
    }
    one_bit_position
}

/// Lowers `lhs_bits + literal_run + carry_in` where `literal_run` has one
/// contiguous run of 1s and zeros elsewhere.
fn gatify_add_const_single_ones_run(
    gb: &mut GateBuilder,
    lhs_bits: &AigBitVector,
    ones_run: Range<usize>,
    carry_in: AigOperand,
) -> AigBitVector {
    let bit_count = lhs_bits.get_bit_count();
    assert!(ones_run.start <= ones_run.end && ones_run.end <= bit_count);

    let mut sum = Vec::with_capacity(bit_count);
    let mut carry = carry_in;

    // For rhs_i=0, this is an increment-by-carry chain:
    // sum_i = lhs_i ^ carry and c_{i+1} = lhs_i & carry.
    for i in 0..ones_run.start {
        let lhs_i = *lhs_bits.get_lsb(i);
        let sum_i = gb.add_xor_binary(lhs_i, carry);
        sum.push(sum_i);
        carry = gb.add_and_binary(lhs_i, carry);
    }

    // For rhs_i=1, carry recurrence is c_{i+1} = lhs_i | carry,
    // and sum_i = !(lhs_i ^ carry).
    for i in ones_run {
        let lhs_i = *lhs_bits.get_lsb(i);
        let lhs_xor_carry = gb.add_xor_binary(lhs_i, carry);
        let sum_i = gb.add_not(lhs_xor_carry);
        sum.push(sum_i);
        carry = gb.add_or_binary(lhs_i, carry);
    }

    for i in sum.len()..bit_count {
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
    literal_bits: &xlsynth_pir::IrBits,
    adder_mapping: AdderMapping,
) -> AigBitVector {
    assert_eq!(sum_bits.get_bit_count(), literal_bits.get_bit_count());
    if let Some(one_bit_position) = get_single_one_bit_position(literal_bits) {
        return gatify_add_const_pow2_with_mapping(gb, sum_bits, one_bit_position, adder_mapping);
    }
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

#[cfg(test)]
mod tests {
    use super::selected_unit_delta_arrival_margin;

    #[test]
    fn selected_unit_delta_arrival_margin_scales_with_output_width() {
        assert_eq!(selected_unit_delta_arrival_margin(0), 2);
        assert_eq!(selected_unit_delta_arrival_margin(1), 2);
        assert_eq!(selected_unit_delta_arrival_margin(4), 2);
        assert_eq!(selected_unit_delta_arrival_margin(7), 3);
        assert_eq!(selected_unit_delta_arrival_margin(8), 3);
        assert_eq!(selected_unit_delta_arrival_margin(9), 4);
        assert_eq!(selected_unit_delta_arrival_margin(12), 4);
        assert_eq!(selected_unit_delta_arrival_margin(25), 5);
        assert_eq!(selected_unit_delta_arrival_margin(54), 6);
    }
}

fn classify_ext_nary_add_term_dimensions(
    bits: &AigBitVector,
    false_bit: AigOperand,
) -> ExtNaryAddTermDimensions {
    let bit_count = bits.get_bit_count();
    let mut weight_shift = 0usize;
    while weight_shift < bit_count && *bits.get_lsb(weight_shift) == false_bit {
        weight_shift += 1;
    }
    if weight_shift == bit_count {
        return ExtNaryAddTermDimensions {
            term_bits: AigBitVector::zeros(0),
            weight_shift,
            active_range: weight_shift..weight_shift,
            fill_kind: ExtNaryAddFillKind::Zero,
        };
    }

    let msb = *bits.get_lsb(bit_count - 1);
    if msb == false_bit {
        let mut active_end = bit_count;
        while active_end > weight_shift && *bits.get_lsb(active_end - 1) == false_bit {
            active_end -= 1;
        }
        return ExtNaryAddTermDimensions {
            term_bits: bits.get_lsb_slice(weight_shift, active_end - weight_shift),
            weight_shift,
            active_range: weight_shift..active_end,
            fill_kind: ExtNaryAddFillKind::Zero,
        };
    }

    let mut active_end = bit_count;
    while active_end > weight_shift + 1 && *bits.get_lsb(active_end - 2) == msb {
        active_end -= 1;
    }
    ExtNaryAddTermDimensions {
        term_bits: bits.get_lsb_slice(weight_shift, active_end - weight_shift),
        weight_shift,
        active_range: weight_shift..active_end,
        fill_kind: ExtNaryAddFillKind::Sign,
    }
}

fn classify_ext_nary_add_unit_correction(
    dimensions: &ExtNaryAddTermDimensions,
    negated: bool,
) -> Option<ExtNaryAddUnitCorrection> {
    if dimensions.term_bits.get_bit_count() != 1 {
        return None;
    }
    if dimensions.active_range != (dimensions.weight_shift..dimensions.weight_shift + 1) {
        return None;
    }

    Some(ExtNaryAddUnitCorrection {
        control: *dimensions.term_bits.get_lsb(0),
        weight_shift: dimensions.weight_shift,
        is_decrement: negated ^ (dimensions.fill_kind == ExtNaryAddFillKind::Sign),
    })
}

/// Adds 1 modulo the bit width of `literal_sum`.
fn increment_literal_sum_by_one(literal_sum: &mut xlsynth_pir::IrBits) {
    let one_bits = xlsynth_pir::IrBits::make_ubits(literal_sum.get_bit_count(), 1)
        .expect("bits[output_width]:1 should construct");
    *literal_sum = literal_sum.add(&one_bits);
}

/// Subtracts 1 modulo the bit width of `literal_sum`.
fn decrement_literal_sum_by_one(literal_sum: &mut xlsynth_pir::IrBits) {
    let one_bits = xlsynth_pir::IrBits::make_ubits(literal_sum.get_bit_count(), 1)
        .expect("bits[output_width]:1 should construct");
    *literal_sum = literal_sum.add(&one_bits.negate());
}

/// Tries to fuse one bit-0 unit correction into the final CPA carry-in.
fn try_fuse_ext_nary_add_unit_correction_into_carry_in(
    gb: &mut GateBuilder,
    unit_correction: ExtNaryAddUnitCorrection,
    literal_sum: &mut xlsynth_pir::IrBits,
    carry_in: &mut Option<AigOperand>,
) -> bool {
    if unit_correction.weight_shift != 0 || carry_in.is_some() {
        return false;
    }
    if unit_correction.is_decrement {
        decrement_literal_sum_by_one(literal_sum);
        *carry_in = Some(gb.add_not(unit_correction.control));
    } else {
        *carry_in = Some(unit_correction.control);
    }
    true
}

/// Falls back to the dense representation for a unit correction.
fn push_ext_nary_add_unit_correction_as_dense_term(
    gb: &mut GateBuilder,
    output_width: usize,
    unit_correction: ExtNaryAddUnitCorrection,
    lowered_terms: &mut Vec<AigBitVector>,
    literal_sum: &mut xlsynth_pir::IrBits,
) {
    if unit_correction.weight_shift >= output_width {
        return;
    }
    let mut shifted_bits = vec![gb.get_false(); output_width];
    shifted_bits[unit_correction.weight_shift] = unit_correction.control;
    let shifted_operand = AigBitVector::from_lsb_is_index_0(&shifted_bits);
    if unit_correction.is_decrement {
        lowered_terms.push(gb.add_not_vec(&shifted_operand));
        increment_literal_sum_by_one(literal_sum);
    } else {
        lowered_terms.push(shifted_operand);
    }
}

fn gatify_dense_ext_nary_add_terms(
    gb: &mut GateBuilder,
    mut lowered_terms: Vec<AigBitVector>,
    literal_sum: &xlsynth_pir::IrBits,
    carry_in: Option<AigOperand>,
    adder_mapping: AdderMapping,
) -> AigBitVector {
    if lowered_terms.is_empty() {
        if carry_in.is_none() {
            return gb.add_literal(literal_sum);
        }
        lowered_terms.push(AigBitVector::zeros(literal_sum.get_bit_count()));
    }

    let mut literal_sum = literal_sum.clone();
    let carry_in = if carry_in.is_none() && is_one(&literal_sum) && !lowered_terms.is_empty() {
        literal_sum = xlsynth_pir::IrBits::zero(literal_sum.get_bit_count());
        Some(gb.get_true())
    } else {
        carry_in
    };

    if lowered_terms.len() == 1 {
        if literal_sum.is_zero() {
            if let Some(carry_in) = carry_in {
                let zero_bits = AigBitVector::zeros(literal_sum.get_bit_count());
                return gatify_add_with_mapping(
                    adder_mapping,
                    &lowered_terms[0],
                    &zero_bits,
                    carry_in,
                    None,
                    gb,
                )
                .1;
            }
            return lowered_terms[0].clone();
        }
        if let Some(ones_run) = get_single_ones_run(&literal_sum)
            && (carry_in.is_some() || ones_run.start == 0 || ones_run.len() != 1)
        {
            return gatify_add_const_single_ones_run(
                gb,
                &lowered_terms[0],
                ones_run,
                carry_in.unwrap_or_else(|| gb.get_false()),
            );
        }
        if carry_in.is_none() {
            return gatify_add_literal_to_dynamic_sum(
                gb,
                &lowered_terms[0],
                &literal_sum,
                adder_mapping,
            );
        }
    }

    if carry_in.is_none() && get_pow2_minus1_k(&literal_sum).is_some() {
        let dynamic_sum = array_add_with_carry_out(gb, &lowered_terms, carry_in, adder_mapping).sum;
        return gatify_add_literal_to_dynamic_sum(gb, &dynamic_sum, &literal_sum, adder_mapping);
    }

    if !literal_sum.is_zero() {
        lowered_terms.push(gb.add_literal(&literal_sum));
    }
    array_add_with_carry_out(gb, &lowered_terms, carry_in, adder_mapping).sum
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
        let lhs_gate_refs = env
            .get_bit_vector(lhs)
            .expect("add lhs should be present for literal-rhs rewrite");
        return gatify_add_literal_to_dynamic_sum(
            g8_builder,
            &lhs_gate_refs,
            &rhs_bits,
            adder_mapping,
        );
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
    enable_rewrite_nary_add: bool,
    g8_builder: &mut GateBuilder,
) -> AigBitVector {
    if output_width == 0 {
        return AigBitVector::zeros(0);
    }
    if terms.is_empty() {
        return AigBitVector::zeros(output_width);
    }

    let false_bit = g8_builder.get_false();
    let selected_negates: Vec<Option<SelectedNegateMatch>> = if enable_rewrite_nary_add {
        terms
            .iter()
            .map(|term| match_selected_negate_term(f, env, term, output_width, false_bit))
            .collect()
    } else {
        vec![None; terms.len()]
    };
    let selected_negate_count = selected_negates.iter().flatten().count();
    let selected_depth = selected_negates
        .iter()
        .flatten()
        .filter_map(|matched| max_aig_depth(g8_builder, &matched.selected_bits))
        .max();
    let other_depth = terms
        .iter()
        .zip(&selected_negates)
        .filter(|(term, matched)| {
            matched.is_none() && literal_bits_if_bits_node(f, term.operand).is_none()
        })
        .filter_map(|(term, _)| env.get_bit_vector(term.operand).ok())
        .filter_map(|bits| max_aig_depth(g8_builder, &bits))
        .max();
    let fuse_selected_negates = match selected_negate_count {
        0 => false,
        1 => selected_depth.is_some_and(|selected_depth| {
            other_depth.unwrap_or(0).saturating_add(2) < selected_depth
        }),
        _ => true,
    };
    if selected_negate_count != 0 {
        log::debug!(
            "selected-negate lowering: adder={} width={} matches={} selected_depth={:?} other_depth={:?} choice={}",
            text_id,
            output_width,
            selected_negate_count,
            selected_depth,
            other_depth,
            if fuse_selected_negates {
                "carry-in"
            } else {
                "mux"
            }
        );
    }

    let mut literal_sum = xlsynth_pir::IrBits::zero(output_width);
    let mut lowered_terms: Vec<AigBitVector> = Vec::with_capacity(terms.len());
    let mut carry_in: Option<AigOperand> = None;
    for (term, selected_negate) in terms.iter().zip(selected_negates) {
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

        let (bits, selected_correction) =
            if fuse_selected_negates && let Some(matched) = selected_negate {
                let control = if matched.invert_control {
                    g8_builder.add_not(matched.control)
                } else {
                    matched.control
                };
                let conditional_complement = AigBitVector::from_lsb_is_index_0(
                    &matched
                        .base_bits
                        .iter_lsb_to_msb()
                        .map(|bit| g8_builder.add_xor_binary(*bit, control))
                        .collect::<Vec<_>>(),
                );
                (
                    conditional_complement,
                    Some(ExtNaryAddUnitCorrection {
                        control,
                        weight_shift: 0,
                        is_decrement: term.negated,
                    }),
                )
            } else {
                (
                    env.get_bit_vector(term.operand)
                        .expect("ext_nary_add operand should be present"),
                    None,
                )
            };
        let resized = if term.signed {
            gatify_sext_or_truncate(g8_builder, text_id, output_width, &bits)
        } else {
            gatify_zext_or_truncate(output_width, &bits)
        };

        let dimensions = classify_ext_nary_add_term_dimensions(&resized, false_bit);
        if let Some(unit_correction) =
            classify_ext_nary_add_unit_correction(&dimensions, term.negated)
        {
            if !try_fuse_ext_nary_add_unit_correction_into_carry_in(
                g8_builder,
                unit_correction,
                &mut literal_sum,
                &mut carry_in,
            ) {
                push_ext_nary_add_unit_correction_as_dense_term(
                    g8_builder,
                    output_width,
                    unit_correction,
                    &mut lowered_terms,
                    &mut literal_sum,
                );
            }
        } else {
            if term.negated {
                lowered_terms.push(g8_builder.add_not_vec(&resized));
                increment_literal_sum_by_one(&mut literal_sum);
            } else {
                lowered_terms.push(resized);
            }
        }

        if let Some(unit_correction) = selected_correction
            && !try_fuse_ext_nary_add_unit_correction_into_carry_in(
                g8_builder,
                unit_correction,
                &mut literal_sum,
                &mut carry_in,
            )
        {
            push_ext_nary_add_unit_correction_as_dense_term(
                g8_builder,
                output_width,
                unit_correction,
                &mut lowered_terms,
                &mut literal_sum,
            );
        }
    }

    let selected_adder_mapping = arch
        .map(AdderMapping::from)
        .unwrap_or(default_adder_mapping);
    let selected_adder_mapping_name = match selected_adder_mapping {
        AdderMapping::RippleCarry => "ripple_carry",
        AdderMapping::BrentKung => "brent_kung",
        AdderMapping::KoggeStone => "kogge_stone",
    };
    let sum = gatify_dense_ext_nary_add_terms(
        g8_builder,
        lowered_terms,
        &literal_sum,
        carry_in,
        selected_adder_mapping,
    );
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
