// SPDX-License-Identifier: Apache-2.0
// Copyright 2020 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Adapted from apfloat::add<8, 23, false> in xls/dslx/stdlib/apfloat.x
// from xlsynth v0.59.0. The two stages split at significand alignment.
// This adder flushes subnormals to zero and rounds to nearest, ties to even.

import std;
import abs_diff;
import float32;

type F32 = float32::F32;

const EXP_SZ = u32:8;
const FRACTION_SZ = u32:23;

// WIDE_EXP: Widened exponent to capture a possible carry bit.
const WIDE_EXP: u32 = EXP_SZ + u32:1;

// CARRY_EXP: WIDE_EXP plus one sign bit.
const CARRY_EXP: u32 = u32:1 + WIDE_EXP;

// WIDE_FRACTION: Widened fraction to contain full precision + rounding
// (sign, hidden as well as guard, round, sticky) bits.
const SIGN_BIT = u32:1;
const HIDDEN_BIT = u32:1;
const GUARD_ROUND_STICKY_BITS = u32:3;
const FRACTION = HIDDEN_BIT + FRACTION_SZ;
const SIGNED_FRACTION = SIGN_BIT + FRACTION;
const WIDE_FRACTION: u32 = SIGNED_FRACTION + GUARD_ROUND_STICKY_BITS;

// CARRY_FRACTION: WIDE_FRACTION plus one bit to capture a possible carry bit.
const CARRY_FRACTION: u32 = u32:1 + WIDE_FRACTION;

// NORMALIZED_FRACTION: WIDE_FRACTION minus one bit for post normalization
// (where the implicit leading 1 bit is dropped).
const NORMALIZED_FRACTION: u32 = WIDE_FRACTION - u32:1;

struct Aligned {
    addend_x: s28,
    addend_y: s28,
    exponent: u8,
    x_sign: bool,
    y_sign: bool,
    is_operand_inf: bool,
    has_pos_inf: bool,
    is_result_nan: bool,
}

fn sign_magnitude_difference<WIDTH: u32>(a: uN[WIDTH], b: uN[WIDTH]) -> (bool, uN[WIDTH]) {
    let abs_diff_result = abs_diff::abs_diff(a, b);
    (!abs_diff::is_x_larger(abs_diff_result), abs_diff::to_corrected(abs_diff_result))
}

fn or_last_bit<WIDTH: u32>(value: bits[WIDTH], lsb: u1) -> bits[WIDTH] {
    bit_slice_update(value, u1:0, (value[0:1] | lsb))
}

pub fn add_f32_cycle0(a: F32, b: F32) -> Aligned {
    // Align to the operand with the larger exponent.
    let (a_is_smaller, shift) = sign_magnitude_difference(a.bexp, b.bexp);
    let (x, y) = if a_is_smaller { (b, a) } else { (a, b) };

    // Restore the hidden significand bit.
    let fraction_x = (u1:1 ++ x.fraction) as uN[FRACTION];
    let fraction_y = (u1:1 ++ y.fraction) as uN[FRACTION];

    // Flush denormals to 0.
    let fraction_x = if x.bexp == uN[EXP_SZ]:0 { uN[FRACTION]:0 } else { fraction_x };
    let fraction_y = if y.bexp == uN[EXP_SZ]:0 { uN[FRACTION]:0 } else { fraction_y };

    // Reserve the guard, round and sticky bits.
    let wide_x = fraction_x as uN[WIDE_FRACTION] << GUARD_ROUND_STICKY_BITS;
    let wide_y = fraction_y as uN[WIDE_FRACTION] << GUARD_ROUND_STICKY_BITS;

    // The larger operand needs no alignment shift. Negate it for subtraction.
    let addend_x = wide_x as sN[WIDE_FRACTION];
    let addend_x = if x.sign != y.sign { -addend_x } else { addend_x };

    // The sticky bit captures any one shifted out of the smaller operand.
    let sticky = std::or_reduce_lsb(wide_y, shift);
    let addend_y = or_last_bit(wide_y >> shift, sticky) as sN[WIDE_FRACTION];

    let inf_x = float32::is_inf(x);
    let inf_y = float32::is_inf(y);
    let has_pos_inf = (inf_x && !x.sign) || (inf_y && !y.sign);
    let has_neg_inf = (inf_x && x.sign) || (inf_y && y.sign);
    Aligned {
        addend_x,
        addend_y,
        exponent: x.bexp,
        x_sign: x.sign,
        y_sign: y.sign,
        is_operand_inf: inf_x || inf_y,
        has_pos_inf,
        is_result_nan: float32::is_nan(x) || float32::is_nan(y) || (has_pos_inf && has_neg_inf),
    }
}

pub fn add_f32_cycle1(p: Aligned) -> F32 {
    // Widen each aligned significand to s29 to capture a carry.
    let fraction = (p.addend_x as sN[CARRY_FRACTION]) + (p.addend_y as sN[CARRY_FRACTION]);
    let fraction_is_zero = fraction == sN[CARRY_FRACTION]:0;
    let result_sign = match (fraction_is_zero, fraction < sN[CARRY_FRACTION]:0) {
        (true, _) => p.x_sign && p.y_sign,  // So that -0.0 + -0.0 results in -0.0
        (false, true) => !p.y_sign,
        _ => p.y_sign,
    };

    // Get the absolute value of the result then chop off the sign bit: s29 -> u28.
    let abs_fraction =
        (if fraction < sN[CARRY_FRACTION]:0 { -fraction } else { fraction }) as uN[WIDE_FRACTION];

    // A carry requires shifting right one bit while preserving the sticky bit.
    let carry_bit = abs_fraction[-1:];
    let carry_fraction = (abs_fraction >> u32:1) as uN[NORMALIZED_FRACTION];
    let carry_fraction = or_last_bit(carry_fraction, abs_fraction[0:1]);

    // If we cancelled higher bits, then we'll need to shift left.
    const CLZ_WIDTH: u32 = std::clog2(WIDE_FRACTION + u32:1);
    let cancel_shift = std::clzt(abs_fraction);

    // Manually apply https://github.com/google/xls/issues/1274
    let cancel_fraction = abs_fraction as uN[WIDE_FRACTION + u32:1] << cancel_shift;
    let cancel_fraction = (cancel_fraction >> u32:1) as uN[NORMALIZED_FRACTION];
    let shifted_fraction = if carry_bit { carry_fraction } else { cancel_fraction };

    let leading_zeroes = if carry_bit { uN[CLZ_WIDTH]:0 } else { cancel_shift };

    // Round up above halfway, or exactly halfway when the retained LSB is odd.
    let normal_chunk = shifted_fraction[0:3];
    let half_way_chunk = shifted_fraction[2:4];
    let do_round_up = (normal_chunk > u3:0x4) || (half_way_chunk == u2:0x3);

    // We again need an extra bit for carry.
    let rounded_fraction = if do_round_up {
        (shifted_fraction as uN[WIDE_FRACTION]) + uN[WIDE_FRACTION]:0x8
    } else {
        shifted_fraction as uN[WIDE_FRACTION]
    };
    let rounding_carry = rounded_fraction[-1:];

    // After rounding, we can chop off the extra precision bits.
    // As with normalization, if we carried, we need to shift right
    // an extra place.
    let fraction_shift =
        GUARD_ROUND_STICKY_BITS as u3 + (if rounded_fraction[-1:] { u3:1 } else { u3:0 });
    let result_fraction = (rounded_fraction >> fraction_shift) as uN[FRACTION_SZ];

    // Finally, adjust the exponent based on addition and rounding -
    // each bit of carry or cancellation moves it by one place.
    let wide_exponent = (p.exponent as sN[CARRY_EXP]) + (rounding_carry as sN[CARRY_EXP]) +
                        sN[CARRY_EXP]:1 - (leading_zeroes as sN[CARRY_EXP]);
    let wide_exponent = if fraction_is_zero { sN[CARRY_EXP]:0 } else { wide_exponent };

    // Chop off the sign bit.
    let wide_exponent = if wide_exponent < sN[CARRY_EXP]:0 {
        uN[WIDE_EXP]:0
    } else {
        wide_exponent as uN[WIDE_EXP]
    };

    // Flush underflow to zero.
    let result_fraction =
        if wide_exponent < uN[WIDE_EXP]:1 { uN[FRACTION_SZ]:0 } else { result_fraction };

    // Handle exponent overflow infinities.
    const MAX_EXPONENT = std::mask_bits<EXP_SZ>();
    const SATURATED_EXPONENT = MAX_EXPONENT as uN[WIDE_EXP];
    let result_fraction =
        if wide_exponent < SATURATED_EXPONENT { result_fraction } else { uN[FRACTION_SZ]:0 };
    let result_exponent =
        if wide_exponent < SATURATED_EXPONENT { wide_exponent as uN[EXP_SZ] } else { MAX_EXPONENT };

    // Handle arg infinities.
    let result_exponent = if p.is_operand_inf { MAX_EXPONENT } else { result_exponent };
    let result_fraction = if p.is_operand_inf { uN[FRACTION_SZ]:0 } else { result_fraction };
    // An infinite result is negative if all infinite operands are negative.
    let result_sign = if p.is_operand_inf { !p.has_pos_inf } else { result_sign };

    // A NaN operand or oppositely signed infinities produce a quiet NaN.
    const FRACTION_HIGH_BIT = uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:1);
    let result_exponent = if p.is_result_nan { MAX_EXPONENT } else { result_exponent };
    let result_fraction = if p.is_result_nan { FRACTION_HIGH_BIT } else { result_fraction };
    let result_sign = if p.is_result_nan { u1:0 } else { result_sign };

    F32 { sign: result_sign, bexp: result_exponent, fraction: result_fraction as uN[FRACTION_SZ] }
}

pub fn add_f32_composed(a: F32, b: F32) -> F32 { add_f32_cycle1(add_f32_cycle0(a, b)) }
