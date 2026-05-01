// SPDX-License-Identifier: Apache-2.0

import add_variants;

/// Multiplies `x * y` by serially accumulating shifted partial products.
///
/// Uarch tradeoff: this is the direct shift-and-add baseline. It emits one
/// conditional partial-product add per multiplier bit and lets the normal DSLX
/// add operator choose the downstream lowering shape.
pub fn mul_shift_add<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    for (i, product): (u32, uN[N]) in u32:0..N {
        let partial = if y[i+:u1] == u1:1 { x << i } else { uN[N]:0 };
        product + partial
    }(uN[N]:0)
}

/// Multiplies `x * y` with shift-and-add using explicit ripple adders.
///
/// Uarch tradeoff: ripple accumulation keeps each add compact and local, but
/// combines the multiplier's serial accumulation with linear carry depth.
pub fn mul_shift_add_ripple<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    for (i, product): (u32, uN[N]) in u32:0..N {
        let partial = if y[i+:u1] == u1:1 { x << i } else { uN[N]:0 };
        add_variants::add_ripple(product, partial)
    }(uN[N]:0)
}

/// Multiplies `x * y` with shift-and-add using explicit prefix adders.
///
/// Uarch tradeoff: prefix accumulation reduces the carry depth of each add at
/// the cost of more generate/propagate logic and broader wiring.
pub fn mul_shift_add_prefix<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    for (i, product): (u32, uN[N]) in u32:0..N {
        let partial = if y[i+:u1] == u1:1 { x << i } else { uN[N]:0 };
        add_variants::add_prefix(product, partial)
    }(uN[N]:0)
}

/// Multiplies `x * y` with shift-and-add using carry-select adders.
///
/// Uarch tradeoff: carry-select accumulation exposes the same `GROUP` tuning
/// knob as the standalone adder variants, trading duplicated block logic for a
/// shorter per-add carry path.
pub fn mul_shift_add_carry_select<N: u32, GROUP: u32 = {u32:4}>
    (x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    for (i, product): (u32, uN[N]) in u32:0..N {
        let partial = if y[i+:u1] == u1:1 { x << i } else { uN[N]:0 };
        add_variants::add_carry_select<N, GROUP>(product, partial)
    }(uN[N]:0)
}

// Keep the proof widths modest; multiplication obligations become expensive
// much faster than the adder checks in the sibling module.
#[quickcheck]
fn qc_mul_u4(x: u4, y: u4) -> bool {
    let expected = x * y;
    mul_shift_add(x, y) == expected && mul_shift_add_ripple(x, y) == expected &&
    mul_shift_add_prefix(x, y) == expected &&
    mul_shift_add_carry_select(x, y) == expected &&
    mul_shift_add_carry_select<u32:4, u32:2>(x, y) == expected
}

#[quickcheck]
fn qc_mul_u6(x: u6, y: u6) -> bool {
    let expected = x * y;
    mul_shift_add(x, y) == expected && mul_shift_add_ripple(x, y) == expected &&
    mul_shift_add_prefix(x, y) == expected &&
    mul_shift_add_carry_select(x, y) == expected &&
    mul_shift_add_carry_select<u32:6, u32:3>(x, y) == expected
}

#[quickcheck]
fn qc_mul_u8(x: u8, y: u8) -> bool {
    let expected = x * y;
    mul_shift_add(x, y) == expected && mul_shift_add_ripple(x, y) == expected &&
    mul_shift_add_prefix(x, y) == expected &&
    mul_shift_add_carry_select(x, y) == expected &&
    mul_shift_add_carry_select<u32:8, u32:4>(x, y) == expected
}
