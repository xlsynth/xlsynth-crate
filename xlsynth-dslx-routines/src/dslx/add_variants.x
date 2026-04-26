// SPDX-License-Identifier: Apache-2.0

import bit_scans;

/// Adds `x + y + carry_in` using a ripple-carry implementation.
///
/// Uarch tradeoff: ripple carry has minimal duplicated logic and very local
/// wiring, but its carry dependency grows linearly with `N`. This variant is
/// the baseline for area-oriented adders and exposes carry-out for chaining.
pub fn add_ripple_with_carry<N: u32>(x: uN[N], y: uN[N], carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    bit_scans::add_ripple_from_carry(x, y, carry_in)
}

/// Adds `x + y` using ripple-carry and discards the carry-out bit.
///
/// Uarch tradeoff: this wrapper preserves the compact ripple structure while
/// matching DSLX wrapping-add semantics. It is useful when the consumer wants a
/// normal add but still wants to select the ripple implementation explicitly.
pub fn add_ripple<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    let (sum, _) = add_ripple_with_carry(x, y, u1:0);
    sum
}

/// Adds `x + y + carry_in` using a parallel-prefix carry implementation.
///
/// Uarch tradeoff: prefix carry reduces carry depth substantially relative to
/// ripple, at the cost of extra generate/propagate logic and wider bit-to-bit
/// communication. This is the depth-oriented baseline and includes carry-out.
pub fn add_prefix_with_carry<N: u32>(x: uN[N], y: uN[N], carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    bit_scans::add_prefix_from_carry(x, y, carry_in)
}

/// Adds `x + y` using prefix carry and discards the carry-out bit.
///
/// Uarch tradeoff: this wrapper keeps the prefix network for normal wrapping
/// add semantics. It is usually larger than ripple but has a much shorter
/// carry path for wider words.
pub fn add_prefix<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    let (sum, _) = add_prefix_with_carry(x, y, u1:0);
    sum
}

/// Adds `x + y + carry_in` using carry-select blocks of `GROUP` bits.
///
/// Uarch tradeoff: carry-select duplicates each block for both possible carry
/// inputs and selects the right result once the incoming carry is known. Smaller
/// `GROUP` values favor depth at higher mux/control overhead; larger values
/// save area but increase local ripple delay.
pub fn add_carry_select_with_carry<N: u32, GROUP: u32 = {u32:4}>
    (x: uN[N], y: uN[N], carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    bit_scans::add_carry_select_from_carry<N, GROUP>(x, y, carry_in)
}

/// Adds `x + y` using carry-select blocks and discards the carry-out bit.
///
/// Uarch tradeoff: this wrapper gives normal wrapping-add behavior while
/// retaining the tunable carry-select shape. It sits between ripple and prefix:
/// more duplicated logic than ripple, often less global prefix wiring than a
/// full prefix adder.
pub fn add_carry_select<N: u32, GROUP: u32 = {u32:4}>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    let (sum, _) = add_carry_select_with_carry<N, GROUP>(x, y, u1:0);
    sum
}

/// Computes the widening-add reference result for carry-capable adders.
///
/// Uarch tradeoff: this is a proof-only specification helper, not an
/// implementation candidate. It intentionally uses a widened builtin add so the
/// quickchecks compare against DSLX arithmetic semantics directly.
fn expected_add_with_carry<N: u32>(x: uN[N], y: uN[N], carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    let full = (x as uN[N + u32:1]) + (y as uN[N + u32:1]) + (carry_in as uN[N + u32:1]);
    (full as uN[N], full[N+:u1])
}

#[quickcheck]
fn qc_add_u8(x: u8, y: u8, carry_in: u1) -> bool {
    let expected_sum = x + y;
    let expected_with_carry = expected_add_with_carry(x, y, carry_in);
    add_ripple(x, y) == expected_sum && add_prefix(x, y) == expected_sum &&
    add_carry_select(x, y) == expected_sum && add_carry_select<u32:8, u32:2>(x, y) == expected_sum &&
    add_ripple_with_carry(x, y, carry_in) == expected_with_carry &&
    add_prefix_with_carry(x, y, carry_in) == expected_with_carry &&
    add_carry_select_with_carry(x, y, carry_in) == expected_with_carry &&
    add_carry_select_with_carry<u32:8, u32:2>(x, y, carry_in) == expected_with_carry
}

#[quickcheck]
fn qc_add_u16(x: u16, y: u16, carry_in: u1) -> bool {
    let expected_sum = x + y;
    let expected_with_carry = expected_add_with_carry(x, y, carry_in);
    add_ripple(x, y) == expected_sum && add_prefix(x, y) == expected_sum &&
    add_carry_select(x, y) == expected_sum && add_carry_select<u32:16, u32:4>(x, y) == expected_sum &&
    add_ripple_with_carry(x, y, carry_in) == expected_with_carry &&
    add_prefix_with_carry(x, y, carry_in) == expected_with_carry &&
    add_carry_select_with_carry(x, y, carry_in) == expected_with_carry &&
    add_carry_select_with_carry<u32:16, u32:4>(x, y, carry_in) == expected_with_carry
}

#[quickcheck]
fn qc_add_u32(x: u32, y: u32, carry_in: u1) -> bool {
    let expected_sum = x + y;
    let expected_with_carry = expected_add_with_carry(x, y, carry_in);
    add_ripple(x, y) == expected_sum && add_prefix(x, y) == expected_sum &&
    add_carry_select(x, y) == expected_sum && add_carry_select<u32:32, u32:8>(x, y) == expected_sum &&
    add_ripple_with_carry(x, y, carry_in) == expected_with_carry &&
    add_prefix_with_carry(x, y, carry_in) == expected_with_carry &&
    add_carry_select_with_carry(x, y, carry_in) == expected_with_carry &&
    add_carry_select_with_carry<u32:32, u32:8>(x, y, carry_in) == expected_with_carry
}
