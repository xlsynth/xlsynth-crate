// SPDX-License-Identifier: Apache-2.0

/// State for a bit-serial adder that advances one bit per `add_step` call.
pub struct AddSeqState<N: u32> {
    remaining_x: uN[N],
    remaining_y: uN[N],
    sum: uN[N],
    bit_mask: uN[N],
    carry: u1,
}

/// Seeds a bit-serial add state for `x + y + carry_in`.
pub fn add_init<N: u32>(x: uN[N], y: uN[N], carry_in: u1) -> AddSeqState<N> {
    const_assert!(N > u32:0);
    AddSeqState {
        remaining_x: x,
        remaining_y: y,
        sum: uN[N]:0,
        bit_mask: uN[N]:1,
        carry: carry_in,
    }
}

/// Advances a bit-serial add by one least-significant-bit step.
///
/// Uarch tradeoff: one call performs one full-adder cell worth of arithmetic,
/// plus the surrounding state shifts. Repeating this state transition across
/// cycles trades latency for very small per-cycle combinational logic.
pub fn add_step<N: u32>(state: AddSeqState<N>) -> AddSeqState<N> {
    const_assert!(N > u32:0);
    let xb = state.remaining_x[0+:u1];
    let yb = state.remaining_y[0+:u1];
    let sum_bit = xb ^ yb ^ state.carry;
    let next_carry = (xb & yb) | (xb & state.carry) | (yb & state.carry);
    let next_sum = if sum_bit == u1:1 { state.sum | state.bit_mask } else { state.sum };
    AddSeqState {
        remaining_x: state.remaining_x >> u32:1,
        remaining_y: state.remaining_y >> u32:1,
        sum: next_sum,
        bit_mask: state.bit_mask << u32:1,
        carry: next_carry,
    }
}

/// Adds `x + y + carry_in` by unrolling the bit-serial state machine.
///
/// Uarch tradeoff: this wrapper is a compositional reference for `add_step`;
/// external callers should schedule `add_step` across cycles when they want the
/// sequential implementation rather than this fully unrolled combinational one.
pub fn add_serial_with_carry<N: u32>(x: uN[N], y: uN[N], carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    let state = for (_, state): (u32, AddSeqState<N>) in u32:0..N {
        add_step(state)
    }(add_init(x, y, carry_in));
    (state.sum, state.carry)
}

/// Adds `x + y` by unrolling the bit-serial state machine.
pub fn add_serial<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    let (sum, _) = add_serial_with_carry(x, y, u1:0);
    sum
}

/// Computes the widening-add reference result for serial adders.
fn expected_add_with_carry<N: u32>(x: uN[N], y: uN[N], carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    let full = (x as uN[N + u32:1]) + (y as uN[N + u32:1]) + (carry_in as uN[N + u32:1]);
    (full as uN[N], full[N+:u1])
}

#[quickcheck]
fn qc_add_unrolled_seq_matches_builtin_u8(x: u8, y: u8, carry_in: u1) -> bool {
    add_serial(x, y) == x + y &&
    add_serial_with_carry(x, y, carry_in) == expected_add_with_carry(x, y, carry_in)
}

#[quickcheck]
fn qc_add_unrolled_seq_matches_builtin_u16(x: u16, y: u16, carry_in: u1) -> bool {
    add_serial(x, y) == x + y &&
    add_serial_with_carry(x, y, carry_in) == expected_add_with_carry(x, y, carry_in)
}

#[quickcheck]
fn qc_add_unrolled_seq_matches_builtin_u32(x: u32, y: u32, carry_in: u1) -> bool {
    add_serial(x, y) == x + y &&
    add_serial_with_carry(x, y, carry_in) == expected_add_with_carry(x, y, carry_in)
}
