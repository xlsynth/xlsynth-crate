// SPDX-License-Identifier: Apache-2.0

/// State for a shift-add multiplier that advances one multiplier bit per step.
pub struct MulSeqState<N: u32> {
    multiplicand: uN[N],
    multiplier: uN[N],
    product: uN[N],
}

/// Seeds a shift-add multiply state for `x * y`.
pub fn mul_init<N: u32>(x: uN[N], y: uN[N]) -> MulSeqState<N> {
    const_assert!(N > u32:0);
    MulSeqState { multiplicand: x, multiplier: y, product: uN[N]:0 }
}

/// Advances a shift-add multiply by one multiplier bit.
///
/// Uarch tradeoff: one call conditionally accumulates the current multiplicand,
/// then shifts both operands for the next cycle. Repeating this transition over
/// time replaces an N-row combinational multiplier with N sequential add steps.
pub fn mul_step<N: u32>(state: MulSeqState<N>) -> MulSeqState<N> {
    const_assert!(N > u32:0);
    let next_product =
        if state.multiplier[0+:u1] == u1:1 { state.product + state.multiplicand } else { state.product };
    MulSeqState {
        multiplicand: state.multiplicand << u32:1,
        multiplier: state.multiplier >> u32:1,
        product: next_product,
    }
}

/// Multiplies `x * y` by unrolling the shift-add state machine.
///
/// Uarch tradeoff: this wrapper is a compositional reference for `mul_step`;
/// external callers should schedule `mul_step` across cycles when they want the
/// sequential implementation rather than this fully unrolled combinational one.
pub fn mul_serial<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    let state = for (_, state): (u32, MulSeqState<N>) in u32:0..N {
        mul_step(state)
    }(mul_init(x, y));
    state.product
}

// Keep the proof widths modest; multiplication obligations become expensive
// much faster than the adder checks in the sibling module.
#[quickcheck]
fn qc_mul_unrolled_seq_matches_builtin_u4(x: u4, y: u4) -> bool { mul_serial(x, y) == x * y }

#[quickcheck]
fn qc_mul_unrolled_seq_matches_builtin_u6(x: u6, y: u6) -> bool { mul_serial(x, y) == x * y }

#[quickcheck]
fn qc_mul_unrolled_seq_matches_builtin_u8(x: u8, y: u8) -> bool { mul_serial(x, y) == x * y }
