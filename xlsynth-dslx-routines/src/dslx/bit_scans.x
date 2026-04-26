// SPDX-License-Identifier: Apache-2.0

/// Returns whether bit `i` of `x` is set.
///
/// Uarch tradeoff: this is just a dynamic bit extraction helper; it should
/// lower to wiring/muxing for the selected bit and carries no scan structure by
/// itself.
pub fn bit_is_set<N: u32>(x: bits[N], i: u32) -> bool { x[i+:u1] == u1:1 }

/// Computes the inclusive prefix OR of `x` from least-significant bit upward.
///
/// Uarch tradeoff: this describes a simple serial "seen one" chain with low
/// local fanout and O(N) logical depth. It is useful as a compact reference or
/// when later optimization can recognize and rebalance the prefix structure.
pub fn prefix_or_lsb<N: u32>(x: bits[N]) -> bits[N] {
    const_assert!(N > u32:0);
    for (i, partial): (u32, bits[N]) in u32:0..N {
        let prior = if i == u32:0 { u1:0 } else { partial[(i - u32:1)+:u1] };
        bit_slice_update(partial, i, prior | x[i+:u1])
    }(zero!<bits[N]>())
}

/// Computes the inclusive prefix OR of `x` from most-significant bit downward.
///
/// Uarch tradeoff: this is the MSB-first form of a serial prefix chain, which
/// maps naturally to leading-zero/priority-style masks. It favors small,
/// regular logic over the shorter depth and higher wiring cost of a tree.
pub fn prefix_or_msb<N: u32>(x: bits[N]) -> bits[N] {
    const_assert!(N > u32:0);
    for (i, partial): (u32, bits[N]) in u32:0..N {
        let bit_index = N - u32:1 - i;
        let prior = if i == u32:0 { u1:0 } else { partial[(bit_index + u32:1)+:u1] };
        bit_slice_update(partial, bit_index, prior | x[bit_index+:u1])
    }(zero!<bits[N]>())
}

/// Counts the set bits in `x`, returning the count in an `N`-bit word.
///
/// Uarch tradeoff: the loop exposes an accumulator-style popcount, so the
/// direct structure has adder-chain depth. It is a straightforward companion
/// for mask-producing scans when code size and clarity matter more than a
/// hand-balanced popcount tree.
pub fn count_set_bits<N: u32>(x: bits[N]) -> uN[N] {
    const_assert!(N > u32:0);
    for (i, count): (u32, uN[N]) in u32:0..N {
        count + (x[i+:u1] as uN[N])
    }(uN[N]:0)
}

/// Adds `x + y + carry_in` with a bit-serial ripple-carry network.
///
/// Uarch tradeoff: ripple carry is the smallest and most local adder shape,
/// with one carry dependency per bit and O(N) carry depth. Use it when area and
/// wiring are more important than the longest combinational path.
pub fn add_ripple_from_carry<N: u32>(x: uN[N], y: uN[N], carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    for (i, state): (u32, (uN[N], u1)) in u32:0..N {
        let (sum, carry) = state;
        let xb = x[i+:u1];
        let yb = y[i+:u1];
        let sum_bit = xb ^ yb ^ carry;
        let next_carry = (xb & yb) | (xb & carry) | (yb & carry);
        (bit_slice_update(sum, i, sum_bit), next_carry)
    }((uN[N]:0, carry_in))
}

/// Adds `x + y` with a bit-serial ripple-carry network.
///
/// Uarch tradeoff: this is the sum-only wrapper around the carry-capable
/// ripple adder, preserving the same compact O(N)-depth structure while
/// discarding carry-out for normal wrapping add semantics.
pub fn add_ripple<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    let (sum, _) = add_ripple_from_carry(x, y, u1:0);
    sum
}

/// Computes parallel-prefix group generate bits from generate/propagate inputs.
///
/// Uarch tradeoff: this is a prefix-carry building block with logarithmic-style
/// dependency distance but more intermediate generate/propagate logic than a
/// ripple chain. It trades area and routing pressure for shorter carry depth.
pub fn prefix_generate<N: u32>(generate: bits[N], propagate: bits[N]) -> bits[N] {
    const_assert!(N > u32:0);
    let (group_generate, _) = for (stage, state): (u32, (bits[N], bits[N])) in u32:0..N {
        let (g, p) = state;
        let distance = u32:1 << stage;
        if distance >= N {
            (g, p)
        } else {
            for (i, inner): (u32, (bits[N], bits[N])) in u32:0..N {
                let (next_g, next_p) = inner;
                if i >= distance {
                    let gi = g[i+:u1];
                    let pi = p[i+:u1];
                    let gj = g[(i - distance)+:u1];
                    let pj = p[(i - distance)+:u1];
                    (
                        bit_slice_update(next_g, i, gi | (pi & gj)),
                        bit_slice_update(next_p, i, pi & pj),
                    )
                } else {
                    (next_g, next_p)
                }
            }((g, p))
        }
    }((generate, propagate));
    group_generate
}

/// Adds `x + y + carry_in` by computing carries with a parallel-prefix tree.
///
/// Uarch tradeoff: prefix carry reduces the long carry dependency of ripple at
/// the cost of more generate/propagate gates and wider communication between
/// bit positions. The carry-in is folded into bit zero so the same prefix
/// network produces both internal carries and carry-out.
pub fn add_prefix_from_carry<N: u32>(x: uN[N], y: uN[N], carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    let propagate = x ^ y;
    let generate = x & y;
    let carry_seeded_generate =
        bit_slice_update(generate, u32:0, generate[0+:u1] | (propagate[0+:u1] & carry_in));
    let carries = prefix_generate(carry_seeded_generate, propagate);
    let sum = for (i, partial): (u32, uN[N]) in u32:0..N {
        let bit_carry_in = if i == u32:0 { carry_in } else { carries[(i - u32:1)+:u1] };
        bit_slice_update(partial, i, propagate[i+:u1] ^ bit_carry_in)
    }(uN[N]:0);
    (sum, carries[(N - u32:1)+:u1])
}

/// Adds `x + y` by computing carries with a parallel-prefix generate tree.
///
/// Uarch tradeoff: this is the sum-only wrapper around the prefix adder. It
/// keeps the shorter carry-depth structure for normal wrapping add semantics,
/// while omitting the observable carry-out bit.
pub fn add_prefix<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    let (sum, _) = add_prefix_from_carry(x, y, u1:0);
    sum
}

/// Ripple-adds one `GROUP`-sized block of `x + y` for carry-select adders.
///
/// Uarch tradeoff: each block keeps a local ripple chain, so `GROUP` controls
/// the local carry delay. Smaller groups shorten local ripple depth but create
/// more block-select boundaries; larger groups reduce boundary overhead but
/// move the design back toward ripple behavior.
pub fn ripple_group<N: u32, GROUP: u32 = {u32:4}>
    (x: uN[N], y: uN[N], group_index: u32, carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    const_assert!(GROUP > u32:0);
    let start = group_index * GROUP;
    let limit = if start + GROUP < N { start + GROUP } else { N };
    for (i, state): (u32, (uN[N], u1)) in u32:0..N {
        let (sum, carry) = state;
        if i >= start && i < limit {
            let xb = x[i+:u1];
            let yb = y[i+:u1];
            let sum_bit = xb ^ yb ^ carry;
            let next_carry = (xb & yb) | (xb & carry) | (yb & carry);
            (bit_slice_update(sum, i, sum_bit), next_carry)
        } else {
            (sum, carry)
        }
    }((uN[N]:0, carry_in))
}

/// Adds `x + y + carry_in` using carry-select blocks of `GROUP` bits.
///
/// Uarch tradeoff: carry-select precomputes each block for carry-in zero and
/// one, then muxes between the candidates. This costs roughly duplicated block
/// logic, but it can cut the carry path to local ripple plus inter-block
/// selection; tune `GROUP` to balance area against depth.
pub fn add_carry_select_from_carry<N: u32, GROUP: u32 = {u32:4}>
    (x: uN[N], y: uN[N], carry_in: u1) -> (uN[N], u1) {
    const_assert!(N > u32:0);
    const_assert!(GROUP > u32:0);
    const GROUPS = (N + GROUP - u32:1) / GROUP;
    for (group_index, state): (u32, (uN[N], u1)) in u32:0..GROUPS {
        let (partial_sum, carry) = state;
        let (sum0, carry0) = ripple_group<N, GROUP>(x, y, group_index, u1:0);
        let (sum1, carry1) = ripple_group<N, GROUP>(x, y, group_index, u1:1);
        if carry == u1:1 { (partial_sum | sum1, carry1) } else { (partial_sum | sum0, carry0) }
    }((uN[N]:0, carry_in))
}

/// Adds `x + y` using carry-select blocks of `GROUP` bits.
///
/// Uarch tradeoff: this sum-only wrapper keeps the carry-select structure for
/// wrapping add semantics. It is typically larger than ripple, but the
/// `GROUP` knob can reduce depth when the extra duplicated block logic is
/// acceptable.
pub fn add_carry_select<N: u32, GROUP: u32 = {u32:4}>(x: uN[N], y: uN[N]) -> uN[N] {
    const_assert!(N > u32:0);
    let (sum, _) = add_carry_select_from_carry<N, GROUP>(x, y, u1:0);
    sum
}
