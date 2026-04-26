// SPDX-License-Identifier: Apache-2.0

import std;
import bit_scans;

/// Counts leading zeros with a simple most-significant-bit-first scan.
///
/// Uarch tradeoff: this is the lowest-concept CLZ structure, carrying a
/// `seen_one` state from MSB to LSB. It tends to be compact and easy to
/// optimize locally, but the detection path is linear in the input width.
pub fn clz_linear<N: u32>(x: bits[N]) -> uN[N] {
    const_assert!(N > u32:0);
    let (_, count) = for (i, state): (u32, (bool, uN[N])) in u32:0..N {
        let (seen_one, count) = state;
        let bit_index = N - u32:1 - i;
        let bit_set = bit_scans::bit_is_set(x, bit_index);
        let next_count = if seen_one || bit_set { count } else { count + uN[N]:1 };
        (seen_one || bit_set, next_count)
    }((false, uN[N]:0));
    count
}

/// Counts leading zeros by building a leading-prefix mask and popcounting it.
///
/// Uarch tradeoff: this separates "which bits are before the first one" from
/// "how many such bits exist", which can expose useful mask and popcount
/// structure to later passes. It generally spends more gates than the linear
/// scan and is most attractive when those substructures are recognized or
/// shared.
pub fn clz_prefix_mask<N: u32>(x: bits[N]) -> uN[N] {
    const_assert!(N > u32:0);
    bit_scans::count_set_bits(!bit_scans::prefix_or_msb(x))
}

/// Counts leading zeros by scanning `GROUP`-bit chunks, where `GROUP` divides `N`.
///
/// Uarch tradeoff: this first finds the leading nonzero chunk, then scans only
/// within that chunk. Smaller `GROUP` values make the chunk search finer and
/// add more inter-group control; larger values reduce group overhead but leave
/// more local scan depth. The default is a conservative middle ground.
pub fn clz_grouped<N: u32, GROUP: u32 = {u32:4}>(x: bits[N]) -> uN[N] {
    const_assert!(N > u32:0);
    const_assert!(GROUP > u32:0);
    const_assert!(GROUP <= N);
    const_assert!(N % GROUP == u32:0);
    const GROUPS = N / GROUP;
    let (_, zero_groups, group_count) =
        for (group, state): (u32, (bool, uN[N], uN[N])) in u32:0..GROUPS {
            let (seen_group, zero_groups, group_count) = state;
            let group_start = N - (group + u32:1) * GROUP;
            let group_has_one = for (i, accum): (u32, u1) in u32:0..GROUP {
                accum | x[(group_start + i)+:u1]
            }(u1:0);
            if seen_group {
                (seen_group, zero_groups, group_count)
            } else if group_has_one == u1:0 {
                (false, zero_groups + uN[N]:1, group_count)
            } else {
                let (_, count) = for (i, inner): (u32, (bool, uN[N])) in u32:0..GROUP {
                    let (seen_one, count) = inner;
                    let bit_index = group_start + GROUP - u32:1 - i;
                    let bit_set = bit_scans::bit_is_set(x, bit_index);
                    let next_count = if seen_one || bit_set { count } else { count + uN[N]:1 };
                    (seen_one || bit_set, next_count)
                }((false, uN[N]:0));
                (true, zero_groups, count)
            }
        }((false, uN[N]:0, uN[N]:0));
    zero_groups * (GROUP as uN[N]) + group_count
}

#[quickcheck]
fn qc_clz_u8(x: u8) -> bool {
    let expected = clz(x);
    clz_linear(x) == expected && clz_prefix_mask(x) == expected && clz_grouped(x) == expected &&
    clz_grouped<u32:8, u32:2>(x) == expected && (std::clzt(x) as u8) == expected
}

#[quickcheck]
fn qc_clz_u16(x: u16) -> bool {
    let expected = clz(x);
    clz_linear(x) == expected && clz_prefix_mask(x) == expected && clz_grouped(x) == expected &&
    clz_grouped<u32:16, u32:4>(x) == expected && (std::clzt(x) as u16) == expected
}

#[quickcheck]
fn qc_clz_u32(x: u32) -> bool {
    let expected = clz(x);
    clz_linear(x) == expected && clz_prefix_mask(x) == expected && clz_grouped(x) == expected &&
    clz_grouped<u32:32, u32:8>(x) == expected && (std::clzt(x) as u32) == expected
}
