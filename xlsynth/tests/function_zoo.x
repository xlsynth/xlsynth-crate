// SPDX-License-Identifier: Apache-2.0

fn sum_elements<N: u32>(elements: u32[N]) -> u32 {
    let result: u32 = for (i, accum) in u32:0..array_size(elements) {
        accum + elements[i]
    }(u32:0);
    result
}

pub fn sum_elements_2(elements: u32[2]) -> u32 {
    sum_elements(elements)
}

pub fn make_u32x2(a: u32, b: u32) -> u32[2] {
    [a, b]
}

/// Creates a two-dimensional array having three rows of `u32[2]` values.
fn make_u32_3x2(a: u32[2], b: u32[2], c: u32[2]) -> u32[2][3] {
    [a, b, c]
}
