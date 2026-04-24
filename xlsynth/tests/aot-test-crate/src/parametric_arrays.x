// SPDX-License-Identifier: Apache-2.0

struct Box<N: u32> {
    value: bits[N],
}

struct ArrayBox<N: u32> {
    items: u8[N],
}

type ArrayBox4 = ArrayBox<u32:4>;
type Box8Array4 = Box<u32:8>[4];

struct OuterBox {
    inner: Box<u32:8>,
    wider: Box<u32:16>,
}

struct ParametricArraysResult {
    array_box: ArrayBox4,
    box_array: Box8Array4,
    outer: OuterBox,
}

pub fn exercise_parametric_arrays(
    array_box: ArrayBox4,
    box_array: Box8Array4,
    outer: OuterBox,
) -> ParametricArraysResult {
    ParametricArraysResult {
        array_box,
        box_array,
        outer,
    }
}
