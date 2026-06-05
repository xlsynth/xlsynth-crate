// SPDX-License-Identifier: Apache-2.0

const WIDTH = u32:7;

struct Box<N: u32> {
    value: bits[N],
}

type Box8 = Box<u32:8>;
type Box16 = Box<u32:16>;
type Box8Array4 = Box8[4];

struct Matrix<R: u32, C: u32> {
    rows: u8[C][R],
}

type Matrix2x3 = Matrix<u32:2, u32:3>;

struct ArrayBox<N: u32> {
    items: u8[N],
}

type ArrayBox4 = ArrayBox<u32:4>;

struct ExprBox<N: u32> {
    value: bits[N],
}

type ExprBox8 = ExprBox<{WIDTH + u32:1}>;

struct SignedTag<S: s32> {
    payload: u8,
}

type NegativeTag = SignedTag<s32:-3>;

struct WideTag<W: uN[128]> {
    payload: u8,
}

type HugeTag = WideTag<uN[128]:18446744073709551616>;

struct WidePair {
    unsigned_value: uN[65],
    signed_value: sN[65],
}

struct ParametricFormsResult {
    box8: Box8,
    box16: Box16,
    matrix: Matrix2x3,
    array_box: ArrayBox4,
    box_array: Box8Array4,
    expr_box: ExprBox8,
    negative: NegativeTag,
    huge: HugeTag,
    wide_pair: WidePair,
}

pub fn exercise_parametric_forms(
    box8: Box8,
    box16: Box16,
    matrix: Matrix2x3,
    array_box: ArrayBox4,
    box_array: Box8Array4,
    expr_box: ExprBox8,
    negative: NegativeTag,
    huge: HugeTag,
    wide_pair: WidePair,
) -> ParametricFormsResult {
    ParametricFormsResult {
        box8,
        box16,
        matrix,
        array_box,
        box_array,
        expr_box,
        negative,
        huge,
        wide_pair,
    }
}
