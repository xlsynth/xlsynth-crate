// SPDX-License-Identifier: Apache-2.0

const WIDTH = u32:7;

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

struct ParametricValuesResult {
    expr_box: ExprBox8,
    negative: NegativeTag,
    huge: HugeTag,
}

pub fn exercise_parametric_values(
    expr_box: ExprBox8,
    negative: NegativeTag,
    huge: HugeTag,
) -> ParametricValuesResult {
    ParametricValuesResult {
        expr_box,
        negative,
        huge,
    }
}
