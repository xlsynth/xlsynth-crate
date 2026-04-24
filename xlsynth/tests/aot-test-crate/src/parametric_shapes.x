// SPDX-License-Identifier: Apache-2.0

struct Box<N: u32> {
    value: bits[N],
}

type Box8 = Box<u32:8>;
type Box16 = Box<u32:16>;

struct Matrix<R: u32, C: u32> {
    rows: u8[C][R],
}

type Matrix2x3 = Matrix<u32:2, u32:3>;

struct ParametricShapesResult {
    box8: Box8,
    box16: Box16,
    matrix: Matrix2x3,
}

pub fn exercise_parametric_shapes(
    box8: Box8,
    box16: Box16,
    matrix: Matrix2x3,
) -> ParametricShapesResult {
    ParametricShapesResult { box8, box16, matrix }
}
