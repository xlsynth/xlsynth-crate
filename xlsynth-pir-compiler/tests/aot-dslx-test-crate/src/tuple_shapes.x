// SPDX-License-Identifier: Apache-2.0

pub fn exercise_tuple_shapes(pair: (u8, u16), increment: u8) -> (u8, u16) {
    let (low, high) = pair;
    (low + increment, high + increment as u16)
}
