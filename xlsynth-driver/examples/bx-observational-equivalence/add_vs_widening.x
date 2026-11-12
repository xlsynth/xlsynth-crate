// SPDX-License-Identifier: Apache-2.0

type FState = u8;
type GState = u16;

pub fn f(i: u8, s: FState) -> (u8, FState) {
    let o = i + s;
    let next_s: FState = s + u8:1;
    (o, next_s)
}

pub fn g(i: u8, s: GState) -> (u8, GState) {
    let i_wide = i as u16;
    let sum = i_wide + s;
    let o = sum as u8;
    let next_s: GState = s + u16:1;
    (o, next_s)
}
