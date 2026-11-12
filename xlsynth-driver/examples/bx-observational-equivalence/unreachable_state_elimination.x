// SPDX-License-Identifier: Apache-2.0

type FState = u2;
type GState = u1;

pub fn f(i: u1, s: FState) -> (u1, FState) {
    let even = u2:0;
    let odd = u2:1;
    let dead = u2:2;
    let o: u1 = if s == odd { u1:1 } else { u1:0 };
    let next_s: FState = if s == even {
        if i == u1:1 { odd } else { even }
    } else if s == odd {
        if i == u1:1 { even } else { odd }
    } else {
        dead
    };
    (o, next_s)
}

pub fn g(i: u1, s: GState) -> (u1, GState) {
    let o: u1 = s;
    let next_s: GState = if i == u1:1 { !s } else { s };
    (o, next_s)
}
