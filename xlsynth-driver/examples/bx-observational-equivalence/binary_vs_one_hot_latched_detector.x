// SPDX-License-Identifier: Apache-2.0

type FState = u2;
type GState = u4;

pub fn f(i: u1, s: FState) -> (u1, FState) {
    let idle = u2:0;
    let armed = u2:1;
    let latched = u2:2;
    let o: u1 = if s == latched { u1:1 } else { u1:0 };
    let next_s: FState = if s == idle {
        if i == u1:1 { armed } else { idle }
    } else if s == armed {
        if i == u1:1 { latched } else { idle }
    } else if s == latched {
        if i == u1:1 { latched } else { idle }
    } else {
        idle
    };
    (o, next_s)
}

pub fn g(i: u1, s: GState) -> (u1, GState) {
    let idle = u4:0b0001;
    let armed = u4:0b0010;
    let latched = u4:0b0100;
    let o: u1 = if s == latched { u1:1 } else { u1:0 };
    let next_s: GState = if s == idle {
        if i == u1:1 { armed } else { idle }
    } else if s == armed {
        if i == u1:1 { latched } else { idle }
    } else if s == latched {
        if i == u1:1 { latched } else { idle }
    } else {
        idle
    };
    (o, next_s)
}
