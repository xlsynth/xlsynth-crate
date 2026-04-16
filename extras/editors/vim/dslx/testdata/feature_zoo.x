// SPDX-License-Identifier: Apache-2.0

#![feature(use_syntax)]

use feature_zoo_imported::IMPORTED_BIAS;

pub const WIDTH = u32:32;

pub type Word = uN[WIDTH];

pub enum Status : u2 {
    IDLE = 0,
    BUSY = 1,
    DONE = 2,
}

pub struct Packet<N: u32> {
    tag: u8,
    payload: bits[N],
}

impl Packet<N> {
    const TOTAL_WIDTH = N + u32:8;

    fn zero() -> Self { Packet<N> { tag: u8:0, payload: zero!<bits[N]>() } }

    fn bumped_tag(self: Self) -> u8 { self.tag + u8:1 }
}

fn fold_and_cast(x: u8) -> u32 {
    let sum = for (i, accum): (u8, u8) in u8:0..u8:4 {
        accum + i
    }(x);
    if sum > u8:0 { sum as u32 } else { u32:0 }
}

fn unrolled_sum() -> u32 {
    unroll_for! (i, accum): (u32, u32) in u32:0..u32:4 {
        accum + i
    }(u32:0)
}

pub fn normalize<N: u32>(x: uN[N], flag: bool) -> uN[N] {
    // Keep this file broad enough to exercise highlighting.
    let zero = zero!<uN[N]>();
    let one = uN[N]:1;
    let neg = s32:-42;
    let hex = u32:0xCAFE_F00D;
    let bin = u8:0b1010_0101;
    let ch = 'x';
    let newline = '\n';
    let imported = u8:3;
    let _text = "tag\nvalue";
    let _braces = "format string with {braces}";
    let status = Status::BUSY;

    trace_fmt!(
        "x={}, hex={:#x}, bin={:#b}, status={}, neg={}, ch={}, newline={}, imported={}, bias={}",
        x, hex, bin, status, neg, ch, newline, imported, IMPORTED_BIAS);
    assert!(flag || x == zero, "flag_or_zero");
    const_assert!(WIDTH == u32:32);

    match flag {
        true => x + one,
        false => zero,
    }
}

proc Counter {
    output: chan<u8> out;

    config(output: chan<u8> out) { (output,) }

    init { u8:0 }

    next(state: u8) {
        let tok = send(join(), output, state);
        state + u8:1
    }
}

#[test_proc]
proc CounterHarness {
    input: chan<u8> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        let (s, r) = chan<u8>("counter");
        spawn Counter(s);
        (r, terminator)
    }

    init { u8:0 }

    next(iter: u8) {
        let (tok, value) = recv(join(), input);
        let tok = send_if(tok, terminator, value == u8:2, true);
        iter + u8:1
    }
}

#[test]
fn packet_zero_test() {
    let p = Packet<u32:8>::zero();
    assert_eq(p.tag, u8:0);
    assert_eq(p.bumped_tag(), u8:1);
    assert_eq(fold_and_cast(u8:1), u32:7);
    assert_eq(unrolled_sum(), u32:6);
}

#[quickcheck]
fn normalize_quickcheck(x: u8) -> bool {
    normalize<u32:8>(x, true) == x + u8:1
}

#[fuzz_test(domains=`u32:0..1, [u32:0, u32:10]`)]
fn normalize_fuzz_sample(x: u32, y: u32) -> bool {
    x + y == y + x
}
