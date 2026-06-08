// SPDX-License-Identifier: Apache-2.0

pub fn exercise_events(x: u8, y: u8, ok: bool, emit: bool) -> u8 {
    cover!("covered", emit);
    cover!("accepted", ok);
    assert!(ok, "bad_condition");
    if emit {
        trace_fmt!("x={} y={:x}", x, y);
    } else {
        ()
    };
    if ok {
        trace_fmt!("accepted x={}", x);
    } else {
        ()
    };
    x + y
}
