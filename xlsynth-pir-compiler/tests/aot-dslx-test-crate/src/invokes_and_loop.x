// SPDX-License-Identifier: Apache-2.0

fn add_one(x: u8) -> u8 {
    x + u8:1
}

fn add_index(carry: u8, index: u8) -> u8 {
    carry + index
}

pub fn exercise_invokes_and_loop(init: u8, increment: u8) -> u8 {
    let invoked = add_one(add_one(init));
    for (i, carry): (u8, u8) in u8:0..u8:3 {
        add_index(carry, i) + increment
    }(invoked)
}
