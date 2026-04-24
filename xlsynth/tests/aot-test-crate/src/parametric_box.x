// SPDX-License-Identifier: Apache-2.0

struct Box<N: u32> {
    value: bits[N],
}

type Box8 = Box<u32:8>;

pub fn echo_box(x: Box8) -> Box8 {
    x
}
