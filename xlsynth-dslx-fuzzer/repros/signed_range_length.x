// SPDX-License-Identifier: Apache-2.0

pub fn main(i: u1) -> s2 {
    (s2:-2..s2:0)[i]
}

#[test]
fn endpoints() {
    assert_eq(main(u1:0), s2:-2);
    assert_eq(main(u1:1), s2:-1);
}
