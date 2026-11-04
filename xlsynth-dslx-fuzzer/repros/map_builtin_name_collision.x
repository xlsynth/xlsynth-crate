// SPDX-License-Identifier: Apache-2.0

fn rev__2(x: u2) -> u2 { x }

pub fn main(x: u2) -> u2 { rev__2(x) ^ map([x], rev)[u32:0] }

#[test]
fn reverse_one() { assert_eq(main(u2:1), u2:3); }
