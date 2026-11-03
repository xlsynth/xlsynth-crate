// SPDX-License-Identifier: Apache-2.0

pub fn main(x: sN[65]) -> sN[65] {
    for (i, acc): (sN[65], sN[65]) in sN[65]:-1..sN[65]:0 { acc + i }(x)
}

#[test]
fn minus_one() {
    assert_eq(main(sN[65]:0), sN[65]:-1);
}
