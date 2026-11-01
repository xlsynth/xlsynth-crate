// SPDX-License-Identifier: Apache-2.0

pub fn muladd_cycle0(x: u8, y: u8, z: u8) -> (u8, u8) { (x * y, z) }

pub fn muladd_cycle1(partials: (u8, u8)) -> u8 {
    let (product, z) = partials;
    product + z
}
