// SPDX-License-Identifier: Apache-2.0

#![feature(use_syntax)]

use std::to_unsigned;

const M1: s32 = s32:-1;

fn main() -> u32 {
    to_unsigned(M1)
}