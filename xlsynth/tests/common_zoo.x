// SPDX-License-Identifier: Apache-2.0

//! Contains sample types to be imported into the "structure zoo".

import std;

pub enum TransactionType : u1 {
    READ = 0,
    WRITE = 1,
}

pub const VALUES_TO_HOLD = u32:255;
pub type MyU8 = bits[std::clog2(VALUES_TO_HOLD)];

/// A sample struct type.
pub struct Point {
    x: u16,
    y: u32,
}