// SPDX-License-Identifier: Apache-2.0

import std;

pub const IMPORTED_BIAS = u32:7;

pub type ImportedByte = u8;

pub fn imported_popcount(x: u32) -> u32 { std::popcount(x) }
