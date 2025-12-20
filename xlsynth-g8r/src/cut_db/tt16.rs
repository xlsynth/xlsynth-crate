// SPDX-License-Identifier: Apache-2.0

//! 4-input single-output Boolean function truth tables.
//!
//! We represent a Boolean function `f(a,b,c,d) -> o` as a `u16` where bit `i`
//! corresponds to the output value on the input assignment encoded by `i`:
//! - `a = (i >> 0) & 1`
//! - `b = (i >> 1) & 1`
//! - `c = (i >> 2) & 1`
//! - `d = (i >> 3) & 1`
//!
//! That is, `a` is the least-significant selector bit and toggles fastest.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct TruthTable16(pub u16);

impl TruthTable16 {
    pub const fn const0() -> Self {
        Self(0x0000)
    }

    pub const fn const1() -> Self {
        Self(0xFFFF)
    }

    /// Returns the truth table for the given input variable index.
    ///
    /// Variable index mapping:
    /// - 0: a (LSB of assignment index)
    /// - 1: b
    /// - 2: c
    /// - 3: d (MSB of assignment index)
    pub const fn var(index: usize) -> Self {
        match index {
            0 => Self(0xAAAA),
            1 => Self(0xCCCC),
            2 => Self(0xF0F0),
            3 => Self(0xFF00),
            _ => panic!("TruthTable16::var index out of range (expected 0..=3)"),
        }
    }

    #[inline]
    pub const fn not(self) -> Self {
        Self(!self.0)
    }

    #[inline]
    pub const fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline]
    pub fn get_bit(self, assignment_index: u8) -> bool {
        debug_assert!(assignment_index < 16);
        ((self.0 >> assignment_index) & 1) != 0
    }

    #[inline]
    pub fn set_bit(&mut self, assignment_index: u8, value: bool) {
        debug_assert!(assignment_index < 16);
        let mask = 1u16 << assignment_index;
        if value {
            self.0 |= mask;
        } else {
            self.0 &= !mask;
        }
    }
}

#[inline]
pub fn decode_assignment(i: u8) -> [bool; 4] {
    debug_assert!(i < 16);
    [
        (i & 0b0001) != 0,
        (i & 0b0010) != 0,
        (i & 0b0100) != 0,
        (i & 0b1000) != 0,
    ]
}

#[inline]
pub fn encode_assignment(bits: [bool; 4]) -> u8 {
    (bits[0] as u8) | ((bits[1] as u8) << 1) | ((bits[2] as u8) << 2) | ((bits[3] as u8) << 3)
}
