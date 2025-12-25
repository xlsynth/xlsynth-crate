// SPDX-License-Identifier: Apache-2.0

use crate::ir;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CornerKind {
    Add = 0,
    Neg = 1,
    SignExt = 2,
    DynamicBitSlice = 3,
    ArrayIndex = 4,
    Shift = 5,
    Shra = 6,
    CompareDistance = 7,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureKind {
    BitSliceOob = 0,
    DynamicBitSliceOob = 1,
    BitSliceUpdateOob = 2,
    ArrayUpdateOobAssumedInBounds = 3,
    ArrayIndexOobAssumedInBounds = 4,
    AssertTriggered = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CornerEvent {
    pub node_ref: ir::NodeRef,
    pub node_text_id: usize,
    pub kind: CornerKind,
    pub tag: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FailureEvent {
    pub node_ref: ir::NodeRef,
    pub node_text_id: usize,
    pub kind: FailureKind,
    pub tag: u8,
}

pub fn bucket_xor_popcount(d: usize) -> u8 {
    match d {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5..=8 => 5,
        9..=16 => 6,
        _ => 7,
    }
}
