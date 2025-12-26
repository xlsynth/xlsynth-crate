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

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddCornerTag {
    LhsIsZero = 0,
    RhsIsZero = 1,
}

impl TryFrom<u8> for AddCornerTag {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(AddCornerTag::LhsIsZero),
            1 => Ok(AddCornerTag::RhsIsZero),
            _ => Err(()),
        }
    }
}

impl From<AddCornerTag> for u8 {
    fn from(value: AddCornerTag) -> Self {
        value as u8
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NegCornerTag {
    OperandIsMinSigned = 0,
    OperandMsbIsOne = 1,
}

impl TryFrom<u8> for NegCornerTag {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(NegCornerTag::OperandIsMinSigned),
            1 => Ok(NegCornerTag::OperandMsbIsOne),
            _ => Err(()),
        }
    }
}

impl From<NegCornerTag> for u8 {
    fn from(value: NegCornerTag) -> Self {
        value as u8
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShiftCornerTag {
    AmtIsZero = 0,
    AmtLtWidth = 1,
    AmtGeWidth = 2,
}

impl TryFrom<u8> for ShiftCornerTag {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ShiftCornerTag::AmtIsZero),
            1 => Ok(ShiftCornerTag::AmtLtWidth),
            2 => Ok(ShiftCornerTag::AmtGeWidth),
            _ => Err(()),
        }
    }
}

impl From<ShiftCornerTag> for u8 {
    fn from(value: ShiftCornerTag) -> Self {
        value as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CornerTag {
    Add(AddCornerTag),
    Neg(NegCornerTag),
    Shift(ShiftCornerTag),
}

pub fn corner_tag_from_kind_and_u8(kind: CornerKind, tag: u8) -> Option<CornerTag> {
    match kind {
        CornerKind::Add => AddCornerTag::try_from(tag).ok().map(CornerTag::Add),
        CornerKind::Neg => NegCornerTag::try_from(tag).ok().map(CornerTag::Neg),
        CornerKind::Shift => ShiftCornerTag::try_from(tag).ok().map(CornerTag::Shift),
        _ => None,
    }
}

pub fn corner_tag_to_u8(tag: CornerTag) -> u8 {
    match tag {
        CornerTag::Add(t) => t.into(),
        CornerTag::Neg(t) => t.into(),
        CornerTag::Shift(t) => t.into(),
    }
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
