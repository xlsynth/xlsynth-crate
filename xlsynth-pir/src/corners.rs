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
    SignExt(SignExtCornerTag),
    DynamicBitSlice(DynamicBitSliceCornerTag),
    ArrayIndex(ArrayIndexCornerTag),
    Shra(ShraCornerTag),
    CompareDistance(CompareDistanceCornerTag),
}

pub fn corner_tag_from_kind_and_u8(kind: CornerKind, tag: u8) -> Option<CornerTag> {
    match kind {
        CornerKind::Add => AddCornerTag::try_from(tag).ok().map(CornerTag::Add),
        CornerKind::Neg => NegCornerTag::try_from(tag).ok().map(CornerTag::Neg),
        CornerKind::Shift => ShiftCornerTag::try_from(tag).ok().map(CornerTag::Shift),
        CornerKind::SignExt => SignExtCornerTag::try_from(tag).ok().map(CornerTag::SignExt),
        CornerKind::DynamicBitSlice => DynamicBitSliceCornerTag::try_from(tag)
            .ok()
            .map(CornerTag::DynamicBitSlice),
        CornerKind::ArrayIndex => ArrayIndexCornerTag::try_from(tag)
            .ok()
            .map(CornerTag::ArrayIndex),
        CornerKind::Shra => ShraCornerTag::try_from(tag).ok().map(CornerTag::Shra),
        CornerKind::CompareDistance => CompareDistanceCornerTag::try_from(tag)
            .ok()
            .map(CornerTag::CompareDistance),
    }
}

pub fn corner_tag_to_u8(tag: CornerTag) -> u8 {
    match tag {
        CornerTag::Add(t) => t.into(),
        CornerTag::Neg(t) => t.into(),
        CornerTag::Shift(t) => t.into(),
        CornerTag::SignExt(t) => t.into(),
        CornerTag::DynamicBitSlice(t) => t.into(),
        CornerTag::ArrayIndex(t) => t.into(),
        CornerTag::Shra(t) => t.into(),
        CornerTag::CompareDistance(t) => t.into(),
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignExtCornerTag {
    MsbIsZero = 0,
}

impl TryFrom<u8> for SignExtCornerTag {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SignExtCornerTag::MsbIsZero),
            _ => Err(()),
        }
    }
}

impl From<SignExtCornerTag> for u8 {
    fn from(value: SignExtCornerTag) -> Self {
        value as u8
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DynamicBitSliceCornerTag {
    InBounds = 0,
    OutOfBounds = 1,
}

impl TryFrom<u8> for DynamicBitSliceCornerTag {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(DynamicBitSliceCornerTag::InBounds),
            1 => Ok(DynamicBitSliceCornerTag::OutOfBounds),
            _ => Err(()),
        }
    }
}

impl From<DynamicBitSliceCornerTag> for u8 {
    fn from(value: DynamicBitSliceCornerTag) -> Self {
        value as u8
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrayIndexCornerTag {
    InBounds = 0,
    Clamped = 1,
}

impl TryFrom<u8> for ArrayIndexCornerTag {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ArrayIndexCornerTag::InBounds),
            1 => Ok(ArrayIndexCornerTag::Clamped),
            _ => Err(()),
        }
    }
}

impl From<ArrayIndexCornerTag> for u8 {
    fn from(value: ArrayIndexCornerTag) -> Self {
        value as u8
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShraCornerTag {
    Msb0AmtLt = 0,
    Msb0AmtGe = 1,
    Msb1AmtLt = 2,
    Msb1AmtGe = 3,
}

impl TryFrom<u8> for ShraCornerTag {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ShraCornerTag::Msb0AmtLt),
            1 => Ok(ShraCornerTag::Msb0AmtGe),
            2 => Ok(ShraCornerTag::Msb1AmtLt),
            3 => Ok(ShraCornerTag::Msb1AmtGe),
            _ => Err(()),
        }
    }
}

impl From<ShraCornerTag> for u8 {
    fn from(value: ShraCornerTag) -> Self {
        value as u8
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareDistanceCornerTag {
    XorPopcount0 = 0,
    XorPopcount1 = 1,
    XorPopcount2 = 2,
    XorPopcount3 = 3,
    XorPopcount4 = 4,
    XorPopcount5To8 = 5,
    XorPopcount9To16 = 6,
    XorPopcount17Plus = 7,
}

impl TryFrom<u8> for CompareDistanceCornerTag {
    type Error = ();
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(CompareDistanceCornerTag::XorPopcount0),
            1 => Ok(CompareDistanceCornerTag::XorPopcount1),
            2 => Ok(CompareDistanceCornerTag::XorPopcount2),
            3 => Ok(CompareDistanceCornerTag::XorPopcount3),
            4 => Ok(CompareDistanceCornerTag::XorPopcount4),
            5 => Ok(CompareDistanceCornerTag::XorPopcount5To8),
            6 => Ok(CompareDistanceCornerTag::XorPopcount9To16),
            7 => Ok(CompareDistanceCornerTag::XorPopcount17Plus),
            _ => Err(()),
        }
    }
}

impl From<CompareDistanceCornerTag> for u8 {
    fn from(value: CompareDistanceCornerTag) -> Self {
        value as u8
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
