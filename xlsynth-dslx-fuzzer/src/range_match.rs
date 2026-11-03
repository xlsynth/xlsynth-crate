// SPDX-License-Identifier: Apache-2.0

//! Bits-valued fragments for range indexing and specialization-time matching.

use crate::ast::{BitsType, Width};
use crate::compound::resize_unsigned;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue};

/// A bounded half-open byte range indexed by a generated lane value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeValue {
    start: u8,
    length: u8,
}

impl RangeValue {
    /// Requires a nonempty range of at most eight bytes with representable
    /// bounds.
    pub fn new(start: u8, length: u8) -> Result<Self, String> {
        if !(1..=8).contains(&length) || start.checked_add(length).is_none() {
            return Err("range must have 1..=8 elements and a u8 exclusive end".to_string());
        }
        Ok(Self { start, length })
    }

    pub(crate) fn to_dslx(self, ty: BitsType, [index, mix, _]: [&str; 3]) -> String {
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let result = ty.to_dslx();
        let Self { start, length } = self;
        let end = start + length;
        format!(
            "{{ let items = u8:{start}..u8:{end}; let i = (({index} as {unsigned}) as u8) % u8:{length}; (items[i] as {result}) ^ {mix} }}"
        )
    }

    /// Computes the selected element arithmetically, without constructing an IR
    /// array.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        [index, mix, _]: [BValue; 3],
        width: usize,
    ) -> Result<BValue, BuilderError> {
        let index = resize_unsigned(builder, index, 8)?;
        let byte =
            |value: u8| IrValue::from_bits(&IrBits::from_lsb_fn(8, |i| value & (1 << i) != 0));
        let length = builder.literal(byte(self.length))?;
        let offset = builder.umod(index, length)?;
        let start = builder.literal(byte(self.start))?;
        let element = builder.add(start, offset)?;
        let element = resize_unsigned(builder, element, width)?;
        builder.xor(element, mix)
    }
}

/// Selects a runtime expression using the low bits of the specialized lane
/// width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstMatch {
    offset: u8,
}

impl ConstMatch {
    /// Checks the two-bit constant used to vary which arm is selected.
    pub fn new(offset: u8) -> Result<Self, String> {
        if offset > 3 {
            return Err("const match offset must fit in u2".to_string());
        }
        Ok(Self { offset })
    }

    pub(crate) fn to_dslx(self, ty: BitsType, [x, y, z]: [&str; 3]) -> String {
        let width = match ty.width {
            Width::Concrete(width) => format!("u32:{width}"),
            Width::Parametric => "N".to_string(),
        };
        // v0.61.0 loses type information for the cast's operand when the cast
        // appears directly in the const-match scrutinee.
        format!(
            "{{ const MATCH_KEY = ({width} as u2) ^ u2:{}; const match MATCH_KEY {{ u2:0 => {x} + {y}, u2:1 => {y} ^ {z}, _ => {z} - {x}, }} }}",
            self.offset
        )
    }

    /// Specializes the width metadata before building the chosen arm's IR.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        [x, y, z]: [BValue; 3],
        width: usize,
    ) -> Result<BValue, BuilderError> {
        match (width & 3) ^ usize::from(self.offset) {
            0 => builder.add(x, y),
            1 => builder.xor(y, z),
            _ => builder.sub(z, x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ConstMatch, RangeValue};

    #[test]
    fn range_bounds() {
        assert!(RangeValue::new(247, 8).is_ok());
        assert!(RangeValue::new(254, 1).is_ok());
        assert!(RangeValue::new(248, 8).is_err());
        assert!(RangeValue::new(255, 1).is_err());
        assert!(RangeValue::new(0, 0).is_err());
        assert!(RangeValue::new(0, 9).is_err());
    }

    #[test]
    fn const_match_offset_must_fit_selector() {
        assert!(ConstMatch::new(3).is_ok());
        assert!(ConstMatch::new(4).is_err());
    }
}
