// SPDX-License-Identifier: Apache-2.0

//! Bits-valued fragments for range indexing, iteration, and match expressions.

use crate::ast::{BitsType, Width};
use crate::compound::resize_unsigned;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue};

/// A small DSLX range with independently typed endpoints. The host integers
/// bound generation and iteration; PIR elements have the full induction width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeValue {
    signed: bool,
    width: usize,
    start: i32,
    length: u8,
    inclusive: bool,
}

impl RangeValue {
    /// Creates the nonempty half-open byte range used by the checkpoint
    /// grammar.
    pub fn new(start: u8, length: u8) -> Result<Self, String> {
        if length == 0 {
            return Err("indexed ranges must be nonempty".to_string());
        }
        Self::typed(false, 8, i16::from(start), length, false)
    }

    /// Requires both written endpoints to fit and at most eight iterations.
    pub fn typed(
        signed: bool,
        width: usize,
        start: i16,
        length: u8,
        inclusive: bool,
    ) -> Result<Self, String> {
        if !(1..=129).contains(&width) || length > 8 || (inclusive && length == 0) {
            return Err("range requires a width in 1..=129, at most eight elements, and a nonempty inclusive interval".to_string());
        }
        let range = Self {
            signed,
            width,
            start: i32::from(start),
            length,
            inclusive,
        };
        for endpoint in [range.start, range.end()] {
            range.literal(endpoint).map_err(|error| error.to_string())?;
        }
        Ok(range)
    }

    /// Old loop descriptors are validated to have starts and trips in 0..=4.
    pub(crate) fn unsigned_loop(start: usize, trips: usize) -> Self {
        Self {
            signed: false,
            width: 32,
            start: start as i32,
            length: trips as u8,
            inclusive: false,
        }
    }

    pub(crate) fn len(self) -> u8 {
        self.length
    }

    fn end(self) -> i32 {
        self.start + i32::from(self.length) - i32::from(self.inclusive)
    }

    fn element_type(self) -> String {
        if !self.signed && matches!(self.width, 8 | 32) {
            // Keep the checkpoint grammars' source spelling.
            format!("u{}", self.width)
        } else {
            BitsType {
                signed: self.signed,
                width: Width::Concrete(self.width),
            }
            .to_dslx()
        }
    }

    fn iterable(self) -> String {
        let ty = self.element_type();
        let bound = if self.inclusive { "..=" } else { ".." };
        format!("{ty}:{}{bound}{ty}:{}", self.start, self.end())
    }

    /// Makes a two's-complement element before applying the source type's
    /// widening rule, including for elements wider than a host word.
    fn literal(self, value: i32) -> Result<IrValue, BuilderError> {
        let literal = if self.signed {
            IrValue::make_sbits(self.width, i64::from(value))
        } else {
            let unsigned = u64::try_from(value).map_err(|_| {
                BuilderError::InvalidOperation("unsigned range endpoint is negative".to_string())
            })?;
            IrValue::make_ubits(self.width, unsigned)
        };
        literal.map_err(|error| BuilderError::InvalidOperation(error.to_string()))
    }

    fn cast_element(
        self,
        builder: &mut FnBuilder,
        value: BValue,
        width: usize,
    ) -> Result<BValue, BuilderError> {
        if self.signed && self.width < width {
            builder.sign_extend(value, width)
        } else {
            resize_unsigned(builder, value, width)
        }
    }

    pub(crate) fn to_dslx(self, ty: BitsType, [index, mix, _]: [&str; 3]) -> String {
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let result = ty.to_dslx();
        let iterable = self.iterable();
        let length = self.length;
        format!(
            "{{ let items = {iterable}; let i = (({index} as {unsigned}) as u8) % u8:{length}; (items[i] as {result}) ^ {mix} }}"
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
        let length = builder.literal(IrValue::from_bits(&IrBits::from_lsb_fn(8, |bit| {
            self.length & (1 << bit) != 0
        })))?;
        let offset = builder.umod(index, length)?;
        let offset = resize_unsigned(builder, offset, self.width)?;
        let start = builder.literal(self.literal(self.start)?)?;
        let element = builder.add(start, offset)?;
        let element = self.cast_element(builder, element, width)?;
        builder.xor(element, mix)
    }

    pub(crate) fn loop_dslx(self, constant: bool, ty: BitsType, [x, y, _]: [&str; 3]) -> String {
        let qualifier = if constant { "const " } else { "" };
        let induction = self.element_type();
        let result = ty.to_dslx();
        let iterable = self.iterable();
        format!(
            "{qualifier}for (i, acc): ({induction}, {result}) in {iterable} {{ (acc + {y}) ^ (i as {result}) }}({x})"
        )
    }

    /// Unrolls the mathematical interval and materializes every induction value
    /// at its declared width, independently of the converter's counted loop.
    pub(crate) fn emit_loop_ir(
        self,
        builder: &mut FnBuilder,
        [x, y, _]: [BValue; 3],
        width: usize,
    ) -> Result<BValue, BuilderError> {
        let mut acc = x;
        for iteration in 0..self.length {
            let i = builder.literal(self.literal(self.start + i32::from(iteration))?)?;
            let i = self.cast_element(builder, i, width)?;
            let sum = builder.add(acc, y)?;
            acc = builder.xor(sum, i)?;
        }
        Ok(acc)
    }
}

/// Two ordered, overlapping ranges in an ordinary match on a runtime selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeMatch {
    arms: [RangeValue; 2],
}

impl RangeMatch {
    /// Requires a common selector type and reachable values for all three arms.
    pub fn new(arms: [RangeValue; 2]) -> Result<Self, String> {
        let [first, second] = arms;
        if (first.signed, first.width) != (second.signed, second.width) {
            return Err("match ranges must have the same selector type".to_string());
        }
        let first_limit = first.start + i32::from(first.length);
        let second_limit = second.start + i32::from(second.length);
        let overlap = first_limit.min(second_limit) - first.start.max(second.start);
        if overlap <= 0 || overlap == i32::from(second.length) {
            return Err("match ranges must overlap and leave the second arm reachable".to_string());
        }
        let covered = i32::from(first.length) + i32::from(second.length) - overlap;
        if 1u32.checked_shl(first.width as u32) == Some(covered as u32) {
            return Err("match ranges must leave the wildcard arm reachable".to_string());
        }
        Ok(Self { arms })
    }

    pub(crate) fn to_dslx(self, [x, y, z]: [&str; 3]) -> String {
        let [first, second] = self.arms;
        let selector = first.element_type();
        let first = first.iterable();
        let second = second.iterable();
        format!("match ({x} as {selector}) {{ {first} => {y}, {second} => !{z}, _ => {x}, }}")
    }

    /// Tests membership by enumerating exact bit patterns instead of comparing
    /// interval bounds. Selecting the earlier arm last preserves DSLX priority.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        [x, y, z]: [BValue; 3],
        width: usize,
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        let selector_width = self.arms[0].width;
        let selector = if signed && width < selector_width {
            builder.sign_extend(x, selector_width)?
        } else {
            resize_unsigned(builder, x, selector_width)?
        };
        let second_result = builder.not(z)?;
        let mut result = x;
        for (range, on_match) in self.arms.into_iter().zip([y, second_result]).rev() {
            let first = builder.literal(range.literal(range.start)?)?;
            let mut matches = builder.eq(selector, first)?;
            for offset in 1..range.length {
                let value = builder.literal(range.literal(range.start + i32::from(offset))?)?;
                let equal = builder.eq(selector, value)?;
                matches = builder.or(matches, equal)?;
            }
            result = builder.select(matches, &[result, on_match], None)?;
        }
        Ok(result)
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
