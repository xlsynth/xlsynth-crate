// SPDX-License-Identifier: Apache-2.0

//! Runtime matches that unpack both a struct and an array of sums.

use crate::ast::BitsType;
use crate::compound::resize_unsigned;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue};

pub(crate) const DECLARATION: &str = r#"
enum FuzzParcel<A: u32, B: u32> {
    Skip,
    Data { frame: FuzzPair<B>, items: FuzzSum<A, B>[2] },
}
"#;

/// Matches an outer enum, projects a bound record field, and matches an
/// independently selected element of the bound array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggregateMatch {
    word_width: usize,
    frame_width: usize,
    second_header: bool,
}

impl AggregateMatch {
    /// Chooses independent widths for the inner enum word and record fields.
    pub fn new(widths: [usize; 2], second_header: bool) -> Result<Self, String> {
        let [word_width, frame_width] = widths;
        if widths.iter().any(|width| !(1..=129).contains(width)) {
            return Err("aggregate match field widths must be in 1..=129".into());
        }
        Ok(Self {
            word_width,
            frame_width,
            second_header,
        })
    }

    /// Builds the alternatives from existing lane values and consumes both
    /// whole-field bindings inside the outer match arm.
    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let [x, y, selector] = args;
        let a = self.word_width;
        let b = self.frame_width;
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let result = ty.to_dslx();
        let header_field = if self.second_header {
            "second"
        } else {
            "first"
        };
        format!(
            r#"{{
        let tags = ({selector} as {unsigned}) as u3;
        let flip = tags[1+:u1];
        let left = if flip {{ FuzzSum<u32:{a}, u32:{b}>::Word(({x} as {unsigned}) as uN[{a}]) }}
                   else {{ FuzzSum<u32:{a}, u32:{b}>::Empty }};
        let right = if flip {{ FuzzSum<u32:{a}, u32:{b}>::Empty }}
                    else {{ FuzzSum<u32:{a}, u32:{b}>::Word(({y} as {unsigned}) as uN[{a}]) }};
        let frame = FuzzPair {{ first: ({y} as {unsigned}) as uN[{b}], second: ({x} as {unsigned}) as uN[{b}] }};
        let parcel = if tags[0+:u1] {{
            FuzzParcel<u32:{a}, u32:{b}>::Data {{ items: [left, right], frame }}
        }} else {{ FuzzParcel<u32:{a}, u32:{b}>::Skip }};
        match parcel {{
            FuzzParcel::Data {{ frame: h, items: entries }} => {{
                let value = match entries[tags[2+:u1]] {{
                    FuzzSum::Word(word) => word,
                    _ => uN[{a}]:0,
                }};
                (((h.{header_field} as {unsigned}) + (value as {unsigned})) as {result})
            }},
            _ => {y},
        }}
    }}"#
        )
    }

    /// Selects scalar inputs directly. A word is present exactly when the
    /// element index and the constructor's flip bit differ.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        width: usize,
    ) -> Result<BValue, BuilderError> {
        let [x, y, selector] = args;
        let tags = resize_unsigned(builder, selector, 3)?;
        let active = builder.bit_slice(tags, 0, 1)?;
        let flip = builder.bit_slice(tags, 1, 1)?;
        let index = builder.bit_slice(tags, 2, 1)?;
        let present = builder.xor(flip, index)?;
        let source = builder.select(index, &[x, y], None)?;
        let word = resize_unsigned(builder, source, self.word_width)?;
        let word = resize_unsigned(builder, word, width)?;
        let zero = builder.literal(IrValue::from_bits(&IrBits::zero(width)))?;
        let value = builder.select(present, &[zero, word], None)?;
        let header = if self.second_header { x } else { y };
        let header = resize_unsigned(builder, header, self.frame_width)?;
        let header = resize_unsigned(builder, header, width)?;
        let sum = builder.add(header, value)?;
        builder.select(active, &[y, sum], None)
    }
}
