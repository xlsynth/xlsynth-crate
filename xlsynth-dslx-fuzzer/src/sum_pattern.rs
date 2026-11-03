// SPDX-License-Identifier: Apache-2.0

//! Runtime construction and destructuring of payload-bearing enums.

use crate::ast::BitsType;
use crate::compound::resize_unsigned;
use xlsynth_pir::{BValue, BuilderError, FnBuilder};

pub(crate) const DECLARATION: &str = r#"
enum FuzzSum<A: u32, B: u32> {
    Empty,
    Word(uN[A]),
    Pair { lo: uN[A], hi: uN[B] },
    Nested((uN[A], (bool, uN[B]))),
}
"#;

/// A sum constructor and the corresponding payload pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumVariant {
    Empty,
    Word,
    Pair,
    Nested,
}

impl SumVariant {
    pub const ALL: [Self; 4] = [Self::Empty, Self::Word, Self::Pair, Self::Nested];

    /// Constructs a variant from the local payload values.
    fn construct(self, type_name: &str) -> String {
        let constructor = match self {
            Self::Empty => "Empty",
            Self::Word => "Word(low)",
            Self::Pair => "Pair { hi: high, lo: low }",
            Self::Nested => "Nested((low, (marked, high)))",
        };
        format!("{type_name}::{constructor}")
    }

    /// Destructures the selected variant, including first-match tuple patterns.
    fn match_arms(self, ty: &str, fallback: &str, tail_when_marked: bool) -> String {
        match self {
            Self::Empty => format!("FuzzSum::Empty => {fallback},"),
            Self::Word => format!("FuzzSum::Word(word) => word as {ty},"),
            Self::Pair => format!(
                "FuzzSum::Pair {{ hi: upper, lo: lower }} => (upper as {ty}) - (lower as {ty}),"
            ),
            Self::Nested if tail_when_marked => format!(
                "FuzzSum::Nested((_, (true, tail))) => tail as {ty},
                 FuzzSum::Nested((head, ..)) => head as {ty},"
            ),
            Self::Nested => format!(
                "FuzzSum::Nested((head, (true, _))) => head as {ty},
                 FuzzSum::Nested((.., (_, tail))) => tail as {ty},"
            ),
        }
    }
}

/// Two runtime alternatives of one enum specialization, with independent
/// payload widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SumPattern {
    /// The selector's low bit chooses between these distinct variants.
    pub variants: [SumVariant; 2],
    /// Bit widths of the low and high payload fields.
    pub widths: [usize; 2],
    /// Selects which nested tuple element is bound by the literal-pattern arm.
    pub tail_when_marked: bool,
}

impl SumPattern {
    pub(crate) fn validate(self) -> Result<(), String> {
        if self.variants[0] == self.variants[1] {
            return Err("sum construction requires distinct variants".into());
        }
        if self.widths.iter().any(|width| !(1..=129).contains(width)) {
            return Err("sum payload widths must be in 1..=129".into());
        }
        Ok(())
    }

    /// Casts composed values into fields, constructs a sum and matches its
    /// payload.
    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let [selector, x, y] = args;
        let [a, b] = self.widths;
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let ty = ty.to_dslx();
        let enum_type = format!("FuzzSum<u32:{a}, u32:{b}>");
        let [off, on] = self.variants.map(|variant| variant.construct(&enum_type));
        let [off_arm, on_arm] = self
            .variants
            .map(|variant| variant.match_arms(&ty, y, self.tail_when_marked));
        format!(
            "{{
        let low = ({x} as {unsigned}) as uN[{a}];
        let high = ({y} as {unsigned}) as uN[{b}];
        let marked = {x} as bool;
        let item = if {selector} as bool {{ {on} }} else {{ {off} }};
        match item {{ {off_arm} {on_arm} _ => {y}, }}
    }}"
        )
    }

    /// Computes the selected arm directly, independent of enum packing or tag
    /// layout.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        width: usize,
    ) -> Result<BValue, BuilderError> {
        let [selector, x, y] = args;
        let [a, b] = self.widths;
        let low = resize_unsigned(builder, x, a)?;
        let high = resize_unsigned(builder, y, b)?;
        let low = resize_unsigned(builder, low, width)?;
        let high = resize_unsigned(builder, high, width)?;
        let marked = builder.bit_slice(x, 0, 1)?;
        let mut arms = Vec::with_capacity(2);
        for variant in self.variants {
            arms.push(match variant {
                SumVariant::Empty => y,
                SumVariant::Word => low,
                SumVariant::Pair => builder.sub(high, low)?,
                SumVariant::Nested => {
                    let fields = if self.tail_when_marked {
                        [low, high]
                    } else {
                        [high, low]
                    };
                    builder.select(marked, &fields, None)?
                }
            });
        }
        let selected = builder.bit_slice(selector, 0, 1)?;
        builder.select(selected, &arms, None)
    }
}
