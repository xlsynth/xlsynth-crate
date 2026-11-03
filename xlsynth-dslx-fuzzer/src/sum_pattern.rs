// SPDX-License-Identifier: Apache-2.0

//! Runtime construction and destructuring of payload-bearing enums.

use crate::ast::BitsType;
use crate::compound::resize_unsigned;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue};

pub(crate) const DECLARATION: &str = r#"
enum FuzzSum<A: u32, B: u32> {
    Empty,
    Word(uN[A]),
    Pair { lo: uN[A], hi: uN[B] },
    Nested((uN[A], (bool, uN[B]))),
}
"#;

pub(crate) const NESTED_DECLARATION: &str = r#"
enum FuzzEnvelope<A: u32, B: u32> {
    Single(FuzzSum<A, B>),
    Bundle { items: FuzzSum<A, B>[2], meta: (uN[B], FuzzSum<A, B>) },
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
    fn construct(self, type_name: &str, low: &str, high: &str, marked: &str) -> String {
        match self {
            Self::Empty => format!("{type_name}::Empty"),
            Self::Word => format!("{type_name}::Word({low})"),
            Self::Pair => format!("{type_name}::Pair {{ hi: {high}, lo: {low} }}"),
            Self::Nested => format!("{type_name}::Nested(({low}, ({marked}, {high})))"),
        }
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

/// Checks the alternatives and payload widths shared by both sum operations.
fn validate_sum(variants: [SumVariant; 2], widths: [usize; 2]) -> Result<(), String> {
    if variants[0] == variants[1] {
        return Err("sum construction requires distinct variants".into());
    }
    if widths.iter().any(|width| !(1..=129).contains(width)) {
        return Err("sum payload widths must be in 1..=129".into());
    }
    Ok(())
}

impl SumPattern {
    pub(crate) fn validate(self) -> Result<(), String> {
        validate_sum(self.variants, self.widths)
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
        let [off, on] = self
            .variants
            .map(|variant| variant.construct(&enum_type, "low", "high", "marked"));
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

/// The enclosing type on which DSLX performs a structural sum comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumShape {
    Bare,
    Tuple,
    Array,
    Nested,
}

impl SumShape {
    pub const ALL: [Self; 4] = [Self::Bare, Self::Tuple, Self::Array, Self::Nested];
}

/// Two runtime-selected enum values compared directly or inside aggregates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SumComparison {
    pub variants: [SumVariant; 2],
    pub widths: [usize; 2],
    pub shape: SumShape,
    pub unequal: bool,
}

impl SumComparison {
    pub(crate) fn validate(self) -> Result<(), String> {
        validate_sum(self.variants, self.widths)
    }

    /// Builds both operands separately so their discriminants and payloads
    /// may differ at runtime.
    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let [x, y, z] = args;
        let [a, b] = self.widths;
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let enum_type = format!("FuzzSum<u32:{a}, u32:{b}>");
        let outer_type = format!("FuzzEnvelope<u32:{a}, u32:{b}>");
        let mut body = format!("{{ let tags = ({z} as {unsigned}) as u4;\n");
        for (side, low_source, high_source, bit) in [("left", x, y, 0), ("right", y, z, 1)] {
            let low = format!("{side}_low");
            let high = format!("{side}_high");
            let marked = format!("{side}_marked");
            let [off, on] = self
                .variants
                .map(|variant| variant.construct(&enum_type, &low, &high, &marked));
            body.push_str(&format!(
                "let {low} = ({low_source} as {unsigned}) as uN[{a}];
        let {high} = ({high_source} as {unsigned}) as uN[{b}];
        let {marked} = and_reduce({low_source} as {unsigned});
        let {side} = if tags[{bit}+:u1] {{ {on} }} else {{ {off} }};\n"
            ));
            let word = format!("{enum_type}::Word({low})");
            let operand = match self.shape {
                SumShape::Bare => side.to_string(),
                SumShape::Tuple => format!("({marked}, {side})"),
                SumShape::Array => format!("[{side}, {word}]"),
                SumShape::Nested => format!(
                    "if tags[{}+:u1] {{
            {outer_type}::Bundle {{ items: [{side}, {word}], meta: ({high}, {side}) }}
        }} else {{ {outer_type}::Single({side}) }}",
                    bit + 2
                ),
            };
            body.push_str(&format!("let {side}_operand = {operand};\n"));
        }
        let op = if self.unequal { "!=" } else { "==" };
        body.push_str(&format!(
            "(left_operand {op} right_operand) as {} }}",
            ty.to_dslx()
        ));
        body
    }

    /// Compares the active fields directly, independent of packed tag values,
    /// payload offsets, padding, or aggregate member order in the converter.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        width: usize,
    ) -> Result<BValue, BuilderError> {
        let [x, y, z] = args;
        let [a, b] = self.widths;
        let left_low = resize_unsigned(builder, x, a)?;
        let right_low = resize_unsigned(builder, y, a)?;
        let left_high = resize_unsigned(builder, y, b)?;
        let right_high = resize_unsigned(builder, z, b)?;
        let low_equal = builder.eq(left_low, right_low)?;
        let high_equal = builder.eq(left_high, right_high)?;
        let fields_equal = builder.and(low_equal, high_equal)?;
        let left_mark = builder.and_reduce(x)?;
        let right_mark = builder.and_reduce(y)?;
        let marks_equal = builder.eq(left_mark, right_mark)?;
        let mut cases = Vec::with_capacity(2);
        for variant in self.variants {
            cases.push(match variant {
                SumVariant::Empty => builder.literal(IrValue::from_bits(&IrBits::all_ones(1)))?,
                SumVariant::Word => low_equal,
                SumVariant::Pair => fields_equal,
                SumVariant::Nested => builder.and(fields_equal, marks_equal)?,
            });
        }
        let tags = resize_unsigned(builder, z, 4)?;
        let left_tag = builder.bit_slice(tags, 0, 1)?;
        let right_tag = builder.bit_slice(tags, 1, 1)?;
        let tags_equal = builder.eq(left_tag, right_tag)?;
        let active_fields_equal = builder.select(left_tag, &cases, None)?;
        let sum_equal = builder.and(tags_equal, active_fields_equal)?;
        let equal = match self.shape {
            SumShape::Bare => sum_equal,
            SumShape::Tuple => builder.and(sum_equal, marks_equal)?,
            SumShape::Array => builder.and(sum_equal, low_equal)?,
            SumShape::Nested => {
                let left_bundle = builder.bit_slice(tags, 2, 1)?;
                let right_bundle = builder.bit_slice(tags, 3, 1)?;
                let same_variant = builder.eq(left_bundle, right_bundle)?;
                let single = builder.not(left_bundle)?;
                let extra_fields_equal = builder.or(single, fields_equal)?;
                let payload_equal = builder.and(sum_equal, extra_fields_equal)?;
                builder.and(same_variant, payload_equal)?
            }
        };
        let result = if self.unequal {
            builder.not(equal)?
        } else {
            equal
        };
        resize_unsigned(builder, result, width)
    }
}
