// SPDX-License-Identifier: Apache-2.0

//! Mission expressions that exercise bound methods, boolean control and casts.

use crate::ast::{BitsType, Comparison, Width};
use crate::compound::{metadata_literal, resize_unsigned};
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrValue};

/// Maps a method bound to a runtime key over a small array, then projects one
/// element. Each key type has a single concrete method signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReceiverMap {
    key_width: usize,
    length: usize,
}

impl ReceiverMap {
    pub const WIDTHS: [usize; 4] = [7, 16, 65, 129];

    /// Restricts receiver types to the concrete declarations emitted per
    /// module.
    pub fn new(key_width: usize, length: usize) -> Result<Self, String> {
        if !Self::WIDTHS.contains(&key_width) || !(2..=4).contains(&length) {
            return Err("receiver map requires a supported key width and 2..=4 elements".into());
        }
        Ok(Self { key_width, length })
    }

    pub(crate) fn key_width(self) -> usize {
        self.key_width
    }

    pub(crate) fn declaration(width: usize) -> String {
        format!(
            r#"
struct FuzzKey{width} {{ key: uN[{width}] }}
impl FuzzKey{width} {{
    fn apply(self, value: uN[{width}]) -> uN[{width}] {{ (value ^ self.key) + self.key }}
}}
"#
        )
    }

    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let [x, y, z] = args;
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let width = self.key_width;
        let length = self.length;
        let lane = ty.to_dslx();
        let fourth = format!("({x} ^ {z})");
        let elements = [x, z, y, &fourth][..length]
            .iter()
            .map(|arg| format!("(({arg} as {unsigned}) as uN[{width}])"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            r#"{{
        let receiver = FuzzKey{width} {{ key: ({y} as {unsigned}) as uN[{width}] }};
        let xs = [{elements}];
        let index = (({z} as {unsigned}) as u32) % u32:{length};
        map(xs, receiver.apply)[index] as {lane}
    }}"#
        )
    }

    /// Selects the source element first and applies the key arithmetically,
    /// independent of DSLX's map, method dispatch and receiver binding.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        width: usize,
    ) -> Result<BValue, BuilderError> {
        let [x, y, z] = args;
        let index = resize_unsigned(builder, z, 32)?;
        let length = metadata_literal(builder, self.length)?;
        let index = builder.umod(index, length)?;
        let mut element = x;
        for slot in 1..self.length {
            let source = match slot {
                1 => z,
                2 => y,
                _ => builder.xor(x, z)?,
            };
            let slot = metadata_literal(builder, slot)?;
            let is_slot = builder.eq(index, slot)?;
            element = builder.select(is_slot, &[element, source], None)?;
        }
        let element = resize_unsigned(builder, element, self.key_width)?;
        let key = resize_unsigned(builder, y, self.key_width)?;
        let masked = builder.xor(element, key)?;
        let mapped = builder.add(masked, key)?;
        resize_unsigned(builder, mapped, width)
    }
}

/// Combines two independently chosen comparisons and selects a lane value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalPredicates {
    comparisons: [Comparison; 2],
    conjunction: bool,
}

impl LogicalPredicates {
    pub fn new(comparisons: [Comparison; 2], conjunction: bool) -> Self {
        Self {
            comparisons,
            conjunction,
        }
    }

    pub(crate) fn to_dslx(self, args: [&str; 3]) -> String {
        let [x, y, z] = args;
        let [left, right] = self.comparisons.map(Comparison::symbol);
        let join = if self.conjunction { "&&" } else { "||" };
        format!("if ({x} {left} {y}) {join} ({y} {right} {z}) {{ {x} }} else {{ {z} }}")
    }

    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        let [x, y, z] = args;
        let [left, right] = self.comparisons;
        let left = left.emit_ir(builder, x, y, signed)?;
        let right = right.emit_ir(builder, y, z, signed)?;
        let condition = if self.conjunction {
            builder.and(left, right)?
        } else {
            builder.or(left, right)?
        };
        builder.select(condition, &[z, x], None)
    }
}

/// Casts to the same signedness and observes the top lane-width bits. When
/// widening, the selected slice includes every extension bit that fits the
/// lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedCast {
    extra: usize,
}

impl CheckedCast {
    pub fn new(extra: usize) -> Result<Self, String> {
        if extra > 4 {
            return Err("checked cast widening requires 0..=4 extra bits".into());
        }
        Ok(Self { extra })
    }

    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        cast_projection_dslx("checked_cast", self.extra, ty, args)
    }

    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        cast_projection_ir(builder, self.extra, args, signed)
    }
}

/// Observes extension bits from a same-signedness `widening_cast`, including
/// the equal-width case and wide extensions across machine-word boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WideningCast {
    extra: usize,
}

impl WideningCast {
    pub fn new(extra: usize) -> Result<Self, String> {
        if extra > 129 {
            return Err("widening cast requires 0..=129 extra bits".into());
        }
        Ok(Self { extra })
    }

    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        cast_projection_dslx("widening_cast", self.extra, ty, args)
    }

    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        cast_projection_ir(builder, self.extra, args, signed)
    }
}

/// Casts and selects the upper lane-width bits, exposing the extension bits.
fn cast_projection_dslx(builtin: &str, extra: usize, ty: BitsType, args: [&str; 3]) -> String {
    let [x, y, _] = args;
    let lane = ty.to_dslx();
    let wide = match ty.width {
        Width::Concrete(width) => (width + extra).to_string(),
        Width::Parametric if extra == 0 => "N".to_string(),
        Width::Parametric => format!("N + u32:{extra}"),
    };
    let sign = if ty.signed { "s" } else { "u" };
    format!(
        r#"{{
        let casted = {builtin}<{sign}N[{wide}]>({x});
        (((casted as uN[{wide}]) >> u32:{extra}) as {lane}) ^ {y}
    }}"#
    )
}

/// The upper lane-width bits equal a right shift of the original word, even
/// when the number of added bits exceeds the original width.
fn cast_projection_ir(
    builder: &mut FnBuilder,
    extra: usize,
    args: [BValue; 3],
    signed: bool,
) -> Result<BValue, BuilderError> {
    let [x, y, _] = args;
    let shift = metadata_literal(builder, extra)?;
    let projected = if signed {
        builder.shra(x, shift)?
    } else {
        builder.shrl(x, shift)?
    };
    builder.xor(projected, y)
}

/// The aggregate tested against a named constant in an ordinary runtime match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamedPatternShape {
    Tuple,
    PayloadEnum,
}

impl NamedPatternShape {
    pub const ALL: [Self; 2] = [Self::Tuple, Self::PayloadEnum];
}

/// Matches two runtime fields against independently chosen constant values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NamedAggregateMatch {
    shape: NamedPatternShape,
    widths: [usize; 2],
    values: [u8; 2],
}

impl NamedAggregateMatch {
    /// Small field widths give both equality outcomes reachable inputs even in
    /// small exhaustive tests. Each byte supplies its field's low bits.
    pub fn new(
        shape: NamedPatternShape,
        widths: [usize; 2],
        values: [u8; 2],
    ) -> Result<Self, String> {
        if widths.iter().any(|width| !(1..=8).contains(width)) {
            return Err("named aggregate pattern fields must have widths in 1..=8".into());
        }
        let values = std::array::from_fn(|i| values[i] & (u8::MAX >> (8 - widths[i])));
        Ok(Self {
            shape,
            widths,
            values,
        })
    }

    pub(crate) fn needs_sum(self) -> bool {
        self.shape == NamedPatternShape::PayloadEnum
    }

    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let [x, y, z] = args;
        let [a, b] = self.widths;
        let [first, second] = self.values;
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let (constant, subject) = match self.shape {
            NamedPatternShape::Tuple => (
                format!("(uN[{a}]:{first}, uN[{b}]:{second})"),
                "(low, high)".into(),
            ),
            NamedPatternShape::PayloadEnum => {
                let nominal = format!("FuzzSum<u32:{a}, u32:{b}>");
                (
                    format!("{nominal}::Pair {{ lo: uN[{a}]:{first}, hi: uN[{b}]:{second} }}"),
                    format!(
                        "if {z} as bool {{ {nominal}::Pair {{ hi: high, lo: low }} }} else {{ {nominal}::Empty }}"
                    ),
                )
            }
        };
        format!(
            r#"{{
        const TARGET = {constant};
        let low = ({x} as {unsigned}) as uN[{a}];
        let high = ({y} as {unsigned}) as uN[{b}];
        let subject = {subject};
        match subject {{ TARGET => !{z}, _ => {z}, }}
    }}"#
        )
    }

    /// Computes each field equality and the enum discriminant directly. Both
    /// arms differ in every output bit regardless of the selected lane value.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
    ) -> Result<BValue, BuilderError> {
        let [x, y, z] = args;
        let equals = |builder: &mut FnBuilder, input, width, value: u8| {
            let field = resize_unsigned(builder, input, width)?;
            let literal = IrValue::make_ubits(width, u64::from(value))
                .map_err(|e| BuilderError::InvalidOperation(e.to_string()))?;
            let literal = builder.literal(literal)?;
            builder.eq(field, literal)
        };
        let low = equals(builder, x, self.widths[0], self.values[0])?;
        let high = equals(builder, y, self.widths[1], self.values[1])?;
        let mut matched = builder.and(low, high)?;
        if self.needs_sum() {
            let pair = builder.bit_slice(z, 0, 1)?;
            matched = builder.and(matched, pair)?;
        }
        let hit = builder.not(z)?;
        builder.select(matched, &[z, hit], None)
    }
}
