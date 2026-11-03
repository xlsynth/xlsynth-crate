// SPDX-License-Identifier: Apache-2.0

//! Mission expressions that exercise bound methods, boolean control and casts.

use crate::ast::{BitsType, Comparison, Width};
use crate::compound::{metadata_literal, resize_unsigned};
use xlsynth_pir::{BValue, BuilderError, FnBuilder};

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
        let [x, y, _] = args;
        let lane = ty.to_dslx();
        let extra = self.extra;
        let wide = match ty.width {
            Width::Concrete(width) => (width + extra).to_string(),
            Width::Parametric if extra == 0 => "N".to_string(),
            Width::Parametric => format!("N + u32:{extra}"),
        };
        let sign = if ty.signed { "s" } else { "u" };
        format!(
            r#"{{
        let casted = checked_cast<{sign}N[{wide}]>({x});
        (((casted as uN[{wide}]) >> u32:{extra}) as {lane}) ^ {y}
    }}"#
        )
    }

    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        let [x, y, _] = args;
        // The top lane-width bits of an extended word equal a signed or
        // unsigned right shift of the original word, even when extra >= width.
        let shift = metadata_literal(builder, self.extra)?;
        let projected = if signed {
            builder.shra(x, shift)?
        } else {
            builder.shrl(x, shift)?
        };
        builder.xor(projected, y)
    }
}
