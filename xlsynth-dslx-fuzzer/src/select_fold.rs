// SPDX-License-Identifier: Apache-2.0

//! Array-fed selects and nested tuple folds with scalar PIR references.

use crate::ast::{BinaryOp, BitsType};
use crate::compound::{metadata_literal, resize_unsigned};
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue};

/// How a DSLX select combines cases enabled by its selector bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectKind {
    OneHot,
    Priority,
}

impl SelectKind {
    pub const ALL: [Self; 2] = [Self::OneHot, Self::Priority];
}

/// The source expression that supplies the case array to a select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseArray {
    Updated,
    Ellipsis,
}

impl CaseArray {
    pub const ALL: [Self; 2] = [Self::Updated, Self::Ellipsis];
}

/// Selects from a dynamically updated array or an ellipsis-extended literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArraySelect {
    kind: SelectKind,
    cases: CaseArray,
    length: usize,
}

impl ArraySelect {
    /// Keeps the selector bounded and leaves at least one repeated case in an
    /// ellipsis array.
    pub fn new(kind: SelectKind, cases: CaseArray, length: usize) -> Result<Self, String> {
        if !(3..=5).contains(&length) {
            return Err("array-fed selects require three to five cases".to_string());
        }
        Ok(Self {
            kind,
            cases,
            length,
        })
    }

    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let [x, y, z] = args;
        let lane = ty.to_dslx();
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let length = self.length;
        let selector = format!("(({x} as {unsigned}) as u{length})");
        let cases = match self.cases {
            CaseArray::Updated => {
                let elements = [y, z, x]
                    .into_iter()
                    .cycle()
                    .take(length)
                    .collect::<Vec<_>>()
                    .join(", ");
                let index = format!("(({y} as {unsigned}) as u3) % u3:{length}");
                format!("update([{elements}], {index}, {x} ^ {z})")
            }
            CaseArray::Ellipsis => format!("{lane}[{length}]:[{y}, {z}, ...]"),
        };
        match self.kind {
            SelectKind::OneHot => format!("one_hot_sel({selector}, {cases})"),
            SelectKind::Priority => format!("priority_sel({selector}, {cases}, {y} + {z})"),
        }
    }

    /// Expands the select into per-bit conditionals and models an update as
    /// an address comparison on each case.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        width: usize,
    ) -> Result<BValue, BuilderError> {
        let [x, y, z] = args;
        let selector = resize_unsigned(builder, x, self.length)?;
        let mut cases = Vec::with_capacity(self.length);
        match self.cases {
            CaseArray::Updated => {
                let index = resize_unsigned(builder, y, 3)?;
                let limit = metadata_literal(builder, self.length)?;
                let limit = resize_unsigned(builder, limit, 3)?;
                let index = builder.umod(index, limit)?;
                let replacement = builder.xor(x, z)?;
                for (i, original) in [y, z, x].into_iter().cycle().take(self.length).enumerate() {
                    let address = metadata_literal(builder, i)?;
                    let address = resize_unsigned(builder, address, 3)?;
                    let written = builder.eq(index, address)?;
                    cases.push(builder.select(written, &[original, replacement], None)?);
                }
            }
            CaseArray::Ellipsis => {
                cases.push(y);
                cases.resize(self.length, z);
            }
        }
        let default = match self.kind {
            SelectKind::OneHot => builder.literal(IrValue::from_bits(&IrBits::zero(width)))?,
            SelectKind::Priority => builder.add(y, z)?,
        };
        let mut selected = default;
        // Walk from high to low so lower enabled bits override higher ones
        // for priority_sel. one_hot_sel instead ORs every enabled value.
        for (i, case) in cases.into_iter().enumerate().rev() {
            let enabled = builder.bit_slice(selector, i, 1)?;
            selected = match self.kind {
                SelectKind::Priority => builder.select(enabled, &[selected, case], None)?,
                SelectKind::OneHot => {
                    let masked = builder.select(enabled, &[default, case], None)?;
                    builder.or(selected, masked)?
                }
            };
        }
        Ok(selected)
    }
}

/// A loop over nested tuple elements and a tuple carry, with an inner fold
/// that captures the current outer element and carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NestedFold {
    outer_length: usize,
    inner_trips: usize,
    combine: BinaryOp,
    projection: usize,
}

impl NestedFold {
    pub const COMBINERS: [BinaryOp; 5] = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Xor,
        BinaryOp::And,
        BinaryOp::Or,
    ];

    /// Bounds unrolling and requires a projection of one of the three carry
    /// fields.
    pub fn new(
        outer_length: usize,
        inner_trips: usize,
        combine: BinaryOp,
        projection: usize,
    ) -> Result<Self, String> {
        if !(2..=4).contains(&outer_length) || !(1..=3).contains(&inner_trips) || projection >= 3 {
            return Err("nested folds require 2..=4 elements, 1..=3 inner trips, and a carry projection in 0..3".to_string());
        }
        Ok(Self {
            outer_length,
            inner_trips,
            combine,
            projection,
        })
    }

    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let [x, y, z] = args;
        let elements = (0..self.outer_length)
            .map(|i| {
                format!(
                    "(({}, {}), {})",
                    args[i % 3],
                    args[(i + 2) % 3],
                    args[(i + 1) % 3]
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let step = self.combine.to_dslx("inner_acc", "prev", ty);
        let end = self.inner_trips + 1;
        let field = ["0.0", "0.1", "1"][self.projection];
        format!(
            r#"{{ let folded = for (((a, _), b), ((prev, carry), total)) in [{elements}] {{
        let inner = for (_, inner_acc) in u3:1..u3:{end} {{ ({step}) + b }}(carry);
        ((inner, total ^ a), total + prev + b)
    }}((({x}, {y}), {z})); folded.{field} }}"#
        )
    }

    /// Unrolls both loops directly over three scalar registers. Tuple
    /// destructuring, captures, and counted-loop IR are all confined to DSLX.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        let [mut prev, mut carry, mut total] = args;
        for i in 0..self.outer_length {
            let a = args[i % 3];
            let b = args[(i + 1) % 3];
            let mut inner = carry;
            for _ in 0..self.inner_trips {
                let step = self.combine.emit_ir(builder, inner, prev, signed)?;
                inner = builder.add(step, b)?;
            }
            carry = builder.xor(total, a)?;
            let sum = builder.add(total, prev)?;
            total = builder.add(sum, b)?;
            prev = inner;
        }
        Ok([prev, carry, total][self.projection])
    }
}
