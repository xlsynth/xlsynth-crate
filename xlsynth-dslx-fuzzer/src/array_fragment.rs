// SPDX-License-Identifier: Apache-2.0

//! Packed array casts and tuple-index updates with scalar PIR references.

use crate::ast::{BitsType, Width};
use crate::compound::{metadata_literal, resize_unsigned};
use xlsynth_pir::{BValue, BuilderError, FnBuilder};

/// Which representation the DSLX cast produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastDirection {
    BitsToArray,
    ArrayToBits,
}

/// A packed array cast whose element order is observable at a runtime index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayCast {
    direction: CastDirection,
    length: usize,
}

impl ArrayCast {
    /// Keeps each element backed by its own input slot.
    pub fn new(direction: CastDirection, length: usize) -> Result<Self, String> {
        if !(2..=3).contains(&length) {
            return Err("packed casts require two or three elements".to_string());
        }
        Ok(Self { direction, length })
    }

    /// Observes the cast through array indexing or a bit slice, respectively.
    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let elements = args[..self.length]
            .iter()
            .map(|arg| format!("({arg} as {unsigned})"))
            .collect::<Vec<_>>();
        let length = self.length;
        let index = format!("(({} as {unsigned}) as u3) % u3:{length}", args[2]);
        let lane = ty.to_dslx();
        match self.direction {
            CastDirection::BitsToArray => {
                let word = elements.join(" ++ ");
                format!(
                    "{{ let words = ({word}) as {unsigned}[{length}]; words[{index}] as {lane} }}"
                )
            }
            CastDirection::ArrayToBits => {
                let elements = elements.join(", ");
                let width = match ty.width {
                    Width::Concrete(width) => format!("u32:{width}"),
                    Width::Parametric => "N".to_string(),
                };
                let last = length - 1;
                format!(
                    "{{ let packed = [{elements}] as uN[u32:{length} * {width}]; let start = ((u3:{last} - ({index})) as u32) * {width}; packed[start+:{unsigned}] as {lane} }}"
                )
            }
        }
    }

    /// The most significant word is element zero, so both projections select
    /// directly from the original inputs without constructing either cast.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
    ) -> Result<BValue, BuilderError> {
        let index = bounded_index(builder, args[2], self.length)?;
        builder.select(index, &args[..self.length], Some(args[0]))
    }
}

/// A rectangular array update observed at a cell chosen independently of the
/// runtime write address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixUpdate {
    rows: usize,
    cols: usize,
    observed_row: usize,
    observed_col: usize,
}

impl MatrixUpdate {
    /// Bounds the emitted matrix and requires an in-bounds observation.
    pub fn new(
        rows: usize,
        cols: usize,
        observed_row: usize,
        observed_col: usize,
    ) -> Result<Self, String> {
        if !(2..=4).contains(&rows)
            || !(2..=4).contains(&cols)
            || observed_row >= rows
            || observed_col >= cols
        {
            return Err("matrix updates require two to four rows and columns and an in-bounds observed cell".to_string());
        }
        Ok(Self {
            rows,
            cols,
            observed_row,
            observed_col,
        })
    }

    /// Checkerboard initialization makes both row and column affect the
    /// original cell value, even when the matrix has an even column count.
    fn original_slot(row: usize, col: usize) -> usize {
        (row + col) % 2
    }

    pub(crate) fn to_dslx(self, ty: BitsType, args: [&str; 3]) -> String {
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let rows = (0..self.rows)
            .map(|row| {
                let cells = (0..self.cols)
                    .map(|col| args[Self::original_slot(row, col)])
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{cells}]")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let [x, y, z] = args;
        format!(
            "{{ let matrix = [{rows}]; let row = (({x} as {unsigned}) as u3) % u3:{}; let col = (({y} as {unsigned}) as u3) % u3:{}; update(matrix, (row, col), {z})[u32:{}][u32:{}] }}",
            self.rows, self.cols, self.observed_row, self.observed_col
        )
    }

    /// The observed cell changes precisely when both write indices match.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
    ) -> Result<BValue, BuilderError> {
        let row = bounded_index(builder, args[0], self.rows)?;
        let col = bounded_index(builder, args[1], self.cols)?;
        let observed_row = metadata_literal(builder, self.observed_row)?;
        let observed_col = metadata_literal(builder, self.observed_col)?;
        let same_row = builder.eq(row, observed_row)?;
        let same_col = builder.eq(col, observed_col)?;
        let replaced = builder.and(same_row, same_col)?;
        let original = args[Self::original_slot(self.observed_row, self.observed_col)];
        builder.select(replaced, &[original, args[2]], None)
    }
}

/// Three index bits exercise every dimension and keep dynamic proofs small.
fn bounded_index(
    builder: &mut FnBuilder,
    input: BValue,
    dimension: usize,
) -> Result<BValue, BuilderError> {
    let value = resize_unsigned(builder, input, 3)?;
    let size = metadata_literal(builder, dimension)?;
    let size = resize_unsigned(builder, size, 3)?;
    let index = builder.umod(value, size)?;
    builder.zero_extend(index, 32)
}
