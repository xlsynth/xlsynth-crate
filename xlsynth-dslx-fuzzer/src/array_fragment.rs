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

/// How the runtime coordinate reaches a multidimensional update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateIndexForm {
    Literal,
    Binding,
    TupleProjection,
    HelperCall,
}

impl UpdateIndexForm {
    pub const ALL: [Self; 4] = [
        Self::Literal,
        Self::Binding,
        Self::TupleProjection,
        Self::HelperCall,
    ];
}

/// The coordinate arity determines how much of the array a write replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateShape {
    WholeMatrix,
    Row,
    MatrixCell,
    /// A cell in one of two oppositely initialized planes.
    CubeCell {
        observed_plane: bool,
    },
}

/// A rectangular array update observed at a cell chosen independently of the
/// runtime write address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixUpdate {
    rows: usize,
    cols: usize,
    observed_row: usize,
    observed_col: usize,
    index_form: UpdateIndexForm,
    shape: UpdateShape,
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
            index_form: UpdateIndexForm::Literal,
            shape: UpdateShape::MatrixCell,
        })
    }

    /// Chooses the address expression for two-coordinate cell writes.
    pub fn with_index_form(mut self, index_form: UpdateIndexForm) -> Self {
        self.index_form = index_form;
        self
    }

    /// Changes the granularity of the write, retaining the observed row and
    /// column.
    pub fn with_shape(mut self, shape: UpdateShape) -> Self {
        self.shape = shape;
        self
    }

    pub(crate) fn needs_coordinate_helper(self) -> bool {
        self.shape == UpdateShape::MatrixCell && self.index_form == UpdateIndexForm::HelperCall
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
        let [x, y, z] = args;
        let matrix = |cells: [&str; 2]| {
            array_literal(self.rows, |row| {
                array_literal(self.cols, |col| {
                    cells[Self::original_slot(row, col)].to_owned()
                })
            })
        };
        let original = matrix([x, y]);
        let row = format!("(({x} as {unsigned}) as u3) % u3:{}", self.rows);
        let col = format!("(({y} as {unsigned}) as u3) % u3:{}", self.cols);
        let observation = format!("[u32:{}][u32:{}]", self.observed_row, self.observed_col);
        match self.shape {
            UpdateShape::WholeMatrix => {
                let replacement = matrix([&format!("{x} ^ {z}"), &format!("{y} + {z}")]);
                format!(
                    "{{ let matrix = {original}; update(matrix, (), {replacement}){observation} }}"
                )
            }
            UpdateShape::Row => {
                let replacement = array_literal(self.cols, |col| {
                    if col % 2 == 0 {
                        format!("{x} ^ {z}")
                    } else {
                        format!("{y} + {z}")
                    }
                });
                format!(
                    "{{ let matrix = {original}; let row = {row}; let coord = (row,); update(matrix, coord, {replacement}){observation} }}"
                )
            }
            UpdateShape::CubeCell { observed_plane } => {
                let other = matrix([y, x]);
                let plane = usize::from(observed_plane);
                format!(
                    "{{ let cube = [{original}, {other}]; let row = {row}; let col = {col}; let coord = (({z} as {unsigned}) as u1, row, col as u32); update(cube, coord, {z})[u32:{plane}]{observation} }}"
                )
            }
            UpdateShape::MatrixCell => {
                let update = match self.index_form {
                    UpdateIndexForm::Literal => format!("update(matrix, (row, col), {z})"),
                    UpdateIndexForm::Binding => {
                        format!("let coord = (row, col); update(matrix, coord, {z})")
                    }
                    UpdateIndexForm::TupleProjection => {
                        format!("let write = ((row, col), {z}); update(matrix, write.0, write.1)")
                    }
                    UpdateIndexForm::HelperCall => {
                        format!("update(matrix, fuzz_coordinate(row, col), {z})")
                    }
                };
                format!(
                    "{{ let matrix = {original}; let row = {row}; let col = {col}; {update}{observation} }}"
                )
            }
        }
    }

    /// Tests only the coordinates needed for the observed scalar; no array or
    /// tuple operations are shared with the converter's lowering.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        args: [BValue; 3],
    ) -> Result<BValue, BuilderError> {
        let slot = Self::original_slot(self.observed_row, self.observed_col);
        let replacement = |builder: &mut FnBuilder, slot| {
            if slot == 0 {
                builder.xor(args[0], args[2])
            } else {
                builder.add(args[1], args[2])
            }
        };
        match self.shape {
            UpdateShape::WholeMatrix => replacement(builder, slot),
            UpdateShape::Row => {
                let row = bounded_index(builder, args[0], self.rows)?;
                let observed_row = metadata_literal(builder, self.observed_row)?;
                let same_row = builder.eq(row, observed_row)?;
                let data = replacement(builder, self.observed_col % 2)?;
                builder.select(same_row, &[args[slot], data], None)
            }
            UpdateShape::MatrixCell | UpdateShape::CubeCell { .. } => {
                let row = bounded_index(builder, args[0], self.rows)?;
                let col = bounded_index(builder, args[1], self.cols)?;
                let observed_row = metadata_literal(builder, self.observed_row)?;
                let observed_col = metadata_literal(builder, self.observed_col)?;
                let same_row = builder.eq(row, observed_row)?;
                let same_col = builder.eq(col, observed_col)?;
                let same_cell = builder.and(same_row, same_col)?;
                let (replaced, original) =
                    if let UpdateShape::CubeCell { observed_plane } = self.shape {
                        let plane = resize_unsigned(builder, args[2], 1)?;
                        let same_plane = if observed_plane {
                            plane
                        } else {
                            builder.not(plane)?
                        };
                        (
                            builder.and(same_cell, same_plane)?,
                            args[slot ^ usize::from(observed_plane)],
                        )
                    } else {
                        (same_cell, args[slot])
                    };
                builder.select(replaced, &[original, args[2]], None)
            }
        }
    }
}

/// Builds a DSLX array literal in index order, including each nested dimension.
fn array_literal(length: usize, element: impl FnMut(usize) -> String) -> String {
    format!(
        "[{}]",
        (0..length).map(element).collect::<Vec<_>>().join(", ")
    )
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
