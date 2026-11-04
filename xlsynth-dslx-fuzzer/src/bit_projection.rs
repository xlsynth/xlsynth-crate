// SPDX-License-Identifier: Apache-2.0

//! Bit and array projections whose intermediate width and signedness differ
//! from the lane.

use crate::ast::{BitsType, UnaryOp};
use crate::compound::resize_unsigned;
use xlsynth_pir::{BValue, BuilderError, FnBuilder};

/// Chooses the representation from which a typed intermediate is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionKind {
    PackArray,
    UnpackElement,
    WidthSlice,
}

impl ProjectionKind {
    pub const ALL: [Self; 3] = [Self::PackArray, Self::UnpackElement, Self::WidthSlice];
}

/// Observes a packed word, array element or width slice after an optional
/// width-preserving unary operation and a cast back to the lane type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitProjection {
    kind: ProjectionKind,
    element_width: usize,
    signed: bool,
    unary: Option<UnaryOp>,
}

impl BitProjection {
    /// Bounds the intermediate word while allowing it to be wider or narrower
    /// than the lane.
    pub fn new(
        kind: ProjectionKind,
        element_width: usize,
        signed: bool,
        unary: Option<UnaryOp>,
    ) -> Result<Self, String> {
        if !(1..=65).contains(&element_width) {
            return Err("projection element width must be in 1..=65".into());
        }
        Ok(Self {
            kind,
            element_width,
            signed,
            unary,
        })
    }

    pub(crate) fn to_dslx(self, ty: BitsType, [x, y, _]: [&str; 3]) -> String {
        let lane = ty.to_dslx();
        let unsigned = BitsType {
            signed: false,
            ..ty
        }
        .to_dslx();
        let element = self.element_width;
        let packed = element * 2;
        let prefix = if self.signed { "sN" } else { "uN" };
        let x = format!("({x} as {unsigned})");
        let y = format!("({y} as {unsigned})");
        let read = match self.kind {
            ProjectionKind::PackArray => {
                format!("[{x} as uN[{element}], {y} as uN[{element}]] as {prefix}[{packed}]")
            }
            ProjectionKind::UnpackElement => {
                format!("(({x} as uN[{packed}]) as {prefix}[{element}][2])[{y} as u1]")
            }
            ProjectionKind::WidthSlice => {
                format!("({x} as uN[{packed}])[{y}+:{prefix}[{element}]]")
            }
        };
        let op = match self.unary {
            None => "",
            Some(UnaryOp::Not) => "!",
            Some(UnaryOp::Neg) => "-",
        };
        format!("{{ let word = {read}; ({op}word) as {lane} }}")
    }

    /// Uses scalar bit operations for packing and indexing, then resizes
    /// according to the declared source type of the final cast.
    pub(crate) fn emit_ir(
        self,
        builder: &mut FnBuilder,
        [x, y, _]: [BValue; 3],
        lane_width: usize,
    ) -> Result<BValue, BuilderError> {
        let element = self.element_width;
        let packed = element * 2;
        let (word, width) = match self.kind {
            ProjectionKind::PackArray => {
                let high = resize_unsigned(builder, x, element)?;
                let low = resize_unsigned(builder, y, element)?;
                (builder.concat(&[high, low])?, packed)
            }
            ProjectionKind::UnpackElement => {
                let bits = resize_unsigned(builder, x, packed)?;
                let high = builder.bit_slice(bits, element, element)?;
                let low = builder.bit_slice(bits, 0, element)?;
                let index = builder.bit_slice(y, 0, 1)?;
                (builder.select(index, &[high, low], None)?, element)
            }
            ProjectionKind::WidthSlice => {
                let bits = resize_unsigned(builder, x, packed)?;
                let shifted = builder.shrl(bits, y)?;
                (builder.bit_slice(shifted, 0, element)?, element)
            }
        };
        let word = match self.unary {
            None => word,
            Some(UnaryOp::Not) => builder.not(word)?,
            Some(UnaryOp::Neg) => builder.neg(word)?,
        };
        if self.signed && lane_width > width {
            builder.sign_extend(word, lane_width)
        } else {
            resize_unsigned(builder, word, lane_width)
        }
    }
}
