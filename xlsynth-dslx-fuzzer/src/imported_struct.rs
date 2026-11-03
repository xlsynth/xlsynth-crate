// SPDX-License-Identifier: Apache-2.0

//! Construction and field updates through aliases of imported nominal types.

use crate::ast::{BinaryOp, BitsType, HelperModule};
use crate::compound::resize_unsigned;
use xlsynth_pir::{BValue, BuilderError, FnBuilder};

/// The public row type of a helper, defined locally or aliased from an import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructExport {
    Definition { widths: [usize; 2] },
    Alias { module: usize },
}

impl StructExport {
    pub(crate) fn declaration(self) -> String {
        match self {
            Self::Definition { widths: [lo, hi] } => format!(
                "pub struct FuzzEntry {{ lo: uN[{lo}], hi: uN[{hi}] }}\npub type FuzzRow = FuzzEntry;\n"
            ),
            Self::Alias { module } => format!("pub type FuzzRow = m{module}::FuzzRow;\n"),
        }
    }

    /// Resolves both field types through the acyclic import graph.
    pub(crate) fn widths(self, available: &[HelperModule]) -> Result<[usize; 2], String> {
        match self {
            Self::Definition { widths } if widths.contains(&0) => {
                Err("exported struct fields must have positive widths".into())
            }
            Self::Definition { widths } => Ok(widths),
            Self::Alias { module } => imported_widths(module, available),
        }
    }
}

pub(crate) fn imported_widths(
    module: usize,
    available: &[HelperModule],
) -> Result<[usize; 2], String> {
    let export = available
        .get(module)
        .and_then(HelperModule::struct_export)
        .ok_or("struct import must reference an earlier helper exporting a row type")?;
    export.widths(&available[..module])
}

/// Spells the constructor with a local alias and the splat update with the
/// qualified type, then observes the changed field and its preserved sibling.
pub(crate) fn update_dslx(
    module: usize,
    widths: [usize; 2],
    ty: BitsType,
    args: [&str; 3],
    update_hi: bool,
    combine: BinaryOp,
) -> String {
    let [x, y, z] = args;
    let [lo, hi] = widths;
    let lane = ty.to_dslx();
    let unsigned = BitsType {
        signed: false,
        ..ty
    }
    .to_dslx();
    let (field, field_width) = if update_hi { ("hi", hi) } else { ("lo", lo) };
    let result = combine.to_dslx(
        &format!("(updated.lo as {lane})"),
        &format!("(updated.hi as {lane})"),
        ty,
    );
    format!(
        r#"{{
        let original = FuzzImportedRow{module} {{ hi: ({y} as {unsigned}) as uN[{hi}], lo: ({x} as {unsigned}) as uN[{lo}] }};
        let changed = original.{field} ^ (({z} as {unsigned}) as uN[{field_width}]);
        let updated = m{module}::FuzzRow {{ {field}: changed, ..original }};
        {result}
    }}"#
    )
}

/// Models the two fields directly as scalars without constructing, updating,
/// or indexing a tuple in the reference.
pub(crate) fn emit_update_ir(
    builder: &mut FnBuilder,
    widths: [usize; 2],
    args: [BValue; 3],
    update_hi: bool,
    combine: BinaryOp,
    lane_width: usize,
    signed: bool,
) -> Result<BValue, BuilderError> {
    let [x, y, z] = args;
    let mut fields = [
        resize_unsigned(builder, x, widths[0])?,
        resize_unsigned(builder, y, widths[1])?,
    ];
    let index = usize::from(update_hi);
    let delta = resize_unsigned(builder, z, widths[index])?;
    fields[index] = builder.xor(fields[index], delta)?;
    let lo = resize_unsigned(builder, fields[0], lane_width)?;
    let hi = resize_unsigned(builder, fields[1], lane_width)?;
    combine.emit_ir(builder, lo, hi, signed)
}
