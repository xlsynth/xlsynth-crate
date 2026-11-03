// SPDX-License-Identifier: Apache-2.0

//! Qualified public constants with a private transitive dependency.

use crate::ast::{BinaryOp, BitsType, Width};
use crate::compound::resize_unsigned;
use crate::constexpr::bits_literal;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrBits, IrValue};

/// The unsigned base, private intermediate, and derived export of a helper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedConstants {
    base: IrBits,
    delta: IrBits,
    combine: [BinaryOp; 2],
}

impl ExportedConstants {
    /// Closed, width-preserving operations used to define and consume exports.
    pub const COMBINERS: [BinaryOp; 5] = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Xor,
        BinaryOp::And,
        BinaryOp::Or,
    ];

    /// Defines a private step from the base and delta, then combines it with
    /// the base.
    pub fn new(base: IrBits, delta: IrBits, combine: [BinaryOp; 2]) -> Result<Self, String> {
        if base.get_bit_count() == 0 || base.get_bit_count() != delta.get_bit_count() {
            return Err("exported constant operands must have the same positive width".to_string());
        }
        Self::validate_combiners(combine)?;
        Ok(Self {
            base,
            delta,
            combine,
        })
    }

    pub(crate) fn validate_combiners(combine: [BinaryOp; 2]) -> Result<(), String> {
        if !combine.iter().all(|op| Self::COMBINERS.contains(op)) {
            return Err("unsupported exported constant combiner".to_string());
        }
        Ok(())
    }

    /// Renders the three definitions in dependency order in the exporting
    /// module.
    pub(crate) fn declarations(&self) -> String {
        let ty = BitsType {
            signed: false,
            width: Width::Concrete(self.base.get_bit_count()),
        };
        let base = bits_literal(ty, &self.base);
        let delta = bits_literal(ty, &self.delta);
        let step = self.combine[0].to_dslx("FUZZ_BASE", &delta, ty);
        let derived = self.combine[1].to_dslx("FUZZ_STEP", "FUZZ_BASE", ty);
        format!(
            "pub const FUZZ_BASE = {base};\nconst FUZZ_STEP = {step};\npub const FUZZ_DERIVED = {derived};\n"
        )
    }

    /// Emits independent PIR for the definitions and their use in the importing
    /// body.
    pub(crate) fn emit_import_ir(
        &self,
        builder: &mut FnBuilder,
        input: BValue,
        combine: [BinaryOp; 2],
        width: usize,
        signed: bool,
    ) -> Result<BValue, BuilderError> {
        let base = builder.literal(IrValue::from_bits(&self.base))?;
        let delta = builder.literal(IrValue::from_bits(&self.delta))?;
        let step = self.combine[0].emit_ir(builder, base, delta, false)?;
        let derived = self.combine[1].emit_ir(builder, step, base, false)?;
        let derived = resize_unsigned(builder, derived, width)?;
        let base = resize_unsigned(builder, base, width)?;
        let intermediate = combine[0].emit_ir(builder, input, derived, signed)?;
        combine[1].emit_ir(builder, intermediate, base, signed)
    }
}

/// Qualifies both public names and converts from their unsigned type into the
/// lane type.
pub(crate) fn import_dslx(
    module: usize,
    arg: &str,
    combine: [BinaryOp; 2],
    ty: BitsType,
) -> String {
    let base = format!("(m{module}::FUZZ_BASE as {})", ty.to_dslx());
    let derived = format!("(m{module}::FUZZ_DERIVED as {})", ty.to_dslx());
    let intermediate = combine[0].to_dslx(arg, &derived, ty);
    combine[1].to_dslx(&format!("({intermediate})"), &base, ty)
}
