// SPDX-License-Identifier: Apache-2.0

//! Check DSLX builtins that take explicit type arguments.

use xlsynth_pir::{IrValue, ir};

use super::{BodyChecker, K, MAX_TYPE_SIZE, bits};
use crate::ast::{Expr, Type};
use crate::builtin::BuiltinOp;
use crate::layout::Layouts;

impl BodyChecker<'_, '_> {
    /// Resolve the type arguments and check the builtin's value operands.
    pub(super) fn type_builtin(
        &mut self,
        name: &str,
        types: &[Type],
        args: &[Expr],
    ) -> Result<(Type, K), String> {
        match (name, types, args) {
            ("all_ones!" | "bit_count" | "element_count", [ty], []) => self.type_constant(name, ty),
            ("decode", [ty], [value]) => {
                let ty = self.source_type(ty)?;
                let Some((false, width)) = bits(&ty) else {
                    return Err(format!(
                        "decode requires an unsigned bit result type, got {ty}"
                    ));
                };
                let value = self.builtin_operand(name, value)?;
                if !matches!(bits(&value.ty), Some((false, _))) {
                    return Err(format!(
                        "decode requires an unsigned bit operand, got {}",
                        value.ty
                    ));
                }
                Ok((
                    ty,
                    K::Builtin {
                        op: BuiltinOp::Decode { width },
                        args: vec![value],
                    },
                ))
            }
            ("widening_cast" | "checked_cast", [ty, source @ ..], [value]) if source.len() <= 1 => {
                self.typed_cast(name, ty, source.first(), value)
            }
            ("all_ones!" | "bit_count" | "element_count", _, _) => Err(format!(
                "{name} requires one type argument and no value arguments"
            )),
            ("decode", _, _) => Err("decode requires one type and one value argument".into()),
            ("widening_cast" | "checked_cast", _, _) => Err(format!(
                "{name} requires one or two type arguments and one value argument"
            )),
            _ => Err(format!("unknown typed builtin '{name}'")),
        }
    }

    /// Materialize an all-ones value or count elements or bits in its layout.
    fn type_constant(&mut self, name: &str, source: &Type) -> Result<(Type, K), String> {
        let ty = self.source_type(source)?;
        let layout = Layouts::new(&self.structs.definitions)
            .lower(&ty)
            .map_err(|e| e.to_string())?;
        if name == "all_ones!" {
            self.structs.check_fill_members(&ty, /* ones= */ true)?;
            return Ok((ty, K::Literal(all_ones_value(&layout)?)));
        }
        let count = match (&layout, name) {
            (ir::Type::Array(array), "element_count") => array.element_count,
            (ir::Type::Tuple(fields), "element_count") => fields.len(),
            _ => layout.bit_count(),
        };
        let value = IrValue::make_ubits(32, count as u64).map_err(|e| e.to_string())?;
        Ok((
            Type::Bits {
                signed: false,
                width: 32,
            },
            K::Literal(value),
        ))
    }

    /// Enforce widening's representability guarantee before emitting a cast.
    fn typed_cast(
        &mut self,
        name: &str,
        destination: &Type,
        source: Option<&Type>,
        value: &Expr,
    ) -> Result<(Type, K), String> {
        let ty = self.source_type(destination)?;
        let value = if let Some(source) = source {
            let source = self.source_type(source)?;
            self.dslx_argument(value, &source)?
        } else {
            self.builtin_operand(name, value)?
        };
        let (Some((source_signed, source_width)), Some((target_signed, width))) =
            (bits(&value.ty), bits(&ty))
        else {
            return Err(format!(
                "{name} requires bit types, got {} and {ty}",
                value.ty
            ));
        };
        if name == "checked_cast" {
            return Ok((
                ty,
                K::Builtin {
                    op: BuiltinOp::CheckedCast {
                        source_signed,
                        target_signed,
                        width,
                    },
                    args: vec![value],
                },
            ));
        }
        let representable = (source_signed == target_signed && width >= source_width)
            || (!source_signed && target_signed && width > source_width);
        if !representable {
            return Err(format!(
                "cannot cast from {} to {ty} with widening_cast",
                value.ty
            ));
        }
        Ok((
            ty,
            K::Cast {
                value: Box::new(value),
            },
        ))
    }
}

/// Fill each leaf while preserving the concrete type of empty arrays.
fn all_ones_value(layout: &ir::Type) -> Result<IrValue, String> {
    match layout {
        ir::Type::Bits(width) => Ok(IrValue::all_ones_bits(*width)),
        ir::Type::Tuple(fields) => {
            let members = fields
                .iter()
                .map(|ty| all_ones_value(ty))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(IrValue::make_tuple(&members))
        }
        ir::Type::Array(array) => {
            let elements = if array.element_count == 0 {
                Vec::new()
            } else {
                vec![all_ones_value(&array.element_type)?; array.element_count]
            };
            IrValue::make_array_typed((*array.element_type).clone(), &elements)
                .map_err(|e| e.to_string())
        }
        ir::Type::Token => Err("all_ones! requires a data type".into()),
    }
}

/// Materialize a byte string as a DSLX array of `u8`.
pub(super) fn byte_string(bytes: &[u8]) -> Result<(Type, K), String> {
    if bytes.is_empty() {
        return Err("zero-length DSLX strings are not supported".into());
    }
    if bytes.len() > (MAX_TYPE_SIZE - 1) / 8 {
        return Err("DSLX string exceeds the maximum concrete type size".into());
    }
    let elements = bytes
        .iter()
        .map(|&byte| IrValue::make_ubits(8, byte.into()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let value =
        IrValue::make_array_typed(ir::Type::Bits(8), &elements).map_err(|e| e.to_string())?;
    let ty = Type::Array {
        element: Box::new(Type::Bits {
            signed: false,
            width: 8,
        }),
        size: bytes.len(),
    };
    Ok((ty, K::Literal(value)))
}

#[cfg(test)]
mod tests {
    use crate::{check_module, parse_module};
    use xlsynth_pir::IrValue;

    #[test]
    fn typed_constants_use_nominal_and_nested_array_layouts() -> Result<(), String> {
        let module = parse_module(
            r#"
enum Mode : u3 { Zero = 0, High = 7 }
struct Packet { lanes: u4[2], mode: Mode }
const BITS = bit_count<Packet>();
const FIELDS = element_count<Packet>();
const ENUM_BITS = element_count<Mode>();
const ARRAY_ELEMENTS = element_count<u4[2][3]>();
const ONES = all_ones!<Packet>();
const EMPTY = all_ones!<Mode[0]>();
"#,
        )?;
        let checked = check_module(&module)?;
        let constants = checked.constants();
        assert_eq!(constants[0].value, IrValue::u32(11));
        assert_eq!(constants[1].value, IrValue::u32(2));
        assert_eq!(constants[2].value, IrValue::u32(3));
        assert_eq!(constants[3].value, IrValue::u32(3));
        assert_eq!(
            constants[4].value,
            IrValue::parse_typed("([bits[4]:15, bits[4]:15], bits[3]:7)")
                .map_err(|e| e.to_string())?,
        );
        assert_eq!(
            constants[5].value.type_(),
            xlsynth_pir::ir::Type::new_array(xlsynth_pir::ir::Type::Bits(3), 0),
        );
        Ok(())
    }

    #[test]
    fn typed_casts_reject_narrowing_and_non_bits() -> Result<(), String> {
        for (source, error) in [
            ("fn f(x: u8) -> s8 { widening_cast<s8>(x) }", "cannot cast"),
            ("fn f(x: s8) -> u9 { widening_cast<u9>(x) }", "cannot cast"),
            ("fn f(x: u8) -> u7 { widening_cast<u7>(x) }", "cannot cast"),
            (
                "fn f(x: u1[2]) -> u2 { checked_cast<u2>(x) }",
                "requires bit types",
            ),
            (
                "fn f(x: s8) -> u8 { decode<u8>(x) }",
                "unsigned bit operand",
            ),
            (
                "fn f(x: u8) -> s8 { decode<s8>(x) }",
                "unsigned bit result type",
            ),
            (
                "enum E : u2 { Zero = 0 } const BAD = all_ones!<(E,)>();",
                "has no all-ones member",
            ),
        ] {
            let module = parse_module(source)?;
            let failure = check_module(&module)
                .err()
                .ok_or("expected checking failure")?;
            assert!(failure.contains(error), "{source}: {failure}");
        }
        Ok(())
    }
}
