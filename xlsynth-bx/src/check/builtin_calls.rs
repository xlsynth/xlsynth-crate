// SPDX-License-Identifier: Apache-2.0

//! Typecheck pure builtin calls with inferred value parameters.

use super::{BodyChecker, ConstantUse, K, MAX_ELABORATION_DEPTH, TypedExpr, bits, minimum_type};
use crate::ast::{Expr, Type};
use crate::builtin::BuiltinOp;
use xlsynth_pir::IrValue;

/// Internal spelling keeps a builtin in a BX library from binding to a
/// declaration with the same name in its importer.
pub(crate) const BUILTIN_PREFIX: &str = "$builtin$";

/// Recognize a pure builtin whose type arguments are inferred from its
/// operands.
pub(crate) fn is_call_builtin(name: &str) -> bool {
    matches!(
        name,
        "rev"
            | "clz"
            | "ctz"
            | "and_reduce"
            | "or_reduce"
            | "xor_reduce"
            | "encode"
            | "one_hot"
            | "one_hot_sel"
            | "priority_sel"
            | "gate!"
            | "bit_slice_update"
            | "signex"
            | "umulp"
            | "smulp"
            | "ceillog2"
            | "configured_value_or"
            | "array_rev"
            | "array_size"
            | "array_slice"
            | "zip"
            | "update"
    )
}

impl BodyChecker<'_, '_> {
    /// An independent parametric needs a type from its argument.
    pub(super) fn builtin_operand(&mut self, name: &str, arg: &Expr) -> Result<TypedExpr, String> {
        if minimum_type(arg)?.is_some() {
            return Err(format!("cannot infer the type of an argument to '{name}'"));
        }
        self.expr(arg, None)
    }

    /// Resolve independent widths from the data arguments, then remove
    /// operands that only witness a type or a constexpr control.
    pub(super) fn builtin_call(&mut self, name: &str, args: &[Expr]) -> Result<(Type, K), String> {
        let arity = match name {
            "array_rev" | "array_size" | "array_slice" | "zip" => {
                return self.array_builtin(name, args);
            }
            "update" => return self.builtin_array_update(args),
            "one_hot"
            | "one_hot_sel"
            | "gate!"
            | "signex"
            | "umulp"
            | "smulp"
            | "configured_value_or" => 2,
            "priority_sel" | "bit_slice_update" => 3,
            _ => 1,
        };
        if args.len() != arity {
            return Err(format!(
                "'{name}' expects {arity} arguments, got {}",
                args.len()
            ));
        }
        let (ty, op, operands) = match name {
            "ceillog2" => return self.ceillog2(&args[0]),
            "configured_value_or" => return self.configured_value_or(&args[0], &args[1]),
            "gate!" => {
                let condition = self.dslx_argument(&args[0], &Type::Bool)?;
                let value = self.builtin_operand(name, &args[1])?;
                (value.ty.clone(), BuiltinOp::Gate, vec![condition, value])
            }
            "one_hot_sel" | "priority_sel" => return self.builtin_selector(name, args),
            "signex" => {
                let value = self.builtin_operand(name, &args[0])?;
                require_bits(name, &value.ty, None)?;
                let witness = self.builtin_operand(name, &args[1])?;
                let width = require_bits(name, &witness.ty, None)?;
                (witness.ty, BuiltinOp::Signex { width }, vec![value])
            }
            "umulp" | "smulp" => {
                let signed = name == "smulp";
                let (left, right) = if minimum_type(&args[0])?.is_some() {
                    let right = self.builtin_operand(name, &args[1])?;
                    let left = self.dslx_argument(&args[0], &right.ty)?;
                    (left, right)
                } else {
                    let left = self.builtin_operand(name, &args[0])?;
                    let right = self.dslx_argument(&args[1], &left.ty)?;
                    (left, right)
                };
                let width = require_bits(name, &left.ty, Some(signed))?;
                let op = if signed {
                    BuiltinOp::Smulp { width }
                } else {
                    BuiltinOp::Umulp { width }
                };
                let unsigned = Type::Bits {
                    signed: false,
                    width,
                };
                (
                    Type::Tuple(vec![unsigned.clone(), unsigned]),
                    op,
                    vec![left, right],
                )
            }
            "bit_slice_update" => {
                let operands = args
                    .iter()
                    .map(|arg| {
                        let value = self.builtin_operand(name, arg)?;
                        require_bits(name, &value.ty, Some(false))?;
                        Ok(value)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                (operands[0].ty.clone(), BuiltinOp::BitSliceUpdate, operands)
            }
            _ => {
                let value = self.builtin_operand(name, &args[0])?;
                let width = require_bits(name, &value.ty, Some(false))?;
                let (ty, op) = match name {
                    "rev" => (value.ty.clone(), BuiltinOp::Rev),
                    "clz" => (value.ty.clone(), BuiltinOp::Clz),
                    "ctz" => (value.ty.clone(), BuiltinOp::Ctz),
                    "and_reduce" => (Type::Bool, BuiltinOp::AndReduce),
                    "or_reduce" => (Type::Bool, BuiltinOp::OrReduce),
                    "xor_reduce" => (Type::Bool, BuiltinOp::XorReduce),
                    "encode" => {
                        let width = usize::BITS - width.saturating_sub(1).leading_zeros();
                        (
                            Type::Bits {
                                signed: false,
                                width: width as usize,
                            },
                            BuiltinOp::Encode,
                        )
                    }
                    "one_hot" => {
                        let priority = self
                            .parametric_value(&args[1], &Type::Bool)
                            .map_err(|e| format!("'one_hot' priority: {e}"))?;
                        let lsb_priority = priority == IrValue::bool(true);
                        let width = width.checked_add(1).ok_or("'one_hot' width overflow")?;
                        (
                            Type::Bits {
                                signed: false,
                                width,
                            },
                            BuiltinOp::OneHot { lsb_priority },
                        )
                    }
                    _ => return Err(format!("unknown builtin '{name}'")),
                };
                (ty, op, vec![value])
            }
        };
        self.structs.validate_type(&ty)?;
        Ok((ty, K::Builtin { op, args: operands }))
    }

    /// Selectors consume an unsigned mask and a same-length array of bit
    /// values. Priority selection also takes a default of the element type.
    fn builtin_selector(&mut self, name: &str, args: &[Expr]) -> Result<(Type, K), String> {
        let cases = self.builtin_operand(name, &args[1])?;
        let Type::Array { element, size } = &cases.ty else {
            return Err(format!(
                "'{name}' requires an array of cases, got {}",
                cases.ty
            ));
        };
        require_bits(name, element, None)?;
        let selector_type = Type::Bits {
            signed: false,
            width: *size,
        };
        let selector = self
            .dslx_argument(&args[0], &selector_type)
            .map_err(|e| format!("'{name}' selector: {e}"))?;
        let work = size.saturating_mul(2).saturating_add(1);
        self.charge_expansion("builtin selection", work)?;
        let ty = element.as_ref().clone();
        let mut args_out = vec![selector, cases];
        let op = if name == "priority_sel" {
            let default = self.dslx_argument(&args[2], &ty)?;
            args_out.push(default);
            BuiltinOp::PrioritySel
        } else {
            BuiltinOp::OneHotSel
        };
        Ok((ty, K::Builtin { op, args: args_out }))
    }

    /// Evaluate the constexpr logarithm at its declared unsigned width.
    fn ceillog2(&mut self, arg: &Expr) -> Result<(Type, K), String> {
        let arg = self.builtin_operand("ceillog2", arg)?;
        let width = require_bits("ceillog2", &arg.ty, Some(false))?;
        self.enable_constant_functions();
        let value = self
            .checked_constant_for(&arg, ConstantUse::ValueParametric)?
            .ok_or("'ceillog2' requires a compile-time argument")?;
        Ok((arg.ty, K::Literal(ceillog2_value(&value, width)?)))
    }

    /// A map of the constexpr callback requires all input elements up front.
    pub(super) fn map_ceillog2(
        &mut self,
        array: &TypedExpr,
        element: &Type,
        size: usize,
    ) -> Result<(Type, K), String> {
        let width = require_bits("ceillog2", element, Some(false))?;
        let shape = self.structs.type_shape(&array.ty, MAX_ELABORATION_DEPTH)?;
        self.charge_expansion("map ceillog2", size.saturating_add(shape.size.div_ceil(64)))?;
        self.enable_constant_functions();
        let array_value = self
            .checked_constant_for(array, ConstantUse::ValueParametric)?
            .ok_or("map callback 'ceillog2' requires a compile-time array")?;
        let values = array_value
            .as_elements()
            .map_err(|e| e.to_string())?
            .iter()
            .map(|value| ceillog2_value(value, width))
            .collect::<Result<Vec<_>, _>>()?;
        let result = IrValue::make_array(&values).map_err(|e| e.to_string())?;
        Ok((array.ty.clone(), K::Literal(result)))
    }

    /// BX has no conversion-time overrides, so evaluate the declared default.
    fn configured_value_or(&mut self, label: &Expr, default: &Expr) -> Result<(Type, K), String> {
        let label = self.expr(label, None)?;
        if !matches!(&label.ty, Type::Array { element, .. } if bits(element) == Some((false, 8))) {
            return Err("'configured_value_or' label requires a byte array".into());
        }
        self.enable_constant_functions();
        let default = self.builtin_operand("configured_value_or", default)?;
        let value = self
            .checked_constant_for(&default, ConstantUse::ValueParametric)?
            .ok_or("'configured_value_or' requires a compile-time default")?;
        Ok((default.ty, K::Literal(value)))
    }
}

/// Compute the rounded-up logarithm without narrowing the input bits.
fn ceillog2_value(value: &IrValue, width: usize) -> Result<IrValue, String> {
    let bits = value.as_bits().map_err(|e| e.to_string())?;
    let result = (0..width)
        .rfind(|&i| bits.get_bit(i).expect("input bit"))
        .map_or(0, |msb| {
            msb + usize::from((0..msb).any(|i| bits.get_bit(i).expect("input bit")))
        });
    IrValue::make_ubits(width, result as u64).map_err(|e| e.to_string())
}

/// Require a bit vector with the specified signedness, when the signature
/// constrains it.
fn require_bits(name: &str, ty: &Type, signed: Option<bool>) -> Result<usize, String> {
    match bits(ty) {
        Some((actual_signed, width)) if signed.is_none_or(|signed| signed == actual_signed) => {
            Ok(width)
        }
        _ => {
            let required = match signed {
                None => "bits",
                Some(false) => "unsigned bits",
                Some(true) => "signed bits",
            };
            Err(format!("'{name}' requires {required}, got {ty}"))
        }
    }
}
