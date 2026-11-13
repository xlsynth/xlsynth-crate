// SPDX-License-Identifier: Apache-2.0

//! Check DSLX array builtins.

use super::{BodyChecker, K, Type, TypedExpr, bits, compatible, minimum_type};
use crate::ast::Expr;
use crate::builtin::BuiltinOp;
use xlsynth_pir::IrValue;

/// Extract an array's checked element type and length.
fn array_type<'a>(ty: &'a Type, name: &str) -> Result<(&'a Type, usize), String> {
    let Type::Array { element, size } = ty else {
        return Err(format!("'{name}' requires an array, got {ty}"));
    };
    Ok((element, *size))
}

impl BodyChecker<'_, '_> {
    /// Check array builtins with value arguments and no explicit parametrics.
    pub(super) fn array_builtin(&mut self, name: &str, args: &[Expr]) -> Result<(Type, K), String> {
        match (name, args) {
            ("array_rev", [array]) => {
                let array = self.expr(array, None)?;
                let (_, size) = array_type(&array.ty, name)?;
                self.charge_expansion(name, size.saturating_mul(2).saturating_add(1))?;
                Ok((
                    array.ty.clone(),
                    K::Builtin {
                        op: BuiltinOp::ArrayRev,
                        args: vec![array],
                    },
                ))
            }
            ("array_size", [array]) => {
                let array = self.expr(array, None)?;
                let (_, size) = array_type(&array.ty, name)?;
                let size = u32::try_from(size).map_err(|_| "array size does not fit in u32")?;
                Ok((Type::unsigned(32), K::Literal(IrValue::u32(size))))
            }
            ("array_slice", [array, start, witness]) => {
                self.builtin_array_slice(array, start, witness)
            }
            ("zip", [left, right]) => self.builtin_zip(left, right),
            _ => Err(format!(
                "invalid argument count {} for '{name}'",
                args.len()
            )),
        }
    }

    /// Check the type witness without retaining a runtime dependency on it.
    fn builtin_array_slice(
        &mut self,
        array: &Expr,
        start: &Expr,
        witness: &Expr,
    ) -> Result<(Type, K), String> {
        if minimum_type(start)?.is_some() {
            return Err("cannot infer the type of 'array_slice' start".into());
        }
        let start = self.expr(start, None)?;
        if !bits(&start.ty).is_some_and(|(signed, _)| !signed) {
            return Err(format!(
                "'array_slice' requires an unsigned start, got {}",
                start.ty
            ));
        }
        let (array, witness) = self.with_parametric_context(|checker| {
            checker.with_scoped_results(&[array, witness], |checker, operands| {
                checker.array_slice_operands(operands[0], operands[1])
            })
        })?;
        let (element, size) = array_type(&array.ty, "array_slice")?;
        let (witness_element, width) = array_type(&witness.ty, "array_slice")?;
        if !compatible(element, witness_element) {
            return Err(format!(
                "'array_slice' witness element expects {element}, got {witness_element}"
            ));
        }
        if size == 0 && width > 0 {
            return Err("'array_slice' cannot fill a nonempty result from an empty array".into());
        }
        let ty = Type::array(element.clone(), width);
        self.structs.validate_type(&ty)?;
        Ok(self.builtin_with_assert_operand(
            ty,
            K::Builtin {
                op: BuiltinOp::ArraySlice { width },
                args: vec![array, start],
            },
            witness,
        ))
    }

    /// Infer the shared element type independently of the two array lengths.
    fn array_slice_operands(
        &mut self,
        array: &Expr,
        witness: &Expr,
    ) -> Result<(TypedExpr, TypedExpr), String> {
        if [array, witness]
            .iter()
            .any(|expr| matches!(expr, Expr::Array(items) if items.is_empty()))
        {
            return Err("empty array literal requires an array type".into());
        }
        match (array, witness) {
            (Expr::Array(items), Expr::Array(witness_items)) => {
                let operands = items.iter().chain(witness_items).collect::<Vec<_>>();
                let mut values = self.unify_values(&operands)?;
                let element = Box::new(values[0].ty.clone());
                let witness_values = values.split_off(items.len());
                let array_ty = Type::Array {
                    element: element.clone(),
                    size: items.len(),
                };
                let witness_ty = Type::Array {
                    element,
                    size: witness_items.len(),
                };
                Ok((
                    self.finish_expr(array, None, array_ty, K::Array(values))?,
                    self.finish_expr(witness, None, witness_ty, K::Array(witness_values))?,
                ))
            }
            (Expr::Array(items), _) => {
                let witness = self.expr(witness, None)?;
                let (element, _) = array_type(&witness.ty, "array_slice")?;
                let ty = Type::array(element.clone(), items.len());
                Ok((self.expr(array, Some(&ty))?, witness))
            }
            (_, Expr::Array(items)) => {
                let array = self.expr(array, None)?;
                let (element, _) = array_type(&array.ty, "array_slice")?;
                let ty = Type::array(element.clone(), items.len());
                Ok((array, self.expr(witness, Some(&ty))?))
            }
            _ => Ok((self.expr(array, None)?, self.expr(witness, None)?)),
        }
    }

    /// Check the shared length before expanding the arrays into pairs.
    fn builtin_zip(&mut self, left: &Expr, right: &Expr) -> Result<(Type, K), String> {
        let left = self.expr(left, None)?;
        let (left_element, size) = array_type(&left.ty, "zip")?;
        let right = self.expr(right, None)?;
        let (right_element, right_size) = array_type(&right.ty, "zip")?;
        if size != right_size {
            return Err(format!(
                "'zip' expects arrays of equal lengths, got {size} and {right_size}"
            ));
        }
        let ty = Type::array(
            Type::Tuple(vec![left_element.clone(), right_element.clone()]),
            size,
        );
        self.structs.validate_type(&ty)?;
        self.charge_expansion("zip", size.saturating_mul(4).saturating_add(1))?;
        Ok((
            ty,
            K::Builtin {
                op: BuiltinOp::Zip,
                args: vec![left, right],
            },
        ))
    }

    /// Check a scalar index or each index in a tuple against the nested array.
    pub(super) fn builtin_array_update(&mut self, args: &[Expr]) -> Result<(Type, K), String> {
        let [array, index, replacement] = args else {
            return Err(format!("'update' expects 3 arguments, got {}", args.len()));
        };
        let array = self.expr(array, None)?;
        array_type(&array.ty, "update")?;
        // DSLX infers the scalar index and replacement independently. An
        // inline unsized scalar cannot supply either type parameter.
        for (arg, role) in [(index, "index"), (replacement, "replacement")] {
            if minimum_type(arg)?.is_some() {
                return Err(format!("cannot infer the type of 'update' {role}"));
            }
        }
        let index = self.expr(index, None)?;
        let (indices, tuple_index) = match &index.ty {
            Type::Tuple(types) => (types.as_slice(), true),
            ty => (std::slice::from_ref(ty), false),
        };
        let mut element = &array.ty;
        for index in indices {
            (element, _) = array_type(element, "update")?;
            if !bits(index).is_some_and(|(signed, _)| !signed) {
                return Err(format!("'update' requires an unsigned index, got {index}"));
            }
        }
        let index_count = indices.len();
        let replacement = self.expr(replacement, None)?;
        if !compatible(element, &replacement.ty) {
            return Err(format!(
                "'update' replacement expects {element}, got {}",
                replacement.ty
            ));
        }
        if !tuple_index {
            return Ok((
                array.ty.clone(),
                K::ArrayUpdate {
                    array: Box::new(array),
                    index: Box::new(index),
                    replacement: Box::new(replacement),
                },
            ));
        }
        Ok((
            array.ty.clone(),
            K::Builtin {
                op: BuiltinOp::ArrayUpdate { index_count },
                args: vec![array, index, replacement],
            },
        ))
    }
}
