// SPDX-License-Identifier: Apache-2.0

//! Elaborate explicit value arguments at the BX call site and ask the native
//! DSLX typechecker to select the concrete imported function.

use super::{BodyChecker, K, MAX_ELABORATION_DEPTH, TypedExpr, has_aggregate_literal};
use crate::ast::Expr;
use crate::dslx_import::NativeDataArg;
use crate::infer_literal::{import_bound_literal_witness, import_literal_witness};
use crate::typed::BoundParametricValue;

/// An independently typed operand can be used directly in IR. A literal
/// needing context is elaborated after DSLX selects the callee's input types.
enum ImportOperand {
    Checked(TypedExpr),
    Contextual(NativeDataArg),
}

impl ImportOperand {
    fn into_evidence(self) -> NativeDataArg {
        match self {
            Self::Checked(value) => NativeDataArg::Typed(value.ty),
            Self::Contextual(arg) => arg,
        }
    }
}

impl BodyChecker<'_, '_> {
    /// Evaluate the explicit prefix in caller scope using the native
    /// declarations' types. DSLX binds defaults and infers the suffix.
    pub(super) fn imported_value_parametrics(
        &mut self,
        module: &str,
        function: &str,
        actuals: &[Expr],
    ) -> Result<Vec<BoundParametricValue>, String> {
        let types = self
            .resolver
            .explicit_parametric_types(module, function, actuals.len())?;
        self.structs.update_imports(self.resolver)?;
        actuals
            .iter()
            .zip(types)
            .enumerate()
            .map(|(i, (expr, ty))| {
                self.structs.validate_type(&ty)?;
                let value = self.parametric_value(expr, &ty).map_err(|e| {
                    format!("value parametric {} of '{module}::{function}': {e}", i + 1)
                })?;
                if self.replaying_body() {
                    self.charge_expansion(
                        "DSLX value parametrics",
                        crate::typed::constant_work(&value),
                    )?;
                }
                Ok(BoundParametricValue { ty, value })
            })
            .collect()
    }

    /// Materialize only the data arguments in IR. The native import ID
    /// identifies the selected implementation, including its value bindings.
    pub(super) fn apply_explicit_import(
        &mut self,
        module: &str,
        function: &str,
        values: Vec<BoundParametricValue>,
        args: Vec<TypedExpr>,
    ) -> Result<TypedExpr, String> {
        let types = args.iter().map(|arg| arg.ty.clone()).collect();
        let (id, ty) = self
            .resolver
            .resolve_explicit(module, function, types, values)?;
        self.structs.update_imports(self.resolver)?;
        self.structs.validate_type(&ty)?;
        Ok(TypedExpr {
            ty,
            kind: K::ImportedCall { id, args },
        })
    }

    /// Preserve literal evidence when an explicit binding can supply an
    /// operand's width, for example math::resize<8>(3).
    pub(super) fn explicit_import_call(
        &mut self,
        module: &str,
        function: &str,
        actuals: &[Expr],
        args: &[Expr],
    ) -> Result<TypedExpr, String> {
        let values = self.imported_value_parametrics(module, function, actuals)?;
        let args = args.iter().collect::<Vec<_>>();
        self.with_parametric_context(|checker| {
            checker.with_aggregate_operands(&args, |checker, args| {
                let mut evidence = Vec::with_capacity(args.len());
                let mut checked = Vec::with_capacity(args.len());
                for arg in args {
                    match checker.import_operand(arg, MAX_ELABORATION_DEPTH)? {
                        ImportOperand::Checked(value) => {
                            evidence.push(NativeDataArg::Typed(value.ty.clone()));
                            checked.push(Some(value));
                        }
                        ImportOperand::Contextual(witness) => {
                            evidence.push(witness);
                            checked.push(None);
                        }
                    }
                }
                if checker.replaying_body() {
                    let work = evidence
                        .iter()
                        .fold(0usize, |sum, arg| sum.saturating_add(arg.storage_work()));
                    checker.charge_expansion("DSLX call arguments", work)?;
                }
                let operands = if checked.iter().any(Option::is_none) {
                    let types = checker
                        .resolver
                        .contextual_data_types(module, function, &values, &evidence)?;
                    checker.structs.update_imports(checker.resolver)?;
                    args.iter()
                        .zip(checked)
                        .zip(types)
                        .map(|((arg, value), ty)| {
                            checker.structs.validate_type(&ty)?;
                            match value {
                                Some(value) if super::compatible(&value.ty, &ty) => Ok(value),
                                _ => checker.dslx_argument(arg, &ty),
                            }
                        })
                        .collect::<Result<_, _>>()?
                } else {
                    checked.into_iter().flatten().collect()
                };
                checker.apply_explicit_import(module, function, values, operands)
            })
        })
    }

    /// Represent the source edges that constrain a data operand's type.
    /// Concrete leaves become typed probe parameters. Literal leaves keep
    /// their minimum width until native inference has run.
    fn import_operand(&mut self, expr: &Expr, depth: usize) -> Result<ImportOperand, String> {
        let child = depth
            .checked_sub(1)
            .ok_or("import data argument nesting exceeds limit")?;
        if let Some((_, ty)) = self.reference_type(expr)
            && let Some(lookup) = self.dslx_widths()
            && let Some(witness) =
                import_bound_literal_witness(expr, ty, lookup, self.width_scope())?
        {
            return Ok(ImportOperand::Contextual(NativeDataArg::BoundLiteral(
                witness,
            )));
        }
        let group = |checker: &mut Self, items: &[Expr]| {
            items
                .iter()
                .map(|item| {
                    checker
                        .import_operand(item, child)
                        .map(ImportOperand::into_evidence)
                })
                .collect::<Result<Vec<_>, String>>()
        };
        let evidence = match expr {
            Expr::Tuple(items) => NativeDataArg::Tuple(group(self, items)?),
            Expr::Array(items) => NativeDataArg::Array {
                members: group(self, items)?,
                has_ellipsis: false,
            },
            Expr::ArrayLiteral {
                members,
                annotation: None,
                has_ellipsis,
            } => NativeDataArg::Array {
                members: group(self, members)?,
                has_ellipsis: *has_ellipsis,
            },
            _ if let Some(seed) = expr.unannotated_loop_seed()
                && has_aggregate_literal(seed) =>
            {
                // The caller's expected accumulator type reaches the seed as
                // well as the loop body during final elaboration.
                self.import_operand(seed, child)?.into_evidence()
            }
            Expr::If {
                then_value,
                else_value,
                ..
            } if has_aggregate_literal(expr) => NativeDataArg::Branches(vec![
                self.import_operand(then_value, child)?.into_evidence(),
                self.import_operand(else_value, child)?.into_evidence(),
            ]),
            Expr::Match { arms, .. } if has_aggregate_literal(expr) => NativeDataArg::Branches(
                arms.iter()
                    .map(|arm| {
                        self.import_operand(&arm.value, child)
                            .map(ImportOperand::into_evidence)
                    })
                    .collect::<Result<_, _>>()?,
            ),
            _ => match import_literal_witness(expr)? {
                Some(witness) => NativeDataArg::Inferred(witness),
                None => return self.expr(expr, None).map(ImportOperand::Checked),
            },
        };
        Ok(ImportOperand::Contextual(evidence))
    }
}
