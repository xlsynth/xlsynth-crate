// SPDX-License-Identifier: Apache-2.0

//! Elaborate explicit value arguments at the BX call site and ask the native
//! DSLX typechecker to select the concrete imported function.

use super::{BodyChecker, K, MAX_ELABORATION_DEPTH, TypedExpr, has_aggregate_literal};
use crate::ast::{Expr, ScopeStatement, Type};
use crate::dslx_import::NativeDataArg;
use crate::infer_literal::{
    ImportLiteralOrigins, has_contextual_result, import_array_witness,
    import_bound_literal_witness, import_literal_witness,
};
use crate::typed::BoundParametricValue;
use std::collections::BTreeMap;

/// An independently typed operand can be used directly in IR. A literal
/// needing context is elaborated after DSLX selects the callee's input types.
enum ImportOperand {
    Checked(TypedExpr),
    Contextual(NativeDataArg),
}

/// Which contextual constraints are being sought along an operand's result
/// edges.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ImportContext {
    Width,
    Value,
    ArrayElement,
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
    /// Native formals can constrain range bounds and unannotated array
    /// elements, including those retained through a let or tuple projection.
    pub(super) fn needs_contextual_import(&self, arg: &Expr) -> bool {
        self.contextual_import_in_scope(arg, &BTreeMap::new(), ImportContext::Value)
    }

    /// Track scoped aliases until preparation replaces them with checked
    /// bindings. Only a binding used by the result can constrain the formal.
    fn contextual_import_in_scope<'e>(
        &self,
        arg: &'e Expr,
        names: &BTreeMap<&'e str, bool>,
        context: ImportContext,
    ) -> bool {
        has_contextual_result(arg, &|expr| match expr {
            Expr::ConstIf { .. } | Expr::Range { .. } => Some(true),
            _ if context != ImportContext::Width
                && let Some((items, _)) = expr.unannotated_array() =>
            {
                Some(items.iter().any(|item| {
                    self.contextual_import_in_scope(item, names, ImportContext::ArrayElement)
                }))
            }
            Expr::Literal { ty: None, .. } => Some(context == ImportContext::ArrayElement),
            Expr::Scope { statements, result } => {
                let mut names = names.clone();
                for stmt in statements {
                    match stmt {
                        ScopeStatement::Let(binding) => {
                            let contextual = binding.ty.is_none()
                                && self.contextual_import_in_scope(&binding.value, &names, context);
                            binding.pattern.visit_names(&mut |name| {
                                names.insert(name, contextual);
                            });
                        }
                        ScopeStatement::Const(def) => {
                            let contextual = def.ty.is_none()
                                && self.contextual_import_in_scope(&def.value, &names, context);
                            names.insert(&def.name, contextual);
                        }
                        ScopeStatement::TypeAlias(_)
                        | ScopeStatement::ConstAssert(_)
                        | ScopeStatement::Expression(_) => { /* No new value binding. */ }
                    }
                }
                Some(self.contextual_import_in_scope(result, &names, context))
            }
            Expr::Name(name) => Some(names.get(name.as_str()).map_or_else(
                || {
                    self.inferred_bindings.get(name).is_some_and(|widths| {
                        context == ImportContext::ArrayElement
                            || widths.infers_array_member(expr, context != ImportContext::Width)
                    })
                },
                // Preparation will resolve the scoped initializer before the
                // probe decides which array elements supply literal evidence.
                |contextual| *contextual || context == ImportContext::ArrayElement,
            )),
            Expr::Index { value, .. } | Expr::Field { value, .. } => {
                Some(self.contextual_import_in_scope(value, names, context))
            }
            _ => None,
        })
    }

    fn inferred_reference<'e>(&'e self, expr: &'e Expr) -> Option<&'e Type> {
        let (binding, ty) = self.reference_type(expr)?;
        let widths = self.inferred_bindings.get(binding)?;
        (widths.infers_at_use(expr) || widths.infers_array_member(expr, true)).then_some(ty)
    }

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

    /// Let native formals or sibling arguments determine a contextual
    /// operand's width, including width evidence from compile-time branches.
    pub(super) fn contextual_import_call(
        &mut self,
        module: &str,
        function: &str,
        actuals: &[Expr],
        args: &[Expr],
    ) -> Result<TypedExpr, String> {
        let values = self.imported_value_parametrics(module, function, actuals)?;
        let source_args = args.iter().collect::<Vec<_>>();
        // Parametric formals, ranges and compile-time selections can supply
        // a width. Other fixed formals keep an ordinary binding's hardware
        // width, as they do when calling a BX-defined function.
        let infer_width = self.resolver.function_has_parametrics(module, function)
            || args.iter().any(|arg| {
                self.contextual_import_in_scope(arg, &BTreeMap::new(), ImportContext::Width)
            });
        self.with_argument_context(infer_width, |checker| {
            checker.with_aggregate_operands(&source_args, |checker, args| {
                let mut evidence = Vec::with_capacity(args.len());
                let mut checked = Vec::with_capacity(args.len());
                let mut origins = ImportLiteralOrigins::default();
                for (arg, source) in args.iter().zip(&source_args) {
                    match checker.import_operand(arg, MAX_ELABORATION_DEPTH, &mut origins)? {
                        ImportOperand::Checked(value) => {
                            evidence.push(NativeDataArg::Typed(value.ty.clone()));
                            checked.push(Some(value));
                        }
                        ImportOperand::Contextual(witness) => {
                            let witness = if let NativeDataArg::Inferred(literal) = &witness
                                && crate::infer_literal::supplies_selected_minimum(source)?
                            {
                                // Native DSLX uses a selected literal as an
                                // independent width witness until an earlier
                                // argument binds its width.
                                NativeDataArg::Inferred(format!(
                                    "const if true {{ {literal} }} else {{ false }}"
                                ))
                            } else {
                                witness
                            };
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
    fn import_operand(
        &mut self,
        expr: &Expr,
        depth: usize,
        origins: &mut ImportLiteralOrigins,
    ) -> Result<ImportOperand, String> {
        let child = depth
            .checked_sub(1)
            .ok_or("import data argument nesting exceeds limit")?;
        if let Some(ty @ (Type::Array { .. } | Type::Tuple(_))) = self.inferred_reference(expr) {
            return self
                .bound_aggregate_evidence(expr, ty, child, origins)
                .map(ImportOperand::Contextual);
        }
        if let Some((_, ty)) = self.reference_type(expr)
            && let Some(lookup) = self.dslx_widths()
            && let Some(witness) =
                import_bound_literal_witness(expr, ty, lookup, self.width_scope(), origins)?
        {
            return Ok(ImportOperand::Contextual(witness));
        }
        let mut group = |checker: &mut Self, items: &[Expr]| {
            items
                .iter()
                .map(|item| {
                    checker
                        .import_operand(item, child, origins)
                        .map(ImportOperand::into_evidence)
                })
                .collect::<Result<Vec<_>, String>>()
        };
        let evidence = match expr {
            Expr::Range {
                start,
                limit,
                inclusive,
            } => {
                let range = self.array_range_bounds(start, limit, *inclusive, None)?;
                self.structs
                    .validate_type(&Type::array(range.ty.clone(), range.count))?;
                if let Some(witness) = import_array_witness(
                    expr,
                    self.dslx_widths(),
                    self.width_scope(),
                    range.count,
                    range.widths.as_ref().map(|_| &range.ty),
                    origins,
                )? {
                    witness
                } else {
                    return self
                        .materialize_range(range, "array range")
                        .map(ImportOperand::Checked);
                }
            }
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
                && (has_aggregate_literal(seed) || self.needs_contextual_import(seed)) =>
            {
                // The caller's expected accumulator type reaches the seed as
                // well as the loop body during final elaboration.
                self.import_operand(seed, child, origins)?.into_evidence()
            }
            Expr::If {
                then_value,
                else_value,
                ..
            } if has_aggregate_literal(expr) || self.needs_contextual_import(expr) => {
                NativeDataArg::Branches(vec![
                    self.import_operand(then_value, child, origins)?
                        .into_evidence(),
                    self.import_operand(else_value, child, origins)?
                        .into_evidence(),
                ])
            }
            Expr::Match { arms, .. }
                if has_aggregate_literal(expr) || self.needs_contextual_import(expr) =>
            {
                NativeDataArg::Branches(
                    arms.iter()
                        .map(|arm| {
                            self.import_operand(&arm.value, child, origins)
                                .map(ImportOperand::into_evidence)
                        })
                        .collect::<Result<_, _>>()?,
                )
            }
            Expr::Scope { result, .. } if self.needs_contextual_import(result) => {
                self.import_operand(result, child, origins)?.into_evidence()
            }
            _ => {
                let Some(witness) = import_literal_witness(
                    expr,
                    self.dslx_widths(),
                    self.needs_contextual_import(expr),
                    origins,
                )?
                else {
                    return self.expr(expr, None).map(ImportOperand::Checked);
                };
                witness
            }
        };
        Ok(ImportOperand::Contextual(evidence))
    }

    /// Reconstruct the inferred leaves of an aggregate binding for native
    /// inference. All positions of an array select the same width constraint.
    fn bound_aggregate_evidence(
        &self,
        expr: &Expr,
        ty: &Type,
        depth: usize,
        origins: &mut ImportLiteralOrigins,
    ) -> Result<NativeDataArg, String> {
        let child = depth
            .checked_sub(1)
            .ok_or("import data argument nesting exceeds limit")?;
        if self.inferred_reference(expr).is_none() {
            return Ok(NativeDataArg::Typed(ty.clone()));
        }
        Ok(match ty {
            Type::Tuple(fields) => NativeDataArg::Tuple(
                fields
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| {
                        let field = Expr::Field {
                            value: Box::new(expr.clone()),
                            name: i.to_string(),
                        };
                        self.bound_aggregate_evidence(&field, ty, child, origins)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Type::Array { element, size } => {
                if let Some(witness) = import_array_witness(
                    expr,
                    self.dslx_widths(),
                    self.width_scope(),
                    *size,
                    None,
                    origins,
                )? {
                    witness
                } else if *size != 0 {
                    let lane = Expr::Index {
                        value: Box::new(expr.clone()),
                        index: Box::new(Expr::Bool(false)),
                    };
                    NativeDataArg::Repeated {
                        element: Box::new(
                            self.bound_aggregate_evidence(&lane, element, child, origins)?,
                        ),
                        size: *size,
                    }
                } else {
                    NativeDataArg::Typed(ty.clone())
                }
            }
            _ => {
                if let Some(lookup) = self.dslx_widths()
                    && let Some(witness) =
                        import_bound_literal_witness(expr, ty, lookup, self.width_scope(), origins)?
                {
                    witness
                } else {
                    NativeDataArg::Typed(ty.clone())
                }
            }
        })
    }
}
