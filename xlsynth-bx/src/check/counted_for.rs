// SPDX-License-Identifier: Apache-2.0

//! Check a counted fold once with runtime bindings for the element and carry.

use super::{BodyChecker, CheckedBinding, CheckedStmt, K, Param, Type, TypedExpr, compatible};
use crate::ast::{Expr, ForIterable, ForLoop};
use crate::infer_literal::{BindingWidths, Member, binding_widths, minimum_type};
use crate::typed::{BoundParametricValue, TypedFor, TypedIterable};
use std::collections::BTreeSet;

/// Static iteration shape and the value from which the loop reads its elements.
pub(super) struct Iteration {
    pub(super) ty: Type,
    pub(super) widths: Option<BindingWidths>,
    pub(super) iterable: TypedIterable,
    pub(super) count: usize,
}

/// Carry type evidence from an expected type, a struct update's partial
/// bindings, or a seed already elaborated while unifying sibling expressions.
pub(super) enum CarryContext<'a> {
    Expected(Option<&'a Type>),
    Prepared(TypedExpr),
    StructPrefix {
        name: &'a str,
        args: &'a [BoundParametricValue],
    },
}

impl BodyChecker<'_, '_> {
    /// Elaborate the carry once, applying the loop annotation before any
    /// partial struct bindings supplied by its enclosing update.
    pub(super) fn loop_seed(
        &mut self,
        seed: &Expr,
        annotation: Option<&Type>,
        context: CarryContext,
        check: impl FnOnce(&mut Self, Option<&Type>) -> Result<TypedExpr, String>,
    ) -> Result<TypedExpr, String> {
        let expected = match context {
            CarryContext::Expected(ty) => ty,
            _ => None,
        };
        let seed = match context {
            CarryContext::Prepared(seed) => seed,
            CarryContext::StructPrefix { name, args } if annotation.is_none() => {
                self.splat_base_with_prefix(seed, name, args)?
            }
            _ => check(self, annotation.or(expected))?,
        };
        if let Some(expected) = expected
            && !compatible(&seed.ty, expected)
        {
            return Err(format!("expected {expected}, got {}", seed.ty));
        }
        Ok(seed)
    }

    /// Resolve the iterable and seed outside the binder scope. The body must
    /// produce the carry type even when the loop will execute zero times.
    pub(super) fn counted_for(
        &mut self,
        fold: &ForLoop,
        context: CarryContext,
    ) -> Result<TypedExpr, String> {
        self.validate_counted_binders(fold)?;
        let annotation = fold
            .annotation
            .as_ref()
            .map(|(e, a)| Ok::<_, String>((self.source_type(e)?, self.source_type(a)?)))
            .transpose()?;
        let element_hint = annotation.as_ref().map(|(e, _)| e);
        let iteration = self.counted_iteration(&fold.iterable, element_hint)?;
        let seed = self.loop_seed(
            &fold.seed,
            annotation.as_ref().map(|(_, a)| a),
            context,
            |checker, carry_hint| {
                checker.with_prepared_value(&fold.seed, |checker, seed| {
                    if carry_hint.is_none() && minimum_type(seed)?.is_some() {
                        return Err(
                            "for needs a typed seed or accumulator for an implicit literal result"
                                .into(),
                        );
                    }
                    checker.elaborate_result(seed, carry_hint, false)
                })
            },
        )?;
        let id = self.next_counted_loop;
        self.next_counted_loop += 1;
        let element = Param {
            name: format!("$for_element_{id}"),
            ty: iteration.ty,
        };
        let accumulator = Param {
            name: format!("$for_carry_{id}"),
            ty: seed.ty.clone(),
        };
        let (bindings, output) =
            self.counted_body(fold, &element, iteration.widths, &accumulator)?;
        Ok(TypedExpr {
            ty: seed.ty.clone(),
            kind: K::For(Box::new(TypedFor::new(
                element,
                accumulator,
                iteration.iterable,
                iteration.count,
                seed,
                bindings,
                output,
            ))),
        })
    }

    /// Check an iterable once, including when a scoped seed has already made
    /// it available to inference through a private lexical alias.
    pub(super) fn counted_iteration(
        &mut self,
        iterable: &ForIterable,
        hint: Option<&Type>,
    ) -> Result<Iteration, String> {
        match iterable {
            ForIterable::Range {
                start,
                limit,
                inclusive,
            } => self.counted_range(start, limit, *inclusive, hint),
            ForIterable::Array(Expr::Name(name))
                if let Some(iteration) = self.prepared_counted_iterables.remove(name) =>
            {
                Ok(iteration)
            }
            ForIterable::Array(value) => self.counted_array(value, hint),
        }
    }

    /// Both patterns introduce names in the same lexical scope.
    pub(super) fn validate_counted_binders(&self, fold: &ForLoop) -> Result<(), String> {
        let mut names = BTreeSet::new();
        let mut duplicate = None;
        for pattern in [&fold.element, &fold.accumulator] {
            self.validate_source_pattern(pattern)?;
            pattern.visit_names(&mut |name| {
                if !names.insert(name.to_owned()) {
                    duplicate = Some(name.to_owned());
                }
            });
        }
        if let Some(name) = duplicate {
            return Err(format!("duplicate loop binding '{name}'"));
        }
        Ok(())
    }

    /// Keep the full-width starting value. Only the number of iterations has
    /// to fit XLS IR's nonnegative int64 trip_count.
    fn counted_range(
        &mut self,
        start: &Expr,
        limit: &Expr,
        inclusive: bool,
        hint: Option<&Type>,
    ) -> Result<Iteration, String> {
        self.with_scoped_results(&[start, limit], |checker, bounds| {
            let range = checker.static_range(bounds[0], bounds[1], hint, inclusive, "for")?;
            if i64::try_from(range.count).is_err() {
                return Err("for range trip count exceeds XLS IR limits".into());
            }
            Ok(Iteration {
                ty: range.ty,
                widths: range.widths,
                iterable: TypedIterable::Range { start: range.first },
                count: range.count,
            })
        })
    }

    /// Use the binder's element annotation as context for array constructors.
    /// A signal, function call, or projection supplies its own array length.
    fn counted_array(&mut self, expr: &Expr, hint: Option<&Type>) -> Result<Iteration, String> {
        self.with_prepared_value(expr, |checker, expr| {
            let length = checker.iterable_length(expr, &[])?;
            let array_hint = hint.zip(length).map(|(ty, size)| Type::Array {
                element: Box::new(ty.clone()),
                size,
            });
            let value = checker.elaborate_result(expr, array_hint.as_ref(), false)?;
            let Type::Array { element, size } = &value.ty else {
                return Err(format!(
                    "for iterable must be an array or range, got {}",
                    value.ty
                ));
            };
            if let Some(hint) = hint
                && !compatible(hint, element)
            {
                return Err(format!("for element expects {hint}, got {element}"));
            }
            let widths = if hint.is_none() {
                binding_widths(expr, &value.ty, &checker.inferred_bindings)?
                    .and_then(BindingWidths::array_element)
            } else {
                None
            };
            Ok(Iteration {
                ty: element.as_ref().clone(),
                widths,
                count: *size,
                iterable: TypedIterable::Array(value),
            })
        })
    }

    /// Recover the length of the selected array without evaluating its source
    /// or indices. Its complete contextual type can then flow through ordinary
    /// literal, projection, and loop seed checking.
    fn iterable_length(&self, expr: &Expr, path: &[Member]) -> Result<Option<usize>, String> {
        match expr {
            Expr::Array(items) if path.is_empty() && items.is_empty() => {
                Err("for cannot infer the element type of an empty array literal".into())
            }
            Expr::Array(items) if path.is_empty() => Ok(Some(items.len())),
            Expr::Array(items) if let [Member::ArrayElement, rest @ ..] = path => {
                items.iter().try_fold(None, |size, item| {
                    Ok(size.or(self.iterable_length(item, rest)?))
                })
            }
            Expr::Tuple(fields)
                if let [Member::TupleField(i), rest @ ..] = path
                    && let Some(field) = fields.get(*i) =>
            {
                self.iterable_length(field, rest)
            }
            Expr::Field { value, name } if let Ok(i) = name.parse() => {
                let mut source_path = vec![Member::TupleField(i)];
                source_path.extend_from_slice(path);
                self.iterable_length(value, &source_path)
            }
            Expr::Index { value, .. } => {
                let mut source_path = vec![Member::ArrayElement];
                source_path.extend_from_slice(path);
                self.iterable_length(value, &source_path)
            }
            Expr::For(fold) if fold.annotation.is_none() => self.iterable_length(&fold.seed, path),
            Expr::If {
                then_value,
                else_value,
                ..
            } => Ok(self
                .iterable_length(then_value, path)?
                .or(self.iterable_length(else_value, path)?)),
            Expr::Match { arms, .. } => arms.iter().try_fold(None, |length, arm| {
                Ok(length.or(self.iterable_length(&arm.value, path)?))
            }),
            _ => {
                let ty = self.reference_type(expr).and_then(|(_, ty)| {
                    path.iter().try_fold(ty, |ty, member| match (ty, member) {
                        (Type::Tuple(fields), Member::TupleField(i)) => fields.get(*i),
                        (Type::Array { element, .. }, Member::ArrayElement) => {
                            Some(element.as_ref())
                        }
                        _ => None,
                    })
                });
                Ok(match ty {
                    Some(Type::Array { size, .. }) => Some(*size),
                    _ => None,
                })
            }
        }
    }

    /// Isolate body-local lets from the containing graph. Stable internal names
    /// let lowering capture outer values even when source binders shadow them.
    fn counted_body(
        &mut self,
        fold: &ForLoop,
        element: &Param,
        element_widths: Option<BindingWidths>,
        accumulator: &Param,
    ) -> Result<(Vec<CheckedBinding>, TypedExpr), String> {
        let start = self.statements.len();
        let outer_counted_body = std::mem::replace(&mut self.in_counted_body, true);
        let mut saved = Vec::new();
        let name = |param: &Param| TypedExpr {
            ty: param.ty.clone(),
            kind: K::Name(param.name.clone()),
        };
        let output = (|| {
            self.bind_pattern(&fold.element, name(element), element_widths, &mut saved)?;
            self.bind_pattern(&fold.accumulator, name(accumulator), None, &mut saved)?;
            self.scoped_result(&fold.body, Some(&accumulator.ty), false)
        })();
        self.leave_scope(saved);
        self.in_counted_body = outer_counted_body;
        let bindings = self.take_pure_bindings(start);
        Ok((bindings, output?))
    }

    /// Move the lets belonging to one generated function out of the caller.
    pub(super) fn take_pure_bindings(&mut self, start: usize) -> Vec<CheckedBinding> {
        self.statements
            .drain(start..)
            .map(|stmt| {
                let CheckedStmt::Let { name, value } = stmt else {
                    unreachable!("pure bodies only emit lexical let bindings")
                };
                self.constants.remove(&name);
                self.constexpr_names.remove(&name);
                CheckedBinding { name, value }
            })
            .collect()
    }
}
