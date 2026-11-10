// SPDX-License-Identifier: Apache-2.0

//! Infer a template's type arguments across all of its data inputs.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::ast::{Expr, Param, Type};
use crate::infer_literal::has_concrete_element_type;
use crate::typed::{TypedExpr, TypedExprKind as K};

use super::generic_blocks::Bindings;
use super::struct_parametrics::StructWitness;
use super::{
    BlockTemplate, BodyChecker, BoundCallback, BranchSelector, MAX_ELABORATION_DEPTH,
    MAX_TYPE_SIZE, MatchSelector, has_aggregate_literal,
};

/// Concrete types, callbacks, and checked port values for one structural call.
pub(super) struct BlockArguments<'m> {
    pub(super) bindings: Bindings,
    pub(super) callbacks: Vec<Rc<BoundCallback<'m>>>,
    pub(super) values: Vec<TypedExpr>,
}

/// A call argument after its explicitly typed operands have supplied bindings.
pub(super) enum Input<'e> {
    Checked {
        expr: Option<&'e Expr>,
        value: TypedExpr,
    },
    Contextual(&'e Expr),
    Aggregate {
        expr: &'e Expr,
        members: Vec<Input<'e>>,
    },
    Branches {
        expr: &'e Expr,
        selector: Option<BranchSelector>,
        results: Vec<Input<'e>>,
    },
}

/// A function may have already checked this operand to infer a dimension.
/// Blocks enter inference with their nominal types and dimensions resolved.
pub(super) enum InputPattern {
    Block,
    Function(Option<TypedExpr>),
}

impl InputPattern {
    fn child(&self, value: Option<TypedExpr>) -> Self {
        match self {
            Self::Block => Self::Block,
            Self::Function(_) => Self::Function(value),
        }
    }

    /// Preserve the checked members when a dimension witness is a constructor.
    fn take_members(&mut self) -> std::vec::IntoIter<TypedExpr> {
        let members = match self {
            Self::Function(value) => match value.take().map(|v| v.kind) {
                Some(K::Tuple(members) | K::Array(members)) => members,
                _ => Vec::new(),
            },
            Self::Block => Vec::new(),
        };
        members.into_iter()
    }

    /// Reuse the selector and results when a conditional has already
    /// supplied a whole-argument dimension witness.
    fn take_branches(&mut self) -> (Option<BranchSelector>, std::vec::IntoIter<TypedExpr>) {
        let Self::Function(value) = self else {
            return (None, Vec::new().into_iter());
        };
        let (selector, results) = match value.take().map(|v| v.kind) {
            Some(K::If {
                condition,
                then_value,
                else_value,
            }) => (
                Some(BranchSelector::If(*condition)),
                vec![*then_value, *else_value],
            ),
            Some(K::Match { subject, arms }) => {
                let (patterns, results) = arms
                    .into_iter()
                    .map(|arm| (arm.patterns, arm.value))
                    .unzip();
                (
                    Some(BranchSelector::Match(MatchSelector {
                        subject: *subject,
                        patterns,
                    })),
                    results,
                )
            }
            _ => (None, Vec::new()),
        };
        (selector, results.into_iter())
    }
}

impl<'e> Input<'e> {
    /// Collect partially typed constructor arguments for each unbound
    /// parameter.
    fn collect_constructors<'a>(
        &'a mut self,
        formal: &'a Type,
        bindings: &Bindings,
        groups: &mut BTreeMap<&'a str, Vec<(&'e Expr, &'a mut Self)>>,
    ) {
        if let (Type::Named(name), Self::Contextual(expr)) = (formal, &*self)
            && bindings.declares(name)
            && bindings.get(name).is_none()
            && matches!(expr, Expr::Tuple(_) | Expr::Array(_))
        {
            groups.entry(name).or_default().push((*expr, self));
            return;
        }
        match (formal, self) {
            (ty, Self::Branches { results, .. }) => {
                for result in results {
                    result.collect_constructors(ty, bindings, groups);
                }
            }
            (Type::Tuple(types), Self::Aggregate { members, .. }) => {
                for (member, ty) in members.iter_mut().zip(types) {
                    member.collect_constructors(ty, bindings, groups);
                }
            }
            (
                Type::Array { element, .. } | Type::ParametricArray { element, .. },
                Self::Aggregate { members, .. },
            ) => {
                for member in members {
                    member.collect_constructors(element, bindings, groups);
                }
            }
            _ => { /* Already checked values and fixed formals need no joint inference. */ }
        }
    }
}

impl<'m> BodyChecker<'_, 'm> {
    /// Infer the template's type parameters before checking contextual
    /// literals.
    pub(super) fn generic_inputs(
        &mut self,
        template: &BlockTemplate<'m>,
        args: &[Expr],
        callbacks: &[String],
    ) -> Result<BlockArguments<'m>, String> {
        let target = &template.block.name;
        let formals = &template.params;
        if formals.len() != args.len() {
            return Err(format!(
                "block '{target}' expects {} data arguments, got {}",
                formals.len(),
                args.len()
            ));
        }
        let previous = self.checking_callback_inputs;
        self.checking_callback_inputs |= !callbacks.is_empty();
        let call = self.with_scope_aliases(|checker| {
            let prepared = formals
                .iter()
                .zip(args)
                .map(|(p, arg)| {
                    checker
                        .prepare_value(arg)
                        .map_err(|e| format!("input '{}', call to block '{target}': {e}", p.name))
                })
                .collect::<Result<Vec<_>, _>>()?;
            checker.infer_prepared_inputs(
                template,
                prepared.iter().map(|arg| arg.as_ref()),
                callbacks,
            )
        });
        self.checking_callback_inputs = previous;
        call
    }

    /// Infer and elaborate every port while references captured from scoped
    /// match arms still carry their original value and literal constraints.
    fn infer_prepared_inputs<'e>(
        &mut self,
        template: &BlockTemplate<'m>,
        args: impl Iterator<Item = &'e Expr>,
        actuals: &[String],
    ) -> Result<BlockArguments<'m>, String> {
        let target = template.block.name.as_str();
        let formals = &template.params;
        let type_params = &template.block.type_params;
        let input_error =
            |p: &Param, e: String| format!("input '{}', call to block '{target}': {e}", p.name);
        let mut bindings = Bindings::new(type_params);
        let mut inputs = Vec::with_capacity(formals.len());
        for (p, arg) in formals.iter().zip(args) {
            inputs.push(
                self.infer_input(&p.ty, arg, &mut bindings, InputPattern::Block)
                    .map_err(|e| input_error(p, e))?,
            );
        }
        self.infer_input_groups(
            &mut inputs,
            formals,
            &mut bindings,
            &format!("block '{target}'"),
        )?;
        let mut callbacks = Vec::with_capacity(actuals.len());
        for (index, (formal, actual)) in template.callback_params.iter().zip(actuals).enumerate() {
            let error = |e: String| {
                format!(
                    "callable parameter '{}' of block '{target}': {e}",
                    formal.signature.name
                )
            };
            let callback = self
                .resolve_block_callback(actual, target, index, formal, &mut bindings)
                .map_err(error)?;
            callbacks.push(callback);
        }
        for name in type_params {
            if bindings.get(name).is_none() {
                return Err(format!(
                    "cannot infer type parameter '{name}' in call to block '{target}'; give a data argument an explicit type"
                ));
            }
        }
        let values = inputs
            .into_iter()
            .zip(formals)
            .map(|(input, p)| {
                let ty = self
                    .bound_block_type(&p.ty, &bindings)
                    .map_err(|e| input_error(p, e))?;
                self.finish_input(input, &ty).map_err(|e| input_error(p, e))
            })
            .collect::<Result<_, _>>()?;
        Ok(BlockArguments {
            bindings,
            callbacks,
            values,
        })
    }

    /// Combine evidence from complementary tuple fields or array elements
    /// across arguments that share a type variable.
    pub(super) fn infer_input_groups(
        &mut self,
        inputs: &mut [Input<'_>],
        formals: &[Param],
        bindings: &mut Bindings,
        callee: &str,
    ) -> Result<(), String> {
        let mut groups = BTreeMap::new();
        for (input, p) in inputs.iter_mut().zip(formals) {
            input.collect_constructors(&p.ty, bindings, &mut groups);
        }
        for (name, constructors) in groups {
            let exprs: Vec<_> = constructors.iter().map(|(expr, _)| *expr).collect();
            if let Some(checked) = self
                .infer_constructors(&exprs, /* infer_literal_widths= */ false)
                .map_err(|e| format!("type parameter '{name}', call to {callee}: {e}"))?
            {
                bindings.match_input(&Type::Named(name.into()), &checked[0].ty, self.structs)?;
                for ((expr, input), value) in constructors.into_iter().zip(checked) {
                    *input = Input::Checked {
                        expr: Some(expr),
                        value,
                    };
                }
            }
        }
        Ok(())
    }

    /// Obtain a dimension witness from the whole constructor type when its
    /// fields can determine that type without the callee's context.
    pub(super) fn constructor_dimension_witness(
        &mut self,
        expr: &Expr,
    ) -> Result<Option<TypedExpr>, String> {
        self.infer_constructors(&[expr], /* infer_literal_widths= */ true)
            .map(|values| values.and_then(|mut values| values.pop()))
    }

    /// Unify corresponding fields against concrete siblings. A type
    /// parameter needs a typed witness. A dimension can use minimum widths.
    fn infer_constructors(
        &mut self,
        items: &[&Expr],
        infer_literal_widths: bool,
    ) -> Result<Option<Vec<TypedExpr>>, String> {
        if items.iter().any(|expr| matches!(expr, Expr::Scope { .. })) {
            return self.with_scoped_results(items, |checker, prepared| {
                checker.infer_constructors(prepared, infer_literal_widths)
            });
        }
        if items
            .iter()
            .any(|expr| matches!(expr, Expr::If { .. } | Expr::Match { .. }))
            && items
                .iter()
                .any(|expr| self.contains_type_struct_constructor(expr))
        {
            let (selectors, branches) = self.result_branches(items)?;
            return Ok(self
                .infer_constructors(&branches, infer_literal_widths)?
                .map(|values| Self::rebuild_branch_results(selectors, values)));
        }
        if let Some(Expr::Tuple(first)) = items.first()
            && let Some(tuples) = items
                .iter()
                .map(|item| match item {
                    Expr::Tuple(fields) if fields.len() == first.len() => Some(fields),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        {
            let mut values = vec![Vec::new(); tuples.len()];
            let mut size = 1usize;
            for index in 0..first.len() {
                let fields: Vec<_> = tuples.iter().map(|tuple| &tuple[index]).collect();
                let Some(checked) = self.infer_constructors(&fields, infer_literal_widths)? else {
                    return Ok(None);
                };
                // Reject an over-budget tuple as it grows, before copying the
                // remaining members' types into every occurrence.
                let shape = self
                    .structs
                    .type_shape(&checked[0].ty, MAX_ELABORATION_DEPTH - 1)?;
                size = size.saturating_add(shape.size);
                if size > MAX_TYPE_SIZE {
                    return Err(format!(
                        "aggregate exceeds maximum concrete type size {MAX_TYPE_SIZE}"
                    ));
                }
                for (tuple, member) in values.iter_mut().zip(checked) {
                    tuple.push(member);
                }
            }
            let checked = items
                .iter()
                .zip(values)
                .map(|(expr, members)| {
                    let ty = Type::Tuple(members.iter().map(|m| m.ty.clone()).collect());
                    self.finish_expr(expr, None, ty, K::Tuple(members))
                })
                .collect::<Result<_, _>>()?;
            return Ok(Some(checked));
        }
        if let Some(Expr::Array(first)) = items.first()
            && let Some(arrays) = items
                .iter()
                .map(|item| match item {
                    Expr::Array(elements) if elements.len() == first.len() => Some(elements),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        {
            let elements: Vec<_> = arrays.into_iter().flatten().collect();
            let Some(checked) = self.infer_constructors(&elements, infer_literal_widths)? else {
                return Ok(None);
            };
            let ty = Type::Array {
                element: Box::new(checked[0].ty.clone()),
                size: first.len(),
            };
            self.structs.validate_type(&ty)?;
            let mut values = checked.into_iter();
            let arrays = items
                .iter()
                .map(|expr| {
                    let members = values.by_ref().take(first.len()).collect();
                    self.finish_expr(expr, None, ty.clone(), K::Array(members))
                })
                .collect::<Result<_, _>>()?;
            return Ok(Some(arrays));
        }
        let mut deferred = false;
        for (index, item) in items.iter().enumerate() {
            if has_concrete_element_type(item, self.common_widths())? {
                let anchor = match self.type_struct_witness(item)? {
                    StructWitness::Other => self.expr(item, None)?,
                    StructWitness::Deferred => {
                        deferred = true;
                        continue;
                    }
                    StructWitness::Checked(value) => value,
                };
                return self.check_siblings(items, index, anchor).map(Some);
            }
        }
        if infer_literal_widths && !deferred && !items.is_empty() {
            return self.unify_values(items).map(Some);
        }
        Ok(None)
    }

    /// Descend matching constructors so each field can supply or receive its
    /// own type, even when another field or port has an unannotated literal.
    pub(super) fn infer_input<'e>(
        &mut self,
        formal: &Type,
        arg: &'e Expr,
        bindings: &mut Bindings,
        mut pattern: InputPattern,
    ) -> Result<Input<'e>, String> {
        if matches!(pattern, InputPattern::Function(None)) && !bindings.has_unbound_variable(formal)
        {
            // Bound formals give their arguments context for literal widths.
            return Ok(Input::Contextual(arg));
        }
        let branches = matches!(arg, Expr::If { .. } | Expr::Match { .. });
        let partial_constructor = branches
            && !matches!(pattern, InputPattern::Function(Some(_)))
            && (self.contains_type_struct_constructor(arg)
                || (has_aggregate_literal(arg)
                    && !has_concrete_element_type(arg, self.common_widths())?));
        if branches
            && !partial_constructor
            && matches!(pattern, InputPattern::Function(None))
            && self.contains_inferred_aggregate(arg)
        {
            // Keep the conditional's common-type evidence for unbound
            // variables, then check its projections with the final types.
            pattern = InputPattern::Function(Some(self.expr(arg, None)?));
        }
        if branches && (matches!(pattern, InputPattern::Function(Some(_))) || partial_constructor) {
            let (selector, mut checked) = pattern.take_branches();
            match arg {
                Expr::If {
                    then_value,
                    else_value,
                    ..
                } => {
                    let results = vec![
                        self.infer_input(
                            formal,
                            then_value,
                            bindings,
                            pattern.child(checked.next()),
                        )?,
                        self.infer_input(
                            formal,
                            else_value,
                            bindings,
                            pattern.child(checked.next()),
                        )?,
                    ];
                    return Ok(Input::Branches {
                        expr: arg,
                        selector,
                        results,
                    });
                }
                Expr::Match { arms, .. } => {
                    let results = self.with_dslx_value_context(|checker| {
                        arms.iter()
                            .map(|arm| {
                                checker.infer_input(
                                    formal,
                                    &arm.value,
                                    bindings,
                                    pattern.child(checked.next()),
                                )
                            })
                            .collect::<Result<_, _>>()
                    })?;
                    return Ok(Input::Branches {
                        expr: arg,
                        selector,
                        results,
                    });
                }
                _ => { /* An ordinary aggregate has a single result constructor below. */ }
            }
        }
        let members = match (formal, arg) {
            (Type::Tuple(types), Expr::Tuple(items)) => {
                if types.len() != items.len() {
                    return Err(format!(
                        "expected {} tuple fields, got {}",
                        types.len(),
                        items.len()
                    ));
                }
                let mut checked = pattern.take_members();
                types
                    .iter()
                    .zip(items)
                    .map(|(ty, item)| {
                        self.infer_input(ty, item, bindings, pattern.child(checked.next()))
                    })
                    .collect::<Result<_, _>>()?
            }
            (
                Type::Array { element, .. } | Type::ParametricArray { element, .. },
                Expr::Array(items),
            ) => {
                if let Type::Array { size, .. } = formal
                    && items.len() != *size
                {
                    return Err(format!("expected array length {size}, got {}", items.len()));
                }
                if items.is_empty() {
                    // The checker constructs an empty IR array from its element
                    // type.
                    return Ok(Input::Contextual(arg));
                }
                let mut checked = pattern.take_members();
                items
                    .iter()
                    .map(|item| {
                        self.infer_input(element, item, bindings, pattern.child(checked.next()))
                    })
                    .collect::<Result<_, _>>()?
            }
            (Type::Bits { .. } | Type::Bool, _)
                if !matches!(pattern, InputPattern::Function(Some(_))) =>
            {
                return Ok(Input::Contextual(arg));
            }
            (Type::Named(name), _)
                if !bindings.declares(name)
                    && !matches!(pattern, InputPattern::Function(Some(_))) =>
            {
                return Ok(Input::Contextual(arg));
            }
            (Type::TypeApplied { .. }, _) if !bindings.has_unbound_variable(formal) => {
                return Ok(Input::Contextual(arg));
            }
            _ => {
                if matches!(arg, Expr::Tuple(_) | Expr::Array(_))
                    && !matches!(pattern, InputPattern::Function(Some(_)))
                    && self.contains_type_struct_constructor(arg)
                {
                    // A whole-aggregate type variable may receive its type
                    // from another argument or from the constructor group.
                    return Ok(Input::Contextual(arg));
                }
                if !has_concrete_element_type(arg, self.common_widths())? {
                    return Ok(Input::Contextual(arg));
                }
                let checked = match &mut pattern {
                    InputPattern::Block => None,
                    InputPattern::Function(value) => value.take(),
                };
                let value = if let Some(checked) = checked {
                    checked
                } else {
                    match self.type_struct_witness(arg)? {
                        StructWitness::Other => self.expr(arg, None)?,
                        StructWitness::Deferred => return Ok(Input::Contextual(arg)),
                        StructWitness::Checked(value) => value,
                    }
                };
                match pattern {
                    InputPattern::Block => {
                        bindings.match_input(formal, &value.ty, self.structs)?;
                    }
                    InputPattern::Function(_) => {
                        let contextual_projection =
                            matches!(arg, Expr::Name(_) | Expr::Field { .. } | Expr::Index { .. })
                                && self.contains_inferred_aggregate(arg);
                        bindings.infer_source_type(
                            formal,
                            &value.ty,
                            self.structs,
                            contextual_projection,
                        )?;
                    }
                }
                return Ok(Input::Checked {
                    expr: Some(arg),
                    value,
                });
            }
        };
        Ok(Input::Aggregate { expr: arg, members })
    }

    /// Assemble constructors and check previously unsized operands once the
    /// full interface is known, reusing the values that supplied its types.
    pub(super) fn finish_input(&mut self, input: Input, ty: &Type) -> Result<TypedExpr, String> {
        let (expr, members) = match input {
            Input::Checked { value, .. } if super::compatible(ty, &value.ty) => return Ok(value),
            Input::Checked {
                expr: Some(expr),
                value,
            } => {
                return self.finish_expr(expr, Some(ty), value.ty, value.kind);
            }
            Input::Checked { expr: None, value } => {
                return Err(format!("expected {ty}, got {}", value.ty));
            }
            Input::Contextual(expr) => return self.expr(expr, Some(ty)),
            Input::Aggregate { expr, members } => (expr, members),
            Input::Branches {
                expr,
                selector,
                results,
            } => {
                return match expr {
                    Expr::If { condition, .. } => {
                        let condition = match selector {
                            Some(BranchSelector::If(value)) => value,
                            None => self.expr(condition, Some(&Type::Bool))?,
                            _ => unreachable!("if selector"),
                        };
                        let mut results = results.into_iter();
                        let then_value =
                            self.finish_input(results.next().expect("then branch"), ty)?;
                        let else_value =
                            self.finish_input(results.next().expect("else branch"), ty)?;
                        Ok(TypedExpr {
                            ty: ty.clone(),
                            kind: K::If {
                                condition: Box::new(condition),
                                then_value: Box::new(then_value),
                                else_value: Box::new(else_value),
                            },
                        })
                    }
                    Expr::Match { subject, arms } => self.with_dslx_value_context(|checker| {
                        let selector = match selector {
                            Some(BranchSelector::Match(selector)) => selector,
                            None => checker.match_selector(subject, arms)?,
                            _ => unreachable!("match selector"),
                        };
                        let values = results
                            .into_iter()
                            .map(|v| checker.finish_input(v, ty))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(selector.with_values(&mut values.into_iter()))
                    }),
                    _ => unreachable!("branches are collected from conditionals and matches"),
                };
            }
        };
        if let Type::Array { size, .. } = ty
            && members.len() != *size
        {
            return Err(format!(
                "expected array length {size}, got {}",
                members.len()
            ));
        }
        let kind = match ty {
            Type::Tuple(types) => K::Tuple(
                members
                    .into_iter()
                    .zip(types)
                    .map(|(member, ty)| self.finish_input(member, ty))
                    .collect::<Result<_, _>>()?,
            ),
            Type::Array { element, .. } => K::Array(
                members
                    .into_iter()
                    .map(|member| self.finish_input(member, element))
                    .collect::<Result<_, _>>()?,
            ),
            _ => unreachable!("an aggregate formal retains its constructor after substitution"),
        };
        self.finish_expr(expr, Some(ty), ty.clone(), kind)
    }
}
