// SPDX-License-Identifier: Apache-2.0

//! Infer a template's type arguments across all of its data inputs.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::ast::{Expr, Param, Type};
use crate::typed::{BoundParametricValue, TypedExpr, TypedExprKind as K};

use super::generic_blocks::Bindings;
use super::struct_parametrics::StructWitness;
use super::{
    BlockParametricArguments, BlockSpecialization, BlockTemplate, BodyChecker, BranchSelector,
    MAX_ELABORATION_DEPTH, MAX_TYPE_SIZE, MatchSelector, TypeResolver, has_aggregate_literal,
};

/// A block's specialized interface and the data arguments checked against it.
pub(super) struct BlockArguments<'m> {
    pub(super) specialization: Rc<BlockSpecialization<'m>>,
    pub(super) inputs: Vec<TypedExpr>,
}

/// A call argument after its explicitly typed operands have supplied bindings.
pub(super) enum Input<'e> {
    Checked {
        expr: Option<&'e Expr>,
        value: TypedExpr,
    },
    Contextual {
        expr: &'e Expr,
        prepared: Option<TypedExpr>,
    },
    Aggregate {
        expr: &'e Expr,
        members: Vec<Input<'e>>,
        /// Implicit operands already budgeted while preparing this array.
        charged_fill: usize,
    },
    Branches {
        expr: &'e Expr,
        selector: Option<BranchSelector>,
        results: Vec<Input<'e>>,
    },
}

/// A resolved port can check its concrete shape during inference. A source
/// annotation may still need dimensions, with a checked operand as a witness.
pub(super) enum InputPattern {
    Block,
    Function(Option<TypedExpr>),
}

impl InputPattern {
    /// Defer to the final formal type while retaining any dimension witness.
    fn contextual(self, expr: &Expr) -> Input<'_> {
        let prepared = match self {
            Self::Block => None,
            Self::Function(value) => value,
        };
        Input::Contextual { expr, prepared }
    }

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
    /// Map callbacks supply a checked element directly and have no source
    /// operand.
    pub(super) fn source(&self) -> Option<&'e Expr> {
        match self {
            Self::Checked { expr, .. } => *expr,
            Self::Contextual { expr, .. }
            | Self::Aggregate { expr, .. }
            | Self::Branches { expr, .. } => Some(expr),
        }
    }

    /// Visit an argument's leaves in source order with their corresponding
    /// formal patterns. Rows and conditional results share their pattern.
    pub(super) fn visit_leaves<'a>(
        &'a mut self,
        formal: &'a Type,
        visit: &mut impl FnMut(&'a Type, &'a mut Self) -> Result<(), String>,
    ) -> Result<(), String> {
        match (formal, self) {
            (ty, Self::Branches { results, .. }) => {
                for result in results {
                    result.visit_leaves(ty, visit)?;
                }
            }
            (Type::Tuple(types), Self::Aggregate { members, .. }) => {
                for (member, ty) in members.iter_mut().zip(types) {
                    member.visit_leaves(ty, visit)?;
                }
            }
            (
                Type::Array { element, .. } | Type::ParametricArray { element, .. },
                Self::Aggregate { members, .. },
            ) => {
                for member in members {
                    member.visit_leaves(element, visit)?;
                }
            }
            (ty, input) => visit(ty, input)?,
        }
        Ok(())
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
        supplied: Vec<BoundParametricValue>,
    ) -> Result<BlockArguments<'m>, String> {
        let target = &template.block.name;
        let formals = &template.ports.params;
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
                    let prepared = if checker.dslx_value_context {
                        checker.prepare_dslx_argument(arg)
                    } else {
                        checker.prepare_value(arg)
                    };
                    prepared
                        .map_err(|e| format!("input '{}', call to block '{target}': {e}", p.name))
                })
                .collect::<Result<Vec<_>, _>>()?;
            checker.infer_prepared_inputs(
                template,
                prepared.iter().map(|arg| arg.as_ref()),
                callbacks,
                supplied,
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
        supplied: Vec<BoundParametricValue>,
    ) -> Result<BlockArguments<'m>, String> {
        let target = template.block.name.as_str();
        // Source patterns retain the distinction between a type formal and
        // a nominal type introduced by expanding a module alias.
        let dependent = template.port_binding_work != 0;
        let formals = if dependent {
            &template.block.params
        } else {
            &template.ports.params
        };
        let type_params = &template.block.type_params;
        let input_error =
            |p: &Param, e: String| format!("input '{}', call to block '{target}': {e}", p.name);
        let mut bindings = Bindings::new(type_params);
        let mut inputs = Vec::with_capacity(formals.len());
        self.charge_expansion("block input types", template.input_pattern_work)?;
        for (p, arg) in formals.iter().zip(args) {
            // An argument of type T[N] can supply T before a default computes N
            // from that type. The concrete port below checks the array length.
            let pattern = if dependent {
                InputPattern::Function(None)
            } else {
                InputPattern::Block
            };
            inputs.push(
                self.infer_input(&p.ty, arg, &mut bindings, pattern)
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
        let values = template.apply_value_defaults(self, supplied, &bindings)?;
        let path = self.specialization.map_or(&[][..], |s| s.path.as_slice());
        let specialization = self
            .specializations
            .as_mut()
            .expect("block elaboration")
            .intern(
                template,
                BlockParametricArguments {
                    bindings,
                    callbacks,
                    values,
                },
                path,
                self.decls.module_dimensions(target),
                &mut TypeResolver {
                    resolver: self.resolver,
                    structs: self.structs,
                    functions: self.function_registry.as_deref_mut(),
                },
            )
            .map_err(|e| format!("call to block '{target}': {e}"))?;
        let inputs = inputs
            .into_iter()
            .zip(&specialization.interface.params)
            .map(|(input, p)| {
                if self.dslx_value_context
                    && let Some(expr) = input.source()
                {
                    self.check_dslx_argument_widths(expr, &p.ty)
                        .map_err(|e| input_error(p, e))?;
                }
                self.finish_input(input, &p.ty)
                    .map_err(|e| input_error(p, e))
            })
            .collect::<Result<_, _>>()?;
        Ok(BlockArguments {
            specialization,
            inputs,
        })
    }

    /// Combine evidence from complementary tuple fields or array elements
    /// across arguments that share a type variable.
    pub(super) fn infer_input_groups<'p>(
        &mut self,
        inputs: &mut [Input<'_>],
        formals: impl IntoIterator<Item = &'p Param>,
        bindings: &mut Bindings,
        callee: &str,
    ) -> Result<(), String> {
        let mut groups = BTreeMap::new();
        for (input, p) in inputs.iter_mut().zip(formals) {
            input.visit_leaves(&p.ty, &mut |ty, input| {
                if let (Type::Named(name), Input::Contextual { expr, .. }) = (ty, &*input)
                    && bindings.declares(name)
                    && bindings.get(name).is_none()
                    && (matches!(expr, Expr::Tuple(_)) || expr.unannotated_array().is_some())
                {
                    groups
                        .entry(name.as_str())
                        .or_insert_with(Vec::new)
                        .push((*expr, input));
                }
                Ok(())
            })?;
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
    pub(super) fn infer_constructors(
        &mut self,
        items: &[&Expr],
        infer_literal_widths: bool,
    ) -> Result<Option<Vec<TypedExpr>>, String> {
        if items.iter().any(|expr| expr.needs_scope_preparation()) {
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
        if let Some(rows) = items
            .iter()
            .map(|item| item.unannotated_array())
            .collect::<Option<Vec<_>>>()
        {
            // An unresolved element (such as Wrap { value: 9 }) must wait for
            // another port to bind its type before the array can be checked.
            return self.check_array_rows(items, rows, |checker, elements| {
                checker.infer_constructors(elements, infer_literal_widths)
            });
        }
        let mut deferred = false;
        for (index, item) in items.iter().enumerate() {
            if self.has_concrete_element_type(item)? {
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
            return Ok(pattern.contextual(arg));
        }
        if let Expr::SelectedOperand(value) = arg
            && super::has_aggregate_literal(value)
        {
            return self.infer_selected_members(formal, value, bindings, pattern);
        }
        let branches = matches!(arg, Expr::If { .. } | Expr::Match { .. });
        let partial_constructor = branches
            && !matches!(pattern, InputPattern::Function(Some(_)))
            && (self.contains_type_struct_constructor(arg)
                || (has_aggregate_literal(arg) && !self.has_concrete_element_type(arg)?));
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
        let mut charged_fill = 0;
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
                Expr::Array(items)
                | Expr::ArrayLiteral {
                    annotation: None,
                    members: items,
                    ..
                },
            ) => {
                if let Type::Array { size, .. } = formal
                    && matches!(arg, Expr::Array(_))
                    && items.len() != *size
                {
                    return Err(format!("expected array length {size}, got {}", items.len()));
                }
                if items.is_empty() {
                    // The checker constructs an empty IR array from its element
                    // type.
                    return Ok(pattern.contextual(arg));
                }
                if let InputPattern::Function(Some(TypedExpr {
                    ty: Type::Array { size, .. },
                    kind: K::Array(written),
                })) = &pattern
                {
                    charged_fill = size - written.len();
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
                return Ok(pattern.contextual(arg));
            }
            (Type::Named(name), _)
                if !bindings.declares(name)
                    && !matches!(pattern, InputPattern::Function(Some(_))) =>
            {
                return Ok(pattern.contextual(arg));
            }
            (Type::TypeApplied { .. } | Type::MixedApplied { .. }, _)
                if !bindings.has_unbound_variable(formal) =>
            {
                return Ok(pattern.contextual(arg));
            }
            _ => {
                if (matches!(arg, Expr::Tuple(_)) || arg.unannotated_array().is_some())
                    && !matches!(pattern, InputPattern::Function(Some(_)))
                    && self.contains_type_struct_constructor(arg)
                {
                    // A whole-aggregate type variable may receive its type
                    // from another argument or from the constructor group.
                    return Ok(pattern.contextual(arg));
                }
                if !matches!(arg, Expr::SelectedOperand(_))
                    && !matches!(pattern, InputPattern::Function(Some(_)))
                    && !self.has_concrete_element_type(arg)?
                {
                    return Ok(pattern.contextual(arg));
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
                        StructWitness::Deferred => return Ok(pattern.contextual(arg)),
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
        Ok(Input::Aggregate {
            expr: arg,
            members,
            charged_fill,
        })
    }

    /// Infer variables from a selected constructor while leaving fields with
    /// a bound type free to evaluate their arithmetic at the completed width.
    fn infer_selected_members<'e>(
        &mut self,
        formal: &Type,
        expr: &'e Expr,
        bindings: &mut Bindings,
        mut pattern: InputPattern,
    ) -> Result<Input<'e>, String> {
        if !bindings.has_unbound_variable(formal) {
            return Ok(pattern.contextual(expr));
        }
        if let (Type::Tuple(types), Expr::Tuple(fields)) = (formal, expr)
            && types.len() == fields.len()
        {
            let mut checked = pattern.take_members();
            let members = types
                .iter()
                .zip(fields)
                .map(|(ty, field)| {
                    self.infer_selected_members(ty, field, bindings, pattern.child(checked.next()))
                })
                .collect::<Result<_, _>>()?;
            return Ok(Input::Aggregate {
                expr,
                members,
                charged_fill: 0,
            });
        }
        // An array's element type reflects every row, even if the selected
        // expression appears in just one of them.
        let value = match pattern {
            InputPattern::Function(Some(value)) => value,
            _ => self.inferred_expr(expr)?,
        };
        bindings.infer_source_type(
            formal,
            &value.ty,
            self.structs,
            /* contextual_projection= */ true,
        )?;
        Ok(Input::Contextual {
            expr,
            prepared: Some(value),
        })
    }

    /// Assemble constructors and check previously unsized operands once the
    /// full interface is known, reusing the values that supplied its types.
    pub(super) fn finish_input(&mut self, input: Input, ty: &Type) -> Result<TypedExpr, String> {
        let (expr, members, charged_fill) = match input {
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
            Input::Contextual {
                prepared: Some(value),
                ..
            } if super::compatible(ty, &value.ty) => {
                return Ok(value);
            }
            Input::Contextual { expr, .. } => return self.expr(expr, Some(ty)),
            Input::Aggregate {
                expr,
                members,
                charged_fill,
            } => (expr, members, charged_fill),
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
        if let Type::Array { size, .. } = ty {
            let fill = matches!(
                expr,
                Expr::ArrayLiteral {
                    has_ellipsis: true,
                    ..
                }
            );
            self.check_array_length(members.len(), *size, fill, charged_fill)?;
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
