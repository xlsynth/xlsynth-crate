// SPDX-License-Identifier: Apache-2.0

use super::unroll::Shadowed;
use super::{AssertionCache, BodyChecker, CheckedStmt, K, Type, TypedExpr};
use crate::ast::{
    BinaryOp, BindingPattern, ConstAssert, Expr, ForIterable, LetBinding, ScopeStatement, Stmt,
    WidthSliceType,
};
use crate::infer_literal::{
    BindingWidths, WidthLookup, binding_widths_using, has_common_bit_type,
    has_concrete_element_type,
};
use crate::typed::TEST_DISCARD_PREFIX;
use crate::typed::{CONSTEXPR_PREFIX, ConstantBinding, ConstantUse};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use xlsynth_pir::IrValue;

/// An elaborated let initializer and the unsized literals retained at its
/// leaves.
pub(super) struct BindingInitializer {
    pub value: TypedExpr,
    pub widths: Option<BindingWidths>,
}

/// Bindings restored on scope exit and expression statements required by
/// constexpr evaluation.
struct EnteredScope {
    saved: Vec<Shadowed>,
    operands: Vec<TypedExpr>,
}

/// The runtime aggregate to project and its shared DSLX constexpr operands.
pub(super) struct ProjectionInput {
    pub(super) value: TypedExpr,
    pub(super) operands: Vec<TypedExpr>,
}

impl ProjectionInput {
    /// Keep each lexical check in its original evaluation mode while exposing
    /// the runtime value for projection.
    pub(super) fn from_shared(mut value: TypedExpr) -> Self {
        let mut operands = Vec::new();
        while let K::Sequence {
            operands: checks,
            result,
            bytecode_only,
        } = value.kind
        {
            if bytecode_only {
                // Keep the conditional check at the use site even when a
                // projection has other, unconditional constexpr operands.
                operands.push(TypedExpr {
                    ty: Type::Tuple(Vec::new()),
                    kind: K::Sequence {
                        operands: checks,
                        result: Box::new(unit()),
                        bytecode_only: true,
                    },
                });
            } else {
                operands.extend(checks);
            }
            value = *result;
        }
        Self { value, operands }
    }
}

fn unit() -> TypedExpr {
    TypedExpr {
        ty: Type::Tuple(Vec::new()),
        kind: K::Literal(IrValue::Tuple(Vec::new().into())),
    }
}

/// Attach the DSLX constexpr dependencies of a result without adding runtime
/// operations.
pub(super) fn sequence(operands: Vec<TypedExpr>, result: TypedExpr) -> TypedExpr {
    checked_sequence(operands, result, false)
}

fn checked_sequence(operands: Vec<TypedExpr>, result: TypedExpr, bytecode_only: bool) -> TypedExpr {
    if operands.is_empty() {
        return result;
    }
    TypedExpr {
        ty: result.ty.clone(),
        kind: K::Sequence {
            operands,
            result: Box::new(result),
            bytecode_only,
        },
    }
}

impl BodyChecker<'_, '_> {
    /// Keep erased type and specialization operands with their own value so
    /// that assertion replay follows the same branch as the original operand.
    pub(super) fn with_assert_operands(
        &mut self,
        check: impl FnOnce(&mut Self) -> Result<TypedExpr, String>,
    ) -> Result<TypedExpr, String> {
        if !self.preserve_assert_inputs {
            return check(self);
        }
        let outer = self.assert_operands.replace(Vec::new());
        let value = check(self);
        let operands = std::mem::replace(&mut self.assert_operands, outer)
            .expect("collecting expression operands");
        value.map(|value| sequence(operands, value))
    }

    pub(super) fn record_assert_operand(&mut self, value: TypedExpr) {
        if let Some(operands) = &mut self.assert_operands
            && !matches!(value.kind, K::Literal(_))
        {
            operands.push(value);
        }
    }

    /// Keep a match's bytecode-only initializers on their checked lexical path.
    pub(super) fn with_match_bytecode<T>(
        &mut self,
        check: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let outer = std::mem::replace(&mut self.in_match_bytecode, true);
        let result = check(self);
        self.in_match_bytecode = outer;
        result
    }

    /// Count concrete callees that may lose their only call when an assertion
    /// is erased.
    fn callee_specialization_count(&self) -> usize {
        self.resolver.call_count()
            + self
                .function_registry
                .as_ref()
                .map_or(0, |registry| registry.generated_count())
    }

    /// Evaluate a declaration in its lexical scope and erase its predicate's
    /// temporary lets.
    pub(super) fn check_const_assert(&mut self, assertion: &ConstAssert) -> Result<(), String> {
        let result = if self.source_template.is_some() && self.function_spec.is_none() {
            // Callback-only templates have a checked abstract body. Its
            // callback has no concrete target until specialization, so only
            // names can be required here.
            self.check_deferred_predicate_names(&assertion.predicate)
        } else {
            let prior_callees = self.callee_specialization_count();
            let start = self.statements.len();
            let had_functions = self.constant_functions.is_some();
            let preserve_inputs = std::mem::replace(&mut self.preserve_assert_inputs, true);
            let outer_use = std::mem::replace(&mut self.dslx_constant_use, ConstantUse::Assertion);
            let outer_match = std::mem::replace(&mut self.in_match_bytecode, false);
            let outer_cache = self.assertion_cache.replace(AssertionCache {
                first_local: start,
                saved: Vec::new(),
            });
            let ids = (
                self.next_map,
                self.next_counted_loop,
                self.unroll_binding_count,
            );
            self.enable_constant_functions();
            let result = self.with_dslx_value_context(|checker| {
                let expr = checker.dslx_argument(&assertion.predicate, &Type::Bool)?;
                let value = checker
                    .checked_constant_for(&expr, ConstantUse::Assertion)?
                    .ok_or("predicate must evaluate to a compile-time constant")?;
                if value.to_bool().map_err(|e| e.to_string())? {
                    Ok(())
                } else {
                    Err("predicate is false".to_owned())
                }
            });
            self.take_pure_bindings(start);
            (
                self.next_map,
                self.next_counted_loop,
                self.unroll_binding_count,
            ) = ids;
            self.preserve_assert_inputs = preserve_inputs;
            self.dslx_constant_use = outer_use;
            self.in_match_bytecode = outer_match;
            let cache = std::mem::replace(&mut self.assertion_cache, outer_cache)
                .expect("active assertion cache");
            for (name, binding) in cache.saved.into_iter().rev() {
                self.constants.insert(name, binding);
            }
            if self.callee_specialization_count() > prior_callees {
                self.decls.assertion_instantiated_callee.set(true);
            }
            if !had_functions {
                self.constant_functions = None;
            }
            result
        };
        result.map_err(|e| {
            format!(
                "{}:{}: const_assert!: {e}",
                assertion.position.line, assertion.position.column
            )
        })
    }

    /// Check each source pattern on the first visit to its enclosing folds.
    /// An empty fold visits the same patterns during lexical name checking.
    pub(super) fn validate_source_pattern(&self, pattern: &BindingPattern) -> Result<(), String> {
        let fields = match pattern {
            BindingPattern::Name(_) => return Ok(()),
            BindingPattern::Rest => return Err("rest marker requires a tuple pattern".into()),
            BindingPattern::Tuple(fields) => fields,
        };
        if self.iteration_path.iter().any(|&ordinal| ordinal != 0) {
            return Ok(());
        }
        let mut names = BTreeSet::new();
        let mut tuples = vec![fields];
        while let Some(fields) = tuples.pop() {
            let mut saw_rest = false;
            for field in fields {
                match field {
                    BindingPattern::Name(name) => {
                        if name != "_" && !names.insert(name.as_str()) {
                            return Err(format!("duplicate name '{name}' in let pattern"));
                        }
                    }
                    BindingPattern::Rest if saw_rest => {
                        return Err("duplicate rest marker in tuple pattern".into());
                    }
                    BindingPattern::Rest => saw_rest = true,
                    BindingPattern::Tuple(nested) => tuples.push(nested),
                }
            }
        }
        Ok(())
    }

    /// Elaborate the declarations of one lexical scope and restore all shadowed
    /// bindings after checking its trailing expression.
    pub(super) fn value_scope(
        &mut self,
        statements: &[ScopeStatement],
        result: &Expr,
        expected: Option<&Type>,
    ) -> Result<TypedExpr, String> {
        self.with_scope_result(statements, |checker| {
            checker.scoped_result(result, expected, false)
        })
    }

    /// Check a trailing value with its contextual type or struct arguments and
    /// lexical operands.
    pub(super) fn with_scope_result(
        &mut self,
        statements: &[ScopeStatement],
        check: impl FnOnce(&mut Self) -> Result<TypedExpr, String>,
    ) -> Result<TypedExpr, String> {
        let scope = self.enter_scope(statements)?;
        let value = check(self);
        self.leave_scope(scope.saved);
        value.map(|value| sequence(scope.operands, value))
    }

    /// Enter each binding only after its initializer has been checked.
    fn enter_scope(&mut self, statements: &[ScopeStatement]) -> Result<EnteredScope, String> {
        if !self.iteration_path.is_empty() {
            self.charge_expansion("unroll_for!", statements.len())?;
        }
        let mut saved = Vec::new();
        let mut operands = Vec::new();
        let result = statements.iter().try_for_each(|statement| {
            let binding = match statement {
                ScopeStatement::Let(binding) => binding,
                ScopeStatement::ConstAssert(assertion) => {
                    return self.check_const_assert(assertion);
                }
                ScopeStatement::Expression(expr) => {
                    let init = self.binding_initializer(expr, None, |checker, expr| {
                        checker.elaborate_result(expr, None, false)
                    })?;
                    let value = self.share_scoped_value(init.value);
                    if self.preserve_assert_inputs && !matches!(value.kind, K::Literal(_)) {
                        operands.push(value);
                    }
                    return Ok(());
                }
            };
            self.validate_source_pattern(&binding.pattern)?;
            let mut widths = None;
            let value = self.with_assert_operands(|checker| {
                let annotation = binding
                    .ty
                    .as_ref()
                    .map(|ty| checker.source_type(ty))
                    .transpose()?;
                let init = checker.binding_initializer(
                    &binding.value,
                    annotation.as_ref(),
                    |checker, expr| checker.elaborate_result(expr, annotation.as_ref(), false),
                )?;
                widths = init.widths;
                Ok(init.value)
            })?;
            self.bind_pattern_initializer(
                &binding.pattern,
                value,
                widths,
                &mut saved,
                Some(&mut operands),
            )
        });
        if let Err(error) = result {
            self.leave_scope(saved);
            return Err(error);
        }
        Ok(EnteredScope { saved, operands })
    }

    /// Capture inferred widths while the bindings that produced the result
    /// are still in scope, including when the initializer is itself a block.
    pub(super) fn binding_initializer(
        &mut self,
        expr: &Expr,
        annotation: Option<&Type>,
        check_value: impl FnOnce(&mut Self, &Expr) -> Result<TypedExpr, String>,
    ) -> Result<BindingInitializer, String> {
        self.with_prepared_value(expr, |checker, expr| {
            let value = check_value(checker, expr)?;
            let widths = if annotation.is_none() {
                binding_widths_using(
                    expr,
                    &value.ty,
                    WidthLookup::DslxLiterals(Some(&checker.inferred_bindings)),
                )?
            } else {
                None
            };
            Ok(BindingInitializer { value, widths })
        })
    }

    /// Keep each arm's lexical values available to common-type inference.
    /// Inference can then examine tuple fields or literal operands across
    /// arms while each scope's initializers are elaborated only once.
    pub(super) fn with_scoped_results<T>(
        &mut self,
        items: &[&Expr],
        check: impl FnOnce(&mut Self, &[&Expr]) -> Result<T, String>,
    ) -> Result<T, String> {
        self.with_scope_aliases(|checker| {
            let prepared = checker.capture_scoped_results(items)?;
            let expressions = prepared.iter().map(|e| e.as_ref()).collect::<Vec<_>>();
            check(checker, &expressions)
        })
    }

    /// Resolve lexical scopes on the argument's common-type edges before
    /// checking its literal constraints. The aliases keep the initializer's
    /// value and width available through both inference and elaboration.
    pub(super) fn dslx_argument(&mut self, expr: &Expr, ty: &Type) -> Result<TypedExpr, String> {
        self.with_prepared_value(expr, |checker, arg| {
            super::check_argument_widths(
                arg,
                ty,
                Some(&checker.inferred_bindings),
                Some(&|| checker.zero_calls_function()),
            )?;
            checker.expr(arg, Some(ty))
        })
    }

    /// Keep scoped initializer values and their type constraints alive until
    /// both elaboration and the caller's width analysis have consumed them.
    pub(super) fn with_prepared_value<T>(
        &mut self,
        expr: &Expr,
        check: impl FnOnce(&mut Self, &Expr) -> Result<T, String>,
    ) -> Result<T, String> {
        self.with_scope_aliases(|checker| {
            let value = checker.prepare_value(expr)?;
            check(checker, &value)
        })
    }

    /// Keep all aliases made during a group of checks alive through its final
    /// elaboration, then release them on success or failure.
    pub(super) fn with_scope_aliases<T>(
        &mut self,
        check: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let first_alias = self.scope_alias_names.len();
        let result = check(self);
        for i in first_alias..self.scope_alias_names.len() {
            let name = format!("$scope_{i}");
            self.function_values.remove(&name);
            self.loop_values.remove(&name);
            self.names.remove(&name);
            self.inferred_bindings.remove(&name);
            self.prepared_counted_iterables.remove(&name);
        }
        self.scope_alias_names.truncate(first_alias);
        result
    }

    /// Replace each scope contributing to a value's inferred type with
    /// its trailing expression. Other edges (such as an index or a nested
    /// call's arguments) have independent contexts and are checked normally.
    pub(super) fn prepare_value<'e>(&mut self, expr: &'e Expr) -> Result<Cow<'e, Expr>, String> {
        self.prepare_scopes(expr, false, &mut |_| Ok(()))
    }

    /// Retain the checked values of concrete constructor members while
    /// inferring their siblings. The caller consumes the aliases in this scope.
    pub(super) fn with_aggregate_operands<T>(
        &mut self,
        items: &[&Expr],
        check: impl FnOnce(&mut Self, &[&Expr]) -> Result<T, String>,
    ) -> Result<T, String> {
        self.with_scope_aliases(|checker| {
            let prepared = items
                .iter()
                .map(|expr| checker.prepare_scopes(expr, true, &mut |_| Ok(())))
                .collect::<Result<Vec<_>, _>>()?;
            let expressions = prepared.iter().map(|e| e.as_ref()).collect::<Vec<_>>();
            check(checker, &expressions)
        })
    }

    /// Check pending loop iterables before entering the first scope that can
    /// supply their seed's type. Scope-free seeds keep the borrowed AST.
    fn prepare_scopes<'e>(
        &mut self,
        expr: &'e Expr,
        capture_anchors: bool,
        before_scope: &mut dyn FnMut(&mut Self) -> Result<(), String>,
    ) -> Result<Cow<'e, Expr>, String> {
        // Constructors and branches share their members' contexts. A
        // self-determined leaf such as a function call can supply its actual
        // type now and use the same checked value during final elaboration.
        // A constructor with missing nominal arguments may need a sibling's
        // type first.
        if capture_anchors
            && !matches!(
                expr,
                Expr::Bool(_)
                    | Expr::Literal { .. }
                    | Expr::Scope { .. }
                    | Expr::Tuple(_)
                    | Expr::Array(_)
                    | Expr::If { .. }
                    | Expr::Match { .. }
            )
            && !super::has_aggregate_literal(expr)
            && self.contextual_struct_constructor(expr).is_none()
            && self.reference_type(expr).is_none()
            && has_concrete_element_type(expr, self.common_widths())?
        {
            before_scope(self)?;
            let mut init = self.binding_initializer(expr, None, |checker, expr| {
                let value = checker.expr(expr, None)?;
                if checker.replaying_body() {
                    let work = checker
                        .structs
                        .type_shape(&value.ty, super::MAX_ELABORATION_DEPTH)?
                        .ir_nodes;
                    checker.charge_expansion("aggregate operand", work)?;
                }
                Ok(value)
            })?;
            // A projection can supply a tuple shape while its literal members
            // still take their final width and signedness from another operand.
            if let Some(widths) = &mut init.widths {
                widths.infer_at_use = true;
            }
            let value = self.share_scoped_value(init.value);
            let alias = self.capture_value_alias(None, value, init.widths);
            return Ok(Cow::Owned(Expr::Name(alias)));
        }
        let mut prepared = Cow::Borrowed(expr);
        match expr {
            Expr::For(fold) if fold.annotation.is_none() => {
                let mut iteration = None;
                let seed = self.with_dslx_value_context(|checker| {
                    checker.prepare_scopes(&fold.seed, false, &mut |checker| {
                        if iteration.is_none() {
                            before_scope(checker)?;
                            checker.validate_counted_binders(fold)?;
                            iteration = Some(checker.counted_iteration(&fold.iterable, None)?);
                        }
                        Ok(())
                    })
                })?;
                if let Some(iteration) = iteration {
                    // The enclosing inference scope owns both the checked
                    // iterable and the lexical aliases in the seed.
                    let alias = format!("$scope_{}", self.scope_alias_names.len());
                    self.scope_alias_names.push(None);
                    self.prepared_counted_iterables
                        .insert(alias.clone(), iteration);
                    let Expr::For(fold) = prepared.to_mut() else {
                        unreachable!()
                    };
                    fold.iterable = ForIterable::Array(Expr::Name(alias));
                    fold.seed = seed.into_owned();
                }
            }
            Expr::Scope { statements, result } if !expr.needs_scope_preparation() => {
                if let Cow::Owned(result) =
                    self.prepare_scopes(result, capture_anchors, before_scope)?
                {
                    prepared = Cow::Owned(Expr::Scope {
                        statements: statements.clone(),
                        result: Box::new(result),
                    });
                }
            }
            Expr::Scope { .. } => {
                before_scope(self)?;
                let mut result = self
                    .capture_scoped_results(&[expr])?
                    .pop()
                    .expect("one scope")
                    .into_owned();
                if let Cow::Owned(nested) =
                    self.prepare_scopes(&result, capture_anchors, before_scope)?
                {
                    result = nested;
                }
                prepared = Cow::Owned(result);
            }
            Expr::Match { .. } if self.dslx_widths().is_none() => {
                // The lexical initializers in a match arm use DSLX inference
                // even if its enclosing block argument uses BX inference.
                return self.with_dslx_value_context(|checker| {
                    checker.prepare_scopes(expr, capture_anchors, before_scope)
                });
            }
            Expr::Match { arms, .. } => {
                for (index, arm) in arms.iter().enumerate() {
                    if let Cow::Owned(value) = self.with_match_bytecode(|checker| {
                        checker.prepare_scopes(&arm.value, capture_anchors, before_scope)
                    })? {
                        let Expr::Match { arms, .. } = prepared.to_mut() else {
                            unreachable!()
                        };
                        arms[index].value = value;
                    }
                }
            }
            Expr::Tuple(items)
            | Expr::Array(items)
            | Expr::ArrayLiteral {
                annotation: None,
                members: items,
                ..
            } => {
                for (i, item) in items.iter().enumerate() {
                    if let Cow::Owned(item) =
                        self.prepare_scopes(item, capture_anchors, before_scope)?
                    {
                        let (Expr::Tuple(items)
                        | Expr::Array(items)
                        | Expr::ArrayLiteral { members: items, .. }) = prepared.to_mut()
                        else {
                            unreachable!()
                        };
                        items[i] = item;
                    }
                }
            }
            Expr::Field { value, .. } | Expr::Index { value, .. } | Expr::Unary { value, .. } => {
                // Only a member's complete expression supplies an anchor.
                // Its internal operands still need their lexical scopes.
                if let Cow::Owned(value) = self.prepare_scopes(value, false, before_scope)? {
                    let (Expr::Field { value: subject, .. }
                    | Expr::Index { value: subject, .. }
                    | Expr::Unary { value: subject, .. }) = prepared.to_mut()
                    else {
                        unreachable!()
                    };
                    **subject = value;
                }
            }
            Expr::Binary { op, left, right }
                if has_common_bit_type(*op) || matches!(op, BinaryOp::Shl | BinaryOp::Shr) =>
            {
                let lhs = self.prepare_scopes(left, false, before_scope)?;
                let rhs = if has_common_bit_type(*op) {
                    self.prepare_scopes(right, false, before_scope)?
                } else {
                    Cow::Borrowed(right.as_ref())
                };
                if matches!(lhs, Cow::Owned(_)) || matches!(rhs, Cow::Owned(_)) {
                    prepared = Cow::Owned(Expr::Binary {
                        op: *op,
                        left: Box::new(lhs.into_owned()),
                        right: Box::new(rhs.into_owned()),
                    });
                }
            }
            Expr::If {
                condition,
                then_value,
                else_value,
            } => {
                let lhs = self.prepare_scopes(then_value, capture_anchors, before_scope)?;
                let rhs = self.prepare_scopes(else_value, capture_anchors, before_scope)?;
                if matches!(lhs, Cow::Owned(_)) || matches!(rhs, Cow::Owned(_)) {
                    prepared = Cow::Owned(Expr::If {
                        condition: condition.clone(),
                        then_value: Box::new(lhs.into_owned()),
                        else_value: Box::new(rhs.into_owned()),
                    });
                }
            }
            _ => { /* This expression has no scopes sharing the parameter's type. */ }
        }
        Ok(prepared)
    }

    fn capture_scoped_results<'e>(
        &mut self,
        items: &[&'e Expr],
    ) -> Result<Vec<Cow<'e, Expr>>, String> {
        let mut prepared = Vec::with_capacity(items.len());
        for &expr in items {
            let Expr::Scope { statements, result } = expr else {
                prepared.push(Cow::Borrowed(expr));
                continue;
            };
            let scope = self.enter_scope(statements)?;
            let mut aliases = BTreeMap::new();
            for statement in statements {
                let ScopeStatement::Let(binding) = statement else {
                    // An assertion introduces no names to capture.
                    continue;
                };
                binding.pattern.visit_names(&mut |name| {
                    if aliases.contains_key(name) {
                        return;
                    }
                    // The temporary alias now owns the local value. Leaving
                    // the scope below restores any shadowed outer binding.
                    let value = if self.in_loop_scope() {
                        self.loop_values.remove(name)
                    } else {
                        self.function_values.remove(name)
                    }
                    .expect("bound scoped result");
                    let widths = self.inferred_bindings.remove(name);
                    let alias = self.capture_value_alias(Some(name), value, widths);
                    aliases.insert(name.to_owned(), alias);
                });
            }
            let mut result = result.as_ref().clone();
            capture_names(&mut result, &mut aliases);
            self.leave_scope(scope.saved);
            if !scope.operands.is_empty() {
                let statements = scope
                    .operands
                    .into_iter()
                    .map(|value| {
                        let alias = self.capture_value_alias(None, value, None);
                        ScopeStatement::Expression(Expr::Name(alias))
                    })
                    .collect();
                result = Expr::Scope {
                    statements,
                    result: Box::new(result),
                };
            }
            prepared.push(Cow::Owned(result));
        }
        Ok(prepared)
    }

    /// Preserve a checked operand until the surrounding group has used its
    /// type and value. The enclosing alias scope releases all three entries.
    fn capture_value_alias(
        &mut self,
        source_name: Option<&str>,
        value: TypedExpr,
        widths: Option<BindingWidths>,
    ) -> String {
        let alias = format!("$scope_{}", self.scope_alias_names.len());
        self.scope_alias_names.push(source_name.map(str::to_owned));
        let values = if self.in_loop_scope() {
            &mut self.loop_values
        } else {
            &mut self.function_values
        };
        values.insert(alias.clone(), value);
        self.names.insert(alias.clone());
        if let Some(mut widths) = widths {
            widths.bind_in_scope(self.width_scope());
            self.inferred_bindings.insert(alias.clone(), widths);
        }
        alias
    }

    /// Materialize an initializer once under a name that survives lexical
    /// shadowing and cannot collide with source or loop-generated names.
    pub(super) fn share_scoped_value(&mut self, value: TypedExpr) -> TypedExpr {
        let result = value.runtime_result();
        if matches!(&result.kind, K::Literal(_) | K::Name(_)) {
            if let K::Sequence {
                operands,
                result: immediate,
                ..
            } = &value.kind
                && (!matches!(
                    operands.as_slice(),
                    [TypedExpr {
                        kind: K::Name(_),
                        ..
                    }]
                ) || matches!(immediate.kind, K::Sequence { .. }))
            {
                let captures = if self.in_match_bytecode {
                    value.match_capture_operands()
                } else {
                    Vec::new()
                };
                if let K::Sequence {
                    operands,
                    result,
                    bytecode_only: true,
                } = value.kind
                {
                    // Share long carry lists by name and keep the bytecode
                    // check on the reference at the use site.
                    let prerequisite = self.retain_constexpr_operand(sequence(operands, unit()));
                    let value = TypedExpr {
                        ty: value.ty,
                        kind: K::Sequence {
                            operands: vec![prerequisite],
                            result,
                            bytecode_only: true,
                        },
                    };
                    return checked_sequence(captures, value, true);
                }
                // Keep lexical aliases of a leaf cheap even when a scope
                // depends on that leaf in both a let initializer and its final
                // result.
                let result = result.clone();
                let prerequisite = self.retain_constexpr_operand(value);
                return checked_sequence(captures, sequence(vec![prerequisite], result), true);
            }
            return value;
        }
        let match_local = self.in_match_bytecode && self.preserve_assert_inputs;
        let prefix = if self.in_erased_operand {
            CONSTEXPR_PREFIX
        } else if match_local {
            crate::typed::MATCH_LET_PREFIX
        } else {
            "$let_"
        };
        let name = format!("{prefix}{}", self.statements.len());
        let constexpr = value.has_constant_inputs(&self.constexpr_names);
        let capture_operands = if match_local {
            value.match_capture_operands()
        } else {
            Vec::new()
        };
        // Guarded statements move into their own body, so a later scoped
        // binding can reuse the same temporary name in the enclosing block.
        if constexpr {
            self.constexpr_names.insert(name.clone());
        } else {
            self.constexpr_names.remove(&name);
        }
        if let Some(constant) = self.fold_constant(&value, constexpr) {
            self.constants.insert(name.clone(), constant);
        } else {
            self.constants.remove(&name);
        }
        let ty = value.ty.clone();
        self.statements.push(CheckedStmt::Let {
            name: name.clone(),
            value,
        });
        let result = TypedExpr {
            ty,
            kind: K::Name(name),
        };
        // A constexpr match captures every free name in its arms. A separate
        // assertion that reads this let inside an arm evaluates just the
        // initializer's selected path.
        checked_sequence(capture_operands, result, true)
    }

    /// Share the compile-time work of an initializer without emitting its
    /// value. The unit reference avoids copying a tuple's type for every
    /// projected field.
    pub(super) fn retain_constexpr_operand(&mut self, value: TypedExpr) -> TypedExpr {
        let index = self.statements.len();
        let name = format!("{CONSTEXPR_PREFIX}{index}");
        self.constants
            .insert(name.clone(), ConstantBinding::Deferred(index));
        self.statements.push(CheckedStmt::Let {
            name: name.clone(),
            value: sequence(vec![value], unit()),
        });
        TypedExpr {
            ty: Type::Tuple(Vec::new()),
            kind: K::Name(name),
        }
    }

    /// Widen the inferred leaves of a bound reference to their use-site
    /// types. Each aggregate is evaluated once before its members are read.
    pub(super) fn widen_projection(
        &mut self,
        value: TypedExpr,
        target: &Type,
    ) -> Result<TypedExpr, String> {
        if super::compatible(&value.ty, target) {
            return Ok(value);
        }
        if let (Some((_, width)), Some((_, target_width))) =
            (super::bits(&value.ty), super::bits(target))
        {
            let kind = if width == target_width {
                value.kind
            } else {
                K::Cast {
                    value: Box::new(value),
                }
            };
            return Ok(TypedExpr {
                ty: target.clone(),
                kind,
            });
        }
        let ProjectionInput {
            value: input,
            operands,
        } = ProjectionInput::from_shared(self.share_scoped_value(value));
        let mut field = |index: usize, source: &Type, destination: &Type| {
            // Aggregate conversions generate one IR projection per member.
            let work = self
                .structs
                .type_shape(source, super::MAX_ELABORATION_DEPTH)?
                .ir_nodes;
            self.charge_expansion("aggregate projection", work)?;
            let kind = match &input.kind {
                K::Literal(value) => {
                    let member = value.get_element(index).map_err(|e| e.to_string())?;
                    self.charge_expansion(
                        "aggregate projection",
                        crate::typed::constant_work(&member),
                    )?;
                    K::Literal(member)
                }
                K::Name(name) if matches!(input.ty, Type::Tuple(_)) => K::BindingTupleIndex {
                    name: name.clone(),
                    index,
                },
                K::Name(_) => K::Index {
                    value: Box::new(input.clone()),
                    index: Box::new(TypedExpr {
                        ty: Type::unsigned(32),
                        kind: K::Literal(
                            IrValue::make_ubits(32, index as u64).map_err(|e| e.to_string())?,
                        ),
                    }),
                },
                _ => unreachable!("shared aggregate is a name or literal"),
            };
            self.widen_projection(
                TypedExpr {
                    ty: source.clone(),
                    kind,
                },
                destination,
            )
        };
        let kind = match (&input.ty, target) {
            (Type::Tuple(source), Type::Tuple(target)) => K::Tuple(
                source
                    .iter()
                    .zip(target)
                    .enumerate()
                    .map(|(i, (s, t))| field(i, s, t))
                    .collect::<Result<_, _>>()?,
            ),
            (
                Type::Array {
                    element: source,
                    size,
                },
                Type::Array {
                    element: target, ..
                },
            ) => K::Array(
                (0..*size)
                    .map(|i| field(i, source, target))
                    .collect::<Result<_, _>>()?,
            ),
            _ => unreachable!("projection compatibility checked"),
        };
        Ok(sequence(
            operands,
            TypedExpr {
                ty: target.clone(),
                kind,
            },
        ))
    }

    /// Project the members selected by a tuple pattern from its initializer.
    pub(super) fn bind_pattern(
        &mut self,
        pattern: &BindingPattern,
        value: TypedExpr,
        widths: Option<BindingWidths>,
        saved: &mut Vec<Shadowed>,
    ) -> Result<(), String> {
        self.bind_pattern_initializer(pattern, value, widths, saved, None)
    }

    /// Preserve the source initializer when a constexpr call executes this let.
    fn bind_pattern_initializer(
        &mut self,
        pattern: &BindingPattern,
        value: TypedExpr,
        widths: Option<BindingWidths>,
        saved: &mut Vec<Shadowed>,
        operands: Option<&mut Vec<TypedExpr>>,
    ) -> Result<(), String> {
        let value = if self.preserve_assert_inputs
            && operands.is_some()
            && !matches!(value.kind, K::Tuple(_))
        {
            self.share_scoped_value(value)
        } else {
            value
        };
        // A tuple constructor is normally split directly into its field
        // computations. DSLX determines whether any projected binder is
        // constexpr from the whole initializer. Keep one deferred definition
        // shared by all of the leaves.
        let prerequisite = if self.preserve_assert_inputs
            && matches!(pattern, BindingPattern::Tuple(_))
            && matches!(&value.kind, K::Tuple(fields) if fields.iter().any(|field| !matches!(field.kind, K::Literal(_))))
        {
            Some(self.retain_constexpr_operand(value.clone()))
        } else {
            None
        };
        if self.preserve_assert_inputs
            && let Some(operands) = operands
        {
            let value = prerequisite.as_ref().unwrap_or(&value);
            if !matches!(value.kind, K::Literal(_)) {
                let unit = unit();
                operands.push(TypedExpr {
                    ty: unit.ty.clone(),
                    kind: K::Sequence {
                        operands: vec![value.clone()],
                        result: Box::new(unit),
                        bytecode_only: true,
                    },
                });
            }
        }
        self.bind_pattern_with_prerequisite(pattern, value, widths, saved, prerequisite.as_ref())
    }

    /// Project the source pattern while retaining its shared tuple prerequisite
    /// at named leaves.
    fn bind_pattern_with_prerequisite(
        &mut self,
        pattern: &BindingPattern,
        value: TypedExpr,
        widths: Option<BindingWidths>,
        saved: &mut Vec<Shadowed>,
        prerequisite: Option<&TypedExpr>,
    ) -> Result<(), String> {
        match pattern {
            BindingPattern::Name(name) => {
                let value = if name != "_"
                    && let Some(prerequisite) = prerequisite
                {
                    sequence(vec![prerequisite.clone()], value)
                } else {
                    value
                };
                let value = self.share_scoped_value(value);
                self.bind_scoped_value(name, value, widths, saved)
            }
            BindingPattern::Rest => return Err("rest marker requires a tuple pattern".into()),
            BindingPattern::Tuple(patterns) => {
                // A tuple constructor or literal already owns its fields.
                // Materialize other expressions once and project by IR name.
                let value = if matches!(&value.kind, K::Tuple(_) | K::Literal(_)) {
                    value
                } else {
                    self.share_scoped_value(value)
                };
                let ProjectionInput { value, operands } = ProjectionInput::from_shared(value);
                let Type::Tuple(types) = value.ty else {
                    return Err(format!("tuple pattern requires a tuple, got {}", value.ty));
                };
                let count = types.len();
                if self.replaying_body() {
                    // Selecting just one field can still scan the tuple or
                    // a long wildcard pattern. Charge once for the traversal.
                    self.charge_expansion("tuple pattern", count.max(patterns.len()))?;
                }
                let rest = patterns
                    .iter()
                    .position(|p| matches!(p, BindingPattern::Rest));
                let required = patterns.len() - usize::from(rest.is_some());
                if required > count || rest.is_none() && required != count {
                    return Err(format!(
                        "tuple pattern with {required} fields cannot bind a {count}-element tuple"
                    ));
                }
                let mut widths = widths.map(|w| w.tuple_fields(count));
                let literal_members = match &value.kind {
                    K::Literal(IrValue::Tuple(fields)) => Some(Arc::clone(fields)),
                    _ => None,
                };
                // A constructor owns the field computations. A named tuple has
                // already evaluated them and a literal has no runtime
                // contracts.
                let evaluate_discards = self.decls.evaluate_discarded_tuple_fields
                    && self.source_kind == super::BodyKind::Function
                    && matches!(&value.kind, K::Tuple(_));
                let mut fields: Box<dyn Iterator<Item = TypedExpr>> = match value.kind {
                    K::Tuple(fields) => Box::new(fields.into_iter()),
                    K::Literal(IrValue::Tuple(fields)) => {
                        Box::new(types.into_iter().enumerate().map(move |(i, ty)| TypedExpr {
                            ty,
                            kind: K::Literal(fields[i].clone()),
                        }))
                    }
                    K::Name(name) => Box::new(types.into_iter().enumerate().map(
                        move |(index, ty)| TypedExpr {
                            ty,
                            kind: K::BindingTupleIndex {
                                name: name.clone(),
                                index,
                            },
                        },
                    )),
                    _ => unreachable!("shared tuple expression is a name, literal, or constructor"),
                };
                let mut cursor = 0;
                for (i, pattern) in patterns.iter().enumerate() {
                    if matches!(pattern, BindingPattern::Rest)
                        || matches!(pattern, BindingPattern::Name(n) if n == "_")
                    {
                        if evaluate_discards {
                            let skipped = if matches!(pattern, BindingPattern::Rest) {
                                count - required
                            } else {
                                1
                            };
                            for field in fields.by_ref().take(skipped) {
                                // Names were already evaluated and literals
                                // have no contracts.
                                if !matches!(field.kind, K::Name(_) | K::Literal(_)) {
                                    self.decls.has_test_discards.set(true);
                                    self.statements.push(CheckedStmt::Let {
                                        name: format!(
                                            "{TEST_DISCARD_PREFIX}{}",
                                            self.statements.len()
                                        ),
                                        value: field,
                                    });
                                }
                            }
                            cursor += skipped;
                        }
                        continue;
                    }
                    let index = if rest.is_some_and(|r| i > r) {
                        count - (patterns.len() - i)
                    } else {
                        i
                    };
                    if let Some(members) = &literal_members {
                        // The iterator copies this selected literal into a
                        // new lexical binding; other tuple fields stay shared.
                        self.charge_reference_copy(super::literal_copy_work(&members[index]))?;
                    }
                    let field = fields
                        .nth(index - cursor)
                        .expect("tuple pattern arity checked");
                    cursor = index + 1;
                    if self.replaying_body() {
                        let work = match &field.kind {
                            K::BindingTupleIndex { .. } => {
                                self.structs
                                    .type_shape(&field.ty, super::MAX_ELABORATION_DEPTH)?
                                    .ir_nodes
                            }
                            K::Literal(value) => crate::typed::constant_work(value),
                            _ => 0, // Tuple constructors already own their checked fields.
                        };
                        self.charge_expansion("tuple pattern projection", work)?;
                    }
                    let widths = widths.as_mut().and_then(|w| w[index].take());
                    self.bind_pattern_with_prerequisite(
                        pattern,
                        sequence(operands.clone(), field),
                        widths,
                        saved,
                        prerequisite,
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Recover the user's binding name when scope preparation has replaced a
/// free reference with its temporary alias.
pub(super) fn source_binding_name<'s>(name: &'s str, aliases: &'s [Option<String>]) -> &'s str {
    name.strip_prefix("$scope_")
        .and_then(|index| index.parse::<usize>().ok())
        .and_then(|index| aliases.get(index))
        .and_then(Option::as_deref)
        .unwrap_or(name)
}

/// Substitute stable aliases for free references to the enclosing scope.
/// Binders introduced by a nested value scope or fold hide outer aliases.
fn capture_names(expr: &mut Expr, aliases: &mut BTreeMap<String, String>) {
    if aliases.is_empty() {
        return;
    }
    fn rename(name: &mut String, aliases: &BTreeMap<String, String>) {
        if let Some(local) = crate::bx_import::local_callee_name(name) {
            if let Some(alias) = aliases.get(local) {
                *name = crate::bx_import::mark_local_callee(alias);
            }
        } else if let Some((head, member)) = name.split_once("::") {
            if let Some(alias) = aliases.get(head) {
                *name = format!("{alias}::{member}");
            }
        } else if let Some(alias) = aliases.get(name) {
            name.clone_from(alias);
        }
    }
    fn rename_type(ty: &mut Type, aliases: &mut BTreeMap<String, String>) {
        match ty {
            Type::Named(name) | Type::Applied { name, .. } => rename(name, aliases),
            Type::ParametricStruct { name, args } => {
                rename(name, aliases);
                for arg in args {
                    capture_names(arg, aliases);
                }
            }
            Type::TypeApplied { name, args } => {
                rename(name, aliases);
                for arg in args {
                    rename_type(arg, aliases);
                }
            }
            Type::Array { element, .. } => rename_type(element, aliases),
            Type::ParametricArray { element, size } => {
                rename_type(element, aliases);
                capture_names(size, aliases);
            }
            Type::ParametricBits { signed, width } => {
                capture_names(signed, aliases);
                capture_names(width, aliases);
            }
            Type::Tuple(fields) => fields.iter_mut().for_each(|f| rename_type(f, aliases)),
            Type::Bits { .. } | Type::Enum { .. } | Type::Bool => { /* Primitive types have no references. */
            }
        }
    }
    fn capture_iterable(iterable: &mut ForIterable, aliases: &mut BTreeMap<String, String>) {
        match iterable {
            ForIterable::Range { start, limit, .. } => {
                capture_names(start, aliases);
                capture_names(limit, aliases);
            }
            ForIterable::Array(array) => capture_names(array, aliases),
        }
    }
    fn capture_binding(
        binding: &mut LetBinding,
        aliases: &mut BTreeMap<String, String>,
        hidden: &mut Vec<(String, String)>,
    ) {
        if let Some(ty) = &mut binding.ty {
            rename_type(ty, aliases);
        }
        capture_names(&mut binding.value, aliases);
        binding
            .pattern
            .visit_names(&mut |name| hidden.extend(aliases.remove_entry(name)));
    }
    match expr {
        Expr::Name(name) => rename(name, aliases),
        Expr::Bool(_) | Expr::ByteString(_) | Expr::Literal { ty: None, .. } => {
            /* No names to capture. */
        }
        Expr::Literal { ty: Some(ty), .. } => rename_type(ty, aliases),
        Expr::Zero(ty) => {
            rename_type(ty, aliases);
            if let Some(alias) = aliases.get("zero!") {
                *expr = Expr::Call {
                    name: alias.clone(),
                    args: Vec::new(),
                };
            }
        }
        Expr::TypeBuiltin { types, args, .. } => {
            for ty in types {
                rename_type(ty, aliases);
            }
            for arg in args {
                capture_names(arg, aliases);
            }
        }
        Expr::Tuple(fields) | Expr::Array(fields) => {
            fields.iter_mut().for_each(|f| capture_names(f, aliases))
        }
        Expr::ArrayLiteral {
            annotation,
            members,
            ..
        } => {
            if let Some(ty) = annotation {
                rename_type(ty, aliases);
            }
            for member in members {
                capture_names(member, aliases);
            }
        }
        Expr::Struct { name, fields, base }
        | Expr::AppliedStruct {
            name, fields, base, ..
        }
        | Expr::ParametricStruct {
            name, fields, base, ..
        }
        | Expr::TypeAppliedStruct {
            name, fields, base, ..
        } => {
            rename(name, aliases);
            for (_, value) in fields {
                capture_names(value, aliases);
            }
            if let Some(base) = base {
                capture_names(&mut base.value, aliases);
            }
            match expr {
                Expr::ParametricStruct { args, .. } => {
                    for arg in args {
                        capture_names(arg, aliases);
                    }
                }
                Expr::TypeAppliedStruct { args, .. } => {
                    for arg in args {
                        rename_type(arg, aliases);
                    }
                }
                _ => { /* The other constructors have no source parametrics to rename. */ }
            }
        }
        Expr::Call { name, args } => {
            rename(name, aliases);
            for arg in args {
                capture_names(arg, aliases);
            }
        }
        Expr::MapBeforeUse {
            name,
            args,
            builtin_args,
        } => {
            rename(name, aliases);
            for arg in args.iter_mut().chain(builtin_args.iter_mut().flatten()) {
                capture_names(arg, aliases);
            }
        }
        Expr::ParametricCall {
            name,
            parametrics,
            args,
        } => {
            rename(name, aliases);
            for arg in parametrics.iter_mut().chain(args) {
                capture_names(arg, aliases);
            }
        }
        Expr::Field { value, .. } | Expr::Unary { value, .. } => capture_names(value, aliases),
        Expr::Index {
            value: left,
            index: right,
        }
        | Expr::Binary { left, right, .. } => {
            capture_names(left, aliases);
            capture_names(right, aliases);
        }
        Expr::Cast { value, ty } => {
            capture_names(value, aliases);
            rename_type(ty, aliases);
        }
        Expr::WidthSlice {
            value,
            start,
            width,
        } => {
            capture_names(value, aliases);
            capture_names(start, aliases);
            match width {
                WidthSliceType::Concrete(ty) => rename_type(ty, aliases),
                WidthSliceType::Bits { signed, bit_count } => {
                    capture_names(signed, aliases);
                    capture_names(bit_count, aliases);
                }
            }
        }
        Expr::Match { subject, arms } => {
            capture_names(subject, aliases);
            for arm in arms {
                for literal in arm.patterns.iter_mut().flat_map(|p| p.literals_mut()) {
                    capture_names(literal, aliases);
                }
                capture_names(&mut arm.value, aliases);
            }
        }
        Expr::If {
            condition,
            then_value,
            else_value,
        } => {
            capture_names(condition, aliases);
            capture_names(then_value, aliases);
            capture_names(else_value, aliases);
        }
        Expr::Scope { statements, result } => {
            let mut hidden = Vec::new();
            for statement in statements {
                match statement {
                    ScopeStatement::Let(binding) => capture_binding(binding, aliases, &mut hidden),
                    ScopeStatement::ConstAssert(assertion) => {
                        capture_names(&mut assertion.predicate, aliases)
                    }
                    ScopeStatement::Expression(expr) => capture_names(expr, aliases),
                }
            }
            capture_names(result, aliases);
            aliases.extend(hidden);
        }
        Expr::UnrollFor(fold) => {
            if let Some((i, a)) = &mut fold.annotation {
                rename_type(i, aliases);
                rename_type(a, aliases);
            }
            capture_names(&mut fold.start, aliases);
            capture_names(&mut fold.limit, aliases);
            capture_names(&mut fold.seed, aliases);
            let mut hidden = Vec::new();
            hidden.extend(aliases.remove_entry(&fold.index));
            hidden.extend(aliases.remove_entry(&fold.accumulator));
            for stmt in &mut fold.body {
                // The checker accepts only lets in a fold body.
                if let Stmt::Let { name, ty, value } = stmt {
                    if let Some(ty) = ty {
                        rename_type(ty, aliases);
                    }
                    capture_names(value, aliases);
                    hidden.extend(aliases.remove_entry(name));
                } else if let Stmt::LetPattern(binding) = stmt {
                    capture_binding(binding, aliases, &mut hidden);
                }
            }
            capture_names(&mut fold.result, aliases);
            aliases.extend(hidden);
        }
        Expr::For(fold) => {
            if let Some((element, carry)) = &mut fold.annotation {
                rename_type(element, aliases);
                rename_type(carry, aliases);
            }
            capture_iterable(&mut fold.iterable, aliases);
            capture_names(&mut fold.seed, aliases);
            let mut hidden = Vec::new();
            for pattern in [&fold.element, &fold.accumulator] {
                pattern.visit_names(&mut |name| hidden.extend(aliases.remove_entry(name)));
            }
            capture_names(&mut fold.body, aliases);
            aliases.extend(hidden);
        }
        Expr::Map(map) => {
            rename(&mut map.target, aliases);
            capture_iterable(&mut map.iterable, aliases);
            if let Some(ty) = &mut map.parameter_type {
                rename_type(ty, aliases);
            }
            let hidden = aliases.remove_entry(&map.parameter);
            if let Some(ty) = &mut map.result_type {
                rename_type(ty, aliases);
            }
            capture_names(&mut map.body, aliases);
            aliases.extend(hidden);
        }
        Expr::NamedMap {
            target,
            iterable,
            callback,
            parametrics,
        } => {
            rename(target, aliases);
            capture_iterable(iterable, aliases);
            rename(callback, aliases);
            for arg in parametrics {
                capture_names(arg, aliases);
            }
        }
    }
}
