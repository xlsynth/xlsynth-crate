// SPDX-License-Identifier: Apache-2.0

use super::unroll::Shadowed;
use super::{BodyChecker, CheckedStmt, K, Type, TypedExpr};
use crate::ast::{BinaryOp, BindingPattern, Expr, ForIterable, LetBinding, Stmt, WidthSliceType};
use crate::infer_literal::{BindingWidths, WidthLookup, binding_widths_using, has_common_bit_type};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use xlsynth_pir::IrValue;

/// An elaborated let initializer and the unsized literals retained at its
/// leaves.
pub(super) struct BindingInitializer {
    pub value: TypedExpr,
    pub widths: Option<BindingWidths>,
}

impl BodyChecker<'_, '_> {
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

    /// Elaborate the pure lets of one lexical scope and restore all shadowed
    /// bindings after checking its trailing expression.
    pub(super) fn value_scope(
        &mut self,
        bindings: &[LetBinding],
        result: &Expr,
        expected: Option<&Type>,
    ) -> Result<TypedExpr, String> {
        let saved = self.enter_scope(bindings)?;
        let value = self.scoped_result(result, expected, false);
        self.leave_scope(saved);
        value
    }

    /// Enter each binding only after its initializer has been checked.
    fn enter_scope(&mut self, bindings: &[LetBinding]) -> Result<Vec<Shadowed>, String> {
        if !self.iteration_path.is_empty() {
            self.charge_unroll(bindings.len())?;
        }
        let mut saved = Vec::new();
        let result = bindings.iter().try_for_each(|binding| {
            self.validate_source_pattern(&binding.pattern)?;
            let annotation = binding
                .ty
                .as_ref()
                .map(|t| self.source_type(t))
                .transpose()?;
            let init =
                self.binding_initializer(&binding.value, annotation.as_ref(), |checker, expr| {
                    checker.elaborate_result(expr, annotation.as_ref(), false)
                })?;
            self.bind_pattern(&binding.pattern, init.value, init.widths, &mut saved)
        });
        if let Err(error) = result {
            self.leave_scope(saved);
            return Err(error);
        }
        Ok(saved)
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
        let first_alias = self.next_scope_alias;
        let result = check(self);
        for i in first_alias..self.next_scope_alias {
            let name = format!("$scope_{i}");
            self.function_values.remove(&name);
            self.loop_values.remove(&name);
            self.names.remove(&name);
            self.inferred_bindings.remove(&name);
            self.prepared_counted_iterables.remove(&name);
        }
        self.next_scope_alias = first_alias;
        result
    }

    /// Replace each scope contributing to a value's inferred type with
    /// its trailing expression. Other edges (such as an index or a nested
    /// call's arguments) have independent contexts and are checked normally.
    pub(super) fn prepare_value<'e>(&mut self, expr: &'e Expr) -> Result<Cow<'e, Expr>, String> {
        self.prepare_scopes(expr, &mut |_| Ok(()))
    }

    /// Check pending loop iterables before entering the first scope that can
    /// supply their seed's type. Scope-free seeds keep the borrowed AST.
    fn prepare_scopes<'e>(
        &mut self,
        expr: &'e Expr,
        before_scope: &mut dyn FnMut(&mut Self) -> Result<(), String>,
    ) -> Result<Cow<'e, Expr>, String> {
        let mut prepared = Cow::Borrowed(expr);
        match expr {
            Expr::For(fold) if fold.annotation.is_none() => {
                let mut iteration = None;
                let seed = self.with_dslx_value_context(|checker| {
                    checker.prepare_scopes(&fold.seed, &mut |checker| {
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
                    let alias = format!("$scope_{}", self.next_scope_alias);
                    self.next_scope_alias += 1;
                    self.prepared_counted_iterables
                        .insert(alias.clone(), iteration);
                    let Expr::For(fold) = prepared.to_mut() else {
                        unreachable!()
                    };
                    fold.iterable = ForIterable::Array(Expr::Name(alias));
                    fold.seed = seed.into_owned();
                }
            }
            Expr::Scope { .. } => {
                before_scope(self)?;
                let mut result = self
                    .capture_scoped_results(&[expr])?
                    .pop()
                    .expect("one scope")
                    .into_owned();
                if let Cow::Owned(nested) = self.prepare_scopes(&result, before_scope)? {
                    result = nested;
                }
                prepared = Cow::Owned(result);
            }
            Expr::Match { .. } if self.dslx_widths().is_none() => {
                // The lexical initializers in a match arm use DSLX inference
                // even if its enclosing block argument uses BX inference.
                return self
                    .with_dslx_value_context(|checker| checker.prepare_scopes(expr, before_scope));
            }
            Expr::Match { arms, .. } => {
                for (index, arm) in arms.iter().enumerate() {
                    if let Cow::Owned(value) = self.prepare_scopes(&arm.value, before_scope)? {
                        let Expr::Match { arms, .. } = prepared.to_mut() else {
                            unreachable!()
                        };
                        arms[index].value = value;
                    }
                }
            }
            Expr::Tuple(items) | Expr::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    if let Cow::Owned(item) = self.prepare_scopes(item, before_scope)? {
                        let (Expr::Tuple(items) | Expr::Array(items)) = prepared.to_mut() else {
                            unreachable!()
                        };
                        items[i] = item;
                    }
                }
            }
            Expr::Field { value, .. } | Expr::Index { value, .. } | Expr::Unary { value, .. } => {
                if let Cow::Owned(value) = self.prepare_scopes(value, before_scope)? {
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
                let lhs = self.prepare_scopes(left, before_scope)?;
                let rhs = if has_common_bit_type(*op) {
                    self.prepare_scopes(right, before_scope)?
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
                let lhs = self.prepare_scopes(then_value, before_scope)?;
                let rhs = self.prepare_scopes(else_value, before_scope)?;
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
            let Expr::Scope { bindings, result } = expr else {
                prepared.push(Cow::Borrowed(expr));
                continue;
            };
            let saved = self.enter_scope(bindings)?;
            let mut aliases = BTreeMap::new();
            for binding in bindings {
                binding.pattern.visit_names(&mut |name| {
                    if aliases.contains_key(name) {
                        return;
                    }
                    let alias = format!("$scope_{}", self.next_scope_alias);
                    self.next_scope_alias += 1;
                    let values = if self.in_loop_scope() {
                        &mut self.loop_values
                    } else {
                        &mut self.function_values
                    };
                    values.insert(alias.clone(), values[name].clone());
                    self.names.insert(alias.clone());
                    if let Some(widths) = self.inferred_bindings.get(name).cloned() {
                        self.inferred_bindings.insert(alias.clone(), widths);
                    }
                    aliases.insert(name.to_owned(), alias);
                });
            }
            let mut result = result.as_ref().clone();
            capture_names(&mut result, &mut aliases);
            self.leave_scope(saved);
            prepared.push(Cow::Owned(result));
        }
        Ok(prepared)
    }

    /// Materialize an initializer once under a name that survives lexical
    /// shadowing and cannot collide with source or loop-generated names.
    fn share_scoped_value(&mut self, value: TypedExpr) -> TypedExpr {
        if matches!(&value.kind, K::Literal(_) | K::Name(_)) {
            return value;
        }
        let name = format!("$let_{}", self.statements.len());
        let constexpr = value.has_constant_inputs(&self.constexpr_names);
        // Guarded statements move into their own body, so a later scoped
        // binding can reuse the same temporary name in the enclosing block.
        if constexpr {
            self.constexpr_names.insert(name.clone());
        } else {
            self.constexpr_names.remove(&name);
        }
        if let Some(constant) = self.fold_constant(&value) {
            self.constants.insert(name.clone(), constant);
        } else {
            self.constants.remove(&name);
        }
        let ty = value.ty.clone();
        self.statements.push(CheckedStmt::Let {
            name: name.clone(),
            value,
        });
        TypedExpr {
            ty,
            kind: K::Name(name),
        }
    }

    /// Project the members selected by a tuple pattern from its initializer.
    pub(super) fn bind_pattern(
        &mut self,
        pattern: &BindingPattern,
        value: TypedExpr,
        widths: Option<BindingWidths>,
        saved: &mut Vec<Shadowed>,
    ) -> Result<(), String> {
        match pattern {
            BindingPattern::Name(name) => {
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
                let Type::Tuple(types) = value.ty else {
                    return Err(format!("tuple pattern requires a tuple, got {}", value.ty));
                };
                let count = types.len();
                if !self.iteration_path.is_empty() {
                    // Selecting just one field can still scan the tuple or
                    // a long wildcard pattern. Charge once for the traversal.
                    self.charge_unroll(count.max(patterns.len()))?;
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
                        continue;
                    }
                    let index = if rest.is_some_and(|r| i > r) {
                        count - (patterns.len() - i)
                    } else {
                        i
                    };
                    let field = fields
                        .nth(index - cursor)
                        .expect("tuple pattern arity checked");
                    cursor = index + 1;
                    let widths = widths.as_mut().and_then(|w| w[index].take());
                    self.bind_pattern(pattern, field, widths, saved)?;
                }
            }
        }
        Ok(())
    }
}

/// Substitute stable aliases for free references to the enclosing scope.
/// Binders introduced by a nested value scope or fold hide outer aliases.
fn capture_names(expr: &mut Expr, aliases: &mut BTreeMap<String, String>) {
    if aliases.is_empty() {
        return;
    }
    fn rename(name: &mut String, aliases: &BTreeMap<String, String>) {
        if let Some((head, member)) = name.split_once("::") {
            if let Some(alias) = aliases.get(head) {
                *name = format!("{alias}::{member}");
            }
        } else if let Some(alias) = aliases.get(name) {
            name.clone_from(alias);
        }
    }
    fn rename_type(ty: &mut Type, aliases: &BTreeMap<String, String>) {
        match ty {
            Type::Named(name) => rename(name, aliases),
            Type::Array { element, .. } => rename_type(element, aliases),
            Type::Tuple(fields) => fields.iter_mut().for_each(|f| rename_type(f, aliases)),
            Type::Bits { .. } | Type::Bool => { /* Primitive types have no references. */ }
        }
    }
    match expr {
        Expr::Name(name) => rename(name, aliases),
        Expr::Bool(_) | Expr::Literal { ty: None, .. } => { /* No names to capture. */ }
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
        Expr::Tuple(fields) | Expr::Array(fields) => {
            fields.iter_mut().for_each(|f| capture_names(f, aliases))
        }
        Expr::Struct { name, fields } => {
            rename(name, aliases);
            for (_, value) in fields {
                capture_names(value, aliases);
            }
        }
        Expr::Call { name, args } => {
            rename(name, aliases);
            for arg in args {
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
        Expr::Scope { bindings, result } => {
            let mut hidden = Vec::new();
            for binding in bindings {
                if let Some(ty) = &mut binding.ty {
                    rename_type(ty, aliases);
                }
                capture_names(&mut binding.value, aliases);
                binding.pattern.visit_names(&mut |n| {
                    if let Some(pair) = aliases.remove_entry(n) {
                        hidden.push(pair);
                    }
                });
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
            match &mut fold.iterable {
                ForIterable::Range { start, limit, .. } => {
                    capture_names(start, aliases);
                    capture_names(limit, aliases);
                }
                ForIterable::Array(expr) => capture_names(expr, aliases),
            }
            capture_names(&mut fold.seed, aliases);
            let mut hidden = Vec::new();
            for pattern in [&fold.element, &fold.accumulator] {
                pattern.visit_names(&mut |name| hidden.extend(aliases.remove_entry(name)));
            }
            capture_names(&mut fold.body, aliases);
            aliases.extend(hidden);
        }
    }
}
