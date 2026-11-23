// SPDX-License-Identifier: Apache-2.0

use super::generic_blocks::callback_argument_name;
use super::*;
use crate::ast::{
    BindingPattern, ForIterable, FunctionParametricRef, MapExpr, ScopeStatement, StructArgumentRef,
    StructParametricRef, UnrollFor,
};
use crate::infer_literal::BindingWidths;
use std::cell::Cell;
use xlsynth_pir::IrBits;

/// Static bounds for a fold. Keep the index in its source bit width and
/// convert only the distance between endpoints to a host iteration count.
pub(super) struct StaticRange {
    pub ty: Type,
    pub first: IrBits,
    pub count: usize,
    pub widths: Option<BindingWidths>,
    pub operands: Vec<TypedExpr>,
}

/// State hidden by a loop-local name. Restore it when leaving the iteration.
pub(super) struct Shadowed {
    pub(super) name: String,
    value: Option<TypedExpr>,
    type_alias: Option<super::scope::LocalTypeAlias>,
    /// A function-local binding in a map can shadow an enclosing loop value.
    outer_loop_value: Option<TypedExpr>,
    widths: Option<BindingWidths>,
    declared: bool,
    /// A nearer binding can hide an eligible constant match pattern.
    outer_constant_pattern: Option<String>,
}

impl BodyChecker<'_, '_> {
    /// Structural iterations and generic specializations elaborate separate
    /// copies of their bodies.
    pub(super) fn replaying_body(&self) -> bool {
        !self.iteration_path.is_empty()
            || self.specialization.is_some()
            || self.function_spec.is_some()
    }

    /// Both elaborated iterations and counted bodies have lexical loop values.
    pub(super) fn in_loop_scope(&self) -> bool {
        self.in_counted_body
            || self.iteration_path.len() > self.map_scope.map_or(0, |s| s.outer_iterations)
    }

    pub(super) fn scoped_widths(&self) -> Option<WidthLookup<'_>> {
        (!self.loop_values.is_empty() || !self.function_values.is_empty())
            .then(|| WidthLookup::scoped_locals(&self.inferred_bindings, &self.loop_values))
    }

    /// Charge expanded folds and range operands to their source body.
    /// Specializations of a generic block share the module's expansion budget.
    pub(super) fn charge_expansion(&mut self, expression: &str, work: usize) -> Result<(), String> {
        let (total, kind) = match (
            &mut self.specializations,
            self.specialization,
            &mut self.function_registry,
            self.function_spec,
            self.source_kind,
        ) {
            (Some(registry), Some(_), _, _, _) => {
                (&mut registry.expansion_work, "generic block specialization")
            }
            (_, _, Some(registry), spec, _) if spec.is_some() || registry.checking_definition() => {
                (&mut registry.expansion_work, "function specialization")
            }
            (_, _, _, _, BodyKind::Block) => (&mut self.expansion_work, "block"),
            (_, _, _, _, BodyKind::Function) => (&mut self.expansion_work, "function"),
        };
        *total = total.saturating_add(work);
        if *total > MAX_TYPE_SIZE {
            return Err(format!(
                "{expression} exceeds the {kind} expansion limit of {MAX_TYPE_SIZE}"
            ));
        }
        Ok(())
    }

    /// Child-producing loops occupy an unconditional binding or block result.
    /// Other expression positions can use the same fold with pure values.
    pub(super) fn scoped_result(
        &mut self,
        expr: &Expr,
        expected: Option<&Type>,
        structural: bool,
    ) -> Result<TypedExpr, String> {
        if self.kind == BodyKind::Function && !matches!(expr, Expr::Scope { .. }) {
            return self.with_prepared_value(expr, |checker, value| {
                checker.elaborate_result(value, expected, structural)
            });
        }
        self.elaborate_result(expr, expected, structural)
    }

    /// Elaborate a result after its contributing lexical scopes are available.
    pub(super) fn elaborate_result(
        &mut self,
        expr: &Expr,
        expected: Option<&Type>,
        structural: bool,
    ) -> Result<TypedExpr, String> {
        match expr {
            Expr::UnrollFor(fold) => {
                self.unroll(fold, CarryContext::Expected(expected), structural)
            }
            _ if expected.is_none() => self.inferred_expr(expr),
            _ => self.expr(expr, expected),
        }
    }

    /// Check an ordinary binding or elaborate its child instance.
    pub(super) fn let_value(
        &mut self,
        name: Option<&str>,
        expr: &Expr,
        expected: Option<&Type>,
        structural: bool,
    ) -> Result<TypedExpr, String> {
        let call = match expr {
            Expr::Call { name, args } | Expr::MapBeforeUse { name, args, .. } => {
                Some((name, &[][..], args))
            }
            Expr::ParametricCall {
                name,
                parametrics,
                args,
            } => Some((name, parametrics.as_slice(), args)),
            _ => None,
        };
        if let Some((target, parametrics, args)) = call
            && self.decls.is_block(target)
            && self.callback_index(target).is_none()
        {
            if self.kind == BodyKind::Function {
                return Err(format!("pure function cannot call block '{target}'"));
            }
            if !structural {
                return Err(format!(
                    "call to block '{target}' requires an unconditional top-level let or unrolled let"
                ));
            }
            if self.loop_values.contains_key(target) {
                return Err(format!("'{target}' names a loop-local value"));
            }
            self.check_block_call_signature(
                target,
                parametrics.len(),
                self.function_values.contains_key(target)
                    || (!parametrics.is_empty() && self.names.contains(target)),
            )?;
            let binding = if let Some(name) = name
                && self.iteration_path.is_empty()
            {
                name.to_owned()
            } else {
                let base = match name {
                    Some(name) => {
                        let mut base = name.to_owned();
                        for index in &self.iteration_path {
                            base.push_str(&format!("_{index}"));
                        }
                        base
                    }
                    // A tuple pattern or wildcard has no source name for the
                    // instance. Unconditional instances occupy distinct
                    // positions in the checked statement list.
                    None => format!("bx_inst_{}", self.statements.len()),
                };
                let mut binding = base.clone();
                let mut suffix = 0;
                while !self.reserved_hardware_names.insert(binding.clone()) {
                    suffix += 1;
                    binding = format!("{base}_{suffix}");
                }
                binding
            };
            return self.block_call(&binding, target, parametrics, args, expected);
        }
        self.scoped_result(expr, expected, structural)
    }

    /// Share a loop-local initializer across uses without copying its
    /// expression tree into the next iteration. Constant carries need no
    /// hardware binding.
    fn share_loop_value(&mut self, value: TypedExpr) -> TypedExpr {
        if matches!(value.runtime_result().kind, K::Name(_) | K::Literal(_)) {
            return self.share_scoped_value(value);
        }
        let ty = value.ty.clone();
        let constexpr = value.has_constant_inputs(&self.constexpr_names);
        let constant = self.fold_constant(&value, constexpr);
        if constexpr {
            match constant {
                Some(ConstantBinding::Known(v)) => {
                    return TypedExpr {
                        ty,
                        kind: K::Literal(v),
                    };
                }
                Some(ConstantBinding::Selected { value: folded, .. }) => {
                    // The ordinary BX carry folded to a literal. Keep the
                    // original expression for stricter constexpr evaluation
                    // without lowering its arithmetic into the runtime loop.
                    let result = TypedExpr {
                        ty,
                        kind: K::Literal(folded),
                    };
                    if self.preserve_assert_inputs {
                        let operand = self.retain_constexpr_operand(value);
                        return scope::sequence(vec![operand], result);
                    }
                    return result;
                }
                _ => { /* A runtime or deferred carry still needs a shared binding. */ }
            }
        }
        // This name is internal to the checked block and cannot occur in BX.
        let name = if self.in_erased_operand {
            format!(
                "{}{}",
                crate::typed::CONSTEXPR_PREFIX,
                self.statements.len()
            )
        } else {
            format!("$unroll_{}", self.unroll_binding_count)
        };
        self.unroll_binding_count += 1;
        if constexpr {
            self.constexpr_names.insert(name.clone());
        }
        if let Some(v) = constant {
            self.constants.insert(name.clone(), v);
        }
        self.statements.push(CheckedStmt::Let {
            name: name.clone(),
            value,
        });
        TypedExpr {
            ty,
            kind: K::Name(name),
        }
    }

    /// Install a source binding, preserving shadowed outer values and literal
    /// inference.
    pub(super) fn bind_scoped_value(
        &mut self,
        name: &str,
        value: TypedExpr,
        widths: Option<BindingWidths>,
        saved: &mut Vec<Shadowed>,
    ) {
        if name == "_" {
            return;
        }
        self.shadow_scoped_name(name, saved);
        let scope = if self.in_loop_scope() {
            &mut self.loop_values
        } else {
            &mut self.function_values
        };
        scope.insert(name.to_owned(), value);
        if let Some(mut widths) = widths {
            widths.bind_in_scope(self.width_scope());
            self.inferred_bindings.insert(name.to_owned(), widths);
        }
    }

    /// Save the current meaning of a name before a declaration of either kind.
    pub(super) fn shadow_scoped_name(&mut self, name: &str, saved: &mut Vec<Shadowed>) {
        let (values, outer_loop_value) = if self.in_loop_scope() {
            (&mut self.loop_values, None)
        } else {
            (&mut self.function_values, self.loop_values.remove(name))
        };
        saved.push(Shadowed {
            name: name.to_owned(),
            value: values.remove(name),
            type_alias: self.local_type_aliases.remove(name),
            outer_loop_value,
            widths: self.inferred_bindings.remove(name),
            declared: self.names.insert(name.to_owned()),
            outer_constant_pattern: self.constant_pattern_names.take(name),
        });
    }

    pub(super) fn leave_scope(&mut self, saved: Vec<Shadowed>) {
        let in_loop = self.in_loop_scope();
        for old in saved.into_iter().rev() {
            self.constant_pattern_names.remove(&old.name);
            if let Some(ty) = old.type_alias {
                self.local_type_aliases.insert(old.name.clone(), ty);
            } else {
                self.local_type_aliases.remove(&old.name);
            }
            if let Some(name) = old.outer_constant_pattern {
                self.constant_pattern_names.insert(name);
            }
            let values = if in_loop {
                &mut self.loop_values
            } else {
                &mut self.function_values
            };
            if let Some(value) = old.value {
                values.insert(old.name.clone(), value);
            } else {
                values.remove(&old.name);
            }
            if let Some(value) = old.outer_loop_value {
                self.loop_values.insert(old.name.clone(), value);
            }
            if let Some(widths) = old.widths {
                self.inferred_bindings.insert(old.name.clone(), widths);
            } else {
                self.inferred_bindings.remove(&old.name);
            }
            if old.declared {
                self.names.remove(&old.name);
            }
        }
    }

    /// Resolve compatible bound types and count their distance without
    /// narrowing the source values, including ranges crossing signed zero.
    pub(super) fn static_range(
        &mut self,
        start: &Expr,
        limit: &Expr,
        hint: Option<&Type>,
        inclusive: bool,
        keyword: &str,
    ) -> Result<StaticRange, String> {
        let (lo, hi) = self.pair(start, limit, hint)?;
        let (signed, width) =
            bits(&lo.ty).ok_or_else(|| format!("{keyword} range requires integer bounds"))?;
        let inferred = hint.is_none()
            && binding_widths(start, &lo.ty, &self.inferred_bindings)?.is_some()
            && binding_widths(limit, &hi.ty, &self.inferred_bindings)?.is_some();
        let mut constant = |bound: &TypedExpr| {
            if bound.has_constant_inputs(&self.constexpr_names)
                && let Some(value) = self.checked_constant(bound)?
                && let Ok(bits) = value.to_bits()
            {
                return Ok(bits);
            }
            Err(format!(
                "{keyword} range bounds must be compile-time constants"
            ))
        };
        let first = constant(&lo)?;
        let last = constant(&hi)?;
        if if signed {
            last.slt(&first)
        } else {
            last.ult(&first)
        } {
            return Err(format!("{keyword} range limit precedes its start"));
        }
        let count = ir_bits_to_usize(&last.sub(&first))
            .and_then(|n| n.checked_add(usize::from(inclusive)))
            .ok_or_else(|| format!("{keyword} range too large"))?;
        let ty = lo.ty.clone();
        let operands = if self.preserve_assert_inputs {
            [lo, hi]
                .into_iter()
                .filter(|v| !matches!(v.kind, K::Literal(_)))
                .collect()
        } else {
            Vec::new()
        };
        Ok(StaticRange {
            ty,
            first,
            count,
            widths: inferred.then(|| BindingWidths::scalar(signed, width)),
            operands,
        })
    }

    /// Fold a constant half-open range in source order. Bounds may be wider
    /// than a host integer; only the iteration count needs to fit in memory.
    pub(super) fn unroll(
        &mut self,
        fold: &UnrollFor,
        context: CarryContext,
        structural: bool,
    ) -> Result<TypedExpr, String> {
        if fold.index == fold.accumulator && fold.index != "_" {
            return Err(format!("duplicate loop binding '{}'", fold.index));
        }
        let annotation = fold
            .annotation
            .as_ref()
            .map(|(i, a)| Ok::<_, String>((self.source_type(i)?, self.source_type(a)?)))
            .transpose()?;
        let index_hint = annotation.as_ref().map(|(i, _)| i);
        let range =
            self.static_range(&fold.start, &fold.limit, index_hint, false, "unroll_for!")?;
        let count = range.count;
        let width = range.first.get_bit_count();
        let mut index = range.first;
        // Charge nested loops to the same block. A small inner range repeated
        // many times consumes the same budget as a single large range.
        self.charge_expansion(
            "unroll_for!",
            count.saturating_mul(fold.body.len().saturating_add(width.div_ceil(64).max(1))),
        )?;
        let seed = self.loop_seed(
            &fold.seed,
            annotation.as_ref().map(|(_, a)| a),
            context,
            |checker, acc_hint| {
                if acc_hint.is_none() && minimum_type(&fold.seed)?.is_some() {
                    return Err(
                    "unroll_for! needs a typed seed or accumulator for an implicit literal result"
                        .into(),
                );
                }
                checker.scoped_result(&fold.seed, acc_hint, false)
            },
        )?;
        let mut carry = self.share_loop_value(seed);
        if count == 0 {
            self.check_empty_loop_names(fold)?;
            return Ok(scope::sequence(range.operands, carry));
        }
        let one = IrBits::from_lsb_fn(width, |i| i == 0);
        let mut previous_carries = Vec::new();
        for ordinal in 0..count {
            self.iteration_path.push(ordinal);
            let mut saved = Vec::new();
            let item = TypedExpr {
                ty: range.ty.clone(),
                kind: K::Literal(IrValue::Bits(index.clone())),
            };
            let acc_type = carry.ty.clone();
            self.bind_scoped_value(&fold.index, item, range.widths.clone(), &mut saved);
            // A constexpr helper executes the seed and each iteration even
            // when the next iteration ignores its accumulator. Nonliteral
            // carries are already shared, so each reference is stored once.
            if self.preserve_assert_inputs && !matches!(carry.kind, K::Literal(_)) {
                previous_carries.push(carry.clone());
            }
            self.bind_scoped_value(&fold.accumulator, carry, None, &mut saved);
            for stmt in &fold.body {
                if let Stmt::LetPattern(binding) = stmt {
                    let init = self
                        .pattern_initializer(binding, structural)
                        .map_err(|e| format!("unroll_for! iteration {ordinal}: {e}"))?;
                    self.bind_pattern(&binding.pattern, init.value, init.widths, &mut saved)
                        .map_err(|e| format!("unroll_for! iteration {ordinal}: {e}"))?;
                    continue;
                }
                let Stmt::Let {
                    name,
                    ty,
                    value: expr,
                } = stmt
                else {
                    return Err(
                        "unroll_for! bodies permit only let bindings before the result".into(),
                    );
                };
                let ty = ty.as_ref().map(|t| self.source_type(t)).transpose()?;
                let value = self
                    .let_value(Some(name), expr, ty.as_ref(), structural)
                    .map_err(|e| format!("unroll_for! iteration {ordinal}, let '{name}': {e}"))?;
                let widths = if ty.is_none() {
                    binding_widths(expr, &value.ty, &self.inferred_bindings)?
                } else {
                    None
                };
                let value = self.share_loop_value(value);
                self.bind_scoped_value(name, value, widths, &mut saved);
            }
            let value = self
                .scoped_result(&fold.result, Some(&acc_type), structural)
                .map_err(|e| format!("unroll_for! iteration {ordinal}: {e}"))?;
            carry = self.share_loop_value(value);
            self.leave_scope(saved);
            self.iteration_path.pop();
            index = index.add(&one);
        }
        let result = if previous_carries.is_empty() {
            carry
        } else {
            TypedExpr {
                ty: carry.ty.clone(),
                kind: K::Sequence {
                    operands: previous_carries,
                    result: Box::new(carry),
                    bytecode_only: true,
                },
            }
        };
        Ok(scope::sequence(range.operands, result))
    }
}

/// Declaration kinds needed for type and match-pattern lookup during a
/// lexical walk without evaluation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LexicalKind {
    Value,
    Constant,
    TypeAlias { can_construct: bool },
}

/// Names visible during lexical checking, including zero-trip bodies.
struct LexicalScope<'a> {
    /// The root borrows names from elaboration so skipped loops need no copy
    /// of the enclosing scope.
    enclosing_names: Option<&'a BTreeSet<String>>,
    declarations: BTreeMap<String, LexicalKind>,
    parent: Option<&'a LexicalScope<'a>>,
    in_lambda: bool,
    /// Shared across nested scopes in this one lexical walk, including
    /// zero-trip bodies that will never produce a typed expression.
    visited: &'a Cell<usize>,
}

impl<'a> LexicalScope<'a> {
    fn root(names: &'a BTreeSet<String>, visited: &'a Cell<usize>) -> Self {
        Self {
            enclosing_names: Some(names),
            declarations: BTreeMap::new(),
            parent: None,
            in_lambda: false,
            visited,
        }
    }

    fn nested(&self) -> LexicalScope<'_> {
        LexicalScope {
            enclosing_names: None,
            declarations: BTreeMap::new(),
            parent: Some(self),
            in_lambda: self.in_lambda,
            visited: self.visited,
        }
    }

    fn bind_pattern(&mut self, pattern: &BindingPattern) {
        let nodes = pattern.visit_names(&mut |name| self.bind_pattern_name(name));
        self.visited.set(self.visited.get().saturating_add(nodes));
    }

    /// An unrolled binding or lambda parameter can discard its value with `_`.
    fn bind_pattern_name(&mut self, name: &str) {
        if name != "_" {
            self.bind(name, LexicalKind::Value);
        }
    }

    fn bind(&mut self, name: &str, kind: LexicalKind) {
        self.declarations.insert(name.to_owned(), kind);
    }

    /// Find the lexical scope that introduced a visible name.
    fn lookup(&self, name: &str) -> Option<&Self> {
        let mut scope = Some(self);
        while let Some(current) = scope {
            if current.declarations.contains_key(name)
                || current
                    .enclosing_names
                    .is_some_and(|names| names.contains(name))
            {
                return Some(current);
            }
            scope = current.parent;
        }
        None
    }

    fn contains(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }
}

/// The syntactic role of a nominal name in a skipped loop body.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NominalUse {
    Annotation,
    StructField,
    Constructor,
    AppliedStruct,
}

impl BodyChecker<'_, '_> {
    /// Combine declaration kinds seen in this walk with the concrete
    /// enclosing scope, preserving lexical shadowing of either kind.
    fn lexical_kind(&self, locals: &LexicalScope<'_>, name: &str) -> Option<LexicalKind> {
        let scope = locals.lookup(name)?;
        Some(scope.declarations.get(name).copied().unwrap_or_else(|| {
            if let Some(alias) = self.local_type_aliases.get(name) {
                LexicalKind::TypeAlias {
                    can_construct: alias.can_construct,
                }
            } else if self.constant_pattern_names.contains(name) {
                LexicalKind::Constant
            } else {
                LexicalKind::Value
            }
        }))
    }

    /// Resolve template names in source order without evaluating any bindings.
    pub(super) fn check_struct_template_names(&mut self, def: &StructDef) -> Result<(), String> {
        self.function_position = self.decls.positions.get(def.name.as_str()).copied();
        self.zero_binding = self.decls.zero_binding(&def.name);
        self.template_name_depth = Some(Cell::new(0));
        let visited = Cell::new(0);
        let mut locals = LexicalScope::root(&self.names, &visited);
        for formal in def.parametric_bindings() {
            match formal {
                StructParametricRef::Type(name) => {
                    self.function_type_bindings
                        .get_or_insert_with(|| super::Bindings::new(&[]))
                        .declare_type_parameter(name);
                }
                StructParametricRef::Value(binding) => {
                    // The formal hides an outer type even in its own
                    // annotation.
                    locals.bind(&binding.name, LexicalKind::Constant);
                    self.check_type_names(&binding.ty, &locals)?;
                    self.constant_pattern_names.insert(binding.name.clone());
                    if let Some(default) = &binding.default {
                        self.check_expr_names(default, &locals)?;
                    }
                }
            }
        }
        for field in &def.fields {
            self.check_annotation_names(&field.ty, &locals, NominalUse::StructField)?;
        }
        Ok(())
    }

    /// Resolve names at a template's declaration while leaving concrete types
    /// and evaluation to the call that supplies its parametric values.
    pub(super) fn check_value_template_names(&mut self, f: &Function) -> Result<(), String> {
        let position = self.decls.positions.get(f.name.as_str()).copied();
        // The signature precedes the binding of the function name.
        self.function_position = position.map(|at| at.saturating_sub(1));
        self.zero_binding = self.decls.zero_binding(&f.name);
        if !f.type_params.is_empty() {
            self.function_type_bindings = Some(super::Bindings::new(&f.type_params));
        }
        self.template_name_depth = Some(Cell::new(0));
        let visited = Cell::new(0);
        let mut locals = LexicalScope::root(&self.names, &visited);
        for formal in f.parametric_bindings() {
            match formal {
                FunctionParametricRef::Value(binding) => {
                    // DSLX introduces the formal before its annotation and
                    // default.
                    locals.bind(&binding.name, LexicalKind::Constant);
                    self.check_type_names(&binding.ty, &locals)?;
                    self.constant_pattern_names.insert(binding.name.clone());
                    if let Some(default) = &binding.default {
                        self.check_expr_names(default, &locals)?;
                    }
                }
                FunctionParametricRef::Callback(callback) => {
                    for ty in callback.params.iter().chain([&callback.result]) {
                        self.check_type_names(ty, &locals)?;
                    }
                    locals.bind_pattern_name(&callback.name);
                }
            }
        }
        // From here each callback has its declared callable role, including as
        // a named map operand. A nearer local may still shadow that binding.
        for callback in &f.callback_params {
            locals.declarations.remove(&callback.name);
        }
        self.template_callbacks = self
            .decls
            .value_template_callbacks
            .get(f.name.as_str())
            .or_else(|| {
                self.decls
                    .functions
                    .get(f.name.as_str())
                    .map(|f| &f.callback_indices)
            });
        for param in &f.params {
            self.check_type_names(&param.ty, &locals)?;
            locals.bind_pattern_name(&param.name);
        }
        self.check_type_names(&f.result, &locals)?;
        self.function_position = position;
        self.check_expr_names(&f.body, &locals)
    }

    /// Expressions and symbolic types can contain one another. Bound their
    /// combined nesting before a template signature is retained or evaluated.
    fn with_template_name_depth(
        &self,
        check: impl FnOnce() -> Result<(), String>,
    ) -> Result<(), String> {
        let Some(depth) = &self.template_name_depth else {
            return check();
        };
        let prior = depth.get();
        if prior > MAX_ELABORATION_DEPTH {
            return Err(format!(
                "value-parametric template nesting exceeds {MAX_ELABORATION_DEPTH} levels"
            ));
        }
        depth.set(prior + 1);
        let result = check();
        depth.set(prior);
        result
    }

    /// Preserve ordinary calls to visible declarations named `map`.
    pub(super) fn is_builtin_map(
        &self,
        target: &str,
        is_local: impl FnOnce(&str) -> bool,
    ) -> Result<bool, String> {
        if target == crate::bx_import::BUILTIN_MAP {
            return Ok(true);
        }
        let spelling = crate::bx_import::library_item(target).map_or(target, |(_, name)| name);
        if spelling != "map"
            || is_local(target)
            || self.callback_index(target).is_some()
            || self
                .type_bindings()
                .is_some_and(|bindings| bindings.declares(target))
        {
            return Ok(false);
        }
        // A block can call imported functions declared later. An imported
        // enum or type alias shadows the builtin once its use enters value
        // scope.
        if !self.decls.visible(target, self.block_position)
            && let Some((subject, member)) = self.resolver.lookup_enum_prefix(target)?
            && self.resolver.is_enum_or_alias(&subject, &member)
        {
            return Ok(true);
        }
        if target != "map" {
            return Ok(false);
        }
        Ok(!self.decls.visible(target, self.function_position)
            || !(self.decls.declares_function(target)
                || self.decls.is_block(target)
                || self.structs.indices.contains_key(target)
                || (self.structs.template(target).is_some()
                    || self.structs.enums.contains_key(target)
                    || self.decls.type_aliases.contains_key(target))
                    && self
                        .decls
                        .visible(target, self.function_position.or(self.block_position))
                || self.module_constant(target).is_some()
                || self.resolver.lookup(target)?.is_some()))
    }

    /// A lambda argument is available only on the built-in map.
    fn check_map_target(&self, map: &MapExpr, locals: &LexicalScope<'_>) -> Result<(), String> {
        if locals.contains(&map.target) {
            return Err("'map' names a local value here".into());
        }
        if !self.is_builtin_map(&map.target, |_| false)? {
            return Err("an inline lambda argument requires the built-in map".into());
        }
        Ok(())
    }

    fn check_map_body_names(
        &self,
        map: &MapExpr,
        enclosing: &LexicalScope<'_>,
    ) -> Result<(), String> {
        if let Some(ty) = &map.parameter_type {
            self.check_type_names(ty, enclosing)?;
        }
        let mut body_scope = enclosing.nested();
        body_scope.in_lambda = true;
        body_scope.bind_pattern_name(&map.parameter);
        if let Some(ty) = &map.result_type {
            self.check_type_names(ty, &body_scope)?;
        }
        self.check_expr_names(&map.body, &body_scope)
    }

    /// Validate the entire lambda subtree, including untaken branches and
    /// zero-trip structural loops, before any constants can be substituted.
    pub(super) fn check_map_names(&mut self, map: &MapExpr) -> Result<(), String> {
        let visited = Cell::new(0);
        let enclosing = LexicalScope::root(&self.names, &visited);
        self.check_map_target(map, &enclosing)?;
        if self.map_scope.is_none() {
            self.check_map_body_names(map, &enclosing)?;
            if self.replaying_body() {
                self.charge_expansion("map lambda", visited.get())?;
            }
        }
        Ok(())
    }

    /// DSLX resolves lexical names before unrolling, even when it never
    /// typechecks an iteration. Keep that distinction for an empty range.
    fn check_empty_loop_names(&mut self, fold: &UnrollFor) -> Result<(), String> {
        if self.map_scope.is_some() {
            // The enclosing callback already resolved names in this subtree.
            return Ok(());
        }
        let visited = Cell::new(0);
        let locals = LexicalScope::root(&self.names, &visited);
        self.check_body_names(fold, &mut locals.nested())?;
        if self.replaying_body() {
            self.charge_expansion("unroll_for!", visited.get())?;
        }
        Ok(())
    }

    /// Resolve lexical names before a template has the bindings to evaluate the
    /// expression.
    pub(super) fn check_deferred_expr_names(&self, expr: &Expr) -> Result<(), String> {
        let visited = Cell::new(0);
        self.check_expr_names(expr, &LexicalScope::root(&self.names, &visited))
    }

    fn check_body_names(
        &self,
        fold: &UnrollFor,
        locals: &mut LexicalScope<'_>,
    ) -> Result<(), String> {
        if fold.index == fold.accumulator && fold.index != "_" {
            return Err(format!("duplicate loop binding '{}'", fold.index));
        }
        locals.bind_pattern_name(&fold.index);
        locals.bind_pattern_name(&fold.accumulator);
        for stmt in &fold.body {
            if let Stmt::LetPattern(binding) = stmt {
                self.validate_source_pattern(&binding.pattern)?;
                if let Some(ty) = &binding.ty {
                    self.check_type_names(ty, locals)?;
                }
                self.check_expr_names(&binding.value, locals)?;
                locals.bind_pattern(&binding.pattern);
                continue;
            }
            let Stmt::Let { name, value, ty } = stmt else {
                return Err("unroll_for! bodies permit only let bindings before the result".into());
            };
            if let Some(ty) = ty {
                self.check_type_names(ty, locals)?;
            }
            self.check_expr_names(value, locals)?;
            locals.bind_pattern_name(name);
        }
        self.check_expr_names(&fold.result, locals)
    }

    /// An enclosing block port shadows a module name inside a symbolic type
    /// argument.
    fn reject_enclosing_type_local(&self, head: &str) -> Result<(), String> {
        if self
            .enclosing_type_locals
            .is_some_and(|contains| contains(head))
        {
            return Err(format!(
                "'{head}' names a local value in this type argument"
            ));
        }
        Ok(())
    }

    /// Resolve the base name of an annotation or constructor without
    /// instantiating a type in a lexical-only walk.
    fn check_nominal_name(
        &self,
        name: &str,
        usage: NominalUse,
        locals: &LexicalScope<'_>,
    ) -> Result<(), String> {
        if let Some(kind) = self.lexical_kind(locals, name) {
            return match kind {
                LexicalKind::TypeAlias { .. } if usage == NominalUse::AppliedStruct => {
                    Err(format!("type alias '{name}' takes no arguments"))
                }
                LexicalKind::TypeAlias {
                    can_construct: false,
                } if usage == NominalUse::Constructor => {
                    Err(format!("type alias '{name}' does not name a struct"))
                }
                LexicalKind::TypeAlias { .. } => Ok(()),
                _ => Err(format!("'{name}' names a local value here")),
            };
        }
        let head = name.split_once("::").map_or(name, |(head, _)| head);
        if !locals.contains(head) {
            self.reject_enclosing_type_local(head)?;
        }
        if self.callback_index(head).is_some() {
            return Err(format!(
                "callable parameter '{head}' may only be used as a callee"
            ));
        }
        if let Some(bindings) = self.type_bindings() {
            if usage != NominalUse::AppliedStruct && bindings.declares(name) {
                // A type parameter can lexically name a record even if this
                // specialization skips the body and binds it to a scalar.
                return Ok(());
            }
            bindings.reject_value(name)?;
        }
        if usage == NominalUse::StructField && self.structs.definition(name).is_ok() {
            // The layout schedule has already checked forward references to
            // ordinary structs that appear directly in field annotations.
            return Ok(());
        }
        self.decls
            .check_type_name_visibility(name, self.function_position)?;
        if let Some(alias) = self.decls.type_aliases.get(name) {
            return match usage {
                NominalUse::AppliedStruct => Err(format!("type alias '{name}' takes no arguments")),
                NominalUse::Constructor if !matches!(alias.ty, Type::Named(_)) => {
                    Err(format!("type alias '{name}' does not name a struct"))
                }
                _ => Ok(()),
            };
        }
        if let Some((module, member)) =
            lookup_use(name, self.resolver, |binding| locals.contains(binding))?
        {
            if usage == NominalUse::AppliedStruct {
                self.resolver.check_struct_template_name(&module, &member)
            } else if usage == NominalUse::Constructor
                && !self.resolver.is_struct_type(&module, &member)
                && self
                    .resolver
                    .struct_template_arity(&module, &member)
                    .is_none()
            {
                Err(format!("'{name}' does not name a struct"))
            } else {
                self.resolver.check_member_name(&module, &member)
            }
        } else if self.structs.template(name).is_some() {
            Ok(())
        } else if usage == NominalUse::AppliedStruct {
            Err(format!("'{name}' requires a struct template"))
        } else if self.structs.declared_records.contains(name) {
            // Name verification only needs the declaration. A specialization
            // that uses this layout will resolve its fields during elaboration.
            Ok(())
        } else if matches!(usage, NominalUse::Annotation | NominalUse::StructField)
            && self.structs.local_enum(name).is_some()
        {
            Ok(())
        } else {
            self.structs.definition(name).map(|_| ())
        }
    }

    /// Resolve the names in a struct application using each argument's kind.
    fn check_struct_argument_names(
        &self,
        args: StructLiteralArgs<'_>,
        locals: &LexicalScope<'_>,
        usage: NominalUse,
    ) -> Result<(), String> {
        for arg in args.iter() {
            match arg {
                StructArgumentRef::Type(ty) => self.check_annotation_names(ty, locals, usage)?,
                StructArgumentRef::Value(value) => self.check_expr_names(value, locals)?,
                StructArgumentRef::Integer(_) => { /* A literal has no names to resolve. */ }
            }
        }
        Ok(())
    }

    fn check_type_names(&self, ty: &Type, locals: &LexicalScope<'_>) -> Result<(), String> {
        self.check_annotation_names(ty, locals, NominalUse::Annotation)
    }

    fn check_annotation_names(
        &self,
        ty: &Type,
        locals: &LexicalScope<'_>,
        usage: NominalUse,
    ) -> Result<(), String> {
        self.with_template_name_depth(|| self.check_type_names_at_depth(ty, locals, usage))
    }

    fn check_type_names_at_depth(
        &self,
        ty: &Type,
        locals: &LexicalScope<'_>,
        usage: NominalUse,
    ) -> Result<(), String> {
        locals.visited.set(locals.visited.get().saturating_add(1));
        match ty {
            Type::Named(name) => self.check_nominal_name(name, usage, locals)?,
            Type::Applied { .. }
            | Type::ParametricStruct { .. }
            | Type::TypeApplied { .. }
            | Type::MixedApplied { .. } => {
                let (name, args) = ty.struct_application().expect("struct application");
                self.check_nominal_name(name, NominalUse::AppliedStruct, locals)?;
                self.check_struct_argument_names(args, locals, usage)?;
            }
            Type::Tuple(items) => {
                for item in items {
                    self.check_annotation_names(item, locals, usage)?;
                }
            }
            Type::Array { element, .. } => self.check_annotation_names(element, locals, usage)?,
            Type::ParametricArray { element, size } => {
                self.check_annotation_names(element, locals, usage)?;
                self.check_expr_names(size, locals)?;
            }
            Type::ParametricBits { signed, width } => {
                self.check_expr_names(signed, locals)?;
                self.check_expr_names(width, locals)?;
            }
            Type::Bool | Type::Bits { .. } | Type::Enum { .. } => {
                // Built-in types and resolved enums require no lexical lookup.
            }
        }
        Ok(())
    }

    /// An unused call still needs a visible callee, but DSLX checks its kind
    /// and arity only when it elaborates an iteration.
    fn check_value_name(
        &self,
        name: &str,
        locals: &LexicalScope<'_>,
        callee: bool,
    ) -> Result<(), String> {
        let name = crate::bx_import::local_callee_name(name).unwrap_or(name);
        let name = self.qualified_extern(name).unwrap_or(name);
        if name == "_" {
            return Err("cannot read the wildcard binding '_'".into());
        }
        let head = name.split_once("::").map_or(name, |(head, _)| head);
        if matches!(
            self.lexical_kind(locals, head),
            Some(LexicalKind::TypeAlias { .. })
        ) {
            if head == name && !callee {
                return Err(format!("type alias '{name}' cannot be used as a value"));
            }
            if name
                .rsplit_once("::")
                .is_some_and(|(prefix, _)| prefix != head)
            {
                return Err(format!(
                    "a member of type alias '{head}' cannot be a qualifier"
                ));
            }
            return Ok(());
        }
        if locals.contains(name) {
            return Ok(());
        }
        if !locals.contains(head) {
            self.reject_enclosing_type_local(head)?;
        }
        if self.callback_index(head).is_some() {
            return Err(format!(
                "callable parameter '{head}' may only be used as a callee"
            ));
        }
        self.reject_type_parameter_value(name)?;
        if callee
            && super::is_call_builtin(name.strip_prefix(super::BUILTIN_PREFIX).unwrap_or(name))
        {
            return Ok(());
        }
        if !self.decls.visible(name, self.function_position) {
            return Err(format!("'{name}' is declared later in the module"));
        }
        if self.module_constant(name).is_some()
            || self.decls.declares_function(name)
            || self.decls.is_block(name)
            || callee
                && (self.structs.local_enum(name).is_some()
                    || self.decls.type_aliases.contains_key(name))
                && self
                    .decls
                    .visible(name, self.function_position.or(self.block_position))
        {
            return Ok(());
        }
        if self.is_enum_member_name(name)? {
            if locals.contains(head) {
                return Err(format!("'{head}' names a local value here"));
            }
            return Ok(());
        }
        if let Some((module, member)) =
            lookup_use(name, self.resolver, |binding| locals.contains(binding))?
        {
            if !callee
                && self.resolver.constant_binding(name).is_some()
                && !self.decls.visible(name, self.block_position)
            {
                return Err(format!("'{name}' is declared later in the module"));
            }
            if !callee && name.contains("::") && self.resolver.is_enum_type(&module, &member) {
                return Err(format!("enum '{name}' cannot be used as a value"));
            }
            return self.resolver.check_member_name(&module, &member);
        }
        Err(format!("unknown value '{name}'"))
    }

    fn check_expr_names(&self, expr: &Expr, locals: &LexicalScope<'_>) -> Result<(), String> {
        self.with_template_name_depth(|| self.check_expr_names_at_depth(expr, locals))
    }

    fn check_expr_names_at_depth(
        &self,
        expr: &Expr,
        locals: &LexicalScope<'_>,
    ) -> Result<(), String> {
        locals.visited.set(locals.visited.get().saturating_add(1));
        match expr {
            Expr::Scope { statements, result } => {
                let mut names = locals.nested();
                for statement in statements {
                    match statement {
                        ScopeStatement::Let(binding) => {
                            self.validate_source_pattern(&binding.pattern)?;
                            if let Some(ty) = &binding.ty {
                                self.check_type_names(ty, &names)?;
                            }
                            self.check_expr_names(&binding.value, &names)?;
                            names.bind_pattern(&binding.pattern);
                        }
                        ScopeStatement::Const(def) => {
                            if let Some(ty) = &def.ty {
                                self.check_type_names(ty, &names)?;
                            }
                            self.check_expr_names(&def.value, &names)?;
                            names.bind(&def.name, LexicalKind::Constant);
                        }
                        ScopeStatement::TypeAlias(alias) => {
                            self.check_type_names(&alias.ty, &names)?;
                            // Record the target's constructor eligibility now,
                            // before another declaration can shadow any name in
                            // the annotation.
                            let can_construct = match &alias.ty {
                                Type::Named(name) => self
                                    .check_nominal_name(name, NominalUse::Constructor, &names)
                                    .is_ok(),
                                Type::Applied { .. }
                                | Type::MixedApplied { .. }
                                | Type::TypeApplied { .. }
                                | Type::ParametricStruct { .. } => true,
                                _ => false,
                            };
                            names.bind(&alias.name, LexicalKind::TypeAlias { can_construct });
                        }
                        ScopeStatement::ConstAssert(assertion) => {
                            self.check_expr_names(&assertion.predicate, &names)?;
                        }
                        ScopeStatement::Expression(expr) => self.check_expr_names(expr, &names)?,
                    }
                }
                self.check_expr_names(result, &names)?;
            }
            Expr::Name(name) => self.check_value_name(name, locals, /* callee= */ false)?,
            Expr::Bool(_) | Expr::ByteString(_) | Expr::Literal { ty: None, .. } => {
                /* No names to resolve. */
            }
            Expr::Zero(ty) | Expr::Literal { ty: Some(ty), .. } => {
                if matches!(expr, Expr::Zero(_))
                    && !locals.contains("zero!")
                    && self.callback_index("zero!").is_some()
                {
                    return Err("callable parameter 'zero!' requires a direct call".into());
                }
                self.check_type_names(ty, locals)?
            }
            Expr::TypeBuiltin { types, args, .. } => {
                for ty in types {
                    self.check_type_names(ty, locals)?;
                }
                for arg in args {
                    self.check_expr_names(arg, locals)?;
                }
            }
            Expr::Call { name, args }
            | Expr::MapBeforeUse {
                name,
                builtin_args: Some(args),
                ..
            } if self.is_builtin_map(name, |target| locals.contains(target))? => {
                // A skipped iteration resolves names without checking call
                // arity.
                for (index, arg) in args.iter().enumerate() {
                    match (index, arg) {
                        (1, Expr::Name(callback)) => {
                            self.check_map_callee_name(callback, locals)?
                        }
                        _ => self.check_expr_names(arg, locals)?,
                    }
                }
            }
            Expr::MapBeforeUse { name, .. }
                if self.is_builtin_map(name, |target| locals.contains(target))? =>
            {
                return Err("invalid operand syntax for the built-in map".into());
            }
            Expr::Tuple(items)
            | Expr::Array(items)
            | Expr::ArrayLiteral { members: items, .. }
            | Expr::Call { args: items, .. }
            | Expr::MapBeforeUse { args: items, .. } => {
                if let Expr::ArrayLiteral {
                    annotation: Some(ty),
                    ..
                } = expr
                {
                    self.check_type_names(ty, locals)?;
                }
                if let Expr::Call { name, .. } | Expr::MapBeforeUse { name, .. } = expr {
                    let local = locals.lookup(name);
                    if self.callback_index(name).is_none() && self.decls.is_block(name) {
                        if local.is_none() {
                            if self.kind == BodyKind::Function || locals.in_lambda {
                                return Err(format!("pure function cannot call block '{name}'"));
                            }
                            self.check_block_call_signature(name, 0, false)?;
                            self.has_block_only_behavior.set(true);
                        } else if self.kind != BodyKind::Function
                            && !locals.in_lambda
                            && local.is_some_and(|scope| scope.parent.is_none())
                            && !self.function_values.contains_key(name)
                            && !self.loop_values.contains_key(name)
                        {
                            // A hardware port or wire can share the structural
                            // child's name. A nearer lexical or iteration
                            // binder shadows it.
                            self.check_block_call_signature(name, 0, false)?;
                            self.has_block_only_behavior.set(true);
                        }
                    }
                    if self.callback_index(name).is_none() || local.is_some() {
                        self.check_value_name(name, locals, /* callee= */ true)?;
                    }
                }
                for item in items {
                    self.check_expr_names(item, locals)?;
                }
            }
            Expr::ParametricCall {
                name,
                parametrics,
                args,
            } => {
                let block = self.decls.is_block(name) && self.callback_index(name).is_none();
                let template = self.decls.function_templates.get(name.as_str());
                if block {
                    self.check_block_call_signature(
                        name,
                        parametrics.len(),
                        locals.contains(name),
                    )?;
                    self.has_block_only_behavior.set(true);
                    if self.kind == BodyKind::Function || locals.in_lambda {
                        return Err(format!("pure function cannot call block '{name}'"));
                    }
                } else if template.is_some_and(|f| !f.has_inferred_parametrics()) {
                    self.callable_template(name, locals.contains(name), parametrics.len())?;
                } else {
                    self.check_value_name(name, locals, /* callee= */ true)?;
                }
                let block_formals = self.decls.templates.get(name.as_str()).filter(|_| block);
                let mut formals = block_formals
                    .into_iter()
                    .flat_map(|t| t.block.parametric_bindings())
                    .chain(template.into_iter().flat_map(|f| f.parametric_bindings()));
                for actual in parametrics {
                    if matches!(formals.next(), Some(FunctionParametricRef::Callback(_))) {
                        self.check_callback_argument_names(actual, locals)?;
                    } else {
                        self.check_expr_names(actual, locals)?;
                    }
                }
                for arg in args {
                    self.check_expr_names(arg, locals)?;
                }
            }
            Expr::Struct { .. }
            | Expr::AppliedStruct { .. }
            | Expr::ParametricStruct { .. }
            | Expr::TypeAppliedStruct { .. }
            | Expr::MixedAppliedStruct { .. } => {
                locals.visited.set(locals.visited.get().saturating_add(1));
                let literal = expr.struct_literal().expect("struct constructor");
                let usage = match literal.args {
                    StructLiteralArgs::Omitted => NominalUse::Constructor,
                    _ => NominalUse::AppliedStruct,
                };
                self.check_nominal_name(literal.name, usage, locals)?;
                self.check_struct_argument_names(literal.args, locals, NominalUse::Annotation)?;
                for (_, value) in literal.fields {
                    self.check_expr_names(value, locals)?;
                }
                if let Some(base) = literal.base {
                    self.check_expr_names(&base.value, locals)?;
                }
            }
            Expr::Cast { value, ty } => {
                self.check_type_names(ty, locals)?;
                self.check_expr_names(value, locals)?;
            }
            Expr::Field { value, .. } | Expr::Unary { value, .. } => {
                self.check_expr_names(value, locals)?
            }
            Expr::Index {
                value: left,
                index: right,
            }
            | Expr::Binary { left, right, .. } => {
                self.check_expr_names(left, locals)?;
                self.check_expr_names(right, locals)?;
            }
            Expr::WidthSlice {
                value,
                start,
                width,
            } => {
                self.check_expr_names(value, locals)?;
                self.check_expr_names(start, locals)?;
                match width {
                    WidthSliceType::Concrete(ty) => self.check_type_names(ty, locals)?,
                    WidthSliceType::Bits { signed, bit_count } => {
                        self.check_expr_names(signed, locals)?;
                        self.check_expr_names(bit_count, locals)?;
                    }
                }
            }
            Expr::If {
                condition,
                then_value,
                else_value,
            } => {
                self.check_expr_names(condition, locals)?;
                self.check_expr_names(then_value, locals)?;
                self.check_expr_names(else_value, locals)?;
            }
            Expr::Match { subject, arms } => {
                for operand in super::match_operands(subject, arms, |name| {
                    let head = name.split_once("::").map_or(name, |(head, _)| head);
                    if let Some(kind) = self.lexical_kind(locals, head) {
                        return Ok(if head == name {
                            kind == LexicalKind::Constant
                        } else {
                            matches!(kind, LexicalKind::TypeAlias { .. })
                        });
                    }
                    Ok(self.callback_index(head).is_none() && self.is_constant_pattern(name)?)
                })? {
                    self.check_expr_names(operand, locals)?;
                }
                for arm in arms {
                    let pattern_nodes = arm.patterns.iter().flat_map(MatchPattern::nodes).count();
                    locals
                        .visited
                        .set(locals.visited.get().saturating_add(pattern_nodes));
                    self.check_expr_names(&arm.value, locals)?;
                }
            }
            Expr::UnrollFor(fold) => {
                self.check_expr_names(&fold.start, locals)?;
                self.check_expr_names(&fold.limit, locals)?;
                self.check_expr_names(&fold.seed, locals)?;
                if let Some((index, acc)) = &fold.annotation {
                    self.check_type_names(index, locals)?;
                    self.check_type_names(acc, locals)?;
                }
                self.check_body_names(fold, &mut locals.nested())?;
            }
            Expr::Map(map) => {
                self.check_map_target(map, locals)?;
                self.check_iterable_names(&map.iterable, locals)?;
                self.check_map_body_names(map, locals)?;
            }
            Expr::NamedMap {
                target,
                iterable,
                callback,
                parametrics,
            } => {
                if !self.is_builtin_map(target, |target| locals.contains(target))? {
                    return Err("this named map requires the built-in map".into());
                }
                self.check_iterable_names(iterable, locals)?;
                self.check_map_callee_name(callback, locals)?;
                for actual in parametrics {
                    self.check_expr_names(actual, locals)?;
                }
            }
            Expr::For(fold) => {
                self.check_iterable_names(&fold.iterable, locals)?;
                if let Some((element, carry)) = &fold.annotation {
                    self.check_type_names(element, locals)?;
                    self.check_type_names(carry, locals)?;
                }
                self.check_expr_names(&fold.seed, locals)?;
                let mut names = locals.nested();
                self.validate_counted_binders(fold)?;
                for pattern in [&fold.element, &fold.accumulator] {
                    names.bind_pattern(pattern);
                }
                self.check_expr_names(&fold.body, &names)?;
            }
        }
        Ok(())
    }

    fn check_map_callee_name(
        &self,
        callback: &str,
        locals: &LexicalScope<'_>,
    ) -> Result<(), String> {
        locals.visited.set(locals.visited.get().saturating_add(1));
        self.map_callee(callback, |head| locals.contains(head))
            .map(|_| ())
    }

    /// Resolve a named callable in a skipped body without binding its
    /// signature.
    fn check_callback_argument_names(
        &self,
        actual: &Expr,
        locals: &LexicalScope<'_>,
    ) -> Result<(), String> {
        locals.visited.set(locals.visited.get().saturating_add(1));
        self.callback_source(callback_argument_name(actual)?, |head| {
            locals.contains(head)
        })
        .map(|_| ())
    }

    fn check_iterable_names(
        &self,
        iterable: &ForIterable,
        locals: &LexicalScope<'_>,
    ) -> Result<(), String> {
        match iterable {
            ForIterable::Range { start, limit, .. } => {
                self.check_expr_names(start, locals)?;
                self.check_expr_names(limit, locals)
            }
            ForIterable::Array(array) => self.check_expr_names(array, locals),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{ImportOptions, check_module, check_module_with_options, parse_module};
    use std::fs;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};

    #[test]
    fn unelaborated_function_scopes_resolve_local_declarations_in_order() {
        for (body, valid) in [
            (
                "type W = u8; const K = W:3; match c { K => c, _ => W:0 }",
                true,
            ),
            ("type M = Mode; match M::A { M::A => c, _ => c }", true),
            ("type W = Bits; type A = W; let y = A { x: c }; c", false),
            ("type A = Mode; let y = A { x: c }; c", false),
            ("type A = uN[c]; let y = A { x: c }; c", false),
            ("type A = S; type B = A; type A = u8; B { x: c }.x", true),
            ("type A = Mode; let y = A::Missing; c", true),
            ("type A = Mode; let y = A::A::Missing; c", false),
            ("const V = c; type W = uN[c]; c", true),
            ("type W = u8; W", false),
            ("type W = u8; const K = missing; c", false),
            ("let y = { type W = u8; W:0 }; let z: W = c; c", false),
        ] {
            for function in [
                format!("fn f<N:u32>(c:u8) -> u8 {{ {body} }}"),
                format!(
                    "fn f(x:u8) -> u8 {{ unroll_for! (_, c) in u32:0..u32:0 {{ {body} }}(x) }}"
                ),
            ] {
                let source = format!(
                    "enum Mode:u1 {{ A=0, B=1 }}\ntype Bits = u8;\nstruct S {{ x:u8 }}\n{function}"
                );
                let native =
                    parse_and_typecheck(&source, "scope.x", "scope", &mut ImportData::default());
                assert_eq!(native.is_ok(), valid, "{source}: {:?}", native.err());
                let bx = parse_module(&source).and_then(|m| check_module(&m));
                assert_eq!(bx.is_ok(), valid, "{source}: {:?}", bx.err());
            }
        }
    }

    #[test]
    fn unused_value_templates_resolve_lexical_names_in_declaration_order() {
        let declarations = "const BASE = u32:4; fn value<N:u32>(x:u32)->u32 { x + N }";
        for (function, valid) in [
            (
                "fn f<N:u32={BASE}, M:u32={N+BASE}>(x:uN[M])->u32 { value<{M+N}>(x as u32) }",
                true,
            ),
            ("fn f<N:u32>(x:uN[N])->u32 { value<N>() }", true),
            ("fn f<N:u32={N}>(x:u32)->u32 { x + N }", true),
            ("fn f<N:u32={M}, M:u32>(x:u32)->u32 { x }", false),
            ("fn f<N:u32>(x:uN[M])->u32 { x as u32 }", false),
            ("fn f<N:u32>(x:u32)->u32 { value<{M+N}>(x) }", false),
            (
                "fn f<N:u32>(x:u32)->u32 { later(x) } fn later(x:u32)->u32 { x }",
                false,
            ),
        ] {
            let source = format!("{declarations}\n{function}");
            let bx = parse_module(&source).and_then(|module| check_module(&module));
            let dslx = parse_and_typecheck(&source, "scope.x", "scope", &mut ImportData::default());
            assert_eq!(dslx.is_ok(), valid, "{source}: {:?}", dslx.err());
            assert_eq!(bx.is_ok(), valid, "{source}: {:?}", bx.err());
        }
    }

    #[test]
    fn template_name_enters_scope_after_its_signature() {
        for (source, valid) in [
            ("fn f<N:u32={f<3>()}>()->u32 { N }", false),
            ("fn f<N:u32>(x:uN[f<3>(u32:0)])->u32 { N }", false),
            ("fn f<N:u32>(x:u32)->u32 { f<N>(x) }", true),
        ] {
            let bx = parse_module(source).and_then(|module| check_module(&module));
            let dslx = parse_and_typecheck(source, "scope.x", "scope", &mut ImportData::default());
            assert_eq!(dslx.is_ok(), valid, "{source}: {:?}", dslx.err());
            assert_eq!(bx.is_ok(), valid, "{source}: {:?}", bx.err());
        }
    }

    #[test]
    fn empty_loops_resolve_names_in_parametric_arguments_and_dimensions() {
        let declarations = "fn value<N:u32>(x:u32)->u32 { x + N }";
        for (expression, valid) in [
            ("value<{i+u32:1}>(a)", true),
            ("value<{missing+u32:1}>(a)", false),
            ("a as uN[i+u32:1]", true),
            ("a as uN[missing+u32:1]", false),
        ] {
            let source = format!(
                "{declarations}\nfn f<N:u32>(x:u32)->u32 {{ unroll_for! (i,a) in u32:0..N {{ let _ = {expression}; a }}(x) }}\nfn sample(x:u32)->u32 {{ f<0>(x) }}"
            );
            let bx = parse_module(&source).and_then(|module| check_module(&module));
            let dslx = parse_and_typecheck(&source, "scope.x", "scope", &mut ImportData::default());
            assert_eq!(dslx.is_ok(), valid, "{source}: {:?}", dslx.err());
            assert_eq!(bx.is_ok(), valid, "{source}: {:?}", bx.err());
        }
    }

    #[test]
    fn a_loop_local_value_shadows_a_struct_name_even_in_an_empty_range() {
        for count in [0, 2] {
            let fold = format!(
                "unroll_for! (_, a) in u8:0..u8:{count} {{ let Item = a; let v = Item {{ data: a }}; a }}(x)"
            );
            let decl = "struct Item { data: u8 }";
            let bx = format!("{decl} block f(clk, x: u8) -> u8 {{ {fold} }}");
            let native = format!("{decl} fn f(x: u8) -> u8 {{ {fold} }}");
            assert!(check_module(&parse_module(&bx).unwrap()).is_err(), "{bx}");
            assert!(
                parse_and_typecheck(&native, "scope.x", "scope", &mut ImportData::default())
                    .is_err(),
                "{native}"
            );
        }
    }

    #[test]
    fn empty_loops_resolve_imported_names_without_checking_calls()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        fs::write(
            dir.path().join("helper.x"),
            r#"
pub fn inc(x: u8) -> u8 { x + u8:1 }
fn hidden(x: u8) -> u8 { x }
pub fn sink() {}
pub const VALUE = u8:0;
"#,
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        for (call, accepted) in [
            ("helper::inc()", true),
            ("helper::sink(x)", true),
            ("helper::VALUE(x)", true),
            ("helper::hidden(x)", false),
            ("helper::missing(x)", false),
            ("_(x)", false),
        ] {
            let fold = format!("unroll_for! (_, a) in u8:0..u8:0 {{ let _ = {call}; a }}(x)");
            let bx = format!("use helper; block f(clk, x: u8) -> u8 {{ {fold} }}");
            let native = format!("import helper; fn f(x: u8) -> u8 {{ {fold} }}");
            let data = &mut ImportData::new(None, &[dir.path()]);
            assert_eq!(
                parse_and_typecheck(&native, "probe.x", "probe", data).is_ok(),
                accepted,
                "{native}"
            );
            assert_eq!(
                check_module_with_options(&parse_module(&bx)?, &options).is_ok(),
                accepted,
                "{bx}"
            );
        }
        Ok(())
    }

    #[test]
    fn skipped_native_specializations_check_names_without_evaluating_actuals()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        fs::write(
            dir.path().join("math.x"),
            "pub fn resize<W:u32, N:u32>(x:uN[W])->uN[N] { x as uN[N] }",
        )?;
        let options = ImportOptions {
            additional_search_paths: vec![dir.path()],
            ..Default::default()
        };
        for (expr, valid) in [
            ("math::resize<x, W>()", true),
            ("({ let resize = x; resize<u32:8>(x) })", true),
            ("({ let resize = x; resize<u32:8,u32:8>(x) })", true),
            ("map([x], resize<x, W>)", true),
            ("math::resize<missing, W>(x)", false),
            ("map([x], resize<x, missing>)", false),
            ("math::missing<W>(x)", false),
            ("map([x], math::missing<W>)", false),
        ] {
            for body in [
                format!("pub fn cold<W:u32>(x:u8)->u8 {{ let _ = {expr}; x }}"),
                format!(
                    "const W=u32:8; pub fn cold(x:u8)->u8 {{ unroll_for! (_,a) in u8:0..u8:0 {{ let _ = {expr}; a }}(x) }}"
                ),
            ] {
                let source = format!("use math; use math::resize; {body}");
                fs::write(dir.path().join("wrapper.bx"), &source)?;
                for root in [source.as_str(), "use wrapper;"] {
                    let result = crate::check_source_with_options(root, &options);
                    assert_eq!(
                        result.is_ok(),
                        valid,
                        "{root} / {source}: {:?}",
                        result.err()
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn nested_loops_share_one_expansion_budget() {
        let source = r#"block f(clk, x: u8) -> u8 {
            unroll_for! (_, acc) in u32:0..u32:2 {
                unroll_for! (_, a) in u32:0..u32:1048576 { a }(acc)
            }(x)
        }"#;
        let err = check_module(&parse_module(source).unwrap()).unwrap_err();
        assert_eq!(
            err,
            "block 'f': result: unroll_for! iteration 0: unroll_for! exceeds the block expansion limit of 1048576"
        );
    }
}
