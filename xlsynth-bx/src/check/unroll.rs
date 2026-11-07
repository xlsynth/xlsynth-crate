// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::ast::UnrollFor;
use crate::infer_literal::BindingWidths;
use xlsynth_pir::IrBits;

/// State hidden by a loop-local name. Restore it when leaving the iteration.
pub(super) struct Shadowed {
    name: String,
    value: Option<TypedExpr>,
    widths: Option<BindingWidths>,
    declared: bool,
}

impl BodyChecker<'_, '_> {
    pub(super) fn loop_widths(&self) -> Option<WidthLookup<'_>> {
        (!self.loop_values.is_empty())
            .then(|| WidthLookup::loop_locals(&self.inferred_bindings, &self.loop_values))
    }

    /// Charge all folds in this function or block to one budget, including
    /// nested ones.
    pub(super) fn charge_unroll(&mut self, work: usize) -> Result<(), String> {
        self.unroll_work = self.unroll_work.saturating_add(work);
        if self.unroll_work > MAX_TYPE_SIZE {
            let kind = match self.kind {
                BodyKind::Block { .. } => "block",
                BodyKind::Function => "function",
            };
            return Err(format!(
                "unroll_for! exceeds the {kind} expansion limit of {MAX_TYPE_SIZE}"
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
            Expr::UnrollFor(fold) => self.unroll(fold, expected, structural),
            _ if expected.is_none() => self.inferred_expr(expr),
            _ => self.expr(expr, expected),
        }
    }

    /// Check an ordinary binding or elaborate its child instance.
    pub(super) fn let_value(
        &mut self,
        name: &str,
        expr: &Expr,
        expected: Option<&Type>,
        structural: bool,
    ) -> Result<TypedExpr, String> {
        if let Expr::Call { name: target, args } = expr
            && let Some(child) = self.decls.blocks.get(target.as_str())
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
            let binding = if self.iteration_path.is_empty() {
                name.to_owned()
            } else {
                let mut base = name.to_owned();
                for index in &self.iteration_path {
                    base.push_str(&format!("_{index}"));
                }
                let mut binding = base.clone();
                let mut suffix = 0;
                while !self.reserved_hardware_names.insert(binding.clone()) {
                    suffix += 1;
                    binding = format!("{base}_{suffix}");
                }
                binding
            };
            return self.block_call(&binding, target, child, args, expected);
        }
        self.scoped_result(expr, expected, structural)
    }

    /// Share a loop-local initializer across uses without copying its
    /// expression tree into the next iteration. Constant carries need no
    /// hardware binding.
    fn share_loop_value(&mut self, value: TypedExpr) -> TypedExpr {
        if matches!(value.kind, K::Name(_) | K::Literal(_)) {
            return value;
        }
        let ty = value.ty.clone();
        let constexpr = value.has_constant_inputs(&self.constexpr_names);
        let constant = value.constant(&self.constants);
        if constexpr && let Some(v) = constant {
            return TypedExpr {
                ty,
                kind: K::Literal(v),
            };
        }
        // This name is internal to the checked block and cannot occur in BX.
        let name = format!("$unroll_{}", self.unroll_binding_count);
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
        let values = if self.iteration_path.is_empty() {
            &mut self.function_values
        } else {
            &mut self.loop_values
        };
        saved.push(Shadowed {
            name: name.to_owned(),
            value: values.insert(name.to_owned(), value),
            widths: self.inferred_bindings.remove(name),
            declared: self.names.insert(name.to_owned()),
        });
        if let Some(widths) = widths {
            self.inferred_bindings.insert(name.to_owned(), widths);
        }
    }

    pub(super) fn leave_scope(&mut self, saved: Vec<Shadowed>) {
        let values = if self.iteration_path.is_empty() {
            &mut self.function_values
        } else {
            &mut self.loop_values
        };
        for old in saved.into_iter().rev() {
            if let Some(value) = old.value {
                values.insert(old.name.clone(), value);
            } else {
                values.remove(&old.name);
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

    /// Fold a constant half-open range in source order. Bounds may be wider
    /// than a host integer; only the iteration count needs to fit in memory.
    pub(super) fn unroll(
        &mut self,
        fold: &UnrollFor,
        expected: Option<&Type>,
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
        let acc_hint = annotation.as_ref().map(|(_, a)| a).or(expected);
        let (lo, hi) = self.pair(&fold.start, &fold.limit, index_hint)?;
        let (signed, width) = bits(&lo.ty).ok_or("unroll_for! range requires integer bounds")?;
        let inferred_index = index_hint.is_none()
            && binding_widths(&fold.start, &lo.ty, &self.inferred_bindings)?.is_some()
            && binding_widths(&fold.limit, &hi.ty, &self.inferred_bindings)?.is_some();
        let bound = |v: &TypedExpr| {
            if !v.has_constant_inputs(&self.constexpr_names) {
                return Err("unroll_for! range bounds must be compile-time constants");
            }
            v.constant(&self.constants)
                .and_then(|v| v.to_bits().ok())
                .ok_or("unroll_for! range bounds must be compile-time constants")
        };
        let mut index = bound(&lo)?;
        let limit = bound(&hi)?;
        if if signed {
            limit.slt(&index)
        } else {
            limit.ult(&index)
        } {
            return Err("unroll_for! range limit precedes its start".into());
        }
        let count = ir_bits_to_usize(&limit.sub(&index)).ok_or("unroll_for! range too large")?;
        // Charge nested loops to the same block. A small inner range repeated
        // many times consumes the same budget as a single large range.
        self.charge_unroll(
            count.saturating_mul(fold.body.len().saturating_add(width.div_ceil(64).max(1))),
        )?;
        if acc_hint.is_none() && minimum_type(&fold.seed)?.is_some() {
            return Err(
                "unroll_for! needs a typed seed or accumulator for an implicit literal result"
                    .into(),
            );
        }
        let seed = self.scoped_result(&fold.seed, acc_hint, false)?;
        if let Some(expected) = expected
            && !compatible(expected, &seed.ty)
        {
            return Err(format!("expected {expected}, got {}", seed.ty));
        }
        let mut carry = self.share_loop_value(seed);
        if count == 0 {
            self.check_empty_loop_names(fold)?;
            return Ok(carry);
        }
        let one = IrBits::from_lsb_fn(width, |i| i == 0);
        for ordinal in 0..count {
            self.iteration_path.push(ordinal);
            let mut saved = Vec::new();
            let item = TypedExpr {
                ty: lo.ty.clone(),
                kind: K::Literal(IrValue::Bits(index.clone())),
            };
            let acc_type = carry.ty.clone();
            let index_widths = inferred_index.then(|| BindingWidths::scalar(signed, width));
            self.bind_scoped_value(&fold.index, item, index_widths, &mut saved);
            self.bind_scoped_value(&fold.accumulator, carry, None, &mut saved);
            for stmt in &fold.body {
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
                    .let_value(name, expr, ty.as_ref(), structural)
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
        Ok(carry)
    }
}

impl BodyChecker<'_, '_> {
    /// DSLX resolves lexical names before unrolling, even when it never
    /// typechecks an iteration. Keep that distinction for an empty range.
    fn check_empty_loop_names(&self, fold: &UnrollFor) -> Result<(), String> {
        let mut locals = self.names.clone();
        self.check_body_names(fold, &mut locals)
    }

    fn check_body_names(
        &self,
        fold: &UnrollFor,
        locals: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        if fold.index == fold.accumulator && fold.index != "_" {
            return Err(format!("duplicate loop binding '{}'", fold.index));
        }
        locals.extend([fold.index.clone(), fold.accumulator.clone()]);
        for stmt in &fold.body {
            let Stmt::Let { name, value, ty } = stmt else {
                return Err("unroll_for! bodies permit only let bindings before the result".into());
            };
            if let Some(ty) = ty {
                self.check_type_names(ty, locals)?;
            }
            self.check_expr_names(value, locals)?;
            locals.insert(name.clone());
        }
        self.check_expr_names(&fold.result, locals)
    }

    fn check_type_names(&self, ty: &Type, locals: &BTreeSet<String>) -> Result<(), String> {
        match ty {
            Type::Named(name) if locals.contains(name) => {
                return Err(format!("'{name}' names a local value here"));
            }
            Type::Named(name) => {
                self.decls
                    .check_type_visibility(ty, self.function_position)?;
                if let Some((module, member)) = lookup_use(name, self.resolver, Some(locals))? {
                    self.resolver.check_member_name(&module, &member)?;
                } else {
                    self.structs.definition(name)?;
                }
            }
            Type::Tuple(items) => {
                for item in items {
                    self.check_type_names(item, locals)?;
                }
            }
            Type::Array { element, .. } => self.check_type_names(element, locals)?,
            _ => { /* Built-in type name. */ }
        }
        Ok(())
    }

    /// An unused call still needs a visible callee, but DSLX checks its kind
    /// and arity only when it elaborates an iteration.
    fn check_value_name(&self, name: &str, locals: &BTreeSet<String>) -> Result<(), String> {
        if name == "_" {
            return Err("cannot read the wildcard binding '_'".into());
        }
        if locals.contains(name) || name == "update" {
            return Ok(());
        }
        if !self.decls.visible(name, self.function_position) {
            return Err(format!("'{name}' is declared later in the module"));
        }
        if self.decls.functions.contains_key(name) || self.decls.blocks.contains_key(name) {
            return Ok(());
        }
        if let Some((module, member)) = lookup_use(name, self.resolver, Some(locals))? {
            return self.resolver.check_member_name(&module, &member);
        }
        Err(format!("unknown value '{name}'"))
    }

    fn check_expr_names(&self, expr: &Expr, locals: &BTreeSet<String>) -> Result<(), String> {
        match expr {
            Expr::Scope { bindings, result } => {
                let mut names = locals.clone();
                for binding in bindings {
                    self.validate_source_pattern(&binding.pattern)?;
                    if let Some(ty) = &binding.ty {
                        self.check_type_names(ty, &names)?;
                    }
                    self.check_expr_names(&binding.value, &names)?;
                    binding.pattern.visit_names(&mut |n| {
                        names.insert(n.to_owned());
                    });
                }
                self.check_expr_names(result, &names)?;
            }
            Expr::Name(name) => self.check_value_name(name, locals)?,
            Expr::Bool(_) | Expr::Literal { ty: None, .. } => { /* No names to resolve. */ }
            Expr::Zero(ty) | Expr::Literal { ty: Some(ty), .. } => {
                self.check_type_names(ty, locals)?
            }
            Expr::Tuple(items) | Expr::Array(items) | Expr::Call { args: items, .. } => {
                if let Expr::Call { name, .. } = expr {
                    if self.kind == BodyKind::Function
                        && !locals.contains(name)
                        && self.decls.blocks.contains_key(name.as_str())
                    {
                        return Err(format!("pure function cannot call block '{name}'"));
                    }
                    self.check_value_name(name, locals)?;
                }
                for item in items {
                    self.check_expr_names(item, locals)?;
                }
            }
            Expr::Struct { name, fields } => {
                self.check_type_names(&Type::Named(name.clone()), locals)?;
                for (_, value) in fields {
                    self.check_expr_names(value, locals)?;
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
                for operand in super::match_operands(subject, arms)? {
                    self.check_expr_names(operand, locals)?;
                }
                for arm in arms {
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
                self.check_body_names(fold, &mut locals.clone())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{ImportOptions, check_module, check_module_with_options, parse_module};
    use std::fs;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};

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
