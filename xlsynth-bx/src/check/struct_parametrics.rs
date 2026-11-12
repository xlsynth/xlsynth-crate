// SPDX-License-Identifier: Apache-2.0

//! Check parametric BX structs and intern their concrete layouts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write};

use xlsynth_pir::ir_value_utils::ir_bits_to_usize;
use xlsynth_pir::{IrFormatPreference, IrValue};

use crate::ast::{Expr, Param, StructDef, StructLiteralArgs, Type};
use crate::builtin::BuiltinOp;
use crate::typed::{BoundParametricValue, TypedExpr, bits};

use super::counted_for::CarryContext;
use super::generic_blocks::Bindings;
use super::generic_inputs::Input;
use super::value_parametrics::{ExplicitScope, InferredBindings, dimension_type};
use super::{
    BodyChecker, BodyKind, FunctionSpecializations, ImportResolver, MAX_ELABORATION_DEPTH,
    MAX_SPECIALIZATIONS, MAX_TYPE_SIZE, ModuleDimensions, StructTable, child_type_budget, declare,
};

/// The source identity and bindings of an interned nominal type.
pub(super) struct StructInstance<'m> {
    pub(super) template: &'m StructDef,
    pub(super) values: Vec<BoundParametricValue>,
    pub(super) types: Vec<Type>,
}

/// Bind a native struct's compile-time parameters in declaration order.
pub(super) enum StructArguments<'a> {
    Values {
        explicit: &'a [BoundParametricValue],
        inferred: Option<&'a InferredBindings<'a>>,
    },
    Types(&'a [Type]),
}

/// A constructor's evaluated prefix, before its fields or base supply the
/// remaining arguments.
#[derive(Clone, Copy)]
pub(super) enum StructApplication<'a> {
    Named(&'a str),
    Values(&'a str, &'a [BoundParametricValue]),
    Types(&'a str, &'a [Type]),
}

impl fmt::Display for StructApplication<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(name) => f.write_str(name),
            Self::Types(name, args) => write!(
                f,
                "{}",
                Type::TypeApplied {
                    name: (*name).to_owned(),
                    args: args.to_vec(),
                }
            ),
            Self::Values(name, args) => {
                let mut bindings = StructName::default();
                for (index, arg) in args.iter().enumerate() {
                    if (index != 0 && bindings.append(format_args!(", ")).is_err())
                        || bindings.append_binding(arg).is_err()
                    {
                        // A diagnostic can describe an application whose name
                        // exceeds the interning limit. Only output errors may
                        // escape Display.
                        return write!(f, "{name}<…>");
                    }
                }
                write!(f, "{name}<{}>", bindings.0)
            }
        }
    }
}

/// A constructor can need type evidence from its surrounding expression.
pub(super) enum StructLiteralError {
    Unbound(String),
    Invalid(String),
}

impl From<String> for StructLiteralError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

impl StructLiteralError {
    pub(super) fn into_message(self) -> String {
        match self {
            Self::Unbound(message) | Self::Invalid(message) => message,
        }
    }
}

/// Whether a bare type-parametric struct can supply a complete type.
pub(super) enum StructWitness {
    Other,
    Deferred,
    Checked(TypedExpr),
}

/// A constructor whose specialization can be supplied by a sibling.
pub(super) enum ContextualStruct<'m> {
    Native(&'m StructDef),
    Imported,
}

/// Bound a specialization name while formatting nested nominal arguments.
#[derive(Default)]
struct StructName(String);

impl Write for StructName {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.0.len().saturating_add(text.len()) > MAX_TYPE_SIZE {
            return Err(fmt::Error);
        }
        self.0.push_str(text);
        Ok(())
    }
}

impl StructName {
    /// Name a partial value application when defaults determine its entire
    /// suffix.
    fn default_prefix(
        template: &StructDef,
        args: &StructArguments<'_>,
    ) -> Result<Option<String>, String> {
        let StructArguments::Values { explicit, .. } = args else {
            return Ok(None);
        };
        let suffix = &template.value_parametrics[explicit.len()..];
        if suffix.is_empty() || suffix.iter().any(|p| p.default.is_none()) {
            return Ok(None);
        }
        let mut prefix = Self::default();
        prefix.append(format_args!("{}<", template.name))?;
        for (index, bound) in explicit.iter().enumerate() {
            if index > 0 {
                prefix.append(format_args!(", "))?;
            }
            prefix.append_binding(bound)?;
        }
        prefix.append(format_args!(">"))?;
        Ok(Some(prefix.0))
    }

    fn append(&mut self, args: fmt::Arguments<'_>) -> Result<(), String> {
        self.write_fmt(args)
            .map_err(|_| format!("struct specialization name exceeds {MAX_TYPE_SIZE} bytes"))
    }

    /// Keep legacy u32 names and include the nominal type of other values.
    fn append_binding(&mut self, bound: &BoundParametricValue) -> Result<(), String> {
        if bound.ty == Type::Bool {
            return self.append(format_args!(
                "{}",
                bound.value.to_bool().map_err(|e| e.to_string())?
            ));
        }
        if bound.ty != dimension_type() {
            self.append(format_args!("{}:", bound.ty))?;
        }
        self.append_value(&bound.value)
    }

    /// Stream aggregate elements so the specialization-name limit applies
    /// before allocating the entire text of a potentially large array.
    fn append_value(&mut self, value: &IrValue) -> Result<(), String> {
        let (open, elements, close) = match value {
            IrValue::Bits(bits) => {
                return self.append(format_args!(
                    "{}",
                    bits.to_string_fmt(IrFormatPreference::Default, false)
                ));
            }
            IrValue::Tuple(elements) => ("(", elements.as_ref(), ")"),
            IrValue::Array(array) => ("[", array.elements(), "]"),
            IrValue::Token => return Err("a struct value parametric cannot be a token".into()),
        };
        self.append(format_args!("{open}"))?;
        for (index, element) in elements.iter().enumerate() {
            if index > 0 {
                self.append(format_args!(", "))?;
            }
            self.append_value(element)?;
        }
        self.append(format_args!("{close}"))
    }
}

impl<'m> StructTable<'m> {
    /// Record a template before checking the concrete declarations that use it.
    pub(super) fn register_template(&mut self, def: &'m StructDef) -> Result<(), String> {
        if !def.is_parametric() {
            return Ok(());
        }
        if !def.type_params.is_empty() && !def.value_parametrics.is_empty() {
            return Err(format!(
                "struct '{}' mixes type and value parameters",
                def.name
            ));
        }
        let mut names = BTreeSet::new();
        for name in &def.type_params {
            if !crate::parser::is_identifier(name) {
                return Err(format!("invalid type parameter name '{name}'"));
            }
            declare(&mut names, name).map_err(|e| format!("struct '{}': {e}", def.name))?;
        }
        for (index, parametric) in def.value_parametrics.iter().enumerate() {
            declare(&mut names, &parametric.name)
                .map_err(|e| format!("struct '{}': {e}", def.name))?;
            if parametric.index != index {
                return Err(format!(
                    "struct '{}' value parametric '{}' has index {}, expected {index}",
                    def.name, parametric.name, parametric.index
                ));
            }
        }
        self.templates.insert(&def.name, def);
        Ok(())
    }

    pub(super) fn template(&self, name: &str) -> Option<&'m StructDef> {
        self.templates.get(name).copied()
    }
}

/// Convert a checked u32 parametric to the canonical binding representation.
fn binding_value(value: &IrValue) -> Result<u32, String> {
    let bits = value.as_bits().map_err(|e| e.to_string())?;
    ir_bits_to_usize(bits)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| "struct value parametric does not fit in u32".into())
}

impl<'m> BodyChecker<'_, 'm> {
    /// Find the value substituted for a bare formal in a struct declaration.
    pub(super) fn direct_struct_binding(&self, expr: &Expr) -> Option<(&Type, &IrValue)> {
        let Expr::Name(name) = expr else { return None };
        if !self.in_struct_definition || !self.value_parametric_names.contains(name) {
            return None;
        }
        let TypedExpr {
            ty,
            kind: super::K::Literal(value),
        } = self.function_values.get(name)?
        else {
            return None;
        };
        Some((ty, value))
    }

    /// Check a directly forwarded scalar against the receiving struct
    /// formal's numeric range, independent of the value's printed radix.
    fn struct_argument_value(&mut self, expr: &Expr, target: &Type) -> Result<IrValue, String> {
        if let Some((source, value)) = self.direct_struct_binding(expr)
            && source != target
            && let Some((source_signed, _)) = bits(source)
            && let Some((target_signed, width)) = bits(target)
        {
            return BuiltinOp::CheckedCast {
                source_signed,
                target_signed,
                width,
            }
            .evaluate(std::slice::from_ref(value))
            .ok_or_else(|| format!("forwarded {source} value does not fit {target}"));
        }
        self.parametric_value(expr, target)
    }

    /// Evaluate the written value prefix using local formal types or the
    /// imported u32 ABI.
    pub(super) fn struct_value_prefix(
        &mut self,
        name: &str,
        args: StructLiteralArgs<'_>,
    ) -> Result<Vec<BoundParametricValue>, String> {
        match (args, self.structs.template(name)) {
            (StructLiteralArgs::Omitted, _) => Ok(Vec::new()),
            (StructLiteralArgs::Types(_), _) => Err(format!(
                "struct '{name}' has type arguments where value arguments are required"
            )),
            (StructLiteralArgs::Expressions(args), Some(template)) => {
                self.explicit_struct_values(template, args.len(), |caller, index, ty| {
                    caller.struct_argument_value(&args[index], ty)
                })
            }
            (StructLiteralArgs::Integers(args), Some(template)) => {
                self.explicit_struct_values(template, args.len(), |caller, index, ty| {
                    // The compact AST stores these as unsized literals.
                    caller.parametric_value(
                        &Expr::Literal {
                            ty: None,
                            text: args[index].to_string(),
                        },
                        ty,
                    )
                })
            }
            (StructLiteralArgs::Expressions(args), None) => {
                let ty = dimension_type();
                args.iter()
                    .map(|arg| {
                        Ok(BoundParametricValue {
                            value: self.struct_argument_value(arg, &ty)?,
                            ty: ty.clone(),
                        })
                    })
                    .collect()
            }
            (StructLiteralArgs::Integers(args), None) => Ok(args
                .iter()
                .map(|&value| BoundParametricValue {
                    ty: dimension_type(),
                    value: IrValue::u32(value),
                })
                .collect()),
        }
    }

    /// Resolve an explicit value application after evaluating its prefix.
    pub(super) fn struct_value_type(
        &mut self,
        name: &str,
        args: &[BoundParametricValue],
        remaining: usize,
    ) -> Result<Type, String> {
        self.check_constructor_name(name)?;
        self.reject_type_parameter_value(name)?;
        if let Some(template) = self.structs.template(name) {
            self.decls
                .check_type_name_visibility(name, self.function_position.or(self.block_position))?;
            self.specialize_struct(
                template,
                StructArguments::Values {
                    explicit: args,
                    inferred: None,
                },
                remaining,
            )
        } else {
            let args = args
                .iter()
                .map(|arg| binding_value(&arg.value))
                .collect::<Result<Vec<_>, _>>()?;
            let ty = self.resolve_dimensions(
                &Type::Applied {
                    name: name.to_owned(),
                    args,
                },
                remaining,
            )?;
            // A constructor can introduce a DSLX specialization with no prior
            // type annotation. Register its reflected layout before checking
            // fields.
            self.structs.update_imports(self.resolver)?;
            Ok(ty)
        }
    }

    /// Resolve a fully determined constructor in the surrounding type scope.
    fn struct_application_type(&mut self, source: StructApplication<'_>) -> Result<Type, String> {
        match source {
            StructApplication::Named(name) => self.source_type(&Type::Named(name.to_owned())),
            StructApplication::Types(name, args) => self.source_type(&Type::TypeApplied {
                name: name.to_owned(),
                args: args.to_vec(),
            }),
            StructApplication::Values(name, args) => {
                self.struct_value_type(name, args, MAX_ELABORATION_DEPTH)
            }
        }
    }

    /// Check the written arity before asking whether a splat's base needs
    /// context. A complete application supplies its own nominal type.
    pub(super) fn contextual_struct_constructor(
        &self,
        expr: &Expr,
    ) -> Option<ContextualStruct<'m>> {
        let literal = expr.struct_literal()?;
        let name = literal.name;
        let supplied = match literal.args {
            StructLiteralArgs::Omitted => 0,
            StructLiteralArgs::Integers(args) => args.len(),
            StructLiteralArgs::Expressions(args) => args.len(),
            StructLiteralArgs::Types(_) => return None,
        };
        let template = if let Some(def) = self.structs.template(name) {
            let bare_type_struct = matches!(literal.args, StructLiteralArgs::Omitted)
                && !def.type_params.is_empty()
                && !self.type_bindings().is_some_and(|b| b.declares(name));
            (bare_type_struct || supplied < def.value_parametrics.len())
                .then_some(ContextualStruct::Native(def))?
        } else {
            let (subject, member) = self.resolver.lookup(name).ok().flatten()?;
            let arity = self.resolver.struct_template_arity(&subject, &member)?;
            (supplied < arity).then_some(ContextualStruct::Imported)?
        };
        // An update on an already concrete base supplies that base's nominal
        // type. An inferred base can receive the missing arguments from a peer.
        literal
            .base
            .is_none_or(|splat| self.contains_inferred_aggregate(&splat.value))
            .then_some(template)
    }

    /// Native declarations additionally support inference from their fields.
    fn inferred_struct_constructor(&self, expr: &Expr) -> Option<&'m StructDef> {
        match self.contextual_struct_constructor(expr)? {
            ContextualStruct::Native(def) => Some(def),
            ContextualStruct::Imported => None,
        }
    }

    /// Find constructors whose type can depend on an enclosing result group.
    pub(super) fn contains_type_struct_constructor(&self, expr: &Expr) -> bool {
        match expr {
            Expr::For(fold) if fold.annotation.is_none() => {
                self.contains_type_struct_constructor(&fold.seed)
            }
            Expr::UnrollFor(fold) if fold.annotation.is_none() => {
                self.contains_type_struct_constructor(&fold.seed)
            }
            Expr::Tuple(items)
            | Expr::Array(items)
            | Expr::ArrayLiteral {
                annotation: None,
                members: items,
                ..
            } => items
                .iter()
                .any(|item| self.contains_type_struct_constructor(item)),
            Expr::If {
                then_value,
                else_value,
                ..
            } => {
                self.contains_type_struct_constructor(then_value)
                    || self.contains_type_struct_constructor(else_value)
            }
            Expr::Match { arms, .. } => arms
                .iter()
                .any(|arm| self.contains_type_struct_constructor(&arm.value)),
            Expr::Scope { result, .. } => self.contains_type_struct_constructor(result),
            _ => self
                .inferred_struct_constructor(expr)
                .is_some_and(|template| !template.type_params.is_empty()),
        }
    }

    /// Use a complete type-parametric constructor to check its siblings.
    /// Value-parametric constructors merge explicit arguments and use the final
    /// constructor's fields to infer the remaining bindings, as DSLX does.
    pub(super) fn infer_struct_siblings(
        &mut self,
        items: &[&Expr],
        prefix: Option<(&str, &[BoundParametricValue])>,
    ) -> Result<Option<Vec<TypedExpr>>, String> {
        let Some((last, siblings)) = items.split_last().filter(|(_, rest)| !rest.is_empty()) else {
            return Ok(None);
        };
        let Some(template) = self.inferred_struct_constructor(last) else {
            return Ok(None);
        };
        let name = template.name.as_str();
        if !siblings.iter().all(|item| {
            self.inferred_struct_constructor(item)
                .is_some_and(|def| def.name == name)
        }) {
            return Ok(None);
        }
        if !template.type_params.is_empty() {
            for (index, item) in items.iter().enumerate() {
                if let StructWitness::Checked(anchor) = self.type_struct_witness(item)? {
                    return self.check_siblings(items, index, anchor).map(Some);
                }
            }
            return Ok(None);
        }
        let mut explicit = prefix
            .filter(|(outer, _)| *outer == name)
            .map_or_else(Vec::new, |(_, args)| args.to_vec());
        let mut members = Vec::with_capacity(items.len());
        for item in items {
            let literal = item
                .struct_literal()
                .expect("sibling is a native struct constructor");
            let args = self.struct_value_prefix(name, literal.args)?;
            if let Some(index) = explicit.iter().zip(&args).position(|(a, b)| a != b) {
                return Err(format!(
                    "conflicting values for argument {} of struct '{name}'",
                    index + 1
                ));
            }
            if args.len() > explicit.len() {
                explicit = args;
            }
            members.push((literal.fields, literal.base));
        }
        let source = if explicit.is_empty() {
            StructApplication::Named(name)
        } else {
            StructApplication::Values(name, &explicit)
        };
        let (fields, base) = members.pop().expect("final constructor fields");
        let (ty, kind) = self.struct_literal(source, fields, base, None)?;
        let anchor = self.finish_expr(last, None, ty, kind)?;
        let mut values = Vec::with_capacity(items.len());
        for (item, (fields, base)) in siblings.iter().zip(members) {
            let expected = Some(&anchor.ty);
            let (ty, kind) = self.struct_literal(source, fields, base, expected)?;
            values.push(self.finish_expr(item, expected, ty, kind)?);
        }
        values.push(anchor);
        Ok(Some(values))
    }

    /// Check a bare type-parametric constructor without requiring it to supply
    /// every argument before the surrounding expression is inferred.
    pub(super) fn type_struct_witness(&mut self, expr: &Expr) -> Result<StructWitness, String> {
        match expr {
            Expr::For(fold)
                if fold.annotation.is_none()
                    && self.contains_type_struct_constructor(&fold.seed) =>
            {
                return self.with_dslx_value_context(|checker| {
                    let Some(seed) = checker.constructor_dimension_witness(&fold.seed)? else {
                        return Ok(StructWitness::Deferred);
                    };
                    checker
                        .counted_for(fold, CarryContext::Prepared(seed))
                        .map(StructWitness::Checked)
                });
            }
            Expr::UnrollFor(fold)
                if fold.annotation.is_none()
                    && self.contains_type_struct_constructor(&fold.seed) =>
            {
                let Some(seed) = self.constructor_dimension_witness(&fold.seed)? else {
                    return Ok(StructWitness::Deferred);
                };
                return self
                    .unroll(fold, CarryContext::Prepared(seed), false)
                    .map(StructWitness::Checked);
            }
            _ => { /* Only an unannotated loop inherits its seed's nominal type. */ }
        }
        let Expr::Struct { name, fields, base } = expr else {
            return Ok(StructWitness::Other);
        };
        if self
            .inferred_struct_constructor(expr)
            .is_none_or(|template| template.type_params.is_empty())
        {
            return Ok(StructWitness::Other);
        }
        let first = self.statements.len();
        let base = if let Some(splat) = base {
            let Some(base) = self.constructor_dimension_witness(&splat.value)? else {
                self.discard_temporary_statements(first);
                return Ok(StructWitness::Deferred);
            };
            Some((splat.as_ref(), base))
        } else {
            None
        };
        match self.check_struct_literal(StructApplication::Named(name), fields, base, None) {
            Ok((ty, kind)) => self
                .finish_expr(expr, None, ty, kind)
                .map(StructWitness::Checked),
            Err(StructLiteralError::Unbound(_)) => {
                self.discard_temporary_statements(first);
                Ok(StructWitness::Deferred)
            }
            Err(error) => Err(error.into_message()),
        }
    }

    /// Pass the update's result type or explicit bindings to its base
    /// expression.
    pub(super) fn struct_splat_base(
        &mut self,
        source: StructApplication<'_>,
        base: &Expr,
        expected: Option<&Type>,
    ) -> Result<TypedExpr, String> {
        if let Some(ty) = expected {
            return self.expr(base, Some(ty));
        }
        match source {
            StructApplication::Values(name, args)
                if self
                    .structs
                    .template(name)
                    .is_none_or(|template| args.len() < template.value_parametrics.len()) =>
            {
                self.splat_base_with_prefix(base, name, args)
            }
            StructApplication::Named(name)
                if !self
                    .type_bindings()
                    .is_some_and(|bindings| bindings.declares(name))
                    && (self.structs.template(name).is_some()
                        || self.resolver.lookup(name)?.is_some_and(|(module, member)| {
                            self.resolver
                                .check_struct_template_name(&module, &member)
                                .is_ok()
                        })) =>
            {
                self.expr(base, None)
            }
            _ => {
                let ty = self.struct_application_type(source)?;
                self.expr(base, Some(&ty))
            }
        }
    }

    /// Merge the outer prefix with a base constructor's own arguments. Its
    /// fields or its splat can still determine the remaining arguments.
    pub(super) fn splat_base_with_prefix(
        &mut self,
        value: &Expr,
        name: &str,
        prefix: &[BoundParametricValue],
    ) -> Result<TypedExpr, String> {
        let literal = match value {
            Expr::For(fold) => {
                return self.with_dslx_value_context(|checker| {
                    checker.counted_for(fold, CarryContext::StructPrefix { name, args: prefix })
                });
            }
            Expr::UnrollFor(fold) => {
                return self.unroll(
                    fold,
                    CarryContext::StructPrefix { name, args: prefix },
                    false,
                );
            }
            Expr::Scope { .. } => {
                return self.with_prepared_value(value, |checker, result| {
                    checker.splat_base_with_prefix(result, name, prefix)
                });
            }
            Expr::If { .. } | Expr::Match { .. } => {
                let mut results = self.splat_branches_with_prefix(&[value], name, prefix)?;
                let result = results.pop().expect("one checked splat base");
                return self.finish_expr(value, None, result.ty, result.kind);
            }
            _ => match value.struct_literal() {
                Some(literal)
                    if !matches!(literal.args, StructLiteralArgs::Types(_))
                        && self.same_struct_template(name, literal.name)? =>
                {
                    literal
                }
                _ => return self.expr(value, None),
            },
        };
        let args = self.struct_value_prefix(literal.name, literal.args)?;
        let args = if prefix.starts_with(&args) {
            prefix.to_vec()
        } else if args.starts_with(prefix) {
            args
        } else {
            return Err(format!(
                "conflicting values for arguments of struct '{name}'"
            ));
        };
        let source = StructApplication::Values(literal.name, &args);
        let (ty, kind) = self.struct_literal(source, literal.fields, literal.base, None)?;
        self.finish_expr(value, None, ty, kind)
    }

    /// Share the prefix and inferred specialization across every branch of
    /// the base, including results inside nested conditionals and matches.
    fn splat_branches_with_prefix(
        &mut self,
        items: &[&Expr],
        name: &str,
        prefix: &[BoundParametricValue],
    ) -> Result<Vec<TypedExpr>, String> {
        if items.iter().any(|item| matches!(item, Expr::Scope { .. })) {
            return self.with_scoped_results(items, |checker, items| {
                checker.splat_branches_with_prefix(items, name, prefix)
            });
        }
        if items
            .iter()
            .any(|item| matches!(item, Expr::If { .. } | Expr::Match { .. }))
        {
            return self.check_result_branches(items, |checker, items| {
                checker.splat_branches_with_prefix(items, name, prefix)
            });
        }
        if let Some(values) = self.infer_struct_siblings(items, Some((name, prefix)))? {
            return Ok(values);
        }
        let index = match self.common_hint(items)? {
            super::CommonTypeHint::FromExpr(index) => index,
            super::CommonTypeHint::Minimum(_) => items.len() - 1,
        };
        let anchor = self.splat_base_with_prefix(items[index], name, prefix)?;
        self.check_siblings(items, index, anchor)
    }

    /// Qualified and directly imported spellings can name the same DSLX
    /// template.
    fn same_struct_template(&self, outer: &str, inner: &str) -> Result<bool, String> {
        if outer == inner {
            return Ok(true);
        }
        let Some(template) = self.resolver.lookup(outer)? else {
            return Ok(false);
        };
        Ok(self.resolver.lookup(inner)?.as_ref() == Some(&template))
    }

    /// Match a constructor to its base, filling omitted bindings from the
    /// base's concrete specialization.
    pub(super) fn check_struct_splat_type(
        &mut self,
        source: StructApplication<'_>,
        base: &Type,
    ) -> Result<(), String> {
        let Type::Named(canonical) = base else {
            return Err(format!("struct splat base must be a struct, got {base}"));
        };
        let prefix = match source {
            StructApplication::Named(name) => Some((name, &[][..])),
            StructApplication::Values(name, args) => Some((name, args)),
            _ => None,
        };
        let matched = match prefix {
            Some((name, prefix)) => self.struct_instance_matches(name, prefix, canonical)?,
            None => None,
        };
        let matches = match matched {
            Some(matches) => matches,
            None => self.struct_application_type(source)? == *base,
        };
        if !matches {
            return Err(format!(
                "struct splat base type {base} does not match '{source}'"
            ));
        }
        Ok(())
    }

    /// Check the written argument prefix against an existing native or
    /// imported specialization of the same template.
    fn struct_instance_matches(
        &self,
        name: &str,
        prefix: &[BoundParametricValue],
        canonical: &str,
    ) -> Result<Option<bool>, String> {
        if self
            .type_bindings()
            .is_some_and(|bindings| bindings.declares(name))
        {
            return Ok(None);
        }
        let matches = if let Some(instance) = self.structs.instances.get(canonical)
            && instance.template.name == name
        {
            instance.values.starts_with(prefix)
        } else if let Some(values) = self.resolver.struct_parametric_values(name, canonical)? {
            prefix.len() <= values.len()
                && prefix.iter().zip(values).all(|(bound, &value)| {
                    bound.ty == dimension_type() && bound.value == IrValue::u32(value)
                })
        } else {
            return Ok(None);
        };
        self.check_constructor_name(name)?;
        Ok(Some(matches))
    }

    /// Resolve the constructor name in the current declaration and lexical
    /// scope.
    fn check_constructor_name(&self, name: &str) -> Result<(), String> {
        self.decls
            .check_type_name_visibility(name, self.function_position)?;
        let binding = name.split_once("::").map_or(name, |(head, _)| head);
        if self.has_local_name(binding) || self.callback_index(binding).is_some() {
            return Err(format!("'{binding}' names a local value here"));
        }
        Ok(())
    }

    /// Resolve the nominal type of a constructor, using context and field
    /// types.
    pub(super) fn struct_literal_type<'e>(
        &mut self,
        source: StructApplication<'_>,
        supplied: &BTreeMap<&str, &'e Expr>,
        expected: Option<&Type>,
    ) -> Result<(Type, BTreeMap<String, Input<'e>>), StructLiteralError> {
        let (name, explicit) = match source {
            StructApplication::Named(name) => (name, &[][..]),
            StructApplication::Values(name, args) => (name, args),
            StructApplication::Types(..) => {
                return Ok((self.struct_application_type(source)?, BTreeMap::new()));
            }
        };
        let Some(template) = self.structs.template(name).filter(|_| {
            !self
                .type_bindings()
                .is_some_and(|bindings| bindings.declares(name))
        }) else {
            if let Some(ty @ Type::Named(canonical)) = expected
                && self.struct_instance_matches(name, explicit, canonical)? == Some(true)
            {
                return Ok((ty.clone(), BTreeMap::new()));
            }
            return Ok((self.struct_application_type(source)?, BTreeMap::new()));
        };
        self.check_constructor_name(name)?;
        let context = match expected {
            Some(Type::Named(name)) => self.structs.instances.get(name),
            _ => None,
        };
        if !template.type_params.is_empty()
            && let Some(instance) = context
            && instance.template.name == template.name
        {
            return Ok((
                expected.expect("matching struct context").clone(),
                BTreeMap::new(),
            ));
        }
        let contextual = context
            .filter(|instance| {
                instance.template.name == template.name && instance.values.starts_with(explicit)
            })
            .map(|instance| instance.values.clone());
        let explicit = contextual.as_deref().unwrap_or(explicit);
        let mut inferred =
            InferredBindings::for_struct(&template.value_parametrics, explicit.len());
        let mut inputs = Vec::with_capacity(template.fields.len());
        let mut types = Bindings::new(&template.type_params);
        for field in &template.fields {
            let value = supplied
                .get(field.name.as_str())
                .ok_or_else(|| format!("missing field '{source}.{}'", field.name))?;
            inferred.literal_lengths(&field.ty, value)?;
            let input = self.infer_type_input(
                &field.ty,
                value,
                &mut types,
                &mut inferred,
                /* tuple_member= */ true,
            )?;
            inputs.push(input);
            inferred.finish_argument();
        }
        let type_args;
        let args = if template.type_params.is_empty() {
            StructArguments::Values {
                explicit,
                inferred: Some(&inferred),
            }
        } else {
            self.infer_input_groups(
                &mut inputs,
                &template.fields,
                &mut types,
                &format!("struct '{name}'"),
            )?;
            type_args = template
                .type_params
                .iter()
                .map(|name| {
                    types.get(name).cloned().ok_or_else(|| {
                        StructLiteralError::Unbound(format!(
                            "cannot infer type parameter '{name}' of struct '{}'",
                            template.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>, StructLiteralError>>()?;
            StructArguments::Types(&type_args)
        };
        let ty = self.specialize_struct(template, args, MAX_ELABORATION_DEPTH)?;
        let inputs = template
            .fields
            .iter()
            .zip(inputs)
            .filter(|(_, input)| !matches!(input, Input::Contextual { .. }))
            .map(|(field, input)| (field.name.clone(), input))
            .collect();
        Ok((ty, inputs))
    }

    /// Resolve each explicit formal's type in declaration scope, then
    /// evaluate its actual in caller scope. Carry preceding bindings between
    /// formal checks so dependent types see precisely their declaration prefix.
    pub(super) fn explicit_struct_values(
        &mut self,
        template: &'m StructDef,
        count: usize,
        mut evaluate: impl FnMut(&mut Self, usize, &Type) -> Result<IrValue, String>,
    ) -> Result<Vec<BoundParametricValue>, String> {
        check_value_arity(template, count)?;
        let mut scope = ExplicitScope::default();
        let mut values = Vec::with_capacity(count);
        for (index, formal) in template.value_parametrics.iter().take(count).enumerate() {
            let ty = self.in_struct_definition(template, |definition| {
                scope.exchange(definition);
                if let Some(previous) = values.last() {
                    definition.bind_parametric_value(
                        &template.value_parametrics[index - 1].name,
                        previous,
                    );
                }
                let result = definition.source_type(&formal.ty);
                scope.exchange(definition);
                result
            })?;
            let value = evaluate(self, index, &ty).map_err(|e| {
                format!(
                    "struct '{}' parametric '{}': {e}",
                    template.name, formal.name
                )
            })?;
            values.push(BoundParametricValue { ty, value });
        }
        Ok(values)
    }

    /// Bind defaults and resolve fields in the template's declaration scope.
    pub(super) fn specialize_struct(
        &mut self,
        template: &'m StructDef,
        args: StructArguments<'_>,
        remaining: usize,
    ) -> Result<Type, String> {
        match &args {
            StructArguments::Values { explicit, .. } => {
                check_value_arity(template, explicit.len())?
            }
            StructArguments::Types(args) => check_type_arity(template, args.len())?,
        }
        let field_budget = child_type_budget(remaining)?;
        self.in_struct_definition(template, |checker| {
            checker.intern_struct(template, args, field_budget)
        })
    }

    /// Check types, defaults, or fields under the template's own module
    /// bindings. Recursive references through any of these share one guard.
    fn in_struct_definition<T>(
        &mut self,
        template: &'m StructDef,
        evaluate: impl FnOnce(&mut BodyChecker<'_, 'm>) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.structs.active_templates.len() >= MAX_ELABORATION_DEPTH {
            return Err(format!(
                "expanded struct nesting exceeds {MAX_ELABORATION_DEPTH} levels"
            ));
        }
        if !self.structs.active_templates.insert(&template.name) {
            return Err(format!(
                "recursive struct dependency involving '{}'",
                template.name
            ));
        }
        let result = {
            let dimensions = self.decls.module_dimensions(&template.name);
            let mut checker = BodyChecker::struct_type_scope(
                dimensions,
                self.resolver,
                self.structs,
                self.function_registry.as_deref_mut(),
            );
            checker.in_struct_definition = true;
            stacker::maybe_grow(1024 * 1024, 2 * 1024 * 1024, || evaluate(&mut checker))
        };
        self.structs.active_templates.remove(template.name.as_str());
        result
    }

    fn intern_struct(
        &mut self,
        template: &'m StructDef,
        args: StructArguments<'_>,
        field_budget: usize,
    ) -> Result<Type, String> {
        // A record-valued default can refer to the same specialization in its
        // annotation and its expression. Reuse it before evaluating either.
        let prefix = StructName::default_prefix(template, &args)?;
        if let Some(index) = prefix
            .as_ref()
            .and_then(|key| self.structs.default_instances.get(key))
        {
            let ty = Type::Named(self.structs.definitions[*index].name.clone());
            self.structs.type_shape(&ty, field_budget + 1)?;
            return Ok(ty);
        }
        let mut name = StructName::default();
        name.append(format_args!("{}<", template.name))?;
        let (values, types) = match args {
            StructArguments::Values { explicit, inferred } => (
                self.bind_struct_values(template, explicit, inferred, &mut name)?,
                Vec::new(),
            ),
            StructArguments::Types(args) => {
                let mut bindings = Bindings::new(&template.type_params);
                let mut types = Vec::with_capacity(args.len());
                for (index, (param, actual)) in template.type_params.iter().zip(args).enumerate() {
                    bindings.match_input(&Type::Named(param.clone()), actual, self.structs)?;
                    let ty = bindings.get(param).expect("bound type argument");
                    if index != 0 {
                        name.append(format_args!(", "))?;
                    }
                    name.append(format_args!("{ty}"))?;
                    types.push(ty.clone());
                }
                self.function_type_bindings = Some(bindings);
                (Vec::new(), types)
            }
        };
        name.append(format_args!(">"))?;
        let name = name.0;
        if !self.structs.instances.contains_key(&name) {
            if self.structs.instances.len() >= MAX_SPECIALIZATIONS {
                return Err(format!(
                    "struct specialization limit of {MAX_SPECIALIZATIONS} exceeded"
                ));
            }
            let fields = template
                .fields
                .iter()
                .map(|field| {
                    let ty = self.resolve_dimensions(&field.ty, field_budget)?;
                    let ty = match &self.function_type_bindings {
                        Some(bindings) => bindings.concrete_type(&ty)?,
                        None => ty,
                    };
                    Ok(Param {
                        name: field.name.clone(),
                        ty,
                    })
                })
                .collect::<Result<_, String>>()?;
            self.structs.update_imports(self.resolver)?;
            let def = StructDef {
                name: name.clone(),
                public: template.public,
                fields,
                value_parametrics: Vec::new(),
                type_params: Vec::new(),
            };
            let shape = self.structs.definition_shape(&def)?;
            // Phantom parameters still retain their value storage and nominal
            // spelling even when the materialized struct has only a tiny field.
            let mut nodes = self
                .structs
                .instance_layout_nodes
                .saturating_add(shape.ir_nodes);
            for bound in &values {
                let layout = self.structs.type_shape(&bound.ty, MAX_ELABORATION_DEPTH)?;
                nodes = nodes.saturating_add(layout.ir_nodes.max(layout.size.div_ceil(64)));
            }
            if !types.is_empty() || values.iter().any(|bound| bound.ty != dimension_type()) {
                nodes = nodes.saturating_add(name.len());
            }
            nodes = nodes.saturating_add(types.iter().map(super::source_type_nodes).sum::<usize>());
            if nodes > MAX_TYPE_SIZE {
                return Err(format!(
                    "struct specializations exceed the limit of {MAX_TYPE_SIZE} type nodes and bindings"
                ));
            }
            let index = self.structs.definitions.len();
            self.structs.definitions.push(def);
            self.structs.indices.insert(name.clone(), (index, shape));
            self.structs.instance_layout_nodes = nodes;
            self.structs.instances.insert(
                name.clone(),
                StructInstance {
                    template,
                    values,
                    types,
                },
            );
        }
        let ty = Type::Named(name.clone());
        self.structs.type_shape(&ty, field_budget + 1)?;
        if let Some(prefix) = prefix {
            let nodes = self
                .structs
                .instance_layout_nodes
                .saturating_add(prefix.len());
            if nodes > MAX_TYPE_SIZE {
                return Err(format!(
                    "struct specializations exceed the limit of {MAX_TYPE_SIZE} type nodes and bindings"
                ));
            }
            let index = self.structs.indices[&name].0;
            self.structs.default_instances.insert(prefix, index);
            self.structs.instance_layout_nodes = nodes;
        }
        Ok(ty)
    }

    /// Evaluate value arguments and defaults in the declaration's scope.
    fn bind_struct_values(
        &mut self,
        template: &StructDef,
        explicit: &[BoundParametricValue],
        inferred: Option<&InferredBindings<'_>>,
        name: &mut StructName,
    ) -> Result<Vec<BoundParametricValue>, String> {
        let mut values = Vec::with_capacity(template.value_parametrics.len());
        for (index, parametric) in template.value_parametrics.iter().enumerate() {
            let bound = if let Some(supplied) = explicit.get(index) {
                // Explicit arguments and contextual prefixes already carry
                // this template's resolved formal type.
                supplied.clone()
            } else {
                let ty = self.source_type(&parametric.ty)?;
                let value = if let Some(default) = &parametric.default {
                    self.names.insert(parametric.name.clone());
                    self.parametric_value(default, &ty)?
                } else {
                    let bound = inferred
                        .map(|b| b.struct_value(index, &ty))
                        .transpose()?
                        .flatten()
                        .ok_or_else(|| {
                            format!(
                                "cannot infer value parametric '{}' of struct '{}'",
                                parametric.name, template.name
                            )
                        })?;
                    if !super::compatible(&bound.ty, &ty) {
                        return Err(format!(
                            "parametric '{}' of struct '{}' expects {ty}, got {}",
                            parametric.name, template.name, bound.ty
                        ));
                    }
                    bound.value
                };
                BoundParametricValue { ty, value }
            };
            if index != 0 {
                name.append(format_args!(", "))?;
            }
            name.append_binding(&bound)?;
            self.bind_parametric_value(&parametric.name, &bound);
            values.push(bound);
        }
        Ok(values)
    }
}

/// An application supplies a prefix of the template's declared value arguments.
fn check_value_arity(template: &StructDef, count: usize) -> Result<(), String> {
    if !template.type_params.is_empty() {
        return Err(format!(
            "struct '{}' requires type arguments",
            template.name
        ));
    }
    if count > template.value_parametrics.len() {
        return Err(format!(
            "struct '{}' expects at most {} value parametrics, got {count}",
            template.name,
            template.value_parametrics.len(),
        ));
    }
    Ok(())
}

/// Check an explicit struct type argument list before deferring any layout.
pub(super) fn check_type_arity(template: &StructDef, count: usize) -> Result<(), String> {
    if template.type_params.is_empty() || count != template.type_params.len() {
        return Err(format!(
            "struct '{}' expects {} type arguments, got {count}",
            template.name,
            template.type_params.len()
        ));
    }
    Ok(())
}

impl<'a, 'm> BodyChecker<'a, 'm> {
    /// Check type arguments in a module scope without inheriting data wires.
    pub(super) fn struct_type_scope(
        dimensions: ModuleDimensions<'a, 'm>,
        resolver: &'a mut ImportResolver,
        structs: &'a mut StructTable<'m>,
        functions: Option<&'a mut FunctionSpecializations<'m>>,
    ) -> Self {
        let mut checker = Self::new(dimensions.decls, resolver, structs, BodyKind::Function);
        checker.function_position = dimensions.position;
        checker.warning_owner = dimensions.warning_owner;
        checker.zero_binding = dimensions.zero_binding();
        checker.function_registry = functions;
        checker.enable_constant_functions();
        checker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_oversized_struct_argument_can_be_printed_in_a_diagnostic() {
        let value = BoundParametricValue {
            ty: Type::Named("A".repeat(MAX_TYPE_SIZE)),
            value: IrValue::make_tuple(&[]),
        };
        assert_eq!(
            StructApplication::Values("Record", &[value]).to_string(),
            "Record<…>"
        );
    }
}
