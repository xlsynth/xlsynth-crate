// SPDX-License-Identifier: Apache-2.0

//! Check parametric BX structs and intern their concrete layouts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write};

use xlsynth_pir::{IrFormatPreference, IrValue};

use crate::ast::{
    Expr, Param, SourceNode, StructArgumentRef, StructDef, StructLiteralArgs, StructParametricRef,
    Type,
};
use crate::builtin::BuiltinOp;
use crate::typed::{BoundParametricValue, TypedExpr, bits};

use super::counted_for::CarryContext;
use super::generic_blocks::Bindings;
use super::generic_inputs::Input;
use super::value_parametrics::{
    ExplicitScope, InferredBindings, TypeArgumentCopies, dimension_type,
};
use super::{
    BodyChecker, BodyKind, FunctionSpecializations, ImportResolver, MAX_ELABORATION_DEPTH,
    MAX_SPECIALIZATIONS, MAX_TYPE_SIZE, ModuleDimensions, StructTable, child_type_budget, declare,
};

/// The source identity and bindings of an interned nominal type.
pub(super) struct StructInstance<'m> {
    pub(super) template: &'m StructDef,
    pub(super) args: Vec<BoundStructArgument>,
}

/// One concrete argument at its position in the struct declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum BoundStructArgument {
    Type(Type),
    Value(BoundParametricValue),
}

/// The explicit prefix and field evidence used to bind the remaining formals.
#[derive(Default)]
pub(super) struct StructArguments<'a> {
    pub(super) explicit: &'a [BoundStructArgument],
    pub(super) inferred_values: Option<&'a InferredBindings<'a>>,
    pub(super) inferred_types: Option<&'a Bindings>,
}

/// A constructor's evaluated prefix, before its fields or base supply the
/// remaining arguments.
#[derive(Clone, Copy)]
pub(super) enum StructApplication<'a> {
    Named(&'a str),
    Applied(&'a str, &'a [BoundStructArgument]),
}

impl<'a> StructApplication<'a> {
    fn parts(self) -> (&'a str, &'a [BoundStructArgument]) {
        match self {
            Self::Named(name) => (name, &[]),
            Self::Applied(name, args) => (name, args),
        }
    }
}

impl fmt::Display for StructApplication<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(name) => f.write_str(name),
            Self::Applied(name, args) => {
                let mut bindings = StructName::default();
                for (index, arg) in args.iter().enumerate() {
                    if (index != 0 && bindings.append(format_args!(", ")).is_err())
                        || bindings.append_argument(arg).is_err()
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
        let explicit = args.explicit;
        let mut suffix = template
            .parametric_bindings()
            .skip(explicit.len())
            .peekable();
        if suffix.peek().is_none()
            || suffix.any(
                |formal| !matches!(formal, StructParametricRef::Value(p) if p.default.is_some()),
            )
        {
            return Ok(None);
        }
        let mut prefix = Self::default();
        prefix.append(format_args!("{}<", template.name))?;
        for (index, bound) in explicit.iter().enumerate() {
            if index > 0 {
                prefix.append(format_args!(", "))?;
            }
            prefix.append_argument(bound)?;
        }
        prefix.append(format_args!(">"))?;
        Ok(Some(prefix.0))
    }

    fn append(&mut self, args: fmt::Arguments<'_>) -> Result<(), String> {
        self.write_fmt(args)
            .map_err(|_| format!("struct specialization name exceeds {MAX_TYPE_SIZE} bytes"))
    }

    fn append_argument(&mut self, argument: &BoundStructArgument) -> Result<(), String> {
        match argument {
            BoundStructArgument::Type(ty) => self.append(format_args!("{ty}")),
            BoundStructArgument::Value(bound) => self.append_binding(bound),
        }
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
        let mut names = BTreeSet::new();
        for name in &def.type_params {
            if !crate::parser::is_identifier(name) {
                return Err(format!("invalid type parameter name '{name}'"));
            }
            declare(&mut names, name).map_err(|e| format!("struct '{}': {e}", def.name))?;
        }
        let arity = def.type_params.len() + def.value_parametrics.len();
        let mut previous = None;
        for parametric in &def.value_parametrics {
            declare(&mut names, &parametric.name)
                .map_err(|e| format!("struct '{}': {e}", def.name))?;
            if parametric.index >= arity || previous.is_some_and(|index| index >= parametric.index)
            {
                return Err(format!(
                    "struct '{}' value parametric '{}' has invalid declaration index {}",
                    def.name, parametric.name, parametric.index
                ));
            }
            previous = Some(parametric.index);
        }
        self.templates.insert(&def.name, def);
        Ok(())
    }

    pub(super) fn template(&self, name: &str) -> Option<&'m StructDef> {
        self.templates.get(name).copied()
    }
}

impl<'m> BodyChecker<'_, 'm> {
    /// Find the value substituted for a bare formal in a struct declaration.
    pub(super) fn direct_struct_binding(&self, expr: &Expr) -> Option<(&Type, &IrValue)> {
        let Expr::Name(name) = expr else { return None };
        if !self.in_struct_definition || !self.constant_pattern_names.contains(name) {
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

    /// Check a written value against the struct formal's declared type.
    fn bind_struct_argument(
        &mut self,
        arg: StructArgumentRef<'_>,
        ty: Type,
    ) -> Result<BoundParametricValue, String> {
        let value = match arg {
            StructArgumentRef::Value(expr) => self.struct_argument_value(expr, &ty)?,
            StructArgumentRef::Integer(value) => self.parametric_value(
                &Expr::Literal {
                    ty: None,
                    text: value.to_string(),
                },
                &ty,
            )?,
            StructArgumentRef::Type(_) => return Err("expected a struct value argument".into()),
        };
        Ok(BoundParametricValue { ty, value })
    }

    /// Bind an independent argument now. A dependent argument gets its
    /// concrete type and value when the block specializes.
    fn bind_or_defer_struct_value(
        &mut self,
        expr: &Expr,
        ty: Type,
        dimensions: ModuleDimensions<'_, 'm>,
    ) -> Result<Option<BoundParametricValue>, String> {
        if dimensions.depends_on_port_binding(expr) {
            return Ok(None);
        }
        self.bind_struct_argument(StructArgumentRef::Value(expr), ty)
            .map(Some)
    }

    /// Evaluate the supplied prefix using the imported declaration's formal
    /// types.
    pub(super) fn imported_struct_prefix(
        &mut self,
        subject: &str,
        member: &str,
        args: StructLiteralArgs<'_>,
    ) -> Result<Vec<BoundParametricValue>, String> {
        let arity = self.resolver.check_struct_template_name(subject, member)?;
        if args.len() > arity {
            return Err(format!(
                "struct '{subject}::{member}' takes at most {arity} arguments, got {}",
                args.len()
            ));
        }
        let mut bound = Vec::with_capacity(args.len());
        for arg in args.iter() {
            let ty = self
                .resolver
                .struct_parametric_type(subject, member, &bound)?;
            bound.push(self.bind_struct_argument(arg, ty)?);
        }
        Ok(bound)
    }

    /// Check every resolvable DSLX formal type, including omitted defaults,
    /// and bind supplied arguments that are already concrete.
    pub(super) fn check_deferred_imported_struct(
        &mut self,
        subject: &str,
        member: &str,
        args: StructLiteralArgs<'_>,
        dimensions: ModuleDimensions<'_, 'm>,
    ) -> Result<(), String> {
        let StructLiteralArgs::Expressions(args) = args else {
            return Err(format!(
                "expected value arguments for '{subject}::{member}'"
            ));
        };
        let arity = self.resolver.check_struct_template_name(subject, member)?;
        if args.len() > arity {
            return Err(format!(
                "struct '{subject}::{member}' takes at most {arity} arguments, got {}",
                args.len()
            ));
        }
        let formals = dimensions.block_formals.expect("dependent block port");
        let mut previous = Vec::with_capacity(args.len());
        for expr in args {
            self.check_block_dimension_names(expr, &formals.values)?;
            let ty = self
                .resolver
                .partial_struct_parametric_type(subject, member, &previous)?;
            let value = match ty {
                Some(ty) => self.bind_or_defer_struct_value(expr, ty, dimensions)?,
                None => None,
            };
            previous.push(value);
        }
        if args.len() < arity {
            self.resolver
                .check_struct_defaults(subject, member, previous)?;
        }
        Ok(())
    }

    /// Check an explicit prefix in order. Formal types are resolved in the
    /// declaration's scope, while each actual is evaluated in the caller.
    pub(super) fn struct_prefix(
        &mut self,
        name: &str,
        args: StructLiteralArgs<'_>,
    ) -> Result<Vec<BoundStructArgument>, String> {
        self.struct_prefix_at(name, args, MAX_ELABORATION_DEPTH, self.function_position)
    }

    /// Carry the surrounding annotation's depth budget through nested
    /// arguments. Constructor expressions also supply their function's source
    /// position. Annotations apply their enclosing declaration's visibility
    /// rules.
    pub(super) fn struct_prefix_at(
        &mut self,
        name: &str,
        args: StructLiteralArgs<'_>,
        remaining: usize,
        visibility_at: Option<usize>,
    ) -> Result<Vec<BoundStructArgument>, String> {
        let depth = child_type_budget(remaining)?;
        let mut copies = Some(TypeArgumentCopies::default());
        self.struct_prefix_with_types(name, args, |checker, ty| {
            checker
                .decls
                .check_type_visibility(ty, visibility_at, |name| {
                    checker.local_type_aliases.contains_key(name)
                        || checker.type_bindings().is_some_and(|b| b.declares(name))
                })?;
            let ty = checker.resolve_annotation(ty, depth, &mut copies)?;
            checker.structs.update_imports(checker.resolver)?;
            checker.structs.validate_type(&ty)?;
            Ok(ty)
        })
    }

    /// Visit formals in order using the type arguments supplied by the caller.
    pub(super) fn struct_prefix_with_types(
        &mut self,
        name: &str,
        args: StructLiteralArgs<'_>,
        mut resolve_type: impl FnMut(&mut Self, &Type) -> Result<Type, String>,
    ) -> Result<Vec<BoundStructArgument>, String> {
        let Some(template) = self.structs.template(name) else {
            if matches!(args, StructLiteralArgs::Omitted) {
                return Ok(Vec::new());
            }
            if matches!(
                args,
                StructLiteralArgs::Types(_) | StructLiteralArgs::Mixed(_)
            ) {
                return Err(format!("unknown type-parametric struct '{name}'"));
            }
            let (subject, member) = self.imported_struct_template(name)?;
            return Ok(self
                .imported_struct_prefix(&subject, &member, args)?
                .into_iter()
                .map(BoundStructArgument::Value)
                .collect());
        };
        let count = args.len();
        if matches!(args, StructLiteralArgs::Types(_)) {
            check_type_arity(template, count)?;
        }
        check_arity(template, count)?;
        let mut scope = ExplicitScope::default();
        let mut previous_value = None;
        let mut result = Vec::with_capacity(count);
        for (index, (formal, arg)) in template.parametric_bindings().zip(args.iter()).enumerate() {
            let argument = match (formal, arg) {
                (StructParametricRef::Type(param), StructArgumentRef::Type(ty)) => {
                    let actual = resolve_type(self, ty)?;
                    let bindings = scope.types.get_or_insert_with(|| Bindings::new(&[]));
                    bindings.declare_type_parameter(param);
                    bindings.match_input(&Type::Named(param.to_owned()), &actual, self.structs)?;
                    BoundStructArgument::Type(
                        bindings.get(param).expect("explicit type binding").clone(),
                    )
                }
                (StructParametricRef::Type(param), _) => {
                    return Err(format!(
                        "struct '{name}' requires type arguments for '{param}'"
                    ));
                }
                (StructParametricRef::Value(formal), arg) => {
                    let ty = self.in_struct_definition(template, |definition| {
                        scope.exchange(definition);
                        if let Some((prior, slot)) = previous_value.take() {
                            let Some(BoundStructArgument::Value(bound)) = result.get(slot) else {
                                unreachable!("preceding struct value argument");
                            };
                            definition.bind_parametric_value(prior, bound);
                        }
                        let result = definition.source_type(&formal.ty);
                        scope.exchange(definition);
                        result
                    })?;
                    if matches!(arg, StructArgumentRef::Type(_)) {
                        return Err(format!(
                            "struct '{name}' requires a value argument for '{}'",
                            formal.name
                        ));
                    }
                    let bound = self.bind_struct_argument(arg, ty).map_err(|e| {
                        format!("struct '{name}' parametric '{}': {e}", formal.name)
                    })?;
                    previous_value = Some((formal.name.as_str(), index));
                    BoundStructArgument::Value(bound)
                }
            };
            result.push(argument);
        }
        Ok(result)
    }

    /// Check concrete operands and omitted defaults while other struct
    /// arguments still depend on the caller's bindings.
    pub(super) fn check_deferred_struct_arguments(
        &mut self,
        template: &'m StructDef,
        source_args: StructLiteralArgs<'_>,
        args: StructLiteralArgs<'_>,
        caller_bindings: &Bindings,
        dimensions: ModuleDimensions<'_, 'm>,
    ) -> Result<(), String> {
        let name = &template.name;
        let mut scope = ExplicitScope::default();
        let mut pending_value = None;
        let mut unresolved = BTreeSet::new();
        for ((formal, arg), source) in template
            .parametric_bindings()
            .zip(args.iter())
            .zip(source_args.iter())
        {
            match (formal, arg) {
                (StructParametricRef::Type(param), StructArgumentRef::Type(ty)) => {
                    // A module alias may resolve to a nominal type sharing a
                    // formal's name.
                    let StructArgumentRef::Type(source_ty) = source else {
                        unreachable!("resolved type argument")
                    };
                    if caller_bindings.has_unbound_variable(source_ty)
                        || dimensions.has_dependent_port_operand(ty)
                    {
                        unresolved.insert(param);
                    } else {
                        let bindings = scope.types.get_or_insert_with(|| Bindings::new(&[]));
                        bindings.declare_type_parameter(param);
                        bindings.match_input(&Type::Named(param.to_owned()), ty, self.structs)?;
                    }
                }
                (StructParametricRef::Value(formal), StructArgumentRef::Value(expr)) => {
                    if let Some(formals) = dimensions.block_formals {
                        self.check_block_dimension_names(expr, &formals.values)?;
                    } else {
                        self.check_deferred_expr_names(expr)?;
                    }
                    if references_formal(SourceNode::Annotation(&formal.ty), |n| {
                        unresolved.contains(n)
                    }) || references_formal(SourceNode::Expression(expr), |n| {
                        caller_bindings.declares(n) && caller_bindings.get(n).is_none()
                    }) {
                        unresolved.insert(formal.name.as_str());
                        continue;
                    }
                    let ty = self.in_struct_definition(template, |definition| {
                        scope.exchange(definition);
                        if let Some((name, bound)) = pending_value.take() {
                            definition.bind_parametric_value(name, &bound);
                        }
                        let ty = definition.source_type(&formal.ty);
                        scope.exchange(definition);
                        ty
                    })?;
                    let bound = self
                        .bind_or_defer_struct_value(expr, ty, dimensions)
                        .map_err(|e| {
                            format!("struct '{name}' parametric '{}': {e}", formal.name)
                        })?;
                    if let Some(bound) = bound {
                        pending_value = Some((formal.name.as_str(), bound));
                    } else {
                        unresolved.insert(formal.name.as_str());
                    }
                }
                (StructParametricRef::Type(param), _) => {
                    return Err(format!(
                        "struct '{name}' requires type arguments for '{param}'"
                    ));
                }
                (StructParametricRef::Value(formal), _) => {
                    return Err(format!(
                        "struct '{name}' requires a value argument for '{}'",
                        formal.name
                    ));
                }
            }
        }
        let values = &template.value_parametrics;
        let first_default = values.partition_point(|formal| formal.index < args.len());
        if first_default == values.len() {
            return Ok(());
        }
        self.in_struct_definition(template, |definition| {
            scope.exchange(definition);
            if let Some((name, bound)) = pending_value {
                definition.bind_parametric_value(name, &bound);
            }
            for (index, formal) in values.iter().enumerate().skip(first_default) {
                // Arity checking requires every omitted formal to have a
                // default.
                let default = formal
                    .default
                    .as_ref()
                    .expect("checked omitted struct argument");
                let is_unresolved = |n: &str| unresolved.contains(n);
                if references_formal(SourceNode::Annotation(&formal.ty), is_unresolved)
                    || references_formal(SourceNode::Expression(default), is_unresolved)
                {
                    unresolved.insert(formal.name.as_str());
                    continue;
                }
                let bound = definition
                    .inferred_struct_value(template, formal, index, None)
                    .map_err(|e| format!("struct '{name}' default for '{}': {e}", formal.name))?;
                definition.bind_parametric_value(&formal.name, &bound);
            }
            Ok(())
        })
    }

    /// Resolve an explicit application after evaluating its prefix.
    pub(super) fn struct_applied_type(
        &mut self,
        name: &str,
        args: &[BoundStructArgument],
        remaining: usize,
    ) -> Result<Type, String> {
        self.check_constructor_name(name)?;
        self.reject_type_parameter_value(name)?;
        if let Some(template) = self.structs.template(name) {
            self.decls
                .check_type_name_visibility(name, self.function_position.or(self.block_position))?;
            self.specialize_struct(
                template,
                StructArguments {
                    explicit: args,
                    ..Default::default()
                },
                remaining,
            )
        } else {
            let (subject, member) = self.imported_struct_template(name)?;
            let values = args
                .iter()
                .map(|arg| match arg {
                    BoundStructArgument::Value(arg) => Ok(arg.clone()),
                    BoundStructArgument::Type(_) => {
                        Err(format!("struct '{name}' requires value arguments"))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let ty = self.resolver.resolve_applied(&subject, &member, &values)?;
            self.structs.update_imports(self.resolver)?;
            Ok(ty)
        }
    }

    /// Look up an applied DSLX type while honoring aliases in the BX scope.
    fn imported_struct_template(&self, name: &str) -> Result<(String, String), String> {
        if self.decls.type_aliases.contains_key(name) || self.local_type_aliases.contains_key(name)
        {
            return Err(format!("type alias '{name}' takes no arguments"));
        }
        self.resolver
            .lookup(name)?
            .ok_or_else(|| match name.split_once("::") {
                Some((qualifier, _)) => format!("unknown use binding '{qualifier}' in '{name}'"),
                None => format!("'{name}' requires an imported DSLX struct template"),
            })
    }

    /// Resolve a fully determined constructor in the surrounding type scope.
    fn struct_application_type(&mut self, source: StructApplication<'_>) -> Result<Type, String> {
        match source {
            StructApplication::Named(name) => self.source_type(&Type::Named(name.to_owned())),
            StructApplication::Applied(name, args) => {
                self.struct_applied_type(name, args, MAX_ELABORATION_DEPTH)
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
        if self.local_type_aliases.contains_key(name) {
            return None;
        }
        if matches!(literal.args, StructLiteralArgs::Types(_)) {
            return None;
        }
        let supplied = literal.args.len();
        let template = if let Some(def) = self.structs.template(name) {
            let bare_type_struct = matches!(literal.args, StructLiteralArgs::Omitted)
                && !def.type_params.is_empty()
                && !self.type_bindings().is_some_and(|b| b.declares(name));
            (bare_type_struct || supplied < def.value_parametrics.len() + def.type_params.len())
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

    /// Infer the common specialization of sibling constructors. Mixed structs
    /// share witnesses across their fields. Value-only structs use the final
    /// constructor's fields after merging their explicit arguments.
    pub(super) fn infer_struct_siblings(
        &mut self,
        items: &[&Expr],
        prefix: Option<(&str, &[BoundStructArgument])>,
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
        if template.value_parametrics.is_empty() {
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
            let args = self.struct_prefix(name, literal.args)?;
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
            StructApplication::Applied(name, &explicit)
        };
        if !template.type_params.is_empty() {
            // A complete sibling or splat base supplies the nominal type.
            // Probe the last constructor first to preserve dimension inference.
            for (index, (fields, base)) in members.iter().enumerate().rev() {
                if let StructWitness::Checked(anchor) =
                    self.struct_witness_with_prefix(items[index], source, fields, *base)?
                {
                    return self.check_siblings(items, index, anchor).map(Some);
                }
            }
            if members.iter().all(|(_, base)| base.is_none()) {
                self.check_constructor_name(name)?;
                let rows = members
                    .iter()
                    .map(|(fields, _)| {
                        let row = struct_field_map(source, fields)?;
                        if row.len() > template.fields.len() {
                            let field = row
                                .keys()
                                .find(|name| !template.fields.iter().any(|p| &p.name == *name))
                                .expect("extra constructor field");
                            return Err(format!("unknown field '{source}.{field}'"));
                        }
                        Ok(row)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let first = self.statements.len();
                let (ty, inputs) = match self.infer_struct_rows(
                    source,
                    template,
                    rows.iter().rev(),
                    Bindings::new(&template.type_params),
                ) {
                    Ok(result) => result,
                    Err(StructLiteralError::Unbound(_)) => {
                        self.discard_temporary_statements(first);
                        return Ok(None);
                    }
                    Err(error) => return Err(error.into_message()),
                };
                let Type::Named(ref canonical) = ty else {
                    unreachable!("interned struct type")
                };
                let fields = self.structs.definition(canonical)?.fields.clone();
                let mut inputs = inputs.into_iter();
                let mut values = Vec::with_capacity(items.len());
                for item in items.iter().rev() {
                    let members = fields
                        .iter()
                        .map(|p| {
                            let input = inputs.next().expect("one checked input per field");
                            self.finish_input(input, &p.ty)
                                .map_err(|e| format!("field '{source}.{}': {e}", p.name))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    values.push(self.finish_expr(
                        item,
                        None,
                        ty.clone(),
                        super::K::Tuple(members),
                    )?);
                }
                values.reverse();
                return Ok(Some(values));
            }
            return Ok(None);
        }
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
            Expr::Scope { statements, result } if !expr.needs_scope_preparation() => {
                return match self.type_struct_witness(result)? {
                    StructWitness::Checked(value) => self
                        .with_scope_result(statements, |_| Ok(value))
                        .map(StructWitness::Checked),
                    witness => Ok(witness),
                };
            }
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
        let Some(literal) = expr.struct_literal() else {
            return Ok(StructWitness::Other);
        };
        if self
            .inferred_struct_constructor(expr)
            .is_none_or(|template| template.type_params.is_empty())
        {
            return Ok(StructWitness::Other);
        }
        let args = self.struct_prefix(literal.name, literal.args)?;
        let source = if args.is_empty() {
            StructApplication::Named(literal.name)
        } else {
            StructApplication::Applied(literal.name, &args)
        };
        self.struct_witness_with_prefix(expr, source, literal.fields, literal.base)
    }

    /// Probe one constructor using a prefix shared by its peers or splat.
    fn struct_witness_with_prefix(
        &mut self,
        expr: &Expr,
        source: StructApplication<'_>,
        fields: &[(String, Expr)],
        base: Option<&crate::ast::StructSplat>,
    ) -> Result<StructWitness, String> {
        let first = self.statements.len();
        let base = if let Some(splat) = base {
            let Some(base) = self.constructor_dimension_witness(&splat.value)? else {
                self.discard_temporary_statements(first);
                return Ok(StructWitness::Deferred);
            };
            Some((splat, base))
        } else {
            None
        };
        match self.check_struct_literal(source, fields, base, None) {
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
            StructApplication::Applied(name, args)
                if self.structs.template(name).is_none_or(|template| {
                    args.len() < template.value_parametrics.len() + template.type_params.len()
                }) =>
            {
                self.splat_base_with_prefix(base, name, args)
            }
            StructApplication::Named(name)
                if !self.local_type_aliases.contains_key(name)
                    && !self
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
        prefix: &[BoundStructArgument],
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
            Expr::Scope { statements, result } => {
                return self.with_scope_result(statements, |checker| {
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
        let args = self.struct_prefix(literal.name, literal.args)?;
        let args = if prefix.starts_with(&args) {
            prefix.to_vec()
        } else if args.starts_with(prefix) {
            args
        } else {
            return Err(format!(
                "conflicting values for arguments of struct '{name}'"
            ));
        };
        let source = StructApplication::Applied(literal.name, &args);
        let (ty, kind) = self.struct_literal(source, literal.fields, literal.base, None)?;
        self.finish_expr(value, None, ty, kind)
    }

    /// Share the prefix and inferred specialization across every branch of
    /// the base, including results inside nested conditionals and matches.
    fn splat_branches_with_prefix(
        &mut self,
        items: &[&Expr],
        name: &str,
        prefix: &[BoundStructArgument],
    ) -> Result<Vec<TypedExpr>, String> {
        if items.iter().any(|item| item.needs_scope_preparation()) {
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
        let (name, prefix) = source.parts();
        let matched = self.struct_instance_matches(name, prefix, canonical)?;
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
        prefix: &[BoundStructArgument],
        canonical: &str,
    ) -> Result<Option<bool>, String> {
        if self.local_type_aliases.contains_key(name)
            || self
                .type_bindings()
                .is_some_and(|bindings| bindings.declares(name))
        {
            return Ok(None);
        }
        let matches = if let Some(instance) = self.structs.instances.get(canonical)
            && instance.template.name == name
        {
            instance.args.starts_with(prefix)
        } else if let Some(values) = self.resolver.struct_parametric_values(name, canonical)? {
            prefix.len() <= values.len()
                && prefix.iter().zip(values).all(|(bound, value)| {
                    matches!(bound, BoundStructArgument::Value(bound) if bound == value)
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
        supplied: &BTreeMap<&'e str, &'e Expr>,
        expected: Option<&Type>,
    ) -> Result<(Type, BTreeMap<String, Input<'e>>), StructLiteralError> {
        let (name, explicit) = source.parts();
        let Some(template) = self.structs.template(name).filter(|_| {
            !self.local_type_aliases.contains_key(name)
                && !self
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
        if let Some(instance) = context
            && instance.template.name == template.name
            && instance.args.starts_with(explicit)
        {
            return Ok((
                expected.expect("matching struct context").clone(),
                BTreeMap::new(),
            ));
        }
        if explicit.len() == template.value_parametrics.len() + template.type_params.len() {
            return Ok((self.struct_application_type(source)?, BTreeMap::new()));
        }
        let (ty, inputs) = self.infer_struct_rows(
            source,
            template,
            [supplied],
            Bindings::new(&template.type_params),
        )?;
        let inputs = template
            .fields
            .iter()
            .zip(inputs)
            .filter(|(_, input)| !matches!(input, Input::Contextual { .. }))
            .map(|(field, input)| (field.name.clone(), input))
            .collect();
        Ok((ty, inputs))
    }

    /// Collect type and dimension witnesses across constructors sharing one
    /// specialization.
    pub(super) fn infer_struct_rows<'e, 'r>(
        &mut self,
        source: StructApplication<'_>,
        template: &'m StructDef,
        rows: impl IntoIterator<Item = &'r BTreeMap<&'e str, &'e Expr>>,
        mut types: Bindings,
    ) -> Result<(Type, Vec<Input<'e>>), StructLiteralError>
    where
        'e: 'r,
    {
        let (name, explicit) = source.parts();
        let mut inferred =
            InferredBindings::for_struct(&template.value_parametrics, explicit.len());
        let mut inputs = Vec::new();
        for (formal, actual) in template.parametric_bindings().zip(explicit) {
            if let (StructParametricRef::Type(name), BoundStructArgument::Type(ty)) =
                (formal, actual)
            {
                types.match_input(&Type::Named(name.to_owned()), ty, self.structs)?;
            }
        }
        for row in rows {
            for field in &template.fields {
                let value = row
                    .get(field.name.as_str())
                    .ok_or_else(|| format!("missing field '{source}.{}'", field.name))?;
                inferred.infer_literal_values(&field.ty, value, self, MAX_ELABORATION_DEPTH)?;
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
        }
        if !template.type_params.is_empty() {
            self.infer_input_groups(
                &mut inputs,
                template.fields.iter().cycle(),
                &mut types,
                &format!("struct '{name}'"),
            )?;
            for param in &template.type_params {
                if types.get(param).is_none() {
                    return Err(StructLiteralError::Unbound(format!(
                        "cannot infer type parameter '{param}' of struct '{name}'"
                    )));
                }
            }
        }
        self.infer_contextual_defaults(
            &mut inputs,
            template.fields.iter().cycle(),
            name,
            &types,
            &mut inferred,
        )?;
        for (index, formal) in template.value_parametrics.iter().enumerate() {
            if formal.index >= explicit.len()
                && formal.default.is_none()
                && inferred.values[index].is_none()
            {
                return Err(StructLiteralError::Unbound(format!(
                    "cannot infer value parametric '{}' of struct '{name}'",
                    formal.name
                )));
            }
        }
        let args = StructArguments {
            explicit,
            inferred_types: Some(&types),
            inferred_values: Some(&inferred),
        };
        let ty = self.specialize_struct(template, args, MAX_ELABORATION_DEPTH)?;
        Ok((ty, inputs))
    }

    /// Bind defaults and resolve fields in the template's declaration scope.
    pub(super) fn specialize_struct(
        &mut self,
        template: &'m StructDef,
        args: StructArguments<'_>,
        remaining: usize,
    ) -> Result<Type, String> {
        check_arity(template, args.explicit.len())?;
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
        let mut result =
            Vec::with_capacity(template.value_parametrics.len() + template.type_params.len());
        let mut value_index = 0;
        for (index, formal) in template.parametric_bindings().enumerate() {
            let bound = match formal {
                StructParametricRef::Type(param) => {
                    let bindings = self
                        .function_type_bindings
                        .get_or_insert_with(|| Bindings::new(&[]));
                    bindings.declare_type_parameter(param);
                    let actual = match args.explicit.get(index) {
                        Some(BoundStructArgument::Type(ty)) => ty,
                        Some(BoundStructArgument::Value(_)) => {
                            return Err(format!(
                                "struct '{}' requires type arguments",
                                template.name
                            ));
                        }
                        None => {
                            args.inferred_types
                                .and_then(|b| b.get(param))
                                .ok_or_else(|| {
                                    format!(
                                        "cannot infer type parameter '{param}' of struct '{}'",
                                        template.name
                                    )
                                })?
                        }
                    };
                    bindings.match_input(&Type::Named(param.to_owned()), actual, self.structs)?;
                    BoundStructArgument::Type(
                        bindings
                            .get(param)
                            .expect("bound struct type argument")
                            .clone(),
                    )
                }
                StructParametricRef::Value(param) => {
                    let bound = match args.explicit.get(index) {
                        Some(BoundStructArgument::Value(value)) => value.clone(),
                        Some(BoundStructArgument::Type(_)) => {
                            return Err(format!(
                                "struct '{}' requires value arguments",
                                template.name
                            ));
                        }
                        None => self.inferred_struct_value(
                            template,
                            param,
                            value_index,
                            args.inferred_values,
                        )?,
                    };
                    self.bind_parametric_value(&param.name, &bound);
                    value_index += 1;
                    BoundStructArgument::Value(bound)
                }
            };
            if index != 0 {
                name.append(format_args!(", "))?;
            }
            name.append_argument(&bound)?;
            result.push(bound);
        }
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
                    let ty = self.resolve_annotation(
                        &field.ty,
                        field_budget,
                        &mut Some(TypeArgumentCopies::default()),
                    )?;
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
                derives_to_bits: template.derives_to_bits,
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
            for bound in &result {
                match bound {
                    BoundStructArgument::Value(bound) => {
                        let layout = self.structs.type_shape(&bound.ty, MAX_ELABORATION_DEPTH)?;
                        nodes = nodes.saturating_add(layout.ir_nodes.max(layout.size.div_ceil(64)));
                    }
                    BoundStructArgument::Type(ty) => {
                        nodes = nodes.saturating_add(super::source_type_nodes(ty));
                    }
                }
            }
            if result.iter().any(|arg| !matches!(arg, BoundStructArgument::Value(bound) if bound.ty == dimension_type())) {
                nodes = nodes.saturating_add(name.len());
            }
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
                    args: result,
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

    /// Evaluate a default or interpret its field witness using the formal's
    /// concrete type.
    fn inferred_struct_value(
        &mut self,
        template: &StructDef,
        parametric: &crate::ast::ParametricBinding,
        index: usize,
        inferred: Option<&InferredBindings<'_>>,
    ) -> Result<BoundParametricValue, String> {
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
        Ok(BoundParametricValue { ty, value })
    }
}

/// Index supplied fields for checking in declaration order.
pub(super) fn struct_field_map<'e>(
    source: StructApplication<'_>,
    fields: &'e [(String, Expr)],
) -> Result<BTreeMap<&'e str, &'e Expr>, String> {
    let mut supplied = BTreeMap::new();
    for (field, value) in fields {
        if supplied.insert(field.as_str(), value).is_some() {
            return Err(format!("duplicate field '{source}.{field}'"));
        }
    }
    Ok(supplied)
}

/// Check whether a type annotation or constexpr expression refers to an
/// enclosing formal, accounting for names rebound in local expressions.
pub(super) fn references_formal(node: SourceNode<'_>, is_formal: impl Fn(&str) -> bool) -> bool {
    node.free_names().any(is_formal)
}

/// An application can supply at most one argument per formal in source order.
pub(super) fn check_arity(template: &StructDef, count: usize) -> Result<(), String> {
    let arity = template.type_params.len() + template.value_parametrics.len();
    if count > arity {
        let kind = if template.type_params.is_empty() {
            "value parametrics"
        } else {
            "arguments"
        };
        return Err(format!(
            "struct '{}' expects at most {arity} {kind}, got {count}",
            template.name
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
        checker.signature_parameters = dimensions.data_parameters;
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
            StructApplication::Applied("Record", &[BoundStructArgument::Value(value)]).to_string(),
            "Record<…>"
        );
    }
}
