// SPDX-License-Identifier: Apache-2.0

//! Check parametric BX structs and intern their concrete layouts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write};

use xlsynth_pir::IrValue;
use xlsynth_pir::ir_value_utils::ir_bits_to_usize;

use crate::ast::{Expr, Param, StructDef, Type};
use crate::typed::{BoundParametricValue, TypedExpr};

use super::generic_blocks::Bindings;
use super::generic_inputs::Input;
use super::value_parametrics::{InferredBindings, dimension_type};
use super::{
    BodyChecker, BodyKind, FunctionSpecializations, ImportResolver, MAX_ELABORATION_DEPTH,
    MAX_SPECIALIZATIONS, MAX_TYPE_SIZE, ModuleDimensions, StructTable, child_type_budget, declare,
};

/// The source identity and bindings of an interned nominal type.
pub(super) struct StructInstance<'m> {
    pub(super) template: &'m StructDef,
    pub(super) values: Vec<u32>,
    pub(super) types: Vec<Type>,
}

/// Bind a native struct's compile-time parameters in declaration order.
pub(super) enum StructArguments<'a> {
    Values {
        explicit: &'a [u32],
        inferred: &'a [Option<BoundParametricValue>],
    },
    Types(&'a [Type]),
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
    fn append(&mut self, args: fmt::Arguments<'_>) -> Result<(), String> {
        self.write_fmt(args)
            .map_err(|_| format!("struct specialization name exceeds {MAX_TYPE_SIZE} bytes"))
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
            if parametric.index != index || parametric.ty != dimension_type() {
                return Err(format!(
                    "struct '{}' value parametric '{}' must be declared as u32",
                    def.name, parametric.name
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
    /// Evaluate an application in its use-site scope before resolving its
    /// layout.
    pub(super) fn struct_actual_type(&mut self, name: &str, args: &[Expr]) -> Result<Type, String> {
        Ok(Type::Applied {
            name: name.to_owned(),
            args: self.struct_argument_values(args)?,
        })
    }

    /// Evaluate explicit struct bindings in the current expression scope.
    fn struct_argument_values(&mut self, args: &[Expr]) -> Result<Vec<u32>, String> {
        args.iter()
            .map(|arg| binding_value(&self.parametric_value(arg, &dimension_type())?))
            .collect()
    }

    /// Identify a native struct constructor with arguments still to infer.
    pub(super) fn inferred_struct_constructor(&self, expr: &Expr) -> Option<&'m StructDef> {
        let (name, supplied) = match expr {
            Expr::Struct { name, .. } => (name, 0),
            Expr::AppliedStruct { name, args, .. } => (name, args.len()),
            Expr::ParametricStruct { name, args, .. } => (name, args.len()),
            _ => return None,
        };
        let def = self.structs.template(name)?;
        let bare_type_struct = matches!(expr, Expr::Struct { .. })
            && !def.type_params.is_empty()
            && !self.type_bindings().is_some_and(|b| b.declares(name));
        (bare_type_struct || supplied < def.value_parametrics.len()).then_some(def)
    }

    /// Find constructors whose type can depend on an enclosing result group.
    pub(super) fn contains_type_struct_constructor(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Tuple(items) | Expr::Array(items) => items
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
        let mut explicit = Vec::new();
        let mut members = Vec::with_capacity(items.len());
        for item in items {
            let (args, fields) = match item {
                Expr::Struct { fields, .. } => (Vec::new(), fields),
                Expr::AppliedStruct { args, fields, .. } => (args.clone(), fields),
                Expr::ParametricStruct { args, fields, .. } => {
                    (self.struct_argument_values(args)?, fields)
                }
                _ => unreachable!("sibling is a native struct constructor"),
            };
            if let Some(index) = explicit.iter().zip(&args).position(|(a, b)| a != b) {
                return Err(format!(
                    "conflicting values for argument {} of struct '{name}'",
                    index + 1
                ));
            }
            if args.len() > explicit.len() {
                explicit = args;
            }
            members.push(fields);
        }
        let source = if explicit.is_empty() {
            Type::Named(name.to_owned())
        } else {
            Type::Applied {
                name: name.to_owned(),
                args: explicit,
            }
        };
        let fields = members.pop().expect("final constructor fields");
        let (ty, kind) = self.struct_literal(source.clone(), fields, None)?;
        let anchor = self.finish_expr(last, None, ty, kind)?;
        let mut values = Vec::with_capacity(items.len());
        for (item, fields) in siblings.iter().zip(members) {
            let expected = Some(&anchor.ty);
            let (ty, kind) = self.struct_literal(source.clone(), fields, expected)?;
            values.push(self.finish_expr(item, expected, ty, kind)?);
        }
        values.push(anchor);
        Ok(Some(values))
    }

    /// Check a bare type-parametric constructor without requiring it to supply
    /// every argument before the surrounding expression is inferred.
    pub(super) fn type_struct_witness(&mut self, expr: &Expr) -> Result<StructWitness, String> {
        let Expr::Struct { name, fields } = expr else {
            return Ok(StructWitness::Other);
        };
        if self
            .inferred_struct_constructor(expr)
            .is_none_or(|template| template.type_params.is_empty())
        {
            return Ok(StructWitness::Other);
        }
        match self.check_struct_literal(&Type::Named(name.clone()), fields, None) {
            Ok((ty, kind)) => self
                .finish_expr(expr, None, ty, kind)
                .map(StructWitness::Checked),
            Err(StructLiteralError::Unbound(_)) => Ok(StructWitness::Deferred),
            Err(error) => Err(error.into_message()),
        }
    }

    /// Resolve the nominal type of a constructor, using context and field
    /// types.
    pub(super) fn struct_literal_type<'e>(
        &mut self,
        source: &Type,
        supplied: &BTreeMap<&str, &'e Expr>,
        expected: Option<&Type>,
    ) -> Result<(Type, BTreeMap<String, Input<'e>>), StructLiteralError> {
        let (name, explicit) = match source {
            Type::Named(name) => (name, &[][..]),
            Type::Applied { name, args } => (name, args.as_slice()),
            Type::TypeApplied { .. } => return Ok((self.source_type(source)?, BTreeMap::new())),
            _ => return Err(format!("'{source}' is not a struct").into()),
        };
        let Some(template) = self.structs.template(name).filter(|_| {
            !self
                .type_bindings()
                .is_some_and(|bindings| bindings.declares(name))
        }) else {
            return Ok((self.source_type(source)?, BTreeMap::new()));
        };
        self.decls
            .check_type_name_visibility(name, self.function_position)?;
        if self.has_local_name(name) {
            return Err(format!("'{name}' names a local value here").into());
        }
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
        let mut inferred = InferredBindings::new(&template.value_parametrics, explicit.len());
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
                inferred: &inferred.values,
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
            .filter(|(_, input)| !matches!(input, Input::Contextual(_)))
            .map(|(field, input)| (field.name.clone(), input))
            .collect();
        Ok((ty, inputs))
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
                if !template.type_params.is_empty() {
                    return Err(format!(
                        "struct '{}' requires type arguments",
                        template.name
                    ));
                }
                if explicit.len() > template.value_parametrics.len() {
                    return Err(format!(
                        "struct '{}' expects at most {} value parametrics, got {}",
                        template.name,
                        template.value_parametrics.len(),
                        explicit.len()
                    ));
                }
            }
            StructArguments::Types(args) => check_type_arity(template, args.len())?,
        }
        let field_budget = child_type_budget(remaining)?;
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
            stacker::maybe_grow(1024 * 1024, 2 * 1024 * 1024, || {
                checker.intern_struct(template, args, field_budget)
            })
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
            let index = self.structs.definitions.len();
            self.structs.definitions.push(StructDef {
                name: name.clone(),
                public: template.public,
                fields,
                value_parametrics: Vec::new(),
                type_params: Vec::new(),
            });
            let shape = self.structs.index(index)?;
            let nodes = self
                .structs
                .instance_layout_nodes
                .saturating_add(shape.ir_nodes)
                .saturating_add(values.len())
                .saturating_add(if types.is_empty() { 0 } else { name.len() })
                .saturating_add(types.iter().map(super::source_type_nodes).sum::<usize>());
            if nodes > MAX_TYPE_SIZE {
                return Err(format!(
                    "struct specializations exceed the limit of {MAX_TYPE_SIZE} type nodes and bindings"
                ));
            }
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
        let ty = Type::Named(name);
        self.structs.type_shape(&ty, field_budget + 1)?;
        Ok(ty)
    }

    /// Evaluate value arguments and defaults in the declaration's scope.
    fn bind_struct_values(
        &mut self,
        template: &StructDef,
        explicit: &[u32],
        inferred: &[Option<BoundParametricValue>],
        name: &mut StructName,
    ) -> Result<Vec<u32>, String> {
        let mut values = Vec::with_capacity(template.value_parametrics.len());
        for (index, parametric) in template.value_parametrics.iter().enumerate() {
            let value = if let Some(&value) = explicit.get(index) {
                IrValue::u32(value)
            } else if let Some(default) = &parametric.default {
                self.names.insert(parametric.name.clone());
                self.parametric_value(default, &parametric.ty)?
            } else if let Some(Some(bound)) = inferred.get(index) {
                if bound.ty != parametric.ty {
                    return Err(format!(
                        "inferred parametric '{}' expects {}",
                        parametric.name, parametric.ty
                    ));
                }
                bound.value.clone()
            } else {
                return Err(format!(
                    "cannot infer value parametric '{}' of struct '{}'",
                    parametric.name, template.name
                ));
            };
            let argument = binding_value(&value)?;
            if index != 0 {
                name.append(format_args!(", "))?;
            }
            name.append(format_args!("{argument}"))?;
            values.push(argument);
            self.bind_parametric_value(
                &parametric.name,
                &BoundParametricValue {
                    ty: parametric.ty.clone(),
                    value,
                },
            );
        }
        Ok(values)
    }
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
        checker.zero_binding = dimensions.zero_binding();
        checker.function_registry = functions;
        checker.enable_constant_functions();
        checker
    }
}
