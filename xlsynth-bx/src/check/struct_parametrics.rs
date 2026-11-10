// SPDX-License-Identifier: Apache-2.0

//! Check value-parametric BX structs and intern their concrete layouts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

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
}

impl<'m> StructTable<'m> {
    /// Record a template before checking the concrete declarations that use it.
    pub(super) fn register_template(&mut self, def: &'m StructDef) -> Result<(), String> {
        if def.value_parametrics.is_empty() {
            return Ok(());
        }
        let mut names = BTreeSet::new();
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
    pub(super) fn inferred_struct_constructor<'e>(&self, expr: &'e Expr) -> Option<&'e str> {
        let (name, supplied) = match expr {
            Expr::Struct { name, .. } => (name, 0),
            Expr::AppliedStruct { name, args, .. } => (name, args.len()),
            Expr::ParametricStruct { name, args, .. } => (name, args.len()),
            _ => return None,
        };
        let def = self.structs.template(name)?;
        (supplied < def.value_parametrics.len()).then_some(name)
    }

    /// Merge explicit arguments from sibling constructors, then use the final
    /// constructor's fields to infer the remaining bindings, as DSLX does.
    pub(super) fn infer_struct_siblings(
        &mut self,
        items: &[&Expr],
    ) -> Result<Option<Vec<TypedExpr>>, String> {
        let Some((last, siblings)) = items.split_last().filter(|(_, rest)| !rest.is_empty()) else {
            return Ok(None);
        };
        let Some(name) = self.inferred_struct_constructor(last) else {
            return Ok(None);
        };
        if !siblings
            .iter()
            .all(|item| self.inferred_struct_constructor(item) == Some(name))
        {
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

    /// Resolve the nominal type of a constructor, using context and field
    /// types.
    pub(super) fn struct_literal_type<'e>(
        &mut self,
        source: &Type,
        supplied: &BTreeMap<&str, &'e Expr>,
        expected: Option<&Type>,
    ) -> Result<(Type, BTreeMap<String, Input<'e>>), String> {
        let (name, explicit) = match source {
            Type::Named(name) => (name, &[][..]),
            Type::Applied { name, args } => (name, args.as_slice()),
            _ => return Err(format!("'{source}' is not a struct")),
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
            return Err(format!("'{name}' names a local value here"));
        }
        let context = match expected {
            Some(Type::Named(name)) => self.structs.instances.get(name),
            _ => None,
        };
        let contextual = context
            .filter(|instance| {
                instance.template.name == template.name && instance.values.starts_with(explicit)
            })
            .map(|instance| instance.values.clone());
        let explicit = contextual.as_deref().unwrap_or(explicit);
        let mut inferred = InferredBindings::new(&template.value_parametrics, explicit.len());
        let mut inputs = BTreeMap::new();
        let mut types = Bindings::new(&[]);
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
            if !matches!(input, Input::Contextual(_)) {
                inputs.insert(field.name.clone(), input);
            }
            inferred.finish_argument();
        }
        let ty = self.specialize_struct(
            template,
            explicit,
            Some(inferred.values),
            MAX_ELABORATION_DEPTH,
        )?;
        Ok((ty, inputs))
    }

    /// Bind defaults and resolve fields in the template's declaration scope.
    pub(super) fn specialize_struct(
        &mut self,
        template: &'m StructDef,
        explicit: &[u32],
        inferred: Option<Vec<Option<BoundParametricValue>>>,
        remaining: usize,
    ) -> Result<Type, String> {
        if explicit.len() > template.value_parametrics.len() {
            return Err(format!(
                "struct '{}' expects at most {} value parametrics, got {}",
                template.name,
                template.value_parametrics.len(),
                explicit.len()
            ));
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
                checker.intern_struct(
                    template,
                    explicit,
                    inferred.as_deref().unwrap_or(&[]),
                    field_budget,
                )
            })
        };
        self.structs.active_templates.remove(template.name.as_str());
        result
    }

    fn intern_struct(
        &mut self,
        template: &'m StructDef,
        explicit: &[u32],
        inferred: &[Option<BoundParametricValue>],
        field_budget: usize,
    ) -> Result<Type, String> {
        let mut values = Vec::with_capacity(template.value_parametrics.len());
        let mut name = format!("{}<", template.name);
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
                name.push_str(", ");
            }
            write!(name, "{argument}").expect("writing to a String");
            values.push(argument);
            self.bind_parametric_value(
                &parametric.name,
                &BoundParametricValue {
                    ty: parametric.ty.clone(),
                    value,
                },
            );
        }
        name.push('>');
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
                    Ok(Param {
                        name: field.name.clone(),
                        ty: self.resolve_dimensions(&field.ty, field_budget)?,
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
            });
            let shape = self.structs.index(index)?;
            let nodes = self
                .structs
                .instance_layout_nodes
                .saturating_add(shape.ir_nodes)
                .saturating_add(values.len());
            if nodes > MAX_TYPE_SIZE {
                return Err(format!(
                    "struct specializations exceed the limit of {MAX_TYPE_SIZE} type nodes and bindings"
                ));
            }
            self.structs.instance_layout_nodes = nodes;
            self.structs
                .instances
                .insert(name.clone(), StructInstance { template, values });
        }
        let ty = Type::Named(name);
        self.structs.type_shape(&ty, field_budget + 1)?;
        Ok(ty)
    }
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
