// SPDX-License-Identifier: Apache-2.0

//! Read constexpr values from their declaring module in the frozen DSLX graph.

use std::collections::BTreeMap;
use std::rc::Rc;

use xlsynth::dslx::{self, ImportData, MatchableModuleMember};
use xlsynth_pir::{
    IrValue, ir, ir_value_utils::zero_ir_value_for_type, libxls_bridge::value_from_libxls,
};

use super::{DeclId, ImportedStructs};
use crate::ast::Type;
use crate::check::ImportedValueShapes;
use crate::dslx_import::TypeWitness;
use crate::layout::Layouts;
use crate::typed::CheckedConstant;

/// Native constant declarations and the context that evaluated their
/// initializers.
struct ConstantOwner {
    context: dslx::TypeInfo,
    declarations: BTreeMap<String, dslx::ConstantDef>,
    templates: BTreeMap<String, dslx::StructDef>,
}

/// Snapshot-scoped native declarations, materialized BX values and their
/// layouts.
#[derive(Default)]
pub(super) struct ImportedConstants {
    owners: BTreeMap<String, ConstantOwner>,
    values: BTreeMap<String, Rc<CheckedConstant>>,
    shapes: ImportedValueShapes,
    layouts: BTreeMap<String, ir::Type>,
}

impl ImportedStructs {
    pub(crate) fn has_public_constant(&self, subject: &str, member: &str) -> bool {
        self.graph.module_reference(subject, &[]).is_some_and(|m| {
            m.outline.public_values.contains(member) && !m.outline.public_functions.contains(member)
        })
    }

    /// Index original AST handles once per owner without materializing any
    /// values.
    fn index_constant_owner(&mut self, subject: &str, data: &mut ImportData) -> Result<(), String> {
        if self.constants.owners.contains_key(subject) {
            return Ok(());
        }
        let checked = self.check_probe(
            &format!("import {subject} as bx_owner;"),
            &format!("0bx_const_owner_{}", self.constants.owners.len()),
            data,
        )?;
        let import = checked
            .get_module()
            .get_member(0)
            .to_import()
            .ok_or("missing DSLX constant owner import")?;
        let root = checked.get_type_info();
        let owner = root
            .get_imported_module(&import)
            .ok_or_else(|| format!("missing imported DSLX module '{subject}'"))?;
        let context = root
            .get_imported_type_info(&owner)
            .ok_or_else(|| format!("missing type information for DSLX module '{subject}'"))?;
        let mut declarations = BTreeMap::new();
        let mut templates = BTreeMap::new();
        for index in 0..owner.get_member_count() {
            match owner.get_member(index).to_matchable() {
                Some(MatchableModuleMember::ConstantDef(def)) => {
                    declarations.insert(def.get_name(), def);
                }
                Some(MatchableModuleMember::StructDef(def)) if def.is_parametric() => {
                    let name = def.get_identifier();
                    if self.public_struct_template_arity(subject, &name).is_some() {
                        templates.insert(name, def);
                    }
                }
                _ => { /* Other declarations supply neither constants nor their templates. */ }
            }
        }
        self.constants.owners.insert(
            subject.to_owned(),
            ConstantOwner {
                context,
                declarations,
                templates,
            },
        );
        Ok(())
    }

    /// Pair an inferred parametric struct with its original public template
    /// handle.
    pub(super) fn index_template_owners(
        &mut self,
        def: &dslx::StructDef,
        data: &mut ImportData,
    ) -> Result<(), String> {
        let name = def.get_identifier();
        let graph = Rc::clone(&self.graph);
        for module in graph.sources().filter(|m| {
            graph
                .public_struct_template_arity(&m.subject, &name)
                .is_some()
        }) {
            self.index_constant_owner(&module.subject, data)?;
            if self.constants.owners[&module.subject].templates.get(&name) == Some(def) {
                self.bind_declaration(
                    def,
                    &DeclId {
                        subject: module.subject.clone(),
                        name,
                    },
                )?;
                break;
            }
        }
        Ok(())
    }

    /// Reflect the declared type and copy its evaluated initializer into PIR on
    /// first use.
    pub(crate) fn resolve_constant(
        &mut self,
        subject: &str,
        member: &str,
        data: &mut ImportData,
    ) -> Result<Rc<CheckedConstant>, String> {
        let name = format!("{subject}::{member}");
        if let Some(value) = self.constants.values.get(&name) {
            return Ok(Rc::clone(value));
        }
        if !self.has_public_constant(subject, member) {
            return Err(format!("'{name}' is not a public DSLX constant"));
        }
        self.index_constant_owner(subject, data)?;
        let owner = &self.constants.owners[subject];
        let def = owner
            .declarations
            .get(member)
            .ok_or_else(|| format!("missing DSLX constant '{name}'"))?;
        let native = owner
            .context
            .try_get_type_for_constant_def(def)
            .ok_or_else(|| format!("missing DSLX type for constant '{name}'"))?;
        let expr = def.get_value();
        let witness = TypeWitness {
            imports: format!("import {subject} as bx_owner;"),
            params: String::new(),
            expr: format!("bx_owner::{member}"),
        };
        let ty = self.reflect_at(&native, data, &witness, None, true, 0)?;
        self.constants.shapes.validate(&ty, &self.definitions)?;
        let layout = Layouts::lower_with(
            &ty,
            &|name| self.named_layouts.get(name).map(|&i| &self.definitions[i]),
            &mut self.constants.layouts,
        )
        .map_err(|e| e.to_string())?;
        let interp = self.constants.owners[subject]
            .context
            .get_const_expr(&expr)
            .map_err(|e| e.to_string())?;
        let value = if layout.bit_count() == 0 {
            zero_ir_value_for_type(&layout)
        } else if contains_empty_array(&layout) {
            self.copy_with_empty_arrays(&ty, &layout, &witness, data)?
        } else {
            let native = interp.convert_to_ir().map_err(|e| e.to_string())?;
            value_from_libxls(&native, &layout).map_err(|e| e.to_string())?
        };
        let constant = Rc::new(CheckedConstant {
            name: name.clone(),
            public: true,
            ty,
            value,
        });
        self.constants.values.insert(name, Rc::clone(&constant));
        Ok(constant)
    }

    /// Native IR values cannot contain empty arrays. Evaluate a projection with
    /// unit in those slots, then restore each empty array using its BX layout.
    fn copy_with_empty_arrays(
        &mut self,
        ty: &Type,
        layout: &ir::Type,
        witness: &TypeWitness,
        data: &mut ImportData,
    ) -> Result<IrValue, String> {
        let expr = self.project_empty_arrays(ty, layout, &witness.expr, 0)?;
        let probe = format!("0bx_const_projection_{}", self.probe_index);
        self.probe_index += 1;
        let source = format!("{}\nconst PROJECTED = {expr};", witness.imports);
        let checked = self.check_probe(&source, &probe, data)?;
        let module = checked.get_module();
        let def = (0..module.get_member_count())
            .find_map(|i| match module.get_member(i).to_matchable() {
                Some(MatchableModuleMember::ConstantDef(def)) => Some(def),
                _ => None,
            })
            .ok_or("missing projected DSLX constant")?;
        let projected = checked
            .get_type_info()
            .get_const_expr(&def.get_value())
            .and_then(|value| value.convert_to_ir())
            .map_err(|e| e.to_string())?;
        restore_empty_arrays(&projected, layout)
    }

    /// Generate one expression per layout member. A map projects all elements
    /// of an array even when its element type is private to another module.
    fn project_empty_arrays(
        &self,
        ty: &Type,
        layout: &ir::Type,
        expr: &str,
        depth: usize,
    ) -> Result<String, String> {
        if !contains_empty_array(layout) {
            return Ok(expr.to_owned());
        }
        if layout.bit_count() == 0 {
            return Ok("()".into());
        }
        match (ty, layout) {
            (Type::Array { element, .. }, ir::Type::Array(array)) => {
                let item = format!("item_{depth}");
                let projected =
                    self.project_empty_arrays(element, &array.element_type, &item, depth + 1)?;
                Ok(format!("map({expr}, |{item}| {projected})"))
            }
            (_, ir::Type::Tuple(fields)) => {
                let container = format!("container_{depth}");
                let members = fields
                    .iter()
                    .enumerate()
                    .map(|(i, layout)| {
                        let (ty, field) = match ty {
                            Type::Named(name) => {
                                let field = &self
                                    .definition(name)
                                    .ok_or("missing DSLX struct definition")?
                                    .fields[i];
                                (&field.ty, field.name.clone())
                            }
                            Type::Tuple(members) => (&members[i], i.to_string()),
                            _ => return Err("expected a DSLX record or tuple".to_owned()),
                        };
                        self.project_empty_arrays(
                            ty,
                            layout,
                            &format!("{container}.{field}"),
                            depth + 1,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let trailing = if members.len() == 1 { "," } else { "" };
                Ok(format!(
                    "{{ let {container} = {expr}; ({}{trailing}) }}",
                    members.join(", ")
                ))
            }
            _ => Err("expected a DSLX aggregate containing an empty array".into()),
        }
    }
}

fn contains_empty_array(ty: &ir::Type) -> bool {
    match ty {
        ir::Type::Array(a) => a.element_count == 0 || contains_empty_array(&a.element_type),
        ir::Type::Tuple(members) => members.iter().any(|ty| contains_empty_array(ty)),
        _ => false,
    }
}

/// Replace projected unit values with typed empty arrays in the original
/// layout.
fn restore_empty_arrays(value: &xlsynth::XlsIrValue, layout: &ir::Type) -> Result<IrValue, String> {
    if !contains_empty_array(layout) {
        return value_from_libxls(value, layout).map_err(|e| e.to_string());
    }
    if layout.bit_count() == 0 {
        return Ok(zero_ir_value_for_type(layout));
    }
    let member = |index, ty| {
        let value = value.get_element(index).map_err(|e| e.to_string())?;
        restore_empty_arrays(&value, ty)
    };
    match layout {
        ir::Type::Array(array) => {
            let elements = (0..array.element_count)
                .map(|i| member(i, &array.element_type))
                .collect::<Result<Vec<_>, _>>()?;
            IrValue::make_array_typed((*array.element_type).clone(), &elements)
                .map_err(|e| e.to_string())
        }
        ir::Type::Tuple(fields) => {
            let elements = fields
                .iter()
                .enumerate()
                .map(|(i, ty)| member(i, ty))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(IrValue::make_tuple(&elements))
        }
        _ => Err("expected a projected aggregate".into()),
    }
}
