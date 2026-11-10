// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use xlsynth_pir::{BuilderError, ir};

use crate::ast::{StructDef, Type};

/// Expands named BX structures to tuples in declaration order, caching their
/// layouts.
pub(crate) struct Layouts<'a> {
    definitions: BTreeMap<&'a str, &'a StructDef>,
    expanded: BTreeMap<String, ir::Type>,
}

impl<'a> Layouts<'a> {
    pub(crate) fn new(structs: &'a [StructDef]) -> Self {
        Self {
            definitions: structs.iter().map(|def| (def.name.as_str(), def)).collect(),
            expanded: BTreeMap::new(),
        }
    }

    pub(crate) fn lower(&mut self, ty: &Type) -> Result<ir::Type, BuilderError> {
        Ok(match ty {
            Type::Bits { width, .. } | Type::Enum { width, .. } => ir::Type::Bits(*width),
            Type::Bool => ir::Type::Bits(1),
            Type::Applied { .. } | Type::ParametricStruct { .. } => {
                return Err(BuilderError::InvalidOperation(format!(
                    "unresolved struct type '{ty}'"
                )));
            }
            Type::ParametricBits { .. } | Type::ParametricArray { .. } => {
                return Err(BuilderError::InvalidOperation(format!(
                    "cannot lower unresolved dimensions in type '{ty}'"
                )));
            }
            Type::Array { element, size } => ir::Type::new_array(self.lower(element)?, *size),
            Type::Tuple(items) => ir::Type::Tuple(
                items
                    .iter()
                    .map(|ty| self.lower(ty).map(Box::new))
                    .collect::<Result<_, _>>()?,
            ),
            Type::Named(name) => {
                if let Some(layout) = self.expanded.get(name) {
                    return Ok(layout.clone());
                }
                let def = self
                    .definitions
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| {
                        BuilderError::InvalidOperation(format!("unknown struct '{name}'"))
                    })?;
                let fields = def
                    .fields
                    .iter()
                    .map(|field| self.lower(&field.ty).map(Box::new))
                    .collect::<Result<_, _>>()?;
                let layout = ir::Type::Tuple(fields);
                self.expanded.insert(name.clone(), layout.clone());
                layout
            }
        })
    }
}
