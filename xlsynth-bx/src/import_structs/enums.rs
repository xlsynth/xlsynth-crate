// SPDX-License-Identifier: Apache-2.0

//! Reflect the defining module's evaluated enum values and backing type.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use xlsynth::dslx::{self, ImportData};
use xlsynth_pir::IrBits;
use xlsynth_pir::libxls_bridge::bits_from_libxls;

use super::ImportedStructs;
use crate::ast::Type;
use crate::typed::CheckedEnum;

/// Public spellings and evaluated layouts keyed by the native declaration
/// handle.
#[derive(Default)]
pub(super) struct ImportedEnums {
    pub(super) exports: HashMap<dslx::EnumDef, String>,
    /// Member names for aliases whose target can only be identified by DSLX.
    pub(super) native_member_names: HashMap<dslx::EnumDef, BTreeSet<String>>,
    reflected: HashMap<dslx::EnumDef, usize>,
    contexts: BTreeMap<String, dslx::TypeInfo>,
    pub(super) definitions: Vec<CheckedEnum>,
}

impl ImportedStructs {
    /// Read member expressions using the type information of their original
    /// module.
    pub(super) fn reflect_enum(
        &mut self,
        ty: &dslx::Type,
        data: &mut ImportData,
    ) -> Result<Type, String> {
        let def = ty.get_enum_def().map_err(|e| e.to_string())?;
        if let Some(&index) = self.enums.reflected.get(&def) {
            return Ok(self.enums.definitions[index].ty());
        }
        let identifier = def.get_identifier();
        let export = self.enums.exports.get(&def).cloned();
        let width = ty.get_total_bit_count().map_err(|e| e.to_string())?;
        let signed = ty.is_signed_bits().map_err(|e| e.to_string())?;
        CheckedEnum::validate_size(width, def.get_member_count())?;
        let members = if def.get_member_count() == 0 {
            BTreeMap::new()
        } else {
            let owner = def.get_member(0).get_value().get_owner_module();
            let native = owner.get_name();
            let subject = self
                .files
                .as_ref()
                .and_then(|f| f.original_subject(&native))
                .ok_or_else(|| format!("missing snapshot owner for DSLX enum '{identifier}'"))?;
            if !self.enums.contexts.contains_key(subject) {
                let checked = self.check_probe(
                    &format!("import {subject} as bx_owner;"),
                    &format!("0bx_enum_owner_{}", self.enums.contexts.len()),
                    data,
                )?;
                let context = checked.get_type_info_for_module(&owner).ok_or_else(|| {
                    format!("missing type info for DSLX enum '{subject}::{identifier}'")
                })?;
                self.enums.contexts.insert(subject.to_owned(), context);
            }
            reflect_values(&def, &self.enums.contexts[subject], width)?
        };
        let index = self.enums.definitions.len();
        let public = export.is_some();
        let name = export.unwrap_or_else(|| format!("0bx_private_enum_{index}::{identifier}"));
        let reflected = CheckedEnum::new(name, public, signed, width, members);
        let ty = reflected.ty();
        self.enums.reflected.insert(def, index);
        self.enums.definitions.push(reflected);
        Ok(ty)
    }
}

/// Evaluate discriminants in the native type information of their declaring
/// module.
fn reflect_values(
    def: &dslx::EnumDef,
    info: &dslx::TypeInfo,
    width: usize,
) -> Result<BTreeMap<String, IrBits>, String> {
    let mut members = BTreeMap::new();
    for i in 0..def.get_member_count() {
        let member = def.get_member(i);
        let interp = info
            .get_const_expr(&member.get_value())
            .map_err(|e| e.to_string())?;
        let value = interp
            .convert_to_ir()
            .and_then(|v| v.to_bits())
            .map_err(|e| e.to_string())?;
        if value.get_bit_count() != width {
            return Err(format!(
                "member '{}' has the wrong width for DSLX enum '{}'",
                member.get_name(),
                def.get_identifier()
            ));
        }
        let bits = bits_from_libxls(&value).map_err(|e| e.to_string())?;
        members.insert(member.get_name(), bits);
    }
    Ok(members)
}
