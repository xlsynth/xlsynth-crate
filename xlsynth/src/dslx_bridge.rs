// SPDX-License-Identifier: Apache-2.0

//! Library for generating Rust code that reflects the types and callables in a
//! DSLX module subtree.
//!
//! We walk the type definitions and callable interfaces present in the DSLX
//! module and generate corresponding Rust code that can be `use`'d into a Rust
//! module.

use crate::{dslx, IrValue, XlsynthError};

/// Abstract interface for building bridge code; i.e. interop to or from DSLX
/// with another language like Rust or SystemVerilog.
pub trait BridgeBuilder {
    fn start_module(&mut self, module_name: &str) -> Result<(), XlsynthError>;

    fn end_module(&mut self, module_name: &str) -> Result<(), XlsynthError>;

    /// `is_signed` indicates whether the bits type underlying the enum is
    /// signed.
    fn add_enum_def(
        &mut self,
        dslx_name: &str,
        is_signed: bool,
        underlying_bit_count: usize,
        members: &[(String, IrValue)],
    ) -> Result<(), XlsynthError>;

    fn add_struct_def(
        &mut self,
        dslx_name: &str,
        members: &[(String, dslx::Type)],
    ) -> Result<(), XlsynthError>;

    /// Invoked when there is a type alias to emit for this module (i.e. not an
    /// import-style type alias that refers, via a ColonRef, to a definition
    /// in a different module).
    fn add_alias(&mut self, dslx_name: &str, ty: dslx::Type) -> Result<(), XlsynthError>;
}

fn enum_as_tups(enum_def: &dslx::EnumDef, type_info: &dslx::TypeInfo) -> Vec<(String, IrValue)> {
    let mut tups = vec![];
    for i in 0..enum_def.get_member_count() {
        let member = enum_def.get_member(i);
        let member_name = member.get_name();
        let member_expr = member.get_value();
        let member_const = type_info
            .get_const_expr(member_expr)
            .expect("enum values should be constexpr");
        let member_const_ir = member_const.convert_to_ir().unwrap();
        tups.push((member_name, member_const_ir));
    }
    tups
}

fn convert_enum(
    enum_def: &dslx::EnumDef,
    type_info: &dslx::TypeInfo,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let tups = enum_as_tups(enum_def, type_info);
    let enum_underlying = type_info.get_type_for_type_annotation(enum_def.get_underlying());
    let (is_signed, underlying_bit_count) = enum_underlying.is_bits_like().expect(&format!(
        "enum underlying type should be bits-like; got: {enum_underlying}"
    ));
    let enum_name = enum_def.get_identifier();
    builder.add_enum_def(&enum_name, is_signed, underlying_bit_count, &tups)
}

fn convert_struct(
    struct_def: &dslx::StructDef,
    type_info: &dslx::TypeInfo,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let struct_name = struct_def.get_identifier();
    let mut members = vec![];
    for i in 0..struct_def.get_member_count() {
        let member = struct_def.get_member(i);
        let member_name = member.get_name();
        let member_type = type_info.get_type_for_struct_member(&member);
        members.push((member_name, member_type));
    }
    builder.add_struct_def(&struct_name, &members)
}

fn convert_type_alias(
    type_alias: &dslx::TypeAlias,
    type_info: &dslx::TypeInfo,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let alias_name = type_alias.get_identifier();

    // If the alias right-hand-side is a ColonRef with a different module as a
    // subject we skip it, because it's an import style re-binding pattern.
    let type_annotation = type_alias.get_type_annotation();

    if let Some(type_ref_type_annotation) = type_annotation.to_type_ref_type_annotation() {
        let type_ref: dslx::TypeRef = type_ref_type_annotation.get_type_ref();
        let type_definition: dslx::TypeDefinition = type_ref.get_type_definition();
        if let Some(colon_ref) = type_definition.to_colon_ref() {
            if let Some(_import) = colon_ref.resolve_import_subject() {
                // Skip the "use-style" typedef.
                return Ok(());
            }
        }
    }

    let alias_type = type_info.get_type_for_type_annotation(type_annotation);
    builder.add_alias(&alias_name, alias_type)
}

pub fn convert_leaf_module(
    import_data: &mut dslx::ImportData,
    dslx_program: &str,
    path: &std::path::Path,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    // If the path is `path/to/foo.x` then the module name is `foo`.
    let module_name = path.file_stem().unwrap().to_str().unwrap();
    let path_str = path.to_str().unwrap();
    let typechecked_module =
        dslx::parse_and_typecheck(dslx_program, path_str, module_name, import_data)?;
    let module = typechecked_module.get_module();
    let type_info = typechecked_module.get_type_info();

    builder.start_module(module_name)?;
    for i in 0..module.get_type_definition_count() {
        let type_def_kind = module.get_type_definition_kind(i);
        match type_def_kind {
            dslx::TypeDefinitionKind::EnumDef => {
                let enum_def = module.get_type_definition_as_enum_def(i).unwrap();
                convert_enum(&enum_def, &type_info, builder)?
            }
            dslx::TypeDefinitionKind::StructDef => {
                let struct_def = module.get_type_definition_as_struct_def(i).unwrap();
                convert_struct(&struct_def, &type_info, builder)?
            }
            dslx::TypeDefinitionKind::TypeAlias => {
                let type_alias = module.get_type_definition_as_type_alias(i).unwrap();
                convert_type_alias(&type_alias, &type_info, builder)?
            }
            dslx::TypeDefinitionKind::ProcDef => {
                todo!("convert impl-style proc definition from DSLX to Rust")
            }
            dslx::TypeDefinitionKind::ColonRef => todo!("convert colon ref from DSLX to Rust"),
        }
    }
    builder.end_module(module_name)?;
    Ok(())
}
