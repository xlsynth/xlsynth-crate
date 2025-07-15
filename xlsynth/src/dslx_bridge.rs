// SPDX-License-Identifier: Apache-2.0

//! Library for generating Rust code that reflects the types and callables in a
//! DSLX module subtree.
//!
//! We walk the type definitions and callable interfaces present in the DSLX
//! module and generate corresponding Rust code that can be `use`'d into a Rust
//! module.

use crate::{dslx, IrValue, XlsynthError};

/// Encapsulates information that the bridge builder gets about a struct member
/// -- the name, type annotation AST node, and deduced concrete type that the
/// annotation corresponds to are all provided.
///
/// The annotation can be used to determine if the type is an external type
/// reference, in which case the bridge builder may want to take different
/// actions.
pub struct StructMemberData {
    pub name: String,
    pub type_annotation: dslx::TypeAnnotation,
    pub concrete_type: dslx::Type,
}

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
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError>;

    /// Invoked when there is a type alias to emit for this module.
    ///
    /// Args:
    /// - `dslx_name`: The name of the type alias (i.e. `Foo` in `type Foo =
    ///   Bar`).
    /// - `type_annotation`: The type annotation for the alias (i.e. `Bar` in
    ///   `type Foo = Bar`).
    /// - `ty`: The concrete type that this resolves to; i.e. after types have
    /// been resolved to concrete values.
    fn add_alias(
        &mut self,
        dslx_name: &str,
        type_annotation: &dslx::TypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<(), XlsynthError>;

    fn add_constant(
        &mut self,
        name: &str,
        constant_def: &dslx::ConstantDef,
        ty: &dslx::Type,
        ir_value: &IrValue,
    ) -> Result<(), XlsynthError>;
}

fn enum_as_tups(enum_def: &dslx::EnumDef, type_info: &dslx::TypeInfo) -> Vec<(String, IrValue)> {
    let mut tups = vec![];
    for i in 0..enum_def.get_member_count() {
        let member = enum_def.get_member(i);
        let member_name = member.get_name();
        let member_expr = member.get_value();
        let member_const = type_info
            .get_const_expr(&member_expr)
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
    let enum_underlying = type_info.get_type_for_type_annotation(&enum_def.get_underlying());
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
        let member_type_annotation = member.get_type();
        let member_type = type_info.get_type_for_struct_member(&member);
        members.push(StructMemberData {
            name: member_name,
            type_annotation: member_type_annotation,
            concrete_type: member_type,
        });
    }
    builder.add_struct_def(&struct_name, &members)
}

fn convert_type_alias(
    type_alias: &dslx::TypeAlias,
    type_info: &dslx::TypeInfo,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let alias_name = type_alias.get_identifier();
    let type_annotation = type_alias.get_type_annotation();
    let alias_type = type_info.get_type_for_type_annotation(&type_annotation);
    builder.add_alias(&alias_name, &type_annotation, &alias_type)
}

fn convert_constant(
    constant_def: &dslx::ConstantDef,
    type_info: &dslx::TypeInfo,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let ty = type_info.get_type_for_constant_def(&constant_def);
    let value = constant_def.get_value();
    let interp_value = type_info.get_const_expr(&value)?;
    let ir_value = interp_value.convert_to_ir()?;
    builder.add_constant(&constant_def.get_name(), &constant_def, &ty, &ir_value)
}

pub fn convert_imported_module(
    typechecked_module: &dslx::TypecheckedModule,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let module = typechecked_module.get_module();
    let type_info = typechecked_module.get_type_info();
    let module_name = module.get_name();
    builder.start_module(&module_name)?;

    for i in 0..module.get_member_count() {
        let member = module.get_member(i);
        let matchable_member = member.to_matchable();
        if matchable_member.is_none() {
            continue;
        }
        match matchable_member.unwrap() {
            dslx::MatchableModuleMember::EnumDef(enum_def) => {
                convert_enum(&enum_def, &type_info, builder)?
            }
            dslx::MatchableModuleMember::StructDef(struct_def) => {
                convert_struct(&struct_def, &type_info, builder)?
            }
            dslx::MatchableModuleMember::TypeAlias(type_alias) => {
                convert_type_alias(&type_alias, &type_info, builder)?
            }
            dslx::MatchableModuleMember::ConstantDef(constant_def) => {
                convert_constant(&constant_def, &type_info, builder)?
            }
            dslx::MatchableModuleMember::Function(_function) => {
                // Functions are not converted by the bridge.
                continue;
            }
            dslx::MatchableModuleMember::QuickCheck(_) => {
                // QuickChecks are currently not converted by the bridge.
                continue;
            }
        }
    }

    builder.end_module(&module_name)?;
    Ok(())
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

    convert_imported_module(&typechecked_module, builder)
}
