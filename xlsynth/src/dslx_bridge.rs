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
    /// Member identifier as written in the DSLX struct definition.
    pub name: String,
    /// Source annotation attached to the member.
    pub type_annotation: dslx::TypeAnnotation,
    /// Concrete DSLX type resolved by typechecking the annotation.
    pub concrete_type: dslx::Type,
}

/// Signature data for one DSLX function parameter.
///
/// Builders can use the source annotation to preserve semantic names such as
/// type aliases or imported module paths while using the concrete type to
/// validate the lowered ABI shape.
pub struct FunctionParamData {
    /// Parameter identifier as written in the DSLX function signature.
    pub name: String,
    /// Source annotation attached to the parameter.
    pub type_annotation: dslx::TypeAnnotation,
    /// Concrete DSLX type when the current `TypeInfo` can resolve it.
    ///
    /// Some callable metadata can be present before every annotation has a
    /// concrete type exposed through the Rust wrapper API; builders that only
    /// need source-level type names should use `type_annotation` instead of
    /// requiring this value.
    pub concrete_type: Option<dslx::Type>,
}

/// Abstract interface for building bridge code; i.e. interop to or from DSLX
/// with another language like Rust or SystemVerilog.
pub trait BridgeBuilder {
    /// Starts collecting declarations for one DSLX module.
    ///
    /// Implementations should reset module-local state here. Returning an error
    /// stops conversion before any members from the module are visited.
    fn start_module(&mut self, module_name: &str) -> Result<(), XlsynthError>;

    /// Starts a module and also provides its DSLX text when a builder needs
    /// source-level annotations that are not exposed through the current AST
    /// wrapper API.
    fn start_module_with_text(
        &mut self,
        module_name: &str,
        _module_text: &str,
    ) -> Result<(), XlsynthError> {
        self.start_module(module_name)
    }

    /// Finishes collecting declarations for one DSLX module.
    ///
    /// Implementations commonly use this to close language-specific namespace
    /// state or append generated epilogues.
    fn end_module(&mut self, module_name: &str) -> Result<(), XlsynthError>;

    /// Adds a DSLX enum definition to the current bridge module.
    ///
    /// `is_signed` indicates whether the bits type underlying the enum is
    /// signed, and `underlying_bit_count` gives its width. The `members` values
    /// are already evaluated constant values, so builders do not need their own
    /// constexpr interpreter.
    fn add_enum_def(
        &mut self,
        dslx_name: &str,
        is_signed: bool,
        underlying_bit_count: usize,
        members: &[(String, IrValue)],
    ) -> Result<(), XlsynthError>;

    /// Adds a DSLX struct definition to the current bridge module.
    ///
    /// Builders receive both each member's source annotation and resolved
    /// concrete type so they can choose between semantic names and structural
    /// fallback types.
    fn add_struct_def(
        &mut self,
        dslx_name: &str,
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError>;

    /// Adds a struct definition while also providing the struct's DSLX text.
    ///
    /// Builders should prefer the structured member annotations in
    /// `StructMemberData`; the text is for nested annotation shapes that are
    /// not yet exposed by the DSLX Rust wrappers.
    fn add_struct_def_with_text(
        &mut self,
        dslx_name: &str,
        _struct_text: &str,
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError> {
        self.add_struct_def(dslx_name, members)
    }

    /// Invoked when there is a type alias to emit for this module.
    ///
    /// Args:
    /// - `dslx_name`: The name of the type alias (i.e. `Foo` in `type Foo =
    ///   Bar`).
    /// - `type_annotation`: The type annotation for the alias (i.e. `Bar` in
    ///   `type Foo = Bar`).
    /// - `ty`: The concrete type that this resolves to; i.e. after types have
    ///   been resolved to concrete values.
    fn add_alias(
        &mut self,
        dslx_name: &str,
        type_annotation: &dslx::TypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<(), XlsynthError>;

    /// Adds a type alias while also providing the alias's DSLX text.
    ///
    /// The default delegates to `add_alias`; builders that need nested source
    /// annotations can parse `_alias_text`.
    fn add_alias_with_text(
        &mut self,
        dslx_name: &str,
        _alias_text: &str,
        type_annotation: &dslx::TypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        self.add_alias(dslx_name, type_annotation, ty)
    }

    /// Adds a DSLX constant definition to the current bridge module.
    ///
    /// The constant expression has already been evaluated to an `IrValue`.
    /// Builders that do not expose constants can implement this as a no-op.
    fn add_constant(
        &mut self,
        name: &str,
        constant_def: &dslx::ConstantDef,
        ty: &dslx::Type,
        ir_value: &IrValue,
    ) -> Result<(), XlsynthError>;

    /// Invoked for function signatures when a builder wants callable metadata.
    ///
    /// The default is intentionally a no-op so existing bridge emitters that
    /// only care about types keep ignoring functions.
    fn add_function_signature(
        &mut self,
        _dslx_name: &str,
        _params: &[FunctionParamData],
        _return_type_annotation: Option<&dslx::TypeAnnotation>,
        _return_type: Option<&dslx::Type>,
    ) -> Result<(), XlsynthError> {
        Ok(())
    }

    /// Adds function signature metadata while also providing the function's
    /// DSLX text.
    fn add_function_signature_with_text(
        &mut self,
        dslx_name: &str,
        _function_text: &str,
        params: &[FunctionParamData],
        return_type_annotation: Option<&dslx::TypeAnnotation>,
        return_type: Option<&dslx::Type>,
    ) -> Result<(), XlsynthError> {
        self.add_function_signature(dslx_name, params, return_type_annotation, return_type)
    }
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
    let enum_underlying = type_info
        .get_type_for_type_annotation(&enum_def.get_underlying())
        .expect("enum underlying type should be present");
    let (is_signed, underlying_bit_count) = enum_underlying
        .is_bits_like()
        .unwrap_or_else(|| panic!("enum underlying type should be bits-like"));
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
    builder.add_struct_def_with_text(&struct_name, &struct_def.to_text(), &members)
}

fn convert_type_alias(
    type_alias: &dslx::TypeAlias,
    type_info: &dslx::TypeInfo,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let alias_name = type_alias.get_identifier();
    let type_annotation = type_alias.get_type_annotation();
    let alias_type = type_info
        .get_type_for_type_annotation(&type_annotation)
        .expect("alias type should be present");
    builder.add_alias_with_text(
        &alias_name,
        &type_alias.to_text(),
        &type_annotation,
        &alias_type,
    )
}

fn convert_constant(
    constant_def: &dslx::ConstantDef,
    type_info: &dslx::TypeInfo,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let ty = type_info.get_type_for_constant_def(constant_def);
    let value = constant_def.get_value();
    let interp_value = type_info.get_const_expr(&value)?;
    let ir_value = interp_value.convert_to_ir()?;
    builder.add_constant(&constant_def.get_name(), constant_def, &ty, &ir_value)
}

fn convert_function(
    function: &dslx::Function,
    type_info: &dslx::TypeInfo,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let params = (0..function.get_param_count())
        .map(|index| {
            let param = function.get_param(index);
            let type_annotation = param.get_type_annotation();
            let concrete_type = type_info.get_type_for_type_annotation(&type_annotation);
            FunctionParamData {
                name: param.get_name(),
                type_annotation,
                concrete_type,
            }
        })
        .collect::<Vec<_>>();
    let return_type_annotation = function.get_return_type();
    let return_type = return_type_annotation
        .as_ref()
        .and_then(|annotation| type_info.get_type_for_type_annotation(annotation));
    builder.add_function_signature_with_text(
        &function.get_identifier(),
        &function.to_text(),
        &params,
        return_type_annotation.as_ref(),
        return_type.as_ref(),
    )
}

/// Converts one typechecked DSLX module through a bridge builder.
///
/// The converter walks module members in source order and dispatches every
/// bridgeable enum, struct, alias, constant, and function signature. Quickcheck
/// definitions are skipped because they are test declarations, not bridgeable
/// interface items.
///
/// # Errors
///
/// Returns an error if the builder rejects any emitted item or if constant
/// evaluation fails while preparing a bridge callback.
pub fn convert_imported_module(
    typechecked_module: &dslx::TypecheckedModule,
    builder: &mut dyn BridgeBuilder,
) -> Result<(), XlsynthError> {
    let module = typechecked_module.get_module();
    let type_info = typechecked_module.get_type_info();
    let module_name = module.get_name();
    builder.start_module_with_text(&module_name, &module.to_text())?;

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
            dslx::MatchableModuleMember::Function(function) => {
                convert_function(&function, &type_info, builder)?
            }
            dslx::MatchableModuleMember::Quickcheck(_) => {
                // Quickchecks are currently not converted by the bridge.
                continue;
            }
        }
    }

    builder.end_module(&module_name)?;
    Ok(())
}

/// Parses, typechecks, and converts one standalone DSLX module.
///
/// This is the convenience entry point for callers that have source text and a
/// filesystem path but not an existing `TypecheckedModule`. Imports are
/// resolved through the provided `ImportData`.
///
/// # Errors
///
/// Returns an error if parsing, typechecking, or bridge emission fails.
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
