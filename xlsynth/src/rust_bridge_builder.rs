// SPDX-License-Identifier: Apache-2.0

//! Builder that creates Rust type definitions from DSLX type definitions.
//!
//! This helps us e.g. call DSLX functions from Rust code, i.e. it enables
//! Rust->DSLX FFI interop.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    dslx,
    dslx_bridge::{BridgeBuilder, FunctionParamData, StructMemberData},
    ir_value::IrFormatPreference,
    IrValue, XlsynthError,
};

/// Emits Rust source that mirrors DSLX public type declarations.
///
/// The builder preserves DSLX module nesting and imported type references so
/// generated AOT wrappers can expose semantic DSLX-facing types instead of
/// only structural tuple and array shapes. It is intentionally text-producing:
/// callers are responsible for placing the resulting source in a build output
/// file and compiling it as normal Rust.
pub struct RustBridgeBuilder {
    lines: Vec<String>,
    module_path: Vec<String>,
    runner_items: Option<String>,
    leading_items: Vec<String>,
    emitted_parametric_structs: BTreeSet<String>,
    defer_parametric_struct_emission: bool,
    mode: RustBridgeMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Selects whether generated bridge types target normal `xlsynth` values or
/// the runtime-neutral standalone AOT surface.
enum RustBridgeMode {
    Xlsynth,
    Standalone,
}

/// Rust items generated for one DSLX module, before parent modules are
/// rendered.
///
/// Fragments let AOT generation collect multiple DSLX modules first and then
/// render one shared tree, which avoids each module independently creating its
/// own top-level namespace.
#[derive(Debug, Clone)]
pub(crate) struct RustModuleFragment {
    pub(crate) path: Vec<String>,
    pub(crate) body: String,
}

#[derive(Debug, Clone, Default)]
struct RustModuleNode {
    body: Vec<String>,
    children: BTreeMap<String, RustModuleNode>,
}

impl RustModuleNode {
    /// Inserts one generated module body into this subtree.
    fn with_fragment(mut self, path: &[String], body: String) -> Self {
        if let Some((module_name, child_path)) = path.split_first() {
            let child = self
                .children
                .remove(module_name)
                .unwrap_or_default()
                .with_fragment(child_path, body);
            self.children.insert(module_name.clone(), child);
            self
        } else {
            self.body.push(body);
            self
        }
    }

    /// Renders this subtree's direct body followed by nested child modules.
    fn render_contents(&self) -> String {
        self.body
            .iter()
            .cloned()
            .chain(
                self.children
                    .iter()
                    .map(|(module_name, child)| child.render_module(module_name)),
            )
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Renders this subtree as one public Rust module.
    fn render_module(&self, module_name: &str) -> String {
        format!("pub mod {module_name} {{\n{}\n}}", self.render_contents())
    }
}

/// Renders generated module fragments into a shared nested Rust module tree.
///
/// Fragments with common path prefixes are merged into the same `pub mod`
/// hierarchy. Passing fragments with duplicate paths appends both bodies in
/// insertion order, which is useful when a caller intentionally emits multiple
/// blocks for one DSLX module.
pub(crate) fn render_rust_module_fragments(
    fragments: impl IntoIterator<Item = RustModuleFragment>,
) -> String {
    fragments
        .into_iter()
        .fold(RustModuleNode::default(), |root, fragment| {
            root.with_fragment(&fragment.path, fragment.body)
        })
        .render_contents()
}

impl RustBridgeBuilder {
    /// Creates a builder that emits Rust definitions for DSLX types.
    pub fn new() -> Self {
        Self {
            lines: vec![],
            module_path: vec![],
            runner_items: None,
            leading_items: vec![],
            emitted_parametric_structs: BTreeSet::new(),
            defer_parametric_struct_emission: false,
            mode: RustBridgeMode::Xlsynth,
        }
    }

    /// Creates a builder that emits runtime-neutral Rust definitions.
    ///
    /// Standalone AOT wrappers use this mode so generated runtime consumers do
    /// not need the `xlsynth` crate merely to name bridge types.
    pub(crate) fn standalone() -> Self {
        Self {
            mode: RustBridgeMode::Standalone,
            ..Self::new()
        }
    }

    /// Appends raw Rust items immediately before the generated module closes.
    ///
    /// This is used by AOT generation to place `Runner` beside the typed DSLX
    /// definitions for the top function.
    pub(crate) fn with_runner_items(mut self, runner_items: impl Into<String>) -> Self {
        self.runner_items = Some(runner_items.into());
        self
    }

    /// Adds generated items immediately after the module preamble.
    ///
    /// Typed AOT uses this hook for owner-module items discovered before normal
    /// bridge rendering. Callers should pass items that are already valid in
    /// the target module's namespace; inserting cross-module paths here
    /// would make generated code depend on the wrong ownership context.
    pub(crate) fn with_leading_items(
        mut self,
        leading_items: impl IntoIterator<Item = String>,
    ) -> Self {
        self.leading_items = leading_items.into_iter().collect();
        self
    }

    /// Defers concrete parametric struct definitions to a package-level pass.
    ///
    /// Typed DSLX AOT generation uses this when it renders several modules
    /// together and has already collected the concrete specializations that
    /// should be emitted in their defining modules.
    pub(crate) fn with_deferred_parametric_struct_emission(mut self) -> Self {
        self.defer_parametric_struct_emission = true;
        self
    }

    /// Finishes the builder and returns Rust source for the generated module
    /// tree.
    ///
    /// Callers should include the returned source from a build output file.
    /// Reusing a builder after `build` is allowed, but doing so observes the
    /// same accumulated lines and epilogue.
    pub fn build(&self) -> String {
        render_rust_module_fragments([self.module_fragment()])
    }

    /// Returns this builder's current module body as a renderable fragment.
    ///
    /// AOT generation uses this to collect bridge modules and the top module
    /// before rendering a single nested module tree.
    pub(crate) fn module_fragment(&self) -> RustModuleFragment {
        RustModuleFragment {
            path: self.module_path.clone(),
            body: self.lines.join("\n"),
        }
    }

    /// Converts a DSLX type and optional source annotation into the generated
    /// Rust type name used by this bridge.
    ///
    /// # Errors
    ///
    /// Returns an error when the concrete DSLX type cannot be represented by
    /// the current Rust bridge surface.
    pub fn rust_type_name(
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        Self::convert_type_with_annotation(RustBridgeMode::Xlsynth, &[], None, type_annotation, ty)
    }

    /// Returns the runtime-neutral Rust type name for a DSLX type in one
    /// module.
    ///
    /// Standalone callers must use this path so imported type references stay
    /// canonical while bits-like values map to `UBits` and `SBits` instead of
    /// `xlsynth` runtime wrappers.
    pub(crate) fn standalone_rust_type_name_from_dslx_module(
        current_module_name: &str,
        type_info: &dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        let module_path = rust_module_path_from_dslx_module_name(current_module_name);
        Self::convert_type_with_annotation(
            RustBridgeMode::Standalone,
            &module_path,
            Some(type_info),
            type_annotation,
            ty,
        )
    }

    fn convert_type_with_annotation(
        mode: RustBridgeMode,
        current_module_path: &[String],
        type_info: Option<&dslx::TypeInfo>,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        if let Some(type_annotation) = type_annotation {
            if let Some(array_annotation) = type_annotation.to_array_type_annotation() {
                if ty.is_array() {
                    let element_annotation = array_annotation.get_element_type();
                    let element_ty = ty.get_array_element_type();
                    let rust_ty = Self::convert_type_with_annotation(
                        mode,
                        current_module_path,
                        type_info,
                        Some(&element_annotation),
                        &element_ty,
                    )?;
                    return Ok(format!("[{rust_ty}; {}]", ty.get_array_size()));
                }
            }
            if let Some(type_ref_annotation) = type_annotation.to_type_ref_type_annotation() {
                if let Some(rust_ty) = Self::convert_type_ref_annotation(
                    current_module_path,
                    type_info,
                    &type_ref_annotation,
                    ty,
                )? {
                    return Ok(rust_ty);
                }
            }
        }
        if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            Ok(match mode {
                RustBridgeMode::Xlsynth => {
                    let signed_str = if is_signed { "S" } else { "U" };
                    format!("Ir{signed_str}Bits<{bit_count}>")
                }
                RustBridgeMode::Standalone => standalone_bits_rust_type(is_signed, bit_count),
            })
        } else if ty.is_enum() {
            let enum_def = ty.get_enum_def()?;
            Ok(enum_def.get_identifier().to_string())
        } else if ty.is_struct() {
            let struct_def = ty.get_struct_def()?;
            Ok(struct_def.get_identifier().to_string())
        } else if ty.is_array() {
            let array_ty = ty.get_array_element_type();
            let array_size = ty.get_array_size();
            let rust_ty = Self::convert_type_with_annotation(
                mode,
                current_module_path,
                type_info,
                None,
                &array_ty,
            )?;
            Ok(format!("[{rust_ty}; {array_size}]"))
        } else {
            Err(XlsynthError(format!(
                "Unsupported type for conversion from DSLX to Rust: {:?}",
                ty.to_string()?
            )))
        }
    }

    /// Converts a typed DSLX type-reference annotation into a Rust bridge path.
    ///
    /// Non-reference annotations return `None` so callers can continue with
    /// concrete structural conversion.
    fn convert_type_ref_annotation(
        current_module_path: &[String],
        type_info: Option<&dslx::TypeInfo>,
        type_ref_annotation: &dslx::TypeRefTypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<Option<String>, XlsynthError> {
        let type_ref = type_ref_annotation.get_type_ref();
        let type_definition = type_ref.get_type_definition();
        let suffix = Self::parametric_type_suffix(type_info, type_ref_annotation, ty)?;
        if let Some(colon_ref) = type_definition.to_colon_ref() {
            let attr = format!("{}{}", colon_ref.get_attr(), suffix);
            if let Some(import) = colon_ref.resolve_import_subject() {
                let module_path = rust_module_path_from_import(&import);
                Ok(Some(rust_type_path_between_module_paths(
                    current_module_path,
                    &module_path,
                    &attr,
                )))
            } else {
                Ok(Some(attr))
            }
        } else {
            Ok(type_definition
                .to_type_alias()
                .map(|alias| alias.get_identifier())
                .or_else(|| {
                    if ty.is_struct() {
                        ty.get_struct_def()
                            .ok()
                            .map(|struct_def| format!("{}{}", struct_def.get_identifier(), suffix))
                    } else if ty.is_enum() {
                        ty.get_enum_def()
                            .ok()
                            .map(|enum_def| enum_def.get_identifier())
                    } else {
                        None
                    }
                }))
        }
    }

    fn parametric_type_suffix(
        type_info: Option<&dslx::TypeInfo>,
        type_ref_annotation: &dslx::TypeRefTypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        let parametric_count = type_ref_annotation.get_parametric_count();
        if parametric_count == 0 {
            return Ok(String::new());
        }
        let type_info = type_info.ok_or_else(|| {
            XlsynthError(
                "DSLX Rust bridge cannot name parametric type without a TypeInfo context"
                    .to_string(),
            )
        })?;
        if !ty.is_struct() {
            return Err(XlsynthError(format!(
                "DSLX Rust bridge only supports parametric struct type references, got `{}`",
                ty.to_string()?
            )));
        }
        let struct_def = ty.get_struct_def()?;
        let binding_count = struct_def.get_parametric_binding_count();
        if binding_count != parametric_count {
            return Err(XlsynthError(format!(
                "DSLX Rust bridge parametric mismatch for `{}`: struct has {binding_count} binding(s), annotation has {parametric_count} argument(s)",
                struct_def.get_identifier()
            )));
        }
        let mut parts = Vec::with_capacity(parametric_count);
        for index in 0..parametric_count {
            let binding = struct_def.get_parametric_binding(index);
            let expr = type_ref_annotation.get_parametric_expr(index).ok_or_else(|| {
                XlsynthError(format!(
                    "DSLX Rust bridge does not support type-valued parametric argument {index} for `{}`",
                    struct_def.get_identifier()
                ))
            })?;
            let is_signed = typed_literal_text_is_signed(&expr.to_text());
            let value = const_parametric_expr_value(type_info, &expr)?;
            parts.push(format!(
                "{}_{}",
                sanitize_type_parametric_segment(&binding.get_identifier()),
                rust_type_parametric_value_suffix(&value, is_signed)?
            ));
        }
        Ok(format!("__{}", parts.join("__")))
    }

    fn emit_concrete_parametric_types_for_type(
        &mut self,
        type_info: &dslx::TypeInfo,
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        if let Some(type_annotation) = type_annotation {
            if let Some(array_annotation) = type_annotation.to_array_type_annotation() {
                if ty.is_array() {
                    let element_annotation = array_annotation.get_element_type();
                    let element_ty = ty.get_array_element_type();
                    self.emit_concrete_parametric_types_for_type(
                        type_info,
                        Some(&element_annotation),
                        &element_ty,
                    )?;
                    return Ok(());
                }
            }
            if let Some(type_ref_annotation) = type_annotation.to_type_ref_type_annotation() {
                self.emit_concrete_parametric_struct_for_type_ref(
                    type_info,
                    &type_ref_annotation,
                    ty,
                )?;
            }
        }
        if ty.is_array() {
            let element_ty = ty.get_array_element_type();
            self.emit_concrete_parametric_types_for_type(type_info, None, &element_ty)?;
        }
        Ok(())
    }

    fn emit_concrete_parametric_struct_for_type_ref(
        &mut self,
        type_info: &dslx::TypeInfo,
        type_ref_annotation: &dslx::TypeRefTypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        if type_ref_annotation.get_parametric_count() == 0 {
            return Ok(());
        }
        if self.defer_parametric_struct_emission {
            return Ok(());
        }
        let Some(rust_ty) = Self::convert_type_ref_annotation(
            &self.module_path,
            Some(type_info),
            type_ref_annotation,
            ty,
        )?
        else {
            return Ok(());
        };
        if rust_ty.contains("::") {
            return Err(XlsynthError(format!(
                "DSLX Rust bridge does not support direct imported parametric struct instantiation `{rust_ty}`; define and use a concrete type alias in the imported module"
            )));
        }
        if !self.emitted_parametric_structs.insert(rust_ty.clone()) {
            return Ok(());
        }
        let struct_def = ty.get_struct_def()?;
        let concrete_member_count = ty.get_struct_member_count();
        let definition_member_count = struct_def.get_member_count();
        if concrete_member_count != definition_member_count {
            return Err(XlsynthError(format!(
                "DSLX Rust bridge parametric struct `{}` has {definition_member_count} definition member(s) but {concrete_member_count} concrete member type(s)",
                struct_def.get_identifier()
            )));
        }
        let mut field_lines = Vec::with_capacity(concrete_member_count);
        for index in 0..concrete_member_count {
            let member = struct_def.get_member(index);
            let field_annotation = member.get_type();
            let field_ty = ty.get_struct_member_type(index);
            self.emit_concrete_parametric_types_for_type(
                type_info,
                Some(&field_annotation),
                &field_ty,
            )?;
            let field_rust_ty = Self::convert_type_with_annotation(
                self.mode,
                &self.module_path,
                Some(type_info),
                Some(&field_annotation),
                &field_ty,
            )?;
            field_lines.push(format!("    pub {}: {},", member.get_name(), field_rust_ty));
        }
        self.lines
            .push("#[allow(non_camel_case_types)]".to_string());
        self.lines
            .push("#[derive(Debug, Clone, PartialEq, Eq)]".to_string());
        self.lines.push(format!("pub struct {rust_ty} {{"));
        self.lines.extend(field_lines);
        self.lines.push("}\n".to_string());
        Ok(())
    }
}

/// Returns the smallest runtime-neutral Rust type that can hold one DSLX bits
/// value.
fn standalone_bits_rust_type(is_signed: bool, bit_count: usize) -> String {
    if is_signed {
        format!("SBits<{bit_count}>")
    } else {
        format!("UBits<{bit_count}>")
    }
}

fn const_parametric_expr_value(
    type_info: &dslx::TypeInfo,
    expr: &dslx::Expr,
) -> Result<IrValue, XlsynthError> {
    if let Ok(value) =
        dslx::InterpValue::from_string(&expr.to_text()).and_then(|value| value.convert_to_ir())
    {
        return Ok(value);
    }
    type_info.get_const_expr(expr)?.convert_to_ir()
}

fn typed_literal_text_is_signed(text: &str) -> bool {
    let text = text.trim();
    text.starts_with('s') || text.starts_with("-")
}

impl Default for RustBridgeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BridgeBuilder for RustBridgeBuilder {
    fn start_module(&mut self, module_name: &str) -> Result<(), XlsynthError> {
        self.module_path = rust_module_path_from_dslx_module_name(module_name);
        self.emitted_parametric_structs.clear();
        let imports = match self.mode {
            RustBridgeMode::Xlsynth => "use xlsynth::{IrValue, IrUBits, IrSBits};\n".to_string(),
            RustBridgeMode::Standalone => {
                let root = std::iter::repeat_n("super", self.module_path.len())
                    .collect::<Vec<_>>()
                    .join("::");
                if root.is_empty() {
                    String::new()
                } else {
                    format!(
                        "use {root}::{{read_leaf_element, write_leaf_element, AotArtifactMetadata, AotElementLayout, AotError, AotRunResult, AotRunnerLayout, SBits, StandaloneRunner, UBits}};\n"
                    )
                }
            }
        };
        self.lines = vec![
            // We allow e.g. enum variants to be unused in consumer code.
            "#![allow(dead_code)]".to_string(),
            "#![allow(unused_imports)]".to_string(),
            imports,
        ];
        self.lines.extend(self.leading_items.clone());
        Ok(())
    }

    fn end_module(&mut self, module_name: &str) -> Result<(), XlsynthError> {
        let _ = module_name;
        if let Some(runner_items) = &self.runner_items {
            self.lines.push(runner_items.clone());
        }
        Ok(())
    }

    fn add_enum_def(
        &mut self,
        dslx_name: &str,
        is_signed: bool,
        _underlying_bit_count: usize,
        members: &[(String, IrValue)],
    ) -> Result<(), XlsynthError> {
        let value_to_string = |value: &IrValue| -> Result<String, XlsynthError> {
            if is_signed {
                value.to_i64().map(|v| v.to_string())
            } else {
                value.to_u64().map(|v| v.to_string())
            }
        };

        self.lines
            .push("#[derive(Debug, Clone, Copy, PartialEq, Eq)]".to_string());
        self.lines.push(format!("pub enum {dslx_name} {{"));
        for (name, value) in members.iter() {
            self.lines
                .push(format!("    {} = {},", name, value_to_string(value)?));
        }
        self.lines.push("}\n".to_string());

        if self.mode == RustBridgeMode::Xlsynth {
            // Emit the converter used by non-standalone bridge consumers that
            // pass generated Rust enums back into interpreter APIs.
            self.lines
                .push(format!("impl From<{dslx_name}> for IrValue {{"));
            self.lines
                .push(format!("    fn from(value: {dslx_name}) -> Self {{"));
            self.lines.push("        match value {".to_string());
            for (member_name, value) in members.iter() {
                let value_str = value_to_string(value)?;
                self.lines.push(format!(
                    "            {}::{} => IrValue::make_{}bits({}, {}).unwrap(),",
                    dslx_name,
                    member_name,
                    if is_signed { "s" } else { "u" },
                    value.bit_count()?,
                    value_str
                ));
            }
            self.lines.push("        }".to_string());
            self.lines.push("    }".to_string());
            self.lines.push("}\n".to_string());
        }
        Ok(())
    }

    fn add_struct_def(
        &mut self,
        dslx_name: &str,
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError> {
        self.lines
            .push("#[derive(Debug, Clone, PartialEq, Eq)]".to_string());
        self.lines.push(format!("pub struct {dslx_name} {{"));
        for member in members {
            let rust_ty = Self::convert_type_with_annotation(
                self.mode,
                &self.module_path,
                None,
                Some(&member.type_annotation),
                &member.concrete_type,
            )?;
            self.lines
                .push(format!("    pub {}: {},", member.name, rust_ty));
        }
        self.lines.push("}\n".to_string());
        Ok(())
    }

    fn add_struct_def_typed(
        &mut self,
        dslx_name: &str,
        type_info: &dslx::TypeInfo,
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError> {
        for member in members {
            self.emit_concrete_parametric_types_for_type(
                type_info,
                Some(&member.type_annotation),
                &member.concrete_type,
            )?;
        }
        self.lines
            .push("#[derive(Debug, Clone, PartialEq, Eq)]".to_string());
        self.lines.push(format!("pub struct {dslx_name} {{"));
        for member in members {
            let rust_ty = Self::convert_type_with_annotation(
                self.mode,
                &self.module_path,
                Some(type_info),
                Some(&member.type_annotation),
                &member.concrete_type,
            )?;
            self.lines
                .push(format!("    pub {}: {},", member.name, rust_ty));
        }
        self.lines.push("}\n".to_string());
        Ok(())
    }

    fn add_alias(
        &mut self,
        dslx_name: &str,
        type_annotation: &dslx::TypeAnnotation,
        concrete_type: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        let rust_ty = Self::convert_type_with_annotation(
            self.mode,
            &self.module_path,
            None,
            Some(type_annotation),
            concrete_type,
        )?;
        self.lines
            .push(format!("pub type {dslx_name} = {rust_ty};\n"));
        Ok(())
    }

    fn add_alias_typed(
        &mut self,
        dslx_name: &str,
        type_info: &dslx::TypeInfo,
        type_annotation: &dslx::TypeAnnotation,
        concrete_type: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        self.emit_concrete_parametric_types_for_type(
            type_info,
            Some(type_annotation),
            concrete_type,
        )?;
        let rust_ty = Self::convert_type_with_annotation(
            self.mode,
            &self.module_path,
            Some(type_info),
            Some(type_annotation),
            concrete_type,
        )?;
        self.lines
            .push(format!("pub type {dslx_name} = {rust_ty};\n"));
        Ok(())
    }

    fn add_function_signature_typed(
        &mut self,
        _dslx_name: &str,
        type_info: &dslx::TypeInfo,
        params: &[FunctionParamData],
        return_type_annotation: Option<&dslx::TypeAnnotation>,
        return_type: Option<&dslx::Type>,
    ) -> Result<(), XlsynthError> {
        for param in params {
            if let Some(concrete_type) = &param.concrete_type {
                self.emit_concrete_parametric_types_for_type(
                    type_info,
                    Some(&param.type_annotation),
                    concrete_type,
                )?;
            }
        }
        if let Some(return_type) = return_type {
            self.emit_concrete_parametric_types_for_type(
                type_info,
                return_type_annotation,
                return_type,
            )?;
        }
        Ok(())
    }

    fn add_constant(
        &mut self,
        _name: &str,
        _constant_def: &dslx::ConstantDef,
        _ty: &dslx::Type,
        _ir_value: &IrValue,
    ) -> Result<(), XlsynthError> {
        Ok(())
    }
}

/// Converts a dotted DSLX module name into sanitized Rust module path segments.
pub(crate) fn rust_module_path_from_dslx_module_name(module_name: &str) -> Vec<String> {
    module_name
        .split('.')
        .filter(|segment| !segment.is_empty())
        .map(sanitize_module_segment)
        .collect()
}

/// Converts a parsed DSLX import subject into sanitized Rust module segments.
fn rust_module_path_from_import(import: &dslx::Import) -> Vec<String> {
    import
        .get_subject()
        .iter()
        .map(|segment| sanitize_module_segment(segment))
        .collect()
}

/// Builds the Rust path from one DSLX module to a type declared in another.
///
/// The result uses `super` segments when needed, so callers should pass DSLX
/// module names rather than already-qualified Rust paths.
pub(crate) fn rust_type_path_between_dslx_modules(
    current_module_name: &str,
    target_module_name: &str,
    type_name: &str,
) -> String {
    let current_path = rust_module_path_from_dslx_module_name(current_module_name);
    let target_path = rust_module_path_from_dslx_module_name(target_module_name);
    rust_type_path_between_module_paths(&current_path, &target_path, type_name)
}

/// Builds the Rust path from one generated module path to a target type path.
///
/// If both paths are the same module, only `type_name` is returned. Otherwise
/// the result walks back to the generated root with `super` and then descends
/// into the target module path.
pub(crate) fn rust_type_path_between_module_paths(
    current_module_path: &[String],
    target_module_path: &[String],
    type_name: &str,
) -> String {
    if current_module_path == target_module_path {
        type_name.to_string()
    } else {
        std::iter::repeat_n("super".to_string(), current_module_path.len())
            .chain(target_module_path.iter().cloned())
            .chain([type_name.to_string()])
            .collect::<Vec<_>>()
            .join("::")
    }
}

/// Rewrites a DSLX module path segment into a Rust module identifier segment.
fn sanitize_module_segment(segment: &str) -> String {
    segment
        .chars()
        .map(|ch| {
            if ch == '_' || ch.is_ascii_alphanumeric() {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn sanitize_type_parametric_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for (index, ch) in segment.chars().enumerate() {
        let valid = ch == '_' || ch.is_ascii_alphanumeric();
        let ch = if valid { ch } else { '_' };
        if index == 0 && ch.is_ascii_digit() {
            out.push('_');
        }
        out.push(ch);
    }
    if out.is_empty() {
        "P".to_string()
    } else {
        out
    }
}

fn rust_type_parametric_value_suffix(
    value: &IrValue,
    is_signed: bool,
) -> Result<String, XlsynthError> {
    let format = if is_signed {
        IrFormatPreference::SignedDecimal
    } else {
        IrFormatPreference::UnsignedDecimal
    };
    let value = value.to_string_fmt_no_prefix(format)?;
    Ok(sanitize_type_parametric_value_segment(&value))
}

fn sanitize_type_parametric_value_segment(value: &str) -> String {
    let value = value.trim();
    if let Some(value) = value.strip_prefix('-') {
        format!("m{}", sanitize_type_parametric_value_atom(value))
    } else {
        sanitize_type_parametric_value_atom(value)
    }
}

fn sanitize_type_parametric_value_atom(value: &str) -> String {
    let out = value
        .chars()
        .map(|ch| {
            if ch == '_' || ch.is_ascii_alphanumeric() {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    if out.is_empty() {
        "P".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::dslx_bridge::{convert_imported_module, convert_leaf_module};

    use super::*;

    // Verifies: enum-only DSLX modules emit Rust enums with IR conversions.
    // Catches: regressions in enum discriminant or conversion rendering.
    #[test]
    fn test_convert_leaf_module_enum_def_only() {
        let dslx = r#"
        enum MyEnum : u2 { A = 0, B = 3 }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"pub mod my_module {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MyEnum {
    A = 0,
    B = 3,
}

impl From<MyEnum> for IrValue {
    fn from(value: MyEnum) -> Self {
        match value {
            MyEnum::A => IrValue::make_ubits(2, 0).unwrap(),
            MyEnum::B => IrValue::make_ubits(2, 3).unwrap(),
        }
    }
}

}"#
        );
    }

    // Verifies: struct-only DSLX modules emit typed Rust fields.
    // Catches: regressions that drop struct fields or signedness wrappers.
    #[test]
    fn test_convert_leaf_module_struct_def_only() {
        let dslx = r#"
        struct MyStruct {
            a: u32,
            b: s16,
        }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"pub mod my_module {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyStruct {
    pub a: IrUBits<32>,
    pub b: IrSBits<16>,
}

}"#
        );
    }

    // Verifies: struct fields can reference enums emitted in the same module.
    // Catches: regressions that lower enum fields to structural bits.
    #[test]
    fn test_convert_leaf_module_struct_with_enum_field() {
        let dslx = r#"
        enum MyEnum : u2 { A = 0, B = 3 }
        struct MyStruct {
            a: MyEnum,
            b: s16,
        }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"pub mod my_module {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MyEnum {
    A = 0,
    B = 3,
}

impl From<MyEnum> for IrValue {
    fn from(value: MyEnum) -> Self {
        match value {
            MyEnum::A => IrValue::make_ubits(2, 0).unwrap(),
            MyEnum::B => IrValue::make_ubits(2, 3).unwrap(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyStruct {
    pub a: MyEnum,
    pub b: IrSBits<16>,
}

}"#
        );
    }

    // Verifies: nested DSLX structs preserve generated Rust struct references.
    // Catches: regressions that flatten nested structs into tuple fields.
    #[test]
    fn test_convert_leaf_module_nested_struct() {
        let dslx = r#"
        struct MyInnerStruct {
            x: u8,
            y: u8,
        }
        struct MyStruct {
            a: u32,
            b: s16,
            c: MyInnerStruct,
        }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"pub mod my_module {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyInnerStruct {
    pub x: IrUBits<8>,
    pub y: IrUBits<8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyStruct {
    pub a: IrUBits<32>,
    pub b: IrSBits<16>,
    pub c: MyInnerStruct,
}

}"#
        );
    }

    // Verifies: DSLX array members render as Rust arrays of bridge types.
    // Catches: regressions in array element type conversion.
    #[test]
    fn test_convert_leaf_module_struct_with_array() {
        let dslx = r#"
        struct MyStruct {
            a: u32,
            b: s16,
            c: u8[4],
        }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"pub mod my_module {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyStruct {
    pub a: IrUBits<32>,
    pub b: IrSBits<16>,
    pub c: [IrUBits<8>; 4],
}

}"#
        );
    }

    // Verifies: DSLX aliases to bits emit Rust type aliases.
    // Catches: regressions that replace aliases with generated structs.
    #[test]
    fn test_convert_leaf_module_type_alias_to_concrete_bits_alias_only() {
        let dslx = "type MyType = u8;";
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"pub mod my_module {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

pub type MyType = IrUBits<8>;

}"#
        );
    }

    // Verifies: the default bridge ignores function signature metadata.
    // Catches: regressions that emit unwanted boundary aliases.
    #[test]
    fn test_function_signatures_do_not_emit_aliases() {
        let dslx = r#"
        type Count = u8;
        struct ResultStruct { value: Count }
        pub fn do_frob(count: Count) -> ResultStruct {
            ResultStruct { value: count }
        }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut default_builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut default_builder).unwrap();
        let output = default_builder.build();
        assert!(!output.contains("DoFrobCount"));
        assert!(!output.contains("DoFrobReturn"));
    }

    // Verifies: imported type references render with canonical sibling paths.
    // Catches: bare imported names that fail outside the defining module.
    #[test]
    fn test_struct_with_extern_type_ref_member() {
        let imported_dslx = "pub struct MyImportedStruct { a: u8 }";
        let importer_dslx = "import imported; struct MyStruct { a: imported::MyImportedStruct }";

        let mut import_data = dslx::ImportData::default();
        let _imported_typechecked =
            dslx::parse_and_typecheck(imported_dslx, "imported.x", "imported", &mut import_data)
                .unwrap();
        let importer_typechecked =
            dslx::parse_and_typecheck(importer_dslx, "importer.x", "importer", &mut import_data)
                .unwrap();

        let mut builder = RustBridgeBuilder::new();
        convert_imported_module(&importer_typechecked, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            "pub mod importer {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyStruct {
    pub a: super::imported::MyImportedStruct,
}

}"
        );
    }

    // Verifies: arrays of imported type references keep canonical paths.
    // Catches: array lowering that loses the imported element qualification.
    #[test]
    fn test_struct_with_extern_type_ref_array_member() {
        let imported_dslx = "pub struct MyImportedStruct { a: u8 }";
        let importer_dslx = "import imported; struct MyStruct { a: imported::MyImportedStruct[2] }";

        let mut import_data = dslx::ImportData::default();
        let _imported_typechecked =
            dslx::parse_and_typecheck(imported_dslx, "imported.x", "imported", &mut import_data)
                .unwrap();
        let importer_typechecked =
            dslx::parse_and_typecheck(importer_dslx, "importer.x", "importer", &mut import_data)
                .unwrap();

        let mut builder = RustBridgeBuilder::new();
        convert_imported_module(&importer_typechecked, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            "pub mod importer {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyStruct {
    pub a: [super::imported::MyImportedStruct; 2],
}

}"
        );
    }

    // Verifies: dotted import aliases render through the generated module tree.
    // Catches: alias handling that drops intermediate module segments.
    #[test]
    fn test_struct_with_dotted_import_alias_type_ref_member() {
        let imported_dslx = "pub struct MyImportedStruct { a: u8 }";
        let importer_dslx =
            "import common.logic.imported as ext; struct MyStruct { a: ext::MyImportedStruct }";

        let tmpdir = xlsynth_test_helpers::make_test_tmpdir("xlsynth_rust_bridge_builder_test");
        let common_logic_dir = tmpdir.path().join("common/logic");
        std::fs::create_dir_all(&common_logic_dir).unwrap();
        let imported_path = common_logic_dir.join("imported.x");
        std::fs::write(&imported_path, imported_dslx).unwrap();
        let importer_path = tmpdir.path().join("importer.x");
        std::fs::write(&importer_path, importer_dslx).unwrap();

        let mut import_data = dslx::ImportData::new(None, &[tmpdir.path()]);
        let importer_typechecked = dslx::parse_and_typecheck(
            importer_dslx,
            importer_path.to_str().unwrap(),
            "importer",
            &mut import_data,
        )
        .unwrap();

        let mut builder = RustBridgeBuilder::new();
        convert_imported_module(&importer_typechecked, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            "pub mod importer {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyStruct {
    pub a: super::common::logic::imported::MyImportedStruct,
}

}"
        );
    }

    // Verifies: arrays preserve dotted import aliases on their element type.
    // Catches: array parsing that strips aliased module qualification.
    #[test]
    fn test_struct_with_dotted_import_alias_type_ref_array_member() {
        let imported_dslx = "pub struct MyImportedStruct { a: u8 }";
        let importer_dslx =
            "import common.logic.imported as ext; struct MyStruct { a: ext::MyImportedStruct[2] }";

        let tmpdir = xlsynth_test_helpers::make_test_tmpdir("xlsynth_rust_bridge_builder_test");
        let common_logic_dir = tmpdir.path().join("common/logic");
        std::fs::create_dir_all(&common_logic_dir).unwrap();
        let imported_path = common_logic_dir.join("imported.x");
        std::fs::write(&imported_path, imported_dslx).unwrap();
        let importer_path = tmpdir.path().join("importer.x");
        std::fs::write(&importer_path, importer_dslx).unwrap();

        let mut import_data = dslx::ImportData::new(None, &[tmpdir.path()]);
        let importer_typechecked = dslx::parse_and_typecheck(
            importer_dslx,
            importer_path.to_str().unwrap(),
            "importer",
            &mut import_data,
        )
        .unwrap();

        let mut builder = RustBridgeBuilder::new();
        convert_imported_module(&importer_typechecked, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            "pub mod importer {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MyStruct {
    pub a: [super::common::logic::imported::MyImportedStruct; 2],
}

}"
        );
    }

    // Verifies: parametric struct aliases emit concrete Rust structs with
    // suffixes based on evaluated parameter values.
    // Catches: accidental fallback to unspecialized field annotations.
    #[test]
    fn test_parametric_struct_alias_emits_concrete_struct() {
        let dslx = r#"
        struct Box<N: u32> {
            value: bits[N],
        }
        type Box8 = Box<u32:8>;
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"pub mod my_module {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Box__N_8 {
    pub value: IrUBits<8>,
}

pub type Box8 = Box__N_8;

}"#
        );
    }

    // Verifies: direct parametric struct references in function signatures
    // still emit the concrete Rust struct even without a DSLX alias.
    // Catches: missing lazy emission from signature-only references.
    #[test]
    fn test_parametric_struct_function_signature_emits_concrete_struct() {
        let dslx = r#"
        struct Box<N: u32> {
            value: bits[N],
        }
        pub fn echo_box(x: Box<u32:8>) -> Box<u32:8> {
            x
        }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"pub mod my_module {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Box__N_8 {
    pub value: IrUBits<8>,
}

}"#
        );
    }

    // Verifies: direct imported parametric instantiations fail before emitting
    // Rust that references a concrete type the imported module may not emit.
    #[test]
    fn test_direct_imported_parametric_struct_reference_errors() {
        let imported_dslx = r#"
        pub struct RemoteBox<N: u32> {
            value: bits[N],
        }
        "#;
        let importer_dslx = r#"
        import imported;
        pub fn echo_box(x: imported::RemoteBox<u32:8>) -> imported::RemoteBox<u32:8> {
            x
        }
        "#;

        let mut import_data = dslx::ImportData::default();
        let _imported_typechecked =
            dslx::parse_and_typecheck(imported_dslx, "imported.x", "imported", &mut import_data)
                .unwrap();
        let importer_typechecked =
            dslx::parse_and_typecheck(importer_dslx, "importer.x", "importer", &mut import_data)
                .unwrap();

        let mut builder = RustBridgeBuilder::new();
        let error = convert_imported_module(&importer_typechecked, &mut builder).unwrap_err();
        assert!(error
            .to_string()
            .contains("direct imported parametric struct instantiation"));
        assert!(error.to_string().contains("concrete type alias"));
    }
}
