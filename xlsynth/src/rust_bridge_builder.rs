// SPDX-License-Identifier: Apache-2.0

//! Builder that creates Rust type definitions from DSLX type definitions.
//!
//! This helps us e.g. call DSLX functions from Rust code, i.e. it enables
//! Rust->DSLX FFI interop.

use std::collections::BTreeMap;

use crate::{
    dslx,
    dslx_bridge::{BridgeBuilder, StructMemberData},
    IrValue, XlsynthError,
};

mod annotation_spelling_fallback;

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
    import_aliases: BTreeMap<String, String>,
}

/// Rust items generated for one DSLX module, before parent modules are
/// rendered.
///
/// Fragments let AOT generation collect multiple DSLX modules first and then
/// render one shared tree, which avoids each module independently creating its
/// own top-level namespace.
#[derive(Debug, Clone)]
pub(crate) struct RustModuleFragment {
    path: Vec<String>,
    body: String,
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
            import_aliases: BTreeMap::new(),
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
        Self::convert_type_with_annotation(&[], type_annotation, ty)
    }

    /// Resolves a DSLX type-reference annotation into a Rust path from a
    /// module.
    ///
    /// This is narrower than `rust_type_name`: it only succeeds for annotations
    /// that name an alias or imported type reference. Use it when the caller
    /// needs to preserve a source-level qualified type instead of falling back
    /// to the concrete structural type.
    pub(crate) fn rust_type_ref_name_from_dslx_module(
        current_module_name: &str,
        type_annotation: &dslx::TypeAnnotation,
    ) -> Option<String> {
        let module_path = rust_module_path_from_dslx_module_name(current_module_name);
        Self::convert_type_ref_annotation(&module_path, type_annotation)
    }

    fn convert_type_with_annotation(
        current_module_path: &[String],
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        Self::convert_type_with_annotation_and_text(
            current_module_path,
            type_annotation,
            None,
            &BTreeMap::new(),
            ty,
        )
    }

    /// Converts a DSLX type while preserving source-spelled type references.
    ///
    /// The typed annotation is preferred when it directly identifies an alias
    /// or imported DSLX type. Source text is consulted only for nested
    /// annotations the current FFI cannot walk, such as array elements.
    fn convert_type_with_annotation_and_text(
        current_module_path: &[String],
        type_annotation: Option<&dslx::TypeAnnotation>,
        annotation_text: Option<&str>,
        import_aliases: &BTreeMap<String, String>,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        if let Some(type_annotation) = type_annotation {
            if let Some(rust_ty) =
                Self::convert_type_ref_annotation(current_module_path, type_annotation)
            {
                return Ok(rust_ty);
            }
        }
        // TODO(xlsynth-ffi): Remove this source-spelling fallback once the DSLX
        // AST/type-annotation bindings expose enough structure to recursively
        // recover imported type references from nested annotations. The typed
        // path above handles a direct `TypeRefTypeAnnotation`, but it cannot
        // currently inspect `ArrayTypeAnnotation` and then continue into the
        // element annotation. This branch exists only to preserve the source
        // spelling for cases such as `imported::Widget[2]` and
        // `alias::Widget` while the FFI is missing those nested annotation
        // accessors. Resolving this TODO should delete this branch and the
        // private `annotation_spelling_fallback` module below, leaving no ad
        // hoc parsing introduced by this PR.
        if let Some(rust_ty) = annotation_text.and_then(|text| {
            annotation_spelling_fallback::rust_type_ref_from_annotation_text(
                current_module_path,
                text,
                import_aliases,
            )
        }) {
            Ok(rust_ty)
        } else if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            let signed_str = if is_signed { "S" } else { "U" };
            Ok(format!("Ir{signed_str}Bits<{bit_count}>"))
        } else if ty.is_enum() {
            let enum_def = ty.get_enum_def()?;
            Ok(enum_def.get_identifier().to_string())
        } else if ty.is_struct() {
            let struct_def = ty.get_struct_def()?;
            Ok(struct_def.get_identifier().to_string())
        } else if ty.is_array() {
            let array_ty = ty.get_array_element_type();
            let array_size = ty.get_array_size();
            // TODO(xlsynth-ffi): Use the array type annotation's typed element
            // annotation here instead of slicing the source spelling. The FFI
            // should expose the DSLX `ArrayTypeAnnotation` node and an accessor
            // for its element `TypeAnnotation`, so this recursive call can pass
            // `Some(&element_type_annotation)` and no text fallback is needed.
            let element_annotation_text =
                annotation_text.and_then(annotation_spelling_fallback::array_element_annotation);
            let rust_ty = Self::convert_type_with_annotation_and_text(
                current_module_path,
                None,
                element_annotation_text,
                import_aliases,
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
        type_annotation: &dslx::TypeAnnotation,
    ) -> Option<String> {
        let type_ref = type_annotation
            .to_type_ref_type_annotation()?
            .get_type_ref();
        let type_definition = type_ref.get_type_definition();
        if let Some(colon_ref) = type_definition.to_colon_ref() {
            let attr = colon_ref.get_attr();
            if let Some(import) = colon_ref.resolve_import_subject() {
                let module_path = rust_module_path_from_import(&import);
                Some(rust_type_path_between_module_paths(
                    current_module_path,
                    &module_path,
                    &attr,
                ))
            } else {
                Some(attr)
            }
        } else {
            type_definition
                .to_type_alias()
                .map(|alias| alias.get_identifier())
        }
    }
}

impl Default for RustBridgeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BridgeBuilder for RustBridgeBuilder {
    fn start_module(&mut self, module_name: &str) -> Result<(), XlsynthError> {
        self.module_path = rust_module_path_from_dslx_module_name(module_name);
        self.import_aliases = BTreeMap::new();
        self.lines = vec![
            // We allow e.g. enum variants to be unused in consumer code.
            "#![allow(dead_code)]".to_string(),
            "#![allow(unused_imports)]".to_string(),
            "use xlsynth::{IrValue, IrUBits, IrSBits};\n".to_string(),
        ];
        Ok(())
    }

    fn start_module_with_text(
        &mut self,
        module_name: &str,
        module_text: &str,
    ) -> Result<(), XlsynthError> {
        self.start_module(module_name)?;
        // TODO(xlsynth-ffi): Replace this import-alias source scan with typed
        // DSLX module import accessors. The FFI should expose each import
        // statement's subject and optional alias from the parsed module, so the
        // bridge can build `alias -> canonical module path` without splitting
        // module text on semicolons or recognizing `import ... as ...` by hand.
        // Once that exists, remove this fallback and the entire
        // `annotation_spelling_fallback` module.
        self.import_aliases = annotation_spelling_fallback::import_aliases_from_module_text(
            module_text,
            sanitize_module_segment,
        );
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

        // Now we emit the converter so we can easily pass our generated Rust enum to IR
        // interpreter functions.
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
        Ok(())
    }

    fn add_struct_def(
        &mut self,
        dslx_name: &str,
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError> {
        self.add_struct_def_with_text(dslx_name, "", members)
    }

    fn add_struct_def_with_text(
        &mut self,
        dslx_name: &str,
        struct_text: &str,
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError> {
        // TODO(xlsynth-ffi): Stop extracting struct member annotations from
        // `StructDef::to_text()`. `StructMember::get_type()` already exposes
        // the immediate member annotation, but nested source-qualified
        // references are not fully walkable through the current Rust bindings.
        // The FFI should expose the annotation node variants needed by
        // `convert_type_with_annotation_and_text` below, especially array
        // annotations and their element annotations. After that, this map and
        // all source-text member parsing should be deleted.
        let member_annotations =
            annotation_spelling_fallback::struct_member_annotations_from_text(struct_text);
        self.lines
            .push("#[derive(Debug, Clone, PartialEq, Eq)]".to_string());
        self.lines.push(format!("pub struct {dslx_name} {{"));
        for member in members {
            let rust_ty = Self::convert_type_with_annotation_and_text(
                &self.module_path,
                Some(&member.type_annotation),
                member_annotations.get(&member.name).map(String::as_str),
                &self.import_aliases,
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
        self.add_alias_with_text(dslx_name, "", type_annotation, concrete_type)
    }

    fn add_alias_with_text(
        &mut self,
        dslx_name: &str,
        alias_text: &str,
        type_annotation: &dslx::TypeAnnotation,
        concrete_type: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        let rust_ty = Self::convert_type_with_annotation_and_text(
            &self.module_path,
            Some(type_annotation),
            // TODO(xlsynth-ffi): Use the existing typed alias annotation plus
            // nested annotation FFI accessors instead of re-reading the
            // right-hand side from `TypeAlias::to_text()`. The replacement
            // should recurse through the `TypeAnnotation` tree directly and
            // should make this text fallback unnecessary for aliases to arrays
            // of imported DSLX types.
            annotation_spelling_fallback::type_alias_annotation_from_text(alias_text),
            &self.import_aliases,
            concrete_type,
        )?;
        self.lines
            .push(format!("pub type {dslx_name} = {rust_ty};\n"));
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
}
