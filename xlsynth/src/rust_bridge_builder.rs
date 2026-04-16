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

pub struct RustBridgeBuilder {
    lines: Vec<String>,
    module_path: Vec<String>,
    module_epilogue: Option<String>,
    import_aliases: BTreeMap<String, String>,
}

/// Rust items generated for one DSLX module, before parent modules are
/// rendered.
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

    fn render_module(&self, module_name: &str) -> String {
        format!(
            "pub mod {module_name} {{\n{}\n}} // pub mod {module_name}",
            self.render_contents()
        )
    }
}

/// Renders generated module fragments into a shared nested Rust module tree.
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
            module_epilogue: None,
            import_aliases: BTreeMap::new(),
        }
    }

    /// Appends raw Rust items immediately before the generated module closes.
    ///
    /// This is used by AOT generation to place `Runner` beside the pretty type
    /// definitions for the DSLX top function.
    pub(crate) fn with_module_epilogue(mut self, module_epilogue: impl Into<String>) -> Self {
        self.module_epilogue = Some(module_epilogue.into());
        self
    }

    pub fn build(&self) -> String {
        render_rust_module_fragments([self.module_fragment()])
    }

    pub(crate) fn module_fragment(&self) -> RustModuleFragment {
        RustModuleFragment {
            path: self.module_path.clone(),
            body: self.lines.join("\n"),
        }
    }

    /// Converts a DSLX type and optional source annotation into the generated
    /// Rust type name used by this bridge.
    pub fn rust_type_name(
        type_annotation: Option<&dslx::TypeAnnotation>,
        ty: &dslx::Type,
    ) -> Result<String, XlsynthError> {
        Self::convert_type_with_annotation(&[], type_annotation, ty)
    }

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
        if let Some(rust_ty) = annotation_text.and_then(|text| {
            rust_type_ref_from_annotation_text(current_module_path, text, import_aliases)
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
            let element_annotation_text = annotation_text.and_then(array_element_annotation_text);
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
        self.import_aliases = import_aliases_from_module_text(module_text);
        Ok(())
    }

    fn end_module(&mut self, module_name: &str) -> Result<(), XlsynthError> {
        let _ = module_name;
        if let Some(module_epilogue) = &self.module_epilogue {
            self.lines.push(module_epilogue.clone());
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
        let member_annotations = struct_member_annotations_from_text(struct_text);
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
        bits_type: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        self.add_alias_with_text(dslx_name, "", type_annotation, bits_type)
    }

    fn add_alias_with_text(
        &mut self,
        dslx_name: &str,
        alias_text: &str,
        type_annotation: &dslx::TypeAnnotation,
        bits_type: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        let rust_ty = Self::convert_type_with_annotation_and_text(
            &self.module_path,
            Some(type_annotation),
            type_alias_annotation_from_text(alias_text),
            &self.import_aliases,
            bits_type,
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

pub(crate) fn rust_module_path_from_dslx_module_name(module_name: &str) -> Vec<String> {
    module_name
        .split('.')
        .filter(|segment| !segment.is_empty())
        .map(sanitize_module_segment)
        .collect()
}

fn rust_module_path_from_import(import: &dslx::Import) -> Vec<String> {
    import
        .get_subject()
        .iter()
        .map(|segment| sanitize_module_segment(segment))
        .collect()
}

pub(crate) fn rust_type_path_between_dslx_modules(
    current_module_name: &str,
    target_module_name: &str,
    type_name: &str,
) -> String {
    let current_path = rust_module_path_from_dslx_module_name(current_module_name);
    let target_path = rust_module_path_from_dslx_module_name(target_module_name);
    rust_type_path_between_module_paths(&current_path, &target_path, type_name)
}

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

fn import_aliases_from_module_text(module_text: &str) -> BTreeMap<String, String> {
    module_text
        .split(';')
        .filter_map(|statement| {
            let import_text = statement.trim().strip_prefix("import ")?;
            let (subject, alias) = if let Some((subject, alias)) = import_text.split_once(" as ") {
                (subject.trim(), alias.trim())
            } else {
                let subject = import_text.trim();
                (subject, subject.rsplit('.').next().unwrap_or(subject))
            };
            let module_name = subject
                .split('.')
                .filter(|segment| !segment.is_empty())
                .map(sanitize_module_segment)
                .collect::<Vec<_>>()
                .join(".");
            Some((alias.to_string(), module_name))
        })
        .collect()
}

fn top_level_split(input: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut square_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut angle_depth = 0usize;
    for (index, ch) in input.char_indices() {
        match ch {
            '[' => square_depth = square_depth.saturating_add(1),
            ']' => square_depth = square_depth.saturating_sub(1),
            '(' => paren_depth = paren_depth.saturating_add(1),
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '{' => brace_depth = brace_depth.saturating_add(1),
            '}' => brace_depth = brace_depth.saturating_sub(1),
            '<' => angle_depth = angle_depth.saturating_add(1),
            '>' => angle_depth = angle_depth.saturating_sub(1),
            _ => {}
        }
        if ch == delimiter
            && square_depth == 0
            && paren_depth == 0
            && brace_depth == 0
            && angle_depth == 0
        {
            parts.push(&input[start..index]);
            start = index + ch.len_utf8();
        }
    }
    parts.push(&input[start..]);
    parts
}

fn split_name_annotation(input: &str) -> Option<(String, String)> {
    let bytes = input.as_bytes();
    for (index, ch) in input.char_indices() {
        if ch == ':'
            && bytes.get(index.wrapping_sub(1)) != Some(&b':')
            && bytes.get(index + 1) != Some(&b':')
        {
            let name = input[..index].trim();
            let annotation = input[index + 1..].trim();
            if !name.is_empty() && !annotation.is_empty() {
                return Some((name.to_string(), annotation.to_string()));
            }
        }
    }
    None
}

fn struct_member_annotations_from_text(struct_text: &str) -> BTreeMap<String, String> {
    let Some(open) = struct_text.find('{') else {
        return BTreeMap::new();
    };
    let Some(close) = struct_text.rfind('}') else {
        return BTreeMap::new();
    };
    top_level_split(&struct_text[open + 1..close], ',')
        .into_iter()
        .filter_map(split_name_annotation)
        .collect()
}

fn type_alias_annotation_from_text(alias_text: &str) -> Option<&str> {
    let equals = alias_text.find('=')?;
    let rest = &alias_text[equals + 1..];
    let semicolon = rest.rfind(';').unwrap_or(rest.len());
    Some(rest[..semicolon].trim()).filter(|annotation| !annotation.is_empty())
}

fn array_element_annotation_text(annotation_text: &str) -> Option<&str> {
    let trimmed = annotation_text.trim();
    if !trimmed.ends_with(']') {
        return None;
    }
    let mut depth = 0usize;
    for (index, ch) in trimmed.char_indices().rev() {
        if ch == ']' {
            depth = depth.saturating_add(1);
        } else if ch == '[' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                let element = trimmed[..index].trim();
                return (!element.is_empty()).then_some(element);
            }
        }
    }
    None
}

fn is_builtin_bits_annotation(annotation_text: &str) -> bool {
    if annotation_text == "bool"
        || annotation_text.starts_with("uN[")
        || annotation_text.starts_with("sN[")
        || annotation_text.starts_with("bits[")
    {
        return true;
    }
    let digits = annotation_text
        .strip_prefix('u')
        .or_else(|| annotation_text.strip_prefix('s'));
    digits.is_some_and(|digits| !digits.is_empty() && digits.chars().all(|ch| ch.is_ascii_digit()))
}

fn rust_type_ref_from_annotation_text(
    current_module_path: &[String],
    annotation_text: &str,
    import_aliases: &BTreeMap<String, String>,
) -> Option<String> {
    let trimmed = annotation_text.trim();
    if trimmed.is_empty() || trimmed.contains('[') || is_builtin_bits_annotation(trimmed) {
        return None;
    }
    if let Some((module_ref, attr)) = trimmed.rsplit_once("::") {
        let module_key = module_ref.rsplit('.').next().unwrap_or(module_ref);
        let module_name = import_aliases
            .get(module_ref)
            .or_else(|| import_aliases.get(module_key))
            .cloned()
            .unwrap_or_else(|| {
                module_ref
                    .split('.')
                    .filter(|segment| !segment.is_empty())
                    .map(sanitize_module_segment)
                    .collect::<Vec<_>>()
                    .join(".")
            });
        let target_module_path = rust_module_path_from_dslx_module_name(&module_name);
        Some(rust_type_path_between_module_paths(
            current_module_path,
            &target_module_path,
            attr,
        ))
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::dslx_bridge::{convert_imported_module, convert_leaf_module};

    use super::*;

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

} // pub mod my_module"#
        );
    }

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

} // pub mod my_module"#
        );
    }

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

} // pub mod my_module"#
        );
    }

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

} // pub mod my_module"#
        );
    }

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

} // pub mod my_module"#
        );
    }

    #[test]
    fn test_convert_leaf_module_type_alias_to_bits_type_only() {
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

} // pub mod my_module"#
        );
    }

    #[test]
    fn test_function_signatures_do_not_emit_aliases() {
        let dslx = r#"
        type Count = u8;
        struct ResultStruct { value: Count }
        pub fn do_route(count: Count) -> ResultStruct {
            ResultStruct { value: count }
        }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut default_builder = RustBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut default_builder).unwrap();
        let output = default_builder.build();
        assert!(!output.contains("DoRouteCount"));
        assert!(!output.contains("DoRouteReturn"));
    }

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

} // pub mod importer"
        );
    }

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

} // pub mod importer"
        );
    }

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

} // pub mod importer"
        );
    }

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

} // pub mod importer"
        );
    }
}
