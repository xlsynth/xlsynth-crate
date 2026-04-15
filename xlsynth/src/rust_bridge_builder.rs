// SPDX-License-Identifier: Apache-2.0

//! Builder that creates Rust type definitions from DSLX type definitions.
//!
//! This helps us e.g. call DSLX functions from Rust code, i.e. it enables
//! Rust->DSLX FFI interop.

use crate::{
    dslx,
    dslx_bridge::{BridgeBuilder, FunctionParamData, StructMemberData},
    IrValue, XlsynthError,
};

pub struct RustBridgeBuilder {
    lines: Vec<String>,
    target_function_name: Option<String>,
}

impl RustBridgeBuilder {
    pub fn new() -> Self {
        Self {
            lines: vec![],
            target_function_name: None,
        }
    }

    /// Creates a builder that emits Rust aliases for one DSLX function
    /// signature.
    pub fn with_function_signature_aliases(function_name: impl Into<String>) -> Self {
        Self {
            lines: vec![],
            target_function_name: Some(function_name.into()),
        }
    }

    pub fn build(&self) -> String {
        self.lines.join("\n")
    }

    fn convert_type(ty: &dslx::Type) -> Result<String, XlsynthError> {
        if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            let signed_str = if is_signed { "S" } else { "U" };
            Ok(format!("Ir{signed_str}Bits<{bit_count}>"))
        } else if ty.is_enum() {
            let enum_def = ty.get_enum_def().unwrap();
            Ok(enum_def.get_identifier().to_string())
        } else if ty.is_struct() {
            let struct_def = ty.get_struct_def().unwrap();
            Ok(struct_def.get_identifier().to_string())
        } else if ty.is_array() {
            let array_ty = ty.get_array_element_type();
            let array_size = ty.get_array_size();
            let rust_ty = Self::convert_type(&array_ty)?;
            Ok(format!("[{rust_ty}; {array_size}]"))
        } else {
            Err(XlsynthError(format!(
                "Unsupported type for conversion from DSLX to Rust: {:?}",
                ty.to_string()?
            )))
        }
    }

    fn convert_type_annotation_or_type(
        type_annotation: Option<&dslx::TypeAnnotation>,
        concrete_type: Option<&dslx::Type>,
    ) -> Result<String, XlsynthError> {
        if let Some(type_ref_type_annotation) =
            type_annotation.and_then(|annotation| annotation.to_type_ref_type_annotation())
        {
            let type_definition = type_ref_type_annotation
                .get_type_ref()
                .get_type_definition();
            if let Some(type_alias) = type_definition.to_type_alias() {
                Ok(type_alias.get_identifier())
            } else if let Some(colon_ref) = type_definition.to_colon_ref() {
                let attr = colon_ref.get_attr();
                let rust_path = if let Some(import) = colon_ref.resolve_import_subject() {
                    let subject = import.get_subject();
                    if let Some(module_name) = subject.last() {
                        format!("super::{module_name}::{attr}")
                    } else {
                        attr
                    }
                } else {
                    attr
                };
                Ok(rust_path)
            } else if let Some(concrete_type) = concrete_type {
                Self::convert_type(concrete_type)
            } else {
                Err(XlsynthError(
                    "DSLX type annotation did not resolve to a Rust type".to_string(),
                ))
            }
        } else if let Some(concrete_type) = concrete_type {
            Self::convert_type(concrete_type)
        } else {
            Err(XlsynthError(
                "DSLX type annotation did not resolve to a Rust type".to_string(),
            ))
        }
    }

    fn upper_camel_identifier(name: &str) -> String {
        name.split('_')
            .filter(|part| !part.is_empty())
            .map(|part| {
                let mut chars = part.chars();
                if let Some(first) = chars.next() {
                    first.to_ascii_uppercase().to_string() + chars.as_str()
                } else {
                    String::new()
                }
            })
            .collect::<String>()
    }
}

impl Default for RustBridgeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BridgeBuilder for RustBridgeBuilder {
    fn start_module(&mut self, module_name: &str) -> Result<(), XlsynthError> {
        self.lines = vec![
            format!("pub mod {module_name} {{"),
            // We allow e.g. enum variants to be unused in consumer code.
            "#![allow(dead_code)]".to_string(),
            "#![allow(unused_imports)]".to_string(),
            "use xlsynth::{IrValue, IrUBits, IrSBits};\n".to_string(),
        ];
        Ok(())
    }

    fn end_module(&mut self, module_name: &str) -> Result<(), XlsynthError> {
        self.lines.push(format!("}} // pub mod {module_name}"));
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

        self.lines.push(format!("pub enum {dslx_name} {{"));
        for (name, value) in members.iter() {
            self.lines
                .push(format!("    {} = {},", name, value_to_string(value)?));
        }
        self.lines.push("}\n".to_string());

        // Now we emit the converter so we can easily pass our generated Rust enum to IR
        // interpreter functions.
        self.lines
            .push(format!("impl Into<IrValue> for {dslx_name} {{"));
        self.lines
            .push("    fn into(self) -> IrValue {".to_string());
        self.lines.push("        match self {".to_string());
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
        self.lines.push(format!("pub struct {dslx_name} {{"));
        for member in members {
            let rust_ty = Self::convert_type_annotation_or_type(
                Some(&member.type_annotation),
                Some(&member.concrete_type),
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
        let rust_ty =
            Self::convert_type_annotation_or_type(Some(type_annotation), Some(bits_type))?;
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

    fn add_function_signature(
        &mut self,
        dslx_name: &str,
        params: &[FunctionParamData],
        return_type_annotation: Option<&dslx::TypeAnnotation>,
        return_type: Option<&dslx::Type>,
    ) -> Result<(), XlsynthError> {
        if self.target_function_name.as_deref() == Some(dslx_name) {
            let function_prefix = Self::upper_camel_identifier(dslx_name);
            self.lines
                .push(format!("// Rust aliases for DSLX function `{dslx_name}`."));
            for param in params {
                let alias_name = format!(
                    "{function_prefix}{}",
                    Self::upper_camel_identifier(&param.name)
                );
                let rust_ty = Self::convert_type_annotation_or_type(
                    Some(&param.type_annotation),
                    param.concrete_type.as_ref(),
                )?;
                self.lines
                    .push(format!("pub type {alias_name} = {rust_ty};"));
            }
            if let Some(return_type) = return_type {
                let rust_ty = Self::convert_type_annotation_or_type(
                    return_type_annotation,
                    Some(return_type),
                )?;
                self.lines
                    .push(format!("pub type {function_prefix}Return = {rust_ty};"));
            }
            self.lines.push("".to_string());
        }
        Ok(())
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

pub enum MyEnum {
    A = 0,
    B = 3,
}

impl Into<IrValue> for MyEnum {
    fn into(self) -> IrValue {
        match self {
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

pub enum MyEnum {
    A = 0,
    B = 3,
}

impl Into<IrValue> for MyEnum {
    fn into(self) -> IrValue {
        match self {
            MyEnum::A => IrValue::make_ubits(2, 0).unwrap(),
            MyEnum::B => IrValue::make_ubits(2, 3).unwrap(),
        }
    }
}

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

pub struct MyInnerStruct {
    pub x: IrUBits<8>,
    pub y: IrUBits<8>,
}

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
    fn test_convert_leaf_module_function_signature_aliases() {
        let dslx = r#"
        struct Cfg {
            a: u32,
        }
        type Samples = u8[4];
        pub fn top(cfg: Cfg, samples: Samples) -> Cfg {
            cfg
        }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = RustBridgeBuilder::with_function_signature_aliases("top");
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"pub mod my_module {
#![allow(dead_code)]
#![allow(unused_imports)]
use xlsynth::{IrValue, IrUBits, IrSBits};

pub struct Cfg {
    pub a: IrUBits<32>,
}

pub type Samples = [IrUBits<8>; 4];

// Rust aliases for DSLX function `top`.
pub type TopCfg = Cfg;
pub type TopSamples = Samples;
pub type TopReturn = Cfg;

} // pub mod my_module"#
        );
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

pub struct MyStruct {
    pub a: super::imported::MyImportedStruct,
}

} // pub mod importer"
        );
    }
}
