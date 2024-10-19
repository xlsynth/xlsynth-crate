// SPDX-License-Identifier: Apache-2.0

//! Library for generating Rust code that reflects the types and callables in a
//! DSLX module subtree.
//!
//! We walk the type definitions and callable interfaces present in the DSLX
//! module and generate corresponding Rust code that can be `use`'d into a Rust
//! module.

use crate::{dslx, IrValue, XlsynthError};

pub trait BridgeBuilder {
    fn start_module(&mut self, module_name: &str) -> Result<(), XlsynthError>;

    fn end_module(&mut self, module_name: &str) -> Result<(), XlsynthError>;

    /// `is_signed` indicates whether the bits type underlying the enum is
    /// signed.
    fn add_enum_def(
        &mut self,
        dslx_name: &str,
        is_signed: bool,
        members: &[(String, IrValue)],
    ) -> Result<(), XlsynthError>;

    fn add_struct_def(
        &mut self,
        dslx_name: &str,
        members: &[(String, dslx::Type)],
    ) -> Result<(), XlsynthError>;
}

pub struct RustBridgeBuilder {
    lines: Vec<String>,
}

impl RustBridgeBuilder {
    pub fn new() -> Self {
        Self { lines: vec![] }
    }

    pub fn build(&self) -> String {
        self.lines.join("\n")
    }

    fn convert_type(ty: &dslx::Type) -> Result<String, XlsynthError> {
        if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            let signed_str = if is_signed { "S" } else { "U" };
            Ok(format!("Ir{}Bits<{}>", signed_str, bit_count))
        } else if ty.is_enum() {
            let enum_def = ty.get_enum_def().unwrap();
            Ok(enum_def.get_identifier().to_string())
        } else if ty.is_struct() {
            let struct_def = ty.get_struct_def().unwrap();
            Ok(struct_def.get_identifier().to_string())
        } else {
            Err(XlsynthError(format!(
                "Unsupported type for conversion from DSLX to Rust: {:?}",
                ty.to_string()?
            )))
        }
    }
}

impl BridgeBuilder for RustBridgeBuilder {
    fn start_module(&mut self, module_name: &str) -> Result<(), XlsynthError> {
        self.lines = vec![
            format!("mod {module_name} {{"),
            // We allow e.g. enum variants to be unused in consumer code.
            "#![allow(dead_code)]".to_string(),
            "use xlsynth::{IrValue, IrUBits, IrSBits};\n".to_string(),
        ];
        Ok(())
    }

    fn end_module(&mut self, module_name: &str) -> Result<(), XlsynthError> {
        self.lines.push(format!("}} // mod {module_name}"));
        Ok(())
    }

    fn add_enum_def(
        &mut self,
        dslx_name: &str,
        is_signed: bool,
        members: &[(String, IrValue)],
    ) -> Result<(), XlsynthError> {
        let value_to_string = |value: &IrValue| -> Result<String, XlsynthError> {
            if is_signed {
                value.to_i64().map(|v| v.to_string())
            } else {
                value.to_u64().map(|v| v.to_string())
            }
        };

        self.lines.push(format!("pub enum {} {{", dslx_name));
        for (name, value) in members.iter() {
            self.lines
                .push(format!("    {} = {},", name, value_to_string(value)?));
        }
        self.lines.push("}\n".to_string());

        // Now we emit the converter so we can easily pass our generated Rust enum to IR
        // interpreter functions.
        self.lines
            .push(format!("impl Into<IrValue> for {} {{", dslx_name));
        self.lines
            .push("    fn into(self) -> IrValue {".to_string());
        self.lines.push("        match self {".to_string());
        for (member_name, value) in members.iter() {
            let value_str = value_to_string(value)?;
            self.lines.push(format!(
                "            {}::{} => IrValue::make_bits({}, {}).unwrap(),",
                dslx_name,
                member_name,
                value.bit_count(),
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
        members: &[(String, dslx::Type)],
    ) -> Result<(), XlsynthError> {
        self.lines.push(format!("pub struct {} {{", dslx_name));
        for (name, ty) in members.iter() {
            let rust_ty = Self::convert_type(ty)?;
            self.lines.push(format!("    pub {}: {},", name, rust_ty));
        }
        self.lines.push("}\n".to_string());
        Ok(())
    }
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
    let enum_underlying = type_info.get_type_for_enum_def(enum_def);
    let is_signed = enum_underlying.is_signed_bits()?;
    let enum_name = enum_def.get_identifier();
    builder.add_enum_def(&enum_name, is_signed, &tups)
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
            dslx::TypeDefinitionKind::TypeAlias => todo!("convert type alias from DSLX to Rust"),
            dslx::TypeDefinitionKind::ColonRef => todo!("convert colon ref from DSLX to Rust"),
        }
    }
    builder.end_module(module_name)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

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
            r#"mod my_module {
#![allow(dead_code)]
use xlsynth::{IrValue, IrUBits, IrSBits};

pub enum MyEnum {
    A = 0,
    B = 3,
}

impl Into<IrValue> for MyEnum {
    fn into(self) -> IrValue {
        match self {
            MyEnum::A => IrValue::make_bits(2, 0).unwrap(),
            MyEnum::B => IrValue::make_bits(2, 3).unwrap(),
        }
    }
}
} // mod my_module"#
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
            r#"mod my_module {
#![allow(dead_code)]
use xlsynth::{IrValue, IrUBits, IrSBits};

pub struct MyStruct {
    pub a: IrUBits<32>,
    pub b: IrSBits<16>,
}

} // mod my_module"#
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
            r#"mod my_module {
#![allow(dead_code)]
use xlsynth::{IrValue, IrUBits, IrSBits};

pub enum MyEnum {
    A = 0,
    B = 3,
}

impl Into<IrValue> for MyEnum {
    fn into(self) -> IrValue {
        match self {
            MyEnum::A => IrValue::make_bits(2, 0).unwrap(),
            MyEnum::B => IrValue::make_bits(2, 3).unwrap(),
        }
    }
}

pub struct MyStruct {
    pub a: MyEnum,
    pub b: IrSBits<16>,
}

} // mod my_module"#
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
            r#"mod my_module {
#![allow(dead_code)]
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

} // mod my_module"#
        );
    }
}
