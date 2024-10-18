// SPDX-License-Identifier: Apache-2.0

//! Library for generating Rust code that reflects the types and callables in a
//! DSLX module subtree.
//!
//! We walk the type definitions and callable interfaces present in the DSLX
//! module and generate corresponding Rust code that can be `use`'d into a Rust
//! module.

use crate::{dslx, IrValue, XlsynthError};

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

fn convert_enum_to_rust(
    enum_def: &dslx::EnumDef,
    type_info: &dslx::TypeInfo,
) -> Result<String, XlsynthError> {
    let tups = enum_as_tups(enum_def, type_info);
    let enum_underlying = type_info.get_type_for_enum_def(enum_def);
    let is_signed = enum_underlying.is_signed_bits()?;

    let value_to_string = |value: &IrValue| -> Result<String, XlsynthError> {
        if is_signed {
            value.to_i64().map(|v| v.to_string())
        } else {
            value.to_u64().map(|v| v.to_string())
        }
    };

    let mut lines: Vec<String> = vec![];
    let enum_name = enum_def.get_identifier();
    lines.push(format!("pub enum {} {{", enum_name));
    for (name, value) in tups.iter() {
        lines.push(format!("    {} = {},", name, value_to_string(value)?));
    }
    lines.push("}\n".to_string());
    lines.push(format!("impl Into<IrValue> for {} {{", enum_name));
    lines.push("    fn into(self) -> IrValue {".to_string());
    lines.push("        match self {".to_string());
    for (member_name, value) in tups.iter() {
        let value_str = value_to_string(value)?;
        lines.push(format!(
            "            {}::{} => IrValue::make_bits({}, {}).unwrap(),",
            enum_name,
            member_name,
            value.bit_count(),
            value_str
        ));
    }
    lines.push("        }".to_string());
    lines.push("    }".to_string());
    lines.push("}".to_string());
    Ok(lines.join("\n"))
}

fn convert_struct_to_rust(
    struct_def: &dslx::StructDef,
    type_info: &dslx::TypeInfo,
) -> Result<String, XlsynthError> {
    let mut lines: Vec<String> = vec![];
    let struct_name = struct_def.get_identifier();
    lines.push(format!("pub struct {} {{", struct_name));
    for i in 0..struct_def.get_member_count() {
        let member = struct_def.get_member(i);
        let member_name = member.get_name();
        let member_type = type_info.get_type_for_struct_member(&member);
        if let Some((is_signed, bit_count)) = member_type.is_bits_like() {
            lines.push(format!(
                "    pub {}: Ir{}Bits<{}>,",
                member_name,
                if is_signed { "S" } else { "U" },
                bit_count
            ));
        } else {
            todo!("convert struct member type to Rust type: {}", member_type);
        }
    }
    lines.push("}\n".to_string());
    Ok(lines.join("\n"))
}

pub fn convert_leaf_module(
    import_data: &mut dslx::ImportData,
    dslx_program: &str,
    path: &std::path::Path,
) -> Result<String, XlsynthError> {
    // If the path is `path/to/foo.x` then the module name is `foo`.
    let module_name = path.file_stem().unwrap().to_str().unwrap();
    let path_str = path.to_str().unwrap();
    let typechecked_module =
        dslx::parse_and_typecheck(dslx_program, path_str, module_name, import_data)?;
    let module = typechecked_module.get_module();
    let type_info = typechecked_module.get_type_info();

    let mut chunks: Vec<String> = vec![
        format!("mod {module_name} {{"),
        // We allow e.g. enum variants to be unused in consumer code.
        "#![allow(dead_code)]".to_string(),
        "use xlsynth::{IrValue, IrUBits, IrSBits};\n".to_string(),
    ];
    for i in 0..module.get_type_definition_count() {
        let type_def_kind = module.get_type_definition_kind(i);
        match type_def_kind {
            dslx::TypeDefinitionKind::EnumDef => {
                let enum_def = module.get_type_definition_as_enum_def(i).unwrap();
                chunks.push(convert_enum_to_rust(&enum_def, &type_info)?)
            }
            dslx::TypeDefinitionKind::StructDef => {
                let struct_def = module.get_type_definition_as_struct_def(i).unwrap();
                chunks.push(convert_struct_to_rust(&struct_def, &type_info)?)
            }
            dslx::TypeDefinitionKind::TypeAlias => todo!("convert type alias from DSLX to Rust"),
            dslx::TypeDefinitionKind::ColonRef => todo!("convert colon ref from DSLX to Rust"),
        }
    }
    chunks.push(format!("}} // mod {module_name}"));
    Ok(chunks.join("\n"))
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
        let rust = convert_leaf_module(&mut import_data, dslx, &path).unwrap();
        assert_eq!(
            rust,
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
        let rust = convert_leaf_module(&mut import_data, dslx, &path).unwrap();
        assert_eq!(
            rust,
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
}
