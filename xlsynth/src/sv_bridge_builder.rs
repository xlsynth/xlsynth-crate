// SPDX-License-Identifier: Apache-2.0

//! Builder that creates SystemVerilog type definitions from DSLX type
//! definitions.

use crate::{
    dslx, dslx_bridge::BridgeBuilder, ir_value::IrFormatPreference, IrValue, XlsynthError,
};

pub struct SvBridgeBuilder {
    lines: Vec<String>,
}

fn camel_to_snake(name: &str) -> String {
    let mut snake = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            snake.push('_');
        }
        snake.push(c.to_ascii_lowercase());
    }
    snake
}

fn make_bit_span_suffix(bit_count: usize) -> String {
    // More study required on how compatible
    assert!(bit_count > 0);
    if bit_count == 1 {
        "".to_string()
    } else {
        format!(" [{}:0]", bit_count - 1)
    }
}

impl SvBridgeBuilder {
    pub fn new() -> Self {
        Self { lines: vec![] }
    }

    pub fn build(&self) -> String {
        self.lines.join("\n")
    }

    fn convert_type(ty: &dslx::Type) -> Result<String, XlsynthError> {
        if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            let leader = if is_signed { "logic signed" } else { "logic" };
            Ok(format!("{}{}", leader, make_bit_span_suffix(bit_count)))
        } else if ty.is_enum() {
            let enum_def = ty.get_enum_def().unwrap();
            Ok(Self::enum_name_to_sv(&enum_def.get_identifier()))
        } else if ty.is_struct() {
            let struct_def = ty.get_struct_def().unwrap();
            Ok(Self::struct_name_to_sv(&struct_def.get_identifier()))
        } else if ty.is_array() {
            let array_ty = ty.get_array_element_type();
            let array_size = ty.get_array_size();
            let sv_ty = Self::convert_type(&array_ty)?;
            Ok(format!("{}[{}]", sv_ty, array_size))
        } else {
            Err(XlsynthError(format!(
                "Unsupported type for conversion from DSLX to SystemVerilog: {:?}",
                ty.to_string()?
            )))
        }
    }

    /// Converts a DSLX enum name in CamelCase to a SystemVerilog enum name in
    /// snake_case with an _e suffix i.e. `MyEnum` -> `my_enum_e`
    fn enum_name_to_sv(dslx_name: &str) -> String {
        format!("{}_e", camel_to_snake(dslx_name))
    }

    fn enum_member_name_to_sv(dslx_name: &str) -> String {
        if !dslx_name.chars().all(char::is_uppercase) {
            camel_to_snake(dslx_name).to_uppercase()
        } else {
            dslx_name.to_string()
        }
    }

    /// Converts a DSLX struct name in CamelCase to a SystemVerilog struct name
    /// in snake_case with a _t suffix
    fn struct_name_to_sv(dslx_name: &str) -> String {
        format!("{}_t", camel_to_snake(dslx_name))
    }
}

impl BridgeBuilder for SvBridgeBuilder {
    fn start_module(&mut self, _module_name: &str) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn end_module(&mut self, _module_name: &str) -> Result<(), XlsynthError> {
        Ok(())
    }

    fn add_enum_def(
        &mut self,
        dslx_name: &str,
        is_signed: bool,
        underlying_bit_count: usize,
        members: &[(String, IrValue)],
    ) -> Result<(), XlsynthError> {
        let mut lines = vec![];
        let sv_name = Self::enum_name_to_sv(dslx_name);
        lines.push(format!(
            "typedef enum logic{} {{",
            make_bit_span_suffix(underlying_bit_count)
        ));
        for (i, (member_name, member_value)) in members.iter().enumerate() {
            let format = if is_signed {
                IrFormatPreference::SignedDecimal
            } else {
                IrFormatPreference::UnsignedDecimal
            };
            let member_value_str = member_value.to_string_fmt(format)?;
            let digits = member_value_str.split(':').nth(1).expect("split success");
            let maybe_comma = if i < members.len() - 1 { "," } else { "" };
            let sv_member_name = Self::enum_member_name_to_sv(member_name);
            lines.push(format!(
                "    {} = {}'d{}{}",
                sv_member_name, underlying_bit_count, digits, maybe_comma
            ));
        }
        lines.push(format!("}} {};\n", sv_name));
        self.lines.push(lines.join("\n"));
        Ok(())
    }

    fn add_struct_def(
        &mut self,
        dslx_name: &str,
        members: &[(String, dslx::Type)],
    ) -> Result<(), XlsynthError> {
        let mut lines = vec![];
        lines.push(format!("typedef struct packed {{"));
        for (member_name, member_ty) in members {
            if member_ty.is_array() {
                // Arrays are displayed differently from other members, the size is after the
                // name, separated from the element type.
                let element_ty = member_ty.get_array_element_type();
                let element_sv_ty = Self::convert_type(&element_ty)?;
                let array_size = member_ty.get_array_size();
                lines.push(format!(
                    "    {} {}[{}];",
                    element_sv_ty, member_name, array_size
                ));
            } else {
                let member_sv_ty = Self::convert_type(member_ty)?;
                lines.push(format!("    {} {};", member_sv_ty, member_name));
            }
        }
        lines.push(format!("}} {};\n", Self::struct_name_to_sv(dslx_name)));
        self.lines.push(lines.join("\n"));
        Ok(())
    }

    fn add_alias(&mut self, dslx_name: &str, bits_type: dslx::Type) -> Result<(), XlsynthError> {
        let sv_ty = Self::convert_type(&bits_type)?;
        let sv_name = format!("{}_t", camel_to_snake(dslx_name));
        self.lines.push(format!("typedef {} {};\n", sv_ty, sv_name));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::dslx_bridge::convert_leaf_module;

    use super::*;

    #[test]
    fn test_convert_leaf_module_enum_def_only() {
        let dslx = r#"
        enum OpType : u2 { READ = 0, WRITE = 1 }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = SvBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"typedef enum logic [1:0] {
    READ = 2'd0,
    WRITE = 2'd1
} op_type_e;
"#
        );
    }

    /// Demonstrates that we convert enums that are defined as CamelCase in DSLX
    /// into enums defined with SCREAMING_SNAKE_CASE in SystemVerilog.
    #[test]
    fn test_convert_leaf_module_enum_def_camel_case() {
        let dslx = r#"
        enum MyEnum : u2 { MyFirstValue = 0, MySecondValue = 1 }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = SvBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"typedef enum logic [1:0] {
    MY_FIRST_VALUE = 2'd0,
    MY_SECOND_VALUE = 2'd1
} my_enum_e;
"#
        );
    }

    #[test]
    fn test_convert_leaf_module_struct_def_only() {
        let dslx = r#"
        struct MyStruct {
            byte_data: u8,
            word_data: u16,
        }
        "#;
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = SvBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(
            builder.build(),
            r#"typedef struct packed {
    logic [7:0] byte_data;
    logic [15:0] word_data;
} my_struct_t;
"#
        );
    }

    #[test]
    fn test_convert_leaf_module_type_alias_to_bits_type_only() {
        let dslx = "type MyType = u8;";
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = SvBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder).unwrap();
        assert_eq!(builder.build(), "typedef logic [7:0] my_type_t;\n");
    }
}
