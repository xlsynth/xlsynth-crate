// SPDX-License-Identifier: Apache-2.0

//! Builder that creates SystemVerilog type definitions from DSLX type
//! definitions.

use std::collections::HashSet;

use crate::{
    dslx,
    dslx_bridge::{BridgeBuilder, StructMemberData},
    ir_value::IrFormatPreference,
    IrValue, XlsynthError,
};

pub struct SvBridgeBuilder {
    lines: Vec<String>,
    /// We keep a record of all the names we define flat within the namespace so
    /// that we can detect and report collisions at generation time instead
    /// of in a subsequent linting step.
    defined: HashSet<String>,
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

fn screaming_snake_to_upper_camel(name: &str) -> String {
    name.split('_')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut chars = s.chars();
            chars
                .next()
                .map(|c| c.to_ascii_uppercase().to_string())
                .unwrap_or_default()
                + &chars.as_str().to_ascii_lowercase()
        })
        .collect()
}

fn is_screaming_snake_case(name: &str) -> bool {
    name.chars().all(|c| {
        if c.is_ascii_alphabetic() {
            c.is_ascii_uppercase()
        } else {
            true
        }
    })
}

fn make_array_span_suffix(array_size: usize) -> String {
    if array_size <= 1 {
        "".to_string()
    } else {
        format!(" [{}:0]", array_size - 1)
    }
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

/// Note: this only supports a very simple package naming and associated
/// hierarchy for the time being.
fn import_to_pkg_name(import: &dslx::Import) -> Result<String, XlsynthError> {
    let subject = import.get_subject();
    assert!(
        subject.len() > 0,
        "import subjects always have at least one token"
    );
    Ok(format!("{}_sv_pkg", subject.last().unwrap()))
}

impl SvBridgeBuilder {
    pub fn new() -> Self {
        Self {
            lines: vec![],
            defined: HashSet::new(),
        }
    }

    pub fn build(&self) -> String {
        self.lines.join("\n")
    }

    fn define_or_error(&mut self, name: &str, ctx: &str) -> Result<(), XlsynthError> {
        let inserted = self.defined.insert(name.to_string());
        if inserted {
            Ok(())
        } else {
            Err(XlsynthError(format!(
                "Building SV; name collision detected for SV name in generated module namespace: `{name}` context: {ctx}"
            )))
        }
    }

    fn convert_type(ty: &dslx::Type, array_size: Option<usize>) -> Result<String, XlsynthError> {
        if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            let leader = if is_signed { "logic signed" } else { "logic" };
            Ok(format!(
                "{}{}{}",
                leader,
                make_array_span_suffix(array_size.unwrap_or(0)),
                make_bit_span_suffix(bit_count)
            ))
        } else if ty.is_enum() {
            let enum_def = ty.get_enum_def().unwrap();
            Ok(format!(
                "{}{}",
                Self::enum_name_to_sv(&enum_def.get_identifier()),
                make_array_span_suffix(array_size.unwrap_or(0))
            ))
        } else if ty.is_struct() {
            let struct_def = ty.get_struct_def().unwrap();
            Ok(format!(
                "{}{}",
                Self::struct_name_to_sv(&struct_def.get_identifier()),
                make_array_span_suffix(array_size.unwrap_or(0))
            ))
        } else if ty.is_array() {
            let array_ty = ty.get_array_element_type();
            let array_size = ty.get_array_size();
            let sv_ty = Self::convert_type(&array_ty, Some(array_size))?;
            println!("sv_ty: {}", sv_ty);
            Ok(sv_ty)
        } else {
            Err(XlsynthError(format!(
                "Unsupported type for conversion from DSLX to SystemVerilog: {:?}",
                ty.to_string()?
            )))
        }
    }

    /// Converts a DSLX enum name in CamelCase to a SystemVerilog enum name in
    /// snake_case with an _t suffix i.e. `MyEnum` -> `my_enum_t`
    fn enum_name_to_sv(dslx_name: &str) -> String {
        format!("{}_t", camel_to_snake(dslx_name))
    }

    fn enum_member_name_to_sv(dslx_name: &str) -> String {
        if is_screaming_snake_case(dslx_name) {
            screaming_snake_to_upper_camel(dslx_name)
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
        let ctx = format!("DSLX enum `{dslx_name}`");
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
            self.define_or_error(&sv_member_name, &ctx)?;
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
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError> {
        let mut lines = vec![];
        lines.push(format!("typedef struct packed {{"));
        for member in members {
            let member_name = &member.name;
            let member_concrete_ty = &member.concrete_type;
            let member_annotated_ty = &member.type_annotation;
            if let Some(type_ref_type_annotation) =
                member_annotated_ty.to_type_ref_type_annotation()
            {
                let type_ref = type_ref_type_annotation.get_type_ref();
                let type_def = type_ref.get_type_definition();
                if let Some(colon_ref) = type_def.to_colon_ref() {
                    if let Some(import) = colon_ref.resolve_import_subject() {
                        let pkg_name = import_to_pkg_name(&import)?;
                        let attr_sv_type_name = Self::convert_type(member_concrete_ty, None)?;
                        let extern_ref = format!("{pkg_name}::{attr_sv_type_name}");
                        lines.push(format!("    {} {};", extern_ref, member_name));
                        continue;
                    }
                }
            }
            if member_concrete_ty.is_array() {
                // Arrays are displayed differently from other members, the size is after the
                // name, separated from the element type.
                let element_ty = member_concrete_ty.get_array_element_type();
                let array_size = member_concrete_ty.get_array_size();
                let struct_string = Self::convert_type(&element_ty, Some(array_size))?;
                lines.push(format!("    {} {};", struct_string, member_name));
            } else {
                let member_sv_ty = Self::convert_type(member_concrete_ty, None)?;
                lines.push(format!("    {} {};", member_sv_ty, member_name));
            }
        }
        lines.push(format!("}} {};\n", Self::struct_name_to_sv(dslx_name)));
        self.lines.push(lines.join("\n"));
        Ok(())
    }

    fn add_alias(&mut self, dslx_name: &str, bits_type: dslx::Type) -> Result<(), XlsynthError> {
        let sv_ty = Self::convert_type(&bits_type, None)?;
        let sv_name = format!("{}_t", camel_to_snake(dslx_name));
        self.lines.push(format!("typedef {} {};\n", sv_ty, sv_name));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::dslx_bridge::{convert_imported_module, convert_leaf_module};

    use super::*;

    fn simple_convert_for_test(dslx: &str) -> Result<String, XlsynthError> {
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = SvBridgeBuilder::new();
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder)?;
        Ok(builder.build())
    }

    /// Demonstrates that we do not change the case of enum members that are
    /// defined as UpperCamelCase in DSLX.
    #[test]
    fn test_convert_leaf_module_enum_def_only() {
        let dslx = r#"
        enum OpType : u2 { Read = 0, Write = 1 }
        "#;
        let sv = simple_convert_for_test(dslx).unwrap();
        test_helpers::assert_valid_sv(&sv);
        assert_eq!(
            sv,
            r#"typedef enum logic [1:0] {
    Read = 2'd0,
    Write = 2'd1
} op_type_t;
"#
        );
    }

    /// Demonstrates that we convert enums that are defined as
    /// SCREAMING_SNAKE_CASE in DSLX into enums defined with UpperCamelCase
    /// in SystemVerilog.
    #[test]
    fn test_convert_leaf_module_enum_def_camel_case() {
        let dslx = r#"
        enum MyEnum : u2 { MY_FIRST_VALUE = 0, MY_SECOND_VALUE = 1 }
        "#;
        let sv = simple_convert_for_test(dslx).unwrap();
        test_helpers::assert_valid_sv(&sv);
        assert_eq!(
            sv,
            r#"typedef enum logic [1:0] {
    MyFirstValue = 2'd0,
    MySecondValue = 2'd1
} my_enum_t;
"#
        );
    }

    #[test]
    fn test_convert_leaf_module_struct_def_only() {
        let dslx = r#"
        struct MyStruct {
            byte_array: u8[10],
            word_data: u16,
        }
        "#;
        let sv = simple_convert_for_test(dslx).unwrap();
        assert_eq!(
            sv,
            r#"typedef struct packed {
    logic [9:0] [7:0] byte_array;
    logic [15:0] word_data;
} my_struct_t;
"#
        );
    }

    #[test]
    fn test_convert_leaf_module_type_alias_to_bits_type_only() {
        let dslx = "type MyType = u8;";
        let sv = simple_convert_for_test(dslx).unwrap();
        assert_eq!(sv, "typedef logic [7:0] my_type_t;\n");
        test_helpers::assert_valid_sv(&sv);
    }

    /// Demonstrates that we get an error when we attempt to emit two enums who
    /// have the same member name -- while this is acceptable in DSLX the
    /// fact we flatten the enum names into a single namespace in SV means
    /// we have an error to flag, in which case we currently expect
    /// user correction.
    #[test]
    fn test_convert_leaf_module_enum_defs_with_collision() {
        let dslx = "enum MyFirstEnum : u1 { A = 0, B = 1 }
        enum MySecondEnum: u3 { A = 3, B = 4 }";
        let result = simple_convert_for_test(dslx);
        // We expect this caused a collision error on `A`.
        let err = result.expect_err("expect collision");
        assert!(err.to_string().contains("name collision detected for SV name in generated module namespace: `A` context: DSLX enum `MySecondEnum`"));
    }

    #[test]
    fn test_is_screaming_snake_case() {
        assert!(is_screaming_snake_case("FOO_BAR"));
        assert!(is_screaming_snake_case("ONEWORD"));

        assert!(!is_screaming_snake_case("FooBar"));
        assert!(!is_screaming_snake_case("blah"));
    }

    #[test]
    fn test_struct_with_extern_type_ref_member_type_ref_member() {
        let imported_dslx = "pub struct MyImportedStruct { a: u8 }";
        let importer_dslx = "import imported; struct MyStruct { a: imported::MyImportedStruct }";

        let mut import_data = dslx::ImportData::default();
        let _imported_typechecked =
            dslx::parse_and_typecheck(imported_dslx, "imported.x", "imported", &mut import_data)
                .unwrap();
        let importer_typechecked =
            dslx::parse_and_typecheck(importer_dslx, "importer.x", "importer", &mut import_data)
                .unwrap();

        let mut builder = SvBridgeBuilder::new();
        convert_imported_module(&importer_typechecked, &mut builder).unwrap();
        let sv = builder.build();
        assert_eq!(
            sv,
            "typedef struct packed {
    imported_sv_pkg::my_imported_struct_t a;
} my_struct_t;
"
        );
    }
}
