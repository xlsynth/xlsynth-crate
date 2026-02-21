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

/// The suffix used when we typedef logic to a type name.
const LOGIC_ALIAS_SUFFIX: &str = "_t";

/// The suffix used when we typedef an enum to a type name.
const ENUM_ALIAS_SUFFIX: &str = "_t";

/// The suffix used when we typedef a struct to a type name.
const STRUCT_ALIAS_SUFFIX: &str = "_t";

/// The suffix used when we typedef a type alias to a type name.
const TYPE_ALIAS_SUFFIX: &str = "_t";

/// Selects how DSLX enum case symbols are emitted into the generated SV
/// namespace.
///
/// This only affects enum member identifiers (for example `Read` vs
/// `OpType_Read`); it does not change the emitted enum typedef name. The bridge
/// still applies the existing case-normalization rules to each component before
/// combining them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SvEnumCaseNamingPolicy {
    /// Emit only the normalized case name (for example `Read`).
    Unqualified,
    /// Emit `<NormalizedEnumName>_<NormalizedCaseName>` (for example
    /// `OpType_Read`).
    EnumQualified,
}

/// Accumulates SV type declarations for a DSLX module while enforcing a flat
/// generated-name namespace.
///
/// DSLX allows enum members from different enums to share the same case name,
/// but the generated SV emitted by this builder places those case symbols in a
/// single namespace. `defined` tracks all emitted symbols so collisions are
/// reported deterministically during generation instead of surfacing later in a
/// downstream parser or linter.
pub struct SvBridgeBuilder {
    lines: Vec<String>,
    /// We keep a record of all the names we define flat within the namespace so
    /// that we can detect and report collisions at generation time instead
    /// of in a subsequent linting step.
    defined: HashSet<String>,
    /// Controls how enum member symbols are derived before collision checks.
    enum_case_naming_policy: SvEnumCaseNamingPolicy,
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
fn import_to_pkg_name(import: &dslx::Import) -> String {
    let subject = import.get_subject();
    assert!(
        !subject.is_empty(),
        "import subjects always have at least one token"
    );
    format!("{}_sv_pkg", subject.last().unwrap())
}

/// Converts a DSLX enum name in CamelCase to a SystemVerilog enum name in
/// snake_case with an _t suffix i.e. `MyEnum` -> `my_enum_t`
fn enum_name_to_sv(dslx_name: &str) -> String {
    format!("{}{}", camel_to_snake(dslx_name), ENUM_ALIAS_SUFFIX)
}

/// Converts a DSLX struct name in CamelCase to a SystemVerilog struct name
/// in snake_case with a _t suffix
fn struct_name_to_sv(dslx_name: &str) -> String {
    format!("{}{}", camel_to_snake(dslx_name), STRUCT_ALIAS_SUFFIX)
}

/// Returns the extern type reference if the type annotation is an extern type.
fn get_extern_type_ref(
    type_annotation: &dslx::TypeAnnotation,
    concrete_ty: &dslx::Type,
) -> Option<String> {
    if let Some(type_ref_type_annotation) = type_annotation.to_type_ref_type_annotation() {
        let type_ref = type_ref_type_annotation.get_type_ref();

        // Inspect whether the type definition is a colon-reference where the subject is
        // another module.
        let type_definition: dslx::TypeDefinition = type_ref.get_type_definition();
        if let Some(colon_ref) = type_definition.to_colon_ref() {
            if let Some(import) = colon_ref.resolve_import_subject() {
                // It is a reference to a type defined in another module -- refer to its in
                // its external module.
                let pkg_name = import_to_pkg_name(&import);
                let extern_ref =
                    convert_extern_type(&pkg_name, Some(&colon_ref.get_attr()), concrete_ty, None)
                        .unwrap();
                return Some(extern_ref);
            }
        }
    }
    None
}

/// A version of `dslx::Type`'s meaningful contents that we can match on in
/// match expressions.
enum MatchableDslxType {
    BitsLike {
        is_signed: bool,
        bit_count: usize,
    },
    Enum(dslx::EnumDef),
    Struct(dslx::StructDef),
    Array {
        element_ty: Box<DslxType>,
        size: usize,
    },
}

struct DslxType {
    ty: dslx::Type,
    matchable_ty: MatchableDslxType,
}

/// Converts a DSLX type into a Rust-matchable version.
fn dslx_type_to_matchable(ty: &dslx::Type) -> Result<DslxType, XlsynthError> {
    if let Some((is_signed, bit_count)) = ty.is_bits_like() {
        Ok(DslxType {
            ty: ty.clone(),
            matchable_ty: MatchableDslxType::BitsLike {
                is_signed,
                bit_count,
            },
        })
    } else if ty.is_enum() {
        Ok(DslxType {
            ty: ty.clone(),
            matchable_ty: MatchableDslxType::Enum(ty.get_enum_def().unwrap()),
        })
    } else if ty.is_struct() {
        Ok(DslxType {
            ty: ty.clone(),
            matchable_ty: MatchableDslxType::Struct(ty.get_struct_def().unwrap()),
        })
    } else if ty.is_array() {
        Ok(DslxType {
            ty: ty.clone(),
            matchable_ty: MatchableDslxType::Array {
                element_ty: Box::new(dslx_type_to_matchable(&ty.get_array_element_type())?),
                size: ty.get_array_size(),
            },
        })
    } else {
        Err(XlsynthError(format!(
            "Unsupported type for conversion from DSLX to matchable type: {:?}",
            ty.to_string()?
        )))
    }
}

/// Helper for making the packed array representation string suffix -- this
/// comes after the type name.
fn make_array_span_suffix(array_sizes: Vec<usize>) -> String {
    let mut suffix_parts = Vec::new();
    for array_size in array_sizes.iter() {
        suffix_parts.push(format!(" [{}:0]", array_size - 1));
    }
    suffix_parts.join("")
}

// Converts a DSLX type into a SystemVerilog type string.
fn convert_type(ty: &dslx::Type, array_sizes: Option<Vec<usize>>) -> Result<String, XlsynthError> {
    let matchable_ty = dslx_type_to_matchable(ty)?;

    match matchable_ty.matchable_ty {
        MatchableDslxType::BitsLike {
            is_signed,
            bit_count,
        } => {
            let leader = if is_signed { "logic signed" } else { "logic" };
            Ok(format!(
                "{}{}{}",
                leader,
                make_array_span_suffix(array_sizes.unwrap_or_default()),
                make_bit_span_suffix(bit_count)
            ))
        }
        MatchableDslxType::Enum(enum_def) => Ok(format!(
            "{}{}",
            enum_name_to_sv(&enum_def.get_identifier()),
            make_array_span_suffix(array_sizes.unwrap_or_default())
        )),
        MatchableDslxType::Struct(struct_def) => Ok(format!(
            "{}{}",
            struct_name_to_sv(&struct_def.get_identifier()),
            make_array_span_suffix(array_sizes.unwrap_or_default())
        )),
        MatchableDslxType::Array { element_ty, size } => {
            let mut array_sizes = array_sizes.unwrap_or_default();
            array_sizes.push(size);
            convert_type(&element_ty.ty, Some(array_sizes))
        }
    }
}

/// Converts a DSLX type -- one that was determined to be an extern type
/// reference -- into a SystemVerilog type string.
fn convert_extern_type(
    pkg_name: &str,
    attr: Option<&str>,
    ty: &dslx::Type,
    array_sizes: Option<Vec<usize>>,
) -> Result<String, XlsynthError> {
    let matchable_ty = dslx_type_to_matchable(ty)?;
    match matchable_ty.matchable_ty {
        MatchableDslxType::BitsLike { .. } => {
            if let Some(attr) = attr {
                let attr_sv = format!("{}{}", camel_to_snake(attr), LOGIC_ALIAS_SUFFIX);
                Ok(format!("{pkg_name}::{attr_sv}"))
            } else {
                convert_type(ty, array_sizes)
            }
        }
        MatchableDslxType::Enum(enum_def) => Ok(format!(
            "{pkg_name}::{ty_name}",
            ty_name = enum_name_to_sv(&enum_def.get_identifier())
        )),
        MatchableDslxType::Struct(struct_def) => Ok(format!(
            "{pkg_name}::{ty_name}",
            ty_name = struct_name_to_sv(&struct_def.get_identifier())
        )),
        MatchableDslxType::Array { element_ty, size } => {
            let mut array_sizes = array_sizes.unwrap_or_default();
            array_sizes.push(size);
            Ok(convert_extern_type(
                pkg_name,
                None,
                &element_ty.ty,
                Some(array_sizes),
            )?)
        }
    }
}

impl SvBridgeBuilder {
    /// Creates a builder that preserves the historical enum-case naming
    /// behavior.
    ///
    /// Callers that need enum-qualified SV case names should use
    /// [`Self::with_enum_case_naming_policy`] to make the policy choice
    /// explicit.
    pub fn new() -> Self {
        Self::with_enum_case_naming_policy(SvEnumCaseNamingPolicy::Unqualified)
    }

    /// Creates a builder with an explicit policy for enum member symbol
    /// naming.
    ///
    /// Using [`SvEnumCaseNamingPolicy::Unqualified`] keeps the previous output
    /// shape, while [`SvEnumCaseNamingPolicy::EnumQualified`] prefixes each case
    /// with the containing enum name to avoid cross-enum collisions in the flat
    /// generated SV namespace.
    pub fn with_enum_case_policy(enum_case_naming_policy: SvEnumCaseNamingPolicy) -> Self {
        Self {
            lines: vec![],
            defined: HashSet::new(),
            enum_case_naming_policy,
        }
    }

    /// Returns the generated SV source accumulated so far.
    ///
    /// Callers typically invoke this after `dslx_bridge` conversion has emitted
    /// all reachable type definitions into the builder.
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

    /// Normalizes one enum-name or case-name component using the existing case
    /// conversion rules.
    fn enum_case_name_component_to_sv(dslx_name: &str) -> String {
        if is_screaming_snake_case(dslx_name) {
            screaming_snake_to_upper_camel(dslx_name)
        } else {
            dslx_name.to_string()
        }
    }

    /// Computes the emitted SV enum member symbol under the active naming policy.
    ///
    /// Both `enum_name` and `member_name` are normalized with the same helper so
    /// the `EnumQualified` policy composes exactly with the historical
    /// `Unqualified` formatting behavior.
    fn enum_member_name_to_sv(&self, enum_name: &str, member_name: &str) -> String {
        match self.enum_case_naming_policy {
            SvEnumCaseNamingPolicy::Unqualified => {
                Self::enum_case_name_component_to_sv(member_name)
            }
            SvEnumCaseNamingPolicy::EnumQualified => format!(
                "{}_{}",
                Self::enum_case_name_component_to_sv(enum_name),
                Self::enum_case_name_component_to_sv(member_name)
            ),
        }
    }
}

impl Default for SvBridgeBuilder {
    fn default() -> Self {
        Self::new()
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
        let sv_name = enum_name_to_sv(dslx_name);
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
            let sv_member_name = self.enum_member_name_to_sv(dslx_name, member_name);
            self.define_or_error(&sv_member_name, &ctx)?;
            lines.push(format!(
                "    {sv_member_name} = {underlying_bit_count}'d{digits}{maybe_comma}"
            ));
        }
        lines.push(format!("}} {sv_name};\n"));
        self.lines.push(lines.join("\n"));
        Ok(())
    }

    fn add_struct_def(
        &mut self,
        dslx_name: &str,
        members: &[StructMemberData],
    ) -> Result<(), XlsynthError> {
        let mut lines = vec![];
        lines.push("typedef struct packed {".to_string());
        for member in members {
            let member_name = &member.name;

            // Note: this is the type that type inference determined the member is; i.e. it
            // will be something like `BitsType`, `StructType`, `ArrayType`,
            // etc.
            let member_concrete_ty = &member.concrete_type;

            let member_annotated_ty = &member.type_annotation;

            if let Some(extern_ref) = get_extern_type_ref(member_annotated_ty, member_concrete_ty) {
                lines.push(format!("    {extern_ref} {member_name};"));
                continue;
            }

            // If the member_annotated_ty is a local alias, we want to emit the type via
            // its local identifier.
            if let Some(type_ref_type_annotation) =
                member_annotated_ty.to_type_ref_type_annotation()
            {
                let type_ref = type_ref_type_annotation.get_type_ref();
                let type_def = type_ref.get_type_definition();
                if let Some(type_alias) = type_def.to_type_alias() {
                    let sv_type_name = struct_name_to_sv(&type_alias.get_identifier());
                    lines.push(format!("    {sv_type_name} {member_name};"));
                    continue;
                }
            }

            let member_sv_ty = convert_type(member_concrete_ty, None)?;
            lines.push(format!("    {member_sv_ty} {member_name};"));
        }
        lines.push(format!("}} {};\n", struct_name_to_sv(dslx_name)));
        self.lines.push(lines.join("\n"));
        Ok(())
    }

    fn add_alias(
        &mut self,
        dslx_name: &str,
        type_annotation: &dslx::TypeAnnotation,
        ty: &dslx::Type,
    ) -> Result<(), XlsynthError> {
        let sv_name = format!("{}{}", camel_to_snake(dslx_name), TYPE_ALIAS_SUFFIX);
        if let Some(extern_ref) = get_extern_type_ref(type_annotation, ty) {
            self.lines
                .push(format!("typedef {extern_ref} {sv_name};\n"));
        } else {
            let sv_ty = convert_type(ty, None)?;
            self.lines.push(format!("typedef {sv_ty} {sv_name};\n"));
        }
        Ok(())
    }

    fn add_constant(
        &mut self,
        name: &str,
        _constant_def: &dslx::ConstantDef,
        ty: &dslx::Type,
        ir_value: &IrValue,
    ) -> Result<(), XlsynthError> {
        if let Some((is_signed, bit_count)) = ty.is_bits_like() {
            let sv_name = if is_screaming_snake_case(name) {
                screaming_snake_to_upper_camel(name)
            } else {
                name.to_string()
            };
            let hex_prefix = if is_signed { "sh" } else { "h" };
            let hex_digits = ir_value
                .to_string_fmt_no_prefix(IrFormatPreference::ZeroPaddedHex)?
                .replace("_", "");

            let value_str = format!("{bit_count}'{hex_prefix}{hex_digits}",);
            self.lines.push(format!(
                "localparam bit {signedness} [{}:0] {name} = {value_str};\n",
                bit_count - 1,
                name = sv_name,
                signedness = if is_signed { "signed" } else { "unsigned" }
            ));
            Ok(())
        } else {
            log::warn!("Unsupported constant type: {ir_value:?}");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::dslx_bridge::{convert_imported_module, convert_leaf_module};

    use super::*;

    /// Reusable scaffolding for converting a single DSLX module contents to SV.
    fn simple_convert_for_test(dslx: &str) -> Result<String, XlsynthError> {
        simple_convert_for_test_with_policy(dslx, SvEnumCaseNamingPolicy::Unqualified)
    }

    fn simple_convert_for_test_with_policy(
        dslx: &str,
        enum_case_naming_policy: SvEnumCaseNamingPolicy,
    ) -> Result<String, XlsynthError> {
        let mut import_data = dslx::ImportData::default();
        let path = std::path::PathBuf::from_str("/memfile/my_module.x").unwrap();
        let mut builder = SvBridgeBuilder::with_enum_case_policy(enum_case_naming_policy);
        convert_leaf_module(&mut import_data, dslx, &path, &mut builder)?;
        Ok(builder.build())
    }

    #[test]
    fn test_type_alias_of_u64_array() {
        let dslx = "type MyType = u64[4];";
        let sv = simple_convert_for_test(dslx).unwrap();
        assert_eq!(sv, "typedef logic [3:0] [63:0] my_type_t;\n");
        xlsynth_test_helpers::assert_valid_sv(&sv);
    }

    /// Demonstrates that we do not change the case of enum members that are
    /// defined as UpperCamelCase in DSLX.
    #[test]
    fn test_convert_leaf_module_enum_def_only() {
        let dslx = r#"
        enum OpType : u2 { Read = 0, Write = 1 }
        "#;
        let sv = simple_convert_for_test(dslx).unwrap();
        xlsynth_test_helpers::assert_valid_sv(&sv);
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
        xlsynth_test_helpers::assert_valid_sv(&sv);
        assert_eq!(
            sv,
            r#"typedef enum logic [1:0] {
    MyFirstValue = 2'd0,
    MySecondValue = 2'd1
} my_enum_t;
"#
        );
    }

    // Verifies: EnumQualified prefixes normalized enum names onto normalized
    // case names in emitted SV.
    // Catches: Regressions where enum-qualified mode reuses the unqualified
    // symbol path or skips normalization.
    #[test]
    fn test_convert_leaf_module_enum_def_enum_qualified_case_names() {
        let dslx = r#"
        enum MyEnum : u2 { MY_FIRST_VALUE = 0, MY_SECOND_VALUE = 1 }
        "#;
        let sv = simple_convert_for_test_with_policy(dslx, SvEnumCaseNamingPolicy::EnumQualified)
            .unwrap();
        xlsynth_test_helpers::assert_valid_sv(&sv);
        assert_eq!(
            sv,
            r#"typedef enum logic [1:0] {
    MyEnum_MyFirstValue = 2'd0,
    MyEnum_MySecondValue = 2'd1
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
        xlsynth_test_helpers::assert_valid_sv(&sv);
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

    // Verifies: EnumQualified allows two DSLX enums to reuse member names
    // without SV symbol collisions.
    // Catches: Regressions where collision detection still sees flat
    // unqualified names under EnumQualified mode.
    #[test]
    fn test_convert_leaf_module_enum_defs_with_enum_qualified_case_names_no_collision() {
        let dslx = "enum MyFirstEnum : u1 { A = 0, B = 1 }
        enum MySecondEnum: u3 { A = 3, B = 4 }";
        let sv = simple_convert_for_test_with_policy(dslx, SvEnumCaseNamingPolicy::EnumQualified)
            .unwrap();
        xlsynth_test_helpers::assert_valid_sv(&sv);
        assert!(sv.contains("MyFirstEnum_A = 1'd0"));
        assert!(sv.contains("MySecondEnum_A = 3'd3"));
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
