// SPDX-License-Identifier: Apache-2.0

//! Scope of module constants inside parameter and value-parametric dimensions.

use crate::audit_contracts::{AuditCase, DimensionDiagnostic};
use crate::converter_contracts::{ContractCase, scalar_identity};

/// Vary whether the parameter shadows its own dimension. If a frontend
/// resolves the dimension to the module constant, the function is an identity
/// at that constant's width.
pub(crate) fn generate(data: &[u8]) -> AuditCase {
    let flags = data.first().copied().unwrap_or_default();
    let widths = [8, 1, 5, 16, 32, 65];
    let width = widths[usize::from(data.get(1).copied().unwrap_or_default()) % widths.len()];
    let names = ["N", "WIDTH", "BIT_COUNT"];
    let name = names[usize::from(data.get(2).copied().unwrap_or_default()) % names.len()];
    let sign = if flags & 4 == 0 { "u" } else { "s" };
    let result_type = if width <= 64 {
        format!("{sign}{width}")
    } else {
        format!("{sign}N[{width}]")
    };
    let form = flags % 4;
    let parameter = if form == 1 { "data" } else { name };
    let dimension = match form {
        2 => width.to_string(),
        3 => "OUTER_WIDTH".to_owned(),
        _ => name.to_owned(),
    };
    let mut source = format!("const {name} = u32:{width};\n");
    if form == 3 {
        source.push_str(&format!("const OUTER_WIDTH = {name};\n"));
    }
    source.push_str(&format!(
        "\npub fn main({parameter}: {sign}N[{dimension}]) -> {result_type} {{\n    {parameter}\n}}\n"
    ));
    AuditCase::DimensionScope {
        case: ContractCase::single("parameter_dimension_scope", source, scalar_identity(width)),
        allowed_rejection: (form == 0).then_some(DimensionDiagnostic::Type),
    }
}

/// Specializes a function or struct whose value-parametric width may refer to
/// a module constant with the same name. Unambiguous controls must lower.
pub(crate) fn parametric(data: &[u8]) -> AuditCase {
    let flags = data.first().copied().unwrap_or_default();
    let widths = [8, 4, 5, 16, 32, 65];
    let width = widths[usize::from(data.get(1).copied().unwrap_or_default()) % widths.len()];
    let names = ["N", "WIDTH", "BIT_COUNT"];
    let name = names[usize::from(data.get(2).copied().unwrap_or_default()) % names.len()];
    let form = flags & 3;
    let parametric = if form == 2 { "K" } else { name };
    let dimension = match form {
        1 => format!("{name} + u32:1"),
        3 => width.to_string(),
        _ => name.to_owned(),
    };
    let value = if flags & 8 == 0 { 8 } else { 3 };
    let binding = format!("{parametric}: uN[{dimension}]");
    let (source, expected_ir) = if flags & 4 == 0 {
        let source = format!(
            r#"const {name} = u32:{width};
fn value<{binding}>() -> u32 {{ {parametric} as u32 }}
pub fn main(x: u32) -> u32 {{ x ^ value<{value}>() }}
"#
        );
        let expected_ir = format!(
            r#"package expected

top fn main(x: bits[32] id=1) -> bits[32] {{
  argument: bits[32] = literal(value={value}, id=2)
  ret result: bits[32] = xor(x, argument, id=3)
}}
"#
        );
        (source, expected_ir)
    } else {
        let source = format!(
            r#"const {name} = u32:{width};
struct Word<{binding}> {{ value: uN[{parametric} as u32] }}
pub fn main(x: u{value}) -> u{value} {{ Word<{value}> {{ value: x }}.value }}
"#
        );
        (source, scalar_identity(value))
    };
    AuditCase::DimensionScope {
        case: ContractCase::single("parametric_dimension_scope", source, expected_ir),
        allowed_rejection: (form < 2).then_some(DimensionDiagnostic::TypeOrNonconstant),
    }
}

#[cfg(test)]
mod tests {
    use super::{generate, parametric};
    use crate::audit_contracts::{AuditCase, DimensionDiagnostic};
    use crate::converter_contracts::scalar_identity;
    use std::path::Path;
    use xlsynth::{XlsIrValue, convert_dslx_to_ir};

    #[test]
    fn short_seed_reuses_the_parameter_name_in_its_dimension() {
        let AuditCase::DimensionScope {
            case,
            allowed_rejection,
        } = generate(&[])
        else {
            panic!("expected a parameter dimension case");
        };
        assert_eq!(allowed_rejection, Some(DimensionDiagnostic::Type));
        assert_eq!(
            case.modules[0].text,
            r#"const N = u32:8;

pub fn main(N: uN[N]) -> u8 {
    N
}
"#
        );
        assert_eq!(case.expected_ir, scalar_identity(8));
    }

    #[test]
    fn unambiguous_dimension_controls_preserve_the_runtime_parameter()
    -> Result<(), Box<dyn std::error::Error>> {
        let input = XlsIrValue::parse_typed("bits[65]:0x1_0000_0000_0000_0001")?;
        for flags in [1, 2, 3, 5, 6, 7] {
            let AuditCase::DimensionScope {
                case,
                allowed_rejection,
            } = generate(&[flags, 5, 2])
            else {
                panic!("expected a parameter dimension case");
            };
            assert_eq!(allowed_rejection, None);
            assert_eq!(case.expected_ir, scalar_identity(65));
            let result = convert_dslx_to_ir(
                &case.modules[0].text,
                Path::new("sample.x"),
                &Default::default(),
            )?;
            assert_eq!(
                result
                    .ir
                    .get_function("__sample__main")?
                    .interpret(std::slice::from_ref(&input))?,
                input
            );
        }
        Ok(())
    }

    #[test]
    fn short_seed_specializes_a_parametric_with_its_own_width() {
        let AuditCase::DimensionScope {
            case,
            allowed_rejection,
        } = parametric(&[])
        else {
            panic!("expected a parametric dimension case");
        };
        assert_eq!(
            allowed_rejection,
            Some(DimensionDiagnostic::TypeOrNonconstant)
        );
        assert_eq!(
            case.modules[0].text,
            r#"const N = u32:8;
fn value<N: uN[N]>() -> u32 { N as u32 }
pub fn main(x: u32) -> u32 { x ^ value<8>() }
"#
        );
    }
}
