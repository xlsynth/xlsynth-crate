// SPDX-License-Identifier: Apache-2.0

//! Scope of module constants inside the dimension of a function parameter.

use crate::audit_contracts::AuditCase;
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
        shadows_dimension: form == 0,
    }
}

#[cfg(test)]
mod tests {
    use super::generate;
    use crate::audit_contracts::AuditCase;
    use crate::converter_contracts::scalar_identity;
    use std::path::Path;
    use xlsynth::{XlsIrValue, convert_dslx_to_ir};

    #[test]
    fn short_seed_reuses_the_parameter_name_in_its_dimension() {
        let AuditCase::DimensionScope {
            case,
            shadows_dimension,
        } = generate(&[])
        else {
            panic!("expected a parameter dimension case");
        };
        assert!(shadows_dimension);
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
                shadows_dimension,
            } = generate(&[flags, 5, 2])
            else {
                panic!("expected a parameter dimension case");
            };
            assert!(!shadows_dimension);
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
}
