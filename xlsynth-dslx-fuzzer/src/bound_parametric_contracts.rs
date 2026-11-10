// SPDX-License-Identifier: Apache-2.0

//! Inferred literal widths at data and value argument positions of free
//! functions.

use crate::audit_contracts::AuditCase;
use crate::converter_contracts::ContractCase;

/// The two argument positions obtain their expected width from different parts
/// of the callee declaration.
#[derive(Clone, Copy)]
enum ArgumentPosition {
    Data,
    Value,
}

/// Checks that a specialization accepts a data operand at its declared width.
pub(crate) fn data_argument(data: &[u8]) -> AuditCase {
    generate(ArgumentPosition::Data, data)
}

/// Observes the width and value of a specialization's parameter directly in
/// the public return type.
pub(crate) fn value_argument(data: &[u8]) -> AuditCase {
    generate(ArgumentPosition::Value, data)
}

/// Varies the literal's width, value, spelling and the location of its type
/// context while keeping the callee and the public result type fixed.
fn generate(position: ArgumentPosition, data: &[u8]) -> AuditCase {
    let byte = |i: usize, default: u8| data.get(i).copied().unwrap_or(default);
    let (family, name, widths, default_value) = match position {
        ArgumentPosition::Data => (
            "bound_parametric_data_argument",
            "x",
            [8, 3, 7, 16, 32, 64],
            3,
        ),
        ArgumentPosition::Value => (
            "bound_parametric_value_argument",
            "n",
            [32, 3, 7, 8, 16, 64],
            4,
        ),
    };
    let width: u32 = widths[usize::from(byte(1, 0)) % widths.len()];
    // Both direct and bound literals are representable at every declared
    // width. A naturally narrower literal exercises contextual inference.
    let value = u32::from(byte(2, default_value)) & ((1 << width.min(8)) - 1);
    let literal = match byte(3, 0) % 3 {
        0 => value.to_string(),
        1 => format!("0x{value:x}"),
        _ => format!("0b{value:b}"),
    };
    let (binding, argument) = match byte(0, 0) % 4 {
        0 => (format!("let {name} = {literal};\n    "), name),
        1 => (format!("let {name} = u{width}:{literal};\n    "), name),
        2 => (String::new(), literal.as_str()),
        _ => (format!("let {name}: u{width} = {literal};\n    "), name),
    };
    let (helper, call) = match position {
        ArgumentPosition::Data => (
            "fn identity<N: u32>(x: uN[N]) -> uN[N] { x }".to_owned(),
            format!("identity<{width}>({argument})"),
        ),
        ArgumentPosition::Value => (
            format!("fn value<N: u{width}>() -> u{width} {{ N }}"),
            format!("value<{argument}>()"),
        ),
    };
    let source = format!(
        r#"{helper}
pub fn main() -> u{width} {{
    {binding}{call}
}}
"#
    );
    let reference = format!(
        r#"package expected

top fn main() -> bits[{width}] {{
  ret result: bits[{width}] = literal(value={value}, id=1)
}}
"#
    );
    AuditCase::BoundCall {
        case: ContractCase::single(family, source, reference),
        runtime_module: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{data_argument, value_argument};
    use crate::audit_contracts::AuditCase;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn short_seeds_reach_both_unannotated_calls() {
        let AuditCase::BoundCall { case: data, .. } = data_argument(&[]) else {
            panic!("expected a bound data argument contract");
        };
        assert_eq!(
            data.modules[0].text,
            r#"fn identity<N: u32>(x: uN[N]) -> uN[N] { x }
pub fn main() -> u8 {
    let x = 3;
    identity<8>(x)
}
"#
        );
        let AuditCase::BoundCall { case: value, .. } = value_argument(&[]) else {
            panic!("expected a bound value argument contract");
        };
        assert_eq!(
            value.modules[0].text,
            r#"fn value<N: u32>() -> u32 { N }
pub fn main() -> u32 {
    let n = 4;
    value<n>()
}
"#
        );
    }

    #[test]
    fn controls_share_the_full_width_literal_oracle() -> Result<(), Box<dyn std::error::Error>> {
        for (generate, width, value) in [
            (data_argument as fn(&[u8]) -> AuditCase, 8, 3),
            (value_argument, 32, 4),
        ] {
            let AuditCase::BoundCall { case: original, .. } = generate(&[]) else {
                panic!("expected a bound call contract");
            };
            for form in 0..4 {
                for spelling in 0..3 {
                    let AuditCase::BoundCall { case, .. } = generate(&[form, 0, value, spelling])
                    else {
                        panic!("expected a bound call contract");
                    };
                    assert_eq!(case.expected_ir, original.expected_ir);
                    if form != 0 {
                        let checked = format!(
                            "{}const_assert!(main() == u{width}:{value});\n",
                            case.modules[0].text
                        );
                        parse_and_typecheck(
                            &checked,
                            "sample.x",
                            "sample",
                            &mut ImportData::default(),
                        )?;
                    }
                }
            }
            let package = IrPackage::parse_ir(&original.expected_ir, None)?;
            assert_eq!(
                package.get_function("main")?.interpret(&[])?,
                XlsIrValue::make_ubits(width, u64::from(value))?,
            );
        }
        Ok(())
    }
}
