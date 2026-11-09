// SPDX-License-Identifier: Apache-2.0

//! Value-parametric calls whose constant argument depends only on the live arm
//! of an `if`, even when the other arm mentions a runtime input.

use crate::converter_contracts::ContractCase;

/// Vary which arm is dead and whether arithmetic consumes the conditional
/// directly or through a binding. Every form must add the selected constant.
pub(crate) fn generate(data: &[u8]) -> ContractCase {
    let flags = data.first().copied().unwrap_or_default();
    let width = [32, 8, 16, 4][usize::from(data.get(1).copied().unwrap_or_default() % 4)];
    let mask = (1u64 << width) - 1;
    let value = u64::from(data.get(2).copied().unwrap_or(7)) & mask;
    let addend = u64::from(data.get(3).copied().unwrap_or_default()) & mask;
    let form = flags & 3;
    // The bare conditional is a control even when the outer addition has a
    // nonzero addend. Fold that addition into its live arm independently.
    let expected = (value + addend) & mask;
    let live = format!("u{width}:{}", if form == 1 { expected } else { value });
    let dead = if flags & 8 == 0 {
        "x".to_string()
    } else {
        format!("u{width}:{}", mask ^ value)
    };
    let conditional = if flags & 4 == 0 {
        format!("if true {{ {live} }} else {{ {dead} }}")
    } else {
        format!("if false {{ {dead} }} else {{ {live} }}")
    };
    let literal = format!("u{width}:{addend}");
    let (binding, argument) = match form {
        0 => (String::new(), format!("({conditional}) + {literal}")),
        1 => (String::new(), conditional),
        2 => (
            format!("    let selected = {conditional};\n"),
            format!("selected + {literal}"),
        ),
        _ => (String::new(), format!("{literal} + ({conditional})")),
    };
    let source = format!(
        r#"fn bias<K: u{width}>(x: u{width}) -> u{width} {{ x + K }}
pub fn main(x: u{width}) -> u{width} {{
{binding}    bias<{{{argument}}}>(x)
}}
"#
    );
    let reference = format!(
        r#"package expected

top fn main(x: bits[{width}] id=1) -> bits[{width}] {{
  bias: bits[{width}] = literal(value={expected}, id=2)
  ret result: bits[{width}] = add(x, bias, id=3)
}}
"#
    );
    ContractCase::single("constexpr_dead_branch_parametric", source, reference)
}

#[cfg(test)]
mod tests {
    use super::generate;
    use std::path::Path;
    use xlsynth::{IrPackage, XlsIrValue, convert_dslx_to_ir};

    #[test]
    fn default_source_wraps_the_conditional_with_zero() {
        assert_eq!(
            generate(&[]).modules[0].text,
            r#"fn bias<K: u32>(x: u32) -> u32 { x + K }
pub fn main(x: u32) -> u32 {
    bias<{(if true { u32:7 } else { x }) + u32:0}>(x)
}
"#
        );
    }

    #[test]
    fn passing_controls_agree_with_the_oracle_through_wraparound()
    -> Result<(), Box<dyn std::error::Error>> {
        // The inner sum is 253 + 10 = 7 modulo 256. Check both polarities of
        // the conditional, constant-only arms, and a bound live value.
        let inline = generate(&[0, 1, 253, 10]);
        let reference = IrPackage::parse_ir(&inline.expected_ir, None)?;
        let oracle = reference.get_function("main")?;
        for flags in [1, 2, 5, 6, 8, 11, 12, 15] {
            let control = generate(&[flags, 1, 253, 10]);
            assert_eq!(control.expected_ir, inline.expected_ir);
            let result = convert_dslx_to_ir(
                &control.modules[0].text,
                Path::new("sample.x"),
                &Default::default(),
            )?;
            let function = result.ir.get_function("__sample__main")?;
            for (x, expected) in [(0, 7), (248, 255), (249, 0), (255, 6)] {
                let inputs = [XlsIrValue::make_ubits(8, x)?];
                let expected = XlsIrValue::make_ubits(8, expected)?;
                assert_eq!(oracle.interpret(&inputs)?, expected);
                assert_eq!(function.interpret(&inputs)?, expected);
            }
        }
        Ok(())
    }
}
