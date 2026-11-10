// SPDX-License-Identifier: Apache-2.0

//! Tracks the sign bit of a literal through a signed function parameter into
//! both bytecode and converted IR.

use crate::SourceModule;
use crate::audit_contracts::AuditCase;
use crate::converter_contracts::ContractCase;

/// Widen a high-bit-set literal across a concrete or parametric signed call.
pub(crate) fn generate(data: &[u8]) -> AuditCase {
    let byte = |index: usize| data.get(index).copied().unwrap_or_default();
    let flags = byte(0);
    let width = [5, 2, 3, 4, 6, 7, 8, 16][usize::from(byte(1) % 8)];
    let result_width = [32, width + 1, width + 4, 64][usize::from(byte(2) % 4)];
    let sign_bit = 1u64 << (width - 1);
    let bits = sign_bit | (u64::from(data.get(3).copied().unwrap_or(5)) & (sign_bit - 1));
    let literal = if flags & 8 == 0 {
        format!("0x{bits:x}")
    } else {
        format!("0b{bits:b}")
    };
    let (signature, callee) = if flags & 4 == 0 {
        (
            "fn widen<N: u32>(x: sN[N])".into(),
            format!("widen<{width}>"),
        )
    } else {
        (format!("fn widen(x: s{width})"), "widen".into())
    };
    let body = match flags & 3 {
        0 => format!("let x = {literal}; {callee}(x)"),
        1 => format!("let x = s{width}:{literal}; {callee}(x)"),
        2 => format!("{callee}({literal})"),
        _ => format!("let x = {literal}; {callee}(x as s{width})"),
    };
    let source = format!(
        r#"{signature} -> u{result_width} {{ x as u{result_width} }}
pub fn main() -> u{result_width} {{ {body} }}
"#
    );
    // Interpret the narrow bit pattern as a negative integer, then express
    // its residue in the declared return width, independently of either DUT.
    let signed = i128::from(bits) - (1i128 << width);
    let expected = (1i128 << result_width) + signed;
    let reference = format!(
        r#"package expected

top fn main() -> bits[{result_width}] {{
  ret result: bits[{result_width}] = literal(value={expected}, id=1)
}}
"#
    );
    let runtime_module = SourceModule {
        name: "sample".into(),
        text: format!(
            r#"{source}#[test]
fn literal_runtime() {{ assert_eq(main(), u{result_width}:0x{expected:x}); }}
"#
        ),
    };
    AuditCase::BoundCall {
        case: ContractCase::single("bound_signed_call_widen", source, reference),
        runtime_module: Some(runtime_module),
    }
}

#[cfg(test)]
mod tests {
    use super::generate;
    use crate::audit_contracts::AuditCase;
    use std::path::Path;
    use xlsynth::{IrPackage, XlsIrValue, convert_dslx_to_ir};

    #[test]
    fn raw_bytecode_observes_the_original_parametric_call() {
        let AuditCase::BoundCall {
            case,
            runtime_module: Some(runtime),
        } = generate(&[])
        else {
            panic!("expected a bound-call runtime comparison");
        };
        let source = r#"fn widen<N: u32>(x: sN[N]) -> u32 { x as u32 }
pub fn main() -> u32 { let x = 0x15; widen<5>(x) }
"#;
        assert_eq!(case.modules[0].text, source);
        assert_eq!(
            runtime.text,
            format!(
                "{source}#[test]\nfn literal_runtime() {{ assert_eq(main(), u32:0xfffffff5); }}\n"
            )
        );
    }

    #[test]
    fn signed_call_controls_agree_with_the_independent_literal()
    -> Result<(), Box<dyn std::error::Error>> {
        for (width_choice, result_choice, low_bits, result_width, expected) in [
            (0, 0, 5, 32, 0xffff_fff5),
            (1, 1, 0, 3, 6),
            (7, 3, 255, 64, 0xffff_ffff_ffff_80ff),
        ] {
            let expected = XlsIrValue::make_ubits(result_width, expected)?;
            for flags in [1, 2, 3, 13, 14, 15] {
                let AuditCase::BoundCall {
                    case,
                    runtime_module: Some(runtime),
                } = generate(&[flags, width_choice, result_choice, low_bits])
                else {
                    panic!("expected a bound-call runtime comparison");
                };
                assert!(runtime.text.starts_with(&case.modules[0].text));
                let oracle = IrPackage::parse_ir(&case.expected_ir, None)?;
                let native = convert_dslx_to_ir(
                    &case.modules[0].text,
                    Path::new("sample.x"),
                    &Default::default(),
                )?;
                assert_eq!(oracle.get_function("main")?.interpret(&[])?, expected);
                assert_eq!(
                    native.ir.get_function("__sample__main")?.interpret(&[])?,
                    expected
                );
            }
        }
        Ok(())
    }
}
