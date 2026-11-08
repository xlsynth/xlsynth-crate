// SPDX-License-Identifier: Apache-2.0

//! Compares bound integer literals whose uses infer a common signed type.

use crate::SourceModule;
use crate::audit_contracts::AuditCase;
use crate::converter_contracts::ContractCase;

/// A prefixed literal supplies a bit pattern. An unprefixed decimal supplies
/// a nonnegative value that must still fit after coercion to signed. The other
/// operand, `-1 - magnitude_minus_one`, requires a sign bit and enough bits
/// for its ones' complement.
fn common_signed_width(positive: u16, magnitude_minus_one: u16, decimal: bool) -> u32 {
    let positive_bits = u16::BITS - positive.leading_zeros();
    let negative_bits = 1 + u16::BITS - magnitude_minus_one.leading_zeros();
    (positive_bits + u32::from(decimal)).max(negative_bits)
}

/// Decodes two independent literals and compares their values at the common
/// width in a function body, module constant or specialization argument.
pub(crate) fn generate(data: &[u8]) -> AuditCase {
    let byte = |i: usize| data.get(i).copied().unwrap_or_default();
    let flags = byte(0);
    let context = byte(1) % 3;
    let radix = byte(2) % 3;
    let positive = u16::from_le_bytes([byte(4) ^ 0xff, byte(6)]);
    let magnitude_minus_one = u16::from_le_bytes([byte(5), byte(7)]);
    let negative = -1 - i32::from(magnitude_minus_one);
    let width = common_signed_width(positive, magnitude_minus_one, radix == 2);

    // All generated operands fit in at most 17 bits. Interpret the positive
    // literal's bit pattern at the comparison width, including its sign bit.
    let unsigned = i32::from(positive);
    let left = if unsigned >= 1 << (width - 1) {
        unsigned - (1 << width)
    } else {
        unsigned
    };
    let (lhs, rhs, a, b) = if flags & 4 == 0 {
        ("lo", "hi", left, negative)
    } else {
        ("hi", "lo", negative, left)
    };
    let (operator, expected) = match byte(3) % 6 {
        0 => ("==", a == b),
        1 => ("!=", a != b),
        2 => ("<", a < b),
        3 => ("<=", a <= b),
        4 => (">", a > b),
        _ => (">=", a >= b),
    };
    let spelling = match radix {
        0 => format!("0x{positive:x}"),
        1 => format!("0b{positive:b}"),
        _ => positive.to_string(),
    };
    let bindings = if flags & 1 == 0 {
        format!("let (lo, hi) = ({spelling}, {negative});")
    } else {
        format!("let lo = {spelling}; let hi = {negative};")
    };
    let render = |cast: bool| {
        let comparison = if cast {
            format!("({lhs} as s{width}) {operator} ({rhs} as s{width})")
        } else {
            format!("{lhs} {operator} {rhs}")
        };
        match context {
            0 => format!("pub fn main() -> bool {{ {bindings} {comparison} }}\n"),
            1 => {
                format!("const K = {{ {bindings} {comparison} }};\npub fn main() -> bool {{ K }}\n")
            }
            _ => format!(
                "fn pick<K: bool>() -> bool {{ K }}\npub fn main() -> bool {{ {bindings} pick<{{ {comparison} }}>() }}\n"
            ),
        }
    };
    let family = [
        "mixed_literal_comparison",
        "mixed_literal_const_comparison",
        "mixed_literal_parametric_comparison",
    ][usize::from(context)];
    let source = render(flags & 2 != 0);
    let with_test = |module: &str| SourceModule {
        name: "sample".into(),
        text: format!(
            "{module}#[test]\nfn literal_runtime() {{ assert_eq(main(), {expected}); }}\n"
        ),
    };
    let expected_ir = format!(
        r#"package expected

top fn main() -> bits[1] {{
  ret result: bits[1] = literal(value={}, id=1)
}}
"#,
        u8::from(expected),
    );
    AuditCase::MixedLiteral {
        raw_runtime_module: with_test(&source),
        runtime_module: with_test(&render(true)),
        case: ContractCase::single(family, source, expected_ir),
    }
}

#[cfg(test)]
mod tests {
    use super::{common_signed_width, generate};
    use crate::audit_contracts::AuditCase;

    #[test]
    fn signed_width_covers_each_decimal_value_and_each_prefixed_bit_pattern() {
        for (positive, magnitude_minus_one, prefixed_width, decimal_width) in [
            (0, 0, 1, 1),
            (1, 0, 1, 2),
            (15, 0, 4, 5),
            (128, 126, 8, 9),
            (255, 127, 8, 9),
            (255, 128, 9, 9),
            (65535, 0, 16, 17),
            (0, 65535, 17, 17),
        ] {
            assert_eq!(
                common_signed_width(positive, magnitude_minus_one, false),
                prefixed_width
            );
            assert_eq!(
                common_signed_width(positive, magnitude_minus_one, true),
                decimal_width
            );
        }
    }

    #[test]
    fn radix_changes_the_reference_value_but_preserves_the_bare_source() {
        for (radix, spelling, width, value) in [
            (0, "0xff", 8, true),
            (1, "0b11111111", 8, true),
            (2, "255", 9, false),
        ] {
            let AuditCase::MixedLiteral {
                case,
                raw_runtime_module,
                runtime_module,
            } = generate(&[0, 1, radix])
            else {
                panic!("expected a mixed-literal comparison");
            };
            let source = format!(
                "const K = {{ let (lo, hi) = ({spelling}, -1); lo == hi }};\npub fn main() -> bool {{ K }}\n"
            );
            let test = format!("#[test]\nfn literal_runtime() {{ assert_eq(main(), {value}); }}\n");
            assert_eq!(case.modules[0].text, source);
            assert_eq!(raw_runtime_module.text, format!("{source}{test}"));
            assert_eq!(
                runtime_module.text,
                raw_runtime_module
                    .text
                    .replace("lo == hi", &format!("(lo as s{width}) == (hi as s{width})"))
            );
            assert_eq!(
                case.expected_ir,
                format!(
                    "package expected\n\ntop fn main() -> bits[1] {{\n  ret result: bits[1] = literal(value={}, id=1)\n}}\n",
                    u8::from(value)
                )
            );
        }
    }

    #[test]
    fn signed_ordering_is_preserved_across_binding_and_specialization_forms() {
        // At s8, 0x80 is -128, which is less than -127. Swapping operands
        // reverses the result even though their natural widths already agree.
        for (reverse, expected) in [(0, true), (4, false)] {
            for context in 0..3 {
                for separate in 0..2 {
                    let flags = reverse | separate;
                    let AuditCase::MixedLiteral {
                        case,
                        runtime_module,
                        ..
                    } = generate(&[flags, context, 0, 2, 0x7f, 126])
                    else {
                        panic!("expected a mixed-literal comparison");
                    };
                    let AuditCase::MixedLiteral {
                        case: control,
                        raw_runtime_module,
                        ..
                    } = generate(&[flags | 2, context, 0, 2, 0x7f, 126])
                    else {
                        panic!("expected a mixed-literal comparison");
                    };
                    let test = format!(
                        "#[test]\nfn literal_runtime() {{ assert_eq(main(), {expected}); }}\n"
                    );
                    assert_eq!(
                        runtime_module.text,
                        format!("{}{test}", control.modules[0].text)
                    );
                    assert_eq!(runtime_module, raw_runtime_module);
                    assert_eq!(case.expected_ir, control.expected_ir);
                }
            }
        }
    }
}
