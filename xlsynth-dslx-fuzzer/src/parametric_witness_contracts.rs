// SPDX-License-Identifier: Apache-2.0

//! Equivalent bit-type spellings used to infer a struct's width parametric.

use crate::{SourceModule, audit_contracts::AuditCase, converter_contracts::ContractCase};

/// Observes the field, its complement and the inferred width after struct
/// construction, or checks that an unrepresentable inferred width is rejected.
pub(crate) fn generate(data: &[u8]) -> AuditCase {
    let flags = data.first().copied().unwrap_or_default();
    let widths = [5_u32, 1, 3, 7, 8, 16, 31];
    let width = widths[usize::from(data.get(1).copied().unwrap_or_default()) % widths.len()];
    let overflows = flags & 64 != 0;
    let width = if overflows { width.max(2) } else { width };
    // A short witness reaches the numeric range check. Long spellings can
    // fail an earlier type check, even for widths that fit.
    let spelling = if overflows { 0 } else { flags & 3 };
    let witness = match spelling {
        0 => format!("u{width}"),
        1 => format!("uN[{width}]"),
        2 => format!("xN[false][{width}]"),
        _ => format!("uN[u32:{width}]"),
    };
    let bound_width = if overflows {
        width.ilog2()
    } else if flags & 4 != 0 {
        32
    } else if flags & 32 != 0 {
        width.ilog2() + 1
    } else {
        5
    };
    let definition = format!("struct Pack<N: u{bound_width}> {{ value: uN[N] }}\n");
    let mut modules = Vec::new();
    let (prefix, qualifier) = if flags & 8 == 0 {
        (definition, "")
    } else {
        modules.push(SourceModule {
            name: "packed".into(),
            text: format!("pub {definition}"),
        });
        ("import packed;\n".into(), "packed::")
    };
    let parametric = if overflows || flags & 16 == 0 {
        String::new()
    } else {
        format!("<u{bound_width}:{width}>")
    };
    modules.push(SourceModule {
        name: "sample".into(),
        text: format!(
            r#"{prefix}fn read_width<N: u{bound_width}>(p: {qualifier}Pack<N>) -> u32 {{ N as u32 }}

pub fn main(x: {witness}) -> (u{width}, u32, u32) {{
    let p = {qualifier}Pack{parametric} {{ value: x }};
    (p.value, (!p.value) as u32, read_width(p))
}}
"#
        ),
    });
    if overflows {
        return AuditCase::ExpectedTypeRejection {
            family: "struct_parametric_width_range",
            modules,
        };
    }
    let expected_ir = format!(
        r#"package expected

top fn main(x: bits[{width}] id=1) -> (bits[{width}], bits[32], bits[32]) {{
  inverted: bits[{width}] = not(x, id=2)
  widened: bits[32] = zero_ext(inverted, new_bit_count=32, id=3)
  inferred_width: bits[32] = literal(value={width}, id=4)
  ret result: (bits[{width}], bits[32], bits[32]) = tuple(x, widened, inferred_width, id=5)
}}
"#
    );
    AuditCase::Semantic(ContractCase {
        family: "struct_parametric_width_witness",
        modules,
        expected_ir,
        max_and_nodes: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{AuditCase, generate};

    #[test]
    fn equivalent_witness_spellings_use_the_same_width_and_value_oracle() {
        let AuditCase::Semantic(canonical) = generate(&[]) else {
            panic!("a fitting inferred width should be proved");
        };
        for (flags, witness) in ["u5", "uN[5]", "xN[false][5]", "uN[u32:5]"]
            .into_iter()
            .enumerate()
        {
            let AuditCase::Semantic(case) = generate(&[flags as u8]) else {
                panic!("equivalent spellings of a fitting width should be proved");
            };
            assert_eq!(
                case.modules[0].text,
                format!(
                    r#"struct Pack<N: u5> {{ value: uN[N] }}
fn read_width<N: u5>(p: Pack<N>) -> u32 {{ N as u32 }}

pub fn main(x: {witness}) -> (u5, u32, u32) {{
    let p = Pack {{ value: x }};
    (p.value, (!p.value) as u32, read_width(p))
}}
"#
                )
            );
            assert_eq!(case.expected_ir, canonical.expected_ir);
        }
    }

    #[test]
    fn explicit_imported_constructor_uses_the_generated_bound_width() {
        let AuditCase::Semantic(case) = generate(&[1 | 8 | 16 | 32, 5]) else {
            panic!("an explicitly supplied fitting width should be proved");
        };
        assert_eq!(
            case.modules[0].text,
            "pub struct Pack<N: u5> { value: uN[N] }\n"
        );
        assert_eq!(
            case.modules[1].text,
            r#"import packed;
fn read_width<N: u5>(p: packed::Pack<N>) -> u32 { N as u32 }

pub fn main(x: uN[16]) -> (u16, u32, u32) {
    let p = packed::Pack<u5:16> { value: x };
    (p.value, (!p.value) as u32, read_width(p))
}
"#
        );
        assert_eq!(
            case.expected_ir,
            r#"package expected

top fn main(x: bits[16] id=1) -> (bits[16], bits[32], bits[32]) {
  inverted: bits[16] = not(x, id=2)
  widened: bits[32] = zero_ext(inverted, new_bit_count=32, id=3)
  inferred_width: bits[32] = literal(value=16, id=4)
  ret result: (bits[16], bits[32], bits[32]) = tuple(x, widened, inferred_width, id=5)
}
"#
        );
        let AuditCase::Semantic(inferred) = generate(&[1 | 4 | 8, 5]) else {
            panic!("a width inferred as u32 should be proved");
        };
        assert_eq!(
            inferred.modules[0].text,
            "pub struct Pack<N: u32> { value: uN[N] }\n"
        );
        assert_eq!(inferred.expected_ir, case.expected_ir);
    }

    #[test]
    fn width_overflow_tests_inference_with_one_too_few_bits() {
        let AuditCase::ExpectedTypeRejection { family, modules } = generate(&[1 | 4 | 16 | 64])
        else {
            panic!("an overflowing inferred width should be rejected");
        };
        assert_eq!(family, "struct_parametric_width_range");
        let AuditCase::Semantic(fitting) = generate(&[32]) else {
            panic!("the same width fits with one more bit");
        };
        assert_eq!(
            modules[0].text,
            fitting.modules[0]
                .text
                .replace("Pack<N: u3>", "Pack<N: u2>")
                .replace("read_width<N: u3>", "read_width<N: u2>")
        );
    }
}
