// SPDX-License-Identifier: Apache-2.0

//! Observes the width and signedness of bound literals at control-flow joins
//! and function boundaries.

use crate::converter_contracts::ContractCase;

/// Joins two distinct literals of equal natural width under a wider declared
/// type, then applies an operation whose value depends on the operand width.
pub(crate) fn generate(data: &[u8]) -> ContractCase {
    let byte = |index: usize| data.get(index).copied().unwrap_or_default();
    let flags = byte(0);
    if flags & 0x80 != 0 {
        return signed_widening(flags, byte(1), byte(2), byte(3));
    }
    let natural_width = 2 + u32::from(byte(1) % 7);
    let width = natural_width + [6, 1, 2, 3, 4, 8, 16, 24][usize::from(byte(2) % 8)];
    let high_bit = 1u64 << (natural_width - 1);
    let low_mask = high_bit - 1;
    let left = high_bit | (u64::from(byte(3)) & low_mask);
    let right = left ^ (1 + u64::from(byte(4)) % low_mask);

    // Evaluate each branch directly at the declared width. This reference
    // avoids both DSLX's inferred producer types and its conditional lowering.
    let values = [left, right];
    let (consumer, result_width, [on_true, on_false]) = match (flags >> 3) & 3 {
        0 => ("v + v", width, values.map(|v| 2 * v)),
        1 => ("!v", width, values.map(|v| ((1u64 << width) - 1) ^ v)),
        2 => ("u1:1 ++ v", width + 1, values.map(|v| (1u64 << width) | v)),
        _ => (
            "rev(v)",
            width,
            values.map(|v| v.reverse_bits() >> (u64::BITS - width)),
        ),
    };
    let ty = format!("u{width}");
    let annotation = if flags & 2 == 0 {
        String::new()
    } else {
        format!("{ty}:")
    };
    let (joined_type, then_arm, else_arm, projection) = match (flags >> 5) % 3 {
        0 => (ty.clone(), "a", "b", "joined"),
        1 => (format!("({ty}, {ty})"), "(a, b)", "(b, a)", "joined.0"),
        _ => (format!("{ty}[2]"), "[a, b]", "[b, a]", "joined[u32:0]"),
    };
    let join = if flags & 1 == 0 {
        format!("if x {{ {then_arm} }} else {{ {else_arm} }}")
    } else {
        format!("match x {{ true => {then_arm}, _ => {else_arm} }}")
    };
    let body = format!(
        r#"{{
    let a = {annotation}{left};
    let b = {annotation}{right};
    let joined: {joined_type} = {join};
    let v = {projection};
    ({consumer}) as u{result_width}
}}"#
    );
    let (family, source) = if flags & 4 == 0 {
        (
            "inferred_branch_width",
            format!("pub fn main(x: bool) -> u{result_width} {body}\n"),
        )
    } else {
        (
            "inferred_branch_width_constexpr",
            format!(
                r#"fn observe(x: bool) -> u{result_width} {body}
const ON_TRUE = observe(true);
const ON_FALSE = observe(false);
pub fn main(x: bool) -> u{result_width} {{ if x {{ ON_TRUE }} else {{ ON_FALSE }} }}
"#
            ),
        )
    };
    ContractCase::single(
        family,
        source,
        selected_reference(result_width, on_true, on_false),
    )
}

/// Observes a high-bit pattern constrained by a signed match or helper return.
/// Converting that signed value to a wider unsigned result must extend its
/// sign.
fn signed_widening(
    flags: u8,
    width_choice: u8,
    extension_choice: u8,
    low_bits: u8,
) -> ContractCase {
    let width = [8, 2, 3, 4, 5, 6, 7, 16][usize::from(width_choice % 8)];
    let result_width = width + [8, 1, 2, 3, 7, 16, 24, 32][usize::from(extension_choice % 8)];
    let modulus = 1u64 << width;
    let bits = (modulus - 1) ^ (u64::from(low_bits) & (modulus / 2 - 1));
    let negative = bits as i64 - modulus as i64;
    let widened = (1u64 << result_width) - (modulus - bits);
    let pattern = if flags & 8 == 0 {
        format!("0x{bits:x}")
    } else {
        format!("0b{bits:b}")
    };
    let literal = if flags & 2 == 0 {
        pattern
    } else {
        format!("s{width}:{pattern}")
    };
    if flags & 1 != 0 {
        let source = format!(
            r#"fn signed_value() -> s{width} {{ let a = {literal}; a }}
const K = signed_value() as u{result_width};
pub fn main(x: bool) -> (u{result_width}, u{result_width}) {{
    if x {{ (signed_value() as u{result_width}, K) }} else {{ (u{result_width}:0, u{result_width}:0) }}
}}
"#
        );
        let ty = format!("(bits[{result_width}], bits[{result_width}])");
        let ir = format!(
            r#"package expected
top fn main(x: bits[1] id=1) -> {ty} {{
  on_true: {ty} = literal(value=({widened}, {widened}), id=2)
  on_false: {ty} = literal(value=(0, 0), id=3)
  ret result: {ty} = sel(x, cases=[on_false, on_true], id=4)
}}
"#
        );
        return ContractCase::single("inferred_signed_return_widen", source, ir);
    }
    // Bare inclusive ranges can prevent constant evaluation altogether. The
    // constexpr variant uses an exact pattern so it still produces data.
    let folded = flags & 4 != 0;
    let arm = if flags & 16 != 0 && !folded {
        let minimum = -(modulus as i64 / 2);
        format!("s{width}:{minimum}..=s{width}:{negative}")
    } else {
        format!("s{width}:{negative}")
    };
    let body = format!(
        "{{ let a = {literal}; match a {{ {arm} => a as u{result_width}, _ => u{result_width}:0 }} }}"
    );
    let source = if !folded {
        format!(
            "pub fn main(x: bool) -> u{result_width} {{ if x {{ {body} }} else {{ u{result_width}:0 }} }}\n"
        )
    } else {
        format!(
            "const K = {body};\npub fn main(x: bool) -> u{result_width} {{ if x {{ K }} else {{ u{result_width}:0 }} }}\n"
        )
    };
    ContractCase::single(
        "inferred_signed_match_widen",
        source,
        selected_reference(result_width, widened, 0),
    )
}

/// Selects between the two independently computed answers in Boolean input
/// order.
fn selected_reference(result_width: u32, on_true: u64, on_false: u64) -> String {
    format!(
        r#"package expected
top fn main(x: bits[1] id=1) -> bits[{result_width}] {{
  on_true: bits[{result_width}] = literal(value={on_true}, id=2)
  on_false: bits[{result_width}] = literal(value={on_false}, id=3)
  ret result: bits[{result_width}] = sel(x, cases=[on_false, on_true], id=4)
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::generate;
    use std::fmt::Write;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn branch_consumers_agree_with_typed_controls_on_both_inputs()
    -> Result<(), Box<dyn std::error::Error>> {
        // The default bindings are 2 and 3. Both consume two natural bits.
        // Their selected value must instead be observed at the declared u8.
        for (op, width, on_true, on_false) in [
            (0, 8, 4, 6),
            (1, 8, 253, 252),
            (2, 9, 258, 259),
            (3, 8, 64, 192),
        ] {
            for shape in 0..3 {
                for context in [0, 1, 4, 5] {
                    let flags = (op << 3) | (shape << 5) | context;
                    let bare = generate(&[flags]);
                    let control = generate(&[flags | 2]);
                    assert_eq!(bare.expected_ir, control.expected_ir);
                    assert_eq!(
                        bare.modules[0]
                            .text
                            .replace("a = 2;", "a = u8:2;")
                            .replace("b = 3;", "b = u8:3;"),
                        control.modules[0].text
                    );
                    let package = IrPackage::parse_ir(&bare.expected_ir, None)?;
                    let function = package.get_function("main")?;
                    let mut source = control.modules[0].text.clone();
                    for (x, value) in [(true, on_true), (false, on_false)] {
                        writeln!(source, "const_assert!(main({x}) == u{width}:{value});")?;
                        assert_eq!(
                            function.interpret(&[XlsIrValue::make_ubits(1, u64::from(x))?])?,
                            XlsIrValue::make_ubits(width, value)?
                        );
                    }
                    parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default())?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn signed_match_and_return_controls_sign_extend_in_both_contexts()
    -> Result<(), Box<dyn std::error::Error>> {
        // s8:0xf8 represents -8, so its u16 representation is 65528.
        for flags in [0x80, 0x84, 0x90, 0x9c, 0x81, 0x89] {
            let bare = generate(&[flags, 0, 0, 7]);
            let control = generate(&[flags | 2, 0, 0, 7]);
            assert_eq!(bare.expected_ir, control.expected_ir);
            let package = IrPackage::parse_ir(&bare.expected_ir, None)?;
            let function = package.get_function("main")?;
            let mut source = control.modules[0].text.clone();
            for (x, value) in [(true, 65528), (false, 0)] {
                let word = XlsIrValue::make_ubits(16, value)?;
                let (expected, dslx_value) = if flags & 1 == 0 {
                    (word, format!("u16:{value}"))
                } else {
                    (
                        XlsIrValue::make_tuple(&[word.clone(), word]),
                        format!("(u16:{value}, u16:{value})"),
                    )
                };
                assert_eq!(function.interpret(&[x.into()])?, expected);
                writeln!(source, "const_assert!(main({x}) == {dslx_value});")?;
            }
            parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default())?;
        }
        Ok(())
    }
}
