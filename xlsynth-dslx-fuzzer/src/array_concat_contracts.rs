// SPDX-License-Identifier: Apache-2.0

//! Element widths at annotated array concatenation and width-sensitive uses.

use crate::converter_contracts::ContractCase;

/// Observes a declared element width in the result type or a width-sensitive
/// value.
pub(crate) fn generate(data: &[u8]) -> ContractCase {
    let byte = |index: usize| data.get(index).copied().unwrap_or_default();
    let flags = byte(0);
    let signed = flags & 16 != 0;
    let natural_width = 2 + byte(1) % 6;
    let widths = [8, 16, 32, 64, 65];
    let width = if flags & 2 != 0 && !signed {
        natural_width
    } else {
        widths[usize::from(byte(2)) % widths.len()]
    };
    let left_len = 1 + usize::from(byte(3) % 3);
    let right_len = 1 + usize::from(byte(4) % 3);
    let length = left_len + right_len;
    let high_bit = 1 << (natural_width - 1);
    let values = (0..length)
        .map(|i| {
            let low_bits = data.get(5 + i).copied().unwrap_or(i as u8);
            (high_bit | (low_bits & (high_bit - 1))).to_string()
        })
        .collect::<Vec<_>>();
    let sign = if signed { 's' } else { 'u' };
    let element = if width <= 64 {
        format!("{sign}{width}")
    } else {
        format!("{sign}N[{width}]")
    };
    let annotation = if flags & 1 == 0 {
        String::new()
    } else {
        format!("{element}:")
    };
    let operand = |elements: &[String]| {
        elements
            .iter()
            .map(|value| format!("{annotation}{value}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let expression = if flags & 4 == 0 {
        let (left, right) = values.split_at(left_len);
        format!("[{}] ++ [{}]", operand(left), operand(right))
    } else {
        format!("[{}]", operand(&values))
    };
    if signed {
        let widened = 2 * u32::from(width);
        let source = format!(
            r#"pub fn main() -> sN[{widened}] {{
    let xs: {element}[{length}] = {expression};
    xs[u32:0] as sN[{widened}]
}}
"#
        );
        // The first literal is positive at the declared width. Its highest
        // inferred bit is set, so sign extension from the narrower IR yields a
        // negative value.
        let expected = format!(
            r#"package expected

top fn main() -> bits[{widened}] {{
  ret result: bits[{widened}] = literal(value={}, id=1)
}}
"#,
            values[0],
        );
        return ContractCase::single("array_concat_signed_widening", source, expected);
    }
    if flags & 8 != 0 {
        let packed = 2 * u32::from(width);
        let last = length - 1;
        let source = format!(
            r#"pub fn main() -> uN[{packed}] {{
    let xs: {element}[{length}] = {expression};
    (xs[u32:0] ++ xs[u32:{last}]) as uN[{packed}]
}}
"#
        );
        let expected = format!(
            r#"package expected

top fn main() -> bits[{packed}] {{
  first: bits[{width}] = literal(value={}, id=1)
  last: bits[{width}] = literal(value={}, id=2)
  ret result: bits[{packed}] = concat(first, last, id=3)
}}
"#,
            values[0], values[last],
        );
        return ContractCase::single("array_concat_element_packing", source, expected);
    }
    let source = format!("pub fn main() -> {element}[{length}] {{ {expression} }}\n");
    let ir_type = format!("bits[{width}][{length}]");
    let expected = format!(
        r#"package expected

top fn main() -> {ir_type} {{
  ret result: {ir_type} = literal(value=[{}], id=1)
}}
"#,
        values.join(", "),
    );
    ContractCase::single("array_concat_return_width", source, expected)
}

#[cfg(test)]
mod tests {
    use super::generate;
    use std::path::Path;
    use xlsynth::{IrPackage, XlsIrValue, convert_dslx_to_ir};

    #[test]
    fn controls_keep_element_width_and_order() -> Result<(), Box<dyn std::error::Error>> {
        let bare = generate(&[]);
        let typed = generate(&[1]);
        assert_eq!(
            bare.modules[0].text,
            "pub fn main() -> u8[2] { [2] ++ [3] }\n"
        );
        assert_eq!(bare.expected_ir, typed.expected_ir);
        for (case, value) in [
            (typed, "[bits[8]:2, bits[8]:3]"),
            (generate(&[2]), "[bits[2]:2, bits[2]:3]"),
            (generate(&[4]), "[bits[8]:2, bits[8]:3]"),
            (generate(&[9]), "bits[16]:515"),
            (generate(&[17]), "bits[16]:2"),
            (
                generate(&[1, 0, 4, 1, 0]),
                "[bits[65]:2, bits[65]:3, bits[65]:2]",
            ),
        ] {
            let expected = IrPackage::parse_ir(&case.expected_ir, None)?;
            let actual = convert_dslx_to_ir(
                &case.modules[0].text,
                Path::new("sample.x"),
                &Default::default(),
            )?;
            let value = XlsIrValue::parse_typed(value)?;
            assert_eq!(expected.get_function("main")?.interpret(&[])?, value);
            assert_eq!(
                actual.ir.get_function("__sample__main")?.interpret(&[])?,
                value
            );
        }
        Ok(())
    }
}
