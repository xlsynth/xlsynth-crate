// SPDX-License-Identifier: Apache-2.0

//! Constexpr builtin widths and tuple-parameter array updates.

use crate::converter_contracts::ContractCase;

/// Checks the width of a constexpr ceil-log2 and its effect on concatenation.
pub(crate) fn ceillog2(data: &[u8]) -> ContractCase {
    let byte = |i: usize| data.get(i).copied().unwrap_or_default();
    let widths = [8, 2, 3, 4, 7, 16, 32, 64];
    let width = widths[usize::from(byte(1)) % widths.len()];
    let power = 1u64 << (1 + u32::from(byte(2)) % (width - 1));
    let operand = match byte(3) % 4 {
        0 => power + 1,
        1 => power,
        2 => power - 1,
        _ => 0,
    };
    let log = u64::BITS - operand.saturating_sub(1).leading_zeros();
    let constant = format!("const K = ceillog2(u{width}:{operand});\n");
    let form = byte(0) % 3;
    if form == 0 {
        let source = format!("{constant}pub fn main() -> u{width} {{ K }}\n");
        let expected = format!(
            r#"package expected

top fn main() -> bits[{width}] {{
  ret result: bits[{width}] = literal(value={log}, id=1)
}}
"#
        );
        return ContractCase::single("ceillog2_const_width", source, expected);
    }

    // A leading one observes the operand's width even when the numeric log
    // is correct. An explicit cast to the declared width serves as a control.
    let (family, operand) = if form == 1 {
        ("ceillog2_const_concat", "K".to_string())
    } else {
        ("ceillog2_const_cast_control", format!("(K as u{width})"))
    };
    let output = width + 1;
    let mask = (1u128 << width) | u128::from(log);
    let source = format!(
        r#"{constant}const MASK = (u1:1 ++ {operand}) as uN[{output}];
pub fn main(x: uN[{output}]) -> uN[{output}] {{ x ^ MASK }}
"#
    );
    let expected = format!(
        r#"package expected

top fn main(x: bits[{output}] id=1) -> bits[{output}] {{
  mask: bits[{output}] = literal(value={mask}, id=2)
  ret result: bits[{output}] = xor(x, mask, id=3)
}}
"#
    );
    ContractCase::single(family, source, expected)
}

/// Updates a matrix using a tuple formal parameter or an explicit projection.
pub(crate) fn tuple_update(data: &[u8]) -> ContractCase {
    let byte = |i: usize| data.get(i).copied().unwrap_or_default();
    let (family, index) = if byte(0) & 1 == 0 {
        ("update_tuple_parameter", "i")
    } else {
        ("update_tuple_projection_control", "(i.0, i.1)")
    };
    let rows = 2 + byte(1) % 3;
    let cols = 2 + byte(2) % 3;
    let widths = [8, 1, 4, 16, 32];
    let width = widths[usize::from(byte(3)) % widths.len()];
    let (row_index, col_index) = match byte(4) % 3 {
        0 => (32, 32),
        1 => (2, 2),
        _ => (2, 8),
    };
    let source = format!(
        r#"pub fn main(xs: u{width}[{cols}][{rows}], i: (u{row_index}, u{col_index}), v: u{width}) -> u{width}[{cols}][{rows}] {{
    update(xs, {index}, v)
}}
"#
    );
    let matrix = format!("bits[{width}][{cols}][{rows}]");
    let expected = format!(
        r#"package expected

top fn main(xs: {matrix} id=1, i: (bits[{row_index}], bits[{col_index}]) id=2, v: bits[{width}] id=3) -> {matrix} {{
  row: bits[{row_index}] = tuple_index(i, index=0, id=4)
  col: bits[{col_index}] = tuple_index(i, index=1, id=5)
  ret result: {matrix} = array_update(xs, v, indices=[row, col], id=6)
}}
"#
    );
    ContractCase::single(family, source, expected)
}
