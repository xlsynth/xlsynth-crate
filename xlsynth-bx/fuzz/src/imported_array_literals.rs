// SPDX-License-Identifier: Apache-2.0

//! Compare the inference of array operands at imported DSLX function calls.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{FunctionProgram, compare_imported_program};
use crate::input_seed;
use xlsynth_pir::{IrBits, IrFormatPreference};

/// Pass a repeated binding directly, through array lets, or through a tuple
/// pattern.
pub fn check_imported_array_literals(data: &[u8]) -> Result<(), String> {
    let [form, kind, width, length, flags, payload] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let width = 1 + usize::from(width);
    let length = 2 + usize::from(length % 4);
    let signed = flags & 1 == 0;
    let element = format!("{}N[{width}]", if signed { 's' } else { 'u' });
    let result = format!("{element}[{length}]");

    // Set the sign bit so the radix determines whether an unsigned spelling
    // can be used as a signed bit pattern at the same width.
    let bits = IrBits::from_lsb_fn(width, |i| i + 1 == width || (payload >> (i % 8)) & 1 != 0);
    let hex = bits.to_string_fmt(IrFormatPreference::Hex, false);
    let decimal = bits.to_string_fmt(IrFormatPreference::UnsignedDecimal, false);
    let kind = kind % 8;
    let binding = match kind {
        1 => format!(
            "let h = {};",
            bits.to_string_fmt(IrFormatPreference::Binary, false)
        ),
        2 => format!("let h = {decimal};"),
        3 => format!("let h: {element} = {hex};"),
        6 => format!("let h: uN[{width}] = {hex};"),
        _ => format!("let h = {hex};"),
    };
    let typed = format!("uN[{width}]:{hex}");
    let mut items = vec!["h"; length];
    items[length - 1] = match kind {
        4 => &decimal,
        5 => &hex,
        7 => &typed,
        _ => "h",
    };
    let array = format!("[{}]", items.join(", "));
    let call = match form % 4 {
        0 => format!("lanes::take({array})"),
        1 => format!("let xs = {array}; lanes::take(xs)"),
        2 => format!("let xs = {array}; let ys = xs; lanes::take(ys)"),
        _ => format!("let (xs, _) = ({array}, flag); lanes::take(xs)"),
    };
    let body = format!("{binding} {call}");
    let params = "flag: bool";
    let library = format!("pub fn take(xs: {result}) -> {result} {{ xs }}");
    let bx = format!("use lanes;\n{}", bx_source(params, &result, &body));
    compare_imported_program(
        "lanes",
        &library,
        &bx,
        FunctionProgram {
            functions: "import lanes;".into(),
            params: params.into(),
            result,
            body,
            // Let the installed DSLX frontend decide whether independently
            // written radix literals retain their signed bit-pattern meaning.
            valid: match (signed, kind) {
                (true, 5) => None,
                (true, 2 | 4 | 6 | 7) => Some(false),
                _ => Some(true),
            },
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
    )
}

#[cfg(test)]
mod tests {
    use super::check_imported_array_literals;

    #[test]
    fn repeated_bindings_and_independent_literals_at_imports() -> Result<(), String> {
        for form in 0..4 {
            for kind in 0..8 {
                // The value 0xff exercises the s8[2] regression. The unsigned
                // and wide controls exercise values through the same oracle.
                for (width, length, flags) in [(7, 0, 0), (0, 3, 1), (128, 1, 0)] {
                    check_imported_array_literals(&[form, kind, width, length, flags, 255])?;
                }
            }
        }
        Ok(())
    }
}
