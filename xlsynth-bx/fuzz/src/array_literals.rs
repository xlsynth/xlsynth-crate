// SPDX-License-Identifier: Apache-2.0

//! Compare typed array construction and trailing ellipses in BX and DSLX.

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use crate::input_seed;

/// Vary the array's element type, its explicit prefix, and where the size comes
/// from.
pub fn check_array_literals(data: &[u8]) -> Result<(), String> {
    let [shape, length, prefix, mode, width, flags] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let length = usize::from(length % 7) + 1;
    let width = usize::from(width % 96) + 2;
    let signed = flags & 1 != 0;
    let word = format!("{}N[{width}]", if signed { 's' } else { 'u' });
    let mut functions = String::new();
    let (element, values) = match shape % 6 {
        0 => (word.clone(), ["1", "x", "y", "x ^ y"]),
        1 => ("bool".into(), ["false", "x == y", "true", "x != y"]),
        2 => (
            format!("({word}, bool)"),
            ["(x, false)", "(y, true)", "(x ^ y, x == y)", "(y, x != y)"],
        ),
        3 => (
            format!("{word}[3]"),
            ["[x, ...]", "[y, x, ...]", "[x, y, x]", "[y, ...]"],
        ),
        4 => {
            functions = format!("struct Packet {{ value: {word} }}\n");
            (
                "Packet".into(),
                [
                    "Packet { value: x }",
                    "Packet { value: y }",
                    "Packet { value: x ^ y }",
                    "Packet { value: 1 }",
                ],
            )
        }
        _ => {
            functions = "enum Code : u2 { Low = 0, High = 1 }\n".into();
            (
                "Code".into(),
                [
                    "Code::Low",
                    "Code::High",
                    "if x == y { Code::Low } else { Code::High }",
                    "Code::Low",
                ],
            )
        }
    };
    let mut result = format!("{element}[{length}]");
    let size = if flags & 16 != 0 {
        functions.push_str(&format!("const SIZE = u32:{length};\n"));
        functions.push_str(if length > 1 {
            "const MASK = bool[SIZE]:[false, true, ...];\n"
        } else {
            "const MASK = bool[SIZE]:[false, ...];\n"
        });
        "SIZE".to_owned()
    } else {
        length.to_string()
    };
    let mode = mode % 22;
    let count = match mode {
        1 => length,     // Exact-length literal without an ellipsis.
        5 => length + 1, // Too many elements with an ellipsis.
        6 => length - 1, // Too few elements without an ellipsis.
        7 => 0,          // No element for the ellipsis to repeat.
        _ => usize::from(prefix) % length + 1,
    };
    let mut items: Vec<_> = (0..count)
        .map(|i| values[(usize::from(flags >> 1) + i) % values.len()])
        .collect();
    if mode == 8 {
        items[0] = "()";
    }
    if !matches!(mode, 1 | 6) {
        items.push("...");
    }
    if mode == 9 {
        items.push(values[0]);
    }
    let comma = if !items.is_empty() && flags & 128 != 0 {
        ","
    } else {
        ""
    };
    let literal = format!("[{}{comma}]", items.join(", "));
    let typed = format!("{element}[{size}]:{literal}");
    let params = format!("x: {word}, y: {word}");
    let body = match mode {
        2 => literal,
        3 => format!("let elements: {result} = {literal}; elements"),
        4 => {
            functions.push_str(&format!(
                "fn elements<N: u32>({params}) -> {element}[N] {{ {element}[N]:{literal} }}"
            ));
            format!("elements<u32:{length}>(x, y)")
        }
        10 => format!("let elements = {literal}; elements"),
        12 => format!("let empty = {element}[0]:[]; empty ++ {typed}"),
        13 if flags & 96 == 32 => {
            functions.push_str(&format!(
                "fn accept<N: u32>(a: {element}[N]) -> {element}[N] {{ a }}"
            ));
            format!("accept<u32:{length}>({literal})")
        }
        13 if flags & 96 == 64 => {
            functions.push_str(&format!(
                "fn accept(a: ({result}, bool)) -> {result} {{ a.0 }}"
            ));
            format!("accept(({literal}, true))")
        }
        11 | 13 | 15 => {
            functions.push_str(&format!("fn accept(a: {result}) -> {result} {{ a }}"));
            match mode {
                11 => format!("accept({typed})"),
                13 if flags & 96 == 96 => {
                    format!("accept(if x == y {{ {literal} }} else {{ {literal} }})")
                }
                15 => format!("accept(if x == y {{ {literal} }} else {{ {typed} }})"),
                _ => format!("accept({literal})"),
            }
        }
        14 => {
            result = "bool".into();
            format!("let elements = {typed}; {literal} == elements")
        }
        16 => {
            // One row fixes the element width and the other fixes the length.
            let bare = (0..length)
                .map(|i| if i % 2 == 0 { "0" } else { "1" })
                .collect::<Vec<_>>()
                .join(", ");
            let fixed_length = format!("[{bare}]");
            let mut rows = ["[x, ...]", fixed_length.as_str()];
            if flags & 2 != 0 {
                rows.reverse();
            }
            if flags & 4 == 0 {
                result = "bool".into();
                rows.join(" == ")
            } else {
                result = format!("{word}[{length}][2]");
                format!("let rows = [{}]; rows", rows.join(", "))
            }
        }
        17 | 18 => {
            functions.push_str(&format!("fn accept(a: {result}) -> {result} {{ a }}"));
            let keyword = if flags & 2 == 0 { "for" } else { "unroll_for!" };
            let annotation = if mode == 18 {
                format!(": (u32, {result})")
            } else {
                String::new()
            };
            format!("accept({keyword} (_, a){annotation} in u32:0..u32:1 {{ a }}({literal}))")
        }
        21 => {
            functions.push_str(&format!(
                "fn accept<N: u32>(a: {element}[N]) -> {element}[N] {{ a }}"
            ));
            let keyword = if flags & 2 == 0 { "for" } else { "unroll_for!" };
            let full_row = vec![values[2]; length].join(", ");
            format!(
                "accept(if x == y {{ {keyword} (_, a) in u32:0..u32:1 {{ a }}({literal}) }} else {{ [{full_row}] }})"
            )
        }
        19 => {
            // A projection of the destructured rows takes its width at use.
            let fill = "[2, ...]".to_owned();
            let full = (0..length)
                .map(|i| if i == 0 { "0" } else { "1" })
                .collect::<Vec<_>>()
                .join(", ");
            let full = format!("[{full}]");
            let mut rows = [fill, full];
            if flags & 2 != 0 {
                rows.reverse();
            }
            result = format!("uN[{width}]");
            // The native evaluators can retain the projection's narrower
            // runtime bits. The binding checks its type and the cast gives
            // the comparison an explicitly sized result.
            format!(
                "let (rows,) = ([{}],); let value: {result} = rows[x == y][0]; value as {result}",
                rows.join(", ")
            )
        }
        20 => {
            // Distinct unsized operands lose a hexadecimal literal's ability
            // to spell a signed bit pattern at its original width.
            result = format!("sN[{width}][3]");
            functions.push_str(&format!("fn accept(a: {result}) -> {result} {{ a }}"));
            let pattern = (1u128 << width) - 1;
            format!("accept(if x == y {{ [{pattern:#x}, 0, ...] }} else {{ {result}:[0, ...] }})")
        }
        _ => typed,
    };
    compare_program(
        FunctionProgram {
            functions,
            params,
            result,
            body,
            valid: Some(!matches!(mode, 5..=10 | 13 | 17 | 20)),
            // The pinned converter rejects empty literals that the interpreter
            // accepts.
            dslx_ir: mode != 12,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::check_array_literals;
    use crate::eval::expr_eval::bx_source;
    use crate::function_oracle::check_sources;

    #[test]
    fn projected_filled_arrays_share_their_written_members_widths() -> Result<(), String> {
        // Native conversion retains the projection's minimum width in this
        // tuple comparison, so compare the two source typecheckers directly.
        for row in ["[0xff, ...]", "[0xff, 0xff]"] {
            let function = format!(
                "fn compare(c: bool) -> bool {{ let (p,) = ((u16:255,),); ([[0xff, 0xff], {row}][c][0],) == p }}"
            );
            let bx = format!("{function}\n{}", bx_source("c: bool", "bool", "compare(c)"));
            check_sources(&bx, &function, Some(true), &[])?;
        }
        Ok(())
    }

    #[test]
    fn compare_array_literal_shapes_and_lengths() -> Result<(), String> {
        for shape in 0..6 {
            for mode in 0..22 {
                check_array_literals(&[shape, 3, 1, mode, 6, 145])?;
            }
            // The explicitly supplied values can already fill the array.
            check_array_literals(&[shape, 1, 1, 0, 78, 2])?;
        }
        for flags in [0, 32, 64, 96] {
            check_array_literals(&[0, 3, 1, 13, 6, flags])?;
        }
        for flags in [0, 2, 4, 6] {
            check_array_literals(&[0, 1, 0, 16, 6, flags])?;
        }
        for mode in [17, 18, 21] {
            check_array_literals(&[0, 3, 1, mode, 6, 2])?;
        }
        for flags in [0, 1, 2, 3] {
            check_array_literals(&[0, 2, 0, 19, 6, flags])?;
        }
        Ok(())
    }
}
