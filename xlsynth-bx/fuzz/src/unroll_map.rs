// SPDX-License-Identifier: Apache-2.0

//! Compare a BX structural map with the pinned DSLX map on concrete inputs.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;

/// Map runtime arrays directly or index them with an unrolled range.
pub fn check_unroll_map(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let length = bytes.next().unwrap() % 6;
    let input_width = 1 + usize::from(bytes.next().unwrap() % 128);
    let output_width = 1 + usize::from(bytes.next().unwrap() % 128);
    let signed = if flags & 1 == 0 { 'u' } else { 's' };
    let input = format!("{signed}N[{input_width}]");
    let output = format!("uN[{output_width}]");
    let (binder, element, value) = if flags & 2 == 0 {
        ("a", input.clone(), "a")
    } else {
        (
            "((a, flip), b)",
            format!("(({input}, bool), {input})"),
            "if flip { b - a } else { a - b }",
        )
    };
    let (result_element, result_expr) = match ((flags >> 2) & 3) % 3 {
        0 => (output.clone(), "mixed"),
        1 => (format!("({output}, bool)"), "(mixed, mixed == offset)"),
        _ => (format!("{output}[2]"), "[mixed, !mixed]"),
    };
    let body = format!("let mixed = (({value}) as {output}) ^ offset; {result_expr}");
    let offset = format!("let offset = bias + {output}:1;");
    let signature = |dim: &str| {
        format!("left: {element}[{dim}], right: {element}[{dim}], pick: bool, bias: {output}")
    };
    let params = if length == 0 {
        format!("bias: {output}")
    } else {
        signature(&length.to_string())
    };
    let array_type = format!("{result_element}[{length}]");
    let (result, bx, native) = if length == 0 {
        // The pinned DSLX map requires a nonempty operand and its IR value
        // API cannot represent an empty array. Compare the length of the
        // contextual BX result to DSLX constexpr execution of a typed zero.
        let source = format!(
            "fn length({params}) -> u32 {{
                {offset}
                let empty: {array_type} = unroll_map! {binder} in zero!<{element}[0]>() {{ {body} }};
                array_size(empty)
            }}"
        );
        (
            "u32".to_owned(),
            format!("{source}\n{}", bx_source(&params, "u32", "length(bias)")),
            format!("array_size(zero!<{array_type}>())"),
        )
    } else {
        let select = "if pick { left } else { right }";
        let map = if flags & 16 == 0 {
            format!("unroll_map! {binder} in {select} {{ {body} }}")
        } else {
            let first = bytes.next().unwrap() % 17;
            let range = if flags & 32 == 0 {
                format!("u32:{first}..(u32:{first} + N)")
            } else {
                format!("u32:{first}..=(u32:{first} + N - u32:1)")
            };
            format!(
                "let items = {select};
                 unroll_map! i in {range} {{ let {binder} = items[i - u32:{first}]; {body} }}"
            )
        };
        let declaration = format!(
            "fn transform<N: u32>({}) -> {result_element}[N] {{ {offset} {map} }}",
            signature("N"),
        );
        let bx = format!(
            "{declaration}\n{}",
            bx_source(&params, &array_type, "transform(left, right, pick, bias)")
        );
        let native = format!(
            "{offset} map({select}, |element| -> {result_element} {{ let {binder} = element; {body} }})"
        );
        (array_type, bx, native)
    };
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: String::new(),
            params,
            result,
            body: native,
            valid: Some(true),
            dslx_ir: length != 0,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

#[cfg(test)]
mod tests {
    use super::check_unroll_map;

    #[test]
    fn typed_maps_match_dslx() -> Result<(), String> {
        for flags in 0..12 {
            for (length, input_width, output_width) in [(0, 8, 1), (1, 1, 128), (5, 97, 11)] {
                check_unroll_map(&[flags, length, input_width - 1, output_width - 1, 0xa7])?;
            }
        }
        Ok(())
    }

    #[test]
    fn indexed_ranges_match_dslx_map() -> Result<(), String> {
        for range in [16, 16 | 32] {
            for flags in [0, 3, 5, 10] {
                for (first, length) in [(0, 1), (7, 1), (0, 5), (9, 4)] {
                    check_unroll_map(&[range | flags, length, 64, 11, first, 0xdb])?;
                }
            }
        }
        Ok(())
    }
}
