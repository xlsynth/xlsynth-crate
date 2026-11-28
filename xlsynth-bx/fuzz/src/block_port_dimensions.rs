// SPDX-License-Identifier: Apache-2.0

//! Compare specialized BX block port dimensions with pinned DSLX signatures.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{
    DslxExecution, FunctionProgram, compare_bx_program, compare_imported_program,
};
use crate::input_seed;

/// A dimension written in terms of block parameters and its concrete size at
/// the caller. Each generated operation independently updates both forms.
struct Dimension {
    source: String,
    size: u32,
}

/// Compose up to three operations, retaining small aggregate sizes for
/// concrete evaluation and admitting zero for both lengths and widths.
fn dimension(code: u8, base: (&str, u32), m: u32, pick: bool, limit: u32) -> Dimension {
    let mut source = base.0.to_owned();
    let mut size = base.1;
    for op in [code & 7, (code >> 3) & 7, code >> 6] {
        (source, size) = match op {
            // This position leaves the expression at its previous depth.
            0 => (source, size),
            1 => (
                format!("ceillog2({source})"),
                u32::BITS - size.saturating_sub(1).leading_zeros(),
            ),
            2 => (format!("({source} + M)"), size + m),
            3 => (format!("({source} * u32:2)"), size * 2),
            4 => (format!("({source} / M)"), size / m),
            5 => (format!("({source} % M)"), size % m),
            6 => (
                format!("(if PICK {{ {source} }} else {{ M - u32:1 }})"),
                if pick { size } else { m - 1 },
            ),
            _ => (format!("bump({source})"), size + 1),
        };
    }
    if size > limit {
        source = format!("({source} % u32:{})", limit + 1);
        size %= limit + 1;
    }
    Dimension { source, size }
}

/// Instantiate the same parameter-dependent port type on a BX block and a
/// DSLX function, checking the argument and expected result independently.
pub fn check_block_port_dimensions(data: &[u8]) -> Result<(), String> {
    let [n_byte, m_byte, width_code, length_code, flags, mutation] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let n = 1 + u32::from(n_byte);
    let m = 1 + u32::from(m_byte % 9);
    if flags & 128 != 0 {
        return compare_word_ports(flags, n, mutation % 6, input_seed(data));
    }
    let typed = flags & 16 != 0;
    let pick = flags & 32 != 0;
    let base = if typed {
        ("bit_count<T>()", m + 1)
    } else {
        ("N", n)
    };
    let width = dimension(width_code, base, m, pick, 129);
    let length = dimension(length_code, base, m, pick, 8);
    let signed = if flags & 64 != 0 {
        pick
    } else {
        flags & 4 != 0
    };
    let scalar = |width: &str| format!("{}N[{width}]", if signed { 's' } else { 'u' });
    let array = flags & 1 != 0;
    let port_type = |word: String, length: &str| match flags & 3 {
        0 => word,
        1 => format!("{word}[{length}]"),
        2 => format!("(bool, {word})"),
        _ => format!("({word}[2], bool)[{length}]"),
    };
    let word = if flags & 64 != 0 {
        format!("xN[PICK][{}]", width.source)
    } else {
        scalar(&width.source)
    };
    let parametric = port_type(word, &length.source);
    let mutation = mutation % 6;
    let concrete = |mismatch: bool| {
        let change_length = array && flags & 8 != 0;
        let w = width.size + u32::from(mismatch && !change_length);
        let l = length.size + u32::from(mismatch && change_length);
        port_type(scalar(&w.to_string()), &l.to_string())
    };
    let input = concrete(mutation == 4);
    let output = concrete(mutation == 5);
    let witness = if typed { ", witness: T" } else { "" };
    let signature = |type_param: &str| {
        format!(
            "port<{type_param}N: u32, M: u32, PICK: bool>(x: {parametric}{witness}) -> {parametric} {{ x }}"
        )
    };
    let witness_arg = if typed {
        format!(", (false, uN[{m}]:0)")
    } else {
        String::new()
    };
    let call = format!("port<{n}, {m}, {pick}>(x{witness_arg})");
    let helper = "fn bump(x: u32) -> u32 { x + u32:1 }";
    let empty = array && length.size == 0;
    let (params, result, body) = if empty {
        // Native IR values cannot carry an empty array. Pass an explicitly
        // typed zero through both signatures and observe its outer length.
        (
            String::new(),
            "u32".to_owned(),
            format!("let x = zero!<{input}>(); let y: {output} = {call}; array_size(y)"),
        )
    } else {
        (format!("x: {input}"), output, format!("let y = {call}; y"))
    };
    let bx = format!(
        "{helper}\nblock {}\n{}",
        signature(if typed { "T, " } else { "" }),
        bx_source(&params, &result, &body)
    );
    compare_bx_program(
        &bx,
        FunctionProgram {
            // Pinned DSLX functions cannot take type formals. Supply the
            // concrete witness type BX infers from the same input.
            functions: format!("type T = (bool, uN[{m}]);\n{helper}\nfn {}", signature("")),
            params,
            result,
            body,
            valid: Some(mutation < 4),
            dslx_ir: !empty,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

/// Pass a value through a nominal port and observe its payload at a wider
/// type. A phantom signedness parameter also tests equal-layout mismatches.
fn compare_word_ports(flags: u8, n: u32, mutation: u8, seed: u64) -> Result<(), String> {
    let signed = flags & 4 != 0;
    let phantom = flags & 8 != 0;
    let imported = flags & 16 != 0;
    let name = if imported { "words::Word" } else { "Word" };
    let field = if phantom { "uN[N]" } else { "xN[S][N]" };
    let library = format!("pub struct Word<S: bool, N: u32> {{ payload: {field} }}");
    let port_type = |word: &str| match flags & 3 {
        0 => word.to_owned(),
        1 => format!("{word}[2]"),
        2 => format!("(bool, {word})"),
        _ => format!("({word}[2], bool)"),
    };
    let (argument, payload) = match flags & 3 {
        0 => ("word", "result.payload"),
        1 => ("[word, word]", "result[u32:1].payload"),
        2 => ("(true, word)", "result.1.payload"),
        _ => ("([word, word], true)", "result.0[u32:0].payload"),
    };
    let (sign_arg, port_signed) = if flags & 32 == 0 {
        ("S", signed)
    } else {
        ("{!S}", !signed)
    };
    let (width_arg, width) = if flags & 64 == 0 {
        ("N", n)
    } else {
        ("{N + u32:1}", n + 1)
    };
    let port = port_type(&format!("{name}<{sign_arg}, {width_arg}>"));
    let signature = format!("port<S: bool, N: u32>(x: {port}) -> {port} {{ x }}");
    let input_signed = port_signed ^ (mutation == 4);
    let word = format!("{name}<{input_signed}, {width}>");
    let output = port_type(&format!(
        "{name}<{}, {width}>",
        port_signed ^ (mutation == 5)
    ));
    let sign = if input_signed && !phantom { 's' } else { 'u' };
    let params = format!("x: uN[{width}]");
    let result = format!("uN[{}]", width + 1);
    let body = format!(
        "let word = {word} {{ payload: x as {sign}N[{width}] }};
         let result: {output} = port<{signed}, {n}>({argument});
         {payload} as {result}"
    );
    let bx_decls = if imported { "use words;" } else { &library };
    let bx = format!(
        "{bx_decls}\nblock {signature}\n{}",
        bx_source(&params, &result, &body)
    );
    let dslx_decls = if imported { "import words;" } else { &library };
    let program = FunctionProgram {
        functions: format!("{dslx_decls}\nfn {signature}"),
        params,
        result,
        body,
        valid: Some(mutation < 4),
        dslx_ir: true,
        dynamic_index: None,
    };
    if imported {
        compare_imported_program("words", &library, &bx, program, seed)
    } else {
        compare_bx_program(
            &bx,
            program,
            seed,
            DslxExecution::ConcreteConstants,
            Vec::new(),
            &[],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::check_block_port_dimensions;

    #[test]
    fn ceillog2_ports_at_power_boundaries() -> Result<(), String> {
        for n in [1, 2, 3, 8, 9, 128, 129] {
            for flags in [0, 3, 4, 9] {
                for mutation in [0, 4, 5] {
                    check_block_port_dimensions(&[n - 1, 2, 1, 1, flags, mutation])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn composed_port_dimensions_match_dslx() -> Result<(), String> {
        for op in 0..8 {
            for mutation in [0, 4, 5] {
                // Mix a different operation with ceillog2, in opposite
                // orders for the element width and outer array length.
                for flags in [15, 31, 63] {
                    check_block_port_dimensions(&[16, 3, op | 8, 1 | (op << 3), flags, mutation])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn dependent_signedness_and_struct_ports_match_dslx() -> Result<(), String> {
        for flags in [64, 97, 128, 133, 138, 175, 144, 213, 154, 255] {
            for mutation in [0, 4, 5] {
                check_block_port_dimensions(&[64, 2, 0, 0, flags, mutation])?;
            }
        }
        Ok(())
    }
}
