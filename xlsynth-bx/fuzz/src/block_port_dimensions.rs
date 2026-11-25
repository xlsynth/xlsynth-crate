// SPDX-License-Identifier: Apache-2.0

//! Compare specialized BX block port dimensions with pinned DSLX signatures.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
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
    let typed = flags & 16 != 0;
    let pick = flags & 32 != 0;
    let base = if typed {
        ("bit_count<T>()", m + 1)
    } else {
        ("N", n)
    };
    let width = dimension(width_code, base, m, pick, 129);
    let length = dimension(length_code, base, m, pick, 8);
    let signed = if flags & 4 == 0 { 'u' } else { 's' };
    let array = flags & 1 != 0;
    let port_type = |width: &str, length: &str| {
        let word = format!("{signed}N[{width}]");
        match flags & 3 {
            0 => word,
            1 => format!("{word}[{length}]"),
            2 => format!("(bool, {word})"),
            _ => format!("({word}[2], bool)[{length}]"),
        }
    };
    let parametric = port_type(&width.source, &length.source);
    let mutation = mutation % 6;
    let concrete = |mismatch: bool| {
        let change_length = array && flags & 8 != 0;
        let w = width.size + u32::from(mismatch && !change_length);
        let l = length.size + u32::from(mismatch && change_length);
        port_type(&w.to_string(), &l.to_string())
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
}
