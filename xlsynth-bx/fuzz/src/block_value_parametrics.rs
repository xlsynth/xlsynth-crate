// SPDX-License-Identifier: Apache-2.0

//! Compare BX block value bindings and unrolls with DSLX function calls.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;

const FOLD: &str = r#"
    let folded = unroll_for! (i, acc): (u32, u32) in u32:0..N {
        (acc ^ i) + seed
    }(seed);
    (x, folded)
"#;

/// Compare calls at distinct depths with otherwise identical signatures.
/// Vary the spelling and scope of actuals and the inferred payload type.
pub fn check_block_value_parametrics(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 129) + 1;
    let first = bytes.next().unwrap() % 9;
    let mut second = bytes.next().unwrap() % 8;
    if second >= first {
        second += 1;
    }
    let mutation = bytes.next().unwrap() % 12;
    let signedness = if flags & 1 == 0 { 'u' } else { 's' };
    let scalar = format!("{signedness}N[{width}]");
    let payload = match flags >> 1 & 3 {
        0 => scalar,
        1 => format!("({scalar}, bool)"),
        2 => format!("{scalar}[2]"),
        _ => format!("(bool, {scalar}[2])"),
    };
    let (block_formals, block_type) = if flags & 8 == 0 {
        ("T, N: u32", "T")
    } else {
        ("N: u32", payload.as_str())
    };
    let (constant, first_actual) = if flags & 16 == 0 {
        (String::new(), format!("{first}"))
    } else {
        (format!("const DEPTH = u32:{first};"), "DEPTH".into())
    };
    let (binding, second_actual) = if flags & 32 == 0 {
        ("".to_owned(), format!("u32:{second}"))
    } else {
        let base = second / 2;
        let increment = second - base;
        (
            format!("let base = u32:{base};"),
            format!("{{base + u32:{increment}}}"),
        )
    };
    let third_actual = match mutation {
        8 => String::new(),
        9 => format!("<u8:{first}>"),
        10 => "<seed>".into(),
        11 => format!("<{first}, {second}>"),
        _ => format!("<u32:{first}>"),
    };
    let mut bx_decls = format!(
        "{constant}\n#[allow(combinational_block)]\n\
         block fold<{block_formals}>(clk, x: {block_type}, seed: u32) -> ({block_type}, u32) {{
             {FOLD}
         }}"
    );
    let mut dslx_decls = format!(
        "{constant}\nfn fold<N: u32>(x: {payload}, seed: u32) -> ({payload}, u32) {{
             {FOLD}
         }}"
    );
    let callee = if flags & 64 == 0 {
        "fold"
    } else {
        bx_decls.push_str(&format!(
            "\nblock forward<{block_formals}>(clk, x: {block_type}, seed: u32) -> ({block_type}, u32) {{
                 let forwarded = fold<N>(x, seed);
                 forwarded
             }}"
        ));
        dslx_decls.push_str(&format!(
            "\nfn forward<N: u32>(x: {payload}, seed: u32) -> ({payload}, u32) {{
                 fold<N>(x, seed)
             }}"
        ));
        "forward"
    };
    let params = format!("x: {payload}, seed: u32");
    let pair = format!("({payload}, u32)");
    let result = format!("({pair}, {pair}, {pair})");
    let body = format!(
        "{binding}
         let first = {callee}<{first_actual}>(x, seed);
         let second = {callee}<{second_actual}>(x, seed);
         let again = {callee}{third_actual}(x, seed);
         (first, second, again)"
    );
    let bx = format!("{bx_decls}\n{}", bx_source(&params, &result, &body));
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: dslx_decls,
            params,
            result,
            body,
            valid: Some(mutation < 8),
            dslx_ir: true,
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
    use super::check_block_value_parametrics;

    #[test]
    fn compare_generated_block_value_parametrics() -> Result<(), String> {
        for flags in [0, 7, 56, 97, 126] {
            for mutation in [0, 8, 9, 10, 11] {
                check_block_value_parametrics(&[flags, 128, 0, 6, mutation])?;
                check_block_value_parametrics(&[flags, 7, 5, 0, mutation])?;
            }
        }
        Ok(())
    }
}
