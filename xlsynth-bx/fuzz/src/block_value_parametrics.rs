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

/// Compare scalar and record value bindings on BX blocks with DSLX functions.
pub fn check_block_value_parametrics(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 129) + 1;
    if flags & 128 != 0 {
        let first = 1 + bytes.next().unwrap() % 8;
        let second = 1 + bytes.next().unwrap() % 8;
        let mutation = bytes.next().unwrap() % 8;
        return compare_record_bindings(flags, width, first, second, mutation, input_seed(data));
    }
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

/// Read both record fields from each specialization and pass an array through
/// its field-dependent input and output ports.
fn compare_record_bindings(
    flags: u8,
    width: usize,
    first_depth: u8,
    second_depth: u8,
    mutation: u8,
    seed: u64,
) -> Result<(), String> {
    let first_pop = flags & 1 != 0;
    let second_pop = flags & 2 != 0;
    let records = format!(
        "struct BufferOpts {{ depth: u32, register_pop: bool }}
         struct OtherOpts {{ depth: u32, register_pop: bool }}
         const BUFFER_OPTS = BufferOpts {{ depth: {first_depth}, register_pop: {first_pop} }};
         const SECOND = BufferOpts {{ register_pop: {second_pop}, depth: {second_depth} }};"
    );
    let word = format!("uN[{width}]");
    let first = format!("{word}[{first_depth}]");
    let second = format!("{word}[{second_depth}]");
    let result_type = |port: &str| format!("({port}, bool, u32)");
    let default = if flags & 4 != 0 {
        " = {BUFFER_OPTS}"
    } else {
        ""
    };
    let emit = |decl: &str, derived: &str, length: &str, clock: &str| {
        let port = format!("{word}[{length}]");
        let result = result_type(&port);
        let formals = format!("CFG: BufferOpts{default}{derived}");
        let mut definitions = format!(
            "{records}\n{decl} buffer<{formals}>({clock}x: {port}) -> {result} {{
                 (x, CFG.register_pop, CFG.depth)
             }}"
        );
        if flags & 8 != 0 {
            definitions.push_str(&format!(
                "\n{decl} forward<{formals}>({clock}x: {port}) -> {result} {{
                     let y = buffer<CFG>(x);
                     y
                 }}"
            ));
        }
        definitions
    };
    let bx_decls = emit(
        "#[allow(combinational_block)]\nblock",
        "",
        "CFG.depth",
        "clk, ",
    );
    // Pinned DSLX represents this port dimension through a derived scalar
    // binding. Both callers still provide only the record-valued argument.
    let dslx_decls = emit("fn", ", D: u32 = {CFG.depth}", "D", "");
    let callee = if flags & 8 != 0 { "forward" } else { "buffer" };
    let first_actual = if default.is_empty() {
        "<BUFFER_OPTS>"
    } else {
        ""
    };
    let second_actual = match flags >> 4 & 3 {
        0 => "<SECOND>".to_owned(),
        1 => "<local>".to_owned(),
        _ => format!("<{{BufferOpts {{ depth: {second_depth}, register_pop: {second_pop} }}}}>"),
    };
    let again_actual = match mutation {
        4 => "<runtime>".to_owned(),
        5 => format!("<{{OtherOpts {{ depth: {first_depth}, register_pop: {first_pop} }}}}>"),
        _ => format!("<{{BufferOpts {{ register_pop: {first_pop}, depth: {first_depth} }}}}>"),
    };
    let first_input = if mutation == 6 {
        format!("{word}[{}]", first_depth + 1)
    } else {
        first.clone()
    };
    let second_output = if mutation == 7 {
        format!("{word}[{}]", second_depth + 1)
    } else {
        second.clone()
    };
    let runtime = if mutation == 4 {
        ", runtime: BufferOpts"
    } else {
        ""
    };
    let params = format!("a: {first_input}, b: {second}{runtime}");
    let result = format!(
        "({}, {}, {})",
        result_type(&first),
        result_type(&second_output),
        result_type(&first)
    );
    let body = format!(
        "let local = SECOND;
         let x = {callee}{first_actual}(a);
         let y = {callee}{second_actual}(b);
         let z = {callee}{again_actual}(a);
         (x, y, z)"
    );
    let bx = format!("{bx_decls}\n{}", bx_source(&params, &result, &body));
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: dslx_decls,
            params,
            result,
            body,
            valid: Some(mutation < 4),
            dslx_ir: true,
            dynamic_index: None,
        },
        seed,
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

    #[test]
    fn compare_record_block_value_parametrics() -> Result<(), String> {
        for flags in [128, 130, 149, 154, 172] {
            for mutation in [0, 4, 5, 6, 7] {
                // Depth three, including the plain BUFFER_OPTS constant with
                // register_pop false. Use equal depths and then different ones.
                check_block_value_parametrics(&[flags, 7, 2, 2, mutation])?;
                check_block_value_parametrics(&[flags, 128, 2, 5, mutation])?;
            }
        }
        Ok(())
    }
}
