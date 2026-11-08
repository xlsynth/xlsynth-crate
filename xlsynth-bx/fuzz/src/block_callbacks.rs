// SPDX-License-Identifier: Apache-2.0

//! Compare combinational BX block callback specializations with direct DSLX
//! calls.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;
use std::fs;
use tempfile::tempdir;

const TRANSFORM: &str = "block transform<I,O,F:fn(I)->O>(clk, x:I)->O { F(x) }";
const COMPOSE: &str = "block compose<I,M,O,F:fn(I)->M,G:fn(M)->O>(clk, x:I)->O { G(F(x)) }";

/// Exercise inferred callback input/output bindings with independently chosen
/// widths and aggregate shapes. The reference uses ordinary concrete calls.
pub fn check_block_callbacks(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 96) + 1;
    let mid_width = usize::from(bytes.next().unwrap() % 96) + 1;
    let out_width = usize::from(bytes.next().unwrap() % 96) + 1;
    let mutation = bytes.next().unwrap() % 8;
    let input = format!("uN[{width}]");
    let mid = format!("uN[{mid_width}]");
    if flags & 128 != 0 {
        return compare_imported(data, &input, &mid, flags & 1 != 0);
    }
    let input = if flags & 1 == 0 {
        input
    } else {
        format!("sN[{width}]")
    };
    let (arg, select) = match flags >> 1 & 3 {
        0 => (format!("({input},bool)"), "if x.1 {x.0} else {!x.0}"),
        1 => (format!("{input}[2]"), "x[u32:0] ^ x[u32:1]"),
        _ => (input, "x"),
    };
    let (middle, make, read) = match flags >> 3 & 3 {
        0 => (
            format!("({mid},bool)"),
            "(v, v == 0)",
            "if x.1 {!x.0} else {x.0}",
        ),
        1 => (format!("{mid}[2]"), "[v,!v]", "x[u32:1]"),
        _ => (mid.clone(), "v", "x"),
    };
    let out = format!("uN[{out_width}]");
    let (result_type, finish) = if flags & 32 != 0 {
        (format!("({out},bool)"), "(v, v != 0)")
    } else {
        (out.clone(), "v")
    };
    let helpers = format!(
        "fn make(v:{mid})->{middle} {{ {make} }}\n\
         fn first(x:{arg})->{middle} {{ make(({select}) as {mid}) }}\n\
         fn other(x:{arg})->{middle} {{ make(!(({select}) as {mid})) }}\n\
         fn pack(v:{out})->{result_type} {{ {finish} }}\n\
         fn finish(x:{middle})->{result_type} {{ pack(({read}) as {out}) }}\n\
         fn finish_alt(x:{middle})->{result_type} {{ pack(!(({read}) as {out})) }}\n\
         fn bad_input(x:({arg},{arg}))-> {middle} {{ first(x.0) }}\n\
         fn bad_middle(x:{arg})-> ({middle},{middle}) {{ (first(x),other(x)) }}"
    );
    let (first, other) = if flags & 64 == 0 {
        ("first", "other")
    } else {
        ("other", "first")
    };
    let target = match mutation {
        6 => "bad_input",
        7 => "bad_middle",
        _ => first,
    };
    let params = format!("a:{arg}, b:{arg}");
    let result = format!("({result_type},{result_type},{result_type},{result_type})");
    let bx_body = format!(
        "let mid_value = transform<{target}>(a); let left = transform<finish>(mid_value);\n\
         let right = compose<{other},finish>(b); let again = compose<{target},finish>(b);\n\
         let another = compose<{other},finish_alt>(b); (left,right,again,another)"
    );
    let bx = format!(
        "{helpers}\n{TRANSFORM}\n{COMPOSE}\n{}",
        bx_source(&params, &result, &bx_body)
    );
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: helpers,
            params,
            result,
            body: format!(
                "(finish({target}(a)), finish({other}(b)), finish({target}(b)), finish_alt({other}(b)))"
            ),
            valid: Some(mutation < 6),
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[],
    )
}

/// Apply one generic imported callback at two types, including two spellings
/// of the same source member, so each concrete signature is checked.
fn compare_imported(data: &[u8], first: &str, second: &str, increment: bool) -> Result<(), String> {
    let dir = tempdir().map_err(|e| e.to_string())?;
    fs::write(
        dir.path().join("arithmetic.x"),
        r#"
pub fn invert<N:u32>(x:uN[N])->uN[N] { !x }
pub fn increment<N:u32>(x:uN[N])->uN[N] { x + uN[N]:1 }
"#,
    )
    .map_err(|e| e.to_string())?;
    let cb = if increment { "increment" } else { "invert" };
    let params = format!("a:{first}, b:{second}");
    let result = format!("({first},{second},{second})");
    let bx_body = format!(
        "let x=transform<arithmetic::{cb}>(a); let y=transform<{cb}>(b); let z=transform<arithmetic::{cb}>(y); (x,y,z)"
    );
    let bx = format!(
        "use arithmetic; use arithmetic::{cb};\n{TRANSFORM}\n{}",
        bx_source(&params, &result, &bx_body)
    );
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: "import arithmetic;".into(),
            params,
            result,
            body: format!(
                "(arithmetic::{cb}(a), arithmetic::{cb}(b), arithmetic::{cb}(arithmetic::{cb}(b)))"
            ),
            valid: Some(true),
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &[dir.path()],
    )
}

#[cfg(test)]
mod tests {
    use super::check_block_callbacks;

    #[test]
    fn compare_generated_block_callbacks() -> Result<(), String> {
        for flags in [0, 1, 2, 8, 35, 64, 127, 128, 129] {
            for mutation in [0, 6, 7] {
                check_block_callbacks(&[flags, 95, 12, 7, mutation])?;
            }
        }
        Ok(())
    }
}
