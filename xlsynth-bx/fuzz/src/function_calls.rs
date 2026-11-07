// SPDX-License-Identifier: Apache-2.0

//! Compare same-file function definitions and their callers in BX and DSLX.

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use crate::generator;

/// Exercise a generated function through nested local calls and a BX block.
pub fn check_function_calls(data: &[u8]) -> Result<(), String> {
    let (&mode, body_bytes) = data.split_first().unwrap_or((&0, &[]));
    let sample = generator::generate(body_bytes, generator::EvaluationDomain::Bytecode);
    let params = &sample.params;
    let result = &sample.result_type;
    let compute = ["compute", "stage", "block", "compute_2"][usize::from(mode >> 6)];
    let args = "x, y, offset, flag";
    let variant = match mode % 8 {
        0 => "!x, y, offset, !flag",
        1 => "x, -y, !offset, flag",
        2 => "x + x * !x, y, offset, flag",
        3 => "x, y ^ y + y, offset, !flag",
        4 => "x, y, 0, !flag",
        5 => "x, y, offset",
        6 => "x, y, offset, flag, flag",
        _ => "x, y, offset, (flag, flag)",
    };
    let functions = format!(
        r#"fn {compute}({params}) -> {result} {{
    {}
    {}
}}
fn choose(flag: bool, left: {result}, right: {result}) -> {result} {{
    if flag {{ left }} else {{ right }}
}}
pub fn route({params}) -> {result} {{
    let first = {compute}({args});
    choose(flag, first, {compute}({variant}))
}}"#,
        sample.bindings, sample.expression,
    );
    let block_variant = if mode % 8 < 5 { variant } else { args };
    let call = match (mode >> 3) % 4 {
        0 => format!("route({args})"),
        1 => format!("{compute}({args})"),
        2 => format!("choose(!flag, route({args}), {compute}({block_variant}))"),
        _ => format!("let left = route({args});\nchoose(flag, left, route({block_variant}))"),
    };
    compare_program(
        FunctionProgram {
            functions,
            params: sample.params,
            result: sample.result_type,
            body: call,
            valid: Some(mode % 8 < 5),
            dslx_ir: true,
            dynamic_index: None,
        },
        crate::input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}
