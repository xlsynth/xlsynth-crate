// SPDX-License-Identifier: Apache-2.0

//! Compare named BX callback specializations with concrete DSLX call graphs.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;
use std::fs;
use tempfile::tempdir;

/// A BX template invocation and its ordinary, concretely bound DSLX body.
struct Case {
    params: String,
    result: String,
    helpers: String,
    template: String,
    bx_body: String,
    dslx_body: String,
    valid: bool,
}

/// Vary callable signatures, arguments, and target identity independently.
pub fn check_function_parametrics(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let family = bytes.next().unwrap() % 3;
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 100) + 1;
    let output_width = usize::from(bytes.next().unwrap() % 100) + 1;
    let length = usize::from(bytes.next().unwrap() % 4) + 1;
    let mutation = bytes.next().unwrap() % 8;
    let signed = if flags & 1 == 0 { "u" } else { "s" };
    let input = format!("{signed}N[{width}]");
    let wide = format!("uN[{output_width}]");

    // Build the concrete reference directly from the selected helpers. It
    // shares the pure helper arithmetic but never consults BX specialization.
    let case = match family {
        0 => {
            let (arg, selected) = match flags >> 1 & 3 {
                0 => (input.clone(), "x"),
                1 => (format!("({input},bool)"), "if x.1 {x.0} else {!x.0}"),
                _ => (format!("{input}[2]"), "x[u32:1] - x[u32:0]"),
            };
            let (ret, value) = match flags >> 3 & 3 {
                0 => (wide.clone(), "v".to_string()),
                1 => (format!("({wide},bool)"), "(v, v == 0)".to_string()),
                _ => (format!("{wide}[2]"), "[v,!v]".to_string()),
            };
            let helpers = format!(
                "fn convert(x:{arg})->{wide} {{ ({selected}) as {wide} }}\n\
                 fn invert(x:{arg})->{wide} {{ !convert(x) }}\n\
                 fn finish(v:{wide})->{ret} {{ {value} }}\n\
                 fn wrong_arity(x:{arg}, y:{arg})->{wide} {{ convert(x) ^ convert(y) }}\n\
                 fn wrong_result(x:{arg})->({wide},{wide}) {{ (convert(x),convert(x)) }}"
            );
            let targets = match mutation {
                6 => ["wrong_arity", "invert", "convert"],
                7 => ["wrong_result", "invert", "convert"],
                _ if flags & 32 == 0 => ["convert", "invert", "convert"],
                _ => ["invert", "convert", "invert"],
            };
            let template = format!(
                "fn apply<F:fn({arg})->{wide}, G:fn({wide})->{ret}>(x:{arg})->{ret} {{ G(F(x)) }}"
            );
            let shadow = if mutation == 5 {
                format!("let {} = a; ", targets[0])
            } else {
                String::new()
            };
            let bx = format!(
                "{shadow}(apply<{},finish>(a),apply<{},finish>(b),apply<{},finish>(b))",
                targets[0], targets[1], targets[2]
            );
            let native = format!(
                "{shadow}(finish({}(a)),finish({}(b)),finish({}(b)))",
                targets[0], targets[1], targets[2]
            );
            Case {
                params: format!("a:{arg}, b:{arg}"),
                result: format!("({ret},{ret},{ret})"),
                helpers,
                template,
                bx_body: bx,
                dslx_body: native,
                valid: mutation < 5,
            }
        }
        1 => {
            let params = format!("xs:{input}[{length}], seed:{input}");
            let result = format!("({input},{input})");
            let helpers = format!(
                "fn sub(a:{input}, x:{input})->{input} {{ a-x }}\nfn rev(a:{input}, x:{input})->{input} {{ x-a }}"
            );
            let (first, second) = if flags & 2 == 0 {
                ("sub", "rev")
            } else {
                ("rev", "sub")
            };
            let template = format!(
                "fn fold<F:fn({input},{input})->{input}, G:fn({input},{input})->{input}>({params})->{input} {{ for (x,a) in xs {{ F(G(a,x),x) }}(seed) }}"
            );
            let bx = format!("(fold<{first},{second}>(xs,seed),fold<{second},{first}>(xs,seed))");
            let native = format!(
                "(for (x,a) in xs {{ {first}({second}(a,x),x) }}(seed),for (x,a) in xs {{ {second}({first}(a,x),x) }}(seed))"
            );
            Case {
                params,
                result,
                helpers,
                template,
                bx_body: bx,
                dslx_body: native,
                valid: true,
            }
        }
        _ => {
            let a = format!("uN[{width}]");
            let b = format!("uN[{output_width}]");
            let product = format!("uN[{}]", width + output_width);
            let params = format!("a:{a}, b:{b}");
            let output = if mutation == 7 {
                format!("uN[{}]", width + output_width + 1)
            } else {
                product.clone()
            };
            let result = format!("({output},{output},{output})");
            let local =
                format!("fn sum(a:{a},b:{b})->{output} {{ (a as {output}) + (b as {output}) }}");
            let template =
                format!("fn apply<F:fn({a},{b})->{output}>({params})->{output} {{ F(a,b) }}");
            let (first, second, native) = if flags & 2 == 0 {
                (
                    "arithmetic::product",
                    "sum",
                    "(arithmetic::product(a,b),sum(a,b),arithmetic::product(a,b))",
                )
            } else {
                (
                    "sum",
                    "product",
                    "(sum(a,b),arithmetic::product(a,b),arithmetic::product(a,b))",
                )
            };
            let bx = format!("(apply<{first}>(a,b),apply<{second}>(a,b),apply<product>(a,b))");
            Case {
                params,
                result,
                helpers: local,
                template,
                bx_body: bx,
                dslx_body: native.into(),
                valid: mutation != 7,
            }
        }
    };
    let Case {
        params,
        result,
        mut helpers,
        template,
        bx_body,
        dslx_body,
        valid,
    } = case;
    let mut bx = format!("{template}\n");
    let imports = if family == 2 {
        let directory = tempdir().map_err(|e| e.to_string())?;
        fs::write(
            directory.path().join("arithmetic.x"),
            r#"
pub fn product<N:u32, M:u32, R:u32={N+M}>(a:uN[N], b:uN[M]) -> uN[R] {
    (a as uN[R]) * (b as uN[R])
}
"#,
        )
        .map_err(|e| e.to_string())?;
        bx.push_str("use arithmetic; use arithmetic::product;\n");
        Some(directory)
    } else {
        None
    };
    bx.push_str(&helpers);
    bx.push('\n');
    if flags & 64 == 0 {
        bx.push_str(&bx_source(&params, &result, &bx_body));
    } else {
        bx.push_str(&format!("fn wrapper({params})->{result} {{ {bx_body} }}\n"));
        let wrapper = if family == 1 {
            "wrapper(xs,seed)"
        } else {
            "wrapper(a,b)"
        };
        bx.push_str(&bx_source(&params, &result, wrapper));
    }
    if imports.is_some() {
        helpers.insert_str(0, "import arithmetic;\n");
    }
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: helpers,
            params,
            result,
            body: dslx_body,
            valid: Some(valid),
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
        &imports.iter().map(|dir| dir.path()).collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::check_function_parametrics;

    #[test]
    fn compare_generated_function_parametrics() -> Result<(), String> {
        for family in 0..3 {
            for flags in [0, 7, 10, 63, 64, 127] {
                for mutation in [0, 5, 6, 7] {
                    check_function_parametrics(&[family, flags, 67, 12, 3, mutation])?;
                }
            }
        }
        Ok(())
    }
}
