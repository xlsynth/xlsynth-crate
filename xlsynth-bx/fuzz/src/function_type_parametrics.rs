// SPDX-License-Identifier: Apache-2.0

//! Compare inferred BX function types and callback results with concrete DSLX.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;

const ZIP_WITH: &str = r#"
fn zip_with<A,B,R,F:fn(A,B)->R,N:u32>(xs:A[N], ys:B[N])->R[N] {
    for (i, acc) in u32:0..N { update(acc,i,F(xs[i],ys[i])) }(zero!<R[N]>())
}
"#;

/// Vary both element types, the independent result type and the array length.
pub fn check_function_type_parametrics(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let left_width = usize::from(bytes.next().unwrap() % 96) + 1;
    let right_width = usize::from(bytes.next().unwrap() % 96) + 1;
    let result_width = usize::from(bytes.next().unwrap() % 96) + 1;
    let len = usize::from(bytes.next().unwrap() % 4) + 1;
    let mutation = bytes.next().unwrap() % 8;
    let left_bits = format!("{}N[{left_width}]", if flags & 1 == 0 { "s" } else { "u" });
    let right_bits = format!("{}N[{right_width}]", if flags & 2 == 0 { "s" } else { "u" });
    let out_bits = format!("uN[{result_width}]");
    let (right, pack_right, read_right) = match (flags >> 2) & 3 {
        0 => ("Right".into(), "Right { value:x }", "b.value"),
        1 => (
            format!("({right_bits},bool)"),
            "(x,x != 0)",
            "if b.1 {b.0} else {!b.0}",
        ),
        2 => (format!("{right_bits}[2]"), "[x,!x]", "b[u32:1]"),
        _ => (right_bits.clone(), "x", "b"),
    };
    let (out, pack_out) = match (flags >> 4) & 3 {
        0 => ("Output".into(), "Output { value:v, flag:v != 0 }"),
        1 => (format!("({out_bits},bool)"), "(v,v == 0)"),
        2 => (format!("{out_bits}[2]"), "[v,!v]"),
        _ => (out_bits.clone(), "v"),
    };
    let helpers = format!(
        r#"
struct Left {{ value:{left_bits} }}
struct Right {{ value:{right_bits} }}
struct Output {{ value:{out_bits}, flag:bool }}
fn left(x:{left_bits})->Left {{ Left {{ value:x }} }}
fn right(x:{right_bits})->{right} {{ {pack_right} }}
fn combine(a:Left,b:{right})->{out} {{
    let v = (a.value as {out_bits}) - (({read_right}) as {out_bits});
    {pack_out}
}}
fn wrong(a:(Left,Left),b:{right})->{out} {{ combine(a.0,b) }}
fn take_left(a:Left,_b:{right})->Left {{ a }}
"#
    );
    let selected = if mutation == 5 { "wrong" } else { "combine" };
    let actual_len = if mutation == 6 { len + 1 } else { len };
    let formal_len = if mutation == 7 { len + 1 } else { len };
    let params = format!("l:{left_bits}[{len}], r:{right_bits}[{actual_len}]");
    let result = format!("({out}[{len}],Left[{len}])");
    let bindings = if flags & 64 == 0 {
        "let xs=map(l,left); let ys=map(r,right);"
    } else {
        "let xs=map(map(l,left),id); let ys=id(map(r,right));"
    };
    let actuals = if flags & 128 == 0 && mutation != 7 {
        selected.to_owned()
    } else {
        format!("{selected},{formal_len}")
    };
    let bx_body = format!("{bindings} (zip_with<{actuals}>(xs,ys), zip_with<take_left>(xs,ys))");
    let bx = format!(
        "{helpers}\nfn id<T>(x:T)->T {{ x }}\n{ZIP_WITH}\n{}",
        bx_source(&params, &result, &bx_body)
    );
    // The reference spells out one call per element. Its concrete parameter
    // types enforce the common length and the selected callback signature.
    let elements = (0..formal_len)
        .map(|i| format!("{selected}(xs[u32:{i}],ys[u32:{i}])"))
        .collect::<Vec<_>>()
        .join(", ");
    let native = format!(
        "#![feature(generics)]\n{helpers}\nfn id<T:type>(x:T)->T {{ x }}\n\
         fn zip_ref(xs:Left[{formal_len}],ys:{right}[{formal_len}])->{out}[{formal_len}] {{ [{elements}] }}"
    );
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: native,
            params,
            result,
            body: format!("{bindings} (zip_ref(xs,ys),xs)"),
            valid: Some(mutation < 5),
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
    use super::check_function_type_parametrics;

    #[test]
    fn compare_generated_function_type_parametrics() -> Result<(), String> {
        for flags in [0, 23, 40, 61, 64, 128, 223] {
            for mutation in [0, 5, 6, 7] {
                check_function_type_parametrics(&[flags, 71, 7, 92, 2, mutation])?;
            }
        }
        Ok(())
    }
}
