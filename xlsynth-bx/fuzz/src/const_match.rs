// SPDX-License-Identifier: Apache-2.0

//! Compare specialization-time pattern selection and selected arm evaluation.

use crate::eval::expr_eval::{bx_source, dslx_source};
use crate::function_oracle::{DslxExecution, FunctionProgram, check_sources, compare_bx_program};
use crate::input_seed;

/// Helper declarations, template expression, and acceptance at one binding.
struct MatchCase {
    declarations: String,
    expression: String,
    valid: bool,
}

/// Vary patterns and result expressions independently of the caller and the
/// spelling of the constexpr subject.
fn match_case(family: u8, subject: &str, selector: u8, literal: u8) -> MatchCase {
    let first = literal % 4;
    let second = (first + 1) % 4;
    let mut declarations = "fn flip(z:u2) -> u2 { z ^ u2:1 }".to_string();
    let (expression, valid) = match family {
        0 => (
            format!(
                r#"let v = const match {subject} {{
                    {first} => (x, !x), {second} => [!x, x], _ => x ^ uN[N]:1
                }};
                const match {subject} {{ {first} => v.1, {second} => v[u32:1], _ => v }}"#
            ),
            true,
        ),
        1 => {
            let mut arms = ["u2:0..u2:2 => x", "u2:1..u2:3 => !x"];
            if literal & 1 != 0 {
                arms.swap(0, 1);
            }
            (
                format!(
                    "const match {subject} {{ {}, _ => x + uN[N]:1 }}",
                    arms.join(", ")
                ),
                true,
            )
        }
        2 => {
            declarations.push_str(&format!(
                "\nconst FIRST = u2:{first}; const SECOND = u2:{second};"
            ));
            (
                format!("const match {subject} {{ FIRST | SECOND => x, _ => !x }}"),
                true,
            )
        }
        3 => (
            format!(
                r#"const match ({subject}, (N, {subject})) {{
                    (0, (n, ..)) => x ^ (n as uN[N]),
                    (1, pair) => !x + (pair.0 as uN[N]),
                    (.., (_, t)) => x ^ (t as uN[N])
                }}"#
            ),
            true,
        ),
        4 => (
            format!("const match ({subject}, N) {{ (0, w) => x + uN[w]:1, _ => !x }}"),
            true,
        ),
        5 => (
            format!(
                "let offset:u16 = const match {subject} {{ 0 => 255 + 1, 1 => {literal}, _ => 9 }}; x ^ (offset as uN[N])"
            ),
            true,
        ),
        6 => (
            format!(
                r#"let neg = -(({subject} as s3) + s3:1);
                const match neg {{ s3:-4..s3:-2 => !x, s3:-3..s3:0 => x, _ => x ^ uN[N]:1 }}"#
            ),
            true,
        ),
        7 => {
            declarations
                .push_str("\nenum Mode:u2 { Keep=u2:0, Swap=u2:1, Invert=u2:2, Plus=u2:3 }");
            (
                format!(
                    "let mode = {subject} as Mode; const match mode {{ Mode::Keep | Mode::Plus => !x, _ => x }}"
                ),
                true,
            )
        }
        8 => {
            // The two pattern constants differ only above bit 63.
            let high = format!("uN[129]:0x1{:016x}", literal);
            declarations.push_str(&format!(
                "\nconst HIGH = {high}; const LOW = uN[129]:{literal};"
            ));
            (
                format!(
                    "let key = if {subject} == u2:0 {{ HIGH }} else {{ LOW }}; const match key {{ LOW => x, HIGH => !x }}"
                ),
                true,
            )
        }
        9 => (
            format!("const match {subject} {{ {first} | {second} => !x }}"),
            selector == first || selector == second,
        ),
        10 => ("const match x { _ => x }".into(), false),
        11 => (
            format!("const match {subject} {{ 0 => x, _ => (x,x) + x }}"),
            selector == 0,
        ),
        12 => (
            format!("const match {subject} {{ 0 => x, _ => {{ const_assert!(false); !x }} }}"),
            selector == 0,
        ),
        13 => (
            format!("const match {subject} {{ 0 => x, _ => missing }}"),
            false,
        ),
        14 => (
            format!("const match {subject} {{ 0 => x, u32:1 => !x, _ => x }}"),
            false,
        ),
        15 => (
            format!("const match ({subject}, N) {{ (0, field) => x, _ => field as uN[N] }}"),
            false,
        ),
        16 => (
            format!("const match {subject} {{ 0 => x, (a,b) => !x, _ => x }}"),
            false,
        ),
        // DSLX checks each alternative for full coverage. The first range
        // covers the entire literal 1, even though the alternative 2 is new.
        17 => (
            format!("const match {subject} {{ u2:0..u2:2 => x, u2:1 | u2:2 => !x, _ => x }}"),
            false,
        ),
        18 => {
            let dependent = match literal % 3 {
                0 => format!("{{ let _ = x; {subject} }}"),
                1 => format!("if true {{ {subject} }} else {{ x as u2 }}"),
                _ => format!(
                    "{{ let (_, key) = (if true {{ u2:1 }} else {{ x as u2 }}, {subject}); key }}"
                ),
            };
            (
                format!("const match {dependent} {{ {first} => x, _ => !x }}"),
                false,
            )
        }
        19 => (
            format!(
                "let key = {{ let _ = x; {subject} }}; const match key {{ {first} => x, _ => !x }}"
            ),
            true,
        ),
        _ => {
            let mask = literal | 0x80;
            let magnitude = 129 + literal % 127;
            (
                format!(
                    "let mask = 0x{mask:x}; let value:s8 = const match {subject} {{ {first} => s8:-{magnitude}, _ => mask }}; x ^ ((value as u8) as uN[N])"
                ),
                true,
            )
        }
    };
    MatchCase {
        declarations,
        expression,
        valid,
    }
}

/// Run a generated const match in a parametric BX function or block and the
/// equivalent DSLX function at the same concrete parameter bindings.
pub fn check_const_match(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let family = bytes.next().unwrap() % 21;
    let flags = bytes.next().unwrap();
    let selector = flags % 4;
    let block = flags & 4 != 0;
    let width = usize::from(bytes.next().unwrap() % 129) + 1;
    let literal = bytes.next().unwrap();
    let (setup, subject) = match (flags >> 3) % 4 {
        0 => ("", "S"),
        1 => ("let tag = S;", "tag"),
        2 if block => ("let tag = flip(S) ^ u2:1;", "tag"),
        2 => ("const TAG = S;", "TAG"),
        _ => ("let tag = flip(flip(S));", "tag"),
    };
    let MatchCase {
        declarations,
        expression,
        valid,
    } = match_case(family, subject, selector, literal);
    // BX block scalar value parameters have type u32 or bool. Both templates
    // match the same two-bit selector after binding the concrete argument.
    let (signature, selector_type, prefix) = if block {
        ("choose<N:u32, K:u32>", "u32", "let S = K as u2;")
    } else {
        ("choose<N:u32, S:u2>", "u2", "")
    };
    let template = format!("{{ {prefix}\n{setup}\n{expression} }}");
    let mut functions = format!("{declarations}\nfn {signature}(x:uN[N]) -> uN[N] {template}");
    let bx_decls = if block {
        format!(
            "{declarations}\n#[allow(combinational_block)]\nblock {signature}(clk, x:uN[N]) -> uN[N] {template}"
        )
    } else {
        functions.clone()
    };
    let word = format!("uN[{width}]");
    let params = format!("x:{word}");
    let (result, body) = if family < 9 {
        // Successive specializations and a repeated call expose stale selected
        // arms or pattern bindings, even when their port widths coincide.
        (
            format!("({word}, {word}, {word}, {word}, {word})"),
            format!(
                r#"let a = choose<u32:{width}, {selector_type}:0>(x);
                let b = choose<u32:{width}, {selector_type}:1>(x);
                let c = choose<u32:{width}, {selector_type}:2>(x);
                let d = choose<u32:{width}, {selector_type}:3>(x);
                let e = choose<u32:{width}, {selector_type}:{selector}>(!x);
                (a, b, c, d, e)"#
            ),
        )
    } else {
        (
            word,
            format!("let selected = choose<u32:{width}, {selector_type}:{selector}>(x); selected"),
        )
    };
    let bx = format!("{bx_decls}\n{}", bx_source(&params, &result, &body));
    if family == 3 {
        // v0.65.2 accepts these tuple binders, but its bytecode emitter and
        // IR converter cannot read them as arm values in a const match.
        // Ordinary match selects the same arm on the same concrete subject.
        let reference = format!("{functions}\n{}", dslx_source(&params, &result, &body));
        check_sources(&bx, &reference, Some(true), &[])?;
        functions = functions.replace("const match ", "match ");
    }
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions,
            params,
            result,
            body,
            valid: Some(valid),
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
    use super::check_const_match;

    #[test]
    fn compare_selected_arm_types_values_and_errors() -> Result<(), String> {
        for family in 0..21 {
            for flags in [0, 1, 6, 7, 10, 13, 16, 23, 25, 28] {
                check_const_match(&[
                    family,
                    flags,
                    if flags & 1 == 0 { 7 } else { 128 },
                    92 + (flags & 1),
                ])?;
            }
        }
        Ok(())
    }
}
