// SPDX-License-Identifier: Apache-2.0

//! Compare native BX struct specializations with the same declarations in DSLX.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program, compare_program};
use crate::input_seed;

/// Generate two instances of a packet template and a nested frame template.
fn packet_program(data: &[u8], turbofish: bool) -> FunctionProgram {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = u32::from(bytes.next().unwrap() % 96) + 1;
    let width_delta = u32::from(bytes.next().unwrap() % 32) + 1;
    let length = (width & 3) + 1;
    let right_width = if flags & 1 == 0 {
        width + width_delta
    } else {
        width
    };
    let right_length = (length % 4) + 1;
    let sign = if flags & 2 == 0 { "u" } else { "s" };
    let separator = if turbofish { "::" } else { "" };
    let (helper, length_default) = if data.get(4).is_some_and(|flags| flags & 2 != 0) {
        (
            "fn label_count<N: u32>() -> u32 { (N & u32:3) + u32:1 }",
            format!("label_count{separator}<N>()"),
        )
    } else {
        ("", "(N & u32:3) + u32:1".to_owned())
    };
    let (signature_arg, constructor_arg) = if flags & 4 == 0 {
        ("", "u32:")
    } else {
        ("u32:", "")
    };
    let left = if flags & 64 == 0 {
        format!("Packet{separator}<{signature_arg}{width}>")
    } else {
        format!("Packet{separator}<{{BASE_WIDTH + u32:{}}}>", width - 1)
    };
    let right =
        format!("Packet{separator}<{signature_arg}{right_width}, {signature_arg}{right_length}>");
    let left_ctor = match (flags >> 3) & 3 {
        0 => format!("Packet{separator}<{constructor_arg}{width}, {constructor_arg}{length}>"),
        1 => format!("Packet{separator}<{constructor_arg}{width}>"),
        _ => "Packet".to_owned(),
    };
    let mutation = bytes.next().unwrap() % 12;
    let right_ctor = if mutation == 9 {
        format!(
            "Packet{separator}<{constructor_arg}{right_width}, {constructor_arg}{right_length}, u32:1>"
        )
    } else {
        format!(
            "Packet{separator}<{constructor_arg}{right_width}, {constructor_arg}{right_length}>"
        )
    };
    let frame_ctor = if flags & 32 == 0 {
        format!("Frame{separator}<N, M>")
    } else {
        "Frame".to_owned()
    };
    let right_source = if mutation == 8 { "x" } else { "y" };
    let functions = format!(
        r#"
const BASE_WIDTH = u32:1;
{helper}
struct Packet<N: u32, M: u32 = {{{length_default}}}> {{
    data: {sign}N[N],
    labels: u8[M],
}}
struct Frame<N: u32, M: u32> {{
    packets: (Packet{separator}<N, M>[2], bool),
}}
struct Other<N: u32, M: u32> {{
    data: {sign}N[N],
    labels: u8[M],
}}
fn make_left(x: {sign}N[{width}], labels: u8[{length}]) -> {left} {{
    let packet = {left_ctor} {{ labels: labels, data: x }};
    packet
}}
fn invert<N: u32, M: u32>(p: Packet{separator}<N, M>) -> Packet{separator}<N, M> {{
    Packet{separator}<N, M> {{ data: !p.data, labels: p.labels }}
}}
fn pack<N: u32, M: u32>(p: Packet{separator}<N, M>, valid: bool) -> Frame{separator}<N, M> {{
    {frame_ctor} {{ packets: ([p, invert{separator}<N, M>(p)], valid) }}
}}
fn unpack<N: u32, M: u32>(f: Frame{separator}<N, M>) -> (Packet{separator}<N, M>[2], bool) {{
    f.packets
}}
fn inspect<N: u32, M: u32>(p: Packet{separator}<N, M>) -> ({sign}N[N], u8[M], u32, u32) {{
    (p.data, p.labels, N, M)
}}
fn read_left(p: {left}) -> ({sign}N[{width}], u8[{length}], u32, u32) {{
    inspect(p)
}}
"#
    );
    let left_expr = match mutation {
        6 => "right".to_owned(),
        7 => format!("Other{separator}<{width}, {length}> {{ data: x, labels: labels }}"),
        _ => "make_left(x, labels)".to_owned(),
    };
    let params = format!(
        "x: {sign}N[{width}], y: {sign}N[{right_width}], labels: u8[{length}], other: u8[{right_length}], valid: bool"
    );
    let left_result = format!("({sign}N[{width}], u8[{length}], u32, u32)");
    let result = format!(
        "({left_result}, {left_result}, ({sign}N[{right_width}], u8[{right_length}], u32, u32), bool)"
    );
    let body = format!(
        r#"
    let right: {right} = {right_ctor} {{ labels: other, data: {right_source} }};
    let left: {left} = {left_expr};
    let (copies, flag) = unpack(pack(left, valid));
    (read_left(copies[u32:0]), inspect(copies[u32:1]), inspect(right), flag)
"#
    );
    FunctionProgram {
        functions,
        params,
        result,
        body,
        valid: Some(match mutation {
            6 | 7 | 9 => false,
            8 => width == right_width,
            _ => true,
        }),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Check how constructor fields bind an unsigned or signed width in declaration
/// order.
fn literal_program(data: &[u8]) -> FunctionProgram {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let magnitude = u32::from(bytes.next().unwrap()) + 1;
    let extra_width = u32::from(bytes.next().unwrap() % 8) + 1;
    let mode = bytes.next().unwrap() % 4;
    let (sign, literal, width) = if flags & 2 == 0 {
        ("u", magnitude.to_string(), magnitude.ilog2() + 1)
    } else {
        let width = u32::BITS - (magnitude - 1).leading_zeros() + 1;
        ("s", format!("-{magnitude}"), width)
    };
    let typed_width = if mode >= 2 {
        width + extra_width
    } else {
        width
    };
    let (first, second, inferred_width) = match mode {
        0 => (literal.as_str(), literal.as_str(), width),
        2 => ("x", literal.as_str(), typed_width),
        _ => (literal.as_str(), "x", width),
    };
    let fields = if flags & 1 == 0 {
        format!("first: {first}, second: {second}")
    } else {
        format!("second: {second}, first: {first}")
    };
    let functions = format!(
        r#"
struct Record<N: u32> {{ first: {sign}N[N], second: {sign}N[N] }}
fn inspect<N: u32>(r: Record<N>) -> ({sign}N[N], {sign}N[N], u32) {{
    (r.first, r.second, N)
}}
"#
    );
    let body = format!(
        r#"
    let r = Record {{ {fields} }};
    let (first, second, width) = inspect(r);
    (first, second, width, x)
"#
    );
    FunctionProgram {
        functions,
        params: format!("x: {sign}N[{typed_width}]"),
        result: format!(
            "({sign}N[{inferred_width}], {sign}N[{inferred_width}], u32, {sign}N[{typed_width}])"
        ),
        body,
        valid: Some(mode != 3),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Compare source acceptance, lowered IR, and concrete calls of each frontend.
pub fn check_struct_parametrics(data: &[u8]) -> Result<(), String> {
    let seed = input_seed(data);
    let program = if data.first().is_some_and(|flags| flags & 128 != 0) {
        literal_program(data)
    } else {
        let program = packet_program(data, false);
        if data.get(4).is_some_and(|flags| flags & 1 != 0) {
            let bx = packet_program(data, true);
            let source = format!(
                "{}\n{}",
                bx.functions,
                bx_source(&bx.params, &bx.result, &bx.body)
            );
            return compare_bx_program(
                &source,
                program,
                seed,
                DslxExecution::ConcreteConstants,
                Vec::new(),
                &[],
            );
        }
        program
    };
    compare_program(program, seed, DslxExecution::ConcreteConstants, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::check_struct_parametrics;

    #[test]
    fn replay_native_parametric_structs() -> Result<(), String> {
        for flags in [0, 7, 8, 15, 16, 23, 32, 55, 64, 119] {
            for mutation in [0, 6, 7, 8, 9] {
                for mode in [0, 1, 2, 3] {
                    check_struct_parametrics(&[flags, 70, 16, mutation, mode])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn replay_struct_literal_field_inference() -> Result<(), String> {
        for flags in 128..132 {
            for magnitude in [0, 130] {
                for mode in 0..4 {
                    check_struct_parametrics(&[flags, magnitude, 3, mode])?;
                }
            }
        }
        Ok(())
    }
}
