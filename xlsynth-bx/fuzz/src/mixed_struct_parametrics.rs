// SPDX-License-Identifier: Apache-2.0

//! Compare mixed struct specialization with native DSLX's generics feature.

use crate::eval::expr_eval::bx_source;
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};
use crate::input_seed;
use crate::struct_type_parameters::Payload;

/// Spell a constructor or type application in the declaration's positional
/// order.
fn application(type_first: bool, ty: &str, length: &str, tag: &str, prefix: usize) -> String {
    let args = if type_first {
        [ty, length, tag]
    } else {
        [length, ty, tag]
    };
    if prefix == 0 {
        "Packet".to_owned()
    } else {
        format!("Packet<{}>", args[..prefix].join(", "))
    }
}

/// Compare positional binding, field inference, defaults and nominal identity.
pub fn check_mixed_struct_parametrics(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let width = usize::from(bytes.next().unwrap() % 129) + 1;
    let shapes = bytes.next().unwrap();
    let lengths = bytes.next().unwrap();
    let tag = usize::from(bytes.next().unwrap());
    let mutation = bytes.next().unwrap() % 12;
    let payload_typed_tag = bytes.next().unwrap() & 1 != 0;
    let type_first = flags & 1 == 0;
    let annotated = flags & 2 != 0;
    let left_prefix = usize::from((flags >> 2) & 3);
    // With a partially applied constructor and a default K: T, pinned DSLX
    // can constrain a second instantiation to the first one's signedness.
    // Sample signedness across programs and tag values across calls.
    let reuse = payload_typed_tag || flags & 16 != 0;
    let left_length = usize::from(lengths & 3) + 1;
    let right_length = usize::from((lengths >> 2) & 3) + 1;
    let left_tag = if annotated || left_prefix == 3 {
        tag
    } else if payload_typed_tag {
        0
    } else {
        left_length + 1
    };
    let right_tag = if flags & 32 == 0 { tag } else { left_tag + 1 };
    let shape_mask = if payload_typed_tag { 1 } else { 15 };
    let left = Payload::new(shapes & shape_mask, width, "LeftPayload", "a[u32:0]");
    let right = if reuse {
        Payload::new(shapes & shape_mask, width, "LeftPayload", "b[u32:0]")
    } else {
        Payload::new(
            (shapes >> 4) & shape_mask,
            width,
            "RightPayload",
            "b[u32:0]",
        )
    };
    let number = |value: usize| {
        if flags & 64 == 0 {
            value.to_string()
        } else {
            format!("u32:{value}")
        }
    };
    let formals = if type_first {
        "T: type, N: u32"
    } else {
        "N: u32, T: type"
    };
    let (tag_type, tag_default) = if payload_typed_tag {
        ("T", "zero!<T>()")
    } else {
        ("u32", "N + u32:1")
    };
    let tag_argument = |payload: &Payload, value: usize| {
        if payload_typed_tag {
            // Hex bit patterns represent both signs, including sN[1]:0x1.
            let bits = value & ((1 << width.min(9)) - 1);
            format!("{}:0x{bits:x}", payload.ty)
        } else {
            number(value)
        }
    };
    let symbolic = application(type_first, "T", "N", "K", 3);
    let right_decl = if reuse { "" } else { &right.declaration };
    let declarations = format!(
        r#"{}
{right_decl}
struct Packet<{formals}, K: {tag_type} = {{{tag_default}}}> {{ data: T, lanes: T[N] }}
fn inspect<T: type, N: u32, K: {tag_type}>(p: {symbolic}) -> (T, T, u32, {tag_type}) {{
    (p.data, p.lanes[N - u32:1], N, K)
}}"#,
        left.declaration
    );
    let left_arg = |prefix| {
        application(
            type_first,
            &left.ty,
            &number(left_length),
            &tag_argument(&left, left_tag),
            prefix,
        )
    };
    let left_annotation = if annotated {
        format!(": {}", left_arg(3))
    } else {
        String::new()
    };
    let annotation_type = if mutation == 6 { &left.ty } else { &right.ty };
    let annotation_tag = if mutation == 7 {
        right_tag ^ 1
    } else {
        right_tag
    };
    let right_annotation = application(
        type_first,
        annotation_type,
        &number(right_length),
        &tag_argument(&right, annotation_tag),
        3,
    );
    let constructor_type = if mutation == 8 { "u32:1" } else { &right.ty };
    let constructor_length = if mutation == 9 {
        "u8".into()
    } else {
        number(right_length)
    };
    let mut right_constructor = application(
        type_first,
        constructor_type,
        &constructor_length,
        &tag_argument(&right, right_tag),
        3,
    );
    if mutation == 10 {
        right_constructor.pop();
        right_constructor.push_str(", u32:0>");
    }
    let right_data = if mutation == 11 {
        &left.argument
    } else {
        &right.argument
    };
    let params = format!(
        "a: {}[{left_length}], b: {}[{right_length}]",
        left.input, right.input
    );
    let row = |payload: &Payload| {
        let tag = if payload_typed_tag {
            &payload.ty
        } else {
            "u32"
        };
        format!("({}, {}, u32, {tag})", payload.input, payload.input)
    };
    let result = format!("({}, {}, {})", row(&left), row(&right), row(&left));
    let projection = |name: &str, payload: &Payload| {
        let field = payload.projection;
        format!("({name}.0{field}, {name}.1{field}, {name}.2, {name}.3)")
    };
    let body = format!(
        r#"let left{left_annotation} = {} {{ data: {}, lanes: {} }};
    let right: {right_annotation} = {right_constructor} {{ lanes: {}, data: {right_data} }};
    let updated = Packet {{ data: left.lanes[u32:{}], ..left }};
    let l = inspect(left);
    let r = inspect(right);
    let u = inspect(updated);
    ({}, {}, {})"#,
        left_arg(left_prefix),
        left.argument,
        left.array_argument("a", left_length),
        right.array_argument("b", right_length),
        left_length - 1,
        projection("l", &left),
        projection("r", &right),
        projection("u", &left),
    );
    let mut bx = format!("{declarations}\n{}", bx_source(&params, &result, &body));
    if flags & 128 != 0 {
        // BX also accepts abbreviated type formals and constructor turbofish.
        bx = bx.replace("T: type", "T");
        // Keep the declaration's plain angle brackets.
        bx = bx
            .replace("Packet<", "Packet::<")
            .replace("struct Packet::<", "struct Packet<");
    }
    let valid = match mutation {
        6 | 11 => left.ty == right.ty,
        7..=10 => false,
        _ => true,
    };
    compare_bx_program(
        &bx,
        FunctionProgram {
            functions: format!("#![feature(generics)]\n{declarations}"),
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
    use super::{bx_source, check_mixed_struct_parametrics};
    use crate::function_oracle::{DslxExecution, FunctionProgram, compare_bx_program};

    #[test]
    fn mixed_contextual_witnesses_match_dslx() -> Result<(), String> {
        let declarations = r#"
struct Row<N: u32, T: type, W: u32 = {bit_count<T>()}> { xs: T[N] }
fn inspect<N: u32, T: type, W: u32>(r: Row<N, T, W>, witness: T) -> (u32, u32, T) {
    (N, W, witness)
}
fn contextual(x: u8) -> (u32, u32, u8) {
    inspect({ const COUNT = u32:2; Row<COUNT> { xs: [254, 255] } }, x)
}
"#;
        let params = "x: u8";
        let result = "(u32, u32, u8)";
        let body = "contextual(x)";
        compare_bx_program(
            &format!("{declarations}\n{}", bx_source(params, result, body)),
            FunctionProgram {
                functions: format!("#![feature(generics)]\n{declarations}"),
                params: params.into(),
                result: result.into(),
                body: body.into(),
                valid: Some(true),
                dslx_ir: true,
                dynamic_index: None,
            },
            0,
            DslxExecution::ConcreteConstants,
            Vec::new(),
            &[],
        )
    }

    #[test]
    fn mixed_specializations_match_pinned_dslx() -> Result<(), String> {
        for order in 0..2 {
            for prefix in 0..4 {
                for annotate in [0, 2] {
                    let flags = order | (prefix << 2) | annotate | 0xe0;
                    check_mixed_struct_parametrics(&[flags, 128, 0x8a, 0x0b, 37])?;
                }
            }
            for mutation in 6..12 {
                check_mixed_struct_parametrics(&[order, 7, 0x88, 0, 9, mutation])?;
            }
            check_mixed_struct_parametrics(&[order | 16, 7, 0x88, 0, 1])?;
            for width_byte in [0, 7, 128] {
                for signed in [0, 1] {
                    for mutation in [0, 7] {
                        check_mixed_struct_parametrics(&[
                            order | 4,
                            width_byte,
                            signed,
                            6,
                            255,
                            mutation,
                            1,
                        ])?;
                    }
                }
            }
        }
        Ok(())
    }
}
