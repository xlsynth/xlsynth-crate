// SPDX-License-Identifier: Apache-2.0

//! Compare DSLX-style struct updates in the BX and pinned DSLX frontends.

use std::fmt::Write;

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use crate::input_seed;

/// Generate the record, its base expression, and a varying set of overrides.
fn update_program(data: &[u8]) -> FunctionProgram {
    let [flags, width, lanes, mask, order, mutation] =
        std::array::from_fn(|index| data.get(index).copied().unwrap_or(0));
    let width = usize::from(width % 96) + 1;
    let lanes = (lanes % 4) + 1;
    let mutation = mutation % 16;
    let parametric = flags & 1 != 0;
    let sign = if flags & 2 == 0 { 'u' } else { 's' };
    let word = format!("{sign}N[{width}]");
    let metadata = |size: &str| {
        if flags & 4 == 0 {
            format!("(bool, u8[{size}])")
        } else {
            format!("(bool, (u8[{size}], u3))")
        }
    };
    let concrete_metadata = metadata(&lanes.to_string());
    let declaration = |name| {
        let (params, member_word, member_metadata) = if parametric {
            ("<N: u32, L: u32>", format!("{sign}N[N]"), metadata("L"))
        } else {
            ("", word.clone(), concrete_metadata.clone())
        };
        format!(
            "struct {name}{params} {{ word: {member_word}, meta: {member_metadata}, tag: u16, valid: bool }}\n"
        )
    };
    let specialization = |name: &str, size: usize| {
        if parametric {
            format!("{name}<{size}, {lanes}>")
        } else {
            name.to_owned()
        }
    };
    let packet = specialization("Packet", width);
    let mut functions = declaration("Packet");
    let initial_fields = "word: x, meta: before, tag, valid: a";
    let mut body = format!("let base = {packet} {{ {initial_fields} }};\n");
    let base_form = (flags >> 4) & 3;
    let valid_update = matches!(mutation, 0 | 7..=15);
    let context = data.get(6).copied().unwrap_or(0) % 7;
    let mut base = match (mutation, parametric, base_form) {
        (3, _, _) => {
            functions.push_str(&declaration("Other"));
            let other = specialization("Other", width);
            writeln!(body, "let other = {other} {{ {initial_fields} }};").unwrap();
            "other".to_owned()
        }
        (5, true, _) => {
            let wider = specialization("Packet", width + 1);
            let wider_word = format!("{sign}N[{}]", width + 1);
            writeln!(
                body,
                "let wider = {wider} {{ word: x as {wider_word}, meta: before, tag, valid: a }};"
            )
            .unwrap();
            "wider".to_owned()
        }
        (0 | 7..=15, _, 1) => {
            writeln!(
                functions,
                "fn make_packet(x: {word}, before: {concrete_metadata}, tag: u16, a: bool) -> {packet} {{ {packet} {{ {initial_fields} }} }}"
            )
            .unwrap();
            "make_packet(x, before, tag, a)".to_owned()
        }
        (0 | 7..=15, _, 2) => format!(
            "(if a {{ base }} else {{ {packet} {{ word: y, meta: after, tag: next_tag, valid: b }} }})"
        ),
        (0 | 7..=15, true, 3) => {
            let literal = "Packet { word: 0, meta: before, tag, valid: a }";
            match context {
                1 => format!("for (_, p) in u32:0..u32:1 {{ p }}({literal})"),
                2 => format!("unroll_for! (_, p) in u32:0..u32:1 {{ p }}({literal})"),
                _ => literal.to_owned(),
            }
        }
        _ => "base".to_owned(),
    };
    // The pinned typechecker needs a named base for computed parametric
    // updates. Both programs still compute the base from the same inputs.
    if parametric && valid_update && matches!(base_form, 1 | 2) {
        writeln!(body, "let computed = {base};").unwrap();
        base = "computed".to_owned();
    }
    let overrides = match mutation {
        1 => "word: x, word: y, ".to_owned(),
        2 => "unknown: y, ".to_owned(),
        4 => format!("word: y as {sign}N[{}], ", width + 1),
        5 if parametric => "tag: next_tag, ".to_owned(),
        _ => {
            let fields = ["word: y", "meta: after", "tag: next_tag", "valid: b"];
            // Include the redundant splat with all four fields overridden.
            // The reference API returns its value even when DSLX warns.
            let mask = mask & 15;
            let mut overrides = (0..fields.len())
                .filter(|index| (mask & (1 << index)) != 0)
                .map(|index| fields[index])
                .collect::<Vec<_>>();
            if !overrides.is_empty() {
                let offset = usize::from(order >> 1) % overrides.len();
                overrides.rotate_left(offset);
                if order & 1 != 0 {
                    overrides.reverse();
                }
                // Leave a separator before the trailing splat expression.
                overrides.push("");
            }
            overrides.join(", ")
        }
    };
    let partial = format!("Packet<{width}>");
    let constructor = if parametric && flags & 128 != 0 && !matches!(mutation, 4 | 5) {
        &partial
    } else if flags & 8 == 0 || mutation == 4 || mutation == 5 {
        &packet
    } else {
        "Packet"
    };
    let comma = if mutation == 6 { "," } else { "" };
    let update = format!("{constructor} {{ {overrides}..{base}{comma} }}");
    // Supply the nested literal's width through a result annotation, a
    // sibling expression, or the explicit update constructor.
    if parametric && valid_update && base_form == 3 && constructor == "Packet" {
        match context {
            3 => writeln!(body, "let updated = if a {{ ({update}) }} else {{ base }};"),
            4 => writeln!(body, "let updated = if a {{ base }} else {{ ({update}) }};"),
            5 => writeln!(body, "let updated = [{update}, base][u32:0];"),
            6 => writeln!(body, "let updated = [base, {update}][u32:1];"),
            _ => writeln!(body, "let updated: {packet} = {update};"),
        }
        .unwrap();
    } else {
        writeln!(body, "let updated = {update};").unwrap();
    }
    body.push_str("(updated.valid, updated.word, updated.meta, updated.tag)");
    let params = format!(
        "x: {word}, y: {word}, before: {concrete_metadata}, after: {concrete_metadata}, tag: u16, next_tag: u16, a: bool, b: bool"
    );
    let result = format!("(bool, {word}, {concrete_metadata}, u16)");
    if flags & 64 != 0 {
        writeln!(
            functions,
            "fn apply_update({params}) -> {result} {{\n{body}\n}}"
        )
        .unwrap();
        body = "apply_update(x, y, before, after, tag, next_tag, a, b)".to_owned();
    }
    FunctionProgram {
        functions,
        params,
        result,
        body,
        valid: Some(match mutation {
            1..=4 | 6 => false,
            5 => !parametric,
            _ => true,
        }),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Compare source acceptance and every resulting field on shared runtime
/// inputs.
pub fn check_struct_updates(data: &[u8]) -> Result<(), String> {
    compare_program(
        update_program(data),
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}
