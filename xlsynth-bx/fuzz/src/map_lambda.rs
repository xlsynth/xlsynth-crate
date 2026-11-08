// SPDX-License-Identifier: Apache-2.0

//! Compare anonymous `map` calls and their lexical captures with pinned DSLX.

use crate::eval::expr_eval::{bx_source, dslx_source};
use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use crate::input_seed;
use xlsynth::dslx::{ImportData, MatchableModuleMember, parse_and_typecheck};
use xlsynth_bx::ast::Type;
use xlsynth_bx::{check_module, parse_module};

/// Vary array and result shapes, lexical captures, nested maps and helpers.
pub fn check_map_lambda(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let shape = bytes.next().unwrap() % 7;
    if shape == 6 {
        return check_counted_captures(data);
    }
    if shape == 5 {
        return check_constant_result(data);
    }
    if shape >= 3 {
        return check_inferred_literal_map(data, shape == 4);
    }
    let flags = bytes.next().unwrap();
    let in_width = usize::from(bytes.next().unwrap() % 128) + 1;
    let out_width = usize::from(bytes.next().unwrap() % 128) + 1;
    let length = usize::from(bytes.next().unwrap() % 4) + 1;
    let output_shape = bytes.next().unwrap() % 3;
    let mutation = bytes.next().unwrap() % 16;
    let length = if mutation == 15 { 0 } else { length };
    let in_sign = if flags & 1 == 0 { 'u' } else { 's' };
    let out_sign = if flags & 2 == 0 { 'u' } else { 's' };
    let input_bits = format!("{in_sign}N[{in_width}]");
    let output_bits = format!("{out_sign}N[{out_width}]");

    let element = match shape {
        0 => input_bits.clone(),
        1 => format!("({input_bits}, bool)"),
        _ => format!("{input_bits}[2]"),
    };
    let leaf_result = match output_shape {
        0 => output_bits.clone(),
        1 => format!("({output_bits}, bool)"),
        _ => format!("{output_bits}[2]"),
    };
    let result_element = if shape == 2 {
        format!("{leaf_result}[2]")
    } else {
        leaf_result.clone()
    };
    let result = format!("{result_element}[{length}]");
    let params = format!(
        "left: {element}[{length}], right: {element}[{length}], pick: bool, bias: {output_bits}"
    );
    let selected = if shape == 0 && flags & 128 != 0 && in_width >= 4 {
        let start: i32 = if in_sign == 's' { -2 } else { 1 };
        let inclusive = bytes.next().unwrap() % 2 != 0;
        let end = start + length as i32 - i32::from(inclusive);
        let operator = if inclusive { "..=" } else { ".." };
        format!("({input_bits}:{start}{operator}{input_bits}:{end})")
    } else if flags & 4 == 0 {
        "left".to_owned()
    } else {
        "if pick {left} else {right}".to_owned()
    };
    let mut argument = selected.as_str();
    let raw = if shape == 1 { "item.0" } else { "item" };
    let converted = if flags & 8 == 0 {
        format!("{raw} as {output_bits}")
    } else {
        format!("convert({raw})")
    };
    let mut functions = if flags & 8 == 0 {
        String::new()
    } else {
        format!("fn convert(x: {input_bits}) -> {output_bits} {{ x as {output_bits} }}")
    };
    let value = if shape == 1 {
        format!("if item.1 {{ {converted} }} else {{ !({converted}) }}")
    } else {
        converted
    };
    let leaf = match output_shape {
        0 => "value ^ (value >> u8:1)",
        1 => "(value, value == 0)",
        _ => "[value, !value]",
    };
    let mut enclosing_let = "";
    let mut outer_lambda_let = "";
    let value = match mutation {
        12 => format!("if pick {{ ({value}) + bias }} else {{ !({value}) ^ bias }}"),
        13 => {
            // The leaf's let reads this aggregate and then shadows it.
            enclosing_let = "let value = (pick, bias);";
            format!("if value.0 {{ ({value}) ^ value.1 }} else {{ ({value}) + value.1 }}")
        }
        14 if shape == 2 => {
            // The inner map reads a different first element for each row.
            // Only the inner callback refers to the enclosing bias.
            outer_lambda_let = "let first = item[u32:0];";
            format!("(({value}) + (first as {output_bits})) ^ bias")
        }
        14 => {
            enclosing_let = "let table = [bias, !bias];";
            format!("({value}) ^ table[pick as u32]")
        }
        _ => value,
    };
    let leaf_body = format!("let value = {value}; {leaf}");
    let mut body = if shape == 2 {
        format!("{outer_lambda_let} map(item, |item| -> {leaf_result} {{ {leaf_body} }})")
    } else {
        leaf_body
    };
    let mut param_ty = if flags & 16 == 0 {
        String::new()
    } else {
        format!(": {element}")
    };
    // Pinned DSLX cannot concretize the result of a captured callback that
    // returns an inferred tuple or array literal.
    let captured = (12..=14).contains(&mutation);
    let needs_result_type = captured && shape != 2 && output_shape != 0;
    let mut return_ty = if flags & 32 == 0 && !needs_result_type {
        String::new()
    } else {
        format!(" -> {result_element}")
    };
    match mutation {
        8 => param_ty = format!(": uN[{}]", in_width + 1),
        9 => return_ty = format!(" -> uN[{}]", out_width + 1),
        10 => argument = "pick",
        11 => body = "unknown_name".into(),
        _ => { /* Leave the generated lambda intact, including for the empty array case. */ }
    }
    let expression =
        format!("{enclosing_let} map({argument}, |item{param_ty}|{return_ty} {{ {body} }})");
    let body = if flags & 64 == 0 {
        expression
    } else {
        functions.push_str(&format!(
            "\nfn transform({params}) -> {result} {{ {expression} }}"
        ));
        "transform(left, right, pick, bias)".to_owned()
    };
    compare_program(
        FunctionProgram {
            functions,
            params,
            result,
            body,
            valid: Some(mutation < 8 || captured),
            // The pinned converter includes an unparsable
            // TypeVariableTypeAnnotation in synthetic function names for
            // inferred tuple/array literals. Concrete DSLX calls still check
            // those result shapes and values.
            dslx_ir: shape == 2 || output_shape == 0 || !return_ty.is_empty(),
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

/// Literal operands determine their width before the lambda is called. The
/// pinned DSLX typechecker accepts a wider annotation but passes narrow values.
fn check_inferred_literal_map(data: &[u8], range: bool) -> Result<(), String> {
    let mut bytes = data.iter().copied().skip(1).chain(std::iter::repeat(0));
    let flags = bytes.next().unwrap();
    let mode = flags % 3;
    let width = 2 + usize::from(bytes.next().unwrap() % 7);
    let high = 1usize << (width - 1);
    let operand = if range {
        format!("{}..{}", high - 1, high + 1)
    } else {
        let first = usize::from(bytes.next().unwrap()) % high;
        let second = high + usize::from(bytes.next().unwrap()) % high;
        format!("[{first},{second}]")
    };
    let annotated = if mode == 2 {
        width + 1 + usize::from(bytes.next().unwrap() % 64)
    } else {
        width
    };
    let result = format!("uN[{annotated}][2]");
    let annotation = if mode == 0 {
        String::new()
    } else {
        format!(": uN[{annotated}]")
    };
    if mode != 2 && flags & 4 != 0 {
        let ty = format!("uN[{width}]");
        let bias = usize::from(bytes.next().unwrap()) % (1 << width);
        let functions = format!(
            "fn transform(bias: {ty}) -> {result} {{ map({operand}, |x{annotation}| x ^ bias) }}\n\
             const MAPPED = transform({ty}:{bias});"
        );
        return compare_program(
            FunctionProgram {
                functions,
                params: format!("bias: {ty}"),
                result: format!("({result}, {result})"),
                body: "(transform(bias), MAPPED)".into(),
                valid: Some(true),
                dslx_ir: true,
                dynamic_index: None,
            },
            input_seed(data),
            DslxExecution::ConcreteConstants,
            Vec::new(),
        );
    }
    let calculation = if mode == 2 { "x" } else { "x + 1" };
    let body = format!("map({operand}, |x{annotation}| {calculation})");
    if mode != 2 {
        return compare_literal_program(data, "", result, body, true);
    }
    let bx = bx_source("", &result, &body);
    let reference = format!(
        "{}\nconst OBSERVED = sample();",
        dslx_source("", &result, &body)
    );
    let inferred = Type::Bits {
        signed: false,
        width,
    };
    let parameter = Type::Bits {
        signed: false,
        width: annotated,
    };
    check_narrow_native_result(
        &bx,
        &reference,
        &format!("map lambda parameter expects {parameter}, got {inferred}"),
        width,
    )
}

/// Compare captures of the counted index, changing carry and loop-local lets.
fn check_counted_captures(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().skip(1).chain(std::iter::repeat(0));
    let mode = bytes.next().unwrap() % 4;
    let count = 1 + bytes.next().unwrap() % 4;
    let value = bytes.next().unwrap();
    let value_width = (u8::BITS - value.leading_zeros()) as usize;
    let index_width = (u8::BITS - count.leading_zeros()) as usize;
    let ty = Type::Bits {
        signed: false,
        width: value_width.max(index_width) + 1 + usize::from(bytes.next().unwrap() % 96),
    };
    let range = if mode == 0 {
        format!("{ty}:0..{ty}:{count}")
    } else {
        format!("0..{count}")
    };
    let local = if mode == 0 {
        format!("acc[u32:1] ^ {ty}:{value}")
    } else {
        value.to_string()
    };
    let callback = match mode {
        0 => "((x ^ acc[u32:0]) + local) ^ i".to_owned(),
        1 => format!("((x ^ acc[u32:0]) + (local as {ty})) ^ (i as {ty})"),
        2 => "local".to_owned(),
        _ => "i".to_owned(),
    };
    let callback = match bytes.next().unwrap() % 3 {
        1 => format!("let alias = {callback}; alias"),
        2 => format!("let (alias,) = ({callback},); alias"),
        _ => callback,
    };
    let result = format!("{ty}[2]");
    let params = format!("xs: {result}");
    let body = format!(
        "for (i, acc) in {range} {{ let local = {local}; map(acc, |x| -> {ty} {{ {callback} }}) }}(xs)"
    );
    if mode < 2 {
        return compare_literal_program(data, &params, result, body, true);
    }
    // Pinned DSLX accepts the wider callback result but its bytecode returns
    // the capture's materialized width. Converting this counted carry can
    // abort.
    let bx = bx_source(&params, &result, &body);
    let reference = format!(
        "{}\nconst OBSERVED = sample([{ty}:0, {ty}:1]);",
        dslx_source(&params, &result, &body)
    );
    let width = if mode == 2 { value_width } else { index_width };
    let inferred = Type::Bits {
        signed: false,
        width,
    };
    check_narrow_native_result(
        &bx,
        &reference,
        &format!("expected {ty}, got {inferred}"),
        width,
    )
}

/// An unannotated constant callback yields the literal's own width even when
/// DSLX accepts a wider declared result. A return annotation provides context.
fn check_constant_result(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().skip(1).chain(std::iter::repeat(0));
    let annotated = bytes.next().unwrap() % 2 != 0;
    let value = bytes.next().unwrap();
    let width = (u8::BITS - value.leading_zeros()) as usize;
    let declared = usize::from(bytes.next().unwrap() % 9);
    let result = format!("uN[{declared}][2]");
    let callback = if annotated {
        format!("|_| -> uN[{declared}] {{ {value} }}")
    } else {
        format!("|_| {value}")
    };
    let body = format!("map(xs, {callback})");
    let params = "xs: u8[2]";
    if annotated || declared <= width {
        return compare_literal_program(data, params, result, body, declared >= width);
    }
    let bx = bx_source(params, &result, &body);
    let reference = format!(
        "{}\nconst OBSERVED = sample([u8:0, u8:1]);",
        dslx_source(params, &result, &body)
    );
    let inferred = Type::Bits {
        signed: false,
        width,
    };
    let expected = Type::Bits {
        signed: false,
        width: declared,
    };
    check_narrow_native_result(
        &bx,
        &reference,
        &format!("expected {expected}[2], got {inferred}[2]"),
        width,
    )
}

/// Compare executable literals, or normal width overflow errors, in both
/// frontends.
fn compare_literal_program(
    data: &[u8],
    params: &str,
    result: String,
    body: String,
    valid: bool,
) -> Result<(), String> {
    compare_program(
        FunctionProgram {
            functions: String::new(),
            params: params.into(),
            result,
            body,
            valid: Some(valid),
            dslx_ir: true,
            dynamic_index: None,
        },
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

/// Check the actual DSLX result ABI without sending inconsistent types to
/// the native IR builder. BX must diagnose the incompatible width.
fn check_narrow_native_result(
    bx: &str,
    reference: &str,
    diagnostic: &str,
    width: usize,
) -> Result<(), String> {
    let module = parse_module(bx).map_err(|error| format!("{error}\n{bx}"))?;
    match check_module(&module) {
        Err(error) if error.contains(diagnostic) => { /* Expected width mismatch. */ }
        Err(error) => {
            return Err(format!(
                "Expected width error containing '{diagnostic}', got {error}\n{bx}"
            ));
        }
        Ok(_) => return Err(format!("BX accepted incompatible map element widths\n{bx}")),
    }
    let native = parse_and_typecheck(reference, "oracle.x", "oracle", &mut ImportData::default())
        .map_err(|e| format!("{e}\n{reference}"))?;
    let module = native.get_module();
    let Some(MatchableModuleMember::ConstantDef(observed)) = module
        .get_member(module.get_member_count() - 1)
        .to_matchable()
    else {
        return Err(format!("Missing native map result\n{reference}"));
    };
    let actual = native
        .get_type_info()
        .get_const_expr(&observed.get_value())
        .and_then(|v| v.convert_to_ir())
        .and_then(|v| v.get_elements())
        .map_err(|e| format!("{e}\n{reference}"))?;
    if actual.len() != 2
        || actual
            .iter()
            .any(|v| !v.to_bits().is_ok_and(|b| b.get_bit_count() == width))
    {
        return Err(format!(
            "Expected native element width {width}, got {actual:?}\n{reference}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_map_lambda;

    #[test]
    fn compare_generated_map_lambdas() -> Result<(), String> {
        for shape in 0..3 {
            for result in 0..3 {
                for flags in [0, 19, 37, 72, 127, 128, 255] {
                    check_map_lambda(&[
                        shape,
                        flags,
                        7 + flags % 103,
                        15 + flags % 94,
                        result,
                        result,
                        0,
                        flags,
                    ])?;
                }
                for mutation in [8, 9, 10, 11, 15] {
                    check_map_lambda(&[shape, 127, 7, 15, 3, result, mutation])?;
                }
                for mutation in 12..=14 {
                    for flags in [0, 127, 128, 255] {
                        check_map_lambda(&[shape, flags, 71, 95, 3, result, mutation])?;
                    }
                }
            }
        }
        for shape in 3..5 {
            for mode in [0, 1, 2, 4, 6] {
                for width in [0, 3, 6] {
                    check_map_lambda(&[shape, mode, width, 31, 19, 55])?;
                }
            }
        }
        for annotated in 0..2 {
            for value in [0, 1, 3, 7, 255] {
                for declared in [0, 1, 3, 8] {
                    check_map_lambda(&[5, annotated, value, declared])?;
                }
            }
        }
        for mode in 0..4 {
            for count in 0..4 {
                for value in [0, 1, 6, 255] {
                    for alias in 0..3 {
                        check_map_lambda(&[6, mode, count, value, 8 + 24 * count, alias])?;
                    }
                }
            }
        }
        Ok(())
    }
}
