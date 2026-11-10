// SPDX-License-Identifier: Apache-2.0

//! Compare array concatenation in BX and DSLX.

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use crate::input_seed;

/// Vary array shapes, the source of operand types, and which expression
/// context evaluates the concatenation.
pub fn check_array_concat(data: &[u8]) -> Result<(), String> {
    let [family, flags, width, left, right] =
        std::array::from_fn(|i| data.get(i).copied().unwrap_or(0));
    let width = usize::from(width % 97);
    let signed = flags & 1 != 0;
    let word = word_type(signed, width);
    let sizes = [usize::from(left % 5) + 1, usize::from(right % 5) + 1];
    let program = match family % 8 {
        0 | 1 => {
            let element = match (flags >> 1) & 3 {
                0 => word,
                1 => "bool".into(),
                2 => format!("({word}, bool)"),
                _ => format!("{word}[{}]", usize::from(flags >> 6) + 1),
            };
            runtime_arrays(&element, sizes, flags)
        }
        2 => nominal_arrays(&word, sizes, flags, false),
        3 => nominal_arrays(&word, sizes, flags, true),
        4 => empty_arrays(&word, sizes, flags, width != 0),
        5 => inferred_arrays(width % 7 + 2, sizes, flags),
        _ => invalid_arrays(width, sizes, flags),
    };
    compare_program(
        program,
        input_seed(data),
        DslxExecution::ConcreteConstants,
        Vec::new(),
    )
}

fn word_type(signed: bool, width: usize) -> String {
    let prefix = if signed { 's' } else { 'u' };
    format!("{prefix}N[{width}]")
}

/// Build the shared source for an accepted runtime expression.
fn runtime_program(params: String, result: String, body: String) -> FunctionProgram {
    FunctionProgram {
        functions: String::new(),
        params,
        result,
        body,
        valid: Some(true),
        dslx_ir: true,
        dynamic_index: None,
    }
}

/// Concatenate parameters directly or in a local function, including
/// three-operand expressions with both association orders.
fn runtime_arrays(element: &str, [left, right]: [usize; 2], flags: u8) -> FunctionProgram {
    let params = format!("left: {element}[{left}], right: {element}[{right}]");
    let (length, expr) = match (flags >> 3) & 3 {
        0 => (left + right, "left ++ right"),
        1 => (left + right, "right ++ left"),
        2 => (2 * left + right, "left ++ right ++ left"),
        _ => (2 * left + right, "left ++ (right ++ left)"),
    };
    let result = format!("{element}[{length}]");
    if flags & 32 != 0 {
        let functions = format!("fn join({params}) -> {result} {{ {expr} }}");
        FunctionProgram {
            functions,
            ..runtime_program(params, result, "join(left, right)".into())
        }
    } else {
        runtime_program(params, result, format!("let joined = {expr}; joined"))
    }
}

/// Vary nominal identity independently of the lowered layout.
fn nominal_arrays(
    word: &str,
    [left, right]: [usize; 2],
    flags: u8,
    is_enum: bool,
) -> FunctionProgram {
    let distinct = flags & 2 != 0;
    let right_type = if distinct { "Other" } else { "Element" };
    let input_type = if is_enum { "bool" } else { word };
    let params = format!("x: {input_type}, y: {input_type}");
    let declaration = |name| {
        if is_enum {
            format!("enum {name} : u2 {{ LOW = 1, HIGH = 2 }}")
        } else {
            format!("struct {name} {{ value: {word} }}")
        }
    };
    let value = |name, input| {
        if is_enum {
            format!("if {input} {{ {name}::LOW }} else {{ {name}::HIGH }}")
        } else {
            format!("{name} {{ value: {input} }}")
        }
    };
    let mut functions = declaration("Element");
    if distinct {
        functions.push('\n');
        functions.push_str(&declaration("Other"));
    }
    let a = array_literal(left, &value("Element", "x"));
    let b = array_literal(right, &value(right_type, "y"));
    let result = format!("Element[{}]", left + right);
    FunctionProgram {
        functions,
        valid: Some(!distinct),
        ..runtime_program(params, result, format!("{a} ++ {b}"))
    }
}

/// Use a nonempty observation for typed empty operands and empty intermediates.
fn empty_arrays(
    word: &str,
    [left, right]: [usize; 2],
    flags: u8,
    has_bits: bool,
) -> FunctionProgram {
    let array_type = format!("{word}[{right}]");
    let constant_prefix = if flags & 8 == 0 { 0 } else { left };
    let constant_items = (0..right)
        .map(|i| format!("{word}:0x{}", usize::from(has_bits) & ((i + 1) % 2)))
        .collect::<Vec<_>>()
        .join(", ");
    let expr = match (flags >> 1) & 3 {
        0 => "empty ++ xs",
        1 => "xs ++ empty",
        2 => "(empty ++ empty) ++ xs",
        _ => "xs ++ (empty ++ empty)",
    };
    FunctionProgram {
        functions: format!(
            "const JOINED = zero!<{word}[{constant_prefix}]>() ++ [{constant_items}];"
        ),
        // Native DSLX constant evaluation handles empty arrays, while its IR
        // converter cannot represent an empty array literal in a function.
        dslx_ir: false,
        ..runtime_program(
            format!("xs: {array_type}"),
            format!("({word}[{}], {array_type})", constant_prefix + right),
            format!("let empty: {word}[0] = []; (JOINED, {expr})"),
        )
    }
}

/// Exercise each operand's independent inference using bare numeric arrays.
fn inferred_arrays(width: usize, [left, right]: [usize; 2], flags: u8) -> FunctionProgram {
    let signed = flags & 1 != 0;
    let widen = flags & 2 != 0;
    let both_literals = !widen && flags & 4 != 0;
    let word = word_type(signed, width + usize::from(widen));
    let high_bit = 1 << (width - 1);
    let literal = if signed { -high_bit } else { high_bit }.to_string();
    let lhs = array_literal(left, &literal);
    let (binding, lhs) = if flags & 16 == 0 {
        (String::new(), lhs)
    } else {
        (format!("let items = {lhs}; "), "items".into())
    };
    let (params, rhs) = if both_literals {
        (String::new(), array_literal(right, &literal))
    } else {
        (format!("xs: {word}[{right}]"), "xs".into())
    };
    let expr = if flags & 8 != 0 {
        format!("{rhs} ++ {lhs}")
    } else {
        format!("{lhs} ++ {rhs}")
    };
    let result = format!("{word}[{}]", left + right);
    FunctionProgram {
        valid: Some(!widen),
        ..runtime_program(params, result, format!("{binding}{expr}"))
    }
}

/// Separate an element type mismatch from an outer length mismatch.
fn invalid_arrays(width: usize, [left, right]: [usize; 2], flags: u8) -> FunctionProgram {
    let word = word_type(false, width);
    let (a, b, result, expr) = match flags & 7 {
        0 => (
            word.clone(),
            word_type(false, width + 1),
            left + right,
            "a ++ b",
        ),
        1 => (word.clone(), word_type(true, width), left + right, "a ++ b"),
        2 => (
            format!("{word}[2]"),
            format!("{word}[3]"),
            left + right,
            "a ++ b",
        ),
        3 => (
            format!("({word}, bool)"),
            format!("({word}, u2)"),
            left + right,
            "a ++ b",
        ),
        4 => (word.clone(), word.clone(), left + right + 1, "a ++ b"),
        5 => (word.clone(), word.clone(), left, "a ++ []"),
        6 => (word.clone(), word.clone(), right, "[] ++ b"),
        _ => (word.clone(), word.clone(), left + right, "a ++ b[u32:0]"),
    };
    FunctionProgram {
        valid: Some(false),
        ..runtime_program(
            format!("a: {a}[{left}], b: {b}[{right}]"),
            format!("{a}[{result}]"),
            expr.into(),
        )
    }
}

fn array_literal(length: usize, element: &str) -> String {
    format!("[{}]", vec![element; length].join(", "))
}

#[cfg(test)]
mod tests {
    use super::check_array_concat;

    #[test]
    fn compare_array_concat_families() -> Result<(), String> {
        for (family, flags) in [
            (0, &[0, 1, 2, 4, 6, 15, 22, 31, 40, 86][..]),
            (2, &[0, 1, 2]),
            (3, &[0, 2]),
            (4, &[0, 3, 4, 7, 8]),
            (5, &[0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 17, 19]),
            (6, &[0, 1, 2, 3, 4, 5, 6, 7]),
        ] {
            for &flag in flags {
                check_array_concat(&[family, flag, 68, 3, 1])?;
            }
        }
        for family in [0, 2, 4, 6] {
            check_array_concat(&[family, 1, 0, 1, 3])?;
        }
        Ok(())
    }
}
