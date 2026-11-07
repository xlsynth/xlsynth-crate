// SPDX-License-Identifier: Apache-2.0

//! Generate local function signatures, argument expressions, and lexical
//! scopes.

use crate::function_oracle::{
    ArrayOperation, DslxExecution, DynamicIndex, FunctionProgram, compare_program,
};

/// Generate a helper and its caller while varying declaration order, binding
/// names, singleton type syntax, and the way arguments acquire their types.
pub fn check_function_syntax(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let family = bytes.next().unwrap() % 8;
    let flags = bytes.next().unwrap();
    let size_choice = bytes.next().unwrap();
    let width = if matches!(family, 0 | 2 | 3) && size_choice & 128 != 0 {
        0
    } else {
        [4usize, 8, 16, 32, 64, 96][usize::from(size_choice) % 6]
    };
    let signed = width != 0 && flags & 1 != 0;
    let prefix = if signed { "sN" } else { "uN" };
    let ty = format!("{prefix}[{width}]");
    let mut name =
        ["transform", "update", "stage", "block"][usize::from(bytes.next().unwrap()) % 4];
    let mode = bytes.next().unwrap();
    let scalar_param = format!("x: {ty}");
    let mut params = format!("{scalar_param}, flag: bool");
    let mut caller_args = "x, flag";
    let mut block_body = None;
    let mut callee_param = scalar_param.clone();
    let mut result_ty = ty.clone();
    let mut caller_result = ty.clone();
    let mut callee_body = String::from("x");
    let mut valid = true;
    let mut dynamic_index = None;
    let mut dslx_ir = true;
    let mut include_callee = true;
    let caller_body = match family {
        0 => {
            let mut argument = String::from("x");
            let mut projection = String::new();
            let mut block_input_type = ty.clone();
            let depth = 1 + mode % 3;
            for i in 0..depth {
                let comma = if flags & (2 << i) != 0 { "," } else { "" };
                result_ty = format!("({result_ty}{comma})");
                argument = format!("({argument},)");
                projection.push_str(".0");
                block_input_type = format!("({block_input_type},)");
            }
            callee_param = format!("x: {result_ty}");
            if flags & 16 != 0 {
                params = format!("x: {block_input_type}, flag: bool");
                argument = "x".into();
            }
            if flags & 32 != 0 {
                // Remove one tuple construction, retaining the callee's type.
                argument = format!("({argument}).0");
                valid = false;
            }
            format!("{name}({argument}){projection}")
        }
        1 => {
            // Vary the literal across the signed boundary and vary whether a
            // typed operand anchors its arithmetic or conditional group.
            let high = if flags & 2 == 0 { '7' } else { '8' };
            let literal = format!("0x{high}{}", "0".repeat(width / 4 - 1));
            let anchored = flags & 4 != 0;
            let other = if anchored { "x" } else { "1" };
            let op = ["+", "-", "&", "|", "^"][usize::from(bytes.next().unwrap()) % 5];
            let arg = match mode % 5 {
                0 => literal.clone(),
                1 => format!("{literal} {op} {other}"),
                2 => format!("{other} {op} {literal}"),
                3 => format!("if flag {{ {literal} }} else {{ {other} }}"),
                _ => format!("if flag {{ {other} }} else {{ {literal} }}"),
            };
            match (flags >> 3) % 3 {
                0 => {
                    valid = mode % 5 == 0 || !signed || anchored || high == '7';
                    format!("{name}({arg})")
                }
                1 => {
                    // Array elements share a type. An untyped high-bit
                    // pattern grouped with a positive integer requires an
                    // additional sign bit, whereas a typed element anchors
                    // the whole array at its signed width.
                    callee_param = format!("values: {ty}[2]");
                    callee_body = format!("values[0] {op} values[1]");
                    valid = !signed || anchored || high == '7';
                    if mode & 1 == 0 {
                        format!("{name}([{literal}, {other}])")
                    } else {
                        format!("{name}([{other}, {literal}])")
                    }
                }
                _ => {
                    callee_param = format!("parts: ({ty}, {ty})");
                    callee_body = format!("parts.0 {op} parts.1");
                    valid = mode % 5 == 0 || !signed || anchored || high == '7';
                    format!("{name}(({arg}, {other}))")
                }
            }
        }
        2 => {
            let pair = if mode & 1 == 0 { "(x, !x)" } else { "(!x, x)" };
            let (pattern, value) = if flags & 2 == 0 {
                ("(first, second)", pair.to_owned())
            } else {
                ("(first, (second, _))", format!("(x, {pair})"))
            };
            let op = ["+", "-", "^"][usize::from(mode >> 1) % 3];
            let scope = format!("let {pattern} = {value}; let x = second; {name}(first {op} x)");
            if flags & 4 == 0 {
                format!("{{ {scope} }}")
            } else {
                format!("if flag {{ {scope} }} else {{ let first = !x; {name}(first) }}")
            }
        }
        3 => {
            // Use the same helper spelling in a disjoint scope or in the scope
            // of the invocation. In the latter the callee resolves to a value.
            if flags & 2 == 0 {
                format!("let shadowed = {{ let {name} = !x; {name} }}; {name}(shadowed)")
            } else {
                valid = false;
                format!("let {name} = !x; {name}(x)")
            }
        }
        4 => {
            let size = 1 + usize::from(mode % 4);
            callee_param = format!("values: {ty}[{size}], index: u3");
            let (operation, call) = if flags & 2 != 0 {
                // The helper's body must call the DSLX builtin update.
                if name == "update" {
                    name = "replace_element";
                }
                callee_param.push_str(&format!(", replacement: {ty}"));
                callee_body = "update(values, index, replacement)".into();
                result_ty = format!("{ty}[{size}]");
                caller_result = result_ty.clone();
                params = callee_param.clone();
                caller_args = "values, index, replacement";
                (
                    ArrayOperation::Update { array_argument: 0 },
                    format!("{name}({caller_args})"),
                )
            } else {
                callee_body = "values[index]".into();
                let call = if flags & 4 != 0 {
                    params = callee_param.clone();
                    caller_args = "values, index";
                    format!("{name}({caller_args})")
                } else {
                    params = format!("{scalar_param}, index: u3");
                    caller_args = "x, index";
                    let elements = (0..size)
                        .map(|i| format!("x + {ty}:{i}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{name}([{elements}], index)")
                };
                (ArrayOperation::Read, call)
            };
            dynamic_index = Some(DynamicIndex {
                index_argument: 1,
                size,
                operation,
            });
            call
        }
        5 => {
            // Compare the builtin to a same-named local definition returning
            // a generated nonzero value. The call spelling stays identical.
            name = "zero!";
            include_callee = mode & 1 == 0;
            dslx_ir = !include_callee;
            callee_param.clear();
            callee_body = format!("{ty}:{}", 1 + (mode >> 1) % 7);
            if flags & 2 != 0 {
                valid = false;
                format!("let zero! = x; zero!<{ty}>()")
            } else {
                format!("zero!<{ty}>()")
            }
        }
        6 => {
            let space = if flags & 2 == 0 { " " } else { "" };
            match (mode >> 1) % 3 {
                0 if mode & 1 == 0 => {
                    // An adjacent bang belongs to the identifier. A separated
                    // bang can start an inequality.
                    valid = !space.is_empty();
                    format!("if x{space}!= !x {{ {name}(x) }} else {{ {name}(!x) }}")
                }
                0 => {
                    dslx_ir = false;
                    format!("let saved!{space}= !x; {name}(saved!)")
                }
                1 => {
                    let (yes, no, postfix) = if mode & 1 == 0 {
                        ("(x,)", "(!x,)", ".0")
                    } else {
                        ("[x]", "[!x]", "[0]")
                    };
                    let condition = format!("if flag {{ {yes} }} else {{ {no} }}");
                    valid = flags & 2 == 0;
                    let projected = if valid {
                        format!("({condition}){postfix}")
                    } else {
                        format!("{condition}{postfix}")
                    };
                    format!("{name}({projected})")
                }
                _ => {
                    valid = flags & 2 == 0;
                    let comment = if valid {
                        "// argument\n"
                    } else {
                        "/* argument */"
                    };
                    if mode & 1 == 0 {
                        format!("{name}({comment} x)")
                    } else {
                        format!("{comment} {name}(x)")
                    }
                }
            }
        }
        _ => {
            // Keep the literal's natural width equal to the parameter width.
            // The pinned native interpreter can return a narrower value for a
            // projected literal even when its caller declares a wider result.
            let high = char::from_digit(u32::from(8 + mode % 8), 16).unwrap();
            let literal = format!("0x{high}{}", "0".repeat(width / 4 - 1));
            let op = ["+", "-", "^"][usize::from(bytes.next().unwrap()) % 3];
            if flags & 32 != 0 {
                block_body = Some(format!(
                    "let raw = ({literal},); let alias = raw.0; route(alias, flag)"
                ));
            }
            if flags & 64 != 0 {
                callee_param = params.clone();
                let (left, right) = if flags & 2 == 0 {
                    ("raw", "x")
                } else {
                    ("x", "raw")
                };
                let expression = match (mode >> 3) % 4 {
                    0 | 1 => {
                        result_ty = "bool".into();
                        caller_result = result_ty.clone();
                        if mode & 8 == 0 {
                            format!("{left} == {right}")
                        } else {
                            let (a, b) = if flags & 4 == 0 {
                                (format!("({left},)"), format!("({right},)"))
                            } else {
                                (format!("[{left}]"), format!("[{right}]"))
                            };
                            let bindings = format!("let a = {a}; let b = {b};");
                            let (mut lhs, mut rhs) = (String::from("a"), String::from("b"));
                            let shape = bytes.next().unwrap();
                            for level in 0..shape % 4 {
                                if shape & (4 << level) == 0 {
                                    lhs = format!("({lhs},)");
                                    rhs = format!("({rhs},)");
                                } else {
                                    lhs = format!("[{lhs}]");
                                    rhs = format!("[{rhs}]");
                                }
                            }
                            format!("{bindings} {lhs} == {rhs}")
                        }
                    }
                    2 => {
                        format!("let selected = if flag {{ {left} }} else {{ {right} }}; selected")
                    }
                    _ => format!("[{left}, {right}][flag]"),
                };
                callee_body = format!("let raw = {literal}; {expression}");
                format!("{name}(x, flag)")
            } else {
                let argument = match (mode >> 3) % 8 {
                    0 => format!("({literal}, x).0"),
                    1 => format!("([{literal}], flag).0[0]"),
                    2 => {
                        valid = !signed;
                        if flags & 4 == 0 {
                            format!("[({literal} {op} 1,), (x,)][flag].0")
                        } else {
                            format!("[{{ let raw = {literal}; (raw {op} 1,) }}, (x,)][flag].0")
                        }
                    }
                    3 => format!("[(x,), ({literal},)][flag].0"),
                    4 => {
                        callee_body = match (flags >> 1) % 6 {
                            0 => format!("({{ let raw = {literal}; (raw, x) }}).0"),
                            1 => format!("let projected: {ty} = ({literal}, x).0; projected"),
                            2 => format!(
                                "let raw = {literal}; let alias = ({{ let nested = raw; (nested,) }}).0; alias"
                            ),
                            3 => format!("let (raw, _) = ({literal}, x); raw"),
                            4 => format!(
                                "let raw = {literal}; let alias = raw; let sum = raw {op} alias; sum"
                            ),
                            _ => {
                                valid = !signed;
                                format!("let raw: uN[{width}] = {literal}; raw")
                            }
                        };
                        "x".into()
                    }
                    5 => {
                        callee_param = format!("parts: ({ty},)");
                        callee_body = "parts.0".into();
                        format!("(({literal},), x).0")
                    }
                    6 => {
                        callee_param = format!("values: {ty}[1]");
                        callee_body = "values[0]".into();
                        format!("([{literal}], flag).0")
                    }
                    _ => {
                        result_ty = format!("({ty}[1], {ty})");
                        caller_result = result_ty.clone();
                        callee_body = format!("(([{literal}], x), x).0");
                        "x".into()
                    }
                };
                format!("{name}({argument})")
            }
        }
    };
    let callee = format!("fn {name}({callee_param}) -> {result_ty} {{ {callee_body} }}");
    let caller = format!("fn route({params}) -> {caller_result} {{ {caller_body} }}");
    let functions = if !include_callee {
        caller
    } else if family == 5 || flags & 128 == 0 {
        format!("{callee}\n{caller}")
    } else {
        // The native source parser requires callees before function callers.
        valid = false;
        format!("{caller}\n{callee}")
    };
    compare_program(
        FunctionProgram {
            functions,
            params,
            result: caller_result,
            body: block_body.unwrap_or_else(|| format!("route({caller_args})")),
            valid: Some(valid),
            dslx_ir,
            dynamic_index,
        },
        crate::input_seed(data),
        DslxExecution::BytecodeTests,
        Vec::new(),
    )
}
