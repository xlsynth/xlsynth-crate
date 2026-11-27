// SPDX-License-Identifier: Apache-2.0

//! Compare nested match coverage and selected results with pinned DSLX.

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use Shape::{Leaf, Tuple};
use xlsynth::dslx::{ImportData, parse_and_typecheck};
use xlsynth_bx::parse_module;
use xlsynth_pir::{IrBits, IrFormatPreference, IrValue};

/// Positions of the independently generated leaves in a tuple subject.
enum Shape {
    Leaf(usize),
    Tuple(Vec<Shape>),
}

impl Shape {
    fn product(selector: u8) -> Self {
        match selector % 6 {
            0 => Tuple(vec![Leaf(0), Leaf(1), Leaf(2)]),
            1 => Tuple(vec![Tuple(vec![Leaf(0), Leaf(2)]), Leaf(1)]),
            2 => Tuple(vec![Leaf(2), Tuple(vec![Leaf(0), Leaf(1)])]),
            3 => Tuple(vec![
                Tuple(vec![]),
                Tuple(vec![Leaf(1), Tuple(vec![Leaf(2)])]),
                Leaf(0),
            ]),
            4 => Tuple(vec![Tuple(vec![Leaf(2), Leaf(1), Leaf(0)])]),
            _ => Tuple(vec![Leaf(1), Leaf(0), Tuple(vec![Leaf(2)])]),
        }
    }

    /// Render types and patterns with the same tuple shape.
    fn render(&self, leaves: &[&str]) -> String {
        match self {
            Self::Leaf(index) => leaves[*index].to_owned(),
            Self::Tuple(members) => {
                let members_text = members
                    .iter()
                    .map(|member| member.render(leaves))
                    .collect::<Vec<_>>()
                    .join(", ");
                let comma = if members.len() == 1 { "," } else { "" };
                format!("({members_text}{comma})")
            }
        }
    }

    /// Construct a concrete IR value using the same field positions.
    fn value(&self, leaves: &[&IrValue]) -> IrValue {
        match self {
            Self::Leaf(index) => leaves[*index].clone(),
            Self::Tuple(members) => IrValue::make_tuple(
                &members
                    .iter()
                    .map(|member| member.value(leaves))
                    .collect::<Vec<_>>(),
            ),
        }
    }
}

/// Generate arm bodies that read projected scalar or tuple fields and cases
/// that reject bindings outside their single-pattern arm.
fn binding_arms(
    mode: u8,
    layout: u8,
    flags: u8,
    shape: &Shape,
    numeric_ty: &str,
    pivot_pattern: &str,
) -> String {
    let aggregate = flags & 16 != 0;
    let pattern = |first, second, numeric| shape.render(&[first, second, numeric]);
    let arm = |pattern: String, value: &str, flag: &str| {
        if aggregate {
            format!("{pattern} => ({value}, {flag})")
        } else {
            format!("{pattern} => {value}")
        }
    };
    let arms = match mode {
        11 => vec![
            arm(
                pattern("false", "carried", pivot_pattern),
                &format!("carried as {numeric_ty}"),
                "carried",
            ),
            arm(
                pattern("carried", "false", "word"),
                &format!("word + (carried as {numeric_ty})"),
                "carried",
            ),
            arm(
                pattern("_", "tag", "carried"),
                &format!("carried ^ (tag as {numeric_ty})"),
                "tag",
            ),
        ],
        12 => {
            let pair_pattern = |tag: &str| {
                if layout & 1 == 0 {
                    format!("({tag}, pair)")
                } else {
                    format!("(pair, {tag})")
                }
            };
            vec![
                arm(pair_pattern("false"), "pair.1", "pair.0"),
                arm(
                    pair_pattern("_"),
                    &format!("pair.1 + (pair.0 as {numeric_ty})"),
                    "!pair.0",
                ),
            ]
        }
        13 => {
            let first = if flags & 4 != 0 { "word" } else { "_" };
            let alternatives =
                [pattern("false", "_", first), pattern("true", "_", "word")].join(" | ");
            let result = if flags & 8 == 0 {
                "word"
            } else {
                pivot_pattern
            };
            vec![arm(alternatives, result, "true")]
        }
        _ => {
            vec![
                arm(pattern("false", "_", "word"), "word", "false"),
                arm("_".into(), "word", "true"),
            ]
        }
    };
    arms.join(", ")
}

/// A generated tuple subject, its match arms and the expected typecheck
/// outcome.
struct TupleCase {
    shape: Shape,
    arms: String,
    result: String,
    valid: bool,
}

/// Generate rest patterns, whole-subject binders and patterns that compare
/// two independent numeric fields with the same constant.
fn rest_and_binding_case(mode: u8, layout: u8, flags: u8, numeric_ty: &str) -> TupleCase {
    let gap = |first: Shape, last: Shape| {
        let mut members = vec![first];
        members.extend((0..(layout / 8) % 4).map(|i| Leaf(usize::from(i % 2))));
        members.push(last);
        Tuple(members)
    };
    let mut result = numeric_ty.to_owned();
    let (shape, arms) = match mode {
        15 => match layout % 5 {
            0 => (
                gap(Leaf(0), Leaf(2)),
                "(false, .., tail) => tail, (.., tail) => !tail".into(),
            ),
            1 => (
                gap(Leaf(2), Leaf(1)),
                "(head, .., true) => head, (head, ..) => !head".into(),
            ),
            2 => (
                gap(gap(Leaf(2), Leaf(3)), Leaf(1)),
                "((.., tail), .., false) => tail, ((head, ..), ..) => head".into(),
            ),
            3 => {
                let shape = if flags & 4 == 0 {
                    Tuple(vec![])
                } else {
                    Shape::product(layout)
                };
                let pattern = if flags & 8 == 0 { "(..)" } else { "(..,)" };
                (shape, format!("{pattern} => {numeric_ty}:1"))
            }
            _ => (
                gap(Tuple(vec![]), Leaf(2)),
                "((..), .., last) => last".into(),
            ),
        },
        16 => {
            let shape = Shape::product(layout);
            result = shape.render(&["bool", "bool", numeric_ty]);
            let special = shape.render(&["false", "_", "_"]);
            let value = shape.render(&["true", "true", &format!("{numeric_ty}:0")]);
            let arms = if flags & 4 == 0 {
                format!("{special} => {value}, whole => whole")
            } else {
                "whole => whole".into()
            };
            (shape, arms)
        }
        17 => {
            let (shape, pattern) = match layout % 4 {
                0 => (Tuple(vec![Leaf(2), Leaf(3)]), "(PIVOT, PIVOT)"),
                1 => (
                    Tuple(vec![Leaf(2), gap(Leaf(1), Leaf(3))]),
                    "(PIVOT, (.., PIVOT))",
                ),
                2 => (
                    Tuple(vec![Tuple(vec![Leaf(0), Leaf(2)]), Leaf(3)]),
                    "((_, PIVOT), PIVOT)",
                ),
                _ => (gap(Leaf(2), Leaf(3)), "(PIVOT, .., PIVOT)"),
            };
            result = "bool".into();
            (shape, format!("{pattern} => true, _ => false"))
        }
        _ => {
            let (shape, pattern) = match layout % 6 {
                0 => (Shape::product(layout), "(.., ..)"),
                1 => (Tuple(vec![Leaf(2)]), "(_, .., _)"),
                2 => (Tuple(vec![Leaf(0), Leaf(2)]), "((..), _)"),
                3 => (Tuple(vec![]), "(.., _)"),
                4 => (gap(Leaf(2), Leaf(3)), "(item, ..) | (.., item)"),
                _ => (Shape::product(layout), ".."),
            };
            let arms = format!("{pattern} => {numeric_ty}:0, _ => {numeric_ty}:1");
            (shape, arms)
        }
    };
    TupleCase {
        shape,
        arms,
        result,
        valid: mode != 18,
    }
}

/// A fresh name in both constant positions is a binding error in BX. The
/// pinned DSLX frontend accepts the same source as an equality test.
fn check_repeated_binder(ty: &str, constant_arms: &str) -> Result<(), String> {
    let arms = constant_arms.replace("PIVOT", "item");
    let source = format!("fn compare(x: {ty}) -> bool {{ match x {{ {arms} }} }}");
    parse_and_typecheck(&source, "oracle.x", "oracle", &mut ImportData::default())
        .map_err(|error| format!("DSLX rejected repeated name: {error}\n{source}"))?;
    match parse_module(&source) {
        Err(error) if error.contains("duplicate match binding 'item'") => Ok(()),
        Err(error) => Err(format!("unexpected BX diagnostic: {error}\n{source}")),
        Ok(_) => Err(format!("BX accepted a duplicate match binding\n{source}")),
    }
}

/// Generate tuple patterns and sample the cross product of the Boolean
/// leaves and the numeric range boundaries, including overlapping arms.
pub fn check_tuple_match(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let mode = bytes.next().unwrap() % 19;
    let layout = bytes.next().unwrap();
    let width_selector = bytes.next().unwrap();
    let flags = bytes.next().unwrap();
    let width = if mode == 9 {
        0
    } else {
        [2, 3, 5, 8, 31, 32, 63, 64, 65, 96, 129][usize::from(width_selector) % 11]
    };
    let signed = width != 0 && flags & 1 != 0;
    let (min, max) = if signed {
        (
            IrBits::signed_min_value(width),
            IrBits::signed_max_value(width),
        )
    } else {
        (IrBits::zero(width), IrBits::all_ones(width))
    };
    let one = IrBits::from_lsb_fn(width, |i| i == 0);
    let entropy = data.get(4..).unwrap_or_default();
    let candidate = IrBits::from_lsb_fn(width, |i| {
        !entropy.is_empty() && entropy[(i / 8) % entropy.len()] & (1 << (i % 8)) != 0
    });
    let pivot = if candidate == min || candidate == max {
        min.add(&one)
    } else {
        candidate
    };
    let numeric_ty = format!("{}N[{width}]", if signed { 's' } else { 'u' });
    let literal = |value: &IrBits| {
        format!(
            "{numeric_ty}:{}",
            value.to_string_fmt(IrFormatPreference::Hex, false)
        )
    };
    let min_text = literal(&min);
    let pivot_text = literal(&pivot);
    let max_text = literal(&max);
    let mut functions = String::new();
    let pivot_pattern: &str = if flags & 2 == 0 && mode != 17 {
        &pivot_text
    } else {
        functions.push_str(&format!("const PIVOT = {pivot_text};\n"));
        "PIVOT"
    };
    let mut valid = Some(true);
    let (shape, binding_case) = if mode >= 15 {
        let case = rest_and_binding_case(mode, layout, flags, &numeric_ty);
        valid = Some(case.valid);
        (case.shape, Some((case.result, case.arms)))
    } else {
        let shape = match mode {
            7 | 9 => Shape::Tuple(vec![Shape::Leaf(2)]),
            8 if layout & 1 == 0 => Shape::Tuple(vec![]),
            8 => Shape::Tuple(vec![Shape::Tuple(vec![])]),
            10 => Shape::Tuple(vec![Shape::Leaf(0), Shape::Leaf(1)]),
            12 => {
                let pair = Tuple(vec![Leaf(1), Leaf(2)]);
                if layout & 1 == 0 {
                    Tuple(vec![Leaf(0), pair])
                } else {
                    Tuple(vec![pair, Leaf(0)])
                }
            }
            _ => Shape::product(layout),
        };
        (shape, None)
    };
    let pattern = |first, second, numeric| shape.render(&[first, second, numeric]);
    let low = format!("{min_text}..{pivot_text}");
    let through_pivot = format!("{min_text}..={pivot_text}");
    let high = format!("{pivot_text}..={max_text}");
    let aggregate = flags & 16 != 0;
    let (result, arms) = if let Some(arms) = binding_case {
        arms
    } else if mode >= 11 {
        valid = Some(matches!(mode, 11 | 12));
        let result = if aggregate {
            format!("({numeric_ty}, bool)")
        } else {
            numeric_ty.clone()
        };
        let arms = binding_arms(mode, layout, flags, &shape, &numeric_ty, pivot_pattern);
        (result, arms)
    } else {
        let mut arms = match mode {
            0 => {
                valid = None;
                let mut patterns = Vec::new();
                for _ in 0..(1 + bytes.next().unwrap() % 4) {
                    let mut alternatives = Vec::new();
                    for _ in 0..(1 + bytes.next().unwrap() % 3) {
                        let mut boolean =
                            || ["false", "true", "_"][usize::from(bytes.next().unwrap()) % 3];
                        let first = boolean();
                        let second = boolean();
                        let numeric = ["_", &min_text, pivot_pattern, &max_text, &low, &high]
                            [usize::from(bytes.next().unwrap()) % 6];
                        alternatives.push(pattern(first, second, numeric));
                    }
                    patterns.push(alternatives.join(" | "));
                }
                if flags & 4 == 0 {
                    patterns.push("_".into());
                }
                patterns
            }
            1 => {
                let mut overlapping = vec![
                    pattern("false", "_", &through_pivot),
                    pattern("_", "false", &high),
                ];
                if flags & 4 != 0 {
                    overlapping.reverse();
                }
                overlapping.push("_".into());
                overlapping
            }
            2 | 3 => {
                let mut partition = vec![
                    [pattern("false", "_", "_"), pattern("true", "false", "_")].join(" | "),
                    pattern("true", "true", &low),
                    pattern("true", "true", &high),
                ];
                if mode == 3 {
                    partition.remove(usize::from(layout) % partition.len());
                    valid = Some(false);
                }
                partition
            }
            4 => {
                valid = Some(false);
                if flags & 4 == 0 {
                    vec![
                        pattern("false", "_", "_"),
                        pattern("false", "true", pivot_pattern),
                        "_".into(),
                    ]
                } else {
                    vec![
                        pattern("false", "true", pivot_pattern),
                        "_".into(),
                        pattern("false", "true", &format!("{pivot_text}..={pivot_text}")),
                    ]
                }
            }
            5 => {
                valid = Some(false);
                vec!["()".into(), "_".into()]
            }
            6 => {
                valid = Some(false);
                let wrong = format!("{}N[{width}]:0", if signed { 'u' } else { 's' });
                vec![pattern("false", "_", &wrong), "_".into()]
            }
            7 => {
                let pattern = |numeric: &str| {
                    if flags & 4 == 0 {
                        format!("({numeric})")
                    } else {
                        format!("({numeric},)")
                    }
                };
                vec![
                    pattern(&low),
                    pattern(pivot_pattern),
                    pattern(&format!("{}..={max_text}", literal(&pivot.add(&one)))),
                ]
            }
            8 => vec![pattern("_", "_", "_")],
            9 => {
                let mut single_value = vec![pattern("_", "_", pivot_pattern)];
                if flags & 4 != 0 {
                    single_value.push(pattern("_", "_", &through_pivot));
                    valid = Some(false);
                }
                single_value
            }
            _ => vec![
                pattern("false", "_", "_"),
                pattern("_", "false", "_"),
                pattern("true", "true", "_"),
            ],
        };
        if flags & 8 != 0 && matches!(mode, 2 | 7) {
            arms.reverse();
        }
        let result = if aggregate { "(u8, bool)" } else { "u8" };
        let arms = arms
            .iter()
            .enumerate()
            .map(|(i, arm)| {
                if aggregate {
                    format!("{arm} => (u8:{i}, {})", i % 2 == 0)
                } else {
                    format!("{arm} => u8:{i}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        (result.to_owned(), arms)
    };
    let ty = shape.render(&["bool", "bool", &numeric_ty, &numeric_ty]);
    if mode == 17 {
        check_repeated_binder(&ty, &arms)?;
    }
    let params = format!("x: {ty}");
    let subject = if flags & 32 == 0 {
        "x"
    } else {
        functions.push_str(&format!("fn key({params}) -> {ty} {{ x }}\n"));
        "key(x)"
    };
    let mut body = format!("match {subject} {{ {arms} }}");
    if mode == 15 && flags & 32 != 0 {
        // Pinned DSLX can infer a zero-field tuple for a direct call whose
        // match arms all use rest markers. Preserve the call's concrete type.
        body = format!("let key_value: {ty} = {subject}; match key_value {{ {arms} }}");
    }
    let check_constant = flags & 128 != 0 && matches!(mode, 11 | 12 | 15..=17);
    if flags & 64 != 0 || check_constant {
        functions.push_str(&format!("fn classify({params}) -> {result} {{ {body} }}\n"));
        body = "classify(x)".into();
        if check_constant {
            let first = if layout & 1 == 0 { "false" } else { "true" };
            let second = if flags & 4 == 0 { "false" } else { "true" };
            let numeric = if flags & 8 == 0 {
                &pivot_text
            } else {
                &min_text
            };
            let value = shape.render(&[first, second, &pivot_text, numeric]);
            functions.push_str(&format!("const SELECTED = classify({value});\n"));
        }
    }

    let mut inputs = Vec::new();
    for numeric in [min, pivot.sub(&one), pivot.add(&one), pivot, max] {
        let neighbor = IrValue::from_bits(&numeric.add(&one));
        let word = IrValue::from_bits(&numeric);
        // At PIVOT and PIVOT-1 this exercises a match and a mismatch in
        // either of the two positions tested against the constant.
        let peers: &[&IrValue] = if mode == 17 {
            &[&word, &neighbor]
        } else {
            &[&neighbor]
        };
        for first in [false, true] {
            let first = IrValue::make_ubits(1, u64::from(first)).map_err(|e| e.to_string())?;
            for second in [false, true] {
                let second =
                    IrValue::make_ubits(1, u64::from(second)).map_err(|e| e.to_string())?;
                for peer in peers {
                    let args = vec![shape.value(&[&first, &second, &word, peer])];
                    if !inputs.contains(&args) {
                        inputs.push(args);
                    }
                }
            }
        }
    }
    compare_program(
        FunctionProgram {
            functions,
            params,
            result,
            body,
            valid,
            dslx_ir: true,
            dynamic_index: None,
        },
        crate::input_seed(data),
        DslxExecution::ConcreteConstants,
        inputs,
    )
}

#[cfg(test)]
mod tests {
    use super::check_tuple_match;

    #[test]
    fn compare_generated_tuple_match_cases() -> Result<(), String> {
        for (mode, layout, width, flags, entropy) in [
            (0, 4, 8, 1, 0),
            (1, 0, 0, 0, 2),
            (1, 1, 9, 117, 0),
            (1, 2, 8, 20, 0x55),
            (2, 3, 2, 32, 11),
            (2, 4, 10, 121, 0x80),
            (3, 2, 1, 1, 0x7e),
            (4, 5, 1, 2, 3),
            (4, 5, 10, 7, 0xfe),
            (5, 2, 3, 1, 0),
            (6, 3, 4, 0, 0),
            (7, 0, 0, 9, 0),
            (7, 0, 9, 102, 0x7f),
            (8, 0, 0, 32, 0),
            (8, 1, 0, 80, 0),
            (9, 0, 0, 0, 0),
            (9, 0, 0, 4, 0),
            (10, 0, 0, 96, 0),
            (11, 1, 8, 115, 0xa5),
            (11, 5, 10, 2, 0xff),
            (11, 3, 10, 150, 0xc2),
            (12, 0, 2, 0, 0x7f),
            (12, 1, 10, 113, 0x80),
            (12, 0, 8, 129, 0x21),
            (13, 1, 2, 0, 3),
            (13, 2, 2, 116, 3),
            (13, 4, 5, 8, 9),
            (14, 3, 4, 16, 0),
            (15, 0, 3, 128, 0x45),
            (15, 16, 8, 65, 0x81),
            (15, 27, 6, 41, 0x55),
            (15, 3, 1, 8, 5),
            (15, 3, 1, 4, 5),
            (15, 4, 10, 0, 0x40),
            (16, 3, 9, 128, 0x9a),
            (16, 5, 1, 4, 5),
            (17, 0, 8, 145, 0x85),
            (17, 1, 6, 56, 0xff),
            (17, 2, 10, 192, 0x66),
            (17, 3, 2, 1, 0x81),
            (17, 4, 8, 144, 0x55),
            (17, 5, 10, 185, 0x80),
            (18, 0, 2, 0, 0),
            (18, 1, 2, 0, 0),
            (18, 2, 2, 0, 0),
            (18, 3, 2, 0, 0),
            (18, 4, 2, 0, 0),
            (18, 5, 2, 0, 0),
        ] {
            check_tuple_match(&[mode, layout, width, flags, entropy])?;
        }
        Ok(())
    }
}
