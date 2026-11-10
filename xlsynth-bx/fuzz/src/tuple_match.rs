// SPDX-License-Identifier: Apache-2.0

//! Compare nested match coverage and selected results with pinned DSLX.

use crate::function_oracle::{DslxExecution, FunctionProgram, compare_program};
use Shape::{Leaf, Tuple};
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
    fn value(&self, leaves: &[IrValue]) -> IrValue {
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

/// Generate tuple patterns and sample the cross product of the Boolean
/// leaves and the numeric range boundaries, including overlapping arms.
pub fn check_tuple_match(data: &[u8]) -> Result<(), String> {
    let mut bytes = data.iter().copied().chain(std::iter::repeat(0));
    let mode = bytes.next().unwrap() % 11;
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
    let pivot_pattern: &str = if flags & 2 == 0 {
        &pivot_text
    } else {
        functions.push_str(&format!("const PIVOT = {pivot_text};\n"));
        "PIVOT"
    };
    let shape = match mode {
        7 | 9 => Shape::Tuple(vec![Shape::Leaf(2)]),
        8 if layout & 1 == 0 => Shape::Tuple(vec![]),
        8 => Shape::Tuple(vec![Shape::Tuple(vec![])]),
        10 => Shape::Tuple(vec![Shape::Leaf(0), Shape::Leaf(1)]),
        _ => Shape::product(layout),
    };
    let pattern = |first, second, numeric| shape.render(&[first, second, numeric]);
    let low = format!("{min_text}..{pivot_text}");
    let through_pivot = format!("{min_text}..={pivot_text}");
    let high = format!("{pivot_text}..={max_text}");
    let mut valid = Some(true);
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
    let aggregate = flags & 16 != 0;
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
    let ty = shape.render(&["bool", "bool", &numeric_ty]);
    let params = format!("x: {ty}");
    let subject = if flags & 32 == 0 {
        "x"
    } else {
        functions.push_str(&format!("fn key({params}) -> {ty} {{ x }}\n"));
        "key(x)"
    };
    let mut body = format!("match {subject} {{ {arms} }}");
    if flags & 64 != 0 {
        functions.push_str(&format!("fn classify({params}) -> {result} {{ {body} }}\n"));
        body = "classify(x)".into();
    }

    let mut inputs = Vec::new();
    for numeric in [min, pivot.sub(&one), pivot.add(&one), pivot, max] {
        for first in [false, true] {
            for second in [false, true] {
                let leaves = [
                    IrValue::make_ubits(1, u64::from(first)).map_err(|e| e.to_string())?,
                    IrValue::make_ubits(1, u64::from(second)).map_err(|e| e.to_string())?,
                    IrValue::from_bits(&numeric),
                ];
                let args = vec![shape.value(&leaves)];
                if !inputs.contains(&args) {
                    inputs.push(args);
                }
            }
        }
    }
    compare_program(
        FunctionProgram {
            functions,
            params,
            result: result.into(),
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
        ] {
            check_tuple_match(&[mode, layout, width, flags, entropy])?;
        }
        Ok(())
    }
}
