// SPDX-License-Identifier: Apache-2.0

//! Tuple bindings whose pattern shape can mutate independently of the input
//! type.

use crate::SourceModule;

/// Varies the type of `x`, the outer pattern, and its two child patterns
/// independently. The constant result isolates pattern acceptance and lowering.
pub(crate) fn generate(data: &[u8]) -> Vec<SourceModule> {
    const INPUT_TYPES: [&str; 16] = [
        "(u8, u8)",
        "((u8,), u8)",
        "(u8, (u8,))",
        "u8",
        "()",
        "(u8,)",
        "(u8, u8, u8)",
        "u8[2]",
        "((u8, u8), u8)",
        "((u8,), (u8,))",
        "(u8, ())",
        "(u8[2], u8)",
        "(u8, (u8, u8))",
        "(((u8,), u8), u8)",
        "((), u8)",
        "(u8, u8[2])",
    ];
    let option = |index: usize| data.get(index).copied().unwrap_or_default();
    let input_type = INPUT_TYPES[usize::from(option(0)) % INPUT_TYPES.len()];
    let left = child_pattern(option(2), "a", "a2");
    let right = child_pattern(option(3), "b", "b2");
    let pattern = match option(1) % 8 {
        0 => format!("({left}, {right})"),
        1 => left,
        2 => format!("({left})"),
        3 => format!("({left},)"),
        4 => "()".to_string(),
        5 => format!("({left}, {right}, c)"),
        6 => format!("({left}, ..)"),
        _ => format!("(.., {right})"),
    };
    vec![SourceModule {
        name: "sample".into(),
        text: format!(
            r#"pub fn main(x: {input_type}) -> bool {{
    let {pattern} = x;
    true
}}
"#
        ),
    }]
}

/// Spells a child with zero, one, or two leaves. Wildcards can replace either
/// name.
fn child_pattern(option: u8, first: &str, second: &str) -> String {
    let first = if option & 8 != 0 { "_" } else { first };
    let second = if option & 16 != 0 { "_" } else { second };
    match option % 8 {
        0 => first.to_string(),
        1 => format!("({first})"),
        2 => format!("({first},)"),
        3 => format!("({first}, {second})"),
        4 => "()".to_string(),
        5 => format!("(.., {second})"),
        6 => format!("({first}, ..)"),
        _ => format!("(({first},), {second})"),
    }
}

#[cfg(test)]
mod tests {
    use super::generate;

    #[test]
    fn singleton_pattern_and_operand_mutate_independently() {
        // Pin the generated syntax without prescribing which mismatches the
        // typechecker must reject.
        for (options, expected) in [
            (
                [0, 0, 1],
                "pub fn main(x: (u8, u8)) -> bool {\n    let ((a), b) = x;\n    true\n}\n",
            ),
            (
                [0, 0, 2],
                "pub fn main(x: (u8, u8)) -> bool {\n    let ((a,), b) = x;\n    true\n}\n",
            ),
            (
                [0, 0, 0],
                "pub fn main(x: (u8, u8)) -> bool {\n    let (a, b) = x;\n    true\n}\n",
            ),
            (
                [1, 0, 1],
                "pub fn main(x: ((u8,), u8)) -> bool {\n    let ((a), b) = x;\n    true\n}\n",
            ),
            (
                [0, 5, 1],
                "pub fn main(x: (u8, u8)) -> bool {\n    let ((a), b, c) = x;\n    true\n}\n",
            ),
        ] {
            let modules = generate(&options);
            assert_eq!(modules.len(), 1);
            assert_eq!(modules[0].name, "sample");
            assert_eq!(modules[0].text, expected);
        }
    }
}
