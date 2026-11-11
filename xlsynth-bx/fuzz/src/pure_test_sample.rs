// SPDX-License-Identifier: Apache-2.0

//! Generate pure test declarations in the common BX and DSLX source language.

use std::fmt::Write;

/// One generated declaration and the first assertion it should fail, if any.
#[derive(Debug)]
pub(crate) struct ExpectedTest {
    pub name: &'static str,
    pub failure_line: Option<usize>,
}

/// A single source module and its expected results in declaration order.
#[derive(Debug)]
pub(crate) struct PureTestSample {
    pub source: String,
    pub expected: [ExpectedTest; 3],
}

impl PureTestSample {
    /// Generate bounded arithmetic, aggregate and control-flow checks, even
    /// for an empty input. Vary the final result to expose a runner that
    /// skips test bodies after an assertion failure.
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut bytes = data.iter().copied().cycle().chain(std::iter::repeat(0));
        let width = 1 + usize::from(bytes.next().unwrap() % 128);
        let flags = bytes.next().unwrap();
        let range = bytes.next().unwrap();
        let mask = u128::MAX >> (128 - width);
        let mut word = || {
            let mut value = 0u128;
            for byte in bytes.by_ref().take(width.div_ceil(8)) {
                value = (value << 8) | u128::from(byte);
            }
            value & mask
        };
        let a = word();
        let b = word();
        let signed = flags & 1 != 0;
        let prefix = if signed { "s" } else { "u" };
        // The explicit cast gives signed data the correct bytecode tag even
        // when the sampled bit pattern has its high bit set.
        let literal = |value: u128| format!("(uN[{width}]:0x{value:x} as Word)");
        let (operator, model): (&str, fn(u128, u128) -> u128) = match (flags >> 1) & 3 {
            0 => ("+", u128::wrapping_add),
            1 => ("-", u128::wrapping_sub),
            2 => ("^", |x, y| x ^ y),
            _ => ("*", u128::wrapping_mul),
        };
        let combine = |x, y| model(x, y) & mask;
        let mixed = combine(a, b);
        let (iterable, term, answer) = if flags & 8 == 0 {
            let start = u32::from(range & 3);
            let limit = start + u32::from((range >> 2) & 7);
            (
                format!("u32:{start}..u32:{limit}"),
                "(item as Word)",
                (start..limit).fold(mixed, |acc, i| combine(acc, u128::from(i) & mask)),
            )
        } else {
            (
                "items".to_owned(),
                "item",
                [a, b].into_iter().fold(mixed, combine),
            )
        };

        let mut source = format!(
            r#"type Word = {prefix}N[{width}];
fn combine(x: Word, y: Word) -> Word {{ x {operator} y }}
fn bundle(x: Word, y: Word) -> (Word, (Word, Word[2])) {{
    (combine(x, y), (x, [x, y]))
}}
"#
        );
        let a_lit = literal(a);
        let b_lit = literal(b);
        let take_then = answer & 1 != 0;
        let selected = usize::from(flags >> 4) % 3;
        let mut emit_test = |name: &'static str, failing: bool| {
            writeln!(
                source,
                r#"
#[test]
fn {name}() {{
    let (mixed, (value, items)) = bundle({a_lit}, {b_lit});
    let answer: Word = for (item, acc) in {iterable} {{ combine(acc, {term}) }}(mixed);"#
            )
            .unwrap();
            let first_line = source.lines().count() + 1;
            let expected_answer = literal(answer ^ u128::from(failing && selected == 0));
            let branch_failure = failing && selected == 1;
            // The unchosen branch always contains a failing assertion. The
            // chosen branch checks that its scoped binding shadows the outer
            // value, which the final assertion reads again after the branch.
            let expected_then = literal((a ^ mixed) ^ u128::from(!take_then || branch_failure));
            let expected_else = literal((b ^ mixed) ^ u128::from(take_then || branch_failure));
            let expected_outer = literal(a ^ u128::from(failing && selected == 2));
            let first =
                format!("    assert_eq((answer, items), ({expected_answer}, [{a_lit}, {b_lit}]));");
            let then_site = format!(r#"        assert!(value == {expected_then}, "then_value");"#);
            let else_site = format!("        assert_eq(value, {expected_else});");
            let last = format!(r#"    assert!(value == {expected_outer}, "outer_value");"#);
            let checks = [
                first.as_str(),
                "    if answer as bool {",
                "        let value = items[u32:0] ^ mixed;",
                then_site.as_str(),
                "    } else {",
                "        let value = items[u32:1] ^ mixed;",
                else_site.as_str(),
                "    };",
                last.as_str(),
            ];
            let selected_site = match selected {
                0 => &first,
                1 if take_then => &then_site,
                1 => &else_site,
                _ => &last,
            };
            let mut failure_line = None;
            for (offset, line) in checks.into_iter().enumerate() {
                if failing && line == selected_site {
                    failure_line = Some(first_line + offset);
                }
                writeln!(source, "{line}").unwrap();
            }
            if failing {
                source.push_str("    assert_eq(true, false);\n");
            }
            source.push_str("}\n");
            ExpectedTest { name, failure_line }
        };
        let first = emit_test("word_checks", false);
        let middle = emit_test("selected_failure", true);
        let fail_after = range & 0x80 != 0;
        writeln!(
            source,
            r#"
#[test]
fn after_failure() {{
    assert_eq(combine({b_lit}, {a_lit}), {});
}}"#,
            literal(combine(b, a) ^ u128::from(fail_after))
        )
        .unwrap();
        let after_line = source.lines().count() - 1;
        Self {
            source,
            expected: [
                first,
                middle,
                ExpectedTest {
                    name: "after_failure",
                    failure_line: fail_after.then_some(after_line),
                },
            ],
        }
    }
}
