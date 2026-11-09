// SPDX-License-Identifier: Apache-2.0

//! Reordering literals in an array or in disjoint match arms must preserve
//! the signed values observed at corresponding positions.

use crate::converter_contracts::ContractCase;

/// Picks two distinct permutations of the same literal set. The first two
/// ranks for a three-element set keep the first literal in place.
pub(crate) fn generate(data: &[u8]) -> ContractCase {
    let byte = |i: usize| data.get(i).copied().unwrap_or_default();
    let shape = byte(0);
    let count = 3 + usize::from((shape >> 1) % 3);
    let literals = literal_spellings(data, count);
    let order_count: usize = (2..=count).product();
    let first = usize::from(byte(8)) % order_count;
    let second = (first + 1 + usize::from(byte(9)) % (order_count - 1)) % order_count;
    let left = permutation(count, first);
    let right = permutation(count, second);
    let (family, source, parameter) = if shape & 1 == 0 {
        (
            "literal_order_array",
            array_comparison(&literals, &left, &right),
            "",
        )
    } else {
        // Both matches keep the same wildcard last. All preceding arms have
        // distinct, typed patterns, so reordering cannot change priority.
        let fallback = if byte(1) % 4 == 3 { "s17:0" } else { "0" };
        let arms = |order: &[usize]| {
            order
                .iter()
                .map(|&i| format!("u8:{i} => {}", literals[i]))
                .chain(std::iter::once(format!("_ => {fallback}")))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let source = format!(
            r#"pub fn main(x: u8) -> bool {{
    let a = match x {{ {} }};
    let b = match x {{ {} }};
    (a as s32) == (b as s32)
}}
"#,
            arms(&left),
            arms(&right),
        );
        ("literal_order_match", source, "x: bits[8] id=1")
    };
    // Permutation invariance does not require choosing a canonical inferred
    // type. Comparing after widening makes a changed sign bit observable.
    let expected_ir = format!(
        r#"package expected

top fn main({parameter}) -> bits[1] {{
  ret result: bits[1] = literal(value=1, id=2)
}}
"#
    );
    ContractCase::single(family, source, expected_ir)
}

/// Varies radix, relative widths and ties across three to five elements.
fn literal_spellings(data: &[u8], count: usize) -> Vec<String> {
    let byte = |i: usize| data.get(i).copied().unwrap_or_default();
    let radix = byte(1) % 4;
    let width = 2 + u32::from(data.get(2).copied().unwrap_or(6) % 15);
    let sign_bit = 1 << (width - 1);
    let offset = u16::from_le_bytes([byte(3), byte(7)]);
    let largest = (2 * sign_bit - 1) - u32::from(offset) % sign_bit;
    let mut literals = vec![match radix {
        0 | 3 => format!("0x{largest:x}"),
        1 => format!("0b{largest:b}"),
        _ => largest.to_string(),
    }];
    let small = u32::from(byte(5)) % sign_bit;
    let tied = sign_bit + small;
    literals.push(match byte(4) % 5 {
        0 => small.to_string(),
        1 => format!("0x{small:x}"),
        2 => tied.to_string(),
        3 => format!("0x{tied:x}"),
        _ => format!("0b{tied:b}"),
    });
    literals.push(format!("-{}", 1 + u32::from(byte(6))));
    for i in 3..count {
        literals.push(i8::from_le_bytes([byte(i + 7)]).to_string());
    }
    if radix == 3 {
        // All generated values fit in s17 even when a positive decimal has
        // the same natural width as the largest prefixed literal.
        for literal in &mut literals {
            *literal = format!("s17:{literal}");
        }
    }
    literals
}

/// Decodes a lexicographic permutation rank for a small literal set.
fn permutation(count: usize, mut rank: usize) -> Vec<usize> {
    let mut remaining: Vec<usize> = (0..count).collect();
    (0..count)
        .rev()
        .map(|suffix_len| {
            let stride: usize = (1..=suffix_len).product();
            let position = rank / stride;
            rank %= stride;
            remaining.remove(position)
        })
        .collect()
}

/// Compares every literal in the left array with its relocated counterpart.
fn array_comparison(literals: &[String], left: &[usize], right: &[usize]) -> String {
    let array = |order: &[usize]| {
        order
            .iter()
            .map(|&i| literals[i].as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut right_index = vec![0; right.len()];
    for (i, &literal) in right.iter().enumerate() {
        right_index[literal] = i;
    }
    let comparisons = left
        .iter()
        .enumerate()
        .map(|(i, &literal)| {
            format!(
                "((a[u32:{i}] as s32) == (b[u32:{}] as s32))",
                right_index[literal]
            )
        })
        .collect::<Vec<_>>()
        .join(" && ");
    format!(
        r#"pub fn main() -> bool {{
    let a = [{}];
    let b = [{}];
    {comparisons}
}}
"#,
        array(left),
        array(right),
    )
}

#[cfg(test)]
mod tests {
    use super::{generate, literal_spellings, permutation};
    use std::collections::BTreeSet;

    #[test]
    fn array_oracle_remaps_every_observation() {
        let case = generate(&[]);
        assert_eq!(case.family, "literal_order_array");
        assert_eq!(
            case.modules[0].text,
            r#"pub fn main() -> bool {
    let a = [0xff, 0, -1];
    let b = [0xff, -1, 0];
    ((a[u32:0] as s32) == (b[u32:0] as s32)) && ((a[u32:1] as s32) == (b[u32:2] as s32)) && ((a[u32:2] as s32) == (b[u32:1] as s32))
}
"#
        );
        assert_eq!(
            case.expected_ir,
            "package expected\n\ntop fn main() -> bits[1] {\n  ret result: bits[1] = literal(value=1, id=2)\n}\n"
        );
    }

    #[test]
    fn match_oracle_preserves_keys_and_final_wildcard() {
        let case = generate(&[1]);
        assert_eq!(case.family, "literal_order_match");
        assert_eq!(
            case.modules[0].text,
            r#"pub fn main(x: u8) -> bool {
    let a = match x { u8:0 => 0xff, u8:1 => 0, u8:2 => -1, _ => 0 };
    let b = match x { u8:0 => 0xff, u8:2 => -1, u8:1 => 0, _ => 0 };
    (a as s32) == (b as s32)
}
"#
        );
        assert_eq!(
            case.expected_ir,
            "package expected\n\ntop fn main(x: bits[8] id=1) -> bits[1] {\n  ret result: bits[1] = literal(value=1, id=2)\n}\n"
        );
    }

    #[test]
    fn width_ties_and_radix_controls_preserve_the_original_literal_spellings() {
        for (radix, first, second, negative, extra) in [
            (0, "0xff", "128", "-1", "-2"),
            (1, "0b11111111", "128", "-1", "-2"),
            (2, "255", "128", "-1", "-2"),
            (3, "s17:0xff", "s17:128", "s17:-1", "s17:-2"),
        ] {
            let data = [2, radix, 6, 0, 2, 0, 0, 0, 0, 0, 254];
            assert_eq!(
                literal_spellings(&data, 4),
                [first, second, negative, extra]
            );
        }
        assert_eq!(
            literal_spellings(&[0, 0, 6, 0, 3], 3),
            ["0xff", "0x80", "-1"]
        );
    }

    #[test]
    fn every_order_of_three_to_five_elements_is_reachable() {
        for count in 3..=5 {
            let ranks: usize = (2..=count).product();
            let orders: BTreeSet<_> = (0..ranks).map(|rank| permutation(count, rank)).collect();
            assert_eq!(orders.len(), ranks);
            for mut order in orders {
                order.sort_unstable();
                assert_eq!(order, (0..count).collect::<Vec<_>>());
            }
        }
    }
}
