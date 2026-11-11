// SPDX-License-Identifier: Apache-2.0

//! Struct splats whose base is a call returning a parametric struct.

use crate::converter_contracts::{ContractCase, scalar_identity};

/// Checks that an inline or let-bound call supplies the retained field and
/// that the splat supplies the replacement field.
pub(crate) fn generate(data: &[u8]) -> ContractCase {
    let flags = data.first().copied().unwrap_or_default();
    let widths = [8, 1, 2, 5, 7, 16, 31, 32];
    let width = widths[usize::from(data.get(1).copied().unwrap_or_default()) % widths.len()];
    let call = if flags & 2 == 0 {
        "make(x)".into()
    } else {
        format!("make<u32:{width}>(x)")
    };
    let splat_type = if flags & 4 == 0 {
        "Pair".into()
    } else {
        format!("Pair<u32:{width}>")
    };
    let (binding, base) = if flags & 1 == 0 {
        (String::new(), call.as_str())
    } else {
        (format!("let base = {call}; "), "base")
    };
    let splat = format!("{splat_type} {{ a: u8:1, ..{base} }}");
    let (result_type, result, expected_ir) = if flags & 8 == 0 {
        (
            format!("u{width}"),
            format!("{splat}.b"),
            scalar_identity(width),
        )
    } else {
        (
            format!("(u8, u{width})"),
            format!("let result = {splat}; (result.a, result.b)"),
            format!(
                r#"package expected

top fn main(x: bits[{width}] id=1) -> (bits[8], bits[{width}]) {{
  replacement: bits[8] = literal(value=1, id=2)
  ret result: (bits[8], bits[{width}]) = tuple(replacement, x, id=3)
}}
"#
            ),
        )
    };
    let source = format!(
        r#"struct Pair<N: u32> {{ a: u8, b: uN[N] }}
fn make<N: u32>(x: uN[N]) -> Pair<N> {{ Pair {{ a: u8:0, b: x }} }}
pub fn main(x: u{width}) -> {result_type} {{ {binding}{result} }}
"#
    );
    ContractCase::single("parametric_splat_base", source, expected_ir)
}

#[cfg(test)]
mod tests {
    use super::generate;
    use crate::converter_contracts::scalar_identity;

    #[test]
    fn inline_and_let_bound_splats_share_an_identity_oracle() {
        let inline = generate(&[]);
        let bound = generate(&[1]);
        assert_eq!(
            inline.modules[0].text,
            r#"struct Pair<N: u32> { a: u8, b: uN[N] }
fn make<N: u32>(x: uN[N]) -> Pair<N> { Pair { a: u8:0, b: x } }
pub fn main(x: u8) -> u8 { Pair { a: u8:1, ..make(x) }.b }
"#
        );
        assert_eq!(
            bound.modules[0].text,
            r#"struct Pair<N: u32> { a: u8, b: uN[N] }
fn make<N: u32>(x: uN[N]) -> Pair<N> { Pair { a: u8:0, b: x } }
pub fn main(x: u8) -> u8 { let base = make(x); Pair { a: u8:1, ..base }.b }
"#
        );
        assert_eq!(inline.expected_ir, scalar_identity(8));
        assert_eq!(inline.expected_ir, bound.expected_ir);
    }
}
