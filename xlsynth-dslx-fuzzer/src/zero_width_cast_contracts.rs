// SPDX-License-Identifier: Apache-2.0

//! Same-width scalar casts preserve their bits, including the empty bit vector.

use crate::converter_contracts::{ContractCase, scalar_identity};

/// Casts a runtime input directly or through a width-parametric free function.
/// The independent identity oracle needs no sign bit, even at width zero.
pub(crate) fn generate(data: &[u8]) -> ContractCase {
    let flags = data.first().copied().unwrap_or_default();
    let widths = [0, 1, 2, 7, 8, 32, 64, 65];
    let width = widths[usize::from(data.get(1).copied().unwrap_or_default()) % widths.len()];
    let from = if flags & 1 == 0 { "sN" } else { "uN" };
    let to = if flags & 2 == 0 { "uN" } else { "sN" };
    let source = if flags & 4 == 0 {
        format!("pub fn main(x: {from}[{width}]) -> {to}[{width}] {{ x as {to}[{width}] }}\n")
    } else {
        format!(
            r#"fn recast<N: u32>(x: {from}[N]) -> {to}[N] {{ x as {to}[N] }}
pub fn main(x: {from}[{width}]) -> {to}[{width}] {{ recast<u32:{width}>(x) }}
"#
        )
    };
    ContractCase::single("same_width_scalar_cast", source, scalar_identity(width))
}

#[cfg(test)]
mod tests {
    use super::generate;
    use crate::converter_contracts::scalar_identity;

    #[test]
    fn signed_zero_cast_retains_the_runtime_operand() {
        let case = generate(&[]);
        assert_eq!(
            case.modules[0].text,
            "pub fn main(x: sN[0]) -> uN[0] { x as uN[0] }\n"
        );
        assert_eq!(case.expected_ir, scalar_identity(0));
    }

    #[test]
    fn parametric_unsigned_control_preserves_a_wide_interface() {
        let case = generate(&[7, 7]);
        assert_eq!(
            case.modules[0].text,
            r#"fn recast<N: u32>(x: uN[N]) -> sN[N] { x as sN[N] }
pub fn main(x: uN[65]) -> sN[65] { recast<u32:65>(x) }
"#
        );
        assert_eq!(case.expected_ir, scalar_identity(65));
    }
}
