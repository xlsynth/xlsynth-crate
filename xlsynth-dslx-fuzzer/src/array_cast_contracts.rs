// SPDX-License-Identifier: Apache-2.0

//! Zero-length bits-to-array and same-type array-to-array casts.

use crate::converter_contracts::{ArrayIdentityForm, ContractCase, array_identity};

const WIDTHS: [usize; 7] = [8, 1, 4, 7, 16, 32, 65];

/// Keep the cast result live through array equality, with a scalar oracle.
/// Materializing the zero-element array must work during optimization.
pub(crate) fn empty(data: &[u8]) -> ContractCase {
    let flags = data.first().copied().unwrap_or_default();
    let width = WIDTHS[usize::from(data.get(1).copied().unwrap_or_default()) % WIDTHS.len()];
    let length = usize::from(flags & 1 != 0);
    let sign = if flags & 2 == 0 { 'u' } else { 's' };
    let element = if width <= 64 {
        format!("{sign}{width}")
    } else {
        format!("{sign}N[{width}]")
    };
    let equal = flags & 4 == 0;
    let comparison = if equal { "==" } else { "!=" };
    let packed_width = width * length;
    let source = format!(
        r#"pub fn main() -> bool {{
    let a = uN[{packed_width}]:0 as {element}[{length}];
    a {comparison} a
}}
"#
    );
    let result = u8::from(equal);
    let expected_ir = format!(
        r#"package expected

top fn main() -> bits[1] {{
  ret result: bits[1] = literal(value={result}, id=1)
}}
"#
    );
    ContractCase::single("empty_array_cast", source, expected_ir)
}

/// Varies the array shape and whether its identity uses an explicit cast.
pub(crate) fn identity(data: &[u8]) -> ContractCase {
    let flags = data.first().copied().unwrap_or_default();
    let form = match flags % 4 {
        0 => ArrayIdentityForm::DirectCast,
        1 => ArrayIdentityForm::Uncast,
        2 => ArrayIdentityForm::AliasCast,
        _ => ArrayIdentityForm::Rebuild,
    };
    let width = WIDTHS[usize::from(data.get(1).copied().unwrap_or_default()) % WIDTHS.len()];
    let length = 2 + usize::from(data.get(2).copied().unwrap_or_default() % 4);
    let sign = if flags & 4 == 0 { 'u' } else { 's' };
    array_identity("array_identity_cast", form, width, length, sign)
}

#[cfg(test)]
mod tests {
    use super::{empty, identity};
    use crate::converter_contracts;
    use std::path::Path;
    use xlsynth::{IrPackage, XlsIrValue, convert_dslx_to_ir, optimize_ir};

    #[test]
    fn short_seed_keeps_the_empty_cast_live() {
        let case = empty(&[]);
        assert_eq!(
            case.modules[0].text,
            r#"pub fn main() -> bool {
    let a = uN[0]:0 as u8[0];
    a == a
}
"#
        );
    }

    #[test]
    fn nonempty_controls_convert_and_optimize_to_the_scalar_oracle()
    -> Result<(), Box<dyn std::error::Error>> {
        for flags in [1, 3, 5, 7] {
            let case = empty(&[flags, 6]);
            let reference = IrPackage::parse_ir(&case.expected_ir, None)?;
            let converted = convert_dslx_to_ir(
                &case.modules[0].text,
                Path::new("sample.x"),
                &Default::default(),
            )?;
            let optimized = optimize_ir(&converted.ir, "__sample__main")?;
            let expected = XlsIrValue::bool(flags & 4 == 0);
            assert_eq!(reference.get_function("main")?.interpret(&[])?, expected);
            assert_eq!(
                optimized.get_function("__sample__main")?.interpret(&[])?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn short_identity_seed_replays_the_original_array_cast() -> Result<(), String> {
        let case = identity(&[]);
        let legacy = converter_contracts::generate(&[3, 1, 7])?;
        assert_eq!(
            case.modules[0].text,
            "pub fn main(x: u8[2]) -> u8[2] { x as u8[2] }\n"
        );
        assert_eq!(case.modules[0].text, legacy.modules[0].text);
        assert_eq!(case.expected_ir, legacy.expected_ir);
        Ok(())
    }

    #[test]
    fn wide_array_identity_controls_preserve_all_element_bits()
    -> Result<(), Box<dyn std::error::Error>> {
        let input = XlsIrValue::parse_typed("[bits[65]:0x10000000000000001, bits[65]:7]")?;
        for flags in [1, 3, 5, 7] {
            let case = identity(&[flags, 6]);
            let reference = IrPackage::parse_ir(&case.expected_ir, None)?;
            let converted = convert_dslx_to_ir(
                &case.modules[0].text,
                Path::new("sample.x"),
                &Default::default(),
            )?;
            assert_eq!(
                reference
                    .get_function("main")?
                    .interpret(std::slice::from_ref(&input))?,
                input
            );
            assert_eq!(
                converted
                    .ir
                    .get_function("__sample__main")?
                    .interpret(std::slice::from_ref(&input))?,
                input
            );
        }
        Ok(())
    }
}
