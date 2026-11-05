// SPDX-License-Identifier: Apache-2.0

//! Explicit and inferred builtin widths compared with scalar IR definitions.

use crate::converter_contracts::ContractCase;
use xlsynth_pir::{FnBuilder, IrValue, ir};

/// Varies a builtin's operand width, legal explicit parameter prefix, and,
/// for decode, its result width on both sides of the reachable index range.
pub fn generate(data: &[u8]) -> Result<ContractCase, String> {
    let byte = |index: usize| data.get(index).copied().unwrap_or_default();
    let widths = [1usize, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 65];
    let input_width = widths[usize::from(byte(1)) % widths.len()];
    let mut builder = FnBuilder::new("main");
    let x = builder
        .param("x", ir::Type::Bits(input_width))
        .map_err(|e| e.to_string())?;

    let spelling = byte(3) % 3;
    let mut parameters = Vec::new();
    let (builtin, family, output_width, result) = match byte(0) % 4 {
        0 => {
            // Use the actual domain when it fits the small-proof budget.
            // Wide operands still exercise shifts by more than a machine word.
            let domain = (input_width <= 7).then(|| 1usize << input_width);
            let boundary = domain.unwrap_or(input_width);
            let output_width = match byte(2) % 4 {
                0 => boundary - 1,
                1 => boundary,
                2 => boundary + 1,
                _ => 1 + usize::from(byte(4)) % 129,
            };
            // DSLX requires the result type even when it can infer the input.
            parameters.push(format!("uN[{output_width}]"));
            let family = match (spelling != 0, domain.is_some_and(|n| output_width > n)) {
                (false, false) => "decode_inferred",
                (true, false) => "decode_explicit_input_width",
                (false, true) => "decode_output_beyond_domain",
                (true, true) => "decode_explicit_beyond_domain",
            };
            // At the declared output width, shifting a one by x either sets
            // bit x or yields zero when x is past the last output bit.
            let one = builder
                .literal(IrValue::make_ubits(output_width, 1).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            ("decode", family, output_width, builder.shll(one, x))
        }
        control => {
            let bits = (0..input_width)
                .map(|index| builder.bit_slice(x, index, 1))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            let (builtin, width, result) = match control {
                // concat takes the most significant bit first.
                1 => ("rev", input_width, builder.concat(&bits)),
                2 => ("and_reduce", 1, builder.and_all(&bits)),
                _ => ("xor_reduce", 1, builder.xor_all(&bits)),
            };
            (builtin, "builtin_width_control", width, result)
        }
    };
    let declaration = match spelling {
        0 => String::new(),
        1 => {
            parameters.push(format!("u32:{input_width}"));
            String::new()
        }
        _ => {
            parameters.push("W".into());
            format!("const W = u32:{input_width};\n")
        }
    };
    let args = if parameters.is_empty() {
        String::new()
    } else {
        format!("<{}>", parameters.join(", "))
    };
    let source = format!(
        "{declaration}pub fn main(x: uN[{input_width}]) -> uN[{output_width}] {{ {builtin}{args}(x) }}\n"
    );
    let expected = builder
        .build_package(result.map_err(|e| e.to_string())?, "expected")
        .map_err(|e| e.to_string())?;
    Ok(ContractCase::single(family, source, expected.to_string()))
}

#[cfg(test)]
mod tests {
    use super::generate;
    use std::fmt::Write;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn legal_builtin_spellings_have_the_same_truth_tables() -> Result<(), Box<dyn std::error::Error>>
    {
        // Two input bits give a four-value domain. Decode to three, four and
        // five bits to check truncation, a full domain, and the spare high bit.
        let cases = [
            (0, 0, 3, [1, 2, 4, 0]),
            (0, 1, 4, [1, 2, 4, 8]),
            (0, 2, 5, [1, 2, 4, 8]),
            (1, 0, 2, [0, 2, 1, 3]),
            (2, 0, 1, [0, 0, 0, 1]),
            (3, 0, 1, [0, 1, 1, 0]),
        ];
        for (operation, width_selector, result_width, table) in cases {
            for spelling in 0..3 {
                let case = generate(&[operation, 1, width_selector, spelling])?;
                let mut source = case.modules[0].text.clone();
                let package = IrPackage::parse_ir(&case.expected_ir, None)?;
                let function = package.get_function("main")?;
                for (index, expected) in table.into_iter().enumerate() {
                    writeln!(
                        source,
                        "const_assert!(main(u2:{index}) == uN[{result_width}]:{expected});"
                    )?;
                    let arg = XlsIrValue::make_ubits(2, index as u64)?;
                    assert_eq!(
                        function.interpret(&[arg])?,
                        XlsIrValue::make_ubits(result_width, expected)?,
                    );
                }
                parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default())?;
            }
        }
        Ok(())
    }

    #[test]
    fn wide_decode_observes_high_input_and_output_bits() -> Result<(), Box<dyn std::error::Error>> {
        let case = generate(&[0, 11, 1, 0])?;
        let package = IrPackage::parse_ir(&case.expected_ir, None)?;
        let function = package.get_function("main")?;
        let high = XlsIrValue::parse_typed("bits[65]:0x1_0000_0000_0000_0000")?;
        assert_eq!(
            function.interpret(&[XlsIrValue::make_ubits(65, 64)?])?,
            high
        );
        assert_eq!(function.interpret(&[high])?, XlsIrValue::make_ubits(65, 0)?);
        Ok(())
    }
}
