// SPDX-License-Identifier: Apache-2.0

//! A free function passed to `map` can be specialized on a constant expression.

use crate::converter_contracts::ContractCase;
use xlsynth_pir::{BuilderError, FnBuilder, IrValue, ir};

/// Applies the independently folded parameter to every input without a map or
/// a call to a specialized function.
fn reference(width: usize, length: usize, amount: u64) -> Result<String, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let element_type = ir::Type::Bits(width);
    let xs = builder.param("xs", ir::Type::new_array(element_type.clone(), length))?;
    let amount = IrValue::make_ubits(width, amount)
        .map_err(|error| BuilderError::InvalidOperation(error.to_string()))?;
    let amount = builder.literal(amount)?;
    let mut results = Vec::with_capacity(length);
    for index in 0..length {
        let index = builder.literal(IrValue::u32(index as u32))?;
        let element = builder.array_index(xs, index)?;
        results.push(builder.add(element, amount)?);
    }
    let result = builder.array(element_type, &results)?;
    Ok(builder.build_package(result, "expected")?.to_string())
}

/// Varies the same specialization between an inline expression, a let binding
/// and a folded literal. The parameter and array element share a concrete
/// width.
pub(crate) fn generate(data: &[u8]) -> Result<ContractCase, String> {
    let byte = |index: usize, default| data.get(index).copied().unwrap_or(default);
    let widths = [8, 2, 4, 7, 16];
    let width = widths[usize::from(byte(2, 0)) % widths.len()];
    let length = 1 + usize::from(byte(3, 1) % 4);
    let mask = (1u64 << width) - 1;
    let left = u64::from(byte(4, 1)) & mask;
    let right = u64::from(byte(5, 1)) & mask;
    let (operator, amount) = match byte(1, 0) % 3 {
        0 => ("+", (left + right) & mask),
        1 => ("-", left.wrapping_sub(right) & mask),
        _ => ("^", left ^ right),
    };
    let expression = format!("u{width}:{left} {operator} u{width}:{right}");
    let (family, binding, argument) = match byte(0, 0) % 3 {
        0 => (
            "map_parametric_expression",
            String::new(),
            format!("{{{expression}}}"),
        ),
        1 => (
            "map_parametric_let",
            format!("let amount = {expression};\n    "),
            "amount".into(),
        ),
        _ => (
            "map_parametric_literal",
            String::new(),
            format!("u{width}:{amount}"),
        ),
    };
    let source = format!(
        r#"fn bias<K: u{width}>(x: u{width}) -> u{width} {{ x + K }}
pub fn main(xs: u{width}[{length}]) -> u{width}[{length}] {{
    {binding}map(xs, bias<{argument}>)
}}
"#
    );
    Ok(ContractCase::single(
        family,
        source,
        reference(width, length, amount).map_err(|e| e.to_string())?,
    ))
}

#[cfg(test)]
mod tests {
    use super::generate;
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn inline_sum_and_bound_control_keep_the_same_map_and_oracle() -> Result<(), String> {
        let inline = generate(&[])?;
        let bound = generate(&[1])?;
        assert_eq!(
            inline.modules[0].text,
            r#"fn bias<K: u8>(x: u8) -> u8 { x + K }
pub fn main(xs: u8[2]) -> u8[2] {
    map(xs, bias<{u8:1 + u8:1}>)
}
"#
        );
        assert_eq!(
            bound.modules[0].text,
            r#"fn bias<K: u8>(x: u8) -> u8 { x + K }
pub fn main(xs: u8[2]) -> u8[2] {
    let amount = u8:1 + u8:1;
    map(xs, bias<amount>)
}
"#
        );
        assert_eq!(inline.expected_ir, bound.expected_ir);
        assert_eq!(inline.expected_ir, generate(&[2])?.expected_ir);
        Ok(())
    }

    #[test]
    fn reference_applies_modular_parameters_to_each_element()
    -> Result<(), Box<dyn std::error::Error>> {
        let inputs = XlsIrValue::parse_typed("[bits[2]:3, bits[2]:1]")?;
        // Two-bit parameters wrap in both directions: 2 + 3 = 1, 2 - 3 = 3.
        for (operator, left, right, expected) in [
            (0, 2, 3, "[bits[2]:0, bits[2]:2]"),
            (1, 2, 3, "[bits[2]:2, bits[2]:0]"),
            (2, 1, 1, "[bits[2]:3, bits[2]:1]"),
        ] {
            let case = generate(&[0, operator, 1, 1, left, right])?;
            let package = IrPackage::parse_ir(&case.expected_ir, None)?;
            let output = package
                .get_function("main")?
                .interpret(std::slice::from_ref(&inputs))?;
            assert_eq!(output, XlsIrValue::parse_typed(expected)?);
        }
        Ok(())
    }
}
