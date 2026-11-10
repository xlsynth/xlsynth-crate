// SPDX-License-Identifier: Apache-2.0

//! Select and slice contracts that vary case and result counts through zero.

use crate::compound::metadata_literal;
use crate::converter_contracts::ContractCase;
use xlsynth_pir::{BuilderError, FnBuilder, IrBits, IrValue, ir};

const WIDTHS: [usize; 7] = [8, 1, 4, 16, 32, 64, 65];

/// Generates a one-hot select with matching runtime mask and case-array sizes.
pub(crate) fn one_hot_select(data: &[u8]) -> Result<ContractCase, String> {
    let (count, width, element) = result_shape(data);
    let source = format!(
        "pub fn main(mask: uN[{count}], cases: {element}[{count}]) -> {element} {{ one_hot_sel(mask, cases) }}\n"
    );
    let expected = one_hot_reference(count, width).map_err(|error| error.to_string())?;
    Ok(ContractCase::single(
        "one_hot_sel_case_count",
        source,
        expected.to_string(),
    ))
}

/// Generates an array slice whose runtime witness specifies only the result
/// length.
pub(crate) fn array_slice(data: &[u8]) -> Result<ContractCase, String> {
    let (length, width, element) = result_shape(data);
    let input_lengths = [2, 1, 3, 4, 5];
    let input_length =
        input_lengths[usize::from(data.get(2).copied().unwrap_or_default()) % input_lengths.len()];
    let source = format!(
        "pub fn main(xs: {element}[{input_length}], start: u32, witness: {element}[{length}]) -> {element}[{length}] {{ array_slice(xs, start, witness) }}\n"
    );
    let expected =
        slice_reference(input_length, length, width).map_err(|error| error.to_string())?;
    Ok(ContractCase::single(
        "array_slice_result_length",
        source,
        expected.to_string(),
    ))
}

/// Chooses an arity and a scalar element type common to both builtins.
fn result_shape(data: &[u8]) -> (usize, usize, String) {
    let count = usize::from(data.first().copied().unwrap_or_default() % 5);
    let width = WIDTHS[usize::from(data.get(1).copied().unwrap_or_default()) % WIDTHS.len()];
    let element = if width <= 64 {
        format!("u{width}")
    } else {
        format!("uN[{width}]")
    };
    (count, width, element)
}

/// ORs the enabled elements, starting from zero even when there are no cases.
fn one_hot_reference(count: usize, width: usize) -> Result<ir::Package, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let mask = builder.param("mask", ir::Type::Bits(count))?;
    let cases = builder.param("cases", ir::Type::new_array(ir::Type::Bits(width), count))?;
    let zero = builder.literal(IrValue::from_bits(&IrBits::zero(width)))?;
    let mut result = zero;
    for index in 0..count {
        let enabled = builder.bit_slice(mask, index, 1)?;
        let address = metadata_literal(&mut builder, index)?;
        let value = builder.array_index(cases, address)?;
        let masked = builder.select(enabled, &[zero, value], None)?;
        result = builder.or(result, masked)?;
    }
    builder.build_package(result, "expected")
}

/// Reads consecutive clamped indices independently of the slice builtin.
fn slice_reference(
    input_length: usize,
    length: usize,
    width: usize,
) -> Result<ir::Package, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let element = ir::Type::Bits(width);
    let xs = builder.param("xs", ir::Type::new_array(element.clone(), input_length))?;
    let start = builder.param("start", ir::Type::Bits(32))?;
    let witness = builder.param("witness", ir::Type::new_array(element.clone(), length))?;
    if length == 0 {
        // The witness already holds the only value of the empty result type.
        return builder.build_package(witness, "expected");
    }

    // Widen before adding offsets to preserve clamping for starts near
    // u32::MAX.
    let start = builder.zero_extend(start, 33)?;
    let mut elements = Vec::with_capacity(length);
    for index in 0..length {
        let offset = metadata_literal(&mut builder, index)?;
        let offset = builder.zero_extend(offset, 33)?;
        let address = builder.add(start, offset)?;
        elements.push(builder.array_index(xs, address)?);
    }
    let result = builder.array(element, &elements)?;
    builder.build_package(result, "expected")
}

#[cfg(test)]
mod tests {
    use super::{array_slice, one_hot_select};
    use std::path::Path;
    use xlsynth::dslx::{ImportData, parse_and_typecheck};
    use xlsynth::{IrPackage, XlsIrValue, convert_dslx_to_ir};

    #[test]
    fn empty_seeds_keep_zero_arity_in_the_typechecked_interface()
    -> Result<(), Box<dyn std::error::Error>> {
        let select = one_hot_select(&[])?;
        assert_eq!(
            select.modules[0].text,
            "pub fn main(mask: uN[0], cases: u8[0]) -> u8 { one_hot_sel(mask, cases) }\n"
        );
        let slice = array_slice(&[])?;
        assert_eq!(
            slice.modules[0].text,
            "pub fn main(xs: u8[2], start: u32, witness: u8[0]) -> u8[0] { array_slice(xs, start, witness) }\n"
        );
        for case in [select, slice] {
            parse_and_typecheck(
                &case.modules[0].text,
                "sample.x",
                "sample",
                &mut ImportData::default(),
            )?;
            IrPackage::parse_ir(&case.expected_ir, None)?;
        }
        Ok(())
    }

    #[test]
    fn one_hot_control_ors_multiple_enabled_cases() -> Result<(), Box<dyn std::error::Error>> {
        let case = one_hot_select(&[2])?;
        let expected = IrPackage::parse_ir(&case.expected_ir, None)?;
        let converted = convert_dslx_to_ir(
            &case.modules[0].text,
            Path::new("sample.x"),
            &Default::default(),
        )?;
        let expected = expected.get_function("main")?;
        let actual = converted.ir.get_function("__sample__main")?;
        let cases = XlsIrValue::parse_typed("[bits[8]:1, bits[8]:128]")?;
        for (mask, value) in [0, 1, 128, 129].into_iter().enumerate() {
            let args = [XlsIrValue::make_ubits(2, mask as u64)?, cases.clone()];
            let value = XlsIrValue::make_ubits(8, value)?;
            assert_eq!(expected.interpret(&args)?, value);
            assert_eq!(actual.interpret(&args)?, value);
        }
        Ok(())
    }

    #[test]
    fn array_slice_control_clamps_without_wrapping() -> Result<(), Box<dyn std::error::Error>> {
        let case = array_slice(&[3])?;
        let expected = IrPackage::parse_ir(&case.expected_ir, None)?;
        let converted = convert_dslx_to_ir(
            &case.modules[0].text,
            Path::new("sample.x"),
            &Default::default(),
        )?;
        let expected = expected.get_function("main")?;
        let actual = converted.ir.get_function("__sample__main")?;
        let xs = XlsIrValue::parse_typed("[bits[8]:3, bits[8]:7]")?;
        let witness = XlsIrValue::parse_typed("[bits[8]:11, bits[8]:13, bits[8]:19]")?;
        for (start, first) in [(0, 3), (1, 7), (u32::MAX, 7)] {
            let args = [
                xs.clone(),
                XlsIrValue::make_ubits(32, u64::from(start))?,
                witness.clone(),
            ];
            let value =
                XlsIrValue::parse_typed(&format!("[bits[8]:{first}, bits[8]:7, bits[8]:7]"))?;
            assert_eq!(expected.interpret(&args)?, value);
            assert_eq!(actual.interpret(&args)?, value);
        }
        Ok(())
    }
}
