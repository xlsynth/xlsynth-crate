// SPDX-License-Identifier: Apache-2.0

//! Focused contracts for derived aggregate packing and method references in
//! maps.

use crate::ast::{BinaryOp, BitsType, Width};
use crate::converter_contracts::ContractCase;
use xlsynth_pir::{BuilderError, FnBuilder, ir};

/// Splits a word into fields from most to least significant, then asks the
/// derived method to put it back together. Unit and zero-width fields consume
/// no input bits. The reference is identity for every grouping of the fields.
pub(crate) fn derive_to_bits(data: &[u8]) -> Result<ContractCase, String> {
    let option = |index: usize| usize::from(data.get(index).copied().unwrap_or_default());
    let shape = option(0) % 5;
    let width = 1 + option(1) % 16;
    let tuple_arity = 1 + option(2) % 3;
    let array_len = [2, 1, 3, 4][option(3) % 4];
    let leaves = if matches!(shape, 1 | 4) {
        1
    } else {
        tuple_arity * array_len
    };
    let packed_width = width * leaves;
    let ty = format!("uN[{width}]");
    let tuple_suffix = if tuple_arity == 1 { "," } else { "" };
    let pieces: Vec<_> = (0..leaves)
        .rev()
        .map(|i| format!("x[{}+: {ty}]", i * width))
        .collect();
    let (family, field_type, initializer) = match shape {
        0 => {
            let tuple = format!(
                "({}{tuple_suffix})",
                vec![ty.as_str(); tuple_arity].join(", ")
            );
            let rows = pieces
                .chunks(tuple_arity)
                .map(|row| format!("({}{tuple_suffix})", row.join(", ")))
                .collect::<Vec<_>>()
                .join(", ");
            (
                "derive_to_bits_tuple_array",
                format!("{tuple}[{array_len}]"),
                format!("[{rows}]"),
            )
        }
        2 => (
            "derive_to_bits_flat_array",
            format!("{ty}[{leaves}]"),
            format!("[{}]", pieces.join(", ")),
        ),
        3 => {
            let row_type = format!("{ty}[{array_len}]");
            let rows = pieces
                .chunks(array_len)
                .map(|row| format!("[{}]", row.join(", ")))
                .collect::<Vec<_>>()
                .join(", ");
            (
                "derive_to_bits_array_tuple",
                format!("({}{tuple_suffix})", vec![row_type; tuple_arity].join(", ")),
                format!("({rows}{tuple_suffix})"),
            )
        }
        // Insert a unit or zero-bit scalar on either side of the data field.
        _ => {
            let (family, empty_type, empty_value) = if shape == 1 {
                ("derive_to_bits_unit", "()", "()")
            } else {
                ("derive_to_bits_zero_width", "uN[0]", "uN[0]:0")
            };
            let piece = &pieces[0];
            if option(4) % 2 == 0 {
                (
                    family,
                    format!("({empty_type}, {ty})"),
                    format!("({empty_value}, {piece})"),
                )
            } else {
                (
                    family,
                    format!("({ty}, {empty_type})"),
                    format!("({piece}, {empty_value})"),
                )
            }
        }
    };
    let source = format!(
        r#"#[derive(ToBits)]
struct Packet {{ lanes: {field_type} }}
pub fn main(x: uN[{packed_width}]) -> uN[{packed_width}] {{
    Packet {{ lanes: {initializer} }}.to_bits()
}}
"#
    );
    let mut builder = FnBuilder::new("main");
    let x = builder
        .param("x", ir::Type::Bits(packed_width))
        .map_err(|e| e.to_string())?;
    let reference = builder
        .build_package(x, "expected")
        .map_err(|e| e.to_string())?;
    Ok(ContractCase::single(family, source, reference.to_string()))
}

/// Varies the method reference independently of the enclosing expression.
/// Since the operators commute, moving the map to either side or a separate
/// binding must preserve both its acceptance and its value.
pub(crate) fn method_map(data: &[u8]) -> Result<ContractCase, String> {
    let option = |index: usize| usize::from(data.get(index).copied().unwrap_or_default());
    let position = option(0) % 3;
    let method = option(1) % 3;
    let width = 1 + option(2) % 16;
    let length = 1 + option(3) % 4;
    let index = option(4) % length;
    let op = [BinaryOp::And, BinaryOp::Or, BinaryOp::Xor][option(5) % 3];
    let bits = BitsType {
        width: Width::Concrete(width),
        signed: false,
    };
    let ty = bits.to_dslx();
    let (definition, mapper) = match method {
        0 => (format!("fn read(self) -> {ty} {{ self.v }}"), "Word::read"),
        1 => (
            format!("fn combine(self, element: {ty}) -> {ty} {{ self.v ^ element }}"),
            "Word { v: mask }.combine",
        ),
        _ => (
            format!("fn read(element: {ty}) -> {ty} {{ element }}"),
            "Word::read",
        ),
    };
    let elements = (0..length)
        .map(|i| {
            let value = if i % 2 == 0 { "x" } else { "!x" };
            if method == 0 {
                format!("Word {{ v: {value} }}")
            } else {
                value.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mapped = format!("map([{elements}], {mapper})[u32:{index}]");
    let body = match position {
        0 => op.to_dslx("mask", &mapped, bits),
        1 => op.to_dslx(&mapped, "mask", bits),
        _ => format!(
            "let selected = {mapped};\n    {}",
            op.to_dslx("mask", "selected", bits)
        ),
    };
    let source = format!(
        r#"struct Word {{ v: {ty} }}
impl Word {{ {definition} }}
pub fn main(x: {ty}, mask: {ty}) -> {ty} {{
    {body}
}}
"#
    );
    let reference = method_reference(width, index, method == 1, op).map_err(|e| e.to_string())?;
    Ok(ContractCase::single("method_map", source, reference))
}

/// Computes the selected element directly, without building a DSLX map,
/// array or method call in the reference.
fn method_reference(
    width: usize,
    index: usize,
    bound: bool,
    op: BinaryOp,
) -> Result<String, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(width))?;
    let mask = builder.param("mask", ir::Type::Bits(width))?;
    let element = if index % 2 == 0 { x } else { builder.not(x)? };
    let selected = if bound {
        builder.xor(mask, element)?
    } else {
        element
    };
    let result = op.emit_ir(&mut builder, mask, selected, false)?;
    Ok(builder.build_package(result, "expected")?.to_string())
}

#[cfg(test)]
mod tests {
    use super::{derive_to_bits, method_map};
    use std::path::Path;
    use xlsynth::{IrPackage, XlsIrValue, convert_dslx_to_ir_text, mangle_dslx_name};

    /// Evaluates the generated source of a passing control through libxls.
    fn interpret_control(
        source: &str,
        inputs: &[XlsIrValue],
    ) -> Result<XlsIrValue, Box<dyn std::error::Error>> {
        let converted =
            convert_dslx_to_ir_text(source, Path::new("sample.x"), &Default::default())?;
        let ir = IrPackage::parse_ir(&converted.ir, None)?;
        let top = ir.get_function(&mangle_dslx_name("sample", "main")?)?;
        Ok(top.interpret(inputs)?)
    }

    #[test]
    fn every_derived_layout_recovers_an_asymmetric_word() -> Result<(), Box<dyn std::error::Error>>
    {
        for shape in 0..5 {
            for length in 0..2 {
                for side in 0..2 {
                    let case = derive_to_bits(&[shape, 2, 1, length, side])?;
                    let width = match (shape, length) {
                        (1 | 4, _) => 3,
                        (_, 0) => 12,
                        _ => 6,
                    };
                    let word = XlsIrValue::make_ubits(width, 0x5a3 & ((1 << width) - 1))?;
                    let ir = IrPackage::parse_ir(&case.expected_ir, None)?;
                    let result = ir
                        .get_function("main")?
                        .interpret(std::slice::from_ref(&word))?;
                    assert_eq!(result, word);
                    // Multi-element tuple arrays and nested units are the
                    // crash candidates. Exercise those in the subprocess
                    // fuzzer.
                    if shape >= 2 || (shape == 0 && length == 1) {
                        assert_eq!(
                            interpret_control(&case.modules[0].text, std::slice::from_ref(&word))?,
                            word
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn every_method_and_placement_matches_the_word_operations()
    -> Result<(), Box<dyn std::error::Error>> {
        for method in 0..3 {
            for position in 0..3 {
                for op in 0..3 {
                    for index in 0..2 {
                        let case = method_map(&[position, method, 2, 1, index, op])?;
                        let ir = IrPackage::parse_ir(&case.expected_ir, None)?;
                        let (x, mask) = (0b110, 0b101);
                        let mut element = if index == 0 { x } else { x ^ 0b111 };
                        if method == 1 {
                            element ^= mask;
                        }
                        let expected =
                            [mask & element, mask | element, mask ^ element][usize::from(op)];
                        let args = [
                            XlsIrValue::make_ubits(3, x)?,
                            XlsIrValue::make_ubits(3, mask)?,
                        ];
                        let expected = XlsIrValue::make_ubits(3, expected)?;
                        assert_eq!(ir.get_function("main")?.interpret(&args)?, expected);
                        // The hoisted form supplies a working control for
                        // every mapped method, input position and operation.
                        if position == 2 {
                            assert_eq!(interpret_control(&case.modules[0].text, &args)?, expected);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
