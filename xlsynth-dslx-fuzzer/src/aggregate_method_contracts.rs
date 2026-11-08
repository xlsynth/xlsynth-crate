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

/// Kind of method passed to DSLX `map`, including how a derived method is
/// first looked up.
#[derive(Clone, Copy)]
enum MappedMethod {
    Read,
    BoundCombine,
    StaticRead,
    DerivedToBits { prior_call: bool },
    ManualToBits,
}

/// Varies the method reference independently of the enclosing expression.
/// Since the operators commute, moving the map to either side or a separate
/// binding must preserve both its acceptance and its value.
pub(crate) fn method_map(data: &[u8]) -> Result<ContractCase, String> {
    let method = match data.get(1).copied().unwrap_or_default() % 3 {
        0 => MappedMethod::Read,
        1 => MappedMethod::BoundCombine,
        _ => MappedMethod::StaticRead,
    };
    method_map_case(data.first().copied().unwrap_or_default(), method, data)
}

/// Compares a qualified reference to a derived method with a preceding dot
/// call or an explicit method implementation in the same map expression.
pub(crate) fn method_map_derived(data: &[u8]) -> Result<ContractCase, String> {
    let method = match data.first().copied().unwrap_or_default() % 3 {
        0 => MappedMethod::DerivedToBits { prior_call: false },
        1 => MappedMethod::DerivedToBits { prior_call: true },
        _ => MappedMethod::ManualToBits,
    };
    method_map_case(data.get(1).copied().unwrap_or_default(), method, data)
}

/// Builds the selected DSLX method call and its independent scalar IR.
fn method_map_case(
    position: u8,
    method: MappedMethod,
    data: &[u8],
) -> Result<ContractCase, String> {
    let option = |index: usize| usize::from(data.get(index).copied().unwrap_or_default());
    let width = 1 + option(2) % 16;
    let length = 1 + option(3) % 4;
    let index = option(4) % length;
    let op = [BinaryOp::And, BinaryOp::Or, BinaryOp::Xor][option(5) % 3];
    let bits = BitsType {
        width: Width::Concrete(width),
        signed: false,
    };
    let ty = bits.to_dslx();
    let word = format!("struct Word {{ v: {ty} }}");
    let (declaration, mapper, prior_call, family) = match method {
        MappedMethod::DerivedToBits { prior_call } => {
            let (prior_call, family) = if prior_call {
                (
                    "let _ = Word { v: x }.to_bits();\n    ",
                    "method_map_derived_after_call",
                )
            } else {
                ("", "method_map_derived")
            };
            (
                format!("#[derive(ToBits)]\n{word}"),
                "Word::to_bits",
                prior_call,
                family,
            )
        }
        MappedMethod::ManualToBits => (
            format!("{word}\nimpl Word {{ fn to_bits(self) -> {ty} {{ self.v }} }}"),
            "Word::to_bits",
            "",
            "method_map_manual_to_bits",
        ),
        MappedMethod::Read => (
            format!("{word}\nimpl Word {{ fn read(self) -> {ty} {{ self.v }} }}"),
            "Word::read",
            "",
            "method_map",
        ),
        MappedMethod::BoundCombine => (
            format!(
                "{word}\nimpl Word {{ fn combine(self, element: {ty}) -> {ty} {{ self.v ^ element }} }}"
            ),
            "Word { v: mask }.combine",
            "",
            "method_map",
        ),
        MappedMethod::StaticRead => (
            format!("{word}\nimpl Word {{ fn read(element: {ty}) -> {ty} {{ element }} }}"),
            "Word::read",
            "",
            "method_map",
        ),
    };
    let elements = (0..length)
        .map(|i| {
            let value = if i % 2 == 0 { "x" } else { "!x" };
            if matches!(
                method,
                MappedMethod::BoundCombine | MappedMethod::StaticRead
            ) {
                value.to_owned()
            } else {
                format!("Word {{ v: {value} }}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let mapped = format!("map([{elements}], {mapper})[u32:{index}]");
    let body = match position % 3 {
        0 => op.to_dslx("mask", &mapped, bits),
        1 => op.to_dslx(&mapped, "mask", bits),
        _ => format!(
            "let selected = {mapped};\n    {}",
            op.to_dslx("mask", "selected", bits)
        ),
    };
    let source = format!(
        r#"{declaration}
pub fn main(x: {ty}, mask: {ty}) -> {ty} {{
    {prior_call}{body}
}}
"#
    );
    let reference = method_reference(
        width,
        index,
        matches!(method, MappedMethod::BoundCombine),
        op,
    )
    .map_err(|e| e.to_string())?;
    Ok(ContractCase::single(family, source, reference))
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
    use super::{derive_to_bits, method_map, method_map_derived};
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

    #[test]
    fn derived_and_handwritten_mappers_share_the_same_reference()
    -> Result<(), Box<dyn std::error::Error>> {
        // The second element is !x. Hoist the map so expression position
        // cannot obscure whether the derived method itself resolves.
        let handwritten = method_map_derived(&[2, 2, 2, 1, 1, 2])?;
        let cold = method_map_derived(&[0, 2, 2, 1, 1, 2])?;
        let after_call = method_map_derived(&[1, 2, 2, 1, 1, 2])?;
        let baseline = method_map(&[2, 0, 2, 1, 1, 2])?;
        assert_eq!(
            handwritten.modules[0].text,
            baseline.modules[0].text.replace("read", "to_bits")
        );
        assert_eq!(
            after_call.modules[0].text,
            cold.modules[0].text.replace(
                "    let selected",
                "    let _ = Word { v: x }.to_bits();\n    let selected",
            )
        );
        assert_eq!(cold.expected_ir, handwritten.expected_ir);
        assert_eq!(after_call.expected_ir, handwritten.expected_ir);
        let args = [XlsIrValue::make_ubits(3, 6)?, XlsIrValue::make_ubits(3, 5)?];
        let expected = XlsIrValue::make_ubits(3, 4)?;
        let reference = IrPackage::parse_ir(&handwritten.expected_ir, None)?;
        assert_eq!(reference.get_function("main")?.interpret(&args)?, expected);
        for control in [handwritten, after_call] {
            assert_eq!(
                interpret_control(&control.modules[0].text, &args)?,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn original_method_map_ignores_trailing_options() -> Result<(), String> {
        for method in 0..3 {
            let bytes = [2, method, 2, 1, 1, 2];
            let original = method_map(&bytes)?;
            for tag in [0, 1, 2, b'd', b'D', b'M', 255] {
                let extended = method_map(&[bytes.as_slice(), &[tag, 0xff]].concat())?;
                assert_eq!(extended.family, original.family);
                assert_eq!(extended.modules, original.modules);
                assert_eq!(extended.expected_ir, original.expected_ir);
            }
        }
        Ok(())
    }
}
