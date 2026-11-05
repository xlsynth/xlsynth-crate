// SPDX-License-Identifier: Apache-2.0

//! Lexically bound values consumed by runtime expressions and specializations.

use crate::converter_contracts::ContractCase;
use xlsynth_pir::{BuilderError, FnBuilder, IrBits, IrFormatPreference, IrValue, ir};

/// Observes the chosen binding in a public output for every input word.
fn reference(value: IrBits) -> Result<String, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(value.get_bit_count()))?;
    let literal = builder.literal(IrValue::Bits(value))?;
    let result = builder.xor(x, literal)?;
    Ok(builder.build_package(result, "expected")?.to_string())
}

/// Draws independent, full-width values for both lexical binders and the
/// expression operand. Missing bytes contribute zero bits in short seeds.
fn input_value(data: &[u8], offset: usize, width: usize) -> IrBits {
    IrBits::from_lsb_fn(width, |bit| {
        let byte = data.get(offset + bit / 8).copied().unwrap_or_default();
        byte & (1 << (bit % 8)) != 0
    })
}

/// Generates a public observation of a local value inside a parametric helper.
/// The oracle uses the lexical definition independently of its rendered name.
pub fn generate(data: &[u8]) -> Result<ContractCase, String> {
    let option = |index: usize| usize::from(data.get(index).copied().unwrap_or_default());
    let widths: [usize; 9] = [4, 2, 3, 5, 9, 17, 33, 65, 97];
    let width = widths[option(2) % widths.len()];
    let ty = format!("uN[{width}]");
    let stride = width.div_ceil(8);
    let local = input_value(data, 6, width);
    let mut outer = input_value(data, 6 + stride, width);
    let operand = input_value(data, 6 + 2 * stride, width);
    let relation = option(1) % 3;
    if relation == 2 {
        outer = local.clone();
    } else if outer == local {
        // Mutations of the name alone preserve the distinguishable value pair.
        outer = outer.xor(&IrBits::from_lsb_fn(width, |bit| bit == 0));
    }
    let outer_name = ["N", "Tag", "Width", "V"][option(3) % 4];
    let local_name = if relation == 1 { "Local" } else { outer_name };
    let literal = |value: &IrBits| value.to_string_fmt(IrFormatPreference::Hex, false);
    let local_literal = literal(&local);
    let binding = match option(4) % 3 {
        0 => format!("let {local_name} = {ty}:{local_literal};"),
        1 => format!("let ({local_name}, _) = ({ty}:{local_literal}, true);"),
        _ => format!("const {local_name} = {ty}:{local_literal};"),
    };
    let operand_literal = literal(&operand);
    let (expression, value) = match option(5) % 4 {
        0 => (
            format!("{local_name} + {ty}:{operand_literal}"),
            local.add(&operand),
        ),
        1 => (
            format!("{local_name} - {ty}:{operand_literal}"),
            local.sub(&operand),
        ),
        2 => (
            format!("{local_name} ^ {ty}:{operand_literal}"),
            local.xor(&operand),
        ),
        _ => (local_name.to_owned(), local),
    };
    let (family, observed, expected) = match option(0) % 3 {
        0 => (
            "lexical_value_argument",
            format!("value<{{{expression}}}>()"),
            value,
        ),
        1 => {
            // The DSLX dimension ranges from 1 to 32 at every binding width.
            // Returning the mask in a fixed-width word exposes a wrong
            // dimension.
            let dimension = 1 + usize::from(value.to_le_bytes()[0] & 31);
            let mask = IrBits::from_lsb_fn(32, |bit| bit < dimension);
            let dimension_expr = format!("((({expression}) as u5) as u32) + u32:1");
            let observed = format!("(all_ones!<uN[{dimension_expr}]>() as u32)");
            ("lexical_type_dimension", observed, mask)
        }
        _ => ("lexical_runtime", format!("({expression})"), value),
    };
    let result_ty = format!("uN[{}]", expected.get_bit_count());
    let outer_literal = literal(&outer);
    let source = format!(
        r#"fn value<K: {ty}>() -> {ty} {{ K }}
fn outer<{outer_name}: {ty}>(x: {result_ty}) -> {result_ty} {{
    {binding}
    x ^ {observed}
}}
pub fn main(x: {result_ty}) -> {result_ty} {{ outer<{ty}:{outer_literal}>(x) }}
"#
    );
    Ok(ContractCase::single(
        family,
        source,
        reference(expected).map_err(|error| error.to_string())?,
    ))
}

#[cfg(test)]
mod tests {
    use super::generate;
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn oracle_uses_nearest_binding_and_wraps_at_declared_width()
    -> Result<(), Box<dyn std::error::Error>> {
        // u4:15 + u4:2 wraps to one. A dimension of 1 + 1 makes mask 3.
        for (consumer, width, expected) in [(0, 4, 1), (1, 32, 3), (2, 4, 1)] {
            for relation in 0..3 {
                for binding in 0..3 {
                    let case = generate(&[consumer, relation, 0, 0, binding, 0, 15, 9, 2])?;
                    let package = IrPackage::parse_ir(&case.expected_ir, None)?;
                    let function = package.get_function("main")?;
                    for x in 0..16 {
                        assert_eq!(
                            function.interpret(&[XlsIrValue::make_ubits(width, x)?])?,
                            XlsIrValue::make_ubits(width, x ^ expected)?,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn alpha_renaming_preserves_an_observable_bit_above_64()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut seed = vec![0; 6 + 3 * 9];
        seed[2] = 7; // 65-bit binding and public result.
        seed[6 + 8] = 1; // Bit 64 of the nearest binding.
        let shadowed = generate(&seed)?;
        seed[1] = 1;
        let renamed = generate(&seed)?;
        assert_ne!(shadowed.modules, renamed.modules);
        assert_eq!(shadowed.expected_ir, renamed.expected_ir);
        let package = IrPackage::parse_ir(&shadowed.expected_ir, None)?;
        let result = package
            .get_function("main")?
            .interpret(&[XlsIrValue::make_ubits(65, 0)?])?;
        assert!(result.to_bits()?.get_bit(64)?);
        Ok(())
    }
}
