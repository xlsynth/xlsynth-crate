// SPDX-License-Identifier: Apache-2.0

//! Checks the returned value and eager input evaluation of DSLX loops when
//! an induction variable or incoming carry is deliberately unused.

use crate::converter_contracts::ContractCase;
use xlsynth_pir::{BuilderError, FnBuilder, IrValue, ir};

/// Selects a loop spelling and binding independently of the property it must
/// satisfy. The ordinary counted loop and unused named bindings are controls.
pub(crate) fn generate(data: &[u8]) -> Result<ContractCase, String> {
    let byte = |i: usize| data.get(i).copied().unwrap_or_default();
    let observation = byte(0) % 5;
    let spelling = ["const for", "unroll_for!", "for"][usize::from(byte(1) % 3)];
    let binding = if byte(2) & 1 == 0 { "_" } else { "unused" };
    let length_option = usize::from(byte(3));
    let flags = byte(4);
    match observation {
        0 | 1 => {
            let length = (length_option + 1) % 5;
            returned_value(spelling, binding, length, flags, observation == 0)
        }
        2 | 3 => checked_operand(
            spelling,
            binding,
            length_option % 4 + 1,
            flags,
            observation == 2,
        ),
        _ => enum_iterable(spelling, byte(2) & 1 != 0, length_option % 4 + 1, flags),
    }
    .map_err(|e| e.to_string())
}

/// An ignored carry yields the final body's value, or the initializer on an
/// empty range. Every generated helper computes identity or negation, so its
/// results on complementary inputs must also differ.
fn returned_value(
    spelling: &str,
    binding: &str,
    length: usize,
    flags: u8,
    compare: bool,
) -> Result<ContractCase, BuilderError> {
    let invert = flags & 1 != 0;
    let constant_elements = flags & 2 != 0;
    let iterable = if length == 0 {
        "u1:0..u1:0".to_owned()
    } else {
        let elements = (0..length)
            .map(|i| match (constant_elements, (i % 2 != 0) ^ invert) {
                (false, false) => "x",
                (false, true) => "!x",
                (true, false) => "false",
                (true, true) => "true",
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{elements}]")
    };
    let body = if constant_elements {
        "element ^ x"
    } else {
        "element"
    };
    let init = if invert { "!x" } else { "x" };
    let loop_expr = format!("{spelling} (element, {binding}) in {iterable} {{ {body} }}({init})");
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(1))?;
    let (family, source, result) = if compare {
        (
            "unrolled_loop_result_equality",
            format!(
                "fn last(x: bool) -> bool {{ {loop_expr} }}\n\
                 pub fn main(x: bool) -> bool {{ last(x) == last(!x) }}\n"
            ),
            builder.literal(IrValue::bool(false))?,
        )
    } else {
        let last_inverts = invert ^ (length > 0 && length % 2 == 0);
        (
            "unrolled_loop_result",
            format!("pub fn main(x: bool) -> bool {{ {loop_expr} }}\n"),
            if last_inverts { builder.not(x)? } else { x },
        )
    };
    let reference = builder.build_package(result, "reference")?;
    Ok(ContractCase::single(family, source, reference.to_string()))
}

/// Both the iterable and the initializer must evaluate their assertion,
/// including when the body ignores the corresponding binding.
fn checked_operand(
    spelling: &str,
    binding: &str,
    length: usize,
    flags: u8,
    check_iterable: bool,
) -> Result<ContractCase, BuilderError> {
    let negated = flags & 1 != 0;
    let predicate = if negated { "!x" } else { "x" };
    let elements = vec!["x"; length].join(", ");
    let assertion = format!(r#"assert!({predicate}, "valid");"#);
    let (family, source) = if check_iterable {
        (
            "unrolled_loop_checked_iterable",
            format!(
                "fn checked(x: bool) -> bool[{length}] {{ {assertion} [{elements}] }}\n\
                 pub fn main(x: bool) -> bool {{ {spelling} ({binding}, acc) in checked(x) {{ acc }}(true) }}\n"
            ),
        )
    } else {
        (
            "unrolled_loop_checked_initializer",
            format!(
                "fn checked(x: bool) {{ {assertion} }}\n\
                 pub fn main(x: bool) -> bool {{ {spelling} (_, {binding}) in [{elements}] {{ () }}(checked(x)); true }}\n"
            ),
        )
    };
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(1))?;
    let predicate = if negated { builder.not(x)? } else { x };
    let token = builder.after_all(&[])?;
    builder.assert(token, predicate, "expected valid", "")?;
    let result = builder.literal(IrValue::bool(true))?;
    let reference = builder.build_package(result, "reference")?;
    Ok(ContractCase::single(family, source, reference.to_string()))
}

/// Iterating an enum directly or inside a singleton tuple selects the same
/// sequence of tags. Compare a parity fold over those tags to Boolean XOR.
fn enum_iterable(
    spelling: &str,
    wrap: bool,
    length: usize,
    flags: u8,
) -> Result<ContractCase, BuilderError> {
    let dynamic = flags & 2 != 0;
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(1))?;
    let mut result = x;
    let mut elements = Vec::with_capacity(length);
    for i in 0..length {
        let one = (i % 2 == 0) ^ (flags & 1 != 0);
        let (tag, other) = if one {
            ("Choice::One", "Choice::Zero")
        } else {
            ("Choice::Zero", "Choice::One")
        };
        let element = if dynamic {
            let is_one = if one { x } else { builder.not(x)? };
            result = builder.xor(result, is_one)?;
            format!("if x {{ {tag} }} else {{ {other} }}")
        } else {
            if one {
                result = builder.not(result)?;
            }
            tag.to_owned()
        };
        elements.push(if wrap {
            format!("({element},)")
        } else {
            element
        });
    }
    let iterable = elements.join(", ");
    let pattern = if wrap { "(element,)" } else { "element" };
    let source = format!(
        "enum Choice : u1 {{ Zero = 0, One = 1 }}\n\
         pub fn main(x: bool) -> bool {{\n\
             {spelling} ({pattern}, acc) in [{iterable}] {{ acc ^ (element == Choice::One) }}(x)\n\
         }}\n"
    );
    let reference = builder.build_package(result, "reference")?;
    Ok(ContractCase::single(
        "unrolled_loop_enum_elements",
        source,
        reference.to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::generate;
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn result_is_the_initializer_or_last_element() -> Result<(), Box<dyn std::error::Error>> {
        // The initializer is x. The array starts with x and alternates.
        for (length_option, expected) in [
            (4, [false, true]), // zero trips
            (0, [false, true]), // one trip
            (1, [true, false]), // two trips
        ] {
            let case = generate(&[1, 0, 0, length_option, 0])?;
            let package = IrPackage::parse_ir(&case.expected_ir, None)?;
            let function = package.get_function("main")?;
            for (x, want) in [false, true].into_iter().zip(expected) {
                assert_eq!(
                    function.interpret(&[XlsIrValue::bool(x)])?,
                    XlsIrValue::bool(want)
                );
            }
        }
        Ok(())
    }
}
