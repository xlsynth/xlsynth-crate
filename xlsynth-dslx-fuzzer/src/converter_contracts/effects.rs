// SPDX-License-Identifier: Apache-2.0

//! Check runtime failure conditions and the size of conditional guard lowering.

use super::ContractCase;
use xlsynth_pir::{BValue, BuilderError, FnBuilder, IrValue, ir};

/// Compares a bounded unsigned word to a generator-selected literal.
fn equals_literal(
    builder: &mut FnBuilder,
    x: BValue,
    width: usize,
    value: usize,
) -> Result<BValue, BuilderError> {
    let value = IrValue::make_ubits(width, value as u64)
        .map_err(|e| BuilderError::InvalidOperation(e.to_string()))?;
    let literal = builder.literal(value)?;
    builder.eq(x, literal)
}

/// Computes membership over the declared representations before comparing to
/// V0.
fn enum_reference(width: usize, members: &[usize]) -> Result<String, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(width))?;
    let matches = members
        .iter()
        .map(|&value| equals_literal(&mut builder, x, width, value))
        .collect::<Result<Vec<_>, _>>()?;
    let declared = builder.or_all(&matches)?;
    let token = builder.after_all(&[])?;
    builder.assert(token, declared, "enum cast requires a declared member", "")?;
    Ok(builder.build_package(matches[0], "reference")?.to_string())
}

/// Varies enum width and member representations, including fully declared
/// domains.
pub(super) fn enum_cast(data: &[u8]) -> Result<ContractCase, String> {
    let width = 1 + usize::from(data.first().copied().unwrap_or_default() % 8);
    let domain = 1 << width;
    let count = 2 + usize::from(data.get(1).copied().unwrap_or_default()) % (domain.min(8) - 1);
    let mut members = Vec::with_capacity(count);
    for index in 0..count {
        let mut value = usize::from(data.get(index + 2).copied().unwrap_or_default()) % domain;
        // The typechecker requires distinct member values. There is always a
        // free representation because count never exceeds the domain size.
        while members.contains(&value) {
            value = (value + 1) % domain;
        }
        members.push(value);
    }
    let declarations = members
        .iter()
        .enumerate()
        .map(|(i, value)| format!("V{i} = {value}"))
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        r#"enum E : u{width} {{ {declarations} }}
pub fn main(x: u{width}) -> bool {{
    let e = x as E;
    e == E::V0
}}
"#
    );
    Ok(ContractCase::single(
        "enum_cast_membership",
        source,
        enum_reference(width, &members).map_err(|error| error.to_string())?,
    ))
}

/// The assertion can fail only if all guard bits are clear and x differs
/// from the requested literal. Every path returns x.
fn guard_reference(width: usize, value: usize, depth: usize) -> Result<String, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(width))?;
    let mut allowed = (0..depth)
        .map(|index| builder.param(&format!("b{index}"), ir::Type::Bits(1)))
        .collect::<Result<Vec<_>, _>>()?;
    allowed.push(equals_literal(&mut builder, x, width, value)?);
    let predicate = builder.or_all(&allowed)?;
    let token = builder.after_all(&[])?;
    builder.assert(token, predicate, "nested conditional assertion", "")?;
    Ok(builder.build_package(x, "reference")?.to_string())
}

/// Nests a single assertion in mixed then/else arms with independent input
/// guards.
pub(super) fn guard_growth(data: &[u8]) -> Result<ContractCase, String> {
    let depth = 1 + usize::from(data.first().copied().unwrap_or_default() % 8);
    let then_arms = data.get(1).copied().unwrap_or_default();
    let width = 1 + usize::from(data.get(2).copied().unwrap_or_default() % 8);
    let value = usize::from(data.get(3).copied().unwrap_or_default()) % (1 << width);
    let params = (0..depth)
        .map(|index| format!("b{index}: bool"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut body = format!("assert_eq(x, u{width}:{value}); x");
    for index in 0..depth {
        body = if then_arms & (1 << index) != 0 {
            format!("if !b{index} {{ {body} }} else {{ x }}")
        } else {
            format!("if b{index} {{ x }} else {{ {body} }}")
        };
    }
    let source = format!("pub fn main(x: u{width}, {params}) -> u{width} {{ {body} }}\n");
    let mut case = ContractCase::single(
        "nested_guard_growth",
        source,
        guard_reference(width, value, depth).map_err(|error| error.to_string())?,
    );
    // Each level adds one guard bit to the assertion's path condition. Allow
    // two AND nodes per level so a small linear lowering change still passes.
    case.max_and_nodes = Some(2 * depth);
    Ok(case)
}
