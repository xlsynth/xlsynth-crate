// SPDX-License-Identifier: Apache-2.0

//! Focused standalone converter properties, separate from mission generation.

mod acceptance;
mod builtin_map;
mod effects;

use crate::SourceModule;
pub(crate) use acceptance::{ArrayIdentityForm, array_identity};

/// Generated DSLX modules, their independent top function and any IR size
/// limit.
#[derive(Debug)]
pub struct ContractCase {
    pub family: &'static str,
    pub modules: Vec<SourceModule>,
    pub expected_ir: String,
    pub max_and_nodes: Option<usize>,
}

impl ContractCase {
    pub(crate) fn single(family: &'static str, text: String, expected_ir: String) -> Self {
        Self {
            family,
            modules: vec![SourceModule {
                name: "sample".into(),
                text,
            }],
            expected_ir,
            max_and_nodes: None,
        }
    }
}

/// Decodes one independently replayable converter property and its operands.
pub fn generate(data: &[u8]) -> Result<ContractCase, String> {
    let (&family, options) = data.split_first().unwrap_or((&0, &[]));
    match family % 8 {
        0 => builtin_map::generate(options),
        1 => acceptance::one_hot(options),
        2 => acceptance::payload_pattern(options),
        3 => Ok(acceptance::array_cast(options)),
        4 => effects::enum_cast(options),
        5 => effects::guard_growth(options),
        6 => Ok(pack_elements(options)),
        _ => Ok(enumerate(options)),
    }
}

/// A pack of the same literal elements must agree at runtime and
/// specialization.
fn pack_elements(data: &[u8]) -> ContractCase {
    let signed = data.first().copied().unwrap_or_default() & 1 != 0;
    let width = 2 + u32::from(data.get(1).copied().unwrap_or_default() % 7);
    let packed = 2 * width;
    let element = if signed {
        format!("s{width}")
    } else {
        format!("u{width}")
    };
    let first = if signed { -1 } else { (1 << width) - 1 };
    let source = format!(
        r#"fn pack(x: {element}[2]) -> u{packed} {{ x as u{packed} }}
fn value<K: u{packed}>() -> u{packed} {{ K }}
pub fn main(x: u{packed}) -> u{packed} {{
    let a = pack([{element}:{first}, {element}:0]);
    x ^ a ^ value<{{pack([{element}:{first}, {element}:0])}}>()
}}
"#
    );
    let ir = scalar_identity(packed);
    ContractCase::single("constexpr_array_pack", source, ir)
}

/// The element selected from enumerate must equal its original array element.
fn enumerate(data: &[u8]) -> ContractCase {
    let width = 1 + u32::from(data.get(1).copied().unwrap_or(6) % 31);
    let (import, width_type, expr) = match data.first().copied().unwrap_or_default() % 3 {
        0 => ("", "uN[32]", "[(u32:0, x)][u32:0].1"),
        1 => ("import std;\n", "uN[32]", "std::enumerate([x])[u32:0].1"),
        _ => ("import std;\n", "u32", "std::enumerate([x])[u32:0].1"),
    };
    let source = format!(
        "{import}const W = {width_type}:{width};\npub fn main(x: uN[W]) -> u{width} {{ {expr} }}\n"
    );
    let ir = scalar_identity(width);
    ContractCase::single("enumerate", source, ir)
}

/// Returns the unchanged input for cancellation and element selection oracles.
pub(crate) fn scalar_identity(width: u32) -> String {
    format!(
        r#"package expected

top fn main(x: bits[{width}] id=1) -> bits[{width}] {{
  ret x: bits[{width}] = param(name=x, id=1)
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::generate;
    use xlsynth_pir::ir_parser::Parser;

    #[test]
    fn all_converter_contract_oracles_parse_with_the_prover_parser() -> Result<(), String> {
        for family in 0..8 {
            for selector in 0..8 {
                let case = generate(&[family, selector, selector, selector, selector])?;
                Parser::new(&case.expected_ir)
                    .parse_package()
                    .map_err(|error| {
                        format!("{} reference for selector {selector}: {error}", case.family)
                    })?;
            }
        }
        Ok(())
    }
}
