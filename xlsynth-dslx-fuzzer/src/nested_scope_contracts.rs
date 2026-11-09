// SPDX-License-Identifier: Apache-2.0

//! Checks which lexical definitions a sibling expression reads during folding.

use crate::converter_contracts::ContractCase;
use xlsynth_pir::{FnBuilder, IrValue, ir};

/// Builds two differently scoped bindings and reads both in one expression.
/// The oracle resolves each operand by its binding, independent of its name.
pub(crate) fn generate(data: &[u8]) -> Result<ContractCase, String> {
    let byte = |i: usize| data.get(i).copied().unwrap_or_default();
    let flags = byte(2);
    let outer = data.get(3).copied().unwrap_or(2);
    let inner = data.get(4).copied().unwrap_or(7);
    let inner_name = if flags & 1 == 0 { "n" } else { "m" };
    let outer_binding = format!("let n = u8:{outer};");
    let inner_block = format!("({{ let {inner_name} = u8:{inner}; {inner_name} }})");
    let inner_operand = (inner_block.as_str(), inner);
    let outer_operand = ("n", outer);
    // Either textual order is legal. In a constexpr consumer the converter
    // must also be able to resolve the outer use before the shadowing block.
    let ((lhs, left_value), (rhs, right_value)) = if flags & 2 == 0 {
        (inner_operand, outer_operand)
    } else {
        (outer_operand, inner_operand)
    };
    let (expression, value) = match byte(1) % 5 {
        0 => (
            format!("{lhs} + {rhs}"),
            left_value.wrapping_add(right_value),
        ),
        1 => (
            format!("{lhs} - {rhs}"),
            left_value.wrapping_sub(right_value),
        ),
        2 => (format!("{lhs} ^ {rhs}"), left_value ^ right_value),
        3 => (
            format!("({lhs} == {rhs}) as u8"),
            u8::from(left_value == right_value),
        ),
        _ => {
            let index = usize::from((flags >> 2) & 1);
            (
                format!("[{lhs}, {rhs}][u32:{index}]"),
                [left_value, right_value][index],
            )
        }
    };

    let (family, declarations, bindings, expected) = match byte(0) % 5 {
        0 => (
            "nested_scope_module_constant",
            format!("const K = {{ {outer_binding} {expression} }};\n"),
            "let observed = K;".into(),
            u16::from(value),
        ),
        1 => (
            "nested_scope_value_argument",
            "fn value<K: u8>() -> u8 { K }\n".into(),
            format!("{outer_binding}\n    let observed = value<{{{expression}}}>();"),
            u16::from(value),
        ),
        2 => (
            "nested_scope_type_dimension",
            String::new(),
            format!(
                "{outer_binding}\n    type Word = uN[(({expression}) as u32) + u32:1];\n    let observed = bit_count<Word>();"
            ),
            // Add one after widening to keep every generated dimension in 1..=256.
            u16::from(value) + 1,
        ),
        3 => (
            "nested_scope_runtime",
            String::new(),
            format!("{outer_binding}\n    let observed = {expression};"),
            u16::from(value),
        ),
        _ => (
            "nested_scope_local_constant",
            String::new(),
            format!("{outer_binding}\n    const K = {expression};\n    let observed = K;"),
            u16::from(value),
        ),
    };
    let source = format!(
        r#"{declarations}pub fn main(x: u16) -> u16 {{
    {bindings}
    x + (observed as u16)
}}
"#
    );
    let mut builder = FnBuilder::new("main");
    let x = builder
        .param("x", ir::Type::Bits(16))
        .map_err(|e| e.to_string())?;
    let literal = IrValue::make_ubits(16, u64::from(expected)).map_err(|e| e.to_string())?;
    let offset = builder.literal(literal).map_err(|e| e.to_string())?;
    let result = builder.add(x, offset).map_err(|e| e.to_string())?;
    let package = builder
        .build_package(result, "expected")
        .map_err(|e| e.to_string())?;
    Ok(ContractCase::single(family, source, package.to_string()))
}

#[cfg(test)]
mod tests {
    use super::generate;
    use std::path::Path;
    use xlsynth::{DslxConvertOptions, IrPackage, XlsIrValue, convert_dslx_to_ir_text};

    #[test]
    fn alpha_renaming_changes_only_the_inner_binding() -> Result<(), String> {
        let shadow = generate(&[])?;
        let renamed = generate(&[0, 0, 1])?;
        assert_eq!(
            shadow.modules[0].text,
            r#"const K = { let n = u8:2; ({ let n = u8:7; n }) + n };
pub fn main(x: u16) -> u16 {
    let observed = K;
    x + (observed as u16)
}
"#
        );
        assert_eq!(
            renamed.modules[0].text,
            shadow.modules[0]
                .text
                .replace("let n = u8:7; n", "let m = u8:7; m")
        );
        assert_eq!(shadow.expected_ir, renamed.expected_ir);
        Ok(())
    }

    #[test]
    fn consumers_observe_both_definitions_in_operand_order()
    -> Result<(), Box<dyn std::error::Error>> {
        // 250 + 7 wraps to 1 at the expression; the public addition is u16.
        for site in 0..5 {
            for (op, flags, value) in [
                (0, 0, 1u16),
                (1, 0, 13),
                (1, 2, 243),
                (2, 0, 253),
                (3, 0, 0),
                (4, 0, 7),
                (4, 4, 250),
                (4, 2, 250),
                (4, 6, 7),
            ] {
                let case = generate(&[site, op, flags, 250, 7])?;
                let package = IrPackage::parse_ir(&case.expected_ir, None)?;
                let function = package.get_function("main")?;
                // The alpha-renamed source exercises the same operation without
                // the known compiler bug in spelling-based environment lookup.
                let renamed = generate(&[site, op, flags | 1, 250, 7])?;
                assert_eq!(case.expected_ir, renamed.expected_ir);
                let converted = convert_dslx_to_ir_text(
                    &renamed.modules[0].text,
                    Path::new("sample.x"),
                    &DslxConvertOptions::default(),
                )?;
                let control_package = IrPackage::parse_ir(&converted.ir, None)?;
                let control = control_package.get_function("__sample__main")?;
                let offset = value + u16::from(site == 2);
                for x in [0u16, 42, u16::MAX] {
                    let args = [XlsIrValue::make_ubits(16, u64::from(x))?];
                    let expected = XlsIrValue::make_ubits(16, u64::from(x.wrapping_add(offset)))?;
                    assert_eq!(function.interpret(&args)?, expected);
                    assert_eq!(control.interpret(&args)?, expected);
                }
            }
        }
        Ok(())
    }
}
