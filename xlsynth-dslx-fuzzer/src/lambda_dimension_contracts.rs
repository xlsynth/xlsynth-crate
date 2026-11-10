// SPDX-License-Identifier: Apache-2.0

//! Inferred and explicit map parameters over a constant signedness and width.

use crate::audit_contracts::AuditCase;
use crate::converter_contracts::ContractCase;

/// Exercises the converter's inferred lambda name even when the entire map is
/// folded. The reference applies the scalar operation to the selected element.
pub(crate) fn generate(data: &[u8]) -> AuditCase {
    let byte = |index: usize| data.get(index).copied().unwrap_or_default();
    let flags = byte(0);
    let signed = flags & 2 != 0;
    let runtime = flags & 4 != 0;
    let widths = [8, 2, 5, 16, 32, 65];
    let width = widths[usize::from(byte(1)) % widths.len()];
    let length = 1 + byte(2) % 4;
    let index = byte(4) % length;
    let (family, parameter) = if flags & 1 == 0 {
        ("map_lambda_inferred_dimensions", "x")
    } else {
        ("map_lambda_explicit_dimensions", "x: xN[SIGNED][WIDTH]")
    };
    let (expr, op) = match byte(3) % 3 {
        0 => ("x + X", "add"),
        1 => ("x ^ X", "xor"),
        _ if signed => ("x >> u32:1", "shra"),
        _ => ("x >> u32:1", "shrl"),
    };
    let sign = if signed { 's' } else { 'u' };
    let result_type = if width <= 64 {
        format!("{sign}{width}")
    } else {
        format!("{sign}N[{width}]")
    };
    let lambda = format!("|{parameter}| {expr}");
    let declarations = format!(
        r#"const SIGNED = {signed};
const WIDTH = u32:{width};
const X = xN[SIGNED][WIDTH]:1;
"#
    );
    let (source, ir_params, operand) = if runtime {
        (
            format!(
                "{declarations}pub fn main(xs: xN[SIGNED][WIDTH][{length}]) -> {result_type} {{ map(xs, {lambda})[u32:{index}] }}\n"
            ),
            format!("xs: bits[{width}][{length}] id=1"),
            format!(
                "i: bits[32] = literal(value={index}, id=3)\n  operand: bits[{width}] = array_index(xs, indices=[i], id=4)"
            ),
        )
    } else {
        let elements = (0..length)
            .map(|i| if i % 2 == 0 { "X" } else { "!X" })
            .collect::<Vec<_>>()
            .join(", ");
        let operand_op = if index % 2 == 0 { "identity" } else { "not" };
        (
            format!(
                "{declarations}const XS = map([{elements}], {lambda});\npub fn main() -> {result_type} {{ XS[u32:{index}] }}\n"
            ),
            String::new(),
            format!("operand: bits[{width}] = {operand_op}(one, id=4)"),
        )
    };
    let expected_ir = format!(
        r#"package expected
top fn main({ir_params}) -> bits[{width}] {{
  one: bits[{width}] = literal(value=1, id=2)
  {operand}
  ret result: bits[{width}] = {op}(operand, one, id=5)
}}
"#
    );
    AuditCase::Semantic(ContractCase::single(family, source, expected_ir))
}

#[cfg(test)]
mod tests {
    use super::generate;
    use crate::audit_contracts::AuditCase;
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn parameter_annotation_is_the_only_change_in_the_constexpr_control()
    -> Result<(), Box<dyn std::error::Error>> {
        let (AuditCase::Semantic(inferred), AuditCase::Semantic(explicit)) =
            (generate(&[]), generate(&[1]))
        else {
            return Err("expected semantic contracts".into());
        };
        let source = r#"const SIGNED = false;
const WIDTH = u32:8;
const X = xN[SIGNED][WIDTH]:1;
const XS = map([X], |x| x + X);
pub fn main() -> u8 { XS[u32:0] }
"#;
        assert_eq!(inferred.modules[0].text, source);
        assert_eq!(
            explicit.modules[0].text,
            source.replace("|x|", "|x: xN[SIGNED][WIDTH]|")
        );
        assert_eq!(inferred.expected_ir, explicit.expected_ir);
        let package = IrPackage::parse_ir(&inferred.expected_ir, None)?;
        assert_eq!(
            package.get_function("main")?.interpret(&[])?,
            XlsIrValue::make_ubits(8, 2)?
        );
        Ok(())
    }

    #[test]
    fn right_shift_observes_signedness_on_the_selected_lane()
    -> Result<(), Box<dyn std::error::Error>> {
        let inputs = [XlsIrValue::parse_typed("[bits[5]:1, bits[5]:30]")?];
        for (flags, expected) in [(0, 15), (2, 31), (4, 15), (6, 31)] {
            let AuditCase::Semantic(case) = generate(&[flags, 2, 1, 2, 1]) else {
                return Err("expected semantic contract".into());
            };
            let package = IrPackage::parse_ir(&case.expected_ir, None)?;
            let args = if flags & 4 == 0 { &[][..] } else { &inputs[..] };
            assert_eq!(
                package.get_function("main")?.interpret(args)?,
                XlsIrValue::make_ubits(5, expected)?
            );
        }
        Ok(())
    }
}
