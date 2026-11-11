// SPDX-License-Identifier: Apache-2.0

//! Array literal typing at indexing and construction sites.

use crate::{SourceModule, audit_contracts::AuditCase, converter_contracts::ContractCase};

/// Checks both literal elements against a wider operand through a dynamic
/// index, or checks the same comparison when folded in a module constant.
pub(crate) fn indexed_comparison(data: &[u8]) -> AuditCase {
    let flags = data.first().copied().unwrap_or_default();
    let widths = [(8, 16), (1, 8), (4, 8), (7, 16), (15, 32), (31, 32)];
    let (literal_width, width) =
        widths[usize::from(data.get(1).copied().unwrap_or_default()) % widths.len()];
    let first = (1_u64 << literal_width) - 1;
    let second = first ^ u64::from(flags & 4 != 0);
    let needle = first ^ u64::from(flags & 8 != 0);
    let equal = flags & 16 == 0;
    let matches = [first, second].map(|value| (value == needle) == equal);
    let operator = if equal { "==" } else { "!=" };
    let array = format!("[{first}, {second}]");
    let typed = flags & 1 != 0;
    let constexpr = flags & 32 != 0;
    let index = usize::from(flags & 64 != 0);
    let (source, assertions) = if constexpr {
        let array = if typed {
            format!("(u{width}[2]:{array})")
        } else {
            array
        };
        (
            format!(
                "const MATCHED = {array}[u32:{index}] {operator} u{width}:{needle};\npub fn main() -> bool {{ MATCHED }}\n"
            ),
            format!("    assert_eq(main(), {});\n", matches[index]),
        )
    } else {
        let comparison = if typed {
            format!("let a: u{width}[2] = {array};\n    a")
        } else {
            array
        };
        (
            format!(
                "pub fn main(i: u32) -> bool {{\n    {comparison}[i] {operator} u{width}:{needle}\n}}\n"
            ),
            format!(
                "    assert_eq((main(u32:0), main(u32:1)), ({}, {}));\n",
                matches[0], matches[1]
            ),
        )
    };
    let family = "array_literal_index_comparison";
    if flags & 2 != 0 {
        return AuditCase::LiteralRuntime {
            family,
            module: SourceModule {
                name: "sample".into(),
                text: format!("{source}#[test]\nfn literal_runtime() {{\n{assertions}}}\n"),
            },
        };
    }
    let [at_zero, at_one] = matches.map(u8::from);
    let expected_ir = if constexpr {
        let result = u8::from(matches[index]);
        format!(
            r#"package expected

top fn main() -> bits[1] {{
  ret result: bits[1] = literal(value={result}, id=1)
}}
"#
        )
    } else {
        // XLS clamps an index above the end to the last element. Bytecode
        // only checks in-range indices because it diagnoses out-of-bounds.
        format!(
            r#"package expected

top fn main(i: bits[32] id=1) -> bits[1] {{
  zero: bits[32] = literal(value=0, id=2)
  first: bits[1] = literal(value={at_zero}, id=3)
  last: bits[1] = literal(value={at_one}, id=4)
  is_first: bits[1] = eq(i, zero, id=5)
  ret result: bits[1] = sel(is_first, cases=[last, first], id=6)
}}
"#
        )
    };
    AuditCase::Semantic(ContractCase::single(family, source, expected_ir))
}

/// Checks that a typed empty literal converts with its declared element type.
pub(crate) fn empty(data: &[u8]) -> AuditCase {
    let flags = data.first().copied().unwrap_or_default();
    let widths = [8, 1, 16, 32, 65];
    let width = widths[usize::from(data.get(1).copied().unwrap_or_default()) % widths.len()];
    let sign = if flags & 4 == 0 { 'u' } else { 's' };
    let element = if width <= 64 {
        format!("{sign}{width}")
    } else {
        format!("{sign}N[{width}]")
    };
    let length = usize::from(flags & 2 != 0);
    let elements = if length == 0 { "" } else { "0" };
    let array_type = format!("{element}[{length}]");
    let prefix = if flags & 1 == 0 {
        format!("{array_type}:")
    } else {
        String::new()
    };
    let source = format!("pub fn main() -> {array_type} {{ {prefix}[{elements}] }}\n");
    let family = "empty_array_literal";
    if flags & 8 != 0 {
        return AuditCase::LiteralRuntime {
            family,
            module: SourceModule {
                name: "sample".into(),
                text: format!(
                    "{source}#[test]\nfn literal_runtime() {{\n    assert_eq(main(), {array_type}:[{elements}]);\n}}\n"
                ),
            },
        };
    }
    let ir_type = format!("bits[{width}][{length}]");
    let nodes = if length == 0 {
        format!("  ret result: {ir_type} = array(id=1)")
    } else {
        format!(
            "  element: bits[{width}] = literal(value=0, id=1)\n  ret result: {ir_type} = array(element, id=2)"
        )
    };
    let expected_ir = format!("package expected\n\ntop fn main() -> {ir_type} {{\n{nodes}\n}}\n");
    AuditCase::Semantic(ContractCase::single(family, source, expected_ir))
}

#[cfg(test)]
mod tests {
    use super::{AuditCase, empty, indexed_comparison};
    use xlsynth::IrPackage;

    #[test]
    fn raw_index_checks_the_same_expression_in_ir_and_bytecode() {
        let AuditCase::Semantic(ir) = indexed_comparison(&[]) else {
            panic!("expected the raw IR contract");
        };
        let AuditCase::LiteralRuntime { module, .. } = indexed_comparison(&[2]) else {
            panic!("expected the raw bytecode contract");
        };
        assert_eq!(
            ir.modules[0].text,
            "pub fn main(i: u32) -> bool {\n    [255, 255][i] == u16:255\n}\n"
        );
        assert_eq!(
            module.text,
            format!(
                "{}#[test]\nfn literal_runtime() {{\n    assert_eq((main(u32:0), main(u32:1)), (true, true));\n}}\n",
                ir.modules[0].text
            )
        );
    }

    #[test]
    fn empty_literal_has_a_typed_zero_element_reference() -> Result<(), Box<dyn std::error::Error>>
    {
        let AuditCase::Semantic(case) = empty(&[]) else {
            panic!("expected the typed empty array IR contract");
        };
        let AuditCase::LiteralRuntime { module, .. } = empty(&[8]) else {
            panic!("expected the typed empty array bytecode contract");
        };
        assert_eq!(
            case.modules[0].text,
            "pub fn main() -> u8[0] { u8[0]:[] }\n"
        );
        assert_eq!(
            module.text,
            format!(
                "{}#[test]\nfn literal_runtime() {{\n    assert_eq(main(), u8[0]:[]);\n}}\n",
                case.modules[0].text
            )
        );
        let package = IrPackage::parse_ir(&case.expected_ir, None)?;
        assert_eq!(
            package
                .get_function("main")?
                .get_type()?
                .return_type()
                .to_string(),
            "bits[8][0]"
        );
        Ok(())
    }
}
