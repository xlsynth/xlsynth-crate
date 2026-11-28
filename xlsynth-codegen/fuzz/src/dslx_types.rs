// SPDX-License-Identifier: Apache-2.0

//! Constructs annotated port types through the production DSLX-to-SV bridge.

use std::fmt::Write;
use std::path::Path;

use xlsynth::dslx::ImportData;
use xlsynth::dslx_bridge::convert_leaf_module;
use xlsynth_codegen::sv_bridge_builder::{SvBridgeBuilder, SvEnumCaseNamingPolicy};
use xlsynth_pir::ir::{NodePayload, Package};

pub(crate) const TYPE_KIND_COUNT: u8 = 11;

/// Returns the actual bounded type family selected by one entropy byte.
pub(crate) fn type_kind(choice: u8) -> &'static str {
    match choice % TYPE_KIND_COUNT {
        0 => "unsigned",
        1 => "signed",
        2 => "packed_array",
        3 => "struct",
        4 => "enum",
        5 => "signed_struct",
        6 => "struct_array",
        7 => "nested_array",
        8 => "enum_struct",
        9 => "enum_array",
        10 => "signed_array_struct",
        _ => unreachable!("type choice reduced modulo kind count"),
    }
}

/// Annotates each positive-width top data port using four reproducible choice
/// bytes and returns the SystemVerilog package containing its converted type.
pub fn apply(package: &mut Package, entropy: &[u8]) -> String {
    let block = package
        .get_top_block_mut()
        .expect("fuzz package has a top block");
    let ports: Vec<_> = block
        .input_ports()
        .chain(block.output_ports())
        .map(|port| (port, block.port_type(port).bit_count()))
        .filter(|(_, width)| *width != 0)
        .collect();
    if ports.is_empty() {
        return String::new();
    }

    let mut source = String::new();
    for (index, (port, width)) in ports.into_iter().enumerate() {
        let mut choices = [0; 4];
        for (offset, value) in choices.iter_mut().enumerate() {
            *value = entropy.get(index * 4 + offset).copied().unwrap_or(0);
        }
        append_type(&mut source, index, width, choices);
        let annotation = Some(format!("fuzz_types::t{index}_t"));
        match &mut block.get_node_mut(port).payload {
            NodePayload::InputPort { sv_type, .. } | NodePayload::OutputPort { sv_type, .. } => {
                *sv_type = annotation;
            }
            _ => unreachable!("block data port has an input/output payload"),
        }
    }

    let mut imports = ImportData::default();
    let mut bridge = SvBridgeBuilder::with_enum_case_policy(SvEnumCaseNamingPolicy::EnumQualified);
    convert_leaf_module(
        &mut imports,
        &source,
        Path::new("fuzz_types.x"),
        &mut bridge,
    )
    .unwrap_or_else(|error| panic!("DSLX type conversion failed: {error}\nDSLX:\n{source}"));
    format!("package fuzz_types;\n{}\nendpackage\n", bridge.build())
}

/// Appends one bounded DSLX type declaration with exactly the required width.
fn append_type(source: &mut String, index: usize, width: usize, choices: [u8; 4]) {
    match choices[0] % TYPE_KIND_COUNT {
        0 => writeln!(source, "type T{index} = uN[{width}];").unwrap(),
        1 => writeln!(source, "type T{index} = sN[{width}];").unwrap(),
        2 => {
            let length = array_length(width, choices[1]);
            writeln!(source, "type T{index} = uN[{}][{length}];", width / length).unwrap();
        }
        3 if width == 1 => writeln!(source, "struct T{index} {{ value: uN[1] }}").unwrap(),
        3 if width >= 3 && choices[2] & 1 != 0 => {
            let tail = 1 + usize::from(choices[1]) % (width - 2);
            let inner = width - tail;
            let upper = 1 + usize::from(choices[3]) % (inner - 1);
            writeln!(
                source,
                "struct T{index}Inner {{ upper: uN[{upper}], lower: uN[{}] }}\nstruct T{index} {{ nested: T{index}Inner, tail: uN[{tail}] }}",
                inner - upper
            )
            .unwrap();
        }
        3 => {
            let upper = 1 + usize::from(choices[1]) % (width - 1);
            writeln!(
                source,
                "struct T{index} {{ upper: uN[{upper}], lower: uN[{}] }}",
                width - upper
            )
            .unwrap();
        }
        4 => writeln!(source, "enum T{index}: uN[{width}] {{ Zero = 0, One = 1 }}").unwrap(),
        5 if width == 1 => writeln!(source, "struct T{index} {{ value: sN[1] }}").unwrap(),
        5 => {
            let signed = 1 + usize::from(choices[1]) % (width - 1);
            writeln!(
                source,
                "struct T{index} {{ upper: sN[{signed}], lower: uN[{}] }}",
                width - signed
            )
            .unwrap();
        }
        6 => {
            let length = array_length(width, choices[1]);
            let element = width / length;
            if element == 1 {
                writeln!(source, "struct T{index}Item {{ value: sN[1] }}").unwrap();
            } else {
                let upper = 1 + usize::from(choices[2]) % (element - 1);
                writeln!(
                    source,
                    "struct T{index}Item {{ upper: sN[{upper}], lower: uN[{}] }}",
                    element - upper
                )
                .unwrap();
            }
            writeln!(source, "type T{index} = T{index}Item[{length}];").unwrap();
        }
        7 => {
            let outer = array_length(width, choices[1]);
            let inner = array_length(width / outer, choices[2]);
            writeln!(
                source,
                "type T{index} = uN[{}][{inner}][{outer}];",
                width / outer / inner
            )
            .unwrap();
        }
        8 => {
            let tag = if width == 1 {
                1
            } else {
                1 + usize::from(choices[1]) % (width - 1)
            };
            writeln!(
                source,
                "enum T{index}Tag: uN[{tag}] {{ Zero = 0, One = 1 }}"
            )
            .unwrap();
            if tag == width {
                writeln!(source, "struct T{index} {{ tag: T{index}Tag }}").unwrap();
            } else {
                writeln!(
                    source,
                    "struct T{index} {{ tag: T{index}Tag, data: sN[{}] }}",
                    width - tag
                )
                .unwrap();
            }
        }
        9 => {
            let length = array_length(width, choices[1]);
            writeln!(
                source,
                "enum T{index}Item: uN[{}] {{ Zero = 0, One = 1 }}",
                width / length
            )
            .unwrap();
            writeln!(source, "type T{index} = T{index}Item[{length}];").unwrap();
        }
        10 if width == 1 => writeln!(source, "struct T{index} {{ values: sN[1][1] }}").unwrap(),
        10 => {
            let tail = 1 + usize::from(choices[1]) % (width - 1);
            let array_width = width - tail;
            let length = array_length(array_width, choices[2]);
            writeln!(
                source,
                "struct T{index} {{ values: sN[{}][{length}], tail: uN[{tail}] }}",
                array_width / length
            )
            .unwrap();
        }
        _ => unreachable!("type choice reduced modulo kind count"),
    }
}

/// Chooses a bounded divisor, retaining a singleton array at prime widths.
fn array_length(width: usize, choice: u8) -> usize {
    let factors: Vec<_> = (2..=width.min(8))
        .filter(|factor| width % factor == 0)
        .collect();
    if factors.is_empty() {
        1
    } else {
        factors[usize::from(choice) % factors.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlsynth_pir::ir_parser::Parser;

    /// Exercises each DSLX shape through the real typechecker and SV bridge.
    #[test]
    fn choices_convert_all_dslx_type_shapes() {
        let source = r#"package typed

top block typed(data: bits[8], result: bits[8]) {
  data: bits[8] = input_port(name=data, id=1)
  result: () = output_port(data, name=result, id=2)
}
"#;
        for (kind, expected) in [
            (0, "typedef logic [7:0] t0_t;"),
            (1, "typedef logic signed [7:0] t0_t;"),
            (2, "typedef logic [1:0] [3:0] t0_t;"),
            (3, "typedef struct packed {"),
            (4, "typedef enum logic [7:0] {"),
        ] {
            let mut package = Parser::new(source).parse_package().unwrap();
            let prelude = apply(&mut package, &[kind, 0, 0, 0, 4, 0, 0, 0]);
            assert!(prelude.contains(expected), "kind {kind}:\n{prelude}");
            assert!(prelude.contains("T1_Zero"), "{prelude}");
            let block = package.get_top_block().unwrap();
            for (index, port) in block.input_ports().chain(block.output_ports()).enumerate() {
                let annotation = match &block.get_node(port).payload {
                    NodePayload::InputPort { sv_type, .. }
                    | NodePayload::OutputPort { sv_type, .. } => sv_type,
                    _ => unreachable!(),
                };
                assert_eq!(
                    annotation.as_deref(),
                    Some(format!("fuzz_types::t{index}_t").as_str())
                );
            }
        }
    }

    /// Converts composite port shapes at nontrivial and singleton widths.
    #[test]
    fn composite_types_convert_and_match_pir() {
        for width in [1, 12] {
            for kind in 5..TYPE_KIND_COUNT {
                let ir = format!(
                    "package composite\n\ntop block composite(data: bits[{width}], result: bits[{width}]) {{\n  data: bits[{width}] = input_port(name=data, id=1)\n  result: () = output_port(data, name=result, id=2)\n}}\n"
                );
                let mut declaration = String::new();
                append_type(&mut declaration, 0, width, [kind, 0, 0, 0]);
                declaration.push_str("fn identity(value: T0) -> T0 { value }");
                let mut imports = ImportData::default();
                let checked = xlsynth::dslx::parse_and_typecheck(
                    &declaration,
                    "composite.x",
                    "composite",
                    &mut imports,
                )
                .unwrap();
                let module = checked.get_module();
                let Some(xlsynth::dslx::MatchableModuleMember::Function(function)) = module
                    .get_member(module.get_member_count() - 1)
                    .to_matchable()
                else {
                    panic!("identity function absent");
                };
                let actual_width = checked
                    .get_type_info()
                    .get_type_for_type_annotation(&function.get_param(0).get_type_annotation())
                    .unwrap()
                    .get_total_bit_count()
                    .unwrap();
                assert_eq!(actual_width, width, "kind={}", type_kind(kind));

                let mut package = Parser::new(&ir).parse_package().unwrap();
                let converted = apply(&mut package, &[kind, 0, 0, 0, kind, 0, 0, 0]);
                assert!(
                    converted.contains("t0_t;"),
                    "kind={} width={width}:\n{converted}",
                    type_kind(kind)
                );
            }
        }
        for kind in 5..TYPE_KIND_COUNT {
            crate::port_types::check(&[0, kind, 0, 0, 0, kind, 0, 0, 0])
                .unwrap_or_else(|error| panic!("kind={}: {error}", type_kind(kind)));
        }
    }

    /// Does not request impossible SV types for ports omitted by codegen.
    #[test]
    fn zero_width_ports_need_no_sv_declarations() {
        let source = r#"package empty

top block empty(data: bits[0], result: bits[0]) {
  data: bits[0] = input_port(name=data, id=1)
  result: () = output_port(data, name=result, id=2)
}
"#;
        let mut package = Parser::new(source).parse_package().unwrap();
        assert!(apply(&mut package, &[4; 8]).is_empty());
    }

    /// Keeps nested structures bounded and preserves their exact packed width.
    #[test]
    fn nested_struct_and_prime_array_types_convert() {
        let mut source = String::new();
        append_type(&mut source, 0, 11, [2, 0, 0, 0]);
        append_type(&mut source, 1, 11, [3, 2, 1, 3]);
        let mut imports = ImportData::default();
        let mut bridge =
            SvBridgeBuilder::with_enum_case_policy(SvEnumCaseNamingPolicy::EnumQualified);
        convert_leaf_module(
            &mut imports,
            &source,
            Path::new("fuzz_types.x"),
            &mut bridge,
        )
        .unwrap();
        let converted = bridge.build();
        assert!(converted.contains("[0:0] [10:0] t0_t;"), "{converted}");
        assert!(converted.contains("t1_inner_t nested;"), "{converted}");
    }
}
