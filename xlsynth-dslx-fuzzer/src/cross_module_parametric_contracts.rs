// SPDX-License-Identifier: Apache-2.0

//! Value-parametric bindings that are forwarded across module boundaries.

use crate::{SourceModule, converter_contracts::ContractCase};

/// Calls two functions whose DSLX names can map to the same IR symbol.
pub(crate) fn specialization_names(data: &[u8]) -> ContractCase {
    let flags = data.first().copied().unwrap_or_default();
    let width = 2 + usize::from(data.get(1).copied().unwrap_or(6) % 31);
    let rename = flags & 1 != 0;
    let apply = "pub fn apply<N: u32>(x: uN[N]) -> uN[N] { x + uN[N]:1 }\n";
    let (family, mut modules, imports, calls) = if flags & 2 == 0 {
        let (nested, flat, renamed) = if flags & 4 == 0 {
            ("lane.ops", "lane_ops", "flat_ops")
        } else {
            ("lane.io.ops", "lane_io_ops", "flat_io_ops")
        };
        let flat = if rename { renamed } else { flat };
        (
            "imported_parametric_module_name_collision",
            vec![
                SourceModule {
                    name: nested.replace('.', "/"),
                    text: apply.into(),
                },
                SourceModule {
                    name: flat.into(),
                    text: "pub fn apply<N: u32>(x: uN[N]) -> uN[N] { x + uN[N]:2 }\n".into(),
                },
            ],
            format!("import {nested} as nested;\nimport {flat} as flat;"),
            "(nested::apply(x), flat::apply(x))".into(),
        )
    } else {
        let name = if rename {
            "apply_other".into()
        } else {
            format!("apply__{width}")
        };
        (
            "imported_parametric_function_name_collision",
            vec![SourceModule {
                name: "ops".into(),
                text: format!(
                    "{apply}pub fn {name}(x: uN[{width}]) -> uN[{width}] {{ x + uN[{width}]:2 }}\n"
                ),
            }],
            "import ops;".into(),
            format!("(ops::apply(x), ops::{name}(x))"),
        )
    };
    modules.push(SourceModule {
        name: "sample".into(),
        text: format!(
            r#"{imports}
pub fn main(x: uN[{width}]) -> (uN[{width}], uN[{width}]) {{ {calls} }}
"#
        ),
    });
    let expected_ir = format!(
        r#"package expected

top fn main(x: bits[{width}] id=1) -> (bits[{width}], bits[{width}]) {{
  one: bits[{width}] = literal(value=1, id=2)
  two: bits[{width}] = literal(value=2, id=3)
  first: bits[{width}] = add(x, one, id=4)
  second: bits[{width}] = add(x, two, id=5)
  ret result: (bits[{width}], bits[{width}]) = tuple(first, second, id=6)
}}
"#
    );
    ContractCase {
        family,
        modules,
        expected_ir,
        max_and_nodes: None,
    }
}

/// Sends two equally wide struct specializations to an imported free function.
/// Reversing the two values changes both the derived signedness and the mask.
pub(crate) fn struct_specializations(data: &[u8]) -> ContractCase {
    let byte = |i: usize, default: u8| u32::from(data.get(i).copied().unwrap_or(default));
    let flags = byte(0, 0);
    let low = 1 + byte(1, 0) % 8;
    let high = low + 1 + byte(2, 1) % 12;
    let width = high + low;
    let record_offset = byte(3, 7);
    let observer_offset = byte(4, 11);
    let (a, b) = if flags & 1 == 0 {
        ("A", "B")
    } else {
        ("InnerA", "InnerB")
    };
    let call = if flags & 2 == 0 {
        "ops::observe<B, A, W, S>(packet)"
    } else {
        "ops::observe(packet)"
    };
    let reverse = flags & 4 != 0;
    let (first, second) = if reverse { (low, high) } else { (high, low) };
    let modules = vec![
        SourceModule {
            name: "records".into(),
            text: format!(
                r#"pub const OFFSET = u32:{record_offset};
pub struct Packet<A: u32, B: u32, W: u32 = {{A + B}}, S: bool = {{A > B}}> {{
    data: xN[S][W],
}}
"#
            ),
        },
        SourceModule {
            name: "ops".into(),
            text: format!(
                r#"import records;
const OFFSET = u32:{observer_offset};
pub fn observe<{b}: u32, {a}: u32, W: u32, S: bool,
               K: u32 = {{{a} * u32:16 + {b} + OFFSET + records::OFFSET}}>(
    packet: records::Packet<{a}, {b}, W, S>
) -> u32 {{
    let {a} = packet.data as u32;
    {a} ^ K
}}
"#
            ),
        },
        SourceModule {
            name: "sample".into(),
            text: format!(
                r#"import records;
import ops;
const A = u32:{first};
const B = u32:{second};
fn relay<A: u32, B: u32, W: u32 = {{A + B}}, S: bool = {{A > B}}>(x: uN[W]) -> u32 {{
    let packet = records::Packet<A, B> {{ data: x as xN[S][W] }};
    {call}
}}
pub fn main(x: uN[{width}], y: uN[{width}]) -> (u32, u32) {{
    (relay<A, B>(x), relay<B, A>(y))
}}
"#
            ),
        },
    ];

    let left_mask = first * 16 + second + record_offset + observer_offset;
    let right_mask = second * 16 + first + record_offset + observer_offset;
    let (left_ext, right_ext) = if reverse {
        ("zero_ext", "sign_ext")
    } else {
        ("sign_ext", "zero_ext")
    };
    let expected_ir = format!(
        r#"package expected

top fn main(x: bits[{width}] id=1, y: bits[{width}] id=2) -> (bits[32], bits[32]) {{
  left: bits[32] = {left_ext}(x, new_bit_count=32, id=3)
  right: bits[32] = {right_ext}(y, new_bit_count=32, id=4)
  left_mask: bits[32] = literal(value={left_mask}, id=5)
  right_mask: bits[32] = literal(value={right_mask}, id=6)
  first: bits[32] = xor(left, left_mask, id=7)
  second: bits[32] = xor(right, right_mask, id=8)
  ret result: (bits[32], bits[32]) = tuple(first, second, id=9)
}}
"#
    );
    ContractCase {
        family: "imported_struct_parametric_specializations",
        modules,
        expected_ir,
        max_and_nodes: None,
    }
}

/// Uses a local binding in an imported specialization or a runtime cast width.
pub(crate) fn shadowed_expression(data: &[u8]) -> ContractCase {
    let byte = |i: usize, default: u8| u32::from(data.get(i).copied().unwrap_or(default));
    let flags = byte(0, 0);
    let outer = 1 + byte(1, 2) % 15;
    let local = outer + 1 + byte(2, 3) % 15;
    let step = 1 + byte(3, 0) % 4;
    let local_name = if flags & 1 == 0 { "N" } else { "Local" };
    if flags & 8 != 0 {
        return shadowed_cast(outer, local, local_name, step);
    }
    let expression = format!("{local_name} + u32:{step}");
    let (binding, argument) = if flags & 4 == 0 {
        (String::new(), format!("{{{expression}}}"))
    } else {
        (
            format!("    let Derived = {expression};\n"),
            "Derived".into(),
        )
    };
    let (family, observe, expected) = if flags & 2 == 0 {
        (
            "imported_function_shadowed_parametric",
            format!("let observed = leaf::value<{argument}>();"),
            local + step,
        )
    } else {
        (
            "imported_struct_shadowed_parametric",
            format!(
                "type LocalWord = leaf::Word<{argument}>;\n    let observed = bit_count<LocalWord>();"
            ),
            local + step + 1,
        )
    };
    let modules = vec![
        SourceModule {
            name: "leaf".into(),
            text: format!(
                r#"pub const N = u32:{local};
pub struct Word<N: u32, W: u32 = {{N + u32:1}}> {{ value: uN[W] }}
pub fn value<N: u32>() -> u32 {{ N }}
"#
            ),
        },
        SourceModule {
            name: "sample".into(),
            text: format!(
                r#"import leaf;
fn relay<N: u32>(x: u32) -> u32 {{
    let {local_name} = leaf::N;
{binding}    {observe}
    x ^ observed
}}
pub fn main(x: u32) -> u32 {{ relay<u32:{outer}>(x) }}
"#
            ),
        },
    ];
    let expected_ir = format!(
        r#"package expected

top fn main(x: bits[32] id=1) -> bits[32] {{
  mask: bits[32] = literal(value={expected}, id=2)
  ret result: bits[32] = xor(x, mask, id=3)
}}
"#
    );
    ContractCase {
        family,
        modules,
        expected_ir,
        max_and_nodes: None,
    }
}

/// Observes a shadowed constexpr cast width in the retained bits of a runtime
/// word.
fn shadowed_cast(outer: u32, local: u32, local_name: &str, step: u32) -> ContractCase {
    let mask = (1u64 << (local + step)) - 1;
    let modules = vec![
        SourceModule {
            name: "leaf".into(),
            text: format!(
                r#"pub fn narrow<N: u32>(x: u64) -> u64 {{
    let {local_name} = u32:{local};
    let narrowed = x as uN[{local_name} + u32:{step}];
    narrowed as u64
}}
"#
            ),
        },
        SourceModule {
            name: "sample".into(),
            text: format!(
                "import leaf;\npub fn main(x: u64) -> u64 {{ leaf::narrow<u32:{outer}>(x) }}\n"
            ),
        },
    ];
    let expected_ir = format!(
        r#"package expected

top fn main(x: bits[64] id=1) -> bits[64] {{
  mask: bits[64] = literal(value={mask}, id=2)
  ret result: bits[64] = and(x, mask, id=3)
}}
"#
    );
    ContractCase {
        family: "imported_shadowed_cast_width",
        modules,
        expected_ir,
        max_and_nodes: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{shadowed_expression, specialization_names, struct_specializations};
    use crate::converter_contracts::ContractCase;
    use xlsynth::{DslxConvertOptions, IrPackage, XlsIrValue, convert_dslx_to_ir};

    /// Compiles the source and parses the independent IR using the same import
    /// tree.
    fn packages(case: &ContractCase) -> Result<(IrPackage, IrPackage), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        for module in &case.modules {
            let path = directory.path().join(format!("{}.x", module.name));
            std::fs::create_dir_all(path.parent().expect("module path has a parent"))?;
            std::fs::write(path, &module.text)?;
        }
        let source = std::fs::read_to_string(directory.path().join("sample.x"))?;
        let actual = convert_dslx_to_ir(
            &source,
            &directory.path().join("sample.x"),
            &DslxConvertOptions {
                additional_search_paths: vec![directory.path()],
                ..Default::default()
            },
        )?;
        let expected = IrPackage::parse_ir(&case.expected_ir, None)?;
        Ok((actual.ir, expected))
    }

    #[test]
    fn same_width_specializations_keep_their_sign_and_parametric_values()
    -> Result<(), Box<dyn std::error::Error>> {
        for flags in 0..8 {
            let (actual, expected) = packages(&struct_specializations(&[flags]))?;
            let actual = actual.get_function("__sample__main")?;
            let expected = expected.get_function("main")?;
            let arg = XlsIrValue::make_ubits(4, 9)?;
            let args = [arg.clone(), arg];
            let mut values = [0xffffffbau32, 44];
            if flags & 4 != 0 {
                values.reverse();
            }
            let [first, second] = values;
            let value = XlsIrValue::parse_typed(&format!("(bits[32]:{first}, bits[32]:{second})"))?;
            assert_eq!(expected.interpret(&args)?, value);
            assert_eq!(actual.interpret(&args)?, value);
        }
        Ok(())
    }

    #[test]
    fn renaming_the_local_preserves_the_independent_shadow_reference()
    -> Result<(), Box<dyn std::error::Error>> {
        for consumer in [0, 2, 4, 6, 8] {
            let shadowed = shadowed_expression(&[consumer]);
            let renamed = shadowed_expression(&[consumer | 1]);
            assert_ne!(shadowed.modules, renamed.modules);
            assert_eq!(shadowed.expected_ir, renamed.expected_ir);
            let (actual, expected) = packages(&renamed)?;
            let (width, result) = match consumer {
                8 => (64, 0xff),
                2 | 6 => (32, 0xff ^ 9),
                _ => (32, 0xff ^ 8),
            };
            let args = [XlsIrValue::make_ubits(width, 0xff)?];
            let value = XlsIrValue::make_ubits(width, result)?;
            assert_eq!(
                actual.get_function("__sample__main")?.interpret(&args)?,
                value
            );
            assert_eq!(expected.get_function("main")?.interpret(&args)?, value);
        }
        Ok(())
    }

    #[test]
    fn renaming_module_or_function_preserves_both_results() -> Result<(), Box<dyn std::error::Error>>
    {
        for form in [0, 2, 4] {
            let collision = specialization_names(&[form]);
            let renamed = specialization_names(&[form | 1]);
            assert_eq!(collision.expected_ir, renamed.expected_ir);
            let (actual, expected) = packages(&renamed)?;
            let args = [XlsIrValue::make_ubits(8, 255)?];
            let value = XlsIrValue::parse_typed("(bits[8]:0, bits[8]:1)")?;
            assert_eq!(
                actual.get_function("__sample__main")?.interpret(&args)?,
                value
            );
            assert_eq!(expected.get_function("main")?.interpret(&args)?, value);
        }
        Ok(())
    }
}
