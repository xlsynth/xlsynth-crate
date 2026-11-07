// SPDX-License-Identifier: Apache-2.0

//! Checks that shared functions specialize independently for module-local
//! types.

use crate::SourceModule;
use crate::converter_contracts::ContractCase;
use xlsynth_pir::{BuilderError, FnBuilder, ir};

/// Returns the two module-local declarations, optionally alpha-renaming one.
fn type_names(flags: u8) -> [&'static str; 2] {
    ["Word", if flags & 1 == 0 { "Word" } else { "Cell" }]
}

/// Generates arrays of independently sized structs in two modules. Enumerating
/// each array preserves the value of its selected element at its original
/// width.
pub fn enumerate(data: &[u8]) -> Result<ContractCase, String> {
    let option = |index: usize| data.get(index).copied().unwrap_or_default();
    let flags = option(0);
    let types = type_names(flags);
    let length = 1 + usize::from(option(3) % 4);
    let mut modules = Vec::new();
    let mut selected = [0; 2];
    for (side, name) in ["left", "right"].into_iter().enumerate() {
        let ty = types[side];
        let width = 1 + option(1 + side) % 8;
        let mask = u8::MAX >> (8 - width);
        let index = usize::from(option(4 + side)) % length;
        let mut elements = Vec::new();
        for i in 0..length {
            let value = option(6 + side * length + i) & mask;
            elements.push(format!("{ty} {{ v: u{width}:{value} }}"));
            if i == index {
                selected[side] = value;
            }
        }
        let elements = elements.join(", ");
        modules.push(SourceModule {
            name: name.to_owned(),
            text: format!(
                r#"import std;
struct {ty} {{ v: u{width} }}
pub fn get() -> u8 {{ std::enumerate([{elements}])[u32:{index}].1.v as u8 }}
"#
            ),
        });
    }
    // Swapping both calls and the condition changes specialization order while
    // preserving the public result for every input.
    let body = if flags & 2 == 0 {
        "if x { left::get() } else { right::get() }"
    } else {
        "if !x { right::get() } else { left::get() }"
    };
    modules.push(SourceModule {
        name: "sample".into(),
        text: format!("import left;\nimport right;\npub fn main(x: bool) -> u8 {{ {body} }}\n"),
    });
    let [left, right] = selected;
    let expected_ir = format!(
        r#"package expected
top fn main(x: bits[1] id=1) -> bits[8] {{
  left_value: bits[8] = literal(value={left}, id=2)
  right_value: bits[8] = literal(value={right}, id=3)
  ret result: bits[8] = sel(x, cases=[right_value, left_value], id=4)
}}
"#
    );
    Ok(ContractCase {
        family: "nominal_enumerate",
        modules,
        expected_ir,
        max_and_nodes: None,
    })
}

/// Exercises user generics in a separate focused family. The two source modules
/// independently choose their field order, field values and method
/// implementation.
pub fn projection(data: &[u8]) -> Result<ContractCase, String> {
    let option = |index: usize| data.get(index).copied().unwrap_or_default();
    let flags = option(0);
    let types = type_names(flags);
    let width = 1 + usize::from(option(1) % 32);
    let method = flags & 4 != 0;
    let fields = ["a", "b"];
    let field = usize::from(option(4) & 1);
    let project = if method { "pick()" } else { fields[field] };
    let mut modules = vec![SourceModule {
        name: "shared".into(),
        text: format!(
            "#![feature(generics)]\npub fn project<T: type>(v: T) -> u{width} {{ v.{project} }}\n"
        ),
    }];
    let mut selected = [0; 2];
    for (side, name) in ["left", "right"].into_iter().enumerate() {
        let ty = types[side];
        let choices = option(2 + side);
        let first = usize::from(choices & 1);
        let binding = usize::from((choices >> 1) & 1);
        let (implementation, observed) = if method {
            let observed = usize::from((choices >> 2) & 1);
            let target = fields[observed];
            (
                format!("impl {ty} {{ fn pick(self) -> u{width} {{ self.{target} }} }}\n"),
                observed,
            )
        } else {
            (String::new(), field)
        };
        let inputs = ["x", "y"];
        selected[side] = observed ^ binding;
        modules.push(SourceModule {
            name: name.to_owned(),
            text: format!(
                r#"import shared;
struct {ty} {{ {}: u{width}, {}: u{width} }}
{implementation}pub fn get(x: u{width}, y: u{width}) -> u{width} {{
    shared::project({ty} {{ a: {}, b: {} }})
}}
"#,
                fields[first],
                fields[1 - first],
                inputs[binding],
                inputs[1 - binding]
            ),
        });
    }
    let mut order = ["left", "right"];
    if flags & 2 != 0 {
        order.reverse();
        selected.reverse();
    }
    let [first, second] = order;
    modules.push(SourceModule {
        name: "sample".into(),
        text: format!(
            r#"import left;
import right;
pub fn main(x: u{width}, y: u{width}) -> (u{width}, u{width}) {{
    ({first}::get(x, y), {second}::get(x, y))
}}
"#
        ),
    });
    Ok(ContractCase {
        family: if method {
            "nominal_method"
        } else {
            "nominal_field"
        },
        modules,
        expected_ir: projection_reference(width, selected).map_err(|error| error.to_string())?,
        max_and_nodes: None,
    })
}

/// Builds the two observable outputs directly from the input bound to each
/// selected field, irrespective of its physical tuple position.
fn projection_reference(width: usize, selected: [usize; 2]) -> Result<String, BuilderError> {
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(width))?;
    let y = builder.param("y", ir::Type::Bits(width))?;
    let inputs = [x, y];
    let result = builder.tuple(&selected.map(|field| inputs[field]))?;
    Ok(builder.build_package(result, "expected")?.to_string())
}

#[cfg(test)]
mod tests {
    use super::{enumerate, projection};
    use xlsynth::{IrPackage, XlsIrValue};

    #[test]
    fn enumerate_observes_the_selected_width_and_index_in_either_order()
    -> Result<(), Box<dyn std::error::Error>> {
        for flags in 0..4 {
            // Two u3 elements in the left module, two u4 elements in the right.
            let case = enumerate(&[flags, 2, 3, 1, 1, 0, 0xff, 0xfa, 0xfc, 0xfe])?;
            let package = IrPackage::parse_ir(&case.expected_ir, None)?;
            let main = package.get_function("main")?;
            for (x, expected) in [(0, 12), (1, 2)] {
                assert_eq!(
                    main.interpret(&[XlsIrValue::make_ubits(1, x)?])?,
                    XlsIrValue::make_ubits(8, expected)?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn projection_observes_each_named_field_and_method_independently()
    -> Result<(), Box<dyn std::error::Error>> {
        for (mode, projected, expected) in [
            (0, 0, [3, 5]), // Field a: left binds x, right binds y.
            (0, 1, [5, 3]), // Field b: left binds y, right binds x.
            (4, 0, [3, 3]), // Left method reads a, right method reads b.
        ] {
            for flags in 0..4 {
                let case = projection(&[flags | mode, 3, 0, 7, projected])?;
                let package = IrPackage::parse_ir(&case.expected_ir, None)?;
                let main = package.get_function("main")?;
                let mut expected = expected;
                if flags & 2 != 0 {
                    expected.reverse();
                }
                let inputs = [XlsIrValue::make_ubits(4, 3)?, XlsIrValue::make_ubits(4, 5)?];
                let expected = XlsIrValue::make_tuple(&[
                    XlsIrValue::make_ubits(4, expected[0])?,
                    XlsIrValue::make_ubits(4, expected[1])?,
                ]);
                assert_eq!(main.interpret(&inputs)?, expected);
            }
        }
        Ok(())
    }
}
