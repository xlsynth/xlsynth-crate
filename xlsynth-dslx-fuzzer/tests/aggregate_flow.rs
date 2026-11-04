// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::path::Path;
use xlsynth::{DslxConvertOptions, IrPackage, XlsIrValue};
use xlsynth_dslx_fuzzer::aggregate_flow::{AggregateFlow, FlowKind, FlowObservation, FlowStep};
use xlsynth_dslx_fuzzer::ast::BinaryOp;
use xlsynth_pir::{FnBuilder, ir};

/// Builds the same three-argument call through imported DSLX helpers and the
/// scalar PIR reference, allowing independent values for all flow inputs.
fn packages(
    flow: &AggregateFlow,
    last_module: usize,
    width: usize,
) -> Result<(IrPackage, IrPackage), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut imports = String::new();
    for module in 0..=last_module {
        let source = format!("{imports}{}", flow.helper_declarations(module));
        std::fs::write(directory.path().join(format!("m{module}.x")), source)?;
        imports.push_str(&format!("import m{module};\n"));
    }
    let source = format!(
        "{imports}fn main(x: sN[{width}], y: sN[{width}], computed: sN[{width}]) -> {} {{ {} }}",
        flow.output_type(),
        flow.caller_expression(["x", "y", "computed"])
    );
    let options = DslxConvertOptions {
        additional_search_paths: vec![directory.path()],
        ..Default::default()
    };
    let conversion = xlsynth::convert_dslx_to_ir_text(&source, Path::new("sample.x"), &options)?;
    let converted = IrPackage::parse_ir(&conversion.ir, None)?;
    let mut builder = FnBuilder::new("main");
    let args = [
        builder.param("x", ir::Type::Bits(width))?,
        builder.param("y", ir::Type::Bits(width))?,
        builder.param("computed", ir::Type::Bits(width))?,
    ];
    let result = flow.emit_ir(&mut builder, args)?;
    let expected = builder.build_package(result, "expected")?;
    Ok((converted, IrPackage::parse_ir(&expected.to_string(), None)?))
}

#[test]
fn chained_updates_observe_preserved_members_and_runtime_tags() -> Result<(), Box<dyn Error>> {
    let steps = vec![
        FlowStep {
            op: BinaryOp::Add,
            invert_index: false,
            delta: 0,
        },
        FlowStep {
            op: BinaryOp::Xor,
            invert_index: true,
            delta: 1,
        },
        FlowStep {
            op: BinaryOp::Sub,
            invert_index: false,
            delta: 2,
        },
    ];
    let mangled = xlsynth::mangle_dslx_name("sample", "main")?;
    for kind in [FlowKind::Array, FlowKind::Struct, FlowKind::Enum] {
        for observation in [FlowObservation::Whole, FlowObservation::Select] {
            for updater in [0, 1] {
                let flow = AggregateFlow::new(
                    0,
                    2,
                    [0, updater],
                    kind,
                    BinaryOp::Add,
                    steps.clone(),
                    observation,
                )?;
                let (converted, reference) = packages(&flow, updater, 2)?;
                let converted = converted.get_function(&mangled)?;
                let reference = reference.get_function("main")?;
                for x in 0..4 {
                    for y in 0..4 {
                        for z in 0..4 {
                            let mut words = [(z + x) & 3, y];
                            let mut tag = x & 1;
                            for (step, delta) in [x, y, z].into_iter().enumerate() {
                                let mut index = (delta & 1) ^ u64::from(step == 1);
                                if kind == FlowKind::Enum {
                                    index ^= tag;
                                    tag = index;
                                }
                                let slot = &mut words[index as usize];
                                *slot = match step {
                                    0 => (*slot + delta) & 3,
                                    1 => *slot ^ delta,
                                    _ => slot.wrapping_sub(delta) & 3,
                                };
                            }
                            let [left, right] =
                                words.map(|n| XlsIrValue::make_ubits(2, n).unwrap());
                            let expected = match (kind, observation) {
                                (FlowKind::Enum, FlowObservation::Whole) => {
                                    XlsIrValue::make_tuple(&[
                                        XlsIrValue::make_ubits(1, tag)?,
                                        left,
                                        right,
                                    ])
                                }
                                (FlowKind::Struct, FlowObservation::Whole) => {
                                    XlsIrValue::make_tuple(&[left, right])
                                }
                                (FlowKind::Array, FlowObservation::Whole) => {
                                    XlsIrValue::make_array(&[left, right])?
                                }
                                (_, FlowObservation::Select) => {
                                    let selector =
                                        (y & 1) ^ if kind == FlowKind::Enum { tag } else { 0 };
                                    XlsIrValue::make_ubits(2, words[selector as usize])?
                                }
                            };
                            let args = [x, y, z].map(|n| XlsIrValue::make_ubits(2, n).unwrap());
                            assert_eq!(reference.interpret(&args)?, expected);
                            assert_eq!(converted.interpret(&args)?, expected);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn cross_field_write_reads_the_preserved_word_including_bit_64() -> Result<(), Box<dyn Error>> {
    let zero = XlsIrValue::make_ubits(65, 0)?;
    let one = XlsIrValue::make_ubits(65, 1)?;
    let two = XlsIrValue::make_ubits(65, 2)?;
    let high = XlsIrValue::parse_typed("bits[65]:0x10000000000000000")?;
    let high_one = XlsIrValue::parse_typed("bits[65]:0x10000000000000001")?;
    let high_two = XlsIrValue::parse_typed("bits[65]:0x10000000000000002")?;
    let mangled = xlsynth::mangle_dslx_name("sample", "main")?;
    for kind in [FlowKind::Array, FlowKind::Struct, FlowKind::Enum] {
        let flow = AggregateFlow::new(
            0,
            65,
            [0, 1],
            kind,
            BinaryOp::Add,
            vec![
                FlowStep {
                    op: BinaryOp::Or,
                    invert_index: false,
                    delta: 0,
                },
                FlowStep {
                    op: BinaryOp::Xor,
                    invert_index: false,
                    delta: 2,
                },
            ],
            FlowObservation::Whole,
        )?;
        let (converted, reference) = packages(&flow.with_cross_field_update(1)?, 1, 65)?;
        // Each case selects a different destination. Bit 64 in the preserved
        // word must propagate into the updated word across the module call.
        for (args, words, tag) in [
            ([&zero, &high_two, &zero], [&high_two, &high_two], false),
            ([&high, &two, &one], [&high_one, &high_two], true),
        ] {
            let args = args.map(Clone::clone);
            let words = words.map(Clone::clone);
            let expected = match kind {
                FlowKind::Array => XlsIrValue::make_array(&words)?,
                FlowKind::Struct => XlsIrValue::make_tuple(&words),
                FlowKind::Enum => XlsIrValue::make_tuple(&[
                    XlsIrValue::make_ubits(1, u64::from(tag))?,
                    words[0].clone(),
                    words[1].clone(),
                ]),
            };
            assert_eq!(reference.get_function("main")?.interpret(&args)?, expected);
            assert_eq!(
                converted.get_function(&mangled)?.interpret(&args)?,
                expected
            );
        }
    }
    Ok(())
}
