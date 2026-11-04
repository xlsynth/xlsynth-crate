// SPDX-License-Identifier: Apache-2.0

use xlsynth::dslx::{ImportData, parse_and_typecheck};
use xlsynth_dslx_fuzzer::ast::{BitsType, Body, Fragment, ValueRef, Width};
use xlsynth_dslx_fuzzer::compound::CompoundOp;
use xlsynth_dslx_fuzzer::range_match::{NestedCapture, RangeMatch, RangeValue};
use xlsynth_dslx_fuzzer::{GenerateOptions, Grammar, Sample, generate_with_grammar};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};

#[test]
fn both_mission_profiles_generate_signed_ranges_and_independent_widths() -> Result<(), String> {
    for grammar in [Grammar::Mission, Grammar::MissionLocal] {
        // Select the first fragment of the first parametric unsigned helper.
        // A signed two-bit range is independent of the enclosing unsigned lane.
        let indexed = [0, 0, 0, 23, 0, 1, 1, 0, 1, 0, 1, 0];
        let looped = [0, 0, 0, 17, 0, 1, 0, 1, 0, 9, 1, 1, 7];
        for (data, expected) in [
            (
                indexed.as_slice(),
                CompoundOp::RangeValue(RangeValue::typed(true, 2, -2, 2, false)?),
            ),
            (
                looped.as_slice(),
                CompoundOp::RangeLoop {
                    range: RangeValue::typed(true, 65, -1, 2, true)?,
                    constant: false,
                },
            ),
        ] {
            let sample = generate_with_grammar(data, GenerateOptions::default(), grammar)?;
            let [Fragment::Compound { op, .. }] = sample.helpers()[0].body(false).fragments()
            else {
                panic!("expected a single compound fragment");
            };
            assert_eq!(*op, expected);
        }
    }
    Ok(())
}

#[test]
fn runtime_range_matches_preserve_arm_order_and_selector_casts() -> Result<(), String> {
    // Seven-bit lanes exercise sign extension to a wide signed key and
    // truncation to a narrow unsigned key. At overlapping points the first
    // result is 5; points unique to the second arm produce !5 (122).
    for (signed, key_width, start, cases) in [
        (
            true,
            65,
            -3,
            [
                (124, 124),
                (125, 5),
                (126, 5),
                (127, 5),
                (0, 122),
                (1, 122),
                (2, 2),
            ],
        ),
        (
            false,
            5,
            2,
            [
                (1, 1),
                (2, 5),
                (35, 5),
                (4, 5),
                (5, 122),
                (6, 122),
                (39, 39),
            ],
        ),
    ] {
        let ranges = RangeMatch::new([
            RangeValue::typed(signed, key_width, start, 3, !signed)?,
            RangeValue::typed(signed, key_width, start + 1, 4, signed)?,
        ])?;
        let ty = BitsType {
            signed,
            width: Width::Concrete(7),
        };
        let mut lane = Body::new(ty)?;
        lane.push(Fragment::Compound {
            op: CompoundOp::RangeMatch(ranges),
            args: [ValueRef::X, ValueRef::Y, ValueRef::Y],
        })?;
        let sample = Sample::new(vec![], vec![lane])?;
        let reference = sample.expected_ir()?;
        let mut source = sample.sources()?.pop().unwrap().text;
        let lane_type = ty.to_dslx();
        for (input, expected) in cases {
            let args = [
                IrValue::make_ubits(7, input).unwrap(),
                IrValue::make_ubits(7, 5).unwrap(),
            ];
            let FnEvalResult::Success(result) = eval_fn(reference.get_top_fn().unwrap(), &args)
            else {
                panic!("could not evaluate the runtime match reference");
            };
            assert_eq!(
                result.value,
                IrValue::make_tuple(&[IrValue::make_ubits(7, expected).unwrap()])
            );
            source.push_str(&format!(
                "const_assert!(main(u7:{input} as {lane_type}, {lane_type}:5) == ((u7:{expected} as {lane_type}),));\n"
            ));
        }
        parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default())
            .map_err(|error| format!("{error}\n{source}"))?;
    }
    Ok(())
}

#[test]
fn range_elements_and_loop_indices_match_dslx_constant_evaluation() -> Result<(), String> {
    for (signed, induction_width, start, length, inclusive) in [
        (true, 2, -2, 2, false),
        (true, 63, -2, 4, false),
        (true, 64, -2, 4, true),
        (true, 65, -1, 1, false),
        (true, 129, -1, 3, true),
        (false, 1, 0, 2, true),
        (false, 2, 1, 2, false),
        (false, 65, 5, 2, true),
    ] {
        let range = RangeValue::typed(signed, induction_width, start, length, inclusive)?;
        for lane_width in [7, 129] {
            let ty = BitsType {
                signed: !signed,
                width: Width::Concrete(lane_width),
            };
            let mut lanes = Vec::new();
            for op in [
                CompoundOp::RangeValue(range),
                CompoundOp::RangeLoop {
                    range,
                    constant: false,
                },
                CompoundOp::RangeLoop {
                    range,
                    constant: true,
                },
            ] {
                let mut body = Body::new(ty)?;
                body.push(Fragment::Compound {
                    op,
                    args: [ValueRef::X, ValueRef::Y, ValueRef::X],
                })?;
                lanes.push(body);
            }
            let sample = Sample::new(vec![], lanes)?;
            let mut source = sample.sources()?.pop().unwrap().text;
            let reference = sample.expected_ir()?;
            let top = reference.get_top_fn().unwrap();
            let lane_type = ty.to_dslx();
            for x in 0..4i64 {
                let pair = [
                    IrValue::make_ubits(lane_width, x as u64).unwrap(),
                    IrValue::make_ubits(lane_width, 0).unwrap(),
                ];
                let args = pair.iter().cycle().take(6).cloned().collect::<Vec<_>>();
                let FnEvalResult::Success(result) = eval_fn(top, &args) else {
                    panic!("range reference evaluation failed");
                };
                let start = i64::from(start);
                let length = i64::from(length);
                let element = start + (x % length);
                let fold = (start..start + length).fold(x, |acc, i| acc ^ i);
                let expected = [element, fold, fold];
                let expected_tuple = expected.map(|n| IrValue::make_sbits(lane_width, n).unwrap());
                assert_eq!(result.value, IrValue::make_tuple(&expected_tuple));
                let arg_pair = format!("{lane_type}:{x}, {lane_type}:0");
                let result_elements =
                    expected.map(|n| format!("(sN[{lane_width}]:{n} as {lane_type})"));
                source.push_str(&format!(
                    "const_assert!(main({arg_pair}, {arg_pair}, {arg_pair}) == ({},));\n",
                    result_elements.join(", ")
                ));
            }
            // DSLX constexpr evaluation checks the same mathematical answers
            // without invoking the C++ converter under test,
            // including for its known failures.
            parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default())
                .map_err(|error| format!("{error}\n{source}"))?;
        }
    }
    Ok(())
}

#[test]
fn ignored_carries_and_nested_captures_match_dslx_constant_evaluation() -> Result<(), String> {
    for (signed_index, start) in [(true, -2i16), (false, 2i16)] {
        let empty = RangeValue::typed(signed_index, 65, start, 0, false)?;
        let singleton = RangeValue::typed(signed_index, 65, start, 1, true)?;
        let outer = RangeValue::typed(signed_index, 65, start, 3, true)?;
        let inner_start = if signed_index { 1i16 } else { -1i16 };
        let inner = RangeValue::typed(!signed_index, 3, inner_start, 2, false)?;
        for lane_width in [8, 129] {
            let ty = BitsType {
                signed: !signed_index,
                width: Width::Concrete(lane_width),
            };
            let mut lanes = Vec::new();
            for op in [
                CompoundOp::IgnoredCarryLoop(empty),
                CompoundOp::IgnoredCarryLoop(singleton),
                CompoundOp::IgnoredCarryLoop(outer),
                CompoundOp::NestedCapture(NestedCapture::new(outer, inner)?),
            ] {
                let mut lane = Body::new(ty)?;
                lane.push(Fragment::Compound {
                    op,
                    args: [ValueRef::X, ValueRef::Y, ValueRef::X],
                })?;
                lanes.push(lane);
            }
            let sample = Sample::new(vec![], lanes)?;
            let reference = sample.expected_ir()?;
            let mut source = sample.sources()?.pop().unwrap().text;
            let lane_type = ty.to_dslx();
            for (x, y) in [(3i64, 5i64), (0, 2), (11, 0)] {
                let start = i64::from(start);
                let inner_start = i64::from(inner_start);
                let nested = (start..start + 3).fold(x, |acc, i| {
                    (inner_start..inner_start + 2)
                        .fold(acc ^ x, |acc, j| ((acc + 1) ^ (i + y)) + j + 2)
                });
                let expected = [x, y + start, y + start + 2, nested];
                let pair = [x, y].map(|v| IrValue::make_sbits(lane_width, v).unwrap());
                let args: Vec<_> = pair.iter().cycle().take(8).cloned().collect();
                let FnEvalResult::Success(result) = eval_fn(reference.get_top_fn().unwrap(), &args)
                else {
                    panic!("captured loop reference evaluation failed");
                };
                assert_eq!(
                    result.value,
                    IrValue::make_tuple(
                        &expected.map(|v| IrValue::make_sbits(lane_width, v).unwrap())
                    )
                );
                let arg_pair = format!("{lane_type}:{x}, {lane_type}:{y}");
                let results = expected.map(|v| format!("(sN[{lane_width}]:{v} as {lane_type})"));
                source.push_str(&format!(
                    "const_assert!(main({arg_pair}, {arg_pair}, {arg_pair}, {arg_pair}) == ({},));\n",
                    results.join(", ")
                ));
            }
            parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default())
                .map_err(|error| format!("{error}\n{source}"))?;
        }
    }
    Ok(())
}
