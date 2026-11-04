// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::path::Path;
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, ValueRef, Width};
use xlsynth_dslx_fuzzer::compound::CompoundOp;
use xlsynth_dslx_fuzzer::mission_expr::ReceiverMap;
use xlsynth_dslx_fuzzer::range_match::GuardedMatch;
use xlsynth_dslx_fuzzer::tuple_pattern::{BindingContext, TupleElement, TuplePattern};
use xlsynth_dslx_fuzzer::{GenerateOptions, Grammar, Sample, generate_with_grammar};
use xlsynth_dslx_fuzzer_fuzz::{check_prepared, prove};
use xlsynth_pir::{FnBuilder, ir};
use xlsynth_prover::prover::types::EquivResult;

#[test]
fn generated_samples_prove() {
    let mut rng = StdRng::from_seed([9; 32]);
    for _ in 0..32 {
        let mut bytes = vec![0; 768];
        rng.fill_bytes(&mut bytes);
        // Small widths keep multiplication/division proofs within the routine
        // solver budget; the fuzz target retains its full 129-bit range.
        // Preserve the historical differential corpus. Newly generated
        // signed ranges can abort the stock converter and are exercised by
        // the fuzz campaign and the dedicated range reference tests.
        let sample =
            generate_with_grammar(&bytes, GenerateOptions { max_width: 8 }, Grammar::MissionV1)
                .unwrap()
                .prepare()
                .unwrap();
        let result = prove(&sample).unwrap();
        assert!(
            matches!(result, EquivResult::Proved),
            "{result:#?}\n{sample:#?}"
        );
    }
}

#[test]
fn versioned_mission_productions_prove_and_preserve_previous_decoders() -> Result<(), String> {
    // Generate one parametric helper production. The top calls it with two
    // independent seven-bit arguments so its result participates in the proof.
    let cases: &[(u8, &[u8])] = &[
        (224, &[0, 1, 64, 4, 3, 1]),      // nested enum inequality
        (225, &[0, 1]),                   // bits to array
        (225, &[1, 1]),                   // array to bits
        (226, &[1, 7, 7, 3, 1, 2, 0, 1]), // overlapping signed ranges
        (227, &[0, 1, 1, 2]),             // rectangular array update
        (200, &[0, 0, 2]),                // one-hot select over an updated array
        (200, &[1, 1, 1]),                // priority select over an ellipsis array
        (201, &[128, 64, 1]),             // match a struct and an array of enums
        (202, &[2, 2, 1, 0]),             // nested folds with a tuple carry
    ];
    for &(draw, options) in cases {
        let mut bytes = vec![0, 0, 0, draw, 0, 1];
        bytes.extend_from_slice(options);
        bytes.extend_from_slice(&[
            0, // third fragment operand
            0, 0, 0, 0, 0, // signed helper and default choice
            0, 0, 0, 0, 2, 0, // observation, unsigned seven-bit lane and literal
            0, 0, 0, 2, 0, // lane computes x + zero
            0, 1, 0, // call unsigned helper with x + zero and y
        ]);
        let versions = if draw >= 224 {
            [
                (Grammar::Mission, Grammar::MissionV2),
                (Grammar::MissionLocal, Grammar::MissionLocalV2),
            ]
        } else {
            [
                (Grammar::MissionV4, Grammar::Mission),
                (Grammar::MissionLocalV4, Grammar::MissionLocal),
            ]
        };
        for (current, previous) in versions {
            let sample = generate_with_grammar(&bytes, GenerateOptions::default(), current)?;
            let [Fragment::Compound { op, .. }] = sample.helpers()[0].body(false).fragments()
            else {
                panic!("expected a compound production in {current:?}");
            };
            assert!(matches!(
                (draw, op),
                (224, CompoundOp::SumComparison(_))
                    | (225, CompoundOp::ArrayCast(_))
                    | (226, CompoundOp::RangeMatch(_))
                    | (227, CompoundOp::MatrixUpdate(_))
                    | (200, CompoundOp::ArraySelect(_))
                    | (201, CompoundOp::AggregateMatch(_))
                    | (202, CompoundOp::NestedFold(_))
            ));
            assert!(matches!(prove(&sample.prepare()?)?, EquivResult::Proved));
            let replay = generate_with_grammar(&bytes, GenerateOptions::default(), previous)?;
            assert_ne!(
                sample.helpers()[0].body(false),
                replay.helpers()[0].body(false)
            );
        }
    }
    Ok(())
}

#[test]
fn v5_helpers_and_imported_constants_compose_and_prove() -> Result<(), String> {
    for (selector, options) in [
        (176, &[2, 2][..]),    // map a bound method over four elements
        (177, &[2, 1, 1][..]), // two runtime comparisons, logical AND
        (177, &[4, 3, 0][..]), // logical OR
        (178, &[0][..]),       // preserve width
        (178, &[4][..]),       // widen by four bits
    ] {
        let mut bytes = vec![0, 0, 0, selector, 0, 1];
        bytes.extend_from_slice(options);
        bytes.extend_from_slice(&[
            1, 0, 0, 0, 0, 0, // third operand, signed helper, no parametric default
            6, 13, 9, 0, 2, // seven-bit exported constant chain
            0, 0, 0, 0, 2, 0,
            0, // observation, unsigned seven-bit lane, literal, one fragment
            179, 0, 1, 0, 2, 0, 0, // combine x with two qualified constants from m0
            0, 1, 0, // call the unsigned helper with the result and y
        ]);
        for (grammar, previous) in [
            (Grammar::MissionV5, Grammar::MissionV4),
            (Grammar::MissionLocalV5, Grammar::MissionLocalV4),
        ] {
            let sample = generate_with_grammar(&bytes, GenerateOptions::default(), grammar)?;
            let [Fragment::Compound { op, .. }] = sample.helpers()[0].body(false).fragments()
            else {
                panic!("expected a v5 helper production");
            };
            assert!(matches!(
                (selector, op),
                (176, CompoundOp::ReceiverMap(_))
                    | (177, CompoundOp::LogicalPredicates(_))
                    | (178, CompoundOp::CheckedCast(_))
            ));
            assert!(matches!(
                sample.lanes()[0].fragments(),
                [
                    Fragment::Literal(_),
                    Fragment::ImportedConstant { module: 0, .. },
                    Fragment::Call { .. }
                ]
            ));
            assert!(matches!(prove(&sample.prepare()?)?, EquivResult::Proved));
            let replay = generate_with_grammar(&bytes, GenerateOptions::default(), previous)?;
            assert_ne!(
                sample.helpers()[0].body(false),
                replay.helpers()[0].body(false)
            );
        }
    }
    Ok(())
}

#[test]
fn v6_runtime_patterns_and_casts_compose_with_imported_struct_updates() -> Result<(), String> {
    for (selector, options) in [
        (152, &[0, 2, 3, 5, 7][..]), // named tuple pattern
        (152, &[1, 3, 2, 9, 2][..]), // named enum pattern
        (153, &[0][..]),             // width-preserving cast
        (153, &[5][..]),             // widen by 64 bits
    ] {
        let mut bytes = vec![0, 0, 0, selector, 0, 1];
        bytes.extend_from_slice(options);
        bytes.extend_from_slice(&[
            1, 0, 0, 0, 0, 0, // third operand, signed helper, no parametric default
            6, 13, 9, 0, 2, 6, 8, // exported constants and seven- and nine-bit struct fields
            0, 0, 0, 0, 2, 0,
            0, // observation, unsigned seven-bit lane, literal, one fragment
            154, 0, 1, 0, 1, 1, 2,
            0, // update the imported struct's high field, then xor its fields
            0, 1, 0, // feed the projected result and y into the unsigned helper
        ]);
        for grammar in [Grammar::MissionV6, Grammar::MissionLocalV6] {
            let sample = generate_with_grammar(&bytes, GenerateOptions::default(), grammar)?;
            let [Fragment::Compound { op, .. }] = sample.helpers()[0].body(false).fragments()
            else {
                panic!("expected a v6 helper production");
            };
            assert!(matches!(
                (selector, op),
                (152, CompoundOp::NamedAggregateMatch(_)) | (153, CompoundOp::WideningCast(_))
            ));
            assert!(matches!(
                sample.lanes()[0].fragments()[1],
                Fragment::ImportedStructUpdate { module: 0, .. }
            ));
            assert!(matches!(prove(&sample.prepare()?)?, EquivResult::Proved));
        }
    }
    Ok(())
}

#[test]
fn mission_typed_range_loops_prove() -> Result<(), String> {
    // A seven-bit lane computes x^y, slices an array, and passes both values
    // to a two-trip 65-bit signed loop starting at +3. The lane returns the
    // loop's result and the independently observable xor.
    let seed = [
        0, 0, 0, 17, 0, 1, 0, 1, 0, 9, 1, 1, 11, 0, // unsigned helper
        0, 0, 0, 0, 0, // signed helper and no parametric default
        0, 0, 0, 0, 2, // observation and lane width selection
        85, 1, // literal and two runtime fragments
        4, 0, 1, 0, // xor(x, y)
        14, 3, 1, 2, 0, 0, // array slice using the xor, y and x
        0, 3, 0, // call the loop helper with the slice and xor
    ];
    for grammar in [Grammar::Mission, Grammar::MissionLocal] {
        let sample =
            generate_with_grammar(&seed, GenerateOptions::default(), grammar)?.prepare()?;
        assert!(matches!(prove(&sample)?, EquivResult::Proved));
    }
    Ok(())
}

#[test]
fn v10_rejects_an_assertion_failure_with_equal_return_values() -> Result<(), String> {
    let mut lane = Body::new(BitsType {
        signed: false,
        width: Width::Concrete(8),
    })?;
    // With another function in the package, a private fallible main can lose
    // its ordinary entry wrapper when the converter processes the whole module.
    let mapped = lane.push(Fragment::Compound {
        op: CompoundOp::ReceiverMap(ReceiverMap::new(7, 2)?),
        args: [ValueRef::X, ValueRef::Y, ValueRef::Y],
    })?;
    lane.push(Fragment::Compound {
        op: CompoundOp::GuardedMatch(GuardedMatch::from_draws(3, 37)?),
        args: [ValueRef::X, mapped, ValueRef::Y],
    })?;
    let mut sample = Sample::new(vec![], vec![lane])?.prepare()?;
    for grammar in [Grammar::MissionV10, Grammar::MissionLocalV10] {
        assert!(matches!(
            check_prepared(&sample, grammar)?,
            EquivResult::Proved
        ));
    }
    // Exclude key 3 from the second arm's assertion while retaining its
    // pattern and value. The old equality-only proof still succeeds.
    let source = &sample.modules.last().unwrap().text;
    let faulty = source.replace("k >= u4:3 &&", "k > u4:3 &&");
    assert_ne!(*source, faulty);
    sample.converted_ir =
        xlsynth::convert_dslx_to_ir_text(&faulty, Path::new("sample.x"), &Default::default())
            .map_err(|error| error.to_string())?
            .ir;
    assert!(matches!(
        check_prepared(&sample, Grammar::MissionV9)?,
        EquivResult::Proved
    ));
    for grammar in [Grammar::MissionV10, Grammar::MissionLocalV10] {
        let failure =
            check_prepared(&sample, grammar).expect_err("v10 must detect the false assertion");
        assert!(failure.contains("Disproved"), "{failure}");
    }
    Ok(())
}

#[test]
fn wrong_reference_is_disproved() {
    let mut body = Body::new(BitsType {
        signed: false,
        width: Width::Concrete(65),
    })
    .unwrap();
    body.push(Fragment::Binary {
        op: BinaryOp::Add,
        lhs: ValueRef::X,
        rhs: ValueRef::Y,
    })
    .unwrap();
    let mut sample = Sample::new(vec![], vec![body]).unwrap().prepare().unwrap();
    assert!(matches!(prove(&sample).unwrap(), EquivResult::Proved));
    let mut builder = FnBuilder::new("main");
    let x = builder.param("x", ir::Type::Bits(65)).unwrap();
    let y = builder.param("y", ir::Type::Bits(65)).unwrap();
    let wrong = builder.sub(x, y).unwrap();
    let tuple = builder.tuple(&[wrong]).unwrap();
    sample.expected_ir = builder.build_package(tuple, "wrong").unwrap().to_string();
    assert!(matches!(
        prove(&sample).unwrap(),
        EquivResult::Disproved { .. }
    ));
}

#[test]
fn nested_tuple_rest_bindings_prove_at_wide_widths() {
    for signed in [false, true] {
        let mut lanes = Vec::new();
        for context in BindingContext::ALL {
            for rest in [0..1, 1..3, 2..3, 1..1, 0..3] {
                let nested = TuplePattern::new(
                    vec![TupleElement::Bind(1), TupleElement::Bind(0)],
                    Some(0..1),
                )
                .unwrap();
                let pattern = TuplePattern::new(
                    vec![
                        TupleElement::Bind(0),
                        TupleElement::Tuple(nested),
                        TupleElement::Bind(1),
                    ],
                    Some(rest),
                )
                .unwrap();
                let mut body = Body::new(BitsType {
                    signed,
                    width: Width::Concrete(65),
                })
                .unwrap();
                body.push(Fragment::TuplePattern {
                    pattern,
                    context,
                    args: [ValueRef::X, ValueRef::Y, ValueRef::X],
                })
                .unwrap();
                lanes.push(body);
            }
        }
        let sample = Sample::new(vec![], lanes).unwrap().prepare().unwrap();
        assert!(matches!(prove(&sample).unwrap(), EquivResult::Proved));
    }
}
