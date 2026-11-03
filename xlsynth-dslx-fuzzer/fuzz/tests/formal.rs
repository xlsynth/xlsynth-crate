// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, ValueRef, Width};
use xlsynth_dslx_fuzzer::tuple_pattern::{BindingContext, TupleElement, TuplePattern};
use xlsynth_dslx_fuzzer::{GenerateOptions, Grammar, Sample, generate_with_grammar};
use xlsynth_dslx_fuzzer_fuzz::prove;
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
