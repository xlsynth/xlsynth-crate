// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use xlsynth_dslx_fuzzer::ast::{BinaryOp, BitsType, Body, Fragment, ValueRef, Width};
use xlsynth_dslx_fuzzer::tuple_pattern::{BindingContext, TupleElement, TuplePattern};
use xlsynth_dslx_fuzzer::{GenerateOptions, Sample, generate_with_options};
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
        let sample = generate_with_options(&bytes, GenerateOptions { max_width: 8 })
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
