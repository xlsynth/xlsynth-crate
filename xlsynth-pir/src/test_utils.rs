// SPDX-License-Identifier: Apache-2.0

//! Test helpers for `xlsynth-pir`.
//!
//! These are intentionally deterministic: helpers use fixed RNG seeds so tests
//! are reproducible.

use rand_pcg::Pcg64Mcg;

use crate::ir;
use crate::ir_eval::{FnEvalResult, eval_fn};
use crate::ir_parser;
use crate::random_inputs::{
    BitValuePattern, generate_biased_irbits_with_rng, generate_pattern_irbits,
};
use xlsynth::{IrBits, IrValue};

/// Deterministically "quickchecks" equivalence of two PIR functions by
/// evaluating both on a mixture of edge cases and pseudo-random samples.
///
/// Notes:
/// - Intended for small/fast regression checks in unit tests.
fn quickcheck_fn_equivalence_ubits_le64(
    f0: &ir::Fn,
    f1: &ir::Fn,
    param_widths: &[usize],
    random_samples: usize,
) {
    let mut rng = Pcg64Mcg::new(0);

    let mut samples_run = 0usize;

    let mut run_case = |case: &[IrBits]| {
        assert_eq!(
            case.len(),
            param_widths.len(),
            "test bug: case arity mismatch"
        );
        let mut args: Vec<IrValue> = Vec::with_capacity(case.len());
        for (&w, bits) in param_widths.iter().zip(case.iter()) {
            assert_eq!(bits.get_bit_count(), w, "test bug: case width mismatch");
            args.push(IrValue::from_bits(bits));
        }

        let got0 = match eval_fn(f0, &args) {
            FnEvalResult::Success(s) => s.value.clone(),
            FnEvalResult::Failure(e) => panic!("unexpected eval failure (lhs): {:?}", e),
        };
        let got1 = match eval_fn(f1, &args) {
            FnEvalResult::Success(s) => s.value.clone(),
            FnEvalResult::Failure(e) => panic!("unexpected eval failure (rhs): {:?}", e),
        };
        assert_eq!(got0, got1, "mismatch on args={args:?}");
        samples_run += 1;
    };

    for pattern in [
        BitValuePattern::Zero,
        BitValuePattern::AllOnes,
        BitValuePattern::Alternating { lsb_is_one: true },
        BitValuePattern::Alternating { lsb_is_one: false },
    ] {
        let case: Vec<IrBits> = param_widths
            .iter()
            .map(|width| generate_pattern_irbits(*width, pattern))
            .collect();
        run_case(&case);
    }

    // Deterministic corner-biased pseudo-random sampling.
    for _ in 0..random_samples {
        let case: Vec<IrBits> = param_widths
            .iter()
            .map(|width| generate_biased_irbits_with_rng(&mut rng, *width))
            .collect();
        run_case(&case);
    }

    assert!(samples_run >= 1, "test bug: quickcheck ran zero samples");
}

/// Deterministically "quickchecks" equivalence of two functions (by name) from
/// two IR texts.
///
/// This helper owns parsing/validation so callers can stay concise.
pub fn quickcheck_ir_text_fn_equivalence_ubits_le64(
    ir_text_0: &str,
    ir_text_1: &str,
    fn_name: &str,
    param_widths: &[usize],
    random_samples: usize,
) {
    let mut p0 = ir_parser::Parser::new(ir_text_0);
    let pkg0 = p0.parse_and_validate_package().expect("parse/validate lhs");
    let f0 = pkg0.get_fn(fn_name).expect("lhs missing function");

    let mut p1 = ir_parser::Parser::new(ir_text_1);
    let pkg1 = p1.parse_and_validate_package().expect("parse/validate rhs");
    let f1 = pkg1.get_fn(fn_name).expect("rhs missing function");

    quickcheck_fn_equivalence_ubits_le64(f0, f1, param_widths, random_samples);
}
