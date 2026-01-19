// SPDX-License-Identifier: Apache-2.0

//! Test helpers for `xlsynth-pir`.
//!
//! These are intentionally deterministic: helpers use fixed RNG seeds so tests
//! are reproducible.

use rand::RngCore;
use rand_pcg::Pcg64Mcg;

use crate::ir;
use crate::ir_eval::{FnEvalResult, eval_fn};
use crate::ir_parser;
use xlsynth::IrValue;

/// Deterministically "quickchecks" equivalence of two PIR functions by
/// evaluating both on a mixture of edge cases and pseudo-random samples.
///
/// Notes:
/// - Only supports unsigned bit-vector parameters up to 64 bits (because
///   `IrValue::make_ubits` takes a `u64`).
/// - Intended for small/fast regression checks in unit tests.
fn quickcheck_fn_equivalence_ubits_le64(
    f0: &ir::Fn,
    f1: &ir::Fn,
    param_widths: &[usize],
    random_samples: usize,
) {
    let mut rng = Pcg64Mcg::new(0);

    let mut samples_run = 0usize;

    let mut run_case = |case: &[u64]| {
        assert_eq!(
            case.len(),
            param_widths.len(),
            "test bug: case arity mismatch"
        );
        let mut args: Vec<IrValue> = Vec::with_capacity(case.len());
        for (&w, &v) in param_widths.iter().zip(case.iter()) {
            assert!(w <= 64, "quickcheck only supports widths <= 64 (got {w})");
            args.push(IrValue::make_ubits(w, v).unwrap());
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

    // Edge cases: all-zeros and all-ones (per width), plus a few simple patterns.
    {
        let zeros: Vec<u64> = vec![0; param_widths.len()];
        run_case(&zeros);

        let ones: Vec<u64> = param_widths
            .iter()
            .map(|&w| if w == 64 { u64::MAX } else { (1u64 << w) - 1 })
            .collect();
        run_case(&ones);

        // Alternating patterns for each arg.
        let alt_a: Vec<u64> = param_widths
            .iter()
            .map(|&w| {
                let v = 0xAAAA_AAAA_AAAA_AAAAu64;
                if w == 64 { v } else { v & ((1u64 << w) - 1) }
            })
            .collect();
        run_case(&alt_a);
        let alt_5: Vec<u64> = param_widths
            .iter()
            .map(|&w| {
                let v = 0x5555_5555_5555_5555u64;
                if w == 64 { v } else { v & ((1u64 << w) - 1) }
            })
            .collect();
        run_case(&alt_5);
    }

    // Deterministic pseudo-random sampling.
    for _ in 0..random_samples {
        let mut case: Vec<u64> = Vec::with_capacity(param_widths.len());
        for &w in param_widths {
            let v = rng.next_u64();
            let masked = if w == 64 { v } else { v & ((1u64 << w) - 1) };
            case.push(masked);
        }
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
