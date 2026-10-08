// SPDX-License-Identifier: Apache-2.0
//! Focused mutation checks for low-prefix nonzero simplification.
use xlsynth_aug_opt::ir_cost::IrCost;
use xlsynth_aug_opt::low_bit_nonzero::rewrite_with_evaluator;
use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::EquivResult;
use xlsynth_prover::prover::{SolverChoice, prover_for_choice_with_limits};

pub const FAMILIES: [&str; 9] = [
    "neg",
    "select",
    "select_reversed",
    "decomposed",
    "retained",
    "offset",
    "gap",
    "not",
    "dead",
];

/// Records accepted rewrites only after their equivalence proof completes.
#[derive(Debug)]
pub struct Outcome {
    pub family: usize,
    pub pipeline: usize,
    pub width: usize,
    pub prefix: usize,
    pub forced: usize,
    pub accepted: usize,
    pub proofs: usize,
}

/// Holds one generated function and its mutation-derived feature labels.
struct Sample {
    original: ir::Fn,
    family: usize,
    pipeline: usize,
    width: usize,
    prefix: usize,
}

/// Generates bounded valid IR with contiguous-prefix and rejection controls.
fn sample(data: &[u8]) -> Sample {
    let byte = |i: usize| data.get(i).copied().unwrap_or(0) as usize;
    let family = byte(0) % FAMILIES.len();
    let pipeline = byte(1) % 3;
    let width = [1usize, 2, 3, 8, 17, 31, 65, 129, 161][byte(2) % 9].max(if family == 3 {
        2
    } else if family == 5 || family == 6 {
        3
    } else {
        1
    });
    let prefix = (1 + byte(3) % width).max(if family == 3 { 2 } else { 1 });
    let source_op = if byte(4) & 1 == 0 {
        "identity(x, id=4)"
    } else {
        "add(x, y, id=4)"
    };
    let neg_op = if family == 7 { "not" } else { "neg" };
    let choice = match family {
        0 => "identity(n, id=6)",
        2 => "sel(p, cases=[n, base], id=6)",
        _ => "sel(p, cases=[base, n], id=6)",
    };
    let mut text = format!(
        "package low_bit_fuzz\n\ntop fn main(x: bits[{width}] id=1, y: bits[{width}] id=2, p: bits[1] id=3) -> RET {{\n  base: bits[{width}] = {source_op}\n  n: bits[{width}] = {neg_op}(base, id=5)\n  v: bits[{width}] = {choice}\n"
    );
    let operand = if family == 0 { "n" } else { "v" };
    if family == 3 || family == 6 {
        let start = if family == 6 { 2 } else { 1 };
        let tail_width = if family == 6 { 1 } else { prefix - 1 };
        let order = if byte(5) & 1 == 0 { "a, br" } else { "br, a" };
        text += &format!(
            "  a: bits[1] = bit_slice({operand}, start=0, width=1, id=7)\n  b: bits[{tail_width}] = bit_slice({operand}, start={start}, width={tail_width}, id=8)\n  br: bits[1] = or_reduce(b, id=9)\n  result: bits[1] = or({order}, id=10)\n"
        );
    } else {
        let start = usize::from(family == 5);
        let size = prefix.min(width - start);
        text += &format!(
            "  lo: bits[{size}] = bit_slice({operand}, start={start}, width={size}, id=7)\n  result: bits[1] = or_reduce(lo, id=10)\n"
        );
    }
    let ret_ty;
    if family == 4 {
        ret_ty = format!("(bits[1], bits[{width}])");
        text += &format!("  ret out: {ret_ty} = tuple(result, v, id=11)\n");
    } else {
        ret_ty = "bits[1]".to_string();
        let ret = if family == 8 { "p" } else { "result" };
        text += &format!("  ret out: bits[1] = identity({ret}, id=11)\n");
    }
    text += "}\n";
    text = text.replace("RET", &ret_ty);
    let package = Parser::new(&text)
        .parse_and_validate_package()
        .unwrap_or_else(|error| panic!("invalid focused sample: {error}\n{text}"));
    Sample {
        original: package.get_top_fn().unwrap().clone(),
        family,
        pipeline,
        width,
        prefix,
    }
}

/// Requires a completed Bitwuzla proof before recording coverage.
fn prove(lhs: &ir::Fn, rhs: &ir::Fn) -> Result<(), String> {
    match prover_for_choice_with_limits(SolverChoice::Bitwuzla, None, crate::fuzz_solver_limits())
        .prove_ir_fn_equiv(lhs, rhs)
    {
        EquivResult::Proved => Ok(()),
        // Inconclusive proofs fail this bounded audit; they never count as coverage.
        other => Err(format!(
            "low-bit equivalence did not complete: {other:?}\n{lhs}\n{rhs}"
        )),
    }
}

/// Checks forced soundness, rollback, and real-cost pipeline acceptance.
pub fn check_input(data: &[u8]) -> Result<Outcome, String> {
    let Sample {
        original,
        family,
        pipeline,
        width,
        prefix,
    } = sample(data);
    let mut forced = original.clone();
    let mut first = true;
    let count = rewrite_with_evaluator(&mut forced, &mut |_| {
        Ok(IrCost {
            area: if std::mem::take(&mut first) { 100 } else { 99 },
            delay: 100.0,
        })
    });
    if (family < 5) != (count > 0) {
        return Err(format!(
            "unexpected forced recognition for {}: {count}\n{original}",
            FAMILIES[family]
        ));
    }
    prove(&original, &forced)?;
    for fail in [false, true] {
        let mut declined = original.clone();
        let count = rewrite_with_evaluator(&mut declined, &mut |_| {
            if fail {
                Err("injected cost error".to_string())
            } else {
                Ok(IrCost {
                    area: 100,
                    delay: 100.0,
                })
            }
        });
        if count != 0 || declined.to_string() != original.to_string() {
            return Err("cost tie/error changed input".to_string());
        }
    }
    let source = format!("package low_bit_fuzz\n\ntop {original}");
    let result = run_aug_opt_over_ir_text_with_stats(
        &source,
        Some("main"),
        AugOptOptions {
            enable: true,
            mode: if pipeline == 0 {
                AugOptMode::PirOnly
            } else {
                AugOptMode::Sandwich
            },
            rounds: if pipeline == 2 { 3 } else { 1 },
            ..Default::default()
        },
    )?;
    let package = Parser::new(&result.output_text)
        .parse_and_validate_package()
        .map_err(|error| format!("invalid optimized sample: {error}"))?;
    prove(
        &original,
        package.get_top_fn().ok_or("missing optimized top")?,
    )?;
    Ok(Outcome {
        family,
        pipeline,
        width,
        prefix,
        forced: count,
        accepted: result.rewrite_stats.low_bit_nonzero_simplified,
        proofs: 2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_audit_requires_proved_acceptance_and_checked_rejections() {
        let mut accepted = [[0usize; 3]; 5];
        let mut rejected = [0usize; 4];
        for family in 0..9 {
            for pipeline in 0..3 {
                for width in [2, 3, 6, 8] {
                    for prefix in [0, 1, 6, 127] {
                        let result = check_input(&[family, pipeline, width, prefix, 1, 1]).unwrap();
                        assert_eq!(result.proofs, 2);
                        if family < 5 {
                            accepted[family as usize][pipeline as usize] += result.accepted;
                        } else {
                            assert_eq!(result.forced, 0);
                            rejected[family as usize - 5] += 1;
                        }
                    }
                }
            }
        }
        for family in [1, 2, 3] {
            for pipeline in 0..3 {
                assert!(accepted[family][pipeline] > 0, "{accepted:?}");
            }
        }
        assert!(rejected.into_iter().all(|count| count > 0));
        eprintln!("low_bit_audit accepted={accepted:?} rejected={rejected:?}");
    }
}
