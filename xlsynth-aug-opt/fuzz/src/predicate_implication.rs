// SPDX-License-Identifier: Apache-2.0
//! Focused mutation checks for structural predicate implication.
use xlsynth_aug_opt::ir_cost::IrCost;
use xlsynth_aug_opt::predicate_implication::rewrite_with_evaluator;
use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::prover::types::EquivResult;
use xlsynth_prover::prover::{SolverChoice, prover_for_choice_with_limits};

pub const FAMILIES: [&str; 13] = [
    "bit",
    "subrange",
    "dual",
    "retained",
    "nested",
    "shared",
    "wrong_source",
    "overlap",
    "wrong_polarity",
    "dead",
    "outside",
    "nonzero_literal",
    "same_range",
];

/// Records completed proofs and actual accepted rewrites for generated inputs.
#[derive(Debug)]
pub struct Outcome {
    pub family: usize,
    pub pipeline: usize,
    pub width: usize,
    pub prefix: usize,
    pub forced: usize,
    pub accepted: usize,
    pub proofs: usize,
    pub features: [usize; 9],
}

/// Retains normalized generator features, excluding unused mutation bytes.
struct Sample {
    original: ir::Fn,
    family: usize,
    pipeline: usize,
    width: usize,
    prefix: usize,
    features: [usize; 9],
}

/// Generates contained ranges and explicit non-contained or unrelated controls.
fn sample(data: &[u8]) -> Sample {
    let byte = |i: usize| data.get(i).copied().unwrap_or(0) as usize;
    let family = byte(0) % FAMILIES.len();
    let pipeline = byte(1) % 3;
    let width = [3usize, 4, 8, 17, 65, 129, 161][byte(2) % 7];
    let input_width = width - 1;
    let mut prefix = if family == 0 {
        1
    } else {
        1 + byte(3) % (width / 2)
    };
    let mut start = width - prefix;
    let (guard_start, guard_width) = if matches!(family, 7 | 10) {
        prefix = if family == 7 { 2 } else { 1 };
        start = 0;
        (1, width - 1)
    } else {
        (0, width)
    };
    if family == 12 {
        prefix = width;
        start = 0;
    }
    let op_index = byte(5) % 4;
    let op = ["and", "nand", "or", "nor"][op_index];
    let plain = if op_index < 2 { "and" } else { "or" };
    let zero = (op_index >= 2) ^ (family == 2);
    let witness_zero = zero ^ (family == 8);
    let arithmetic = byte(4) & 1;
    let base_op = if arithmetic == 1 {
        "add(sx, sy, id=6)"
    } else {
        "concat(p, x, id=6)"
    };
    let mut text = format!(
        r#"package implication_fuzz
top fn main(x: bits[{input_width}] id=1, y: bits[{input_width}] id=2, p: bits[1] id=3) -> RET {{
  sx: bits[{width}] = sign_ext(x, new_bit_count={width}, id=4)
  sy: bits[{width}] = sign_ext(y, new_bit_count={width}, id=5)
  base: bits[{width}] = {base_op}
  other: bits[{width}] = xor(sx, sy, id=7)
"#
    );
    if family == 4 {
        let inner_width = width - 1;
        let inner_start = start - 1;
        text += &format!(
            "  middle: bits[{inner_width}] = bit_slice(base, start=1, width={inner_width}, id=8)\n  a: bits[{prefix}] = bit_slice(middle, start={inner_start}, width={prefix}, id=9)\n"
        );
    } else {
        text += &format!(
            "  a: bits[{prefix}] = bit_slice(base, start={start}, width={prefix}, id=9)\n"
        );
    }
    let guard_source = if family == 6 { "other" } else { "base" };
    let literal = usize::from(family == 11);
    let compare = if zero { "eq" } else { "ne" };
    text += &format!(
        "  b: bits[{guard_width}] = bit_slice({guard_source}, start={guard_start}, width={guard_width}, id=10)\n  literal: bits[{guard_width}] = literal(value={literal}, id=11)\n  nz: bits[1] = or_reduce(a, id=12)\n"
    );
    let witness_op = if witness_zero {
        "not(nz, id=13)"
    } else {
        "identity(nz, id=13)"
    };
    text += &format!(
        "  witness: bits[1] = {witness_op}\n  guard: bits[1] = {compare}(literal, b, id=14)\n"
    );
    // Positive witnesses use the original reduction; identity wrappers are not
    // part of the recognizer and should not hide the directed stimulus.
    let witness = if witness_zero { "witness" } else { "nz" };
    // Keep whole-pipeline sharing below the existing priority matcher's
    // multiplicity-expansion limit. Deeper DAGs have separate direct-pass
    // regression/proof coverage; the baseline pipeline also fails them.
    let depth = if family == 5 { 1 + byte(7) % 8 } else { 0 };
    let mut previous = witness.to_string();
    for index in 0..depth {
        let name = format!("shared_{index}");
        text += &format!(
            "  {name}: bits[1] = {plain}({previous}, {previous}, id={})\n",
            20 + index
        );
        previous = name;
    }
    let order = byte(6) & 1;
    let operands = if order == 0 {
        format!("{previous}, guard")
    } else {
        format!("guard, {previous}")
    };
    text += &format!("  result: bits[1] = {op}({operands}, id=100)\n");
    let ret_ty = if family == 3 {
        let ty = format!("(bits[1], bits[{width}], bits[1], bits[1])");
        text += &format!("  ret out: {ty} = tuple(result, base, guard, {witness}, id=101)\n");
        ty
    } else {
        let ret = if family == 9 { "p" } else { "result" };
        text += &format!("  ret out: bits[1] = identity({ret}, id=101)\n");
        "bits[1]".to_string()
    };
    text += "}\n";
    let text = text.replace("RET", &ret_ty);
    let package = Parser::new(&text)
        .parse_and_validate_package()
        .unwrap_or_else(|error| panic!("invalid focused sample: {error}\n{text}"));
    Sample {
        original: package.get_top_fn().unwrap().clone(),
        family,
        pipeline,
        width,
        prefix,
        features: [
            family, pipeline, width, prefix, start, arithmetic, op_index, order, depth,
        ],
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
            "predicate implication equivalence did not complete: {other:?}\n{lhs}\n{rhs}"
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
        features,
    } = sample(data);
    let mut forced = original.clone();
    let mut first = true;
    let count = rewrite_with_evaluator(&mut forced, &mut |_| {
        Ok(IrCost {
            area: if std::mem::take(&mut first) { 100 } else { 99 },
            delay: 100.0,
        })
    });
    if count != usize::from(family < 6 || family == 12) {
        return Err(format!(
            "unexpected forced recognition for {}: {count}\n{original}",
            FAMILIES[family]
        ));
    }
    if count == 0 && !first {
        return Err("no-match input invoked costing".to_string());
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
    let source = format!("package implication_fuzz\n\ntop {original}");
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
        accepted: result.rewrite_stats.predicate_implication_simplified,
        proofs: 2,
        features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_audit_requires_proved_acceptance_and_checked_rejections() {
        let mut accepted = [[0usize; 3]; 13];
        let mut rejected = [0usize; 6];
        for family in 0..13 {
            for pipeline in 0..3 {
                for width in [2, 3, 4, 6] {
                    for op in 0..4 {
                        eprintln!(
                            "audit_case family={family} pipeline={pipeline} width={width} op={op}"
                        );
                        let result =
                            check_input(&[family, pipeline, width, 6, 1, op, 1, 63]).unwrap();
                        assert_eq!(result.proofs, 2);
                        accepted[family as usize][pipeline as usize] += result.accepted;
                        if (6..12).contains(&family) {
                            assert_eq!(result.forced, 0);
                            rejected[family as usize - 6] += 1;
                        }
                    }
                }
            }
        }
        for family in [0, 1, 4, 5] {
            for pipeline in 0..3 {
                assert!(accepted[family][pipeline] > 0, "{accepted:?}");
            }
        }
        assert!(rejected.into_iter().all(|count| count > 0));
        eprintln!("implication_audit accepted={accepted:?} rejected={rejected:?}");
    }
}
