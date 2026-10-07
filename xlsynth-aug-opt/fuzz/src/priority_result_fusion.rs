// SPDX-License-Identifier: Apache-2.0

//! Independent equivalence checks and semantic coverage for priority fusion.

use std::collections::BTreeMap;

use xlsynth_aug_opt::ir_cost::IrCost;
use xlsynth_aug_opt::priority_result_fusion::rewrite_with_evaluator;
use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir;
use xlsynth_pir::ir_eval::eval_fn;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_verify::verify_function;
use xlsynth_prover::prover::types::EquivResult;
use xlsynth_prover::prover::{SolverChoice, prover_for_choice_with_limits};

use crate::fuzz_solver_limits;
use crate::priority_result_fusion_sample::{CASE_NAMES, SCALAR_DIRECTIONS, generate_sample};

/// Distinguishes reached shapes, accepted rewrites, and completed proofs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Coverage {
    pub samples: u64,
    pub forced_fusions: u64,
    pub forced_proofs: u64,
    pub real_fusions: u64,
    pub real_proofs: u64,
    pub rejection_checks: u64,
    pub proofs: u64,
    pub inconclusive: u64,
    pub exhaustive_assignments: u64,
    pub zero_input_assignments: u64,
}

impl Coverage {
    fn accumulate(&mut self, other: Self) {
        self.samples += other.samples;
        self.forced_fusions += other.forced_fusions;
        self.forced_proofs += other.forced_proofs;
        self.real_fusions += other.real_fusions;
        self.real_proofs += other.real_proofs;
        self.rejection_checks += other.rejection_checks;
        self.proofs += other.proofs;
        self.inconclusive += other.inconclusive;
        self.exhaustive_assignments += other.exhaustive_assignments;
        self.zero_input_assignments += other.zero_input_assignments;
    }
}

/// A deterministic report keyed by case and live generated feature names.
#[derive(Debug, Default)]
pub struct CoverageReport {
    pub total: Coverage,
    pub cases: BTreeMap<String, Coverage>,
    pub features: BTreeMap<String, Coverage>,
}

impl CoverageReport {
    /// Combines completed samples without discarding their named coverage.
    pub fn accumulate(&mut self, other: Self) {
        self.total.accumulate(other.total);
        for (name, counts) in other.cases {
            self.cases.entry(name).or_default().accumulate(counts);
        }
        for (name, counts) in other.features {
            self.features.entry(name).or_default().accumulate(counts);
        }
    }

    /// Rejects a corpus that lacks exercised rejection paths or proved fusions.
    pub fn validate(&self) -> Result<(), String> {
        for (case, name) in CASE_NAMES.iter().enumerate() {
            let coverage = self
                .cases
                .get(*name)
                .ok_or_else(|| format!("missing case {name}"))?;
            if matches!(case, 4..=9 | 13 | 15 | 25 | 26 | 30..=33 | 36) {
                if coverage.rejection_checks == 0 || coverage.proofs == 0 {
                    return Err(format!(
                        "case {name} lacks a checked rejection and completed proof"
                    ));
                }
            } else if coverage.forced_proofs == 0 {
                return Err(format!("case {name} lacks a proved fusion"));
            }
        }
        for name in ["scalar_group", "scalar_group_dead_user"] {
            if self.cases.get(name).is_none_or(|c| c.real_proofs == 0) {
                return Err(format!(
                    "grouped scalar case {name} lacks a proved real-cost fusion"
                ));
            }
        }
        for feature in [
            "scalar_group_msb",
            "scalar_group_lsb",
            "scalar_group_pir_only",
            "scalar_group_sandwich_1",
            "scalar_group_sandwich_3",
        ] {
            if self
                .features
                .get(feature)
                .is_none_or(|c| c.real_proofs == 0)
            {
                return Err(format!(
                    "grouped scalar feature {feature} lacks a proved real-cost fusion"
                ));
            }
        }
        for name in &CASE_NAMES[16..25] {
            if self.cases.get(*name).is_none_or(|c| c.real_proofs == 0) {
                return Err(format!(
                    "scalar family {name} lacks a proved real-cost fusion"
                ));
            }
        }
        for name in SCALAR_DIRECTIONS.into_iter().flatten() {
            if self
                .features
                .get(&format!("{name}_pir_only"))
                .is_none_or(|c| c.forced_proofs == 0 || c.real_proofs == 0)
            {
                return Err(format!(
                    "scalar family/direction {name} lacks a proved PIR-only real-cost fusion"
                ));
            }
            if self
                .features
                .get(name)
                .is_none_or(|c| c.forced_proofs == 0 || c.real_proofs == 0)
            {
                return Err(format!(
                    "scalar family/direction {name} lacks forced and real-cost proofs"
                ));
            }
        }
        for name in [
            "scalar_sentinel",
            "scalar_wrapping",
            "scalar_wide_constant",
            "scalar_widen_before",
            "scalar_widen_after",
            "scalar_sliced",
            "scalar_full_reduction",
            "scalar_literal_left",
            "scalar_literal_right",
            "scalar_wide_input",
            "scalar_narrow_input",
        ] {
            if self.features.get(name).is_none_or(|c| c.forced_proofs == 0) {
                return Err(format!("scalar feature {name} lacks a proved fusion"));
            }
        }
        if self.total.inconclusive != 0 {
            return Err(format!(
                "{} inconclusive proofs in audited corpus",
                self.total.inconclusive
            ));
        }
        if self.total.zero_input_assignments == 0 {
            return Err("no concrete zero-input sentinel checks".to_string());
        }
        for feature in [
            "msb_priority",
            "lsb_priority",
            "masked",
            "unmasked",
            "narrow_input",
            "wide_input",
            "wide_constant",
            "decode",
            "one_shift",
            "expanded_concat",
            "expanded_xor",
            "add_right",
            "add_left",
            "subtract",
            "reflect",
            "zero_extend",
            "zero_prefix",
            "flattened_concat",
            "split_padding",
            "redundant_leaf_mask",
            "data_dependent_predicate",
            "wrapping",
            "overshift",
            "operation_sequence",
            "pir_only",
            "sandwich_1",
            "sandwich_3",
        ] {
            if self
                .features
                .get(feature)
                .is_none_or(|counts| counts.forced_proofs == 0)
            {
                return Err(format!("feature {feature} lacks a proved fusion"));
            }
        }
        for pipeline in ["pir_only", "sandwich_1", "sandwich_3"] {
            if self
                .features
                .get(pipeline)
                .is_none_or(|counts| counts.real_proofs == 0)
            {
                return Err(format!(
                    "pipeline {pipeline} lacks a proved real-cost fusion"
                ));
            }
        }
        if self.total.real_proofs == 0 {
            return Err("no completed proof exercised fusion with real g8r costing".to_string());
        }
        Ok(())
    }
}

/// Uses direct PIR-to-SMT translation, independently of the mapper's costs.
fn prove(lhs: &ir::Fn, rhs: &ir::Fn) -> EquivResult {
    prover_for_choice_with_limits(SolverChoice::Bitwuzla, None, fuzz_solver_limits())
        .prove_ir_fn_equiv(lhs, rhs)
}

/// Counts solver limits as inconclusive samples, never as successful proofs.
fn check_equivalence(lhs: &ir::Fn, rhs: &ir::Fn, counts: &mut Coverage) -> Result<bool, String> {
    match prove(lhs, rhs) {
        EquivResult::Proved => {
            counts.proofs += 1;
            Ok(true)
        }
        EquivResult::Inconclusive(reason) => {
            // Resource limits do not establish inequivalence; retain their
            // count so a campaign cannot claim them as completed proofs.
            counts.inconclusive += 1;
            log::debug!("priority-fusion proof inconclusive: {reason}");
            Ok(false)
        }
        result => Err(format!(
            "priority-fusion equivalence failed: {result:?}\nBefore:\n{lhs}\nAfter:\n{rhs}"
        )),
    }
}

/// Checks zero sentinels at every width and exhausts small signatures.
fn check_small_inputs(lhs: &ir::Fn, rhs: &ir::Fn, counts: &mut Coverage) -> Result<(), String> {
    let zero_args = lhs
        .param_nodes()
        .map(|node| IrValue::make_ubits(node.ty.bit_count(), 0).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if eval_fn(lhs, &zero_args) != eval_fn(rhs, &zero_args) {
        return Err("zero-input sentinel interpreter mismatch".to_string());
    }
    counts.zero_input_assignments += 1;
    let total_bits: usize = lhs.param_nodes().map(|node| node.ty.bit_count()).sum();
    if total_bits > 8 {
        // Larger signatures are covered by the quantified SMT proof instead
        // of exponential concrete enumeration.
        return Ok(());
    }
    for assignment in 0..(1usize << total_bits) {
        let mut remaining = assignment;
        let args = lhs
            .param_nodes()
            .map(|node| {
                let width = node.ty.bit_count();
                let value = remaining & ((1usize << width) - 1);
                remaining >>= width;
                IrValue::make_ubits(width, value as u64).map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let before = eval_fn(lhs, &args);
        let after = eval_fn(rhs, &args);
        if before != after {
            return Err(format!(
                "priority-fusion interpreter mismatch for {args:?}: {before:?} != {after:?}"
            ));
        }
        counts.exhaustive_assignments += 1;
    }
    Ok(())
}

/// Checks recognition independently of profitability, then the real pipeline.
pub fn check_input(data: &[u8]) -> Result<CoverageReport, String> {
    let sample = generate_sample(data)?;
    let original = &sample.function;
    let mut candidate = original.clone();
    let mut calls = 0;
    let fusions = rewrite_with_evaluator(&mut candidate, &mut |_| {
        calls += 1;
        Ok(IrCost {
            area: if calls == 1 { 100 } else { 99 },
            delay: 100.0,
        })
    });
    if sample.must_reject && fusions != 0 {
        return Err(format!(
            "{} unexpectedly fused\n{original}",
            CASE_NAMES[sample.case]
        ));
    }
    if !sample.must_reject && !sample.may_decline && fusions == 0 {
        return Err(format!(
            "{} never reached forced acceptance\n{original}",
            CASE_NAMES[sample.case]
        ));
    }
    if fusions == 0 && candidate.to_string() != original.to_string() {
        return Err("declined fusion changed the input".to_string());
    }
    verify_function(&candidate).map_err(|error| format!("invalid fusion candidate: {error}"))?;
    let mut counts = Coverage {
        samples: 1,
        forced_fusions: fusions as u64,
        ..Default::default()
    };
    if sample.must_reject {
        counts.rejection_checks += 1;
    }
    if check_equivalence(original, &candidate, &mut counts)? && fusions != 0 {
        counts.forced_proofs += 1;
    }
    check_small_inputs(original, &candidate, &mut counts)?;

    // Ties and evaluator failures must roll back even for recognized shapes.
    for fail in [false, true] {
        let mut rejected = original.clone();
        let result = rewrite_with_evaluator(&mut rejected, &mut |_| {
            if fail {
                Err("injected fuzz cost failure".to_string())
            } else {
                Ok(IrCost {
                    area: 100,
                    delay: 100.0,
                })
            }
        });
        if result != 0 || rejected.to_string() != original.to_string() {
            return Err("fusion leaked a speculative change after a cost tie/error".to_string());
        }
    }

    let source = format!("package priority_fuzz\n\ntop {original}");
    Parser::new(&source)
        .parse_and_validate_package()
        .map_err(|error| format!("cannot parse generated IR: {error}\n{source}"))?;
    for fuse_priority_results in [false, true] {
        let result = run_aug_opt_over_ir_text_with_stats(
            &source,
            Some("main"),
            AugOptOptions {
                enable: true,
                mode: if sample.pipeline == 0 {
                    AugOptMode::PirOnly
                } else {
                    AugOptMode::Sandwich
                },
                rounds: if sample.pipeline == 2 { 3 } else { 1 },
                recover_split_adders: false,
                fuse_priority_results,
            },
        )
        .map_err(|error| format!("priority-fusion pipeline failed: {error}\n{source}"))?;
        if !fuse_priority_results && result.rewrite_stats.priority_results_fused != 0 {
            return Err("disabled priority fusion ran".to_string());
        }
        let package = Parser::new(&result.output_text)
            .parse_and_validate_package()
            .map_err(|error| format!("cannot parse rewritten IR: {error}"))?;
        let rewritten = package
            .get_top_fn()
            .ok_or("rewritten IR has no top function")?;
        let fused = result.rewrite_stats.priority_results_fused as u64;
        counts.real_fusions += fused;
        if check_equivalence(original, rewritten, &mut counts)? && fused != 0 {
            counts.real_proofs += 1;
        }
        check_small_inputs(original, rewritten, &mut counts)?;
    }
    let mut features: BTreeMap<String, Coverage> = sample
        .features
        .iter()
        .map(|name| (name.to_string(), counts))
        .collect();
    if sample.pipeline == 0 && (16..25).contains(&sample.case) {
        for name in SCALAR_DIRECTIONS[sample.case - 16] {
            if sample.features.contains(name) {
                features.insert(format!("{name}_pir_only"), counts);
            }
        }
    }
    Ok(CoverageReport {
        total: counts,
        cases: BTreeMap::from([(CASE_NAMES[sample.case].to_string(), counts)]),
        features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::priority_result_fusion_sample::validation_inputs;
    use xlsynth_pir::ir::{NodePayload, Type};

    #[test]
    fn deterministic_corpus_exercises_required_semantic_cases() {
        let mut coverage = CoverageReport::default();
        for input in validation_inputs() {
            coverage.accumulate(
                check_input(&input).unwrap_or_else(|error| panic!("input={input:?}: {error}")),
            );
        }
        coverage
            .validate()
            .unwrap_or_else(|error| panic!("{error}\n{coverage:#?}"));
        assert_eq!(coverage.total.inconclusive, 0);
        assert!(coverage.total.exhaustive_assignments > 0);
        let real = coverage.cases["scalar_eq"].real_proofs;
        coverage.cases.get_mut("scalar_eq").unwrap().real_proofs = 0;
        assert!(
            coverage.validate().is_err(),
            "forced scalar proofs cannot replace real-cost hits"
        );
        coverage.cases.get_mut("scalar_eq").unwrap().real_proofs = real;
        coverage
            .features
            .get_mut("scalar_eq_lsb")
            .unwrap()
            .real_proofs = 0;
        assert!(
            coverage.validate().is_err(),
            "the other priority direction cannot fill a missing hit"
        );
    }

    #[test]
    fn coverage_audit_rejects_unexercised_and_unproved_cases() {
        assert!(CoverageReport::default().validate().is_err());
        let mut coverage = CoverageReport::default();
        for input in validation_inputs() {
            let sample = generate_sample(&input).unwrap();
            let counts = Coverage {
                samples: 1,
                forced_fusions: 1,
                real_fusions: 1,
                ..Default::default()
            };
            coverage
                .cases
                .insert(CASE_NAMES[sample.case].to_string(), counts);
            for name in sample.features {
                coverage.features.insert(name.to_string(), counts);
            }
        }
        // Visiting every generated label is insufficient without completed
        // proofs of actual accepted rewrites and checked rejection paths.
        assert!(coverage.validate().is_err());
    }

    #[test]
    fn independent_oracle_detects_a_wrong_wiring_destination() {
        let sample = generate_sample(&[0, 0, 0, 0, 0, 1, 0]).unwrap();
        let mut candidate = sample.function.clone();
        let mut first = true;
        assert_eq!(
            rewrite_with_evaluator(&mut candidate, &mut |_| {
                Ok(IrCost {
                    area: if std::mem::take(&mut first) { 100 } else { 99 },
                    delay: 100.0,
                })
            }),
            1
        );
        let node = candidate
            .nodes
            .iter_mut()
            .find(|node| matches!(node.payload, NodePayload::BitSlice { start: 0, .. }))
            .unwrap();
        let NodePayload::BitSlice { start, .. } = &mut node.payload else {
            unreachable!()
        };
        *start = 1;
        assert_eq!(node.ty, Type::Bits(1));
        verify_function(&candidate).unwrap();
        assert!(matches!(
            prove(&sample.function, &candidate),
            EquivResult::Disproved { .. }
        ));
    }
}
