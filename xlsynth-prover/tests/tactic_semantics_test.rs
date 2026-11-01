// SPDX-License-Identifier: Apache-2.0

//! Regressions for preserving the proof contract through tactic obligations.

use std::collections::BTreeSet;

use xlsynth_prover::dslx_tactics::obligations::{
    LecObligation, LecProofOptions, LecSide, ObligationPayload, ProverObligation, SourceFile,
};
use xlsynth_prover::dslx_tactics::plans::build_plan_from_obligations;
use xlsynth_prover::dslx_tactics::tactics::cosliced::{CoslicedTactic, NamedSlice};
use xlsynth_prover::dslx_tactics::tactics::focus::{FocusPair, FocusTactic};
use xlsynth_prover::dslx_tactics::tactics::{IsTactic, Tactic};
use xlsynth_prover::proof_plan::config::{ProverPlan, ProverTask};
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism};

/// Builds an obligation whose helper can be abstracted by either tactic.
fn root_obligation() -> ProverObligation {
    let source = "fn helper(x: u8) -> u8 { x } fn main(x: u8) -> u8 { helper(x) }";
    ProverObligation {
        selector_segment: "root".to_string(),
        description: None,
        payload: ObligationPayload::Lec(LecObligation {
            lhs: LecSide::from_text("main", source),
            rhs: LecSide::from_text("main", source),
            options: LecProofOptions::default(),
        }),
    }
}

/// Encoding "helper:other" as the function would tell the child to use
/// "helper".
#[test]
fn a_function_name_cannot_become_part_of_a_worker_uf_symbol() {
    let mut root = root_obligation();
    let ObligationPayload::Lec(lec) = &mut root.payload else {
        panic!("expected equivalence obligation");
    };
    lec.lhs.uf_map.insert("helper:other".into(), "F".into());
    let result = build_plan_from_obligations(&[("root".into(), root)], None, &[], None, None);
    assert_eq!(
        result.err().as_deref(),
        Some(r#"Invalid DSLX function name in UF mapping: "helper:other""#)
    );
}

/// Provides both concrete APIs, independently of script-level validation.
fn decomposition_tactics() -> [Tactic; 2] {
    let slice = NamedSlice {
        func_name: "slice".to_string(),
        code: SourceFile::Text("fn slice(x: u8) -> u8 { x }".to_string()),
    };
    let composed = NamedSlice {
        func_name: "composed".to_string(),
        code: SourceFile::Text("fn composed(x: u8) -> u8 { slice(x) }".to_string()),
    };
    [
        Tactic::Focus(FocusTactic {
            pairs: vec![FocusPair {
                lhs: "helper".to_string(),
                rhs: "helper".to_string(),
            }],
        }),
        Tactic::Cosliced(CoslicedTactic {
            lhs_slices: vec![slice.clone()],
            rhs_slices: vec![slice],
            lhs_composed: composed.clone(),
            rhs_composed: composed,
        }),
    ]
}

#[test]
fn process_plan_preserves_explicit_proof_options() {
    for semantics in [
        AssertionSemantics::Ignore,
        AssertionSemantics::Never,
        AssertionSemantics::Same,
        AssertionSemantics::Assume,
        AssertionSemantics::Implies,
    ] {
        for assume_enum_in_bound in [false, true] {
            let mut root = root_obligation();
            let ObligationPayload::Lec(lec) = &mut root.payload else {
                panic!("expected equivalence obligation");
            };
            lec.options = LecProofOptions {
                assertion_semantics: semantics,
                assume_enum_in_bound,
                flatten_aggregates: true,
                drop_params: vec!["__token".to_string(), "ignored".to_string()],
                assert_label_filter: Some("^check_[ab]$".to_string()),
                parallelism: EquivParallelism::OutputBits,
            };
            lec.lhs.fixed_implicit_activation = true;
            lec.rhs.fixed_implicit_activation = false;

            let plan =
                build_plan_from_obligations(&[("root".to_string(), root)], None, &[], None, None)
                    .expect("build root-only proof plan");
            let ProverPlan::Group { tasks, .. } = &plan.plan else {
                panic!("expected task group");
            };
            assert_eq!(tasks.len(), 1);
            let ProverPlan::Task {
                task: ProverTask::DslxEquiv { config },
                ..
            } = &tasks[0]
            else {
                panic!("expected equivalence task");
            };
            assert_eq!(config.assertion_semantics, Some(semantics));
            assert_eq!(config.assume_enum_in_bound, Some(assume_enum_in_bound));
            assert_eq!(config.flatten_aggregates, Some(true));
            assert_eq!(
                config.drop_params.as_deref(),
                Some(["__token".to_string(), "ignored".to_string()].as_slice())
            );
            assert_eq!(config.assert_label_filter.as_deref(), Some("^check_[ab]$"));
            assert_eq!(
                config.parallelism_strategy,
                Some(EquivParallelism::OutputBits)
            );
            assert_eq!(config.lhs_fixed_implicit_activation, Some(true));
            assert_eq!(config.rhs_fixed_implicit_activation, Some(false));
        }
    }
}

#[test]
fn concrete_tactics_reject_assertion_sensitive_contracts() {
    for tactic in decomposition_tactics() {
        for semantics in [
            AssertionSemantics::Never,
            AssertionSemantics::Same,
            AssertionSemantics::Assume,
            AssertionSemantics::Implies,
        ] {
            let mut root = root_obligation();
            let ObligationPayload::Lec(lec) = &mut root.payload else {
                panic!("expected equivalence obligation");
            };
            lec.options.assertion_semantics = semantics;
            assert!(
                tactic.apply(&root).is_err(),
                "{} must reject {semantics:?} before abstracting assertions",
                tactic.name()
            );
        }
    }
}

#[test]
fn concrete_tactics_reject_unsupported_signature_transformations() {
    for tactic in decomposition_tactics() {
        for unsupported in [
            "drop_params",
            "flatten_aggregates",
            "lhs_fixed_implicit_activation",
            "rhs_fixed_implicit_activation",
            "output_bits",
            "input_bit_split",
        ] {
            let mut root = root_obligation();
            let ObligationPayload::Lec(lec) = &mut root.payload else {
                panic!("expected equivalence obligation");
            };
            match unsupported {
                "drop_params" => lec.options.drop_params.push("x".to_string()),
                "flatten_aggregates" => lec.options.flatten_aggregates = true,
                "lhs_fixed_implicit_activation" => lec.lhs.fixed_implicit_activation = true,
                "rhs_fixed_implicit_activation" => lec.rhs.fixed_implicit_activation = true,
                "output_bits" => lec.options.parallelism = EquivParallelism::OutputBits,
                "input_bit_split" => lec.options.parallelism = EquivParallelism::InputBitSplit,
                _ => unreachable!(),
            }
            assert!(
                tactic.apply(&root).is_err(),
                "{} must reject {unsupported}",
                tactic.name()
            );
        }
    }
}

#[test]
fn concrete_tactics_remove_enum_bounds_from_every_child_only() {
    for tactic in decomposition_tactics() {
        let mut root = root_obligation();
        let ObligationPayload::Lec(lec) = &mut root.payload else {
            panic!("expected equivalence obligation");
        };
        lec.options.assume_enum_in_bound = true;
        let children = tactic.apply(&root).expect("supported decomposition");
        assert!(!children.is_empty());
        for child in children {
            let ObligationPayload::Lec(lec) = child.payload else {
                panic!("expected equivalence child");
            };
            assert!(
                !lec.options.assume_enum_in_bound,
                "{} child {} must cover internally constructed enum values",
                tactic.name(),
                child.selector_segment
            );
        }
        let ObligationPayload::Lec(lec) = root.payload else {
            panic!("expected equivalence parent");
        };
        assert!(lec.options.assume_enum_in_bound);
    }
}

#[test]
fn self_equivalence_clones_preserve_the_selected_sides_activation() {
    let mut root = root_obligation();
    let ObligationPayload::Lec(lec) = &mut root.payload else {
        panic!("expected equivalence obligation");
    };
    lec.lhs.fixed_implicit_activation = true;
    lec.rhs.fixed_implicit_activation = false;
    lec.options.assertion_semantics = AssertionSemantics::Never;
    lec.options.assume_enum_in_bound = true;

    let lhs_self = lec.clone_lhs_self();
    assert!(lhs_self.lhs.fixed_implicit_activation);
    assert!(lhs_self.rhs.fixed_implicit_activation);
    let rhs_self = lec.clone_rhs_self();
    assert!(!rhs_self.lhs.fixed_implicit_activation);
    assert!(!rhs_self.rhs.fixed_implicit_activation);
    for child in [lhs_self, rhs_self] {
        assert_eq!(child.options.assertion_semantics, AssertionSemantics::Never);
        assert!(child.options.assume_enum_in_bound);
    }
}

#[test]
fn process_plan_preserves_module_basenames_without_collisions() {
    let source_dir = tempfile::tempdir().expect("source directory");
    let lhs_path = source_dir.path().join("lhs").join("module.x");
    let rhs_path = source_dir.path().join("rhs").join("module.x");
    std::fs::create_dir_all(lhs_path.parent().unwrap()).expect("lhs directory");
    std::fs::create_dir_all(rhs_path.parent().unwrap()).expect("rhs directory");
    std::fs::write(&lhs_path, "fn main(x: u8) -> u8 { x }").expect("lhs source");
    std::fs::write(&rhs_path, "fn main(x: u8) -> u8 { x + u8:1 }").expect("rhs source");
    let mut root = root_obligation();
    let ObligationPayload::Lec(lec) = &mut root.payload else {
        panic!("expected equivalence obligation");
    };
    lec.lhs = LecSide::from_path("main", &lhs_path);
    lec.rhs = LecSide::from_path("main", &rhs_path);
    let plan = build_plan_from_obligations(
        &[
            ("first".to_string(), root.clone()),
            ("second".to_string(), root),
        ],
        None,
        &[],
        None,
        None,
    )
    .expect("build independent task files");
    let ProverPlan::Group { tasks, .. } = &plan.plan else {
        panic!("expected task group");
    };
    assert_eq!(tasks.len(), 2);
    let mut paths = BTreeSet::new();
    for task in tasks {
        let ProverPlan::Task {
            task: ProverTask::DslxEquiv { config },
            ..
        } = task
        else {
            panic!("expected equivalence task");
        };
        for (actual, original) in [
            (&config.lhs_dslx_file, &lhs_path),
            (&config.rhs_dslx_file, &rhs_path),
        ] {
            assert_eq!(actual.file_name(), original.file_name());
            assert_eq!(
                std::fs::read_to_string(actual).expect("generated source"),
                std::fs::read_to_string(original).expect("original source")
            );
            assert!(paths.insert(actual.clone()), "task source path collision");
        }
    }
}

#[cfg(feature = "has-bitwuzla")]
mod solver_regressions {
    use std::collections::HashMap;
    use std::path::Path;

    use super::*;
    use xlsynth_prover::dslx_equiv::{DslxEquivRequest, DslxModule, run_dslx_equiv};
    use xlsynth_prover::prover::SolverChoice;
    use xlsynth_prover::prover::types::EquivResult;

    const LHS_SOURCE: &str = r#"
enum E : u3 { A = 0, B = 1 }
fn helper(e: E) -> u32 { match e { E::A => u32:0, _ => u32:1 } }
fn main(e: E) -> u32 { helper(((e as u3) + u3:2) as E) }
"#;
    const RHS_SOURCE: &str = r#"
enum E : u3 { A = 0, B = 1 }
fn helper(e: E) -> u32 { match e { E::B => u32:1, _ => u32:0 } }
fn main(e: E) -> u32 { helper(((e as u3) + u3:2) as E) }
"#;

    /// Solves a typed obligation directly, including its complete proof
    /// contract.
    fn prove(lec: &LecObligation) -> Result<EquivResult, String> {
        let mut modules = Vec::new();
        for (side, module_name) in [(&lec.lhs, "lhs"), (&lec.rhs, "rhs")] {
            let uf_map = side
                .uf_map
                .iter()
                .map(|(name, alias)| {
                    xlsynth::mangle_dslx_name(module_name, name)
                        .map(|name| (name, alias.clone()))
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<HashMap<_, _>, _>>()?;
            let path = if module_name == "lhs" {
                "lhs.x"
            } else {
                "rhs.x"
            };
            modules.push(
                DslxModule::new(&side.file.text, &side.top_func)
                    .with_path(Some(Path::new(path)))
                    .with_owned_uf_map(uf_map)
                    .with_fixed_implicit_activation(side.fixed_implicit_activation),
            );
        }
        let rhs = modules.pop().expect("rhs module");
        let lhs = modules.pop().expect("lhs module");
        let request = DslxEquivRequest::new(lhs, rhs)
            .with_solver(Some(SolverChoice::Bitwuzla))
            .with_assertion_semantics(lec.options.assertion_semantics)
            .with_assume_enum_in_bound(lec.options.assume_enum_in_bound)
            .with_drop_params(&lec.options.drop_params)
            .with_flatten_aggregates(lec.options.flatten_aggregates)
            .with_parallelism(lec.options.parallelism)
            .with_assert_label_filter(lec.options.assert_label_filter.as_deref());
        Ok(run_dslx_equiv(&request)?.result)
    }

    /// Checks both the counterexample and the false proof from restricting
    /// helpers.
    fn check_enum_decomposition(tactic: &impl IsTactic, pair_segment: &str) {
        let root = ProverObligation {
            selector_segment: "root".to_string(),
            description: None,
            payload: ObligationPayload::Lec(LecObligation {
                lhs: LecSide::from_text("main", LHS_SOURCE),
                rhs: LecSide::from_text("main", RHS_SOURCE),
                options: LecProofOptions {
                    assume_enum_in_bound: true,
                    ..LecProofOptions::default()
                },
            }),
        };
        let ObligationPayload::Lec(parent) = &root.payload else {
            panic!("expected equivalence parent");
        };
        assert!(matches!(
            prove(parent).unwrap(),
            EquivResult::Disproved { .. }
        ));
        let children = tactic.apply(&root).expect("enum decomposition");
        let mut checked_pair = false;
        for child in children {
            let ObligationPayload::Lec(lec) = child.payload else {
                panic!("expected equivalence child");
            };
            let result = prove(&lec).expect("solve derived obligation");
            if child.selector_segment == pair_segment {
                checked_pair = true;
                assert!(
                    matches!(result, EquivResult::Disproved { .. }),
                    "{result:?}"
                );
                let mut unsoundly_bounded = lec.clone();
                unsoundly_bounded.options.assume_enum_in_bound = true;
                assert!(matches!(
                    prove(&unsoundly_bounded).unwrap(),
                    EquivResult::Proved
                ));
            } else {
                assert!(matches!(result, EquivResult::Proved), "{result:?}");
            }
        }
        assert!(checked_pair, "missing targeted helper equivalence check");
    }

    #[test]
    fn focus_checks_helper_on_internally_constructed_enum_values() {
        let tactic = FocusTactic {
            pairs: vec![FocusPair {
                lhs: "helper".to_string(),
                rhs: "helper".to_string(),
            }],
        };
        check_enum_decomposition(&tactic, "pair_1");
    }

    #[test]
    fn cosliced_checks_helper_on_internally_constructed_enum_values() {
        let composed = NamedSlice {
            func_name: "composed".to_string(),
            code: SourceFile::Text(
                "fn composed(e: E) -> u32 { slice(((e as u3) + u3:2) as E) }".to_string(),
            ),
        };
        let tactic = CoslicedTactic {
            lhs_slices: vec![NamedSlice {
                func_name: "slice".to_string(),
                code: SourceFile::Text(
                    "fn slice(e: E) -> u32 { match e { E::A => u32:0, _ => u32:1 } }".to_string(),
                ),
            }],
            rhs_slices: vec![NamedSlice {
                func_name: "slice".to_string(),
                code: SourceFile::Text(
                    "fn slice(e: E) -> u32 { match e { E::B => u32:1, _ => u32:0 } }".to_string(),
                ),
            }],
            lhs_composed: composed.clone(),
            rhs_composed: composed,
        };
        check_enum_decomposition(&tactic, "slice_1");
    }
}
