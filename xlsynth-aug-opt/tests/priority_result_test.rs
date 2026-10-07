// SPDX-License-Identifier: Apache-2.0

use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_g8r::check_equivalence::check_equivalence_via_toolchain;
use xlsynth_pir::ir_parser::Parser;

const PRIORITY8: &str = include_str!("fixtures/mffc_regressions/priority_u8.ir");
const ZERO: &str = include_str!("fixtures/mffc_regressions/priority_zero.ir");

#[test]
fn priority_fusion_defaults_on_when_aug_opt_is_enabled() {
    assert!(AugOptOptions::default().fuse_priority_results);
    assert!(!AugOptOptions::default().enable);
    for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
        let options = AugOptOptions {
            enable: true,
            mode,
            ..Default::default()
        };
        let implicit =
            run_aug_opt_over_ir_text_with_stats(PRIORITY8, Some("main"), options).unwrap();
        let explicit = run_aug_opt_over_ir_text_with_stats(
            PRIORITY8,
            Some("main"),
            AugOptOptions {
                fuse_priority_results: true,
                ..options
            },
        )
        .unwrap();
        let disabled = run_aug_opt_over_ir_text_with_stats(
            PRIORITY8,
            Some("main"),
            AugOptOptions {
                fuse_priority_results: false,
                ..options
            },
        )
        .unwrap();
        assert_eq!(implicit.rewrite_stats.priority_results_fused, 1, "{mode:?}");
        assert_eq!(explicit.rewrite_stats.priority_results_fused, 1, "{mode:?}");
        assert_eq!(disabled.rewrite_stats.priority_results_fused, 0, "{mode:?}");
        assert_eq!(implicit.output_text, explicit.output_text, "{mode:?}");
        assert_ne!(implicit.output_text, disabled.output_text, "{mode:?}");
        assert_eq!(implicit.total_rewrites, explicit.total_rewrites, "{mode:?}");
        check_equivalence_via_toolchain(PRIORITY8, &implicit.output_text).unwrap();
    }
}

#[test]
fn disabled_aug_opt_is_an_exact_passthrough_even_with_priority_fusion_requested() {
    for text in [PRIORITY8, ZERO] {
        for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
            for fuse_priority_results in [false, true] {
                let result = run_aug_opt_over_ir_text_with_stats(
                    text,
                    None,
                    AugOptOptions {
                        enable: false,
                        mode,
                        fuse_priority_results,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(result.output_text, text);
                assert_eq!(result.total_rewrites, 0);
                assert_eq!(result.rewrite_stats.total(), 0);
                assert_eq!(result.rewrite_stats.priority_results_fused, 0);
            }
        }
    }
}

#[test]
fn sandwich_counts_initial_fusion_once_and_zero_rounds_skips_it() {
    for text in [
        PRIORITY8,
        include_str!("fixtures/mffc_regressions/priority_shared_hot63.ir"),
    ] {
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let top = &package.get_top_fn().unwrap().name;
        for rounds in [0, 1, 3] {
            let options = AugOptOptions {
                enable: true,
                rounds,
                mode: AugOptMode::Sandwich,
                fuse_priority_results: true,
                ..Default::default()
            };
            let enabled = run_aug_opt_over_ir_text_with_stats(text, Some(top), options).unwrap();
            assert_eq!(
                enabled.rewrite_stats.priority_results_fused,
                usize::from(rounds > 0)
            );
            assert_eq!(enabled.total_rewrites, enabled.rewrite_stats.total());
            check_equivalence_via_toolchain(text, &enabled.output_text).unwrap();
            if rounds == 0 {
                let disabled = run_aug_opt_over_ir_text_with_stats(
                    text,
                    Some(top),
                    AugOptOptions {
                        fuse_priority_results: false,
                        ..options
                    },
                )
                .unwrap();
                assert_eq!(enabled.output_text, disabled.output_text);
                assert_eq!(enabled.total_rewrites, 0);
            }
        }
    }
}
