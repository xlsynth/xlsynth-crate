// SPDX-License-Identifier: Apache-2.0

use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_g8r::check_equivalence::check_equivalence_via_toolchain;
use xlsynth_pir::ir_parser::Parser;

const PRIORITY8: &str = include_str!("testdata/priority_result/priority8.ir");
const ZERO: &str = include_str!("testdata/priority_result/zero.ir");

#[test]
fn exclusive_priority_cones_prove_and_sharing_controls_are_respected() {
    let cases = [
        (
            "shared input",
            include_str!("testdata/priority_result/shared-input63.ir"),
            1,
        ),
        ("priority8", PRIORITY8, 1),
        (
            "historical priority",
            include_str!("testdata/priority_result/historical-priority.ir"),
            1,
        ),
        (
            "shared one-hot",
            include_str!("testdata/priority_result/shared-hot63.ir"),
            1,
        ),
        (
            "shared amount",
            include_str!("testdata/priority_result/shared-amount63.ir"),
            0,
        ),
        (
            "shared count",
            include_str!("testdata/priority_result/shared-count63.ir"),
            0,
        ),
        (
            "truncated count",
            include_str!("testdata/priority_result/truncated-count.ir"),
            0,
        ),
        (
            "reflected index",
            include_str!("testdata/priority_result/reflected-index32.ir"),
            1,
        ),
    ];
    // Preserve the directed sharing and truncation shapes while exercising
    // the public optimizer and its whole-function profitability check.
    let options = AugOptOptions {
        enable: true,
        mode: AugOptMode::PirOnly,
        ..Default::default()
    };
    for (name, text, expected_fusions) in cases {
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let top = &package.get_top_fn().unwrap().name;
        let disabled = run_aug_opt_over_ir_text_with_stats(text, Some(top), options).unwrap();
        let enabled = run_aug_opt_over_ir_text_with_stats(
            text,
            Some(top),
            AugOptOptions {
                fuse_priority_results: true,
                ..options
            },
        )
        .unwrap();

        assert_eq!(disabled.rewrite_stats.priority_results_fused, 0, "{name}");
        assert_eq!(
            enabled.rewrite_stats.priority_results_fused, expected_fusions,
            "{name}"
        );
        assert_eq!(
            disabled.output_text != enabled.output_text,
            expected_fusions != 0,
            "{name}"
        );
        // Both full pipelines must preserve the source function. Passing the
        // emitted IR directly also checks that the result uses basis IR.
        check_equivalence_via_toolchain(text, &disabled.output_text)
            .unwrap_or_else(|error| panic!("{name}, fusion disabled: {error}"));
        check_equivalence_via_toolchain(text, &enabled.output_text)
            .unwrap_or_else(|error| panic!("{name}, fusion enabled: {error}"));
    }
}

#[test]
fn zero_width_control_is_unchanged_by_priority_fusion() {
    let options = AugOptOptions {
        enable: true,
        mode: AugOptMode::PirOnly,
        ..Default::default()
    };
    let disabled = run_aug_opt_over_ir_text_with_stats(ZERO, Some("main"), options).unwrap();
    let enabled = run_aug_opt_over_ir_text_with_stats(
        ZERO,
        Some("main"),
        AugOptOptions {
            fuse_priority_results: true,
            ..options
        },
    )
    .unwrap();
    assert_eq!(disabled.rewrite_stats.priority_results_fused, 0);
    assert_eq!(enabled.rewrite_stats.priority_results_fused, 0);
    assert_eq!(disabled.output_text, enabled.output_text);
    check_equivalence_via_toolchain(ZERO, &enabled.output_text).unwrap();
}

#[test]
fn priority_fusion_defaults_off_when_aug_opt_is_enabled() {
    assert!(!AugOptOptions::default().fuse_priority_results);
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
                fuse_priority_results: false,
                ..options
            },
        )
        .unwrap();
        assert_eq!(implicit.rewrite_stats.priority_results_fused, 0, "{mode:?}");
        assert_eq!(explicit.rewrite_stats.priority_results_fused, 0, "{mode:?}");
        assert_eq!(implicit.output_text, explicit.output_text, "{mode:?}");
        assert_eq!(implicit.total_rewrites, explicit.total_rewrites, "{mode:?}");
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
        include_str!("testdata/priority_result/shared-hot63.ir"),
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
