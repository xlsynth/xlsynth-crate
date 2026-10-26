// SPDX-License-Identifier: Apache-2.0

use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_g8r::aig::gate::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig::graph_logical_effort::{
    GraphLogicalEffortOptions, score_graph_logical_effort,
};
use xlsynth_g8r::aig_sim::gate_sim;
use xlsynth_g8r::check_equivalence::check_equivalence_via_toolchain;
use xlsynth_g8r::process_ir_path::{
    CanonicalG8rOptions, canonical_ir_text_to_g8r_lowering_artifacts,
};
use xlsynth_pir::IrBits;

const INPUT: &str = include_str!("fixtures/priority_count_bit.ir");

/// Uses the normal native preparation and AIG passes, before FRAIG or ABC.
fn native_gates(text: &str) -> GateFn {
    canonical_ir_text_to_g8r_lowering_artifacts(
        text,
        Some("main"),
        &CanonicalG8rOptions {
            fraig: false,
            ..Default::default()
        },
    )
    .expect("canonical native lowering")
    .gate_fn
}

#[test]
fn priority_count_bit_improves_beyond_normal_xls_optimization() {
    let package = xlsynth::IrPackage::parse_ir(INPUT, None).unwrap();
    let optimized = xlsynth::optimize_ir(&package, "main").unwrap();
    let optimized_again = xlsynth::optimize_ir(&optimized, "main").unwrap();
    assert_eq!(optimized.to_string(), optimized_again.to_string());
    let baseline = native_gates(&optimized.to_string());
    let before = get_aig_stats(&baseline);
    // The opportunity remains after ordinary XLS opt and native AIG passes.
    assert!(before.and_nodes > 200, "{before:?}");
    for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
        let options = AugOptOptions {
            enable: true,
            mode,
            ..Default::default()
        };
        let enabled =
            run_aug_opt_over_ir_text_with_stats(&optimized.to_string(), Some("main"), options)
                .unwrap();
        let disabled = run_aug_opt_over_ir_text_with_stats(
            &optimized.to_string(),
            Some("main"),
            AugOptOptions {
                fuse_priority_results: false,
                ..options
            },
        )
        .unwrap();
        assert_eq!(enabled.rewrite_stats.priority_results_fused, 1, "{mode:?}");
        assert_eq!(disabled.rewrite_stats.priority_results_fused, 0, "{mode:?}");
        assert_eq!(
            get_aig_stats(&native_gates(&disabled.output_text)).and_nodes,
            before.and_nodes
        );
        check_equivalence_via_toolchain(INPUT, &enabled.output_text).unwrap();

        let gates = native_gates(&enabled.output_text);
        let after = get_aig_stats(&gates);
        assert!(after.and_nodes <= 45, "{mode:?}: {after:?}");
        assert!(after.max_depth < before.max_depth, "{mode:?}");
        let le = GraphLogicalEffortOptions {
            beta1: 1.0,
            beta2: 0.0,
        };
        assert!(
            score_graph_logical_effort(&gates, &le).delay
                < score_graph_logical_effort(&baseline, &le).delay,
            "{mode:?}"
        );
        for value in [0, 1, 2, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_ffff] {
            let result = gate_sim::eval(
                &gates,
                &[IrBits::make_ubits(32, value).unwrap()],
                gate_sim::Collect::None,
            );
            assert_eq!(
                result.outputs,
                vec![IrBits::make_ubits(1, u64::from(value > 0x8000_0000)).unwrap()],
                "{mode:?}: input={value}"
            );
        }
    }
}
