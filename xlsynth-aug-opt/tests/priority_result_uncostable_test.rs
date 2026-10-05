// SPDX-License-Identifier: Apache-2.0

use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_g8r::check_equivalence::check_equivalence_with_top_via_toolchain;
use xlsynth_pir::ir::{NodePayload, Package};
use xlsynth_pir::ir_parser::Parser;

const SOURCE: &str = r#"package uncostable_priority

fn helper(y: bits[8] id=1) -> bits[8] {
  ret result: bits[8] = not(y, id=2)
}

top fn main(x: bits[32] id=3, y: bits[8] id=4) -> (bits[33], bits[8]) {
  hot: bits[33] = one_hot(x, lsb_prio=false, id=5)
  count: bits[6] = encode(hot, id=6)
  decoded: bits[33] = decode(count, width=33, id=7)
  called: bits[8] = invoke(y, to_apply=helper, id=8)
  ret result: (bits[33], bits[8]) = tuple(decoded, called, id=9)
}
"#;

#[test]
fn priority_fusion_declines_uncostable_package_calls_without_changing_them() {
    let source_package = Parser::new(SOURCE).parse_and_validate_package().unwrap();
    for rounds in [1, 3] {
        let options = AugOptOptions {
            enable: true,
            rounds,
            mode: AugOptMode::PirOnly,
            ..Default::default()
        };
        let disabled = run_aug_opt_over_ir_text_with_stats(SOURCE, Some("main"), options).unwrap();
        let enabled = run_aug_opt_over_ir_text_with_stats(
            SOURCE,
            Some("main"),
            AugOptOptions {
                fuse_priority_results: true,
                ..options
            },
        )
        .unwrap();
        assert_eq!(enabled.rewrite_stats.priority_results_fused, 0);
        assert_eq!(disabled.output_text, enabled.output_text);
        assert_eq!(disabled.total_rewrites, enabled.total_rewrites);
        let result_package: Package = Parser::new(&enabled.output_text)
            .parse_and_validate_package()
            .unwrap();
        assert_eq!(
            source_package.get_fn("helper").unwrap().to_string(),
            result_package.get_fn("helper").unwrap().to_string()
        );
        assert!(
            result_package
                .get_top_fn()
                .unwrap()
                .nodes
                .iter()
                .any(|node| { matches!(node.payload, NodePayload::Invoke { .. }) })
        );
        check_equivalence_with_top_via_toolchain(SOURCE, &enabled.output_text, Some("main"))
            .unwrap();

        // The live invoke prevents whole-function costing. Replacing only that
        // call exposes the same priority cone to a successful cost comparison.
        let costable = SOURCE.replace("invoke(y, to_apply=helper, id=8)", "not(y, id=8)");
        let fused = run_aug_opt_over_ir_text_with_stats(
            &costable,
            Some("main"),
            AugOptOptions {
                fuse_priority_results: true,
                ..options
            },
        )
        .unwrap();
        assert_eq!(fused.rewrite_stats.priority_results_fused, 1);
        check_equivalence_with_top_via_toolchain(&costable, &fused.output_text, Some("main"))
            .unwrap();
    }
}
