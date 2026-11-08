// SPDX-License-Identifier: Apache-2.0

use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_g8r::check_equivalence::check_equivalence_with_top_via_toolchain;
use xlsynth_pir::ir::{NodePayload, Unop};
use xlsynth_pir::ir_parser::Parser;

const BVC_CONE: &str = include_str!("fixtures/clz_predicate_c0865afd0d0b6b09.ir");
const BVC_TOP: &str = "__k3_cone_c0865afd0d0b6b09";

#[test]
fn clz_predicate_bvc_cone_reduces_to_input_msb_and_stays_stable() {
    let direct_msb = r#"package direct_msb

top fn __k3_cone_c0865afd0d0b6b09(leaf_138: bits[161] id=1) -> bits[1] {
  ret msb: bits[1] = bit_slice(leaf_138, start=160, width=1, id=2)
}
"#;
    for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
        let mut first_output = None;
        for rounds in [1, 3] {
            let rewritten = run_aug_opt_over_ir_text_with_stats(
                BVC_CONE,
                Some(BVC_TOP),
                AugOptOptions {
                    enable: true,
                    rounds,
                    mode,
                    ..Default::default()
                },
            )
            .expect("aug-opt on BVC cone");
            assert_eq!(rewritten.rewrite_stats.priority_results_fused, 1);
            check_equivalence_with_top_via_toolchain(
                BVC_CONE,
                &rewritten.output_text,
                Some(BVC_TOP),
            )
            .expect("rewritten BVC cone is equivalent for every input");
            check_equivalence_with_top_via_toolchain(
                &rewritten.output_text,
                direct_msb,
                Some(BVC_TOP),
            )
            .expect("BVC cone is equivalent to input MSB");

            if mode == AugOptMode::Sandwich {
                // The following libxls pass absorbs the predicate and removes
                // the now-unused priority encoder.
                let pkg = Parser::new(&rewritten.output_text)
                    .parse_and_validate_package()
                    .expect("parse/validate sandwich output");
                let f = pkg.get_fn(BVC_TOP).expect("BVC function");
                let mut result = f.get_node(f.ret_node_ref.expect("return node"));
                while let NodePayload::Unop(Unop::Identity, arg) = result.payload {
                    result = f.get_node(arg);
                }
                let NodePayload::BitSlice {
                    arg,
                    start: 160,
                    width: 1,
                } = result.payload
                else {
                    panic!("expected direct input MSB, got {:?}", result.payload);
                };
                assert!(matches!(f.get_node(arg).payload, NodePayload::Param));
            }
            if let Some(first_output) = &first_output {
                assert_eq!(
                    first_output, &rewritten.output_text,
                    "extra rounds should preserve the result in {mode:?} mode"
                );
            } else {
                first_output = Some(rewritten.output_text);
            }
        }
    }
}
