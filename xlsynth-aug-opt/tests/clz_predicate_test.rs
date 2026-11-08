// SPDX-License-Identifier: Apache-2.0

use xlsynth_aug_opt::{AugOptMode, AugOptOptions, run_aug_opt_over_ir_text_with_stats};
use xlsynth_g8r::check_equivalence::check_equivalence_with_top_via_toolchain;
use xlsynth_pir::ir::{NodePayload, Unop};
use xlsynth_pir::ir_parser::Parser;

const BVC_CONE: &str = include_str!("fixtures/clz_predicate_c0865afd0d0b6b09.ir");
const BVC_TOP: &str = "__k3_cone_c0865afd0d0b6b09";

#[test]
fn clz_equalities_lower_to_input_bits_in_both_modes() {
    // The threshold makes the shared-count group strictly cheaper. An isolated
    // equality can already map as cheaply as the direct input expression.
    for op in ["eq", "ne"] {
        let source = format!(
            r#"package equality

top fn main(x: bits[8] id=1) -> (bits[1], bits[1]) {{
  reversed: bits[8] = reverse(x, id=2)
  hot: bits[9] = one_hot(reversed, lsb_prio=true, id=3)
  count: bits[4] = encode(hot, id=4)
  limit: bits[4] = literal(value=3, id=5)
  equality: bits[1] = {op}(count, limit, id=6)
  high_count: bits[3] = bit_slice(count, start=1, width=3, id=7)
  at_least_two: bits[1] = or_reduce(high_count, id=8)
  ret out: (bits[1], bits[1]) = tuple(equality, at_least_two, id=9)
}}
"#
        );
        for mode in [AugOptMode::PirOnly, AugOptMode::Sandwich] {
            let rewritten = run_aug_opt_over_ir_text_with_stats(
                &source,
                Some("main"),
                AugOptOptions {
                    enable: true,
                    mode,
                    ..Default::default()
                },
            )
            .expect("aug-opt on CLZ equality");
            assert_eq!(rewritten.rewrite_stats.priority_results_fused, 2);
            let pkg = Parser::new(&rewritten.output_text)
                .parse_and_validate_package()
                .expect("parse/validate equality output");
            let f = pkg.get_top_fn().expect("top function");
            let live = xlsynth_pir::dce::remove_dead_nodes(f);
            assert!(live.nodes.iter().all(|node| !matches!(
                node.payload,
                NodePayload::Encode { .. } | NodePayload::OneHot { .. }
            )));
            check_equivalence_with_top_via_toolchain(&source, &rewritten.output_text, Some("main"))
                .expect("direct equality predicate is equivalent");
        }
    }
}

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
