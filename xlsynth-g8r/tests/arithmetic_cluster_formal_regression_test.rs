// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::ir_parser::Parser;

/// Proves complete result vectors and verifies that the actual extraction plan
/// deferred the intended products. These cases need no diagnostic trace hooks.
#[test]
fn arithmetic_cluster_resolution_boundaries_formal() {
    for (source, deferred_ids) in [
        (
            r#"package identical
top fn f(a: bits[4] id=1, b: bits[4] id=2) -> bits[8] {
  p: bits[8] = umul(a, b, id=3)
  q: bits[8] = umul(a, b, id=4)
  ret r: bits[8] = ext_nary_add(p, q, signed=[false, false], negated=[false, false], id=5)
}"#,
            vec![3, 4],
        ),
        (
            r#"package signed_truncation
top fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[8] id=3) -> bits[8] {
  p: bits[10] = smul(a, b, id=4)
  ret r: bits[8] = ext_nary_add(p, c, signed=[true, false], negated=[true, false], id=5)
}"#,
            vec![4],
        ),
        (
            r#"package selected_negate
top fn f(a: bits[3] id=1, b: bits[3] id=2, c: bits[6] id=3, s: bits[1] id=4) -> bits[6] {
  p: bits[6] = smul(a, b, id=5)
  n: bits[6] = neg(c, id=6)
  v: bits[6] = sel(s, cases=[c, n], id=7)
  ret r: bits[6] = ext_nary_add(p, v, signed=[false, false], negated=[false, false], id=8)
}"#,
            vec![5],
        ),
    ] {
        let package = Parser::new(source).parse_and_validate_package().unwrap();
        let f = package.get_top_fn().unwrap();
        let mapped = gatify_prepared_fn(
            f,
            GatifyOptions {
                fold: true,
                hash: true,
                enable_rewrite_nary_add: true,
                adder_mapping: AdderMapping::BrentKung,
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        for text_id in deferred_ids {
            let index = f.nodes.iter().position(|n| n.text_id == text_id).unwrap();
            assert!(
                !mapped
                    .lowering_map
                    .contains_key(&xlsynth_pir::ir::NodeRef { index }),
                "product {text_id} was not deferred"
            );
        }
        xlsynth_g8r::check_equivalence::validate_same_fn_with_solver(
            f,
            &mapped.gate_fn,
            xlsynth_prover::prover::SolverChoice::Bitwuzla,
            None,
        )
        .unwrap();
    }
}
