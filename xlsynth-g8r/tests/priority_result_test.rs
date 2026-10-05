// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::prep_for_gatify::{
    PrepForGatifyOptions, prep_for_gatify, prep_for_gatify_with_priority_result,
};
use xlsynth_pir::desugar_extensions::desugar_extensions_in_fn;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn exclusive_priority_cones_prove_and_sharing_controls_are_respected() {
    let cases = [
        (
            include_str!("testdata/priority_result/shared-input63.ir"),
            true,
        ),
        (include_str!("testdata/priority_result/priority8.ir"), true),
        (
            include_str!("testdata/priority_result/historical-priority.ir"),
            true,
        ),
        (
            include_str!("testdata/priority_result/shared-hot63.ir"),
            true,
        ),
        (
            include_str!("testdata/priority_result/shared-amount63.ir"),
            false,
        ),
        (
            include_str!("testdata/priority_result/shared-count63.ir"),
            false,
        ),
        (
            include_str!("testdata/priority_result/truncated-count.ir"),
            false,
        ),
        (
            include_str!("testdata/priority_result/wrong-size.ir"),
            false,
        ),
    ];
    for (text, should_change) in cases {
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let source = package.get_top_fn().unwrap();
        let options = PrepForGatifyOptions::all_opts_enabled();
        let normal = prep_for_gatify(source, None, options);
        let candidate = prep_for_gatify_with_priority_result(source, None, options);
        assert_eq!(normal.to_string() != candidate.to_string(), should_change);
        let mut basis = candidate.clone();
        desugar_extensions_in_fn(&mut basis).unwrap();
        check_equivalence::check_equivalence_via_toolchain(
            text,
            &format!(
                "package result

top {basis}"
            ),
        )
        .unwrap();
    }
}

#[test]
fn zero_width_control_and_disabled_priority_rewrite_are_unchanged() {
    for text in [
        include_str!("testdata/priority_result/zero.ir"),
        include_str!("testdata/priority_result/priority8.ir"),
    ] {
        let package = Parser::new(text).parse_and_validate_package().unwrap();
        let source = package.get_top_fn().unwrap();
        let options = PrepForGatifyOptions::all_opts_disabled();
        assert_eq!(
            prep_for_gatify(source, None, options).to_string(),
            prep_for_gatify_with_priority_result(source, None, options).to_string(),
        );
    }
}
