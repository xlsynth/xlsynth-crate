// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_query;

#[test]
fn width_matcher_respects_typed_placeholder() {
    let ir_text = r#"package test

top fn main(x: bits[8] id=1, y: bits[16] id=2) -> bits[16] {
  bs8: bits[8] = bit_slice(x, start=0, width=8, id=3)
  ret bs16: bits[16] = bit_slice(y, start=0, width=16, id=4)
}
"#;

    let mut parser = Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().unwrap();
    let f = pkg.get_top_fn().unwrap();

    // `$width(x: bits[8])` is a reference-time assertion: x must already be bound
    // to a bits[8] node, otherwise the `$width(...)` expression evaluates to no
    // solution and matching fails.
    let query = ir_query::parse_query("bit_slice(x, start=0, width=$width(x: bits[8]))").unwrap();
    let matches = ir_query::find_matching_nodes(f, &query);
    assert_eq!(matches.len(), 1);
    assert_eq!(ir::node_textual_id(f, matches[0]), "bs8");
}
