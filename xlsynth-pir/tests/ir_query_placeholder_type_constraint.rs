// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_query;

#[test]
fn query_placeholder_type_constraint_filters_matches() {
    let ir_text = r#"package test

top fn main(x: bits[16] id=1, b1: bits[1] id=2, b8: bits[8] id=3) -> bits[16] {
  se1: bits[16] = sign_ext(b1, new_bit_count=16, id=4)
  se8: bits[16] = sign_ext(b8, new_bit_count=16, id=5)
  and1: bits[16] = and(x, se1, id=6)
  and8: bits[16] = and(x, se8, id=7)
  ret out: bits[16] = or(and1, and8, id=8)
}
"#;

    let mut parser = Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().unwrap();
    let f = pkg.get_top_fn().unwrap();

    let query = ir_query::parse_query("and(x, sign_ext(b: bits[1]))").unwrap();
    let matches = ir_query::find_matching_nodes(f, &query);
    assert_eq!(matches.len(), 1);
    assert_eq!(ir::node_textual_id(f, matches[0]), "and1");
}
