// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_query;

#[test]
fn query_number_matches_wide_literal_when_upper_bits_are_zero() {
    let ir_text = r#"package test

top fn main(x: bits[128] id=1) -> bits[1] {
  literal.2: bits[128] = literal(value=1, id=2)
  ret eq.3: bits[1] = eq(x, literal.2, id=3)
}
"#;

    let mut parser = Parser::new(ir_text);
    let pkg = parser.parse_and_validate_package().unwrap();
    let f = pkg.get_top_fn().unwrap();

    let q_lit = ir_query::parse_query("literal(1)").unwrap();
    let lit_matches = ir_query::find_matching_nodes(f, &q_lit);
    assert_eq!(lit_matches.len(), 1);
    assert!(
        matches!(
            f.get_node(lit_matches[0]).payload,
            ir::NodePayload::Literal(_)
        ),
        "expected literal match, got: {:?}",
        f.get_node(lit_matches[0]).payload
    );

    let q_eq = ir_query::parse_query("eq(_, 1)").unwrap();
    let eq_matches = ir_query::find_matching_nodes(f, &q_eq);
    assert_eq!(eq_matches.len(), 1);
    assert_eq!(f.get_node(eq_matches[0]).payload.get_operator(), "eq");
}
