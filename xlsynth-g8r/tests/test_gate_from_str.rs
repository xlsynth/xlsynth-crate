// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::gate_parser::ParseError;
use xlsynth_g8r::test_utils::{
    interesting_ir_roundtrip_cases, load_interesting_ir_roundtrip_case,
    setup_graph_with_redundancies, setup_simple_graph, structurally_equivalent,
};

fn assert_text_roundtrip(case_name: &str, g: &GateFn) -> Result<(), ParseError> {
    let text = g.to_string();
    let parsed = GateFn::try_from(text.as_str())?;
    assert!(
        structurally_equivalent(g, &parsed),
        "GateFn text roundtrip changed structure for {case_name}"
    );
    Ok(())
}

#[test]
fn test_round_trip_simple() -> Result<(), ParseError> {
    assert_text_roundtrip("simple_graph", &setup_simple_graph().g)
}

#[test]
fn test_round_trip_redundant() -> Result<(), ParseError> {
    assert_text_roundtrip("redundant_graph", &setup_graph_with_redundancies().g)
}

#[test]
fn test_round_trip_interesting_signatures() -> Result<(), ParseError> {
    for case in interesting_ir_roundtrip_cases() {
        let sample = load_interesting_ir_roundtrip_case(case);
        assert_text_roundtrip(case.name, &sample.gate_fn)?;
    }
    Ok(())
}
