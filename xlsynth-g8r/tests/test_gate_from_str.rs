// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig_serdes::gate_parser::ParseError;
use xlsynth_g8r::test_utils::{
    setup_graph_with_redundancies, setup_simple_graph, structurally_equivalent,
};

#[test]
fn test_round_trip_simple() -> Result<(), ParseError> {
    let g = setup_simple_graph().g;
    let text = g.to_string();
    let parsed = GateFn::try_from(text.as_str())?;
    assert!(structurally_equivalent(&g, &parsed));
    Ok(())
}

#[test]
fn test_round_trip_redundant() -> Result<(), ParseError> {
    let g = setup_graph_with_redundancies().g;
    let text = g.to_string();
    let parsed = GateFn::try_from(text.as_str())?;
    assert!(structurally_equivalent(&g, &parsed));
    Ok(())
}
