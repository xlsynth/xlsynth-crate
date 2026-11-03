// SPDX-License-Identifier: Apache-2.0

use super::MAX_LANES;
use crate::aig::AigNode;
use crate::gatify::ir2gate::TerminalObservationMode;
use crate::ir2gates::{Ir2GatesOptions, ir2gates_with_terminal_observations};

#[test]
fn single_wide_select_with_one_shared_output_gate() {
    let ir = r#"package wide

top fn f(a: bits[1] id=1, b: bits[1] id=2, s: bits[2] id=3) -> bits[4096] {
  x: bits[4096] = sign_ext(a, new_bit_count=4096, id=4)
  y: bits[4096] = sign_ext(b, new_bit_count=4096, id=5)
  zero: bits[4096] = literal(value=0, id=6)
  ret word: bits[4096] = sel(s, cases=[x, y, zero, zero], id=7)
}"#;
    let (result, capture) = ir2gates_with_terminal_observations(
        ir,
        None,
        Ir2GatesOptions::all_opts_disabled(),
        TerminalObservationMode::WordChoices,
    )
    .unwrap();
    let graph = result.gatify_output.gate_fn;
    let word = &graph.outputs[0].bit_vector;
    assert_eq!(word.get_bit_count(), 4096);
    let selected = word.get_lsb(0);
    assert!(matches!(
        graph.gates[selected.node.id],
        AigNode::And2 { .. }
    ));
    assert!(word.iter_lsb_to_msb().all(|bit| bit == selected));
    assert!(capture.word_groups.unwrap().is_empty());
}

#[test]
fn capture_checks_lane_limit_before_expanding_fanout() {
    // A single selected bit feeds both its original output and every bit of
    // the sign extension. The two select words account for four more slots.
    for extra in [MAX_LANES - 4, MAX_LANES - 3] {
        let ir = format!(
            r#"package fanout

top fn f(a: bits[2] id=1, b: bits[2] id=2, s: bits[2] id=3) -> (bits[2], bits[2], bits[{extra}]) {{
  zero: bits[2] = literal(value=0, id=4)
  left: bits[2] = sel(s, cases=[a, b, zero, zero], id=5)
  right: bits[2] = sel(s, cases=[a, b, a, zero], id=6)
  low: bits[1] = bit_slice(left, start=0, width=1, id=7)
  repeated: bits[{extra}] = sign_ext(low, new_bit_count={extra}, id=8)
  ret outputs: (bits[2], bits[2], bits[{extra}]) = tuple(left, right, repeated, id=9)
}}"#
        );
        let (_, capture) = ir2gates_with_terminal_observations(
            &ir,
            None,
            Ir2GatesOptions::all_opts_disabled(),
            TerminalObservationMode::WordChoices,
        )
        .unwrap();
        let groups = capture.word_groups.unwrap();
        if extra + 4 == MAX_LANES {
            assert_eq!(groups.len(), 1);
            assert_eq!(
                groups[0]
                    .group
                    .words
                    .iter()
                    .map(|w| w.lanes.len())
                    .sum::<usize>(),
                MAX_LANES
            );
        } else {
            assert!(groups.is_empty());
        }
    }
}
