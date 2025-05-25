// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::validate_equiv::{prove_gate_fn_equiv, Ctx, EquivResult};

#[test]
fn test_simple_equivalence() {
    let mut gb = GateBuilder::new("xor".to_string(), GateBuilderOptions::opt());
    let a = gb.add_input("a".to_string(), 1).get_lsb(0).clone();
    let b = gb.add_input("b".to_string(), 1).get_lsb(0).clone();
    let x = gb.add_xor_binary(a, b);
    gb.add_output("out".to_string(), x.into());
    let g1 = gb.build();

    let mut ctx = Ctx::new();
    assert_eq!(prove_gate_fn_equiv(&g1, &g1, &mut ctx), EquivResult::Proved);
}

#[test]
fn test_simple_inequivalence() {
    let mut gb1 = GateBuilder::new("xor".to_string(), GateBuilderOptions::opt());
    let a1 = gb1.add_input("a".to_string(), 1).get_lsb(0).clone();
    let b1 = gb1.add_input("b".to_string(), 1).get_lsb(0).clone();
    let x1 = gb1.add_xor_binary(a1, b1);
    gb1.add_output("out".to_string(), x1.into());
    let g1 = gb1.build();

    let mut gb2 = GateBuilder::new("and".to_string(), GateBuilderOptions::opt());
    let a2 = gb2.add_input("a".to_string(), 1).get_lsb(0).clone();
    let b2 = gb2.add_input("b".to_string(), 1).get_lsb(0).clone();
    let y = gb2.add_and_binary(a2, b2);
    gb2.add_output("out".to_string(), y.into());
    let g2 = gb2.build();

    let mut ctx = Ctx::new();
    match prove_gate_fn_equiv(&g1, &g2, &mut ctx) {
        EquivResult::Proved => panic!("Expected inequivalent"),
        EquivResult::Disproved(_) => (),
    }
}
