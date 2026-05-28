// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrBits;
use xlsynth_g8r::aig::{
    ClockPort, RegisterBinding, ResetSpec, SequentialGateFn, TransitionInputId, TransitionOutputId,
};
use xlsynth_g8r::aig_serdes::g8r::{decode_g8r_binary, emit_g8r, encode_g8r_binary, parse_g8r};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::test_utils::structurally_equivalent;

#[test]
fn zero_register_sequential_gate_fn_may_omit_clock() {
    let mut builder = GateBuilder::new("passthrough".to_string(), GateBuilderOptions::opt());
    let x = builder.add_input("x".to_string(), 2);
    builder.add_output("y".to_string(), x);

    let sequential = SequentialGateFn::new(
        "passthrough".to_string(),
        builder.build(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(0)],
        None,
        vec![],
    )
    .unwrap();

    assert!(sequential.registers.is_empty());
    assert!(sequential.clock.is_none());
}

#[test]
fn native_g8r_stores_combinational_design_as_sequential_gate_fn() {
    let mut builder = GateBuilder::new("passthrough".to_string(), GateBuilderOptions::opt());
    let x = builder.add_input("x".to_string(), 2);
    builder.add_output("y".to_string(), x);
    let transition = builder.build();
    let sequential = SequentialGateFn::new(
        "transition".to_string(),
        transition.clone(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(0)],
        None,
        vec![],
    )
    .unwrap();

    let text = emit_g8r(&sequential);
    assert!(text.starts_with("g8r_v1\n"));
    let parsed = parse_g8r(&text).unwrap();
    assert_eq!(parsed.name, "transition");
    assert!(parsed.registers.is_empty());
    assert!(structurally_equivalent(&transition, &parsed.transition));
    let binary = encode_g8r_binary(&sequential).unwrap();
    assert!(binary.starts_with(b"g8rbin_v1\n"));
    assert!(decode_g8r_binary(&binary).unwrap().registers.is_empty());

    let wrapped = SequentialGateFn::from_gate_fn(transition.clone());
    let extracted = parse_g8r(&emit_g8r(&wrapped))
        .unwrap()
        .try_into_gate_fn()
        .unwrap();
    assert!(structurally_equivalent(&transition, &extracted));
}

#[test]
fn register_can_bind_data_load_enable_and_reset_transition_outputs() {
    let mut builder =
        GateBuilder::new("pipeline_transition".to_string(), GateBuilderOptions::opt());
    let data = builder.add_input("data".to_string(), 8);
    let load_enable = builder.add_input("load_enable".to_string(), 1);
    let reset = builder.add_input("reset".to_string(), 1);
    let state_q = builder.add_input("state_q".to_string(), 8);
    builder.add_output("result".to_string(), state_q);
    builder.add_output("state_d".to_string(), data);
    builder.add_output("state_load_enable".to_string(), load_enable);
    builder.add_output("state_reset".to_string(), reset);

    let sequential = SequentialGateFn::new(
        "pipeline".to_string(),
        builder.build(),
        vec![
            TransitionInputId::new(0),
            TransitionInputId::new(1),
            TransitionInputId::new(2),
        ],
        vec![TransitionOutputId::new(0)],
        Some(ClockPort {
            name: "clk".to_string(),
        }),
        vec![RegisterBinding {
            name: "state".to_string(),
            q: TransitionInputId::new(3),
            d: TransitionOutputId::new(1),
            load_enable: Some(TransitionOutputId::new(2)),
            reset: Some(ResetSpec {
                signal: TransitionOutputId::new(3),
                asynchronous: false,
                active_low: false,
                value: IrBits::make_ubits(8, 3).unwrap(),
            }),
            initial_value: None,
        }],
    )
    .unwrap();

    assert_eq!(sequential.registers[0].name, "state");
    assert_eq!(
        sequential.registers[0]
            .reset
            .as_ref()
            .unwrap()
            .value
            .get_bit_count(),
        8
    );

    let text = emit_g8r(&sequential);
    assert!(text.contains("\nclock clk\n"));
    assert!(!text.contains("\nclock posedge "));
    assert!(!text.contains("\nclock negedge "));

    for round_tripped in [
        parse_g8r(&text).unwrap(),
        decode_g8r_binary(&encode_g8r_binary(&sequential).unwrap()).unwrap(),
    ] {
        assert_eq!(round_tripped.clock, sequential.clock);
        assert_eq!(round_tripped.inputs, sequential.inputs);
        assert_eq!(round_tripped.outputs, sequential.outputs);
        assert_eq!(round_tripped.registers, sequential.registers);
        assert!(structurally_equivalent(
            &round_tripped.transition,
            &sequential.transition
        ));
    }

    assert_eq!(
        sequential.try_into_gate_fn().unwrap_err(),
        "cannot convert design 'pipeline' to GateFn: design contains 1 register(s) and clock 'clk'"
    );
}

#[test]
fn combinational_conversion_restores_external_port_order() {
    let mut builder = GateBuilder::new("transition".to_string(), GateBuilderOptions::opt());
    let a = builder.add_input("a".to_string(), 1);
    let b = builder.add_input("b".to_string(), 1);
    builder.add_output("first".to_string(), a);
    builder.add_output("second".to_string(), b);

    let sequential = SequentialGateFn::new(
        "ordered".to_string(),
        builder.build(),
        vec![TransitionInputId::new(1), TransitionInputId::new(0)],
        vec![TransitionOutputId::new(1), TransitionOutputId::new(0)],
        None,
        vec![],
    )
    .unwrap();
    let gate_fn = sequential.try_into_gate_fn().unwrap();

    assert_eq!(gate_fn.name, "ordered");
    assert_eq!(gate_fn.inputs[0].name, "b");
    assert_eq!(gate_fn.inputs[1].name, "a");
    assert_eq!(gate_fn.outputs[0].name, "second");
    assert_eq!(gate_fn.outputs[1].name, "first");
}

#[test]
fn combinational_conversion_rejects_declared_clock_without_registers() {
    let mut builder = GateBuilder::new("clocked".to_string(), GateBuilderOptions::opt());
    let x = builder.add_input("x".to_string(), 1);
    builder.add_output("y".to_string(), x);
    let sequential = SequentialGateFn::new(
        "clocked".to_string(),
        builder.build(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(0)],
        Some(ClockPort {
            name: "clk".to_string(),
        }),
        vec![],
    )
    .unwrap();

    assert_eq!(
        sequential.try_into_gate_fn().unwrap_err(),
        "cannot convert design 'clocked' to GateFn: design declares clock 'clk'"
    );
}

#[test]
fn registered_sequential_gate_fn_requires_clock() {
    let mut builder =
        GateBuilder::new("register_transition".to_string(), GateBuilderOptions::opt());
    let state_q = builder.add_input("state_q".to_string(), 1);
    builder.add_output("state_d".to_string(), state_q);

    let error = SequentialGateFn::new(
        "register".to_string(),
        builder.build(),
        vec![],
        vec![],
        None,
        vec![RegisterBinding {
            name: "state".to_string(),
            q: TransitionInputId::new(0),
            d: TransitionOutputId::new(0),
            load_enable: None,
            reset: None,
            initial_value: None,
        }],
    )
    .unwrap_err();

    assert_eq!(error, "a SequentialGateFn with registers must have a clock");
}

#[test]
fn register_data_width_must_match_state_width() {
    let mut builder = GateBuilder::new("bad_transition".to_string(), GateBuilderOptions::opt());
    let state_q = builder.add_input("state_q".to_string(), 8);
    builder.add_output("state_d".to_string(), state_q.get_lsb_slice(0, 1));

    let error = SequentialGateFn::new(
        "bad".to_string(),
        builder.build(),
        vec![],
        vec![],
        Some(ClockPort {
            name: "clk".to_string(),
        }),
        vec![RegisterBinding {
            name: "state".to_string(),
            q: TransitionInputId::new(0),
            d: TransitionOutputId::new(0),
            load_enable: None,
            reset: None,
            initial_value: None,
        }],
    )
    .unwrap_err();

    assert_eq!(error, "register 'state' has Q width 8 but D width 1");
}
