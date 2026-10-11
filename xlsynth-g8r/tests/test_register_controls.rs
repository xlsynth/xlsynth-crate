// SPDX-License-Identifier: Apache-2.0

//! Explicit-control evaluation, legalization, and consumer boundary checks.

use xlsynth_g8r::aig::{
    ClockPort, RegisterBinding, RegisterControl, SequentialGateFn, SynchronousReset,
    TransitionInputId, TransitionOutputId,
};
use xlsynth_g8r::aig_serdes::blif::emit_blif;
use xlsynth_g8r::aig_serdes::emit_netlist::emit_netlist;
use xlsynth_g8r::aig_serdes::g8r::{decode_g8r_binary, emit_g8r, encode_g8r_binary, parse_g8r};
use xlsynth_g8r::aig_serdes::sequential2ir::sequential_gate_fn_to_pir_block_package;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::aig_sim::sequential::{
    SequentialState, count_sequential_toggle_activity, simulate,
};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_pir::IrBits;

/// Uses shared output roles and deliberately non-interface-order external IDs.
fn controlled_register(width: usize) -> SequentialGateFn {
    let mut builder = GateBuilder::new("controlled".into(), GateBuilderOptions::no_opt());
    let data = builder.add_input("data".into(), width);
    let q = builder.add_input("q".into(), width);
    let enable = builder.add_input("enable".into(), 1);
    let reset = builder.add_input("reset".into(), 1);
    builder.add_output("data".into(), data);
    builder.add_output("q".into(), q);
    builder.add_output("enable".into(), enable);
    builder.add_output("reset".into(), reset);
    SequentialGateFn::new(
        "controlled".into(),
        builder.build(),
        vec![
            TransitionInputId::new(3),
            TransitionInputId::new(0),
            TransitionInputId::new(2),
        ],
        vec![
            TransitionOutputId::new(2),
            TransitionOutputId::new(1),
            TransitionOutputId::new(0),
        ],
        Some(ClockPort { name: "clk".into() }),
        vec![RegisterBinding {
            name: "state".into(),
            q: TransitionInputId::new(1),
            d: TransitionOutputId::new(0),
            enable: Some(RegisterControl {
                signal: TransitionOutputId::new(2),
                active_low: false,
            }),
            reset: Some(SynchronousReset {
                control: RegisterControl {
                    signal: TransitionOutputId::new(3),
                    active_low: false,
                },
                value: IrBits::from_lsb_fn(width, |i| i % 3 == 0),
            }),
        }],
    )
    .unwrap()
}

#[test]
fn legalizes_controls_for_arbitrary_state_and_wide_packed_values() {
    for width in [0, 1, 8, 129] {
        for enable_low in [false, true] {
            for reset_low in [false, true] {
                let mut source = controlled_register(width);
                source.registers[0].enable.as_mut().unwrap().active_low = enable_low;
                source.registers[0]
                    .reset
                    .as_mut()
                    .unwrap()
                    .control
                    .active_low = reset_low;
                let legalized = source.legalize_register_controls().unwrap();
                assert_eq!(legalized.inputs, source.inputs);
                assert_eq!(legalized.registers.len(), 1);
                assert_eq!(legalized.registers[0].q, source.registers[0].q);
                assert_eq!(legalized.clock, source.clock);
                assert_eq!(
                    legalized
                        .transition
                        .outputs
                        .iter()
                        .map(|o| o.name.as_str())
                        .collect::<Vec<_>>(),
                    ["enable", "q", "data", "state__d"],
                );
                // All control combinations and distinct arbitrary data/state
                // patterns, including the cycle before reset was ever asserted.
                for pattern in 0..8 {
                    for enable in [false, true] {
                        for reset in [false, true] {
                            let data = IrBits::from_lsb_fn(width, |i| (i + pattern) % 3 == 0);
                            let q = IrBits::from_lsb_fn(width, |i| (i + pattern) % 4 == 1);
                            let result = gate_sim::eval(
                                &legalized.transition,
                                &[
                                    data.clone(),
                                    q.clone(),
                                    IrBits::bool(enable),
                                    IrBits::bool(reset),
                                ],
                                Collect::None,
                            );
                            let expected_d = if reset != reset_low {
                                source.registers[0].reset.as_ref().unwrap().value.clone()
                            } else if enable != enable_low {
                                data.clone()
                            } else {
                                q.clone()
                            };
                            assert_eq!(
                                result.outputs,
                                vec![IrBits::bool(enable), q, data, expected_d]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn absent_controls_load_unconditionally_and_self_feedback_holds() {
    let mut source = controlled_register(8);
    source.registers[0].enable = None;
    source.registers[0].reset = None;
    // Keep the former reset endpoint externally visible so every output is
    // bound.
    source.outputs.push(TransitionOutputId::new(3));
    for hold in [false, true] {
        if hold {
            source.registers[0].d = TransitionOutputId::new(1);
        }
        let legalized = source.legalize_register_controls().unwrap();
        let data = IrBits::make_ubits(8, 51).unwrap();
        let q = IrBits::make_ubits(8, 123).unwrap();
        let result = gate_sim::eval(
            &legalized.transition,
            &[
                data.clone(),
                q.clone(),
                IrBits::bool(false),
                IrBits::bool(true),
            ],
            Collect::None,
        );
        assert_eq!(
            &result.outputs[legalized.registers[0].d.index()],
            if hold { &q } else { &data }
        );
    }
}

#[test]
fn shared_controls_and_data_do_not_merge_registers() {
    let mut source = controlled_register(1);
    let mut second = source.registers[0].clone();
    second.name = "second".into();
    second.q = TransitionInputId::new(0);
    source.inputs.retain(|id| *id != second.q);
    source.registers.push(second);
    // The reset control is also the data and the enable of both registers.
    for register in &mut source.registers {
        register.d = TransitionOutputId::new(3);
        register.enable.as_mut().unwrap().signal = TransitionOutputId::new(3);
    }
    let legalized = source.legalize_register_controls().unwrap();
    assert_eq!(legalized.registers.len(), 2);
    for a in [false, true] {
        for b in [false, true] {
            for reset in [false, true] {
                let result = gate_sim::eval(
                    &legalized.transition,
                    &[
                        IrBits::bool(a),
                        IrBits::bool(b),
                        IrBits::bool(false),
                        IrBits::bool(reset),
                    ],
                    Collect::None,
                );
                assert_eq!(
                    result.outputs[3],
                    IrBits::bool(if reset { true } else { b })
                );
                assert_eq!(
                    result.outputs[4],
                    IrBits::bool(if reset { true } else { a })
                );
            }
        }
    }
}

#[test]
fn rejects_invalid_register_metadata() {
    let source = controlled_register(8);
    let mut bad = source.clone();
    bad.registers[0].enable.as_mut().unwrap().signal = TransitionOutputId::new(0);
    assert_eq!(
        bad.validate().unwrap_err(),
        "register 'state' control 'data' must have width 1, got 8"
    );
    let mut bad = source.clone();
    bad.registers[0].reset.as_mut().unwrap().value = IrBits::bool(false);
    assert_eq!(
        bad.validate().unwrap_err(),
        "register 'state' has Q width 8 but reset value width 1"
    );
    let mut bad = source.clone();
    bad.registers[0].d = TransitionOutputId::new(2);
    assert_eq!(
        bad.validate().unwrap_err(),
        "register 'state' has Q width 8 but D width 1"
    );
    let mut bad = source.clone();
    bad.registers[0].reset.as_mut().unwrap().control.signal = TransitionOutputId::new(42);
    assert_eq!(
        bad.validate().unwrap_err(),
        "register control references transition output index 42, but only 4 outputs exist"
    );
    let mut bad = source.clone();
    bad.clock = None;
    assert!(bad.validate().is_err());
    let mut bad = source.clone();
    bad.registers.push(bad.registers[0].clone());
    assert_eq!(
        bad.validate().unwrap_err(),
        "duplicate register name 'state'"
    );
    let mut bad = source.clone();
    bad.inputs.push(TransitionInputId::new(1));
    assert!(bad.validate().is_err());
    let mut bad = source.clone();
    bad.outputs.push(bad.outputs[0]);
    assert!(bad.validate().is_err());
    let mut bad = source.clone();
    bad.registers[0].reset = None;
    assert_eq!(
        bad.validate().unwrap_err(),
        "transition output 'reset' at index 3 is not bound as an external output or register input"
    );
    let mut bad = source;
    bad.inputs.pop();
    assert!(bad.validate().is_err());
}

#[test]
fn supports_no_registers_and_no_output_ports() {
    let builder = GateBuilder::new("empty".into(), GateBuilderOptions::no_opt());
    let combinational = xlsynth_g8r::aig::GateFn {
        name: builder.name,
        inputs: builder.inputs,
        outputs: builder.outputs,
        gates: builder.gates,
    };
    let source =
        SequentialGateFn::new("empty".into(), combinational, vec![], vec![], None, vec![]).unwrap();
    let legalized = source.legalize_register_controls().unwrap();
    assert!(legalized.registers.is_empty());
    assert!(legalized.transition.outputs.is_empty());
}

#[test]
fn constant_controls_fold_without_removing_state() {
    let mut source = controlled_register(8);
    let zero = xlsynth_g8r::aig::AigOperand {
        node: xlsynth_g8r::aig::AigRef { id: 0 },
        negated: false,
    };
    source.transition.outputs[2].bit_vector = zero.into();
    source.transition.outputs[3].bit_vector = zero.into();
    source.transition.outputs[1].name = "state__d".into();
    let legalized = source.legalize_register_controls().unwrap();
    assert_eq!(legalized.registers.len(), 1);
    assert_eq!(legalized.transition.outputs[3].name, "state__d__1");
    assert_eq!(
        legalized.transition.outputs[3]
            .bit_vector
            .iter_lsb_to_msb()
            .collect::<Vec<_>>(),
        legalized.transition.inputs[1]
            .bit_vector
            .iter_lsb_to_msb()
            .collect::<Vec<_>>(),
    );
    assert!(
        !legalized
            .transition
            .gates
            .iter()
            .any(|node| { matches!(node, xlsynth_g8r::aig::AigNode::And2 { .. }) })
    );
}

#[test]
fn serialization_preserves_dead_nodes_in_unlegalized_graphs() {
    let mut builder = GateBuilder::new("dead_nodes".into(), GateBuilderOptions::no_opt());
    let data = builder.add_input("data".into(), 2);
    let q = builder.add_input("q".into(), 2);
    let dead = builder.add_and_binary(*data.get_lsb(0), *q.get_lsb(0));
    builder.add_tag(dead.node, "unused".into());
    builder.add_pir_node_id(dead.node, 42);
    let enable = builder.add_and_binary(*data.get_lsb(1), *q.get_lsb(1));
    builder.add_output("out".into(), q);
    builder.add_output("raw_data".into(), data);
    builder.add_output("enable".into(), enable.into());
    let source = SequentialGateFn::new(
        "dead_nodes".into(),
        builder.build(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(0)],
        Some(ClockPort { name: "clk".into() }),
        vec![RegisterBinding {
            name: "state".into(),
            q: TransitionInputId::new(1),
            d: TransitionOutputId::new(1),
            enable: Some(RegisterControl {
                signal: TransitionOutputId::new(2),
                active_low: false,
            }),
            reset: None,
        }],
    )
    .unwrap();
    let text = emit_g8r(&source).unwrap();
    xlsynth_test_helpers::compare_golden_text(
        &format!("{text}\n"),
        "tests/goldens/register_controls.g8r",
    );
    let decoded = parse_g8r(&text).unwrap();
    assert_eq!(
        decoded.transition.gates.len(),
        source.transition.gates.len()
    );
    assert_eq!(
        decoded.transition.gates[dead.node.id].get_pir_node_ids(),
        &[42]
    );
    assert_eq!(decoded.registers, source.registers);
    assert_eq!(emit_g8r(&decoded).unwrap(), text);
}

#[test]
fn explicit_controls_simulate_identically_before_and_after_legalization() {
    for width in [0, 1, 8, 129] {
        for active_low in [false, true] {
            for controls in 0..4 {
                let mut source = controlled_register(width);
                source.outputs.push(TransitionOutputId::new(3));
                source.registers[0].enable.as_mut().unwrap().active_low = active_low;
                source.registers[0]
                    .reset
                    .as_mut()
                    .unwrap()
                    .control
                    .active_low = !active_low;
                if controls & 1 == 0 {
                    source.registers[0].enable = None;
                }
                if controls & 2 == 0 {
                    source.registers[0].reset = None;
                }
                let legalized = source.legalize_register_controls().unwrap();
                legalized.validate_legalized().unwrap();
                assert!(!legalized.has_register_controls());
                // The external order is reset, data, enable. No reset edge is
                // assumed before cycle zero; every trial starts with distinct
                // arbitrary state shared by the two representations.
                for startup in 0..4 {
                    let state = vec![IrBits::from_lsb_fn(width, |i| (i + startup) % 3 == 1)];
                    let stimulus = (0..32)
                        .map(|cycle| {
                            vec![
                                IrBits::bool(cycle & 4 != 0),
                                IrBits::from_lsb_fn(width, |i| (i + cycle) % 7 < 3),
                                IrBits::bool(cycle & 2 != 0),
                            ]
                        })
                        .collect::<Vec<_>>();
                    let actual = simulate(
                        &source,
                        &stimulus,
                        SequentialState::from_register_values(&source, state.clone()).unwrap(),
                    )
                    .unwrap();
                    let expected = simulate(
                        &legalized,
                        &stimulus,
                        SequentialState::from_register_values(&legalized, state).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(actual.external_outputs(), expected.external_outputs());
                    assert_eq!(actual.final_state(), expected.final_state());
                }
                // A second call must not apply controls twice or rebuild an
                // already-legalized graph, including its interface aliases.
                let twice = legalized.legalize_register_controls().unwrap();
                assert_eq!(emit_g8r(&legalized).unwrap(), emit_g8r(&twice).unwrap());
            }
        }
    }
}

#[test]
fn mixed_controlled_and_plain_registers_update_simultaneously() {
    let mut source = controlled_register(8);
    // Repurpose the old data input as Q of an unconditional register that
    // captures the controlled register's previous Q.
    source.inputs.retain(|id| id.index() != 0);
    source.registers.push(RegisterBinding {
        name: "plain".into(),
        q: TransitionInputId::new(0),
        d: TransitionOutputId::new(1),
        enable: None,
        reset: None,
    });
    let legalized = source.legalize_register_controls().unwrap();
    let values = vec![
        IrBits::make_ubits(8, 3).unwrap(),
        IrBits::make_ubits(8, 9).unwrap(),
    ];
    let stimulus = vec![
        vec![IrBits::bool(false), IrBits::bool(true)],
        vec![IrBits::bool(false), IrBits::bool(true)],
        vec![IrBits::bool(true), IrBits::bool(false)],
    ];
    let actual = simulate(
        &source,
        &stimulus,
        SequentialState::from_register_values(&source, values.clone()).unwrap(),
    )
    .unwrap();
    let expected = simulate(
        &legalized,
        &stimulus,
        SequentialState::from_register_values(&legalized, values).unwrap(),
    )
    .unwrap();
    assert_eq!(actual.external_outputs(), expected.external_outputs());
    assert_eq!(actual.final_state(), expected.final_state());
    assert_eq!(
        actual.external_outputs()[0][1],
        IrBits::make_ubits(8, 3).unwrap()
    );
    assert_eq!(
        actual.external_outputs()[1][1],
        IrBits::make_ubits(8, 9).unwrap()
    );
    assert_eq!(
        actual.external_outputs()[2][1],
        IrBits::make_ubits(8, 3).unwrap()
    );
}

#[test]
fn effective_d_consumers_reject_explicit_controls() {
    let mut source = controlled_register(8);
    // HDL ports share a namespace; use distinct input/output names here.
    for output in &mut source.transition.outputs {
        output.name = format!("out_{}", output.name);
    }
    source.validate().unwrap();
    let error = "register 'state' has explicit reset/enable controls; call legalize_register_controls() first";
    assert_eq!(source.validate_legalized().unwrap_err(), error);
    assert_eq!(
        emit_blif(&source).unwrap_err(),
        format!("cannot emit BLIF for invalid SequentialGateFn: {error}")
    );
    assert_eq!(
        emit_netlist(&source, true).unwrap_err(),
        format!("emit_netlist: invalid SequentialGateFn: {error}")
    );
    assert_eq!(
        sequential_gate_fn_to_pir_block_package(&source, "test").unwrap_err(),
        format!("sequential2ir: invalid SequentialGateFn: {error}")
    );
    let stimulus = vec![vec![
        IrBits::bool(false),
        IrBits::make_ubits(8, 42).unwrap(),
        IrBits::bool(true),
    ]];
    let trace = simulate(&source, &stimulus, SequentialState::all_zeros(&source)).unwrap();
    // Logical evaluation is supported; physical gate/toggle accounting still
    // needs the reset/enable gates to exist.
    assert_eq!(
        count_sequential_toggle_activity(&source, &trace).unwrap_err(),
        error
    );
    let legalized = source.legalize_register_controls().unwrap();
    assert!(emit_g8r(&legalized).is_ok());
    assert!(encode_g8r_binary(&legalized).is_ok());
    assert!(emit_blif(&legalized).is_ok());
    assert!(emit_netlist(&legalized, true).is_ok());
    assert!(sequential_gate_fn_to_pir_block_package(&legalized, "test").is_ok());
}

#[test]
fn explicit_control_serialization_preserves_wide_values_and_evaluation() {
    for width in [0, 1, 8, 129, 257] {
        let source = controlled_register(width);
        let stimulus = vec![
            vec![
                IrBits::bool(false),
                IrBits::zero(width),
                IrBits::bool(false),
            ],
            vec![IrBits::bool(true), IrBits::zero(width), IrBits::bool(false)],
            vec![
                IrBits::bool(false),
                IrBits::from_lsb_fn(width, |_| true),
                IrBits::bool(true),
            ],
        ];
        let initial = vec![IrBits::from_lsb_fn(width, |i| i % 2 == 0)];
        let expected = simulate(
            &source,
            &stimulus,
            SequentialState::from_register_values(&source, initial.clone()).unwrap(),
        )
        .unwrap();
        for decoded in [
            parse_g8r(&emit_g8r(&source).unwrap()).unwrap(),
            decode_g8r_binary(&encode_g8r_binary(&source).unwrap()).unwrap(),
        ] {
            assert_eq!(decoded.registers, source.registers);
            assert_eq!(emit_g8r(&decoded).unwrap(), emit_g8r(&source).unwrap());
            let actual = simulate(
                &decoded,
                &stimulus,
                SequentialState::from_register_values(&decoded, initial.clone()).unwrap(),
            )
            .unwrap();
            assert_eq!(actual.external_outputs(), expected.external_outputs());
            assert_eq!(actual.final_state(), expected.final_state());
        }
    }
}

#[test]
fn text_control_metadata_rejects_partial_and_invalid_groups() {
    let source = controlled_register(8);
    let text = emit_g8r(&source).unwrap();
    for (from, to, expected) in [
        (
            " enable_active_low=false",
            "",
            "register is missing 'enable_active_low' field",
        ),
        (
            " reset_value=bits[8]:0x49",
            "",
            "register is missing 'reset_value' field",
        ),
        (
            "enable=2",
            "enable=42",
            "register control references transition output index 42, but only 4 outputs exist",
        ),
        (
            "reset_value=bits[8]:0x49",
            "reset_value=bits[1]:0",
            "register 'state' has Q width 8 but reset value width 1",
        ),
    ] {
        let malformed = text.replace(from, to);
        assert_ne!(malformed, text, "fixture must contain {from}");
        let error = parse_g8r(&malformed).unwrap_err();
        if expected.starts_with("register 'state'") || expected.starts_with("register control") {
            assert_eq!(
                error,
                format!("invalid SequentialGateFn payload: {expected}")
            );
        } else {
            assert_eq!(error, expected);
        }
    }
    assert_eq!(
        parse_g8r(&text.replacen("g8r_v4", "g8r_v3", 1)).unwrap_err(),
        "unsupported g8r header 'g8r_v3'; expected 'g8r_v4'"
    );
    let mut legacy = encode_g8r_binary(&source).unwrap();
    legacy[b"g8rbin_v".len()] = b'3';
    assert_eq!(
        decode_g8r_binary(&legacy).unwrap_err(),
        "unsupported g8rbin header; expected 'g8rbin_v4'"
    );
}
