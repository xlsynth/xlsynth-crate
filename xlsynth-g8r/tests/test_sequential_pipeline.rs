// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::{ClockPort, SequentialPipelineOptions, stitch_gate_fns_into_pipeline};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};

fn passthrough_stage(
    name: &str,
    input_name: &str,
    output_name: &str,
    width: usize,
) -> xlsynth_g8r::aig::GateFn {
    let mut builder = GateBuilder::new(name.to_string(), GateBuilderOptions::no_opt());
    let input = builder.add_input(input_name.to_string(), width);
    builder.add_output(output_name.to_string(), input);
    builder.build()
}

#[test]
fn stitches_adjacent_stage_ports_through_registers() {
    let mut stage0 = GateBuilder::new("stage0".to_string(), GateBuilderOptions::no_opt());
    let a = stage0.add_input("a".to_string(), 2);
    let b = stage0.add_input("b".to_string(), 1);
    stage0.add_output("low".to_string(), a);
    stage0.add_output("high".to_string(), b);

    let mut stage1 = GateBuilder::new("stage1".to_string(), GateBuilderOptions::no_opt());
    let lhs = stage1.add_input("lhs".to_string(), 2);
    let _rhs = stage1.add_input("rhs".to_string(), 1);
    stage1.add_output("result".to_string(), lhs);

    let design = stitch_gate_fns_into_pipeline(
        &[stage0.build(), stage1.build()],
        &SequentialPipelineOptions {
            name: "stitched".to_string(),
            clock: ClockPort {
                name: "clk".to_string(),
            },
            flop_inputs: false,
            flop_outputs: false,
            input_valid_signal: None,
            output_valid_signal: None,
            reset_signal: None,
            reset_active_low: false,
        },
    )
    .unwrap();

    assert_eq!(design.name, "stitched");
    assert_eq!(design.clock.unwrap().name, "clk");
    assert_eq!(design.inputs.len(), 2);
    assert_eq!(design.outputs.len(), 1);
    assert_eq!(design.registers.len(), 2);
    assert_eq!(design.registers[0].name, "p0_lhs");
    assert_eq!(design.registers[1].name, "p0_rhs");
}

#[test]
fn inserts_requested_external_register_layers() {
    let stages = [
        passthrough_stage("stage0", "data", "sum", 4),
        passthrough_stage("stage1", "sum", "result", 4),
    ];
    let design = stitch_gate_fns_into_pipeline(
        &stages,
        &SequentialPipelineOptions {
            name: "registered".to_string(),
            clock: ClockPort {
                name: "clk".to_string(),
            },
            flop_inputs: true,
            flop_outputs: true,
            input_valid_signal: None,
            output_valid_signal: None,
            reset_signal: None,
            reset_active_low: false,
        },
    )
    .unwrap();

    let register_names = design
        .registers
        .iter()
        .map(|register| register.name.as_str())
        .collect::<Vec<&str>>();
    assert_eq!(design.registers.len(), 3);
    assert!(register_names.contains(&"p0_data"));
    assert!(register_names.contains(&"p1_sum"));
    assert!(register_names.contains(&"p2_result"));
}

#[test]
fn rejects_adjacent_stages_with_different_flat_widths() {
    let stages = [
        passthrough_stage("stage0", "x", "out", 2),
        passthrough_stage("stage1", "in", "out", 3),
    ];
    let error = stitch_gate_fns_into_pipeline(
        &stages,
        &SequentialPipelineOptions {
            name: "bad".to_string(),
            clock: ClockPort {
                name: "clk".to_string(),
            },
            flop_inputs: false,
            flop_outputs: false,
            input_valid_signal: None,
            output_valid_signal: None,
            reset_signal: None,
            reset_active_low: false,
        },
    )
    .unwrap_err();

    assert_eq!(
        error,
        "cannot stitch stages 0 and 1: output width 2 does not match input width 3"
    );
}

#[test]
fn rejects_clock_name_colliding_with_a_pipeline_input() {
    let stage = passthrough_stage("stage", "clk", "out", 1);
    let error = stitch_gate_fns_into_pipeline(
        &[stage],
        &SequentialPipelineOptions {
            name: "bad_clock".to_string(),
            clock: ClockPort {
                name: "clk".to_string(),
            },
            flop_inputs: true,
            flop_outputs: false,
            input_valid_signal: None,
            output_valid_signal: None,
            reset_signal: None,
            reset_active_low: false,
        },
    )
    .unwrap_err();

    assert_eq!(error, "clock name 'clk' collides with a stage input");
}

#[test]
fn valid_and_reset_add_load_enabled_data_and_valid_state() {
    let stages = [
        passthrough_stage("stage0", "data", "sum", 1),
        passthrough_stage("stage1", "sum", "result", 1),
    ];
    let design = stitch_gate_fns_into_pipeline(
        &stages,
        &SequentialPipelineOptions {
            name: "enabled".to_string(),
            clock: ClockPort {
                name: "clk".to_string(),
            },
            flop_inputs: true,
            flop_outputs: true,
            input_valid_signal: Some("in_valid".to_string()),
            output_valid_signal: Some("out_valid".to_string()),
            reset_signal: Some("rst".to_string()),
            reset_active_low: false,
        },
    )
    .unwrap();

    let register_names = design
        .registers
        .iter()
        .map(|register| register.name.as_str())
        .collect::<Vec<&str>>();
    assert_eq!(design.inputs.len(), 3);
    assert_eq!(design.outputs.len(), 2);
    assert_eq!(design.registers.len(), 6);
    assert!(register_names.contains(&"p0_in_valid"));
    assert!(register_names.contains(&"p1_in_valid"));
    assert!(register_names.contains(&"p2_in_valid"));
}

#[test]
fn stitches_a_deep_stage_without_recursive_graph_import() {
    let mut stage0 = GateBuilder::new("deep".to_string(), GateBuilderOptions::no_opt());
    let x = stage0.add_input("x".to_string(), 1);
    let mut value = *x.get_lsb(0);
    for _ in 0..20_000 {
        value = stage0.add_and_binary(value, *x.get_lsb(0));
    }
    stage0.add_output("value".to_string(), value.into());
    let stages = [
        stage0.build(),
        passthrough_stage("consumer", "value", "out", 1),
    ];

    let design = stitch_gate_fns_into_pipeline(
        &stages,
        &SequentialPipelineOptions {
            name: "deep_pipeline".to_string(),
            clock: ClockPort {
                name: "clk".to_string(),
            },
            flop_inputs: false,
            flop_outputs: false,
            input_valid_signal: None,
            output_valid_signal: None,
            reset_signal: None,
            reset_active_low: false,
        },
    )
    .unwrap();

    assert_eq!(design.registers.len(), 1);
}
