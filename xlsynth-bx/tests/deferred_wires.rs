// SPDX-License-Identifier: Apache-2.0

mod support;

use std::error::Error;
use std::fs;
use std::path::Path;
use support::search_paths;

use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_source_file_with_options, compile, empty_package,
    lower_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

const FLOW_STAGES: &str = r#"
struct FlowOut<T> { push_ready: bool, pop_valid: bool, pop_data: T }

block flow_reg_rev<T>(clk, rst, push_valid: bool, push_data: T, pop_ready: bool) -> FlowOut<T> {
    reg saved_valid: bool = false;
    reg saved_data: T = zero!<T>();
    let hold = !saved_valid && push_valid && !pop_ready;
    saved_valid := (saved_valid && !pop_ready) || hold;
    if hold { saved_data := push_data; }
    FlowOut {
        push_ready: !saved_valid,
        pop_valid: saved_valid || push_valid,
        pop_data: if saved_valid { saved_data } else { push_data },
    }
}

block flow_reg_fwd<T>(clk, rst, push_valid: bool, push_data: T, pop_ready: bool) -> FlowOut<T> {
    reg valid_q: bool = false;
    reg data_q: T = zero!<T>();
    let push_ready = !valid_q || pop_ready;
    if push_ready { valid_q := push_valid; }
    if push_ready && push_valid { data_q := push_data; }
    FlowOut { push_ready, pop_valid: valid_q, pop_data: data_q }
}
"#;

const DEFERRED_FLOW: &str = r#"
block flow_reg_both<T>(clk, rst, push_valid: bool, push_data: T, pop_ready: bool) -> FlowOut<T> {
    let mid_ready: bool;
    let rev = flow_reg_rev(push_valid, push_data, mid_ready);
    let fwd = flow_reg_fwd(rev.pop_valid, rev.pop_data, pop_ready);
    mid_ready = fwd.push_ready;
    FlowOut { push_ready: rev.push_ready, pop_valid: fwd.pop_valid, pop_data: fwd.pop_data }
}

pub block pipeline(clk, rst, push_valid: bool, push_data: u8, pop_ready: bool) -> FlowOut<u8> {
    let flow = flow_reg_both(push_valid, push_data, pop_ready);
    flow
}
"#;

const EXPLICIT_FLOW: &str = r#"
block rev_byte(clk, rst, push_valid: bool, push_data: u8, pop_ready: bool) -> FlowOut<u8> {
    let stage = flow_reg_rev(push_valid, push_data, pop_ready);
    stage
}

block fwd_byte(clk, rst, push_valid: bool, push_data: u8, pop_ready: bool) -> FlowOut<u8> {
    let stage = flow_reg_fwd(push_valid, push_data, pop_ready);
    stage
}

pub block pipeline(clk, rst, push_valid: bool, push_data: u8, pop_ready: bool) -> FlowOut<u8> {
    inst rev: rev_byte;
    inst fwd: fwd_byte;
    rev.push_valid := push_valid;
    rev.push_data := push_data;
    rev.pop_ready := fwd.push_ready;
    fwd.push_valid := rev.pop_valid;
    fwd.push_data := rev.pop_data;
    fwd.pop_ready := pop_ready;
    FlowOut { push_ready: rev.push_ready, pop_valid: fwd.pop_valid, pop_data: fwd.pop_data }
}
"#;

/// Simulate a two-entry flow register through fill, stall, drain, and reset.
fn run_flow(source: &str) -> Result<Vec<[u64; 3]>, Box<dyn Error>> {
    let mut package = compile(
        &format!("{FLOW_STAGES}{source}"),
        empty_package("flow"),
        ResetBehavior::default(),
    )?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    inline_all_blocks_in_package(&mut package)?;
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default())?;
    let module = compile_pipeline_module(&rtl.system_verilog).unwrap();
    let mut runner = FixtureRunner::new(&module).unwrap();
    let ports = runner.port_bindings();
    let reset = ports.input("rst").unwrap();
    let valid = ports.input("push_valid").unwrap();
    let data = ports.input("push_data").unwrap();
    let ready = ports.input("pop_ready").unwrap();
    let mut trace = Vec::new();
    for (rst, push_valid, push_data, pop_ready) in [
        (true, false, 0, false),
        (false, true, 11, false),
        (false, true, 22, false),
        (false, true, 33, false),
        (false, true, 33, true),
        (false, true, 33, true),
        (false, false, 0, true),
        (true, true, 44, false),
    ] {
        let cycle = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_bool(&reset, rst)?;
                drive.set_bool(&valid, push_valid)?;
                drive.set_u64(&data, push_data)?;
                drive.set_bool(&ready, pop_ready)
            })
            .unwrap();
        trace.push(["push_ready", "pop_valid", "pop_data"].map(|port| {
            cycle.outputs[port]
                .to_u64_if_known()
                .expect("output is known after reset")
        }));
    }
    Ok(trace)
}

#[test]
fn deferred_ready_wire_matches_explicit_instances_under_backpressure() -> Result<(), Box<dyn Error>>
{
    let reference = run_flow(EXPLICIT_FLOW)?;
    assert_eq!(
        reference,
        [
            [1, 0, 0],
            [1, 1, 11],
            [0, 1, 11],
            [0, 1, 11],
            [1, 1, 22],
            [1, 1, 33],
            [1, 0, 33],
            [1, 0, 0],
        ]
    );
    assert_eq!(run_flow(DEFERRED_FLOW)?, reference);
    Ok(())
}

#[test]
fn deferred_aggregate_wires_can_drive_earlier_expressions() -> Result<(), Box<dyn Error>> {
    let source = r#"
#[allow(combinational_block)]
pub block select_pair(clk, choose: bool, left: (u8, u16), right: (u8, u16)) -> (u8, u16) {
    let pending: (u8, u16);
    let incremented = pending.0 + u8:1;
    pending = if choose { left } else { right };
    (incremented, pending.1)
}
"#;
    let package = compile(source, empty_package("pairs"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    Ok(())
}

#[test]
fn deferred_wires_require_one_driver_of_the_declared_type() {
    for (body, expected) in [
        ("let signal: u8; signal", "wire 'signal' has no driver"),
        (
            "let signal: u8; signal = u8:1; signal = u8:2; signal",
            "multiple drivers for wire 'signal'",
        ),
        (
            "let signal: u8; signal = u16:1; signal",
            "driver of wire 'signal': expected u8, got u16",
        ),
        (
            "let signal: u8; if true { signal = u8:1; } signal",
            "conditional driver for wire 'signal'",
        ),
        (
            "if true { let signal: u8; } u8:0",
            "conditional declaration of wire 'signal'",
        ),
        (
            "let signal: u8; signal := u8:1; signal",
            "wire 'signal' must be driven with '='",
        ),
        (
            "let signal = u8:0; signal = u8:1; signal",
            "'signal' is not a declared wire",
        ),
    ] {
        let source = format!("block sample(clk) -> u8 {{ {body} }}");
        let error = compile(&source, empty_package("wires"), ResetBehavior::default()).unwrap_err();
        assert_eq!(error, format!("block 'sample': {expected}"), "{body}");
    }
}

#[test]
fn combinational_feedback_reports_driver_and_child_call_spans() {
    let source = r#"block pass_through<T>(clk, data: T) -> T { data }

pub block cycle(clk) -> u8 {
    let feedback: u8;
    let forwarded = pass_through(feedback);
    feedback = forwarded;
    forwarded
}
"#;
    let checked =
        check_source_file_with_options(source, Path::new("feedback.bx"), &ImportOptions::default())
            .unwrap();
    let error = lower_module(
        &checked,
        empty_package("feedback"),
        ResetBehavior::default(),
    )
    .unwrap_err();
    let expected = r#"error: combinational cycle in block 'cycle'
 --> feedback.bx:6:5
  |
5 |     let forwarded = pass_through(feedback);
  |     --------------------------------------- `forwarded.out` depends on `forwarded.data`
6 |     feedback = forwarded;
  |     ^^^^^^^^^^^^^^^^^^^^^ driver of `feedback`
  |
  = note: dependency cycle: forwarded.out -> forwarded.data -> feedback -> forwarded.out"#;
    assert_eq!(error, expected);
}

#[test]
fn imported_generic_block_cycle_reports_its_library_name_and_spans() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let stage = dir.path().join("stage.bx");
    fs::write(
        &stage,
        r#"block pass<T>(clk, value: T) -> T {
    value
}

pub block feedback<T>(clk, unused: T) -> T {
    let wire: T;
    let result = pass(wire); // keep $bx$foo$bar
    wire = result;
    result
}
"#,
    )?;
    let source = r#"use stage::feedback;

pub block wrapper(clk, input: u8) -> u8 {
    let result = feedback(input);
    result
}
"#;
    let options = search_paths(&[dir.path()]);
    let checked = check_source_file_with_options(source, &dir.path().join("design.bx"), &options)?;
    let error =
        lower_module(&checked, empty_package("design"), ResetBehavior::default()).unwrap_err();
    let expected = r#"error: combinational cycle in block 'stage::feedback'
 --> <stage>:8:5
  |
7 |     let result = pass(wire); // keep $bx$foo$bar
  |     ------------------------ `result.out` depends on `result.value`
8 |     wire = result;
  |     ^^^^^^^^^^^^^^ driver of `wire`
  |
  = note: dependency cycle: result.out -> result.value -> wire -> result.out"#
        .replace("<stage>", &stage.display().to_string());
    assert_eq!(error, expected);
    Ok(())
}

#[test]
fn mutually_driven_wire_aliases_are_rejected() {
    let expected = r#"error: combinational cycle in block 'cycle'
 --> <bx>:4:5
  |
4 |     first = second;
  |     ^^^^^^^^^^^^^^^ driver of `first`
5 |     second = first;
  |     --------------- driver of `second`
  |
  = note: dependency cycle: first -> second -> first"#;
    for (annotation, output) in [("", "u8:0"), (" @0", "first")] {
        let source = format!(
            r#"pub block cycle(clk) -> u8{annotation} {{
    let first: u8;
    let second: u8;
    first = second;
    second = first;
    {output}
}}
"#
        );
        let error =
            compile(&source, empty_package("aliases"), ResetBehavior::default()).unwrap_err();
        assert_eq!(error, expected);
    }
}

#[test]
fn deferred_wire_output_cycle_follows_its_driver() -> Result<(), Box<dyn Error>> {
    let check_cycle = |cycle| {
        let source = format!(
            r#"pub block delayed(clk, x: u8) -> u8 @{cycle} {{
    reg q: u8;
    let pending: u8;
    let out = pending;
    pending = q;
    q := x;
    out
}}
"#
        );
        compile(&source, empty_package("delay"), ResetBehavior::default())
    };
    check_cycle(1)?;
    assert_eq!(
        check_cycle(0).unwrap_err(),
        "block 'delayed': expected cycle 0, got cycle 1"
    );
    Ok(())
}
