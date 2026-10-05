// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::Command;
use std::process::Output;

use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::test_utils::structurally_equivalent;
use xlsynth_pir::IrBits;

fn run_gv2aig(netlist_text: &str, liberty_text: &str) -> (tempfile::TempDir, PathBuf, Output) {
    run_netlist2aig("gv2aig", netlist_text, Some(liberty_text), None)
}

fn run_ugv2aig(netlist_text: &str) -> (tempfile::TempDir, PathBuf, Output) {
    run_netlist2aig("ugv2aig", netlist_text, None, None)
}

fn run_netlist2aig(
    command_name: &str,
    netlist_text: &str,
    liberty_text: Option<&str>,
    module_name: Option<&str>,
) -> (tempfile::TempDir, PathBuf, Output) {
    run_netlist2aig_with_options(command_name, netlist_text, liberty_text, module_name, &[])
}

fn run_netlist2aig_with_options(
    command_name: &str,
    netlist_text: &str,
    liberty_text: Option<&str>,
    module_name: Option<&str>,
    extra_args: &[&str],
) -> (tempfile::TempDir, PathBuf, Output) {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("netlist.v");
    let out_path = temp_dir.path().join("out.aag");

    std::fs::write(&netlist_path, netlist_text).expect("write netlist");

    let mut command = Command::new(driver);
    command
        .arg(command_name)
        .arg("--netlist")
        .arg(netlist_path.as_os_str())
        .arg("--aiger-out")
        .arg(out_path.as_os_str());

    if let Some(module_name) = module_name {
        command.arg("--module_name").arg(module_name);
    }

    if let Some(liberty_text) = liberty_text {
        let liberty_path = temp_dir.path().join("lib.textproto");
        std::fs::write(&liberty_path, liberty_text).expect("write liberty");
        command.arg("--liberty_proto").arg(liberty_path.as_os_str());
    }
    command.args(extra_args);

    let output = command
        .output()
        .expect("netlist-to-AIG invocation should run");
    (temp_dir, out_path, output)
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "netlist-to-AIG failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn gv2aig_emits_parseable_aiger() {
    let liberty_text = r#"
format_magic: 5496997758177923663
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  area: 1.0
}
cells: {
  name: "AND2"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 5 }
  area: 1.0
}
interned_strings: ["A", "B", "Y", "(!A)", "(A & B)"]
"#;
    let netlist_text = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  wire n1;
  AND2 u1 (.A(a), .B(b), .Y(n1));
  INV u2 (.A(n1), .Y(y));
endmodule
"#;

    let (_temp_dir, out_path, output) = run_gv2aig(netlist_text, liberty_text);
    assert_success(&output);

    let loaded = load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load liberty-backed aiger");
    assert!(
        !loaded.gate_fn.gates.is_empty(),
        "expected non-empty GateFn"
    );
}

#[test]
fn gv2aig_load_enable_feedback_requires_opt_in() {
    let liberty_text = r#"
format_magic: 5496997758177923663
cells: {
  name: "DFFEN"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT }
  pins: { name_string_id: 3 direction: INPUT }
  pins: { name_string_id: 4 direction: INPUT }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 6 direction: OUTPUT function_string_id: 7 }
  sequential: {
    state_var: "S"
    next_state: "(PRIOR & HOLD) | (DATA & LOAD)"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
cells: {
  name: "INV"
  pins: { name_string_id: 8 direction: INPUT }
  pins: { name_string_id: 9 direction: OUTPUT function_string_id: 10 }
}
interned_strings: ["DATA", "LOAD", "HOLD", "PRIOR", "CLK", "Q", "S", "A", "Y", "(!A)"]
"#;
    let netlist_text = r#"
module top(data, enable, clk, y);
  input data;
  input enable;
  input clk;
  output y;
  wire hold;
  INV u_hold (.A(enable), .Y(hold));
  DFFEN u_pipe (.DATA(data), .LOAD(enable), .HOLD(hold), .PRIOR(y),
                 .CLK(clk), .Q(y));
endmodule
"#;

    let (_default_dir, _default_path, default_output) = run_gv2aig(netlist_text, liberty_text);
    assert!(!default_output.status.success());
    assert_eq!(
        String::from_utf8_lossy(&default_output.stderr),
        include_str!("golden/gv2aig_load_enable_feedback.golden.txt")
    );

    let (_enabled_dir, enabled_path, enabled_output) = run_netlist2aig_with_options(
        "gv2aig",
        netlist_text,
        Some(liberty_text),
        None,
        &["--collapse_load_enable_feedback"],
    );
    assert_success(&enabled_output);
    let loaded = load_aiger_auto_from_path(&enabled_path, GateBuilderOptions::no_opt())
        .expect("load AIGER with collapsed load-enable feedback");
    let mut sim = PreparedGateSim::new(&loaded.gate_fn);
    for data in 0..=1 {
        for enable in 0..=1 {
            let outputs = sim.eval_outputs(&[
                IrBits::make_ubits(1, data).unwrap(),
                IrBits::make_ubits(1, enable).unwrap(),
                IrBits::make_ubits(1, 0).unwrap(),
            ]);
            assert_eq!(outputs, vec![IrBits::make_ubits(1, data & enable).unwrap()]);
        }
    }
}

#[test]
fn ugv2aig_emits_parseable_aiger_for_structural_assigns() {
    let netlist_text = r#"
module top(a, b, y);
  input a;
  input b;
  output y;
  wire n;
  assign n = a & b;
  assign y = ~n;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_ugv2aig(netlist_text);
    assert_success(&output);

    let loaded = load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load structural assign aiger");
    assert!(
        !loaded.gate_fn.gates.is_empty(),
        "expected non-empty GateFn"
    );
}

#[test]
fn ugv2aig_supports_vector_xor() {
    let netlist_text = r#"
module top(a, b, y);
  input [1:0] a;
  input [1:0] b;
  output [1:0] y;
  assign y = a ^ b;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_ugv2aig(netlist_text);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load vector xor aiger");
}

#[test]
fn ugv2aig_supports_bus_slice_assembly() {
    let netlist_text = r#"
module top(lo, hi, y);
  input [1:0] lo;
  input [1:0] hi;
  output [3:0] y;
  assign y[1:0] = lo;
  assign y[3:2] = hi;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_ugv2aig(netlist_text);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load bus slice assembly aiger");
}

#[test]
fn ugv2aig_supports_bare_literal_tieoff_resize() {
    let netlist_text = r#"
module top(y);
  output [3:0] y;
  assign y = 1'b0;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_ugv2aig(netlist_text);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load literal tie-off structural aiger");
}

#[test]
fn ugv2aig_supports_declarations_after_assigns() {
    let netlist_text = r#"
module top(a, y);
  assign y[3:0] = a;
  input [3:0] a;
  output [3:0] y;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_ugv2aig(netlist_text);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load declarations-after-assigns structural aiger");
}

#[test]
fn ugv2aig_supports_acyclic_overlapping_slice_dependencies() {
    let netlist_text = r#"
module top(a, y);
  input a;
  output [3:0] y;
  assign y[3:1] = y[2:0];
  assign y[0] = a;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_ugv2aig(netlist_text);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load overlapping-slice structural aiger");
}

#[test]
fn ugv2aig_supports_ascending_packed_range_selects() {
    let netlist_text = r#"
module top(a, y);
  input [0:3] a;
  output [0:3] y;
  assign y[0:2] = a[0:2];
  assign y[3] = a[3];
endmodule
"#;

    let (_temp_dir, out_path, output) = run_ugv2aig(netlist_text);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load ascending-range structural aiger");
}

#[test]
fn ugv2aig_rejects_mixed_width_bitwise_ops() {
    let netlist_text = r#"
module top(a, y);
  input [3:0] a;
  output [3:0] y;
  assign y = a & 1'b1;
endmodule
"#;

    let (_temp_dir, _out_path, output) = run_ugv2aig(netlist_text);
    assert!(
        !output.status.success(),
        "mixed-width bitwise ops should fail\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("exact-width bitwise operands"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn ugv2aig_scopes_port_lookup_to_selected_module() {
    let netlist_text = r#"
module helper(a, y);
  input a;
  output y;
  assign y = a;
endmodule

module top(a, y);
  input [1:0] a;
  output [1:0] y;
  assign y = a;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_netlist2aig("ugv2aig", netlist_text, None, Some("top"));
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load selected-module structural aiger");
}

#[test]
fn ugv2aig_rejects_cycles() {
    let netlist_text = r#"
module top(a, y);
  input a;
  output y;
  wire n;
  assign n = y;
  assign y = n;
endmodule
"#;

    let (_temp_dir, _out_path, output) = run_ugv2aig(netlist_text);
    assert!(
        !output.status.success(),
        "cycle should fail\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("dependency cycle") || stderr.contains("unresolved recursion"),
        "unexpected stderr: {}",
        stderr
    );
}

#[test]
fn gv2aig_supports_preserved_wiring_assigns() {
    let liberty_text = r#"
format_magic: 5496997758177923663
cells: {
  name: "BUF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 1 }
  area: 1.0
}

interned_strings: ["A", "Y"]
"#;
    let netlist_text = r#"
module top(a, y);
  input [1:0] a;
  output y;
  wire [1:0] tmp;
  assign tmp = {a[0], a[1]};
  BUF u0 (.A(tmp[1]), .Y(y));
endmodule
"#;

    let (_temp_dir, out_path, output) = run_gv2aig(netlist_text, liberty_text);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load liberty-backed preserved assign aiger");
}

#[test]
fn gv2aig_requires_liberty() {
    let source = "module top(a, y); input a; output y; assign y = a; endmodule";
    let (_dir, _path, output) = run_netlist2aig("gv2aig", source, None, None);
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        include_str!("goldens/gv2aig_requires_liberty.golden.txt")
    );
}

#[test]
fn aig2ugv_output_roundtrips_through_ugv2aig() {
    let dir = tempfile::tempdir().unwrap();
    let input_path = dir.path().join("input.aag");
    std::fs::write(
        &input_path,
        r#"aag 3 2 0 1 1
2
4
6
6 2 5
i0 a
i1 b
o0 y
c
"#,
    )
    .unwrap();
    let emitted = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("aig2ugv")
        .arg(&input_path)
        .args(["--module-name", "top"])
        .output()
        .unwrap();
    assert_success(&emitted);
    let ugv = String::from_utf8(emitted.stdout).unwrap();
    let (_output_dir, output_path, output) = run_ugv2aig(&ugv);
    assert_success(&output);
    let original = load_aiger_auto_from_path(&input_path, GateBuilderOptions::no_opt()).unwrap();
    let rebuilt = load_aiger_auto_from_path(&output_path, GateBuilderOptions::no_opt()).unwrap();
    assert!(structurally_equivalent(&original.gate_fn, &rebuilt.gate_fn));
}
