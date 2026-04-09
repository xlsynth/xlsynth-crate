// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::process::Command;
use std::process::Output;

use xlsynth_g8r::aig_serdes::load_aiger_auto::load_aiger_auto_from_path;
use xlsynth_g8r::gate_builder::GateBuilderOptions;

fn run_gv2aig(
    netlist_text: &str,
    liberty_text: Option<&str>,
) -> (tempfile::TempDir, PathBuf, Output) {
    run_gv2aig_with_module_name(netlist_text, liberty_text, None)
}

fn run_gv2aig_with_module_name(
    netlist_text: &str,
    liberty_text: Option<&str>,
    module_name: Option<&str>,
) -> (tempfile::TempDir, PathBuf, Output) {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("netlist.v");
    let out_path = temp_dir.path().join("out.aag");

    std::fs::write(&netlist_path, netlist_text).expect("write netlist");

    let mut command = Command::new(driver);
    command
        .arg("gv2aig")
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

    let output = command.output().expect("gv2aig invocation should run");
    (temp_dir, out_path, output)
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "gv2aig failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn gv2aig_emits_parseable_aiger() {
    let liberty_text = r#"
cells: {
  name: "INV"
  pins: { name: "A" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "(!A)" }
  area: 1.0
}
cells: {
  name: "AND2"
  pins: { name: "A" direction: INPUT }
  pins: { name: "B" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "(A & B)" }
  area: 1.0
}
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

    let (_temp_dir, out_path, output) = run_gv2aig(netlist_text, Some(liberty_text));
    assert_success(&output);

    let loaded = load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load liberty-backed aiger");
    assert!(
        !loaded.gate_fn.gates.is_empty(),
        "expected non-empty GateFn"
    );
}

#[test]
fn gv2aig_without_liberty_emits_parseable_aiger_for_structural_assigns() {
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

    let (_temp_dir, out_path, output) = run_gv2aig(netlist_text, None);
    assert_success(&output);

    let loaded = load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load structural assign aiger");
    assert!(
        !loaded.gate_fn.gates.is_empty(),
        "expected non-empty GateFn"
    );
}

#[test]
fn gv2aig_without_liberty_supports_vector_xor() {
    let netlist_text = r#"
module top(a, b, y);
  input [1:0] a;
  input [1:0] b;
  output [1:0] y;
  assign y = a ^ b;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_gv2aig(netlist_text, None);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load vector xor aiger");
}

#[test]
fn gv2aig_without_liberty_supports_bus_slice_assembly() {
    let netlist_text = r#"
module top(lo, hi, y);
  input [1:0] lo;
  input [1:0] hi;
  output [3:0] y;
  assign y[1:0] = lo;
  assign y[3:2] = hi;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_gv2aig(netlist_text, None);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load bus slice assembly aiger");
}

#[test]
fn gv2aig_without_liberty_supports_acyclic_overlapping_slice_dependencies() {
    let netlist_text = r#"
module top(a, y);
  input a;
  output [3:0] y;
  assign y[3:1] = y[2:0];
  assign y[0] = a;
endmodule
"#;

    let (_temp_dir, out_path, output) = run_gv2aig(netlist_text, None);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load overlapping-slice structural aiger");
}

#[test]
fn gv2aig_without_liberty_supports_ascending_packed_range_selects() {
    let netlist_text = r#"
module top(a, y);
  input [0:3] a;
  output [0:3] y;
  assign y[0:2] = a[0:2];
  assign y[3] = a[3];
endmodule
"#;

    let (_temp_dir, out_path, output) = run_gv2aig(netlist_text, None);
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load ascending-range structural aiger");
}

#[test]
fn gv2aig_without_liberty_scopes_port_lookup_to_selected_module() {
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

    let (_temp_dir, out_path, output) =
        run_gv2aig_with_module_name(netlist_text, None, Some("top"));
    assert_success(&output);
    load_aiger_auto_from_path(&out_path, GateBuilderOptions::no_opt())
        .expect("load selected-module structural aiger");
}

#[test]
fn gv2aig_without_liberty_rejects_cycles() {
    let netlist_text = r#"
module top(a, y);
  input a;
  output y;
  wire n;
  assign n = y;
  assign y = n;
endmodule
"#;

    let (_temp_dir, _out_path, output) = run_gv2aig(netlist_text, None);
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
fn gv2aig_with_liberty_rejects_preserved_assigns() {
    let liberty_text = r#"
cells: {
  name: "BUF"
  pins: { name: "A" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "A" }
  area: 1.0
}
"#;
    let netlist_text = r#"
module top(a, y);
  input a;
  output y;
  assign y = a;
endmodule
"#;

    let (_temp_dir, _out_path, output) = run_gv2aig(netlist_text, Some(liberty_text));
    assert!(
        !output.status.success(),
        "preserved assigns with liberty should fail\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("does not support preserved continuous assigns"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );
}
