// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

fn make_timing_enabled_inv_liberty_textproto() -> &'static str {
    r#"
format_magic: 5496997758228255311
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT capacitance: 1.0 }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 3
    timing_arcs: {
      related_pin_string_id: 1
      timing_sense: TIMING_SENSE_NEGATIVE_UNATE
      timing_type: TIMING_TYPE_COMBINATIONAL
      tables: { kind: TIMING_TABLE_KIND_CELL_RISE shape_id: 1 values: 2.0 }
      tables: { kind: TIMING_TABLE_KIND_CELL_FALL shape_id: 1 values: 3.0 }
      tables: { kind: TIMING_TABLE_KIND_RISE_TRANSITION shape_id: 1 values: 0.2 }
      tables: { kind: TIMING_TABLE_KIND_FALL_TRANSITION shape_id: 1 values: 0.3 }
    }
  }
}
units: { time_unit: "1ps" capacitance_unit: "1pf" }
interned_strings: ["A", "Y", "!A"]
lut_shapes: {}
"#
}

#[test]
fn gv_sta_uses_selected_module_net_scope_when_names_repeat() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let netlist_path = temp_dir.path().join("multi.gv");
    let liberty_path = temp_dir.path().join("lib.textproto");

    std::fs::write(
        &netlist_path,
        r#"
module fast (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 ( .A(a), .Y(y) );
endmodule

module slow (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  INV u0 ( .A(a), .Y(n0) );
  INV u1 ( .A(n0), .Y(y) );
endmodule
"#,
    )
    .expect("write netlist");
    std::fs::write(&liberty_path, make_timing_enabled_inv_liberty_textproto())
        .expect("write liberty");

    let output = Command::new(driver)
        .arg("gv-sta")
        .arg("--netlist")
        .arg(netlist_path.as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_path.as_os_str())
        .arg("--module_name")
        .arg("fast")
        .output()
        .expect("gv-sta invocation should run");

    assert!(
        output.status.success(),
        "gv-sta failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("worst_output_arrival: 3.000000"),
        "unexpected stdout: {}",
        stdout
    );
    assert!(
        stdout.contains("output y rise_arrival=2.000000 fall_arrival=3.000000"),
        "unexpected stdout: {}",
        stdout
    );
}

#[test]
fn gv_sta_affine_wire_capacitance_ff_preserves_pin_and_output_loads() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");
    let dir = tempfile::tempdir().unwrap();
    let lib = dir.path().join("test.lib");
    let proto = dir.path().join("test.proto");
    let net = dir.path().join("test.v");
    std::fs::write(
        &lib,
        r#"
library(test) {
 time_unit : "1ns"; capacitive_load_unit(1,pf);
 lu_table_template(t) { variable_1: total_output_net_capacitance; index_1("0,4"); }
 cell(BUF) {
  area:1;
  pin(A) { direction:input; capacitance:1; }
  pin(Y) { direction:output; function:"A";
   timing() { related_pin:"A"; timing_sense:positive_unate;
    cell_rise(t) { values("0,4"); } cell_fall(t) { values("0,4"); }
    rise_transition(t) { values("0,4"); } fall_transition(t) { values("0,4"); }
   }
  }
 }
}
"#,
    )
    .unwrap();
    std::fs::write(&net, "module m(a,y); input a; output y; wire n; BUF a0(.A(a),.Y(n)); BUF a1(.A(n),.Y(y)); endmodule").unwrap();
    let import = Command::new(driver)
        .args(["lib2proto", "--include-timing", "--output"])
        .arg(&proto)
        .arg(&lib)
        .output()
        .unwrap();
    assert!(
        import.status.success(),
        "{}",
        String::from_utf8_lossy(&import.stderr)
    );
    let mut baseline = None;
    for (index, flags, expected) in [
        (0, vec![], 3.0),
        (
            1,
            vec![
                "--wire-capacitance-constant-ff=0",
                "--wire-capacitance-per-sink-ff=0",
            ],
            3.0,
        ),
        (2, vec!["--wire-capacitance-constant-ff=500"], 4.0),
        (3, vec!["--wire-capacitance-per-sink-ff=250"], 3.25),
        (
            4,
            vec![
                "--wire-capacitance-constant-ff=500",
                "--wire-capacitance-per-sink-ff=250",
            ],
            4.25,
        ),
    ] {
        let output = dir.path().join(format!("result{index}.json"));
        let run = Command::new(driver)
            .args(["gv-sta", "--netlist"])
            .arg(&net)
            .arg("--liberty_proto")
            .arg(&proto)
            .args(["--module_output_load=2", "--json_out"])
            .arg(&output)
            .args(flags)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let bytes = std::fs::read(&output).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["worst_output_arrival"].as_f64().unwrap(), expected);
        if index == 0 {
            baseline = Some(bytes);
        } else if index == 1 {
            assert_eq!(baseline.as_ref().unwrap(), &bytes);
        }
    }
    for flag in [
        "--wire-capacitance-constant-ff=-1",
        "--wire-capacitance-per-sink-ff=NaN",
        "--wire-capacitance-constant-ff=inf",
    ] {
        let run = Command::new(driver)
            .args(["gv-sta", "--netlist"])
            .arg(&net)
            .arg("--liberty_proto")
            .arg(&proto)
            .arg(flag)
            .output()
            .unwrap();
        assert!(!run.status.success());
    }
}
