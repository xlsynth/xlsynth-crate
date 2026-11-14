// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use xlsynth_test_helpers::compare_golden_text;

#[test]
fn gv_dump_cone_basic_csv_output() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    // Minimal Liberty textproto with a single inverter cell.
    let liberty_text = r#"
format_magic: 5496997758228255311
cells: {
  name: "INV"
  pins: {
    name_string_id: 1
    direction: INPUT
  }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 3
  }
  area: 1.0
}
interned_strings: ["A", "Y", "(!A)"]
"#;

    // Minimal netlist: a -> INV u1 -> n1 -> INV u2 -> y
    let netlist_text = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n1;
  INV u1 (.A(a), .Y(n1));
  INV u2 (.A(n1), .Y(y));
endmodule
"#;

    let mut liberty_file = tempfile::NamedTempFile::new().expect("create liberty temp file");
    std::io::Write::write_all(&mut liberty_file, liberty_text.as_bytes())
        .expect("write liberty text");

    let mut netlist_file = tempfile::NamedTempFile::new().expect("create netlist temp file");
    std::io::Write::write_all(&mut netlist_file, netlist_text.as_bytes())
        .expect("write netlist text");

    let output = Command::new(driver)
        .arg("gv-dump-cone")
        .arg(netlist_file.path().as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_file.path().as_os_str())
        .arg("--instance")
        .arg("u1")
        .arg("--traverse")
        .arg("fanout")
        .arg("--stop-at-levels")
        .arg("1")
        .output()
        .expect("gv-dump-cone invocation should run");

    assert!(
        output.status.success(),
        "gv-dump-cone failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let got = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");

    let want = "\
instance_type,instance_name,traversal_pin,levels
INV,u1,Y,0
INV,u2,A,1
";

    assert_eq!(got, want);
}

#[test]
fn gv_dump_cone_traverses_preserved_wiring_assigns() {
    let driver = env!("CARGO_BIN_EXE_xlsynth-driver");

    let liberty_text = r#"
format_magic: 5496997758228255311
cells: {
  name: "BUF"
  pins: {
    name_string_id: 1
    direction: INPUT
  }
  pins: {
    name_string_id: 2
    direction: OUTPUT
    function_string_id: 1
  }
  area: 1.0
}
interned_strings: ["A", "Y"]
"#;

    let netlist_text = r#"
module top (a, y);
  input a;
  output y;
  wire n0;
  wire n1;
  BUF u0 (.A(a), .Y(n0));
  assign n1 = n0;
  BUF u1 (.A(n1), .Y(y));
endmodule
"#;

    let mut liberty_file = tempfile::NamedTempFile::new().expect("create liberty temp file");
    std::io::Write::write_all(&mut liberty_file, liberty_text.as_bytes())
        .expect("write liberty text");

    let mut netlist_file = tempfile::NamedTempFile::new().expect("create netlist temp file");
    std::io::Write::write_all(&mut netlist_file, netlist_text.as_bytes())
        .expect("write netlist text");

    let output = Command::new(driver)
        .arg("gv-dump-cone")
        .arg(netlist_file.path().as_os_str())
        .arg("--liberty_proto")
        .arg(liberty_file.path().as_os_str())
        .arg("--instance")
        .arg("u0")
        .arg("--traverse")
        .arg("fanout")
        .arg("--stop-at-levels")
        .arg("1")
        .output()
        .expect("gv-dump-cone invocation should run");

    assert!(
        output.status.success(),
        "gv-dump-cone failed: status={:?}\nstdout={}\nstderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let got = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");

    let want = "\
instance_type,instance_name,traversal_pin,levels
BUF,u0,Y,0
BUF,u1,A,1
";

    assert_eq!(got, want);
}

#[test]
fn gv_dump_cone_stops_at_liberty_flip_flops_unless_overridden() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let liberty_path = temp_dir.path().join("cells.textproto");
    std::fs::write(
        &liberty_path,
        r#"
format_magic: 5496997758228255311
interned_strings: ["A", "Y", "D", "Q", "CLK", "EN", "IQ"]
cells: {
  name: "BUF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 1 }
}
cells: {
  name: "LATCH"
  pins: { name_string_id: 3 direction: INPUT }
  pins: { name_string_id: 4 direction: OUTPUT function_string_id: 7 }
  pins: { name_string_id: 6 direction: INPUT is_clocking_pin: true }
  sequential: { state_var: "IQ" next_state: "D" clock_expr: "EN" kind: SEQUENTIAL_KIND_LATCH }
}
cells: {
  name: "REG"
  pins: { name_string_id: 3 direction: INPUT }
  pins: { name_string_id: 4 direction: OUTPUT function_string_id: 7 }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  sequential: { state_var: "IQ" next_state: "D" clock_expr: "CLK" kind: SEQUENTIAL_KIND_FF }
}
"#,
    )
    .expect("write Liberty textproto");
    let netlist_path = temp_dir.path().join("mapped.gv");
    std::fs::write(
        &netlist_path,
        r#"
module top (a, en, clk, y);
  input a;
  input en;
  input clk;
  output y;
  wire n0;
  wire n1;
  wire n2;
  wire n3;
  BUF start (.A(a), .Y(n0));
  LATCH latch0 (.D(n0), .EN(en), .Q(n1));
  REG ff0 (.D(n1), .CLK(clk), .Q(n2));
  BUF after_ff (.A(n2), .Y(n3));
  LATCH latch1 (.D(n3), .EN(en), .Q(y));
endmodule
"#,
    )
    .expect("write netlist");

    let run_cone = |override_cells: Option<&str>| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"));
        command
            .env_remove("CLICOLOR_FORCE")
            .arg("gv-dump-cone")
            .arg(&netlist_path)
            .arg("--liberty_proto")
            .arg(&liberty_path)
            .args([
                "--instance",
                "start",
                "--traverse",
                "fanout",
                "--stop-at-dff",
            ]);
        if let Some(override_cells) = override_cells {
            command.args(["--dff_cells", override_cells]);
        }
        command.output().expect("run gv-dump-cone")
    };
    for (override_cells, golden_path) in [
        (None, "tests/goldens/gv_dump_cone_liberty_dff.csv"),
        (Some("BUF"), "tests/goldens/gv_dump_cone_dff_override.csv"),
    ] {
        let output = run_cone(override_cells);
        assert!(
            output.status.success(),
            "gv-dump-cone failed: {}",
            String::from_utf8_lossy(&output.stderr),
        );
        let got = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
        compare_golden_text(&got, golden_path);
    }

    let output = run_cone(Some(","));
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        "xlsynth-driver: --dff-cells must contain one or more cell names when supplied with --stop-at-dff\n"
    );
}
