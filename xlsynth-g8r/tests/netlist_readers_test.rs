// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::{
    read_gv_for_roundtrip_from_path, read_gv_from_str, read_ugv_from_str,
};

const MIXED_NETLIST: &str = r#"module top(a, y);
  input a;
  output y;
  wire n;
  assign n = ~a;
  BUF u_buf (.A(n), .Y(y));
endmodule
"#;

#[test]
fn readers_accept_ansi_ports_and_shared_wiring() {
    let source = r#"module top(input wire [1:0] a, output wire [1:0] y);
  assign y = a;
endmodule
"#;
    for parsed in [read_gv_from_str(source), read_ugv_from_str(source)] {
        let parsed = parsed.unwrap();
        assert_eq!(parsed.modules.len(), 1);
        let ports = &parsed.modules[0].ports;
        assert_eq!(ports.len(), 2);
        assert_eq!(parsed.interner.resolve(ports[0].name), Some("a"));
        assert_eq!(parsed.interner.resolve(ports[1].name), Some("y"));
        assert_eq!(ports[0].width, Some((1, 0)));
        assert_eq!(ports[1].width, Some((1, 0)));
    }
}

#[test]
fn readers_preserve_escaped_keyword_identifiers() {
    let source = r#"module top(input wire \reg , output wire \always_ff );
  assign \always_ff  = \reg ;
endmodule
"#;
    for parsed in [read_gv_from_str(source), read_ugv_from_str(source)] {
        let parsed = parsed.unwrap();
        let ports = &parsed.modules[0].ports;
        assert_eq!(parsed.interner.resolve(ports[0].name), Some("reg"));
        assert_eq!(parsed.interner.resolve(ports[1].name), Some("always_ff"));
    }
}

#[test]
fn gv_reader_rejects_logic_assignments_in_mixed_input() {
    let error = read_gv_from_str(MIXED_NETLIST)
        .err()
        .expect("mixed GV must fail");
    assert_eq!(
        error.to_string(),
        include_str!("goldens/gv_logic_assignment.golden.txt").trim_end()
    );
}

#[test]
fn ugv_reader_rejects_leaf_cells_in_mixed_input() {
    let error = read_ugv_from_str(MIXED_NETLIST)
        .err()
        .expect("mixed UGV must fail");
    assert_eq!(
        error.to_string(),
        include_str!("goldens/ugv_leaf_cell.golden.txt").trim_end()
    );
}

#[test]
fn ugv_reader_distinguishes_helper_modules_from_leaf_cells() {
    let source = r#"module child(a, y);
  input a;
  output y;
  assign y = ~a;
endmodule
module top(a, y);
  input a;
  output y;
  child u_child (.a(a), .y(y));
endmodule
"#;
    let parsed = read_ugv_from_str(source).unwrap();
    assert_eq!(parsed.modules.len(), 2);
}

#[test]
fn ugv_reader_reports_procedural_registers() {
    let source = r#"module top(clk, a, y);
  input clk;
  input a;
  output y;
  always_ff @ (posedge clk) y <= a;
endmodule
"#;
    let error = read_ugv_from_str(source)
        .err()
        .expect("procedural UGV must fail");
    assert_eq!(
        error.to_string(),
        include_str!("goldens/ugv_procedural.golden.txt").trim_end()
    );
}

#[test]
fn ugv_reader_reports_unsupported_arithmetic() {
    let source = r#"module top(a, b, y);
  input a;
  input b;
  output y;
  assign y = a + b;
endmodule
"#;
    let error = read_ugv_from_str(source)
        .err()
        .expect("arithmetic UGV must fail");
    assert_eq!(
        error.to_string(),
        include_str!("goldens/ugv_arithmetic.golden.txt").trim_end()
    );
}

#[test]
fn roundtrip_reader_rejects_non_ansi_slots_that_would_be_lost() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("ports.gv");
    for (header, declarations) in [
        ("a,", "input a;"),
        ("a,a", "input a;"),
        ("a,b", "input a;"),
        ("a", "input a,b;"),
        ("a,a", "input a; input a;"),
    ] {
        let source = format!("module top({header}); {declarations} endmodule");
        std::fs::write(&path, &source)?;
        assert!(read_gv_for_roundtrip_from_path(&path).is_err(), "{source}");
        read_gv_from_str(&source)?;
    }
    Ok(())
}

#[test]
fn escaped_statement_keywords_roundtrip_as_cell_types() -> Result<()> {
    let source = r#"module top();
  \assign  u_assign ();
  \tran  u_tran ();
endmodule
"#;
    let parsed = read_gv_from_str(source)?;
    let module = &parsed.modules[0];
    assert_eq!(module.instances.len(), 2);
    assert_eq!(
        parsed.interner.resolve(module.instances[0].type_name),
        Some("assign")
    );
    assert_eq!(
        parsed.interner.resolve(module.instances[1].type_name),
        Some("tran")
    );
    assert_eq!(
        emit_module_as_netlist_text(module, &parsed.nets, &parsed.interner)?,
        source
    );
    Ok(())
}
