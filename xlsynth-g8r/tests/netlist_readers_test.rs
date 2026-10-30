// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::{
    read_gv_for_proof_from_path, read_gv_for_roundtrip_from_path, read_gv_from_str,
    read_ugv_from_str,
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
fn proof_gv_reader_requires_sized_nonempty_concatenation_operands() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("concat.gv");
    for (expression, diagnostic) in [
        (
            "{0,x}",
            "unsized decimal in concatenation requires an explicit literal width",
        ),
        (
            "{x,{1,x}}",
            "unsized decimal in concatenation requires an explicit literal width",
        ),
        ("{x,}", "expected a concatenation operand before '}'"),
        ("{}", "expected a concatenation operand before '}'"),
    ] {
        std::fs::write(
            &path,
            format!("module top(input x, output [32:0] y);\nassign y = {expression};\nendmodule"),
        )?;
        let error = read_gv_for_proof_from_path(&path)
            .err()
            .expect("invalid concatenation");
        let message = error.to_string();
        assert!(
            message.starts_with(&format!("GV reader: {diagnostic} @ {}:2:", path.display())),
            "{message}"
        );
    }
    std::fs::write(
        &path,
        "module top(input x, output [32:0] y); assign y = {32'd0, {x}}; endmodule",
    )?;
    read_gv_for_proof_from_path(&path)?;
    Ok(())
}

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
    for newline in ["\n", "\r", "\r\n"] {
        let source = MIXED_NETLIST.replace('\n', newline);
        let error = read_gv_from_str(&source).err().expect("mixed GV must fail");
        assert_eq!(
            error.to_string(),
            include_str!("goldens/gv_logic_assignment.golden.txt").trim_end()
        );
    }
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

#[test]
fn roundtrip_reader_requires_a_declaration_site_for_each_net() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("implicit.gv");
    for body in [
        "assign y = misspelled;",
        "assign y = {misspelled, also_absent};",
        "assign y = misspelled[0];",
        "BUF inst(.A(misspelled[0]), .Y(y));",
    ] {
        // A declaration in a preceding module must not leak into this one.
        let source = format!(
            "module other(); wire misspelled; endmodule\nmodule top(output y); {body} endmodule"
        );
        std::fs::write(&path, &source)?;
        let error = read_gv_for_roundtrip_from_path(&path)
            .err()
            .expect("missing declaration site");
        assert_eq!(
            error.to_string().split(" @ ").next().unwrap(),
            "GV reader: net 'misspelled' has no declaration or implicit declaration site in this module"
        );
        read_gv_from_str(&source)?;
    }
    Ok(())
}

#[test]
fn proof_gv_reader_rejects_part_selects_opposite_to_the_declaration() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("select.gv");
    for (statement, select, declaration) in [
        ("assign y = x[0:1];", "x[0:1]", "[1:0]"),
        ("assign y[0:1] = x;", "y[0:1]", "[1:0]"),
        ("assign y = {{w[1:0]}};", "w[1:0]", "[0:1]"),
        ("tran(w[1:0], x);", "w[1:0]", "[0:1]"),
        ("tran(w[0], x[0:1]);", "x[0:1]", "[1:0]"),
        ("BUF inst(.A({x[0:1]}), .Y(y));", "x[0:1]", "[1:0]"),
    ] {
        let source = format!(
            "module top(x, y); {statement} input [1:0] x; output [1:0] y; wire [0:1] w; endmodule"
        );
        std::fs::write(&path, source)?;
        let error = read_gv_for_proof_from_path(&path)
            .err()
            .expect("a reversed part-select must not become a bit reversal");
        assert_eq!(
            error.to_string().split(" @ ").next().unwrap(),
            format!(
                "GV reader: part-select '{select}' does not follow its declared packed range {declaration}"
            )
        );
    }
    Ok(())
}

#[test]
fn proof_gv_reader_accepts_ascending_and_singleton_part_selects() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("ascending.gv");
    std::fs::write(
        &path,
        r#"module top(input [1:0] x, output [1:0] y);
  assign w[0:1] = x[1:0];
  assign y = {w[0:0], w[1:1]};
  wire [0:1] w;
endmodule"#,
    )?;
    read_gv_for_proof_from_path(&path)?;
    Ok(())
}

#[test]
fn proof_gv_reader_requires_packed_declarations_for_selects() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("scalar.gv");
    for (body, select, net) in [
        ("assign y = x[0];", "x[0]", "x"),
        ("assign y = {{x[0:0]}};", "x[0:0]", "x"),
        ("assign y[0] = x;", "y[0]", "y"),
        ("BUF inst(.A(x[0:0]), .Y(y));", "x[0:0]", "x"),
        ("tran(y, x[0]);", "x[0]", "x"),
        ("assign w = x; assign y = w[0];", "w[0]", "w"),
    ] {
        let source = format!("module top(x, y); {body} input x; output y; endmodule");
        std::fs::write(&path, &source)?;
        let error = read_gv_for_proof_from_path(&path)
            .err()
            .expect("a scalar cannot be indexed");
        assert_eq!(
            error.to_string().split(" @ ").next().unwrap(),
            format!("GV reader: select '{select}' requires a packed net. '{net}' is scalar")
        );
        read_gv_from_str(&source)?;
    }

    // A packed [0:0] net supports both forms of select.
    std::fs::write(
        &path,
        "module top(input [0:0] x, output y); wire [0:0] w; assign w[0] = x[0:0]; assign y = w[0:0]; endmodule",
    )?;
    read_gv_for_proof_from_path(&path)?;
    Ok(())
}

#[test]
fn preserved_gv_readers_reject_conflicting_scalar_and_packed_declarations() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("declarations.gv");
    for declarations in [
        "input x; wire [0:0] x;",
        "wire [0:0] x; input x;",
        "input [1:0] x; wire x;",
        "wire x; input [1:0] x;",
    ] {
        let source = format!("module top(x, y); assign y = x; {declarations} output y; endmodule");
        std::fs::write(&path, &source)?;
        for result in [
            read_gv_for_proof_from_path(&path),
            read_gv_for_roundtrip_from_path(&path),
        ] {
            let error = result.err().expect("incompatible explicit declarations");
            assert_eq!(
                error.to_string().split(" @ ").next().unwrap(),
                "GV reader: net 'x' has both scalar and packed declarations"
            );
        }
        read_gv_from_str(&source)?;
    }
    Ok(())
}

#[test]
fn proof_gv_reader_requires_scalar_tran_terminals_before_elaboration() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("tran.gv");
    for (lhs, rhs, terminal) in [("out", "x", "out"), ("out[0]", "x", "x")] {
        std::fs::write(
            &path,
            format!("module top(input [1:0] x, output [1:0] out); tran({lhs}, {rhs}); endmodule"),
        )?;
        let error = read_gv_for_proof_from_path(&path)
            .err()
            .expect("vector tran terminal");
        assert_eq!(
            error.to_string().split(" @ ").next().unwrap(),
            format!(
                "GV reader: plain tran terminal '{terminal}' has width 2. Use one tran primitive per bit"
            )
        );
        // The round-trip codec can preserve the original terminal structure.
        read_gv_for_roundtrip_from_path(&path)?;
    }
    std::fs::write(
        &path,
        "module top(input [1:0] x, output [1:0] out); tran(out[0], x[0]); tran(out[1], x[1]); endmodule",
    )?;
    read_gv_for_proof_from_path(&path)?;
    Ok(())
}

#[test]
fn roundtrip_reader_accepts_implicit_declarations_and_forward_references() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("implicit.gv");
    let source = r#"module top(input a, output [5:0] y);
  BUF inst(.A({cell_input}), .Y(cell_output));
  tran(tran_left, a);
  tran(a, tran_right);
  assign {alias_net} = a;
  assign y = {cell_input, cell_output, tran_left, tran_right, alias_net, later_wire};
  wire later_wire;
endmodule
"#;
    std::fs::write(&path, source)?;
    read_gv_for_roundtrip_from_path(&path)?;
    Ok(())
}
