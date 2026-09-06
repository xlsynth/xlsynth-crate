// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::path::Path;
use tempfile::NamedTempFile;
use xlsynth_g8r::netlist::gv2block::convert_gv2block_paths;

/// Requires every successful fixture to produce both PIR-valid and XLS-valid
/// IR.
fn convert_gv2block_paths_to_string(netlist: &Path, liberty: &Path) -> anyhow::Result<String> {
    let package = convert_gv2block_paths(netlist, liberty)?;
    xlsynth_pir::ir_verify::verify_package(&package).expect("generated package must verify");
    let text = package.to_string();
    xlsynth::IrPackage::parse_ir(&text, None)
        .unwrap_or_else(|error| panic!("generated package must parse in XLS: {error}\n{text}"));
    Ok(text)
}

/// Converts an inline netlist with the common test library and checks its
/// package.
fn convert_netlist(netlist: &str) -> anyhow::Result<String> {
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{netlist}").unwrap();
    convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path())
}

const LIBERTY_TEXTPROTO: &str = r#"
format_magic: 5496997758177923663
cells: {
  name: "INV"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 3 }
  area: 1.0
}
cells: {
  name: "BUF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 1 }
  area: 1.0
}
cells: {
  name: "CKG"
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 6 direction: OUTPUT }
  area: 1.0
  clock_gate: {
    clock_pin: "CLK"
    output_pin: "GCLK"
  }
}
cells: {
  name: "DFF"
  pins: { name_string_id: 7 direction: INPUT }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 8 direction: OUTPUT function_string_id: 8 }
  area: 1.0
  sequential: {
    state_var: "Q"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
cells: {
  name: "DFFNEDGE"
  pins: { name_string_id: 7 direction: INPUT }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 8 direction: OUTPUT function_string_id: 8 }
  area: 1.0
  sequential: {
    state_var: "Q"
    next_state: "D"
    clock_expr: "!CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
cells: {
  name: "DFFQN"
  pins: { name_string_id: 7 direction: INPUT }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 8 direction: OUTPUT function_string_id: 9 }
  pins: { name_string_id: 10 direction: OUTPUT function_string_id: 11 }
  area: 1.0
  sequential: {
    state_var: "IQ"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
    complementary_state_var: "IQN"
  }
}
cells: {
  name: "DFFCLR"
  pins: { name_string_id: 7 direction: INPUT }
  pins: { name_string_id: 12 direction: INPUT }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 8 direction: OUTPUT function_string_id: 8 }
  area: 1.0
  sequential: {
    state_var: "Q"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
    clear_expr: "RST"
  }
}
cells: {
  name: "DFFPRE"
  pins: { name_string_id: 7 direction: INPUT }
  pins: { name_string_id: 13 direction: INPUT }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 8 direction: OUTPUT function_string_id: 8 }
  area: 1.0
  sequential: {
    state_var: "Q"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
    preset_expr: "!RSTN"
  }
}
cells: {
  name: "AND2"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 14 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 15 }
  area: 1.0
}
cells: {
  name: "DFFNAND"
  pins: { name_string_id: 7 direction: INPUT }
  pins: { name_string_id: 16 direction: INPUT }
  pins: { name_string_id: 5 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 8 direction: OUTPUT function_string_id: 8 }
  area: 1.0
  sequential: {
    state_var: "Q"
    next_state: "!(D * EN)"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
interned_strings: ["A", "Y", "(!A)", "unused", "CLK", "GCLK", "D", "Q", "IQ", "QN", "IQN", "RST", "RSTN", "B", "(A * B)", "EN"]
"#;

#[test]
fn test_gv2block_inverter() {
    let netlist = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u1 (.A(a), .Y(y));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let want = r#"package top_

block INV(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  not.2: bits[1] = not(A, id=2)
  Y: () = output_port(not.2, name=Y, id=3)
}

top block top_(a: bits[1], y: bits[1]) {
  instantiation u1(block=INV, kind=block)
  a: bits[1] = input_port(name=a, id=4)
  u1_Y: bits[1] = instantiation_output(instantiation=u1, port_name=Y, id=5)
  u1_A: () = instantiation_input(a, instantiation=u1, port_name=A, id=6)
  y: () = output_port(u1_Y, name=y, id=7)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_dff_cell() {
    let netlist = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  wire d;
  wire clk;
  wire q;
  DFF u1 (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let want = r#"package top_

block DFF(CLK: clock, D: bits[1], Q: bits[1]) {
  reg Q_reg(bits[1])
  D: bits[1] = input_port(name=D, id=1)
  Q_q: bits[1] = register_read(register=Q_reg, id=2)
  Q_d: () = register_write(D, register=Q_reg, id=3)
  Q: () = output_port(Q_q, name=Q, id=4)
}

top block top_(clk: clock, d: bits[1], q: bits[1]) {
  instantiation u1(block=DFF, kind=block)
  d: bits[1] = input_port(name=d, id=5)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=6)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=7)
  q: () = output_port(u1_Q, name=q, id=8)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_rejects_negative_edge_ff() {
    let netlist = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  wire d;
  wire clk;
  wire q;
  DFFNEDGE u1 (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let error = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path())
        .expect_err("negative-edge FFs should be rejected");
    assert!(
        format!("{error:#}").contains("gv2block only supports positive-edge FFs"),
        "{error:#}"
    );
}

#[test]
fn test_gv2block_dff_complementary_output_uses_inverted_register() {
    let netlist = r#"
module top (d, clk, q, qn);
  input d;
  input clk;
  output q;
  output qn;
  wire d;
  wire clk;
  wire q;
  wire qn;
  DFFQN u1 (.D(d), .CLK(clk), .Q(q), .QN(qn));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let want = r#"package top_

block DFFQN(CLK: clock, D: bits[1], Q: bits[1], QN: bits[1]) {
  reg IQ_reg(bits[1])
  D: bits[1] = input_port(name=D, id=1)
  IQ_q: bits[1] = register_read(register=IQ_reg, id=2)
  IQN_q: bits[1] = not(IQ_q, id=3)
  IQ_d: () = register_write(D, register=IQ_reg, id=4)
  Q: () = output_port(IQ_q, name=Q, id=5)
  QN: () = output_port(IQN_q, name=QN, id=6)
}

top block top_(clk: clock, d: bits[1], q: bits[1], qn: bits[1]) {
  instantiation u1(block=DFFQN, kind=block)
  d: bits[1] = input_port(name=d, id=7)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=8)
  u1_QN: bits[1] = instantiation_output(instantiation=u1, port_name=QN, id=9)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=10)
  q: () = output_port(u1_Q, name=q, id=11)
  qn: () = output_port(u1_QN, name=qn, id=12)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_dff_with_logic() {
    let netlist = r#"
module top (d, en, clk, q);
  input d;
  input en;
  input clk;
  output q;
  wire d;
  wire en;
  wire clk;
  wire q;
  DFFNAND u1 (.D(d), .EN(en), .CLK(clk), .Q(q));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let want = r#"package top_

block DFFNAND(CLK: clock, D: bits[1], EN: bits[1], Q: bits[1]) {
  reg Q_reg(bits[1])
  D: bits[1] = input_port(name=D, id=1)
  EN: bits[1] = input_port(name=EN, id=2)
  Q_q: bits[1] = register_read(register=Q_reg, id=3)
  and.4: bits[1] = and(D, EN, id=4)
  not.5: bits[1] = not(and.4, id=5)
  Q_d: () = register_write(not.5, register=Q_reg, id=6)
  Q: () = output_port(Q_q, name=Q, id=7)
}

top block top_(clk: clock, d: bits[1], en: bits[1], q: bits[1]) {
  instantiation u1(block=DFFNAND, kind=block)
  d: bits[1] = input_port(name=d, id=8)
  en: bits[1] = input_port(name=en, id=9)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=10)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=11)
  u1_EN: () = instantiation_input(en, instantiation=u1, port_name=EN, id=12)
  q: () = output_port(u1_Q, name=q, id=13)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_dff_with_clear() {
    let netlist = r#"
module top (d, rst, clk, q);
  input d;
  input rst;
  input clk;
  output q;
  wire d;
  wire rst;
  wire clk;
  wire q;
  DFFCLR u1 (.D(d), .RST(rst), .CLK(clk), .Q(q));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let want = r#"package top_

block DFFCLR(CLK: clock, D: bits[1], RST: bits[1], Q: bits[1]) {
  #![reset(port="RST", asynchronous=true, active_low=false)]
  reg Q_reg(bits[1], reset_value=0)
  D: bits[1] = input_port(name=D, id=1)
  RST: bits[1] = input_port(name=RST, id=2)
  Q_q: bits[1] = register_read(register=Q_reg, id=3)
  Q_d: () = register_write(D, register=Q_reg, reset=RST, id=4)
  Q: () = output_port(Q_q, name=Q, id=5)
}

top block top_(clk: clock, d: bits[1], rst: bits[1], q: bits[1]) {
  instantiation u1(block=DFFCLR, kind=block)
  d: bits[1] = input_port(name=d, id=6)
  rst: bits[1] = input_port(name=rst, id=7)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=8)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=9)
  u1_RST: () = instantiation_input(rst, instantiation=u1, port_name=RST, id=10)
  q: () = output_port(u1_Q, name=q, id=11)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_dff_with_preset() {
    let netlist = r#"
module top (d, rstn, clk, q);
  input d;
  input rstn;
  input clk;
  output q;
  wire d;
  wire rstn;
  wire clk;
  wire q;
  DFFPRE u1 (.D(d), .RSTN(rstn), .CLK(clk), .Q(q));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let want = r#"package top_

block DFFPRE(CLK: clock, D: bits[1], RSTN: bits[1], Q: bits[1]) {
  #![reset(port="RSTN", asynchronous=true, active_low=true)]
  reg Q_reg(bits[1], reset_value=1)
  D: bits[1] = input_port(name=D, id=1)
  RSTN: bits[1] = input_port(name=RSTN, id=2)
  Q_q: bits[1] = register_read(register=Q_reg, id=3)
  Q_d: () = register_write(D, register=Q_reg, reset=RSTN, id=4)
  Q: () = output_port(Q_q, name=Q, id=5)
}

top block top_(clk: clock, d: bits[1], rstn: bits[1], q: bits[1]) {
  instantiation u1(block=DFFPRE, kind=block)
  d: bits[1] = input_port(name=d, id=6)
  rstn: bits[1] = input_port(name=rstn, id=7)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=8)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=9)
  u1_RSTN: () = instantiation_input(rstn, instantiation=u1, port_name=RSTN, id=10)
  q: () = output_port(u1_Q, name=q, id=11)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_vector_outputs_from_multiple_cells() {
    let netlist = r#"
module top (a, y);
  input [2:0] a;
  output [2:0] y;
  wire [2:0] a;
  wire [2:0] y;
  INV u0 (.A(a[0]), .Y(y[0]));
  BUF u1 (.A(a[1]), .Y(y[1]));
  INV u2 (.A(a[2]), .Y(y[2]));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let want = r#"package top_

block BUF(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  Y: () = output_port(A, name=Y, id=2)
}

block INV(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=3)
  not.4: bits[1] = not(A, id=4)
  Y: () = output_port(not.4, name=Y, id=5)
}

top block top_(a: bits[3], y: bits[3]) {
  instantiation u0(block=INV, kind=block)
  instantiation u1(block=BUF, kind=block)
  instantiation u2(block=INV, kind=block)
  a: bits[3] = input_port(name=a, id=6)
  u0_Y: bits[1] = instantiation_output(instantiation=u0, port_name=Y, id=7)
  u1_Y: bits[1] = instantiation_output(instantiation=u1, port_name=Y, id=8)
  u2_Y: bits[1] = instantiation_output(instantiation=u2, port_name=Y, id=9)
  bit_slice.10: bits[1] = bit_slice(a, start=0, width=1, id=10)
  u0_A: () = instantiation_input(bit_slice.10, instantiation=u0, port_name=A, id=11)
  bit_slice.12: bits[1] = bit_slice(a, start=1, width=1, id=12)
  u1_A: () = instantiation_input(bit_slice.12, instantiation=u1, port_name=A, id=13)
  bit_slice.14: bits[1] = bit_slice(a, start=2, width=1, id=14)
  u2_A: () = instantiation_input(bit_slice.14, instantiation=u2, port_name=A, id=15)
  concat.16: bits[3] = concat(u2_Y, u1_Y, u0_Y, id=16)
  y: () = output_port(concat.16, name=y, id=17)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_vector_output_with_unused_bit() {
    let netlist = r#"
module top (a, y);
  input a;
  output [1:0] y;
  wire a;
  wire [1:0] y;
  INV u0 (.A(a), .Y(y[0]));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let want = r#"package top_

block INV(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  not.2: bits[1] = not(A, id=2)
  Y: () = output_port(not.2, name=Y, id=3)
}

top block top_(a: bits[1], y: bits[2]) {
  instantiation u0(block=INV, kind=block)
  a: bits[1] = input_port(name=a, id=4)
  u0_Y: bits[1] = instantiation_output(instantiation=u0, port_name=Y, id=5)
  u0_A: () = instantiation_input(a, instantiation=u0, port_name=A, id=6)
  literal.7: bits[1] = literal(value=0, id=7)
  concat.8: bits[2] = concat(literal.7, u0_Y, id=8)
  y: () = output_port(concat.8, name=y, id=9)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_elides_clock_gate_cell() {
    let netlist = r#"
module top (clk, d, q);
  input clk;
  input d;
  output q;
  wire clk;
  wire d;
  wire q;
  wire gclk;
  CKG u_cg (.CLK(clk), .GCLK(gclk));
  DFF u1 (.D(d), .CLK(gclk), .Q(q));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let want = r#"package top_

block DFF(CLK: clock, D: bits[1], Q: bits[1]) {
  reg Q_reg(bits[1])
  D: bits[1] = input_port(name=D, id=1)
  Q_q: bits[1] = register_read(register=Q_reg, id=2)
  Q_d: () = register_write(D, register=Q_reg, id=3)
  Q: () = output_port(Q_q, name=Q, id=4)
}

top block top_(clk: clock, d: bits[1], q: bits[1]) {
  instantiation u1(block=DFF, kind=block)
  d: bits[1] = input_port(name=d, id=5)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=6)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=7)
  q: () = output_port(u1_Q, name=q, id=8)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_rejects_derived_clock() {
    let netlist = r#"
module top (clk, en, d, q);
  input clk;
  input en;
  input d;
  output q;
  wire clk;
  wire en;
  wire d;
  wire q;
  wire gclk;
  AND2 u0 (.A(clk), .B(en), .Y(gclk));
  DFF u1 (.D(d), .CLK(gclk), .Q(q));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let err = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path())
        .expect_err("expected derived clock rejection");
    assert!(err.to_string().contains("derived clock 'gclk'"));
}

#[test]
fn test_gv2block_rejects_multiple_clocks() {
    let netlist = r#"
module top (clk0, clk1, d0, d1, q0, q1);
  input clk0;
  input clk1;
  input d0;
  input d1;
  output q0;
  output q1;
  wire clk0;
  wire clk1;
  wire d0;
  wire d1;
  wire q0;
  wire q1;
  DFF u0 (.D(d0), .CLK(clk0), .Q(q0));
  DFF u1 (.D(d1), .CLK(clk1), .Q(q1));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let err = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path())
        .expect_err("expected multi-clock rejection");
    let err_text = err.to_string();
    assert!(err_text.contains("multiple clock nets detected"));
    assert!(err_text.contains("clk0"));
    assert!(err_text.contains("clk1"));
}

#[test]
fn test_gv2block_rejects_clock_net_used_as_data() {
    let netlist = r#"
module top (clk, d, q, nclk);
  input clk;
  input d;
  output q;
  output nclk;
  wire clk;
  wire d;
  wire q;
  wire nclk;
  DFF u0 (.D(d), .CLK(clk), .Q(q));
  INV u1 (.A(clk), .Y(nclk));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let err = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path())
        .expect_err("expected rejection when selected clock net is used as data");
    let err_text = err.to_string();
    assert!(err_text.contains("clock net 'clk' is connected to non-clock input"));
    assert!(err_text.contains("u1.A"));
}

#[test]
fn test_gv2block_legalizes_escaped_instance_identifier() {
    let netlist = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  AND2 \foo[0]_blah (.A(a), .B(b), .Y(y));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();

    let legalized = "foo_0__blah";
    assert!(got.contains(&format!(
        "instantiation {}(block=AND2, kind=block)",
        legalized
    )));
    assert!(got.contains(&format!(
        "{}_Y: bits[1] = instantiation_output(instantiation={}, port_name=Y",
        legalized, legalized
    )));
    assert!(got.contains(&format!(
        "{}_A: () = instantiation_input(a, instantiation={}, port_name=A",
        legalized, legalized
    )));
}

#[test]
fn test_gv2block_accepts_preserved_wiring_assigns() {
    let netlist = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n;
  BUF u0 (.A(a), .Y(n));
  assign y = n;
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();
    assert!(got.contains("u0_A: () = instantiation_input(a, instantiation=u0, port_name=A"));
    assert!(got.contains("y: () = output_port(u0_Y, name=y"));
}

#[test]
fn test_gv2block_accepts_preserved_tran_aliases() {
    let netlist = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire y_alias;
  BUF u0 (.A(a), .Y(y_alias));
  tran(y, y_alias);
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();
    assert!(got.contains("y: () = output_port(u0_Y, name=y"));
}

#[test]
fn test_gv2block_rejects_preserved_combinational_assigns() {
    let netlist = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  assign y = a & b;
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", LIBERTY_TEXTPROTO).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();

    let err = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path())
        .expect_err("expected top-level logic assign to be rejected");
    let err_text = err.to_string();
    assert!(err_text.contains("gv2block only supports techmapped netlists"));
    assert!(err_text.contains("run technology mapping first"));
}

#[test]
fn test_gv2block_legalizes_reserved_names_before_resolving_collisions() {
    let netlist = r#"
module top (stage, stage_, clock);
  input stage;
  input stage_;
  output clock;
  AND2 ret (.A(stage), .B(stage_), .Y(clock));
endmodule
"#;
    let got = convert_netlist(netlist).unwrap();
    let want = r#"package top_

block AND2(A: bits[1], B: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  B: bits[1] = input_port(name=B, id=2)
  and.3: bits[1] = and(A, B, id=3)
  Y: () = output_port(and.3, name=Y, id=4)
}

top block top_(stage_: bits[1], stage__1: bits[1], clock_: bits[1]) {
  instantiation ret_(block=AND2, kind=block)
  stage_: bits[1] = input_port(name=stage_, id=5)
  stage__1: bits[1] = input_port(name=stage__1, id=6)
  ret__Y: bits[1] = instantiation_output(instantiation=ret_, port_name=Y, id=7)
  ret__A: () = instantiation_input(stage_, instantiation=ret_, port_name=A, id=8)
  ret__B: () = instantiation_input(stage__1, instantiation=ret_, port_name=B, id=9)
  clock_: () = output_port(ret__Y, name=clock_, id=10)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_reserves_clock_and_late_output_names_before_internal_aliases() {
    let netlist = r#"
module aliases (u_D, u_Q, u_Q_1);
  input u_D;
  input u_Q;
  output u_Q_1;
  DFF u (.CLK(u_D), .D(u_Q), .Q(u_Q_1));
endmodule
"#;
    let got = convert_netlist(netlist).unwrap();
    let want = r#"package aliases

block DFF(CLK: clock, D: bits[1], Q: bits[1]) {
  reg Q_reg(bits[1])
  D: bits[1] = input_port(name=D, id=1)
  Q_q: bits[1] = register_read(register=Q_reg, id=2)
  Q_d: () = register_write(D, register=Q_reg, id=3)
  Q: () = output_port(Q_q, name=Q, id=4)
}

top block aliases(u_D: clock, u_Q: bits[1], u_Q_1: bits[1]) {
  instantiation u(block=DFF, kind=block)
  u_Q: bits[1] = input_port(name=u_Q, id=5)
  u_Q_2: bits[1] = instantiation_output(instantiation=u, port_name=Q, id=6)
  u_D_1: () = instantiation_input(u_Q, instantiation=u, port_name=D, id=7)
  u_Q_1: () = output_port(u_Q_2, name=u_Q_1, id=8)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_uniquifies_aliases_from_ambiguous_instance_pin_boundaries() {
    let liberty = r#"
format_magic: 5496997758177923663
cells: {
  name: "X"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: OUTPUT function_string_id: 1 }
}
cells: {
  name: "Y"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 1 }
}
interned_strings: ["A", "B_C", "C"]
"#;
    let netlist = r#"
module aliases (a, y0, y1);
  input a;
  output y0;
  output y1;
  X u (.A(a), .B_C(y0));
  Y u_B (.A(a), .C(y1));
endmodule
"#;
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{liberty}").unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{netlist}").unwrap();
    let got = convert_gv2block_paths_to_string(netlist_file.path(), liberty_file.path()).unwrap();
    let want = r#"package aliases

block X(A: bits[1], B_C: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  B_C: () = output_port(A, name=B_C, id=2)
}

block Y(A: bits[1], C: bits[1]) {
  A: bits[1] = input_port(name=A, id=3)
  C: () = output_port(A, name=C, id=4)
}

top block aliases(a: bits[1], y0: bits[1], y1: bits[1]) {
  instantiation u(block=X, kind=block)
  instantiation u_B(block=Y, kind=block)
  a: bits[1] = input_port(name=a, id=5)
  u_B_C: bits[1] = instantiation_output(instantiation=u, port_name=B_C, id=6)
  u_B_C_1: bits[1] = instantiation_output(instantiation=u_B, port_name=C, id=7)
  u_A: () = instantiation_input(a, instantiation=u, port_name=A, id=8)
  u_B_A: () = instantiation_input(a, instantiation=u_B, port_name=A, id=9)
  y0: () = output_port(u_B_C, name=y0, id=10)
  y1: () = output_port(u_B_C_1, name=y1, id=11)
}
"#;
    assert_eq!(got, want);
}

#[test]
fn test_gv2block_preserves_clock_legalization_after_data_ports() {
    let netlist = r#"
module clock_collision (\clk[0] , clk_0_, q);
  input \clk[0] ;
  input clk_0_;
  output q;
  DFF u (.CLK(\clk[0] ), .D(clk_0_), .Q(q));
endmodule
"#;
    let text = convert_netlist(netlist).unwrap();
    let package = xlsynth_pir::ir_parser::Parser::new(&text)
        .parse_and_verify_package()
        .unwrap();
    let block = package.get_top_block().unwrap();
    assert_eq!(block.clock_port_name(), Some("clk_0__1"));
    assert_eq!(
        block
            .input_ports()
            .map(|port| block.port_name(port))
            .collect::<Vec<_>>(),
        vec!["clk_0_"]
    );
}

#[test]
fn test_gv2block_rejects_missing_or_incorrectly_sized_instance_inputs() {
    let missing = r#"
module missing (y);
  output y;
  INV u (.Y(y));
endmodule
"#;
    assert_eq!(
        convert_netlist(missing).unwrap_err().to_string(),
        "function 'missing' instantiation 'u' missing input ports: [\"A\"]"
    );
    let wide = r#"
module wide (a, y);
  input [1:0] a;
  output y;
  INV u (.A(a), .Y(y));
endmodule
"#;
    assert_eq!(
        convert_netlist(wide).unwrap_err().to_string(),
        "instance input 'A' requires bits[1], got bits[2]"
    );
}

#[test]
fn test_gv2block_empty_instance_input_remains_tied_to_zero() {
    let netlist = r#"
module tied (y);
  output y;
  INV u (.A(), .Y(y));
endmodule
"#;
    let got = convert_netlist(netlist).unwrap();
    let want = r#"package tied

block INV(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  not.2: bits[1] = not(A, id=2)
  Y: () = output_port(not.2, name=Y, id=3)
}

top block tied(y: bits[1]) {
  instantiation u(block=INV, kind=block)
  u_Y: bits[1] = instantiation_output(instantiation=u, port_name=Y, id=4)
  literal.5: bits[1] = literal(value=0, id=5)
  u_A: () = instantiation_input(literal.5, instantiation=u, port_name=A, id=6)
  y: () = output_port(u_Y, name=y, id=7)
}
"#;
    assert_eq!(got, want);
}
