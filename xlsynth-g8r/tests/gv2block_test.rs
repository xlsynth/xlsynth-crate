// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use tempfile::NamedTempFile;
use xlsynth_g8r::netlist::gv2block::convert_gv2block_paths_to_string;

const LIBERTY_TEXTPROTO: &str = r#"
cells: {
  name: "INV"
  pins: { name: "A" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "(!A)" }
  area: 1.0
}
cells: {
  name: "BUF"
  pins: { name: "A" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "A" }
  area: 1.0
}
cells: {
  name: "CKG"
  pins: { name: "CLK" direction: INPUT is_clocking_pin: true }
  pins: { name: "GCLK" direction: OUTPUT }
  area: 1.0
  clock_gate: {
    clock_pin: "CLK"
    output_pin: "GCLK"
  }
}
cells: {
  name: "DFF"
  pins: { name: "D" direction: INPUT }
  pins: { name: "CLK" direction: INPUT is_clocking_pin: true }
  pins: { name: "Q" direction: OUTPUT function: "Q" }
  area: 1.0
  sequential: {
    state_var: "Q"
    next_state: "D"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
cells: {
  name: "DFFCLR"
  pins: { name: "D" direction: INPUT }
  pins: { name: "RST" direction: INPUT }
  pins: { name: "CLK" direction: INPUT is_clocking_pin: true }
  pins: { name: "Q" direction: OUTPUT function: "Q" }
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
  pins: { name: "D" direction: INPUT }
  pins: { name: "RSTN" direction: INPUT }
  pins: { name: "CLK" direction: INPUT is_clocking_pin: true }
  pins: { name: "Q" direction: OUTPUT function: "Q" }
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
  pins: { name: "A" direction: INPUT }
  pins: { name: "B" direction: INPUT }
  pins: { name: "Y" direction: OUTPUT function: "(A * B)" }
  area: 1.0
}
cells: {
  name: "DFFNAND"
  pins: { name: "D" direction: INPUT }
  pins: { name: "EN" direction: INPUT }
  pins: { name: "CLK" direction: INPUT is_clocking_pin: true }
  pins: { name: "Q" direction: OUTPUT function: "Q" }
  area: 1.0
  sequential: {
    state_var: "Q"
    next_state: "!(D * EN)"
    clock_expr: "CLK"
    kind: SEQUENTIAL_KIND_FF
  }
}
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

    let want = r#"package top

block INV(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  not.2: bits[1] = not(A, id=2)
  Y: () = output_port(not.2, name=Y, id=3)
}

top block top(a: bits[1], y: bits[1]) {
  instantiation u1(block=INV, kind=block)
  a: bits[1] = input_port(name=a, id=1)
  u1_Y: bits[1] = instantiation_output(instantiation=u1, port_name=Y, id=2)
  u1_A: () = instantiation_input(a, instantiation=u1, port_name=A, id=3)
  y: () = output_port(u1_Y, name=y, id=4)
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

    let want = r#"package top

block DFF(CLK: clock, D: bits[1], Q: bits[1]) {
  reg Q_reg(bits[1])
  D: bits[1] = input_port(name=D, id=1)
  Q_q: bits[1] = register_read(register=Q_reg, id=2)
  Q_d: () = register_write(D, register=Q_reg, id=3)
  Q: () = output_port(Q_q, name=Q, id=4)
}

top block top(clk: clock, d: bits[1], q: bits[1]) {
  instantiation u1(block=DFF, kind=block)
  d: bits[1] = input_port(name=d, id=1)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=2)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=3)
  q: () = output_port(u1_Q, name=q, id=4)
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

    let want = r#"package top

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

top block top(clk: clock, d: bits[1], en: bits[1], q: bits[1]) {
  instantiation u1(block=DFFNAND, kind=block)
  d: bits[1] = input_port(name=d, id=1)
  en: bits[1] = input_port(name=en, id=2)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=3)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=4)
  u1_EN: () = instantiation_input(en, instantiation=u1, port_name=EN, id=5)
  q: () = output_port(u1_Q, name=q, id=6)
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

    let want = r#"package top

block DFFCLR(CLK: clock, D: bits[1], RST: bits[1], Q: bits[1]) {
  #![reset(port="RST", asynchronous=true, active_low=false)]
  reg Q_reg(bits[1], reset_value=0)
  D: bits[1] = input_port(name=D, id=1)
  RST: bits[1] = input_port(name=RST, id=2)
  Q_q: bits[1] = register_read(register=Q_reg, id=3)
  Q_d: () = register_write(D, register=Q_reg, reset=RST, id=4)
  Q: () = output_port(Q_q, name=Q, id=5)
}

top block top(clk: clock, d: bits[1], rst: bits[1], q: bits[1]) {
  instantiation u1(block=DFFCLR, kind=block)
  d: bits[1] = input_port(name=d, id=1)
  rst: bits[1] = input_port(name=rst, id=2)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=3)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=4)
  u1_RST: () = instantiation_input(rst, instantiation=u1, port_name=RST, id=5)
  q: () = output_port(u1_Q, name=q, id=6)
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

    let want = r#"package top

block DFFPRE(CLK: clock, D: bits[1], RSTN: bits[1], Q: bits[1]) {
  #![reset(port="RSTN", asynchronous=true, active_low=true)]
  reg Q_reg(bits[1], reset_value=1)
  D: bits[1] = input_port(name=D, id=1)
  RSTN: bits[1] = input_port(name=RSTN, id=2)
  Q_q: bits[1] = register_read(register=Q_reg, id=3)
  Q_d: () = register_write(D, register=Q_reg, reset=RSTN, id=4)
  Q: () = output_port(Q_q, name=Q, id=5)
}

top block top(clk: clock, d: bits[1], rstn: bits[1], q: bits[1]) {
  instantiation u1(block=DFFPRE, kind=block)
  d: bits[1] = input_port(name=d, id=1)
  rstn: bits[1] = input_port(name=rstn, id=2)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=3)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=4)
  u1_RSTN: () = instantiation_input(rstn, instantiation=u1, port_name=RSTN, id=5)
  q: () = output_port(u1_Q, name=q, id=6)
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

    let want = r#"package top

block BUF(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  Y: () = output_port(A, name=Y, id=2)
}

block INV(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  not.2: bits[1] = not(A, id=2)
  Y: () = output_port(not.2, name=Y, id=3)
}

top block top(a: bits[3], y: bits[3]) {
  instantiation u0(block=INV, kind=block)
  instantiation u1(block=BUF, kind=block)
  instantiation u2(block=INV, kind=block)
  a: bits[3] = input_port(name=a, id=1)
  u0_Y: bits[1] = instantiation_output(instantiation=u0, port_name=Y, id=2)
  u1_Y: bits[1] = instantiation_output(instantiation=u1, port_name=Y, id=3)
  u2_Y: bits[1] = instantiation_output(instantiation=u2, port_name=Y, id=4)
  bit_slice.5: bits[1] = bit_slice(a, start=0, width=1, id=5)
  u0_A: () = instantiation_input(bit_slice.5, instantiation=u0, port_name=A, id=6)
  bit_slice.7: bits[1] = bit_slice(a, start=1, width=1, id=7)
  u1_A: () = instantiation_input(bit_slice.7, instantiation=u1, port_name=A, id=8)
  bit_slice.9: bits[1] = bit_slice(a, start=2, width=1, id=9)
  u2_A: () = instantiation_input(bit_slice.9, instantiation=u2, port_name=A, id=10)
  concat.11: bits[3] = concat(u2_Y, u1_Y, u0_Y, id=11)
  y: () = output_port(concat.11, name=y, id=12)
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

    let want = r#"package top

block INV(A: bits[1], Y: bits[1]) {
  A: bits[1] = input_port(name=A, id=1)
  not.2: bits[1] = not(A, id=2)
  Y: () = output_port(not.2, name=Y, id=3)
}

top block top(a: bits[1], y: bits[2]) {
  instantiation u0(block=INV, kind=block)
  a: bits[1] = input_port(name=a, id=1)
  u0_Y: bits[1] = instantiation_output(instantiation=u0, port_name=Y, id=2)
  u0_A: () = instantiation_input(a, instantiation=u0, port_name=A, id=3)
  literal.4: bits[1] = literal(value=0, id=4)
  concat.5: bits[2] = concat(literal.4, u0_Y, id=5)
  y: () = output_port(concat.5, name=y, id=6)
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

    let want = r#"package top

block DFF(CLK: clock, D: bits[1], Q: bits[1]) {
  reg Q_reg(bits[1])
  D: bits[1] = input_port(name=D, id=1)
  Q_q: bits[1] = register_read(register=Q_reg, id=2)
  Q_d: () = register_write(D, register=Q_reg, id=3)
  Q: () = output_port(Q_q, name=Q, id=4)
}

top block top(clk: clock, d: bits[1], q: bits[1]) {
  instantiation u1(block=DFF, kind=block)
  d: bits[1] = input_port(name=d, id=1)
  u1_Q: bits[1] = instantiation_output(instantiation=u1, port_name=Q, id=2)
  u1_D: () = instantiation_input(d, instantiation=u1, port_name=D, id=3)
  q: () = output_port(u1_Q, name=q, id=4)
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
