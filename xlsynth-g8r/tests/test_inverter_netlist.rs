// SPDX-License-Identifier: Apache-2.0

//! Integration test for a single inverter netlist and matching Liberty proto.

const LIBERTY_INVERTER_AND_BUF_TEXTPROTO: &str = r#"
cells: {
  name: "INVX1"
  pins: {
    name: "A"
    direction: INPUT
  }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "(!A)"
  }
  area: 1.0
}
cells: {
  name: "BUFX1"
  pins: {
    name: "A"
    direction: INPUT
  }
  pins: {
    name: "Y"
    direction: OUTPUT
    function: "A"
  }
  area: 1.0
}
"#;

use prost::Message;
use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;
use xlsynth_g8r::liberty_proto::Library;
use xlsynth_g8r::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
use xlsynth_g8r::netlist::parse::{Parser as NetlistParser, TokenScanner};

#[test]
fn test_single_inverter_netlist_and_liberty() {
    // Use only the INVX1 cell from the constant
    let liberty_textproto = LIBERTY_INVERTER_AND_BUF_TEXTPROTO
        .lines()
        .take_while(|line| !line.contains("BUFX1"))
        .collect::<Vec<_>>()
        .join("\n");

    // Minimal netlist (.v) as a string
    let netlist = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n1;
  INVX1 u1 (.A(a), .Y(y));
endmodule
"#;

    // Write Liberty textproto to a temp file
    let mut liberty_file = NamedTempFile::new().unwrap();
    write!(liberty_file, "{}", liberty_textproto).unwrap();

    // Write netlist to a temp file
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(netlist_file, "{}", netlist).unwrap();
    let netlist_path = netlist_file.path();

    // Convert Liberty textproto to binary proto using prost-reflect
    // (simulate what the real toolchain does)
    let descriptor_pool = prost_reflect::DescriptorPool::decode(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/liberty.bin"
    )) as &[u8])
    .unwrap();
    let msg_desc = descriptor_pool
        .get_message_by_name("liberty.Library")
        .unwrap();
    let dyn_msg =
        prost_reflect::DynamicMessage::parse_text_format(msg_desc, &liberty_textproto).unwrap();
    let mut liberty_bin = Vec::new();
    dyn_msg.encode(&mut liberty_bin).unwrap();
    let mut liberty_bin_file = NamedTempFile::new().unwrap();
    liberty_bin_file.write_all(&liberty_bin).unwrap();

    // Parse netlist
    let file = File::open(netlist_path).unwrap();
    let scanner = TokenScanner::from_file_with_path(file, netlist_path.to_path_buf());
    let mut parser = NetlistParser::new(scanner);
    let modules = parser.parse_file().unwrap();
    assert_eq!(modules.len(), 1);
    let module = &modules[0];

    // Parse Liberty proto
    let liberty_lib = Library::decode(&*liberty_bin).unwrap();

    // Use the helper to build the GateFn
    let gate_fn = project_gatefn_from_netlist_and_liberty(
        module,
        &parser.nets,
        &parser.interner,
        &liberty_lib,
    )
    .unwrap();
    let s = gate_fn.to_string();
    // Check that the output is the negation of the input
    assert!(s.contains("y[0] = not("), "GateFn output: {}", s);
    assert!(s.contains("not("), "GateFn output: {}", s);
}

#[test]
fn test_single_driver_multiple_consumer_netlist() {
    // Use the full constant for both INVX1 and BUFX1
    let liberty_textproto = LIBERTY_INVERTER_AND_BUF_TEXTPROTO;

    // Netlist: a -> INVX1 -> n; n -> BUFX1 -> y1; n -> BUFX1 -> y2
    let netlist = r#"
module top (a, y1, y2);
  input a;
  output y1;
  output y2;
  wire a;
  wire n;
  wire y1;
  wire y2;
  INVX1 u_inv (.A(a), .Y(n));
  BUFX1 u_buf1 (.A(n), .Y(y1));
  BUFX1 u_buf2 (.A(n), .Y(y2));
endmodule
"#;

    // Write Liberty textproto to a temp file
    let mut liberty_file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut liberty_file, liberty_textproto.as_bytes()).unwrap();

    // Write netlist to a temp file
    let mut netlist_file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut netlist_file, netlist.as_bytes()).unwrap();
    let netlist_path = netlist_file.path();

    // Convert Liberty textproto to binary proto using prost-reflect
    let descriptor_pool = prost_reflect::DescriptorPool::decode(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/liberty.bin"
    )) as &[u8])
    .unwrap();
    let msg_desc = descriptor_pool
        .get_message_by_name("liberty.Library")
        .unwrap();
    let dyn_msg =
        prost_reflect::DynamicMessage::parse_text_format(msg_desc, liberty_textproto).unwrap();
    let mut liberty_bin = Vec::new();
    dyn_msg.encode(&mut liberty_bin).unwrap();

    // Parse netlist
    let file = std::fs::File::open(netlist_path).unwrap();
    let scanner = xlsynth_g8r::netlist::parse::TokenScanner::from_file_with_path(
        file,
        netlist_path.to_path_buf(),
    );
    let mut parser = xlsynth_g8r::netlist::parse::Parser::new(scanner);
    let modules = parser.parse_file().unwrap();
    assert_eq!(modules.len(), 1);
    let module = &modules[0];

    // Parse Liberty proto
    let liberty_lib = xlsynth_g8r::liberty_proto::Library::decode(&*liberty_bin).unwrap();

    // Use the helper to build the GateFn
    let gate_fn =
        xlsynth_g8r::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty(
            module,
            &parser.nets,
            &parser.interner,
            &liberty_lib,
        )
        .unwrap();
    let s = gate_fn.to_string();
    // Check that both y1[0] and y2[0] are assigned to the same value, which is
    // not(a) and that n[0] is not(a)
    assert!(s.contains("n[0] = not("), "GateFn output: {}", s);
    assert!(s.contains("y1[0] = n[0]"), "GateFn output: {}", s);
    assert!(s.contains("y2[0] = n[0]"), "GateFn output: {}", s);
}
