// SPDX-License-Identifier: Apache-2.0

//! Block IR emission retains valid bit-select feedback through electrical nets.

use xlsynth_codegen::emit_system_verilog;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn emits_false_resolved_net_feedback_without_changing_its_wiring() {
    let source = r#"package inout_feedback
 top block root(seed: bits[1], pad: bits[4], result: bits[4]) {
  ext_inout_net bus(bits[4])
  seed: bits[1] = input_port(name=seed, id=1)
  pad: () = ext_inout_port(name=pad, net=bus, id=2)
  one: bits[1] = literal(value=1, id=3)
  read: bits[4] = ext_inout_read(net=bus, id=4)
  low: bits[3] = bit_slice(read, start=0, width=3, id=5)
  data: bits[4] = concat(low, seed, id=6)
  drive: () = ext_inout_drive(data, one, net=bus, id=7)
  result: () = output_port(read, name=result, id=8)
}"#;
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    let before = package.to_string();
    let generated = emit_system_verilog(&package, &Default::default()).unwrap();
    assert!(generated.system_verilog.contains("inout"));
    assert!(generated.system_verilog.contains("pad"));
    assert_eq!(package.to_string(), before);
}

#[test]
fn modeled_external_feedback_keeps_module_parameters_and_port_casts_in_sv() {
    let source = r#"package modeled_feedback
block model(d: bits[4], q: bits[4]) {
  d: bits[4] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(seed: bits[1], result: bits[4]) {
  instantiation u(block=model, module_name=external_leaf, parameters={WIDTH:"4"}, input_casts={d:"pkg::word_t"}, kind=behavioral_extern_block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=4)
  low: bits[3] = bit_slice(q, start=0, width=3, id=5)
  driver: bits[4] = concat(low, seed, id=6)
  drive: () = instantiation_input(driver, instantiation=u, port_name=d, id=7)
  result: () = output_port(q, name=result, id=8)
}"#;
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    let generated = emit_system_verilog(&package, &Default::default())
        .unwrap()
        .system_verilog;
    assert!(generated.contains("external_leaf"), "{generated}");
    assert!(generated.contains("WIDTH"), "{generated}");
    assert!(generated.contains("pkg::word_t'("), "{generated}");
    assert!(!generated.contains("module model"), "{generated}");
}
