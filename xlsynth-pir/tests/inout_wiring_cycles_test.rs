// SPDX-License-Identifier: Apache-2.0

//! Structural cycle refinement preserves resolved-net and passive-alias
//! semantics.

use xlsynth_pir::BuilderError;
use xlsynth_pir::ir_builder::{verify_block_combinational_cycles, verify_block_wiring_cycles};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_verify::verify_package;

/// Builds a four-bit resolved-net recurrence, optionally crossing a child
/// alias.
fn feedback(aliased: bool, start: usize) -> String {
    let child = if aliased {
        r#"block tie(a: bits[4], b: bits[4]) {
  ext_inout_net common(bits[4])
  a: () = ext_inout_port(name=a, net=common, id=10)
  b: () = ext_inout_port(name=b, net=common, id=11)
}
"#
    } else {
        ""
    };
    let extra_net = if aliased {
        "  ext_inout_net other(bits[4])\n  instantiation short(block=tie, kind=block)\n  a: () = ext_instantiation_inout(instantiation=short, port_name=a, net=bus, id=12)\n  b: () = ext_instantiation_inout(instantiation=short, port_name=b, net=other, id=13)\n"
    } else {
        ""
    };
    let read_net = if aliased { "other" } else { "bus" };
    format!(
        r#"package inout_feedback
{child}top block root(seed: bits[1], result: bits[4]) {{
  ext_inout_net bus(bits[4])
{extra_net}  seed: bits[1] = input_port(name=seed, id=1)
  one: bits[1] = literal(value=1, id=2)
  sample: bits[4] = ext_inout_read(net={read_net}, id=3)
  lower: bits[3] = bit_slice(sample, start={start}, width=3, id=4)
  data: bits[4] = concat(lower, seed, id=5)
  drive: () = ext_inout_drive(data, one, net=bus, id=6)
  result: () = output_port(sample, name=result, id=7)
}}"#
    )
}

#[test]
fn false_feedback_through_a_resolved_net_or_passive_child_alias_is_valid() {
    for aliased in [false, true] {
        let source = feedback(aliased, 0);
        let package = Parser::new(&source).parse_and_verify_package().unwrap();
        let top = package.get_top_block().unwrap();
        assert!(matches!(
            verify_block_combinational_cycles(top, &package),
            Err(BuilderError::CombinationalCycle { .. })
        ));
        verify_block_wiring_cycles(top, &package).unwrap();
    }
}

#[test]
fn true_feedback_through_a_resolved_net_or_passive_child_alias_is_rejected() {
    for aliased in [false, true] {
        let source = feedback(aliased, 1);
        let package = Parser::new(&source).parse_package().unwrap();
        assert!(verify_package(&package).is_err());
        assert!(matches!(
            verify_block_wiring_cycles(package.get_top_block().unwrap(), &package),
            Err(BuilderError::CombinationalCycle { .. })
        ));
    }
}

#[test]
fn modeled_external_private_net_drivers_are_traced() {
    let source = r#"package extern_feedback
block model(data: bits[4], sample: bits[4]) {
  ext_inout_net bus(bits[4])
  data: bits[4] = input_port(name=data, id=1)
  one: bits[1] = literal(value=1, id=2)
  drive: () = ext_inout_drive(data, one, net=bus, id=3)
  read: bits[4] = ext_inout_read(net=bus, id=4)
  sample: () = output_port(read, name=sample, id=5)
}
top block root(seed: bits[1], result: bits[4]) {
  instantiation device(block=model, module_name=external_device, kind=behavioral_extern_block)
  seed: bits[1] = input_port(name=seed, id=6)
  q: bits[4] = instantiation_output(instantiation=device, port_name=sample, id=7)
  low: bits[3] = bit_slice(q, start=0, width=3, id=8)
  d: bits[4] = concat(low, seed, id=9)
  bind: () = instantiation_input(d, instantiation=device, port_name=data, id=10)
  result: () = output_port(q, name=result, id=11)
}"#;
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    verify_block_wiring_cycles(package.get_top_block().unwrap(), &package).unwrap();
    let cyclic = source.replace("start=0, width=3", "start=1, width=3");
    let package = Parser::new(&cyclic).parse_package().unwrap();
    assert!(verify_package(&package).is_err());
}

#[test]
fn a_separate_false_feedback_cone_cannot_hide_a_real_inout_cycle() {
    let source = r#"package mixed_feedback
block identity(d: bits[2], q: bits[2]) {
  d: bits[2] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(seed: bits[1]) {
  ext_inout_net bus(bits[1])
  instantiation u(block=identity, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[2] = instantiation_output(instantiation=u, port_name=q, id=4)
  low: bits[1] = bit_slice(q, start=0, width=1, id=5)
  d: bits[2] = concat(low, seed, id=6)
  bind: () = instantiation_input(d, instantiation=u, port_name=d, id=7)
  one: bits[1] = literal(value=1, id=8)
  read: bits[1] = ext_inout_read(net=bus, id=9)
  drive: () = ext_inout_drive(one, read, net=bus, id=10)
}"#;
    let package = Parser::new(source).parse_package().unwrap();
    assert!(matches!(
        verify_block_wiring_cycles(package.get_top_block().unwrap(), &package),
        Err(BuilderError::CombinationalCycle { .. })
    ));
}
