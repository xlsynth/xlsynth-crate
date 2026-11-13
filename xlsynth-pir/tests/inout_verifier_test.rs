// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_verify::{VerifyError, verify_package};

/// Runs the package verifier on a syntactically valid test fixture.
fn verify(source: &str) -> Result<(), VerifyError> {
    let package = Parser::new(source).parse_package().expect("valid test IR");
    verify_package(&package)
}

#[test]
fn private_pin_dependencies_reach_ancestors_with_only_ordinary_ports() {
    let source = r#"package private_wrapper
block passive(io: bits[1]) {
  ext_inout io(bits[1])
}
block wrapper(x: bits[1], y: bits[1]) {
  instantiation child(block=passive, kind=block)
  x: bits[1] = input_port(name=x, id=1)
  one: bits[1] = literal(value=1, id=2)
  drive_pin: () = ext_instantiation_inout_input(x, one, instantiation=child, port_name=io, id=3)
  read_pin: bits[1] = ext_instantiation_inout_output(instantiation=child, port_name=io, id=4)
  y: () = output_port(read_pin, name=y, id=5)
}
top block grandparent(x: bits[1]) {
  instantiation wrapped(block=wrapper, kind=block)
  x: bits[1] = input_port(name=x, id=6)
  feedback: bits[1] = instantiation_output(instantiation=wrapped, port_name=y, id=7)
  connect: () = instantiation_input(x, instantiation=wrapped, port_name=x, id=8)
}
"#;
    verify(source).unwrap();
    let cyclic = source.replace("instantiation_input(x,", "instantiation_input(feedback,");
    assert!(matches!(
        verify(&cyclic),
        Err(VerifyError::BlockInvariantViolation { block, reason })
            if block == "grandparent" && reason.contains("combinational cycle")
    ));
}

#[test]
fn aliasing_two_child_ports_to_one_net_can_close_a_cycle() {
    let source = r#"package aliases
block child(a: bits[1], b: bits[1]) {
  ext_inout a(bits[1])
  ext_inout b(bits[1])
  read_a: bits[1] = ext_inout_port_input(name=a, id=1)
  one: bits[1] = literal(value=1, id=2)
  drive_b: () = ext_inout_port_output(one, read_a, name=b, id=3)
}
top block parent(left: bits[1], right: bits[1]) {
  ext_inout left(bits[1])
  ext_inout right(bits[1])
  instantiation c(block=child, kind=block)
  bind_a: () = ext_instantiation_inout(instantiation=c, port_name=a, parent_port=left, id=4)
  bind_b: () = ext_instantiation_inout(instantiation=c, port_name=b, parent_port=right, id=5)
}
"#;
    verify(source).unwrap();
    let cyclic = source.replace("parent_port=right", "parent_port=left");
    assert!(matches!(
        verify(&cyclic),
        Err(VerifyError::BlockInvariantViolation { block, reason })
            if block == "parent" && reason.contains("combinational cycle")
    ));
}

#[test]
fn private_pins_are_distinct_per_instance_and_may_be_read_or_drive_only() {
    let source = r#"package independent
block passive(io: bits[1]) {
  ext_inout io(bits[1])
}
top block wrapper() {
  instantiation a(block=passive, kind=block)
  instantiation b(block=passive, kind=block)
  read_a: bits[1] = ext_instantiation_inout_output(instantiation=a, port_name=io, id=1)
  another_read_a: bits[1] = ext_instantiation_inout_output(instantiation=a, port_name=io, id=2)
  drive_b: () = ext_instantiation_inout_input(read_a, another_read_a, instantiation=b, port_name=io, id=3)
}
"#;
    verify(source).unwrap();
    let duplicate_driver = source.replace(
        "port_name=io, id=3)",
        "port_name=io, id=3)\n  other: () = ext_instantiation_inout_input(read_a, another_read_a, instantiation=b, port_name=io, id=4)",
    );
    assert!(matches!(
        verify(&duplicate_driver),
        Err(VerifyError::BlockInvariantViolation { reason, .. })
            if reason.contains("duplicate driver for inout b.io")
    ));
}

#[test]
fn directly_bound_pins_cannot_also_have_private_accesses() {
    let prefix = r#"package mixed
block passive(io: bits[1]) {
  ext_inout io(bits[1])
}
top block wrapper(pin: bits[1]) {
  ext_inout pin(bits[1])
  instantiation c(block=passive, kind=block)
"#;
    let binding =
        "  b: () = ext_instantiation_inout(instantiation=c, port_name=io, parent_port=pin, id=1)";
    let read = "  r: bits[1] = ext_instantiation_inout_output(instantiation=c, port_name=io, id=2)";
    for (first, second) in [(binding, read), (read, binding)] {
        let mixed = format!("{prefix}{first}\n{second}\n}}");
        assert!(matches!(
            verify(&mixed),
            Err(VerifyError::BlockInvariantViolation { reason, .. })
                if reason.contains("cannot both bind and access inout c.io")
        ));
    }
}

#[test]
fn local_reads_and_driver_match_the_declared_pin_type() {
    let source = r#"package local
top block pads(io: bits[4], data: bits[4], oe: bits[1]) {
  ext_inout io(bits[4])
  data: bits[4] = input_port(name=data, id=1)
  oe: bits[1] = input_port(name=oe, id=2)
  read_a: bits[4] = ext_inout_port_input(name=io, id=3)
  read_b: bits[4] = ext_inout_port_input(name=io, id=4)
  drive: () = ext_inout_port_output(data, oe, name=io, id=5)
}
"#;
    verify(source).unwrap();
    for (from, to) in [
        ("name=io, id=4", "name=data, id=4"),
        ("read_a: bits[4]", "read_a: bits[1]"),
        ("output(data, oe", "output(oe, oe"),
        ("output(data, oe", "output(data, data"),
        ("drive: ()", "drive: bits[4]"),
        (
            "read_b: bits[4] = ext_inout_port_input(",
            "read_b: () = ext_inout_port_output(data, oe, ",
        ),
        ("output(data, oe", "output(read_a, oe"),
    ] {
        let invalid = source.replace(from, to);
        assert!(verify(&invalid).is_err(), "unexpectedly accepted {invalid}");
    }
}
