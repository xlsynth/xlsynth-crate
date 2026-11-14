// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{BlockPort, InoutNet, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_verify::{VerifyError, verify_package};

/// Runs the package verifier on a syntactically valid test fixture.
fn verify(source: &str) -> Result<(), VerifyError> {
    let package = Parser::new(source).parse_package().expect("valid test IR");
    verify_package(&package)
}

#[test]
fn private_net_dependencies_reach_ancestors_with_only_ordinary_ports() {
    let source = r#"package private_wrapper
block wrapper(x: bits[1], y: bits[1]) {
  ext_inout_net bus(bits[1])
  x: bits[1] = input_port(name=x, id=1)
  one: bits[1] = literal(value=1, id=2)
  drive_bus: () = ext_inout_drive(x, one, net=bus, id=3)
  read_bus: bits[1] = ext_inout_read(net=bus, id=4)
  y: () = output_port(read_bus, name=y, id=5)
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
fn child_public_aliases_join_distinct_parent_nets_transitively() {
    let source = r#"package aliases
block tie(x: bits[1], y: bits[1]) {
  ext_inout_net shorted(bits[1], ports=[y, x])
}
block relay(a: bits[1], b: bits[1]) {
  ext_inout_net left(bits[1], ports=[a])
  ext_inout_net right(bits[1], ports=[b])
  instantiation wire(block=tie, kind=block)
  x: () = ext_instantiation_inout(instantiation=wire, port_name=x, net=left, id=1)
  y: () = ext_instantiation_inout(instantiation=wire, port_name=y, net=right, id=2)
}
top block top(clk: clock, data: bits[1]) {
  reg state(bits[1])
  ext_inout_net p(bits[1])
  ext_inout_net q(bits[1])
  instantiation wrapped(block=relay, kind=block)
  a: () = ext_instantiation_inout(instantiation=wrapped, port_name=a, net=p, id=3)
  b: () = ext_instantiation_inout(instantiation=wrapped, port_name=b, net=q, id=4)
  data: bits[1] = input_port(name=data, id=5)
  sampled: bits[1] = ext_inout_read(net=p, id=6)
  written: () = register_write(sampled, register=state, id=7)
  delayed: bits[1] = register_read(register=state, id=8)
  drive: () = ext_inout_drive(delayed, data, net=q, id=9)
}
"#;
    verify(source).unwrap();
    // Both declarations may have local drivers even when a child joins them.
    let two_drivers = source.replace(
        "net=q, id=9)",
        "net=q, id=9)\n  drive_p: () = ext_inout_drive(data, data, net=p, id=10)",
    );
    verify(&two_drivers).unwrap();
    for operands in ["sampled, data", "data, sampled"] {
        let cyclic = source.replace("drive(delayed, data", &format!("drive({operands}"));
        assert!(matches!(
            verify(&cyclic),
            Err(VerifyError::BlockInvariantViolation { block, reason })
                if block == "top" && reason.contains("combinational cycle")
        ));
        let independent = cyclic.replace(
            "shorted(bits[1], ports=[y, x])",
            "x_net(bits[1], ports=[x])\n  ext_inout_net y_net(bits[1], ports=[y])",
        );
        verify(&independent).unwrap();
    }
}

#[test]
fn shared_private_nets_collect_dependencies_from_each_instance() {
    let source = r#"package sibling_drivers
block cell(io: bits[1], data: bits[1], sense: bits[1]) {
  ext_inout_net pad(bits[1], ports=[io])
  data: bits[1] = input_port(name=data, id=1)
  one: bits[1] = literal(value=1, id=2)
  drive: () = ext_inout_drive(data, one, net=pad, id=3)
  read: bits[1] = ext_inout_read(net=pad, id=4)
  sense: () = output_port(read, name=sense, id=5)
}
top block container(x: bits[1]) {
  ext_inout_net shared(bits[1])
  ext_inout_net isolated(bits[1])
  instantiation a(block=cell, kind=block)
  instantiation b(block=cell, kind=block)
  x: bits[1] = input_port(name=x, id=6)
  a_pin: () = ext_instantiation_inout(instantiation=a, port_name=io, net=shared, id=7)
  b_pin: () = ext_instantiation_inout(instantiation=b, port_name=io, net=shared, id=8)
  a_data: () = instantiation_input(x, instantiation=a, port_name=data, id=9)
  a_sense: bits[1] = instantiation_output(instantiation=a, port_name=sense, id=10)
  b_sense: bits[1] = instantiation_output(instantiation=b, port_name=sense, id=11)
  b_data: () = instantiation_input(x, instantiation=b, port_name=data, id=12)
}
"#;
    verify(source).unwrap();
    let cyclic = source.replace(
        "instantiation_input(x, instantiation=b",
        "instantiation_input(a_sense, instantiation=b",
    );
    assert!(matches!(
        verify(&cyclic),
        Err(VerifyError::BlockInvariantViolation { block, reason })
            if block == "container" && reason.contains("combinational cycle")
    ));
    verify(&cyclic.replace("net=shared, id=8", "net=isolated, id=8")).unwrap();
}

#[test]
fn binding_independent_child_ports_to_the_same_net_can_close_a_cycle() {
    let source = r#"package local_short
block bridge(a: bits[1], b: bits[1]) {
  ext_inout_net input_wire(bits[1], ports=[a])
  ext_inout_net output_wire(bits[1], ports=[b])
  read_a: bits[1] = ext_inout_read(net=input_wire, id=1)
  one: bits[1] = literal(value=1, id=2)
  drive_b: () = ext_inout_drive(one, read_a, net=output_wire, id=3)
}
top block parent() {
  ext_inout_net p(bits[1])
  ext_inout_net q(bits[1])
  instantiation c(block=bridge, kind=block)
  bind_a: () = ext_instantiation_inout(instantiation=c, port_name=a, net=p, id=4)
  bind_b: () = ext_instantiation_inout(instantiation=c, port_name=b, net=q, id=5)
}
"#;
    verify(source).unwrap();
    assert!(matches!(
        verify(&source.replace("net=q, id=5", "net=p, id=5")),
        Err(VerifyError::BlockInvariantViolation { block, reason })
            if block == "parent" && reason.contains("combinational cycle")
    ));
}

#[test]
fn bindings_require_each_child_alias_once_with_exact_aggregate_type() {
    let source = r#"package types
block cell(io: bits[2][2], other: bits[2][2]) {
  ext_inout_net tied(bits[2][2], ports=[io, other])
}
top block parent() {
  ext_inout_net bus(bits[2][2])
  instantiation a(block=cell, kind=block)
  first: () = ext_instantiation_inout(instantiation=a, port_name=io, net=bus, id=1)
  second: () = ext_instantiation_inout(instantiation=a, port_name=other, net=bus, id=2)
}
"#;
    verify(source).unwrap();
    for (from, to, expected) in [
        ("bus(bits[2][2])", "bus(bits[4])", "type bits[4]"),
        (
            "port_name=other",
            "port_name=io",
            "duplicate inout binding a.io",
        ),
        (
            "port_name=other",
            "port_name=unknown",
            "no block instance inout named a.unknown",
        ),
        ("net=bus, id=2", "net=io, id=2", "undeclared inout net 'io'"),
        (
            "second: () = ext_instantiation_inout(instantiation=a, port_name=other, net=bus, id=2)",
            "",
            "missing inout binding a.other",
        ),
    ] {
        let error = verify(&source.replace(from, to)).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn malformed_net_declarations_are_rejected_even_without_accesses() {
    let source = r#"package declarations
top block top(io: bits[4]) {
  ext_inout_net bus(bits[4], ports=[io])
}
"#;
    let original = Parser::new(source).parse_and_verify_package().unwrap();
    for (name, ty, ports) in [
        ("duplicate_alias", Type::Bits(4), vec!["io".to_string()]),
        ("missing_port", Type::Bits(4), vec!["unknown".to_string()]),
        ("bus", Type::Bits(4), vec![]),
        ("", Type::Bits(4), vec![]),
        ("empty", Type::Bits(0), vec![]),
        (
            "tokens",
            Type::Tuple(vec![Box::new(Type::Bits(4)), Box::new(Type::Token)]),
            vec![],
        ),
    ] {
        let mut package = original.clone();
        package
            .get_block_mut("top")
            .unwrap()
            .inout_nets
            .push(InoutNet {
                name: name.into(),
                ty,
                ports,
                sv_type: None,
            });
        assert!(verify_package(&package).is_err(), "{name:?}");
    }
    let mut package = original;
    package
        .get_block_mut("top")
        .unwrap()
        .ports
        .push(BlockPort::Inout("orphan".into()));
    assert!(verify_package(&package).is_err());
}

#[test]
fn local_reads_and_driver_match_the_declared_net_type() {
    let source = r#"package local
top block pads(io: bits[4], data: bits[4], oe: bits[1]) {
  ext_inout_net bus(bits[4], ports=[io])
  data: bits[4] = input_port(name=data, id=1)
  oe: bits[1] = input_port(name=oe, id=2)
  read_a: bits[4] = ext_inout_read(net=bus, id=3)
  read_b: bits[4] = ext_inout_read(net=bus, id=4)
  drive: () = ext_inout_drive(data, oe, net=bus, id=5)
}
"#;
    verify(source).unwrap();
    // Ordinary inputs and net declarations have independent namespaces.
    verify(
        &source
            .replace("bus(", "data(")
            .replace("net=bus", "net=data"),
    )
    .unwrap();
    for (from, to) in [
        ("net=bus, id=4", "net=data, id=4"),
        ("net=bus, id=4", "net=io, id=4"),
        ("read_a: bits[4]", "read_a: bits[1]"),
        ("drive(data, oe", "drive(oe, oe"),
        ("drive(data, oe", "drive(data, data"),
        ("drive: ()", "drive: bits[4]"),
        (
            "read_b: bits[4] = ext_inout_read(",
            "read_b: () = ext_inout_drive(data, oe, ",
        ),
        ("drive(data, oe", "drive(read_a, oe"),
    ] {
        let invalid = source.replace(from, to);
        assert!(verify(&invalid).is_err(), "unexpectedly accepted {invalid}");
    }
}
