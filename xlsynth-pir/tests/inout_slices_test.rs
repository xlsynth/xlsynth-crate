// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::BlockBuilder;
use xlsynth_pir::ir::{InoutBinding, InoutPiece, Type};
use xlsynth_pir::ir_parser::Parser;

const PORT: &str = r#"package bits

top block pads(io: bits[4]) {
  ext_inout_net bus(bits[4])
  io: () = ext_inout_port(name=io, net={bus[2:1], bus[3], bus[0]}, id=1)
}
"#;

#[test]
fn binding_width_shape_bounds_and_bit_coverage_are_checked() {
    let pkg = Parser::new(PORT).parse_and_verify_package().unwrap();
    assert_eq!(pkg.to_string(), PORT);
    let expr = "{bus[2:1], bus[3], bus[0]}";
    for (from, to, reason) in [
        (expr, "bus[2:0]", "width 3"),
        (expr, "bus[0:3]", "invalid select"),
        (expr, "bus[4:1]", "invalid select"),
        (expr, "{bus[2:1], bad[1:0]}", "undeclared inout net"),
        (expr, "{bus[1:0], bus[1:0]}", "uncovered bits [3:2]"),
        ("io: bits[4]", "io: bits[2][2]", "require a bits port"),
        ("bus(bits[4])", "bus(bits[2][2])", "require a bits net"),
    ] {
        let ir = PORT.replace(from, to);
        let error = Parser::new(&ir).parse_and_verify_package().unwrap_err();
        assert!(error.to_string().contains(reason), "{ir}: {error}");
    }
    for invalid in [
        "{}",
        "bus[-1]",
        "bus[bad]",
        "{bus[0],}",
        "bus[184467440737095516160:0]",
    ] {
        assert!(
            Parser::new(&PORT.replace(expr, invalid))
                .parse_package()
                .is_err(),
            "{invalid}"
        );
    }
    for access in [
        "read: bits[4] = ext_inout_read(net=bus, id=2)",
        "d: bits[4] = literal(value=0, id=2) oe: bits[1] = literal(value=0, id=3) drv: () = ext_inout_drive(d, oe, net=bus, id=4)",
    ] {
        let ir = PORT
            .replace(expr, "{bus[0], bus[0], bus[0], bus[0]}")
            .replace("\n}", &format!("\n  {access}\n}}"));
        Parser::new(&ir).parse_and_verify_package().unwrap();
    }
}

#[test]
fn builder_checks_handles_before_installing_a_composite_binding() {
    let mut child = BlockBuilder::new("child");
    let x = child.declare_inout_net("x", Type::Bits(4)).unwrap();
    let pieces = |net| {
        InoutBinding::Concat(vec![
            InoutPiece::Select { net, hi: 1, lo: 0 },
            InoutPiece::Select { net, hi: 3, lo: 2 },
        ])
    };
    child.ext_inout_port("p", Type::Bits(4), pieces(x)).unwrap();
    let mut pkg = child.build_package("pins").unwrap();
    let mut parent = BlockBuilder::new("parent");
    let y = parent.declare_inout_net("y", Type::Bits(4)).unwrap();
    assert!(
        parent
            .ext_inout_port("p", Type::Bits(4), pieces(x))
            .is_err()
    );
    parent.ext_inout_port("p", Type::Bits(4), y).unwrap();
    let u = parent
        .instantiate("u", pkg.get_block("child").unwrap())
        .unwrap();
    assert!(parent.bind_inout_net(u, "p", pieces(x)).is_err());
    parent.bind_inout_net(u, "p", pieces(y)).unwrap();
    assert!(parent.bind_inout_net(u, "p", y).is_err());
    parent.build_into_package(&mut pkg).unwrap();
    let text = pkg.to_string();
    assert_eq!(
        Parser::new(&text)
            .parse_and_verify_package()
            .unwrap()
            .to_string(),
        text
    );
}

#[test]
fn swapped_lanes_and_repeated_bits_allow_inout_feedback_across_two_levels() {
    let ir = r#"package wiring
block leaf(io: bits[2], sense: bits[1], data: bits[1]) {
  ext_inout_net hi(bits[1])
  ext_inout_net lo(bits[1])
  io: () = ext_inout_port(name=io, net={hi, lo}, id=1)
  x: bits[1] = ext_inout_read(net=lo, id=2)
  sense: () = output_port(x, name=sense, id=3)
  data: bits[1] = input_port(name=data, id=4)
  one: bits[1] = literal(value=1, id=5)
  tx: () = ext_inout_drive(data, one, net=hi, id=6)
}
block relay(p: bits[2], y: bits[1], d: bits[1]) {
  ext_inout_net bus(bits[2])
  instantiation u(block=leaf, kind=block)
  p: () = ext_inout_port(name=p, net=bus, id=7)
  c: () = ext_instantiation_inout(instantiation=u, port_name=io, net={bus[0], bus[1]}, id=8)
  x: bits[1] = instantiation_output(instantiation=u, port_name=sense, id=9)
  y: () = output_port(x, name=y, id=10)
  d: bits[1] = input_port(name=d, id=11)
  set: () = instantiation_input(d, instantiation=u, port_name=data, id=12)
}
top block host(pad: bits[2]) {
  ext_inout_net wire(bits[2])
  instantiation r(block=relay, kind=block)
  pin: () = ext_inout_port(name=pad, net=wire, id=13)
  bind: () = ext_instantiation_inout(instantiation=r, port_name=p, net=wire, id=14)
  sensed: bits[1] = instantiation_output(instantiation=r, port_name=y, id=15)
  feed: () = instantiation_input(sensed, instantiation=r, port_name=d, id=16)
}
"#;
    for (drive, loop_drive) in [("data, one", "data, one"), ("data, one", "one, data")] {
        let base = ir.replace(drive, loop_drive);
        Parser::new(&base).parse_and_verify_package().unwrap();
        for short in [
            "port_name=p, net={wire[1], wire[1]}",
            "port_name=p, net={wire[0], wire[0]}",
        ] {
            let cyclic = base.replace("port_name=p, net=wire", short);
            Parser::new(&cyclic).parse_and_verify_package().unwrap();
        }
    }
}

#[test]
fn fixed_slices_preserve_ordinary_port_dependencies_in_an_inout_hierarchy() {
    let ir = r#"package feedback
block route(d: bits[2], o: bits[2]) {
  d: bits[2] = input_port(name=d, id=1)
  high: bits[1] = bit_slice(d, start=1, width=1, id=2)
  low: bits[1] = bit_slice(d, start=0, width=1, id=3)
  swapped: bits[2] = concat(low, high, id=4)
  o: () = output_port(swapped, name=o, id=5)
}
top block parent(pad: bits[1]) {
  ext_inout_net pin(bits[1])
  pad: () = ext_inout_port(name=pad, net=pin, id=6)
  instantiation u(block=route, kind=block)
  obs: bits[2] = instantiation_output(instantiation=u, port_name=o, id=7)
  bit: bits[1] = bit_slice(obs, start=1, width=1, id=8)
  zero: bits[1] = literal(value=0, id=9)
  data: bits[2] = concat(bit, zero, id=10)
  write: () = instantiation_input(data, instantiation=u, port_name=d, id=11)
}
"#;
    // o[1] depends on the constant d[0]; o[0] depends on the fed-back d[1].
    Parser::new(ir).parse_and_verify_package().unwrap();
    let cyclic = ir.replace("bit_slice(obs, start=1", "bit_slice(obs, start=0");
    let error = Parser::new(&cyclic).parse_and_verify_package().unwrap_err();
    assert!(error.to_string().contains("combinational cycle"), "{error}");
}

#[test]
fn wide_unfragmented_bindings_use_ranges() {
    let ir = PORT
        .replace("bits[4]", "bits[1000000000]")
        .replace("{bus[2:1], bus[3], bus[0]}", "{bus[999999999:1], bus[0]}");
    Parser::new(&ir).parse_and_verify_package().unwrap();
}

#[test]
fn child_passive_ties_allow_feedback_in_ancestors() {
    let source = r#"package ties
block leaf(p: bits[4]) {
  ext_inout_net shared(bits[1])
  ext_inout_net lanes(bits[2])
  p: () = ext_inout_port(name=p, net={shared, lanes, shared}, id=1)
}
block relay(io: bits[4]) {
  ext_inout_net bus(bits[4])
  io: () = ext_inout_port(name=io, net=bus, id=2)
  instantiation u(block=leaf, kind=block)
  conn: () = ext_instantiation_inout(instantiation=u, port_name=p, net={bus[1:0], bus[3:2]}, id=3)
}
top block host() {
  ext_inout_net high(bits[2])
  ext_inout_net low(bits[2])
  instantiation u(block=relay, kind=block)
  joined: () = ext_instantiation_inout(instantiation=u, port_name=io, net={high, low}, id=4)
  read: bits[2] = ext_inout_read(net=low, id=5)
  x: bits[1] = bit_slice(read, start=0, width=1, id=6)
  one: bits[1] = literal(value=1, id=7)
  data: bits[2] = concat(one, x, id=8)
  tx: () = ext_inout_drive(data, one, net=high, id=9)
}
"#;
    // The leaf joins relay bits 1 and 2, which are low[1] and high[0].
    Parser::new(source).parse_and_verify_package().unwrap();
    let cyclic = source.replace("start=0", "start=1");
    Parser::new(&cyclic).parse_and_verify_package().unwrap();
    let isolated = cyclic
        .replace("shared(bits[1])", "shared(bits[2])")
        .replace("{shared, lanes, shared}", "{shared[1], lanes, shared[0]}");
    Parser::new(&isolated).parse_and_verify_package().unwrap();
}

#[test]
fn builder_compares_public_alias_bits_independently_of_net_spelling() {
    let source = r#"package child_pins
block cell(p: bits[4]) {
  ext_inout_net n(bits[2])
  p: () = ext_inout_port(name=p, net={n, n}, id=1)
}
"#;
    let original = Parser::new(source).parse_and_verify_package().unwrap();
    for (expr, accepted) in [("{n[1], n[0], n}", true), ("{n[0], n[1], n}", false)] {
        let ir = source
            .replace("{n, n}", expr)
            .replace("n(bits", "renamed(bits")
            .replace("n[", "renamed[")
            .replace(", n}", ", renamed}");
        let mut pkg = Parser::new(&ir).parse_and_verify_package().unwrap();
        let mut parent = BlockBuilder::new("parent");
        let instance = parent
            .instantiate("u", original.get_block("cell").unwrap())
            .unwrap();
        let net = parent.declare_inout_net("pin", Type::Bits(4)).unwrap();
        parent.bind_inout_net(instance, "p", net).unwrap();
        assert_eq!(
            parent.build_into_package(&mut pkg).is_ok(),
            accepted,
            "{ir}"
        );
    }
}

#[test]
fn binding_width_overflow_is_reported_before_elaboration() {
    let source = PORT
        .replace("bus(bits[4])", &format!("bus(bits[{}])", usize::MAX))
        .replace("{bus[2:1], bus[3], bus[0]}", "{bus, bus}");
    let err = Parser::new(&source).parse_and_verify_package().unwrap_err();
    assert!(err.to_string().contains("width overflow"), "{err}");
}
