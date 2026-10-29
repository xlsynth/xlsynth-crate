// SPDX-License-Identifier: Apache-2.0

use super::pb::r#ref::Kind;
use super::{decode_netlist, encode_netlist, pb};
use crate::netlist::emit::emit_module_as_netlist_text;
use crate::netlist::io::{ParsedNetlist, read_gv_from_str};
use anyhow::Result;
use prost::Message;

const MAPPED_GV: &str = r#"
module leaf(bus, y, pad);
  output [72:0] y;
  inout pad;
  input bus;
  wire [1:3] bus;
  wire [0:0] w;
  assign y[72:69] = {{bus[1:2], bus[3]}, 1'bx};
  assign y[68:0] = 69'h10000000000000005;
  tran(pad, w);
  AND2 u1(.A(bus[1]), .B(w), .Y(implicit_wire));
endmodule

module top(w, pad, y);
  input [1:3] w;
  inout pad;
  output [72:0] y;
  leaf child(.bus(w), .y(y), .pad(pad));
  AND2 \gate/2 (.A(w[1]), .B(w[2]), .Y());
  AND2 u3(.B(w[2]), .A(w[1]), .Y());
  AND2 u4(.A(w[1]), .Y());
endmodule
"#;

/// Renders all modules in file order, including implicit nets through their
/// uses.
fn emit(parsed: &ParsedNetlist) -> Result<String> {
    parsed
        .modules
        .iter()
        .map(|module| emit_module_as_netlist_text(module, &parsed.nets, &parsed.interner))
        .collect()
}

#[test]
fn round_trips_structural_gv_and_shares_ordered_interfaces() -> Result<()> {
    let parsed = read_gv_from_str(MAPPED_GV)?;
    let bytes = encode_netlist(&parsed)?;
    assert_eq!(bytes, encode_netlist(&read_gv_from_str(MAPPED_GV)?)?);

    let wire = pb::Netlist::decode(bytes.as_slice())?;
    assert_eq!(wire.interfaces.len(), 4);
    let interface = wire.modules[0].instances[0].interface;
    let top_instances = &wire.modules[1].instances;
    assert_eq!(interface, top_instances[1].interface);
    assert_ne!(interface, top_instances[2].interface);
    assert_ne!(interface, top_instances[3].interface);
    assert_eq!(wire.interfaces[interface as usize].pins, ["A", "B", "Y"]);
    assert_eq!(
        wire.interfaces[top_instances[2].interface as usize].pins,
        ["B", "A", "Y"]
    );
    assert_eq!(
        wire.interfaces[top_instances[3].interface as usize].pins,
        ["A", "Y"]
    );

    let restored = decode_netlist(&bytes)?;
    assert_eq!(encode_netlist(&restored)?, bytes);
    let text = emit(&restored)?;
    assert_eq!(text, emit(&parsed)?);
    assert_eq!(text, emit(&read_gv_from_str(&text)?)?);
    Ok(())
}

#[test]
fn rejects_malformed_references_in_assignments() -> Result<()> {
    let parsed = read_gv_from_str("module top(output y); assign y = 1'b0; endmodule")?;
    let mut wire = pb::Netlist::decode(encode_netlist(&parsed)?.as_slice())?;
    wire.modules[0].assigns[0].rhs = None;
    assert!(decode_netlist(&wire.encode_to_vec()).is_err());

    let open = Kind::Open(pb::Open {});
    let cases = [
        ("missing reference kind", None),
        ("out-of-range net", Some(Kind::Net(1))),
        (
            "wrong literal length",
            Some(Kind::Literal(pb::Literal {
                width: 9,
                little_endian: vec![1],
            })),
        ),
        (
            "nonzero literal padding",
            Some(Kind::Literal(pb::Literal {
                width: 1,
                little_endian: vec![2],
            })),
        ),
        (
            "open inside concat",
            Some(Kind::Concat(pb::Concat {
                parts: vec![pb::Ref {
                    kind: Some(open.clone()),
                }],
            })),
        ),
        ("open assignment operand", Some(open)),
    ];
    for (description, kind) in cases {
        wire.modules[0].assigns[0].rhs = Some(pb::Ref { kind });
        assert!(
            decode_netlist(&wire.encode_to_vec()).is_err(),
            "{description}"
        );
    }
    Ok(())
}

#[test]
fn rejects_invalid_declarations_and_interfaces() -> Result<()> {
    let parsed = read_gv_from_str("module top(input a); INV u(.A(a)); endmodule")?;
    let wire = pb::Netlist::decode(encode_netlist(&parsed)?.as_slice())?;
    let cases: &[(&str, fn(&mut pb::Netlist))] = &[
        ("unknown direction", |m| {
            m.modules[0].ports[0].direction = 42
        }),
        ("bad wire index", |m| m.modules[0].wires.push(1)),
        ("bad port index", |m| m.modules[0].ports[0].net = 1),
        ("bad interface index", |m| {
            m.modules[0].instances[0].interface = 1
        }),
        ("bad packed net index", |m| {
            let instance = &mut m.modules[0].instances[0];
            instance.connections.clear();
            instance.nets = vec![1];
        }),
        ("wrong pin count", |m| m.interfaces[0].pins.push("Y".into())),
        ("duplicate net name", |m| {
            let nets = &mut m.modules[0].nets;
            nets.push(nets[0].clone());
        }),
        ("duplicate instance name", |m| {
            let instances = &mut m.modules[0].instances;
            instances.push(instances[0].clone());
        }),
        ("unrepresentable pin name", |m| {
            m.interfaces[0].pins[0] = "two words".into()
        }),
    ];
    for (description, corrupt) in cases {
        let mut broken = wire.clone();
        corrupt(&mut broken);
        assert!(
            decode_netlist(&broken.encode_to_vec()).is_err(),
            "{description}"
        );
    }
    assert!(decode_netlist(&[0xff]).is_err());
    Ok(())
}

#[test]
fn concat_nesting_limit_keeps_encoded_literals_decodable() -> Result<()> {
    for depth in [48, 49] {
        let source = format!(
            "module top(output y); assign y = {}1'b1{}; endmodule",
            "{".repeat(depth),
            "}".repeat(depth),
        );
        let parsed = read_gv_from_str(&source)?;
        let encoded = encode_netlist(&parsed);
        if depth == 48 {
            assert_eq!(emit(&decode_netlist(&encoded?)?)?, emit(&parsed)?);
        } else {
            assert!(encoded.is_err());
        }
    }
    Ok(())
}
