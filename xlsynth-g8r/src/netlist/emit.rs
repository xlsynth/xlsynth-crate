// SPDX-License-Identifier: Apache-2.0

//! Helpers for rendering parsed gate-level netlist structures as text.

use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistModule, PortDirection};
use anyhow::{Result, anyhow};
use std::fmt::Write as FmtWrite;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};
use xlsynth::IrBits;

fn resolve_symbol(
    interner: &StringInterner<StringBackend<SymbolU32>>,
    sym: SymbolU32,
    what: &str,
) -> Result<String> {
    interner
        .resolve(sym)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("could not resolve {what} symbol"))
}

fn net_name(
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    idx: NetIndex,
) -> Result<String> {
    let net = nets
        .get(idx.0)
        .ok_or_else(|| anyhow!("net index {} is out of range ({} nets)", idx.0, nets.len()))?;
    resolve_symbol(interner, net.name, "net")
}

fn width_suffix(width: Option<(u32, u32)>) -> String {
    match width {
        Some((msb, lsb)) => format!("[{}:{}] ", msb, lsb),
        None => "".to_string(),
    }
}

fn is_simple_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "wire" | "module" | "endmodule" | "input" | "output" | "inout" | "assign"
    )
}

fn render_identifier(name: &str) -> String {
    if is_simple_identifier(name) && !is_keyword(name) {
        name.to_string()
    } else {
        format!("\\{} ", name)
    }
}

fn render_literal(bits: &IrBits) -> String {
    let bit_count = bits.get_bit_count();
    let mut value = String::with_capacity(bit_count.max(1));
    if bit_count == 0 {
        value.push('0');
    } else {
        for i in (0..bit_count).rev() {
            value.push(if bits.get_bit(i).unwrap_or(false) {
                '1'
            } else {
                '0'
            });
        }
    }
    format!("{}'b{}", bit_count, value)
}

fn render_netref(
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    netref: &NetRef,
) -> Result<String> {
    match netref {
        NetRef::Simple(idx) => Ok(render_identifier(&net_name(nets, interner, *idx)?)),
        NetRef::BitSelect(idx, bit) => Ok(format!(
            "{}[{}]",
            render_identifier(&net_name(nets, interner, *idx)?),
            bit
        )),
        NetRef::PartSelect(idx, msb, lsb) => Ok(format!(
            "{}[{}:{}]",
            render_identifier(&net_name(nets, interner, *idx)?),
            msb,
            lsb
        )),
        NetRef::Literal(bits) => Ok(render_literal(bits)),
        NetRef::Unconnected => Ok("".to_string()),
        NetRef::Concat(parts) => {
            let mut rendered: Vec<String> = Vec::with_capacity(parts.len());
            for part in parts {
                rendered.push(render_netref(nets, interner, part)?);
            }
            Ok(format!("{{{}}}", rendered.join(", ")))
        }
    }
}

pub fn emit_module_as_netlist_text(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<String> {
    if !module.assigns.is_empty() {
        return Err(anyhow!(
            "netlist emission does not support preserved continuous assigns"
        ));
    }
    let module_name = render_identifier(&resolve_symbol(interner, module.name, "module name")?);
    let mut out = String::new();

    let mut port_names: Vec<String> = Vec::with_capacity(module.ports.len());
    for port in &module.ports {
        port_names.push(render_identifier(&resolve_symbol(
            interner,
            port.name,
            "port name",
        )?));
    }

    if port_names.is_empty() {
        writeln!(&mut out, "module {}();", module_name).unwrap();
    } else {
        writeln!(
            &mut out,
            "module {}({});",
            module_name,
            port_names.join(", ")
        )
        .unwrap();
    }

    for port in &module.ports {
        let port_name = render_identifier(&resolve_symbol(interner, port.name, "port name")?);
        let dir = match port.direction {
            PortDirection::Input => "input",
            PortDirection::Output => "output",
            PortDirection::Inout => "inout",
        };
        writeln!(
            &mut out,
            "  {} {}{};",
            dir,
            width_suffix(port.width),
            port_name
        )
        .unwrap();
    }

    for wire_idx in &module.wires {
        let net = nets.get(wire_idx.0).ok_or_else(|| {
            anyhow!(
                "wire index {} is out of range ({} nets)",
                wire_idx.0,
                nets.len()
            )
        })?;
        let wire_name = render_identifier(&resolve_symbol(interner, net.name, "wire name")?);
        writeln!(&mut out, "  wire {}{};", width_suffix(net.width), wire_name).unwrap();
    }

    for inst in &module.instances {
        let type_name =
            render_identifier(&resolve_symbol(interner, inst.type_name, "instance type")?);
        let inst_name = render_identifier(&resolve_symbol(
            interner,
            inst.instance_name,
            "instance name",
        )?);
        if inst.connections.is_empty() {
            writeln!(&mut out, "  {} {} ();", type_name, inst_name).unwrap();
            continue;
        }
        let mut rendered_conns: Vec<String> = Vec::with_capacity(inst.connections.len());
        for (pin_sym, netref) in &inst.connections {
            let pin = render_identifier(&resolve_symbol(interner, *pin_sym, "pin name")?);
            let rhs = render_netref(nets, interner, netref)?;
            rendered_conns.push(format!(".{}({})", pin, rhs));
        }
        writeln!(
            &mut out,
            "  {} {} ({});",
            type_name,
            inst_name,
            rendered_conns.join(", ")
        )
        .unwrap();
    }

    writeln!(&mut out, "endmodule").unwrap();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::netlist::parse::{Parser as NetlistParser, TokenScanner};
    use crate::netlist::techmap::{StructuralTechMapOptions, map_gatefn_to_structural_netlist};

    #[test]
    fn emitted_techmap_netlist_round_trips_through_parser() {
        let mut gb = GateBuilder::new("round_trip".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("a".to_string(), 1);
        let b = gb.add_input("b".to_string(), 1);
        let y = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        gb.add_output("y".to_string(), y.into());
        let gate_fn = gb.build();

        let mapped =
            map_gatefn_to_structural_netlist(&gate_fn, &StructuralTechMapOptions::default())
                .expect("mapping should succeed");

        let text =
            emit_module_as_netlist_text(&mapped.module, mapped.nets.as_slice(), &mapped.interner)
                .expect("emission should succeed");

        let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        let lookup = move |lineno: u32| lines.get((lineno - 1) as usize).cloned();
        let scanner = TokenScanner::with_line_lookup(
            std::io::Cursor::new(text.into_bytes()),
            Box::new(lookup),
        );
        let mut parser = NetlistParser::new(scanner);
        let modules = parser.parse_file().expect("emitted netlist should parse");
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn emitted_netlist_escapes_non_simple_and_keyword_identifiers() {
        let mut gb = GateBuilder::new("module".to_string(), GateBuilderOptions::no_opt());
        let a = gb.add_input("p0[0]".to_string(), 1);
        let b = gb.add_input("input".to_string(), 1);
        let _unused = gb.add_input("assign".to_string(), 1);
        let y = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
        gb.add_output("output".to_string(), y.into());
        let gate_fn = gb.build();

        let mapped =
            map_gatefn_to_structural_netlist(&gate_fn, &StructuralTechMapOptions::default())
                .expect("mapping should succeed");
        let text =
            emit_module_as_netlist_text(&mapped.module, mapped.nets.as_slice(), &mapped.interner)
                .expect("emission should succeed");

        assert!(text.contains("module \\module "));
        assert!(text.contains("\\p0[0] "));
        assert!(text.contains("\\input "));
        assert!(text.contains("\\assign "));
        assert!(text.contains("\\output "));

        let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        let lookup = move |lineno: u32| lines.get((lineno - 1) as usize).cloned();
        let scanner = TokenScanner::with_line_lookup(
            std::io::Cursor::new(text.into_bytes()),
            Box::new(lookup),
        );
        let mut parser = NetlistParser::new(scanner);
        let modules = parser.parse_file().expect("emitted netlist should parse");
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn emitter_rejects_preserved_continuous_assigns() {
        let text = r#"
module top(a, y);
  input a;
  output y;
  assign y = a;
endmodule
"#;
        let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        let lookup = move |lineno: u32| lines.get((lineno - 1) as usize).cloned();
        let scanner =
            TokenScanner::with_line_lookup(std::io::Cursor::new(text.as_bytes()), Box::new(lookup));
        let mut parser = NetlistParser::new(scanner);
        let mut modules = parser.parse_file().expect("input netlist should parse");
        let module = modules.pop().expect("one module expected");

        let err = emit_module_as_netlist_text(&module, &parser.nets, &parser.interner)
            .expect_err("assign-bearing modules should be rejected");
        assert!(
            err.to_string()
                .contains("does not support preserved continuous assigns")
        );
    }
}
