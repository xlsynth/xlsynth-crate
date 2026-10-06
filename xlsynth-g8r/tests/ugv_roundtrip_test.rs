// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::{GateFn, SequentialGateFn};
use xlsynth_g8r::aig_serdes::emit_netlist::{
    NetlistPortStyle, emit_netlist_with_version_and_port_style,
};
use xlsynth_g8r::aig_serdes::gate2ir::{
    GateFnInterfacePort, GateFnInterfaceSchema, repack_gate_fn_interface_with_schema,
};
use xlsynth_g8r::netlist::assigns_to_gatefn::project_gatefn_from_structural_assigns;
use xlsynth_g8r::netlist::io::{read_ugv_from_str, select_module};
use xlsynth_g8r::test_utils::{
    interesting_ir_roundtrip_cases, load_interesting_ir_roundtrip_case, structurally_equivalent,
};
use xlsynth_g8r::verilog_version::VerilogVersion;
use xlsynth_pir::ir::Type;

fn interface_schema(gate_fn: &GateFn) -> GateFnInterfaceSchema {
    GateFnInterfaceSchema {
        input_ports: gate_fn
            .inputs
            .iter()
            .map(|port| GateFnInterfacePort {
                name: port.name.clone(),
                ty: Type::Bits(port.get_bit_count()),
            })
            .collect(),
        output_ports: gate_fn
            .outputs
            .iter()
            .map(|port| GateFnInterfacePort {
                name: port.name.clone(),
                ty: Type::Bits(port.get_bit_count()),
            })
            .collect(),
        return_type: gate_fn.get_flat_type().return_type,
    }
}

fn expected_ports<'a>(
    ports: impl Iterator<Item = (&'a str, usize)>,
    style: NetlistPortStyle,
) -> Vec<(String, usize)> {
    ports
        .flat_map(|(name, width)| match style {
            NetlistPortStyle::PackedBits => vec![(name.to_string(), width)],
            NetlistPortStyle::ScalarBits if width == 1 => vec![(name.to_string(), 1)],
            NetlistPortStyle::ScalarBits => {
                (0..width).map(|bit| (format!("{name}_{bit}"), 1)).collect()
            }
        })
        .collect()
}

#[test]
fn combinational_ugv_roundtrips_interesting_signatures() {
    for case in interesting_ir_roundtrip_cases() {
        let sample = load_interesting_ir_roundtrip_case(case);
        if sample
            .gate_fn
            .inputs
            .iter()
            .any(|port| port.get_bit_count() == 0)
            || sample
                .gate_fn
                .outputs
                .iter()
                .any(|port| port.get_bit_count() == 0)
        {
            // Verilog has no zero-width port with which to preserve this
            // interface.
            continue;
        }
        let schema = interface_schema(&sample.gate_fn);
        let design = SequentialGateFn::from_gate_fn(sample.gate_fn.clone());
        for version in [VerilogVersion::Verilog, VerilogVersion::SystemVerilog] {
            for style in [NetlistPortStyle::ScalarBits, NetlistPortStyle::PackedBits] {
                let text =
                    emit_netlist_with_version_and_port_style(&design, version, style).unwrap();
                let parsed = read_ugv_from_str(&text).unwrap_or_else(|error| {
                    panic!("{} {version:?} {style:?}: {error}\n{text}", case.name)
                });
                let module = select_module(&parsed, None).unwrap();
                let rebuilt =
                    project_gatefn_from_structural_assigns(module, &parsed.nets, &parsed.interner)
                        .unwrap();
                assert_eq!(
                    rebuilt
                        .inputs
                        .iter()
                        .map(|port| (port.name.clone(), port.get_bit_count()))
                        .collect::<Vec<_>>(),
                    expected_ports(
                        sample
                            .gate_fn
                            .inputs
                            .iter()
                            .map(|port| (port.name.as_str(), port.get_bit_count())),
                        style
                    ),
                    "{} {version:?} {style:?}",
                    case.name
                );
                assert_eq!(
                    rebuilt
                        .outputs
                        .iter()
                        .map(|port| (port.name.clone(), port.get_bit_count()))
                        .collect::<Vec<_>>(),
                    expected_ports(
                        sample
                            .gate_fn
                            .outputs
                            .iter()
                            .map(|port| (port.name.as_str(), port.get_bit_count())),
                        style
                    ),
                    "{} {version:?} {style:?}",
                    case.name
                );
                let rebuilt = repack_gate_fn_interface_with_schema(rebuilt, &schema).unwrap();
                assert_eq!(
                    rebuilt.get_flat_type(),
                    sample.gate_fn.get_flat_type(),
                    "{} {version:?} {style:?}",
                    case.name
                );
                assert!(
                    structurally_equivalent(&sample.gate_fn, &rebuilt),
                    "{} {version:?} {style:?}",
                    case.name
                );
            }
        }
    }
}
