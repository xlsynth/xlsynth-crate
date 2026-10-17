// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::liberty_model::{Cell, LibraryBuilder, Pin, PinDirection};
use xlsynth_g8r::netlist::parse::{AssignExpr, NetRef};
use xlsynth_g8r::techmap::area::{AreaMappingLibrary, map_gatefn_to_area_netlist};

/// Checks the mapped physical pin wiring against the Liberty truth table
/// without evaluating the function through the shared formula parser.
#[test]
fn area_mapping_respects_space_and_and_xor_precedence() {
    for (function, mixed) in [("A B", false), ("A B ^ C", true), ("A * B ^ C", true)] {
        let mut lib = LibraryBuilder::new();
        let pin_names: &[&str] = if mixed { &["A", "B", "C"] } else { &["A", "B"] };
        let mut pins = pin_names
            .iter()
            .map(|name| Pin {
                name: lib.intern_string(name).unwrap(),
                direction: PinDirection::Input as i32,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        pins.push(Pin {
            name: lib.intern_string("Y").unwrap(),
            direction: PinDirection::Output as i32,
            function: lib.intern_string(function).unwrap(),
            ..Default::default()
        });
        lib.cells.push(Cell {
            name: "TEST".into(),
            pins,
            area: 1.0,
            ..Default::default()
        });
        let library = lib.finish();
        let index = AreaMappingLibrary::new(&library).unwrap();

        let mut builder = GateBuilder::new("formula".into(), GateBuilderOptions::no_opt());
        let a = *builder.add_input("a".into(), 1).get_lsb(0);
        let b = *builder.add_input("b".into(), 1).get_lsb(0);
        let rhs = if mixed {
            let c = *builder.add_input("c".into(), 1).get_lsb(0);
            builder.add_xor_binary(b, c)
        } else {
            b
        };
        let output = builder.add_and_binary(a, rhs);
        builder.add_output("out".into(), output.into());
        let mapped = map_gatefn_to_area_netlist(&builder.build(), &index).unwrap();
        let netlist = &mapped.netlist;
        assert_eq!(netlist.module.instances.len(), 1, "{function}");
        let cell = &netlist.module.instances[0];
        assert_eq!(netlist.interner.resolve(cell.type_name), Some("TEST"));
        let connections = cell
            .connections
            .iter()
            .map(|(pin, net)| {
                let NetRef::Simple(net) = net else {
                    panic!("expected a scalar connection for {function}: {net:?}");
                };
                (netlist.interner.resolve(*pin).unwrap(), *net)
            })
            .collect::<HashMap<_, _>>();
        assert_eq!(netlist.module.assigns.len(), 1);
        let NetRef::Simple(out_net) = &netlist.module.assigns[0].lhs else {
            panic!("expected a scalar output assignment for {function}");
        };
        assert_eq!(
            netlist.interner.resolve(netlist.nets[out_net.0].name),
            Some("out")
        );
        assert_eq!(
            netlist.module.assigns[0].rhs,
            AssignExpr::Leaf(NetRef::Simple(connections["Y"]))
        );

        for assignment in 0..1usize << pin_names.len() {
            let values = HashMap::from([
                ("a", assignment & 1 != 0),
                ("b", assignment & 2 != 0),
                ("c", assignment & 4 != 0),
            ]);
            let physical_value = |pin: &str| {
                let net = &netlist.nets[connections[pin].0];
                let name = netlist.interner.resolve(net.name).unwrap();
                values[name]
            };
            let physical_rhs = if mixed {
                physical_value("B") ^ physical_value("C")
            } else {
                physical_value("B")
            };
            let expected_rhs = if mixed {
                values["b"] ^ values["c"]
            } else {
                values["b"]
            };
            assert_eq!(
                physical_value("A") && physical_rhs,
                values["a"] && expected_rhs,
                "{function}, assignment {assignment}"
            );
        }
    }
}
