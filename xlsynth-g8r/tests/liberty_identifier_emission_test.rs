// SPDX-License-Identifier: Apache-2.0

//! Tests Liberty names across area mapping and Verilog emission.

use std::io::Write;

use tempfile::NamedTempFile;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::liberty::parser::{
    LibertyPayloadOptions, parse_liberty_files_with_payload_options,
};
use xlsynth_g8r::netlist::emit::emit_module_as_netlist_text;
use xlsynth_g8r::netlist::io::read_gv_from_str;
use xlsynth_g8r::techmap::area::{AreaMappingLibrary, map_gatefn_to_area_netlist};

#[test]
fn liberty_cell_and_pin_names_are_validated_during_verilog_emission() {
    let mut builder = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".into(), 1).get_lsb(0);
    let b = *builder.add_input("b".into(), 1).get_lsb(0);
    let result = builder.add_and_binary(a, b);
    builder.add_output("y".into(), result.into());
    let gatefn = builder.build();

    for (cell, output, invalid) in [
        ("AND 2", "Y", Some("AND 2")),
        ("AND2", "Y 0", Some("Y 0")),
        ("AND/2", "Y[0]", None),
    ] {
        let source = format!(
            r#"library(test) {{
                cell("{cell}") {{ area: 1;
                    pin(A, B) {{ direction: input; }}
                    pin("{output}") {{ direction: output; function: "A * B"; }}
                }}
            }}"#
        );
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(source.as_bytes()).unwrap();
        let library =
            parse_liberty_files_with_payload_options(&[file.path()], LibertyPayloadOptions::all())
                .unwrap();
        let index = AreaMappingLibrary::new(&library).unwrap();
        let mapped = map_gatefn_to_area_netlist(&gatefn, &index).unwrap();
        let emitted = emit_module_as_netlist_text(
            &mapped.netlist.module,
            &mapped.netlist.nets,
            &mapped.netlist.interner,
        );
        if let Some(bad_name) = invalid {
            assert_eq!(
                emitted.unwrap_err().to_string(),
                format!("name cannot be represented as a Verilog identifier: {bad_name:?}")
            );
        } else {
            let parsed = read_gv_from_str(&emitted.unwrap()).unwrap();
            let instance = &parsed.modules[0].instances[0];
            assert_eq!(parsed.interner.resolve(instance.type_name), Some(cell));
            assert!(
                instance
                    .connections
                    .iter()
                    .any(|(pin, _)| { parsed.interner.resolve(*pin) == Some(output) })
            );
        }
    }
}
