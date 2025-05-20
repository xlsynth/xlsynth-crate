// SPDX-License-Identifier: Apache-2.0

use prost::Message;
use prost_reflect::DynamicMessage;
use std::fs::File;
use std::io::Read;
use xlsynth_g8r::gate2ir::gate_fn_to_xlsynth_ir;
use xlsynth_g8r::liberty::descriptor::liberty_descriptor_pool;
use xlsynth_g8r::liberty_proto::Library;
use xlsynth_g8r::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
use xlsynth_g8r::netlist::parse::{Parser as NetlistParser, TokenScanner};

pub fn handle_gv2ir(matches: &clap::ArgMatches) {
    let netlist_path = matches.get_one::<String>("netlist").unwrap();
    let liberty_proto_path = matches.get_one::<String>("liberty_proto").unwrap();

    // Parse netlist
    let file = File::open(netlist_path).expect("failed to open netlist");
    let scanner = TokenScanner::from_file_with_path(file, netlist_path.into());
    let mut parser = NetlistParser::new(scanner);
    let modules = match parser.parse_file() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to parse netlist: {} at {:?}", e.message, e.span);
            if let Some(line) = parser.get_line(e.span.start.lineno) {
                eprintln!("{}", line);
                let col = (e.span.start.colno as usize).saturating_sub(1);
                eprintln!("{}^", " ".repeat(col));
            }
            std::process::exit(1);
        }
    };
    assert_eq!(modules.len(), 1);
    let module = &modules[0];

    // Parse Liberty proto (try binary, then textproto)
    let mut buf = Vec::new();
    File::open(liberty_proto_path)
        .expect("failed to open liberty proto")
        .read_to_end(&mut buf)
        .expect("failed to read liberty proto");
    let liberty_lib = Library::decode(&buf[..])
        .or_else(|_| {
            // Try textproto using the canonical embedded descriptor
            let descriptor_pool = liberty_descriptor_pool();
            let msg_desc = descriptor_pool
                .get_message_by_name("liberty.Library")
                .unwrap();
            let dyn_msg =
                DynamicMessage::parse_text_format(msg_desc, std::str::from_utf8(&buf).unwrap())
                    .unwrap();
            let encoded = dyn_msg.encode_to_vec();
            Ok::<Library, prost::DecodeError>(Library::decode(&encoded[..]).unwrap())
        })
        .expect("failed to decode liberty proto");

    // Parse dff_cells flag
    let dff_cells: std::collections::HashSet<String> = matches
        .get_one::<String>("dff_cells")
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    // Project to GateFn
    let gate_fn = match project_gatefn_from_netlist_and_liberty(
        module,
        &parser.nets,
        &parser.interner,
        &liberty_lib,
        &dff_cells,
    ) {
        Ok(gf) => gf,
        Err(e) => {
            eprintln!("Failed to project GateFn: {}", e);
            std::process::exit(1);
        }
    };

    // Convert to IR and print
    let flat_type = gate_fn.get_flat_type();
    let ir_pkg = gate_fn_to_xlsynth_ir(&gate_fn, "gate", &flat_type).unwrap();
    println!("{}", ir_pkg.to_string());
}
