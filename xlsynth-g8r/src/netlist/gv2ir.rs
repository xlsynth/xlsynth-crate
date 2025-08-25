// SPDX-License-Identifier: Apache-2.0

use crate::gate2ir::gate_fn_to_xlsynth_ir;
use crate::liberty::descriptor::liberty_descriptor_pool;
use crate::liberty_proto::Library;
use crate::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
use crate::netlist::parse::{Parser as NetlistParser, TokenScanner};
use anyhow::{Result, anyhow};
use flate2::bufread::MultiGzDecoder as BufMultiGzDecoder;
use prost::Message;
use prost_reflect::DynamicMessage;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

fn open_reader(path: &Path) -> Result<(Box<dyn Read>, bool)> {
    let is_gz = path.extension().map(|e| e == "gz").unwrap_or(false);
    if is_gz {
        let f = File::open(path)?;
        let br = BufReader::new(f);
        Ok((Box::new(BufMultiGzDecoder::new(br)), true))
    } else {
        let f = File::open(path)?;
        Ok((Box::new(f), false))
    }
}

fn line_lookup(path: &Path, is_gz: bool) -> Box<dyn Fn(u32) -> Option<String>> {
    let path = path.to_path_buf();
    Box::new(move |lineno| {
        let f = File::open(&path).ok()?;
        if is_gz {
            let br = BufReader::new(f);
            let gz = BufMultiGzDecoder::new(br);
            let rdr = BufReader::new(gz);
            rdr.lines().nth((lineno - 1) as usize).and_then(Result::ok)
        } else {
            let rdr = BufReader::new(f);
            rdr.lines().nth((lineno - 1) as usize).and_then(Result::ok)
        }
    })
}

fn load_liberty_proto(path: &Path) -> Result<Library> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    let lib = Library::decode(&buf[..]).or_else(|_| {
        let descriptor_pool = liberty_descriptor_pool();
        let msg_desc = descriptor_pool
            .get_message_by_name("liberty.Library")
            .ok_or_else(|| anyhow!("missing liberty.Library descriptor"))?;
        let dyn_msg = DynamicMessage::parse_text_format(msg_desc, std::str::from_utf8(&buf)?)?;
        let encoded = dyn_msg.encode_to_vec();
        Ok::<Library, anyhow::Error>(Library::decode(&encoded[..])?)
    })?;
    Ok(lib)
}

pub fn convert_gv2ir_paths(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    dff_cells: &HashSet<String>,
) -> Result<String> {
    // Netlist parse
    let (reader, is_gz) = open_reader(netlist_path)?;
    let lookup = line_lookup(netlist_path, is_gz);
    let scanner = TokenScanner::with_line_lookup(reader, lookup);
    let mut parser: NetlistParser<Box<dyn Read>> = NetlistParser::new(scanner);
    let modules = parser.parse_file().map_err(|e| {
        anyhow!(format!(
            "{} at {:?}\n{}\n{}^",
            e.message,
            e.span,
            parser
                .get_line(e.span.start.lineno)
                .unwrap_or_else(|| "<line unavailable>".to_string()),
            " ".repeat((e.span.start.colno as usize).saturating_sub(1))
        ))
    })?;
    if modules.len() != 1 {
        return Err(anyhow!(format!(
            "expected exactly one module, got {}",
            modules.len()
        )));
    }
    let module = &modules[0];

    // Liberty
    let liberty_lib = load_liberty_proto(liberty_proto_path)?;

    // Project GateFn
    let gate_fn = project_gatefn_from_netlist_and_liberty(
        module,
        &parser.nets,
        &parser.interner,
        &liberty_lib,
        dff_cells,
    )
    .map_err(|e| anyhow!(e))?;

    // Convert to IR text
    let flat_type = gate_fn.get_flat_type();
    let ir_pkg = gate_fn_to_xlsynth_ir(&gate_fn, "gate", &flat_type)?;
    Ok(ir_pkg.to_string())
}
