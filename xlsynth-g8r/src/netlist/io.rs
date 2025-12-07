// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for reading and parsing gate-level netlists from disk.
//!
//! This module centralizes the logic for:
//! - Handling plain `.gv` and `.gv.gz` inputs.
//! - Wiring up `TokenScanner::with_line_lookup` so that parse errors can show
//!   source-line context.
//! - Producing the parsed modules together with the global `nets` array and
//!   `StringInterner`.

use crate::liberty::descriptor::liberty_descriptor_pool;
use crate::liberty_proto::Library;
use crate::netlist::parse::{Net, NetlistModule, Parser as NetlistParser, TokenScanner};
use anyhow::{Result, anyhow};
use flate2::read::MultiGzDecoder;
use prost::Message;
use prost_reflect::DynamicMessage;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Parsed netlist plus the global nets and interner.
pub struct ParsedNetlist {
    pub modules: Vec<NetlistModule>,
    pub nets: Vec<Net>,
    pub interner: StringInterner<StringBackend<SymbolU32>>,
}

/// Parse a gate-level netlist (optionally gzipped) into modules, nets, and the
/// interner, with rich error messages including source-line context.
pub fn parse_netlist_from_path(path: &Path) -> Result<ParsedNetlist> {
    let file = File::open(path)
        .map_err(|e| anyhow!(format!("opening netlist '{}': {}", path.display(), e)))?;
    let is_gz = path.extension().map(|e| e == "gz").unwrap_or(false);
    let reader: Box<dyn Read> = if is_gz {
        Box::new(MultiGzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(file)
    };

    let lookup_path = path.to_path_buf();
    let lookup_is_gz = is_gz;
    let lookup = move |lineno: u32| -> Option<String> {
        let f = File::open(&lookup_path).ok()?;
        if lookup_is_gz {
            let br = BufReader::new(f);
            let gz = MultiGzDecoder::new(br);
            let rdr = BufReader::new(gz);
            rdr.lines().nth((lineno - 1) as usize).and_then(Result::ok)
        } else {
            let rdr = BufReader::new(f);
            rdr.lines().nth((lineno - 1) as usize).and_then(Result::ok)
        }
    };

    let scanner = TokenScanner::with_line_lookup(reader, Box::new(lookup));
    let mut parser: NetlistParser<Box<dyn Read>> = NetlistParser::new(scanner);
    let modules = parser.parse_file().map_err(|e| {
        anyhow!(format!(
            "{} @ {}\n{}\n{}^",
            e.message,
            e.span.to_human_string(),
            parser
                .get_line(e.span.start.lineno)
                .unwrap_or_else(|| "<line unavailable>".to_string()),
            " ".repeat((e.span.start.colno as usize).saturating_sub(1))
        ))
    })?;

    Ok(ParsedNetlist {
        modules,
        nets: parser.nets,
        interner: parser.interner,
    })
}

/// Load a Liberty proto (binary or textproto) into a `Library`.
///
/// This helper is shared by higher-level routines that need to work from
/// Liberty files but want to keep I/O concerns out of their core logic.
pub fn load_liberty_from_path(path: &Path) -> Result<Library> {
    let mut buf: Vec<u8> = Vec::new();
    File::open(path)
        .map_err(|e| anyhow!(format!("opening liberty proto '{}': {}", path.display(), e)))?
        .read_to_end(&mut buf)
        .map_err(|e| anyhow!(format!("reading liberty proto '{}': {}", path.display(), e)))?;

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
