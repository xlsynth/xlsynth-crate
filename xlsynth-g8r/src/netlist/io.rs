// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for reading and parsing gate-level netlists from disk.
//!
//! This module centralizes the logic for:
//! - Handling plain `.gv` and `.gv.gz` inputs.
//! - Wiring up `TokenScanner::with_line_lookup` so that parse errors can show
//!   source-line context.
//! - Producing the parsed modules together with the global `nets` array and
//!   `StringInterner`.

use crate::liberty::load::{
    Library, LibraryWithTimingData, load_library_from_path, load_library_with_timing_data_from_path,
};
use crate::netlist::parse::{Net, NetlistModule, Parser as NetlistParser, PortId, TokenScanner};
use anyhow::{Result, anyhow};
use flate2::read::MultiGzDecoder;
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

/// Resolves one interned netlist symbol with an actionable error.
pub fn resolve_symbol(
    interner: &StringInterner<StringBackend<SymbolU32>>,
    sym: PortId,
    what: &str,
) -> Result<String> {
    interner
        .resolve(sym)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("could not resolve {} symbol", what))
}

/// Selects one module by name, or the only module when no name is provided.
pub fn select_module<'a>(
    parsed: &'a ParsedNetlist,
    module_name: Option<&str>,
) -> Result<&'a NetlistModule> {
    if let Some(name) = module_name {
        for module in &parsed.modules {
            let resolved = resolve_symbol(&parsed.interner, module.name, "module name")?;
            if resolved == name {
                return Ok(module);
            }
        }
        let mut available: Vec<String> = parsed
            .modules
            .iter()
            .map(|m| {
                parsed
                    .interner
                    .resolve(m.name)
                    .unwrap_or("<unknown>")
                    .to_string()
            })
            .collect();
        available.sort();
        return Err(anyhow!(
            "module '{}' not found in netlist; available modules: [{}]",
            name,
            available.join(", ")
        ));
    }

    if parsed.modules.len() == 1 {
        return Ok(&parsed.modules[0]);
    }

    let mut names = Vec::with_capacity(parsed.modules.len());
    for module in &parsed.modules {
        names.push(resolve_symbol(
            &parsed.interner,
            module.name,
            "module name",
        )?);
    }
    names.sort();
    Err(anyhow!(
        "netlist contains {} modules; specify --module_name; available modules: [{}]",
        parsed.modules.len(),
        names.join(", ")
    ))
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
    load_library_from_path(path)
}

/// Load a Liberty proto (binary or textproto) with full timing payloads.
pub fn load_liberty_with_timing_data_from_path(path: &Path) -> Result<LibraryWithTimingData> {
    load_library_with_timing_data_from_path(path)
}
