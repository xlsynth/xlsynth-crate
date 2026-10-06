// SPDX-License-Identifier: Apache-2.0

//! Explicit GV and UGV readers with shared source loading and diagnostics.
//!
//! This module centralizes the logic for:
//! - Handling plain and gzip-compressed inputs.
//! - Wiring up `TokenScanner::with_line_lookup` so that parse errors can show
//!   source-line context.
//! - Producing the parsed modules together with the global `nets` array and
//!   `StringInterner`.

use crate::liberty::Library;
use crate::liberty::load::{load_library_from_path, load_library_with_timing_data_from_path};
use crate::netlist::form::{validate_gv_form, validate_ugv_form};
use crate::netlist::parse::{
    Net, NetlistModule, Parser as NetlistParser, PortId, ScanError, TokenScanner,
};
use anyhow::{Result, anyhow};
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::Path;
use std::time::{Duration, Instant};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Parsed netlist plus the global nets and interner.
pub struct ParsedNetlist {
    pub modules: Vec<NetlistModule>,
    pub nets: Vec<Net>,
    pub interner: StringInterner<StringBackend<SymbolU32>>,
}

/// Parsed input with syntax-parsing time, excluding format validation.
pub(crate) struct TimedParsedNetlist {
    pub parsed: ParsedNetlist,
    pub parse_duration: Duration,
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

#[derive(Clone, Copy)]
enum NetlistFormat {
    Gv,
    Ugv,
}

impl NetlistFormat {
    fn name(self) -> &'static str {
        match self {
            Self::Gv => "GV",
            Self::Ugv => "UGV",
        }
    }

    fn guidance(self) -> &'static str {
        match self {
            Self::Gv => {
                "GV supports cell instances and wiring assignments; procedural RTL and logic assignments are unsupported."
            }
            Self::Ugv => {
                "UGV supports combinational Boolean assignments; procedural registers and leaf cells are unsupported."
            }
        }
    }
}

/// Reads GV containing cell instances and wiring-only assignments.
pub fn read_gv_from_path(path: &Path) -> Result<ParsedNetlist> {
    read_netlist_from_path(path, NetlistFormat::Gv).map(|result| result.parsed)
}

/// Reads combinational UGV containing Boolean assignments and no leaf cells.
pub fn read_ugv_from_path(path: &Path) -> Result<ParsedNetlist> {
    read_netlist_from_path(path, NetlistFormat::Ugv).map(|result| result.parsed)
}

/// Reads GV while retaining the parser-only duration used by `gv-read-stats`.
pub(crate) fn read_gv_from_path_timed(path: &Path) -> Result<TimedParsedNetlist> {
    read_netlist_from_path(path, NetlistFormat::Gv)
}

/// Reads GV from an in-memory string.
pub fn read_gv_from_str(source: &str) -> Result<ParsedNetlist> {
    read_netlist_from_str(source, NetlistFormat::Gv).map(|result| result.parsed)
}

/// Reads combinational UGV from an in-memory string.
pub fn read_ugv_from_str(source: &str) -> Result<ParsedNetlist> {
    read_netlist_from_str(source, NetlistFormat::Ugv).map(|result| result.parsed)
}

fn read_netlist_from_str(source: &str, format: NetlistFormat) -> Result<TimedParsedNetlist> {
    let lines = source.lines().map(str::to_string).collect::<Vec<_>>();
    let scanner = TokenScanner::with_line_lookup(
        Cursor::new(source.as_bytes().to_vec()),
        Box::new(move |lineno| lines.get((lineno - 1) as usize).cloned()),
    );
    read_netlist(scanner, "<string>", format)
}

fn read_netlist_from_path(path: &Path, format: NetlistFormat) -> Result<TimedParsedNetlist> {
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
    read_netlist(scanner, &path.display().to_string(), format)
}

fn read_netlist<R: Read + 'static>(
    scanner: TokenScanner<R>,
    source_name: &str,
    format: NetlistFormat,
) -> Result<TimedParsedNetlist> {
    let mut parser = NetlistParser::new(scanner);
    let render_error = |error: ScanError, parser: &NetlistParser<R>| {
        anyhow!(
            "{} reader: {} @ {}:{}\n{}\n{}^\n{}",
            format.name(),
            error.message,
            source_name,
            error.span.to_human_string(),
            parser
                .get_line(error.span.start.lineno)
                .unwrap_or_else(|| "<line unavailable>".to_string()),
            " ".repeat((error.span.start.colno as usize).saturating_sub(1)),
            format.guidance()
        )
    };
    let parse_start = Instant::now();
    let modules = parser
        .parse_file()
        .map_err(|error| render_error(error, &parser))?;
    let parse_duration = parse_start.elapsed();
    match format {
        NetlistFormat::Gv => validate_gv_form(&modules, &parser.nets, &parser.interner),
        NetlistFormat::Ugv => validate_ugv_form(&modules, &parser.interner),
    }
    .map_err(|error| render_error(error, &parser))?;

    Ok(TimedParsedNetlist {
        parsed: ParsedNetlist {
            modules,
            nets: parser.nets,
            interner: parser.interner,
        },
        parse_duration,
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
pub fn load_liberty_with_timing_data_from_path(path: &Path) -> Result<Library> {
    load_library_with_timing_data_from_path(path)
}
