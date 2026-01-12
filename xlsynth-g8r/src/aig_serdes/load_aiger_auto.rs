// SPDX-License-Identifier: Apache-2.0

//! Auto-detect and load either ASCII or binary AIGER files.

use crate::aig_serdes::load_aiger::{LoadAigerResult, load_aiger};
use crate::aig_serdes::load_aiger_binary::load_aiger_binary;
use crate::gate_builder::GateBuilderOptions;
use std::fs;
use std::path::Path;

/// Parses an ASCII (`.aag`) or binary (`.aig`) AIGER file from disk.
pub fn load_aiger_auto_from_path(
    path: &Path,
    opts: GateBuilderOptions,
) -> Result<LoadAigerResult, String> {
    let contents =
        fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    load_aiger_auto(&contents, opts)
}

/// Parses either ASCII or binary AIGER bytes, dispatching based on the header.
pub fn load_aiger_auto(src: &[u8], opts: GateBuilderOptions) -> Result<LoadAigerResult, String> {
    let header_line_end = src
        .iter()
        .position(|b| *b == b'\n')
        .ok_or_else(|| "unterminated AIGER header".to_string())?;
    let header_line = std::str::from_utf8(&src[..header_line_end])
        .map_err(|e| format!("invalid UTF-8 in AIGER header: {}", e))?;
    let header_tokens: Vec<&str> = header_line.split_whitespace().collect();
    let header_kind = header_tokens
        .get(0)
        .ok_or_else(|| "missing AIGER header token".to_string())?;
    match *header_kind {
        "aag" => {
            let text = std::str::from_utf8(src)
                .map_err(|e| format!("invalid UTF-8 in ASCII AIGER input: {}", e))?;
            load_aiger(text, opts)
        }
        "aig" => load_aiger_binary(src, opts),
        other => Err(format!("unknown AIGER header '{}'", other)),
    }
}
