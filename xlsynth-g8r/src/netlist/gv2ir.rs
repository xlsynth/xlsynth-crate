// SPDX-License-Identifier: Apache-2.0

use crate::aig_serdes::gate2ir::gate_fn_to_xlsynth_ir;
use crate::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
use crate::netlist::io::load_liberty_from_path;
use crate::netlist::parse::{Parser as NetlistParser, TokenScanner};
use anyhow::{Context, Result, anyhow};
use flate2::bufread::MultiGzDecoder as BufMultiGzDecoder;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

fn open_reader(path: &Path) -> Result<(Box<dyn Read>, bool)> {
    let is_gz = path.extension().map(|e| e == "gz").unwrap_or(false);
    if is_gz {
        let f =
            File::open(path).with_context(|| format!("opening netlist '{}'", path.display()))?;
        let br = BufReader::new(f);
        Ok((Box::new(BufMultiGzDecoder::new(br)), true))
    } else {
        let f =
            File::open(path).with_context(|| format!("opening netlist '{}'", path.display()))?;
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

pub fn convert_gv2ir_paths(
    netlist_path: &Path,
    liberty_proto_path: &Path,
    dff_cells: &HashSet<String>,
    dff_cell_formula: Option<&str>,
    dff_cell_invert_formula: Option<&str>,
) -> Result<String> {
    // Netlist parse
    let (reader, is_gz) = open_reader(netlist_path)?;
    let lookup = line_lookup(netlist_path, is_gz);
    let scanner = TokenScanner::with_line_lookup(reader, lookup);
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
    if modules.len() != 1 {
        return Err(anyhow!(format!(
            "expected exactly one module, got {}",
            modules.len()
        )));
    }
    let module = &modules[0];

    // Liberty
    let liberty_lib = load_liberty_from_path(liberty_proto_path)?;

    // Build DFF cell sets: identity and inverted. Start from provided names
    // (identity), then add by matching formula strings (if provided).
    let mut dff_cells_identity: HashSet<String> = dff_cells.clone();
    if let Some(target_formula) = dff_cell_formula {
        for cell in &liberty_lib.cells {
            if cell
                .pins
                .iter()
                .any(|p| p.direction == 1 && p.function == target_formula)
            {
                dff_cells_identity.insert(cell.name.clone());
            }
        }
    }
    let mut dff_cells_inverted: HashSet<String> = HashSet::new();
    if let Some(invert_formula) = dff_cell_invert_formula {
        for cell in &liberty_lib.cells {
            if cell
                .pins
                .iter()
                .any(|p| p.direction == 1 && p.function == invert_formula)
            {
                dff_cells_inverted.insert(cell.name.clone());
            }
        }
    }

    // Project GateFn
    let gate_fn = project_gatefn_from_netlist_and_liberty(
        module,
        &parser.nets,
        &parser.interner,
        &liberty_lib,
        &dff_cells_identity,
        &dff_cells_inverted,
    )
    .map_err(|e| anyhow!(e))?;

    // Convert to IR text
    let flat_type = gate_fn.get_flat_type();
    let ir_pkg = gate_fn_to_xlsynth_ir(&gate_fn, "gate", &flat_type)?;
    Ok(ir_pkg.to_string())
}
