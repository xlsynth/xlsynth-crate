// SPDX-License-Identifier: Apache-2.0

//! Compute summary statistics for gate-level netlists.

use crate::netlist::io::{ParsedNetlist, read_gv_from_path_timed};
use crate::netlist::parse::{
    Net, NetIndex, NetRef, NetlistInstance, NetlistModule, NetlistPort, PortId,
};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::mem::size_of;
use std::path::Path;
use std::time::Duration;

/// Summary statistics for a parsed netlist.
#[derive(Debug)]
pub struct NetlistStats {
    pub num_instances: usize,
    pub num_nets: usize,
    pub memory_bytes: usize,
    pub cell_counts: Vec<(String, usize)>,
    pub parse_duration: Duration,
}

/// Reads and parses the netlist at `path`, returning summary statistics.
pub fn read_gv_stats(path: &Path) -> Result<NetlistStats> {
    log::info!("read_gv_stats: begin path='{}'", path.display());
    let result = read_gv_from_path_timed(path)?;
    let parsed = result.parsed;
    let modules = &parsed.modules;
    let parse_duration = result.parse_duration;
    log::info!(
        "read_gv_stats: parse done; modules={}, nets={}, duration_ms={}",
        modules.len(),
        parsed.nets.len(),
        parse_duration.as_millis()
    );

    // Strong invariant: a valid gate-level netlist must contain at least one
    // module. Returning zeroed statistics is misleading; instead, report an
    // error so callers can surface actionable feedback (e.g. invalid gzip,
    // unsupported preprocessor directives, or non-gate-level input).
    if modules.is_empty() {
        return Err(anyhow!(format!(
            "no modules parsed from '{}'; ensure the file is a readable gate-level Verilog netlist{}",
            path.display(),
            if path.extension().map(|e| e == "gz").unwrap_or(false) {
                " and a valid gzip stream"
            } else {
                ""
            }
        )));
    }

    let num_instances: usize = modules.iter().map(|m| m.instances.len()).sum();
    let num_nets = parsed.nets.len();

    let mut counts: HashMap<PortId, usize> = HashMap::new();
    for m in modules {
        for inst in &m.instances {
            *counts.entry(inst.type_name).or_insert(0) += 1;
        }
    }

    let mut cell_counts: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(sym, count)| {
            let name = parsed
                .interner
                .resolve(sym)
                .map(|s| s.to_string())
                .unwrap_or_default();
            (name, count)
        })
        .collect();
    cell_counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let memory_bytes = estimate_memory_bytes(&parsed);

    Ok(NetlistStats {
        num_instances,
        num_nets,
        memory_bytes,
        cell_counts,
        parse_duration,
    })
}

fn estimate_memory_bytes(parsed: &ParsedNetlist) -> usize {
    let modules = &parsed.modules;
    let mut memory_bytes = parsed.nets.capacity() * size_of::<Net>();
    for m in modules {
        memory_bytes += size_of::<NetlistModule>();
        memory_bytes += m.ports.capacity() * size_of::<NetlistPort>();
        memory_bytes += m.wires.capacity() * size_of::<NetIndex>();
        memory_bytes += m.instances.capacity() * size_of::<NetlistInstance>();
        for inst in &m.instances {
            memory_bytes += inst.connections.capacity() * size_of::<(PortId, NetRef)>();
        }
    }
    memory_bytes += parsed.interner.len() * size_of::<String>();
    for (_sym, s) in parsed.interner.clone().into_iter() {
        memory_bytes += s.len();
    }
    memory_bytes
}
