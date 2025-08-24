// SPDX-License-Identifier: Apache-2.0

//! Compute summary statistics for gate-level netlists.

use crate::netlist::parse::{
    Net, NetIndex, NetRef, NetlistInstance, NetlistModule, NetlistPort, Parser as NetlistParser,
    PortId, TokenScanner,
};
use anyhow::{Result, anyhow};
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::mem::size_of;
use std::path::Path;
use std::time::{Duration, Instant};

/// Summary statistics for a parsed netlist.
#[derive(Debug)]
pub struct NetlistStats {
    pub num_instances: usize,
    pub num_nets: usize,
    pub memory_bytes: usize,
    pub cell_counts: Vec<(String, usize)>,
    pub parse_duration: Duration,
}

fn open_reader(path: &Path) -> Result<Box<dyn Read>> {
    let file = File::open(path)?;
    let is_gz = path.extension().map(|e| e == "gz").unwrap_or(false);
    if is_gz {
        Ok(Box::new(GzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

fn line_lookup(path: &Path) -> Box<dyn Fn(u32) -> Option<String>> {
    let path = path.to_path_buf();
    let is_gz = path.extension().map(|e| e == "gz").unwrap_or(false);
    Box::new(move |lineno| {
        let file = File::open(&path).ok()?;
        if is_gz {
            let gz = GzDecoder::new(file);
            let reader = BufReader::new(gz);
            reader
                .lines()
                .nth((lineno - 1) as usize)
                .and_then(|r| r.ok())
        } else {
            let reader = BufReader::new(file);
            reader
                .lines()
                .nth((lineno - 1) as usize)
                .and_then(|r| r.ok())
        }
    })
}

/// Reads and parses the netlist at `path`, returning summary statistics.
pub fn read_netlist_stats(path: &Path) -> Result<NetlistStats> {
    let reader = open_reader(path)?;
    let lookup = line_lookup(path);
    let scanner = TokenScanner::with_line_lookup(reader, lookup);
    let mut parser: NetlistParser<Box<dyn Read>> = NetlistParser::new(scanner);
    let start = Instant::now();
    let modules = parser.parse_file().map_err(|e| anyhow!(e.message))?;
    let parse_duration = start.elapsed();

    let num_instances: usize = modules.iter().map(|m| m.instances.len()).sum();
    let num_nets = parser.nets.len();

    let mut counts: HashMap<PortId, usize> = HashMap::new();
    for m in &modules {
        for inst in &m.instances {
            *counts.entry(inst.type_name).or_insert(0) += 1;
        }
    }

    let mut cell_counts: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(sym, count)| {
            let name = parser
                .interner
                .resolve(sym)
                .map(|s| s.to_string())
                .unwrap_or_default();
            (name, count)
        })
        .collect();
    cell_counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let memory_bytes = estimate_memory_bytes(&parser, &modules);

    Ok(NetlistStats {
        num_instances,
        num_nets,
        memory_bytes,
        cell_counts,
        parse_duration,
    })
}

fn estimate_memory_bytes<R: Read + 'static>(
    parser: &NetlistParser<R>,
    modules: &[NetlistModule],
) -> usize {
    let mut memory_bytes = parser.nets.capacity() * size_of::<Net>();
    for m in modules {
        memory_bytes += size_of::<NetlistModule>();
        memory_bytes += m.ports.capacity() * size_of::<NetlistPort>();
        memory_bytes += m.wires.capacity() * size_of::<NetIndex>();
        memory_bytes += m.instances.capacity() * size_of::<NetlistInstance>();
        for inst in &m.instances {
            memory_bytes += inst.connections.capacity() * size_of::<(PortId, NetRef)>();
        }
    }
    memory_bytes += parser.interner.len() * size_of::<String>();
    for (_sym, s) in parser.interner.clone().into_iter() {
        memory_bytes += s.len();
    }
    memory_bytes
}
