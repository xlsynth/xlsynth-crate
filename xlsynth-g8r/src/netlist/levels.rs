// SPDX-License-Identifier: Apache-2.0

//! Compute longest combinational path depths between netlist boundaries.
//!
//! We define boundaries as:
//! - Start boundaries: module inputs and DFF outputs.
//! - Sink boundaries: DFF inputs and module outputs.
//!
//! Depth is measured as the number of **combinational cell instances** on the
//! longest boundary-to-boundary path. Direct boundary→boundary connectivity has
//! depth 0.

use crate::liberty::IndexedLibrary;
use crate::liberty_proto::PinDirection;
use crate::netlist::connectivity::NetlistConnectivity;
use crate::netlist::parse::{InstIndex, Net, NetIndex, NetlistModule, PortDirection, PortId};
use anyhow::{Result, anyhow};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OriginKind {
    /// Origin is not a boundary start (module input or DFF output).
    ///
    /// In practice this shows up when a net is only driven by constants or has
    /// no drivers that participate in our boundary seeding (e.g. undriven
    /// wires, missing connectivity due to malformed netlists, or combinational
    /// fanin that is ultimately constant-only). We treat this as input-like
    /// for category bucketing (i.e. it contributes to input-to-reg/output).
    Other = 0,
    Input = 1,
    Reg = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathInfo {
    pub depth: u32,
    pub origin: OriginKind,
}

impl PathInfo {
    /// Returns true if `self` should replace `other` as the "best" (most
    /// relevant) provenance for a net.
    ///
    /// Real-world intent:
    /// - Prefer the **longest combinational depth** because we are computing
    ///   worst-case path depth histograms.
    /// - When depths tie, prefer the **more register-like origin** to make the
    ///   bucketing deterministic and conservative for reg-to-* classification:
    ///   \(Reg > Input > Other\).
    fn better_than(&self, other: &PathInfo) -> bool {
        self.depth > other.depth || (self.depth == other.depth && self.origin > other.origin)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LevelsCategory {
    InputToReg,
    RegToReg,
    RegToOutput,
    InputToOutput,
}

impl LevelsCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            LevelsCategory::InputToReg => "input-to-reg",
            LevelsCategory::RegToReg => "reg-to-reg",
            LevelsCategory::RegToOutput => "reg-to-output",
            LevelsCategory::InputToOutput => "input-to-output",
        }
    }
}
#[derive(Debug, Clone)]
pub struct LevelsReport {
    pub histograms: BTreeMap<LevelsCategory, BTreeMap<u32, u64>>,
    pub num_instances: usize,
    pub num_dff_instances: usize,
    pub num_output_ports: usize,
    /// For each category, a deterministic example showing one sink sample that
    /// attains the maximum depth observed in that category.
    pub max_examples: BTreeMap<LevelsCategory, LevelsMaxExample>,
}

#[derive(Debug, Clone)]
pub struct LevelsMaxExample {
    pub depth: u32,
    /// Human-readable sink identifier (e.g. `port:y` or `inst:udff0`).
    pub sink: String,
    /// Path of combinational instances along a max-depth path, ordered from
    /// boundary start to sink.
    pub instance_path: Vec<String>,
}

fn resolve_to_string(
    interner: &StringInterner<StringBackend<SymbolU32>>,
    sym: SymbolU32,
) -> String {
    interner
        .resolve(sym)
        .expect("symbol should always resolve in interner")
        .to_string()
}

fn validate_liberty_coverage(
    module: &NetlistModule,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    indexed_lib: &IndexedLibrary,
) -> Result<()> {
    for inst in &module.instances {
        let cell_name = resolve_to_string(interner, inst.type_name);
        let cell = indexed_lib
            .get_cell(cell_name.as_str())
            .ok_or_else(|| anyhow!(format!("cell type '{}' is missing from Liberty", cell_name)))?;
        let mut pins: HashMap<&str, PinDirection> = HashMap::new();
        for pin in &cell.pins {
            let dir_val = pin.direction;
            if dir_val == PinDirection::Input as i32 {
                pins.insert(pin.name.as_str(), PinDirection::Input);
            } else if dir_val == PinDirection::Output as i32 {
                pins.insert(pin.name.as_str(), PinDirection::Output);
            } else {
                pins.insert(pin.name.as_str(), PinDirection::Invalid);
            }
        }
        for (port_sym, _netref) in &inst.connections {
            let pin_name = resolve_to_string(interner, *port_sym);
            let dir = pins
                .get(pin_name.as_str())
                .copied()
                .unwrap_or(PinDirection::Invalid);
            if dir == PinDirection::Invalid {
                return Err(anyhow!(format!(
                    "pin '{}' on cell type '{}' is missing or has invalid direction in Liberty",
                    pin_name, cell_name
                )));
            }
        }
    }
    Ok(())
}

fn build_net_index_by_sym(nets: &[Net]) -> HashMap<SymbolU32, NetIndex> {
    let mut out: HashMap<SymbolU32, NetIndex> = HashMap::new();
    for (i, net) in nets.iter().enumerate() {
        out.insert(net.name, NetIndex(i));
    }
    out
}

fn add_hist_sample(
    histograms: &mut BTreeMap<LevelsCategory, BTreeMap<u32, u64>>,
    cat: LevelsCategory,
    depth: u32,
) {
    let entry = histograms.entry(cat).or_insert_with(BTreeMap::new);
    *entry.entry(depth).or_insert(0) += 1;
}

#[derive(Debug, Clone)]
enum TracePrev {
    InstOut {
        inst: InstIndex,
        input_net: Option<NetIndex>,
    },
}

#[derive(Debug, Clone)]
struct NetTrace {
    info: PathInfo,
    prev: Option<TracePrev>,
}

fn default_trace() -> NetTrace {
    NetTrace {
        info: PathInfo {
            depth: 0,
            origin: OriginKind::Other,
        },
        prev: None,
    }
}

fn prev_tie_key(prev: &Option<TracePrev>) -> (u32, u32) {
    match prev {
        None => (u32::MAX, u32::MAX),
        Some(TracePrev::InstOut { inst, input_net }) => (
            inst.0 as u32,
            input_net.map(|n| n.0 as u32).unwrap_or(u32::MAX),
        ),
    }
}

fn trace_better_than(a: &NetTrace, b: &NetTrace) -> bool {
    if a.info.better_than(&b.info) {
        return true;
    }
    if b.info.better_than(&a.info) {
        return false;
    }
    // Same PathInfo: deterministically prefer the smaller predecessor key.
    prev_tie_key(&a.prev) < prev_tie_key(&b.prev)
}

fn format_instance_id(
    module: &NetlistModule,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    inst: InstIndex,
) -> String {
    let i = &module.instances[inst.0];
    let inst_name = resolve_to_string(interner, i.instance_name);
    let cell_name = resolve_to_string(interner, i.type_name);
    format!("{}({})", inst_name, cell_name)
}

fn build_instance_path_for_net(
    module: &NetlistModule,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    net_traces: &[Option<NetTrace>],
    sink_net: NetIndex,
) -> Vec<String> {
    let mut out_rev: Vec<String> = Vec::new();
    let mut cur = Some(sink_net);
    // Bound the traversal to avoid accidental infinite loops if invariants are
    // violated.
    for _ in 0..(module.instances.len() + 1) {
        let Some(net_idx) = cur else {
            break;
        };
        let Some(trace) = &net_traces[net_idx.0] else {
            break;
        };
        match &trace.prev {
            None => break,
            Some(TracePrev::InstOut { inst, input_net }) => {
                out_rev.push(format_instance_id(module, interner, *inst));
                cur = *input_net;
            }
        }
    }
    out_rev.reverse();
    out_rev
}

fn maybe_update_max_example(
    max_examples: &mut BTreeMap<LevelsCategory, LevelsMaxExample>,
    cat: LevelsCategory,
    depth: u32,
    sink: String,
    instance_path: Vec<String>,
) {
    match max_examples.get(&cat) {
        None => {
            max_examples.insert(
                cat,
                LevelsMaxExample {
                    depth,
                    sink,
                    instance_path,
                },
            );
        }
        Some(existing) => {
            if depth > existing.depth
                || (depth == existing.depth
                    && (sink < existing.sink
                        || (sink == existing.sink && instance_path < existing.instance_path)))
            {
                max_examples.insert(
                    cat,
                    LevelsMaxExample {
                        depth,
                        sink,
                        instance_path,
                    },
                );
            }
        }
    }
}

/// Computes per-category depth histograms for the given module.
pub fn compute_levels(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    indexed_lib: &IndexedLibrary,
    dff_cell_types: &HashSet<String>,
    module_port_dirs: Option<&HashMap<PortId, HashMap<PortId, PinDirection>>>,
) -> Result<LevelsReport> {
    validate_liberty_coverage(module, interner, indexed_lib)?;
    let conn = NetlistConnectivity::new(module, nets, interner, indexed_lib, module_port_dirs);
    let net_index_by_sym = build_net_index_by_sym(nets);

    let mut is_dff_inst: Vec<bool> = Vec::with_capacity(module.instances.len());
    is_dff_inst.resize_with(module.instances.len(), || false);
    for (i, inst) in module.instances.iter().enumerate() {
        let cell = resolve_to_string(interner, inst.type_name);
        is_dff_inst[i] = dff_cell_types.contains(&cell);
    }

    let num_dff_instances = is_dff_inst.iter().filter(|v| **v).count();

    // Seed per-net path info with boundaries, plus predecessor information for
    // reconstructing example max-depth paths.
    let mut net_traces: Vec<Option<NetTrace>> = vec![None; nets.len()];

    // Module inputs are start boundaries.
    for port in &module.ports {
        if port.direction != PortDirection::Input {
            continue;
        }
        let Some(net_idx) = net_index_by_sym.get(&port.name).copied() else {
            continue;
        };
        let candidate = PathInfo {
            depth: 0,
            origin: OriginKind::Input,
        };
        let candidate_trace = NetTrace {
            info: candidate,
            prev: None,
        };
        match &mut net_traces[net_idx.0] {
            Some(existing) => {
                if trace_better_than(&candidate_trace, existing) {
                    *existing = candidate_trace;
                }
            }
            None => net_traces[net_idx.0] = Some(candidate_trace),
        }
    }

    // DFF outputs are start boundaries.
    for (inst_idx_raw, _inst) in module.instances.iter().enumerate() {
        if !is_dff_inst[inst_idx_raw] {
            continue;
        }
        let inst_idx = InstIndex(inst_idx_raw);
        for port in conn.instance_ports(inst_idx, interner) {
            if port.dir != PinDirection::Output {
                continue;
            }
            let candidate = PathInfo {
                depth: 0,
                origin: OriginKind::Reg,
            };
            let candidate_trace = NetTrace {
                info: candidate,
                prev: None,
            };
            for net_idx in &port.nets {
                match &mut net_traces[net_idx.0] {
                    Some(existing) => {
                        if trace_better_than(&candidate_trace, existing) {
                            *existing = candidate_trace.clone();
                        }
                    }
                    None => net_traces[net_idx.0] = Some(candidate_trace.clone()),
                }
            }
        }
    }

    // Build combinational dependency graph: comb_driver -> comb_load.
    let mut is_comb_inst: Vec<bool> = Vec::with_capacity(module.instances.len());
    is_comb_inst.resize_with(module.instances.len(), || false);
    for i in 0..module.instances.len() {
        is_comb_inst[i] = !is_dff_inst[i];
    }

    let mut indegree: Vec<u32> = vec![0; module.instances.len()];
    let mut adj: Vec<Vec<InstIndex>> = vec![Vec::new(); module.instances.len()];

    for (load_idx_raw, _inst) in module.instances.iter().enumerate() {
        if !is_comb_inst[load_idx_raw] {
            continue;
        }
        let load_idx = InstIndex(load_idx_raw);

        for port in conn.instance_ports(load_idx, interner) {
            if port.dir != PinDirection::Input {
                continue;
            }
            for net_idx in &port.nets {
                let drivers = conn.drivers_for_net(*net_idx);
                let mut comb_driver: Option<InstIndex> = None;
                for (drv_inst, _drv_port) in drivers {
                    if is_comb_inst[drv_inst.0] {
                        if comb_driver.is_some() {
                            return Err(anyhow!(format!(
                                "net '{}' has multiple combinational drivers; expected at most one",
                                resolve_to_string(interner, nets[net_idx.0].name)
                            )));
                        }
                        comb_driver = Some(*drv_inst);
                    }
                }
                if let Some(drv) = comb_driver {
                    adj[drv.0].push(load_idx);
                    indegree[load_idx.0] += 1;
                }
            }
        }
    }

    for neighbors in &mut adj {
        neighbors.sort_by_key(|idx| idx.0);
        neighbors.dedup();
    }

    let mut queue: VecDeque<InstIndex> = VecDeque::new();
    for i in 0..module.instances.len() {
        if is_comb_inst[i] && indegree[i] == 0 {
            queue.push_back(InstIndex(i));
        }
    }

    let mut processed_comb = 0usize;
    while let Some(inst_idx) = queue.pop_front() {
        processed_comb += 1;

        // Best input path among this combinational instance's inputs.
        let mut best = default_trace();
        let mut best_input_net: Option<NetIndex> = None;
        for port in conn.instance_ports(inst_idx, interner) {
            if port.dir != PinDirection::Input {
                continue;
            }
            let mut nets_sorted = port.nets.clone();
            nets_sorted.sort_by_key(|n| n.0);
            for net_idx in &nets_sorted {
                let candidate = net_traces[net_idx.0].clone().unwrap_or_else(default_trace);
                if trace_better_than(&candidate, &best)
                    || (candidate.info.depth == best.info.depth
                        && candidate.info.origin == best.info.origin
                        && (best_input_net.is_none()
                            || net_idx.0 < best_input_net.expect("checked").0))
                {
                    best = candidate;
                    best_input_net = Some(*net_idx);
                }
            }
        }

        let out_trace = NetTrace {
            info: PathInfo {
                depth: best.info.depth + 1,
                origin: best.info.origin,
            },
            prev: Some(TracePrev::InstOut {
                inst: inst_idx,
                input_net: best_input_net,
            }),
        };

        for port in conn.instance_ports(inst_idx, interner) {
            if port.dir != PinDirection::Output {
                continue;
            }
            for net_idx in &port.nets {
                match &mut net_traces[net_idx.0] {
                    Some(existing) => {
                        if trace_better_than(&out_trace, existing) {
                            *existing = out_trace.clone();
                        }
                    }
                    None => net_traces[net_idx.0] = Some(out_trace.clone()),
                }
            }
        }

        for nbr in &adj[inst_idx.0] {
            indegree[nbr.0] = indegree[nbr.0].saturating_sub(1);
            if indegree[nbr.0] == 0 {
                queue.push_back(*nbr);
            }
        }
    }

    let total_comb = is_comb_inst.iter().filter(|v| **v).count();
    if processed_comb != total_comb {
        let mut remaining: Vec<usize> = Vec::new();
        for i in 0..module.instances.len() {
            if is_comb_inst[i] && indegree[i] > 0 {
                remaining.push(i);
            }
        }
        remaining.sort();
        let mut sample: Vec<String> = Vec::new();
        for idx in remaining.iter().take(5) {
            let inst = &module.instances[*idx];
            let inst_name = resolve_to_string(interner, inst.instance_name);
            let cell_name = resolve_to_string(interner, inst.type_name);
            sample.push(format!("{}:{}", inst_name, cell_name));
        }
        return Err(anyhow!(format!(
            "cycle detected in combinational connectivity; sample remaining instances: {}",
            sample.join(", ")
        )));
    }

    // Build histograms for sinks.
    let mut histograms: BTreeMap<LevelsCategory, BTreeMap<u32, u64>> = BTreeMap::new();
    let mut max_examples: BTreeMap<LevelsCategory, LevelsMaxExample> = BTreeMap::new();

    // DFF sink samples: one per DFF instance (max over input pins).
    for (inst_idx_raw, _inst) in module.instances.iter().enumerate() {
        if !is_dff_inst[inst_idx_raw] {
            continue;
        }
        let inst_idx = InstIndex(inst_idx_raw);
        let mut best = default_trace();
        let mut best_sink_net: Option<NetIndex> = None;
        for port in conn.instance_ports(inst_idx, interner) {
            if port.dir != PinDirection::Input {
                continue;
            }
            let mut nets_sorted = port.nets.clone();
            nets_sorted.sort_by_key(|n| n.0);
            for net_idx in &nets_sorted {
                let candidate = net_traces[net_idx.0].clone().unwrap_or_else(default_trace);
                if trace_better_than(&candidate, &best)
                    || (candidate.info.depth == best.info.depth
                        && candidate.info.origin == best.info.origin
                        && (best_sink_net.is_none()
                            || net_idx.0 < best_sink_net.expect("checked").0))
                {
                    best = candidate;
                    best_sink_net = Some(*net_idx);
                }
            }
        }
        let cat = if matches!(best.info.origin, OriginKind::Reg) {
            LevelsCategory::RegToReg
        } else {
            LevelsCategory::InputToReg
        };
        add_hist_sample(&mut histograms, cat, best.info.depth);
        let inst_name = resolve_to_string(interner, module.instances[inst_idx_raw].instance_name);
        let sink = format!("inst:{}", inst_name);
        let instance_path = best_sink_net
            .map(|n| build_instance_path_for_net(module, interner, &net_traces, n))
            .unwrap_or_default();
        maybe_update_max_example(&mut max_examples, cat, best.info.depth, sink, instance_path);
    }

    // Output port samples: one per module output port.
    let mut num_output_ports = 0usize;
    for port in &module.ports {
        if port.direction != PortDirection::Output {
            continue;
        }
        num_output_ports += 1;
        let Some(net_idx) = net_index_by_sym.get(&port.name).copied() else {
            continue;
        };
        let best = net_traces[net_idx.0].clone().unwrap_or_else(default_trace);
        let cat = if matches!(best.info.origin, OriginKind::Reg) {
            LevelsCategory::RegToOutput
        } else {
            LevelsCategory::InputToOutput
        };
        add_hist_sample(&mut histograms, cat, best.info.depth);
        let port_name = resolve_to_string(interner, port.name);
        let sink = format!("port:{}", port_name);
        let instance_path = build_instance_path_for_net(module, interner, &net_traces, net_idx);
        maybe_update_max_example(&mut max_examples, cat, best.info.depth, sink, instance_path);
    }

    Ok(LevelsReport {
        histograms,
        num_instances: module.instances.len(),
        num_dff_instances,
        num_output_ports,
        max_examples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::IndexedLibrary;
    use crate::liberty::test_utils::{make_test_library, make_test_library_with_dff};
    use crate::liberty_proto::Library;
    use crate::netlist::parse::{NetRef, NetlistInstance, NetlistPort};
    use pretty_assertions::assert_eq;
    use string_interner::symbol::SymbolU32;
    use string_interner::{StringInterner, backend::StringBackend};

    fn sym(interner: &mut StringInterner<StringBackend<SymbolU32>>, s: &str) -> SymbolU32 {
        interner.get_or_intern(s)
    }

    fn simple_ports(interner: &mut StringInterner<StringBackend<SymbolU32>>) -> Vec<NetlistPort> {
        vec![
            NetlistPort {
                direction: PortDirection::Input,
                width: None,
                name: sym(interner, "a"),
            },
            NetlistPort {
                direction: PortDirection::Output,
                width: None,
                name: sym(interner, "y"),
            },
        ]
    }

    fn nets_for_names(
        interner: &mut StringInterner<StringBackend<SymbolU32>>,
        names: &[&str],
    ) -> Vec<Net> {
        names
            .iter()
            .map(|n| Net {
                name: sym(interner, n),
                width: None,
            })
            .collect()
    }

    #[test]
    fn input_to_output_depth_two() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let invx1 = sym(&mut interner, "INV");
        let u1 = sym(&mut interner, "u1");
        let u2 = sym(&mut interner, "u2");
        let top = sym(&mut interner, "top");

        let nets = nets_for_names(&mut interner, &["a", "n1", "n2", "y"]);

        let instances = vec![
            NetlistInstance {
                type_name: invx1,
                instance_name: u1,
                connections: vec![
                    (sym(&mut interner, "A"), NetRef::Simple(NetIndex(0))),
                    (sym(&mut interner, "Y"), NetRef::Simple(NetIndex(1))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
            NetlistInstance {
                type_name: invx1,
                instance_name: u2,
                connections: vec![
                    (sym(&mut interner, "A"), NetRef::Simple(NetIndex(1))),
                    (sym(&mut interner, "Y"), NetRef::Simple(NetIndex(3))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
        ];

        let module = NetlistModule {
            name: top,
            ports: simple_ports(&mut interner),
            wires: vec![NetIndex(0), NetIndex(1), NetIndex(2), NetIndex(3)],
            instances,
        };

        let lib: Library = make_test_library();
        let indexed = IndexedLibrary::new(lib);
        let dff_cells: HashSet<String> = HashSet::new();

        let report = compute_levels(&module, &nets, &interner, &indexed, &dff_cells, None).unwrap();
        let h = report
            .histograms
            .get(&LevelsCategory::InputToOutput)
            .unwrap();
        assert_eq!(h.get(&2).copied(), Some(1));
    }

    #[test]
    fn reg_to_reg_depth_one() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let invx1 = sym(&mut interner, "INV");
        let dffx1 = sym(&mut interner, "DFFX1");
        let u_dff0 = sym(&mut interner, "udff0");
        let u_inv = sym(&mut interner, "uinv");
        let u_dff1 = sym(&mut interner, "udff1");
        let top = sym(&mut interner, "top");

        let nets = nets_for_names(&mut interner, &["a", "q0", "n1", "q1", "y"]);

        let instances = vec![
            // DFF0: Q drives q0. We do not model D/CLK semantics; for boundary
            // purposes the output pin is a start boundary.
            NetlistInstance {
                type_name: dffx1,
                instance_name: u_dff0,
                connections: vec![
                    (sym(&mut interner, "D"), NetRef::Simple(NetIndex(0))),
                    (sym(&mut interner, "Q"), NetRef::Simple(NetIndex(1))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
            // INV: q0 -> n1
            NetlistInstance {
                type_name: invx1,
                instance_name: u_inv,
                connections: vec![
                    (sym(&mut interner, "A"), NetRef::Simple(NetIndex(1))),
                    (sym(&mut interner, "Y"), NetRef::Simple(NetIndex(2))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
            // DFF1: n1 -> q1 (sink boundary on input pin D)
            NetlistInstance {
                type_name: dffx1,
                instance_name: u_dff1,
                connections: vec![
                    (sym(&mut interner, "D"), NetRef::Simple(NetIndex(2))),
                    (sym(&mut interner, "Q"), NetRef::Simple(NetIndex(3))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
        ];

        let module = NetlistModule {
            name: top,
            ports: vec![
                NetlistPort {
                    direction: PortDirection::Input,
                    width: None,
                    name: sym(&mut interner, "a"),
                },
                NetlistPort {
                    direction: PortDirection::Output,
                    width: None,
                    name: sym(&mut interner, "y"),
                },
            ],
            wires: vec![
                NetIndex(0),
                NetIndex(1),
                NetIndex(2),
                NetIndex(3),
                NetIndex(4),
            ],
            instances,
        };

        let lib: Library = make_test_library_with_dff();
        let indexed = IndexedLibrary::new(lib);
        let mut dff_cells: HashSet<String> = HashSet::new();
        dff_cells.insert("DFFX1".to_string());

        let report = compute_levels(&module, &nets, &interner, &indexed, &dff_cells, None).unwrap();
        let h = report.histograms.get(&LevelsCategory::RegToReg).unwrap();
        assert_eq!(h.get(&1).copied(), Some(1));
    }

    #[test]
    fn input_to_reg_depth_one() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let invx1 = sym(&mut interner, "INV");
        let dffx1 = sym(&mut interner, "DFFX1");
        let u_inv = sym(&mut interner, "uinv");
        let u_dff0 = sym(&mut interner, "udff0");
        let top = sym(&mut interner, "top");

        let nets = nets_for_names(&mut interner, &["a", "n1", "q0", "y"]);

        let instances = vec![
            // INV: a -> n1
            NetlistInstance {
                type_name: invx1,
                instance_name: u_inv,
                connections: vec![
                    (sym(&mut interner, "A"), NetRef::Simple(NetIndex(0))),
                    (sym(&mut interner, "Y"), NetRef::Simple(NetIndex(1))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
            // DFF0: D sinks n1; Q drives q0.
            NetlistInstance {
                type_name: dffx1,
                instance_name: u_dff0,
                connections: vec![
                    (sym(&mut interner, "D"), NetRef::Simple(NetIndex(1))),
                    (sym(&mut interner, "Q"), NetRef::Simple(NetIndex(2))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
        ];

        let module = NetlistModule {
            name: top,
            ports: vec![
                NetlistPort {
                    direction: PortDirection::Input,
                    width: None,
                    name: sym(&mut interner, "a"),
                },
                NetlistPort {
                    direction: PortDirection::Output,
                    width: None,
                    name: sym(&mut interner, "y"),
                },
            ],
            wires: vec![NetIndex(0), NetIndex(1), NetIndex(2), NetIndex(3)],
            instances,
        };

        let lib: Library = make_test_library_with_dff();
        let indexed = IndexedLibrary::new(lib);
        let mut dff_cells: HashSet<String> = HashSet::new();
        dff_cells.insert("DFFX1".to_string());

        let report = compute_levels(&module, &nets, &interner, &indexed, &dff_cells, None).unwrap();
        let h = report.histograms.get(&LevelsCategory::InputToReg).unwrap();
        assert_eq!(h.get(&1).copied(), Some(1));
    }

    #[test]
    fn reg_to_output_depth_one() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let invx1 = sym(&mut interner, "INV");
        let dffx1 = sym(&mut interner, "DFFX1");
        let u_dff0 = sym(&mut interner, "udff0");
        let u_inv = sym(&mut interner, "uinv");
        let top = sym(&mut interner, "top");

        let nets = nets_for_names(&mut interner, &["a", "q0", "n1", "y"]);

        let instances = vec![
            // DFF0: Q drives q0.
            NetlistInstance {
                type_name: dffx1,
                instance_name: u_dff0,
                connections: vec![
                    (sym(&mut interner, "D"), NetRef::Simple(NetIndex(0))),
                    (sym(&mut interner, "Q"), NetRef::Simple(NetIndex(1))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
            // INV: q0 -> y
            NetlistInstance {
                type_name: invx1,
                instance_name: u_inv,
                connections: vec![
                    (sym(&mut interner, "A"), NetRef::Simple(NetIndex(1))),
                    (sym(&mut interner, "Y"), NetRef::Simple(NetIndex(3))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
        ];

        let module = NetlistModule {
            name: top,
            ports: vec![
                NetlistPort {
                    direction: PortDirection::Input,
                    width: None,
                    name: sym(&mut interner, "a"),
                },
                NetlistPort {
                    direction: PortDirection::Output,
                    width: None,
                    name: sym(&mut interner, "y"),
                },
            ],
            wires: vec![NetIndex(0), NetIndex(1), NetIndex(2), NetIndex(3)],
            instances,
        };

        let lib: Library = make_test_library_with_dff();
        let indexed = IndexedLibrary::new(lib);
        let mut dff_cells: HashSet<String> = HashSet::new();
        dff_cells.insert("DFFX1".to_string());

        let report = compute_levels(&module, &nets, &interner, &indexed, &dff_cells, None).unwrap();
        let h = report.histograms.get(&LevelsCategory::RegToOutput).unwrap();
        assert_eq!(h.get(&1).copied(), Some(1));
    }

    #[test]
    fn detects_combinational_cycle() {
        let mut interner: StringInterner<StringBackend<SymbolU32>> = StringInterner::new();
        let invx1 = sym(&mut interner, "INV");
        let u1 = sym(&mut interner, "u1");
        let u2 = sym(&mut interner, "u2");
        let top = sym(&mut interner, "top");

        let nets = nets_for_names(&mut interner, &["n1", "n2"]);

        let instances = vec![
            NetlistInstance {
                type_name: invx1,
                instance_name: u1,
                connections: vec![
                    (sym(&mut interner, "A"), NetRef::Simple(NetIndex(1))),
                    (sym(&mut interner, "Y"), NetRef::Simple(NetIndex(0))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
            NetlistInstance {
                type_name: invx1,
                instance_name: u2,
                connections: vec![
                    (sym(&mut interner, "A"), NetRef::Simple(NetIndex(0))),
                    (sym(&mut interner, "Y"), NetRef::Simple(NetIndex(1))),
                ],
                inst_lineno: 0,
                inst_colno: 0,
            },
        ];

        let module = NetlistModule {
            name: top,
            ports: vec![],
            wires: vec![NetIndex(0), NetIndex(1)],
            instances,
        };

        let lib: Library = make_test_library();
        let indexed = IndexedLibrary::new(lib);
        let dff_cells: HashSet<String> = HashSet::new();

        let err = compute_levels(&module, &nets, &interner, &indexed, &dff_cells, None)
            .expect_err("cycle should be detected");
        assert!(
            format!("{}", err).contains("cycle detected"),
            "unexpected error: {}",
            err
        );
    }
}
