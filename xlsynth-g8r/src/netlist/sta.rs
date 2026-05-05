// SPDX-License-Identifier: Apache-2.0

//! Basic static timing analysis (STA) for parsed gate-level netlists.
//!
//! This module implements a limited-scope, combinational max-arrival analysis:
//! - Uses Liberty combinational timing arcs (`cell_rise`, `cell_fall`,
//!   `rise_transition`, `fall_transition`).
//! - Propagates rise/fall arrival and transition values net-by-net.
//! - Assumes fixed transition at primary-input sources.
//! - Uses summed input-pin capacitance on each net plus a fixed module-output
//!   load to query timing tables.
//! - Treats primary-input sources as externally launched; it does not model
//!   sequential launch/capture timing such as clock-to-Q or setup checks.
//! - Rejects sequential output pins and conditional (`when`) timing arcs rather
//!   than approximating unsupported timing semantics in this limited-scope
//!   pass.
//! - Rejects non-monotone timing tables because later frontier reduction relies
//!   on larger transition/load queries not producing smaller delay/slew values.
//!
//! At a high level, the analysis is:
//! 1. Index each instance's cell/pin connectivity, recording one driver and all
//!    loads for each net.
//! 2. Build a combinational dependency graph between instances and compute the
//!    edge-specific capacitive load on each net.
//! 3. Seed primary inputs with zero arrival and the configured source
//!    transition; reject undriven non-input nets that feed logic or module
//!    outputs.
//! 4. Visit instances in topological order. For each output timing arc:
//!    - choose the relevant input edge candidates from the arc sense
//!      (`positive_unate`, `negative_unate`, or `non_unate`);
//!    - query Liberty delay and transition tables using input transition and
//!      output load;
//!    - produce output `(arrival, transition)` candidates and retain the
//!      non-dominated frontier while combining related pins/arcs. Keeping that
//!      frontier preserves cases where the latest-arriving candidate and the
//!      highest-transition candidate differ; pruning dominated candidates only
//!      removes candidates that are worse in both dimensions under the intended
//!      monotone timing model.
//! 5. Before storing an output-net result, collapse each rise/fall candidate
//!    set to its conservative envelope: maximum arrival and maximum transition.
//! 6. Report the maximum rise/fall arrival observed at module outputs.

use crate::liberty::LibraryWithTimingData;
use crate::liberty::timing_table::TimingTableArrayView;
use crate::liberty_proto::{LuTableTemplate, Pin, PinDirection, TimingArc, TimingTable};
use crate::netlist::parse::{Net, NetIndex, NetRef, NetlistModule, PortDirection};
use anyhow::{Result, anyhow};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeTiming {
    pub arrival: f64,
    pub transition: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignalTiming {
    pub rise: EdgeTiming,
    pub fall: EdgeTiming,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct EdgeTimingSet {
    values: Vec<EdgeTiming>,
}

impl EdgeTimingSet {
    fn from_single(edge: EdgeTiming) -> Self {
        let mut set = Self::default();
        set.insert(edge);
        set
    }

    fn insert(&mut self, candidate: EdgeTiming) {
        if self.values.contains(&candidate) {
            return;
        }

        if self
            .values
            .iter()
            .any(|existing| edge_timing_dominates(*existing, candidate))
        {
            return;
        }
        self.values
            .retain(|existing| !edge_timing_dominates(candidate, *existing));

        self.values.push(candidate);
        self.values.sort_by(|lhs, rhs| {
            lhs.arrival
                .partial_cmp(&rhs.arrival)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    lhs.transition
                        .partial_cmp(&rhs.transition)
                        .unwrap_or(Ordering::Equal)
                })
        });
    }

    fn extend_from(&mut self, rhs: &Self) {
        for edge in &rhs.values {
            self.insert(*edge);
        }
    }

    fn max_arrival_edge(&self) -> Option<EdgeTiming> {
        self.values
            .iter()
            .copied()
            .reduce(choose_worse_edge_timing_by_arrival)
    }

    fn iter(&self) -> impl Iterator<Item = EdgeTiming> + '_ {
        self.values.iter().copied()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct SignalTimingSet {
    rise: EdgeTimingSet,
    fall: EdgeTimingSet,
}

impl SignalTimingSet {
    fn from_single(signal: SignalTiming) -> Self {
        Self {
            rise: EdgeTimingSet::from_single(signal.rise),
            fall: EdgeTimingSet::from_single(signal.fall),
        }
    }

    fn merge(mut self, rhs: &Self) -> Self {
        self.rise.extend_from(&rhs.rise);
        self.fall.extend_from(&rhs.fall);
        self
    }

    fn as_report_signal_timing(&self) -> Option<SignalTiming> {
        Some(SignalTiming {
            rise: self.rise.max_arrival_edge()?,
            fall: self.fall.max_arrival_edge()?,
        })
    }

    fn worst_arrival(&self) -> Option<f64> {
        Some(
            self.rise
                .max_arrival_edge()?
                .arrival
                .max(self.fall.max_arrival_edge()?.arrival),
        )
    }
}

fn edge_timing_dominates(lhs: EdgeTiming, rhs: EdgeTiming) -> bool {
    (lhs.arrival >= rhs.arrival && lhs.transition >= rhs.transition)
        && (lhs.arrival > rhs.arrival || lhs.transition > rhs.transition)
}

fn parse_trace_env_flag(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sta_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("XLSYNTH_G8R_STA_TRACE")
            .map(|value| parse_trace_env_flag(value.as_str()))
            .unwrap_or(false)
    })
}

fn sta_trace(msg: impl FnOnce() -> String) {
    if sta_trace_enabled() {
        log::info!(target: "xlsynth_g8r::netlist::sta_trace", "{}", msg());
    }
}

fn format_edge_timing(edge: EdgeTiming) -> String {
    format!("(arr={:.6},tr={:.6})", edge.arrival, edge.transition)
}

fn format_edge_timing_set(set: &EdgeTimingSet) -> String {
    let parts: Vec<String> = set.values.iter().copied().map(format_edge_timing).collect();
    format!("[{}]", parts.join(", "))
}

fn format_optional_edge_timing(edge: Option<EdgeTiming>) -> String {
    edge.map(format_edge_timing)
        .unwrap_or_else(|| "<none>".to_string())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaOptions {
    /// Transition applied to primary-input source nets.
    pub primary_input_transition: f64,
    /// Extra load added to nets attached to module outputs.
    pub module_output_load: f64,
}

impl Default for StaOptions {
    fn default() -> Self {
        Self {
            primary_input_transition: 0.01,
            module_output_load: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StaReport {
    pub net_timing: Vec<Option<SignalTiming>>,
    pub worst_output_arrival: f64,
}

impl StaReport {
    pub fn timing_for_net(&self, net: NetIndex) -> Option<SignalTiming> {
        self.net_timing.get(net.0).copied().flatten()
    }
}

struct StaLibraryIndex<'a> {
    library: &'a crate::liberty_proto::Library,
    cell_by_name: HashMap<String, usize>,
    pin_by_cell: Vec<HashMap<String, usize>>,
}

impl<'a> StaLibraryIndex<'a> {
    fn new(library: &'a crate::liberty_proto::Library) -> Result<Self> {
        let mut cell_by_name = HashMap::new();
        let mut pin_by_cell = Vec::with_capacity(library.cells.len());
        for (cell_idx, cell) in library.cells.iter().enumerate() {
            if cell_by_name.insert(cell.name.clone(), cell_idx).is_some() {
                return Err(anyhow!(
                    "library defines cell '{}' more than once; duplicate cell names are unsupported in basic STA",
                    cell.name
                ));
            }
            let mut pin_map = HashMap::new();
            for (pin_idx, pin) in cell.pins.iter().enumerate() {
                if pin_map.insert(pin.name.clone(), pin_idx).is_some() {
                    return Err(anyhow!(
                        "library cell '{}' defines pin '{}' more than once; duplicate pin names are unsupported in basic STA",
                        cell.name,
                        pin.name
                    ));
                }
            }
            pin_by_cell.push(pin_map);
        }
        Ok(Self {
            library,
            cell_by_name,
            pin_by_cell,
        })
    }

    fn cell_index(&self, cell_name: &str) -> Option<usize> {
        self.cell_by_name.get(cell_name).copied()
    }

    fn pin(&self, cell_idx: usize, pin_name: &str) -> Option<&Pin> {
        let pin_idx = *self.pin_by_cell.get(cell_idx)?.get(pin_name)?;
        self.library.cells.get(cell_idx)?.pins.get(pin_idx)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EdgeLoadCapacitance {
    rise: f64,
    fall: f64,
}

#[derive(Clone, Debug)]
struct NetEndpoint {
    inst_idx: usize,
    pin_name: String,
}

pub fn analyze_combinational_max_arrival(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &LibraryWithTimingData,
    options: StaOptions,
) -> Result<StaReport> {
    analyze_combinational_max_arrival_proto(module, nets, interner, library.as_proto(), options)
}

fn analyze_combinational_max_arrival_proto(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &crate::liberty_proto::Library,
    options: StaOptions,
) -> Result<StaReport> {
    if !module.assigns.is_empty() {
        return Err(anyhow!(
            "module contains {} continuous assign statement(s); basic STA only supports structural netlists without assigns",
            module.assigns.len()
        ));
    }
    if !options.primary_input_transition.is_finite() {
        return Err(anyhow!(
            "primary_input_transition must be finite; got {}",
            options.primary_input_transition
        ));
    }
    if options.primary_input_transition < 0.0 {
        return Err(anyhow!(
            "primary_input_transition must be non-negative; got {}",
            options.primary_input_transition
        ));
    }
    if !options.module_output_load.is_finite() {
        return Err(anyhow!(
            "module_output_load must be finite; got {}",
            options.module_output_load
        ));
    }
    if options.module_output_load < 0.0 {
        return Err(anyhow!(
            "module_output_load must be non-negative; got {}",
            options.module_output_load
        ));
    }

    let lib = StaLibraryIndex::new(library)?;
    let instance_count = module.instances.len();

    let mut instance_cell_indices: Vec<usize> = Vec::with_capacity(instance_count);
    let mut instance_cell_names: Vec<String> = Vec::with_capacity(instance_count);
    let mut instance_pin_nets: Vec<HashMap<String, Vec<NetIndex>>> =
        Vec::with_capacity(instance_count);
    let mut instance_timing_related_input_pins: Vec<HashSet<String>> =
        Vec::with_capacity(instance_count);

    let mut net_drivers: Vec<Vec<NetEndpoint>> = vec![Vec::new(); nets.len()];
    let mut net_loads: Vec<Vec<NetEndpoint>> = vec![Vec::new(); nets.len()];

    for (inst_idx, inst) in module.instances.iter().enumerate() {
        let cell_name = resolve_symbol(interner, inst.type_name, "cell type")?;
        let cell_idx = lib.cell_index(cell_name.as_str()).ok_or_else(|| {
            anyhow!(
                "instance '{}' references unknown cell '{}'",
                resolve_symbol(interner, inst.instance_name, "instance name")
                    .unwrap_or_else(|_| "<unknown>".to_string()),
                cell_name
            )
        })?;
        instance_cell_indices.push(cell_idx);
        instance_cell_names.push(cell_name.clone());
        let mut timing_related_input_pins = HashSet::new();
        for output_pin in lib.library.cells[cell_idx]
            .pins
            .iter()
            .filter(|pin| pin.direction == PinDirection::Output as i32)
        {
            for arc in output_pin
                .timing_arcs
                .iter()
                .filter(|arc| StaTimingType::from_raw(arc.timing_type.as_str()).is_combinational())
            {
                for related_pin_name in split_related_pin_names(arc.related_pin.as_str()) {
                    let related_pin = lib.pin(cell_idx, related_pin_name).ok_or_else(|| {
                        anyhow!(
                            "cell '{}' output pin '{}' has timing arc with unknown related pin '{}'",
                            cell_name,
                            output_pin.name,
                            related_pin_name
                        )
                    })?;
                    if related_pin.direction != PinDirection::Input as i32 {
                        return Err(anyhow!(
                            "cell '{}' output pin '{}' has unsupported non-input related pin '{}'; basic STA only supports input-related combinational arcs",
                            cell_name,
                            output_pin.name,
                            related_pin_name
                        ));
                    }
                    timing_related_input_pins.insert(related_pin_name.to_string());
                }
            }
        }

        let mut pin_nets: HashMap<String, Vec<NetIndex>> = HashMap::new();
        for (port_sym, netref) in &inst.connections {
            let pin_name = resolve_symbol(interner, *port_sym, "pin name")?;
            if pin_nets.contains_key(pin_name.as_str()) {
                return Err(anyhow!(
                    "instance '{}' of '{}' connects pin '{}' more than once; duplicate pin bindings are unsupported in basic STA",
                    resolve_symbol(interner, inst.instance_name, "instance name")
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                    cell_name,
                    pin_name
                ));
            }
            let pin = lib.pin(cell_idx, pin_name.as_str()).ok_or_else(|| {
                anyhow!(
                    "instance '{}' of '{}' references unknown pin '{}'",
                    resolve_symbol(interner, inst.instance_name, "instance name")
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                    cell_name,
                    pin_name
                )
            })?;
            if pin.direction == PinDirection::Output as i32
                && matches!(netref, NetRef::Literal(_) | NetRef::Unconnected)
            {
                return Err(anyhow!(
                    "instance '{}' output pin '{}.{}' uses unsupported literal or unconnected binding",
                    resolve_symbol(interner, inst.instance_name, "instance name")
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                    cell_name,
                    pin_name
                ));
            }
            let uses_vector_connectivity = match netref {
                NetRef::Simple(net_idx) => nets
                    .get(net_idx.0)
                    .map(|net| net_width_is_multibit(net.width))
                    .unwrap_or(false),
                NetRef::BitSelect(_, _) | NetRef::PartSelect(_, _, _) | NetRef::Concat(_) => true,
                NetRef::Literal(_) | NetRef::Unconnected => false,
            };
            if uses_vector_connectivity {
                return Err(anyhow!(
                    "instance '{}' pin '{}.{}' uses vector connectivity; basic STA currently requires scalar net connections",
                    resolve_symbol(interner, inst.instance_name, "instance name")
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                    cell_name,
                    pin_name
                ));
            }
            if pin.direction == PinDirection::Input as i32
                && timing_related_input_pins.contains(pin_name.as_str())
                && matches!(netref, NetRef::Literal(_))
            {
                return Err(anyhow!(
                    "instance '{}' timing-related input pin '{}.{}' uses a literal binding; basic STA does not model constant-tied timing inputs",
                    resolve_symbol(interner, inst.instance_name, "instance name")
                        .unwrap_or_else(|_| "<unknown>".to_string()),
                    cell_name,
                    pin_name
                ));
            }
            let mut connected_nets = Vec::new();
            netref.collect_net_indices(&mut connected_nets);
            for net_idx in &connected_nets {
                if net_idx.0 >= nets.len() {
                    return Err(anyhow!(
                        "instance '{}' pin '{}' references out-of-range net index {}",
                        resolve_symbol(interner, inst.instance_name, "instance name")
                            .unwrap_or_else(|_| "<unknown>".to_string()),
                        pin_name,
                        net_idx.0
                    ));
                }
                match pin.direction {
                    d if d == PinDirection::Output as i32 => {
                        net_drivers[net_idx.0].push(NetEndpoint {
                            inst_idx,
                            pin_name: pin_name.clone(),
                        });
                    }
                    d if d == PinDirection::Input as i32 => {
                        net_loads[net_idx.0].push(NetEndpoint {
                            inst_idx,
                            pin_name: pin_name.clone(),
                        });
                    }
                    _ => {
                        return Err(anyhow!(
                            "instance '{}' pin '{}.{}' has unsupported direction value {}",
                            resolve_symbol(interner, inst.instance_name, "instance name")
                                .unwrap_or_else(|_| "<unknown>".to_string()),
                            cell_name,
                            pin_name,
                            pin.direction
                        ));
                    }
                }
            }
            pin_nets.insert(pin_name, connected_nets);
        }
        instance_pin_nets.push(pin_nets);
        instance_timing_related_input_pins.push(timing_related_input_pins);
    }

    let mut module_output_nets: Vec<NetIndex> = Vec::new();
    let mut has_module_output = vec![false; nets.len()];
    let mut is_module_input = vec![false; nets.len()];
    for port in &module.ports {
        let port_name = resolve_symbol(interner, port.name, "port name")
            .unwrap_or_else(|_| "<unknown>".to_string());
        match port.direction {
            PortDirection::Input => {
                if let Some(net_idx) = module.find_net_index(port.name, nets) {
                    is_module_input[net_idx.0] = true;
                }
            }
            PortDirection::Output => {
                if let Some(net_idx) = module.find_net_index(port.name, nets)
                    && !has_module_output[net_idx.0]
                {
                    has_module_output[net_idx.0] = true;
                    module_output_nets.push(net_idx);
                }
            }
            PortDirection::Inout => {
                return Err(anyhow!(
                    "module port '{}' is inout; basic STA only supports input and output ports",
                    port_name
                ));
            }
        }
    }
    module_output_nets.sort_by_key(|n| n.0);

    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); instance_count];
    let mut indegree: Vec<usize> = vec![0; instance_count];
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
    for (net_idx, drivers) in net_drivers.iter().enumerate() {
        if is_module_input[net_idx] && !drivers.is_empty() {
            let net_name = net_name_for_index(nets, interner, NetIndex(net_idx));
            return Err(anyhow!(
                "module input net '{}' also has an instance driver; basic STA does not support multiply driven primary inputs",
                net_name
            ));
        }
        if drivers.len() > 1 {
            let net_name = net_name_for_index(nets, interner, NetIndex(net_idx));
            return Err(anyhow!(
                "net '{}' has {} drivers; wired multi-driver nets are unsupported in basic STA",
                net_name,
                drivers.len()
            ));
        }
        let Some(driver) = drivers.first() else {
            continue;
        };
        for load in &net_loads[net_idx] {
            if !instance_timing_related_input_pins[load.inst_idx].contains(&load.pin_name) {
                continue;
            }
            if load.inst_idx == driver.inst_idx {
                continue;
            }
            if seen_edges.insert((driver.inst_idx, load.inst_idx)) {
                successors[driver.inst_idx].push(load.inst_idx);
                indegree[load.inst_idx] += 1;
            }
        }
    }

    let mut net_load_capacitance = vec![EdgeLoadCapacitance::default(); nets.len()];
    for (net_idx, loads) in net_loads.iter().enumerate() {
        let mut cap = EdgeLoadCapacitance::default();
        for load in loads {
            let cell_idx = instance_cell_indices[load.inst_idx];
            let pin = lib.pin(cell_idx, load.pin_name.as_str()).ok_or_else(|| {
                anyhow!(
                    "could not resolve load pin '{}.{}' while computing output load",
                    instance_cell_names[load.inst_idx],
                    load.pin_name
                )
            })?;
            let pin_cap = effective_input_capacitance_by_edge(
                pin,
                &format!(
                    "load pin '{}.{}'",
                    instance_cell_names[load.inst_idx], load.pin_name
                ),
            )?;
            cap.rise += pin_cap.rise;
            cap.fall += pin_cap.fall;
        }
        if has_module_output[net_idx] {
            cap.rise += options.module_output_load;
            cap.fall += options.module_output_load;
        }
        net_load_capacitance[net_idx] = cap;
    }

    let source_timing = SignalTiming {
        rise: EdgeTiming {
            arrival: 0.0,
            transition: options.primary_input_transition,
        },
        fall: EdgeTiming {
            arrival: 0.0,
            transition: options.primary_input_transition,
        },
    };
    let source_timing_set = SignalTimingSet::from_single(source_timing);
    let mut net_timing_sets: Vec<Option<SignalTimingSet>> = vec![None; nets.len()];

    for (net_idx, drivers) in net_drivers.iter().enumerate() {
        if !drivers.is_empty() {
            continue;
        }
        if is_module_input[net_idx] {
            net_timing_sets[net_idx] = Some(source_timing_set.clone());
            continue;
        }
        if !net_loads[net_idx].is_empty() || has_module_output[net_idx] {
            return Err(anyhow!(
                "net '{}' is undriven and is not a module input; basic STA does not support floating source nets",
                net_name_for_index(nets, interner, NetIndex(net_idx))
            ));
        }
    }

    let mut queue = VecDeque::new();
    for (idx, deg) in indegree.iter().enumerate() {
        if *deg == 0 {
            queue.push_back(idx);
        }
    }

    let mut processed = 0usize;
    while let Some(inst_idx) = queue.pop_front() {
        processed += 1;

        let cell_idx = instance_cell_indices[inst_idx];
        let cell_name = &instance_cell_names[inst_idx];
        let inst_pin_map = &instance_pin_nets[inst_idx];
        let instance = &module.instances[inst_idx];
        let instance_name = resolve_symbol(interner, instance.instance_name, "instance name")
            .unwrap_or_else(|_| "<unknown>".to_string());

        for pin in &lib.library.cells[cell_idx].pins {
            if pin.direction != PinDirection::Output as i32 {
                continue;
            }
            let Some(output_nets) = inst_pin_map.get(pin.name.as_str()) else {
                continue;
            };
            if output_nets.is_empty() {
                continue;
            }
            if let Some(unsupported_arc) = pin
                .timing_arcs
                .iter()
                .find(|arc| !StaTimingType::from_raw(arc.timing_type.as_str()).is_combinational())
            {
                return Err(anyhow!(
                    "basic STA only supports combinational output pins; instance '{}' output pin '{}.{}' has unsupported timing type '{}'",
                    instance_name,
                    cell_name,
                    pin.name,
                    unsupported_arc.timing_type
                ));
            }
            let combinational_arcs: Vec<&TimingArc> = pin
                .timing_arcs
                .iter()
                .filter(|arc| StaTimingType::from_raw(arc.timing_type.as_str()).is_combinational())
                .collect();
            if combinational_arcs.is_empty() {
                return Err(anyhow!(
                    "basic STA only supports combinational output pins; instance '{}' output pin '{}.{}' has no combinational timing arcs",
                    instance_name,
                    cell_name,
                    pin.name
                ));
            }
            if let Some(conditional_arc) =
                combinational_arcs.iter().find(|arc| !arc.when.is_empty())
            {
                return Err(anyhow!(
                    "basic STA does not support conditional timing arcs; instance '{}' output pin '{}.{}' has related pin '{}' with when='{}'",
                    instance_name,
                    cell_name,
                    pin.name,
                    conditional_arc.related_pin,
                    conditional_arc.when
                ));
            }

            for output_net in output_nets {
                let output_load = net_load_capacitance[output_net.0];
                let output_net_name = net_name_for_index(nets, interner, *output_net);
                let mut accumulated: Option<SignalTimingSet> = None;

                for arc in &combinational_arcs {
                    for related_pin_name in split_related_pin_names(arc.related_pin.as_str()) {
                        let related_nets =
                            inst_pin_map.get(related_pin_name).ok_or_else(|| {
                                anyhow!(
                                    "instance '{}' output pin '{}.{}' requires timing-related input pin '{}' to be connected",
                                    instance_name,
                                    cell_name,
                                    pin.name,
                                    related_pin_name
                                )
                            })?;
                        if related_nets.is_empty() {
                            return Err(anyhow!(
                                "instance '{}' output pin '{}.{}' has unconnected timing-related input pin '{}'",
                                instance_name,
                                cell_name,
                                pin.name,
                                related_pin_name
                            ));
                        }
                        for related_net in related_nets {
                            let input_timing_set = net_timing_sets[related_net.0].as_ref().ok_or_else(|| {
                            anyhow!(
                                "missing source timing for net '{}' feeding '{}.{}' (related pin '{}')",
                                net_name_for_index(nets, interner, *related_net),
                                cell_name,
                                pin.name,
                                related_pin_name
                            )
                        })?;
                            let related_net_name = net_name_for_index(nets, interner, *related_net);
                            let context = format!(
                                "{}.{} (instance '{}') related_pin '{}'",
                                cell_name, pin.name, instance_name, related_pin_name
                            );
                            let candidate = evaluate_arc_set(
                                lib.library,
                                arc,
                                input_timing_set,
                                output_load,
                                &context,
                            )?;
                            sta_trace(|| {
                                format!(
                                    "inst={} cell={} out_pin={} out_net={} related_pin={} related_net={} when={} sense={} type={} rise_candidates={} fall_candidates={} rise_pick={} fall_pick={}",
                                    instance_name,
                                    cell_name,
                                    pin.name,
                                    output_net_name,
                                    related_pin_name,
                                    related_net_name,
                                    if arc.when.is_empty() {
                                        "<empty>"
                                    } else {
                                        arc.when.as_str()
                                    },
                                    arc.timing_sense,
                                    arc.timing_type,
                                    format_edge_timing_set(&candidate.rise),
                                    format_edge_timing_set(&candidate.fall),
                                    format_optional_edge_timing(candidate.rise.max_arrival_edge()),
                                    format_optional_edge_timing(candidate.fall.max_arrival_edge()),
                                )
                            });
                            accumulated = Some(match accumulated {
                                Some(prev) => prev.merge(&candidate),
                                None => candidate,
                            });
                        }
                    }
                }

                let mut out_timing_set = accumulated.ok_or_else(|| {
                    anyhow!(
                        "no usable combinational timing arcs for instance '{}' pin '{}.{}'",
                        instance_name,
                        cell_name,
                        pin.name
                    )
                })?;
                // Use a conservative per-edge envelope: max arrival and max
                // transition may come from different source candidates.
                collapse_signal_timing_set_to_envelope(&mut out_timing_set);
                sta_trace(|| {
                    format!(
                        "inst={} cell={} out_pin={} out_net={} merged_rise={} merged_fall={} merged_rise_pick={} merged_fall_pick={}",
                        instance_name,
                        cell_name,
                        pin.name,
                        output_net_name,
                        format_edge_timing_set(&out_timing_set.rise),
                        format_edge_timing_set(&out_timing_set.fall),
                        format_optional_edge_timing(out_timing_set.rise.max_arrival_edge()),
                        format_optional_edge_timing(out_timing_set.fall.max_arrival_edge()),
                    )
                });

                net_timing_sets[output_net.0] =
                    Some(match net_timing_sets[output_net.0].as_ref() {
                        Some(prev) => prev.clone().merge(&out_timing_set),
                        None => out_timing_set,
                    });
            }
        }

        for succ in &successors[inst_idx] {
            indegree[*succ] = indegree[*succ].saturating_sub(1);
            if indegree[*succ] == 0 {
                queue.push_back(*succ);
            }
        }
    }

    if processed != instance_count {
        return Err(anyhow!(
            "combinational cycle detected or unresolved dependencies: processed {} of {} instances",
            processed,
            instance_count
        ));
    }

    let mut worst_output_arrival: Option<f64> = None;
    for net_idx in &module_output_nets {
        let timing_set = net_timing_sets[net_idx.0].as_ref().ok_or_else(|| {
            anyhow!(
                "missing timing result for module output net '{}'",
                net_name_for_index(nets, interner, *net_idx)
            )
        })?;
        let output_arrival = timing_set.worst_arrival().ok_or_else(|| {
            anyhow!(
                "missing edge timing candidates for module output net '{}'",
                net_name_for_index(nets, interner, *net_idx)
            )
        })?;
        worst_output_arrival = Some(
            worst_output_arrival
                .map(|current| current.max(output_arrival))
                .unwrap_or(output_arrival),
        );
    }

    let net_timing: Vec<Option<SignalTiming>> = net_timing_sets
        .into_iter()
        .map(|timing_set| timing_set.and_then(|set| set.as_report_signal_timing()))
        .collect();

    Ok(StaReport {
        net_timing,
        worst_output_arrival: worst_output_arrival.unwrap_or(0.0),
    })
}
fn effective_input_capacitance_by_edge(pin: &Pin, context: &str) -> Result<EdgeLoadCapacitance> {
    // max_capacitance is a design-rule limit, not nominal pin capacitance.
    // Prefer edge-specific capacitance when provided; fall back to nominal.
    let capacitance = EdgeLoadCapacitance {
        rise: pin
            .rise_capacitance
            .or(pin.capacitance)
            .or(pin.fall_capacitance)
            .unwrap_or(0.0),
        fall: pin
            .fall_capacitance
            .or(pin.capacitance)
            .or(pin.rise_capacitance)
            .unwrap_or(0.0),
    };
    validate_non_negative_finite(capacitance.rise, "rise capacitance", context)?;
    validate_non_negative_finite(capacitance.fall, "fall capacitance", context)?;
    Ok(capacitance)
}

#[cfg(test)]
fn effective_input_capacitance(pin: &Pin) -> f64 {
    // `max_capacitance` is a design-rule limit, not nominal pin capacitance.
    // Net load should be built from sink-pin capacitance, not that limit.
    if let Some(cap) = pin.capacitance {
        return cap;
    }
    [pin.rise_capacitance, pin.fall_capacitance]
        .into_iter()
        .flatten()
        .fold(0.0, f64::max)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaTimingType {
    Combinational,
    CombinationalRise,
    CombinationalFall,
    Other,
}

impl StaTimingType {
    fn from_raw(raw: &str) -> Self {
        match raw {
            "" | "combinational" => Self::Combinational,
            "combinational_rise" => Self::CombinationalRise,
            "combinational_fall" => Self::CombinationalFall,
            _ => Self::Other,
        }
    }

    fn is_combinational(self) -> bool {
        !matches!(self, Self::Other)
    }

    fn produces_rise(self) -> bool {
        !matches!(self, Self::CombinationalFall)
    }

    fn produces_fall(self) -> bool {
        !matches!(self, Self::CombinationalRise)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaTimingSense {
    Unspecified,
    PositiveUnate,
    NegativeUnate,
    NonUnate,
    Other,
}

impl StaTimingSense {
    fn from_raw(raw: &str) -> Self {
        match raw {
            "" => Self::Unspecified,
            "positive_unate" => Self::PositiveUnate,
            "negative_unate" => Self::NegativeUnate,
            "non_unate" => Self::NonUnate,
            _ => Self::Other,
        }
    }

    fn may_use_either_input_edge(self) -> bool {
        matches!(self, Self::Unspecified | Self::NonUnate)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaTimingTableKind {
    CellRise,
    CellFall,
    RiseTransition,
    FallTransition,
}

impl StaTimingTableKind {
    fn as_raw(self) -> &'static str {
        match self {
            Self::CellRise => "cell_rise",
            Self::CellFall => "cell_fall",
            Self::RiseTransition => "rise_transition",
            Self::FallTransition => "fall_transition",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LibertyTableKind {
    CellRise,
    CellFall,
    RiseTransition,
    FallTransition,
    RisePower,
    FallPower,
    Other,
}

impl LibertyTableKind {
    fn from_raw(raw: &str) -> Self {
        match raw {
            "cell_rise" => Self::CellRise,
            "cell_fall" => Self::CellFall,
            "rise_transition" => Self::RiseTransition,
            "fall_transition" => Self::FallTransition,
            "rise_power" => Self::RisePower,
            "fall_power" => Self::FallPower,
            _ => Self::Other,
        }
    }

    fn template_kind(self) -> Option<LuTableTemplateKind> {
        match self {
            Self::CellRise | Self::CellFall | Self::RiseTransition | Self::FallTransition => {
                Some(LuTableTemplateKind::Timing)
            }
            Self::RisePower | Self::FallPower => Some(LuTableTemplateKind::Power),
            Self::Other => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LuTableTemplateKind {
    Timing,
    Power,
    Other,
}

impl LuTableTemplateKind {
    fn from_raw(raw: &str) -> Self {
        match raw {
            "lu_table_template" => Self::Timing,
            "power_lut_template" => Self::Power,
            _ => Self::Other,
        }
    }

    fn as_raw(self) -> &'static str {
        match self {
            Self::Timing => "lu_table_template",
            Self::Power => "power_lut_template",
            Self::Other => "<unsupported>",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AxisVariable {
    Unspecified,
    InputTransition,
    OutputLoad,
    Other,
}

impl AxisVariable {
    fn from_raw(raw: &str) -> Self {
        match raw.trim() {
            "" => Self::Unspecified,
            "input_net_transition" | "input_transition_time" => Self::InputTransition,
            "total_output_net_capacitance" => Self::OutputLoad,
            _ => Self::Other,
        }
    }
}

fn split_related_pin_names(related_pin: &str) -> impl Iterator<Item = &str> {
    related_pin.split_whitespace()
}

fn net_width_is_multibit(width: Option<(u32, u32)>) -> bool {
    matches!(width, Some((msb, lsb)) if msb != lsb)
}

#[cfg(test)]
fn evaluate_arc(
    library: &crate::liberty_proto::Library,
    arc: &TimingArc,
    input_timing: SignalTiming,
    output_load: f64,
    context: &str,
) -> Result<SignalTiming> {
    let set = evaluate_arc_set(
        library,
        arc,
        &SignalTimingSet::from_single(input_timing),
        EdgeLoadCapacitance {
            rise: output_load,
            fall: output_load,
        },
        context,
    )?;
    set.as_report_signal_timing().ok_or_else(|| {
        anyhow!("{context}: no output timing candidates were produced after arc evaluation")
    })
}

fn evaluate_arc_set(
    library: &crate::liberty_proto::Library,
    arc: &TimingArc,
    input_timing: &SignalTimingSet,
    output_load: EdgeLoadCapacitance,
    context: &str,
) -> Result<SignalTimingSet> {
    let timing_type = StaTimingType::from_raw(arc.timing_type.as_str());
    let timing_sense = StaTimingSense::from_raw(arc.timing_sense.as_str());
    let all_inputs = if timing_sense.may_use_either_input_edge() {
        let mut combined = input_timing.rise.clone();
        combined.extend_from(&input_timing.fall);
        Some(combined)
    } else {
        None
    };
    let source_edges = |output_edge_is_rise: bool| -> Result<&EdgeTimingSet> {
        match (timing_sense, output_edge_is_rise) {
            (StaTimingSense::PositiveUnate, true) => Ok(&input_timing.rise),
            (StaTimingSense::PositiveUnate, false) => Ok(&input_timing.fall),
            (StaTimingSense::NegativeUnate, true) => Ok(&input_timing.fall),
            (StaTimingSense::NegativeUnate, false) => Ok(&input_timing.rise),
            (StaTimingSense::Unspecified | StaTimingSense::NonUnate, _) => Ok(all_inputs
                .as_ref()
                .expect("non-unate input set should be built")),
            _ => Err(anyhow!(
                "{context}: unsupported timing_sense '{}'",
                arc.timing_sense
            )),
        }
    };

    let mut output = SignalTimingSet::default();
    if timing_type.produces_rise() {
        let cell_rise = find_unique_table(arc, StaTimingTableKind::CellRise, context)?;
        let rise_transition = find_unique_table(arc, StaTimingTableKind::RiseTransition, context)?;
        output.rise = evaluate_output_edge_set(
            library,
            cell_rise,
            rise_transition,
            source_edges(true)?,
            output_load.rise,
            context,
            StaTimingTableKind::CellRise,
            StaTimingTableKind::RiseTransition,
        )?;
    }
    if timing_type.produces_fall() {
        let cell_fall = find_unique_table(arc, StaTimingTableKind::CellFall, context)?;
        let fall_transition = find_unique_table(arc, StaTimingTableKind::FallTransition, context)?;
        output.fall = evaluate_output_edge_set(
            library,
            cell_fall,
            fall_transition,
            source_edges(false)?,
            output_load.fall,
            context,
            StaTimingTableKind::CellFall,
            StaTimingTableKind::FallTransition,
        )?;
    }
    Ok(output)
}

fn evaluate_output_edge_set(
    library: &crate::liberty_proto::Library,
    delay_table: &TimingTable,
    slew_table: &TimingTable,
    source_edges: &EdgeTimingSet,
    output_load: f64,
    context: &str,
    delay_kind: StaTimingTableKind,
    slew_kind: StaTimingTableKind,
) -> Result<EdgeTimingSet> {
    let mut outputs = EdgeTimingSet::default();

    for source_edge in source_edges.iter() {
        let delay = evaluate_table(
            library,
            delay_table,
            source_edge.transition,
            output_load,
            &format!("{context} {}", delay_kind.as_raw()),
        )?;
        let transition = evaluate_table(
            library,
            slew_table,
            source_edge.transition,
            output_load,
            &format!("{context} {}", slew_kind.as_raw()),
        )?;
        validate_non_negative_finite(
            transition,
            &format!("{} result", slew_kind.as_raw()),
            context,
        )?;
        let arrival = source_edge.arrival + delay;
        if !arrival.is_finite() {
            return Err(anyhow!(
                "{context}: propagated arrival must be finite; got {} + {} = {}",
                source_edge.arrival,
                delay,
                arrival
            ));
        }
        outputs.insert(EdgeTiming {
            arrival,
            transition,
        });
    }

    if outputs.values.is_empty() {
        return Err(anyhow!(
            "{context}: no source edge candidates for '{}'/'{}' evaluation",
            delay_kind.as_raw(),
            slew_kind.as_raw()
        ));
    }

    Ok(outputs)
}

fn collapse_signal_timing_set_to_envelope(signal: &mut SignalTimingSet) {
    collapse_edge_timing_set_to_envelope(&mut signal.rise);
    collapse_edge_timing_set_to_envelope(&mut signal.fall);
}

fn collapse_edge_timing_set_to_envelope(set: &mut EdgeTimingSet) {
    let max_arrival = set.values.iter().map(|edge| edge.arrival).reduce(f64::max);
    let max_transition = set
        .values
        .iter()
        .map(|edge| edge.transition)
        .reduce(f64::max);
    if let (Some(arrival), Some(transition)) = (max_arrival, max_transition) {
        set.values.clear();
        set.values.push(EdgeTiming {
            arrival,
            transition,
        });
    }
}

fn choose_worse_edge_timing_by_arrival(lhs: EdgeTiming, rhs: EdgeTiming) -> EdgeTiming {
    if lhs.arrival > rhs.arrival {
        lhs
    } else if rhs.arrival > lhs.arrival {
        rhs
    } else {
        EdgeTiming {
            arrival: lhs.arrival,
            transition: lhs.transition.max(rhs.transition),
        }
    }
}

fn find_unique_table<'a>(
    arc: &'a TimingArc,
    kind: StaTimingTableKind,
    context: &str,
) -> Result<&'a TimingTable> {
    let mut matches = arc
        .tables
        .iter()
        .filter(|table| table.kind == kind.as_raw());
    let first = matches
        .next()
        .ok_or_else(|| anyhow!("{context}: missing '{}' timing table", kind.as_raw()))?;
    if matches.next().is_some() {
        return Err(anyhow!(
            "{context}: multiple '{}' timing tables are unsupported in basic STA",
            kind.as_raw()
        ));
    }
    Ok(first)
}

fn expected_template_kind_for_timing_table(table: &TimingTable) -> Result<LuTableTemplateKind> {
    LibertyTableKind::from_raw(table.kind.as_str())
        .template_kind()
        .ok_or_else(|| anyhow!("unsupported Liberty table kind '{}'", table.kind))
}

fn evaluate_table(
    library: &crate::liberty_proto::Library,
    table: &TimingTable,
    input_transition: f64,
    output_load: f64,
    context: &str,
) -> Result<f64> {
    validate_non_negative_finite(input_transition, "input transition query", context)?;
    validate_non_negative_finite(output_load, "output load query", context)?;
    let array = TimingTableArrayView::from_timing_table(table)
        .map_err(|e| anyhow!("{context}: invalid timing table payload: {e}"))?;
    for value in &table.values {
        if !value.is_finite() {
            return Err(anyhow!(
                "{context}: timing table contains non-finite value {}",
                value
            ));
        }
    }

    let template: Option<&LuTableTemplate> = if table.template_id == 0 {
        None
    } else {
        let idx = (table.template_id - 1) as usize;
        let tmpl = library.lu_table_templates.get(idx).ok_or_else(|| {
            anyhow!(
                "{context}: template_id {} out of range ({} templates)",
                table.template_id,
                library.lu_table_templates.len()
            )
        })?;
        let expected_kind = expected_template_kind_for_timing_table(table)
            .map_err(|e| anyhow!("{context}: {e}"))?;
        let actual_kind = LuTableTemplateKind::from_raw(tmpl.kind.as_str());
        if actual_kind != expected_kind {
            return Err(anyhow!(
                "{context}: template_id {} kind mismatch; got '{}' expected '{}'",
                table.template_id,
                tmpl.kind,
                expected_kind.as_raw()
            ));
        }
        Some(tmpl)
    };

    let axis_1 = effective_axis(&table.index_1, template.map(|t| t.index_1.as_slice()));
    let axis_2 = effective_axis(&table.index_2, template.map(|t| t.index_2.as_slice()));
    let axis_3 = effective_axis(&table.index_3, template.map(|t| t.index_3.as_slice()));

    let variable_1 = template.map(|t| t.variable_1.as_str()).unwrap_or("");
    let variable_2 = template.map(|t| t.variable_2.as_str()).unwrap_or("");
    let variable_3 = template.map(|t| t.variable_3.as_str()).unwrap_or("");

    let rank = array.rank();
    if rank > 3 {
        return Err(anyhow!(
            "{context}: rank-{} timing table is unsupported in basic STA",
            rank
        ));
    }

    let all_axes = [axis_1, axis_2, axis_3];
    let all_variables = [variable_1, variable_2, variable_3];
    validate_effective_axes(table, all_axes, context)?;
    validate_monotone_timing_table(&array, table.dimensions.as_slice(), context)?;
    if rank == 0 {
        return array
            .get(&[])
            .ok_or_else(|| anyhow!("{context}: scalar timing table had no value"));
    }
    let mut bounds: Vec<(usize, usize, f64)> = Vec::with_capacity(rank);

    for axis_idx in 0..rank {
        let axis = all_axes[axis_idx];
        if axis.is_empty() {
            return Err(anyhow!(
                "{context}: missing axis data for rank-{} table on axis {}",
                rank,
                axis_idx + 1
            ));
        }
        if let Some(value) = axis.iter().copied().find(|value| !value.is_finite()) {
            return Err(anyhow!(
                "{context}: axis {} contains non-finite value {}",
                axis_idx + 1,
                value
            ));
        }
        if !is_strictly_increasing(axis) {
            return Err(anyhow!(
                "{context}: axis {} is not strictly increasing",
                axis_idx + 1
            ));
        }
        let query = axis_query_value(
            all_variables[axis_idx],
            axis_idx,
            input_transition,
            output_load,
            context,
        )?;
        let axis_lo = axis[0];
        let axis_hi = axis[axis.len() - 1];
        if query < axis_lo || query > axis_hi {
            sta_trace(|| {
                format!(
                    "table_query_out_of_range context='{}' axis={} var='{}' query={:.6} axis_lo={:.6} axis_hi={:.6}",
                    context,
                    axis_idx + 1,
                    all_variables[axis_idx],
                    query,
                    axis_lo,
                    axis_hi,
                )
            });
        }
        bounds.push(bracket_axis(axis, query));
    }

    let mut indices = vec![0usize; rank];
    let mut varying_axes: Vec<(usize, usize, usize, f64)> = Vec::with_capacity(rank);
    for (axis_idx, (lo, hi, t)) in bounds.iter().copied().enumerate() {
        if lo == hi {
            indices[axis_idx] = lo;
        } else {
            varying_axes.push((axis_idx, lo, hi, t));
        }
    }

    let corner_count = 1usize << varying_axes.len();
    let mut result = 0.0;
    for corner in 0..corner_count {
        let mut weight = 1.0;
        for (bit_idx, (axis_idx, lo, hi, t)) in varying_axes.iter().enumerate() {
            if ((corner >> bit_idx) & 1) == 1 {
                indices[*axis_idx] = *hi;
                weight *= *t;
            } else {
                indices[*axis_idx] = *lo;
                weight *= 1.0 - *t;
            }
        }
        let value = array
            .get(indices.as_slice())
            .ok_or_else(|| anyhow!("{context}: could not index timing table at {:?}", indices))?;
        result += weight * value;
    }
    if !result.is_finite() {
        return Err(anyhow!(
            "{context}: timing table evaluation produced non-finite result {}",
            result
        ));
    }
    Ok(result)
}

fn validate_non_negative_finite(value: f64, what: &str, context: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(anyhow!("{context}: {what} must be finite; got {}", value));
    }
    if value < 0.0 {
        return Err(anyhow!(
            "{context}: {what} must be non-negative; got {}",
            value
        ));
    }
    Ok(())
}

fn axis_rank(index_1: &[f64], index_2: &[f64], index_3: &[f64]) -> usize {
    if !index_3.is_empty() {
        3
    } else if !index_2.is_empty() {
        2
    } else if !index_1.is_empty() {
        1
    } else {
        0
    }
}

fn axes_are_contiguous(index_1: &[f64], index_2: &[f64], index_3: &[f64]) -> bool {
    if !index_3.is_empty() && (index_1.is_empty() || index_2.is_empty()) {
        return false;
    }
    if !index_2.is_empty() && index_1.is_empty() {
        return false;
    }
    true
}

fn validate_effective_axes(table: &TimingTable, axes: [&[f64]; 3], context: &str) -> Result<()> {
    if !axes_are_contiguous(axes[0], axes[1], axes[2]) {
        return Err(anyhow!(
            "{context}: timing table has non-contiguous effective axes (index_1={}, index_2={}, index_3={})",
            axes[0].len(),
            axes[1].len(),
            axes[2].len()
        ));
    }
    let expected_rank = axis_rank(axes[0], axes[1], axes[2]);
    if table.dimensions.len() != expected_rank {
        return Err(anyhow!(
            "{context}: timing table dimension rank {} does not match effective axis rank {}",
            table.dimensions.len(),
            expected_rank
        ));
    }
    for (axis_idx, axis) in axes.iter().take(expected_rank).enumerate() {
        let dimension = table.dimensions[axis_idx] as usize;
        if dimension != axis.len() {
            return Err(anyhow!(
                "{context}: timing table axis {} dimension {} does not match effective axis length {}",
                axis_idx + 1,
                dimension,
                axis.len()
            ));
        }
    }
    Ok(())
}

/// Rejects tables that decrease along any axis; envelope reduction requires
/// monotone queries.
fn validate_monotone_timing_table(
    array: &TimingTableArrayView<'_>,
    dimensions: &[u32],
    context: &str,
) -> Result<()> {
    if dimensions.is_empty() {
        return Ok(());
    }

    let mut indices = vec![0usize; dimensions.len()];
    loop {
        let current = array.get(indices.as_slice()).ok_or_else(|| {
            anyhow!(
                "{context}: could not index timing table at {:?} while checking monotonicity",
                indices
            )
        })?;

        for (axis_idx, dimension) in dimensions.iter().copied().enumerate() {
            if indices[axis_idx] + 1 >= dimension as usize {
                continue;
            }
            let mut next_indices = indices.clone();
            next_indices[axis_idx] += 1;
            let next = array.get(next_indices.as_slice()).ok_or_else(|| {
                anyhow!(
                    "{context}: could not index timing table at {:?} while checking monotonicity",
                    next_indices
                )
            })?;
            if next < current {
                return Err(anyhow!(
                    "{context}: timing table decreases along axis {} between {:?}={} and {:?}={}; basic STA requires monotone timing tables",
                    axis_idx + 1,
                    indices,
                    current,
                    next_indices,
                    next
                ));
            }
        }

        let mut axis_idx = dimensions.len();
        while axis_idx > 0 {
            axis_idx -= 1;
            indices[axis_idx] += 1;
            if indices[axis_idx] < dimensions[axis_idx] as usize {
                break;
            }
            indices[axis_idx] = 0;
        }
        if axis_idx == 0 && indices[0] == 0 {
            break;
        }
    }

    Ok(())
}

fn effective_axis<'a>(table_axis: &'a [f64], template_axis: Option<&'a [f64]>) -> &'a [f64] {
    if table_axis.is_empty() {
        template_axis.unwrap_or(&[])
    } else {
        table_axis
    }
}

fn axis_query_value(
    variable_name: &str,
    axis_idx: usize,
    input_transition: f64,
    output_load: f64,
    context: &str,
) -> Result<f64> {
    match AxisVariable::from_raw(variable_name) {
        AxisVariable::Unspecified => match axis_idx {
            0 => Ok(input_transition),
            1 => Ok(output_load),
            _ => Err(anyhow!(
                "{context}: missing variable name for axis {}; cannot infer query value",
                axis_idx + 1
            )),
        },
        AxisVariable::InputTransition => Ok(input_transition),
        AxisVariable::OutputLoad => Ok(output_load),
        AxisVariable::Other => Err(anyhow!(
            "{context}: unsupported axis variable '{}' for basic STA",
            variable_name
        )),
    }
}

fn is_strictly_increasing(values: &[f64]) -> bool {
    values
        .windows(2)
        .all(|w| w.first().copied().unwrap_or(0.0) < w.get(1).copied().unwrap_or(0.0))
}

fn bracket_axis(axis: &[f64], query: f64) -> (usize, usize, f64) {
    debug_assert!(!axis.is_empty());
    if axis.len() == 1 {
        return (0, 0, 0.0);
    }

    let last = axis.len() - 1;
    if query <= axis[0] {
        let lo = 0usize;
        let hi = 1usize;
        let denom = axis[hi] - axis[lo];
        if denom.abs() < f64::EPSILON {
            return (0, 0, 0.0);
        }
        return (lo, hi, (query - axis[lo]) / denom);
    }
    if query >= axis[last] {
        let lo = last - 1;
        let hi = last;
        let denom = axis[hi] - axis[lo];
        if denom.abs() < f64::EPSILON {
            return (last, last, 0.0);
        }
        return (lo, hi, (query - axis[lo]) / denom);
    }

    for idx in 0..last {
        let lo = axis[idx];
        let hi = axis[idx + 1];
        if query >= lo && query <= hi {
            let denom = hi - lo;
            if denom.abs() < f64::EPSILON {
                return (idx, idx, 0.0);
            }
            let t = (query - lo) / denom;
            return (idx, idx + 1, t);
        }
    }

    (last, last, 0.0)
}

fn resolve_symbol(
    interner: &StringInterner<StringBackend<SymbolU32>>,
    sym: SymbolU32,
    what: &str,
) -> Result<String> {
    interner
        .resolve(sym)
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("could not resolve {} symbol {:?}", what, sym))
}

fn net_name_for_index(
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    net_idx: NetIndex,
) -> String {
    nets.get(net_idx.0)
        .and_then(|n| interner.resolve(n.name))
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("<net:{}>", net_idx.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty_proto::{Cell, Pin, TimingArc, TimingTable};
    use crate::netlist::bench_synth_netlist;
    use crate::netlist::parse::{Parser, TokenScanner};

    fn parse_single_module(
        src: &str,
    ) -> (
        NetlistModule,
        Vec<Net>,
        StringInterner<StringBackend<SymbolU32>>,
    ) {
        let bytes = src.as_bytes().to_vec();
        let lines: Vec<String> = src.lines().map(ToString::to_string).collect();
        let lookup = move |lineno: u32| lines.get((lineno - 1) as usize).cloned();
        let scanner = TokenScanner::with_line_lookup(std::io::Cursor::new(bytes), Box::new(lookup));
        let mut parser = Parser::new(scanner);
        let mut modules = parser.parse_file().expect("parse should succeed");
        assert_eq!(modules.len(), 1);
        (modules.remove(0), parser.nets, parser.interner)
    }

    fn find_net_index(
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        name: &str,
    ) -> NetIndex {
        nets.iter()
            .enumerate()
            .find_map(|(idx, n)| {
                if interner.resolve(n.name) == Some(name) {
                    Some(NetIndex(idx))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("missing net '{}'", name))
    }

    fn scalar_table(kind: &str, value: f64) -> TimingTable {
        TimingTable {
            kind: kind.to_string(),
            values: vec![value],
            dimensions: vec![],
            ..Default::default()
        }
    }

    #[test]
    fn axis_query_value_accepts_only_explicit_supported_variable_names() {
        assert_eq!(
            axis_query_value("input_net_transition", 0, 0.2, 0.7, "axis").unwrap(),
            0.2
        );
        assert_eq!(
            axis_query_value("input_transition_time", 0, 0.2, 0.7, "axis").unwrap(),
            0.2
        );
        assert_eq!(
            axis_query_value("total_output_net_capacitance", 1, 0.2, 0.7, "axis").unwrap(),
            0.7
        );
        assert!(
            axis_query_value("made_up_transition", 0, 0.2, 0.7, "axis")
                .unwrap_err()
                .to_string()
                .contains("unsupported axis variable")
        );
        assert!(
            axis_query_value("made_up_capacitance", 1, 0.2, 0.7, "axis")
                .unwrap_err()
                .to_string()
                .contains("unsupported axis variable")
        );
    }

    #[test]
    fn template_kind_lookup_rejects_unknown_power_named_table_kinds() {
        let table = scalar_table("made_up_power", 1.0);
        assert!(
            expected_template_kind_for_timing_table(&table)
                .unwrap_err()
                .to_string()
                .contains("unsupported Liberty table kind")
        );
    }

    fn assert_close(lhs: f64, rhs: f64) {
        assert!(
            (lhs - rhs).abs() <= 1e-9,
            "expected {} ~= {} (|diff|={})",
            lhs,
            rhs,
            (lhs - rhs).abs()
        );
    }

    fn scalar_inv_library() -> crate::liberty_proto::Library {
        crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "negative_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 2.0),
                                scalar_table("cell_fall", 3.0),
                                scalar_table("rise_transition", 0.2),
                                scalar_table("fall_transition", 0.3),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn sta_rejects_continuous_assigns() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n;
  INV u0 (.A(a), .Y(n));
  assign y = n;
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("continuous assigns should be rejected");
        assert!(error.to_string().contains("continuous assign"));
    }

    #[test]
    fn sta_rejects_non_finite_options() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let transition_error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions {
                primary_input_transition: f64::NAN,
                module_output_load: 0.0,
            },
        )
        .expect_err("NaN primary-input transition should be rejected");
        assert!(
            transition_error
                .to_string()
                .contains("primary_input_transition must be finite")
        );

        let load_error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions {
                primary_input_transition: 0.01,
                module_output_load: f64::INFINITY,
            },
        )
        .expect_err("infinite module-output load should be rejected");
        assert!(
            load_error
                .to_string()
                .contains("module_output_load must be finite")
        );
    }

    #[test]
    fn sta_rejects_negative_options() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let transition_error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions {
                primary_input_transition: -0.1,
                module_output_load: 0.0,
            },
        )
        .expect_err("negative primary-input transition should be rejected");
        assert!(
            transition_error
                .to_string()
                .contains("primary_input_transition must be non-negative")
        );

        let load_error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions {
                primary_input_transition: 0.01,
                module_output_load: -0.1,
            },
        )
        .expect_err("negative module-output load should be rejected");
        assert!(
            load_error
                .to_string()
                .contains("module_output_load must be non-negative")
        );
    }

    #[test]
    fn sta_rejects_vector_pin_connectivity_until_bit_level_timing_is_supported() {
        let src = r#"
module top (a, y);
  input [1:0] a;
  output [1:0] y;
  wire [1:0] a;
  wire [1:0] y;
  INV u0 (.A(a[0]), .Y(y[0]));
  INV u1 (.A(a[1]), .Y(y[1]));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("vector pin connectivity should be rejected");
        assert!(error.to_string().contains("vector connectivity"));
    }

    #[test]
    fn sta_rejects_whole_vector_simple_connectivity() {
        let src = r#"
module top (a, y);
  input [1:0] a;
  output [1:0] y;
  wire [1:0] a;
  wire [1:0] y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("whole-vector simple connectivity should be rejected");
        assert!(error.to_string().contains("vector connectivity"));
    }

    #[test]
    fn sta_accepts_implicit_scalar_connectivity() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(n));
  INV u1 (.A(n), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect("implicit scalar wire should be accepted");
        assert_close(report.worst_output_arrival, 5.0);
    }

    #[test]
    fn sta_rejects_floating_internal_source_nets() {
        let src = r#"
module top (y);
  output y;
  wire y;
  wire floating_internal;
  INV u0 (.A(floating_internal), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("floating internal source nets should be rejected");
        assert!(error.to_string().contains("undriven"));
    }

    #[test]
    fn sta_rejects_missing_timing_related_input_pins() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  NAND2 u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "B".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                            TimingArc {
                                related_pin: "B".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("missing timing-related input pins should be rejected");
        assert!(
            error
                .to_string()
                .contains("requires timing-related input pin 'B' to be connected")
        );
    }

    #[test]
    fn sta_rejects_literal_tied_timing_inputs() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  NAND2 u0 (.A(a), .B(1'b0), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "B".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                            TimingArc {
                                related_pin: "B".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 4.0),
                                    scalar_table("cell_fall", 5.0),
                                    scalar_table("rise_transition", 0.4),
                                    scalar_table("fall_transition", 0.5),
                                ],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("literal-tied timing inputs should be rejected");
        assert!(
            error
                .to_string()
                .contains("does not model constant-tied timing inputs")
        );
    }

    #[test]
    fn sta_rejects_non_input_related_pins() {
        let src = r#"
module top (a, y0, y1);
  input a;
  output y0;
  output y1;
  wire a;
  wire y0;
  wire y1;
  DUALOUT u0 (.A(a), .Y0(y0), .Y1(y1));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "DUALOUT".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y0".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 1.0),
                                scalar_table("cell_fall", 1.0),
                                scalar_table("rise_transition", 0.1),
                                scalar_table("fall_transition", 0.1),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y1".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "Y0".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 1.0),
                                scalar_table("cell_fall", 1.0),
                                scalar_table("rise_transition", 0.1),
                                scalar_table("fall_transition", 0.1),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("non-input related pins should be rejected");
        assert!(
            error
                .to_string()
                .contains("unsupported non-input related pin 'Y0'")
        );
    }

    #[test]
    fn sta_rejects_primary_inputs_with_instance_drivers() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(y), .Y(a));
  INV u1 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("primary inputs with instance drivers should be rejected");
        assert!(error.to_string().contains("also has an instance driver"));
    }

    #[test]
    fn sta_rejects_duplicate_instance_pin_bindings() {
        let src = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  INV u0 (.A(a), .A(b), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("duplicate instance pin bindings should be rejected");
        assert!(
            error
                .to_string()
                .contains("connects pin 'A' more than once")
        );
    }

    #[test]
    fn sta_rejects_output_pins_bound_to_literals() {
        let src = r#"
module top (a);
  input a;
  wire a;
  INV u0 (.A(a), .Y(1'b0));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("literal output pin bindings should be rejected");
        assert!(
            error
                .to_string()
                .contains("uses unsupported literal or unconnected binding")
        );
    }

    #[test]
    fn sta_rejects_inout_ports() {
        let src = r#"
module top (io);
  inout io;
  wire io;
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &scalar_inv_library(),
            StaOptions::default(),
        )
        .expect_err("inout ports should be rejected");
        assert!(error.to_string().contains("is inout"));
    }

    #[test]
    fn sta_combines_one_sided_combinational_rise_and_fall_arcs() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational_rise".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("rise_transition", 0.2),
                                ],
                                ..Default::default()
                            },
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational_fall".to_string(),
                                tables: vec![
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.1,
                module_output_load: 0.0,
            },
        )
        .expect("one-sided combinational arcs should combine");
        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 2.0);
        assert_close(y_timing.fall.arrival, 3.0);
        assert_close(y_timing.rise.transition, 0.2);
        assert_close(y_timing.fall.transition, 0.3);
        assert_close(report.worst_output_arrival, 3.0);
    }

    #[test]
    fn sta_reports_negative_worst_output_arrival_without_zero_clamp() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", -2.0),
                                scalar_table("cell_fall", -1.0),
                                scalar_table("rise_transition", 0.2),
                                scalar_table("fall_transition", 0.3),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("sta should succeed");
        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, -2.0);
        assert_close(y_timing.fall.arrival, -1.0);
        assert_close(report.worst_output_arrival, -1.0);
    }

    #[test]
    fn sta_ignores_non_timing_input_pins_when_building_dependencies() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  PASS_WITH_EN u0 (.A(a), .EN(y), .Y(n0));
  INV u1 (.A(n0), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![
                Cell {
                    name: "PASS_WITH_EN".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "EN".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "INV".to_string(),
                    pins: scalar_inv_library().cells[0].pins.clone(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("non-timing EN feedback should not create a timing cycle");
        assert_close(report.worst_output_arrival, 5.0);
    }

    #[test]
    fn sta_splits_multi_pin_related_pin_names() {
        let src = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  NAND2 u0 (.A(a), .B(b), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "B".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A B".to_string(),
                            timing_sense: "negative_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 2.0),
                                scalar_table("cell_fall", 3.0),
                                scalar_table("rise_transition", 0.2),
                                scalar_table("fall_transition", 0.3),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect("multi-pin related_pin names should be expanded");
        assert_close(report.worst_output_arrival, 3.0);
    }

    #[test]
    fn sta_rejects_non_combinational_output_pins() {
        let src = r#"
module top (d, clk, q);
  input d;
  input clk;
  output q;
  wire d;
  wire clk;
  wire q;
  DFF u0 (.D(d), .CLK(clk), .Q(q));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "DFF".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "D".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "CLK".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Q".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "CLK".to_string(),
                            timing_type: "rising_edge".to_string(),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("sequential output pin should be rejected");
        assert!(
            error
                .to_string()
                .contains("basic STA only supports combinational output pins")
        );
    }

    #[test]
    fn sta_rejects_conditional_timing_arcs() {
        let src = r#"
module top (a, en, y);
  input a;
  input en;
  output y;
  wire a;
  wire en;
  wire y;
  BUF_EN u0 (.A(a), .EN(en), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "BUF_EN".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "EN".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            when: "EN".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 2.0),
                                scalar_table("cell_fall", 3.0),
                                scalar_table("rise_transition", 0.2),
                                scalar_table("fall_transition", 0.3),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("conditional timing arcs should be rejected");
        assert!(
            error
                .to_string()
                .contains("does not support conditional timing arcs")
        );
    }

    #[test]
    fn sta_rejects_mixed_output_timing_types() {
        let src = r#"
module top (a, clk, y);
  input a;
  input clk;
  output y;
  wire a;
  wire clk;
  wire y;
  MIXED u0 (.A(a), .CLK(clk), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "MIXED".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "CLK".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 2.0),
                                    scalar_table("cell_fall", 3.0),
                                    scalar_table("rise_transition", 0.2),
                                    scalar_table("fall_transition", 0.3),
                                ],
                                ..Default::default()
                            },
                            TimingArc {
                                related_pin: "CLK".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "rising_edge".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 4.0),
                                    scalar_table("cell_fall", 5.0),
                                    scalar_table("rise_transition", 0.4),
                                    scalar_table("fall_transition", 0.5),
                                ],
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let error = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions::default(),
        )
        .expect_err("mixed output timing types should be rejected");
        assert!(
            error
                .to_string()
                .contains("has unsupported timing type 'rising_edge'")
        );
    }

    #[test]
    fn effective_input_capacitance_ignores_max_capacitance_constraint() {
        let with_nominal = Pin {
            capacitance: Some(0.6),
            rise_capacitance: Some(0.7),
            fall_capacitance: Some(0.5),
            max_capacitance: Some(46.1),
            ..Default::default()
        };
        assert_close(effective_input_capacitance(&with_nominal), 0.6);

        let no_nominal = Pin {
            rise_capacitance: Some(0.6),
            fall_capacitance: Some(0.5),
            max_capacitance: Some(46.1),
            ..Default::default()
        };
        assert_close(effective_input_capacitance(&no_nominal), 0.6);

        let only_max = Pin {
            max_capacitance: Some(46.1),
            ..Default::default()
        };
        assert_close(effective_input_capacitance(&only_max), 0.0);
    }

    #[test]
    fn effective_input_capacitance_by_edge_rejects_negative_values() {
        let negative = Pin {
            rise_capacitance: Some(-0.1),
            fall_capacitance: Some(0.2),
            ..Default::default()
        };
        let error = effective_input_capacitance_by_edge(&negative, "pin")
            .expect_err("negative capacitance should be rejected");
        assert!(
            error
                .to_string()
                .contains("rise capacitance must be non-negative")
        );
    }

    #[test]
    fn sta_uses_nominal_input_capacitance_for_net_load() {
        let src = r#"
module top (a, n, y);
  input a;
  output n;
  output y;
  wire n;
  wire y;
  INV u0 (.A(a), .Y(n));
  INV u1 (.A(n), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2, 2],
            values,
            ..Default::default()
        };
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_2d".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![0.0, 1.0],
                index_2: vec![0.0, 10.0],
                ..Default::default()
            }],
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        capacitance: Some(1.0),
                        max_capacitance: Some(1000.0),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                table("cell_rise", vec![0.0, 10.0, 0.0, 10.0]),
                                table("cell_fall", vec![0.0, 10.0, 0.0, 10.0]),
                                table("rise_transition", vec![0.0, 0.0, 0.0, 0.0]),
                                table("fall_transition", vec![0.0, 0.0, 0.0, 0.0]),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.0,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let n_idx = find_net_index(&nets, &interner, "n");
        let y_idx = find_net_index(&nets, &interner, "y");
        let n_timing = report.timing_for_net(n_idx).expect("timing for n");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");

        assert_close(n_timing.rise.arrival, 1.0);
        assert_close(n_timing.fall.arrival, 1.0);
        assert_close(y_timing.rise.arrival, 1.0);
        assert_close(y_timing.fall.arrival, 1.0);
    }

    #[test]
    fn sta_inv_chain_propagates_arrival_and_transition() {
        let src = bench_synth_netlist::make_chain_netlist(2);
        let (module, nets, interner) = parse_single_module(src.as_str());
        let lib = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                scalar_table("cell_rise", 2.0),
                                scalar_table("cell_fall", 3.0),
                                scalar_table("rise_transition", 0.4),
                                scalar_table("fall_transition", 0.5),
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.2,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 4.0);
        assert_close(y_timing.fall.arrival, 6.0);
        assert_close(y_timing.rise.transition, 0.4);
        assert_close(y_timing.fall.transition, 0.5);
        assert_close(report.worst_output_arrival, 6.0);
    }

    #[test]
    fn sta_negative_unate_uses_opposite_input_edge() {
        let src = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  wire n0;
  INV u0 (.A(a), .Y(n0));
  NAND2 u1 (.A(n0), .B(b), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let inv_arc = TimingArc {
            related_pin: "A".to_string(),
            timing_sense: "positive_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                scalar_table("cell_rise", 2.0),
                scalar_table("cell_fall", 7.0),
                scalar_table("rise_transition", 0.2),
                scalar_table("fall_transition", 0.7),
            ],
            ..Default::default()
        };
        let nand_arc_a = TimingArc {
            related_pin: "A".to_string(),
            timing_sense: "negative_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                scalar_table("cell_rise", 100.0),
                scalar_table("cell_fall", 10.0),
                scalar_table("rise_transition", 1.0),
                scalar_table("fall_transition", 1.0),
            ],
            ..Default::default()
        };
        let nand_arc_b = TimingArc {
            related_pin: "B".to_string(),
            timing_sense: "negative_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                scalar_table("cell_rise", 1.0),
                scalar_table("cell_fall", 1.0),
                scalar_table("rise_transition", 0.5),
                scalar_table("fall_transition", 0.5),
            ],
            ..Default::default()
        };
        let lib = crate::liberty_proto::Library {
            cells: vec![
                Cell {
                    name: "INV".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![inv_arc],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "NAND2".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "B".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![nand_arc_a, nand_arc_b],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.1,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 107.0);
        assert_close(y_timing.fall.arrival, 12.0);
        assert_close(report.worst_output_arrival, 107.0);
    }

    #[test]
    fn sta_non_unate_keeps_arrival_and_transition_correlated_per_source_edge() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1d_transition".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![1.0, 5.0],
                ..Default::default()
            }],
            ..Default::default()
        };

        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2],
            values,
            ..Default::default()
        };

        let arc = TimingArc {
            related_pin: "A".to_string(),
            timing_sense: "non_unate".to_string(),
            timing_type: "combinational".to_string(),
            tables: vec![
                table("cell_rise", vec![10.0, 50.0]),
                table("cell_fall", vec![4.0, 20.0]),
                table("rise_transition", vec![2.0, 8.0]),
                table("fall_transition", vec![3.0, 7.0]),
            ],
            ..Default::default()
        };

        let input = SignalTiming {
            rise: EdgeTiming {
                arrival: 200.0,
                transition: 1.0,
            },
            fall: EdgeTiming {
                arrival: 100.0,
                transition: 5.0,
            },
        };

        let output = evaluate_arc(&lib, &arc, input, 0.0, "non_unate_test").expect("evaluate arc");

        // Rise edge should come from rise input edge (200 + 10), not mixed with
        // fall-edge transition.
        assert_close(output.rise.arrival, 210.0);
        assert_close(output.rise.transition, 2.0);

        // Fall edge should also come from rise input edge (200 + 4).
        assert_close(output.fall.arrival, 204.0);
        assert_close(output.fall.transition, 3.0);
    }
    #[test]
    fn sta_composed_logic_keeps_non_dominated_input_candidates() {
        let src = r#"
module top (a, b, y);
  input a;
  input b;
  output y;
  wire a;
  wire b;
  wire y;
  wire n0;
  MERGE u0 (.A(a), .B(b), .Y(n0));
  LOADSENS u1 (.A(n0), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);

        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2],
            values,
            ..Default::default()
        };

        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1d_transition".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![1.0, 5.0],
                ..Default::default()
            }],
            cells: vec![
                Cell {
                    name: "MERGE".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "B".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![
                                TimingArc {
                                    related_pin: "A".to_string(),
                                    timing_sense: "positive_unate".to_string(),
                                    timing_type: "combinational".to_string(),
                                    tables: vec![
                                        scalar_table("cell_rise", 10.0),
                                        scalar_table("cell_fall", 10.0),
                                        scalar_table("rise_transition", 1.0),
                                        scalar_table("fall_transition", 1.0),
                                    ],
                                    ..Default::default()
                                },
                                TimingArc {
                                    related_pin: "B".to_string(),
                                    timing_sense: "positive_unate".to_string(),
                                    timing_type: "combinational".to_string(),
                                    tables: vec![
                                        scalar_table("cell_rise", 8.0),
                                        scalar_table("cell_fall", 8.0),
                                        scalar_table("rise_transition", 5.0),
                                        scalar_table("fall_transition", 5.0),
                                    ],
                                    ..Default::default()
                                },
                            ],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "LOADSENS".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    table("cell_rise", vec![10.0, 50.0]),
                                    table("cell_fall", vec![10.0, 50.0]),
                                    table("rise_transition", vec![1.0, 5.0]),
                                    table("fall_transition", vec![1.0, 5.0]),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.0,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 60.0);
        assert_close(y_timing.fall.arrival, 60.0);
        assert_close(report.worst_output_arrival, 60.0);
    }

    #[test]
    fn sta_composed_logic_handles_mixed_sense_arcs_across_stages() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  wire n0;
  wire n1;
  SKEW u0 (.A(a), .Y(n0));
  DUAL u1 (.A(n0), .Y(n1));
  LOADSENS u2 (.A(n1), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);

        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2],
            values,
            ..Default::default()
        };

        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1d_transition".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![1.0, 5.0],
                ..Default::default()
            }],
            cells: vec![
                Cell {
                    name: "SKEW".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    scalar_table("cell_rise", 40.0),
                                    scalar_table("cell_fall", 0.0),
                                    scalar_table("rise_transition", 1.0),
                                    scalar_table("fall_transition", 5.0),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "DUAL".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![
                                TimingArc {
                                    related_pin: "A".to_string(),
                                    timing_sense: "positive_unate".to_string(),
                                    timing_type: "combinational".to_string(),
                                    tables: vec![
                                        scalar_table("cell_rise", 10.0),
                                        scalar_table("cell_fall", 10.0),
                                        scalar_table("rise_transition", 1.0),
                                        scalar_table("fall_transition", 1.0),
                                    ],
                                    ..Default::default()
                                },
                                TimingArc {
                                    related_pin: "A".to_string(),
                                    timing_sense: "negative_unate".to_string(),
                                    timing_type: "combinational".to_string(),
                                    tables: vec![
                                        scalar_table("cell_rise", 20.0),
                                        scalar_table("cell_fall", 20.0),
                                        scalar_table("rise_transition", 5.0),
                                        scalar_table("fall_transition", 5.0),
                                    ],
                                    ..Default::default()
                                },
                            ],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "LOADSENS".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    table("cell_rise", vec![10.0, 50.0]),
                                    table("cell_fall", vec![10.0, 50.0]),
                                    table("rise_transition", vec![1.0, 5.0]),
                                    table("fall_transition", vec![1.0, 5.0]),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.0,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");

        assert_close(y_timing.rise.arrival, 100.0);
        assert_close(y_timing.fall.arrival, 110.0);
        assert_close(report.worst_output_arrival, 110.0);
    }

    #[test]
    fn sta_interpolates_two_dimensional_tables() {
        let src = r#"
module top (a, y);
  input a;
  output y;
  wire a;
  wire y;
  INV u0 (.A(a), .Y(y));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_2d".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![0.1, 0.3],
                index_2: vec![1.0, 3.0],
                ..Default::default()
            }],
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        name: "Y".to_string(),
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "positive_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            tables: vec![
                                TimingTable {
                                    kind: "cell_rise".to_string(),
                                    template_id: 1,
                                    dimensions: vec![2, 2],
                                    values: vec![10.0, 20.0, 30.0, 40.0],
                                    ..Default::default()
                                },
                                TimingTable {
                                    kind: "cell_fall".to_string(),
                                    template_id: 1,
                                    dimensions: vec![2, 2],
                                    values: vec![5.0, 7.0, 9.0, 11.0],
                                    ..Default::default()
                                },
                                TimingTable {
                                    kind: "rise_transition".to_string(),
                                    template_id: 1,
                                    dimensions: vec![2, 2],
                                    values: vec![1.0, 2.0, 3.0, 4.0],
                                    ..Default::default()
                                },
                                TimingTable {
                                    kind: "fall_transition".to_string(),
                                    template_id: 1,
                                    dimensions: vec![2, 2],
                                    values: vec![2.0, 4.0, 6.0, 8.0],
                                    ..Default::default()
                                },
                            ],
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.2,
                module_output_load: 2.0,
            },
        )
        .expect("sta should succeed");

        let y_idx = find_net_index(&nets, &interner, "y");
        let y_timing = report.timing_for_net(y_idx).expect("timing for y");
        assert_close(y_timing.rise.arrival, 25.0);
        assert_close(y_timing.fall.arrival, 8.0);
        assert_close(y_timing.rise.transition, 2.5);
        assert_close(y_timing.fall.transition, 5.0);
        assert_close(report.worst_output_arrival, 25.0);
    }

    #[test]
    fn sta_uses_edge_specific_input_capacitance_for_net_load() {
        let src = r#"
module top (a, n);
  input a;
  output n;
  wire a;
  wire n;
  DRV u0 (.A(a), .Y(n));
  SINK u1 (.A(n));
endmodule
"#;
        let (module, nets, interner) = parse_single_module(src);

        let table = |kind: &str, values: Vec<f64>| TimingTable {
            kind: kind.to_string(),
            template_id: 1,
            dimensions: vec![2],
            values,
            ..Default::default()
        };

        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1d_load".to_string(),
                variable_1: "total_output_net_capacitance".to_string(),
                index_1: vec![1.0, 10.0],
                ..Default::default()
            }],
            cells: vec![
                Cell {
                    name: "DRV".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            name: "A".to_string(),
                            ..Default::default()
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            name: "Y".to_string(),
                            timing_arcs: vec![TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                tables: vec![
                                    table("cell_rise", vec![10.0, 20.0]),
                                    table("cell_fall", vec![30.0, 40.0]),
                                    table("rise_transition", vec![0.0, 0.0]),
                                    table("fall_transition", vec![0.0, 0.0]),
                                ],
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                },
                Cell {
                    name: "SINK".to_string(),
                    pins: vec![Pin {
                        direction: PinDirection::Input as i32,
                        name: "A".to_string(),
                        capacitance: Some(9.0),
                        rise_capacitance: Some(1.0),
                        fall_capacitance: Some(10.0),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let report = analyze_combinational_max_arrival_proto(
            &module,
            &nets,
            &interner,
            &lib,
            StaOptions {
                primary_input_transition: 0.0,
                module_output_load: 0.0,
            },
        )
        .expect("sta should succeed");

        let n_idx = find_net_index(&nets, &interner, "n");
        let n_timing = report.timing_for_net(n_idx).expect("timing for n");

        // Rising output uses rise_capacitance (1.0); falling output uses
        // fall_capacitance (10.0), even when nominal capacitance is present.
        assert_close(n_timing.rise.arrival, 10.0);
        assert_close(n_timing.fall.arrival, 40.0);
    }

    #[test]
    fn evaluate_table_does_not_duplicate_weight_on_singleton_axes() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_singleton_x_axis".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![0.0],
                index_2: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![1, 2],
            values: vec![1.0, 2.0],
            ..Default::default()
        };

        let low = evaluate_table(&lib, &table, 7.0, 0.0, "singleton_axis_low").expect("table eval");
        assert_close(low, 1.0);

        let mid = evaluate_table(&lib, &table, 7.0, 0.5, "singleton_axis_mid").expect("table eval");
        assert_close(mid, 1.5);
    }

    #[test]
    fn evaluate_table_extrapolates_below_axis_range() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_2d_extrap".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![5.0, 10.0],
                index_2: vec![0.72, 1.44],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2, 2],
            values: vec![6.90715, 9.84125, 8.69936, 11.6159],
            ..Default::default()
        };

        let extrapolated =
            evaluate_table(&lib, &table, 0.0, 0.619_928, "extrapolated").expect("table eval");
        assert!(
            (extrapolated - 4.704_690_972_222_221).abs() <= 2e-6,
            "expected 4.704690972222221 ~= {extrapolated}"
        );
    }

    #[test]
    fn evaluate_table_rejects_axis_dimension_mismatch() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_bad_extent".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![3],
            values: vec![1.0, 2.0, 3.0],
            ..Default::default()
        };

        let error = evaluate_table(&lib, &table, 0.5, 0.0, "bad_extent")
            .expect_err("axis extent mismatch should be rejected");
        assert!(error.to_string().contains("axis 1 dimension 3"));
    }

    #[test]
    fn evaluate_table_rejects_axis_rank_mismatch() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_bad_rank".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                index_1: vec![0.0, 1.0],
                index_2: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2],
            values: vec![1.0, 2.0],
            ..Default::default()
        };

        let error = evaluate_table(&lib, &table, 0.5, 0.5, "bad_rank")
            .expect_err("axis rank mismatch should be rejected");
        assert!(error.to_string().contains("dimension rank 1"));
    }

    #[test]
    fn evaluate_table_rejects_non_finite_axes_and_values() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_non_finite_axis".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![0.0, f64::INFINITY],
                ..Default::default()
            }],
            ..Default::default()
        };
        let axis_table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2],
            values: vec![1.0, 2.0],
            ..Default::default()
        };
        let axis_error = evaluate_table(&lib, &axis_table, 0.5, 0.0, "bad_axis")
            .expect_err("non-finite axes should be rejected");
        assert!(
            axis_error
                .to_string()
                .contains("axis 1 contains non-finite")
        );

        let value_table = TimingTable {
            kind: "cell_rise".to_string(),
            values: vec![f64::NAN],
            ..Default::default()
        };
        let value_error = evaluate_table(
            &crate::liberty_proto::Library::default(),
            &value_table,
            0.0,
            0.0,
            "bad_value",
        )
        .expect_err("non-finite table values should be rejected");
        assert!(
            value_error
                .to_string()
                .contains("timing table contains non-finite value")
        );
    }

    #[test]
    fn evaluate_table_rejects_duplicate_axis_points() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_duplicate_axis".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![0.0, 0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![3],
            values: vec![1.0, 2.0, 3.0],
            ..Default::default()
        };

        let error = evaluate_table(&lib, &table, 0.5, 0.0, "duplicate_axis")
            .expect_err("duplicate axis points should be rejected");
        assert!(
            error
                .to_string()
                .contains("axis 1 is not strictly increasing")
        );
    }

    #[test]
    fn evaluate_table_rejects_non_monotone_values() {
        let lib = crate::liberty_proto::Library {
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_non_monotone".to_string(),
                variable_1: "input_net_transition".to_string(),
                index_1: vec![0.0, 1.0],
                ..Default::default()
            }],
            ..Default::default()
        };
        let table = TimingTable {
            kind: "cell_rise".to_string(),
            template_id: 1,
            dimensions: vec![2],
            values: vec![2.0, 1.0],
            ..Default::default()
        };

        let error = evaluate_table(&lib, &table, 0.5, 0.0, "non_monotone")
            .expect_err("non-monotone values should be rejected");
        assert!(
            error
                .to_string()
                .contains("basic STA requires monotone timing tables")
        );
    }

    #[test]
    fn evaluate_output_edge_set_rejects_negative_transition_results() {
        let input_timing = EdgeTimingSet::from_single(EdgeTiming {
            arrival: 0.0,
            transition: 0.1,
        });
        let delay_table = scalar_table("cell_rise", 1.0);
        let slew_table = scalar_table("rise_transition", -0.1);

        let error = evaluate_output_edge_set(
            &crate::liberty_proto::Library::default(),
            &delay_table,
            &slew_table,
            &input_timing,
            0.0,
            "negative_transition",
            StaTimingTableKind::CellRise,
            StaTimingTableKind::RiseTransition,
        )
        .expect_err("negative transition outputs should be rejected");
        assert!(
            error
                .to_string()
                .contains("rise_transition result must be non-negative")
        );
    }

    #[test]
    fn evaluate_output_edge_set_rejects_non_finite_arrival_results() {
        let input_timing = EdgeTimingSet::from_single(EdgeTiming {
            arrival: f64::MAX,
            transition: 0.1,
        });
        let delay_table = scalar_table("cell_rise", f64::MAX);
        let slew_table = scalar_table("rise_transition", 0.1);

        let error = evaluate_output_edge_set(
            &crate::liberty_proto::Library::default(),
            &delay_table,
            &slew_table,
            &input_timing,
            0.0,
            "overflow_arrival",
            StaTimingTableKind::CellRise,
            StaTimingTableKind::RiseTransition,
        )
        .expect_err("overflowed arrival outputs should be rejected");
        assert!(
            error
                .to_string()
                .contains("propagated arrival must be finite")
        );
    }

    #[test]
    fn sta_rejects_duplicate_library_cells_and_pins() {
        let duplicate_cells = crate::liberty_proto::Library {
            cells: vec![
                Cell {
                    name: "INV".to_string(),
                    ..Default::default()
                },
                Cell {
                    name: "INV".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let cell_error = match StaLibraryIndex::new(&duplicate_cells) {
            Ok(_) => panic!("duplicate cells should fail"),
            Err(error) => error,
        };
        assert!(
            cell_error
                .to_string()
                .contains("defines cell 'INV' more than once")
        );

        let duplicate_pins = crate::liberty_proto::Library {
            cells: vec![Cell {
                name: "INV".to_string(),
                pins: vec![
                    Pin {
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    Pin {
                        name: "A".to_string(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let pin_error = match StaLibraryIndex::new(&duplicate_pins) {
            Ok(_) => panic!("duplicate pins should fail"),
            Err(error) => error,
        };
        assert!(
            pin_error
                .to_string()
                .contains("defines pin 'A' more than once")
        );
    }
}
