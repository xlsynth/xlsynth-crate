// SPDX-License-Identifier: Apache-2.0

//! Structural register-stage partitioning and complete area accounting.

use crate::liberty_model::{Library, PinDirection};
use crate::netlist::normalized::{BitExpr, BitSource, NormalizedNetlistModule};
use crate::netlist::parse::{Net, NetlistModule, PortDirection};
use crate::netlist::sta::is_sequential_boundary_cell;
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Outcome of attempting to assign sequential instances to pipeline stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StagePartitionStatus {
    NoRegisters,
    Partitioned,
    NotPartitionable,
    Ambiguous,
}

/// Structural stage and area analysis for one mapped netlist module.
#[derive(Clone, Debug, PartialEq)]
pub struct RegisterStageAnalysis {
    pub status: StagePartitionStatus,
    pub register_indices: Vec<usize>,
    pub register_levels: Vec<Option<usize>>,
    pub sequential_area: f64,
    pub non_stage_combinational_area: f64,
    pub stage_areas: BTreeMap<usize, f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InstanceInfo {
    cell_idx: usize,
    sequential: bool,
}

#[derive(Clone, Debug, Default)]
struct CombinationalParticipation {
    inputs: BTreeSet<usize>,
    outputs: BTreeSet<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Origin {
    PrimaryInput,
    Register(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Sink {
    ModuleOutput,
    Register(usize),
}

/// Partitions registers into feed-forward stages and assigns each cell's area.
pub fn analyze_register_stages(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
) -> Result<RegisterStageAnalysis> {
    let normalized = NormalizedNetlistModule::new(module, nets, interner)?;
    let mut cell_by_name = HashMap::new();
    for (cell_idx, cell) in library.cells.iter().enumerate() {
        if cell_by_name.insert(cell.name.as_str(), cell_idx).is_some() {
            return Err(anyhow!(
                "library defines cell '{}' more than once; duplicate cell names are unsupported in stage analysis",
                cell.name
            ));
        }
    }

    let mut infos = Vec::with_capacity(normalized.instances.len());
    for instance in &normalized.instances {
        let cell_name = interner
            .resolve(instance.type_name)
            .ok_or_else(|| anyhow!("could not resolve cell type symbol"))?;
        let cell_idx = *cell_by_name.get(cell_name).ok_or_else(|| {
            anyhow!(
                "instance references unknown cell '{}' during stage analysis",
                cell_name
            )
        })?;
        infos.push(InstanceInfo {
            cell_idx,
            sequential: is_sequential_boundary_cell(&library.cells[cell_idx]),
        });
    }

    let bit_count = normalized.bit_count();
    let mut successors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); bit_count];
    let mut predecessors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); bit_count];
    for assign in &normalized.assigns {
        for (lhs, rhs) in assign.lhs_bits.iter().copied().zip(assign.rhs_bits.iter()) {
            if let BitExpr::Source(BitSource::Bit(source)) = rhs {
                add_edge(*source, lhs, &mut successors, &mut predecessors);
            }
        }
    }

    let mut register_outputs: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); infos.len()];
    let mut register_inputs: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); infos.len()];
    let mut combinational = vec![CombinationalParticipation::default(); infos.len()];
    for (inst_idx, instance) in normalized.instances.iter().enumerate() {
        let cell = &library.cells[infos[inst_idx].cell_idx];
        let connections = connection_map(instance, interner)?;
        if infos[inst_idx].sequential {
            for pin in &cell.pins {
                let Some(bits) = connections.get(library.resolve_string(&pin.name)) else {
                    continue;
                };
                if pin.direction == PinDirection::Output as i32 {
                    extend_bit_set(&mut register_outputs[inst_idx], bits);
                } else if pin.direction == PinDirection::Input as i32 && !pin.is_clocking_pin {
                    extend_bit_set(&mut register_inputs[inst_idx], bits);
                }
            }
            continue;
        }

        for output_pin in cell
            .pins
            .iter()
            .filter(|pin| pin.direction == PinDirection::Output as i32)
        {
            let Some(output_bits) = connections.get(library.resolve_string(&output_pin.name))
            else {
                continue;
            };
            for arc in output_pin
                .timing_arcs
                .iter()
                .filter(|arc| is_combinational_timing_type(arc.timing_type_str(library)))
            {
                for related_pin in library.resolve_string(&arc.related_pin).split_whitespace() {
                    let Some(input_bits) = connections.get(related_pin) else {
                        continue;
                    };
                    for input in input_bits.iter().filter_map(as_bit) {
                        combinational[inst_idx].inputs.insert(input);
                        for output in output_bits.iter().filter_map(as_bit) {
                            combinational[inst_idx].outputs.insert(output);
                            add_edge(input, output, &mut successors, &mut predecessors);
                        }
                    }
                }
            }
        }
    }

    let register_indices: Vec<usize> = infos
        .iter()
        .enumerate()
        .filter_map(|(idx, info)| info.sequential.then_some(idx))
        .collect();
    let sequential_area: f64 = register_indices
        .iter()
        .map(|idx| library.cells[infos[*idx].cell_idx].area)
        .fold(0.0, |sum, area| sum + area);
    if register_indices.is_empty() {
        let non_stage_combinational_area = infos
            .iter()
            .map(|info| library.cells[info.cell_idx].area)
            .fold(0.0, |sum, area| sum + area);
        return Ok(RegisterStageAnalysis {
            status: StagePartitionStatus::NoRegisters,
            register_indices,
            register_levels: vec![None; infos.len()],
            sequential_area,
            non_stage_combinational_area,
            stage_areas: BTreeMap::new(),
        });
    }

    let mut origins: Vec<BTreeSet<Origin>> = vec![BTreeSet::new(); bit_count];
    for port in normalized
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Input)
    {
        for bit in &port.bits {
            origins[*bit].insert(Origin::PrimaryInput);
        }
    }
    for reg_idx in &register_indices {
        for bit in &register_outputs[*reg_idx] {
            origins[*bit].insert(Origin::Register(*reg_idx));
        }
    }
    propagate_sets(&mut origins, &successors);

    let mut sinks: Vec<BTreeSet<Sink>> = vec![BTreeSet::new(); bit_count];
    for port in normalized
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
    {
        for bit in &port.bits {
            sinks[*bit].insert(Sink::ModuleOutput);
        }
    }
    for reg_idx in &register_indices {
        for bit in &register_inputs[*reg_idx] {
            sinks[*bit].insert(Sink::Register(*reg_idx));
        }
    }
    propagate_sets(&mut sinks, &predecessors);

    let mut register_edges = BTreeSet::new();
    for dst in &register_indices {
        for bit in &register_inputs[*dst] {
            for origin in &origins[*bit] {
                if let Origin::Register(src) = origin {
                    register_edges.insert((*src, *dst));
                }
            }
        }
    }
    log::debug!(
        "stage analysis identified {} register instances and {} register edges",
        register_indices.len(),
        register_edges.len()
    );
    for (source, sink) in &register_edges {
        log::trace!(
            "stage register edge {} '{}' -> {} '{}'",
            source,
            interner
                .resolve(normalized.instances[*source].instance_name)
                .unwrap_or("<unresolved>"),
            sink,
            interner
                .resolve(normalized.instances[*sink].instance_name)
                .unwrap_or("<unresolved>")
        );
    }

    let partition = assign_register_levels(&register_indices, &register_edges);
    let (status, register_levels) = match partition {
        LevelAssignment::Partitioned(levels) => (StagePartitionStatus::Partitioned, levels),
        LevelAssignment::NotPartitionable => (
            StagePartitionStatus::NotPartitionable,
            vec![None; infos.len()],
        ),
        LevelAssignment::Ambiguous => (StagePartitionStatus::Ambiguous, vec![None; infos.len()]),
    };

    let mut non_stage_combinational_area = 0.0;
    let mut stage_areas = BTreeMap::new();
    if status == StagePartitionStatus::Partitioned {
        for (src, dst) in &register_edges {
            let src_level = register_levels[*src].expect("partitioned source has a level");
            let dst_level = register_levels[*dst].expect("partitioned sink has a level");
            if dst_level == src_level + 1 {
                stage_areas.entry(src_level).or_insert(0.0);
            }
        }
    }
    for (inst_idx, info) in infos.iter().enumerate() {
        if info.sequential {
            continue;
        }
        let area = library.cells[info.cell_idx].area;
        let stage = if status == StagePartitionStatus::Partitioned {
            classify_combinational_stage(
                &combinational[inst_idx],
                &origins,
                &sinks,
                &register_levels,
            )
        } else {
            None
        };
        if let Some(stage) = stage {
            *stage_areas.entry(stage).or_insert(0.0) += area;
        } else {
            non_stage_combinational_area += area;
        }
    }

    Ok(RegisterStageAnalysis {
        status,
        register_indices,
        register_levels,
        sequential_area,
        non_stage_combinational_area,
        stage_areas,
    })
}

fn connection_map<'a>(
    instance: &'a crate::netlist::normalized::NormalizedInstance,
    interner: &StringInterner<StringBackend<SymbolU32>>,
) -> Result<HashMap<String, &'a [BitSource]>> {
    let mut map = HashMap::new();
    for connection in &instance.connections {
        let pin = interner
            .resolve(connection.port)
            .ok_or_else(|| anyhow!("could not resolve instance pin symbol"))?;
        if map
            .insert(pin.to_string(), connection.bits.as_slice())
            .is_some()
        {
            return Err(anyhow!(
                "instance connects pin '{}' more than once during stage analysis",
                pin
            ));
        }
    }
    Ok(map)
}

fn as_bit(source: &BitSource) -> Option<usize> {
    match source {
        BitSource::Bit(bit) => Some(*bit),
        BitSource::Literal(_) | BitSource::Unknown => None,
    }
}

fn extend_bit_set(target: &mut BTreeSet<usize>, bits: &[BitSource]) {
    target.extend(bits.iter().filter_map(as_bit));
}

fn add_edge(
    source: usize,
    sink: usize,
    successors: &mut [BTreeSet<usize>],
    predecessors: &mut [BTreeSet<usize>],
) {
    successors[source].insert(sink);
    predecessors[sink].insert(source);
}

fn is_combinational_timing_type(raw: &str) -> bool {
    matches!(
        raw,
        "" | "combinational" | "combinational_rise" | "combinational_fall"
    )
}

fn propagate_sets<T: Clone + Ord>(sets: &mut [BTreeSet<T>], successors: &[BTreeSet<usize>]) {
    let mut queue: VecDeque<usize> = sets
        .iter()
        .enumerate()
        .filter_map(|(idx, values)| (!values.is_empty()).then_some(idx))
        .collect();
    let mut queued = vec![false; sets.len()];
    for source in &queue {
        queued[*source] = true;
    }
    while let Some(source) = queue.pop_front() {
        queued[source] = false;
        let propagated = sets[source].clone();
        for sink in &successors[source] {
            let old_len = sets[*sink].len();
            sets[*sink].extend(propagated.iter().cloned());
            if sets[*sink].len() != old_len && !queued[*sink] {
                queued[*sink] = true;
                queue.push_back(*sink);
            }
        }
    }
}

fn classify_combinational_stage(
    participation: &CombinationalParticipation,
    origins: &[BTreeSet<Origin>],
    sinks: &[BTreeSet<Sink>],
    register_levels: &[Option<usize>],
) -> Option<usize> {
    let mut cell_origins = BTreeSet::new();
    for bit in &participation.inputs {
        cell_origins.extend(origins[*bit].iter().copied());
    }
    let mut cell_sinks = BTreeSet::new();
    for bit in &participation.outputs {
        cell_sinks.extend(sinks[*bit].iter().copied());
    }
    if cell_origins.contains(&Origin::PrimaryInput) || cell_sinks.contains(&Sink::ModuleOutput) {
        return None;
    }
    let source_registers: Vec<usize> = cell_origins
        .iter()
        .filter_map(|origin| match origin {
            Origin::Register(idx) => Some(*idx),
            Origin::PrimaryInput => None,
        })
        .collect();
    let sink_registers: Vec<usize> = cell_sinks
        .iter()
        .filter_map(|sink| match sink {
            Sink::Register(idx) => Some(*idx),
            Sink::ModuleOutput => None,
        })
        .collect();
    if source_registers.is_empty() || sink_registers.is_empty() {
        return None;
    }
    let mut stage = None;
    for source in &source_registers {
        let source_level = register_levels[*source]?;
        for sink in &sink_registers {
            let sink_level = register_levels[*sink]?;
            if sink_level != source_level + 1 {
                return None;
            }
            if stage.is_some_and(|current| current != source_level) {
                return None;
            }
            stage = Some(source_level);
        }
    }
    stage
}

enum LevelAssignment {
    Partitioned(Vec<Option<usize>>),
    NotPartitionable,
    Ambiguous,
}

/// Solves adjacent-stage constraints, considering only cycle-closing edges as
/// eligible feedback. Cyclic components with more than one valid layering are
/// deliberately reported as ambiguous.
fn assign_register_levels(
    register_indices: &[usize],
    raw_edges: &BTreeSet<(usize, usize)>,
) -> LevelAssignment {
    let local_by_raw: HashMap<usize, usize> = register_indices
        .iter()
        .enumerate()
        .map(|(local, raw)| (*raw, local))
        .collect();
    let count = register_indices.len();
    let mut edges = vec![BTreeSet::new(); count];
    let mut reverse_edges = vec![BTreeSet::new(); count];
    for (raw_source, raw_sink) in raw_edges {
        let source = local_by_raw[raw_source];
        let sink = local_by_raw[raw_sink];
        edges[source].insert(sink);
        reverse_edges[sink].insert(source);
    }
    let components = strongly_connected_components(&edges, &reverse_edges);
    let mut component_of = vec![0usize; count];
    for (component_idx, component) in components.iter().enumerate() {
        for node in component {
            component_of[*node] = component_idx;
        }
    }
    let mut component_edges = vec![BTreeSet::new(); components.len()];
    let mut component_indegree = vec![0usize; components.len()];
    for (source, successors) in edges.iter().enumerate() {
        for sink in successors {
            let source_component = component_of[source];
            let sink_component = component_of[*sink];
            if source_component != sink_component
                && component_edges[source_component].insert(sink_component)
            {
                component_indegree[sink_component] += 1;
            }
        }
    }
    let mut component_queue: VecDeque<usize> = component_indegree
        .iter()
        .enumerate()
        .filter_map(|(idx, indegree)| (*indegree == 0).then_some(idx))
        .collect();
    let mut component_order = Vec::new();
    while let Some(component) = component_queue.pop_front() {
        component_order.push(component);
        for successor in &component_edges[component] {
            component_indegree[*successor] -= 1;
            if component_indegree[*successor] == 0 {
                component_queue.push_back(*successor);
            }
        }
    }

    let mut levels = vec![None; count];
    for component_idx in component_order {
        let component = &components[component_idx];
        let cyclic = component.len() > 1 || edges[component[0]].contains(&component[0]);
        if !cyclic {
            let node = component[0];
            let required: BTreeSet<usize> = reverse_edges[node]
                .iter()
                .filter_map(|predecessor| levels[*predecessor].map(|level| level + 1))
                .collect();
            if required.len() > 1 {
                let predecessor_levels: Vec<(usize, Option<usize>)> = reverse_edges[node]
                    .iter()
                    .map(|predecessor| (register_indices[*predecessor], levels[*predecessor]))
                    .collect();
                log::debug!(
                    "register instance {} has conflicting required stage levels {:?} from {:?}",
                    register_indices[node],
                    required,
                    predecessor_levels
                );
                return LevelAssignment::NotPartitionable;
            }
            levels[node] = Some(required.iter().next().copied().unwrap_or(0));
            continue;
        }

        let component_nodes: BTreeSet<usize> = component.iter().copied().collect();
        let mut candidates: BTreeSet<Vec<usize>> = BTreeSet::new();
        for root in component {
            let Some(relative) = relative_cycle_levels(*root, &component_nodes, &edges) else {
                continue;
            };
            let mut offset: Option<isize> = None;
            let mut valid = true;
            for node in component {
                for predecessor in &reverse_edges[*node] {
                    if component_nodes.contains(predecessor) {
                        continue;
                    }
                    let predecessor_level =
                        levels[*predecessor].expect("predecessor SCC was assigned first");
                    let required = predecessor_level as isize + 1 - relative[*node] as isize;
                    if offset.is_some_and(|existing| existing != required) {
                        valid = false;
                    }
                    offset = Some(required);
                }
            }
            let offset = offset.unwrap_or(0);
            if !valid
                || component
                    .iter()
                    .any(|node| relative[*node] as isize + offset < 0)
            {
                continue;
            }
            let absolute: Vec<usize> = component
                .iter()
                .map(|node| (relative[*node] as isize + offset) as usize)
                .collect();
            candidates.insert(absolute);
        }
        if candidates.is_empty() {
            return LevelAssignment::NotPartitionable;
        }
        if candidates.len() > 1 {
            return LevelAssignment::Ambiguous;
        }
        let selected = candidates.into_iter().next().expect("one candidate");
        for (node, level) in component.iter().zip(selected) {
            levels[*node] = Some(level);
        }
    }
    for (source, successors) in edges.iter().enumerate() {
        for sink in successors {
            if component_of[source] != component_of[*sink]
                && levels[*sink] != levels[source].map(|level| level + 1)
            {
                log::debug!(
                    "register edge {} -> {} violates adjacent stage levels {:?} -> {:?}",
                    register_indices[source],
                    register_indices[*sink],
                    levels[source],
                    levels[*sink]
                );
                return LevelAssignment::NotPartitionable;
            }
        }
    }
    let mut raw_levels = vec![None; register_indices.iter().max().copied().unwrap_or(0) + 1];
    for (local, raw) in register_indices.iter().enumerate() {
        raw_levels[*raw] = levels[local];
    }
    LevelAssignment::Partitioned(raw_levels)
}

fn relative_cycle_levels(
    root: usize,
    component: &BTreeSet<usize>,
    edges: &[BTreeSet<usize>],
) -> Option<Vec<usize>> {
    let mut levels = vec![usize::MAX; edges.len()];
    levels[root] = 0;
    let mut queue = VecDeque::from([root]);
    while let Some(source) = queue.pop_front() {
        for sink in &edges[source] {
            if component.contains(sink) && levels[*sink] == usize::MAX {
                levels[*sink] = levels[source] + 1;
                queue.push_back(*sink);
            }
        }
    }
    if component.iter().any(|node| levels[*node] == usize::MAX) {
        return None;
    }
    for source in component {
        for sink in edges[*source]
            .iter()
            .filter(|sink| component.contains(sink))
        {
            if levels[*sink] > levels[*source] + 1 {
                return None;
            }
        }
    }
    Some(levels)
}

fn strongly_connected_components(
    edges: &[BTreeSet<usize>],
    reverse_edges: &[BTreeSet<usize>],
) -> Vec<Vec<usize>> {
    fn visit(node: usize, edges: &[BTreeSet<usize>], seen: &mut [bool], order: &mut Vec<usize>) {
        if seen[node] {
            return;
        }
        seen[node] = true;
        for successor in &edges[node] {
            visit(*successor, edges, seen, order);
        }
        order.push(node);
    }
    fn collect(
        node: usize,
        edges: &[BTreeSet<usize>],
        seen: &mut [bool],
        component: &mut Vec<usize>,
    ) {
        if seen[node] {
            return;
        }
        seen[node] = true;
        component.push(node);
        for successor in &edges[node] {
            collect(*successor, edges, seen, component);
        }
    }
    let mut order = Vec::new();
    let mut seen = vec![false; edges.len()];
    for node in 0..edges.len() {
        visit(node, edges, &mut seen, &mut order);
    }
    seen.fill(false);
    let mut components = Vec::new();
    for node in order.into_iter().rev() {
        if seen[node] {
            continue;
        }
        let mut component = Vec::new();
        collect(node, reverse_edges, &mut seen, &mut component);
        component.sort_unstable();
        components.push(component);
    }
    components
}
