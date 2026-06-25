// SPDX-License-Identifier: Apache-2.0

//! Incremental register-boundary timing for equivalent-cell substitutions.

use super::*;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// Result of one incremental equivalent-cell substitution evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct IncrementalCellSubstitutionEvaluation {
    pub report: StaReport,
    pub recomputed_instances: usize,
}

#[derive(Clone)]
struct IncrementalCheckpoint {
    instance_index: usize,
    cell_index: usize,
    cell_name: String,
    output_pin_order: Vec<usize>,
    changed_loads: Vec<(usize, EdgeLoadCapacitance)>,
    changed_timings: Vec<(usize, Option<SignalTimingSet>)>,
    changed_instance_diagnostics: Vec<(usize, TimingQueryDiagnosticCounts)>,
    register_input_timings: Vec<Option<RegisterCaptureTimingCandidate>>,
    register_diagnostics: Vec<TimingQueryDiagnosticCounts>,
}

/// Reusable STA state for repeated equivalent-cell substitution trials.
///
/// Connectivity and source normalization are prepared once. A trial updates
/// input-pin loads, recomputes only the affected forward cone, rebuilds capture
/// endpoints, and then restores the prior state unless the change is committed.
pub struct IncrementalRegisterBoundarySta<'a> {
    library: &'a crate::liberty_proto::Library,
    lib: StaLibraryIndex<'a>,
    launch_register_instances: HashSet<usize>,
    nets_len: usize,
    bit_net_indices: Vec<usize>,
    bit_names: Vec<String>,
    instance_names: Vec<String>,
    instance_cell_indices: Vec<usize>,
    instance_cell_names: Vec<String>,
    instance_pin_sources: Vec<HashMap<String, Vec<PinBitSource>>>,
    instance_known_pin_values: Vec<HashMap<String, bool>>,
    instance_timing_related_input_pins: Vec<HashSet<String>>,
    instance_output_pin_orders: Vec<Vec<usize>>,
    instance_is_sequential: Vec<bool>,
    bit_drivers: Vec<Vec<NetEndpoint>>,
    resolved_timing_sources: Vec<ResolvedTimingSource>,
    module_output_bits: Vec<usize>,
    successors: Vec<Vec<usize>>,
    topo_order: Vec<usize>,
    cell_levels: usize,
    bit_load_capacitance: Vec<EdgeLoadCapacitance>,
    literal_source_timing_set: SignalTimingSet,
    bit_timing_sets: Vec<Option<SignalTimingSet>>,
    instance_diagnostics: Vec<TimingQueryDiagnosticCounts>,
    register_input_timings: Vec<Option<RegisterCaptureTimingCandidate>>,
    register_diagnostics: Vec<TimingQueryDiagnosticCounts>,
    validated_timing_tables: HashSet<*const TimingTable>,
}

impl<'a> IncrementalRegisterBoundarySta<'a> {
    /// Prepares reusable register-boundary STA state and performs one full
    /// pass.
    pub fn new(
        module: &NetlistModule,
        nets: &[Net],
        interner: &StringInterner<StringBackend<SymbolU32>>,
        library: &'a LibraryWithTimingData,
        options: StaOptions,
        launch_primary_inputs: bool,
        launch_register_instances: &[usize],
    ) -> Result<Self> {
        validate_options(options)?;
        let library = library.as_proto();
        let lib = StaLibraryIndex::new(library)?;
        let normalized = NormalizedNetlistModule::new(module, nets, interner)?;
        let instance_count = normalized.instances.len();
        let bit_count = normalized.bit_count();
        let assign_analysis = analyze_assign_sources(&normalized, nets, interner)?;
        let assign_sources = assign_analysis.bit_sources;
        let resolved_timing_sources =
            resolve_timing_sources(assign_sources.as_slice(), &normalized, nets, interner)?;
        let launch_register_instances: HashSet<usize> =
            launch_register_instances.iter().copied().collect();

        let mut instance_names = Vec::with_capacity(instance_count);
        let mut instance_cell_indices = Vec::with_capacity(instance_count);
        let mut instance_cell_names = Vec::with_capacity(instance_count);
        let mut instance_pin_sources = Vec::with_capacity(instance_count);
        let mut instance_known_pin_values = Vec::with_capacity(instance_count);
        let mut instance_timing_related_input_pins = Vec::with_capacity(instance_count);
        let mut instance_output_pin_orders = Vec::with_capacity(instance_count);
        let mut instance_is_sequential = Vec::with_capacity(instance_count);
        let mut bit_drivers: Vec<Vec<NetEndpoint>> = vec![Vec::new(); bit_count];
        let mut bit_loads: Vec<Vec<NetEndpoint>> = vec![Vec::new(); bit_count];
        let mut bit_constant_values = vec![None; bit_count];

        for (inst_idx, inst) in normalized.instances.iter().enumerate() {
            let instance_name = resolve_symbol(interner, inst.instance_name, "instance name")?;
            let cell_name = resolve_symbol(interner, inst.type_name, "cell type")?;
            let cell_idx = lib.cell_index(cell_name.as_str()).ok_or_else(|| {
                anyhow!(
                    "instance '{}' references unknown cell '{}'",
                    instance_name,
                    cell_name
                )
            })?;
            let is_sequential = is_sequential_boundary_cell(&library.cells[cell_idx]);
            let timing_related_input_pins =
                timing_related_input_pins(&lib, cell_idx, is_sequential)?;
            let mut pin_sources = HashMap::new();
            let mut known_pin_values = HashMap::new();
            for connection in &inst.connections {
                let pin_name = resolve_symbol(interner, connection.port, "pin name")?;
                if pin_sources.contains_key(pin_name.as_str()) {
                    return Err(anyhow!(
                        "instance '{}' of '{}' connects pin '{}' more than once",
                        instance_name,
                        cell_name,
                        pin_name
                    ));
                }
                let pin = lib.pin(cell_idx, pin_name.as_str()).ok_or_else(|| {
                    anyhow!(
                        "instance '{}' of '{}' references unknown pin '{}'",
                        instance_name,
                        cell_name,
                        pin_name
                    )
                })?;
                let pin_bit_sources = if pin.direction == PinDirection::Input as i32 {
                    connection
                        .bits
                        .iter()
                        .copied()
                        .map(|source| {
                            canonicalize_pin_bit_source(source, resolved_timing_sources.as_slice())
                        })
                        .collect()
                } else {
                    connection.bits.clone()
                };
                if pin_bit_sources.len() > 1 {
                    return Err(anyhow!(
                        "instance '{}' pin '{}.{}' connects {} bits; incremental STA requires scalar cell pins",
                        instance_name,
                        cell_name,
                        pin_name,
                        pin_bit_sources.len()
                    ));
                }
                if pin.direction == PinDirection::Output as i32
                    && !matches!(pin_bit_sources.as_slice(), [PinBitSource::Bit(_)])
                {
                    return Err(anyhow!(
                        "instance '{}' output pin '{}.{}' is not connected to a net bit",
                        instance_name,
                        cell_name,
                        pin_name
                    ));
                }
                for source in &pin_bit_sources {
                    match pin.direction {
                        direction if direction == PinDirection::Output as i32 => {
                            let PinBitSource::Bit(bit_idx) = source else {
                                unreachable!("output binding was validated above");
                            };
                            bit_drivers[*bit_idx].push(NetEndpoint {
                                inst_idx,
                                pin_name: pin_name.clone(),
                            });
                            if pin.timing_arcs.is_empty() {
                                bit_constant_values[*bit_idx] =
                                    constant_output_function_value(cell_name.as_str(), pin)?;
                            }
                        }
                        direction if direction == PinDirection::Input as i32 => match source {
                            PinBitSource::Bit(bit_idx) => {
                                bit_loads[*bit_idx].push(NetEndpoint {
                                    inst_idx,
                                    pin_name: pin_name.clone(),
                                });
                            }
                            PinBitSource::Literal(value) => {
                                known_pin_values.insert(pin_name.clone(), *value);
                            }
                            PinBitSource::Unknown => {}
                        },
                        _ => {
                            return Err(anyhow!(
                                "instance '{}' pin '{}.{}' has unsupported direction {}",
                                instance_name,
                                cell_name,
                                pin_name,
                                pin.direction
                            ));
                        }
                    }
                }
                pin_sources.insert(pin_name, pin_bit_sources);
            }
            let output_pin_order = output_pin_order(&lib, cell_idx, is_sequential, &pin_sources)?;
            instance_names.push(instance_name);
            instance_cell_indices.push(cell_idx);
            instance_cell_names.push(cell_name);
            instance_pin_sources.push(pin_sources);
            instance_known_pin_values.push(known_pin_values);
            instance_timing_related_input_pins.push(timing_related_input_pins);
            instance_output_pin_orders.push(output_pin_order);
            instance_is_sequential.push(is_sequential);
        }

        for (inst_idx, (pin_sources, known_pin_values)) in instance_pin_sources
            .iter()
            .zip(instance_known_pin_values.iter_mut())
            .enumerate()
        {
            for (pin_name, sources) in pin_sources {
                let pin = lib
                    .pin(instance_cell_indices[inst_idx], pin_name.as_str())
                    .expect("instance pins were validated above");
                if pin.direction != PinDirection::Input as i32 {
                    continue;
                }
                let [PinBitSource::Bit(bit_idx)] = sources.as_slice() else {
                    continue;
                };
                if let Some(value) = bit_constant_values[*bit_idx] {
                    known_pin_values.insert(pin_name.clone(), value);
                }
            }
        }

        let mut module_output_bits = Vec::new();
        let mut has_module_output = vec![false; bit_count];
        let mut is_module_input = vec![false; bit_count];
        for port in &normalized.ports {
            match port.direction {
                PortDirection::Input => {
                    for bit_idx in &port.bits {
                        is_module_input[*bit_idx] = true;
                    }
                }
                PortDirection::Output => {
                    for bit_idx in &port.bits {
                        if !has_module_output[*bit_idx] {
                            has_module_output[*bit_idx] = true;
                            module_output_bits.push(*bit_idx);
                        }
                    }
                }
                PortDirection::Inout => {
                    return Err(anyhow!(
                        "incremental STA does not support inout module ports"
                    ));
                }
            }
        }
        module_output_bits.sort_unstable();
        let mut has_resolved_module_output = vec![false; bit_count];
        for bit_idx in &module_output_bits {
            if let ResolvedTimingSource::Bit(source_bit_idx) = resolved_timing_sources[*bit_idx] {
                has_resolved_module_output[source_bit_idx] = true;
            }
        }
        reject_live_unsupported_assigns(
            assign_analysis.unsupported_sources.as_slice(),
            assign_sources.as_slice(),
            bit_loads.as_slice(),
            has_module_output.as_slice(),
            &normalized,
            nets,
            interner,
        )?;

        validate_single_drivers(
            &normalized,
            nets,
            interner,
            assign_sources.as_slice(),
            is_module_input.as_slice(),
            bit_drivers.as_slice(),
        )?;
        let (successors, topo_order, instance_levels) = build_instance_graph(
            instance_count,
            bit_drivers.as_slice(),
            bit_loads.as_slice(),
            instance_timing_related_input_pins.as_slice(),
        )?;
        let cell_levels = module_output_bits
            .iter()
            .filter_map(|bit_idx| match resolved_timing_sources[*bit_idx] {
                ResolvedTimingSource::Bit(source_bit_idx) => bit_drivers[source_bit_idx]
                    .first()
                    .map(|driver| instance_levels[driver.inst_idx]),
                ResolvedTimingSource::Literal(_) | ResolvedTimingSource::Unknown => None,
            })
            .max()
            .unwrap_or(0);

        let mut bit_load_capacitance = vec![EdgeLoadCapacitance::default(); bit_count];
        for (bit_idx, loads) in bit_loads.iter().enumerate() {
            for load in loads {
                let cell_name = &instance_cell_names[load.inst_idx];
                let pin = lib
                    .pin(instance_cell_indices[load.inst_idx], load.pin_name.as_str())
                    .expect("load pin was validated above");
                let capacitance = effective_input_capacitance_by_edge(
                    pin,
                    &format!("load pin '{}.{}'", cell_name, load.pin_name),
                )?;
                bit_load_capacitance[bit_idx].rise += capacitance.rise;
                bit_load_capacitance[bit_idx].fall += capacitance.fall;
            }
        }
        for bit_idx in &module_output_bits {
            if let ResolvedTimingSource::Bit(source_bit_idx) = resolved_timing_sources[*bit_idx] {
                bit_load_capacitance[source_bit_idx].rise += options.module_output_load;
                bit_load_capacitance[source_bit_idx].fall += options.module_output_load;
            }
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
        let source_timing_set = SignalTimingSet::from_primary_input_launch(source_timing);
        let literal_source_timing_set = SignalTimingSet::from_single(SignalTiming {
            rise: EdgeTiming {
                arrival: 0.0,
                transition: 0.0,
            },
            fall: EdgeTiming {
                arrival: 0.0,
                transition: 0.0,
            },
        });
        let mut bit_timing_sets = vec![None; bit_count];
        for bit_idx in 0..bit_count {
            if !bit_drivers[bit_idx].is_empty() {
                continue;
            }
            match resolved_timing_sources[bit_idx] {
                ResolvedTimingSource::Literal(_) | ResolvedTimingSource::Unknown => continue,
                ResolvedTimingSource::Bit(source_bit_idx) if source_bit_idx != bit_idx => continue,
                ResolvedTimingSource::Bit(_) => {}
            }
            if is_module_input[bit_idx] && launch_primary_inputs {
                bit_timing_sets[bit_idx] = Some(source_timing_set.clone());
            } else if (!bit_loads[bit_idx].is_empty() || has_resolved_module_output[bit_idx])
                && !is_module_input[bit_idx]
            {
                // Floating sources are allowed in register-boundary mode and
                // simply do not contribute timing.
            }
        }

        let bit_net_indices = (0..bit_count)
            .map(|bit_idx| normalized.bit(bit_idx).net.0)
            .collect();
        let bit_names = (0..bit_count)
            .map(|bit_idx| normalized.render_bit(bit_idx, nets, interner))
            .collect();
        let mut result = Self {
            library,
            lib,
            launch_register_instances,
            nets_len: nets.len(),
            bit_net_indices,
            bit_names,
            instance_names,
            instance_cell_indices,
            instance_cell_names,
            instance_pin_sources,
            instance_known_pin_values,
            instance_timing_related_input_pins,
            instance_output_pin_orders,
            instance_is_sequential,
            bit_drivers,
            resolved_timing_sources,
            module_output_bits,
            successors,
            topo_order,
            cell_levels,
            bit_load_capacitance,
            literal_source_timing_set,
            bit_timing_sets,
            instance_diagnostics: vec![TimingQueryDiagnosticCounts::default(); instance_count],
            register_input_timings: vec![None; instance_count],
            register_diagnostics: vec![TimingQueryDiagnosticCounts::default(); instance_count],
            validated_timing_tables: HashSet::new(),
        };
        let topo_order = result.topo_order.clone();
        for inst_idx in topo_order {
            result.recompute_instance(inst_idx)?;
        }
        result.refresh_register_endpoints()?;
        Ok(result)
    }

    /// Returns the report for the currently committed cell choices.
    pub fn report(&self) -> Result<StaReport> {
        self.build_report()
    }

    /// Evaluates a substitution and rolls the timing state back afterward.
    pub fn evaluate_cell_substitution(
        &mut self,
        instance_index: usize,
        new_cell_name: &str,
    ) -> Result<IncrementalCellSubstitutionEvaluation> {
        self.substitute_cell(instance_index, new_cell_name, false)
    }

    /// Applies a substitution permanently to the reusable timing state.
    pub fn commit_cell_substitution(
        &mut self,
        instance_index: usize,
        new_cell_name: &str,
    ) -> Result<IncrementalCellSubstitutionEvaluation> {
        self.substitute_cell(instance_index, new_cell_name, true)
    }
}

fn validate_options(options: StaOptions) -> Result<()> {
    if !options.primary_input_transition.is_finite() || options.primary_input_transition < 0.0 {
        return Err(anyhow!(
            "primary_input_transition must be non-negative and finite; got {}",
            options.primary_input_transition
        ));
    }
    if !options.module_output_load.is_finite() || options.module_output_load < 0.0 {
        return Err(anyhow!(
            "module_output_load must be non-negative and finite; got {}",
            options.module_output_load
        ));
    }
    Ok(())
}

fn timing_related_input_pins(
    lib: &StaLibraryIndex<'_>,
    cell_idx: usize,
    is_sequential: bool,
) -> Result<HashSet<String>> {
    if is_sequential {
        return Ok(HashSet::new());
    }
    let cell = &lib.library.cells[cell_idx];
    let mut result = HashSet::new();
    for output_pin in cell
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
                        "cell '{}' output pin '{}' references unknown timing pin '{}'",
                        cell.name,
                        output_pin.name,
                        related_pin_name
                    )
                })?;
                if related_pin.direction == PinDirection::Input as i32 {
                    result.insert(related_pin_name.to_string());
                } else if related_pin.direction != PinDirection::Output as i32 {
                    return Err(anyhow!(
                        "cell '{}' timing-related pin '{}' has unsupported direction {}",
                        cell.name,
                        related_pin_name,
                        related_pin.direction
                    ));
                }
            }
        }
    }
    Ok(result)
}

fn output_pin_order(
    lib: &StaLibraryIndex<'_>,
    cell_idx: usize,
    is_sequential: bool,
    pin_sources: &HashMap<String, Vec<PinBitSource>>,
) -> Result<Vec<usize>> {
    if is_sequential {
        Ok(lib.library.cells[cell_idx]
            .pins
            .iter()
            .enumerate()
            .filter(|(_, pin)| {
                pin.direction == PinDirection::Output as i32
                    && pin_sources.contains_key(pin.name.as_str())
            })
            .map(|(pin_idx, _)| pin_idx)
            .collect())
    } else {
        combinational_output_pin_evaluation_order(lib, cell_idx, pin_sources)
    }
}

fn validate_single_drivers(
    normalized: &NormalizedNetlistModule<'_>,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    assign_sources: &[Option<BitAssignSource>],
    is_module_input: &[bool],
    bit_drivers: &[Vec<NetEndpoint>],
) -> Result<()> {
    for (bit_idx, drivers) in bit_drivers.iter().enumerate() {
        let has_assign_source = assign_sources[bit_idx].is_some();
        if is_module_input[bit_idx] && (!drivers.is_empty() || has_assign_source) {
            return Err(anyhow!(
                "module input bit '{}' also has an internal driver",
                normalized.render_bit(bit_idx, nets, interner)
            ));
        }
        if drivers.len() > 1 || (!drivers.is_empty() && has_assign_source) {
            return Err(anyhow!(
                "net bit '{}' has {} drivers",
                normalized.render_bit(bit_idx, nets, interner),
                drivers.len() + usize::from(has_assign_source)
            ));
        }
    }
    Ok(())
}

fn build_instance_graph(
    instance_count: usize,
    bit_drivers: &[Vec<NetEndpoint>],
    bit_loads: &[Vec<NetEndpoint>],
    timing_related_input_pins: &[HashSet<String>],
) -> Result<(Vec<Vec<usize>>, Vec<usize>, Vec<usize>)> {
    let mut successors = vec![Vec::new(); instance_count];
    let mut indegree = vec![0usize; instance_count];
    let mut seen_edges = HashSet::new();
    for (bit_idx, loads) in bit_loads.iter().enumerate() {
        let Some(driver) = bit_drivers[bit_idx].first() else {
            continue;
        };
        for load in loads {
            if !timing_related_input_pins[load.inst_idx].contains(&load.pin_name)
                || load.inst_idx == driver.inst_idx
            {
                continue;
            }
            if seen_edges.insert((driver.inst_idx, load.inst_idx)) {
                successors[driver.inst_idx].push(load.inst_idx);
                indegree[load.inst_idx] += 1;
            }
        }
    }
    for next in &mut successors {
        next.sort_unstable();
    }
    let mut queue = VecDeque::new();
    let mut levels = vec![1usize; instance_count];
    for (inst_idx, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(inst_idx);
        }
    }
    let mut order = Vec::with_capacity(instance_count);
    while let Some(inst_idx) = queue.pop_front() {
        order.push(inst_idx);
        for successor in &successors[inst_idx] {
            levels[*successor] = levels[*successor].max(levels[inst_idx] + 1);
            indegree[*successor] -= 1;
            if indegree[*successor] == 0 {
                queue.push_back(*successor);
            }
        }
    }
    if order.len() != instance_count {
        return Err(anyhow!(
            "combinational cycle detected while preparing incremental STA"
        ));
    }
    Ok((successors, order, levels))
}

impl IncrementalRegisterBoundarySta<'_> {
    fn substitute_cell(
        &mut self,
        instance_index: usize,
        new_cell_name: &str,
        commit: bool,
    ) -> Result<IncrementalCellSubstitutionEvaluation> {
        let new_cell_index = self
            .lib
            .cell_index(new_cell_name)
            .ok_or_else(|| anyhow!("library does not contain cell '{}'", new_cell_name))?;
        let new_output_pin_order =
            self.validate_substitution(instance_index, new_cell_index, new_cell_name)?;
        let changed_loads = self.changed_input_loads(instance_index, new_cell_index)?;
        let dirty_instances = self.dirty_instance_closure(instance_index, &changed_loads);
        let checkpoint = self.checkpoint(
            instance_index,
            changed_loads.as_slice(),
            dirty_instances.as_slice(),
        );
        let result = (|| {
            self.instance_cell_indices[instance_index] = new_cell_index;
            self.instance_cell_names[instance_index] = new_cell_name.to_string();
            self.instance_output_pin_orders[instance_index] = new_output_pin_order;
            for (bit_idx, new_load) in &changed_loads {
                self.bit_load_capacitance[*bit_idx] = *new_load;
            }
            for inst_idx in &dirty_instances {
                self.recompute_instance(*inst_idx)?;
            }
            self.refresh_register_endpoints()?;
            Ok(IncrementalCellSubstitutionEvaluation {
                report: self.build_report()?,
                recomputed_instances: dirty_instances.len(),
            })
        })();
        if result.is_err() || !commit {
            self.restore(checkpoint);
        }
        result
    }

    fn validate_substitution(
        &self,
        instance_index: usize,
        new_cell_index: usize,
        new_cell_name: &str,
    ) -> Result<Vec<usize>> {
        if instance_index >= self.instance_cell_indices.len() {
            return Err(anyhow!(
                "instance index {} is out of range for {} instances",
                instance_index,
                self.instance_cell_indices.len()
            ));
        }
        let old_cell_index = self.instance_cell_indices[instance_index];
        let old_cell = &self.library.cells[old_cell_index];
        let new_cell = &self.library.cells[new_cell_index];
        let old_is_sequential = self.instance_is_sequential[instance_index];
        let new_is_sequential = is_sequential_boundary_cell(new_cell);
        if old_is_sequential != new_is_sequential {
            return Err(anyhow!(
                "incremental substitution changes sequential-boundary classification: '{}' -> '{}'",
                old_cell.name,
                new_cell_name
            ));
        }
        for pin_name in self.instance_pin_sources[instance_index].keys() {
            let old_pin = self.lib.pin(old_cell_index, pin_name).ok_or_else(|| {
                anyhow!(
                    "old cell '{}' has no connected pin '{}'",
                    old_cell.name,
                    pin_name
                )
            })?;
            let new_pin = self.lib.pin(new_cell_index, pin_name).ok_or_else(|| {
                anyhow!(
                    "new cell '{}' has no connected pin '{}'",
                    new_cell_name,
                    pin_name
                )
            })?;
            if old_pin.direction != new_pin.direction {
                return Err(anyhow!(
                    "incremental substitution changes direction of pin '{}': '{}' -> '{}'",
                    pin_name,
                    old_cell.name,
                    new_cell_name
                ));
            }
            if old_pin.direction == PinDirection::Output as i32
                && old_pin.timing_arcs.is_empty()
                && new_pin.timing_arcs.is_empty()
                && constant_output_function_value(old_cell.name.as_str(), old_pin)?
                    != constant_output_function_value(new_cell.name.as_str(), new_pin)?
            {
                return Err(anyhow!(
                    "incremental substitution changes constant output '{}.{}'",
                    new_cell_name,
                    pin_name
                ));
            }
        }
        let new_related = timing_related_input_pins(&self.lib, new_cell_index, new_is_sequential)?;
        if new_related != self.instance_timing_related_input_pins[instance_index] {
            return Err(anyhow!(
                "incremental substitution changes timing dependency pins: '{}' -> '{}'",
                old_cell.name,
                new_cell_name
            ));
        }
        output_pin_order(
            &self.lib,
            new_cell_index,
            new_is_sequential,
            &self.instance_pin_sources[instance_index],
        )
    }

    fn changed_input_loads(
        &self,
        instance_index: usize,
        new_cell_index: usize,
    ) -> Result<Vec<(usize, EdgeLoadCapacitance)>> {
        let old_cell_index = self.instance_cell_indices[instance_index];
        let mut changed = HashMap::<usize, EdgeLoadCapacitance>::new();
        for (pin_name, sources) in &self.instance_pin_sources[instance_index] {
            let old_pin = self
                .lib
                .pin(old_cell_index, pin_name)
                .expect("old cell pins were validated during preparation");
            if old_pin.direction != PinDirection::Input as i32 {
                continue;
            }
            let new_pin = self
                .lib
                .pin(new_cell_index, pin_name)
                .expect("new cell pins were validated before load calculation");
            let old_cap = effective_input_capacitance_by_edge(
                old_pin,
                &format!(
                    "load pin '{}.{}'",
                    self.instance_cell_names[instance_index], pin_name
                ),
            )?;
            let new_cap = effective_input_capacitance_by_edge(
                new_pin,
                &format!(
                    "load pin '{}.{}'",
                    self.library.cells[new_cell_index].name, pin_name
                ),
            )?;
            for source in sources {
                let PinBitSource::Bit(bit_idx) = source else {
                    continue;
                };
                let load = changed
                    .entry(*bit_idx)
                    .or_insert(self.bit_load_capacitance[*bit_idx]);
                load.rise += new_cap.rise - old_cap.rise;
                load.fall += new_cap.fall - old_cap.fall;
                if load.rise < -1e-12 || load.fall < -1e-12 {
                    return Err(anyhow!(
                        "incremental load update produced negative capacitance on '{}'",
                        self.bit_names[*bit_idx]
                    ));
                }
                load.rise = load.rise.max(0.0);
                load.fall = load.fall.max(0.0);
            }
        }
        let mut result: Vec<(usize, EdgeLoadCapacitance)> = changed.into_iter().collect();
        result.sort_by_key(|(bit_idx, _)| *bit_idx);
        Ok(result)
    }

    fn dirty_instance_closure(
        &self,
        instance_index: usize,
        changed_loads: &[(usize, EdgeLoadCapacitance)],
    ) -> Vec<usize> {
        let mut dirty = BTreeSet::new();
        let mut queue = VecDeque::new();
        dirty.insert(instance_index);
        queue.push_back(instance_index);
        for (bit_idx, _) in changed_loads {
            if let Some(driver) = self.bit_drivers[*bit_idx].first()
                && dirty.insert(driver.inst_idx)
            {
                queue.push_back(driver.inst_idx);
            }
        }
        while let Some(inst_idx) = queue.pop_front() {
            for successor in &self.successors[inst_idx] {
                if dirty.insert(*successor) {
                    queue.push_back(*successor);
                }
            }
        }
        self.topo_order
            .iter()
            .copied()
            .filter(|inst_idx| dirty.contains(inst_idx))
            .collect()
    }

    fn checkpoint(
        &self,
        instance_index: usize,
        changed_loads: &[(usize, EdgeLoadCapacitance)],
        dirty_instances: &[usize],
    ) -> IncrementalCheckpoint {
        let changed_loads = changed_loads
            .iter()
            .map(|(bit_idx, _)| (*bit_idx, self.bit_load_capacitance[*bit_idx]))
            .collect();
        let mut output_bits = BTreeSet::new();
        for inst_idx in dirty_instances {
            output_bits.extend(self.connected_output_bits(*inst_idx));
        }
        let changed_timings = output_bits
            .into_iter()
            .map(|bit_idx| (bit_idx, self.bit_timing_sets[bit_idx].clone()))
            .collect();
        let changed_instance_diagnostics = dirty_instances
            .iter()
            .map(|inst_idx| (*inst_idx, self.instance_diagnostics[*inst_idx]))
            .collect();
        IncrementalCheckpoint {
            instance_index,
            cell_index: self.instance_cell_indices[instance_index],
            cell_name: self.instance_cell_names[instance_index].clone(),
            output_pin_order: self.instance_output_pin_orders[instance_index].clone(),
            changed_loads,
            changed_timings,
            changed_instance_diagnostics,
            register_input_timings: self.register_input_timings.clone(),
            register_diagnostics: self.register_diagnostics.clone(),
        }
    }

    fn restore(&mut self, checkpoint: IncrementalCheckpoint) {
        self.instance_cell_indices[checkpoint.instance_index] = checkpoint.cell_index;
        self.instance_cell_names[checkpoint.instance_index] = checkpoint.cell_name;
        self.instance_output_pin_orders[checkpoint.instance_index] = checkpoint.output_pin_order;
        for (bit_idx, load) in checkpoint.changed_loads {
            self.bit_load_capacitance[bit_idx] = load;
        }
        for (bit_idx, timing) in checkpoint.changed_timings {
            self.bit_timing_sets[bit_idx] = timing;
        }
        for (inst_idx, diagnostics) in checkpoint.changed_instance_diagnostics {
            self.instance_diagnostics[inst_idx] = diagnostics;
        }
        self.register_input_timings = checkpoint.register_input_timings;
        self.register_diagnostics = checkpoint.register_diagnostics;
    }

    fn connected_output_bits(&self, instance_index: usize) -> Vec<usize> {
        let cell_idx = self.instance_cell_indices[instance_index];
        let mut result = Vec::new();
        for pin in self.library.cells[cell_idx]
            .pins
            .iter()
            .filter(|pin| pin.direction == PinDirection::Output as i32)
        {
            let Some(sources) = self.instance_pin_sources[instance_index].get(&pin.name) else {
                continue;
            };
            for source in sources {
                if let PinBitSource::Bit(bit_idx) = source {
                    result.push(*bit_idx);
                }
            }
        }
        result.sort_unstable();
        result.dedup();
        result
    }

    fn recompute_instance(&mut self, inst_idx: usize) -> Result<()> {
        let cell_idx = self.instance_cell_indices[inst_idx];
        let cell_name = self.instance_cell_names[inst_idx].clone();
        let instance_name = self.instance_names[inst_idx].clone();
        let pin_sources = self.instance_pin_sources[inst_idx].clone();
        let known_pin_values = self.instance_known_pin_values[inst_idx].clone();
        let output_pin_order = self.instance_output_pin_orders[inst_idx].clone();
        let is_sequential = self.instance_is_sequential[inst_idx];
        let mut diagnostics = TimingQueryDiagnosticCounts::default();
        let mut local_output_timing_sets = HashMap::<String, SignalTimingSet>::new();
        let mut output_updates = HashMap::<usize, SignalTimingSet>::new();

        for bit_idx in self.connected_output_bits(inst_idx) {
            self.bit_timing_sets[bit_idx] = None;
        }
        for pin_idx in output_pin_order {
            let pin = &self.library.cells[cell_idx].pins[pin_idx];
            let output_sources = pin_sources.get(pin.name.as_str());
            if is_sequential {
                if self.launch_register_instances.contains(&inst_idx) {
                    let output_bit_idx = match output_sources.and_then(|sources| sources.first()) {
                        Some(PinBitSource::Bit(bit_idx)) => *bit_idx,
                        Some(PinBitSource::Literal(_) | PinBitSource::Unknown) | None => {
                            unreachable!("connected sequential outputs use net bits")
                        }
                    };
                    output_updates.insert(
                        output_bit_idx,
                        evaluate_register_launch_output_set(
                            self.library,
                            cell_name.as_str(),
                            pin,
                            &known_pin_values,
                            self.bit_load_capacitance[output_bit_idx],
                            inst_idx,
                            &mut self.validated_timing_tables,
                            &mut diagnostics,
                            &format!(
                                "{}.{} (instance '{}') clock-to-output",
                                cell_name, pin.name, instance_name
                            ),
                        )?,
                    );
                }
                continue;
            }
            if let Some(unsupported_arc) = pin
                .timing_arcs
                .iter()
                .find(|arc| !StaTimingType::from_raw(arc.timing_type.as_str()).is_combinational())
            {
                return Err(anyhow!(
                    "instance '{}' output pin '{}.{}' has unsupported timing type '{}'",
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
            let output_bit_idx = match output_sources.and_then(|sources| sources.first()) {
                Some(PinBitSource::Bit(bit_idx)) => Some(*bit_idx),
                Some(PinBitSource::Literal(_) | PinBitSource::Unknown) => {
                    unreachable!("connected outputs use net bits")
                }
                None => None,
            };
            if combinational_arcs.is_empty() {
                if constant_output_function_value(cell_name.as_str(), pin)?.is_some() {
                    let timing = self.literal_source_timing_set.clone();
                    local_output_timing_sets.insert(pin.name.clone(), timing.clone());
                    if let Some(bit_idx) = output_bit_idx {
                        output_updates.insert(bit_idx, timing);
                    }
                    continue;
                }
                return Err(anyhow!(
                    "instance '{}' output pin '{}.{}' has no usable timing arcs",
                    instance_name,
                    cell_name,
                    pin.name
                ));
            }
            validate_timing_tables_once(
                self.library,
                cell_name.as_str(),
                pin.name.as_str(),
                combinational_arcs.as_slice(),
                &mut self.validated_timing_tables,
            )?;
            let output_load = output_bit_idx
                .map(|bit_idx| self.bit_load_capacitance[bit_idx])
                .unwrap_or_default();
            let mut accumulated: Option<SignalTimingSet> = None;
            for arc in &combinational_arcs {
                let arc_context = format!(
                    "cell '{}' output pin '{}' timing arc related_pin '{}'",
                    cell_name, pin.name, arc.related_pin
                );
                if !arc_when_may_apply(arc, &known_pin_values, arc_context.as_str())? {
                    continue;
                }
                for related_pin_name in split_related_pin_names(arc.related_pin.as_str()) {
                    let related_pin =
                        self.lib.pin(cell_idx, related_pin_name).ok_or_else(|| {
                            anyhow!(
                                "cell '{}' output pin '{}' references unknown timing pin '{}'",
                                cell_name,
                                pin.name,
                                related_pin_name
                            )
                        })?;
                    let mut related_timing_sets = Vec::<&SignalTimingSet>::new();
                    if related_pin.direction == PinDirection::Output as i32 {
                        let Some(timing) = local_output_timing_sets.get(related_pin_name) else {
                            continue;
                        };
                        related_timing_sets.push(timing);
                    } else {
                        let related_sources =
                            pin_sources.get(related_pin_name).ok_or_else(|| {
                                anyhow!(
                                    "instance '{}' output pin '{}.{}' requires connected pin '{}'",
                                    instance_name,
                                    cell_name,
                                    pin.name,
                                    related_pin_name
                                )
                            })?;
                        if related_sources.is_empty() {
                            return Err(anyhow!(
                                "instance '{}' timing-related pin '{}' is unconnected",
                                instance_name,
                                related_pin_name
                            ));
                        }
                        for source in related_sources {
                            match source {
                                PinBitSource::Bit(bit_idx) => {
                                    if let Some(timing) = self.bit_timing_sets[*bit_idx].as_ref() {
                                        related_timing_sets.push(timing);
                                    }
                                }
                                PinBitSource::Literal(_) | PinBitSource::Unknown => {}
                            }
                        }
                    }
                    for input_timing_set in related_timing_sets {
                        let candidate = evaluate_arc_set(
                            self.library,
                            arc,
                            input_timing_set,
                            output_load,
                            Some(inst_idx),
                            &mut diagnostics,
                            &format!(
                                "{}.{} (instance '{}') related_pin '{}'",
                                cell_name, pin.name, instance_name, related_pin_name
                            ),
                        )?;
                        accumulated = Some(match accumulated {
                            Some(previous) => previous.merge(&candidate),
                            None => candidate,
                        });
                    }
                }
            }
            let Some(mut output_timing_set) = accumulated else {
                continue;
            };
            collapse_signal_timing_set_to_envelope(&mut output_timing_set);
            local_output_timing_sets.insert(pin.name.clone(), output_timing_set.clone());
            if let Some(bit_idx) = output_bit_idx {
                output_updates.insert(bit_idx, output_timing_set);
            }
        }
        for (bit_idx, timing) in output_updates {
            self.bit_timing_sets[bit_idx] = Some(timing);
        }
        self.instance_diagnostics[inst_idx] = diagnostics;
        Ok(())
    }

    fn refresh_register_endpoints(&mut self) -> Result<()> {
        self.register_input_timings.fill(None);
        self.register_diagnostics
            .fill(TimingQueryDiagnosticCounts::default());
        for inst_idx in 0..self.instance_cell_indices.len() {
            if !self.instance_is_sequential[inst_idx] {
                continue;
            }
            let cell_idx = self.instance_cell_indices[inst_idx];
            let cell_name = self.instance_cell_names[inst_idx].clone();
            let instance_name = self.instance_names[inst_idx].clone();
            let mut diagnostics = TimingQueryDiagnosticCounts::default();
            let mut endpoint_timing: Option<RegisterCaptureTimingCandidate> = None;
            for pin in self.library.cells[cell_idx]
                .pins
                .iter()
                .filter(|pin| pin.direction == PinDirection::Input as i32 && !pin.is_clocking_pin)
            {
                let setup_arcs: Vec<&TimingArc> = pin
                    .timing_arcs
                    .iter()
                    .filter(|arc| StaTimingType::from_raw(arc.timing_type.as_str()).is_setup())
                    .collect();
                if setup_arcs.is_empty() {
                    continue;
                }
                validate_constraint_tables_once(
                    self.library,
                    cell_name.as_str(),
                    pin.name.as_str(),
                    setup_arcs.as_slice(),
                    &mut self.validated_timing_tables,
                )?;
                let Some(sources) = self.instance_pin_sources[inst_idx].get(pin.name.as_str())
                else {
                    continue;
                };
                for source in sources {
                    let PinBitSource::Bit(bit_idx) = source else {
                        continue;
                    };
                    let Some(data_timing) = self.bit_timing_sets[*bit_idx].as_ref() else {
                        continue;
                    };
                    let Some(candidate) = evaluate_register_setup_capture_arrival(
                        self.library,
                        setup_arcs.as_slice(),
                        &self.instance_known_pin_values[inst_idx],
                        data_timing,
                        inst_idx,
                        &mut diagnostics,
                        &format!(
                            "{}.{} (instance '{}') setup",
                            cell_name, pin.name, instance_name
                        ),
                    )?
                    else {
                        continue;
                    };
                    endpoint_timing = Some(match endpoint_timing {
                        Some(current) => choose_worse_register_capture_timing(current, candidate),
                        None => candidate,
                    });
                }
            }
            self.register_input_timings[inst_idx] = endpoint_timing;
            self.register_diagnostics[inst_idx] = diagnostics;
        }
        Ok(())
    }

    fn build_report(&self) -> Result<StaReport> {
        let mut worst_output: Option<EdgeTimingCandidate> = None;
        for bit_idx in &self.module_output_bits {
            let timing_source = self.resolved_timing_sources[*bit_idx];
            let timing_set = timing_set_for_resolved_source(
                timing_source,
                self.bit_timing_sets.as_slice(),
                &self.literal_source_timing_set,
            );
            let Some(timing_set) = timing_set else {
                continue;
            };
            let output_timing = timing_set.worst_arrival_candidate().ok_or_else(|| {
                anyhow!(
                    "missing edge timing candidates for module output '{}'",
                    self.bit_names[*bit_idx]
                )
            })?;
            worst_output = Some(match worst_output {
                Some(current) => {
                    choose_worse_edge_timing_candidate_by_arrival(current, output_timing)
                }
                None => output_timing,
            });
        }

        let mut aggregate_sets = vec![None::<SignalTimingSet>; self.nets_len];
        for (bit_idx, source) in self.resolved_timing_sources.iter().copied().enumerate() {
            let Some(timing_set) = timing_set_for_resolved_source(
                source,
                self.bit_timing_sets.as_slice(),
                &self.literal_source_timing_set,
            ) else {
                continue;
            };
            let net_idx = self.bit_net_indices[bit_idx];
            aggregate_sets[net_idx] = Some(match aggregate_sets[net_idx].take() {
                Some(previous) => previous.merge(timing_set),
                None => timing_set.clone(),
            });
        }
        let net_timing = aggregate_sets
            .into_iter()
            .map(|timing_set| timing_set.and_then(|set| set.as_report_signal_timing()))
            .collect();
        let worst_register_input = self
            .register_input_timings
            .iter()
            .flatten()
            .cloned()
            .reduce(choose_worse_register_capture_timing);
        let register_input_arrivals = self
            .register_input_timings
            .iter()
            .map(|timing| timing.as_ref().map(|timing| timing.arrival))
            .collect();
        let register_input_breakdowns = self
            .register_input_timings
            .iter()
            .map(|timing| {
                timing
                    .as_ref()
                    .and_then(|timing| timing.register_path_breakdown)
            })
            .collect();
        let register_input_critical_paths = self
            .register_input_timings
            .iter()
            .map(|timing| {
                timing
                    .as_ref()
                    .map(|timing| timing.path_instances.clone())
                    .unwrap_or_default()
            })
            .collect();
        let mut timing_query_diagnostic_counts = TimingQueryDiagnosticCounts::default();
        for diagnostics in self
            .instance_diagnostics
            .iter()
            .chain(self.register_diagnostics.iter())
        {
            timing_query_diagnostic_counts += *diagnostics;
        }
        Ok(StaReport {
            net_timing,
            worst_output_arrival: worst_output
                .as_ref()
                .map(|timing| timing.timing.arrival)
                .unwrap_or(0.0),
            worst_output_breakdown: worst_output
                .as_ref()
                .and_then(|timing| timing.register_path_breakdown),
            worst_register_input_arrival: worst_register_input
                .as_ref()
                .map(|timing| timing.arrival)
                .unwrap_or(0.0),
            worst_register_input_breakdown: worst_register_input
                .as_ref()
                .and_then(|timing| timing.register_path_breakdown),
            register_input_arrivals,
            register_input_breakdowns,
            register_input_critical_paths,
            worst_register_input_critical_path: worst_register_input
                .map(|timing| timing.path_instances)
                .unwrap_or_default(),
            timing_query_diagnostic_counts,
            cell_levels: self.cell_levels,
        })
    }
}
