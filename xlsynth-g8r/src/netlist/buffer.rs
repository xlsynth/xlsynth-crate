// SPDX-License-Identifier: Apache-2.0

//! Shared Liberty-buffer options and disabled legacy buffering.

use crate::liberty_model::Library;
#[cfg(test)]
use crate::liberty_model::PinDirection;
#[cfg(test)]
use crate::netlist::cell_catalog::{CatalogCell, CellCatalog};
use crate::netlist::parse::{Net, NetlistModule};
#[cfg(test)]
use crate::netlist::parse::{NetIndex, NetRef, NetlistInstance, PortDirection};
#[cfg(test)]
use crate::netlist::sta::{CombinationalOutputLoad, effective_input_capacitance_for_mapping};
#[cfg(test)]
use crate::netlist::utils::scalar_constant_output_assignments;
use anyhow::Result;
#[cfg(test)]
use anyhow::anyhow;
use serde::Serialize;
#[cfg(test)]
use std::collections::HashMap;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Keeps inserted buffers away from the steep end of their timing tables.
#[cfg(test)]
const MAX_BUFFER_OUTPUT_LOAD_FRACTION: f64 = 0.35;

/// Electrical and fanout bounds for timing-aware mapped-netlist buffering.
#[derive(Clone, Debug, PartialEq)]
pub struct BufferOptions {
    /// Largest number of directly connected input pins at a tree level.
    pub max_fanout: usize,
    /// Optional rise/fall load bound in the units of the Liberty library.
    pub target_load: Option<f64>,
    /// Extra capacitive load attached to each module output.
    pub module_output_load: f64,
    /// Whether data primary inputs may be roots of inserted buffer trees.
    pub buffer_primary_inputs: bool,
}

impl Default for BufferOptions {
    fn default() -> Self {
        Self {
            max_fanout: 12,
            target_load: None,
            module_output_load: 0.0,
            buffer_primary_inputs: false,
        }
    }
}

/// Deterministic accounting for one buffer-insertion pass.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct BufferStats {
    pub buffered_nets: usize,
    pub buffers_inserted: usize,
    pub area_added: f64,
    pub max_fanout_before: usize,
    pub max_fanout_after: usize,
    pub max_load_before: f64,
    pub max_load_after: f64,
    pub unresolved_overloaded_nets: usize,
    /// Complete combinational or register-aware delay before buffering.
    pub initial_worst_delay: Option<f64>,
    /// Independently recomputed worst-path delay after accepted buffering.
    pub final_worst_delay: Option<f64>,
    /// Number of complete Liberty STA evaluations used to accept buffer edits.
    pub timing_evaluations: usize,
    /// Candidate batches rejected because their exact timing did not improve.
    pub rejected_timing_batches: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SinkTarget {
    InstancePin {
        instance_index: usize,
        connection_index: usize,
    },
    ModuleOutput {
        net: NetIndex,
    },
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct BufferSink {
    target: SinkTarget,
    load: CombinationalOutputLoad,
}

#[cfg(test)]
#[derive(Clone, Debug, Default)]
struct NetFanout {
    sinks: Vec<BufferSink>,
    driver: Option<(usize, usize)>,
    is_primary_input: bool,
    protected_clock: bool,
}

#[cfg(test)]
#[derive(Default)]
struct SinkGroup {
    sinks: Vec<BufferSink>,
    load: CombinationalOutputLoad,
}

/// Rejects the retired capacitance-only buffer-insertion entry point.
#[track_caller]
pub fn insert_buffers(
    _module: &mut NetlistModule,
    _nets: &mut Vec<Net>,
    _interner: &mut StringInterner<StringBackend<SymbolU32>>,
    _library: &Library,
    _options: &BufferOptions,
) -> Result<BufferStats> {
    panic!("capacitance-balanced buffer insertion is disabled; use insert_timing_aware_buffers");
}

/// Preserves the retired algorithm strictly for focused test characterization.
#[cfg(test)]
fn insert_capacitance_balanced_buffers_for_characterization(
    module: &mut NetlistModule,
    nets: &mut Vec<Net>,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
) -> Result<BufferStats> {
    validate_options(options)?;
    if module.net_index_range.end != nets.len() {
        return Err(anyhow!(
            "buffer insertion requires the selected module to own the end of the net table"
        ));
    }

    let catalog = CellCatalog::new(library)?;
    let initial_net_count = nets.len();
    let original_fanouts = collect_net_fanouts(module, nets, interner, library, options)?;
    let mut stats = BufferStats {
        max_fanout_before: original_fanouts
            .iter()
            .map(|fanout| fanout.sinks.len())
            .max()
            .unwrap_or(0),
        max_load_before: original_fanouts
            .iter()
            .map(|fanout| sink_load(fanout.sinks.as_slice()))
            .map(max_load)
            .fold(0.0, f64::max),
        ..BufferStats::default()
    };
    let mut names = FreshNames::default();

    for (net_index, fanout) in original_fanouts
        .into_iter()
        .enumerate()
        .take(initial_net_count)
    {
        if fanout.protected_clock
            || (fanout.is_primary_input && !options.buffer_primary_inputs)
            || (fanout.driver.is_none() && !fanout.is_primary_input)
        {
            continue;
        }

        let source_net = NetIndex(net_index);
        let target_load = root_load_limit(&fanout, module, interner, library, &catalog, options)?;
        if !is_overloaded(fanout.sinks.as_slice(), options.max_fanout, target_load) {
            continue;
        }
        if catalog.buffers().next().is_none() {
            return Err(anyhow!(
                "buffer insertion found an overloaded net but Liberty has no usable identity buffer"
            ));
        }

        let mut tree_root = source_net;
        if fanout
            .sinks
            .iter()
            .any(|sink| matches!(sink.target, SinkTarget::ModuleOutput { .. }))
        {
            let Some((driver_index, connection_index)) = fanout.driver else {
                // An ideal primary-input/output alias has no internal driver to split.
                continue;
            };
            tree_root = fresh_wire(module, nets, interner, &mut names);
            module.instances[driver_index].connections[connection_index].1 =
                NetRef::Simple(tree_root);
        }

        let mut level_sinks = fanout.sinks;
        let mut inserted_at_this_root = false;
        loop {
            if inserted_at_this_root
                && !is_overloaded(level_sinks.as_slice(), options.max_fanout, target_load)
            {
                break;
            }
            let previous_count = level_sinks.len();
            let previous_load = max_load(sink_load(level_sinks.as_slice()));
            let groups = partition_sinks(level_sinks, options.max_fanout, target_load);
            let mut next_sinks = Vec::with_capacity(groups.len());
            for group in groups {
                let buffer = choose_buffer(&catalog, group.load)?;
                let output_net = group
                    .sinks
                    .iter()
                    .find_map(|sink| match sink.target {
                        SinkTarget::ModuleOutput { net } => Some(net),
                        SinkTarget::InstancePin { .. } => None,
                    })
                    .unwrap_or_else(|| fresh_wire(module, nets, interner, &mut names));

                for sink in &group.sinks {
                    if let SinkTarget::InstancePin {
                        instance_index,
                        connection_index,
                    } = sink.target
                    {
                        module.instances[instance_index].connections[connection_index].1 =
                            NetRef::Simple(output_net);
                    }
                }

                let (instance_index, input_connection_index) = append_buffer_instance(
                    module, interner, library, buffer, tree_root, output_net, &mut names,
                )?;
                stats.buffers_inserted += 1;
                stats.area_added += buffer.area;
                next_sinks.push(BufferSink {
                    target: SinkTarget::InstancePin {
                        instance_index,
                        connection_index: input_connection_index,
                    },
                    load: buffer.input_capacitances[0],
                });
            }

            inserted_at_this_root = true;
            if next_sinks.len() >= previous_count
                && max_load(sink_load(next_sinks.as_slice())) >= previous_load - f64::EPSILON
            {
                // Another identical tree level cannot reduce either fanout or
                // electrical load; report the remaining violation instead of
                // repeatedly inserting buffers for an infeasible constraint.
                break;
            }
            level_sinks = next_sinks;
        }
        if inserted_at_this_root {
            stats.buffered_nets += 1;
        }
    }

    let final_fanouts = collect_net_fanouts(module, nets, interner, library, options)?;
    stats.max_fanout_after = final_fanouts
        .iter()
        .map(|fanout| fanout.sinks.len())
        .max()
        .unwrap_or(0);
    stats.max_load_after = final_fanouts
        .iter()
        .map(|fanout| max_load(sink_load(fanout.sinks.as_slice())))
        .fold(0.0, f64::max);
    for fanout in &final_fanouts {
        if fanout.protected_clock
            || (fanout.is_primary_input && !options.buffer_primary_inputs)
            || (fanout.driver.is_none() && !fanout.is_primary_input)
        {
            continue;
        }
        let limit = root_load_limit(fanout, module, interner, library, &catalog, options)?;
        if is_overloaded(fanout.sinks.as_slice(), options.max_fanout, limit) {
            stats.unresolved_overloaded_nets += 1;
        }
    }
    Ok(stats)
}

/// Validates buffer-tree bounds before inspecting or changing the netlist.
#[cfg(test)]
fn validate_options(options: &BufferOptions) -> Result<()> {
    if options.max_fanout < 2 {
        return Err(anyhow!("buffer max_fanout must be at least 2"));
    }
    if !options.module_output_load.is_finite() || options.module_output_load < 0.0 {
        return Err(anyhow!(
            "buffer module_output_load must be non-negative and finite"
        ));
    }
    if options
        .target_load
        .is_some_and(|load| !load.is_finite() || load <= 0.0)
    {
        return Err(anyhow!("buffer target_load must be positive and finite"));
    }
    Ok(())
}

/// Collects exact scalar sink pins and module-output loads for each net.
#[cfg(test)]
fn collect_net_fanouts(
    module: &NetlistModule,
    nets: &[Net],
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &BufferOptions,
) -> Result<Vec<NetFanout>> {
    let constant_outputs = scalar_constant_output_assignments(module, nets)?;
    let cell_by_name: HashMap<&str, usize> = library
        .cells
        .iter()
        .enumerate()
        .map(|(index, cell)| (cell.name.as_str(), index))
        .collect();
    let mut fanouts = vec![NetFanout::default(); nets.len()];

    for port in &module.ports {
        let net = module.find_net_index(port.name, nets).ok_or_else(|| {
            anyhow!(
                "module port '{}' has no corresponding net",
                interner.resolve(port.name).unwrap_or("<unknown>")
            )
        })?;
        match port.direction {
            PortDirection::Input => fanouts[net.0].is_primary_input = true,
            PortDirection::Output => fanouts[net.0].sinks.push(BufferSink {
                target: SinkTarget::ModuleOutput { net },
                load: CombinationalOutputLoad {
                    rise: options.module_output_load,
                    fall: options.module_output_load,
                },
            }),
            PortDirection::Inout => fanouts[net.0].protected_clock = true,
        }
    }

    for (instance_index, instance) in module.instances.iter().enumerate() {
        let cell_name = interner
            .resolve(instance.type_name)
            .ok_or_else(|| anyhow!("cannot resolve mapped instance cell type"))?;
        let cell_index = cell_by_name
            .get(cell_name)
            .copied()
            .ok_or_else(|| anyhow!("mapped instance uses unknown Liberty cell '{cell_name}'"))?;
        let cell = &library.cells[cell_index];

        for (connection_index, (port, net_ref)) in instance.connections.iter().enumerate() {
            let pin_name = interner
                .resolve(*port)
                .ok_or_else(|| anyhow!("cannot resolve mapped instance pin"))?;
            let pin = cell
                .pins
                .iter()
                .find(|pin| library.resolve_string(&pin.name) == pin_name)
                .ok_or_else(|| anyhow!("cell '{cell_name}' has no pin '{pin_name}'"))?;
            let net = match net_ref {
                NetRef::Simple(net) => *net,
                NetRef::Literal(_) | NetRef::UnknownLiteral(_) | NetRef::Unconnected => continue,
                _ => {
                    return Err(anyhow!(
                        "buffer insertion currently requires scalar simple-net cell connections"
                    ));
                }
            };
            if net.0 >= fanouts.len() {
                return Err(anyhow!("mapped instance references an out-of-range net"));
            }
            if pin.direction == PinDirection::Input as i32 {
                if pin.is_clocking_pin {
                    fanouts[net.0].protected_clock = true;
                }
                fanouts[net.0].sinks.push(BufferSink {
                    target: SinkTarget::InstancePin {
                        instance_index,
                        connection_index,
                    },
                    load: effective_input_capacitance_for_mapping(
                        pin,
                        format!("buffer sink '{cell_name}.{pin_name}'").as_str(),
                    )?,
                });
            } else if pin.direction == PinDirection::Output as i32 {
                if fanouts[net.0]
                    .driver
                    .replace((instance_index, connection_index))
                    .is_some()
                {
                    return Err(anyhow!(
                        "buffer insertion requires exactly one driver per mapped net"
                    ));
                }
            }
        }
    }
    for net in constant_outputs.keys() {
        if fanouts[*net].driver.is_some() {
            return Err(anyhow!(
                "buffer insertion found a cell driving a constant-tied output"
            ));
        }
    }
    Ok(fanouts)
}

/// Uses an explicit target, the existing driver limit, or a real buffer limit.
#[cfg(test)]
fn root_load_limit(
    fanout: &NetFanout,
    module: &NetlistModule,
    interner: &StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    catalog: &CellCatalog,
    options: &BufferOptions,
) -> Result<Option<f64>> {
    if let Some(limit) = options.target_load {
        return Ok(Some(limit));
    }
    if let Some((instance_index, connection_index)) = fanout.driver {
        let instance = &module.instances[instance_index];
        let cell_name = interner
            .resolve(instance.type_name)
            .ok_or_else(|| anyhow!("cannot resolve buffer driver cell"))?;
        let output_name = interner
            .resolve(instance.connections[connection_index].0)
            .ok_or_else(|| anyhow!("cannot resolve buffer driver output pin"))?;
        if let Some(pin) = catalog
            .by_name(cell_name)
            .map(|entry| &library.cells[entry.cell_index])
            .and_then(|cell| {
                cell.pins
                    .iter()
                    .find(|pin| library.resolve_string(&pin.name) == output_name)
            })
            && let Some(limit) = pin.max_capacitance
            && limit.is_finite()
            && limit > 0.0
        {
            return Ok(Some(limit));
        }
    }
    Ok(catalog
        .buffers()
        .filter_map(|buffer| buffer.output_max_capacitance)
        .find(|limit| limit.is_finite() && *limit > 0.0))
}

/// Partitions sinks deterministically by both count and rise/fall capacitance.
#[cfg(test)]
fn partition_sinks(
    mut sinks: Vec<BufferSink>,
    max_fanout: usize,
    target_load: Option<f64>,
) -> Vec<SinkGroup> {
    sinks.sort_by(|lhs, rhs| {
        max_load(rhs.load)
            .total_cmp(&max_load(lhs.load))
            .then_with(|| sink_target_order(lhs.target, rhs.target))
    });
    let mut groups: Vec<SinkGroup> = Vec::new();
    for sink in sinks {
        let eligible = groups
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                group.sinks.len() < max_fanout
                    && target_load.is_none_or(|limit| {
                        group.load.rise + sink.load.rise <= limit
                            && group.load.fall + sink.load.fall <= limit
                    })
            })
            .min_by(|(lhs_index, lhs), (rhs_index, rhs)| {
                max_load(lhs.load)
                    .total_cmp(&max_load(rhs.load))
                    .then_with(|| lhs_index.cmp(rhs_index))
            })
            .map(|(index, _)| index);
        let group_index = eligible.unwrap_or_else(|| {
            groups.push(SinkGroup::default());
            groups.len() - 1
        });
        groups[group_index].load.rise += sink.load.rise;
        groups[group_index].load.fall += sink.load.fall;
        groups[group_index].sinks.push(sink);
    }
    groups
}

/// Keeps equal-load sink grouping independent of map or Liberty ordering.
#[cfg(test)]
fn sink_target_order(lhs: SinkTarget, rhs: SinkTarget) -> std::cmp::Ordering {
    match (lhs, rhs) {
        (
            SinkTarget::InstancePin {
                instance_index: lhs_instance,
                connection_index: lhs_connection,
            },
            SinkTarget::InstancePin {
                instance_index: rhs_instance,
                connection_index: rhs_connection,
            },
        ) => (lhs_instance, lhs_connection).cmp(&(rhs_instance, rhs_connection)),
        (SinkTarget::InstancePin { .. }, SinkTarget::ModuleOutput { .. }) => {
            std::cmp::Ordering::Less
        }
        (SinkTarget::ModuleOutput { .. }, SinkTarget::InstancePin { .. }) => {
            std::cmp::Ordering::Greater
        }
        (SinkTarget::ModuleOutput { net: lhs }, SinkTarget::ModuleOutput { net: rhs }) => {
            lhs.0.cmp(&rhs.0)
        }
    }
}

/// Picks the smallest buffer with usable timing headroom for the actual load.
#[cfg(test)]
fn choose_buffer(catalog: &CellCatalog, load: CombinationalOutputLoad) -> Result<&CatalogCell> {
    if let Some(buffer) = catalog.buffers().find(|buffer| {
        buffer
            .output_max_capacitance
            .is_none_or(|limit| max_load(load) <= limit * MAX_BUFFER_OUTPUT_LOAD_FRACTION)
    }) {
        return Ok(buffer);
    }
    if let Some(buffer) = catalog.buffers().find(|buffer| {
        buffer
            .output_max_capacitance
            .is_none_or(|limit| max_load(load) <= limit)
    }) {
        return Ok(buffer);
    }
    catalog
        .buffers()
        .max_by(|lhs, rhs| {
            lhs.output_max_capacitance
                .unwrap_or(0.0)
                .total_cmp(&rhs.output_max_capacitance.unwrap_or(0.0))
                .then_with(|| rhs.area.total_cmp(&lhs.area))
                .then_with(|| rhs.name.cmp(&lhs.name))
        })
        .ok_or_else(|| anyhow!("Liberty has no usable identity buffer"))
}

/// Returns whether a direct net exceeds count or characterized load bounds.
#[cfg(test)]
fn is_overloaded(sinks: &[BufferSink], max_fanout: usize, target_load: Option<f64>) -> bool {
    sinks.len() > max_fanout
        || target_load.is_some_and(|limit| {
            let load = sink_load(sinks);
            load.rise > limit || load.fall > limit
        })
}

/// Sums the separate rise and fall capacitances of sink pins.
#[cfg(test)]
fn sink_load(sinks: &[BufferSink]) -> CombinationalOutputLoad {
    sinks
        .iter()
        .fold(CombinationalOutputLoad::default(), |mut total, sink| {
            total.rise += sink.load.rise;
            total.fall += sink.load.fall;
            total
        })
}

/// Returns a conservative scalar for rise/fall electrical comparisons.
#[cfg(test)]
fn max_load(load: CombinationalOutputLoad) -> f64 {
    load.rise.max(load.fall)
}

#[cfg(test)]
#[derive(Default)]
struct FreshNames {
    wire: usize,
    instance: usize,
}

/// Adds a deterministic internal scalar wire to the selected module.
#[cfg(test)]
fn fresh_wire(
    module: &mut NetlistModule,
    nets: &mut Vec<Net>,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    names: &mut FreshNames,
) -> NetIndex {
    loop {
        let name = format!("n_buf_{}", names.wire);
        names.wire += 1;
        if interner.get(name.as_str()).is_some() {
            continue;
        }
        let index = NetIndex(nets.len());
        nets.push(Net {
            name: interner.get_or_intern(name.as_str()),
            width: None,
        });
        module.wires.push(index);
        module.net_index_range.end = nets.len();
        return index;
    }
}

/// Appends one real Liberty buffer and returns its mutable input endpoint.
#[cfg(test)]
fn append_buffer_instance(
    module: &mut NetlistModule,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    buffer: &CatalogCell,
    input_net: NetIndex,
    output_net: NetIndex,
    names: &mut FreshNames,
) -> Result<(usize, usize)> {
    let instance_name = loop {
        let candidate = format!("u_buf_{}", names.instance);
        names.instance += 1;
        if interner.get(candidate.as_str()).is_none() {
            break interner.get_or_intern(candidate.as_str());
        }
    };
    let cell = &library.cells[buffer.cell_index];
    let input_name = library.resolve_string(&cell.pins[buffer.input_pin_indices[0]].name);
    let output_name = library.resolve_string(&cell.pins[buffer.output_pin_index].name);
    let mut connections = vec![
        (
            interner.get_or_intern(input_name),
            NetRef::Simple(input_net),
        ),
        (
            interner.get_or_intern(output_name),
            NetRef::Simple(output_net),
        ),
    ];
    connections.sort_by(|lhs, rhs| {
        interner
            .resolve(lhs.0)
            .unwrap_or("")
            .cmp(interner.resolve(rhs.0).unwrap_or(""))
    });
    let input_connection_index = connections
        .iter()
        .position(|(pin, _)| interner.resolve(*pin) == Some(input_name))
        .ok_or_else(|| anyhow!("inserted buffer input connection is missing"))?;
    let instance_index = module.instances.len();
    module.instances.push(NetlistInstance {
        type_name: interner.get_or_intern(buffer.name.as_str()),
        instance_name,
        connections,
        inst_lineno: 1,
        inst_colno: 1,
    });
    Ok((instance_index, input_connection_index))
}

#[cfg(test)]
mod tests {
    use super::{
        BufferOptions, choose_buffer, insert_buffers,
        insert_capacitance_balanced_buffers_for_characterization,
    };
    use crate::netlist::cell_catalog::CellCatalog;
    use crate::netlist::cell_catalog::test_utils::{parse_module, sizing_library};
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::report::build_sta_report;
    use crate::netlist::sta::{CombinationalOutputLoad, StaOptions};

    #[test]
    #[should_panic(
        expected = "capacitance-balanced buffer insertion is disabled; use insert_timing_aware_buffers"
    )]
    fn legacy_capacitance_balanced_inserter_panics() {
        let library = sizing_library();
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, y);
  input a;
  output y;
  BUF driver (.A(a), .Y(y));
endmodule
"#,
        );

        let _ = insert_buffers(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions::default(),
        );
    }

    #[test]
    fn selects_a_buffer_with_characterized_output_load_headroom() {
        let library = sizing_library();
        let catalog = CellCatalog::new(&library).unwrap();
        let buffer = choose_buffer(
            &catalog,
            CombinationalOutputLoad {
                rise: 0.4,
                fall: 0.4,
            },
        )
        .expect("a stronger legal buffer should retain timing-table headroom");

        assert_eq!(buffer.name, "BUF_FAST");
    }

    #[test]
    fn splits_high_fanout_into_a_balanced_buffer_tree() {
        let source = r#"
module top(a, y0, y1, y2, y3, y4, y5, y6, y7, y8);
  input a;
  output y0, y1, y2, y3, y4, y5, y6, y7, y8;
  wire root;
  BUF driver (.A(a), .Y(root));
  BUF use0 (.A(root), .Y(y0));
  BUF use1 (.A(root), .Y(y1));
  BUF use2 (.A(root), .Y(y2));
  BUF use3 (.A(root), .Y(y3));
  BUF use4 (.A(root), .Y(y4));
  BUF use5 (.A(root), .Y(y5));
  BUF use6 (.A(root), .Y(y6));
  BUF use7 (.A(root), .Y(y7));
  BUF use8 (.A(root), .Y(y8));
endmodule
"#;
        let library = sizing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        let stats = insert_capacitance_balanced_buffers_for_characterization(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 3,
                ..BufferOptions::default()
            },
        )
        .unwrap();

        assert_eq!(stats.max_fanout_before, 9);
        assert!(stats.max_fanout_after <= 3);
        assert_eq!(stats.buffers_inserted, 3);
        assert_eq!(stats.unresolved_overloaded_nets, 0);
        build_sta_report(&module, &nets, &interner, &library, StaOptions::default())
            .expect("buffered module should remain a valid timing-complete netlist");
    }

    #[test]
    fn preserves_a_high_fanout_primary_output_name() {
        let source = r#"
module top(a, y, y0, y1, y2);
  input a;
  output y, y0, y1, y2;
  BUF driver (.A(a), .Y(y));
  BUF use0 (.A(y), .Y(y0));
  BUF use1 (.A(y), .Y(y1));
  BUF use2 (.A(y), .Y(y2));
endmodule
"#;
        let library = sizing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        insert_capacitance_balanced_buffers_for_characterization(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 2,
                ..BufferOptions::default()
            },
        )
        .unwrap();

        let report = build_sta_report(&module, &nets, &interner, &library, StaOptions::default())
            .expect("all original output names should remain driven");
        assert!(report.outputs.iter().any(|output| output.output == "y"));
    }

    #[test]
    fn buffering_is_deterministic() {
        let source = r#"
module top(a, y0, y1, y2);
  input a;
  output y0, y1, y2;
  wire root;
  BUF driver (.A(a), .Y(root));
  BUF use0 (.A(root), .Y(y0));
  BUF use1 (.A(root), .Y(y1));
  BUF use2 (.A(root), .Y(y2));
endmodule
"#;
        let library = sizing_library();
        let options = BufferOptions {
            max_fanout: 2,
            ..BufferOptions::default()
        };
        let render = || {
            let (mut module, mut nets, mut interner) = parse_module(source);
            insert_capacitance_balanced_buffers_for_characterization(
                &mut module,
                &mut nets,
                &mut interner,
                &library,
                &options,
            )
            .unwrap();
            emit_module_as_netlist_text(&module, &nets, &interner).unwrap()
        };
        assert_eq!(render(), render());
    }

    #[test]
    fn infeasible_load_bound_terminates_and_reports_remaining_overload() {
        let source = r#"
module top(a, y);
  input a;
  output y;
  wire root;
  BUF driver (.A(a), .Y(root));
  BUF use (.A(root), .Y(y));
endmodule
"#;
        let library = sizing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        let stats = insert_capacitance_balanced_buffers_for_characterization(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                target_load: Some(0.05),
                ..BufferOptions::default()
            },
        )
        .unwrap();

        assert_eq!(stats.buffers_inserted, 1);
        assert!(stats.unresolved_overloaded_nets > 0);
        build_sta_report(&module, &nets, &interner, &library, StaOptions::default())
            .expect("an infeasible electrical target must still leave a valid netlist");
    }

    #[test]
    fn preserves_constant_output_assignments_while_buffering_logic() {
        let source = r#"
module top(a, y0, y1, y2, zero);
  input a;
  output y0, y1, y2, zero;
  wire root;
  assign zero = 1'b0;
  BUF driver (.A(a), .Y(root));
  BUF use0 (.A(root), .Y(y0));
  BUF use1 (.A(root), .Y(y1));
  BUF use2 (.A(root), .Y(y2));
endmodule
"#;
        let library = sizing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        let stats = insert_capacitance_balanced_buffers_for_characterization(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &BufferOptions {
                max_fanout: 2,
                ..BufferOptions::default()
            },
        )
        .expect("constant output tie-offs should not prevent buffering");

        assert!(stats.buffers_inserted > 0);
        assert_eq!(module.assigns.len(), 1);
        let report = build_sta_report(&module, &nets, &interner, &library, StaOptions::default())
            .expect("buffered constant-output module should remain timing-complete");
        assert!(report.outputs.iter().any(|output| output.output == "zero"));
    }
}
