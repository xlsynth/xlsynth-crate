// SPDX-License-Identifier: Apache-2.0

use crate::liberty_proto as wire;
use anyhow::{Result, bail};
use std::collections::HashMap;

/// Normalized in-memory Liberty library used by parsers and evaluators.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Library {
    pub cells: Vec<Cell>,
    pub units: Option<wire::LibraryUnits>,
    pub lu_table_templates: Vec<LuTableTemplate>,
    pub threshold_voltage_groups: Vec<String>,
    pub default_threshold_voltage_group_id: u32,
    pub threshold_voltage_group_class_indices: Vec<i32>,
    pub nominal_voltage: Option<f64>,
    pub provenance: String,
    pub source_files: Vec<String>,
}

/// Normalized cell data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cell {
    pub name: String,
    pub pins: Vec<Pin>,
    pub area: f64,
    pub sequential: Vec<wire::Sequential>,
    pub clock_gate: Option<wire::ClockGate>,
    pub threshold_voltage_group_id: u32,
    pub dont_use: Option<bool>,
}

/// Normalized pin data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pin {
    pub direction: i32,
    pub function: String,
    pub name: String,
    pub is_clocking_pin: bool,
    pub capacitance: Option<f64>,
    pub rise_capacitance: Option<f64>,
    pub fall_capacitance: Option<f64>,
    pub max_capacitance: Option<f64>,
    pub timing_arcs: Vec<TimingArc>,
    pub internal_power: Vec<InternalPower>,
}

/// Normalized LUT template with inline float64 axes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LuTableTemplate {
    pub kind: String,
    pub name: String,
    pub variable_1: String,
    pub variable_2: String,
    pub variable_3: String,
    pub index_1: Vec<f64>,
    pub index_2: Vec<f64>,
    pub index_3: Vec<f64>,
}

/// Normalized timing arc.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimingArc {
    pub related_pin: String,
    pub timing_sense: String,
    pub timing_type: String,
    pub when: String,
    pub tables: Vec<TimingTable>,
}

/// Normalized timing table with inline axes and float64 values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimingTable {
    pub kind: String,
    pub template_id: u32,
    pub index_1: Vec<f64>,
    pub index_2: Vec<f64>,
    pub index_3: Vec<f64>,
    pub values: Vec<f64>,
    pub dimensions: Vec<u32>,
    pub template_name: String,
}

/// Normalized internal-power selector and table data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InternalPower {
    pub related_pins: Vec<String>,
    pub when: String,
    pub related_pg_pin: String,
    pub tables: Vec<PowerTable>,
}

/// Normalized power table with inline axes and float64 values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PowerTable {
    pub transition: i32,
    pub template_id: u32,
    pub index_1: Vec<f64>,
    pub index_2: Vec<f64>,
    pub index_3: Vec<f64>,
    pub values: Vec<f64>,
    pub dimensions: Vec<u32>,
    pub template_name: String,
}

fn retain_templates_and_remap_ids(
    library: &mut Library,
    mut keep: impl FnMut(&LuTableTemplate) -> bool,
) {
    let old_templates = std::mem::take(&mut library.lu_table_templates);
    let mut id_remap = vec![0u32; old_templates.len() + 1];
    for (old_index, template) in old_templates.into_iter().enumerate() {
        if keep(&template) {
            let new_id = library.lu_table_templates.len() as u32 + 1;
            id_remap[old_index + 1] = new_id;
            library.lu_table_templates.push(template);
        }
    }
    let remap = |id: &mut u32| {
        if *id != 0 {
            *id = id_remap.get(*id as usize).copied().unwrap_or(0);
        }
    };
    for cell in &mut library.cells {
        for pin in &mut cell.pins {
            for arc in &mut pin.timing_arcs {
                for table in &mut arc.tables {
                    remap(&mut table.template_id);
                }
            }
            for group in &mut pin.internal_power {
                for table in &mut group.tables {
                    remap(&mut table.template_id);
                }
            }
        }
    }
}

/// Removes timing arcs and timing templates from normalized library data.
pub fn strip_timing_data(library: &mut Library) {
    for cell in &mut library.cells {
        for pin in &mut cell.pins {
            pin.timing_arcs.clear();
        }
    }
    retain_templates_and_remap_ids(library, |template| template.kind != "lu_table_template");
}

/// Removes dynamic-power payloads and templates from normalized library data.
pub fn strip_power_data(library: &mut Library) {
    library.nominal_voltage = None;
    for cell in &mut library.cells {
        for pin in &mut cell.pins {
            pin.internal_power.clear();
        }
    }
    retain_templates_and_remap_ids(library, |template| template.kind != "power_lut_template");
}

/// Wire-format discriminator stored in every serialized Liberty library.
pub const LIBERTY_FORMAT_MAGIC: u64 = 0x4c49_4256_3250_524f;

/// Returns whether a timing-table kind is consumed by the deterministic STA.
pub fn is_evaluator_timing_table_kind(kind: &str) -> bool {
    matches!(
        kind,
        "cell_rise"
            | "cell_fall"
            | "rise_transition"
            | "fall_transition"
            | "rise_constraint"
            | "fall_constraint"
    )
}

/// Returns whether a decoded library has the expected wire-format magic.
pub fn has_valid_header(library: &wire::Library) -> bool {
    library.format_magic == LIBERTY_FORMAT_MAGIC
}

fn axis_key(axis: &[f64]) -> Vec<u64> {
    axis.iter().map(|value| value.to_bits()).collect()
}

fn record_axis(axis: &[f64], frequencies: &mut HashMap<Vec<u64>, u64>) {
    if !axis.is_empty() {
        *frequencies.entry(axis_key(axis)).or_default() += 1;
    }
}

fn axis_id(axis: &[f64], id_by_axis: &HashMap<Vec<u64>, u32>) -> u32 {
    if axis.is_empty() {
        0
    } else {
        *id_by_axis
            .get(&axis_key(axis))
            .expect("every nonempty LUT axis was interned")
    }
}

fn values_to_f32(values: Vec<f64>) -> Result<Vec<f32>> {
    let mut result = Vec::with_capacity(values.len());
    for value in values {
        let compact = value as f32;
        if value.is_finite() && !compact.is_finite() {
            bail!("LUT value {value} is outside the finite float32 range");
        }
        result.push(compact);
    }
    Ok(result)
}

fn timing_table_to_wire(
    table: TimingTable,
    id_by_axis: &HashMap<Vec<u64>, u32>,
) -> Result<wire::TimingTable> {
    Ok(wire::TimingTable {
        kind: table.kind,
        template_id: table.template_id,
        index_1_id: axis_id(&table.index_1, id_by_axis),
        index_2_id: axis_id(&table.index_2, id_by_axis),
        index_3_id: axis_id(&table.index_3, id_by_axis),
        values: values_to_f32(table.values)?,
        dimensions: table.dimensions,
        template_name: table.template_name,
    })
}

fn power_table_to_wire(
    table: PowerTable,
    id_by_axis: &HashMap<Vec<u64>, u32>,
) -> Result<wire::PowerTable> {
    Ok(wire::PowerTable {
        transition: table.transition,
        template_id: table.template_id,
        index_1_id: axis_id(&table.index_1, id_by_axis),
        index_2_id: axis_id(&table.index_2, id_by_axis),
        index_3_id: axis_id(&table.index_3, id_by_axis),
        values: values_to_f32(table.values)?,
        dimensions: table.dimensions,
        template_name: table.template_name,
    })
}

/// Converts normalized data into the compact protobuf representation.
///
/// Timing-table kinds not consumed by the deterministic evaluator are omitted.
pub fn library_to_proto(library: Library) -> Result<wire::Library> {
    let mut frequencies = HashMap::new();
    for template in &library.lu_table_templates {
        record_axis(&template.index_1, &mut frequencies);
        record_axis(&template.index_2, &mut frequencies);
        record_axis(&template.index_3, &mut frequencies);
    }
    for cell in &library.cells {
        for pin in &cell.pins {
            for arc in &pin.timing_arcs {
                for table in &arc.tables {
                    if !is_evaluator_timing_table_kind(&table.kind) {
                        continue;
                    }
                    record_axis(&table.index_1, &mut frequencies);
                    record_axis(&table.index_2, &mut frequencies);
                    record_axis(&table.index_3, &mut frequencies);
                }
            }
            for group in &pin.internal_power {
                for table in &group.tables {
                    record_axis(&table.index_1, &mut frequencies);
                    record_axis(&table.index_2, &mut frequencies);
                    record_axis(&table.index_3, &mut frequencies);
                }
            }
        }
    }

    // Frequent axes receive low IDs, minimizing protobuf varint references.
    // Bit-pattern ordering makes ties deterministic.
    let mut axes: Vec<(Vec<u64>, u64)> = frequencies.into_iter().collect();
    axes.sort_by(|(lhs_axis, lhs_frequency), (rhs_axis, rhs_frequency)| {
        rhs_frequency
            .cmp(lhs_frequency)
            .then_with(|| lhs_axis.cmp(rhs_axis))
    });
    let mut id_by_axis = HashMap::with_capacity(axes.len());
    let mut lut_axes = Vec::with_capacity(axes.len());
    for (index, (axis, _frequency)) in axes.into_iter().enumerate() {
        let id = u32::try_from(index + 1).map_err(|_| anyhow::anyhow!("too many LUT axes"))?;
        id_by_axis.insert(axis.clone(), id);
        lut_axes.push(wire::LutAxis {
            values: axis.into_iter().map(f64::from_bits).collect(),
        });
    }

    let lu_table_templates = library
        .lu_table_templates
        .into_iter()
        .map(|template| wire::LuTableTemplate {
            kind: template.kind,
            name: template.name,
            variable_1: template.variable_1,
            variable_2: template.variable_2,
            variable_3: template.variable_3,
            index_1_id: axis_id(&template.index_1, &id_by_axis),
            index_2_id: axis_id(&template.index_2, &id_by_axis),
            index_3_id: axis_id(&template.index_3, &id_by_axis),
        })
        .collect();
    let cells = library
        .cells
        .into_iter()
        .map(|cell| {
            let pins = cell
                .pins
                .into_iter()
                .map(|pin| {
                    let timing_arcs = pin
                        .timing_arcs
                        .into_iter()
                        .map(|arc| {
                            let tables = arc
                                .tables
                                .into_iter()
                                .filter(|table| is_evaluator_timing_table_kind(&table.kind))
                                .map(|table| timing_table_to_wire(table, &id_by_axis))
                                .collect::<Result<Vec<_>>>()?;
                            Ok(wire::TimingArc {
                                related_pin: arc.related_pin,
                                timing_sense: arc.timing_sense,
                                timing_type: arc.timing_type,
                                when: arc.when,
                                tables,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let internal_power = pin
                        .internal_power
                        .into_iter()
                        .map(|group| {
                            let tables = group
                                .tables
                                .into_iter()
                                .map(|table| power_table_to_wire(table, &id_by_axis))
                                .collect::<Result<Vec<_>>>()?;
                            Ok(wire::InternalPower {
                                related_pins: group.related_pins,
                                when: group.when,
                                related_pg_pin: group.related_pg_pin,
                                tables,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(wire::Pin {
                        direction: pin.direction,
                        function: pin.function,
                        name: pin.name,
                        is_clocking_pin: pin.is_clocking_pin,
                        capacitance: pin.capacitance,
                        rise_capacitance: pin.rise_capacitance,
                        fall_capacitance: pin.fall_capacitance,
                        max_capacitance: pin.max_capacitance,
                        timing_arcs,
                        internal_power,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(wire::Cell {
                name: cell.name,
                pins,
                area: cell.area,
                sequential: cell.sequential,
                clock_gate: cell.clock_gate,
                threshold_voltage_group_id: cell.threshold_voltage_group_id,
                dont_use: cell.dont_use,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(wire::Library {
        format_magic: LIBERTY_FORMAT_MAGIC,
        cells,
        units: library.units,
        lu_table_templates,
        threshold_voltage_groups: library.threshold_voltage_groups,
        default_threshold_voltage_group_id: library.default_threshold_voltage_group_id,
        threshold_voltage_group_class_indices: library.threshold_voltage_group_class_indices,
        nominal_voltage: library.nominal_voltage,
        provenance: library.provenance,
        source_files: library.source_files,
        lut_axes,
    })
}

fn resolve_axis(axis_id: u32, axes: &[wire::LutAxis]) -> Result<Vec<f64>> {
    if axis_id == 0 {
        return Ok(Vec::new());
    }
    let index = (axis_id - 1) as usize;
    let axis = axes
        .get(index)
        .ok_or_else(|| anyhow::anyhow!("LUT axis ID {axis_id} is out of range"))?;
    Ok(axis.values.clone())
}

fn timing_table_from_wire(table: wire::TimingTable, axes: &[wire::LutAxis]) -> Result<TimingTable> {
    Ok(TimingTable {
        kind: table.kind,
        template_id: table.template_id,
        index_1: resolve_axis(table.index_1_id, axes)?,
        index_2: resolve_axis(table.index_2_id, axes)?,
        index_3: resolve_axis(table.index_3_id, axes)?,
        values: table.values.into_iter().map(f64::from).collect(),
        dimensions: table.dimensions,
        template_name: table.template_name,
    })
}

fn power_table_from_wire(table: wire::PowerTable, axes: &[wire::LutAxis]) -> Result<PowerTable> {
    Ok(PowerTable {
        transition: table.transition,
        template_id: table.template_id,
        index_1: resolve_axis(table.index_1_id, axes)?,
        index_2: resolve_axis(table.index_2_id, axes)?,
        index_3: resolve_axis(table.index_3_id, axes)?,
        values: table.values.into_iter().map(f64::from).collect(),
        dimensions: table.dimensions,
        template_name: table.template_name,
    })
}

/// Expands a compact protobuf payload into normalized float64 data.
pub fn library_from_proto(library: wire::Library) -> Result<Library> {
    if !has_valid_header(&library) {
        bail!(
            "invalid Liberty proto header: magic=0x{:016x}",
            library.format_magic
        );
    }
    let axes = library.lut_axes;
    let lu_table_templates = library
        .lu_table_templates
        .into_iter()
        .map(|template| {
            Ok(LuTableTemplate {
                kind: template.kind,
                name: template.name,
                variable_1: template.variable_1,
                variable_2: template.variable_2,
                variable_3: template.variable_3,
                index_1: resolve_axis(template.index_1_id, &axes)?,
                index_2: resolve_axis(template.index_2_id, &axes)?,
                index_3: resolve_axis(template.index_3_id, &axes)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let cells = library
        .cells
        .into_iter()
        .map(|cell| {
            let pins = cell
                .pins
                .into_iter()
                .map(|pin| {
                    let timing_arcs = pin
                        .timing_arcs
                        .into_iter()
                        .map(|arc| {
                            let tables = arc
                                .tables
                                .into_iter()
                                .map(|table| timing_table_from_wire(table, &axes))
                                .collect::<Result<Vec<_>>>()?;
                            Ok(TimingArc {
                                related_pin: arc.related_pin,
                                timing_sense: arc.timing_sense,
                                timing_type: arc.timing_type,
                                when: arc.when,
                                tables,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let internal_power = pin
                        .internal_power
                        .into_iter()
                        .map(|group| {
                            let tables = group
                                .tables
                                .into_iter()
                                .map(|table| power_table_from_wire(table, &axes))
                                .collect::<Result<Vec<_>>>()?;
                            Ok(InternalPower {
                                related_pins: group.related_pins,
                                when: group.when,
                                related_pg_pin: group.related_pg_pin,
                                tables,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Pin {
                        direction: pin.direction,
                        function: pin.function,
                        name: pin.name,
                        is_clocking_pin: pin.is_clocking_pin,
                        capacitance: pin.capacitance,
                        rise_capacitance: pin.rise_capacitance,
                        fall_capacitance: pin.fall_capacitance,
                        max_capacitance: pin.max_capacitance,
                        timing_arcs,
                        internal_power,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Cell {
                name: cell.name,
                pins,
                area: cell.area,
                sequential: cell.sequential,
                clock_gate: cell.clock_gate,
                threshold_voltage_group_id: cell.threshold_voltage_group_id,
                dont_use: cell.dont_use,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Library {
        cells,
        units: library.units,
        lu_table_templates,
        threshold_voltage_groups: library.threshold_voltage_groups,
        default_threshold_voltage_group_id: library.default_threshold_voltage_group_id,
        threshold_voltage_group_class_indices: library.threshold_voltage_group_class_indices,
        nominal_voltage: library.nominal_voltage,
        provenance: library.provenance,
        source_files: library.source_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proto_conversion_roundtrips_with_float32_precision() {
        let shared_axis = vec![0.01, 0.02];
        let library = Library {
            lu_table_templates: vec![LuTableTemplate {
                index_1: shared_axis.clone(),
                ..Default::default()
            }],
            cells: vec![Cell {
                pins: vec![Pin {
                    timing_arcs: vec![TimingArc {
                        tables: vec![TimingTable {
                            kind: "cell_rise".to_string(),
                            index_1: shared_axis.clone(),
                            values: vec![0.1, 0.2],
                            dimensions: vec![2],
                            ..Default::default()
                        }],
                        ..Default::default()
                    }],
                    internal_power: vec![InternalPower {
                        tables: vec![PowerTable {
                            transition: wire::PowerTransition::Rise as i32,
                            index_1: shared_axis,
                            values: vec![0.3, 0.4],
                            dimensions: vec![2],
                            ..Default::default()
                        }],
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };

        let proto = library_to_proto(library).unwrap();

        assert!(has_valid_header(&proto));
        assert_eq!(proto.lut_axes.len(), 1);
        assert_eq!(proto.lu_table_templates[0].index_1_id, 1);
        assert_eq!(
            proto.cells[0].pins[0].timing_arcs[0].tables[0].values,
            vec![0.1_f32, 0.2_f32]
        );
        let roundtrip = library_from_proto(proto).unwrap();
        assert_eq!(
            roundtrip.cells[0].pins[0].timing_arcs[0].tables[0].values,
            vec![f64::from(0.1_f32), f64::from(0.2_f32)]
        );
        assert_eq!(
            roundtrip.cells[0].pins[0].internal_power[0].tables[0].values,
            vec![f64::from(0.3_f32), f64::from(0.4_f32)]
        );
    }

    #[test]
    fn proto_conversion_rejects_invalid_header() {
        let error = library_from_proto(wire::Library::default()).unwrap_err();
        assert!(format!("{error:#}").contains("invalid Liberty proto header"));
    }
}
