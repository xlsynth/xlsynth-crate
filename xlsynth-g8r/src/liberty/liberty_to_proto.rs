// SPDX-License-Identifier: Apache-2.0

use crate::liberty::cell_formula::parse_formula;
use crate::liberty::liberty_parser::{Block, BlockMember, Value};
use crate::liberty::util::human_readable_size;
use crate::liberty::{CharReader, LibertyParser};
use crate::liberty_proto::{
    Cell, ClockGate, Library, LibraryUnits, LuTableTemplate, Pin, PinDirection, Sequential,
    SequentialKind, TimingArc, TimingTable,
};
use flate2::bufread::GzDecoder;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

fn value_to_f64(value: &crate::liberty::liberty_parser::Value) -> f64 {
    match value {
        crate::liberty::liberty_parser::Value::Number(n) => *n,
        _ => panic!("Expected number for area attribute"),
    }
}

fn value_to_string(value: &crate::liberty::liberty_parser::Value) -> String {
    match value {
        crate::liberty::liberty_parser::Value::String(s) => s.clone(),
        crate::liberty::liberty_parser::Value::Identifier(s) => s.clone(),
        _ => panic!("Expected string or identifier for function attribute"),
    }
}

fn value_to_bool(value: &crate::liberty::liberty_parser::Value) -> Option<bool> {
    match value {
        crate::liberty::liberty_parser::Value::Identifier(s) => {
            if s.eq_ignore_ascii_case("true") {
                Some(true)
            } else if s.eq_ignore_ascii_case("false") {
                Some(false)
            } else {
                None
            }
        }
        crate::liberty::liberty_parser::Value::String(_) => None,
        crate::liberty::liberty_parser::Value::Number(_) => None,
        crate::liberty::liberty_parser::Value::Tuple(_) => None,
    }
}

fn direction_from_str(s: &str) -> i32 {
    match s {
        "input" => PinDirection::Input as i32,
        "output" => PinDirection::Output as i32,
        _ => PinDirection::Invalid as i32,
    }
}

fn qualifier_to_string(value: &crate::liberty::liberty_parser::Value) -> Option<String> {
    match value {
        crate::liberty::liberty_parser::Value::Identifier(s)
        | crate::liberty::liberty_parser::Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn value_to_attr_string(value: &Value) -> String {
    match value {
        Value::String(s) | Value::Identifier(s) => s.clone(),
        Value::Number(n) => format!("{n}"),
        Value::Tuple(xs) => {
            let parts: Vec<String> = xs.iter().map(|v| value_to_attr_string(v)).collect();
            format!("({})", parts.join(","))
        }
    }
}

fn parse_csv_f64s(text: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for part in text.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed = trimmed
            .parse::<f64>()
            .map_err(|e| format!("failed parsing '{trimmed}' as f64: {e}"))?;
        out.push(parsed);
    }
    Ok(out)
}

fn parse_scalar_list(value: &Value) -> Result<Vec<f64>, String> {
    match value {
        Value::Number(n) => Ok(vec![*n]),
        Value::String(s) | Value::Identifier(s) => parse_csv_f64s(s),
        Value::Tuple(xs) => {
            let mut out = Vec::new();
            for x in xs {
                out.extend(parse_scalar_list(x)?);
            }
            Ok(out)
        }
    }
}

#[derive(Debug)]
struct NumericTensor {
    values: Vec<f64>,
    dimensions: Vec<u32>,
}

fn parse_numeric_tensor(value: &Value, collapse_singletons: bool) -> Result<NumericTensor, String> {
    match value {
        Value::Number(n) => Ok(NumericTensor {
            values: vec![*n],
            dimensions: Vec::new(),
        }),
        Value::String(s) | Value::Identifier(s) => {
            let values = parse_csv_f64s(s)?;
            Ok(NumericTensor {
                dimensions: vec![values.len() as u32],
                values,
            })
        }
        Value::Tuple(xs) => {
            if xs.is_empty() {
                return Ok(NumericTensor {
                    values: Vec::new(),
                    dimensions: vec![0],
                });
            }
            let mut children = Vec::with_capacity(xs.len());
            for x in xs {
                children.push(parse_numeric_tensor(x, collapse_singletons)?);
            }
            if collapse_singletons && xs.len() == 1 {
                return Ok(children.into_iter().next().expect("single child expected"));
            }
            let first_dims = children[0].dimensions.clone();
            for child in &children[1..] {
                if child.dimensions != first_dims {
                    return Err(format!(
                        "ragged tensor in Liberty values payload: first child dims={:?}, other dims={:?}",
                        first_dims, child.dimensions
                    ));
                }
            }
            let mut values = Vec::new();
            for child in children {
                values.extend(child.values);
            }
            let mut dimensions = vec![xs.len() as u32];
            dimensions.extend(first_dims);
            Ok(NumericTensor { values, dimensions })
        }
    }
}

fn value_to_optional_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => Some(*n),
        Value::String(s) | Value::Identifier(s) => s.trim().parse::<f64>().ok(),
        Value::Tuple(_) => None,
    }
}

fn parse_library_units(block: &Block) -> LibraryUnits {
    let mut units = LibraryUnits::default();
    for member in &block.members {
        let BlockMember::BlockAttr(attr) = member else {
            continue;
        };
        let value = value_to_attr_string(&attr.value);
        match attr.attr_name.as_str() {
            "time_unit" => units.time_unit = value,
            "capacitance_unit" => units.capacitance_unit = value,
            "voltage_unit" => units.voltage_unit = value,
            "current_unit" => units.current_unit = value,
            "resistance_unit" => units.resistance_unit = value,
            "pulling_resistance_unit" => units.pulling_resistance_unit = value,
            "leakage_power_unit" => units.leakage_power_unit = value,
            _ => {
                // This helper only extracts the standard Liberty unit
                // declarations.
            }
        }
    }
    units
}

fn parse_lu_table_templates(block: &Block) -> Vec<LuTableTemplate> {
    let mut out = Vec::new();
    for member in &block.members {
        let BlockMember::SubBlock(sub_block) = member else {
            continue;
        };
        if !sub_block.block_type.ends_with("template") {
            continue;
        }
        let mut tmpl = LuTableTemplate {
            kind: sub_block.block_type.clone(),
            name: sub_block
                .qualifiers
                .first()
                .and_then(qualifier_to_string)
                .unwrap_or_default(),
            ..Default::default()
        };
        for member in &sub_block.members {
            let BlockMember::BlockAttr(attr) = member else {
                continue;
            };
            match attr.attr_name.as_str() {
                "variable_1" => tmpl.variable_1 = value_to_attr_string(&attr.value),
                "variable_2" => tmpl.variable_2 = value_to_attr_string(&attr.value),
                "variable_3" => tmpl.variable_3 = value_to_attr_string(&attr.value),
                "index_1" => match parse_scalar_list(&attr.value) {
                    Ok(values) => tmpl.index_1 = values,
                    Err(e) => {
                        log::warn!(
                            "Failed to parse {} index_1 {:?}: {}",
                            tmpl.name,
                            value_to_attr_string(&attr.value),
                            e
                        );
                    }
                },
                "index_2" => match parse_scalar_list(&attr.value) {
                    Ok(values) => tmpl.index_2 = values,
                    Err(e) => {
                        log::warn!(
                            "Failed to parse {} index_2 {:?}: {}",
                            tmpl.name,
                            value_to_attr_string(&attr.value),
                            e
                        );
                    }
                },
                "index_3" => match parse_scalar_list(&attr.value) {
                    Ok(values) => tmpl.index_3 = values,
                    Err(e) => {
                        log::warn!(
                            "Failed to parse {} index_3 {:?}: {}",
                            tmpl.name,
                            value_to_attr_string(&attr.value),
                            e
                        );
                    }
                },
                _ => {
                    // Keep only typed template fields we consume
                    // (variables/index axes).
                }
            }
        }
        out.push(tmpl);
    }
    out.sort_by(|a, b| a.kind.cmp(&b.kind).then(a.name.cmp(&b.name)));
    out
}

fn is_timing_table_block(block: &Block) -> bool {
    block.members.iter().any(|m| {
        let BlockMember::BlockAttr(attr) = m else {
            return false;
        };
        matches!(
            attr.attr_name.as_str(),
            "index_1" | "index_2" | "index_3" | "values"
        )
    })
}

fn parse_timing_table(block: &Block) -> Option<TimingTable> {
    if !is_timing_table_block(block) {
        return None;
    }
    let mut table = TimingTable {
        kind: block.block_type.clone(),
        template_name: block
            .qualifiers
            .first()
            .and_then(qualifier_to_string)
            .unwrap_or_default(),
        ..Default::default()
    };
    for member in &block.members {
        let BlockMember::BlockAttr(attr) = member else {
            continue;
        };
        match attr.attr_name.as_str() {
            "index_1" => match parse_scalar_list(&attr.value) {
                Ok(values) => table.index_1 = values,
                Err(e) => {
                    log::warn!(
                        "Failed to parse {}.{} index_1 {:?}: {}",
                        table.kind,
                        table.template_name,
                        value_to_attr_string(&attr.value),
                        e
                    );
                }
            },
            "index_2" => match parse_scalar_list(&attr.value) {
                Ok(values) => table.index_2 = values,
                Err(e) => {
                    log::warn!(
                        "Failed to parse {}.{} index_2 {:?}: {}",
                        table.kind,
                        table.template_name,
                        value_to_attr_string(&attr.value),
                        e
                    );
                }
            },
            "index_3" => match parse_scalar_list(&attr.value) {
                Ok(values) => table.index_3 = values,
                Err(e) => {
                    log::warn!(
                        "Failed to parse {}.{} index_3 {:?}: {}",
                        table.kind,
                        table.template_name,
                        value_to_attr_string(&attr.value),
                        e
                    );
                }
            },
            "values" => {
                match parse_numeric_tensor(&attr.value, /* collapse_singletons= */ true) {
                    Ok(tensor) => {
                        table.values = tensor.values;
                        table.dimensions = tensor.dimensions;
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to parse {}.{} values {:?}: {}",
                            table.kind,
                            table.template_name,
                            value_to_attr_string(&attr.value),
                            e
                        );
                    }
                }
            }
            _ => {
                // Keep only typed table payload fields (axes and numeric
                // values).
            }
        }
    }
    Some(table)
}

fn parse_timing_arc(block: &Block) -> TimingArc {
    let mut arc = TimingArc::default();
    for member in &block.members {
        match member {
            BlockMember::BlockAttr(attr) => match attr.attr_name.as_str() {
                "related_pin" => arc.related_pin = value_to_attr_string(&attr.value),
                "timing_sense" => arc.timing_sense = value_to_attr_string(&attr.value),
                "timing_type" => arc.timing_type = value_to_attr_string(&attr.value),
                "when" => arc.when = value_to_attr_string(&attr.value),
                _ => {
                    // Keep only typed arc selector fields used by timing
                    // lookup.
                }
            },
            BlockMember::SubBlock(sub_block) => {
                if let Some(table) = parse_timing_table(sub_block) {
                    arc.tables.push(table);
                }
            }
        }
    }
    arc.tables.sort_by(|a, b| a.kind.cmp(&b.kind));
    arc
}

fn build_template_id_map_by_name(
    templates: &[LuTableTemplate],
) -> (HashMap<String, u32>, HashSet<String>) {
    let mut id_by_name: HashMap<String, u32> = HashMap::new();
    let mut ambiguous_names: HashSet<String> = HashSet::new();
    for (i, tmpl) in templates.iter().enumerate() {
        if ambiguous_names.contains(&tmpl.name) {
            continue;
        }
        let id = (i as u32) + 1;
        match id_by_name.entry(tmpl.name.clone()) {
            Entry::Vacant(v) => {
                v.insert(id);
            }
            Entry::Occupied(_) => {
                ambiguous_names.insert(tmpl.name.clone());
            }
        }
    }
    for name in &ambiguous_names {
        id_by_name.remove(name);
    }
    (id_by_name, ambiguous_names)
}

fn canonicalize_timing_table_references(library: &mut Library) {
    let (template_id_by_name, ambiguous_names) =
        build_template_id_map_by_name(&library.lu_table_templates);
    for template_name in &ambiguous_names {
        log::warn!(
            "Template name '{}' is ambiguous across template kinds; keeping per-table template_name strings",
            template_name
        );
    }

    let template_axes: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = library
        .lu_table_templates
        .iter()
        .map(|tmpl| {
            (
                tmpl.index_1.clone(),
                tmpl.index_2.clone(),
                tmpl.index_3.clone(),
            )
        })
        .collect();

    let mut resolved_template_refs = 0usize;
    let mut elided_index_1 = 0usize;
    let mut elided_index_2 = 0usize;
    let mut elided_index_3 = 0usize;
    for cell in &mut library.cells {
        for pin in &mut cell.pins {
            for arc in &mut pin.timing_arcs {
                for table in &mut arc.tables {
                    if table.template_id == 0 && !table.template_name.is_empty() {
                        if let Some(template_id) = template_id_by_name.get(&table.template_name) {
                            table.template_id = *template_id;
                        } else if !ambiguous_names.contains(&table.template_name) {
                            log::warn!(
                                "Could not resolve timing table template '{}' for {}.{}; keeping legacy name",
                                table.template_name,
                                cell.name,
                                pin.name
                            );
                        }
                    }
                    if table.template_id == 0 {
                        continue;
                    }
                    let template_idx = (table.template_id - 1) as usize;
                    if template_idx >= template_axes.len() {
                        log::warn!(
                            "Timing table {} has out-of-range template_id={} for {}.{}",
                            table.kind,
                            table.template_id,
                            cell.name,
                            pin.name
                        );
                        continue;
                    }
                    resolved_template_refs += 1;
                    let (tmpl_index_1, tmpl_index_2, tmpl_index_3) = &template_axes[template_idx];
                    if table.index_1 == *tmpl_index_1 && !table.index_1.is_empty() {
                        table.index_1.clear();
                        elided_index_1 += 1;
                    }
                    if table.index_2 == *tmpl_index_2 && !table.index_2.is_empty() {
                        table.index_2.clear();
                        elided_index_2 += 1;
                    }
                    if table.index_3 == *tmpl_index_3 && !table.index_3.is_empty() {
                        table.index_3.clear();
                        elided_index_3 += 1;
                    }
                    table.template_name.clear();
                }
            }
        }
    }
    log::info!(
        "Canonicalized {} timing tables: elided index_1={} index_2={} index_3={}",
        resolved_template_refs,
        elided_index_1,
        elided_index_2,
        elided_index_3
    );
}

fn extract_sequential_blocks(
    cell_block: &crate::liberty::liberty_parser::Block,
) -> Vec<Sequential> {
    let mut sequential = Vec::new();
    for cell_member in &cell_block.members {
        let crate::liberty::liberty_parser::BlockMember::SubBlock(sub_block) = cell_member else {
            continue;
        };
        let kind = match sub_block.block_type.as_str() {
            "ff" => SequentialKind::Ff,
            "latch" => SequentialKind::Latch,
            _ => continue,
        } as i32;

        let state_vars: Vec<String> = sub_block
            .qualifiers
            .iter()
            .filter_map(qualifier_to_string)
            .collect();
        if state_vars.is_empty() {
            continue;
        }

        let mut next_state = String::new();
        let mut data_in = String::new();
        let mut clock_expr = String::new();
        let mut clear_expr = String::new();
        let mut preset_expr = String::new();
        for seq_member in &sub_block.members {
            let crate::liberty::liberty_parser::BlockMember::BlockAttr(attr) = seq_member else {
                continue;
            };
            if attr.attr_name == "next_state" {
                next_state = value_to_string(&attr.value);
            } else if sub_block.block_type == "latch" && attr.attr_name == "data_in" {
                data_in = value_to_string(&attr.value);
            } else if sub_block.block_type == "ff" && attr.attr_name == "clear" {
                clear_expr = value_to_string(&attr.value);
            } else if sub_block.block_type == "ff" && attr.attr_name == "preset" {
                preset_expr = value_to_string(&attr.value);
            } else if (sub_block.block_type == "ff" && attr.attr_name == "clocked_on")
                || (sub_block.block_type == "latch" && attr.attr_name == "enable")
            {
                clock_expr = value_to_string(&attr.value);
            }
        }

        if next_state.is_empty() && kind == (SequentialKind::Latch as i32) && !data_in.is_empty() {
            next_state = data_in;
        }

        if next_state.is_empty() && clock_expr.is_empty() {
            continue;
        }

        for state_var in state_vars {
            sequential.push(Sequential {
                state_var,
                next_state: next_state.clone(),
                clock_expr: clock_expr.clone(),
                kind,
                clear_expr: clear_expr.clone(),
                preset_expr: preset_expr.clone(),
            });
        }
    }
    sequential
}

fn block_to_proto_cells(block: &Block) -> Vec<Cell> {
    let mut cells = Vec::new();
    for member in &block.members {
        let BlockMember::SubBlock(cell_block) = member else {
            continue;
        };
        if cell_block.block_type != "cell" {
            continue;
        }
        let name = cell_block
            .qualifiers
            .first()
            .and_then(qualifier_to_string)
            .unwrap_or_default();
        let mut area = 0.0;
        let mut clocking_pins: HashSet<String> = HashSet::new();
        let sequential = extract_sequential_blocks(cell_block);

        // First pass: gather cell-level scalar properties (like area) and any
        // clocking pins referenced by sequential blocks via clock expressions.
        for cell_member in &cell_block.members {
            if let BlockMember::BlockAttr(attr) = cell_member {
                if attr.attr_name == "area" {
                    area = value_to_f64(&attr.value);
                }
            }
        }
        for seq in &sequential {
            if seq.clock_expr.is_empty() {
                continue;
            }
            match parse_formula(&seq.clock_expr) {
                Ok(term) => {
                    for input in term.inputs() {
                        clocking_pins.insert(input);
                    }
                }
                Err(e) => {
                    log::warn!(
                        "Failed to parse sequential clock expression {:?}: {}",
                        seq.clock_expr,
                        e
                    );
                }
            }
        }

        // Second pass: build Pin protos, marking pins that are referred to by any
        // clocked_on attribute in an ff block as clocking pins.
        let mut pins = Vec::new();
        let mut clock_gate_clock_pins = Vec::new();
        let mut clock_gate_output_pins = Vec::new();
        let mut clock_gate_enable_pins = Vec::new();
        let mut clock_gate_test_pins = Vec::new();
        for cell_member in &cell_block.members {
            let BlockMember::SubBlock(pin_block) = cell_member else {
                continue;
            };
            if pin_block.block_type != "pin" {
                continue;
            }
            let mut direction = PinDirection::Invalid as i32;
            let mut function = String::new();
            let pin_name = pin_block
                .qualifiers
                .first()
                .and_then(qualifier_to_string)
                .unwrap_or_default();
            let mut pin_is_clock_gate_clock = false;
            let mut pin_is_clock_gate_out = false;
            let mut pin_is_clock_gate_enable = false;
            let mut pin_is_clock_gate_test = false;
            let mut timing_arcs = Vec::new();
            let mut capacitance = None;
            let mut rise_capacitance = None;
            let mut fall_capacitance = None;
            let mut max_capacitance = None;
            for pin_member in &pin_block.members {
                match pin_member {
                    BlockMember::BlockAttr(attr) => {
                        // For an example of these cells see ASAP7 `ICG` cells`.
                        if attr.attr_name == "direction" {
                            if let Value::Identifier(s) = &attr.value {
                                direction = direction_from_str(s);
                            }
                        } else if attr.attr_name == "function" {
                            function = value_to_string(&attr.value);
                        } else if attr.attr_name == "clock_gate_clock_pin" {
                            pin_is_clock_gate_clock = value_to_bool(&attr.value).unwrap_or(false);
                        } else if attr.attr_name == "clock_gate_out_pin" {
                            pin_is_clock_gate_out = value_to_bool(&attr.value).unwrap_or(false);
                        } else if attr.attr_name == "clock_gate_enable_pin" {
                            pin_is_clock_gate_enable = value_to_bool(&attr.value).unwrap_or(false);
                        } else if attr.attr_name == "clock_gate_test_pin" {
                            pin_is_clock_gate_test = value_to_bool(&attr.value).unwrap_or(false);
                        } else if attr.attr_name == "capacitance" {
                            capacitance = value_to_optional_f64(&attr.value);
                        } else if attr.attr_name == "rise_capacitance" {
                            rise_capacitance = value_to_optional_f64(&attr.value);
                        } else if attr.attr_name == "fall_capacitance" {
                            fall_capacitance = value_to_optional_f64(&attr.value);
                        } else if attr.attr_name == "max_capacitance" {
                            max_capacitance = value_to_optional_f64(&attr.value);
                        }
                    }
                    BlockMember::SubBlock(sub_block) => {
                        if sub_block.block_type == "timing" {
                            timing_arcs.push(parse_timing_arc(sub_block));
                        }
                    }
                }
            }
            if pin_is_clock_gate_clock {
                clock_gate_clock_pins.push(pin_name.clone());
            }
            if pin_is_clock_gate_out {
                clock_gate_output_pins.push(pin_name.clone());
            }
            if pin_is_clock_gate_enable {
                clock_gate_enable_pins.push(pin_name.clone());
            }
            if pin_is_clock_gate_test {
                clock_gate_test_pins.push(pin_name.clone());
            }
            let is_clocking_pin = clocking_pins.contains(&pin_name) || pin_is_clock_gate_clock;
            timing_arcs.sort_by(|a, b| {
                a.related_pin
                    .cmp(&b.related_pin)
                    .then(a.timing_type.cmp(&b.timing_type))
                    .then(a.timing_sense.cmp(&b.timing_sense))
            });
            pins.push(Pin {
                direction,
                function,
                name: pin_name,
                is_clocking_pin,
                capacitance,
                rise_capacitance,
                fall_capacitance,
                max_capacitance,
                timing_arcs,
            });
        }
        let has_clock_gate_roles = !clock_gate_clock_pins.is_empty()
            || !clock_gate_output_pins.is_empty()
            || !clock_gate_enable_pins.is_empty()
            || !clock_gate_test_pins.is_empty();
        if clock_gate_clock_pins.len() > 1 {
            log::warn!(
                "Cell '{}' has multiple clock_gate_clock_pin pins; selecting first: {:?}",
                name,
                clock_gate_clock_pins
            );
        }
        if clock_gate_output_pins.len() > 1 {
            log::warn!(
                "Cell '{}' has multiple clock_gate_out_pin pins; selecting first: {:?}",
                name,
                clock_gate_output_pins
            );
        }
        let clock_pin = if let Some(first) = clock_gate_clock_pins.first() {
            first.clone()
        } else {
            String::new()
        };
        let output_pin = clock_gate_output_pins.first().cloned().unwrap_or_default();
        let clock_gate = if has_clock_gate_roles {
            Some(ClockGate {
                clock_pin,
                output_pin,
                enable_pins: clock_gate_enable_pins,
                test_pins: clock_gate_test_pins,
            })
        } else {
            None
        };
        cells.push(Cell {
            area,
            pins,
            name,
            sequential,
            clock_gate,
        });
    }
    cells.sort_by(|a, b| a.name.cmp(&b.name));
    cells
}

pub fn parse_liberty_files_to_proto<P: AsRef<Path>>(paths: &[P]) -> Result<Library, String> {
    let mut libraries = Vec::new();
    for path in paths {
        log::info!("Parsing Liberty file: {}", path.as_ref().display());
        // Log the human-readable file size
        match std::fs::metadata(&path) {
            Ok(meta) => {
                let size = meta.len();
                let human_size = human_readable_size(size);
                log::info!("  Size: {}", human_size);
            }
            Err(e) => {
                log::warn!("  Could not stat file: {}", e);
            }
        }
        let file = File::open(&path)
            .map_err(|e| format!("Failed to open {}: {}", path.as_ref().display(), e))?;
        let streamer: Box<dyn std::io::Read> = if path
            .as_ref()
            .extension()
            .map(|e| e == "gz")
            .unwrap_or(false)
        {
            Box::new(GzDecoder::new(BufReader::with_capacity(256 * 1024, file)))
        } else {
            Box::new(BufReader::new(file))
        };
        let char_reader = CharReader::new(streamer);
        let mut parser = LibertyParser::new_from_iter(char_reader);
        let library = parser
            .parse()
            .map_err(|e| format!("Parse error in {}: {}", path.as_ref().display(), e))?;
        libraries.push(library);
    }
    log::info!("Flattening cells from {} libraries...", libraries.len());
    let mut all_cells = Vec::new();
    let mut all_templates = Vec::new();
    let mut units: Option<LibraryUnits> = None;
    for lib in &libraries {
        all_cells.extend(block_to_proto_cells(lib));
        all_templates.extend(parse_lu_table_templates(lib));
        let parsed_units = parse_library_units(lib);
        let parsed_units_populated = !parsed_units.time_unit.is_empty()
            || !parsed_units.capacitance_unit.is_empty()
            || !parsed_units.voltage_unit.is_empty()
            || !parsed_units.current_unit.is_empty()
            || !parsed_units.resistance_unit.is_empty()
            || !parsed_units.pulling_resistance_unit.is_empty()
            || !parsed_units.leakage_power_unit.is_empty();
        if !parsed_units_populated {
            continue;
        }
        if let Some(existing) = &units {
            if existing != &parsed_units {
                log::warn!(
                    "Multiple Liberty libraries provided different unit declarations; keeping first: {:?}, ignoring: {:?}",
                    existing,
                    parsed_units
                );
            }
        } else {
            units = Some(parsed_units);
        }
    }
    all_templates.sort_by(|a, b| a.kind.cmp(&b.kind).then(a.name.cmp(&b.name)));
    let mut deduped_templates: Vec<LuTableTemplate> = Vec::with_capacity(all_templates.len());
    for tmpl in all_templates {
        if let Some(last) = deduped_templates.last() {
            if last.kind == tmpl.kind && last.name == tmpl.name {
                if last != &tmpl {
                    log::warn!(
                        "Conflicting template definitions for {}({}); keeping first",
                        tmpl.kind,
                        tmpl.name
                    );
                }
                continue;
            }
        }
        deduped_templates.push(tmpl);
    }
    log::info!("Flattened {} cells", all_cells.len());
    let mut library = Library {
        cells: all_cells,
        units: Some(units.unwrap_or_default()),
        lu_table_templates: deduped_templates,
    };
    canonicalize_timing_table_references(&mut library);
    Ok(library)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::descriptor::LIBERTY_DESCRIPTOR;
    use prost::Message;
    use prost_reflect::{DescriptorPool, DynamicMessage};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_liberty_files_to_proto_and_pretty_print() {
        let liberty_text = r#"
        library (my_library) {
            cell (my_and) {
                area: 1.0;
                pin (Y) {
                    direction: output;
                    function: "(A * B)";
                }
                pin (A) {
                    direction: input;
                }
                pin (B) {
                    direction: input;
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        assert_eq!(lib.cells.len(), 1);
        assert_eq!(lib.cells[0].name, "my_and");
        assert_eq!(lib.cells[0].area, 1.0);
        assert_eq!(lib.cells[0].pins.len(), 3);
        assert!(!lib.cells[0].pins[0].is_clocking_pin);
        assert!(!lib.cells[0].pins[1].is_clocking_pin);
        assert!(!lib.cells[0].pins[2].is_clocking_pin);
        // Pretty-print using prost-reflect
        let descriptor_pool = DescriptorPool::decode(LIBERTY_DESCRIPTOR).unwrap();
        let msg_desc = descriptor_pool
            .get_message_by_name("liberty.Library")
            .unwrap();
        let mut buf = Vec::new();
        lib.encode(&mut buf).unwrap();
        let dyn_msg = DynamicMessage::decode(msg_desc, &*buf).unwrap();
        let textproto = dyn_msg.to_text_format();
        println!("Pretty-printed textproto:\n{}", textproto);
        assert!(textproto.contains("cells:["));
        assert!(textproto.contains("name:\"my_and\""));
    }

    #[test]
    fn test_timing_templates_and_arcs_are_captured() {
        let liberty_text = r#"
        library (my_library) {
            time_unit : "1ns";
            capacitance_unit : "1pf";
            lu_table_template (tmpl_2x2) {
                variable_1 : input_net_transition;
                variable_2 : total_output_net_capacitance;
                index_1 ("0.01, 0.02");
                index_2 ("0.10, 0.20");
            }
            cell (NAND2) {
                area: 1.1;
                pin (A) {
                    direction: input;
                    capacitance: 0.01;
                }
                pin (B) {
                    direction: input;
                    capacitance: 0.02;
                }
                pin (Y) {
                    direction: output;
                    function: "!(A * B)";
                    max_capacitance: 0.2;
                    timing () {
                        related_pin : "A";
                        timing_sense : negative_unate;
                        timing_type : combinational;
                        cell_rise (tmpl_2x2) {
                            values ("0.10, 0.20", "0.30, 0.40");
                        }
                        cell_fall (tmpl_2x2) {
                            index_1 ("0.01, 0.02");
                            index_2 ("0.10, 0.20");
                            values ("0.20, 0.30", "0.40, 0.50");
                        }
                    }
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        let units = lib.units.as_ref().expect("units should be present");
        assert_eq!(units.time_unit, "1ns");
        assert_eq!(units.capacitance_unit, "1pf");
        assert_eq!(lib.lu_table_templates.len(), 1);
        let tmpl = &lib.lu_table_templates[0];
        assert_eq!(tmpl.kind, "lu_table_template");
        assert_eq!(tmpl.name, "tmpl_2x2");
        assert_eq!(tmpl.index_1, vec![0.01, 0.02]);
        assert_eq!(tmpl.index_2, vec![0.10, 0.20]);

        assert_eq!(lib.cells.len(), 1);
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "NAND2");
        let pin_a = cell.pins.iter().find(|p| p.name == "A").unwrap();
        assert_eq!(pin_a.capacitance, Some(0.01));

        let pin_y = cell.pins.iter().find(|p| p.name == "Y").unwrap();
        assert_eq!(pin_y.max_capacitance, Some(0.2));
        assert_eq!(pin_y.timing_arcs.len(), 1);
        let arc = &pin_y.timing_arcs[0];
        assert_eq!(arc.related_pin, "A");
        assert_eq!(arc.timing_sense, "negative_unate");
        assert_eq!(arc.timing_type, "combinational");
        assert_eq!(arc.tables.len(), 2);
        let cell_fall = arc.tables.iter().find(|t| t.kind == "cell_fall").unwrap();
        assert_eq!(cell_fall.template_id, 1);
        assert_eq!(cell_fall.template_name, "");
        assert_eq!(cell_fall.index_1, Vec::<f64>::new());
        assert_eq!(cell_fall.index_2, Vec::<f64>::new());
        assert_eq!(cell_fall.dimensions, vec![2, 2]);
        assert_eq!(cell_fall.values, vec![0.20, 0.30, 0.40, 0.50]);
        let cell_rise = arc.tables.iter().find(|t| t.kind == "cell_rise").unwrap();
        assert_eq!(cell_rise.template_id, 1);
        assert_eq!(cell_rise.template_name, "");
        assert_eq!(cell_rise.index_1, Vec::<f64>::new());
        assert_eq!(cell_rise.index_2, Vec::<f64>::new());
        assert_eq!(cell_rise.dimensions, vec![2, 2]);
        assert_eq!(cell_rise.values, vec![0.10, 0.20, 0.30, 0.40]);
    }

    #[test]
    fn test_clocked_on_marks_clocking_pin() {
        let liberty_text = r#"
        library (my_library) {
            cell (my_ff) {
                area: 2.0;
                pin (CLK) {
                    direction: input;
                }
                pin (D) {
                    direction: input;
                }
                pin (Q) {
                    direction: output;
                }
                ff (IQN) {
                    clocked_on : "CLK";
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        assert_eq!(lib.cells.len(), 1);
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_ff");
        assert_eq!(cell.pins.len(), 3);
        let mut clk_is_clocking = None;
        let mut d_is_clocking = None;
        let mut q_is_clocking = None;
        for pin in &cell.pins {
            match pin.name.as_str() {
                "CLK" => clk_is_clocking = Some(pin.is_clocking_pin),
                "D" => d_is_clocking = Some(pin.is_clocking_pin),
                "Q" => q_is_clocking = Some(pin.is_clocking_pin),
                other => panic!("Unexpected pin name in test: {}", other),
            }
        }
        assert_eq!(clk_is_clocking, Some(true));
        assert_eq!(d_is_clocking, Some(false));
        assert_eq!(q_is_clocking, Some(false));
    }

    #[test]
    fn test_clocked_on_multiple_clocks_all_marked() {
        let liberty_text = r#"
        library (my_library) {
            cell (my_ff) {
                area: 2.0;
                pin (CLK1) {
                    direction: input;
                }
                pin (CLK2) {
                    direction: input;
                }
                pin (D) {
                    direction: input;
                }
                pin (Q) {
                    direction: output;
                }
                ff (IQN) {
                    clocked_on : "CLK1 | CLK2";
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        assert_eq!(lib.cells.len(), 1);
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_ff");
        assert_eq!(cell.pins.len(), 4);
        let mut clk1 = None;
        let mut clk2 = None;
        let mut d = None;
        let mut q = None;
        for pin in &cell.pins {
            match pin.name.as_str() {
                "CLK1" => clk1 = Some(pin.is_clocking_pin),
                "CLK2" => clk2 = Some(pin.is_clocking_pin),
                "D" => d = Some(pin.is_clocking_pin),
                "Q" => q = Some(pin.is_clocking_pin),
                other => panic!("Unexpected pin name in test: {}", other),
            }
        }
        assert_eq!(clk1, Some(true));
        assert_eq!(clk2, Some(true));
        assert_eq!(d, Some(false));
        assert_eq!(q, Some(false));
    }

    #[test]
    fn test_ff_next_state_is_captured() {
        // Inspired by the ASAP7 `SDFH*` scan-flop `ff { next_state: ... }` pattern.
        let liberty_text = r#"
        library (my_library) {
            cell (my_scan_ff) {
                area: 3.0;
                pin (CLK) {
                    direction: input;
                }
                pin (D) {
                    direction: input;
                }
                pin (SE) {
                    direction: input;
                }
                pin (SI) {
                    direction: input;
                }
                pin (Q) {
                    direction: output;
                    function: "IQ";
                }
                ff (IQ, IQN) {
                    clocked_on : "CLK";
                    next_state : "(!D * !SE) + (SE * SI)";
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_scan_ff");
        assert_eq!(cell.sequential.len(), 2);
        let mut iq = None;
        let mut iqn = None;
        for seq in &cell.sequential {
            match seq.state_var.as_str() {
                "IQ" => iq = Some(seq),
                "IQN" => iqn = Some(seq),
                other => panic!("Unexpected state var in test: {}", other),
            }
        }
        let iq = iq.expect("missing IQ state");
        let iqn = iqn.expect("missing IQN state");
        assert_eq!(iq.next_state, "(!D * !SE) + (SE * SI)");
        assert_eq!(iq.clock_expr, "CLK");
        assert_eq!(iq.kind, SequentialKind::Ff as i32);
        assert_eq!(iqn.next_state, "(!D * !SE) + (SE * SI)");
        assert_eq!(iqn.clock_expr, "CLK");
        assert_eq!(iqn.kind, SequentialKind::Ff as i32);
    }

    #[test]
    fn test_ff_multiple_state_vars_are_captured() {
        // Mirrors ASAP7 `DFFLQNx1_ASAP7_75t_R` with `ff (IQ,IQN)`.
        let liberty_text = r#"
        library (my_library) {
            cell (my_ff) {
                area: 2.0;
                pin (CLK) {
                    direction: input;
                }
                pin (D) {
                    direction: input;
                }
                pin (Q) {
                    direction: output;
                    function: "IQ";
                }
                pin (QN) {
                    direction: output;
                    function: "IQN";
                }
                ff (IQ, IQN) {
                    clocked_on : "CLK";
                    next_state : "D";
                    power_down_function : "(!VDD) + (VSS)";
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_ff");
        assert_eq!(cell.sequential.len(), 2);
        let mut iq = None;
        let mut iqn = None;
        for seq in &cell.sequential {
            match seq.state_var.as_str() {
                "IQ" => iq = Some(seq),
                "IQN" => iqn = Some(seq),
                other => panic!("Unexpected state var in test: {}", other),
            }
        }
        let iq = iq.expect("missing IQ state");
        let iqn = iqn.expect("missing IQN state");
        assert_eq!(iq.next_state, "D");
        assert_eq!(iq.clock_expr, "CLK");
        assert_eq!(iq.kind, SequentialKind::Ff as i32);
        assert_eq!(iqn.next_state, "D");
        assert_eq!(iqn.clock_expr, "CLK");
        assert_eq!(iqn.kind, SequentialKind::Ff as i32);
    }

    #[test]
    fn test_latch_enable_marks_clocking_pin_and_data_in_captured() {
        // Mirrors ASAP7 `DHLx2_ASAP7_75t_R` latch block with data_in/enable.
        let liberty_text = r#"
        library (my_library) {
            cell (my_latch) {
                area: 1.5;
                pin (CLK) {
                    direction: input;
                }
                pin (D) {
                    direction: input;
                }
                pin (Q) {
                    direction: output;
                    function: "IQ";
                }
                latch (IQ, IQN) {
                    data_in : "D";
                    enable : "CLK";
                    power_down_function : "(!VDD) + (VSS)";
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        assert_eq!(lib.cells.len(), 1);
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_latch");
        assert_eq!(cell.sequential.len(), 2);
        let mut iq = None;
        let mut iqn = None;
        for seq in &cell.sequential {
            match seq.state_var.as_str() {
                "IQ" => iq = Some(seq),
                "IQN" => iqn = Some(seq),
                other => panic!("Unexpected state var in test: {}", other),
            }
        }
        let iq = iq.expect("missing IQ state");
        let iqn = iqn.expect("missing IQN state");
        assert_eq!(iq.next_state, "D");
        assert_eq!(iq.clock_expr, "CLK");
        assert_eq!(iq.kind, SequentialKind::Latch as i32);
        assert_eq!(iqn.next_state, "D");
        assert_eq!(iqn.clock_expr, "CLK");
        assert_eq!(iqn.kind, SequentialKind::Latch as i32);
        let mut clk_is_clocking = None;
        let mut d_is_clocking = None;
        let mut q_is_clocking = None;
        for pin in &cell.pins {
            match pin.name.as_str() {
                "CLK" => clk_is_clocking = Some(pin.is_clocking_pin),
                "D" => d_is_clocking = Some(pin.is_clocking_pin),
                "Q" => q_is_clocking = Some(pin.is_clocking_pin),
                other => panic!("Unexpected pin name in test: {}", other),
            }
        }
        assert_eq!(clk_is_clocking, Some(true));
        assert_eq!(d_is_clocking, Some(false));
        assert_eq!(q_is_clocking, Some(false));
    }

    #[test]
    fn test_clock_gate_pin_roles_are_captured() {
        // This is similar to ASAP7 `ICG` cells`.
        let liberty_text = r#"
        library (my_library) {
            cell (my_icg) {
                area: 1.5;
                pin (CLK) {
                    direction: input;
                    clock_gate_clock_pin: true;
                }
                pin (EN) {
                    direction: input;
                    clock_gate_enable_pin: true;
                }
                pin (TE) {
                    direction: input;
                    clock_gate_test_pin: true;
                }
                pin (GCLK) {
                    direction: output;
                    clock_gate_out_pin: true;
                }
            }
        }
        "#;
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", liberty_text).unwrap();
        let lib = parse_liberty_files_to_proto(&[tmp.path()]).unwrap();
        assert_eq!(lib.cells.len(), 1);
        let cell = &lib.cells[0];
        assert_eq!(cell.name, "my_icg");
        let clock_gate = cell
            .clock_gate
            .as_ref()
            .expect("clock_gate should be present for clock_gate_* annotated cell");
        assert_eq!(clock_gate.clock_pin, "CLK");
        assert_eq!(clock_gate.output_pin, "GCLK");
        assert_eq!(clock_gate.enable_pins, vec!["EN"]);
        assert_eq!(clock_gate.test_pins, vec!["TE"]);

        let mut clk_is_clocking = None;
        let mut en_is_clocking = None;
        let mut te_is_clocking = None;
        let mut gclk_is_clocking = None;
        for pin in &cell.pins {
            match pin.name.as_str() {
                "CLK" => clk_is_clocking = Some(pin.is_clocking_pin),
                "EN" => en_is_clocking = Some(pin.is_clocking_pin),
                "TE" => te_is_clocking = Some(pin.is_clocking_pin),
                "GCLK" => gclk_is_clocking = Some(pin.is_clocking_pin),
                other => panic!("Unexpected pin name in test: {}", other),
            }
        }
        assert_eq!(clk_is_clocking, Some(true));
        assert_eq!(en_is_clocking, Some(false));
        assert_eq!(te_is_clocking, Some(false));
        assert_eq!(gclk_is_clocking, Some(false));
    }

    #[test]
    fn test_committed_liberty_bin_matches_generated() {
        if let Ok(value) = std::env::var("XLSYNTH_CRATE_NO_PROTO_CHECK") {
            if value.trim() == "1" {
                eprintln!(
                    "Skipping descriptor byte comparison: XLSYNTH_CRATE_NO_PROTO_CHECK={}",
                    value.trim()
                );
                eprintln!("To run this test, unset XLSYNTH_CRATE_NO_PROTO_CHECK or set it to 0.");
                return;
            }
        }
        // Check protoc version
        let output = std::process::Command::new("protoc")
            .arg("--version")
            .output()
            .expect("failed to run protoc");
        let version_str = String::from_utf8_lossy(&output.stdout);
        let expected_version = "libprotoc 29.1"; // Update to your canonical version
        let found_version = version_str.trim();
        let committed = include_bytes!("../../proto/liberty.bin") as &[u8];
        // Generate a fresh descriptor set in a temp dir
        let tmp = tempfile::tempdir().unwrap();
        let descriptor_path = tmp.path().join("liberty.bin");
        // Use absolute paths to proto files for robustness.
        let proto_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("proto");
        let liberty_proto = proto_dir.join("liberty.proto");
        let result_proto = proto_dir.join("result.proto");
        let liberty_proto_str = liberty_proto.to_str().unwrap();
        let result_proto_str = result_proto.to_str().unwrap();
        prost_build::Config::new()
            // Keep in sync with build.rs for descriptor determinism.
            .protoc_arg("--experimental_allow_proto3_optional")
            .file_descriptor_set_path(&descriptor_path)
            .compile_protos(
                &[liberty_proto_str, result_proto_str],
                &[proto_dir.to_str().unwrap()],
            )
            .expect("Failed to compile proto");
        let generated = std::fs::read(&descriptor_path).expect("read generated liberty.bin");
        let message = format_args!(
            concat!(
                "Committed proto/liberty.bin does not match the generated file.\n",
                "Expected protoc version: '{expected_version}', installed version: '{found_version}'.\n",
                "If these differ, the version mismatch is likely the reason for the failure.\n",
                "Otherwise, to rebuild proto/liberty.bin, use a matching protoc, run ",
                "`cargo build -p xlsynth-g8r`, then copy ",
                "`target/*/build/xlsynth-g8r-*/out/liberty.bin` to ",
                "`xlsynth-g8r/proto/liberty.bin`.\n",
                "If the test still fails, this test may be out of sync with how build.rs generates the file.\n",
                "To skip this test, set XLSYNTH_CRATE_NO_PROTO_CHECK=1 or true.\n",
            ),
            expected_version = expected_version,
            found_version = found_version
        );
        assert!(committed == generated, "{message}");
    }
}
