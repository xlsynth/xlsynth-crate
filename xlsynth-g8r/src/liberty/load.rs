// SPDX-License-Identifier: Apache-2.0

use crate::liberty::descriptor::liberty_descriptor_pool;
use crate::liberty_proto;
use anyhow::{Context, Result, anyhow};
use prost::Message;
use prost_reflect::DynamicMessage;
use std::fs::File;
use std::io::Read;
use std::ops::Deref;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Library {
    proto: liberty_proto::Library,
}

#[derive(Clone, Debug)]
pub struct LibraryWithTimingData {
    proto: liberty_proto::Library,
}

impl Library {
    fn from_proto(mut proto: liberty_proto::Library) -> Self {
        strip_timing_data(&mut proto);
        Self { proto }
    }

    pub fn as_proto(&self) -> &liberty_proto::Library {
        &self.proto
    }

    pub fn into_proto(self) -> liberty_proto::Library {
        self.proto
    }
}

impl LibraryWithTimingData {
    fn from_proto(proto: liberty_proto::Library) -> Self {
        Self { proto }
    }

    pub fn as_proto(&self) -> &liberty_proto::Library {
        &self.proto
    }

    pub fn into_proto(self) -> liberty_proto::Library {
        self.proto
    }
}

impl Deref for Library {
    type Target = liberty_proto::Library;

    fn deref(&self) -> &Self::Target {
        &self.proto
    }
}

impl Deref for LibraryWithTimingData {
    type Target = liberty_proto::Library;

    fn deref(&self) -> &Self::Target {
        &self.proto
    }
}

impl AsRef<liberty_proto::Library> for Library {
    fn as_ref(&self) -> &liberty_proto::Library {
        &self.proto
    }
}

impl AsRef<liberty_proto::Library> for LibraryWithTimingData {
    fn as_ref(&self) -> &liberty_proto::Library {
        &self.proto
    }
}

impl From<Library> for liberty_proto::Library {
    fn from(value: Library) -> Self {
        value.proto
    }
}

impl From<LibraryWithTimingData> for liberty_proto::Library {
    fn from(value: LibraryWithTimingData) -> Self {
        value.proto
    }
}

impl From<LibraryWithTimingData> for Library {
    fn from(value: LibraryWithTimingData) -> Self {
        Library::from_proto(value.proto)
    }
}

fn is_timing_template_kind(kind: &str) -> bool {
    kind == "lu_table_template"
}

pub fn strip_timing_data(proto: &mut liberty_proto::Library) {
    proto
        .lu_table_templates
        .retain(|tmpl| !is_timing_template_kind(&tmpl.kind));
    for cell in &mut proto.cells {
        for pin in &mut cell.pins {
            pin.timing_arcs.clear();
        }
    }
}

fn has_timing_data(proto: &liberty_proto::Library) -> bool {
    for cell in &proto.cells {
        for pin in &cell.pins {
            if !pin.timing_arcs.is_empty() {
                return true;
            }
        }
    }
    false
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimingTableSummary {
    pub cells: usize,
    pub timing_tables: usize,
}

pub fn count_timing_tables(proto: &liberty_proto::Library) -> usize {
    let mut count = 0usize;
    for cell in &proto.cells {
        for pin in &cell.pins {
            for arc in &pin.timing_arcs {
                count += arc.tables.len();
            }
        }
    }
    count
}

pub fn count_timing_values(proto: &liberty_proto::Library) -> usize {
    let mut values = 0usize;
    for cell in &proto.cells {
        for pin in &cell.pins {
            for arc in &pin.timing_arcs {
                for table in &arc.tables {
                    values += table.values.len();
                }
            }
        }
    }
    values
}

#[derive(Clone, PartialEq, Message)]
struct LibrarySkipTimingTableValuesPayload {
    #[prost(message, repeated, tag = "1")]
    cells: Vec<CellSkipTimingTableValuesPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct CellSkipTimingTableValuesPayload {
    #[prost(message, repeated, tag = "2")]
    pins: Vec<PinSkipTimingTableValuesPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct PinSkipTimingTableValuesPayload {
    #[prost(message, repeated, tag = "9")]
    timing_arcs: Vec<TimingArcSkipTimingTableValuesPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct TimingArcSkipTimingTableValuesPayload {
    #[prost(message, repeated, tag = "5")]
    tables: Vec<TimingTableSkipTimingTableValuesPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct TimingTableSkipTimingTableValuesPayload {
    #[prost(double, repeated, tag = "3")]
    index_1: Vec<f64>,
    #[prost(double, repeated, tag = "4")]
    index_2: Vec<f64>,
    #[prost(double, repeated, tag = "5")]
    index_3: Vec<f64>,
    #[prost(uint32, repeated, tag = "7")]
    dimensions: Vec<u32>,
}

fn count_timing_tables_skip_values(proto: &LibrarySkipTimingTableValuesPayload) -> usize {
    let mut count = 0usize;
    for cell in &proto.cells {
        for pin in &cell.pins {
            for arc in &pin.timing_arcs {
                count += arc.tables.len();
            }
        }
    }
    count
}

pub fn decode_timing_table_summary_skip_values_from_bytes(
    bytes: &[u8],
    source_name: &str,
) -> Result<TimingTableSummary> {
    match LibrarySkipTimingTableValuesPayload::decode(bytes) {
        Ok(decoded) => Ok(TimingTableSummary {
            cells: decoded.cells.len(),
            timing_tables: count_timing_tables_skip_values(&decoded),
        }),
        Err(skip_decode_err) => {
            let full = decode_full_binary_or_text(bytes, source_name).with_context(|| {
                format!(
                    "skip-timing-values binary decode failed for '{}': {}",
                    source_name, skip_decode_err
                )
            })?;
            Ok(TimingTableSummary {
                cells: full.cells.len(),
                timing_tables: count_timing_tables(&full),
            })
        }
    }
}

#[derive(Clone, PartialEq, Message)]
struct LibraryNoTimingPayload {
    #[prost(message, repeated, tag = "1")]
    cells: Vec<CellNoTimingPayload>,
    #[prost(message, optional, tag = "2")]
    units: Option<LibraryUnitsNoTimingPayload>,
    #[prost(message, repeated, tag = "3")]
    lu_table_templates: Vec<LuTableTemplateNoTimingPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct CellNoTimingPayload {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, repeated, tag = "2")]
    pins: Vec<PinNoTimingPayload>,
    #[prost(double, tag = "3")]
    area: f64,
    #[prost(message, repeated, tag = "4")]
    sequential: Vec<SequentialNoTimingPayload>,
    #[prost(message, optional, tag = "5")]
    clock_gate: Option<ClockGateNoTimingPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct PinNoTimingPayload {
    #[prost(enumeration = "liberty_proto::PinDirection", tag = "1")]
    direction: i32,
    #[prost(string, tag = "2")]
    function: String,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(bool, tag = "4")]
    is_clocking_pin: bool,
    #[prost(double, optional, tag = "5")]
    capacitance: Option<f64>,
    #[prost(double, optional, tag = "6")]
    rise_capacitance: Option<f64>,
    #[prost(double, optional, tag = "7")]
    fall_capacitance: Option<f64>,
    #[prost(double, optional, tag = "8")]
    max_capacitance: Option<f64>,
}

#[derive(Clone, PartialEq, Message)]
struct ClockGateNoTimingPayload {
    #[prost(string, tag = "1")]
    clock_pin: String,
    #[prost(string, tag = "2")]
    output_pin: String,
    #[prost(string, repeated, tag = "3")]
    enable_pins: Vec<String>,
    #[prost(string, repeated, tag = "4")]
    test_pins: Vec<String>,
}

#[derive(Clone, PartialEq, Message)]
struct LibraryUnitsNoTimingPayload {
    #[prost(string, tag = "1")]
    time_unit: String,
    #[prost(string, tag = "2")]
    capacitance_unit: String,
    #[prost(string, tag = "3")]
    voltage_unit: String,
    #[prost(string, tag = "4")]
    current_unit: String,
    #[prost(string, tag = "5")]
    resistance_unit: String,
    #[prost(string, tag = "6")]
    pulling_resistance_unit: String,
    #[prost(string, tag = "7")]
    leakage_power_unit: String,
}

#[derive(Clone, PartialEq, Message)]
struct LuTableTemplateNoTimingPayload {
    #[prost(string, tag = "1")]
    kind: String,
    #[prost(string, tag = "2")]
    name: String,
    #[prost(string, tag = "3")]
    variable_1: String,
    #[prost(string, tag = "4")]
    variable_2: String,
    #[prost(string, tag = "5")]
    variable_3: String,
    #[prost(double, repeated, tag = "6")]
    index_1: Vec<f64>,
    #[prost(double, repeated, tag = "7")]
    index_2: Vec<f64>,
    #[prost(double, repeated, tag = "8")]
    index_3: Vec<f64>,
}

#[derive(Clone, PartialEq, Message)]
struct SequentialNoTimingPayload {
    #[prost(string, tag = "1")]
    state_var: String,
    #[prost(string, tag = "2")]
    next_state: String,
    #[prost(string, tag = "3")]
    clock_expr: String,
    #[prost(enumeration = "liberty_proto::SequentialKind", tag = "4")]
    kind: i32,
    #[prost(string, tag = "5")]
    clear_expr: String,
    #[prost(string, tag = "6")]
    preset_expr: String,
}

impl From<LibraryNoTimingPayload> for liberty_proto::Library {
    fn from(value: LibraryNoTimingPayload) -> Self {
        liberty_proto::Library {
            cells: value.cells.into_iter().map(Into::into).collect(),
            units: value.units.map(Into::into),
            lu_table_templates: value
                .lu_table_templates
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<CellNoTimingPayload> for liberty_proto::Cell {
    fn from(value: CellNoTimingPayload) -> Self {
        liberty_proto::Cell {
            name: value.name,
            pins: value.pins.into_iter().map(Into::into).collect(),
            area: value.area,
            sequential: value.sequential.into_iter().map(Into::into).collect(),
            clock_gate: value.clock_gate.map(Into::into),
        }
    }
}

impl From<PinNoTimingPayload> for liberty_proto::Pin {
    fn from(value: PinNoTimingPayload) -> Self {
        liberty_proto::Pin {
            direction: value.direction,
            function: value.function,
            name: value.name,
            is_clocking_pin: value.is_clocking_pin,
            capacitance: value.capacitance,
            rise_capacitance: value.rise_capacitance,
            fall_capacitance: value.fall_capacitance,
            max_capacitance: value.max_capacitance,
            timing_arcs: vec![],
        }
    }
}

impl From<ClockGateNoTimingPayload> for liberty_proto::ClockGate {
    fn from(value: ClockGateNoTimingPayload) -> Self {
        liberty_proto::ClockGate {
            clock_pin: value.clock_pin,
            output_pin: value.output_pin,
            enable_pins: value.enable_pins,
            test_pins: value.test_pins,
        }
    }
}

impl From<LibraryUnitsNoTimingPayload> for liberty_proto::LibraryUnits {
    fn from(value: LibraryUnitsNoTimingPayload) -> Self {
        liberty_proto::LibraryUnits {
            time_unit: value.time_unit,
            capacitance_unit: value.capacitance_unit,
            voltage_unit: value.voltage_unit,
            current_unit: value.current_unit,
            resistance_unit: value.resistance_unit,
            pulling_resistance_unit: value.pulling_resistance_unit,
            leakage_power_unit: value.leakage_power_unit,
        }
    }
}

impl From<LuTableTemplateNoTimingPayload> for liberty_proto::LuTableTemplate {
    fn from(value: LuTableTemplateNoTimingPayload) -> Self {
        liberty_proto::LuTableTemplate {
            kind: value.kind,
            name: value.name,
            variable_1: value.variable_1,
            variable_2: value.variable_2,
            variable_3: value.variable_3,
            index_1: value.index_1,
            index_2: value.index_2,
            index_3: value.index_3,
        }
    }
}

impl From<SequentialNoTimingPayload> for liberty_proto::Sequential {
    fn from(value: SequentialNoTimingPayload) -> Self {
        liberty_proto::Sequential {
            state_var: value.state_var,
            next_state: value.next_state,
            clock_expr: value.clock_expr,
            kind: value.kind,
            clear_expr: value.clear_expr,
            preset_expr: value.preset_expr,
        }
    }
}

fn decode_full_binary_or_text(bytes: &[u8], source_name: &str) -> Result<liberty_proto::Library> {
    match liberty_proto::Library::decode(bytes) {
        Ok(lib) => Ok(lib),
        Err(binary_err) => {
            let descriptor_pool = liberty_descriptor_pool();
            let msg_desc = descriptor_pool
                .get_message_by_name("liberty.Library")
                .ok_or_else(|| anyhow!("missing liberty.Library descriptor"))?;
            let text = std::str::from_utf8(bytes).with_context(|| {
                format!(
                    "binary decode failed for '{}': {}; textproto fallback input is not UTF-8",
                    source_name, binary_err
                )
            })?;
            let dyn_msg = DynamicMessage::parse_text_format(msg_desc, text).with_context(|| {
                format!(
                    "binary decode failed for '{}': {}; textproto fallback parse failed",
                    source_name, binary_err
                )
            })?;
            let encoded = dyn_msg.encode_to_vec();
            liberty_proto::Library::decode(encoded.as_slice()).with_context(|| {
                format!(
                    "decoding textproto fallback for '{}' into liberty.Library",
                    source_name
                )
            })
        }
    }
}

fn decode_library_from_bytes(bytes: &[u8], source_name: &str) -> Result<Library> {
    match LibraryNoTimingPayload::decode(bytes) {
        Ok(decoded) => Ok(Library::from_proto(decoded.into())),
        Err(no_timing_err) => {
            let lib = decode_full_binary_or_text(bytes, source_name).with_context(|| {
                format!(
                    "no-timing binary decode failed for '{}': {}",
                    source_name, no_timing_err
                )
            })?;
            Ok(Library::from_proto(lib))
        }
    }
}

fn decode_library_with_timing_data_from_bytes(
    bytes: &[u8],
    source_name: &str,
) -> Result<LibraryWithTimingData> {
    let lib = decode_full_binary_or_text(bytes, source_name)?;
    if !has_timing_data(&lib) {
        return Err(anyhow!(
            "liberty proto '{}' has no timing payloads; load a timing-enabled proto or use the non-timing loader",
            source_name
        ));
    }
    Ok(LibraryWithTimingData::from_proto(lib))
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    File::open(path)
        .with_context(|| format!("opening liberty proto '{}'", path.display()))?
        .read_to_end(&mut buf)
        .with_context(|| format!("reading liberty proto '{}'", path.display()))?;
    Ok(buf)
}

pub fn load_library_from_path(path: &Path) -> Result<Library> {
    let buf = read_file(path)?;
    decode_library_from_bytes(&buf, &path.display().to_string())
}

pub fn load_library_with_timing_data_from_path(path: &Path) -> Result<LibraryWithTimingData> {
    let buf = read_file(path)?;
    decode_library_with_timing_data_from_bytes(&buf, &path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_with_timing_payload() -> liberty_proto::Library {
        liberty_proto::Library {
            cells: vec![liberty_proto::Cell {
                name: "NAND2".to_string(),
                pins: vec![liberty_proto::Pin {
                    direction: liberty_proto::PinDirection::Output as i32,
                    function: "!(A*B)".to_string(),
                    name: "Y".to_string(),
                    timing_arcs: vec![liberty_proto::TimingArc {
                        related_pin: "A".to_string(),
                        timing_sense: "negative_unate".to_string(),
                        timing_type: "combinational".to_string(),
                        tables: vec![liberty_proto::TimingTable {
                            kind: "cell_rise".to_string(),
                            index_1: vec![0.1],
                            values: vec![1.0],
                            dimensions: vec![1],
                            ..Default::default()
                        }],
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            lu_table_templates: vec![liberty_proto::LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1".to_string(),
                index_1: vec![0.1],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_without_timing_payload() -> liberty_proto::Library {
        liberty_proto::Library {
            cells: vec![liberty_proto::Cell {
                name: "INV".to_string(),
                pins: vec![
                    liberty_proto::Pin {
                        direction: liberty_proto::PinDirection::Input as i32,
                        function: "".to_string(),
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    liberty_proto::Pin {
                        direction: liberty_proto::PinDirection::Output as i32,
                        function: "!A".to_string(),
                        name: "Y".to_string(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn make_templates_only_payload() -> liberty_proto::Library {
        liberty_proto::Library {
            cells: vec![liberty_proto::Cell {
                name: "INV".to_string(),
                pins: vec![
                    liberty_proto::Pin {
                        direction: liberty_proto::PinDirection::Input as i32,
                        function: "".to_string(),
                        name: "A".to_string(),
                        ..Default::default()
                    },
                    liberty_proto::Pin {
                        direction: liberty_proto::PinDirection::Output as i32,
                        function: "!A".to_string(),
                        name: "Y".to_string(),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            lu_table_templates: vec![liberty_proto::LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_1".to_string(),
                index_1: vec![0.1],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn strip_timing_data_preserves_non_timing_templates() {
        let mut lib = make_with_timing_payload();
        lib.lu_table_templates.push(liberty_proto::LuTableTemplate {
            kind: "power_lut_template".to_string(),
            name: "power_tmpl".to_string(),
            index_1: vec![0.1],
            ..Default::default()
        });

        strip_timing_data(&mut lib);

        assert_eq!(lib.lu_table_templates.len(), 1);
        assert_eq!(lib.lu_table_templates[0].kind, "power_lut_template");
        assert!(lib.cells[0].pins[0].timing_arcs.is_empty());
    }

    #[test]
    fn decode_without_timing_removes_timing_payloads() {
        let with_timing = make_with_timing_payload();
        let bytes = with_timing.encode_to_vec();
        let loaded = decode_library_from_bytes(&bytes, "unit-test").unwrap();
        assert!(loaded.lu_table_templates.is_empty());
        assert_eq!(loaded.cells.len(), 1);
        assert_eq!(loaded.cells[0].pins.len(), 1);
        assert!(loaded.cells[0].pins[0].timing_arcs.is_empty());
    }

    #[test]
    fn decode_without_timing_preserves_non_timing_templates_for_binary() {
        let mut with_timing = make_with_timing_payload();
        with_timing
            .lu_table_templates
            .push(liberty_proto::LuTableTemplate {
                kind: "power_lut_template".to_string(),
                name: "power_tmpl".to_string(),
                index_1: vec![0.1],
                ..Default::default()
            });

        let bytes = with_timing.encode_to_vec();
        let loaded = decode_library_from_bytes(&bytes, "unit-test").unwrap();

        assert_eq!(loaded.lu_table_templates.len(), 1);
        assert_eq!(loaded.lu_table_templates[0].kind, "power_lut_template");
        assert_eq!(loaded.lu_table_templates[0].name, "power_tmpl");
        assert_eq!(loaded.lu_table_templates[0].index_1, vec![0.1]);
        assert!(loaded.cells[0].pins[0].timing_arcs.is_empty());
    }

    #[test]
    fn timing_counters_measure_tables_and_values() {
        let with_timing = make_with_timing_payload();
        assert_eq!(count_timing_tables(&with_timing), 1);
        assert_eq!(count_timing_values(&with_timing), 1);
    }

    #[test]
    fn decode_skip_timing_values_summary_counts_tables() {
        let with_timing = make_with_timing_payload();
        let bytes = with_timing.encode_to_vec();
        let summary =
            decode_timing_table_summary_skip_values_from_bytes(&bytes, "unit-test").unwrap();
        assert_eq!(summary.cells, 1);
        assert_eq!(summary.timing_tables, 1);
    }

    #[test]
    fn decode_with_timing_preserves_timing_payloads() {
        let with_timing = make_with_timing_payload();
        let bytes = with_timing.encode_to_vec();
        let loaded = decode_library_with_timing_data_from_bytes(&bytes, "unit-test").unwrap();
        assert_eq!(loaded.lu_table_templates.len(), 1);
        assert_eq!(loaded.cells[0].pins[0].timing_arcs.len(), 1);
        assert_eq!(
            loaded.cells[0].pins[0].timing_arcs[0].tables[0].values,
            vec![1.0]
        );
    }

    #[test]
    fn converting_with_timing_wrapper_to_without_timing_strips_data() {
        let wrapped = LibraryWithTimingData::from_proto(make_with_timing_payload());
        let stripped: Library = wrapped.into();
        assert!(stripped.lu_table_templates.is_empty());
        assert!(stripped.cells[0].pins[0].timing_arcs.is_empty());
    }

    #[test]
    fn decode_with_timing_errors_when_payload_absent() {
        let without_timing = make_without_timing_payload();
        let bytes = without_timing.encode_to_vec();
        let err = decode_library_with_timing_data_from_bytes(&bytes, "unit-test").unwrap_err();
        assert!(
            format!("{err:#}").contains("has no timing payloads"),
            "expected missing-timing error, got: {err:#}"
        );
    }

    #[test]
    fn decode_with_timing_errors_when_only_templates_are_present() {
        let templates_only = make_templates_only_payload();
        let bytes = templates_only.encode_to_vec();
        let err = decode_library_with_timing_data_from_bytes(&bytes, "unit-test").unwrap_err();
        assert!(
            format!("{err:#}").contains("has no timing payloads"),
            "expected missing-timing error, got: {err:#}"
        );
    }
}
