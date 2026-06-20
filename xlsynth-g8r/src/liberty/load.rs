// SPDX-License-Identifier: Apache-2.0

use crate::liberty::descriptor::liberty_descriptor_pool;
pub use crate::liberty::model::strip_timing_data;
use crate::liberty::model::{LIBERTY_FORMAT_MAGIC, library_from_proto};
use crate::liberty_model;
use crate::liberty_proto;
use anyhow::{Context, Result, anyhow};
use flate2::read::MultiGzDecoder;
use prost::Message;
use prost_reflect::DynamicMessage;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Deref;
use std::path::Path;

#[derive(Debug)]
pub struct Library {
    proto: liberty_model::Library,
}

#[derive(Debug)]
pub struct LibraryWithTimingData {
    proto: liberty_model::Library,
}

impl Library {
    /// Wraps Liberty model data after dropping timing payloads.
    pub fn from_model(mut model: liberty_model::Library) -> Self {
        strip_timing_data(&mut model);
        Self { proto: model }
    }

    pub fn as_model(&self) -> &liberty_model::Library {
        &self.proto
    }

    pub fn into_model(self) -> liberty_model::Library {
        self.proto
    }
}

impl LibraryWithTimingData {
    /// Wraps fully populated Liberty model data while preserving timing
    /// payloads.
    pub fn from_model(model: liberty_model::Library) -> Self {
        Self { proto: model }
    }

    pub fn as_model(&self) -> &liberty_model::Library {
        &self.proto
    }

    pub fn into_model(self) -> liberty_model::Library {
        self.proto
    }
}

impl Deref for Library {
    type Target = liberty_model::Library;

    fn deref(&self) -> &Self::Target {
        &self.proto
    }
}

impl Deref for LibraryWithTimingData {
    type Target = liberty_model::Library;

    fn deref(&self) -> &Self::Target {
        self.as_model()
    }
}

impl AsRef<liberty_model::Library> for Library {
    fn as_ref(&self) -> &liberty_model::Library {
        &self.proto
    }
}

impl AsRef<liberty_model::Library> for LibraryWithTimingData {
    fn as_ref(&self) -> &liberty_model::Library {
        self.as_model()
    }
}

impl From<Library> for liberty_model::Library {
    fn from(value: Library) -> Self {
        value.proto
    }
}

impl From<LibraryWithTimingData> for liberty_model::Library {
    fn from(value: LibraryWithTimingData) -> Self {
        value.into_model()
    }
}

impl From<LibraryWithTimingData> for Library {
    fn from(value: LibraryWithTimingData) -> Self {
        Library::from_model(value.into_model())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimingTableSummary {
    pub cells: usize,
    pub timing_tables: usize,
}

pub fn count_timing_tables(proto: &liberty_model::Library) -> usize {
    proto
        .cells
        .iter()
        .flat_map(|cell| &cell.pins)
        .flat_map(|pin| &pin.timing_arcs)
        .map(|arc| arc.tables.len())
        .sum()
}

pub fn count_timing_values(proto: &liberty_model::Library) -> usize {
    proto
        .cells
        .iter()
        .flat_map(|cell| &cell.pins)
        .flat_map(|pin| &pin.timing_arcs)
        .flat_map(|arc| &arc.tables)
        .map(|table| table.value_range().len())
        .sum()
}

#[derive(Clone, PartialEq, Message)]
struct LibraryTimingSummaryPayload {
    #[prost(fixed64, tag = "1")]
    format_magic: u64,
    #[prost(message, repeated, tag = "2")]
    cells: Vec<CellTimingSummaryPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct CellTimingSummaryPayload {
    #[prost(message, repeated, tag = "2")]
    pins: Vec<PinTimingSummaryPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct PinTimingSummaryPayload {
    #[prost(message, repeated, tag = "9")]
    timing_arcs: Vec<TimingArcSummaryPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct TimingArcSummaryPayload {
    #[prost(message, repeated, tag = "5")]
    tables: Vec<TimingTableSummaryPayload>,
}

#[derive(Clone, PartialEq, Message)]
struct TimingTableSummaryPayload {}

fn count_summary_tables(payload: &LibraryTimingSummaryPayload) -> usize {
    payload
        .cells
        .iter()
        .flat_map(|cell| &cell.pins)
        .flat_map(|pin| &pin.timing_arcs)
        .map(|arc| arc.tables.len())
        .sum()
}

pub(crate) fn decode_library_binary_or_text(
    bytes: &[u8],
    source_name: &str,
) -> Result<liberty_model::Library> {
    let binary_error = match liberty_proto::Library::decode(bytes) {
        Ok(proto) => {
            return library_from_proto(proto)
                .with_context(|| format!("decoding Liberty proto '{source_name}'"));
        }
        Err(error) => error,
    };

    let text = std::str::from_utf8(bytes).with_context(|| {
        format!(
            "binary decode failed for '{}': {}; textproto fallback input is not UTF-8",
            source_name, binary_error
        )
    })?;
    let descriptor = liberty_descriptor_pool()
        .get_message_by_name("liberty.Library")
        .ok_or_else(|| anyhow!("missing liberty.Library descriptor"))?;
    let dynamic = DynamicMessage::parse_text_format(descriptor, text).with_context(|| {
        format!(
            "binary decode failed for '{}': {}; textproto fallback parse failed",
            source_name, binary_error
        )
    })?;
    let wire = liberty_proto::Library::decode(dynamic.encode_to_vec().as_slice())
        .with_context(|| format!("decoding textproto fallback for '{source_name}'"))?;
    library_from_proto(wire).with_context(|| format!("expanding Liberty proto '{source_name}'"))
}

pub fn decode_timing_table_summary_skip_values_from_bytes(
    bytes: &[u8],
    source_name: &str,
) -> Result<TimingTableSummary> {
    if let Ok(payload) = LibraryTimingSummaryPayload::decode(bytes) {
        if payload.format_magic == LIBERTY_FORMAT_MAGIC {
            return Ok(TimingTableSummary {
                cells: payload.cells.len(),
                timing_tables: count_summary_tables(&payload),
            });
        }
    }
    let proto = decode_library_binary_or_text(bytes, source_name)?;
    Ok(TimingTableSummary {
        cells: proto.cells.len(),
        timing_tables: count_timing_tables(&proto),
    })
}

fn decode_library_from_bytes(bytes: &[u8], source_name: &str) -> Result<Library> {
    Ok(Library::from_model(decode_library_binary_or_text(
        bytes,
        source_name,
    )?))
}

fn decode_library_with_timing_data_from_bytes(
    bytes: &[u8],
    source_name: &str,
) -> Result<LibraryWithTimingData> {
    let model = decode_library_binary_or_text(bytes, source_name)?;
    if count_timing_tables(&model) == 0 {
        return Err(anyhow!(
            "liberty proto '{}' has no timing payloads; load a timing-enabled proto or use the non-timing loader",
            source_name
        ));
    }
    Ok(LibraryWithTimingData::from_model(model))
}

/// Reads a binary proto or textproto, transparently decompressing `.gz` inputs.
pub fn read_liberty_proto_bytes_from_path(path: &Path) -> Result<Vec<u8>> {
    let file =
        File::open(path).with_context(|| format!("opening liberty proto '{}'", path.display()))?;
    let is_gz = path.extension().map(|ext| ext == "gz").unwrap_or(false);
    let mut reader: Box<dyn Read> = if is_gz {
        Box::new(MultiGzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading liberty proto '{}'", path.display()))?;
    Ok(bytes)
}

pub fn load_library_from_path(path: &Path) -> Result<Library> {
    let bytes = read_liberty_proto_bytes_from_path(path)?;
    decode_library_from_bytes(&bytes, &path.display().to_string())
}

pub fn load_library_with_timing_data_from_path(path: &Path) -> Result<LibraryWithTimingData> {
    let bytes = read_liberty_proto_bytes_from_path(path)?;
    decode_library_with_timing_data_from_bytes(&bytes, &path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::model::library_to_proto;
    use crate::liberty_model::{
        Cell, InternalPower, LibraryBuilder, LuTableTemplate, Pin, PowerTransition, Sequential,
        SequentialKind,
    };
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    #[derive(Clone, PartialEq, Message)]
    struct OldLibraryPayload {
        #[prost(message, repeated, tag = "1")]
        cells: Vec<OldCellPayload>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct OldCellPayload {
        #[prost(string, tag = "1")]
        name: String,
    }

    fn make_payload() -> liberty_model::Library {
        let mut builder = LibraryBuilder::new();
        let timing_table = builder
            .add_timing_table_f64(
                liberty_proto::TimingTableKind::CellRise,
                1,
                vec![],
                vec![],
                vec![],
                vec![0.25],
                vec![1],
                "",
            )
            .unwrap();
        let power_table = builder
            .add_power_table_f64(
                PowerTransition::Rise,
                2,
                vec![],
                vec![],
                vec![],
                vec![0.5],
                vec![1],
                "",
            )
            .unwrap();
        builder.nominal_voltage = Some(0.7);
        builder.provenance = "test provenance".to_string();
        builder.source_files = vec!["cells.lib".to_string()];
        builder.lu_table_templates = vec![
            LuTableTemplate {
                kind: "lu_table_template".to_string().into(),
                index_1: vec![0.1],
                ..Default::default()
            },
            LuTableTemplate {
                kind: "power_lut_template".to_string().into(),
                index_1: vec![0.1],
                ..Default::default()
            },
        ];
        let name = builder.intern_string("Y").unwrap();
        let arc = builder
            .add_timing_arc("A", "", "", "", vec![timing_table])
            .unwrap();
        builder.cells = vec![Cell {
            name: "INV".to_string().into(),
            pins: vec![Pin {
                name,
                timing_arcs: vec![arc],
                internal_power: vec![InternalPower {
                    tables: vec![power_table],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }];
        builder.finish()
    }

    #[test]
    fn compact_binary_roundtrips_to_runtime_data() {
        let wire = library_to_proto(make_payload()).unwrap();
        let loaded = decode_library_binary_or_text(&wire.encode_to_vec(), "unit-test").unwrap();

        assert_eq!(loaded.provenance, "test provenance");
        assert_eq!(loaded.lu_table_templates[0].index_1, vec![0.1]);
        let timing_table = &loaded.cells[0].pins[0].timing_arcs[0].tables[0];
        assert_eq!(loaded.timing_table_values(timing_table), &[0.25_f32]);
        let power_table = &loaded.cells[0].pins[0].internal_power[0].tables[0];
        assert_eq!(loaded.power_table_values(power_table), &[0.5_f32]);
    }

    #[test]
    fn sequential_complement_is_optional_in_binary_payloads() {
        let sequential = Sequential {
            state_var: "IQ".to_string().into(),
            next_state: "D".to_string().into(),
            clock_expr: "CLK".to_string().into(),
            kind: SequentialKind::Ff as i32,
            complementary_state_var: None,
            ..Default::default()
        };
        let mut payload_without_complement = make_payload();
        payload_without_complement.cells[0].sequential = vec![sequential.clone()];
        let wire_without_complement = library_to_proto(payload_without_complement).unwrap();
        let loaded_without_complement = decode_library_binary_or_text(
            &wire_without_complement.encode_to_vec(),
            "without-complement.proto",
        )
        .unwrap();
        assert_eq!(
            loaded_without_complement.cells[0].sequential[0].complementary_state_var,
            None
        );

        let mut payload_with_complement = make_payload();
        payload_with_complement.cells[0].sequential = vec![Sequential {
            complementary_state_var: Some("IQN".to_string()),
            ..sequential
        }];
        let wire_with_complement = library_to_proto(payload_with_complement).unwrap();
        let loaded_with_complement = decode_library_binary_or_text(
            &wire_with_complement.encode_to_vec(),
            "with-complement.proto",
        )
        .unwrap();
        assert_eq!(
            loaded_with_complement.cells[0].sequential[0]
                .complementary_state_var
                .as_deref(),
            Some("IQN")
        );
    }

    #[test]
    fn timing_summary_skips_compact_table_values() {
        let wire = library_to_proto(make_payload()).unwrap();

        let summary =
            decode_timing_table_summary_skip_values_from_bytes(&wire.encode_to_vec(), "unit-test")
                .unwrap();

        assert_eq!(summary.cells, 1);
        assert_eq!(summary.timing_tables, 1);
    }

    #[test]
    fn rejects_the_pre_breaking_change_wire_format() {
        let old = OldLibraryPayload {
            cells: vec![OldCellPayload {
                name: "INV".to_string().into(),
            }],
        };

        let error = decode_library_binary_or_text(&old.encode_to_vec(), "old.proto").unwrap_err();

        assert!(format!("{error:#}").contains("binary decode failed"));
    }

    #[test]
    fn loads_gzipped_proto_into_runtime_model() {
        let wire = library_to_proto(make_payload()).unwrap();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&wire.encode_to_vec()).unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("library.proto.gz");
        std::fs::write(&path, encoder.finish().unwrap()).unwrap();

        let loaded = load_library_with_timing_data_from_path(&path).unwrap();
        assert_eq!(loaded.cells.len(), 1);
        assert_eq!(loaded.cells[0].pins[0].timing_arcs.len(), 1);
        assert_eq!(loaded.cells[0].pins[0].timing_arcs[0].tables.len(), 1);
    }

    #[test]
    fn non_timing_loader_strips_timing_and_remaps_templates() {
        let wire = library_to_proto(make_payload()).unwrap();
        let loaded = decode_library_from_bytes(&wire.encode_to_vec(), "unit-test").unwrap();

        assert!(loaded.cells[0].pins[0].timing_arcs.is_empty());
        assert_eq!(loaded.lu_table_templates.len(), 1);
        assert_eq!(
            loaded.lu_table_templates[0].kind_str(&loaded),
            "power_lut_template"
        );
        assert_eq!(loaded.lut_shapes.len(), 1);
        assert_eq!(loaded.lut_values, vec![0.5_f32]);
        assert_eq!(
            loaded
                .power_table_shape(&loaded.cells[0].pins[0].internal_power[0].tables[0])
                .template_id,
            1
        );
    }
}
