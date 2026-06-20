// SPDX-License-Identifier: Apache-2.0

use crate::liberty::load::{decode_library_binary_or_text, read_liberty_proto_bytes_from_path};
use crate::liberty::model::Library;
use crate::liberty_model::PowerTransition;
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Structural and file metadata for one serialized Liberty proto.
#[derive(Clone, Debug, PartialEq)]
pub struct LibertyProtoInfo {
    pub size_bytes: u64,
    pub sha256: String,
    pub provenance: String,
    pub source_files: Vec<String>,
    pub cells: usize,
    pub combinational_cells: usize,
    pub sequential_cells: usize,
    pub dont_use_cells: usize,
    pub lut_templates: usize,
    pub timing_arcs: usize,
    pub internal_power_groups: usize,
    pub rise_power_tables: usize,
    pub fall_power_tables: usize,
    pub both_power_tables: usize,
    pub unknown_power_tables: usize,
    pub nominal_voltage: Option<f64>,
}

/// Summarizes one decoded Liberty proto and its stored-file identity.
fn summarize_liberty_proto(size_bytes: u64, sha256: String, proto: &Library) -> LibertyProtoInfo {
    let mut timing_arcs = 0;
    let mut internal_power_groups = 0;
    let mut rise_power_tables = 0;
    let mut fall_power_tables = 0;
    let mut both_power_tables = 0;
    let mut unknown_power_tables = 0;
    for cell in &proto.cells {
        for pin in &cell.pins {
            timing_arcs += pin.timing_arcs.len();
            internal_power_groups += pin.internal_power.len();
            for group in &pin.internal_power {
                for table in &group.tables {
                    match PowerTransition::try_from(table.transition)
                        .unwrap_or(PowerTransition::Unknown)
                    {
                        PowerTransition::Rise => rise_power_tables += 1,
                        PowerTransition::Fall => fall_power_tables += 1,
                        PowerTransition::Both => both_power_tables += 1,
                        PowerTransition::Unknown => unknown_power_tables += 1,
                    }
                }
            }
        }
    }
    let sequential_cells = proto
        .cells
        .iter()
        .filter(|cell| !cell.sequential.is_empty())
        .count();
    let dont_use_cells = proto
        .cells
        .iter()
        .filter(|cell| cell.dont_use == Some(true))
        .count();
    LibertyProtoInfo {
        size_bytes,
        sha256,
        provenance: proto.provenance.clone(),
        source_files: proto.source_files.clone(),
        cells: proto.cells.len(),
        combinational_cells: proto.cells.len() - sequential_cells,
        sequential_cells,
        dont_use_cells,
        lut_templates: proto.lu_table_templates.len(),
        timing_arcs,
        internal_power_groups,
        rise_power_tables,
        fall_power_tables,
        both_power_tables,
        unknown_power_tables,
        nominal_voltage: proto.nominal_voltage,
    }
}

fn stored_file_identity(path: &Path) -> Result<(u64, String)> {
    let file =
        File::open(path).with_context(|| format!("opening Liberty proto '{}'", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut size_bytes = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buf)
            .with_context(|| format!("reading Liberty proto '{}'", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buf[..count]);
        size_bytes += count as u64;
    }
    Ok((size_bytes, format!("{:x}", digest.finalize())))
}

/// Loads and summarizes a binary Liberty proto or textproto, optionally
/// gzipped.
pub fn liberty_proto_info_from_path(path: &Path) -> Result<LibertyProtoInfo> {
    let (size_bytes, sha256) = stored_file_identity(path)?;
    let bytes = read_liberty_proto_bytes_from_path(path)?;
    let proto = decode_library_binary_or_text(&bytes, &path.display().to_string())?;
    Ok(summarize_liberty_proto(size_bytes, sha256, &proto))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::model::{library_from_proto, library_to_proto};
    use crate::liberty_model::{Cell, InternalPower, LibraryBuilder, Pin, Sequential, TimingArc};
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use prost::Message;
    use std::io::Write;

    #[test]
    fn summarizes_combinational_sequential_timing_and_power_data() {
        let mut builder = LibraryBuilder::new();
        let rise = builder
            .add_power_table_f64(
                PowerTransition::Rise,
                0,
                vec![],
                vec![],
                vec![],
                vec![0.0],
                vec![],
                "",
            )
            .unwrap();
        let fall = builder
            .add_power_table_f64(
                PowerTransition::Fall,
                0,
                vec![],
                vec![],
                vec![],
                vec![0.0],
                vec![],
                "",
            )
            .unwrap();
        builder.provenance = "test provenance".to_string();
        builder.source_files = vec![
            "combinational.lib".to_string(),
            "sequential.lib".to_string(),
        ];
        builder.nominal_voltage = Some(0.7);
        builder.lu_table_templates = vec![Default::default(), Default::default()];
        builder.cells = vec![
            Cell {
                name: "INV".to_string().into(),
                dont_use: Some(true),
                pins: vec![Pin {
                    timing_arcs: vec![TimingArc::default()],
                    internal_power: vec![InternalPower {
                        tables: vec![rise, fall],
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            },
            Cell {
                name: "DFF".to_string().into(),
                sequential: vec![Sequential::default()],
                ..Default::default()
            },
        ];
        let wire = library_to_proto(builder.finish()).unwrap();
        let bytes = wire.encode_to_vec();
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let proto = library_from_proto(wire).unwrap();
        let info = summarize_liberty_proto(bytes.len() as u64, sha256, &proto);

        assert_eq!(info.size_bytes, bytes.len() as u64);
        assert_eq!(info.sha256.len(), 64);
        assert_eq!(info.provenance, "test provenance");
        assert_eq!(
            info.source_files,
            vec!["combinational.lib", "sequential.lib"]
        );
        assert_eq!(info.cells, 2);
        assert_eq!(info.combinational_cells, 1);
        assert_eq!(info.sequential_cells, 1);
        assert_eq!(info.dont_use_cells, 1);
        assert_eq!(info.lut_templates, 2);
        assert_eq!(info.timing_arcs, 1);
        assert_eq!(info.internal_power_groups, 1);
        assert_eq!(info.rise_power_tables, 1);
        assert_eq!(info.fall_power_tables, 1);
        assert_eq!(info.both_power_tables, 0);
        assert_eq!(info.unknown_power_tables, 0);
        assert_eq!(info.nominal_voltage, Some(0.7));
    }

    #[test]
    fn reads_gzipped_proto_and_reports_stored_file_identity() {
        let proto = Library {
            provenance: "compressed test".to_string().into(),
            cells: vec![Cell {
                name: "INV".to_string().into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(&library_to_proto(proto).unwrap().encode_to_vec())
            .unwrap();
        let compressed = encoder.finish().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("library.proto.gz");
        std::fs::write(&path, &compressed).unwrap();

        let info = liberty_proto_info_from_path(&path).unwrap();

        assert_eq!(info.size_bytes, compressed.len() as u64);
        assert_eq!(info.sha256, format!("{:x}", Sha256::digest(&compressed)));
        assert_eq!(info.provenance, "compressed test");
        assert_eq!(info.cells, 1);
    }
}
