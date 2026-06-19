// SPDX-License-Identifier: Apache-2.0

use flate2::Compression;
use flate2::write::GzEncoder;
use prost::Message;
use std::io::Write;
use std::process::Command;
use xlsynth_g8r::liberty::model::LIBERTY_FORMAT_MAGIC;
use xlsynth_g8r::liberty_proto::Library;
use xlsynth_test_helpers::compare_golden_text;

#[test]
fn lib2proto_records_generation_metadata_and_info_reports_it() {
    let temp_dir = tempfile::tempdir().expect("create temp directory");
    let liberty_path = temp_dir.path().join("test.lib");
    let minimal_path = temp_dir.path().join("test.notiming.nopower.proto");
    let full_path = temp_dir.path().join("test.timing.power.proto");
    let full_gz_path = temp_dir.path().join("test.timing.power.proto.gz");
    let full_textproto_path = temp_dir.path().join("test.timing.power.textproto");
    std::fs::write(
        &liberty_path,
        r#"
library (test) {
  nom_voltage : 0.7;
  time_unit : "1ns";
  capacitive_load_unit (1, pf);
  voltage_unit : "1V";
  lu_table_template (timing_tmpl) {
    variable_1 : input_net_transition;
    index_1 ("0.1");
  }
  power_lut_template (power_tmpl) {
    variable_1 : input_transition_time;
    index_1 ("0.1");
  }
  cell (INV) {
    area : 1.0;
    pin (A) {
      direction : input;
      capacitance : 0.1;
      internal_power () {
        related_pg_pin : VDD;
        rise_power (power_tmpl) {
          values ("0.2");
        }
        fall_power (power_tmpl) {
          values ("0.3");
        }
      }
    }
    pin (Y) {
      direction : output;
      function : "!A";
      timing () {
        related_pin : "A";
        timing_sense : negative_unate;
        timing_type : combinational;
        cell_rise (timing_tmpl) {
          values ("0.4");
        }
      }
    }
  }
}
"#,
    )
    .expect("write Liberty input");

    let provenance = "asap7 commit abcde1234, generated locally";
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("lib2proto")
        .arg("--output")
        .arg(&minimal_path)
        .arg("--provenance")
        .arg(provenance)
        .arg(&liberty_path)
        .output()
        .expect("run lib2proto");
    assert!(
        output.status.success(),
        "lib2proto failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let minimal_bytes = std::fs::read(&minimal_path).expect("read generated proto");
    let minimal = Library::decode(minimal_bytes.as_slice()).expect("decode generated proto");
    assert_eq!(minimal.format_magic, LIBERTY_FORMAT_MAGIC);
    assert_eq!(minimal.provenance, provenance);
    assert_eq!(minimal.source_files, vec!["test.lib"]);
    assert_eq!(minimal.nominal_voltage, None);
    assert!(minimal.lu_table_templates.is_empty());
    assert!(minimal.cells[0].pins[0].internal_power.is_empty());
    assert!(minimal.cells[0].pins[1].timing_arcs.is_empty());

    let full_output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("lib2proto")
        .arg("--output")
        .arg(&full_path)
        .arg("--provenance")
        .arg(provenance)
        .arg("--include-timing")
        .arg("--include-power")
        .arg(&liberty_path)
        .output()
        .expect("run lib2proto with timing and power");
    assert!(
        full_output.status.success(),
        "lib2proto --include-timing --include-power failed: {}",
        String::from_utf8_lossy(&full_output.stderr)
    );
    let full_bytes = std::fs::read(&full_path).expect("read full proto");
    let full = Library::decode(full_bytes.as_slice()).expect("decode full proto");
    assert_eq!(full.format_magic, LIBERTY_FORMAT_MAGIC);
    assert_eq!(full.provenance, provenance);
    assert_eq!(full.source_files, vec!["test.lib"]);
    assert_eq!(full.nominal_voltage, Some(0.7));
    assert_eq!(full.lu_table_templates.len(), 2);
    assert_eq!(full.cells[0].pins[0].internal_power.len(), 1);
    assert_eq!(full.cells[0].pins[1].timing_arcs.len(), 1);
    assert_eq!(full.lut_axes.len(), 1);
    assert_ne!(full.lu_table_templates[0].index_1_id, 0);
    let timing_table = &full.cells[0].pins[1].timing_arcs[0].tables[0];
    assert_eq!(timing_table.index_1_id, 0);
    assert_eq!(timing_table.values, vec![0.4_f32]);

    let textproto_output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("lib2proto")
        .arg("--output")
        .arg(&full_textproto_path)
        .arg("--provenance")
        .arg(provenance)
        .arg("--include-timing")
        .arg("--include-power")
        .arg(&liberty_path)
        .output()
        .expect("run lib2proto for pretty textproto");
    assert!(
        textproto_output.status.success(),
        "lib2proto textproto failed: {}",
        String::from_utf8_lossy(&textproto_output.stderr)
    );
    compare_golden_text(
        &std::fs::read_to_string(&full_textproto_path).expect("read generated textproto"),
        "tests/lib2proto_pretty.golden.textproto",
    );

    let info_output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("liberty-proto-info")
        .arg(&full_path)
        .output()
        .expect("run liberty-proto-info");
    assert!(
        info_output.status.success(),
        "liberty-proto-info failed: {}",
        String::from_utf8_lossy(&info_output.stderr)
    );
    compare_golden_text(
        &String::from_utf8(info_output.stdout).expect("UTF-8 info output"),
        "tests/liberty_proto_info.golden.txt",
    );

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&full_bytes).expect("compress full proto");
    std::fs::write(&full_gz_path, encoder.finish().expect("finish gzip stream"))
        .expect("write compressed proto");
    let gzip_info_output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("liberty-proto-info")
        .arg(&full_gz_path)
        .output()
        .expect("inspect compressed proto");
    assert!(
        gzip_info_output.status.success(),
        "liberty-proto-info rejected compressed proto: {}",
        String::from_utf8_lossy(&gzip_info_output.stderr)
    );
    compare_golden_text(
        &String::from_utf8(gzip_info_output.stdout).expect("UTF-8 compressed info output"),
        "tests/liberty_proto_info_gz.golden.txt",
    );

    let textproto_info = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("liberty-proto-info")
        .arg(&full_textproto_path)
        .output()
        .expect("inspect generated textproto");
    assert!(
        textproto_info.status.success(),
        "liberty-proto-info rejected generated textproto: {}",
        String::from_utf8_lossy(&textproto_info.stderr)
    );
}

#[test]
fn lib2proto_applies_ordered_cell_filter_policy() {
    let temp_dir = tempfile::tempdir().expect("create temp directory");
    let liberty_path = temp_dir.path().join("filter.lib");
    let policy_path = temp_dir.path().join("filter.policy");
    let output_path = temp_dir.path().join("filtered.notiming.nopower.proto");
    std::fs::write(
        &liberty_path,
        r#"
library (filter_test) {
  cell (BLOCK_NATIVE) {
    area : 1.0;
    dont_use : true;
    pin (A) { direction : input; }
    pin (Y) { direction : output; function : "A"; }
  }
  cell (BLOCK_POLICY) {
    area : 1.0;
    pin (A) { direction : input; }
    pin (Y) { direction : output; function : "A"; }
  }
  cell (CLOCK_GATE) {
    area : 1.0;
    dont_use : true;
    pin (A) { direction : input; }
    pin (Y) { direction : output; function : "A"; }
  }
  cell (KEEP) {
    area : 1.0;
    pin (A) { direction : input; }
    pin (Y) { direction : output; function : "A"; }
  }
}
"#,
    )
    .expect("write filter Liberty input");
    std::fs::write(
        &policy_path,
        r#"
# Last matching rule wins.
exclude ^BLOCK_POLICY$
exclude .*GATE.*
include ^CLOCK_GATE$
"#,
    )
    .expect("write cell-filter policy");

    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("lib2proto")
        .arg("--output")
        .arg(&output_path)
        .arg("--cell-filter-policy")
        .arg(&policy_path)
        .arg(&liberty_path)
        .output()
        .expect("run filtered lib2proto");
    assert!(
        output.status.success(),
        "filtered lib2proto failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(&output_path).expect("read filtered proto");
    let library = Library::decode(bytes.as_slice()).expect("decode filtered proto");
    assert_eq!(
        library
            .cells
            .iter()
            .map(|cell| cell.name.as_str())
            .collect::<Vec<_>>(),
        vec!["CLOCK_GATE", "KEEP"]
    );
    assert!(
        library
            .cells
            .iter()
            .all(|cell| cell.dont_use == Some(false))
    );
}
